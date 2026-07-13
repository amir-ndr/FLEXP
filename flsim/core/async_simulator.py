"""
core/async_simulator.py: Discrete-event simulator for async federated learning.

Simulation model (FedAsync, Xie et al. 2019)
---------------------------------------------
The paper has two concurrent threads:

  Thread Scheduler  — periodically dispatches clients, sending them the
                      current global model with its timestamp τ.
  Thread Updater    — whenever a (x_new, τ) pair arrives from any worker,
                      immediately updates: x_t = (1-α_t)*x_{t-1} + α_t*x_new.

This simulator models both threads with a **discrete-event priority queue**:

  1. DISPATCH  (at virtual time t_dispatch, current epoch τ)
       - Select a client via algorithm.select_clients()
       - Take a snapshot of the current global model (at epoch τ)
       - Run local PyTorch training on that snapshot (wall-clock ignored)
       - Compute simulated finish time:
             arrival_time = t_dispatch + t_compute + t_upload
       - Push an ArrivalEvent(arrival_time, τ, trained_weights) onto the queue

  2. UPDATER LOOP  (process epochs 0 … T-1)
       - Pop the earliest ArrivalEvent from the queue
       - Compute staleness k = (current epoch) − τ
       - Compute alpha_t = algorithm.mixing_weight(base_alpha, k)
       - Call algorithm.aggregate_async(global_model, update, epoch, k, alpha_t)
       - Load result into global_model
       - Dispatch ONE replacement client (sliding-window: always keep
         window_size clients in flight)

Sliding-window concurrency
--------------------------
The simulator keeps exactly `window_size` clients in flight at all times:
  - On startup: dispatch window_size clients simultaneously (all see model at epoch 0)
  - After each Updater step: dispatch 1 replacement client (sees current model)

This naturally produces staleness: clients dispatched earlier but slow to arrive
will be processed while the model has already been updated by faster peers.

Bandwidth is divided equally among the window_size concurrent clients
(effective bandwidth per client = total_bandwidth / window_size) to model
real contention on the uplink channel.

Extending
---------
  - Change selection policy  → override AsyncFederatedAlgorithm.select_clients()
  - Change staleness decay   → override AsyncFederatedAlgorithm.mixing_weight()
  - Change update rule       → override AsyncFederatedAlgorithm.aggregate_async()
  - Change concurrency level → async_fl.window_size in config
"""

import copy
import heapq
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from flsim.core.client import ClientUpdate
from flsim.core.async_logger import AsyncRoundResult
from flsim.interfaces.async_algorithm import AsyncFederatedAlgorithm
from flsim.system.energy import EnergyModel


# ---------------------------------------------------------------------------
# Internal event type for the priority queue
# ---------------------------------------------------------------------------

@dataclass
class _ArrivalEvent:
    """
    One entry in the pending-arrivals priority queue.

    Fields:
        arrival_time: simulated time when the update reaches the server.
        seq:          insertion counter for stable tie-breaking.
        tau:          global epoch at which the client was dispatched
                      (= the model version the client trained on).
        update:       the ClientUpdate produced by local training.
    """
    arrival_time: float
    seq:          int
    tau:          int
    update:       Any

    def __lt__(self, other):
        """Primary sort: arrival_time. Tie-break: seq (FIFO)."""
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        return self.seq < other.seq


# ---------------------------------------------------------------------------
# AsyncSimulator
# ---------------------------------------------------------------------------

class AsyncSimulator:
    """
    Async FL simulation orchestrator.

    Constructor arguments mirror Simulator for drop-in comparability.
    The server's algorithm must be an AsyncFederatedAlgorithm instance.

    Key config fields (under async_fl in YAML or config_overrides):
        alpha       (float): base mixing hyperparameter. Default 0.1.
        window_size (int):   number of clients in flight. Default = clients_per_round.

    The existing learning.global_rounds field controls how many server
    updates (= global epochs) to run.
    """

    def __init__(
        self,
        server,
        clients: list,
        time_model,
        channel_model,
        allocator,
        energy_model: EnergyModel,
        evaluator,
        logger,
        config,
        rng: np.random.RandomState,
        device: torch.device,
    ):
        if not isinstance(server.algorithm, AsyncFederatedAlgorithm):
            raise TypeError(
                f"AsyncSimulator requires an AsyncFederatedAlgorithm, "
                f"got {type(server.algorithm).__name__}. "
                f"Pass a FedAsync (or subclass) instance via "
                f"components={{\"algorithm\": FedAsync(...)}}."
            )

        self.server        = server
        self.clients       = clients
        self.time_model    = time_model
        self.channel_model = channel_model
        self.allocator     = allocator
        self.energy_model  = energy_model
        self.evaluator     = evaluator
        self.logger        = logger
        self.config        = config
        self.rng           = rng
        self.device        = device

        # Resolve upload size once at startup
        from flsim.core.simulator import _resolve_upload_bits
        self._upload_bits = _resolve_upload_bits(config.wireless, server.global_model)
        self._p_max_w     = config._p_max_w
        self._f_max_hz    = config._f_max_hz

        # Async-specific config (async_fl section in YAML)
        # Alpha priority: algorithm.alpha > config.async_fl.alpha > default 0.1
        # This lets FedAsync(alpha=0.6) override the YAML without needing a config edit.
        async_cfg = getattr(config, "async_fl", None)
        _alg_alpha = getattr(server.algorithm, "alpha", None)
        if _alg_alpha is not None:
            self._base_alpha = float(_alg_alpha)
        else:
            self._base_alpha = float(getattr(async_cfg, "alpha", 0.1) if async_cfg else 0.1)
        self._window_size  = int(  getattr(async_cfg, "window_size",
                                           config.learning.clients_per_round)
                                   if async_cfg else config.learning.clients_per_round)

        # Effective bandwidth per client = total / window_size
        # (models uplink contention when window_size clients transmit concurrently)
        self._effective_bw = config.wireless.total_bandwidth_hz / max(1, self._window_size)

        # If downlink_negligible is set, model-broadcast (download) time is 0,
        # so arrival_time = dispatch_time + t_compute + t_upload only.
        self._downlink_negligible = bool(
            getattr(config.wireless, "downlink_negligible", False)
        )

        # Insertion counter for stable priority-queue tie-breaking
        self._seq = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full async FL experiment.

        Runs learning.global_rounds server updates (global epochs).
        Evaluates every evaluation.evaluate_every epochs.
        Saves plots at the end.
        """
        T           = self.config.learning.global_rounds
        cfg_eval    = self.config.evaluation

        print(
            f"\n[AsyncSimulator] Starting FedAsync: "
            f"T={T} epochs, window={self._window_size}, "
            f"alpha={self._base_alpha}, "
            f"staleness_func={getattr(self.server.algorithm, 'staleness_func', 'custom')}, "
            f"device={self.device}"
        )
        print(
            f"[AsyncSimulator] Upload size: {self._upload_bits / 1e3:.1f} kbits "
            f"(mode={self.config.wireless.upload_size_mode}), "
            f"effective BW/client: {self._effective_bw / 1e6:.2f} MHz"
        )

        # ---- initial dispatch: fill the window ----
        # Extra kwargs passed to select_clients so selection policies that need
        # channel or bandwidth info (e.g. FedAsyncTopKFastTotal) can use them.
        selection_ctx = {
            "channel_model":     self.channel_model,
            "noise_psd_w_per_hz": self.config._noise_psd_w_per_hz,
            "bw_per_client_hz":  self._effective_bw,
            "upload_size_bits":  self._upload_bits,
        }

        pending: list = []
        initial_batch = self.server.algorithm.select_clients(
            self.clients, self._window_size, self.rng, **selection_ctx
        )
        for client in initial_batch:
            ev = self._dispatch_client(client, current_time=0.0, current_epoch=0)
            heapq.heappush(pending, ev)

        global_epoch    = 0
        simulated_time  = 0.0

        # ---- main updater loop ----
        while global_epoch < T:
            if not pending:
                print(
                    f"[AsyncSimulator] Warning: pending queue empty at epoch "
                    f"{global_epoch}. Stopping early."
                )
                break

            # Pop earliest arriving update
            event = heapq.heappop(pending)
            simulated_time = event.arrival_time

            staleness = global_epoch - event.tau
            alpha_t   = self.server.algorithm.mixing_weight(self._base_alpha, staleness)

            # Apply update to global model
            new_state = self.server.algorithm.aggregate_async(
                self.server.global_model,
                event.update,
                global_epoch,
                staleness,
                alpha_t,
            )
            self.server.global_model.load_state_dict(new_state)
            self.server.round_idx = global_epoch + 1

            # Build result record
            upd = event.update
            result = AsyncRoundResult(
                global_epoch=global_epoch,
                arrival_time_s=simulated_time,
                staleness=staleness,
                alpha_used=alpha_t,
                client_id=upd.client_id,
                compute_time_s=upd.compute_time_s,
                upload_time_s=upd.upload_time_s,
                total_time_s=upd.total_time_s,
                compute_energy_j=upd.compute_energy_j,
                tx_energy_j=upd.tx_energy_j,
                total_energy_j=upd.total_energy_j,
                channel_gain=upd.channel_gain,
                achievable_rate_bps=upd.achievable_rate_bps,
                train_loss=upd.train_loss,
            )

            # Evaluate
            eval_result = None
            if global_epoch % cfg_eval.evaluate_every == 0:
                eval_result = self.evaluator.evaluate(
                    self.server.global_model, device=self.device
                )

            self.logger.log_epoch(global_epoch, simulated_time, result, eval_result)

            if eval_result is not None:
                print(
                    f"  Epoch {global_epoch:5d} | "
                    f"sim_time={simulated_time:10.2f}s | "
                    f"staleness={staleness:3d} | "
                    f"α_t={alpha_t:.4f} | "
                    f"acc={eval_result.test_accuracy:.4f} | "
                    f"loss={eval_result.test_loss:.4f}"
                )

            global_epoch += 1

            # Dispatch one replacement to keep the window full
            if global_epoch < T:
                replacement_batch = self.server.algorithm.select_clients(
                    self.clients, 1, self.rng, **selection_ctx
                )
                new_ev = self._dispatch_client(
                    replacement_batch[0],
                    current_time=simulated_time,
                    current_epoch=global_epoch,
                )
                heapq.heappush(pending, new_ev)

        print("[AsyncSimulator] Done. Saving plots …")
        self.logger.plot_results()

    # ------------------------------------------------------------------
    # Internal: dispatch one client
    # ------------------------------------------------------------------

    def _dispatch_client(
        self,
        client,
        current_time: float,
        current_epoch: int,
    ) -> _ArrivalEvent:
        """
        Dispatch one client:
          - Compute simulated timing and energy
          - Run local PyTorch training on a snapshot of the current global model
          - Return an ArrivalEvent that will be pushed onto the priority queue

        The local training runs on the global model AS IT STANDS at
        current_epoch (the snapshot).  This is correct: the client trains
        on the model it received, and we record τ = current_epoch so that
        staleness can be computed later when the update arrives.

        Bandwidth is divided by window_size to model concurrent uplink.
        """
        cid       = client.client_id
        cfg       = self.config.learning
        noise_psd = self.config._noise_psd_w_per_hz

        # ---- channel + resource allocation ----
        gain = self.channel_model.channel_gain(client.profile, self.rng)

        # Pass only this client's profile; effective_bw already accounts for sharing
        profiles = [client.profile]
        bw_alloc = self.allocator.allocate_bandwidth(
            profiles, self._effective_bw, channel_gains={cid: gain}
        )
        pw_alloc = self.allocator.allocate_power(
            profiles, self._p_max_w, channel_gains={cid: gain}
        )
        fq_alloc = self.allocator.allocate_cpu_freq(
            profiles, self._f_max_hz, channel_gains={cid: gain}
        )

        bw_hz = bw_alloc[cid]
        p_w   = pw_alloc[cid]
        f_hz  = fq_alloc[cid]

        rate_bps = self.channel_model.achievable_rate_bps(
            bandwidth_hz=bw_hz,
            tx_power_w=p_w,
            channel_gain=gain,
            noise_psd_w_per_hz=noise_psd,
        )

        # ---- simulated timing ----
        t_comp = self.time_model.compute_training_time(
            client.profile,
            num_samples=client.num_samples,
            local_epochs=cfg.local_epochs,
            batch_size=cfg.batch_size,
            cpu_freq_hz=f_hz,
        )
        t_up = self.time_model.compute_upload_time(
            client.profile,
            size_bits=self._upload_bits,
            bandwidth_hz=bw_hz,
            channel_gain=gain,
        )
        t_dn = 0.0 if self._downlink_negligible else \
            self.time_model.compute_download_time(
                client.profile,
                size_bits=self._upload_bits,
                bandwidth_hz=bw_hz,
                channel_gain=gain,
            )

        # ---- energy ----
        e_comp = self.energy_model.compute_energy_j(
            client.profile,
            local_epochs=cfg.local_epochs,
            num_samples=client.num_samples,
            cpu_freq_hz=f_hz,
        )
        e_tx = self.energy_model.transmission_energy_j(
            client.profile, upload_time_s=t_up, tx_power_w=p_w,
        )

        # ---- local training (on snapshot of current global model) ----
        self.server.algorithm.configure_client(
            client, self.server.global_model, current_epoch
        )
        state_dict, n_samples, train_loss = client.train(
            global_model=self.server.global_model,
            local_epochs=cfg.local_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            device=self.device,
            proximal_mu=getattr(client, "proximal_mu", 0.0),
        )

        update = ClientUpdate(
            client_id=cid,
            state_dict=state_dict,
            num_samples=n_samples,
            train_loss=train_loss,
            compute_time_s=t_comp,
            upload_time_s=t_up,
            download_time_s=t_dn,
            total_time_s=t_comp + t_up + t_dn,
            compute_energy_j=e_comp,
            tx_energy_j=e_tx,
            total_energy_j=e_comp + e_tx,
            channel_gain=gain,
            achievable_rate_bps=rate_bps,
            allocated_bandwidth_hz=bw_hz,
        )

        # Arrival = dispatch_time + t_download + t_compute + t_upload
        # (matches total_time_s stored in the ClientUpdate and is consistent
        #  with the sync Simulator which uses total_time_s = t_comp+t_up+t_dn
        #  for its round duration).
        arrival_time = current_time + t_dn + t_comp + t_up
        self._seq   += 1
        return _ArrivalEvent(
            arrival_time=arrival_time,
            seq=self._seq,
            tau=current_epoch,
            update=update,
        )
