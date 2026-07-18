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
       - Run local PyTorch training on that snapshot (wall-clock ignored),
         solving g_xt(x;z) = f(x;z) + (rho/2)||x-xt||^2 when rho > 0
       - Compute simulated finish time:
             arrival_time = t_dispatch + t_compute + t_upload
       - Push an ArrivalEvent(arrival_time, τ, trained_weights) onto the queue

  2. UPDATER LOOP  (process epochs 0 … T-1)
       - Pop the B earliest ArrivalEvents from the queue (B = algorithm.buffer_size,
         default 1 -> fully async, exactly the paper's Algorithm 1). Because the
         queue is a min-heap on arrival_time, these are — by construction — the B
         clients that finished first among the window_size currently in flight.
       - Compute staleness k_i = (current epoch) − τ_i for each of the B updates;
         representative staleness = max(k_i) (conservative: the worst update in
         the batch drives the mixing weight, see Remark 2 in the paper).
       - Compute alpha_t = algorithm.mixing_weight(base_alpha, max(k_i))
       - B == 1: call algorithm.aggregate_async(global_model, update, epoch, k, alpha_t)
         B  > 1: call algorithm.aggregate_buffered(global_model, updates, epoch, k_list, alpha_t)
           — semi-async: the B updates are combined into ONE global model update;
             the remaining (window_size - B) clients keep training uninterrupted.
       - Load result into global_model
       - Dispatch B replacement clients (sliding-window: always keep
         window_size clients in flight)

Sliding-window concurrency
--------------------------
The simulator keeps exactly `window_size` clients in flight at all times:
  - On startup: dispatch window_size clients simultaneously (all see model at epoch 0)
  - After each Updater step: dispatch B replacement clients (sees current model),
    where B is the number of updates just consumed (default 1)

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
  - Go semi-async (buffer B updates per model update)
                             → set algorithm.buffer_size = B > 1 and override
                               AsyncFederatedAlgorithm.aggregate_buffered()
                               (see FedAsyncTopKFastTotal)
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
from flsim.core.training_utils import effective_work_samples, local_iters
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
        forced_staleness: for controlled-staleness algorithms (those exposing
                      sample_dispatch_staleness), the k that was sampled at
                      dispatch and on whose stale model snapshot x_{t-k} the
                      client was actually trained. When set, the updater uses
                      this k as THE staleness (for the mixing weight and the
                      log) instead of the real timing-based staleness. None for
                      ordinary async, where staleness = arrival_epoch - tau.
    """
    arrival_time: float
    seq:          int
    tau:          int
    update:       Any
    forced_staleness: Any = None

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
        rho         (float): proximal regularization weight for the local worker
                             objective g_xt(x;z) = f(x;z) + (rho/2)||x-xt||^2
                             (paper §3). Default 0.0 (plain local SGD). An
                             algorithm's own `rho` attribute (e.g. FedAsync(rho=..))
                             takes priority over this YAML value.

    Buffering (semi-async): the algorithm's own `buffer_size` attribute (default
    1) controls how many arrivals are aggregated per global model update — this
    is set by the algorithm (e.g. FedAsyncTopKFastTotal(k=5)), not via YAML.
    Must satisfy buffer_size <= window_size (validated below).

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

        # Buffer size k (semi-async): algorithm.buffer_size, default 1 (fully async).
        # k=1 processes one arrival at a time (aggregate_async); k>1 buffers the
        # k fastest-to-arrive clients per epoch and aggregates them together
        # (aggregate_buffered). See AsyncFederatedAlgorithm / FedAsyncTopKFastTotal.
        self._buffer_size = int(getattr(server.algorithm, "buffer_size", 1))
        if self._buffer_size < 1:
            raise ValueError(
                f"{type(server.algorithm).__name__}.buffer_size must be >= 1, "
                f"got {self._buffer_size}."
            )
        if self._buffer_size > self._window_size:
            raise ValueError(
                f"{type(server.algorithm).__name__}.buffer_size (k={self._buffer_size}) "
                f"cannot exceed async_fl.window_size ({self._window_size}) — the "
                f"server can never buffer more arrivals than are ever in flight. "
                f"Increase window_size or decrease k."
            )

        # Rho priority: algorithm.rho > config.async_fl.rho > default 0.0
        # (proximal regularization weight for the local worker objective, paper §3)
        _alg_rho = getattr(server.algorithm, "rho", None)
        if _alg_rho is not None:
            self._rho = float(_alg_rho)
        else:
            self._rho = float(getattr(async_cfg, "rho", 0.0) if async_cfg else 0.0)

        # Effective bandwidth per client = total / window_size
        # (models uplink contention when window_size clients transmit concurrently)
        self._effective_bw = config.wireless.total_bandwidth_hz / max(1, self._window_size)

        # If downlink_negligible is set, model-broadcast (download) time is 0,
        # so arrival_time = dispatch_time + t_compute + t_upload only.
        self._downlink_negligible = bool(
            getattr(config.wireless, "downlink_negligible", False)
        )
        # Optional BS downlink power P^DL — same unified downlink convention
        # as the sync Simulator / SplitCostModel (rate at BS power + downlink
        # energy charged when set; None = original symmetric behaviour).
        self._downlink_tx_power_w = getattr(
            config.wireless, "downlink_tx_power_w", None
        )

        # Insertion counter for stable priority-queue tie-breaking
        self._seq = 0

        # Controlled-staleness support (e.g. FedAsyncSimulatedStaleness): if the
        # algorithm exposes sample_dispatch_staleness(), the simulator keeps a
        # rolling history of past global-model snapshots so a dispatched client
        # can be trained on a genuinely OLD model x_{t-k} (k sampled by the
        # algorithm), rather than the current one. This makes the sampled
        # staleness affect the actual update (stale model contributes), not just
        # the mixing weight. Disabled (no history kept, zero overhead) for
        # ordinary async algorithms, whose staleness already arises naturally
        # from real dispatch/arrival timing when window_size > 1.
        self._uses_forced_staleness = hasattr(server.algorithm, "sample_dispatch_staleness")
        self._model_history: list = []      # newest-last list of state_dict snapshots
        self._max_history = int(getattr(server.algorithm, "max_staleness", 0)) + 1
        self._stale_holder = None            # lazy scratch nn.Module for loading snapshots

    # ------------------------------------------------------------------
    # Controlled-staleness helpers
    # ------------------------------------------------------------------

    def _snapshot_global_model(self) -> None:
        """Append a detached CPU-independent clone of the current global model
        state_dict to the rolling history (bounded to _max_history)."""
        snap = {k: v.detach().clone() for k, v in self.server.global_model.state_dict().items()}
        self._model_history.append(snap)
        if len(self._model_history) > self._max_history:
            self._model_history.pop(0)

    def _resolve_training_model(self, current_epoch: int):
        """
        Return (model_to_train_on, forced_staleness).

        For controlled-staleness algorithms: ask the algorithm for a staleness k
        (bounded by how many snapshots we actually have), load the snapshot from
        k versions ago into a scratch model, and return (that_model, k). The
        client will train on the genuinely stale model x_{t-k}.

        For ordinary async: return (current global model, None).
        """
        if not self._uses_forced_staleness or not self._model_history:
            return self.server.global_model, None

        max_available = len(self._model_history) - 1   # 0 => only current model exists
        k = int(self.server.algorithm.sample_dispatch_staleness(current_epoch, max_available))
        k = max(0, min(k, max_available))
        snapshot = self._model_history[-1 - k]          # k versions back (k=0 => current)

        if self._stale_holder is None:
            self._stale_holder = copy.deepcopy(self.server.global_model)
        self._stale_holder.load_state_dict(snapshot)
        return self._stale_holder, k

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full async FL experiment.

        Stopping condition (mutually exclusive):
          - Default: run exactly learning.global_rounds server updates (global epochs).
          - If learning.stop_by_time_s is set (> 0): ignore global_rounds and
            instead run epochs until cumulative simulated_time_s reaches that
            budget. See _should_continue().

        Evaluates every evaluation.evaluate_every epochs.
        Saves plots at the end.
        """
        T           = self.config.learning.global_rounds
        cfg_eval    = self.config.evaluation

        stop_by_time_s = getattr(self.config.learning, "stop_by_time_s", None)
        time_based     = stop_by_time_s is not None and stop_by_time_s > 0

        if time_based:
            print(
                f"\n[AsyncSimulator] Starting FedAsync: "
                f"stop_by_time_s={stop_by_time_s:.0f}s (time-based), "
                f"window={self._window_size}, buffer_size={self._buffer_size}, "
                f"alpha={self._base_alpha}, rho={self._rho}, "
                f"staleness_func={getattr(self.server.algorithm, 'staleness_func', 'custom')}, "
                f"device={self.device}"
            )
        else:
            print(
                f"\n[AsyncSimulator] Starting FedAsync: "
                f"T={T} epochs, window={self._window_size}, buffer_size={self._buffer_size}, "
                f"alpha={self._base_alpha}, rho={self._rho}, "
                f"staleness_func={getattr(self.server.algorithm, 'staleness_func', 'custom')}, "
                f"device={self.device}"
            )
        print(
            f"[AsyncSimulator] Upload size: {self._upload_bits / 1e3:.1f} kbits "
            f"(mode={self.config.wireless.upload_size_mode}), "
            f"effective BW/client: {self._effective_bw / 1e6:.2f} MHz"
        )

        # ---- initial dispatch: fill the window ----
        # Extra kwargs passed to select_clients so custom selection policies that
        # need channel or bandwidth info can use them (unused by the default
        # uniform-random selection).
        selection_ctx = {
            "channel_model":     self.channel_model,
            "noise_psd_w_per_hz": self.config._noise_psd_w_per_hz,
            "bw_per_client_hz":  self._effective_bw,
            "upload_size_bits":  self._upload_bits,
        }

        # Seed the model history with the initial (epoch-0) global model so the
        # first dispatched clients can already be assigned a (small) staleness.
        if self._uses_forced_staleness:
            self._snapshot_global_model()

        pending: list = []
        initial_batch = self.server.algorithm.select_clients(
            self.clients, self._window_size, self.rng, **selection_ctx
        )
        for client in initial_batch:
            ev = self._dispatch_client(client, current_time=0.0, current_epoch=0)
            heapq.heappush(pending, ev)

        global_epoch    = 0
        simulated_time  = 0.0
        B               = self._buffer_size

        # ---- main updater loop ----
        while self._should_continue(global_epoch, simulated_time, T, time_based, stop_by_time_s):
            if not pending:
                print(
                    f"[AsyncSimulator] Warning: pending queue empty at epoch "
                    f"{global_epoch}. Stopping early."
                )
                break

            # Pop the B earliest-arriving updates (B=1: fully async, exactly the
            # old behavior. B>1: semi-async — these are, by construction of the
            # priority queue, the B clients that finished first among the
            # window_size currently in flight).
            batch_size   = min(B, len(pending))
            batch_events = [heapq.heappop(pending) for _ in range(batch_size)]
            simulated_time = max(ev.arrival_time for ev in batch_events)

            # Staleness per update: for controlled-staleness algorithms use the
            # k that was sampled at dispatch (and on whose stale model snapshot
            # the client was actually trained); otherwise the real timing-based
            # staleness = arrival_epoch - tau.
            stalenesses = [
                ev.forced_staleness if ev.forced_staleness is not None
                else global_epoch - ev.tau
                for ev in batch_events
            ]
            # Representative staleness for the batch's mixing weight: the WORST
            # (max) staleness, so one stale update in the batch isn't hidden by
            # fresher ones — conservative per paper Remark 2 (larger staleness
            # -> more error -> lower alpha).
            rep_staleness = max(stalenesses)
            alpha_t = self.server.algorithm.mixing_weight(self._base_alpha, rep_staleness)

            updates = [ev.update for ev in batch_events]
            if batch_size == 1:
                new_state = self.server.algorithm.aggregate_async(
                    self.server.global_model,
                    updates[0],
                    global_epoch,
                    stalenesses[0],
                    alpha_t,
                )
            else:
                new_state = self.server.algorithm.aggregate_buffered(
                    self.server.global_model,
                    updates,
                    global_epoch,
                    stalenesses,
                    alpha_t,
                )
            self.server.global_model.load_state_dict(new_state)
            self.server.round_idx = global_epoch + 1

            # Record the new global-model version so future dispatches can be
            # trained on it as a stale snapshot (controlled-staleness only).
            if self._uses_forced_staleness:
                self._snapshot_global_model()

            # Build result record. Single-arrival epochs (B=1) log that one
            # update's own fields, exactly as before. Buffered epochs (B>1) log
            # the batch as a whole: time fields use the batch's bottleneck (the
            # slowest of the B, which determined when the epoch could fire —
            # same convention as the sync Simulator's per-round max); energy
            # fields are summed across the batch (total energy spent this
            # epoch, same convention as the sync Simulator's per-round sum).
            if batch_size == 1:
                upd = updates[0]
                result = AsyncRoundResult(
                    global_epoch=global_epoch,
                    arrival_time_s=simulated_time,
                    staleness=stalenesses[0],
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
            else:
                result = AsyncRoundResult(
                    global_epoch=global_epoch,
                    arrival_time_s=simulated_time,
                    staleness=rep_staleness,
                    alpha_used=alpha_t,
                    client_id=[u.client_id for u in updates],
                    compute_time_s=max(u.compute_time_s for u in updates),
                    upload_time_s=max(u.upload_time_s for u in updates),
                    total_time_s=max(u.total_time_s for u in updates),
                    compute_energy_j=sum(u.compute_energy_j for u in updates),
                    tx_energy_j=sum(u.tx_energy_j for u in updates),
                    total_energy_j=sum(u.total_energy_j for u in updates),
                    channel_gain=sum(u.channel_gain for u in updates) / len(updates),
                    achievable_rate_bps=sum(u.achievable_rate_bps for u in updates) / len(updates),
                    train_loss=sum(u.train_loss for u in updates) / len(updates),
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
                    f"staleness={rep_staleness:3d} | "
                    f"α_t={alpha_t:.4f} | "
                    f"acc={eval_result.test_accuracy:.4f} | "
                    f"loss={eval_result.test_loss:.4f}"
                )

            global_epoch += 1

            # Dispatch batch_size replacements to keep the window full
            if self._should_continue(global_epoch, simulated_time, T, time_based, stop_by_time_s):
                replacement_batch = self.server.algorithm.select_clients(
                    self.clients, batch_size, self.rng, **selection_ctx
                )
                for client in replacement_batch:
                    new_ev = self._dispatch_client(
                        client,
                        current_time=simulated_time,
                        current_epoch=global_epoch,
                    )
                    heapq.heappush(pending, new_ev)

        print("[AsyncSimulator] Done. Saving plots …")
        self.logger.plot_results()

    @staticmethod
    def _should_continue(global_epoch: int, simulated_time: float, T: int,
                          time_based: bool, stop_by_time_s: float) -> bool:
        """Round-based: global_epoch < T. Time-based: simulated_time < stop_by_time_s."""
        if time_based:
            return simulated_time < stop_by_time_s
        return global_epoch < T

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
        # Per-round work (sample-passes): H*b when learning.local_iters is set,
        # else num_samples*local_epochs — same coherent quantity used for time,
        # energy, and (via max_iters) the actual local training below.
        work = effective_work_samples(cfg, client.num_samples)
        t_comp = self.time_model.compute_training_time(
            client.profile,
            num_samples=work,
            local_epochs=1,
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
                tx_power_w=self._downlink_tx_power_w,
            )

        # ---- energy ----
        e_comp = self.energy_model.compute_energy_j(
            client.profile,
            local_epochs=1,
            num_samples=work,
            cpu_freq_hz=f_hz,
        )
        e_tx = self.energy_model.transmission_energy_j(
            client.profile, upload_time_s=t_up, tx_power_w=p_w,
        )
        # Downlink TX energy P^DL·t_dn — charged only when a BS downlink power
        # is configured (matches SplitCostModel / sync Simulator convention).
        if self._downlink_tx_power_w is not None and t_dn > 0.0:
            e_tx += self._downlink_tx_power_w * t_dn

        # ---- local training (on snapshot of the global model the client received) ----
        # Ordinarily this is the CURRENT global model (staleness then arises
        # naturally from dispatch/arrival timing when window_size > 1). For a
        # controlled-staleness algorithm it is a genuinely OLD snapshot x_{t-k}
        # with k sampled by the algorithm, so the sampled staleness affects the
        # actual update, not just the mixing weight.
        train_model, forced_staleness = self._resolve_training_model(current_epoch)

        # Local objective solved here is g_xt(x; z) = f(x; z) + (rho/2)||x - xt||^2
        # (paper §3, Algorithm 1 "Process Worker"). rho comes from the algorithm's
        # `rho` attribute if set, else config.async_fl.rho, else 0.0 (plain SGD).
        # configure_client() may still override per-client via client.proximal_mu.
        self.server.algorithm.configure_client(
            client, train_model, current_epoch
        )
        state_dict, n_samples, train_loss = client.train(
            global_model=train_model,
            local_epochs=cfg.local_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            device=self.device,
            proximal_mu=getattr(client, "proximal_mu", self._rho),
            max_iters=local_iters(cfg),
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
            forced_staleness=forced_staleness,
        )
