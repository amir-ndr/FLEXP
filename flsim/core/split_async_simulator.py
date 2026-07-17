"""
core/split_async_simulator.py: Discrete-event simulator for asynchronous /
semi-asynchronous split federated learning.

Mirrors flsim.core.async_simulator.AsyncSimulator's discrete-event
priority-queue design (see that module's docstring for the general
Scheduler/Updater pattern), adapted for split learning's two-sided
(client-side, server-side) model pair — the same adaptation
flsim.core.split_simulator.SplitSimulator makes to the SYNCHRONOUS Simulator.
This is the asynchronous counterpart to SplitSimulator, exactly as
AsyncSimulator is the asynchronous counterpart to Simulator.

Why a new simulator (not reusing AsyncSimulator or SplitSimulator)
--------------------------------------------------------------------
AsyncSimulator aggregates ONE model's state_dict per arrival. SplitSimulator
trains a (client-side, server-side) pair via the forward/backward relay (see
flsim.core.split_client.SplitClient) but is purely synchronous — every
selected client trains and is awaited together, every global epoch. Neither
fits "clients train continuously and independently; the server aggregates
whenever a subset finishes; stragglers keep training uninterrupted" for a
two-sided model — hence this module.

Per-device timing/energy/traffic — NOT re-derived here
--------------------------------------------------------
Every per-device term (model download, device FP+BP, smashed-data uplink,
server FP+BP, gradient downlink, model upload — and the matching DVFS
compute energy / uplink TX energy / traffic byte counts) is computed by
flsim.system.split_cost.SplitCostModel.device_cost() — the exact same cost
model flsim.core.split_simulator.SplitSimulator (synchronous SL/SFLV1/SFLV2)
already uses, verified term-by-term against the semi-async split-FL paper's
eq. (6)-(13). What's new here is only the ORCHESTRATION: when each device is
dispatched, when its update arrives relative to others, and how arriving
updates combine — the eq. (14)-(18) / Fig. 2(a) semi-async workflow layered
on top of that same per-device physics. A device's own total round-trip time
(model-down + H co-training iterations of FP/upload/server-compute/download/
BP + model-up) is exactly `SplitCostModel.device_cost(...).full_path_s` — H
arises naturally from local_epochs x num_batches, exactly as it does for the
synchronous engine.

Simulation model (mirrors AsyncSimulator; see that module's docstring for
the general pattern)
--------------------------------------------------------------------------
  DISPATCH — select_clients() picks a device
           — snapshot (client_model, server_model) at the current epoch tau
           — train it via SplitClient.train_local() (the verified
             forward/backward relay, unchanged from the sync engine)
           — compute this device's own arrival_time = dispatch_time +
             device_cost.full_path_s
           — push an ArrivalEvent onto the priority queue

  UPDATER  — pop the B earliest arrivals (B = algorithm.buffer_size, the
             paper's |S_t|; B=1 is the paper's own "fully asynchronous"
             special case)
           — staleness_i = global_epoch - tau_i (same convention as
             AsyncSimulator/FedAsync)
           — algorithm.aggregate_buffered(...) combines the B arriving pairs
             into the new global (client_model, server_model) — see
             flsim.interfaces.split_async_algorithm for exactly what this
             does and does not do
           — dispatch B replacements, keeping window_size devices in flight

Extensibility — same three independent override points as
AsyncFederatedAlgorithm (see flsim.interfaces.split_async_algorithm):
  - WHO trains       -> SplitAsyncAlgorithm.select_clients()
  - HOW MUCH each
    arrival counts    -> SplitAsyncAlgorithm.participation_weight()
  - HOW arrivals
    combine           -> SplitAsyncAlgorithm.aggregate_buffered()
  - Resource allocation (bandwidth/power/frequency) is a SEPARATE pluggable
    component (`allocator`, a ResourceAllocator), independent of the
    algorithm — exactly as in AsyncSimulator/Simulator.
"""

import copy
import heapq
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from flsim.core.training_utils import effective_work_samples, local_iters
from flsim.system.flops import measure_activation_and_split
from flsim.interfaces.split_async_algorithm import SplitAsyncAlgorithm
from flsim.system.split_model import SplitFullModel


# ---------------------------------------------------------------------------
# Internal event type for the priority queue
# ---------------------------------------------------------------------------

@dataclass
class _SplitArrivalEvent:
    """One entry in the pending-arrivals priority queue."""
    arrival_time: float
    seq:          int
    tau:          int
    client_id:    int
    client_state_dict: Any
    server_state_dict: Any
    num_samples:  int
    train_loss:   float
    device_cost:  Any     # DevicePerRound — see flsim.system.split_cost

    def __lt__(self, other):
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        return self.seq < other.seq


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SplitAsyncEpochResult:
    """
    Metrics for one global epoch (= one aggregation event, consuming
    buffer_size arrivals) of asynchronous/semi-asynchronous split learning.

    Field names/semantics mirror flsim.core.async_logger.AsyncRoundResult
    (staleness, client_id, train_loss, energy) and
    flsim.core.split_simulator.SplitEpochResult (round_latency_s,
    simulated_time_s, traffic_bytes, cumulative_*) so a future
    logger/experiment layer can reuse the same CSV/plotting conventions.
    """
    global_epoch: int
    arrival_time_s: float
    staleness: int              # representative (max over the batch)
    client_id: Any               # single id (buffer_size=1) or list (>1)
    train_loss: float
    num_clients: int
    test_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    round_latency_s: float = 0.0          # max full_path_s among the batch
    simulated_time_s: float = 0.0         # = arrival_time_s (cumulative virtual time)
    traffic_bytes: float = 0.0            # sum over the batch
    cumulative_traffic_bytes: float = 0.0
    total_energy_j: float = 0.0           # sum over the batch
    cumulative_energy_j: float = 0.0


# ---------------------------------------------------------------------------
# SplitAsyncSimulator
# ---------------------------------------------------------------------------

class SplitAsyncSimulator:
    """
    Asynchronous/semi-asynchronous split-FL simulation orchestrator.

    Args:
        clients (list[SplitClient]): all participating clients.
        client_model (nn.Module): initial global client-side sub-model (from
            split_model()). Owned/mutated by this simulator.
        server_model (nn.Module): initial global server-side sub-model.
        algorithm (SplitAsyncAlgorithm): controls client selection,
            per-device weighting, and how buffered arrivals combine — see
            flsim.interfaces.split_async_algorithm. Pass e.g.
            flsim.algorithms.safsl.SAFSL(k=5, gamma=1.0).
        evaluator (Evaluator): evaluates the combined model on the test set.
        cost_model (SplitCostModel): supplies per-device latency/energy/
            traffic — the SAME cost model the synchronous SplitSimulator
            uses (flsim.system.split_cost).
        profiles (list): one ClientSystemProfile per client (indexed by
            client_id) — CPU frequency, transmit power for the cost model.
        channel_model: read from cost_model.channel_model (not a separate
            argument — mirrors SplitSimulator's own convention).
        allocator (ResourceAllocator): per-dispatch bandwidth/power/frequency
            allocation — e.g. flsim.allocators.equal_split.EqualSplitAllocator.
            Independent of `algorithm` — swap it to change resource policy
            without touching selection/aggregation logic.
        config: SimpleNamespace config. Reads learning.{local_epochs,
            batch_size, learning_rate, global_rounds, stop_by_time_s},
            evaluation.evaluate_every, wireless.total_bandwidth_hz,
            wireless.downlink_negligible, async_fl.window_size (defaults to
            len(clients) if absent — same fallback as AsyncSimulator).
        rng (np.random.RandomState): controls client selection/ordering.
        device (torch.device): training device.

    This class does NOT:
    - Decide which dataset/model to use (both come in pre-built).
    - Implement any specific selection/weighting/combination policy — that's
      entirely `algorithm`'s job (see flsim.interfaces.split_async_algorithm).
    """

    def __init__(
        self,
        clients: list,
        client_model,
        server_model,
        algorithm: SplitAsyncAlgorithm,
        evaluator,
        cost_model,
        profiles: list,
        allocator,
        config,
        rng: np.random.RandomState,
        device: torch.device,
    ):
        if not isinstance(algorithm, SplitAsyncAlgorithm):
            raise TypeError(
                f"SplitAsyncSimulator requires a SplitAsyncAlgorithm, got "
                f"{type(algorithm).__name__}. Pass e.g. "
                f"flsim.algorithms.safsl.SAFSL(...) via algorithm=..."
            )

        self.clients      = clients
        self.client_model = client_model.to(device)
        self.server_model = server_model.to(device)
        self.algorithm    = algorithm
        self.evaluator    = evaluator
        self.cost_model   = cost_model
        self.profiles     = profiles
        self.allocator    = allocator
        self.config       = config
        self.rng          = rng
        self.device       = device
        self.history: list = []

        async_cfg = getattr(config, "async_fl", None)
        self._window_size = int(
            getattr(async_cfg, "window_size", len(clients)) if async_cfg else len(clients)
        )

        self._buffer_size = int(getattr(algorithm, "buffer_size", 1))
        if self._buffer_size < 1:
            raise ValueError(f"{type(algorithm).__name__}.buffer_size must be >= 1, got {self._buffer_size}")
        if self._buffer_size > self._window_size:
            raise ValueError(
                f"{type(algorithm).__name__}.buffer_size (k={self._buffer_size}) "
                f"cannot exceed async_fl.window_size ({self._window_size})."
            )

        self._effective_bw = config.wireless.total_bandwidth_hz / max(1, self._window_size)
        self._downlink_negligible = bool(getattr(config.wireless, "downlink_negligible", False))

        # Measured once (don't change across rounds) — same quantities
        # SplitSimulator measures via its own _measure_split_sizes().
        self._client_param_count = sum(t.numel() for t in self.client_model.state_dict().values())
        self._activation_numel, self._device_compute_fraction = measure_activation_and_split(
            self.client_model, self.server_model, clients[0], device
        )

        self._seq = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> list:
        """
        Execute the full asynchronous split-FL run.

        Stopping condition (mutually exclusive, same convention as
        AsyncSimulator):
          - Default: run exactly learning.global_rounds aggregations.
          - If learning.stop_by_time_s is set (> 0): run until cumulative
            simulated_time_s reaches that budget instead.
        """
        cfg_learn = self.config.learning
        cfg_eval  = self.config.evaluation
        T = cfg_learn.global_rounds

        stop_by_time_s = getattr(cfg_learn, "stop_by_time_s", None)
        time_based = stop_by_time_s is not None and stop_by_time_s > 0

        print(
            f"\n[SplitAsyncSimulator] Starting {type(self.algorithm).__name__}: "
            + (f"stop_by_time_s={stop_by_time_s:.0f}s" if time_based else f"T={T} epochs")
            + f", window={self._window_size}, buffer_size={self._buffer_size}, device={self.device}"
        )

        pending: list = []
        initial_batch = self.algorithm.select_clients(self.clients, self._window_size, self.rng)
        for client in initial_batch:
            ev = self._dispatch_client(client, current_time=0.0, current_epoch=0)
            heapq.heappush(pending, ev)

        global_epoch   = 0
        simulated_time = 0.0
        B = self._buffer_size

        while self._should_continue(global_epoch, simulated_time, T, time_based, stop_by_time_s):
            if not pending:
                print(f"[SplitAsyncSimulator] Warning: pending queue empty at epoch {global_epoch}. Stopping early.")
                break

            batch_size = min(B, len(pending))
            batch_events = [heapq.heappop(pending) for _ in range(batch_size)]
            simulated_time = max(ev.arrival_time for ev in batch_events)

            stalenesses = [global_epoch - ev.tau for ev in batch_events]
            rep_staleness = max(stalenesses)

            client_state_dicts = [ev.client_state_dict for ev in batch_events]
            server_state_dicts = [ev.server_state_dict for ev in batch_events]
            num_samples_list   = [ev.num_samples for ev in batch_events]

            new_client_state, new_server_state = self.algorithm.aggregate_buffered(
                client_state_dicts=client_state_dicts,
                server_state_dicts=server_state_dicts,
                num_samples_list=num_samples_list,
                stalenesses=stalenesses,
                global_epoch=global_epoch,
            )
            self.client_model.load_state_dict(new_client_state)
            self.server_model.load_state_dict(new_server_state)

            traffic = sum(ev.device_cost.traffic_bytes for ev in batch_events)
            energy  = sum(ev.device_cost.total_energy_j for ev in batch_events)
            round_latency = max(ev.device_cost.full_path_s for ev in batch_events)
            mean_loss = sum(ev.train_loss * ev.num_samples for ev in batch_events) / max(sum(num_samples_list), 1)

            self._cum_traffic = getattr(self, "_cum_traffic", 0.0) + traffic
            self._cum_energy  = getattr(self, "_cum_energy", 0.0) + energy

            eval_result = None
            if global_epoch % cfg_eval.evaluate_every == 0:
                combined = SplitFullModel(self.client_model, self.server_model)
                eval_result = self.evaluator.evaluate(combined, device=self.device)

            result = SplitAsyncEpochResult(
                global_epoch=global_epoch,
                arrival_time_s=simulated_time,
                staleness=rep_staleness,
                client_id=batch_events[0].client_id if batch_size == 1 else [e.client_id for e in batch_events],
                train_loss=mean_loss,
                num_clients=batch_size,
                test_loss=eval_result.test_loss if eval_result else None,
                test_accuracy=eval_result.test_accuracy if eval_result else None,
                round_latency_s=round_latency,
                simulated_time_s=simulated_time,
                traffic_bytes=traffic,
                cumulative_traffic_bytes=self._cum_traffic,
                total_energy_j=energy,
                cumulative_energy_j=self._cum_energy,
            )
            self.history.append(result)

            if eval_result is not None:
                print(
                    f"  Epoch {global_epoch:5d} | sim_time={simulated_time:10.2f}s | "
                    f"staleness={rep_staleness:3d} | acc={eval_result.test_accuracy:.4f} | "
                    f"loss={eval_result.test_loss:.4f}"
                )

            global_epoch += 1

            if self._should_continue(global_epoch, simulated_time, T, time_based, stop_by_time_s):
                replacements = self.algorithm.select_clients(self.clients, batch_size, self.rng)
                for client in replacements:
                    new_ev = self._dispatch_client(client, current_time=simulated_time, current_epoch=global_epoch)
                    heapq.heappush(pending, new_ev)

        print("[SplitAsyncSimulator] Done.")
        return self.history

    @staticmethod
    def _should_continue(global_epoch, simulated_time, T, time_based, stop_by_time_s) -> bool:
        if time_based:
            return simulated_time < stop_by_time_s
        return global_epoch < T

    # ------------------------------------------------------------------
    # Internal: dispatch one client
    # ------------------------------------------------------------------

    def _dispatch_client(self, client, current_time: float, current_epoch: int) -> _SplitArrivalEvent:
        """
        Dispatch one client: snapshot the current global (client_model,
        server_model), train it via the relay, compute its own arrival time
        via SplitCostModel.device_cost(...).full_path_s, and return an
        ArrivalEvent for the priority queue. Mirrors
        AsyncSimulator._dispatch_client's "train eagerly now, queue the
        result for a later arrival_time" pattern.
        """
        cid = client.client_id
        cfg = self.config.learning
        profile = self.profiles[cid]

        gain = self.cost_model.channel_model.channel_gain(profile, self.rng)
        bw_alloc = self.allocator.allocate_bandwidth([profile], self._effective_bw, channel_gains={cid: gain})
        pw_alloc = self.allocator.allocate_power([profile], profile.tx_power_w, channel_gains={cid: gain})
        fq_alloc = self.allocator.allocate_cpu_freq([profile], profile.cpu_frequency_hz, channel_gains={cid: gain})
        bw_hz = bw_alloc[cid]

        # Snapshot the global pair at dispatch time (tau = current_epoch) —
        # deep copies, since SplitClient.train_local() mutates in place.
        client_model_copy = copy.deepcopy(self.client_model)
        server_model_copy = copy.deepcopy(self.server_model)

        c_sd, s_sd, n_samples, train_loss = client.train_local(
            client_model=client_model_copy,
            server_model=server_model_copy,
            local_epochs=cfg.local_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            device=self.device,
            max_iters=local_iters(cfg),
        )

        device_cost = self.cost_model.device_cost(
            profile=profile,
            num_samples=client.num_samples,
            local_epochs=cfg.local_epochs,
            cycles_per_sample=profile.cycles_per_sample,
            device_compute_fraction=self._device_compute_fraction,
            activation_numel=self._activation_numel,
            client_param_count=self._client_param_count,
            bandwidth_hz=bw_hz,
            channel_gain=gain,
            work_samples=effective_work_samples(cfg, client.num_samples),
        )

        arrival_time = current_time + device_cost.full_path_s
        self._seq += 1
        return _SplitArrivalEvent(
            arrival_time=arrival_time,
            seq=self._seq,
            tau=current_epoch,
            client_id=cid,
            client_state_dict=c_sd,
            server_state_dict=s_sd,
            num_samples=n_samples,
            train_loss=train_loss,
            device_cost=device_cost,
        )
