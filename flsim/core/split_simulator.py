"""
core/split_simulator.py: Orchestrator for split learning (SL / SFLV1 / SFLV2),
per Thapa, Chamikara Mahawaga Arachchige, Camtepe & Sun, "SplitFed: When
Federated Learning Meets Split Learning" (AAAI-22, arXiv:2004.12088),
Algorithm 1 & 2, and the "Variants of Splitfed Learning" discussion.

Why a new simulator (not FederatedAlgorithm.aggregate())
------------------------------------------------------------
Every other algorithm in this framework (FedAvg, FedProx, FedAsync, FedOTA)
trains one FULL model per client and aggregates complete state_dicts. Split
learning trains TWO cooperating sub-models (client-side, server-side) via a
forward/backward RELAY across a simulated communication boundary (see
flsim.core.split_client.SplitClient) — a fundamentally different mechanic
that doesn't fit the aggregate(global_model, client_updates) contract. So,
exactly as AsyncSimulator exists alongside Simulator for async's different
mechanics, SplitSimulator exists alongside both for split learning's.

The client_mode / server_mode framing
------------------------------------------
The paper describes three variants, but they reduce to two independent,
reusable choices — how the CLIENT side is combined across clients, and how
the SERVER side is:

  "sequential"      : ONE persistent model instance. Clients are processed
                       one at a time (order re-shuffled each global epoch);
                       each client's local training continues directly from
                       wherever the previous client left off. No aggregation
                       is ever computed — there's only ever one copy.
  "parallel_fedavg"  : Every selected client gets an INDEPENDENT deep copy of
                       the current global sub-model, trains it locally for
                       `local_epochs` epochs, and at the end of the global
                       epoch all copies are combined via sample-weighted
                       averaging (FedAvg): W_{t+1} = sum_k (n_k/n) W_{k,t}.

The paper's three named variants are exactly these two axes:

  SL     (client_mode="sequential",      server_mode="sequential")
         Relay-based: literally one client-side and one server-side model,
         handed from client to client. Matches Table 1's "Client-side
         training: Sequential" and "Model aggregation: No" for SL.
  SFLV1  (client_mode="parallel_fedavg", server_mode="parallel_fedavg")
         "the server-side models of all clients are executed separately in
         parallel and then aggregated to obtain the global server-side model
         at each global epoch" (paper, "Variants of Splitfed Learning").
         Client-side aggregated by the fed server the same way.
  SFLV2  (client_mode="parallel_fedavg", server_mode="sequential")
         "SFLV2 processes the forward-backward propagations of the
         server-side model sequentially with respect to the client's smashed
         data (no FedAvg of the server-side models). The client order is
         chosen randomly in the server-side operations" — client-side
         unchanged from SFLV1 ("The client-side operation remains the same
         as in the SFLV1").

This orthogonal framing is also why the module docstring you're reading
promises this is easy to extend: a hypothetical "SplitFedAvg" variant, or any
other custom combination, is just a different choice on these same two axes
(or a custom aggregation-weight function — see `weight_fn`) — no new
orchestration code needed.

System-cost metrics (latency / traffic / energy)
--------------------------------------------------
Pass a `cost_model` (flsim.system.split_cost.SplitCostModel) + per-client
`profiles` to have each round's simulated latency, communication traffic
(bytes), and energy computed on the SAME physical base as the sync/async/OTA
simulators — FDMA Shannon rate for links, DVFS (kappa·f²) for compute energy,
tx_power·time for uplink energy — so all paradigms are directly comparable.
What differs for split learning is only the WORKFLOW (device FP/BP + smashed
uplink + server FP/BP + gradient downlink + device-model up/down) and that
server-side compute runs at a faster edge-server frequency; the device/server
compute split at the cut layer is measured automatically (flsim.system.flops).
These land in SplitEpochResult (simulated_time_s, round_latency_s,
traffic_bytes, cumulative_energy_j, …) and the run's CSV. Omit the cost model
and those metrics stay 0 (the algorithm still runs).

The ALGORITHM itself (forward/backward relay, the three aggregation patterns,
cut_layer) is verified exact against the paper — see flsim/core/split_client.py's
docstring for the proof that split-relay training is numerically IDENTICAL to
training the unsplit model directly.
"""

import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from flsim.core.training_utils import effective_work_samples, local_iters
from flsim.system.split_model import SplitFullModel


@dataclass
class SplitEpochResult:
    """Metrics for one global epoch of split-learning training."""
    global_epoch: int
    train_loss: float          # sample-weighted mean training loss this epoch
    num_clients: int           # number of clients processed this epoch
    test_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    # System-cost metrics (0 when no cost_model is attached) — computed on the
    # same physical base as the sync/async/OTA simulators; see SplitCostModel.
    round_latency_s:         float = 0.0   # this round's simulated duration
    simulated_time_s:        float = 0.0   # cumulative simulated time
    traffic_bytes:           float = 0.0   # bytes communicated this round
    cumulative_traffic_bytes: float = 0.0
    total_energy_j:          float = 0.0   # energy this round
    cumulative_energy_j:     float = 0.0


def _weighted_average_state_dicts(state_dicts: list, weights: list) -> OrderedDict:
    """
    Sample-weighted average of a list of state_dicts (paper's FedAvg
    aggregation, applied identically to whichever side is in parallel_fedavg
    mode): W = sum_k (n_k / sum(n)) * W_k.

    Args:
        state_dicts: list[OrderedDict], all with identical keys/shapes.
        weights:     list[float], same length (e.g. num_samples per client).

    Returns:
        OrderedDict: the weighted-average state_dict (detached float32 tensors).
    """
    total = float(sum(weights))
    assert total > 0, "weights must sum to > 0"
    agg = OrderedDict()
    ref = state_dicts[0]
    for key, param in ref.items():
        agg[key] = torch.zeros_like(param, dtype=torch.float32)
    for sd, w in zip(state_dicts, weights):
        frac = w / total
        for key, param in sd.items():
            agg[key] += frac * param.float().detach()
    return agg


class SplitSimulator:
    """
    Runs split learning (SL / SFLV1 / SFLV2, or any custom client_mode /
    server_mode combination) for a configured number of global epochs.

    Args:
        clients (list[SplitClient]): all participating clients.
        client_model (nn.Module): initial global client-side sub-model
            (from split_model()). Owned/mutated by this simulator.
        server_model (nn.Module): initial global server-side sub-model.
        evaluator (Evaluator): evaluates the combined model on the test set.
        config: SimpleNamespace config (reads learning.global_rounds,
            learning.local_epochs, learning.batch_size, learning.learning_rate,
            evaluation.evaluate_every; learning.clients_per_round selects a
            subset each epoch if set, else all clients participate — matching
            the paper's own experimental setting "all participants update the
            model in each global epoch, i.e. C=1").
        rng (np.random.RandomState): controls client-order shuffling
            (sequential modes) and client selection (if clients_per_round < N).
        device (torch.device): training device.
        client_mode (str): "sequential" | "parallel_fedavg" — see module docstring.
        server_mode (str): "sequential" | "parallel_fedavg" — see module docstring.
        cost_model (SplitCostModel, optional): if given, per-round latency,
            traffic, and energy are computed on the same physical base as the
            sync/async/OTA simulators (see flsim.system.split_cost). Requires
            `profiles`. If None, those metrics stay 0 (algorithm still runs).
        profiles (list, optional): one ClientSystemProfile per client (same
            order as `clients`) — supplies each device's CPU frequency, transmit
            power, and channel gain for the cost model.

    This class does NOT:
    - Decide which dataset/model to use (both come in pre-built).
    """

    def __init__(
        self,
        clients: list,
        client_model: nn.Module,
        server_model: nn.Module,
        evaluator,
        config,
        rng: np.random.RandomState,
        device: torch.device,
        client_mode: str = "parallel_fedavg",
        server_mode: str = "parallel_fedavg",
        cost_model=None,
        profiles: list = None,
    ):
        valid_modes = ("sequential", "parallel_fedavg")
        if client_mode not in valid_modes or server_mode not in valid_modes:
            raise ValueError(
                f"client_mode/server_mode must be one of {valid_modes}, "
                f"got client_mode={client_mode!r}, server_mode={server_mode!r}"
            )
        self.clients      = clients
        self.client_model = client_model.to(device)
        self.server_model = server_model.to(device)
        self.evaluator     = evaluator
        self.config        = config
        self.rng           = rng
        self.device        = device
        self.client_mode   = client_mode
        self.server_mode   = server_mode
        self.cost_model    = cost_model
        self.profiles      = profiles
        self.history: list = []   # list[SplitEpochResult], filled by run()

        # Dedicated RNG for the cost model's per-round channel draws, seeded
        # independently of the training RNG. This is essential for FAIR
        # comparison: SL/SFLV1/SFLV2 advance the training RNG by different
        # amounts (different training paths), so without a separate stream they
        # would see DIFFERENT channel realizations. Seeding this identically
        # (from experiment.seed) makes every variant see the SAME channels.
        seed = int(getattr(getattr(config, "experiment", None), "seed", 0))
        self.cost_rng = np.random.RandomState(seed)

        # Measure the split-specific sizes ONCE (they don't change across rounds):
        #   activation_numel        — smashed-data elements per sample (client output)
        #   client_param_count      — device-side model size in elements, counted
        #       from the state_dict (params + buffers, e.g. BatchNorm running
        #       stats) — the SAME accounting the sync/async simulators use for
        #       model transfer, so device-model traffic is consistent across
        #       paradigms even for models that carry buffers.
        #   device_compute_fraction — share of FLOPs on the device side of the cut
        self._activation_numel = 0
        self._client_param_count = sum(t.numel() for t in self.client_model.state_dict().values())
        self._device_compute_fraction = 0.5
        if self.cost_model is not None:
            self._measure_split_sizes()

    def _cost_mode(self) -> str:
        """Map (client_mode, server_mode) → cost-model variant key."""
        if self.client_mode == "sequential" and self.server_mode == "sequential":
            return "sl"
        if self.client_mode == "parallel_fedavg" and self.server_mode == "sequential":
            return "sflv2"
        return "sflv1"   # parallel_fedavg × parallel_fedavg (and any other combo)

    def _measure_split_sizes(self) -> None:
        """Run one real batch through the split to measure smashed-data size and
        the device/server FLOP split (see flsim.system.flops)."""
        from torch.utils.data import DataLoader, Subset
        from flsim.system.flops import compute_split_fraction
        c0 = self.clients[0]
        loader = DataLoader(Subset(c0.dataset, c0.indices[: min(8, len(c0.indices))]),
                            batch_size=min(8, len(c0.indices)))
        x, _ = next(iter(loader))
        x = x.to(self.device)
        with torch.no_grad():
            smashed = self.client_model(x)
        self._activation_numel = smashed[0].numel()   # per sample
        self._device_compute_fraction = compute_split_fraction(
            self.client_model, self.server_model, x
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> list:
        """
        Execute global_rounds global epochs. Returns self.history (also
        stored on the instance for later inspection/plotting).
        """
        cfg_learn = self.config.learning
        cfg_eval  = self.config.evaluation
        T = cfg_learn.global_rounds
        k = int(getattr(cfg_learn, "clients_per_round", 0) or len(self.clients))
        k = min(k, len(self.clients))

        mode_label = f"client={self.client_mode}, server={self.server_mode}"
        cost_label = f", cost_model={self._cost_mode()}" if self.cost_model else ""
        print(f"\n[SplitSimulator] Starting: T={T} epochs, {k}/{len(self.clients)} "
              f"clients/epoch, {mode_label}{cost_label}, device={self.device}")

        cum_time = 0.0
        cum_traffic = 0.0
        cum_energy = 0.0

        for epoch in range(T):
            selected = self._select_clients(k)

            if self.server_mode == "sequential":
                mean_loss = self._run_epoch_server_sequential(selected)
            else:
                mean_loss = self._run_epoch_server_parallel_fedavg(selected)

            # ---- system cost for this round (same physical base as other sims) ----
            round_cost = self._round_cost(selected)
            cum_time    += round_cost.latency_s
            cum_traffic += round_cost.traffic_bytes
            cum_energy  += round_cost.total_energy_j

            eval_result = None
            if epoch % cfg_eval.evaluate_every == 0:
                combined = SplitFullModel(self.client_model, self.server_model)
                eval_result = self.evaluator.evaluate(combined, device=self.device)

            result = SplitEpochResult(
                global_epoch=epoch,
                train_loss=mean_loss,
                num_clients=len(selected),
                test_loss=eval_result.test_loss if eval_result else None,
                test_accuracy=eval_result.test_accuracy if eval_result else None,
                round_latency_s=round_cost.latency_s,
                simulated_time_s=cum_time,
                traffic_bytes=round_cost.traffic_bytes,
                cumulative_traffic_bytes=cum_traffic,
                total_energy_j=round_cost.total_energy_j,
                cumulative_energy_j=cum_energy,
            )
            self.history.append(result)

            if eval_result is not None:
                extra = (f" | t={cum_time:.0f}s | traffic={cum_traffic/1e6:.1f}MB | "
                         f"E={cum_energy:.1f}J") if self.cost_model else ""
                print(
                    f"  Epoch {epoch:4d} | train_loss={mean_loss:.4f} | "
                    f"acc={eval_result.test_accuracy:.4f} | loss={eval_result.test_loss:.4f}{extra}"
                )

        print("[SplitSimulator] Done.")
        return self.history

    # ------------------------------------------------------------------
    # Per-round system cost (latency / traffic / energy)
    # ------------------------------------------------------------------

    def _round_cost(self, selected: list):
        """Compute this round's SplitRoundCost via the cost model, or an all-zero
        cost if no cost model is attached."""
        from flsim.system.split_cost import SplitRoundCost
        if self.cost_model is None or self.profiles is None:
            return SplitRoundCost(latency_s=0.0, traffic_bytes=0.0, total_energy_j=0.0)

        cfg = self.config.learning
        bw_per_client = self.config.wireless.total_bandwidth_hz / max(1, len(selected))
        per_device = []
        for client in selected:
            profile = self.profiles[client.client_id]
            gain = self.cost_model.channel_model.channel_gain(profile, self.cost_rng)
            per_device.append(self.cost_model.device_cost(
                profile=profile,
                num_samples=client.num_samples,
                local_epochs=cfg.local_epochs,
                cycles_per_sample=profile.cycles_per_sample,
                device_compute_fraction=self._device_compute_fraction,
                activation_numel=self._activation_numel,
                client_param_count=self._client_param_count,
                bandwidth_hz=bw_per_client,
                channel_gain=gain,
                work_samples=effective_work_samples(cfg, client.num_samples),
            ))
        return self.cost_model.combine(self._cost_mode(), per_device)

    # ------------------------------------------------------------------
    # Client selection
    # ------------------------------------------------------------------

    def _select_clients(self, k: int) -> list:
        """Uniform random selection of k of N clients (default k=N, matching
        the paper's C=1 full-participation setting)."""
        if k >= len(self.clients):
            return list(self.clients)
        indices = self.rng.choice(len(self.clients), size=k, replace=False)
        return [self.clients[i] for i in indices]

    # ------------------------------------------------------------------
    # One global epoch — server_mode="sequential" branch
    # (covers SL: client_mode="sequential" too, and SFLV2:
    #  client_mode="parallel_fedavg")
    # ------------------------------------------------------------------

    def _run_epoch_server_sequential(self, selected: list) -> float:
        cfg = self.config.learning
        order = self.rng.permutation(len(selected))

        client_state_dicts, client_weights = [], []
        total_loss, total_samples = 0.0, 0

        for idx in order:
            client = selected[idx]

            # client_mode governs whether this client trains an independent
            # copy of the client-side model (parallel_fedavg, aggregated at
            # the end of the epoch) or the ONE shared instance directly (SL).
            if self.client_mode == "parallel_fedavg":
                client_model = copy.deepcopy(self.client_model)
            else:  # "sequential"
                client_model = self.client_model

            # server_mode="sequential": always the SAME shared server_model
            # object, mutated in place across clients in this random order —
            # this is the defining trait of both SL and SFLV2's server side.
            c_sd, s_sd, n, loss = client.train_local(
                client_model=client_model,
                server_model=self.server_model,
                local_epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                device=self.device,
                max_iters=local_iters(cfg),
            )
            # server_model already updated in place — nothing further to do
            # for the server side.

            if self.client_mode == "parallel_fedavg":
                client_state_dicts.append(c_sd)
                client_weights.append(n)
            # else ("sequential"): self.client_model IS client_model, already
            # updated in place — nothing further to do for the client side.

            total_loss += loss * n
            total_samples += n

        if self.client_mode == "parallel_fedavg":
            new_client_state = _weighted_average_state_dicts(client_state_dicts, client_weights)
            self.client_model.load_state_dict(new_client_state)

        return total_loss / max(total_samples, 1)

    # ------------------------------------------------------------------
    # One global epoch — server_mode="parallel_fedavg" branch
    # (covers SFLV1: client_mode="parallel_fedavg" too)
    # ------------------------------------------------------------------

    def _run_epoch_server_parallel_fedavg(self, selected: list) -> float:
        cfg = self.config.learning

        client_state_dicts, client_weights = [], []
        server_state_dicts, server_weights = [], []
        total_loss, total_samples = 0.0, 0

        for client in selected:
            # Every client gets an independent copy of BOTH sides, starting
            # from this epoch's global snapshot — "executed separately in
            # parallel" per the paper.
            client_model = (
                copy.deepcopy(self.client_model) if self.client_mode == "parallel_fedavg"
                else self.client_model
            )
            server_model = copy.deepcopy(self.server_model)

            c_sd, s_sd, n, loss = client.train_local(
                client_model=client_model,
                server_model=server_model,
                local_epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                device=self.device,
                max_iters=local_iters(cfg),
            )

            if self.client_mode == "parallel_fedavg":
                client_state_dicts.append(c_sd)
                client_weights.append(n)
            server_state_dicts.append(s_sd)
            server_weights.append(n)

            total_loss += loss * n
            total_samples += n

        if self.client_mode == "parallel_fedavg":
            new_client_state = _weighted_average_state_dicts(client_state_dicts, client_weights)
            self.client_model.load_state_dict(new_client_state)

        new_server_state = _weighted_average_state_dicts(server_state_dicts, server_weights)
        self.server_model.load_state_dict(new_server_state)

        return total_loss / max(total_samples, 1)
