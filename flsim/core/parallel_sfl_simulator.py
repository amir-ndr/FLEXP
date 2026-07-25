"""
core/parallel_sfl_simulator.py: Orchestrator for ParallelSFL — cluster-based
split federated learning (Liao et al., ACM MobiCom '24).

The counterpart to flsim.core.split_simulator.SplitSimulator, but for the
cluster paradigm: workers are partitioned into clusters (each = 1 top worker
holding the TOP submodel + N_c bottom workers holding BOTTOM submodels), each
cluster runs an intra-cluster split-training loop (bottom workers exchange
smashed data / gradients with the top WORKER, not the PS), the top worker
aggregates its bottom submodels, and the PS aggregates the (bottom, top) pairs
across clusters. See flsim.interfaces.parallel_sfl_algorithm for the clustering
/ frequency / aggregation hooks and flsim.algorithms.parallel_sfl.ParallelSFL
for the paper-faithful strategies.

What this module owns (vs. the algorithm)
-----------------------------------------
The ALGORITHM decides who clusters with whom, how many local iterations each
cluster runs (tau_c), and how submodels combine. This SIMULATOR owns the
models and the mechanics:

  * The intra-cluster relay training loop (paper Eq. 2/3): a SHARED top submodel
    on the top worker, updated each iteration with the gradient AVERAGED over
    the cluster's N_c bottom workers (Eq. 3), while each bottom worker updates
    its own bottom submodel with its full gradient (Eq. 2). This reuses the
    exact detach()+requires_grad_() relay boundary proven bitwise-correct in
    flsim.core.split_client.
  * Global aggregation is done on the (bottom, top) pair directly: aggregating
    the cluster full models w_c = [w_b,c, w_p,c] with weights rho_c is, because
    the two param sets are disjoint, the same as FedAvg-ing the bottoms across
    clusters and the tops across clusters separately (Eq. 18). So no explicit
    "splicing" is needed — the global model is kept as (global_bottom, global_top).
  * The per-worker system quantities the clustering / Eq. 17 hooks need
    (label distribution V_i, compute times mu_b/mu_p, smashed/full transmission
    times beta) are measured here from the client partitions + system profiles
    + channel model, and handed to the algorithm as WorkerInfo. NOTE: the top
    submodel runs on a DEVICE (the top worker), so its compute uses the top
    worker's own CPU frequency — NOT a fast edge server (that is the whole
    point of ParallelSFL: no PS/edge-server compute bottleneck).
"""

import copy
from collections import OrderedDict
from dataclasses import dataclass
from itertools import cycle
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from flsim.core.split_simulator import _weighted_average_state_dicts
from flsim.interfaces.parallel_sfl_algorithm import ParallelSFLAlgorithm, WorkerInfo
from flsim.system.split_model import SplitFullModel

_BITS_PER_ELEMENT = 32


@dataclass
class ParallelSFLEpochResult:
    """Metrics for one ParallelSFL aggregation round."""
    round: int
    train_loss: float
    num_clusters: int
    num_workers_trained: int          # sum of N_c over clusters (top workers excluded)
    test_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    round_latency_s: float = 0.0      # max_c (tau_c * t_c,o + beta_c)  (Eq. 15/16)
    simulated_time_s: float = 0.0     # cumulative round_latency_s
    avg_waiting_time_s: float = 0.0   # mean over clusters of the intra-cluster waiting time (Eq. 11/16)
    traffic_bytes: float = 0.0        # this round's communication (smashed + models)
    cumulative_traffic_bytes: float = 0.0
    total_energy_j: float = 0.0       # this round's energy (see _round_cost / energy_scope)
    cumulative_energy_j: float = 0.0


class ParallelSFLSimulator:
    """
    Runs ParallelSFL for a configured number of global aggregation rounds.

    Args:
        clients (list[SplitClient]): all workers (each owns local data via
            .dataset / .indices). A worker chosen as a cluster's TOP worker does
            not train on its own data that round (paper Sec 4.3).
        bottom_model (nn.Module): initial global bottom submodel (from
            split_model()). Owned/mutated by this simulator.
        top_model (nn.Module): initial global top submodel.
        algorithm (ParallelSFLAlgorithm): clustering / frequency / aggregation
            strategy (e.g. flsim.algorithms.parallel_sfl.ParallelSFL).
        evaluator (Evaluator): evaluates the combined model on the test set.
        config: reads learning.{global_rounds, batch_size, learning_rate},
            evaluation.evaluate_every, wireless.total_bandwidth_hz, and (for the
            timing estimates) system + split settings.
        rng (np.random.RandomState): reproducibility for clustering.
        device (torch.device): training device.
        profiles (list, optional): one ClientSystemProfile per client (indexed
            by client_id) — supplies cpu_frequency / tx_power / distance for the
            WorkerInfo timing estimates. If None, timing is left at 0 and the
            algorithm must not depend on it.
        cost_model (SplitCostModel, optional): supplies the channel model (for
            link rates) + kappa/q. If None, transmission times are left at 0.

    This class does NOT:
    - Decide which dataset/model to use (both come in pre-built).
    - Implement clustering/frequency/aggregation — that is the algorithm's job.
    """

    def __init__(
        self,
        clients: list,
        bottom_model: nn.Module,
        top_model: nn.Module,
        algorithm: ParallelSFLAlgorithm,
        evaluator,
        config,
        rng: np.random.RandomState,
        device: torch.device,
        profiles: list = None,
        cost_model=None,
        num_classes: int = None,
    ):
        if not isinstance(algorithm, ParallelSFLAlgorithm):
            raise TypeError(
                f"ParallelSFLSimulator requires a ParallelSFLAlgorithm, got "
                f"{type(algorithm).__name__}."
            )
        self._num_classes_arg = num_classes
        self.clients = clients
        self.global_bottom = bottom_model.to(device)
        self.global_top = top_model.to(device)
        self.algorithm = algorithm
        self.evaluator = evaluator
        self.config = config
        self.rng = rng
        self.device = device
        self.profiles = profiles
        self.cost_model = cost_model
        self.history: list = []

        # Dedicated RNG for cost/channel draws, seeded independently of the
        # clustering RNG (same fairness rationale as SplitSimulator.cost_rng).
        seed = int(getattr(getattr(config, "experiment", None), "seed", 0))
        self.cost_rng = np.random.RandomState(seed)

        # ---- label distributions V_i (statistical heterogeneity) ----
        self._num_classes = self._infer_num_classes()
        self._label_dists = self._build_label_dists()

        # ---- split sizes for the timing estimates (measured once) ----
        self._bottom_param_count = sum(t.numel() for t in self.global_bottom.state_dict().values())
        self._top_param_count = sum(t.numel() for t in self.global_top.state_dict().values())
        self._activation_numel = 0
        self._device_compute_fraction = 0.5
        if self.cost_model is not None and self.profiles is not None:
            self._measure_split_sizes()

    # ------------------------------------------------------------------
    # One-time measurements
    # ------------------------------------------------------------------

    def _infer_num_classes(self) -> int:
        """
        Number of label classes M. Prefer an explicit num_classes (constructor
        arg or config.data.num_classes); otherwise infer ROBUSTLY from the FULL
        dataset — never a subset, since a non-IID client's first samples can
        miss the higher labels and undersize the histogram (IndexError).
        """
        if self._num_classes_arg is not None:
            return int(self._num_classes_arg)
        nc = getattr(getattr(self.config, "data", None), "num_classes", None)
        if nc:
            return int(nc)
        # Robust fallback: max label + 1 over the whole dataset (use .targets
        # when available, else scan every client's every index).
        ds = self.clients[0].dataset
        tgts = getattr(ds, "targets", None)
        if tgts is not None:
            arr = tgts.tolist() if hasattr(tgts, "tolist") else list(tgts)
            return int(max(int(t) for t in arr)) + 1
        m = 0
        for c in self.clients:
            for idx in c.indices:
                _, y = c.dataset[idx]
                m = max(m, int(y))
        return m + 1

    def _build_label_dists(self) -> dict:
        """V_i — normalized label histogram over M classes, per client."""
        dists = {}
        for c in self.clients:
            hist = np.zeros(self._num_classes, dtype=float)
            for idx in c.indices:
                _, y = c.dataset[idx]
                hist[int(y)] += 1.0
            total = hist.sum()
            dists[c.client_id] = hist / total if total > 0 else hist
        return dists

    def _measure_split_sizes(self) -> None:
        """Measure smashed-data size + bottom/top FLOP fraction from one batch."""
        from flsim.system.flops import measure_activation_and_split
        self._activation_numel, self._device_compute_fraction = measure_activation_and_split(
            self.global_bottom, self.global_top, self.clients[0], self.device
        )

    # ------------------------------------------------------------------
    # Per-round WorkerInfo (timing + label distribution) for the algorithm
    # ------------------------------------------------------------------

    def _build_worker_infos(self) -> List[WorkerInfo]:
        cfg = self.config.learning
        b = int(cfg.batch_size)
        phi = float(getattr(self.config.system, "cycles_per_sample_max", 1.0e7))  # C_k (FLOPs/sample)
        frac = self._device_compute_fraction
        q_dev = float(getattr(getattr(self.config, "split", None), "q_device", 1.0))
        total_bw = float(getattr(self.config.wireless, "total_bandwidth_hz", 2.0e7))
        bw_per = total_bw / max(1, len(self.clients))
        noise = getattr(self.config, "_noise_psd_w_per_hz", None)

        smashed_bits = self._activation_numel * _BITS_PER_ELEMENT
        full_bits = (self._bottom_param_count + self._top_param_count) * _BITS_PER_ELEMENT

        # Cache the per-worker system quantities used by _round_cost (freq, tx
        # power, link rate), drawn once here so cost and timing are consistent.
        self._sys = {}
        infos = []
        for c in self.clients:
            cid = c.client_id
            if self.profiles is not None and self.cost_model is not None and noise is not None:
                prof = self.profiles[cid]
                f = prof.cpu_frequency_hz
                gain = self.cost_model.channel_model.channel_gain(prof, self.cost_rng)
                rate = max(self.cost_model.channel_model.achievable_rate_bps(
                    bandwidth_hz=bw_per, tx_power_w=prof.tx_power_w,
                    channel_gain=gain, noise_psd_w_per_hz=noise), 1.0)
                mu_b = (frac * phi) * b / (f * q_dev)               # bottom compute / iter
                mu_p = ((1.0 - frac) * phi) * b / (f * q_dev)       # top compute / iter (this worker as top)
                beta_s = (smashed_bits * b) / rate                 # smashed tx / iter
                beta_f = full_bits / rate                          # full-model tx (this worker as top)
                ingress = total_bw
                self._sys[cid] = (f, prof.tx_power_w, rate)
            else:
                mu_b = mu_p = beta_s = beta_f = 0.0
                ingress = total_bw
                self._sys[cid] = (0.0, 0.0, 1.0)
            infos.append(WorkerInfo(
                client_id=cid, label_dist=self._label_dists[cid], num_samples=c.num_samples,
                mu_b=mu_b, mu_p=mu_p, beta_smashed=beta_s, beta_full=beta_f, ingress_bw_hz=ingress,
            ))
        return infos

    # ------------------------------------------------------------------
    # Per-round energy + traffic (same physical base as the other sims)
    # ------------------------------------------------------------------

    def _round_cost(self, clusters) -> tuple:
        """
        (traffic_bytes, energy_j) for one round, on the framework's physical
        base (DVFS compute energy kappa*f^2*FLOPs/q, tx energy p*t, byte counts).

        KEY ParallelSFL difference: the top submodel runs on the top WORKER — a
        resource-constrained DEVICE, not a plugged-in edge server. So under
        energy_scope="device" (battery) the top worker's compute + transmit
        energy DO count (there is no free server). Communication that crosses
        the PS is only the full model, once per cluster per round (the point of
        ParallelSFL: no per-sample smashed data on the PS link).
        """
        if self.cost_model is None or self.profiles is None:
            return 0.0, 0.0
        cfg = self.config.learning
        b = int(cfg.batch_size)
        phi = float(getattr(self.config.system, "cycles_per_sample_max", 1.0e7))
        frac = self._device_compute_fraction
        q = float(getattr(getattr(self.config, "split", None), "q_device", 1.0))
        kappa = float(self.config.system.switched_capacitance)
        scope = getattr(self.config.system, "energy_scope", "total")
        act_bytes = self._activation_numel * 4
        smashed_bits = self._activation_numel * _BITS_PER_ELEMENT
        bottom_bytes = self._bottom_param_count * 4
        full_bytes = (self._bottom_param_count + self._top_param_count) * 4
        bottom_bits = self._bottom_param_count * _BITS_PER_ELEMENT
        full_bits = full_bytes * 8

        traffic, energy = 0.0, 0.0
        for c in clusters:
            n_c, tau = c.size, c.tau
            work_bottom = tau * b            # sample-passes per bottom worker
            work_top = tau * n_c * b         # top serves all N_c bottoms
            # ---- bottom workers (devices) ----
            for w in c.bottoms:
                f, p, rate = self._sys[w.client_id]
                energy += kappa * (frac * phi) * work_bottom * (f ** 2) / q            # compute
                energy += p * (smashed_bits * work_bottom) / rate                     # smashed uplink
                energy += p * (bottom_bits) / rate                                    # bottom-model upload (agg)
            # ---- top worker (also a device) ----
            ft, pt, ratet = self._sys[c.top.client_id]
            energy += kappa * ((1.0 - frac) * phi) * work_top * (ft ** 2) / q          # top compute
            energy += pt * (smashed_bits * work_top) / ratet                          # gradient downlink to bottoms
            energy += pt * (full_bits) / ratet                                        # full-model upload to PS
            energy += pt * (bottom_bits * n_c) / ratet                                # bottom-model download to bottoms
            if scope != "device":
                # PS transmits the full model down to the top worker (infrastructure).
                energy += pt * (full_bits) / ratet
            # ---- traffic (both directions) ----
            traffic += n_c * (2 * act_bytes * work_bottom)     # smashed up + gradients down
            traffic += n_c * (2 * bottom_bytes)                # bottom model down + up (intra-cluster)
            traffic += 2 * full_bytes                          # full model PS<->top (once per cluster)
        return traffic, energy

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> list:
        cfg = self.config.learning
        cfg_eval = self.config.evaluation
        T = cfg.global_rounds

        print(f"\n[ParallelSFLSimulator] Starting {type(self.algorithm).__name__}: "
              f"T={T} rounds, {len(self.clients)} workers, device={self.device}")

        cum_time = 0.0
        cum_traffic = 0.0
        cum_energy = 0.0
        for rnd in range(T):
            worker_infos = self._build_worker_infos()
            clusters = self.algorithm.cluster_workers(worker_infos, self.rng)
            self.algorithm.assign_frequencies(clusters)

            g_bottom = self.global_bottom.state_dict()
            g_top = self.global_top.state_dict()

            cluster_bottoms, cluster_tops = [], []
            total_loss, total_samples, workers_trained = 0.0, 0, 0
            for c in clusters:
                w_b, w_p, loss, nsamp = self._train_cluster(c, g_bottom, g_top)
                cluster_bottoms.append(w_b)
                cluster_tops.append(w_p)
                total_loss += loss * nsamp
                total_samples += nsamp
                workers_trained += c.size

            # ---- global aggregation (Eq. 18): same rho_c for bottom & top ----
            rho = self.algorithm.global_weights(clusters)
            self.global_bottom.load_state_dict(_weighted_average_state_dicts(cluster_bottoms, rho))
            self.global_top.load_state_dict(_weighted_average_state_dicts(cluster_tops, rho))

            # ---- round latency (Eq. 15/16) + cross-cluster waiting time (Eq. 16) ----
            round_latency, avg_wait = self._round_timing(clusters)
            cum_time += round_latency
            traffic, energy = self._round_cost(clusters)
            cum_traffic += traffic
            cum_energy += energy
            mean_loss = total_loss / max(total_samples, 1)

            eval_result = None
            if rnd % cfg_eval.evaluate_every == 0:
                combined = SplitFullModel(self.global_bottom, self.global_top)
                eval_result = self.evaluator.evaluate(combined, device=self.device)

            self.history.append(ParallelSFLEpochResult(
                round=rnd, train_loss=mean_loss, num_clusters=len(clusters),
                num_workers_trained=workers_trained,
                test_loss=eval_result.test_loss if eval_result else None,
                test_accuracy=eval_result.test_accuracy if eval_result else None,
                round_latency_s=round_latency, simulated_time_s=cum_time,
                avg_waiting_time_s=avg_wait,
                traffic_bytes=traffic, cumulative_traffic_bytes=cum_traffic,
                total_energy_j=energy, cumulative_energy_j=cum_energy,
            ))
            if eval_result is not None:
                print(f"  Round {rnd:4d} | clusters={len(clusters)} | "
                      f"train_loss={mean_loss:.4f} | acc={eval_result.test_accuracy:.4f} | "
                      f"loss={eval_result.test_loss:.4f} | t={cum_time:.0f}s | "
                      f"traffic={cum_traffic/1e6:.1f}MB")

        print("[ParallelSFLSimulator] Done.")
        return self.history

    # ------------------------------------------------------------------
    # Intra-cluster training (paper Eq. 2/3): shared top + averaged gradient
    # ------------------------------------------------------------------

    def _train_cluster(self, cluster, global_bottom_state, global_top_state):
        """
        Train one cluster for tau_c local iterations. Returns
        (aggregated_bottom_state, top_state, mean_loss, total_samples).

        Each iteration: every bottom worker forwards a mini-batch -> smashed
        data -> the SHARED top submodel; the top submodel takes ONE SGD step
        with the gradient AVERAGED over the cluster's N_c bottom workers
        (Eq. 3), and each bottom worker takes one SGD step with its own full
        gradient (Eq. 2). After tau_c iterations the bottom submodels are
        averaged (Eq. 4, via algorithm.aggregate_bottom).
        """
        cfg = self.config.learning
        lr = cfg.learning_rate
        b = int(cfg.batch_size)
        n_c = cluster.size
        criterion = nn.CrossEntropyLoss()

        # One shared top submodel; one bottom submodel per bottom worker.
        top = copy.deepcopy(self.global_top).to(self.device)
        top.load_state_dict(global_top_state)
        top.train()
        top_opt = torch.optim.SGD(top.parameters(), lr=lr)

        bottoms, bottom_opts, loaders = [], [], []
        for w in cluster.bottoms:
            client = self.clients[w.client_id]
            bm = copy.deepcopy(self.global_bottom).to(self.device)
            bm.load_state_dict(global_bottom_state)
            bm.train()
            bottoms.append(bm)
            bottom_opts.append(torch.optim.SGD(bm.parameters(), lr=lr))
            loader = DataLoader(Subset(client.dataset, client.indices),
                                batch_size=b, shuffle=True, drop_last=False)
            loaders.append(cycle(loader))   # cycle so tau iters can exceed one epoch

        total_loss, steps = 0.0, 0
        for _ in range(max(1, cluster.tau)):
            # ---- forward all bottoms -> top; accumulate top grads over N_c ----
            top_opt.zero_grad()
            smasheds, relays = [], []
            for bm, ld in zip(bottoms, loaders):
                x, y = next(ld)
                x, y = x.to(self.device), y.to(self.device)
                smashed = bm(x)
                relay = smashed.detach().requires_grad_(True)
                out = top(relay)
                loss = criterion(out, y)
                loss.backward()          # accumulates top.grad (sum_i) and relay.grad (full for bottom i)
                smasheds.append(smashed)
                relays.append(relay)
                total_loss += loss.item()
                steps += 1
            # Eq. 3: top gradient averaged over the N_c bottom workers.
            for p in top.parameters():
                if p.grad is not None:
                    p.grad.div_(n_c)
            top_opt.step()
            # Eq. 2: each bottom worker backprops its FULL relay gradient and steps.
            for bm, opt, smashed, relay in zip(bottoms, bottom_opts, smasheds, relays):
                opt.zero_grad()
                smashed.backward(relay.grad)
                opt.step()

        # ---- Eq. 4: aggregate bottom submodels on the top worker ----
        bottom_states = [bm.state_dict() for bm in bottoms]
        num_samples = [self.clients[w.client_id].num_samples for w in cluster.bottoms]
        agg_bottom = self.algorithm.aggregate_bottom(bottom_states, num_samples)
        mean_loss = total_loss / max(steps, 1)
        return agg_bottom, top.state_dict(), mean_loss, sum(num_samples)

    # ------------------------------------------------------------------
    # Round latency (paper Eq. 15/16)
    # ------------------------------------------------------------------

    def _round_timing(self, clusters) -> tuple:
        """
        (round_latency, avg_waiting_time) for one round.

        round_latency = max_c t_c   where t_c = tau_c * t_c,o + beta_c (Eq. 15) —
            the round advances at the slowest cluster's completion time.
        avg_waiting_time = mean_c (max_c t_c - t_c)  (Eq. 16) — the cross-cluster
            idle time the Eq. 17 frequency optimization minimizes.
        """
        if not clusters:
            return 0.0, 0.0
        t_c = []
        for c in clusters:
            ts = [w.mu_b + w.beta_smashed + c.top.mu_p for w in c.bottoms] or [c.top.mu_p]
            t_c.append(c.tau * max(ts) + c.top.beta_full)
        t_h = max(t_c)
        avg_wait = float(np.mean([t_h - t for t in t_c]))
        return t_h, avg_wait
