"""
core/csa_sfl_simulator.py: Orchestrator for CSA-SFL (Clustered Semi-Asynchronous
Split Federated Learning).

Relationship to the other split simulators
-------------------------------------------
  * SplitSimulator / SplitAsyncSimulator (SAFSL) — device-level split FL on the
    edge server (PS), synchronous / semi-async over DEVICES.
  * ParallelSFLSimulator — cluster-level, but the top submodel runs on a peer
    DEVICE and inter-cluster aggregation is SYNCHRONOUS every round.
  * CSASFLSimulator (this) — cluster-level, the server-side submodel runs on the
    PS (one per cluster), and inter-cluster aggregation is SEMI-ASYNCHRONOUS:
    a discrete-event loop where each cluster COMPLETION triggers a global round.

Faithful to the paper (see flsim.algorithms.csa_sfl for the two PS mechanisms):
  * Intra-cluster split co-training is SYNCHRONOUS (SFLV1-style on the PS): one
    shared cluster server-side submodel updated each iteration with the
    DATA-SIZE-weighted sum of device server-gradients (Eq. server_update);
    device-side submodels update with their own gradient (Eq. device_update)
    and are data-size-averaged after E iterations (Eq. intra_agg). The cluster
    model is [w_bar_n^c, w_n^{s,E}] (Eq. form).
  * Inter-cluster aggregation is ASYNCHRONOUS with a BUFFER: the PS keeps each
    cluster's most-recent completed model and, when a cluster completes at
    global round t, aggregates ALL N buffered models with data-size-and-
    staleness-aware weights phi_{n,t} (Eq. phi / global_update). tau_n = t - t_n.
    The new global model is assigned ONLY to the completed cluster (its
    staleness resets to 0); all other clusters keep training, staleness += 1.
  * Dynamic clustering: cosine-distance K-means on device-side gradients, redone
    every H global rounds (a synchronous gradient-recollection halt).

Timing (fair with the SAFSL / SFLv2 baselines — same SplitCostModel base)
-------------------------------------------------------------------------
Each cluster's training duration for E iterations is its SFLV1-in-cluster cost:
max over the cluster's devices of SplitCostModel.device_cost(...).full_path_s,
with the edge server frequency f_S SHARED across all concurrently-training
devices (f_S / K), exactly like SAFSL's f_S/window sharing. Clusters run
concurrently; the earliest completion advances simulated_time_s. There is NO
inter-cluster waiting (async) — the only synchronous wait is intra-cluster
straggler (max - mean device path), reported as avg_waiting_time_s.
"""

import copy
import heapq
from dataclasses import dataclass
from itertools import cycle
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from flsim.core.split_simulator import _weighted_average_state_dicts
from flsim.system.split_model import SplitFullModel

_BITS_PER_ELEMENT = 32
_BYTES_PER_ELEMENT = 4


@dataclass
class CSASFLEpochResult:
    """Metrics for one CSA-SFL global aggregation round (a cluster completion)."""
    round: int
    train_loss: float
    num_clusters: int
    completed_cluster_size: int
    staleness: float = 0.0                    # completed cluster's tau_n^t (versions)
    test_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    round_latency_s: float = 0.0              # gap since the previous completion
    simulated_time_s: float = 0.0             # wall-clock of this completion
    avg_waiting_time_s: float = 0.0           # intra-cluster straggler wait (no inter-cluster wait)
    traffic_bytes: float = 0.0
    cumulative_traffic_bytes: float = 0.0
    total_energy_j: float = 0.0
    cumulative_energy_j: float = 0.0


class CSASFLSimulator:
    def __init__(
        self,
        clients: list,
        bottom_model: nn.Module,      # global device-side submodel w^c
        server_model: nn.Module,      # global server-side submodel w^s (on the PS)
        algorithm,                    # flsim.algorithms.csa_sfl.CSASFL
        evaluator,
        config,
        rng: np.random.RandomState,
        device: torch.device,
        profiles: list = None,
        cost_model=None,
    ):
        self.clients = clients
        self.global_bottom = bottom_model.to(device)
        self.global_server = server_model.to(device)
        self.algo = algorithm
        self.evaluator = evaluator
        self.config = config
        self.rng = rng
        self.device = device
        self.profiles = profiles
        self.cost_model = cost_model
        self.history: list = []

        seed = int(getattr(getattr(config, "experiment", None), "seed", 0))
        self.cost_rng = np.random.RandomState(seed)

        self.data_sizes = {c.client_id: int(c.num_samples) for c in self.clients}
        self._bottom_params = sum(t.numel() for t in self.global_bottom.state_dict().values())
        self._server_params = sum(t.numel() for t in self.global_server.state_dict().values())
        self._activation_numel = 0
        self._device_frac = 0.5
        if self.cost_model is not None and self.profiles is not None:
            from flsim.system.flops import measure_activation_and_split
            self._activation_numel, self._device_frac = measure_activation_and_split(
                self.global_bottom, self.global_server, self.clients[0], self.device
            )

    # ==================================================================
    # Public entry point — the discrete-event semi-async loop
    # ==================================================================

    def run(self) -> list:
        cfg = self.config.learning
        cfg_eval = self.config.evaluation
        T = int(self.algo.T if self.algo.T is not None else cfg.global_rounds)
        N = self.algo.N
        E = self.algo.E
        Hr = self.algo.H

        print(f"\n[CSASFLSimulator] Starting CSA-SFL: T={T} global rounds, N={N} clusters, "
              f"E={E} local iters, recluster every H={Hr}, {len(self.clients)} devices, device={self.device}")

        cum_time = 0.0
        cum_traffic = 0.0
        cum_energy = 0.0

        # ---- initial clustering (gradient K-means, or random one-time ablation) ----
        clusters, g0_dur, g0_traf, g0_energy = self._form_clusters()
        N = len(clusters)
        cum_time += g0_dur; cum_traffic += g0_traf; cum_energy += g0_energy

        # per-cluster state: the model each cluster trains FROM, its buffer, and
        # the global round at which it was last refreshed (for staleness).
        g_bottom = {k: v.clone() for k, v in self.global_bottom.state_dict().items()}
        g_server = {k: v.clone() for k, v in self.global_server.state_dict().items()}
        start = [self._clone_pair(g_bottom, g_server) for _ in clusters]
        buffer = [self._clone_pair(g_bottom, g_server) for _ in clusters]
        last_refresh = [0 for _ in clusters]

        # schedule every cluster's first completion (all start at cum_time)
        heap = []   # (completion_time, seq, cluster_idx)
        seq = 0
        durations = [self._cluster_duration_cost(c)[0] for c in clusters]
        for n in range(N):
            heapq.heappush(heap, (cum_time + durations[n], seq, n)); seq += 1

        prev_time = cum_time
        t = 0
        while t < T and heap:
            comp_time, _, n = heapq.heappop(heap)
            cum_time = comp_time

            # ---- train the completed cluster (E iters) from its start model ----
            b_agg, s_state, loss, wait = self._train_cluster(clusters[n], start[n], E)
            buffer[n] = self._clone_pair(b_agg, s_state)

            # ---- staleness of EVERY cluster at this global round t ----
            stale = [t - last_refresh[m] for m in range(N)]
            phi = self.algo.agg_weights(
                [self._cluster_data_size(clusters[m]) for m in range(N)], stale
            )

            # ---- global aggregation of ALL buffered models (Eq. global_update) ----
            new_bottom = _weighted_average_state_dicts([buffer[m][0] for m in range(N)], list(phi))
            new_server = _weighted_average_state_dicts([buffer[m][1] for m in range(N)], list(phi))
            self.global_bottom.load_state_dict(new_bottom)
            self.global_server.load_state_dict(new_server)

            # ---- assign the new global model ONLY to the completed cluster ----
            start[n] = self._clone_pair(new_bottom, new_server)
            last_refresh[n] = t + 1                      # tau_n^{t+1} = 0 (paper)

            # ---- cost of the completed cluster's training this round ----
            dur_n, traf_n, energy_n = self._cluster_duration_cost(clusters[n])
            cum_traffic += traf_n
            cum_energy += energy_n

            # ---- reschedule the completed cluster (keeps training, async) ----
            heapq.heappush(heap, (cum_time + dur_n, seq, n)); seq += 1

            # ---- evaluate periodically on the aggregated global model ----
            eval_result = None
            if t % cfg_eval.evaluate_every == 0 or t == T - 1:
                combined = SplitFullModel(self.global_bottom, self.global_server)
                eval_result = self.evaluator.evaluate(combined, device=self.device)

            self.history.append(CSASFLEpochResult(
                round=t, train_loss=loss, num_clusters=N,
                completed_cluster_size=len(clusters[n]), staleness=float(stale[n]),
                test_loss=eval_result.test_loss if eval_result else None,
                test_accuracy=eval_result.test_accuracy if eval_result else None,
                round_latency_s=cum_time - prev_time, simulated_time_s=cum_time,
                avg_waiting_time_s=wait,
                traffic_bytes=traf_n, cumulative_traffic_bytes=cum_traffic,
                total_energy_j=energy_n, cumulative_energy_j=cum_energy,
            ))
            if eval_result is not None:
                print(f"  Round {t:4d} | cluster {n} (size {len(clusters[n])}, stale {stale[n]}) "
                      f"| loss={loss:.4f} | acc={eval_result.test_accuracy:.4f} "
                      f"| t={cum_time:.0f}s | traffic={cum_traffic/1e6:.1f}MB")
            prev_time = cum_time
            t += 1

            # ---- dynamic re-clustering every H global rounds (gradient mode only;
            #      random one-time clustering never re-clusters) ----
            if t < T and self.algo.clustering == "gradient" and t % Hr == 0:
                clusters, gc_dur, gc_traf, gc_energy = self._form_clusters()
                N = len(clusters)
                cum_time += gc_dur; cum_traffic += gc_traf; cum_energy += gc_energy
                g_bottom = {k: v.clone() for k, v in self.global_bottom.state_dict().items()}
                g_server = {k: v.clone() for k, v in self.global_server.state_dict().items()}
                # all (re-formed) clusters restart from the current global model
                start = [self._clone_pair(g_bottom, g_server) for _ in clusters]
                buffer = [self._clone_pair(g_bottom, g_server) for _ in clusters]
                last_refresh = [t for _ in clusters]
                heap = []
                for m in range(N):
                    dur_m = self._cluster_duration_cost(clusters[m])[0]
                    heapq.heappush(heap, (cum_time + dur_m, seq, m)); seq += 1
                prev_time = cum_time

        print("[CSASFLSimulator] Done.")
        return self.history

    def _form_clusters(self):
        """Form clusters + return the synchronous-halt cost (duration, traffic,
        energy). Gradient mode collects device-side gradients (a co-training halt)
        then cosine K-means; the random ablation clusters once with no halt."""
        if self.algo.clustering == "random":
            clusters = self.algo.random_clusters([c.client_id for c in self.clients], self.rng)
            return clusters, 0.0, 0.0, 0.0
        dur, traf, energy = self._collect_gradients_cost()
        gradients = self._collect_gradients()
        clusters = self.algo.cluster_by_gradient(gradients, self.rng)
        return clusters, dur, traf, energy

    # ==================================================================
    # Intra-cluster split co-training (paper Eq. server_update / device_update /
    # intra_agg) — one shared server-side, data-size-weighted server gradient,
    # per-device device-sides averaged by data size.
    # ==================================================================

    def _train_cluster(self, cluster_cids, start_pair, E):
        cfg = self.config.learning
        lr = float(cfg.learning_rate)
        b = int(cfg.batch_size)
        criterion = nn.CrossEntropyLoss()
        start_bottom, start_server = start_pair
        Dn = float(self._cluster_data_size(cluster_cids)) or 1.0

        # shared cluster server-side submodel (on the PS)
        server = copy.deepcopy(self.global_server).to(self.device)
        server.load_state_dict(start_server)
        server.train()
        server_opt = torch.optim.SGD(server.parameters(), lr=lr)

        # one device-side submodel per device, all initialized from w^{c,t_n}
        bottoms, bopts, loaders, weights = [], [], [], []
        for cid in cluster_cids:
            client = self.clients[cid]
            bm = copy.deepcopy(self.global_bottom).to(self.device)
            bm.load_state_dict(start_bottom)
            bm.train()
            bottoms.append(bm)
            bopts.append(torch.optim.SGD(bm.parameters(), lr=lr))
            loaders.append(cycle(DataLoader(Subset(client.dataset, client.indices),
                                            batch_size=b, shuffle=True, drop_last=False)))
            weights.append(float(self.data_sizes[cid]) / Dn)   # |D_k| / |D_n|

        total_loss, steps = 0.0, 0
        for _ in range(max(1, E)):
            server_opt.zero_grad()
            smasheds, relays = [], []
            for bm, ld, wk in zip(bottoms, loaders, weights):
                x, y = next(ld)
                x, y = x.to(self.device), y.to(self.device)
                smashed = bm(x)
                relay = smashed.detach().requires_grad_(True)
                loss = criterion(server(relay), y)
                # scale by |D_k|/|D_n| so the ACCUMULATED server grad is
                # sum_k (|D_k|/|D_n|) g_k^s  (Eq. server_update). The relay grad
                # is then w_k * (full device grad) — undone below.
                (wk * loss).backward()
                smasheds.append(smashed); relays.append(relay)
                total_loss += loss.item(); steps += 1
            server_opt.step()                                   # w^{s,e+1} (Eq. server_update)
            # device-side updates use each device's OWN full gradient (Eq. device_update)
            for bm, opt, smashed, relay, wk in zip(bottoms, bopts, smasheds, relays, weights):
                opt.zero_grad()
                full_relay_grad = relay.grad / wk if wk > 0 else relay.grad
                smashed.backward(full_relay_grad)
                opt.step()

        # data-size-weighted aggregation of device-side submodels (Eq. intra_agg)
        b_states = [bm.state_dict() for bm in bottoms]
        n_samples = [self.data_sizes[cid] for cid in cluster_cids]
        b_agg = _weighted_average_state_dicts(b_states, n_samples)
        wait = self._intra_cluster_wait(cluster_cids)
        return b_agg, server.state_dict(), total_loss / max(steps, 1), wait

    # ==================================================================
    # Gradient collection for clustering (one co-training pass per device)
    # ==================================================================

    def _collect_gradients(self) -> dict:
        """Device-side gradient g_k for every device, from one co-training pass on
        the CURRENT global model (paper's initial/periodic gradient recollection)."""
        cfg = self.config.learning
        b = int(cfg.batch_size)
        criterion = nn.CrossEntropyLoss()
        server = copy.deepcopy(self.global_server).to(self.device); server.eval()
        grads = {}
        for c in self.clients:
            bm = copy.deepcopy(self.global_bottom).to(self.device); bm.train()
            loader = DataLoader(Subset(c.dataset, c.indices), batch_size=b, shuffle=True)
            x, y = next(iter(loader))
            x, y = x.to(self.device), y.to(self.device)
            bm.zero_grad()
            smashed = bm(x)
            relay = smashed.detach().requires_grad_(True)
            loss = criterion(server(relay), y)
            loss.backward()                                   # relay.grad = server-returned smashed grad
            smashed.backward(relay.grad)                      # -> device-side gradient g_k
            g = torch.cat([p.grad.detach().flatten() for p in bm.parameters() if p.grad is not None])
            grads[c.client_id] = g.cpu().numpy()
        return grads

    # ==================================================================
    # Cost / timing (same SplitCostModel base as SAFSL/SFLv2 -> fair)
    # ==================================================================

    def _server_freq_shared(self):
        """f_S shared across ALL concurrently-training devices (f_S / K), matching
        SAFSL's f_S/window sharing. None if no cost model."""
        if self.cost_model is None:
            return None
        shared = bool(getattr(getattr(self.config, "split", None), "server_frequency_shared", True))
        K = max(1, len(self.clients))
        return (self.cost_model.f_server / K) if shared else None

    def _device_cost(self, cid, work_samples):
        from flsim.core.training_utils import effective_work_samples
        cfg = self.config.learning
        prof = self.profiles[cid]
        bw_per = self.config.wireless.total_bandwidth_hz / max(1, len(self.clients))
        gain = self.cost_model.channel_model.channel_gain(prof, self.cost_rng)
        return self.cost_model.device_cost(
            profile=prof, num_samples=self.data_sizes[cid], local_epochs=cfg.local_epochs,
            cycles_per_sample=prof.cycles_per_sample, device_compute_fraction=self._device_frac,
            activation_numel=self._activation_numel, client_param_count=self._bottom_params,
            bandwidth_hz=bw_per, channel_gain=gain, work_samples=work_samples,
            server_freq_hz=self._server_freq_shared(),
        )

    def _cluster_duration_cost(self, cluster_cids):
        """(duration_s, traffic_bytes, energy_j) for one cluster training E iters.
        Duration = max device full_path (SFLV1 in-cluster); traffic/energy sum."""
        if self.cost_model is None or self.profiles is None:
            return 0.0, 0.0, 0.0
        work = self.algo.E * int(self.config.learning.batch_size)
        paths, traffic, energy = [], 0.0, 0.0
        for cid in cluster_cids:
            dc = self._device_cost(cid, work)
            paths.append(dc.full_path_s)
            traffic += dc.traffic_bytes
            energy += dc.total_energy_j
        return (max(paths) if paths else 0.0), traffic, energy

    def _intra_cluster_wait(self, cluster_cids) -> float:
        """Intra-cluster straggler wait = max - mean device full_path (the only
        synchronous wait; there is NO inter-cluster waiting under async)."""
        if self.cost_model is None or self.profiles is None:
            return 0.0
        work = self.algo.E * int(self.config.learning.batch_size)
        paths = [self._device_cost(cid, work).full_path_s for cid in cluster_cids]
        return (max(paths) - float(np.mean(paths))) if paths else 0.0

    def _collect_gradients_cost(self):
        """(duration, traffic, energy) of one synchronous gradient-recollection
        halt: every device does ONE co-training iteration; duration = max device
        full_path (all halt together)."""
        if self.cost_model is None or self.profiles is None:
            return 0.0, 0.0, 0.0
        work = int(self.config.learning.batch_size)   # one mini-batch
        paths, traffic, energy = [], 0.0, 0.0
        for c in self.clients:
            dc = self._device_cost(c.client_id, work)
            paths.append(dc.full_path_s); traffic += dc.traffic_bytes; energy += dc.total_energy_j
        return (max(paths) if paths else 0.0), traffic, energy

    # ==================================================================
    # helpers
    # ==================================================================

    def _cluster_data_size(self, cluster_cids) -> int:
        return int(sum(self.data_sizes[cid] for cid in cluster_cids))

    @staticmethod
    def _clone_pair(bottom_state, server_state):
        return ({k: v.clone() for k, v in bottom_state.items()},
                {k: v.clone() for k, v in server_state.items()})
