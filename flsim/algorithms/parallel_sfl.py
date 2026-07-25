"""
algorithms/parallel_sfl.py: ParallelSFL — cluster-based split federated learning
that tackles system + statistical heterogeneity (Liao et al., ACM MobiCom '24).

Implements the paper's two contributions as the two clustering/frequency hooks
of flsim.interfaces.parallel_sfl_algorithm.ParallelSFLAlgorithm:

  cluster_workers()   -> Algorithm 1 (greedy worker clustering)
  assign_frequencies() -> Eq. (17) local-updating-frequency optimization

The intra-cluster relay training (Eq. 2/3), the bottom aggregation (Eq. 4,
inherited default = uniform mean), and the global aggregation (Eq. 18,
inherited default = N_c*tau_c weights) live in the base class / simulator — see
their docstrings. This file is purely the two heterogeneity-aware strategies,
operating on the per-worker WorkerInfo numbers the simulator measures.

Algorithm 1 (worker clustering, paper Sec 4.3)
----------------------------------------------
Goal: partition workers into clusters that are each (a) internally balanced in
COMPUTE/COMM time (small waiting time W_c, system heterogeneity) and (b) close
to IID in aggregate label distribution (small KL(Phi_c||Phi_0), statistical
heterogeneity), subject to the top worker's bandwidth (Eq. 8) and throughput
(Eq. 10) limits. The utility U_c = lambda*W_c + (1-lambda)*KL balances the two.

  1. K-means the workers by label distribution into K = N/5 sets.
  2. Greedily build clusters: for each new cluster, pick the highest-ingress-
     bandwidth worker from the largest remaining set as the top worker, then
     pull in workers (one candidate per set, the slowest by t_i) that most
     reduce KL(Phi_c||Phi_0), while respecting Eq. (8)/(10).
  3. Fine-tune: swap workers between clusters to further lower sum_c U_c,
     without violating the constraints.

Eq. (17) local updating frequency (paper Sec 4.4)
-------------------------------------------------
Faster clusters (smaller per-iteration completion time t_c,o) run MORE local
iterations tau_c so that every cluster's round-completion time
t_c = tau_c*t_c,o + beta_c is aligned to the fastest cluster running the
default maximum frequency tau_max — this shrinks the cross-cluster waiting time
under the synchronization barrier. tau_c is then also the per-cluster weight in
the global aggregation (Eq. 18), so a cluster that did more work counts more.
"""

from typing import List

import numpy as np

from flsim.interfaces.parallel_sfl_algorithm import (
    Cluster,
    ParallelSFLAlgorithm,
    WorkerInfo,
)

_EPS = 1e-12


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p||q) with epsilon smoothing (both are probability vectors)."""
    p = np.asarray(p, dtype=float) + _EPS
    q = np.asarray(q, dtype=float) + _EPS
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _kmeans_label_dists(dists: np.ndarray, k: int, rng, iters: int = 25):
    """
    Tiny K-means on the label-distribution vectors (Euclidean; a light,
    dependency-free stand-in for the paper's KL-based grouping — both put
    similar-distribution workers together). Returns a list of index lists,
    one per (non-empty) set.

    Args:
        dists (np.ndarray): (N, M) label distributions.
        k (int): number of sets.
        rng: numpy RandomState.
    """
    n = len(dists)
    k = max(1, min(k, n))
    centers = dists[rng.choice(n, k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        # assign
        d2 = ((dists[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            labels = new_labels
            break
        labels = new_labels
        # update
        for j in range(k):
            members = dists[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)
    sets = [list(np.where(labels == j)[0]) for j in range(k)]
    return [s for s in sets if s]


class ParallelSFL(ParallelSFLAlgorithm):
    """
    ParallelSFL with the paper's greedy Algorithm 1 clustering and Eq. (17)
    frequency optimization.

    Args:
        lam (float): lambda in the utility U_c = lambda*W_c + (1-lambda)*KL
            (Eq. 14). Higher => prioritise low waiting time (system
            heterogeneity); lower => prioritise IID clusters (statistical
            heterogeneity). Default 0.5.
        k_ratio (float): K = max(1, round(N * k_ratio)) sets for the K-means
            step (paper uses K = N/5, i.e. k_ratio = 0.2). Default 0.2.
        tau_max (int): default maximum local updating frequency assigned to the
            fastest cluster in Eq. (17); slower clusters get fewer iterations.
            Default 5.
        bandwidth_per_worker_hz (float): b in Eq. (8) — bandwidth one bottom
            worker occupies at the top worker; a cluster's bottom count is
            capped at floor(B_c / b). Default 5e6 (5 MHz). Set together with the
            per-worker ingress_bw_hz the simulator supplies.
        max_cluster_size (int): hard cap on bottom workers per cluster, applied
            on top of the Eq. (8)/(10) limits (safety / small-N runs). Default 8.
        fine_tune_passes (int): number of swap passes in Algorithm 1 line 11.
            Default 2.
    """

    def __init__(
        self,
        lam: float = 0.5,
        k_ratio: float = 0.2,
        tau_max: int = 5,
        bandwidth_per_worker_hz: float = 5.0e6,
        max_cluster_size: int = 8,
        fine_tune_passes: int = 2,
    ):
        super().__init__()
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        self.lam = lam
        self.k_ratio = k_ratio
        self.tau_max = tau_max
        self.b = bandwidth_per_worker_hz
        self.max_cluster_size = max_cluster_size
        self.fine_tune_passes = fine_tune_passes

    # ==================================================================
    # Hook 1 — Algorithm 1: worker clustering
    # ==================================================================

    def cluster_workers(self, workers: List[WorkerInfo], rng) -> List[Cluster]:
        n = len(workers)
        if n < 2:
            raise ValueError("ParallelSFL needs >= 2 workers (1 top + >= 1 bottom).")

        phi0 = np.mean([w.label_dist for w in workers], axis=0)   # IID reference
        dists = np.array([w.label_dist for w in workers])

        # Line 1-2: K-means on label distributions -> K sets (index lists into `workers`).
        K = max(1, round(n * self.k_ratio))
        sets = _kmeans_label_dists(dists, K, rng)
        remaining = [list(s) for s in sets]   # mutable copy

        clusters: List[Cluster] = []

        # Line 4-10: greedily build clusters.
        while any(remaining):
            # Line 5: top worker = max ingress bandwidth in the set with most workers.
            biggest = max((s for s in remaining if s), key=len)
            top_local = max(biggest, key=lambda gi: workers[gi].ingress_bw_hz)
            biggest.remove(top_local)
            top = workers[top_local]

            bottoms: List[WorkerInfo] = []
            phi_c = np.zeros_like(phi0)

            # Cap bottom count by the bandwidth constraint (Eq. 8) and the safety cap.
            cap = min(self.max_cluster_size,
                      max(1, int(top.ingress_bw_hz // max(self.b, 1.0))))

            # Line 8-9: keep adding the candidate (one per set) that most reduces
            # KL(Phi_c||Phi0), while respecting Eq. (8)/(10).
            while len(bottoms) < cap and any(remaining):
                # candidate pool A: from each non-empty set, the worker with the
                # largest t_i (slowest) — paper line 8.
                cand = []
                for s in remaining:
                    if not s:
                        continue
                    j = max(s, key=lambda gi: self._t_i(workers[gi], top))
                    cand.append(j)
                if not cand:
                    break
                # choose the candidate that minimises the resulting KL(Phi_c||Phi0)
                best_j, best_kl = None, None
                for j in cand:
                    trial_n = len(bottoms) + 1
                    trial_phi = (phi_c * len(bottoms) + workers[j].label_dist) / trial_n
                    trial_kl = _kl(trial_phi, phi0)
                    if best_kl is None or trial_kl < best_kl:
                        best_kl, best_j = trial_kl, j
                if best_j is None:
                    break
                # Eq. (10): adding best_j must keep the top from being the
                # bottleneck (top serves N_c batches; N_c*mu_p <= slowest bottom).
                if bottoms and not self._eq10_ok(top, bottoms + [workers[best_j]]):
                    break
                # commit
                bottoms.append(workers[best_j])
                phi_c = (phi_c * (len(bottoms) - 1) + workers[best_j].label_dist) / len(bottoms)
                for s in remaining:
                    if best_j in s:
                        s.remove(best_j)
                        break

            if not bottoms:
                # No bottoms could be added (tiny leftovers) — pull any remaining
                # worker so the top isn't wasted; if none remain, drop the top back.
                leftovers = [gi for s in remaining for gi in s]
                if leftovers:
                    j = leftovers[0]
                    bottoms.append(workers[j])
                    for s in remaining:
                        if j in s:
                            s.remove(j); break
                else:
                    # lone top worker with nothing to pair — attach to an existing
                    # cluster as a bottom (its data will train), or skip if none.
                    if clusters:
                        clusters[-1].bottoms.append(top)
                    continue
            clusters.append(Cluster(top=top, bottoms=bottoms))

        # Line 11: fine-tune by swapping to reduce sum U_c.
        self._fine_tune(clusters, phi0, workers)
        return clusters

    # ------------------------------------------------------------------
    # Algorithm-1 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _t_i(w: WorkerInfo, top: WorkerInfo) -> float:
        """t_i = mu_b,i + beta_i + mu_p,c  (paper Eq. 9)."""
        return w.mu_b + w.beta_smashed + top.mu_p

    def _eq10_ok(self, top: WorkerInfo, bottoms: List[WorkerInfo]) -> bool:
        """Eq. (10): N_c * mu_p,c <= max_i(mu_b,i + beta_i,c) — top not the bottleneck.

        Always allows a single bottom worker (n_c <= 1) so a cluster can never
        end up empty; only additional workers are gated by the throughput bound.
        """
        n_c = len(bottoms)
        if n_c <= 1:
            return True
        max_bottom = max(w.mu_b + w.beta_smashed for w in bottoms)
        return n_c * top.mu_p <= max_bottom + _EPS

    def _waiting_time(self, cluster: Cluster) -> float:
        """W_c = mean_i(t_c,o - t_i), t_c,o = max t_i  (paper Eq. 11)."""
        if not cluster.bottoms:
            return 0.0
        ts = [self._t_i(w, cluster.top) for w in cluster.bottoms]
        t_o = max(ts)
        return float(np.mean([t_o - t for t in ts]))

    def _utility(self, cluster: Cluster, phi0: np.ndarray) -> float:
        """U_c = lambda*W_c + (1-lambda)*KL(Phi_c||Phi0)  (paper Eq. 14).

        W_c is normalised by the max t_c,o so it is comparable to the KL term
        (both roughly in [0, 1]); this mirrors the paper's 'normalize W_c and KL'.
        """
        if not cluster.bottoms:
            return 0.0
        phi_c = np.mean([w.label_dist for w in cluster.bottoms], axis=0)
        kl = _kl(phi_c, phi0)
        ts = [self._t_i(w, cluster.top) for w in cluster.bottoms]
        t_o = max(ts) if ts else 1.0
        w_norm = (np.mean([t_o - t for t in ts]) / t_o) if t_o > 0 else 0.0
        return self.lam * w_norm + (1.0 - self.lam) * kl

    def _fine_tune(self, clusters: List[Cluster], phi0: np.ndarray, workers) -> None:
        """Line 11: try swapping bottom workers between cluster pairs to lower
        sum_c U_c, keeping Eq. (10) satisfied. A few greedy passes."""
        for _ in range(self.fine_tune_passes):
            improved = False
            for a in range(len(clusters)):
                for b in range(a + 1, len(clusters)):
                    ca, cb = clusters[a], clusters[b]
                    if not ca.bottoms or not cb.bottoms:
                        continue
                    base = self._utility(ca, phi0) + self._utility(cb, phi0)
                    best = None
                    for ia, wa in enumerate(ca.bottoms):
                        for ib, wb in enumerate(cb.bottoms):
                            na = ca.bottoms[:ia] + [wb] + ca.bottoms[ia + 1:]
                            nb = cb.bottoms[:ib] + [wa] + cb.bottoms[ib + 1:]
                            if not self._eq10_ok(ca.top, na) or not self._eq10_ok(cb.top, nb):
                                continue
                            trial = (self._utility(Cluster(ca.top, na), phi0)
                                     + self._utility(Cluster(cb.top, nb), phi0))
                            if trial < base - 1e-9 and (best is None or trial < best[0]):
                                best = (trial, ia, ib)
                    if best is not None:
                        _, ia, ib = best
                        ca.bottoms[ia], cb.bottoms[ib] = cb.bottoms[ib], ca.bottoms[ia]
                        improved = True
            if not improved:
                break

    # ==================================================================
    # Hook 2 — Eq. (17): local updating frequency
    # ==================================================================

    def assign_frequencies(self, clusters: List[Cluster]) -> None:
        """
        Set tau_c per cluster so round-completion times align to the fastest
        cluster running tau_max (Eq. 17). t_c,o = max_i t_i (slowest bottom's
        per-iter time); the fastest cluster (min t_c,o) gets tau_max, and
        slower clusters get fewer iterations:
            tau_c = clamp_1( floor( (tau_max*t_l,o + beta_l - beta_c) / t_c,o ) )
        """
        if not clusters:
            return
        t_o = []
        beta = []
        for c in clusters:
            ts = [self._t_i(w, c.top) for w in c.bottoms] or [c.top.mu_p]
            t_o.append(max(ts))
            beta.append(c.top.beta_full)
        l = int(np.argmin(t_o))           # fastest cluster
        t_ref = self.tau_max * t_o[l] + beta[l]
        for i, c in enumerate(clusters):
            if i == l:
                c.tau = self.tau_max
            else:
                raw = (t_ref - beta[i]) / max(t_o[i], _EPS)
                c.tau = max(1, int(np.floor(raw)))
