"""
algorithms/csa_sfl.py: CSA-SFL (Clustered Semi-Asynchronous Split Federated
Learning) parameters + the two PS-side mechanisms.

This class is a lightweight strategy/params holder. The event-driven
semi-asynchronous orchestration (per-cluster server-side submodels, buffered +
staleness-weighted global aggregation, cluster completion scheduling, dynamic
re-clustering) lives in flsim.core.csa_sfl_simulator.CSASFLSimulator.

The two mechanisms it owns, both run on the PS:
  1. cluster_by_gradient — dynamic device clustering by GRADIENT SIMILARITY:
     cosine-distance K-means on the device-side gradients g_k (paper's
     "min_{C} sum_n sum_{k in C_n} (1 - cos(g_k, mu_n))"). Coherent-gradient
     devices are grouped, reducing intra-cluster update inconsistency.
  2. agg_weights — data-size-aware AND staleness-aware inter-cluster weights
     phi_{n,t} = |D_n|/(1+tau_n) / sum_n' |D_n'|/(1+tau_n')  (paper Eq. phi),
     giving fresher (small tau) and larger (big |D_n|) clusters more influence.

Arguments (N, H, E, T) — as requested:
    N (num_clusters)     — number of clusters the K devices are partitioned into.
    H (recluster_every)  — re-collect gradients and re-cluster every H GLOBAL
                           aggregation rounds (paper's "every H rounds").
    E (local_iters)      — intra-cluster local work: E mini-batch co-training
                           iterations per cluster training pass. Kept as
                           ITERATIONS (not full epochs) so CSA-SFL's per-round
                           local work equals the other baselines' H mini-batch
                           steps in the comparison experiment (fair timing).
    T (global_rounds)    — number of global aggregation rounds (cluster
                           completions) to simulate.
"""

import numpy as np


class CSASFL:
    def __init__(
        self,
        num_clusters: int = 5,
        recluster_every: int = 100,
        local_iters: int = 5,
        global_rounds: int = None,
        kmeans_iters: int = 50,
        clustering: str = "gradient",
        aggregation: str = "weighted",
    ):
        self.N = int(num_clusters)          # N — number of clusters
        self.H = int(recluster_every)       # H — re-cluster every H global rounds
        self.E = int(local_iters)           # E — intra-cluster local mini-batch iterations
        self.T = global_rounds              # T — number of global rounds (None -> read config)
        self.kmeans_iters = int(kmeans_iters)
        # Ablation switches (Exp 3): the full method is gradient + weighted.
        #   clustering  = "gradient" (cosine K-means, re-clustered every H) |
        #                 "random"   (random ONE-TIME clustering, no re-clustering)
        #   aggregation = "weighted" (phi = |D_n|/(1+tau_n), normalized)     |
        #                 "uniform"  (phi = 1/N — drops data-size AND staleness)
        self.clustering = clustering
        self.aggregation = aggregation
        if self.N < 1:
            raise ValueError(f"num_clusters must be >= 1, got {self.N}")
        if self.H < 1:
            raise ValueError(f"recluster_every must be >= 1, got {self.H}")
        if self.E < 1:
            raise ValueError(f"local_iters must be >= 1, got {self.E}")
        if clustering not in ("gradient", "random"):
            raise ValueError(f"clustering must be 'gradient' or 'random', got {clustering!r}")
        if aggregation not in ("weighted", "uniform"):
            raise ValueError(f"aggregation must be 'weighted' or 'uniform', got {aggregation!r}")

    def random_clusters(self, client_ids, rng: np.random.RandomState) -> list:
        """Random one-time clustering (ablation): shuffle devices, split into N
        near-equal contiguous groups. No gradient info used."""
        ids = list(client_ids)
        rng.shuffle(ids)
        N = min(self.N, len(ids))
        return [list(g) for g in np.array_split(ids, N) if len(g) > 0]

    # ------------------------------------------------------------------
    # Mechanism 1 — cosine-distance K-means on device-side gradients
    # ------------------------------------------------------------------

    def cluster_by_gradient(self, gradients: dict, rng: np.random.RandomState) -> list:
        """
        Cosine-distance K-means over device-side gradients.

        Args:
            gradients: {client_id -> 1-D np.ndarray} device-side gradient g_k.
            rng:       seeded RNG for reproducible centroid initialization.

        Returns:
            list[list[client_id]] — N non-empty clusters (empties are refilled).
        """
        cids = list(gradients.keys())
        G = np.stack([np.asarray(gradients[c], dtype=np.float64) for c in cids])
        norms = np.linalg.norm(G, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        U = G / norms                                   # unit vectors; cos(a,b) = <a_hat, b_hat>
        N = min(self.N, len(cids))

        centroids = self._init_centroids(U, N, rng)     # (N, d), unit rows
        labels = np.full(len(cids), -1, dtype=int)
        for it in range(self.kmeans_iters):
            sims = U @ centroids.T                       # cosine similarity to each centroid
            new_labels = sims.argmax(axis=1)             # assign = max cos = min (1-cos)
            if it > 0 and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for n in range(N):
                members = U[labels == n]
                if len(members) == 0:
                    # re-seed an empty centroid to the current worst-fit point
                    centroids[n] = U[sims.max(axis=1).argmin()]
                else:
                    m = members.mean(axis=0)
                    mn = np.linalg.norm(m)
                    centroids[n] = m / mn if mn > 0 else m

        clusters = [[] for _ in range(N)]
        for i, c in enumerate(cids):
            clusters[int(labels[i])].append(c)
        return self._fix_empty(clusters)

    def _init_centroids(self, U: np.ndarray, N: int, rng) -> np.ndarray:
        """k-means++-style seeding on the unit sphere (cosine distance = 1-cos)."""
        chosen = [int(rng.randint(len(U)))]
        for _ in range(1, N):
            sims = (U @ U[chosen].T).max(axis=1)         # max sim to any chosen centroid
            dist = np.clip(1.0 - sims, 0.0, None)         # cosine distance
            s = dist.sum()
            probs = dist / s if s > 0 else np.full(len(U), 1.0 / len(U))
            chosen.append(int(rng.choice(len(U), p=probs)))
        return U[chosen].copy()

    @staticmethod
    def _fix_empty(clusters: list) -> list:
        """Guarantee no empty cluster: move a member from the largest cluster."""
        clusters = [list(c) for c in clusters]
        while any(len(c) == 0 for c in clusters):
            biggest = max(clusters, key=len)
            if len(biggest) <= 1:
                break
            empty = next(c for c in clusters if len(c) == 0)
            empty.append(biggest.pop())
        return [c for c in clusters if len(c) > 0]

    # ------------------------------------------------------------------
    # Mechanism 2 — data-size-aware AND staleness-aware aggregation weights
    # ------------------------------------------------------------------

    def agg_weights(self, data_sizes, stalenesses) -> np.ndarray:
        """
        phi_{n,t} = |D_n| * 1/(1+tau_n) / sum_n' |D_n'| * 1/(1+tau_n')  (paper Eq. phi).
        Returns a length-N array summing to 1. The "uniform" ablation returns 1/N
        (drops both the data-size and the staleness weighting).
        """
        n = len(data_sizes)
        if self.aggregation == "uniform":
            return np.full(n, 1.0 / max(1, n))
        d = np.asarray(data_sizes, dtype=float)
        tau = np.asarray(stalenesses, dtype=float)
        raw = d / (1.0 + tau)
        s = raw.sum()
        return raw / s if s > 0 else np.full(n, 1.0 / max(1, n))
