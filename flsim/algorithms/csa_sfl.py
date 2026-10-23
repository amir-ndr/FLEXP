"""
algorithms/csa_sfl.py: CSA-SFL (Clustered Semi-Asynchronous Split Federated
Learning) parameters + the two PS-side mechanisms.

This class is a lightweight strategy/params holder. The event-driven
semi-asynchronous orchestration (per-cluster server-side submodels, buffered +
staleness-weighted global aggregation, cluster completion scheduling, dynamic
re-clustering) lives in flsim.core.csa_sfl_simulator.CSASFLSimulator (the
channel-cost experiment, examples/parallelsfl_experiment.py) and in
flsim.csasfl.methods.run_csasfl (the paper's MNIST setup, examples/CSASFL/).

The two mechanisms it owns, both run on the PS:
  1. cluster_by_gradient — dynamic device clustering by GRADIENT SIMILARITY:
     cosine-distance K-means on the device-side gradients g_k (paper's
     "min_{C} sum_n sum_{k in C_n} (1 - cos(g_k, mu_n))", mu_n = mean of the
     cluster's g_k). Coherent-gradient devices are grouped, reducing
     intra-cluster update inconsistency.
  2. agg_weights — data-size-aware AND staleness-aware inter-cluster weights
     phi_{n,t} = |D_n|/(1+tau_n) / sum_n' |D_n'|/(1+tau_n')  (paper Eq. phi),
     giving fresher (small tau) and larger (big |D_n|) clusters more influence.

Arguments (N, H, E, T) — as requested:
    N (num_clusters)     — number of clusters the K devices are partitioned into.
    H (recluster_every)  — re-collect gradients and re-cluster every H GLOBAL
                           aggregation rounds (paper's "every H rounds").
    E (local_iters)      — intra-cluster local work per cluster training pass.
                           CSASFLSimulator reads it as E mini-batch ITERATIONS
                           (so its local work equals the other baselines' H
                           steps in parallelsfl_experiment.py); the paper's
                           setup (flsim.csasfl) reads it as E local EPOCHS.
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
        staleness_exp: float = 1.0,
    ):
        self.N = int(num_clusters)          # N — number of clusters
        self.H = int(recluster_every)       # H — re-cluster every H global rounds
        self.E = int(local_iters)           # E — intra-cluster local mini-batch iterations
        self.T = global_rounds              # T — number of global rounds (None -> read config)
        self.kmeans_iters = int(kmeans_iters)
        # Ablation switches (Exp 3): the full method is gradient + weighted.
        #   clustering  = "gradient" (cosine K-means, re-clustered every H) |
        #                 "random"   (random ONE-TIME clustering, no re-clustering) |
        #                 "system"   (ONE-TIME K-means on the devices' round times only, no gradients)
        #   aggregation = "weighted" (phi = |D_n|/(1+tau_n), normalized)     |
        #                 "uniform"  (phi = 1/N — drops data-size AND staleness)
        self.clustering = clustering
        self.aggregation = aggregation
        self.staleness_exp = float(staleness_exp)   # a in phi = |D_n| / (1+tau_n)^a  (paper: a = 1)
        if self.N < 1:
            raise ValueError(f"num_clusters must be >= 1, got {self.N}")
        if self.H < 1:
            raise ValueError(f"recluster_every must be >= 1, got {self.H}")
        if self.E < 1:
            raise ValueError(f"local_iters must be >= 1, got {self.E}")
        if clustering not in ("gradient", "random", "system"):
            raise ValueError(f"clustering must be 'gradient', 'random' or 'system', got {clustering!r}")
        if aggregation not in ("weighted", "uniform"):
            raise ValueError(f"aggregation must be 'weighted' or 'uniform', got {aggregation!r}")

    def system_clusters(self, system: dict, rng: np.random.RandomState) -> list:
        """System-only clustering (ablation of the system-aware variant without its gradient term):
        K-means on the scalar system feature z_k alone (the same k-means++ / Lloyd procedure as
        cluster_system_aware, with the gradient term constant), i.e. speed tiers."""
        return self.cluster_system_aware({c: np.ones(1) for c in system}, system, 1.0, rng)

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

    @staticmethod
    def _labels_from(init, cids) -> np.ndarray:
        """Assignment of cids from a partition (list of member lists); every cid must be in one."""
        of = {c: n for n, members in enumerate(init) for c in members}
        return np.array([of[c] for c in cids])

    def cluster_by_gradient(self, gradients: dict, rng: np.random.RandomState, init=None) -> list:
        """
        Cosine-distance K-means over device-side gradients (paper Eq. clustering):
            min_{C_1..C_N} sum_n sum_{k in C_n} (1 - cos(g_k, mu_n)),   mu_n = (1/|C_n|) sum_{k in C_n} g_k,
        i.e. the centroid is the MEAN OF THE RAW GRADIENTS (not of their unit vectors). Lloyd
        iterations: assign every device to the centroid of largest cosine, recompute mu_n as
        defined; since this centroid is not the exact minimizer of the cosine objective, the
        partition with the lowest objective value seen is returned.

        Args:
            gradients: {client_id -> 1-D np.ndarray} device-side gradient g_k.
            rng:       seeded RNG for reproducible centroid initialization.
            init:      optional current partition (N member lists): warm start, Lloyd iterations
                       begin from it instead of k-means++ seeding (rng unused).

        Returns:
            list[list[client_id]] — N non-empty clusters.
        """
        cids = list(gradients.keys())
        G = np.stack([np.asarray(gradients[c], dtype=np.float64) for c in cids])
        norms = np.linalg.norm(G, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        U = G / norms                                   # unit vectors; cos(a,b) = <a_hat, b_hat>
        N = min(self.N, len(cids))

        if init is not None:
            labels, C = self._labels_from(init, cids), None
            N = len(init)
        else:
            C = self._init_centroids(U, N, rng)         # (N, d) unit centroid directions
            labels = (U @ C.T).argmax(axis=1)
        best_obj, best = np.inf, None
        for _ in range(self.kmeans_iters):
            labels = self._fill_empty(labels, U, C, N)
            mu = np.stack([G[labels == n].mean(axis=0) for n in range(N)])     # mu_n (paper)
            mn = np.linalg.norm(mu, axis=1, keepdims=True)
            C = mu / np.where(mn > 0, mn, 1.0)
            obj = float(np.sum(1.0 - np.einsum("ij,ij->i", U, C[labels])))     # sum_k (1 - cos(g_k, mu_{n(k)}))
            if obj < best_obj - 1e-12:
                best_obj, best = obj, labels.copy()
            new_labels = (U @ C.T).argmax(axis=1)       # assign = max cos = min (1 - cos)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
        self.last_objective = best_obj

        clusters = [[] for _ in range(N)]
        for i, c in enumerate(cids):
            clusters[int(best[i])].append(c)
        return [c for c in clusters if c]

    def cluster_system_aware(self, gradients: dict, system: dict, lam: float,
                             rng: np.random.RandomState, init=None) -> list:
        """
        Gradient + system-behaviour K-means (improvement 1; lam = 0 is the paper's clustering,
        use cluster_by_gradient for it). Each device k carries its gradient g_k and a scalar
        system feature z_k (the caller passes the standardized log of its local-round time), and
            min sum_n sum_{k in C_n} [ (1 - cos(g_k, mu_n)) + lam * (z_k - c_n)^2 ],
            mu_n = mean of the cluster's g_k (paper),  c_n = mean of the cluster's z_k,
        so devices are grouped by update direction AND by how long their local round takes,
        which bounds the intra-cluster straggler wait. Lloyd iterations with k-means++ seeding
        on the same cost (or, with init = the current partition, from that partition: warm
        start); the partition with the lowest objective seen is returned.
        """
        cids = list(gradients.keys())
        G = np.stack([np.asarray(gradients[c], dtype=np.float64) for c in cids])
        norms = np.linalg.norm(G, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        U = G / norms
        z = np.array([float(system[c]) for c in cids])
        N = min(self.N, len(cids))
        lam = float(lam)

        def cost(C, c):                                          # (K, N) assignment cost
            return (1.0 - U @ C.T) + lam * (z[:, None] - c[None, :]) ** 2

        if init is not None:                                     # warm start from the current partition
            labels, N = self._labels_from(init, cids), len(init)
        else:
            # k-means++ seeding on the joint point-to-point cost
            chosen = [int(rng.randint(len(U)))]
            for _ in range(1, N):
                dist = np.min(np.stack([(1.0 - U @ U[j]) + lam * (z - z[j]) ** 2 for j in chosen]), axis=0)
                dist = np.clip(dist, 0.0, None)
                s = dist.sum()
                chosen.append(int(rng.choice(len(U), p=dist / s if s > 0 else np.full(len(U), 1.0 / len(U)))))
            C, c = U[chosen].copy(), z[chosen].copy()
            labels = cost(C, c).argmin(axis=1)
        best_obj, best = np.inf, None
        for _ in range(self.kmeans_iters):
            for n in range(N):                                   # refill an empty cluster with the worst-fit device
                if np.any(labels == n):
                    continue
                own = cost(C, c)[np.arange(len(U)), labels]
                sizes = np.bincount(labels, minlength=N)
                for i in np.argsort(-own):
                    if sizes[labels[i]] > 1:
                        labels[i] = n
                        break
            mu = np.stack([G[labels == n].mean(axis=0) for n in range(N)])
            mn = np.linalg.norm(mu, axis=1, keepdims=True)
            C = mu / np.where(mn > 0, mn, 1.0)
            c = np.array([z[labels == n].mean() for n in range(N)])
            obj = float(cost(C, c)[np.arange(len(U)), labels].sum())
            if obj < best_obj - 1e-12:
                best_obj, best = obj, labels.copy()
            new_labels = cost(C, c).argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
        self.last_objective = best_obj
        clusters = [[] for _ in range(N)]
        for i, cid in enumerate(cids):
            clusters[int(best[i])].append(cid)
        return [cl for cl in clusters if cl]

    @staticmethod
    def _fill_empty(labels: np.ndarray, U: np.ndarray, C: np.ndarray, N: int) -> np.ndarray:
        """Give every empty cluster the worst-fitting device of a cluster with > 1 member."""
        labels = labels.copy()
        for n in range(N):
            if np.any(labels == n):
                continue
            fit = np.einsum("ij,ij->i", U, C[labels])               # cos to its own centroid
            sizes = np.bincount(labels, minlength=N)
            for i in np.argsort(fit):
                if sizes[labels[i]] > 1:
                    labels[i] = n
                    break
        return labels

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

    # ------------------------------------------------------------------
    # Mechanism 2 — data-size-aware AND staleness-aware aggregation weights
    # ------------------------------------------------------------------

    def agg_weights(self, data_sizes, stalenesses) -> np.ndarray:
        """
        phi_{n,t} = |D_n| * 1/(1+tau_n) / sum_n' |D_n'| * 1/(1+tau_n')  (paper Eq. phi).
        With staleness_exp a != 1: |D_n| / (1+tau_n)^a (a < 1: a gentler staleness discount).
        Returns a length-N array summing to 1. The "uniform" ablation returns 1/N
        (drops both the data-size and the staleness weighting).
        """
        n = len(data_sizes)
        if self.aggregation == "uniform":
            return np.full(n, 1.0 / max(1, n))
        d = np.asarray(data_sizes, dtype=float)
        tau = np.asarray(stalenesses, dtype=float)
        raw = d / (1.0 + tau) if self.staleness_exp == 1.0 else d / (1.0 + tau) ** self.staleness_exp
        s = raw.sum()
        return raw / s if s > 0 else np.full(n, 1.0 / max(1, n))
