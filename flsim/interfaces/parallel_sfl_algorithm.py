"""
interfaces/parallel_sfl_algorithm.py: Base class for ParallelSFL-style
cluster-based split federated learning algorithms.

Based on Liao, Xu, Xu, Yao, Huang & Qiao, "ParallelSFL: A Novel Split
Federated Learning Framework Tackling Heterogeneity Issues" (ACM MobiCom '24).

What makes ParallelSFL different from SplitFed (SL/SFLV1/SFLV2)
--------------------------------------------------------------
In SplitFed the PS/edge server holds the top submodel for EVERY worker. In
ParallelSFL the workers are partitioned into CLUSTERS, and within each cluster
ONE worker (the "top worker") holds the top submodel while the others ("bottom
workers") hold bottom submodels and exchange smashed data / gradients with the
top worker (not the PS). So each cluster runs an SFLV1-style split-training loop
internally, then the top worker aggregates the bottom submodels and sends the
cluster's (bottom, top) pair to the PS, which aggregates across clusters. This
avoids the PS communication bottleneck (only full models cross the PS link,
once per round per cluster) and lets the framework tackle system + statistical
heterogeneity through the choice of clusters and per-cluster local frequencies.

This class is the SINGLE extension point for that choice. Following the
framework's "override only what changes" pattern (see the README), it exposes
four independent hooks — the two the paper contributes (clustering, local
updating frequency) plus the two aggregation rules — each with a working
default so a custom algorithm overrides only the piece it changes:

  cluster_workers(workers, rng) -> list[Cluster]
      Partition workers into clusters (each = 1 top worker + N_c bottom
      workers). Default: capability-balanced greedy grouping. The paper's
      data-and-system-aware greedy Algorithm 1 is the override in
      flsim.algorithms.parallel_sfl.ParallelSFL.

  assign_frequencies(clusters) -> None   (mutates cluster.tau in place)
      Per-cluster local updating frequency tau_c (number of local iterations).
      Default: uniform default_tau for every cluster. The paper's Eq. (17)
      waiting-time-alignment rule is ParallelSFL's override.

  aggregate_bottom(bottom_state_dicts, num_samples_list) -> OrderedDict
      Intra-cluster bottom-submodel aggregation on the top worker (paper
      Eq. 4). Default: uniform mean (Eq. 4 is a plain 1/N_c average).

  global_weights(clusters) -> list[float]
      Per-cluster weights rho_c for the PS's cross-cluster aggregation
      (paper Eq. 18). Default: N_c * tau_c, normalized by the simulator.

The models, the intra-cluster relay training (Eq. 2/3), and the physical
timing/energy live in flsim.core.parallel_sfl_simulator.ParallelSFLSimulator —
this class only decides WHO is grouped with whom, HOW MANY local iterations
each cluster runs, and HOW submodels combine. The per-worker quantities the
clustering/frequency hooks need (label distribution, compute/comm times,
ingress bandwidth) are measured by the simulator and handed in as WorkerInfo,
so the algorithm is a pure function of those numbers (easy to unit-test).
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from flsim.core.split_simulator import _weighted_average_state_dicts


@dataclass
class WorkerInfo:
    """
    Everything the clustering / frequency hooks need to know about one worker,
    measured by the simulator once per round (label distribution + system
    capabilities). All times are per ONE local iteration (one mini-batch).

    Fields:
        client_id (int):        the worker's id (indexes into the client list).
        label_dist (np.ndarray): V_i — normalized label histogram over M classes
                                 (paper Sec 4.2). Sums to 1.
        num_samples (int):      |D_i| — local dataset size (aggregation weight).
        mu_b (float):           mu_b,i — bottom-submodel compute time per iter (s).
        mu_p (float):           mu_p,i — top-submodel compute time per iter for
                                 ONE mini-batch, IF this worker were the top
                                 worker (s). In a cluster the top worker serves
                                 N_c batches, so its top time is N_c * mu_p.
        beta_smashed (float):   beta_i — smashed-data / gradient transmission time
                                 per iter from this worker (bottom) to its top
                                 worker (s). (Estimated from this worker's link
                                 rate; a true D2D rate would also depend on the
                                 top worker — see ParallelSFLSimulator.)
        beta_full (float):      transmission time for the FULL model from this
                                 worker (as top) to the PS per round (s) — the
                                 paper's beta_c (Eq. 15).
        ingress_bw_hz (float):  B_i — available ingress bandwidth if this worker
                                 is the top worker (paper Eq. 8 constraint).
    """
    client_id: int
    label_dist: np.ndarray
    num_samples: int
    mu_b: float = 0.0
    mu_p: float = 0.0
    beta_smashed: float = 0.0
    beta_full: float = 0.0
    ingress_bw_hz: float = 0.0


@dataclass
class Cluster:
    """
    One cluster: a designated top worker (holds the top submodel; its own local
    data is NOT used for training — paper Sec 4.3) and N_c bottom workers (train
    the bottom submodels). tau is the cluster's local updating frequency, set by
    assign_frequencies().
    """
    top: WorkerInfo
    bottoms: List[WorkerInfo]
    tau: int = 1

    @property
    def size(self) -> int:
        """N_c — number of bottom workers (the top worker doesn't train)."""
        return len(self.bottoms)

    @property
    def bottom_ids(self) -> list:
        return [w.client_id for w in self.bottoms]


class ParallelSFLAlgorithm:
    """
    Base class for ParallelSFL cluster-based split-FL algorithms. Provides the
    four override hooks with simple, working defaults; subclass and override
    only what differs (see flsim.algorithms.parallel_sfl.ParallelSFL for the
    paper-faithful strategies).

    Args:
        cluster_size (int): target number of bottom workers per cluster for the
            DEFAULT clustering (ignored by subclasses that override
            cluster_workers). Default 4.
        default_tau (int): local updating frequency used by the DEFAULT
            assign_frequencies (ignored by subclasses that override it). Default 1.
    """

    def __init__(self, cluster_size: int = 4, default_tau: int = 1):
        self.cluster_size = cluster_size
        self.default_tau = default_tau

    # ------------------------------------------------------------------
    # Hook 1: worker clustering (the paper's Algorithm 1 is the override)
    # ------------------------------------------------------------------

    def cluster_workers(self, workers: List[WorkerInfo], rng) -> List[Cluster]:
        """
        Partition workers into clusters (1 top + N_c bottoms each).

        Default: shuffle workers, chop into groups of (cluster_size + 1), and
        make the first of each group the top worker. Data/system heterogeneity
        is ignored here — that is exactly what ParallelSFL overrides.

        Args:
            workers: all WorkerInfo this round.
            rng: numpy RandomState for reproducibility.

        Returns:
            list[Cluster] — every worker appears in exactly one cluster (as top
            or bottom); clusters have >= 1 bottom worker.
        """
        idx = rng.permutation(len(workers))
        group = self.cluster_size + 1          # +1 for the top worker
        clusters = []
        for start in range(0, len(workers), group):
            members = [workers[i] for i in idx[start:start + group]]
            if len(members) < 2:
                # leftover single worker — attach as a bottom to the last cluster
                if clusters:
                    clusters[-1].bottoms.append(members[0])
                continue
            clusters.append(Cluster(top=members[0], bottoms=members[1:]))
        return clusters

    # ------------------------------------------------------------------
    # Hook 2: per-cluster local updating frequency (paper Eq. 17 override)
    # ------------------------------------------------------------------

    def assign_frequencies(self, clusters: List[Cluster]) -> None:
        """
        Set each cluster's tau (local updating frequency) IN PLACE.

        Default: uniform default_tau for every cluster. ParallelSFL overrides
        with Eq. (17)'s waiting-time-alignment rule (faster clusters get more
        local iterations).
        """
        for c in clusters:
            c.tau = self.default_tau

    # ------------------------------------------------------------------
    # Hook 3: intra-cluster bottom aggregation (paper Eq. 4)
    # ------------------------------------------------------------------

    def aggregate_bottom(self, bottom_state_dicts: list, num_samples_list: list) -> OrderedDict:
        """
        Aggregate a cluster's bottom submodels on the top worker (Eq. 4).

        Default: uniform mean (Eq. 4 is exactly w_b,c = (1/N_c) sum_i w_b,i).
        Override for a sample-weighted or otherwise custom bottom aggregation.
        """
        weights = [1.0] * len(bottom_state_dicts)
        return _weighted_average_state_dicts(bottom_state_dicts, weights)

    # ------------------------------------------------------------------
    # Hook 4: cross-cluster global weights (paper Eq. 18)
    # ------------------------------------------------------------------

    def global_weights(self, clusters: List[Cluster]) -> List[float]:
        """
        Un-normalized per-cluster weights rho_c for the PS's global aggregation
        (Eq. 18). The simulator normalizes them and applies the same weight to
        the cluster's bottom and top submodels (aggregating full models = agg-
        regating bottom and top separately, since the param sets are disjoint).

        Default (paper Eq. 18): rho_c = N_c * tau_c — clusters that trained more
        (more workers, more local iterations) count proportionally more.
        """
        return [c.size * c.tau for c in clusters]
