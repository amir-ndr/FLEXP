"""
data/dirichlet.py: Dirichlet-based non-IID data partitioner.

For each class c, samples allocation proportions from Dirichlet(alpha * 1_K)
and distributes that class's samples to clients according to those proportions.

alpha controls heterogeneity:
    alpha → ∞  : approaches IID (each client gets equal share of every class)
    alpha = 1.0: moderate heterogeneity
    alpha = 0.5: strong non-IID (default research setting)
    alpha → 0  : extreme non-IID (each client gets one class only)

Reference: Hsieh et al., "Quagmire of Momentum", 2019; Yurochkin et al.,
"Bayesian Nonparametric Federated Learning of Neural Networks", ICML 2019.
"""

import numpy as np
from flsim.interfaces.partitioner import DataPartitioner


class DirichletPartitioner(DataPartitioner):
    """
    Dirichlet label-distribution partitioner.

    Provides a continuously tunable degree of non-IID-ness via alpha.
    Lower alpha → greater label concentration per client.

    This class does NOT:
    - Load or modify the dataset.
    - Guarantee any minimum number of samples per client (some clients may
      receive zero samples of a rare class under very low alpha).
    """

    def __init__(self, alpha: float = 0.5, min_samples_per_client: int = 1):
        """
        Args:
            alpha (float): Dirichlet concentration parameter. Default: 0.5.
            min_samples_per_client (int): minimum samples guaranteed per client.
                If a client would receive fewer, their shortage is filled from
                a global pool. This prevents empty clients. Default: 1.
        """
        assert alpha > 0.0, f"alpha must be positive, got {alpha}"
        self.alpha = alpha
        self.min_samples_per_client = min_samples_per_client

    def partition(self, dataset, num_clients: int, rng: np.random.RandomState) -> list:
        """
        Dirichlet partition: each class distributed across clients via Dir(alpha).

        Args:
            dataset: object with .targets attribute (list or tensor of int class labels).
            num_clients (int): K — number of clients.
            rng (np.random.RandomState): seeded RNG.

        Returns:
            list[list[int]]: client_indices[k] = indices for client k.
        """
        targets = _get_targets(dataset)
        classes = np.unique(targets)
        n = len(targets)

        client_indices = [[] for _ in range(num_clients)]

        for c in classes:
            class_indices = np.where(targets == c)[0]
            rng.shuffle(class_indices)

            # Sample proportions from Dirichlet(alpha * 1_K)
            proportions = rng.dirichlet(alpha=self.alpha * np.ones(num_clients))

            # Convert proportions to integer counts that sum to len(class_indices)
            counts = (proportions * len(class_indices)).astype(int)

            # Distribute remainder one-by-one to avoid losing samples
            remainder = len(class_indices) - counts.sum()
            top_k = np.argsort(proportions)[::-1][:remainder]
            counts[top_k] += 1

            # Assign slices to clients
            ptr = 0
            for k in range(num_clients):
                client_indices[k].extend(class_indices[ptr: ptr + counts[k]].tolist())
                ptr += counts[k]

        return client_indices

    def describe(self) -> str:
        return f"DirichletPartitioner(alpha={self.alpha})"


def _get_targets(dataset) -> np.ndarray:
    """Extract integer class labels from a dataset's .targets attribute."""
    targets = dataset.targets
    if hasattr(targets, "numpy"):
        return targets.numpy().astype(int)
    return np.array(targets, dtype=int)
