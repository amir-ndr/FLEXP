"""
data/iid.py: IID (independent and identically distributed) data partitioner.

Shuffles all dataset indices uniformly at random, then splits into K equal
(or near-equal) chunks. Each client gets approximately the same number of
samples with the same class distribution as the full dataset.

This is the homogeneous baseline — research typically shows FL algorithms
perform best under IID data and degrades as heterogeneity increases.
"""

import numpy as np
from flsim.interfaces.partitioner import DataPartitioner


class IIDPartitioner(DataPartitioner):
    """
    IID partition: shuffle all indices, split equally into K chunks.

    Each client receives len(dataset) // K samples (remainder distributed
    to the first clients so all samples are assigned exactly once).

    This class does NOT:
    - Load or modify the dataset.
    - Guarantee any class distribution per client (though it is close to uniform
      due to the global shuffle).
    """

    def partition(self, dataset, num_clients: int, rng: np.random.RandomState) -> list:
        """
        Randomly shuffle all indices and split into num_clients equal chunks.

        Args:
            dataset: dataset with len() defined.
            num_clients (int): K — number of clients.
            rng (np.random.RandomState): seeded RNG.

        Returns:
            list[list[int]]: client_indices[k] = list of indices for client k.
                             Every index 0..len(dataset)-1 appears exactly once.
        """
        n = len(dataset)
        indices = np.arange(n)
        rng.shuffle(indices)

        # np.array_split handles uneven splits; first (n % K) clients get one extra
        splits = np.array_split(indices, num_clients)
        return [split.tolist() for split in splits]

    def describe(self) -> str:
        return "IIDPartitioner()"
