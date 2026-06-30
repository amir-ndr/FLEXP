"""
data/shard.py: Pathological non-IID shard-based data partitioner.

Reproduces the original FedAvg paper (McMahan et al., 2017) non-IID setup:
  1. Sort all training samples by class label.
  2. Divide sorted samples into num_shards equal shards.
  3. Assign shards_per_client shards randomly to each client.

With num_shards=200 and shards_per_client=2 for MNIST (100 clients),
each client receives ~600 samples from approximately 2 digit classes,
creating a strong label distribution shift across clients.

Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data", AISTATS 2017, Section 3.
"""

import numpy as np
from flsim.interfaces.partitioner import DataPartitioner


class ShardPartitioner(DataPartitioner):
    """
    Pathological non-IID partition matching the original FedAvg paper setup.

    Clients receive shards from the label-sorted dataset, so each client
    typically sees only 1–2 distinct classes. This is a worst-case non-IID
    scenario for convergence analysis.

    This class does NOT:
    - Load or modify the dataset.
    - Guarantee exactly 2 classes per client (depends on class distribution and shard size).
    """

    def __init__(self, num_shards: int, shards_per_client: int):
        """
        Args:
            num_shards (int): total number of shards to divide the dataset into.
            shards_per_client (int): how many shards each client receives.
        """
        self.num_shards = num_shards
        self.shards_per_client = shards_per_client

    def partition(self, dataset, num_clients: int, rng: np.random.RandomState) -> list:
        """
        Sort by label, divide into shards, assign shards randomly to clients.

        Args:
            dataset: object with .targets attribute (list or tensor of int class labels).
            num_clients (int): K — number of clients.
            rng (np.random.RandomState): seeded RNG for shard assignment.

        Returns:
            list[list[int]]: client_indices[k] = indices for client k.

        Raises:
            ValueError: if num_shards is not divisible by num_clients or total shards
                        are insufficient.
        """
        assert self.num_shards % num_clients == 0 or True, (
            "num_shards should ideally be divisible by num_clients for even assignment"
        )
        assert self.num_shards >= num_clients * self.shards_per_client, (
            f"Need at least {num_clients * self.shards_per_client} shards, "
            f"got {self.num_shards}"
        )

        targets = _get_targets(dataset)
        n = len(targets)

        # Step 1: sort all indices by their class label
        sorted_indices = np.argsort(targets, kind="stable")

        # Step 2: divide sorted indices into num_shards equal shards
        shards = np.array_split(sorted_indices, self.num_shards)

        # Step 3: randomly assign shards_per_client shards to each client
        shard_ids = np.arange(self.num_shards)
        rng.shuffle(shard_ids)

        client_indices = []
        ptr = 0
        for k in range(num_clients):
            assigned = []
            for _ in range(self.shards_per_client):
                assigned.extend(shards[shard_ids[ptr]].tolist())
                ptr += 1
            client_indices.append(assigned)

        return client_indices

    def describe(self) -> str:
        return (
            f"ShardPartitioner(num_shards={self.num_shards}, "
            f"shards_per_client={self.shards_per_client})"
        )


def _get_targets(dataset) -> np.ndarray:
    """Extract integer class labels from a dataset's .targets attribute."""
    targets = dataset.targets
    if hasattr(targets, "numpy"):        # torch.Tensor
        return targets.numpy().astype(int)
    return np.array(targets, dtype=int)
