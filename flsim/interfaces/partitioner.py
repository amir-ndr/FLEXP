"""
interfaces/partitioner.py: Abstract base class for data partitioning strategies.

Data heterogeneity is a core research variable in FL. Adding a new partitioning
strategy (pathological non-IID, label-quantity skew, feature skew, …) only
requires subclassing DataPartitioner — no changes to any other module.
"""

from abc import ABC, abstractmethod


class DataPartitioner(ABC):
    """
    Base class for all data partitioning strategies.

    Takes a dataset and splits it into per-client index lists.
    Adding a new strategy = subclass this, implement partition(), done.

    This class does NOT:
    - Load datasets (that is handled by data/loaders/).
    - Hold references to the dataset after partition() returns.
    - Perform any model training.
    """

    @abstractmethod
    def partition(self, dataset, num_clients: int, rng) -> list:
        """
        Partition a dataset into per-client index lists.

        Args:
            dataset: torchvision dataset or any object with a .targets attribute
                     (list or tensor of integer class labels, length == len(dataset)).
            num_clients (int): number of federated clients K.
            rng: numpy RandomState for reproducibility.

        Returns:
            list[list[int]]: client_indices where client_indices[k] is the list
                             of dataset indices assigned to client k.
                             Union of all lists must equal {0, …, len(dataset)-1}.
        """

    @abstractmethod
    def describe(self) -> str:
        """
        Returns a human-readable string describing partition settings.

        Used in logging and saved configs so experiments are self-documenting.

        Returns:
            str: description, e.g. "DirichletPartitioner(alpha=0.5)".
        """
