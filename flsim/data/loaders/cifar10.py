"""
data/loaders/cifar10.py: CIFAR-10 dataset loader.

Returns the standard torchvision CIFAR-10 train and test datasets with
per-channel normalisation (ImageNet-style mean/std computed on CIFAR-10).
Data is downloaded to ~/.flsim/data/ on first use.
"""

import os
import torchvision
import torchvision.transforms as T

# CIFAR-10 per-channel statistics (computed over training set)
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

_DEFAULT_DATA_ROOT = os.path.expanduser("~/.flsim/data")


def load_cifar10(data_root: str = _DEFAULT_DATA_ROOT):
    """
    Load CIFAR-10 train and test datasets with standard normalisation.

    Args:
        data_root (str): directory for storing downloaded data.

    Returns:
        tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
            (train_dataset, test_dataset)
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])

    train = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    test = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )
    return train, test
