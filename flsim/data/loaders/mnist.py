"""
data/loaders/mnist.py: MNIST dataset loader.

Returns the standard torchvision MNIST train and test datasets with
appropriate normalisation (zero mean, unit std for the MNIST pixel range).
Data is downloaded to ~/.flsim/data/ on first use.
"""

import os
import torchvision
import torchvision.transforms as T

# MNIST pixel statistics (computed over training set)
_MNIST_MEAN = (0.1307,)
_MNIST_STD  = (0.3081,)

_DEFAULT_DATA_ROOT = os.path.expanduser("~/.flsim/data")


def load_mnist(data_root: str = _DEFAULT_DATA_ROOT):
    """
    Load MNIST train and test datasets with standard normalisation.

    Args:
        data_root (str): directory for storing downloaded data.

    Returns:
        tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
            (train_dataset, test_dataset)
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(_MNIST_MEAN, _MNIST_STD),
    ])

    train = torchvision.datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    return train, test
