"""
data/loaders/cifar100.py: CIFAR-100 dataset loader.

Returns the standard torchvision CIFAR-100 (fine-label, 100-class) train and
test datasets with per-channel normalisation. Data is downloaded to
~/.flsim/data/ on first use. Native resolution is already 32x32x3 — identical
to CIFAR-10 — so it drops into every existing 3-channel model in this
framework unchanged; only the classifier's num_classes needs to be 100 (see
flsim.models.factory.create_model(..., num_classes=100), already wired
automatically via flsim.experiments.wiring._num_classes_for_dataset).
"""

import os
import torchvision
import torchvision.transforms as T

# CIFAR-100 per-channel statistics (computed over training set)
_CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
_CIFAR100_STD  = (0.2673, 0.2564, 0.2762)

_DEFAULT_DATA_ROOT = os.path.expanduser("~/.flsim/data")


def load_cifar100(data_root: str = _DEFAULT_DATA_ROOT):
    """
    Load CIFAR-100 (fine labels, 100 classes) train and test datasets with
    standard normalisation.

    Args:
        data_root (str): directory for storing downloaded data.

    Returns:
        tuple[torchvision.datasets.CIFAR100, torchvision.datasets.CIFAR100]:
            (train_dataset, test_dataset). Both expose .targets (fine label,
            0-99), consumed directly by flsim.data.{shard,dirichlet}.
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
    ])

    train = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform
    )
    test = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=transform
    )
    return train, test
