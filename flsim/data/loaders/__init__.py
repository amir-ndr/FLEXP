"""data/loaders/: Dataset download and loading utilities."""
from flsim.data.loaders.mnist import load_mnist
from flsim.data.loaders.cifar10 import load_cifar10

__all__ = ["load_mnist", "load_cifar10"]
