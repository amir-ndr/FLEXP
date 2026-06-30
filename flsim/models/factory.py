"""
models/factory.py: Model factory — maps config string names to nn.Module instances.

Adding a new model: register it in create_model() with a unique string key.
No other file needs to change.
"""

import torch.nn as nn

from flsim.models.mnist_cnn import MnistCNN
from flsim.models.cifar_cnn import CifarCNN

# Registry of all available models: name → constructor
_MODEL_REGISTRY = {
    "mnist_cnn":  MnistCNN,
    "cifar_cnn":  CifarCNN,
}


def create_model(name: str) -> nn.Module:
    """
    Instantiate a model by name.

    Args:
        name (str): model name as used in config (e.g. "mnist_cnn", "cifar_cnn").

    Returns:
        nn.Module: freshly constructed model with randomly initialised weights.

    Raises:
        ValueError: if name is not in the model registry.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name]()


def list_models() -> list:
    """Return sorted list of registered model names."""
    return sorted(_MODEL_REGISTRY.keys())
