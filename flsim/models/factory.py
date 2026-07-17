"""
models/factory.py: Model factory — maps config string names to nn.Module instances.

Adding a new model: register it in create_model() with a unique string key.
No other file needs to change.
"""

import torch.nn as nn

from flsim.models.mnist_cnn import MnistCNN
from flsim.models.cifar_cnn import CifarCNN
from flsim.models.alexnet import AlexNet
from flsim.models.vgg import VGG16, VGG19
from flsim.models.resnet import ResNet18, ResNet34

# Registry of all available models: name → constructor
_MODEL_REGISTRY = {
    "mnist_cnn":  MnistCNN,
    "cifar_cnn":  CifarCNN,
    "alexnet":    AlexNet,
    "vgg16":      VGG16,
    "vgg19":      VGG19,
    "resnet18":   ResNet18,
    "resnet34":   ResNet34,
}


def create_model(name: str, num_classes: int = None) -> nn.Module:
    """
    Instantiate a model by name.

    Args:
        name (str): model name as used in config (e.g. "mnist_cnn", "resnet18").
        num_classes (int, optional): output classes. Every registered model
            accepts this (default 10). Pass the dataset's actual class count
            (e.g. 100 for cifar100, 7 for ham10000) — see
            flsim.experiments.wiring._num_classes_for_dataset(), which is
            wired into every experiment base class automatically. None uses
            each model's own default (10).

    Returns:
        nn.Module: freshly constructed model with randomly initialised weights.

    Raises:
        ValueError: if name is not in the model registry.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(_MODEL_REGISTRY.keys())}"
        )
    cls = _MODEL_REGISTRY[name]
    if num_classes is None:
        return cls()
    return cls(num_classes=num_classes)


def list_models() -> list:
    """Return sorted list of registered model names."""
    return sorted(_MODEL_REGISTRY.keys())
