"""
models/mnist_cnn.py: Lightweight CNN for MNIST classification.

Architecture matches the standard FL benchmark model from McMahan et al. (2017):
  Conv(1→32, 5×5) → ReLU → MaxPool(2)
  Conv(32→64, 5×5) → ReLU → MaxPool(2)
  Flatten → Linear(1024→512) → ReLU → Linear(512→10)

Input: (B, 1, 28, 28) — single-channel 28×28 MNIST images
Output: (B, 10) — logits for 10 digit classes

Splittable (flsim.interfaces.splittable): ordered_layers() exposes the 10
layers below as a flat list (indices 0-5 = features, 6-9 = classifier) so
flsim.system.split_model.split_model() can cut the network for split
learning (SL/SFLV1/SFLV2) at any index in [1, 9]. See that module for the
cut_layer validity rule.
"""

import torch.nn as nn

from flsim.interfaces.splittable import Splittable


class MnistCNN(nn.Module, Splittable):
    """
    Lightweight 2-layer CNN for MNIST.

    This class does NOT:
    - Manage training loops.
    - Load data or apply normalisation.
    - Hold optimizer state.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),    # (B,1,28,28) → (B,32,24,24)
            nn.ReLU(),
            nn.MaxPool2d(2),                     # → (B,32,12,12)
            nn.Conv2d(32, 64, kernel_size=5),    # → (B,64,8,8)
            nn.ReLU(),
            nn.MaxPool2d(2),                     # → (B,64,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # → (B, 1024)
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape (B, 1, 28, 28).

        Returns:
            tensor of shape (B, 10) — unnormalized logits.
        """
        return self.classifier(self.features(x))

    def ordered_layers(self) -> list:
        """10 layers in forward order: features[0:6] then classifier[0:4]."""
        return list(self.features) + list(self.classifier)
