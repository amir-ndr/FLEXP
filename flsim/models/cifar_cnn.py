"""
models/cifar_cnn.py: Small CNN for CIFAR-10 classification.

Architecture:
  Conv(3→32, 3×3, pad=1) → BN → ReLU → MaxPool(2)
  Conv(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2)
  Conv(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2)
  Flatten → Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→10)

Input: (B, 3, 32, 32) — 3-channel 32×32 CIFAR-10 images
Output: (B, 10) — logits for 10 object classes

Splittable (flsim.interfaces.splittable): ordered_layers() exposes the 17
layers below as a flat list (indices 0-11 = features, 12-16 = classifier) so
flsim.system.split_model.split_model() can cut the network for split
learning (SL/SFLV1/SFLV2) at any index in [1, 16].
"""

import torch.nn as nn

from flsim.interfaces.splittable import Splittable


class CifarCNN(nn.Module, Splittable):
    """
    Small 3-layer CNN for CIFAR-10 with Batch Normalisation.

    This class does NOT:
    - Manage training loops.
    - Load data or apply normalisation.
    - Hold optimizer state.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (B,3,32,32) → (B,32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → (B,32,16,16)
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (B,64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → (B,64,8,8)
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # → (B,128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → (B,128,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                  # → (B, 2048)
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape (B, 3, 32, 32).

        Returns:
            tensor of shape (B, 10) — unnormalized logits.
        """
        return self.classifier(self.features(x))

    def ordered_layers(self) -> list:
        """17 layers in forward order: features[0:12] then classifier[0:5]."""
        return list(self.features) + list(self.classifier)
