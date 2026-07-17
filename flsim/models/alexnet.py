"""
models/alexnet.py: AlexNet (Krizhevsky et al., 2012), adapted for CIFAR-10.

The original AlexNet is sized for 224x224 ImageNet input (11x11 stride-4 stem,
5 conv layers, three 4096-wide FC layers). Applied verbatim to a 32x32
CIFAR-10 image, that stem alone collapses the spatial dimension to <=1 before
the later conv layers ever run. This is the standard CIFAR-10 adaptation used
throughout the FL/vision literature: the same 5 conv + 3 maxpool + 3 FC
structure as the original, with kernel sizes/strides shrunk to fit CIFAR-10's
much smaller input instead of assuming 224x224.

Architecture:
  Conv(3->64, 3x3, pad=1)    -> ReLU -> MaxPool(2)            # 32 -> 16
  Conv(64->192, 3x3, pad=1)  -> ReLU -> MaxPool(2)             # 16 -> 8
  Conv(192->384, 3x3, pad=1) -> ReLU
  Conv(384->256, 3x3, pad=1) -> ReLU
  Conv(256->256, 3x3, pad=1) -> ReLU -> MaxPool(2)             # 8 -> 4
  Flatten -> Dropout(0.5) -> Linear(4096->4096) -> ReLU
          -> Dropout(0.5) -> Linear(4096->4096) -> ReLU
          -> Linear(4096->10)

Input: (B, 3, 32, 32)
Output: (B, 10)

Splittable (flsim.interfaces.splittable): ordered_layers() exposes the 21
layers below as a flat list (indices 0-12 = features, 13-20 = classifier), so
flsim.system.split_model.split_model() can cut the network at any index in
[1, 20].
"""

import torch.nn as nn

from flsim.interfaces.splittable import Splittable


class AlexNet(nn.Module, Splittable):
    """
    AlexNet, CIFAR-10-adapted (see module docstring).

    This class does NOT:
    - Manage training loops.
    - Load data or apply normalisation.
    - Hold optimizer state.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),     # (B,3,32,32) -> (B,64,32,32)
            nn.ReLU(),
            nn.MaxPool2d(2),                                 # -> (B,64,16,16)
            nn.Conv2d(64, 192, kernel_size=3, padding=1),   # -> (B,192,16,16)
            nn.ReLU(),
            nn.MaxPool2d(2),                                 # -> (B,192,8,8)
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # -> (B,384,8,8)
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # -> (B,256,8,8)
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # -> (B,256,8,8)
            nn.ReLU(),
            nn.MaxPool2d(2),                                 # -> (B,256,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                    # -> (B, 4096)
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape (B, 3, 32, 32).

        Returns:
            tensor of shape (B, num_classes) — unnormalized logits.
        """
        return self.classifier(self.features(x))

    def ordered_layers(self) -> list:
        """21 layers in forward order: features[0:13] then classifier[0:8]."""
        return list(self.features) + list(self.classifier)
