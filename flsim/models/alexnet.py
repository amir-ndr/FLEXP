"""
models/alexnet.py: AlexNet (Krizhevsky et al., 2012), STANDARD architecture —
resolution-flexible.

Uses the standard (torchvision-style) AlexNet: an 11x11 stride-4 conv stem,
5 conv layers, 3 max-pools, an AdaptiveAvgPool((6,6)), and three FC layers
(4096-4096-num_classes). This is the architecture whose parameter/FLOP counts
match the SAFSL paper (AlexNet ~= 60M params, ~0.098 GFLOPs forward at 64x64).
The AdaptiveAvgPool makes it run at ANY input resolution (32x32, 64x64, ...)
without changing the classifier dimension.

NOTE — earlier this file used a CIFAR-adapted variant (small 3x3 kernels, no
stride-4 stem) sized for 32x32. That was switched to the standard architecture
so the measured params/FLOPs match the paper.

Input: (B, 3, H, W)   (the SAFSL paper resizes CIFAR-10 / HAM10000 to 64x64)
Output: (B, num_classes)

Splittable: ordered_layers() exposes every layer (features + adaptive pool +
classifier) as a flat list — a pure feed-forward stack, so split_model() may
cut at any index in [1, N-1].
"""

import torch.nn as nn

from flsim.interfaces.splittable import Splittable


class AlexNet(nn.Module, Splittable):
    """
    AlexNet, standard architecture (see module docstring).

    This class does NOT:
    - Manage training loops.
    - Load data or apply normalisation.
    - Hold optimizer state.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))   # resolution-flexible; keeps 256*6*6=9216 FC input
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape (B, 3, H, W).

        Returns:
            tensor of shape (B, num_classes) — unnormalized logits.
        """
        return self.classifier(self.avgpool(self.features(x)))

    def ordered_layers(self) -> list:
        """features + adaptive pool + classifier, in forward order (flat, any cut valid)."""
        return list(self.features) + [self.avgpool] + list(self.classifier)
