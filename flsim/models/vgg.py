"""
models/vgg.py: VGG-16 / VGG-19 (Simonyan & Zisserman, 2014), STANDARD
architecture — resolution-flexible.

Uses the standard (torchvision-style) VGG with BatchNorm: the configuration-D
(VGG-16) / configuration-E (VGG-19) conv stack, an AdaptiveAvgPool((7,7)), and
three FC layers (4096-4096-num_classes). This is the architecture whose
parameter/FLOP counts match the SAFSL paper (VGG-19 ~= 143M params, ~1.72
GFLOPs forward at 64x64). The AdaptiveAvgPool makes it run at ANY input
resolution without changing the classifier dimension.

NOTE — earlier this file used a CIFAR-adapted variant (single Linear(512,10)
classifier, no 4096 FC layers) sized for 32x32. That was switched to the
standard architecture so the measured params/FLOPs match the paper.

cfg 16 (VGG-16 / D): [64,64,'M', 128,128,'M', 256,256,256,'M', 512,512,512,'M', 512,512,512,'M']
cfg 19 (VGG-19 / E): [64,64,'M', 128,128,'M', 256,256,256,256,'M', 512,512,512,512,'M', 512,512,512,512,'M']
(each number = Conv(in->n,3x3,pad=1) -> BatchNorm2d(n) -> ReLU; 'M' = MaxPool2d(2))

Input: (B, 3, H, W)   (the SAFSL paper resizes CIFAR-10 / HAM10000 to 64x64)
Output: (B, num_classes)

Splittable: ordered_layers() exposes every layer as a flat list (no skip
connections) — split_model() may cut at any index in [1, N-1].
"""

import torch.nn as nn

from flsim.interfaces.splittable import Splittable


_CFGS = {
    16: [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    19: [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _make_feature_layers(cfg: list) -> list:
    """Conv(3x3,pad=1)->BatchNorm->ReLU per number in cfg; MaxPool2d(2) per 'M'."""
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2))
        else:
            layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU())
            in_channels = v
    return layers


class VGG(nn.Module, Splittable):
    """
    Standard VGG-BN (see module docstring). Use VGG16 / VGG19 below rather than
    constructing this directly.

    This class does NOT:
    - Manage training loops.
    - Load data or apply normalisation.
    - Hold optimizer state.
    """

    def __init__(self, cfg_key: int, num_classes: int = 10):
        super().__init__()
        if cfg_key not in _CFGS:
            raise ValueError(f"cfg_key must be one of {sorted(_CFGS)}, got {cfg_key}")
        self.features = nn.Sequential(*_make_feature_layers(_CFGS[cfg_key]))
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))   # resolution-flexible; keeps 512*7*7 FC input
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
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


class VGG16(VGG):
    """VGG-16 (configuration D), standard VGG-BN architecture."""

    def __init__(self, num_classes: int = 10):
        super().__init__(cfg_key=16, num_classes=num_classes)


class VGG19(VGG):
    """VGG-19 (configuration E), standard VGG-BN architecture."""

    def __init__(self, num_classes: int = 10):
        super().__init__(cfg_key=19, num_classes=num_classes)
