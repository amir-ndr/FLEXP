"""
models/vgg.py: VGG-16 / VGG-19 (Simonyan & Zisserman, 2014), adapted for CIFAR-10.

The original VGG is sized for 224x224 ImageNet input, ending in a 7x7x512
feature map fed to three 4096-wide FC layers. Applied to CIFAR-10's 32x32
input, the same 5-maxpool conv stack instead collapses to a 1x1x512 feature
map (32 / 2^5 = 1) — so the classifier only needs ONE Linear(512, 10) layer,
not the three 4096-wide FC layers ImageNet requires. This is the standard
CIFAR-10 VGG adaptation used throughout the FL/vision literature, and
additionally folds BatchNorm into every conv block (VGG-BN): plain VGG is
notoriously hard to train from scratch at CIFAR-10 scale without it — this
mirrors CifarCNN's existing use of BatchNorm in this codebase.

cfg 16 (VGG-16 / configuration D):
  [64,64,'M', 128,128,'M', 256,256,256,'M', 512,512,512,'M', 512,512,512,'M']
cfg 19 (VGG-19 / configuration E):
  [64,64,'M', 128,128,'M', 256,256,256,256,'M', 512,512,512,512,'M', 512,512,512,512,'M']
(each number = Conv(in->n, 3x3, pad=1) -> BatchNorm2d(n) -> ReLU; 'M' = MaxPool2d(2))

Input: (B, 3, 32, 32)
Output: (B, 10)

Splittable (flsim.interfaces.splittable): ordered_layers() exposes every
Conv/BatchNorm/ReLU/MaxPool/Flatten/Linear as a flat list (no skip
connections anywhere in VGG), so flsim.system.split_model.split_model() can
cut the network at ANY index in [1, N-1] (N = 46 for VGG-16, 55 for VGG-19).
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
    VGG, CIFAR-10-adapted (see module docstring). Use VGG16 / VGG19 below
    rather than constructing this directly.

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
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # (B,512,1,1) -> (B, 512)
            nn.Linear(512, num_classes),
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
        """features (Conv/BatchNorm/ReLU/MaxPool, in forward order) then classifier."""
        return list(self.features) + list(self.classifier)


class VGG16(VGG):
    """VGG-16 (configuration D), CIFAR-10-adapted."""

    def __init__(self, num_classes: int = 10):
        super().__init__(cfg_key=16, num_classes=num_classes)


class VGG19(VGG):
    """VGG-19 (configuration E), CIFAR-10-adapted."""

    def __init__(self, num_classes: int = 10):
        super().__init__(cfg_key=19, num_classes=num_classes)
