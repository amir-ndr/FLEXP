"""
models/resnet.py: ResNet-18 / ResNet-34 (He et al., 2015), adapted for CIFAR-10.

The original ResNet is sized for 224x224 ImageNet input (7x7 stride-2 stem +
3x3 stride-2 maxpool, downsampling by 4x before the first residual block).
Applied to CIFAR-10's 32x32 input, that stem alone would shrink the image to
8x8 before any residual block runs, discarding most of the (already small)
spatial detail. This is the standard CIFAR-10 ResNet adaptation used
throughout the FL/vision literature: the same BasicBlock / stage structure
and the same per-stage block counts as the ImageNet ResNet-18 ([2,2,2,2]
blocks) and ResNet-34 ([3,4,6,3] blocks), but the stem is a single 3x3
stride-1 conv (no initial maxpool) so the first residual stage still sees the
full 32x32 resolution.

Stage output sizes for a 32x32 input:
  stem               -> (B,64,32,32)
  layer1 (stride 1)  -> (B,64,32,32)
  layer2 (stride 2)  -> (B,128,16,16)
  layer3 (stride 2)  -> (B,256,8,8)
  layer4 (stride 2)  -> (B,512,4,4)
  avgpool(4)         -> (B,512,1,1) -> Linear(512,10)

Input: (B, 3, 32, 32)
Output: (B, 10)

Splittable — IMPORTANT caveat: unlike MnistCNN/CifarCNN/AlexNet/VGG (pure
feed-forward stacks), a BasicBlock has an internal skip connection
(out = relu(conv-bn-conv-bn(x) + shortcut(x))), so it CANNOT be decomposed
into its individual Conv/BatchNorm/ReLU layers the way split_model() expects
— chaining those pieces one at a time in a plain nn.Sequential would drop the
skip path entirely and silently produce a different, broken network. Each
BasicBlock is therefore exposed as ONE ATOMIC element of ordered_layers():
split_model() can cut BETWEEN blocks (or between stages), never inside one.
This matches how split learning is applied to ResNets in the literature (cut
points chosen at stage/block boundaries). ordered_layers() has 14 elements
for ResNet-18 (3 stem + 8 blocks + 3 head) and 22 for ResNet-34 (3 stem + 16
blocks + 3 head); valid cut_layer is [1, N-1].
"""

import torch.nn as nn

from flsim.interfaces.splittable import Splittable


class BasicBlock(nn.Module):
    """
    Standard ResNet basic block: two 3x3 convs + identity/projection shortcut.

    NOT independently Splittable — see module docstring. Used only as an
    atomic building block inside ResNet.
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNet(nn.Module, Splittable):
    """
    ResNet, CIFAR-10-adapted (see module docstring). Use ResNet18 / ResNet34
    below rather than constructing this directly.

    This class does NOT:
    - Manage training loops.
    - Load data or apply normalisation.
    - Hold optimizer state.
    """

    def __init__(self, num_blocks: list, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.layer1 = self._make_stage(64, num_blocks[0], stride=1)
        self.layer2 = self._make_stage(128, num_blocks[1], stride=2)
        self.layer3 = self._make_stage(256, num_blocks[2], stride=2)
        self.layer4 = self._make_stage(512, num_blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_stage(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*blocks)

    def forward(self, x):
        """
        Args:
            x: tensor of shape (B, 3, 32, 32).

        Returns:
            tensor of shape (B, num_classes) — unnormalized logits.
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        return self.linear(out)

    def ordered_layers(self) -> list:
        """Stem (conv1, bn1, relu) + every BasicBlock (atomic) + (avgpool, flatten, linear)."""
        return (
            [self.conv1, self.bn1, self.relu]
            + list(self.layer1) + list(self.layer2) + list(self.layer3) + list(self.layer4)
            + [self.avgpool, self.flatten, self.linear]
        )


class ResNet18(ResNet):
    """ResNet-18 ([2,2,2,2] BasicBlocks), CIFAR-10-adapted."""

    def __init__(self, num_classes: int = 10):
        super().__init__(num_blocks=[2, 2, 2, 2], num_classes=num_classes)


class ResNet34(ResNet):
    """ResNet-34 ([3,4,6,3] BasicBlocks), CIFAR-10-adapted."""

    def __init__(self, num_classes: int = 10):
        super().__init__(num_blocks=[3, 4, 6, 3], num_classes=num_classes)
