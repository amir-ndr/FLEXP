"""
models/resnet.py: ResNet-18 / ResNet-34 (He et al., 2015), STANDARD ImageNet
architecture — resolution-flexible.

Uses the paper's / torchvision's standard stem: a 7x7 stride-2 conv followed by
a 3x3 stride-2 max-pool, downsampling the input by 4x before the first residual
stage. This is the architecture whose FLOP counts match the SAFSL paper
(ResNet-34 ~= 0.30 GFLOPs forward at 64x64 input). An `AdaptiveAvgPool2d(1)`
before the classifier makes the network run at ANY input resolution (32x32,
64x64, ...) without changing the classifier dimension.

Stage output sizes for a 64x64 input (the SAFSL paper resizes CIFAR-10 /
HAM10000 to 64x64):
  stem conv 7x7 s2   -> (B,64,32,32)
  stem maxpool 3x3 s2-> (B,64,16,16)
  layer1 (stride 1)  -> (B,64,16,16)
  layer2 (stride 2)  -> (B,128,8,8)
  layer3 (stride 2)  -> (B,256,4,4)
  layer4 (stride 2)  -> (B,512,2,2)
  adaptiveavgpool(1) -> (B,512,1,1) -> Linear(512,num_classes)

Output: (B, num_classes)

NOTE — earlier this file used a CIFAR-adapted stem (3x3 stride-1, no maxpool)
that KEPT full resolution and therefore did ~8x MORE compute per sample than
the paper's numbers. That was switched to the standard downsampling stem so the
measured FLOPs match the paper. If you specifically want the high-resolution
CIFAR variant, restore a 3x3 stride-1 conv1 and drop self.maxpool.

Splittable — IMPORTANT caveat: a BasicBlock has an internal skip connection
(out = relu(conv-bn-conv-bn(x) + shortcut(x))), so it CANNOT be decomposed into
its individual Conv/BN/ReLU layers. Each BasicBlock is therefore ONE ATOMIC
element of ordered_layers(): split_model() may cut BETWEEN blocks/stages, never
inside one (the paper notes the ResNet split layer must sit at block
boundaries). ordered_layers() has 15 elements for ResNet-18 (4 stem + 8 blocks
+ 3 head) and 23 for ResNet-34 (4 stem + 16 blocks + 3 head); valid cut_layer
is [1, N-1].
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
    Standard ImageNet ResNet (see module docstring). Use ResNet18 / ResNet34
    below rather than constructing this directly.

    This class does NOT:
    - Manage training loops.
    - Load data or apply normalisation.
    - Hold optimizer state.
    """

    def __init__(self, num_blocks: list, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64

        # Standard ImageNet stem: 7x7 stride-2 conv + 3x3 stride-2 maxpool (4x downsample).
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_stage(64, num_blocks[0], stride=1)
        self.layer2 = self._make_stage(128, num_blocks[1], stride=2)
        self.layer3 = self._make_stage(256, num_blocks[2], stride=2)
        self.layer4 = self._make_stage(512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # resolution-flexible
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
            x: tensor of shape (B, 3, H, W) — any resolution (e.g. 64x64).

        Returns:
            tensor of shape (B, num_classes) — unnormalized logits.
        """
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        return self.linear(out)

    def ordered_layers(self) -> list:
        """Stem (conv1, bn1, relu, maxpool) + every BasicBlock (atomic) + (avgpool, flatten, linear)."""
        return (
            [self.conv1, self.bn1, self.relu, self.maxpool]
            + list(self.layer1) + list(self.layer2) + list(self.layer3) + list(self.layer4)
            + [self.avgpool, self.flatten, self.linear]
        )


class ResNet18(ResNet):
    """ResNet-18 ([2,2,2,2] BasicBlocks), standard ImageNet architecture."""

    def __init__(self, num_classes: int = 10):
        super().__init__(num_blocks=[2, 2, 2, 2], num_classes=num_classes)


class ResNet34(ResNet):
    """ResNet-34 ([3,4,6,3] BasicBlocks), standard ImageNet architecture."""

    def __init__(self, num_classes: int = 10):
        super().__init__(num_blocks=[3, 4, 6, 3], num_classes=num_classes)
