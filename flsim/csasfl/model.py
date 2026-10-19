"""
flsim/csasfl/model.py: the 12-layer chain-topology LeNet of the CSA-SFL paper (layer order and
convolution widths of Wu et al., JSAC 2023, Table III; FC widths chosen to give the paper's
~1.4 M parameters / ~5.6 MB in FP32).

    index  layer   configuration                     output (MNIST 28x28)
      1    CONV1   32 filters 3x3, pad 1, ReLU        32 x 28 x 28
      2    CONV2   32 filters 3x3, pad 1, ReLU        32 x 28 x 28
      3    POOL1   max 2x2                            32 x 14 x 14
      4    CONV3   64 filters 3x3, pad 1, ReLU        64 x 14 x 14
      5    CONV4   64 filters 3x3, pad 1, ReLU        64 x 14 x 14
      6    POOL2   max 2x2                            64 x  7 x  7
      7    CONV5   128 filters 3x3, pad 1, ReLU      128 x  7 x  7
      8    CONV6   128 filters 3x3, pad 1, ReLU      128 x  7 x  7   <- split layer (cut after 8)
      9    POOL3   max 2x2, ceil                     128 x  4 x  4
     10    FC1     2048 -> 512, ReLU
     11    FC2     512 -> 128, ReLU
     12    FC3     128 -> 10 (logits; softmax in the cross-entropy loss)

    parameters: 1,402,474 (5.61 MB at 32 bits); device side (layers 1-8) 286,432,
    server side (layers 9-12) 1,116,042; smashed data 128*7*7 = 6,272 values per sample;
    the device side carries ~96% of the forward MACs. Initialization: He (Kaiming-normal) weights,
    zero biases (see LeNet12).

Two views of the same network:
  * LeNet12 (nn.Module, Splittable) -- parameter initialization, and usable by the rest of
    flsim (registered as "lenet12" in flsim.models.factory);
  * forward_layers(spec, params, x, start, stop) -- the same forward written functionally over
    a flat list of parameter tensors (in LeNet12.parameters() order), which is what the
    batched training engine (flsim.csasfl.engine) vmaps over devices.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

FC_WIDTHS = (512, 128)
CUT_LAYER = 8                     # device side = layers 1..8 (CONV1..CONV6)
CONV_WIDTHS = (32, 32, "P", 64, 64, "P", 128, 128, "P")


def layer_spec(num_classes: int = 10, in_ch: int = 1, image: int = 28, fc=FC_WIDTHS) -> list:
    """The 12 layers as ("conv", c_in, c_out) / ("pool", ceil_mode) / ("fc", n_in, n_out, relu)."""
    spec, c, s = [], in_ch, image
    pools = 0
    for w in CONV_WIDTHS:
        if w == "P":
            pools += 1
            ceil = pools == 3                    # POOL3 keeps the 7x7 border row/column (7 -> 4)
            s = -(-s // 2) if ceil else s // 2
            spec.append(("pool", ceil))
        else:
            spec.append(("conv", c, w))
            c = w
    n_in = c * s * s
    for i, n_out in enumerate(tuple(fc) + (num_classes,)):
        spec.append(("fc", n_in, n_out, i < len(fc)))
        n_in = n_out
    return spec


class LeNet12(nn.Module):
    """nn.Module form of layer_spec(); ordered_layers() returns the 12 layers (Splittable).

    init="he" (default): Kaiming-normal weights (fan_in, ReLU gain) and zero biases. With
    PyTorch's default init (gain 1/sqrt(3) per layer) this 9-ReLU-deep plain network does not
    leave the loss plateau under plain SGD at eta = 0.01 within an epoch of MNIST (loss stays
    2.30); with He init it reaches ~94% test accuracy after 300 steps. init="torch" keeps the
    PyTorch default."""

    def __init__(self, num_classes: int = 10, in_ch: int = 1, image: int = 28, fc=FC_WIDTHS, init: str = "he"):
        super().__init__()
        self.spec = layer_spec(num_classes, in_ch, image, fc)
        layers, n_fc, n_conv, n_pool = [], 0, 0, 0
        for kind, *a in self.spec:
            if kind == "conv":
                n_conv += 1
                layers.append((f"conv{n_conv}", nn.Sequential(nn.Conv2d(a[0], a[1], 3, padding=1), nn.ReLU())))
            elif kind == "pool":
                n_pool += 1
                layers.append((f"pool{n_pool}", nn.MaxPool2d(2, ceil_mode=a[0])))
            else:
                n_fc += 1
                mods = ([nn.Flatten()] if n_fc == 1 else []) + [nn.Linear(a[0], a[1])] + ([nn.ReLU()] if a[2] else [])
                layers.append((f"fc{n_fc}", nn.Sequential(*mods)))
        self.layers = nn.Sequential(OrderedDict(layers))
        if init == "he":
            for mod in self.modules():
                if isinstance(mod, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
                    nn.init.zeros_(mod.bias)
        elif init != "torch":
            raise ValueError(f"init must be 'he' or 'torch', got {init!r}")

    def forward(self, x):
        return self.layers(x)

    def ordered_layers(self) -> list:
        return list(self.layers)


def forward_layers(spec: list, params: list, x: torch.Tensor, start: int = 0, stop: int = None) -> torch.Tensor:
    """Functional forward of spec[start:stop]; `params` holds exactly those layers' (weight, bias)
    tensors in order. Works per example inside torch.func.vmap (x then has no device axis)."""
    i = 0
    for kind, *a in spec[start:stop]:
        if kind == "conv":
            x = F.relu(F.conv2d(x, params[i], params[i + 1], padding=1))
            i += 2
        elif kind == "pool":
            x = F.max_pool2d(x, 2, ceil_mode=a[0])
        else:
            if x.dim() > 2:
                x = x.flatten(1)
            x = F.linear(x, params[i], params[i + 1])
            i += 2
            if a[2]:
                x = F.relu(x)
    return x


def n_param_tensors(spec: list, start: int = 0, stop: int = None) -> int:
    return sum(2 for kind, *_ in spec[start:stop] if kind != "pool")


def layer_macs(spec: list, image: int = 28) -> list:
    """Forward multiply-accumulates per sample of each layer (conv: H*W*C_out*C_in*9, fc: n_in*n_out)."""
    out, s = [], image
    for kind, *a in spec:
        if kind == "conv":
            out.append(s * s * a[1] * a[0] * 9)
        elif kind == "pool":
            s = -(-s // 2) if a[0] else s // 2
            out.append(0)
        else:
            out.append(a[0] * a[1])
    return out


def activation_numel(spec: list, cut: int, in_ch: int = 1, image: int = 28) -> int:
    """Values per sample of the smashed data leaving layer `cut`."""
    with torch.no_grad():
        m = LeNet12(spec[-1][2], in_ch, image)
        z = torch.zeros(1, in_ch, image, image)
        for L in m.ordered_layers()[:cut]:
            z = L(z)
    return int(z[0].numel())
