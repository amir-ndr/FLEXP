"""
flsim/airsfl/model.py: CIFAR ResNet-18 for AirSFL (roadmap Sec. 4).

ResNet-18 BasicBlock [2,2,2,2] with a 3x3 stride-1 CIFAR stem (NO initial
max-pool), GroupNorm (32 groups) instead of BatchNorm, global average pool, and
a 512 -> num_classes head. GroupNorm is required so there are NO unsynchronized
batch-statistic buffers: the per-client server-side suffix copies and the
prefix stay plain parameters, so noiseless AirSFL / digital SFL-V1 coincide with
full-model FedAvg up to floating-point order.

split_at_stage(model, s) partitions at the END of residual stage s (s = 1..4):
  prefix (client)  = stem + layer[1..s]        -> outputs the cut activation
  suffix (server)  = layer[s+1..4] + pool + fc  -> per-client copy in SFL-V1
The profiled sizes (roadmap Sec. 6) are reproduced exactly by check_profiled_dims().
"""

import torch
import torch.nn as nn


def _gn(channels: int, groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(min(groups, channels), channels)


class BasicBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1, groups: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.gn1 = _gn(out_c, groups)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.gn2 = _gn(out_c, groups)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False), _gn(out_c, groups))

    def forward(self, x):
        idn = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.relu(out + idn)


class CifarResNet18GN(nn.Module):
    """ResNet-18 (GroupNorm, 3x3 stride-1 stem, no maxpool) for 32x32 inputs."""

    def __init__(self, num_classes: int = 10, groups: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False), _gn(64, groups), nn.ReLU(inplace=True))
        self.layer1 = self._make(64, 64, 2, 1, groups)
        self.layer2 = self._make(64, 128, 2, 2, groups)
        self.layer3 = self._make(128, 256, 2, 2, groups)
        self.layer4 = self._make(256, 512, 2, 2, groups)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make(in_c, out_c, blocks, stride, groups):
        layers = [BasicBlock(in_c, out_c, stride, groups)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_c, out_c, 1, groups))
        return nn.Sequential(*layers)

    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def forward(self, x, cut=None, perturb=None):
        """Full forward. If `perturb` is given, it is applied to the activation at the
        end of residual stage `cut` (used to inject the ZF reconstruction error)."""
        x = self.stem(x)
        for i, s in enumerate(self.stages(), start=1):
            x = s(x)
            if perturb is not None and i == cut:
                x = perturb(x)
        return self.fc(self.pool(x).flatten(1))

    def no_inplace(self):
        """Disable in-place ReLU (identical numerics; required by torch.func transforms)."""
        for mod in self.modules():
            if isinstance(mod, nn.ReLU):
                mod.inplace = False
        return self


def split_at_stage(model: CifarResNet18GN, stage: int):
    """Return (prefix, suffix) modules split at the end of residual `stage` (1..4)."""
    if not (1 <= stage <= 4):
        raise ValueError(f"stage must be in 1..4, got {stage}")
    stages = model.stages()
    prefix = nn.Sequential(model.stem, *stages[:stage])
    suffix = nn.Sequential(*(list(stages[stage:]) + [model.pool, nn.Flatten(1), model.fc]))
    return prefix, suffix


def _num_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def check_profiled_dims(batch_size: int = 16, verbose: bool = True) -> dict:
    """Build a fresh model and confirm d_c (prefix params), d_s (suffix params),
    and d_a (cut activation numel incl. batch) match the roadmap Sec. 6 table."""
    golden = {  # (d_c, d_s, d_a@B=16)
        1: (149824, 11024138, 1048576),
        2: (675392, 10498570, 524288),
        3: (2775104, 8398858, 262144),
        4: (11168832, 5130, 131072),
    }
    model = CifarResNet18GN(num_classes=10)
    x = torch.randn(batch_size, 3, 32, 32)
    out = {}
    ok = True
    for stage in (1, 2, 3, 4):
        prefix, suffix = split_at_stage(model, stage)
        with torch.no_grad():
            act = prefix(x)
        d_c, d_s, d_a = _num_params(prefix), _num_params(suffix), act.numel()
        g_c, g_s, g_a = golden[stage]
        match = (d_c == g_c and d_s == g_s and d_a == g_a * (batch_size // 16 if batch_size % 16 == 0 else 1)) \
            if batch_size == 16 else (d_c == g_c and d_s == g_s)
        ok &= (d_c == g_c and d_s == g_s and (batch_size != 16 or d_a == g_a))
        out[stage] = {"d_c": d_c, "d_s": d_s, "d_a": d_a}
        if verbose:
            print(f"  stage {stage}: d_c={d_c:>9,} (golden {g_c:>9,}) | "
                  f"d_s={d_s:>9,} (golden {g_s:>9,}) | d_a={d_a:>9,} (golden {g_a:>9,})  "
                  f"{'OK' if (d_c==g_c and d_s==g_s and (batch_size!=16 or d_a==g_a)) else 'MISMATCH'}")
    # total (prefix+suffix) must equal the whole model
    total_ok = _num_params(model) == _num_params(split_at_stage(model, 1)[0]) + _num_params(split_at_stage(model, 1)[1])
    if verbose:
        print(f"  full-model param check (prefix+suffix == whole): {total_ok}")
        print(f"  PROFILED_DIMS: {'PASS' if ok and total_ok else 'FAIL'}")
    out["ok"] = ok and total_ok
    return out


if __name__ == "__main__":
    print("=== AirSFL CIFAR ResNet-18 (GroupNorm) — profiled dimension check ===")
    check_profiled_dims()
