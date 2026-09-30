"""
flsim/airsfl/compute.py: computation-time model for AirSFL and its baselines, so that
end-to-end training time = uplink communication time + computation time (downlinks
remain ideal and untimed, as in the paper).

FLOP counting (per training sample), same convention as flsim.system.flops:
  * only Conv2d and Linear layers are counted (GroupNorm / ReLU / residual adds /
    pooling add < 1% and are omitted); FLOPs = 2 x multiply-accumulates;
  * forward (FP) of a layer = 2 * MACs;
  * backward (BP) of a layer = weight-gradient (2 * MACs) + input-gradient (2 * MACs),
    except the very first convolution, whose input is the image and needs no
    gradient (weight-gradient only). The suffix's first layer DOES compute its input
    gradient -- that is the cut derivative returned to the client.
  Prefix + suffix FLOPs therefore equal the full model's FLOPs at every cut.

Per-round computation latency (sequential stages of each co-split step, as in
AdaptSFL; clients run in parallel, the slowest client paces every step):
  split methods (AirSFL, digital SFL-V1 [OFDMA or ZF], hybrid FDMA/ZF-AirComp SFL):
      client FP : tau * b * C_FP / f_min            C_FP  = client-side FP FLOPs/sample
      server FP : tau * N * b * Phi_FP / f_s         Phi_FP = server-side FP FLOPs/sample
      server BP : tau * N * b * Phi_BP / f_s         (the M-server runs all N suffix copies
      client BP : tau * b * C_BP / f_min              on its SHARED capacity f_s)
  FL methods (AirComp-FL, digital FedAvg): the client trains the whole model
      client FP : tau * b * W_FP / f_min,   client BP : tau * b * W_BP / f_min
  with f_min = min_i f_i. Server-side aggregation (a weighted sum of N vectors) costs
  ~2*N*d FLOPs, i.e. microseconds at f_s, and is omitted.

Computing capabilities (AdaptSFL setting): f_i ~ U[1, 2] TFLOPS per edge device (drawn
once per seed, identical for every method), f_s = 20 TFLOPS for the M-server. The
uniform draws are stored so any other [lo, hi] / f_s can be re-applied exactly later.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from flsim.airsfl.model import CifarResNet18GN
from flsim.airsfl.timing import SPLIT_METHODS   # noqa: F401  (re-exported: split methods run SFL compute)
from flsim.system.flops import _conv2d_macs, _linear_macs


@dataclass
class ComputeConfig:
    client_tflops_lo: float = 1.0      # edge-device capability range (TFLOPS), uniform
    client_tflops_hi: float = 2.0
    server_tflops: float = 20.0        # M-server capability (TFLOPS), shared by all suffix copies


def _stage_of(name: str) -> int:
    """0 = stem, 1..4 = residual stages, 5 = head (pool + fc)."""
    if name.startswith("stem"):
        return 0
    for s in (1, 2, 3, 4):
        if name.startswith(f"layer{s}."):
            return s
    return 5


@lru_cache(maxsize=None)
def layer_flops(num_classes: int = 10) -> tuple:
    """Per-sample (name, stage, fp_flops, bp_flops) for every Conv2d / Linear layer."""
    model = CifarResNet18GN(num_classes)
    rows, handles = [], []
    first_conv = next(n for n, m in model.named_modules() if isinstance(m, nn.Conv2d))

    def hook(name):
        def f(mod, inp, out):
            macs = _conv2d_macs(mod, out) if isinstance(mod, nn.Conv2d) else _linear_macs(mod)
            fp = 2 * macs
            bp = 2 * macs + (0 if name == first_conv else 2 * macs)   # weight grad + input grad
            rows.append((name, _stage_of(name), fp, bp))
        return f

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            handles.append(mod.register_forward_hook(hook(name)))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32))
    for h in handles:
        h.remove()
    return tuple(rows)


def split_flops(stage: Optional[int], num_classes: int = 10) -> dict:
    """Per-sample FLOPs of the client prefix (stem + stages 1..stage) and the server
    suffix. stage=None -> no split: the client holds the whole model."""
    cut = 5 if stage is None else int(stage)
    out = {"client_fp": 0, "client_bp": 0, "server_fp": 0, "server_bp": 0}
    for _, st, fp, bp in layer_flops(num_classes):
        side = "client" if st <= cut else "server"
        out[f"{side}_fp"] += fp
        out[f"{side}_bp"] += bp
    return out


def client_draws(N: int, seed: int) -> np.ndarray:
    """Uniform U[0,1] draws that fix each client's capability for a seed (paired across
    methods): f_i = lo + (hi - lo) * u_i."""
    return np.random.RandomState(int(seed) * 7919 + 43).uniform(size=int(N))


def compute_time_breakdown(method: str, flops: dict, N: int, b: int, tau: int,
                           f_client_min: float, f_server: float) -> dict:
    """Per-round computation seconds {client_fp, client_bp, server_fp, server_bp, total}.
    flops: split_flops(...) for the method's cut (FL methods: stage=None)."""
    ct = tau * b / f_client_min
    if method in SPLIT_METHODS:
        st = tau * N * b / f_server
        out = {"client_fp": ct * flops["client_fp"], "client_bp": ct * flops["client_bp"],
               "server_fp": st * flops["server_fp"], "server_bp": st * flops["server_bp"]}
    else:
        w_fp = flops["client_fp"] + flops["server_fp"]
        w_bp = flops["client_bp"] + flops["server_bp"]
        out = {"client_fp": ct * w_fp, "client_bp": ct * w_bp, "server_fp": 0.0, "server_bp": 0.0}
    out["total"] = out["client_fp"] + out["client_bp"] + out["server_fp"] + out["server_bp"]
    return out


def verify_compute(verbose: bool = True) -> bool:
    """Checks: (1) conv/linear MACs == closed form for CIFAR ResNet-18 and == the
    framework counter; (2) prefix + suffix == full model at every cut, FP and BP;
    (3) BP / FP = 2 except the stem's weight-only gradient; (4) limit case N = 1,
    f_s = f_client: splitting does not change the total computation time;
    (5) a hand calculation of the default per-round times."""
    from flsim.system.flops import forward_macs
    ok = True
    macs = sum(fp for _, _, fp, _ in layer_flops()) // 2
    stage_macs = lambda cin, c, hw: (cin * c * 9 + 3 * c * c * 9 + cin * c) * hw * hw   # conv1+3 convs+1x1 ds
    closed = (3 * 64 * 9 * 32 * 32 + 4 * 64 * 64 * 9 * 32 * 32                 # stem, layer1
              + stage_macs(64, 128, 16) + stage_macs(128, 256, 8) + stage_macs(256, 512, 4)
              + 512 * 10)                                                       # fc
    fw = forward_macs(CifarResNet18GN(), torch.zeros(1, 3, 32, 32))
    ok &= macs == closed == fw
    full = split_flops(None)
    for s in (1, 2, 3, 4):
        f = split_flops(s)
        ok &= f["client_fp"] + f["server_fp"] == full["client_fp"]
        ok &= f["client_bp"] + f["server_bp"] == full["client_bp"]
    stem = next(fp for n, st, fp, bp in layer_flops() if st == 0)
    ok &= full["client_bp"] == 2 * full["client_fp"] - stem
    # (4) N = 1, equal capability: SFL computation == FL computation
    f1 = 1.5e12
    for s in (1, 2, 3, 4):
        sfl = compute_time_breakdown("airsfl", split_flops(s), 1, 16, 5, f1, f1)["total"]
        fl = compute_time_breakdown("aircomp_fl", split_flops(None), 1, 16, 5, f1, f1)["total"]
        ok &= abs(sfl - fl) <= 1e-12 * fl
    # (5) hand check at cut 2, N=30, b=16, tau=5, f_min=1 TFLOPS, f_s=20 TFLOPS
    f2 = split_flops(2)
    hand_client = 5 * 16 * (f2["client_fp"] + f2["client_bp"]) / 1e12
    hand_server = 5 * 30 * 16 * (f2["server_fp"] + f2["server_bp"]) / 20e12
    got = compute_time_breakdown("digital_sflv1", f2, 30, 16, 5, 1e12, 20e12)
    ok &= abs(got["client_fp"] + got["client_bp"] - hand_client) < 1e-12
    ok &= abs(got["server_fp"] + got["server_bp"] - hand_server) < 1e-12
    if verbose:
        print("=== computation model (Conv2d + Linear FLOPs, FLOPs = 2 x MACs) ===")
        print(f"  forward MACs/sample = {macs:,} (closed form {closed:,}, framework counter {fw:,})")
        print(f"  full model: FP {full['client_fp']/1e9:.4f} GFLOPs, BP {full['client_bp']/1e9:.4f} GFLOPs per sample")
        for s in (1, 2, 3, 4):
            f = split_flops(s)
            print(f"  cut {s}: client FP {f['client_fp']/1e9:.4f} BP {f['client_bp']/1e9:.4f} | "
                  f"server FP {f['server_fp']/1e9:.4f} BP {f['server_bp']/1e9:.4f} GFLOPs/sample "
                  f"(client share {f['client_fp']/full['client_fp']:.1%})")
        print(f"  cut 2, N=30, b=16, tau=5, f_min=1 TFLOPS, f_s=20 TFLOPS: client {hand_client:.4f} s, "
              f"server {hand_server:.4f} s per round")
        print(f"  COMPUTE_CHECKS: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


if __name__ == "__main__":
    verify_compute()
