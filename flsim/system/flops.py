"""
system/flops.py: Lightweight forward-pass FLOP (MAC) counter for splitting a
model's compute between the device-side and server-side sub-networks.

Split learning charges device-side FP/BP on the (weak) device CPU frequency and
server-side FP/BP on the (fast) edge-server frequency. To do that fairly we need
to know what FRACTION of the model's compute lives on each side of the cut. This
module measures it directly by counting multiply-accumulate operations (MACs)
for the layer types that dominate CNN compute — Conv2d and Linear. Everything
else (ReLU, MaxPool, BatchNorm, Flatten, Dropout) contributes negligible
arithmetic and is ignored.

We count FORWARD MACs; the device/server *fraction* of forward MACs is used as
the fraction of the config's total `cycles_per_sample` assigned to each side
(backward is ~2x forward on both sides, so the fraction is unchanged). This
keeps the split-learning compute model anchored to the SAME `cycles_per_sample`
that the sync/async simulators use — only its allocation across the cut changes.
"""

import torch
import torch.nn as nn


def _conv2d_macs(module: nn.Conv2d, out) -> int:
    """MACs for a Conv2d forward: out_elems × (in_ch/groups × kH × kW), per sample."""
    out_h, out_w = out.shape[-2], out.shape[-1]
    kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
    in_per_group = module.in_channels // module.groups
    return module.out_channels * out_h * out_w * in_per_group * kh * kw


def _linear_macs(module: nn.Linear) -> int:
    """MACs for a Linear forward: in_features × out_features, per sample."""
    return module.in_features * module.out_features


def forward_macs(model: nn.Module, sample_input: torch.Tensor) -> int:
    """
    Count forward multiply-accumulate operations for one sample through `model`.

    Args:
        model (nn.Module): the (sub-)model to profile.
        sample_input (torch.Tensor): a real input batch. The per-layer MAC
            formulas use output spatial/channel dims (not the batch dim), so the
            result is already PER SAMPLE regardless of the batch size passed.

    Returns:
        int: forward MACs per sample (Conv2d + Linear only).
    """
    total = {"macs": 0}
    handles = []

    def make_hook(mod):
        def hook(m, inp, out):
            if isinstance(m, nn.Conv2d):
                total["macs"] += _conv2d_macs(m, out)
            elif isinstance(m, nn.Linear):
                total["macs"] += _linear_macs(m)
        return hook

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(make_hook(m)))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(sample_input)
    if was_training:
        model.train()
    for h in handles:
        h.remove()

    return total["macs"]


def compute_split_fraction(
    client_model: nn.Module,
    server_model: nn.Module,
    sample_input: torch.Tensor,
) -> float:
    """
    Fraction of forward MACs that live on the CLIENT (device) side of the cut.

    Runs `sample_input` through the client model to get the smashed activations,
    then those through the server model, counting MACs on each side.

    Args:
        client_model, server_model: the two halves from split_model().
        sample_input (torch.Tensor): a real input batch (e.g. one training batch).

    Returns:
        float: device_macs / (device_macs + server_macs), in [0, 1]. If a model
        has no Conv2d/Linear layers on one side (all MACs on the other), returns
        the appropriate 0.0 or 1.0.
    """
    client_model = client_model.to(sample_input.device)
    server_model = server_model.to(sample_input.device)

    dev_macs = forward_macs(client_model, sample_input)
    with torch.no_grad():
        smashed = client_model(sample_input)
    srv_macs = forward_macs(server_model, smashed)

    total = dev_macs + srv_macs
    if total == 0:
        # Neither side has counted compute (unlikely) — split the cycles evenly.
        return 0.5
    return dev_macs / total
