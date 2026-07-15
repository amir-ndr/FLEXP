"""
system/split_model.py: Cuts a Splittable model into client-side / server-side
sub-networks for split learning (Vepakomma et al. 2018; Thapa et al. 2022
"SplitFed", arXiv:2004.12088).

Client-side model W^C = layers[0:cut_layer]
Server-side model W^S = layers[cut_layer:]

`cut_layer` is the paper's "cut layer" concept exactly: the boundary index
after which activations ("smashed data") cross the wire instead of raw model
weights. It is exposed as a config option (learning.cut_layer in YAML) —
see flsim/experiments/split_wiring.py — so you don't need to touch code to
try a different split point.
"""

import copy
from typing import Tuple

import torch
import torch.nn as nn


def split_model(model: nn.Module, cut_layer: int) -> Tuple[nn.Module, nn.Module]:
    """
    Split a Splittable model into independent (client_side, server_side) copies.

    The two returned modules are FRESH deep copies of the relevant layers —
    not references into `model` — so training either one never mutates
    `model` or the other half. This matters for split learning specifically:
    SFLV1/SFLV2 need multiple independent per-client copies of each side per
    global epoch, and aliased parameters between them would silently corrupt
    every other copy's training.

    Args:
        model (nn.Module): a model implementing Splittable.ordered_layers()
            (e.g. MnistCNN, CifarCNN). Only used as a layer/weight source —
            not mutated.
        cut_layer (int): number of layers (from ordered_layers()) given to
            the client. Must satisfy 1 <= cut_layer <= N-1 for an N-layer
            model, so neither side is empty (cut_layer=0 would give the
            client nothing to compute; cut_layer=N would give the server
            nothing to compute — split learning requires both to do real work).

    Returns:
        (client_side, server_side): two nn.Sequential modules whose chained
        forward passes (server_side(client_side(x))) reproduce model(x)
        exactly, at cut_layer's initial weights.

    Raises:
        TypeError: if model doesn't implement ordered_layers().
        ValueError: if cut_layer is out of the valid [1, N-1] range.
    """
    if not hasattr(model, "ordered_layers"):
        raise TypeError(
            f"{type(model).__name__} does not implement ordered_layers() — "
            f"it can't be used with split learning. See flsim.interfaces.splittable.Splittable."
        )
    layers = model.ordered_layers()
    n = len(layers)
    if not (1 <= cut_layer <= n - 1):
        raise ValueError(
            f"cut_layer must be in [1, {n - 1}] for this {n}-layer "
            f"{type(model).__name__} (client and server must each get >= 1 layer), "
            f"got cut_layer={cut_layer}."
        )

    client_layers = copy.deepcopy(nn.ModuleList(layers[:cut_layer]))
    server_layers = copy.deepcopy(nn.ModuleList(layers[cut_layer:]))
    client_side = nn.Sequential(*client_layers)
    server_side = nn.Sequential(*server_layers)
    return client_side, server_side


def num_layers(model: nn.Module) -> int:
    """len(model.ordered_layers()) — for validating cut_layer against a model before splitting."""
    if not hasattr(model, "ordered_layers"):
        raise TypeError(
            f"{type(model).__name__} does not implement ordered_layers() — "
            f"see flsim.interfaces.splittable.Splittable."
        )
    return len(model.ordered_layers())


class SplitFullModel(nn.Module):
    """
    Chains a client-side and server-side sub-model into one nn.Module for
    evaluation (Evaluator.evaluate() expects a single model(x) call).

    Only used for computing test accuracy/loss on the combined model — split
    learning never trains through this wrapper directly (client-side and
    server-side are always trained/exchanged separately via the relay
    mechanism in flsim.core.split_client). Rebuild a fresh SplitFullModel
    whenever you want to evaluate a new (client_side, server_side) snapshot;
    it holds references (not copies) to the two sub-models you pass in, so
    evaluating always reflects their current weights.

    Args:
        client_side (nn.Module): the current client-side sub-model.
        server_side (nn.Module): the current server-side sub-model.
    """

    def __init__(self, client_side: nn.Module, server_side: nn.Module):
        super().__init__()
        self.client_side = client_side
        self.server_side = server_side

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.server_side(self.client_side(x))
