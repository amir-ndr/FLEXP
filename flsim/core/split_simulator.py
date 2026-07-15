"""
core/split_simulator.py: Orchestrator for split learning (SL / SFLV1 / SFLV2),
per Thapa, Chamikara Mahawaga Arachchige, Camtepe & Sun, "SplitFed: When
Federated Learning Meets Split Learning" (AAAI-22, arXiv:2004.12088),
Algorithm 1 & 2, and the "Variants of Splitfed Learning" discussion.

Why a new simulator (not FederatedAlgorithm.aggregate())
------------------------------------------------------------
Every other algorithm in this framework (FedAvg, FedProx, FedAsync, FedOTA)
trains one FULL model per client and aggregates complete state_dicts. Split
learning trains TWO cooperating sub-models (client-side, server-side) via a
forward/backward RELAY across a simulated communication boundary (see
flsim.core.split_client.SplitClient) — a fundamentally different mechanic
that doesn't fit the aggregate(global_model, client_updates) contract. So,
exactly as AsyncSimulator exists alongside Simulator for async's different
mechanics, SplitSimulator exists alongside both for split learning's.

The client_mode / server_mode framing
------------------------------------------
The paper describes three variants, but they reduce to two independent,
reusable choices — how the CLIENT side is combined across clients, and how
the SERVER side is:

  "sequential"      : ONE persistent model instance. Clients are processed
                       one at a time (order re-shuffled each global epoch);
                       each client's local training continues directly from
                       wherever the previous client left off. No aggregation
                       is ever computed — there's only ever one copy.
  "parallel_fedavg"  : Every selected client gets an INDEPENDENT deep copy of
                       the current global sub-model, trains it locally for
                       `local_epochs` epochs, and at the end of the global
                       epoch all copies are combined via sample-weighted
                       averaging (FedAvg): W_{t+1} = sum_k (n_k/n) W_{k,t}.

The paper's three named variants are exactly these two axes:

  SL     (client_mode="sequential",      server_mode="sequential")
         Relay-based: literally one client-side and one server-side model,
         handed from client to client. Matches Table 1's "Client-side
         training: Sequential" and "Model aggregation: No" for SL.
  SFLV1  (client_mode="parallel_fedavg", server_mode="parallel_fedavg")
         "the server-side models of all clients are executed separately in
         parallel and then aggregated to obtain the global server-side model
         at each global epoch" (paper, "Variants of Splitfed Learning").
         Client-side aggregated by the fed server the same way.
  SFLV2  (client_mode="parallel_fedavg", server_mode="sequential")
         "SFLV2 processes the forward-backward propagations of the
         server-side model sequentially with respect to the client's smashed
         data (no FedAvg of the server-side models). The client order is
         chosen randomly in the server-side operations" — client-side
         unchanged from SFLV1 ("The client-side operation remains the same
         as in the SFLV1").

This orthogonal framing is also why the module docstring you're reading
promises this is easy to extend: a hypothetical "SplitFedAvg" variant, or any
other custom combination, is just a different choice on these same two axes
(or a custom aggregation-weight function — see `weight_fn`) — no new
orchestration code needed.

Scope note (communication time / energy)
--------------------------------------------
This implementation focuses on getting the ALGORITHM (forward/backward relay,
the three aggregation patterns, cut_layer) exactly right — verified against
the paper (see flsim/core/split_client.py's docstring for the relay
correctness proof: split-relay training is numerically IDENTICAL to training
the unsplit model directly). It does not simulate communication time or
energy the way the sync/async/OTA simulators do (TimeModel/EnergyModel/
ChannelModel) — the paper's own communication-cost analysis (its Table 2:
comms. per client, total comms., total training time as closed-form
expressions in |W|, p, q, K, R, T_fedavg) is a separate, analytically-defined
model that isn't wired in here. This can be added as a follow-up using the
formulas straight from Table 2 if you want communication/energy plots
alongside the accuracy curves this module produces.
"""

import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from flsim.system.split_model import SplitFullModel


@dataclass
class SplitEpochResult:
    """Metrics for one global epoch of split-learning training."""
    global_epoch: int
    train_loss: float          # sample-weighted mean training loss this epoch
    num_clients: int           # number of clients processed this epoch
    test_loss: Optional[float] = None
    test_accuracy: Optional[float] = None


def _weighted_average_state_dicts(state_dicts: list, weights: list) -> OrderedDict:
    """
    Sample-weighted average of a list of state_dicts (paper's FedAvg
    aggregation, applied identically to whichever side is in parallel_fedavg
    mode): W = sum_k (n_k / sum(n)) * W_k.

    Args:
        state_dicts: list[OrderedDict], all with identical keys/shapes.
        weights:     list[float], same length (e.g. num_samples per client).

    Returns:
        OrderedDict: the weighted-average state_dict (detached float32 tensors).
    """
    total = float(sum(weights))
    assert total > 0, "weights must sum to > 0"
    agg = OrderedDict()
    ref = state_dicts[0]
    for key, param in ref.items():
        agg[key] = torch.zeros_like(param, dtype=torch.float32)
    for sd, w in zip(state_dicts, weights):
        frac = w / total
        for key, param in sd.items():
            agg[key] += frac * param.float().detach()
    return agg


class SplitSimulator:
    """
    Runs split learning (SL / SFLV1 / SFLV2, or any custom client_mode /
    server_mode combination) for a configured number of global epochs.

    Args:
        clients (list[SplitClient]): all participating clients.
        client_model (nn.Module): initial global client-side sub-model
            (from split_model()). Owned/mutated by this simulator.
        server_model (nn.Module): initial global server-side sub-model.
        evaluator (Evaluator): evaluates the combined model on the test set.
        config: SimpleNamespace config (reads learning.global_rounds,
            learning.local_epochs, learning.batch_size, learning.learning_rate,
            evaluation.evaluate_every; learning.clients_per_round selects a
            subset each epoch if set, else all clients participate — matching
            the paper's own experimental setting "all participants update the
            model in each global epoch, i.e. C=1").
        rng (np.random.RandomState): controls client-order shuffling
            (sequential modes) and client selection (if clients_per_round < N).
        device (torch.device): training device.
        client_mode (str): "sequential" | "parallel_fedavg" — see module docstring.
        server_mode (str): "sequential" | "parallel_fedavg" — see module docstring.

    This class does NOT:
    - Compute simulated time, energy, or channel metrics (see module docstring).
    - Decide which dataset/model to use (both come in pre-built).
    """

    def __init__(
        self,
        clients: list,
        client_model: nn.Module,
        server_model: nn.Module,
        evaluator,
        config,
        rng: np.random.RandomState,
        device: torch.device,
        client_mode: str = "parallel_fedavg",
        server_mode: str = "parallel_fedavg",
    ):
        valid_modes = ("sequential", "parallel_fedavg")
        if client_mode not in valid_modes or server_mode not in valid_modes:
            raise ValueError(
                f"client_mode/server_mode must be one of {valid_modes}, "
                f"got client_mode={client_mode!r}, server_mode={server_mode!r}"
            )
        self.clients      = clients
        self.client_model = client_model.to(device)
        self.server_model = server_model.to(device)
        self.evaluator     = evaluator
        self.config        = config
        self.rng           = rng
        self.device        = device
        self.client_mode   = client_mode
        self.server_mode   = server_mode
        self.history: list = []   # list[SplitEpochResult], filled by run()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> list:
        """
        Execute global_rounds global epochs. Returns self.history (also
        stored on the instance for later inspection/plotting).
        """
        cfg_learn = self.config.learning
        cfg_eval  = self.config.evaluation
        T = cfg_learn.global_rounds
        k = int(getattr(cfg_learn, "clients_per_round", 0) or len(self.clients))
        k = min(k, len(self.clients))

        mode_label = f"client={self.client_mode}, server={self.server_mode}"
        print(f"\n[SplitSimulator] Starting: T={T} epochs, {k}/{len(self.clients)} "
              f"clients/epoch, {mode_label}, device={self.device}")

        for epoch in range(T):
            selected = self._select_clients(k)

            if self.server_mode == "sequential":
                mean_loss = self._run_epoch_server_sequential(selected)
            else:
                mean_loss = self._run_epoch_server_parallel_fedavg(selected)

            eval_result = None
            if epoch % cfg_eval.evaluate_every == 0:
                combined = SplitFullModel(self.client_model, self.server_model)
                eval_result = self.evaluator.evaluate(combined, device=self.device)

            result = SplitEpochResult(
                global_epoch=epoch,
                train_loss=mean_loss,
                num_clients=len(selected),
                test_loss=eval_result.test_loss if eval_result else None,
                test_accuracy=eval_result.test_accuracy if eval_result else None,
            )
            self.history.append(result)

            if eval_result is not None:
                print(
                    f"  Epoch {epoch:4d} | train_loss={mean_loss:.4f} | "
                    f"acc={eval_result.test_accuracy:.4f} | loss={eval_result.test_loss:.4f}"
                )

        print("[SplitSimulator] Done.")
        return self.history

    # ------------------------------------------------------------------
    # Client selection
    # ------------------------------------------------------------------

    def _select_clients(self, k: int) -> list:
        """Uniform random selection of k of N clients (default k=N, matching
        the paper's C=1 full-participation setting)."""
        if k >= len(self.clients):
            return list(self.clients)
        indices = self.rng.choice(len(self.clients), size=k, replace=False)
        return [self.clients[i] for i in indices]

    # ------------------------------------------------------------------
    # One global epoch — server_mode="sequential" branch
    # (covers SL: client_mode="sequential" too, and SFLV2:
    #  client_mode="parallel_fedavg")
    # ------------------------------------------------------------------

    def _run_epoch_server_sequential(self, selected: list) -> float:
        cfg = self.config.learning
        order = self.rng.permutation(len(selected))

        client_state_dicts, client_weights = [], []
        total_loss, total_samples = 0.0, 0

        for idx in order:
            client = selected[idx]

            # client_mode governs whether this client trains an independent
            # copy of the client-side model (parallel_fedavg, aggregated at
            # the end of the epoch) or the ONE shared instance directly (SL).
            if self.client_mode == "parallel_fedavg":
                client_model = copy.deepcopy(self.client_model)
            else:  # "sequential"
                client_model = self.client_model

            # server_mode="sequential": always the SAME shared server_model
            # object, mutated in place across clients in this random order —
            # this is the defining trait of both SL and SFLV2's server side.
            c_sd, s_sd, n, loss = client.train_local(
                client_model=client_model,
                server_model=self.server_model,
                local_epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                device=self.device,
            )
            # server_model already updated in place — nothing further to do
            # for the server side.

            if self.client_mode == "parallel_fedavg":
                client_state_dicts.append(c_sd)
                client_weights.append(n)
            # else ("sequential"): self.client_model IS client_model, already
            # updated in place — nothing further to do for the client side.

            total_loss += loss * n
            total_samples += n

        if self.client_mode == "parallel_fedavg":
            new_client_state = _weighted_average_state_dicts(client_state_dicts, client_weights)
            self.client_model.load_state_dict(new_client_state)

        return total_loss / max(total_samples, 1)

    # ------------------------------------------------------------------
    # One global epoch — server_mode="parallel_fedavg" branch
    # (covers SFLV1: client_mode="parallel_fedavg" too)
    # ------------------------------------------------------------------

    def _run_epoch_server_parallel_fedavg(self, selected: list) -> float:
        cfg = self.config.learning

        client_state_dicts, client_weights = [], []
        server_state_dicts, server_weights = [], []
        total_loss, total_samples = 0.0, 0

        for client in selected:
            # Every client gets an independent copy of BOTH sides, starting
            # from this epoch's global snapshot — "executed separately in
            # parallel" per the paper.
            client_model = (
                copy.deepcopy(self.client_model) if self.client_mode == "parallel_fedavg"
                else self.client_model
            )
            server_model = copy.deepcopy(self.server_model)

            c_sd, s_sd, n, loss = client.train_local(
                client_model=client_model,
                server_model=server_model,
                local_epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                device=self.device,
            )

            if self.client_mode == "parallel_fedavg":
                client_state_dicts.append(c_sd)
                client_weights.append(n)
            server_state_dicts.append(s_sd)
            server_weights.append(n)

            total_loss += loss * n
            total_samples += n

        if self.client_mode == "parallel_fedavg":
            new_client_state = _weighted_average_state_dicts(client_state_dicts, client_weights)
            self.client_model.load_state_dict(new_client_state)

        new_server_state = _weighted_average_state_dicts(server_state_dicts, server_weights)
        self.server_model.load_state_dict(new_server_state)

        return total_loss / max(total_samples, 1)
