"""
core/client.py: Federated learning client — owns local data and performs local training.

Client.train() executes real PyTorch gradient steps and returns the trained
state dict. It does NOT compute simulated time or energy — those are computed
by TimeModel and EnergyModel in the Simulator after train() returns.

IMPORTANT: The wall-clock time of train() is NEVER used as simulated time.
"""

import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from flsim.core.training_utils import iter_local_batches


@dataclass
class ClientUpdate:
    """
    All information produced by one client in one communication round.

    Separates actual training results (state_dict, loss) from simulated
    system metrics (time, energy, channel) so each can be reasoned about
    independently.
    """
    client_id: int
    state_dict: OrderedDict          # trained model weights
    num_samples: int                 # D_k used in this round

    # Training metrics (real PyTorch values)
    train_loss: float

    # Simulated time components (seconds) — filled by Simulator, not by Client
    compute_time_s: float = 0.0
    upload_time_s:  float = 0.0
    download_time_s: float = 0.0
    total_time_s:   float = 0.0      # sum of above three

    # Energy (joules) — filled by Simulator
    compute_energy_j: float = 0.0
    tx_energy_j:      float = 0.0
    total_energy_j:   float = 0.0

    # Channel info — filled by Simulator
    channel_gain: float = 0.0
    achievable_rate_bps: float = 0.0
    allocated_bandwidth_hz: float = 0.0


class Client:
    """
    Federated learning client.

    Owns a local dataset (as a list of dataset indices) and performs local
    SGD training when train() is called.

    This class does NOT:
    - Compute simulated time (that belongs to CellularTimeModel).
    - Compute energy (that belongs to EnergyModel).
    - Know about the communication channel.
    - Aggregate model updates (that belongs to the algorithm).
    """

    def __init__(self, client_id: int, dataset, indices: list, profile):
        """
        Args:
            client_id (int): unique identifier for this client.
            dataset: full training dataset; client accesses via indices.
            indices (list[int]): dataset indices assigned to this client.
            profile: ClientSystemProfile with system heterogeneity parameters.
        """
        self.client_id = client_id
        self.dataset = dataset
        self.indices = indices
        self.profile = profile

    @property
    def num_samples(self) -> int:
        """D_k: number of local training samples."""
        return len(self.indices)

    def train(
        self,
        global_model: nn.Module,
        local_epochs: int,
        batch_size: int,
        learning_rate: float,
        device: torch.device,
        proximal_mu: float = 0.0,
        max_iters: int = None,
    ) -> tuple:
        """
        Perform local SGD on a deep copy of the global model.

        When proximal_mu > 0 (FedProx), adds a proximal term to each batch:
            loss += (mu/2) * ||w - w_global||²
        This penalises drift from the global model and stabilises training
        under statistical heterogeneity.

        Args:
            global_model (nn.Module): current global model (read-only reference).
            local_epochs (int): I_k — number of local training epochs (used only
                when max_iters is None).
            batch_size (int): mini-batch size for SGD.
            learning_rate (float): local SGD learning rate.
            device (torch.device): device to train on.
            proximal_mu (float): FedProx proximal coefficient μ ≥ 0.
                0.0 (default) = standard FedAvg (no proximal term).
            max_iters (int, optional): H — if set, do exactly H mini-batch SGD
                steps this round instead of local_epochs full passes (the
                "local iterations" unit of semi-async split-FL papers). See
                flsim.core.training_utils.iter_local_batches. None (default)
                keeps the original full-epoch behaviour.

        Returns:
            tuple[OrderedDict, int, float]:
                (trained_state_dict, num_samples_used, mean_train_loss)
        """
        local_model = copy.deepcopy(global_model).to(device)
        local_model.train()

        # Keep a frozen reference of the global weights for the proximal term.
        # Only materialised when mu > 0 to avoid the copy overhead for FedAvg.
        global_params = (
            [p.detach().clone() for p in global_model.parameters()]
            if proximal_mu > 0.0 else None
        )

        subset = Subset(self.dataset, self.indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        total_loss  = 0.0
        total_batches = 0

        for x, y in iter_local_batches(loader, local_epochs, max_iters):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(local_model(x), y)

            # FedProx proximal term: (μ/2) * ||w - w_global||²
            if proximal_mu > 0.0:
                prox = sum(
                    ((p - g.to(device)) ** 2).sum()
                    for p, g in zip(local_model.parameters(), global_params)
                )
                loss = loss + (proximal_mu / 2.0) * prox

            loss.backward()
            optimizer.step()
            total_loss  += loss.item()
            total_batches += 1

        mean_loss = total_loss / max(total_batches, 1)
        return local_model.state_dict(), self.num_samples, mean_loss
