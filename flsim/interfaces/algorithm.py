"""
interfaces/algorithm.py: Base class for all federated learning algorithms.

Design principle — minimal override:
  Only aggregate() is @abstractmethod.
  select_clients() and configure_client() have sensible defaults so a new
  algorithm only needs to override the methods it actually changes.

  Typical override patterns:
    Custom aggregation only       → override aggregate()
    Custom selection only         → override select_clients()
    Custom per-client config only → override configure_client()
    Full custom algorithm         → override all three

  Example — channel-aware selection with FedAvg aggregation:

      class ChannelAwareAlg(FederatedAlgorithm):
          def select_clients(self, all_clients, num_to_select, rng):
              # sort by channel quality, pick top-K
              ...
          # aggregate() not overridden → uses sample-weighted FedAvg
          # configure_client() not overridden → no-op

  Example — FedProx (just add proximal term config):

      class FedProx(FederatedAlgorithm):
          def __init__(self, mu: float):
              self.mu = mu
          def configure_client(self, client, global_model, round_idx):
              client.proximal_mu = self.mu   # client.train() reads this
          # select_clients() not overridden → uniform random
          # aggregate() not overridden → sample-weighted FedAvg
"""

import copy
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch


class FederatedAlgorithm(ABC):
    """
    Base class for all FL algorithms.

    Provides ready-to-use default implementations for selection and
    client configuration. Only aggregate() must be implemented.

    This class does NOT:
    - Perform local training (Client.train() does that).
    - Know about the communication channel or time model.
    - Store or manage global model state (Server does that).
    """

    # ------------------------------------------------------------------
    # Default: uniform random client selection (used by FedAvg and most
    # baselines — override for channel-aware or importance-based selection)
    # ------------------------------------------------------------------

    def select_clients(
        self, all_clients: list, num_to_select: int, rng, **kwargs
    ) -> list:
        """
        Uniform random sampling without replacement.

        Override this to implement:
          - Channel-aware selection (pick clients with best channel)
          - Importance sampling (pick clients with most informative data)
          - Clustered selection
          - Any other custom policy

        The **kwargs carry system context the Simulator passes each round, so a
        channel-aware selector can estimate client speed BEFORE selection. The
        keys match the async AsyncFederatedAlgorithm.select_clients contract, so
        the same custom selector works in both sync and async:
            channel_model      — ChannelModel (call .channel_gain / .achievable_rate_bps)
            noise_psd_w_per_hz  — noise PSD in W/Hz
            bw_per_client_hz    — bandwidth each client would get (B / clients_per_round)
            round_idx           — current round index (sync only)
        Use what you need; ignore the rest.

        Args:
            all_clients (list[Client]): all K available clients.
            num_to_select (int): number of clients m to select this round.
            rng: numpy RandomState for reproducible selection.

        Returns:
            list[Client]: m selected clients.
        """
        indices = rng.choice(len(all_clients), size=num_to_select, replace=False)
        return [all_clients[i] for i in indices]

    # ------------------------------------------------------------------
    # Default: no-op configuration (used by FedAvg)
    # Override for FedProx (set mu), SCAFFOLD (distribute control variates),
    # or any algorithm that needs per-round per-client setup.
    # ------------------------------------------------------------------

    def configure_client(self, client, global_model, round_idx: int) -> None:
        """
        Set any per-round client hyperparameters before local training.

        Default: no-op (correct for FedAvg and FedSGD).

        Override for:
          - FedProx: client.proximal_mu = self.mu
          - SCAFFOLD: distribute control variates to client
          - Any algorithm needing per-round client state

        Args:
            client: Client object to configure.
            global_model: current global nn.Module (read-only).
            round_idx (int): current round index (0-based).
        """
        pass

    # ------------------------------------------------------------------
    # Abstract: aggregation — every algorithm MUST define this.
    # ------------------------------------------------------------------

    @abstractmethod
    def aggregate(self, global_model, client_updates: list) -> OrderedDict:
        """
        Combine client updates into a new global model state dict.

        This is the core of any FL algorithm and must always be overridden.

        Args:
            global_model: current global nn.Module (for shape reference).
            client_updates (list[ClientUpdate]): updates from selected clients.
                Each has: state_dict, num_samples, train_loss, compute_time_s, …

        Returns:
            OrderedDict: new global model state_dict (detached float32 tensors).
        """
