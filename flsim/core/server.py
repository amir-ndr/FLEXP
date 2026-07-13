"""
core/server.py: Federation server — holds the global model and delegates to the algorithm.

Server is a thin coordinator: it owns the global model and calls the injected
FederatedAlgorithm for client selection and aggregation. It never trains models
or computes time/energy — those responsibilities belong to Client and Simulator.
"""

import torch.nn as nn

from flsim.interfaces.algorithm import FederatedAlgorithm


class Server:
    """
    Federation server.

    Holds the global model and delegates client selection and aggregation
    to the injected FederatedAlgorithm.

    This class does NOT:
    - Perform local training.
    - Compute simulated time, energy, or channel metrics.
    - Implement any algorithm-specific logic directly.
    """

    def __init__(self, model: nn.Module, algorithm: FederatedAlgorithm):
        """
        Args:
            model (nn.Module): initial global model (randomly initialised).
            algorithm (FederatedAlgorithm): aggregation and selection strategy.
        """
        self.global_model = model
        self.algorithm = algorithm
        self.round_idx = 0

    def select_clients(
        self, all_clients: list, num_to_select: int, rng, **kwargs
    ) -> list:
        """
        Delegate client selection to the algorithm.

        Args:
            all_clients (list[Client]): all available clients.
            num_to_select (int): number to select for this round.
            rng: numpy RandomState.
            **kwargs: system context (channel_model, noise_psd_w_per_hz,
                bw_per_client_hz, round_idx) forwarded to the algorithm so
                channel-aware selectors can rank clients before selection.

        Returns:
            list[Client]: selected clients.
        """
        return self.algorithm.select_clients(
            all_clients, num_to_select, rng, **kwargs
        )

    def aggregate(self, client_updates: list) -> None:
        """
        Aggregate client updates into the global model in-place.

        Calls algorithm.aggregate() then loads the result into global_model.
        Increments round_idx.

        Args:
            client_updates (list[ClientUpdate]): updates from selected clients.
        """
        new_state = self.algorithm.aggregate(self.global_model, client_updates)
        self.global_model.load_state_dict(new_state)
        self.round_idx += 1
