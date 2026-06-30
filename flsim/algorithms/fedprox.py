"""
algorithms/fedprox.py: FedProx — Federated Optimisation with Proximal Term.

Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020.

The only difference from FedAvg is the proximal term added to each client's
local objective:

    min F_k(w) + (μ/2) * ||w − w_global||²

This penalises client drift and improves convergence under statistical
heterogeneity (non-IID data). The proximal term is implemented in
Client.train() via the proximal_mu argument; FedProx sets it here in
configure_client() before each round.

select_clients() and aggregate() are inherited from FedAvg unchanged.
"""

from flsim.algorithms.fedavg import FedAvg


class FedProx(FedAvg):
    """
    FedProx: FedAvg + per-round proximal regularisation on each client.

    Overrides only configure_client() — selection and aggregation are
    identical to FedAvg.

    Args:
        mu (float): proximal coefficient μ ≥ 0.
            0.0  → identical to FedAvg (no regularisation).
            0.01 → light regularisation (good starting point).
            0.1  → stronger regularisation (use for highly non-IID data).
            1.0  → very strong (rarely needed; may slow convergence).
    """

    def __init__(self, mu: float = 0.01):
        self.mu = mu

    def configure_client(self, client, global_model, round_idx: int) -> None:
        """
        Set the proximal coefficient on the client before local training.

        Client.train() reads client.proximal_mu and adds the term
        (mu/2)*||w - w_global||² to each batch loss when mu > 0.
        """
        client.proximal_mu = self.mu

    # select_clients() → inherited: uniform random sampling
    # aggregate()      → inherited: sample-weighted FedAvg
