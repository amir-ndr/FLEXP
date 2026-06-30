"""
algorithms/fedavg.py: FedAvg — Federated Averaging.

McMahan et al., "Communication-Efficient Learning of Deep Networks from
Decentralized Data", AISTATS 2017.

select_clients()    → inherited default: uniform random sampling
configure_client()  → inherited default: no-op
aggregate()         → sample-size weighted average of full state_dicts
"""

from collections import OrderedDict

import torch

from flsim.interfaces.algorithm import FederatedAlgorithm


class FedAvg(FederatedAlgorithm):
    """
    FedAvg: sample-size weighted aggregation, uniform random selection.

    Only aggregate() is overridden — selection and client config use
    the defaults defined in FederatedAlgorithm.
    """

    def aggregate(self, global_model, client_updates: list) -> OrderedDict:
        """
        Sample-size weighted federated averaging over the full state_dict.

        Formula: w_global = sum_k (n_k / N) * w_k
        where n_k = num_samples for client k, N = total samples.

        Includes BatchNorm buffers (running_mean, running_var, num_batches_tracked)
        because they are part of the state_dict.

        Args:
            global_model: current global nn.Module (shape reference).
            client_updates (list[ClientUpdate]): updates from selected clients.

        Returns:
            OrderedDict: aggregated state_dict.
        """
        total_samples = sum(u.num_samples for u in client_updates)
        assert total_samples > 0, "All client updates have zero samples"

        agg_state = OrderedDict()
        ref_state  = client_updates[0].state_dict
        for key, param in ref_state.items():
            agg_state[key] = torch.zeros_like(param, dtype=torch.float32)

        for update in client_updates:
            weight = update.num_samples / total_samples
            for key, param in update.state_dict.items():
                agg_state[key] += weight * param.float().detach()

        return agg_state
