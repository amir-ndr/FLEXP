"""
core/split_client.py: Split-learning client — local training via the
forward/backward RELAY mechanism (Vepakomma et al. 2018, "SplitFed" Alg. 2).

Unlike flsim.core.client.Client (which trains a single full model locally),
SplitClient trains a (client_side, server_side) MODEL PAIR together: the
client does a forward pass up to the cut layer, "sends" the resulting
activations (smashed data) across a simulated wire to the server, the server
finishes the forward pass and computes the loss, then gradients flow back
across the same wire so the client can finish its own backward pass. Both
sides' parameters are updated by ordinary SGD.

This class does NOT decide whether client_side/server_side are shared across
clients or per-client independent copies — it just trains whatever model
objects it's given, IN PLACE (mirroring the caller-controls-copying pattern
used throughout this codebase's Simulator, which deep-copies before calling
Client.train() when it wants an independent copy). That decision belongs to
the orchestrator (flsim.core.split_simulator.SplitSimulator), which is what
distinguishes SL / SFLV1 / SFLV2 from each other — see its module docstring.
"""

from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn


class SplitClient:
    """
    Split-learning client: owns local data, trains a client/server model pair
    via the forward/backward relay.

    This class does NOT:
    - Decide aggregation strategy (FedAvg, sequential hand-off, etc.) — that's
      the orchestrator's job.
    - Copy the models it's given — callers control that (see module docstring).
    - Compute simulated time/energy/channel metrics (out of scope for this
      first split-learning implementation — see SplitSimulator's module
      docstring for why).
    """

    def __init__(self, client_id: int, dataset, indices: list):
        """
        Args:
            client_id (int): unique identifier for this client.
            dataset: full training dataset; client accesses via indices.
            indices (list[int]): dataset indices assigned to this client.
        """
        self.client_id = client_id
        self.dataset = dataset
        self.indices = indices

    @property
    def num_samples(self) -> int:
        """D_k: number of local training samples."""
        return len(self.indices)

    def train_local(
        self,
        client_model: nn.Module,
        server_model: nn.Module,
        local_epochs: int,
        batch_size: int,
        learning_rate: float,
        device: torch.device,
    ) -> tuple:
        """
        Train client_model and server_model together via the relay mechanism,
        for local_epochs passes over this client's own data. Both models are
        updated IN PLACE by their optimizers.

        Per batch (paper Algorithm 2 / Algorithm 1's inner loop):
          1. Client forward pass up to the cut layer -> smashed data A.
          2. "Send" A across the wire: detach it and mark it as a fresh leaf
             requiring grad, exactly simulating what the server would receive
             (no direct autograd link back into the client's graph).
          3. Server forward pass on the received A -> loss.
          4. loss.backward() populates BOTH the server's parameter gradients
             AND the relayed leaf's gradient (dA) in one call — that .grad IS
             the "gradient of the smashed data" the paper has the server send
             back to the client.
          5. Server optimizer step.
          6. Relay dA back: call backward on the ORIGINAL (non-detached)
             smashed data with dA as the upstream gradient, continuing
             backprop into the client's own layers.
          7. Client optimizer step.

        Args:
            client_model: the client-side sub-model (layers before the cut).
            server_model: the server-side sub-model (layers from the cut on).
            local_epochs (int): E — number of local epochs per Algorithm 2.
            batch_size (int): mini-batch size for SGD.
            learning_rate (float): SGD learning rate (shared by both optimizers,
                matching the paper's single eta for the whole split network).
            device (torch.device): device to train on.

        Returns:
            tuple[OrderedDict, OrderedDict, int, float]:
                (client_state_dict, server_state_dict, num_samples_used, mean_train_loss)
        """
        client_model = client_model.to(device)
        server_model = server_model.to(device)
        client_model.train()
        server_model.train()

        client_opt = torch.optim.SGD(client_model.parameters(), lr=learning_rate)
        server_opt = torch.optim.SGD(server_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        subset = Subset(self.dataset, self.indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False)

        total_loss = 0.0
        total_batches = 0

        for _ in range(local_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)

                client_opt.zero_grad()
                server_opt.zero_grad()

                # ---- client forward (up to cut layer) ----
                smashed = client_model(x)

                # ---- "send" smashed data across the wire ----
                # Fresh leaf tensor, disconnected from the client's graph —
                # this IS the simulated communication boundary.
                smashed_relay = smashed.detach().requires_grad_(True)

                # ---- server forward + backward ----
                output = server_model(smashed_relay)
                loss = criterion(output, y)
                loss.backward()   # fills server_model grads AND smashed_relay.grad (= dA)
                server_opt.step()

                # ---- relay dA back to the client, finish client backward ----
                smashed.backward(smashed_relay.grad)
                client_opt.step()

                total_loss += loss.item()
                total_batches += 1

        mean_loss = total_loss / max(total_batches, 1)
        return client_model.state_dict(), server_model.state_dict(), self.num_samples, mean_loss
