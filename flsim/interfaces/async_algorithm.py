"""
interfaces/async_algorithm.py: Base class for all asynchronous FL algorithms.

Design — three override points, only one is required:

  select_clients(all_clients, num_to_trigger, rng, **kwargs)
      Which clients the Scheduler dispatches each cycle.
      Default: uniform random without replacement.

  mixing_weight(base_alpha, staleness) -> float
      Staleness-adaptive alpha_t = base_alpha * s(t - tau).
      Default: constant s = 1 (no staleness decay).
      Override for polynomial, hinge, or any custom decay.

  aggregate_async(global_model, update, global_epoch, staleness, alpha_t) -> OrderedDict
      REQUIRED. How one arriving update changes the global model.
      alpha_t is pre-computed by the simulator via mixing_weight() so it
      can be logged; the algorithm receives the final value directly.

Typical patterns
----------------
  # Only change the mixing rule:
  class MyDecay(FedAsync):
      def mixing_weight(self, alpha, staleness):
          return alpha * math.exp(-0.1 * staleness)

  # Only change client selection (pick the 1 fastest always):
  class FastOnly(FedAsync):
      def select_clients(self, all_clients, num_to_trigger, rng, **kwargs):
          return sorted(all_clients,
                        key=lambda c: c.profile.cycles_per_sample)[:num_to_trigger]

  # Custom update rule (momentum mixing):
  class MomentumAsync(AsyncFederatedAlgorithm):
      def __init__(self, beta=0.9, alpha=0.1):
          self.beta, self.alpha = beta, alpha
          self._v = None   # momentum buffer
      def aggregate_async(self, global_model, update, epoch, staleness, alpha_t):
          ...

Client selection modes (use or subclass these ready-made variants):
  FedAsync              — random selection, tunable staleness function
  FedAsyncTopKFastTotal — always pick the K fastest clients (compute + upload)
  FedAsyncFixedFast     — random from the top-M fastest clients
"""

from abc import ABC, abstractmethod
from collections import OrderedDict


class AsyncFederatedAlgorithm(ABC):
    """
    Base class for asynchronous FL algorithms.

    The simulator calls these methods in order every global epoch:
      1. select_clients()    — Scheduler: which client to dispatch next
      2. configure_client()  — per-dispatch client setup (optional)
      3. mixing_weight()     — compute alpha_t (logged by simulator)
      4. aggregate_async()   — Updater: apply the arriving update
    """

    # ------------------------------------------------------------------
    # Scheduler: client selection
    # ------------------------------------------------------------------

    def select_clients(
        self,
        all_clients: list,
        num_to_trigger: int,
        rng,
        **kwargs,
    ) -> list:
        """
        Select clients for one Scheduler dispatch cycle.

        Default: uniform random sampling without replacement.

        Override for:
          - Top-K fastest (sort by compute time estimate)
          - Fixed fast pool (always the same N fast clients)
          - Ratio-based (return int(ratio * len(all_clients)) clients,
            ignoring num_to_trigger)
          - Any custom policy

        The **kwargs may carry extra context passed by the simulator
        (e.g., current_virtual_time, time_model) — use what you need,
        ignore the rest.

        Args:
            all_clients:     all Client objects in the federation.
            num_to_trigger:  how many to select this cycle.
            rng:             numpy RandomState for reproducibility.

        Returns:
            list[Client]: selected clients (length == num_to_trigger
            unless the override deliberately returns a different size).
        """
        indices = rng.choice(len(all_clients), size=num_to_trigger, replace=False)
        return [all_clients[i] for i in indices]

    # ------------------------------------------------------------------
    # Per-dispatch client configuration (optional)
    # ------------------------------------------------------------------

    def configure_client(self, client, global_model, global_epoch: int) -> None:
        """
        Set per-dispatch client state before local training.

        Called at dispatch time with the snapshot of the global model
        the client will train on.  Default: no-op.

        Override for FedProx-style proximal terms, control variates, etc.
        """
        pass

    # ------------------------------------------------------------------
    # Updater: staleness-adaptive mixing weight
    # ------------------------------------------------------------------

    def mixing_weight(self, base_alpha: float, staleness: int) -> float:
        """
        Compute the effective mixing weight alpha_t.

        alpha_t = base_alpha * s(staleness)

        The simulator calls this to get alpha_t, logs the value, and
        passes it to aggregate_async() — so mixing_weight() is called
        exactly once per global epoch.

        Default: constant s = 1  →  alpha_t = base_alpha always.

        Override to implement:
          Polynomial : s(k) = (k + 1)^{-a}
          Hinge      : s(k) = 1 if k <= b, else 1 / (a*(k-b) + 1)
          Custom     : any monotonically non-increasing function of k

        Args:
            base_alpha: the base hyperparameter α ∈ (0, 1).
            staleness:  t − τ ≥ 0.

        Returns:
            float: effective alpha_t ∈ (0, base_alpha].
        """
        return base_alpha

    # ------------------------------------------------------------------
    # Updater: global model update (must override)
    # ------------------------------------------------------------------

    @abstractmethod
    def aggregate_async(
        self,
        global_model,
        update,
        global_epoch: int,
        staleness: int,
        alpha_t: float,
    ) -> OrderedDict:
        """
        Compute the new global model state dict from one arriving update.

        Standard FedAsync rule:
            x_t = (1 - alpha_t) * x_{t-1} + alpha_t * x_new

        Args:
            global_model:  current global nn.Module (read state_dict from this).
            update:        ClientUpdate — has .state_dict, .num_samples,
                           .train_loss, .compute_time_s, …
            global_epoch:  t — number of model updates processed so far
                           (0-based, incremented AFTER this call).
            staleness:     t − τ ≥ 0.  τ is the epoch at which the client
                           received the global model it trained on.
            alpha_t:       effective mixing weight, already computed by
                           mixing_weight(base_alpha, staleness).

        Returns:
            OrderedDict: new global model state_dict (detached float32 tensors).
        """
