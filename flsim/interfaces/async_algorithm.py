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

Semi-asynchronous (buffered) algorithms
----------------------------------------
Set `buffer_size = k > 1` (default 1) to switch the simulator from fully
async (one arrival at a time) to semi-async: the server buffers the k
clients that finish first among the `window_size` currently in flight,
aggregates them together with ONE mixing step via aggregate_buffered(),
and immediately re-dispatches k replacements — the remaining
(window_size - k) clients keep training uninterrupted in the background.
k=1 (default) is standard FedAsync. k=window_size degenerates to
synchronous FedAvg-style batching. See FedAsyncTopKFastTotal.

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

Client selection / concurrency modes (use or subclass these ready-made variants):
  FedAsync              — random selection, tunable staleness function, fully async (k=1)
  FedAsyncTopKFastTotal — semi-async: buffers the k fastest-to-arrive clients per epoch
"""

from abc import ABC, abstractmethod
from collections import OrderedDict


class AsyncFederatedAlgorithm(ABC):
    """
    Base class for asynchronous FL algorithms.

    The simulator calls these methods in order every global epoch:
      1. select_clients()    — Scheduler: which client(s) to dispatch next
      2. configure_client()  — per-dispatch client setup (optional)
      3. mixing_weight()     — compute alpha_t (logged by simulator)
      4. aggregate_async()   — Updater: apply one arriving update
                                (buffer_size == 1, the default)
         or aggregate_buffered() — apply k arriving updates together
                                (buffer_size == k > 1, semi-async)

    Class attribute:
      buffer_size (int): number of client updates the simulator buffers
          before triggering one global model update. Default 1 (fully
          async — every arrival triggers aggregate_async() immediately).
          Set > 1 (e.g. via a subclass's __init__) to switch to semi-async
          buffered aggregation via aggregate_buffered(). Must be
          <= async_fl.window_size (validated by AsyncSimulator).
    """

    buffer_size: int = 1

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
          - Ratio-based (return int(ratio * len(all_clients)) clients,
            ignoring num_to_trigger)
          - Any custom policy

        Note: for semi-async (buffer_size > 1) algorithms, "fastest k
        clients" does not need to be implemented here — it emerges
        naturally from the simulator buffering the k earliest arrivals
        among whichever clients are currently in flight. See
        FedAsyncTopKFastTotal.

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

    # ------------------------------------------------------------------
    # Updater: semi-async buffered update (only used when buffer_size > 1)
    # ------------------------------------------------------------------

    def aggregate_buffered(
        self,
        global_model,
        updates: list,
        global_epoch: int,
        stalenesses: list,
        alpha_t: float,
    ) -> OrderedDict:
        """
        Compute the new global model state dict from k buffered updates
        (called instead of aggregate_async() when buffer_size == k > 1).

        Only needs to be overridden by semi-async algorithms; the default
        raises NotImplementedError since aggregating multiple updates is
        not well-defined for the standard one-at-a-time FedAsync rule.

        Args:
            global_model:  current global nn.Module (read state_dict from this).
            updates:       list[ClientUpdate] — the k updates buffered this epoch,
                           in arrival order (earliest first).
            global_epoch:  t — number of model updates processed so far.
            stalenesses:   list[int] — t - tau_i for each update, same order as `updates`.
            alpha_t:       effective mixing weight, already computed by
                           mixing_weight(base_alpha, representative_staleness)
                           — the simulator uses max(stalenesses) as the
                           representative staleness for the whole batch.

        Returns:
            OrderedDict: new global model state_dict (detached float32 tensors).
        """
        raise NotImplementedError(
            f"{type(self).__name__} has buffer_size > 1 but does not "
            f"override aggregate_buffered()."
        )
