"""
algorithms/fedasync.py: FedAsync — Asynchronous Federated Optimization (Xie et al., 2019).

Global update rule:
    x_t = (1 - alpha_t) * x_{t-1} + alpha_t * x_new
    alpha_t = base_alpha * s(t - tau)

Local worker objective (paper §3, Algorithm 1 "Process Worker"):
    g_xt(x; z) = f(x; z) + (rho / 2) * ||x - xt||^2
Set FedAsync(rho=...) (or async_fl.rho in YAML) to enable the proximal term;
rho=0 (default) reduces local training to plain SGD. The paper requires
rho > mu (the weak-convexity constant of f) for the Theorem 5 convergence
guarantee.

Three built-in staleness strategies (paper §5.2):
  "constant"   : s(k) = 1                              → FedAsync+Const
  "polynomial" : s(k) = (k + 1)^{-a}                  → FedAsync+Poly
  "hinge"      : s(k) = 1           if k <= b          → FedAsync+Hinge
                         1 / (a*(k-b)+1)  otherwise

Two built-in concurrency modes:
  FedAsync              — fully async (buffer_size=1): every arrival triggers
                          an immediate global model update, uniform random
                          client selection.
  FedAsyncTopKFastTotal — semi-async (buffer_size=k>1): buffers the k clients
                          that finish first among the window_size clients
                          currently in flight, aggregates them together with
                          ONE mixing step, and immediately re-dispatches k
                          replacements. The other (window_size - k) clients
                          keep training uninterrupted. k=1 reduces to plain
                          FedAsync; k=window_size degenerates to synchronous
                          batching.

Writing a custom async algorithm
---------------------------------
Inherit from AsyncFederatedAlgorithm (or FedAsync for the standard mixing rule)
and override the methods you want to change:

    from flsim.interfaces.async_algorithm import AsyncFederatedAlgorithm
    from collections import OrderedDict
    import torch

    class MyAsyncAlg(AsyncFederatedAlgorithm):

        def select_clients(self, all_clients, num_to_trigger, rng, **kwargs):
            # Example: always pick the client with the best channel gain
            # (requires passing channel_gain info via kwargs or profiling)
            return sorted(all_clients,
                          key=lambda c: -c.profile.cpu_frequency_hz)[:num_to_trigger]

        def mixing_weight(self, base_alpha, staleness):
            # Custom exponential decay
            import math
            return base_alpha * math.exp(-0.05 * staleness)

        def aggregate_async(self, global_model, update, global_epoch, staleness, alpha_t):
            # Custom update: weighted by num_samples as well
            w = update.num_samples / max(update.num_samples, 1)
            effective_alpha = alpha_t * w
            current = global_model.state_dict()
            new_state = OrderedDict()
            for key in current:
                new_state[key] = (
                    (1.0 - effective_alpha) * current[key].float()
                    + effective_alpha * update.state_dict[key].float()
                )
            return new_state
"""

from collections import OrderedDict

import numpy as np

from flsim.interfaces.async_algorithm import AsyncFederatedAlgorithm


class FedAsync(AsyncFederatedAlgorithm):
    """
    Standard FedAsync with pluggable staleness function and random client selection.

    Args:
        alpha (float):          base mixing hyperparameter α ∈ (0, 1). Default 0.1.
        staleness_func (str):   "constant" | "polynomial" | "hinge". Default "constant".
        a (float):              exponent for polynomial; slope coefficient for hinge.
        b (float):              staleness threshold for hinge (ignored otherwise).
        rho (float):            proximal regularization weight ρ ≥ 0 for the local
                                 worker objective g_xt(x; z) = f(x; z) + (ρ/2)‖x−xt‖²
                                 (paper §3, Algorithm 1). Default 0.0 = no proximal
                                 term (plain local SGD). The paper requires ρ > μ,
                                 the weak-convexity constant of f, for the Theorem 5
                                 convergence guarantee to hold. If omitted here, the
                                 simulator falls back to config.async_fl.rho (YAML).

    Examples:
        FedAsync(alpha=0.1)
        FedAsync(alpha=0.5, staleness_func="polynomial", a=0.5)
        FedAsync(alpha=0.5, staleness_func="hinge", a=10.0, b=4.0)
        FedAsync(alpha=0.1, rho=0.01)   # with proximal regularization
    """

    def __init__(
        self,
        alpha: float = 0.1,
        staleness_func: str = "constant",
        a: float = 1.0,
        b: float = 4.0,
        rho: float = None,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if staleness_func not in ("constant", "polynomial", "hinge"):
            raise ValueError(
                f"staleness_func must be 'constant', 'polynomial', or 'hinge', "
                f"got '{staleness_func}'"
            )
        if rho is not None and rho < 0.0:
            raise ValueError(f"rho must be >= 0, got {rho}")
        self.alpha         = alpha
        self.staleness_func = staleness_func
        self.a             = a
        self.b             = b
        self.rho           = rho

    # ------------------------------------------------------------------
    # Staleness-adaptive mixing weight
    # ------------------------------------------------------------------

    def mixing_weight(self, base_alpha: float, staleness: int) -> float:
        """alpha_t = base_alpha * s(staleness)"""
        return base_alpha * self._s(staleness)

    def _s(self, staleness: int) -> float:
        """Staleness scaling function s(k), always in (0, 1]."""
        if self.staleness_func == "constant":
            return 1.0
        elif self.staleness_func == "polynomial":
            # sa(k) = (k + 1)^{-a}   →   s(0) = 1, decreasing
            return float(staleness + 1) ** (-self.a)
        else:  # "hinge"
            # sa,b(k) = 1 if k <= b, else 1 / (a*(k-b) + 1)
            if staleness <= self.b:
                return 1.0
            return 1.0 / (self.a * (staleness - self.b) + 1.0)

    # ------------------------------------------------------------------
    # Core update rule
    # ------------------------------------------------------------------

    def aggregate_async(
        self,
        global_model,
        update,
        global_epoch: int,
        staleness: int,
        alpha_t: float,
    ) -> OrderedDict:
        """
        x_t = (1 - alpha_t) * x_{t-1} + alpha_t * x_new

        alpha_t is passed in pre-computed by the simulator (via mixing_weight).
        """
        current   = global_model.state_dict()
        new_state = OrderedDict()
        for key in current:
            new_state[key] = (
                (1.0 - alpha_t) * current[key].float()
                + alpha_t       * update.state_dict[key].float()
            )
        return new_state


# ---------------------------------------------------------------------------
# Semi-asynchronous (buffered top-K) variant
# ---------------------------------------------------------------------------

class FedAsyncTopKFastTotal(FedAsync):
    """
    Semi-asynchronous FedAsync: buffers the k fastest-to-arrive clients per
    global model update, instead of updating on every single arrival.

    The simulator keeps `window_size` clients training concurrently at all
    times. With plain FedAsync (buffer_size=1) every single arrival
    immediately triggers a global model update. Here, buffer_size=k: the
    server waits for k arrivals — necessarily the k clients that happen to
    finish first among the window_size in flight, since arrivals are
    processed earliest-first — aggregates their k updates together with
    ONE mixing step (aggregate_buffered), and immediately re-dispatches k
    replacements. The remaining (window_size - k) clients are left
    completely undisturbed, continuing the local training they already
    started.

    "Fastest k" therefore isn't a static ranking heuristic — it falls out
    directly from the discrete-event simulation's real completion order,
    which is both simpler and more faithful than estimating client speed
    from profile data ahead of time. This also avoids a selection-bias trap:
    a fixed profile-based ranking would keep re-dispatching the same one or
    two objectively-fastest clients forever, starving the rest of the
    federation of representation. Random dispatch (inherited from FedAsync)
    keeps who's in the window varied, while buffering picks up whichever of
    them happen to finish first each cycle.

    k=1 is exactly plain FedAsync (fully async). k=window_size means the
    server always waits for the entire window before updating — equivalent
    to synchronous round-based FedAvg batching. Intermediate k values trade
    off staleness/variance (favors smaller k) against per-update robustness
    from averaging multiple clients (favors larger k) — this is the
    synchronous/asynchronous trade-off the paper's mixing hyperparameter α
    targets from a different angle.

    Aggregation rule (once k arrivals are buffered):
        x_avg   = sum_i (n_i / sum_j n_j) * x_new_i     — sample-weighted
                  average of the k arriving models
        alpha_t = base_alpha * s(max_i staleness_i)     — driven by the
                  WORST staleness in the batch (conservative: one very
                  stale update in the batch shouldn't be hidden by k-1
                  fresh ones)
        x_t     = (1 - alpha_t) * x_{t-1} + alpha_t * x_avg

    Args:
        k (int): buffer size — number of client updates aggregated together
            per global model update. Must satisfy 1 <= k <= window_size
            (async_fl.window_size in config); validated by AsyncSimulator
            at construction time. Default 5.
        Remaining args (alpha, staleness_func, a, b, rho) are the same as FedAsync.

    Example:
        # Buffer the fastest 5 of 10 concurrently-training clients per update.
        FedAsyncTopKFastTotal(alpha=0.1, k=5)   # needs async_fl.window_size >= 5
    """

    def __init__(self, k: int = 5, **kwargs):
        super().__init__(**kwargs)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.buffer_size = k

    def aggregate_buffered(
        self,
        global_model,
        updates: list,
        global_epoch: int,
        stalenesses: list,
        alpha_t: float,
    ) -> OrderedDict:
        """
        x_avg = sample-weighted average of the k arriving models
        x_t   = (1 - alpha_t) * x_{t-1} + alpha_t * x_avg

        alpha_t is passed in pre-computed by the simulator, from
        mixing_weight(base_alpha, max(stalenesses)).
        """
        current       = global_model.state_dict()
        total_samples = sum(u.num_samples for u in updates)
        new_state     = OrderedDict()
        for key in current:
            x_avg = sum(
                (u.num_samples / total_samples) * u.state_dict[key].float()
                for u in updates
            )
            new_state[key] = (1.0 - alpha_t) * current[key].float() + alpha_t * x_avg
        return new_state


# ---------------------------------------------------------------------------
# Paper replication: simulated-staleness variant (Xie et al. 2019 §5.2)
# ---------------------------------------------------------------------------

class FedAsyncSimulatedStaleness(FedAsync):
    """
    FedAsync where staleness is sampled from Uniform{0, ..., max_staleness}
    instead of being derived from real client timing differences.

    This replicates the evaluation setup of Xie et al. (2019) §5.2:
        "We simulate the asynchrony by randomly sampling the staleness (t−τ)
        from a uniform distribution."

    How it works
    ------------
    The simulator always calls mixing_weight(base_alpha, real_staleness) first,
    then logs alpha_t, then calls aggregate_async(..., alpha_t).

    Here, mixing_weight() IGNORES the real timing-based staleness and samples
    a new one from Uniform{0, ..., max_staleness}. The sampled value is stored
    so aggregate_async can use it; the logged alpha_used correctly reflects the
    sampled staleness (not the real one).

    This means:
      - The staleness column in the CSV = real timing staleness (always ~0 with
        window_size=1, or small with larger windows).
      - alpha_used column = alpha * s(sampled_staleness) — this is what the paper
        studies and what drives convergence.

    Args:
        max_staleness (int): upper bound K. Paper uses 4 (low) or 16 (high).
        seed (int):          seed for the staleness sampling RNG — independent
                             from the main simulator RNG so results are
                             reproducible regardless of other randomness.
        alpha, staleness_func, a, b: same as FedAsync.

    Examples (matching the paper's Figure 2):
        FedAsyncSimulatedStaleness(max_staleness=4,  seed=0, alpha=0.6, staleness_func="constant")
        FedAsyncSimulatedStaleness(max_staleness=16, seed=0, alpha=0.9, staleness_func="polynomial", a=0.5)
        FedAsyncSimulatedStaleness(max_staleness=4,  seed=0, alpha=0.6, staleness_func="hinge", a=10, b=4)
    """

    def __init__(self, max_staleness: int, seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        if max_staleness < 0:
            raise ValueError(f"max_staleness must be >= 0, got {max_staleness}")
        self.max_staleness    = max_staleness
        self._staleness_rng   = np.random.RandomState(seed)
        self._sampled_staleness = 0   # set by mixing_weight, read by aggregate_async

    def mixing_weight(self, base_alpha: float, staleness: int) -> float:
        """
        Sample k ~ Uniform{0, ..., max_staleness}, return base_alpha * s(k).

        The real timing-based staleness argument is intentionally ignored.
        The sampled k is cached so aggregate_async can access it for record-keeping.
        """
        self._sampled_staleness = int(
            self._staleness_rng.randint(0, self.max_staleness + 1)
        )
        return base_alpha * self._s(self._sampled_staleness)

    def aggregate_async(
        self, global_model, update, global_epoch, staleness, alpha_t
    ) -> OrderedDict:
        """
        Standard FedAsync mixing.  alpha_t was already computed from the sampled
        staleness in mixing_weight() — just use it directly.
        """
        return super().aggregate_async(
            global_model, update, global_epoch,
            self._sampled_staleness,   # pass sampled value (not real timing staleness)
            alpha_t,
        )
