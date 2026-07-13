"""
algorithms/fedasync.py: FedAsync — Asynchronous Federated Optimization (Xie et al., 2019).

Global update rule:
    x_t = (1 - alpha_t) * x_{t-1} + alpha_t * x_new
    alpha_t = base_alpha * s(t - tau)

Three built-in staleness strategies (paper §5.2):
  "constant"   : s(k) = 1                              → FedAsync+Const
  "polynomial" : s(k) = (k + 1)^{-a}                  → FedAsync+Poly
  "hinge"      : s(k) = 1           if k <= b          → FedAsync+Hinge
                         1 / (a*(k-b)+1)  otherwise

Three built-in client selection policies:
  FedAsync              — uniform random (default)
  FedAsyncTopKFastTotal — always pick the K fastest clients by total
                          (compute + upload) time; falls back to compute-only
                          when no channel/upload context is available
  FedAsyncFixedFast     — pick randomly from a fixed pool of the M fastest clients

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

    Examples:
        FedAsync(alpha=0.1)
        FedAsync(alpha=0.5, staleness_func="polynomial", a=0.5)
        FedAsync(alpha=0.5, staleness_func="hinge", a=10.0, b=4.0)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        staleness_func: str = "constant",
        a: float = 1.0,
        b: float = 4.0,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if staleness_func not in ("constant", "polynomial", "hinge"):
            raise ValueError(
                f"staleness_func must be 'constant', 'polynomial', or 'hinge', "
                f"got '{staleness_func}'"
            )
        self.alpha         = alpha
        self.staleness_func = staleness_func
        self.a             = a
        self.b             = b

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
# Client selection variants
# ---------------------------------------------------------------------------

class FedAsyncTopKFastTotal(FedAsync):
    """
    FedAsync that always dispatches the K clients with the smallest TOTAL time.

    Total time estimate:
        t_total = t_comp + t_upload
        t_comp  = cycles_per_sample × num_samples / cpu_frequency_hz
        t_up    = upload_bits / (B × log2(1 + SNR))   (Shannon rate at full power)

    Ranking by total time (rather than compute alone) correctly deprioritises a
    client that computes fast but has a weak channel — its slow upload makes it
    slow end-to-end.  Download time is not included (server broadcast is assumed
    cheap; see wireless.downlink_negligible).

    Channel/upload context is supplied automatically by the simulator via kwargs
    (channel_model, noise_psd_w_per_hz, bw_per_client_hz, upload_size_bits), so
    no manual wiring is needed:

        FedAsyncTopKFastTotal(alpha=0.1)          # uses simulator's upload size
        FedAsyncTopKFastTotal(alpha=0.1, upload_size_bits=1e6)   # override

    If the channel context is missing (e.g. used outside the simulator), it
    falls back to ranking by compute time only.

    With num_to_trigger=1 this selects the single fastest client each dispatch;
    set async_fl.window_size in config to control the concurrency level.

    Args:
        upload_size_bits (float): override the model size in bits used for the
            upload estimate. If 0 (default), the simulator-provided value is used.
        Remaining args are the same as FedAsync.
    """

    def __init__(self, upload_size_bits: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self._upload_bits = upload_size_bits

    def select_clients(self, all_clients, num_to_trigger, rng, **kwargs):
        """Sort by estimated total time (compute + upload), return K fastest."""
        channel_model = kwargs.get("channel_model", None)
        noise_psd     = kwargs.get("noise_psd_w_per_hz", None)
        bw_per_client = kwargs.get("bw_per_client_hz", None)
        # Prefer an explicit constructor override; otherwise use the simulator's.
        upload_bits   = self._upload_bits or kwargs.get("upload_size_bits", 0.0)

        def _t_total_estimate(client):
            p = client.profile
            t_comp = p.cycles_per_sample * client.num_samples / p.cpu_frequency_hz
            if (channel_model is None or noise_psd is None
                    or bw_per_client is None or not upload_bits):
                return t_comp  # fallback: compute only
            # Estimate upload rate. Pass the selection rng so channel models that
            # need it (exp_fading draws ρ~Exp(1)) work; path_loss ignores it.
            gain = channel_model.channel_gain(p, rng)
            rate = channel_model.achievable_rate_bps(
                bandwidth_hz=bw_per_client,
                tx_power_w=p.tx_power_w,
                channel_gain=gain,
                noise_psd_w_per_hz=noise_psd,
            )
            t_up = upload_bits / max(rate, 1.0)
            return t_comp + t_up

        k = min(num_to_trigger, len(all_clients))
        return sorted(all_clients, key=_t_total_estimate)[:k]


class FedAsyncFixedFast(FedAsync):
    """
    FedAsync that samples uniformly from a fixed pool of the M fastest clients.

    The fast pool is computed once on the first call and cached. This gives
    variety within a fast subset while still excluding slow clients entirely.

    Args:
        pool_size (int): size of the fast client pool. Default 10.
        Same as FedAsync for remaining args.

    Example:
        # Keep a pool of 20 fastest clients; each dispatch picks 5 of them.
        FedAsyncFixedFast(pool_size=20, alpha=0.1)
        # Then set async_fl.window_size: 5 in the config.
    """

    def __init__(self, pool_size: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.pool_size   = pool_size
        self._fast_pool  = None   # lazily built on first call

    def select_clients(self, all_clients, num_to_trigger, rng, **kwargs):
        """Sample without replacement from the top-pool_size fastest clients."""
        if self._fast_pool is None:
            def _t_estimate(client):
                p = client.profile
                return p.cycles_per_sample * client.num_samples / p.cpu_frequency_hz
            sorted_all = sorted(all_clients, key=_t_estimate)
            self._fast_pool = sorted_all[:self.pool_size]

        k = min(num_to_trigger, len(self._fast_pool))
        indices = rng.choice(len(self._fast_pool), size=k, replace=False)
        return [self._fast_pool[i] for i in indices]


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
