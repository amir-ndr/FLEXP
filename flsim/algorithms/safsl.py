"""
algorithms/safsl.py: SAFSL — Semi-Asynchronous Federated Split Learning over
wireless networks.

Implements the paper's aggregation rule (eq. 4) exactly:
    w_{t+1} = sum_{n in S_t} m_{n,t} * rho_{n,t} * w^H_{n,t}
    rho_{n,t} = |D_n|^gamma * alpha_{n,t}  /  sum_{k in S_t} |D_k|^gamma * alpha_{k,t}
where alpha_{n,t} = t - r_{n,t} is the device's participation interval (the
same quantity flsim.algorithms.fedasync.FedAsync calls "staleness"), and S_t
is the set of devices the server buffers before this aggregation — see
flsim.core.split_async_simulator.SplitAsyncSimulator, which supplies
`buffer_size` devices per aggregation (|S_t| = buffer_size).

IMPORTANT — no blend with the previous global model. Unlike FedAsync's
x_t = (1-alpha_t)*x_{t-1} + alpha_t*x_new, eq. (4) has NO w_t term on the
right-hand side at all: it is a pure weighted average of the CURRENTLY
ARRIVING batch. With buffer_size=1 (the paper's own "fully asynchronous"
special case — see class docstring below), the single arriving device's own
trained (client-side, server-side) pair REPLACES the global pair outright.
This is intentional and matches the paper exactly (each device's starting
point w^0_{n,t} = w(r_{n,t}) already carries forward the recent global state,
so the combination step itself doesn't need to blend again) — override
aggregate_buffered() (inherited from SplitAsyncAlgorithm) if you want a
blended variant instead.

NOTE on alpha_{n,t} = 0. In a fresh simulation, the very first single-device
arrival can legitimately have staleness 0 (dispatched and returned with
nothing else having happened in between — same edge case
flsim.algorithms.fedasync.FedAsync's polynomial/hinge staleness functions
avoid by using (k+1) rather than k). Since alpha multiplies the weight
directly here (not through an exponent), a raw alpha=0 would zero out that
device's entire contribution — clearly not intended. This implementation
uses (staleness + 1) in its place, mirroring FedAsync's own (k+1) convention
in this codebase.
"""

from flsim.interfaces.split_async_algorithm import SplitAsyncAlgorithm


class SAFSL(SplitAsyncAlgorithm):
    """
    Semi-Asynchronous Federated Split Learning.

    Concurrency, via `buffer_size` (same convention as
    flsim.algorithms.fedasync.FedAsyncTopKFastTotal):
        k=1              -> fully asynchronous (paper's own limiting case,
                             "scenarios with extreme device heterogeneity")
        1 < k < window    -> semi-asynchronous (the paper's actual proposal)
        k=window_size    -> degenerates to synchronous batch aggregation

    Args:
        k (int): buffer size |S_t| — number of devices aggregated together
            per global-model update. Default 5.
        gamma (float): data-size exponent γ in rho_{n,t} (paper Section
            III-A). γ>1 favors devices with more data over staleness; γ<1
            favors fresher (less stale) devices; γ=1 (default) weighs data
            size and participation interval equally.

    Example:
        SAFSL(k=5, gamma=1.0)   # needs async_fl.window_size >= 5
    """

    def __init__(self, k: int = 5, gamma: float = 1.0):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        self.k = k
        self.buffer_size = k
        self.gamma = gamma

    def participation_weight(self, num_samples: int, staleness: int, **kwargs) -> float:
        """rho_{n,t}'s unnormalized numerator: |D_n|^gamma * (staleness + 1)."""
        return (float(num_samples) ** self.gamma) * (staleness + 1)

    # select_clients()      -> inherited (uniform random; the paper doesn't
    #                          specify a selection policy).
    # aggregate_buffered()  -> inherited from SplitAsyncAlgorithm: normalized
    #                          weighted average using participation_weight()
    #                          above — exactly eq. (4) with rho_{n,t} defined
    #                          as this method computes it.
