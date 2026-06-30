"""
interfaces/channel_model.py: Abstract base class for wireless channel models.

Separating ChannelModel from TimeModel allows independent swapping of:
- Path-loss formula (urban macro, rural, free-space, …)
- Achievable-rate formula (Shannon, realistic MCS table, …)

Bandwidth allocation policy lives in ResourceAllocator (interfaces/allocator.py),
keeping channel physics separate from resource optimization policy.
"""

from abc import ABC, abstractmethod


class ChannelModel(ABC):
    """
    Base class for wireless channel models.

    Computes channel gain and achievable rate for each client.

    This class does NOT:
    - Compute training or propagation time (that belongs to TimeModel).
    - Allocate compute resources.
    - Hold any per-round mutable state.
    """

    @abstractmethod
    def channel_gain(self, profile, rng) -> float:
        """
        Returns the linear channel power gain g_k for one client.

        Combines deterministic path loss with stochastic shadow fading.
        The shadowing component is drawn once at profile creation and stored
        in profile.shadowing_db, so this method is deterministic given a profile.

        Args:
            profile: ClientSystemProfile (contains distance_m, shadowing_db).
            rng: numpy RandomState (used only if the model draws fresh fading each call).

        Returns:
            float: linear channel power gain g_k  (dimensionless, 0 < g_k < 1).
        """

    @abstractmethod
    def achievable_rate_bps(
        self,
        bandwidth_hz: float,
        tx_power_w: float,
        channel_gain: float,
        noise_psd_w_per_hz: float,
    ) -> float:
        """
        Returns achievable uplink rate in bits per second (Shannon capacity).

        Formula: r = B * log2(1 + g * p / (N0 * B))

        Args:
            bandwidth_hz (float): allocated bandwidth B in Hz.
            tx_power_w (float): transmit power p in watts.
            channel_gain (float): linear channel gain g (dimensionless).
            noise_psd_w_per_hz (float): noise power spectral density N0 in W/Hz.

        Returns:
            float: achievable uplink rate in bits per second.
        """

