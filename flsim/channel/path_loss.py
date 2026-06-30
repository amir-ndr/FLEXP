"""
channel/path_loss.py: 3GPP Urban Macro path-loss channel model.

Implements the ChannelModel interface using the standard 3GPP path-loss formula
(PL = 128.1 + 37.6 * log10(d_km)) plus log-normal shadow fading drawn once
per client at profile creation time (stored in profile.shadowing_db).

Bandwidth allocation has moved to ResourceAllocator (allocators/equal_split.py).

Reference: 3GPP TR 36.814, Table A.2.1.1-2.
"""

import math
import numpy as np

from flsim.interfaces.channel_model import ChannelModel
from flsim.channel.conversions import db_to_linear

# 3GPP Urban Macro path-loss constants (dB)
_PATH_LOSS_INTERCEPT_DB = 128.1   # A in PL = A + B * log10(d_km)
_PATH_LOSS_EXPONENT_DB  = 37.6    # B


class PathLossChannelModel(ChannelModel):
    """
    3GPP Urban Macro path-loss + log-normal shadowing channel model.

    Channel gain g_k = 10^(-(PL_k + X_k) / 10) where:
        PL_k = 128.1 + 37.6 * log10(d_k_km)   [dB]
        X_k  = profile.shadowing_db             [dB, drawn at profile creation]

    This class does NOT:
    - Perform bandwidth allocation (that belongs to ResourceAllocator).
    - Compute time or energy (those belong to TimeModel / EnergyModel).
    - Draw fresh shadowing each call — shadowing is frozen in the profile.
    """

    def __init__(
        self,
        total_bandwidth_hz: float,
        noise_psd_w_per_hz: float,
        min_distance_m: float = 1.0,
    ):
        """
        Args:
            total_bandwidth_hz (float): total system bandwidth B in Hz.
            noise_psd_w_per_hz (float): thermal noise PSD N0 in W/Hz.
            min_distance_m (float): minimum distance clamp to avoid log singularity at d=0.
        """
        self.total_bandwidth_hz = total_bandwidth_hz
        self.noise_psd_w_per_hz = noise_psd_w_per_hz
        self.min_distance_m = min_distance_m

    # ------------------------------------------------------------------
    # ChannelModel interface
    # ------------------------------------------------------------------

    def channel_gain(self, profile, rng) -> float:
        """
        Compute linear channel power gain g_k for one client.

        Uses frozen shadowing stored in profile.shadowing_db (drawn once at
        profile creation — not re-drawn here, ensuring reproducibility per round).

        Args:
            profile: ClientSystemProfile with distance_m and shadowing_db.
            rng: unused here (shadowing is frozen); kept for interface compliance.

        Returns:
            float: linear channel gain g_k in (0, 1).
        """
        d_m = max(profile.distance_m, self.min_distance_m)
        d_km = d_m / 1000.0

        # 3GPP path loss in dB
        pl_db = _PATH_LOSS_INTERCEPT_DB + _PATH_LOSS_EXPONENT_DB * math.log10(d_km)

        # Total channel loss including shadow fading (frozen in profile)
        loss_db = pl_db + profile.shadowing_db

        # Convert loss to linear gain: g = 10^(-L_dB / 10)
        g_k = 10.0 ** (-loss_db / 10.0)
        return g_k

    def achievable_rate_bps(
        self,
        bandwidth_hz: float,
        tx_power_w: float,
        channel_gain: float,
        noise_psd_w_per_hz: float,
    ) -> float:
        """
        Shannon capacity: r = B * log2(1 + g * p / (N0 * B)).

        Args:
            bandwidth_hz (float): allocated bandwidth B in Hz.
            tx_power_w (float): transmit power p in watts.
            channel_gain (float): linear channel gain g (dimensionless).
            noise_psd_w_per_hz (float): noise PSD N0 in W/Hz.

        Returns:
            float: achievable uplink rate in bits per second (bps).
        """
        noise_power_w = noise_psd_w_per_hz * bandwidth_hz
        snr = (channel_gain * tx_power_w) / noise_power_w
        rate_bps = bandwidth_hz * math.log2(1.0 + snr)
        return rate_bps

