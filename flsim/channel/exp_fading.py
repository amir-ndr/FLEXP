"""
channel/exp_fading.py: Exponential small-scale fading channel model.

Implements the channel model used in many FL-over-wireless papers:

    h_{n,t} = h_0 · ρ_{n,t} · d_n^{-2}

where:
    h_0          — path loss constant (absorbs frequency & geometry factors)
    ρ_{n,t}      — small-scale fading power gain, ρ ~ Exp(1), RE-DRAWN each round
    d_n          — distance from client n to edge server (metres)
    exponent = 2 — free-space path loss (not 3GPP's 37.6·log10(d_km))

Key difference from PathLossChannelModel:
  ρ is drawn fresh every time channel_gain() is called (i.e., every round).
  This is correct — the channel changes between rounds due to mobility/scattering.
  PathLossChannelModel uses frozen log-normal shadowing, which is a different
  physical assumption (quasi-static large-scale fading only).
"""

import math
import numpy as np

from flsim.interfaces.channel_model import ChannelModel


class ExpFadingChannelModel(ChannelModel):
    """
    Free-space path loss (exponent=2) with per-round exponential small-scale fading.

    Channel gain: g_{n,t} = h_0 · ρ_{n,t} · d_n^{-2},  ρ_{n,t} ~ Exp(1)

    The Exp(1) draw happens inside channel_gain() so ρ is fresh each round —
    do NOT cache the return value of channel_gain() across rounds.

    This class does NOT:
    - Store per-round fading state (stateless between rounds).
    - Compute time or energy.
    - Make scheduling decisions.
    """

    def __init__(
        self,
        h0: float,
        total_bandwidth_hz: float,
        noise_psd_w_per_hz: float,
        min_distance_m: float = 1.0,
    ):
        """
        Args:
            h0 (float): path loss constant (dimensionless).
                        Choose so that SNR is reasonable at cell edge.
                        Example: h0 = 1e-6 gives ~5–15 dB SNR at 100–500m
                        with p=10 dBm and B=2 MHz.
            total_bandwidth_hz (float): total system bandwidth B in Hz.
            noise_psd_w_per_hz (float): thermal noise PSD N0 in W/Hz.
            min_distance_m (float): minimum distance clamp (avoids g→∞ at d=0).
        """
        self.h0 = h0
        self.total_bandwidth_hz = total_bandwidth_hz
        self.noise_psd_w_per_hz = noise_psd_w_per_hz
        self.min_distance_m = min_distance_m

    # ------------------------------------------------------------------
    # ChannelModel interface
    # ------------------------------------------------------------------

    def channel_gain(self, profile, rng) -> float:
        """
        Compute linear channel power gain for one client in one round.

        g = h0 · ρ · d^{-2},   ρ ~ Exp(1)

        ρ is drawn fresh here every call — this is intentional.
        Each round has an independent fading realisation.

        Args:
            profile: ClientSystemProfile with distance_m.
            rng: numpy RandomState — must be passed (not None) for ρ draw.

        Returns:
            float: linear channel gain g (dimensionless, > 0).
        """
        d = max(profile.distance_m, self.min_distance_m)
        rho = rng.exponential(scale=1.0)   # ρ ~ Exp(1), fresh each round
        g = self.h0 * rho / (d ** 2)
        return g

    def achievable_rate_bps(
        self,
        bandwidth_hz: float,
        tx_power_w: float,
        channel_gain: float,
        noise_psd_w_per_hz: float,
    ) -> float:
        """
        Shannon capacity: r = B · log2(1 + g·p / (N0·B)).

        Args:
            bandwidth_hz (float): allocated bandwidth B in Hz.
            tx_power_w (float): transmit power p in watts.
            channel_gain (float): linear channel gain g.
            noise_psd_w_per_hz (float): noise PSD N0 in W/Hz.

        Returns:
            float: achievable uplink rate in bits per second.
        """
        snr = (channel_gain * tx_power_w) / (noise_psd_w_per_hz * bandwidth_hz)
        return bandwidth_hz * math.log2(1.0 + snr)

