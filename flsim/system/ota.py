"""
system/ota.py: Over-the-air computation (AirComp) aggregation for federated
learning, per Yang, Jiang, Shi & Ding, "Federated Learning via Over-the-Air
Computation" (arXiv:1812.11750).

Physical model (paper §II-B, single-antenna base station, N=1)
-----------------------------------------------------------------
Instead of each client uploading its model over an orthogonal slot (FDMA, as
used by the rest of this simulator) and the server digitally averaging them,
OTA computation has every selected client transmit SIMULTANEOUSLY over the
same channel. The wireless channel's own signal superposition physically
computes the weighted sum — the server only has to apply one scalar
post-processing factor. Communication cost no longer scales with the number
of participating clients, at the cost of channel-induced aggregation noise.

Scope: single-antenna base station (N=1)
------------------------------------------
The paper's main contribution (§III-V) is a device-selection + RECEIVER
BEAMFORMING co-design for a multi-antenna (N>1) base station — solved via a
difference-of-convex-functions (DC) program over sparse and low-rank matrix
variables, since with N>1 antennas the receive beamforming vector m couples
every device's feasibility together, making joint device selection genuinely
combinatorial (NP-hard in general).

This module implements the N=1 case (matching this framework's existing
ChannelModel, which only models a scalar real power gain per client — see
flsim/interfaces/channel_model.py — not a channel vector). With N=1 there is
no beamforming design freedom (m is a scalar, its value cancels out of every
ratio in the MSE formula), so the device-selection problem (paper eq. 11)
collapses from a coupled combinatorial program to an INDEPENDENT per-device
feasibility test — provably, not approximately: with m fixed, feasibility of
device i depends only on i's own (phi_i, channel_gain_i), never on which
other devices are selected. The paper itself notes this simpler "truncation"
style of device selection as a known simpler alternative to its N>1 DC
algorithm (see the discussion after eq. 11 in the paper, citing [24]). The
full N>1 DC/SDP machinery (Ky Fan k-norms, matrix lifting, semidefinite
subproblems) is NOT implemented here — it would need a general SDP solver as
a new dependency and is a separate, much larger undertaking than an
aggregation primitive.

Real- vs complex-valued channel
---------------------------------
The paper's signals are complex baseband (s_i, h_i, b_i, m in C). This
framework's ChannelModel.channel_gain() only returns a real linear POWER
GAIN g_i = |h_i|^2 (no phase). Consistently with that existing convention,
this module treats the channel coefficient as the real, non-negative
h_i = sqrt(g_i) (i.e. zero-phase / coherent channel), which makes the
zero-forcing transmit scalar b_i real too. This is the standard reduction
used throughout the OTA-FL literature for tractability and matches every
other channel-related computation already in this codebase (which is also
real-power-gain-only).

Key equations implemented (paper §II-B, specialized to N=1, m=1)
--------------------------------------------------------------------
Zero-forcing transmit scalar (Proposition 1, eq. 8):
    b_i = sqrt(eta) * phi_i / h_i                      , h_i = sqrt(g_i)

Power-normalizing factor (eq. 9), set by the worst selected channel:
    eta = min_{i in S} P0 * g_i / phi_i^2

Aggregation MSE (eq. 10):
    MSE(S) = sigma^2 / eta = sigma^2 * max_{i in S} (phi_i^2 / (P0 * g_i))

Device-selection feasibility (eq. 11, specialized to N=1 — see docstring of
select_feasible_devices for the independence argument):
    device i is includable  <=>  phi_i^2 / g_i <= gamma * P0 / sigma^2

Post-processed aggregate (eq. 2, 6): with zero-forcing, h_i*b_i/sqrt(eta) =
phi_i exactly, so the received signal (before post-processing) is the exact
weighted sum plus a single shared noise term:
    ghat = sum_i phi_i * s_i + n/sqrt(eta) ,   n ~ N(0, sigma^2)
    zhat = ghat / sum_i phi_i
         = (sum_i phi_i * s_i) / (sum_i phi_i)  +  noise
         = [ordinary FedAvg weighted average]   +  noise
    where noise ~ N(0, MSE(S) / (sum_i phi_i)^2), applied i.i.d. per scalar
    model dimension.

This is the practically important takeaway: OTA aggregation IS FedAvg's
weighted average, plus zero-mean Gaussian noise whose variance is set by the
achieved MSE(S) — everything below exists to compute that one MSE value (and
the per-device transmit energy that produced it) correctly.

Transmission energy
--------------------
Per the transmit power constraint (paper eq. 5), device i's per-symbol
transmit energy is the squared norm of its transmitted signal:
    E_i(one symbol) = |b_i * s_i|^2 = |b_i|^2          (since Var(s_i) = 1)
For a full model vector of d scalar dimensions, transmitted over d channel
uses (one per parameter, as in the paper's "time slot j in {1,...,d}"):
    E_i(one round) = d * |b_i|^2
By construction of eta (eq. 9), |b_i|^2 <= P0 for every selected device,
with equality for whichever device was the limiting (worst-channel) one.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class OTASelectionResult:
    """Result of one feasibility-based device-selection pass."""
    selected_ids: list          # client ids satisfying the MSE threshold
    excluded_ids: list          # client ids that would violate it
    mse_threshold_linear: float  # gamma used (linear scale)


class OTAChannel:
    """
    Reusable over-the-air aggregation primitive (single-antenna BS, N=1).

    Owns the two OTA-specific physical parameters that are NOT part of the
    FDMA wireless config elsewhere in this framework (P0, the per-device
    peak transmit power, and sigma^2, the receiver noise power per channel
    use) — these are conceptually different from `tx_power_dbm` /
    `noise_psd_dbm_per_hz` in wireless config, which are power-spectral
    quantities for a bandwidth-limited Shannon-capacity link, not a
    per-symbol AirComp link.

    Stateless otherwise — every method takes its inputs explicitly, so this
    class can be constructed once and reused across rounds/algorithms, or
    instantiated fresh anywhere a custom algorithm wants OTA aggregation.

    Args:
        p0_w (float):           per-device peak transmit power P0, in watts.
        noise_power_w (float):  receiver noise power sigma^2, in watts,
                                 for ONE scalar channel use (not a PSD).

    Example — building block for a custom algorithm:
        from flsim.system.ota import OTAChannel

        class MyOTAAlgorithm(FederatedAlgorithm):
            def __init__(self):
                self.ota = OTAChannel(p0_w=0.1, noise_power_w=1e-10)

            def select_clients(self, all_clients, num_to_select, rng, **kwargs):
                channel_model = kwargs["channel_model"]
                candidates = [
                    (c.client_id, c.num_samples,
                     channel_model.channel_gain(c.profile, rng))
                    for c in all_clients
                ]
                result = self.ota.select_feasible_devices(candidates, gamma=10.0)
                return [c for c in all_clients if c.client_id in result.selected_ids]

            def aggregate(self, global_model, client_updates):
                state_dicts = [u.state_dict for u in client_updates]
                phis   = [u.num_samples   for u in client_updates]
                gains  = [u.channel_gain  for u in client_updates]
                agg_state, mse = self.ota.aggregate_state_dicts(state_dicts, phis, gains)
                self.last_mse = mse   # stash for logging if you want it
                return agg_state
    """

    def __init__(self, p0_w: float, noise_power_w: float):
        if p0_w <= 0.0:
            raise ValueError(f"p0_w must be > 0, got {p0_w}")
        if noise_power_w <= 0.0:
            raise ValueError(f"noise_power_w must be > 0, got {noise_power_w}")
        self.p0_w = p0_w
        self.noise_power_w = noise_power_w

    # ------------------------------------------------------------------
    # Core physical-layer quantities (paper eq. 8-10, N=1 specialization)
    # ------------------------------------------------------------------

    def compute_eta(self, phis: list, gains: list) -> float:
        """
        Power-normalizing factor eta (paper eq. 9, m=1):
            eta = min_{i in S} P0 * g_i / phi_i^2

        Set by whichever selected device has the worst "effective channel"
        g_i/phi_i^2 — that device transmits at exactly P0; every other
        device backs off to match it (zero-forcing power control).

        Args:
            phis:  pre-processing weights phi_i (num_samples) for each
                   selected device, same order as `gains`.
            gains: channel power gains g_i = |h_i|^2 for each selected
                   device (from ChannelModel.channel_gain()).

        Returns:
            float: eta > 0.

        Raises:
            ValueError: if phis/gains are empty or mismatched in length, or
                any gain is <= 0 (a zero/negative gain makes eta collapse to
                0, i.e. infinite required power — not physically meaningful).
        """
        self._validate_phi_gain(phis, gains)
        ratios = [self.p0_w * g / (phi ** 2) for phi, g in zip(phis, gains)]
        return min(ratios)

    def zero_forcing_scalar(self, phi_i: float, gain_i: float, eta: float) -> float:
        """
        Zero-forcing transmit scalar b_i (paper Proposition 1, eq. 8, real/N=1):
            b_i = sqrt(eta) * phi_i / sqrt(gain_i)

        Args:
            phi_i:  pre-processing weight (num_samples) for this device.
            gain_i: this device's channel power gain g_i.
            eta:    power-normalizing factor (from compute_eta()).

        Returns:
            float: real transmit scalar b_i. |b_i|^2 <= p0_w by construction
            when eta was computed from a set that includes this device.
        """
        if gain_i <= 0.0:
            raise ValueError(f"gain_i must be > 0, got {gain_i}")
        return (eta ** 0.5) * phi_i / (gain_i ** 0.5)

    def mse(self, phis: list, gains: list) -> float:
        """
        Aggregation MSE achieved by selecting exactly this set of devices
        (paper eq. 10, N=1):
            MSE(S) = sigma^2 / eta = sigma^2 * max_i (phi_i^2 / (P0 * g_i))

        Args:
            phis, gains: see compute_eta().

        Returns:
            float: MSE(S) >= 0. Smaller is better (less aggregation noise).
        """
        eta = self.compute_eta(phis, gains)
        return self.noise_power_w / eta

    def transmission_energy_j(self, phi_i: float, gain_i: float, eta: float,
                               num_symbols: int) -> float:
        """
        Transmission energy for one device over one round, in joules.

        Per the paper's power constraint (eq. 5), one symbol's transmit
        energy is the squared norm of the transmitted signal:
            E(one symbol) = |b_i|^2
        A full model vector needs num_symbols channel uses (one per scalar
        model parameter, matching the paper's "time slot j in {1,...,d}"):
            E(round) = num_symbols * |b_i|^2

        Args:
            phi_i, gain_i, eta: see zero_forcing_scalar().
            num_symbols (int): number of scalar model dimensions transmitted
                (e.g. total element count of the model's state_dict).

        Returns:
            float: transmission energy in joules for this device this round.
        """
        if num_symbols < 0:
            raise ValueError(f"num_symbols must be >= 0, got {num_symbols}")
        b_i = self.zero_forcing_scalar(phi_i, gain_i, eta)
        return num_symbols * (b_i ** 2)

    # ------------------------------------------------------------------
    # Device selection (paper eq. 11, N=1 specialization)
    # ------------------------------------------------------------------

    def select_feasible_devices(self, candidates: list, gamma: float) -> OTASelectionResult:
        """
        Maximize the number of selected devices subject to an MSE
        requirement (paper eq. 11): find the largest S with MSE(S) <= gamma.

        With N=1 (no receive-beamforming degrees of freedom — see module
        docstring), MSE(S) = sigma^2 * max_{i in S} (phi_i^2/(P0*g_i)) is a
        MAX over the selected set of a per-device quantity. A set's MSE is
        therefore <= gamma if and only if EVERY member individually
        satisfies phi_i^2/(P0*g_i) <= gamma/sigma^2 — whether device i is
        includable never depends on which other devices are also selected.
        So the (paper eq. 11) combinatorial maximization — which is genuinely
        NP-hard for N>1 because m couples all devices' feasibility together —
        has a trivial, exactly-optimal closed form here: include every
        device that individually clears the threshold. This is the same
        "channel truncation" idea the paper itself contrasts its N>1 DC
        algorithm against (discussion following eq. 11).

        Args:
            candidates: list of (client_id, phi_i, gain_i) tuples — the full
                candidate pool to evaluate (typically ALL available clients,
                matching the paper's framing of maximizing |S| out of the
                whole device population, not a pre-capped subset).
            gamma (float): target MSE threshold (linear scale, i.e. already
                converted from dB if needed — see flsim.channel.conversions.db_to_linear).

        Returns:
            OTASelectionResult with selected_ids sorted by ascending
            phi_i^2/gain_i (best effective channel first) and excluded_ids
            likewise. selected_ids may be empty if gamma is infeasible for
            every candidate — callers should treat that as an error
            (see FedOTA.select_clients for an example that raises).
        """
        if not candidates:
            raise ValueError("candidates must be non-empty")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be > 0, got {gamma}")

        threshold = gamma * self.p0_w / self.noise_power_w
        scored = []
        for client_id, phi_i, gain_i in candidates:
            if gain_i <= 0.0:
                raise ValueError(f"gain for client {client_id} must be > 0, got {gain_i}")
            scored.append((client_id, (phi_i ** 2) / gain_i))

        scored.sort(key=lambda t: t[1])
        selected = [cid for cid, score in scored if score <= threshold]
        excluded = [cid for cid, score in scored if score > threshold]
        return OTASelectionResult(
            selected_ids=selected, excluded_ids=excluded, mse_threshold_linear=gamma
        )

    # ------------------------------------------------------------------
    # Aggregation (paper eq. 2, 6 — the practical FedAvg-plus-noise result)
    # ------------------------------------------------------------------

    def aggregate_state_dicts(
        self,
        state_dicts: list,
        phis: list,
        gains: list,
        rng: Optional[np.random.RandomState] = None,
    ) -> tuple:
        """
        Combine selected clients' model updates via simulated over-the-air
        superposition: the exact sample-weighted FedAvg average, plus
        zero-mean Gaussian noise whose variance matches the physically
        achieved aggregation MSE (see module docstring's derivation).

        Args:
            state_dicts: list[OrderedDict] — one per selected client, same
                order as phis/gains.
            phis:  pre-processing weights phi_i (num_samples) per client.
            gains: channel power gains g_i per client.
            rng:   numpy RandomState for the injected noise. If None, a
                   fresh unseeded RandomState is used (non-reproducible) —
                   pass one explicitly for reproducible experiments.

        Returns:
            (agg_state, mse): OrderedDict of the noisy aggregated
            state_dict (detached float32 tensors), and the float MSE(S)
            that was actually injected (paper eq. 10) — useful for logging.
        """
        self._validate_phi_gain(phis, gains)
        if len(state_dicts) != len(phis):
            raise ValueError(
                f"state_dicts ({len(state_dicts)}) and phis ({len(phis)}) "
                f"must have the same length"
            )
        if rng is None:
            rng = np.random.RandomState()

        total_phi = float(sum(phis))
        mse = self.mse(phis, gains)
        # Post-processed noise variance: MSE(S) / (sum_i phi_i)^2 — see
        # module docstring's derivation of zhat = weighted_avg + noise.
        noise_std = (mse ** 0.5) / total_phi

        agg_state = OrderedDict()
        ref_state = state_dicts[0]
        for key, param in ref_state.items():
            agg_state[key] = torch.zeros_like(param, dtype=torch.float32)

        for state_dict, phi_i in zip(state_dicts, phis):
            weight = phi_i / total_phi
            for key, param in state_dict.items():
                agg_state[key] += weight * param.float().detach()

        for key, tensor in agg_state.items():
            noise = torch.from_numpy(
                rng.normal(loc=0.0, scale=noise_std, size=tuple(tensor.shape))
            ).to(dtype=tensor.dtype)
            agg_state[key] = tensor + noise

        return agg_state, mse

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_phi_gain(phis: list, gains: list) -> None:
        if not phis or not gains:
            raise ValueError("phis and gains must be non-empty")
        if len(phis) != len(gains):
            raise ValueError(
                f"phis ({len(phis)}) and gains ({len(gains)}) must have the same length"
            )
        for g in gains:
            if g <= 0.0:
                raise ValueError(f"all gains must be > 0, got {g}")
        for phi in phis:
            if phi <= 0.0:
                raise ValueError(f"all phis must be > 0, got {phi}")
