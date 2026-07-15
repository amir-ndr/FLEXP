"""
algorithms/fedota.py: FedOTA — Federated Averaging with over-the-air (AirComp)
aggregation. Yang, Jiang, Shi & Ding, "Federated Learning via Over-the-Air
Computation" (arXiv:1812.11750), Algorithm 1 (the paper's own FedAvg-style
training loop) combined with the paper's OTA aggregation instead of digital
per-client upload + averaging.

See flsim/system/ota.py for the underlying physics (OTAChannel) and its
scope note (single-antenna base station, N=1 — the paper's harder N>1
device-selection+beamforming DC/SDP algorithm is not implemented here).

FedOTA is a thin reference algorithm: it wires OTAChannel's two building
blocks (select_feasible_devices, aggregate_state_dicts) into the standard
FederatedAlgorithm interface, exactly as FedAvg/FedProx wire in their own
selection/aggregation rules. Write your own OTA-based algorithm the same way
— see the "Writing a custom OTA algorithm" section below.

Round structure implemented (paper §II-A "FedAvg" + §II-B "AirComp"):
  select_clients() : maximize the number of participating devices subject to
                      an MSE budget gamma (paper eq. 11, N=1 closed form)
  configure_client(): inherited no-op (plain local SGD, as in the paper)
  aggregate()       : weighted average via simulated AirComp superposition,
                      i.e. FedAvg's weighted average plus injected Gaussian
                      noise whose variance matches the achieved MSE(S)

Uplink physics (time + energy)
--------------------------------
FedOTA overrides the Simulator's default FDMA uplink model via the
recompute_uplink_physics() hook (Simulator._run_round calls it if present).
This rewrites each ClientUpdate's uplink fields to OTA physics BEFORE the
RoundResult / CSV is built, so the standard energy/time columns are correct
for OTA — no side-channel reading required:
  * upload_time_s: CONSTANT across all clients and rounds. In OTA every
    selected client transmits SIMULTANEOUSLY over the full bandwidth (signal
    superposition), so the uplink takes num_symbols / total_bandwidth_hz
    seconds regardless of how many devices participate — the communication
    win the paper advertises. (Contrast FDMA, where each client gets a
    bandwidth slice B/M and upload time grows with M.)
  * tx_energy_j: the SQUARED NORM of the transmitted signal (paper eq. 5),
    num_symbols * |b_i|^2, where b_i is the zero-forcing transmit scalar.
    Not p·t.
  * total_time_s / total_energy_j: recomputed from the above.

The round duration (max over clients of total_time_s) is therefore
max(compute) + the constant simultaneous-upload time.

FedOTA also accumulates OTA-specific quantities that have no column in the
standard CSV, for convenience when plotting/analyzing:
    self.mse_history      : list[float]           — MSE(S) achieved each round
    self.energy_history   : list[dict[int,float]] — {client_id: joules} each round
    self.excluded_history : list[list[int]]       — client ids infeasible each round

Writing a custom OTA algorithm
---------------------------------
Reuse OTAChannel directly rather than subclassing FedOTA if you want a
different selection rule with the same aggregation physics:

    from flsim.system.ota import OTAChannel
    from flsim.interfaces.algorithm import FederatedAlgorithm

    class MyOTAAlgorithm(FederatedAlgorithm):
        def __init__(self, p0_w, noise_power_w, mu):
            self.ota = OTAChannel(p0_w=p0_w, noise_power_w=noise_power_w)
            self.mu  = mu   # e.g. add a FedProx-style proximal term

        def configure_client(self, client, global_model, round_idx):
            client.proximal_mu = self.mu

        def select_clients(self, all_clients, num_to_select, rng, **kwargs):
            # your own selection rule — doesn't have to be MSE-threshold based
            ...

        def aggregate(self, global_model, client_updates):
            state_dicts = [u.state_dict  for u in client_updates]
            phis        = [u.num_samples for u in client_updates]
            gains       = [u.channel_gain for u in client_updates]
            agg_state, mse = self.ota.aggregate_state_dicts(state_dicts, phis, gains)
            return agg_state
"""

from collections import OrderedDict

import numpy as np

from flsim.channel.conversions import db_to_linear
from flsim.interfaces.algorithm import FederatedAlgorithm
from flsim.system.ota import OTAChannel


class FedOTA(FederatedAlgorithm):
    """
    FedAvg-style training with over-the-air aggregation (paper Algorithm 1 + §II-B).

    Args:
        p0_w (float):          per-device peak transmit power P0, in watts.
        noise_power_w (float): receiver noise power sigma^2 (per scalar
                                channel use), in watts.
        gamma_linear (float):  target aggregation MSE budget, linear scale.
                                Exactly one of gamma_linear / gamma_db must
                                be given.
        gamma_db (float):      target aggregation MSE budget, in dB
                                (converted via db_to_linear). The paper's
                                own CIFAR-10/SVM experiment (§VI-C) uses
                                gamma=5dB.
        seed (int):             seeds the aggregation-noise RNG, independent
                                of the simulator's main RNG (so results are
                                reproducible regardless of other randomness) —
                                same pattern as FedAsyncSimulatedStaleness.

    Choosing gamma: the achievable MSE for a given candidate depends on your
    deployment's phi_i (num_samples) and channel_gain() scale — there's no
    universal default. A quick way to calibrate: construct an OTAChannel
    with your intended p0_w/noise_power_w and call .mse(phis, gains) for the
    full candidate pool (all devices) to see the MSE if everyone were
    selected; pick gamma relative to that.

    Raises RuntimeError from select_clients() if gamma is infeasible for
    every candidate this round (empty selection would otherwise crash later
    aggregation) — loosen gamma, raise p0_w, or lower noise_power_w.

    Example:
        FedOTA(p0_w=0.1, noise_power_w=1e-10, gamma_db=5.0, seed=0)
    """

    def __init__(
        self,
        p0_w: float,
        noise_power_w: float,
        gamma_linear: float = None,
        gamma_db: float = None,
        seed: int = 0,
    ):
        if (gamma_linear is None) == (gamma_db is None):
            raise ValueError(
                "Provide exactly one of gamma_linear or gamma_db, got "
                f"gamma_linear={gamma_linear}, gamma_db={gamma_db}"
            )
        self.p0_w = p0_w
        self.noise_power_w = noise_power_w
        self.gamma = gamma_linear if gamma_linear is not None else db_to_linear(gamma_db)

        self.ota = OTAChannel(p0_w=p0_w, noise_power_w=noise_power_w)
        self._noise_rng = np.random.RandomState(seed)

        # Per-round OTA metrics with no standard CSV column — see the module
        # docstring's "Uplink physics" section.
        self.mse_history:      list = []
        self.energy_history:   list = []
        self.excluded_history: list = []

    # ------------------------------------------------------------------
    # Scheduler: maximize participating devices under the MSE budget
    # ------------------------------------------------------------------

    def select_clients(self, all_clients: list, num_to_select: int, rng, **kwargs) -> list:
        """
        Select every device that individually satisfies the MSE budget
        (paper eq. 11, N=1 closed form — see OTAChannel.select_feasible_devices).

        num_to_select is IGNORED: the paper's objective is to maximize the
        number of participating devices, not to hit a fixed count — capping
        at num_to_select would defeat that purpose. All of `all_clients` is
        treated as the candidate pool each round.

        Channel gains are drawn here (via the simulator-provided
        channel_model/rng in kwargs) purely to evaluate feasibility ahead of
        dispatch — same look-ahead pattern used by other channel-aware
        selectors in this codebase (e.g. FedAsyncTopKFastTotal's predecessor,
        README's ChannelAwareSelection). For a re-drawing channel model
        (e.g. ExpFadingChannelModel), this look-ahead value may differ from
        the gain drawn again later in the round for the actual transmission
        — a pre-existing characteristic of this framework's selection
        contract, not something specific to FedOTA.
        """
        channel_model = kwargs.get("channel_model")
        if channel_model is None:
            raise RuntimeError(
                "FedOTA.select_clients requires 'channel_model' in kwargs — "
                "make sure it's being driven by Simulator/AsyncSimulator, "
                "which supply it automatically."
            )

        candidates = [
            (c.client_id, float(c.num_samples), channel_model.channel_gain(c.profile, rng))
            for c in all_clients
        ]
        result = self.ota.select_feasible_devices(candidates, gamma=self.gamma)

        if not result.selected_ids:
            raise RuntimeError(
                f"FedOTA: no devices satisfy the MSE budget gamma={self.gamma:.4g} "
                f"this round (p0_w={self.p0_w}, noise_power_w={self.noise_power_w}). "
                f"Loosen gamma, raise p0_w, or lower noise_power_w."
            )

        self.excluded_history.append(result.excluded_ids)
        id_to_client = {c.client_id: c for c in all_clients}
        return [id_to_client[cid] for cid in result.selected_ids]

    # ------------------------------------------------------------------
    # Uplink physics override — called by Simulator._run_round BEFORE
    # aggregate(), replaces the default FDMA per-client time/energy with OTA.
    # ------------------------------------------------------------------

    def recompute_uplink_physics(
        self,
        client_updates: list,
        total_bandwidth_hz: float,
        **kwargs,
    ) -> None:
        """
        Rewrite each ClientUpdate's uplink fields in place to OTA physics
        (see the module docstring's "Uplink physics" section). Runs once per
        round, before aggregate(), so RoundResult / CSV reflect OTA — no
        side-channel reads needed.

        - upload_time_s = num_symbols / total_bandwidth_hz  (CONSTANT: all
          clients transmit simultaneously over the full band, so the uplink
          time is independent of the number of participating devices).
        - tx_energy_j   = num_symbols * |b_i|^2  (squared norm of the
          transmitted signal, paper eq. 5), b_i = zero-forcing scalar.
        - total_time_s / total_energy_j recomputed from the above (download
          time left as the Simulator already computed it — 0 when the
          Simulator's wireless.downlink_negligible is set).

        Also records this round's per-device OTA energy into energy_history.

        Args:
            client_updates: the selected clients' ClientUpdates for this round.
            total_bandwidth_hz: full system uplink bandwidth B (Hz).
            **kwargs: extra round context forwarded by the Simulator
                (e.g. downlink_negligible); unused here, accepted so the hook
                contract can grow without breaking implementers.
        """
        phis  = [float(u.num_samples) for u in client_updates]
        gains = [u.channel_gain for u in client_updates]
        eta   = self.ota.compute_eta(phis, gains)

        num_symbols = sum(t.numel() for t in client_updates[0].state_dict.values())
        upload_time_s = num_symbols / total_bandwidth_hz

        energies_j = {}
        for u in client_updates:
            e_tx = self.ota.transmission_energy_j(
                float(u.num_samples), u.channel_gain, eta, num_symbols
            )
            energies_j[u.client_id] = e_tx
            u.upload_time_s = upload_time_s
            u.tx_energy_j   = e_tx
            u.total_time_s   = u.compute_time_s + u.upload_time_s + u.download_time_s
            u.total_energy_j = u.compute_energy_j + u.tx_energy_j

        self.energy_history.append(energies_j)

    # ------------------------------------------------------------------
    # Updater: over-the-air aggregation
    # ------------------------------------------------------------------

    def aggregate(self, global_model, client_updates: list) -> OrderedDict:
        """
        Simulated AirComp superposition: FedAvg's weighted average plus
        Gaussian noise matching the achieved aggregation MSE (paper eq. 2,
        6, 10 — see flsim/system/ota.py module docstring for the full
        derivation). Records the achieved MSE into self.mse_history.

        The per-device transmission energy for this round is computed in
        recompute_uplink_physics() (called by the Simulator just before this),
        not here — keeping physical-layer (time/energy) and model-layer
        (aggregation) concerns separate.

        client_updates' .channel_gain fields are already populated by the
        Simulator earlier in the round (the same channel draw used for the
        round's time/energy computation) — reused here directly rather than
        re-drawing, so aggregation is self-consistent with whatever channel
        realization actually produced these updates.
        """
        state_dicts = [u.state_dict for u in client_updates]
        phis  = [float(u.num_samples) for u in client_updates]
        gains = [u.channel_gain for u in client_updates]

        agg_state, mse = self.ota.aggregate_state_dicts(
            state_dicts, phis, gains, rng=self._noise_rng
        )
        self.mse_history.append(mse)
        return agg_state
