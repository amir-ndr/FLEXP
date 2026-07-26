"""
system/split_cost.py: Physically-grounded latency + energy + traffic model for
split learning (SL / SFLV1 / SFLV2).

Same physical base as the sync/async/OTA simulators — so cross-paradigm
comparison is fair
---------------------------------------------------------------------------------
This model reuses the framework's existing primitives, NOT a new set of formulas:
  * FDMA link rate  = ChannelModel.achievable_rate_bps(...)  (Shannon capacity),
                      identical to what the sync/async CellularTimeModel uses.
  * Compute time    = cycles / frequency,  compute energy = kappa · f² · cycles
                      (DVFS), identical to EnergyModel.compute_energy_j's form.
  * TX energy       = tx_power · transmission_time  (uplink only, as in
                      EnergyModel.transmission_energy_j — the base station's
                      downlink energy is not charged to devices, matching FL).

What split learning ADDS on top of that shared base is only its different
WORKFLOW (per the co-training equations in SAFSL-style split-FL papers):
per device per round the pipeline is
    model-download → [ device FP → smashed-data uplink → server FP+BP →
                       gradient downlink → device BP ] × H iterations → model-upload
and the model's compute is charged partly on the (weak) device CPU and partly on
the (fast) edge server, split at the cut layer by measured FLOP fraction
(flsim.system.flops.compute_split_fraction).

Modelling choices (fixed at build time by the experiment / user):
  * server compute runs at a separate `server_cpu_frequency_hz` (edge server),
    while device compute runs at each device's own `profile.cpu_frequency_hz`.
  * downlink (model-download, gradient) uses the same symmetric Shannon rate as
    uplink (the framework's default assumption); it counts toward LATENCY but,
    like FL's downlink, its energy is the base station's and is not charged to
    devices. Set wireless.downlink_negligible to zero downlink TIME as well.
  * latency combination per variant (energy always sums over all devices):
        SL     : sum over devices          (clients processed sequentially)
        SFLV1  : max over devices          (clients + server fully parallel)
        SFLV2  : staged synchronization barriers, faithful to AdaptSFL eq.
                 (16)-(25) with client-side MA every round (I=1):
                   maxᵢ(FP + activation-uplink)
                 + Σᵢ(server FP+BP)          [single edge server, workload summed]
                 + maxᵢ(gradient-downlink + client BP)
                 + maxᵢ(client-model uplink)   + maxᵢ(client-model downlink)
                 Each device-side phase waits for its own straggler (a sum of
                 per-phase maxes), rather than one max over the whole pipeline.

Everything is a pure function of measured sizes + rates — cheap to call each
round, easy to reuse in your own split-based experiment.
"""

from dataclasses import dataclass
from typing import List

BITS_PER_ELEMENT = 32     # float32
BYTES_PER_ELEMENT = 4


@dataclass
class DevicePerRound:
    """Timing/energy/traffic breakdown for ONE device in one round."""
    # latency components (seconds)
    t_model_down:   float
    t_dev_compute:  float
    t_smashed_up:   float
    t_srv_compute:  float
    t_grad_down:    float
    t_model_up:     float
    # energy (joules), split by WHO spends it:
    #   dev_compute_energy_j, up_tx_energy_j  -> the DEVICE (battery)
    #   srv_compute_energy_j, dn_tx_energy_j  -> the INFRASTRUCTURE (edge server + BS)
    dev_compute_energy_j: float
    srv_compute_energy_j: float
    up_tx_energy_j:       float   # device uplink TX (smashed-data up + device-model up)
    dn_tx_energy_j:       float   # BS downlink TX (model down + gradients down)
    # traffic (bytes)
    traffic_bytes:  float
    # FP/BP split of device compute, with t_dev_fp + t_dev_bp == t_dev_compute
    # (exactly). Used by the SFLV2 staged-barrier latency (server sits BETWEEN
    # device FP and device BP), so FP joins the activation-uplink phase and BP
    # joins the gradient-downlink phase. Default 0 keeps any externally- or
    # legacy-constructed DevicePerRound valid; every other consumer reads only
    # t_dev_compute / device_path_s / full_path_s, which are unchanged.
    t_dev_fp: float = 0.0
    t_dev_bp: float = 0.0
    # which parts count toward total_energy_j: "total" (device + infrastructure)
    # or "device" (battery only — device compute + device uplink TX). See
    # SplitCostModel(energy_scope=...).
    energy_scope:   str = "total"

    @property
    def tx_energy_j(self) -> float:
        """All TX energy (uplink + downlink) — kept for back-compat / reporting."""
        return self.up_tx_energy_j + self.dn_tx_energy_j

    @property
    def device_path_s(self) -> float:
        """Critical path EXCLUDING server compute (device compute + its own comms)."""
        return (self.t_model_down + self.t_dev_compute + self.t_smashed_up
                + self.t_grad_down + self.t_model_up)

    @property
    def full_path_s(self) -> float:
        """Critical path INCLUDING server compute (server runs inline / in parallel)."""
        return self.device_path_s + self.t_srv_compute

    @property
    def total_energy_j(self) -> float:
        """
        Energy attributed to this device-round. "device" scope = battery only
        (device compute + device uplink TX) — excludes the plugged-in edge
        server's compute and the BS's downlink TX. "total" = everything.
        """
        if self.energy_scope == "device":
            return self.dev_compute_energy_j + self.up_tx_energy_j
        return (self.dev_compute_energy_j + self.srv_compute_energy_j
                + self.up_tx_energy_j + self.dn_tx_energy_j)


@dataclass
class SplitRoundCost:
    """Aggregated cost of one global round."""
    latency_s:      float   # this round's simulated duration (mode-dependent)
    traffic_bytes:  float   # total bytes communicated (mode-independent)
    total_energy_j: float   # sum over all devices (compute + uplink TX)


class SplitCostModel:
    """
    Analytic per-round split-learning cost, reusing the framework's channel model
    and DVFS energy form (see module docstring).

    Args:
        channel_model:            a ChannelModel (Shannon `achievable_rate_bps`).
        noise_psd_w_per_hz (float): N0, W/Hz (same value the sync/async sims use).
        kappa (float):            DVFS switched-capacitance κ (system.switched_capacitance).
        server_cpu_frequency_hz (float): edge-server frequency f_S (cycles/s).
        downlink_negligible (bool): if True, downlink transmissions take 0 time
            (base station assumed to have unlimited power/bandwidth).
    """

    def __init__(
        self,
        channel_model,
        noise_psd_w_per_hz: float,
        kappa: float,
        server_cpu_frequency_hz: float,
        downlink_negligible: bool = False,
        q_device: float = 1.0,
        q_server: float = 1.0,
        downlink_tx_power_w: float = None,
        energy_scope: str = "total",
    ):
        """
        Args (extending the class docstring):
            q_device (float): device FLOPs-per-cycle q_n (paper eq. 7/12).
                Device compute time = FLOPs / (f_device * q_device). Default 1.0
                (so "cycles_per_sample" == FLOPs and time == cycles/freq, i.e.
                the framework's original behaviour — fully backward compatible).
            q_server (float): edge-server FLOPs-per-cycle q_S (paper eq. 10),
                usually > q_device (server does more FLOPs/cycle). Default 1.0.
            downlink_tx_power_w (float): BS transmit power for downlink
                (model-download and gradient-download), P^DL in paper eq. 5/16.
                When set, downlink LATENCY uses a rate computed at this (BS)
                power instead of the device's uplink power, AND downlink energy
                P^DL*(t_model_down + t_grad_down) is charged (paper eq. 16
                includes these P^DL terms). When None (default), downlink reuses
                the uplink rate and NO downlink energy is charged — the
                framework's original symmetric-link, uplink-only-energy
                convention. Ignored entirely if downlink_negligible=True.
            energy_scope (str): which parts count toward each device-round's
                total_energy_j. "total" (default): device compute + server
                compute + uplink TX + downlink TX (everything, paper eq. 16).
                "device": DEVICE/battery only — device compute + device uplink
                TX; excludes the plugged-in edge server's compute and the BS's
                downlink TX. Use "device" for a battery-cost comparison where
                offloading to the server should SAVE energy.
        """
        self.channel_model = channel_model
        self.noise_psd = noise_psd_w_per_hz
        self.kappa = kappa
        self.f_server = server_cpu_frequency_hz
        self.downlink_negligible = downlink_negligible
        self.q_device = q_device
        self.q_server = q_server
        self.downlink_tx_power_w = downlink_tx_power_w
        self.energy_scope = energy_scope

    # ------------------------------------------------------------------
    # Per-device cost (one round)
    # ------------------------------------------------------------------

    def device_cost(
        self,
        profile,
        num_samples: int,
        local_epochs: int,
        cycles_per_sample: float,
        device_compute_fraction: float,
        activation_numel: int,
        client_param_count: int,
        bandwidth_hz: float,
        channel_gain: float,
        work_samples: float = None,
        server_freq_hz: float = None,
    ) -> DevicePerRound:
        """
        Cost of one device's participation in one global round.

        Args:
            profile:               ClientSystemProfile (reads cpu_frequency_hz, tx_power_w).
            num_samples (int):     n_k — this device's local sample count.
            local_epochs (int):    E — local passes per round.
            cycles_per_sample (float): C_k — total (full-model) CPU cycles (or
                FLOPs when q!=1; see q_device/q_server) per sample.
            device_compute_fraction (float): fraction of cycles on the device side
                (from flops.compute_split_fraction); server gets (1 - fraction).
            activation_numel (int): smashed-data elements per sample (client-model output).
            client_param_count (int): device-side model size in elements.
            bandwidth_hz (float):  B_n allocated to this device (FDMA).
            channel_gain (float):  g_n linear channel power gain.
            work_samples (float, optional): total sample-passes this round. If
                given, OVERRIDES num_samples * local_epochs — use it to pass the
                paper's H*b (H local iterations of a b-sample mini-batch) instead
                of full-epoch work. None (default) keeps num_samples*local_epochs.
            server_freq_hz (float, optional): the server frequency f^S_{n} this
                device is ALLOCATED for this round. The paper's f^S,max is the
                BS's total capacity, split across concurrently-served devices
                (sum_n f^S_n <= f^S,max) — so parallel variants (SFLV1, async
                split) pass f_S/n_concurrent here, while sequential-server
                variants (SL, SFLV2) pass None (= full f_S, one job at a time).

        Returns:
            DevicePerRound.
        """
        f_srv = server_freq_hz if server_freq_hz is not None else self.f_server
        # ---- FDMA link rate (Shannon), same primitive as the sync/async sims ----
        rate_bps = self.channel_model.achievable_rate_bps(
            bandwidth_hz=bandwidth_hz,
            tx_power_w=profile.tx_power_w,
            channel_gain=channel_gain,
            noise_psd_w_per_hz=self.noise_psd,
        )
        rate_bps = max(rate_bps, 1.0)

        # Downlink rate: 0 (negligible), a separate BS-power rate (paper P^DL),
        # or symmetric with the uplink (framework default). Same channel_gain as
        # uplink (reciprocal link), only the transmit power differs.
        if self.downlink_negligible:
            dl_rate = 0.0
        elif self.downlink_tx_power_w is not None:
            dl_rate = max(self.channel_model.achievable_rate_bps(
                bandwidth_hz=bandwidth_hz,
                tx_power_w=self.downlink_tx_power_w,
                channel_gain=channel_gain,
                noise_psd_w_per_hz=self.noise_psd,
            ), 1.0)
        else:
            dl_rate = rate_bps

        work = work_samples if work_samples is not None else num_samples * local_epochs
        dev_cycles = cycles_per_sample * device_compute_fraction
        srv_cycles = cycles_per_sample * (1.0 - device_compute_fraction)

        # ---- compute times: FLOPs / (frequency * FLOPs-per-cycle q) (paper eq. 7/10/12) ----
        t_dev_compute = (dev_cycles * work) / (profile.cpu_frequency_hz * self.q_device)
        t_srv_compute = (srv_cycles * work) / (f_srv * self.q_server)

        # FP/BP split of the device compute: backward ≈ 2× forward (see
        # flsim.system.flops), so FP is 1/3 of FP+BP. t_dev_bp is the exact
        # remainder so t_dev_fp + t_dev_bp == t_dev_compute (no float drift) —
        # device_path_s / full_path_s stay byte-identical for the other modes.
        t_dev_fp = t_dev_compute / 3.0
        t_dev_bp = t_dev_compute - t_dev_fp

        # ---- communication times (bits / rate) ----
        smashed_bits = activation_numel * BITS_PER_ELEMENT      # per sample
        model_bits   = client_param_count * BITS_PER_ELEMENT
        t_smashed_up = (smashed_bits * work) / rate_bps         # activations uplink
        t_grad_down  = 0.0 if dl_rate == 0.0 else (smashed_bits * work) / dl_rate  # gradients downlink
        t_model_down = 0.0 if dl_rate == 0.0 else model_bits / dl_rate
        t_model_up   = model_bits / rate_bps

        # ---- energy (paper eq. 16) ----
        # compute: DVFS  kappa*f^3*t = kappa * FLOPs * work * f^2 / q  (device & server)
        dev_compute_energy = self.kappa * dev_cycles * work * (profile.cpu_frequency_hz ** 2) / self.q_device
        srv_compute_energy = self.kappa * srv_cycles * work * (f_srv ** 2) / self.q_server
        # TX energy, split by who transmits:
        #   uplink   P^UL*(smashed_up + model_up)  -> the DEVICE (battery)
        #   downlink P^DL*(model_down + grad_down)  -> the BS (only when a BS
        #            downlink power is configured; else no downlink charge)
        up_tx_energy = profile.tx_power_w * (t_smashed_up + t_model_up)
        dn_tx_energy = 0.0
        if self.downlink_tx_power_w is not None and not self.downlink_negligible:
            dn_tx_energy = self.downlink_tx_power_w * (t_model_down + t_grad_down)

        # ---- traffic (bytes): smashed both ways + device model both ways ----
        smashed_bytes = 2 * activation_numel * work * BYTES_PER_ELEMENT
        model_bytes   = 2 * client_param_count * BYTES_PER_ELEMENT
        traffic_bytes = smashed_bytes + model_bytes

        return DevicePerRound(
            t_model_down=t_model_down, t_dev_compute=t_dev_compute,
            t_smashed_up=t_smashed_up, t_srv_compute=t_srv_compute,
            t_grad_down=t_grad_down, t_model_up=t_model_up,
            dev_compute_energy_j=dev_compute_energy,
            srv_compute_energy_j=srv_compute_energy,
            up_tx_energy_j=up_tx_energy, dn_tx_energy_j=dn_tx_energy,
            traffic_bytes=traffic_bytes,
            t_dev_fp=t_dev_fp, t_dev_bp=t_dev_bp,
            energy_scope=self.energy_scope,
        )

    # ------------------------------------------------------------------
    # Combine per-device costs into a round cost (mode-dependent latency)
    # ------------------------------------------------------------------

    def combine(self, mode: str, per_device: List[DevicePerRound]) -> SplitRoundCost:
        """
        Combine per-device costs into one round. Energy and traffic always sum
        over devices; latency depends on the variant (see module docstring).
        """
        mode = mode.lower()
        traffic = sum(d.traffic_bytes for d in per_device)
        energy  = sum(d.total_energy_j for d in per_device)

        if mode == "sl":
            latency = sum(d.full_path_s for d in per_device)              # sequential
        elif mode == "sflv1":
            latency = max(d.full_path_s for d in per_device)              # fully parallel
        elif mode == "sflv2":
            # Staged synchronization barriers per AdaptSFL eq. (16)-(25), I=1
            # (client-side model aggregation every round). Each device-side phase
            # blocks on its own straggler; the single edge server processes all
            # devices' activations sequentially (summed workload).
            phase_fp_up      = max(d.t_dev_fp + d.t_smashed_up for d in per_device)  # eq. 16+17
            server_seq       = sum(d.t_srv_compute for d in per_device)             # eq. 18+19
            phase_grad_bp    = max(d.t_grad_down + d.t_dev_bp for d in per_device)  # eq. 20+21
            phase_model_up   = max(d.t_model_up   for d in per_device)              # eq. 22
            phase_model_down = max(d.t_model_down for d in per_device)              # eq. 24
            latency = (phase_fp_up + server_seq + phase_grad_bp
                       + phase_model_up + phase_model_down)
        else:
            raise ValueError(f"mode must be 'sl'|'sflv1'|'sflv2', got {mode!r}")

        return SplitRoundCost(latency_s=latency, traffic_bytes=traffic, total_energy_j=energy)

    # ------------------------------------------------------------------
    # Centralized ("Normal") baseline — compute only, on the edge server
    # ------------------------------------------------------------------

    def centralized_cost(
        self, total_samples: int, local_epochs: int, cycles_per_sample: float
    ) -> SplitRoundCost:
        """
        Cost of one epoch of centralized training on the edge server (one
        powerful machine, no communication). Latency = full-model compute over
        all data at the server frequency; energy = DVFS compute energy; traffic = 0.
        """
        work = total_samples * local_epochs
        latency = (cycles_per_sample * work) / (self.f_server * self.q_server)
        energy = self.kappa * cycles_per_sample * work * (self.f_server ** 2) / self.q_server
        return SplitRoundCost(latency_s=latency, traffic_bytes=0.0, total_energy_j=energy)
