"""
system/cellular_time.py: Cellular system time model for federated learning.

Implements TimeModel using the standard formulas from the FL-over-wireless
literature. All times are in simulated seconds derived analytically from
system parameters — wall-clock PyTorch training time is NEVER used.

Computation time formula:
    tau_k = (I_k * C_k * D_k) / f_k

Upload/Download time formula:
    t_k_up = size_bits / rate_bps
    (rate_bps provided by the ChannelModel, injected at construction time)
"""

from flsim.interfaces.time_model import TimeModel


class CellularTimeModel(TimeModel):
    """
    Simulated time model for a cellular FL deployment.

    Computation time is derived from CPU frequency and cycles-per-sample.
    Upload/download time is derived from the Shannon rate supplied by the
    injected channel model.

    This class does NOT:
    - Measure actual PyTorch wall-clock time.
    - Perform training or model operations.
    - Make scheduling decisions.
    """

    def __init__(self, channel_model, noise_psd_w_per_hz: float):
        """
        Args:
            channel_model: ChannelModel instance used to compute achievable rates.
            noise_psd_w_per_hz (float): thermal noise PSD N0 in W/Hz.
        """
        self.channel_model = channel_model
        self.noise_psd_w_per_hz = noise_psd_w_per_hz

    # ------------------------------------------------------------------
    # TimeModel interface
    # ------------------------------------------------------------------

    def compute_training_time(
        self, profile, num_samples: int, local_epochs: int, batch_size: int,
        cpu_freq_hz: float = None,
    ) -> float:
        """
        Simulated training time in seconds.

        Formula: tau_k = (I_k * C_k * D_k) / f_k

        Args:
            profile: ClientSystemProfile with cpu_frequency_hz, cycles_per_sample.
            num_samples (int): D_k — number of local training samples.
            local_epochs (int): I_k — local training epochs per round.
            batch_size (int): reserved for future mini-batch-level models; unused here.
            cpu_freq_hz (float, optional): override f_k from the allocator.
                If None, profile.cpu_frequency_hz is used.

        Returns:
            float: simulated training time in seconds.
        """
        f_k = cpu_freq_hz if cpu_freq_hz is not None else profile.cpu_frequency_hz
        tau_k = (local_epochs * profile.cycles_per_sample * num_samples) / f_k
        return tau_k

    def compute_upload_time(
        self, profile, size_bits: float, bandwidth_hz: float,
        channel_gain: float = None,
    ) -> float:
        """
        Simulated upload (uplink) time in seconds.

        Formula: t_k_up = size_bits / rate_bps

        Args:
            profile: ClientSystemProfile with tx_power_w.
            size_bits (float): size of model update in bits.
            bandwidth_hz (float): allocated uplink bandwidth B_k in Hz.
            channel_gain (float, optional): pre-computed linear channel gain.
                Pass this when the caller already holds the round's gain
                (e.g. from ExpFadingChannelModel where ρ is drawn once per
                round and must not be re-drawn here).  If None, the gain is
                recomputed via the channel model (safe for frozen models like
                PathLossChannelModel).

        Returns:
            float: simulated upload time in seconds.
        """
        if channel_gain is None:
            channel_gain = self.channel_model.channel_gain(profile, rng=None)
        rate_bps = self.channel_model.achievable_rate_bps(
            bandwidth_hz=bandwidth_hz,
            tx_power_w=profile.tx_power_w,
            channel_gain=channel_gain,
            noise_psd_w_per_hz=self.noise_psd_w_per_hz,
        )
        t_up = size_bits / rate_bps
        return t_up

    def compute_download_time(
        self, profile, size_bits: float, bandwidth_hz: float,
        channel_gain: float = None,
    ) -> float:
        """
        Simulated download (downlink) time in seconds.

        Uses the same uplink Shannon rate as a simplifying assumption
        (symmetric channel). Downlink asymmetry can be added by injecting
        a separate downlink channel model in a future extension.

        Args:
            profile: ClientSystemProfile.
            size_bits (float): size of global model in bits.
            bandwidth_hz (float): allocated downlink bandwidth in Hz.
            channel_gain (float, optional): pre-computed gain (see compute_upload_time).

        Returns:
            float: simulated download time in seconds.
        """
        # NOTE: Using uplink rate as a proxy for downlink (symmetric assumption).
        # To add asymmetry, inject a separate downlink ChannelModel.
        return self.compute_upload_time(profile, size_bits, bandwidth_hz, channel_gain)
