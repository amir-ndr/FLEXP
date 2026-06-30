"""
interfaces/time_model.py: Abstract base class for simulated time models.

IMPORTANT: Simulated time is computed analytically from system parameters —
it is NEVER the wall-clock time of PyTorch training. This separation means
the simulator can study time-sensitive FL without GPU speed affecting results.
"""

from abc import ABC, abstractmethod


class TimeModel(ABC):
    """
    Base class for all simulated-time models.

    Computes how long each phase of a client round takes in simulated seconds.
    Actual PyTorch wall-clock time is NEVER used as simulated time.

    This class does NOT:
    - Perform any PyTorch operations.
    - Access the global model directly.
    - Make scheduling decisions.
    """

    @abstractmethod
    def compute_training_time(
        self, profile, num_samples: int, local_epochs: int, batch_size: int
    ) -> float:
        """
        Returns simulated training time in seconds for one client.

        Formula: tau_k = (local_epochs * cycles_per_sample * num_samples) / cpu_frequency_hz

        Args:
            profile: ClientSystemProfile with cpu_frequency_hz and cycles_per_sample.
            num_samples (int): number of local training samples D_k.
            local_epochs (int): number of local epochs I_k.
            batch_size (int): local mini-batch size (not used in basic formula; reserved).

        Returns:
            float: simulated training time in seconds.
        """

    @abstractmethod
    def compute_upload_time(
        self, profile, size_bits: float, bandwidth_hz: float
    ) -> float:
        """
        Returns simulated upload time in seconds.

        Formula: t_up = size_bits / achievable_rate_bps

        Args:
            profile: ClientSystemProfile (contains tx_power_w).
            size_bits (float): size of the model update in bits.
            bandwidth_hz (float): allocated uplink bandwidth for this client in Hz.

        Returns:
            float: simulated upload time in seconds.
        """

    @abstractmethod
    def compute_download_time(
        self, profile, size_bits: float, bandwidth_hz: float
    ) -> float:
        """
        Returns simulated download time in seconds.

        Args:
            profile: ClientSystemProfile.
            size_bits (float): size of the global model in bits.
            bandwidth_hz (float): allocated downlink bandwidth in Hz.

        Returns:
            float: simulated download time in seconds.
        """
