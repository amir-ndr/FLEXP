"""
system/energy.py: Energy consumption model for federated learning clients.

Energy formulas from the standard dynamic voltage and frequency scaling (DVFS)
model used in FL-over-wireless literature.

Computation energy:
    E_k_comp = kappa * I_k * C_k * D_k * f_k^2
    where kappa = 1e-28 (effective switched capacitance coefficient)

Transmission energy:
    E_k_tx = p_k * t_k_up

All energy values are in joules (J).
"""

# Effective switched capacitance for DVFS energy model (F·s²/cycle²)
# Standard value from the FL-over-wireless literature; do not change without citation.
DEFAULT_KAPPA = 1.0e-28


class EnergyModel:
    """
    DVFS-based energy model for federated learning clients.

    Computes computation and transmission energy separately so that
    the trade-off between local training and communication cost can be studied.

    This class does NOT:
    - Perform any PyTorch operations.
    - Compute simulated time (use CellularTimeModel for that).
    - Make scheduling decisions.
    """

    def __init__(self, kappa: float = DEFAULT_KAPPA):
        """
        Args:
            kappa (float): effective switched capacitance in F·s²/cycle².
                           Default: 1e-28 (from literature).
        """
        self.kappa = kappa

    def compute_energy_j(
        self,
        profile,
        local_epochs: int,
        num_samples: int,
        cpu_freq_hz: float = None,
    ) -> float:
        """
        Simulated computation energy in joules.

        Formula: E_k_comp = kappa * I_k * C_k * D_k * f_k^2

        Args:
            profile: ClientSystemProfile with cpu_frequency_hz, cycles_per_sample.
            local_epochs (int): I_k — local epochs per round.
            num_samples (int): D_k — number of local training samples.
            cpu_freq_hz (float, optional): override f_k from the allocator.
                If None, profile.cpu_frequency_hz is used.

        Returns:
            float: computation energy in joules.
        """
        f_k = cpu_freq_hz if cpu_freq_hz is not None else profile.cpu_frequency_hz
        C_k = profile.cycles_per_sample
        E_comp = self.kappa * local_epochs * C_k * num_samples * (f_k ** 2)
        return E_comp

    def transmission_energy_j(
        self,
        profile,
        upload_time_s: float,
        tx_power_w: float = None,
    ) -> float:
        """
        Simulated transmission (uplink) energy in joules.

        Formula: E_k_tx = p_k * t_k_up

        Args:
            profile: ClientSystemProfile with tx_power_w (p_k in watts).
            upload_time_s (float): simulated upload time t_k_up in seconds.
            tx_power_w (float, optional): override p_k from the allocator.
                If None, profile.tx_power_w is used.

        Returns:
            float: transmission energy in joules.
        """
        p_k = tx_power_w if tx_power_w is not None else profile.tx_power_w
        return p_k * upload_time_s

    def total_energy_j(
        self,
        profile,
        local_epochs: int,
        num_samples: int,
        upload_time_s: float,
    ) -> float:
        """
        Total energy consumed by one client in one round (joules).

        Formula: E_k = E_k_comp + E_k_tx

        Args:
            profile: ClientSystemProfile.
            local_epochs (int): I_k.
            num_samples (int): D_k.
            upload_time_s (float): simulated uplink time in seconds.

        Returns:
            float: total energy in joules.
        """
        e_comp = self.compute_energy_j(profile, local_epochs, num_samples)
        e_tx   = self.transmission_energy_j(profile, upload_time_s)
        return e_comp + e_tx
