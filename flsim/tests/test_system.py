"""
tests/test_system.py: Unit tests for system/cellular_time.py and system/energy.py.

All tests verify that formulas match the paper's analytical definitions exactly,
not the actual PyTorch wall-clock training time.
"""

import math
import pytest
from dataclasses import dataclass

from flsim.channel.conversions import dbm_to_watts
from flsim.channel.fdma import FDMAChannelModel
from flsim.system.cellular_time import CellularTimeModel
from flsim.system.energy import EnergyModel, DEFAULT_KAPPA


# ---------------------------------------------------------------------------
# Fake channel model that returns fixed gain/rate for deterministic tests
# ---------------------------------------------------------------------------

class _FixedChannelModel:
    """Minimal channel stub returning fixed gain and rate."""
    def __init__(self, gain: float, rate_bps: float):
        self._gain = gain
        self._rate_bps = rate_bps

    def channel_gain(self, profile, rng):
        return self._gain

    def achievable_rate_bps(self, bandwidth_hz, tx_power_w, channel_gain, noise_psd_w_per_hz):
        return self._rate_bps

    def allocate_bandwidth(self, selected_client_ids, total_bandwidth_hz):
        n = len(selected_client_ids)
        return {cid: total_bandwidth_hz / n for cid in selected_client_ids}


# ---------------------------------------------------------------------------
# Fake profile stub
# ---------------------------------------------------------------------------

@dataclass
class _FakeProfile:
    client_id: int = 0
    x_m: float = 100.0
    y_m: float = 0.0
    distance_m: float = 100.0
    cpu_frequency_hz: float = 2.0e9    # 2 GHz
    cycles_per_sample: float = 2.0e4   # 20k cycles/sample
    tx_power_w: float = 0.01           # 10 dBm
    shadowing_db: float = 0.0
    num_samples: int = 600


NOISE_PSD = 10.0 ** ((-174.0 - 30.0) / 10.0)   # -174 dBm/Hz → W/Hz
BANDWIDTH = 2.0e7   # 20 MHz
RATE_BPS  = 5.0e6   # 5 Mbps (fixed for tests)


# ---------------------------------------------------------------------------
# CellularTimeModel tests
# ---------------------------------------------------------------------------

class TestCellularTimeModel:
    def _time_model(self, rate_bps=RATE_BPS):
        channel = _FixedChannelModel(gain=1e-10, rate_bps=rate_bps)
        return CellularTimeModel(channel_model=channel, noise_psd_w_per_hz=NOISE_PSD)

    def test_training_time_formula(self):
        """tau_k = (I * C * D) / f"""
        model = self._time_model()
        profile = _FakeProfile(
            cpu_frequency_hz=2.0e9,
            cycles_per_sample=2.0e4,
            num_samples=600,
        )
        I, C, D, f = 1, profile.cycles_per_sample, profile.num_samples, profile.cpu_frequency_hz
        expected = (I * C * D) / f
        result = model.compute_training_time(profile, num_samples=D, local_epochs=I, batch_size=32)
        assert abs(result - expected) < 1e-12

    def test_training_time_scales_with_epochs(self):
        model = self._time_model()
        profile = _FakeProfile()
        t1 = model.compute_training_time(profile, profile.num_samples, local_epochs=1, batch_size=32)
        t5 = model.compute_training_time(profile, profile.num_samples, local_epochs=5, batch_size=32)
        assert abs(t5 - 5 * t1) < 1e-12

    def test_training_time_scales_with_samples(self):
        model = self._time_model()
        profile = _FakeProfile()
        t600  = model.compute_training_time(profile, num_samples=600,  local_epochs=1, batch_size=32)
        t1200 = model.compute_training_time(profile, num_samples=1200, local_epochs=1, batch_size=32)
        assert abs(t1200 - 2 * t600) < 1e-12

    def test_upload_time_formula(self):
        """t_up = size_bits / rate_bps"""
        rate = 5.0e6
        model = self._time_model(rate_bps=rate)
        profile = _FakeProfile()
        size_bits = 28100.0
        expected = size_bits / rate
        result = model.compute_upload_time(profile, size_bits=size_bits, bandwidth_hz=BANDWIDTH / 10)
        assert abs(result - expected) < 1e-12

    def test_download_time_positive(self):
        model = self._time_model()
        profile = _FakeProfile()
        t = model.compute_download_time(profile, size_bits=28100.0, bandwidth_hz=BANDWIDTH / 10)
        assert t > 0.0

    def test_upload_equals_download(self):
        """Current model uses symmetric rates."""
        model = self._time_model()
        profile = _FakeProfile()
        t_up = model.compute_upload_time(profile, 28100.0, BANDWIDTH / 10)
        t_dn = model.compute_download_time(profile, 28100.0, BANDWIDTH / 10)
        assert abs(t_up - t_dn) < 1e-12


# ---------------------------------------------------------------------------
# EnergyModel tests
# ---------------------------------------------------------------------------

class TestEnergyModel:
    def test_compute_energy_formula(self):
        """E_comp = kappa * I * C * D * f^2"""
        model = EnergyModel(kappa=DEFAULT_KAPPA)
        profile = _FakeProfile(cpu_frequency_hz=2.0e9, cycles_per_sample=2.0e4)
        I, D = 1, 600
        expected = DEFAULT_KAPPA * I * profile.cycles_per_sample * D * (profile.cpu_frequency_hz ** 2)
        result = model.compute_energy_j(profile, local_epochs=I, num_samples=D)
        assert abs(result - expected) < 1e-40

    def test_compute_energy_positive(self):
        model = EnergyModel()
        profile = _FakeProfile()
        e = model.compute_energy_j(profile, local_epochs=1, num_samples=600)
        assert e > 0.0

    def test_transmission_energy_formula(self):
        """E_tx = p * t_up"""
        model = EnergyModel()
        profile = _FakeProfile(tx_power_w=0.01)
        t_up = 10.0   # 10 seconds
        expected = 0.01 * 10.0
        result = model.transmission_energy_j(profile, upload_time_s=t_up)
        assert abs(result - expected) < 1e-15

    def test_total_energy_is_sum(self):
        """total = comp + tx"""
        model = EnergyModel()
        profile = _FakeProfile()
        t_up = 5.0
        e_comp = model.compute_energy_j(profile, local_epochs=1, num_samples=600)
        e_tx   = model.transmission_energy_j(profile, upload_time_s=t_up)
        total  = model.total_energy_j(profile, local_epochs=1, num_samples=600, upload_time_s=t_up)
        assert abs(total - (e_comp + e_tx)) < 1e-20

    def test_energy_scales_with_epochs(self):
        model = EnergyModel()
        profile = _FakeProfile()
        e1 = model.compute_energy_j(profile, local_epochs=1, num_samples=600)
        e3 = model.compute_energy_j(profile, local_epochs=3, num_samples=600)
        assert abs(e3 - 3 * e1) < 1e-20
