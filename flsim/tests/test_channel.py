"""
tests/test_channel.py: Unit tests for channel/conversions.py, channel/path_loss.py,
channel/fdma.py, and allocators/equal_split.py.
"""

import math
import pytest
import numpy as np
from dataclasses import dataclass

from flsim.channel.conversions import dbm_to_watts, db_to_linear, watts_to_dbm
from flsim.channel.path_loss import PathLossChannelModel
from flsim.channel.fdma import FDMAChannelModel
from flsim.allocators.equal_split import EqualSplitAllocator


# ---------------------------------------------------------------------------
# Minimal profile stub used only in channel tests
# ---------------------------------------------------------------------------

@dataclass
class _FakeProfile:
    distance_m: float
    shadowing_db: float = 0.0
    tx_power_w: float = 0.01   # 10 dBm
    client_id: int = 0
    cpu_frequency_hz: float = 1.0e9


# ---------------------------------------------------------------------------
# conversions.py
# ---------------------------------------------------------------------------

class TestConversions:
    def test_dbm_to_watts_10dbm(self):
        """10 dBm == 10 mW == 0.01 W"""
        assert abs(dbm_to_watts(10.0) - 0.01) < 1e-10

    def test_dbm_to_watts_30dbm(self):
        """30 dBm == 1 W"""
        assert abs(dbm_to_watts(30.0) - 1.0) < 1e-10

    def test_dbm_to_watts_0dbm(self):
        """0 dBm == 1 mW == 0.001 W"""
        assert abs(dbm_to_watts(0.0) - 0.001) < 1e-10

    def test_db_to_linear_0db(self):
        assert abs(db_to_linear(0.0) - 1.0) < 1e-10

    def test_db_to_linear_10db(self):
        assert abs(db_to_linear(10.0) - 10.0) < 1e-10

    def test_db_to_linear_negative(self):
        """−10 dB == 0.1"""
        assert abs(db_to_linear(-10.0) - 0.1) < 1e-10

    def test_watts_to_dbm_roundtrip(self):
        """watts_to_dbm(dbm_to_watts(x)) == x for several values"""
        for dbm in [-10.0, 0.0, 10.0, 23.0, 30.0]:
            assert abs(watts_to_dbm(dbm_to_watts(dbm)) - dbm) < 1e-8

    def test_watts_to_dbm_1w(self):
        assert abs(watts_to_dbm(1.0) - 30.0) < 1e-8

    def test_watts_to_dbm_001w(self):
        assert abs(watts_to_dbm(0.01) - 10.0) < 1e-8


# ---------------------------------------------------------------------------
# path_loss.py
# ---------------------------------------------------------------------------

NOISE_PSD = 10.0 ** ((-174.0 - 30.0) / 10.0)   # -174 dBm/Hz → W/Hz
BANDWIDTH = 2.0e7                                 # 20 MHz
TX_POWER  = dbm_to_watts(10.0)                   # 10 dBm → 0.01 W


class TestPathLoss:
    def _model(self):
        return PathLossChannelModel(
            total_bandwidth_hz=BANDWIDTH,
            noise_psd_w_per_hz=NOISE_PSD,
        )

    def test_channel_gain_positive(self):
        model = self._model()
        profile = _FakeProfile(distance_m=200.0, shadowing_db=0.0)
        g = model.channel_gain(profile, rng=None)
        assert g > 0.0

    def test_channel_gain_less_than_one(self):
        model = self._model()
        profile = _FakeProfile(distance_m=200.0, shadowing_db=0.0)
        g = model.channel_gain(profile, rng=None)
        assert g < 1.0

    def test_channel_gain_decreases_with_distance(self):
        model = self._model()
        g_near = model.channel_gain(_FakeProfile(distance_m=100.0), rng=None)
        g_far  = model.channel_gain(_FakeProfile(distance_m=400.0), rng=None)
        assert g_near > g_far

    def test_channel_gain_min_distance_clamp(self):
        """d=0 should not raise; min_distance_m clamp prevents log10(0)."""
        model = self._model()
        g = model.channel_gain(_FakeProfile(distance_m=0.0), rng=None)
        assert g > 0.0

    def test_achievable_rate_positive(self):
        model = self._model()
        profile = _FakeProfile(distance_m=200.0)
        g = model.channel_gain(profile, rng=None)
        rate = model.achievable_rate_bps(
            bandwidth_hz=BANDWIDTH,
            tx_power_w=TX_POWER,
            channel_gain=g,
            noise_psd_w_per_hz=NOISE_PSD,
        )
        assert rate > 0.0

    def test_achievable_rate_increases_with_gain(self):
        model = self._model()
        r_low  = model.achievable_rate_bps(BANDWIDTH, TX_POWER, 1e-12, NOISE_PSD)
        r_high = model.achievable_rate_bps(BANDWIDTH, TX_POWER, 1e-8,  NOISE_PSD)
        assert r_high > r_low

    # allocate_bandwidth has moved to EqualSplitAllocator — tested in TestEqualSplitAllocator


# ---------------------------------------------------------------------------
# fdma.py
# ---------------------------------------------------------------------------

class TestFDMA:
    def _model(self):
        return FDMAChannelModel(
            total_bandwidth_hz=BANDWIDTH,
            noise_psd_w_per_hz=NOISE_PSD,
        )

    def test_fdma_inherits_channel_gain(self):
        """FDMAChannelModel must still compute channel gain correctly."""
        model = self._model()
        profile = _FakeProfile(distance_m=300.0, shadowing_db=0.0)
        g = model.channel_gain(profile, rng=None)
        assert 0.0 < g < 1.0


# ---------------------------------------------------------------------------
# allocators/equal_split.py
# ---------------------------------------------------------------------------

def _profiles(n):
    return [_FakeProfile(distance_m=100.0, client_id=i, cpu_frequency_hz=float(i + 1) * 1e8)
            for i in range(n)]


class TestEqualSplitAllocator:
    def _alloc(self):
        return EqualSplitAllocator()

    def test_bandwidth_sums_to_total(self):
        alloc = self._alloc()
        profiles = _profiles(5)
        result = alloc.allocate_bandwidth(profiles, BANDWIDTH)
        assert abs(sum(result.values()) - BANDWIDTH) < 1.0

    def test_bandwidth_equal_split(self):
        alloc = self._alloc()
        profiles = _profiles(3)
        result = alloc.allocate_bandwidth(profiles, BANDWIDTH)
        expected = BANDWIDTH / 3
        for v in result.values():
            assert abs(v - expected) < 1.0

    def test_bandwidth_empty(self):
        alloc = self._alloc()
        result = alloc.allocate_bandwidth([], BANDWIDTH)
        assert result == {}

    def test_power_all_at_pmax(self):
        alloc = self._alloc()
        profiles = _profiles(4)
        p_max = 0.01
        result = alloc.allocate_power(profiles, p_max)
        for v in result.values():
            assert v == p_max

    def test_cpu_freq_from_profile(self):
        alloc = self._alloc()
        profiles = _profiles(3)
        result = alloc.allocate_cpu_freq(profiles, f_max_hz=2e9)
        for p in profiles:
            assert result[p.client_id] == p.cpu_frequency_hz

    def test_all_client_ids_present(self):
        alloc = self._alloc()
        profiles = _profiles(5)
        bw = alloc.allocate_bandwidth(profiles, BANDWIDTH)
        pw = alloc.allocate_power(profiles, 0.01)
        fq = alloc.allocate_cpu_freq(profiles, 2e9)
        ids = {p.client_id for p in profiles}
        assert set(bw) == ids
        assert set(pw) == ids
        assert set(fq) == ids
