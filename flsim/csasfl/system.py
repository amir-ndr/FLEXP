"""
flsim/csasfl/system.py: the CSA-SFL paper's system model (Sec. Performance Evaluation), applied
identically to every method.

Computation heterogeneity
    s_k ~ LogNormal(0, sigma_s^2), sigma_s = 2, fixed per device (drawn per seed, identical for
    all methods of that seed). Training one mini-batch takes t_0 / s_k (t_0 = 0.2 s at unit
    speed), so E local epochs take the paper's
        t_k = E * ceil(|D_k| / B) * t_0 / s_k.
    Time is computation only, as in the paper: transmission and PS computation are not timed.
    A synchronous group (an intra-cluster co-training pass, an FL / SFLv2 round) finishes at
    max_k of its members' t_k; an asynchronous arrival (SAFSL) at dispatch + t_k.
    Method-specific extensions (stated where used):
      * CSA-SFL gradient recollection: every device runs one mini-batch, all clusters halt,
        halt = max_k t_0 / s_k.
      * ParallelSFL (iteration-based): a bottom worker's iteration costs t_0 / s_i, the top
        worker's per-batch top-submodel compute r * t_0 / s_top with r = MACs(layers 9-12) /
        MACs(layers 1-8) (the top worker is a device, not the PS); its Eq. (15) round time is
        max_c tau_c * max_i (t_0/s_i + r t_0/s_top).

Communication overhead (paper: actual tensor dimensions at 32 bits, cumulative MB = 1e6 bytes)
    split device-round : device-side model download + upload (2 d_c), and per processed sample
                         the smashed data up (a values), its label up (1) and the returned
                         smashed-data gradient down (a)  -> 4 * (2 d_c + n_samples (2a + 1)) bytes
    full-model round   : 4 * 2 d (FL; ParallelSFL's PS <-> top-worker exchange)
    gradient probe     : smashed + label + gradient for the probe batch, plus the upload of g_k (d_c)
"""

import math
from dataclasses import dataclass

import numpy as np

BYTES = 4                        # 32-bit floating point (labels are counted as 32-bit values too)


def device_speeds(K: int, sigma_s: float, seed: int) -> np.ndarray:
    """s_k ~ LogNormal(0, sigma_s^2), one fixed draw per device."""
    return np.random.RandomState(int(seed) * 7919 + 1).lognormal(0.0, float(sigma_s), int(K))


def batches(n_samples: int, B: int) -> int:
    return int(math.ceil(int(n_samples) / int(B)))


def train_time(n_samples: int, E: int, B: int, t0: float, s: float) -> float:
    """t_k = E * ceil(|D_k| / B) * t_0 / s_k (paper)."""
    return E * batches(n_samples, B) * t0 / s


@dataclass
class Dims:
    d_c: int                     # device-side parameters (layers 1..cut)
    d_s: int                     # server-side parameters
    act: int                     # smashed-data values per sample

    @property
    def d(self) -> int:
        return self.d_c + self.d_s

    def split_round(self, n_samples: float) -> float:
        """Bytes of one device's split co-training pass over n_samples samples."""
        return BYTES * (2 * self.d_c + n_samples * (2 * self.act + 1))

    def smashed(self, n_samples: float) -> float:
        """Bytes of the per-sample exchange only (no model transfers)."""
        return BYTES * n_samples * (2 * self.act + 1)

    def full_round(self) -> float:
        return BYTES * 2 * self.d

    def device_model(self) -> float:
        return BYTES * self.d_c

    def probe(self, n_samples: float) -> float:
        """Gradient recollection of one device: probe batch exchange + upload of g_k."""
        return self.smashed(n_samples) + BYTES * self.d_c
