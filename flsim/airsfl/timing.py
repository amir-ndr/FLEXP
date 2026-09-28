"""
flsim/airsfl/timing.py: OFDM uplink communication-time model for AirSFL and its
baselines (AirSFL WCNC draft, Eq. 11-13 and 25-26; evaluation roadmap Sec. 3, 5).

The metric of the AirSFL study is MODELED UPLINK COMMUNICATION TIME: client ->
M-server activation uploads plus client -> F-server model-update uploads (plus the
reliable label upload). Computation is excluded. Both server downlinks (cut
derivatives, prefix broadcast) are idealized as reliable and non-bottleneck, so
they are NOT timed for any method (paper Sec. II-C).

Two servers, each with its own receive array: the M-server (Nr antennas) receives
activations; the F-server (Nr_F antennas) receives model differences. Both uplink
stages use the same S subcarriers (spacing df, W = S*df) at different times; each
client has the same total power Pmax in each stage. Reference SNR before array
combining and before OFDMA bandwidth division: rho = Pmax*lambda_ref/(N0*W).

  ANALOG OFDM (AirSFL activations via ZF, AirComp aggregation) -- all clients
  occupy all S tones; a length-D real tensor packs into ceil(D/2) complex values,
  S in parallel per OFDM symbol of useful duration 1/df (Eq. 13):
        A_X(D) = ceil( ceil(D/2) / S ) / (eps_X * df)   ~=  D / (2 eps_X W)
  No Shannon rate is assigned to analog coordinates; SNR enters only as distortion.

  DIGITAL OFDMA -- client n gets S_n = S/N disjoint tones and concentrates its
  power Pmax on them; maximum-ratio reception over the receiving server's array;
  q = 32 bits per real value (Eq. 11-12):
        Rbar_n = eps_D * df * S_n * c_D,   c_D = E[log2(1 + rho * N * X)],
        X = sum_{a=1..Nr}|g_a|^2 ~ Gamma(Nr, 1),     D(D) = max_n q*D / Rbar_n.
  (per-tone SNR = Pmax*lambda*X/(S_n N0 df) = rho*(S/S_n)*X = rho*N*X). The
  expectation is a fluid/ergodic goodput estimated once from a long channel sample.
  The rate to the M-server uses Nr, to the F-server Nr_F (equal by default, then
  the two rates are identical: R_A,n = R_U,n, paper Sec. III-C).

Per-round UPLINK time (Eq. 25-26; d_a = activation size incl. batch B, d_c prefix
params, d_s suffix params, d = d_c + d_s, tau local steps, ell_y label time):

  AirSFL                  tau*[A_U(d_a) + ell_y] + A_A(d_c)
  Digital SFL-V1          tau*[D_U(d_a) + ell_y] + D_A(d_c)
  Sun-inspired FDMA-AC    tau*[D_U(d_a) + ell_y] + A_A(d_c)
  AirComp-FL              A_A(d)                      (full model, no labels)
  Digital FedAvg          D_A(d)

Formula regression: verify_reference_example() reproduces the earlier roadmap's
analytical example (N=8, Nr=32, S=64, eps=0.8, stage-3 cut) exactly.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

Q_BITS = 32  # bits per FP32 real value (digital serialization)


# ---------------------------------------------------------------------------
# Radio configuration
# ---------------------------------------------------------------------------

@dataclass
class RadioConfig:
    """OFDM/MIMO radio parameters (evaluation roadmap Sec. 3 defaults)."""
    N: int = 30                    # clients
    Nr: int = 64                   # M-server receive antennas (activation ZF)
    Nr_F: Optional[int] = None     # F-server receive antennas (AirComp); None -> Nr
    S: int = 120                   # subcarriers
    df_hz: float = 15e3            # subcarrier spacing (W = S*df = 1.8 MHz)
    Pmax_w: float = 0.1            # per-client total transmit power in each uplink stage (W)
    N0_dbm_per_hz: float = -167.0  # effective noise PSD (-174 dBm/Hz + 7 dB noise figure)
    eps_D: float = 0.6             # digital payload-efficiency fraction
    eps_U: float = 0.6             # analog activation-uplink efficiency
    eps_A: float = 0.6             # analog aggregation efficiency
    rho_db: float = 20.0           # reference SNR (per client, pre-combining, full band)
    num_classes: int = 10
    batch_size: int = 16

    def __post_init__(self):
        if self.Nr_F is None:
            self.Nr_F = self.Nr

    @property
    def W_hz(self) -> float:
        return self.S * self.df_hz

    @property
    def N0_w_per_hz(self) -> float:
        return 10.0 ** ((self.N0_dbm_per_hz - 30.0) / 10.0)

    @property
    def rho_lin(self) -> float:
        return 10.0 ** (self.rho_db / 10.0)

    @property
    def Sn(self) -> float:
        """Digital OFDMA tones per client, S_n = S / N."""
        return self.S / self.N

    @property
    def lambda_ref(self) -> float:
        """Path gain implied by rho: lambda = rho * N0 * W / Pmax (a calibration, not a distance)."""
        return self.rho_lin * self.N0_w_per_hz * self.W_hz / self.Pmax_w


# ---------------------------------------------------------------------------
# Digital OFDMA: long-term goodput (fluid service estimate)
# ---------------------------------------------------------------------------

def digital_spectral_efficiency(radio: RadioConfig, rng: Optional[np.random.RandomState] = None,
                                n_samples: int = 400000, Nr: Optional[int] = None) -> float:
    """Per-tone spectral efficiency c_D = E[log2(1 + rho*N*X)], X ~ Gamma(Nr, 1)
    (MRC over the receiving server's Nr antennas). Monte-Carlo over a long sample."""
    rng = rng if rng is not None else np.random.RandomState(0)
    X = rng.gamma(shape=Nr if Nr is not None else radio.Nr, scale=1.0, size=n_samples)
    return float(np.mean(np.log2(1.0 + radio.rho_lin * radio.N * X)))


def digital_rate_bps(radio: RadioConfig, c_D: Optional[float] = None,
                     rng: Optional[np.random.RandomState] = None, Nr: Optional[int] = None) -> float:
    """Rbar_n = eps_D * df * S_n * c_D  (bits/s, equal-link long-term goodput)."""
    if c_D is None:
        c_D = digital_spectral_efficiency(radio, rng, Nr=Nr)
    return radio.eps_D * radio.df_hz * radio.Sn * c_D


def digital_rates(radio: RadioConfig, seed: int = 2026) -> dict:
    """Goodputs to the two servers: {"U": client->M-server, "A": client->F-server}.
    Same channel sample seed for both, so equal arrays give identical rates."""
    cU = digital_spectral_efficiency(radio, np.random.RandomState(seed), Nr=radio.Nr)
    cA = cU if radio.Nr_F == radio.Nr else \
        digital_spectral_efficiency(radio, np.random.RandomState(seed), Nr=radio.Nr_F)
    return {"U": digital_rate_bps(radio, c_D=cU), "A": digital_rate_bps(radio, c_D=cA),
            "c_U": cU, "c_A": cA}


# ---------------------------------------------------------------------------
# Per-tensor uplink times
# ---------------------------------------------------------------------------

def analog_time_s(D: int, eps_X: float, radio: RadioConfig) -> float:
    """A_X(D) = ceil(ceil(D/2)/S) / (eps_X*df)  [~ D/(2 eps_X W)] (Eq. 13)."""
    symbols = math.ceil(math.ceil(D / 2) / radio.S)
    return symbols / (eps_X * radio.df_hz)


def digital_time_s(D: int, rate_bps: float) -> float:
    """D(D) = q*D / Rbar_n (Eq. 12; equal links -> the max over n is any n)."""
    return Q_BITS * D / rate_bps


def label_time_s(radio: RadioConfig, rate_bps: float) -> float:
    """ell_y = B*ceil(log2 J) / Rbar_n (labels per client per co-split step, reliable)."""
    return radio.batch_size * math.ceil(math.log2(radio.num_classes)) / rate_bps


# ---------------------------------------------------------------------------
# Per-round uplink time per method (Eq. 25-26)
# ---------------------------------------------------------------------------

METHODS = ("airsfl", "digital_sflv1", "sun_fdma_aircomp", "aircomp_fl", "digital_fedavg")

_ALIASES = {
    "airsfl": "airsfl",
    "digital_sflv1": "digital_sflv1", "digital-sfl-v1": "digital_sflv1", "sflv1": "digital_sflv1",
    "sun_fdma_aircomp": "sun_fdma_aircomp", "sun": "sun_fdma_aircomp", "fdma_aircomp": "sun_fdma_aircomp",
    "aircomp_fl": "aircomp_fl", "aircomp-fl": "aircomp_fl",
    "digital_fedavg": "digital_fedavg", "fedavg": "digital_fedavg", "fl": "digital_fedavg",
}


def uplink_time_breakdown(method: str, dims: dict, radio: RadioConfig, tau: int,
                          rates: Optional[dict] = None) -> dict:
    """Per-round uplink seconds split into {activation, labels, aggregation, total}.
    activation/labels go to the M-server, aggregation to the F-server.
    dims: {"d_c", "d_s", "d_a"} (d_a includes the batch dimension)."""
    m = _ALIASES.get(method.lower())
    if m is None:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    d_c, d_s, d_a = int(dims["d_c"]), int(dims["d_s"]), int(dims["d_a"])
    rates = rates if rates is not None else digital_rates(radio)
    AU = lambda D: analog_time_s(D, radio.eps_U, radio)
    AA = lambda D: analog_time_s(D, radio.eps_A, radio)
    DU = lambda D: digital_time_s(D, rates["U"])
    DA = lambda D: digital_time_s(D, rates["A"])
    # labels go up only when the M-server computes the loss, i.e. when there IS a split
    # (d_a > 0); in the no-split limit the SFL rows reduce exactly to the FL rows
    lab = tau * label_time_s(radio, rates["U"]) if d_a > 0 else 0.0
    act = agg = 0.0
    if m == "airsfl":
        act, agg = tau * AU(d_a), AA(d_c)
    elif m == "digital_sflv1":
        act, agg = tau * DU(d_a), DA(d_c)
    elif m == "sun_fdma_aircomp":
        act, agg = tau * DU(d_a), AA(d_c)
    elif m == "aircomp_fl":
        lab, agg = 0.0, AA(d_c + d_s)
    elif m == "digital_fedavg":
        lab, agg = 0.0, DA(d_c + d_s)
    return {"activation": act, "labels": lab, "aggregation": agg, "total": act + lab + agg}


def uplink_time_per_round(method: str, dims: dict, radio: RadioConfig, tau: int,
                          rates: Optional[dict] = None) -> float:
    """Modeled UPLINK seconds for one round (Eq. 25-26)."""
    return uplink_time_breakdown(method, dims, radio, tau, rates)["total"]


# ---------------------------------------------------------------------------
# Source-equivalent volume (roadmap: "do not turn concurrency into fake MB")
# ---------------------------------------------------------------------------

def source_equivalent_mb_per_round(method: str, dims: dict, radio: RadioConfig, tau: int) -> float:
    """Uplink source-equivalent MB per round (q_ref = 32):
        SFL (all three): N*[q(tau*d_a + d_c) + tau*B*ceil(log2 J)] / 8e6
        FL  (both):      N*q*d / 8e6
    The three matched SFL variants have IDENTICAL volume -- AirSFL saves AIRTIME,
    not source bytes."""
    m = _ALIASES.get(method.lower())
    d_c, d_s, d_a = int(dims["d_c"]), int(dims["d_s"]), int(dims["d_a"])
    label_bits = radio.batch_size * math.ceil(math.log2(radio.num_classes)) if d_a > 0 else 0
    if m in ("airsfl", "digital_sflv1", "sun_fdma_aircomp"):
        return radio.N * (Q_BITS * (tau * d_a + d_c) + tau * label_bits) / 8e6
    return radio.N * Q_BITS * (d_c + d_s) / 8e6


# ---------------------------------------------------------------------------
# Profiled model dimensions (CIFAR ResNet-18 GroupNorm; roadmap Sec. 2 table)
# ---------------------------------------------------------------------------

def profiled_dims(batch_size: int = 16) -> dict:
    """d_a(stage) = channels*H*W*B: stage1 64*32*32, stage2 128*16*16, stage3 256*8*8,
    stage4 512*4*4; d_c / d_s = prefix / suffix parameter counts."""
    da_per_sample = {1: 64 * 32 * 32, 2: 128 * 16 * 16, 3: 256 * 8 * 8, 4: 512 * 4 * 4}
    dc = {1: 149824, 2: 675392, 3: 2775104, 4: 11168832}
    ds = {1: 11024138, 2: 10498570, 3: 8398858, 4: 5130}
    return {stage: {"d_c": dc[stage], "d_s": ds[stage], "d_a": da_per_sample[stage] * batch_size}
            for stage in (1, 2, 3, 4)}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_reference_example(verbose: bool = True) -> dict:
    """Formula regression against the earlier roadmap's analytical example:
    N=8, Nr=32, S=64 (W=0.96 MHz), eps=0.8, rho=20 dB, tau=5, stage-3 cut ->
    c_D ~= 14.62; AirSFL 2.66, digital SFL-V1 93.1, Sun-style 31.7, AirComp-FL 7.27 s."""
    radio = RadioConfig(N=8, Nr=32, S=64, eps_D=0.8, eps_U=0.8, eps_A=0.8, rho_db=20.0)
    rates = digital_rates(radio, seed=2026)
    dims = profiled_dims(radio.batch_size)[3]
    t = lambda m: uplink_time_per_round(m, dims, radio, 5, rates)
    out = {"c_D": rates["c_U"], "airsfl_s": t("airsfl"), "digital_sflv1_s": t("digital_sflv1"),
           "sun_s": t("sun_fdma_aircomp"), "aircomp_fl_s": t("aircomp_fl")}
    golden = {"c_D": 14.62, "airsfl_s": 2.66, "digital_sflv1_s": 93.1, "sun_s": 31.7, "aircomp_fl_s": 7.27}
    out["ok"] = all(abs(out[k] - v) / v < 0.01 for k, v in golden.items())
    if verbose:
        print("=== timing formula regression (earlier roadmap example, stage-3 cut) ===")
        for k, v in golden.items():
            print(f"  {k:16s} = {out[k]:8.2f}   [golden {v}]")
        print(f"  REFERENCE_EXAMPLE: {'PASS' if out['ok'] else 'FAIL'}")
    return out


def verify_limit_cases(verbose: bool = True) -> bool:
    """Structural reductions of Eq. 25-26:
      cut after the LAST layer (d_s = 0, d_a = 0): SFL-V1 -> FedAvg, AirSFL -> AirComp-FL,
      Sun -> AirComp-FL;  Sun = SFL-V1 activation + labels + AirSFL aggregation at every
      cut; each phase is counted exactly once (total = activation + labels + aggregation)."""
    radio = RadioConfig()
    rates = digital_rates(radio)
    full = profiled_dims(radio.batch_size)[1]
    nosplit = {"d_c": full["d_c"] + full["d_s"], "d_s": 0, "d_a": 0}
    ul = lambda m, dims: uplink_time_per_round(m, dims, radio, 5, rates)
    checks = {
        "SFL-V1(no split) == FedAvg": (ul("digital_sflv1", nosplit), ul("digital_fedavg", nosplit)),
        "AirSFL(no split) == AirComp-FL": (ul("airsfl", nosplit), ul("aircomp_fl", nosplit)),
        "Sun(no split)    == AirComp-FL": (ul("sun_fdma_aircomp", nosplit), ul("aircomp_fl", nosplit)),
    }
    for stage, dims in profiled_dims(radio.batch_size).items():
        bs = uplink_time_breakdown("sun_fdma_aircomp", dims, radio, 5, rates)
        bd = uplink_time_breakdown("digital_sflv1", dims, radio, 5, rates)
        ba = uplink_time_breakdown("airsfl", dims, radio, 5, rates)
        checks[f"stage {stage}: Sun = SFLV1.act + labels + AirSFL.agg"] = \
            (bs["total"], bd["activation"] + bd["labels"] + ba["aggregation"])
        checks[f"stage {stage}: AirSFL total = sum of its phases"] = \
            (ba["total"], ba["activation"] + ba["labels"] + ba["aggregation"])
    ok = True
    if verbose:
        print("=== Eq. 25-26 limit-case / structural checks ===")
    for name, (a, b) in checks.items():
        good = abs(a - b) <= 1e-9 * max(1.0, abs(b))
        ok &= good
        if verbose:
            print(f"  {'OK ' if good else 'BAD'} {name:48s} {a:12.4f} vs {b:12.4f}")
    if verbose:
        print(f"  LIMIT_CASES: {'PASS' if ok else 'FAIL'}")
    return ok


def environment_summary(radio: Optional[RadioConfig] = None, stage: int = 2, tau: int = 5) -> None:
    """Print the per-round uplink budget of the default environment."""
    radio = radio or RadioConfig()
    rates = digital_rates(radio)
    dims = profiled_dims(radio.batch_size)[stage]
    print(f"=== environment: N={radio.N}, Nr_M={radio.Nr}, Nr_F={radio.Nr_F}, S={radio.S} "
          f"(W={radio.W_hz/1e6:.2f} MHz, S_n={radio.Sn:g}), rho={radio.rho_db:g} dB, "
          f"eps=({radio.eps_D},{radio.eps_U},{radio.eps_A}), cut {stage}, tau={tau} ===")
    print(f"  c_D = {rates['c_U']:.2f} bit/tone-use, digital goodput per client = {rates['U']/1e3:.1f} kbit/s")
    for m in METHODS:
        b = uplink_time_breakdown(m, dims, radio, tau, rates)
        print(f"  {m:17s} act {b['activation']:9.3f}  labels {b['labels']:.4f}  agg {b['aggregation']:9.3f}"
              f"  total {b['total']:9.3f} s/round")


if __name__ == "__main__":
    verify_reference_example()
    print()
    verify_limit_cases()
    print()
    environment_summary()
