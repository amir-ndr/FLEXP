"""
flsim/airsfl/timing.py: OFDM uplink communication-time model for AirSFL and its
baselines (AirSFL WCNC draft Table I; Evaluation Roadmap Sec. 2-3).

The PRIMARY metric of the AirSFL study is MODELED UPLINK COMMUNICATION TIME to an
accuracy target -- NOT computation time (excluded) and NOT the compute-based
simulated_time of the rest of flsim. Downlink is a separate sensitivity study.

Two transports share one OFDM uplink. Radio: N clients, Nr server antennas, S
subcarriers, spacing df, bandwidth W = S*df, per-client power Pmax, noise PSD N0,
reference (pre-array-combining, full-band) SNR rho = Pmax*lambda_ref/(N0*W).

  ANALOG OFDM (AirSFL activations via ZF, aggregation via AirComp) -- all clients
  occupy all S tones; a length-D real tensor packs into ceil(D/2) complex values
  (both quadratures fill a symbol), sent S-at-a-time:
        A_X(D) = ceil( ceil(D/2) / S ) / (eps_X * df)   ~=  D / (2 eps_X W)
  The useful OFDM symbol duration is 1/df (NOT 1/W): S coordinates go in parallel.

  DIGITAL OFDMA (baselines) -- client n gets S_n = S/N disjoint tones, MRC over Nr
  antennas, q = 32 bits/value:
        Rbar_n = eps_D * df * sum_{j in S_n} E[ log2(1 + Pmax||h_nj||^2/(S_n N0 df)) ]
        D_dig(D) = max_n q*D / Rbar_n
  Under equal links Rbar_n = eps_D * df * S_n * c_D, with per-tone spectral
  efficiency c_D = E[log2(1 + rho * N * X)], X = sum_{a=1..Nr}|g_a|^2 ~ Gamma(Nr,1)
  (g_a ~ CN(0,1)). The per-tone SNR is rho*N*X because digital OFDMA concentrates
  each client's power into S_n = S/N tones (S/S_n = N).

Per-round UPLINK time (Table I; d_a = activation size incl. batch B, d_c = prefix
params, d_s = suffix params, d = d_c+d_s, tau local steps, ell_y = label time):

  AirSFL                  tau*A_U(d_a) + A_A(d_c) + tau*ell_y
  Digital SFL-V1          tau*D_dig(d_a) + D_dig(d_c) + tau*ell_y
  Sun-style FDMA-AirComp  tau*D_dig(d_a) + A_A(d_c) + tau*ell_y
  AirComp-FL              A_A(d)                       (full model, no split)
  Digital FedAvg (ctrl)   D_dig(d)

Verified against the roadmap's analytical example (N=8, Nr=32, rho=20 dB,
W=0.96 MHz, eps=0.8, stage-3 cut): see verify_reference_example().
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

Q_BITS = 32  # bits per FP32 real value


# ---------------------------------------------------------------------------
# Radio configuration
# ---------------------------------------------------------------------------

@dataclass
class RadioConfig:
    """OFDM/MIMO radio parameters (roadmap Sec. 4 defaults)."""
    N: int = 8                     # clients
    Nr: int = 32                   # server receive antennas
    S: int = 64                    # subcarriers
    df_hz: float = 15e3            # subcarrier spacing  (W = S*df = 0.96 MHz)
    Pmax_w: float = 0.1            # per-client transmit power (W)
    N0_dbm_per_hz: float = -167.0  # effective noise PSD (thermal -174 + 7 dB NF)
    eps_D: float = 0.8             # digital payload-efficiency fraction
    eps_U: float = 0.8             # analog activation-uplink efficiency
    eps_A: float = 0.8             # analog aggregation efficiency
    rho_db: float = 20.0           # reference SNR (per client, pre-combining, full band)
    num_classes: int = 10
    batch_size: int = 16
    # ---- downlink (sensitivity study only; excluded from the primary uplink metric) ----
    # Reliable digital OFDMA downlink, common to every SFL variant (paper Sec. II-C,
    # V-C). Default rate derived by channel reciprocity: same Rayleigh h_n, Nr-antenna
    # MRT (array gain), same S_n = S/N tone split, BS power Pdl per client. Pdl = 0.3 W
    # follows SAFSL (BS 0.3 W per device vs device 0.1-0.2 W); AdaptSFL likewise uses a
    # separate, faster downlink (370 vs 75-80 Mbps). dl_scale sweeps the goodput.
    Pdl_w: float = 0.3
    eps_DL: float = 0.8

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
        """Path gain implied by rho: lambda = rho * N0 * W / Pmax."""
        return self.rho_lin * self.N0_w_per_hz * self.W_hz / self.Pmax_w


# ---------------------------------------------------------------------------
# Digital OFDMA: long-term goodput (fluid service estimate)
# ---------------------------------------------------------------------------

def digital_spectral_efficiency(radio: RadioConfig, rng: Optional[np.random.RandomState] = None,
                                n_samples: int = 400000) -> float:
    """
    Per-tone spectral efficiency c_D = E[log2(1 + rho*N*X)] bits/complex-use,
    with X = sum_{a=1..Nr} |g_a|^2 ~ Gamma(Nr, 1)  (post-MRC over Nr antennas).
    Monte-Carlo over an independent long channel sample (roadmap: estimate Rbar
    once from a long sample, reuse across matched methods).
    """
    rng = rng if rng is not None else np.random.RandomState(0)
    X = rng.gamma(shape=radio.Nr, scale=1.0, size=n_samples)   # sum of Nr unit exponentials
    per_tone_snr = radio.rho_lin * radio.N * X
    return float(np.mean(np.log2(1.0 + per_tone_snr)))


def digital_rate_bps(radio: RadioConfig, c_D: Optional[float] = None,
                     rng: Optional[np.random.RandomState] = None) -> float:
    """Rbar_n = eps_D * df * S_n * c_D  (bits/s, equal-link long-term goodput)."""
    if c_D is None:
        c_D = digital_spectral_efficiency(radio, rng)
    return radio.eps_D * radio.df_hz * radio.Sn * c_D


# ---------------------------------------------------------------------------
# Per-tensor uplink times
# ---------------------------------------------------------------------------

def analog_time_s(D: int, eps_X: float, radio: RadioConfig) -> float:
    """A_X(D) = ceil(ceil(D/2)/S) / (eps_X*df)  [~ D/(2 eps_X W)]."""
    symbols = math.ceil(math.ceil(D / 2) / radio.S)
    return symbols / (eps_X * radio.df_hz)


def digital_time_s(D: int, radio: RadioConfig, rate_bps: Optional[float] = None,
                   c_D: Optional[float] = None, rng: Optional[np.random.RandomState] = None) -> float:
    """D_dig(D) = q*D / Rbar_n."""
    if rate_bps is None:
        rate_bps = digital_rate_bps(radio, c_D=c_D, rng=rng)
    return Q_BITS * D / rate_bps


def label_time_s(radio: RadioConfig, rate_bps: Optional[float] = None,
                 c_D: Optional[float] = None, rng: Optional[np.random.RandomState] = None) -> float:
    """ell_y = B*ceil(log2 J) / Rbar_n  (label bits per local step, digital)."""
    if rate_bps is None:
        rate_bps = digital_rate_bps(radio, c_D=c_D, rng=rng)
    label_bits = radio.batch_size * math.ceil(math.log2(radio.num_classes))
    return label_bits / rate_bps


# ---------------------------------------------------------------------------
# Per-round uplink time per method (Table I)
# ---------------------------------------------------------------------------

METHODS = ("airsfl", "digital_sflv1", "sun_fdma_aircomp", "aircomp_fl", "digital_fedavg")

_ALIASES = {
    "airsfl": "airsfl",
    "digital_sflv1": "digital_sflv1", "digital-sfl-v1": "digital_sflv1", "sflv1": "digital_sflv1",
    "sun_fdma_aircomp": "sun_fdma_aircomp", "sun": "sun_fdma_aircomp", "fdma_aircomp": "sun_fdma_aircomp",
    "aircomp_fl": "aircomp_fl", "aircomp-fl": "aircomp_fl",
    "digital_fedavg": "digital_fedavg", "fedavg": "digital_fedavg", "fl": "digital_fedavg",
}


def uplink_time_per_round(method: str, dims: dict, radio: RadioConfig, tau: int,
                          include_labels: bool = True, c_D: Optional[float] = None,
                          rng: Optional[np.random.RandomState] = None) -> float:
    """
    Modeled UPLINK seconds for one round (Table I).

    dims: {"d_c":..., "d_s":..., "d_a":...} (d_a includes the batch dimension B).
    tau : local co-split steps per round.
    """
    m = _ALIASES.get(method.lower())
    if m is None:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    d_c, d_s, d_a = int(dims["d_c"]), int(dims["d_s"]), int(dims["d_a"])
    d = d_c + d_s
    rate = digital_rate_bps(radio, c_D=c_D, rng=rng)
    ell_y = label_time_s(radio, rate_bps=rate)

    def AU(D): return analog_time_s(D, radio.eps_U, radio)
    def AA(D): return analog_time_s(D, radio.eps_A, radio)
    def Dd(D): return digital_time_s(D, radio, rate_bps=rate)

    # Labels go up only when the server computes the loss, i.e. when there IS a split
    # (d_a > 0). In the no-split limit (cut after the last layer: d_s = d_a = 0) the
    # SFL rows reduce EXACTLY to the FL rows (SFL-V1 -> FedAvg, AirSFL/Sun -> AirComp-FL).
    lab = tau * ell_y if (include_labels and d_a > 0) else 0.0
    if m == "airsfl":
        return tau * AU(d_a) + AA(d_c) + lab
    if m == "digital_sflv1":
        return tau * Dd(d_a) + Dd(d_c) + lab
    if m == "sun_fdma_aircomp":
        return tau * Dd(d_a) + AA(d_c) + lab
    if m == "aircomp_fl":
        return AA(d)          # full-model difference, one AirComp per round
    if m == "digital_fedavg":
        return Dd(d)          # full-model digital upload, one per round


def uplink_time_breakdown(method: str, dims: dict, radio: RadioConfig, tau: int,
                          c_D: Optional[float] = None, rng: Optional[np.random.RandomState] = None) -> dict:
    """Per-round uplink time split into {activation, model, labels, total} seconds
    (for the stacked per-round phase-timing figure)."""
    m = _ALIASES.get(method.lower())
    d_c, d_s, d_a = int(dims["d_c"]), int(dims["d_s"]), int(dims["d_a"])
    d = d_c + d_s
    rate = digital_rate_bps(radio, c_D=c_D, rng=rng)
    ell_y = label_time_s(radio, rate_bps=rate)

    def AU(D): return analog_time_s(D, radio.eps_U, radio)
    def AA(D): return analog_time_s(D, radio.eps_A, radio)
    def Dd(D): return digital_time_s(D, radio, rate_bps=rate)

    act = mod = lab = 0.0
    lab_split = tau * ell_y if d_a > 0 else 0.0
    if m == "airsfl":
        act, mod, lab = tau * AU(d_a), AA(d_c), lab_split
    elif m == "digital_sflv1":
        act, mod, lab = tau * Dd(d_a), Dd(d_c), lab_split
    elif m == "sun_fdma_aircomp":
        act, mod, lab = tau * Dd(d_a), AA(d_c), lab_split
    elif m == "aircomp_fl":
        mod = AA(d)
    elif m == "digital_fedavg":
        mod = Dd(d)
    return {"activation": act, "model": mod, "labels": lab, "total": act + mod + lab}


# ---------------------------------------------------------------------------
# Downlink (sensitivity study; paper Eq. 25 + Table I downlink column)
# ---------------------------------------------------------------------------

def downlink_spectral_efficiency(radio: RadioConfig, rng: Optional[np.random.RandomState] = None,
                                 n_samples: int = 400000) -> float:
    """Per-tone DL spectral efficiency under reciprocity: same h_n, Nr-antenna MRT,
    S_n tones, BS power Pdl per client -> per-tone SNR = (Pdl/Pmax)*rho*N*X."""
    rng = rng if rng is not None else np.random.RandomState(1)
    X = rng.gamma(shape=radio.Nr, scale=1.0, size=n_samples)
    snr = (radio.Pdl_w / radio.Pmax_w) * radio.rho_lin * radio.N * X
    return float(np.mean(np.log2(1.0 + snr)))


def downlink_rate_bps(radio: RadioConfig, c_DL: Optional[float] = None, dl_scale: float = 1.0,
                      rng: Optional[np.random.RandomState] = None) -> float:
    """Rbar^DL_n = dl_scale * eps_DL * df * S_n * c_DL (per-client distinct-stream goodput)."""
    if c_DL is None:
        c_DL = downlink_spectral_efficiency(radio, rng)
    return dl_scale * radio.eps_DL * radio.df_hz * radio.Sn * c_DL


def downlink_time_per_round(method: str, dims: dict, radio: RadioConfig, tau: int,
                            c_DL: Optional[float] = None, dl_scale: float = 1.0,
                            bc_scale: float = 1.0,
                            rng: Optional[np.random.RandomState] = None) -> float:
    """Table I downlink column:
        SFL variants (AirSFL, digital SFL-V1, Sun):  tau*D_DL(d_a) + B_DL(d_c)
        FL variants  (AirComp-FL, digital FedAvg):   B_DL(d_c + d_s)
    D_DL(D) = max_n q*D/Rbar^DL_n (a distinct cut derivative to every client, OMA).
    B_DL(D) = q*D/R^DL_bc (one common vector), R^DL_bc = bc_scale * Rbar^DL_n. The
    paper leaves R^DL_bc as a swept service rate; the two natural bounds are
      bc_scale = 1 : the common model goes over each client's own S_n tones
                     (AdaptSFL per-device download) -- pessimistic for a broadcast;
      bc_scale = N : one common stream over the full band W = S*df at the same
                     per-tone efficiency (ideal multicast) -- optimistic.
    Only B_DL depends on bc_scale, so it matters most for the FL baselines, whose
    whole downlink is the broadcast. All matched SFL methods share the SAME
    downlink service (identical DL time)."""
    m = _ALIASES.get(method.lower())
    d_c, d_s, d_a = int(dims["d_c"]), int(dims["d_s"]), int(dims["d_a"])
    r = downlink_rate_bps(radio, c_DL=c_DL, dl_scale=dl_scale, rng=rng)
    DDL = lambda D: Q_BITS * D / r
    BDL = lambda D: Q_BITS * D / (bc_scale * r)
    if m in ("airsfl", "digital_sflv1", "sun_fdma_aircomp"):
        return tau * DDL(d_a) + BDL(d_c)
    return BDL(d_c + d_s)


# ---------------------------------------------------------------------------
# Source-equivalent volume (roadmap Sec. 3: "do not turn concurrency into fake MB")
# ---------------------------------------------------------------------------

def source_equivalent_mb_per_round(method: str, dims: dict, radio: RadioConfig, tau: int) -> float:
    """Uplink source-equivalent MB per round (q_ref = 32):
        SFL (all three): N*[q(tau*d_a + d_c) + tau*B*ceil(log2 J)] / 8e6
        FL  (both):      N*q*d / 8e6
    The three matched SFL variants have IDENTICAL volume -- AirSFL saves AIRTIME,
    not source bytes (analog symbols carry no digital bitstream of this size)."""
    m = _ALIASES.get(method.lower())
    d_c, d_s, d_a = int(dims["d_c"]), int(dims["d_s"]), int(dims["d_a"])
    label_bits = radio.batch_size * math.ceil(math.log2(radio.num_classes)) if d_a > 0 else 0
    if m in ("airsfl", "digital_sflv1", "sun_fdma_aircomp"):
        return radio.N * (Q_BITS * (tau * d_a + d_c) + tau * label_bits) / 8e6
    return radio.N * Q_BITS * (d_c + d_s) / 8e6


# ---------------------------------------------------------------------------
# Profiled model dimensions (CIFAR ResNet-18 GroupNorm, B=16; roadmap Sec. 6)
# ---------------------------------------------------------------------------

def profiled_dims(batch_size: int = 16) -> dict:
    """Roadmap Sec. 6 table (cut = end of residual stage). d_a scales with B.
    d_a(stage) = channels * H * W * B: stage1 64*32*32, stage2 128*16*16,
    stage3 256*8*8, stage4 512*4*4."""
    da_per_sample = {1: 64 * 32 * 32, 2: 128 * 16 * 16, 3: 256 * 8 * 8, 4: 512 * 4 * 4}
    dc = {1: 149824, 2: 675392, 3: 2775104, 4: 11168832}
    ds = {1: 11024138, 2: 10498570, 3: 8398858, 4: 5130}
    return {stage: {"d_c": dc[stage], "d_s": ds[stage], "d_a": da_per_sample[stage] * batch_size}
            for stage in (1, 2, 3, 4)}


# ---------------------------------------------------------------------------
# Reference-example verification (roadmap Sec. 6 analytical illustration)
# ---------------------------------------------------------------------------

def verify_reference_example(verbose: bool = True) -> dict:
    """Reproduce the roadmap's analytical example and check the golden numbers:
    N=8, Nr=32, rho=20 dB, W=0.96 MHz, eps=0.8, tau=5, stage-3 cut ->
    c_D ~= 14.62; AirSFL 2.66, digital SFL-V1 93.1, Sun-style 31.7, AirComp-FL 7.27 s."""
    radio = RadioConfig()  # roadmap defaults (N=8, Nr=32, rho=20 dB, ...)
    rng = np.random.RandomState(2026)
    c_D = digital_spectral_efficiency(radio, rng)
    dims = profiled_dims(radio.batch_size)[3]   # stage 3
    tau = 5
    out = {
        "c_D": c_D,
        "digital_rate_Mbps": digital_rate_bps(radio, c_D=c_D) / 1e6,
        "airsfl_s": uplink_time_per_round("airsfl", dims, radio, tau, c_D=c_D),
        "digital_sflv1_s": uplink_time_per_round("digital_sflv1", dims, radio, tau, c_D=c_D),
        "sun_s": uplink_time_per_round("sun_fdma_aircomp", dims, radio, tau, c_D=c_D),
        "aircomp_fl_s": uplink_time_per_round("aircomp_fl", dims, radio, tau, c_D=c_D),
        "digital_fedavg_s": uplink_time_per_round("digital_fedavg", dims, radio, tau, c_D=c_D),
    }
    out["digital_analog_ratio"] = out["digital_sflv1_s"] / out["airsfl_s"]
    if verbose:
        print("=== AirSFL timing model — roadmap reference example (stage-3 cut) ===")
        print(f"  c_D (per-tone spectral eff)  = {out['c_D']:.2f}  bits/use   [golden ~14.62]")
        print(f"  digital goodput per client   = {out['digital_rate_Mbps']:.3f} Mbps")
        print(f"  AirSFL                        = {out['airsfl_s']:.2f} s/round  [golden 2.66]")
        print(f"  Digital SFL-V1                = {out['digital_sflv1_s']:.1f} s/round  [golden 93.1]")
        print(f"  Sun-style FDMA-AirComp        = {out['sun_s']:.1f} s/round  [golden 31.7]")
        print(f"  AirComp-FL                    = {out['aircomp_fl_s']:.2f} s/round  [golden 7.27]")
        print(f"  Digital FedAvg (control)      = {out['digital_fedavg_s']:.1f} s/round")
        print(f"  digital/analog ratio          = {out['digital_analog_ratio']:.1f}    [golden ~35 (SFLV1/AirSFL)]")
    return out


def verify_limit_cases(verbose: bool = True) -> bool:
    """Structural reductions of Table I (UL and DL):
      cut after the LAST layer (whole model on the client: d_s = 0, d_a = 0):
        digital SFL-V1 -> digital FedAvg ;  AirSFL -> AirComp-FL ;  Sun -> AirComp-FL
      Sun = (digital SFL-V1 activation part) + (AirSFL model part), at every cut.
      All SFL variants share the same downlink time at a given cut."""
    radio = RadioConfig()
    c_D = digital_spectral_efficiency(radio, np.random.RandomState(2026))
    c_DL = downlink_spectral_efficiency(radio, np.random.RandomState(2027))
    full = profiled_dims(radio.batch_size)[1]
    d = full["d_c"] + full["d_s"]
    nosplit = {"d_c": d, "d_s": 0, "d_a": 0}
    tau = 5
    ul = lambda m, dims: uplink_time_per_round(m, dims, radio, tau, c_D=c_D)
    dl = lambda m, dims: downlink_time_per_round(m, dims, radio, tau, c_DL=c_DL)
    dlN = lambda m, dims: downlink_time_per_round(m, dims, radio, tau, c_DL=c_DL, bc_scale=radio.N)
    checks = {
        "UL SFL-V1(no split) == FedAvg": (ul("digital_sflv1", nosplit), ul("digital_fedavg", nosplit)),
        "UL AirSFL(no split) == AirComp-FL": (ul("airsfl", nosplit), ul("aircomp_fl", nosplit)),
        "UL Sun(no split)    == AirComp-FL": (ul("sun_fdma_aircomp", nosplit), ul("aircomp_fl", nosplit)),
        "DL SFL-V1(no split) == FedAvg": (dl("digital_sflv1", nosplit), dl("digital_fedavg", nosplit)),
        "DL AirSFL(no split) == AirComp-FL": (dl("airsfl", nosplit), dl("aircomp_fl", nosplit)),
        "DL AirSFL(no split) == AirComp-FL, bc=N": (dlN("airsfl", nosplit), dlN("aircomp_fl", nosplit)),
        "DL FedAvg: full-band bc = per-client / N": (dlN("digital_fedavg", nosplit),
                                                     dl("digital_fedavg", nosplit) / radio.N),
    }
    for stage, dims in profiled_dims(radio.batch_size).items():
        bs = uplink_time_breakdown("sun_fdma_aircomp", dims, radio, tau, c_D=c_D)
        bd = uplink_time_breakdown("digital_sflv1", dims, radio, tau, c_D=c_D)
        ba = uplink_time_breakdown("airsfl", dims, radio, tau, c_D=c_D)
        checks[f"stage {stage}: Sun = SFLV1.act + AirSFL.model"] = (bs["total"], bd["activation"] + ba["model"] + bs["labels"])
        checks[f"stage {stage}: DL identical across SFL variants"] = (dl("airsfl", dims), dl("digital_sflv1", dims))
    ok = True
    if verbose:
        print("=== Table I limit-case / structural checks ===")
    for name, (a, b) in checks.items():
        good = abs(a - b) <= 1e-9 * max(1.0, abs(b))
        ok &= good
        if verbose:
            print(f"  {'OK ' if good else 'BAD'} {name:45s} {a:12.4f} vs {b:12.4f}")
    if verbose:
        print(f"  LIMIT_CASES: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    verify_reference_example()
    print()
    verify_limit_cases()
