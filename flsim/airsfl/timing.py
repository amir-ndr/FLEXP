"""
flsim/airsfl/timing.py: OFDM uplink communication-time model for AirSFL and its
baselines (AirSFL WCNC draft, Eq. 11-13 and 25-26; evaluation roadmap Sec. 3, 5).

The metric of the AirSFL study is MODELED UPLINK COMMUNICATION TIME: client ->
M-server activation uploads plus client -> F-server model-update uploads (plus the
reliable label upload); computation is modelled separately (compute.py). Both server downlinks (cut
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
  q bits per real value (radio.q_bits: 16 = FP16, the experiments' default; 32 = FP32)
  (Eq. 11-12):
        Rbar_n = eps_D * df * S_n * c_D,   c_D = E[log2(1 + rho * N * X)],
        X = sum_{a=1..Nr}|g_a|^2 ~ Gamma(Nr, 1),     D(D) = max_n q*D / Rbar_n.
  (per-tone SNR = Pmax*lambda*X/(S_n N0 df) = rho*(S/S_n)*X = rho*N*X). The
  expectation is a fluid/ergodic goodput estimated once from a long channel sample.
  The rate to the M-server uses Nr, to the F-server Nr_F (equal by default, then
  the two rates are identical: R_A,n = R_U,n, paper Sec. III-C).

  DIGITAL MULTI-USER ZF (stronger digital baselines, not OMA) -- every client sends its
  own coded stream (q bits per value) on ALL S tones with Pmax/S per tone (like the analog stages);
  the receiving server separates the N streams with the same ZF filter AirSFL uses and
  decodes each one. Post-ZF SNR = rho*Y, Y = 1/[(H~^H H~)^-1]_nn ~ Gamma(Nr-N+1, 1):
        Rbar_ZF = eps_D * df * S * c_ZF,   c_ZF = E[log2(1 + rho * Y)],   Z(D) = q*D / Rbar_ZF.
  At N = 1 it equals OFDMA (single-user MRC on all tones). Its gap to AirSFL is only the
  signalling cost q/c_ZF vs 1/2 tone-use per value (plus the smaller ZF array gain);
  the OFDMA gap additionally contains the N-fold spatial reuse.

  UNEQUAL PATH GAINS (optional robustness setting): lambda_n = g_n lambda_ref, the same to
  both servers; rho is the median client's SNR (path_gain_offsets_db). Every digital
  upload ends with its slowest client (Eq. 12), so the digital rates above are evaluated at
  the weakest client's SNR rho * g_min. Analog airtime does not depend on the gains; they
  enter only the analog distortion (simulator).

Per-round UPLINK time (Eq. 25-26; d_a = activation size incl. batch B, d_c prefix
params, d_s suffix params, d = d_c + d_s, tau local steps, ell_y label time):

  AirSFL                  tau*[A_U(d_a) + ell_y] + A_A(d_c)
  Digital SFL-V1          tau*[D_U(d_a) + ell_y] + D_A(d_c)
  Hybrid FDMA-AirComp     tau*[D_U(d_a) + ell_y] + A_A(d_c)      (Sun-inspired)
  AirComp-FL              A_A(d)                      (full model, no labels)
  Digital FedAvg          D_A(d)
  Digital SFL-V1 (ZF)     tau*[Z_U(d_a) + ell_y] + Z_A(d_c)      (ell_y at the ZF rate)
  Hybrid ZF-AirComp       tau*[Z_U(d_a) + ell_y] + A_A(d_c)
  Digital FedAvg (ZF)     Z_A(d)

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
    gains_db: Optional[tuple] = None   # per-client long-term path gains lambda_n / lambda_ref in dB, the
                                       # same to both servers (draft: h ~ CN(0, lambda_n I)); None = equal
                                       # gains. rho refers to lambda_ref (see path_gain_offsets_db)
    q_bits: int = 32                   # bits per real value of every reliable DIGITAL payload (activations,
                                       # model differences): 32 = FP32, 16 = FP16 (the experiments' default,
                                       # run_airsfl.py --digital-q; the simulator then really rounds those
                                       # tensors); labels and analog stages are unaffected

    def __post_init__(self):
        if self.Nr_F is None:
            self.Nr_F = self.Nr
        if self.gains_db is not None:
            self.gains_db = tuple(float(g) for g in self.gains_db)
            if len(self.gains_db) != self.N:
                raise ValueError(f"{len(self.gains_db)} path gains for N={self.N} clients")
        self.q_bits = int(self.q_bits)
        if self.q_bits not in (16, 32):
            raise ValueError(f"q_bits must be 32 (FP32) or 16 (FP16), got {self.q_bits}")

    @property
    def g_lin(self) -> np.ndarray:
        """Per-client path gains relative to lambda_ref (linear); ones for equal gains."""
        if self.gains_db is None:
            return np.ones(self.N)
        return 10.0 ** (np.asarray(self.gains_db, dtype=np.float64) / 10.0)

    @property
    def unequal_gains(self) -> bool:
        return self.gains_db is not None and any(g != 0.0 for g in self.gains_db)

    @property
    def g_min_db(self) -> float:
        return 0.0 if self.gains_db is None else min(self.gains_db)

    @property
    def rho_min_lin(self) -> float:
        """SNR of the weakest client, which paces every digital upload (Eq. 12: max over n)."""
        return self.rho_lin * 10.0 ** (self.g_min_db / 10.0)

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
    (MRC over the receiving server's Nr antennas). Monte-Carlo over a long sample.
    With unequal path gains it is the weakest client's (rho -> rho * g_min): the digital
    upload finishes when the slowest client does (Eq. 12)."""
    rng = rng if rng is not None else np.random.RandomState(0)
    X = rng.gamma(shape=Nr if Nr is not None else radio.Nr, scale=1.0, size=n_samples)
    return float(np.mean(np.log2(1.0 + radio.rho_min_lin * radio.N * X)))


def digital_rate_bps(radio: RadioConfig, c_D: Optional[float] = None,
                     rng: Optional[np.random.RandomState] = None, Nr: Optional[int] = None) -> float:
    """Rbar_n = eps_D * df * S_n * c_D  (bits/s, equal-link long-term goodput)."""
    if c_D is None:
        c_D = digital_spectral_efficiency(radio, rng, Nr=Nr)
    return radio.eps_D * radio.df_hz * radio.Sn * c_D


def digital_zf_spectral_efficiency(radio: RadioConfig, rng: Optional[np.random.RandomState] = None,
                                   n_samples: int = 400000, Nr: Optional[int] = None) -> float:
    """Per-tone spectral efficiency of DIGITAL multi-user MIMO with ZF reception: every client
    transmits its own coded stream on all S tones (power Pmax/S per tone, like the analog
    stages) and the server separates the N streams with the ZF matrix C = H(H^H H)^-1 before
    decoding each one. Post-ZF SNR of client n = (Pmax/S) / (N0 df [(H^H H)^-1]_nn) = rho g_n Y,
    Y = 1/[(H~^H H~)^-1]_nn ~ Gamma(Nr - N + 1, 1) for i.i.d. Rayleigh ([(H^H H)^-1]_nn =
    [(H~^H H~)^-1]_nn / g_n for H = H~ diag(sqrt(g))); c_ZF = E[log2(1 + rho g Y)] of the weakest
    client, which paces the upload."""
    rng = rng if rng is not None else np.random.RandomState(0)
    Nr = Nr if Nr is not None else radio.Nr
    if radio.N > Nr:
        return float("nan")                          # ZF cannot separate more streams than antennas
    Y = rng.gamma(shape=Nr - radio.N + 1, scale=1.0, size=n_samples)
    return float(np.mean(np.log2(1.0 + radio.rho_min_lin * Y)))


def path_gain_offsets_db(N: int, spread_db: float, seed: int) -> Optional[tuple]:
    """Unequal long-term path gains (draft: h_n ~ CN(0, lambda_n I), matched to both servers):
    N offsets 10 log10(lambda_n / lambda_ref), equally spaced over [-spread/2, +spread/2] dB
    (stratified uniform-in-dB, so every seed has the same weakest client, exactly
    spread * (1 - 1/N) / 2 dB below the reference) and assigned to clients by a seeded
    permutation (which data partition is weak changes with the seed; identical for every
    method of a seed). The reference SNR rho is the median client's. spread 0 -> None."""
    if not spread_db:
        return None
    offs = spread_db * ((np.arange(N) + 0.5) / N - 0.5)
    perm = np.random.RandomState(int(seed) * 7919 + 59).permutation(N)
    return tuple(round(float(x), 6) for x in offs[perm])


def digital_rates(radio: RadioConfig, seed: int = 2026) -> dict:
    """Goodputs to the two servers: {"U": client->M-server, "A": client->F-server} for digital
    OFDMA (S/N tones per client, MRC) and {"U_zf", "A_zf"} for digital multi-user ZF (all S
    tones, ZF separation). Same channel sample seed for both servers, so equal arrays give
    identical rates. With unequal path gains these are the weakest client's goodputs: the
    clients upload concurrently and the stage ends when the slowest finishes (Eq. 12)."""
    cU = digital_spectral_efficiency(radio, np.random.RandomState(seed), Nr=radio.Nr)
    cA = cU if radio.Nr_F == radio.Nr else \
        digital_spectral_efficiency(radio, np.random.RandomState(seed), Nr=radio.Nr_F)
    # same sample seed as OFDMA: at N = 1 both schemes are single-user MRC on all S tones
    # and the two rates coincide exactly (verify_zf)
    zU = digital_zf_spectral_efficiency(radio, np.random.RandomState(seed), Nr=radio.Nr)
    zA = zU if radio.Nr_F == radio.Nr else \
        digital_zf_spectral_efficiency(radio, np.random.RandomState(seed), Nr=radio.Nr_F)
    zf = lambda c: radio.eps_D * radio.df_hz * radio.S * c
    return {"U": digital_rate_bps(radio, c_D=cU), "A": digital_rate_bps(radio, c_D=cA),
            "c_U": cU, "c_A": cA, "U_zf": zf(zU), "A_zf": zf(zA), "c_zf_U": zU, "c_zf_A": zA}


# ---------------------------------------------------------------------------
# Per-tensor uplink times
# ---------------------------------------------------------------------------

def analog_time_s(D: int, eps_X: float, radio: RadioConfig) -> float:
    """A_X(D) = ceil(ceil(D/2)/S) / (eps_X*df)  [~ D/(2 eps_X W)] (Eq. 13)."""
    symbols = math.ceil(math.ceil(D / 2) / radio.S)
    return symbols / (eps_X * radio.df_hz)


def digital_time_s(D: int, rate_bps: float, q: int = Q_BITS) -> float:
    """D(D) = q*D / Rbar_n (Eq. 12; equal links -> the max over n is any n); q bits per value."""
    return q * D / rate_bps


def label_time_s(radio: RadioConfig, rate_bps: float) -> float:
    """ell_y = B*ceil(log2 J) / Rbar_n (labels per client per co-split step, reliable)."""
    return radio.batch_size * math.ceil(math.log2(radio.num_classes)) / rate_bps


# ---------------------------------------------------------------------------
# Per-round uplink time per method (Eq. 25-26)
# ---------------------------------------------------------------------------

METHODS = ("airsfl", "digital_sflv1", "sun_fdma_aircomp", "aircomp_fl", "digital_fedavg",
           "digital_sflv1_zf", "hybrid_zf_aircomp", "digital_fedavg_zf")
SPLIT_METHODS = ("airsfl", "digital_sflv1", "sun_fdma_aircomp", "digital_sflv1_zf", "hybrid_zf_aircomp")
ZF_METHODS = ("digital_sflv1_zf", "hybrid_zf_aircomp", "digital_fedavg_zf")          # digital multi-user ZF

_ALIASES = {
    "airsfl": "airsfl",
    "digital_sflv1": "digital_sflv1", "digital-sfl-v1": "digital_sflv1", "sflv1": "digital_sflv1",
    "sun_fdma_aircomp": "sun_fdma_aircomp", "sun": "sun_fdma_aircomp", "fdma_aircomp": "sun_fdma_aircomp",
    "aircomp_fl": "aircomp_fl", "aircomp-fl": "aircomp_fl",
    "digital_fedavg": "digital_fedavg", "fedavg": "digital_fedavg", "fl": "digital_fedavg",
    "digital_sflv1_zf": "digital_sflv1_zf", "sflv1_zf": "digital_sflv1_zf",
    "hybrid_zf_aircomp": "hybrid_zf_aircomp", "zf_aircomp": "hybrid_zf_aircomp",
    "digital_fedavg_zf": "digital_fedavg_zf",
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
    q = radio.q_bits                                        # digital payload bits per value (32 or 16)
    AU = lambda D: analog_time_s(D, radio.eps_U, radio)
    AA = lambda D: analog_time_s(D, radio.eps_A, radio)
    DU = lambda D: digital_time_s(D, rates["U"], q)         # OFDMA (S/N tones per client, concurrent)
    DA = lambda D: digital_time_s(D, rates["A"], q)
    ZU = lambda D: digital_time_s(D, rates["U_zf"], q)      # digital multi-user ZF (all tones, concurrent)
    ZA = lambda D: digital_time_s(D, rates["A_zf"], q)
    # labels go up only when the M-server computes the loss, i.e. when there IS a split
    # (d_a > 0); in the no-split limit the SFL rows reduce exactly to the FL rows. They use
    # the method's digital activation link (OFDMA or ZF); label bits do not depend on q.
    link = "U_zf" if m in ZF_METHODS else "U"
    lab = tau * label_time_s(radio, rates[link]) if d_a > 0 else 0.0
    act = agg = 0.0
    if m == "airsfl":
        act, agg = tau * AU(d_a), AA(d_c)
    elif m == "digital_sflv1":
        act, agg = tau * DU(d_a), DA(d_c)
    elif m == "sun_fdma_aircomp":
        act, agg = tau * DU(d_a), AA(d_c)
    elif m == "digital_sflv1_zf":
        act, agg = tau * ZU(d_a), ZA(d_c)
    elif m == "hybrid_zf_aircomp":
        act, agg = tau * ZU(d_a), AA(d_c)
    elif m == "aircomp_fl":
        lab, agg = 0.0, AA(d_c + d_s)
    elif m == "digital_fedavg":
        lab, agg = 0.0, DA(d_c + d_s)
    elif m == "digital_fedavg_zf":
        lab, agg = 0.0, ZA(d_c + d_s)
    return {"activation": act, "labels": lab, "aggregation": agg, "total": act + lab + agg}


def uplink_time_per_round(method: str, dims: dict, radio: RadioConfig, tau: int,
                          rates: Optional[dict] = None) -> float:
    """Modeled UPLINK seconds for one round (Eq. 25-26)."""
    return uplink_time_breakdown(method, dims, radio, tau, rates)["total"]


# ---------------------------------------------------------------------------
# Source-equivalent volume (roadmap: "do not turn concurrency into fake MB")
# ---------------------------------------------------------------------------

def source_equivalent_mb_per_round(method: str, dims: dict, radio: RadioConfig, tau: int,
                                   q_ref: int = Q_BITS) -> float:
    """Uplink source-equivalent MB per round at q_ref bits per value (default 32; the CSVs
    record this FP32 reference, the figures use the drawn digital payload's q):
        SFL (all):  N*[q_ref(tau*d_a + d_c) + tau*B*ceil(log2 J)] / 8e6
        FL  (all):  N*q_ref*d / 8e6
    The matched SFL variants have IDENTICAL volume -- AirSFL saves AIRTIME, not source bytes."""
    m = _ALIASES.get(method.lower())
    d_c, d_s, d_a = int(dims["d_c"]), int(dims["d_s"]), int(dims["d_a"])
    label_bits = radio.batch_size * math.ceil(math.log2(radio.num_classes)) if d_a > 0 else 0
    if m in SPLIT_METHODS:
        return radio.N * (q_ref * (tau * d_a + d_c) + tau * label_bits) / 8e6
    return radio.N * q_ref * (d_c + d_s) / 8e6


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
    checks["HybridZF(no split) == AirComp-FL"] = (ul("hybrid_zf_aircomp", nosplit), ul("aircomp_fl", nosplit))
    checks["SFLV1-ZF(no split) == q d / R_ZF"] = (ul("digital_sflv1_zf", nosplit),
                                                  Q_BITS * nosplit["d_c"] / rates["A_zf"])
    checks["SFLV1-ZF(no split) == FedAvg-ZF"] = (ul("digital_sflv1_zf", nosplit), ul("digital_fedavg_zf", nosplit))
    for stage, dims in profiled_dims(radio.batch_size).items():
        bh = uplink_time_breakdown("hybrid_zf_aircomp", dims, radio, 5, rates)
        bz = uplink_time_breakdown("digital_sflv1_zf", dims, radio, 5, rates)
        ba = uplink_time_breakdown("airsfl", dims, radio, 5, rates)
        checks[f"stage {stage}: HybridZF = SFLV1-ZF.act + labels + AirSFL.agg"] = \
            (bh["total"], bz["activation"] + bz["labels"] + ba["aggregation"])
    ok = True
    if verbose:
        print("=== Eq. 25-26 limit-case / structural checks ===")
    for name, (a, b) in checks.items():
        good = abs(a - b) <= 1e-9 * max(1.0, abs(b))
        ok &= good
        if verbose:
            print(f"  {'OK ' if good else 'BAD'} {name:58s} {a:12.4f} vs {b:12.4f}")
    if verbose:
        print(f"  LIMIT_CASES: {'PASS' if ok else 'FAIL'}")
    return ok


def verify_zf(verbose: bool = True) -> bool:
    """Digital multi-user ZF rate:
      (1) the Gamma(Nr-N+1, 1) post-ZF gain used for c_ZF == Monte-Carlo over explicit
          channels H ~ CN(0, I) and the ZF filter (H^H H)^-1 H^H, at 20 dB and -10 dB;
      (2) N = 1: ZF == OFDMA exactly (single-user MRC on all S tones);
      (3) golden values of the default environment (cut 2, tau 5, 20 dB):
          c_ZF ~= 11.75, Digital SFL-V1 (ZF) ~= 8.31 s, Hybrid ZF-AirComp ~= 6.92 s per round."""
    ok = True
    rng = np.random.RandomState(7)
    base = RadioConfig()
    lines = []
    for rho_db in (20.0, -10.0):
        radio = RadioConfig(rho_db=rho_db)
        N, Nr = radio.N, radio.Nr
        H = (rng.normal(size=(3000, Nr, N)) + 1j * rng.normal(size=(3000, Nr, N))) / math.sqrt(2)
        Y = 1.0 / np.real(np.diagonal(np.linalg.inv(np.conj(np.swapaxes(H, 1, 2)) @ H), axis1=1, axis2=2))
        mc = float(np.mean(np.log2(1.0 + radio.rho_lin * Y)))
        th = digital_zf_spectral_efficiency(radio, np.random.RandomState(2026))
        good = abs(mc - th) / th < 0.01 and abs(Y.mean() - (Nr - N + 1)) / (Nr - N + 1) < 0.01
        ok &= good
        lines.append(f"  (1) {rho_db:+5.0f} dB: c_ZF Gamma {th:.4f} vs explicit ZF {mc:.4f} bit/tone-use, "
                     f"E[Y] {Y.mean():.2f} vs Nr-N+1 = {Nr - N + 1}  {'OK' if good else 'BAD'}")
    one = RadioConfig(N=1)
    r1 = digital_rates(one)
    good = r1["U_zf"] == r1["U"] and r1["A_zf"] == r1["A"]
    ok &= good
    lines.append(f"  (2) N=1: R_ZF = {r1['U_zf']/1e6:.4f} Mbit/s == R_OFDMA = {r1['U']/1e6:.4f} Mbit/s  "
                 f"{'OK' if good else 'BAD'}")
    rates = digital_rates(base)
    dims = profiled_dims(base.batch_size)[2]
    got = {"c_ZF": rates["c_zf_U"], "sflv1_zf_s": uplink_time_per_round("digital_sflv1_zf", dims, base, 5, rates),
           "hybrid_zf_s": uplink_time_per_round("hybrid_zf_aircomp", dims, base, 5, rates)}
    golden = {"c_ZF": 11.75, "sflv1_zf_s": 8.31, "hybrid_zf_s": 6.92}
    good = all(abs(got[k] - v) / v < 0.01 for k, v in golden.items())
    ok &= good
    lines.append("  (3) default environment: " + ", ".join(f"{k} = {got[k]:.3f} [golden {v}]"
                                                         for k, v in golden.items()) + f"  {'OK' if good else 'BAD'}")
    if verbose:
        print("=== digital multi-user ZF rate ===")
        print("\n".join(lines))
        print(f"  ZF_RATE: {'PASS' if ok else 'FAIL'}")
    return ok


def verify_path_gains(verbose: bool = True) -> bool:
    """Unequal path gains, timing side:
      (1) explicit all-zero offsets == equal gains, bitwise;
      (2) the digital rates equal those of an equal-gain link at the weakest client's SNR
          (rho + g_min dB), i.e. the max over clients of Eq. 12;
      (3) analog phases are unchanged (AirSFL's labels are digital and follow the weakest
          client too); digital times grow with the spread;
      (4) the offsets are stratified: weakest client -spread (1 - 1/N)/2 dB, median 0 dB,
          same multiset for every seed, different client order."""
    ok = True
    lines = []
    base = RadioConfig()
    dims = profiled_dims(base.batch_size)[2]
    r0 = digital_rates(base)
    rz = digital_rates(RadioConfig(gains_db=(0.0,) * base.N))
    good = all(r0[k] == rz[k] for k in r0)
    ok &= good
    lines.append(f"  (1) all-zero offsets == equal gains (bitwise): {'OK' if good else 'BAD'}")
    prev = {m: uplink_time_per_round(m, dims, base, 5, r0) for m in METHODS}
    b0 = uplink_time_breakdown("airsfl", dims, base, 5, r0)
    for spread in (10.0, 20.0, 40.0):
        g = path_gain_offsets_db(base.N, spread, seed=11)
        rg = RadioConfig(gains_db=g)
        rates = digital_rates(rg)
        ref = digital_rates(RadioConfig(rho_db=base.rho_db + min(g)))
        good = all(abs(rates[k] - ref[k]) <= 1e-9 * abs(ref[k]) for k in ("U", "A", "U_zf", "A_zf"))
        t = {m: uplink_time_per_round(m, dims, rg, 5, rates) for m in METHODS}
        b = uplink_time_breakdown("airsfl", dims, rg, 5, rates)
        good &= b["activation"] == b0["activation"] and b["aggregation"] == b0["aggregation"]  # analog phases
        good &= t["aircomp_fl"] == prev["aircomp_fl"]
        good &= all(t[m] > prev[m] for m in ("digital_sflv1", "digital_sflv1_zf", "sun_fdma_aircomp",
                                               "hybrid_zf_aircomp", "digital_fedavg"))
        prev = {m: (t[m] if m != "aircomp_fl" else prev[m]) for m in METHODS}
        other = path_gain_offsets_db(base.N, spread, seed=22)
        good &= sorted(other) == sorted(g) and other != g and abs(float(np.median(g))) < 1e-9
        good &= abs(min(g) + spread * (1 - 1 / base.N) / 2) < 1e-6
        ok &= good
        lines.append(f"  spread {spread:4.0f} dB: weakest {min(g):6.2f} dB | SFL-V1 {t['digital_sflv1']:7.1f} s, "
                     f"SFL-V1 (ZF) {t['digital_sflv1_zf']:6.2f} s, AirSFL {t['airsfl']:.3f} s per round | "
                     f"(2)-(4) {'OK' if good else 'BAD'}")
    if verbose:
        print("=== unequal path gains (timing) ===")
        print("\n".join(lines))
        print(f"  PATH_GAINS_TIMING: {'PASS' if ok else 'FAIL'}")
    return ok


def verify_fp16(verbose: bool = True) -> bool:
    """FP16 digital payload (q = 16, the default of the experiments):
      (1) every digital phase of every method is exactly half its FP32 value, while the labels
          (integer class indices) and the analog phases are unchanged;
      (2) the same holds with unequal path gains (the digital links are paced by the weakest
          client for either q)."""
    ok = True
    lines = []
    dims = profiled_dims(16)[2]
    for name, gains in (("equal gains", None), ("20 dB spread", path_gain_offsets_db(RadioConfig().N, 20.0, 11))):
        r32, r16 = RadioConfig(gains_db=gains), RadioConfig(gains_db=gains, q_bits=16)
        rates = digital_rates(r32)
        good = True
        for m in METHODS:
            b32 = uplink_time_breakdown(m, dims, r32, 5, rates)
            b16 = uplink_time_breakdown(m, dims, r16, 5, rates)
            analog = m in ("airsfl", "aircomp_fl")
            for ph in ("activation", "aggregation"):
                digital_phase = not analog and not (ph == "aggregation" and "aircomp" in m)
                want = b32[ph] / 2 if digital_phase else b32[ph]
                good &= abs(b16[ph] - want) <= 1e-12 * max(1.0, want)
            good &= b16["labels"] == b32["labels"]
        ok &= good
        t32 = uplink_time_per_round("digital_sflv1", dims, r32, 5, rates)
        t16 = uplink_time_per_round("digital_sflv1", dims, r16, 5, rates)
        lines.append(f"  {name}: digital phases halved, labels and analog phases unchanged for all "
                     f"{len(METHODS)} methods (SFL-V1 {t32:.1f} -> {t16:.1f} s per round)  {'OK' if good else 'BAD'}")
    if verbose:
        print("=== FP16 digital payload (timing) ===")
        print("\n".join(lines))
        print(f"  FP16_TIMING: {'PASS' if ok else 'FAIL'}")
    return ok


def environment_summary(radio: Optional[RadioConfig] = None, stage: int = 2, tau: int = 5) -> None:
    """Print the per-round uplink budget of the default environment."""
    radio = radio or RadioConfig()
    rates = digital_rates(radio)
    dims = profiled_dims(radio.batch_size)[stage]
    print(f"=== environment: N={radio.N}, Nr_M={radio.Nr}, Nr_F={radio.Nr_F}, S={radio.S} "
          f"(W={radio.W_hz/1e6:.2f} MHz, S_n={radio.Sn:g}), rho={radio.rho_db:g} dB, "
          f"eps=({radio.eps_D},{radio.eps_U},{radio.eps_A}), cut {stage}, tau={tau} ===")
    if radio.unequal_gains:
        print(f"  unequal path gains: {min(radio.gains_db):.2f} ... {max(radio.gains_db):.2f} dB around the "
              f"reference (median); digital rates below are the weakest client's")
    print(f"  OFDMA: c_D = {rates['c_U']:.2f} bit/tone-use on {radio.Sn:g} tones, goodput per client = "
          f"{rates['U']/1e6:.3f} Mbit/s")
    print(f"  ZF:    c_ZF = {rates['c_zf_U']:.2f} bit/tone-use on {radio.S} tones, goodput per client = "
          f"{rates['U_zf']/1e6:.3f} Mbit/s")
    for m in METHODS:
        b = uplink_time_breakdown(m, dims, radio, tau, rates)
        print(f"  {m:17s} act {b['activation']:9.3f}  labels {b['labels']:.4f}  agg {b['aggregation']:9.3f}"
              f"  total {b['total']:9.3f} s/round")


if __name__ == "__main__":
    verify_reference_example()
    print()
    verify_limit_cases()
    print()
    verify_zf()
    print()
    verify_path_gains()
    print()
    verify_fp16()
    print()
    environment_summary()
