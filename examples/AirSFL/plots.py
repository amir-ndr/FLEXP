"""
examples/AirSFL/plots.py: figures + tables for the AirSFL study from the CSVs written by
run_airsfl.py (works on whatever results exist; missing experiments are skipped).

Two time axes, both carried by every CSV row (downlinks ideal and untimed):
  * accumulated UPLINK communication time (paper Eq. 25-26, 28): activations + labels
    (client -> M-server) and model differences (client -> F-server);
  * TRAINING time = uplink + computation (flsim.airsfl.compute: client FP/BP on the
    slowest client, suffix FP/BP of all N copies on the shared M-server; FL: client
    FP/BP of the whole model). Client/server FLOPS can be overridden here
    (--client-tflops LO HI --server-tflops F); computation is recomputed exactly from
    the stored FLOP counts and client draws -- no retraining.

Targets: prespecified VALIDATION accuracies (--targets, e.g. 0.75 0.8; the first is the
primary target of the bar figures). Without --targets, the rule A* = integer % below 95%
of the error-free reference's final validation accuracy is used. Attainment is per seed:
the first evaluated checkpoint (round >= 1) with val_acc >= A. Failures are reported
(k/n seeds) and drawn as "not reached", never given a finite time; times are averaged
over the seeds that reached the target and speed-ups are PAIRED per seed. Curves show
TEST accuracy; target markers sit at the checkpoint where VALIDATION reached the target.

Digital learning does not depend on SNR, cut, efficiency or antenna count (ideal
decoding): for those sweeps its time axes are recomputed analytically from the same runs.

Digital multi-user ZF baselines (stronger digital references than OFDMA): "Digital SFL-V1
(ZF)" and "Hybrid ZF-AirComp SFL" learn exactly like digital SFL-V1 / Hybrid FDMA-AirComp
SFL (reliable digital links either way; same AirComp stream), so they are derived from those
runs with the uplink time recomputed at the ZF rate (flsim.airsfl.timing; verified against
real runs in flsim.airsfl.checks). --methods selects which methods are drawn.

Digital payload (--digital-q, default 16 = FP16): the methods that upload tensors digitally
(digital SFL-V1, the hybrid, digital FedAvg and their ZF variants) are drawn from their runs
with that payload (FP16: really rounded uploads, timed with q = 16 bits) -- FP16 -> figures/,
tables/; FP32 (--digital-q 32) -> figures_q32/, tables_q32/; --digital-q 16 32 draws both.
FP16 runs come from each experiment's folder and, for main / snr, also from fp16/ (--exp fp16).
AirSFL and AirComp-FL upload only labels digitally: the same runs serve both payloads. In both
sets the exact FP32 digital SFL-V1 is the error-free reference and the FP32 FedAvg the
reference of the target rule (the drawn runs if the FP32 ones are absent). A point whose
payload runs are missing is listed and left out -- never filled with the other payload's
learning. Source-equivalent MB are counted at the drawn payload precision.

Only CSVs of the current schema with the calibrated initial LR and augmentation setting
(results/lr/chosen_lr.json, or --lr / --augment) and one epoch budget are loaded.

Figures -> <results>/figures/, tables -> <results>/tables/; fig1e, fig2c, fig2_snr and fig7 are also
saved panel by panel (one file per subfigure) into <results>/paper_results/{1e,2c,2_snr,7}/ (--panels):
  fig1_acc_vs_uplink_time     test accuracy vs accumulated uplink time
  fig1c_acc_vs_training_time  test accuracy vs training time (uplink + computation)
  fig1b_acc_vs_epochs         test accuracy vs global-epoch equivalents (learning only)
  fig1e_acc_vs_uplink_time_linear  Sun et al.-style curves on a linear time axis, 20 dB and low SNR
  fig2_snr                    final accuracy, uplink / training time to target vs SNR
  fig2b_airsfl_snr_curves     AirSFL accuracy vs uplink time at every SNR (+ error-free AirSFL,
                              digital SFL-V1 (ZF / OFDMA) at the lowest SNR and at 20 dB)
  fig2c_acc_at_budget         test accuracy reached within a fixed time budget vs SNR
  fig3_uplink_breakdown       per-round uplink phases vs N (analytic)
  fig3b_uplink_vs_cut         per-round uplink time vs cut (analytic; only with --also fig3b)
  fig3c_round_time            per-round uplink + computation, and its composition (analytic)
  fig4_comm_overhead          source-equivalent MB vs airtime per round; GB to target
  fig5_nr                     AirSFL vs M-server antennas (if run)
  fig6_time_to_target         uplink and training time to the primary target (bars)
  fig7_nsweep                 N in {20,30,40}: time to a common target (if run)
  fig8_efficiency             time to target vs analog efficiency (0.4/0.6/0.7, ideal 1.0)
  fig9_cut_tau                time to target vs cut and vs tau (if run)
  fig10_compute_regimes       time to target: uplink only / edge-GPU / IoT-CPU (Sun et al.) devices
  fig11_path_gains            unequal path gains (exp `pathloss`; only with --also fig11): accuracy, time to
                              target, s/round vs spread (+ fig11b curves, table_pathloss)
  fig12_digital_transport     digital baselines: access (OFDMA / ZF) x payload (FP32 / FP16)
  fig2d_curves_all_snr_<axis>, fig11b_curves_all_spreads_<axis>
                              learning curves of all methods at every SNR / path-gain spread (--sweep-curves)
  fig1d_acc_vs_training_time_iot  accuracy vs training time with IoT-CPU devices
  tables: table_main, table_snr, table_budget, table_cuts, table_round_time, table_nsweep,
          table_efficiency, table_cut_tau  (.csv + .md + .tex)
"""

import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flsim.airsfl.compute import SPLIT_METHODS, compute_time_breakdown, split_flops
from flsim.airsfl.timing import (RadioConfig, digital_rates, path_gain_offsets_db, profiled_dims,
                                 source_equivalent_mb_per_round,
                                 uplink_time_breakdown, uplink_time_per_round)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIG = os.path.join(RESULTS, "figures")
TAB = os.path.join(RESULTS, "tables")
SCHEMA = 4
FILTER = {}                 # base_lr / augment / epochs_budget of the loaded campaign (set in main)
TARGETS = None              # prespecified validation targets (set in main)
COMPUTE = dict(client_tflops_lo=1.0, client_tflops_hi=2.0, server_tflops=20.0)   # table/analytic default
BUDGETS = {"uplink": None, "training": None}
BUDGET_MULTS = [1.0]        # fig2c: budgets = base budget x each multiplier (--budget-mult)
LINEAR = {"snrs": None, "xmax": None}   # fig1e: SNR panels and x range (--linear-snrs, --linear-xmax)
SWEEP_CURVES = {"axes": ["epochs", "uplink"], "xscale": "log"}   # fig2d / fig11b (--sweep-curves, --curves-xscale)
ALSO = set()                # optional figures (--also): fig3b (uplink vs cut), fig11 (path-gain experiment)
# figures also saved panel by panel (--panels) into <results>/paper_results/<key>/ (FP16 set; FP32: paper_results_q32)
PANELS = {"figs": {"1e", "2c", "2_snr", "7"}, "dir": None}

STYLE = {
    "airsfl":            dict(label="AirSFL (proposed)", color="#d62728", marker="o", lw=2.6, ls="-"),
    "aircomp_fl":        dict(label="AirComp-FL", color="#2ca02c", marker="D", lw=1.8, ls="-"),
    "hybrid_zf_aircomp": dict(label="Hybrid ZF-AirComp SFL", color="#8c564b", marker="X", lw=1.8, ls="-."),
    "sun_fdma_aircomp":  dict(label="Hybrid FDMA-AirComp SFL", color="#9467bd", marker="^", lw=1.8, ls="-"),
    "digital_sflv1_zf":  dict(label="Digital SFL-V1 (ZF)", color="#17becf", marker="P", lw=1.8, ls="-."),
    "digital_sflv1":     dict(label="Digital SFL-V1 (OFDMA)", color="#1f77b4", marker="s", lw=1.8, ls="-"),
    "digital_fedavg":    dict(label="Digital FedAvg (OFDMA)", color="#7f7f7f", marker="v", lw=1.6, ls="--"),
    "airsfl_errfree":    dict(label="AirSFL, error-free (upper bound)", color="#d62728", marker="*", lw=1.6, ls=":"),
    # sensitivity variant (fig12 / table_digital_transport; drawn elsewhere only via --methods)
    "digital_fedavg_zf":   dict(label="Digital FedAvg (ZF)", color="#bcbd22", marker="1", lw=1.4, ls="-."),
}
# computation presets, all recomputed from the stored FLOP counts and client draws (no retraining):
# (client TFLOPS lo, hi, M-server TFLOPS)
PRESETS = [
    ("uplink only (no computation)", None),
    ("edge GPU [AdaptSFL]: clients U[1,2] TFLOPS, server 20 TFLOPS", (1.0, 2.0, 20.0)),
    ("IoT CPU [Sun et al.]: 16 FLOP/cycle x U[0.1,2] GHz, server 320 GFLOPS", (0.0016, 0.032, 0.32)),
]
PART_NAME = {"iid": "IID", "dirichlet": "Non-IID (Dirichlet)"}   # alpha filled in from the data (main)
# families side by side: analog, hybrid (ZF / OFDMA activations), digital SFL (ZF / OFDMA), FedAvg
ALL_METHODS = ["airsfl", "aircomp_fl", "hybrid_zf_aircomp", "sun_fdma_aircomp", "digital_sflv1_zf",
               "digital_sflv1", "digital_fedavg"]
EXTRA_METHODS = ["digital_fedavg_zf"]
DEFAULT_METHODS = [m for m in ALL_METHODS if m != "digital_fedavg"]   # drawn by default (FedAvg: --methods)
ORDER = list(DEFAULT_METHODS)         # methods drawn (plots.py --methods)
# digital baselines derived from their parent runs (identical learning, multi-user ZF timing:
# all tones, concurrent streams separated by ZF)
DERIVED = {"digital_sflv1_zf": "digital_sflv1", "hybrid_zf_aircomp": "sun_fdma_aircomp",
           "digital_fedavg_zf": "digital_fedavg"}
DIGITAL = ("digital_sflv1", "digital_fedavg", "digital_sflv1_zf", "digital_fedavg_zf")   # SNR-independent learning
# ENV0["q_bits"] = digital payload of the drawn set (--digital-q; set per set in main)
ENV0 = dict(N=30, Nr=64, Nr_F=64, S=120, eps_D=0.6, eps_U=0.6, eps_A=0.6, rho_db=20.0, batch_size=16, q_bits=16)
FP16_PARENTS = ("digital_sflv1", "sun_fdma_aircomp", "digital_fedavg")   # trained methods with digital payloads
PAYLOAD_METHODS = FP16_PARENTS + tuple(DERIVED)       # drawn from their runs with the chosen payload
REFS = ("errfree_ref", "target_ref")                  # FP32 digital SFL-V1 / FedAvg kept as references
TAU0, CUT0, B0 = 5, 2, 16
AXES = {"uplink": ("uplink_s", "Accumulated uplink communication time (s)"),
        "training": ("training_time_s", "Training time: uplink + computation (s)")}

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "legend.fontsize": 9, "figure.dpi": 110})


# ---------------------------------------------------------------------------
# loading, time recomputation
# ---------------------------------------------------------------------------

def _read(exp):
    dfs = []
    for f in sorted(glob.glob(os.path.join(RESULTS, exp, "*.csv"))):
        d = pd.read_csv(f)
        if "schema" not in d or int(d["schema"].iloc[0]) != SCHEMA:
            continue                                    # older experiment generation: ignore
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _keep_alpha(df, alpha):
    """Drop Dirichlet runs of another concentration (IID rows have no alpha and are kept)."""
    if df.empty or alpha is None or "dirichlet_alpha" not in df:
        return df
    a = pd.to_numeric(df["dirichlet_alpha"], errors="coerce")
    return df[(df.partition != "dirichlet") | np.isclose(a.fillna(-1.0), alpha)]


def _filter(df, skip=()):
    if df.empty:
        return df
    keep = np.ones(len(df), dtype=bool)
    for col, val in FILTER.items():
        if val is None or col not in df or col in skip:
            continue
        keep &= np.isclose(df[col].astype(float), float(val)) if isinstance(val, float) else (df[col] == val).values
    return df[keep]


_RATE_CACHE = {}


def _rates(radio):
    # digital rates depend on the gains through the weakest client (Eq. 12); the payload
    # precision q enters only the times
    gains = tuple(sorted(radio.gains_db)) if radio.unequal_gains else None
    key = (radio.N, radio.Nr, radio.Nr_F, radio.S, radio.eps_D, radio.rho_db, gains)
    if key not in _RATE_CACHE:
        _RATE_CACHE[key] = digital_rates(radio)
    return _RATE_CACHE[key]


def _radio(**kw):
    return RadioConfig(**{**ENV0, **kw})


def _gains_of(r0):
    """Per-client path gains (dB) a run was trained with; None for equal gains / older CSVs."""
    g = r0.get("path_gains_db") if hasattr(r0, "get") else None
    return tuple(json.loads(g)) if isinstance(g, str) and g.strip() else None


def _q_of(r0):
    """Digital payload bits a run was trained / timed with (32 for older CSVs)."""
    q = r0.get("q_bits") if hasattr(r0, "get") else None
    return 32 if q is None or (isinstance(q, float) and math.isnan(q)) else int(q)


def _radio_of(run, **ov):
    r0 = run.iloc[0]
    kw = dict(N=int(r0.N), Nr=int(r0.Nr), Nr_F=int(r0.Nr_F), S=int(r0.S), eps_D=float(r0.eps_D),
              eps_U=float(r0.eps_U), eps_A=float(r0.eps_A), rho_db=float(r0.rho_db), batch_size=int(r0.B),
              gains_db=_gains_of(r0), q_bits=_q_of(r0))
    kw.update(ov)
    return RadioConfig(**kw)


def _f_min(r0):
    lo, hi = COMPUTE["client_tflops_lo"], COMPUTE["client_tflops_hi"]
    return (lo + (hi - lo) * float(r0.u_client_min)) * 1e12


def _compute_per_round(method, run, stage=None):
    """Computation seconds per round of a run (optionally for another cut)."""
    r0 = run.iloc[0]
    if stage is None:
        fl = {k: float(r0[f"flops_{k}"]) for k in ("client_fp", "client_bp", "server_fp", "server_bp")}
    else:
        fl = split_flops(stage if method in SPLIT_METHODS else None)
    return compute_time_breakdown(method, fl, int(r0.N), int(r0.B), int(r0.tau), _f_min(r0),
                                  COMPUTE["server_tflops"] * 1e12)


def _set_times(run, ul, comp):
    """Replace a run's per-round uplink/compute totals and the cumulative time axes."""
    k = run["round"]
    return run.assign(ul_s_per_round=ul, uplink_s=k * ul, cumulative_ul_s=k * ul,
                      compute_s_per_round=comp, cumulative_compute_s=k * comp,
                      e2e_s_per_round=ul + comp, training_time_s=k * (ul + comp))


def _retime(method, run, dims=None, stage=None, **ov):
    """Same training run, time axes (and uplink phases) recomputed for `method` under radio
    overrides / another cut. Several runs (seeds) are retimed one by one: each keeps its own
    computation time (its own slowest-client draw)."""
    if "run_id" in run and run.run_id.nunique() > 1:
        return _per_run(run, lambda g: _retime(method, g, dims, stage, **ov))
    r0 = run.iloc[0]
    radio = _radio_of(run, **ov)
    dims = dims or {"d_c": int(r0.d_c), "d_s": int(r0.d_s), "d_a": int(r0.d_a)}
    b = uplink_time_breakdown(method, dims, radio, int(r0.tau), _rates(radio))
    comp = _compute_per_round(method, run, stage)["total"] if stage is not None else float(r0.compute_s_per_round)
    out = _set_times(run, b["total"], comp).assign(activation_ul_s=b["activation"], labels_ul_s=b["labels"],
                                                   aggregation_ul_s=b["aggregation"])
    for k in ("rho_db", "eps_D", "eps_U", "eps_A"):      # the retimed run now describes that radio setting
        if k in ov:
            out[k] = ov[k]
    return out


def _derive(df):
    """Add the derived digital baselines (multi-user ZF) to a set of runs. Their learning
    is exactly that of the parent run (the digital links are reliable either way, and the hybrids
    share the parent's AirComp stream), so every parent run is copied and only its uplink time is
    recomputed for that run's own N, antennas, SNR, gains, payload precision, cut and tau
    (flsim.airsfl.checks trains both and verifies this)."""
    if df.empty:
        return df
    out = [df]
    for der_m, parent in DERIVED.items():
        src = df[df.method == parent]
        if src.empty or (df.method == der_m).any():
            continue
        der = _per_run(src, lambda g: _retime(der_m, g))
        out.append(der.assign(method=der_m, label=STYLE[der_m]["label"],
                              run_id=der.run_id.astype(str) + "|" + der_m))
    return pd.concat(out, ignore_index=True)


_derive_zf = _derive                     # former name


def _q_col(df):
    """Digital payload bits of every row (32 for CSVs written before the column existed)."""
    if "q_bits" not in df:
        return pd.Series(32, index=df.index)
    return pd.to_numeric(df["q_bits"], errors="coerce").fillna(32).astype(int)


def _select_payload(d, q):
    """The data sets drawn with a q-bit digital payload. Methods that upload tensors digitally
    (PAYLOAD_METHODS) keep only their q-bit runs: for q = 16 those in each experiment's folder
    plus, for main / snr, the runs of --exp fp16 (20 dB -> main, other SNRs -> snr); a run found
    in two folders (same run_id) counts once. AirSFL / AirComp-FL keep their runs (trained with
    the FP32 identity; their training and timing do not depend on q). Each experiment's FP32
    digital SFL-V1 / FedAvg runs are added as "errfree_ref" / "target_ref" -- the exact
    error-free reference and the reference of the target rule -- so both payload sets share
    the same targets and the same upper bound."""
    fp = d.get("fp16", pd.DataFrame())
    out = {}
    for e, df in d.items():
        if e == "fp16" or df.empty:
            if e != "fp16":
                out[e] = df
            continue
        qc = _q_col(df)
        pay = df.method.isin(PAYLOAD_METHODS)
        other = df[~pay]
        oq = _q_col(other)
        # an analog method trained with q = 16 (older runner) duplicates its q = 32 run: keep one
        other = other[(oq == 32) | ~other.method.isin(set(other.method[oq == 32]))]
        parts = [df[pay & (qc == q)], other]
        if q == 16 and not fp.empty and e in ("main", "snr"):
            f = fp[fp.method.isin(PAYLOAD_METHODS) & (_q_col(fp) == 16)]
            parts.append(f[f.rho_db == 20.0] if e == "main" else f[f.rho_db != 20.0])
        exact = df[qc == 32]
        for m, ref in (("digital_sflv1", "errfree_ref"), ("digital_fedavg", "target_ref")):
            r = exact[exact.method == m]
            if not r.empty:
                parts.append(r.assign(method=ref, run_id=r.run_id.astype(str) + "|" + ref))
        parts = [x for x in parts if not x.empty]
        sel = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
        out[e] = sel.drop_duplicates(subset=["run_id", "round"]).reset_index(drop=True)
    return out


_KEY = ["method", "partition", "seed", "rho_db", "N", "Nr", "cut", "tau", "path_gain_spread_db"]


def _report_missing(d, sel, q):
    """List the trained digital-payload runs that exist with another payload but not with q:
    those points are left out of the q-bit figures (never replaced by the other payload)."""
    groups = {}
    for e, df in d.items():
        if e == "fp16" or df.empty:
            continue
        cols = [c for c in _KEY if c in df]
        keys = lambda x: set(map(tuple, x[cols].drop_duplicates().itertuples(index=False, name=None)))
        have = sel[e][sel[e].method.isin(FP16_PARENTS)] if not sel[e].empty else sel[e]
        want = df[df.method.isin(FP16_PARENTS) & (_q_col(df) != q)]
        for k in sorted(keys(want) - (keys(have) if not have.empty else set())):
            r = dict(zip(cols, k))
            groups.setdefault((e, r["method"]), []).append(
                f"{r['partition']} s{int(r['seed'])} {float(r['rho_db']):g} dB N={int(r['N'])}"
                + (f" spread {float(r['path_gain_spread_db']):g} dB" if float(r.get("path_gain_spread_db", 0) or 0)
                   else ""))
    if not groups:
        print(f"[plots] FP{q} digital payload: every digital-payload run has its FP{q} version")
    else:
        print(f"[plots] FP{q} digital payload: these runs exist only with another payload, so they are LEFT OUT of "
              f"the FP{q} figures (train them with run_airsfl.py --digital-q {q}; finished runs are skipped):")
        for (e, m), pts in sorted(groups.items()):
            print(f"    {e:9s} {m:17s} {len(pts):3d} run(s): " + "; ".join(pts[:6]) + (" ..." if len(pts) > 6 else ""))
    main = sel.get("main", pd.DataFrame())
    if q == 32 or main.empty:
        return                                  # FP32 set: a missing FP32 run is already listed above
    seeds = lambda m: set(zip(main.partition[main.method == m], main.seed[main.method == m].astype(int)))
    for ref, src, what in (("errfree_ref", "digital_sflv1", "error-free curve"),
                           ("target_ref", "digital_fedavg", "target rule")):
        gap = sorted(seeds("airsfl") - seeds(ref))
        if gap:
            print(f"[plots] FP32 {src} run (main) missing for {gap}: the {what} uses the other seeds "
                  f"(train it with run_airsfl.py --digital-q 32)")


def _payload_of(m, run):
    """Digital payload of a method's uploads as recorded by its run."""
    if m in PAYLOAD_METHODS:
        return f"FP{_q_of(run.iloc[0])}"
    return "labels only" if m in SPLIT_METHODS else "none"


def _pl():
    return f"digital FP{ENV0['q_bits']}"


def _mb_per_round(run):
    """Source-equivalent uplink MB per round of a run, counted at the drawn digital payload
    precision (the CSV column mb_per_round is the FP32 reference)."""
    r0 = run.iloc[0]
    dims = {"d_c": int(r0.d_c), "d_s": int(r0.d_s), "d_a": int(r0.d_a)}
    return source_equivalent_mb_per_round(r0.method, dims, _radio_of(run), int(r0.tau), q_ref=ENV0["q_bits"])


def _with_compute(run, preset):
    """One run's time axes under a computation preset (None = uplink only)."""
    r0 = run.iloc[0]
    ul = float(r0.ul_s_per_round)
    if preset is None:
        return _set_times(run, ul, 0.0)
    lo, hi, fs = preset
    fl = {k: float(r0[f"flops_{k}"]) for k in ("client_fp", "client_bp", "server_fp", "server_bp")}
    f_min = (lo + (hi - lo) * float(r0.u_client_min)) * 1e12
    c = compute_time_breakdown(r0.method, fl, int(r0.N), int(r0.B), int(r0.tau), f_min, fs * 1e12)
    return _set_times(run, ul, c["total"])


def _per_run(df, fn):
    return pd.concat([fn(g) for _, g in df.groupby("run_id")], ignore_index=True) if not df.empty else df


def _airsfl_error_free(main, scheme):
    """Noise-free AirSFL = digital SFL-V1's learning trajectory (bitwise identical, see checks)
    placed on AirSFL's per-round uplink time: the error-free upper bound. The computation per
    round of the two is identical for a seed (same cut, same client draws), so each seed keeps
    its own. `main` holds one operating SNR (20 dB, or the one chosen with --snr). The exact FP32
    digital SFL-V1 runs ("errfree_ref", _select_payload) are used whatever the drawn payload."""
    ref = "errfree_ref" if (main.method == "errfree_ref").any() else "digital_sflv1"
    dig = main[(main.partition == scheme) & (main.method == ref)]
    air = main[(main.partition == scheme) & (main.method == "airsfl")]
    if dig.empty or air.empty:
        return pd.DataFrame()
    ul = float(air.ul_s_per_round.iloc[0])            # seed-independent analog airtime (+ tiny digital labels)
    return _per_run(dig, lambda g: _set_times(g, ul, float(g.compute_s_per_round.iloc[0])))


def _main_at(main, snr, rho):
    """The operating-point data set at SNR rho, from the CSVs: analog methods from the SNR sweep
    at rho (ZF hybrid derived from its runs), digital methods (SNR-independent learning)
    retimed at rho, plus the error-free / target references. rho = 20 dB is `main` itself."""
    if rho == 20.0:
        return main
    frames = [r for scheme in _schemes(main) for m in ALL_METHODS
              for r in [_snr_runs(main, snr, scheme, m, rho)] if not r.empty]
    frames.append(main[main.method.isin(REFS)])
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _apply_compute(df):
    """Recompute every run's computation / training time under the COMPUTE setting."""
    if df.empty:
        return df
    out = []
    for _, run in df.groupby("run_id"):
        c = _compute_per_round(run.method.iloc[0], run)
        run = run.assign(client_fp_s=c["client_fp"], client_bp_s=c["client_bp"], server_fp_s=c["server_fp"],
                         server_bp_s=c["server_bp"])
        out.append(_set_times(run, float(run.ul_s_per_round.iloc[0]), c["total"]))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# seed statistics, targets
# ---------------------------------------------------------------------------

def _curve(df):
    """Mean over paired seeds at each evaluation round (numeric columns)."""
    if df.empty:
        return df
    return df.select_dtypes("number").groupby("round", as_index=False).mean().sort_values("round")


def _seed_runs(df):
    return [g.sort_values("round") for _, g in df.groupby("seed")] if not df.empty else []


def _nseeds(df):
    return int(df.seed.nunique()) if not df.empty else 0


def _seed_txt(n):
    return f"mean of {n} seeds, band = seed min-max" if n > 1 else "1 seed"


def rule_target(df, scheme):
    """95% rule on the error-free reference: the FP32 FedAvg runs, else the FP32 digital SFL-V1 runs
    (identical learning: exact transport makes SFL-V1 full-model SGD), else the drawn ones."""
    sub = df[df.partition == scheme]
    have = set(sub.method)
    m = next((k for k in ("target_ref", "digital_fedavg", "errfree_ref", "digital_sflv1") if k in have), None)
    ref = sub[sub.method == m] if m else sub.iloc[0:0]
    if ref.empty:
        return None
    return math.floor(100 * 0.95 * float(_curve(ref)["val_acc"].iloc[-1])) / 100.0


def targets_for(df, scheme):
    if TARGETS:
        return list(TARGETS)
    A = rule_target(df, scheme)
    return [A] if A is not None else []


def primary_target(df, scheme):
    t = targets_for(df, scheme)
    return t[0] if t else None


def time_to_target(run_df, A, xcol):
    """First evaluated round (>= 1) with val_acc >= A -> (time, round)."""
    hit = run_df[(run_df.val_acc >= A) & (run_df["round"] >= 1)].sort_values("round")
    if hit.empty:
        return None, None
    return float(hit[xcol].iloc[0]), int(hit["round"].iloc[0])


def ttt(df, A, xcol="uplink_s"):
    """Per-seed attainment: n seeds, hits, per-seed (time, round), mean/min/max over hits."""
    per = {}
    for g in _seed_runs(df):
        t, k = time_to_target(g, A, xcol) if A is not None else (None, None)
        per[int(g.seed.iloc[0])] = (t, k)
    hits = {s: v for s, v in per.items() if v[0] is not None}
    out = {"n": len(per), "hit": len(hits), "per_seed": per, "t": np.nan, "tmin": np.nan, "tmax": np.nan,
           "k": np.nan, "epochs": np.nan}
    if hits:
        ts = [v[0] for v in hits.values()]
        ks = [v[1] for v in hits.values()]
        out.update(t=float(np.mean(ts)), tmin=float(min(ts)), tmax=float(max(ts)), k=float(np.mean(ks)))
        r0 = df.iloc[0]
        out["epochs"] = out["k"] * int(r0.N) * int(r0.tau) * int(r0.B) / 45000.0
    return out


def paired_speedup(s_ref, s_m):
    """Per-seed ratios t_ref / t_m over seeds where both reached the target."""
    r = [s_ref["per_seed"][k][0] / s_m["per_seed"][k][0] for k in s_m["per_seed"]
         if k in s_ref["per_seed"] and s_ref["per_seed"][k][0] is not None and s_m["per_seed"][k][0] is not None]
    return (float(np.mean(r)), float(min(r)), float(max(r)), len(r)) if r else (np.nan, np.nan, np.nan, 0)


def final_acc(df, col="test_acc"):
    v = [100 * float(g[col].iloc[-1]) for g in _seed_runs(df)]
    return (float(np.mean(v)), float(min(v)), float(max(v))) if v else (np.nan, np.nan, np.nan)


def acc_at_budget(df, T, xcol):
    """Per-seed TEST accuracy of the last checkpoint whose time <= T (mean, min, max)."""
    b = budget_point(df, T, xcol)
    return b["a"], b["lo"], b["hi"]


def budget_point(df, T, xcol):
    """Accuracy within a time budget T, per seed: TEST accuracy of the last evaluated checkpoint
    with time <= T. Also returns how many rounds fit into T and the last evaluated round, and
    `evaluated` = False when no seed has a trained checkpoint (round >= 1) inside T -- the
    value is then only the initial model's accuracy, not the (unevaluated) model at T."""
    acc, last, fit = [], [], []
    for g in _seed_runs(df):
        ok = g[g[xcol] <= T]
        if ok.empty:
            continue
        acc.append(100 * float(ok.test_acc.iloc[-1]))
        last.append(int(ok["round"].iloc[-1]))
        per_round = float(g[xcol].iloc[-1]) / max(1, int(g["round"].iloc[-1]))
        fit.append(min(int(g["round"].iloc[-1]), int(T // per_round)) if per_round > 0 else int(g["round"].iloc[-1]))
    if not acc:
        return {"a": np.nan, "lo": np.nan, "hi": np.nan, "last_round": np.nan, "rounds_in_budget": np.nan,
                "evaluated": False}
    return {"a": float(np.mean(acc)), "lo": float(min(acc)), "hi": float(max(acc)),
            "last_round": float(np.mean(last)), "rounds_in_budget": float(np.mean(fit)),
            "evaluated": max(last) >= 1}


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[fig] {name}")


def _export_panels(fig, key, named_axes, hollow_note=False):
    """Also save every panel of a finished figure as its own file, <results>/<dir>/<key>/<name>.png
    (+ .pdf): a copy of the figure keeps only that panel, exactly as drawn (title, axis labels,
    markers), without the figure-wide title and legend, and gets its own legend below the axes."""
    if PANELS["dir"] is None or key not in PANELS["figs"]:
        return
    import pickle
    out = os.path.join(RESULTS, PANELS["dir"], key)
    os.makedirs(out, exist_ok=True)
    blob = pickle.dumps(fig)
    n = 0
    for ax0, name in named_axes:
        if not ax0.has_data():
            continue                                  # empty panel (e.g. no runs for that partition)
        k = fig.axes.index(ax0)
        f = pickle.loads(blob)
        ax = f.axes[k]
        for other in [a for a in f.axes if a is not ax]:
            f.delaxes(other)
        f.legends.clear()
        if getattr(f, "_suptitle", None) is not None:
            f._suptitle.set_visible(False)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        h, lab = ax.get_legend_handles_labels()
        if hollow_note and any(ln.get_linestyle() == "None" and str(ln.get_markerfacecolor()).lower() in ("white", "w")
                               for ln in ax.get_lines()):
            h.append(plt.Line2D([], [], marker="o", mfc="white", mec="k", ls=""))
            lab.append("hollow marker: no evaluated checkpoint within the budget (initial model)")
        if h:                                         # legend just below the x-axis label, never on the data
            f.canvas.draw()
            y0 = ax.get_tightbbox(f.canvas.get_renderer()).transformed(ax.transAxes.inverted()).y0
            ax.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, y0 - 0.02), ncol=2, fontsize=8,
                      frameon=False)
        for ext in ("png", "pdf"):
            f.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
        plt.close(f)
        n += 1
    print(f"[panels] {key}: {n} panels -> {out}")


def _write_table(df, name, floatfmt=".4g"):
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(os.path.join(TAB, f"{name}.csv"), index=False)
    with open(os.path.join(TAB, f"{name}.md"), "w") as f:
        try:
            f.write(df.to_markdown(index=False, floatfmt=floatfmt))
        except ImportError:
            f.write(df.to_string(index=False))
    with open(os.path.join(TAB, f"{name}.tex"), "w") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.4g}", escape=True))
    print(f"[table] {name}")


def _plot_curve(ax, df, st, xcol, label=True, ls=None, color=None, lw=None, ycol="test_acc", first_round=1,
                markers=False):
    """Seed-mean curve (rounds < first_round dropped; round 0 has time 0, which a log axis
    cannot show) with a min-max seed band when >1 seed."""
    run = _curve(df)
    run = run[run["round"] >= first_round]
    if run.empty:
        return run
    c = color or st["color"]
    mk = dict(marker=st["marker"], markevery=max(1, len(run) // 8), ms=5) if markers else {}
    ax.plot(run[xcol], 100 * run[ycol], color=c, ls=ls or st["ls"], lw=lw or st["lw"],
            label=st["label"] if label is True else (label or None), **mk)
    if df.seed.nunique() > 1:
        g = df[df["round"] >= first_round].groupby("round")[ycol]
        lo, hi = g.min().sort_index(), g.max().sort_index()
        ax.fill_between(run[xcol].values, 100 * lo.values, 100 * hi.values, color=c, alpha=0.15, lw=0)
    return run


def _schemes(df):
    return [s for s in ("iid", "dirichlet") if not df.empty and s in set(df.partition)]


def _fig_legend(fig, axes, ncol=5):
    """One legend for the whole figure, below the panels (never covers a curve)."""
    handles = {}
    for ax in np.ravel(axes):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    if handles:
        fig.legend(handles.values(), handles.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.0),
                   ncol=min(ncol, len(handles)), fontsize=8.5, frameon=False)


def _mark_target(ax, run, A, st, xcol):
    """Marker at the first checkpoint whose VALIDATION accuracy reached A (y = test acc there)."""
    t, k = time_to_target(run, A, xcol) if A is not None else (None, None)
    if t is not None:
        ax.plot([t], [100 * float(run[run["round"] == k].test_acc.iloc[0])], marker=st["marker"],
                color=st["color"], ms=9, mec="k", zorder=5)


# ---------------------------------------------------------------------------
# Figure 1 / 1b / 1c + table_main  (roadmap B, C)
# ---------------------------------------------------------------------------

def fig1_and_table(main):
    schemes = _schemes(main)
    if not schemes:
        return
    ns = _nseeds(main)
    env = f"N=30, Nr=64 (M & F), W=1.8 MHz, {ENV0['rho_db']:g} dB, cut 2, tau=5, {_pl()}; {_seed_txt(ns)}"
    for axis, fname in (("uplink", "fig1_acc_vs_uplink_time"), ("training", "fig1c_acc_vs_training_time")):
        xcol, xlab = AXES[axis]
        fig, axes = plt.subplots(1, len(schemes), figsize=(6.3 * len(schemes), 4.7), squeeze=False)
        for ax, scheme in zip(axes[0], schemes):
            sub = main[main.partition == scheme]
            A = primary_target(main, scheme)
            for m in ORDER:
                dm = sub[sub.method == m]
                if dm.empty:
                    continue
                run = _plot_curve(ax, dm, STYLE[m], xcol)
                _mark_target(ax, run, A, STYLE[m], xcol)
            ef = _airsfl_error_free(main, scheme)          # Sun et al.'s "Error-free" benchmark
            if not ef.empty:
                _plot_curve(ax, ef, STYLE["airsfl_errfree"], xcol, lw=1.4)
            if A is not None:
                ax.plot([], [], ls="", marker="o", mfc="w", mec="k", ms=8,
                        label="marker: first checkpoint whose validation accuracy reaches the target")
            ax.set_xscale("log")
            ax.set_xlabel(xlab)
            ax.set_ylabel("Test accuracy (%)")
            ax.set_title(f"CIFAR-10, ResNet-18 — {PART_NAME[scheme]}"
                         + (f" (target {100 * A:.0f}%)" if A is not None else ""))
        _fig_legend(fig, axes, ncol=4)
        fig.suptitle(f"Accuracy vs {xlab.split(' (')[0].lower()} ({env})", y=1.02, fontsize=11)
        _save(fig, fname)

    fig, axes = plt.subplots(1, len(schemes), figsize=(6.3 * len(schemes), 4.7), squeeze=False)
    for ax, scheme in zip(axes[0], schemes):
        sub = main[main.partition == scheme]
        for m in ORDER:
            if m in DERIVED and DERIVED[m] in ORDER:
                continue                      # same learning curve as its OFDMA parent
            if not sub[sub.method == m].empty:
                _plot_curve(ax, sub[sub.method == m], STYLE[m], "epoch_equiv")
        ax.set_xlabel("Global-epoch equivalents")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"CIFAR-10, ResNet-18 — {PART_NAME[scheme]}")
    axes[0][0].legend(loc="lower right")
    fig.suptitle(f"Accuracy vs sample exposure ({env}); gaps = analog distortion\n"
                 f"(the ZF digital baselines learn exactly like their OFDMA counterparts)", y=1.04, fontsize=11)
    _save(fig, "fig1b_acc_vs_epochs")

    rows = []
    for scheme in schemes:
        sub = main[main.partition == scheme]
        for A in targets_for(main, scheme):
            st_ul = {m: ttt(sub[sub.method == m], A, "uplink_s") for m in ORDER if not sub[sub.method == m].empty}
            st_tr = {m: ttt(sub[sub.method == m], A, "training_time_s") for m in st_ul}
            for m in st_ul:
                dm = sub[sub.method == m]
                r0 = dm.iloc[0]
                acc, amin, amax = final_acc(dm)
                su = paired_speedup(st_ul["digital_sflv1"], st_ul[m]) if "digital_sflv1" in st_ul else (np.nan,) * 4
                sr = paired_speedup(st_tr["digital_sflv1"], st_tr[m]) if "digital_sflv1" in st_tr else (np.nan,) * 4
                xu = paired_speedup(st_ul[m], st_ul["airsfl"]) if "airsfl" in st_ul else (np.nan,) * 4
                xr = paired_speedup(st_tr[m], st_tr["airsfl"]) if "airsfl" in st_tr else (np.nan,) * 4
                mb = _mb_per_round(dm)
                rows.append({"partition": PART_NAME[scheme], "method": STYLE[m]["label"],
                             "digital payload": _payload_of(m, dm),
                             "SNR (dB)": float(r0.rho_db), "seeds": st_ul[m]["n"],
                             "UL s/round": float(r0.ul_s_per_round), "act s/round": float(r0.activation_ul_s),
                             "labels s/round": float(r0.labels_ul_s), "agg s/round": float(r0.aggregation_ul_s),
                             "compute s/round": float(r0.compute_s_per_round),
                             "client FP+BP s/round": float(r0.client_fp_s + r0.client_bp_s),
                             "server FP+BP s/round": float(r0.server_fp_s + r0.server_bp_s),
                             "training s/round": float(r0.e2e_s_per_round), f"MB/round (FP{ENV0['q_bits']})": mb,
                             "final test acc (%)": acc, "acc min": amin, "acc max": amax,
                             "target val acc (%)": 100 * A, "reached": f"{st_ul[m]['hit']}/{st_ul[m]['n']}",
                             "rounds to target": st_ul[m]["k"], "epochs to target": st_ul[m]["epochs"],
                             "UL time to target (s)": st_ul[m]["t"],
                             "training time to target (s)": st_tr[m]["t"],
                             f"MB to target (FP{ENV0['q_bits']})": st_ul[m]["k"] * mb,
                             "UL time / AirSFL (paired)": xu[0], "training time / AirSFL (paired)": xr[0],
                             "UL speed-up vs SFL-V1 (paired)": su[0], "training speed-up vs SFL-V1 (paired)": sr[0]})
    if rows:
        _write_table(pd.DataFrame(rows), "table_main")


# ---------------------------------------------------------------------------
# Figure 2 / 2b / 2c + table_snr / table_budget  (roadmap D)
# ---------------------------------------------------------------------------

def _snr_runs(main, snr, scheme, m, rho):
    base = main[(main.partition == scheme) & (main.method == m) & (main.rho_db == 20.0)]
    if m in DIGITAL:
        return base if base.empty else _retime(m, base, rho_db=rho)
    if rho == 20.0:
        return base
    if snr.empty:
        return snr
    return snr[(snr.partition == scheme) & (snr.method == m) & (snr.rho_db == rho)]


def _spread_runs(main, snr, pl, scheme, m, spread, rho=20.0):
    """Runs of method m with unequal path gains of nominal spread (dB) at SNR rho. Spread 0:
    the equal-gain runs. Analog methods: the `pathloss` runs. Digital methods (learning does
    not depend on the gains): the equal-gain runs retimed at the weakest client's rate."""
    base = _snr_runs(main, snr, scheme, m, rho)
    if not spread:
        return base
    if m in DIGITAL:
        if base.empty:
            return base
        r0 = base.iloc[0]      # stratified offsets: the weakest client (all digital timing needs) is seed-free
        return _retime(m, base, rho_db=rho, gains_db=path_gain_offsets_db(int(r0.N), spread, int(r0.seed)))
    if pl.empty or "path_gain_spread_db" not in pl:
        return pl.iloc[0:0]
    return pl[(pl.partition == scheme) & (pl.method == m) & (pl.rho_db == rho)
              & np.isclose(pl.path_gain_spread_db.astype(float), spread)]


def _unreached(ax, x, color):
    ax.plot([x], [0.96], marker="x", color=color, ms=9, mew=2, transform=ax.get_xaxis_transform(), clip_on=False)


def fig2_snr(main, snr):
    schemes = _schemes(main)
    if not schemes:
        return
    rhos = sorted(set([20.0] + ([] if snr.empty else list(snr.rho_db.unique()))))
    rows = []
    fig, axes = plt.subplots(len(schemes), 3, figsize=(17, 4.4 * len(schemes)), squeeze=False)
    for i, scheme in enumerate(schemes):
        A = primary_target(main, scheme)
        allacc = []
        for m in ORDER:
            st = STYLE[m]
            acc, tt = [], {"uplink": [], "training": []}
            for rho in rhos:
                run = _snr_runs(main, snr, scheme, m, rho)
                if run.empty:
                    continue
                a, amin, amax = final_acc(run)
                su, sr = ttt(run, A, "uplink_s"), ttt(run, A, "training_time_s")
                acc.append((rho, a, amin, amax))
                tt["uplink"].append((rho, su))
                tt["training"].append((rho, sr))
                allacc += [amin, amax]
                nsr = run[run["round"] >= 1]["act_nsr_db"].mean() if m == "airsfl" else np.nan
                agg = run[run["round"] >= 1]["agg_nsr_db"].mean() if m in ("airsfl", "sun_fdma_aircomp",
                                                                          "hybrid_zf_aircomp", "aircomp_fl") \
                    else np.nan
                rows.append({"partition": PART_NAME[scheme], "method": st["label"],
                             "digital payload": _payload_of(m, run), "SNR (dB)": rho,
                             "UL s/round": float(run.ul_s_per_round.iloc[0]),
                             "training s/round": float(run.e2e_s_per_round.iloc[0]),
                             "seeds": su["n"], "final test acc (%)": a, "acc min": amin, "acc max": amax,
                             "target val acc (%)": 100 * A if A else np.nan, "reached": f"{su['hit']}/{su['n']}",
                             "UL time to target (s)": su["t"], "training time to target (s)": sr["t"],
                             "activation NSR (dB)": nsr, "aggregation NSR (dB)": agg})
            if acc:
                r, a, lo, hi = map(np.array, zip(*acc))
                axes[i][0].errorbar(r, a, yerr=[a - lo, hi - a], color=st["color"], marker=st["marker"],
                                    ls=st["ls"], lw=st["lw"], capsize=3, label=st["label"])
            for j, axis in ((1, "uplink"), (2, "training")):
                ok = [(r, s) for r, s in tt[axis] if s["hit"] > 0]
                if ok:
                    r = np.array([x for x, _ in ok])
                    t = np.array([s["t"] for _, s in ok])
                    lo = np.array([s["tmin"] for _, s in ok])
                    hi = np.array([s["tmax"] for _, s in ok])
                    axes[i][j].errorbar(r, t, yerr=[t - lo, hi - t], color=st["color"], marker=st["marker"],
                                        ls=st["ls"], lw=st["lw"], capsize=3, label=st["label"])
                    for rr, s in ok:
                        if s["hit"] < s["n"]:
                            axes[i][j].annotate(f"{s['hit']}/{s['n']}", (rr, s["t"]), textcoords="offset points",
                                                xytext=(4, 4), fontsize=7, color=st["color"])
                for rr, s in tt[axis]:
                    if s["hit"] == 0:
                        _unreached(axes[i][j], rr, st["color"])
        if allacc:
            lo, hi = min(allacc), max(allacc)
            if hi - lo < 6:
                mid = 0.5 * (hi + lo)
                lo, hi = mid - 3, mid + 3
            axes[i][0].set_ylim(lo - 1, hi + 1)
        axes[i][0].set_xlabel("Reference SNR rho (dB)")
        axes[i][0].set_ylabel("Final test accuracy (%)")
        axes[i][0].set_title(f"{PART_NAME[scheme]}: final accuracy vs SNR (bars = seed min-max)\n"
                             f"digital (FP{ENV0['q_bits']}) learning is SNR-independent; ZF variants = their OFDMA twins",
                             fontsize=10.5)
        for j, axis in ((1, "uplink"), (2, "training")):
            axes[i][j].set_yscale("log")
            lo_y, hi_y = axes[i][j].get_ylim()
            axes[i][j].set_ylim(lo_y, hi_y * 10)       # headroom: the "not reached" x never sits on a data point
            axes[i][j].set_xlabel("Reference SNR rho (dB)")
            axes[i][j].set_ylabel(f"{'Uplink' if axis == 'uplink' else 'Training'} time to target (s)")
            axes[i][j].set_title(f"{PART_NAME[scheme]}: {axis} time to {100*A:.0f}% val. acc.\n"
                                 "(x at the top = not reached)" if A else PART_NAME[scheme], fontsize=10.5)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    _export_panels(fig, "2_snr", [(axes[i][j], f"{scheme}_{name}") for i, scheme in enumerate(schemes)
                                  for j, name in enumerate(("final_accuracy", "uplink_time_to_target",
                                                            "training_time_to_target"))])
    _save(fig, "fig2_snr")
    if rows:
        _write_table(pd.DataFrame(rows), "table_snr")

    # 2b: AirSFL learning curves at every SNR, with digital SFL-V1 at the lowest SNR and at 20 dB
    fig, axes = plt.subplots(1, len(schemes), figsize=(6.4 * len(schemes), 4.6), squeeze=False)
    cmap = plt.get_cmap("Reds")
    for ax, scheme in zip(axes[0], schemes):
        for j, rho in enumerate(rhos):
            run = _snr_runs(main, snr, scheme, "airsfl", rho)
            if not run.empty:
                _plot_curve(ax, run, STYLE["airsfl"], "uplink_s", label=f"AirSFL, {rho:g} dB",
                            color=cmap(0.3 + 0.7 * j / max(1, len(rhos) - 1)), lw=2.0, ls="-")
        ef = _airsfl_error_free(main, scheme)
        if not ef.empty:
            _plot_curve(ax, ef, STYLE["airsfl_errfree"], "uplink_s", lw=1.8)
        for m in ("digital_sflv1_zf", "digital_sflv1"):
            if m not in ORDER:
                continue
            for rho, ls in sorted({(min(rhos), "--"), (20.0, "-")}):
                run = _snr_runs(main, snr, scheme, m, rho)
                if not run.empty:
                    _plot_curve(ax, run, STYLE[m], "uplink_s", label=f"{STYLE[m]['label']}, {rho:g} dB",
                                ls=ls, lw=1.8)
        ax.set_xscale("log")
        ax.set_xlabel(AXES["uplink"][1])
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"SNR sweep — {PART_NAME[scheme]}")
    _fig_legend(fig, axes, ncol=4)
    fig.suptitle(f"AirSFL: airtime is SNR-independent, distortion is not; digital (FP{ENV0['q_bits']}): accuracy fixed, "
                 "rate falls at low SNR", y=1.02, fontsize=11)
    _save(fig, "fig2b_airsfl_snr_curves")

    # 2c: accuracy reached within a fixed time budget vs SNR (uplink and training axes)
    # budgets: base T per axis (--budget-uplink / --budget-training, default AirSFL's full-run
    # time at 20 dB), times each multiplier in BUDGET_MULTS (--budget-mult); one column each
    cols = [(axis, mult) for axis in ("uplink", "training") for mult in BUDGET_MULTS]
    rows = []
    fig, axes = plt.subplots(len(schemes), len(cols), figsize=(6 * len(cols), 4.5 * len(schemes)), squeeze=False)
    for i, scheme in enumerate(schemes):
        for j, (axis, mult) in enumerate(cols):
            ax = axes[i][j]
            xcol = AXES[axis][0]
            T = BUDGETS[axis]
            if T is None:                            # default: AirSFL's full-run time at 20 dB (seed mean)
                air = _snr_runs(main, snr, scheme, "airsfl", 20.0)
                if air.empty:
                    continue
                T = float(np.mean([g[xcol].iloc[-1] for g in _seed_runs(air)]))
            T *= mult
            for m in ORDER:
                pts = []
                for rho in rhos:
                    run = _snr_runs(main, snr, scheme, m, rho)
                    if run.empty:
                        continue
                    b = budget_point(run, T, xcol)
                    pts.append((rho, b))
                    rows.append({"partition": PART_NAME[scheme], "axis": axis, "budget (s)": T,
                                 "method": STYLE[m]["label"], "SNR (dB)": rho, "test acc at budget (%)": b["a"],
                                 "min": b["lo"], "max": b["hi"], "rounds within budget": b["rounds_in_budget"],
                                 "last evaluated round": b["last_round"], "evaluated": b["evaluated"]})
                st = STYLE[m]
                ev = [(r, b) for r, b in pts if b["evaluated"]]
                if ev:
                    r = np.array([x for x, _ in ev])
                    a, lo, hi = (np.array([b[k] for _, b in ev]) for k in ("a", "lo", "hi"))
                    ax.errorbar(r, a, yerr=[a - lo, hi - a], color=st["color"], marker=st["marker"], ls=st["ls"],
                                lw=st["lw"], capsize=3, label=st["label"])
                labeled = bool(ev)
                for r, b in pts:                     # no trained checkpoint inside T: initial model only
                    if not b["evaluated"] and not np.isnan(b["a"]):
                        ax.plot([r], [b["a"]], marker=st["marker"], mfc="white", mec=st["color"], ms=8, ls="",
                                label=None if labeled else st["label"])
                        labeled = True
            ef = _airsfl_error_free(main, scheme)
            if not ef.empty:                         # SNR-independent upper bound on AirSFL's time axis
                b = budget_point(ef, T, xcol)
                st = STYLE["airsfl_errfree"]
                ax.plot([min(rhos), max(rhos)], [b["a"], b["a"]], color=st["color"], ls=st["ls"], lw=st["lw"],
                        label=st["label"])
                rows.append({"partition": PART_NAME[scheme], "axis": axis, "budget (s)": T, "method": st["label"],
                             "SNR (dB)": np.nan, "test acc at budget (%)": b["a"], "min": b["lo"], "max": b["hi"],
                             "rounds within budget": b["rounds_in_budget"], "last evaluated round": b["last_round"],
                             "evaluated": b["evaluated"]})
            ax.set_xlabel("Reference SNR rho (dB)")
            ax.set_ylabel("Test accuracy within the budget (%)")
            ax.set_title(f"{PART_NAME[scheme]}: accuracy after {T:.4g} s of {axis} time"
                         + (f" ({mult:g} x AirSFL's run)" if mult != 1 or BUDGETS[axis] is None else ""),
                         fontsize=10)
    handles = {}                                     # one legend for the figure, below the panels
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    handles["hollow marker: no evaluated checkpoint within the budget (value = initial model)"] = \
        plt.Line2D([], [], marker="o", mfc="white", mec="k", ls="")
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=min(5, len(handles)), fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07 if len(schemes) > 1 else 0.12, 1, 1))
    _export_panels(fig, "2c", [(axes[i][j], f"{scheme}_{axis}_budget_{mult:g}x") for i, scheme in enumerate(schemes)
                               for j, (axis, mult) in enumerate(cols)], hollow_note=True)
    _save(fig, "fig2c_acc_at_budget")
    if rows:
        _write_table(pd.DataFrame(rows), "table_budget")


def _sweep_curves(fname, title, schemes, panels, runs_of, axis, xscale):
    """Learning curves of every drawn method at every point of a sweep: rows = partitions,
    columns = sweep points; runs_of(scheme, method, point) -> runs ("airsfl_errfree" = the
    error-free upper bound). On the epochs axis the ZF variants coincide with their OFDMA
    parents and the error-free curve with digital SFL-V1, so those duplicates are skipped."""
    xcol, xlab = {"epochs": ("epoch_equiv", "Global-epoch equivalents"), "uplink": AXES["uplink"],
                  "training": AXES["training"]}[axis]
    ncol = min(len(panels), 5)                      # at most 5 panels per row (e.g. 9 SNRs -> 5 + 4)
    per = math.ceil(len(panels) / ncol)             # grid rows per partition
    nrows = per * len(schemes)
    fig, axes = plt.subplots(nrows, ncol, figsize=(4.9 * ncol, 4.2 * nrows), squeeze=False)
    for i, scheme in enumerate(schemes):
        block = []
        for j, (label, point) in enumerate(panels):
            ax = axes[i * per + j // ncol][j % ncol]
            block.append(ax)
            for m in ORDER + ["airsfl_errfree"]:
                if axis == "epochs" and ((m in DERIVED and DERIVED[m] in ORDER) or
                                         (m == "airsfl_errfree" and "digital_sflv1" in ORDER)):
                    continue
                run = runs_of(scheme, m, point)
                if run is None or run.empty:
                    continue
                _plot_curve(ax, run, STYLE[m], xcol, first_round=1 if (xscale == "log" and axis != "epochs") else 0,
                            lw=1.5 if m == "airsfl_errfree" else None)
            if axis != "epochs":
                ax.set_xscale(xscale)
            ax.set_title(f"{PART_NAME[scheme]}, {label}", fontsize=10)
            ax.set_xlabel(xlab, fontsize=9)
            if j % ncol == 0:
                ax.set_ylabel("Test accuracy (%)")
        lo = min(ax.get_ylim()[0] for ax in block)      # one accuracy scale per partition
        hi = max(ax.get_ylim()[1] for ax in block)
        for ax in block:
            ax.set_ylim(lo, hi)
        for j in range(len(panels), per * ncol):
            axes[i * per + j // ncol][j % ncol].axis("off")
    handles = {}
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=min(4, max(1, len(handles))), fontsize=8.5)
    fig.suptitle(title, y=1.01, fontsize=11)
    fig.tight_layout(rect=(0, min(0.14, 0.75 / (4.2 * nrows)), 1, 1))
    _save(fig, fname)


def fig_sweep_curves(main, snr, pl):
    """Learning curves of all methods at every swept SNR (fig2d_*) and at every path-gain
    spread (fig11b_*), on the axes chosen with --sweep-curves (epochs / uplink / training)."""
    schemes = _schemes(main)
    if not schemes or not SWEEP_CURVES["axes"]:
        return
    ef = lambda scheme: _airsfl_error_free(main, scheme)
    rhos = sorted(set([20.0] + ([] if snr.empty else [float(r) for r in snr.rho_db.unique()])))
    if len(rhos) > 1:
        for axis in SWEEP_CURVES["axes"]:
            _sweep_curves(f"fig2d_curves_all_snr_{axis}",
                          f"Learning curves at every SNR ({_pl()}; digital learning is SNR-independent, its rate "
                          f"is not)",
                          schemes, [(f"rho = {r:g} dB", r) for r in rhos],
                          lambda s, m, r: ef(s) if m == "airsfl_errfree" else _snr_runs(main, snr, s, m, r),
                          axis, SWEEP_CURVES["xscale"])
    if not pl.empty:
        spreads = [0.0] + sorted(set(float(x) for x in pl.path_gain_spread_db))
        pl_rhos = sorted(set(float(r) for r in pl.rho_db))
        for rho in pl_rhos:                     # one figure per median SNR of the pathloss runs
            tag = f"_rho{rho:g}" if len(pl_rhos) > 1 else ""
            for axis in SWEEP_CURVES["axes"]:
                _sweep_curves(f"fig11b_curves_all_spreads_{axis}{tag}",
                              f"Learning curves at every path-gain spread ({rho:g} dB median client, {_pl()}; digital "
                              f"uploads wait for the weakest client)", schemes, [(f"spread {s:g} dB", s) for s in spreads],
                              lambda sc, m, s, rho=rho: ef(sc) if m == "airsfl_errfree"
                              else _spread_runs(main, snr, pl, sc, m, s, rho),
                              axis, SWEEP_CURVES["xscale"])


def fig11_path_gains(main, snr, pl):
    """Unequal path gains (the draft's robustness test; rho = median client's SNR): final
    accuracy, uplink time to the primary target and uplink seconds per round vs the spread.
    Digital uploads wait for the weakest client (Eq. 12); analog airtime does not depend on
    the gains, which enter only the distortion (weak clients get noisier ZF activations and
    set the AirComp scaling)."""
    if pl.empty or "path_gain_spread_db" not in pl:
        return
    schemes = _schemes(pl)
    spreads = [0.0] + sorted(set(float(x) for x in pl.path_gain_spread_db))
    rows = []
    for rho in sorted(set(float(r) for r in pl.rho_db)):
        fig, axes = plt.subplots(len(schemes), 3, figsize=(17, 4.4 * len(schemes)), squeeze=False)
        for i, scheme in enumerate(schemes):
            A = primary_target(main, scheme)
            for m in ORDER:
                st = STYLE[m]
                pts = []
                for s in spreads:
                    run = _spread_runs(main, snr, pl, scheme, m, s, rho)
                    if run.empty:
                        continue
                    a, lo, hi = final_acc(run)
                    tt = ttt(run, A, "uplink_s")
                    pts.append((s, a, lo, hi, tt, float(run.ul_s_per_round.iloc[0])))
                    rows.append({"partition": PART_NAME[scheme], "SNR (dB)": rho, "spread (dB)": s,
                                 "weakest client (dB)": -s * (1 - 1 / int(run.N.iloc[0])) / 2,
                                 "method": st["label"],
                                 "UL s/round": pts[-1][5], "final test acc (%)": a, "acc min": lo, "acc max": hi,
                                 "reached": f"{tt['hit']}/{tt['n']}", "UL time to target (s)": tt["t"],
                                 "training time to target (s)": ttt(run, A, "training_time_s")["t"],
                                 "activation NSR (dB)": run[run["round"] >= 1]["act_nsr_db"].mean(),
                                 "aggregation NSR (dB)": run[run["round"] >= 1]["agg_nsr_db"].mean()})
                if not pts:
                    continue
                x = np.array([p[0] for p in pts])
                a, lo, hi = (np.array([p[k] for p in pts]) for k in (1, 2, 3))
                axes[i][0].errorbar(x, a, yerr=[a - lo, hi - a], color=st["color"], marker=st["marker"],
                                    ls=st["ls"], lw=st["lw"], capsize=3, label=st["label"])
                ok = [(p[0], p[4]) for p in pts if p[4]["hit"] > 0]
                if ok:
                    axes[i][1].plot([v for v, _ in ok], [s_["t"] for _, s_ in ok], color=st["color"],
                                    marker=st["marker"], ls=st["ls"], lw=st["lw"], label=st["label"])
                for p in pts:
                    if p[4]["hit"] == 0:
                        _unreached(axes[i][1], p[0], st["color"])
                axes[i][2].plot(x, [p[5] for p in pts], color=st["color"], marker=st["marker"], ls=st["ls"],
                                lw=st["lw"], label=st["label"])
            for j, ylab in enumerate(("Final test accuracy (%)", "Uplink time to target (s)",
                                      "Uplink seconds per round")):
                axes[i][j].set_xlabel("Path-gain spread (dB) around the median client")
                axes[i][j].set_ylabel(ylab)
                if j:
                    axes[i][j].set_yscale("log")
            lo_a, hi_a = axes[i][0].get_ylim()           # at least 6 points of range: seed noise stays small
            if hi_a - lo_a < 6:
                mid = 0.5 * (lo_a + hi_a)
                axes[i][0].set_ylim(mid - 3, mid + 3)
            lo_y, hi_y = axes[i][1].get_ylim()
            axes[i][1].set_ylim(lo_y, hi_y * 10)         # headroom for the "not reached" x
            axes[i][0].set_title(f"{PART_NAME[scheme]}, rho = {rho:g} dB: final accuracy", fontsize=10.5)
            axes[i][1].set_title(f"uplink time to {100 * A:.0f}% val. acc. (x at the top = not reached)"
                                 if A else "uplink time to target", fontsize=10.5)
            axes[i][2].set_title(f"per-round uplink time ({_pl()}: paced by the weakest client)", fontsize=10.5)
        fig.tight_layout()
        _fig_legend(fig, axes, ncol=4)
        _save(fig, "fig11_path_gains" + (f"_rho{rho:g}" if len(set(pl.rho_db)) > 1 else ""))
    if rows:
        _write_table(pd.DataFrame(rows), "table_pathloss")


def fig1e_linear_time(main, snr):
    """Learning curves in the style of Sun et al. (Fig. 2a/b): test accuracy vs accumulated
    uplink time on a LINEAR axis, at 20 dB and at the lowest swept SNR (--linear-snrs). On the
    log axis of fig1 a per-round-time ratio is a pure shift, so every curve has the same shape;
    on a linear axis it is a different slope. All methods run the same equal-period schedule,
    so the curves differ by per-round airtime and, at low SNR, by analog distortion (AirSFL
    vs its error-free twin). Dotted: the run's epoch budget is used up (final value carried)."""
    schemes = _schemes(main)
    if not schemes:
        return
    snrs = LINEAR["snrs"] or ([20.0] + ([float(snr.rho_db.min())] if not snr.empty else []))
    fig, axes = plt.subplots(len(schemes), len(snrs), figsize=(6.6 * len(snrs), 4.7 * len(schemes)),
                             squeeze=False)
    for i, scheme in enumerate(schemes):
        xmax = LINEAR["xmax"]
        if xmax is None:                     # default: AirComp-FL's full run at 20 dB (fastest baseline)
            ends = [float(r.uplink_s.max()) for r in (_snr_runs(main, snr, scheme, m, 20.0)
                                                       for m in ("aircomp_fl", "airsfl")) if not r.empty]
            xmax = 1.05 * max(ends) if ends else None
        for j, rho in enumerate(snrs):
            ax = axes[i][j]
            curves = [(m, _snr_runs(main, snr, scheme, m, rho)) for m in ORDER]
            curves.append(("airsfl_errfree", _airsfl_error_free(main, scheme)))
            for m, run in curves:
                if run.empty:
                    continue
                st = STYLE[m]
                c = _plot_curve(ax, run, st, "uplink_s", first_round=0, markers=True,
                                lw=1.6 if m == "airsfl_errfree" else None)
                if xmax and not c.empty and float(c.uplink_s.iloc[-1]) < xmax:
                    ax.plot([float(c.uplink_s.iloc[-1]), xmax], [100 * float(c.test_acc.iloc[-1])] * 2,
                            color=st["color"], ls=":", lw=1.0, alpha=0.7)
            if xmax:
                ax.set_xlim(0, xmax)
            ax.set_xlabel(AXES["uplink"][1])
            ax.set_ylabel("Test accuracy (%)")
            ax.set_title(f"{PART_NAME[scheme]}, rho = {rho:g} dB", fontsize=10.5)
    fig.suptitle(f"Accuracy vs uplink time, linear axis, {_pl()} (same equal-period schedule for all methods: curves "
                 "differ by airtime per round and, at low SNR, by analog distortion; dotted = epoch budget used up)",
                 y=1.01, fontsize=10.5)
    fig.tight_layout()
    _fig_legend(fig, axes, ncol=4)
    _export_panels(fig, "1e", [(axes[i][j], f"{scheme}_snr{rho:g}") for i, scheme in enumerate(schemes)
                               for j, rho in enumerate(snrs)])
    _save(fig, "fig1e_acc_vs_uplink_time_linear")


# ---------------------------------------------------------------------------
# Figure 3 / 3b / 3c + table_cuts / table_round_time (analytic, roadmap A)
# ---------------------------------------------------------------------------

def _analytic_compute(m, N=30, stage=CUT0, tau=TAU0, u_min=0.0):
    """Computation per round in the default environment at the slowest possible client
    (u_min = 0 -> f_min = lo TFLOPS), for analytic figures."""
    lo, hi = COMPUTE["client_tflops_lo"], COMPUTE["client_tflops_hi"]
    return compute_time_breakdown(m, split_flops(stage if m in SPLIT_METHODS else None), N, B0, tau,
                                  (lo + (hi - lo) * u_min) * 1e12, COMPUTE["server_tflops"] * 1e12)


def _xlabels(ms):
    return [STYLE[m]["label"].replace(" (proposed)", "") for m in ms]


def _ratio_txt(v):
    return f"{v:.0f}x" if v >= 10 else f"{v:.1f}x"


def fig3_breakdown(Ns=(20, 30, 40), stage=CUT0, tau=TAU0):
    dims = profiled_dims(B0)[stage]
    sfl = [m for m in ORDER if m in SPLIT_METHODS]
    npan = len(sfl) + 1                                   # one stacked panel per split method + all methods
    ncol = min(3, npan)
    nrow = math.ceil(npan / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.6 * nrow), squeeze=False)
    axes = axes.ravel()
    for ax in axes[npan:]:
        ax.axis("off")
    phases = [("activation", "#ff9896", "activations (client -> M-server)"),
              ("labels", "#ffbb78", "labels"),
              ("aggregation", "#aec7e8", "prefix differences (client -> F-server)")]
    x = np.arange(len(Ns))
    tot = {m: [] for m in ORDER}
    for m in ORDER:
        for N in Ns:
            r = _radio(N=N)
            tot[m].append(uplink_time_breakdown(m, dims, r, tau, _rates(r)))
    for ax, m in zip(axes, sfl):
        bottom = np.zeros(len(Ns))
        for key, col, lab in phases:
            vals = np.array([b[key] for b in tot[m]])
            ax.bar(x, vals, bottom=bottom, color=col, edgecolor="k", lw=0.5, width=0.6, label=lab)
            bottom += vals
        for i, b in enumerate(tot[m]):
            ax.text(i, b["total"] * 1.02, f"{b['total']:.2f} s" if b["total"] < 10 else f"{b['total']:.0f} s",
                    ha="center", va="bottom", fontsize=9, fontweight="bold" if m == "airsfl" else None)
        ax.set_ylim(0, max(b["total"] for b in tot[m]) * 1.45)
        ax.set_xticks(x)
        tones = "all 120 tones" if m not in ("digital_sflv1", "sun_fdma_aircomp") else None
        ax.set_xticklabels([f"N={N}\n({tones or f'{120 // N} tones'})" for N in Ns], fontsize=9)
        ax.set_title(STYLE[m]["label"], color=STYLE[m]["color"], fontweight="bold")
        ax.set_ylabel("Uplink seconds per round")
    axes[0].legend(fontsize=7.5, loc="upper left")
    ax = axes[len(sfl)]
    w = 0.8 / len(ORDER)
    for j, m in enumerate(ORDER):
        ax.bar(x + (j - (len(ORDER) - 1) / 2) * w, [b["total"] for b in tot[m]], width=w, color=STYLE[m]["color"],
               edgecolor="k", lw=0.4, label=STYLE[m]["label"])
    ref = [m for m in ("digital_sflv1_zf", "digital_sflv1") if m in tot]
    ticks = []
    for i, N in enumerate(Ns):
        txt = f"N={N}"
        if "airsfl" in tot and ref:
            txt += "\nSFL-V1 / AirSFL:\n" + ", ".join(
                f"{'ZF' if m.endswith('_zf') else 'OFDMA'} {_ratio_txt(tot[m][i]['total'] / tot['airsfl'][i]['total'])}"
                for m in ref)
        ticks.append(txt)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 1e4)
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, fontsize=8.5)
    ax.set_title("All methods (log scale; FL has no activation phase)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle(f"Per-round uplink airtime by phase (cut {stage}, tau={tau}, Nr=64 at both servers, W=1.8 MHz, "
                 f"{ENV0['rho_db']:g} dB, eps=0.6, {_pl()}):\nanalog time is independent of N; digital OFDMA time grows ~N (S/N tones "
                 f"per client), digital ZF only through the ZF gain Nr-N+1", y=1.02, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig3_uplink_breakdown")

    r = _radio()
    # 3b: per-round uplink time vs cut (N=30) -- optional (plots.py --also fig3b)
    if "fig3b" in ALSO:
        rows = []
        fig, ax = plt.subplots(figsize=(11, 4.8))
        wb = 0.8 / len(ORDER)
        dims_all = profiled_dims(B0)
        for j, m in enumerate(ORDER):
            vals = []
            for s in (1, 2, 3, 4):
                b = uplink_time_breakdown(m, dims_all[s], r, tau, _rates(r))
                c = _analytic_compute(m, stage=s)
                vals.append(b["total"])
                rows.append({"cut": s, "method": STYLE[m]["label"], "UL s/round": b["total"],
                             "activation": b["activation"], "labels": b["labels"], "aggregation": b["aggregation"],
                             "compute s/round (f_min=lo)": c["total"], "training s/round": b["total"] + c["total"],
                             "d_c": dims_all[s]["d_c"], "d_s": dims_all[s]["d_s"], "d_a": dims_all[s]["d_a"],
                             "tau*d_a < d_s": tau * dims_all[s]["d_a"] < dims_all[s]["d_s"],
                             f"MB/round (FP{r.q_bits})": source_equivalent_mb_per_round(m, dims_all[s], r, tau,
                                                                                        q_ref=r.q_bits)})
            ax.bar(np.arange(4) + (j - (len(ORDER) - 1) / 2) * wb, vals, width=wb, color=STYLE[m]["color"],
                   edgecolor="k", lw=0.4, label=STYLE[m]["label"])
        ax.set_yscale("log")
        ax.set_ylim(0.5, 1e4)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels([f"cut {s}{' (default)' if s == stage else ''}\nd_a={dims_all[s]['d_a']/1e6:.2f}M\n"
                            f"d_c={dims_all[s]['d_c']/1e6:.2f}M" for s in (1, 2, 3, 4)], fontsize=9)
        ax.set_ylabel("Uplink seconds per round")
        ax.set_title(f"Per-round uplink time vs cut (N=30, {ENV0['rho_db']:g} dB, {_pl()}): with equal analog efficiencies and\n"
                     "negligible labels, AirSFL < AirComp-FL when tau*d_a < d_s", fontsize=10.5)
        ax.legend(fontsize=8, ncol=4, loc="upper center")
        _save(fig, "fig3b_uplink_vs_cut")
        _write_table(pd.DataFrame(rows), "table_cuts")

    # 3c: per-round training time = uplink + computation, and its composition
    parts = [("client FP", "#c7e9c0"), ("activation UL", "#ff9896"), ("labels UL", "#ffbb78"),
             ("server FP+BP", "#9ecae1"), ("client BP", "#74c476"), ("aggregation UL", "#aec7e8")]
    rows = []
    comp = {}
    for m in ORDER:
        b = uplink_time_breakdown(m, dims, r, tau, _rates(r))
        c = _analytic_compute(m)
        comp[m] = {"client FP": c["client_fp"], "activation UL": b["activation"], "labels UL": b["labels"],
                   "server FP+BP": c["server_fp"] + c["server_bp"], "client BP": c["client_bp"],
                   "aggregation UL": b["aggregation"]}
        rows.append({"method": STYLE[m]["label"], **comp[m], "uplink": b["total"], "compute": c["total"],
                     "training (s/round)": b["total"] + c["total"]})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
    x = np.arange(len(ORDER))
    ul = np.array([rw["uplink"] for rw in rows])
    trn = np.array([rw["training (s/round)"] for rw in rows])
    ax1.bar(x - 0.2, ul, width=0.4, color="#fdae6b", edgecolor="k", lw=0.5, label="uplink only")
    ax1.bar(x + 0.2, trn, width=0.4, color="#6baed6", edgecolor="k", lw=0.5, label="uplink + computation")
    for i in range(len(ORDER)):
        ax1.text(i + 0.2, trn[i] * 1.3, f"{trn[i]:.2f} s" if trn[i] < 10 else f"{trn[i]:.0f} s", ha="center",
                 fontsize=8.5, fontweight="bold" if ORDER[i] == "airsfl" else None)
    ax1.set_yscale("log")
    ax1.set_ylim(0.1, trn.max() * 20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(_xlabels(ORDER), rotation=25, ha="right")
    ax1.set_ylabel("Seconds per round")
    ax1.set_title(f"(a) Per-round time (cut 2, N=30, {ENV0['rho_db']:g} dB, {_pl()}, slowest client 1 TFLOPS, "
                  f"M-server 20 TFLOPS)", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")
    left = np.zeros(len(ORDER))
    for name, col in parts:
        frac = np.array([comp[m][name] / trn[i] for i, m in enumerate(ORDER)])
        ax2.barh(x, frac, left=left, color=col, edgecolor="k", lw=0.4, label=name)
        left += frac
    ax2.set_yticks(x)
    ax2.set_yticklabels(_xlabels(ORDER))
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Fraction of the round (sequential stages of each co-split step)")
    ax2.set_title("(b) Where a training round's time goes", fontsize=10)
    ax2.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    _save(fig, "fig3c_round_time")
    _write_table(pd.DataFrame(rows), "table_round_time")


# ---------------------------------------------------------------------------
# Figure 4: communication overhead (source-equivalent MB vs airtime)
# ---------------------------------------------------------------------------

def fig4_overhead(main):
    r = _radio()
    dims = profiled_dims(B0)[CUT0]
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(14, 5.2))
    for m in ORDER:
        mb = source_equivalent_mb_per_round(m, dims, r, TAU0, q_ref=r.q_bits)
        ul = uplink_time_per_round(m, dims, r, TAU0, _rates(r))
        st = STYLE[m]
        ax.scatter([mb], [ul], s=130, color=st["color"], marker=st["marker"], edgecolor="k", zorder=5,
                   label=st["label"])
        # the split variants share one x (same bytes): digital SFL-V1 labelled on the left, the others right
        left = m in ("digital_sflv1", "digital_sflv1_zf")
        ax.annotate(f"{ul:.2f} s" if ul < 10 else f"{ul:.0f} s", (mb, ul), textcoords="offset points",
                    xytext=(-10, 2) if left else (10, -6 if m.startswith("hybrid") or m.startswith("sun") else -3),
                    ha="right" if left else "left", fontsize=8.5, color=st["color"], fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper center", ncol=2)
    sfl_mb = source_equivalent_mb_per_round("airsfl", dims, r, TAU0, q_ref=r.q_bits)
    fl_mb = source_equivalent_mb_per_round("aircomp_fl", dims, r, TAU0, q_ref=r.q_bits)
    ax.axvline(sfl_mb, color="k", ls=":", lw=1)
    t_air = uplink_time_per_round("airsfl", dims, r, TAU0, _rates(r))
    n_sfl = sum(m in SPLIT_METHODS for m in ORDER)
    txt = f"all {n_sfl} SFL variants carry the same\n{sfl_mb:.0f} MB/round of source data;\nAirSFL airtime per round ="
    for m in ("digital_sflv1_zf", "digital_sflv1"):
        if m in ORDER:
            t = uplink_time_per_round(m, dims, r, TAU0, _rates(r))
            txt += f"\n  1/{_ratio_txt(t / t_air)[:-1]} of {STYLE[m]['label']}"
    ax.text(sfl_mb + 0.3 * fl_mb, 0.45, txt, fontsize=8.5, va="center",
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.set_yscale("log")
    ax.set_ylim(0.1, 3e4)                                  # headroom for the legend
    ax.set_xlim(0, fl_mb * 1.3)
    ax.set_xlabel(f"Source-equivalent uplink volume per round (MB, {r.q_bits}-bit values)")
    ax.set_ylabel("Uplink airtime per round (s)")
    ax.set_title(f"(a) Bytes vs airtime per round (cut {CUT0}, {ENV0['rho_db']:g} dB, {_pl()})")
    x = np.arange(len(ORDER))
    any_bar = False
    schemes = _schemes(main)
    width = 0.8 / max(1, len(schemes))
    for si, scheme in enumerate(schemes):
        A = primary_target(main, scheme)
        vals = []
        for m in ORDER:
            dm = main[(main.partition == scheme) & (main.method == m)]
            vals.append(ttt(dm, A)["k"] * _mb_per_round(dm) / 1e3 if not dm.empty else np.nan)
        xs = x + (si - (len(schemes) - 1) / 2) * width
        axb.bar(xs, vals, width=width, color=["#1f77b4", "#ff7f0e"][si], alpha=0.85, label=PART_NAME[scheme])
        for xi, v in zip(xs, vals):
            if np.isnan(v):
                axb.text(xi, 0.02, "not\nreached", ha="center", fontsize=7, transform=axb.get_xaxis_transform())
            else:
                any_bar = True
    axb.set_xticks(x)
    axb.set_xticklabels(_xlabels(ORDER), rotation=25, ha="right")
    axb.set_ylabel(f"Source-equivalent GB to target ({ENV0['q_bits']}-bit values)")
    axb.set_title("(b) Uplink source volume to reach the target (not airtime)")
    if any_bar:
        axb.legend(fontsize=8)
    elif not schemes:
        axb.text(0.5, 0.5, "needs the `main` runs", ha="center", va="center", transform=axb.transAxes,
                 fontsize=10, color="0.4")
    fig.tight_layout()
    _save(fig, "fig4_comm_overhead")


# ---------------------------------------------------------------------------
# Figure 5: M-server antenna margin (optional experiment `nr`)
# ---------------------------------------------------------------------------

def fig5_nr(main, nr):
    if nr.empty:
        return
    frames = [nr] + ([main[main.method == "airsfl"]] if not main.empty else [])
    allr = pd.concat(frames, ignore_index=True)
    schemes = _schemes(allr)
    fig, axes = plt.subplots(1, len(schemes), figsize=(6.3 * len(schemes), 4.6), squeeze=False)
    cmap = plt.get_cmap("Reds")
    for ax, scheme in zip(axes[0], schemes):
        sub = allr[(allr.partition == scheme) & (allr.method == "airsfl") & (allr.rho_db == 20.0)]
        nrs = sorted(set(sub.Nr))
        for i, Nr in enumerate(nrs):
            dm = sub[sub.Nr == Nr]
            _plot_curve(ax, dm, STYLE["airsfl"], "uplink_s", color=cmap(0.35 + 0.6 * i / max(1, len(nrs) - 1)),
                        lw=2.0, ls="-", label=f"AirSFL, Nr_M={Nr}, Nr_F={int(dm.Nr_F.iloc[0])} "
                                              f"(Nr_M-N={Nr - int(dm.N.iloc[0])})")
        for m in ("digital_sflv1_zf", "digital_sflv1"):
            ref = main[(main.partition == scheme) & (main.method == m)] if not main.empty else main
            if m in ORDER and not ref.empty:
                _plot_curve(ax, ref, STYLE[m], "uplink_s", label=f"{STYLE[m]['label']} (Nr=64, FP{ENV0['q_bits']})")
        ax.set_xscale("log")
        ax.set_xlabel(AXES["uplink"][1])
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"M-server antenna margin (20 dB) — {PART_NAME[scheme]}")
        ax.legend(fontsize=8, loc="lower right")
    _save(fig, "fig5_nr")


# ---------------------------------------------------------------------------
# Figure 6: uplink and training time to the primary target (headline bars)
# ---------------------------------------------------------------------------

def fig6_time_to_target(main):
    schemes = _schemes(main)
    if not schemes:
        return
    fig, axes = plt.subplots(2, len(schemes), figsize=(1.15 * len(ORDER) * len(schemes), 9.8), squeeze=False)
    x = np.arange(len(ORDER))
    for row, axis in enumerate(("uplink", "training")):
        xcol = AXES[axis][0]
        for ax, scheme in zip(axes[row], schemes):
            A = primary_target(main, scheme)
            stats = {m: ttt(main[(main.partition == scheme) & (main.method == m)], A, xcol) for m in ORDER}
            air = stats.get("airsfl", {"hit": 0})
            for i, m in enumerate(ORDER):
                s, st = stats[m], STYLE[m]
                if s["n"] == 0:
                    continue
                if s["hit"] == 0:
                    ax.text(i, 0.03, "not\nreached", ha="center", fontsize=8, color=st["color"],
                            transform=ax.get_xaxis_transform())
                    continue
                ax.bar(i, s["t"], color=st["color"], edgecolor="k", lw=0.5, width=0.62,
                       yerr=[[s["t"] - s["tmin"]], [s["tmax"] - s["t"]]] if s["hit"] > 1 else None, capsize=4)
                txt = f"{s['t']:.3g} s"
                if m != "airsfl" and air["hit"] > 0:
                    txt += f"\n{_ratio_txt(paired_speedup(s, air)[0])}"
                if s["hit"] < s["n"]:
                    txt += f"\n({s['hit']}/{s['n']})"
                ax.text(i, s["tmax"] * 1.25, txt, ha="center", va="bottom", fontsize=8,
                        fontweight="bold" if m == "airsfl" else None)
            ax.set_yscale("log")
            ymax = max((s["tmax"] for s in stats.values() if s["hit"] > 0), default=10)
            ax.set_ylim(top=ymax * 40)
            ax.set_xticks(x)
            ax.set_xticklabels(_xlabels(ORDER), rotation=25, ha="right")
            ax.set_ylabel(f"{'Uplink' if axis == 'uplink' else 'Training (uplink + computation)'} time (s)")
            ax.set_title(f"{PART_NAME[scheme]}: {axis} time to {100*A:.0f}% validation accuracy" if A
                         else PART_NAME[scheme], fontsize=10.5)
    ns = _nseeds(main)
    fig.suptitle(f"Time to the target accuracy at {ENV0['rho_db']:g} dB, {_pl()} ({_seed_txt(ns)}; labels: time, method time / AirSFL time "
                 f"paired per seed, (k/n) seeds that reached it)", y=1.01, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig6_time_to_target")


# ---------------------------------------------------------------------------
# Figure 7 + table_nsweep: number of clients (optional experiment `nsweep`)
# ---------------------------------------------------------------------------

def fig7_nsweep(main, nsweep):
    if nsweep.empty:
        return
    allr = pd.concat([f for f in (main, nsweep) if not f.empty], ignore_index=True)
    allr = allr[(allr.rho_db == 20.0) & (allr.tau == TAU0) & allr.cut.isin([CUT0, 0])]   # 0 = FL (no cut)
    Ns = tuple(sorted({int(n) for n in allr.N}))
    schemes = _schemes(allr)
    fig, axes = plt.subplots(2, len(schemes), figsize=(7.4 * len(schemes), 9.4), squeeze=False)
    x = np.arange(len(Ns))
    w = 0.84 / len(ORDER)
    rows = []
    for pi, scheme in enumerate(schemes):
        sub = allr[allr.partition == scheme]
        if TARGETS:
            A = TARGETS[0]
        else:
            tg = [rule_target(sub[sub.N == N], scheme) for N in Ns]
            if any(t is None for t in tg):
                continue
            A = min(tg)                      # common target reachable by every N
        for row, axis in enumerate(("uplink", "training")):
            ax = axes[row][pi]
            xcol = AXES[axis][0]
            for j, m in enumerate(ORDER):
                vals, ran = [], []
                for N in Ns:
                    dm = sub[(sub.N == N) & (sub.method == m)]
                    ran.append(not dm.empty)
                    if dm.empty:
                        vals.append(np.nan)
                        continue
                    s = ttt(dm, A, xcol)
                    vals.append(s["t"])
                    if row == 0:
                        rows.append({"partition": PART_NAME[scheme], "N": N, "method": STYLE[m]["label"],
                                     "UL s/round": float(dm.ul_s_per_round.iloc[0]),
                                     "training s/round": float(dm.e2e_s_per_round.iloc[0]),
                                     "target (%)": 100 * A, "reached": f"{s['hit']}/{s['n']}",
                                     "rounds to target": s["k"], "epochs to target": s["epochs"],
                                     "UL time to target (s)": s["t"],
                                     "training time to target (s)": ttt(dm, A, "training_time_s")["t"],
                                     "final test acc (%)": final_acc(dm)[0]})
                xs = x + (j - (len(ORDER) - 1) / 2) * w
                ax.bar(xs, vals, width=w, color=STYLE[m]["color"], edgecolor="k", lw=0.4, label=STYLE[m]["label"])
                for xi, v, r in zip(xs, vals, ran):
                    if r and np.isnan(v):
                        ax.text(xi, 0.02, "×", ha="center", color=STYLE[m]["color"], fontsize=10,
                                transform=ax.get_xaxis_transform())
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels([f"N={N}" for N in Ns])
            ax.set_ylabel(f"{axis.capitalize()} time to target (s)")
            ax.set_title(f"{PART_NAME[scheme]}: {axis} time to {100*A:.0f}%  (× = not reached)", fontsize=10.5)
    axes[0][0].legend(fontsize=7.5, ncol=2, loc="upper left")
    fig.suptitle(f"Client count at fixed Nr=64 and W=1.8 MHz ({_pl()}): time to a common target", y=1.01, fontsize=11)
    fig.tight_layout()
    _export_panels(fig, "7", [(axes[row][pi], f"{scheme}_{axis}_time_to_target") for pi, scheme in enumerate(schemes)
                              for row, axis in enumerate(("uplink", "training"))])
    _save(fig, "fig7_nsweep")
    if rows:
        _write_table(pd.DataFrame(rows), "table_nsweep")


# ---------------------------------------------------------------------------
# Figure 8 + table_efficiency: analog efficiency sensitivity (recomputed, no re-run)
# ---------------------------------------------------------------------------

def fig8_efficiency(main, eps_list=(0.4, 0.6, 0.7)):
    schemes = _schemes(main)
    if not schemes:
        return
    cases = [(f"analog eps={e}", dict(eps_U=e, eps_A=e)) for e in eps_list] + \
            [("ideal payload-only (all eps=1)", dict(eps_U=1.0, eps_A=1.0, eps_D=1.0))]
    fig, axes = plt.subplots(1, len(schemes), figsize=(1.1 * len(ORDER) * len(schemes), 5.0), squeeze=False)
    colors = ["#fcbba1", "#fb6a4a", "#cb181d", "#67000d"]
    rows = []
    x = np.arange(len(ORDER))
    w = 0.8 / len(cases)
    for ax, scheme in zip(axes[0], schemes):
        A = primary_target(main, scheme)
        for ci, (name, ov) in enumerate(cases):
            vals = []
            for m in ORDER:
                dm = main[(main.partition == scheme) & (main.method == m)]
                if dm.empty:
                    vals.append(np.nan)
                    continue
                rt = _retime(m, dm, **ov)
                s = ttt(rt, A)
                vals.append(s["t"])
                rows.append({"partition": PART_NAME[scheme], "case": name, "method": STYLE[m]["label"],
                             "UL time to target (s)": s["t"], "training time to target (s)":
                             ttt(rt, A, "training_time_s")["t"], "reached": f"{s['hit']}/{s['n']}"})
            ax.bar(x + (ci - (len(cases) - 1) / 2) * w, vals, width=w, color=colors[ci], edgecolor="k", lw=0.4,
                   label=name)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(_xlabels(ORDER), rotation=25, ha="right")
        ax.set_ylabel("Uplink time to target (s)")
        ax.set_title(f"{PART_NAME[scheme]}: uplink time to {100*A:.0f}% (digital eps_D=0.6 unless ideal)"
                     if A else PART_NAME[scheme], fontsize=10.5)
        ax.legend(fontsize=7.5)
    fig.suptitle(f"Efficiency sensitivity at {ENV0['rho_db']:g} dB, {_pl()}: declared overhead factors (same training runs, "
                 f"time recomputed)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig8_efficiency")
    if rows:
        _write_table(pd.DataFrame(rows), "table_efficiency")


# ---------------------------------------------------------------------------
# Figure 9 + table_cut_tau: architectural trade-off (optional experiments `cuts`, `tau`)
# ---------------------------------------------------------------------------

def fig9_cut_tau(main, cuts, tau):
    if cuts.empty and tau.empty:
        return
    panels = [p for p, d in (("cut", cuts), ("tau", tau)) if not d.empty]
    schemes = _schemes(main)
    fig, axes = plt.subplots(2 * len(schemes), len(panels), figsize=(6.6 * len(panels), 4.2 * 2 * len(schemes)),
                             squeeze=False)
    rows = []
    dims_all = profiled_dims(B0)
    for i, scheme in enumerate(schemes):
        A = primary_target(main, scheme)
        for j, panel in enumerate(panels):
            for m in ORDER:
                pts = []
                if panel == "cut":
                    for c in (1, 2, 3, 4):
                        if m in ("airsfl", "sun_fdma_aircomp", "hybrid_zf_aircomp"):
                            src = main if c == CUT0 else cuts
                            dm = src[(src.partition == scheme) & (src.method == m) & (src.cut == c)]
                        elif m in ("digital_sflv1", "digital_sflv1_zf"):   # learning cut-independent:
                            dm = main[(main.partition == scheme) & (main.method == m)]   # retime the main run
                            dm = dm if dm.empty else _retime(m, dm, dims=dims_all[c], stage=c)
                        else:                           # FL: no cut
                            dm = main[(main.partition == scheme) & (main.method == m)]
                        if not dm.empty:
                            pts.append((c, dm))
                else:
                    for t in sorted(set([TAU0] + list(tau.tau.unique()))):
                        src = main if t == TAU0 else tau
                        dm = src[(src.partition == scheme) & (src.method == m) & (src.tau == t)]
                        if not dm.empty:
                            pts.append((t, dm))
                for row, axis in enumerate(("uplink", "training")):
                    ax = axes[2 * i + row][j]
                    stats = [(v, ttt(dm, A, AXES[axis][0])) for v, dm in pts]
                    for v, s in stats:
                        rows.append({"partition": PART_NAME[scheme], "sweep": panel, "value": v, "axis": axis,
                                     "method": STYLE[m]["label"], "reached": f"{s['hit']}/{s['n']}",
                                     "time to target (s)": s["t"]})
                    ok = [(v, s) for v, s in stats if s["hit"] > 0]
                    if ok:
                        st = STYLE[m]
                        ax.plot([v for v, _ in ok], [s["t"] for _, s in ok], color=st["color"], marker=st["marker"],
                                ls=st["ls"], lw=st["lw"], label=st["label"])
                    for v, s in stats:
                        if s["hit"] == 0:
                            _unreached(ax, v, STYLE[m]["color"])
                    ax.set_yscale("log")
                    ax.set_xlabel("cut after residual stage" if panel == "cut" else "local steps tau")
                    ax.set_ylabel(f"{axis.capitalize()} time to target (s)")
                    ax.set_title(f"{PART_NAME[scheme]}: {axis} time to {100*A:.0f}% vs {panel}" if A
                                 else PART_NAME[scheme], fontsize=10.5)
                    if panel == "cut":
                        ax.set_xticks([1, 2, 3, 4])
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig9_cut_tau")
    if rows:
        _write_table(pd.DataFrame(rows), "table_cut_tau")


# ---------------------------------------------------------------------------
# Figure 10 / 1d + table_compute_regimes: how much computation changes the picture
# ---------------------------------------------------------------------------

def fig10_compute_regimes(main):
    """Training time to the primary target under three accountings, recomputed from the same
    runs: uplink only, edge-GPU devices (AdaptSFL setting) and IoT-CPU devices (Sun et al.:
    16 FLOPs/cycle x U[0.1, 2] GHz, 20 GHz server). Plus fig1d: accuracy vs training time
    in the IoT-CPU setting, where computation dominates the round."""
    schemes = _schemes(main)
    if not schemes:
        return
    fig, axes = plt.subplots(1, len(schemes), figsize=(1.3 * len(ORDER) * len(schemes), 5.4), squeeze=False)
    colors = ["#fdae6b", "#6baed6", "#31a354"]
    x = np.arange(len(ORDER))
    w = 0.8 / len(PRESETS)
    rows = []
    for ax, scheme in zip(axes[0], schemes):
        A = primary_target(main, scheme)
        for pi, (name, preset) in enumerate(PRESETS):
            stats = {}
            for m in ORDER:
                dm = main[(main.partition == scheme) & (main.method == m)]
                if dm.empty:
                    continue
                rt = _per_run(dm, lambda g: _with_compute(g, preset))
                stats[m] = (ttt(rt, A, "training_time_s"), float(rt.e2e_s_per_round.iloc[0]))
            vals = [stats[m][0]["t"] if m in stats else np.nan for m in ORDER]
            xs = x + (pi - (len(PRESETS) - 1) / 2) * w
            ax.bar(xs, vals, width=w, color=colors[pi], edgecolor="k", lw=0.4, label=name)
            air = stats["airsfl"][0] if "airsfl" in stats else None
            for xi, m, v in zip(xs, ORDER, vals):
                if m not in stats:
                    continue
                s, per_round = stats[m]
                ratio = paired_speedup(s, air)[0] if air is not None and air["hit"] > 0 else np.nan
                if np.isnan(v):
                    ax.text(xi, 0.03, "×", ha="center", fontsize=10, transform=ax.get_xaxis_transform())
                elif m != "airsfl" and not np.isnan(ratio):
                    ax.text(xi, v * 1.15, _ratio_txt(ratio), ha="center", va="bottom", fontsize=7, rotation=90)
                rows.append({"partition": PART_NAME[scheme], "accounting": name, "method": STYLE[m]["label"],
                             "s/round": per_round, "reached": f"{s['hit']}/{s['n']}",
                             "time to target (s)": s["t"], "x AirSFL (paired)": ratio})
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(_xlabels(ORDER), rotation=25, ha="right")
        ax.set_ylabel("Time to target (s)")
        ax.set_title(f"{PART_NAME[scheme]}: time to {100*A:.0f}% validation accuracy\n"
                     f"(labels: method time / AirSFL time)" if A else PART_NAME[scheme], fontsize=10.5)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.set_ylim(top=ax.get_ylim()[1] * 30)
    fig.suptitle(f"How much computation changes the comparison at {ENV0['rho_db']:g} dB, {_pl()} (same training runs; "
                 f"downlinks ideal)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig10_compute_regimes")
    if rows:
        _write_table(pd.DataFrame(rows), "table_compute_regimes")

    # fig1d: accuracy vs training time with IoT-CPU devices
    name, preset = PRESETS[2]
    fig, axes = plt.subplots(1, len(schemes), figsize=(6.3 * len(schemes), 4.7), squeeze=False)
    for ax, scheme in zip(axes[0], schemes):
        A = primary_target(main, scheme)
        for m in ORDER:
            dm = main[(main.partition == scheme) & (main.method == m)]
            if dm.empty:
                continue
            rt = _per_run(dm, lambda g: _with_compute(g, preset))
            run = _plot_curve(ax, rt, STYLE[m], "training_time_s")
            _mark_target(ax, run, A, STYLE[m], "training_time_s")
        ax.set_xscale("log")
        ax.set_xlabel("Training time: uplink + computation (s)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"CIFAR-10, ResNet-18 — {PART_NAME[scheme]}"
                     + (f" (target {100 * A:.0f}%)" if A is not None else ""))
    _fig_legend(fig, axes, ncol=4)
    fig.suptitle(f"Accuracy vs training time at {ENV0['rho_db']:g} dB, {_pl()}, {name}", y=1.02, fontsize=11)
    _save(fig, "fig1d_acc_vs_training_time_iot")


# ---------------------------------------------------------------------------
# Figure 12 + table_digital_transport: sensitivity of the digital baselines to the transport
# ---------------------------------------------------------------------------

TRANSPORT_FAMILIES = [
    ("Digital SFL-V1", {"OFDMA": "digital_sflv1", "ZF": "digital_sflv1_zf"}),
    ("Hybrid SFL (digital act. + AirComp)", {"OFDMA": "sun_fdma_aircomp", "ZF": "hybrid_zf_aircomp"}),
    ("Digital FedAvg", {"OFDMA": "digital_fedavg", "ZF": "digital_fedavg_zf"}),
]
ACCESS_STYLE = {"OFDMA": ("#1f77b4", "S/N tones each, concurrent (paper)"),
                "ZF": ("#17becf", "all tones, concurrent, ZF separation")}


def fig12_digital_transport(sets):
    """Sensitivity of the digital baselines to their transport at the nominal SNR: access scheme
    (OFDMA = the paper's; multi-user ZF) x payload precision (FP16 = the default, trained with
    really rounded uploads; FP32). sets = {q: main data set drawn with q-bit payloads}. (a)
    uplink seconds per round (analytic), then the uplink time to the primary target per
    partition, each payload from its own runs. AirSFL (the same runs for both payloads) is the
    dashed line; labels = method / AirSFL (paired per seed)."""
    main = sets[ENV0["q_bits"]]
    schemes = _schemes(main)
    fams = [(f, ms) for f, ms in TRANSPORT_FAMILIES if ms["OFDMA"] in ORDER]   # families of the drawn methods
    dims = profiled_dims(B0)[CUT0]
    radios = {q: _radio(q_bits=q) for q in (32, 16)}
    t_air = uplink_time_per_round("airsfl", dims, radios[32], TAU0, _rates(radios[32]))
    combos = [(acc, q) for acc in ACCESS_STYLE for q in (32, 16)]
    w = 0.8 / len(combos)
    fig, axes = plt.subplots(1, 1 + len(schemes), figsize=(6.4 * (1 + len(schemes)), 5.4), squeeze=False)
    axes = axes[0]
    rows = []

    def draw(ax, fi, ci, val, lab_ratio):
        acc, q = combos[ci]
        col = ACCESS_STYLE[acc][0]
        x = fi + (ci - (len(combos) - 1) / 2) * w
        if np.isnan(val):
            ax.text(x, 0.02, "no\nrun", ha="center", fontsize=6, color=col, transform=ax.get_xaxis_transform())
            return
        ax.bar(x, val, width=w, color=col, alpha=1.0 if q == 32 else 0.55, hatch=None if q == 32 else "//",
               edgecolor="k", lw=0.4)
        if not np.isnan(lab_ratio):
            ax.text(x, val * 1.15, _ratio_txt(lab_ratio), ha="center", va="bottom", fontsize=6.5, rotation=90)

    # (a) per-round uplink time, analytic
    for fi, (fam, ms) in enumerate(fams):
        for ci, (acc, q) in enumerate(combos):
            t = uplink_time_per_round(ms[acc], dims, radios[q], TAU0, _rates(radios[q]))
            draw(axes[0], fi, ci, t, t / t_air)
            rows.append({"partition": "analytic (per round)", "family": fam, "access": acc, "payload": f"FP{q}",
                         "method": ms[acc], "UL s/round": t, "UL s/round / AirSFL": t / t_air})
    axes[0].axhline(t_air, color=STYLE["airsfl"]["color"], ls="--", lw=1.5)
    axes[0].set_title(f"(a) uplink seconds per round (cut {CUT0}, N=30, 20 dB); AirSFL = {t_air:.2f} s",
                      fontsize=10.5)
    axes[0].set_ylabel("Uplink seconds per round")
    # (b, c) uplink time to the primary target (learning from the runs of each payload)
    for pi, scheme in enumerate(schemes):
        ax = axes[1 + pi]
        A = primary_target(main, scheme)
        air = ttt(main[(main.partition == scheme) & (main.method == "airsfl")], A, "uplink_s")
        for fi, (fam, ms) in enumerate(fams):
            for ci, (acc, q) in enumerate(combos):
                src = sets.get(q, pd.DataFrame())
                dm = src[(src.partition == scheme) & (src.method == ms[acc])] if not src.empty else src
                if dm.empty:
                    draw(ax, fi, ci, np.nan, np.nan)
                    rows.append({"partition": PART_NAME[scheme], "family": fam, "access": acc, "payload": f"FP{q}",
                                 "method": ms[acc], "note": f"no FP{q} run"})
                    continue
                s = ttt(dm, A, "uplink_s")
                ratio = paired_speedup(s, air)[0] if air["hit"] > 0 and s["hit"] > 0 else np.nan
                draw(ax, fi, ci, s["t"] if s["hit"] > 0 else np.nan, ratio)
                acc_m, lo, hi = final_acc(dm)
                rows.append({"partition": PART_NAME[scheme], "family": fam, "access": acc, "payload": f"FP{q}",
                             "method": ms[acc], "seeds": s["n"], "UL s/round": float(dm.ul_s_per_round.iloc[0]),
                             "UL s/round / AirSFL": float(dm.ul_s_per_round.iloc[0]) / t_air,
                             "final test acc (%)": acc_m, "target val acc (%)": 100 * A if A else np.nan,
                             "reached": f"{s['hit']}/{s['n']}", "UL time to target (s)": s["t"],
                             "UL time / AirSFL (paired)": ratio,
                             "training time to target (s)": ttt(dm, A, "training_time_s")["t"]})
        if air["hit"] > 0:
            ax.axhline(air["t"], color=STYLE["airsfl"]["color"], ls="--", lw=1.5)
        ax.set_title(f"({'bc'[pi]}) {PART_NAME[scheme]}: uplink time to {100 * A:.0f}% val. acc."
                     + (f"; AirSFL = {air['t']:.3g} s" if air["hit"] > 0 else "") if A else PART_NAME[scheme],
                     fontsize=10.5)
        ax.set_ylabel("Uplink time to target (s)")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(top=ax.get_ylim()[1] * 20)
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels([f for f, _ in fams], fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACCESS_STYLE[a][0], ec="k", lw=0.4) for a in ACCESS_STYLE] + \
        [plt.Rectangle((0, 0), 1, 1, fc="white", ec="k", lw=0.4),
         plt.Rectangle((0, 0), 1, 1, fc="white", ec="k", lw=0.4, hatch="//"),
         plt.Line2D([], [], color=STYLE["airsfl"]["color"], ls="--", lw=1.5)]
    labels = [f"{a}: {ACCESS_STYLE[a][1]}" for a in ACCESS_STYLE] + \
        ["FP32 payload", "FP16 payload (default; rounded uploads)", "AirSFL"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle("Digital-transport sensitivity at 20 dB: access scheme x payload precision "
                 "(bar labels = method / AirSFL)", y=1.02, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig12_digital_transport")
    _write_table(pd.DataFrame(rows), "table_digital_transport")


def main():
    import argparse
    global RESULTS, FIG, TAB, TARGETS
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=None, help="results root (default examples/AirSFL/results)")
    p.add_argument("--lr", type=float, default=None, help="initial LR of the runs to load (default: chosen_lr.json)")
    p.add_argument("--augment", choices=["on", "off"], default=None,
                   help="augmentation setting of the runs to load (default: chosen_lr.json)")
    p.add_argument("--epochs", type=float, default=None,
                   help="epoch budget of the runs to load (default: the most common in `main`)")
    p.add_argument("--targets", nargs="+", type=float, default=None,
                   help="prespecified validation targets, e.g. 0.75 0.8 (first = primary)")
    p.add_argument("--client-tflops", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                   help="recompute computation time with client capabilities U[LO, HI] TFLOPS")
    p.add_argument("--server-tflops", type=float, default=None, help="recompute with this M-server TFLOPS")
    p.add_argument("--budget-uplink", type=float, default=None, help="fig2c budget on the uplink axis (s)")
    p.add_argument("--budget-training", type=float, default=None, help="fig2c budget on the training axis (s)")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=ALL_METHODS + EXTRA_METHODS,
                   help="methods to draw (default: AirSFL, AirComp-FL, the two hybrids and digital SFL-V1 (ZF / "
                        "OFDMA); add digital_fedavg (and digital_fedavg_zf) to draw digital FedAvg. The ZF variants "
                        "are derived from the digital_sflv1 / sun_fdma_aircomp / digital_fedavg runs)")
    p.add_argument("--digital-q", nargs="+", type=int, default=[16], choices=[16, 32],
                   help="digital payload(s) of the drawn figure sets: 16 = the FP16 runs (default) -> figures/, "
                        "tables/; 32 = the FP32 runs -> figures_q32/, tables_q32/ (--digital-q 16 32: both)")
    p.add_argument("--budget-mult", nargs="+", type=float, default=[1.0],
                   help="fig2c: budgets = base budget x each value, e.g. 1 3 10 (base = AirSFL's full run)")
    p.add_argument("--linear-snrs", nargs="+", type=float, default=None,
                   help="fig1e: SNR panels (default 20 dB and the lowest swept SNR)")
    p.add_argument("--linear-xmax", type=float, default=None,
                   help="fig1e: x range in uplink seconds (default AirComp-FL's full run at 20 dB)")
    p.add_argument("--sweep-curves", nargs="*", default=["epochs", "uplink"], choices=["epochs", "uplink", "training"],
                   help="learning curves of all methods at every SNR (fig2d) and path-gain spread (fig11b) on these "
                        "axes; pass the flag without values to skip them")
    p.add_argument("--curves-xscale", choices=["log", "linear"], default="log",
                   help="time-axis scale of the sweep learning curves")
    p.add_argument("--path-gain-spread", type=float, default=0.0,
                   help="load main/snr/... runs trained with this path-gain spread (dB; 0 = equal gains)")
    p.add_argument("--dirichlet-alpha", type=float, default=None,
                   help="Dirichlet concentration of the non-IID runs to load (default: the one in `main`)")
    p.add_argument("--panels", nargs="*", default=["1e", "2c", "2_snr", "7"], choices=["1e", "2c", "2_snr", "7"],
                   help="figures also saved panel by panel, one file per subfigure, into <results>/paper_results/"
                        "<1e|2c|2_snr|7>/ (FP16 set; the FP32 set -> paper_results_q32/); --panels alone: none")
    p.add_argument("--also", nargs="+", default=[], choices=["fig3b", "fig11"],
                   help="optional figures, off by default: fig3b (per-round uplink time vs cut, table_cuts) and fig11 "
                        "(path-gain experiment: fig11, fig11b, table_pathloss; loads the pathloss runs)")
    p.add_argument("--snr", nargs="+", type=float, default=[20.0],
                   help="operating SNR(s) of the main-point figures (fig1/1b/1c/1d, 3/3b/3c, 4, 6, 8, 10, table_main). "
                        "20 = the nominal set (all figures); any other swept SNR is rebuilt from the CSVs into "
                        "figures_snr<SNR>/ and tables_snr<SNR>/, e.g. --snr 20 -20")
    a = p.parse_args()
    ALSO.update(a.also)
    PANELS["figs"] = set(a.panels)
    BUDGET_MULTS[:] = a.budget_mult
    LINEAR.update(snrs=a.linear_snrs, xmax=a.linear_xmax)
    SWEEP_CURVES.update(axes=list(a.sweep_curves), xscale=a.curves_xscale)
    FILTER["path_gain_spread_db"] = float(a.path_gain_spread)
    if a.results:
        RESULTS = os.path.abspath(a.results)
        FIG, TAB = os.path.join(RESULTS, "figures"), os.path.join(RESULTS, "tables")
    TARGETS = a.targets
    ORDER[:] = [m for m in ALL_METHODS + EXTRA_METHODS if m in a.methods]
    BUDGETS["uplink"], BUDGETS["training"] = a.budget_uplink, a.budget_training
    cal_path = os.path.join(RESULTS, "lr", "chosen_lr.json")
    cal = json.load(open(cal_path)) if os.path.exists(cal_path) else {}
    FILTER["base_lr"] = a.lr if a.lr is not None else (float(cal["lr"]) if "lr" in cal else None)
    FILTER["augment"] = (a.augment == "on") if a.augment else cal.get("augment")
    recompute = a.client_tflops is not None or a.server_tflops is not None
    if a.client_tflops:
        COMPUTE["client_tflops_lo"], COMPUTE["client_tflops_hi"] = a.client_tflops
    if a.server_tflops:
        COMPUTE["server_tflops"] = a.server_tflops
    skip = lambda e: ("path_gain_spread_db",) if e == "pathloss" else ()     # the spread is its sweep variable
    exps = ["main", "snr", "nr", "nsweep", "cuts", "tau", "fp16"] + (["pathloss"] if "fig11" in ALSO else [])
    d = {e: _filter(_read(e), skip(e)) for e in exps}
    d.setdefault("pathloss", pd.DataFrame())            # not loaded unless --also fig11
    budgets = sorted(set(d["main"].epochs_budget)) if not d["main"].empty else []
    FILTER["epochs_budget"] = a.epochs if a.epochs is not None else (
        float(d["main"].groupby("epochs_budget").run_id.nunique().idxmax()) if budgets else None)
    if len(budgets) > 1 and a.epochs is None:
        print(f"[plots] WARNING: `main` holds epoch budgets {budgets}; using {FILTER['epochs_budget']} "
              f"(choose with --epochs)")
    d = {e: _filter(df, skip(e)) for e, df in d.items()}
    m0 = d["main"]
    dir_alphas = (pd.to_numeric(m0.loc[m0.partition == "dirichlet", "dirichlet_alpha"], errors="coerce").dropna()
                  if not m0.empty and "dirichlet_alpha" in m0 else pd.Series(dtype=float))
    alpha = a.dirichlet_alpha if a.dirichlet_alpha is not None else (
        float(dir_alphas.round(6).mode().iloc[0]) if len(dir_alphas) else None)
    if len(set(dir_alphas.round(6))) > 1 and a.dirichlet_alpha is None:
        print(f"[plots] WARNING: `main` holds Dirichlet alphas {sorted(set(dir_alphas.round(6)))}; using {alpha:g} "
              f"(choose with --dirichlet-alpha)")
    if alpha is not None:
        d = {e: _keep_alpha(df, alpha) for e, df in d.items()}
        PART_NAME["dirichlet"] = f"Non-IID (Dir-{alpha:g})"
    if recompute:
        d = {e: _apply_compute(df) for e, df in d.items()}
    d = {e: _derive(df) for e, df in d.items()}
    print(f"[plots] results={RESULTS} | filter {FILTER} | compute {COMPUTE} "
          f"({'recomputed' if recompute else 'as stored'}) | targets = "
          f"{TARGETS or 'rule (95% of the error-free reference)'} | methods {ORDER}")
    sets = {q: _select_payload(d, q) for q in (16, 32)}     # both: fig12 compares the payloads
    for q in a.digital_q:
        ENV0["q_bits"] = q
        x = sets[q]
        tag = "" if q == 16 else f"_q{q}"                   # FP16 (default) -> figures/, FP32 -> figures_q32/
        _report_missing(d, x, q)
        if not x["main"].empty and not (x["main"].method == "errfree_ref").any():
            print(f"[plots] FP{q} set: no FP32 digital SFL-V1 run -> the error-free curve and the target rule use "
                  f"the drawn digital runs")
        for rho in a.snr:
            FIG = os.path.join(RESULTS, f"figures{tag}" + ("" if rho == 20.0 else f"_snr{rho:g}"))
            TAB = os.path.join(RESULTS, f"tables{tag}" + ("" if rho == 20.0 else f"_snr{rho:g}"))
            print(f"[plots] digital payload FP{q}, {rho:g} dB -> {FIG}, {TAB}")
            # subfigure files of the nominal pass: FP16 -> paper_results/, FP32 -> paper_results_q32/
            PANELS["dir"] = ("paper_results" + tag) if rho == 20.0 else None
            if rho == 20.0:                  # nominal point: every figure, sweeps included
                ENV0["rho_db"] = 20.0
                fig3_breakdown()
                fig4_overhead(x["main"])
                fig1_and_table(x["main"])
                fig2_snr(x["main"], x["snr"])
                fig1e_linear_time(x["main"], x["snr"])
                fig11_path_gains(x["main"], x["snr"], x["pathloss"])
                fig_sweep_curves(x["main"], x["snr"], x["pathloss"])
                fig5_nr(x["main"], x["nr"])
                fig6_time_to_target(x["main"])
                fig7_nsweep(x["main"], x["nsweep"])
                fig8_efficiency(x["main"])
                fig9_cut_tau(x["main"], x["cuts"], x["tau"])
                fig10_compute_regimes(x["main"])
                fig12_digital_transport({qq: sets[qq]["main"] for qq in (32, 16)})
                continue
            # another operating SNR: the main-point figures rebuilt from the CSVs (analog methods from
            # the SNR sweep at rho, digital methods retimed at rho) into figures_snr<rho>/, tables_snr<rho>/.
            # The N / cut / tau / antenna sweeps exist only at 20 dB and are not redrawn.
            if x["snr"].empty or not np.isclose(x["snr"].rho_db.astype(float), rho).any():
                print(f"[plots] --snr {rho:g}: no SNR-sweep runs at {rho:g} dB in {RESULTS}/snr -> skipped")
                continue
            main_rho = _main_at(x["main"], x["snr"], rho)
            ENV0["rho_db"] = rho
            print(f"[plots]   methods present at {rho:g} dB: "
                  f"{sorted(set(main_rho.method)) if not main_rho.empty else []}")
            fig3_breakdown()
            fig4_overhead(main_rho)
            fig1_and_table(main_rho)
            fig6_time_to_target(main_rho)
            fig8_efficiency(main_rho)
            fig10_compute_regimes(main_rho)
    ENV0["rho_db"], ENV0["q_bits"], PANELS["dir"] = 20.0, 16, None
    FIG, TAB = os.path.join(RESULTS, "figures"), os.path.join(RESULTS, "tables")


if __name__ == "__main__":
    main()
