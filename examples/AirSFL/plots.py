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

Only CSVs of the current schema with the calibrated initial LR and augmentation setting
(results/lr/chosen_lr.json, or --lr / --augment) and one epoch budget are loaded.

Figures -> <results>/figures/, tables -> <results>/tables/:
  fig1_acc_vs_uplink_time     test accuracy vs accumulated uplink time
  fig1c_acc_vs_training_time  test accuracy vs training time (uplink + computation)
  fig1b_acc_vs_epochs         test accuracy vs global-epoch equivalents (learning only)
  fig2_snr                    final accuracy, uplink / training time to target vs SNR
  fig2b_airsfl_snr_curves     AirSFL accuracy vs uplink time at every SNR (+ digital SFL-V1)
  fig2c_acc_at_budget         test accuracy reached within a fixed time budget vs SNR
  fig3_uplink_breakdown       per-round uplink phases vs N (analytic)
  fig3b_uplink_vs_cut         per-round uplink time vs cut (analytic)
  fig3c_round_time            per-round uplink + computation, and its composition (analytic)
  fig4_comm_overhead          source-equivalent MB vs airtime per round; GB to target
  fig5_nr                     AirSFL vs M-server antennas (if run)
  fig6_time_to_target         uplink and training time to the primary target (bars)
  fig7_nsweep                 N in {20,30,40}: time to a common target (if run)
  fig8_efficiency             time to target vs analog efficiency (0.4/0.6/0.7, ideal 1.0)
  fig9_cut_tau                time to target vs cut and vs tau (if run)
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
from flsim.airsfl.timing import (RadioConfig, digital_rates, profiled_dims, source_equivalent_mb_per_round,
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

STYLE = {
    "airsfl":           dict(label="AirSFL (proposed)", color="#d62728", marker="o", lw=2.6, ls="-"),
    "digital_sflv1":    dict(label="Digital SFL-V1", color="#1f77b4", marker="s", lw=1.8, ls="-"),
    "sun_fdma_aircomp": dict(label="FDMA-AirComp SFL", color="#9467bd", marker="^", lw=1.8, ls="-"),
    "aircomp_fl":       dict(label="AirComp-FL", color="#2ca02c", marker="D", lw=1.8, ls="-"),
    "digital_fedavg":   dict(label="Digital FedAvg", color="#7f7f7f", marker="v", lw=1.6, ls="--"),
}
PART_NAME = {"iid": "IID", "dirichlet": "Non-IID (Dir-0.5)"}
ORDER = ["airsfl", "sun_fdma_aircomp", "aircomp_fl", "digital_sflv1", "digital_fedavg"]
SFL = ["airsfl", "sun_fdma_aircomp", "digital_sflv1"]
DIGITAL = ("digital_sflv1", "digital_fedavg")
ENV0 = dict(N=30, Nr=64, Nr_F=64, S=120, eps_D=0.6, eps_U=0.6, eps_A=0.6, rho_db=20.0, batch_size=16)
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


def _filter(df):
    if df.empty:
        return df
    keep = np.ones(len(df), dtype=bool)
    for col, val in FILTER.items():
        if val is None or col not in df:
            continue
        keep &= np.isclose(df[col].astype(float), float(val)) if isinstance(val, float) else (df[col] == val).values
    return df[keep]


_RATE_CACHE = {}


def _rates(radio):
    key = (radio.N, radio.Nr, radio.Nr_F, radio.S, radio.eps_D, radio.rho_db)
    if key not in _RATE_CACHE:
        _RATE_CACHE[key] = digital_rates(radio)
    return _RATE_CACHE[key]


def _radio(**kw):
    return RadioConfig(**{**ENV0, **kw})


def _radio_of(run, **ov):
    r0 = run.iloc[0]
    kw = dict(N=int(r0.N), Nr=int(r0.Nr), Nr_F=int(r0.Nr_F), S=int(r0.S), eps_D=float(r0.eps_D),
              eps_U=float(r0.eps_U), eps_A=float(r0.eps_A), rho_db=float(r0.rho_db), batch_size=int(r0.B))
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
    """Same training run, time axes recomputed under radio overrides / another cut."""
    r0 = run.iloc[0]
    radio = _radio_of(run, **ov)
    dims = dims or {"d_c": int(r0.d_c), "d_s": int(r0.d_s), "d_a": int(r0.d_a)}
    ul = uplink_time_per_round(method, dims, radio, int(r0.tau), _rates(radio))
    comp = _compute_per_round(method, run, stage)["total"] if stage is not None else float(r0.compute_s_per_round)
    return _set_times(run, ul, comp)


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
    ref = df[(df.partition == scheme) & (df.method == "digital_fedavg")]
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
    v = []
    for g in _seed_runs(df):
        ok = g[g[xcol] <= T]
        v.append(100 * float(ok.test_acc.iloc[-1]) if not ok.empty else np.nan)
    v = [x for x in v if not np.isnan(x)]
    return (float(np.mean(v)), float(min(v)), float(max(v))) if v else (np.nan, np.nan, np.nan)


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[fig] {name}")


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


def _plot_curve(ax, df, st, xcol, label=True, ls=None, color=None, lw=None, ycol="test_acc"):
    """Seed-mean curve (round 0 dropped) with a min-max seed band when >1 seed."""
    run = _curve(df)
    run = run[run["round"] >= 1]
    if run.empty:
        return run
    c = color or st["color"]
    ax.plot(run[xcol], 100 * run[ycol], color=c, ls=ls or st["ls"], lw=lw or st["lw"],
            label=st["label"] if label is True else (label or None))
    if df.seed.nunique() > 1:
        g = df[df["round"] >= 1].groupby("round")[ycol]
        lo, hi = g.min().sort_index(), g.max().sort_index()
        ax.fill_between(run[xcol].values, 100 * lo.values, 100 * hi.values, color=c, alpha=0.15, lw=0)
    return run


def _schemes(df):
    return [s for s in ("iid", "dirichlet") if not df.empty and s in set(df.partition)]


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
    env = f"N=30, Nr=64 (M & F), W=1.8 MHz, 20 dB, cut 2, tau=5; {_seed_txt(ns)}"
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
            if A is not None:
                ax.plot([], [], ls="", marker="o", mfc="w", mec="k", ms=8,
                        label=f"first checkpoint with validation acc >= {100*A:.0f}%")
            ax.set_xscale("log")
            ax.set_xlabel(xlab)
            ax.set_ylabel("Test accuracy (%)")
            ax.set_title(f"CIFAR-10, ResNet-18 — {PART_NAME[scheme]}")
        axes[0][0].legend(loc="lower right", fontsize=8)
        fig.suptitle(f"Accuracy vs {xlab.split(' (')[0].lower()} ({env})", y=1.02, fontsize=11)
        _save(fig, fname)

    fig, axes = plt.subplots(1, len(schemes), figsize=(6.3 * len(schemes), 4.7), squeeze=False)
    for ax, scheme in zip(axes[0], schemes):
        sub = main[main.partition == scheme]
        for m in ORDER:
            if not sub[sub.method == m].empty:
                _plot_curve(ax, sub[sub.method == m], STYLE[m], "epoch_equiv")
        ax.set_xlabel("Global-epoch equivalents")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"CIFAR-10, ResNet-18 — {PART_NAME[scheme]}")
    axes[0][0].legend(loc="lower right")
    fig.suptitle(f"Accuracy vs sample exposure ({env}); gaps = analog distortion", y=1.02, fontsize=11)
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
                rows.append({"partition": PART_NAME[scheme], "method": STYLE[m]["label"], "seeds": st_ul[m]["n"],
                             "UL s/round": float(r0.ul_s_per_round), "act s/round": float(r0.activation_ul_s),
                             "labels s/round": float(r0.labels_ul_s), "agg s/round": float(r0.aggregation_ul_s),
                             "compute s/round": float(r0.compute_s_per_round),
                             "client FP+BP s/round": float(r0.client_fp_s + r0.client_bp_s),
                             "server FP+BP s/round": float(r0.server_fp_s + r0.server_bp_s),
                             "training s/round": float(r0.e2e_s_per_round), "MB/round": float(r0.mb_per_round),
                             "final test acc (%)": acc, "acc min": amin, "acc max": amax,
                             "target val acc (%)": 100 * A, "reached": f"{st_ul[m]['hit']}/{st_ul[m]['n']}",
                             "rounds to target": st_ul[m]["k"], "epochs to target": st_ul[m]["epochs"],
                             "UL time to target (s)": st_ul[m]["t"],
                             "training time to target (s)": st_tr[m]["t"],
                             "MB to target": st_ul[m]["k"] * float(r0.mb_per_round),
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
                                                                          "aircomp_fl") else np.nan
                rows.append({"partition": PART_NAME[scheme], "method": st["label"], "SNR (dB)": rho,
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
        axes[i][0].set_title(f"{PART_NAME[scheme]}: final accuracy vs SNR\n"
                             "(digital learning is SNR-independent; bars = seed min-max)", fontsize=10.5)
        for j, axis in ((1, "uplink"), (2, "training")):
            axes[i][j].set_yscale("log")
            axes[i][j].set_xlabel("Reference SNR rho (dB)")
            axes[i][j].set_ylabel(f"{'Uplink' if axis == 'uplink' else 'Training'} time to target (s)")
            axes[i][j].set_title(f"{PART_NAME[scheme]}: {axis} time to {100*A:.0f}% val. acc.\n"
                                 "(x at the top = not reached)" if A else PART_NAME[scheme], fontsize=10.5)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
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
        for rho, ls in sorted({(min(rhos), "--"), (20.0, "-")}):
            run = _snr_runs(main, snr, scheme, "digital_sflv1", rho)
            if not run.empty:
                _plot_curve(ax, run, STYLE["digital_sflv1"], "uplink_s", label=f"Digital SFL-V1, {rho:g} dB",
                            ls=ls, lw=1.8)
        ax.set_xscale("log")
        ax.set_xlabel(AXES["uplink"][1])
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"SNR sweep — {PART_NAME[scheme]}")
        ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle("AirSFL: airtime is SNR-independent, distortion is not; digital: accuracy fixed, "
                 "rate falls at low SNR", y=1.02, fontsize=11)
    _save(fig, "fig2b_airsfl_snr_curves")

    # 2c: accuracy reached within a fixed time budget vs SNR (uplink and training axes)
    rows = []
    fig, axes = plt.subplots(len(schemes), 2, figsize=(12, 4.4 * len(schemes)), squeeze=False)
    for i, scheme in enumerate(schemes):
        for j, axis in enumerate(("uplink", "training")):
            xcol = AXES[axis][0]
            T = BUDGETS[axis]
            if T is None:                            # default: AirSFL's full-run time at 20 dB (seed mean)
                air = _snr_runs(main, snr, scheme, "airsfl", 20.0)
                if air.empty:
                    continue
                T = float(np.mean([g[xcol].iloc[-1] for g in _seed_runs(air)]))
            for m in ORDER:
                pts = []
                for rho in rhos:
                    run = _snr_runs(main, snr, scheme, m, rho)
                    if run.empty:
                        continue
                    a, lo, hi = acc_at_budget(run, T, xcol)
                    pts.append((rho, a, lo, hi))
                    rows.append({"partition": PART_NAME[scheme], "axis": axis, "budget (s)": T,
                                 "method": STYLE[m]["label"], "SNR (dB)": rho, "test acc at budget (%)": a,
                                 "min": lo, "max": hi})
                if pts:
                    st = STYLE[m]
                    r, a, lo, hi = map(np.array, zip(*pts))
                    axes[i][j].errorbar(r, a, yerr=[a - lo, hi - a], color=st["color"], marker=st["marker"],
                                        ls=st["ls"], lw=st["lw"], capsize=3, label=st["label"])
            axes[i][j].set_xlabel("Reference SNR rho (dB)")
            axes[i][j].set_ylabel("Test accuracy within the budget (%)")
            axes[i][j].set_title(f"{PART_NAME[scheme]}: accuracy after {T:.4g} s of {axis} time", fontsize=10.5)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig2c_acc_at_budget")
    if rows:
        _write_table(pd.DataFrame(rows), "table_budget")


# ---------------------------------------------------------------------------
# Figure 3 / 3b / 3c + table_cuts / table_round_time (analytic, roadmap A)
# ---------------------------------------------------------------------------

def _analytic_compute(m, N=30, stage=CUT0, tau=TAU0, u_min=0.0):
    """Computation per round in the default environment at the slowest possible client
    (u_min = 0 -> f_min = lo TFLOPS), for analytic figures."""
    lo, hi = COMPUTE["client_tflops_lo"], COMPUTE["client_tflops_hi"]
    return compute_time_breakdown(m, split_flops(stage if m in SPLIT_METHODS else None), N, B0, tau,
                                  (lo + (hi - lo) * u_min) * 1e12, COMPUTE["server_tflops"] * 1e12)


def fig3_breakdown(Ns=(20, 30, 40), stage=CUT0, tau=TAU0):
    dims = profiled_dims(B0)[stage]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    phases = [("activation", "#ff9896", "activations (client -> M-server)"),
              ("labels", "#ffbb78", "labels"),
              ("aggregation", "#aec7e8", "prefix differences (client -> F-server)")]
    x = np.arange(len(Ns))
    tot = {m: [] for m in ORDER}
    for m in ORDER:
        for N in Ns:
            r = _radio(N=N)
            tot[m].append(uplink_time_breakdown(m, dims, r, tau, _rates(r)))
    for ax, m in zip(axes[:3], SFL):
        bottom = np.zeros(len(Ns))
        for key, col, lab in phases:
            vals = np.array([b[key] for b in tot[m]])
            ax.bar(x, vals, bottom=bottom, color=col, edgecolor="k", lw=0.5, width=0.6, label=lab)
            bottom += vals
        for i, b in enumerate(tot[m]):
            ax.text(i, b["total"] * 1.02, f"{b['total']:.2f} s" if b["total"] < 10 else f"{b['total']:.0f} s",
                    ha="center", va="bottom", fontsize=9, fontweight="bold" if m == "airsfl" else None)
        ax.set_ylim(0, max(b["total"] for b in tot[m]) * 1.18)
        ax.set_xticks(x)
        ax.set_xticklabels([f"N={N}\n({120 // N} tones)" for N in Ns], fontsize=9)
        ax.set_title(STYLE[m]["label"], color=STYLE[m]["color"], fontweight="bold")
        ax.set_ylabel("Uplink seconds per round")
    axes[0].legend(fontsize=7.5, loc="lower right")
    w = 0.16
    for j, m in enumerate(ORDER):
        axes[3].bar(x + (j - 2) * w, [b["total"] for b in tot[m]], width=w, color=STYLE[m]["color"],
                    edgecolor="k", lw=0.4, label=STYLE[m]["label"])
    ratios = [tot["digital_sflv1"][i]["total"] / tot["airsfl"][i]["total"] for i in range(len(Ns))]
    axes[3].set_yscale("log")
    axes[3].set_ylim(0.5, 1e4)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels([f"N={N}\nSFL-V1 / AirSFL = {ratios[i]:.0f}" for i, N in enumerate(Ns)], fontsize=9)
    axes[3].set_title("All methods (log scale; FL has no activation phase)")
    axes[3].legend(fontsize=7.5, ncol=2, loc="upper left")
    fig.suptitle(f"Per-round uplink airtime by phase (cut {stage}, tau={tau}, Nr=64 at both servers, "
                 f"W=1.8 MHz, 20 dB, eps=0.6): analog time is independent of N, OFDMA time grows with N",
                 y=1.03, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig3_uplink_breakdown")

    # 3b: per-round uplink time vs cut (N=30)
    r = _radio()
    rows = []
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    wb = 0.16
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
                         "MB/round": source_equivalent_mb_per_round(m, dims_all[s], r, tau)})
        ax.bar(np.arange(4) + (j - 2) * wb, vals, width=wb, color=STYLE[m]["color"], edgecolor="k", lw=0.4,
               label=STYLE[m]["label"])
    ax.set_yscale("log")
    ax.set_ylim(0.5, 1e4)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([f"cut {s}{' (default)' if s == stage else ''}\nd_a={dims_all[s]['d_a']/1e6:.2f}M\n"
                        f"d_c={dims_all[s]['d_c']/1e6:.2f}M" for s in (1, 2, 3, 4)], fontsize=9)
    ax.set_ylabel("Uplink seconds per round")
    ax.set_title("Per-round uplink time vs cut (N=30): with equal analog efficiencies and\n"
                 "negligible labels, AirSFL < AirComp-FL when tau*d_a < d_s", fontsize=10.5)
    ax.legend(fontsize=8, ncol=3, loc="upper center")
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
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
    ax1.set_xticklabels([STYLE[m]["label"].replace(" (proposed)", "") for m in ORDER], rotation=20, ha="right")
    ax1.set_ylabel("Seconds per round")
    ax1.set_title("(a) Per-round time (cut 2, N=30, slowest client 1 TFLOPS, M-server 20 TFLOPS)", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")
    left = np.zeros(len(ORDER))
    for name, col in parts:
        frac = np.array([comp[m][name] / trn[i] for i, m in enumerate(ORDER)])
        ax2.barh(x, frac, left=left, color=col, edgecolor="k", lw=0.4, label=name)
        left += frac
    ax2.set_yticks(x)
    ax2.set_yticklabels([STYLE[m]["label"].replace(" (proposed)", "") for m in ORDER])
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
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(13, 4.8))
    for m in ORDER:
        mb = source_equivalent_mb_per_round(m, dims, r, TAU0)
        ul = uplink_time_per_round(m, dims, r, TAU0, _rates(r))
        st = STYLE[m]
        ax.scatter([mb], [ul], s=160, color=st["color"], marker=st["marker"], edgecolor="k", zorder=5,
                   label=st["label"])
        off = {"digital_sflv1": (12, 4), "sun_fdma_aircomp": (12, -22)}.get(m, (12, -4))   # avoid overlap
        ax.annotate(f"{st['label'].replace(' (proposed)', '')}\n{ul:.2f} s" if ul < 10 else
                    f"{st['label'].replace(' (proposed)', '')}\n{ul:.0f} s",
                    (mb, ul), textcoords="offset points", xytext=off, fontsize=8.5)
    sfl_mb = source_equivalent_mb_per_round("airsfl", dims, r, TAU0)
    fl_mb = source_equivalent_mb_per_round("aircomp_fl", dims, r, TAU0)
    ax.axvline(sfl_mb, color="k", ls=":", lw=1)
    t_air = uplink_time_per_round("airsfl", dims, r, TAU0, _rates(r))
    t_dig = uplink_time_per_round("digital_sflv1", dims, r, TAU0, _rates(r))
    ax.text(sfl_mb + 0.08 * fl_mb, 12, f"three SFL variants: identical\n{sfl_mb:.0f} MB/round of source data;\n"
            f"AirSFL airtime = {100 * t_air / t_dig:.2f}% of digital\nSFL-V1 (a {t_dig / t_air:.1f}-fold ratio)",
            fontsize=8.5, va="center", bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.set_yscale("log")
    ax.set_ylim(0.3, 5e3)
    ax.set_xlim(0, fl_mb * 1.3)
    ax.set_xlabel("Source-equivalent uplink volume per round (MB)")
    ax.set_ylabel("Uplink airtime per round (s)")
    ax.set_title(f"(a) Bytes vs airtime per round (cut {CUT0})")
    x = np.arange(len(ORDER))
    any_bar = False
    schemes = _schemes(main)
    width = 0.8 / max(1, len(schemes))
    for si, scheme in enumerate(schemes):
        A = primary_target(main, scheme)
        vals = []
        for m in ORDER:
            dm = main[(main.partition == scheme) & (main.method == m)]
            vals.append(ttt(dm, A)["k"] * float(dm.mb_per_round.iloc[0]) / 1e3 if not dm.empty else np.nan)
        xs = x + (si - (len(schemes) - 1) / 2) * width
        axb.bar(xs, vals, width=width, color=["#1f77b4", "#ff7f0e"][si], alpha=0.85, label=PART_NAME[scheme])
        for xi, v in zip(xs, vals):
            if np.isnan(v):
                axb.text(xi, 0.02, "not\nreached", ha="center", fontsize=7, transform=axb.get_xaxis_transform())
            else:
                any_bar = True
    axb.set_xticks(x)
    axb.set_xticklabels([STYLE[m]["label"].replace(" (proposed)", "") for m in ORDER], rotation=20, ha="right")
    axb.set_ylabel("Source-equivalent GB to target")
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
        ref = main[(main.partition == scheme) & (main.method == "digital_sflv1")] if not main.empty else main
        if not ref.empty:
            _plot_curve(ax, ref, STYLE["digital_sflv1"], "uplink_s", label="Digital SFL-V1 (Nr=64)")
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
    fig, axes = plt.subplots(2, len(schemes), figsize=(6.4 * len(schemes), 9.4), squeeze=False)
    x = np.arange(len(ORDER))
    for row, axis in enumerate(("uplink", "training")):
        xcol = AXES[axis][0]
        for ax, scheme in zip(axes[row], schemes):
            A = primary_target(main, scheme)
            stats = {m: ttt(main[(main.partition == scheme) & (main.method == m)], A, xcol) for m in ORDER}
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
                if m != "airsfl" and stats["airsfl"]["hit"] > 0:
                    sp = paired_speedup(s, stats["airsfl"])[0]
                    txt += f"\n{sp:.0f}x AirSFL" if sp >= 10 else f"\n{sp:.1f}x AirSFL"
                if s["hit"] < s["n"]:
                    txt += f"\n({s['hit']}/{s['n']} seeds)"
                ax.text(i, s["tmax"] * 1.25, txt, ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold" if m == "airsfl" else None)
            ax.set_yscale("log")
            ymax = max((s["tmax"] for s in stats.values() if s["hit"] > 0), default=10)
            ax.set_ylim(top=ymax * 40)
            ax.set_xticks(x)
            ax.set_xticklabels([STYLE[m]["label"].replace(" (proposed)", "") for m in ORDER], rotation=20,
                               ha="right")
            ax.set_ylabel(f"{'Uplink' if axis == 'uplink' else 'Training (uplink + computation)'} time (s)")
            ax.set_title(f"{PART_NAME[scheme]}: {axis} time to {100*A:.0f}% validation accuracy" if A
                         else PART_NAME[scheme], fontsize=10.5)
    ns = _nseeds(main)
    fig.suptitle(f"Time to the target accuracy ({_seed_txt(ns)}; labels: method time / AirSFL time, "
                 f"paired per seed)", y=1.01, fontsize=11)
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
    fig, axes = plt.subplots(2, len(schemes), figsize=(6.4 * len(schemes), 9.2), squeeze=False)
    x = np.arange(len(Ns))
    w = 0.16
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
                vals = []
                for N in Ns:
                    dm = sub[(sub.N == N) & (sub.method == m)]
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
                xs = x + (j - 2) * w
                ax.bar(xs, vals, width=w, color=STYLE[m]["color"], edgecolor="k", lw=0.4, label=STYLE[m]["label"])
                for xi, v in zip(xs, vals):
                    if np.isnan(v):
                        ax.text(xi, 0.02, "×", ha="center", color=STYLE[m]["color"], fontsize=10,
                                transform=ax.get_xaxis_transform())
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels([f"N={N}" for N in Ns])
            ax.set_ylabel(f"{axis.capitalize()} time to target (s)")
            ax.set_title(f"{PART_NAME[scheme]}: {axis} time to {100*A:.0f}%  (× = not reached)", fontsize=10.5)
    axes[0][0].legend(fontsize=7.5, ncol=2, loc="upper left")
    fig.suptitle("Client count at fixed Nr=64 and W=1.8 MHz: time to a common target", y=1.01, fontsize=11)
    fig.tight_layout()
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
    fig, axes = plt.subplots(1, len(schemes), figsize=(6.6 * len(schemes), 4.8), squeeze=False)
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
        ax.set_xticklabels([STYLE[m]["label"].replace(" (proposed)", "") for m in ORDER], rotation=20, ha="right")
        ax.set_ylabel("Uplink time to target (s)")
        ax.set_title(f"{PART_NAME[scheme]}: uplink time to {100*A:.0f}% (digital eps_D=0.6 unless ideal)"
                     if A else PART_NAME[scheme], fontsize=10.5)
        ax.legend(fontsize=7.5)
    fig.suptitle("Efficiency sensitivity: declared overhead factors (same training runs, time recomputed)",
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
                        if m in ("airsfl", "sun_fdma_aircomp"):
                            src = main if c == CUT0 else cuts
                            dm = src[(src.partition == scheme) & (src.method == m) & (src.cut == c)]
                        elif m == "digital_sflv1":      # learning cut-independent -> retime the main run
                            dm = main[(main.partition == scheme) & (main.method == m)]
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
    a = p.parse_args()
    if a.results:
        RESULTS = os.path.abspath(a.results)
        FIG, TAB = os.path.join(RESULTS, "figures"), os.path.join(RESULTS, "tables")
    TARGETS = a.targets
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
    d = {e: _filter(_read(e)) for e in ("main", "snr", "nr", "nsweep", "cuts", "tau")}
    budgets = sorted(set(d["main"].epochs_budget)) if not d["main"].empty else []
    FILTER["epochs_budget"] = a.epochs if a.epochs is not None else (
        float(d["main"].groupby("epochs_budget").run_id.nunique().idxmax()) if budgets else None)
    if len(budgets) > 1 and a.epochs is None:
        print(f"[plots] WARNING: `main` holds epoch budgets {budgets}; using {FILTER['epochs_budget']} "
              f"(choose with --epochs)")
    d = {e: _filter(df) for e, df in d.items()}
    if recompute:
        d = {e: _apply_compute(df) for e, df in d.items()}
    print(f"[plots] results={RESULTS} | filter {FILTER} | compute {COMPUTE} "
          f"({'recomputed' if recompute else 'as stored'}) | targets = "
          f"{TARGETS or 'rule (95% of the error-free reference)'}")
    fig3_breakdown()
    fig4_overhead(d["main"])
    fig1_and_table(d["main"])
    fig2_snr(d["main"], d["snr"])
    fig5_nr(d["main"], d["nr"])
    fig6_time_to_target(d["main"])
    fig7_nsweep(d["main"], d["nsweep"])
    fig8_efficiency(d["main"])
    fig9_cut_tau(d["main"], d["cuts"], d["tau"])


if __name__ == "__main__":
    main()
