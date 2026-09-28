"""
examples/AirSFL/plots.py: figures + tables for the AirSFL study from the CSVs written
by run_airsfl.py (works on whatever results exist; missing experiments are skipped).

Metric: accumulated UPLINK communication time (paper Eq. 25-26, 28): client -> M-server
activations + labels and client -> F-server model differences. Downlinks are ideal and
not timed; computation is excluded. Axis label: "accumulated uplink communication time".

Targets: prespecified validation accuracies (--targets, e.g. 0.6 0.7); the first is the
primary target of the bar figures. Without --targets, the rule A* = integer % below
95% of the error-free reference's final validation accuracy is used. Attainment is per
seed: the first evaluated checkpoint (round >= 1) with val_acc >= A. Failures are
reported (k/n seeds), never assigned a finite time; times are averaged over the seeds
that reached the target, and speed-ups are PAIRED per seed.

Learning of the digital methods is independent of SNR, cut, efficiency and antenna
count (ideal decoding); for those sweeps their time axis is recomputed analytically
from the same training runs (no re-run), exactly like the analog methods' time.

Only schema-3 CSVs trained with the calibrated initial LR (results/lr/chosen_lr.json,
or --lr) are loaded.

Figures -> <results>/figures/, tables -> <results>/tables/:
  fig1_acc_vs_uplink_time   test accuracy vs accumulated uplink seconds (IID | non-IID)
  fig1b_acc_vs_epochs       test accuracy vs global-epoch equivalents (learning only)
  fig2_snr                  final accuracy and uplink time to target vs SNR
  fig2b_airsfl_snr_curves   AirSFL accuracy vs uplink time at every SNR (+ digital SFL-V1)
  fig3_uplink_breakdown     per-round uplink phases (activation / labels / aggregation) vs N
  fig3b_uplink_vs_cut       per-round uplink time vs cut (analytic)
  fig4_comm_overhead        source-equivalent MB vs airtime per round; GB to target
  fig5_nr                   AirSFL accuracy vs uplink time for several Nr (if run)
  fig6_time_to_target       headline bars: uplink time to the primary target per method
  fig7_nsweep               N in {20,30,40}: per-round uplink + time to a common target (if run)
  fig8_efficiency           time to target vs analog efficiency (0.4/0.6/0.7, ideal 1.0)
  fig9_cut_tau              time to target vs cut and vs tau (if run)
  table_main / table_snr / table_cuts / table_nsweep / table_efficiency / table_cut_tau
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

from flsim.airsfl.timing import (RadioConfig, digital_rates, profiled_dims, source_equivalent_mb_per_round,
                                 uplink_time_breakdown, uplink_time_per_round)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIG = os.path.join(RESULTS, "figures")
TAB = os.path.join(RESULTS, "tables")
LR_FILTER = None            # initial LR whose runs are loaded (set in main)
TARGETS = None              # prespecified validation targets (set in main)
SCHEMA = 3

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
TAU0, CUT0 = 5, 2
XLAB = "Accumulated uplink communication time (s)"

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "legend.fontsize": 9, "figure.dpi": 110})


# ---------------------------------------------------------------------------
# loading, timing recomputation, seed statistics, targets
# ---------------------------------------------------------------------------

def _load(exp):
    dfs = []
    for f in sorted(glob.glob(os.path.join(RESULTS, exp, "*.csv"))):
        d = pd.read_csv(f)
        if "schema" not in d or int(d["schema"].iloc[0]) != SCHEMA:
            continue                              # older experiment generation: ignore
        if LR_FILTER is not None and not np.isclose(float(d.base_lr.iloc[0]), LR_FILTER):
            continue
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


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


def _retime(method, run, dims=None, **ov):
    """Same training run, uplink time axis recomputed under radio overrides / other dims."""
    r0 = run.iloc[0]
    radio = _radio_of(run, **ov)
    dims = dims or {"d_c": int(r0.d_c), "d_s": int(r0.d_s), "d_a": int(r0.d_a)}
    ul = uplink_time_per_round(method, dims, radio, int(r0.tau), _rates(radio))
    return run.assign(uplink_s=run["round"] * ul, cumulative_ul_s=run["round"] * ul, ul_s_per_round=ul)


def _curve(df):
    """Mean over paired seeds at each evaluation round (numeric columns)."""
    if df.empty:
        return df
    num = df.select_dtypes("number")
    return num.groupby("round", as_index=False).mean().sort_values("round")


def _seed_runs(df):
    return [g.sort_values("round") for _, g in df.groupby("seed")] if not df.empty else []


def _nseeds(df):
    return int(df.seed.nunique()) if not df.empty else 0


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


def time_to_target(run_df, A, xcol="uplink_s"):
    """First evaluated round (>= 1) with val_acc >= A -> (time, round)."""
    hit = run_df[(run_df.val_acc >= A) & (run_df["round"] >= 1)].sort_values("round")
    if hit.empty:
        return None, None
    return float(hit[xcol].iloc[0]), int(hit["round"].iloc[0])


def ttt(df, A, xcol="uplink_s"):
    """Per-seed attainment: n seeds, hits, per-seed times, mean/min/max time and rounds over hits."""
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


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[fig] {name}")


def _write_table(df, name, floatfmt=".3g"):
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(os.path.join(TAB, f"{name}.csv"), index=False)
    with open(os.path.join(TAB, f"{name}.md"), "w") as f:
        try:
            f.write(df.to_markdown(index=False, floatfmt=floatfmt))
        except ImportError:
            f.write(df.to_string(index=False))
    with open(os.path.join(TAB, f"{name}.tex"), "w") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.3g}", escape=True))
    print(f"[table] {name}")


def _plot_curve(ax, df, st, xcol, label=True, ls=None, color=None, lw=None):
    """Seed-mean curve (round 0 dropped) with a min-max seed band."""
    run = _curve(df)
    run = run[run["round"] >= 1]
    if run.empty:
        return run
    c = color or st["color"]
    ax.plot(run[xcol], 100 * run.test_acc, color=c, ls=ls or st["ls"], lw=lw or st["lw"],
            label=st["label"] if label is True else (label or None))
    if df.seed.nunique() > 1:
        g = df[df["round"] >= 1].groupby("round")["test_acc"]
        lo, hi = g.min().sort_index(), g.max().sort_index()
        ax.fill_between(run[xcol].values, 100 * lo.values, 100 * hi.values, color=c, alpha=0.15, lw=0)
    return run


def _schemes(df):
    return [s for s in ("iid", "dirichlet") if not df.empty and s in set(df.partition)]


def _target_lines(ax, targets):
    for A in targets:
        ax.axhline(100 * A, color="k", ls=":", lw=1)
        ax.text(0.01, 100 * A + 0.6, f"target {100*A:.0f}%", fontsize=8.5, transform=ax.get_yaxis_transform())


# ---------------------------------------------------------------------------
# Figure 1 / 1b + table_main  (roadmap B, C)
# ---------------------------------------------------------------------------

def fig1_and_table(main):
    schemes = _schemes(main)
    if not schemes:
        return
    fig, axes = plt.subplots(1, len(schemes), figsize=(6.3 * len(schemes), 4.7), squeeze=False)
    fige, axe = plt.subplots(1, len(schemes), figsize=(6.3 * len(schemes), 4.7), squeeze=False)
    rows = []
    for ax, a2, scheme in zip(axes[0], axe[0], schemes):
        sub = main[main.partition == scheme]
        tg = targets_for(main, scheme)
        stats = {}
        for m in ORDER:
            dm = sub[sub.method == m]
            if dm.empty:
                continue
            st = STYLE[m]
            run = _plot_curve(ax, dm, st, "uplink_s")
            _plot_curve(a2, dm, st, "epoch_equiv")
            if tg:
                t, k = time_to_target(run, tg[0])
                if t is not None:
                    ax.plot([t], [100 * float(run[run["round"] == k].test_acc.iloc[0])], marker=st["marker"],
                            color=st["color"], ms=9, mec="k", zorder=5)
            stats[m] = {A: ttt(dm, A) for A in tg}
        for m in ORDER:
            dm = sub[sub.method == m]
            if dm.empty:
                continue
            r0 = dm.iloc[0]
            acc, amin, amax = final_acc(dm)
            for A in tg:
                s = stats[m][A]
                sp = paired_speedup(stats["digital_sflv1"][A], s) if "digital_sflv1" in stats else (np.nan,) * 3 + (0,)
                rows.append({"partition": PART_NAME[scheme], "method": STYLE[m]["label"], "seeds": s["n"],
                             "UL s/round": float(r0.ul_s_per_round),
                             "act s/round": float(r0.activation_ul_s), "labels s/round": float(r0.labels_ul_s),
                             "agg s/round": float(r0.aggregation_ul_s), "MB/round": float(r0.mb_per_round),
                             "final test acc (%)": acc, "acc min": amin, "acc max": amax,
                             "target (%)": 100 * A, "reached": f"{s['hit']}/{s['n']}",
                             "rounds to target": s["k"], "epochs to target": s["epochs"],
                             "UL time to target (s)": s["t"], "MB to target": s["k"] * float(r0.mb_per_round),
                             "speed-up vs SFL-V1 (paired mean)": sp[0], "speed-up min": sp[1], "speed-up max": sp[2]})
        for a in (ax, a2):
            _target_lines(a, tg)
            a.set_ylabel("Test accuracy (%)")
            a.set_title(f"CIFAR-10, ResNet-18 — {PART_NAME[scheme]}")
        ax.set_xscale("log")
        ax.set_xlabel(XLAB)
        a2.set_xlabel("Global-epoch equivalents")
    ns = _nseeds(main)
    axes[0][0].legend(loc="lower right")
    axe[0][0].legend(loc="lower right")
    env = f"N={ENV0['N']}, Nr=64 (M & F), W=1.8 MHz, 20 dB, cut 2, tau=5; mean of {ns} seed{'s' * (ns > 1)}"
    fig.suptitle(f"Accuracy vs accumulated uplink time ({env}, band = min-max)", y=1.02, fontsize=11)
    fige.suptitle(f"Accuracy vs sample exposure ({env}); gaps = analog distortion", y=1.02, fontsize=11)
    _save(fig, "fig1_acc_vs_uplink_time")
    _save(fige, "fig1b_acc_vs_epochs")
    if rows:
        _write_table(pd.DataFrame(rows), "table_main")


# ---------------------------------------------------------------------------
# Figure 2 / 2b + table_snr  (roadmap D)
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


def fig2_snr(main, snr):
    schemes = _schemes(main)
    if not schemes:
        return
    rhos = sorted(set([20.0] + ([] if snr.empty else list(snr.rho_db.unique()))))
    rows = []
    fig, axes = plt.subplots(len(schemes), 2, figsize=(12, 4.4 * len(schemes)), squeeze=False)
    for i, scheme in enumerate(schemes):
        A = primary_target(main, scheme)
        allacc = []
        for m in ORDER:
            st = STYLE[m]
            acc, ttl = [], []
            for rho in rhos:
                run = _snr_runs(main, snr, scheme, m, rho)
                if run.empty:
                    continue
                a, amin, amax = final_acc(run)
                s = ttt(run, A)
                acc.append((rho, a, amin, amax))
                ttl.append((rho, s))
                allacc += [amin, amax]
                nsr = run[run["round"] >= 1]["act_nsr_db"].mean() if m == "airsfl" else np.nan
                rows.append({"partition": PART_NAME[scheme], "method": st["label"], "SNR (dB)": rho,
                             "seeds": s["n"], "final test acc (%)": a, "acc min": amin, "acc max": amax,
                             "target (%)": 100 * A if A else np.nan, "reached": f"{s['hit']}/{s['n']}",
                             "UL time to target (s)": s["t"], "AirSFL activation NSR (dB)": nsr})
            if acc:
                r, a, lo, hi = map(np.array, zip(*acc))
                axes[i][0].errorbar(r, a, yerr=[a - lo, hi - a], color=st["color"], marker=st["marker"],
                                    ls=st["ls"], lw=st["lw"], capsize=3, label=st["label"])
            ok = [(r, s) for r, s in ttl if s["hit"] > 0]
            if ok:
                r = np.array([x for x, _ in ok])
                t = np.array([s["t"] for _, s in ok])
                lo = np.array([s["tmin"] for _, s in ok])
                hi = np.array([s["tmax"] for _, s in ok])
                axes[i][1].errorbar(r, t, yerr=[t - lo, hi - t], color=st["color"], marker=st["marker"],
                                    ls=st["ls"], lw=st["lw"], capsize=3, label=st["label"])
                for rr, s in ok:
                    if s["hit"] < s["n"]:
                        axes[i][1].annotate(f"{s['hit']}/{s['n']}", (rr, s["t"]), textcoords="offset points",
                                            xytext=(4, 4), fontsize=7, color=st["color"])
            for rr, s in ttl:
                if s["hit"] == 0:
                    axes[i][1].text(rr, 0.03, "×", color=st["color"], fontsize=12, ha="center",
                                    transform=axes[i][1].get_xaxis_transform())
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
        axes[i][1].set_yscale("log")
        axes[i][1].set_xlabel("Reference SNR rho (dB)")
        axes[i][1].set_ylabel("Uplink time to target (s)")
        axes[i][1].set_title(f"{PART_NAME[scheme]}: uplink time to {100*A:.0f}% (validation)\n"
                             "(× = no seed reached it)" if A else PART_NAME[scheme], fontsize=10.5)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig2_snr")
    if rows:
        _write_table(pd.DataFrame(rows), "table_snr")

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
        _target_lines(ax, targets_for(main, scheme)[:1])
        ax.set_xscale("log")
        ax.set_xlabel(XLAB)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"SNR sweep — {PART_NAME[scheme]}")
        ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle("AirSFL: airtime is SNR-independent, distortion is not; digital: accuracy fixed, "
                 "rate falls at low SNR", y=1.02, fontsize=11)
    _save(fig, "fig2b_airsfl_snr_curves")


# ---------------------------------------------------------------------------
# Figure 3 / 3b + table_cuts: per-round uplink breakdown (analytic, roadmap A)
# ---------------------------------------------------------------------------

def fig3_breakdown(Ns=(20, 30, 40), stage=CUT0, tau=TAU0):
    dims = profiled_dims(16)[stage]
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
    axes[3].set_xticklabels([f"N={N}\nAirSFL {ratios[i]:.0f}x < SFL-V1" for i, N in enumerate(Ns)], fontsize=9)
    axes[3].set_title("All methods (log scale; FL has no activation phase)")
    axes[3].legend(fontsize=7.5, ncol=2, loc="upper left")
    fig.suptitle(f"Per-round uplink airtime by phase (cut {stage}, tau={tau}, Nr=64 at both servers, "
                 f"W=1.8 MHz, 20 dB, eps=0.6): analog time is independent of N, OFDMA time grows with N",
                 y=1.03, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig3_uplink_breakdown")

    # 3b: per-round time vs cut (N=30)
    r = _radio()
    rows = []
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    wb = 0.16
    dims_all = profiled_dims(16)
    for j, m in enumerate(ORDER):
        vals = []
        for s in (1, 2, 3, 4):
            b = uplink_time_breakdown(m, dims_all[s], r, tau, _rates(r))
            vals.append(b["total"])
            rows.append({"cut": s, "method": STYLE[m]["label"], "UL s/round": b["total"],
                         "activation": b["activation"], "labels": b["labels"], "aggregation": b["aggregation"],
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
    ax.set_title("Per-round uplink time vs cut (N=30): AirSFL < AirComp-FL iff tau*d_a < d_s")
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    _save(fig, "fig3b_uplink_vs_cut")
    _write_table(pd.DataFrame(rows), "table_cuts")


# ---------------------------------------------------------------------------
# Figure 4: communication overhead (source-equivalent MB vs airtime)
# ---------------------------------------------------------------------------

def fig4_overhead(main):
    r = _radio()
    dims = profiled_dims(16)[CUT0]
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(13, 4.8))
    for m in ORDER:
        mb = source_equivalent_mb_per_round(m, dims, r, TAU0)
        ul = uplink_time_per_round(m, dims, r, TAU0, _rates(r))
        st = STYLE[m]
        ax.scatter([mb], [ul], s=160, color=st["color"], marker=st["marker"], edgecolor="k", zorder=5,
                   label=st["label"])
        ax.annotate(f"{st['label'].replace(' (proposed)', '')}\n{ul:.2f} s" if ul < 10 else
                    f"{st['label'].replace(' (proposed)', '')}\n{ul:.0f} s",
                    (mb, ul), textcoords="offset points", xytext=(12, -4), fontsize=8.5)
    sfl_mb = source_equivalent_mb_per_round("airsfl", dims, r, TAU0)
    fl_mb = source_equivalent_mb_per_round("aircomp_fl", dims, r, TAU0)
    ax.axvline(sfl_mb, color="k", ls=":", lw=1)
    ratio = uplink_time_per_round("digital_sflv1", dims, r, TAU0, _rates(r)) / \
        uplink_time_per_round("airsfl", dims, r, TAU0, _rates(r))
    ax.text(sfl_mb + 0.08 * fl_mb, 12, f"three SFL variants:\nidentical {sfl_mb:.0f} MB/round,\n"
            f"AirSFL needs {ratio:.0f}x less airtime\nthan digital SFL-V1", fontsize=8.5, va="center",
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
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
    fig.tight_layout()
    _save(fig, "fig4_comm_overhead")


# ---------------------------------------------------------------------------
# Figure 5: antenna margin (optional experiment `nr`)
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
                        lw=2.0, ls="-", label=f"AirSFL, Nr={Nr} (Nr-N={Nr - int(dm.N.iloc[0])})")
        ref = main[(main.partition == scheme) & (main.method == "digital_sflv1")] if not main.empty else main
        if not ref.empty:
            _plot_curve(ax, ref, STYLE["digital_sflv1"], "uplink_s", label="Digital SFL-V1 (Nr=64)")
        _target_lines(ax, targets_for(main, scheme)[:1] if not main.empty else [])
        ax.set_xscale("log")
        ax.set_xlabel(XLAB)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"Antenna margin (20 dB) — {PART_NAME[scheme]}")
        ax.legend(fontsize=8, loc="lower right")
    _save(fig, "fig5_nr")


# ---------------------------------------------------------------------------
# Figure 6: uplink time to the primary target (headline bars)
# ---------------------------------------------------------------------------

def fig6_time_to_target(main):
    schemes = _schemes(main)
    if not schemes:
        return
    fig, axes = plt.subplots(1, len(schemes), figsize=(6.4 * len(schemes), 5.0), squeeze=False)
    x = np.arange(len(ORDER))
    for ax, scheme in zip(axes[0], schemes):
        A = primary_target(main, scheme)
        stats = {m: ttt(main[(main.partition == scheme) & (main.method == m)], A) for m in ORDER}
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
                txt += f"\nAirSFL {sp:.0f}x faster" if sp >= 10 else f"\nAirSFL {sp:.1f}x faster"
            if s["hit"] < s["n"]:
                txt += f"\n({s['hit']}/{s['n']} seeds)"
            ax.text(i, s["tmax"] * 1.25, txt, ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold" if m == "airsfl" else None)
        ax.set_yscale("log")
        ymax = max((s["tmax"] for s in stats.values() if s["hit"] > 0), default=10)
        ax.set_ylim(top=ymax * 40)
        ax.set_xticks(x)
        ax.set_xticklabels([STYLE[m]["label"].replace(" (proposed)", "") for m in ORDER], rotation=20, ha="right")
        ax.set_ylabel("Uplink communication time to target (s)")
        ax.set_title(f"{PART_NAME[scheme]}: time to {100*A:.0f}% validation accuracy" if A else PART_NAME[scheme])
    ns = _nseeds(main)
    fig.suptitle(f"Accumulated uplink time to the target (activations + labels + client->F-server "
                 f"updates; mean of {ns} seed{'s' * (ns > 1)}, bars = min-max, speed-ups paired per seed)",
                 y=1.02, fontsize=11)
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
    fig, axes = plt.subplots(1, len(schemes), figsize=(6.4 * len(schemes), 4.8), squeeze=False)
    x = np.arange(len(Ns))
    w = 0.16
    rows = []
    for pi, (ax, scheme) in enumerate(zip(axes[0], schemes)):
        sub = allr[allr.partition == scheme]
        if TARGETS:
            A = TARGETS[0]
        else:
            tg = [rule_target(sub[sub.N == N], scheme) for N in Ns]
            if any(t is None for t in tg):
                continue
            A = min(tg)                      # common target reachable by every N
        for j, m in enumerate(ORDER):
            vals = []
            for N in Ns:
                dm = sub[(sub.N == N) & (sub.method == m)]
                if dm.empty:
                    vals.append(np.nan)
                    continue
                s = ttt(dm, A)
                vals.append(s["t"])
                rows.append({"partition": PART_NAME[scheme], "N": N, "method": STYLE[m]["label"],
                             "UL s/round": float(dm.ul_s_per_round.iloc[0]), "target (%)": 100 * A,
                             "reached": f"{s['hit']}/{s['n']}", "rounds to target": s["k"],
                             "epochs to target": s["epochs"], "UL time to target (s)": s["t"],
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
        ax.set_ylabel("Uplink time to target (s)")
        ax.set_title(f"{PART_NAME[scheme]}: time to {100*A:.0f}%  (× = not reached)", fontsize=10.5)
    axes[0][0].legend(fontsize=7.5, ncol=2, loc="upper left")
    fig.suptitle("Client count at fixed Nr=64 and W=1.8 MHz: time to a common target", y=1.03, fontsize=11)
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
                s = ttt(_retime(m, dm, **ov), A)
                vals.append(s["t"])
                rows.append({"partition": PART_NAME[scheme], "case": name, "method": STYLE[m]["label"],
                             "UL time to target (s)": s["t"], "reached": f"{s['hit']}/{s['n']}"})
            ax.bar(x + (ci - (len(cases) - 1) / 2) * w, vals, width=w, color=colors[ci], edgecolor="k", lw=0.4,
                   label=name)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([STYLE[m]["label"].replace(" (proposed)", "") for m in ORDER], rotation=20, ha="right")
        ax.set_ylabel("Uplink time to target (s)")
        ax.set_title(f"{PART_NAME[scheme]}: time to {100*A:.0f}% (digital eps_D=0.6 unless ideal)"
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
    fig, axes = plt.subplots(len(schemes), len(panels), figsize=(6.6 * len(panels), 4.4 * len(schemes)),
                             squeeze=False)
    rows = []
    dims_all = profiled_dims(16)
    for i, scheme in enumerate(schemes):
        A = primary_target(main, scheme)
        for j, panel in enumerate(panels):
            ax = axes[i][j]
            for m in ORDER:
                pts = []
                if panel == "cut":
                    for c in (1, 2, 3, 4):
                        if m in ("airsfl", "sun_fdma_aircomp"):
                            src = main if c == CUT0 else cuts
                            dm = src[(src.partition == scheme) & (src.method == m) & (src.cut == c)]
                        elif m == "digital_sflv1":      # learning cut-independent -> retime the main run
                            dm = main[(main.partition == scheme) & (main.method == m)]
                            dm = dm if dm.empty else _retime(m, dm, dims=dims_all[c])
                        else:                           # FL: no cut
                            dm = main[(main.partition == scheme) & (main.method == m)]
                        if not dm.empty:
                            pts.append((c, ttt(dm, A)))
                else:
                    for t in sorted(set([TAU0] + list(tau.tau.unique()))):
                        src = main if t == TAU0 else tau
                        dm = src[(src.partition == scheme) & (src.method == m) & (src.tau == t)]
                        if not dm.empty:
                            pts.append((t, ttt(dm, A)))
                ok = [(v, s) for v, s in pts if s["hit"] > 0]
                for v, s in pts:
                    rows.append({"partition": PART_NAME[scheme], "sweep": panel, "value": v,
                                 "method": STYLE[m]["label"], "reached": f"{s['hit']}/{s['n']}",
                                 "UL time to target (s)": s["t"]})
                if ok:
                    st = STYLE[m]
                    ax.plot([v for v, _ in ok], [s["t"] for _, s in ok], color=st["color"], marker=st["marker"],
                            ls=st["ls"], lw=st["lw"], label=st["label"])
            ax.set_yscale("log")
            ax.set_xlabel("cut after residual stage" if panel == "cut" else "local steps tau")
            ax.set_ylabel("Uplink time to target (s)")
            ax.set_title(f"{PART_NAME[scheme]}: time to {100*A:.0f}% vs {panel}" if A else PART_NAME[scheme])
            if panel == "cut":
                ax.set_xticks([1, 2, 3, 4])
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig9_cut_tau")
    if rows:
        _write_table(pd.DataFrame(rows), "table_cut_tau")


def main():
    import argparse
    global RESULTS, FIG, TAB, LR_FILTER, TARGETS
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=None, help="results root (default examples/AirSFL/results)")
    p.add_argument("--lr", type=float, default=None,
                   help="load only runs with this initial LR (default: results/lr/chosen_lr.json)")
    p.add_argument("--targets", nargs="+", type=float, default=None,
                   help="prespecified validation targets, e.g. 0.6 0.7 (first = primary)")
    a = p.parse_args()
    if a.results:
        RESULTS = os.path.abspath(a.results)
        FIG, TAB = os.path.join(RESULTS, "figures"), os.path.join(RESULTS, "tables")
    LR_FILTER, TARGETS = a.lr, a.targets
    cal = os.path.join(RESULTS, "lr", "chosen_lr.json")
    if LR_FILTER is None and os.path.exists(cal):
        with open(cal) as f:
            LR_FILTER = float(json.load(f)["lr"])
    print(f"[plots] results={RESULTS} | initial lr = {LR_FILTER if LR_FILTER is not None else 'any'} | "
          f"targets = {TARGETS or 'rule (95% of the error-free reference)'}")
    d = {e: _load(e) for e in ("main", "snr", "nr", "nsweep", "cuts", "tau")}
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
