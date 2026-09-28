"""
examples/AirSFL/plots.py: figures + tables for the AirSFL study from the CSVs
written by run_airsfl.py (works on whatever results exist so far).

Metric: modeled UPLINK communication seconds (paper Table 1). Every row already
contains BOTH client->server phases: the tau co-split activation uploads and the
once-per-round client->fed-server model(-difference) upload, plus labels. The
downlink is a reliable decoded link and is not timed here. Computation excluded.

Target accuracy A* (roadmap Sec. 5): the integer percentage point below 95% of
the noiseless reference's final clean VALIDATION accuracy (digital FedAvg, mean
over seeds, per partition), frozen for every method in that task. Attainment is
per seed: the first evaluated round >= 1 whose validation accuracy >= A*. Missed
targets are reported as failures, never assigned a finite time; times are
averaged over the seeds that reached A* and the success count is reported.

Only runs trained with the calibrated LR (results/lr/chosen_lr.json, or --lr)
are loaded, so results from an older LR never mix in.

Figures -> results/figures/, tables -> results/tables/:
  fig1_acc_vs_uplink_time   test accuracy vs uplink seconds (IID | non-IID), seed band
  fig1b_acc_vs_rounds       test accuracy vs rounds (learning, independent of transport speed)
  fig2_snr                  final accuracy and uplink time-to-target vs SNR
  fig2b_airsfl_snr_curves   AirSFL accuracy vs uplink time at every SNR (+ digital SFL-V1)
  fig3_uplink_latency       per-round uplink seconds by phase (cut 3) and vs cut (analytic)
  fig4_comm_overhead        source-equivalent MB vs airtime per round, and GB to target
  fig5_nr                   AirSFL accuracy vs uplink time for several Nr (if run)
  fig6_time_to_target       uplink time to A* per method (bars, speed-up annotated)
  fig7_nsweep               N in {20,30,40}: per-round uplink bars + time to a common target
  table_main / table_snr / table_cuts / table_nsweep  (.csv + .md + .tex)
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

from flsim.airsfl.timing import (RadioConfig, digital_spectral_efficiency, profiled_dims,
                                 source_equivalent_mb_per_round, uplink_time_breakdown,
                                 uplink_time_per_round)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIG = os.path.join(RESULTS, "figures")
TAB = os.path.join(RESULTS, "tables")
LR_FILTER = None            # base LR whose runs are loaded (set in main)

STYLE = {
    "airsfl":           dict(label="AirSFL (proposed)", color="#d62728", marker="o", lw=2.6, ls="-"),
    "digital_sflv1":    dict(label="Digital SFL-V1", color="#1f77b4", marker="s", lw=1.8, ls="-"),
    "sun_fdma_aircomp": dict(label="FDMA-AirComp SFL", color="#9467bd", marker="^", lw=1.8, ls="-"),
    "aircomp_fl":       dict(label="AirComp-FL", color="#2ca02c", marker="D", lw=1.8, ls="-"),
    "digital_fedavg":   dict(label="Digital FedAvg", color="#7f7f7f", marker="v", lw=1.6, ls="--"),
}
PART_NAME = {"iid": "IID", "dirichlet": "Non-IID (Dir-0.5)"}
ORDER = ["airsfl", "sun_fdma_aircomp", "aircomp_fl", "digital_sflv1", "digital_fedavg"]
DIGITAL = ("digital_sflv1", "digital_fedavg")
MAIN_RHO, MAIN_N, MAIN_NR = 20.0, 30, 128

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "legend.fontsize": 9, "figure.dpi": 110})


# ---------------------------------------------------------------------------
# loading, seed statistics, target attainment
# ---------------------------------------------------------------------------

def _load(exp):
    dfs = []
    for f in sorted(glob.glob(os.path.join(RESULTS, exp, "*.csv"))):
        d = pd.read_csv(f)
        if "base_lr" not in d:                   # written before the current schedule: ignore
            continue
        if LR_FILTER is not None and not np.isclose(float(d.base_lr.iloc[0]), LR_FILTER):
            continue
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _curve(df):
    """Mean over paired seeds at each evaluation round (numeric columns)."""
    if df.empty:
        return df
    num = df.select_dtypes("number").drop(columns=["diverged"], errors="ignore")
    return num.groupby("round", as_index=False).mean().sort_values("round")


def _band(df, col):
    """Per-round min / max across seeds (for shaded bands)."""
    g = df.groupby("round")[col]
    return g.min().sort_index(), g.max().sort_index()


def _seed_runs(df):
    return [g.sort_values("round") for _, g in df.groupby("seed")] if not df.empty else []


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
        except ImportError:                      # `tabulate` not installed on this machine
            f.write(df.to_string(index=False))
    with open(os.path.join(TAB, f"{name}.tex"), "w") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.3g}", escape=True))
    print(f"[table] {name}")


def target_accuracy(df, scheme):
    """A*: integer % below 95% of the noiseless reference's final validation accuracy."""
    ref = df[(df.partition == scheme) & (df.method == "digital_fedavg")]
    if ref.empty:
        return None
    final_val = float(_curve(ref)["val_acc"].iloc[-1])
    return math.floor(100 * 0.95 * final_val) / 100.0


def time_to_target(run_df, A, xcol="uplink_s"):
    """First evaluated round (>= 1, i.e. after training started) with val_acc >= A."""
    hit = run_df[(run_df.val_acc >= A) & (run_df["round"] >= 1)].sort_values("round")
    if hit.empty:
        return None, None
    return float(hit[xcol].iloc[0]), int(hit["round"].iloc[0])


def ttt(df, A, xcol="uplink_s"):
    """Per-seed attainment. Returns n seeds, hits, and mean/min/max time and rounds over hits."""
    ts, ks = [], []
    runs = _seed_runs(df)
    for g in runs:
        t, k = time_to_target(g, A, xcol) if A is not None else (None, None)
        if t is not None:
            ts.append(t)
            ks.append(k)
    out = {"n": len(runs), "hit": len(ts), "t": np.nan, "tmin": np.nan, "tmax": np.nan, "k": np.nan}
    if ts:
        out.update(t=float(np.mean(ts)), tmin=float(min(ts)), tmax=float(max(ts)), k=float(np.mean(ks)))
    return out


def final_acc(df, col="test_acc"):
    """Per-seed final accuracy (%): mean, min, max."""
    v = [100 * float(g[col].iloc[-1]) for g in _seed_runs(df)]
    return (float(np.mean(v)), float(min(v)), float(max(v))) if v else (np.nan, np.nan, np.nan)


def _ul_per_round(method, run, rho):
    r0 = run.iloc[0]
    r = RadioConfig(N=int(r0.N), Nr=int(r0.Nr), S=int(r0.S), rho_db=rho, batch_size=16)
    c = digital_spectral_efficiency(r, np.random.RandomState(2026))
    dims = {"d_c": r0.d_c, "d_s": r0.d_s, "d_a": r0.d_a}
    return uplink_time_per_round(method, dims, r, 5, c_D=c)


def _plot_curve(ax, df, st, xcol, label=True, ls=None, color=None, lw=None, band=True):
    """Seed-mean curve (round 0 dropped on log axes) with a min-max seed band."""
    run = _curve(df)
    run = run[run["round"] >= 1]
    if run.empty:
        return run
    c = color or st["color"]
    ax.plot(run[xcol], 100 * run.test_acc, color=c, ls=ls or st["ls"], lw=lw or st["lw"],
            label=st["label"] if label is True else (label or None))
    if band and df.seed.nunique() > 1:
        lo, hi = _band(df[df["round"] >= 1], "test_acc")
        ax.fill_between(run[xcol].values, 100 * lo.values, 100 * hi.values, color=c, alpha=0.15, lw=0)
    return run


def _nseeds(df):
    return int(df.seed.nunique()) if not df.empty else 0


# ---------------------------------------------------------------------------
# Figure 1 / 1b + table_main
# ---------------------------------------------------------------------------

def fig1_and_table(main):
    if main.empty:
        return
    schemes = [s for s in ("iid", "dirichlet") if s in set(main.partition)]
    fig, axes = plt.subplots(1, len(schemes), figsize=(6.2 * len(schemes), 4.6), squeeze=False)
    figr, axr = plt.subplots(1, len(schemes), figsize=(6.2 * len(schemes), 4.6), squeeze=False)
    rows = []
    for ax, a2, scheme in zip(axes[0], axr[0], schemes):
        A = target_accuracy(main, scheme)
        sub = main[main.partition == scheme]
        for m in ORDER:
            dm = sub[sub.method == m]
            if dm.empty:
                continue
            st = STYLE[m]
            run = _plot_curve(ax, dm, st, "uplink_s")
            _plot_curve(a2, dm, st, "round")
            s = ttt(dm, A)
            if A is not None and not np.isnan(s["t"]):
                t_mean, k_mean = time_to_target(run, A)
                if t_mean is not None:
                    ax.plot([t_mean], [100 * float(run[run["round"] == k_mean].test_acc.iloc[0])],
                            marker=st["marker"], color=st["color"], ms=9, mec="k", zorder=5)
            acc, amin, amax = final_acc(dm)
            r0 = dm.iloc[0]
            rows.append({"partition": PART_NAME[scheme], "method": st["label"], "seeds": s["n"],
                         "UL s/round": float(r0.ul_s_per_round),
                         "UL activation s/round": float(r0.ul_activation_s),
                         "UL model s/round": float(r0.ul_model_s),
                         "MB/round": float(r0.mb_per_round),
                         "final test acc (%)": acc, "acc min": amin, "acc max": amax,
                         "target A* (%)": 100 * A if A is not None else np.nan,
                         "reached A*": f"{s['hit']}/{s['n']}",
                         "rounds to A*": s["k"], "UL time to A* (s)": s["t"],
                         "MB to A*": s["k"] * float(r0.mb_per_round)})
        for a, xlab in ((ax, "Modeled uplink communication time (s)"), (a2, "Communication round")):
            if A is not None:
                a.axhline(100 * A, color="k", ls=":", lw=1)
                a.text(0.02, 100 * A + 0.8, f"target {100*A:.0f}%", fontsize=9, transform=a.get_yaxis_transform())
            a.set_xlabel(xlab)
            a.set_ylabel("Test accuracy (%)")
            a.set_title(f"CIFAR-10, ResNet-18 — {PART_NAME[scheme]}")
        ax.set_xscale("log")
    ns = _nseeds(main)
    axes[0][0].legend(loc="lower right")
    axr[0][0].legend(loc="lower right")
    fig.suptitle(f"Accuracy vs uplink time (N={MAIN_N}, Nr={MAIN_NR}, W=0.9 MHz, rho=20 dB, cut 3; "
                 f"mean of {ns} seed{'s' if ns > 1 else ''}, band = min-max)", y=1.02)
    figr.suptitle("Accuracy vs rounds: the learning trajectories; gaps = analog distortion", y=1.02)
    _save(fig, "fig1_acc_vs_uplink_time")
    _save(figr, "fig1b_acc_vs_rounds")

    tab = pd.DataFrame(rows)
    if not tab.empty:
        key = "UL time to A* (s)"
        for scheme in tab.partition.unique():
            ref = tab[(tab.partition == scheme) & (tab.method == STYLE["digital_sflv1"]["label"])]
            if not ref.empty and not np.isnan(ref[key].iloc[0]):
                tab.loc[tab.partition == scheme, "speed-up vs digital SFL-V1"] = \
                    float(ref[key].iloc[0]) / tab.loc[tab.partition == scheme, key]
        _write_table(tab, "table_main")


# ---------------------------------------------------------------------------
# Figure 2 / 2b + table_snr (digital methods: SNR-independent learning, analytic time)
# ---------------------------------------------------------------------------

def _snr_runs(main, snr, scheme, m, rho):
    """Runs of method m at SNR rho. Digital learning is SNR-independent (reliable
    transport): reuse the 20 dB runs and recompute only their time axis."""
    base = main[(main.partition == scheme) & (main.method == m) & (main.rho_db == MAIN_RHO)]
    if m in DIGITAL:
        if base.empty:
            return base
        return base.assign(uplink_s=base["round"] * _ul_per_round(m, base, rho))
    if rho == MAIN_RHO:
        return base
    if snr.empty:
        return snr
    n0, nr0 = int(main.N.iloc[0]), int(main.Nr.iloc[0])          # same environment as `main`
    return snr[(snr.partition == scheme) & (snr.method == m) & (snr.rho_db == rho) &
               (snr.N == n0) & (snr.Nr == nr0)]


def fig2_snr(main, snr):
    if main.empty:
        return
    schemes = [s for s in ("iid", "dirichlet") if s in set(main.partition)]
    rhos = sorted(set([MAIN_RHO] + ([] if snr.empty else list(snr.rho_db.unique()))))
    rows = []
    fig, axes = plt.subplots(len(schemes), 2, figsize=(12, 4.4 * len(schemes)), squeeze=False)
    for i, scheme in enumerate(schemes):
        A = target_accuracy(main, scheme)
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
                rows.append({"partition": PART_NAME[scheme], "method": st["label"], "SNR (dB)": rho,
                             "seeds": s["n"], "final test acc (%)": a, "acc min": amin, "acc max": amax,
                             "reached A*": f"{s['hit']}/{s['n']}", "UL time to A* (s)": s["t"]})
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
                             f"(digital learning is SNR-independent; bars = seed min-max)", fontsize=10.5)
        axes[i][1].set_yscale("log")
        axes[i][1].set_xlabel("Reference SNR rho (dB)")
        axes[i][1].set_ylabel("Uplink time to target (s)")
        axes[i][1].set_title(f"{PART_NAME[scheme]}: uplink time to {100*A:.0f}%\n(× = no seed reached it)"
                             if A is not None else PART_NAME[scheme], fontsize=10.5)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig2_snr")
    if rows:
        _write_table(pd.DataFrame(rows), "table_snr")

    # 2b: AirSFL learning curves at every SNR, with digital SFL-V1 at the best/worst SNR
    fig, axes = plt.subplots(1, len(schemes), figsize=(6.4 * len(schemes), 4.6), squeeze=False)
    cmap = plt.get_cmap("Reds")
    for ax, scheme in zip(axes[0], schemes):
        A = target_accuracy(main, scheme)
        for j, rho in enumerate(rhos):
            run = _snr_runs(main, snr, scheme, "airsfl", rho)
            if run.empty:
                continue
            col = cmap(0.3 + 0.7 * j / max(1, len(rhos) - 1))
            _plot_curve(ax, run, STYLE["airsfl"], "uplink_s", label=f"AirSFL, {rho:g} dB", color=col,
                        lw=2.0, ls="-")
        for rho, ls in ((min(rhos), "--"), (MAIN_RHO, "-")):
            run = _snr_runs(main, snr, scheme, "digital_sflv1", rho)
            if not run.empty:
                _plot_curve(ax, run, STYLE["digital_sflv1"], "uplink_s", label=f"Digital SFL-V1, {rho:g} dB",
                            ls=ls, lw=1.8)
            if min(rhos) == MAIN_RHO:
                break
        if A is not None:
            ax.axhline(100 * A, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("Modeled uplink communication time (s)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"SNR sweep — {PART_NAME[scheme]}")
        ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle("AirSFL trades analog distortion for airtime; digital keeps accuracy but slows down "
                 "at low SNR", y=1.02)
    _save(fig, "fig2b_airsfl_snr_curves")


# ---------------------------------------------------------------------------
# Figure 3 + table_cuts: per-round uplink latency (analytic, N=30 environment)
# ---------------------------------------------------------------------------

def fig3_latency(N=30, Nr=128, S=60, rho=20.0, tau=5, B=16):
    r = RadioConfig(N=N, Nr=Nr, S=S, rho_db=rho, batch_size=B)
    c = digital_spectral_efficiency(r, np.random.RandomState(2026))
    dims_all = profiled_dims(B)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0), gridspec_kw={"width_ratios": [1, 1.45]})

    # (a) grouped bars on a log axis (stacking on a log axis would hide small parts)
    names = [STYLE[m]["label"].replace(" (proposed)", "") for m in ORDER]
    br = [uplink_time_breakdown(m, dims_all[3], r, tau, c_D=c) for m in ORDER]
    act = np.array([b["activation"] for b in br])
    mod = np.array([b["model"] for b in br])
    tot = np.array([b["total"] for b in br])
    x = np.arange(len(ORDER))
    w = 0.38
    ax1.bar(x - w / 2, np.where(act > 0, act, np.nan), width=w, color="#ff9896", edgecolor="k", lw=0.5,
            label=f"co-split activation uplink ({tau} per round)")
    ax1.bar(x + w / 2, mod, width=w, color="#aec7e8", edgecolor="k", lw=0.5,
            label="client -> fed-server model uplink")
    for i, t in enumerate(tot):
        ax1.text(i, max(act[i], mod[i]) * 1.6, f"total\n{t:.2f} s" if t < 10 else f"total\n{t:.0f} s",
                 ha="center", fontsize=9, fontweight="bold" if ORDER[i] == "airsfl" else None)
    ax1.set_yscale("log")
    ax1.set_ylim(0.3, tot.max() * 12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylabel("Uplink seconds per round")
    ax1.set_title("(a) Per-round uplink latency by phase (cut 3)")
    ax1.legend(fontsize=8, loc="upper left")

    # (b) all methods at every cut
    rows = []
    wb = 0.16
    for j, m in enumerate(ORDER):
        vals = []
        for stage in (1, 2, 3, 4):
            t = uplink_time_per_round(m, dims_all[stage], r, tau, c_D=c)
            vals.append(t)
            rows.append({"cut": stage, "method": STYLE[m]["label"], "UL s/round": t,
                         "d_c": dims_all[stage]["d_c"], "d_s": dims_all[stage]["d_s"],
                         "d_a": dims_all[stage]["d_a"],
                         "MB/round": source_equivalent_mb_per_round(m, dims_all[stage], r, tau)})
        ax2.bar(np.arange(4) + (j - 2) * wb, vals, width=wb, color=STYLE[m]["color"], edgecolor="k", lw=0.4,
                label=STYLE[m]["label"])
    ax2.set_yscale("log")
    ax2.set_ylim(1, 5e3)
    ax2.set_xticks(np.arange(4))
    ax2.set_xticklabels([f"cut {s}\nd_a={dims_all[s]['d_a']/1e6:.2f}M\nd_c={dims_all[s]['d_c']/1e6:.2f}M"
                         for s in (1, 2, 3, 4)], fontsize=9)
    ax2.set_ylabel("Uplink seconds per round")
    ax2.set_title("(b) Per-round uplink latency vs cut layer")
    ax2.legend(fontsize=8, ncol=3, loc="upper center")
    fig.suptitle(f"N={N}, Nr={Nr}, W={r.W_hz/1e6:.2f} MHz, rho={rho:g} dB, tau={tau}, B={B}, eps=0.8 "
                 f"(labels < 1 ms/round, omitted)", y=1.01)
    fig.tight_layout()
    _save(fig, "fig3_uplink_latency")
    _write_table(pd.DataFrame(rows), "table_cuts")


# ---------------------------------------------------------------------------
# Figure 4: communication overhead (source-equivalent MB vs airtime)
# ---------------------------------------------------------------------------

def fig4_overhead(main):
    r = RadioConfig(N=30, Nr=128, S=60, rho_db=20.0)
    c = digital_spectral_efficiency(r, np.random.RandomState(2026))
    dims = profiled_dims(16)[3]
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(13, 4.8))

    # (a) MB/round (x) vs airtime/round (y): same bytes, very different airtime
    for m in ORDER:
        mb = source_equivalent_mb_per_round(m, dims, r, 5)
        ul = uplink_time_per_round(m, dims, r, 5, c_D=c)
        st = STYLE[m]
        ax.scatter([mb], [ul], s=160, color=st["color"], marker=st["marker"], edgecolor="k", zorder=5,
                   label=st["label"])
        ax.annotate(f"{st['label'].replace(' (proposed)', '')}\n{ul:.2f} s" if ul < 10 else
                    f"{st['label'].replace(' (proposed)', '')}\n{ul:.0f} s",
                    (mb, ul), textcoords="offset points", xytext=(12, -4), fontsize=8.5)
    sfl_mb = source_equivalent_mb_per_round("airsfl", dims, r, 5)
    ax.axvline(sfl_mb, color="k", ls=":", lw=1)
    ratio = uplink_time_per_round("digital_sflv1", dims, r, 5, c_D=c) / uplink_time_per_round("airsfl", dims, r, 5, c_D=c)
    ax.text(620, 25, f"three SFL variants:\nidentical {sfl_mb:.0f} MB/round,\n"
            f"AirSFL needs {ratio:.0f}x less airtime\nthan digital SFL-V1", fontsize=8.5, va="center",
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.set_yscale("log")
    ax.set_ylim(0.5, 5e3)
    ax.set_xlim(0, 1700)
    ax.set_xlabel("Source-equivalent uplink volume per round (MB)")
    ax.set_ylabel("Uplink airtime per round (s)")
    ax.set_title("(a) Bytes vs airtime per round (cut 3)")

    # (b) volume needed to reach the target accuracy
    x = np.arange(len(ORDER))
    names = [STYLE[m]["label"].replace(" (proposed)", "") for m in ORDER]
    any_bar = False
    if not main.empty:
        schemes = [s for s in ("iid", "dirichlet") if s in set(main.partition)]
        width = 0.8 / max(1, len(schemes))
        for si, scheme in enumerate(schemes):
            A = target_accuracy(main, scheme)
            vals = []
            for m in ORDER:
                dm = main[(main.partition == scheme) & (main.method == m)]
                s = ttt(dm, A) if not dm.empty else {"k": np.nan}
                vals.append(s["k"] * float(dm.mb_per_round.iloc[0]) / 1e3 if not dm.empty else np.nan)
            xs = x + (si - (len(schemes) - 1) / 2) * width
            axb.bar(xs, vals, width=width, color=["#1f77b4", "#ff7f0e"][si], alpha=0.85, label=PART_NAME[scheme])
            for xi, v in zip(xs, vals):
                if np.isnan(v):
                    axb.text(xi, 0.02, "not\nreached", ha="center", fontsize=7, transform=axb.get_xaxis_transform())
                else:
                    any_bar = True
    axb.set_xticks(x)
    axb.set_xticklabels(names, rotation=20, ha="right")
    axb.set_ylabel("Source-equivalent GB to target")
    axb.set_title("(b) Uplink volume to reach the target accuracy")
    if any_bar:
        axb.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig4_comm_overhead")


# ---------------------------------------------------------------------------
# Figure 5: antennas (Nr) sweep (optional experiment)
# ---------------------------------------------------------------------------

def fig5_nr(nr):
    if nr.empty:
        return
    schemes = [s for s in ("iid", "dirichlet") if s in set(nr.partition)]
    fig, axes = plt.subplots(1, len(schemes), figsize=(6.2 * len(schemes), 4.5), squeeze=False)
    cmap = plt.get_cmap("Reds")
    for ax, scheme in zip(axes[0], schemes):
        sub = nr[nr.partition == scheme]
        nrs = sorted(set(sub[sub.method == "airsfl"].Nr))
        for i, Nr in enumerate(nrs):
            dm = sub[(sub.method == "airsfl") & (sub.Nr == Nr)]
            _plot_curve(ax, dm, STYLE["airsfl"], "uplink_s", color=cmap(0.35 + 0.6 * i / max(1, len(nrs) - 1)),
                        lw=2.0, ls="-", label=f"AirSFL, Nr={Nr} (Nr-N={Nr - int(dm.N.iloc[0])})")
        ref = sub[sub.method == "digital_sflv1"]
        if not ref.empty:
            _plot_curve(ax, ref, STYLE["digital_sflv1"], "uplink_s", label="Digital SFL-V1 (reliable)")
        ax.set_xscale("log")
        ax.set_xlabel("Modeled uplink communication time (s)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"Receive antennas, rho={sub.rho_db.iloc[0]:g} dB — {PART_NAME[scheme]}")
        ax.legend(fontsize=8, loc="lower right")
    _save(fig, "fig5_nr")


# ---------------------------------------------------------------------------
# Figure 6: uplink communication time to the target (the headline bar plot)
# ---------------------------------------------------------------------------

def fig6_time_to_target(main):
    if main.empty:
        return
    schemes = [s for s in ("iid", "dirichlet") if s in set(main.partition)]
    fig, axes = plt.subplots(1, len(schemes), figsize=(6.4 * len(schemes), 4.9), squeeze=False)
    x = np.arange(len(ORDER))
    for ax, scheme in zip(axes[0], schemes):
        A = target_accuracy(main, scheme)
        stats = {m: ttt(main[(main.partition == scheme) & (main.method == m)], A) for m in ORDER}
        t_air = stats["airsfl"]["t"]
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
            if m != "airsfl" and not np.isnan(t_air):
                txt += f"\n{s['t'] / t_air:.0f}x AirSFL" if s["t"] / t_air >= 10 else \
                       f"\n{s['t'] / t_air:.1f}x AirSFL"
            if s["hit"] < s["n"]:
                txt += f"\n({s['hit']}/{s['n']} seeds)"
            ax.text(i, s["tmax"] * 1.25, txt, ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold" if m == "airsfl" else None)
        ax.set_yscale("log")
        ymax = max((s["tmax"] for s in stats.values() if s["hit"] > 0), default=10)
        ax.set_ylim(top=ymax * 30)
        ax.set_xticks(x)
        ax.set_xticklabels([STYLE[m]["label"].replace(" (proposed)", "") for m in ORDER], rotation=20, ha="right")
        ax.set_ylabel("Uplink communication time to target (s)")
        ax.set_title(f"{PART_NAME[scheme]}: time to A* = {100*A:.0f}% (validation)" if A is not None
                     else PART_NAME[scheme])
    ns = _nseeds(main)
    fig.suptitle(f"Uplink time to the target accuracy (activations + client->fed-server model uploads; "
                 f"mean of {ns} seed{'s' if ns > 1 else ''}, bars = min-max)", y=1.02)
    fig.tight_layout()
    _save(fig, "fig6_time_to_target")


# ---------------------------------------------------------------------------
# Figure 7 + table_nsweep: number of clients N at fixed Nr and S
# ---------------------------------------------------------------------------

def fig7_nsweep(main, nsweep, Nr=128, S=60, rho=20.0, tau=5, B=16):
    """(a) analytic per-round uplink seconds vs N; (b, c) uplink time to a COMMON target
    per partition (the lowest A*(N) over the swept N, so every N aims at the same accuracy).
    N=30 runs come from `main`."""
    frames = [f for f in (main, nsweep) if not f.empty]
    allr = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not allr.empty:
        allr = allr[allr.rho_db == rho]
    Ns = tuple(sorted({int(n) for n in allr.N})) if not nsweep.empty else (20, 30, 40)
    schemes = [s for s in ("iid", "dirichlet") if not nsweep.empty and s in set(allr.partition)]
    fig, axes = plt.subplots(1, 1 + len(schemes), figsize=(6.3 * (1 + len(schemes)), 4.8), squeeze=False)
    axes = axes[0]
    x = np.arange(len(Ns))
    w = 0.16
    dims = profiled_dims(B)[3]

    # (a) analytic per-round uplink time
    per_round = {}
    for j, m in enumerate(ORDER):
        vals = []
        for N in Ns:
            r = RadioConfig(N=N, Nr=Nr, S=S, rho_db=rho, batch_size=B)
            c = digital_spectral_efficiency(r, np.random.RandomState(2026))
            vals.append(uplink_time_per_round(m, dims, r, tau, c_D=c))
        per_round[m] = vals
        axes[0].bar(x + (j - 2) * w, vals, width=w, color=STYLE[m]["color"], edgecolor="k", lw=0.4,
                    label=STYLE[m]["label"])
    ratios = [per_round["digital_sflv1"][i] / per_round["airsfl"][i] for i in range(len(Ns))]
    axes[0].set_yscale("log")
    axes[0].set_ylim(1, 1e4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"N={N}, S_n={S / N:g}\nAirSFL {ratios[i]:.0f}x < SFL-V1"
                             for i, N in enumerate(Ns)], fontsize=8.5)
    axes[0].set_ylabel("Uplink seconds per round")
    axes[0].set_title("(a) Per-round uplink time (cut 3, analytic)", fontsize=10.5)
    axes[0].legend(fontsize=7.5, ncol=2, loc="upper left")

    # (b, c) uplink time to a common target
    rows = []
    for pi, (ax, scheme) in enumerate(zip(axes[1:], schemes)):
        sub = allr[allr.partition == scheme]
        targets = [target_accuracy(sub[sub.N == N], scheme) for N in Ns]
        if any(t is None for t in targets):
            continue
        A = min(targets)
        for j, m in enumerate(ORDER):
            vals = []
            for N, AN in zip(Ns, targets):
                dm = sub[(sub.N == N) & (sub.method == m)]
                if dm.empty:
                    vals.append(np.nan)
                    continue
                s = ttt(dm, A)
                vals.append(s["t"])
                acc, _, _ = final_acc(dm)
                rows.append({"partition": PART_NAME[scheme], "N": N, "method": STYLE[m]["label"],
                             "UL s/round": float(dm.ul_s_per_round.iloc[0]),
                             "common target (%)": 100 * A, "own A*(N) (%)": 100 * AN,
                             "reached": f"{s['hit']}/{s['n']}", "rounds to target": s["k"],
                             "UL time to target (s)": s["t"], "final test acc (%)": acc})
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
        ax.set_title(f"({'bc'[pi]}) {PART_NAME[scheme]}: time to common {100*A:.0f}%  (× = not reached)",
                     fontsize=10.5)
    fig.suptitle(f"Number of clients at fixed Nr={Nr}, W={S * 15e3 / 1e6:.2f} MHz, rho={rho:g} dB: "
                 "digital OFDMA splits the band, AirSFL reuses it", y=1.03)
    fig.tight_layout()
    _save(fig, "fig7_nsweep")
    if rows:
        tab = pd.DataFrame(rows)
        for (p, N), g in tab.groupby(["partition", "N"]):
            ref = g[g.method == STYLE["digital_sflv1"]["label"]]["UL time to target (s)"]
            if len(ref) and not np.isnan(ref.iloc[0]):
                idx = (tab.partition == p) & (tab.N == N)
                tab.loc[idx, "speed-up vs digital SFL-V1"] = float(ref.iloc[0]) / tab.loc[idx, "UL time to target (s)"]
        _write_table(tab, "table_nsweep")


def main():
    import argparse
    global RESULTS, FIG, TAB, LR_FILTER
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=None, help="results root (default examples/AirSFL/results)")
    p.add_argument("--lr", type=float, default=None,
                   help="load only runs with this base LR (default: results/lr/chosen_lr.json)")
    a = p.parse_args()
    if a.results:
        RESULTS = os.path.abspath(a.results)
        FIG, TAB = os.path.join(RESULTS, "figures"), os.path.join(RESULTS, "tables")
    LR_FILTER = a.lr
    cal = os.path.join(RESULTS, "lr", "chosen_lr.json")
    if LR_FILTER is None and os.path.exists(cal):
        with open(cal) as f:
            LR_FILTER = float(json.load(f)["lr"])
    print(f"[plots] results={RESULTS} | runs with base lr = {LR_FILTER if LR_FILTER is not None else 'any'}")
    main_df, snr_df, nr_df, ns_df = _load("main"), _load("snr"), _load("nr"), _load("nsweep")
    fig3_latency()
    fig4_overhead(main_df)
    fig1_and_table(main_df)
    fig2_snr(main_df, snr_df)
    fig5_nr(nr_df)
    fig6_time_to_target(main_df)
    fig7_nsweep(main_df, ns_df)


if __name__ == "__main__":
    main()
