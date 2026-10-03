"""
examples/AirSFL/paper_figures.py: the 7 paper figures of the AirSFL study, one file each, seed MEANS
only (no error bars, no seed bands), written to <results>/finalz/ together with one CSV per figure:

  fig1_noniid_final_accuracy_vs_snr            final test accuracy vs SNR (a ZF variant learns exactly
                                               like its OFDMA twin, so each pair is drawn once)
  fig2_noniid_accuracy_vs_uplink_time_20dB     test accuracy vs accumulated uplink time (linear), 20 dB
  fig3_noniid_uplink_time_to_target_vs_snr     uplink time to the target vs SNR
  fig4_noniid_accuracy_within_uplink_budget    test accuracy within AirSFL's full-run uplink time vs SNR
  fig5_noniid_uplink_time_to_target_vs_N       uplink time to a common target vs number of clients
  fig6_iid_accuracy_vs_uplink_time_20dB        as fig2, IID
  fig7_iid_uplink_time_to_target_vs_N          as fig5, IID

Two steps:
  1. DATA: every number is computed from the run CSVs with plots.py (FP16 digital payload, ZF variants
     derived, same targets / time-to-target / budget rules and seed handling as plots.py) and saved as
     finalz/<figure>.csv (+ <figure>_summary.csv for the two learning-curve figures);
  2. PLOT: the figures are drawn from those CSVs only. Change the look in the FORMAT / METHODS blocks
     below and redraw in seconds with --plot-only (no recomputation).

CSV columns: one column per method (names in COLUMN below), seed means; "<method> [reached]" = seeds
that reached the target / seeds; "seeds" = seeds per point (20 dB uses --main-seeds when given).
finalz/provenance.csv lists, for every point of every figure and every method, the digital payload,
the seeds and the run files it is computed from.

Usage (from the repo root):
  python examples/AirSFL/paper_figures.py --results examples/AirSFL/results_final_s1 --main-seeds 11
  python examples/AirSFL/paper_figures.py --results examples/AirSFL/results_final_s1 --plot-only
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------------
# FORMAT -- edit freely, then: python examples/AirSFL/paper_figures.py --results <folder> --plot-only
# ---------------------------------------------------------------------------------------------------
FORMAT = dict(
    figsize=(6.4, 4.6),          # inches per figure
    dpi=300,                     # PNG resolution
    file_types=("png", "pdf"),
    font=11,                     # base font size
    title=True,                  # panel titles on/off (papers often put this in the caption instead)
    legend="below",              # "below" the axes, "inside" (best spot) or "none"
    legend_font=8.5,
    legend_cols=2,
    grid_alpha=0.3,
    marker_size=6,
    curve_markers=8,             # markers per learning curve (fig2 / fig6)
    extend_curves=True,          # dotted line at the final accuracy after a run's epoch budget is used up
    # fig2 / fig6 inset: the slow baselines over their WHOLE run (the main axis stops at AirComp-FL's run)
    inset_methods=("sun_fdma_aircomp", "digital_sflv1"),   # () = no inset
    inset_bounds={                                          # (left, bottom, width, height) in axes fractions
        "fig2_noniid_accuracy_vs_uplink_time_20dB": (0.42, 0.45, 0.48, 0.30),
        "fig6_iid_accuracy_vs_uplink_time_20dB": (0.42, 0.55, 0.48, 0.26),
    },
    inset_font=7.5,
    inset_title="Full training",                            # "" = no title
    snr_label="Reference SNR $\\rho$ (dB)",
)

# drawing order; label, color, marker, line style, line width (per method)
METHODS = {
    "airsfl":            dict(label="AirSFL (proposed)",       color="#d62728", marker="o", ls="-",  lw=2.6),
    "aircomp_fl":        dict(label="AirComp-FL",              color="#2ca02c", marker="D", ls="-",  lw=1.8),
    "hybrid_zf_aircomp": dict(label="Hybrid ZF-AirComp SFL",   color="#8c564b", marker="X", ls="-.", lw=1.8),
    "sun_fdma_aircomp":  dict(label="Hybrid FDMA-AirComp SFL", color="#9467bd", marker="^", ls="-",  lw=1.8),
    "digital_sflv1_zf":  dict(label="Digital SFL-V1 (ZF)",     color="#17becf", marker="P", ls="-.", lw=1.8),
    "digital_sflv1":     dict(label="Digital SFL-V1 (OFDMA)",  color="#1f77b4", marker="s", ls="-",  lw=1.8),
}
# fig1 draws each ZF variant once with its OFDMA twin (identical learning), under this label
JOINT = {"sun_fdma_aircomp": "Hybrid FDMA / ZF-AirComp SFL", "digital_sflv1": "Digital SFL-V1 (OFDMA / ZF)"}
TWINS = {"hybrid_zf_aircomp": "sun_fdma_aircomp", "digital_sflv1_zf": "digital_sflv1"}

# CSV column name of each method (fixed: the plot step finds the data by these names)
COLUMN = {"airsfl": "AirSFL", "aircomp_fl": "AirComp-FL", "hybrid_zf_aircomp": "Hybrid ZF-AirComp",
          "sun_fdma_aircomp": "Hybrid FDMA-AirComp", "digital_sflv1_zf": "Digital SFL-V1 (ZF)",
          "digital_sflv1": "Digital SFL-V1 (OFDMA)"}
TRAINED = list(METHODS)

FIGS = {   # file name -> (partition, kind)
    "fig1_noniid_final_accuracy_vs_snr": ("dirichlet", "final_acc_vs_snr"),
    "fig2_noniid_accuracy_vs_uplink_time_20dB": ("dirichlet", "curves_20dB"),
    "fig3_noniid_uplink_time_to_target_vs_snr": ("dirichlet", "uplink_time_to_target_vs_snr"),
    "fig4_noniid_accuracy_within_uplink_budget": ("dirichlet", "budget_vs_snr"),
    "fig5_noniid_uplink_time_to_target_vs_N": ("dirichlet", "time_to_target_vs_N"),
    "fig6_iid_accuracy_vs_uplink_time_20dB": ("iid", "curves_20dB"),
    "fig7_iid_uplink_time_to_target_vs_N": ("iid", "time_to_target_vs_N"),
}


# ---------------------------------------------------------------------------------------------------
# 1. DATA (from the run CSVs, through plots.py)
# ---------------------------------------------------------------------------------------------------

def _seeds_txt(ns):
    ns = [n for n in ns if n]
    return "" if not ns else (str(min(ns)) if min(ns) == max(ns) else f"{min(ns)}-{max(ns)}")


def build_data(results, main_seeds, out):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import plots as P
    P.MAIN_SEEDS["seeds"] = set(main_seeds) if main_seeds else None
    d, _ = P.load_results(results)
    x = P._select_payload(d, 16)                       # FP16 digital payload
    P.ENV0["q_bits"] = 16
    P.TARGET_SRC["df"] = x["main"]                     # 95% target: every seed of FP16 digital SFL-V1
    main, snr, nsweep = x["main"], x["snr"], x["nsweep"]
    P._report_unpaired({**x, "main": P._main20(main)})
    rhos = sorted(set([20.0] + [float(r) for r in snr.rho_db.unique()]))
    os.makedirs(out, exist_ok=True)
    prov = []

    def note(fig, point, m, r):
        """Record which payload, seeds and run files a plotted point of method m comes from."""
        q = sorted(set(P._q_col(r)))
        prov.append({"figure": fig, "point": point, "method": COLUMN[m],
                     "digital payload": "/".join(f"FP{v}" for v in q) if m in P.PAYLOAD_METHODS else
                     ("none (analog; labels only)" if m == "airsfl" else "none (analog)"),
                     "seeds": ",".join(str(v) for v in sorted(set(r.seed.astype(int)))),
                     "run files": "; ".join(sorted(set(r.source_file)))})

    def runs(scheme, m, rho):
        return P._snr_runs(main, snr, scheme, m, rho)

    def final_acc_vs_snr(fig, scheme):
        rows = []
        for rho in rhos:
            row, ns = {"SNR (dB)": rho}, []
            for m in TRAINED:
                r = runs(scheme, m, rho)
                if not r.empty:
                    row[COLUMN[m]] = P.final_acc(r)[0]
                    ns.append(P._nseeds(r))
                    note(fig, f"{rho:g} dB", m, r)
            row["seeds"] = _seeds_txt(ns)
            rows.append(row)
        return pd.DataFrame(rows)

    def time_to_target_vs_snr(fig, scheme, xcol="training_time_s"):
        A = P.primary_target(main, scheme)
        rows = []
        for rho in rhos:
            row, ns = {"SNR (dB)": rho, "target val acc (%)": 100 * A}, []
            for m in TRAINED:
                r = runs(scheme, m, rho)
                if r.empty:
                    continue
                note(fig, f"{rho:g} dB", m, r)
                s = P.ttt(r, A, xcol)
                row[COLUMN[m]] = s["t"]
                row[f"{COLUMN[m]} [reached]"] = f"{s['hit']}/{s['n']}"
                ns.append(s["n"])
            row["seeds"] = _seeds_txt(ns)
            rows.append(row)
        return pd.DataFrame(rows)

    def budget_vs_snr(fig, scheme, xcol="uplink_s"):
        air = runs(scheme, "airsfl", 20.0)
        T = float(np.mean([g[xcol].iloc[-1] for g in P._seed_runs(air)]))   # AirSFL's full run at 20 dB
        rows = []
        for rho in rhos:
            row, ns, hollow = {"SNR (dB)": rho, "budget (uplink s)": T}, [], []
            for m in TRAINED:
                r = runs(scheme, m, rho)
                if r.empty:
                    continue
                note(fig, f"{rho:g} dB", m, r)
                b = P.budget_point(r, T, xcol)
                row[COLUMN[m]] = b["a"]
                if not b["evaluated"]:
                    hollow.append(COLUMN[m])
                ns.append(P._nseeds(r))
            row["no evaluated checkpoint in budget"] = "; ".join(hollow)
            row["seeds"] = _seeds_txt(ns)
            rows.append(row)
        return pd.DataFrame(rows)

    def time_to_target_vs_N(fig, scheme, xcol="uplink_s"):
        def frame(mn):
            a = pd.concat([f for f in (mn, nsweep) if not f.empty], ignore_index=True)
            return a[(a.rho_db == 20.0) & (a.tau == P.TAU0) & a.cut.isin([P.CUT0, 0]) & (a.partition == scheme)]
        sub, sub_t = frame(P._main20(main)), frame(main)   # bars: --main-seeds at N = 30; target: every seed
        Ns = sorted({int(n) for n in sub.N})
        tg = [P.rule_target(sub_t[sub_t.N == N], scheme) for N in Ns]
        A = min(t for t in tg if t is not None)            # common target reachable by every N
        rows = []
        for N in Ns:
            row, ns = {"N": N, "target val acc (%)": 100 * A}, []
            for m in TRAINED:
                dm = sub[(sub.N == N) & (sub.method == m)]
                if dm.empty:
                    continue
                note(fig, f"N={N}", m, dm)
                s = P.ttt(dm, A, xcol)
                row[COLUMN[m]] = s["t"]
                row[f"{COLUMN[m]} [reached]"] = f"{s['hit']}/{s['n']}"
                ns.append(s["n"])
            row["seeds"] = _seeds_txt(ns)
            rows.append(row)
        return pd.DataFrame(rows)

    def curves_20dB(fig, scheme):
        A = P.primary_target(main, scheme)
        curves, summary = [], []
        air_t = None
        for m in TRAINED:
            r = runs(scheme, m, 20.0)
            if r.empty:
                continue
            note(fig, "20 dB", m, r)
            c = P._curve(r)
            curves.append(pd.DataFrame({"method": COLUMN[m], "round": c["round"].astype(int),
                                        "uplink time (s)": c.uplink_s, "test acc (%)": 100 * c.test_acc}))
            s = P.ttt(r, A, "uplink_s")
            if m == "airsfl":
                air_t = s["t"]
            summary.append({"method": COLUMN[m], "seeds": s["n"], "final test acc (%)": P.final_acc(r)[0],
                            "UL s/round": float(r.ul_s_per_round.iloc[0]),
                            "UL time of the full run (s)": float(np.mean([g.uplink_s.iloc[-1] for g in P._seed_runs(r)])),
                            "target val acc (%)": 100 * A, "reached": f"{s['hit']}/{s['n']}",
                            "UL time to target (s)": s["t"]})
        summary = pd.DataFrame(summary)
        if air_t:
            summary["UL time to target / AirSFL"] = summary["UL time to target (s)"] / air_t
        return pd.concat(curves, ignore_index=True), summary

    for name, (scheme, kind) in FIGS.items():
        if kind == "curves_20dB":
            data, summary = curves_20dB(name, scheme)
            summary.to_csv(os.path.join(out, f"{name}_summary.csv"), index=False, float_format="%.6g")
        else:
            data = {"final_acc_vs_snr": final_acc_vs_snr,
                    "uplink_time_to_target_vs_snr": lambda f, sc: time_to_target_vs_snr(f, sc, xcol="uplink_s"),
                    "training_time_to_target_vs_snr": lambda f, sc: time_to_target_vs_snr(f, sc, "training_time_s"),
                    "budget_vs_snr": budget_vs_snr, "time_to_target_vs_N": time_to_target_vs_N}[kind](name, scheme)
        data.to_csv(os.path.join(out, f"{name}.csv"), index=False, float_format="%.6g")
        print(f"[data] {name}.csv")
    prov = pd.DataFrame(prov)
    prov.to_csv(os.path.join(out, "provenance.csv"), index=False)
    pay = prov[prov["digital payload"].str.startswith("FP")]["digital payload"].unique()
    print(f"[data] provenance.csv: digital payloads used {sorted(pay)}; seeds per point "
          f"{prov.groupby('seeds').size().to_dict()}")


# ---------------------------------------------------------------------------------------------------
# 2. PLOT (from the CSVs only)
# ---------------------------------------------------------------------------------------------------

def _style(m):
    return METHODS[m]


def _legend(ax, fig):
    if FORMAT["legend"] == "none":
        return
    h, lab = ax.get_legend_handles_labels()
    if not h:
        return
    if FORMAT["legend"] == "inside":
        ax.legend(h, lab, fontsize=FORMAT["legend_font"], loc="best")
        return
    fig.canvas.draw()                                  # just below the x-axis label, never on the data
    y0 = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted()).y0
    ax.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, y0 - 0.02), ncol=FORMAT["legend_cols"],
              fontsize=FORMAT["legend_font"], frameon=False)


def _finish(fig, ax, out, name, title):
    if FORMAT["title"] and title:
        ax.set_title(title, fontsize=FORMAT["font"])
    ax.grid(True, alpha=FORMAT["grid_alpha"])
    _legend(ax, fig)
    for ext in FORMAT["file_types"]:
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", dpi=FORMAT["dpi"])
    plt.close(fig)
    print(f"[fig] {name}")


def _not_reached(ax, x, color):
    ax.plot([x], [0.96], marker="x", color=color, ms=9, mew=2, transform=ax.get_xaxis_transform(), clip_on=False)


def _part(name):
    return "Non-IID (Dir-0.1)" if "noniid" in name else "IID"


def plot_final_acc_vs_snr(df, out, name):
    fig, ax = plt.subplots(figsize=FORMAT["figsize"])
    for m in TRAINED:
        if m in TWINS or COLUMN[m] not in df:
            continue                                   # a ZF variant = its OFDMA twin: drawn once
        st = _style(m)
        ax.plot(df["SNR (dB)"], df[COLUMN[m]], color=st["color"], marker=st["marker"], ls=st["ls"], lw=st["lw"],
                ms=FORMAT["marker_size"], label=JOINT.get(m, st["label"]))
    ax.set_xlabel(FORMAT["snr_label"])
    ax.set_ylabel("Final test accuracy (%)")
    _finish(fig, ax, out, name, f"{_part(name)}: final accuracy vs SNR")


def plot_time_to_target_vs_snr(df, out, name, axis="Training"):
    fig, ax = plt.subplots(figsize=FORMAT["figsize"])
    for m in TRAINED:
        c = COLUMN[m]
        if c not in df:
            continue
        st = _style(m)
        ok = df[df[c].notna()]
        ax.plot(ok["SNR (dB)"], ok[c], color=st["color"], marker=st["marker"], ls=st["ls"], lw=st["lw"],
                ms=FORMAT["marker_size"], label=st["label"])
        for _, r in df.iterrows():
            if not isinstance(r.get(f"{c} [reached]"), str):
                continue                               # no run of this method at this SNR
            hit, n = (int(v) for v in r[f"{c} [reached]"].split("/"))
            if hit == 0:
                _not_reached(ax, r["SNR (dB)"], st["color"])
            elif hit < n:
                ax.annotate(f"{hit}/{n}", (r["SNR (dB)"], r[c]), textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color=st["color"])
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 10)                           # headroom: "not reached" x sits above the data
    ax.set_xlabel(FORMAT["snr_label"])
    ax.set_ylabel(f"{axis} time to target (s)")
    _finish(fig, ax, out, name, f"{_part(name)}: {axis.lower()} time to {df['target val acc (%)'].iloc[0]:.0f}% "
                                f"val. acc. (x = not reached)")


def plot_budget_vs_snr(df, out, name):
    fig, ax = plt.subplots(figsize=FORMAT["figsize"])
    for m in TRAINED:
        c = COLUMN[m]
        if c not in df:
            continue
        st = _style(m)
        hollow = df["no evaluated checkpoint in budget"].fillna("").str.contains(c, regex=False)
        ok = df[~hollow]
        ax.plot(ok["SNR (dB)"], ok[c], color=st["color"], marker=st["marker"], ls=st["ls"], lw=st["lw"],
                ms=FORMAT["marker_size"], label=st["label"])
        if hollow.any():                               # initial model only (nothing evaluated inside the budget)
            ax.plot(df["SNR (dB)"][hollow], df[c][hollow], marker=st["marker"], mfc="white", mec=st["color"],
                    ms=FORMAT["marker_size"] + 2, ls="")
    ax.set_xlabel(FORMAT["snr_label"])
    ax.set_ylabel("Test accuracy within the budget (%)")
    _finish(fig, ax, out, name, f"{_part(name)}: accuracy after {df['budget (uplink s)'].iloc[0]:.0f} s of uplink "
                                f"time (AirSFL's run)")


def plot_time_to_target_vs_N(df, out, name):
    fig, ax = plt.subplots(figsize=FORMAT["figsize"])
    ms = [m for m in TRAINED if COLUMN[m] in df]
    x = np.arange(len(df))
    w = 0.84 / len(ms)
    for j, m in enumerate(ms):
        st, c = _style(m), COLUMN[m]
        xs = x + (j - (len(ms) - 1) / 2) * w
        ax.bar(xs, df[c], width=w, color=st["color"], edgecolor="k", lw=0.4, label=st["label"])
        for xi, v in zip(xs, df[c]):
            if np.isnan(v):
                ax.text(xi, 0.02, "×", ha="center", color=st["color"], fontsize=10, transform=ax.get_xaxis_transform())
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={int(n)}" for n in df["N"]])
    ax.set_ylabel("Uplink time to target (s)")
    _finish(fig, ax, out, name, f"{_part(name)}: uplink time to {df['target val acc (%)'].iloc[0]:.0f}% "
                                f"(× = not reached)")


def plot_curves(df, out, name):
    fig, ax = plt.subplots(figsize=FORMAT["figsize"])
    ends = df[df.method.isin([COLUMN["aircomp_fl"], COLUMN["airsfl"]])].groupby("method")["uplink time (s)"].max()
    xmax = 1.05 * float(ends.max()) if len(ends) else None    # AirComp-FL's full run (the fastest baseline)
    for m in METHODS:
        c = df[df.method == COLUMN[m]].sort_values("round")
        if c.empty:
            continue
        st = _style(m)
        ax.plot(c["uplink time (s)"], c["test acc (%)"], color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"],
                markevery=max(1, len(c) // FORMAT["curve_markers"]), ms=FORMAT["marker_size"] - 1, label=st["label"])
        if FORMAT["extend_curves"] and xmax and float(c["uplink time (s)"].iloc[-1]) < xmax:
            ax.plot([float(c["uplink time (s)"].iloc[-1]), xmax], [float(c["test acc (%)"].iloc[-1])] * 2,
                    color=st["color"], ls=":", lw=1.0, alpha=0.7)
    if xmax:
        ax.set_xlim(0, xmax)
    ax.set_xlabel("Accumulated uplink communication time (s)")
    ax.set_ylabel("Test accuracy (%)")
    if FORMAT["inset_methods"] and name in FORMAT["inset_bounds"]:
        _inset_full_runs(ax, df, FORMAT["inset_bounds"][name])
    _finish(fig, ax, out, name, f"{_part(name)}, $\\rho$ = 20 dB")


def _inset_full_runs(ax, df, bounds):
    """Small box inside the axes with the whole run of the slow baselines (same colors and line styles
    as the main plot; uplink time in thousands of seconds: 0, 50k, 100k, 150k)."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    ins = ax.inset_axes(bounds)
    for m in FORMAT["inset_methods"]:
        c = df[df.method == COLUMN[m]].sort_values("round")
        if c.empty:
            continue
        st = _style(m)
        ins.plot(c["uplink time (s)"], c["test acc (%)"], color=st["color"], ls=st["ls"], lw=max(1.0, 0.7 * st["lw"]))
    ins.set_xlim(0, None)
    ins.xaxis.set_major_locator(MaxNLocator(4))
    ins.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:g}k"))
    ins.yaxis.set_major_locator(MaxNLocator(4))
    ins.tick_params(labelsize=FORMAT["inset_font"])
    ins.grid(True, alpha=FORMAT["grid_alpha"])
    if FORMAT["inset_title"]:
        ins.set_title(FORMAT["inset_title"], fontsize=FORMAT["inset_font"], pad=2)


def plot_all(out):
    plt.rcParams.update({"font.size": FORMAT["font"]})
    draw = {"final_acc_vs_snr": plot_final_acc_vs_snr,
            "uplink_time_to_target_vs_snr": lambda d, o, n: plot_time_to_target_vs_snr(d, o, n, axis="Uplink"),
            "training_time_to_target_vs_snr": lambda d, o, n: plot_time_to_target_vs_snr(d, o, n, axis="Training"),
            "budget_vs_snr": plot_budget_vs_snr, "time_to_target_vs_N": plot_time_to_target_vs_N,
            "curves_20dB": plot_curves}
    for name, (_, kind) in FIGS.items():
        path = os.path.join(out, f"{name}.csv")
        if not os.path.exists(path):
            print(f"[fig] {name}: {path} missing (run without --plot-only first)")
            continue
        draw[kind](pd.read_csv(path), out, name)


def main():
    p = argparse.ArgumentParser(description="The 7 AirSFL paper figures (seed means) + their CSVs -> <results>/finalz/")
    p.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_final_s1"))
    p.add_argument("--main-seeds", nargs="+", type=int, default=None,
                   help="seeds of the 20 dB (main) points, e.g. 11 while seeds 22/33 of main are missing")
    p.add_argument("--plot-only", action="store_true", help="redraw from finalz/*.csv (after a FORMAT change)")
    a = p.parse_args()
    out = os.path.join(os.path.abspath(a.results), "finalz")
    if not a.plot_only:
        build_data(a.results, a.main_seeds, out)
    plot_all(out)


if __name__ == "__main__":
    main()
