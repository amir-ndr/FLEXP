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

Low-precision digital baselines (--payload-results <folder of the INT8 / INT4 runs of run_airsfl.py
--digital-q 8 / 4>): the same figures with the INT-q versions of the ZF baselines added (VARIANTS
below), the payload figure fig8 (AirSFL vs Hybrid-Z / SFL-Z at FP16 / INT8 / INT4) and the payload
table (table_payload.csv with every method x payload x SNR / N, table_payload.tex / .md compact) go to
<results>/finalz_payload/; finalz/ is not touched. Targets are those of finalz (from the six FP16
methods), so every FP16 number is the same in both folders.

Usage (from the repo root):
  python examples/AirSFL/paper_figures.py --results examples/AirSFL/results_final_s1 --main-seeds 11
  python examples/AirSFL/paper_figures.py --results examples/AirSFL/results_final_s1 --plot-only
  python examples/AirSFL/paper_figures.py --results examples/AirSFL/results_final_s1 \\
      --payload-results examples/AirSFL/results_qbits            (+ --plot-only to redraw)
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

# DATA setting (needs a rebuild, not --plot-only): target of fig3 (uplink time to target vs SNR)
#   "common": the highest integer accuracy that EVERY drawn method reaches in EVERY seed at EVERY SNR, so
#             every point has a value (no "not reached"); "rule": 95% of digital SFL-V1's final accuracy
#             (the target of the other figures); or a number, e.g. 65 (= 65% validation accuracy)
FIG3_TARGET = "common"

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
    # figures drawn without title and legend: a shared legend figure (legend.png / .pdf) goes with them
    shared_legend_figs=("fig2_noniid_accuracy_vs_uplink_time_20dB", "fig6_iid_accuracy_vs_uplink_time_20dB"),
    # figures with their own legend INSIDE the axes: loc, columns, anchor (axes fractions) and extra headroom on
    # the y axis (factor on the log range top / on the linear span) so the legend covers no curve
    inside_legend={
        "fig1_noniid_final_accuracy_vs_snr": dict(loc="lower right"),
        "fig3_noniid_uplink_time_to_target_vs_snr": dict(loc="upper center", ncol=3, headroom=8.0),
        "fig4_noniid_accuracy_within_uplink_budget": dict(loc="lower right", ncol=2, bbox=(0.99, 0.16)),
        "fig4_noniid_accuracy_within_uplink_budget_2000": dict(loc="lower right", ncol=2, bbox=(0.99, 0.16)),
        "fig5_noniid_uplink_time_to_target_vs_N": dict(loc="upper center", ncol=3, headroom=6.0),
        "fig7_iid_uplink_time_to_target_vs_N": dict(loc="upper center", ncol=3, headroom=6.0),
    },
    inside_legend_frame=False,   # False = transparent legend (no box); True = white box
    inside_legend_font=10.5,     # font size of the legends inside the axes
    # figures drawn at their PRINT size (one IEEE column, include with width=\columnwidth): same physical size and
    # font as fig_payload_precision, so the text matches 1:1 in the paper. figsize / font / ylabel / legend
    # (legend replaces the figure's inside_legend spec) override the defaults above for that figure only
    # Same canvas (figsize above) for every figure except fig5; only the text, lines and markers are enlarged so that
    # they stay readable after LaTeX shrinks the figure to its \includegraphics width (IEEEtran: \columnwidth 3.5 in,
    # \textwidth 7.16 in). font: axis labels / ticks; legend: inside-legend spec; lw: line-width factor; ms: marker
    # size; xlabel / ylabel: shorter axis labels where the long ones do not fit at that font; xk: ticks as 5k, 10k
    compact={
        # \includegraphics[width=\columnwidth]: drawn at 3.45 x 2.45 in, like fig_payload_precision
        "fig5_noniid_uplink_time_to_target_vs_N": dict(
            figsize=(3.45, 2.45), font=9, ylabel="Uplink time to target (s)", ybottom=100.0,   # log bars start at 100 s
            legend=dict(loc="upper center", ncol=3, headroom=12.0, font=8.5, cs=0.8, hl=1.2)),
        # width=0.48\columnwidth (1.68 in; the 6.4-in canvas prints at ~0.29x): 22 pt -> ~6.5 pt in print
        "fig2_noniid_accuracy_vs_uplink_time_20dB": dict(
            font=22, lw=1.6, ms=9, xk=True, xlabel="Accumulated uplink time (s)", inset_font=16),
        "fig6_iid_accuracy_vs_uplink_time_20dB": dict(
            font=22, lw=1.6, ms=9, xk=True, xlabel="Accumulated uplink time (s)", inset_font=16),
        # width=0.32\textwidth (2.29 in; prints at ~0.41x): 19 pt -> ~7.5-8 pt in print
        "fig1_noniid_final_accuracy_vs_snr": dict(
            font=19, lw=1.4, ms=9, legend=dict(loc="lower right", font=16)),
        "fig3_noniid_uplink_time_to_target_vs_snr": dict(
            font=19, lw=1.4, ms=9, ylabel="Uplink time to target (s)",
            legend=dict(loc="upper center", ncol=3, headroom=20.0, font=15, cs=0.8, hl=1.6)),
        "fig4_noniid_accuracy_within_uplink_budget_2000": dict(
            font=19, lw=1.4, ms=9, ylabel="Test accuracy in budget (%)",
            legend=dict(loc="upper center", ncol=3, headroom=1.42, font=15, cs=0.8, hl=1.6, max_tick=80)),
    },
    no_title_figs=("fig1_noniid_final_accuracy_vs_snr", "fig2_noniid_accuracy_vs_uplink_time_20dB",
                   "fig3_noniid_uplink_time_to_target_vs_snr", "fig4_noniid_accuracy_within_uplink_budget",
                   "fig4_noniid_accuracy_within_uplink_budget_2000", "fig5_noniid_uplink_time_to_target_vs_N", "fig6_iid_accuracy_vs_uplink_time_20dB",
                   "fig7_iid_uplink_time_to_target_vs_N", "fig8_noniid_payload_uplink_time_to_target_vs_snr"),
    # fig1: a ZF variant has exactly the accuracy of its OFDMA twin. "merge" draws each pair once under a joint
    # label (SFL-O / SFL-Z); "overlay" draws both (Z dash-dot with its own marker on top of O)
    fig1_twins="merge",
    legend_figure=dict(cols=6, font=9, width=7.0,      # one row across a two-column figure* (inches)
                       order=("airsfl", "digital_sflv1", "digital_sflv1_zf", "sun_fdma_aircomp",
                              "hybrid_zf_aircomp", "aircomp_fl"),
                       labels={"airsfl": "AirSFL"}),           # label overrides for the legend figure only
)

# drawing order; label, color, marker, line style, line width (per method). Paper names: SFL-O / SFL-Z =
# digital SFL-V1 over OFDMA / multi-user ZF; Hybrid-O / Hybrid-Z = digital smashed data over OFDMA / ZF with
# AirComp aggregation; AirComp-FL = full model trained locally, AirComp aggregation
METHODS = {
    "airsfl":            dict(label="AirSFL (proposed)", color="#d62728", marker="o", ls="-",  lw=2.6),
    "aircomp_fl":        dict(label="AirComp-FL",        color="#2ca02c", marker="D", ls="-",  lw=1.8),
    "hybrid_zf_aircomp": dict(label="Hybrid-Z",          color="#8c564b", marker="X", ls="-.", lw=1.8),
    "sun_fdma_aircomp":  dict(label="Hybrid-O",          color="#9467bd", marker="^", ls="-",  lw=1.8),
    "digital_sflv1_zf":  dict(label="SFL-Z",             color="#17becf", marker="P", ls="-.", lw=1.8),
    "digital_sflv1":     dict(label="SFL-O",             color="#1f77b4", marker="s", ls="-",  lw=1.8),
}
# fig1 draws each ZF variant once with its OFDMA twin (identical learning), under this label
JOINT = {"sun_fdma_aircomp": "Hybrid-O / Hybrid-Z", "digital_sflv1": "SFL-O / SFL-Z"}
TWINS = {"hybrid_zf_aircomp": "sun_fdma_aircomp", "digital_sflv1_zf": "digital_sflv1"}

# CSV column name of each method (fixed: the plot step finds the data by these names)
COLUMN = {"airsfl": "AirSFL", "aircomp_fl": "AirComp-FL", "hybrid_zf_aircomp": "Hybrid-Z",
          "sun_fdma_aircomp": "Hybrid-O", "digital_sflv1_zf": "SFL-Z", "digital_sflv1": "SFL-O"}
TRAINED = list(METHODS)

# ---------------------------------------------------------------------------------------------------
# LOW-PRECISION DIGITAL BASELINES (only with --payload-results; output -> <results>/finalz_payload/)
# ---------------------------------------------------------------------------------------------------
# key "<method>@int<q>" = that method trained with a q-bit INT payload (run_airsfl.py --digital-q q); hollow
# markers, hatched bars. Add e.g. "sun_fdma_aircomp@int8" (style it) to draw an OFDMA variant too.
VARIANTS = {
    "hybrid_zf_aircomp@int8": dict(label="Hybrid-Z (INT8)", color="#ff7f0e", marker="X", ls="--", lw=1.6,
                                   mfc="white", hatch="///"),
    "hybrid_zf_aircomp@int4": dict(label="Hybrid-Z (INT4)", color="#e377c2", marker="X", ls=":", lw=2.0,
                                   mfc="white", hatch="..."),
    "digital_sflv1_zf@int8":  dict(label="SFL-Z (INT8)", color="#bcbd22", marker="P", ls="--", lw=1.6,
                                   mfc="white", hatch="///"),
    "digital_sflv1_zf@int4":  dict(label="SFL-Z (INT4)", color="#7f7f7f", marker="P", ls=":", lw=2.0,
                                   mfc="white", hatch="..."),
}
PAYLOAD = dict(
    bits=(8, 4),                                        # INT-q payloads read from --payload-results
    # payload figure (fig8): AirSFL against the ZF baselines at every payload, uplink time to fig3's target vs SNR
    fig8_methods=("airsfl", "hybrid_zf_aircomp", "hybrid_zf_aircomp@int8", "hybrid_zf_aircomp@int4",
                  "digital_sflv1_zf", "digital_sflv1_zf@int8", "digital_sflv1_zf@int4"),
    # table_payload.tex / .md (compact; table_payload.csv has everything): rows (method, q), SNR columns
    table_rows=(("airsfl", None), ("hybrid_zf_aircomp", 16), ("hybrid_zf_aircomp", 8), ("hybrid_zf_aircomp", 4),
                ("digital_sflv1_zf", 16), ("digital_sflv1_zf", 8), ("digital_sflv1_zf", 4)),
    table_snrs=(20.0, 10.0, 0.0, -20.0),
    # finalz_payload labels: the FP16 baselines say "(FP16)"; fig1 draws each O/Z pair once ("Hybrid-O/Z ...")
    labels={"airsfl": "AirSFL", "hybrid_zf_aircomp": "Hybrid-Z (FP16)", "sun_fdma_aircomp": "Hybrid-O (FP16)",
            "digital_sflv1_zf": "SFL-Z (FP16)", "digital_sflv1": "SFL-O (FP16)"},
    joint={"sun_fdma_aircomp": "Hybrid-O/Z (FP16)", "digital_sflv1": "SFL-O/Z (FP16)",
           "hybrid_zf_aircomp@int8": "Hybrid-O/Z (INT8)", "hybrid_zf_aircomp@int4": "Hybrid-O/Z (INT4)",
           "digital_sflv1_zf@int8": "SFL-O/Z (INT8)", "digital_sflv1_zf@int4": "SFL-O/Z (INT4)"},
    # legends of finalz_payload (more entries than finalz): overrides of FORMAT["inside_legend"] / legend_figure.
    # "rows": the legend as a grid, one method family per row, one payload per column (None = empty cell)
    # (footroom: extra room below the data, as a fraction of the y span; cs / hl: column spacing / handle length)
    inside_legend={
        "fig1_noniid_final_accuracy_vs_snr": dict(loc="lower right", font=8.5, footroom=0.45, rows=(
            ("airsfl", "aircomp_fl", None),
            ("digital_sflv1", "digital_sflv1_zf@int8", "digital_sflv1_zf@int4"),
            ("sun_fdma_aircomp", "hybrid_zf_aircomp@int8", "hybrid_zf_aircomp@int4"))),
        "fig3_noniid_uplink_time_to_target_vs_snr": dict(loc="upper center", headroom=60.0, font=7.5, cs=0.8, hl=1.8,
                                                         rows=(
            ("airsfl", "aircomp_fl", None, None),
            ("digital_sflv1", "digital_sflv1_zf", "digital_sflv1_zf@int8", "digital_sflv1_zf@int4"),
            ("sun_fdma_aircomp", "hybrid_zf_aircomp", "hybrid_zf_aircomp@int8", "hybrid_zf_aircomp@int4"))),
        "fig4_noniid_accuracy_within_uplink_budget": dict(loc="lower right", ncol=2, bbox=(0.99, 0.10), font=8.5),
        "fig4_noniid_accuracy_within_uplink_budget_2000": dict(loc="lower right", ncol=2, bbox=(0.99, 0.10), font=8.5),
        "fig5_noniid_uplink_time_to_target_vs_N": dict(loc="upper center", headroom=40.0, font=7.5, cs=0.8, hl=1.8,
                                                       rows=(
            ("airsfl", "aircomp_fl", None, None),
            ("digital_sflv1", "digital_sflv1_zf", "digital_sflv1_zf@int8", "digital_sflv1_zf@int4"),
            ("sun_fdma_aircomp", "hybrid_zf_aircomp", "hybrid_zf_aircomp@int8", "hybrid_zf_aircomp@int4"))),
        "fig7_iid_uplink_time_to_target_vs_N": dict(loc="upper center", headroom=40.0, font=7.5, cs=0.8, hl=1.8,
                                                    rows=(
            ("airsfl", "aircomp_fl", None, None),
            ("digital_sflv1", "digital_sflv1_zf", "digital_sflv1_zf@int8", "digital_sflv1_zf@int4"),
            ("sun_fdma_aircomp", "hybrid_zf_aircomp", "hybrid_zf_aircomp@int8", "hybrid_zf_aircomp@int4"))),
        "fig8_noniid_payload_uplink_time_to_target_vs_snr": dict(loc="upper center", headroom=25.0, font=9.5, rows=(
            ("airsfl", None, None),
            ("digital_sflv1_zf", "digital_sflv1_zf@int8", "digital_sflv1_zf@int4"),
            ("hybrid_zf_aircomp", "hybrid_zf_aircomp@int8", "hybrid_zf_aircomp@int4"))),
    },
    legend_figure=dict(cols=5, font=9, width=7.0,
                       order=("airsfl", "digital_sflv1", "digital_sflv1_zf", "digital_sflv1_zf@int8",
                              "digital_sflv1_zf@int4", "aircomp_fl", "sun_fdma_aircomp", "hybrid_zf_aircomp",
                              "hybrid_zf_aircomp@int8", "hybrid_zf_aircomp@int4"),
                       rows=(("airsfl", "digital_sflv1", "digital_sflv1_zf", "digital_sflv1_zf@int8",
                              "digital_sflv1_zf@int4"),
                             ("aircomp_fl", "sun_fdma_aircomp", "hybrid_zf_aircomp", "hybrid_zf_aircomp@int8",
                              "hybrid_zf_aircomp@int4"))),
)
PAYLOAD_FIGS = {   # extra figures of finalz_payload (same specs as FIGS)
    "fig8_noniid_payload_uplink_time_to_target_vs_snr": ("dirichlet", "uplink_time_to_target_vs_snr",
                                                         {"methods": PAYLOAD["fig8_methods"]}),
}
# compact one-column payload figure (finalz_payload/fig_payload_precision): uplink time to fig3's target of each
# ZF baseline / AirSFL's, vs SNR -- color = payload, line style + marker = scheme; AirSFL's own points sit at 1
PRECISION = dict(
    figsize=(3.45, 2.45), font=9, legend_font=8.5, marker_size=4.5, lw=1.4,
    payloads={"FP16": "#0f7c8c", "INT8": "#c06a00", "INT4": "#8350a8"},          # payload -> color
    schemes={"SFL-Z": dict(ls="-", marker="o"), "Hybrid-Z": dict(ls="--", marker="s")},
    airsfl=dict(label="AirSFL", color="#d62728", marker="o", ms=5.5, ls="-", lw=1.8),
    ylabel="Uplink time (× AirSFL)",  # = uplink time to target / AirSFL's at the same SNR
    shade="Digital faster",          # label of the region below 1 ("" = no shading)
    airsfl_seconds=False,            # True: AirSFL's own time to target (s) written next to each of its points
)
DRAWN = TRAINED + list(VARIANTS)                         # drawing order (variants exist only in finalz_payload)
COLUMN.update({m: st["label"] for m, st in VARIANTS.items()})


def _variant(m):
    """'<method>@int<q>' -> (method, q); a plain method -> (method, None)."""
    if "@int" not in m:
        return m, None
    base, q = m.split("@int")
    return base, int(q)

FIGS = {   # file name -> (partition, kind[, options])
    "fig1_noniid_final_accuracy_vs_snr": ("dirichlet", "final_acc_vs_snr"),
    "fig2_noniid_accuracy_vs_uplink_time_20dB": ("dirichlet", "curves_20dB"),
    "fig3_noniid_uplink_time_to_target_vs_snr": ("dirichlet", "uplink_time_to_target_vs_snr"),
    "fig4_noniid_accuracy_within_uplink_budget": ("dirichlet", "budget_vs_snr"),    # budget = AirSFL's full run
    "fig4_noniid_accuracy_within_uplink_budget_2000": ("dirichlet", "budget_vs_snr", {"budget_s": 2000.0}),
    # fig5 target "common": the highest accuracy every method reaches in every seed at every N (as fig3)
    "fig5_noniid_uplink_time_to_target_vs_N": ("dirichlet", "time_to_target_vs_N", {"target": "common"}),
    "fig6_iid_accuracy_vs_uplink_time_20dB": ("iid", "curves_20dB"),
    "fig7_iid_uplink_time_to_target_vs_N": ("iid", "time_to_target_vs_N"),
}


# ---------------------------------------------------------------------------------------------------
# 1. DATA (from the run CSVs, through plots.py)
# ---------------------------------------------------------------------------------------------------

def _seeds_txt(ns):
    ns = [n for n in ns if n]
    return "" if not ns else (str(min(ns)) if min(ns) == max(ns) else f"{min(ns)}-{max(ns)}")


def _only_seeds(sets, seeds):
    """Every run set restricted to these seeds (None = all): a paired comparison when one set has fewer seeds."""
    if not seeds:
        return sets
    return {e: df if df.empty else df[df.seed.astype(int).isin(seeds)].reset_index(drop=True) for e, df in sets.items()}


def build_data(results, main_seeds, out, fig3_target=FIG3_TARGET, payload_results=None, payload_bits=PAYLOAD["bits"],
               seeds=None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import plots as P
    from flsim.airsfl.timing import breakeven_bits, payload_name
    P.MAIN_SEEDS["seeds"] = set(main_seeds) if main_seeds else None
    d, _ = P.load_results(results)
    x = _only_seeds(P._select_payload(d, 16), seeds)   # FP16 digital payload
    P.ENV0["q_bits"] = 16
    P.TARGET_SRC["df"] = x["main"]                     # 95% target: every seed of FP16 digital SFL-V1
    main, snr, nsweep = x["main"], x["snr"], x["nsweep"]
    P._report_unpaired({**x, "main": P._main20(main)})
    xq = {}                                            # INT-q sets (finalz_payload): {q: {experiment: runs}}
    if payload_results:
        dq, _ = P.load_results(payload_results)
        for q in payload_bits:
            xq[q] = _only_seeds(P._select_payload(dq, q), seeds)
            n = {e: int(df[df.method.isin(P.FP16_PARENTS)].run_id.nunique()) for e, df in xq[q].items()
                 if not df.empty and e in ("main", "snr", "nsweep")}
            print(f"[data] {payload_name(q)} runs in {payload_results}: {n or 'none'}")
            if n:
                P._report_unpaired({**xq[q], "main": P._main20(xq[q]["main"])})
    drawn = DRAWN if payload_results else TRAINED
    rhos = sorted(set([20.0] + [float(r) for r in snr.rho_db.unique()]))
    os.makedirs(out, exist_ok=True)
    prov = []
    targets = {}                                       # figure -> its target (payload table)

    def note(fig, point, m, r):
        """Record which payload, seeds and run files a plotted point of method m comes from."""
        base, qv = _variant(m)
        q = sorted(set(P._q_col(r)))
        files = sorted(set(r.source_file))
        if qv is not None:                             # INT-q runs: path relative to the repo's AirSFL folder
            files = [os.path.join(os.path.basename(os.path.normpath(payload_results)), f) for f in files]
        prov.append({"figure": fig, "point": point, "method": COLUMN[m],
                     "digital payload": "/".join(payload_name(v) for v in q) if base in P.PAYLOAD_METHODS else
                     ("none (analog; labels only)" if base == "airsfl" else "none (analog)"),
                     "seeds": ",".join(str(v) for v in sorted(set(r.seed.astype(int)))),
                     "run files": "; ".join(files)})

    def runs_q(scheme, m, q, rho):
        """Runs of method m at SNR rho with a q-bit digital payload (None / 16: the FP16 set)."""
        if q in (None, 16):
            return P._snr_runs(main, snr, scheme, m, rho)
        s = xq.get(q, {})
        mn, sn = s.get("main", pd.DataFrame()), s.get("snr", pd.DataFrame())
        if mn.empty:                                   # no 20 dB runs (yet): the hybrid's SNR-sweep runs only
            if rho == 20.0 or m in P.DIGITAL or sn.empty:
                return pd.DataFrame()
            return sn[(sn.partition == scheme) & (sn.method == m) & (sn.rho_db == rho)]
        return P._snr_runs(mn, sn, scheme, m, rho)

    def runs(scheme, m, rho):
        return runs_q(scheme, *_variant(m), rho)

    def frame(mn, ns, scheme):
        """20 dB runs of the N sweep (N = 30 from main) of one partition."""
        a = pd.concat([f for f in (mn, ns) if not f.empty], ignore_index=True)
        if a.empty:
            return a
        return a[(a.rho_db == 20.0) & (a.tau == P.TAU0) & a.cut.isin([P.CUT0, 0]) & (a.partition == scheme)]

    def runs_N(scheme, m, N, every_seed=False):
        """Runs of method m (or '<method>@int<q>') at N clients, 20 dB (--main-seeds at N = 30 unless every_seed)."""
        base, q = _variant(m)
        s = x if q is None else xq.get(q, {})
        mn = s.get("main", pd.DataFrame())
        f = frame(mn if every_seed else P._main20(mn), s.get("nsweep", pd.DataFrame()), scheme)
        return f if f.empty else f[(f.N == N) & (f.method == base)]

    def final_acc_vs_snr(fig, scheme):
        rows = []
        for rho in rhos:
            row, ns = {"SNR (dB)": rho}, []
            for m in drawn:
                r = runs(scheme, m, rho)
                if not r.empty:
                    row[COLUMN[m]] = P.final_acc(r)[0]
                    ns.append(P._nseeds(r))
                    note(fig, f"{rho:g} dB", m, r)
            row["seeds"] = _seeds_txt(ns)
            rows.append(row)
        return pd.DataFrame(rows)

    def common_target(scheme):
        """Highest integer accuracy every drawn method reaches (best validation checkpoint) in every seed
        at every SNR."""
        best = [float(g[g["round"] >= 1].val_acc.max()) for rho in rhos for m in TRAINED
                for g in P._seed_runs(runs(scheme, m, rho))]
        return math.floor(100 * min(best)) / 100

    def target_of(scheme, setting):
        if setting == "rule":
            return P.primary_target(main, scheme)
        if setting == "common":
            return common_target(scheme)
        v = float(setting)
        return v / 100 if v > 1 else v

    def time_to_target_vs_snr(fig, scheme, xcol="training_time_s", target="rule"):
        A = target_of(scheme, target)
        rows = []
        for rho in rhos:
            row, ns = {"SNR (dB)": rho, "target val acc (%)": 100 * A}, []
            for m in drawn:
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

    def budget_vs_snr(fig, scheme, xcol="uplink_s", budget_s=None):
        """Test accuracy of the last evaluated checkpoint within an uplink budget T (per seed, then mean):
        T = budget_s, or AirSFL's full-run uplink time at 20 dB when not given."""
        if budget_s is None:
            air = runs(scheme, "airsfl", 20.0)
            T, source = float(np.mean([g[xcol].iloc[-1] for g in P._seed_runs(air)])), "AirSFL's full run"
        else:
            T, source = float(budget_s), "fixed"
        rows = []
        for rho in rhos:
            row, ns, hollow = {"SNR (dB)": rho, "budget (uplink s)": T, "budget": source}, [], []
            for m in drawn:
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

    def time_to_target_vs_N(fig, scheme, xcol="uplink_s", target="rule"):
        # bars: --main-seeds at N = 30; target: every seed (and the six FP16 methods only, as in finalz)
        sub, sub_t = frame(P._main20(main), nsweep, scheme), frame(main, nsweep, scheme)
        Ns = sorted({int(n) for n in sub.N})
        if target == "common":                              # highest accuracy every drawn run reaches
            A = math.floor(100 * min(float(g[g["round"] >= 1].val_acc.max()) for N in Ns for m in TRAINED
                                     for g in P._seed_runs(sub[(sub.N == N) & (sub.method == m)]))) / 100
        elif target == "rule":                              # lowest per-N 95% target, reachable by every N
            tg = [P.rule_target(sub_t[sub_t.N == N], scheme) for N in Ns]
            A = min(t for t in tg if t is not None)
        else:
            A = float(target) / 100 if float(target) > 1 else float(target)
        rows = []
        for N in Ns:
            row, ns = {"N": N, "target val acc (%)": 100 * A}, []
            for m in drawn:
                dm = runs_N(scheme, m, N)
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
        for m in drawn:
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

    def payload_table():
        """table_payload.csv: AirSFL, AirComp-FL and the ZF / OFDMA baselines at FP16 and every INT-q payload, at
        the targets of the figures -- non-IID SNR sweep (fig3), 20 dB non-IID / IID (fig2 / fig6), N sweep
        (fig5 / fig7): seeds, final test accuracy, uplink s/round (and / AirSFL's), uplink time to target (mean
        over the seeds that reached it; / AirSFL's), seeds reached, rounds to target, activation and aggregate
        error-to-signal ratios (AirSFL: ZF / AirComp error; digital: payload rounding) and, per SNR, the
        break-even payload q* = c_ZF / 2 (the digital ZF payload with AirSFL's airtime per value)."""
        meths = [("airsfl", None), ("aircomp_fl", None)] + [
            (m, q) for m in ("hybrid_zf_aircomp", "sun_fdma_aircomp", "digital_sflv1_zf", "digital_sflv1")
            for q in [16] + list(xq)]
        rows = []

        def add(section, scheme, point, A, get):
            first = len(rows)
            for m, q in meths:
                r = get(m, q)
                if r.empty:
                    continue
                s = P.ttt(r, A, "uplink_s")
                ev = r[r["round"] >= 1]
                db = lambda c: float(ev[c].mean()) if c in ev and ev[c].notna().any() else np.nan
                rows.append({"section": section, "partition": scheme, **point, "method": COLUMN[m],
                             "payload": "analog" if q is None else payload_name(q), "seeds": P._nseeds(r),
                             "final test acc (%)": P.final_acc(r)[0], "UL s/round": float(r.ul_s_per_round.iloc[0]),
                             "target val acc (%)": 100 * A, "UL time to target (s)": s["t"],
                             "reached": f"{s['hit']}/{s['n']}", "rounds to target": s["k"],
                             "activation NSR (dB)": db("act_nsr_db"), "aggregation NSR (dB)": db("agg_nsr_db")})
            air = [w for w in rows[first:] if w["method"] == COLUMN["airsfl"]]
            for w in rows[first:]:
                w["UL s/round / AirSFL"] = w["UL s/round"] / air[0]["UL s/round"] if air else np.nan
                w["UL time to target / AirSFL"] = w["UL time to target (s)"] / air[0]["UL time to target (s)"] \
                    if air else np.nan

        A3 = targets["fig3_noniid_uplink_time_to_target_vs_snr"]
        for rho in rhos:
            radio = P._radio(rho_db=rho)
            add("snr", "dirichlet", {"SNR (dB)": rho, "N": 30, "q* = c_ZF/2 (bits)": breakeven_bits(radio, P._rates(radio))},
                A3, lambda m, q: runs_q("dirichlet", m, q, rho))
        for scheme, fig in (("dirichlet", "fig2_noniid_accuracy_vs_uplink_time_20dB"),
                            ("iid", "fig6_iid_accuracy_vs_uplink_time_20dB")):
            add("20dB", scheme, {"SNR (dB)": 20.0, "N": 30}, targets[fig], lambda m, q: runs_q(scheme, m, q, 20.0))
        for scheme, fig in (("dirichlet", "fig5_noniid_uplink_time_to_target_vs_N"),
                            ("iid", "fig7_iid_uplink_time_to_target_vs_N")):
            for N in sorted({int(n) for n in frame(main, nsweep, scheme).N}):
                add("N", scheme, {"SNR (dB)": 20.0, "N": N}, targets[fig],
                    lambda m, q: runs_N(scheme, m if q in (None, 16) else f"{m}@int{q}", N))
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(out, "table_payload.csv"), index=False, float_format="%.6g")
        _payload_tex(t, out)
        _precision_csv(t, out)
        print(f"[data] table_payload.csv ({len(t)} rows), table_payload.tex / .md")

    figs = {**FIGS, **(PAYLOAD_FIGS if payload_results else {})}
    for name, spec in figs.items():
        scheme, kind = spec[:2]
        opts = spec[2] if len(spec) > 2 else {}
        if kind == "curves_20dB":
            data, summary = curves_20dB(name, scheme)
            summary.to_csv(os.path.join(out, f"{name}_summary.csv"), index=False, float_format="%.6g")
            targets[name] = float(summary["target val acc (%)"].iloc[0]) / 100
        else:
            data = {"final_acc_vs_snr": final_acc_vs_snr,
                    "uplink_time_to_target_vs_snr": lambda f, sc: time_to_target_vs_snr(f, sc, "uplink_s", fig3_target),
                    "training_time_to_target_vs_snr": lambda f, sc: time_to_target_vs_snr(f, sc, "training_time_s"),
                    "budget_vs_snr": lambda f, sc: budget_vs_snr(f, sc, budget_s=opts.get("budget_s")),
                    "time_to_target_vs_N": lambda f, sc: time_to_target_vs_N(f, sc, target=opts.get("target", "rule"))
                    }[kind](name, scheme)
            if "target val acc (%)" in data:
                targets[name] = float(data["target val acc (%)"].iloc[0]) / 100
        data.to_csv(os.path.join(out, f"{name}.csv"), index=False, float_format="%.6g")
        tg = f" (target {data['target val acc (%)'].iloc[0]:g}%)" if "target val acc (%)" in data else ""
        print(f"[data] {name}.csv{tg}")
    if payload_results:
        payload_table()
    prov = pd.DataFrame(prov)
    prov.to_csv(os.path.join(out, "provenance.csv"), index=False)
    pay = prov[prov["digital payload"].str.match(r"(FP|INT)\d")]["digital payload"].unique()
    print(f"[data] provenance.csv: digital payloads used {sorted(pay)}; seeds per point "
          f"{prov.groupby('seeds').size().to_dict()}")


def _precision_csv(t, out):
    """fig_payload_precision.csv: per SNR, AirSFL's uplink time to fig3's target (s) and, per ZF scheme and payload,
    the ratio of its mean time to AirSFL's (+ seeds reached)."""
    snr = t[t.section == "snr"]
    rows = []
    for rho, g in snr.groupby("SNR (dB)"):
        air = g[g.method == COLUMN["airsfl"]]
        row = {"SNR (dB)": rho, "target val acc (%)": float(g["target val acc (%)"].iloc[0]),
               "AirSFL (s)": float(air["UL time to target (s)"].iloc[0]) if not air.empty else np.nan,
               "AirSFL [reached]": air["reached"].iloc[0] if not air.empty else ""}
        for sc in PRECISION["schemes"]:
            for pl in PRECISION["payloads"]:
                w = g[(g.method == sc) & (g.payload == pl)]
                if not w.empty:
                    row[f"{sc} {pl}"] = float(w["UL time to target / AirSFL"].iloc[0])
                    row[f"{sc} {pl} [reached]"] = w["reached"].iloc[0]
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out, "fig_payload_precision.csv"), index=False, float_format="%.6g")


def plot_payload_precision(df, out, name="fig_payload_precision"):
    """Compact one-column figure: time to target of each ZF baseline at each payload / AirSFL's vs SNR; AirSFL's
    own (measured) points at 1, the region below 1 (digital faster) shaded."""
    from matplotlib.ticker import FixedLocator, NullLocator, StrMethodFormatter
    fp = PRECISION
    with plt.rc_context({"font.size": fp["font"]}):
        fig, ax = plt.subplots(figsize=fp["figsize"])
        x = df["SNR (dB)"]
        entries = {}
        for sc, ss in fp["schemes"].items():
            for pl, col in fp["payloads"].items():
                c = f"{sc} {pl}"
                if c not in df:
                    continue
                ok = df[c].notna()
                ax.plot(x[ok], df[c][ok], color=col, ls=ss["ls"], marker=ss["marker"], ms=fp["marker_size"], mfc="white",
                        lw=fp["lw"])
        a = fp["airsfl"]
        ax.plot(x, np.ones(len(df)), color=a["color"], marker=a["marker"], ms=a["ms"], ls=a["ls"], lw=a["lw"], zorder=1.9)
        if fp["airsfl_seconds"] and "AirSFL (s)" in df:
            for xi, s in zip(x, df["AirSFL (s)"]):
                ax.annotate(f"{s:,.0f}", (xi, 1.0), textcoords="offset points", xytext=(0, 5), ha="center",
                            fontsize=fp["font"] - 2.5, color=a["color"])
        ax.set_yscale("log")
        vals = df[[c for c in df if c.split(" ")[-1] in fp["payloads"]]].to_numpy(dtype=float)
        lo, hi = 0.85 * min(1.0, np.nanmin(vals)), 1.15 * np.nanmax(vals)
        ax.set_ylim(lo, hi)
        if fp["shade"]:
            ax.axhspan(lo, 1.0, color="#8350a8", alpha=0.08, lw=0)
            ax.text(0.03, (np.log(1.0) - np.log(lo)) / (np.log(hi) - np.log(lo)) / 2, fp["shade"], transform=ax.transAxes,
                    va="center", fontsize=fp["font"] - 0.5, color="0.35")
        ticks = [v for v in (0.5, 1, 2, 5, 10, 20, 50, 100, 200) if lo <= v <= hi]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.set_xticks([v for v in (-20, -10, 0, 10, 20) if x.min() <= v <= x.max()])
        ax.set_xlabel(FORMAT["snr_label"])
        ax.set_ylabel(fp["ylabel"])
        ax.grid(True, alpha=FORMAT["grid_alpha"])
        # legend above the axes: payload colors (row 1), scheme styles and AirSFL (row 2)
        for pl, col in fp["payloads"].items():
            entries[pl] = (plt.Line2D([], [], color=col, lw=fp["lw"] + 0.6), pl)
        for sc, ss in fp["schemes"].items():
            entries[sc] = (plt.Line2D([], [], color="0.25", ls=ss["ls"], marker=ss["marker"], mfc="white",
                                      ms=fp["marker_size"], lw=fp["lw"]), sc)
        entries["AirSFL"] = (plt.Line2D([], [], color=a["color"], marker=a["marker"], ms=a["ms"] - 0.5, ls=a["ls"],
                                        lw=a["lw"]), a["label"])
        h, lab = _grid(entries, (tuple(fp["payloads"]), tuple(fp["schemes"]) + ("AirSFL",)))
        ax.legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False,
                  fontsize=fp["legend_font"], handlelength=2.4, columnspacing=1.2)
        for ext in FORMAT["file_types"]:
            fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", dpi=FORMAT["dpi"])
        plt.close(fig)
    print(f"[fig] {name}")


def _payload_tex(t, out):
    """Compact payload table (PAYLOAD["table_rows"] x PAYLOAD["table_snrs"]) as LaTeX (booktabs) and Markdown:
    uplink s/round and final test accuracy at 20 dB, then the uplink time to fig3's target per SNR -- AirSFL in
    seconds, every other row relative to AirSFL ("k/n" = seeds that reached it when not all; "--" = none)."""
    snr = t[t.section == "snr"]
    if snr.empty:
        return
    A = float(snr["target val acc (%)"].iloc[0])
    cols = [c for c in PAYLOAD["table_snrs"] if np.isclose(snr["SNR (dB)"], c).any()]
    name = {v: k for k, v in COLUMN.items()}

    def cell(m, pl, c, tex):
        w = snr[(snr.method == COLUMN[m]) & (snr.payload == pl) & np.isclose(snr["SNR (dB)"], c)]
        if w.empty:
            return "n/a"
        w = w.iloc[0]
        hit, n = (int(v) for v in w["reached"].split("/"))
        if hit == 0:
            return "--"
        v = f"{w['UL time to target (s)']:,.0f} s" if m == "airsfl" else \
            (f"${w['UL time to target / AirSFL']:.2f}\\times$" if tex else f"{w['UL time to target / AirSFL']:.2f}x")
        return v + (f" ({hit}/{n})" if hit < n else "")

    head = ["Scheme", "Payload", "s/round", r"Acc. (\%)"] + [f"${c:g}$ dB" for c in cols]
    body, md = [], []
    prev = None
    for m, q in PAYLOAD["table_rows"]:
        pl = "analog" if q is None else ("FP16" if q == 16 else f"INT{q}")
        w20 = snr[(snr.method == COLUMN[m]) & (snr.payload == pl) & np.isclose(snr["SNR (dB)"], 20.0)]
        if w20.empty and not any(cell(m, pl, c, True) != "n/a" for c in cols):
            continue
        lab = METHODS[m]["label"].replace(" (proposed)", "") if m in METHODS else name.get(m, m)
        cells = [f"{w20['UL s/round'].iloc[0]:.2f}" if not w20.empty else "n/a",
                 f"{w20['final test acc (%)'].iloc[0]:.1f}" if not w20.empty else "n/a"]
        body.append(" & ".join([lab if lab != prev else "", pl] + cells + [cell(m, pl, c, True) for c in cols]) + r" \\")
        md.append("| " + " | ".join([lab, pl] + cells + [cell(m, pl, c, False) for c in cols]) + " |")
        prev = lab
    qs = [f"{snr[np.isclose(snr['SNR (dB)'], c)]['q* = c_ZF/2 (bits)'].iloc[0]:.1f}" for c in cols]
    body += [r"\midrule", " & ".join([r"$q^\star = c_{\mathrm{ZF}}/2$ (bits)", "", "", ""] + qs) + r" \\"]
    md.append("| " + " | ".join(["q* = c_ZF/2 (bits)", "", "", ""] + qs) + " |")
    tex = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
           r"\caption{Uplink communication time to " + f"{A:g}" + r"\% validation accuracy (non-IID, $N=30$): AirSFL in "
           r"seconds, the baselines relative to AirSFL; uplink time per round and final test accuracy at 20~dB. "
           r"$q^\star$: digital ZF payload with AirSFL's airtime per value.}",
           r"\label{tab:payload}", r"\begin{tabular}{ll" + "c" * (len(head) - 2) + "}", r"\toprule",
           " & ".join(head) + r" \\", r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(out, "table_payload.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    with open(os.path.join(out, "table_payload.md"), "w") as f:
        f.write(f"Uplink time to {A:g}% validation accuracy (non-IID, N = 30): AirSFL in seconds, others x AirSFL; "
                f"s/round and final test accuracy at 20 dB\n\n")
        f.write("| " + " | ".join(h.replace("$", "").replace("\\%", "%") for h in head) + " |\n|" + "---|" * len(head)
                + "\n")
        f.write("\n".join(md) + "\n")


# ---------------------------------------------------------------------------------------------------
# 2. PLOT (from the CSVs only)
# ---------------------------------------------------------------------------------------------------

def _style(m):
    return METHODS[m] if m in METHODS else VARIANTS[m]


def _figsize(name):
    return FORMAT["compact"].get(name, {}).get("figsize", FORMAT["figsize"])


def _subplots(name):
    return plt.subplots(figsize=_figsize(name))


def _ylabel(name, default):
    return FORMAT["compact"].get(name, {}).get("ylabel", default)


def _xlabel(name, default):
    return FORMAT["compact"].get(name, {}).get("xlabel", default)


def _lw(name, lw):
    """Line width of a method, scaled for a print-size figure."""
    return lw * FORMAT["compact"].get(name, {}).get("lw", 1.0)


def _ms(name, delta=0.0):
    """Marker size (FORMAT["marker_size"] + delta), or the print-size figure's."""
    c = FORMAT["compact"].get(name, {})
    return c["ms"] + delta * c["ms"] / FORMAT["marker_size"] if "ms" in c else FORMAT["marker_size"] + delta


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
    if FORMAT["title"] and title and name not in FORMAT["no_title_figs"]:
        ax.set_title(title, fontsize=FORMAT["font"])
    ax.grid(True, alpha=FORMAT["grid_alpha"])
    if "legend" in FORMAT["compact"].get(name, {}):     # print-size figure: its own legend spec
        _inside_legend(ax, FORMAT["compact"][name]["legend"])
    elif name in FORMAT["inside_legend"]:               # own legend inside the axes
        _inside_legend(ax, FORMAT["inside_legend"][name])
    elif name not in FORMAT["shared_legend_figs"]:      # shared ones: no legend, legend.png goes with them
        _legend(ax, fig)
    for ext in FORMAT["file_types"]:
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", dpi=FORMAT["dpi"])
    plt.close(fig)
    print(f"[fig] {name}")


def _inside_legend(ax, spec):
    """Legend inside the axes, entries in the order of legend.png and with its labels (e.g. "AirSFL")."""
    if spec.get("headroom"):                           # make room above the data
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * spec["headroom"] if ax.get_yscale() == "log" else lo + (hi - lo) * spec["headroom"])
    if spec.get("max_tick") is not None:               # no ticks in the legend's headroom (e.g. accuracy above 80%)
        lo, hi = ax.get_ylim()
        ax.set_yticks([t for t in ax.get_yticks() if lo <= t <= spec["max_tick"]])
        ax.set_ylim(lo, hi)
    if spec.get("footroom"):                           # make room below the data (linear axes)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - (hi - lo) * spec["footroom"], hi)
    lf = FORMAT["legend_figure"]
    key = {_style(m)["label"]: m for m in DRAWN}
    key.update({v: m for m, v in JOINT.items()})
    rank = {m: i for i, m in enumerate(lf["order"])}
    h, lab = ax.get_legend_handles_labels()
    # joint labels (fig1: one line per O/Z pair) stay as drawn; the others take the legend figure's labels
    text = lambda l: l if l in JOINT.values() else lf.get("labels", {}).get(key.get(l), l)
    if spec.get("rows"):                               # grid: one family per row, one payload per column
        h, lab = _grid({key.get(l): (hh, text(l)) for hh, l in zip(h, lab)}, spec["rows"])
        ncol = max(len(r) for r in spec["rows"])
    else:
        pairs = sorted(zip(h, lab), key=lambda p: rank.get(key.get(p[1]), len(rank)))
        h, lab, ncol = [p[0] for p in pairs], [text(p[1]) for p in pairs], spec.get("ncol", 1)
    ax.legend(h, lab, loc=spec.get("loc", "best"), ncol=ncol, bbox_to_anchor=spec.get("bbox"),
              fontsize=spec.get("font", FORMAT["inside_legend_font"]), frameon=FORMAT["inside_legend_frame"],
              framealpha=0.9, **({"columnspacing": spec.get("cs", 1.0), "handlelength": spec.get("hl", 2.2)}
                                 if spec.get("rows") or "cs" in spec or "hl" in spec else {}))


def _grid(entries, rows):
    """Handles and labels in matplotlib's column-major legend order so that the legend reads as `rows`
    (method keys; None or a method not drawn = empty cell). entries: method -> (handle, label)."""
    blank = plt.Line2D([], [], alpha=0)
    ncol = max(len(r) for r in rows)
    cells = [entries.get(r[c]) if c < len(r) and r[c] is not None else None for c in range(ncol) for r in rows]
    return [e[0] if e else blank for e in cells], [e[1] if e else "" for e in cells]


def _not_reached(ax, x, color):
    ax.plot([x], [0.96], marker="x", color=color, ms=9, mew=2, transform=ax.get_xaxis_transform(), clip_on=False)


def _part(name):
    return "Non-IID (Dir-0.1)" if "noniid" in name else "IID"


def plot_final_acc_vs_snr(df, out, name):
    fig, ax = _subplots(name)
    overlay = FORMAT["fig1_twins"] == "overlay"
    order = [m for m in DRAWN if m not in TWINS] + ([m for m in DRAWN if m in TWINS] if overlay else [])
    for m in order:                                    # OFDMA first, ZF twins (identical values) on top
        if COLUMN[m] not in df:
            continue
        st = _style(m)
        ax.plot(df["SNR (dB)"], df[COLUMN[m]], color=st["color"], marker=st["marker"], ls=st["ls"], lw=_lw(name, st["lw"]),
                ms=_ms(name, -1 if m in TWINS else 0), mfc=st.get("mfc"),
                label=st["label"] if overlay else JOINT.get(m, st["label"]))
    ax.set_xlabel(FORMAT["snr_label"])
    ax.set_ylabel(_ylabel(name, "Final test accuracy (%)"))
    _finish(fig, ax, out, name, f"{_part(name)}: final accuracy vs SNR")


def plot_time_to_target_vs_snr(df, out, name, axis="Training", methods=None):
    fig, ax = _subplots(name)
    unreached = False
    for m in methods or DRAWN:
        c = COLUMN[m]
        if c not in df:
            continue
        st = _style(m)
        ok = df[df[c].notna()]
        ax.plot(ok["SNR (dB)"], ok[c], color=st["color"], marker=st["marker"], ls=st["ls"], lw=_lw(name, st["lw"]),
                ms=_ms(name), mfc=st.get("mfc"), label=st["label"])
        for _, r in df.iterrows():
            if not isinstance(r.get(f"{c} [reached]"), str):
                continue                               # no run of this method at this SNR
            hit, n = (int(v) for v in r[f"{c} [reached]"].split("/"))
            if hit == 0:
                _not_reached(ax, r["SNR (dB)"], st["color"])
                unreached = True
            elif hit < n:
                ax.annotate(f"{hit}/{n}", (r["SNR (dB)"], r[c]), textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color=st["color"])
    ax.set_yscale("log")
    if unreached:                                      # headroom: the "not reached" x sits above the data
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 10)
    ax.set_xlabel(FORMAT["snr_label"])
    ax.set_ylabel(_ylabel(name, "Uplink communication time to target (s)" if axis == "Uplink" else
                          f"{axis} time to target (s)"))
    _finish(fig, ax, out, name, f"{_part(name)}: {axis.lower()} time to {df['target val acc (%)'].iloc[0]:.0f}% "
                                f"val. acc." + (" (x = not reached)" if unreached else ""))


def plot_budget_vs_snr(df, out, name):
    fig, ax = _subplots(name)
    for m in DRAWN:
        c = COLUMN[m]
        if c not in df:
            continue
        st = _style(m)
        hollow = df["no evaluated checkpoint in budget"].fillna("").str.split("; ").apply(lambda v: c in v)
        ok = df[~hollow]
        ax.plot(ok["SNR (dB)"], ok[c], color=st["color"], marker=st["marker"], ls=st["ls"], lw=_lw(name, st["lw"]),
                ms=_ms(name), mfc=st.get("mfc"), label=st["label"])
        if hollow.any():                               # initial model only (nothing evaluated inside the budget)
            ax.plot(df["SNR (dB)"][hollow], df[c][hollow], marker=st["marker"], mfc="white", mec=st["color"],
                    ms=_ms(name, 2), ls="")
    ax.set_xlabel(FORMAT["snr_label"])
    ax.set_ylabel(_ylabel(name, "Test accuracy within the budget (%)"))
    own = "budget" in df and str(df["budget"].iloc[0]) == "AirSFL's full run"
    _finish(fig, ax, out, name, f"{_part(name)}: accuracy after {df['budget (uplink s)'].iloc[0]:.0f} s of uplink "
                                f"time" + (" (AirSFL's run)" if own else ""))


def plot_time_to_target_vs_N(df, out, name):
    fig, ax = _subplots(name)
    ms = [m for m in DRAWN if COLUMN[m] in df]
    x = np.arange(len(df))
    w = 0.84 / len(ms)
    unreached = False
    for j, m in enumerate(ms):
        st, c = _style(m), COLUMN[m]
        xs = x + (j - (len(ms) - 1) / 2) * w
        ax.bar(xs, df[c], width=w, color=st["color"], edgecolor="k", lw=0.4, hatch=st.get("hatch"), label=st["label"])
        for xi, v in zip(xs, df[c]):
            if np.isnan(v):
                ax.text(xi, 0.02, "×", ha="center", color=st["color"], fontsize=10, transform=ax.get_xaxis_transform())
                unreached = True
    ax.set_yscale("log")
    if FORMAT["compact"].get(name, {}).get("ybottom"):
        ax.set_ylim(bottom=FORMAT["compact"][name]["ybottom"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(n)) for n in df["N"]])
    ax.set_xlabel("Number of clients $N$")
    ax.set_ylabel(_ylabel(name, "Uplink communication time to target (s)"))
    _finish(fig, ax, out, name, f"{_part(name)}: uplink time to {df['target val acc (%)'].iloc[0]:.0f}% val. acc."
                                + (" (× = not reached)" if unreached else ""))


def plot_curves(df, out, name):
    fig, ax = _subplots(name)
    ends = df[df.method.isin([COLUMN["aircomp_fl"], COLUMN["airsfl"]])].groupby("method")["uplink time (s)"].max()
    xmax = 1.05 * float(ends.max()) if len(ends) else None    # AirComp-FL's full run (the fastest baseline)
    for m in DRAWN:
        c = df[df.method == COLUMN[m]].sort_values("round")
        if c.empty:
            continue
        st = _style(m)
        ax.plot(c["uplink time (s)"], c["test acc (%)"], color=st["color"], ls=st["ls"], lw=_lw(name, st["lw"]), marker=st["marker"],
                markevery=max(1, len(c) // FORMAT["curve_markers"]), ms=_ms(name, -1), mfc=st.get("mfc"),
                label=st["label"])
        if FORMAT["extend_curves"] and xmax and float(c["uplink time (s)"].iloc[-1]) < xmax:
            ax.plot([float(c["uplink time (s)"].iloc[-1]), xmax], [float(c["test acc (%)"].iloc[-1])] * 2,
                    color=st["color"], ls=":", lw=_lw(name, 1.0), alpha=0.7)
    if xmax:
        ax.set_xlim(0, xmax)
    if FORMAT["compact"].get(name, {}).get("xk"):     # narrow panel: 0, 5k, 10k
        from matplotlib.ticker import FuncFormatter, MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:g}k"))
    ax.set_xlabel(_xlabel(name, "Accumulated uplink communication time (s)"))
    ax.set_ylabel("Test accuracy (%)")
    if FORMAT["inset_methods"] and name in FORMAT["inset_bounds"]:
        c = FORMAT["compact"].get(name, {})
        _inset_full_runs(ax, df, c.get("inset_bounds", FORMAT["inset_bounds"][name]),
                         c.get("inset_font", FORMAT["inset_font"]), c.get("lw", 1.0))
    _finish(fig, ax, out, name, f"{_part(name)}, $\\rho$ = 20 dB")


def _inset_full_runs(ax, df, bounds, font=None, lw_scale=1.0):
    """Small box inside the axes with the whole run of the slow baselines (same colors and line styles
    as the main plot; uplink time in thousands of seconds: 0, 50k, 100k, 150k)."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    font = font or FORMAT["inset_font"]
    ins = ax.inset_axes(bounds)
    for m in FORMAT["inset_methods"]:
        c = df[df.method == COLUMN[m]].sort_values("round")
        if c.empty:
            continue
        st = _style(m)
        ins.plot(c["uplink time (s)"], c["test acc (%)"], color=st["color"], ls=st["ls"], lw=lw_scale * max(1.0, 0.7 * st["lw"]))
    ins.set_xlim(0, None)
    ins.xaxis.set_major_locator(MaxNLocator(4))
    ins.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:g}k"))
    ins.yaxis.set_major_locator(MaxNLocator(4))
    ins.tick_params(labelsize=font, length=2, pad=1)
    ins.grid(True, alpha=FORMAT["grid_alpha"])
    if FORMAT["inset_title"]:
        ins.set_title(FORMAT["inset_title"], fontsize=font, pad=2)


def plot_all(out, payload=False):
    plt.rcParams.update({"font.size": FORMAT["font"]})
    if payload:                                        # finalz_payload: more legend entries, "(FP16)" labels
        FORMAT["inside_legend"] = {**FORMAT["inside_legend"], **PAYLOAD["inside_legend"]}
        FORMAT["legend_figure"] = {**PAYLOAD["legend_figure"], "labels": PAYLOAD["labels"]}
        JOINT.update(PAYLOAD["joint"])
    draw = {"final_acc_vs_snr": plot_final_acc_vs_snr,
            "uplink_time_to_target_vs_snr": lambda d, o, n, **k: plot_time_to_target_vs_snr(d, o, n, axis="Uplink", **k),
            "training_time_to_target_vs_snr": lambda d, o, n, **k: plot_time_to_target_vs_snr(d, o, n, axis="Training",
                                                                                               **k),
            "budget_vs_snr": plot_budget_vs_snr, "time_to_target_vs_N": plot_time_to_target_vs_N,
            "curves_20dB": plot_curves}
    for name, spec in {**FIGS, **(PAYLOAD_FIGS if payload else {})}.items():
        kind = spec[1]
        path = os.path.join(out, f"{name}.csv")
        if not os.path.exists(path):
            print(f"[fig] {name}: {path} missing (run without --plot-only first)")
            continue
        extra = {"methods": spec[2]["methods"]} if len(spec) > 2 and "methods" in spec[2] else {}
        with plt.rc_context({"font.size": FORMAT["compact"].get(name, {}).get("font", FORMAT["font"])}):
            draw[kind](pd.read_csv(path), out, name, **extra)
    if payload and os.path.exists(os.path.join(out, "fig_payload_precision.csv")):
        plot_payload_precision(pd.read_csv(os.path.join(out, "fig_payload_precision.csv")), out)
    if FORMAT["shared_legend_figs"]:
        plot_legend(out)


def plot_legend(out, name="legend"):
    """Stand-alone legend of all methods (line style, color and marker as in the figures), for the
    figures drawn without one (FORMAT["shared_legend_figs"])."""
    lf = FORMAT["legend_figure"]
    entries = {m: (plt.Line2D([], [], color=_style(m)["color"], ls=_style(m)["ls"], lw=_style(m)["lw"],
                              marker=_style(m)["marker"], mfc=_style(m).get("mfc"), ms=FORMAT["marker_size"] - 1),
                   lf.get("labels", {}).get(m, _style(m)["label"])) for m in lf["order"]}
    if lf.get("rows"):                                 # grid: one family per row
        handles, labels = _grid(entries, lf["rows"])
        cols, nrow = max(len(r) for r in lf["rows"]), len(lf["rows"])
    else:
        handles, labels = [entries[m][0] for m in lf["order"]], [entries[m][1] for m in lf["order"]]
        cols, nrow = lf["cols"], math.ceil(len(lf["order"]) / lf["cols"])
    fig = plt.figure(figsize=(lf["width"], 0.4 * nrow))
    fig.legend(handles, labels, loc="center", ncol=cols,
               fontsize=lf["font"], frameon=False, handlelength=2.6, columnspacing=1.4)
    for ext in FORMAT["file_types"]:
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", dpi=FORMAT["dpi"])
    plt.close(fig)
    print(f"[fig] {name}")


def main():
    p = argparse.ArgumentParser(description="The 7 AirSFL paper figures (seed means) + their CSVs -> <results>/finalz/")
    p.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_final_s1"))
    p.add_argument("--main-seeds", nargs="+", type=int, default=None,
                   help="seeds of the 20 dB (main) points, e.g. 11 while seeds 22/33 of main are missing")
    p.add_argument("--plot-only", action="store_true", help="redraw from finalz/*.csv (after a FORMAT change)")
    p.add_argument("--fig3-target", default=FIG3_TARGET,
                   help='fig3 target: "common" (default: highest accuracy every method reaches in every seed at every '
                        'SNR), "rule" (95%% of digital SFL-V1, as the other figures) or a number such as 65')
    p.add_argument("--payload-results", default=None,
                   help="folder of the INT-q runs (run_airsfl.py --digital-q 8 / 4, e.g. examples/AirSFL/results_qbits): "
                        "the figures with the INT-q baselines (VARIANTS), fig8 and table_payload -> <results>/finalz_payload/")
    p.add_argument("--payload-bits", nargs="+", type=int, default=list(PAYLOAD["bits"]),
                   help="INT-q payloads read from --payload-results (default 8 4)")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="use only these seeds everywhere, for every method (paired comparison, e.g. 11 22 33 while the "
                        "INT-q runs have three seeds); targets are recomputed from them unless given (--fig3-target)")
    p.add_argument("--out", default=None, help="output folder (default <results>/finalz or <results>/finalz_payload)")
    a = p.parse_args()
    out = a.out or os.path.join(os.path.abspath(a.results), "finalz_payload" if a.payload_results else "finalz")
    if not a.plot_only:
        build_data(a.results, a.main_seeds, out, a.fig3_target, a.payload_results, a.payload_bits, a.seeds)
    plot_all(out, payload=bool(a.payload_results))


if __name__ == "__main__":
    main()
