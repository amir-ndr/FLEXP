"""
examples/AirSFL/plots.py: figures + tables for the AirSFL study from the CSVs
written by run_airsfl.py (works on whatever results exist so far).

Target accuracy A* (roadmap Sec. 5): the integer percentage point below 95% of
the noiseless reference's final clean VALIDATION accuracy (digital FedAvg, per
partition), frozen for every method in that task. Attainment = first evaluated
round whose validation accuracy >= A*. Missed targets are reported as failures,
never assigned a finite time.

Figures -> results/figures/, tables -> results/tables/:
  fig1_acc_vs_uplink_time   test accuracy vs modeled uplink seconds (IID | non-IID)
  fig2_snr                  final accuracy and uplink time-to-target vs SNR
  fig3_uplink_latency       per-round uplink seconds, stacked activation/model/labels
                            (a) methods at cut 3, (b) all methods at cuts 1-4
  fig4_comm_overhead        source-equivalent MB vs airtime per round, and to target
  fig5_nr                   AirSFL accuracy vs uplink time for several Nr (10 dB)
  fig6_two_way              time to target: uplink only vs uplink+downlink (broadcast
                            rate bracketed: per-client and full-band)
  fig7_nsweep               N in {20,30,40}: per-round uplink bars + uplink time to A*(N)
  table_main / table_snr / table_cuts / table_two_way / table_nsweep  (.csv + .md + .tex)
"""

import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flsim.airsfl.timing import (METHODS, RadioConfig, digital_spectral_efficiency,
                                 downlink_spectral_efficiency, downlink_time_per_round, profiled_dims,
                                 source_equivalent_mb_per_round, uplink_time_breakdown,
                                 uplink_time_per_round)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIG = os.path.join(RESULTS, "figures")
TAB = os.path.join(RESULTS, "tables")

STYLE = {
    "airsfl":           dict(label="AirSFL (proposed)", color="#d62728", marker="o", lw=2.6, ls="-"),
    "digital_sflv1":    dict(label="Digital SFL-V1", color="#1f77b4", marker="s", lw=1.8, ls="-"),
    "sun_fdma_aircomp": dict(label="FDMA-AirComp SFL", color="#9467bd", marker="^", lw=1.8, ls="-"),
    "aircomp_fl":       dict(label="AirComp-FL", color="#2ca02c", marker="D", lw=1.8, ls="-"),
    "digital_fedavg":   dict(label="Digital FedAvg", color="#7f7f7f", marker="v", lw=1.6, ls="--"),
}
PART_NAME = {"iid": "IID", "dirichlet": "Non-IID (Dir-0.5)"}
ORDER = ["airsfl", "sun_fdma_aircomp", "aircomp_fl", "digital_sflv1", "digital_fedavg"]

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "legend.fontsize": 9, "figure.dpi": 110})


def _load(exp):
    files = glob.glob(os.path.join(RESULTS, exp, "*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def _curve(df):
    """Mean over paired seeds at each evaluation round (numeric columns)."""
    if df.empty:
        return df
    return df.select_dtypes("number").groupby("round", as_index=False).mean().sort_values("round")


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


def target_accuracy(main_df, scheme):
    """A*: integer % below 95% of the noiseless reference's final validation accuracy."""
    ref = main_df[(main_df.partition == scheme) & (main_df.method == "digital_fedavg")]
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


def _ul_per_round(method, dims, rho, Nr, tau=5, B=16):
    r = RadioConfig(N=int(dims.get("N", 30)), Nr=Nr, S=int(dims.get("S", 60)), rho_db=rho, batch_size=B)
    c = digital_spectral_efficiency(r, np.random.RandomState(2026))
    return uplink_time_per_round(method, dims, r, tau, c_D=c)


# ---------------------------------------------------------------------------
# Figure 1 + table_main
# ---------------------------------------------------------------------------

def fig1_and_table(main):
    if main.empty:
        return
    schemes = [s for s in ("iid", "dirichlet") if s in set(main.partition)]
    fig, axes = plt.subplots(1, len(schemes), figsize=(6.2 * len(schemes), 4.6), squeeze=False)
    rows = []
    for ax, scheme in zip(axes[0], schemes):
        A = target_accuracy(main, scheme)
        sub = main[main.partition == scheme]
        for m in ORDER:
            run = _curve(sub[sub.method == m])
            if run.empty:
                continue
            st = STYLE[m]
            x = run.uplink_s.clip(lower=1e-1)
            ax.plot(x, 100 * run.test_acc, color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])
            if A is not None:
                t, k = time_to_target(run, A)
                if t is not None:
                    ax.plot([max(t, 1e-1)], [100 * float(run[run["round"] == k].test_acc.iloc[0])],
                            marker=st["marker"], color=st["color"], ms=9, mec="k", zorder=5)
            t, k = time_to_target(run, A) if A is not None else (None, None)
            fin = run.iloc[-1]
            rows.append({"partition": PART_NAME[scheme], "method": st["label"],
                         "UL s/round": fin.ul_s_per_round,
                         "DL s/round (per-client bc)": fin.dl_s_per_round,
                         "MB/round": fin.mb_per_round,
                         "final test acc (%)": 100 * fin.test_acc,
                         "best val acc (%)": 100 * run.val_acc.max(),
                         "target A* (%)": 100 * A if A is not None else np.nan,
                         "rounds to A*": k if k is not None else np.nan,
                         "UL time to A* (s)": t if t is not None else np.nan,
                         "MB to A*": (k * fin.mb_per_round) if k is not None else np.nan})
        if A is not None:
            ax.axhline(100 * A, color="k", ls=":", lw=1)
            ax.text(ax.get_xlim()[0] if False else 0.12, 100 * A + 0.8, f"target {100*A:.0f}%", fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("Modeled uplink communication time (s)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"CIFAR-10, ResNet-18 — {PART_NAME[scheme]}")
    axes[0][0].legend(loc="lower right")
    fig.suptitle("Accuracy vs uplink time (N=30, Nr=128, W=0.9 MHz, rho=20 dB, cut 3)", y=1.02)
    _save(fig, "fig1_acc_vs_uplink_time")

    tab = pd.DataFrame(rows)
    if not tab.empty:
        for scheme in tab.partition.unique():
            ref = tab[(tab.partition == scheme) & (tab.method == STYLE["digital_sflv1"]["label"])]
            key = "UL time to A* (s)"
            if not ref.empty and not np.isnan(ref[key].iloc[0]):
                tab.loc[tab.partition == scheme, "speedup vs digital SFL-V1"] = \
                    float(ref[key].iloc[0]) / tab.loc[tab.partition == scheme, key]
        _write_table(tab, "table_main")


# ---------------------------------------------------------------------------
# Figure 2 + table_snr (digital methods: SNR-independent learning, analytic time)
# ---------------------------------------------------------------------------

def fig2_snr(main, snr):
    if main.empty:
        return
    base = main[main.rho_db == 20.0]
    frames = [base] + ([snr] if not snr.empty else [])
    allr = pd.concat(frames, ignore_index=True)
    schemes = [s for s in ("iid", "dirichlet") if s in set(allr.partition)]
    rhos = sorted(set(allr.rho_db))
    rows = []
    fig, axes = plt.subplots(len(schemes), 2, figsize=(11, 4.0 * len(schemes)), squeeze=False)
    for i, scheme in enumerate(schemes):
        A = target_accuracy(main, scheme)
        for m in ORDER:
            accs, times, fails = [], [], []
            for rho in rhos:
                if m in ("digital_sflv1", "digital_fedavg"):
                    run = _curve(base[(base.partition == scheme) & (base.method == m)])
                    if run.empty:
                        continue
                    dims = {"d_c": run.d_c.iloc[0], "d_s": run.d_s.iloc[0], "d_a": run.d_a.iloc[0],
                            "N": run.N.iloc[0], "S": run.S.iloc[0]}
                    ul = _ul_per_round(m, dims, rho, int(run.Nr.iloc[0]))
                    run = run.assign(uplink_s=run["round"] * ul)
                else:
                    run = _curve(allr[(allr.partition == scheme) & (allr.method == m) & (allr.rho_db == rho)
                                      & (allr.Nr == 128)])
                    if run.empty:
                        continue
                acc = 100 * float(run.test_acc.iloc[-1])
                t, k = time_to_target(run, A) if A is not None else (None, None)
                accs.append((rho, acc))
                times.append((rho, t))
                rows.append({"partition": PART_NAME[scheme], "method": STYLE[m]["label"], "SNR (dB)": rho,
                             "final test acc (%)": acc, "UL time to A* (s)": t if t is not None else np.nan,
                             "reached A*": t is not None})
            st = STYLE[m]
            if accs:
                axes[i][0].plot(*zip(*accs), color=st["color"], marker=st["marker"], ls=st["ls"],
                                lw=st["lw"], label=st["label"])
            ok = [(r, t) for r, t in times if t is not None]
            if ok:
                axes[i][1].plot(*zip(*ok), color=st["color"], marker=st["marker"], ls=st["ls"],
                                lw=st["lw"], label=st["label"])
            for r, t in times:
                if t is None:
                    axes[i][1].scatter([r], [np.nan], marker="x", color=st["color"])
        axes[i][0].set_xlabel("Reference SNR rho (dB)")
        axes[i][0].set_ylabel("Final test accuracy (%)")
        axes[i][0].set_title(f"{PART_NAME[scheme]}: accuracy vs SNR")
        axes[i][1].set_yscale("log")
        axes[i][1].set_xlabel("Reference SNR rho (dB)")
        axes[i][1].set_ylabel("Uplink time to target (s)")
        axes[i][1].set_title(f"{PART_NAME[scheme]}: time to {100*A:.0f}% (missing = not reached)"
                             if A is not None else PART_NAME[scheme])
    axes[0][0].legend()
    _save(fig, "fig2_snr")
    if rows:
        _write_table(pd.DataFrame(rows), "table_snr")


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
            label=f"activation uplink ({tau} per round)")
    ax1.bar(x + w / 2, mod, width=w, color="#aec7e8", edgecolor="k", lw=0.5,
            label="model / prefix-difference uplink")
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
                run = _curve(main[(main.partition == scheme) & (main.method == m)])
                t, k = time_to_target(run, A) if (A is not None and not run.empty) else (None, None)
                vals.append(k * run.mb_per_round.iloc[0] / 1e3 if k is not None else np.nan)
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
# Figure 5: antennas (Nr) sweep; Figure 6: two-way time (downlink sensitivity)
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
            run = _curve(sub[(sub.method == "airsfl") & (sub.Nr == Nr)])
            ax.plot(run.uplink_s.clip(lower=1e-1), 100 * run.test_acc, color=cmap(0.35 + 0.6 * i / max(1, len(nrs) - 1)),
                    lw=2, label=f"AirSFL, Nr={Nr} (Nr-N={Nr - int(run.N.iloc[0])})")
        ref = _curve(sub[sub.method == "digital_sflv1"])
        if not ref.empty:
            ax.plot(ref.uplink_s, 100 * ref.test_acc, color=STYLE["digital_sflv1"]["color"], lw=1.8,
                    label="Digital SFL-V1 (reliable)")
        ax.set_xscale("log")
        ax.set_xlabel("Modeled uplink communication time (s)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"Receive antennas, rho=10 dB — {PART_NAME[scheme]}")
        ax.legend(fontsize=8, loc="lower right")
    _save(fig, "fig5_nr")


def _dl_per_round(method, run, bc_scale):
    """Per-round downlink seconds (paper Eq. 25 / Table 1) for a run's environment."""
    r = RadioConfig(N=int(run.N.iloc[0]), Nr=int(run.Nr.iloc[0]), S=int(run.S.iloc[0]),
                    rho_db=float(run.rho_db.iloc[0]))
    c = downlink_spectral_efficiency(r, np.random.RandomState(2027))
    dims = {"d_c": run.d_c.iloc[0], "d_s": run.d_s.iloc[0], "d_a": run.d_a.iloc[0]}
    return downlink_time_per_round(method, dims, r, 5, c_DL=c, bc_scale=bc_scale)


def fig6_two_way(main):
    """Downlink sensitivity (paper Sec. 5.3, Eq. 29): communication time to A* under
    three accountings. The common-model broadcast rate R_bc is bracketed:
    per-client (R_bc = Rbar^DL_n) and full-band (R_bc = N * Rbar^DL_n)."""
    if main.empty:
        return
    schemes = [s for s in ("iid", "dirichlet") if s in set(main.partition)]
    acct = [("uplink only", "#bbbbbb", None),
            ("UL + DL, per-client broadcast", "#6baed6", "one"),
            ("UL + DL, full-band broadcast", "#08519c", "N")]
    fig, axes = plt.subplots(1, len(schemes), figsize=(6.6 * len(schemes), 4.8), squeeze=False)
    rows = []
    x = np.arange(len(ORDER))
    w = 0.27
    for ax, scheme in zip(axes[0], schemes):
        A = target_accuracy(main, scheme)
        sub = main[main.partition == scheme]
        for j, (name, color, bc) in enumerate(acct):
            vals = []
            for m in ORDER:
                run = _curve(sub[sub.method == m])
                if run.empty or A is None:
                    vals.append(np.nan)
                    continue
                N = int(run.N.iloc[0])
                dl = 0.0 if bc is None else _dl_per_round(m, run, 1.0 if bc == "one" else N)
                run = run.assign(tw=run["round"] * (run.ul_s_per_round + dl))
                t, k = time_to_target(run, A, xcol="tw")
                vals.append(t if t is not None else np.nan)
                rows.append({"partition": PART_NAME[scheme], "method": STYLE[m]["label"], "accounting": name,
                             "UL s/round": float(run.ul_s_per_round.iloc[0]), "DL s/round": dl,
                             "rounds to A*": k if k is not None else np.nan,
                             "time to A* (s)": t if t is not None else np.nan,
                             "final test acc (%)": 100 * float(run.test_acc.iloc[-1])})
            xs = x + (j - 1) * w
            ax.bar(xs, vals, width=w, color=color, edgecolor="k", lw=0.4, label=name)
            for xi, v in zip(xs, vals):
                if np.isnan(v):
                    ax.text(xi, 0.02, "×", ha="center", color=color, fontsize=10,
                            transform=ax.get_xaxis_transform())
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([STYLE[m]["label"].replace(" (proposed)", "") for m in ORDER], rotation=20, ha="right")
        ax.set_ylabel("Communication time to target (s)")
        ax.set_title(f"{PART_NAME[scheme]}: time to {100*A:.0f}%  (× = not reached)" if A is not None
                     else PART_NAME[scheme])
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Downlink sensitivity: OMA cut derivatives + common-model broadcast (bracketed)", y=1.02)
    fig.tight_layout()
    _save(fig, "fig6_two_way")
    if rows:
        _write_table(pd.DataFrame(rows), "table_two_way")


# ---------------------------------------------------------------------------
# Figure 7 + table_nsweep: number of clients N at fixed Nr and S
# ---------------------------------------------------------------------------

def fig7_nsweep(main, nsweep, Nr=128, S=60, rho=20.0, tau=5, B=16):
    """(a) analytic per-round uplink seconds vs N; (b, c) uplink time to A*(N) per partition.
    A*(N) comes from the digital FedAvg run at that N; N=30 runs come from `main`."""
    allr = pd.concat([f for f in (main, nsweep) if not f.empty], ignore_index=True) \
        if not (main.empty and nsweep.empty) else pd.DataFrame()
    if not allr.empty:
        allr = allr[allr.rho_db == rho]
    Ns = tuple(sorted({int(n) for n in allr.N})) if not nsweep.empty else (20, 30, 40)
    schemes = [s for s in ("iid", "dirichlet") if not allr.empty and s in set(allr.partition)]
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

    # (b, c) uplink time to A*(N)
    rows = []
    for ax, scheme in zip(axes[1:], schemes):
        sub = allr[allr.partition == scheme]
        for j, m in enumerate(ORDER):
            vals = []
            for N in Ns:
                subN = sub[sub.N == N]
                A = target_accuracy(subN, scheme)
                run = _curve(subN[subN.method == m])
                if run.empty or A is None:
                    vals.append(np.nan)
                    continue
                t, k = time_to_target(run, A)
                vals.append(t if t is not None else np.nan)
                rows.append({"partition": PART_NAME[scheme], "N": N, "method": STYLE[m]["label"],
                             "UL s/round": float(run.ul_s_per_round.iloc[0]),
                             "rounds/epoch": float(run["round"].iloc[-1] / run.epoch_equiv.iloc[-1]),
                             "target A* (%)": 100 * A, "rounds to A*": k if k is not None else np.nan,
                             "UL time to A* (s)": t if t is not None else np.nan,
                             "final test acc (%)": 100 * float(run.test_acc.iloc[-1])})
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
        ax.set_title(f"({'bc'[schemes.index(scheme)]}) {PART_NAME[scheme]}: time to A*(N)  (× = not reached)",
                     fontsize=10.5)
    fig.suptitle(f"Number of clients at fixed Nr={Nr}, W={S * 15e3 / 1e6:.2f} MHz, rho={rho:g} dB: "
                 "digital OFDMA splits the band, AirSFL reuses it", y=1.03)
    fig.tight_layout()
    _save(fig, "fig7_nsweep")
    if rows:
        tab = pd.DataFrame(rows)
        for (p, N), g in tab.groupby(["partition", "N"]):
            ref = g[g.method == STYLE["digital_sflv1"]["label"]]["UL time to A* (s)"]
            if len(ref) and not np.isnan(ref.iloc[0]):
                idx = (tab.partition == p) & (tab.N == N)
                tab.loc[idx, "speedup vs digital SFL-V1"] = float(ref.iloc[0]) / tab.loc[idx, "UL time to A* (s)"]
        _write_table(tab, "table_nsweep")


def main():
    import argparse
    global RESULTS, FIG, TAB
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=None, help="results root (default examples/AirSFL/results)")
    a = p.parse_args()
    if a.results:
        RESULTS = os.path.abspath(a.results)
        FIG, TAB = os.path.join(RESULTS, "figures"), os.path.join(RESULTS, "tables")
    main_df, snr_df, nr_df, ns_df = _load("main"), _load("snr"), _load("nr"), _load("nsweep")
    fig3_latency()
    fig4_overhead(main_df)
    fig1_and_table(main_df)
    fig2_snr(main_df, snr_df)
    fig5_nr(nr_df)
    fig6_two_way(main_df)
    fig7_nsweep(main_df, ns_df)


if __name__ == "__main__":
    main()
