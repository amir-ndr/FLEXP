"""
examples/CSASFL/plots.py: figures + tables of mnist_exp (examples/CSASFL/run_mnist_exp.py).

Reads every complete run in <results>/runs/ and averages over seeds (the paper averages 20 runs).
Event-driven methods are evaluated at different times in every seed, so a seed's accuracy is put
on a common x grid (simulated time or cumulative MB) by zero-order hold (the accuracy of the last
evaluation at or before x); the mean is drawn up to the shortest seed (every seed contributes to
every plotted point) with a +-1 std band. Every figure is written as PNG + PDF with the CSV of the
plotted means next to it, into <results>/figures/:

  main_delta<d>_{time,time_log,comm,round}   CSA-SFL vs FL / SFLv2 / SAFSL / ParallelSFL
  nsweep_{time,comm}                          CSA-SFL, N sweep (delta 0.1, H 20)        [paper Fig. a, b]
  hsweep_final, hsweep_time                   CSA-SFL, H sweep (delta 0.1, N 7)         [paper Fig. c]
  ablation_time                               full vs w/o weighted aggregation vs w/o gradient clustering
  tables.txt / *_table.csv                    final accuracy, time- and MB-to-target (mean +- std, reached/n)

Usage:  python examples/CSASFL/plots.py [--results DIR] [--targets 0.9 0.95 0.97] [--tmax SECONDS]
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_ORDER = ["fl", "sflv2", "safsl", "parallelsfl", "csasfl"]
LABEL = {"fl": "FL", "sflv2": "SFLv2", "safsl": "SAFSL", "parallelsfl": "ParallelSFL", "csasfl": "CSA-SFL"}
COLOR = {"fl": "#1f77b4", "sflv2": "#ff7f0e", "safsl": "#2ca02c", "parallelsfl": "#9467bd", "csasfl": "#d62728"}
VARIANT_LABEL = {"full": "CSA-SFL (full)", "uniform_agg": "w/o weighted aggregation",
                 "random_cluster": "w/o gradient clustering"}
XLAB = {"sim_time_s": "Training time (s)", "comm_mb": "Communication overhead (MB)", "round": "Global round"}


def load(results: str) -> pd.DataFrame:
    frames, partial = [], []
    for f in sorted(glob.glob(os.path.join(results, "runs", "*.csv"))):
        d = pd.read_csv(f)
        tb = float(d["time_budget"].iloc[0]) if "time_budget" in d and len(d) else 0.0
        if len(d) and (d["round"].max() >= d["T"].iloc[0] or d["diverged"].astype(bool).any()
                       or (tb > 0 and d["sim_time_s"].max() >= tb)):
            frames.append(d.assign(file=os.path.basename(f)))
        else:
            partial.append(os.path.basename(f))
    if partial:
        print(f"[plots] {len(partial)} incomplete runs ignored (still running / timed out): {partial[:5]}...")
    if not frames:
        raise SystemExit(f"[plots] no complete runs in {results}/runs")
    df = pd.concat(frames, ignore_index=True)
    print(f"[plots] {df['file'].nunique()} complete runs")
    return df


CONFIG_COLS = ["method", "variant", "delta", "seed", "K", "E", "B", "lr", "t0", "sigma_s", "N", "H"]  # not T: a cap


def common_budget(df: pd.DataFrame, budget=None) -> pd.DataFrame:
    """One run per configuration and seed (the longest time budget wins), every run cut at the
    common budget: its rows up to the first round at or past it -- exactly the run that budget
    would have produced, since the budget only decides when a run stops. Default budget: the
    smallest one present (all runs then cover the same simulated-time window)."""
    tbs = df.groupby("file")["time_budget"].first().fillna(0.0)
    if budget is None:
        budget = float(tbs[tbs > 0].min()) if (tbs > 0).any() else 0.0
    key = df[CONFIG_COLS].astype(str).agg("|".join, axis=1)
    longest = (df.assign(_k=key, _tb=df["time_budget"].fillna(0.0).replace(0.0, np.inf))
                 .groupby("_k")[["_tb", "file"]].apply(lambda g: g.sort_values("_tb").iloc[-1]["file"]))
    df = df[df["file"].isin(set(longest))]
    if budget > 0:
        parts = []
        for _, g in df.groupby("file"):
            g = g.sort_values("round")
            past = np.flatnonzero(g["sim_time_s"].to_numpy() >= budget)
            parts.append(g.iloc[:past[0] + 1] if len(past) else g)
        df = pd.concat(parts, ignore_index=True)
        df["time_budget"] = budget
        n_cut = int((tbs > budget).sum())
        print(f"[plots] common time budget {budget:g} s" + (f" ({n_cut} longer runs cut)" if n_cut else ""))
    return df


def variant_label(v: str) -> str:
    """'full+sys1+slot0.5+delta+sw+lazy' -> readable legend label."""
    tags = v.split("+")[1:]
    if not tags:
        return "CSA-SFL (paper)"
    kinds = {("sys" if t.startswith("sys") else "slot" if t.startswith("slot") else t) for t in tags}
    if kinds == {"sys", "slot", "delta", "sw", "lazy"}:
        return "CSA-SFL + all 4 improvements"
    if kinds == {"slot", "delta", "sw", "lazy"}:
        return "CSA-SFL + all but sys-aware clust."
    names = []
    for tg in tags:
        if tg.startswith("sys"):
            names.append(f"sys-aware clust. (lam={tg[3:]})")
        elif tg.startswith("dl"):
            names.append(f"deadline (q={tg[2:]})")
        elif tg.startswith("slot"):
            names.append(f"time slot (q={tg[4:]})")
        elif tg == "delta":
            names.append("update agg.")
        elif tg == "sw":
            names.append("sweeps staleness")
        elif tg == "lazy":
            names.append("lazy re-clust.")
        elif tg == "batch":
            names.append("batched arrivals")
    return "+ " + ", ".join(names)


def runs_of(df, **sel):
    m = np.ones(len(df), dtype=bool)
    for k, v in sel.items():
        m &= np.isclose(df[k].astype(float), float(v)) if isinstance(v, (int, float)) else (df[k] == v)
    sub = df[m]
    return [g.sort_values("round") for _, g in sub.groupby("file")]


def hold(x, y, grid):
    i = np.searchsorted(x, grid, side="right") - 1
    out = np.where(i >= 0, y[np.clip(i, 0, None)], np.nan)
    out[grid > x[-1] + 1e-9] = np.nan
    return out


def persistent(r):
    """The model state that actually persists at each simulated instant: the last row per
    timestamp. Several clusters can finish at the same instant (CSA-SFL time-slot variants); the
    PS aggregates them one after another and logs an evaluation after each, but the intermediate
    global models exist for zero simulated time and must not count for time / MB to accuracy."""
    return r.groupby("sim_time_s", sort=True).tail(1)


def mean_curve(runs, xcol, log=False, n=400):
    if not runs:
        return None
    if xcol != "round":
        runs = [persistent(r) for r in runs]
    end = min(r[xcol].iloc[-1] for r in runs)
    if log:
        lo = min(r[xcol][r[xcol] > 0].min() for r in runs)
        grid = np.geomspace(lo, end, n) if end > lo else np.array([end])
    else:
        grid = np.linspace(0.0, end, n)
    Y = np.stack([hold(r[xcol].to_numpy(float), r["test_acc"].to_numpy(float), grid) for r in runs])
    ok = ~np.isnan(Y).any(0)
    return pd.DataFrame({xcol: grid[ok], "acc_mean": Y[:, ok].mean(0), "acc_std": Y[:, ok].std(0), "n_seeds": len(runs)})


def plot_curves(curves: dict, xcol, path, title, log=False, xmax=None, styles=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    table = []
    for key, c in curves.items():
        if c is None or c.empty:
            continue
        st = (styles or {}).get(key, {})
        x, mu, sd = c[xcol].to_numpy(), 100 * c["acc_mean"].to_numpy(), 100 * c["acc_std"].to_numpy()
        ax.plot(x, mu, label=f"{key}", lw=st.get("lw", 1.6), color=st.get("color"), ls=st.get("ls", "-"))
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.12, color=ax.lines[-1].get_color(), lw=0)
        table.append(c.assign(curve=key))
    if log:
        ax.set_xscale("log")
    if xmax:
        ax.set_xlim(right=xmax)
    ax.set_xlabel(XLAB.get(xcol, xcol) + (" (log)" if log else ""))
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)
    if table:
        pd.concat(table, ignore_index=True).to_csv(f"{path}.csv", index=False)
    print(f"[plots] {path}.png")


def first_hit(r, col, target):
    hit = r.index[r["test_acc"] >= target]
    return float(r.loc[hit[0], col]) if len(hit) else np.nan


def summary_rows(groups: dict, targets):
    """groups: name -> list of runs. Final accuracy + time/MB to each target (mean +- std over seeds)."""
    rows = []
    for name, runs in groups.items():
        if not runs:
            continue
        runs = [persistent(r) for r in runs]
        fin = np.array([r["test_acc"].iloc[-1] for r in runs])
        last10 = np.array([r["test_acc"].iloc[-10:].mean() for r in runs])
        row = {"config": name, "n_seeds": len(runs), "final_acc_mean": fin.mean(), "final_acc_std": fin.std(),
               "last10_acc_mean": last10.mean(),
               "end_time_s_mean": np.mean([r["sim_time_s"].iloc[-1] for r in runs]),
               "end_comm_mb_mean": np.mean([r["comm_mb"].iloc[-1] for r in runs])}
        for A in targets:
            for col, tag in (("sim_time_s", "time_s"), ("comm_mb", "mb")):
                v = np.array([first_hit(r, col, A) for r in runs])
                row[f"{tag}_to_{int(round(100 * A))}_mean"] = np.nanmean(v) if np.isfinite(v).any() else np.nan
                row[f"{tag}_to_{int(round(100 * A))}_std"] = np.nanstd(v) if np.isfinite(v).any() else np.nan
            row[f"reached_{int(round(100 * A))}"] = f"{int(np.isfinite(v).sum())}/{len(runs)}"
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_table(t: pd.DataFrame, title: str, targets) -> str:
    if t.empty:
        return ""
    lines = [f"== {title} ==",
             f"{'config':32s} {'n':>3s} {'final acc (%)':>15s} " +
             " ".join(f"{'t->' + str(int(100 * A)) + '% (s)':>17s} {'MB->' + str(int(100 * A)) + '%':>14s} {'hit':>5s}"
                      for A in targets)]
    for _, r in t.iterrows():
        s = f"{r['config']:32s} {int(r['n_seeds']):3d} {100 * r['final_acc_mean']:7.2f} +-{100 * r['final_acc_std']:5.2f} "
        for A in targets:
            a = int(round(100 * A))
            tm, ts, mm = r[f"time_s_to_{a}_mean"], r[f"time_s_to_{a}_std"], r[f"mb_to_{a}_mean"]
            s += (f"{tm:9.0f} +-{ts:6.0f} {mm:14.0f} " if np.isfinite(tm) else f"{'--':>17s} {'--':>14s} ") + f"{r[f'reached_{a}']:>5s} "
        lines.append(s)
    return "\n".join(lines) + "\n\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.path.join(HERE, "results_mnist_exp"))
    p.add_argument("--targets", nargs="+", type=float, default=[0.90, 0.95, 0.97])
    p.add_argument("--tmax", type=float, default=None, help="x limit (s) of the linear time panels")
    p.add_argument("--noniid", type=float, default=0.1)
    p.add_argument("--time-budget", type=float, default=None,
                   help="cut every run at this simulated time (s); default: the smallest budget among the runs")
    p.add_argument("--N", type=int, default=7, help="default N (fixed in the H sweep)")
    p.add_argument("--H", type=int, default=20, help="default H (fixed in the N sweep)")
    args = p.parse_args()
    df = common_budget(load(args.results), args.time_budget)
    out = os.path.join(args.results, "figures")
    os.makedirs(out, exist_ok=True)
    report = ""
    # runs are selected by configuration (not by the experiment that trained them: a shared run,
    # e.g. CSA-SFL N=7 H=20 delta=0.1, belongs to the comparison, both sweeps and the ablation)

    # ---- comparison (main) ----
    main_df = df
    for delta in sorted(main_df["delta"].unique()):
        groups = {}
        for m in METHOD_ORDER:
            sel = dict(method=m, delta=delta)
            if m == "csasfl":
                sel.update(N=args.N, H=args.H, variant="full")
            groups[m] = runs_of(main_df, **sel)
        styles = {LABEL[m]: {"color": COLOR[m], "lw": 2.4 if m == "csasfl" else 1.5} for m in METHOD_ORDER}
        tag = f"main_delta{delta:g}"
        name = f"MNIST, delta = {delta:g}, K = 50 (mean of {max(len(v) for v in groups.values())} seeds)"
        cs = groups["csasfl"]
        tb = float(main_df["time_budget"].max()) if "time_budget" in main_df else 0.0
        tmax = args.tmax or (tb if tb > 0 else (1.5 * max(r["sim_time_s"].iloc[-1] for r in cs) if cs else None))
        for xcol, log, suf, xm in (("sim_time_s", False, "time", tmax), ("sim_time_s", True, "time_log", None),
                                   ("comm_mb", True, "comm", None), ("round", False, "round", None)):
            curves = {LABEL[m]: mean_curve(groups[m], xcol, log=log) for m in METHOD_ORDER if groups[m]}
            plot_curves(curves, xcol, os.path.join(out, f"{tag}_{suf}"), name, log=log, xmax=xm, styles=styles)
        t = summary_rows({LABEL[m]: groups[m] for m in METHOD_ORDER}, args.targets)
        t.to_csv(os.path.join(out, f"{tag}_table.csv"), index=False)
        report += fmt_table(t, f"comparison, delta = {delta:g}", args.targets)

    # ---- N sweep ----
    sw = df[(df["method"] == "csasfl") & np.isclose(df["delta"], args.noniid) & (df["variant"] == "full")]
    ns = sorted(sw[np.isclose(sw["H"], args.H)]["N"].dropna().unique())
    if len(ns) > 1:
        groups = {f"N={int(n)}": runs_of(sw, N=n, H=args.H) for n in ns}
        for xcol, suf, log in (("sim_time_s", "time", False), ("comm_mb", "comm", False)):
            plot_curves({k: mean_curve(v, xcol, log=log) for k, v in groups.items()}, xcol,
                        os.path.join(out, f"nsweep_{suf}"), f"CSA-SFL, delta = {args.noniid:g}, H = {args.H}: varying N")
        t = summary_rows(groups, args.targets)
        t.to_csv(os.path.join(out, "nsweep_table.csv"), index=False)
        report += fmt_table(t, f"N sweep (delta {args.noniid:g}, H {args.H})", args.targets)

    # ---- H sweep ----
    hs = sorted(sw[np.isclose(sw["N"], args.N)]["H"].dropna().unique())
    if len(hs) > 1:
        groups = {f"H={int(h)}": runs_of(sw, N=args.N, H=h) for h in hs}
        plot_curves({k: mean_curve(v, "sim_time_s") for k, v in groups.items()}, "sim_time_s",
                    os.path.join(out, "hsweep_time"), f"CSA-SFL, delta = {args.noniid:g}, N = {args.N}: varying H")
        t = summary_rows(groups, args.targets)
        t.to_csv(os.path.join(out, "hsweep_table.csv"), index=False)
        report += fmt_table(t, f"H sweep (delta {args.noniid:g}, N {args.N})", args.targets)
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        ax.errorbar([int(h) for h in hs], 100 * t["final_acc_mean"], yerr=100 * t["final_acc_std"],
                    marker="o", capsize=4, color=COLOR["csasfl"])
        ax.set_xlabel("Re-clustering interval H (global rounds)")
        ax.set_ylabel("Final test accuracy (%)")
        tb = float(sw["time_budget"].max()) if "time_budget" in sw else 0.0
        ax.set_title(f"Effect of H (delta = {args.noniid:g}, N = {args.N}, "
                     + (f"time budget {tb:g} s)" if tb > 0 else f"T = {int(sw['T'].iloc[0])})"), fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out, f"hsweep_final.{ext}"), dpi=200)
        plt.close(fig)
        t[["config", "n_seeds", "final_acc_mean", "final_acc_std", "last10_acc_mean"]].to_csv(
            os.path.join(out, "hsweep_final.csv"), index=False)
        print(f"[plots] {out}/hsweep_final.png")

    # ---- ablation ----
    ab = df[(df["method"] == "csasfl") & np.isclose(df["delta"], args.noniid)]
    if ab["variant"].isin(["uniform_agg", "random_cluster"]).any():
        groups = {VARIANT_LABEL.get(v, v): runs_of(ab, variant=v, N=args.N, H=args.H)
                  for v in ["full", "uniform_agg", "random_cluster"] if (ab["variant"] == v).any()}
        styles = {VARIANT_LABEL["full"]: {"color": COLOR["csasfl"], "lw": 2.4},
                  VARIANT_LABEL["uniform_agg"]: {"color": "#555555", "ls": "--"},
                  VARIANT_LABEL["random_cluster"]: {"color": "#8c564b", "ls": "-."}}
        plot_curves({k: mean_curve(v, "sim_time_s") for k, v in groups.items()}, "sim_time_s",
                    os.path.join(out, "ablation_time"), f"Ablation, delta = {args.noniid:g}, N = {args.N}, H = {args.H}",
                    styles=styles)
        t = summary_rows(groups, args.targets)
        t.to_csv(os.path.join(out, "ablation_table.csv"), index=False)
        report += fmt_table(t, "ablation", args.targets)

    # ---- CSA-SFL improvement variants (run_mnist_exp.py --exp variants) ----
    cv = df[(df["method"] == "csasfl") & np.isclose(df["N"], args.N) & np.isclose(df["H"], args.H)
            & df["variant"].astype(str).str.startswith("full")]
    for delta in sorted(cv["delta"].unique()):
        names = sorted(cv[np.isclose(cv["delta"], delta)]["variant"].unique(), key=lambda v: (v.count("+"), v))
        if len(names) < 2:
            continue
        groups = {variant_label(v): runs_of(cv, delta=delta, variant=v) for v in names}
        ref = {LABEL[m]: runs_of(df, method=m, delta=delta) for m in ("parallelsfl", "safsl", "sflv2", "fl")}
        ref = {k: v for k, v in ref.items() if v}
        palette = plt.cm.tab10.colors
        styles = {lab: {"color": palette[i % 10], "lw": 2.2 if i in (0, len(groups) - 1) else 1.5}
                  for i, lab in enumerate(groups)}
        styles.update({LABEL[m]: {"color": "#999999", "ls": ls, "lw": 1.0}
                       for m, ls in (("parallelsfl", ":"), ("safsl", "--"), ("sflv2", "-."), ("fl", (0, (1, 3))))})
        tb = float(cv["time_budget"].max()) if "time_budget" in cv else 0.0
        title = f"CSA-SFL improvements, delta = {delta:g} (grey: baselines)"
        for xcol, log, suf, xm in (("sim_time_s", False, "time", args.tmax or (tb if tb > 0 else None)),
                                   ("sim_time_s", True, "time_log", None), ("comm_mb", True, "comm", None)):
            curves = {k: mean_curve(v, xcol, log=log) for k, v in {**groups, **ref}.items()}
            plot_curves(curves, xcol, os.path.join(out, f"variants_delta{delta:g}_{suf}"), title, log=log,
                        xmax=xm, styles=styles)
        t = summary_rows({**groups, **ref}, args.targets)
        t.to_csv(os.path.join(out, f"variants_delta{delta:g}_table.csv"), index=False)
        report += fmt_table(t, f"CSA-SFL improvements, delta = {delta:g}", args.targets)

    with open(os.path.join(out, "tables.txt"), "w") as f:
        f.write(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
