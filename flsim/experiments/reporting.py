"""
experiments/reporting.py: automatic, paradigm-agnostic run outputs.

Every run — no matter which paradigm or algorithm produced it — can be turned
into ONE canonical CSV and a standard set of plots by a single call, so you
never write per-algorithm CSV/plot code. The experiment base classes call this
automatically at the end of run_single / run_single_async / run_single_split
(and any custom experiment can call `experiment.finalize_run(result, name)`
for a simulator it wired by hand).

It works by normalizing whatever columns a run's CSV happens to have onto a
fixed CANONICAL schema (aliasing each simulator's quirks, filling absent
columns with NaN), then emitting:

  <run_dir>/<name>_unified.csv     — the canonical schema (identical for every run)
  <run_dir>/<name>_<metric>.png    — the standard plot set

so any two runs are directly comparable and you can plot anything later from
the unified CSVs. This is additive: each simulator's own logger/CSV (and any
extra columns like staleness / num_clusters) is left untouched alongside.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# The common schema every run is normalized to.
CANONICAL_COLUMNS = [
    "round", "train_loss", "test_accuracy", "test_loss",
    "simulated_time_s", "round_latency_s", "avg_waiting_time_s",
    "traffic_bytes", "cumulative_traffic_bytes",
    "total_energy_j", "cumulative_energy_j",
]

# canonical name -> raw column names to look for, in priority order (covers the
# sync / async / split / async-split / cluster simulators' differing headers).
_ALIASES = {
    "round":                    ["round", "global_epoch"],
    "train_loss":               ["train_loss", "mean_train_loss"],
    "test_accuracy":            ["test_accuracy"],
    "test_loss":                ["test_loss"],
    "simulated_time_s":         ["simulated_time_s"],
    "round_latency_s":          ["round_latency_s", "round_duration_s", "total_time_s"],
    "avg_waiting_time_s":       ["avg_waiting_time_s"],
    "traffic_bytes":            ["traffic_bytes"],
    "cumulative_traffic_bytes": ["cumulative_traffic_bytes"],
    "total_energy_j":           ["total_energy_j"],
    "cumulative_energy_j":      ["cumulative_energy_j"],
}


def _first_present(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map any run's DataFrame onto CANONICAL_COLUMNS (alias-or-NaN, never raises).

    avg_waiting_time_s: used directly if present (cluster paradigm); else a
    proxy is derived for synchronous runs that expose the per-client timing
    breakdown (round_duration - mean per-client completion); else NaN.
    """
    out = pd.DataFrame()
    for canon in CANONICAL_COLUMNS:
        if canon == "avg_waiting_time_s":
            continue
        col = _first_present(df, _ALIASES.get(canon, [canon]))
        out[canon] = df[col] if col is not None else np.nan

    if "avg_waiting_time_s" in df.columns:
        out["avg_waiting_time_s"] = df["avg_waiting_time_s"]
    elif {"round_duration_s", "mean_compute_time_s", "mean_upload_time_s"}.issubset(df.columns):
        done = (df["mean_compute_time_s"] + df["mean_upload_time_s"]
                + df.get("mean_download_time_s", 0.0))
        out["avg_waiting_time_s"] = (df["round_duration_s"] - done).clip(lower=0.0)
    else:
        out["avg_waiting_time_s"] = np.nan

    return out[CANONICAL_COLUMNS]


def _lineplot(x, y, xlabel, ylabel, title, out, scale_y=1.0):
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(y)
    if m.sum() == 0:
        return   # nothing to plot (column absent for this paradigm) — skip silently
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.asarray(x)[m], y[m] * scale_y, marker="o", markersize=3,
            linewidth=1.5, color="tab:blue")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def standard_plots(ndf: pd.DataFrame, label: str, out_dir: str, prefix: str) -> None:
    """The standard per-run plot set (skips any metric a paradigm doesn't have)."""
    ev = ndf.dropna(subset=["test_accuracy"])   # evaluated rows for acc/loss
    specs = [
        (ev["round"], ev["test_accuracy"], "Round", "Test accuracy (%)", "Accuracy vs Round", "acc_vs_round", 100.0),
        (ev["simulated_time_s"], ev["test_accuracy"], "Simulated time (s)", "Test accuracy (%)", "Accuracy vs Time", "acc_vs_time", 100.0),
        (ev["round"], ev["test_loss"], "Round", "Test loss", "Test loss vs Round", "test_loss", 1.0),
        (ndf["round"], ndf["train_loss"], "Round", "Train loss", "Train loss vs Round", "train_loss", 1.0),
        (ndf["round"], ndf["cumulative_energy_j"], "Round", "Cumulative energy (J)", "Cumulative energy vs Round", "cumulative_energy", 1.0),
        (ndf["round"], ndf["cumulative_traffic_bytes"] / 1e6, "Round", "Cumulative traffic (MB)", "Cumulative traffic vs Round", "cumulative_traffic", 1.0),
        (ndf["round"], ndf["round_latency_s"], "Round", "Round latency (s)", "Per-round latency", "round_latency", 1.0),
        (ndf["round"], ndf["avg_waiting_time_s"], "Round", "Avg. waiting time (s)", "Average waiting time", "avg_waiting_time", 1.0),
    ]
    for x, y, xl, yl, title, fname, sc in specs:
        _lineplot(x, y, xl, yl, f"{label} — {title}",
                  os.path.join(out_dir, f"{prefix}_{fname}.png"), scale_y=sc)


def write_standard_outputs(df: pd.DataFrame, label: str, out_dir: str, prefix: str) -> pd.DataFrame:
    """
    Write <prefix>_unified.csv (canonical schema) + the standard plot set into
    out_dir. Returns the normalized DataFrame.
    """
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.basename(prefix)   # never let a slashed prefix write into a missing subdir
    ndf = normalize_df(df)
    ndf.to_csv(os.path.join(out_dir, f"{prefix}_unified.csv"), index=False)
    standard_plots(ndf, label, out_dir, prefix)
    return ndf
