"""
plot_results.py — standalone plotter for flsim CSV output.

Usage:
    python plot_results.py outputs/mnist_fedavg_shard.csv
    python plot_results.py outputs/mnist_fedavg_shard.csv --out my_plots/

Reads a CSV produced by flsim's Logger and saves the same set of plots
that Logger.plot_results() generates at the end of a run. Useful for
re-plotting after the fact or comparing multiple runs.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _savefig(fig, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # total_energy_j in the CSV is the per-client sum of compute+tx;
    # tx energy is the difference.
    df["tx_energy_j"] = df["total_energy_j"] - df["mean_compute_energy_j"]
    df["mean_rate_mbps"] = df["mean_rate_bps"] / 1e6
    df["cumulative_energy_j"] = df["total_energy_j"].cumsum()
    return df


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_acc_vs_round(df, out_dir, name):
    edf = df.dropna(subset=["test_accuracy"])
    if edf.empty:
        return
    fig, ax = plt.subplots()
    ax.plot(edf["round"], edf["test_accuracy"], marker="o", linewidth=1.5, markersize=4)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"{name} — accuracy vs round")
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_dir, f"{name}_acc_vs_round.png")


def plot_acc_vs_time(df, out_dir, name):
    edf = df.dropna(subset=["test_accuracy"])
    if edf.empty:
        return
    fig, ax = plt.subplots()
    ax.plot(edf["simulated_time_s"], edf["test_accuracy"],
            marker="o", linewidth=1.5, markersize=4, color="tab:orange")
    ax.set_xlabel("Simulated time (s)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"{name} — accuracy vs simulated time")
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_dir, f"{name}_acc_vs_time.png")


def plot_test_loss_vs_round(df, out_dir, name):
    edf = df.dropna(subset=["test_loss"])
    if edf.empty:
        return
    fig, ax = plt.subplots()
    ax.plot(edf["round"], edf["test_loss"],
            marker="o", linewidth=1.5, markersize=4, color="tab:red")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test loss")
    ax.set_title(f"{name} — test loss vs round")
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_dir, f"{name}_test_loss_vs_round.png")


def plot_timing_breakdown(df, out_dir, name):
    fig, ax = plt.subplots()
    ax.plot(df["round"], df["mean_compute_time_s"], label="Mean compute",   linewidth=1.5)
    ax.plot(df["round"], df["mean_upload_time_s"],  label="Mean upload",    linewidth=1.5)
    ax.plot(df["round"], df["max_compute_time_s"],  label="Max compute",    linewidth=1.5, linestyle="--")
    ax.plot(df["round"], df["round_duration_s"],    label="Round duration", linewidth=1.5, linestyle=":")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Simulated time (s)")
    ax.set_title(f"{name} — timing breakdown")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_dir, f"{name}_timing_breakdown.png")


def plot_energy_breakdown(df, out_dir, name):
    # Subsample to ≤50 bars for readability
    step = max(1, len(df) // 50)
    sub  = df.iloc[::step]

    fig, ax = plt.subplots()
    ax.bar(sub["round"], sub["mean_compute_energy_j"],
           label="Compute", width=step * 0.8)
    ax.bar(sub["round"], sub["tx_energy_j"],
           label="Transmission", width=step * 0.8,
           bottom=sub["mean_compute_energy_j"])
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Energy per client per round (J)")
    ax.set_title(f"{name} — energy breakdown")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    _savefig(fig, out_dir, f"{name}_energy_breakdown.png")


def plot_cumulative_energy(df, out_dir, name):
    fig, ax = plt.subplots()
    ax.plot(df["round"], df["cumulative_energy_j"], linewidth=1.5, color="tab:brown")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Cumulative energy (J)")
    ax.set_title(f"{name} — cumulative energy")
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_dir, f"{name}_cumulative_energy.png")


def plot_channel_gain(df, out_dir, name):
    fig, ax = plt.subplots()
    ax.plot(df["round"], df["mean_channel_gain"], linewidth=1.0, color="tab:cyan", alpha=0.8)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Mean channel gain (linear)")
    ax.set_yscale("log")
    ax.set_title(f"{name} — channel gain vs round")
    ax.grid(True, alpha=0.3, which="both")
    _savefig(fig, out_dir, f"{name}_channel_gain_vs_round.png")


def plot_rate(df, out_dir, name):
    fig, ax = plt.subplots()
    ax.plot(df["round"], df["mean_rate_mbps"], linewidth=1.0, color="tab:olive", alpha=0.8)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Mean rate (Mbps)")
    ax.set_title(f"{name} — achievable rate vs round")
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_dir, f"{name}_rate_vs_round.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Re-plot flsim CSV results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv", help="Path to the flsim output CSV file.")
    parser.add_argument(
        "--out", default=None,
        help="Output directory for plots. Defaults to the same folder as the CSV.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"Error: file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)

    # Derive a base name from the CSV filename (without extension)
    name = os.path.splitext(os.path.basename(args.csv))[0]

    print(f"Reading  : {args.csv}")
    print(f"Output   : {out_dir}/")
    print(f"Basename : {name}")
    print()

    df = _load(args.csv)
    print(f"Loaded {len(df)} rounds.\n")

    plot_acc_vs_round(df, out_dir, name)
    plot_acc_vs_time(df, out_dir, name)
    plot_test_loss_vs_round(df, out_dir, name)
    plot_timing_breakdown(df, out_dir, name)
    plot_energy_breakdown(df, out_dir, name)
    plot_cumulative_energy(df, out_dir, name)
    plot_channel_gain(df, out_dir, name)
    plot_rate(df, out_dir, name)

    print(f"\nDone. {8} plots written to {out_dir}/")


if __name__ == "__main__":
    main()
