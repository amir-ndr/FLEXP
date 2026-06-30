"""
examples/fedavg_vs_fedprox.py: FedAvg vs FedProx comparison experiment.

Runs both algorithms under identical conditions (same config, same seed,
same data partition) and produces side-by-side comparison plots.

Results saved to:
    outputs/fedavg_vs_fedprox/
        fedavg/fedavg.csv
        fedprox_mu0p01/fedprox_mu0p01.csv
        fedprox_mu0p1/fedprox_mu0p1.csv
        comparison_acc_vs_round.png
        comparison_acc_vs_time.png
        comparison_cumulative_energy.png
        bar_final_accuracy.png

Run from the FLEXP/ root:
    python examples/fedavg_vs_fedprox.py
    python examples/fedavg_vs_fedprox.py --rounds 50
    python examples/fedavg_vs_fedprox.py --rounds 50 --mu 0.01 0.1 0.5
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flsim.algorithms.fedavg import FedAvg
from flsim.algorithms.fedprox import FedProx
from flsim.experiments import AlgorithmComparison


BASE_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "flsim", "configs", "mnist_fedavg.yaml"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "fedavg_vs_fedprox"
)


def plot_cumulative_energy(results: dict, output_dir: str) -> None:
    """Plot cumulative energy (J) vs round for all runs on one axes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors
    for i, (label, result) in enumerate(results.items()):
        df = result.df.copy()
        df["cumulative_energy_j"] = df["total_energy_j"].cumsum()
        ax.plot(df["round"], df["cumulative_energy_j"],
                linewidth=1.5, label=label, color=colors[i % 10])
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Cumulative energy (J)")
    ax.set_title("Cumulative energy consumption")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_cumulative_energy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Experiment] Saved {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare FedAvg vs FedProx on MNIST (non-IID shard partition)."
    )
    parser.add_argument("--rounds",  type=int,   default=None,
                        help="Override learning.global_rounds")
    parser.add_argument("--mu",      type=float, nargs="+", default=[0.01, 0.1],
                        help="FedProx μ values to compare (space-separated). Default: 0.01 0.1")
    parser.add_argument("--clients", type=int,   default=None,
                        help="Override learning.clients_per_round")
    parser.add_argument("--epochs",  type=int,   default=None,
                        help="Override learning.local_epochs")
    args = parser.parse_args()

    overrides = {}
    if args.rounds  is not None: overrides["learning.global_rounds"]     = args.rounds
    if args.clients is not None: overrides["learning.clients_per_round"] = args.clients
    if args.epochs  is not None: overrides["learning.local_epochs"]      = args.epochs

    # Build algorithm dict: FedAvg + one entry per μ value
    algorithms = {"FedAvg": FedAvg()}
    for mu in args.mu:
        algorithms[f"FedProx(μ={mu})"] = FedProx(mu=mu)

    exp = AlgorithmComparison(
        base_config      = BASE_CONFIG,
        output_dir       = OUTPUT_DIR,
        algorithms       = algorithms,
        config_overrides = overrides,
    )
    results = exp.run()

    # 1. Accuracy vs communication round
    exp.plot_comparison(
        results,
        plot_configs = [{"metric": "test_accuracy",
                         "x":      "round",
                         "ylabel": "Test Accuracy"}],
        out_prefix   = "comparison_acc_vs_round",
    )

    # 2. Accuracy vs simulated time
    exp.plot_comparison(
        results,
        plot_configs = [{"metric": "test_accuracy",
                         "x":      "simulated_time_s",
                         "ylabel": "Test Accuracy"}],
        out_prefix   = "comparison_acc_vs_time",
    )

    # 3. Cumulative energy
    plot_cumulative_energy(results, OUTPUT_DIR)

    # 4. Final accuracy bar chart
    exp.plot_bar(
        results,
        metric   = "final_accuracy",
        ylabel   = "Final Test Accuracy",
        out_name = "bar_final_accuracy",
    )

    print("\nAll results saved to:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()