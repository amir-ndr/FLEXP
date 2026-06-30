"""
examples/fedavg_experiment.py: Canonical FedAvg experiment.

Runs FedAvg on MNIST with the shard partition (non-IID) as configured
in configs/mnist_fedavg.yaml, saves all CSV files and plots to:

    outputs/fedAVG/

Run from the FLEXP/ root:
    python examples/fedavg_experiment.py
    python examples/fedavg_experiment.py --rounds 50
    python examples/fedavg_experiment.py --rounds 50 --clients_per_round 5
"""

import argparse
import os
import sys

# Make sure flsim is importable when run from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flsim.algorithms.fedavg import FedAvg
from flsim.experiments import Experiment


# ---------------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------------

class FedAvgExperiment(Experiment):
    """
    Single FedAvg run.

    Reads all hyperparameters from configs/mnist_fedavg.yaml.
    Any value can be overridden via config_overrides without editing the YAML.
    """

    def __init__(self, config_overrides: dict = None):
        super().__init__(
            base_config = os.path.join(
                os.path.dirname(__file__), "..", "flsim", "configs", "mnist_fedavg.yaml"
            ),
            output_dir = os.path.join(
                os.path.dirname(__file__), "..", "outputs", "fedAVG"
            ),
        )
        self.config_overrides = config_overrides or {}

    def run(self):
        # Single run — inject FedAvg explicitly so the algorithm is
        # visible in this file even though the YAML already says "fedavg".
        result = self.run_single(
            run_name         = "fedavg",
            label            = "FedAvg",
            config_overrides = self.config_overrides,
            components       = {"algorithm": FedAvg()},
        )

        # Per-run plots (driven by the plots: list in base.yaml)
        self.plot_single(result)

        # Print summary
        print("\n" + "=" * 50)
        print("FedAvg experiment complete")
        print(f"  Final accuracy : {result.final_accuracy:.4f}")
        print(f"  Best accuracy  : {result.best_accuracy:.4f}")
        print(f"  Final loss     : {result.final_loss:.4f}")
        print(f"  Total energy   : {result.total_energy_j:.4e} J")
        print(f"  Simulated time : {result.total_simulated_time_s:.1f} s")
        print(f"  CSV saved to   : {result.csv_path}")
        print("=" * 50 + "\n")

        return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run FedAvg on MNIST and save results to outputs/fedAVG/"
    )
    parser.add_argument("--rounds",           type=int,   default=None,
                        help="Override learning.global_rounds")
    parser.add_argument("--clients_per_round",type=int,   default=None,
                        help="Override learning.clients_per_round")
    parser.add_argument("--local_epochs",     type=int,   default=None,
                        help="Override learning.local_epochs")
    parser.add_argument("--lr",               type=float, default=None,
                        help="Override learning.learning_rate")
    parser.add_argument("--seed",             type=int,   default=None,
                        help="Override experiment.seed")
    args = parser.parse_args()

    # Build override dict from CLI args (skip None values)
    overrides = {}
    if args.rounds            is not None: overrides["learning.global_rounds"]        = args.rounds
    if args.clients_per_round is not None: overrides["learning.clients_per_round"]    = args.clients_per_round
    if args.local_epochs      is not None: overrides["learning.local_epochs"]         = args.local_epochs
    if args.lr                is not None: overrides["learning.learning_rate"]        = args.lr
    if args.seed              is not None: overrides["experiment.seed"]               = args.seed

    exp = FedAvgExperiment(config_overrides=overrides)
    exp.run()


if __name__ == "__main__":
    main()
