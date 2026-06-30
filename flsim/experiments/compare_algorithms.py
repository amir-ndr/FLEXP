"""
experiments/compare_algorithms.py: Side-by-side algorithm comparison.

Runs the same config with multiple algorithms and generates comparison plots
for every metric listed in the YAML `plots:` section.

Usage
-----
    from flsim.experiments import AlgorithmComparison
    from flsim.algorithms.fedavg import FedAvg

    exp = AlgorithmComparison(
        base_config = "flsim/configs/mnist_fedavg.yaml",
        output_dir  = "outputs/compare_algs/",
        algorithms  = {
            "FedAvg":        FedAvg(),
            "FedProx(0.1)":  FedProx(mu=0.1),
            "ChannelAware":  ChannelAwareAlg(),
        },
    )
    exp.run()

Each algorithm run writes:
    outputs/compare_algs/<slug>/  ← per-run CSV + per-run plots
Comparison plots are written to:
    outputs/compare_algs/comparison_<metric>_vs_<x>.png
A final-accuracy bar chart is written to:
    outputs/compare_algs/bar_final_accuracy.png
"""

from flsim.experiments.base import Experiment


class AlgorithmComparison(Experiment):
    """
    Run the same experiment with N algorithms; plot them together.

    Args:
        base_config (str): path to base YAML config.
        output_dir  (str): root output directory.
        algorithms  (dict): {label: FederatedAlgorithm instance}
        config_overrides (dict): optional config patches applied to ALL runs.
        extra_components (dict): optional extra component injections (e.g.
            {"allocator": MyAllocator()}) applied to ALL runs. The algorithm
            is always taken from the algorithms dict.
    """

    def __init__(
        self,
        base_config: str,
        algorithms: dict,
        output_dir: str = "outputs/experiments/",
        config_overrides: dict = None,
        extra_components: dict = None,
    ):
        super().__init__(base_config=base_config, output_dir=output_dir)
        self.algorithms       = algorithms        # {label: FederatedAlgorithm}
        self.config_overrides = config_overrides or {}
        self.extra_components = extra_components  or {}

    def run(self) -> dict:
        """
        Run all algorithms and generate comparison plots.

        Returns:
            dict: {label: RunResult} — one entry per algorithm.
        """
        results = {}
        for label, algorithm in self.algorithms.items():
            slug = _to_slug(label)
            components = {**self.extra_components, "algorithm": algorithm}
            result = self.run_single(
                run_name         = slug,
                label            = label,
                config_overrides = self.config_overrides,
                components       = components,
            )
            results[label] = result

        # Comparison plots (driven by YAML plots: list)
        self.plot_comparison(results, out_prefix="comparison")

        # Bar chart of final accuracy
        self.plot_bar(
            results,
            metric   = "final_accuracy",
            ylabel   = "Final Test Accuracy",
            out_name = "bar_final_accuracy",
        )

        # Summary table to stdout
        print("\n" + "="*50)
        print(f"{'Algorithm':<20} {'Final Acc':>10} {'Best Acc':>10} {'Energy (J)':>12}")
        print("-"*50)
        for label, r in results.items():
            print(f"{label:<20} {r.final_accuracy:>10.4f} "
                  f"{r.best_accuracy:>10.4f} {r.total_energy_j:>12.4e}")
        print("="*50 + "\n")

        return results


def _to_slug(label: str) -> str:
    """Convert a human label to a filesystem-safe slug."""
    return (label.lower()
                 .replace(" ", "_")
                 .replace("(", "")
                 .replace(")", "")
                 .replace("=", "")
                 .replace(".", "p"))
