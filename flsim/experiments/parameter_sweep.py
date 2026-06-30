"""
experiments/parameter_sweep.py: Single-parameter sensitivity study.

Varies one config parameter across a list of values, runs the same
algorithm for each, and plots how each metric changes with the parameter.

Usage
-----
    from flsim.experiments import ParameterSweep

    exp = ParameterSweep(
        base_config  = "flsim/configs/mnist_fedavg.yaml",
        output_dir   = "outputs/bw_sweep/",
        param        = "wireless.total_bandwidth_hz",
        values       = [5e6, 10e6, 20e6, 40e6],
        labels       = ["5 MHz", "10 MHz", "20 MHz", "40 MHz"],
        param_label  = "Total Bandwidth",
    )
    exp.run()

Each value run writes:
    outputs/bw_sweep/<slug>/   ← per-run CSV + per-run plots
Sweep line plots are written to:
    outputs/bw_sweep/sweep_<metric>_vs_param.png
"""

from flsim.experiments.base import Experiment, _get_plot_configs


class ParameterSweep(Experiment):
    """
    Sweep one config parameter and plot final metrics vs that parameter.

    Args:
        base_config    (str):  path to base YAML config.
        output_dir     (str):  root output directory.
        param          (str):  dot-notation config key to sweep.
                               e.g. "wireless.total_bandwidth_hz"
        values         (list): values to sweep over.
        labels         (list): human-readable labels for each value
                               (used in legends and filenames).
                               Defaults to str(v) for each v.
        param_label    (str):  x-axis label for sweep plots.
                               Defaults to param name.
        components     (dict): fixed component injections applied to ALL runs.
        config_overrides(dict): fixed config patches applied to ALL runs
                               (on top of the swept value).
    """

    def __init__(
        self,
        base_config: str,
        param: str,
        values: list,
        output_dir: str = "outputs/experiments/",
        labels: list = None,
        param_label: str = None,
        components: dict = None,
        config_overrides: dict = None,
    ):
        super().__init__(base_config=base_config, output_dir=output_dir)
        self.param            = param
        self.values           = values
        self.labels           = labels or [str(v) for v in values]
        self.param_label      = param_label or param
        self.components       = components       or {}
        self.config_overrides = config_overrides or {}

        assert len(self.labels) == len(self.values), \
            "labels and values must have the same length"

    def run(self) -> list:
        """
        Run the sweep and generate plots.

        Returns:
            list[RunResult]: one entry per swept value, same order as self.values.
        """
        results = []
        for value, label in zip(self.values, self.labels):
            slug = f"sweep_{_to_slug(self.param)}_{_to_slug(str(label))}"
            overrides = {**self.config_overrides, self.param: value}
            result = self.run_single(
                run_name         = slug,
                label            = label,
                config_overrides = overrides,
                components       = self.components,
            )
            results.append(result)

        # Dict keyed by label for comparison plots
        results_dict = {r.label: r for r in results}

        # 1. Convergence curve comparison (all values on one axes per metric)
        self.plot_comparison(results_dict, out_prefix="sweep_curves")

        # 2. Sweep line: final/best accuracy vs parameter value
        for metric, ylabel in [
            ("final_accuracy",        "Final Test Accuracy"),
            ("best_accuracy",         "Best Test Accuracy"),
            ("total_energy_j",        "Total Energy (J)"),
            ("total_simulated_time_s","Total Simulated Time (s)"),
        ]:
            self.plot_sweep(
                param_values = self.values,
                results      = results,
                param_label  = self.param_label,
                metric       = metric,
                ylabel       = ylabel,
                out_name     = f"sweep_{metric}_vs_param",
            )

        # 3. Summary table
        print("\n" + "="*60)
        print(f"{'Value':<15} {'Final Acc':>10} {'Best Acc':>10} "
              f"{'Energy (J)':>12} {'Time (s)':>10}")
        print("-"*60)
        for label, r in results_dict.items():
            print(f"{label:<15} {r.final_accuracy:>10.4f} {r.best_accuracy:>10.4f} "
                  f"{r.total_energy_j:>12.4e} {r.total_simulated_time_s:>10.1f}")
        print("="*60 + "\n")

        return results


def _to_slug(s: str) -> str:
    return (s.lower()
             .replace(" ", "_")
             .replace("(", "").replace(")", "")
             .replace("/", "_").replace(".", "p"))
