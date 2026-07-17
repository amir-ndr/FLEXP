"""
experiments/base.py: Base experiment class and RunResult container.

Design goals
------------
1. Minimal-override: subclass Experiment, implement run(), done.
2. Config overrides via dot-notation dict — no need to write a new YAML per variant.
3. Component injection — swap any object (algorithm, allocator, channel model, …).
4. Each run writes its own CSV; comparison plots read those CSVs afterward.
5. Which plots to generate is controlled by the YAML `plots:` list.

Quick-start
-----------
    class MyExp(Experiment):
        def run(self):
            r1 = self.run_single("fedavg", components={"algorithm": FedAvg()})
            r2 = self.run_single("fedprox", components={"algorithm": FedProx(mu=0.1)})
            self.plot_comparison({"FedAvg": r1, "FedProx": r2})

    MyExp(base_config="configs/mnist_fedavg.yaml",
          output_dir="outputs/my_exp/").run()
"""

import copy
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Cross-layer imports — experiments/ is an entry-point module like run.py
from flsim.channel.conversions import dbm_to_watts
from flsim.core.client import Client
from flsim.core.evaluator import Evaluator
from flsim.core.logger import Logger
from flsim.core.server import Server
from flsim.core.simulator import Simulator
from flsim.models.factory import create_model
from flsim.experiments.wiring import (
    _load_dataset,
    _make_algorithm,
    _make_allocator,
    _make_channel_model,
    _make_partitioner,
    _make_profiles,
    _model_name_for_dataset,
    _num_classes_for_dataset,
    load_config,
    set_seeds,
)
from flsim.system.cellular_time import CellularTimeModel
from flsim.system.energy import EnergyModel


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """
    Output of a single experiment run.

    Attributes:
        name     : filesystem-safe slug (used for filenames).
        label    : human-readable label for plot legends.
        config   : the merged SimpleNamespace config that produced this run.
        csv_path : absolute path to the per-run CSV file.
        df       : CSV loaded as a DataFrame.
    """
    name:     str
    label:    str
    config:   object          # SimpleNamespace
    csv_path: str
    df:       pd.DataFrame

    # ------------------------------------------------------------------
    # Convenience properties — avoid digging into df manually
    # ------------------------------------------------------------------

    @property
    def final_accuracy(self) -> float:
        """Test accuracy at the last evaluated round."""
        edf = self.df.dropna(subset=["test_accuracy"])
        return float(edf["test_accuracy"].iloc[-1]) if not edf.empty else float("nan")

    @property
    def best_accuracy(self) -> float:
        """Best test accuracy across all evaluated rounds."""
        edf = self.df.dropna(subset=["test_accuracy"])
        return float(edf["test_accuracy"].max()) if not edf.empty else float("nan")

    @property
    def final_loss(self) -> float:
        edf = self.df.dropna(subset=["test_loss"])
        return float(edf["test_loss"].iloc[-1]) if not edf.empty else float("nan")

    @property
    def total_energy_j(self) -> float:
        """Sum of total_energy_j across all rounds."""
        return float(self.df["total_energy_j"].sum())

    @property
    def total_simulated_time_s(self) -> float:
        """Cumulative simulated time at the end of training."""
        return float(self.df["simulated_time_s"].iloc[-1])

    @property
    def avg_staleness(self) -> float:
        """
        Mean staleness (t - tau) across all epochs.

        0.0 for synchronous runs (no `staleness` column — clients always train
        on the same model, so staleness is zero by construction).
        """
        if "staleness" not in self.df.columns:
            return 0.0
        return float(self.df["staleness"].mean())

    def metric(self, col: str) -> pd.Series:
        """Return a column from the CSV as a pandas Series."""
        return self.df[col]

    def __repr__(self):
        return (f"RunResult(label={self.label!r}, "
                f"final_acc={self.final_accuracy:.4f}, "
                f"best_acc={self.best_accuracy:.4f})")


# ---------------------------------------------------------------------------
# Experiment base
# ---------------------------------------------------------------------------

class Experiment:
    """
    Base class for all experiments.

    Subclass this and implement run(). Use run_single() to execute runs
    with config overrides and component injections. Use plot_comparison()
    to compare multiple RunResults. Use plot_single() for per-run plots.

    Args:
        base_config (str): path to base YAML config.
        output_dir  (str): root directory for all output files.
    """

    def __init__(self, base_config: str, output_dir: str = "outputs/experiments/"):
        self.base_config = base_config
        self.output_dir  = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # build_run — mirror of build_experiment() with override support
    # ------------------------------------------------------------------

    def build_run(
        self,
        run_name: str,
        config_path: str = None,
        config_overrides: dict = None,
        components: dict = None,
    ) -> tuple:
        """
        Wire a fully configured Simulator.

        Args:
            run_name (str): unique slug — used for the output subdirectory
                and CSV filename.
            config_path (str): YAML to load. Defaults to self.base_config.
            config_overrides (dict): dot-notation patches applied to the config
                before any component is built. Examples:
                    {"learning.global_rounds": 50}
                    {"wireless.total_bandwidth_hz": 1e7,
                     "learning.local_epochs": 3}
            components (dict): pre-built objects that bypass the factory.
                Valid keys (all optional):
                    "algorithm"     — FederatedAlgorithm instance
                    "allocator"     — ResourceAllocator instance
                    "channel_model" — ChannelModel instance
                    "time_model"    — TimeModel instance
                    "energy_model"  — EnergyModel instance

        Returns:
            (Simulator, SimpleNamespace): ready-to-run simulator + config used.
        """
        config_overrides = config_overrides or {}
        components       = components       or {}
        config_path      = config_path or self.base_config

        config = load_config(config_path)
        config = _apply_config_overrides(config, config_overrides)

        set_seeds(config.experiment.seed)
        rng    = np.random.RandomState(config.experiment.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset
        train_ds, test_ds = _load_dataset(config)

        partitioner    = _make_partitioner(config.data)
        client_indices = partitioner.partition(train_ds, config.data.num_clients, rng)

        global_model = create_model(
            _model_name_for_dataset(config.data.dataset, getattr(config.data, "model_name", None)),
            num_classes=_num_classes_for_dataset(config.data.dataset, getattr(config.data, "num_classes", None)),
        ).to(device)

        # Pre-converted scalars
        noise_psd = dbm_to_watts(config.wireless.noise_psd_dbm_per_hz)
        config._noise_psd_w_per_hz = noise_psd
        config._p_max_w            = dbm_to_watts(config.wireless.tx_power_dbm)
        config._f_max_hz           = getattr(config.system, "cpu_frequency_hz", 2.0e9)

        channel_model = components.get("channel_model",
                                       _make_channel_model(config, noise_psd))
        allocator     = components.get("allocator", _make_allocator(config))

        profiles = _make_profiles(config, [len(i) for i in client_indices], rng)

        time_model   = components.get("time_model",
                                      CellularTimeModel(channel_model, noise_psd))
        energy_model = components.get("energy_model",
                                      EnergyModel(config.system.switched_capacitance))

        clients = [
            Client(client_id=k, dataset=train_ds,
                   indices=client_indices[k], profile=profiles[k])
            for k in range(config.data.num_clients)
        ]

        algorithm = components.get("algorithm",
                                   _make_algorithm(config.learning.algorithm))
        server    = Server(model=global_model, algorithm=algorithm)

        run_dir   = os.path.join(self.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        evaluator = Evaluator(test_dataset=test_ds)
        logger    = Logger(output_dir=run_dir, experiment_name=run_name)

        simulator = Simulator(
            server=server, clients=clients,
            time_model=time_model, channel_model=channel_model,
            allocator=allocator, energy_model=energy_model,
            evaluator=evaluator, logger=logger,
            config=config, rng=rng, device=device,
        )

        return simulator, config

    # ------------------------------------------------------------------
    # run_single
    # ------------------------------------------------------------------

    def run_single(
        self,
        run_name: str,
        label: str = None,
        config_path: str = None,
        config_overrides: dict = None,
        components: dict = None,
    ) -> RunResult:
        """
        Build, execute, and return results for one run.

        The CSV is written to:
            <output_dir>/<run_name>/<run_name>.csv

        Args:
            run_name (str): unique slug for directories/filenames.
            label (str): legend label in plots (defaults to run_name).
            config_path (str): YAML override.
            config_overrides (dict): config patches.
            components (dict): component injections.

        Returns:
            RunResult: loaded CSV + convenience properties.
        """
        label = label or run_name
        _print_header(label, config_overrides, components)

        sim, config = self.build_run(
            run_name=run_name,
            config_path=config_path,
            config_overrides=config_overrides,
            components=components,
        )
        sim.run()

        csv_path = os.path.join(self.output_dir, run_name, f"{run_name}.csv")
        df       = pd.read_csv(csv_path)
        result   = RunResult(name=run_name, label=label,
                             config=config, csv_path=csv_path, df=df)
        print(f"[Experiment] Done — final_acc={result.final_accuracy:.4f}  "
              f"best_acc={result.best_accuracy:.4f}\n")
        return result

    # ------------------------------------------------------------------
    # Entry point — implement this in your subclass
    # ------------------------------------------------------------------

    def run(self):
        """
        Define your experiment here.

        Typical pattern:
            r1 = self.run_single("alg_a", components={"algorithm": AlgA()})
            r2 = self.run_single("alg_b", components={"algorithm": AlgB()})
            self.plot_comparison({"AlgA": r1, "AlgB": r2})
        """
        raise NotImplementedError("Subclass Experiment and implement run()")

    # ------------------------------------------------------------------
    # Plotting — driven by the YAML plots: list
    # ------------------------------------------------------------------

    def plot_single(self, result: RunResult) -> None:
        """
        Generate all plots listed in config.plots for a single run.
        Plots are saved inside the run's own subdirectory.
        """
        plot_cfgs = _get_plot_configs(result.config)
        run_dir   = os.path.join(self.output_dir, result.name)
        for pc in plot_cfgs:
            _plot_one(
                dfs    = {result.label: result.df},
                metric = pc["metric"],
                x      = pc.get("x", "round"),
                ylabel = pc.get("ylabel", pc["metric"]),
                title  = pc.get("title", None),
                log_y  = pc.get("log_scale", False),
                out_path = os.path.join(
                    run_dir,
                    f"{result.name}_{pc['metric']}_vs_{pc.get('x','round')}.png"
                ),
            )

    def plot_comparison(
        self,
        results: Dict[str, RunResult],
        plot_configs: list = None,
        out_prefix: str = "comparison",
    ) -> None:
        """
        Generate comparison plots for multiple runs side by side.

        Which plots to generate is controlled by:
          1. plot_configs argument (explicit list of dicts), OR
          2. the `plots:` section of the first result's config YAML, OR
          3. a sensible default (accuracy + loss + energy).

        Each plot is saved as:
            <output_dir>/<out_prefix>_<metric>_vs_<x>.png

        Args:
            results (dict): {label: RunResult}
            plot_configs (list): optional explicit list, same format as YAML:
                [{"metric": "test_accuracy", "x": "round", "ylabel": "Accuracy"}, ...]
            out_prefix (str): prefix for output filenames.
        """
        if plot_configs is None:
            first = next(iter(results.values()))
            plot_configs = _get_plot_configs(first.config)

        dfs = {label: r.df for label, r in results.items()}
        for pc in plot_configs:
            metric = pc["metric"]
            x      = pc.get("x", "round")
            _plot_one(
                dfs      = dfs,
                metric   = metric,
                x        = x,
                ylabel   = pc.get("ylabel", metric),
                title    = pc.get("title", None),
                log_y    = pc.get("log_scale", False),
                out_path = os.path.join(
                    self.output_dir,
                    f"{out_prefix}_{metric}_vs_{x}.png"
                ),
            )

    def plot_bar(
        self,
        results: Dict[str, RunResult],
        metric: str = "final_accuracy",
        ylabel: str = "Final test accuracy",
        out_name: str = "bar_comparison",
        title: str = None,
    ) -> str:
        """
        Bar chart of any RunResult scalar property across runs.

        metric can be any RunResult property:
            "final_accuracy", "best_accuracy", "final_loss",
            "total_energy_j", "total_simulated_time_s"

        Returns:
            str: path to the saved PNG.
        """
        labels = list(results.keys())
        values = [getattr(r, metric) for r in results.values()]
        fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.4), 4))
        colors = plt.cm.tab10.colors
        bars = ax.bar(labels, values,
                      color=[colors[i % 10] for i in range(len(labels))])
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title or ylabel)
        ax.set_ylim(0, max(values) * 1.15 if values else 1.0)
        ax.grid(True, alpha=0.3, axis="y")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        return self._save(fig, out_name)

    def plot_sweep(
        self,
        param_values: list,
        results: List[RunResult],
        param_label: str,
        metric: str = "final_accuracy",
        ylabel: str = "Final test accuracy",
        out_name: str = "sweep",
        title: str = None,
    ) -> str:
        """
        Line plot of a scalar metric vs a swept parameter.
        Designed for ParameterSweep but usable anywhere.

        Args:
            param_values (list): x-axis tick values (the swept parameter).
            results (list): RunResult list aligned with param_values.
            param_label (str): x-axis label (e.g. "Total bandwidth (Hz)").
            metric (str): RunResult property for the y-axis.
            ylabel (str): y-axis label.
            out_name (str): output filename without extension.

        Returns:
            str: path to saved PNG.
        """
        y = [getattr(r, metric) for r in results]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(param_values, y, marker="o", linewidth=2, markersize=6)
        for xv, yv in zip(param_values, y):
            ax.annotate(f"{yv:.4f}", (xv, yv),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8)
        ax.set_xlabel(param_label)
        ax.set_ylabel(ylabel)
        ax.set_title(title or f"{ylabel} vs {param_label}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return self._save(fig, out_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save(self, fig, out_name: str) -> str:
        path = os.path.join(self.output_dir, f"{out_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Experiment] Saved {path}")
        return path


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _apply_config_overrides(config, overrides: dict):
    """
    Apply dot-notation overrides to a SimpleNamespace config in-place (deep copy).

    Example:
        overrides = {
            "learning.global_rounds": 50,
            "wireless.total_bandwidth_hz": 1.0e+7,
        }
    """
    config = copy.deepcopy(config)
    for key, value in overrides.items():
        parts = key.split(".")
        obj   = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return config


def _get_plot_configs(config) -> list:
    """
    Extract the plots list from the config namespace.
    Falls back to a minimal default if the config has no plots section.
    """
    plots = getattr(config, "plots", None)
    if plots is None:
        return [
            {"metric": "test_accuracy",  "x": "round",           "ylabel": "Test Accuracy"},
            {"metric": "test_accuracy",  "x": "simulated_time_s", "ylabel": "Test Accuracy"},
            {"metric": "test_loss",      "x": "round",           "ylabel": "Test Loss"},
            {"metric": "total_energy_j", "x": "round",           "ylabel": "Energy (J)"},
        ]
    # SimpleNamespace list → plain dicts
    if isinstance(plots, list):
        result = []
        for p in plots:
            if hasattr(p, "__dict__"):
                result.append(vars(p))
            elif isinstance(p, dict):
                result.append(p)
        return result
    return []


def _plot_one(dfs: dict, metric: str, x: str, ylabel: str,
              title: str, log_y: bool, out_path: str) -> None:
    """Plot one metric for multiple runs and save to out_path."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = plt.cm.tab10.colors
    for i, (label, df) in enumerate(dfs.items()):
        sub = df.dropna(subset=[metric]) if metric in ["test_accuracy", "test_loss"] else df
        if metric not in sub.columns:
            continue
        ax.plot(sub[x], sub[metric], marker="o", markersize=3,
                linewidth=1.5, label=label, color=colors[i % 10])
    ax.set_xlabel("Communication round" if x == "round" else "Simulated time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{ylabel} vs {'round' if x == 'round' else 'time'}")
    if log_y:
        ax.set_yscale("log")
    if len(dfs) > 1:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Experiment] Saved {out_path}")


def _print_header(label, config_overrides, components):
    print(f"\n{'='*60}")
    print(f"[Experiment] Run: {label}")
    if config_overrides:
        for k, v in config_overrides.items():
            print(f"  override: {k} = {v}")
    if components:
        for k, v in components.items():
            print(f"  component: {k} = {type(v).__name__}")
    print(f"{'='*60}")
