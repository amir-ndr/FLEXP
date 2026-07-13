"""
examples/paper_fedasync_cifar10.py
Replicates Figure 2 of Xie et al. (2019) "Asynchronous Federated Optimization".

All training hyperparameters (dataset, num_clients, batch_size, local_epochs,
learning_rate, …) are read from the YAML config.  Pass --config to change them.
The paper-specific knobs (staleness conditions, alpha values, seeds, total rounds)
are exposed as CLI arguments.

Usage
-----
  # Default: base.yaml, 1 seed, staleness ∈ {4, 16}
  python examples/paper_fedasync_cifar10.py

  # Point at a custom config
  python examples/paper_fedasync_cifar10.py --config flsim/configs/base.yaml

  # Override individual YAML keys on the command line
  python examples/paper_fedasync_cifar10.py --set data.dataset=cifar10 data.num_clients=100

  # Paper-quality: 10 seeds, both staleness panels
  python examples/paper_fedasync_cifar10.py --seeds 10 --staleness 4 16

  # Only the high-staleness panel, more FedAvg rounds
  python examples/paper_fedasync_cifar10.py --staleness 16 --fedavg-rounds 300

Gradient alignment
------------------
  FedAvg:   grads/round  = clients_per_round × ceil(N_client / batch_size) × local_epochs
  FedAsync: grads/epoch  = 1               × ceil(N_client / batch_size) × local_epochs
  --async-epochs defaults to: fedavg_rounds × clients_per_round  (equal total gradients)

Output
------
  <output_dir>/figure2_max_staleness_<K>.png  for each K in --staleness
  <output_dir>/<run_name>/<run_name>.csv      per-run logs
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flsim.algorithms.fedavg import FedAvg
from flsim.algorithms.fedasync import FedAsyncSimulatedStaleness
from flsim.experiments.async_base import AsyncExperiment
from flsim.experiments.wiring import load_config

# ---- known training-set sizes (for gradient counting without loading the data) ----
_DATASET_TRAIN_SIZES = {
    "mnist":   60_000,
    "cifar10": 50_000,
}

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "flsim", "configs", "base.yaml"
)

# ---- algorithm catalogue ----
# Keys appear in the legend; values are factories (max_staleness, seed) → algorithm.
_ALGORITHM_FACTORIES = {
    "FedAsync+Const α=0.6":  lambda ms, s: FedAsyncSimulatedStaleness(ms, seed=s, alpha=0.6, staleness_func="constant"),
    "FedAsync+Const α=0.9":  lambda ms, s: FedAsyncSimulatedStaleness(ms, seed=s, alpha=0.9, staleness_func="constant"),
    "FedAsync+Poly α=0.6":   lambda ms, s: FedAsyncSimulatedStaleness(ms, seed=s, alpha=0.6, staleness_func="polynomial", a=0.5),
    "FedAsync+Poly α=0.9":   lambda ms, s: FedAsyncSimulatedStaleness(ms, seed=s, alpha=0.9, staleness_func="polynomial", a=0.5),
    "FedAsync+Hinge α=0.6":  lambda ms, s: FedAsyncSimulatedStaleness(ms, seed=s, alpha=0.6, staleness_func="hinge", a=10.0, b=4.0),
    "FedAsync+Hinge α=0.9":  lambda ms, s: FedAsyncSimulatedStaleness(ms, seed=s, alpha=0.9, staleness_func="hinge", a=10.0, b=4.0),
}

_PLOT_STYLES = {
    "FedAvg":               dict(color="black",      ls="-",   lw=2.0),
    "FedAsync+Const α=0.6": dict(color="tab:blue",   ls="--",  lw=1.8),
    "FedAsync+Const α=0.9": dict(color="tab:cyan",   ls="--",  lw=1.8),
    "FedAsync+Poly α=0.6":  dict(color="tab:orange", ls="-.",  lw=1.8),
    "FedAsync+Poly α=0.9":  dict(color="tab:red",    ls="-.",  lw=1.8),
    "FedAsync+Hinge α=0.6": dict(color="tab:green",  ls=":",   lw=1.8),
    "FedAsync+Hinge α=0.9": dict(color="tab:purple", ls=":",   lw=1.8),
}


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------

class PaperFedAsyncExperiment(AsyncExperiment):
    """
    Replicates Xie et al. (2019) Figure 2: test accuracy vs. # of gradients.

    All dataset / model / system settings come from the YAML config.
    Paper-specific knobs are passed to run().
    """

    def run(
        self,
        config_overrides: dict = None,
        fedavg_rounds: int = None,
        async_epochs: int = None,
        max_staleness_values: list = None,
        num_seeds: int = 1,
    ) -> None:
        """
        Args:
            config_overrides:     extra YAML overrides applied to every run
                                  (e.g. {"data.dataset": "cifar10"}).
            fedavg_rounds:        number of FedAvg rounds (default: from YAML global_rounds).
            async_epochs:         number of async server updates per seed
                                  (default: fedavg_rounds × clients_per_round).
            max_staleness_values: list of max-staleness K values to run
                                  (default [4, 16] → both Figure 2 panels).
            num_seeds:            seeds 0 … num_seeds-1 are averaged (default 1).
        """
        config_overrides     = config_overrides or {}
        max_staleness_values = max_staleness_values or [4, 16]

        # ---- derive round counts from (possibly overridden) config ----
        base_cfg = load_config(self.base_config)
        # apply top-level overrides so we see the right values
        from flsim.experiments.base import _apply_config_overrides
        merged_cfg = _apply_config_overrides(base_cfg, config_overrides)

        k        = int(merged_cfg.learning.clients_per_round)
        _dataset = merged_cfg.data.dataset
        _n_cl    = int(merged_cfg.data.num_clients)
        _bsz     = int(merged_cfg.learning.batch_size)
        _le      = int(merged_cfg.learning.local_epochs)

        total_train = _DATASET_TRAIN_SIZES.get(_dataset, None)
        if total_train is None:
            raise ValueError(
                f"Unknown dataset '{_dataset}'. "
                f"Known: {list(_DATASET_TRAIN_SIZES)}. "
                f"Add it to _DATASET_TRAIN_SIZES in this file."
            )
        samples_per_client = total_train // _n_cl
        batches_per_client = math.ceil(samples_per_client / _bsz)

        grads_per_fedavg_round = k       * batches_per_client * _le
        grads_per_async_epoch  = 1       * batches_per_client * _le

        if fedavg_rounds is None:
            fedavg_rounds = int(merged_cfg.learning.global_rounds)
        if async_epochs is None:
            # Match total gradients: fedavg_rounds × k clients = async_epochs × 1 client
            async_epochs = fedavg_rounds * k

        total_grads = fedavg_rounds * grads_per_fedavg_round

        # FedAvg: evaluate every 10% of rounds (at least 1)
        fedavg_eval_every = max(1, fedavg_rounds // 10)
        # FedAsync: evaluate at the same gradient checkpoints
        async_eval_every  = max(1, async_epochs // 10)

        print(f"\n{'='*70}")
        print(f"  Paper FedAsync CIFAR-10 experiment")
        print(f"  Dataset: {_dataset}, clients: {_n_cl}, "
              f"batch_size: {_bsz}, local_epochs: {_le}")
        print(f"  FedAvg: k={k}, rounds={fedavg_rounds}, "
              f"grads/round={grads_per_fedavg_round}")
        print(f"  FedAsync: epochs={async_epochs}, "
              f"grads/epoch={grads_per_async_epoch}")
        print(f"  Target total gradients: {total_grads}")
        print(f"  Seeds: {num_seeds},  Staleness conditions: {max_staleness_values}")
        print(f"{'='*70}\n")

        for max_staleness in max_staleness_values:
            print(f"\n{'─'*60}")
            print(f"  CONDITION: max_staleness = {max_staleness}")
            print(f"{'─'*60}")

            seed_dfs: dict = {}
            for seed in range(num_seeds):
                seed_dfs[seed] = self._run_one_seed(
                    seed=seed,
                    max_staleness=max_staleness,
                    fedavg_rounds=fedavg_rounds,
                    async_epochs=async_epochs,
                    fedavg_eval_every=fedavg_eval_every,
                    async_eval_every=async_eval_every,
                    grads_per_fedavg_round=grads_per_fedavg_round,
                    grads_per_async_epoch=grads_per_async_epoch,
                    config_overrides=config_overrides,
                )

            avg = _average_over_seeds(seed_dfs)
            out = os.path.join(
                self.output_dir, f"figure2_max_staleness_{max_staleness}.png"
            )
            _plot_accuracy_vs_gradients(avg, max_staleness, total_grads, out)

        print(f"\n[PaperFedAsyncExperiment] All done. Plots in {self.output_dir}")

    def _run_one_seed(
        self,
        seed: int,
        max_staleness: int,
        fedavg_rounds: int,
        async_epochs: int,
        fedavg_eval_every: int,
        async_eval_every: int,
        grads_per_fedavg_round: int,
        grads_per_async_epoch: int,
        config_overrides: dict,
    ) -> dict:
        """Run all algorithms for one (max_staleness, seed) pair."""
        results = {}

        # ---- shared overrides per run type ----
        fedavg_overrides = {
            **config_overrides,
            "learning.global_rounds":     fedavg_rounds,
            "evaluation.evaluate_every":  fedavg_eval_every,
            "experiment.seed":            seed,
        }
        async_overrides = {
            **config_overrides,
            "learning.global_rounds":     async_epochs,
            "learning.clients_per_round": 1,   # window=1 for simulated staleness
            "async_fl.window_size":       1,
            "evaluation.evaluate_every":  async_eval_every,
            "experiment.seed":            seed,
        }

        # ---- FedAvg baseline ----
        r = self.run_single(
            run_name=f"fedavg_ms{max_staleness}_s{seed}",
            label="FedAvg",
            config_overrides=fedavg_overrides,
            components={"algorithm": FedAvg()},
        )
        df = _add_gradient_column(r.df, grads_per_fedavg_round)
        results["FedAvg"] = df

        # ---- FedAsync variants ----
        for label, factory in _ALGORITHM_FACTORIES.items():
            alg  = factory(max_staleness, seed * 1000 + max_staleness)
            slug = _slugify(label)
            r = self.run_single_async(
                run_name=f"async_{slug}_ms{max_staleness}_s{seed}",
                label=label,
                config_overrides=async_overrides,
                components={"algorithm": alg},
            )
            df = _add_gradient_column(r.df, grads_per_async_epoch)
            results[label] = df

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
print('hi')

def _add_gradient_column(df: pd.DataFrame, grads_per_step: int) -> pd.DataFrame:
    df = df.copy()
    df["cumulative_gradients"] = (df["round"] + 1) * grads_per_step
    return df


def _average_over_seeds(seed_dfs: dict) -> dict:
    """Average test_accuracy at common gradient checkpoints across all seeds."""
    labels = list(next(iter(seed_dfs.values())).keys())
    avg = {}
    for label in labels:
        frames = []
        for label_dict in seed_dfs.values():
            df = label_dict[label].dropna(subset=["test_accuracy"])
            frames.append(df[["cumulative_gradients", "test_accuracy"]])
        combined = pd.concat(frames, ignore_index=True)
        avg[label] = (
            combined.groupby("cumulative_gradients")["test_accuracy"]
            .mean()
            .reset_index()
        )
    return avg


def _plot_accuracy_vs_gradients(
    avg_results: dict, max_staleness: int, total_grads: int, out_path: str
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, df in avg_results.items():
        style = _PLOT_STYLES.get(label, dict(color="gray", ls="-", lw=1.5))
        ax.plot(df["cumulative_gradients"], df["test_accuracy"] * 100.0,
                label=label, **style)
    ax.set_xlabel("Number of gradients", fontsize=12)
    ax.set_ylabel("Top-1 test accuracy (%)", fontsize=12)
    ax.set_title(
        f"FedAsync vs FedAvg — staleness t−τ ~ Uniform{{0,…,{max_staleness}}}",
        fontsize=13,
    )
    ax.set_xlim(0, total_grads)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


def _slugify(label: str) -> str:
    return (label.lower()
            .replace(" ", "_").replace("=", "").replace(".", "")
            .replace("+", "p").replace("α", "a"))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Replicate Xie et al. (2019) Figure 2 (FedAsync vs FedAvg)"
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help="Path to base YAML config (default: flsim/configs/base.yaml). "
             "Dataset, clients, model, system params are all read from here."
    )
    parser.add_argument(
        "--seeds", type=int, default=1, metavar="N",
        help="Number of random seeds to average over (default 1; paper uses 10)."
    )
    parser.add_argument(
        "--staleness", type=int, nargs="+", default=[4, 16], metavar="K",
        help="Max-staleness condition(s), e.g. --staleness 4 16 (default)."
    )
    parser.add_argument(
        "--fedavg-rounds", type=int, default=None,
        help="FedAvg global rounds (default: global_rounds from YAML)."
    )
    parser.add_argument(
        "--async-epochs", type=int, default=None,
        help="FedAsync server-update count "
             "(default: fedavg_rounds × clients_per_round, matching total gradients)."
    )
    parser.add_argument(
        "--output-dir", default="outputs/paper_fedasync/",
        help="Directory for plots and CSVs (default: outputs/paper_fedasync/)."
    )
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override any YAML key in dot notation, e.g.: "
             "--set data.dataset=cifar10 data.num_clients=100 learning.batch_size=50 "
             "learning.learning_rate=0.1"
    )
    return parser.parse_args()


def _parse_set_overrides(set_args: list) -> dict:
    """Convert ["a.b=v", "x=y"] to {"a.b": "v", "x": "y"} with type coercion."""
    overrides = {}
    for item in set_args:
        if "=" not in item:
            raise ValueError(f"--set entries must be KEY=VALUE, got: {item!r}")
        key, val = item.split("=", 1)
        # coerce to int / float if possible, else keep as string
        for cast in (int, float):
            try:
                val = cast(val)
                break
            except ValueError:
                pass
        overrides[key] = val
    return overrides


if __name__ == "__main__":
    args = _parse_args()
    overrides = _parse_set_overrides(args.set)

    PaperFedAsyncExperiment(
        base_config=args.config,
        output_dir=args.output_dir,
    ).run(
        config_overrides=overrides,
        fedavg_rounds=args.fedavg_rounds,
        async_epochs=args.async_epochs,
        max_staleness_values=args.staleness,
        num_seeds=args.seeds,
    )
