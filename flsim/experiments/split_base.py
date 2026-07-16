"""
experiments/split_base.py: Experiment base class for split-learning runs
(SL / SFLV1 / SFLV2 — see flsim/core/split_simulator.py).

Mirrors flsim/experiments/async_base.py's pattern (a focused Experiment
subclass adding one build_*_run() + one run_single_*() pair) rather than
extending the sync Experiment directly, since split learning shares almost
none of the sync Simulator's wiring (no ChannelModel/TimeModel/EnergyModel/
ResourceAllocator — see split_simulator.py's scope note).

Produces a standard RunResult (same class used by every other experiment in
this framework), so split-learning runs plug directly into the existing
plot_comparison() / plot_bar() machinery — no separate plotting code needed.

Quick-start
-----------
    from flsim.experiments.split_base import SplitExperiment

    class MyExp(SplitExperiment):
        def run(self):
            r_sl    = self.run_single_split("sl",    label="SL",    client_mode="sequential",      server_mode="sequential")
            r_sflv1 = self.run_single_split("sflv1", label="SFLV1", client_mode="parallel_fedavg",  server_mode="parallel_fedavg")
            r_sflv2 = self.run_single_split("sflv2", label="SFLV2", client_mode="parallel_fedavg",  server_mode="sequential")
            self.plot_comparison({"SL": r_sl, "SFLV1": r_sflv1, "SFLV2": r_sflv2})

    MyExp(base_config="flsim/configs/mnist_fedavg.yaml",
          output_dir="outputs/my_split_exp/").run()
"""

import csv
import os

import numpy as np
import pandas as pd
import torch

from flsim.channel.conversions import dbm_to_watts
from flsim.core.evaluator import Evaluator
from flsim.core.split_client import SplitClient
from flsim.core.split_simulator import SplitSimulator
from flsim.experiments.base import Experiment, RunResult, _apply_config_overrides, _print_header
from flsim.experiments.wiring import (
    _make_channel_model,
    _make_partitioner,
    _make_profiles,
    _model_name_for_dataset,
    load_config,
    set_seeds,
)
from flsim.models.factory import create_model
from flsim.system.split_model import split_model, num_layers
from flsim.system.split_cost import SplitCostModel
from flsim.data.loaders.mnist import load_mnist
from flsim.data.loaders.cifar10 import load_cifar10


_SPLIT_CSV_COLUMNS = [
    "round", "train_loss", "test_loss", "test_accuracy", "num_clients",
    # system-cost columns (same physical base as the sync/async/OTA CSVs, so
    # these are directly comparable): simulated_time_s / cumulative_energy_j
    # share the exact names those CSVs use.
    "round_latency_s", "simulated_time_s",
    "traffic_bytes", "cumulative_traffic_bytes",
    "total_energy_j", "cumulative_energy_j",
]


class SplitExperiment(Experiment):
    """
    Experiment subclass for split learning. Inherits plot_comparison(),
    plot_bar(), etc. from Experiment; adds build_split_run() / run_single_split().
    """

    def build_split_run(
        self,
        run_name: str,
        config_path: str = None,
        config_overrides: dict = None,
        cut_layer: int = None,
        client_mode: str = "parallel_fedavg",
        server_mode: str = "parallel_fedavg",
    ) -> tuple:
        """
        Wire a fully configured SplitSimulator.

        Args:
            run_name (str): unique slug for output directory / CSV name.
            config_path (str): YAML to load (defaults to self.base_config).
            config_overrides (dict): dot-notation patches, e.g.
                {"learning.global_rounds": 200, "learning.cut_layer": 4}.
            cut_layer (int): overrides learning.cut_layer from config if given
                (so you can sweep cut points without editing YAML/overrides).
            client_mode, server_mode (str): "sequential" | "parallel_fedavg"
                — see split_simulator.py's module docstring. Defaults give
                SFLV1; pass ("sequential","sequential") for SL or
                ("parallel_fedavg","sequential") for SFLV2.

        Returns:
            (SplitSimulator, config, csv_path)
        """
        config_overrides = config_overrides or {}
        config_path      = config_path or self.base_config

        config = load_config(config_path)
        config = _apply_config_overrides(config, config_overrides)

        set_seeds(config.experiment.seed)
        rng    = np.random.RandomState(config.experiment.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- dataset ----
        if config.data.dataset == "mnist":
            train_ds, test_ds = load_mnist()
        elif config.data.dataset == "cifar10":
            train_ds, test_ds = load_cifar10()
        else:
            raise ValueError(f"Unknown dataset: {config.data.dataset}")

        partitioner    = _make_partitioner(config.data)
        client_indices = partitioner.partition(train_ds, config.data.num_clients, rng)

        # ---- split the model at cut_layer ----
        model_name = _model_name_for_dataset(config.data.dataset)
        full_model = create_model(model_name)
        n_layers   = num_layers(full_model)
        resolved_cut = cut_layer if cut_layer is not None else getattr(config.learning, "cut_layer", None)
        if resolved_cut is None:
            raise ValueError(
                "cut_layer not set — pass cut_layer=... or set learning.cut_layer in config."
            )
        if not (1 <= resolved_cut <= n_layers - 1):
            raise ValueError(
                f"cut_layer={resolved_cut} invalid for {model_name} ({n_layers} layers); "
                f"must be in [1, {n_layers - 1}]."
            )
        client_model, server_model = split_model(full_model, cut_layer=resolved_cut)

        # ---- clients ----
        clients = [
            SplitClient(client_id=k, dataset=train_ds, indices=client_indices[k])
            for k in range(config.data.num_clients)
        ]

        # ---- system-cost model (same physical base as sync/async/OTA) ----
        # Reuses the wireless channel model, DVFS kappa, and per-device profiles;
        # adds only the edge-server frequency (split.server_cpu_frequency_hz).
        noise_psd = dbm_to_watts(config.wireless.noise_psd_dbm_per_hz)
        config._noise_psd_w_per_hz = noise_psd
        channel_model = _make_channel_model(config, noise_psd)
        profiles = _make_profiles(config, [len(i) for i in client_indices], rng)
        server_freq = float(getattr(getattr(config, "split", None), "server_cpu_frequency_hz", 3.0e9))
        cost_model = SplitCostModel(
            channel_model=channel_model,
            noise_psd_w_per_hz=noise_psd,
            kappa=config.system.switched_capacitance,
            server_cpu_frequency_hz=server_freq,
            downlink_negligible=bool(getattr(config.wireless, "downlink_negligible", False)),
        )

        evaluator = Evaluator(test_dataset=test_ds)

        simulator = SplitSimulator(
            clients=clients,
            client_model=client_model,
            server_model=server_model,
            evaluator=evaluator,
            config=config,
            rng=rng,
            device=device,
            client_mode=client_mode,
            server_mode=server_mode,
            cost_model=cost_model,
            profiles=profiles,
        )

        run_dir = os.path.join(self.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        csv_path = os.path.join(run_dir, f"{run_name}.csv")

        return simulator, config, csv_path

    def run_single_split(
        self,
        run_name: str,
        label: str = None,
        config_path: str = None,
        config_overrides: dict = None,
        cut_layer: int = None,
        client_mode: str = "parallel_fedavg",
        server_mode: str = "parallel_fedavg",
    ) -> RunResult:
        """
        Build, execute, and return results for one split-learning run.

        The CSV is written to <output_dir>/<run_name>/<run_name>.csv. It reuses
        the sync/async column names (round, test_accuracy, test_loss,
        simulated_time_s, cumulative_energy_j) plus split-specific cost columns
        (round_latency_s, traffic_bytes, cumulative_traffic_bytes), so this
        RunResult works with plot_comparison()/plot_bar() unmodified.
        """
        label = label or run_name
        header_overrides = {**(config_overrides or {}), "client_mode": client_mode, "server_mode": server_mode}
        _print_header(label, header_overrides, None)

        sim, config, csv_path = self.build_split_run(
            run_name=run_name,
            config_path=config_path,
            config_overrides=config_overrides,
            cut_layer=cut_layer,
            client_mode=client_mode,
            server_mode=server_mode,
        )
        history = sim.run()

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_SPLIT_CSV_COLUMNS)
            writer.writeheader()
            for r in history:
                writer.writerow({
                    "round":         r.global_epoch,
                    "train_loss":    f"{r.train_loss:.6f}",
                    "test_loss":     f"{r.test_loss:.6f}" if r.test_loss is not None else "",
                    "test_accuracy": f"{r.test_accuracy:.6f}" if r.test_accuracy is not None else "",
                    "num_clients":   r.num_clients,
                    "round_latency_s":          f"{r.round_latency_s:.6f}",
                    "simulated_time_s":         f"{r.simulated_time_s:.6f}",
                    "traffic_bytes":            f"{r.traffic_bytes:.1f}",
                    "cumulative_traffic_bytes": f"{r.cumulative_traffic_bytes:.1f}",
                    "total_energy_j":           f"{r.total_energy_j:.6e}",
                    "cumulative_energy_j":      f"{r.cumulative_energy_j:.6e}",
                })

        df = pd.read_csv(csv_path)
        result = RunResult(name=run_name, label=label, config=config, csv_path=csv_path, df=df)
        print(
            f"[SplitExperiment] Done — "
            f"final_acc={result.final_accuracy:.4f}  best_acc={result.best_accuracy:.4f}\n"
        )
        return result
