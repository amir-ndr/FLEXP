"""
experiments/async_base.py: Experiment base class for async FL runs.

Extends the synchronous Experiment with build_async_run() and
run_single_async(), which wire an AsyncSimulator instead of Simulator.

The returned RunResult is compatible with the synchronous RunResult
(same pandas-accessible columns: test_accuracy, simulated_time_s,
total_energy_j) so you can pass both to plot_comparison().

Quick-start
-----------
    from flsim.experiments.async_base import AsyncExperiment
    from flsim.algorithms.fedasync import FedAsync, FedAsyncTopKFastTotal

    class MyAsyncExp(AsyncExperiment):
        def run(self):
            # ---- async variants ----
            r_const = self.run_single_async(
                "fedasync_const",
                components={"algorithm": FedAsync(alpha=0.1)},
            )
            r_poly = self.run_single_async(
                "fedasync_poly",
                components={
                    "algorithm": FedAsync(alpha=0.1,
                                         staleness_func="polynomial", a=0.5)
                },
            )
            r_topk = self.run_single_async(
                "fedasync_topk",
                components={"algorithm": FedAsyncTopKFastTotal(alpha=0.1)},
                config_overrides={"async_fl.window_size": 5},
            )

            # ---- sync baseline ----
            r_sync = self.run_single("fedavg_sync",
                                     components={"algorithm": FedAvg()})

            # ---- compare on accuracy ----
            self.plot_comparison({
                "FedAsync+Const": r_const,
                "FedAsync+Poly":  r_poly,
                "FedAsync+TopK":  r_topk,
                "FedAvg (sync)":  r_sync,
            })

    MyAsyncExp(
        base_config="flsim/configs/mnist_fedavg.yaml",
        output_dir="outputs/async_exp/",
    ).run()

Config additions (async_fl section in YAML or config_overrides)
---------------------------------------------------------------
    async_fl:
      alpha:       0.1   # base mixing hyperparameter α
      window_size: 10    # concurrent clients in flight (default = clients_per_round)
"""

import os

import numpy as np
import pandas as pd
import torch

from flsim.channel.conversions import dbm_to_watts
from flsim.core.async_logger import AsyncLogger
from flsim.core.async_simulator import AsyncSimulator
from flsim.core.client import Client
from flsim.core.evaluator import Evaluator
from flsim.core.server import Server
from flsim.experiments.base import Experiment, RunResult, _apply_config_overrides, _print_header
from flsim.experiments.wiring import (
    _make_algorithm,
    _make_allocator,
    _make_channel_model,
    _make_partitioner,
    _make_profiles,
    _model_name_for_dataset,
    load_config,
    set_seeds,
)
from flsim.models.factory import create_model
from flsim.system.cellular_time import CellularTimeModel
from flsim.system.energy import EnergyModel
from flsim.data.loaders.mnist import load_mnist
from flsim.data.loaders.cifar10 import load_cifar10


class AsyncExperiment(Experiment):
    """
    Experiment subclass that adds async FL support.

    Inherits all synchronous methods (run_single, build_run, plot_comparison …)
    and adds async counterparts (run_single_async, build_async_run).

    You can mix sync and async runs in the same experiment and compare them
    with plot_comparison() — both return RunResult which exposes
    test_accuracy, simulated_time_s, and total_energy_j.
    """

    # ------------------------------------------------------------------
    # build_async_run
    # ------------------------------------------------------------------

    def build_async_run(
        self,
        run_name: str,
        config_path: str = None,
        config_overrides: dict = None,
        components: dict = None,
    ) -> tuple:
        """
        Wire a fully configured AsyncSimulator.

        Mirrors build_run() but creates AsyncSimulator + AsyncLogger.

        Args:
            run_name (str): unique slug for output directory / CSV name.
            config_path (str): YAML to load (defaults to self.base_config).
            config_overrides (dict): dot-notation patches, e.g.:
                {"async_fl.alpha": 0.2, "async_fl.window_size": 5,
                 "learning.global_rounds": 200}
            components (dict): pre-built objects that bypass the factory.
                Valid keys (all optional):
                    "algorithm"     — AsyncFederatedAlgorithm instance  ← required
                    "allocator"     — ResourceAllocator
                    "channel_model" — ChannelModel
                    "time_model"    — TimeModel
                    "energy_model"  — EnergyModel

                If "algorithm" is not provided, the factory falls back to
                the YAML learning.algorithm string (fedavg / fedprox).
                For async runs you almost always want to pass your own
                AsyncFederatedAlgorithm instance here.

        Returns:
            (AsyncSimulator, config): ready-to-run simulator + merged config.
        """
        config_overrides = config_overrides or {}
        components       = components       or {}
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

        global_model = create_model(_model_name_for_dataset(config.data.dataset)).to(device)

        # ---- pre-converted scalars ----
        noise_psd             = dbm_to_watts(config.wireless.noise_psd_dbm_per_hz)
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

        # ---- algorithm — must be AsyncFederatedAlgorithm ----
        algorithm = components.get("algorithm",
                                   _make_algorithm(config.learning.algorithm))
        server    = Server(model=global_model, algorithm=algorithm)

        # ---- async logger (separate from sync Logger) ----
        run_dir   = os.path.join(self.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        evaluator = Evaluator(test_dataset=test_ds)
        logger    = AsyncLogger(output_dir=run_dir, experiment_name=run_name)

        simulator = AsyncSimulator(
            server=server, clients=clients,
            time_model=time_model, channel_model=channel_model,
            allocator=allocator, energy_model=energy_model,
            evaluator=evaluator, logger=logger,
            config=config, rng=rng, device=device,
        )

        return simulator, config

    # ------------------------------------------------------------------
    # run_single_async
    # ------------------------------------------------------------------

    def run_single_async(
        self,
        run_name: str,
        label: str = None,
        config_path: str = None,
        config_overrides: dict = None,
        components: dict = None,
    ) -> RunResult:
        """
        Build, execute, and return results for one async run.

        The CSV is written to:
            <output_dir>/<run_name>/<run_name>.csv

        Returns a RunResult whose .df DataFrame has columns:
            global_epoch, simulated_time_s, test_accuracy, test_loss,
            staleness, alpha_used, client_id, compute_time_s, …

        This RunResult is compatible with plot_comparison() — any shared
        column (test_accuracy, simulated_time_s, total_energy_j) can be
        plotted against a sync RunResult.

        Args:
            run_name (str): unique slug for directories / filenames.
            label (str):    legend label in plots (defaults to run_name).
            config_path (str): YAML override.
            config_overrides (dict): config patches (dot notation).
            components (dict): component injections (see build_async_run).

        Returns:
            RunResult with .df, .final_accuracy, .best_accuracy, etc.
        """
        label = label or run_name
        _print_header(label, config_overrides, components)

        sim, config = self.build_async_run(
            run_name=run_name,
            config_path=config_path,
            config_overrides=config_overrides,
            components=components,
        )
        sim.run()

        csv_path = os.path.join(self.output_dir, run_name, f"{run_name}.csv")
        df       = pd.read_csv(csv_path)

        # RunResult expects a "round" column for some plot helpers — alias it.
        if "round" not in df.columns and "global_epoch" in df.columns:
            df["round"] = df["global_epoch"]

        result = RunResult(
            name=run_name, label=label,
            config=config, csv_path=csv_path, df=df,
        )
        print(
            f"[AsyncExperiment] Done — "
            f"final_acc={result.final_accuracy:.4f}  "
            f"best_acc={result.best_accuracy:.4f}\n"
        )
        return result
