"""
examples/parallelsfl_experiment.py: ParallelSFL vs FL / SFLv2 / SAFSL on
MNIST + MnistCNN under DATA and SYSTEM heterogeneity.

Compares four paradigms, all on ONE coherent physical base (FDMA Shannon rate,
DVFS compute energy, FLOPs/(f*q) compute time) so the comparison is meaningful:

  ParallelSFL — cluster-based split FL: workers grouped into clusters, each with
                a top worker holding the top submodel; adaptive per-cluster local
                frequency tau_c (Eq. 17) + KL-clustering (Alg. 1).
                (flsim.algorithms.parallel_sfl.ParallelSFL)
  FL          — synchronous FedAvg (full model on every device).
  SFLv2       — synchronous split FL, server side sequential (edge server holds
                the top submodel for all workers).
  SAFSL       — semi-asynchronous split FL (buffered, staleness-weighted).

Fairness / coherence
--------------------
  * Data heterogeneity: Dirichlet(alpha) partition across NUM_CLIENTS workers.
  * System heterogeneity: per-device CPU frequency ~ U[0.1,2] GHz, transmit
    power ~ U[0.1,0.2] W, distance ~ U[100,1000] m (exp-fading channel + SNR
    floor) — the same coherent base every paradigm reads.
  * Local work: the baselines do H = LOCAL_ITERS mini-batch steps per round;
    ParallelSFL's fastest cluster does tau_max = LOCAL_ITERS and slower clusters
    adapt DOWN per Eq. 17 (its contribution). Same maximum local work per round.
  * Energy scope = device/battery. NOTE: in ParallelSFL the top submodel runs on
    a peer DEVICE, so its compute+tx energy counts; in SFLv2/SAFSL the top runs
    on the plugged-in edge server, excluded from device energy. That asymmetry
    is real (ParallelSFL offloads to a device, not a free server) and is
    reflected faithfully.

Outputs (the ParallelSFL paper's metrics: accuracy, training time, traffic):
  accuracy vs simulated time, accuracy vs round, a time-to-accuracy table, and
  energy(J) / communication-overhead(MB) bars at target accuracies, plus the
  average-waiting-time curve (ParallelSFL's Eq. 17 story).

RUNTIME: 50 workers x split relay over many rounds is a CLUSTER/GPU job. For a
quick local check, cut NUM_CLIENTS / MAX_GLOBAL_ROUNDS hard.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from flsim.algorithms.fedavg import FedAvg
from flsim.algorithms.safsl import SAFSL
from flsim.algorithms.parallel_sfl import ParallelSFL
from flsim.allocators.equal_split import EqualSplitAllocator
from flsim.channel.conversions import dbm_to_watts
from flsim.core.evaluator import Evaluator
from flsim.core.parallel_sfl_simulator import ParallelSFLSimulator
from flsim.core.split_async_simulator import SplitAsyncSimulator
from flsim.core.split_client import SplitClient
from flsim.experiments.async_base import AsyncExperiment
from flsim.experiments.base import RunResult, _apply_config_overrides
from flsim.experiments.split_base import SplitExperiment
from flsim.experiments.wiring import (
    _load_dataset, _make_channel_model, _make_partitioner, _make_profiles,
    _model_name_for_dataset, _num_classes_for_dataset, load_config, set_seeds,
)
from flsim.models.factory import create_model
from flsim.system.flops import forward_macs
from flsim.system.split_cost import SplitCostModel
from flsim.system.split_model import split_model


BASE_CONFIG = os.path.join(os.path.dirname(__file__), "..", "flsim", "configs", "base.yaml")
OUTPUT_DIR = "outputs/parallelsfl_experiment/"

# ---- problem ----
DATASET   = "mnist"
MODEL     = "mnist_cnn"
CUT_LAYER = 6            # MnistCNN features/classifier boundary (10 layers)

# ---- federation / heterogeneity ----
NUM_CLIENTS      = 50
DIRICHLET_ALPHA  = 0.3   # data non-IID level (smaller = more heterogeneous)

# ---- local work (fair: baselines H == ParallelSFL tau_max) ----
LOCAL_ITERS   = 5        # H mini-batch steps / round (baselines) = ParallelSFL tau_max
BATCH_SIZE    = 64
LEARNING_RATE = 0.01

# ---- system heterogeneity (coherent physical base) ----
BANDWIDTH_HZ         = 50.0e6
DIST_MIN_M, DIST_MAX_M = 100.0, 1000.0
DEV_FREQ_MIN_GHZ, DEV_FREQ_MAX_GHZ = 0.1, 2.0
SERVER_FREQ_HZ       = 3.0e10
Q_DEVICE, Q_SERVER   = 1.0, 2.0
DEV_TX_POWER_MIN_W, DEV_TX_POWER_MAX_W = 0.1, 0.2
BS_DOWNLINK_POWER_W  = 0.3
NOISE_PSD_DBM_PER_HZ = -150.0
PATH_FADING_EXPONENT = 1.3
H0_CHANNEL_CONST     = 1.0e-6
MIN_SNR_DB           = 0.0
KAPPA                = 1.0e-28
ENERGY_SCOPE         = "device"

# ---- SAFSL semi-async cohort ----
SAFSL_WINDOW = NUM_CLIENTS
SAFSL_BUFFER = int(round(0.8 * NUM_CLIENTS))   # >= 0.8 N aggregated per round

# ---- run length / eval / targets ----
MAX_GLOBAL_ROUNDS = 150
EVALUATE_EVERY    = 5
ACC_TARGETS       = [0.80, 0.85, 0.90, 0.95]

METHOD_ORDER = ["FL", "SFLv2", "SAFSL", "ParallelSFL"]


def _phi_flops_per_sample() -> float:
    """Full-model FP+BP FLOPs/sample (~6*MACs) for MnistCNN, fixed for all clients."""
    m = create_model(MODEL, num_classes=_num_classes_for_dataset(DATASET))
    x = torch.randn(2, 1, 28, 28)
    return 6.0 * forward_macs(m, x)


PHI_FLOPS_PER_SAMPLE = _phi_flops_per_sample()


SHARED_OVERRIDES = {
    "data.dataset":               DATASET,
    "data.model_name":            MODEL,
    "data.num_clients":           NUM_CLIENTS,
    "data.partition":             "dirichlet",
    "data.dirichlet_alpha":       DIRICHLET_ALPHA,
    "learning.batch_size":        BATCH_SIZE,
    "learning.learning_rate":     LEARNING_RATE,
    "learning.local_iters":       LOCAL_ITERS,     # H for the baselines
    "learning.local_epochs":      1,
    "learning.cut_layer":         CUT_LAYER,
    "evaluation.evaluate_every":  EVALUATE_EVERY,
    # wireless / channel (heterogeneous system)
    "wireless.channel_model":            "exp_fading",
    "wireless.exp_fading_path_exponent": PATH_FADING_EXPONENT,
    "wireless.h0_path_loss_constant":    H0_CHANNEL_CONST,
    "wireless.deployment_shape":         "distance_range",
    "wireless.dist_min_m":               DIST_MIN_M,
    "wireless.dist_max_m":               DIST_MAX_M,
    "wireless.total_bandwidth_hz":       BANDWIDTH_HZ,
    "wireless.tx_power_w_min":           DEV_TX_POWER_MIN_W,
    "wireless.tx_power_w_max":           DEV_TX_POWER_MAX_W,
    "wireless.noise_psd_dbm_per_hz":     NOISE_PSD_DBM_PER_HZ,
    "wireless.min_snr_db":               MIN_SNR_DB,
    "wireless.min_distance_m":           1.0,
    "wireless.downlink_tx_power_w":      BS_DOWNLINK_POWER_W,
    "wireless.upload_size_mode":         "model",
    # compute (heterogeneous device frequency; fixed per-sample FLOPs)
    "system.cpu_freq_mode":       "uniform_ghz",
    "system.cpu_freq_min_ghz":    DEV_FREQ_MIN_GHZ,
    "system.cpu_freq_max_ghz":    DEV_FREQ_MAX_GHZ,
    "system.cycles_per_sample_min": PHI_FLOPS_PER_SAMPLE,
    "system.cycles_per_sample_max": PHI_FLOPS_PER_SAMPLE,
    "system.switched_capacitance":  KAPPA,
    "system.energy_scope":          ENERGY_SCOPE,
    # split cost model
    "split.server_cpu_frequency_hz": SERVER_FREQ_HZ,
    "split.q_device":                Q_DEVICE,
    "split.q_server":                Q_SERVER,
}


class ParallelSFLComparison(SplitExperiment, AsyncExperiment):
    """Drives ParallelSFL + FL + SFLv2 + SAFSL, then the full metric set."""

    def run(self):
        results = {}

        # 1. FL — synchronous FedAvg, all workers participate.
        results["FL"] = self.run_single(
            run_name="psfl_fl", label="FL",
            config_overrides={**SHARED_OVERRIDES,
                              "learning.global_rounds": MAX_GLOBAL_ROUNDS,
                              "learning.clients_per_round": NUM_CLIENTS},
            components={"algorithm": FedAvg()},
        )
        self._add_full_model_traffic(results["FL"], per_step_clients=NUM_CLIENTS)

        # 2. SFLv2 — sync split, server side sequential.
        results["SFLv2"] = self.run_single_split(
            run_name="psfl_sflv2", label="SFLv2",
            config_overrides={**SHARED_OVERRIDES,
                              "learning.global_rounds": MAX_GLOBAL_ROUNDS,
                              "learning.clients_per_round": NUM_CLIENTS},
            client_mode="parallel_fedavg", server_mode="sequential",
        )

        # 3. SAFSL — semi-async split, buffer >= 0.8N. Extra aggregations so the
        #    total worker-training work is comparable to the round-based methods.
        safsl_rounds = int(np.ceil(MAX_GLOBAL_ROUNDS * NUM_CLIENTS / SAFSL_BUFFER))
        results["SAFSL"] = self._run_safsl(
            run_name="psfl_safsl", label="SAFSL",
            global_rounds=safsl_rounds,
            evaluate_every=int(np.ceil(EVALUATE_EVERY * NUM_CLIENTS / SAFSL_BUFFER)),
        )

        # 4. ParallelSFL — clusters + adaptive frequency.
        results["ParallelSFL"] = self._run_parallelsfl(
            run_name="psfl_parallelsfl", label="ParallelSFL",
            global_rounds=MAX_GLOBAL_ROUNDS, evaluate_every=EVALUATE_EVERY,
        )

        self._plot_accuracy_vs_time(results)
        self._plot_accuracy_vs_round(results)
        self._plot_bars_at_targets(results)
        self._plot_waiting_time(results)
        self._time_to_accuracy_table(results)
        return results

    # ==================================================================
    # ParallelSFL run builder
    # ==================================================================

    def _run_parallelsfl(self, run_name, label, global_rounds, evaluate_every) -> RunResult:
        print(f"\n{'='*60}\n[ParallelSFLComparison] Run: {label}\n{'='*60}")
        overrides = {**SHARED_OVERRIDES, "learning.global_rounds": global_rounds,
                     "evaluation.evaluate_every": evaluate_every}
        config = _apply_config_overrides(load_config(BASE_CONFIG), overrides)
        set_seeds(config.experiment.seed)
        rng = np.random.RandomState(config.experiment.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_ds, test_ds = _load_dataset(config)
        idx = _make_partitioner(config.data).partition(train_ds, config.data.num_clients, rng)
        full = create_model(_model_name_for_dataset(config.data.dataset, getattr(config.data, "model_name", None)),
                            num_classes=_num_classes_for_dataset(config.data.dataset, getattr(config.data, "num_classes", None)))
        bottom, top = split_model(full, cut_layer=config.learning.cut_layer)
        clients = [SplitClient(client_id=k, dataset=train_ds, indices=idx[k]) for k in range(config.data.num_clients)]

        noise = dbm_to_watts(config.wireless.noise_psd_dbm_per_hz)
        config._noise_psd_w_per_hz = noise
        channel_model = _make_channel_model(config, noise)
        profiles = _make_profiles(config, [len(i) for i in idx], rng)
        cost_model = SplitCostModel(
            channel_model=channel_model, noise_psd_w_per_hz=noise,
            kappa=config.system.switched_capacitance,
            server_cpu_frequency_hz=float(config.split.server_cpu_frequency_hz),
            q_device=float(config.split.q_device), q_server=float(config.split.q_server),
            downlink_tx_power_w=getattr(config.wireless, "downlink_tx_power_w", None),
            energy_scope=getattr(config.system, "energy_scope", "total"),
        )
        sim = ParallelSFLSimulator(
            clients=clients, bottom_model=bottom, top_model=top,
            algorithm=ParallelSFL(lam=0.5, tau_max=LOCAL_ITERS),
            evaluator=Evaluator(test_dataset=test_ds), config=config, rng=rng, device=device,
            profiles=profiles, cost_model=cost_model,
        )
        history = sim.run()

        run_dir = os.path.join(self.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        csv_path = os.path.join(run_dir, f"{run_name}.csv")
        cols = ["round", "test_accuracy", "test_loss", "simulated_time_s",
                "traffic_bytes", "cumulative_traffic_bytes",
                "total_energy_j", "cumulative_energy_j",
                "round_latency_s", "avg_waiting_time_s", "num_clusters"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in history:
                w.writerow({
                    "round": r.round,
                    "test_accuracy": f"{r.test_accuracy:.6f}" if r.test_accuracy is not None else "",
                    "test_loss": f"{r.test_loss:.6f}" if r.test_loss is not None else "",
                    "simulated_time_s": f"{r.simulated_time_s:.6f}",
                    "traffic_bytes": f"{r.traffic_bytes:.1f}",
                    "cumulative_traffic_bytes": f"{r.cumulative_traffic_bytes:.1f}",
                    "total_energy_j": f"{r.total_energy_j:.6e}",
                    "cumulative_energy_j": f"{r.cumulative_energy_j:.6e}",
                    "round_latency_s": f"{r.round_latency_s:.6f}",
                    "avg_waiting_time_s": f"{r.avg_waiting_time_s:.6f}",
                    "num_clusters": r.num_clusters,
                })
        df = pd.read_csv(csv_path)
        return RunResult(name=run_name, label=label, config=config, csv_path=csv_path, df=df)

    # ==================================================================
    # SAFSL run builder (same as examples/SAFSL_experiment.py)
    # ==================================================================

    def _run_safsl(self, run_name, label, global_rounds, evaluate_every) -> RunResult:
        print(f"\n{'='*60}\n[ParallelSFLComparison] Run: {label}\n{'='*60}")
        overrides = {**SHARED_OVERRIDES, "learning.global_rounds": global_rounds,
                     "learning.clients_per_round": NUM_CLIENTS,
                     "evaluation.evaluate_every": evaluate_every,
                     "async_fl.window_size": SAFSL_WINDOW}
        config = _apply_config_overrides(load_config(BASE_CONFIG), overrides)
        set_seeds(config.experiment.seed)
        rng = np.random.RandomState(config.experiment.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_ds, test_ds = _load_dataset(config)
        idx = _make_partitioner(config.data).partition(train_ds, config.data.num_clients, rng)
        full = create_model(_model_name_for_dataset(config.data.dataset, getattr(config.data, "model_name", None)),
                            num_classes=_num_classes_for_dataset(config.data.dataset, getattr(config.data, "num_classes", None)))
        client_model, server_model = split_model(full, cut_layer=config.learning.cut_layer)
        clients = [SplitClient(client_id=k, dataset=train_ds, indices=idx[k]) for k in range(config.data.num_clients)]

        noise = dbm_to_watts(config.wireless.noise_psd_dbm_per_hz)
        config._noise_psd_w_per_hz = noise
        channel_model = _make_channel_model(config, noise)
        profiles = _make_profiles(config, [len(i) for i in idx], rng)
        cost_model = SplitCostModel(
            channel_model=channel_model, noise_psd_w_per_hz=noise,
            kappa=config.system.switched_capacitance,
            server_cpu_frequency_hz=float(config.split.server_cpu_frequency_hz),
            q_device=float(config.split.q_device), q_server=float(config.split.q_server),
            downlink_tx_power_w=getattr(config.wireless, "downlink_tx_power_w", None),
            energy_scope=getattr(config.system, "energy_scope", "total"),
        )
        sim = SplitAsyncSimulator(
            clients=clients, client_model=client_model, server_model=server_model,
            algorithm=SAFSL(k=SAFSL_BUFFER, gamma=1.0),
            evaluator=Evaluator(test_dataset=test_ds), cost_model=cost_model, profiles=profiles,
            allocator=EqualSplitAllocator(), config=config, rng=rng, device=device,
        )
        history = sim.run()

        run_dir = os.path.join(self.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        csv_path = os.path.join(run_dir, f"{run_name}.csv")
        cols = ["round", "test_accuracy", "test_loss", "simulated_time_s",
                "traffic_bytes", "cumulative_traffic_bytes",
                "total_energy_j", "cumulative_energy_j", "round_latency_s"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in history:
                w.writerow({
                    "round": r.global_epoch,
                    "test_accuracy": f"{r.test_accuracy:.6f}" if r.test_accuracy is not None else "",
                    "test_loss": f"{r.test_loss:.6f}" if r.test_loss is not None else "",
                    "simulated_time_s": f"{r.simulated_time_s:.6f}",
                    "traffic_bytes": f"{r.traffic_bytes:.1f}",
                    "cumulative_traffic_bytes": f"{r.cumulative_traffic_bytes:.1f}",
                    "total_energy_j": f"{r.total_energy_j:.6e}",
                    "cumulative_energy_j": f"{r.cumulative_energy_j:.6e}",
                    "round_latency_s": f"{r.round_latency_s:.6f}",
                })
        df = pd.read_csv(csv_path)
        return RunResult(name=run_name, label=label, config=config, csv_path=csv_path, df=df)

    # ------------------------------------------------------------------
    # FL full-model traffic column (its CSV has time+energy but no traffic)
    # ------------------------------------------------------------------

    def _add_full_model_traffic(self, result: RunResult, per_step_clients: int) -> None:
        model = create_model(
            _model_name_for_dataset(result.config.data.dataset, getattr(result.config.data, "model_name", None)),
            num_classes=_num_classes_for_dataset(result.config.data.dataset, getattr(result.config.data, "num_classes", None)),
        )
        elems = sum(t.numel() for t in model.state_dict().values())
        per_step = 2 * per_step_clients * elems * 4
        df = result.df
        df["traffic_bytes"] = float(per_step)
        df["cumulative_traffic_bytes"] = per_step * (np.arange(len(df)) + 1)

    # ==================================================================
    # Analysis + plots
    # ==================================================================

    @staticmethod
    def _curve(df):
        sub = df.dropna(subset=["test_accuracy"]).copy().sort_values("simulated_time_s")
        return (sub["test_accuracy"].to_numpy(),
                sub["simulated_time_s"].to_numpy(),
                sub["cumulative_energy_j"].to_numpy(),
                sub["cumulative_traffic_bytes"].to_numpy() / 1e6)

    @classmethod
    def _first_crossing(cls, df, target):
        acc, t, e, mb = cls._curve(df)
        hit = np.where(acc >= target)[0]
        if len(hit) == 0:
            return (np.nan, np.nan, np.nan)
        i = hit[0]
        return (t[i], e[i], mb[i])

    def _plot_accuracy_vs_time(self, results):
        os.makedirs(self.output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.tab10.colors
        for i, name in enumerate([m for m in METHOD_ORDER if m in results]):
            acc, t, _, _ = self._curve(results[name].df)
            ax.plot(t, acc * 100, marker="o", markersize=3, linewidth=1.6, label=name, color=colors[i % 10])
        ax.set_xlabel("Training latency (simulated seconds)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"Accuracy vs Training Latency — {MODEL} / {DATASET} (alpha={DIRICHLET_ALPHA})")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
        out = os.path.join(self.output_dir, "psfl_accuracy_vs_time.png")
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        print(f"[ParallelSFLComparison] Saved {out}")

    def _plot_accuracy_vs_round(self, results):
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.tab10.colors
        for i, name in enumerate([m for m in METHOD_ORDER if m in results]):
            df = results[name].df.dropna(subset=["test_accuracy"])
            ax.plot(df["round"], df["test_accuracy"] * 100, marker="o", markersize=3,
                    linewidth=1.6, label=name, color=colors[i % 10])
        ax.set_xlabel("Aggregation round"); ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"Accuracy vs Round — {MODEL} / {DATASET} (alpha={DIRICHLET_ALPHA})")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
        out = os.path.join(self.output_dir, "psfl_accuracy_vs_round.png")
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        print(f"[ParallelSFLComparison] Saved {out}")

    def _plot_bars_at_targets(self, results):
        methods = [m for m in METHOD_ORDER if m in results]
        colors = plt.cm.tab10.colors
        for metric_idx, ylabel, fname, title in [
            (1, "Energy consumption (J)", "psfl_energy_to_accuracy.png", "Energy to reach target accuracy"),
            (2, "Communication overhead (MB)", "psfl_overhead_to_accuracy.png", "Communication overhead to reach target accuracy"),
        ]:
            fig, ax = plt.subplots(figsize=(9, 5))
            n_groups, n_methods = len(ACC_TARGETS), len(methods)
            width = 0.8 / n_methods; x = np.arange(n_groups)
            for mi, name in enumerate(methods):
                vals = [self._first_crossing(results[name].df, thr)[metric_idx] for thr in ACC_TARGETS]
                ax.bar(x + mi * width, vals, width, label=name, color=colors[mi % 10])
            ax.set_xticks(x + width * (n_methods - 1) / 2)
            ax.set_xticklabels([f"{int(t*100)}%" for t in ACC_TARGETS])
            ax.set_xlabel("Target test accuracy"); ax.set_ylabel(ylabel)
            ax.set_title(f"{title} — {MODEL} / {DATASET}")
            ax.grid(True, alpha=0.3, axis="y"); ax.legend(fontsize=9)
            out = os.path.join(self.output_dir, fname)
            fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
            print(f"[ParallelSFLComparison] Saved {out}")

    def _waiting_series(self, df):
        """
        Per-round average waiting time, by what each simulator exposes:
          * ParallelSFL — exact (Eq. 16 avg_waiting_time_s column).
          * FL (sync)   — proxy: round_duration - mean per-client completion.
          * SAFSL       — async, no synchronization barrier -> 0.
          * SFLv2       — split-sync, per-client times not exposed -> NaN (omitted).
        """
        r = df["round"].to_numpy()
        if "avg_waiting_time_s" in df.columns:                       # ParallelSFL (exact)
            return r, df["avg_waiting_time_s"].to_numpy()
        need = {"round_duration_s", "mean_compute_time_s", "mean_upload_time_s"}
        if need.issubset(df.columns):                               # FL sync proxy
            dn = (df["mean_compute_time_s"] + df["mean_upload_time_s"]
                  + df.get("mean_download_time_s", 0.0))
            return r, (df["round_duration_s"] - dn).clip(lower=0.0).to_numpy()
        if "staleness" in df.columns:                              # SAFSL async, no barrier
            return r, np.zeros(len(df))
        return r, np.full(len(df), np.nan)                         # SFLv2: not exposed

    def _plot_waiting_time(self, results):
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.tab10.colors
        for i, name in enumerate([m for m in METHOD_ORDER if m in results]):
            r, w = self._waiting_series(results[name].df)
            ax.plot(r, w, marker="o", markersize=2, linewidth=1.3, label=name, color=colors[i % 10])
        ax.set_xlabel("Aggregation round"); ax.set_ylabel("Avg. waiting time (s)")
        ax.set_title(f"Average waiting time per round — {MODEL} / {DATASET}")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
        out = os.path.join(self.output_dir, "psfl_waiting_time.png")
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        print(f"[ParallelSFLComparison] Saved {out}")

    def _time_to_accuracy_table(self, results):
        methods = [m for m in METHOD_ORDER if m in results]
        header = "  " + f"{'target':>8s}" + "".join(f"{m:>14s}" for m in methods)
        lines = ["=" * len(header), "  TIME-TO-ACCURACY (simulated seconds)", header]
        for thr in ACC_TARGETS:
            row = f"  {int(thr*100):>7d}%"
            for name in methods:
                t = self._first_crossing(results[name].df, thr)[0]
                row += (f"{t:>14.0f}" if np.isfinite(t) else f"{'--':>14s}")
            lines.append(row)
        lines.append("=" * len(header))
        table = "\n".join(lines)
        print("\n" + table)
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "psfl_time_to_accuracy.txt"), "w") as f:
            f.write(table + "\n")
        print(f"[ParallelSFLComparison] Saved {self.output_dir}psfl_time_to_accuracy.txt")


if __name__ == "__main__":
    ParallelSFLComparison(base_config=BASE_CONFIG, output_dir=OUTPUT_DIR).run()
