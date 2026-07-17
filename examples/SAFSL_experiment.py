"""
examples/SAFSL_experiment.py: Semi-Asynchronous Federated Split Learning (SAFSL)
comparison on ResNet-18 / CIFAR-10, reproducing the paper's
"test accuracy vs training latency" figure plus energy- and
communication-overhead-to-accuracy bar charts.

WHAT IT COMPARES (all six on ONE coherent physical base — see below)
--------------------------------------------------------------------
  SAFSL     — semi-asynchronous split FL (flsim.algorithms.safsl.SAFSL over
              flsim.core.split_async_simulator.SplitAsyncSimulator); buffers
              >= 0.8 N devices per aggregation, stragglers keep training.
  SFLV1     — synchronous split FL, server+client sides FedAvg-aggregated.
  SFLV2     — synchronous split FL, client side FedAvg, server side sequential.
  SL        — vanilla (sequential) split learning.
  FL        — synchronous FedAvg (full model on device).
  FedAsync  — asynchronous full-model FL (Xie et al. 2019).

HOW "TIME" IS TREATED (the paper's methodology)
-----------------------------------------------
There is NO fixed global-iteration or wall-clock budget. Each method runs for
enough rounds/epochs that even the slowest crosses the top target accuracy,
evaluating frequently and recording, at every evaluation,
    (cumulative simulated_time_s, cumulative_energy_j, cumulative_traffic, acc).
Then:
  * "Accuracy vs simulated time" is plotted directly — each curve naturally
    extends to a different x-position because each paradigm has a different
    per-round latency (the paper's Fig.).
  * For each target accuracy (40/50/60/70 %), we read off the FIRST evaluation
    where acc >= target and take the cumulative energy / traffic / time there —
    giving the "energy (J) / overhead (MB) / latency (s) to reach X%" bars and
    the time-to-accuracy table (the paper's textual result, e.g. "FL takes
    75661 s to reach 60% while SAFSL takes 19380 s").

FAIRNESS / COHERENCE (identical local work + one physical base for ALL methods)
-------------------------------------------------------------------------------
  * Local work per round is IDENTICAL for every method: H = LOCAL_ITERS
    mini-batch SGD steps of BATCH_SIZE samples (learning.local_iters). So each
    round does the same local work — the only differences measured are the
    paradigm's own (compute offload to the BS, comm pattern, staleness).
  * Latency/energy/traffic use ONE physical base for all six: FDMA Shannon rate
    (exp-fading channel), DVFS compute energy kappa*f^2*FLOPs/q, and
    FLOPs/(f*q) compute time. Split methods additionally offload part of the
    per-cut-layer compute to the fast edge server (q_server) and pay smashed-
    data + gradient communication; FL/FedAsync run the whole model on-device
    and exchange the full model. The device-side compute physics is the SAME
    formula for every method (q_device=1).

PAPER PARAMETERS (SAFSL system model)
-------------------------------------
  N=10 devices; >= 0.8N aggregated/round; device dist ~ U[100,1000] m;
  device freq ~ U[0.1,2]e9 cyc/s (q_n=1); edge server f_S=3e10 (q_S=2);
  B=50 MHz; device tx power ~ U[0.1,0.2] W; BS downlink power 0.3 W;
  N0=1e-18 W/Hz (=-150 dBm/Hz); path-fading exponent 1.3, complex-Gaussian
  small-scale fading; b=64; kappa=1e-28. Model split at CUT_LAYER of ResNet-18.

RUNTIME NOTE
------------
ResNet-18/CIFAR-10 to 60-70% over thousands of rounds is a CLUSTER/GPU job, not
a laptop run. MAX_GLOBAL_ROUNDS below is set for a real run; for a quick local
sanity check, drop it (and dataset size) drastically. A SLURM script is in
slurm/ (mirror run_splitfed.slurm).
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
from flsim.algorithms.fedasync import FedAsync
from flsim.algorithms.safsl import SAFSL
from flsim.allocators.equal_split import EqualSplitAllocator
from flsim.channel.conversions import dbm_to_watts
from flsim.core.evaluator import Evaluator
from flsim.core.split_async_simulator import SplitAsyncSimulator
from flsim.core.split_client import SplitClient
from flsim.experiments.async_base import AsyncExperiment
from flsim.experiments.base import RunResult, _apply_config_overrides
from flsim.experiments.split_base import SplitExperiment
from flsim.experiments.wiring import (
    _load_dataset,
    _make_channel_model,
    _make_partitioner,
    _make_profiles,
    _model_name_for_dataset,
    _num_classes_for_dataset,
    load_config,
    set_seeds,
)
from flsim.models.factory import create_model
from flsim.system.flops import forward_macs
from flsim.system.split_cost import SplitCostModel
from flsim.system.split_model import split_model


BASE_CONFIG = os.path.join(os.path.dirname(__file__), "..", "flsim", "configs", "base.yaml")
OUTPUT_DIR = "outputs/safsl_experiment/"

# ---- Problem ----
DATASET   = "cifar10"
MODEL     = "resnet18"
CUT_LAYER = 6            # ResNet-18: device keeps stem+stage1+first stage2 conv (dev ~38% FLOPs)

# ---- Fleet / federation (paper) ----
NUM_DEVICES        = 10
PARTICIPATION_FRAC = 0.8                                  # >= 0.8 N aggregated per round
BUFFER_K           = int(round(PARTICIPATION_FRAC * NUM_DEVICES))   # SAFSL |S_t| = 8
WINDOW_SIZE        = NUM_DEVICES                          # concurrent in-flight devices

# ---- Local training (identical for ALL methods -> fair) ----
LOCAL_ITERS   = 10      # H: mini-batch SGD steps per round (paper's local iterations)
BATCH_SIZE    = 64
LEARNING_RATE = 0.0001

# ---- Physical parameters (paper) ----
BANDWIDTH_HZ         = 50.0e6      # B = 50 MHz
DIST_MIN_M           = 100.0
DIST_MAX_M           = 1000.0
DEV_FREQ_MIN_GHZ     = 0.1         # device f ~ U[0.1, 2] x 1e9
DEV_FREQ_MAX_GHZ     = 2.0
SERVER_FREQ_HZ       = 3.0e10      # f_S = 3 x 1e10
Q_DEVICE             = 1.0         # q_n
Q_SERVER             = 2.0         # q_S
DEV_TX_POWER_MIN_W   = 0.1         # p_n ~ U[0.1, 0.2] W
DEV_TX_POWER_MAX_W   = 0.2
BS_DOWNLINK_POWER_W  = 0.3         # P^DL
NOISE_PSD_DBM_PER_HZ = -150.0      # N0 = 1e-18 W/Hz  ->  10*log10(1e-18)+30 = -150 dBm/Hz
PATH_FADING_EXPONENT = 1.3
KAPPA                = 1.0e-28
H0_CHANNEL_CONST     = 1.0e-6      # exp-fading SNR-calibration constant (paper cites [39];
                                   # tuned here for a reasonable cell-edge SNR — adjust to taste)

# ---- Run length & evaluation (time-to-accuracy is extracted post-hoc) ----
# CLUSTER SETTING. For a laptop sanity check, cut these hard (see RUNTIME NOTE).
MAX_GLOBAL_ROUNDS = 2000
EVALUATE_EVERY    = 10             # evaluate every N rounds/aggregations

# ---- Target accuracies for the energy/overhead bars + latency table ----
ACC_TARGETS = [0.40, 0.50, 0.60, 0.70]

METHOD_ORDER = ["FL", "FedAsync", "SL", "SFLV1", "SFLV2", "SAFSL"]


def _resnet_flops_per_sample() -> float:
    """
    Full-model FLOPs per sample for the chosen model (FP + BP), fixed for every
    client. 1 MAC = 2 FLOPs; forward = 2*MACs; backward ~= 2*forward; so
    FP+BP ~= 6*MACs (standard training-FLOPs convention). This is the paper's
    per-sample computing workload Phi; the split cost model divides it between
    device and server by the measured cut-layer FLOP fraction.
    """
    m = create_model(MODEL, num_classes=_num_classes_for_dataset(DATASET))
    x = torch.randn(2, 3, 32, 32)
    return 6.0 * forward_macs(m, x)


PHI_FLOPS_PER_SAMPLE = _resnet_flops_per_sample()


# The physical/system overrides shared by EVERY method (the coherent base).
SHARED_OVERRIDES = {
    "data.dataset":               DATASET,
    "data.model_name":            MODEL,
    "data.num_clients":           NUM_DEVICES,
    "data.partition":             "iid",
    "learning.batch_size":        BATCH_SIZE,
    "learning.learning_rate":     LEARNING_RATE,
    "learning.local_iters":       LOCAL_ITERS,     # H — identical local work for all
    "learning.local_epochs":      1,               # ignored while local_iters is set
    "evaluation.evaluate_every":  EVALUATE_EVERY,
    # --- wireless / channel (paper) ---
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
    "wireless.min_distance_m":           1.0,
    # FL/FedAsync exchange the FULL model -> size their up/down comm by the real net.
    "wireless.upload_size_mode":         "model",
    # --- compute (paper): device f ~ U[0.1,2] GHz, fixed per-sample FLOPs ---
    "system.cpu_freq_mode":       "uniform_ghz",
    "system.cpu_freq_min_ghz":    DEV_FREQ_MIN_GHZ,
    "system.cpu_freq_max_ghz":    DEV_FREQ_MAX_GHZ,
    "system.cycles_per_sample_min": PHI_FLOPS_PER_SAMPLE,   # Phi, same for all clients
    "system.cycles_per_sample_max": PHI_FLOPS_PER_SAMPLE,
    "system.switched_capacitance":  KAPPA,
    # --- split cost model (edge server + asymmetric downlink) ---
    "split.server_cpu_frequency_hz": SERVER_FREQ_HZ,
    "split.q_device":                Q_DEVICE,
    "split.q_server":                Q_SERVER,
    "split.downlink_tx_power_w":     BS_DOWNLINK_POWER_W,
    "learning.cut_layer":            CUT_LAYER,
}


class SAFSLExperiment(SplitExperiment, AsyncExperiment):
    """Drives all six methods, then produces the acc-vs-time plot, the
    energy/overhead-to-accuracy bars, and the time-to-accuracy table."""

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self):
        results = {}

        # 1. FL — synchronous FedAvg, full model on device, all N participate.
        results["FL"] = self.run_single(
            run_name="safsl_fl", label="FL",
            config_overrides={**SHARED_OVERRIDES,
                              "learning.global_rounds": MAX_GLOBAL_ROUNDS,
                              "learning.clients_per_round": NUM_DEVICES},
            components={"algorithm": FedAvg()},
        )
        self._add_full_model_traffic(results["FL"], per_step_clients=NUM_DEVICES)

        # 2. FedAsync — full-model async baseline (1 update per epoch). Runs
        #    NUM_DEVICES x more epochs so total client-training work is comparable.
        results["FedAsync"] = self.run_single_async(
            run_name="safsl_fedasync", label="FedAsync",
            config_overrides={**SHARED_OVERRIDES,
                              "learning.global_rounds": MAX_GLOBAL_ROUNDS * NUM_DEVICES,
                              "learning.clients_per_round": NUM_DEVICES,
                              "async_fl.window_size": WINDOW_SIZE,
                              "evaluation.evaluate_every": EVALUATE_EVERY * NUM_DEVICES},
            components={"algorithm": FedAsync(alpha=0.1)},
        )
        self._add_full_model_traffic(results["FedAsync"], per_step_clients=1)

        # 3. SL — vanilla sequential split learning.
        results["SL"] = self.run_single_split(
            run_name="safsl_sl", label="SL",
            config_overrides={**SHARED_OVERRIDES,
                              "learning.global_rounds": MAX_GLOBAL_ROUNDS,
                              "learning.clients_per_round": NUM_DEVICES},
            client_mode="sequential", server_mode="sequential",
        )

        # 4. SFLV1 — parallel client + parallel (FedAvg) server.
        results["SFLV1"] = self.run_single_split(
            run_name="safsl_sflv1", label="SFLV1",
            config_overrides={**SHARED_OVERRIDES,
                              "learning.global_rounds": MAX_GLOBAL_ROUNDS,
                              "learning.clients_per_round": NUM_DEVICES},
            client_mode="parallel_fedavg", server_mode="parallel_fedavg",
        )

        # 5. SFLV2 — parallel client + sequential server.
        results["SFLV2"] = self.run_single_split(
            run_name="safsl_sflv2", label="SFLV2",
            config_overrides={**SHARED_OVERRIDES,
                              "learning.global_rounds": MAX_GLOBAL_ROUNDS,
                              "learning.clients_per_round": NUM_DEVICES},
            client_mode="parallel_fedavg", server_mode="sequential",
        )

        # 6. SAFSL — semi-async split FL; buffer BUFFER_K (>= 0.8N) per aggregation.
        #    Runs enough aggregations to match the total client-training work of
        #    the round-based methods (each sync round aggregates NUM_DEVICES
        #    devices; each SAFSL aggregation buffers BUFFER_K).
        safsl_aggregations = int(np.ceil(MAX_GLOBAL_ROUNDS * NUM_DEVICES / BUFFER_K))
        results["SAFSL"] = self._run_safsl(
            run_name="safsl_safsl", label="SAFSL",
            global_rounds=safsl_aggregations,
            evaluate_every=int(np.ceil(EVALUATE_EVERY * NUM_DEVICES / BUFFER_K)),
        )

        # ---- analysis + figures ----
        self._plot_accuracy_vs_time(results)
        self._plot_bars_at_targets(results)
        self._time_to_accuracy_table(results)
        return results

    # ------------------------------------------------------------------
    # SAFSL run builder (no experiment base exists for the async split
    # simulator — wire it here, mirroring SplitExperiment.build_split_run).
    # ------------------------------------------------------------------

    def _run_safsl(self, run_name: str, label: str, global_rounds: int,
                   evaluate_every: int) -> RunResult:
        print(f"\n{'='*60}\n[SAFSLExperiment] Run: {label}\n{'='*60}")
        overrides = {**SHARED_OVERRIDES,
                     "learning.global_rounds": global_rounds,
                     "learning.clients_per_round": NUM_DEVICES,
                     "evaluation.evaluate_every": evaluate_every,
                     "async_fl.window_size": WINDOW_SIZE}

        config = _apply_config_overrides(load_config(BASE_CONFIG), overrides)
        set_seeds(config.experiment.seed)
        rng = np.random.RandomState(config.experiment.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_ds, test_ds = _load_dataset(config)
        partitioner = _make_partitioner(config.data)
        client_indices = partitioner.partition(train_ds, config.data.num_clients, rng)

        full_model = create_model(
            _model_name_for_dataset(config.data.dataset, getattr(config.data, "model_name", None)),
            num_classes=_num_classes_for_dataset(config.data.dataset, getattr(config.data, "num_classes", None)),
        )
        client_model, server_model = split_model(full_model, cut_layer=config.learning.cut_layer)

        clients = [SplitClient(client_id=k, dataset=train_ds, indices=client_indices[k])
                   for k in range(config.data.num_clients)]

        noise_psd = dbm_to_watts(config.wireless.noise_psd_dbm_per_hz)
        config._noise_psd_w_per_hz = noise_psd
        channel_model = _make_channel_model(config, noise_psd)
        profiles = _make_profiles(config, [len(i) for i in client_indices], rng)
        cost_model = SplitCostModel(
            channel_model=channel_model,
            noise_psd_w_per_hz=noise_psd,
            kappa=config.system.switched_capacitance,
            server_cpu_frequency_hz=float(config.split.server_cpu_frequency_hz),
            q_device=float(config.split.q_device),
            q_server=float(config.split.q_server),
            downlink_tx_power_w=getattr(config.split, "downlink_tx_power_w", None),
        )
        evaluator = Evaluator(test_dataset=test_ds)

        sim = SplitAsyncSimulator(
            clients=clients, client_model=client_model, server_model=server_model,
            algorithm=SAFSL(k=BUFFER_K, gamma=1.0),
            evaluator=evaluator, cost_model=cost_model, profiles=profiles,
            allocator=EqualSplitAllocator(), config=config, rng=rng, device=device,
        )
        history = sim.run()

        run_dir = os.path.join(self.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        csv_path = os.path.join(run_dir, f"{run_name}.csv")
        cols = ["round", "test_accuracy", "test_loss", "simulated_time_s",
                "traffic_bytes", "cumulative_traffic_bytes",
                "total_energy_j", "cumulative_energy_j", "round_latency_s", "staleness"]
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
                    "staleness": r.staleness,
                })
        df = pd.read_csv(csv_path)
        return RunResult(name=run_name, label=label, config=config, csv_path=csv_path, df=df)

    # ------------------------------------------------------------------
    # Full-model traffic column for FL / FedAsync (their CSVs have time+energy
    # but no traffic — comm = full model down+up, per participating client).
    # ------------------------------------------------------------------

    def _add_full_model_traffic(self, result: RunResult, per_step_clients: int) -> None:
        model = create_model(
            _model_name_for_dataset(result.config.data.dataset, getattr(result.config.data, "model_name", None)),
            num_classes=_num_classes_for_dataset(result.config.data.dataset, getattr(result.config.data, "num_classes", None)),
        )
        elems = sum(t.numel() for t in model.state_dict().values())
        per_step = 2 * per_step_clients * elems * 4   # bytes: down + up, float32
        df = result.df
        df["traffic_bytes"] = float(per_step)
        df["cumulative_traffic_bytes"] = per_step * (np.arange(len(df)) + 1)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _curve(df: pd.DataFrame):
        """(acc, time_s, cum_energy_J, cum_traffic_MB) at evaluated rows only,
        sorted by time — the common shape every method's df is reduced to."""
        sub = df.dropna(subset=["test_accuracy"]).copy()
        sub = sub.sort_values("simulated_time_s")
        return (sub["test_accuracy"].to_numpy(),
                sub["simulated_time_s"].to_numpy(),
                sub["cumulative_energy_j"].to_numpy(),
                sub["cumulative_traffic_bytes"].to_numpy() / 1e6)

    @classmethod
    def _first_crossing(cls, df: pd.DataFrame, target: float):
        """(time_s, energy_J, traffic_MB) at the FIRST evaluation with
        acc >= target, or (nan, nan, nan) if the target is never reached."""
        acc, t, e, mb = cls._curve(df)
        hit = np.where(acc >= target)[0]
        if len(hit) == 0:
            return (np.nan, np.nan, np.nan)
        i = hit[0]
        return (t[i], e[i], mb[i])

    def _plot_accuracy_vs_time(self, results: dict) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.tab10.colors
        for i, name in enumerate([m for m in METHOD_ORDER if m in results]):
            acc, t, _, _ = self._curve(results[name].df)
            ax.plot(t, acc * 100.0, marker="o", markersize=3, linewidth=1.6,
                    label=name, color=colors[i % 10])
        ax.set_xlabel("Training latency (simulated seconds)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"Accuracy vs Training Latency — {MODEL} / {DATASET}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        out = os.path.join(self.output_dir, "safsl_accuracy_vs_time.png")
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        print(f"[SAFSLExperiment] Saved {out}")

    def _plot_bars_at_targets(self, results: dict) -> None:
        methods = [m for m in METHOD_ORDER if m in results]
        colors = plt.cm.tab10.colors
        # energy (index 1) and traffic-MB (index 2) of _first_crossing
        for metric_idx, ylabel, fname, title in [
            (1, "Energy consumption (J)", "safsl_energy_to_accuracy.png",
             "Energy to reach target accuracy"),
            (2, "Communication overhead (MB)", "safsl_overhead_to_accuracy.png",
             "Communication overhead to reach target accuracy"),
        ]:
            fig, ax = plt.subplots(figsize=(9, 5))
            n_groups = len(ACC_TARGETS)
            n_methods = len(methods)
            width = 0.8 / n_methods
            x = np.arange(n_groups)
            for mi, name in enumerate(methods):
                vals = [self._first_crossing(results[name].df, thr)[metric_idx] for thr in ACC_TARGETS]
                ax.bar(x + mi * width, vals, width, label=name, color=colors[mi % 10])
            ax.set_xticks(x + width * (n_methods - 1) / 2)
            ax.set_xticklabels([f"{int(t*100)}%" for t in ACC_TARGETS])
            ax.set_xlabel("Target test accuracy")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} — {MODEL} / {DATASET}")
            ax.grid(True, alpha=0.3, axis="y")
            ax.legend(fontsize=9)
            out = os.path.join(self.output_dir, fname)
            fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
            print(f"[SAFSLExperiment] Saved {out}")

    def _time_to_accuracy_table(self, results: dict) -> None:
        methods = [m for m in METHOD_ORDER if m in results]
        lines = []
        header = "  " + f"{'target':>8s}" + "".join(f"{m:>12s}" for m in methods)
        lines.append("=" * len(header)); lines.append("  TIME-TO-ACCURACY (simulated seconds)"); lines.append(header)
        for thr in ACC_TARGETS:
            row = f"  {int(thr*100):>7d}%"
            for name in methods:
                t = self._first_crossing(results[name].df, thr)[0]
                row += (f"{t:>12.0f}" if np.isfinite(t) else f"{'--':>12s}")
            lines.append(row)
        lines.append("=" * len(header))
        table = "\n".join(lines)
        print("\n" + table)
        out = os.path.join(self.output_dir, "safsl_time_to_accuracy.txt")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(out, "w") as f:
            f.write(table + "\n")
        print(f"[SAFSLExperiment] Saved {out}")


if __name__ == "__main__":
    SAFSLExperiment(base_config=BASE_CONFIG, output_dir=OUTPUT_DIR).run()
