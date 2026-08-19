"""
examples/fl_vs_fedasync_experiment.py: FL (sync FedAvg) vs FedAsync on
ResNet-34 / CIFAR-10, under SYSTEM heterogeneity, at TWO data-heterogeneity
levels (IID and high non-IID). Four runs:

    FL       (IID)   |  FedAsync (IID)
    FL   (non-IID)   |  FedAsync (non-IID)

Everything sits on ONE coherent physical base (FDMA Shannon uplink rate, DVFS
compute energy, compute time = C·D / (f·q)) so FL and FedAsync are directly
comparable: identical local work per round (H = LOCAL_ITERS mini-batch steps),
identical device population, identical channel.


THE TIMING SCALE — paper-faithful (Φ = MACs, q = 1)
---------------------------------------------------
Compute time per round is

        t_compute = Φ · (H·b) / (f · q)

where Φ = cycles_per_sample, f = device clock (Hz), q = FLOPs-per-cycle. We
follow the split-FL papers (SAFSL / AdaptSFL / ASAFL) exactly:

  * Φ = the model's MACs — measured AUTOMATICALLY by the framework
    (system.cycles_per_sample_mode="model_macs", the default), so it always
    matches whatever model you pick. The papers quote ResNet-34 as "0.3 billion
    FLOPs", which equals its MAC count (≈ 0.30e9 @ 64×64). NOT 6·MACs (the true
    fwd+bwd training cost, 6× larger and NOT what the papers plug in).
  * q = 1 — the papers set device q_n = 1 (edge-server q_S = 2 lives in the
    split cost model, not here). With Φ = MACs and q = 1, f·q is the device's
    stated compute capability in FLOP/s, and ResNet-34 lands at the papers'
    ~1e4-scale for modest per-round work.

COMPUTE (Φ, MACs) drives compute time/energy; COMMUNICATION (model PARAMETERS,
here ResNet-34 = 21 M ≈ 85 MB/transfer) drives upload time/traffic/TX energy —
two independent costs. The total scale is set by Φ · (H·b) · rounds / (f·q)
for compute plus (params · rounds / rate) for comm; tune LOCAL_ITERS (H) /
MAX_GLOBAL_ROUNDS to place the total where you want it.

RUNTIME: ResNet-34 × 20 clients × 300 rounds is a GPU job. For a quick local
smoke, use `--smoke` (see the __main__ guard).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flsim.algorithms.fedavg import FedAvg
from flsim.algorithms.fedasync import FedAsync
from flsim.experiments.async_base import AsyncExperiment


BASE_CONFIG = os.path.join(os.path.dirname(__file__), "..", "flsim", "configs", "base.yaml")
OUTPUT_DIR  = "outputs/fl_vs_fedasync_experiment/"

# ---- problem ----
DATASET    = "cifar10"
MODEL      = "resnet34"
IMAGE_SIZE = 64            # standard ImageNet stem needs >= 64×64

# ---- federation ----
NUM_CLIENTS       = 20
MAX_GLOBAL_ROUNDS = 300
EVALUATE_EVERY    = 5
NONIID_ALPHA      = 0.1    # high data heterogeneity (Dirichlet); IID run uses partition="iid"

# ---- local work (fair: FL and FedAsync do the SAME per-round work) ----
LOCAL_ITERS   = 16         # H mini-batch SGD steps / round
BATCH_SIZE    = 64
LEARNING_RATE = 0.05

# ---- system heterogeneity: compute ----
# Device computing capability ~ U[0.1, 2] × 1e9 cycles/s with q = 1 (SAFSL
# convention: "[0.1, 2] × 10^9 cycles/s with q_n = 1"). With Φ = MACs, f·q is
# the device's FLOP/s. Edge-server q_S = 2 lives in the split cost model only.
DEV_FREQ_MIN_GHZ, DEV_FREQ_MAX_GHZ = 0.1, 2.0
FLOPS_PER_CYCLE   = 1.0     # q_n = 1 (paper). Do NOT raise this — Φ=MACs already
                            # matches the papers; q>1 would double-count.
KAPPA             = 1.0e-28
ENERGY_SCOPE      = "device"

# ---- system heterogeneity: wireless (coherent base, shared by all runs) ----
BANDWIDTH_HZ           = 50.0e6
DIST_MIN_M, DIST_MAX_M = 100.0, 1000.0
DEV_TX_POWER_MIN_W, DEV_TX_POWER_MAX_W = 0.1, 0.2
BS_DOWNLINK_POWER_W    = 0.3
NOISE_PSD_DBM_PER_HZ   = -150.0
PATH_FADING_EXPONENT   = 1.3
H0_CHANNEL_CONST       = 1.0e-6
MIN_SNR_DB             = 0.0

# ---- FedAsync ----
ASYNC_ALPHA     = 0.5
ASYNC_STALENESS = "polynomial"    # α_t = α·(staleness+1)^(-a)
ASYNC_POLY_A    = 0.5
ASYNC_WINDOW    = NUM_CLIENTS      # in-flight concurrency (pure async)

# ---- targets for the time-to-accuracy table ----
ACC_TARGETS = [0.40, 0.50, 0.60, 0.70]

METHOD_ORDER = ["FL (IID)", "FedAsync (IID)", "FL (non-IID)", "FedAsync (non-IID)"]


# Φ (per-sample compute workload) is measured AUTOMATICALLY from the model by
# the framework (system.cycles_per_sample_mode="model_macs", the default) — it
# equals the model's MACs, which is the paper convention (a model's quoted
# "FLOPs" is its MAC count; ResNet-34 @ 64×64 = 0.30e9). No manual FLOP-setting.


# Base shared by every run (problem, local work, system heterogeneity). Only the
# data partition (IID vs Dirichlet) and global_rounds are added per-run.
SHARED_OVERRIDES = {
    "data.dataset":               DATASET,
    "data.model_name":            MODEL,
    "data.image_size":            IMAGE_SIZE,
    "data.num_clients":           NUM_CLIENTS,
    "learning.clients_per_round": NUM_CLIENTS,      # full participation (FL)
    "learning.batch_size":        BATCH_SIZE,
    "learning.learning_rate":     LEARNING_RATE,
    "learning.local_iters":       LOCAL_ITERS,      # H — same local work for FL & async
    "learning.local_epochs":      1,
    "evaluation.evaluate_every":  EVALUATE_EVERY,
    # ---- device compute (clock × FLOPs-per-cycle; heterogeneous) ----
    "system.cpu_freq_mode":         "uniform_ghz",
    "system.cpu_freq_min_ghz":      DEV_FREQ_MIN_GHZ,
    "system.cpu_freq_max_ghz":      DEV_FREQ_MAX_GHZ,
    # Φ (cycles_per_sample) auto-measured from the model — mode defaults to
    # "model_macs"; no manual override needed.
    "system.flops_per_cycle":       FLOPS_PER_CYCLE,        # q = 1 (paper); effective FLOP/s = f·q
    "system.switched_capacitance":  KAPPA,
    "system.energy_scope":          ENERGY_SCOPE,
    # ---- wireless (heterogeneous channel) ----
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
    "wireless.upload_size_mode":         "model",     # real ResNet-34 update size
    # ---- FedAsync knobs (ignored by the sync FL runs) ----
    "async_fl.alpha":       ASYNC_ALPHA,
    "async_fl.window_size": ASYNC_WINDOW,
}

# Data-heterogeneity axis: IID vs high non-IID (Dirichlet α small).
DATA_SETTINGS = {
    "IID":     {"data.partition": "iid"},
    "non-IID": {"data.partition": "dirichlet", "data.dirichlet_alpha": NONIID_ALPHA},
}


class FLvsFedAsync(AsyncExperiment):
    """FL (sync FedAvg) vs FedAsync (polynomial staleness) at two data-
    heterogeneity levels. Each run auto-emits its unified CSV + standard plots
    (framework reporting); this class adds the cross-method comparison plots."""

    def run(self):
        results = {}
        for data_tag, data_ovr in DATA_SETTINGS.items():
            ovr = {**SHARED_OVERRIDES, **data_ovr,
                   "learning.global_rounds": MAX_GLOBAL_ROUNDS}
            slug = data_tag.lower().replace("-", "")

            # ---- FL: synchronous FedAvg ----
            results[f"FL ({data_tag})"] = self.run_single(
                f"fl_{slug}", label=f"FL ({data_tag})",
                config_overrides=ovr,
                components={"algorithm": FedAvg()},
            )

            # ---- FedAsync: pure async, polynomial staleness ----
            results[f"FedAsync ({data_tag})"] = self.run_single_async(
                f"fedasync_{slug}", label=f"FedAsync ({data_tag})",
                config_overrides=ovr,
                components={"algorithm": FedAsync(
                    alpha=ASYNC_ALPHA, staleness_func=ASYNC_STALENESS, a=ASYNC_POLY_A)},
            )

        self._comparison_plots(results)
        self._staleness_plot(results)
        self._scalar_bars(results)
        self._time_to_accuracy_table(results)
        return results

    # ------------------------------------------------------------------
    # Cross-method comparison plots (accuracy vs round / time, energy)
    # ------------------------------------------------------------------

    def _comparison_plots(self, results):
        ordered = {k: results[k] for k in METHOD_ORDER if k in results}
        self.plot_comparison(
            ordered,
            plot_configs=[
                {"metric": "test_accuracy", "x": "round",
                 "ylabel": "Test accuracy", "title": "Accuracy vs round — ResNet-34 / CIFAR-10"},
                {"metric": "test_accuracy", "x": "simulated_time_s",
                 "ylabel": "Test accuracy", "title": "Accuracy vs simulated time"},
                {"metric": "cumulative_energy_j", "x": "round",
                 "ylabel": "Cumulative energy (J)", "title": "Cumulative device energy"},
            ],
            out_prefix="cmp",
        )

    # ------------------------------------------------------------------
    # Staleness (async-only) — one curve per FedAsync run
    # ------------------------------------------------------------------

    def _staleness_plot(self, results):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        drew = False
        for label, r in results.items():
            if "staleness" not in r.df.columns:
                continue   # sync FL has no staleness
            df = r.df.dropna(subset=["staleness"])
            if df.empty:
                continue
            ax.plot(df["round"], df["staleness"], marker=".", markersize=3,
                    linewidth=1.0, alpha=0.8, label=label)
            drew = True
        if not drew:
            plt.close(fig)
            return
        ax.set_xlabel("Aggregation index"); ax.set_ylabel("Staleness (versions)")
        ax.set_title("FedAsync staleness"); ax.grid(True, alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "cmp_staleness.png"), dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Scalar bars: best accuracy, total energy, total simulated time
    # ------------------------------------------------------------------

    def _scalar_bars(self, results):
        ordered = {k: results[k] for k in METHOD_ORDER if k in results}
        self.plot_bar(ordered, metric="best_accuracy",
                      ylabel="Best test accuracy", out_name="bar_best_accuracy")
        self.plot_bar(ordered, metric="total_energy_j",
                      ylabel="Total device energy (J)", out_name="bar_total_energy")
        self.plot_bar(ordered, metric="total_simulated_time_s",
                      ylabel="Total simulated time (s)", out_name="bar_total_time")

    # ------------------------------------------------------------------
    # Time-to-accuracy table (simulated seconds to first reach each target)
    # ------------------------------------------------------------------

    def _time_to_accuracy_table(self, results):
        def _t_to_acc(r, target):
            df = r.df.dropna(subset=["test_accuracy"])
            hit = df[df["test_accuracy"] >= target]
            return None if hit.empty else float(hit.iloc[0]["simulated_time_s"])

        lines = ["=" * 82, "  TIME-TO-ACCURACY (simulated seconds)"]
        present = [k for k in METHOD_ORDER if k in results]
        lines.append("    target " + "".join(f"{k:>18}" for k in present))
        for tgt in ACC_TARGETS:
            row = f"    {int(tgt*100):>4}%   "
            for k in present:
                t = _t_to_acc(results[k], tgt)
                row += f"{('--' if t is None else f'{t:,.0f}'):>18}"
            lines.append(row)
        lines.append("=" * 82)
        text = "\n".join(lines)
        print("\n" + text + "\n")
        with open(os.path.join(self.output_dir, "time_to_accuracy.txt"), "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    import sys

    # Quick local smoke: `python examples/fl_vs_fedasync_experiment.py --smoke`
    # shrinks the problem so it runs on a laptop; drop it for the real run.
    if "--smoke" in sys.argv:
        NUM_CLIENTS = 4
        MAX_GLOBAL_ROUNDS = 4
        EVALUATE_EVERY = 2
        SHARED_OVERRIDES["data.num_clients"] = NUM_CLIENTS
        SHARED_OVERRIDES["learning.clients_per_round"] = NUM_CLIENTS
        SHARED_OVERRIDES["async_fl.window_size"] = NUM_CLIENTS
        SHARED_OVERRIDES["evaluation.evaluate_every"] = EVALUATE_EVERY

    FLvsFedAsync(base_config=BASE_CONFIG, output_dir=OUTPUT_DIR).run()
