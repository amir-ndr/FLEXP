"""
examples/ota_experiment.py: FedOTA (over-the-air aggregation) vs digital FedAvg.

Demonstrates the trade-off at the heart of Yang, Jiang, Shi & Ding,
"Federated Learning via Over-the-Air Computation" (arXiv:1812.11750) §II-A:
looser MSE budgets let more devices participate (helping convergence) at the
cost of more aggregation noise. Compares:

  1. FedAvg (digital) — the paper's "Benchmark": all devices participate, exact
                       weighted averaging, no aggregation error (orthogonal
                       FDMA upload).
  2. FedOTA, gamma=-3dB — tight MSE budget: few devices selected, low noise.
  3. FedOTA, gamma=+5dB — the paper's operating point (§VI-C uses gamma=5dB).
  4. FedOTA, gamma=+10dB — loose budget: most devices selected, more noise.

Relationship to the paper's experiment (honest scope)
-------------------------------------------------------
This is NOT a literal reproduction of the paper's Figure 7. It shares the
core OTA aggregation mechanism and the "Benchmark vs OTA-with-device-selection"
comparison, but differs in three ways:
  * N=1 antenna (this framework's ChannelModel is a scalar real power gain).
    The paper's N=6-antenna DC/SDP device-selection algorithm (its main
    contribution) is not implemented — see flsim/system/ota.py's scope note.
    With N=1, device selection is a closed-form per-device MSE threshold, so
    the paper's algorithm comparison (l1+SDR / Reweighted+SDR / Proposed DC)
    collapses to a single trivial rule and there is nothing to compare among
    them — hence we sweep gamma instead of comparing selection algorithms.
  * MNIST CNN, not the paper's SVM-on-CIFAR-10 (no SVM in this framework).
    To try CIFAR-10 instead, point BASE_CONFIG at a cifar10 config; the OTA
    parameters below would then need re-calibrating (see below).
  * 30 devices, not 20.

gamma (dB) values and the noise_power_w below were calibrated for THIS
config's actual channel_gain() scale (flsim/configs/mnist_fedavg.yaml, 30
clients, exp_fading channel) — NOT copied from the paper's own numbers,
which assume a unit-variance Rayleigh channel unrelated to this framework's
path-loss-based gains. See the calibration method in the comment below;
re-run it if you change num_clients, the channel model, or its parameters.

Standard accuracy/loss/energy plots come from the usual RunResult/CSV
machinery. FedOTA overrides the Simulator's uplink physics
(recompute_uplink_physics), so the CSV's upload_time_s / tx_energy_j /
total_energy_j / cumulative_energy_j columns already reflect OTA (constant
simultaneous upload, squared-norm transmit energy) — not FDMA. A few OTA
quantities still have no CSV column (the achieved aggregation MSE, the count
of devices selected, and the per-device energy breakdown); those are read
directly off each FedOTA instance's .mse_history / .energy_history /
.excluded_history after the run and plotted separately below.

Energy caveat: the cumulative-energy comparison plot puts FedAvg (FDMA p*t
energy) and FedOTA (AirComp squared-norm energy) on the same Joule axis. They
are genuine Joules but computed under two different physical layers, so read
absolute cross-family differences with that in mind; within each family the
comparison is clean.

Run from the repo root:
    python examples/ota_experiment.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flsim.algorithms.fedavg import FedAvg
from flsim.algorithms.fedota import FedOTA
from flsim.channel.conversions import dbm_to_watts
from flsim.experiments.base import Experiment


BASE_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "flsim", "configs", "mnist_fedavg.yaml"
)
OUTPUT_DIR = "outputs/ota_experiment/"

# ---------------------------------------------------------------------------
# OTA parameter calibration for THIS config (30 clients, exp_fading channel,
# ~2000 samples/client). Method:
#   1. p0_w = dbm_to_watts(wireless.tx_power_dbm)  — reuse the existing
#      per-device transmit power budget from the wireless config.
#   2. Draw channel_gain() for all client profiles, compute
#      score_i = phi_i^2 / gain_i  (phi_i = num_samples).
#   3. Pick noise_power_w so that gamma=0dB (linear gamma=1) selects roughly
#      the median device: noise_power_w = p0_w / median(score).
#   4. gamma_db then sweeps which fraction of devices clear the threshold —
#      here -3/+3/+10 dB gives roughly 10/30, 21/30, 28/30 selected (see the
#      calibration script output this was checked against).
# Re-run this calibration (flsim.system.ota.OTAChannel(...).mse(...) against
# the full candidate pool) if num_clients / channel model / channel params change.
# ---------------------------------------------------------------------------
P0_W = dbm_to_watts(10.0)
NOISE_POWER_W = 1.03e-20
GAMMA_DB_VALUES = [-3.0, 5.0, 10.0]   # 5 dB is the paper's operating point (§VI-C)

NUM_CLIENTS = 30           # matches data.num_clients in the config
GLOBAL_ROUNDS = 30
EVALUATE_EVERY = 2


class OTAExperiment(Experiment):
    """Compare FedOTA at several MSE budgets against digital FedAvg."""

    def run(self):
        results = {}
        ota_algorithms = {}   # label -> FedOTA instance, for reading histories after the run

        # ------------------------------------------------------------------
        # 1. Digital FedAvg baseline — full participation (matches
        #    data.num_clients=30), no aggregation noise.
        # ------------------------------------------------------------------
        results["FedAvg (digital)"] = self.run_single(
            run_name="ota_fedavg_baseline",
            label="FedAvg (digital)",
            config_overrides={
                "learning.global_rounds": GLOBAL_ROUNDS,
                "learning.clients_per_round": 30,
                "evaluation.evaluate_every": EVALUATE_EVERY,
            },
            components={"algorithm": FedAvg()},
        )

        # ------------------------------------------------------------------
        # 2-4. FedOTA at increasing MSE budgets (tight -> loose).
        #    clients_per_round is irrelevant for FedOTA (it selects every
        #    feasible device, see FedOTA.select_clients) but Simulator still
        #    reads it for the bandwidth-context passed to select_clients —
        #    harmless since FedOTA ignores it.
        # ------------------------------------------------------------------
        for gamma_db in GAMMA_DB_VALUES:
            label = f"FedOTA (gamma={gamma_db:+.0f}dB)"
            alg = FedOTA(p0_w=P0_W, noise_power_w=NOISE_POWER_W, gamma_db=gamma_db, seed=0)
            ota_algorithms[label] = alg
            results[label] = self.run_single(
                run_name=f"ota_gamma_{gamma_db:+.0f}db".replace("+", "p").replace("-", "m"),
                label=label,
                config_overrides={
                    "learning.global_rounds": GLOBAL_ROUNDS,
                    "learning.clients_per_round": 30,
                    "evaluation.evaluate_every": EVALUATE_EVERY,
                },
                components={"algorithm": alg},
            )

        # ------------------------------------------------------------------
        # Standard plots. For the FedOTA runs the CSV energy/time columns now
        # reflect true OTA physics (recompute_uplink_physics); for the FedAvg
        # run they are the usual FDMA values — so the energy axis mixes two
        # physical layers and is only meaningful within each algorithm family.
        # ------------------------------------------------------------------
        print("\n[OTAExperiment] Generating comparison plots …")
        self.plot_comparison(
            results,
            plot_configs=[
                {"metric": "test_accuracy", "x": "round",
                 "ylabel": "Test Accuracy",
                 "title": "FedOTA vs FedAvg — Accuracy vs Round"},
                # Cumulative energy for ALL runs (FedAvg baseline + every FedOTA
                # budget). cumulative_energy_j is written by the logger each
                # round; for FedOTA it is the OTA squared-norm energy, for
                # FedAvg the FDMA p*t energy (see the module docstring caveat).
                {"metric": "cumulative_energy_j", "x": "round",
                 "ylabel": "Cumulative energy (J)",
                 "title": "FedOTA vs FedAvg — cumulative energy vs round"},
            ],
            out_prefix="ota_comparison",
        )
        self.plot_bar(
            results,
            metric="best_accuracy",
            ylabel="Best test accuracy",
            out_name="ota_best_acc_bar",
            title="FedOTA (by MSE budget) vs FedAvg — best accuracy",
        )
        # Total end-of-run energy across all runs, as a bar chart.
        self.plot_bar(
            results,
            metric="total_energy_j",
            ylabel="Total energy over run (J)",
            out_name="ota_total_energy_bar",
            title="FedOTA (by MSE budget) vs FedAvg — total energy",
        )

        # ------------------------------------------------------------------
        # OTA-specific plots — read directly from each FedOTA instance's
        # accumulated histories (not from the CSV).
        # ------------------------------------------------------------------
        self._plot_mse_vs_round(ota_algorithms)
        self._plot_devices_selected_vs_round(ota_algorithms)
        self._plot_total_energy_vs_round(ota_algorithms)

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for label, r in results.items():
            extra = ""
            if label in ota_algorithms:
                alg = ota_algorithms[label]
                avg_dev = sum(len(e) for e in alg.energy_history) / len(alg.energy_history)
                avg_mse = sum(alg.mse_history) / len(alg.mse_history)
                extra = f"  avg_devices={avg_dev:.1f}/{NUM_CLIENTS}  avg_mse={avg_mse:.4g}"
            print(
                f"  {label:<24s}  best={r.best_accuracy:.4f}  "
                f"final={r.final_accuracy:.4f}  total_energy={r.total_energy_j:.4g}J{extra}"
            )
        print("=" * 60)

    # ------------------------------------------------------------------
    # Custom OTA plots
    # ------------------------------------------------------------------

    def _plot_mse_vs_round(self, ota_algorithms: dict) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, alg in ota_algorithms.items():
            ax.plot(range(len(alg.mse_history)), alg.mse_history, marker="o",
                    markersize=3, linewidth=1.5, label=label)
        ax.set_xlabel("Round")
        ax.set_ylabel("Achieved aggregation MSE(S)")
        ax.set_yscale("log")
        ax.set_title("FedOTA — achieved MSE per round vs MSE budget gamma")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        self._save(fig, "ota_mse_vs_round")

    def _plot_devices_selected_vs_round(self, ota_algorithms: dict) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, alg in ota_algorithms.items():
            counts = [len(e) for e in alg.energy_history]
            ax.plot(range(len(counts)), counts, marker="o",
                    markersize=3, linewidth=1.5, label=label)
        ax.set_xlabel("Round")
        ax.set_ylabel("Number of devices selected")
        ax.set_title("FedOTA — participating devices per round vs MSE budget gamma")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        self._save(fig, "ota_devices_selected_vs_round")

    def _plot_total_energy_vs_round(self, ota_algorithms: dict) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, alg in ota_algorithms.items():
            totals = [sum(e.values()) for e in alg.energy_history]
            ax.plot(range(len(totals)), totals, marker="o",
                    markersize=3, linewidth=1.5, label=label)
        ax.set_xlabel("Round")
        ax.set_ylabel("Total OTA transmission energy (J)")
        ax.set_title("FedOTA — true AirComp transmission energy per round (paper eq. 5)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        self._save(fig, "ota_total_energy_vs_round")


if __name__ == "__main__":
    OTAExperiment(
        base_config=BASE_CONFIG,
        output_dir=OUTPUT_DIR,
    ).run()
