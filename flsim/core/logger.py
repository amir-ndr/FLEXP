"""
core/logger.py: CSV logger and result plotter for federated learning experiments.

Writes one row per round to a CSV file with full system metrics.
Also provides plot_results() to generate accuracy-vs-round and
accuracy-vs-simulated-time plots saved to the outputs/ directory.
"""

import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in headless environments
import matplotlib.pyplot as plt


# CSV column order (matches the spec exactly)
_CSV_COLUMNS = [
    "round",
    "simulated_time_s",
    "test_accuracy",
    "test_loss",
    "round_duration_s",
    "mean_compute_time_s",
    "max_compute_time_s",
    "mean_upload_time_s",
    "max_upload_time_s",
    "mean_download_time_s",
    "max_download_time_s",
    "mean_compute_energy_j",
    "total_energy_j",
    "cumulative_energy_j",
    "mean_channel_gain",
    "mean_rate_bps",
    "num_selected_clients",
    "selected_client_ids",
]


@dataclass
class RoundResult:
    """Aggregated metrics for one communication round."""
    round_idx: int

    # Duration (seconds, simulated)
    round_duration_s: float

    # Per-client timing statistics
    compute_times_s:  List[float]
    upload_times_s:   List[float]
    download_times_s: List[float]

    # Per-client energy statistics
    compute_energies_j: List[float]
    tx_energies_j:      List[float]

    # Channel statistics
    channel_gains: List[float]
    rates_bps:     List[float]

    # Client selection info
    selected_client_ids: List[int]

    # Training stats
    train_losses: List[float]

    @property
    def mean_compute_time_s(self)  -> float: return _mean(self.compute_times_s)
    @property
    def max_compute_time_s(self)   -> float: return max(self.compute_times_s, default=0.0)
    @property
    def mean_upload_time_s(self)   -> float: return _mean(self.upload_times_s)
    @property
    def max_upload_time_s(self)    -> float: return max(self.upload_times_s, default=0.0)
    @property
    def mean_download_time_s(self) -> float: return _mean(self.download_times_s)
    @property
    def max_download_time_s(self)  -> float: return max(self.download_times_s, default=0.0)
    @property
    def mean_compute_energy_j(self) -> float: return _mean(self.compute_energies_j)
    @property
    def total_energy_j(self)        -> float: return sum(self.compute_energies_j) + sum(self.tx_energies_j)
    @property
    def mean_channel_gain(self)     -> float: return _mean(self.channel_gains)
    @property
    def mean_rate_bps(self)         -> float: return _mean(self.rates_bps)
    @property
    def num_selected_clients(self)  -> int:   return len(self.selected_client_ids)


class Logger:
    """
    Experiment logger.

    Writes one CSV row per round. Optionally logs evaluation metrics when
    they are available. Provides plot_results() for post-run visualisation.

    This class does NOT:
    - Perform model evaluation.
    - Compute simulated time or energy.
    - Log anything outside the defined CSV columns.
    """

    def __init__(self, output_dir: str, experiment_name: str):
        """
        Args:
            output_dir (str): directory where CSV and plots are written.
            experiment_name (str): used as the base filename.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.csv_path = os.path.join(output_dir, f"{experiment_name}.csv")
        self._rows = []   # in-memory cache for plotting
        self._cum_energy_j = 0.0   # running total energy across all rounds

        # Open CSV and write header
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            writer.writeheader()

        print(f"[Logger] Writing to {self.csv_path}")

    def log_round(
        self,
        round_idx: int,
        simulated_time_s: float,
        round_result: RoundResult,
        eval_result=None,   # EvalResult | None
    ) -> None:
        """
        Append one CSV row for the given round.

        Args:
            round_idx (int): 0-based round index.
            simulated_time_s (float): cumulative simulated time in seconds.
            round_result (RoundResult): system metrics for this round.
            eval_result: EvalResult if evaluation was run this round, else None.
        """
        self._cum_energy_j += round_result.total_energy_j

        row = {
            "round":                  round_idx,
            "simulated_time_s":       f"{simulated_time_s:.4f}",
            "test_accuracy":          f"{eval_result.test_accuracy:.6f}" if eval_result else "",
            "test_loss":              f"{eval_result.test_loss:.6f}"     if eval_result else "",
            "round_duration_s":       f"{round_result.round_duration_s:.4f}",
            "mean_compute_time_s":    f"{round_result.mean_compute_time_s:.4f}",
            "max_compute_time_s":     f"{round_result.max_compute_time_s:.4f}",
            "mean_upload_time_s":     f"{round_result.mean_upload_time_s:.4f}",
            "max_upload_time_s":      f"{round_result.max_upload_time_s:.4f}",
            "mean_download_time_s":   f"{round_result.mean_download_time_s:.4f}",
            "max_download_time_s":    f"{round_result.max_download_time_s:.4f}",
            "mean_compute_energy_j":  f"{round_result.mean_compute_energy_j:.6e}",
            "total_energy_j":         f"{round_result.total_energy_j:.6e}",
            "cumulative_energy_j":    f"{self._cum_energy_j:.6e}",
            "mean_channel_gain":      f"{round_result.mean_channel_gain:.6e}",
            "mean_rate_bps":          f"{round_result.mean_rate_bps:.2f}",
            "num_selected_clients":   round_result.num_selected_clients,
            "selected_client_ids":    str(round_result.selected_client_ids),
        }

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            writer.writerow(row)

        self._rows.append({
            "round":                 round_idx,
            "simulated_time_s":      simulated_time_s,
            "test_accuracy":         eval_result.test_accuracy if eval_result else None,
            "test_loss":             eval_result.test_loss     if eval_result else None,
            "round_duration_s":      round_result.round_duration_s,
            "mean_compute_time_s":   round_result.mean_compute_time_s,
            "max_compute_time_s":    round_result.max_compute_time_s,
            "mean_upload_time_s":    round_result.mean_upload_time_s,
            "max_upload_time_s":     round_result.max_upload_time_s,
            "mean_compute_energy_j": round_result.mean_compute_energy_j,
            "total_energy_j":        round_result.total_energy_j,
            "cumulative_energy_j":   self._cum_energy_j,
            "mean_channel_gain":     round_result.mean_channel_gain,
            "mean_rate_bps":         round_result.mean_rate_bps,
            "mean_train_loss":       _mean(round_result.train_losses),
        })

    def plot_results(self) -> None:
        """
        Generate and save plots to the output directory.

        Plots produced (all rounds unless noted):
          1. Test accuracy vs communication round          [eval rounds only]
          2. Test accuracy vs simulated time               [eval rounds only]
          3. Test loss vs communication round              [eval rounds only]
          4. Training loss vs communication round          [all rounds]
          5. Round duration — mean compute / upload / total  [all rounds]
          6. Per-round energy — compute vs TX              [all rounds]
          7. Cumulative energy vs round                    [all rounds]
          8. Mean channel gain vs round                    [all rounds]
          9. Mean achievable rate (Mbps) vs round          [all rounds]
        """
        rows      = self._rows
        all_r     = [r["round"] for r in rows]
        eval_rows = [r for r in rows if r["test_accuracy"] is not None]

        def _savefig(fig, suffix):
            path = os.path.join(self.output_dir, f"{self.experiment_name}_{suffix}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[Logger] Saved {path}")

        # ------------------------------------------------------------------
        # 1. Accuracy vs round
        # ------------------------------------------------------------------
        if eval_rows:
            e_rounds = [r["round"] for r in eval_rows]
            accs     = [r["test_accuracy"] for r in eval_rows]
            fig, ax  = plt.subplots()
            ax.plot(e_rounds, accs, marker="o", linewidth=1.5, markersize=4)
            ax.set_xlabel("Communication round")
            ax.set_ylabel("Test accuracy")
            ax.set_title(f"{self.experiment_name} — accuracy vs round")
            ax.grid(True, alpha=0.3)
            _savefig(fig, "acc_vs_round")

        # ------------------------------------------------------------------
        # 2. Accuracy vs simulated time
        # ------------------------------------------------------------------
        if eval_rows:
            e_times = [r["simulated_time_s"] for r in eval_rows]
            fig, ax = plt.subplots()
            ax.plot(e_times, accs, marker="o", linewidth=1.5, markersize=4, color="tab:orange")
            ax.set_xlabel("Simulated time (s)")
            ax.set_ylabel("Test accuracy")
            ax.set_title(f"{self.experiment_name} — accuracy vs simulated time")
            ax.grid(True, alpha=0.3)
            _savefig(fig, "acc_vs_time")

        # ------------------------------------------------------------------
        # 3. Test loss vs round
        # ------------------------------------------------------------------
        if eval_rows:
            losses = [r["test_loss"] for r in eval_rows]
            fig, ax = plt.subplots()
            ax.plot(e_rounds, losses, marker="o", linewidth=1.5, markersize=4, color="tab:red")
            ax.set_xlabel("Communication round")
            ax.set_ylabel("Test loss")
            ax.set_title(f"{self.experiment_name} — test loss vs round")
            ax.grid(True, alpha=0.3)
            _savefig(fig, "test_loss_vs_round")

        # ------------------------------------------------------------------
        # 4. Training loss vs round (all rounds)
        # ------------------------------------------------------------------
        train_losses = [r["mean_train_loss"] for r in rows]
        fig, ax = plt.subplots()
        ax.plot(all_r, train_losses, linewidth=1.5, color="tab:purple")
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Mean training loss")
        ax.set_title(f"{self.experiment_name} — training loss vs round")
        ax.grid(True, alpha=0.3)
        _savefig(fig, "train_loss_vs_round")

        # ------------------------------------------------------------------
        # 5. Round timing breakdown (mean compute, mean upload, total)
        # ------------------------------------------------------------------
        t_comp  = [r["mean_compute_time_s"] for r in rows]
        t_up    = [r["mean_upload_time_s"]  for r in rows]
        t_max   = [r["max_compute_time_s"]  for r in rows]
        t_total = [r["round_duration_s"]    for r in rows]

        fig, ax = plt.subplots()
        ax.plot(all_r, t_comp,  label="Mean compute",   linewidth=1.5)
        ax.plot(all_r, t_up,    label="Mean upload",    linewidth=1.5)
        ax.plot(all_r, t_max,   label="Max compute",    linewidth=1.5, linestyle="--")
        ax.plot(all_r, t_total, label="Round duration", linewidth=1.5, linestyle=":")
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Simulated time (s)")
        ax.set_title(f"{self.experiment_name} — timing breakdown")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _savefig(fig, "timing_breakdown")

        # ------------------------------------------------------------------
        # 6. Per-round energy: compute vs TX (stacked bar, sampled if many rounds)
        # ------------------------------------------------------------------
        e_comp = [r["mean_compute_energy_j"] for r in rows]
        # TX energy per round = total - compute (total already sums both)
        e_total_list = [r["total_energy_j"] for r in rows]
        e_tx   = [et - ec for et, ec in zip(e_total_list, e_comp)]

        # Subsample for readability if > 50 rounds
        step = max(1, len(all_r) // 50)
        r_s   = all_r[::step]
        ec_s  = e_comp[::step]
        etx_s = e_tx[::step]

        fig, ax = plt.subplots()
        ax.bar(r_s, ec_s,  label="Compute",      width=step * 0.8)
        ax.bar(r_s, etx_s, label="Transmission", width=step * 0.8, bottom=ec_s)
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Energy per client per round (J)")
        ax.set_title(f"{self.experiment_name} — energy breakdown")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        _savefig(fig, "energy_breakdown")

        # ------------------------------------------------------------------
        # 7. Cumulative total energy vs round
        # ------------------------------------------------------------------
        cum_energy = []
        total = 0.0
        for r in rows:
            total += r["total_energy_j"]
            cum_energy.append(total)

        fig, ax = plt.subplots()
        ax.plot(all_r, cum_energy, linewidth=1.5, color="tab:brown")
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Cumulative energy (J)")
        ax.set_title(f"{self.experiment_name} — cumulative energy")
        ax.grid(True, alpha=0.3)
        _savefig(fig, "cumulative_energy")

        # ------------------------------------------------------------------
        # 8. Mean channel gain vs round
        # ------------------------------------------------------------------
        gains = [r["mean_channel_gain"] for r in rows]
        fig, ax = plt.subplots()
        ax.plot(all_r, gains, linewidth=1.0, color="tab:cyan", alpha=0.8)
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Mean channel gain (linear)")
        ax.set_yscale("log")
        ax.set_title(f"{self.experiment_name} — channel gain vs round")
        ax.grid(True, alpha=0.3, which="both")
        _savefig(fig, "channel_gain_vs_round")

        # ------------------------------------------------------------------
        # 9. Mean achievable rate vs round
        # ------------------------------------------------------------------
        rates_mbps = [r["mean_rate_bps"] / 1e6 for r in rows]
        fig, ax = plt.subplots()
        ax.plot(all_r, rates_mbps, linewidth=1.0, color="tab:olive", alpha=0.8)
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Mean rate (Mbps)")
        ax.set_title(f"{self.experiment_name} — achievable rate vs round")
        ax.grid(True, alpha=0.3)
        _savefig(fig, "rate_vs_round")

        if not eval_rows:
            print("[Logger] No evaluation data — accuracy/loss plots skipped.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0
