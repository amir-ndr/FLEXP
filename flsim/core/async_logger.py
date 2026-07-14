"""
core/async_logger.py: Logger for asynchronous federated learning experiments.

One CSV row per global epoch (= one server model update, from one client in
fully-async mode, or from a buffered batch of `k` clients in semi-async mode —
see AsyncFederatedAlgorithm.buffer_size). Extends the sync Logger columns with
async-specific fields:
  staleness  — t - tau (how stale the arriving update is; max over the batch
               when buffer_size > 1)
  alpha_used — effective mixing weight alpha_t for this epoch

Also produces async-specific plots:
  staleness_vs_epoch  — tracks update freshness over time
  alpha_vs_epoch      — tracks the effective mixing weight over time
  acc_vs_epoch        — standard accuracy curve (vs global epoch count)
  acc_vs_time         — accuracy vs simulated wall-clock time
  train_loss_vs_epoch — training loss of each arriving update
  energy_vs_epoch     — per-epoch energy cost
"""

import csv
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_ASYNC_CSV_COLUMNS = [
    "global_epoch",
    "simulated_time_s",
    "test_accuracy",
    "test_loss",
    "staleness",
    "alpha_used",
    "client_id",
    "compute_time_s",
    "upload_time_s",
    "total_time_s",
    "compute_energy_j",
    "tx_energy_j",
    "total_energy_j",
    "cumulative_energy_j",
    "channel_gain",
    "achievable_rate_bps",
    "train_loss",
]


@dataclass
class AsyncRoundResult:
    """
    All metrics produced by one global epoch in async FL.

    One arriving ClientUpdate triggers one server update = one global epoch
    (fully async, buffer_size=1 — the default). This struct bundles the
    timing, energy, channel, and staleness info for that single event so the
    logger can record it.

    Semi-async (buffer_size=k>1): one epoch instead aggregates k arriving
    ClientUpdates together (see AsyncSimulator / aggregate_buffered). In that
    case:
      client_id            — list[int] of the k client ids in the batch
                              (instead of a single int)
      compute/upload/total_time_s — max over the batch (the bottleneck that
                              gated this epoch, same convention as the sync
                              Simulator's per-round max)
      *_energy_j fields     — sum over the batch (total energy spent this
                              epoch, same convention as the sync Simulator's
                              per-round sum)
      channel_gain, achievable_rate_bps, train_loss — mean over the batch
      staleness             — max over the batch (see AsyncSimulator)
    """
    global_epoch:        int
    arrival_time_s:      float   # simulated time the update reached the server
    staleness:           int     # t - tau  (0 = fresh, >0 = stale); max over batch if buffered
    alpha_used:          float   # effective alpha_t = base_alpha * s(staleness)

    # From the arriving ClientUpdate(s) — see class docstring for the
    # single-vs-buffered convention of each field.
    client_id:           object  # int, or list[int] when buffer_size > 1
    compute_time_s:      float
    upload_time_s:       float
    total_time_s:        float
    compute_energy_j:    float
    tx_energy_j:         float
    total_energy_j:      float
    channel_gain:        float
    achievable_rate_bps: float
    train_loss:          float


class AsyncLogger:
    """
    Logger for async FL experiments.

    Mirrors the sync Logger interface but writes one row per global epoch
    (not per round) and adds staleness/alpha columns.

    This class does NOT:
    - Perform model evaluation.
    - Compute simulated time or energy.
    """

    def __init__(self, output_dir: str, experiment_name: str):
        """
        Args:
            output_dir (str):      directory for CSV and plot files.
            experiment_name (str): base filename (no extension).
        """
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir       = output_dir
        self.experiment_name  = experiment_name
        self.csv_path         = os.path.join(output_dir, f"{experiment_name}.csv")
        self._rows            = []   # in-memory cache for plotting
        self._cum_energy_j    = 0.0  # running total energy across all epochs

        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_ASYNC_CSV_COLUMNS).writeheader()

        print(f"[AsyncLogger] Writing to {self.csv_path}")

    def log_epoch(
        self,
        global_epoch: int,
        simulated_time_s: float,
        result: AsyncRoundResult,
        eval_result=None,   # EvalResult | None
    ) -> None:
        """
        Append one CSV row for this global epoch.

        Args:
            global_epoch (int):      0-based epoch index.
            simulated_time_s (float): virtual clock at this epoch.
            result (AsyncRoundResult): metrics for the arriving update.
            eval_result: EvalResult if evaluation ran this epoch, else None.
        """
        self._cum_energy_j += result.total_energy_j

        row = {
            "global_epoch":       global_epoch,
            "simulated_time_s":   f"{simulated_time_s:.4f}",
            "test_accuracy":      f"{eval_result.test_accuracy:.6f}" if eval_result else "",
            "test_loss":          f"{eval_result.test_loss:.6f}"     if eval_result else "",
            "staleness":          result.staleness,
            "alpha_used":         f"{result.alpha_used:.6f}",
            "client_id":          result.client_id,
            "compute_time_s":     f"{result.compute_time_s:.4f}",
            "upload_time_s":      f"{result.upload_time_s:.4f}",
            "total_time_s":       f"{result.total_time_s:.4f}",
            "compute_energy_j":   f"{result.compute_energy_j:.6e}",
            "tx_energy_j":        f"{result.tx_energy_j:.6e}",
            "total_energy_j":     f"{result.total_energy_j:.6e}",
            "cumulative_energy_j": f"{self._cum_energy_j:.6e}",
            "channel_gain":       f"{result.channel_gain:.6e}",
            "achievable_rate_bps": f"{result.achievable_rate_bps:.2f}",
            "train_loss":         f"{result.train_loss:.6f}",
        }

        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_ASYNC_CSV_COLUMNS).writerow(row)

        self._rows.append({
            "global_epoch":     global_epoch,
            "simulated_time_s": simulated_time_s,
            "test_accuracy":    eval_result.test_accuracy if eval_result else None,
            "test_loss":        eval_result.test_loss     if eval_result else None,
            "staleness":        result.staleness,
            "alpha_used":       result.alpha_used,
            "client_id":        result.client_id,
            "compute_time_s":   result.compute_time_s,
            "upload_time_s":    result.upload_time_s,
            "total_energy_j":   result.total_energy_j,
            "cumulative_energy_j": self._cum_energy_j,
            "train_loss":       result.train_loss,
        })

    def plot_results(self) -> None:
        """
        Generate and save all async plots to output_dir.

        Plots produced:
          1. Test accuracy vs global epoch        [eval epochs only]
          2. Test accuracy vs simulated time      [eval epochs only]
          3. Test loss vs global epoch            [eval epochs only]
          4. Training loss vs global epoch        [all epochs]
          5. Staleness vs global epoch            [all epochs]
          6. Effective alpha_t vs global epoch    [all epochs]
          7. Energy per epoch vs global epoch     [all epochs]
        """
        rows      = self._rows
        if not rows:
            return

        epochs    = [r["global_epoch"]     for r in rows]
        eval_rows = [r for r in rows if r["test_accuracy"] is not None]

        def _savefig(fig, suffix):
            path = os.path.join(self.output_dir,
                                f"{self.experiment_name}_{suffix}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[AsyncLogger] Saved {path}")

        # ------------------------------------------------------------------
        # 1. Accuracy vs global epoch
        # ------------------------------------------------------------------
        if eval_rows:
            e_ep  = [r["global_epoch"]     for r in eval_rows]
            accs  = [r["test_accuracy"]    for r in eval_rows]
            times = [r["simulated_time_s"] for r in eval_rows]

            fig, ax = plt.subplots()
            ax.plot(e_ep, accs, marker="o", linewidth=1.5, markersize=4)
            ax.set_xlabel("Global epoch")
            ax.set_ylabel("Test accuracy")
            ax.set_title(f"{self.experiment_name} — accuracy vs epoch")
            ax.grid(True, alpha=0.3)
            _savefig(fig, "acc_vs_epoch")

            # ------------------------------------------------------------------
            # 2. Accuracy vs simulated time
            # ------------------------------------------------------------------
            fig, ax = plt.subplots()
            ax.plot(times, accs, marker="o", linewidth=1.5, markersize=4,
                    color="tab:orange")
            ax.set_xlabel("Simulated time (s)")
            ax.set_ylabel("Test accuracy")
            ax.set_title(f"{self.experiment_name} — accuracy vs simulated time")
            ax.grid(True, alpha=0.3)
            _savefig(fig, "acc_vs_time")

            # ------------------------------------------------------------------
            # 3. Test loss vs epoch
            # ------------------------------------------------------------------
            losses = [r["test_loss"] for r in eval_rows]
            fig, ax = plt.subplots()
            ax.plot(e_ep, losses, marker="o", linewidth=1.5, markersize=4,
                    color="tab:red")
            ax.set_xlabel("Global epoch")
            ax.set_ylabel("Test loss")
            ax.set_title(f"{self.experiment_name} — test loss vs epoch")
            ax.grid(True, alpha=0.3)
            _savefig(fig, "test_loss_vs_epoch")

        # ------------------------------------------------------------------
        # 4. Training loss
        # ------------------------------------------------------------------
        train_losses = [r["train_loss"] for r in rows]
        fig, ax = plt.subplots()
        ax.plot(epochs, train_losses, linewidth=1.0, color="tab:purple", alpha=0.8)
        ax.set_xlabel("Global epoch")
        ax.set_ylabel("Training loss (arriving update)")
        ax.set_title(f"{self.experiment_name} — training loss vs epoch")
        ax.grid(True, alpha=0.3)
        _savefig(fig, "train_loss_vs_epoch")

        # ------------------------------------------------------------------
        # 5. Staleness
        # ------------------------------------------------------------------
        staleness = [r["staleness"] for r in rows]
        fig, ax = plt.subplots()
        ax.plot(epochs, staleness, linewidth=1.0, color="tab:red", alpha=0.7)
        ax.set_xlabel("Global epoch")
        ax.set_ylabel("Staleness  t − τ")
        ax.set_title(f"{self.experiment_name} — update staleness vs epoch")
        ax.grid(True, alpha=0.3)
        _savefig(fig, "staleness_vs_epoch")

        # ------------------------------------------------------------------
        # 6. Effective alpha_t
        # ------------------------------------------------------------------
        alphas = [r["alpha_used"] for r in rows]
        fig, ax = plt.subplots()
        ax.plot(epochs, alphas, linewidth=1.0, color="tab:green", alpha=0.8)
        ax.set_xlabel("Global epoch")
        ax.set_ylabel("Effective α_t")
        ax.set_title(f"{self.experiment_name} — mixing weight vs epoch")
        ax.grid(True, alpha=0.3)
        _savefig(fig, "alpha_vs_epoch")

        # ------------------------------------------------------------------
        # 7. Energy per epoch
        # ------------------------------------------------------------------
        energies = [r["total_energy_j"] for r in rows]
        fig, ax = plt.subplots()
        ax.plot(epochs, energies, linewidth=1.0, color="tab:brown", alpha=0.8)
        ax.set_xlabel("Global epoch")
        ax.set_ylabel("Energy per epoch (J)")
        ax.set_title(f"{self.experiment_name} — energy vs epoch")
        ax.grid(True, alpha=0.3)
        _savefig(fig, "energy_vs_epoch")

        if not eval_rows:
            print("[AsyncLogger] No evaluation data — accuracy/loss plots skipped.")
