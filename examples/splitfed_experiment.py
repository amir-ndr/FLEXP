"""
examples/splitfed_experiment.py: A cross-paradigm distributed-learning
comparison, structured like Figure 2 in Thapa, Chamikara Mahawaga Arachchige,
Camtepe & Sun, "SplitFed: When Federated Learning Meets Split Learning"
(AAAI-22, arXiv:2004.12088), extended with an asynchronous baseline. Six
paradigms are compared on accuracy, energy, traffic, and latency:
  Normal    — centralized training (the benchmark ceiling)
  FL        — synchronous FedAvg (full model)
  FedAsync  — ASYNCHRONOUS FedAvg (full model); updates on each arrival
  SL        — split learning, sequential
  SFLV1     — splitfed, both sides parallel + FedAvg
  SFLV2     — splitfed, client parallel + FedAvg, server sequential

Fairness across paradigms
-------------------------
All six are computed on the SAME physical base (FDMA Shannon rate, DVFS
compute energy, uplink TX energy — flsim.system.split_cost / the sync/async
simulators), so latency/energy/traffic are directly comparable. The one care
point is FedAsync: it has no "rounds" — one async "epoch" = ONE client update,
whereas a sync/split "round" = NUM_CLIENTS client-trainings. So FedAsync is run
for NUM_CLIENTS × GLOBAL_ROUNDS epochs (same total client-training work) and is
compared on the genuinely apples-to-apples axes — SIMULATED TIME and CUMULATIVE
energy/traffic — where its per-update granularity lines up with the round-based
methods. The per-round plots additionally show it at its per-update grain.

Honest scope note — this is NOT a literal reproduction of Figure 2
------------------------------------------------------------------
Figure 2 specifically uses ResNet18 on the HAM10000 medical dataset with 5
clients. This framework has no HAM10000 loader (a Harvard Dataverse medical
imaging dataset with nontrivial preprocessing — out of scope to add here) and
no ResNet18 wiring. This script substitutes MNIST + the existing MnistCNN
model instead — note this substitution isn't arbitrary: the paper's own
Table 5 separately reports "MNIST + ResNet18" and "MNIST + AlexNet" results,
so MNIST is a combination the paper itself validates, just not with this
exact (smaller) CNN. What IS faithfully reproduced is the paradigm
comparison structure Figure 2 demonstrates — Normal/FL/SL/SFLV1/SFLV2 test
accuracy convergence over global epochs with 5 clients — and, more
importantly, the underlying SL/SFLV1/SFLV2 mechanics have been verified
against the paper's algorithm descriptions directly (see flsim/core/
split_simulator.py and flsim/core/split_client.py's module docstrings for
the correctness proofs — split-relay training is numerically IDENTICAL to
training the unsplit model directly, and each variant's client/server
sharing pattern was checked to match Table 1 and the "Variants of Splitfed
Learning" section exactly).

System-cost metrics (latency / energy / traffic) ARE modeled, on the same
physical base as the sync/async/OTA simulators (FDMA Shannon rate, DVFS
compute energy, uplink TX energy) — see flsim.system.split_cost.SplitCostModel.
This lets the script compare all five paradigms fairly on six axes: accuracy
vs global epoch, accuracy vs simulated time, cumulative energy, per-round
communication traffic, and per-round training latency (plus a best-accuracy
bar). Normal (centralized, compute-only) and FL (its own FDMA cost from the
sync Simulator, plus a computed traffic column) are made comparable to the
split runs.

Run from the repo root:
    python examples/splitfed_experiment.py
"""

import copy
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flsim.algorithms.fedavg import FedAvg
from flsim.algorithms.fedasync import FedAsync
from flsim.experiments.base import RunResult
from flsim.experiments.split_base import SplitExperiment
from flsim.experiments.async_base import AsyncExperiment
from flsim.experiments.wiring import load_config, set_seeds, _model_name_for_dataset
from flsim.models.factory import create_model
from flsim.core.evaluator import Evaluator
from flsim.system.split_cost import SplitCostModel
from flsim.data.loaders.mnist import load_mnist
from flsim.data.loaders.cifar10 import load_cifar10


BASE_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "flsim", "configs", "mnist_fedavg.yaml"
)
OUTPUT_DIR = "outputs/splitfed_experiment/"

# Paper's Figure 2 setting: 5 clients, C=1 (full participation every epoch).
NUM_CLIENTS    = 5
GLOBAL_ROUNDS  = 30
LOCAL_EPOCHS   = 1
BATCH_SIZE     = 32
LEARNING_RATE  = 0.01
CUT_LAYER      = 6     # mnist_cnn's features/classifier boundary (10 layers total)
EVALUATE_EVERY = 1

# FedAsync (full-model async baseline) settings. To do the SAME total work as
# the round-based methods (each of their rounds = NUM_CLIENTS client-trainings),
# async runs NUM_CLIENTS × GLOBAL_ROUNDS epochs (each epoch = ONE client update)
# and evaluates every NUM_CLIENTS epochs — so it hits the same work checkpoints.
ASYNC_ALPHA      = 0.5          # base mixing weight
ASYNC_STALENESS  = "constant"   # "constant" | "polynomial" | "hinge"
ASYNC_WINDOW     = NUM_CLIENTS  # concurrent in-flight clients (matches FL's cohort)


# Multiple inheritance: SplitExperiment and AsyncExperiment both extend
# Experiment and add non-overlapping methods (run_single_split / run_single_async),
# so this class can drive sync, split, AND async runs from one place.
class SplitFedFigure2Experiment(SplitExperiment, AsyncExperiment):
    """
    Compare Normal (centralized) / FL / SL / SFLV1 / SFLV2 test-accuracy
    convergence over global epochs, all under identical data partitioning,
    model, and hyperparameters — the same comparison structure as Figure 2.
    """

    SHARED_OVERRIDES = {
        "learning.global_rounds":     GLOBAL_ROUNDS,
        "learning.local_epochs":      LOCAL_EPOCHS,
        "learning.batch_size":        BATCH_SIZE,
        "learning.learning_rate":     LEARNING_RATE,
        "learning.cut_layer":         CUT_LAYER,
        "learning.clients_per_round": NUM_CLIENTS,
        "data.num_clients":           NUM_CLIENTS,
        "data.partition":             "iid",
        "evaluation.evaluate_every":  EVALUATE_EVERY,
        # Lock the model-size accounting to the REAL model (state_dict elements),
        # not the fixed 28,100-bit placeholder, so FL/FedAsync upload time & energy
        # use the true network size — matching how split and the traffic helpers
        # measure it. Robust even if the base config defaults to "fixed".
        "wireless.upload_size_mode":  "model",
    }

    def run(self):
        results = {}

        # ------------------------------------------------------------------
        # 1. Normal — centralized training on the full pooled dataset.
        #    No client partitioning, no split, no aggregation: the paper's
        #    own benchmark ceiling.
        # ------------------------------------------------------------------
        results["Normal"] = self._run_normal()

        # ------------------------------------------------------------------
        # 2. FL — standard synchronous FedAvg (reuses the existing sync
        #    Simulator/FedAvg — no split-learning code involved).
        # ------------------------------------------------------------------
        results["FL"] = self.run_single(
            run_name="splitfed_fl",
            label="FL",
            config_overrides=self.SHARED_OVERRIDES,
            components={"algorithm": FedAvg()},
        )
        self._add_fl_traffic(results["FL"])   # add a comparable traffic column

        # ------------------------------------------------------------------
        # 2b. FedAsync — full-model ASYNCHRONOUS FL baseline. Same physical
        #    base (compute/energy/channel) as FL and split, but updates the
        #    global model on each single arrival instead of per round. Runs
        #    NUM_CLIENTS × GLOBAL_ROUNDS epochs so its total client-training
        #    work matches the round-based methods (see the module docstring's
        #    fairness note). Its "round" column is an epoch index (1 update),
        #    so compare it on SIMULATED TIME / CUMULATIVE axes, not per-round.
        # ------------------------------------------------------------------
        async_overrides = {k: v for k, v in self.SHARED_OVERRIDES.items()
                           if k not in ("learning.global_rounds", "evaluation.evaluate_every")}
        results["FedAsync"] = self.run_single_async(
            run_name="splitfed_fedasync",
            label="FedAsync",
            config_overrides={
                **async_overrides,
                "learning.global_rounds":    GLOBAL_ROUNDS * NUM_CLIENTS,
                "async_fl.window_size":      ASYNC_WINDOW,
                "async_fl.alpha":            ASYNC_ALPHA,
                "evaluation.evaluate_every": EVALUATE_EVERY * NUM_CLIENTS,
            },
            components={"algorithm": FedAsync(alpha=ASYNC_ALPHA, staleness_func=ASYNC_STALENESS)},
        )
        self._add_async_traffic(results["FedAsync"])

        # ------------------------------------------------------------------
        # 3. SL — sequential relay, both sides shared/unaggregated.
        # ------------------------------------------------------------------
        results["SL"] = self.run_single_split(
            run_name="splitfed_sl",
            label="SL",
            config_overrides=self.SHARED_OVERRIDES,
            client_mode="sequential",
            server_mode="sequential",
        )

        # ------------------------------------------------------------------
        # 4. SFLV1 — both sides parallel + FedAvg aggregated.
        # ------------------------------------------------------------------
        results["SFLV1"] = self.run_single_split(
            run_name="splitfed_sflv1",
            label="SFLV1",
            config_overrides=self.SHARED_OVERRIDES,
            client_mode="parallel_fedavg",
            server_mode="parallel_fedavg",
        )

        # ------------------------------------------------------------------
        # 5. SFLV2 — client-side parallel + FedAvg (same as SFLV1), server
        #    side sequential (same as SL) with per-client random order.
        # ------------------------------------------------------------------
        results["SFLV2"] = self.run_single_split(
            run_name="splitfed_sflv2",
            label="SFLV2",
            config_overrides=self.SHARED_OVERRIDES,
            client_mode="parallel_fedavg",
            server_mode="sequential",
        )

        # ------------------------------------------------------------------
        # Comparison plots (Figure 2's shape: accuracy vs global epoch)
        # ------------------------------------------------------------------
        print("\n[SplitFedFigure2Experiment] Generating comparison plots …")
        # All runs share these column names on the SAME physical base
        # (simulated_time_s, cumulative_energy_j, traffic_bytes, round_latency_s),
        # so the comparison is fair. plot_comparison silently skips any run
        # missing a column, so this is robust even if a baseline lacks one.
        #
        # FAIRNESS NOTE — sync/split methods do NUM_CLIENTS client-trainings per
        # "round"; FedAsync does ONE per "epoch" (its round column is an epoch
        # index). So the genuinely apples-to-apples axes across ALL paradigms
        # are SIMULATED TIME and CUMULATIVE quantities — the "*_vs_time" plots
        # below and cumulative energy/traffic. The per-round plots (per-round
        # traffic, per-round latency, and vs-round curves) are natural for the
        # round-structured methods; FedAsync appears there at its per-UPDATE
        # granularity (1/NUM_CLIENTS the per-step work, NUM_CLIENTS× more steps),
        # which only lines up on the cumulative/time axes.
        self.plot_comparison(
            results,
            plot_configs=[
                # ---- fairest axes: everything vs SIMULATED TIME ----
                # (1) accuracy vs simulated time
                {"metric": "test_accuracy", "x": "simulated_time_s",
                 "ylabel": "Test Accuracy",
                 "title": "Accuracy vs Simulated Time"},
                # (2) cumulative energy vs simulated time
                {"metric": "cumulative_energy_j", "x": "simulated_time_s",
                 "ylabel": "Cumulative energy (J)",
                 "title": "Cumulative Energy vs Simulated Time"},
                # (3) cumulative traffic vs simulated time
                {"metric": "cumulative_traffic_bytes", "x": "simulated_time_s",
                 "ylabel": "Cumulative traffic (bytes)",
                 "title": "Cumulative Communication Traffic vs Simulated Time"},

                # ---- convergence view (round/epoch index) ----
                {"metric": "test_accuracy", "x": "round",
                 "ylabel": "Test Accuracy",
                 "title": "Accuracy vs Round / Epoch"},
                {"metric": "test_loss", "x": "round",
                 "ylabel": "Test Loss",
                 "title": "Loss vs Round / Epoch"},
                {"metric": "cumulative_energy_j", "x": "round",
                 "ylabel": "Cumulative energy (J)",
                 "title": "Cumulative Energy vs Round / Epoch"},

                # ---- per-step views (natural for round-structured methods) ----
                # per-round/epoch communication traffic
                {"metric": "traffic_bytes", "x": "round",
                 "ylabel": "Traffic per step (bytes)",
                 "title": "Communication Traffic per Round / Epoch"},
                # per-round/epoch training latency
                {"metric": "round_latency_s", "x": "round",
                 "ylabel": "Latency per step (s)",
                 "title": "Per-Round / Per-Epoch Training Latency"},
            ],
            out_prefix="splitfed_comparison",
        )
        self.plot_bar(
            results,
            metric="best_accuracy",
            ylabel="Best test accuracy",
            out_name="splitfed_best_acc_bar",
            title="Normal / FL / FedAsync / SL / SFLV1 / SFLV2 — best accuracy",
        )

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print("\n" + "=" * 78)
        print(f"  {'':<8s}{'best_acc':>10s}{'sim_time(s)':>14s}{'energy(J)':>13s}{'traffic(MB)':>14s}")
        print("=" * 78)
        for label, r in results.items():
            df = r.df
            t = df["simulated_time_s"].iloc[-1] if "simulated_time_s" in df.columns else float("nan")
            e = df["cumulative_energy_j"].iloc[-1] if "cumulative_energy_j" in df.columns else float("nan")
            traffic = (df["cumulative_traffic_bytes"].iloc[-1] / 1e6
                       if "cumulative_traffic_bytes" in df.columns else float("nan"))
            print(f"  {label:<8s}{r.best_accuracy:>10.4f}{t:>14.0f}{e:>13.1f}{traffic:>14.1f}")
        print("=" * 78)

    # ------------------------------------------------------------------
    # Normal (centralized) baseline — no FL/split machinery at all.
    # ------------------------------------------------------------------

    def _run_normal(self) -> RunResult:
        """
        Train the full (unsplit) model directly on the ENTIRE pooled training
        set for GLOBAL_ROUNDS epochs — the "Normal" row in Table 5 / Figure 2,
        i.e. the centralized-training ceiling every distributed method is
        compared against.
        """
        run_name = "splitfed_normal"
        print(f"\n{'='*60}\n[Experiment] Run: Normal (centralized)\n{'='*60}")

        config = load_config(BASE_CONFIG)
        from flsim.experiments.base import _apply_config_overrides
        config = _apply_config_overrides(config, self.SHARED_OVERRIDES)
        set_seeds(config.experiment.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.data.dataset == "mnist":
            train_ds, test_ds = load_mnist()
        elif config.data.dataset == "cifar10":
            train_ds, test_ds = load_cifar10()
        else:
            raise ValueError(f"Unknown dataset: {config.data.dataset}")

        model = create_model(_model_name_for_dataset(config.data.dataset)).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning.learning_rate)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(train_ds, batch_size=config.learning.batch_size, shuffle=True)
        evaluator = Evaluator(test_dataset=test_ds)

        # ---- cost model: centralized training on the edge server (same physical
        #      base as split/FL — compute only, no communication) ----
        cost_model = SplitCostModel(
            channel_model=None, noise_psd_w_per_hz=0.0,
            kappa=config.system.switched_capacitance,
            server_cpu_frequency_hz=float(getattr(getattr(config, "split", None),
                                                  "server_cpu_frequency_hz", 3.0e9)),
        )
        cycles = (config.system.cycles_per_sample_min + config.system.cycles_per_sample_max) / 2.0
        total_samples = len(train_ds)

        run_dir = os.path.join(self.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        csv_path = os.path.join(run_dir, f"{run_name}.csv")
        rows = []
        cum_time = cum_energy = 0.0

        for epoch in range(config.learning.global_rounds):
            model.train()
            total_loss, total_batches = 0.0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_batches += 1
            train_loss = total_loss / max(total_batches, 1)

            rc = cost_model.centralized_cost(total_samples, config.learning.local_epochs, cycles)
            cum_time   += rc.latency_s
            cum_energy += rc.total_energy_j

            eval_result = None
            if epoch % config.evaluation.evaluate_every == 0:
                eval_result = evaluator.evaluate(model, device=device)
                print(f"  Epoch {epoch:4d} | train_loss={train_loss:.4f} | "
                      f"acc={eval_result.test_accuracy:.4f} | loss={eval_result.test_loss:.4f}")

            rows.append({
                "round": epoch,
                "train_loss": f"{train_loss:.6f}",
                "test_loss": f"{eval_result.test_loss:.6f}" if eval_result else "",
                "test_accuracy": f"{eval_result.test_accuracy:.6f}" if eval_result else "",
                "round_latency_s": f"{rc.latency_s:.6f}",
                "simulated_time_s": f"{cum_time:.6f}",
                "traffic_bytes": "0.0",
                "cumulative_traffic_bytes": "0.0",
                "total_energy_j": f"{rc.total_energy_j:.6e}",
                "cumulative_energy_j": f"{cum_energy:.6e}",
            })

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "round", "train_loss", "test_loss", "test_accuracy",
                "round_latency_s", "simulated_time_s", "traffic_bytes",
                "cumulative_traffic_bytes", "total_energy_j", "cumulative_energy_j",
            ])
            writer.writeheader()
            writer.writerows(rows)

        df = pd.read_csv(csv_path)
        result = RunResult(name=run_name, label="Normal", config=config, csv_path=csv_path, df=df)
        print(f"[Experiment] Done — final_acc={result.final_accuracy:.4f}  best_acc={result.best_accuracy:.4f}\n")
        return result

    # ------------------------------------------------------------------
    # FL uses the sync Simulator, whose CSV already has simulated_time_s,
    # round_duration_s, cumulative_energy_j on the SAME physical base — it
    # only lacks a traffic column. Add it so FL appears on the traffic plot.
    # ------------------------------------------------------------------

    @staticmethod
    def _model_size_elements(dataset: str) -> int:
        """
        Full-model size in ELEMENTS, counted from the state_dict (parameters +
        buffers, e.g. BatchNorm running stats) — the EXACT quantity the sync/
        async simulators use for upload time/energy (_resolve_upload_bits with
        mode="model"). Using state_dict (not just .parameters()) keeps traffic
        perfectly consistent with time/energy for models that carry buffers
        (e.g. CifarCNN's BatchNorm); for MnistCNN there are no buffers, so it is
        identical to the parameter count.
        """
        model = create_model(_model_name_for_dataset(dataset))
        return sum(t.numel() for t in model.state_dict().values())

    def _add_fl_traffic(self, result: RunResult) -> None:
        """FedAvg traffic per round = 2·K·|W| (full model down + up), constant."""
        elems = self._model_size_elements(result.config.data.dataset)
        K = int(result.config.learning.clients_per_round)
        per_round = 2 * K * elems * 4   # bytes (float32)
        df = result.df
        df["traffic_bytes"] = float(per_round)
        df["cumulative_traffic_bytes"] = per_round * (df["round"] + 1)
        # alias the sync per-round-latency column to the shared name
        if "round_duration_s" in df.columns:
            df["round_latency_s"] = df["round_duration_s"]

    def _add_async_traffic(self, result: RunResult) -> None:
        """
        FedAsync traffic per EPOCH = 2·|W| (one client downloads + uploads the
        full model). Note this is 1/K of FL's per-round traffic, but async runs
        K× more epochs, so the CUMULATIVE traffic matches FL — which is why the
        fair comparison for async is on cumulative / time axes, not per-round.
        """
        elems = self._model_size_elements(result.config.data.dataset)
        per_epoch = 2 * elems * 4   # bytes: one client, down + up
        df = result.df
        df["traffic_bytes"] = float(per_epoch)
        df["cumulative_traffic_bytes"] = per_epoch * (df["round"] + 1)
        # async's per-update total time is the natural per-step latency analog
        if "total_time_s" in df.columns:
            df["round_latency_s"] = df["total_time_s"]


if __name__ == "__main__":
    SplitFedFigure2Experiment(
        base_config=BASE_CONFIG,
        output_dir=OUTPUT_DIR,
    ).run()
