"""
examples/splitfed_experiment.py: Replicates the shape of Figure 2 in Thapa,
Chamikara Mahawaga Arachchige, Camtepe & Sun, "SplitFed: When Federated
Learning Meets Split Learning" (AAAI-22, arXiv:2004.12088) — testing
convergence under 5 learning paradigms: Normal (centralized), FL, SL,
SFLV1, SFLV2.

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

Communication time/energy are NOT modeled (see split_simulator.py's scope
note) — this script only reproduces the ACCURACY convergence comparison,
not Table 2's communication-cost analysis.

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
from flsim.experiments.base import RunResult
from flsim.experiments.split_base import SplitExperiment
from flsim.experiments.wiring import load_config, set_seeds, _model_name_for_dataset
from flsim.models.factory import create_model
from flsim.core.evaluator import Evaluator
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


class SplitFedFigure2Experiment(SplitExperiment):
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
        self.plot_comparison(
            results,
            plot_configs=[
                {"metric": "test_accuracy", "x": "round",
                 "ylabel": "Test Accuracy",
                 "title": "Normal vs FL vs SL vs SFLV1 vs SFLV2 — Accuracy vs Global Epoch"},
                {"metric": "test_loss", "x": "round",
                 "ylabel": "Test Loss",
                 "title": "Normal vs FL vs SL vs SFLV1 vs SFLV2 — Loss vs Global Epoch"},
            ],
            out_prefix="splitfed_comparison",
        )
        self.plot_bar(
            results,
            metric="best_accuracy",
            ylabel="Best test accuracy",
            out_name="splitfed_best_acc_bar",
            title="Normal / FL / SL / SFLV1 / SFLV2 — best accuracy",
        )

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for label, r in results.items():
            print(f"  {label:<10s}  best={r.best_accuracy:.4f}  final={r.final_accuracy:.4f}")
        print("=" * 60)

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

        run_dir = os.path.join(self.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        csv_path = os.path.join(run_dir, f"{run_name}.csv")
        rows = []

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
            })

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "train_loss", "test_loss", "test_accuracy"])
            writer.writeheader()
            writer.writerows(rows)

        df = pd.read_csv(csv_path)
        result = RunResult(name=run_name, label="Normal", config=config, csv_path=csv_path, df=df)
        print(f"[Experiment] Done — final_acc={result.final_accuracy:.4f}  best_acc={result.best_accuracy:.4f}\n")
        return result


if __name__ == "__main__":
    SplitFedFigure2Experiment(
        base_config=BASE_CONFIG,
        output_dir=OUTPUT_DIR,
    ).run()
