"""
examples/fedasync_experiment.py: Canonical FedAsync experiment.

Demonstrates:
  1. FedAsync+Const     — constant alpha, random client selection
  2. FedAsync+Poly      — polynomial staleness decay
  3. FedAsync+Hinge     — hinge staleness decay
  4. FedAsync+TopKFast  — always pick the fastest clients (compute + upload)
  5. FedAsync+FixedFast — random from a fixed fast pool
  6. FedAvg (sync)      — synchronous baseline for comparison

All runs share the same dataset, model, channel, and system config
(flsim/configs/mnist_fedavg.yaml).  Only the async algorithm changes.

Run from the repo root:
    python examples/fedasync_experiment.py

Writing a custom async algorithm
---------------------------------
Inherit from AsyncFederatedAlgorithm and override what you need:

    from flsim.interfaces.async_algorithm import AsyncFederatedAlgorithm
    from collections import OrderedDict

    class MyAsyncAlg(AsyncFederatedAlgorithm):

        def select_clients(self, all_clients, num_to_trigger, rng, **kwargs):
            # Example: round-robin selection
            if not hasattr(self, "_rr_idx"):
                self._rr_idx = 0
            start = self._rr_idx % len(all_clients)
            selected = []
            for i in range(num_to_trigger):
                selected.append(all_clients[(start + i) % len(all_clients)])
            self._rr_idx += num_to_trigger
            return selected

        def mixing_weight(self, base_alpha, staleness):
            # Exponential decay
            import math
            return base_alpha * math.exp(-0.1 * staleness)

        def aggregate_async(self, global_model, update, global_epoch, staleness, alpha_t):
            # Standard mixing rule with the custom alpha_t from mixing_weight
            current   = global_model.state_dict()
            new_state = {}
            for key in current:
                new_state[key] = (
                    (1 - alpha_t) * current[key].float()
                    + alpha_t     * update.state_dict[key].float()
                )
            return new_state

    # Then use it exactly like any other async algorithm:
    r = exp.run_single_async("my_alg", components={"algorithm": MyAsyncAlg()})
"""

import os
import sys

# Make sure the repo root is on the path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flsim.algorithms.fedavg import FedAvg
from flsim.algorithms.fedasync import (
    FedAsync, FedAsyncTopKFastTotal, FedAsyncFixedFast,
)
from flsim.experiments.async_base import AsyncExperiment


BASE_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "flsim", "configs", "mnist_fedavg.yaml"
)
OUTPUT_DIR = "outputs/fedasync_experiment/"


class FedAsyncExperiment(AsyncExperiment):
    """
    Compare FedAsync variants and the synchronous FedAvg baseline.

    Config overrides applied to all runs:
      - 100 global rounds (FedAsync epochs = 100 server updates)
      - 10 clients in flight (async window size)
      - Evaluate every 10 epochs to keep it fast
    """

    SHARED_OVERRIDES = {
        "learning.global_rounds": 500,
        "async_fl.window_size":   10,
        "evaluation.evaluate_every": 10,
    }

    def run(self):
        results = {}

        # ------------------------------------------------------------------
        # 1. FedAsync + Constant α  (baseline async, no staleness adaptation)
        # ------------------------------------------------------------------
        results["FedAsync+Const"] = self.run_single_async(
            run_name="fedasync_const",
            label="FedAsync+Const",
            config_overrides={
                **self.SHARED_OVERRIDES,
                "async_fl.alpha": 0.1,
            },
            components={
                "algorithm": FedAsync(alpha=0.1, staleness_func="constant"),
            },
        )

        # ------------------------------------------------------------------
        # 2. FedAsync + Polynomial decay  s(k) = (k+1)^{-0.5}
        # ------------------------------------------------------------------
        results["FedAsync+Poly"] = self.run_single_async(
            run_name="fedasync_poly",
            label="FedAsync+Poly (a=0.5)",
            config_overrides={
                **self.SHARED_OVERRIDES,
                "async_fl.alpha": 0.1,
            },
            components={
                "algorithm": FedAsync(
                    alpha=0.1,
                    staleness_func="polynomial",
                    a=0.5,
                ),
            },
        )

        # ------------------------------------------------------------------
        # 3. FedAsync + Hinge  s(k) = 1 if k <= 4, else 1/(10*(k-4)+1)
        # ------------------------------------------------------------------
        results["FedAsync+Hinge"] = self.run_single_async(
            run_name="fedasync_hinge",
            label="FedAsync+Hinge (a=10, b=4)",
            config_overrides={
                **self.SHARED_OVERRIDES,
                "async_fl.alpha": 0.1,
            },
            components={
                "algorithm": FedAsync(
                    alpha=0.1,
                    staleness_func="hinge",
                    a=10.0,
                    b=4.0,
                ),
            },
        )

        # ------------------------------------------------------------------
        # 4. FedAsync + Top-K fastest clients by TOTAL time (compute + upload)
        #    Always dispatches the fastest clients end-to-end. The simulator
        #    supplies channel + upload-size context automatically, so a client
        #    that computes fast but has a weak channel is correctly deprioritised.
        # ------------------------------------------------------------------
        results["FedAsync+TopKFast"] = self.run_single_async(
            run_name="fedasync_topkfast",
            label="FedAsync+TopKFast",
            config_overrides={
                **self.SHARED_OVERRIDES,
                "async_fl.alpha":       0.1,
                "async_fl.window_size": 10,
            },
            components={
                "algorithm": FedAsyncTopKFastTotal(alpha=0.1, staleness_func="constant"),
            },
        )

        # ------------------------------------------------------------------
        # 6. FedAsync + Fixed fast pool (random from top-20 fastest)
        # ------------------------------------------------------------------
        results["FedAsync+FixedPool"] = self.run_single_async(
            run_name="fedasync_fixedpool",
            label="FedAsync+FixedFast (pool=20)",
            config_overrides={
                **self.SHARED_OVERRIDES,
                "async_fl.alpha": 0.1,
            },
            components={
                "algorithm": FedAsyncFixedFast(
                    pool_size=20, alpha=0.1, staleness_func="constant"
                ),
            },
        )

        # ------------------------------------------------------------------
        # 7. Synchronous FedAvg baseline
        #    Same number of rounds, same clients_per_round as window_size
        # ------------------------------------------------------------------
        results["FedAvg (sync)"] = self.run_single(
            run_name="fedavg_sync_baseline",
            label="FedAvg (sync)",
            config_overrides={
                "learning.global_rounds": 100,
                "learning.clients_per_round": 10,
                "evaluation.evaluate_every": 10,
            },
            components={
                "algorithm": FedAvg(),
            },
        )

        # ------------------------------------------------------------------
        # Comparison plots
        # ------------------------------------------------------------------
        print("\n[FedAsyncExperiment] Generating comparison plots …")

        # Accuracy vs round / epoch
        self.plot_comparison(
            results,
            plot_configs=[
                {"metric": "test_accuracy",  "x": "round",
                 "ylabel": "Test Accuracy",
                 "title": "FedAsync variants vs FedAvg — Accuracy vs Epoch"},
                {"metric": "test_accuracy",  "x": "simulated_time_s",
                 "ylabel": "Test Accuracy",
                 "title": "FedAsync variants vs FedAvg — Accuracy vs Simulated Time"},
                {"metric": "total_energy_j", "x": "round",
                 "ylabel": "Energy per epoch (J)",
                 "title": "Energy comparison"},
            ],
            out_prefix="fedasync_comparison",
        )

        # Summary bar chart
        self.plot_bar(
            results,
            metric="best_accuracy",
            ylabel="Best test accuracy",
            out_name="fedasync_best_acc_bar",
            title="FedAsync variants — best accuracy",
        )

        # Print final summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for label, r in results.items():
            print(
                f"  {label:<30s}  "
                f"best={r.best_accuracy:.4f}  "
                f"final={r.final_accuracy:.4f}  "
                f"time={r.total_simulated_time_s:.0f}s"
            )
        print("=" * 60)


if __name__ == "__main__":
    FedAsyncExperiment(
        base_config=BASE_CONFIG,
        output_dir=OUTPUT_DIR,
    ).run()
