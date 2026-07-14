"""
examples/fedasync_experiment.py: Canonical FedAsync experiment.

30 clients total (data.num_clients in mnist_fedavg.yaml), window_size=30 for
all async variants (so every client can be in flight at once).

Demonstrates:
  1. FedAsync+Const  — constant alpha, random client selection.       1000 epochs.
  2. FedAsync+Poly   — polynomial staleness decay.                    1000 epochs.
  3. FedAsync+Hinge  — hinge staleness decay (b=25, a=0.03 — retuned  1000 epochs.
                       for this config's actual staleness distribution,
                       see note below).
  4. FedAsync+TopK   — semi-async: buffers the k=5 fastest-to-arrive   500 epochs.
                       clients per global update (FedAsyncTopKFastTotal).
  5. FedAvg (sync)   — full participation (clients_per_round=30 = all  100 rounds.
                       clients every round).

All runs share the same dataset, model, channel, and system config
(flsim/configs/mnist_fedavg.yaml).  Only the async algorithm changes.

Hinge retuning note
--------------------
Hinge's s(k) = 1 if k<=b else 1/(a*(k-b)+1) only trusts an update fully up to
staleness b, then decays. The old defaults (b=4, a=10) assumed staleness stays
in the single digits — but empirically, with window_size=30 (all 30 clients
racing at once) and this config's system heterogeneity (cpu_freq 0.1-0.8 GHz,
exp_fading channel), a 250-epoch pilot measured: mean=26.4, median=21, p75=29,
p90=46, p99=92.5, max=101. With b=4, essentially every update fell into the
decay branch, and a=10 crushed alpha_t to ~0.001x base alpha by staleness~90
— the model was frozen almost immediately (see the earlier run's flat
accuracy curve). b=25 (~median) and a=0.03 keeps alpha_t at ~0.33x base alpha
even at staleness=92, and ~0.11x even at staleness=300 (the 1000-epoch full
run will see somewhat worse tail events than this 250-epoch pilot) — gentle
enough that Hinge can still learn instead of stalling. Re-run the pilot and
recheck these percentiles if you change window_size, num_clients, or the
system/wireless heterogeneity settings — the right b/a depend on them.

Alternate stopping condition (stop_by_time_s)
-----------------------------------------------
Every run below stops after a fixed number of rounds/epochs. To instead stop
by elapsed simulated time (e.g. after 5000 simulated seconds, regardless of
how many rounds that took), add to any run's config_overrides:
    "learning.stop_by_time_s": 5000
This applies to both the sync Simulator and AsyncSimulator; global_rounds is
ignored for that run while it's set. See flsim/configs/base.yaml.

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
from flsim.algorithms.fedasync import FedAsync, FedAsyncTopKFastTotal
from flsim.experiments.async_base import AsyncExperiment


BASE_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "flsim", "configs", "mnist_fedavg.yaml"
)
OUTPUT_DIR = "outputs/fedasync_experiment/"


class FedAsyncExperiment(AsyncExperiment):
    """
    Compare FedAsync variants and the synchronous FedAvg baseline.

    30 clients total. window_size=30 for every async variant (all clients can
    be in flight simultaneously). Round budgets differ by variant (see
    ROUNDS_* below) since semi-async and sync need fewer server updates to
    cover a comparable amount of client-training work.
    """

    # Shared by every async variant (Const/Poly/Hinge/TopK).
    ASYNC_OVERRIDES = {
        "async_fl.window_size":      30,
        "evaluation.evaluate_every": 10,
    }

    ROUNDS_FULLY_ASYNC = 1000   # Const / Poly / Hinge (buffer_size=1)
    ROUNDS_SEMI_ASYNC  = 500    # TopK (buffer_size=k=5)
    ROUNDS_SYNC        = 100    # FedAvg

    def run(self):
        results = {}

        # ------------------------------------------------------------------
        # 1. FedAsync + Constant α  (baseline async, no staleness adaptation)
        # ------------------------------------------------------------------
        results["FedAsync+Const"] = self.run_single_async(
            run_name="fedasync_const",
            label="FedAsync+Const",
            config_overrides={
                **self.ASYNC_OVERRIDES,
                "learning.global_rounds": self.ROUNDS_FULLY_ASYNC,
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
                **self.ASYNC_OVERRIDES,
                "learning.global_rounds": self.ROUNDS_FULLY_ASYNC,
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
        # 3. FedAsync + Hinge  s(k) = 1 if k <= 25, else 1/(0.03*(k-25)+1)
        #    b/a retuned for this config's actual staleness scale — see the
        #    module docstring for the calibration data behind these numbers.
        # ------------------------------------------------------------------
        results["FedAsync+Hinge"] = self.run_single_async(
            run_name="fedasync_hinge",
            label="FedAsync+Hinge (a=0.03, b=25)",
            config_overrides={
                **self.ASYNC_OVERRIDES,
                "learning.global_rounds": self.ROUNDS_FULLY_ASYNC,
                "async_fl.alpha": 0.1,
            },
            components={
                "algorithm": FedAsync(
                    alpha=0.1,
                    staleness_func="hinge",
                    a=0.03,
                    b=25.0,
                ),
            },
        )

        # ------------------------------------------------------------------
        # 4. FedAsync + Semi-async top-K buffering
        #    Buffers the k=5 fastest-to-arrive clients (out of window_size=30
        #    in flight) per global update, aggregates them together with one
        #    mixing step, and immediately re-dispatches 5 replacements. The
        #    other 25 clients keep training uninterrupted.
        # ------------------------------------------------------------------
        results["FedAsync+TopK"] = self.run_single_async(
            run_name="fedasync_topk",
            label="FedAsync+TopK (k=5)",
            config_overrides={
                **self.ASYNC_OVERRIDES,
                "learning.global_rounds": self.ROUNDS_SEMI_ASYNC,
                "async_fl.alpha": 0.1,
            },
            components={
                "algorithm": FedAsyncTopKFastTotal(alpha=0.1, k=5, staleness_func="constant"),
            },
        )

        # ------------------------------------------------------------------
        # 5. Synchronous FedAvg baseline — full participation (all 30 clients
        #    selected every round, matching data.num_clients=30).
        # ------------------------------------------------------------------
        results["FedAvg (sync)"] = self.run_single(
            run_name="fedavg_sync_baseline",
            label="FedAvg (sync)",
            config_overrides={
                "learning.global_rounds": self.ROUNDS_SYNC,
                "learning.clients_per_round": 30,
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
                # Staleness has no meaning for the sync FedAvg baseline (no
                # `staleness` column in its CSV) — plot_comparison silently
                # skips any run missing the requested column, so only the
                # async variants show up here.
                {"metric": "staleness",      "x": "round",
                 "ylabel": "Staleness (t − τ)",
                 "title": "FedAsync variants — staleness vs epoch"},
            ],
            out_prefix="fedasync_comparison",
        )

        # Summary bar charts
        self.plot_bar(
            results,
            metric="best_accuracy",
            ylabel="Best test accuracy",
            out_name="fedasync_best_acc_bar",
            title="FedAsync variants — best accuracy",
        )
        self.plot_bar(
            results,
            metric="avg_staleness",
            ylabel="Average staleness (t − τ)",
            out_name="fedasync_avg_staleness_bar",
            title="FedAsync variants — average staleness (0 for sync FedAvg)",
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
                f"avg_staleness={r.avg_staleness:.2f}  "
                f"time={r.total_simulated_time_s:.0f}s"
            )
        print("=" * 60)


if __name__ == "__main__":
    FedAsyncExperiment(
        base_config=BASE_CONFIG,
        output_dir=OUTPUT_DIR,
    ).run()
