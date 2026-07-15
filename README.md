# flsim — Federated Learning Simulator

A research-grade FL simulator with a clean, layered architecture designed for easy extension.
It supports four training paradigms, each with its own orchestrator but a shared
component/experiment/plotting stack:

| Paradigm | Orchestrator | Built-in algorithms |
|---|---|---|
| **Synchronous** FL | `Simulator` | `FedAvg`, `FedProx` |
| **Asynchronous** FL | `AsyncSimulator` | `FedAsync` (+ Const/Poly/Hinge staleness), `FedAsyncTopKFastTotal` (semi-async buffering), `FedAsyncSimulatedStaleness` |
| **Over-the-air** aggregation (AirComp) | `Simulator` + uplink-physics hook | `FedOTA` |
| **Split learning** (SL / SplitFed) | `SplitSimulator` | SL, SFLV1, SFLV2 (via `client_mode` × `server_mode`) |

The recurring design principle across all four: **write a new algorithm or
experiment by overriding only the methods that change.** Each section below ends
with copy-pasteable patterns for doing exactly that.

---

## Quick start

```bash
# Install
pip install -e flsim/

# Synchronous FedAvg
python examples/fedavg_experiment.py

# Asynchronous FedAsync — Const/Poly/Hinge + semi-async TopK + FedAvg baseline
python examples/fedasync_experiment.py

# Over-the-air aggregation — FedOTA (several MSE budgets) vs digital FedAvg
python examples/ota_experiment.py

# Split learning — Normal / FL / SL / SFLV1 / SFLV2 (Figure-2-style comparison)
python examples/splitfed_experiment.py

# With CLI overrides
python examples/fedavg_experiment.py --rounds 50 --clients_per_round 5 --lr 0.005

# Re-plot any saved CSV
python plot_results.py outputs/fedAVG/fedavg/fedavg.csv
```

For long runs on a GPU cluster, ready-made SLURM scripts live in `slurm/`
(`run_fedasync.slurm`, `run_ota.slurm`, `run_splitfed.slurm`, …) — each
auto-detects CUDA and logs the GPU it landed on.

> **Note:** There is no `run.py`. The entry point is always an experiment file
> in `examples/` (or your own script). All wiring utilities live in `flsim/experiments/wiring.py`.

---

## Project layout

```
FLEXP/
├── flsim/
│   ├── interfaces/              # ABCs — the contracts everything must satisfy
│   │   ├── algorithm.py           FederatedAlgorithm    (sync: select_clients, configure_client, aggregate)
│   │   ├── async_algorithm.py     AsyncFederatedAlgorithm (async: select_clients, mixing_weight, aggregate_async)
│   │   ├── allocator.py           ResourceAllocator     (allocate_bandwidth, allocate_power, allocate_cpu_freq)
│   │   ├── channel_model.py       ChannelModel          (channel_gain, achievable_rate_bps)
│   │   ├── time_model.py          TimeModel             (compute_training_time, compute_upload_time, …)
│   │   └── partitioner.py         DataPartitioner       (partition, describe)
│   │
│   ├── algorithms/              # FL algorithms
│   │   ├── fedavg.py              FedAvg   (sync, sample-weighted aggregation)
│   │   ├── fedprox.py             FedProx  (sync, proximal regularisation)
│   │   └── fedasync.py            FedAsync (async) + FedAsyncTopKFastTotal + FedAsyncFixedFast
│   │
│   ├── allocators/              # Resource allocation policies
│   │   └── equal_split.py         EqualSplitAllocator (FDMA equal split, max power, profile freq)
│   │
│   ├── channel/                 # Channel physics
│   │   ├── path_loss.py           3GPP UMa + frozen log-normal shadowing
│   │   ├── fdma.py                Alias for PathLossChannelModel
│   │   └── exp_fading.py          h = h0·ρ·d⁻², ρ~Exp(1) per round
│   │
│   ├── system/                  # Computation + energy formulas
│   │   ├── cellular_time.py       τ = (I·C·D)/f
│   │   └── energy.py              E_comp = κ·I·C·D·f²,  E_tx = p·t_up
│   │
│   ├── data/                    # Dataset loaders + partitioners
│   ├── models/                  # PyTorch model definitions
│   ├── profiles/                # Client system profiles (distance, freq, power)
│   ├── core/
│   │   ├── simulator.py           Synchronous FL simulator
│   │   ├── async_simulator.py     Asynchronous FL simulator (discrete-event)
│   │   ├── server.py              Server (holds global model + algorithm)
│   │   ├── client.py              Client (local PyTorch training)
│   │   ├── evaluator.py           Test-set evaluation
│   │   ├── logger.py              Sync CSV logger + plots
│   │   └── async_logger.py        Async CSV logger + plots (staleness, alpha_t columns)
│   ├── configs/                 # YAML experiment configs
│   ├── experiments/
│   │   ├── base.py                Experiment + RunResult  (sync)
│   │   ├── async_base.py          AsyncExperiment         (async, extends Experiment)
│   │   ├── compare_algorithms.py  AlgorithmComparison
│   │   ├── parameter_sweep.py     ParameterSweep
│   │   └── wiring.py              Config loading + component factories
│   └── tests/                   # Unit tests
│
├── examples/
│   ├── fedavg_experiment.py       Canonical FedAvg run (sync)
│   └── fedasync_experiment.py     FedAsync variants + FedAvg baseline (async)
│
├── plot_results.py                Standalone CSV → plots tool
└── README.md
```

---

## Writing a custom algorithm

Every algorithm subclasses `FederatedAlgorithm` from `flsim/interfaces/algorithm.py`.  
**Only `aggregate()` is abstract** — `select_clients()` and `configure_client()` have
working defaults (uniform random selection, no-op configuration).  
Override only the methods that differ from the baseline.

### Pattern A — custom aggregation only

```python
# flsim/algorithms/fed_median.py
from collections import OrderedDict
import torch
from flsim.interfaces.algorithm import FederatedAlgorithm

class FedMedian(FederatedAlgorithm):
    """Replace mean aggregation with coordinate-wise median (Byzantine-robust)."""

    def aggregate(self, global_model, client_updates) -> OrderedDict:
        agg = OrderedDict()
        for key in client_updates[0].state_dict:
            stack = torch.stack([u.state_dict[key].float() for u in client_updates])
            agg[key] = stack.median(dim=0).values
        return agg

    # select_clients()    → inherited: uniform random
    # configure_client()  → inherited: no-op
```

### Pattern B — custom selection only (inherit FedAvg aggregation)

```python
# flsim/algorithms/channel_aware.py
from flsim.algorithms.fedavg import FedAvg

class ChannelAwareSelection(FedAvg):
    """Pick the K clients with the shortest distance (best average channel)."""

    def select_clients(self, all_clients, num_to_select, rng, **kwargs):
        sorted_clients = sorted(all_clients, key=lambda c: c.profile.distance_m)
        return sorted_clients[:num_to_select]

    # aggregate()        → inherited from FedAvg (sample-weighted average)
    # configure_client() → inherited: no-op
```

`select_clients` receives system context via `**kwargs` (identical keys in sync
and async, so the same selector class works in both):

| kwarg | Meaning |
|-------|---------|
| `channel_model` | call `.channel_gain(profile, rng)` / `.achievable_rate_bps(...)` |
| `noise_psd_w_per_hz` | noise PSD in W/Hz |
| `bw_per_client_hz` | bandwidth each client would get (`B / clients_per_round`) |
| `round_idx` | current round (sync only) |

Example ranking by estimated **uplink rate** instead of raw distance:

```python
class RateAwareSelection(FedAvg):
    def select_clients(self, all_clients, num_to_select, rng, **kwargs):
        cm  = kwargs["channel_model"]
        n0  = kwargs["noise_psd_w_per_hz"]
        bw  = kwargs["bw_per_client_hz"]
        def rate(c):
            g = cm.channel_gain(c.profile, rng)
            return cm.achievable_rate_bps(bw, c.profile.tx_power_w, g, n0)
        return sorted(all_clients, key=rate, reverse=True)[:num_to_select]
```

### Pattern C — FedProx (proximal term, same selection + aggregation)

```python
# flsim/algorithms/fedprox.py
from flsim.algorithms.fedavg import FedAvg

class FedProx(FedAvg):
    def __init__(self, mu: float):
        self.mu = mu

    def configure_client(self, client, global_model, round_idx):
        # Client.train() reads client.proximal_mu if it exists
        client.proximal_mu = self.mu

    # select_clients() → inherited: uniform random
    # aggregate()      → inherited: FedAvg sample-weighted average
```

### Wiring a new algorithm

Add one line to `_make_algorithm()` in `flsim/experiments/wiring.py`:

```python
def _make_algorithm(name: str):
    if name == "fedavg":    return FedAvg()
    if name == "fedmedian": return FedMedian()
    if name == "fedprox":   return FedProx(mu=0.01)
    raise ValueError(f"Unknown algorithm: {name}")
```

Then set `learning.algorithm: "fedmedian"` in your YAML, or pass it directly via
`components={"algorithm": FedMedian()}` in an experiment.

---

## Asynchronous federated learning (FedAsync)

### How async FL differs from sync FL

| Property | Sync (FedAvg) | Async (FedAsync) |
|---|---|---|
| **Server waits for** | All K selected clients | No one — first arrival triggers update |
| **Update rule** | Weighted average of K updates | `x_t = (1−α_t)·x_{t-1} + α_t·x_new` |
| **Round duration** | `max(K clients' times)` — straggler-bounded | Individual arrival time — no straggler penalty |
| **Simulated time** | Accumulates `max(K times)` per round | Advances to each individual arrival timestamp |
| **Staleness** | Zero (all train on the same model) | `t − τ ≥ 0` where `τ` = model version client received |

> **Simulated time semantics in async FL:** `simulated_time_s` at global epoch `t` is the
> virtual wall-clock time when the `t`-th update arrives at the server.  
> It is **not** the max time of any group of clients — the server never waits for a group.
>
> Each client's arrival time = `dispatch_time + t_download + t_compute + t_upload`
> (this matches `total_time_s` in the ClientUpdate, consistent with the sync simulator).  
>
> Example with window=3: clients A (10s total), B (25s), C (40s), all dispatched at t=0 →
> epoch 0 fires at t=10s, epoch 1 at t=25s, epoch 2 at t=40s.
> A replacement is dispatched immediately after each arrival, training on the updated model.

### Simulation model (sliding-window discrete-event)

```
Initial dispatch (time=0):
  Select window_size clients, all train on model@epoch0 (τ=0)
  Arrival time per client = 0 + t_comp_k + t_upload_k

Updater loop (runs T times):
  Pop earliest arrival from priority queue
    staleness k = current_epoch − τ
    alpha_t    = algorithm.mixing_weight(base_alpha, k)
    x_t        = algorithm.aggregate_async(global_model, update, epoch, k, alpha_t)
    simulated_time = arrival_time_of_this_update

  Scheduler: dispatch 1 replacement client
    Replacement trains on the UPDATED model (current_epoch+1)
    arrival_time = simulated_time + t_comp_new + t_upload_new
```

Bandwidth is divided equally among `window_size` concurrent clients to model uplink contention:
`effective_bw_per_client = total_bandwidth / window_size`.

### Built-in algorithms and selection modes

```python
from flsim.algorithms.fedasync import (
    FedAsync, FedAsyncTopKFastTotal, FedAsyncFixedFast,
)

# Constant alpha (FedAsync+Const from the paper)
FedAsync(alpha=0.1)

# Polynomial staleness decay: alpha_t = alpha * (staleness + 1)^{-a}
FedAsync(alpha=0.1, staleness_func="polynomial", a=0.5)

# Hinge staleness decay: alpha_t = alpha if staleness <= b, else alpha / (a*(k-b)+1)
FedAsync(alpha=0.1, staleness_func="hinge", a=10.0, b=4.0)

# Always dispatch the K fastest clients by TOTAL time (compute + upload).
# The simulator passes channel_model / bw_per_client / upload_size automatically
# via kwargs, so a fast-compute but weak-channel client is correctly deprioritised.
# Falls back to compute-only ranking if no channel context is available.
FedAsyncTopKFastTotal(alpha=0.1)

# Sample randomly from a fixed pool of the M fastest-compute clients
FedAsyncFixedFast(pool_size=20, alpha=0.1)

# Ratio-based concurrency: set window_size = int(ratio * num_clients) in config
# e.g., 10% of 100 clients → async_fl.window_size: 10
```

**Selection modes compared:**

| Class | Sorts by | Best when |
|---|---|---|
| `FedAsync` | Random | Baseline / IID clients |
| `FedAsyncTopKFastTotal` | Compute + upload estimate (falls back to compute-only) | Fastest end-to-end clients (heterogeneous CPUs *and* channels) |
| `FedAsyncFixedFast` | Random within top-M fastest | Variety among fast clients |

### Running an async experiment

```python
from flsim.experiments.async_base import AsyncExperiment
from flsim.algorithms.fedasync import FedAsync, FedAsyncTopKFastTotal
from flsim.algorithms.fedavg import FedAvg

class MyExp(AsyncExperiment):
    def run(self):
        # Async run
        r_async = self.run_single_async(
            run_name="fedasync",
            label="FedAsync+Poly",
            config_overrides={
                "learning.global_rounds": 200,
                "async_fl.alpha":         0.1,
                "async_fl.window_size":   10,
                "evaluation.evaluate_every": 10,
            },
            components={
                "algorithm": FedAsync(alpha=0.1, staleness_func="polynomial", a=0.5),
            },
        )

        # Sync baseline for comparison
        r_sync = self.run_single(
            run_name="fedavg",
            label="FedAvg (sync)",
            config_overrides={"learning.global_rounds": 200},
            components={"algorithm": FedAvg()},
        )

        # Compare on accuracy vs simulated time
        self.plot_comparison(
            {"FedAsync+Poly": r_async, "FedAvg (sync)": r_sync},
            plot_configs=[
                {"metric": "test_accuracy", "x": "simulated_time_s", "ylabel": "Accuracy"},
                {"metric": "test_accuracy", "x": "round",            "ylabel": "Accuracy"},
            ],
        )

MyExp(
    base_config="flsim/configs/mnist_fedavg.yaml",
    output_dir="outputs/my_async_exp/",
).run()
```

### Writing a custom async algorithm

Every async algorithm subclasses `AsyncFederatedAlgorithm` from  
`flsim/interfaces/async_algorithm.py`.

There are **three override hooks** — all optional except `aggregate_async`:

| Method | Required | Purpose |
|---|---|---|
| `aggregate_async(global_model, update, global_epoch, staleness, alpha_t)` | **Yes** | How one arriving update changes the global model |
| `mixing_weight(base_alpha, staleness)` | No | Staleness-adaptive `alpha_t = base_alpha * s(staleness)`. Default: constant. |
| `select_clients(all_clients, num_to_trigger, rng, **kwargs)` | No | Which client to dispatch next. Default: uniform random. |
| `configure_client(client, global_model, global_epoch)` | No | Per-dispatch client setup. Default: no-op. |

#### Pattern A — only change the update rule

```python
from flsim.interfaces.async_algorithm import AsyncFederatedAlgorithm
from collections import OrderedDict

class MomentumAsync(AsyncFederatedAlgorithm):
    """Adds a momentum buffer on top of the standard FedAsync mixing."""

    def __init__(self, alpha=0.1, beta=0.9):
        self.alpha = alpha
        self.beta  = beta
        self._v    = None   # momentum buffer (same shape as global model)

    def aggregate_async(self, global_model, update, global_epoch, staleness, alpha_t):
        current = global_model.state_dict()
        new_state = OrderedDict()
        for key in current:
            delta = update.state_dict[key].float() - current[key].float()
            if self._v is None:
                # Initialise momentum buffer on first call
                self._v = {k: t.clone().zero_() for k, t in current.items()}
            self._v[key] = self.beta * self._v[key] + (1 - self.beta) * delta
            new_state[key] = current[key].float() + alpha_t * self._v[key]
        return new_state

    # mixing_weight()  → inherited: constant (alpha_t = base_alpha)
    # select_clients() → inherited: uniform random
```

#### Pattern B — only change the staleness decay

```python
import math
from flsim.algorithms.fedasync import FedAsync   # inherit the standard mixing rule

class ExponentialDecay(FedAsync):
    """alpha_t = alpha * exp(-lambda * staleness)"""

    def __init__(self, alpha=0.1, lam=0.05):
        super().__init__(alpha=alpha)
        self.lam = lam

    def mixing_weight(self, base_alpha, staleness):
        return base_alpha * math.exp(-self.lam * staleness)

    # aggregate_async() → inherited from FedAsync: standard (1-α)x + αx_new
    # select_clients()  → inherited: uniform random
```

#### Pattern C — only change client selection

```python
from flsim.algorithms.fedasync import FedAsync

class RoundRobinAsync(FedAsync):
    """Dispatch clients in strict round-robin order."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rr_idx = 0

    def select_clients(self, all_clients, num_to_trigger, rng, **kwargs):
        selected = []
        for i in range(num_to_trigger):
            selected.append(all_clients[self._rr_idx % len(all_clients)])
            self._rr_idx += 1
        return selected

    # mixing_weight()   → inherited from FedAsync (constant by default)
    # aggregate_async() → inherited from FedAsync: standard mixing
```

#### Pattern D — full custom algorithm (all three hooks)

```python
import math
from flsim.interfaces.async_algorithm import AsyncFederatedAlgorithm
from collections import OrderedDict

class FullCustomAsync(AsyncFederatedAlgorithm):
    """
    Custom async algorithm:
      - Selects: always the single fastest client
      - Decay:   exponential staleness
      - Update:  weighted by num_samples as well as alpha_t
    """

    def __init__(self, alpha=0.1, lam=0.1):
        self.alpha = alpha
        self.lam   = lam

    def select_clients(self, all_clients, num_to_trigger, rng, **kwargs):
        # Always pick the num_to_trigger fastest (by compute estimate)
        return sorted(all_clients,
                      key=lambda c: c.profile.cycles_per_sample *
                                    c.num_samples /
                                    c.profile.cpu_frequency_hz
                      )[:num_to_trigger]

    def mixing_weight(self, base_alpha, staleness):
        return base_alpha * math.exp(-self.lam * staleness)

    def aggregate_async(self, global_model, update, global_epoch, staleness, alpha_t):
        # Standard FedAsync mixing; alpha_t already decayed by mixing_weight above
        current   = global_model.state_dict()
        new_state = OrderedDict()
        for key in current:
            new_state[key] = (
                (1.0 - alpha_t) * current[key].float()
                + alpha_t       * update.state_dict[key].float()
            )
        return new_state
```

#### Wiring a custom algorithm

No factory registration needed — pass it directly via `components`:

```python
r = exp.run_single_async(
    "my_custom_alg",
    components={"algorithm": FullCustomAsync(alpha=0.15, lam=0.08)},
    config_overrides={
        "async_fl.window_size": 5,
        "learning.global_rounds": 300,
    },
)
```

### Async config reference

Set these in your YAML or via `config_overrides`:

```yaml
async_fl:
  alpha:       0.1   # base mixing hyperparameter α ∈ (0,1)
                     # smaller → conservative/slow, larger → aggressive/fast
  window_size: 10    # concurrent in-flight clients (= concurrency level)
                     # defaults to learning.clients_per_round if not set
                     # effective_bw_per_client = total_bandwidth / window_size
```

Useful `config_overrides` for async experiments:

| Key | Effect |
|---|---|
| `"async_fl.alpha": 0.2` | More aggressive model mixing |
| `"async_fl.window_size": 20` | More concurrency → more staleness |
| `"learning.global_rounds": 500` | Total server updates (not sync rounds) |
| `"learning.local_epochs": 3` | Local SGD steps per client dispatch |
| `"evaluation.evaluate_every": 50` | Evaluate every 50 global epochs |

### Async CSV columns

One row per global epoch (= one server update). Key columns:

| Column | Description |
|---|---|
| `global_epoch` | Server update counter (0-based) |
| `simulated_time_s` | Virtual wall-clock time of this update's arrival |
| `staleness` | `t − τ`: how many model versions behind this update is |
| `alpha_used` | Effective mixing weight `α_t = base_alpha × s(staleness)` |
| `client_id` | Which client sent this update |
| `compute_time_s` | Simulated training time for this client |
| `upload_time_s` | Simulated upload time for this client |
| `total_time_s` | `compute + upload + download` for this client |
| `total_energy_j` | Energy consumed by this client this epoch (compute + TX) |
| `cumulative_energy_j` | Running total energy across all epochs so far |
| `train_loss` | Training loss of this arriving update |
| `test_accuracy` | Global model accuracy (only at evaluation epochs) |

> **Comparing energy fairly (sync vs async).** Per-step energy is not comparable
> — a sync round logs ~`clients_per_round` clients' energy, an async epoch logs
> just one. Use the `cumulative_energy_j` column (present in **both** sync and
> async CSVs) for a fair energy curve:
>
> ```python
> self.plot_comparison(results, plot_configs=[
>     {"metric": "cumulative_energy_j", "x": "round",            "ylabel": "Cumulative energy (J)"},
>     {"metric": "cumulative_energy_j", "x": "simulated_time_s", "ylabel": "Cumulative energy (J)"},
> ])
> ```

### Async plots generated automatically

| File | Description |
|---|---|
| `acc_vs_epoch.png` | Test accuracy vs global epoch |
| `acc_vs_time.png` | Test accuracy vs simulated wall-clock time |
| `test_loss_vs_epoch.png` | Test loss vs global epoch |
| `train_loss_vs_epoch.png` | Training loss of each arriving update |
| `staleness_vs_epoch.png` | Staleness `t − τ` over training |
| `alpha_vs_epoch.png` | Effective `α_t` over training |
| `energy_vs_epoch.png` | Per-epoch energy cost |

---

## Writing a custom resource allocator

Every allocator subclasses `ResourceAllocator` from `flsim/interfaces/allocator.py`.
All three methods have implementations in `EqualSplitAllocator` — subclass it and
override only what you optimise.

```python
# flsim/allocators/channel_proportional.py
from flsim.allocators.equal_split import EqualSplitAllocator

class ChannelProportionalBW(EqualSplitAllocator):
    """
    Allocate bandwidth proportional to channel gain:
        b_k = B · g_k / sum(g_j)
    Clients with better channels get more bandwidth.
    """

    def allocate_bandwidth(self, selected_profiles, total_bandwidth_hz, **kwargs):
        gains = kwargs.get("channel_gains", {})   # {client_id: float}
        total_gain = sum(gains.get(p.client_id, 1.0) for p in selected_profiles)
        if total_gain == 0:
            return super().allocate_bandwidth(selected_profiles, total_bandwidth_hz)
        return {
            p.client_id: total_bandwidth_hz * gains.get(p.client_id, 1.0) / total_gain
            for p in selected_profiles
        }

    # allocate_power()    → inherited: all clients at p_max
    # allocate_cpu_freq() → inherited: profile frequency
```

Wire it in `_make_allocator()` in `flsim/experiments/wiring.py` (or inject directly in an experiment):

```python
components = {"allocator": ChannelProportionalBW()}
```

---

## Writing experiments

### Option 1 — bare custom experiment

```python
# my_experiment.py
from flsim.experiments import Experiment
from flsim.algorithms.fedavg import FedAvg

class MyExp(Experiment):
    def run(self):
        result = self.run_single(
            run_name         = "fedavg_run",
            label            = "FedAvg",
            config_overrides = {"learning.global_rounds": 50},
            components       = {"algorithm": FedAvg()},
        )
        self.plot_single(result)   # plots driven by YAML plots: list

MyExp(
    base_config = "flsim/configs/mnist_fedavg.yaml",
    output_dir  = "outputs/my_exp/",
).run()
```

### Option 2 — compare algorithms

```python
from flsim.experiments import AlgorithmComparison
from flsim.algorithms.fedavg import FedAvg

AlgorithmComparison(
    base_config = "flsim/configs/mnist_fedavg.yaml",
    output_dir  = "outputs/compare/",
    algorithms  = {
        "FedAvg":       FedAvg(),
        "FedMedian":    FedMedian(),
        "ChannelAware": ChannelAwareSelection(),
    },
    config_overrides = {"learning.global_rounds": 100},
).run()
```

Output per-run CSVs:
```
outputs/compare/fedavg/fedavg.csv
outputs/compare/fedmedian/fedmedian.csv
outputs/compare/channelaware/channelaware.csv
```

Comparison plots (one per entry in `plots:` list):
```
outputs/compare/comparison_test_accuracy_vs_round.png
outputs/compare/comparison_total_energy_j_vs_round.png
...
outputs/compare/bar_final_accuracy.png
```

### Option 3 — parameter sweep

```python
from flsim.experiments import ParameterSweep

ParameterSweep(
    base_config = "flsim/configs/mnist_fedavg.yaml",
    output_dir  = "outputs/bw_sweep/",
    param       = "wireless.total_bandwidth_hz",
    values      = [5e6, 10e6, 20e6, 40e6],
    labels      = ["5 MHz", "10 MHz", "20 MHz", "40 MHz"],
    param_label = "Total Bandwidth",
).run()
```

Sweep plots:
```
outputs/bw_sweep/sweep_final_accuracy_vs_param.png
outputs/bw_sweep/sweep_total_energy_j_vs_param.png
...
```

---

## Config overrides (dot-notation)

Any experiment accepts `config_overrides` — a dict that patches config values without
editing a YAML file. Keys use dot-notation matching the YAML structure:

| Override | Effect |
|---|---|
| `"learning.global_rounds": 50` | Run for 50 rounds |
| `"learning.local_epochs": 5` | 5 local epochs per round |
| `"learning.learning_rate": 0.001` | Change SGD learning rate |
| `"learning.clients_per_round": 20` | Select 20 clients per round |
| `"wireless.total_bandwidth_hz": 1e7` | 10 MHz total bandwidth |
| `"wireless.channel_model": "exp_fading"` | Switch channel model |
| `"wireless.deployment_shape": "circle"` | Circular deployment |
| `"system.cpu_freq_mode": "discrete_ghz"` | Heterogeneous CPU freqs |
| `"data.partition": "iid"` | Switch to IID partition |
| `"experiment.seed": 123` | Change random seed |

---

## Controlling which plots are generated

Edit the `plots:` list in `flsim/configs/base.yaml` (or in your experiment YAML):

```yaml
plots:
  - metric: test_accuracy
    x: round
    ylabel: "Test Accuracy"
  - metric: test_accuracy
    x: simulated_time_s
    ylabel: "Test Accuracy vs Time"
  - metric: total_energy_j
    x: round
    ylabel: "Energy (J)"
    log_scale: false
  - metric: mean_rate_bps
    x: round
    ylabel: "Mean Rate (bps)"
  - metric: round_duration_s
    x: round
    ylabel: "Round Duration (s)"
```

Available `metric` values (CSV column names):

| Column | Description |
|---|---|
| `test_accuracy` | Global model test accuracy |
| `test_loss` | Global model test loss |
| `round_duration_s` | Simulated round duration (bottlenecked by slowest client) |
| `mean_compute_time_s` | Mean client computation time |
| `max_compute_time_s` | Max client computation time (= straggler) |
| `mean_upload_time_s` | Mean client upload time |
| `max_upload_time_s` | Max client upload time |
| `mean_compute_energy_j` | Mean client computation energy |
| `total_energy_j` | Total energy (compute + TX) summed over selected clients this round |
| `cumulative_energy_j` | Running total energy across all rounds so far |
| `mean_channel_gain` | Mean linear channel gain of selected clients |
| `mean_rate_bps` | Mean achievable uplink rate |

---

## Re-plotting from a saved CSV

```bash
# All plots from a single run
python plot_results.py outputs/fedAVG/fedavg/fedavg.csv

# Save plots to a specific folder
python plot_results.py outputs/fedAVG/fedavg/fedavg.csv --out figures/fedavg/
```

---

## Config reference

Key parameters in `flsim/configs/base.yaml`:

```yaml
data:
  dataset:     mnist | cifar10
  num_clients: 100
  partition:   iid | shard | dirichlet
  dirichlet_alpha: 0.5    # only for dirichlet
  num_shards: 200         # only for shard
  shards_per_client: 2    # only for shard

learning:
  algorithm:         fedavg
  global_rounds:     100
  clients_per_round: 10
  local_epochs:      5
  batch_size:        32
  learning_rate:     0.01

system:
  cpu_freq_mode:        fixed | discrete_ghz
  cpu_frequency_hz:     2.0e+9   # used when fixed
  cpu_freq_min_ghz:     0.1      # used when discrete_ghz
  cpu_freq_max_ghz:     0.8
  cpu_freq_step_ghz:    0.1
  cycles_per_sample_min: 1.0e+7
  cycles_per_sample_max: 1.0e+7
  switched_capacitance: 1.0e-28

wireless:
  channel_model:         path_loss | exp_fading
  deployment_shape:      square | circle
  area_side_m:           500.0    # used when square
  area_radius_m:         500.0    # used when circle
  total_bandwidth_hz:    2.0e+7
  tx_power_dbm:          10.0
  noise_psd_dbm_per_hz: -174.0
  upload_size_mode:      fixed | model
  upload_size_bits:      28100    # used when fixed
  downlink_negligible:   false    # true → model-broadcast (download) time = 0
                                  # applies to BOTH sync and async simulators

# Async FL only — ignored by synchronous Simulator
async_fl:
  alpha:       0.1   # base mixing weight α ∈ (0, 1)
  window_size: 10    # concurrent in-flight clients (default = clients_per_round)
```

**Downlink time.** By default download (server → client model broadcast) time is
computed symmetrically to upload. Set `wireless.downlink_negligible: true` (or
`--config-override` / `config_overrides={"wireless.downlink_negligible": True}`)
to treat the broadcast as instantaneous — a common FL assumption since the base
station has far more power/bandwidth than clients. Round duration (sync) and
arrival time (async) then count only **compute + upload**.
