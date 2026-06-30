# flsim — Federated Learning Simulator

A research-grade FL simulator with clean 4-layer architecture designed for easy extension.  
Write a new algorithm, allocator, or experiment by overriding only the methods that change.

---

## Quick start

```bash
# Install
pip install -e flsim/

# Run the canonical FedAvg experiment
python examples/fedavg_experiment.py

# With CLI overrides
python examples/fedavg_experiment.py --rounds 50 --clients_per_round 5 --lr 0.005

# Re-plot any saved CSV
python plot_results.py outputs/fedAVG/fedavg/fedavg.csv
```

> **Note:** There is no `run.py`. The entry point is always an experiment file
> in `examples/` (or your own script). All wiring utilities live in `flsim/experiments/wiring.py`.

---

## Project layout

```
FLEXP/
├── flsim/
│   ├── interfaces/          # ABCs — the contracts everything must satisfy
│   │   ├── algorithm.py       FederatedAlgorithm (select_clients, configure_client, aggregate)
│   │   ├── allocator.py       ResourceAllocator  (allocate_bandwidth, allocate_power, allocate_cpu_freq)
│   │   ├── channel_model.py   ChannelModel       (channel_gain, achievable_rate_bps)
│   │   ├── time_model.py      TimeModel          (compute_training_time, compute_upload_time, …)
│   │   └── partitioner.py     DataPartitioner    (partition, describe)
│   │
│   ├── algorithms/          # FL algorithms
│   │   └── fedavg.py          FedAvg (sample-weighted aggregation)
│   │
│   ├── allocators/          # Resource allocation policies
│   │   └── equal_split.py     EqualSplitAllocator (FDMA equal split, max power, profile freq)
│   │
│   ├── channel/             # Channel physics
│   │   ├── path_loss.py       3GPP UMa + frozen log-normal shadowing
│   │   ├── fdma.py            Alias for PathLossChannelModel
│   │   └── exp_fading.py      h = h0·ρ·d⁻², ρ~Exp(1) per round
│   │
│   ├── system/              # Computation + energy formulas
│   │   ├── cellular_time.py   τ = (I·C·D)/f
│   │   └── energy.py          E_comp = κ·I·C·D·f²,  E_tx = p·t_up
│   │
│   ├── data/                # Dataset loaders + partitioners
│   ├── models/              # PyTorch model definitions
│   ├── profiles/            # Client system profiles (distance, freq, power)
│   ├── core/                # Simulator, Server, Client, Evaluator, Logger
│   ├── configs/             # YAML experiment configs
│   ├── experiments/         # Experiment runner classes
│   │   ├── base.py            Experiment base + RunResult
│   │   ├── compare_algorithms.py  AlgorithmComparison
│   │   └── parameter_sweep.py     ParameterSweep
│   │   └── wiring.py          Config loading + component factories (add new algs/allocators here)
│   └── tests/               # 55 unit tests
│
├── examples/
│   └── fedavg_experiment.py   Canonical FedAvg run
│
├── plot_results.py            Standalone CSV → plots tool
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

    def select_clients(self, all_clients, num_to_select, rng):
        sorted_clients = sorted(all_clients, key=lambda c: c.profile.distance_m)
        return sorted_clients[:num_to_select]

    # aggregate()        → inherited from FedAvg (sample-weighted average)
    # configure_client() → inherited: no-op
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
| `total_energy_j` | Total energy (compute + TX) summed over selected clients |
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
```
