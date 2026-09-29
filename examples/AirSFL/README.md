# AirSFL — Split FL with Zero-Forcing reception and AirComp aggregation

Implementation and evaluation of **AirSFL** (WCNC draft, two-server version, and its
evaluation roadmap) against four baselines on **CIFAR-10 / ResNet-18 (GroupNorm)**.

Two servers: the **M-server** receives activations (ZF, 64 antennas), keeps one suffix
copy per client and returns cut derivatives; the **F-server** receives prefix differences
(AirComp, 64 antennas) and broadcasts the prefix. Both downlinks are ideal (reliable,
non-bottleneck, not timed for any method).

| Method | Activation uplink → M-server (τ per round) | Model uplink → F-server (once per round) |
|---|---|---|
| **AirSFL** (proposed) | analog OFDM, **ZF** separates the N streams (Eq. 9-10) | analog **AirComp** weighted sum (Eq. 19-23) |
| Digital SFL-V1 | digital OFDMA, reliable FP32 | digital OFDMA, exact |
| FDMA-AirComp SFL (Sun-inspired, matched) | digital OFDMA, reliable FP32 | same AirComp receiver as AirSFL |
| AirComp-FL | — (no split) | AirComp of the **full** model difference |
| Digital FedAvg (control) | — (no split) | digital OFDMA, full model |

All split methods are equal-period **SFL-V1**: suffix copies stay independent for τ
steps and are averaged exactly (locally at the M-server) at the round boundary.

## Time model (every CSV row carries all of it, per phase and cumulative)

**Uplink** (paper Eq. 11-13, 25-26): analog time = OFDM symbol count, digital time =
bits / Shannon fluid goodput on S/N tones with MRC at the receiving server.

**Computation** (`flsim/airsfl/compute.py`, AdaptSFL-style sequential stages of every
co-split step; the slowest client paces each step):

| | per round |
|---|---|
| client FP | τ · b · C_FP / f_min |
| M-server FP + BP | τ · **N** · b · (Φ_FP + Φ_BP) / f_s — the M-server runs all N suffix copies on its shared capacity |
| client BP | τ · b · C_BP / f_min |
| FL (AirComp-FL, FedAvg) | τ · b · (W_FP + W_BP) / f_min — the client trains the whole model |

FLOPs per sample = 2 × Conv2d/Linear MACs (the framework's convention); BP = weight
gradient + input gradient (the first convolution needs no input gradient). Prefix +
suffix FLOPs equal the full model's at every cut. At cut 2: client 0.574 / 1.144 GFLOPs
(FP / BP), server 0.537 / 1.074 GFLOPs per sample. Capabilities (AdaptSFL setting):
f_i ~ U[1, 2] TFLOPS per client (drawn once per seed, identical for all methods),
f_s = 20 TFLOPS. Server aggregation and ZF/AirComp receiver processing cost
microseconds and are omitted. `plots.py --client-tflops LO HI --server-tflops F`
recomputes computation exactly from the stored FLOP counts (no retraining).

**Training time** = uplink + computation (downlinks ideal). Per round at the defaults
(slowest client 1 TFLOPS): AirSFL 1.53 + 0.33 s, digital SFL-V1 167.1 + 0.33 s,
Sun-inspired 133.2 + 0.33 s, AirComp-FL 5.17 + 0.27 s, digital FedAvg 566.3 + 0.27 s.

## Code

| File | What it is | Verified by |
|---|---|---|
| `flsim/airsfl/timing.py` | uplink time per phase, source MB | formula regression, limit cases, each phase counted once |
| `flsim/airsfl/compute.py` | FLOP profile per cut, computation time per phase | closed-form MACs, framework counter, prefix+suffix = full, N=1 limit, hand calculation |
| `flsim/airsfl/model.py` | ResNet-18: 3x3 stem, no max-pool, GroupNorm-32, cuts after stages 1-4 | roadmap d_c / d_s / d_a |
| `flsim/airsfl/radio.py` | dimensional ZF / AirComp physics | gates (packing, power, C^H H = I, noiseless AirComp, zero packets); MC MSE vs Eq. 10 / 23 |
| `flsim/airsfl/data.py` | 45k/5k split, IID / Dirichlet(0.5), matched per-client streams, crop+flip | vectorized crop == reference |
| `flsim/airsfl/simulator.py` | training for all 5 methods; joint ZF error per packet; AirComp error; radio streams seeded by (seed, link, round, step); two engines | `python -m flsim.airsfl.checks` |

**Engines.** `loop` (default: least memory, robust on shared GPUs) trains clients one
after another with relay back-prop; `vmap` trains all clients in parallel (torch.func).
Both equal float64 relay back-prop to 6e-15; `--engine vmap --fast-cudnn` was the
fastest setting on an idle A30 (1.53 vs 1.77 s/round).

## Environment (defaults in `run_airsfl.py`)

N=30 clients, Nr=64 at the M-server and the F-server, S=120 × 15 kHz (W=1.8 MHz), P_max=0.1 W,
N0=−167 dBm/Hz, ε=0.6, ρ=20 dB, block Rayleigh per 128-symbol packet, perfect CSI. Cut
after stage 2, τ=5, B=16, plain SGD, cosine decay to 1 % (fixed within a round), initial LR
calibrated on the error-free digital reference from {0.01, 0.03, 0.1, 0.3}, random crop +
flip, evaluation once per global-epoch equivalent, FP32 (TF32 disabled). SNR sweep
ρ ∈ {−20, −10, 0, 10, 20} dB.

**Run identity.** Every run's full resolved configuration (radio, training, schedule,
horizon, data/partition, augmentation, evaluation sizes, schema) is hashed; the hash is in
the file name and the configuration is saved as `<run>.json`. A finished run is reused only
when the configuration is identical; an LR calibration is refused under another setup.

**Targets.** Prespecified validation accuracies (`plots.py --targets 0.75 0.8`); curves show
test accuracy, markers the checkpoint where validation reached the target; failures are
reported as k/n seeds; speed-ups are paired per seed.

## Experiments (`run_airsfl.py --exp ...`)

| exp | runs | figures |
|---|---|---|
| `bench` | seconds/round of both engines × cuDNN mode | — |
| `lr` | FedAvg (error-free), IID, LR grid, 20 epochs | — |
| `main` | 5 methods × {IID, Dir-0.5} at 20 dB | fig1, 1b, 1c, 4b, 6, 8, table_main |
| `snr` | 3 analog methods × {−20, −10, 0, 10} dB | fig2, 2b, 2c, table_snr, table_budget |
| `nsweep` | 5 methods × N=20, 40 | fig7 |
| `cuts` | AirSFL + Sun × cuts 1, 3, 4 | fig9 |
| `tau` | 5 methods × τ = 1, 10, 20, 50 | fig9 |
| `nr` | AirSFL × Nr_M = 32, 40, 48 (Nr_F = 64) | fig5 |

Analytic figures (always drawn): fig3 (uplink phases vs N), fig3b (vs cut), fig3c (uplink +
computation per round and its composition), fig4a (bytes vs airtime). fig2c plots the test
accuracy reached within a fixed uplink / training-time budget vs SNR (`--budget-uplink`,
`--budget-training`; default: AirSFL's full-run time at 20 dB).

On the cluster: see the header of `slurm/run_airsfl.slurm`.
