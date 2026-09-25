# AirSFL — Split FL with Zero-Forcing reception and AirComp aggregation

Implementation and evaluation of **AirSFL** (WCNC methods draft + evaluation
roadmap) against four baselines, on **CIFAR-10 / ResNet-18 (GroupNorm)**.

| Method | Activation uplink (τ per round) | Prefix / model uplink (once per round) |
|---|---|---|
| **AirSFL** (proposed) | analog OFDM, **ZF** separates the N streams (noisy, Eq. 9-10) | analog **AirComp** weighted sum (noisy, Eq. 16-20) |
| Digital SFL-V1 | digital OFDMA, reliable FP32 | digital OFDMA, exact |
| FDMA-AirComp SFL (Sun-style control) | digital OFDMA, reliable FP32 | analog AirComp (noisy) |
| AirComp-FL | — (no split) | analog AirComp of the **full** model |
| Digital FedAvg (control) | — (no split) | digital OFDMA, full model |

All split methods are **SFL-V1**: the server keeps one suffix copy per client,
averaged exactly at the round boundary.

## Code

| File | What it is | How it was verified |
|---|---|---|
| `flsim/airsfl/timing.py` | OFDM analog/digital uplink time (Table I), reciprocity downlink, source-equivalent MB | reproduces the roadmap example exactly: c_D=14.62, AirSFL 2.66 / SFL-V1 93.2 / Sun 31.7 / AirComp-FL 7.27 s/round; Table I limit cases |
| `flsim/airsfl/model.py` | CIFAR ResNet-18: 3x3 stem, no max-pool, GroupNorm-32, cuts after stages 1-4 | reproduces the roadmap's profiled d_c / d_s / d_a at all 4 cuts |
| `flsim/airsfl/radio.py` | MIMO Rayleigh, ZF recovery, AirComp combiner (dimensional Eq. 9-20) | Monte-Carlo MSE vs Eq. 10 (0.11 %) and Eq. 20 (0.02 %) |
| `flsim/airsfl/data.py` | 45k/5k stratified split (seed 2026), train-only crop+flip, IID / Dirichlet(α), matched per-client streams | — |
| `flsim/airsfl/simulator.py` | SFL-V1 training for all 5 methods, joint ZF noise per packet, per-packet AirComp noise, uplink-time metric | see checks below |
| `flsim/airsfl/checks.py` | the full verification suite | `python -m flsim.airsfl.checks` |

### Verification (`python -m flsim.airsfl.checks` — all PASS)

* relay back-prop (Eq. 11-13) == full-model back-prop at every cut: **bitwise**
* **noiseless** digital SFL-V1 (cuts 1-4), AirSFL, Sun-style, AirComp-FL == digital FedAvg: **bitwise**
* **no-split limits with the noise on** (cut after the last layer):
  SFL-V1 == FedAvg, AirSFL == AirComp-FL, Sun == AirComp-FL: **bitwise**
* empirical ZF activation NSR = 1/(ρ(Nr−N)) (complex-Wishart mean): −20.78 vs −20.79 dB

## Environment (defaults in `run_airsfl.py`)

N=30 clients, Nr=128 server antennas, S=60 subcarriers × 15 kHz (W=0.9 MHz),
P_max=0.1 W, N0=−167 dBm/Hz, ε_D=ε_U=ε_A=0.8, ρ=20 dB, block Rayleigh per
128-symbol packet; cut after stage 3, τ=5, B=16, plain SGD, 50 global-epoch
equivalents, LR from the calibration grid {0.01, 0.03, 0.1}, LR ×0.1 at 50 % and 75 %.

**Fairness.** Every method uses the same data split, partition, initialization,
per-client minibatch *and augmentation* streams (own seeded generator per
client), SGD steps, weights a_n = D_n/ΣD, bandwidth, per-client power and
antennas. Digital transports are ideal (error-free FP32 at Shannon goodput,
favourable to digital); analog ones carry ZF/AirComp distortion. That trade-off
is exactly what is being measured.

**Metric.** Modeled **uplink communication seconds** (Table I), accumulated per
round. Computation time is excluded. Source-equivalent MB is reported separately;
the three SFL variants have **identical** MB, since AirSFL saves airtime, not
bytes.

**Uplink rates.** Digital transports use Shannon goodput on S_n = S/N OFDMA tones
per client (Eq. 22-23, per-tone SNR ρN·X, X~Gamma(Nr,1), q=32 bits/value). Analog
transports (AirSFL ZF activations, AirComp aggregation) have **no Shannon rate**:
time is OFDM symbol count, 2 real values per tone per symbol on all S tones shared
by all clients (Eq. 24), and the SNR enters only as distortion (Eq. 10, 20). When
N does not divide S (N=40 in the N sweep), S_n is fractional (time-shared tones).

**Downlink (sensitivity, `fig6`).** Paper Eq. 25: distinct cut derivatives go over
OMA downlinks at R̄ᴰᴸ_n, the prefix/model over a common broadcast at R_bc; identical
for all SFL variants. Default R̄ᴰᴸ_n mirrors Eq. 22 with BS power 0.3 W per client
(SAFSL convention; AdaptSFL also uses a separate downlink), so it is only
slightly faster than the uplink (log rate). R_bc is bracketed: per-client
(R_bc = R̄ᴰᴸ_n) and full-band (R_bc = N·R̄ᴰᴸ_n). The bracket decides AirSFL vs
AirComp-FL in two-way time (downlink analogue of Eq. 27: τ·d_a·R_bc/R̄ᴰᴸ_n < d_s).

**Target accuracy A\*.** The integer percent below 95 % of the noiseless
reference's final validation accuracy (per partition), frozen for all
methods. Attainment is the first evaluation at or above A\*. Failures are
reported, never assigned a finite time.

## Running

```bash
python -m flsim.airsfl.checks                         # verification (CPU, ~minutes)
python examples/AirSFL/run_airsfl.py --exp lr         # 1) LR calibration (run first)
python examples/AirSFL/run_airsfl.py --exp main       # 2) 5 methods x {IID, non-IID}
python examples/AirSFL/run_airsfl.py --exp snr nr     # 3) SNR sweep, antenna sweep
python examples/AirSFL/run_airsfl.py --exp nsweep     # 4) N = 20, 40 (N = 30 from main)
python examples/AirSFL/plots.py                       # figures + tables
```
On the cluster: `sbatch --export=ALL,EXP="main" slurm/run_airsfl.slurm` (see the file header).

## Outputs (`examples/AirSFL/results/`)

| Figure | Question it answers |
|---|---|
| `fig1_acc_vs_uplink_time` | How fast does each method reach a target accuracy, in uplink seconds? (IID and non-IID) |
| `fig2_snr` | How does analog noise (0–30 dB) trade accuracy against time? Digital learning is SNR-independent; only its rate changes. |
| `fig3_uplink_latency` | Where does per-round uplink time go (activation vs model), and how does it vary with the cut? |
| `fig4_comm_overhead` | Same source-equivalent MB, very different airtime; MB needed to reach the target |
| `fig5_nr` | How many receive antennas does AirSFL need (ZF conditioning as Nr → N)? |
| `fig6_two_way` | Does the advantage survive once downlink time is counted? (time to A\*: uplink only / two-way with both broadcast bounds) |
| `fig7_nsweep` | How does per-round uplink time and time to A\* scale with N = 20, 30, 40 at fixed Nr and W? |

Tables: `table_main` (per method: s/round, MB/round, final accuracy, rounds / seconds / MB to A\*, speed-up vs digital SFL-V1), `table_snr`, `table_cuts`, `table_two_way`, `table_nsweep`.
