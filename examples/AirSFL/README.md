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
| `flsim/airsfl/timing.py` | OFDM analog/digital uplink time (Table I), source-equivalent MB (a downlink model is kept but not used in the figures) | reproduces the roadmap example exactly: c_D=14.62, AirSFL 2.66 / SFL-V1 93.2 / Sun 31.7 / AirComp-FL 7.27 s/round; Table I limit cases |
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
P_max=0.1 W, N0=−167 dBm/Hz, ε_D=ε_U=ε_A=0.8, ρ=20 dB (implied path loss
λ = ρN0W/Pmax = 107.5 dB, equal for all clients), block Rayleigh per 128-symbol
packet; cut after stage 3, τ=5, B=16, plain SGD, 50 global-epoch equivalents,
2 evaluations per epoch, seeds 11/22/33. LR: calibrated on the noiseless
reference from {0.05, 0.1, 0.2, 0.4} (10 epochs), 1-epoch linear warmup,
×0.1 at 50 % and 75 % of the rounds. SNR sweep ρ ∈ {−30, −20, −10, 0, 10, 20} dB:
the ZF activation NSR is 1/(ρ(Nr−N)), i.e. −40 dB at 20 dB, so the accuracy
cost of analog transport only appears at negative ρ.

**Fairness.** Every method uses the same data split, partition, initialization,
per-client minibatch *and augmentation* streams (own seeded generator per
client), LR schedule, SGD steps, weights a_n = D_n/ΣD, bandwidth, per-client
power and antennas. Digital transports are ideal (error-free FP32, q=32, at
Shannon goodput; favourable to digital); analog ones carry ZF/AirComp distortion
(perfect CSI, favourable to analog). That trade-off is what is measured.
Divergent runs (non-finite loss) stop and count as "target not reached".

**Metric.** Modeled **uplink communication seconds** (Table I), accumulated per
round. Each row contains **both** client→server phases: the τ co-split
activation uploads (client → main server) and the once-per-round model upload
(client → fed server; the two server roles are co-located at the edge node). SFL
suffix copies are averaged locally at the server (no wireless traffic). The
downlink (cut derivatives, model broadcast) is a reliable decoded link and is not
timed. Computation time is excluded. Source-equivalent MB is reported separately;
the three SFL variants have **identical** MB, since AirSFL saves airtime, not bytes.

**Uplink rates.** Digital transports use Shannon goodput on S_n = S/N OFDMA tones
per client (Eq. 22-23, per-tone SNR ρN·X, X~Gamma(Nr,1), q=32 bits/value). Analog
transports (AirSFL ZF activations, AirComp aggregation) have **no Shannon rate**:
time is OFDM symbol count, 2 real values per tone per symbol on all S tones shared
by all clients (Eq. 24), and the SNR enters only as distortion (Eq. 10, 20). When
N does not divide S (N=40 in the N sweep), S_n is fractional (time-shared tones).

**Target accuracy A\*.** The integer percent below 95 % of the noiseless
reference's final validation accuracy (seed mean, per partition), frozen for all
methods. Attainment is per seed: the first evaluation at or above A\*. Times are
averaged over the seeds that reached A\* and the success count is reported;
failures are never assigned a finite time.

## Running

```bash
python -m flsim.airsfl.checks                                    # verification (CPU, ~minutes)
python examples/AirSFL/run_airsfl.py --exp lr                    # 1) LR calibration (run first)
python examples/AirSFL/run_airsfl.py --exp main --seeds 11 22 33 # 2) 5 methods x {IID, non-IID}
python examples/AirSFL/run_airsfl.py --exp snr  --seeds 11 22 33 # 3) SNR sweep (analog methods)
python examples/AirSFL/plots.py                                  # figures + tables
```
Optional: `--exp nr` (antenna sweep), `--exp nsweep` (N = 20, 40). On the cluster see
the header of `slurm/run_airsfl.slurm`. Plots load only CSVs written by the current
schedule and trained with the calibrated LR.

## Outputs (`examples/AirSFL/results/`)

| Figure | Question it answers |
|---|---|
| `fig1_acc_vs_uplink_time` | How fast does each method reach the target, in uplink seconds? (IID and non-IID, seed band) |
| `fig1b_acc_vs_rounds` | Do the methods learn the same per round? (gaps = analog distortion) |
| `fig2_snr` | Final accuracy and uplink time to target vs SNR (−30…20 dB). Digital learning is SNR-independent; only its rate changes. |
| `fig2b_airsfl_snr_curves` | AirSFL accuracy vs uplink time at every SNR, with digital SFL-V1 at the lowest SNR and at 20 dB |
| `fig3_uplink_latency` | Per-round uplink time by phase (activations vs client→fed-server model) and vs cut (analytic) |
| `fig4_comm_overhead` | Same source-equivalent MB, very different airtime; GB needed to reach the target |
| `fig6_time_to_target` | Headline bars: uplink time to A\* per method, with the speed-up of AirSFL |
| `fig5_nr`, `fig7_nsweep` | Optional antenna and client-count sweeps (fig7(a) is analytic and always drawn) |

Tables: `table_main` (per method: s/round split by phase, MB/round, final accuracy with seed range, target reached k/n, rounds / seconds / MB to A\*, speed-up vs digital SFL-V1), `table_snr`, `table_cuts`, `table_nsweep`.
