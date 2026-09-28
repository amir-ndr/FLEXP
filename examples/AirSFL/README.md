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

## Code

| File | What it is | Verified by |
|---|---|---|
| `flsim/airsfl/timing.py` | Eq. 11-13, 25-26: analog symbol-count time, digital OFDMA fluid goodput (MRC at the receiving server), labels, per-phase breakdown, source MB | formula regression (earlier roadmap example), limit cases, each phase counted once |
| `flsim/airsfl/model.py` | ResNet-18: 3x3 stem, no max-pool, GroupNorm-32, no conv bias, cuts after stages 1-4 | roadmap d_c / d_s / d_a at all cuts |
| `flsim/airsfl/radio.py` | dimensional ZF / AirComp physics | gates 1-3, 5 (packing, packet power, C^H H = I, noiseless AirComp, zero packets); Monte-Carlo MSE vs Eq. 10 (0.11 %) and Eq. 23 (0.02 %) |
| `flsim/airsfl/data.py` | fixed stratified 45k/5k split, IID / Dirichlet(0.5) (≥ 1 minibatch per client, redraws logged), matched per-client streams, optional crop+flip | vectorized crop == per-image reference |
| `flsim/airsfl/simulator.py` | training for all 5 methods; joint ZF error per packet (shared M-server noise); AirComp error per packet; separate seeded M-link / F-link traces (paired across methods); two engines | see below |
| `flsim/airsfl/checks.py` | the verification suite | `python -m flsim.airsfl.checks` |

**Engines.** `vmap` (default) trains all N clients in parallel with `torch.func` (needs
torch ≥ 2.0): one forward/backward of Φ(h_c(x_c) + e) with the ZF error e detached,
which has identity Jacobian at the cut, so the prefix receives exactly the returned
derivative q̂ (Eq. 14-16). `loop` is the per-client relay implementation
(`relay.grad` → `z.backward(q̂)`), kept as the reference.

### Verification (`python -m flsim.airsfl.checks` — all PASS)
* relay back-prop == full-model back-prop at every cut: bitwise
* noiseless digital SFL-V1 (cuts 1-4), AirSFL, Sun, AirComp-FL == digital FedAvg: bitwise, both engines (gate 6)
* no-split limits with the analog errors on: SFL-V1 == FedAvg, AirSFL == AirComp-FL, Sun == AirComp-FL: bitwise
* vmap step with ZF error == float64 relay back-prop: 6e-15 relative (gate 7)
* empirical ZF activation NSR = 1/(ρ(Nr−N)): −20.78 vs −20.79 dB; zero packets → zero error

## Environment (defaults in `run_airsfl.py`, roadmap Sec. 2-3)

N=30 clients (sweep 20/30/40), Nr=64 at the M-server and the F-server, S=120 × 15 kHz
(W=1.8 MHz; OFDMA 6/4/3 tones per client), P_max=0.1 W per uplink stage (analog 0.1/120 W
per tone, digital 0.1/S_n W per tone), N0=−167 dBm/Hz, ε_D=ε_U=ε_A=0.6, ρ=20 dB
(λ_ref = ρ·N0·W/P_max, equal for all clients and both servers), block Rayleigh per
128-symbol packet, perfect CSI. Cut after stage 2, τ=5, B=16, plain SGD, cosine decay
to 1 % (fixed within a round), initial LR calibrated on the error-free digital reference
from {0.01, 0.03, 0.1, 0.3}, 100 global-epoch equivalents, evaluation once per
global-epoch equivalent, FP32 (TF32 disabled), seeds 11/22/33/44/55. Augmentation off
by default (`--augment` enables crop+flip).

Per-round uplink at these defaults (cut 2): AirSFL 1.53 s (act 1.21 + agg 0.31), digital
SFL-V1 167 s, Sun-inspired 133 s, AirComp-FL 5.17 s, digital FedAvg 566 s.

**Metric.** Accumulated uplink communication time (Eq. 28): client→M-server activations
+ labels, client→F-server model differences. Not end-to-end training time (computation
excluded, downlinks ideal). Source-equivalent MB is reported separately: the three SFL
variants have identical MB; AirSFL saves airtime, not bytes.

**Targets.** Prespecified validation accuracies (`plots.py --targets 0.6 0.7`; fix them
after the quick test, before the final runs). Attainment per seed = first checkpoint
with val acc ≥ target; failures are reported as k/n seeds; speed-ups are paired per seed.

## Experiments (`run_airsfl.py --exp ...`; every CSV keeps the full record)

| exp | runs | figures |
|---|---|---|
| `bench` | times both engines on the device | — |
| `lr` | FedAvg (error-free), IID, LR grid, 20 epochs | — |
| `main` | 5 methods × {IID, Dir-0.5} at 20 dB | fig1, fig1b, fig4b, fig6, fig8, table_main |
| `snr` | 3 analog methods × SNRs (default 0/10/30; `--snrs -20 -10 0 10 30`) | fig2, fig2b, table_snr |
| `nsweep` | 5 methods × N=20, 40 | fig7, table_nsweep |
| `cuts` | AirSFL + Sun × cuts 1, 3, 4 | fig9 |
| `tau` | 5 methods × τ = 1, 10 | fig9 |
| `nr` | AirSFL × Nr = 32, 48, 128 | fig5 |

Analytic figures (always drawn): fig3 (per-round phases vs N, roadmap A), fig3b (vs cut),
fig4a (bytes vs airtime), fig8 (efficiency 0.4/0.6/0.7 and ideal ε=1, from the main runs).
Digital learning does not depend on SNR, cut, ε or Nr (ideal decoding), so for those
sweeps only its time axis is recomputed from the same runs.

On the cluster: see the header of `slurm/run_airsfl.slurm` (quick test, then full runs).
