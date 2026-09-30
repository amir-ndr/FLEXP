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
| Digital SFL-V1 (OFDMA) | digital OFDMA, reliable FP16 | digital OFDMA, FP16 |
| Hybrid FDMA-AirComp SFL (Sun-inspired, matched) | digital OFDMA, reliable FP16 | same AirComp receiver as AirSFL |
| AirComp-FL | — (no split) | AirComp of the **full** model difference |
| Digital FedAvg (OFDMA, control) | — (no split) | digital OFDMA, full model, FP16 |
| Digital SFL-V1 (ZF) † | digital **multi-user ZF**, reliable FP16 | digital multi-user ZF, FP16 |
| Hybrid ZF-AirComp SFL † | digital **multi-user ZF**, reliable FP16 | same AirComp receiver as AirSFL |

All split methods are equal-period **SFL-V1**: suffix copies stay independent for τ
steps and are averaged exactly (locally at the M-server) at the round boundary.

**Digital payload: FP16 (default).** Every tensor a digital baseline uploads (activations,
prefix / model differences) is sent as FP16 (q = 16 bits per value) and is *really rounded*
in training (draft: "a 16-bit transport check must actually round transmitted tensors"): per
256-value block the block maximum is side information and the values travel as IEEE half,
which avoids the FP16 underflow of small late-training model differences (rounding error
≈ −74 dB of the activations, ≈ −77 dB of the aggregates). Labels (integer class indices)
and the analog stages are unaffected, so AirSFL and AirComp-FL have the same runs for both
payloads. FP32 (`--digital-q 32`) remains available: its digital SFL-V1 runs are the exact
error-free reference, its FedAvg runs the reference of the target rule, and `fig12` compares
the two payloads.

† Stronger digital baselines (non-orthogonal access): every client sends its own coded
stream on all S tones and the server separates the N streams with the same ZF filter
AirSFL uses. Their learning is *identical* to Digital SFL-V1 / Hybrid FDMA-AirComp SFL
(reliable digital links either way, same AirComp stream), so `plots.py` derives them from
those runs with the ZF uplink time — no extra GPU runs. `python -m flsim.airsfl.checks`
trains them for real and confirms the equality bit for bit.

## Time model (every CSV row carries all of it, per phase and cumulative)

**Uplink** (paper Eq. 11-13, 25-26): analog time = OFDM symbol count, digital time =
q · values / Shannon fluid goodput (q = 16). Digital OFDMA: S/N tones per client, MRC at the receiving
server, per-tone SNR ρ·N·X with X ~ Gamma(Nr, 1). Digital multi-user ZF: all S tones per
client, post-ZF SNR ρ·Y with Y ~ Gamma(Nr − N + 1, 1), goodput ε_D·Δf·S·E[log2(1 + ρY)]
(= OFDMA at N = 1; the Gamma law is checked against explicit ZF channels).

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
(20 dB, slowest client 1 TFLOPS; FP32 payload in brackets):

| | uplink s/round | × AirSFL | + computation |
|---|---|---|---|
| AirSFL | 1.53 | 1 | + 0.33 |
| AirComp-FL | 5.17 | 3.4 | + 0.27 |
| Hybrid ZF-AirComp SFL | 3.62 (6.92) | 2.4 (4.5) | + 0.33 |
| Digital SFL-V1 (ZF) | 4.16 (8.31) | 2.7 (5.4) | + 0.33 |
| Hybrid FDMA-AirComp SFL | 66.7 (133.2) | 44 (87) | + 0.33 |
| Digital SFL-V1 (OFDMA) | 83.5 (167.1) | 55 (109) | + 0.33 |
| Digital FedAvg (OFDMA) | 283.1 (566.3) | 185 (371) | + 0.27 |

The digital rates fall with the SNR while the analog airtime does not: at −20 dB Digital
SFL-V1 (ZF) needs 113 s/round and Hybrid ZF-AirComp 90 s/round (AirSFL still 1.53 s,
but with activation distortion).
Whether computation matters depends on the devices: `fig10_compute_regimes` and
`fig1d_acc_vs_training_time_iot` recompute the same runs with the IoT-CPU setting of Sun et
al. (16 FLOPs/cycle × U[0.1, 2] GHz, 20 GHz server = 320 GFLOPS), where a round costs
AirSFL 1.5 + 84 s, digital SFL-V1 83.5 + 84 s, AirComp-FL 5.2 + 140 s.

**Why digital and analog uplink times differ by ~55×** (cut 2, one activation upload):
digital OFDMA gives each client S/N = 4 tones and needs 16 bits / c_D = 0.91 tone-uses per
value (c_D = 17.5 bit at the 52.8 dB per-tone SNR — already beyond practical modulation);
analog needs 0.5 tone-use per value (two values per complex symbol) and every client uses
all 120 tones (ZF separates the streams). Ratio = 30 (spatial reuse) × 1.82 (signalling) =
55 = 2Nq/c_D (Eq. 26; 109 with FP32). It is structural (independent of the bandwidth) and
should be reported as a gain over digital **OMA** SFL. The multi-user ZF baselines remove the
spatial-reuse factor (every digital client also uses all tones); what remains is the
signalling cost 2q/c_ZF = 32/11.75 = 2.7 (5.4 with FP32; c_ZF < c_D because ZF has
Nr − N + 1 = 35 instead of Nr·N of array/power gain per tone) — the gain of analog
transmission itself.

## Code

| File | What it is | Verified by |
|---|---|---|
| `flsim/airsfl/timing.py` | uplink time per phase, source MB; OFDMA and multi-user ZF digital rates; FP16 / FP32 payload | formula regression, limit cases, each phase counted once, ZF rate vs explicit ZF channels, N=1: ZF = OFDMA, q = 16 halves exactly the digital phases |
| `flsim/airsfl/compute.py` | FLOP profile per cut, computation time per phase | closed-form MACs, framework counter, prefix+suffix = full, N=1 limit, hand calculation |
| `flsim/airsfl/model.py` | ResNet-18: 3x3 stem, no max-pool, GroupNorm-32, cuts after stages 1-4 | roadmap d_c / d_s / d_a |
| `flsim/airsfl/radio.py` | dimensional ZF / AirComp physics | gates (packing, power, C^H H = I, noiseless AirComp, zero packets); MC MSE vs Eq. 10 / 23 |
| `flsim/airsfl/data.py` | 45k/5k split, IID / Dirichlet(α) (default α = 0.1), matched per-client streams, crop+flip | vectorized crop == reference |
| `flsim/airsfl/simulator.py` | training for all methods (the 5 trained ones + the ZF digital baselines); joint ZF error per packet; AirComp error; FP16 rounding of digital uploads; radio streams seeded by (seed, link, round, step); two engines | `python -m flsim.airsfl.checks` (incl. ZF baselines == parents' learning, bitwise; FP16 error bound, analog methods untouched by q) |

**Engines.** `vmap` (default) trains all clients in parallel (torch.func) with fast cuDNN
(1.53 s/round on an idle A30); `loop` trains clients one after another with relay back-prop
(1.77 s/round, 3.3 GB instead of 7.7 GB). Both equal float64 relay back-prop to 6e-15. A run
that hits GPU out-of-memory restarts automatically on the loop engine (same configuration,
same minibatches). `--deterministic` gives bitwise-reproducible reruns (slower).

## Environment (defaults in `run_airsfl.py`)

N=30 clients, Nr=64 at the M-server and the F-server, S=120 × 15 kHz (W=1.8 MHz), P_max=0.1 W,
N0=−167 dBm/Hz, ε=0.6, ρ=20 dB, block Rayleigh per 128-symbol packet, perfect CSI. Cut
after stage 2, τ=5, B=16, plain SGD, cosine decay to 1 % (fixed within a round), initial LR
calibrated on the error-free digital reference from {0.01, 0.03, 0.1, 0.3}, no augmentation
(`--augment` enables crop + flip; in the 50-epoch quick test it lowered accuracy from 76 % to
60 % and pushed the LR calibration to the grid edge), evaluation once per global-epoch
equivalent (`--evals-per-epoch`), FP16 digital payload (`--digital-q`), FP32 arithmetic
(TF32 disabled). SNR sweep ρ ∈ {−20, −10, 0, 10, 20} dB. The SNR figures
also show **AirSFL, error-free**: digital SFL-V1's trajectory (identical to noiseless
AirSFL) on AirSFL's time axis — the upper bound, like the "Error-free" curve of Sun et al.

**Run identity.** Every run's full resolved configuration (radio, training, schedule,
horizon, data/partition, augmentation, evaluation sizes, schema) is hashed; the hash is in
the file name and the configuration is saved as `<run>.json`. A finished run is reused only
when the configuration is identical — also across experiment folders (e.g. the FP16 runs of
`--exp fp16` are copied into `main/` instead of being trained again); an LR calibration is
refused under another setup. The FP16 payload enters the identity of the digital-payload
methods only, so AirSFL / AirComp-FL runs made before FP16 became the default are reused.

**Targets.** Prespecified validation accuracies (`plots.py --targets 0.75 0.8`); curves show
test accuracy, markers the checkpoint where validation reached the target; failures are
reported as k/n seeds; speed-ups are paired per seed.

## Experiments (`run_airsfl.py --exp ...`)

| exp | runs | figures |
|---|---|---|
| `bench` | seconds/round of both engines × cuDNN mode | — |
| `lr` | FedAvg (error-free, FP32), IID, LR grid, 20 epochs | — |
| `main` | 5 methods × {IID, Dir-α} at 20 dB (α = 0.1, `--dirichlet-alpha`) | fig1, 1b, 1c, 4b, 6, 8, table_main |
| `snr` | 3 analog methods × {−20, −10, 0, 10} dB | fig2, 2b, 2c, table_snr, table_budget |
| `nsweep` | 5 methods × N=20, 40 | fig7 |
| `cuts` | AirSFL + Sun × cuts 1, 3, 4 | fig9 |
| `tau` | 5 methods × τ = 1, 10, 20, 50 | fig9 |
| `nr` | AirSFL × Nr_M = 32, 40, 48 (Nr_F = 64) | fig5 |
| `pathloss` | 3 analog methods × path-gain spread 10, 20, 30, 40 dB at 20 dB (`--spreads`, `--pathloss-snrs`) | fig11, fig11b, table_pathloss |
| `fp16` | digital SFL-V1 + digital FedAvg at 20 dB, hybrid FDMA-AirComp at 20 dB and every `--snrs` value, FP16 payload (now the default of `main` / `snr`, which reuse these runs) | all figures |

**Unequal path gains** (the draft's robustness test, optional). Client n has long-term gain
λ_n = g_n λ_ref, the same to both servers (h ~ CN(0, λ_n I)). The offsets 10 log10 g_n are
equally spaced over [−Δ/2, +Δ/2] dB and assigned to clients by a seeded permutation, identical
for every method of a seed; ρ is the median client's SNR and the weakest client sits
Δ(1 − 1/N)/2 dB below it. Physics: client n's ZF error scales by 1/√g_n (joint correlation
kept), the AirComp combiner norm becomes dᵀG̃⁻¹d with d = 1/√g (the weakest clients set the
AirComp scaling), and every digital upload finishes with its slowest client (Eq. 12), so
the digital rates are the weakest client's. Analog airtime does not change. Per round at
20 dB (FP16): digital SFL-V1 83.5 → 102 s (Δ = 20 dB) → 132 s (40 dB); SFL-V1 (ZF) 4.2 → 5.7 →
9.1 s; AirSFL 1.53 s throughout. `run_airsfl.py --path-gain-spread Δ` applies the gains to every run
of a call (e.g. the SNR sweep); `plots.py --path-gain-spread Δ` loads such runs.

**Payload sets in `plots.py`.** `--digital-q 16` (default) draws every figure and table with
the FP16 runs of the digital-payload methods (from each experiment's folder and, for `main` /
`snr`, also from `fp16/`) into `figures/` and `tables/`; `--digital-q 32` draws the FP32 runs into
`figures_q32/` and `tables_q32/`; `--digital-q 16 32` draws both (`--snr` folders get the same
`_q32` tag). Both sets use the FP32 digital SFL-V1 as the error-free bound and the FP32 FedAvg
for the target rule, so their targets are identical. A point whose FP16 run is missing is
listed at the start and left out — it is never filled with FP32 learning. Source-equivalent MB
are counted at the drawn payload (the CSV column `mb_per_round` is the FP32 reference).

**Digital-transport sensitivity** (`fig12_digital_transport`, `table_digital_transport`):
every digital baseline (digital SFL-V1, hybrid with digital activations + AirComp, digital
FedAvg) under two access schemes (OFDMA, multi-user ZF) and two payload precisions (FP16,
FP32), each from its own runs; q = 16 halves every digital phase exactly.

**Main-point figures at another SNR**: `plots.py --snr 20 -20` also rebuilds fig1/1b/1c/1d,
fig3/3b/3c, fig4, fig6, fig8, fig10 and `table_main` at −20 dB from the CSVs (analog methods
from the SNR sweep at that SNR, digital ones retimed there) into `figures_snr-20/` and
`tables_snr-20/`; any swept SNR works. The N / cut / τ / antenna sweeps exist only at 20 dB.
Targets are the same for every SNR of one call; for an SNR where AirSFL's accuracy is lower,
call it alone with its own `--targets` (it writes only its own folder).

**Learning curves of every sweep point**: `fig2d_curves_all_snr_<axis>` (every SNR) and
`fig11b_curves_all_spreads_<axis>` (every spread) for all methods; `plots.py --sweep-curves
epochs uplink training` chooses the axes, `--curves-xscale linear` the time scale.

The two ZF digital baselines appear in every figure without runs of their own (derived from
`digital_sflv1` / `sun_fdma_aircomp` runs of the same experiment, e.g. the hybrid at every
SNR of `snr`). `plots.py --methods ...` selects the methods drawn, e.g.
`--methods airsfl aircomp_fl hybrid_zf_aircomp digital_sflv1_zf digital_sflv1`.

`fig1e` draws the learning curves on a **linear** uplink-time axis (the style of Sun et al.,
Fig. 2a/b) at 20 dB and at the lowest swept SNR: on the log axis of fig1 a per-round-time
ratio is a pure shift, on a linear axis it is a different slope. `--budget-mult 1 3` adds
fig2c columns at multiples of the default budget; `run_airsfl.py --early-evals` adds
checkpoints after rounds 1, 2, 4, 8, 16 so slow baselines have evaluated accuracies inside
small budgets (training unchanged; opt-in, so existing runs keep their identity).

Analytic figures (always drawn): fig3 (uplink phases vs N), fig3b (vs cut), fig3c (uplink +
computation per round and its composition), fig4a (bytes vs airtime). fig2c plots the test
accuracy reached within a fixed uplink / training-time budget vs SNR (`--budget-uplink`,
`--budget-training`; default: AirSFL's full-run time at 20 dB). fig10 / fig1d compare the
computation regimes (uplink only, edge GPU, IoT CPU) from the same `main` runs.

On the cluster: see the header of `slurm/run_airsfl.slurm`.
