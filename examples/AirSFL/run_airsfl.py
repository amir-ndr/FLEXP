"""
examples/AirSFL/run_airsfl.py: AirSFL vs baselines on CIFAR-10 / ResNet-18 (GroupNorm),
following the AirSFL WCNC draft (two servers, ideal downlinks) and its evaluation roadmap.

One matched environment; all methods share data, init, model, minibatches, augmentation,
plain SGD, the LR schedule, tau, full participation, weights a_n, bandwidth, per-client
power, the two server arrays, the channel statistics and the client computing
capabilities (paired per seed):

  AirSFL | Digital SFL-V1 | Hybrid FDMA-AirComp SFL (Sun-inspired) | AirComp-FL | Digital FedAvg

(The digital multi-user ZF baselines "Digital SFL-V1 (ZF)" and "Hybrid ZF-AirComp SFL" learn
exactly like Digital SFL-V1 / Hybrid FDMA-AirComp SFL; plots.py derives them from those runs
with the ZF uplink time -- they need no runs of their own.)

Default environment: N=30 clients; M-server (activations, ZF) and F-server (model
differences, AirComp) with 64 antennas each; S=120 subcarriers x 15 kHz (W=1.8 MHz, 4
tones per client in OFDMA); Pmax=0.1 W; N0=-167 dBm/Hz; eps_D=eps_U=eps_A=0.6; rho=20 dB;
cut after stage 2; tau=5; B=16; plain SGD with cosine decay to 1% (fixed within a round);
digital payload FP16 (--digital-q 16: the tensors digital SFL-V1, the hybrid and digital FedAvg
upload are really rounded to FP16 and timed with q = 16 bits; --digital-q 32 for FP32; AirSFL and
AirComp-FL upload nothing digitally but labels and are unaffected); evaluation once per
global-epoch equivalent; FP32 arithmetic (TF32 disabled); no augmentation (--augment
enables crop+flip); parallel "vmap" engine with fast cuDNN (--engine loop / --deterministic
to change; an out-of-memory run restarts on the loop engine). Computation: client
f_i ~ U[1, 2] TFLOPS, M-server f_s = 20 TFLOPS (flsim.airsfl.compute); plots.py also shows
the IoT-CPU setting of Sun et al. from the same runs. Every CSV row carries the uplink time,
the computation time and the end-to-end training time (uplink + computation), cumulative
and per phase, so any axis can be plotted later.

Experiments (--exp, any subset). Every run is identified by a hash of its complete
resolved configuration (saved as <run>.json next to <run>.csv): a finished run with the
same configuration is skipped (also when it finished under another experiment: it is then
copied, e.g. FP16 runs of --exp fp16 into main/), any changed setting produces a new run.
  bench   seconds per round of both training engines (x deterministic / fast cuDNN)
  lr      LR calibration on the error-free digital reference (FedAvg, IID, FP32); the chosen
          initial LR is then used by ALL methods (refused if calibrated under another setup)
  main    5 methods x {IID, Dirichlet-alpha (default 0.1, --dirichlet-alpha)} at 20 dB
  snr     analog methods x SNR (default -20, -10, 0, 10 dB; 20 dB from main). Digital
          learning is SNR-independent (ideal decoding): only its time is recomputed in plots
  nsweep  5 methods x N in {20, 40} (N=30 from main)
  cuts    AirSFL + Sun-inspired x cuts {1, 3, 4} (2 from main; digital SFL-V1 learning is
          cut-independent -> reused, times recomputed)
  tau     5 methods x tau in {1, 10, 20, 50} (5 from main)
  nr      AirSFL x M-server antennas Nr_M in {32, 40, 48}, F-server fixed at 64 (64 from main)
  pathloss  analog methods x unequal path gains, spread in {10, 20, 30, 40} dB around the
          reference (rho = median client; 0 dB from main); --pathloss-snrs adds SNR points.
          Digital learning is gain-independent: plots.py retimes the main runs at the weakest
          client's rate. --path-gain-spread X applies unequal gains to every run of a call.
  fp16    FP16 digital payload (really rounded, q = 16): digital SFL-V1 + digital FedAvg at 20 dB,
          hybrid FDMA-AirComp at 20 dB and every --snrs value (its learning depends on the SNR).
          FP16 is now the default payload of every experiment, so main / snr already contain these
          runs (finished fp16 runs are reused there); kept for older job scripts.
The FP32 runs of the digital-payload methods (--digital-q 32) are the exact error-free reference,
the reference of the target rule and the FP32 rows of plots.py's payload-sensitivity figure.

Usage:
  python examples/AirSFL/run_airsfl.py --exp bench lr
  python examples/AirSFL/run_airsfl.py --exp main --partition iid dirichlet --seeds 11 22 33 44 55
  python examples/AirSFL/run_airsfl.py --exp snr --snrs -20 -10 0 10
"""

import argparse
import dataclasses
import glob
import hashlib
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

from flsim.airsfl.compute import ComputeConfig, compute_time_breakdown, split_flops
from flsim.airsfl.data import (PARTITION_REDRAWS, ClientStream, load_cifar10_tensors, make_eval_tensors,
                               partition)
from flsim.airsfl.simulator import PACKET_REAL, SCHEMA, AirSFLSimulator, RunConfig
from flsim.airsfl.timing import METHODS as ALL_TIMED
from flsim.airsfl.timing import SPLIT_METHODS, RadioConfig, environment_summary, path_gain_offsets_db

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

METHODS = ["airsfl", "digital_sflv1", "sun_fdma_aircomp", "aircomp_fl", "digital_fedavg"]   # trained
ANALOG_METHODS = ["airsfl", "sun_fdma_aircomp", "aircomp_fl"]
# the methods that upload tensors digitally (activations and/or model differences): --digital-q
# applies to them; AirSFL / AirComp-FL upload only labels digitally and always keep q = 32 in their
# identity (their training and timing do not depend on q, so their runs are shared by both payloads)
FP16_METHODS = ["digital_sflv1", "sun_fdma_aircomp", "digital_fedavg"]

ENV = dict(N=30, Nr=64, Nr_F=64, S=120, df_hz=15e3, Pmax_w=0.1, N0_dbm_per_hz=-167.0,
           eps_D=0.6, eps_U=0.6, eps_A=0.6, rho_db=20.0)
COMPUTE = dict(client_tflops_lo=1.0, client_tflops_hi=2.0, server_tflops=20.0)
TRAIN = dict(stage=2, tau=5, batch_size=16, epochs=100, lr=0.1, lr_schedule="cosine", lr_min_frac=0.01,
             dirichlet_alpha=0.1, min_per_client=16, seed=11, evals_per_epoch=1.0, eval_rounds=(),
             augment=False, engine="vmap", vmap_chunk=None)
EARLY_EVALS = (1, 2, 4, 8, 16)         # --early-evals: extra checkpoints before the first per-epoch one
# augment=False: with plain SGD and a 50-100 epoch budget the model is still in the under-fitting
# regime, where crop+flip slows fitting (quick test: 60% vs 76%) and pushed the LR calibration to the
# grid edge (0.01). engine="vmap" + fast cuDNN was the fastest setting on an idle A30; a run that hits
# GPU out-of-memory (shared GPU) is restarted automatically with the loop engine.
LR_GRID = [0.01, 0.03, 0.1, 0.3]       # roadmap {0.01, 0.03, 0.1} + 0.3 to detect a grid-edge optimum
LR_CAL_EPOCHS = 20
SNR_SWEEP = [-20.0, -10.0, 0.0, 10.0]  # + 20 dB from main
N_SWEEP = [20, 40]
CUT_SWEEP = [1, 3, 4]
TAU_SWEEP = [1, 10, 20, 50]            # stage-2 per-round crossover with AirComp-FL near tau = d_s/d_a ~ 20
NR_SWEEP = [32, 40, 48]                # M-server antennas; F-server fixed
SPREAD_SWEEP = [10.0, 20.0, 30.0, 40.0]   # path-gain spreads (dB) for --exp pathloss (0 = main)
PATH_GAIN = {"spread_db": 0.0}         # --path-gain-spread: unequal path gains for every experiment
DIGITAL_Q = {"q_bits": 16}             # --digital-q: payload bits of FP16_METHODS (16 FP16 default, 32 FP32)
TARGETS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]   # reached_XX columns (validation)


class Env:
    """Loads CIFAR once; builds matched client streams per (partition, N, seed)."""

    def __init__(self, device, augment):
        self.device, self.augment = device, augment
        self.data = load_cifar10_tensors()
        d = self.data
        self.n_train = int(d["x_train"].shape[0])
        self.eval_sets = {
            "val": (make_eval_tensors(d["x_val"], d["mean"], d["std"], device), d["y_val"].to(device)),
            "test": (make_eval_tensors(d["x_test"], d["mean"], d["std"], device), d["y_test"].to(device)),
        }
        self._parts = {}

    def parts(self, scheme, N, seed):
        key = (scheme, N, seed)
        if key not in self._parts:
            self._parts[key] = partition(self.data["y_train"].numpy(), N, scheme,
                                         alpha=TRAIN["dirichlet_alpha"],
                                         min_per_client=TRAIN["min_per_client"], seed=seed)
        return self._parts[key]

    def streams(self, scheme, N, B, seed):
        d = self.data
        return [ClientStream(d["x_train"][p], d["y_train"][p], d["mean"], d["std"], B,
                             seed * 1000 + k, self.device, augment=self.augment)
                for k, p in enumerate(self.parts(scheme, N, seed))]

    def weights(self, scheme, N, seed):
        w = np.array([len(p) for p in self.parts(scheme, N, seed)], dtype=np.float64)
        return w / w.sum()


def rounds_for(epochs, N, tau, B, n_train):
    return int(math.ceil(epochs * n_train / (N * tau * B)))


def run_manifest(env, method, scheme, radio, cfg, epochs):
    """Everything that determines a run's learning trajectory. Implementation details that
    only change float rounding (engine, cuDNN mode) and the computation-capability model
    (which only rescales the time columns; plots.py can recompute it) are recorded in the
    CSV but excluded from the identity."""
    d = env.data
    part_sizes = [len(p) for p in env.parts(scheme, radio.N, cfg.seed)]
    return {
        "schema": SCHEMA, "method": method, "partition": scheme,
        "dirichlet_alpha": TRAIN["dirichlet_alpha"] if scheme == "dirichlet" else None,
        "min_per_client": TRAIN["min_per_client"],
        "partition_sizes_sha": hashlib.sha256(json.dumps(part_sizes).encode()).hexdigest()[:12],
        # path gains / FP16 payloads enter the identity only when set (equal-gain FP32 runs keep their hash)
        "radio": {k: v for k, v in dataclasses.asdict(radio).items()
                  if not ((k == "gains_db" and v is None) or (k == "q_bits" and v == 32))},
        # extra evaluation rounds enter the identity only when used (runs without them keep their hash)
        "train": {k: v for k, v in dataclasses.asdict(cfg).items()
                  if k not in ("engine", "vmap_chunk") and not (k == "eval_rounds" and not v)},
        "epochs_budget": float(epochs),
        "data": {"augment": env.augment, "n_train": env.n_train, "split_seed": 2026,
                 "val_n": int(env.eval_sets["val"][0].shape[0]), "test_n": int(env.eval_sets["test"][0].shape[0]),
                 "mean": [round(float(v), 6) for v in d["mean"]]},
        "packet_real": PACKET_REAL,
    }


def run_one(env: Env, exp: str, method: str, scheme: str, lr: float, epochs: float, seed: int,
            rho=None, N=None, Nr=None, Nr_F=None, stage=None, tau=None, noiseless=False, spread=None,
            q_bits=None) -> str:
    rho = ENV["rho_db"] if rho is None else float(rho)
    N = ENV["N"] if N is None else int(N)
    Nr = ENV["Nr"] if Nr is None else int(Nr)
    Nr_F = ENV["Nr_F"] if Nr_F is None else int(Nr_F)
    stage = TRAIN["stage"] if stage is None else stage
    tau = TRAIN["tau"] if tau is None else int(tau)
    spread = PATH_GAIN["spread_db"] if spread is None else float(spread)
    if q_bits is None:
        q_bits = DIGITAL_Q["q_bits"] if method in FP16_METHODS else 32
    q_bits = int(q_bits)
    B = TRAIN["batch_size"]
    R = rounds_for(epochs, N, tau, B, env.n_train)
    radio = RadioConfig(**{**ENV, "N": N, "Nr": Nr, "Nr_F": Nr_F, "rho_db": rho, "batch_size": B,
                           "gains_db": path_gain_offsets_db(N, spread, seed), "q_bits": q_bits})
    cfg = RunConfig(method=method, stage=stage, tau=tau, batch_size=B, lr=lr,
                    lr_schedule=TRAIN["lr_schedule"], lr_min_frac=TRAIN["lr_min_frac"], rounds=R,
                    evals_per_epoch=TRAIN["evals_per_epoch"], eval_rounds=tuple(TRAIN["eval_rounds"]),
                    noiseless=noiseless, seed=seed,
                    engine=TRAIN["engine"], vmap_chunk=TRAIN["vmap_chunk"])
    manifest = run_manifest(env, method, scheme, radio, cfg, epochs)
    rid = hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode()).hexdigest()[:10]
    name = (f"{method}_{scheme}_N{N}_Nr{Nr}-{Nr_F}_cut{stage}_tau{tau}_rho{rho:g}"
            + (f"_gs{spread:g}" if spread else "") + (f"_q{q_bits}" if q_bits != 32 else "")
            + f"_lr{lr:g}_s{seed}_{rid}")
    out_dir = os.path.join(RESULTS, exp)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, name + ".csv")
    complete = lambda done: len(done) and (int(done["round"].max()) >= R or done["diverged"].astype(bool).any())
    if os.path.exists(csv_path):
        if complete(pd.read_csv(csv_path)):
            print(f"[skip] {name} (same configuration, complete)")
            return csv_path
    else:       # the identical run (same name = same identity hash) finished under another experiment
        for other in sorted(glob.glob(os.path.join(RESULTS, "*", name + ".csv"))):
            done = pd.read_csv(other)
            if complete(done):
                done.assign(exp=exp).to_csv(csv_path, index=False)
                for ext in (".json", ".log"):
                    if os.path.exists(other[:-4] + ext):
                        shutil.copyfile(other[:-4] + ext, os.path.join(out_dir, name + ext))
                print(f"[reuse] {name}: same configuration finished in "
                      f"{os.path.basename(os.path.dirname(other))}/ -> copied to {exp}/")
                return csv_path
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump({"run_id": rid, **manifest}, f, indent=1, sort_keys=True, default=str)
    with open(os.path.join(out_dir, name + ".log"), "w") as logf:
        def log(msg):
            print(msg, flush=True)
            logf.write(msg + "\n")
            logf.flush()

        def simulate(c):   # fresh streams every attempt: identical minibatches in a retry
            return AirSFLSimulator(c, radio, env.streams(scheme, N, B, seed), env.weights(scheme, N, seed),
                                   env.eval_sets, env.device, env.n_train, log=log,
                                   compute=ComputeConfig(**COMPUTE)).run()
        try:
            hist = simulate(cfg)
        except torch.cuda.OutOfMemoryError:
            if cfg.engine == "loop":
                raise
            torch.cuda.empty_cache()
            log("[oom] vmap engine ran out of GPU memory (shared GPU?) -> restarting this run with the "
                "loop engine (same math, ~1/3 of the memory)")
            hist = simulate(dataclasses.replace(cfg, engine="loop"))
    df = pd.DataFrame(hist)
    df["run_id"] = rid
    df["partition"] = scheme
    df["exp"] = exp
    df["path_gain_spread_db"] = spread          # nominal spread (0 = equal gains); gains in path_gains_db
    df["augment"] = env.augment
    df["epochs_budget"] = float(epochs)
    df["dirichlet_alpha"] = TRAIN["dirichlet_alpha"] if scheme == "dirichlet" else np.nan
    df["partition_redraws"] = PARTITION_REDRAWS.get((scheme, N, TRAIN["dirichlet_alpha"], seed), 0)
    best = df["val_acc"].cummax()
    for A in TARGETS:
        df[f"reached_{int(round(100 * A))}"] = best >= A
    df.to_csv(csv_path, index=False)
    print(f"[done] {name} -> {csv_path}")
    return csv_path


def _lr_setup(env):
    """The setup an LR calibration is valid for."""
    return {"augment": env.augment, "lr_schedule": TRAIN["lr_schedule"], "lr_min_frac": TRAIN["lr_min_frac"],
            "schema": SCHEMA, "N": ENV["N"], "B": TRAIN["batch_size"], "tau": TRAIN["tau"]}


def chosen_lr(env) -> float:
    """Initial LR picked by `--exp lr`; refuses a calibration made under another setup."""
    path = os.path.join(RESULTS, "lr", "chosen_lr.json")
    if not os.path.exists(path):
        raise SystemExit(f"[lr] {path} not found: run --exp lr first (or pass --lr)")
    with open(path) as f:
        cal = json.load(f)
    mism = {k: (cal.get("setup", {}).get(k), v) for k, v in _lr_setup(env).items()
            if cal.get("setup", {}).get(k) != v}
    if mism:
        raise SystemExit(f"[lr] {path} was calibrated under a different setup (stored, current): {mism}. "
                         f"Rerun --exp lr in this results folder (or pass --lr).")
    return float(cal["lr"])


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------

def exp_bench(env, args):
    """Seconds per round of both engines x cuDNN mode on this device (AirSFL, default
    environment), and the parameter difference after the same rounds."""
    rounds = 3
    out = {}
    vecs = {}
    small_eval = {k: (x[:200], y[:200]) for k, (x, y) in env.eval_sets.items()}   # keep eval out of timing

    def make(engine, R):
        cfg = RunConfig(method="airsfl", stage=TRAIN["stage"], tau=TRAIN["tau"], batch_size=TRAIN["batch_size"],
                        lr=0.05, rounds=R, evals_per_epoch=1e-9, seed=11, engine=engine,
                        vmap_chunk=TRAIN["vmap_chunk"])
        return AirSFLSimulator(cfg, RadioConfig(**ENV), env.streams("iid", ENV["N"], TRAIN["batch_size"], 11),
                               env.weights("iid", ENV["N"], 11), small_eval, env.device, env.n_train,
                               log=lambda *a: None)
    if env.device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        out["gpu_free_GB_at_start"] = free / 2 ** 30
        if free < 0.9 * total:
            print(f"[bench] WARNING: only {free / 2**30:.1f} of {total / 2**30:.1f} GB free -- another process "
                  f"shares this GPU, so timings are pessimistic and vmap may run out of memory")
    saved = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    for fast in (False, True):                                 # deterministic vs fastest cuDNN algorithms
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = fast, not fast
        for engine in ("loop", "vmap"):
            key = engine + ("_fast_cudnn" if fast else "")
            try:
                make(engine, 1).run()                          # warm-up (CUDA/cuDNN initialization)
                sim = make(engine, rounds)
                if env.device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                sim.run()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                out[key] = {"s_per_round": float("inf"), "error": "CUDA out of memory"}
                print(f"[bench] {key:17s}: CUDA out of memory (skipped)")
                continue
            if env.device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.time() - t0) / rounds
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if env.device.type == "cuda" else float("nan")
            out[key] = {"s_per_round": dt, "peak_mem_GB": mem,
                        "est_hours_per_100_epochs": dt * rounds_for(100, ENV["N"], TRAIN["tau"],
                                                                    TRAIN["batch_size"], env.n_train) / 3600}
            vecs[key] = sim.global_vector()
            print(f"[bench] {key:17s}: {dt:.2f} s/round, peak GPU memory {mem:.2f} GB, "
                  f"~{out[key]['est_hours_per_100_epochs']:.2f} h per 100-epoch run")
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = saved
    if "vmap" in vecs and "loop" in vecs:
        diff = vecs["vmap"] - vecs["loop"]
        out["max_param_diff"] = float(diff.abs().max())
        out["median_param_diff"] = float(diff.abs().median())
    best = min((k for k in out if isinstance(out[k], dict)), key=lambda k: out[k]["s_per_round"])
    out["fastest"] = best
    print(f"[bench] fastest setting: {best}")
    os.makedirs(os.path.join(RESULTS, "bench"), exist_ok=True)
    with open(os.path.join(RESULTS, "bench", f"bench_{env.device.type}.json"), "w") as f:
        json.dump(out, f, indent=2)


def exp_lr(env, args):
    rows = []
    for lr in LR_GRID:   # error-free (FP32) reference: path gains change only its time, never its learning
        p = run_one(env, "lr", "digital_fedavg", "iid", lr, args.lr_epochs, seed=args.seeds[0], spread=0.0,
                    q_bits=32)
        df = pd.read_csv(p)
        diverged = bool(df["diverged"].astype(bool).any())
        rows.append({"lr": lr, "final_val_acc": float("nan") if diverged else float(df["val_acc"].iloc[-1]),
                     "best_val_acc": float(df["val_acc"].max()), "diverged": diverged})
    ok = [r for r in rows if not math.isnan(r["final_val_acc"])]
    best = max(ok, key=lambda r: r["final_val_acc"])
    with open(os.path.join(RESULTS, "lr", "chosen_lr.json"), "w") as f:
        json.dump({"lr": best["lr"], "grid": rows, "epochs": args.lr_epochs, "setup": _lr_setup(env),
                   "augment": env.augment, "schedule": TRAIN["lr_schedule"]}, f, indent=2)
    print(f"[lr] grid {rows} -> chosen lr = {best['lr']}")
    if best["lr"] in (min(LR_GRID), max(LR_GRID)):
        print(f"[lr] WARNING: chosen lr {best['lr']} is at the edge of the grid {LR_GRID}")


def _lr(env, args):
    return args.lr if args.lr is not None else chosen_lr(env)


def exp_main(env, args):
    lr = _lr(env, args)
    for seed in args.seeds:
        for scheme in args.partition:
            for method in args.methods:
                run_one(env, "main", method, scheme, lr, args.epochs, seed)


def exp_snr(env, args):
    lr = _lr(env, args)
    for seed in args.seeds:
        for scheme in args.partition:
            for rho in args.snrs:
                for method in [m for m in args.methods if m in ANALOG_METHODS]:
                    run_one(env, "snr", method, scheme, lr, args.epochs, seed, rho=rho)


def exp_nsweep(env, args):
    lr = _lr(env, args)
    for seed in args.seeds:
        for scheme in args.partition:
            for N in args.ns:
                for method in args.methods:
                    run_one(env, "nsweep", method, scheme, lr, args.epochs, seed, N=N)


def exp_cuts(env, args):
    lr = _lr(env, args)
    for seed in args.seeds:
        for scheme in args.partition:
            for stage in args.cuts:
                for method in [m for m in args.methods if m in ("airsfl", "sun_fdma_aircomp")]:
                    run_one(env, "cuts", method, scheme, lr, args.epochs, seed, stage=stage)


def exp_tau(env, args):
    lr = _lr(env, args)
    for seed in args.seeds:
        for scheme in args.partition:
            for tau in args.taus:
                for method in args.methods:
                    run_one(env, "tau", method, scheme, lr, args.epochs, seed, tau=tau)


def exp_nr(env, args):
    lr = _lr(env, args)
    for seed in args.seeds:
        for scheme in args.partition:
            for Nr in args.nrs:
                run_one(env, "nr", "airsfl", scheme, lr, args.epochs, seed, Nr=Nr, Nr_F=args.nr_f)


def exp_pathloss(env, args):
    """Unequal path gains (draft's robustness test): analog methods x path-gain spread at the
    nominal SNR, or at every --snrs value when --pathloss-snrs is given. Digital learning does
    not depend on the gains (ideal decoding): plots.py recomputes its time from the main runs
    at the weakest client's rate."""
    lr = _lr(env, args)
    rhos = args.pathloss_snrs or [ENV["rho_db"]]
    for seed in args.seeds:
        for scheme in args.partition:
            for rho in rhos:
                for spread in args.spreads:
                    for method in [m for m in args.methods if m in ANALOG_METHODS]:
                        run_one(env, "pathloss", method, scheme, lr, args.epochs, seed, rho=rho, spread=spread)


def exp_fp16(env, args):
    """FP16 as the digital payload (the draft's 16-bit transport check: "must actually round
    transmitted tensors"): the methods that upload tensors digitally, trained with FP16-rounded
    uploads and timed with q = 16 -- digital SFL-V1 and digital FedAvg at the nominal SNR, and
    the hybrid (digital activations + AirComp) at the nominal SNR and at every --snrs value,
    since its learning depends on the SNR. Their ZF variants are derived in plots.py; AirSFL
    and AirComp-FL upload nothing digitally except labels and are unaffected. (FP16 is now the
    default payload of main / snr, which reuse these runs.)"""
    lr = _lr(env, args)
    for seed in args.seeds:
        for scheme in args.partition:
            for method in [m for m in args.methods if m in FP16_METHODS]:
                rhos = [ENV["rho_db"]] + ([r for r in args.snrs if r != ENV["rho_db"]]
                                         if method == "sun_fdma_aircomp" else [])
                for rho in rhos:
                    run_one(env, "fp16", method, scheme, lr, args.epochs, seed, rho=rho, q_bits=16)


EXPERIMENTS = {"bench": exp_bench, "lr": exp_lr, "main": exp_main, "snr": exp_snr, "nsweep": exp_nsweep,
               "cuts": exp_cuts, "tau": exp_tau, "nr": exp_nr, "pathloss": exp_pathloss, "fp16": exp_fp16}


def print_budget():
    """Per-round uplink + computation budget of the default environment."""
    radio = RadioConfig(**ENV, gains_db=path_gain_offsets_db(ENV["N"], PATH_GAIN["spread_db"], TRAIN["seed"]),
                        q_bits=DIGITAL_Q["q_bits"])
    print(f"  digital payload: FP{radio.q_bits} (q = {radio.q_bits} bits per value) for {', '.join(FP16_METHODS)}")
    environment_summary(radio, stage=TRAIN["stage"], tau=TRAIN["tau"])
    lo, hi, fs = COMPUTE["client_tflops_lo"], COMPUTE["client_tflops_hi"], COMPUTE["server_tflops"]
    print(f"  computation: client f_i ~ U[{lo}, {hi}] TFLOPS (slowest client paces each step), "
          f"M-server {fs} TFLOPS shared by the {radio.N} suffix copies")
    for m in ALL_TIMED:
        fl = split_flops(TRAIN["stage"] if m in SPLIT_METHODS else None)
        c = compute_time_breakdown(m, fl, radio.N, TRAIN["batch_size"], TRAIN["tau"], lo * 1e12, fs * 1e12)
        print(f"  {m:17s} compute {c['total']:.3f} s/round at f_min={lo} TFLOPS (client FP {c['client_fp']:.3f}, "
              f"server FP {c['server_fp']:.3f}, server BP {c['server_bp']:.3f}, client BP {c['client_bp']:.3f})")


def main():
    p = argparse.ArgumentParser(description="AirSFL vs baselines (CIFAR-10, ResNet-18 GN)")
    p.add_argument("--exp", nargs="+", default=["main"], choices=list(EXPERIMENTS))
    p.add_argument("--partition", nargs="+", default=["iid", "dirichlet"], choices=["iid", "dirichlet"])
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--epochs", type=float, default=TRAIN["epochs"], help="global-epoch equivalents")
    p.add_argument("--lr-epochs", type=float, default=LR_CAL_EPOCHS)
    p.add_argument("--lr", type=float, default=None, help="override the calibrated initial LR")
    p.add_argument("--seeds", nargs="+", type=int, default=[TRAIN["seed"]],
                   help="paired seeds (roadmap: 11 22 33 44 55): partitions, init, data order, channels, f_i")
    p.add_argument("--snrs", nargs="+", type=float, default=SNR_SWEEP)
    p.add_argument("--ns", nargs="+", type=int, default=N_SWEEP)
    p.add_argument("--cuts", nargs="+", type=int, default=CUT_SWEEP)
    p.add_argument("--taus", nargs="+", type=int, default=TAU_SWEEP)
    p.add_argument("--nrs", nargs="+", type=int, default=NR_SWEEP, help="M-server antennas for --exp nr")
    p.add_argument("--nr-f", type=int, default=ENV["Nr_F"], help="F-server antennas for --exp nr (fixed)")
    p.add_argument("--spreads", nargs="+", type=float, default=SPREAD_SWEEP,
                   help="path-gain spreads (dB) for --exp pathloss")
    p.add_argument("--pathloss-snrs", nargs="+", type=float, default=None,
                   help="SNRs for --exp pathloss (default: the nominal 20 dB only)")
    p.add_argument("--digital-q", type=int, default=DIGITAL_Q["q_bits"], choices=[16, 32],
                   help="digital payload bits of the methods that upload tensors digitally (digital SFL-V1, hybrid, "
                        "digital FedAvg): 16 = FP16 (default; the uploaded tensors are really rounded), 32 = FP32. "
                        "AirSFL / AirComp-FL are unaffected (same runs for both); --exp fp16 uses 16, --exp lr 32")
    p.add_argument("--path-gain-spread", type=float, default=0.0,
                   help="unequal path gains for EVERY run of this call: client gains spread uniformly (dB) over "
                        "this range around the reference (rho = median client); 0 = equal gains")
    p.add_argument("--augment", action="store_true", help="random crop + flip (default off)")
    p.add_argument("--no-augment", action="store_true", help="(default; kept for old commands)")
    p.add_argument("--engine", default=TRAIN["engine"], choices=["vmap", "loop"])
    p.add_argument("--vmap-chunk", type=int, default=None, help="clients per vmap chunk (GPU memory)")
    p.add_argument("--deterministic", action="store_true",
                   help="deterministic cuDNN algorithms (bitwise-reproducible reruns, slower)")
    p.add_argument("--fast-cudnn", action="store_true", help="(default; kept for old commands)")
    p.add_argument("--dirichlet-alpha", type=float, default=TRAIN["dirichlet_alpha"],
                   help="label-Dirichlet concentration of the non-IID partition (default 0.1; the draft's 0.5 is "
                        "milder). IID runs do not depend on it")
    p.add_argument("--evals-per-epoch", type=float, default=TRAIN["evals_per_epoch"],
                   help="evaluations per global-epoch equivalent (default 1 = every ~19 rounds at N=30; 4 = every "
                        "~5 rounds). One evaluation (5k val + 10k test images) costs about 1.5-2 training rounds")
    p.add_argument("--early-evals", action="store_true",
                   help=f"also evaluate after rounds {EARLY_EVALS} (real accuracies of slow methods inside small "
                        f"time budgets; training is unchanged, the runs get their own identity)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results", default=None, help="results root (default examples/AirSFL/results)")
    p.add_argument("--eval-subset", type=int, default=0,
                   help="evaluate on the first K val/test images only (smoke tests); 0 = full sets")
    args = p.parse_args()

    global RESULTS
    if args.results:
        RESULTS = os.path.abspath(args.results)
    TRAIN["engine"], TRAIN["vmap_chunk"] = args.engine, args.vmap_chunk
    TRAIN["eval_rounds"] = EARLY_EVALS if args.early_evals else ()
    TRAIN["evals_per_epoch"] = float(args.evals_per_epoch)
    TRAIN["dirichlet_alpha"] = float(args.dirichlet_alpha)
    PATH_GAIN["spread_db"] = float(args.path_gain_spread)
    DIGITAL_Q["q_bits"] = int(args.digital_q)
    # FP32 everywhere (roadmap): no TF32 tensor-core shortcuts on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = not args.deterministic
    torch.backends.cudnn.deterministic = bool(args.deterministic)
    env = Env(torch.device(args.device), augment=bool(args.augment) and not args.no_augment)
    if args.eval_subset > 0:
        env.eval_sets = {k: (x[:args.eval_subset], y[:args.eval_subset]) for k, (x, y) in env.eval_sets.items()}
    print(f"[env] device={args.device} | results={RESULTS} | {ENV} | {COMPUTE} | {TRAIN} | augment={env.augment} "
          f"| epochs={args.epochs} | seeds={args.seeds} | n_train={env.n_train}")
    print_budget()
    for e in args.exp:
        EXPERIMENTS[e](env, args)


if __name__ == "__main__":
    sys.exit(main())
