"""
examples/AirSFL/run_airsfl.py: AirSFL vs baselines on CIFAR-10 / ResNet-18 (GroupNorm),
following the AirSFL WCNC draft (two servers, ideal downlinks) and its evaluation roadmap.

One matched environment; all methods share data, init, model, minibatches, plain SGD,
the LR schedule, tau, full participation, weights a_n, bandwidth, per-client power,
the two server arrays and the channel statistics (paired seeded channel traces):

  AirSFL | Digital SFL-V1 | FDMA-AirComp SFL (Sun-inspired) | AirComp-FL | Digital FedAvg

Default environment (roadmap Sec. 2-3): N=30 clients, M-server and F-server with 64
antennas each, S=120 subcarriers x 15 kHz (W=1.8 MHz, 4 tones per client in OFDMA),
Pmax=0.1 W, N0=-167 dBm/Hz, eps_D=eps_U=eps_A=0.6, rho=20 dB, cut after stage 2,
tau=5, B=16, plain SGD with cosine decay to 1% (fixed within a round), 100 global-epoch
equivalents, evaluation once per global-epoch equivalent, FP32 (TF32 disabled).
Augmentation is off by default (--augment turns on crop+flip).

Experiments (--exp, any subset; every run is resumable -- a finished CSV is skipped;
every CSV keeps the full per-checkpoint record, so any figure can be redrawn later):
  bench   time the two training engines on this device (s/round) and compare them
  lr      LR calibration on the error-free digital reference (FedAvg, IID); the chosen
          initial LR is then used by ALL methods (transport-isolation experiment)
  main    5 methods x {IID, Dirichlet-0.5} at 20 dB          (roadmap B, C; fig 1, 3, 4, 6)
  snr     analog methods x SNR (default 0, 10, 30 dB; 20 dB from main)   (roadmap D; fig 2)
          digital learning is SNR-independent (ideal decoding): only its time changes,
          recomputed analytically in plots.py -- no re-run
  nsweep  5 methods x N in {20, 40} (N=30 from main)            (roadmap A + time to target)
  cuts    AirSFL + Sun-inspired x cuts {1, 3, 4} (2 from main)  (roadmap E; digital SFL-V1
          learning is cut-independent -> reused, time recomputed)
  tau     5 methods x tau in {1, 10} (5 from main)              (roadmap E)
  nr      AirSFL x Nr_M = Nr_F in {32, 48, 128} (64 from main)  (roadmap F, antenna margin)

Outputs: <results>/<exp>/<run>.csv (+ .log); then python examples/AirSFL/plots.py.

Usage:
  python examples/AirSFL/run_airsfl.py --exp bench
  python examples/AirSFL/run_airsfl.py --exp lr
  python examples/AirSFL/run_airsfl.py --exp main --partition iid dirichlet --seeds 11 22 33 44 55
  python examples/AirSFL/run_airsfl.py --exp snr --snrs -20 -10 0 10 30
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from flsim.airsfl.data import (PARTITION_REDRAWS, ClientStream, load_cifar10_tensors, make_eval_tensors,
                               partition)
from flsim.airsfl.simulator import SCHEMA, AirSFLSimulator, RunConfig
from flsim.airsfl.timing import RadioConfig, environment_summary

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

METHODS = ["airsfl", "digital_sflv1", "sun_fdma_aircomp", "aircomp_fl", "digital_fedavg"]
ANALOG_METHODS = ["airsfl", "sun_fdma_aircomp", "aircomp_fl"]

ENV = dict(N=30, Nr=64, Nr_F=64, S=120, df_hz=15e3, Pmax_w=0.1, N0_dbm_per_hz=-167.0,
           eps_D=0.6, eps_U=0.6, eps_A=0.6, rho_db=20.0)
TRAIN = dict(stage=2, tau=5, batch_size=16, epochs=100, lr=0.1, lr_schedule="cosine", lr_min_frac=0.01,
             dirichlet_alpha=0.5, min_per_client=16, seed=11, evals_per_epoch=1.0, augment=False,
             engine="loop", vmap_chunk=None)   # quick-test bench: loop 3.6 s/rnd vs vmap 4.3 s/rnd on the GPU
LR_GRID = [0.01, 0.03, 0.1, 0.3]       # roadmap {0.01, 0.03, 0.1} + 0.3 to detect a grid-edge optimum
LR_CAL_EPOCHS = 20
SNR_SWEEP = [0.0, 10.0, 30.0]          # 20 dB from main; add negatives with --snrs to probe low SNR
N_SWEEP = [20, 40]
CUT_SWEEP = [1, 3, 4]
TAU_SWEEP = [1, 10]
NR_SWEEP = [32, 48, 128]
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


def run_name(method, scheme, N, Nr, stage, tau, rho, lr, seed):
    return f"{method}_{scheme}_N{N}_Nr{Nr}_cut{stage}_tau{tau}_rho{rho:g}_lr{lr:g}_s{seed}"


def run_one(env: Env, exp: str, method: str, scheme: str, lr: float, epochs: float, seed: int,
            rho=None, N=None, Nr=None, stage=None, tau=None, noiseless=False) -> str:
    rho = ENV["rho_db"] if rho is None else float(rho)
    N = ENV["N"] if N is None else int(N)
    Nr = ENV["Nr"] if Nr is None else int(Nr)
    stage = TRAIN["stage"] if stage is None else stage
    tau = TRAIN["tau"] if tau is None else int(tau)
    B = TRAIN["batch_size"]
    name = run_name(method, scheme, N, Nr, stage, tau, rho, lr, seed)
    out_dir = os.path.join(RESULTS, exp)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, name + ".csv")
    R = rounds_for(epochs, N, tau, B, env.n_train)
    if os.path.exists(csv_path):
        done = pd.read_csv(csv_path)
        if len(done) and "schema" in done and int(done["schema"].iloc[0]) == SCHEMA and \
                (int(done["round"].max()) >= R or done["diverged"].astype(bool).any()):
            print(f"[skip] {name} (complete)")
            return csv_path
    radio = RadioConfig(**{**ENV, "N": N, "Nr": Nr, "Nr_F": Nr, "rho_db": rho})
    cfg = RunConfig(method=method, stage=stage, tau=tau, batch_size=B, lr=lr,
                    lr_schedule=TRAIN["lr_schedule"], lr_min_frac=TRAIN["lr_min_frac"], rounds=R,
                    evals_per_epoch=TRAIN["evals_per_epoch"], noiseless=noiseless, seed=seed,
                    engine=TRAIN["engine"], vmap_chunk=TRAIN["vmap_chunk"])
    with open(os.path.join(out_dir, name + ".log"), "w") as logf:
        def log(msg):
            print(msg, flush=True)
            logf.write(msg + "\n")
            logf.flush()
        sim = AirSFLSimulator(cfg, radio, env.streams(scheme, N, B, seed), env.weights(scheme, N, seed),
                              env.eval_sets, env.device, env.n_train, log=log)
        hist = sim.run()
    df = pd.DataFrame(hist)
    df["partition"] = scheme
    df["exp"] = exp
    df["augment"] = env.augment
    df["dirichlet_alpha"] = TRAIN["dirichlet_alpha"] if scheme == "dirichlet" else np.nan
    df["partition_redraws"] = PARTITION_REDRAWS.get((scheme, N, TRAIN["dirichlet_alpha"], seed), 0)
    best = df["val_acc"].cummax()
    for A in TARGETS:
        df[f"reached_{int(round(100 * A))}"] = best >= A
    df.to_csv(csv_path, index=False)
    print(f"[done] {name} -> {csv_path}")
    return csv_path


def chosen_lr() -> float:
    """Initial LR picked by `--exp lr`; falls back to TRAIN['lr'] (with a warning)."""
    path = os.path.join(RESULTS, "lr", "chosen_lr.json")
    if os.path.exists(path):
        with open(path) as f:
            return float(json.load(f)["lr"])
    print(f"[lr] WARNING: {path} not found -> using the default lr={TRAIN['lr']}")
    return TRAIN["lr"]


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------

def exp_bench(env, args):
    """Seconds per round of both engines on this device (AirSFL, default environment),
    and the parameter difference after the same rounds (float32 kernels differ slightly)."""
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
    saved = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    for fast in (False, True):                                 # deterministic vs fastest cuDNN algorithms
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = fast, not fast
        for engine in ("loop", "vmap"):
            key = engine + ("_fast_cudnn" if fast else "")
            make(engine, 1).run()                              # warm-up (CUDA/cuDNN initialization)
            sim = make(engine, rounds)
            if env.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            sim.run()
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
    diff = vecs["vmap"] - vecs["loop"]
    out["max_param_diff"] = float(diff.abs().max())
    out["median_param_diff"] = float(diff.abs().median())
    best = min((k for k in out if isinstance(out[k], dict)), key=lambda k: out[k]["s_per_round"])
    out["fastest"] = best
    print(f"[bench] after {rounds} rounds: max|vmap - loop| = {out['max_param_diff']:.2e}, "
          f"median {out['median_param_diff']:.1e}; fastest setting: {best}")
    os.makedirs(os.path.join(RESULTS, "bench"), exist_ok=True)
    with open(os.path.join(RESULTS, "bench", f"bench_{env.device.type}.json"), "w") as f:
        json.dump(out, f, indent=2)


def exp_lr(env, args):
    rows = []
    for lr in LR_GRID:
        p = run_one(env, "lr", "digital_fedavg", "iid", lr, args.lr_epochs, seed=args.seeds[0])
        df = pd.read_csv(p)
        diverged = bool(df["diverged"].astype(bool).any())
        rows.append({"lr": lr, "final_val_acc": float("nan") if diverged else float(df["val_acc"].iloc[-1]),
                     "best_val_acc": float(df["val_acc"].max()), "diverged": diverged})
    ok = [r for r in rows if not math.isnan(r["final_val_acc"])]
    best = max(ok, key=lambda r: r["final_val_acc"])
    with open(os.path.join(RESULTS, "lr", "chosen_lr.json"), "w") as f:
        json.dump({"lr": best["lr"], "grid": rows, "epochs": args.lr_epochs, "schedule": TRAIN["lr_schedule"],
                   "augment": env.augment}, f, indent=2)
    print(f"[lr] grid {rows} -> chosen lr = {best['lr']}")
    if best["lr"] in (min(LR_GRID), max(LR_GRID)):
        print(f"[lr] WARNING: chosen lr {best['lr']} is at the edge of the grid {LR_GRID}")


def _lr(args):
    return args.lr if args.lr is not None else chosen_lr()


def exp_main(env, args):
    lr = _lr(args)
    for seed in args.seeds:
        for scheme in args.partition:
            for method in args.methods:
                run_one(env, "main", method, scheme, lr, args.epochs, seed)


def exp_snr(env, args):
    lr = _lr(args)
    for seed in args.seeds:
        for scheme in args.partition:
            for rho in args.snrs:
                for method in [m for m in args.methods if m in ANALOG_METHODS]:
                    run_one(env, "snr", method, scheme, lr, args.epochs, seed, rho=rho)


def exp_nsweep(env, args):
    lr = _lr(args)
    for seed in args.seeds:
        for scheme in args.partition:
            for N in args.ns:
                for method in args.methods:
                    run_one(env, "nsweep", method, scheme, lr, args.epochs, seed, N=N)


def exp_cuts(env, args):
    lr = _lr(args)
    for seed in args.seeds:
        for scheme in args.partition:
            for stage in args.cuts:
                for method in [m for m in args.methods if m in ("airsfl", "sun_fdma_aircomp")]:
                    run_one(env, "cuts", method, scheme, lr, args.epochs, seed, stage=stage)


def exp_tau(env, args):
    lr = _lr(args)
    for seed in args.seeds:
        for scheme in args.partition:
            for tau in args.taus:
                for method in args.methods:
                    run_one(env, "tau", method, scheme, lr, args.epochs, seed, tau=tau)


def exp_nr(env, args):
    lr = _lr(args)
    for seed in args.seeds:
        for scheme in args.partition:
            for Nr in args.nrs:
                run_one(env, "nr", "airsfl", scheme, lr, args.epochs, seed, Nr=Nr)


EXPERIMENTS = {"bench": exp_bench, "lr": exp_lr, "main": exp_main, "snr": exp_snr, "nsweep": exp_nsweep,
               "cuts": exp_cuts, "tau": exp_tau, "nr": exp_nr}


def main():
    p = argparse.ArgumentParser(description="AirSFL vs baselines (CIFAR-10, ResNet-18 GN)")
    p.add_argument("--exp", nargs="+", default=["main"], choices=list(EXPERIMENTS))
    p.add_argument("--partition", nargs="+", default=["iid", "dirichlet"], choices=["iid", "dirichlet"])
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--epochs", type=float, default=TRAIN["epochs"], help="global-epoch equivalents")
    p.add_argument("--lr-epochs", type=float, default=LR_CAL_EPOCHS)
    p.add_argument("--lr", type=float, default=None, help="override the calibrated initial LR")
    p.add_argument("--seeds", nargs="+", type=int, default=[TRAIN["seed"]],
                   help="paired seeds (roadmap: 11 22 33 44 55): partitions, init, data order, channels")
    p.add_argument("--snrs", nargs="+", type=float, default=SNR_SWEEP)
    p.add_argument("--ns", nargs="+", type=int, default=N_SWEEP)
    p.add_argument("--cuts", nargs="+", type=int, default=CUT_SWEEP)
    p.add_argument("--taus", nargs="+", type=int, default=TAU_SWEEP)
    p.add_argument("--nrs", nargs="+", type=int, default=NR_SWEEP)
    p.add_argument("--augment", action="store_true", help="random crop + flip (default off)")
    p.add_argument("--engine", default=TRAIN["engine"], choices=["vmap", "loop"])
    p.add_argument("--vmap-chunk", type=int, default=None, help="clients per vmap chunk (GPU memory)")
    p.add_argument("--fast-cudnn", action="store_true",
                   help="let cuDNN pick the fastest (non-deterministic) FP32 algorithms")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results", default=None, help="results root (default examples/AirSFL/results)")
    p.add_argument("--eval-subset", type=int, default=0,
                   help="evaluate on the first K val/test images only (smoke tests); 0 = full sets")
    args = p.parse_args()

    global RESULTS
    if args.results:
        RESULTS = os.path.abspath(args.results)
    TRAIN["engine"], TRAIN["vmap_chunk"] = args.engine, args.vmap_chunk
    # FP32 everywhere (roadmap): no TF32 tensor-core shortcuts on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = bool(args.fast_cudnn)
    torch.backends.cudnn.deterministic = not args.fast_cudnn
    env = Env(torch.device(args.device), augment=args.augment)
    if args.eval_subset > 0:
        env.eval_sets = {k: (x[:args.eval_subset], y[:args.eval_subset]) for k, (x, y) in env.eval_sets.items()}
    print(f"[env] device={args.device} | results={RESULTS} | {ENV} | {TRAIN} | epochs={args.epochs} | "
          f"seeds={args.seeds} | n_train={env.n_train}")
    environment_summary(RadioConfig(**ENV), stage=TRAIN["stage"], tau=TRAIN["tau"])
    for e in args.exp:
        EXPERIMENTS[e](env, args)


if __name__ == "__main__":
    sys.exit(main())
