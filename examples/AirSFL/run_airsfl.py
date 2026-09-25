"""
examples/AirSFL/run_airsfl.py: AirSFL vs baselines on CIFAR-10 / ResNet-18 (GroupNorm).

One matched environment (roadmap Sec. 3-4), all methods share data, init, model,
minibatches, augmentation, plain SGD, tau, full participation, weights a_n,
bandwidth, per-client power and server antennas:

  AirSFL | Digital SFL-V1 | FDMA-AirComp SFL (Sun-style) | AirComp-FL | Digital FedAvg

Default environment ("reasonable & clear"): N=30 clients, Nr=128 antennas,
S=60 subcarriers x 15 kHz (W=0.9 MHz), Pmax=0.1 W, N0=-167 dBm/Hz, eps=0.8,
rho=20 dB, cut after residual stage 3, tau=5, B=16, 50 global-epoch equivalents.

Experiments (--exp, any subset; each run is resumable -- a finished CSV is skipped):
  lr      LR calibration on the noiseless reference (digital FedAvg), grid
          {0.01, 0.03, 0.1} (roadmap Sec. 4); the chosen LR is then used by ALL methods.
  main    5 methods x {IID, Dirichlet(0.5)} at rho = 20 dB  -> accuracy vs uplink time.
  snr     analog methods x rho in {0, 10, 30} dB (20 dB comes from `main`). Digital
          methods' learning is SNR-independent (reliable transport): only their
          time axis changes, which plots.py recomputes analytically -- no re-run.
  nr      AirSFL x Nr in {32, 48, 64, 128} at rho = 10 dB (ZF conditioning / spatial load).
  nsweep  5 methods x N in {20, 40} at rho = 20 dB, fixed Nr and S (N = 30 from `main`):
          digital OFDMA splits the band (S_n = S/N) while AirSFL reuses it.

Outputs: examples/AirSFL/results/<exp>/<run>.csv (one row per evaluation),
then run examples/AirSFL/plots.py to build the figures and tables.

Usage:
  python examples/AirSFL/run_airsfl.py --exp lr
  python examples/AirSFL/run_airsfl.py --exp main --partition iid dirichlet
  python examples/AirSFL/run_airsfl.py --exp snr nr
  python examples/AirSFL/run_airsfl.py --exp nsweep --partition iid dirichlet
"""

import argparse
import copy
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch

from flsim.airsfl.data import ClientStream, load_cifar10_tensors, make_eval_tensors, partition
from flsim.airsfl.simulator import AirSFLSimulator, RunConfig
from flsim.airsfl.timing import RadioConfig

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

METHODS = ["airsfl", "digital_sflv1", "sun_fdma_aircomp", "aircomp_fl", "digital_fedavg"]
ANALOG_METHODS = ["airsfl", "sun_fdma_aircomp", "aircomp_fl"]

ENV = dict(N=30, Nr=128, S=60, df_hz=15e3, Pmax_w=0.1, N0_dbm_per_hz=-167.0,
           eps_D=0.8, eps_U=0.8, eps_A=0.8, rho_db=20.0, Pdl_w=0.3)
TRAIN = dict(stage=3, tau=5, batch_size=16, epochs=50, lr=0.03, dirichlet_alpha=0.5,
             min_per_client=100, seed=11, evals_per_epoch=1)
LR_GRID = [0.01, 0.03, 0.1]
LR_CAL_EPOCHS = 10
SNR_SWEEP = [0.0, 10.0, 30.0]
NR_SWEEP = [32, 48, 64, 128]
NR_SWEEP_SNR = 10.0
N_SWEEP = [20, 40]      # client-count sweep at fixed Nr, S (N=30 comes from `main`);
                        # N=40 -> S_n = 1.5 tones (fluid OFDMA goodput, Eq. 22)


class Env:
    """Loads CIFAR once; builds matched client streams per (partition, seed)."""

    def __init__(self, device):
        self.device = device
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
                             seed * 1000 + k, self.device) for k, p in enumerate(self.parts(scheme, N, seed))]

    def weights(self, scheme, N, seed):
        w = np.array([len(p) for p in self.parts(scheme, N, seed)], dtype=np.float64)
        return w / w.sum()


def rounds_for(epochs, N, tau, B, n_train):
    return int(math.ceil(epochs * n_train / (N * tau * B)))


def run_one(env: Env, exp: str, method: str, scheme: str, rho_db: float, Nr: int, lr: float,
            epochs: float, stage=None, seed=None, tag="", noiseless=False, N=None) -> str:
    stage = TRAIN["stage"] if stage is None else stage
    seed = TRAIN["seed"] if seed is None else seed
    B, tau = TRAIN["batch_size"], TRAIN["tau"]
    if N is None:
        N = ENV["N"]
    else:
        tag = f"_N{N}{tag}"
    name = f"{method}_{scheme}_rho{rho_db:g}_Nr{Nr}_cut{stage}_lr{lr:g}_s{seed}{tag}"
    out_dir = os.path.join(RESULTS, exp)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, name + ".csv")
    R = rounds_for(epochs, N, tau, B, env.n_train)
    if os.path.exists(csv_path):
        done = pd.read_csv(csv_path)
        if len(done) and int(done["round"].max()) >= R:
            print(f"[skip] {name} (complete)")
            return csv_path
    radio = RadioConfig(**{**ENV, "N": N, "Nr": Nr, "rho_db": rho_db})
    rounds_per_epoch = env.n_train / (N * tau * B)
    eval_every = max(1, int(round(rounds_per_epoch / TRAIN["evals_per_epoch"])))
    cfg = RunConfig(method=method, stage=stage, tau=tau, batch_size=B, lr=lr, rounds=R,
                    eval_every=eval_every, noiseless=noiseless, seed=seed)
    log_path = os.path.join(out_dir, name + ".log")
    with open(log_path, "w") as logf:
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
    df["ul_s_per_round"] = sim.ul_s
    df["dl_s_per_round"] = sim.dl_s
    df["mb_per_round"] = sim.mb
    df["ul_activation_s"] = sim.ul_break["activation"]
    df["ul_model_s"] = sim.ul_break["model"]
    df["ul_labels_s"] = sim.ul_break["labels"]
    df["d_c"], df["d_s"], df["d_a"] = sim.dims["d_c"], sim.dims["d_s"], sim.dims["d_a"]
    df.to_csv(csv_path, index=False)
    print(f"[done] {name} -> {csv_path}")
    return csv_path


def chosen_lr() -> float:
    """LR picked by `--exp lr` (best final validation accuracy of the noiseless
    reference); falls back to TRAIN['lr'] if calibration has not been run."""
    path = os.path.join(RESULTS, "lr", "chosen_lr.json")
    if os.path.exists(path):
        with open(path) as f:
            return float(json.load(f)["lr"])
    return TRAIN["lr"]


def exp_lr(env, args):
    rows = []
    for lr in LR_GRID:
        p = run_one(env, "lr", "digital_fedavg", "iid", ENV["rho_db"], ENV["Nr"], lr,
                    args.lr_epochs, noiseless=True)
        df = pd.read_csv(p)
        rows.append({"lr": lr, "final_val_acc": float(df["val_acc"].iloc[-1])})
    best = max(rows, key=lambda r: r["final_val_acc"])
    with open(os.path.join(RESULTS, "lr", "chosen_lr.json"), "w") as f:
        json.dump({"lr": best["lr"], "grid": rows, "epochs": args.lr_epochs}, f, indent=2)
    print(f"[lr] grid {rows} -> chosen lr = {best['lr']}")


def exp_main(env, args):
    lr = args.lr or chosen_lr()
    for seed in args.seeds:
        for scheme in args.partition:
            for method in args.methods:
                run_one(env, "main", method, scheme, ENV["rho_db"], ENV["Nr"], lr, args.epochs, seed=seed)


def exp_snr(env, args):
    lr = args.lr or chosen_lr()
    for seed in args.seeds:
        for scheme in args.partition:
            for rho in SNR_SWEEP:
                for method in [m for m in args.methods if m in ANALOG_METHODS]:
                    run_one(env, "snr", method, scheme, rho, ENV["Nr"], lr, args.epochs, seed=seed)


def exp_nr(env, args):
    lr = args.lr or chosen_lr()
    for seed in args.seeds:
        for scheme in args.partition:
            for Nr in NR_SWEEP:
                run_one(env, "nr", "airsfl", scheme, NR_SWEEP_SNR, Nr, lr, args.epochs, seed=seed)
            run_one(env, "nr", "digital_sflv1", scheme, NR_SWEEP_SNR, ENV["Nr"], lr, args.epochs, seed=seed)


def exp_nsweep(env, args):
    """All methods at N in N_SWEEP (fixed Nr, S, rho = 20 dB, same epochs and LR).
    Digital FedAvg is included: it sets A* for each N. N = 30 is reused from `main`."""
    lr = args.lr or chosen_lr()
    for seed in args.seeds:
        for scheme in args.partition:
            for N in N_SWEEP:
                for method in args.methods:
                    run_one(env, "nsweep", method, scheme, ENV["rho_db"], ENV["Nr"], lr, args.epochs,
                            seed=seed, N=N)


def main():
    p = argparse.ArgumentParser(description="AirSFL vs baselines (CIFAR-10, ResNet-18 GN)")
    p.add_argument("--exp", nargs="+", default=["main"], choices=["lr", "main", "snr", "nr", "nsweep"])
    p.add_argument("--partition", nargs="+", default=["iid", "dirichlet"], choices=["iid", "dirichlet"])
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--epochs", type=float, default=TRAIN["epochs"])
    p.add_argument("--lr-epochs", type=float, default=LR_CAL_EPOCHS)
    p.add_argument("--lr", type=float, default=None, help="override the calibrated LR")
    p.add_argument("--seeds", nargs="+", type=int, default=[TRAIN["seed"]],
                   help="paired seeds (roadmap reported seeds: 11 22 33 44 55); each seed pairs "
                        "init, data streams and radio noise across methods")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results", default=None, help="results root (default examples/AirSFL/results)")
    p.add_argument("--eval-subset", type=int, default=0,
                   help="evaluate on the first K val/test images only (smoke tests); 0 = full sets")
    args = p.parse_args()

    global RESULTS
    if args.results:
        RESULTS = os.path.abspath(args.results)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    env = Env(torch.device(args.device))
    if args.eval_subset > 0:
        env.eval_sets = {k: (x[:args.eval_subset], y[:args.eval_subset]) for k, (x, y) in env.eval_sets.items()}
    print(f"[env] device={args.device} | {ENV} | {TRAIN} | n_train={env.n_train}")
    for e in args.exp:
        {"lr": exp_lr, "main": exp_main, "snr": exp_snr, "nr": exp_nr, "nsweep": exp_nsweep}[e](env, args)


if __name__ == "__main__":
    sys.exit(main())
