"""
examples/CSASFL/run_mnist_exp.py: "mnist_exp" -- the CSA-SFL paper's simulation setup
(Sec. Performance Evaluation) for CSA-SFL and the baselines of examples/parallelsfl_experiment.py
(FL, SFLv2, SAFSL, ParallelSFL; IFCA left out for now).

Setup (paper; every value a default of flsim.csasfl.methods.RunConfig):
  K = 50 devices, s_k ~ LogNormal(0, 2^2) fixed per device, t_k = E ceil(|D_k|/B) t_0 / s_k with
  t_0 = 0.2 s (computation-only time); MNIST (60k train / 10k test), 12-layer chain LeNet
  (1,402,474 params = 5.61 MB, split after layer 8); per-class Dirichlet(delta) partition,
  delta in {0.1, 10}, >= 10 samples per device; SGD eta = 0.01, E = 5 local epochs, B = 64;
  CSA-SFL N = 7, H = 20. Stopping: every method trains until its simulated clock reaches the same
  --time-budget (default 50,000 s, ~2000 CSA-SFL rounds; the paper's T = 150 CSA-SFL rounds is
  far too short at sigma_s = 2), --rounds caps each method's own rounds (default 5000; with
  --time-budget 0 every method runs exactly --rounds rounds). Communication = actual FP32
  tensors (device-model down/up, smashed data + labels up, smashed gradients down), cumulative MB;
  the 10k test set is evaluated after every global round. See flsim/csasfl/ for the exact
  semantics of every method.

Experiments (--exp, any subset; every run is stored once in <results>/runs/):
  main      FL, SFLv2, SAFSL, ParallelSFL, CSA-SFL  x  delta in --deltas (default 0.1 10)
  nsweep    CSA-SFL x N in --ns (default 3 5 7 9 11), H = 20, delta = 0.1
  hsweep    CSA-SFL x H in --hs (default 10 20 30 40 50), N = 7, delta = 0.1
  ablation  CSA-SFL full / w/o weighted aggregation (phi = 1/N) / w/o gradient clustering (random
            one-time clustering), delta = 0.1
Each run is identified by a hash of its complete configuration; a finished run (all T rounds or
diverged) is skipped, so a timed-out job is resubmitted unchanged, and the run shared by several
experiments (CSA-SFL N=7 H=20 delta=0.1 is in all four) is trained once. Paired seeds: the same
seed gives every method the same partition, device speeds, initial model and data streams.

GPU: the data live on the GPU and every group of devices trains in lock-step through one
batched vmap kernel (flsim.csasfl.engine); a run that runs out of GPU memory restarts with
chunked vmap, then with the loop engine. Several processes can share one GPU: --shard i/n runs the
i-th of n cost-balanced slices of the run list (slurm/run_csasfl_mnist.slurm starts them).

Usage:
  python examples/CSASFL/run_mnist_exp.py --exp main nsweep hsweep ablation --seeds 1 2 3
  python examples/CSASFL/run_mnist_exp.py --exp main --seeds 1 --shard 0/4
  python examples/CSASFL/run_mnist_exp.py --exp main --seeds 1 --list
"""

import argparse
import dataclasses
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from flsim.csasfl.data import PARTITION_REDRAWS, load_mnist
from flsim.csasfl.engine import Engine
from flsim.csasfl.methods import METHOD_LABELS, RunConfig, run
from flsim.csasfl.model import CUT_LAYER, FC_WIDTHS, layer_spec

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results_mnist_exp")
SCHEMA = 1
METHODS = ["fl", "sflv2", "safsl", "parallelsfl", "csasfl"]
DELTAS = [0.1, 10.0]
NONIID = 0.1
N_SWEEP = [3, 5, 7, 9, 11]
H_SWEEP = [10, 20, 30, 40, 50]
ABLATION = [("gradient", "weighted"), ("gradient", "uniform"), ("random", "weighted")]
# relative GPU time per run (sharding). Training-free count, seeds 1-5, 50,000 s budget: lock-steps
# (each one batched GPU step) / MNIST-epochs of local training per run -- ParallelSFL 346k / 4160,
# CSA-SFL 347k / 1400 (only ~4 devices per step: launch-overhead bound), SAFSL 215k / 3650,
# FL & SFLv2 19k / 400 (SFLv2: + a 50-device sequential server loop per step).
COST = {"parallelsfl": 4.0, "safsl": 3.5, "csasfl": 3.0, "sflv2": 0.8, "fl": 0.4}


def build_runs(args) -> list:
    """(experiments, RunConfig) for every requested run, deduplicated by identity."""
    base = dict(K=args.K, T=args.rounds, time_budget=args.time_budget, E=args.E, B=args.B, lr=args.lr,
                t0=args.t0, sigma_s=args.sigma_s)
    out = {}

    def add(exp, **kw):
        cfg = RunConfig(**{**base, **kw})
        rid = run_id(cfg)
        out.setdefault(rid, [set(), cfg])[0].add(exp)

    for seed in args.seeds:
        if "main" in args.exp:
            for delta in args.deltas:
                for m in args.methods:
                    add("main", method=m, delta=delta, seed=seed)
        if "nsweep" in args.exp:
            for N in args.ns:
                add("nsweep", method="csasfl", delta=NONIID, seed=seed, N=N)
        if "hsweep" in args.exp:
            for H in args.hs:
                add("hsweep", method="csasfl", delta=NONIID, seed=seed, H=H)
        if "ablation" in args.exp:
            for cl, ag in ABLATION:
                add("ablation", method="csasfl", delta=NONIID, seed=seed, clustering=cl, aggregation=ag)
    return [(sorted(e), c) for e, c in out.values()]


def manifest(cfg: RunConfig) -> dict:
    d = dataclasses.asdict(cfg)
    if cfg.method != "csasfl":                   # CSA-SFL-only knobs do not change a baseline's run
        for k in ("N", "H", "clustering", "aggregation"):
            d.pop(k)
    if cfg.method != "safsl":
        d.pop("safsl_buffer_frac"); d.pop("safsl_gamma")
    if cfg.method != "parallelsfl":
        d.pop("psfl_lambda"); d.pop("psfl_max_cluster")
    return {"schema": SCHEMA, "model": f"lenet12-fc{'-'.join(map(str, FC_WIDTHS))}-he", "dataset": "mnist60k", **d}


def run_id(cfg: RunConfig) -> str:
    return hashlib.sha256(json.dumps(manifest(cfg), sort_keys=True).encode()).hexdigest()[:10]


def run_name(cfg: RunConfig) -> str:
    extra = f"_N{cfg.N}_H{cfg.H}_{cfg.variant()}" if cfg.method == "csasfl" else ""
    stop = f"_tb{cfg.time_budget:g}" if cfg.time_budget > 0 else ""
    return f"{cfg.method}_d{cfg.delta:g}_s{cfg.seed}{extra}_T{cfg.T}{stop}_{run_id(cfg)}"


def shard(runs: list, i: int, n: int) -> list:
    """Cost-balanced (LPT) deterministic split of the run list into n slices; slice i."""
    order = sorted(range(len(runs)), key=lambda j: (-COST.get(runs[j][1].method, 1.0), j))
    load, own = [0.0] * n, [[] for _ in range(n)]
    for j in order:
        s = int(np.argmin(load))
        load[s] += COST.get(runs[j][1].method, 1.0)
        own[s].append(j)
    return [runs[j] for j in sorted(own[i])]


def is_complete(path: str, T: int, time_budget: float = 0.0) -> bool:
    """All T rounds, the simulated-time budget reached, or diverged."""
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    return len(df) > 0 and (int(df["round"].max()) >= T or bool(df["diverged"].astype(bool).any())
                            or (time_budget > 0 and float(df["sim_time_s"].max()) >= time_budget))


def covering_run(cfg: RunConfig, rdir: str):
    """A finished run of the same configuration with a LONGER (or no) time budget. The budget
    only decides when a run stops -- nothing before that depends on it -- so the first rows of
    the longer run, up to the first round at or past cfg.time_budget, ARE this run (plots.py
    cuts every run at the common budget)."""
    if cfg.time_budget <= 0:
        return None
    mine = manifest(cfg)
    mine.pop("time_budget")
    for js in sorted(glob.glob(os.path.join(rdir, f"{cfg.method}_d{cfg.delta:g}_s{cfg.seed}_*.json"))):
        with open(js) as f:
            other = json.load(f)
        tb = float(other.get("time_budget", 0.0))
        if not (tb <= 0 or tb > cfg.time_budget):
            continue
        if {k: v for k, v in other.items() if k not in ("time_budget", "run_id", "experiments")} != mine:
            continue
        if is_complete(js[:-5] + ".csv", int(other["T"]), tb):
            return os.path.basename(js)[:-5]
    return None


def run_one(cfg: RunConfig, exps: list, data: dict, args) -> str:
    name = run_name(cfg)
    rdir = os.path.join(RESULTS, "runs")
    os.makedirs(rdir, exist_ok=True)
    csv_path = os.path.join(rdir, name + ".csv")
    if is_complete(csv_path, cfg.T, cfg.time_budget):
        print(f"[skip] {name} (complete)")
        return csv_path
    cover = covering_run(cfg, rdir)
    if cover is not None:
        print(f"[skip] {name}: covered by the finished longer-budget run {cover}")
        return os.path.join(rdir, cover + ".csv")
    with open(os.path.join(rdir, name + ".json"), "w") as f:
        json.dump({"run_id": run_id(cfg), "experiments": exps, **manifest(cfg)}, f, indent=1, sort_keys=True)
    spec = layer_spec()
    with open(os.path.join(rdir, name + ".log"), "w") as logf:
        def log(msg):
            print(msg, flush=True)
            logf.write(msg + "\n")
            logf.flush()

        attempts = [(args.engine, args.vmap_chunk)]
        if args.engine == "vmap":
            attempts += [("vmap", 8), ("loop", None)]
        oom = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
        t0 = time.time()
        for i, (eng_name, chunk) in enumerate(attempts):
            try:
                eng = Engine(spec, CUT_LAYER, cfg.lr, data["x_train"].device, engine=eng_name, chunk=chunk)
                rows = run(cfg, data, eng, log=log)
                break
            except oom as e:
                if i == len(attempts) - 1 or (oom is RuntimeError and "out of memory" not in str(e)):
                    raise
                torch.cuda.empty_cache()
                log(f"[oom] {eng_name} (chunk={chunk}) ran out of GPU memory -> retrying with "
                    f"{attempts[i + 1][0]} (chunk={attempts[i + 1][1]})")
    df = pd.DataFrame(rows)
    for k, v in [("method", cfg.method), ("label", METHOD_LABELS[cfg.method]), ("variant", cfg.variant()),
                 ("delta", cfg.delta), ("seed", cfg.seed), ("K", cfg.K), ("E", cfg.E), ("B", cfg.B),
                 ("lr", cfg.lr), ("T", cfg.T), ("time_budget", cfg.time_budget), ("t0", cfg.t0),
                 ("sigma_s", cfg.sigma_s),
                 ("N", cfg.N if cfg.method == "csasfl" else np.nan), ("H", cfg.H if cfg.method == "csasfl" else np.nan),
                 ("engine", eng.engine), ("run_id", run_id(cfg)), ("experiments", "+".join(exps)),
                 ("partition_redraws", PARTITION_REDRAWS.get((cfg.K, float(cfg.delta), int(cfg.seed),
                                                              int(cfg.min_per_device)), -1))]:
        df[k] = v
    tmp = csv_path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)
    print(f"[done] {name}: final acc {df['test_acc'].iloc[-1]:.4f}, sim time {df['sim_time_s'].iloc[-1]:.0f}s, "
          f"comm {df['comm_mb'].iloc[-1]:.0f} MB, wall {time.time() - t0:.0f}s -> {csv_path}")
    return csv_path


def main():
    global RESULTS
    p = argparse.ArgumentParser(description="mnist_exp: CSA-SFL paper setup (comparison, N/H sweeps, ablation)")
    p.add_argument("--exp", nargs="+", default=["main"], choices=["main", "nsweep", "hsweep", "ablation"])
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS, help="methods of --exp main")
    p.add_argument("--deltas", nargs="+", type=float, default=DELTAS, help="Dirichlet deltas of --exp main")
    p.add_argument("--seeds", nargs="+", type=int, default=[1], help="paired seeds (paper: 20 runs, e.g. 1..20)")
    p.add_argument("--ns", nargs="+", type=int, default=N_SWEEP)
    p.add_argument("--hs", nargs="+", type=int, default=H_SWEEP)
    p.add_argument("--rounds", type=int, default=5000, help="cap on every method's own global rounds")
    p.add_argument("--time-budget", type=float, default=50000.0,
                   help="stop every method once its simulated time reaches this (s); 0 = exactly --rounds rounds")
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--E", type=int, default=5)
    p.add_argument("--B", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--t0", type=float, default=0.2)
    p.add_argument("--sigma-s", type=float, default=2.0)
    p.add_argument("--engine", default="vmap", choices=["vmap", "loop"])
    p.add_argument("--vmap-chunk", type=int, default=None, help="devices per vmap chunk (GPU memory)")
    p.add_argument("--shard", default="0/1", help="i/n: run only the i-th of n cost-balanced slices")
    p.add_argument("--results", default=None, help=f"results root (default {RESULTS})")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tf32", action="store_true", help="allow TF32 tensor cores (default: exact FP32)")
    p.add_argument("--deterministic", action="store_true", help="deterministic cuDNN (slower)")
    p.add_argument("--eval-subset", type=int, default=0, help="evaluate on the first K test images (smoke tests)")
    p.add_argument("--train-subset", type=int, default=0, help="partition only the first K training images (smoke tests)")
    p.add_argument("--list", action="store_true", help="print this shard's run list and exit")
    args = p.parse_args()

    if args.results:
        RESULTS = os.path.abspath(args.results)
    si, sn = (int(v) for v in args.shard.split("/"))
    runs = build_runs(args)
    mine = shard(runs, si, sn)
    print(f"[mnist_exp] {len(runs)} runs requested, shard {si}/{sn}: {len(mine)} runs | results={RESULTS}")
    if args.list:
        for exps, cfg in mine:
            print(f"  {run_name(cfg):70s} {'+'.join(exps)}"
                  f"{'  (complete)' if is_complete(os.path.join(RESULTS, 'runs', run_name(cfg) + '.csv'), cfg.T, cfg.time_budget) else ''}")
        return 0
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.benchmark = not args.deterministic
    torch.backends.cudnn.deterministic = bool(args.deterministic)
    data = load_mnist(device=torch.device(args.device))
    if args.eval_subset > 0:
        data["x_test"], data["y_test"] = data["x_test"][:args.eval_subset], data["y_test"][:args.eval_subset]
    if args.train_subset > 0:        # smoke tests only: such runs land in their own results folder
        if args.results is None:
            raise SystemExit("--train-subset needs an explicit --results folder (keeps smoke runs apart)")
        for k in ("x_train", "y_train", "y_train_np"):
            data[k] = data[k][:args.train_subset]
    print(f"[mnist_exp] device={args.device} | train {tuple(data['x_train'].shape)} test {tuple(data['x_test'].shape)}")
    for exps, cfg in mine:
        run_one(cfg, exps, data, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
