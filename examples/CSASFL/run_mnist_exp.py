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
  far too short at sigma_s = 2), --rounds caps each method's own rounds (default 50000; with
  --time-budget 0 every method runs exactly --rounds rounds). Communication = actual FP32
  tensors (device-model down/up, smashed data + labels up, smashed gradients down), cumulative MB;
  the 10k test set is evaluated after every global round. See flsim/csasfl/ for the exact
  semantics of every method.

Experiments (--exp, any subset; every run is stored once in <results>/runs/):
  main      FL, SFLv2, SAFSL, ParallelSFL, CSA-SFL  x  delta in --deltas (default 0.1 10)
  nsweep    CSA-SFL x N in --ns (default 3 5 7 9 11), H = 20, delta = 0.1   (variant: --csa-variant)
  hsweep    CSA-SFL x H in --hs (default 10 20 30 40 50), N = 7, delta = 0.1 (variant: --csa-variant)
  ablation  CSA-SFL full / w/o weighted aggregation (phi = 1/N) / w/o gradient clustering (random
            one-time clustering), delta = 0.1
  variants  CSA-SFL improvement screen (flsim.csasfl.methods.run_csasfl): paper / +system-aware
            clustering / +time slot / +update-based aggregation with sweeps staleness / +lazy
            re-clustering / all four / all but system-aware clustering, at --variant-deltas (default 0.1)
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
# CSA-SFL improvement screen (--exp variants). Each improvement alone, all four, and all but (1):
#   sys       (1) clustering on gradient direction + local-round time, lambda = --sys-lambda
#   slot      (2) time slot L = --deadline-q quantile of t_k: work capped at L, every round lasts >= L,
#                 FedNova-normalized device sides  ("deadline" = the cap without the >= L floor; with
#                 system-aware clustering it floods the PS with tiny fast-cluster rounds, so not in the default)
#   delta     (3) each cluster update applied once + staleness in sweeps (t - t_n)/N
#   lazy      (4) re-clustering from held updates, no halt, no discarded work
# Second screen (targets the problems found in the first one):
#   sys_lazy             (1)+(4): system-aware clustering without halts/restarts (slow clusters can finish)
#   slot_batch           (2) with simultaneous arrivals aggregated together (no restarts from partial models)
#   slot_batch_lazy      + lazy re-clustering (no halts)
#   slot_batch_lazy_q25  the same with a shorter slot (L = 25th percentile of t_k): less local drift on
#                        label-homogeneous clusters, twice as frequent mixing
#   sys_warm             (1) with warm-started re-clustering (Lloyd from the current partition, not a fresh
#                        k-means++ seeding): only clusters whose members really moved restart
VARIANT_NAMES = ["paper", "sys", "slot", "deadline", "delta", "lazy", "all", "all_nosys",
                 "sys_lazy", "slot_batch", "slot_batch_lazy", "slot_batch_lazy_q25", "sys_warm"]
SCREEN2 = ["sys_lazy", "slot_batch", "slot_batch_lazy", "slot_batch_lazy_q25"]
DEFAULT_VARIANTS = ["paper", "sys", "slot", "delta", "lazy", "all", "all_nosys"]


def variant_overrides(name: str, args) -> dict:
    one = {"sys": {"sys_lambda": args.sys_lambda},
           "slot": {"local_work": "slot", "deadline_q": args.deadline_q},
           "deadline": {"local_work": "deadline", "deadline_q": args.deadline_q},
           "delta": {"agg_mode": "delta", "staleness": "sweeps"},
           "lazy": {"recluster": "lazy"}}
    if name == "paper":
        return {}
    if name == "sys_lazy":
        return {**one["sys"], **one["lazy"]}
    if name == "sys_warm":
        return {**one["sys"], "recluster_init": "warm"}
    if name.startswith("slot_batch"):
        o = {**one["slot"], "batch_arrivals": True}
        if "lazy" in name:
            o.update(one["lazy"])
        if name.endswith("_q25"):
            o["deadline_q"] = 0.25
        return o
    if name in ("all", "all_nosys"):
        keys = ["sys", "slot", "delta", "lazy"] if name == "all" else ["slot", "delta", "lazy"]
        return {k: v for key in keys for k, v in one[key].items()}
    return dict(one[name])


# relative GPU time per run (sharding). Training-free count, seeds 1-5, 50,000 s budget: lock-steps
# (each one batched GPU step) / MNIST-epochs of local training per run -- ParallelSFL 346k / 4160,
# CSA-SFL 347k / 1400 (only ~4 devices per step: launch-overhead bound), SAFSL 215k / 3650,
# FL & SFLv2 19k / 400 (SFLv2: + a 50-device sequential server loop per step).
COST = {"parallelsfl": 4.0, "safsl": 3.5, "csasfl": 3.0, "sflv2": 0.8, "fl": 0.4}


def run_cost(cfg: RunConfig) -> float:
    """COST of the method, scaled for the CSA-SFL improvement variants by their measured local
    training per simulated second relative to the paper (system-aware clustering alone ~12x,
    time slot / deadline ~5x)."""
    c = COST.get(cfg.method, 1.0)
    if cfg.method == "csasfl":
        if cfg.local_work != "epochs":
            c *= 5.0 * (2.0 if cfg.deadline_q <= 0.25 else 1.0)
        elif cfg.sys_lambda > 0 or cfg.clustering == "system":
            c *= 12.0
    return c


def ablation_configs(name: str, args) -> list:
    """(label, RunConfig overrides) of the ablation of CSA-SFL variant `name`. Paper: the paper's ablation
    (full / uniform phi / random one-time clustering). An improved variant: the paper reference, the
    variant itself, the variant with each of its improvements removed (one at a time), with uniform phi
    and with random one-time clustering; a variant with system-aware clustering also gets
    "w/o gradient term" (one-time speed-only clusters). A removal that leaves the paper itself is
    merged into the paper entry."""
    if name == "paper":
        return [("paper", {}), ("w/o weighted aggregation", {"aggregation": "uniform"}),
                ("w/o gradient clustering", {"clustering": "random"})]
    v = variant_overrides(name, args)
    drop = lambda *ks: {k: x for k, x in v.items() if k not in ks}
    rem = []
    if v.get("local_work") in ("slot", "deadline"):
        rem.append(("time slot", drop("local_work", "deadline_q", "batch_arrivals")))
    if v.get("recluster") == "lazy":
        rem.append(("lazy re-clustering", drop("recluster")))
    if v.get("recluster_init") == "warm":
        rem.append(("warm re-clustering", drop("recluster_init")))
    if v.get("batch_arrivals"):
        rem.append(("batched arrivals", drop("batch_arrivals")))
    if v.get("sys_lambda"):
        rem.append(("system term", drop("sys_lambda")))
    if v.get("agg_mode") == "delta":
        rem.append(("update aggregation", drop("agg_mode", "staleness")))
    paper_label = "paper" + "".join(f" = w/o {lab}" for lab, ov in rem if not ov)
    out = [(paper_label, {}), ("full", dict(v))] + [(f"w/o {lab}", ov) for lab, ov in rem if ov]
    if v.get("sys_lambda"):
        out.append(("w/o gradient term (speed-only clusters)",
                    {**drop("sys_lambda", "recluster", "recluster_init"), "clustering": "system"}))
    out.append(("w/o weighted aggregation", {**v, "aggregation": "uniform"}))
    out.append(("w/o gradient clustering (random)",       # one-time clusterings: no re-clustering options
                {**drop("recluster", "recluster_init", "sys_lambda"), "clustering": "random"}))
    return out


ABLATION_PARTS = ["all", "full", "paper", "uniform", "random", "speed", "removals"]


def ablation_part(label: str) -> str:
    """--ablation-parts key of an ablation_configs label."""
    if label == "full":
        return "full"
    if label.startswith("paper"):
        return "paper"
    if label == "w/o weighted aggregation":
        return "uniform"
    if label.endswith("(random)") or label == "w/o gradient clustering":
        return "random"
    if "speed-only" in label:
        return "speed"
    return "removals"


def build_runs(args) -> list:
    """(experiments, RunConfig) for every requested run, deduplicated by identity."""
    base = dict(K=args.K, T=args.rounds, time_budget=args.time_budget, E=args.E, B=args.B, lr=args.lr,
                t0=args.t0, sigma_s=args.sigma_s, speed_drift_rho=args.speed_drift_rho, speed_drift_dt=args.speed_drift_dt)
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
        for csa in args.csa_variant:                              # CSA-SFL variant(s) of the sweeps / ablation
            cv = variant_overrides(csa, args)
            if "nsweep" in args.exp:
                for N in args.ns:
                    add("nsweep", method="csasfl", delta=NONIID, seed=seed, N=N, **cv)
            if "hsweep" in args.exp:
                for H in args.hs:
                    add("hsweep", method="csasfl", delta=NONIID, seed=seed, H=H, **cv)
            if "ablation" in args.exp:
                for lab, ov in ablation_configs(csa, args):
                    if "all" in args.ablation_parts or ablation_part(lab) in args.ablation_parts:
                        add("ablation", method="csasfl", delta=NONIID, seed=seed, **ov)
        if "variants" in args.exp:
            names = [v for n in args.variants for v in (SCREEN2 if n == "screen2" else [n])]
            for delta in args.variant_deltas:
                for name in names:
                    add("variants", method="csasfl", delta=delta, seed=seed, **variant_overrides(name, args))
    return [(sorted(e), c) for e, c in out.values()]


def manifest(cfg: RunConfig) -> dict:
    d = dataclasses.asdict(cfg)
    # improvement options / time-varying speeds enter the identity only when they differ from the paper
    # (finished runs keep their hash)
    for k, v in RunConfig.SYSTEM_DEFAULTS.items():
        if d[k] == v:
            d.pop(k)
    for k, v in RunConfig.IMPROVEMENT_DEFAULTS.items():
        if cfg.method != "csasfl" or d[k] == v:
            d.pop(k)
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
    if cfg.speed_drift_rho < 1.0:
        stop = f"_drift{cfg.speed_drift_rho:g}-{cfg.speed_drift_dt:g}" + stop
    return f"{cfg.method}_d{cfg.delta:g}_s{cfg.seed}{extra}_T{cfg.T}{stop}_{run_id(cfg)}"


def shard(runs: list, i: int, n: int) -> list:
    """Cost-balanced (LPT) deterministic split of the run list into n slices; slice i."""
    order = sorted(range(len(runs)), key=lambda j: (-run_cost(runs[j][1]), j))
    load, own = [0.0] * n, [[] for _ in range(n)]
    for j in order:
        s = int(np.argmin(load))
        load[s] += run_cost(runs[j][1])
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
    """A finished run of the same configuration that already contains this run. The time budget
    and the round cap T only decide WHEN a run stops (nothing before that depends on them), so
    another run covers this one if its rows reach this run's stopping point: the first round at
    or past cfg.time_budget, round cfg.T, or divergence. plots.py cuts every run at the common
    budget."""
    mine = manifest(cfg)
    for k in ("time_budget", "T"):
        mine.pop(k)
    for js in sorted(glob.glob(os.path.join(rdir, f"{cfg.method}_d{cfg.delta:g}_s{cfg.seed}_*.json"))):
        if os.path.basename(js)[:-5] == run_name(cfg):
            continue
        with open(js) as f:
            other = json.load(f)
        if {k: v for k, v in other.items() if k not in ("time_budget", "T", "run_id", "experiments")} != mine:
            continue
        path = js[:-5] + ".csv"
        if not is_complete(path, int(other["T"]), float(other.get("time_budget", 0.0))):
            continue
        df = pd.read_csv(path)
        stop = (df["round"] >= cfg.T) | df["diverged"].astype(bool)
        if cfg.time_budget > 0:
            stop |= df["sim_time_s"] >= cfg.time_budget
        if stop.any():
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
                 ("speed_drift_rho", cfg.speed_drift_rho), ("speed_drift_dt", cfg.speed_drift_dt),
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
    p.add_argument("--exp", nargs="+", default=["main"], choices=["main", "nsweep", "hsweep", "ablation", "variants"])
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=VARIANT_NAMES + ["screen2"],
                   help="CSA-SFL improvement variants of --exp variants ('screen2' = " + " ".join(SCREEN2) + ")")
    p.add_argument("--variant-deltas", nargs="+", type=float, default=[NONIID], help="deltas of --exp variants")
    p.add_argument("--csa-variant", nargs="+", default=["paper"], choices=VARIANT_NAMES,
                   help="CSA-SFL variant(s) used by --exp nsweep / hsweep / ablation (default: the paper's algorithm); "
                        "several = the sweeps and ablation of each")
    p.add_argument("--ablation-parts", nargs="+", default=["all"], choices=ABLATION_PARTS,
                   help="subset of --exp ablation: full (the variant itself), paper (reference), uniform (w/o weighted "
                        "aggregation), random (w/o gradient clustering), speed (speed-only clusters), removals (the "
                        "variant with each improvement removed); default all")
    p.add_argument("--sys-lambda", type=float, default=1.0, help="(1) weight of the system term in the clustering")
    p.add_argument("--deadline-q", type=float, default=0.5, help="(2) time slot L = this quantile of the devices' t_k")
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS, help="methods of --exp main")
    p.add_argument("--deltas", nargs="+", type=float, default=DELTAS, help="Dirichlet deltas of --exp main")
    p.add_argument("--seeds", nargs="+", type=int, default=[1], help="paired seeds (paper: 20 runs, e.g. 1..20)")
    p.add_argument("--ns", nargs="+", type=int, default=N_SWEEP)
    p.add_argument("--hs", nargs="+", type=int, default=H_SWEEP)
    p.add_argument("--rounds", type=int, default=50000,
                   help="safety cap on every method's own global rounds (the time budget normally stops a run first; "
                        "runs that stopped by time under a smaller cap are reused)")
    p.add_argument("--time-budget", type=float, default=50000.0,
                   help="stop every method once its simulated time reaches this (s); 0 = exactly --rounds rounds")
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--E", type=int, default=5)
    p.add_argument("--B", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--t0", type=float, default=0.2)
    p.add_argument("--sigma-s", type=float, default=2.0)
    p.add_argument("--speed-drift-rho", type=float, default=1.0,
                   help="time-varying device speeds: AR(1) coefficient of log s_k per --speed-drift-dt seconds "
                        "(1 = the paper's fixed speeds; e.g. 0.9). The LogNormal(0, sigma_s^2) heterogeneity is kept")
    p.add_argument("--speed-drift-dt", type=float, default=100.0, help="speed update interval (s)")
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
