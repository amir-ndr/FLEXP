"""
flsim/csasfl/methods.py: CSA-SFL and the baselines (FL, SFLv2, SAFSL, ParallelSFL) on the CSA-SFL
paper's simulation setup (flsim.csasfl.system / data / model), all trained with the batched
engine (flsim.csasfl.engine) and timed with the same computation model.

Every method of a seed shares: the partition, the device speeds s_k, the initial model, plain
SGD with eta, B and E local epochs, the evaluation (10k MNIST test images after every global
round) and the communication accounting. Stopping (user 2026-10-19): every method trains until its
simulated clock reaches the same time budget (default 50,000 s, ~2000 CSA-SFL rounds) -- a
global round means very different work per method (one cluster for CSA-SFL, all K devices for
FL/SFLv2) -- with T as a cap on each method's own rounds (time_budget = 0: exactly T rounds).

CSA-SFL (paper Sec. Mechanism Design + Algorithm 1)
  * clustering: device-side gradients g_k from one co-training mini-batch, cosine K-means
    (flsim.algorithms.csa_sfl.CSASFL.cluster_by_gradient, centroids mu_n = mean of the g_k);
    initially on w^0, then every H global rounds (if t+1 < T) on each device's OWN current
    model version (its cluster's w^{t_n}, device side + that cluster's server side).
    All clusters halt for the recollection (halt = max_k t_0/s_k, every device runs one batch).
    A new cluster whose member set equals an old one is unaffected and resumes its ongoing
    round (its completion is pushed back by the halt; buffer, t_n kept); every other cluster
    is affected: reinitialized from w^{t+1}, buffer <- w^{t+1}, t_m <- t+1, devices download
    w^{c,t+1}. The just-completed cluster always starts its next round from w^{t+1}.
  * intra-cluster co-training from w^{t_n} for E local epochs: device k makes its own
    E*ceil(|D_k|/B) mini-batch steps; at every step the cluster's single server-side submodel
    takes  w^s <- w^s - eta * sum_k |D_k| g_k^s / sum_k |D_k|  over the devices still running
    (= Eq. server_update while all are running), device sides step with their own g_k^c
    (Eq. device_update); then  w_bar^c = sum_k |D_k| w_k^{c,E} / |D_n|  (Eq. intra_agg) and the
    cluster model is [w_bar^c, w^{s,E}] (Eq. form). Duration = max_{k in C_n} t_k.
  * inter-cluster: the completed model replaces the cluster's buffer; tau_n^t = t - t_n;
    phi_{n,t} = |D_n|/(1+tau_n^t) / sum (Eq. phi; CSASFL.agg_weights);
    w^{t+1} = sum_n phi_{n,t} w_bar_n^b over ALL N buffers (Eq. global_update);
    w^{t+1} goes only to the completed cluster (t_n <- t+1); the others keep training.
  Ablations: aggregation="uniform" (phi = 1/N), clustering="random" (random one-time
  clustering, no gradient collection, no re-clustering).

Baselines (the same set as examples/parallelsfl_experiment.py)
  * FL      synchronous FedAvg, all K devices train the full model E epochs, |D_k|-weighted
            average; round time max_k t_k.
  * SFLv2   synchronous split FL with ONE server-side submodel processed sequentially: per
            lock-step every running device sends its batch, the PS processes the devices in a
            random order (fresh per round), one SGD step each, returning each device's
            smashed-data gradient; device sides are |D_k|-averaged after E epochs; round time
            max_k t_k (devices compute in parallel, PS time not modelled).
  * SAFSL   semi-asynchronous split FL: all K devices in flight, each trains its own
            (device, server) pair from the global model it was dispatched with for E epochs and
            arrives after t_k; the PS aggregates the k = 0.8K earliest arrivals with
            rho = |D_n|^gamma (staleness+1) (flsim.algorithms.safsl.SAFSL, gamma = 1) and
            re-dispatches them from the new model.
  * ParallelSFL  clustered split FL (flsim.algorithms.parallel_sfl.ParallelSFL: Algorithm 1
            clustering by label distribution + waiting time, Eq. 17 local frequency, Eq. 3/4/18
            aggregation) re-run every round; the top worker (a device, not training its own data)
            holds the cluster's top submodel. Its local updating frequency is in iterations
            (its native unit): the fastest cluster runs tau_max = E*ceil(mean|D_k|/B) iterations
            (E epochs of an average device), slower clusters fewer (Eq. 17); bottom worker
            iteration time t_0/s_i, top-submodel time r t_0/s_top per batch (no transmission
            time, as in the paper's model; equal link capacity -> the fastest device of the set
            becomes the top worker). Round time max_c tau_c t_{c,o} (Eq. 15).
"""

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from flsim.algorithms.csa_sfl import CSASFL
from flsim.algorithms.parallel_sfl import ParallelSFL
from flsim.algorithms.safsl import SAFSL
from flsim.csasfl.data import DeviceStream, dirichlet_partition, label_histograms
from flsim.csasfl.model import activation_numel, layer_macs
from flsim.csasfl.system import Dims, SpeedProcess, batches, device_speeds, train_time
from flsim.interfaces.parallel_sfl_algorithm import WorkerInfo

METHOD_LABELS = {"fl": "FL", "cfl": "CFL", "sflv2": "SFLv2", "safsl": "SAFSL", "parallelsfl": "ParallelSFL",
                 "csasfl": "CSA-SFL"}


@dataclass
class RunConfig:
    method: str                      # fl | sflv2 | safsl | parallelsfl | csasfl
    delta: float = 0.1               # Dirichlet concentration
    seed: int = 1
    K: int = 50                      # devices
    E: int = 5                       # local epochs
    B: int = 64                      # mini-batch size
    lr: float = 0.01                 # SGD learning rate eta
    T: int = 5000                    # cap on global rounds (each method's own)
    time_budget: float = 50000.0     # stop once simulated time >= this (s); 0 = run exactly T rounds
    t0: float = 0.2                  # s per mini-batch at unit speed
    sigma_s: float = 2.0             # s_k ~ LogNormal(0, sigma_s^2)
    min_per_device: int = 10         # partition: minimum samples per device
    cut: int = 8                     # split layer
    # CSA-SFL
    N: int = 7                       # clusters
    H: int = 20                      # re-clustering interval (global rounds)
    clustering: str = "gradient"     # gradient | random | system (ablations: one-time random / speed-only clusters)
    aggregation: str = "weighted"    # weighted | uniform (ablation)
    # CSA-SFL improvements (defaults = the paper's algorithm; see run_csasfl)
    sys_lambda: float = 0.0          # (1) system-aware clustering weight (0 = paper)
    local_work: str = "epochs"       # (2) epochs (paper: E full epochs) | deadline (work capped at slot L) |
                                     #     slot (capped AND every cluster round lasts >= L)
    deadline_q: float = 0.5          # (2) slot L = deadline_q-quantile of all devices' t_k
    agg_mode: str = "buffer"         # (3) buffer (paper Eq. global_update) | delta (each update applied once)
    staleness: str = "rounds"        # (3) rounds (paper tau = t - t_n) | sweeps (tau = (t - t_n) / N)
    recluster: str = "halt"          # (4) halt (paper: halt, re-probe, restart affected) | lazy
    batch_arrivals: bool = False     # (2b) clusters finishing at the same instant -> one aggregation (slot mode)
    recluster_init: str = "seed"     # (4b) seed (fresh k-means++ every re-clustering) | warm (Lloyd from the current partition)
    staleness_exp: float = 1.0       # (3b) a in phi = |D_n| / (1+tau_n)^a (paper: 1; < 1 = gentler staleness discount)
    # SAFSL
    safsl_buffer_frac: float = 0.8   # aggregate the 0.8K earliest arrivals
    safsl_gamma: float = 1.0
    # ParallelSFL
    psfl_lambda: float = 0.5
    psfl_max_cluster: int = 8
    # CFL (Sattler et al., TNNLS 2021): split a cluster when ||mean update|| < eps1 and max_k ||update_k|| > eps2,
    # eps1 = cfl_eps1_frac * (largest ||mean update|| of the cluster so far), eps2 = cfl_eps2_mult * eps1,
    # and the bipartition passes sqrt((1 - alpha_cross_max) / 2) > cfl_gamma_max (Eq. 36)
    cfl_eps1_frac: float = 0.1
    cfl_eps2_mult: float = 4.0
    cfl_gamma_max: float = 0.0
    # time-varying device speeds (flsim.csasfl.system.SpeedProcess; rho = 1: the paper's fixed speeds)
    speed_drift_rho: float = 1.0     # AR(1) coefficient of log s_k per interval
    speed_drift_dt: float = 100.0    # interval length (s); correlation time = dt / (1 - rho)

    SYSTEM_DEFAULTS = {"speed_drift_rho": 1.0, "speed_drift_dt": 100.0}
    IMPROVEMENT_DEFAULTS = {"sys_lambda": 0.0, "local_work": "epochs", "deadline_q": 0.5,
                            "agg_mode": "buffer", "staleness": "rounds", "recluster": "halt",
                            "batch_arrivals": False, "recluster_init": "seed",
                            "staleness_exp": 1.0}

    def improvements(self) -> dict:
        """The CSA-SFL improvement options that differ from the paper's algorithm."""
        return {k: getattr(self, k) for k, v in self.IMPROVEMENT_DEFAULTS.items() if getattr(self, k) != v}

    def variant(self) -> str:
        if self.method != "csasfl":
            return "-"
        base = {("gradient", "weighted"): "full", ("gradient", "uniform"): "uniform_agg",
                ("random", "weighted"): "random_cluster",
                ("system", "weighted"): "system_cluster"}.get((self.clustering, self.aggregation),
                                                              f"{self.clustering}_{self.aggregation}")
        imp = self.improvements()
        tags = []
        if "sys_lambda" in imp:
            tags.append(f"sys{self.sys_lambda:g}")
        if self.local_work == "deadline":
            tags.append(f"dl{self.deadline_q:g}")
        if self.local_work == "slot":
            tags.append(f"slot{self.deadline_q:g}")
        if self.agg_mode == "delta":
            tags.append("delta")
        if self.staleness == "sweeps":
            tags.append("sw")
        if self.staleness_exp != 1.0:
            tags.append(f"a{self.staleness_exp:g}")
        if self.recluster == "lazy":
            tags.append("lazy")
        if self.batch_arrivals:
            tags.append("batch")
        if self.recluster_init == "warm":
            tags.append("warm")
        return base + "".join("+" + t for t in tags)


class _Slot:
    """State of one CSA-SFL cluster at the PS."""
    _ids = 0

    def __init__(self, members, D, start, t_ref):
        _Slot._ids += 1
        self.sid = _Slot._ids
        self.members = sorted(int(k) for k in members)
        self.D = float(D)
        self.start = start               # (w^{c,t_n}, w^{s,t_n}) the cluster trains from
        self.buf = start                 # buffered cluster model w_bar_n^b
        self.t_ref = int(t_ref)          # t_n
        self.began = 0.0
        self.done = 0.0
        self.token = 0


class Context:
    """Per-run state shared by the methods (data, partition, speeds, streams, init, engine)."""

    def __init__(self, cfg: RunConfig, data: dict, engine, log=print):
        self.cfg, self.eng, self.log = cfg, engine, log
        self.device = engine.device
        self.Xtr, self.ytr, self.Xte, self.yte = data["x_train"], data["y_train"], data["x_test"], data["y_test"]
        labels = data["y_train_np"]
        self.parts = dirichlet_partition(labels, cfg.K, cfg.delta, cfg.seed, cfg.min_per_device)
        self.sizes = np.array([len(p) for p in self.parts], dtype=np.int64)
        self.label_dists = label_histograms(labels, self.parts)
        self.speeds = device_speeds(cfg.K, cfg.sigma_s, cfg.seed)
        self.streams = [DeviceStream(p, cfg.seed * 100_003 + k, self.device) for k, p in enumerate(self.parts)]
        self.dims = Dims(engine.d_c, engine.d_s, activation_numel(engine.spec, cfg.cut))
        self.tk = np.array([train_time(n, cfg.E, cfg.B, cfg.t0, s) for n, s in zip(self.sizes, self.speeds)])
        self.nb_full = np.array([cfg.E * batches(int(n), cfg.B) for n in self.sizes])
        self.sp = SpeedProcess(self.speeds, cfg.sigma_s, cfg.speed_drift_rho, cfg.speed_drift_dt, cfg.seed)
        self.dyn = self.sp.dyn                                       # time-varying speeds?
        self.rng = np.random.RandomState(cfg.seed * 7919 + 2)          # clustering / orders
        self.dev0, self.srv0 = engine.init_params(cfg.seed)
        self.D_t = torch.tensor(self.sizes, dtype=torch.float32, device=self.device)
        self.wall0 = time.time()
        self.rows = []

    # ------------------------------------------------------------------
    # schedules + lock-step group training
    # ------------------------------------------------------------------

    def schedule(self, members, iters=None, caps=None):
        """Stack the members' batch schedules, longest first. Epoch mode (iters None): E
        epochs of each device's own data, optionally cut to the first caps[i] batches (CSA-SFL
        deadline mode); iteration mode: iters[i] full batches for member i.
        Returns (order, I (n,L,B), M (n,L,B), active counts per step)."""
        cfg = self.cfg
        if iters is None:
            sch = [self.streams[k].epochs(cfg.E, cfg.B) for k in members]
            if caps is not None:
                sch = [(I_[:int(c)], M_[:int(c)]) for (I_, M_), c in zip(sch, caps)]
        else:
            sch = [self.streams[k].iters(int(n), cfg.B) for k, n in zip(members, iters)]
        lens = np.array([s[0].shape[0] for s in sch])
        order = list(np.argsort(-lens, kind="stable"))
        L = int(lens.max())
        I = torch.zeros(len(members), L, cfg.B, dtype=torch.long, device=self.device)
        M = torch.zeros(len(members), L, cfg.B, device=self.device)
        for row, i in enumerate(order):
            n = lens[i]
            I[row, :n], M[row, :n] = sch[i]
        sl = lens[order]
        active = [int((sl > j).sum()) for j in range(L)]
        return order, I, M, active

    def run_group(self, dev, srv, kind, sched, wts=None, gidx=None, seq_order=None) -> float:
        """Lock-step co-training of a stacked group (rows already in schedule order); dev /
        srv are updated in place. kind: shared | per_device | grouped | sequential."""
        _, I, M, active = sched
        loss_sum = torch.zeros((), device=self.device)
        steps = 0
        for j, a in enumerate(active):
            idx = I[:a, j]
            X, Y, Mj = self.Xtr[idx], self.ytr[idx], M[:a, j]
            if kind == "shared":
                w = wts[:a] / wts[:a].sum()                  # |D_k| renormalized over running devices
                l = self.eng.step(dev[:a], srv, X, Y, Mj, "shared", wts=w)
            elif kind == "per_device":
                l = self.eng.step(dev[:a], srv[:a], X, Y, Mj, "per_device")
            elif kind == "grouped":
                l = self.eng.step(dev[:a], srv, X, Y, Mj, "grouped", wts=wts[:a], gidx=gidx[:a])
            else:
                l = self.eng.step_sequential_server(dev[:a], srv, X, Y, Mj, [p for p in seq_order if p < a])
            loss_sum += l.sum()
            steps += a
        return float(loss_sum) / max(steps, 1)

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------

    def record(self, r, sim_time, comm_bytes, round_bytes, dev, srv, train_loss, acc_loss=None, **extra) -> bool:
        """Evaluate (w^c, w^s) on the test set and append a row (acc_loss: the method's own (acc, loss), e.g. a
        mean over several models; dev / srv are then only checked for divergence). Returns False if diverged."""
        acc, loss = self.eng.evaluate(dev, srv, self.Xte, self.yte) if acc_loss is None else acc_loss
        ok = bool(torch.isfinite(dev).all() and torch.isfinite(srv).all()) and math.isfinite(loss)
        self.rows.append({"round": int(r), "sim_time_s": float(sim_time), "comm_mb": comm_bytes / 1e6,
                          "round_comm_mb": round_bytes / 1e6, "test_acc": acc, "test_loss": loss,
                          "train_loss": train_loss, "diverged": not ok, "wall_s": time.time() - self.wall0, **extra})
        if r % 25 == 0 or r == self.cfg.T or self.over(sim_time) or not ok:
            self.log(f"  [{METHOD_LABELS[self.cfg.method]}] round {r:4d} | t={sim_time:11.1f}s | "
                     f"comm={comm_bytes / 1e6:11.1f}MB | acc={acc:.4f} loss={loss:.4f} | "
                     f"train_loss={train_loss:.4f} | wall={time.time() - self.wall0:.0f}s")
        return ok

    def device_times(self, t_start: float) -> np.ndarray:
        """(K,) time each device needs for E local epochs started at t_start (= t_k with fixed speeds)."""
        if not self.dyn:
            return self.tk
        return np.array([self.sp.finish(k, t_start, self.nb_full[k] * self.cfg.t0) - t_start for k in range(self.cfg.K)])

    def finish_epochs(self, k: int, t_start: float) -> float:
        """Time device k completes E local epochs started at t_start."""
        if not self.dyn:
            return t_start + float(self.tk[k])
        return self.sp.finish(k, t_start, self.nb_full[k] * self.cfg.t0)

    def over(self, sim_time) -> bool:
        """Simulated-time budget reached (checked after recording a round)."""
        return self.cfg.time_budget > 0 and sim_time >= self.cfg.time_budget

    def wavg(self, mats, weights):
        """sum_i w_i * mats[i] for a list of flat vectors (or an (n, d) matrix)."""
        Wm = mats if torch.is_tensor(mats) else torch.stack(mats)
        w = torch.as_tensor(np.asarray(weights, dtype=np.float64) / float(np.sum(weights)),
                            dtype=torch.float32, device=self.device)
        return w @ Wm


# ======================================================================
# FL (FedAvg)
# ======================================================================

def run_fl(ctx: Context):
    cfg, K = ctx.cfg, ctx.cfg.K
    g_dev, g_srv = ctx.dev0, ctx.srv0
    now = comm = 0.0
    ctx.record(0, now, comm, 0.0, g_dev, g_srv, float("nan"))
    members = list(range(K))
    for r in range(cfg.T):
        sched = ctx.schedule(members)
        mem = [members[i] for i in sched[0]]
        dev, srv = g_dev.repeat(K, 1), g_srv.repeat(K, 1)
        loss = ctx.run_group(dev, srv, "per_device", sched)
        g_dev, g_srv = ctx.wavg(dev, ctx.sizes[mem]), ctx.wavg(srv, ctx.sizes[mem])
        dts = ctx.device_times(now)
        lat = float(dts.max())
        now += lat
        rb = K * ctx.dims.full_round()
        comm += rb
        if not ctx.record(r + 1, now, comm, rb, g_dev, g_srv, loss, round_latency_s=lat,
                          avg_wait_s=float(np.mean(lat - dts))) or ctx.over(now):
            break
    return ctx.rows


# ======================================================================
# CFL (clustered FL, Sattler et al., IEEE TNNLS 2021)
# ======================================================================

def cfl_bipartition(S: np.ndarray):
    """Paper Algorithm 1: the bipartition c1, c2 minimizing max_{i in c1, j in c2} S_ij (Eq. 33). Pairs are merged in
    order of decreasing similarity until two groups remain (single linkage). Returns (c1, c2) as sorted index lists."""
    n = len(S)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    groups = n
    for idx in np.argsort(-S, axis=None, kind="stable"):
        if groups == 2:
            break
        a, b = root(int(idx) // n), root(int(idx) % n)
        if a != b:
            parent[b] = a
            groups -= 1
    r0 = root(0)
    c1 = [i for i in range(n) if root(i) == r0]
    return c1, [i for i in range(n) if root(i) != r0]


def run_cfl(ctx: Context):
    """CFL (Sattler, Mueller, Samek, IEEE TNNLS 2021; Algorithm 5 with weight-updates, no permutation P) on the
    paper's system model. Every round all K devices train the FULL model of their cluster for E local epochs from
    the cluster model (as FL), Delta_k = w_k - w_c; each cluster aggregates |D_k|-weighted (Eq. 2), so CFL is FL
    until its first split. Split test per cluster c (Algorithm 5 l.14-19):
        ||Delta_c|| < eps1  and  max_{k in c} ||Delta_k|| > eps2,   Delta_c = sum_k |D_k|/|D_c| Delta_k,
        eps1 = cfl_eps1_frac * max_{rounds so far} ||Delta_c||  (paper: "a tenth of the maximum average update norm",
               the running maximum of the cluster, inherited from its parent),  eps2 = cfl_eps2_mult * eps1,
    then the cosine similarities alpha_ij of the members' updates (Eq. 40), the bipartition of Algorithm 1 (Eq. 33),
    and the split is kept if sqrt((1 - alpha_cross_max) / 2) > cfl_gamma_max (Eq. 36). A split takes effect in
    the same round: each child aggregates only its own members' updates (no work discarded) and keeps the parent's
    running maximum. Round time = the slowest device (all clusters train together, as FL); traffic = FL's (full
    model down + up per device). Reported test_acc / test_loss: |D_k|-weighted mean over devices of their cluster
    model's accuracy on the full test set; acc_personal: the same with each device's model scored on its own label
    distribution (per-class test accuracy weighted by the device's training-label shares)."""
    cfg, K = ctx.cfg, ctx.cfg.K
    sizes = ctx.sizes
    D_tot = float(sizes.sum())
    pk = ctx.label_dists / ctx.label_dists.sum(1, keepdims=True)              # (K, 10) training-label shares
    clusters = [list(range(K))]                                               # C
    models = [(ctx.dev0, ctx.srv0)]                                           # w_c
    peak = [0.0]                                                              # running max ||Delta_c||

    def evaluate():
        """(acc, loss) over devices, acc_personal, per-cluster accuracies."""
        accs, losses, pers = [], [], 0.0
        for C, (d, s_) in zip(clusters, models):
            a, l = ctx.eng.evaluate(d, s_, ctx.Xte, ctx.yte)
            pc = ctx.eng.evaluate_per_class(d, s_, ctx.Xte, ctx.yte)
            accs.append(a); losses.append(l)
            pers += float(sum(sizes[k] * (pk[k] @ np.nan_to_num(pc)) for k in C))
        w = np.array([sizes[C].sum() for C in clusters]) / D_tot
        return float(w @ accs), float(w @ losses), pers / D_tot, accs

    def info():
        return {"num_clusters": len(clusters), "cluster_sizes": "-".join(str(len(C)) for C in sorted(clusters, key=len, reverse=True))}

    trace = getattr(ctx, "trace", None)                                       # verification hook (checks.py)
    now = comm = 0.0
    acc, loss_t, pers, _ = evaluate()
    ctx.record(0, now, comm, 0.0, ctx.dev0, ctx.srv0, float("nan"), acc_personal=pers, **info())
    members = list(range(K))
    for r in range(cfg.T):
        sched = ctx.schedule(members)
        mem = [members[i] for i in sched[0]]
        row = {k: i for i, k in enumerate(mem)}
        cid = {k: c for c, C in enumerate(clusters) for k in C}
        if len(clusters) == 1:                                                # (= FL's g.repeat)
            dev, srv = models[0][0].repeat(K, 1), models[0][1].repeat(K, 1)
        else:
            dev = torch.stack([models[cid[k]][0] for k in mem])
            srv = torch.stack([models[cid[k]][1] for k in mem])
        dev_s, srv_s = dev.clone(), srv.clone()
        loss = ctx.run_group(dev, srv, "per_device", sched)
        dW = torch.cat([dev - dev_s, srv - srv_s], 1)                         # (K, d) weight updates (Eq. 1)
        dnorm = dW.norm(dim=1)
        del dev_s, srv_s
        new_clusters, new_models, new_peak, n_split = [], [], [], 0
        norms_mean, eps1_all, decisions = [], [], []
        for c, C in enumerate(clusters):
            rows = sorted(row[k] for k in C)                                  # schedule order (as FL's wavg)
            ks = [mem[i] for i in rows]
            p = torch.as_tensor(sizes[ks] / float(sizes[ks].sum()), dtype=torch.float32, device=ctx.device)
            mean_norm = float((p @ dW[rows]).norm())                          # ||Delta_c||
            max_norm = float(dnorm[rows].max())                               # max_k ||Delta_k||
            pk_c = max(peak[c], mean_norm)
            eps1 = cfg.cfl_eps1_frac * pk_c
            eps2 = cfg.cfl_eps2_mult * eps1
            norms_mean.append(mean_norm); eps1_all.append(eps1)
            parts, dec = [list(range(len(rows)))], {}
            if len(rows) >= 2 and mean_norm < eps1 and max_norm > eps2:
                U = dW[rows] / dnorm[rows].clamp_min(1e-30)[:, None]
                S = (U @ U.T).double().cpu().numpy()                          # alpha_ij (Eq. 40)
                c1, c2 = cfl_bipartition(S)
                a_cross = float(S[np.ix_(c1, c2)].max())
                if math.sqrt(max(0.0, 1.0 - a_cross) / 2.0) > cfg.cfl_gamma_max:
                    parts, n_split = [c1, c2], n_split + 1
                dec = {"S": S, "c1": [ks[i] for i in c1], "c2": [ks[i] for i in c2], "a_cross": a_cross}
            for part in parts:
                prow = [rows[i] for i in part]
                pks = [mem[i] for i in prow]
                new_clusters.append(sorted(pks))
                new_models.append((ctx.wavg(dev[prow], sizes[pks]), ctx.wavg(srv[prow], sizes[pks])))   # Eq. 2
                new_peak.append(pk_c)
            decisions.append({"members": ks, "mean_norm": mean_norm, "max_norm": max_norm, "peak_before": peak[c],
                              "eps1": eps1, "eps2": eps2, "split": len(parts) == 2, **dec})
        if trace is not None:
            trace.append({"r": r + 1, "mem": list(mem), "dev": dev.detach().cpu().clone(), "srv": srv.detach().cpu().clone(),
                          "decisions": decisions, "clusters_after": [list(C) for C in new_clusters],
                          "models_after": [(m[0].detach().cpu().clone(), m[1].detach().cpu().clone()) for m in new_models],
                          "peak_after": list(new_peak)})
        clusters, models, peak = new_clusters, new_models, new_peak
        del dev, srv, dW
        dts = ctx.device_times(now)
        lat = float(dts.max())
        now += lat
        rb = K * ctx.dims.full_round()
        comm += rb
        acc, loss_t, pers, _ = evaluate()
        if len(models) == 1:                                                  # one model: recorded exactly as FL
            dchk, schk, al = models[0][0], models[0][1], None
        else:                                                                 # several: mean over devices, all checked
            dchk, schk, al = torch.stack([m[0] for m in models]), torch.stack([m[1] for m in models]), (acc, loss_t)
        ok = ctx.record(r + 1, now, comm, rb, dchk, schk, loss, acc_loss=al,
                        round_latency_s=lat, avg_wait_s=float(np.mean(lat - dts)), acc_personal=pers, n_split=n_split,
                        cfl_mean_norm=max(norms_mean), cfl_max_norm=float(dnorm.max()), cfl_eps1=min(eps1_all), **info())
        if not ok or ctx.over(now):
            break
    return ctx.rows


# ======================================================================
# SFLv2 (sequential server side)
# ======================================================================

def run_sflv2(ctx: Context):
    cfg, K = ctx.cfg, ctx.cfg.K
    g_dev, g_srv = ctx.dev0, ctx.srv0.clone()
    now = comm = 0.0
    ctx.record(0, now, comm, 0.0, g_dev, g_srv, float("nan"))
    members = list(range(K))
    for r in range(cfg.T):
        perm = ctx.rng.permutation(K)                         # server-side processing order
        sched = ctx.schedule(members)
        mem = [members[i] for i in sched[0]]
        row_of = {k: i for i, k in enumerate(mem)}
        dev = g_dev.repeat(K, 1)
        loss = ctx.run_group(dev, g_srv, "sequential", sched, seq_order=[row_of[k] for k in perm])
        g_dev = ctx.wavg(dev, ctx.sizes[mem])
        dts = ctx.device_times(now)
        lat = float(dts.max())
        now += lat
        rb = sum(ctx.dims.split_round(cfg.E * n) for n in ctx.sizes)
        comm += rb
        if not ctx.record(r + 1, now, comm, rb, g_dev, g_srv, loss, round_latency_s=lat,
                          avg_wait_s=float(np.mean(lat - dts))) or ctx.over(now):
            break
    return ctx.rows


# ======================================================================
# SAFSL (semi-asynchronous, buffered)
# ======================================================================

def run_safsl(ctx: Context):
    cfg, K = ctx.cfg, ctx.cfg.K
    kbuf = max(1, int(round(cfg.safsl_buffer_frac * K)))
    algo = SAFSL(k=kbuf, gamma=cfg.safsl_gamma)
    g_dev, g_srv = ctx.dev0, ctx.srv0
    inflight, heap, seq = {}, [], [0]

    def dispatch(devs, t_now, tau, g_dev, g_srv):
        sched = ctx.schedule(devs)
        mem = [devs[i] for i in sched[0]]
        dev, srv = g_dev.repeat(len(mem), 1), g_srv.repeat(len(mem), 1)
        loss = ctx.run_group(dev, srv, "per_device", sched)
        for i, k in enumerate(mem):
            inflight[k] = (tau, dev[i], srv[i], loss)
            seq[0] += 1
            heapq.heappush(heap, (ctx.finish_epochs(k, t_now), seq[0], k))

    now = comm = 0.0
    ctx.record(0, now, comm, 0.0, g_dev, g_srv, float("nan"))
    dispatch(list(range(K)), 0.0, 0, g_dev, g_srv)
    for e in range(cfg.T):
        arr = [heapq.heappop(heap) for _ in range(min(kbuf, len(heap)))]
        prev = now
        now = max(a[0] for a in arr)
        ks = [a[2] for a in arr]
        stale = [e - inflight[k][0] for k in ks]
        w = [algo.participation_weight(int(ctx.sizes[k]), s) for k, s in zip(ks, stale)]
        g_dev = ctx.wavg([inflight[k][1] for k in ks], w)
        g_srv = ctx.wavg([inflight[k][2] for k in ks], w)
        loss = float(np.mean([inflight[k][3] for k in ks]))
        rb = sum(ctx.dims.split_round(cfg.E * int(ctx.sizes[k])) for k in ks)
        comm += rb
        for k in ks:
            del inflight[k]
        if not ctx.record(e + 1, now, comm, rb, g_dev, g_srv, loss, round_latency_s=now - prev,
                          staleness_max=max(stale), staleness_mean=float(np.mean(stale))) or ctx.over(now):
            break
        if e + 1 < cfg.T:
            dispatch(ks, now, e + 1, g_dev, g_srv)
    return ctx.rows


# ======================================================================
# ParallelSFL (clustered, synchronous)
# ======================================================================

class _ParallelSFLCompute(ParallelSFL):
    """ParallelSFL with the top worker picked by link capacity, ties (equal capacity in the
    paper's computation-only model) broken by the fastest top-submodel compute."""

    def _top_key(self, w: WorkerInfo):
        return (w.ingress_bw_hz, -w.mu_p)


def run_parallelsfl(ctx: Context):
    cfg, K = ctx.cfg, ctx.cfg.K
    tau_max = cfg.E * batches(float(np.mean(ctx.sizes)), cfg.B)
    algo = _ParallelSFLCompute(lam=cfg.psfl_lambda, tau_max=tau_max, max_cluster_size=cfg.psfl_max_cluster)
    macs = layer_macs(ctx.eng.spec)
    r_top = sum(macs[cfg.cut:]) / sum(macs[:cfg.cut])
    def worker_infos(sp):                       # per-worker timing from the speeds the round starts with
        return [WorkerInfo(client_id=k, label_dist=ctx.label_dists[k], num_samples=int(ctx.sizes[k]),
                           mu_b=cfg.t0 / sp[k], mu_p=r_top * cfg.t0 / sp[k],
                           beta_smashed=0.0, beta_full=0.0,
                           ingress_bw_hz=algo.b * cfg.psfl_max_cluster) for k in range(K)]
    infos = worker_infos(ctx.speeds)
    g_dev, g_srv = ctx.dev0, ctx.srv0
    now = comm = 0.0
    ctx.record(0, now, comm, 0.0, g_dev, g_srv, float("nan"))
    for r in range(cfg.T):
        if ctx.dyn:
            infos = worker_infos(ctx.sp.at(now))   # Algorithm 1 / Eq. 17 see the current speeds
        clusters = algo.cluster_workers(infos, ctx.rng)
        algo.assign_frequencies(clusters)
        bott = [(ci, w.client_id, c.tau, c.size) for ci, c in enumerate(clusters) for w in c.bottoms]
        sched = ctx.schedule([b[1] for b in bott], iters=[b[2] for b in bott])
        bs = [bott[i] for i in sched[0]]
        gidx = torch.tensor([b[0] for b in bs], device=ctx.device)
        wts = torch.tensor([1.0 / b[3] for b in bs], device=ctx.device)        # Eq. 3: 1/N_c
        dev, srv = g_dev.repeat(len(bs), 1), g_srv.repeat(len(clusters), 1)
        loss = ctx.run_group(dev, srv, "grouped", sched, wts=wts, gidx=gidx)
        bottom = [dev[gidx == ci].mean(0) for ci in range(len(clusters))]      # Eq. 4 (uniform)
        rho = algo.global_weights(clusters)                                     # Eq. 18: N_c tau_c
        g_dev, g_srv = ctx.wavg(bottom, rho), ctx.wavg(srv, rho)
        if not ctx.dyn:
            t_c = [c.tau * max(w.mu_b + w.beta_smashed + c.top.mu_p for w in c.bottoms) + c.top.beta_full
                   for c in clusters]                                           # Eq. 15
        else:                                       # Eq. 15 iteration by iteration, integrated over the drifting speeds
            t_c = []
            for c in clusters:
                tt = now
                for _ in range(c.tau):
                    tb = max(ctx.sp.finish(w.client_id, tt, cfg.t0) for w in c.bottoms)
                    tt = ctx.sp.finish(c.top.client_id, tb, r_top * cfg.t0)
                t_c.append(tt - now)
        lat = max(t_c)
        now += lat
        rb = sum(c.size * ctx.dims.split_round(c.tau * cfg.B) + ctx.dims.full_round() for c in clusters)
        comm += rb
        taus = [c.tau for c in clusters]
        if not ctx.record(r + 1, now, comm, rb, g_dev, g_srv, loss, round_latency_s=lat,
                          avg_wait_s=float(np.mean([lat - t for t in t_c])), num_clusters=len(clusters),
                          tau_min=min(taus), tau_max=max(taus)) or ctx.over(now):
            break
    return ctx.rows


# ======================================================================
# CSA-SFL
# ======================================================================

def run_csasfl(ctx: Context):
    """CSA-SFL (paper Algorithm 1) with the optional improvements of RunConfig (all off = paper):
      (1) sys_lambda > 0   clustering on gradient direction AND local-round time
                           (CSASFL.cluster_system_aware; z_k = standardized log of the device's round time)
      (2) local_work = deadline   global time slot L = deadline_q-quantile of all t_k: device k makes
                           n_k = min(E ceil(|D_k|/B), max(1, floor(L s_k / t_0))) mini-batch steps (the first
                           n_k of its E-epoch schedule), cluster time = max_k n_k t_0 / s_k; device sides are
                           combined FedNova-style  w^c = w^c_0 - tau_eff sum_k p_k (w^c_0 - w^c_k) / n_k,
                           p_k = |D_k|/|D_n|, tau_eff = sum_k p_k n_k  (= Eq. intra_agg when all n_k are equal)
          local_work = slot      the same work cap, and every cluster round lasts at least L:
                           cluster time = max(L, max_k n_k t_0 / s_k). All clusters then complete at the same
                           cadence (no flood of tiny fast-cluster rounds, no update-frequency bias)
          batch_arrivals = True  clusters that complete at the same simulated instant (the norm in slot mode)
                           are aggregated together as ONE update and all restart from the resulting model;
                           without it they are aggregated one after another and the first ones restart from
                           a partial global model (zero-duration intermediate states). Each arrival still
                           counts as one global round (t, H and staleness keep the paper's units).
      (3) agg_mode = delta  each cluster update applied once to the CURRENT global:
                           w^{t+1} = w^t + psi_n (w_bar_n - w^{t_n}),  psi_n = |D_n|/|D| * 1/(1+tau_n)
                           (uniform ablation: psi_n = 1/N); no buffer of old models
          staleness = sweeps   tau_n = (t - t_n) / N (in buffer or delta mode)
          staleness_exp = a    staleness discount (1+tau_n)^a instead of (1+tau_n) in phi / psi (a < 1: gentler)
      time-varying speeds (RunConfig.speed_drift_rho < 1, flsim.csasfl.system.SpeedProcess): every round's
                           member finish times are integrated over the drifting speeds from its start; a halt
                           pauses in-flight rounds (remaining work resumes at the speeds after the halt); the
                           system feature z_k of system-aware clustering uses the speeds at clustering time
      (4) recluster = lazy  every H rounds the PS re-clusters from the device updates it already holds
                           (g_k = (w^c_start - w^c_k)/n_k of the device's last round; the initial probe before
                           that) -- no halt, no probe traffic; an unchanged cluster keeps running, every other
                           in-flight cluster finishes its round normally (its update is aggregated) and then
                           dissolves; a new cluster starts from the current global as soon as all its members
                           are free. No work is ever discarded.
          recluster_init = warm  every re-clustering runs Lloyd's iterations from the PS's current partition instead
                           of a fresh k-means++ seeding: a cluster changes only when its members' gradients / round
                           times have moved, not because the seeding relabelled the partition (fewer affected
                           clusters, less discarded in-flight work); the first clustering is seeded as usual."""
    cfg, K, E, B, t0 = ctx.cfg, ctx.cfg.K, ctx.cfg.E, ctx.cfg.B, ctx.cfg.t0
    algo = CSASFL(num_clusters=cfg.N, recluster_every=cfg.H, local_iters=E, global_rounds=cfg.T,
                  clustering=cfg.clustering, aggregation=cfg.aggregation, staleness_exp=cfg.staleness_exp)
    dims, s, sizes = ctx.dims, ctx.speeds, ctx.sizes
    for opt, ok in (("local_work", ("epochs", "deadline", "slot")), ("agg_mode", ("buffer", "delta")),
                    ("staleness", ("rounds", "sweeps")), ("recluster", ("halt", "lazy")),
                    ("recluster_init", ("seed", "warm"))):
        if getattr(cfg, opt) not in ok:
            raise ValueError(f"{opt} must be one of {ok}, got {getattr(cfg, opt)!r}")
    deadline = cfg.local_work in ("deadline", "slot")          # local work capped by the slot L
    floor = cfg.local_work == "slot"                            # ... and rounds last at least L
    lazy = cfg.recluster == "lazy"
    nb_full = np.array([E * batches(int(n), B) for n in sizes])
    if deadline:
        L_slot = float(np.quantile(ctx.tk, cfg.deadline_q))
        # (+1e-9: a device whose t_k equals L exactly must keep all its batches despite float rounding)
        nb_cap = np.minimum(nb_full, np.maximum(1, np.floor(L_slot * s / t0 + 1e-9).astype(np.int64)))
        rt = nb_cap * t0 / s                                    # per-device round time
    else:
        L_slot, nb_cap, rt = None, nb_full, ctx.tk
    lt = np.log(rt)
    zsys = {k: float(v) for k, v in enumerate((lt - lt.mean()) / max(lt.std(), 1e-12))}
    dur = ((lambda mem: max(L_slot, float(max(rt[k] for k in mem)))) if floor
           else (lambda mem: float(max(rt[k] for k in mem))))
    D_of = lambda mem: float(sizes[list(mem)].sum())
    D_tot = float(sizes.sum())
    heap, seq = [], [0]
    last_dir = None                                             # (K, d_c) last update direction (lazy)
    dyn = ctx.dyn                                               # time-varying speeds (SpeedProcess)

    def zsys_at(t):
        """System feature z_k of the clustering: standardized log of each device's local-round time
        at time t (the PS observes the devices' recent round times); fixed speeds: zsys."""
        if not dyn:
            return zsys
        s_now = ctx.sp.at(t)
        n = (np.minimum(nb_full, np.maximum(1, np.floor(L_slot * s_now / t0 + 1e-9).astype(np.int64)))
             if deadline else nb_full)
        l = np.log(n * t0 / s_now)
        return {k: float(v) for k, v in enumerate((l - l.mean()) / max(l.std(), 1e-12))}

    def schedule(sl, t_start):
        if dyn:                                                 # exact per-member finish times under drifting speeds
            sl.began = sl.seg = sl.started = t_start           # began moves with halts, started does not
            sl.caps, sl.rem, sl.fin = {}, {}, {}
            for k in sl.members:
                n = int(nb_full[k])
                if deadline:                                    # steps that fit in the slot at the speeds it sees
                    n = min(n, max(1, int(math.floor(ctx.sp.work(k, t_start, t_start + L_slot) / t0 + 1e-9))))
                sl.caps[k], sl.rem[k] = n, n * t0
                sl.fin[k] = ctx.sp.finish(k, t_start, n * t0)
            sl.floor_end = t_start + L_slot if floor else t_start
            sl.done = max(max(sl.fin.values()), sl.floor_end)
        else:
            sl.began, sl.done = t_start, t_start + dur(sl.members)
        sl.token += 1
        seq[0] += 1
        heapq.heappush(heap, (sl.done, seq[0], sl.sid, sl.token))

    partition = None                                            # the PS's last clustering result

    def cluster(grads, t_now):
        nonlocal partition
        init = partition if cfg.recluster_init == "warm" else None
        if cfg.sys_lambda > 0:
            partition = algo.cluster_system_aware(grads, zsys_at(t_now), cfg.sys_lambda, ctx.rng, init=init)
        else:
            partition = algo.cluster_by_gradient(grads, ctx.rng, init=init)
        return partition

    def probe(owner, t_now):
        """Device-side gradients g_k of every device on its own current version (owner[k] =
        its cluster's (w^c, w^s)); returns ({k: g_k}, (K, d_c) tensor, halt time, bytes)."""
        P = [ctx.streams[k].probe(B) for k in range(K)]
        I, M = torch.stack([p[0] for p in P]), torch.stack([p[1] for p in P])
        dev2d = torch.stack([owner[k][0] for k in range(K)])
        srv2d = torch.stack([owner[k][1] for k in range(K)])
        G = ctx.eng.device_grads(dev2d, srv2d, ctx.Xtr[I], ctx.ytr[I], M)
        del dev2d, srv2d
        grads = {k: G[k].double().cpu().numpy() for k in range(K)}
        halt = (float(max(ctx.sp.finish(k, t_now, t0) - t_now for k in range(K))) if dyn
                else float(max(t0 / s[k] for k in range(K))))
        nbytes = sum(dims.probe(min(B, int(sizes[k]))) for k in range(K))
        return grads, G, halt, nbytes

    def train(sl):
        """One intra-cluster round from sl.start. Returns ((w^c_bar, w^s), loss, {k: samples})."""
        mem = sl.members
        capk = (lambda k: sl.caps[k]) if dyn else (lambda k: nb_cap[k])   # mini-batch steps of this round
        sched = ctx.schedule(mem, caps=[capk(k) for k in mem] if deadline else None)
        m_sorted = [mem[i] for i in sched[0]]
        Dk = ctx.D_t[m_sorted]
        dev = sl.start[0].repeat(len(mem), 1)
        srv = sl.start[1].clone()
        loss = ctx.run_group(dev, srv, "shared", sched, wts=Dk)            # Eq. server_update / device_update
        steps = torch.tensor([float(capk(k)) for k in m_sorted], device=ctx.device)
        if deadline:                                                       # FedNova-normalized Eq. intra_agg
            p = Dk / Dk.sum()
            d_k = (sl.start[0][None, :] - dev) / steps[:, None]
            dev_bar = sl.start[0] - (p * steps).sum() * (p @ d_k)
            samples = sched[2].sum(dim=(1, 2)).tolist()
            samp = {k: float(n) for k, n in zip(m_sorted, samples)}
        else:
            dev_bar = (Dk / Dk.sum()) @ dev                                # Eq. intra_agg
            samp = {k: E * int(sizes[k]) for k in mem}
        if lazy:                                                           # g_k for the next lazy re-clustering
            last_dir[m_sorted] = (sl.start[0][None, :] - dev) / steps[:, None]
        return (dev_bar, srv), loss, samp                                  # Eq. form

    def stale_of(x, t):
        return (t - x.t_ref) / cfg.N if cfg.staleness == "sweeps" else t - x.t_ref

    g = (ctx.dev0, ctx.srv0)
    comm = K * dims.device_model()                       # every device downloads w^{c,0}
    now = 0.0
    if algo.clustering == "gradient":
        grads, G0, h0, b0 = probe({k: g for k in range(K)}, now)
        clusters = cluster(grads, now)
        now, comm = now + h0, comm + b0
        if lazy:
            last_dir = G0.clone()
    elif algo.clustering == "system":                     # ablation: speed tiers only, one-time
        clusters = algo.system_clusters(zsys_at(now), ctx.rng)
    else:
        clusters = algo.random_clusters(range(K), ctx.rng)
    slots = {}
    for C in clusters:
        sl = _Slot(C, D_of(C), g, 0)
        slots[sl.sid] = sl
        schedule(sl, now)
    ctx.record(0, now, comm, comm, g[0], g[1], float("nan"), num_clusters=len(slots),
               slot_s=L_slot if deadline else np.nan)

    pending, idle = [], set()                            # lazy re-clustering state
    trace = getattr(ctx, "trace", None)                  # verification hook (flsim.csasfl.checks)
    skipped = 0
    t = 0
    prev = now
    while t < cfg.T and heap:
        when, _, sid, tok = heapq.heappop(heap)
        sl = slots.get(sid)
        if sl is None or tok != sl.token:
            skipped += 1
            continue                                     # event of a cluster dissolved / rescheduled
        now = when
        arr = [sl]                                       # clusters completing at this instant
        if cfg.batch_arrivals:                           # (2b) all of them form ONE aggregation, and all restart
            while heap and heap[0][0] <= now:            #      from the resulting model (never from a partial one)
                _, _, sid2, tok2 = heapq.heappop(heap)
                x2 = slots.get(sid2)
                if x2 is None or tok2 != x2.token:
                    skipped += 1
                    continue
                arr.append(x2)
        m = len(arr)
        arr_ids = {x.sid for x in arr}
        g_before = g
        res = [train(x) for x in arr]                    # (upd, loss, samples) per arrival
        order = list(slots.values())
        stale = [stale_of(x, t) for x in order]
        idx = [order.index(x) for x in arr]
        if cfg.agg_mode == "buffer":
            for x, r in zip(arr, res):
                x.buf = r[0]
            phi = algo.agg_weights([x.D for x in order], stale)                                 # Eq. phi
            g = (ctx.wavg([x.buf[0] for x in order], phi), ctx.wavg([x.buf[1] for x in order], phi))  # Eq. global_update
            ws = [float(phi[i]) for i in idx]
        else:                                                    # (3) each update applied once
            disc = ((lambda tau: 1.0 + tau) if cfg.staleness_exp == 1.0 else (lambda tau: (1.0 + tau) ** cfg.staleness_exp))
            ws = [(1.0 / len(order) if algo.aggregation == "uniform" else x.D / D_tot / disc(stale[i]))
                  for x, i in zip(arr, idx)]
            g0, g1 = g
            for x, r, w in zip(arr, res, ws):
                g0, g1 = g0 + w * (r[0][0] - x.start[0]), g1 + w * (r[0][1] - x.start[1])
            g = (g0, g1)
            for x, r in zip(arr, res):
                x.buf = r[0]
        taus = [float(stale[i]) for i in idx]
        if trace is not None:
            trace.append({"t": t, "now": now, "g_before": g_before, "g_after": g, "upd": res[0][0], "start": arr[0].start,
                          "w_n": ws[0], "D_n": arr[0].D, "tau_n": taus[0], "members": list(arr[0].members),
                          "bufs": [x.buf for x in order], "phi": (list(phi) if cfg.agg_mode == "buffer" else None),
                          "stale": list(stale), "samp": dict(res[0][2]), "began": arr[0].began, "done": arr[0].done,
                          "batch": [{"sid": x.sid, "members": list(x.members), "upd": r[0], "start": x.start, "w": w,
                                     "D": x.D, "tau": tau, "samp": dict(r[2]), "began": x.began, "done": x.done,
                                     "started": getattr(x, "started", None), "fin": dict(getattr(x, "fin", {})),
                                     "caps": dict(getattr(x, "caps", {}))}
                                    for x, r, w, tau in zip(arr, res, ws, taus)]})
        wait = (float(np.mean([x.done - x.fin[k] for x in arr for k in x.members])) if dyn            # straggler wait
                else float(np.mean([x.done - x.began - rt[k] for x in arr for k in x.members])))
        rb = sum(sum(dims.split_round(r[2][k]) for k in x.members) for x, r in zip(arr, res))  # exchange + upload + broadcast
        busy = {k for x in arr for k in x.members}       # devices that just received the new model
        t_new = t + m
        for x in arr:
            x.start, x.t_ref = g, t_new
        n_aff, h = 0, 0.0
        recl =(algo.clustering == "gradient" and any(r % algo.H == 0 for r in range(t + 1, t_new + 1))
                and t_new < cfg.T)
        if recl and not lazy:
            owner = {k: x.start for x in slots.values() for k in x.members}
            grads, _, h, pb = probe(owner, now)
            rb += pb
            new_clusters = cluster(grads, now)
            old = {tuple(x.members): x for x in slots.values()}
            new_slots, kept = {}, set()
            for C in new_clusters:
                x = old.get(tuple(sorted(int(k) for k in C)))
                if x is not None:                         # unaffected: resume its current state
                    kept.add(x.sid)
                    if x.sid in arr_ids:
                        schedule(x, now + h)
                    elif dyn:                                   # paused for h, then resumes at the speeds after the halt
                        for k in x.members:
                            if x.fin[k] <= now:                 # already done: its finish moves with the paused round
                                x.rem[k], x.fin[k] = 0.0, x.fin[k] + h
                            else:
                                x.rem[k] = max(0.0, x.rem[k] - ctx.sp.work(k, x.seg, now))
                                x.fin[k] = ctx.sp.finish(k, now + h, x.rem[k])
                        x.seg, x.floor_end, x.began = now + h, x.floor_end + h, x.began + h
                        x.done, x.token = max(max(x.fin.values()), x.floor_end), x.token + 1
                        seq[0] += 1
                        heapq.heappush(heap, (x.done, seq[0], x.sid, x.token))
                    else:
                        x.began, x.done, x.token = x.began + h, x.done + h, x.token + 1
                        seq[0] += 1
                        heapq.heappush(heap, (x.done, seq[0], x.sid, x.token))
                else:                                     # affected: restart from w^{t+1}
                    x = _Slot(C, D_of(C), g, t_new)
                    schedule(x, now + h)
                    # devices of the just-completed cluster(s) already hold w^{c,t+1} (broadcast above)
                    rb += sum(1 for k in x.members if k not in busy) * dims.device_model()
                    n_aff += 1
                new_slots[x.sid] = x
            for x in slots.values():                      # partial exchange of interrupted rounds
                if x.sid not in kept and x.sid not in arr_ids:
                    el = max(0.0, now - x.began)
                    for k in x.members:
                        if dyn:
                            wdone = x.caps[k] * t0 - x.rem[k] + ctx.sp.work(k, x.seg, now)
                            nb = min(int(x.caps[k]), int(math.floor(wdone / t0 + 1e-9)))
                        else:
                            nb = min(int(nb_cap[k]), int(math.floor(el * s[k] / t0)))
                        rb += dims.smashed(min(E * int(sizes[k]), nb * B))
            slots = new_slots
        else:
            if recl:                                      # (4) lazy: new partition from the held updates
                grads = {k: last_dir[k].double().cpu().numpy() for k in range(K)}
                new_clusters = cluster(grads, now)
                by_members = {tuple(x.members): x for x in slots.values()}
                kept, pending = set(), []
                for C in new_clusters:
                    key = tuple(sorted(int(k) for k in C))
                    if key in by_members:
                        kept.add(by_members[key].sid)
                    else:
                        pending.append(key)
                for x in slots.values():
                    x.dissolve = x.sid not in kept
                n_aff = len(pending)
            for x in arr:
                if getattr(x, "dissolve", False):         # finished its round under an old partition
                    del slots[x.sid]
                    idle |= set(x.members)
                else:
                    schedule(x, now)
            for C in list(pending):                       # start every new cluster whose members are free
                if all(k in idle for k in C):
                    x = _Slot(C, D_of(C), g, t_new)
                    slots[x.sid] = x
                    schedule(x, now)
                    idle -= set(C)
                    pending.remove(C)
                    rb += sum(1 for k in C if k not in busy) * dims.device_model()
        comm += rb
        if trace is not None:
            trace[-1].update(active=[(x.sid, tuple(x.members), bool(getattr(x, "dissolve", False)), x.began, x.done)
                                     for x in slots.values()], idle=set(idle), pending=list(pending),
                             skipped=skipped, rb=rb, recl=recl, halt=h,
                             restart_from_final=all(x.start[0] is g[0] and x.start[1] is g[1] for x in arr))
        ok = ctx.record(t_new, now, comm, rb, g[0], g[1], float(np.mean([r[1] for r in res])),
                        round_latency_s=now - prev,
                        cluster_id=arr[0].sid, cluster_size=sum(len(x.members) for x in arr),
                        staleness=float(np.mean(taus)), phi=float(sum(ws)),
                        staleness_mean=float(np.mean(stale)), staleness_max=float(max(stale)),
                        cluster_time_s=(max(x.done - x.began for x in arr) if dyn else max(dur(x.members) for x in arr)),
                        avg_wait_s=wait, recluster=int(recl), n_affected=n_aff, num_clusters=len(slots),
                        pending=len(pending), idle_devices=len(idle), batch=m)
        prev = now
        t = t_new
        if not ok or ctx.over(now):
            break
    return ctx.rows


RUNNERS = {"fl": run_fl, "cfl": run_cfl, "sflv2": run_sflv2, "safsl": run_safsl, "parallelsfl": run_parallelsfl,
           "csasfl": run_csasfl}


def run(cfg: RunConfig, data: dict, engine, log=print) -> list:
    ctx = Context(cfg, data, engine, log)
    log(f"[{METHOD_LABELS[cfg.method]}] delta={cfg.delta:g} seed={cfg.seed} K={cfg.K} E={cfg.E} B={cfg.B} "
        f"lr={cfg.lr:g} T<={cfg.T} budget={cfg.time_budget:g}s | t0={cfg.t0:g}s sigma_s={cfg.sigma_s:g} | |D_k| min/med/max = "
        f"{ctx.sizes.min()}/{int(np.median(ctx.sizes))}/{ctx.sizes.max()} | s_k min/med/max = "
        f"{ctx.speeds.min():.3g}/{np.median(ctx.speeds):.3g}/{ctx.speeds.max():.3g} | t_k min/med/max = "
        f"{ctx.tk.min():.1f}/{np.median(ctx.tk):.1f}/{ctx.tk.max():.1f}s"
        + (f" | N={cfg.N} H={cfg.H} variant={cfg.variant()}" if cfg.method == "csasfl" else "")
        + f" | engine={engine.engine}")
    return RUNNERS[cfg.method](ctx)
