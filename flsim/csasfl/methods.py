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
from flsim.csasfl.system import Dims, batches, device_speeds, train_time
from flsim.interfaces.parallel_sfl_algorithm import WorkerInfo

METHOD_LABELS = {"fl": "FL", "sflv2": "SFLv2", "safsl": "SAFSL", "parallelsfl": "ParallelSFL",
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
    clustering: str = "gradient"     # gradient | random (ablation)
    aggregation: str = "weighted"    # weighted | uniform (ablation)
    # SAFSL
    safsl_buffer_frac: float = 0.8   # aggregate the 0.8K earliest arrivals
    safsl_gamma: float = 1.0
    # ParallelSFL
    psfl_lambda: float = 0.5
    psfl_max_cluster: int = 8

    def variant(self) -> str:
        if self.method != "csasfl":
            return "-"
        return {("gradient", "weighted"): "full", ("gradient", "uniform"): "uniform_agg",
                ("random", "weighted"): "random_cluster"}.get((self.clustering, self.aggregation),
                                                               f"{self.clustering}_{self.aggregation}")


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
        self.rng = np.random.RandomState(cfg.seed * 7919 + 2)          # clustering / orders
        self.dev0, self.srv0 = engine.init_params(cfg.seed)
        self.D_t = torch.tensor(self.sizes, dtype=torch.float32, device=self.device)
        self.wall0 = time.time()
        self.rows = []

    # ------------------------------------------------------------------
    # schedules + lock-step group training
    # ------------------------------------------------------------------

    def schedule(self, members, iters=None):
        """Stack the members' batch schedules, longest first. Epoch mode (iters None): E
        epochs of each device's own data; iteration mode: iters[i] full batches for member i.
        Returns (order, I (n,L,B), M (n,L,B), active counts per step)."""
        cfg = self.cfg
        if iters is None:
            sch = [self.streams[k].epochs(cfg.E, cfg.B) for k in members]
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

    def record(self, r, sim_time, comm_bytes, round_bytes, dev, srv, train_loss, **extra) -> bool:
        """Evaluate (w^c, w^s) on the test set and append a row. Returns False if diverged."""
        acc, loss = self.eng.evaluate(dev, srv, self.Xte, self.yte)
        ok = bool(torch.isfinite(dev).all() and torch.isfinite(srv).all()) and math.isfinite(loss)
        self.rows.append({"round": int(r), "sim_time_s": float(sim_time), "comm_mb": comm_bytes / 1e6,
                          "round_comm_mb": round_bytes / 1e6, "test_acc": acc, "test_loss": loss,
                          "train_loss": train_loss, "diverged": not ok, "wall_s": time.time() - self.wall0, **extra})
        if r % 25 == 0 or r == self.cfg.T or self.over(sim_time) or not ok:
            self.log(f"  [{METHOD_LABELS[self.cfg.method]}] round {r:4d} | t={sim_time:11.1f}s | "
                     f"comm={comm_bytes / 1e6:11.1f}MB | acc={acc:.4f} loss={loss:.4f} | "
                     f"train_loss={train_loss:.4f} | wall={time.time() - self.wall0:.0f}s")
        return ok

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
        lat = float(ctx.tk.max())
        now += lat
        rb = K * ctx.dims.full_round()
        comm += rb
        if not ctx.record(r + 1, now, comm, rb, g_dev, g_srv, loss, round_latency_s=lat,
                          avg_wait_s=float(np.mean(lat - ctx.tk))) or ctx.over(now):
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
        lat = float(ctx.tk.max())
        now += lat
        rb = sum(ctx.dims.split_round(cfg.E * n) for n in ctx.sizes)
        comm += rb
        if not ctx.record(r + 1, now, comm, rb, g_dev, g_srv, loss, round_latency_s=lat,
                          avg_wait_s=float(np.mean(lat - ctx.tk))) or ctx.over(now):
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
            heapq.heappush(heap, (t_now + float(ctx.tk[k]), seq[0], k))

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
    infos = [WorkerInfo(client_id=k, label_dist=ctx.label_dists[k], num_samples=int(ctx.sizes[k]),
                        mu_b=cfg.t0 / ctx.speeds[k], mu_p=r_top * cfg.t0 / ctx.speeds[k],
                        beta_smashed=0.0, beta_full=0.0,
                        ingress_bw_hz=algo.b * cfg.psfl_max_cluster) for k in range(K)]
    g_dev, g_srv = ctx.dev0, ctx.srv0
    now = comm = 0.0
    ctx.record(0, now, comm, 0.0, g_dev, g_srv, float("nan"))
    for r in range(cfg.T):
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
        t_c = [c.tau * max(w.mu_b + w.beta_smashed + c.top.mu_p for w in c.bottoms) + c.top.beta_full
               for c in clusters]                                               # Eq. 15
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
    cfg, K, E, B, t0 = ctx.cfg, ctx.cfg.K, ctx.cfg.E, ctx.cfg.B, ctx.cfg.t0
    algo = CSASFL(num_clusters=cfg.N, recluster_every=cfg.H, local_iters=E, global_rounds=cfg.T,
                  clustering=cfg.clustering, aggregation=cfg.aggregation)
    dims, s, sizes = ctx.dims, ctx.speeds, ctx.sizes
    dur = lambda mem: float(max(ctx.tk[k] for k in mem))
    D_of = lambda mem: float(sizes[list(mem)].sum())
    heap, seq = [], [0]

    def schedule(sl, t_start):
        sl.began, sl.done = t_start, t_start + dur(sl.members)
        sl.token += 1
        seq[0] += 1
        heapq.heappush(heap, (sl.done, seq[0], sl.sid, sl.token))

    def probe(owner):
        """Device-side gradients g_k of every device on its own current version (owner[k] =
        its cluster's (w^c, w^s)); returns ({k: g_k}, halt time, bytes)."""
        P = [ctx.streams[k].probe(B) for k in range(K)]
        I, M = torch.stack([p[0] for p in P]), torch.stack([p[1] for p in P])
        dev2d = torch.stack([owner[k][0] for k in range(K)])
        srv2d = torch.stack([owner[k][1] for k in range(K)])
        G = ctx.eng.device_grads(dev2d, srv2d, ctx.Xtr[I], ctx.ytr[I], M)
        del dev2d, srv2d
        grads = {k: G[k].double().cpu().numpy() for k in range(K)}
        halt = float(max(t0 / s[k] for k in range(K)))
        nbytes = sum(dims.probe(min(B, int(sizes[k]))) for k in range(K))
        return grads, halt, nbytes

    def train(sl):
        mem = sl.members
        sched = ctx.schedule(mem)
        m_sorted = [mem[i] for i in sched[0]]
        Dk = ctx.D_t[m_sorted]
        dev = sl.start[0].repeat(len(mem), 1)
        srv = sl.start[1].clone()
        loss = ctx.run_group(dev, srv, "shared", sched, wts=Dk)            # Eq. server_update / device_update
        dev_bar = (Dk / Dk.sum()) @ dev                                    # Eq. intra_agg
        return (dev_bar, srv), loss                                        # Eq. form

    g = (ctx.dev0, ctx.srv0)
    comm = K * dims.device_model()                       # every device downloads w^{c,0}
    now = 0.0
    if algo.clustering == "gradient":
        grads, h0, b0 = probe({k: g for k in range(K)})
        clusters = algo.cluster_by_gradient(grads, ctx.rng)
        now, comm = now + h0, comm + b0
    else:
        clusters = algo.random_clusters(range(K), ctx.rng)
    slots = {}
    for C in clusters:
        sl = _Slot(C, D_of(C), g, 0)
        slots[sl.sid] = sl
        schedule(sl, now)
    ctx.record(0, now, comm, comm, g[0], g[1], float("nan"), num_clusters=len(slots))

    t = 0
    prev = now
    while t < cfg.T and heap:
        when, _, sid, tok = heapq.heappop(heap)
        sl = slots.get(sid)
        if sl is None or tok != sl.token:
            continue                                     # event of a cluster dissolved / rescheduled
        now = when
        sl.buf, loss = train(sl)
        order = list(slots.values())
        stale = [t - x.t_ref for x in order]
        phi = algo.agg_weights([x.D for x in order], stale)                                 # Eq. phi
        g = (ctx.wavg([x.buf[0] for x in order], phi), ctx.wavg([x.buf[1] for x in order], phi))  # Eq. global_update
        i_n = order.index(sl)
        tau_n, phi_n = int(stale[i_n]), float(phi[i_n])
        wait = float(np.mean([sl.done - sl.began - ctx.tk[k] for k in sl.members]))     # intra-cluster straggler wait
        rb = sum(dims.split_round(E * int(sizes[k])) for k in sl.members)              # its exchange + upload + broadcast
        sl.start, sl.t_ref = g, t + 1
        n_aff = 0
        recl = algo.clustering == "gradient" and (t + 1) % algo.H == 0 and t + 1 < cfg.T
        if recl:
            owner = {k: x.start for x in slots.values() for k in x.members}
            grads, h, pb = probe(owner)
            rb += pb
            new_clusters = algo.cluster_by_gradient(grads, ctx.rng)
            old = {tuple(x.members): x for x in slots.values()}
            new_slots, kept = {}, set()
            for C in new_clusters:
                x = old.get(tuple(sorted(int(k) for k in C)))
                if x is not None:                         # unaffected: resume its current state
                    kept.add(x.sid)
                    if x is sl:
                        schedule(x, now + h)
                    else:
                        x.began, x.done, x.token = x.began + h, x.done + h, x.token + 1
                        seq[0] += 1
                        heapq.heappush(heap, (x.done, seq[0], x.sid, x.token))
                else:                                     # affected: restart from w^{t+1}
                    x = _Slot(C, D_of(C), g, t + 1)
                    schedule(x, now + h)
                    # devices of the just-completed cluster already hold w^{c,t+1} (broadcast above)
                    rb += sum(1 for k in x.members if k not in sl.members) * dims.device_model()
                    n_aff += 1
                new_slots[x.sid] = x
            for x in slots.values():                      # partial exchange of interrupted rounds
                if x.sid not in kept and x is not sl:
                    el = max(0.0, now - x.began)
                    for k in x.members:
                        nb = min(E * batches(int(sizes[k]), B), int(math.floor(el * s[k] / t0)))
                        rb += dims.smashed(min(E * int(sizes[k]), nb * B))
            slots = new_slots
        else:
            schedule(sl, now)
        comm += rb
        ok = ctx.record(t + 1, now, comm, rb, g[0], g[1], loss, round_latency_s=now - prev,
                        cluster_id=sl.sid, cluster_size=len(sl.members), staleness=tau_n, phi=phi_n,
                        staleness_mean=float(np.mean(stale)), staleness_max=int(max(stale)),
                        cluster_time_s=dur(sl.members),
                        avg_wait_s=wait, recluster=int(recl), n_affected=n_aff, num_clusters=len(slots))
        prev = now
        t += 1
        if not ok or ctx.over(now):
            break
    return ctx.rows


RUNNERS = {"fl": run_fl, "sflv2": run_sflv2, "safsl": run_safsl, "parallelsfl": run_parallelsfl,
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
