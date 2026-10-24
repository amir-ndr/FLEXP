"""
flsim/csasfl/checks.py: correctness checks for the CSA-SFL paper setup (run before GPU jobs):

    python -m flsim.csasfl.checks          (CPU, ~1-2 min)

  1 model       12 layers, 1,402,474 parameters (5.61 MB), split 8: d_c = 286,432, smashed 6,272;
                functional forward == nn.Module forward
  2 relay       one engine step == the split relay of flsim.core.split_client (detach +
                requires_grad, server backward, smashed.backward(relay.grad)) on nn.Modules
  3 server_update  the "shared" step uses w^s - eta sum_k |D_k| g_k^s / |D_n| (explicit sum)
  4 engines     vmap == loop for every step kind, SFLv2's sequential server, the probe gradients
  5 phi         Eq. phi normalized, decreasing in staleness, uniform ablation = 1/N
  6 k-means     recovers planted gradient groups; the returned objective uses mu_n = mean g_k
  7 partition   disjoint cover of the training set, >= min_per_device, delta controls skew
  8 runs        tiny CSA-SFL / SAFSL / FL / SFLv2 / ParallelSFL runs: event-loop invariants
                (time non-decreasing, staleness, re-clustering every H, comm formulas)
  9 variants    (python -m flsim.csasfl.checks --variants, ~10 min) the four CSA-SFL improvement
                options, each formula / invariant verified exactly through run_csasfl's trace hook
 10 drift       (python -m flsim.csasfl.checks --drift, ~10 min) time-varying speeds: the AR(1)
                process keeps LogNormal(0, sigma^2) with lag-1 correlation rho and is deterministic;
                the time-varying code path with frozen speeds reproduces the fixed-speed runs of every
                method; with drifting speeds every method's timing (FL / SFLv2 rounds, SAFSL arrivals,
                ParallelSFL Eq. 15, CSA-SFL rounds, caps, halts, paused rounds) matches an independent
                re-simulation
 11 cfl         (python -m flsim.csasfl.checks --cfl, ~3 min) the CFL baseline (Sattler et al. 2021): Algorithm 1 gives
                the exact optimum of Eq. 33 (brute force); without splits CFL is bitwise FL (fixed and drifting speeds);
                with forced splits every decision (eps1 / eps2 rule, Eq. 36), child model (Eq. 2), the reported
                accuracies, the clock and the traffic are recomputed independently
"""

import math
import sys

import numpy as np
import torch
import torch.nn as nn

from flsim.algorithms.csa_sfl import CSASFL
from flsim.csasfl.data import dirichlet_partition, label_histograms, load_mnist
from flsim.csasfl.engine import _HAS_FUNC, Engine
from flsim.csasfl.methods import RunConfig, run
from flsim.csasfl.model import CUT_LAYER, LeNet12, activation_numel, forward_layers, layer_spec
from flsim.csasfl.system import Dims, train_time

_fails = []


def check(name, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _fails.append(name)


def _batch(a=3, B=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(a, B, 1, 28, 28, generator=g)
    Y = torch.randint(0, 10, (a, B), generator=g)
    M = torch.ones(a, B)
    M[-1, B // 2:] = 0.0                                   # a short last batch
    return X, Y, M


def main():
    torch.set_num_threads(min(8, torch.get_num_threads()))
    spec, cpu = layer_spec(), torch.device("cpu")
    print("1 model")
    m = LeNet12()
    n = sum(p.numel() for p in m.parameters())
    eng = Engine(spec, CUT_LAYER, 0.01, cpu, "vmap" if _HAS_FUNC else "loop")
    check("12 ordered layers", len(m.ordered_layers()) == 12)
    check("1,402,474 parameters (5.61 MB)", n == 1_402_474, f"{n}, {n * 4 / 1e6:.2f} MB")
    check("device side 286,432 / smashed 6,272", eng.d_c == 286_432 and activation_numel(spec, 8) == 6272,
          f"d_c={eng.d_c}, act={activation_numel(spec, 8)}")
    x = torch.randn(5, 1, 28, 28)
    params = list(m.parameters())
    nd = len(eng.sd)
    f = forward_layers(spec, params[nd:], forward_layers(spec, params[:nd], x, 0, 8), 8)
    check("functional forward == nn.Module", torch.allclose(f, m(x), atol=1e-5))

    print("2 relay (flsim.core.split_client mechanics) == engine step")
    dev0, srv0 = eng.init_params(3)
    X, Y, M = _batch(1, 32)
    M[:] = 1.0
    full = LeNet12()
    flat = torch.cat([dev0, srv0])
    off = 0
    with torch.no_grad():
        for p in full.parameters():
            p.copy_(flat[off:off + p.numel()].view_as(p))
            off += p.numel()
    cm = nn.Sequential(*full.ordered_layers()[:8])
    sm = nn.Sequential(*full.ordered_layers()[8:])
    co, so = torch.optim.SGD(cm.parameters(), lr=0.01), torch.optim.SGD(sm.parameters(), lr=0.01)
    smashed = cm(X[0])
    relay = smashed.detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(sm(relay), Y[0])
    loss.backward()
    so.step()
    smashed.backward(relay.grad)
    co.step()
    d2, s2 = dev0[None].clone(), srv0.clone()
    eng.step(d2, s2, X, Y, M, "shared", wts=torch.ones(1))
    rd = torch.cat([p.detach().reshape(-1) for p in cm.parameters()])
    rs = torch.cat([p.detach().reshape(-1) for p in sm.parameters()])
    check("device side identical", torch.allclose(d2[0], rd, atol=1e-6), f"{(d2[0] - rd).abs().max():.2e}")
    check("server side identical", torch.allclose(s2, rs, atol=1e-6), f"{(s2 - rs).abs().max():.2e}")

    print("3 Eq. server_update")
    X, Y, M = _batch(3, 16, seed=1)
    D = torch.tensor([100.0, 300.0, 600.0])
    w = D / D.sum()
    d2, s2 = dev0.repeat(3, 1), srv0.clone()
    ref = Engine(spec, CUT_LAYER, 0.01, cpu, "loop")
    gd, gs, _ = ref._loop_grads(d2, s2, X, Y, M, "shared", None)
    eng.step(d2, s2, X, Y, M, "shared", wts=w)
    expect = srv0 - 0.01 * sum(w[k] * gs[k] for k in range(3))
    check("w^s <- w^s - eta sum |D_k| g_k^s / |D_n|", torch.allclose(s2, expect, atol=1e-6))
    check("w_k^c <- w_k^c - eta g_k^c", torch.allclose(d2, dev0 - 0.01 * gd, atol=1e-6))

    print("4 engines (vmap == loop)")
    if not _HAS_FUNC:
        print("  [skip] torch.func unavailable (torch < 2.0): loop engine only")
    else:
        X, Y, M = _batch(4, 16, seed=2)
        gidx, wts = torch.tensor([0, 1, 0, 1]), torch.tensor([0.5, 0.25, 0.5, 0.75])
        for kind in ("shared", "per_device", "grouped"):
            out = []
            for e in (eng, ref):
                d2 = dev0.repeat(4, 1)
                s2 = {"shared": srv0.clone(), "per_device": srv0.repeat(4, 1), "grouped": srv0.repeat(2, 1)}[kind]
                for _ in range(2):
                    e.step(d2, s2, X, Y, M, kind, wts=wts if kind != "per_device" else None, gidx=gidx)
                out.append((d2, s2))
            err = max((out[0][0] - out[1][0]).abs().max(), (out[0][1] - out[1][1]).abs().max())
            check(f"{kind} step", err < 1e-5, f"max diff {err:.2e}")
        out = []
        for e in (eng, ref):
            d2, s2 = dev0.repeat(4, 1), srv0.clone()
            for _ in range(2):
                e.step_sequential_server(d2, s2, X, Y, M, [3, 1, 0, 2])
            out.append((d2, s2))
        err = max((out[0][0] - out[1][0]).abs().max(), (out[0][1] - out[1][1]).abs().max())
        check("SFLv2 sequential server", err < 1e-5, f"max diff {err:.2e}")
        g1 = eng.device_grads(dev0.repeat(4, 1), srv0.repeat(4, 1), X, Y, M)
        g2 = ref.device_grads(dev0.repeat(4, 1), srv0.repeat(4, 1), X, Y, M)
        check("probe gradients", (g1 - g2).abs().max() < 1e-6, f"max diff {(g1 - g2).abs().max():.2e}")

    print("5 phi")
    a = CSASFL(num_clusters=3)
    phi = a.agg_weights([100, 200, 300], [0, 1, 3])
    raw = np.array([100 / 1, 200 / 2, 300 / 4])
    check("phi = |D_n|/(1+tau) normalized", np.allclose(phi, raw / raw.sum()) and abs(phi.sum() - 1) < 1e-12)
    check("phi decreasing in staleness", a.agg_weights([1, 1], [0, 5])[0] > a.agg_weights([1, 1], [0, 5])[1])
    check("uniform ablation = 1/N", np.allclose(CSASFL(num_clusters=3, aggregation="uniform").agg_weights([1, 5, 9], [0, 2, 4]), 1 / 3))

    print("6 k-means")
    rng = np.random.RandomState(0)
    dirs = rng.randn(4, 200)
    grads, truth = {}, {}
    for k in range(40):
        c = k % 4
        grads[k] = dirs[c] * rng.uniform(0.5, 5.0) + 0.2 * rng.randn(200)
        truth[k] = c
    a = CSASFL(num_clusters=4)
    cl = a.cluster_by_gradient(grads, np.random.RandomState(1))
    pure = all(len({truth[k] for k in c}) == 1 for c in cl) and len(cl) == 4
    check("planted groups recovered", pure, f"sizes {[len(c) for c in cl]}")
    obj = 0.0
    for c in cl:
        G = np.stack([grads[k] for k in c])
        mu = G.mean(0)
        obj += sum(1 - g @ mu / (np.linalg.norm(g) * np.linalg.norm(mu)) for g in G)
    check("objective with mu_n = mean g_k", abs(obj - a.last_objective) < 1e-8, f"{obj:.6f} vs {a.last_objective:.6f}")

    print("7 partition")
    data = load_mnist(device=cpu)
    y = data["y_train_np"]
    for delta in (0.1, 10.0):
        parts = dirichlet_partition(y, 50, delta, seed=1, min_per_device=10)
        allidx = np.concatenate(parts)
        H = label_histograms(y, parts)
        check(f"delta={delta:g}: disjoint cover, min >= 10", len(allidx) == len(y) == len(np.unique(allidx))
              and min(len(p) for p in parts) >= 10,
              f"sizes min/med/max {min(map(len, parts))}/{int(np.median(list(map(len, parts))))}/{max(map(len, parts))}, "
              f"mean top-class share {H.max(1).mean():.2f}")
    p1 = label_histograms(y, dirichlet_partition(y, 50, 0.1, seed=1)).max(1).mean()
    p10 = label_histograms(y, dirichlet_partition(y, 50, 10.0, seed=1)).max(1).mean()
    check("delta=0.1 far more skewed than delta=10", p1 > 0.6 and p10 < 0.3, f"{p1:.2f} vs {p10:.2f}")
    check("t_k = E ceil(|D_k|/B) t0/s_k", math.isclose(train_time(130, 5, 64, 0.2, 0.5), 5 * 3 * 0.2 / 0.5))

    print("8 tiny runs (event-loop invariants)")
    small = {"x_train": data["x_train"][:2400], "y_train": data["y_train"][:2400], "y_train_np": y[:2400],
             "x_test": data["x_test"][:300], "y_test": data["y_test"][:300]}
    quiet = lambda *_: None
    base = dict(delta=0.1, seed=2, K=8, E=1, B=32, lr=0.01, T=12, time_budget=0.0)
    dims = Dims(eng.d_c, eng.d_s, 6272)
    rows = run(RunConfig(method="csasfl", N=3, H=4, **base), small, Engine(spec, 8, 0.01, cpu, eng.engine), quiet)
    tt = [r["sim_time_s"] for r in rows]
    check("CSA-SFL: T rounds, time non-decreasing", len(rows) == 13 and all(b >= a for a, b in zip(tt, tt[1:])))
    rec = [r["round"] for r in rows if r.get("recluster") == 1]
    check("CSA-SFL: re-clustering at rounds H, 2H (t+1 < T)", rec == [4, 8], f"{rec}")
    check("CSA-SFL: phi in (0,1], staleness >= 0",
          all(0 < r["phi"] <= 1 and r["staleness"] >= 0 for r in rows[1:]))
    check("CSA-SFL: <= N clusters", all(r["num_clusters"] <= 3 for r in rows))
    rows = run(RunConfig(method="fl", **base), small, Engine(spec, 8, 0.01, cpu, eng.engine), quiet)
    check("FL: comm = rounds * K * 2 d * 4 B",
          math.isclose(rows[-1]["comm_mb"] * 1e6, 12 * 8 * dims.full_round(), rel_tol=1e-9))
    rows = run(RunConfig(method="sflv2", **base), small, Engine(spec, 8, 0.01, cpu, eng.engine), quiet)
    n_tr = 2400
    check("SFLv2: comm = rounds * sum_k (2 d_c + E|D_k|(2a+1)) * 4 B",
          math.isclose(rows[-1]["comm_mb"] * 1e6, 12 * (8 * 2 * eng.d_c + n_tr * (2 * 6272 + 1)) * 4, rel_tol=1e-9))
    rows = run(RunConfig(method="safsl", **base), small, Engine(spec, 8, 0.01, cpu, eng.engine), quiet)
    tt = [r["sim_time_s"] for r in rows]
    check("SAFSL: time non-decreasing, staleness >= 0",
          all(b >= a for a, b in zip(tt, tt[1:])) and all(r["staleness_max"] >= 0 for r in rows[1:]))
    rows = run(RunConfig(method="parallelsfl", **base), small, Engine(spec, 8, 0.01, cpu, eng.engine), quiet)
    check("ParallelSFL: T rounds, tau in [1, tau_max]",
          len(rows) == 13 and all(1 <= r["tau_min"] <= r["tau_max"] for r in rows[1:]))

    if "--variants" in sys.argv:
        check_variants(small, spec, cpu)
    if "--drift" in sys.argv:
        check_drift(small, spec, cpu)
    if "--cfl" in sys.argv:
        check_cfl(small, spec, cpu)
    print(f"\n{'ALL CHECKS PASSED' if not _fails else f'{len(_fails)} FAILED: {_fails}'}")
    return 1 if _fails else 0


def check_variants(small, spec, cpu):
    """9 CSA-SFL improvement options (RunConfig sys_lambda / local_work / agg_mode / staleness /
    recluster), verified exactly on small runs through the trace hook of run_csasfl."""
    from flsim.csasfl.methods import Context, run_csasfl
    from flsim.csasfl.system import batches
    print("9 CSA-SFL improvements (--variants)")
    base = dict(method="csasfl", delta=0.1, seed=2, K=12, E=2, B=32, lr=0.01, T=24, time_budget=0.0, N=3, H=4)

    def go(**kw):
        cfg = RunConfig(**{**base, **kw})
        ctx = Context(cfg, small, Engine(spec, 8, 0.01, cpu, "vmap" if _HAS_FUNC else "loop"), log=lambda *a: None)
        ctx.trace = []
        return cfg, ctx, run_csasfl(ctx), ctx.trace

    close = lambda a, b, tol=1e-5: float((a - b).abs().max()) <= tol * max(1.0, float(b.abs().max()))
    # (1) system-aware clustering
    r0 = np.random.RandomState(0)
    gr = {k: r0.randn(300) + (k % 3) * r0.randn(300) for k in range(30)}
    z = {k: float(r0.randn()) for k in range(30)}
    c1 = CSASFL(num_clusters=4).cluster_by_gradient(gr, np.random.RandomState(5))
    c2 = CSASFL(num_clusters=4).cluster_system_aware(gr, z, 0.0, np.random.RandomState(5))
    check("(1) lam = 0 reproduces the paper's clustering", sorted(map(sorted, c1)) == sorted(map(sorted, c2)))
    c3 = CSASFL(num_clusters=4).cluster_system_aware(gr, z, 1e6, np.random.RandomState(5))
    spans = sorted((min(z[k] for k in c), max(z[k] for k in c)) for c in c3)
    check("(1) lam -> inf gives speed tiers", all(spans[i][1] < spans[i + 1][0] for i in range(len(spans) - 1)))
    # (4b) warm-started re-clustering
    dirs = r0.randn(4, 300)
    gp = {k: dirs[k % 4] * r0.uniform(0.5, 3.0) + 0.3 * r0.randn(300) for k in range(40)}
    zp = {k: 0.1 * float(r0.randn()) for k in range(40)}
    planted = [[k for k in range(40) if k % 4 == n] for n in range(4)]
    a4 = CSASFL(num_clusters=4)
    w1 = a4.cluster_by_gradient(gp, np.random.RandomState(9), init=planted)
    w2 = a4.cluster_system_aware(gp, zp, 1.0, np.random.RandomState(9), init=planted)
    check("(4b) warm start from a well-separated partition keeps it (gradient / system-aware)",
          sorted(map(sorted, w1)) == planted and sorted(map(sorted, w2)) == planted)
    noisy = {k: v + 2.0 * r0.randn(300) for k, v in gr.items()}
    p0 = CSASFL(num_clusters=4).cluster_system_aware(gr, z, 1.0, np.random.RandomState(5))
    obj = lambda P, g: sum(sum((1 - g[k] @ mu / (np.linalg.norm(g[k]) * np.linalg.norm(mu))) + (z[k] - cz) ** 2 for k in C)
                           for C in P for mu, cz in [(np.mean([g[k] for k in C], 0), np.mean([z[k] for k in C]))])
    a5 = CSASFL(num_clusters=4)
    p1 = a5.cluster_system_aware(noisy, z, 1.0, np.random.RandomState(5), init=p0)
    check("(4b) warm start never ends worse than the partition it starts from", obj(p1, noisy) <= obj(p0, noisy) + 1e-9
          and math.isclose(a5.last_objective, obj(p1, noisy), rel_tol=1e-9), f"{obj(p0, noisy):.3f} -> {obj(p1, noisy):.3f}")
    # (2) deadline local work
    cfg, ctx, rows, tr = go(local_work="deadline", deadline_q=0.5)
    L, sp = float(np.quantile(ctx.tk, 0.5)), ctx.speeds
    ncap = {k: min(2 * batches(int(ctx.sizes[k]), 32), max(1, int(math.floor(L * sp[k] / 0.2 + 1e-9)))) for k in range(12)}
    check("(2) cluster time = max_k n_k t0/s_k (n_k = steps that fit in the slot, >= 1, <= E epochs)",
          all(math.isclose(e["done"] - e["began"], max(ncap[k] * 0.2 / sp[k] for k in e["members"]), rel_tol=1e-9) for e in tr))
    cfg, ctx, rows, tr = go(local_work="slot", deadline_q=0.5)
    check("(2) slot: same work cap, and cluster time = max(L, max_k n_k t0/s_k) (every round lasts >= L)",
          all(math.isclose(e["done"] - e["began"], max(L, max(ncap[k] * 0.2 / sp[k] for k in e["members"])), rel_tol=1e-9)
              for e in tr) and min(e["done"] - e["began"] for e in tr) >= L - 1e-9)
    p = torch.rand(5); p /= p.sum(); w0 = torch.randn(20); W = torch.randn(5, 20); st = torch.full((5,), 7.0)
    check("(2) FedNova combination = Eq. intra_agg when all step counts are equal",
          close(w0 - (p * st).sum() * (p @ ((w0[None] - W) / st[:, None])), p @ W, 1e-6))
    _, _, ra, _ = go(local_work="deadline", deadline_q=1.0, H=1000)
    _, _, rb, _ = go(H=1000)
    check("(2) deadline q=1 (nobody cut), no re-clustering: times and traffic = paper",
          len(ra) == len(rb) and all(math.isclose(a["sim_time_s"], b["sim_time_s"], rel_tol=1e-12)
                                     and math.isclose(a["comm_mb"], b["comm_mb"], rel_tol=1e-12) for a, b in zip(ra, rb)))
    # (3) update-based aggregation + sweeps staleness
    cfg, ctx, rows, tr = go(agg_mode="delta", staleness="sweeps")
    Dt = float(ctx.sizes.sum())
    check("(3) w^(t+1) = w^t + psi_n (w_bar_n - w^(t_n)) at every round",
          all(close(e["g_after"][i], e["g_before"][i] + e["w_n"] * (e["upd"][i] - e["start"][i])) for e in tr for i in (0, 1)))
    check("(3) psi_n = |D_n|/|D| / (1 + (t - t_n)/N)",
          all(math.isclose(e["w_n"], e["D_n"] / Dt / (1 + e["tau_n"]), rel_tol=1e-12) for e in tr)
          and all(abs(e["tau_n"] * 3 - round(e["tau_n"] * 3)) < 1e-12 for e in tr))
    cfg, ctx, rows, tr = go(staleness="sweeps")
    check("(3) buffer mode with sweeps staleness keeps Eq. global_update: w = sum phi_m b_m",
          all(close(e["g_after"][0], sum(f * bb[0] for f, bb in zip(e["phi"], e["bufs"]))) for e in tr))
    # (2b) simultaneous arrivals aggregated together (slot mode)
    for agg in ("buffer", "delta"):
        cfg, ctx, rows, tr = go(local_work="slot", deadline_q=0.5, batch_arrivals=True, agg_mode=agg,
                                staleness="sweeps" if agg == "delta" else "rounds", T=40, H=12)
        sizes = [len(e["batch"]) for e in tr]
        same_instant = all(len({b["done"] for b in e["batch"]}) == 1 for e in tr)
        distinct = all(b["now"] > a["now"] for a, b in zip(tr, tr[1:]))
        if agg == "buffer":
            agg_ok = all(close(e["g_after"][0], sum(f * bb[0] for f, bb in zip(e["phi"], e["bufs"]))) for e in tr)
        else:
            agg_ok = all(close(e["g_after"][i], e["g_before"][i] + sum(b["w"] * (b["upd"][i] - b["start"][i]) for b in e["batch"]))
                         for e in tr for i in (0, 1))
        rounds_ok = [r["round"] for r in rows[1:]] == list(np.cumsum(sizes)[:len(rows) - 1])
        check(f"(2b) [{agg}] one aggregation per instant (batches up to {max(sizes)} clusters), distinct instants",
              same_instant and distinct and max(sizes) > 1)
        check(f"(2b) [{agg}] the new model uses every fresh cluster model at once", agg_ok)
        check(f"(2b) [{agg}] every arrival restarts from the final model; each arrival = one global round",
              all(e["restart_from_final"] for e in tr) and rounds_ok)
    # (4) lazy re-clustering, alone and with everything else
    for kw in ({}, {"agg_mode": "delta", "staleness": "sweeps", "sys_lambda": 1.0, "local_work": "slot"},
               {"local_work": "slot", "batch_arrivals": True}, {"sys_lambda": 1.0}):
        cfg, ctx, rows, tr = go(recluster="lazy", T=40, **kw)
        tag = "lazy" + ("+" + "+".join(f"{k}={v}" for k, v in kw.items()) if kw else "")
        part = all(len(m) == len(set(m)) and set(m).isdisjoint(e["idle"]) and set(m) | e["idle"] == set(range(12))
                   for e in tr for m in [[x for _, ms, _, _, _ in e["active"] for x in ms]])
        wait_ok = all(set(x for C in e["pending"] for x in C) <= (e["idle"] | {x for _, ms, dz, _, _ in e["active"] if dz for x in ms})
                      for e in tr)
        dm = ctx.dims.device_model()
        extra = [e["rb"] - sum(ctx.dims.split_round(b["samp"][k]) for b in e["batch"] for k in b["members"]) for e in tr]
        done = set(k for e in tr for b in e["batch"] for k in b["members"])
        busy_end = set(m for _, ms, _, _, _ in tr[-1]["active"] for m in ms)
        check(f"(4) [{tag}] no work discarded, partition consistent, pending waits only for free/finishing devices",
              tr[-1]["skipped"] == 0 and part and wait_ok)
        check(f"(4) [{tag}] no halts and no probe traffic after the start; every device trained or is training",
              all(x >= -1e-6 and abs(x / dm - round(x / dm)) < 1e-9 for x in extra)
              and sum(r["recluster"] for r in rows[1:]) > 0 and done | busy_end == set(range(12)))


def check_cfl(small, spec, cpu):
    """11 CFL baseline."""
    import itertools
    from flsim.csasfl.methods import RUNNERS, Context, cfl_bipartition
    print("11 CFL (--cfl)")
    r0 = np.random.RandomState(0)
    ok = True
    for _ in range(200):                                   # Algorithm 1 == brute-force argmin of Eq. 33
        n = int(r0.randint(2, 10))
        A = r0.randn(n, 6) + (r0.randint(0, 3, n)[:, None] * r0.randn(1, 6))
        U = A / np.linalg.norm(A, axis=1, keepdims=True)
        S = U @ U.T
        c1, c2 = cfl_bipartition(S)
        best = min(S[np.ix_([i for i in range(n) if m >> i & 1], [i for i in range(n) if not m >> i & 1])].max()
                   for m in range(1, 2 ** n - 1) if m & 1)
        ok &= sorted(c1 + c2) == list(range(n)) and len(c1) > 0 and len(c2) > 0 and abs(S[np.ix_(c1, c2)].max() - best) < 1e-12
    check("Algorithm 1 bipartition = exact argmin_{c1,c2} max_{i in c1, j in c2} alpha_ij (Eq. 33), 200 random cases", ok)

    quiet = lambda *_: None
    eng = lambda: Engine(spec, 8, 0.01, cpu, "vmap" if _HAS_FUNC else "loop")
    base = dict(delta=0.1, seed=2, K=10, E=1, B=32, lr=0.01, time_budget=0.0, T=8)
    eq = lambda u, v: u == v or (isinstance(u, float) and isinstance(v, float) and math.isnan(u) and math.isnan(v))

    def go(method, **kw):
        ctx = Context(RunConfig(method=method, **{**base, **kw}), small, eng(), quiet)
        ctx.trace = []
        return ctx, RUNNERS[method](ctx)
    for kw, tag in (({}, "fixed speeds"), ({"speed_drift_rho": 0.6, "speed_drift_dt": 5.0}, "drifting speeds")):
        _, rf = go("fl", **kw)
        _, rc = go("cfl", cfl_eps1_frac=0.0, **kw)            # eps1 = 0: never splits
        same = len(rf) == len(rc) and all(eq(a[k], b[k]) for a, b in zip(rf, rc) for k in a if k != "wall_s")
        check(f"CFL without a split == FL bitwise ({tag}, {len(rc)} rows)", same and all(r["num_clusters"] == 1 for r in rc))

    # forced splits: eps1 = running max (split as soon as the mean update shrinks), eps2 = 0, gamma_max = 0
    ctx, rows = go("cfl", cfl_eps1_frac=1.0, cfl_eps2_mult=0.0, T=10)
    _, rf = go("fl", T=10)
    sizes, D = ctx.sizes, float(ctx.sizes.sum())
    pk = ctx.label_dists / ctx.label_dists.sum(1, keepdims=True)
    models = {k: (ctx.dev0, ctx.srv0) for k in range(ctx.cfg.K)}     # each device's cluster model
    peaks = {k: 0.0 for k in range(ctx.cfg.K)}
    dec_ok = mod_ok = part_ok = acc_ok = True
    n_splits = 0
    for e, row in zip(ctx.trace, rows[1:]):
        mem = e["mem"]
        start = torch.stack([torch.cat(models[k]) for k in mem])
        trained = torch.cat([e["dev"], e["srv"]], 1)
        dW = trained - start
        for d in e["decisions"]:
            ks = d["members"]
            rows_ = [mem.index(k) for k in ks]
            p = torch.tensor(sizes[ks] / sizes[ks].sum(), dtype=torch.float32)
            mn = float((p @ dW[rows_]).norm())
            mx = float(dW[rows_].norm(dim=1).max())
            pkc = max(peaks[ks[0]], mn)
            want = len(ks) >= 2 and mn < 1.0 * pkc and mx > 0.0
            dec_ok &= (abs(d["mean_norm"] - mn) <= 1e-5 * max(1, mn) and abs(d["eps1"] - pkc) <= 1e-5 * max(1, pkc)
                       and d["split"] == want and d["peak_before"] == peaks[ks[0]])
            if d["split"]:
                n_splits += 1
                U = dW[rows_] / dW[rows_].norm(dim=1, keepdim=True)
                S = (U @ U.T).double().numpy()
                c1, c2 = cfl_bipartition(S)
                dec_ok &= np.allclose(S, d["S"], atol=1e-6) and sorted(d["c1"]) == sorted(ks[i] for i in c1) \
                    and sorted(d["c2"]) == sorted(ks[i] for i in c2)
        after = e["clusters_after"]
        part_ok &= sorted(k for C in after for k in C) == list(range(ctx.cfg.K))
        new = {}
        for C, (md, ms), pkk in zip(after, e["models_after"], e["peak_after"]):
            w = torch.tensor(sizes[C] / sizes[C].sum(), dtype=torch.float32)
            idx = [mem.index(k) for k in C]
            mod_ok &= bool(torch.allclose(md, w @ e["dev"][idx], atol=1e-6) and torch.allclose(ms, w @ e["srv"][idx], atol=1e-6))
            for k in C:
                new[k] = (md, ms)
                peaks[k] = pkk
        models = new
        accs, pers = 0.0, 0.0
        for C, (md, ms) in zip(after, e["models_after"]):
            a, _ = ctx.eng.evaluate(md, ms, ctx.Xte, ctx.yte)
            pc = ctx.eng.evaluate_per_class(md, ms, ctx.Xte, ctx.yte)
            accs += sizes[C].sum() / D * a
            pers += sum(sizes[k] * (pk[k] @ pc) for k in C) / D
        acc_ok &= abs(row["test_acc"] - accs) < 1e-9 and abs(row["acc_personal"] - pers) < 1e-9 and row["num_clusters"] == len(after)
    check("forced splits: split iff ||Delta_c|| < eps1 and max_k ||Delta_k|| > eps2 (eps1 = frac * running max), "
          "cosine matrix and Algorithm 1 recomputed", dec_ok and n_splits > 0, f"{n_splits} splits, final clusters {rows[-1]['cluster_sizes']}")
    check("forced splits: clusters partition the devices; every cluster model = |D_k|-weighted mean of its members (Eq. 2)",
          part_ok and mod_ok)
    check("forced splits: test_acc = sum_c |D_c|/|D| acc(w_c) on the full test set; acc_personal on each device's labels", acc_ok)
    check("CFL clock and traffic = FL's (slowest device per round, full model down + up per device)",
          all(math.isclose(a["sim_time_s"], b["sim_time_s"], rel_tol=1e-12) and math.isclose(a["comm_mb"], b["comm_mb"], rel_tol=1e-12)
              for a, b in zip(rows, rf)))


class _RefClock:
    """Independent integrator of the piecewise-constant speeds of a SpeedProcess: cumulative work
    W_k(t) tabulated at the interval boundaries and inverted directly (SpeedProcess.finish / .work
    walk interval by interval instead). Built on a fresh process of the same seed."""

    def __init__(self, sp):
        self.sp, self.dt = sp, sp.dt
        self.S = np.zeros((0, sp.K))

    def _grow(self, J):
        if len(self.S) <= J + 1:
            self.S = np.exp(np.stack([self.sp._log(j) for j in range(J + 64)]))
            self.C = np.vstack([np.zeros((1, self.sp.K)), np.cumsum(self.S * self.dt, axis=0)])

    def W(self, k, t):
        J = int(np.floor(t / self.dt))
        self._grow(J)
        return self.C[J, k] + (t - J * self.dt) * self.S[J, k]

    def work(self, k, a, b):
        return self.W(k, b) - self.W(k, a)

    def finish(self, k, a, w):
        target, J = self.W(k, a) + w, int(np.floor(a / self.dt))
        while True:
            self._grow(J + 1)
            if self.C[J + 1, k] > target:
                return max(a, J * self.dt + (target - self.C[J, k]) / self.S[J, k])
            J += 1


def check_drift(small, spec, cpu):
    """10 time-varying device speeds (RunConfig.speed_drift_rho < 1)."""
    import heapq
    from flsim.csasfl.methods import RUNNERS, Context, _ParallelSFLCompute
    from flsim.csasfl.model import layer_macs
    from flsim.csasfl.system import SpeedProcess, batches
    from flsim.interfaces.parallel_sfl_algorithm import WorkerInfo
    print("10 time-varying speeds (--drift)")
    rel = lambda a, b, tol: abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    # the process itself
    sp = SpeedProcess(np.random.RandomState(0).lognormal(0.0, 2.0, 4000), 2.0, 0.9, 10.0, 7)
    L = np.stack([np.log(sp.at(10.0 * j + 1.0)) for j in range(150)])            # (150, 4000)
    corr = float(np.mean([np.corrcoef(L[j], L[j + 1])[0, 1] for j in range(149)]))
    check("AR(1) on log s_k keeps LogNormal(0, sigma^2), lag-1 correlation rho",
          abs(L.mean()) < 0.03 and abs(L.std() - 2.0) < 0.03 and abs(corr - 0.9) < 0.01,
          f"mean {L.mean():.3f}, std {L.std():.3f}, corr {corr:.3f}")
    a = SpeedProcess(np.ones(5), 2.0, 0.7, 3.0, 11)
    b = SpeedProcess(np.ones(5), 2.0, 0.7, 3.0, 11)
    late = a.at(500.0).copy()
    check("deterministic per seed, independent of query order",
          all(np.array_equal(a.at(3.0 * j), b.at(3.0 * j)) for j in range(200)) and np.array_equal(late, b.at(500.0)))
    ref, r0 = _RefClock(SpeedProcess(np.ones(5), 2.0, 0.7, 3.0, 11)), np.random.RandomState(1)
    qs = [(int(r0.randint(5)), float(r0.uniform(0, 400)), float(r0.exponential(20.0))) for _ in range(300)]
    check("finish / work == independent integrator, finish inverts work",
          all(rel(a.finish(k, t, w), ref.finish(k, t, w), 1e-9) and rel(a.work(k, t, t + w), ref.work(k, t, t + w), 1e-9)
              and rel(a.work(k, t, a.finish(k, t, w)), w, 1e-9) for k, t, w in qs))
    st = SpeedProcess(np.array([0.5, 2.0]), 2.0, 1.0, 100.0, 3)
    check("rho = 1: fixed speeds, t + work / s_k", not st.dyn and st.finish(0, 7.0, 3.0) == 7.0 + 3.0 / 0.5
          and np.allclose(st.at(1e5), [0.5, 2.0]))

    quiet = lambda *_: None
    eng = lambda: Engine(spec, 8, 0.01, cpu, "vmap" if _HAS_FUNC else "loop")
    base = dict(delta=0.1, seed=2, K=10, E=1, B=32, lr=0.01, time_budget=0.0, N=3, H=4)
    csa = {"paper": {}, "sys": {"sys_lambda": 1.0}, "system": {"clustering": "system"},
           "uniform": {"aggregation": "uniform", "sys_lambda": 1.0},
           "slot": {"local_work": "slot", "deadline_q": 0.5},
           "slot+batch+lazy": {"local_work": "slot", "deadline_q": 0.25, "batch_arrivals": True, "recluster": "lazy"},
           "sys+warm H=2": {"sys_lambda": 1.0, "recluster_init": "warm", "H": 2},       # kept clusters: paused, resumed
           "slot+warm H=2": {"local_work": "slot", "deadline_q": 0.5, "recluster_init": "warm", "H": 2}}
    cases = [(m, {}, 10) for m in ("fl", "sflv2", "safsl", "parallelsfl")] + [("csasfl", kw, 20) for kw in csa.values()]
    tags = {id(kw): n for n, kw in csa.items()}

    def go(method, kw, T, rho=1.0, dt=100.0, frozen_dt=None):
        cfg = RunConfig(method=method, T=T, speed_drift_rho=rho, speed_drift_dt=dt, **{**base, **kw})
        ctx = Context(cfg, small, eng(), quiet)
        if frozen_dt is not None:                       # the time-varying code path, speeds frozen at s_k(0)
            ctx.sp = SpeedProcess(ctx.speeds, cfg.sigma_s, 1.0, frozen_dt, cfg.seed)
            ctx.sp.dyn = ctx.dyn = True
        ctx.trace = []
        return cfg, ctx, RUNNERS[method](ctx)

    # (a) frozen speeds through the time-varying code: identical training, same clocks / traffic
    timing = ("sim_time_s", "comm_mb", "round_comm_mb", "round_latency_s", "avg_wait_s", "cluster_time_s",
              "staleness", "staleness_mean", "staleness_max", "phi")
    eq = lambda u, v: u == v or (isinstance(u, float) and isinstance(v, float) and math.isnan(u) and math.isnan(v))
    for m, kw, T in cases:
        _, _, ra = go(m, kw, T)
        _, _, rb = go(m, kw, T, frozen_dt=3.7)
        same = len(ra) == len(rb) and all(eq(x[c], y[c]) for x, y in zip(ra, rb) for c in ("test_acc", "test_loss", "train_loss"))
        worst = max((abs(x[c] - y[c]) / max(1.0, abs(x[c])) for x, y in zip(ra, rb) for c in timing
                     if c in x and isinstance(x[c], float) and not math.isnan(x[c])), default=0.0)
        rest = all(eq(x[c], y[c]) for x, y in zip(ra, rb) for c in x if c not in timing and c not in ("wall_s", "cluster_id"))
        check(f"(a) frozen drift == fixed speeds: {m + (' ' + tags[id(kw)] if m == 'csasfl' else '')}",
              same and rest and worst < 1e-9, f"{len(rb)} rows, max rel. timing diff {worst:.1e}")

    # (b) drifting speeds: every method's clock vs an independent re-simulation (_RefClock)
    rho, dt = 0.6, 5.0
    for m, kw, T in cases:
        cfg, ctx, rows = go(m, kw, T, rho, dt)
        ref = _RefClock(SpeedProcess(ctx.speeds, cfg.sigma_s, rho, dt, cfg.seed))
        K, t0, nbf = cfg.K, cfg.t0, ctx.nb_full
        tt = [r["sim_time_s"] for r in rows]
        mono = all(y >= x for x, y in zip(tt, tt[1:]))
        name = m + (" " + tags[id(kw)] if m == "csasfl" else "")
        if m in ("fl", "sflv2"):
            t, ok = 0.0, True
            for r in rows[1:]:
                d = [ref.finish(k, t, nbf[k] * t0) - t for k in range(K)]
                t += max(d)
                ok &= rel(r["round_latency_s"], max(d), 1e-7) and rel(r["sim_time_s"], t, 1e-7) \
                    and rel(r["avg_wait_s"], float(np.mean(max(d) - np.array(d))), 1e-6)
            check(f"(b) {name}: round = max_k (finish of E epochs from the round start)", ok and mono)
        elif m == "safsl":
            kbuf = max(1, int(round(cfg.safsl_buffer_frac * K)))
            heap = [(ref.finish(k, 0.0, nbf[k] * t0), k) for k in range(K)]
            heapq.heapify(heap)
            ver, ok = {k: 0 for k in range(K)}, True
            for e, r in enumerate(rows[1:]):
                arr = [heapq.heappop(heap) for _ in range(min(kbuf, len(heap)))]
                now = max(x[0] for x in arr)
                stale = [e - ver[x[1]] for x in arr]
                ok &= rel(r["sim_time_s"], now, 1e-7) and r["staleness_max"] == max(stale)
                for _, k in arr:
                    ver[k] = e + 1
                    heapq.heappush(heap, (ref.finish(k, now, nbf[k] * t0), k))
            check(f"(b) {name}: arrival = dispatch + E epochs at the drifting speeds, k earliest aggregated", ok and mono)
        elif m == "parallelsfl":
            ctx2 = Context(cfg, small, eng(), quiet)                       # replays the same rng stream
            algo = _ParallelSFLCompute(lam=cfg.psfl_lambda, tau_max=cfg.E * batches(float(np.mean(ctx2.sizes)), cfg.B),
                                       max_cluster_size=cfg.psfl_max_cluster)
            macs = layer_macs(spec)
            r_top = sum(macs[cfg.cut:]) / sum(macs[:cfg.cut])
            t, ok = 0.0, True
            for r in rows[1:]:
                s_now = np.exp(ref.sp._log(int(np.floor(t / dt))))
                infos = [WorkerInfo(client_id=k, label_dist=ctx2.label_dists[k], num_samples=int(ctx2.sizes[k]),
                                    mu_b=t0 / s_now[k], mu_p=r_top * t0 / s_now[k], beta_smashed=0.0, beta_full=0.0,
                                    ingress_bw_hz=algo.b * cfg.psfl_max_cluster) for k in range(K)]
                cl = algo.cluster_workers(infos, ctx2.rng)
                algo.assign_frequencies(cl)
                tc = []
                for c in cl:
                    x = t
                    for _ in range(c.tau):
                        x = ref.finish(c.top.client_id, max(ref.finish(w.client_id, x, t0) for w in c.bottoms), r_top * t0)
                    tc.append(x - t)
                t += max(tc)
                ok &= rel(r["round_latency_s"], max(tc), 1e-7) and rel(r["sim_time_s"], t, 1e-7) \
                    and r["tau_max"] == max(c.tau for c in cl) and r["num_clusters"] == len(cl)
            check(f"(b) {name}: Algorithm 1 on the current speeds, Eq. 15 integrated over the drift", ok and mono)
        else:
            tr = ctx.trace
            deadline, floor = cfg.local_work in ("deadline", "slot"), cfg.local_work == "slot"
            L = float(np.quantile(ctx.tk, cfg.deadline_q)) if deadline else 0.0
            halts = [(e["now"], e["halt"]) for e in tr if e["halt"] > 0]
            h_ok = all(rel(h, max(ref.finish(k, tn, t0) - tn for k in range(K)), 1e-7) for tn, h in halts)
            if cfg.clustering == "gradient":                         # initial probe on w^0, then the first rounds
                h_ok &= rel(rows[0]["sim_time_s"], max(ref.finish(k, 0.0, t0) for k in range(K)), 1e-7)
            starts = {0.0, rows[0]["sim_time_s"]} | {e["now"] for e in tr} | {tn + h for tn, h in halts}
            caps_ok = fin_ok = done_ok = start_ok = paused = True
            n_paused = 0
            for e in tr:
                for bb in e["batch"]:
                    s0 = bb["started"]
                    start_ok &= any(rel(s0, x, 1e-12) for x in starts)
                    inflight = [(tn, h) for tn, h in halts if s0 <= tn < bb["done"]]
                    n_paused += bool(inflight)
                    fins = []
                    for k in bb["members"]:
                        n = int(nbf[k])
                        if deadline:
                            n = min(n, max(1, int(math.floor(ref.work(k, s0, s0 + L) / t0 + 1e-9))))
                        caps_ok &= bb["caps"][k] == n and 0 < bb["samp"][k] <= n * cfg.B
                        tt_, rem, A = s0, n * t0, None                   # its own work, paused by every halt
                        for tn, h in inflight:
                            w = ref.work(k, tt_, tn)
                            if rem <= w:
                                A = ref.finish(k, tt_, rem)
                                break
                            rem, tt_ = rem - w, tn + h
                        A = ref.finish(k, tt_, rem) if A is None else A
                        fin = A + sum(h for tn, h in inflight if tn >= A)  # a finished member waits through later halts
                        fin_ok &= rel(bb["fin"][k], fin, 1e-7)
                        fins.append(fin)
                    done = max(fins + ([s0 + L + sum(h for _, h in inflight)] if floor else []))
                    done_ok &= rel(bb["done"], done, 1e-7) and rel(e["now"], bb["done"], 1e-12)
            check(f"(b) {name}: halts = max_k (one batch at the current speeds), rounds start at events / after halts",
                  h_ok and start_ok and mono, f"{len(halts)} halts")
            check(f"(b) {name}: per-member steps (E epochs{' or what fits in L' if deadline else ''}) and finish times, "
                  f"round end = max(finishes{', start + L' if floor else ''}), paused by halts",
                  caps_ok and fin_ok and done_ok, f"{sum(len(e['batch']) for e in tr)} rounds, {n_paused} paused by a halt")


if __name__ == "__main__":
    sys.exit(main())
