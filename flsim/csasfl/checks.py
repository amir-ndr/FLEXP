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


if __name__ == "__main__":
    sys.exit(main())
