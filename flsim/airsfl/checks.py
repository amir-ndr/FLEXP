
"""
flsim/airsfl/checks.py: correctness checks for the AirSFL environment
(evaluation roadmap Sec. 9, implementation gates). Run:  python -m flsim.airsfl.checks

  1. Timing: formula regression + Eq. 25-26 limit cases (each phase counted once); digital
     multi-user ZF rate (Gamma(Nr-N+1) gain vs explicit ZF channels, N=1 == OFDMA, golden).
  2. Model: roadmap d_c / d_s / d_a at all four cuts.
  3. Radio: gates 1-3, 5 (packing, packet power, ZF identity, noiseless AirComp, zero
     packets); Monte-Carlo MSE vs Eq. 10 / 23; scale-free == dimensional formulas.
  4. Training (small CIFAR subset, CPU):
     a. relay back-prop (Eq. 14-16) == full-model back-prop at every cut;
     b. NOISELESS: digital SFL-V1 (cuts 1-4), AirSFL, Sun, AirComp-FL == digital FedAvg,
        for both engines (gate 6);
     c. NO-SPLIT limits with the analog errors ON: SFL-V1 == FedAvg, AirSFL == AirComp-FL,
        Sun == AirComp-FL (bitwise);
     d. engine equivalence: the parallel "vmap" engine (detached additive ZF error)
        == the reference "loop" engine (relay back-prop), noiseless and noisy (gate 7);
     e. empirical ZF activation NSR ~= 1/(rho (Nr-N));
     f. zero activation packets / zero aggregates carry zero error;
     h. recorded computation / training-time columns == the computation model;
     i. radio draws depend only on (seed, link, round, step);
     j. derived digital baselines (multi-user ZF; FP32 and FP16) == their OFDMA parents' learning
        (bitwise), timing == retiming;
     l. FP16 digital payload: transport error bound, analog methods unaffected (bitwise),
        digital methods rounded at the FP16 level, both engines agree;
     k. unequal path gains: scale-free == dimensional, sampler Monte Carlo, uniform offsets ==
        shifted SNR, all-zero offsets == equal gains;
     g. vectorized crop+flip == per-image reference.
  Computation model (flsim.airsfl.compute.verify_compute): FLOP counts vs closed form and
  the framework counter, prefix + suffix == full model, N=1 limit, hand calculation.
"""

import copy
import math

import numpy as np
import torch

from flsim.airsfl import radio as R
from flsim.airsfl.compute import compute_time_breakdown, split_flops, verify_compute
from flsim.airsfl.data import ClientStream, load_cifar10_tensors, make_eval_tensors, partition
from flsim.airsfl.model import CifarResNet18GN, check_profiled_dims
from flsim.airsfl.simulator import AirSFLSimulator, RunConfig, _HAS_FUNC
from flsim.airsfl.timing import (RadioConfig, digital_rates, uplink_time_breakdown, verify_limit_cases,
                                 verify_fp16, verify_path_gains, verify_reference_example, verify_zf)

_OK = lambda b: "PASS" if b else "FAIL"
ENGINES = ("loop", "vmap") if _HAS_FUNC else ("loop",)


def check_radio_scale_free(verbose=True) -> bool:
    """Scale-free simulator formulas == dimensional radio.py formulas on one channel."""
    radio = RadioConfig(N=6, Nr=16, S=12, rho_db=10.0)
    rng = np.random.RandomState(3)
    Ht = (rng.normal(size=(radio.Nr, radio.N)) + 1j * rng.normal(size=(radio.Nr, radio.N))) / math.sqrt(2)
    H = math.sqrt(radio.lambda_ref) * Ht                                   # physical channel
    Gt_inv = np.linalg.inv(Ht.conj().T @ Ht)
    z = torch.randn(256, dtype=torch.float64)
    m = 128
    ok = True
    for n in range(radio.N):
        dim = R.zf_noise_var_per_coord(z, R.zf_column_norms_sq(H)[n], radio)     # dimensional Eq. 10
        sf = Gt_inv[n, n].real * float(z.pow(2).sum()) / (2 * radio.rho_lin * m)  # simulator form
        ok &= abs(dim - sf) / dim < 1e-9
    deltas = [torch.randn(256, dtype=torch.float64) for _ in range(radio.N)]
    a = [1.0 / radio.N] * radio.N
    alpha = R.aircomp_alpha(deltas, a, R.aircomp_v_norm_sq(H), radio)
    dim_A = R.sigma2(radio) / (2 * alpha ** 2)                                   # dimensional Eq. 23
    maxterm = max(a[n] ** 2 * float(deltas[n].pow(2).sum()) for n in range(radio.N))
    sf_A = Gt_inv.sum().real * maxterm / (2 * radio.rho_lin * m)
    ok &= abs(dim_A - sf_A) / dim_A < 1e-9
    if verbose:
        print(f"  scale-free == dimensional (ZF Eq. 10, AirComp Eq. 23): {_OK(ok)}")
    return ok


def _small_env(N=4, Nr=16, S=8, per_client=64, n_val=200, device=torch.device("cpu")):
    data = load_cifar10_tensors()
    parts = partition(data["y_train"].numpy(), N, "iid", seed=5)
    parts = [p[:per_client] for p in parts]
    weights = np.array([len(p) for p in parts], dtype=np.float64)
    weights /= weights.sum()
    ev = {"val": (make_eval_tensors(data["x_val"][:n_val], data["mean"], data["std"], device),
                  data["y_val"][:n_val].to(device))}

    def make_streams(B, seed, augment=True):
        return [ClientStream(data["x_train"][p], data["y_train"][p], data["mean"], data["std"],
                             B, seed * 1000 + k, device, augment=augment) for k, p in enumerate(parts)]
    radio = RadioConfig(N=N, Nr=Nr, S=S, rho_db=10.0)
    return radio, make_streams, weights, ev


def _run(method, stage, radio, make_streams, weights, ev, noiseless, engine="loop", rounds=2, tau=2, B=8,
         seed=11, eval_rounds=()):
    cfg = RunConfig(method=method, stage=stage, tau=tau, batch_size=B, lr=0.05, rounds=rounds,
                    noiseless=noiseless, seed=seed, engine=engine, eval_rounds=eval_rounds)
    sim = AirSFLSimulator(cfg, copy.deepcopy(radio), make_streams(B, seed), weights, ev,
                          torch.device("cpu"), n_train=45000, log=lambda *a: None)
    sim.run()
    return sim


def check_relay_gradient(verbose=True) -> bool:
    torch.manual_seed(0)
    M = CifarResNet18GN()
    x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
    M2 = copy.deepcopy(M)
    torch.nn.functional.cross_entropy(M(x), y).backward()
    full = torch.cat([p.grad.reshape(-1) for p in M.parameters()])
    ok = True
    for stage in (1, 2, 3, 4):
        M3 = copy.deepcopy(M2)
        z = M3.stem(x)
        for L in M3.stages()[:stage]:
            z = L(z)
        relay = z.detach().clone().requires_grad_(True)
        zz = relay
        for L in M3.stages()[stage:]:
            zz = L(zz)
        torch.nn.functional.cross_entropy(M3.fc(M3.pool(zz).flatten(1)), y).backward()
        z.backward(relay.grad)
        split = torch.cat([p.grad.reshape(-1) for p in M3.parameters()])
        diff = float((full - split).abs().max())
        ok &= diff < 1e-6
        if verbose:
            print(f"  relay back-prop == full back-prop at cut {stage}: max|diff| = {diff:.2e}")
    if verbose:
        print(f"  RELAY_GRADIENT: {_OK(ok)}")
    return ok


def check_training_equivalences(verbose=True) -> bool:
    radio, make_streams, weights, ev = _small_env()
    ok = True
    cases = [("digital_sflv1", s) for s in (1, 2, 3, 4)] + \
            [("airsfl", 2), ("airsfl", 1), ("sun_fdma_aircomp", 2), ("aircomp_fl", None)]
    refs = {}
    for engine in ENGINES:
        ref = _run("digital_fedavg", None, radio, make_streams, weights, ev, True, engine).global_vector()
        refs[engine] = ref
        if verbose:
            print(f"  (b) NOISELESS equivalence vs digital FedAvg [{engine} engine]:")
        for method, stage in cases:
            v = _run(method, stage, radio, make_streams, weights, ev, True, engine).global_vector()
            diff = float((v - ref).abs().max())
            good = diff < 1e-5
            ok &= good
            if verbose:
                print(f"      {method:18s} cut={stage}: max|theta - theta_FedAvg| = {diff:.2e}  {_OK(good)}")
        if verbose:
            print(f"  (c) NO-SPLIT limits with analog errors ON [{engine} engine]:")
        fed = _run("digital_fedavg", None, radio, make_streams, weights, ev, False, engine).global_vector()
        acfl = _run("aircomp_fl", None, radio, make_streams, weights, ev, False, engine).global_vector()
        for method, target, name in [("digital_sflv1", fed, "FedAvg"), ("airsfl", acfl, "AirComp-FL"),
                                     ("sun_fdma_aircomp", acfl, "AirComp-FL")]:
            v = _run(method, None, radio, make_streams, weights, ev, False, engine).global_vector()
            diff = float((v - target).abs().max())
            ok &= diff == 0.0
            if verbose:
                print(f"      {method:18s}(no split) == {name:10s}: max|diff| = {diff:.2e}  {_OK(diff == 0.0)}")
    if len(ENGINES) == 2:
        ok &= check_vmap_step_vs_relay64(radio, make_streams, weights, ev, verbose)
        if verbose:
            print("  (d2) ENGINE trajectories, vmap vs loop over 2 rounds x 2 steps (plumbing check; float32\n"
                  "       kernels differ, so the tolerance is loose but far below any plumbing error):")
        for method, stage, noiseless in [("digital_fedavg", None, True), ("airsfl", 2, False),
                                         ("sun_fdma_aircomp", 2, False)]:
            a = _run(method, stage, radio, make_streams, weights, ev, noiseless, "loop").global_vector()
            b = _run(method, stage, radio, make_streams, weights, ev, noiseless, "vmap").global_vector()
            diff = float((a - b).abs().max())
            med = float((a - b).abs().median())
            good = diff < 1e-2 and med < 1e-5
            ok &= good
            if verbose:
                print(f"      {method:18s} noiseless={noiseless!s:5s}: max|loop - vmap| = {diff:.2e}, "
                      f"median {med:.1e}  {_OK(good)}")
    noisy = _run("airsfl", 2, radio, make_streams, weights, ev, False, ENGINES[-1]).global_vector()
    moved = float((noisy - refs[ENGINES[-1]]).abs().max())
    ok &= moved > 0
    if verbose:
        print(f"      sanity: noisy AirSFL differs from the noiseless reference (max|diff| = {moved:.2e}) "
              f"{_OK(moved > 0)}")
    return ok


def check_vmap_step_vs_relay64(radio, make_streams, weights, ev, verbose=True) -> bool:
    """(d1) One co-split step WITH ZF error, both implementations in float64 (so the
    comparison tests the math, not float32 kernel choices): the vmap engine's
    per-client gradients (detached additive error, identity Jacobian at the cut) ==
    relay back-prop (Eq. 14-16) with the same error realization."""
    import torch.nn.functional as F
    from torch.func import grad_and_value, vmap
    cfg = RunConfig(method="airsfl", stage=2, tau=1, batch_size=8, lr=0.05, rounds=1, seed=11, engine="vmap")
    sim = AirSFLSimulator(cfg, copy.deepcopy(radio), make_streams(8, 11), weights, ev,
                          torch.device("cpu"), n_train=45000, log=lambda *a: None)
    batches = [s.next_batch() for s in sim.streams]
    X = torch.stack([b[0] for b in batches]).double()
    Y = torch.stack([b[1] for b in batches])
    E = sim._sample_zf_joint().double()
    params64 = {k: v.double() for k, v in sim.stacked.items()}
    g, _ = vmap(grad_and_value(sim._loss_noisy, has_aux=True))(params64, X, Y, E)
    worst = 0.0
    for n in range(sim.N):
        M = copy.deepcopy(sim.global_model).double()
        z = sim._prefix(M, X[n])
        zt = z.detach().clone()
        relay = (zt + sim._zf_noise(zt, E[n])).requires_grad_(True)
        F.cross_entropy(sim._suffix(M, relay), Y[n]).backward()      # suffix grad (15), q (14)
        z.backward(relay.grad)                                          # prefix grad (16)
        for name, p in M.named_parameters():
            ref = p.grad
            worst = max(worst, float((g[name][n] - ref).abs().max()) / max(float(ref.abs().max()), 1e-30))
    ok = worst < 1e-9
    if verbose:
        print(f"  (d1) vmap step (additive detached ZF error) == float64 relay back-prop: "
              f"max relative grad error = {worst:.1e}  {_OK(ok)}")
    return ok


def check_activation_nsr_and_zero(verbose=True) -> bool:
    radio, make_streams, weights, ev = _small_env()
    sim = _run("airsfl", 2, radio, make_streams, weights, ev, False, ENGINES[-1], rounds=3)
    emp_db = np.nanmean([h["act_nsr_db"] for h in sim.history[1:]])
    theory_db = 10 * math.log10(1.0 / (radio.rho_lin * (radio.Nr - radio.N)))
    good = abs(emp_db - theory_db) < 1.0
    # zero packets: an all-zero activation and an all-zero aggregate get exactly zero error
    Er = torch.randn(sim.K_act, 256)
    z0 = torch.zeros(sim.dims["d_a"])
    zero_act = float(sim._zf_noise(z0, Er).abs().max()) == 0.0
    zero_agg = float(sim._aircomp_noise(torch.zeros(5), 5 * 256).abs().max()) == 0.0
    ok = good and zero_act and zero_agg
    if verbose:
        print(f"  (e) ZF activation NSR: empirical {emp_db:.2f} dB vs theory 1/(rho(Nr-N)) = {theory_db:.2f} dB  "
              f"{_OK(good)}")
        print(f"  (f) zero activation packets -> zero error: {zero_act}; zero aggregate -> zero error: "
              f"{zero_agg}  {_OK(zero_act and zero_agg)}")
    return ok


def check_time_columns_and_streams(verbose=True) -> bool:
    """(h) Every recorded row: training time = round * (uplink + computation) per round, the
    computation equals compute_time_breakdown(...) for the run's cut / N / b / tau / f_min /
    f_s, and FL methods have no server computation. (i) A round's radio draws depend only on
    (seed, link, round, step): re-seeding reproduces them regardless of earlier draws."""
    radio, make_streams, weights, ev = _small_env()
    ok = True
    for method, stage in (("airsfl", 2), ("digital_sflv1", 1), ("aircomp_fl", None), ("digital_fedavg", None)):
        sim = _run(method, stage, radio, make_streams, weights, ev, False, ENGINES[-1], rounds=2)
        c = compute_time_breakdown(sim.method, split_flops(sim.stage), sim.N, sim.cfg.batch_size, sim.cfg.tau,
                                   sim.f_client_min, sim.f_server)
        good = abs(c["total"] - sim.comp_s) < 1e-15 and (sim.stage is not None or c["server_fp"] == 0.0)
        for h in sim.history:
            good &= abs(h["training_time_s"] - h["round"] * (h["ul_s_per_round"] + h["compute_s_per_round"])) < 1e-9
            good &= abs(h["cumulative_compute_s"] + h["uplink_s"] - h["training_time_s"]) < 1e-9
        ok &= good
        if verbose:
            print(f"  (h) time columns [{method:14s} cut={stage}]: compute {sim.comp_s:.4g} s/round "
                  f"(f_min={sim.f_client_min/1e12:.2f} TFLOPS), training = uplink + compute  {_OK(good)}")
    sim = _run("airsfl", 2, radio, make_streams, weights, ev, False, ENGINES[-1], rounds=1)
    sim.gen_U.manual_seed(sim._stream_seed(1, 3, 2))
    e1 = sim._sample_zf_joint()
    sim._sample_zf_joint()                                    # consume more randomness
    sim.gen_U.manual_seed(sim._stream_seed(1, 3, 2))
    e2 = sim._sample_zf_joint()
    seeds = {sim._stream_seed(l, r, i) for l in (1, 2) for r in range(200) for i in range(60)}
    good = torch.equal(e1, e2) and len(seeds) == 2 * 200 * 60
    ok &= good
    if verbose:
        print(f"  (i) radio draws depend only on (seed, link, round, step), no seed collisions: {_OK(good)}")
    # extra evaluation rounds (--early-evals) only add rows; the training trajectory is untouched
    a = _run("airsfl", 2, radio, make_streams, weights, ev, False, ENGINES[-1], rounds=3)
    b = _run("airsfl", 2, radio, make_streams, weights, ev, False, ENGINES[-1], rounds=3, eval_rounds=(1, 2))
    good = torch.equal(a.global_vector(), b.global_vector()) and \
        [h["round"] for h in b.history] == sorted({h["round"] for h in a.history} | {1, 2})
    ok &= good
    if verbose:
        print(f"      extra evaluation rounds leave the trajectory bitwise unchanged: {_OK(good)}")
    return ok


DERIVED_PAIRS = (("digital_sflv1_zf", "digital_sflv1", 2), ("hybrid_zf_aircomp", "sun_fdma_aircomp", 2),
                 ("digital_fedavg_zf", "digital_fedavg", None))


def check_zf_baselines(verbose=True) -> bool:
    """(j) The derived digital baselines (multi-user ZF) learn exactly like their OFDMA
    parents (the digital links are reliable either way; the hybrids share the parent's AirComp
    stream): identical models, losses and accuracies. Their recorded uplink columns equal the
    parent run's rows retimed with uplink_time_breakdown -- the derivation plots.py uses instead
    of training them again -- and their computation is the parent's. Also for FP16 payloads."""
    radio, make_streams, weights, ev = _small_env()
    same = lambda x, y: x == y or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y))
    ok = True
    import dataclasses
    for q in (32, 16):
        rq = dataclasses.replace(radio, q_bits=q)
        for der, parent, stage in DERIVED_PAIRS:
            if q == 16 and der not in ("digital_sflv1_zf", "digital_fedavg_zf"):
                continue                          # FP16: one split and one FL pair are enough
            a = _run(parent, stage, rq, make_streams, weights, ev, False, ENGINES[-1])
            b = _run(der, stage, rq, make_streams, weights, ev, False, ENGINES[-1])
            good = torch.equal(a.global_vector(), b.global_vector()) and len(a.history) == len(b.history)
            h0 = a.history[0]
            rd = RadioConfig(N=h0["N"], Nr=h0["Nr"], Nr_F=h0["Nr_F"], S=h0["S"], eps_D=h0["eps_D"],
                             eps_U=h0["eps_U"], eps_A=h0["eps_A"], rho_db=h0["rho_db"], batch_size=h0["B"],
                             q_bits=h0["q_bits"])
            br = uplink_time_breakdown(der, {k: h0[k] for k in ("d_c", "d_s", "d_a")}, rd, h0["tau"],
                                       digital_rates(rd))
            for ha, hb in zip(a.history, b.history):
                good &= all(same(ha[k], hb[k]) for k in ("round", "train_loss", "val_acc", "val_loss", "agg_nsr_db",
                                                         "act_nsr_db", "compute_s_per_round", "mb_per_round"))
                good &= all(abs(hb[c] - br[k]) <= 1e-12 * max(1.0, br[k]) for c, k in
                            (("activation_ul_s", "activation"), ("labels_ul_s", "labels"),
                             ("aggregation_ul_s", "aggregation"), ("ul_s_per_round", "total")))
                good &= abs(hb["training_time_s"] - hb["round"] * (br["total"] + ha["compute_s_per_round"])) < 1e-9
            ok &= good
            if verbose:
                print(f"  (j) {der:19s} == {parent:16s} q={q}: learning bitwise, uplink {b.ul_s:9.3f} vs "
                      f"{a.ul_s:9.3f} s/round == retiming of the parent run  {_OK(good)}")
    return ok


def check_fp16(verbose=True) -> bool:
    """(l) FP16 digital payload:
      1. _fp16_transport: per-256-value block error <= 2^-11 of the block maximum (2^-25 in the
         subnormal range), over 12 decades of magnitude; zero blocks stay zero; FP16-representable
         blocks are reproduced exactly;
      2. AirSFL and AirComp-FL upload nothing digitally except labels: with q = 16 their training
         and their uplink time are bitwise those of q = 32;
      3. digital SFL-V1 and digital FedAvg with q = 16: the rounding error is recorded at the
         FP16 level (activation / aggregate NSR between -80 and -55 dB), the trajectory stays close
         to FP32, the uplink time is that of q = 16, and both engines agree (loose, as in (d2))."""
    import dataclasses
    from flsim.airsfl.simulator import _fp16_transport
    ok = True
    g = torch.Generator().manual_seed(0)
    x = torch.randn(40, 256, generator=g, dtype=torch.float64)
    x = x * (10.0 ** torch.linspace(-9, 3, 40, dtype=torch.float64))[:, None]     # 12 decades across blocks
    x[7] = 0.0
    xf = x.float()
    y = _fp16_transport(xf.reshape(-1)).reshape(40, 256).double()
    s = xf.double().abs().amax(dim=1, keepdim=True)
    good = bool(((y - xf.double()).abs() <= 2.0 ** -11 * s * (1 + 1e-6) + 1e-45).all()) and bool((y[7] == 0).all())
    rep = torch.tensor([0.5, -0.25, 1.0, 0.125] * 64)                       # exactly representable after scaling
    good &= torch.equal(_fp16_transport(rep), rep)
    ok &= good
    if verbose:
        rel = float(((y - xf.double()).abs() / s.clamp_min(1e-300)).max())
        print(f"  (l1) FP16 transport: max error / block max = {rel:.2e} (<= 2^-11 = {2.0 ** -11:.2e}), zero block "
              f"-> zero, representable -> exact  {_OK(good)}")
    radio, make_streams, weights, ev = _small_env()
    r16 = dataclasses.replace(radio, q_bits=16)
    for m, stage in (("airsfl", 2), ("aircomp_fl", None)):
        a = _run(m, stage, radio, make_streams, weights, ev, False, ENGINES[-1])
        b = _run(m, stage, r16, make_streams, weights, ev, False, ENGINES[-1])
        good = torch.equal(a.global_vector(), b.global_vector()) and a.ul_s == b.ul_s
        ok &= good
        if verbose:
            print(f"  (l2) {m:14s} q=16 == q=32 (training bitwise, uplink {b.ul_s:.4f} s/round)  {_OK(good)}")
    for m, stage in (("digital_sflv1", 2), ("digital_fedavg", None), ("sun_fdma_aircomp", 2)):
        a = _run(m, stage, radio, make_streams, weights, ev, False, ENGINES[-1], rounds=3)
        b = _run(m, stage, r16, make_streams, weights, ev, False, ENGINES[-1], rounds=3)
        c = _run(m, stage, r16, make_streams, weights, ev, False, "loop", rounds=3)
        dv = (a.global_vector() - b.global_vector()).abs()
        de = (b.global_vector() - c.global_vector()).abs()
        mean_db = lambda k: (lambda v: float(np.mean(v)) if v else float("nan"))(
            [h[k] for h in b.history[1:] if not math.isnan(h[k])])
        act, agg = mean_db("act_nsr_db"), mean_db("agg_nsr_db")
        want_act, want_agg = stage is not None, m != "sun_fdma_aircomp"
        good = float(dv.max()) > 0 and float(dv.max()) < 1e-2                        # moved, but only slightly
        good &= (-80 < act < -55) if want_act else math.isnan(act)
        good &= (-80 < agg < -55) if want_agg else agg > -55                          # hybrid: AirComp noise
        rd = dataclasses.replace(r16)
        rd.batch_size = b.cfg.batch_size
        good &= abs(b.ul_s - uplink_time_breakdown(m, b.dims, rd, b.cfg.tau, digital_rates(rd))["total"]) < 1e-12
        good &= float(de.max()) < 1e-2 and float(de.median()) < 1e-5
        ok &= good
        if verbose:
            print(f"  (l3) {m:16s} q=16: act NSR {act:6.1f} dB, agg NSR {agg:6.1f} dB, max|theta16 - theta32| = "
                  f"{float(dv.max()):.1e}, loop vs vmap max {float(de.max()):.1e} / median {float(de.median()):.1e}, "
                  f"uplink {b.ul_s:.3f} vs {a.ul_s:.3f} s/round  {_OK(good)}")
    return ok


def check_path_gains(verbose=True) -> bool:
    """(k) Unequal path gains lambda_n = g_n lambda_ref (optional robustness setting):
      1. the simulator's scale-free forms == dimensional Eq. 10 / 23 on a channel with
         unequal gains ([G^-1]_nn = [G~^-1]_nn / g_n; ||v||^2 = d^T G~^-1 d, d = 1/sqrt(g));
      2. Monte Carlo through the simulator's own samplers: joint ZF error variance per client
         E[G~inv_nn] / (2 g_n) = 1 / (2 (Nr - N) g_n); AirComp E[||v~||^2] = sum_n (1/g_n) / (Nr - N);
      3. training: equal offsets of -3 dB for every client == equal gains at rho - 3 dB (same
         trajectory to float rounding, same time columns);
      4. all-zero offsets take the equal-gain code path (bitwise identical run)."""
    import dataclasses
    from flsim.airsfl.timing import path_gain_offsets_db
    ok = True
    # 1. scale-free == dimensional on one channel
    radio = RadioConfig(N=6, Nr=16, S=12, rho_db=10.0, gains_db=path_gain_offsets_db(6, 20.0, 3))
    g = radio.g_lin
    rng = np.random.RandomState(3)
    Ht = (rng.normal(size=(radio.Nr, radio.N)) + 1j * rng.normal(size=(radio.Nr, radio.N))) / math.sqrt(2)
    H = math.sqrt(radio.lambda_ref) * Ht * np.sqrt(g)[None, :]                   # physical channel
    Gt_inv = np.linalg.inv(Ht.conj().T @ Ht)
    z = torch.randn(256, dtype=torch.float64)
    m = 128
    good = True
    for n in range(radio.N):
        dim = R.zf_noise_var_per_coord(z, R.zf_column_norms_sq(H)[n], radio)
        sf = Gt_inv[n, n].real / g[n] * float(z.pow(2).sum()) / (2 * radio.rho_lin * m)
        good &= abs(dim - sf) / dim < 1e-9
    deltas = [torch.randn(256, dtype=torch.float64) for _ in range(radio.N)]
    a = [1.0 / radio.N] * radio.N
    dim_A = R.sigma2(radio) / (2 * R.aircomp_alpha(deltas, a, R.aircomp_v_norm_sq(H), radio) ** 2)
    dvec = 1.0 / np.sqrt(g)
    maxterm = max(a[n] ** 2 * float(deltas[n].pow(2).sum()) for n in range(radio.N))
    sf_A = float(np.real(dvec @ Gt_inv @ dvec)) * maxterm / (2 * radio.rho_lin * m)
    good &= abs(dim_A - sf_A) / dim_A < 1e-9
    ok &= good
    if verbose:
        print(f"  (k1) unequal gains: scale-free == dimensional (ZF Eq. 10, AirComp Eq. 23): {_OK(good)}")
    # 2. Monte Carlo through the simulator's samplers
    base, make_streams, weights, ev = _small_env()
    rg = dataclasses.replace(base, gains_db=path_gain_offsets_db(base.N, 20.0, 5))
    cfg = RunConfig(method="airsfl", stage=2, tau=1, batch_size=8, lr=0.05, rounds=1, seed=11, engine="loop")
    sim = AirSFLSimulator(cfg, copy.deepcopy(rg), make_streams(8, 11), weights, ev, torch.device("cpu"),
                          n_train=45000, log=lambda *a: None)
    gl = rg.g_lin
    E = torch.cat([sim._sample_zf_joint() for _ in range(4)], dim=1).double()       # (N, 4K, 256)
    emp = E.pow(2).mean(dim=(1, 2)).numpy()
    theory = 1.0 / (2 * (rg.Nr - rg.N) * gl)
    err_zf = float(np.max(np.abs(emp / theory - 1)))
    sim.gen_A.manual_seed(123)
    K = 4096
    noise = sim._aircomp_noise(torch.ones(K), K * 256).double()
    emp_v2 = float(noise.pow(2).mean()) * 2 * rg.rho_lin * 128
    th_v2 = float(np.sum(1.0 / gl)) / (rg.Nr_F - rg.N)
    err_ac = abs(emp_v2 / th_v2 - 1)
    good = err_zf < 0.03 and err_ac < 0.03
    ok &= good
    if verbose:
        print(f"  (k2) simulator samplers vs theory: ZF per-client variance max rel. error {100 * err_zf:.2f}%, "
              f"AirComp ||v||^2 rel. error {100 * err_ac:.2f}%  {_OK(good)}")
    # 3. uniform -3 dB offsets == equal gains at rho - 3 dB
    a3 = _run("airsfl", 2, dataclasses.replace(base, gains_db=(-3.0,) * base.N), make_streams, weights, ev,
              False, ENGINES[-1])
    b3 = _run("airsfl", 2, dataclasses.replace(base, rho_db=base.rho_db - 3.0), make_streams, weights, ev,
              False, ENGINES[-1])
    dv = (a3.global_vector() - b3.global_vector()).abs()
    diff, med = float(dv.max()), float(dv.median())
    tdiff = max(abs(ha["uplink_s"] - hb["uplink_s"]) for ha, hb in zip(a3.history, b3.history))
    # same noise realizations up to float32 rounding (which can flip a ReLU mask): tolerance as in (d2);
    # a wrong gain scaling would move the parameters by the size of the noise itself
    good = a3.unequal and diff < 1e-2 and med < 1e-6 and tdiff < 1e-9
    ok &= good
    if verbose:
        print(f"  (k3) offsets -3 dB for all == equal gains at rho-3 dB: max|theta diff| = {diff:.1e}, "
              f"median {med:.1e}, time diff {tdiff:.1e}  {_OK(good)}")
    # 4. all-zero offsets take the equal-gain path
    a0 = _run("airsfl", 2, dataclasses.replace(base, gains_db=(0.0,) * base.N), make_streams, weights, ev,
              False, ENGINES[-1])
    b0 = _run("airsfl", 2, base, make_streams, weights, ev, False, ENGINES[-1])
    good = (not a0.unequal) and torch.equal(a0.global_vector(), b0.global_vector())
    ok &= good
    if verbose:
        print(f"  (k4) all-zero offsets == equal gains (bitwise): {_OK(good)}")
    return ok


def check_crop(verbose=True) -> bool:
    """Vectorized crop+flip == the per-image reference with the same random draws."""
    x = torch.rand(16, 3, 32, 32)
    s = ClientStream(torch.zeros(1, 3, 32, 32, dtype=torch.uint8), torch.zeros(1, dtype=torch.long),
                     torch.zeros(3), torch.ones(3), 16, 7, torch.device("cpu"))
    got = s._crop_flip(x)
    g = torch.Generator().manual_seed(7)
    padded = torch.nn.functional.pad(x, (4, 4, 4, 4))
    offs = torch.randint(0, 9, (16, 2), generator=g)
    flips = torch.rand(16, generator=g) < 0.5
    ref = torch.stack([(torch.flip(padded[i, :, r:r + 32, c:c + 32], dims=[2]) if bool(flips[i])
                        else padded[i, :, r:r + 32, c:c + 32])
                       for i, (r, c) in enumerate(offs.tolist())])
    ok = torch.equal(got, ref)
    if verbose:
        print(f"  (g) vectorized crop+flip == per-image reference: {_OK(ok)}")
    return ok


def run_all() -> bool:
    print("\n########## 1. TIMING ##########")
    ok = verify_reference_example()["ok"]
    ok &= verify_limit_cases()
    ok &= verify_zf()
    ok &= verify_path_gains()
    ok &= verify_fp16()
    print("\n########## 2. MODEL + COMPUTATION ##########")
    ok &= check_profiled_dims()["ok"]
    ok &= verify_compute()
    print("\n########## 3. RADIO ##########")
    ok &= R.run_radio_checks()
    ok &= check_radio_scale_free()
    print(f"\n########## 4. TRAINING (engines: {', '.join(ENGINES)}) ##########")
    ok &= check_relay_gradient()
    ok &= check_training_equivalences()
    ok &= check_activation_nsr_and_zero()
    ok &= check_time_columns_and_streams()
    ok &= check_zf_baselines()
    ok &= check_fp16()
    ok &= check_path_gains()
    ok &= check_crop()
    print(f"\n==== ALL AIRSFL CHECKS: {_OK(ok)} ====")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all() else 1)
