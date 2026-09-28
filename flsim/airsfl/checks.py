"""
flsim/airsfl/checks.py: correctness checks for the AirSFL environment
(evaluation roadmap Sec. 9, implementation gates). Run:  python -m flsim.airsfl.checks

  1. Timing: formula regression + Eq. 25-26 limit cases (each phase counted once).
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
     g. vectorized crop+flip == per-image reference.
"""

import copy
import math

import numpy as np
import torch

from flsim.airsfl import radio as R
from flsim.airsfl.data import ClientStream, load_cifar10_tensors, make_eval_tensors, partition
from flsim.airsfl.model import CifarResNet18GN, check_profiled_dims
from flsim.airsfl.simulator import AirSFLSimulator, RunConfig, _HAS_FUNC
from flsim.airsfl.timing import RadioConfig, verify_limit_cases, verify_reference_example

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
         seed=11):
    cfg = RunConfig(method=method, stage=stage, tau=tau, batch_size=B, lr=0.05, rounds=rounds,
                    noiseless=noiseless, seed=seed, engine=engine)
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
    print("\n########## 2. MODEL ##########")
    ok &= check_profiled_dims()["ok"]
    print("\n########## 3. RADIO ##########")
    ok &= R.run_radio_checks()
    ok &= check_radio_scale_free()
    print(f"\n########## 4. TRAINING (engines: {', '.join(ENGINES)}) ##########")
    ok &= check_relay_gradient()
    ok &= check_training_equivalences()
    ok &= check_activation_nsr_and_zero()
    ok &= check_crop()
    print(f"\n==== ALL AIRSFL CHECKS: {_OK(ok)} ====")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all() else 1)
