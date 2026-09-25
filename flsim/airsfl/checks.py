"""
flsim/airsfl/checks.py: correctness checks for the AirSFL environment.
Run:  python -m flsim.airsfl.checks

  1. Timing: roadmap golden numbers + Table I limit cases (UL and DL).
  2. Model: roadmap profiled d_c / d_s / d_a at all four cuts.
  3. Radio: Monte-Carlo MSE vs Eq. 10 / 20; scale-free (simulator) noise formulas
     equal the dimensional (radio.py) ones; torch CN(0,1) sampling is correct.
  4. Training (small CIFAR subset, CPU):
     a. relay back-prop (Eq. 11-13) == full-model back-prop, one step;
     b. NOISELESS equivalence: digital FedAvg == digital SFL-V1 (cuts 1-4) ==
        AirSFL == Sun-style == AirComp-FL  (final parameters);
     c. NO-SPLIT limits with the noise ON (cut after the last layer):
        SFL-V1 == FedAvg, AirSFL == AirComp-FL, Sun == AirComp-FL  (bitwise);
     d. empirical ZF activation NSR ~= 1/(rho (Nr-N)) (complex-Wishart mean).
"""

import copy
import math

import numpy as np
import torch

from flsim.airsfl import radio as R
from flsim.airsfl.data import ClientStream, load_cifar10_tensors, make_eval_tensors, partition
from flsim.airsfl.model import CifarResNet18GN, check_profiled_dims
from flsim.airsfl.simulator import AirSFLSimulator, RunConfig
from flsim.airsfl.timing import RadioConfig, verify_limit_cases, verify_reference_example

_OK = lambda b: "PASS" if b else "FAIL"


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
    dim_A = R.sigma2(radio) / (2 * alpha ** 2)                                   # dimensional Eq. 20
    maxterm = max(a[n] ** 2 * float(deltas[n].pow(2).sum()) for n in range(radio.N))
    sf_A = Gt_inv.sum().real * maxterm / (2 * radio.rho_lin * m)
    ok &= abs(dim_A - sf_A) / dim_A < 1e-9
    g = torch.Generator().manual_seed(0)
    x = torch.randn(400000, dtype=torch.complex128, generator=g)
    cn_ok = abs(float(x.abs().pow(2).mean()) - 1) < 0.01 and abs(float(x.real.var()) - 0.5) < 0.01
    ok &= cn_ok
    if verbose:
        print(f"  scale-free == dimensional (ZF Eq.10, AirComp Eq.20); torch CN(0,1) ok={cn_ok}: {_OK(ok)}")
    return ok


def _small_env(N=4, Nr=16, S=8, per_client=64, n_val=200, device=torch.device("cpu")):
    data = load_cifar10_tensors()
    parts = partition(data["y_train"].numpy(), N, "iid", seed=5)
    parts = [p[:per_client] for p in parts]
    weights = np.array([len(p) for p in parts], dtype=np.float64)
    weights /= weights.sum()
    ev = {"val": (make_eval_tensors(data["x_val"][:n_val], data["mean"], data["std"], device),
                  data["y_val"][:n_val].to(device))}

    def make_streams(B, seed):
        return [ClientStream(data["x_train"][p], data["y_train"][p], data["mean"], data["std"],
                             B, seed * 1000 + k, device) for k, p in enumerate(parts)]
    radio = RadioConfig(N=N, Nr=Nr, S=S, rho_db=10.0)
    return radio, make_streams, weights, ev


def _run(method, stage, radio, make_streams, weights, ev, noiseless, rounds=2, tau=2, B=8, seed=11):
    cfg = RunConfig(method=method, stage=stage, tau=tau, batch_size=B, lr=0.05, rounds=rounds,
                    eval_every=rounds, noiseless=noiseless, seed=seed)
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
    ref = _run("digital_fedavg", None, radio, make_streams, weights, ev, noiseless=True).global_vector()
    if verbose:
        print("  (b) NOISELESS equivalence vs digital FedAvg (2 rounds, tau=2, N=4):")
    cases = [("digital_sflv1", s) for s in (1, 2, 3, 4)] + \
            [("airsfl", 3), ("airsfl", 1), ("sun_fdma_aircomp", 3), ("aircomp_fl", None)]
    for method, stage in cases:
        v = _run(method, stage, radio, make_streams, weights, ev, noiseless=True).global_vector()
        diff = float((v - ref).abs().max())
        good = diff < 1e-4
        ok &= good
        if verbose:
            print(f"      {method:18s} cut={stage}: max|theta - theta_FedAvg| = {diff:.2e}  {_OK(good)}")
    if verbose:
        print("  (c) NO-SPLIT limits with analog noise ON (cut after the last layer):")
    fed = _run("digital_fedavg", None, radio, make_streams, weights, ev, noiseless=False).global_vector()
    acfl = _run("aircomp_fl", None, radio, make_streams, weights, ev, noiseless=False).global_vector()
    limits = [("digital_sflv1", fed, "FedAvg"), ("airsfl", acfl, "AirComp-FL"),
              ("sun_fdma_aircomp", acfl, "AirComp-FL")]
    for method, target, name in limits:
        v = _run(method, None, radio, make_streams, weights, ev, noiseless=False).global_vector()
        diff = float((v - target).abs().max())
        good = diff == 0.0
        ok &= good
        if verbose:
            print(f"      {method:18s}(no split) == {name:10s}: max|diff| = {diff:.2e}  {_OK(good)}")
    noisy = _run("airsfl", 3, radio, make_streams, weights, ev, noiseless=False).global_vector()
    moved = float((noisy - ref).abs().max())
    ok &= moved > 0
    if verbose:
        print(f"      sanity: noisy AirSFL differs from noiseless reference (max|diff| = {moved:.2e}) "
              f"{_OK(moved > 0)}")
    return ok


def check_activation_nsr(verbose=True) -> bool:
    radio, make_streams, weights, ev = _small_env()
    sim = _run("airsfl", 3, radio, make_streams, weights, ev, noiseless=False, rounds=3)
    emp_db = np.nanmean([h["act_nsr_db"] for h in sim.history[1:]])
    theory_db = 10 * math.log10(1.0 / (radio.rho_lin * (radio.Nr - radio.N)))
    good = abs(emp_db - theory_db) < 1.0
    if verbose:
        print(f"  (d) ZF activation NSR: empirical {emp_db:.2f} dB vs theory 1/(rho(Nr-N)) = "
              f"{theory_db:.2f} dB  {_OK(good)}")
    return good


def run_all() -> bool:
    print("\n########## 1. TIMING ##########")
    verify_reference_example()
    ok = verify_limit_cases()
    print("\n########## 2. MODEL ##########")
    ok &= check_profiled_dims()["ok"]
    print("\n########## 3. RADIO ##########")
    ok &= R.run_radio_checks()
    ok &= check_radio_scale_free()
    print("\n########## 4. TRAINING ##########")
    ok &= check_relay_gradient()
    ok &= check_training_equivalences()
    ok &= check_activation_nsr()
    print(f"\n==== ALL AIRSFL CHECKS: {_OK(ok)} ====")
    return ok


if __name__ == "__main__":
    run_all()
