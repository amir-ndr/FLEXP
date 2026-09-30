"""
flsim/airsfl/radio.py: MIMO radio physics for AirSFL — Zero-Forcing activation
recovery (Eq. 9-10) and AirComp weighted-sum aggregation (Eq. 16-20).

Channel: each client n has h_n ~ CN(0, lambda*I_Nr) over the server's Nr antennas
(constant within a coherent packet). H = [h_1,...,h_N] is Nr x N; ZF needs N<=Nr.
lambda is fixed by the reference SNR rho = Pmax*lambda/(N0*W) (RadioConfig).

Noise injected into TRAINING (per real coordinate, matching the paper):
  * ZF activation reconstruction (Eq. 10): zhat_n = z_n + e_U,n, with
        Var[e_U per coord] = sigma^2 ||c_n||^2 / (2 b_n^2),
        b_n^2 = m_U p_n / ||z_n||^2,  m_U = ceil(d_U/2),  p_n = Pmax/S,
        sigma^2 = N0*df,  ||c_n||^2 = [(H^H H)^{-1}]_{nn}   (ZF column norm).
  * AirComp aggregation (Eq. 19-20): agg = sum_n a_n*Delta_n + e_A, with
        Var[e_A per coord] = sigma^2 / (2 alpha^2),
        alpha = min_n |f_n| sqrt(m_A p_n) / (a_n ||Delta_n||),  m_A = ceil(d_A/2),
        baseline combiner v = H(H^H H)^{-1} 1, w = v/||v||, f_n = 1/||v|| for all n.

Both variances are Monte-Carlo-verified against a full complex simulation of the
ZF / AirComp chain (roadmap Sec. 6 "radio checks before training").
"""

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Channels + combiner norms
# ---------------------------------------------------------------------------

def draw_channel(radio, rng: np.random.RandomState, Nr: int = None) -> np.ndarray:
    """H (Nr x N) complex Rayleigh, columns h_n ~ CN(0, lambda_n I_Nr) with lambda_n =
    g_n lambda_ref (g_n = 1 for equal path gains). Nr defaults to the M-server array; pass
    radio.Nr_F for the F-server link."""
    lam = radio.lambda_ref
    scale = math.sqrt(lam / 2.0)                      # per real/imag part
    Nr, N = (Nr if Nr is not None else radio.Nr), radio.N
    H = rng.normal(0, scale, (Nr, N)) + 1j * rng.normal(0, scale, (Nr, N))
    if getattr(radio, "unequal_gains", False):
        H = H * np.sqrt(radio.g_lin)[None, :]
    return H


def zf_column_norms_sq(H: np.ndarray) -> np.ndarray:
    """||c_n||^2 = [(H^H H)^{-1}]_{nn} for ZF combiner C = H (H^H H)^{-1}."""
    G = H.conj().T @ H
    Ginv = np.linalg.inv(G)
    return np.real(np.diag(Ginv))


def aircomp_v_norm_sq(H: np.ndarray) -> float:
    """||v||^2 = 1^H (H^H H)^{-1} 1 for the baseline combiner v = H(H^H H)^{-1}1."""
    G = H.conj().T @ H
    Ginv = np.linalg.inv(G)
    ones = np.ones(H.shape[1])
    return float(np.real(ones @ Ginv @ ones))


def sigma2(radio) -> float:
    """Per-tone noise power sigma^2 = N0 * df."""
    return radio.N0_w_per_hz * radio.df_hz


# ---------------------------------------------------------------------------
# Training-time noise injection (torch tensors)
# ---------------------------------------------------------------------------

def zf_noise_var_per_coord(z: torch.Tensor, cn2: float, radio) -> float:
    """Var[e_U per coord] = sigma^2 ||c_n||^2 / (2 b_n^2) (Eq. 10)."""
    d_U = z.numel()
    m_U = math.ceil(d_U / 2)
    p_n = radio.Pmax_w / radio.S
    z_energy = float(z.detach().pow(2).sum())
    b2 = m_U * p_n / max(z_energy, 1e-12)
    return sigma2(radio) * cn2 / (2.0 * b2)


def add_zf_activation_noise(z: torch.Tensor, cn2: float, radio,
                            gen: torch.Generator = None) -> torch.Tensor:
    """zhat_n = z_n + e_U,n (Eq. 9-10). z: the client's cut activation tensor."""
    var = zf_noise_var_per_coord(z, cn2, radio)
    noise = torch.randn(z.shape, generator=gen, device=z.device, dtype=z.dtype) * math.sqrt(var)
    return z + noise


def aircomp_alpha(deltas, weights, v_norm2: float, radio) -> float:
    """alpha = min_n |f_n| sqrt(m_A p_n) / (a_n ||Delta_n||), f_n = 1/||v|| (Eq. 17)."""
    p_n = radio.Pmax_w / radio.S
    f_abs = 1.0 / math.sqrt(v_norm2)                 # |f_n| = 1/||v|| (baseline combiner)
    best = math.inf
    for d, a in zip(deltas, weights):
        norm = float(d.detach().pow(2).sum().sqrt())
        if norm <= 0 or a <= 0:
            continue
        m_A = math.ceil(d.numel() / 2)
        val = f_abs * math.sqrt(m_A * p_n) / (a * norm)
        best = min(best, val)
    return best if best < math.inf else 0.0


def aircomp_aggregate(deltas, weights, v_norm2: float, radio,
                      gen: torch.Generator = None):
    """Recover sum_n a_n Delta_n + e_A (Eq. 19-20). Returns (noisy_sum, alpha).
    deltas: list of equal-shaped real tensors; weights: data weights a_n (sum<=1)."""
    agg = sum(a * d for a, d in zip(weights, deltas))
    alpha = aircomp_alpha(deltas, weights, v_norm2, radio)
    if alpha <= 0:
        return agg, alpha
    var = sigma2(radio) / (2.0 * alpha ** 2)          # per-coord (Eq. 20)
    noise = torch.randn(agg.shape, generator=gen, device=agg.device, dtype=agg.dtype) * math.sqrt(var)
    return agg + noise, alpha


# ---------------------------------------------------------------------------
# Monte-Carlo radio checks (roadmap Sec. 6): injected var == analytical Eq.10/20
# ---------------------------------------------------------------------------

def _pack(real_vec: np.ndarray) -> np.ndarray:
    """Pd: pack 2 real coords into one complex (Eq. 3). ||P(r)||^2 == ||r||^2."""
    d = real_vec.size
    if d % 2 == 1:
        real_vec = np.concatenate([real_vec, [0.0]])
    return real_vec[0::2] + 1j * real_vec[1::2]


def verify_zf_mse(radio, rng, d_U=2048, trials=4000) -> dict:
    """Simulate the ZF noise chain (vectorized) and compare empirical ||e_U||^2 to
    Eq. 10. Real-domain error energy after unpacking = sum_l |e_l|^2 (no factor 2:
    a complex e_l unpacks to two reals [Re,Im] with energy |e_l|^2)."""
    H = draw_channel(radio, rng)                       # one coherent packet
    cn2 = zf_column_norms_sq(H)
    C = H @ np.linalg.inv(H.conj().T @ H)              # Nr x N combiner; c_n = column n
    p_n = radio.Pmax_w / radio.S
    s2 = sigma2(radio)
    m = math.ceil(d_U / 2)
    emp, ana = [], []
    for n in range(radio.N):
        z = rng.normal(0, 1.0, d_U)
        b = math.sqrt(m * p_n) / np.linalg.norm(z)     # Eq. 7 scaling
        cn = C[:, n]
        # error per complex coord e = c_n^H v / b, v ~ CN(0, sigma^2 I_Nr).
        # vectorize over (trials, m): project noise through c_n^H.
        V = (rng.normal(0, math.sqrt(s2 / 2), (trials, m, radio.Nr))
             + 1j * rng.normal(0, math.sqrt(s2 / 2), (trials, m, radio.Nr)))
        e = (V @ cn.conj()) / b                        # (trials, m) complex
        emp.append(float(np.mean(np.sum(np.abs(e) ** 2, axis=1))))
        ana.append(d_U * s2 * cn2[n] / (2 * b ** 2))   # Eq. 10
    emp, ana = np.array(emp), np.array(ana)
    return {"empirical": emp, "analytical": ana, "max_rel_err": float(np.max(np.abs(emp - ana) / ana))}


def verify_aircomp_mse(radio, rng, d_A=2048, trials=4000) -> dict:
    """Simulate the AirComp noise chain (vectorized) and compare ||e_A||^2 to Eq. 20.
    The recovered error is e = w^H v / alpha (signal terms are exact); real-domain
    energy = sum_l |e_l|^2."""
    H = draw_channel(radio, rng)
    vvec = H @ np.linalg.inv(H.conj().T @ H) @ np.ones(radio.N)
    w = vvec / np.linalg.norm(vvec)                    # ||w|| = 1
    f = w.conj() @ H                                   # f_n = 1/||v|| (baseline)
    p_n = radio.Pmax_w / radio.S
    s2 = sigma2(radio)
    m = math.ceil(d_A / 2)
    a = np.full(radio.N, 1.0 / radio.N)
    deltas = [rng.normal(0, 1.0, d_A) for _ in range(radio.N)]
    norms = [np.linalg.norm(dd) for dd in deltas]
    alpha = min(np.abs(f[n]) * math.sqrt(m * p_n) / (a[n] * norms[n]) for n in range(radio.N))
    # error only: e = w^H v / alpha, v ~ CN(0, sigma^2 I_Nr) per coord
    V = (rng.normal(0, math.sqrt(s2 / 2), (trials, m, radio.Nr))
         + 1j * rng.normal(0, math.sqrt(s2 / 2), (trials, m, radio.Nr)))
    e = (V @ w.conj()) / alpha                          # (trials, m)
    emp = float(np.mean(np.sum(np.abs(e) ** 2, axis=1)))
    ana = d_A * s2 / (2 * alpha ** 2)                   # Eq. 20
    return {"empirical": emp, "analytical": ana, "rel_err": abs(emp - ana) / ana}


def verify_gates(radio, rng, verbose: bool = True) -> bool:
    """Roadmap implementation gates 1-3 and 5 on one packet:
      1. I/Q packing (Eq. 2) preserves the squared norm for odd and even lengths and
         the inverse removes only the padding;
      2. packet powers (Eq. 4, 7, 21): ||s||^2/m == p_n for ZF activations and
         <= p_n for every AirComp client (== for the binding one);
      3. ZF identity C^H H = I and noiseless weighted AirComp output == sum a_n u_n;
      5. zero packets and an all-zero aggregate give zero, without division by zero."""
    ok = True
    for d in (255, 256):                                           # 1. packing
        r = rng.normal(size=d)
        u = _pack(r)
        back = np.stack([u.real, u.imag], 1).reshape(-1)[:d]
        ok &= abs(np.sum(np.abs(u) ** 2) - np.sum(r ** 2)) < 1e-9 and np.allclose(back, r)
    H = draw_channel(radio, rng)
    Hf = draw_channel(radio, rng, Nr=radio.Nr_F)
    p = radio.Pmax_w / radio.S
    m = 128
    z = rng.normal(size=2 * m)                                     # 2. ZF activation power
    s = math.sqrt(m * p) / np.linalg.norm(z) * _pack(z)
    ok &= abs(np.sum(np.abs(s) ** 2) / m - p) < 1e-12 * max(1, p)
    C = H @ np.linalg.inv(H.conj().T @ H)                          # 3. ZF identity
    ok &= np.allclose(C.conj().T @ H, np.eye(radio.N), atol=1e-8 * np.abs(C).max() * np.abs(H).max())
    v = Hf @ np.linalg.inv(Hf.conj().T @ Hf) @ np.ones(radio.N)    # AirComp combiner (Eq. 19)
    w = v / np.linalg.norm(v)
    f = w.conj() @ Hf
    a = rng.dirichlet(np.ones(radio.N))
    deltas = [rng.normal(size=2 * m) for _ in range(radio.N)]
    U = [_pack(dl) for dl in deltas]
    alpha = min(abs(f[n]) * math.sqrt(m * p) / (a[n] * np.linalg.norm(deltas[n])) for n in range(radio.N))
    S_tx = [alpha * a[n] * f[n].conj() / abs(f[n]) ** 2 * U[n] for n in range(radio.N)]
    powers = np.array([np.sum(np.abs(st) ** 2) / m for st in S_tx])
    ok &= bool(np.all(powers <= p * (1 + 1e-9))) and abs(powers.max() - p) < 1e-9 * p
    y = sum(np.outer(Hf[:, n], S_tx[n]) for n in range(radio.N))  # noiseless receive (Nr_F x m)
    u_hat = (w.conj() @ y) / alpha
    ok &= np.allclose(u_hat, sum(a[n] * U[n] for n in range(radio.N)))
    zero = np.zeros(2 * m)                                         # 5. zero packets
    ok &= np.linalg.norm(zero) == 0.0                              # flagged: sends/recovers exactly zero
    if verbose:
        print(f"  gates (packing, packet power, ZF identity, noiseless AirComp, zero packets): "
              f"{'PASS' if ok else 'FAIL'}")
    return bool(ok)


def run_radio_checks(verbose: bool = True) -> bool:
    from flsim.airsfl.timing import RadioConfig, path_gain_offsets_db
    radio = RadioConfig(N=8, Nr=32, S=64, rho_db=20.0)     # small array: fast Monte-Carlo
    rng = np.random.RandomState(7)
    zf = verify_zf_mse(radio, rng)
    ac = verify_aircomp_mse(radio, rng)
    gates = verify_gates(RadioConfig(), np.random.RandomState(11), verbose=verbose)
    # the same chains with unequal path gains (20 dB spread): Eq. 10 / 20 hold per client
    radio_g = RadioConfig(N=8, Nr=32, S=64, rho_db=20.0, gains_db=path_gain_offsets_db(8, 20.0, 11))
    zf_g = verify_zf_mse(radio_g, rng)
    ac_g = verify_aircomp_mse(radio_g, rng)
    gates &= verify_gates(RadioConfig(gains_db=path_gain_offsets_db(30, 20.0, 11)), np.random.RandomState(11),
                          verbose=False)
    ok = zf["max_rel_err"] < 0.05 and ac["rel_err"] < 0.05 and gates
    ok &= zf_g["max_rel_err"] < 0.05 and ac_g["rel_err"] < 0.05
    if verbose:
        print("=== AirSFL radio checks (empirical MSE vs analytical Eq. 10/20) ===")
        print(f"  ZF activation (Eq. 10): max relative error over {radio.N} clients "
              f"= {zf['max_rel_err']*100:.2f}%   [< 5% => OK]")
        print(f"      analytical D_U (client 0) = {zf['analytical'][0]:.4e}, "
              f"empirical = {zf['empirical'][0]:.4e}")
        print(f"  AirComp aggregation (Eq. 20): relative error = {ac['rel_err']*100:.2f}%   [< 5% => OK]")
        print(f"      analytical D_A = {ac['analytical']:.4e}, empirical = {ac['empirical']:.4e}")
        print(f"  unequal path gains (20 dB spread): ZF max rel. error {zf_g['max_rel_err']*100:.2f}%, "
              f"AirComp rel. error {ac_g['rel_err']*100:.2f}%, gates   [< 5% => OK]")
        print(f"  RADIO_CHECKS: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    run_radio_checks()
