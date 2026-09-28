"""
flsim/airsfl/simulator.py: full-participation, equal-period SFL-V1 training for
AirSFL and its four baselines on ONE matched environment (AirSFL draft Sec. II-III;
evaluation roadmap Sec. 1-5).

Two servers. The M-server (Nr antennas) receives activations and keeps one suffix
copy per client; the F-server (Nr_F antennas) receives prefix (or full-model)
differences. Both downlinks (cut derivatives, prefix broadcast) are ideal.

  method              split?  activation uplink (M-server)   aggregation uplink (F-server)
  airsfl              yes     analog ZF   (+ Eq. 10 error)   analog AirComp (+ Eq. 23 error)
  digital_sflv1       yes     digital OFDMA (reliable)       digital OFDMA (exact)
  sun_fdma_aircomp    yes     digital OFDMA (reliable)       analog AirComp (+ Eq. 23 error)
  aircomp_fl          no      -                              analog AirComp of the FULL model
  digital_fedavg      no      -                              digital OFDMA, full model (exact)

All methods share data, init, model, minibatches, (optional) augmentation, plain
SGD, the LR schedule, tau, participation, weights a_n = D_n/sum D, bandwidth,
power and antennas. Suffix copies stay independent for tau steps and are averaged
exactly at the round boundary (Eq. 24).

Training step of a split method (Eq. 6, 9, 14-17): with zhat = z + e, e detached,
    q = grad_z Phi(zhat), g_s = grad_xs Phi(zhat), g_c = grad_xc <q, h_c(x_c)>.
Two engines compute exactly this:
  "loop"  per client: relay = zhat.requires_grad_(); loss.backward(); z.backward(relay.grad)
  "vmap"  all clients at once (torch.func): one forward/backward of
          Phi(h_c(x_c) + e_detached) -- the additive detached error has identity
          Jacobian w.r.t. z, so the prefix receives exactly q (straight-through).
The two agree to float precision (flsim.airsfl.checks). With exact transport the
update is full-model SGD, so noiseless AirSFL == digital SFL-V1 == digital FedAvg.

Noise (scale-free form of Eq. 10 / 23 with H~ = H/sqrt(lambda) ~ CN(0, I) and
sigma^2/(lambda p_n) = 1/rho):
  ZF, per real coord of packet k of client n:  [G~^-1]_nn ||z_nk||^2 / (2 rho m_k),
      sampled JOINTLY across clients (Cov = G~^-1): errors are correlated through the
      shared M-server noise.
  AirComp, per real coord of packet k: 1^T G~^-1 1 * max_n a_n^2 ||Delta_nk||^2 / (2 rho m_k)
      (baseline combiner v = H(H^H H)^-1 1, alpha_k = min over clients, Eq. 19-20).
Packets: 128 complex payload symbols (256 real coords) per coherent per-tone packet,
i.i.d. Rayleigh across packets; zero packets carry no error. The M-link and F-link
use SEPARATE, seeded channel/noise streams, so every analog-aggregation method
sees the same F-link trace for a given seed (paired comparison).

Metric: modeled uplink communication seconds (Eq. 25-26), accumulated per round
and logged per phase (activation / labels / aggregation). Computation excluded.
"""

import copy
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from flsim.airsfl.model import CifarResNet18GN
from flsim.airsfl.timing import (_ALIASES, RadioConfig, digital_rates, source_equivalent_mb_per_round,
                                 uplink_time_breakdown)

try:
    from torch.func import functional_call, grad_and_value, vmap
    _HAS_FUNC = True
except ImportError:                      # torch < 2.0
    _HAS_FUNC = False

PACKET_REAL = 256          # 128 complex payload symbols per coherent per-tone packet
_CHUNK = 1024              # packets per channel-sampling chunk (memory bound)
SCHEMA = 3                 # CSV schema version (plots ignore older files)

METHOD_SPEC = {
    "airsfl":           {"split": True,  "zf_act": True,  "aircomp": True},
    "digital_sflv1":    {"split": True,  "zf_act": False, "aircomp": False},
    "sun_fdma_aircomp": {"split": True,  "zf_act": False, "aircomp": True},
    "aircomp_fl":       {"split": False, "zf_act": False, "aircomp": True},
    "digital_fedavg":   {"split": False, "zf_act": False, "aircomp": False},
}

METHOD_LABELS = {
    "airsfl": "AirSFL",
    "digital_sflv1": "Digital SFL-V1",
    "sun_fdma_aircomp": "FDMA-AirComp SFL (Sun-inspired)",
    "aircomp_fl": "AirComp-FL",
    "digital_fedavg": "Digital FedAvg",
}


@dataclass
class RunConfig:
    method: str
    stage: Optional[int] = 2        # cut after residual stage 1..4; None = no split
    tau: int = 5                    # local co-split SGD steps per round
    batch_size: int = 16
    lr: float = 0.1                 # initial learning rate
    lr_schedule: str = "cosine"     # "cosine" (to lr_min_frac*lr) | "step" | "const"
    lr_min_frac: float = 0.01
    lr_decay_fracs: tuple = (0.5, 0.75)   # "step" schedule only
    lr_decay: float = 0.1
    warmup_rounds: int = 0          # optional linear warmup (same for every method)
    rounds: int = 100
    evals_per_epoch: float = 1.0    # evaluate at the first round boundary after each 1/x epoch
    noiseless: bool = False         # exact transport (analog errors switched off)
    seed: int = 11                  # model init + radio streams (data streams seeded outside)
    engine: str = "vmap"            # "vmap" (all clients in parallel) | "loop" (reference)
    vmap_chunk: Optional[int] = None  # clients per vmap chunk (memory), None = all


def _prefix_param_names(model: CifarResNet18GN, stage: Optional[int]):
    names = [n for n, _ in model.named_parameters()]
    if stage is None:
        return names
    keep = ("stem.",) + tuple(f"layer{i}." for i in range(1, stage + 1))
    return [n for n in names if n.startswith(keep)]


def _packet_lengths(d: int, K: int, device) -> torch.Tensor:
    lens = torch.full((K,), PACKET_REAL, dtype=torch.float64, device=device)
    lens[-1] = d - (K - 1) * PACKET_REAL
    return lens


class AirSFLSimulator:
    def __init__(self, cfg: RunConfig, radio: RadioConfig, streams: list, weights,
                 eval_sets: dict, device, n_train: int, log=print):
        m = _ALIASES.get(cfg.method.lower())
        if m not in METHOD_SPEC:
            raise ValueError(f"unknown method {cfg.method!r}")
        if len(streams) != radio.N:
            raise ValueError(f"{len(streams)} client streams but radio.N = {radio.N}")
        if radio.N > radio.Nr:
            raise ValueError(f"ZF needs N <= Nr at the M-server (got N={radio.N}, Nr={radio.Nr})")
        if radio.N > radio.Nr_F:
            raise ValueError(f"the baseline AirComp combiner needs N <= Nr_F (got N={radio.N}, Nr_F={radio.Nr_F})")
        if radio.N > radio.S:
            raise ValueError(f"digital OFDMA needs N <= S (got S={radio.S}, N={radio.N})")
        spec = METHOD_SPEC[m]
        self.method, self.cfg, self.radio = m, cfg, radio
        self.device, self.log, self.n_train = device, log, n_train
        self.engine = cfg.engine
        if self.engine == "vmap" and not _HAS_FUNC:
            log("[AirSFL] torch.func unavailable -> using the loop engine")
            self.engine = "loop"
        self.split = spec["split"] and cfg.stage is not None
        self.stage = cfg.stage if self.split else None
        self.zf_act = spec["zf_act"] and self.split and not cfg.noiseless
        self.aircomp = spec["aircomp"] and not cfg.noiseless
        self.streams, self.eval_sets = streams, eval_sets
        self.N = radio.N
        self.a = torch.tensor(np.asarray(weights, dtype=np.float64), dtype=torch.float32, device=device)

        # identical initialization across methods
        torch.manual_seed(cfg.seed)
        self.global_model = CifarResNet18GN(radio.num_classes).no_inplace().to(device)
        self.names = [n for n, _ in self.global_model.named_parameters()]
        pref = set(_prefix_param_names(self.global_model, self.stage))
        self.pref_idx = [i for i, n in enumerate(self.names) if n in pref]
        self.suf_idx = [i for i, n in enumerate(self.names) if n not in pref]
        self.gparams = list(self.global_model.parameters())
        if self.engine == "loop":
            self.clients = [copy.deepcopy(self.global_model) for _ in range(self.N)]
            self.cparams = [list(c.parameters()) for c in self.clients]
        else:
            self.fmodel = copy.deepcopy(self.global_model)       # structure for functional_call
            self.stacked = {n: p.detach().unsqueeze(0).repeat(self.N, *([1] * p.dim())).contiguous()
                            for n, p in self.global_model.named_parameters()}
            self.cparams = [[self.stacked[n][k] for n in self.names] for k in range(self.N)]   # views

        # radio streams, independent of the data streams: M-link (ZF) and F-link (AirComp)
        self.rdev = device if device.type == "cuda" else torch.device("cpu")
        self.gen_U = torch.Generator(device=self.rdev)
        self.gen_U.manual_seed(int(cfg.seed) * 7919 + 17)
        self.gen_A = torch.Generator(device=self.rdev)
        self.gen_A.manual_seed(int(cfg.seed) * 7919 + 29)

        # dimensions (measured from the actual model)
        d_c = sum(self.gparams[i].numel() for i in self.pref_idx)
        d_s = sum(self.gparams[i].numel() for i in self.suf_idx)
        d_a = 0
        if self.split:
            with torch.no_grad():
                z = self._prefix(self.global_model, torch.zeros(cfg.batch_size, 3, 32, 32, device=device))
            d_a = int(z.numel())
            self.K_act = math.ceil(d_a / PACKET_REAL)
            self._act_m = torch.ceil(_packet_lengths(d_a, self.K_act, device) / 2.0).float()
        self.dims = {"d_c": d_c, "d_s": d_s, "d_a": d_a}

        # timing constants (fluid goodput estimated once, fixed seed -> identical across methods)
        radio.batch_size = cfg.batch_size
        self.rates = digital_rates(radio, seed=2026)
        self.ul_break = uplink_time_breakdown(m, self.dims, radio, cfg.tau, self.rates)
        self.ul_s = self.ul_break["total"]
        self.mb = source_equivalent_mb_per_round(m, self.dims, radio, cfg.tau)
        self.history = []
        self._reset_stats()

    # ------------------------------------------------------------------
    # model halves (loop engine / dimension probe)
    # ------------------------------------------------------------------

    def _prefix(self, M, x):
        x = M.stem(x)
        for L in M.stages()[: self.stage]:
            x = L(x)
        return x

    def _suffix(self, M, z):
        for L in M.stages()[self.stage:]:
            z = L(z)
        return M.fc(M.pool(z).flatten(1))

    # ------------------------------------------------------------------
    # radio: ZF activation error (joint across clients) + AirComp error
    # ------------------------------------------------------------------

    def _sample_zf_joint(self) -> torch.Tensor:
        """Whitened joint ZF error for one co-split step, real-interleaved:
        returns (N, K, 256); per packet k and complex symbol l the N clients' errors
        are CN(0, G~inv_k) (shared M-server noise), before per-client amplitude scaling."""
        K, N, Nr = self.K_act, self.N, self.radio.Nr
        out = torch.empty((K, PACKET_REAL // 2, N), dtype=torch.complex64, device=self.rdev)
        for s in range(0, K, _CHUNK):
            k = min(_CHUNK, K - s)
            H = torch.randn((k, Nr, N), dtype=torch.complex128, generator=self.gen_U, device=self.rdev)
            Ginv = torch.linalg.inv(H.conj().transpose(-1, -2) @ H)
            Ginv = 0.5 * (Ginv + Ginv.conj().transpose(-1, -2))
            L = torch.linalg.cholesky(Ginv)
            w = torch.randn((k, PACKET_REAL // 2, N), dtype=torch.complex128, generator=self.gen_U,
                            device=self.rdev)
            out[s:s + k] = torch.einsum("kij,klj->kli", L, w).to(torch.complex64)
        # client-major, real/imag interleaved per symbol (inverse of the packing Eq. 2)
        return torch.view_as_real(out.permute(2, 0, 1).contiguous()).reshape(N, K, PACKET_REAL).to(self.device)

    def _zf_noise(self, z: torch.Tensor, Er: torch.Tensor) -> torch.Tensor:
        """Client n's real reconstruction error for its activation z (Eq. 9-10).
        z is detached by the caller; Er: (K, 256) whitened joint error of this client."""
        flat = z.reshape(-1)
        d = flat.numel()
        K = Er.shape[0]
        pk = F.pad(flat, (0, K * PACKET_REAL - d)).reshape(K, PACKET_REAL)
        scale = pk.norm(dim=1) / torch.sqrt(self.radio.rho_lin * self._act_m)   # zero packet -> 0
        return (Er * scale[:, None]).reshape(-1)[:d].reshape(z.shape)

    def _aircomp_noise(self, maxterm: torch.Tensor, d: int) -> torch.Tensor:
        """F-server AirComp error on the recovered weighted sum (Eq. 22-23)."""
        K, N, Nr = maxterm.numel(), self.N, self.radio.Nr_F
        v2 = torch.empty(K, dtype=torch.float64, device=self.rdev)
        for s in range(0, K, _CHUNK):
            k = min(_CHUNK, K - s)
            H = torch.randn((k, Nr, N), dtype=torch.complex128, generator=self.gen_A, device=self.rdev)
            Ginv = torch.linalg.inv(H.conj().transpose(-1, -2) @ H)
            v2[s:s + k] = Ginv.sum(dim=(-1, -2)).real                 # ||v~||^2 = 1^T G~inv 1
        m = torch.ceil(_packet_lengths(d, K, self.rdev) / 2.0)
        var = v2 * maxterm.to(self.rdev, torch.float64) / (2.0 * self.radio.rho_lin * m)
        noise = torch.randn((K, PACKET_REAL), generator=self.gen_A, device=self.rdev) \
            * torch.sqrt(var).to(torch.float32)[:, None]
        return noise.reshape(-1)[:d].to(self.device)

    # ------------------------------------------------------------------
    # one co-split step for all clients (Eq. 6, 9, 14-17)
    # ------------------------------------------------------------------

    def _local_step_loop(self, n: int, x, y, lr: float, Er):
        M = self.clients[n]
        for p in M.parameters():
            p.grad = None
        if self.split:
            z = self._prefix(M, x)
            zt = z.detach().clone()
            if Er is not None:
                noise = self._zf_noise(zt, Er)
                self._act_noise_e += noise.pow(2).sum()
                self._act_sig_e += zt.pow(2).sum()
                self._act_coords += zt.numel()
                zt = zt + noise
            relay = zt.requires_grad_(True)
            loss = F.cross_entropy(self._suffix(M, relay), y)
            loss.backward()                      # suffix gradient (15) and cut derivative q (14)
            z.backward(relay.grad)               # prefix gradient through the retained graph (16)
        else:
            loss = F.cross_entropy(M(x), y)
            loss.backward()
        with torch.no_grad():
            for p in M.parameters():
                p.add_(p.grad, alpha=-lr)        # (17)
        return loss.detach()

    def _loss_clean(self, params, x, y):
        return F.cross_entropy(functional_call(self.fmodel, params, (x,)), y)

    def _loss_noisy(self, params, x, y, Er):
        stats = {}

        def perturb(z):
            zd = z.detach()
            e = self._zf_noise(zd, Er)
            stats["ne"], stats["se"] = e.pow(2).sum(), zd.pow(2).sum()
            return z + e                         # identity Jacobian w.r.t. z: prefix receives q exactly

        logits = functional_call(self.fmodel, params, (x,), {"cut": self.stage, "perturb": perturb})
        return F.cross_entropy(logits, y), (stats["ne"], stats["se"])

    def _step_vmap(self, X, Y, Er, lr: float):
        if Er is None:
            fn = vmap(grad_and_value(self._loss_clean), in_dims=(0, 0, 0), chunk_size=self.cfg.vmap_chunk)
            g, loss = fn(self.stacked, X, Y)
        else:
            fn = vmap(grad_and_value(self._loss_noisy, has_aux=True), in_dims=(0, 0, 0, 0),
                      chunk_size=self.cfg.vmap_chunk)
            g, (loss, (ne, se)) = fn(self.stacked, X, Y, Er)
            self._act_noise_e += ne.sum()
            self._act_sig_e += se.sum()
            self._act_coords += self.dims["d_a"] * self.N
        with torch.no_grad():
            for k, t in self.stacked.items():
                t.add_(g[k], alpha=-lr)          # (17)
        return loss.sum()

    # ------------------------------------------------------------------
    # round-boundary aggregation (Eq. 18, 22, 24)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _aggregate(self):
        for idx, air in ((self.pref_idx, self.aircomp), (self.suf_idx, False)):
            if not idx:
                continue
            g = torch.cat([self.gparams[i].reshape(-1) for i in idx])
            d = g.numel()
            K = math.ceil(d / PACKET_REAL)
            agg = torch.zeros_like(g)
            maxterm = torch.zeros(K, dtype=torch.float32, device=g.device) if air else None
            for n in range(self.N):
                delta = torch.cat([self.cparams[n][i].reshape(-1) for i in idx]) - g   # Eq. 18
                agg.add_(delta, alpha=float(self.a[n]))
                if air:
                    pk = F.pad(delta, (0, K * PACKET_REAL - d)).view(K, PACKET_REAL)
                    maxterm = torch.maximum(maxterm, (self.a[n] ** 2) * pk.pow(2).sum(dim=1))
            if air:
                noise = self._aircomp_noise(maxterm, d)
                self._agg_noise_e += noise.pow(2).sum()
                self._agg_sig_e += agg.pow(2).sum()
                self._agg_coords += d
                agg = agg + noise
            new = g + agg                                                  # Eq. 24 (no extra stepsize)
            off = 0
            for i in idx:
                k = self.gparams[i].numel()
                self.gparams[i].copy_(new[off:off + k].view_as(self.gparams[i]))
                off += k

    @torch.no_grad()
    def _broadcast(self):
        """Round start (Eq. 5): every client prefix and suffix copy <- global blocks."""
        if self.engine == "loop":
            for n in range(self.N):
                for cp, gp in zip(self.cparams[n], self.gparams):
                    cp.copy_(gp)
        else:
            for name, gp in zip(self.names, self.gparams):
                self.stacked[name].copy_(gp.unsqueeze(0).expand_as(self.stacked[name]))

    # ------------------------------------------------------------------
    # evaluation + main loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict:
        M = self.global_model
        out = {}
        for name, (x, y) in self.eval_sets.items():
            correct, loss_sum = 0, 0.0
            for s in range(0, x.shape[0], 500):
                logits = M(x[s:s + 500])
                loss_sum += float(F.cross_entropy(logits, y[s:s + 500], reduction="sum"))
                correct += int((logits.argmax(1) == y[s:s + 500]).sum())
            out[f"{name}_acc"] = correct / x.shape[0]
            out[f"{name}_loss"] = loss_sum / x.shape[0]
        return out

    def _lr_at(self, r: int) -> float:
        cfg = self.cfg
        if cfg.lr_schedule == "cosine":           # progress in global-epoch equivalents = r / R
            c = 0.5 * (1.0 + math.cos(math.pi * r / cfg.rounds))
            lr = cfg.lr * (cfg.lr_min_frac + (1.0 - cfg.lr_min_frac) * c)
        elif cfg.lr_schedule == "step":
            drops = sum(r >= int(f * cfg.rounds) for f in cfg.lr_decay_fracs)
            lr = cfg.lr * (cfg.lr_decay ** drops)
        else:
            lr = cfg.lr
        if cfg.warmup_rounds > 0:
            lr *= min(1.0, (r + 1) / cfg.warmup_rounds)
        return lr

    def _reset_stats(self):
        z = lambda: torch.zeros((), device=self.device)
        self._act_noise_e, self._act_sig_e, self._agg_noise_e, self._agg_sig_e = z(), z(), z(), z()
        self._act_coords = self._agg_coords = 0

    def _record(self, r: int, lr: float, train_loss: float, t0: float, diverged: bool = False):
        ane, ase, gne, gse = (float(t) for t in (self._act_noise_e, self._act_sig_e,
                                                 self._agg_noise_e, self._agg_sig_e))
        db = lambda n, s: 10 * math.log10(n / s) if n > 0 and s > 0 else float("nan")
        spr = self.N * self.cfg.tau * self.cfg.batch_size
        rec = {
            "schema": SCHEMA, "method": self.method, "label": METHOD_LABELS[self.method],
            "cut": self.stage if self.stage is not None else 0, "tau": self.cfg.tau, "B": self.cfg.batch_size,
            "rho_db": self.radio.rho_db, "N": self.N, "Nr": self.radio.Nr, "Nr_F": self.radio.Nr_F,
            "S": self.radio.S, "eps_D": self.radio.eps_D, "eps_U": self.radio.eps_U, "eps_A": self.radio.eps_A,
            "seed": self.cfg.seed, "noiseless": self.cfg.noiseless, "base_lr": self.cfg.lr,
            "lr_schedule": self.cfg.lr_schedule, "engine": self.engine,
            "round": r, "processed_samples": r * spr, "epoch_equiv": r * spr / self.n_train,
            "lr": lr, "train_loss": train_loss, "diverged": diverged,
            "activation_ul_s": self.ul_break["activation"], "labels_ul_s": self.ul_break["labels"],
            "aggregation_ul_s": self.ul_break["aggregation"], "ul_s_per_round": self.ul_s,
            "uplink_s": r * self.ul_s, "cumulative_ul_s": r * self.ul_s,
            "mb_per_round": self.mb, "source_mb": r * self.mb,
            "activation_mse": ane / self._act_coords if self._act_coords else float("nan"),
            "aggregation_mse": gne / self._agg_coords if self._agg_coords else float("nan"),
            "act_nsr_db": db(ane, ase), "agg_nsr_db": db(gne, gse),
            "d_a": self.dims["d_a"], "d_c": self.dims["d_c"], "d_s": self.dims["d_s"],
            "wall_s": time.time() - t0,
        }
        rec.update(self.evaluate())
        self.history.append(rec)
        self._reset_stats()
        return rec

    def run(self) -> list:
        cfg, t0 = self.cfg, time.time()
        self.log(f"[AirSFL] {METHOD_LABELS[self.method]} | cut={self.stage} | rho={self.radio.rho_db:g} dB | "
                 f"N={self.N} Nr_M={self.radio.Nr} Nr_F={self.radio.Nr_F} S={self.radio.S} | "
                 f"tau={cfg.tau} B={cfg.batch_size} | rounds={cfg.rounds} | lr={cfg.lr:g} ({cfg.lr_schedule}) | "
                 f"UL {self.ul_s:.3f} s/rnd (act {self.ul_break['activation']:.3f}, labels "
                 f"{self.ul_break['labels']:.4f}, agg {self.ul_break['aggregation']:.3f}) | "
                 f"engine={self.engine} | noiseless={cfg.noiseless}")
        self._record(0, self._lr_at(0), float("nan"), t0)
        spr = self.N * cfg.tau * cfg.batch_size
        eval_step = self.n_train / cfg.evals_per_epoch
        next_eval = eval_step
        for r in range(cfg.rounds):
            lr = self._lr_at(r)                  # fixed within a round
            self._broadcast()
            loss_sum = torch.zeros((), device=self.device)
            for _ in range(cfg.tau):
                E = self._sample_zf_joint() if self.zf_act else None       # one per step, all clients
                batches = [s.next_batch() for s in self.streams]
                if self.engine == "loop":
                    for n, (x, y) in enumerate(batches):
                        loss_sum += self._local_step_loop(n, x, y, lr, E[n] if E is not None else None)
                else:
                    X = torch.stack([b[0] for b in batches])
                    Y = torch.stack([b[1] for b in batches])
                    loss_sum += self._step_vmap(X, Y, E, lr)
            self._aggregate()
            if not bool(torch.isfinite(loss_sum)):
                # training blew up (e.g. very low SNR): record the failure and stop; the run
                # counts as "target not reached" and is not retried on resume
                self._record(r + 1, lr, float("nan"), t0, diverged=True)
                self.log(f"  rnd {r+1:5d} DIVERGED (non-finite training loss) -> stopping this run")
                break
            processed = (r + 1) * spr
            if processed >= next_eval or r + 1 == cfg.rounds:
                while next_eval <= processed:
                    next_eval += eval_step
                rec = self._record(r + 1, lr, float(loss_sum) / (cfg.tau * self.N), t0)
                self.log(f"  rnd {r+1:5d} ep {rec['epoch_equiv']:6.2f} | lr {lr:.4f} | loss {rec['train_loss']:.3f} | "
                         f"val {rec.get('val_acc', float('nan')):.4f} test {rec.get('test_acc', float('nan')):.4f} | "
                         f"UL {rec['uplink_s']:.1f}s | actNSR {rec['act_nsr_db']:.1f}dB aggNSR {rec['agg_nsr_db']:.1f}dB "
                         f"| wall {rec['wall_s']:.0f}s ({rec['wall_s'] / (r + 1):.2f} s/rnd)")
        return self.history

    def global_vector(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.gparams])
