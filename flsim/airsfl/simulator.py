"""
flsim/airsfl/simulator.py: full-participation SFL-V1 training for AirSFL and its
four baselines on ONE matched environment (AirSFL draft Sec. II-IV; roadmap Sec. 3).

Methods (all share data, init, model, minibatches, augmentation, plain SGD, tau,
participation, weights a_n = D_n/sum D, bandwidth, power and antennas):

  method              split?  activation transport       prefix/model aggregation
  airsfl              yes     analog ZF   (+ Eq.10 noise) analog AirComp (+ Eq.20 noise)
  digital_sflv1       yes     digital OFDMA (reliable)    exact (digital)
  sun_fdma_aircomp    yes     digital OFDMA (reliable)    analog AirComp (+ Eq.20 noise)
  aircomp_fl          no      -                           analog AirComp on FULL model
  digital_fedavg      no      -                           exact (digital)

In every split method the server keeps a SEPARATE suffix copy per client
(SFL-V1) and averages the copies EXACTLY at the round boundary (server-local).

Training step of a split method (Eq. 6, 9, 11-14):
    z = prefix(x);  zhat = z.detach() [+ ZF noise]      # noise detached from graph
    relay = zhat.requires_grad_();  loss = CE(suffix(relay), y)
    loss.backward()           -> suffix gradient (12) and cut derivative q = relay.grad (11)
    z.backward(q)             -> prefix gradient through the RETAINED forward graph (13)
    SGD step on both blocks at the pre-update pair (14)
With exact transport this is exactly full-model SGD, so noiseless AirSFL ==
digital SFL-V1 == digital FedAvg up to float order (verified in flsim.airsfl.checks).

Noise (scale-free form of Eq. 10/20 with normalized channels H~ = H/sqrt(lambda)
~ CN(0, I); sigma^2/lambda = Pmax/(rho*S) makes power and N0 cancel):
  ZF, per real coord of packet k of client n:  G~inv_nn * ||z_nk||^2 / (2 rho m_k),
      sampled JOINTLY across the N clients of a packet (Cov = G~inv), so the
      cross-client correlation of ZF errors through the shared receiver noise is kept.
  AirComp, per real coord of packet k:  1^T G~inv 1 * max_n a_n^2 ||Delta_nk||^2 / (2 rho m_k)
      (baseline combiner v = H(H^H H)^{-1}1, alpha_k = min over clients, Eq. 16-17).
Packets: 128 complex payload symbols (256 real coords) per coherent per-tone
packet, i.i.d. Rayleigh across packets (roadmap Sec. 4). Zero packets carry no noise.

Metric: modeled UPLINK communication seconds (Table I) accumulated per round.
Downlink (sensitivity) and source-equivalent MB are recorded alongside.
Computation time is excluded (paper Sec. II-C).
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
from flsim.airsfl.timing import (
    _ALIASES, RadioConfig, digital_spectral_efficiency, downlink_spectral_efficiency,
    downlink_time_per_round, source_equivalent_mb_per_round, uplink_time_breakdown,
    uplink_time_per_round,
)

PACKET_REAL = 256          # 128 complex payload symbols per coherent per-tone packet
_CHUNK = 1024              # packets per channel-sampling chunk (memory bound)

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
    "sun_fdma_aircomp": "FDMA-AirComp SFL (Sun-style)",
    "aircomp_fl": "AirComp-FL",
    "digital_fedavg": "Digital FedAvg",
}


@dataclass
class RunConfig:
    method: str
    stage: Optional[int] = 3        # cut after residual stage 1..4; None = no split
    tau: int = 5                    # local co-split SGD steps per round
    batch_size: int = 16
    lr: float = 0.03
    lr_decay_fracs: tuple = (0.5, 0.75)
    lr_decay: float = 0.1
    warmup_rounds: int = 0          # linear LR warmup (same schedule for every method)
    rounds: int = 100
    eval_every: int = 20
    noiseless: bool = False         # exact transport (analog noise switched off)
    seed: int = 11                  # model init + data streams + radio noise
    dl_scale: float = 1.0           # downlink goodput multiplier (sensitivity)


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
            raise ValueError(f"ZF needs N <= Nr (got N={radio.N}, Nr={radio.Nr})")
        # Digital OFDMA gives S_n = S/N tones per client. When N does not divide S the
        # fluid goodput (Eq. 22) is used with fractional S_n (tones time-shared across
        # resource blocks); per-tone SNR stays rho*N. At least one tone per client.
        if radio.N > radio.S:
            raise ValueError(f"digital OFDMA needs N <= S (got S={radio.S}, N={radio.N})")
        spec = METHOD_SPEC[m]
        self.method, self.cfg, self.radio = m, cfg, radio
        self.device, self.log, self.n_train = device, log, n_train
        self.split = spec["split"] and cfg.stage is not None
        self.stage = cfg.stage if self.split else None
        self.zf_act = spec["zf_act"] and self.split and not cfg.noiseless
        self.aircomp = spec["aircomp"] and not cfg.noiseless
        self.streams, self.eval_sets = streams, eval_sets
        self.N = radio.N
        self.a = torch.tensor(np.asarray(weights, dtype=np.float64), dtype=torch.float32, device=device)

        # identical initialization across methods
        torch.manual_seed(cfg.seed)
        self.global_model = CifarResNet18GN(radio.num_classes).to(device)
        self.clients = [copy.deepcopy(self.global_model) for _ in range(self.N)]
        names = [n for n, _ in self.global_model.named_parameters()]
        pref = set(_prefix_param_names(self.global_model, self.stage))
        self.pref_idx = [i for i, n in enumerate(names) if n in pref]
        self.suf_idx = [i for i, n in enumerate(names) if n not in pref]
        self.gparams = list(self.global_model.parameters())
        self.cparams = [list(c.parameters()) for c in self.clients]

        # radio RNG, independent of the data streams
        self.rdev = device if device.type == "cuda" else torch.device("cpu")
        self.rgen = torch.Generator(device=self.rdev)
        self.rgen.manual_seed(int(cfg.seed) * 7919 + 17)

        # dimensions (measured from the actual model)
        d_c = sum(self.gparams[i].numel() for i in self.pref_idx)
        d_s = sum(self.gparams[i].numel() for i in self.suf_idx)
        d_a = 0
        if self.split:
            with torch.no_grad():
                z = self._prefix(self.global_model, torch.zeros(cfg.batch_size, 3, 32, 32, device=device))
            d_a = int(z.numel())
            self.K_act = math.ceil(d_a / PACKET_REAL)
        self.dims = {"d_c": d_c, "d_s": d_s, "d_a": d_a}

        # timing constants: fluid long-term goodput estimated ONCE from a long channel
        # sample and reused (fixed seeds -> identical across matched methods)
        radio.batch_size = cfg.batch_size
        self.c_D = digital_spectral_efficiency(radio, np.random.RandomState(2026))
        self.c_DL = downlink_spectral_efficiency(radio, np.random.RandomState(2027))
        self.ul_s = uplink_time_per_round(m, self.dims, radio, cfg.tau, c_D=self.c_D)
        self.ul_break = uplink_time_breakdown(m, self.dims, radio, cfg.tau, c_D=self.c_D)
        self.dl_s = downlink_time_per_round(m, self.dims, radio, cfg.tau, c_DL=self.c_DL,
                                            dl_scale=cfg.dl_scale)
        self.mb = source_equivalent_mb_per_round(m, self.dims, radio, cfg.tau)
        self.history = []

    # ------------------------------------------------------------------
    # model halves
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
    # radio: ZF activation noise (joint across clients) + AirComp noise
    # ------------------------------------------------------------------

    def _sample_zf_joint(self) -> torch.Tensor:
        """Whitened joint ZF error for one co-split step: E[k, l, :] ~ CN(0, G~inv_k),
        k = packet, l = complex symbol (128), last dim = the N clients."""
        K, N, Nr = self.K_act, self.N, self.radio.Nr
        out = torch.empty((K, PACKET_REAL // 2, N), dtype=torch.complex64, device=self.rdev)
        for s in range(0, K, _CHUNK):
            k = min(_CHUNK, K - s)
            H = torch.randn((k, Nr, N), dtype=torch.complex128, generator=self.rgen, device=self.rdev)
            G = H.conj().transpose(-1, -2) @ H
            Ginv = torch.linalg.inv(G)
            Ginv = 0.5 * (Ginv + Ginv.conj().transpose(-1, -2))
            L = torch.linalg.cholesky(Ginv)
            w = torch.randn((k, PACKET_REAL // 2, N), dtype=torch.complex128, generator=self.rgen,
                            device=self.rdev)
            out[s:s + k] = torch.einsum("kij,klj->kli", L, w).to(torch.complex64)
        return out.to(self.device)

    def _zf_noise(self, z: torch.Tensor, En: torch.Tensor) -> torch.Tensor:
        """Client n's real reconstruction error for its activation z (Eq. 9-10)."""
        flat = z.reshape(-1)
        d = flat.numel()
        K = En.shape[0]
        pk = F.pad(flat, (0, K * PACKET_REAL - d)).view(K, PACKET_REAL)
        norms = pk.norm(dim=1)
        m = torch.ceil(_packet_lengths(d, K, z.device) / 2.0).to(z.dtype)
        scale = norms / torch.sqrt(self.radio.rho_lin * m)        # zero packet -> zero noise
        noise = torch.view_as_real(En).reshape(K, PACKET_REAL) * scale[:, None]
        return noise.reshape(-1)[:d].view_as(z)

    def _aircomp_noise(self, maxterm: torch.Tensor, d: int) -> torch.Tensor:
        """AirComp receiver noise on the recovered weighted sum (Eq. 19-20)."""
        K, N, Nr = maxterm.numel(), self.N, self.radio.Nr
        v2 = torch.empty(K, dtype=torch.float64, device=self.rdev)
        for s in range(0, K, _CHUNK):
            k = min(_CHUNK, K - s)
            H = torch.randn((k, Nr, N), dtype=torch.complex128, generator=self.rgen, device=self.rdev)
            Ginv = torch.linalg.inv(H.conj().transpose(-1, -2) @ H)
            v2[s:s + k] = Ginv.sum(dim=(-1, -2)).real                 # ||v~||^2 = 1^T G~inv 1
        m = torch.ceil(_packet_lengths(d, K, self.rdev) / 2.0)
        var = v2 * maxterm.to(self.rdev, torch.float64) / (2.0 * self.radio.rho_lin * m)
        noise = torch.randn((K, PACKET_REAL), generator=self.rgen, device=self.rdev) \
            * torch.sqrt(var).to(torch.float32)[:, None]
        return noise.reshape(-1)[:d].to(self.device)

    # ------------------------------------------------------------------
    # one local step (Eq. 6, 9, 11-14)
    # ------------------------------------------------------------------

    def _local_step(self, n: int, x, y, lr: float, En):
        M = self.clients[n]
        for p in M.parameters():
            p.grad = None
        if self.split:
            z = self._prefix(M, x)
            zt = z.detach().clone()
            if En is not None:
                noise = self._zf_noise(zt, En)
                self._act_noise_e += noise.pow(2).sum()
                self._act_sig_e += zt.pow(2).sum()
                zt = zt + noise
            relay = zt.requires_grad_(True)
            loss = F.cross_entropy(self._suffix(M, relay), y)
            loss.backward()
            z.backward(relay.grad)
        else:
            loss = F.cross_entropy(M(x), y)
            loss.backward()
        with torch.no_grad():
            for p in M.parameters():
                p.add_(p.grad, alpha=-lr)
        return loss.detach()

    # ------------------------------------------------------------------
    # round-boundary aggregation (Eq. 15, 19, 21)
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
                delta = torch.cat([self.cparams[n][i].reshape(-1) for i in idx]) - g   # Eq. 15
                agg.add_(delta, alpha=float(self.a[n]))
                if air:
                    pk = F.pad(delta, (0, K * PACKET_REAL - d)).view(K, PACKET_REAL)
                    maxterm = torch.maximum(maxterm, (self.a[n] ** 2) * pk.pow(2).sum(dim=1))
            if air:
                noise = self._aircomp_noise(maxterm, d)
                self._agg_noise_e += noise.pow(2).sum()
                self._agg_sig_e += agg.pow(2).sum()
                agg = agg + noise
            new = g + agg                                                  # Eq. 21
            off = 0
            for i in idx:
                k = self.gparams[i].numel()
                self.gparams[i].copy_(new[off:off + k].view_as(self.gparams[i]))
                off += k

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
        drops = sum(r >= int(f * self.cfg.rounds) for f in self.cfg.lr_decay_fracs)
        warm = min(1.0, (r + 1) / self.cfg.warmup_rounds) if self.cfg.warmup_rounds > 0 else 1.0
        return self.cfg.lr * warm * (self.cfg.lr_decay ** drops)

    def _record(self, r: int, lr: float, train_loss: float, act_nsr: float, agg_nsr: float, t0: float,
                diverged: bool = False):
        rec = {
            "method": self.method, "label": METHOD_LABELS[self.method],
            "stage": self.stage if self.stage is not None else 0,
            "rho_db": self.radio.rho_db, "N": self.N, "Nr": self.radio.Nr, "S": self.radio.S,
            "seed": self.cfg.seed, "noiseless": self.cfg.noiseless, "base_lr": self.cfg.lr,
            "round": r, "epoch_equiv": r * self.N * self.cfg.tau * self.cfg.batch_size / self.n_train,
            "lr": lr, "train_loss": train_loss, "diverged": diverged,
            "uplink_s": r * self.ul_s, "downlink_s": r * self.dl_s,
            "twoway_s": r * (self.ul_s + self.dl_s), "source_mb": r * self.mb,
            "act_nsr_db": 10 * math.log10(act_nsr) if act_nsr > 0 else float("nan"),
            "agg_nsr_db": 10 * math.log10(agg_nsr) if agg_nsr > 0 else float("nan"),
            "wall_s": time.time() - t0,
        }
        rec.update(self.evaluate())
        self.history.append(rec)
        return rec

    def run(self) -> list:
        cfg, t0 = self.cfg, time.time()
        self.log(f"[AirSFL] {METHOD_LABELS[self.method]} | stage={self.stage} | rho={self.radio.rho_db} dB | "
                 f"N={self.N} Nr={self.radio.Nr} S={self.radio.S} | tau={cfg.tau} B={cfg.batch_size} | "
                 f"rounds={cfg.rounds} | UL {self.ul_s:.3f} s/rnd, DL {self.dl_s:.3f} s/rnd, "
                 f"{self.mb:.1f} MB/rnd | noiseless={cfg.noiseless}")
        self._record(0, cfg.lr, float("nan"), 0.0, 0.0, t0)
        for r in range(cfg.rounds):
            lr = self._lr_at(r)
            with torch.no_grad():
                for n in range(self.N):
                    for cp, gp in zip(self.cparams[n], self.gparams):
                        cp.copy_(gp)
            self._act_noise_e = torch.zeros((), device=self.device)
            self._act_sig_e = torch.zeros((), device=self.device)
            self._agg_noise_e = torch.zeros((), device=self.device)
            self._agg_sig_e = torch.zeros((), device=self.device)
            loss_sum = torch.zeros((), device=self.device)
            for _ in range(cfg.tau):
                E = self._sample_zf_joint() if self.zf_act else None       # one step, all clients
                for n in range(self.N):
                    x, y = self.streams[n].next_batch()
                    loss_sum += self._local_step(n, x, y, lr, E[:, :, n] if E is not None else None)
            self._aggregate()
            if not bool(torch.isfinite(loss_sum)):
                # training blew up (e.g. very low SNR): record the failure and stop; the
                # run counts as "target not reached" and is not retried on resume
                self._record(r + 1, lr, float("nan"), 0.0, 0.0, t0, diverged=True)
                self.log(f"  rnd {r+1:5d} DIVERGED (non-finite training loss) -> stopping this run")
                break
            if (r + 1) % cfg.eval_every == 0 or r + 1 == cfg.rounds:
                act = float(self._act_noise_e / self._act_sig_e) if float(self._act_sig_e) > 0 else 0.0
                agg = float(self._agg_noise_e / self._agg_sig_e) if float(self._agg_sig_e) > 0 else 0.0
                rec = self._record(r + 1, lr, float(loss_sum) / (cfg.tau * self.N), act, agg, t0)
                self.log(f"  rnd {r+1:5d} ep {rec['epoch_equiv']:6.2f} | loss {rec['train_loss']:.3f} | "
                         f"val {rec.get('val_acc', float('nan')):.4f} test {rec.get('test_acc', float('nan')):.4f} | "
                         f"UL {rec['uplink_s']:.1f}s | actNSR {rec['act_nsr_db']:.1f}dB aggNSR {rec['agg_nsr_db']:.1f}dB "
                         f"| wall {rec['wall_s']:.0f}s")
        return self.history

    def global_vector(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.gparams])
