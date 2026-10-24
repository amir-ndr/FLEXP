"""
flsim/csasfl/engine.py: batched split co-training steps for all methods of the CSA-SFL study.

Models are kept as FLAT parameter vectors: device side w^c in R^{d_c}, server side w^s in
R^{d_s}; a group of n devices is an (n, d_c) matrix (one row per device-side submodel), so
aggregation is a weighted row sum and an "active prefix" of a group is a view, not a copy.

One lock-step co-training step of a group (paper Eq. server_update / device_update, and the
corresponding steps of the baselines): every active device k forward-propagates its mini-batch
B_k through its device-side submodel w_k^c, the server side finishes the forward pass, and

    g_k^s = (1/|B_k|) sum_{xi in B_k} grad_{w^s} l(w^s; w_k^c(xi))       (server-side gradient)
    g_k^c = (1/|B_k|) sum_{xi in B_k} grad_{w^c} l(w_k^c; xi)            (device-side gradient,
                                                                          BP of the returned
                                                                          smashed-data gradient)
    w_k^c <- w_k^c - eta g_k^c
    server side, by `kind`:
      "shared"     one server-side submodel for the group (CSA-SFL cluster):
                   w^s <- w^s - eta sum_k wts_k g_k^s    (wts_k = |D_k| / sum_active |D_j|)
      "grouped"    one server-side submodel per group, gidx maps devices to groups
                   (ParallelSFL clusters, wts_k = 1/N_c, Eq. 3)
      "per_device" every device trains its own server-side copy (FL's full model, SAFSL)
All g_k^s are taken at the SAME w^s (the PS computes the step's gradients, then updates once),
and the smashed-data gradient is taken at w^s before the update -- the standard relay.
step_sequential_server() is SFLv2: the server side processes the devices' smashed data one
device after another in a given order, updating after each.

Two engines compute exactly this:
  "vmap"  torch.func: vmap(grad(loss)) over the devices -- one batched forward/backward of the
          whole group (all devices' convolutions in one kernel launch); per-device
          server-side gradients come out stacked and are combined with `wts`.
  "loop"  reference: per device torch.autograd.grad (torch < 2.0, OOM fallback, checks).
Taking the gradient of the composed loss l(w^s; w^c(x)) w.r.t. (w^c, w^s) is exactly the split
relay (the smashed-data gradient is the chain-rule factor); flsim.csasfl.checks verifies
vmap == loop == relay (detach + backward) to float precision.
"""

import math

import torch
import torch.nn.functional as F

from flsim.csasfl.model import LeNet12, forward_layers, n_param_tensors

try:
    from torch.func import grad, grad_and_value, vjp, vmap
    _HAS_FUNC = True
except ImportError:                      # torch < 2.0 -> loop engine
    _HAS_FUNC = False


class Engine:
    def __init__(self, spec: list, cut: int, lr: float, device, engine: str = "vmap", chunk: int = None,
                 num_classes: int = 10):
        self.spec, self.cut, self.lr, self.device = spec, cut, float(lr), device
        self.num_classes = num_classes
        self.engine = engine if (engine == "loop" or _HAS_FUNC) else "loop"
        self.chunk = chunk
        m = LeNet12(num_classes)
        shapes = [tuple(p.shape) for p in m.parameters()]
        nd = n_param_tensors(spec, 0, cut)
        self.sd, self.ss = shapes[:nd], shapes[nd:]
        self.nd = [math.prod(s) for s in self.sd]
        self.ns = [math.prod(s) for s in self.ss]
        self.d_c, self.d_s = sum(self.nd), sum(self.ns)
        self._fns = {}

    # ------------------------------------------------------------------
    # parameters <-> flat vectors
    # ------------------------------------------------------------------

    def init_params(self, seed: int):
        """Identical initialization for every method of a seed: (w^{c,0}, w^{s,0}) flat."""
        torch.manual_seed(int(seed))
        m = LeNet12(self.num_classes)
        flat = torch.cat([p.detach().reshape(-1) for p in m.parameters()]).to(self.device)
        return flat[:self.d_c].clone(), flat[self.d_c:].clone()

    @staticmethod
    def _views(flat, shapes, numels):
        out, o = [], 0
        for s, n in zip(shapes, numels):
            out.append(flat[o:o + n].view(s))
            o += n
        return out

    @staticmethod
    def _views2d(flat2d, shapes, numels):
        out, o, a = [], 0, flat2d.shape[0]
        for s, n in zip(shapes, numels):
            out.append(flat2d[:, o:o + n].view(a, *s))
            o += n
        return out

    @staticmethod
    def _flat(grads):
        return torch.cat([g.reshape(g.shape[0], -1) for g in grads], dim=1)

    def dev_views(self, x, two_d=True):
        return self._views2d(x, self.sd, self.nd) if two_d else self._views(x, self.sd, self.nd)

    def srv_views(self, x, two_d=True):
        return self._views2d(x, self.ss, self.ns) if two_d else self._views(x, self.ss, self.ns)

    # ------------------------------------------------------------------
    # losses (per device; x: (B,1,28,28), y: (B,), m: (B,) 0/1 mask of real samples)
    # ------------------------------------------------------------------

    def _loss(self, pd, ps, x, y, m):
        z = forward_layers(self.spec, pd, x, 0, self.cut)
        logits = forward_layers(self.spec, ps, z, self.cut)
        l = F.cross_entropy(logits, y, reduction="none")
        return (l * m).sum() / m.sum().clamp_min(1.0)

    def _srv_loss(self, z, ps, y, m):
        l = F.cross_entropy(forward_layers(self.spec, ps, z, self.cut), y, reduction="none")
        return (l * m).sum() / m.sum().clamp_min(1.0)

    def _fwd_dev(self, pd, x):
        return forward_layers(self.spec, pd, x, 0, self.cut)

    def _fn(self, key, in_ps):
        if key not in self._fns:
            if key[0] == "step":
                f = grad_and_value(self._loss, argnums=(0, 1))
            else:                                                   # "dgrad": device-side gradient only
                f = grad(self._loss, argnums=0)
            self._fns[key] = vmap(f, in_dims=(0, in_ps, 0, 0, 0), chunk_size=self.chunk)
        return self._fns[key]

    # ------------------------------------------------------------------
    # one lock-step of a group (see module docstring)
    # ------------------------------------------------------------------

    def step(self, dev2d, srv, X, Y, M, kind: str, wts=None, gidx=None) -> torch.Tensor:
        """Updates dev2d (a, d_c) and srv in place; returns the (a,) per-device batch losses."""
        if self.engine == "vmap":
            pd = self.dev_views(dev2d)
            if kind == "shared":
                ps, in_ps = self.srv_views(srv, two_d=False), None
            elif kind == "per_device":
                ps, in_ps = self.srv_views(srv), 0
            else:
                ps, in_ps = self.srv_views(srv[gidx]), 0
            (gd, gs), loss = self._fn(("step", in_ps), in_ps)(pd, ps, X, Y, M)
            gd, gs = self._flat(gd), self._flat(gs)
        else:
            gd, gs, loss = self._loop_grads(dev2d, srv, X, Y, M, kind, gidx)
        with torch.no_grad():
            dev2d.sub_(gd, alpha=self.lr)
            if kind == "shared":
                srv.sub_(wts @ gs, alpha=self.lr)
            elif kind == "per_device":
                srv.sub_(gs, alpha=self.lr)
            else:
                srv.index_add_(0, gidx, gs * (-self.lr * wts)[:, None])
        return loss.detach()

    def _loop_grads(self, dev2d, srv, X, Y, M, kind, gidx, want_srv=True):
        gds, gss, losses = [], [], []
        for i in range(dev2d.shape[0]):
            src = srv if kind == "shared" else (srv[i] if kind == "per_device" else srv[gidx[i]])
            pd = [p.detach().clone().requires_grad_(True) for p in self.dev_views(dev2d[i], two_d=False)]
            ps = [p.detach().clone().requires_grad_(want_srv) for p in self.srv_views(src, two_d=False)]
            loss = self._loss(pd, ps, X[i], Y[i], M[i])
            g = torch.autograd.grad(loss, pd + (ps if want_srv else []))
            gds.append(torch.cat([t.reshape(-1) for t in g[:len(pd)]]))
            if want_srv:
                gss.append(torch.cat([t.reshape(-1) for t in g[len(pd):]]))
            losses.append(loss.detach())
        return torch.stack(gds), (torch.stack(gss) if want_srv else None), torch.stack(losses)

    def step_sequential_server(self, dev2d, srv, X, Y, M, order) -> torch.Tensor:
        """SFLv2 lock-step: device FP of all active devices, then the single server-side
        submodel processes device order[0], order[1], ... sequentially (one SGD step each, the
        returned smashed-data gradient taken before that device's server step), then every
        device back-propagates its returned gradient. Updates dev2d and srv in place."""
        a = dev2d.shape[0]
        losses = torch.zeros(a, device=dev2d.device)
        if self.engine == "vmap":
            pd = self.dev_views(dev2d)
            Z, vjp_fn = vjp(lambda P: vmap(self._fwd_dev)(P, X), pd)
            dZ = torch.zeros_like(Z)
            gfn = grad_and_value(self._srv_loss, argnums=(0, 1))
            for i in order:
                (gz, gs), l = gfn(Z[i].detach(), self.srv_views(srv, two_d=False), Y[i], M[i])
                with torch.no_grad():
                    srv.sub_(torch.cat([t.reshape(-1) for t in gs]), alpha=self.lr)
                dZ[i], losses[i] = gz, l.detach()
            gd = self._flat(vjp_fn(dZ)[0])
        else:
            leaves, Zs = [], []
            for i in range(a):
                pd = [p.detach().clone().requires_grad_(True) for p in self.dev_views(dev2d[i], two_d=False)]
                leaves.append(pd)
                Zs.append(self._fwd_dev(pd, X[i]))
            dZ = [None] * a
            for i in order:
                z = Zs[i].detach().requires_grad_(True)
                ps = [p.detach().clone().requires_grad_(True) for p in self.srv_views(srv, two_d=False)]
                l = self._srv_loss(z, ps, Y[i], M[i])
                g = torch.autograd.grad(l, [z] + ps)
                with torch.no_grad():
                    srv.sub_(torch.cat([t.reshape(-1) for t in g[1:]]), alpha=self.lr)
                dZ[i], losses[i] = g[0], l.detach()
            gd = torch.stack([torch.cat([t.reshape(-1) for t in torch.autograd.grad(Zs[i], leaves[i], dZ[i])])
                              for i in range(a)])
        with torch.no_grad():
            dev2d.sub_(gd, alpha=self.lr)
        return losses

    def device_grads(self, dev2d, srv2d, X, Y, M) -> torch.Tensor:
        """(a, d_c) device-side gradients g_k (no update): device k's submodel dev2d[k] with the
        server side srv2d[k] it trains against (CSA-SFL clustering probe)."""
        if self.engine == "vmap":
            return self._flat(self._fn(("dgrad", 0), 0)(self.dev_views(dev2d), self.srv_views(srv2d), X, Y, M))
        return self._loop_grads(dev2d, srv2d, X, Y, M, "per_device", None, want_srv=False)[0]

    # ------------------------------------------------------------------
    # evaluation of a (w^c, w^s) pair
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_per_class(self, dev, srv, X, Y, n_classes: int = 10, bs: int = 2500):
        """(n_classes,) test accuracy of (dev, srv) on each true class (NaN for an absent class)."""
        pd, ps = self.dev_views(dev, two_d=False), self.srv_views(srv, two_d=False)
        hit = torch.zeros(n_classes, device=X.device)
        tot = torch.zeros(n_classes, device=X.device)
        with torch.no_grad():
            for s in range(0, X.shape[0], bs):
                logits = forward_layers(self.spec, ps, forward_layers(self.spec, pd, X[s:s + bs], 0, self.cut), self.cut)
                y = Y[s:s + bs]
                hit += torch.bincount(y[logits.argmax(1) == y], minlength=n_classes).float()
                tot += torch.bincount(y, minlength=n_classes).float()
        return (hit / tot).cpu().numpy()

    def evaluate(self, dev, srv, X, Y, bs: int = 2500):
        pd, ps = self.dev_views(dev, two_d=False), self.srv_views(srv, two_d=False)
        correct, loss = 0, 0.0
        for s in range(0, X.shape[0], bs):
            logits = forward_layers(self.spec, ps, forward_layers(self.spec, pd, X[s:s + bs], 0, self.cut), self.cut)
            loss += float(F.cross_entropy(logits, Y[s:s + bs], reduction="sum"))
            correct += int((logits.argmax(1) == Y[s:s + bs]).sum())
        return correct / X.shape[0], loss / X.shape[0]
