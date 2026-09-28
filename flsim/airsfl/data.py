"""
flsim/airsfl/data.py: CIFAR-10 pipeline for the AirSFL study (roadmap Sec. 4).

  * Fixed stratified 45,000 / 5,000 train/validation split of the official
    training set (seed 2026); the official 10,000 test images are untouched.
  * Normalization statistics computed from the 45k TRAINING split only.
  * Optional train-only augmentation: random crop (zero padding 4) + horizontal flip.
  * Partition: IID or label-Dirichlet(alpha), minimum `min_per_client` examples
    per client (redraw if violated); aggregation weights a_n = D_n / sum D.

MATCHED STREAMS. Every client owns a ClientStream with its OWN torch.Generator
(seeded by (seed, client_id)) that drives its index shuffling, crop offsets and
flips. A method calls next_batch() exactly tau times per client per round, so
all five methods consume IDENTICAL minibatches and augmentations. This is what
makes the noiseless equivalence (AirSFL == digital SFL-V1 == FedAvg) testable.
Radio noise uses a separate generator and never perturbs the data streams.
"""

import os

import numpy as np
import torch

_DEFAULT_ROOT = os.path.expanduser("~/.flsim/data")


def _read_cifar_batches(root: str):
    """Read the standard python-pickle CIFAR-10 files (same arrays torchvision builds)."""
    import pickle
    base = os.path.join(root, "cifar-10-batches-py")

    def load(names):
        xs, ys = [], []
        for nm in names:
            with open(os.path.join(base, nm), "rb") as f:
                d = pickle.load(f, encoding="latin1")
            xs.append(np.asarray(d["data"], dtype=np.uint8).reshape(-1, 3, 32, 32))
            ys.extend(d["labels"])
        return np.concatenate(xs), ys
    xtr, ytr = load([f"data_batch_{i}" for i in range(1, 6)])
    xte, yte = load(["test_batch"])
    return xtr, ytr, xte, yte


def load_cifar10_tensors(root: str = _DEFAULT_ROOT, n_val: int = 5000, split_seed: int = 2026):
    """Returns dict of uint8 NCHW tensors + int64 labels and the train mean/std."""
    try:
        import torchvision
        tr = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
        te = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
        xtr, ytr = tr.data.transpose(0, 3, 1, 2), tr.targets        # HWC -> CHW
        xte, yte = te.data.transpose(0, 3, 1, 2), te.targets
    except ImportError:                                              # no torchvision: read the pickles
        xtr, ytr, xte, yte = _read_cifar_batches(root)
    x_all = torch.from_numpy(np.ascontiguousarray(xtr))              # uint8 N,3,32,32
    y_all = torch.tensor(ytr, dtype=torch.long)
    x_te = torch.from_numpy(np.ascontiguousarray(xte))
    y_te = torch.tensor(yte, dtype=torch.long)

    # stratified split: n_val/10 validation images per class
    rng = np.random.RandomState(split_seed)
    per_class = n_val // 10
    val_idx, train_idx = [], []
    y_np = y_all.numpy()
    for c in range(10):
        idx = np.where(y_np == c)[0]
        rng.shuffle(idx)
        val_idx.extend(idx[:per_class].tolist())
        train_idx.extend(idx[per_class:].tolist())
    train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)

    x_tr, y_tr = x_all[train_idx], y_all[train_idx]
    x_va, y_va = x_all[val_idx], y_all[val_idx]
    xf = x_tr.float() / 255.0
    mean = xf.mean(dim=(0, 2, 3))
    std = xf.std(dim=(0, 2, 3))
    return {"x_train": x_tr, "y_train": y_tr, "x_val": x_va, "y_val": y_va,
            "x_test": x_te, "y_test": y_te, "mean": mean, "std": std}


PARTITION_REDRAWS = {}   # (scheme, N, alpha, seed) -> number of Dirichlet redraws (disclosed)


def partition(labels: np.ndarray, num_clients: int, scheme: str = "iid", alpha: float = 0.5,
              min_per_client: int = 16, seed: int = 0, max_tries: int = 200) -> list:
    """IID or label-Dirichlet(alpha) partition; redraw until every client has at
    least min_per_client examples (roadmap: at least one minibatch per client).
    The number of redraws is stored in PARTITION_REDRAWS for disclosure."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    if scheme == "iid":
        perm = rng.permutation(n)
        return [np.sort(p) for p in np.array_split(perm, num_clients)]
    if scheme != "dirichlet":
        raise ValueError(f"scheme must be 'iid' or 'dirichlet', got {scheme!r}")
    classes = np.unique(labels)
    for tries in range(max_tries):
        PARTITION_REDRAWS[(scheme, num_clients, alpha, seed)] = tries
        parts = [[] for _ in range(num_clients)]
        for c in classes:
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            props = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_per_client:
            return [np.sort(np.array(p)) for p in parts]
    raise RuntimeError(f"Dirichlet({alpha}) partition failed min_per_client={min_per_client} "
                       f"after {max_tries} redraws")


class ClientStream:
    """Per-client minibatch stream with its own generator (matched across methods).
    Epoch-wise reshuffle; each batch is exactly B examples (the tail of an epoch
    is carried into the next shuffle, so every batch is full)."""

    def __init__(self, x_u8: torch.Tensor, y: torch.Tensor, mean, std, batch_size: int,
                 seed: int, device, augment: bool = True):
        self.x = x_u8.to(device)
        self.y = y.to(device)
        self.mean = mean.view(1, 3, 1, 1).to(device)
        self.std = std.view(1, 3, 1, 1).to(device)
        self.B = int(batch_size)
        self.augment = augment
        self.device = device
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(int(seed))
        self._queue = torch.empty(0, dtype=torch.long)

    def __len__(self):
        return int(self.x.shape[0])

    def _next_indices(self) -> torch.Tensor:
        while self._queue.numel() < self.B:
            perm = torch.randperm(len(self), generator=self.gen)
            self._queue = torch.cat([self._queue, perm])
        idx, self._queue = self._queue[: self.B], self._queue[self.B:]
        return idx

    def next_batch(self):
        idx = self._next_indices()
        x = self.x[idx.to(self.device)].float() / 255.0
        if self.augment:
            x = self._crop_flip(x)
        x = (x - self.mean) / self.std
        return x, self.y[idx.to(self.device)]

    def _crop_flip(self, x: torch.Tensor) -> torch.Tensor:
        """Random 32x32 crop of the zero-padded (4) image + horizontal flip, one gather."""
        B = x.shape[0]
        padded = torch.nn.functional.pad(x, (4, 4, 4, 4))                 # zero pad 4
        offs = torch.randint(0, 9, (B, 2), generator=self.gen)            # crop offsets 0..8
        flips = torch.rand(B, generator=self.gen) < 0.5
        ar = torch.arange(32)
        rows = offs[:, 0:1] + ar                                          # B x 32
        cols = offs[:, 1:2] + ar
        cols = torch.where(flips[:, None], cols.flip(1), cols)            # flip = reversed columns
        dev = x.device
        bi = torch.arange(B, device=dev)[:, None, None, None]
        ci = torch.arange(x.shape[1], device=dev)[None, :, None, None]
        return padded[bi, ci, rows.to(dev)[:, None, :, None], cols.to(dev)[:, None, None, :]]


def make_eval_tensors(x_u8: torch.Tensor, mean, std, device):
    x = x_u8.float().div(255.0)
    x = (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    return x.to(device)
