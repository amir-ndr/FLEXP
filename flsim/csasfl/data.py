"""
flsim/csasfl/data.py: MNIST on the GPU, the label-Dirichlet partition, and per-device
mini-batch streams for the CSA-SFL paper's simulation (Sec. Performance Evaluation).

  * MNIST: the 60,000 training images are partitioned over the K devices; the 10,000 test
    images evaluate the global model. Normalization (0.1307, 0.3081). Read straight from the
    IDX files (torchvision not needed); the whole set lives on the training device as one
    float tensor, so a mini-batch is a single gather (no DataLoader).
  * Partition (user choice 2026-10-19): per-class Dirichlet -- for every class c the class's
    samples are split over the K devices with proportions ~ Dir(delta * 1_K); smaller delta =
    stronger heterogeneity (delta = 0.1 strongly non-IID, delta = 10 near-IID). Device sizes
    |D_k| are therefore unequal. A draw is repeated until every device holds at least
    `min_per_device` samples (default 10, the NIID-Bench convention); the number of redraws
    is recorded in PARTITION_REDRAWS and written to every run's CSV.
  * DeviceStream: one per device, with its own seeded generators (training order, ParallelSFL
    iteration stream, clustering mini-batch), so a method's data order never depends on how
    much randomness another component consumed.
      epochs(E, B)  E full passes, each a fresh permutation cut into ceil(|D_k|/B) batches
                    (the last one smaller, padded + masked) -> the paper's E*ceil(|D_k|/B) steps;
      iters(n, B)   n batches of exactly B (epoch-wise reshuffle, the tail carried into the
                    next permutation) -> ParallelSFL's iteration-based local updating;
      probe(B)      one batch of min(B, |D_k|) samples for the clustering gradient g_k.
"""

import gzip
import os
import struct

import numpy as np
import torch

_DEFAULT_ROOT = os.path.expanduser("~/.flsim/data")
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
PARTITION_REDRAWS = {}      # (K, delta, seed, min_per_device) -> redraws needed


def _read_idx(path: str) -> np.ndarray:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def _find(root: str, name: str) -> str:
    for d in (os.path.join(root, "MNIST", "raw"), os.path.join(root, "mnist"), root):
        for ext in ("", ".gz"):
            p = os.path.join(d, name + ext)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"{name} not found under {root} (MNIST/raw/); download MNIST there first")


def load_mnist(root: str = _DEFAULT_ROOT, device="cpu") -> dict:
    """{'x_train','y_train','x_test','y_test'}: normalized float (n,1,28,28) and int64 labels on
    `device`; the training labels are also returned as numpy (for partitioning)."""
    if not any(os.path.exists(os.path.join(root, "MNIST", "raw", f)) for f in
               ("train-images-idx3-ubyte", "train-images-idx3-ubyte.gz")):
        try:                                                  # download once via torchvision if present
            import torchvision
            torchvision.datasets.MNIST(root=root, train=True, download=True)
            torchvision.datasets.MNIST(root=root, train=False, download=True)
        except ImportError:
            pass
    arr = {k: _read_idx(_find(root, f)) for k, f in (
        ("xtr", "train-images-idx3-ubyte"), ("ytr", "train-labels-idx1-ubyte"),
        ("xte", "t10k-images-idx3-ubyte"), ("yte", "t10k-labels-idx1-ubyte"))}
    norm = lambda x: ((torch.from_numpy(x.copy()).float().div_(255.0) - MNIST_MEAN) / MNIST_STD).unsqueeze(1)
    return {
        "x_train": norm(arr["xtr"]).to(device), "y_train": torch.from_numpy(arr["ytr"].astype(np.int64)).to(device),
        "x_test": norm(arr["xte"]).to(device), "y_test": torch.from_numpy(arr["yte"].astype(np.int64)).to(device),
        "y_train_np": arr["ytr"].astype(np.int64),
    }


def dirichlet_partition(labels: np.ndarray, K: int, delta: float, seed: int,
                        min_per_device: int = 10, max_tries: int = 5000) -> list:
    """Per-class label-Dirichlet partition (see module docstring). Returns K sorted index arrays."""
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)
    by_class = [np.where(labels == c)[0] for c in classes]
    for tries in range(max_tries):
        parts = [[] for _ in range(K)]
        for idx in by_class:
            idx = idx.copy()
            rng.shuffle(idx)
            props = rng.dirichlet(delta * np.ones(K))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_per_device:
            PARTITION_REDRAWS[(K, float(delta), int(seed), int(min_per_device))] = tries
            return [np.sort(np.asarray(p, dtype=np.int64)) for p in parts]
    raise RuntimeError(f"Dirichlet({delta}) partition: no draw with >= {min_per_device} samples per device "
                       f"in {max_tries} tries (K={K})")


def label_histograms(labels: np.ndarray, parts: list, num_classes: int = 10) -> np.ndarray:
    """(K, M) normalized label distribution V_k of every device."""
    H = np.zeros((len(parts), num_classes))
    for k, p in enumerate(parts):
        H[k] = np.bincount(labels[p], minlength=num_classes)
    return H / np.maximum(H.sum(axis=1, keepdims=True), 1.0)


class DeviceStream:
    """Mini-batch index streams of one device (indices into the GPU training tensor)."""

    def __init__(self, indices: np.ndarray, seed: int, device):
        self.idx = torch.as_tensor(indices, dtype=torch.long)
        self.n = int(self.idx.numel())
        self.device = device
        self.g_train = torch.Generator().manual_seed(int(seed))
        self.g_iter = torch.Generator().manual_seed(int(seed) + 1_000_003)
        self.g_probe = torch.Generator().manual_seed(int(seed) + 2_000_003)
        self._queue = torch.empty(0, dtype=torch.long)

    def batches_per_epoch(self, B: int) -> int:
        return -(-self.n // B)

    def epochs(self, E: int, B: int):
        """(idx (E*nb, B) long, mask (E*nb, B) float) on the device; padded slots have mask 0."""
        nb = self.batches_per_epoch(B)
        rows = []
        for _ in range(E):
            perm = self.idx[torch.randperm(self.n, generator=self.g_train)]
            pad = torch.full((nb * B - self.n,), -1, dtype=torch.long)
            rows.append(torch.cat([perm, pad]).view(nb, B))
        I = torch.cat(rows)
        M = (I >= 0).float()
        return I.clamp_min(0).to(self.device), M.to(self.device)

    def iters(self, n_iter: int, B: int):
        """n_iter full batches of B (reshuffle per epoch, tail carried over)."""
        need = n_iter * B
        while self._queue.numel() < need:
            self._queue = torch.cat([self._queue, self.idx[torch.randperm(self.n, generator=self.g_iter)]])
        I, self._queue = self._queue[:need].view(n_iter, B), self._queue[need:]
        return I.to(self.device), torch.ones(n_iter, B, device=self.device)

    def probe(self, B: int):
        """One batch of min(B, |D_k|) distinct samples, padded to B (for the clustering gradient)."""
        m = min(B, self.n)
        sel = self.idx[torch.randperm(self.n, generator=self.g_probe)[:m]]
        I = torch.cat([sel, torch.zeros(B - m, dtype=torch.long)])
        M = torch.cat([torch.ones(m), torch.zeros(B - m)])
        return I.to(self.device), M.to(self.device)
