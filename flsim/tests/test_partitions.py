"""
tests/test_partitions.py: Unit tests for data partitioning strategies.

Uses a lightweight synthetic dataset (no torchvision download required) to
keep tests fast and offline-compatible.
"""

import numpy as np
import pytest
from dataclasses import dataclass

from flsim.data.iid import IIDPartitioner
from flsim.data.shard import ShardPartitioner
from flsim.data.dirichlet import DirichletPartitioner


# ---------------------------------------------------------------------------
# Minimal dataset stub
# ---------------------------------------------------------------------------

class _FakeDataset:
    """Synthetic dataset with .targets attribute. No download needed."""

    def __init__(self, n_samples: int = 6000, n_classes: int = 10, seed: int = 0):
        rng = np.random.RandomState(seed)
        # Balanced classes: n_samples must be divisible by n_classes for exact balance
        per_class = n_samples // n_classes
        self.targets = np.repeat(np.arange(n_classes), per_class)
        self.targets = self.targets[:n_samples]    # trim any leftover
        rng.shuffle(self.targets)

    def __len__(self):
        return len(self.targets)


def _all_indices_covered(partition, dataset_size: int) -> bool:
    """Every index 0..n-1 appears exactly once across all clients."""
    flat = []
    for part in partition:
        flat.extend(part)
    return sorted(flat) == list(range(dataset_size))


# ---------------------------------------------------------------------------
# IID tests
# ---------------------------------------------------------------------------

class TestIIDPartitioner:
    def _rng(self):
        return np.random.RandomState(42)

    def test_all_samples_assigned_once(self):
        ds = _FakeDataset(n_samples=6000)
        part = IIDPartitioner().partition(ds, num_clients=100, rng=self._rng())
        assert _all_indices_covered(part, len(ds))

    def test_number_of_clients(self):
        ds = _FakeDataset(n_samples=6000)
        part = IIDPartitioner().partition(ds, num_clients=100, rng=self._rng())
        assert len(part) == 100

    def test_roughly_equal_sizes(self):
        """Each client should have roughly 6000/100 = 60 samples (±1 for remainder)."""
        ds = _FakeDataset(n_samples=6000)
        part = IIDPartitioner().partition(ds, num_clients=100, rng=self._rng())
        sizes = [len(p) for p in part]
        assert min(sizes) >= 59 and max(sizes) <= 61

    def test_no_empty_clients(self):
        ds = _FakeDataset(n_samples=600)
        part = IIDPartitioner().partition(ds, num_clients=50, rng=self._rng())
        assert all(len(p) > 0 for p in part)

    def test_describe(self):
        assert "IID" in IIDPartitioner().describe()


# ---------------------------------------------------------------------------
# Shard tests
# ---------------------------------------------------------------------------

class TestShardPartitioner:
    def _rng(self):
        return np.random.RandomState(42)

    def _partitioner(self):
        return ShardPartitioner(num_shards=200, shards_per_client=2)

    def test_all_samples_assigned_once(self):
        ds = _FakeDataset(n_samples=6000)
        part = self._partitioner().partition(ds, num_clients=100, rng=self._rng())
        assert _all_indices_covered(part, len(ds))

    def test_number_of_clients(self):
        ds = _FakeDataset(n_samples=6000)
        part = self._partitioner().partition(ds, num_clients=100, rng=self._rng())
        assert len(part) == 100

    def test_clients_have_at_most_2_classes(self):
        """With shards_per_client=2 on a balanced 10-class dataset, clients should
        typically have samples from ≤ 2 distinct classes."""
        ds = _FakeDataset(n_samples=6000, n_classes=10, seed=0)
        # Restore original ordered targets for label-sort test
        ds.targets = np.repeat(np.arange(10), 600)   # balanced, ordered
        part = self._partitioner().partition(ds, num_clients=100, rng=self._rng())
        targets = ds.targets
        for client_part in part:
            labels = set(targets[i] for i in client_part)
            assert len(labels) <= 2, f"Expected ≤ 2 classes, got {labels}"

    def test_no_empty_clients(self):
        ds = _FakeDataset(n_samples=6000)
        ds.targets = np.repeat(np.arange(10), 600)
        part = self._partitioner().partition(ds, num_clients=100, rng=self._rng())
        assert all(len(p) > 0 for p in part)

    def test_describe(self):
        p = self._partitioner()
        desc = p.describe()
        assert "200" in desc and "2" in desc


# ---------------------------------------------------------------------------
# Dirichlet tests
# ---------------------------------------------------------------------------

class TestDirichletPartitioner:
    def _rng(self):
        return np.random.RandomState(42)

    def test_all_samples_assigned_once(self):
        ds = _FakeDataset(n_samples=6000)
        part = DirichletPartitioner(alpha=0.5).partition(ds, num_clients=100, rng=self._rng())
        assert _all_indices_covered(part, len(ds))

    def test_number_of_clients(self):
        ds = _FakeDataset(n_samples=6000)
        part = DirichletPartitioner(alpha=0.5).partition(ds, num_clients=50, rng=self._rng())
        assert len(part) == 50

    def test_low_alpha_produces_higher_label_skew_than_high_alpha(self):
        """
        Lower alpha → clients concentrate on fewer classes → higher per-client
        label entropy variance. We measure skew as mean KL divergence from
        uniform class distribution: higher = more non-IID.
        """
        ds = _FakeDataset(n_samples=6000, n_classes=10, seed=7)
        rng_low  = np.random.RandomState(0)
        rng_high = np.random.RandomState(0)

        part_low  = DirichletPartitioner(alpha=0.1).partition(ds, num_clients=50, rng=rng_low)
        part_high = DirichletPartitioner(alpha=100.0).partition(ds, num_clients=50, rng=rng_high)

        targets = ds.targets
        n_classes = 10

        def mean_skew(partition):
            skews = []
            for p in partition:
                if len(p) == 0:
                    continue
                labels = [targets[i] for i in p]
                counts = np.bincount(labels, minlength=n_classes).astype(float)
                dist = counts / counts.sum()
                uniform = np.ones(n_classes) / n_classes
                # KL divergence from uniform; add epsilon to avoid log(0)
                eps = 1e-9
                kl = np.sum(dist * np.log((dist + eps) / (uniform + eps)))
                skews.append(kl)
            return np.mean(skews)

        assert mean_skew(part_low) > mean_skew(part_high), (
            "Low alpha should produce higher label skew than high alpha"
        )

    def test_describe(self):
        assert "0.5" in DirichletPartitioner(alpha=0.5).describe()
