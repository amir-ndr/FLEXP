"""
tests/test_fedavg.py: Unit tests for FedAvg algorithm.

Tests cover client selection, aggregation correctness, and sample-size weighting.
Uses tiny MnistCNN models to keep tests fast (no dataset download needed).
"""

import copy
import numpy as np
import pytest
import torch
from collections import OrderedDict

from flsim.algorithms.fedavg import FedAvg
from flsim.core.client import ClientUpdate
from flsim.models.mnist_cnn import MnistCNN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_update(model: torch.nn.Module, num_samples: int, client_id: int = 0) -> ClientUpdate:
    """Create a ClientUpdate from a model's state dict."""
    return ClientUpdate(
        client_id=client_id,
        state_dict=copy.deepcopy(model.state_dict()),
        num_samples=num_samples,
        train_loss=0.0,
    )


def _state_dicts_equal(sd1: OrderedDict, sd2: OrderedDict) -> bool:
    """Check if two state dicts have identical tensors."""
    for k in sd1:
        if not torch.allclose(sd1[k].float(), sd2[k].float(), atol=1e-6):
            return False
    return True


# ---------------------------------------------------------------------------
# Minimal client stub for select_clients tests
# ---------------------------------------------------------------------------

class _FakeClient:
    def __init__(self, cid):
        self.client_id = cid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFedAvg:
    def _rng(self, seed=42):
        return np.random.RandomState(seed)

    def test_select_clients_returns_exact_count(self):
        algo = FedAvg()
        clients = [_FakeClient(i) for i in range(50)]
        selected = algo.select_clients(clients, num_to_select=10, rng=self._rng())
        assert len(selected) == 10

    def test_select_clients_no_duplicates(self):
        algo = FedAvg()
        clients = [_FakeClient(i) for i in range(50)]
        selected = algo.select_clients(clients, num_to_select=10, rng=self._rng())
        ids = [c.client_id for c in selected]
        assert len(ids) == len(set(ids)), "Duplicate clients selected"

    def test_select_clients_subset_of_all(self):
        algo = FedAvg()
        clients = [_FakeClient(i) for i in range(50)]
        selected = algo.select_clients(clients, num_to_select=10, rng=self._rng())
        all_ids = {c.client_id for c in clients}
        for c in selected:
            assert c.client_id in all_ids

    def test_select_clients_reproducible(self):
        algo = FedAvg()
        clients = [_FakeClient(i) for i in range(100)]
        sel1 = algo.select_clients(clients, 10, self._rng(seed=7))
        sel2 = algo.select_clients(clients, 10, self._rng(seed=7))
        assert [c.client_id for c in sel1] == [c.client_id for c in sel2]

    def test_aggregate_identical_models_returns_same(self):
        """Aggregating K identical models should return the same model."""
        algo = FedAvg()
        model = MnistCNN()
        updates = [_make_update(model, num_samples=100, client_id=k) for k in range(5)]
        result = algo.aggregate(model, updates)
        assert _state_dicts_equal(result, model.state_dict()), (
            "Aggregation of identical models should produce the same model"
        )

    def test_aggregate_is_sample_size_weighted(self):
        """
        Manual check: two clients with different weights should produce
        a weighted average, not a simple mean.

        Client A: all weights = 1.0, 100 samples
        Client B: all weights = 0.0,  50 samples
        Expected: 100/(100+50) * 1.0 + 50/(100+50) * 0.0 = 2/3
        """
        algo = FedAvg()
        ref_model = MnistCNN()

        # Client A: set all params to 1.0
        model_a = copy.deepcopy(ref_model)
        with torch.no_grad():
            for p in model_a.parameters():
                p.fill_(1.0)

        # Client B: set all params to 0.0
        model_b = copy.deepcopy(ref_model)
        with torch.no_grad():
            for p in model_b.parameters():
                p.fill_(0.0)

        update_a = _make_update(model_a, num_samples=100, client_id=0)
        update_b = _make_update(model_b, num_samples=50,  client_id=1)

        result = algo.aggregate(ref_model, [update_a, update_b])

        expected = 100 / 150  # weight of client A
        for key, tensor in result.items():
            # Only check floating-point parameters (not int buffers like num_batches_tracked)
            if tensor.dtype.is_floating_point:
                assert torch.allclose(tensor, torch.full_like(tensor, expected), atol=1e-5), (
                    f"Layer {key}: expected {expected}, got {tensor.mean().item()}"
                )

    def test_aggregate_handles_batchnorm_buffers(self):
        """Aggregation must handle BatchNorm running_mean/var (non-gradient buffers)."""
        from flsim.models.cifar_cnn import CifarCNN
        algo = FedAvg()
        model = CifarCNN()
        updates = [_make_update(model, num_samples=100, client_id=k) for k in range(3)]
        result = algo.aggregate(model, updates)
        # All keys in original state dict should be present in result
        assert set(result.keys()) == set(model.state_dict().keys())

    def test_configure_client_is_noop(self):
        """configure_client should not raise and should return None."""
        algo = FedAvg()
        client = _FakeClient(0)
        model = MnistCNN()
        result = algo.configure_client(client, model, round_idx=0)
        assert result is None
