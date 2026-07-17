"""
core/training_utils.py: Small shared helper for local training loops.

Kept separate so flsim.core.client.Client (full-model training) and
flsim.core.split_client.SplitClient (split relay training) share ONE
definition of "what makes up one local round." Both use mini-batches of size
`batch_size` for SGD — the difference is only HOW MANY mini-batch steps a round
contains:

  * Full-epoch mode (max_iters=None, the framework's original FedAvg behaviour):
    E = local_epochs complete passes over the local data. One epoch =
    ceil(|D_k| / batch_size) mini-batch steps, so total = E * that. This is
    standard FedAvg — mini-batches are still used within each epoch.

  * Iteration mode (max_iters=H): exactly H mini-batch SGD steps, regardless of
    how many that is relative to an epoch. This is the "H local iterations" of
    semi-asynchronous split-FL papers (SAFSL), where "In the tau-th iteration,
    the device conducts FP ... based on a random mini-batch B^tau ⊆ D_n" — i.e.
    one mini-batch PER iteration, so per-round work = H * batch_size samples,
    independent of the client's full local dataset size. H may be smaller than
    one epoch's worth of batches (the loader is cycled if H exceeds it).

The two are different UNITS for local work; they are not interchangeable
(H=10 steps is generally a fractional number of epochs), which is why this
helper exists rather than reusing local_epochs for both.

FAIRNESS NOTE — for a cross-algorithm comparison, the SAME local-work setting
(same max_iters=H, or same local_epochs) must be used for EVERY algorithm, so
that each round does identical local work and the only differences measured
are the paradigm's own (compute offload, communication pattern, staleness).
The experiments enforce this by setting one shared learning.local_iters value
applied uniformly to all methods.
"""


def local_iters(cfg_learn):
    """learning.local_iters (H) if set (> 0), else None (full-epoch mode)."""
    li = getattr(cfg_learn, "local_iters", None)
    if li is not None and int(li) > 0:
        return int(li)
    return None


def effective_work_samples(cfg_learn, num_samples: int) -> int:
    """
    Per-round sample-passes used for time/energy/traffic — the SAME quantity
    for every paradigm, so cost is coherent across FL/FedAsync/SL/SFL/SAFSL.

    If learning.local_iters (H) is set: H * batch_size (H mini-batch iterations
    of a b-sample batch, the paper's local-iterations unit). Otherwise the
    original full-epoch work: num_samples * local_epochs. Pair with max_iters=
    local_iters(cfg_learn) passed to Client.train / SplitClient.train_local so
    the TRAINING work and the COSTED work always match.
    """
    H = local_iters(cfg_learn)
    if H is not None:
        return H * int(cfg_learn.batch_size)
    return int(num_samples) * int(cfg_learn.local_epochs)


def iter_local_batches(loader, local_epochs: int, max_iters: int = None):
    """
    Yield (x, y) mini-batches for one local round (see module docstring).

    Args:
        loader: a DataLoader (already shuffled) over the client's local data.
        local_epochs (int): E — full passes (used only when max_iters is None).
        max_iters (int, optional): H — number of mini-batch steps; overrides
            local_epochs when set (> 0).

    Yields:
        (x, y) batches. In iteration mode the loader is cycled (re-iterated,
        reshuffling each cycle) until H batches have been yielded.
    """
    if max_iters is not None:
        if max_iters <= 0:
            raise ValueError(f"max_iters must be > 0 when set, got {max_iters}")
        count = 0
        while count < max_iters:
            for batch in loader:
                yield batch
                count += 1
                if count >= max_iters:
                    return
    else:
        for _ in range(local_epochs):
            for batch in loader:
                yield batch
