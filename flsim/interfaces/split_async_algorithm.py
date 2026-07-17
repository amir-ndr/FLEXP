"""
interfaces/split_async_algorithm.py: Base class for asynchronous / semi-
asynchronous split-FL algorithms.

Mirrors flsim.interfaces.async_algorithm.AsyncFederatedAlgorithm's override
pattern, adapted for split learning's two-sided (client-side, server-side)
model pair — the same adaptation flsim.core.split_simulator.SplitSimulator
makes to the synchronous case. A split-FL algorithm never aggregates a
single state_dict; it always combines a (client_state_dict, server_state_dict)
pair together (same weight applied to both — see aggregate_buffered below),
since the paper's per-device model w^H_{n,t} = {device-side, server-side}.

Design — three independent override points, only one has no default:

  select_clients(all_clients, num_to_trigger, rng, **kwargs)
      Which device(s) the Scheduler dispatches each cycle.
      Default: uniform random without replacement (identical default to
      AsyncFederatedAlgorithm).

  participation_weight(num_samples, staleness, **kwargs) -> float
      UNNORMALIZED per-device weight used by the default aggregate_buffered()
      below (normalized across the buffer there). Default: num_samples (pure
      FedAvg-style data-size weighting, staleness ignored) — override this
      alone for a different weighting scheme without touching the
      combination mechanics themselves. See flsim.algorithms.safsl.SAFSL for
      the paper-faithful override.

  aggregate_buffered(client_state_dicts, server_state_dicts, num_samples_list,
                      stalenesses, global_epoch) -> (client_state_dict, server_state_dict)
      HAS a default: the normalized-weight average of the B buffered updates
      (weights from participation_weight()), applied identically to the
      client-side and server-side dicts. Override this directly only if you
      want something other than a weighted average (e.g. blending with the
      previous global model — see the note in SAFSL's docstring about eq. 4
      having no such blend term).

Resource allocation (bandwidth/power/frequency across concurrently-training
devices) is deliberately NOT part of this class — like AsyncFederatedAlgorithm,
it is a separately swappable component (the `allocator` passed to
SplitAsyncSimulator), so client selection, aggregation, and resource
allocation are each independently overridable.

buffer_size (class attribute, default 1): number of arrivals the simulator
buffers before triggering one aggregation — the paper's |S_t|. 1 = fully
asynchronous ("evolves into the fully asynchronous... setting" per the
semi-async split-FL paper, in scenarios of extreme device heterogeneity).
window_size (the number of concurrent in-flight devices) = fully synchronous
batching. Same convention as AsyncFederatedAlgorithm.buffer_size.
"""

from flsim.core.split_simulator import _weighted_average_state_dicts


class SplitAsyncAlgorithm:
    """Base class for asynchronous/semi-asynchronous split-FL algorithms."""

    buffer_size: int = 1

    # ------------------------------------------------------------------
    # Scheduler: client selection
    # ------------------------------------------------------------------

    def select_clients(
        self,
        all_clients: list,
        num_to_trigger: int,
        rng,
        **kwargs,
    ) -> list:
        """
        Select clients for one Scheduler dispatch cycle.

        Default: uniform random sampling without replacement (identical to
        AsyncFederatedAlgorithm.select_clients — override for any custom
        policy, e.g. channel-aware or compute-aware selection).

        Args:
            all_clients:    all SplitClient objects in the federation.
            num_to_trigger: how many to select this cycle.
            rng:            numpy RandomState for reproducibility.

        Returns:
            list[SplitClient]: selected clients.
        """
        indices = rng.choice(len(all_clients), size=num_to_trigger, replace=False)
        return [all_clients[i] for i in indices]

    # ------------------------------------------------------------------
    # Updater: per-device weight (the piece you override most often)
    # ------------------------------------------------------------------

    def participation_weight(self, num_samples: int, staleness: int, **kwargs) -> float:
        """
        Unnormalized weight for one arriving device's update.

        Default: num_samples (pure data-size / FedAvg-style weighting,
        staleness ignored). Override for a custom weighting scheme — e.g.
        SAFSL's data-size-and-staleness rule (flsim.algorithms.safsl.SAFSL).

        Args:
            num_samples: this device's local sample count.
            staleness:   global_epoch (at aggregation) - tau (at dispatch).

        Returns:
            float: unnormalized weight (normalized across the buffer by the
            default aggregate_buffered() below).
        """
        return float(num_samples)

    # ------------------------------------------------------------------
    # Updater: combine buffered arrivals into the new global pair
    # ------------------------------------------------------------------

    def aggregate_buffered(
        self,
        client_state_dicts: list,
        server_state_dicts: list,
        num_samples_list: list,
        stalenesses: list,
        global_epoch: int,
    ) -> tuple:
        """
        Combine B buffered (client-side, server-side) updates into the new
        global (client_state_dict, server_state_dict).

        Default: normalized-weight average — weights from
        participation_weight(num_samples_i, staleness_i), normalized to sum
        to 1 ACROSS THIS BUFFER ONLY. There is no blend with the previous
        global model (unlike flsim.algorithms.fedasync.FedAsync's
        (1-alpha)*old + alpha*new rule) — with buffer_size=1 the single
        arriving device's own trained pair REPLACES the global pair outright.
        This matches the semi-async split-FL paper's aggregation rule (eq. 4)
        exactly when participation_weight() implements its ρ_{n,t} formula —
        see flsim.algorithms.safsl.SAFSL.

        Override this method directly (instead of just participation_weight)
        if you want different combination mechanics entirely — e.g. blending
        with the previous global model, momentum, or coordinate-wise trust
        weighting.

        Args:
            client_state_dicts: list[OrderedDict], one per buffered device.
            server_state_dicts: list[OrderedDict], one per buffered device.
            num_samples_list:   list[int], same order.
            stalenesses:        list[int], same order.
            global_epoch:       number of aggregations processed so far.

        Returns:
            tuple[OrderedDict, OrderedDict]: (new_client_state, new_server_state).
        """
        weights = [
            self.participation_weight(n, s)
            for n, s in zip(num_samples_list, stalenesses)
        ]
        new_client_state = _weighted_average_state_dicts(client_state_dicts, weights)
        new_server_state = _weighted_average_state_dicts(server_state_dicts, weights)
        return new_client_state, new_server_state
