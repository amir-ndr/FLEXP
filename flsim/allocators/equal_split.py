"""
allocators/equal_split.py: Default equal-split resource allocator.

Implements the simplest possible allocation policy:
  - Bandwidth : equal FDMA split  b_k = B / K
  - Power     : every client transmits at p_max
  - CPU freq  : each client keeps the frequency in its profile
                (already drawn heterogeneously at startup)

This is the baseline used by most FL papers and was previously hardcoded
inside FDMAChannelModel. Moving it here decouples the channel physics from
the allocation policy.

To write a custom allocator, subclass ResourceAllocator and override the
methods you want to optimize. Return the equal-split value in the others.

Example skeleton:

    from flsim.allocators.equal_split import EqualSplitAllocator

    class MyAllocator(EqualSplitAllocator):
        def allocate_bandwidth(self, selected_profiles, total_bandwidth_hz, **kwargs):
            # channel_gains available via kwargs["channel_gains"]
            gains = kwargs.get("channel_gains", {})
            # ... your optimization here ...
            return {p.client_id: bw for p, bw in zip(selected_profiles, bws)}
"""

from flsim.interfaces.allocator import ResourceAllocator


class EqualSplitAllocator(ResourceAllocator):
    """
    Equal-split resource allocation — the standard FL baseline.

    Bandwidth  : b_k = B / K  (FDMA equal split)
    Power      : p_k = p_max  (all clients at max power)
    CPU freq   : f_k = profile.cpu_frequency_hz  (keep profile assignment)

    This class does NOT:
    - Solve any optimization problem.
    - Inspect channel gains or client distances.
    - Hold any mutable state.
    """

    def allocate_bandwidth(
        self,
        selected_profiles: list,
        total_bandwidth_hz: float,
        **kwargs,
    ) -> dict:
        """
        Equal FDMA split: b_k = B / K for all selected clients.

        Args:
            selected_profiles: list of ClientSystemProfile.
            total_bandwidth_hz (float): total system bandwidth B in Hz.

        Returns:
            dict[int, float]: {client_id: total_bandwidth_hz / K}.
        """
        n = len(selected_profiles)
        if n == 0:
            return {}
        share = total_bandwidth_hz / n
        return {p.client_id: share for p in selected_profiles}

    def allocate_power(
        self,
        selected_profiles: list,
        p_max_w: float,
        **kwargs,
    ) -> dict:
        """
        Each client transmits at its own profile power p_k = profile.tx_power_w.

        In the homogeneous default (wireless.tx_power_dbm, no per-device
        range) every profile's tx_power_w equals dbm_to_watts(tx_power_dbm)
        == p_max_w, so this is IDENTICAL to the old "everyone at p_max"
        behaviour. With a per-device power range (wireless.tx_power_w_min/max,
        e.g. p_n ~ U[0.1, 0.2] W), the profile power is each device's actual
        max — using it keeps the sync/async uplink physics consistent with
        the split cost model, which reads profile.tx_power_w directly.

        Args:
            selected_profiles: list of ClientSystemProfile.
            p_max_w (float): system-level max transmit power in watts (kept
                for custom allocators that optimise against a shared cap;
                unused by this default policy — the profile power IS the cap).

        Returns:
            dict[int, float]: {client_id: profile.tx_power_w}.
        """
        return {p.client_id: p.tx_power_w for p in selected_profiles}

    def allocate_cpu_freq(
        self,
        selected_profiles: list,
        f_max_hz: float,
        **kwargs,
    ) -> dict:
        """
        Each client uses its profile's CPU frequency (set at startup).

        The profile frequency is already heterogeneous when cpu_freq_mode
        is "discrete_ghz". This method preserves that assignment — it does
        not override it with f_max.

        Args:
            selected_profiles: list of ClientSystemProfile.
            f_max_hz (float): maximum CPU frequency (unused here).

        Returns:
            dict[int, float]: {client_id: profile.cpu_frequency_hz}.
        """
        return {p.client_id: p.cpu_frequency_hz for p in selected_profiles}
