"""
interfaces/allocator.py: Abstract base class for resource allocators.

A ResourceAllocator decides, each round, how radio and compute resources
are distributed among the selected clients. It is the single extension
point for implementing optimization algorithms such as:

  - Water-filling bandwidth allocation
  - Max-min fairness power control
  - Energy-aware CPU frequency scheduling
  - Joint bandwidth + power optimization

Separation from ChannelModel:
  ChannelModel answers "what is the channel?" (physics).
  ResourceAllocator answers "how do we use it?" (policy/optimization).

Separation from FederatedAlgorithm:
  FederatedAlgorithm answers "which clients participate and how are their
  updates combined?" (learning).
  ResourceAllocator answers "what resources does each client get?" (system).

To write a new allocator:
  1. Subclass ResourceAllocator.
  2. Override any of the three methods you want to optimize.
  3. Return the fixed/default value in the others.
  4. Inject via run.py — no other file needs to change.
"""

from abc import ABC, abstractmethod


class ResourceAllocator(ABC):
    """
    Base class for per-round resource allocation.

    All three methods receive the list of selected ClientSystemProfiles
    so the allocator can make decisions based on distance, channel gain,
    CPU capability, number of samples, etc.

    Contract:
      - allocate_bandwidth values must sum to total_bandwidth_hz.
      - allocate_power values must be <= p_max_w for each client.
      - allocate_cpu_freq values must be <= f_max_hz for each client.
      - All dicts are keyed by client_id (int).
    """

    @abstractmethod
    def allocate_bandwidth(
        self,
        selected_profiles: list,
        total_bandwidth_hz: float,
        **kwargs,
    ) -> dict:
        """
        Assign uplink bandwidth to each selected client.

        Args:
            selected_profiles: list of ClientSystemProfile for selected clients.
            total_bandwidth_hz (float): total system bandwidth B in Hz.
            **kwargs: extra context (e.g. channel_gains dict) passed by Simulator.

        Returns:
            dict[int, float]: {client_id: allocated_bandwidth_hz}.
                              Values should sum to total_bandwidth_hz.
        """

    @abstractmethod
    def allocate_power(
        self,
        selected_profiles: list,
        p_max_w: float,
        **kwargs,
    ) -> dict:
        """
        Assign transmit power to each selected client.

        Args:
            selected_profiles: list of ClientSystemProfile for selected clients.
            p_max_w (float): maximum transmit power per client in watts.
            **kwargs: extra context passed by Simulator.

        Returns:
            dict[int, float]: {client_id: tx_power_w}.
                              Values should be in (0, p_max_w].
        """

    @abstractmethod
    def allocate_cpu_freq(
        self,
        selected_profiles: list,
        f_max_hz: float,
        **kwargs,
    ) -> dict:
        """
        Assign CPU frequency to each selected client for this round.

        In most paper setups the CPU frequency is fixed per client (drawn once
        at profile creation). Override this method when your algorithm jointly
        optimizes computation and communication resources.

        Args:
            selected_profiles: list of ClientSystemProfile for selected clients.
            f_max_hz (float): maximum CPU frequency in Hz.
            **kwargs: extra context passed by Simulator.

        Returns:
            dict[int, float]: {client_id: cpu_frequency_hz}.
                              Values should be in (0, f_max_hz].
        """
