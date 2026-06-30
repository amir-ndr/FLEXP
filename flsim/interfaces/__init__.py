"""
interfaces/: Abstract base classes for every extensible component in flsim.

Rules:
- Zero logic; only ABCs.
- Zero imports from other flsim modules.
- Every new algorithm / time model / channel model / partitioner must subclass from here.
"""
from flsim.interfaces.algorithm import FederatedAlgorithm
from flsim.interfaces.time_model import TimeModel
from flsim.interfaces.channel_model import ChannelModel
from flsim.interfaces.partitioner import DataPartitioner

__all__ = ["FederatedAlgorithm", "TimeModel", "ChannelModel", "DataPartitioner"]
