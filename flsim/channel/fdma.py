"""
channel/fdma.py: kept for backwards compatibility and as a named alias.

FDMAChannelModel is now just PathLossChannelModel — bandwidth allocation
has moved to EqualSplitAllocator (allocators/equal_split.py).

The class is retained so existing configs and imports don't break.
"""

from flsim.channel.path_loss import PathLossChannelModel

# Alias — no additional behaviour needed now that allocation is separate.
FDMAChannelModel = PathLossChannelModel
