"""
channel/conversions.py: Unit conversion helpers for wireless system quantities.

All functions are pure (no side effects) and operate on scalars or numpy arrays.
Physical unit labels appear explicitly in parameter and return-value docstrings.
"""

import math


def dbm_to_watts(dbm: float) -> float:
    """
    Convert power from dBm to watts.

    Formula: P_W = 10^((P_dBm - 30) / 10)

    Args:
        dbm (float): power in dBm.

    Returns:
        float: power in watts (W).

    Example:
        >>> dbm_to_watts(10.0)   # 10 dBm == 10 mW == 0.01 W
        0.01
        >>> dbm_to_watts(30.0)   # 30 dBm == 1 W
        1.0
    """
    return 10.0 ** ((dbm - 30.0) / 10.0)


def db_to_linear(db: float) -> float:
    """
    Convert a power ratio from dB to linear scale.

    Formula: ratio = 10^(dB / 10)

    Args:
        db (float): power ratio in decibels.

    Returns:
        float: linear power ratio (dimensionless).

    Example:
        >>> db_to_linear(0.0)    # 0 dB == gain of 1
        1.0
        >>> db_to_linear(10.0)   # 10 dB == gain of 10
        10.0
    """
    return 10.0 ** (db / 10.0)


def watts_to_dbm(w: float) -> float:
    """
    Convert power from watts to dBm.

    Formula: P_dBm = 10 * log10(P_W) + 30

    Args:
        w (float): power in watts (W). Must be > 0.

    Returns:
        float: power in dBm.

    Example:
        >>> watts_to_dbm(0.01)   # 0.01 W == 10 dBm
        10.0
        >>> watts_to_dbm(1.0)    # 1 W == 30 dBm
        30.0
    """
    return 10.0 * math.log10(w) + 30.0
