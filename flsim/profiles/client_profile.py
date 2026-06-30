"""
profiles/client_profile.py: ClientSystemProfile dataclass.

A ClientSystemProfile captures all system-heterogeneity parameters for one
federated client: position, compute capability, wireless link parameters,
and dataset size. It is created once at simulation startup and treated as
immutable for the duration of the experiment.

Physical units are encoded in every field name (e.g. _m for metres, _hz for
hertz, _w for watts, _db/_dbm for decibel quantities).
"""

import math
from dataclasses import dataclass, field


@dataclass
class ClientSystemProfile:
    """
    Immutable system profile for one federated learning client.

    All quantities are in SI units unless stated otherwise.
    Shadowing is drawn once at profile creation and frozen here —
    the channel model reads profile.shadowing_db rather than re-drawing.

    This class does NOT:
    - Perform any computation.
    - Hold references to datasets or model weights.
    - Change after creation (treat as frozen).
    """

    client_id: int

    # ---- Position (base station at origin) --------------------------------
    x_m: float          # x-coordinate in metres
    y_m: float          # y-coordinate in metres
    distance_m: float   # Euclidean distance to BS in metres (derived from x, y)

    # ---- Compute capability -----------------------------------------------
    cpu_frequency_hz: float     # f_k: CPU clock speed in cycles per second
    cycles_per_sample: float    # C_k: CPU cycles required per training sample

    # ---- Wireless link ----------------------------------------------------
    tx_power_w: float       # p_k: average transmit power in watts
    shadowing_db: float     # X_k: log-normal shadow fading component in dB
                            #      drawn from N(0, sigma_db^2) once at creation

    # ---- Dataset ----------------------------------------------------------
    num_samples: int        # D_k: number of local training samples

    @staticmethod
    def from_position(
        client_id: int,
        x_m: float,
        y_m: float,
        cpu_frequency_hz: float,
        cycles_per_sample: float,
        tx_power_w: float,
        shadowing_db: float,
        num_samples: int,
    ) -> "ClientSystemProfile":
        """
        Construct a profile from raw (x, y) coordinates; distance is computed.

        Args:
            client_id (int): unique client identifier.
            x_m (float): x-position in metres.
            y_m (float): y-position in metres.
            cpu_frequency_hz (float): CPU frequency f_k in Hz.
            cycles_per_sample (float): cycles per sample C_k.
            tx_power_w (float): transmit power p_k in watts.
            shadowing_db (float): frozen shadowing draw X_k in dB.
            num_samples (int): local dataset size D_k.

        Returns:
            ClientSystemProfile
        """
        distance_m = math.sqrt(x_m ** 2 + y_m ** 2)
        return ClientSystemProfile(
            client_id=client_id,
            x_m=x_m,
            y_m=y_m,
            distance_m=distance_m,
            cpu_frequency_hz=cpu_frequency_hz,
            cycles_per_sample=cycles_per_sample,
            tx_power_w=tx_power_w,
            shadowing_db=shadowing_db,
            num_samples=num_samples,
        )
