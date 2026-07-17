"""
profiles/factory.py: Factory for generating a fleet of ClientSystemProfiles.

Supports two deployment geometries and two CPU-frequency heterogeneity modes
so the factory can match different paper setups without modifying other code.

Deployment modes (deployment_shape):
  "square"          — uniform in [-L/2, L/2]^2  (original baseline)
  "circle"          — uniform in disk of radius R (many FL-over-wireless papers)
  "distance_range"  — distance to BS drawn directly ~ U[dist_min_m, dist_max_m]
                       (placed along +x so distance_m == the sampled distance),
                       matching papers that specify a device-to-BS distance range
                       rather than an area (e.g. SAFSL: d ~ U[100, 1000] m)

CPU frequency modes (cpu_freq_mode):
  "fixed"         — all clients use cpu_frequency_hz
  "discrete_ghz"  — each client draws independently from
                     {f_min, f_min+step, ..., f_max} GHz (paper setup)
  "uniform_ghz"   — each client draws CONTINUOUSLY ~ U[f_min, f_max] GHz
                     (SAFSL: f ~ U[0.1, 2] x 1e9 cycles/s)

Transmit power:
  Homogeneous by default (tx_power_dbm, one value for all clients). Pass
  tx_power_w_min/tx_power_w_max to instead draw each device's power
  CONTINUOUSLY ~ U[min, max] watts (SAFSL: p_n ~ U[0.1, 0.2] W).
"""

import math
import numpy as np
from typing import List

from flsim.profiles.client_profile import ClientSystemProfile
from flsim.channel.conversions import dbm_to_watts


def create_client_profiles(
    num_clients: int,
    num_samples_list: List[int],
    tx_power_dbm: float,
    min_distance_m: float,
    rng: np.random.RandomState,
    # --- Deployment geometry ---
    deployment_shape: str = "square",
    area_side_m: float = 500.0,       # used when deployment_shape="square"
    area_radius_m: float = 500.0,     # used when deployment_shape="circle"
    dist_min_m: float = 100.0,        # used when deployment_shape="distance_range"
    dist_max_m: float = 1000.0,       # used when deployment_shape="distance_range"
    # --- CPU frequency ---
    cpu_freq_mode: str = "fixed",
    cpu_frequency_hz: float = 2.0e9,  # used when cpu_freq_mode="fixed"
    cpu_freq_min_ghz: float = 0.1,    # discrete_ghz / uniform_ghz
    cpu_freq_max_ghz: float = 0.8,    # discrete_ghz / uniform_ghz
    cpu_freq_step_ghz: float = 0.1,   # used when cpu_freq_mode="discrete_ghz"
    # --- Transmit power (per-device range overrides tx_power_dbm when both set) ---
    tx_power_w_min: float = None,
    tx_power_w_max: float = None,
    # --- Compute cycles ---
    cycles_per_sample_min: float = 1.0e6,
    cycles_per_sample_max: float = 1.0e7,
    # --- Shadowing (used only with 3GPP path-loss channel model) ---
    shadowing_std_db: float = 0.0,
) -> List[ClientSystemProfile]:
    """
    Generate K client system profiles.

    Args:
        num_clients (int): number of clients K.
        num_samples_list (list[int]): D_k per client (from partitioner).
        tx_power_dbm (float): transmit power p_k in dBm (homogeneous).
        min_distance_m (float): minimum distance clamp (avoids g→∞ at d=0).
        rng (np.random.RandomState): seeded RNG.

        deployment_shape (str): "square" or "circle".
        area_side_m (float): square side length L in metres.
        area_radius_m (float): circle radius R in metres.

        cpu_freq_mode (str): "fixed" or "discrete_ghz".
        cpu_frequency_hz (float): fixed CPU freq for all clients (fixed mode).
        cpu_freq_min_ghz (float): lower bound in GHz (discrete_ghz mode).
        cpu_freq_max_ghz (float): upper bound in GHz (discrete_ghz mode).
        cpu_freq_step_ghz (float): step size in GHz (discrete_ghz mode).

        cycles_per_sample_min (float): lower bound C_min for C_k.
        cycles_per_sample_max (float): upper bound C_max for C_k.
        shadowing_std_db (float): σ for log-normal shadowing (0 = no shadowing).

    Returns:
        list[ClientSystemProfile]: K profiles.
    """
    assert len(num_samples_list) == num_clients

    # --- Positions ---
    if deployment_shape == "square":
        half = area_side_m / 2.0
        xs = rng.uniform(-half, half, size=num_clients)
        ys = rng.uniform(-half, half, size=num_clients)
    elif deployment_shape == "circle":
        # Uniform in disk: r = R·sqrt(U[0,1]),  θ = U[0,2π]
        angles  = rng.uniform(0.0, 2.0 * math.pi, size=num_clients)
        radii   = area_radius_m * np.sqrt(rng.uniform(0.0, 1.0, size=num_clients))
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
    elif deployment_shape == "distance_range":
        # Distance drawn DIRECTLY ~ U[dist_min_m, dist_max_m] (paper convention),
        # placed along +x so distance_m == the sampled distance exactly.
        xs = rng.uniform(dist_min_m, dist_max_m, size=num_clients)
        ys = np.zeros(num_clients)
    else:
        raise ValueError(
            f"Unknown deployment_shape '{deployment_shape}'. "
            f"Use 'square', 'circle', or 'distance_range'."
        )

    # --- CPU frequencies ---
    if cpu_freq_mode == "fixed":
        cpu_freqs = np.full(num_clients, cpu_frequency_hz)
    elif cpu_freq_mode == "discrete_ghz":
        # Build discrete set {f_min, f_min+step, ..., f_max} in GHz, then convert to Hz
        steps = round((cpu_freq_max_ghz - cpu_freq_min_ghz) / cpu_freq_step_ghz)
        choices_hz = np.array([
            (cpu_freq_min_ghz + i * cpu_freq_step_ghz) * 1e9
            for i in range(steps + 1)
        ])
        cpu_freqs = rng.choice(choices_hz, size=num_clients, replace=True)
    elif cpu_freq_mode == "uniform_ghz":
        # Continuous f_k ~ U[f_min, f_max] GHz -> Hz (paper convention)
        cpu_freqs = rng.uniform(cpu_freq_min_ghz, cpu_freq_max_ghz, size=num_clients) * 1e9
    else:
        raise ValueError(
            f"Unknown cpu_freq_mode '{cpu_freq_mode}'. "
            f"Use 'fixed', 'discrete_ghz', or 'uniform_ghz'."
        )

    # --- Transmit power ---
    # Default (unchanged): homogeneous p_k = dbm_to_watts(tx_power_dbm) for every
    # client — exactly as before, and what achievable_rate_bps reads for the data
    # rate. Opt-in: a per-device range draws p_n ~ U[min, max] W (paper convention),
    # the only way to express heterogeneous per-device transmit power.
    if tx_power_w_min is not None and tx_power_w_max is not None:
        tx_powers = rng.uniform(tx_power_w_min, tx_power_w_max, size=num_clients)
    else:
        tx_powers = np.full(num_clients, dbm_to_watts(tx_power_dbm))

    # --- Cycles per sample and shadowing ---
    cycles  = rng.uniform(cycles_per_sample_min, cycles_per_sample_max, size=num_clients)
    shadows = rng.normal(0.0, shadowing_std_db, size=num_clients) if shadowing_std_db > 0.0 \
              else np.zeros(num_clients)

    # --- Assemble profiles ---
    profiles = []
    for k in range(num_clients):
        profile = ClientSystemProfile.from_position(
            client_id=k,
            x_m=float(xs[k]),
            y_m=float(ys[k]),
            cpu_frequency_hz=float(cpu_freqs[k]),
            cycles_per_sample=float(cycles[k]),
            tx_power_w=float(tx_powers[k]),
            shadowing_db=float(shadows[k]),
            num_samples=num_samples_list[k],
        )
        if profile.distance_m < min_distance_m:
            object.__setattr__(profile, "distance_m", min_distance_m)
        profiles.append(profile)

    return profiles
