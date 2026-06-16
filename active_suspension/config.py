"""Configuration dataclasses for the active-suspension project.

All physical constants live here so that the simulator, controllers, metrics and
domain-randomization sampler share a single source of truth. The corner order is
fixed everywhere as: 0=FL, 1=FR, 2=RL, 3=RR.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List

import numpy as np

CORNER_NAMES = ("FL", "FR", "RL", "RR")
N_CORNERS = 4
GRAVITY = 9.81


@dataclass
class VehicleParams:
    """7-DOF full-car parameters (heave, pitch, roll + 4 unsprung verticals)."""

    m_s: float = 1500.0          # sprung mass [kg]
    I_pitch: float = 2160.0      # pitch inertia about y [kg m^2]
    I_roll: float = 460.0        # roll inertia about x [kg m^2]
    m_u: float = 45.0            # unsprung mass per corner [kg]

    a: float = 1.4               # CG -> front axle [m]
    b: float = 1.6               # CG -> rear axle [m]
    track_f: float = 1.55        # front track [m]
    track_r: float = 1.55        # rear track [m]

    # Suspension (passive component; active force is added on top).
    k_s_front: float = 35000.0   # spring [N/m]
    k_s_rear: float = 38000.0
    c_s_front: float = 1500.0    # base passive damping [N s/m]
    c_s_rear: float = 1600.0

    # Tire
    k_t: float = 200000.0        # tire vertical stiffness [N/m]
    c_t: float = 150.0           # tire damping [N s/m]

    # Hard limits used by constraints / safety filter
    susp_travel_limit: float = 0.08   # +/- suspension working space [m]

    def corner_positions(self) -> np.ndarray:
        """Return (4,2) array of (x, y) corner offsets from CG.

        x>0 forward, y>0 left.
        """
        xf, xr = self.a, -self.b
        yfl, yfr = self.track_f / 2.0, -self.track_f / 2.0
        yrl, yrr = self.track_r / 2.0, -self.track_r / 2.0
        return np.array(
            [[xf, yfl], [xf, yfr], [xr, yrl], [xr, yrr]], dtype=float
        )

    def k_s(self) -> np.ndarray:
        return np.array(
            [self.k_s_front, self.k_s_front, self.k_s_rear, self.k_s_rear],
            dtype=float,
        )

    def c_s(self) -> np.ndarray:
        return np.array(
            [self.c_s_front, self.c_s_front, self.c_s_rear, self.c_s_rear],
            dtype=float,
        )

    def static_corner_loads(self) -> np.ndarray:
        """Static vertical tire load per corner [N] from sprung + unsprung weight."""
        # Distribute sprung weight front/rear by CG position; left/right evenly.
        w_s = self.m_s * GRAVITY
        front_frac = self.b / (self.a + self.b)
        rear_frac = self.a / (self.a + self.b)
        sprung = np.array(
            [front_frac / 2, front_frac / 2, rear_frac / 2, rear_frac / 2]
        ) * w_s
        unsprung = self.m_u * GRAVITY
        return sprung + unsprung


@dataclass
class ActuatorParams:
    """Forward-only actuator model: (i1, i2, n_pump) per corner -> force.

    The map is intentionally non-invertible / redundant (3 inputs -> 1 scalar
    force). Damping currents shape a dissipative (semi-active) force that opposes
    the suspension relative velocity; the pump rotation speed injects an active
    force that can both push and pull. A first-order lag plus pure transport delay
    model the real force delay over time.
    """

    i_max: float = 2.0           # max damping current per valve [A]
    n_pump_max: float = 3000.0   # max pump rotation speed [rpm], +/- (bidirectional)

    c_base: float = 800.0        # base (zero-current) damping [N s/m]
    k_curr1: float = 1200.0      # current1 -> damping gain [N s/m per A]
    k_curr2: float = 1000.0      # current2 -> damping gain [N s/m per A]
    damp_vel_limit: float = 1.5  # rel-velocity saturation for damping force [m/s]

    k_pump: float = 0.9          # pump speed -> active force gain [N per rpm]

    force_sat: float = 5000.0    # |actuator force| saturation [N]
    tau: float = 0.02            # first-order force lag time constant [s]
    transport_delay_s: float = 0.01  # pure transport delay [s]


@dataclass
class PreviewParams:
    """ADS road-preview sensor description."""

    resolution_m: float = 0.05   # 5 cm along heading
    horizon_m: float = 8.0       # 8 m look-ahead
    # Confidence is highest near the car and decays with look-ahead distance.
    conf_near: float = 0.99
    conf_far: float = 0.55
    # Error injection (scaled by (1 - confidence)).
    height_noise_std: float = 0.004   # [m] at far range
    dropout_prob: float = 0.02
    latency_s: float = 0.0
    lateral_jitter_m: float = 0.02
    # Downsampling for the RL observation (use every k-th sample).
    obs_stride: int = 4

    @property
    def n_samples(self) -> int:
        return int(round(self.horizon_m / self.resolution_m))


@dataclass
class SimConfig:
    """Top-level simulation / episode configuration."""

    dt: float = 0.001            # integration step [s] (1 kHz)
    control_decimation: int = 5  # controller runs every N steps -> 200 Hz control
    speed: float = 15.0          # nominal forward speed [m/s]
    episode_time: float = 4.0    # [s]
    history_len: int = 5         # action/obs history window for the policy
    seed: int = 0

    @property
    def control_dt(self) -> float:
        return self.dt * self.control_decimation

    @property
    def n_steps(self) -> int:
        return int(round(self.episode_time / self.dt))


@dataclass
class FullConfig:
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    actuator: ActuatorParams = field(default_factory=ActuatorParams)
    preview: PreviewParams = field(default_factory=PreviewParams)
    sim: SimConfig = field(default_factory=SimConfig)

    def copy(self) -> "FullConfig":
        return replace(
            self,
            vehicle=replace(self.vehicle),
            actuator=replace(self.actuator),
            preview=replace(self.preview),
            sim=replace(self.sim),
        )


def default_config() -> FullConfig:
    return FullConfig()
