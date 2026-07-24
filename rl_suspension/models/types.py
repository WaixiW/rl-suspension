"""Shared state and parameter types for the suspension simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class VehicleParams:
    """Physical parameters for a 7-DOF vertical vehicle model."""

    sprung_mass: float = 1200.0
    unsprung_masses: FloatArray = field(
        default_factory=lambda: np.array([45.0, 45.0, 50.0, 50.0], dtype=np.float64)
    )
    pitch_inertia: float = 2100.0
    roll_inertia: float = 650.0
    suspension_stiffness: FloatArray = field(
        default_factory=lambda: np.array([28000.0, 28000.0, 24000.0, 24000.0], dtype=np.float64)
    )
    tire_stiffness: FloatArray = field(
        default_factory=lambda: np.array([210000.0, 210000.0, 220000.0, 220000.0], dtype=np.float64)
    )
    passive_damping: FloatArray = field(
        default_factory=lambda: np.array([1600.0, 1600.0, 1800.0, 1800.0], dtype=np.float64)
    )
    half_wheelbase_front: float = 1.35
    half_wheelbase_rear: float = 1.45
    half_track_front: float = 0.78
    half_track_rear: float = 0.78
    gravity: float = 9.81
    suspension_travel_limit: float = 0.12
    tire_load_min: float = 100.0

    @property
    def corner_x(self) -> FloatArray:
        """Corner longitudinal offsets from CG: FL, FR, RL, RR."""

        return np.array(
            [
                self.half_wheelbase_front,
                self.half_wheelbase_front,
                -self.half_wheelbase_rear,
                -self.half_wheelbase_rear,
            ],
            dtype=np.float64,
        )

    @property
    def corner_y(self) -> FloatArray:
        """Corner lateral offsets from CG: FL, FR, RL, RR."""

        return np.array(
            [
                self.half_track_front,
                -self.half_track_front,
                self.half_track_rear,
                -self.half_track_rear,
            ],
            dtype=np.float64,
        )


@dataclass
class SuspensionState:
    """7-DOF model state.

    Generalized coordinates are body heave, pitch, roll, and four unsprung
    vertical displacements. Velocities follow the same ordering.
    """

    q: FloatArray
    qd: FloatArray

    @classmethod
    def zeros(cls) -> "SuspensionState":
        return cls(q=np.zeros(7, dtype=np.float64), qd=np.zeros(7, dtype=np.float64))

    def copy(self) -> "SuspensionState":
        return SuspensionState(q=self.q.copy(), qd=self.qd.copy())

    def as_vector(self) -> FloatArray:
        return np.concatenate([self.q, self.qd]).astype(np.float64)

    @classmethod
    def from_vector(cls, x: FloatArray) -> "SuspensionState":
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape != (14,):
            raise ValueError(f"SuspensionState vector must have shape (14,), got {arr.shape}")
        return cls(q=arr[:7].copy(), qd=arr[7:].copy())


@dataclass
class ActuatorState:
    """Per-corner actuator memory used for lag and rate limits."""

    currents: FloatArray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    pump_speeds: FloatArray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    forces: FloatArray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))

    def copy(self) -> "ActuatorState":
        return ActuatorState(
            currents=self.currents.copy(),
            pump_speeds=self.pump_speeds.copy(),
            forces=self.forces.copy(),
        )

    def as_vector(self) -> FloatArray:
        return np.concatenate([self.currents, self.pump_speeds, self.forces]).astype(np.float64)


@dataclass(frozen=True)
class SuspensionOutput:
    """Diagnostics returned by one simulator step."""

    body_acceleration: float
    pitch_acceleration: float
    roll_acceleration: float
    suspension_deflections: FloatArray
    suspension_velocities: FloatArray
    tire_deflections: FloatArray
    tire_loads: FloatArray
    actual_corner_forces: FloatArray
    actuator_state: ActuatorState
    constraint_violations: dict[str, float]


@dataclass(frozen=True)
class StepResult:
    """Standard plant step result."""

    next_state: SuspensionState
    output: SuspensionOutput
