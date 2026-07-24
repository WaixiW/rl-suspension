"""Versioned contracts at the private-server integration boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ActionSchema:
    """Physical ordering, bounds, and slew limits for twelve commands."""

    version: str = "action.direct12.v1"
    names: tuple[str, ...] = (
        "fl_current_compression_a",
        "fl_current_rebound_a",
        "fl_pump_rpm",
        "fr_current_compression_a",
        "fr_current_rebound_a",
        "fr_pump_rpm",
        "rl_current_compression_a",
        "rl_current_rebound_a",
        "rl_pump_rpm",
        "rr_current_compression_a",
        "rr_current_rebound_a",
        "rr_pump_rpm",
    )
    minimum: tuple[float, ...] = (0.0, 0.0, 0.0) * 4
    maximum: tuple[float, ...] = (2.0, 2.0, 5000.0) * 4
    slew_per_second: tuple[float, ...] = (10.0, 10.0, 12000.0) * 4
    safe_action: tuple[float, ...] = (0.0, 0.0, 0.0) * 4

    @property
    def dimension(self) -> int:
        return len(self.names)

    @classmethod
    def load(cls, path: str | Path) -> "ActionSchema":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("names", "minimum", "maximum", "slew_per_second", "safe_action"):
            if key in payload:
                payload[key] = tuple(payload[key])
        return cls(**payload)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def validate(self, action: NDArray[np.floating], *, bounded: bool = True) -> FloatArray:
        value = np.asarray(action, dtype=np.float64)
        if value.shape != (self.dimension,):
            raise ValueError(f"action must have shape ({self.dimension},), got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError("action contains NaN or Inf")
        if bounded:
            low, high = np.asarray(self.minimum), np.asarray(self.maximum)
            if np.any(value < low - 1e-9) or np.any(value > high + 1e-9):
                raise ValueError("action violates physical bounds")
        return value

    def normalize(self, action: NDArray[np.floating]) -> FloatArray:
        value = self.validate(action)
        low, high = np.asarray(self.minimum), np.asarray(self.maximum)
        return ((value - low) / np.maximum(high - low, 1e-12)).astype(np.float64)

    def denormalize(self, action_01: NDArray[np.floating]) -> FloatArray:
        value = np.asarray(action_01, dtype=np.float64)
        if value.shape != (self.dimension,) or not np.all(np.isfinite(value)):
            raise ValueError("normalized action is invalid")
        low, high = np.asarray(self.minimum), np.asarray(self.maximum)
        return (low + np.clip(value, 0.0, 1.0) * (high - low)).astype(np.float64)

    def project(
        self,
        action: NDArray[np.floating],
        previous_action: NDArray[np.floating],
        dt: float,
    ) -> FloatArray:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        proposed = self.validate(action, bounded=False)
        previous = self.validate(previous_action)
        low, high = np.asarray(self.minimum), np.asarray(self.maximum)
        max_delta = np.asarray(self.slew_per_second) * dt
        bounded = np.clip(proposed, low, high)
        return (previous + np.clip(bounded - previous, -max_delta, max_delta)).astype(
            np.float64
        )


@dataclass(frozen=True)
class ObservationSchema:
    version: str = "observation.preview12.v1"
    vehicle_state_dim: int = 14
    sensor_feature_dim: int = 8
    actuator_state_dim: int = 16
    action_dim: int = 12
    road_points: int = 217
    road_resolution_m: float = 0.05
    road_start_m: float = -2.8
    road_stop_m: float = 8.0
    control_period_s: float = 0.01

    @property
    def state_vector_dim(self) -> int:
        return (
            self.vehicle_state_dim
            + self.sensor_feature_dim
            + self.actuator_state_dim
            + self.action_dim
            + 1
            + 2
        )

    @property
    def road_channels(self) -> int:
        return 4

    @classmethod
    def load(cls, path: str | Path) -> "ObservationSchema":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ObservationV1:
    timestamp_ns: int
    vehicle_state: FloatArray
    sensor_features: FloatArray
    actuator_state: FloatArray
    previous_action_12d: FloatArray
    speed_mps: float
    road_left_m: FloatArray
    road_right_m: FloatArray
    road_validity: FloatArray
    sensor_validity: FloatArray = field(
        default_factory=lambda: np.ones(2, dtype=np.float64)
    )
    schema_version: str = "observation.preview12.v1"

    def validate(
        self,
        schema: ObservationSchema,
        action_schema: ActionSchema,
    ) -> "ObservationV1":
        expected = {
            "vehicle_state": (schema.vehicle_state_dim,),
            "sensor_features": (schema.sensor_feature_dim,),
            "actuator_state": (schema.actuator_state_dim,),
            "previous_action_12d": (schema.action_dim,),
            "road_left_m": (schema.road_points,),
            "road_right_m": (schema.road_points,),
            "road_validity": (2, schema.road_points),
            "sensor_validity": (2,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains NaN or Inf")
        if self.schema_version != schema.version:
            raise ValueError(
                f"observation schema {self.schema_version!r} != {schema.version!r}"
            )
        if not np.isfinite(self.speed_mps) or self.speed_mps < 0.0:
            raise ValueError("speed_mps must be finite and nonnegative")
        action_schema.validate(self.previous_action_12d)
        if np.any(np.asarray(self.road_validity) < 0.0) or np.any(
            np.asarray(self.road_validity) > 1.0
        ):
            raise ValueError("road_validity must be within [0, 1]")
        return self

    def state_vector(self) -> FloatArray:
        return np.concatenate(
            [
                np.asarray(self.vehicle_state, dtype=np.float64),
                np.asarray(self.sensor_features, dtype=np.float64),
                np.asarray(self.actuator_state, dtype=np.float64),
                np.asarray(self.previous_action_12d, dtype=np.float64),
                np.array([self.speed_mps], dtype=np.float64),
                np.asarray(self.sensor_validity, dtype=np.float64),
            ]
        )

    def road_tensor(self) -> FloatArray:
        return np.stack(
            [
                np.asarray(self.road_left_m, dtype=np.float64),
                np.asarray(self.road_right_m, dtype=np.float64),
                np.asarray(self.road_validity[0], dtype=np.float64),
                np.asarray(self.road_validity[1], dtype=np.float64),
            ],
            axis=0,
        )


@dataclass(frozen=True)
class SolverDiagnostics:
    status: str
    objective: float
    iterations: int
    solve_time_ms: float
    feasibility_margin: float
    fallback: bool = False
    timeout: bool = False
    extra: dict[str, float | int | str | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class MpcSolveResult:
    action_12d: FloatArray
    valid: bool
    diagnostics: SolverDiagnostics
    horizon_summary: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class StepDiagnostics:
    reward: float
    reward_components: dict[str, float]
    body_acceleration: float
    pitch_acceleration: float
    roll_acceleration: float
    suspension_travel: FloatArray
    tire_loads: FloatArray
    constraint_violations: dict[str, float]
    terminated: bool = False
    truncated: bool = False
    extra: dict[str, float | int | str | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulatorStepResult:
    observation: ObservationV1
    diagnostics: StepDiagnostics


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int
    split: str
    bump_family: str
    parameters: dict[str, Any]
    version: str = "scenario.v1"


@runtime_checkable
class MpcAdapter(Protocol):
    name: str

    def reset(self, scenario: Scenario, simulator_snapshot: bytes) -> None: ...

    def solve(
        self,
        observation: ObservationV1,
        simulator_snapshot: bytes,
    ) -> MpcSolveResult: ...


@runtime_checkable
class SimulatorAdapter(Protocol):
    name: str
    done: bool

    def reset(self, scenario: Scenario, seed: int) -> ObservationV1: ...

    def step(self, action_12d: FloatArray) -> SimulatorStepResult: ...

    def snapshot(self) -> bytes: ...

    def restore(self, snapshot: bytes) -> None: ...


@runtime_checkable
class PolicyAdapter(Protocol):
    name: str

    def predict(self, observation: ObservationV1) -> FloatArray: ...


@runtime_checkable
class SafeControllerAdapter(PolicyAdapter, Protocol):
    pass


def contract_payload(
    observation_schema: ObservationSchema,
    action_schema: ActionSchema,
) -> dict[str, Any]:
    return {
        "observation_schema": asdict(observation_schema),
        "action_schema": asdict(action_schema),
    }


DEFAULT_ACTION_SCHEMA = ActionSchema()
DEFAULT_OBSERVATION_SCHEMA = ObservationSchema()
