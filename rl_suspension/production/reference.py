"""Deterministic reference adapters used for contract and pipeline smoke tests.

These are not vehicle models. Private deployments replace them via plugin
factories while preserving the production contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import pickle

import numpy as np

from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    DEFAULT_OBSERVATION_SCHEMA,
    ActionSchema,
    MpcSolveResult,
    ObservationSchema,
    ObservationV1,
    Scenario,
    SimulatorStepResult,
    SolverDiagnostics,
    StepDiagnostics,
)


@dataclass
class ReferenceDirect12Simulator:
    action_schema: ActionSchema = field(default_factory=lambda: DEFAULT_ACTION_SCHEMA)
    observation_schema: ObservationSchema = field(
        default_factory=lambda: DEFAULT_OBSERVATION_SCHEMA
    )
    name: str = "reference_direct12_simulator"

    def __post_init__(self) -> None:
        self.done = False
        self._scenario = Scenario("uninitialized", 0, "train", "flat", {})
        self._step = 0
        self._vehicle_x = 0.0
        self._vehicle_state = np.zeros(14, dtype=np.float64)
        self._actual_action = np.asarray(
            self.action_schema.safe_action,
            dtype=np.float64,
        )
        self._previous_action = self._actual_action.copy()

    def reset(self, scenario: Scenario, seed: int) -> ObservationV1:
        del seed
        self._scenario = scenario
        self._step = 0
        self._vehicle_x = 0.0
        self.done = False
        self._vehicle_state.fill(0.0)
        self._actual_action = np.asarray(
            self.action_schema.safe_action,
            dtype=np.float64,
        )
        self._previous_action = self._actual_action.copy()
        return self._observation()

    def step(self, action_12d: np.ndarray) -> SimulatorStepResult:
        action = self.action_schema.validate(action_12d)
        dt = self.observation_schema.control_period_s
        alpha = 0.25
        self._actual_action += alpha * (action - self._actual_action)
        road = self._road_height(self._vehicle_x)
        pump_indices = np.array([2, 5, 8, 11])
        pump_fraction = np.mean(
            self._actual_action[pump_indices]
            / np.asarray(self.action_schema.maximum)[pump_indices]
        )
        body_acceleration = 45.0 * road - 1.6 * pump_fraction - 0.8 * self._vehicle_state[7]
        self._vehicle_state[7] += body_acceleration * dt
        self._vehicle_state[0] += self._vehicle_state[7] * dt
        self._vehicle_state[8:10] *= 0.98
        self._vehicle_state[1:3] += self._vehicle_state[8:10] * dt

        speed = float(self._scenario.parameters.get("speed_mps", 12.0))
        self._vehicle_x += speed * dt
        self._step += 1
        episode_steps = int(self._scenario.parameters.get("episode_steps", 100))
        self.done = self._step >= episode_steps
        travel = np.repeat(self._vehicle_state[0], 4)
        tire_loads = np.full(4, 3500.0 - 5000.0 * abs(road))
        violations = {
            "suspension_travel": float(np.maximum(np.abs(travel) - 0.12, 0.0).sum()),
            "tire_lift": float(np.maximum(100.0 - tire_loads, 0.0).sum()),
        }
        action_rate = np.mean(np.abs(action - self._previous_action)) / dt
        reward_components = {
            "body_acceleration": -(body_acceleration**2),
            "action_rate": -1e-4 * float(action_rate),
            "violation": -20.0 * sum(violations.values()),
        }
        reward = float(sum(reward_components.values()))
        self._previous_action = action.copy()
        return SimulatorStepResult(
            observation=self._observation(),
            diagnostics=StepDiagnostics(
                reward=reward,
                reward_components=reward_components,
                body_acceleration=float(body_acceleration),
                pitch_acceleration=0.0,
                roll_acceleration=0.0,
                suspension_travel=travel,
                tire_loads=tire_loads,
                constraint_violations=violations,
                truncated=self.done,
            ),
        )

    def snapshot(self) -> bytes:
        return pickle.dumps(
            (
                self._scenario,
                self._step,
                self._vehicle_x,
                self._vehicle_state,
                self._actual_action,
                self._previous_action,
                self.done,
            ),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    def restore(self, snapshot: bytes) -> None:
        (
            self._scenario,
            self._step,
            self._vehicle_x,
            self._vehicle_state,
            self._actual_action,
            self._previous_action,
            self.done,
        ) = pickle.loads(snapshot)

    def _observation(self) -> ObservationV1:
        offsets = np.linspace(
            self.observation_schema.road_start_m,
            self.observation_schema.road_stop_m,
            self.observation_schema.road_points,
        )
        left = np.asarray(
            [self._road_height(self._vehicle_x + offset) for offset in offsets],
            dtype=np.float64,
        )
        asymmetry = float(self._scenario.parameters.get("asymmetry", 0.0))
        right = np.maximum(left - asymmetry * (left > 0.0), 0.0)
        actuator_state = np.concatenate(
            [self._actual_action, np.zeros(4, dtype=np.float64)]
        )
        return ObservationV1(
            timestamp_ns=int(
                self._step * self.observation_schema.control_period_s * 1e9
            ),
            vehicle_state=self._vehicle_state.copy(),
            sensor_features=np.zeros(
                self.observation_schema.sensor_feature_dim,
                dtype=np.float64,
            ),
            actuator_state=actuator_state,
            previous_action_12d=self._previous_action.copy(),
            speed_mps=float(self._scenario.parameters.get("speed_mps", 12.0)),
            road_left_m=left,
            road_right_m=right,
            road_validity=np.ones((2, self.observation_schema.road_points)),
        )

    def _road_height(self, x: float) -> float:
        start = float(self._scenario.parameters.get("bump_start_m", 2.0))
        width = float(self._scenario.parameters.get("bump_width_m", 0.5))
        height = float(self._scenario.parameters.get("bump_height_m", 0.05))
        if self._scenario.bump_family == "flat":
            return 0.0
        value = 0.0
        if start <= x <= start + width:
            phase = (x - start) / max(width, 1e-9)
            value += 0.5 * height * (1.0 - np.cos(2.0 * np.pi * phase))
        if self._scenario.bump_family == "double_bump":
            second_start = start + width + float(
                self._scenario.parameters.get("double_spacing_m", 1.0)
            )
            if second_start <= x <= second_start + width:
                second_phase = (x - second_start) / width
                value += 0.5 * height * (
                    1.0 - np.cos(2.0 * np.pi * second_phase)
                )
        return float(value)


@dataclass
class ReferenceMpcAdapter:
    action_schema: ActionSchema = field(default_factory=lambda: DEFAULT_ACTION_SCHEMA)
    observation_schema: ObservationSchema = field(
        default_factory=lambda: DEFAULT_OBSERVATION_SCHEMA
    )
    name: str = "reference_mpc"

    def reset(self, scenario: Scenario, simulator_snapshot: bytes) -> None:
        del scenario, simulator_snapshot

    def solve(
        self,
        observation: ObservationV1,
        simulator_snapshot: bytes,
    ) -> MpcSolveResult:
        del simulator_snapshot
        peak = float(
            max(np.max(observation.road_left_m), np.max(observation.road_right_m))
        )
        normalized = np.zeros(12, dtype=np.float64)
        for pump_index in (2, 5, 8, 11):
            normalized[pump_index] = np.clip(peak / 0.08, 0.0, 1.0)
        for current_index in (0, 1, 3, 4, 6, 7, 9, 10):
            normalized[current_index] = 0.4 * np.clip(peak / 0.08, 0.0, 1.0)
        requested = self.action_schema.denormalize(normalized)
        action = self.action_schema.project(
            requested,
            observation.previous_action_12d,
            self.observation_schema.control_period_s,
        )
        return MpcSolveResult(
            action_12d=action,
            valid=True,
            diagnostics=SolverDiagnostics(
                status="optimal",
                objective=peak,
                iterations=10,
                solve_time_ms=2.0,
                feasibility_margin=1.0,
            ),
            horizon_summary={"road_peak_m": peak},
        )
