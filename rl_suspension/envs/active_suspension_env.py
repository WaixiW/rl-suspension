"""Gymnasium environment for single-agent active suspension RL."""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_suspension.actuators import ActuatorAllocator, ActuatorLimits
from rl_suspension.models import ActuatorState, SevenDofSuspensionModel, SuspensionState, VehicleParams
from rl_suspension.models.types import FloatArray, SuspensionOutput
from rl_suspension.road import BumpScenario, RoadProfile, ScenarioConfig


@dataclass(frozen=True)
class RewardWeights:
    body_acc: float = 1.0
    pitch_acc: float = 0.2
    roll_acc: float = 0.2
    suspension_travel: float = 2.0
    tire_load_variation: float = 1e-7
    force: float = 1e-8
    force_rate: float = 1e-7
    violation: float = 20.0


@dataclass(frozen=True)
class EnvConfig:
    dt: float = 0.01
    max_force: float = 5000.0
    curriculum_stage: int = 1
    randomize_scenario: bool = True
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    vehicle_params: VehicleParams = field(default_factory=VehicleParams)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    actuator_limits: ActuatorLimits = field(default_factory=ActuatorLimits)


class ActiveSuspensionEnv(gym.Env):
    """Single centralized agent: observation -> four desired corner forces."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.model = SevenDofSuspensionModel(self.config.vehicle_params)
        self.allocator = ActuatorAllocator(self.config.actuator_limits)
        self.scenario_factory = BumpScenario(self.config.scenario)
        self.state = SuspensionState.zeros()
        self.actuator_state = ActuatorState()
        self.road: RoadProfile = self.scenario_factory.make_profile()
        self.vehicle_x = 0.0
        self.time = 0.0
        self.previous_forces = np.zeros(4, dtype=np.float64)
        self.last_output: SuspensionOutput | None = None
        self.episode_accelerations: list[float] = []
        self.episode_forces: list[FloatArray] = []

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # 14 plant states + 4 suspension deflections + 4 suspension velocities
        # + 4 previous forces + 8 currents + 4 pump speeds + 4 actual forces
        # + speed + 7 compressed ADS features.
        self.observation_dim = 50
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[FloatArray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.scenario_factory = BumpScenario(self.config.scenario, seed=seed)
        stage = int((options or {}).get("curriculum_stage", self.config.curriculum_stage))
        self.road = (
            self.scenario_factory.randomized(stage)
            if self.config.randomize_scenario
            else self.scenario_factory.make_profile()
        )
        self.state = SuspensionState.zeros()
        self.actuator_state = ActuatorState()
        self.vehicle_x = 0.0
        self.time = 0.0
        self.previous_forces = np.zeros(4, dtype=np.float64)
        self.last_output = None
        self.episode_accelerations = []
        self.episode_forces = []
        obs = self._build_observation()
        return obs, {"scenario": self.road.config}

    def step(self, action: FloatArray) -> tuple[FloatArray, float, bool, bool, dict]:
        normalized_action = np.asarray(action, dtype=np.float64)
        if normalized_action.shape != (4,):
            raise ValueError(f"action must have shape (4,), got {normalized_action.shape}")
        desired_forces = np.clip(normalized_action, -1.0, 1.0) * self.config.max_force

        road_heights = self.road.wheel_heights(self.vehicle_x)
        suspension_velocities = (
            self.last_output.suspension_velocities if self.last_output is not None else np.zeros(4)
        )
        allocation = self.allocator.allocate(
            desired_forces=desired_forces,
            suspension_velocities=suspension_velocities,
            previous_state=self.actuator_state,
            dt=self.config.dt,
        )
        self.actuator_state = allocation.actuator_state

        result = self.model.step(
            state=self.state,
            action_12d=allocation.action_12d,
            road_profile=road_heights,
            dt=self.config.dt,
            actuator_state=self.actuator_state,
        )
        self.state = result.next_state
        self.last_output = result.output
        self.vehicle_x += self.road.config.speed * self.config.dt
        self.time += self.config.dt

        force_rate = (allocation.feasible_forces - self.previous_forces) / self.config.dt
        reward = self._reward(result.output, allocation.feasible_forces, force_rate)
        self.previous_forces = allocation.feasible_forces.copy()

        self.episode_accelerations.append(result.output.body_acceleration)
        self.episode_forces.append(allocation.feasible_forces.copy())

        terminated = self._is_unsafe(result.output)
        truncated = self.time >= self.road.config.episode_time
        obs = self._build_observation()
        info = self._info(result.output, allocation.feasible_forces, force_rate)
        return obs, reward, terminated, truncated, info

    def _build_observation(self) -> FloatArray:
        if self.last_output is None:
            suspension_deflections = np.zeros(4, dtype=np.float64)
            suspension_velocities = np.zeros(4, dtype=np.float64)
        else:
            suspension_deflections = self.last_output.suspension_deflections
            suspension_velocities = self.last_output.suspension_velocities

        obs = np.concatenate(
            [
                self.state.as_vector(),
                suspension_deflections,
                suspension_velocities,
                self.previous_forces / max(self.config.max_force, 1.0),
                self.actuator_state.currents / max(self.config.actuator_limits.current_max, 1e-6),
                self.actuator_state.pump_speeds / max(self.config.actuator_limits.pump_speed_max, 1.0),
                self.actuator_state.forces / max(self.config.max_force, 1.0),
                np.array([self.road.config.speed], dtype=np.float64),
                self.road.features(self.vehicle_x),
            ]
        ).astype(np.float32)
        if obs.shape != (self.observation_dim,):
            raise RuntimeError(f"Observation shape {obs.shape} does not match {self.observation_dim}")
        return obs

    def _reward(self, output: SuspensionOutput, forces: FloatArray, force_rate: FloatArray) -> float:
        w = self.config.reward_weights
        travel = output.suspension_deflections
        static_load = self.config.vehicle_params.unsprung_masses * self.config.vehicle_params.gravity
        tire_variation = output.tire_loads - static_load
        violation = sum(output.constraint_violations.values())
        value = (
            -w.body_acc * output.body_acceleration**2
            - w.pitch_acc * output.pitch_acceleration**2
            - w.roll_acc * output.roll_acceleration**2
            - w.suspension_travel * float(np.mean(travel**2))
            - w.tire_load_variation * float(np.mean(tire_variation**2))
            - w.force * float(np.mean(forces**2))
            - w.force_rate * float(np.mean(force_rate**2))
            - w.violation * violation
        )
        return float(value)

    def _is_unsafe(self, output: SuspensionOutput) -> bool:
        return any(value > 0.05 for value in output.constraint_violations.values())

    def _info(self, output: SuspensionOutput, forces: FloatArray, force_rate: FloatArray) -> dict:
        accelerations = np.asarray(self.episode_accelerations or [output.body_acceleration])
        return {
            "rms_body_acceleration": float(np.sqrt(np.mean(accelerations**2))),
            "peak_body_acceleration": float(np.max(np.abs(accelerations))),
            "pitch_peak": float(abs(self.state.q[1])),
            "roll_peak": float(abs(self.state.q[2])),
            "max_suspension_travel": float(np.max(np.abs(output.suspension_deflections))),
            "tire_load_min": float(np.min(output.tire_loads)),
            "actuator_energy": float(np.mean(forces**2)),
            "action_rate": float(np.mean(np.abs(force_rate))),
            "constraint_violations": dict(output.constraint_violations),
        }
