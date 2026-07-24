"""Generic wrapper for an existing private suspension simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    DEFAULT_OBSERVATION_SCHEMA,
    ActionSchema,
    ObservationSchema,
    ObservationV1,
    Scenario,
    SimulatorStepResult,
)


@dataclass
class CallableSimulatorAdapter:
    reset_fn: Callable[[Scenario, int], ObservationV1]
    step_fn: Callable
    snapshot_fn: Callable[[], bytes]
    restore_fn: Callable[[bytes], None]
    done_fn: Callable[[], bool]
    observation_schema: ObservationSchema = DEFAULT_OBSERVATION_SCHEMA
    action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA
    name: str = "private_simulator"

    @property
    def done(self) -> bool:
        return bool(self.done_fn())

    def reset(self, scenario: Scenario, seed: int) -> ObservationV1:
        observation = self.reset_fn(scenario, seed)
        return observation.validate(self.observation_schema, self.action_schema)

    def step(self, action_12d) -> SimulatorStepResult:
        action = self.action_schema.validate(action_12d)
        result = self.step_fn(action)
        if not isinstance(result, SimulatorStepResult):
            raise TypeError("private simulator step must return SimulatorStepResult")
        result.observation.validate(self.observation_schema, self.action_schema)
        return result

    def snapshot(self) -> bytes:
        snapshot = self.snapshot_fn()
        if not isinstance(snapshot, bytes) or not snapshot:
            raise ValueError("private simulator snapshot must be nonempty bytes")
        return snapshot

    def restore(self, snapshot: bytes) -> None:
        self.restore_fn(snapshot)
