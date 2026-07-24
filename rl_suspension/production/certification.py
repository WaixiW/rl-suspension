"""Contract certification for private MPC and simulator adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    DEFAULT_OBSERVATION_SCHEMA,
    ActionSchema,
    MpcAdapter,
    ObservationSchema,
    Scenario,
    SimulatorAdapter,
)


@dataclass(frozen=True)
class CertificationReport:
    passed: bool
    checks: dict[str, bool]
    maximum_replay_error: float
    solver_status: str
    action_schema_version: str
    observation_schema_version: str

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def certify_integration(
    mpc: MpcAdapter,
    simulator: SimulatorAdapter,
    scenario: Scenario,
    *,
    action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
    observation_schema: ObservationSchema = DEFAULT_OBSERVATION_SCHEMA,
    replay_tolerance: float = 1e-10,
) -> CertificationReport:
    observation = simulator.reset(scenario, scenario.seed)
    observation.validate(observation_schema, action_schema)
    snapshot = simulator.snapshot()
    if not isinstance(snapshot, bytes) or not snapshot:
        raise ValueError("simulator snapshot must be nonempty bytes")

    mpc.reset(scenario, snapshot)
    solve = mpc.solve(observation, snapshot)
    action = action_schema.validate(solve.action_12d)
    first = simulator.step(action)
    first.observation.validate(observation_schema, action_schema)

    simulator.restore(snapshot)
    replay = simulator.step(action)
    replay.observation.validate(observation_schema, action_schema)
    replay_error = _maximum_observation_error(first.observation, replay.observation)

    projected = action_schema.project(
        np.asarray(action_schema.maximum, dtype=np.float64) * 2.0,
        np.asarray(action_schema.safe_action, dtype=np.float64),
        observation_schema.control_period_s,
    )
    max_delta = (
        np.asarray(action_schema.slew_per_second)
        * observation_schema.control_period_s
    )
    projected_delta = np.abs(
        projected - np.asarray(action_schema.safe_action, dtype=np.float64)
    )

    checks = {
        "observation_contract": True,
        "action_contract": True,
        "solver_valid": bool(solve.valid and not solve.diagnostics.fallback),
        "deterministic_snapshot_replay": replay_error <= replay_tolerance,
        "action_projection_bounds": _within_bounds(projected, action_schema),
        "action_projection_slew": bool(np.all(projected_delta <= max_delta + 1e-12)),
        "finite_step_diagnostics": _finite_diagnostics(first.diagnostics.reward_components.values()),
    }
    return CertificationReport(
        passed=all(checks.values()),
        checks=checks,
        maximum_replay_error=replay_error,
        solver_status=solve.diagnostics.status,
        action_schema_version=action_schema.version,
        observation_schema_version=observation_schema.version,
    )


def _maximum_observation_error(first, second) -> float:
    values = [
        np.max(np.abs(first.state_vector() - second.state_vector())),
        np.max(np.abs(first.road_tensor() - second.road_tensor())),
    ]
    return float(max(values))


def _within_bounds(action: np.ndarray, schema: ActionSchema) -> bool:
    return bool(
        np.all(action >= np.asarray(schema.minimum) - 1e-12)
        and np.all(action <= np.asarray(schema.maximum) + 1e-12)
    )


def _finite_diagnostics(values: Iterable[float]) -> bool:
    return bool(np.all(np.isfinite(np.asarray(list(values), dtype=np.float64))))
