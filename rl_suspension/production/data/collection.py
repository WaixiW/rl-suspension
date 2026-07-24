"""Direct-12D expert collection against black-box production protocols."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Iterable, Union

import numpy as np
from numpy.typing import NDArray

from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    DEFAULT_OBSERVATION_SCHEMA,
    ActionSchema,
    MpcAdapter,
    ObservationSchema,
    ObservationV1,
    PolicyAdapter,
    Scenario,
    SimulatorAdapter,
)
from rl_suspension.production.data.dataset import EpisodeRecord, EpisodeShardWriter


class EpisodePhase(IntEnum):
    FLAT = 0
    PREVIEW = 1
    FRONT_CONTACT = 2
    REAR_CONTACT = 3
    RECOVERY = 4


PHASE_NAMES = {
    EpisodePhase.FLAT: "flat",
    EpisodePhase.PREVIEW: "preview",
    EpisodePhase.FRONT_CONTACT: "front_contact",
    EpisodePhase.REAR_CONTACT: "rear_contact",
    EpisodePhase.RECOVERY: "recovery",
}

PhaseClassifier = Callable[[ObservationV1], Union[EpisodePhase, int]]


@dataclass(frozen=True)
class CollectionSummary:
    requested: int
    collected: int
    resumed: int
    transitions: int
    records: tuple[EpisodeRecord, ...]


def infer_episode_phase(
    observation: ObservationV1,
    *,
    observation_schema: ObservationSchema = DEFAULT_OBSERVATION_SCHEMA,
    height_threshold_m: float = 1e-5,
) -> EpisodePhase:
    """Infer preview/contact/recovery phase from the spatial road window."""

    offsets = np.linspace(
        observation_schema.road_start_m,
        observation_schema.road_stop_m,
        observation_schema.road_points,
    )
    road = np.maximum(
        np.abs(np.asarray(observation.road_left_m, dtype=np.float64)),
        np.abs(np.asarray(observation.road_right_m, dtype=np.float64)),
    )
    active = road > height_threshold_m
    if not np.any(active):
        return EpisodePhase.FLAT
    current = int(np.argmin(np.abs(offsets)))
    rear = 0
    contact_radius = max(1, int(round(0.15 / observation_schema.road_resolution_m)))
    if np.any(active[max(0, current - contact_radius) : current + contact_radius + 1]):
        return EpisodePhase.FRONT_CONTACT
    if np.any(active[rear : rear + contact_radius + 1]):
        return EpisodePhase.REAR_CONTACT
    if np.any(active[offsets > 0.0]):
        return EpisodePhase.PREVIEW
    return EpisodePhase.RECOVERY


def collect_episode(
    simulator: SimulatorAdapter,
    mpc: MpcAdapter,
    scenario: Scenario,
    writer: EpisodeShardWriter,
    *,
    behavior_policy: PolicyAdapter | None = None,
    phase_classifier: PhaseClassifier | None = None,
    episode_id: str | int | None = None,
    maximum_steps: int = 100_000,
    action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
    observation_schema: ObservationSchema = DEFAULT_OBSERVATION_SCHEMA,
) -> EpisodeRecord:
    """Collect one complete episode, or return its verified existing shard."""

    existing = writer.record_for_scenario(scenario.scenario_id)
    if existing is not None:
        # write_episode performs checksum verification for resumed scenarios.
        return writer.write_episode(
            transitions={},
            scenario=scenario,
            episode_id=existing.episode_id,
        )
    if maximum_steps <= 0:
        raise ValueError("maximum_steps must be positive")

    observation = simulator.reset(scenario, scenario.seed)
    observation.validate(observation_schema, action_schema)
    mpc.reset(scenario, simulator.snapshot())
    reset_behavior = getattr(behavior_policy, "reset", None)
    if callable(reset_behavior):
        reset_behavior()
    previous_behavior = np.asarray(action_schema.safe_action, dtype=np.float64)
    classifier = phase_classifier or (
        lambda value: infer_episode_phase(
            value,
            observation_schema=observation_schema,
        )
    )
    rows: dict[str, list[object]] = {
        "state_observations": [],
        "road_observations": [],
        "next_state_observations": [],
        "next_road_observations": [],
        "expert_actions_physical": [],
        "expert_actions_normalized": [],
        "behavior_actions_physical": [],
        "behavior_actions_normalized": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
        "timestamps_ns": [],
        "phases": [],
        "expert_valid": [],
        "solver_status": [],
        "solver_objective": [],
        "solver_iterations": [],
        "solver_solve_time_ms": [],
        "solver_feasibility_margin": [],
        "solver_fallback": [],
        "solver_timeout": [],
        "constraint_violation": [],
    }

    for _ in range(maximum_steps):
        if simulator.done:
            break
        solve = mpc.solve(observation, simulator.snapshot())
        raw_expert, label_valid = _expert_label(
            solve.action_12d,
            solve.valid,
            action_schema,
        )
        maximum_delta = (
            np.asarray(action_schema.slew_per_second, dtype=np.float64)
            * observation_schema.control_period_s
        )
        label_valid = bool(
            label_valid
            and np.all(
                np.abs(raw_expert - observation.previous_action_12d)
                <= maximum_delta + 1e-9
            )
        )
        expert = raw_expert
        if behavior_policy is None:
            proposed_behavior = expert
        else:
            proposed_behavior = np.asarray(
                behavior_policy.predict(observation),
                dtype=np.float64,
            )
            if (
                proposed_behavior.shape != (action_schema.dimension,)
                or not np.all(np.isfinite(proposed_behavior))
            ):
                proposed_behavior = np.asarray(
                    action_schema.safe_action,
                    dtype=np.float64,
                )
        behavior = action_schema.project(
            proposed_behavior,
            previous_behavior,
            observation_schema.control_period_s,
        )
        step = simulator.step(behavior)
        step.observation.validate(observation_schema, action_schema)
        phase = int(classifier(observation))
        if phase not in {int(item) for item in EpisodePhase}:
            raise ValueError(f"phase classifier returned invalid label {phase}")
        diagnostics = solve.diagnostics
        constraint_violation = float(
            sum(
                max(float(value), 0.0)
                for value in step.diagnostics.constraint_violations.values()
            )
        )

        rows["state_observations"].append(observation.state_vector())
        rows["road_observations"].append(observation.road_tensor())
        rows["next_state_observations"].append(step.observation.state_vector())
        rows["next_road_observations"].append(step.observation.road_tensor())
        rows["expert_actions_physical"].append(expert)
        rows["expert_actions_normalized"].append(action_schema.normalize(expert))
        rows["behavior_actions_physical"].append(behavior)
        rows["behavior_actions_normalized"].append(action_schema.normalize(behavior))
        rows["rewards"].append(step.diagnostics.reward)
        rows["terminated"].append(step.diagnostics.terminated)
        rows["truncated"].append(step.diagnostics.truncated)
        rows["timestamps_ns"].append(observation.timestamp_ns)
        rows["phases"].append(phase)
        rows["expert_valid"].append(
            label_valid
            and not diagnostics.fallback
            and not diagnostics.timeout
        )
        rows["solver_status"].append(diagnostics.status)
        rows["solver_objective"].append(diagnostics.objective)
        rows["solver_iterations"].append(diagnostics.iterations)
        rows["solver_solve_time_ms"].append(diagnostics.solve_time_ms)
        rows["solver_feasibility_margin"].append(diagnostics.feasibility_margin)
        rows["solver_fallback"].append(diagnostics.fallback)
        rows["solver_timeout"].append(diagnostics.timeout)
        rows["constraint_violation"].append(constraint_violation)

        observation = step.observation
        previous_behavior = behavior
        if step.diagnostics.terminated or step.diagnostics.truncated or simulator.done:
            break
    else:
        raise RuntimeError(
            f"scenario {scenario.scenario_id!r} exceeded maximum_steps={maximum_steps}"
        )

    if not rows["rewards"]:
        raise ValueError("simulator produced an empty episode")
    transitions = {name: np.asarray(values) for name, values in rows.items()}
    return writer.write_episode(
        transitions=transitions,
        scenario=scenario,
        episode_id=episode_id,
    )


def collect_scenarios(
    simulator_factory: Callable[[], SimulatorAdapter],
    mpc: MpcAdapter,
    scenarios: Iterable[Scenario],
    writer: EpisodeShardWriter,
    **episode_kwargs: object,
) -> CollectionSummary:
    scenario_list = list(scenarios)
    records: list[EpisodeRecord] = []
    resumed = 0
    transitions = 0
    for scenario in scenario_list:
        existed = writer.is_completed(scenario.scenario_id)
        record = collect_episode(
            simulator_factory(),
            mpc,
            scenario,
            writer,
            **episode_kwargs,
        )
        records.append(record)
        resumed += int(existed)
        if not existed:
            transitions += record.transitions
    return CollectionSummary(
        requested=len(scenario_list),
        collected=len(scenario_list) - resumed,
        resumed=resumed,
        transitions=transitions,
        records=tuple(records),
    )


def _expert_label(
    action: NDArray[np.floating],
    solver_valid: bool,
    action_schema: ActionSchema,
) -> tuple[NDArray[np.float64], bool]:
    value = np.asarray(action, dtype=np.float64)
    valid = bool(solver_valid)
    try:
        action_schema.validate(value)
    except ValueError:
        valid = False
        value = np.asarray(action_schema.safe_action, dtype=np.float64)
    return value, valid
