"""Five-round DAgger orchestration over production adapter contracts."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
from threading import BoundedSemaphore
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np

from rl_suspension.production.adapters.safe_controller import SafetyProjector
from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    DEFAULT_OBSERVATION_SCHEMA,
    ActionSchema,
    MpcAdapter,
    MpcSolveResult,
    ObservationSchema,
    ObservationV1,
    PolicyAdapter,
    Scenario,
    SimulatorAdapter,
)
from rl_suspension.production.training.bc import (
    BCDatasetArrays,
    concatenate_datasets,
)


DEFAULT_BETA_SCHEDULE = (1.0, 0.75, 0.5, 0.25, 0.0)


class RoundTrainer(Protocol):
    def __call__(
        self,
        aggregate: BCDatasetArrays,
        fixed_validation: BCDatasetArrays,
        round_index: int,
    ) -> PolicyAdapter: ...


class ValidationEvaluator(Protocol):
    def __call__(
        self,
        policy: PolicyAdapter,
        fixed_validation: BCDatasetArrays,
    ) -> float: ...


class SnapshotQueryQueue:
    """Bounded asynchronous MPC queries against immutable simulator snapshots."""

    def __init__(
        self,
        expert: MpcAdapter,
        *,
        max_workers: int = 1,
        max_pending: int = 32,
    ) -> None:
        if max_workers <= 0 or max_pending <= 0:
            raise ValueError("max_workers and max_pending must be positive")
        self.expert = expert
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="snapshot-expert",
        )
        self._slots = BoundedSemaphore(max_pending)
        self._closed = False

    def submit(
        self,
        observation: ObservationV1,
        simulator_snapshot: bytes,
    ) -> Future[MpcSolveResult]:
        if self._closed:
            raise RuntimeError("snapshot query queue is closed")
        if not isinstance(simulator_snapshot, bytes) or not simulator_snapshot:
            raise ValueError("simulator snapshot must be nonempty bytes")
        self._slots.acquire()
        try:
            future = self._executor.submit(
                self.expert.solve,
                observation,
                simulator_snapshot,
            )
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        return future

    def close(self, *, wait: bool = True) -> None:
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def __enter__(self) -> "SnapshotQueryQueue":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(wait=True)


@dataclass(frozen=True)
class DaggerConfig:
    beta_schedule: tuple[float, ...] = DEFAULT_BETA_SCHEDULE
    episodes_per_round: int = 1
    maximum_steps_per_episode: int = 10_000
    seed: int = 0
    asynchronous_queries: bool = False
    query_workers: int = 1
    max_pending_queries: int = 32
    validation_regression_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if tuple(self.beta_schedule) != DEFAULT_BETA_SCHEDULE:
            raise ValueError(
                "production DAgger uses five rounds with beta schedule "
                f"{list(DEFAULT_BETA_SCHEDULE)}"
            )
        if self.episodes_per_round <= 0 or self.maximum_steps_per_episode <= 0:
            raise ValueError("episode and step counts must be positive")
        if self.query_workers <= 0 or self.max_pending_queries <= 0:
            raise ValueError("query queue limits must be positive")
        if self.validation_regression_tolerance < 0.0:
            raise ValueError("validation_regression_tolerance must be nonnegative")


@dataclass(frozen=True)
class DaggerRoundReport:
    round_index: int
    beta: float
    transitions: int
    expert_steps: int
    invalid_labels: int
    validation_score: float
    accepted: bool
    aggregate_size: int


@dataclass(frozen=True)
class DaggerResult:
    final_policy: PolicyAdapter
    aggregate: BCDatasetArrays
    reports: tuple[DaggerRoundReport, ...]
    best_validation_score: float


@dataclass
class _EpisodeBuffers:
    states: list[np.ndarray] = field(default_factory=list)
    roads: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    phases: list[int] = field(default_factory=list)
    quality: list[float] = field(default_factory=list)
    episode_ids: list[int] = field(default_factory=list)
    sequence_indices: list[int] = field(default_factory=list)
    valid: list[bool] = field(default_factory=list)


def default_phase_classifier(observation: ObservationV1) -> int:
    """Classify flat versus preview event without simulator-specific state."""

    valid = np.asarray(observation.road_validity, dtype=np.float64)
    road = np.stack((observation.road_left_m, observation.road_right_m))
    visible_peak = float(np.max(np.abs(road) * valid))
    return 0 if visible_peak < 1e-5 else 1


class DaggerTrainer:
    """Collect expert labels, retrain, and reject validation regressions."""

    def __init__(
        self,
        *,
        simulator_factory: Callable[[], SimulatorAdapter],
        expert: MpcAdapter,
        initial_policy: PolicyAdapter,
        train_round: RoundTrainer,
        validation_evaluator: ValidationEvaluator,
        config: DaggerConfig | None = None,
        safety_projector: SafetyProjector | None = None,
        action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
        observation_schema: ObservationSchema = DEFAULT_OBSERVATION_SCHEMA,
        phase_classifier: Callable[[ObservationV1], int] = default_phase_classifier,
    ) -> None:
        self.simulator_factory = simulator_factory
        self.expert = expert
        self.initial_policy = initial_policy
        self.train_round = train_round
        self.validation_evaluator = validation_evaluator
        self.config = config or DaggerConfig()
        self.action_schema = action_schema
        self.observation_schema = observation_schema
        self.safety_projector = safety_projector or SafetyProjector(action_schema)
        self.phase_classifier = phase_classifier
        self._rng = np.random.default_rng(self.config.seed)
        self._next_episode_id = 0

    def run(
        self,
        initial_dataset: BCDatasetArrays,
        fixed_validation: BCDatasetArrays,
        scenarios: Sequence[Sequence[Scenario]],
    ) -> DaggerResult:
        if len(scenarios) != len(DEFAULT_BETA_SCHEDULE):
            raise ValueError("scenarios must provide one sequence for each DAgger round")
        if any(len(items) != self.config.episodes_per_round for items in scenarios):
            raise ValueError("each DAgger round must contain episodes_per_round scenarios")

        validation = _freeze_dataset(fixed_validation.fixed_copy())
        validation_fingerprint = _dataset_fingerprint(validation)
        aggregate = initial_dataset
        current_policy = self.initial_policy
        best_score = _finite_score(
            self.validation_evaluator(current_policy, validation)
        )
        self._next_episode_id = int(np.max(initial_dataset.episode_ids)) + 1
        reports: list[DaggerRoundReport] = []

        for round_index, beta in enumerate(DEFAULT_BETA_SCHEDULE):
            collected, expert_steps, invalid_labels = self._collect_round(
                current_policy,
                scenarios[round_index],
                beta,
            )
            aggregate = concatenate_datasets((aggregate, collected))
            candidate = self.train_round(aggregate, validation, round_index)
            if _dataset_fingerprint(validation) != validation_fingerprint:
                raise RuntimeError("the fixed validation dataset was modified")
            candidate_score = _finite_score(
                self.validation_evaluator(candidate, validation)
            )
            accepted = (
                candidate_score
                <= best_score + self.config.validation_regression_tolerance
            )
            if accepted:
                current_policy = candidate
                best_score = candidate_score
            reports.append(
                DaggerRoundReport(
                    round_index=round_index + 1,
                    beta=beta,
                    transitions=len(collected),
                    expert_steps=expert_steps,
                    invalid_labels=invalid_labels,
                    validation_score=candidate_score,
                    accepted=accepted,
                    aggregate_size=len(aggregate),
                )
            )

        return DaggerResult(
            final_policy=current_policy,
            aggregate=aggregate,
            reports=tuple(reports),
            best_validation_score=best_score,
        )

    def _collect_round(
        self,
        policy: PolicyAdapter,
        scenarios: Sequence[Scenario],
        beta: float,
    ) -> tuple[BCDatasetArrays, int, int]:
        datasets: list[BCDatasetArrays] = []
        expert_steps = 0
        invalid_labels = 0
        for scenario in scenarios:
            episode, episode_expert_steps, episode_invalid = self._collect_episode(
                policy,
                scenario,
                beta,
            )
            datasets.append(episode)
            expert_steps += episode_expert_steps
            invalid_labels += episode_invalid
        return concatenate_datasets(datasets), expert_steps, invalid_labels

    def _collect_episode(
        self,
        policy: PolicyAdapter,
        scenario: Scenario,
        beta: float,
    ) -> tuple[BCDatasetArrays, int, int]:
        simulator = self.simulator_factory()
        observation = simulator.reset(scenario, scenario.seed)
        observation.validate(self.observation_schema, self.action_schema)
        reset_policy = getattr(policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        initial_snapshot = simulator.snapshot()
        self.expert.reset(scenario, initial_snapshot)
        queue = (
            SnapshotQueryQueue(
                self.expert,
                max_workers=self.config.query_workers,
                max_pending=self.config.max_pending_queries,
            )
            if self.config.asynchronous_queries
            else None
        )
        buffers = _EpisodeBuffers()
        previous_action = np.asarray(
            observation.previous_action_12d,
            dtype=np.float64,
        )
        expert_steps = 0
        invalid_labels = 0

        try:
            for step_index in range(self.config.maximum_steps_per_episode):
                snapshot = simulator.snapshot()
                query = (
                    queue.submit(observation, snapshot)
                    if queue is not None
                    else self.expert.solve(observation, snapshot)
                )
                use_expert = bool(self._rng.random() < beta)
                student_action = np.asarray(policy.predict(observation), dtype=np.float64)
                solve = None
                if use_expert:
                    solve = query.result() if isinstance(query, Future) else query
                    expert_usable = (
                        solve.valid
                        and not solve.diagnostics.fallback
                        and not solve.diagnostics.timeout
                    )
                    proposed = solve.action_12d if expert_usable else student_action
                    expert_steps += 1
                else:
                    proposed = student_action
                behavior_action = self._project_or_safe(proposed, previous_action)
                step_result = simulator.step(behavior_action)

                if solve is None:
                    solve = query.result() if isinstance(query, Future) else query
                label, label_valid, quality = self._label(
                    solve,
                    observation.previous_action_12d,
                )
                invalid_labels += int(not label_valid)
                buffers.states.append(observation.state_vector().astype(np.float32))
                buffers.roads.append(observation.road_tensor().astype(np.float32))
                buffers.actions.append(label.astype(np.float32))
                buffers.phases.append(int(self.phase_classifier(observation)))
                buffers.quality.append(quality)
                buffers.episode_ids.append(self._next_episode_id)
                buffers.sequence_indices.append(step_index)
                buffers.valid.append(label_valid)

                observation = step_result.observation
                observation.validate(self.observation_schema, self.action_schema)
                previous_action = behavior_action
                if (
                    simulator.done
                    or step_result.diagnostics.terminated
                    or step_result.diagnostics.truncated
                ):
                    break
            else:
                raise RuntimeError("DAgger episode exceeded maximum_steps_per_episode")
        finally:
            if queue is not None:
                queue.close(wait=True)

        self._next_episode_id += 1
        if not buffers.states:
            raise RuntimeError("DAgger episode produced no transitions")
        return (
            BCDatasetArrays(
                states=np.stack(buffers.states),
                roads=np.stack(buffers.roads),
                actions=np.stack(buffers.actions),
                phases=np.asarray(buffers.phases),
                quality=np.asarray(buffers.quality),
                episode_ids=np.asarray(buffers.episode_ids),
                sequence_indices=np.asarray(buffers.sequence_indices),
                valid=np.asarray(buffers.valid),
                actions_normalized=False,
                observation_schema=self.observation_schema,
                action_schema=self.action_schema,
            ),
            expert_steps,
            invalid_labels,
        )

    def _project_or_safe(
        self,
        proposed: np.ndarray,
        previous_action: np.ndarray,
    ) -> np.ndarray:
        try:
            projected = self.safety_projector.project(
                proposed,
                previous_action,
                self.observation_schema.control_period_s,
            )
            return self.action_schema.validate(projected)
        except (TypeError, ValueError, FloatingPointError):
            return self.action_schema.project(
                np.asarray(self.action_schema.safe_action, dtype=np.float64),
                previous_action,
                self.observation_schema.control_period_s,
            )

    def _label(
        self,
        solve: MpcSolveResult,
        previous_action: np.ndarray,
    ) -> tuple[np.ndarray, bool, float]:
        valid = bool(
            solve.valid
            and not solve.diagnostics.fallback
            and not solve.diagnostics.timeout
        )
        try:
            action = self.action_schema.validate(solve.action_12d)
        except ValueError:
            valid = False
            action = np.asarray(self.action_schema.safe_action, dtype=np.float64)
        maximum_delta = (
            np.asarray(self.action_schema.slew_per_second, dtype=np.float64)
            * self.observation_schema.control_period_s
        )
        if np.any(
            np.abs(action - np.asarray(previous_action, dtype=np.float64))
            > maximum_delta + 1e-9
        ):
            valid = False
        margin = float(solve.diagnostics.feasibility_margin)
        quality = max(margin, 0.05) if valid and np.isfinite(margin) else 0.05
        return action, valid, quality


def _freeze_dataset(dataset: BCDatasetArrays) -> BCDatasetArrays:
    for value in dataset.__dict__.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    return dataset


def _dataset_fingerprint(dataset: BCDatasetArrays) -> str:
    digest = hashlib.sha256()
    for name in (
        "states",
        "roads",
        "actions",
        "phases",
        "quality",
        "episode_ids",
        "sequence_indices",
        "valid",
        "state_history",
        "road_history",
    ):
        raw_value = getattr(dataset, name)
        if raw_value is None:
            digest.update(name.encode("utf-8"))
            digest.update(b"none")
            continue
        value = np.ascontiguousarray(raw_value)
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _finite_score(value: float) -> float:
    score = float(value)
    if not np.isfinite(score):
        raise ValueError("validation evaluator returned a non-finite score")
    return score
