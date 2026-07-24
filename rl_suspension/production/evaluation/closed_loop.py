"""Paired closed-loop qualification against passive, MPC, and student control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter_ns
from typing import Callable, Iterable

import numpy as np

from rl_suspension.production.adapters.safe_controller import ConstantSafeController
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
from rl_suspension.production.evaluation.metrics import (
    BootstrapInterval,
    bootstrap_confidence_interval,
)


@dataclass(frozen=True)
class ClosedLoopEpisode:
    controller: str
    scenario_id: str
    seed: int
    steps: int
    return_total: float
    rms_body_acceleration: float
    peak_body_acceleration: float
    rms_pitch_acceleration: float
    rms_roll_acceleration: float
    maximum_suspension_travel: float
    minimum_tire_load: float
    constraint_violation_total: float
    action_bounds_violations: int
    action_slew_violations: int
    actuator_energy: float
    mean_action_rate: float
    p99_latency_ms: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)

    def numeric_metrics(self) -> dict[str, float]:
        payload = self.to_dict()
        return {
            key: float(value)
            for key, value in payload.items()
            if key not in {"controller", "scenario_id", "seed", "steps"}
        }


@dataclass(frozen=True)
class ControllerSummary:
    controller: str
    episodes: int
    metrics: dict[str, BootstrapInterval]

    def to_dict(self) -> dict[str, object]:
        return {
            "controller": self.controller,
            "episodes": self.episodes,
            "metrics": {
                name: interval.to_dict() for name, interval in self.metrics.items()
            },
        }


@dataclass(frozen=True)
class PairedClosedLoopReport:
    """Episode results plus paired candidate-minus-baseline intervals."""

    episodes: dict[str, tuple[ClosedLoopEpisode, ...]]
    summaries: dict[str, ControllerSummary]
    paired_differences: dict[str, dict[str, BootstrapInterval]]

    def to_dict(self) -> dict[str, object]:
        return {
            "episodes": {
                name: [episode.to_dict() for episode in episodes]
                for name, episodes in self.episodes.items()
            },
            "summaries": {
                name: summary.to_dict() for name, summary in self.summaries.items()
            },
            "paired_differences": {
                comparison: {
                    metric: interval.to_dict() for metric, interval in metrics.items()
                }
                for comparison, metrics in self.paired_differences.items()
            },
        }


class PairedClosedLoopEvaluator:
    """Evaluate all controllers on identical scenario/seed pairs."""

    def __init__(
        self,
        simulator_factory: Callable[[], SimulatorAdapter],
        *,
        mpc: MpcAdapter,
        student: PolicyAdapter,
        passive: PolicyAdapter | None = None,
        action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
        observation_schema: ObservationSchema = DEFAULT_OBSERVATION_SCHEMA,
        confidence: float = 0.95,
        bootstrap_resamples: int = 2_000,
        bootstrap_seed: int = 0,
    ) -> None:
        self.simulator_factory = simulator_factory
        self.mpc = mpc
        self.student = student
        self.passive = passive or ConstantSafeController(action_schema)
        self.action_schema = action_schema
        self.observation_schema = observation_schema
        self.confidence = confidence
        self.bootstrap_resamples = bootstrap_resamples
        self.bootstrap_seed = bootstrap_seed

    def evaluate(self, scenarios: Iterable[Scenario]) -> PairedClosedLoopReport:
        scenario_list = tuple(scenarios)
        if not scenario_list:
            raise ValueError("at least one scenario is required")

        controllers = (
            (self.passive.name, self.passive),
            (self.mpc.name, self.mpc),
            (self.student.name, self.student),
        )
        names = [name for name, _ in controllers]
        if len(set(names)) != len(names):
            raise ValueError("controller names must be unique")

        episode_map: dict[str, list[ClosedLoopEpisode]] = {
            name: [] for name in names
        }
        for scenario in scenario_list:
            for name, controller in controllers:
                episode_map[name].append(
                    self._run_episode(scenario, name, controller, controller is self.mpc)
                )

        frozen_episodes = {
            name: tuple(episodes) for name, episodes in episode_map.items()
        }
        summaries = {
            name: self._summarize(name, episodes)
            for name, episodes in frozen_episodes.items()
        }
        student_name = self.student.name
        paired = {}
        for baseline_name in (self.passive.name, self.mpc.name):
            paired[f"{student_name}_minus_{baseline_name}"] = self._paired_metrics(
                frozen_episodes[student_name],
                frozen_episodes[baseline_name],
            )
        return PairedClosedLoopReport(frozen_episodes, summaries, paired)

    def _run_episode(
        self,
        scenario: Scenario,
        controller_name: str,
        controller: PolicyAdapter | MpcAdapter,
        is_mpc: bool,
    ) -> ClosedLoopEpisode:
        simulator = self.simulator_factory()
        observation = simulator.reset(scenario, scenario.seed)
        observation.validate(self.observation_schema, self.action_schema)
        if is_mpc:
            assert isinstance(controller, MpcAdapter)
            controller.reset(scenario, simulator.snapshot())
        else:
            reset_controller = getattr(controller, "reset", None)
            if callable(reset_controller):
                reset_controller()

        previous = np.asarray(self.action_schema.safe_action, dtype=np.float64)
        body: list[float] = []
        pitch: list[float] = []
        roll: list[float] = []
        travel: list[float] = []
        tire_load: list[float] = []
        violations: list[float] = []
        action_energy: list[float] = []
        action_rate: list[float] = []
        latency_ms: list[float] = []
        total_return = 0.0
        bounds_violations = 0
        slew_violations = 0
        dt = self.observation_schema.control_period_s

        while not simulator.done:
            started = perf_counter_ns()
            if is_mpc:
                solve = controller.solve(observation, simulator.snapshot())  # type: ignore[union-attr]
                proposed = solve.action_12d
                if not solve.valid:
                    proposed = np.asarray(self.action_schema.safe_action)
            else:
                proposed = controller.predict(observation)  # type: ignore[union-attr]
            latency_ms.append((perf_counter_ns() - started) / 1e6)

            raw = np.asarray(proposed, dtype=np.float64)
            if raw.shape != (self.action_schema.dimension,) or not np.all(
                np.isfinite(raw)
            ):
                bounds_violations += 1
                action = np.asarray(self.action_schema.safe_action, dtype=np.float64)
            else:
                low = np.asarray(self.action_schema.minimum)
                high = np.asarray(self.action_schema.maximum)
                bounds_violations += int(np.count_nonzero((raw < low) | (raw > high)))
                bounded = np.clip(raw, low, high)
                max_delta = np.asarray(self.action_schema.slew_per_second) * dt
                slew_violations += int(
                    np.count_nonzero(np.abs(bounded - previous) > max_delta + 1e-12)
                )
                action = self.action_schema.project(raw, previous, dt)

            result = simulator.step(action)
            diagnostics = result.diagnostics
            total_return += diagnostics.reward
            body.append(diagnostics.body_acceleration)
            pitch.append(diagnostics.pitch_acceleration)
            roll.append(diagnostics.roll_acceleration)
            travel.append(float(np.max(np.abs(diagnostics.suspension_travel))))
            tire_load.append(float(np.min(diagnostics.tire_loads)))
            violations.append(float(sum(diagnostics.constraint_violations.values())))
            action_energy.append(float(np.sum(np.square(action))) * dt)
            action_rate.append(float(np.mean(np.abs(action - previous))) / dt)
            previous = action
            observation = result.observation
            observation.validate(self.observation_schema, self.action_schema)
            if diagnostics.terminated or diagnostics.truncated:
                break

        return ClosedLoopEpisode(
            controller=controller_name,
            scenario_id=scenario.scenario_id,
            seed=scenario.seed,
            steps=len(body),
            return_total=float(total_return),
            rms_body_acceleration=_rms(body),
            peak_body_acceleration=float(np.max(np.abs(body))),
            rms_pitch_acceleration=_rms(pitch),
            rms_roll_acceleration=_rms(roll),
            maximum_suspension_travel=float(np.max(travel)),
            minimum_tire_load=float(np.min(tire_load)),
            constraint_violation_total=float(np.sum(violations)),
            action_bounds_violations=bounds_violations,
            action_slew_violations=slew_violations,
            actuator_energy=float(np.sum(action_energy)),
            mean_action_rate=float(np.mean(action_rate)),
            p99_latency_ms=float(np.quantile(latency_ms, 0.99)),
        )

    def _summarize(
        self, name: str, episodes: tuple[ClosedLoopEpisode, ...]
    ) -> ControllerSummary:
        metric_names = tuple(episodes[0].numeric_metrics())
        return ControllerSummary(
            controller=name,
            episodes=len(episodes),
            metrics={
                metric: self._interval(
                    [episode.numeric_metrics()[metric] for episode in episodes],
                    seed_offset=index,
                )
                for index, metric in enumerate(metric_names)
            },
        )

    def _paired_metrics(
        self,
        candidate: tuple[ClosedLoopEpisode, ...],
        baseline: tuple[ClosedLoopEpisode, ...],
    ) -> dict[str, BootstrapInterval]:
        if len(candidate) != len(baseline):
            raise ValueError("paired episode counts do not match")
        metrics = tuple(candidate[0].numeric_metrics())
        return {
            metric: self._interval(
                [
                    first.numeric_metrics()[metric] - second.numeric_metrics()[metric]
                    for first, second in zip(candidate, baseline)
                ],
                seed_offset=index,
            )
            for index, metric in enumerate(metrics)
        }

    def _interval(
        self, values: list[float], *, seed_offset: int
    ) -> BootstrapInterval:
        return bootstrap_confidence_interval(
            values,
            confidence=self.confidence,
            resamples=self.bootstrap_resamples,
            seed=self.bootstrap_seed + seed_offset,
        )


def evaluate_paired_closed_loop(
    scenarios: Iterable[Scenario],
    simulator_factory: Callable[[], SimulatorAdapter],
    *,
    mpc: MpcAdapter,
    student: PolicyAdapter,
    passive: PolicyAdapter | None = None,
    **kwargs,
) -> PairedClosedLoopReport:
    """Functional wrapper around :class:`PairedClosedLoopEvaluator`."""

    return PairedClosedLoopEvaluator(
        simulator_factory,
        mpc=mpc,
        student=student,
        passive=passive,
        **kwargs,
    ).evaluate(scenarios)


def _rms(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))
