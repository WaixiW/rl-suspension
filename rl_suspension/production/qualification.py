"""Paired passive-versus-MPC qualification with explicit production gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

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


SimulatorFactory = Callable[[], SimulatorAdapter]


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int


@dataclass(frozen=True)
class RolloutMetrics:
    steps: int
    rms_body_acceleration: float
    rms_pitch_acceleration: float
    rms_roll_acceleration: float
    total_reward: float
    maximum_suspension_travel: float
    minimum_tire_load: float
    safety_violation_steps: int


@dataclass(frozen=True)
class PairedEpisodeResult:
    scenario_id: str
    split: str
    passive: RolloutMetrics
    mpc: RolloutMetrics
    comfort_improvement: float
    return_improvement: float


@dataclass(frozen=True)
class QualificationConfig:
    confidence: float = 0.95
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 0
    maximum_steps: int = 10_000
    minimum_comfort_improvement_lower_bound: float = 0.0
    minimum_return_improvement_lower_bound: float = 0.0
    minimum_solver_valid_rate: float = 0.99
    maximum_fallback_rate: float = 0.0
    maximum_timeout_rate: float = 0.0
    maximum_solve_time_ms: float = 20.0
    maximum_raw_slew_violation_rate: float = 0.0
    maximum_safety_violation_rate: float = 1.0
    maximum_safety_regression: float = 0.0
    safety_tolerance: float = 1e-9
    allowed_solver_statuses: tuple[str, ...] = ("optimal", "optimal_inaccurate")

    def validate(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be between zero and one")
        if self.bootstrap_samples <= 0 or self.maximum_steps <= 0:
            raise ValueError("bootstrap_samples and maximum_steps must be positive")
        for name in (
            "minimum_solver_valid_rate",
            "maximum_fallback_rate",
            "maximum_timeout_rate",
            "maximum_raw_slew_violation_rate",
            "maximum_safety_violation_rate",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.maximum_solve_time_ms <= 0.0 or self.safety_tolerance < 0.0:
            raise ValueError("latency must be positive and safety tolerance nonnegative")
        if self.maximum_safety_regression < 0.0:
            raise ValueError("maximum_safety_regression must be nonnegative")


@dataclass(frozen=True)
class QualificationReport:
    passed: bool
    gates: dict[str, bool]
    comfort_improvement: BootstrapInterval
    return_improvement: BootstrapInterval
    paired_episodes: tuple[PairedEpisodeResult, ...]
    solver_valid_rate: float
    solver_fallback_rate: float
    solver_timeout_rate: float
    solver_status_rate: float
    solve_time_p99_ms: float
    raw_action_bound_violations: int
    raw_action_slew_violation_rate: float
    applied_action_violations: int
    safety_violation_rate: float
    passive_safety_violation_rate: float
    solver_calls: int

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        temporary.replace(target)


@dataclass
class _SolverCounters:
    calls: int = 0
    valid: int = 0
    fallback: int = 0
    timeout: int = 0
    allowed_status: int = 0
    raw_bound_violations: int = 0
    raw_slew_violations: int = 0
    applied_violations: int = 0

    def __post_init__(self) -> None:
        self.solve_times_ms: list[float] = []
        self.nonfinite_solve_times = 0


def bootstrap_confidence_interval(
    values: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 0,
) -> BootstrapInterval:
    """Deterministic percentile bootstrap of the paired sample mean."""

    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("bootstrap values must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(observations)):
        raise ValueError("bootstrap values must be finite")
    if not 0.0 < confidence < 1.0 or samples <= 0:
        raise ValueError("invalid confidence or bootstrap sample count")

    estimate = float(np.mean(observations))
    if observations.size == 1:
        return BootstrapInterval(
            estimate=estimate,
            lower=estimate,
            upper=estimate,
            confidence=confidence,
            samples=samples,
        )
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    # Batching avoids allocating samples x episodes for large qualification suites.
    batch_size = min(samples, 2048)
    offset = 0
    while offset < samples:
        size = min(batch_size, samples - offset)
        indices = rng.integers(0, observations.size, size=(size, observations.size))
        means[offset : offset + size] = observations[indices].mean(axis=1)
        offset += size
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return BootstrapInterval(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        samples=samples,
    )


def qualify_mpc(
    mpc: MpcAdapter,
    simulator_factory: SimulatorFactory,
    scenarios: Iterable[Scenario],
    *,
    config: QualificationConfig | None = None,
    action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
    observation_schema: ObservationSchema = DEFAULT_OBSERVATION_SCHEMA,
) -> QualificationReport:
    """Run paired rollouts and evaluate comfort, solver, action, and safety gates."""

    settings = config or QualificationConfig()
    settings.validate()
    scenario_list = list(scenarios)
    if not scenario_list:
        raise ValueError("qualification requires at least one scenario")

    paired: list[PairedEpisodeResult] = []
    counters = _SolverCounters()
    total_mpc_steps = 0
    total_safety_steps = 0
    total_passive_steps = 0
    total_passive_safety_steps = 0
    for scenario in scenario_list:
        passive_metrics = _rollout_passive(
            simulator_factory(),
            scenario,
            settings,
            action_schema,
            observation_schema,
        )
        mpc_metrics = _rollout_mpc(
            simulator_factory(),
            mpc,
            scenario,
            settings,
            action_schema,
            observation_schema,
            counters,
        )
        denominator = max(passive_metrics.rms_body_acceleration, 1e-12)
        improvement = (
            passive_metrics.rms_body_acceleration - mpc_metrics.rms_body_acceleration
        ) / denominator
        return_improvement = (
            mpc_metrics.total_reward - passive_metrics.total_reward
        ) / max(abs(passive_metrics.total_reward), 1.0)
        paired.append(
            PairedEpisodeResult(
                scenario_id=scenario.scenario_id,
                split=scenario.split,
                passive=passive_metrics,
                mpc=mpc_metrics,
                comfort_improvement=float(improvement),
                return_improvement=float(return_improvement),
            )
        )
        total_mpc_steps += mpc_metrics.steps
        total_safety_steps += mpc_metrics.safety_violation_steps
        total_passive_steps += passive_metrics.steps
        total_passive_safety_steps += passive_metrics.safety_violation_steps

    interval = bootstrap_confidence_interval(
        [item.comfort_improvement for item in paired],
        confidence=settings.confidence,
        samples=settings.bootstrap_samples,
        seed=settings.bootstrap_seed,
    )
    return_interval = bootstrap_confidence_interval(
        [item.return_improvement for item in paired],
        confidence=settings.confidence,
        samples=settings.bootstrap_samples,
        seed=settings.bootstrap_seed + 1,
    )
    calls = max(counters.calls, 1)
    solver_valid_rate = counters.valid / calls
    fallback_rate = counters.fallback / calls
    timeout_rate = counters.timeout / calls
    status_rate = counters.allowed_status / calls
    raw_slew_rate = counters.raw_slew_violations / calls
    safety_rate = total_safety_steps / max(total_mpc_steps, 1)
    passive_safety_rate = total_passive_safety_steps / max(total_passive_steps, 1)
    solve_time_p99 = (
        float(np.quantile(counters.solve_times_ms, 0.99))
        if counters.solve_times_ms
        else 0.0
    )
    gates = {
        "paired_comfort_improvement": (
            interval.lower >= settings.minimum_comfort_improvement_lower_bound
        ),
        "paired_return_improvement": (
            return_interval.lower >= settings.minimum_return_improvement_lower_bound
        ),
        "solver_validity": solver_valid_rate >= settings.minimum_solver_valid_rate,
        "solver_status": status_rate >= settings.minimum_solver_valid_rate,
        "solver_fallback": fallback_rate <= settings.maximum_fallback_rate,
        "solver_timeout": timeout_rate <= settings.maximum_timeout_rate,
        "solver_latency": (
            counters.nonfinite_solve_times == 0
            and solve_time_p99 <= settings.maximum_solve_time_ms
        ),
        "action_bounds": counters.raw_bound_violations == 0,
        "action_slew": (
            raw_slew_rate <= settings.maximum_raw_slew_violation_rate
            and counters.applied_violations == 0
        ),
        "safety_absolute": safety_rate <= settings.maximum_safety_violation_rate,
        "safety_vs_passive": (
            safety_rate
            <= passive_safety_rate + settings.maximum_safety_regression
        ),
    }
    return QualificationReport(
        passed=all(gates.values()),
        gates=gates,
        comfort_improvement=interval,
        return_improvement=return_interval,
        paired_episodes=tuple(paired),
        solver_valid_rate=solver_valid_rate,
        solver_fallback_rate=fallback_rate,
        solver_timeout_rate=timeout_rate,
        solver_status_rate=status_rate,
        solve_time_p99_ms=solve_time_p99,
        raw_action_bound_violations=counters.raw_bound_violations,
        raw_action_slew_violation_rate=raw_slew_rate,
        applied_action_violations=counters.applied_violations,
        safety_violation_rate=safety_rate,
        passive_safety_violation_rate=passive_safety_rate,
        solver_calls=counters.calls,
    )


qualify_controller = qualify_mpc


def _rollout_passive(
    simulator: SimulatorAdapter,
    scenario: Scenario,
    config: QualificationConfig,
    action_schema: ActionSchema,
    observation_schema: ObservationSchema,
) -> RolloutMetrics:
    observation = simulator.reset(scenario, scenario.seed)
    observation.validate(observation_schema, action_schema)
    action = np.asarray(action_schema.safe_action, dtype=np.float64)
    values: list[object] = []
    for _ in range(config.maximum_steps):
        if simulator.done:
            break
        step = simulator.step(action)
        step.observation.validate(observation_schema, action_schema)
        values.append(step.diagnostics)
        if step.diagnostics.terminated or step.diagnostics.truncated or simulator.done:
            break
    return _summarize_rollout(values, config.safety_tolerance)


def _rollout_mpc(
    simulator: SimulatorAdapter,
    mpc: MpcAdapter,
    scenario: Scenario,
    config: QualificationConfig,
    action_schema: ActionSchema,
    observation_schema: ObservationSchema,
    counters: _SolverCounters,
) -> RolloutMetrics:
    observation = simulator.reset(scenario, scenario.seed)
    observation.validate(observation_schema, action_schema)
    initial_snapshot = simulator.snapshot()
    mpc.reset(scenario, initial_snapshot)
    previous_action = np.asarray(action_schema.safe_action, dtype=np.float64)
    values: list[object] = []
    dt = observation_schema.control_period_s
    maximum_delta = np.asarray(action_schema.slew_per_second, dtype=np.float64) * dt
    low = np.asarray(action_schema.minimum, dtype=np.float64)
    high = np.asarray(action_schema.maximum, dtype=np.float64)

    for _ in range(config.maximum_steps):
        if simulator.done:
            break
        result = mpc.solve(observation, simulator.snapshot())
        counters.calls += 1
        diagnostics = result.diagnostics
        counters.valid += int(bool(result.valid))
        counters.fallback += int(bool(diagnostics.fallback))
        counters.timeout += int(bool(diagnostics.timeout))
        counters.allowed_status += int(diagnostics.status in config.allowed_solver_statuses)
        if np.isfinite(diagnostics.solve_time_ms):
            counters.solve_times_ms.append(float(diagnostics.solve_time_ms))
        else:
            counters.nonfinite_solve_times += 1

        raw = np.asarray(result.action_12d, dtype=np.float64)
        raw_shape_finite = raw.shape == (action_schema.dimension,) and np.all(
            np.isfinite(raw)
        )
        if not raw_shape_finite:
            counters.raw_bound_violations += 1
            raw = np.asarray(action_schema.safe_action, dtype=np.float64)
        elif np.any(raw < low - 1e-9) or np.any(raw > high + 1e-9):
            counters.raw_bound_violations += 1
        if np.any(np.abs(raw - previous_action) > maximum_delta + 1e-9):
            counters.raw_slew_violations += 1

        applied = action_schema.project(raw, previous_action, dt)
        if (
            np.any(applied < low - 1e-9)
            or np.any(applied > high + 1e-9)
            or np.any(np.abs(applied - previous_action) > maximum_delta + 1e-9)
        ):
            counters.applied_violations += 1
        step = simulator.step(applied)
        step.observation.validate(observation_schema, action_schema)
        values.append(step.diagnostics)
        previous_action = applied
        observation = step.observation
        if step.diagnostics.terminated or step.diagnostics.truncated or simulator.done:
            break
    return _summarize_rollout(values, config.safety_tolerance)


def _summarize_rollout(values: Sequence[object], safety_tolerance: float) -> RolloutMetrics:
    if not values:
        raise ValueError("simulator produced an empty rollout")
    body = np.asarray([item.body_acceleration for item in values], dtype=np.float64)
    pitch = np.asarray([item.pitch_acceleration for item in values], dtype=np.float64)
    roll = np.asarray([item.roll_acceleration for item in values], dtype=np.float64)
    rewards = np.asarray([item.reward for item in values], dtype=np.float64)
    if not all(np.all(np.isfinite(item)) for item in (body, pitch, roll, rewards)):
        raise ValueError("rollout diagnostics contain NaN or Inf")
    safety_steps = sum(
        any(
            not np.isfinite(value) or value > safety_tolerance
            for value in item.constraint_violations.values()
        )
        for item in values
    )
    return RolloutMetrics(
        steps=len(values),
        rms_body_acceleration=float(np.sqrt(np.mean(np.square(body)))),
        rms_pitch_acceleration=float(np.sqrt(np.mean(np.square(pitch)))),
        rms_roll_acceleration=float(np.sqrt(np.mean(np.square(roll)))),
        total_reward=float(np.sum(rewards)),
        maximum_suspension_travel=float(
            max(np.max(np.abs(item.suspension_travel)) for item in values)
        ),
        minimum_tire_load=float(min(np.min(item.tire_loads) for item in values)),
        safety_violation_steps=int(safety_steps),
    )
