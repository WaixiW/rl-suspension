"""ECU-facing safety supervisor around an untrusted learned policy."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from time import monotonic_ns, perf_counter_ns
from typing import Callable, Sequence

import numpy as np

from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    DEFAULT_OBSERVATION_SCHEMA,
    ActionSchema,
    ObservationSchema,
    ObservationV1,
    PolicyAdapter,
)


@dataclass(frozen=True)
class EngineeringEnvelope:
    """Engineering-domain limits, separate from statistical normalization."""

    vehicle_state_minimum: float | Sequence[float] = -100.0
    vehicle_state_maximum: float | Sequence[float] = 100.0
    sensor_feature_minimum: float | Sequence[float] = -1_000.0
    sensor_feature_maximum: float | Sequence[float] = 1_000.0
    actuator_state_minimum: float | Sequence[float] = -10_000.0
    actuator_state_maximum: float | Sequence[float] = 10_000.0
    minimum_speed_mps: float = 0.0
    maximum_speed_mps: float = 80.0
    minimum_road_height_m: float = -0.30
    maximum_road_height_m: float = 0.30
    minimum_road_valid_fraction: float = 0.50
    minimum_sensor_validity: float = 0.50

    def violations(self, observation: ObservationV1) -> tuple[str, ...]:
        reasons: list[str] = []
        _append_range_violation(
            reasons,
            "vehicle_state_ood",
            observation.vehicle_state,
            self.vehicle_state_minimum,
            self.vehicle_state_maximum,
        )
        _append_range_violation(
            reasons,
            "sensor_features_ood",
            observation.sensor_features,
            self.sensor_feature_minimum,
            self.sensor_feature_maximum,
        )
        _append_range_violation(
            reasons,
            "actuator_state_ood",
            observation.actuator_state,
            self.actuator_state_minimum,
            self.actuator_state_maximum,
        )
        if not self.minimum_speed_mps <= observation.speed_mps <= self.maximum_speed_mps:
            reasons.append("speed_ood")
        road = np.concatenate(
            [observation.road_left_m, observation.road_right_m]
        )
        if np.any(road < self.minimum_road_height_m) or np.any(
            road > self.maximum_road_height_m
        ):
            reasons.append("road_height_ood")
        if float(np.mean(observation.road_validity)) < self.minimum_road_valid_fraction:
            reasons.append("road_invalid")
        if np.any(
            np.asarray(observation.sensor_validity)
            < self.minimum_sensor_validity
        ):
            reasons.append("sensor_invalid")
        return tuple(reasons)


@dataclass(frozen=True)
class SupervisorConfig:
    maximum_observation_age_ms: float = 25.0
    maximum_future_skew_ms: float = 1.0
    deadline_ms: float = 10.0
    faults_to_latch_fallback: int = 1
    minimum_fallback_cycles: int = 5
    healthy_cycles_to_recover: int = 3
    ring_buffer_capacity: int = 2_048
    enforce_freshness: bool = True

    def __post_init__(self) -> None:
        if (
            self.maximum_observation_age_ms <= 0.0
            or self.maximum_future_skew_ms < 0.0
            or self.deadline_ms <= 0.0
        ):
            raise ValueError("supervisor time limits are invalid")
        if (
            self.faults_to_latch_fallback < 1
            or self.minimum_fallback_cycles < 0
            or self.healthy_cycles_to_recover < 1
            or self.ring_buffer_capacity < 1
        ):
            raise ValueError("supervisor cycle counts are invalid")


@dataclass(frozen=True)
class SupervisorDecision:
    action: np.ndarray
    proposed_action: np.ndarray
    source: str
    fallback_active: bool
    projected: bool
    reasons: tuple[str, ...]
    primary_latency_ms: float
    total_latency_ms: float


@dataclass(frozen=True)
class SupervisorLogRecord:
    sequence: int
    decision_timestamp_ns: int
    observation_timestamp_ns: int
    source: str
    fallback_active: bool
    projected: bool
    reasons: tuple[str, ...]
    primary_latency_ms: float
    total_latency_ms: float
    proposed_action: tuple[float, ...]
    applied_action: tuple[float, ...]


class SafetySupervisor:
    """Validate, supervise, project, and log every learned-policy command."""

    name = "safety_supervisor"

    def __init__(
        self,
        primary: PolicyAdapter,
        fallback: PolicyAdapter,
        *,
        config: SupervisorConfig = SupervisorConfig(),
        envelope: EngineeringEnvelope = EngineeringEnvelope(),
        action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
        observation_schema: ObservationSchema = DEFAULT_OBSERVATION_SCHEMA,
        clock_ns: Callable[[], int] = monotonic_ns,
        latency_clock_ns: Callable[[], int] = perf_counter_ns,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.config = config
        self.envelope = envelope
        self.action_schema = action_schema
        self.observation_schema = observation_schema
        self.clock_ns = clock_ns
        self.latency_clock_ns = latency_clock_ns
        self._records: deque[SupervisorLogRecord] = deque(
            maxlen=config.ring_buffer_capacity
        )
        self.reset()

    @property
    def fallback_active(self) -> bool:
        return self._fallback_active

    def reset(self, previous_action: np.ndarray | None = None) -> None:
        self._previous_action = self.action_schema.validate(
            np.asarray(
                self.action_schema.safe_action
                if previous_action is None
                else previous_action,
                dtype=np.float64,
            )
        ).copy()
        self._fallback_active = False
        self._consecutive_faults = 0
        self._healthy_recovery_cycles = 0
        self._fallback_cycles = 0
        self._sequence = 0
        self._records.clear()

    def predict(self, observation: ObservationV1) -> np.ndarray:
        """PolicyAdapter-compatible action-only interface."""

        return self.decide(observation).action

    def decide(
        self,
        observation: ObservationV1,
        *,
        now_ns: int | None = None,
    ) -> SupervisorDecision:
        total_started = self.latency_clock_ns()
        decision_time = self.clock_ns() if now_ns is None else int(now_ns)
        reasons = list(self._observation_reasons(observation, decision_time))

        primary_latency_ms = 0.0
        proposed: np.ndarray | None = None
        if not reasons:
            started = self.latency_clock_ns()
            try:
                proposed = np.asarray(
                    self.primary.predict(observation), dtype=np.float64
                )
            except Exception:
                reasons.append("primary_exception")
            primary_latency_ms = (self.latency_clock_ns() - started) / 1e6
            if primary_latency_ms > self.config.deadline_ms:
                reasons.append("deadline_miss")
            if proposed is not None and (
                proposed.shape != (self.action_schema.dimension,)
                or not np.all(np.isfinite(proposed))
            ):
                reasons.append("primary_action_invalid")

        healthy = not reasons
        self._update_fallback_state(healthy)
        use_fallback = self._fallback_active or not healthy
        source = "fallback" if use_fallback else "primary"
        if use_fallback:
            proposed = self._fallback_action(observation, reasons)
        assert proposed is not None

        action = self.action_schema.project(
            proposed,
            self._previous_action,
            self.observation_schema.control_period_s,
        )
        projected = not np.allclose(action, proposed, rtol=0.0, atol=1e-12)
        self._previous_action = action.copy()
        total_latency_ms = (self.latency_clock_ns() - total_started) / 1e6
        decision = SupervisorDecision(
            action=action,
            proposed_action=proposed.copy(),
            source=source,
            fallback_active=self._fallback_active,
            projected=projected,
            reasons=tuple(dict.fromkeys(reasons)),
            primary_latency_ms=float(primary_latency_ms),
            total_latency_ms=float(total_latency_ms),
        )
        self._record(observation, decision_time, decision)
        return decision

    def log_snapshot(self) -> tuple[SupervisorLogRecord, ...]:
        return tuple(self._records)

    def write_log(self, path: str | Path) -> Path:
        """Persist the bounded ring snapshot as newline-delimited JSON."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for record in self._records:
                stream.write(json.dumps(asdict(record), allow_nan=False) + "\n")
        temporary.replace(target)
        return target

    def _observation_reasons(
        self,
        observation: ObservationV1,
        decision_time_ns: int,
    ) -> tuple[str, ...]:
        try:
            observation.validate(self.observation_schema, self.action_schema)
        except (ValueError, TypeError):
            return ("observation_invalid",)
        reasons = list(self.envelope.violations(observation))
        if self.config.enforce_freshness:
            age_ns = decision_time_ns - int(observation.timestamp_ns)
            if age_ns > self.config.maximum_observation_age_ms * 1e6:
                reasons.append("observation_stale")
            if age_ns < -self.config.maximum_future_skew_ms * 1e6:
                reasons.append("observation_from_future")
        return tuple(reasons)

    def _fallback_action(
        self,
        observation: ObservationV1,
        reasons: list[str],
    ) -> np.ndarray:
        try:
            fallback = np.asarray(
                self.fallback.predict(observation), dtype=np.float64
            )
            if fallback.shape != (self.action_schema.dimension,) or not np.all(
                np.isfinite(fallback)
            ):
                raise ValueError("invalid fallback action")
            return fallback
        except Exception:
            reasons.append("fallback_invalid_safe_action_used")
            return np.asarray(self.action_schema.safe_action, dtype=np.float64)

    def _update_fallback_state(self, healthy: bool) -> None:
        if self._fallback_active:
            self._fallback_cycles += 1
            if healthy:
                self._healthy_recovery_cycles += 1
            else:
                self._healthy_recovery_cycles = 0
            if (
                self._fallback_cycles >= self.config.minimum_fallback_cycles
                and self._healthy_recovery_cycles
                >= self.config.healthy_cycles_to_recover
            ):
                self._fallback_active = False
                self._consecutive_faults = 0
                self._healthy_recovery_cycles = 0
                self._fallback_cycles = 0
            return

        if healthy:
            self._consecutive_faults = 0
            return
        self._consecutive_faults += 1
        if self._consecutive_faults >= self.config.faults_to_latch_fallback:
            self._fallback_active = True
            self._fallback_cycles = 0
            self._healthy_recovery_cycles = 0

    def _record(
        self,
        observation: ObservationV1,
        decision_time_ns: int,
        decision: SupervisorDecision,
    ) -> None:
        self._records.append(
            SupervisorLogRecord(
                sequence=self._sequence,
                decision_timestamp_ns=decision_time_ns,
                observation_timestamp_ns=int(observation.timestamp_ns),
                source=decision.source,
                fallback_active=decision.fallback_active,
                projected=decision.projected,
                reasons=decision.reasons,
                primary_latency_ms=decision.primary_latency_ms,
                total_latency_ms=decision.total_latency_ms,
                proposed_action=tuple(float(value) for value in decision.proposed_action),
                applied_action=tuple(float(value) for value in decision.action),
            )
        )
        self._sequence += 1


EcuSafetySupervisor = SafetySupervisor


def _append_range_violation(
    reasons: list[str],
    name: str,
    values: np.ndarray,
    minimum: float | Sequence[float],
    maximum: float | Sequence[float],
) -> None:
    array = np.asarray(values, dtype=np.float64)
    low = np.asarray(minimum, dtype=np.float64)
    high = np.asarray(maximum, dtype=np.float64)
    try:
        invalid = np.any(array < low) or np.any(array > high)
    except ValueError as error:
        raise ValueError(f"engineering envelope for {name} is not broadcastable") from error
    if invalid:
        reasons.append(name)
