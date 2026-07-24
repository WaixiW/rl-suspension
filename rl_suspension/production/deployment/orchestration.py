"""Hardware-in-loop, shadow, and fault-injection orchestration interfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter_ns
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    ActionSchema,
    ObservationV1,
    PolicyAdapter,
)
from rl_suspension.production.deployment.runtime import SafetySupervisor


@dataclass(frozen=True)
class HilTestCase:
    case_id: str
    duration_s: float
    configuration: Mapping[str, Any] = field(default_factory=dict)
    required_checks: tuple[str, ...] = ()


@dataclass(frozen=True)
class HilCaseResult:
    case_id: str
    passed: bool
    checks: Mapping[str, bool]
    measurements: Mapping[str, float] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    detail: str = ""


@runtime_checkable
class HilBackend(Protocol):
    """Private HIL adapter; implementations own hardware setup and teardown."""

    def prepare(self) -> None: ...

    def run_case(self, case: HilTestCase) -> HilCaseResult: ...

    def shutdown(self) -> None: ...


@dataclass(frozen=True)
class HilCampaignReport:
    passed: bool
    results: tuple[HilCaseResult, ...]


class HilOrchestrator:
    def __init__(self, backend: HilBackend) -> None:
        self.backend = backend

    def run(self, cases: Iterable[HilTestCase]) -> HilCampaignReport:
        case_list = tuple(cases)
        if not case_list:
            raise ValueError("HIL campaign must contain at least one case")
        ids = [case.case_id for case in case_list]
        if len(ids) != len(set(ids)):
            raise ValueError("HIL case IDs must be unique")
        results: list[HilCaseResult] = []
        self.backend.prepare()
        try:
            for case in case_list:
                result = self.backend.run_case(case)
                if result.case_id != case.case_id:
                    raise ValueError("HIL backend returned a mismatched case ID")
                missing = set(case.required_checks) - set(result.checks)
                if missing:
                    result = HilCaseResult(
                        case_id=result.case_id,
                        passed=False,
                        checks=result.checks,
                        measurements=result.measurements,
                        artifacts=result.artifacts,
                        detail=f"missing required checks: {sorted(missing)}",
                    )
                results.append(result)
        finally:
            self.backend.shutdown()
        return HilCampaignReport(
            passed=all(result.passed for result in results),
            results=tuple(results),
        )


HILOrchestrator = HilOrchestrator


@dataclass(frozen=True)
class ShadowSample:
    timestamp_ns: int
    maximum_absolute_delta: float
    rms_delta: float
    candidate_latency_ms: float


@dataclass(frozen=True)
class ShadowReport:
    samples: int
    maximum_absolute_delta: float
    rms_delta: float
    p99_candidate_latency_ms: float
    candidate_never_applied: bool
    records: tuple[ShadowSample, ...]


class ShadowOrchestrator:
    """Evaluate a candidate beside production without returning its command."""

    def __init__(
        self,
        production: PolicyAdapter,
        candidate: PolicyAdapter,
        *,
        action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
    ) -> None:
        self.production = production
        self.candidate = candidate
        self.action_schema = action_schema

    def run(self, observations: Iterable[ObservationV1]) -> ShadowReport:
        records: list[ShadowSample] = []
        all_delta: list[np.ndarray] = []
        for observation in observations:
            production = self.action_schema.validate(
                np.asarray(self.production.predict(observation), dtype=np.float64)
            )
            started = perf_counter_ns()
            candidate = self.action_schema.validate(
                np.asarray(self.candidate.predict(observation), dtype=np.float64)
            )
            latency = (perf_counter_ns() - started) / 1e6
            delta = candidate - production
            all_delta.append(delta)
            records.append(
                ShadowSample(
                    timestamp_ns=int(observation.timestamp_ns),
                    maximum_absolute_delta=float(np.max(np.abs(delta))),
                    rms_delta=float(np.sqrt(np.mean(np.square(delta)))),
                    candidate_latency_ms=float(latency),
                )
            )
        if not records:
            raise ValueError("shadow run requires at least one observation")
        flattened = np.concatenate(all_delta)
        return ShadowReport(
            samples=len(records),
            maximum_absolute_delta=float(np.max(np.abs(flattened))),
            rms_delta=float(np.sqrt(np.mean(np.square(flattened)))),
            p99_candidate_latency_ms=float(
                np.quantile([record.candidate_latency_ms for record in records], 0.99)
            ),
            candidate_never_applied=True,
            records=tuple(records),
        )


@runtime_checkable
class FaultInjector(Protocol):
    name: str

    def inject(self, observation: ObservationV1, step: int) -> ObservationV1: ...


@dataclass(frozen=True)
class FaultCampaignResult:
    injector: str
    steps: int
    fallback_steps: int
    fault_reasons: tuple[str, ...]
    decisions: tuple[Mapping[str, Any], ...]

    @property
    def fallback_observed(self) -> bool:
        return self.fallback_steps > 0


class FaultInjectionOrchestrator:
    """Replay observations through each injector and the real supervisor."""

    def run(
        self,
        observations: Sequence[ObservationV1],
        supervisor: SafetySupervisor,
        injectors: Iterable[FaultInjector],
    ) -> tuple[FaultCampaignResult, ...]:
        if not observations:
            raise ValueError("fault campaign requires observations")
        results: list[FaultCampaignResult] = []
        for injector in injectors:
            supervisor.reset()
            decisions: list[Mapping[str, Any]] = []
            reasons: set[str] = set()
            fallback_steps = 0
            for step, observation in enumerate(observations):
                injected = injector.inject(observation, step)
                decision = supervisor.decide(
                    injected,
                    now_ns=int(observation.timestamp_ns),
                )
                fallback_steps += int(decision.source == "fallback")
                reasons.update(decision.reasons)
                decisions.append(
                    {
                        **asdict(decision),
                        "action": decision.action.tolist(),
                        "proposed_action": decision.proposed_action.tolist(),
                    }
                )
            results.append(
                FaultCampaignResult(
                    injector=injector.name,
                    steps=len(observations),
                    fallback_steps=fallback_steps,
                    fault_reasons=tuple(sorted(reasons)),
                    decisions=tuple(decisions),
                )
            )
        return tuple(results)
