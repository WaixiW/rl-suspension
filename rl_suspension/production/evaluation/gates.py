"""Auditable promotion gates for a production student controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

import numpy as np

from rl_suspension.production.evaluation.closed_loop import PairedClosedLoopReport


@dataclass(frozen=True)
class PromotionGateConfig:
    comfort_metric: str = "rms_body_acceleration"
    minimum_mpc_improvement_retention: float = 0.80
    maximum_safety_regression_fraction: float = 0.0
    maximum_constraint_violation_total: float = 0.0
    maximum_suspension_travel: float = 0.12
    minimum_tire_load: float = 0.0
    maximum_action_bounds_violations: int = 0
    maximum_action_slew_violations: int = 0
    maximum_p99_latency_ms: float = 10.0
    require_confident_passive_improvement: bool = False
    additional_maximums: Mapping[str, float] = field(default_factory=dict)
    additional_minimums: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_mpc_improvement_retention <= 1.0:
            raise ValueError("minimum MPC improvement retention must be within [0, 1]")
        if self.maximum_safety_regression_fraction < 0.0:
            raise ValueError("maximum safety regression fraction must be nonnegative")
        if self.maximum_p99_latency_ms <= 0.0:
            raise ValueError("maximum p99 latency must be positive")


PromotionCriteria = PromotionGateConfig


@dataclass(frozen=True)
class PromotionEvidence:
    passive_metrics: Mapping[str, float]
    mpc_metrics: Mapping[str, float]
    student_metrics: Mapping[str, float]
    student_p99_latency_ms: float
    action_bounds_violations: int
    action_slew_violations: int
    student_minus_passive_ci_upper: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_closed_loop_report(
        cls,
        report: PairedClosedLoopReport,
        *,
        passive_name: str,
        mpc_name: str,
        student_name: str,
        runtime_p99_latency_ms: float | None = None,
    ) -> "PromotionEvidence":
        def estimates(controller: str) -> dict[str, float]:
            return {
                name: interval.estimate
                for name, interval in report.summaries[controller].metrics.items()
            }

        student = estimates(student_name)
        comparison = report.paired_differences[
            f"{student_name}_minus_{passive_name}"
        ]
        return cls(
            passive_metrics=estimates(passive_name),
            mpc_metrics=estimates(mpc_name),
            student_metrics=student,
            student_p99_latency_ms=float(
                runtime_p99_latency_ms
                if runtime_p99_latency_ms is not None
                else student["p99_latency_ms"]
            ),
            action_bounds_violations=int(
                round(student["action_bounds_violations"])
            ),
            action_slew_violations=int(round(student["action_slew_violations"])),
            student_minus_passive_ci_upper={
                name: interval.upper for name, interval in comparison.items()
            },
        )


@dataclass(frozen=True)
class GateCheck:
    passed: bool
    observed: float
    threshold: float
    relation: str
    detail: str = ""

    def to_dict(self) -> dict[str, bool | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionDecision:
    passed: bool
    checks: Mapping[str, GateCheck]
    mpc_improvement_retention: float

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "mpc_improvement_retention": self.mpc_improvement_retention,
            "checks": {
                name: check.to_dict() for name, check in self.checks.items()
            },
        }


def evaluate_promotion_gates(
    evidence: PromotionEvidence,
    config: PromotionGateConfig = PromotionGateConfig(),
) -> PromotionDecision:
    """Evaluate all gates; no failed safety gate can be averaged away."""

    metric = config.comfort_metric
    passive = _metric(evidence.passive_metrics, metric, "passive")
    mpc = _metric(evidence.mpc_metrics, metric, "MPC")
    student = _metric(evidence.student_metrics, metric, "student")
    mpc_improvement = passive - mpc
    student_improvement = passive - student
    retention = (
        student_improvement / mpc_improvement
        if mpc_improvement > 0.0
        else float("-inf")
    )

    checks: dict[str, GateCheck] = {}
    checks["mpc_improves_passive"] = GateCheck(
        passed=mpc_improvement > 0.0,
        observed=mpc_improvement,
        threshold=0.0,
        relation=">",
        detail=f"lower {metric} is better",
    )
    checks["mpc_improvement_retention"] = _minimum_check(
        retention,
        config.minimum_mpc_improvement_retention,
        "student retains the required fraction of MPC improvement",
    )

    constraint_total = _metric(
        evidence.student_metrics,
        "constraint_violation_total",
        "student",
    )
    mpc_constraint = _metric(
        evidence.mpc_metrics,
        "constraint_violation_total",
        "MPC",
    )
    allowed_regression = min(
        config.maximum_constraint_violation_total,
        mpc_constraint * (1.0 + config.maximum_safety_regression_fraction),
    )
    checks["constraint_violations"] = _maximum_check(
        constraint_total,
        allowed_regression,
        "student safety violations must not exceed the absolute/regression limit",
    )
    checks["suspension_travel"] = _maximum_check(
        _metric(evidence.student_metrics, "maximum_suspension_travel", "student"),
        config.maximum_suspension_travel,
    )
    checks["tire_load"] = _minimum_check(
        _metric(evidence.student_metrics, "minimum_tire_load", "student"),
        config.minimum_tire_load,
    )
    checks["action_bounds"] = _maximum_check(
        float(evidence.action_bounds_violations),
        float(config.maximum_action_bounds_violations),
    )
    checks["action_slew"] = _maximum_check(
        float(evidence.action_slew_violations),
        float(config.maximum_action_slew_violations),
    )
    checks["p99_latency"] = _maximum_check(
        evidence.student_p99_latency_ms,
        config.maximum_p99_latency_ms,
    )

    if config.require_confident_passive_improvement:
        upper = evidence.student_minus_passive_ci_upper.get(metric, float("inf"))
        checks["confident_passive_improvement"] = GateCheck(
            passed=bool(np.isfinite(upper) and upper < 0.0),
            observed=float(upper),
            threshold=0.0,
            relation="<",
            detail="upper paired-CI bound for student minus passive",
        )

    for name, threshold in config.additional_maximums.items():
        checks[f"maximum:{name}"] = _maximum_check(
            _metric(evidence.student_metrics, name, "student"),
            float(threshold),
        )
    for name, threshold in config.additional_minimums.items():
        checks[f"minimum:{name}"] = _minimum_check(
            _metric(evidence.student_metrics, name, "student"),
            float(threshold),
        )

    return PromotionDecision(
        passed=all(check.passed for check in checks.values()),
        checks=checks,
        mpc_improvement_retention=float(retention),
    )


def _metric(metrics: Mapping[str, float], name: str, controller: str) -> float:
    if name not in metrics:
        raise KeyError(f"{controller} metrics do not include {name!r}")
    value = float(metrics[name])
    if not np.isfinite(value):
        raise ValueError(f"{controller} metric {name!r} is not finite")
    return value


def _maximum_check(
    observed: float, threshold: float, detail: str = ""
) -> GateCheck:
    return GateCheck(
        passed=bool(np.isfinite(observed) and observed <= threshold),
        observed=float(observed),
        threshold=float(threshold),
        relation="<=",
        detail=detail,
    )


def _minimum_check(
    observed: float, threshold: float, detail: str = ""
) -> GateCheck:
    return GateCheck(
        passed=bool(np.isfinite(observed) and observed >= threshold),
        observed=float(observed),
        threshold=float(threshold),
        relation=">=",
        detail=detail,
    )
