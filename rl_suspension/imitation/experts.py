"""Replaceable expert interface and temporary-teacher qualification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from rl_suspension.baselines import PassivePolicy, PreviewRulePolicy, SkyhookGroundhookPolicy


FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class ExpertResult:
    """One expert query result."""

    action: FloatArray
    valid: bool = True
    quality: float = 1.0
    diagnostics: dict[str, float | str] = field(default_factory=dict)


@runtime_checkable
class Expert(Protocol):
    """Interface shared by rule teachers and the future MPC teacher."""

    name: str

    def predict(self, observation: NDArray[np.floating]) -> ExpertResult:
        """Return a normalized four-corner-force label."""


@dataclass
class PolicyExpert:
    """Adapt a baseline policy to the expert protocol."""

    name: str
    policy: object
    quality: float = 1.0

    def predict(self, observation: NDArray[np.floating]) -> ExpertResult:
        action, _ = self.policy.predict(observation, deterministic=True)
        array = np.asarray(action, dtype=np.float32)
        valid = bool(array.shape == (4,) and np.all(np.isfinite(array)))
        if not valid:
            array = np.zeros(4, dtype=np.float32)
        clipped = np.clip(array, -1.0, 1.0).astype(np.float32)
        return ExpertResult(
            action=clipped,
            valid=valid,
            quality=float(self.quality),
            diagnostics={"teacher": self.name},
        )


@dataclass(frozen=True)
class QualificationScore:
    name: str
    mean_return: float
    mean_constraint_violation: float
    mean_rms_acceleration: float
    mean_solve_time_ms: float = 0.0
    p95_solve_time_ms: float = 0.0
    fallback_rate: float = 0.0


@dataclass(frozen=True)
class QualificationResult:
    expert: Expert
    selected: QualificationScore
    passive: QualificationScore
    candidates: tuple[QualificationScore, ...]
    qualified: bool


def temporary_experts() -> list[PolicyExpert]:
    """Return a small gain grid for selecting a useful mock teacher."""

    experts: list[PolicyExpert] = [
        PolicyExpert("skyhook_default", SkyhookGroundhookPolicy()),
    ]
    for preview_gain in (0.25, 0.45, 0.75, 1.0, -0.25, -0.5):
        for damping_gain in (0.02, 0.04, 0.08, -0.04):
            name = f"preview_p{preview_gain:g}_d{damping_gain:g}"
            experts.append(
                PolicyExpert(
                    name,
                    PreviewRulePolicy(
                        preview_gain=preview_gain,
                        damping_gain=damping_gain,
                    ),
                )
            )
    return experts


def qualify_temporary_expert(
    env_factory: Callable[[], object],
    seeds: list[int] | tuple[int, ...],
    minimum_relative_improvement: float = 0.01,
    require_improvement: bool = True,
) -> QualificationResult:
    """Select the best feasible temporary expert on fixed scenario seeds."""

    passive_expert = PolicyExpert("passive", PassivePolicy())
    passive_score = _score_expert(passive_expert, env_factory, seeds)

    experts = temporary_experts()
    scores = tuple(_score_expert(expert, env_factory, seeds) for expert in experts)
    feasible_indices = [
        index
        for index, score in enumerate(scores)
        if score.mean_constraint_violation <= passive_score.mean_constraint_violation + 1e-8
    ]
    pool = feasible_indices or list(range(len(scores)))
    best_index = max(pool, key=lambda index: scores[index].mean_return)
    selected_score = scores[best_index]

    required_gain = minimum_relative_improvement * max(abs(passive_score.mean_return), 1.0)
    qualified = bool(
        selected_score.mean_return >= passive_score.mean_return + required_gain
        and selected_score.mean_constraint_violation
        <= passive_score.mean_constraint_violation + 1e-8
    )
    if require_improvement and not qualified:
        raise RuntimeError(
            "No temporary expert passed the qualification gate: "
            f"passive return={passive_score.mean_return:.6g}, "
            f"best={selected_score.name} return={selected_score.mean_return:.6g}. "
            "Use a stronger teacher or explicitly allow an unqualified plumbing-only run."
        )

    return QualificationResult(
        expert=experts[best_index],
        selected=selected_score,
        passive=passive_score,
        candidates=scores,
        qualified=qualified,
    )


def qualify_expert_against_passive(
    expert: Expert,
    env_factory: Callable[[], object],
    seeds: list[int] | tuple[int, ...],
    minimum_relative_improvement: float = 0.01,
    require_improvement: bool = True,
    maximum_fallback_rate: float = 0.01,
) -> QualificationResult:
    """Apply the deployment gate to a supplied expert, including MPC."""

    passive = PolicyExpert("passive", PassivePolicy())
    passive_score = _score_expert(passive, env_factory, seeds)
    selected_score = _score_expert(expert, env_factory, seeds)
    required_gain = minimum_relative_improvement * max(
        abs(passive_score.mean_return),
        1.0,
    )
    qualified = bool(
        selected_score.mean_return >= passive_score.mean_return + required_gain
        and selected_score.mean_constraint_violation
        <= passive_score.mean_constraint_violation + 1e-8
        and selected_score.fallback_rate <= maximum_fallback_rate
    )
    if require_improvement and not qualified:
        raise RuntimeError(
            f"{expert.name} failed qualification: "
            f"return={selected_score.mean_return:.6g} versus "
            f"passive={passive_score.mean_return:.6g}, "
            f"violations={selected_score.mean_constraint_violation:.6g} versus "
            f"{passive_score.mean_constraint_violation:.6g}, "
            f"fallback_rate={selected_score.fallback_rate:.3%}."
        )
    return QualificationResult(
        expert=expert,
        selected=selected_score,
        passive=passive_score,
        candidates=(selected_score,),
        qualified=qualified,
    )


def _score_expert(
    expert: Expert,
    env_factory: Callable[[], object],
    seeds: list[int] | tuple[int, ...],
) -> QualificationScore:
    episode_returns: list[float] = []
    violations: list[float] = []
    rms_accelerations: list[float] = []
    solve_times_ms: list[float] = []
    fallbacks: list[float] = []

    for seed in seeds:
        env = env_factory()
        reset_expert = getattr(expert, "reset", None)
        if callable(reset_expert):
            reset_expert()
        observation, _ = env.reset(seed=int(seed))
        total_reward = 0.0
        total_violation = 0.0
        done = False
        final_info: dict = {}
        while not done:
            label = expert.predict(observation)
            if "latency_ms" in label.diagnostics:
                solve_times_ms.append(float(label.diagnostics["latency_ms"]))
            if "fallback" in label.diagnostics:
                fallbacks.append(float(label.diagnostics["fallback"]))
            observation, reward, terminated, truncated, final_info = env.step(label.action)
            total_reward += float(reward)
            total_violation += float(sum(final_info["constraint_violations"].values()))
            done = bool(terminated or truncated)
        episode_returns.append(total_reward)
        violations.append(total_violation)
        rms_accelerations.append(float(final_info["rms_body_acceleration"]))
        env.close()

    return QualificationScore(
        name=expert.name,
        mean_return=float(np.mean(episode_returns)),
        mean_constraint_violation=float(np.mean(violations)),
        mean_rms_acceleration=float(np.mean(rms_accelerations)),
        mean_solve_time_ms=(
            float(np.mean(solve_times_ms)) if solve_times_ms else 0.0
        ),
        p95_solve_time_ms=(
            float(np.quantile(solve_times_ms, 0.95)) if solve_times_ms else 0.0
        ),
        fallback_rate=float(np.mean(fallbacks)) if fallbacks else 0.0,
    )
