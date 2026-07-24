"""RL-implementation-neutral actor initialization and fine-tuning hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np
from torch import Tensor, nn

from rl_suspension.production.adapters.safe_controller import SafetyProjector
from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    DEFAULT_OBSERVATION_SCHEMA,
    ActionSchema,
    ObservationSchema,
)
from rl_suspension.production.training.bc import BCDatasetArrays
from rl_suspension.production.training.losses import (
    normalized_weighted_huber_loss,
)


@dataclass(frozen=True)
class SafetyEvent:
    step: int
    proposed_action: np.ndarray
    projected_action: np.ndarray
    previous_action: np.ndarray
    intervened: bool


class SafetyCallback(Protocol):
    def __call__(self, event: SafetyEvent) -> None: ...


class PhysicalSafetyFilter:
    """Project each RL action through physical bounds and slew constraints."""

    def __init__(
        self,
        *,
        projector: SafetyProjector | None = None,
        action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
        observation_schema: ObservationSchema = DEFAULT_OBSERVATION_SCHEMA,
        callbacks: Sequence[SafetyCallback] = (),
    ) -> None:
        self.action_schema = action_schema
        self.observation_schema = observation_schema
        self.projector = projector or SafetyProjector(action_schema)
        self.callbacks = tuple(callbacks)

    def project(
        self,
        proposed_action: np.ndarray,
        previous_action: np.ndarray,
        *,
        step: int,
    ) -> np.ndarray:
        proposed = np.asarray(proposed_action, dtype=np.float64)
        previous = self.action_schema.validate(previous_action)
        try:
            projected = self.projector.project(
                proposed,
                previous,
                self.observation_schema.control_period_s,
            )
            projected = self.action_schema.validate(projected)
        except (TypeError, ValueError, FloatingPointError):
            projected = self.action_schema.project(
                np.asarray(self.action_schema.safe_action, dtype=np.float64),
                previous,
                self.observation_schema.control_period_s,
            )
        comparable = proposed.shape == projected.shape and np.all(np.isfinite(proposed))
        intervened = not comparable or not np.allclose(
            proposed,
            projected,
            rtol=0.0,
            atol=1e-12,
        )
        event = SafetyEvent(
            step=int(step),
            proposed_action=proposed.copy(),
            projected_action=projected.copy(),
            previous_action=previous.copy(),
            intervened=bool(intervened),
        )
        for callback in self.callbacks:
            callback(event)
        return projected


@dataclass(frozen=True)
class BCRegularizationConfig:
    initial_coefficient: float = 1.0
    final_coefficient: float = 0.0
    decay_steps: int = 100_000
    schedule: str = "linear"
    huber_delta: float = 0.05
    channel_weights: tuple[float, ...] = (1.0,) * 12

    def __post_init__(self) -> None:
        if self.initial_coefficient < 0.0 or self.final_coefficient < 0.0:
            raise ValueError("BC coefficients must be nonnegative")
        if self.final_coefficient > self.initial_coefficient:
            raise ValueError("BC regularization must decay, not increase")
        if self.decay_steps <= 0:
            raise ValueError("decay_steps must be positive")
        if self.schedule not in {"linear", "exponential"}:
            raise ValueError("schedule must be 'linear' or 'exponential'")


class DecayingBCRegularizer:
    """Callable loss term an arbitrary RL adapter can apply to actor batches."""

    def __init__(
        self,
        config: BCRegularizationConfig | None = None,
        *,
        action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
    ) -> None:
        self.config = config or BCRegularizationConfig()
        self.action_schema = action_schema
        if len(self.config.channel_weights) != action_schema.dimension:
            raise ValueError("channel_weights do not match the action schema")

    def coefficient(self, step: int) -> float:
        progress = min(max(float(step) / self.config.decay_steps, 0.0), 1.0)
        initial = self.config.initial_coefficient
        final = self.config.final_coefficient
        if self.config.schedule == "linear":
            return initial + progress * (final - initial)
        if initial == 0.0:
            return 0.0
        ratio = max(final, 1e-12) / initial
        value = initial * ratio**progress
        return 0.0 if progress >= 1.0 and final == 0.0 else value

    def __call__(
        self,
        predicted_normalized: Tensor,
        expert_actions: Tensor,
        step: int,
        *,
        expert_actions_normalized: bool = False,
        sample_weights: Tensor | None = None,
    ) -> Tensor:
        loss = normalized_weighted_huber_loss(
            predicted_normalized,
            expert_actions,
            action_schema=self.action_schema,
            target_is_normalized=expert_actions_normalized,
            channel_weights=self.config.channel_weights,
            sample_weights=sample_weights,
            delta=self.config.huber_delta,
        )
        return self.coefficient(step) * loss


class RLTrainingAdapter(Protocol):
    """Minimal seam implemented by a project-specific RL algorithm wrapper."""

    def initialize_actor(self, student: nn.Module) -> None: ...

    def fine_tune(
        self,
        *,
        total_steps: int,
        demonstrations: BCDatasetArrays | None,
        bc_regularizer: DecayingBCRegularizer,
        safety_filter: PhysicalSafetyFilter,
    ) -> object: ...


@dataclass(frozen=True)
class FineTuneConfig:
    total_steps: int = 100_000
    initialize_actor: bool = True
    run_fine_tuning: bool = True

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")


@dataclass(frozen=True)
class FineTuneResult:
    initialized: bool
    fine_tuned: bool
    adapter_result: object | None


class FineTuneOrchestrator:
    """Wire a student into an RL adapter without choosing an RL algorithm."""

    def __init__(
        self,
        adapter: RLTrainingAdapter,
        *,
        config: FineTuneConfig | None = None,
        regularizer: DecayingBCRegularizer | None = None,
        safety_filter: PhysicalSafetyFilter | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config or FineTuneConfig()
        self.regularizer = regularizer or DecayingBCRegularizer()
        self.safety_filter = safety_filter or PhysicalSafetyFilter()

    def run(
        self,
        student: nn.Module,
        demonstrations: BCDatasetArrays | None = None,
    ) -> FineTuneResult:
        initialized = False
        if self.config.initialize_actor:
            self.adapter.initialize_actor(student)
            initialized = True

        adapter_result = None
        if self.config.run_fine_tuning:
            adapter_result = self.adapter.fine_tune(
                total_steps=self.config.total_steps,
                demonstrations=demonstrations,
                bc_regularizer=self.regularizer,
                safety_filter=self.safety_filter,
            )
        return FineTuneResult(
            initialized=initialized,
            fine_tuned=self.config.run_fine_tuning,
            adapter_result=adapter_result,
        )


def intervention_counter() -> tuple[SafetyCallback, Callable[[], int]]:
    """Create a lightweight safety callback and count accessor."""

    count = 0

    def callback(event: SafetyEvent) -> None:
        nonlocal count
        count += int(event.intervened)

    return callback, lambda: count
