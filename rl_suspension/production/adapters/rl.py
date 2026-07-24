"""Callable seam around the private RL training framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import nn

from rl_suspension.production.training.bc import BCDatasetArrays
from rl_suspension.production.training.finetune import (
    DecayingBCRegularizer,
    PhysicalSafetyFilter,
)


@dataclass
class CallableRLTrainingAdapter:
    initialize_actor_fn: Callable[[nn.Module], None]
    fine_tune_fn: Callable[..., object]

    def initialize_actor(self, student: nn.Module) -> None:
        self.initialize_actor_fn(student)

    def fine_tune(
        self,
        *,
        total_steps: int,
        demonstrations: BCDatasetArrays | None,
        bc_regularizer: DecayingBCRegularizer,
        safety_filter: PhysicalSafetyFilter,
    ) -> object:
        return self.fine_tune_fn(
            total_steps=total_steps,
            demonstrations=demonstrations,
            bc_regularizer=bc_regularizer,
            safety_filter=safety_filter,
        )
