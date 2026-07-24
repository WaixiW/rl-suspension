"""Deterministic safe-controller and physical command projection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    ActionSchema,
    ObservationV1,
)


@dataclass
class ConstantSafeController:
    action_schema: ActionSchema = field(default_factory=ActionSchema)
    name: str = "constant_safe"

    def predict(self, observation: ObservationV1) -> np.ndarray:
        del observation
        return np.asarray(self.action_schema.safe_action, dtype=np.float64)


@dataclass
class SafetyProjector:
    action_schema: ActionSchema = field(default_factory=lambda: DEFAULT_ACTION_SCHEMA)

    def project(
        self,
        proposed_action: np.ndarray,
        previous_action: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        return self.action_schema.project(proposed_action, previous_action, dt)
