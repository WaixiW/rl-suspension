"""Simple force-level baselines for active suspension benchmarking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_suspension.models.types import FloatArray


class BaselinePolicy:
    """Policy interface matching the environment's normalized 4D action."""

    def predict(self, observation: FloatArray, deterministic: bool = True) -> tuple[FloatArray, None]:
        raise NotImplementedError


@dataclass
class PassivePolicy(BaselinePolicy):
    """Zero active force baseline."""

    def predict(self, observation: FloatArray, deterministic: bool = True) -> tuple[FloatArray, None]:
        return np.zeros(4, dtype=np.float32), None


@dataclass
class SkyhookGroundhookPolicy(BaselinePolicy):
    """Classical velocity feedback baseline in normalized force units."""

    skyhook_gain: float = 0.08
    groundhook_gain: float = 0.03

    def predict(self, observation: FloatArray, deterministic: bool = True) -> tuple[FloatArray, None]:
        obs = np.asarray(observation, dtype=np.float64)
        body_velocity = obs[7]
        wheel_velocities = obs[10:14]
        suspension_velocities = obs[18:22]
        force = -self.skyhook_gain * body_velocity - self.groundhook_gain * wheel_velocities
        force += -0.02 * suspension_velocities
        return np.clip(force, -1.0, 1.0).astype(np.float32), None


@dataclass
class PreviewRulePolicy(BaselinePolicy):
    """Feedforward rule using compressed ADS bump features."""

    preview_gain: float = 0.45
    damping_gain: float = 0.04

    def predict(self, observation: FloatArray, deterministic: bool = True) -> tuple[FloatArray, None]:
        obs = np.asarray(observation, dtype=np.float64)
        suspension_velocities = obs[18:22]
        speed = max(float(obs[42]), 0.1)
        peak_distance, peak_height, _, asymmetry, *_ = obs[43:50]
        time_to_bump = peak_distance / speed
        timing = np.exp(-((time_to_bump - 0.12) ** 2) / 0.02)
        base = -self.preview_gain * np.sign(peak_height) * abs(peak_height) * timing
        left = base - 0.5 * self.preview_gain * asymmetry * timing
        right = base + 0.5 * self.preview_gain * asymmetry * timing
        force = np.array([left, right, 0.5 * left, 0.5 * right], dtype=np.float64)
        force += -self.damping_gain * suspension_velocities
        return np.clip(force, -1.0, 1.0).astype(np.float32), None
