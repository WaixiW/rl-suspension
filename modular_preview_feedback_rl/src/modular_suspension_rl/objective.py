from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor

from .config import RewardConfig


@dataclass
class RewardSignals:
    vertical_acceleration_mps2: Tensor
    pitch_rate_radps: Tensor
    pitch_acceleration_radps2: Tensor
    roll_rate_radps: Tensor
    roll_acceleration_radps2: Tensor
    passenger_acceleration_mps2: Tensor
    jerk_mps3: Tensor
    suspension_travel_m: Tensor
    tire_load_n: Tensor
    tire_load_variation_n: Tensor
    wheel_hop_mps: Tensor
    electrical_power_w: Tensor
    command_slew_normalized: Tensor
    force_tracking_error_n: Tensor
    gate: Tensor
    previous_gate: Tensor
    preview_confidence: Tensor
    end_stop_event: Tensor
    hardware_violation: Tensor

    @property
    def batch_size(self) -> int:
        return self.vertical_acceleration_mps2.shape[0]

    def validate(self) -> None:
        batch = self.batch_size
        scalar_names = (
            "vertical_acceleration_mps2",
            "pitch_rate_radps",
            "pitch_acceleration_radps2",
            "roll_rate_radps",
            "roll_acceleration_radps2",
            "jerk_mps3",
            "electrical_power_w",
            "command_slew_normalized",
            "end_stop_event",
            "hardware_violation",
        )
        for name in scalar_names:
            value = getattr(self, name)
            if tuple(value.shape) not in ((batch,), (batch, 1)):
                raise ValueError("{} must be scalar per batch item.".format(name))
        wheel_names = (
            "suspension_travel_m",
            "tire_load_n",
            "tire_load_variation_n",
            "wheel_hop_mps",
            "force_tracking_error_n",
            "gate",
            "previous_gate",
            "preview_confidence",
        )
        for name in wheel_names:
            if tuple(getattr(self, name).shape) != (batch, 4):
                raise ValueError("{} must have shape [batch, 4].".format(name))
        if self.passenger_acceleration_mps2.ndim != 2:
            raise ValueError("passenger_acceleration_mps2 must have shape [batch, occupants].")


@dataclass
class RewardBreakdown:
    reward: Tensor
    terms: Dict[str, Tensor]
    hard_violation: Tensor


class SuspensionReward:
    """Normalized comfort, road-holding, actuation, and safety objective."""

    def __init__(self, config: RewardConfig):
        self.config = config

    @staticmethod
    def _scalar(value: Tensor) -> Tensor:
        return value.reshape(value.shape[0], -1).mean(dim=-1)

    @staticmethod
    def _mean_square(value: Tensor, scale: float) -> Tensor:
        return (value / scale).square().reshape(value.shape[0], -1).mean(dim=-1)

    def __call__(self, signals: RewardSignals) -> RewardBreakdown:
        signals.validate()
        scales = self.config.scales
        weights = self.config.weights

        terms = {
            "vertical_acceleration": self._mean_square(
                signals.vertical_acceleration_mps2,
                scales.vertical_acceleration_mps2,
            ),
            "pitch": self._mean_square(
                signals.pitch_rate_radps, scales.angular_rate_radps
            )
            + self._mean_square(
                signals.pitch_acceleration_radps2,
                scales.angular_acceleration_radps2,
            ),
            "roll": self._mean_square(
                signals.roll_rate_radps, scales.angular_rate_radps
            )
            + self._mean_square(
                signals.roll_acceleration_radps2,
                scales.angular_acceleration_radps2,
            ),
            "passenger_acceleration": self._mean_square(
                signals.passenger_acceleration_mps2,
                scales.vertical_acceleration_mps2,
            ),
            "jerk": self._mean_square(signals.jerk_mps3, scales.jerk_mps3),
            "suspension_travel": self._mean_square(
                torch.relu(
                    signals.suspension_travel_m.abs()
                    - self.config.travel_soft_limit_m
                ),
                max(
                    scales.suspension_travel_m - self.config.travel_soft_limit_m,
                    1e-4,
                ),
            ),
            "tire_load_variation": self._mean_square(
                signals.tire_load_variation_n, scales.tire_load_n
            ),
            "wheel_hop": self._mean_square(
                signals.wheel_hop_mps, scales.wheel_hop_mps
            ),
            "electrical_power": self._mean_square(
                signals.electrical_power_w, scales.electrical_power_w
            ),
            "command_slew": self._mean_square(
                signals.command_slew_normalized, scales.command_slew
            ),
            "force_tracking": self._mean_square(
                signals.force_tracking_error_n, scales.force_error_n
            ),
            "gate_smoothness": (signals.gate - signals.previous_gate)
            .square()
            .mean(dim=-1),
            "invalid_preview_use": (
                signals.gate * (1.0 - signals.preview_confidence.clamp(0.0, 1.0))
            )
            .square()
            .mean(dim=-1),
        }

        weighted = (
            weights.vertical_acceleration * terms["vertical_acceleration"]
            + weights.pitch * terms["pitch"]
            + weights.roll * terms["roll"]
            + weights.passenger_acceleration * terms["passenger_acceleration"]
            + weights.jerk * terms["jerk"]
            + weights.suspension_travel * terms["suspension_travel"]
            + weights.tire_load_variation * terms["tire_load_variation"]
            + weights.wheel_hop * terms["wheel_hop"]
            + weights.electrical_power * terms["electrical_power"]
            + weights.command_slew * terms["command_slew"]
            + weights.force_tracking * terms["force_tracking"]
            + weights.gate_smoothness * terms["gate_smoothness"]
            + weights.invalid_preview_use * terms["invalid_preview_use"]
        )
        tire_contact_violation = (
            signals.tire_load_n < self.config.tire_load_min_n
        ).any(dim=-1)
        hard_violation = torch.maximum(
            self._scalar(signals.end_stop_event).clamp(0.0, 1.0),
            self._scalar(signals.hardware_violation).clamp(0.0, 1.0),
        )
        hard_violation = torch.maximum(
            hard_violation, tire_contact_violation.to(weighted.dtype)
        )
        reward = -(weighted + weights.hard_violation * hard_violation)
        return RewardBreakdown(
            reward=reward,
            terms=terms,
            hard_violation=hard_violation,
        )
