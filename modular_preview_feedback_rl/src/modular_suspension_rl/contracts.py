from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor, nn

from .config import ObservationConfig, WHEEL_ORDER


@dataclass
class ObservationBatch:
    """Normalized outer-loop observation with an explicit, validated layout."""

    road: Tensor
    feedback_history: Tensor
    speed: Tensor
    preview_confidence: Tensor
    previous_gate: Tensor
    suspension_velocity: Tensor
    actuator_force: Tensor
    previous_commands: Tensor

    @property
    def batch_size(self) -> int:
        return self.road.shape[0]

    def validate(self, config: ObservationConfig) -> None:
        expected_road = (
            self.batch_size,
            len(WHEEL_ORDER),
            config.preview_feature_count,
            config.preview_points,
        )
        expected_history = (
            self.batch_size,
            config.feedback_history,
            config.feedback_dim,
        )
        if tuple(self.road.shape) != expected_road:
            raise ValueError(
                "road must have shape {}, got {}".format(expected_road, tuple(self.road.shape))
            )
        if tuple(self.feedback_history.shape) != expected_history:
            raise ValueError(
                "feedback_history must have shape {}, got {}".format(
                    expected_history, tuple(self.feedback_history.shape)
                )
            )
        for name in ("preview_confidence", "previous_gate", "suspension_velocity", "actuator_force"):
            value = getattr(self, name)
            if tuple(value.shape) != (self.batch_size, len(WHEEL_ORDER)):
                raise ValueError("{} must have shape [batch, 4].".format(name))
        if tuple(self.speed.shape) != (self.batch_size, 1):
            raise ValueError("speed must have shape [batch, 1].")
        if tuple(self.previous_commands.shape) != (self.batch_size, 12):
            raise ValueError("previous_commands must have shape [batch, 12].")
        tensors = (
            self.road,
            self.feedback_history,
            self.speed,
            self.preview_confidence,
            self.previous_gate,
            self.suspension_velocity,
            self.actuator_force,
            self.previous_commands,
        )
        if not all(torch.isfinite(value).all() for value in tensors):
            raise ValueError("Observation contains NaN or infinite values.")

    def to(self, device: torch.device) -> "ObservationBatch":
        return ObservationBatch(
            **{
                name: value.to(device)
                for name, value in self.__dict__.items()
            }
        )

    def detach(self) -> "ObservationBatch":
        return ObservationBatch(
            **{
                name: value.detach()
                for name, value in self.__dict__.items()
            }
        )


class RoadPreviewProcessor(nn.Module):
    """Converts four geometry-aligned height tracks into six spatial features."""

    HEIGHT = 0
    SLOPE = 1
    CURVATURE = 2
    CONFIDENCE = 3
    ARRIVAL_TIME = 4
    VALID = 5

    def __init__(self, config: ObservationConfig):
        super().__init__()
        config.validate()
        self.config = config
        distance = torch.arange(config.preview_points, dtype=torch.float32)
        distance = distance * config.preview_resolution_m
        self.register_buffer("distance_m", distance, persistent=False)

    def forward(
        self,
        heights_m: Tensor,
        confidence: Tensor,
        speed_mps: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        expected = (heights_m.shape[0], len(WHEEL_ORDER), self.config.preview_points)
        if tuple(heights_m.shape) != expected or tuple(confidence.shape) != expected:
            raise ValueError("heights and confidence must have shape [batch, 4, 160].")
        if tuple(speed_mps.shape) != (heights_m.shape[0], 1):
            raise ValueError("speed_mps must have shape [batch, 1].")
        if valid_mask is None:
            valid_mask = torch.ones_like(heights_m)
        if tuple(valid_mask.shape) != expected:
            raise ValueError("valid_mask must have shape [batch, 4, 160].")

        confidence = confidence.clamp(0.0, 1.0) * valid_mask
        slope = self._first_derivative(heights_m)
        curvature = self._first_derivative(slope)

        distance = self.distance_m.to(heights_m).view(1, 1, -1)
        axle_delay_distance = heights_m.new_tensor(
            [0.0, 0.0, self.config.wheelbase_m, self.config.wheelbase_m]
        ).view(1, 4, 1)
        speed = speed_mps.abs().clamp_min(self.config.min_speed_mps).view(-1, 1, 1)
        arrival_time = (distance + axle_delay_distance) / speed
        arrival_time = arrival_time.expand_as(heights_m)

        return torch.stack(
            (
                heights_m,
                slope,
                curvature,
                confidence,
                arrival_time,
                valid_mask.clamp(0.0, 1.0),
            ),
            dim=2,
        )

    def _first_derivative(self, values: Tensor) -> Tensor:
        derivative = torch.empty_like(values)
        spacing = self.config.preview_resolution_m
        derivative[..., 1:-1] = (values[..., 2:] - values[..., :-2]) / (2.0 * spacing)
        derivative[..., 0] = (values[..., 1] - values[..., 0]) / spacing
        derivative[..., -1] = (values[..., -1] - values[..., -2]) / spacing
        return derivative


class PhysicalNormalizer(nn.Module):
    """Deterministic normalization using documented physical scales."""

    def __init__(self, road_scales: Iterable[float], feedback_scales: Iterable[float]):
        super().__init__()
        road = torch.as_tensor(list(road_scales), dtype=torch.float32)
        feedback = torch.as_tensor(list(feedback_scales), dtype=torch.float32)
        if road.numel() != 6:
            raise ValueError("road_scales must contain six feature scales.")
        if (road <= 0).any() or (feedback <= 0).any():
            raise ValueError("All normalization scales must be positive.")
        self.register_buffer("road_scales", road.view(1, 1, 6, 1))
        self.register_buffer("feedback_scales", feedback.view(1, 1, -1))

    def forward(self, road: Tensor, feedback_history: Tensor) -> Dict[str, Tensor]:
        if feedback_history.shape[-1] != self.feedback_scales.shape[-1]:
            raise ValueError("feedback_scales do not match the feedback vector.")
        return {
            "road": road / self.road_scales,
            "feedback_history": feedback_history / self.feedback_scales,
        }


def concatenate_feedback_state(
    named_components: Dict[str, Tensor],
    expected_dim: int,
) -> Tensor:
    """Concatenates explicitly named state components without hiding ordering."""

    if not named_components:
        raise ValueError("At least one feedback component is required.")
    batch_sizes = {value.shape[0] for value in named_components.values()}
    if len(batch_sizes) != 1:
        raise ValueError("All feedback components must share a batch dimension.")
    flattened = [value.reshape(value.shape[0], -1) for value in named_components.values()]
    result = torch.cat(flattened, dim=-1)
    if result.shape[-1] != expected_dim:
        details = ", ".join(
            "{}={}".format(name, value[0].numel())
            for name, value in named_components.items()
        )
        raise ValueError(
            "Feedback dimension is {}, expected {} ({})".format(
                result.shape[-1], expected_dim, details
            )
        )
    return result
