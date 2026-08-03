from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .config import ObservationConfig


class RoadScenario(str, Enum):
    SINGLE_BUMP = "single_bump"
    DOUBLE_BUMP = "double_bump"
    ASYMMETRIC_BUMP = "asymmetric_bump"
    WAVY = "wavy"
    MIXED = "mixed"


@dataclass(frozen=True)
class ParameterRange:
    minimum: float
    maximum: float

    def sample(
        self, batch_size: int, generator: torch.Generator, severity: float
    ) -> Tensor:
        severity = min(max(severity, 0.0), 1.0)
        midpoint = 0.5 * (self.minimum + self.maximum)
        half_width = 0.5 * (self.maximum - self.minimum) * severity
        values = torch.rand(batch_size, generator=generator)
        return midpoint + (2.0 * values - 1.0) * half_width


@dataclass
class DomainSample:
    vehicle: Dict[str, Tensor]
    sensor: Dict[str, Tensor]
    actuator: Dict[str, Tensor]
    ads: Dict[str, Tensor]


class DomainRandomizer:
    """Samples the complete vehicle-to-sensor causal chain by curriculum stage."""

    def __init__(self, seed: int = 0, maximum_stage: int = 5):
        self.generator = torch.Generator().manual_seed(seed)
        self.maximum_stage = maximum_stage
        self.vehicle_ranges = {
            "sprung_mass_scale": ParameterRange(0.8, 1.25),
            "unsprung_mass_scale": ParameterRange(0.85, 1.15),
            "cg_longitudinal_shift_m": ParameterRange(-0.15, 0.15),
            "cg_lateral_shift_m": ParameterRange(-0.05, 0.05),
            "spring_scale": ParameterRange(0.85, 1.15),
            "tire_stiffness_scale": ParameterRange(0.85, 1.15),
            "speed_scale": ParameterRange(0.75, 1.25),
        }
        self.sensor_ranges = {
            "noise_scale": ParameterRange(0.0, 1.0),
            "bias_scale": ParameterRange(0.0, 1.0),
            "delay_s": ParameterRange(0.0, 0.012),
            "quantization_scale": ParameterRange(0.0, 1.0),
        }
        self.actuator_ranges = {
            "force_curve_scale": ParameterRange(0.8, 1.2),
            "time_constant_scale": ParameterRange(0.75, 1.4),
            "hysteresis_scale": ParameterRange(0.0, 1.0),
            "efficiency_scale": ParameterRange(0.75, 1.0),
            "communication_delay_s": ParameterRange(0.0, 0.01),
        }
        self.ads_ranges = {
            "noise_scale": ParameterRange(0.0, 1.0),
            "bias_m": ParameterRange(-0.008, 0.008),
            "missing_probability": ParameterRange(0.0, 0.12),
            "dropout_probability": ParameterRange(0.0, 0.08),
            "timestamp_error_s": ParameterRange(-0.015, 0.015),
            "registration_error_m": ParameterRange(-0.1, 0.1),
        }

    def sample(self, batch_size: int, stage: int) -> DomainSample:
        if not 0 <= stage <= self.maximum_stage:
            raise ValueError("stage must be between zero and maximum_stage.")
        severity = stage / max(self.maximum_stage, 1)

        def sample_group(ranges: Dict[str, ParameterRange]) -> Dict[str, Tensor]:
            return {
                name: bounds.sample(batch_size, self.generator, severity)
                for name, bounds in ranges.items()
            }

        return DomainSample(
            vehicle=sample_group(self.vehicle_ranges),
            sensor=sample_group(self.sensor_ranges),
            actuator=sample_group(self.actuator_ranges),
            ads=sample_group(self.ads_ranges),
        )


class RoadScenarioGenerator:
    def __init__(self, config: ObservationConfig, seed: int = 0):
        self.config = config
        self.generator = torch.Generator().manual_seed(seed)
        self.distance = (
            torch.arange(config.preview_points, dtype=torch.float32)
            * config.preview_resolution_m
        )

    def generate(
        self,
        batch_size: int,
        scenario: Optional[RoadScenario] = None,
        severity: float = 1.0,
    ) -> Tuple[Tensor, Tuple[RoadScenario, ...]]:
        road = torch.zeros(batch_size, 4, self.config.preview_points)
        selected = []
        choices = list(RoadScenario)
        severity = min(max(severity, 0.05), 1.0)
        for batch_index in range(batch_size):
            current = scenario
            if current is None:
                choice_index = int(
                    torch.randint(
                        len(choices), (1,), generator=self.generator
                    ).item()
                )
                current = choices[choice_index]
            selected.append(current)
            road[batch_index] = self._single_scenario(current, severity)
        return road, tuple(selected)

    def _uniform(self, low: float, high: float) -> float:
        value = torch.rand(1, generator=self.generator).item()
        return low + value * (high - low)

    def _bump(self, center_m: float, width_m: float, height_m: float) -> Tensor:
        relative = (self.distance - (center_m - width_m * 0.5)) / width_m
        inside = (relative >= 0.0) & (relative <= 1.0)
        bump = 0.5 * height_m * (1.0 - torch.cos(2.0 * torch.pi * relative))
        return torch.where(inside, bump, torch.zeros_like(bump))

    def _single_scenario(self, scenario: RoadScenario, severity: float) -> Tensor:
        center = self._uniform(1.0, 6.0)
        width = self._uniform(0.25, 1.2)
        height = self._uniform(0.015, 0.11) * severity
        base_bump = self._bump(center, width, height)
        road = base_bump.repeat(4, 1)

        if scenario == RoadScenario.DOUBLE_BUMP:
            spacing = self._uniform(0.5, 1.8)
            second = self._bump(
                min(center + spacing, 7.2),
                self._uniform(0.25, 1.0),
                self._uniform(0.5, 1.0) * height,
            )
            road = road + second.repeat(4, 1)
        elif scenario == RoadScenario.ASYMMETRIC_BUMP:
            left_side = bool(
                torch.randint(2, (1,), generator=self.generator).item()
            )
            wheel_mask = torch.tensor(
                [1.0, 0.0, 1.0, 0.0]
                if left_side
                else [0.0, 1.0, 0.0, 1.0]
            ).unsqueeze(-1)
            road = road * wheel_mask
        elif scenario == RoadScenario.WAVY:
            wavelength = self._uniform(1.0, 5.0)
            phase = self._uniform(0.0, 2.0 * torch.pi)
            wave = height * torch.sin(
                2.0 * torch.pi * self.distance / wavelength + phase
            )
            left_gain = self._uniform(0.7, 1.0)
            right_gain = self._uniform(0.7, 1.0)
            road = torch.stack(
                (left_gain * wave, right_gain * wave, left_gain * wave, right_gain * wave)
            )
        elif scenario == RoadScenario.MIXED:
            wavelength = self._uniform(1.5, 4.5)
            phase = self._uniform(0.0, 2.0 * torch.pi)
            wave = 0.35 * height * torch.sin(
                2.0 * torch.pi * self.distance / wavelength + phase
            )
            side_gain = torch.tensor(
                [
                    self._uniform(0.5, 1.0),
                    self._uniform(0.5, 1.0),
                    self._uniform(0.5, 1.0),
                    self._uniform(0.5, 1.0),
                ]
            ).unsqueeze(-1)
            road = side_gain * (road + wave)
        return road


class ADSCorruptor:
    """Applies range-dependent noise, missing points, registration error, and dropout."""

    def __init__(self, config: ObservationConfig, seed: int = 0):
        self.config = config
        self.generator = torch.Generator().manual_seed(seed)
        self.distance_fraction = torch.linspace(0.0, 1.0, config.preview_points)

    def apply(
        self,
        clean_height_m: Tensor,
        domain: DomainSample,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch = clean_height_m.shape[0]
        if tuple(clean_height_m.shape[1:]) != (4, self.config.preview_points):
            raise ValueError("clean_height_m must have shape [batch, 4, 160].")
        range_weight = (0.25 + 0.75 * self.distance_fraction).view(1, 1, -1)
        noise_scale = domain.ads["noise_scale"].view(batch, 1, 1)
        noise = torch.randn(
            clean_height_m.shape, generator=self.generator
        ) * (0.002 + 0.012 * noise_scale) * range_weight
        bias = domain.ads["bias_m"].view(batch, 1, 1)
        corrupted = clean_height_m + noise + bias

        registration = domain.ads["registration_error_m"]
        point_shift = torch.round(
            registration / self.config.preview_resolution_m
        ).to(torch.int64)
        for index in range(batch):
            corrupted[index] = torch.roll(
                corrupted[index], int(point_shift[index].item()), dims=-1
            )

        missing_probability = domain.ads["missing_probability"].view(batch, 1, 1)
        missing = torch.rand(
            clean_height_m.shape, generator=self.generator
        ) < missing_probability
        dropout_probability = domain.ads["dropout_probability"]
        complete_dropout = torch.rand(
            batch, generator=self.generator
        ) < dropout_probability
        missing[complete_dropout] = True
        valid = (~missing).to(clean_height_m.dtype)

        confidence = (1.0 - range_weight) * 0.35 + 0.65
        confidence = confidence.expand_as(clean_height_m)
        confidence = confidence * valid * (1.0 - noise_scale.clamp(0.0, 1.0))
        corrupted = torch.where(valid.bool(), corrupted, torch.zeros_like(corrupted))
        return corrupted, confidence.clamp(0.0, 1.0), valid


class SensorCorruptor:
    def __init__(self, seed: int = 0):
        self.generator = torch.Generator().manual_seed(seed)

    def apply(
        self,
        clean_signal: Tensor,
        noise_standard_deviation: Tensor,
        bias_limit: Tensor,
        domain: DomainSample,
    ) -> Tensor:
        batch = clean_signal.shape[0]
        noise_scale = domain.sensor["noise_scale"].view(
            batch, *([1] * (clean_signal.ndim - 1))
        )
        bias_scale = domain.sensor["bias_scale"].view(
            batch, *([1] * (clean_signal.ndim - 1))
        )
        noise = torch.randn(
            clean_signal.shape, generator=self.generator
        ) * noise_standard_deviation * noise_scale
        bias_direction = 2.0 * torch.rand(
            (batch,) + (1,) * (clean_signal.ndim - 1),
            generator=self.generator,
        ) - 1.0
        return clean_signal + noise + bias_direction * bias_limit * bias_scale
