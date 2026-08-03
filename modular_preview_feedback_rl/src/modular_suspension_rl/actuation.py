from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn

from .config import ActuatorConfig


def pack_commands(currents_a: Tensor, rpm: Tensor) -> Tensor:
    """Packs [B,4,2] currents and [B,4] rpm as eight currents then four rpm."""

    if currents_a.ndim != 3 or tuple(currents_a.shape[1:]) != (4, 2):
        raise ValueError("currents_a must have shape [batch, 4, 2].")
    if tuple(rpm.shape) != (currents_a.shape[0], 4):
        raise ValueError("rpm must have shape [batch, 4].")
    return torch.cat((currents_a.reshape(currents_a.shape[0], 8), rpm), dim=-1)


def unpack_commands(commands: Tensor) -> Tuple[Tensor, Tensor]:
    if commands.ndim != 2 or commands.shape[-1] != 12:
        raise ValueError("commands must have shape [batch, 12].")
    return commands[:, :8].reshape(commands.shape[0], 4, 2), commands[:, 8:]


@dataclass
class ProjectionResult:
    force_n: Tensor
    correction_n: Tensor
    active: Tensor


class ForceProjector(nn.Module):
    """Projects force requests using explicit force, travel, and tire-load limits.

    Sign convention: positive requested force increases positive suspension
    travel and increases tire normal load. Integrations with another convention
    must transform signs before calling this projector.
    """

    def __init__(
        self,
        force_bounds_n: Tuple[float, float],
        travel_soft_limit_m: float = 0.07,
        travel_hard_limit_m: float = 0.09,
        tire_load_min_n: float = 100.0,
        tire_load_margin_n: float = 500.0,
    ):
        super().__init__()
        if force_bounds_n[0] >= force_bounds_n[1]:
            raise ValueError("force_bounds_n must be ordered.")
        if not 0.0 < travel_soft_limit_m < travel_hard_limit_m:
            raise ValueError("Travel limits must satisfy 0 < soft < hard.")
        self.force_bounds_n = force_bounds_n
        self.travel_soft_limit_m = travel_soft_limit_m
        self.travel_hard_limit_m = travel_hard_limit_m
        self.tire_load_min_n = tire_load_min_n
        self.tire_load_margin_n = tire_load_margin_n

    def forward(
        self,
        requested_force_n: Tensor,
        suspension_travel_m: Tensor,
        tire_load_n: Tensor,
    ) -> ProjectionResult:
        if requested_force_n.ndim != 2 or requested_force_n.shape[-1] != 4:
            raise ValueError("requested_force_n must have shape [batch, 4].")
        if suspension_travel_m.shape != requested_force_n.shape:
            raise ValueError("suspension_travel_m must match requested force.")
        if tire_load_n.shape != requested_force_n.shape:
            raise ValueError("tire_load_n must match requested force.")

        projected = requested_force_n.clamp(*self.force_bounds_n)

        travel_scale = (
            (self.travel_hard_limit_m - suspension_travel_m.abs())
            / (self.travel_hard_limit_m - self.travel_soft_limit_m)
        ).clamp(0.0, 1.0)
        worsens_travel = projected * suspension_travel_m > 0.0
        projected = torch.where(worsens_travel, projected * travel_scale, projected)

        load_scale = (
            (tire_load_n - self.tire_load_min_n) / self.tire_load_margin_n
        ).clamp(0.0, 1.0)
        unloads_tire = projected < 0.0
        projected = torch.where(unloads_tire, projected * load_scale, projected)

        correction = projected - requested_force_n
        return ProjectionResult(
            force_n=projected,
            correction_n=correction,
            active=correction.abs() > 1e-5,
        )


class NonlinearActuatorModel(nn.Module):
    """Compact dynamic model for two directional valves and one pump per corner."""

    def __init__(self, config: ActuatorConfig):
        super().__init__()
        self.config = config

    def direction_weights(self, suspension_velocity_mps: Tensor) -> Tuple[Tensor, Tensor]:
        compression = torch.sigmoid(-self.config.direction_softness * suspension_velocity_mps)
        rebound = 1.0 - compression
        return compression, rebound

    def steady_force(
        self,
        currents_a: Tensor,
        rpm: Tensor,
        suspension_velocity_mps: Tensor,
    ) -> Tensor:
        compression, rebound = self.direction_weights(suspension_velocity_mps)
        directional_current = (
            compression * currents_a[..., 0] + rebound * currents_a[..., 1]
        )
        damping = (
            self.config.passive_damping_ns_per_m
            + self.config.current_damping_gain_ns_per_m_per_a * directional_current
        )
        damping_force = -damping * suspension_velocity_mps
        pump_force = self.config.pump_force_gain_n_per_rpm * rpm
        return damping_force + pump_force

    def next_force(
        self,
        commands: Tensor,
        suspension_velocity_mps: Tensor,
        current_force_n: Tensor,
    ) -> Tensor:
        currents, rpm = unpack_commands(commands)
        target_force = self.steady_force(currents, rpm, suspension_velocity_mps)
        alpha = min(
            self.config.sample_time_s / self.config.force_time_constant_s,
            1.0,
        )
        return current_force_n + alpha * (target_force - current_force_n)

    def normalized_linear_model(
        self,
        suspension_velocity_mps: Tensor,
        current_force_n: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns base, coefficients, command center, and scale in normalized space."""

        batch = suspension_velocity_mps.shape[0]
        compression, rebound = self.direction_weights(suspension_velocity_mps)
        alpha = min(
            self.config.sample_time_s / self.config.force_time_constant_s,
            1.0,
        )
        velocity = suspension_velocity_mps
        current_gain = self.config.current_damping_gain_ns_per_m_per_a

        physical_coefficients = torch.stack(
            (
                -alpha * current_gain * velocity * compression,
                -alpha * current_gain * velocity * rebound,
                torch.full_like(velocity, alpha * self.config.pump_force_gain_n_per_rpm),
            ),
            dim=-1,
        )
        base = (
            (1.0 - alpha) * current_force_n
            - alpha * self.config.passive_damping_ns_per_m * velocity
        )

        current_min, current_max = self.config.current_bounds_a
        rpm_min, rpm_max = self.config.rpm_bounds
        center = suspension_velocity_mps.new_tensor(
            [
                (current_min + current_max) * 0.5,
                (current_min + current_max) * 0.5,
                (rpm_min + rpm_max) * 0.5,
            ]
        ).view(1, 1, 3).expand(batch, 4, 3)
        scale = suspension_velocity_mps.new_tensor(
            [
                (current_max - current_min) * 0.5,
                (current_max - current_min) * 0.5,
                (rpm_max - rpm_min) * 0.5,
            ]
        ).view(1, 1, 3).expand(batch, 4, 3)

        normalized_base = base + (physical_coefficients * center).sum(dim=-1)
        normalized_coefficients = physical_coefficients * scale
        return normalized_base, normalized_coefficients, center, scale


@dataclass
class AllocationResult:
    commands: Tensor
    predicted_force_n: Tensor
    tracking_error_n: Tensor
    saturated: Tensor


class DynamicActuatorAllocator:
    """Projected model-based allocation; it is intentionally outside the RL policy."""

    def __init__(self, config: ActuatorConfig, model: NonlinearActuatorModel):
        self.config = config
        self.model = model

    def allocate(
        self,
        target_force_n: Tensor,
        suspension_velocity_mps: Tensor,
        current_force_n: Tensor,
        previous_commands: Tensor,
    ) -> AllocationResult:
        if target_force_n.ndim != 2 or target_force_n.shape[-1] != 4:
            raise ValueError("target_force_n must have shape [batch, 4].")
        if suspension_velocity_mps.shape != target_force_n.shape:
            raise ValueError("suspension_velocity_mps must match target force.")
        if current_force_n.shape != target_force_n.shape:
            raise ValueError("current_force_n must match target force.")

        previous_currents, previous_rpm = unpack_commands(previous_commands)
        previous_corner_commands = torch.cat(
            (previous_currents, previous_rpm.unsqueeze(-1)), dim=-1
        )
        base, coefficients, center, scale = self.model.normalized_linear_model(
            suspension_velocity_mps, current_force_n
        )
        previous_normalized = ((previous_corner_commands - center) / scale).clamp(-1.0, 1.0)

        force_scale = max(abs(value) for value in self.config.force_bounds_n)
        coefficient_column = coefficients.unsqueeze(-1)
        identity = torch.eye(3, device=target_force_n.device, dtype=target_force_n.dtype)
        identity = identity.view(1, 1, 3, 3)
        regularization = self.config.energy_weight + self.config.slew_weight
        matrix = (
            coefficient_column @ coefficient_column.transpose(-1, -2)
        ) / (force_scale ** 2) + regularization * identity
        rhs = (
            coefficients * (target_force_n - base).unsqueeze(-1) / (force_scale ** 2)
            + self.config.slew_weight * previous_normalized
        )
        normalized = torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)

        slew_physical = target_force_n.new_tensor(
            [
                self.config.current_slew_a_per_s,
                self.config.current_slew_a_per_s,
                self.config.rpm_slew_per_s,
            ]
        ).view(1, 1, 3)
        max_normalized_change = slew_physical * self.config.sample_time_s / scale
        lower = torch.maximum(
            previous_normalized - max_normalized_change,
            torch.full_like(previous_normalized, -1.0),
        )
        upper = torch.minimum(
            previous_normalized + max_normalized_change,
            torch.full_like(previous_normalized, 1.0),
        )
        normalized = normalized.clamp(min=lower, max=upper)

        for _ in range(self.config.allocator_iterations):
            predicted = base + (coefficients * normalized).sum(dim=-1)
            error = (predicted - target_force_n) / force_scale
            gradient = (
                2.0 * coefficients * error.unsqueeze(-1) / force_scale
                + 2.0 * self.config.energy_weight * normalized
                + 2.0 * self.config.slew_weight * (normalized - previous_normalized)
            )
            normalized = (normalized - self.config.allocator_step_size * gradient).clamp(
                min=lower, max=upper
            )

        corner_commands = center + scale * normalized
        currents = corner_commands[..., :2]
        rpm = corner_commands[..., 2]
        commands = pack_commands(currents, rpm)
        predicted_force = self.model.next_force(
            commands, suspension_velocity_mps, current_force_n
        )
        tracking_error = predicted_force - target_force_n
        saturated = (normalized <= lower + 1e-5) | (normalized >= upper - 1e-5)
        return AllocationResult(
            commands=commands.detach(),
            predicted_force_n=predicted_force.detach(),
            tracking_error_n=tracking_error.detach(),
            saturated=saturated.any(dim=-1).detach(),
        )
