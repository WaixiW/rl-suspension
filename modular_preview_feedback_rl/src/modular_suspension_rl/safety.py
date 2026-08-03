from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
from torch import Tensor

from .actuation import (
    AllocationResult,
    DynamicActuatorAllocator,
    ForceProjector,
    ProjectionResult,
    unpack_commands,
)
from .contracts import ObservationBatch
from .networks import ActorOutput, ModularActor


@dataclass(frozen=True)
class SafetyConfig:
    minimum_preview_confidence: float = 0.05
    maximum_force_discontinuity_n: float = 50.0
    maximum_tracking_error_n: float = 500.0
    settling_band_fraction: float = 0.05
    maximum_settling_time_s: float = 0.2


@dataclass
class SafetyState:
    suspension_travel_m: Tensor
    tire_load_n: Tensor
    sensor_valid: Tensor
    hardware_healthy: Tensor


@dataclass
class SafetyDecision:
    requested_force_n: Tensor
    projected: ProjectionResult
    allocation: AllocationResult
    fallback_active: Tensor
    preview_available: Tensor


FallbackController = Callable[[ObservationBatch], Tensor]


class SafetySupervisor:
    """Runtime shield around policy force, projection, and command allocation."""

    def __init__(
        self,
        projector: ForceProjector,
        allocator: DynamicActuatorAllocator,
        fallback: Optional[FallbackController] = None,
        config: SafetyConfig = SafetyConfig(),
    ):
        self.projector = projector
        self.allocator = allocator
        self.fallback = fallback or self._passive_fallback
        self.config = config

    @staticmethod
    def _passive_fallback(observation: ObservationBatch) -> Tensor:
        return observation.suspension_velocity.new_zeros(
            observation.batch_size, 4
        )

    def execute(
        self,
        observation: ObservationBatch,
        actor_output: ActorOutput,
        safety_state: SafetyState,
    ) -> SafetyDecision:
        batch = observation.batch_size
        if tuple(safety_state.sensor_valid.shape) not in ((batch,), (batch, 1)):
            raise ValueError("sensor_valid must be scalar per batch item.")
        if tuple(safety_state.hardware_healthy.shape) not in ((batch,), (batch, 1)):
            raise ValueError("hardware_healthy must be scalar per batch item.")
        healthy = (
            safety_state.sensor_valid.reshape(batch).bool()
            & safety_state.hardware_healthy.reshape(batch).bool()
        )
        fallback_force = self.fallback(observation)
        requested = torch.where(
            healthy.unsqueeze(-1), actor_output.raw_force_n, fallback_force
        )
        projected = self.projector(
            requested,
            safety_state.suspension_travel_m,
            safety_state.tire_load_n,
        )
        allocation = self.allocator.allocate(
            projected.force_n,
            observation.suspension_velocity,
            observation.actuator_force,
            observation.previous_commands,
        )
        preview_available = (
            observation.preview_confidence.mean(dim=-1)
            >= self.config.minimum_preview_confidence
        )
        return SafetyDecision(
            requested_force_n=requested,
            projected=projected,
            allocation=allocation,
            fallback_active=~healthy,
            preview_available=preview_available,
        )


@dataclass
class BandwidthReport:
    settling_time_s: float
    final_tracking_error_n: float
    settled: bool


def measure_allocator_step_response(
    allocator: DynamicActuatorAllocator,
    target_force_n: float = 2500.0,
    duration_s: float = 0.5,
    suspension_velocity_mps: float = 0.0,
    settling_band_fraction: float = 0.05,
) -> BandwidthReport:
    config = allocator.config
    steps = int(round(duration_s / config.sample_time_s))
    target = torch.full((1, 4), target_force_n)
    velocity = torch.full((1, 4), suspension_velocity_mps)
    force = torch.zeros(1, 4)
    commands = torch.zeros(1, 12)
    commands[:, :8] = config.current_bounds_a[0]
    settled_index = None
    errors = []
    band = max(abs(target_force_n) * settling_band_fraction, 1.0)

    for index in range(steps):
        result = allocator.allocate(target, velocity, force, commands)
        commands = result.commands
        force = result.predicted_force_n
        error = float((force - target).abs().max())
        errors.append(error)
        if error <= band and settled_index is None:
            settled_index = index
        if error > band:
            settled_index = None

    settling_time = (
        float("inf")
        if settled_index is None
        else (settled_index + 1) * config.sample_time_s
    )
    return BandwidthReport(
        settling_time_s=settling_time,
        final_tracking_error_n=errors[-1],
        settled=settled_index is not None,
    )


@dataclass
class PolicySafetyReport:
    finite_output: bool
    gate_in_range: bool
    dropout_matches_feedback: bool
    dropout_force_difference_n: float
    command_limits_respected: bool
    command_slew_respected: bool
    force_limits_respected: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.finite_output,
                self.gate_in_range,
                self.dropout_matches_feedback,
                self.command_limits_respected,
                self.command_slew_respected,
                self.force_limits_respected,
            )
        )


@torch.no_grad()
def verify_policy_safety(
    actor: ModularActor,
    observation: ObservationBatch,
    supervisor: SafetySupervisor,
    safety_state: SafetyState,
) -> PolicySafetyReport:
    normal_output = actor(observation)
    dropout_observation = observation.detach()
    dropout_observation.preview_confidence = torch.zeros_like(
        dropout_observation.preview_confidence
    )
    dropout_observation.road = dropout_observation.road.clone()
    dropout_observation.road[:, :, 3, :] = 0.0
    dropout_observation.road[:, :, 5, :] = 0.0
    dropout_output = actor(dropout_observation)
    feedback_output = actor(dropout_observation, feedback_only=True)
    difference = float(
        (
            dropout_output.raw_force_n - feedback_output.raw_force_n
        )
        .abs()
        .max()
    )
    decision = supervisor.execute(observation, normal_output, safety_state)
    currents, rpm = unpack_commands(decision.allocation.commands)
    previous_currents, previous_rpm = unpack_commands(
        observation.previous_commands
    )
    actuator_config = supervisor.allocator.config
    current_slew = (
        currents - previous_currents
    ).abs() / actuator_config.sample_time_s
    rpm_slew = (
        rpm - previous_rpm
    ).abs() / actuator_config.sample_time_s
    return PolicySafetyReport(
        finite_output=bool(
            torch.isfinite(normal_output.raw_force_n).all()
            and torch.isfinite(decision.allocation.commands).all()
        ),
        gate_in_range=bool(
            (normal_output.gate >= 0.0).all()
            and (normal_output.gate <= 1.0).all()
        ),
        dropout_matches_feedback=(
            difference <= supervisor.config.maximum_force_discontinuity_n
        ),
        dropout_force_difference_n=difference,
        command_limits_respected=bool(
            (currents >= actuator_config.current_bounds_a[0] - 1e-5).all()
            and (currents <= actuator_config.current_bounds_a[1] + 1e-5).all()
            and (rpm >= actuator_config.rpm_bounds[0] - 1e-5).all()
            and (rpm <= actuator_config.rpm_bounds[1] + 1e-5).all()
        ),
        command_slew_respected=bool(
            (
                current_slew
                <= actuator_config.current_slew_a_per_s + 1e-3
            ).all()
            and (rpm_slew <= actuator_config.rpm_slew_per_s + 1e-3).all()
        ),
        force_limits_respected=bool(
            (
                decision.projected.force_n
                >= supervisor.projector.force_bounds_n[0] - 1e-5
            ).all()
            and (
                decision.projected.force_n
                <= supervisor.projector.force_bounds_n[1] + 1e-5
            ).all()
        ),
    )


@dataclass
class GoNoGoReport:
    approved: bool
    checks: Dict[str, bool]


def deployment_go_no_go(
    feedback_metrics: Dict[str, float],
    residual_metrics: Dict[str, float],
    dropout_force_difference_n: float,
    policy_safety_passed: bool,
    bandwidth: BandwidthReport,
    config: SafetyConfig = SafetyConfig(),
) -> GoNoGoReport:
    checks = {
        "comfort_improved": residual_metrics["rms_vertical_acceleration"]
        < feedback_metrics["rms_vertical_acceleration"],
        "hard_violations_not_increased": residual_metrics[
            "hard_violation_count"
        ]
        <= feedback_metrics["hard_violation_count"],
        "travel_not_increased": residual_metrics["max_suspension_travel"]
        <= feedback_metrics["max_suspension_travel"],
        "dropout_is_continuous": dropout_force_difference_n
        <= config.maximum_force_discontinuity_n,
        "runtime_safety_passed": policy_safety_passed,
        "allocator_settles": bandwidth.settled
        and bandwidth.settling_time_s <= config.maximum_settling_time_s
        and bandwidth.final_tracking_error_n
        <= config.maximum_tracking_error_n,
    }
    return GoNoGoReport(approved=all(checks.values()), checks=checks)
