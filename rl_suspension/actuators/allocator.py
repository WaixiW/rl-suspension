"""Deterministic corner-force allocator for the active suspension actuators."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rl_suspension.models.types import ActuatorState, FloatArray


@dataclass(frozen=True)
class ActuatorLimits:
    """Physical and numerical limits for one four-corner actuator set."""

    current_min: float = 0.0
    current_max: float = 2.0
    pump_speed_min: float = 0.0
    pump_speed_max: float = 5000.0
    current_rate_limit: float = 10.0
    pump_accel_limit: float = 12000.0
    max_force: FloatArray = field(
        default_factory=lambda: np.array([5000.0, 5000.0, 5000.0, 5000.0], dtype=np.float64)
    )
    force_time_constant: float = 0.04


@dataclass(frozen=True)
class AllocationResult:
    """Result of converting desired forces to physical commands."""

    action_12d: FloatArray
    actuator_state: ActuatorState
    desired_forces: FloatArray
    feasible_forces: FloatArray
    tracking_error: FloatArray
    saturated: FloatArray


class ActuatorAllocator:
    """Map desired corner forces to two damping currents and one pump speed.

    The allocator is deliberately simple: it clips desired force to the feasible
    range, assigns rebound/compression current according to force sign and
    relative velocity, maps force magnitude to pump speed, applies command rate
    limits, and adds first-order force lag.
    """

    def __init__(self, limits: ActuatorLimits | None = None) -> None:
        self.limits = limits or ActuatorLimits()

    def allocate(
        self,
        desired_forces: FloatArray,
        suspension_velocities: FloatArray,
        previous_state: ActuatorState,
        dt: float,
    ) -> AllocationResult:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        desired = np.asarray(desired_forces, dtype=np.float64)
        velocities = np.asarray(suspension_velocities, dtype=np.float64)
        if desired.shape != (4,):
            raise ValueError(f"desired_forces must have shape (4,), got {desired.shape}")
        if velocities.shape != (4,):
            raise ValueError(f"suspension_velocities must have shape (4,), got {velocities.shape}")

        max_force = np.asarray(self.limits.max_force, dtype=np.float64)
        feasible = np.clip(desired, -max_force, max_force)
        saturated = np.abs(feasible - desired) > 1e-9

        target_currents = self._target_currents(feasible, velocities, max_force)
        target_pumps = self._target_pump_speeds(feasible, max_force)

        currents = self._rate_limit(
            target_currents,
            previous_state.currents,
            self.limits.current_rate_limit,
            dt,
        )
        pumps = self._rate_limit(
            target_pumps,
            previous_state.pump_speeds,
            self.limits.pump_accel_limit,
            dt,
        )
        currents = np.clip(currents, self.limits.current_min, self.limits.current_max)
        pumps = np.clip(pumps, self.limits.pump_speed_min, self.limits.pump_speed_max)

        realized_target = self._forward_force(currents, pumps, velocities, max_force)
        alpha = float(np.clip(dt / max(self.limits.force_time_constant, dt), 0.0, 1.0))
        forces = previous_state.forces + alpha * (realized_target - previous_state.forces)

        action_12d = np.empty(12, dtype=np.float64)
        for corner in range(4):
            action_12d[3 * corner] = currents[2 * corner]
            action_12d[3 * corner + 1] = currents[2 * corner + 1]
            action_12d[3 * corner + 2] = pumps[corner]

        next_state = ActuatorState(
            currents=currents.astype(np.float64),
            pump_speeds=pumps.astype(np.float64),
            forces=forces.astype(np.float64),
        )
        return AllocationResult(
            action_12d=action_12d,
            actuator_state=next_state,
            desired_forces=desired,
            feasible_forces=feasible.astype(np.float64),
            tracking_error=(feasible - forces).astype(np.float64),
            saturated=saturated,
        )

    def _target_currents(
        self,
        feasible_forces: FloatArray,
        suspension_velocities: FloatArray,
        max_force: FloatArray,
    ) -> FloatArray:
        currents = np.zeros(8, dtype=np.float64)
        normalized_force = np.abs(feasible_forces) / np.maximum(max_force, 1.0)
        base_current = self.limits.current_min + normalized_force * (
            self.limits.current_max - self.limits.current_min
        )

        for corner, force in enumerate(feasible_forces):
            compression_channel = 2 * corner
            rebound_channel = compression_channel + 1
            use_compression = (force >= 0.0 and suspension_velocities[corner] >= 0.0) or (
                force < 0.0 and suspension_velocities[corner] < 0.0
            )
            if use_compression:
                currents[compression_channel] = base_current[corner]
                currents[rebound_channel] = 0.25 * base_current[corner]
            else:
                currents[compression_channel] = 0.25 * base_current[corner]
                currents[rebound_channel] = base_current[corner]
        return currents

    def _target_pump_speeds(self, feasible_forces: FloatArray, max_force: FloatArray) -> FloatArray:
        normalized_force = np.abs(feasible_forces) / np.maximum(max_force, 1.0)
        return self.limits.pump_speed_min + normalized_force * (
            self.limits.pump_speed_max - self.limits.pump_speed_min
        )

    def _forward_force(
        self,
        currents: FloatArray,
        pump_speeds: FloatArray,
        suspension_velocities: FloatArray,
        max_force: FloatArray,
    ) -> FloatArray:
        forces = np.zeros(4, dtype=np.float64)
        pump_gain = pump_speeds / max(self.limits.pump_speed_max, 1.0)
        for corner in range(4):
            c0 = currents[2 * corner]
            c1 = currents[2 * corner + 1]
            signed_current = c0 - c1
            valve_gain = signed_current / max(self.limits.current_max, 1e-6)
            velocity_gain = np.tanh(6.0 * suspension_velocities[corner])
            direction = np.sign(valve_gain + 0.2 * velocity_gain)
            if direction == 0.0:
                direction = np.sign(velocity_gain)
            forces[corner] = direction * max_force[corner] * np.clip(pump_gain[corner], 0.0, 1.0)
        return forces

    @staticmethod
    def _rate_limit(target: FloatArray, previous: FloatArray, rate_limit: float, dt: float) -> FloatArray:
        max_delta = abs(rate_limit) * dt
        return previous + np.clip(target - previous, -max_delta, max_delta)
