"""Simplified nonlinear 7-DOF full-car vertical suspension model."""

from __future__ import annotations

import numpy as np

from rl_suspension.models.types import (
    ActuatorState,
    FloatArray,
    StepResult,
    SuspensionOutput,
    SuspensionState,
    VehicleParams,
)


class SevenDofSuspensionModel:
    """Numerical 7-DOF suspension model with actuator-force inputs.

    This model is intentionally compact and explicit. It provides the training
    API and physically meaningful diagnostics while leaving room to replace the
    equations with a higher-fidelity plant later.
    """

    def __init__(self, params: VehicleParams | None = None) -> None:
        self.params = params or VehicleParams()

    def step(
        self,
        state: SuspensionState,
        action_12d: FloatArray,
        road_profile: FloatArray,
        dt: float,
        actuator_state: ActuatorState | None = None,
    ) -> StepResult:
        """Advance the plant one fixed time step.

        `action_12d` is accepted to keep the final simulator boundary aligned
        with the real controller. The current model uses `actuator_state.forces`
        as the realized corner force after the allocator/actuator dynamics.
        """

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        action = np.asarray(action_12d, dtype=np.float64)
        if action.shape != (12,):
            raise ValueError(f"action_12d must have shape (12,), got {action.shape}")

        road = np.asarray(road_profile, dtype=np.float64)
        if road.shape != (4,):
            raise ValueError(f"road_profile must have shape (4,), got {road.shape}")

        act_state = actuator_state or ActuatorState()
        acceleration, output = self._accelerations(state, road, act_state)

        # Semi-implicit Euler is stable enough for the scaffold and easy to audit.
        next_qd = state.qd + acceleration * dt
        next_q = state.q + next_qd * dt
        next_state = SuspensionState(q=next_q.astype(np.float64), qd=next_qd.astype(np.float64))

        return StepResult(next_state=next_state, output=output)

    def _accelerations(
        self,
        state: SuspensionState,
        road_heights: FloatArray,
        actuator_state: ActuatorState,
    ) -> tuple[FloatArray, SuspensionOutput]:
        p = self.params
        q = state.q
        qd = state.qd

        z_body = q[0]
        pitch = q[1]
        roll = q[2]
        z_wheel = q[3:7]
        zd_body = qd[0]
        pitch_rate = qd[1]
        roll_rate = qd[2]
        zd_wheel = qd[3:7]

        corner_body_z = z_body + p.corner_x * pitch + p.corner_y * roll
        corner_body_zd = zd_body + p.corner_x * pitch_rate + p.corner_y * roll_rate

        suspension_deflections = corner_body_z - z_wheel
        suspension_velocities = corner_body_zd - zd_wheel
        tire_deflections = road_heights - z_wheel

        spring_forces = -p.suspension_stiffness * suspension_deflections
        damping_forces = -self._nonlinear_passive_damping(suspension_velocities)
        active_forces = np.asarray(actuator_state.forces, dtype=np.float64)

        # Positive corner force acts upward on the body and downward on the wheel.
        body_corner_forces = spring_forces + damping_forces + active_forces
        tire_loads = p.tire_stiffness * tire_deflections + p.unsprung_masses * p.gravity

        body_acc = float(np.sum(body_corner_forces) / p.sprung_mass)
        pitch_acc = float(np.sum(body_corner_forces * p.corner_x) / p.pitch_inertia)
        roll_acc = float(np.sum(body_corner_forces * p.corner_y) / p.roll_inertia)
        wheel_acc = (-body_corner_forces + p.tire_stiffness * tire_deflections) / p.unsprung_masses

        acceleration = np.concatenate(
            [
                np.array([body_acc, pitch_acc, roll_acc], dtype=np.float64),
                wheel_acc.astype(np.float64),
            ]
        )

        violations = {
            "suspension_travel": float(
                np.maximum(np.abs(suspension_deflections) - p.suspension_travel_limit, 0.0).sum()
            ),
            "tire_lift": float(np.maximum(p.tire_load_min - tire_loads, 0.0).sum()),
        }

        output = SuspensionOutput(
            body_acceleration=body_acc,
            pitch_acceleration=pitch_acc,
            roll_acceleration=roll_acc,
            suspension_deflections=suspension_deflections.astype(np.float64),
            suspension_velocities=suspension_velocities.astype(np.float64),
            tire_deflections=tire_deflections.astype(np.float64),
            tire_loads=tire_loads.astype(np.float64),
            actual_corner_forces=active_forces.astype(np.float64),
            actuator_state=actuator_state.copy(),
            constraint_violations=violations,
        )
        return acceleration, output

    def _nonlinear_passive_damping(self, velocities: FloatArray) -> FloatArray:
        p = self.params
        linear = p.passive_damping * velocities
        high_speed = 250.0 * np.tanh(8.0 * velocities)
        return linear + high_speed
