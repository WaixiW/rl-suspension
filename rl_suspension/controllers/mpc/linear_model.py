"""Successively linearized 18-state full-car suspension model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rl_suspension.models import SuspensionState, VehicleParams

FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True)
class LinearizedSuspensionModel:
    """Discrete affine model and output maps for one MPC solve."""

    A: FloatMatrix
    B: FloatMatrix
    E: FloatMatrix
    c: FloatMatrix
    acceleration_x: FloatMatrix
    acceleration_u: FloatMatrix
    acceleration_w: FloatMatrix
    acceleration_offset: FloatMatrix
    suspension_travel_x: FloatMatrix
    tire_variation_x: FloatMatrix
    tire_variation_w: FloatMatrix
    damping_slopes: FloatMatrix


def build_linear_model(
    state: SuspensionState,
    realized_forces: NDArray[np.floating],
    *,
    dt: float,
    force_time_constant: float,
    params: VehicleParams | None = None,
) -> LinearizedSuspensionModel:
    """Linearize passive damping and discretize with semi-implicit Euler.

    The plant applies newly realized actuator forces during the current step,
    so acceleration depends on ``F_next = (1-alpha) F + alpha u``.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    p = params or VehicleParams()
    forces = np.asarray(realized_forces, dtype=np.float64)
    if forces.shape != (4,):
        raise ValueError("realized_forces must have shape (4,)")

    h = _corner_deflection_matrix(p)
    velocity = h @ np.asarray(state.qd, dtype=np.float64)
    damping_value = p.passive_damping * velocity + 250.0 * np.tanh(8.0 * velocity)
    damping_slopes = p.passive_damping + 2000.0 / np.cosh(8.0 * velocity) ** 2
    damping_offset = damping_value - damping_slopes * velocity

    force_to_acceleration = _corner_force_to_acceleration(p)
    acceleration_q = force_to_acceleration @ (-np.diag(p.suspension_stiffness) @ h)
    acceleration_q[3:7, 3:7] -= np.diag(
        p.tire_stiffness / p.unsprung_masses
    )
    acceleration_qd = force_to_acceleration @ (-np.diag(damping_slopes) @ h)
    acceleration_road = np.zeros((7, 4), dtype=np.float64)
    acceleration_road[3:7] = np.diag(p.tire_stiffness / p.unsprung_masses)
    acceleration_affine = force_to_acceleration @ (-damping_offset)

    alpha = float(np.clip(dt / max(force_time_constant, dt), 0.0, 1.0))
    retained_force_gain = (1.0 - alpha) * force_to_acceleration
    command_gain = alpha * force_to_acceleration

    acceleration_x = np.zeros((7, 18), dtype=np.float64)
    acceleration_x[:, :7] = acceleration_q
    acceleration_x[:, 7:14] = acceleration_qd
    acceleration_x[:, 14:18] = retained_force_gain
    acceleration_u = command_gain

    a = np.zeros((18, 18), dtype=np.float64)
    b = np.zeros((18, 4), dtype=np.float64)
    e = np.zeros((18, 4), dtype=np.float64)
    c = np.zeros(18, dtype=np.float64)

    a[:7, :7] = np.eye(7) + dt * dt * acceleration_q
    a[:7, 7:14] = dt * np.eye(7) + dt * dt * acceleration_qd
    a[:7, 14:18] = dt * dt * retained_force_gain
    b[:7] = dt * dt * command_gain
    e[:7] = dt * dt * acceleration_road
    c[:7] = dt * dt * acceleration_affine

    a[7:14, :7] = dt * acceleration_q
    a[7:14, 7:14] = np.eye(7) + dt * acceleration_qd
    a[7:14, 14:18] = dt * retained_force_gain
    b[7:14] = dt * command_gain
    e[7:14] = dt * acceleration_road
    c[7:14] = dt * acceleration_affine

    a[14:18, 14:18] = (1.0 - alpha) * np.eye(4)
    b[14:18] = alpha * np.eye(4)

    travel_x = np.zeros((4, 18), dtype=np.float64)
    travel_x[:, :7] = h
    tire_x = np.zeros((4, 18), dtype=np.float64)
    tire_x[:, 3:7] = -np.diag(p.tire_stiffness)
    tire_w = np.diag(p.tire_stiffness)

    return LinearizedSuspensionModel(
        A=a,
        B=b,
        E=e,
        c=c,
        acceleration_x=acceleration_x,
        acceleration_u=acceleration_u,
        acceleration_w=acceleration_road,
        acceleration_offset=acceleration_affine,
        suspension_travel_x=travel_x,
        tire_variation_x=tire_x,
        tire_variation_w=tire_w,
        damping_slopes=damping_slopes.astype(np.float64),
    )


def augmented_state(
    state: SuspensionState,
    realized_forces: NDArray[np.floating],
) -> FloatMatrix:
    forces = np.asarray(realized_forces, dtype=np.float64)
    if forces.shape != (4,):
        raise ValueError("realized_forces must have shape (4,)")
    return np.concatenate([state.as_vector(), forces]).astype(np.float64)


def _corner_deflection_matrix(params: VehicleParams) -> FloatMatrix:
    h = np.zeros((4, 7), dtype=np.float64)
    h[:, 0] = 1.0
    h[:, 1] = params.corner_x
    h[:, 2] = params.corner_y
    h[:, 3:7] = -np.eye(4)
    return h


def _corner_force_to_acceleration(params: VehicleParams) -> FloatMatrix:
    mapping = np.zeros((7, 4), dtype=np.float64)
    mapping[0] = 1.0 / params.sprung_mass
    mapping[1] = params.corner_x / params.pitch_inertia
    mapping[2] = params.corner_y / params.roll_inertia
    mapping[3:7] = -np.diag(1.0 / params.unsprung_masses)
    return mapping
