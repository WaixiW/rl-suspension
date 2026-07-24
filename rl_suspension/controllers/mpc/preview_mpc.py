"""OSQP-backed full-car road-preview MPC."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from rl_suspension.controllers.mpc.linear_model import (
    augmented_state,
    build_linear_model,
)
from rl_suspension.envs.observation import OBSERVATION_SPEC, ObservationSpec
from rl_suspension.models import SuspensionState, VehicleParams
from rl_suspension.road import four_wheel_time_preview


@dataclass(frozen=True)
class MpcWeights:
    heave_acceleration: float = 8.0
    pitch_acceleration: float = 1.0
    roll_acceleration: float = 1.0
    suspension_travel: float = 2.0
    tire_load_variation: float = 100.0
    force_effort: float = 0.1
    force_rate: float = 10.0
    travel_slack: float = 1.0e5


@dataclass(frozen=True)
class PreviewMpcConfig:
    horizon: int = 40
    dt: float = 0.01
    force_time_constant: float = 0.04
    max_force: tuple[float, float, float, float] = (
        5000.0,
        5000.0,
        5000.0,
        5000.0,
    )
    force_rate_limit: float = 50_000.0
    travel_limit: float = 0.12
    osqp_max_iter: int = 10_000
    osqp_eps_abs: float = 1.0e-3
    osqp_eps_rel: float = 1.0e-3
    weights: MpcWeights = field(default_factory=MpcWeights)


@dataclass(frozen=True)
class MpcResult:
    action: NDArray[np.float32]
    desired_forces: NDArray[np.float64]
    status: str
    objective: float
    solve_time: float
    iterations: int
    predicted_violation: float
    constraint_margin: float
    fallback: bool


class PreviewMPC:
    """Parameterized convex MPC problem reused with OSQP warm starts."""

    def __init__(
        self,
        config: PreviewMpcConfig | None = None,
        params: VehicleParams | None = None,
        observation_spec: ObservationSpec = OBSERVATION_SPEC,
    ) -> None:
        self.config = config or PreviewMpcConfig()
        self.params = params or VehicleParams()
        self.spec = observation_spec
        if not 30 <= self.config.horizon <= 60:
            raise ValueError("MPC horizon must be between 30 and 60 steps")
        self.max_force = np.asarray(self.config.max_force, dtype=np.float64)
        self.last_feasible_command = np.zeros(4, dtype=np.float64)
        self._build_problem()

    @property
    def name(self) -> str:
        return "preview_mpc"

    def reset(self) -> None:
        self.last_feasible_command.fill(0.0)
        self.x.value = None
        self.u.value = None
        self.slack.value = None

    def solve(self, observation: NDArray[np.floating]) -> MpcResult:
        obs = self.spec.validate(np.asarray(observation))
        state = SuspensionState.from_vector(obs[self.spec.state])
        realized_forces = (
            np.asarray(obs[self.spec.actual_forces], dtype=np.float64) * self.max_force
        )
        previous_command = (
            np.asarray(obs[self.spec.previous_forces], dtype=np.float64) * self.max_force
        )
        road = four_wheel_time_preview(
            obs,
            horizon=self.config.horizon,
            dt=self.config.dt,
            spec=self.spec,
        )
        model = build_linear_model(
            state,
            realized_forces,
            dt=self.config.dt,
            force_time_constant=self.config.force_time_constant,
            params=self.params,
        )
        self._assign_parameters(
            augmented_state(state, realized_forces),
            previous_command,
            road,
            model,
        )

        started = time.perf_counter()
        try:
            self.problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                max_iter=self.config.osqp_max_iter,
                eps_abs=self.config.osqp_eps_abs,
                eps_rel=self.config.osqp_eps_rel,
                polishing=True,
            )
            solve_time = time.perf_counter() - started
        except (cp.SolverError, ValueError):
            solve_time = time.perf_counter() - started
            return self._fallback_result("solver_error", solve_time)

        status = str(self.problem.status)
        success = status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        if not success or self.u.value is None or not np.all(np.isfinite(self.u.value[0])):
            return self._fallback_result(status, solve_time)

        normalized_command = np.clip(
            np.asarray(self.u.value[0], dtype=np.float64),
            -1.0,
            1.0,
        )
        command = np.clip(
            normalized_command * self.max_force,
            -self.max_force,
            self.max_force,
        )
        self.last_feasible_command = command.copy()
        slack_value = (
            np.asarray(self.slack.value, dtype=np.float64)
            if self.slack.value is not None
            else np.zeros((self.config.horizon, 4), dtype=np.float64)
        )
        predicted_violation = float(np.max(np.maximum(slack_value, 0.0)))
        constraint_margin = self._constraint_margin(previous_command)
        stats = self.problem.solver_stats
        iterations = int(getattr(stats, "num_iters", 0) or 0)
        objective = float(self.problem.value)
        return MpcResult(
            action=(command / self.max_force).astype(np.float32),
            desired_forces=command,
            status=status,
            objective=objective,
            solve_time=solve_time,
            iterations=iterations,
            predicted_violation=predicted_violation,
            constraint_margin=constraint_margin,
            fallback=False,
        )

    def config_dict(self) -> dict:
        return asdict(self.config)

    def _build_problem(self) -> None:
        n = self.config.horizon
        self.x = cp.Variable((n + 1, 18), name="state")
        self.u = cp.Variable((n, 4), name="force_command")
        self.slack = cp.Variable((n, 4), nonneg=True, name="travel_slack")

        self.p_x0 = cp.Parameter(18)
        self.p_previous_u = cp.Parameter(4)
        self.p_road = cp.Parameter((n, 4))
        self.p_A = cp.Parameter((18, 18))
        self.p_B = cp.Parameter((18, 4))
        self.p_c = cp.Parameter(18)

        nominal_model = build_linear_model(
            SuspensionState.zeros(),
            np.zeros(4, dtype=np.float64),
            dt=self.config.dt,
            force_time_constant=self.config.force_time_constant,
            params=self.params,
        )
        state_scale = np.ones(18, dtype=np.float64)
        state_scale[14:18] = self.max_force
        self.scaled_road_matrix = (
            (1.0 / state_scale)[:, None] * nominal_model.E
        )

        h = np.zeros((4, 18), dtype=np.float64)
        h[:, 0] = 1.0
        h[:, 1] = self.params.corner_x
        h[:, 2] = self.params.corner_y
        h[:, 3:7] = -np.eye(4)
        tire_x = np.zeros((4, 18), dtype=np.float64)
        tire_x[:, 3:7] = -np.diag(self.params.tire_stiffness)
        tire_w = np.diag(self.params.tire_stiffness)

        weights = self.config.weights
        constraints: list[cp.Constraint] = [self.x[0] == self.p_x0]
        objective = 0.0
        previous_u = self.p_previous_u
        max_delta = (
            self.config.force_rate_limit * self.config.dt / self.max_force
        )

        for k in range(n):
            acceleration = (
                self.x[k + 1, 7:14] - self.x[k, 7:14]
            ) / self.config.dt
            travel = h @ self.x[k]
            tire_variation = tire_x @ self.x[k] + tire_w @ self.p_road[k]
            delta_u = self.u[k] - previous_u
            objective += weights.heave_acceleration * cp.square(acceleration[0])
            objective += weights.pitch_acceleration * cp.square(acceleration[1])
            objective += weights.roll_acceleration * cp.square(acceleration[2])
            objective += weights.suspension_travel * cp.sum_squares(
                travel / self.config.travel_limit
            )
            objective += weights.tire_load_variation * cp.sum_squares(
                tire_variation / 5000.0
            )
            objective += weights.force_effort * cp.sum_squares(
                self.u[k]
            )
            objective += weights.force_rate * cp.sum_squares(
                delta_u
            )
            objective += weights.travel_slack * cp.sum_squares(
                self.slack[k]
            )

            constraints.extend(
                [
                    self.x[k + 1]
                    == self.p_A @ self.x[k]
                    + self.p_B @ self.u[k]
                    + self.scaled_road_matrix @ self.p_road[k]
                    + self.p_c,
                    self.u[k] <= 1.0,
                    self.u[k] >= -1.0,
                    delta_u <= max_delta,
                    delta_u >= -max_delta,
                    h @ self.x[k + 1]
                    <= self.config.travel_limit * (1.0 + self.slack[k]),
                    h @ self.x[k + 1]
                    >= -self.config.travel_limit * (1.0 + self.slack[k]),
                ]
            )
            previous_u = self.u[k]

        self.problem = cp.Problem(cp.Minimize(objective), constraints)

    def _assign_parameters(self, x0, previous_u, road, model) -> None:
        state_scale = np.ones(18, dtype=np.float64)
        state_scale[14:18] = self.max_force
        inverse_scale = 1.0 / state_scale
        self.p_x0.value = x0 * inverse_scale
        self.p_previous_u.value = np.clip(
            previous_u / self.max_force,
            -1.0,
            1.0,
        )
        self.p_road.value = road
        self.p_A.value = inverse_scale[:, None] * model.A * state_scale[None, :]
        self.p_B.value = inverse_scale[:, None] * model.B * self.max_force[None, :]
        self.p_c.value = inverse_scale * model.c

    def _constraint_margin(self, previous_command: NDArray[np.float64]) -> float:
        if self.u.value is None or self.x.value is None:
            return float("-inf")
        force_margin = float(np.min(1.0 - np.abs(self.u.value)))
        previous_normalized = previous_command / self.max_force
        commands = np.vstack([previous_normalized, self.u.value])
        max_delta = self.config.force_rate_limit * self.config.dt / self.max_force
        rate_margin = float(
            np.min(max_delta - np.abs(np.diff(commands, axis=0)))
        )
        h = np.zeros((4, 18), dtype=np.float64)
        h[:, 0] = 1.0
        h[:, 1] = self.params.corner_x
        h[:, 2] = self.params.corner_y
        h[:, 3:7] = -np.eye(4)
        travel_margin = float(
            np.min(self.config.travel_limit - np.abs(self.x.value[1:] @ h.T))
        )
        return min(
            force_margin,
            rate_margin / float(np.max(max_delta)),
            travel_margin / self.config.travel_limit,
        )

    def _fallback_result(self, status: str, solve_time: float) -> MpcResult:
        command = np.clip(
            self.last_feasible_command,
            -self.max_force,
            self.max_force,
        )
        return MpcResult(
            action=(command / self.max_force).astype(np.float32),
            desired_forces=command.copy(),
            status=status,
            objective=float("inf"),
            solve_time=solve_time,
            iterations=0,
            predicted_violation=float("inf"),
            constraint_margin=float("-inf"),
            fallback=True,
        )
