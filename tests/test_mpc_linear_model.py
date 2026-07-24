import numpy as np
import pytest

from rl_suspension.controllers.mpc.linear_model import (
    augmented_state,
    build_linear_model,
)
from rl_suspension.models import (
    ActuatorState,
    SevenDofSuspensionModel,
    SuspensionState,
    VehicleParams,
)


@pytest.mark.parametrize("perturbed", [False, True])
def test_affine_model_matches_nonlinear_step_at_linearization_point(perturbed):
    rng = np.random.default_rng(3)
    state = SuspensionState.zeros()
    forces = np.zeros(4)
    command = np.zeros(4)
    road = np.zeros(4)
    if perturbed:
        state.q = rng.normal(0.0, 0.01, 7)
        state.qd = rng.normal(0.0, 0.08, 7)
        forces = rng.normal(0.0, 200.0, 4)
        command = rng.normal(0.0, 300.0, 4)
        road = rng.normal(0.0, 0.005, 4)

    dt = 0.01
    time_constant = 0.04
    alpha = dt / time_constant
    realized_next = forces + alpha * (command - forces)
    params = VehicleParams()
    nonlinear = SevenDofSuspensionModel(params).step(
        state,
        np.zeros(12),
        road,
        dt,
        ActuatorState(forces=realized_next),
    )
    expected = augmented_state(nonlinear.next_state, realized_next)

    local = build_linear_model(
        state,
        forces,
        dt=dt,
        force_time_constant=time_constant,
        params=params,
    )
    predicted = (
        local.A @ augmented_state(state, forces)
        + local.B @ command
        + local.E @ road
        + local.c
    )

    np.testing.assert_allclose(predicted, expected, atol=1e-10, rtol=1e-9)


def test_linear_model_dimensions_and_finite_difference():
    rng = np.random.default_rng(7)
    state = SuspensionState(
        q=rng.normal(0.0, 0.005, 7),
        qd=rng.normal(0.0, 0.05, 7),
    )
    forces = rng.normal(0.0, 100.0, 4)
    model = build_linear_model(
        state,
        forces,
        dt=0.01,
        force_time_constant=0.04,
    )

    assert model.A.shape == (18, 18)
    assert model.B.shape == (18, 4)
    assert model.E.shape == (18, 4)
    assert model.acceleration_x.shape == (7, 18)
    assert np.all(model.damping_slopes > 0.0)

    plant = SevenDofSuspensionModel()
    x0 = augmented_state(state, forces)
    u0 = rng.normal(0.0, 200.0, 4)
    w0 = rng.normal(0.0, 0.003, 4)

    def nonlinear_map(x, u, road):
        local_state = SuspensionState.from_vector(x[:14])
        next_force = x[14:] + 0.25 * (u - x[14:])
        result = plant.step(
            local_state,
            np.zeros(12),
            road,
            0.01,
            ActuatorState(forces=next_force),
        )
        return augmented_state(result.next_state, next_force)

    epsilon = 1.0e-6
    for column in range(18):
        delta = np.zeros(18)
        delta[column] = epsilon
        numerical = (
            nonlinear_map(x0 + delta, u0, w0)
            - nonlinear_map(x0 - delta, u0, w0)
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(numerical, model.A[:, column], atol=2e-5, rtol=2e-4)
