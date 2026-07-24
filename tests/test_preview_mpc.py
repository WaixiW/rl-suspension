import cvxpy as cp
import numpy as np

from rl_suspension.controllers.mpc import PreviewMPC, PreviewMpcConfig
from rl_suspension.envs import ActiveSuspensionEnv, EnvConfig
from rl_suspension.road import ScenarioConfig


def make_observation(kind="single_bump"):
    env = ActiveSuspensionEnv(
        EnvConfig(
            randomize_scenario=False,
            terminate_on_violation=False,
            scenario=ScenarioConfig(
                kind=kind,
                speed=12.0,
                bump_start=1.0,
                bump_width=0.5,
            ),
        )
    )
    observation, _ = env.reset(seed=0)
    return env, observation


def test_zero_road_equilibrium_and_qp_constraints():
    env, observation = make_observation(kind="flat")
    controller = PreviewMPC(PreviewMpcConfig(horizon=30))

    result = controller.solve(observation)

    assert result.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    assert not result.fallback
    np.testing.assert_allclose(result.action, 0.0, atol=1e-5)
    assert np.max(np.abs(controller.u.value)) <= 1.0 + 1e-5
    assert np.max(np.abs(np.diff(controller.u.value, axis=0))) <= 0.1 + 1e-5
    env.close()


def test_warm_start_and_solver_diagnostics():
    env, observation = make_observation()
    controller = PreviewMPC(PreviewMpcConfig(horizon=30))

    first = controller.solve(observation)
    second = controller.solve(observation)

    assert not first.fallback
    assert not second.fallback
    assert first.iterations > 0
    assert second.iterations > 0
    assert np.isfinite(second.objective)
    assert second.predicted_violation >= 0.0
    assert np.all(np.abs(second.action) <= 1.0 + 1e-6)
    env.close()


def test_solver_failure_returns_previous_feasible_command(monkeypatch):
    env, observation = make_observation()
    controller = PreviewMPC(PreviewMpcConfig(horizon=30))
    controller.last_feasible_command = np.array([100.0, -200.0, 300.0, -400.0])

    def fail_solve(*args, **kwargs):
        raise cp.SolverError("forced test failure")

    monkeypatch.setattr(cp.Problem, "solve", fail_solve)
    result = controller.solve(observation)

    assert result.fallback
    np.testing.assert_allclose(
        result.desired_forces,
        [100.0, -200.0, 300.0, -400.0],
    )
    assert not np.all(result.action == 0.0)
    env.close()
