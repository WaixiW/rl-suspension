"""Fast end-to-end sanity check of the suspension stack (no RL training).

Runs each component, then a short closed-loop episode for passive / skyhook / MPC
controllers and prints comfort metrics. Exits non-zero on failure.
"""

from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_suspension.config import default_config
from active_suspension.full_car import FullCar7DOF
from active_suspension.actuator import ForwardActuator, OfflineActuatorInverse
from active_suspension.env import SuspensionEnv
from active_suspension.baselines import PassiveController, SkyhookController
from active_suspension.mpc import MPCTeacher
from active_suspension.domain_randomization import DRConfig


def test_car_stability():
    cfg = default_config()
    car = FullCar7DOF(cfg.vehicle)
    Ad, _, _ = car.discretize(cfg.sim.dt)
    eig = np.linalg.eigvals(Ad)
    assert np.all(np.abs(eig) <= 1.0 + 1e-6), f"unstable discretization: {np.abs(eig).max()}"
    # free response from a bump should decay
    x = car.initial_state()
    x[3] = 0.05  # FL unsprung displaced
    for _ in range(2000):
        x = car.step(x, np.zeros(4), np.zeros(4), np.zeros(4), cfg.sim.dt)
    assert np.linalg.norm(x) < 1e-2, f"free response did not decay: {np.linalg.norm(x)}"
    print("[ok] car discretization stable + decays")


def test_actuator_forward_only():
    cfg = default_config()
    act = ForwardActuator(cfg.actuator, cfg.sim.dt)
    cmd = np.array([[1.0, 1.0, 1000.0]] * 4)
    rel = np.array([0.2, -0.2, 0.1, -0.1])
    f0 = act.step(cmd, rel)
    for _ in range(50):
        f = act.step(cmd, rel)
    assert np.all(np.abs(f) <= cfg.actuator.force_sat + 1e-6)
    # delay: realized force lags target
    assert np.linalg.norm(f) > np.linalg.norm(f0), "force did not build up through lag"
    print(f"[ok] forward actuator with delay; steady force={np.round(f,1)}")


def test_inverse_offline():
    cfg = default_config()
    inv = OfflineActuatorInverse(cfg.actuator)
    act = ForwardActuator(cfg.actuator, cfg.sim.dt)
    rel = np.array([0.3, -0.3, 0.05, -0.05])
    f_des = np.array([800.0, -600.0, 1200.0, -200.0])
    cmd = inv.invert(f_des, rel)
    f_real = act.target_force(cmd, rel)
    err = np.abs(f_real - f_des)
    print(f"[ok] offline inverse label error (N): {np.round(err,1)}")


def run_controller(env, controller, n=None, rl_policy=None):
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    n = n or env._max_control_steps()
    for _ in range(n):
        if rl_policy is not None:
            action = rl_policy(obs)
            res = env.apply_command(ForwardActuator.normalize_action(action, env.cfg.actuator))
        else:
            inp = env.controller_inputs()
            if isinstance(controller, MPCTeacher):
                cmd, _ = controller.act(inp["x_hat"], inp["rel_vel"], inp["road_future"])
            else:
                cmd = controller.act(inp["est"])
            res = env.apply_command(cmd)
        obs = res.obs
        if res.terminated:
            break
    return env.episode_metrics()


def test_episode_controllers():
    cfg = default_config()
    env = SuspensionEnv(cfg, dr=DRConfig(enabled=False), randomize=False)
    env.set_scenario("single_bump")

    passive = PassiveController(cfg.actuator)
    skyhook = SkyhookController(cfg.actuator)
    mpc = MPCTeacher(env.car, cfg.actuator, cfg.sim.control_dt)

    m_pass = run_controller(env, passive)
    m_sky = run_controller(env, skyhook)
    # rebuild MPC on the (fixed) car after reset
    mpc = MPCTeacher(env.car, cfg.actuator, cfg.sim.control_dt)
    m_mpc = run_controller(env, mpc)

    print(f"[metrics] single_bump comfort Wk-RMS [m/s^2]:")
    print(f"    passive : {m_pass.comfort_wk_rms:.4f}  (heave {m_pass.heave_acc_rms:.3f})")
    print(f"    skyhook : {m_sky.comfort_wk_rms:.4f}  (heave {m_sky.heave_acc_rms:.3f})")
    print(f"    mpc     : {m_mpc.comfort_wk_rms:.4f}  (heave {m_mpc.heave_acc_rms:.3f})")
    assert np.isfinite(m_pass.comfort_wk_rms)
    assert m_mpc.sws_max < 0.16, "MPC violated travel grossly"
    print("[ok] closed-loop episodes ran for passive/skyhook/mpc")


def test_env_gym_api():
    cfg = default_config()
    env = SuspensionEnv(cfg, randomize=True, use_safety=True)
    obs, info = env.reset(seed=0)
    assert obs.shape[0] == env.observation_space.shape[0]
    total_r = 0.0
    interventions = 0
    for _ in range(50):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        interventions += int(info["safety_intervened"])
        if term or trunc:
            break
    print(f"[ok] gym API: obs_dim={obs.shape[0]}, random-policy return={total_r:.1f}, "
          f"safety interventions={interventions}")


if __name__ == "__main__":
    test_car_stability()
    test_actuator_forward_only()
    test_inverse_offline()
    test_episode_controllers()
    test_env_gym_api()
    print("\nALL SMOKE TESTS PASSED")
