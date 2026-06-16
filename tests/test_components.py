"""Lightweight component tests (run with pytest or directly).

These guard the core physics/estimation/controller invariants so future changes
don't silently break the stack. Run:  python tests/test_components.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_suspension.config import default_config
from active_suspension.full_car import FullCar7DOF
from active_suspension.actuator import ForwardActuator, OfflineActuatorInverse
from active_suspension.env import SuspensionEnv
from active_suspension.estimator import KalmanEstimator
from active_suspension.baselines import PassiveController
from active_suspension.mpc import MPCTeacher
from active_suspension.evaluation import rollout
from active_suspension.domain_randomization import DRConfig


def test_discretization_stable():
    cfg = default_config()
    car = FullCar7DOF(cfg.vehicle)
    Ad, _, _ = car.discretize(cfg.sim.dt)
    assert np.max(np.abs(np.linalg.eigvals(Ad))) <= 1.0 + 1e-6


def test_actuator_is_forward_and_dissipative_sign():
    cfg = default_config()
    act = ForwardActuator(cfg.actuator, cfg.sim.dt)
    # pure positive rel_vel with zero current -> positive (dissipative) force
    f = act.target_force(np.zeros((4, 3)), np.array([0.5, 0.5, 0.5, 0.5]))
    assert np.all(f > 0), "base damping must produce force in +rel_vel direction"


def test_action_neutral_is_passive():
    cfg = default_config()
    cmd0 = ForwardActuator.normalize_action(np.zeros(12), cfg.actuator)
    assert np.allclose(cmd0, 0.0), "neutral action must map to zero command (passive)"


def test_offline_inverse_recovers_force():
    cfg = default_config()
    inv = OfflineActuatorInverse(cfg.actuator)
    act = ForwardActuator(cfg.actuator, cfg.sim.dt)
    rel = np.array([0.3, -0.2, 0.1, -0.4])
    f_des = np.array([700.0, -500.0, 1100.0, -900.0])
    cmd = inv.invert(f_des, rel)
    f = act.target_force(cmd, rel)
    assert np.max(np.abs(f - f_des)) < 50.0


def test_estimator_tracks_truth():
    cfg = default_config()
    env = SuspensionEnv(cfg, dr=DRConfig(enabled=False), randomize=False)
    env.set_scenario("single_bump")
    env.reset()
    passive = PassiveController(cfg.actuator)
    for _ in range(200):
        inp = env.controller_inputs()
        env.apply_command(passive.act(inp["est"]))
    # estimated suspension deflection should track the true state reasonably
    est = env.estimator.decode(env.F_act)
    n = env.car.n
    true_defl = env.car.J @ env.x_true[:3] - env.x_true[3:7]
    err = np.max(np.abs(est["susp_defl"] - true_defl))
    assert err < 5e-3, f"estimator deflection error too large: {err}"


def test_mpc_beats_passive_on_random_road():
    cfg = default_config()
    env = SuspensionEnv(cfg, dr=DRConfig(enabled=False), randomize=False)
    env.set_scenario("iso_C")
    m_pass, _ = rollout(env, "passive", PassiveController(cfg.actuator))
    env2 = SuspensionEnv(cfg, dr=DRConfig(enabled=False), randomize=False)
    env2.set_scenario("iso_C")
    m_mpc, _ = rollout(env2, "mpc", None)
    assert m_mpc.comfort_wk_rms < m_pass.comfort_wk_rms


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"[ok] {fn.__name__}")
    print("\nALL COMPONENT TESTS PASSED")
