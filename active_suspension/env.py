"""Gymnasium environment for RL-first active suspension control.

Action: (12,) in [-1, 1] -> raw commands (2 currents + pump speed) x 4 corners.
The forward-only actuator model lives inside the environment; the policy never
needs an inverse.

The environment exposes a unified low-level interface so that RL, baselines and
the MPC teacher all drive the same physics:
    inputs = env.controller_inputs()     # decoded estimate + previewed road
    result = env.apply_command(cmd)       # advance one control step
RL's step(action) simply normalizes the action and calls apply_command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except Exception:  # pragma: no cover
    _HAS_GYM = False
    gym = object  # type: ignore

from .config import FullConfig, default_config, N_CORNERS
from .full_car import FullCar7DOF
from .actuator import ForwardActuator
from .preview import PreviewSensor
from .estimator import KalmanEstimator, ObservationBuilder
from .reward import RewardFunction, RewardWeights
from .metrics import MetricRecorder
from .safety import CommandSafetyFilter
from .domain_randomization import DRConfig, sample_config
from .road import RoadField, make_scenario, scenario_library


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict


_Base = gym.Env if _HAS_GYM else object


class SuspensionEnv(_Base):
    metadata = {"render_modes": []}

    def __init__(
        self,
        base_config: Optional[FullConfig] = None,
        dr: Optional[DRConfig] = None,
        reward_weights: Optional[RewardWeights] = None,
        use_safety: bool = False,
        scenarios: Optional[Tuple[str, ...]] = None,
        randomize: bool = True,
        seed: int = 0,
    ):
        super().__init__() if _HAS_GYM else None
        self.base_cfg = base_config or default_config()
        self.dr = dr or DRConfig(enabled=randomize)
        self.reward_weights = reward_weights or RewardWeights()
        self.use_safety = use_safety
        self.randomize = randomize
        self.scenario_names = scenarios or (
            "single_bump", "double_bump", "asymmetric_bump",
            "pothole", "speed_bump", "iso_C",
        )
        self._rng = np.random.default_rng(seed)
        self._fixed_scenario: Optional[str] = None

        # Build once with nominal config to size spaces (dims are config-stable).
        self._build(self.base_cfg.copy(), make_scenario("flat"))

        if _HAS_GYM:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(self.obs_builder.dim,), dtype=np.float32
            )

    # ------------------------------------------------------------------ build
    def _build(self, cfg: FullConfig, road: RoadField) -> None:
        self.cfg = cfg
        self.car = FullCar7DOF(cfg.vehicle)
        self.actuator = ForwardActuator(cfg.actuator, cfg.sim.dt)
        self.preview = PreviewSensor(
            cfg.preview, cfg.vehicle, road, cfg.sim.speed,
            seed=int(self._rng.integers(0, 2**31 - 1)),
        )
        self.estimator = KalmanEstimator(self.car, cfg.sim.control_dt)
        if not hasattr(self, "obs_builder"):
            self.obs_builder = ObservationBuilder(
                cfg.sim.history_len,
                self.preview.n_preview_obs,
                self.preview.n_conf_obs,
            )
        self.reward_fn = RewardFunction(self.reward_weights, cfg.vehicle, cfg.actuator)
        self.recorder = MetricRecorder(
            1.0 / cfg.sim.control_dt,
            cfg.vehicle.static_corner_loads(),
            cfg.vehicle.susp_travel_limit,
        )
        self.safety = CommandSafetyFilter(
            self.car, cfg.actuator, cfg.vehicle, cfg.sim.control_dt
        ) if self.use_safety else None

        self.x_true = self.car.initial_state()
        self.F_act = np.zeros(N_CORNERS)
        self.prev_cmd = np.zeros((N_CORNERS, 3))
        self.prev_action = np.zeros(12)
        self._step_count = 0
        self._safety_flag = False

    # --------------------------------------------------------------- scenario
    def set_scenario(self, name: Optional[str]) -> None:
        self._fixed_scenario = name

    def _pick_scenario(self) -> RoadField:
        if self._fixed_scenario is not None:
            return make_scenario(self._fixed_scenario)
        name = self.scenario_names[int(self._rng.integers(len(self.scenario_names)))]
        return make_scenario(name)

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = (
            sample_config(self.base_cfg, self.dr, self._rng)
            if self.randomize else self.base_cfg.copy()
        )
        road = self._pick_scenario()
        self._build(cfg, road)
        self.preview.reset(s0=0.0)
        self.estimator.reset()
        self.recorder.reset()
        self.obs_builder.reset()
        obs = self._make_obs()
        info = {"scenario": road.name, "speed": cfg.sim.speed}
        if _HAS_GYM:
            return obs, info
        return obs

    # ------------------------------------------------------ controller inputs
    def _road_future(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Clean previewed road over the next n control steps (privileged path).

        Used by the MPC teacher and the safety filter. Returns (z_r_seq (n,4),
        zd_r_seq (n,4), w_seq (n,8)).
        """
        dt = self.cfg.sim.control_dt
        v = self.cfg.sim.speed
        pos = self.cfg.vehicle.corner_positions()
        cx, cy = pos[:, 0], pos[:, 1]
        z_seq = np.zeros((n_steps, 4))
        zd_seq = np.zeros((n_steps, 4))
        for h in range(n_steps):
            s = self.preview.s + v * dt * h
            xw = s + cx
            z_seq[h] = self.preview.road.height(xw, cy)
            zd_seq[h] = v * self.preview.road.height_dx(xw, cy)
        w_seq = np.concatenate([z_seq, zd_seq], axis=1)
        return z_seq, zd_seq, w_seq

    def controller_inputs(self) -> dict:
        est = self.estimator.decode(self.F_act)
        z_seq, zd_seq, w_seq = self._road_future(max(self.safety.H if self.safety else 6,
                                                     30))
        return {
            "x_hat": self.estimator.x.copy(),
            "est": est,
            "rel_vel": est["rel_vel"],
            "corner_body_vel": est["corner_body_vel"],
            "road_future": w_seq,
            "road_future_zr": z_seq,
            "road_future_zdr": zd_seq,
            "preview_conf": float(np.mean(self.preview.confidence)),
            "prev_cmd": self.prev_cmd.copy(),
        }

    # -------------------------------------------------------- apply a command
    def apply_command(self, cmd: np.ndarray) -> StepResult:
        cmd = np.asarray(cmd, float).reshape(N_CORNERS, 3)
        intervened = False
        if self.safety is not None:
            z_seq, zd_seq, _ = self._road_future(self.safety.H)
            cmd, intervened = self.safety.filter(
                cmd, self.prev_cmd, self.estimator.x.copy(), (z_seq, zd_seq)
            )

        dt = self.cfg.sim.dt
        last_out = None
        for _ in range(self.cfg.sim.control_decimation):
            z_r, zd_r = self.preview.true_road()
            n = self.car.n
            qd = self.x_true[n:]
            rel_vel = qd[3:7] - self.car.J @ qd[:3]
            self.F_act = self.actuator.step(cmd, rel_vel)
            self.x_true = self.car.step(self.x_true, self.F_act, z_r, zd_r, dt)
            self.preview.advance(dt)
            last_out = self.car.outputs(self.x_true, self.F_act, z_r, zd_r)

        # --- synthetic sensor measurements + estimator update
        meas = self._measure(last_out)
        self.estimator.step(meas, self.F_act)

        # --- reward (from ground truth) + metrics
        action_norm = ForwardActuator.denormalize_command(cmd, self.cfg.actuator)
        cmd_rate_norm = action_norm - self.prev_action
        rinfo = self.reward_fn(
            heave_acc=last_out.heave_acc,
            pitch_acc=last_out.pitch_acc,
            roll_acc=last_out.roll_acc,
            susp_defl=last_out.susp_defl,
            dyn_tire_load=last_out.dyn_tire_load,
            actuator_force=self.F_act,
            rel_vel=last_out.rel_vel,
            command_rate_norm=cmd_rate_norm,
            safety_intervened=intervened,
        )
        self.recorder.record(
            heave_acc=last_out.heave_acc,
            pitch_acc=last_out.pitch_acc,
            roll_acc=last_out.roll_acc,
            susp_defl=last_out.susp_defl,
            dyn_tire_load=last_out.dyn_tire_load,
            actuator_force=self.F_act,
            rel_vel=last_out.rel_vel,
            command=action_norm,
            safety_intervened=intervened,
        )

        self.prev_cmd = cmd
        self.prev_action = action_norm
        self.obs_builder.push_action(action_norm)
        self._step_count += 1
        self._safety_flag = intervened

        terminated = self._check_failure(last_out)
        truncated = self._step_count >= self._max_control_steps()
        reward = rinfo.reward - (50.0 if terminated else 0.0)
        obs = self._make_obs()
        info = {
            "reward_terms": rinfo.__dict__,
            "safety_intervened": intervened,
            "outputs": last_out,
        }
        return StepResult(obs, reward, terminated, truncated, info)

    # ------------------------------------------------------------- gym step
    def step(self, action: np.ndarray):
        cmd = ForwardActuator.normalize_action(action, self.cfg.actuator)
        res = self.apply_command(cmd)
        if _HAS_GYM:
            return res.obs, res.reward, res.terminated, res.truncated, res.info
        return res

    # --------------------------------------------------------------- helpers
    def _measure(self, out) -> np.ndarray:
        mn_h, mn_r, mn_a = 1e-4, 2e-3, 3e-2
        height = out.susp_defl + self._rng.normal(0, mn_h, size=4)
        pitch_rate = self.x_true[self.car.n + 1] + self._rng.normal(0, mn_r)
        roll_rate = self.x_true[self.car.n + 2] + self._rng.normal(0, mn_r)
        heave_acc = out.heave_acc + self._rng.normal(0, mn_a)
        return np.concatenate([height, [pitch_rate, roll_rate, heave_acc]])

    def _make_obs(self) -> np.ndarray:
        est = self.estimator.decode(self.F_act)
        preview_obs = self.preview.preview_obs()
        conf_obs = self.preview.confidence_obs()
        return self.obs_builder.build(est, preview_obs, conf_obs, self.cfg.sim.speed)

    def _check_failure(self, out) -> bool:
        if not np.all(np.isfinite(self.x_true)):
            return True
        if np.any(np.abs(out.susp_defl) > 1.6 * self.cfg.vehicle.susp_travel_limit):
            return True
        if abs(out.heave_acc) > 80.0:
            return True
        return False

    def _max_control_steps(self) -> int:
        return int(round(self.cfg.sim.episode_time / self.cfg.sim.control_dt))

    def episode_metrics(self):
        return self.recorder.compute()
