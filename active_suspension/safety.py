"""Command-level safety filter and fallback state machine.

Because the policy outputs raw commands (no force inverse exists), the filter
works in command space:

  1. Box clamp every current/pump to its limits (always valid).
  2. Rate clamp the change from the previous command (anti bang-bang hardware).
  3. Forward-model predictive check: roll the full-car model forward a short
     horizon under the candidate command (held) and the previewed road; if
     suspension travel or tire-load limits would be violated, blend the command
     toward a known-safe max-dissipative command (currents max, pump off).

The FallbackSupervisor downgrades authority when preview confidence, estimator
health, or compute timing degrade: RL -> skyhook preset -> passive-safe.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

import numpy as np

from .full_car import FullCar7DOF
from .actuator import ForwardActuator
from .config import ActuatorParams, VehicleParams, N_CORNERS


class CommandSafetyFilter:
    def __init__(
        self,
        car: FullCar7DOF,
        actuator: ActuatorParams,
        vehicle: VehicleParams,
        control_dt: float,
        horizon: int = 6,
        travel_margin: float = 0.9,
        dtl_limit: float = 1.0,
        max_rate: Optional[np.ndarray] = None,
    ):
        self.car = car
        self.ap = actuator
        self.vp = vehicle
        self.dt = control_dt
        self.H = horizon
        self.travel_limit = vehicle.susp_travel_limit * travel_margin
        self.dtl_limit = dtl_limit
        self.static_loads = vehicle.static_corner_loads()
        # default per-channel max change per control step
        if max_rate is None:
            max_rate = np.array([
                actuator.i_max * 0.5, actuator.i_max * 0.5,
                actuator.n_pump_max * 0.5,
            ])
        self.max_rate = max_rate
        self._fwd = ForwardActuator(actuator, control_dt)

    # ---------------------------------------------------------------- clamps
    def _box_clamp(self, cmd: np.ndarray) -> np.ndarray:
        cmd = cmd.reshape(N_CORNERS, 3).copy()
        cmd[:, 0] = np.clip(cmd[:, 0], 0, self.ap.i_max)
        cmd[:, 1] = np.clip(cmd[:, 1], 0, self.ap.i_max)
        cmd[:, 2] = np.clip(cmd[:, 2], -self.ap.n_pump_max, self.ap.n_pump_max)
        return cmd

    def _rate_clamp(self, cmd: np.ndarray, prev: np.ndarray) -> np.ndarray:
        delta = np.clip(cmd - prev, -self.max_rate, self.max_rate)
        return prev + delta

    def safe_command(self) -> np.ndarray:
        """Max-dissipative command: currents at max, pump off."""
        c = np.zeros((N_CORNERS, 3))
        c[:, 0] = self.ap.i_max
        c[:, 1] = self.ap.i_max
        return c

    # ---------------------------------------------------- predictive rollout
    def _predict_violation(
        self,
        cmd: np.ndarray,
        x0: np.ndarray,
        road_future: Tuple[np.ndarray, np.ndarray],
    ) -> bool:
        """Roll the model forward H steps with cmd held; return True if unsafe."""
        z_r_seq, zd_r_seq = road_future
        x = x0.copy()
        n = self.car.n
        for h in range(self.H):
            qd = x[n:]
            corner_body_vel = self.car.J @ qd[:3]
            rel_vel = qd[3:7] - corner_body_vel
            f = self._fwd.target_force(cmd, rel_vel)
            z_r = z_r_seq[min(h, len(z_r_seq) - 1)]
            zd_r = zd_r_seq[min(h, len(zd_r_seq) - 1)]
            x = self.car.step(x, f, z_r, zd_r, self.dt)
            q = x[:n]
            susp_defl = self.car.J @ q[:3] - q[3:7]
            if np.any(np.abs(susp_defl) > self.travel_limit):
                return True
            tire_defl = q[3:7] - z_r
            dyn_load = self.car.p.k_t * tire_defl
            ratio = np.abs(dyn_load) / np.maximum(self.static_loads, 1e-6)
            if np.any(ratio > self.dtl_limit + 0.2):
                return True
        return False

    def filter(
        self,
        cmd: np.ndarray,
        prev_cmd: np.ndarray,
        x_hat: np.ndarray,
        road_future: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[np.ndarray, bool]:
        """Return (safe_command (4,3), intervened)."""
        cmd = self._box_clamp(np.asarray(cmd, float))
        cmd = self._rate_clamp(cmd, prev_cmd.reshape(N_CORNERS, 3))
        cmd = self._box_clamp(cmd)

        if not self._predict_violation(cmd, x_hat, road_future):
            return cmd, False

        # blend toward the safe command until predicted-safe or fully safe
        safe = self.safe_command()
        for beta in (0.34, 0.67, 1.0):
            blended = (1 - beta) * cmd + beta * safe
            blended = self._box_clamp(blended)
            if not self._predict_violation(blended, x_hat, road_future):
                return blended, True
        return safe, True


class FallbackMode(Enum):
    RL = "rl"
    RL_LIMITED = "rl_limited"
    SKYHOOK = "skyhook"
    PASSIVE = "passive"


class FallbackSupervisor:
    """Decides controller authority from runtime health signals."""

    def __init__(self, min_conf: float = 0.4):
        self.min_conf = min_conf

    def decide(
        self,
        preview_conf: float,
        estimator_ok: bool,
        compute_ok: bool,
        state_ok: bool,
    ) -> FallbackMode:
        if not state_ok or not estimator_ok:
            return FallbackMode.PASSIVE
        if not compute_ok:
            return FallbackMode.SKYHOOK
        if preview_conf < self.min_conf:
            return FallbackMode.RL_LIMITED
        return FallbackMode.RL
