"""State estimation and RL observation construction.

Sensors available on the car:
  * IMU on the sprung mass  -> vertical acceleration, pitch rate, roll rate
  * 4 wheel-height sensors   -> suspension deflection (z_s_i - z_u_i) per corner

The estimator is a linear Kalman filter on the 7-DOF model. It uses the *known*
realized actuator force for prediction and treats the *unknown* road input as
process noise. It never sees the simulator ground-truth state. The observation
builder then assembles the policy input from the estimate + preview + history.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import numpy as np

from .full_car import FullCar7DOF
from .config import N_CORNERS


class KalmanEstimator:
    def __init__(self, car: FullCar7DOF, dt: float,
                 meas_noise: Optional[Dict[str, float]] = None):
        self.car = car
        self.dt = dt
        self.nx = car.nx
        self.n = car.n
        Ad, Bud, _ = car.discretize(dt)
        self.Ad = Ad
        self.Bud = Bud

        mn = {"height": 1e-4, "rate": 2e-3, "acc": 3e-2}
        if meas_noise:
            mn.update(meas_noise)
        self._build_measurement(mn)

        # process noise: large on unsprung velocities (road excitation)
        q = np.full(self.nx, 1e-7)
        q[self.n + 3: self.n + 7] = 5.0       # unsprung vertical velocities
        q[3:7] = 1e-5
        self.Q = np.diag(q)
        self.reset()

    def _build_measurement(self, mn: Dict[str, float]) -> None:
        n = self.n
        rows = []
        Du = []           # feedthrough on actuator force for each measurement
        Rdiag = []
        # 4 height sensors: J[i] @ q_sprung - z_u_i
        for i in range(N_CORNERS):
            h = np.zeros(self.nx)
            h[0:3] = self.car.J[i]
            h[3 + i] = -1.0
            rows.append(h)
            Du.append(np.zeros(N_CORNERS))
            Rdiag.append(mn["height"] ** 2)
        # pitch rate (qd theta) and roll rate (qd phi)
        for idx in (n + 1, n + 2):
            h = np.zeros(self.nx)
            h[idx] = 1.0
            rows.append(h)
            Du.append(np.zeros(N_CORNERS))
            Rdiag.append(mn["rate"] ** 2)
        # heave acceleration = continuous xd[n] = A[n,:] x + Bu[n,:] F
        h = self.car.A[n, :].copy()
        rows.append(h)
        Du.append(self.car.Bu[n, :].copy())
        Rdiag.append(mn["acc"] ** 2)

        self.H = np.array(rows)
        self.Du = np.array(Du)
        self.R = np.diag(Rdiag)

    def reset(self) -> None:
        self.x = np.zeros(self.nx)
        self.P = np.eye(self.nx) * 1e-3

    def predict(self, F_act: np.ndarray) -> None:
        self.x = self.Ad @ self.x + self.Bud @ F_act
        self.P = self.Ad @ self.P @ self.Ad.T + self.Q

    def update(self, meas: np.ndarray, F_act: np.ndarray) -> None:
        y = meas - (self.H @ self.x + self.Du @ F_act)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.nx) - K @ self.H) @ self.P

    def step(self, meas: np.ndarray, F_act: np.ndarray) -> np.ndarray:
        self.predict(F_act)
        self.update(meas, F_act)
        return self.x.copy()

    # ----------------------------------------------------- decoded estimate
    def decode(self, F_act: np.ndarray) -> Dict[str, np.ndarray]:
        n = self.n
        x = self.x
        q = x[:n]
        qd = x[n:]
        J = self.car.J
        susp_defl = J @ q[:3] - q[3:7]
        corner_body_vel = J @ qd[:3]
        susp_defl_vel = corner_body_vel - qd[3:7]
        rel_vel = qd[3:7] - corner_body_vel
        xd = self.car.A @ x + self.car.Bu @ F_act
        return {
            "heave_vel": float(qd[0]),
            "pitch_angle": float(q[1]),
            "roll_angle": float(q[2]),
            "pitch_rate": float(qd[1]),
            "roll_rate": float(qd[2]),
            "heave_acc": float(xd[n]),
            "pitch_acc": float(xd[n + 1]),
            "roll_acc": float(xd[n + 2]),
            "susp_defl": susp_defl,
            "susp_defl_vel": susp_defl_vel,
            "rel_vel": rel_vel,
            "corner_body_vel": corner_body_vel,
        }


class ObservationBuilder:
    """Builds the flat observation vector for the policy."""

    def __init__(self, history_len: int, n_preview_obs: int, n_conf_obs: int,
                 speed_scale: float = 30.0):
        self.history_len = history_len
        self.n_preview = n_preview_obs
        self.n_conf = n_conf_obs
        self.speed_scale = speed_scale
        # core estimate features (see _core below)
        self.n_core = 5 + 4 * 3   # body(5) + per-corner(defl,defl_vel,rel_vel)
        self.n_action = 12
        self.reset()

    def reset(self) -> None:
        self.action_hist = deque(
            [np.zeros(self.n_action) for _ in range(self.history_len)],
            maxlen=self.history_len,
        )

    def push_action(self, action: np.ndarray) -> None:
        self.action_hist.append(np.asarray(action, float).reshape(-1))

    @property
    def dim(self) -> int:
        return (
            self.n_core
            + self.n_preview
            + self.n_conf
            + 1                       # speed
            + self.history_len * self.n_action
        )

    def _core(self, est: Dict[str, np.ndarray]) -> np.ndarray:
        body = np.array([
            est["heave_vel"], est["pitch_angle"], est["roll_angle"],
            est["pitch_rate"], est["roll_rate"],
        ])
        corner = np.concatenate([
            est["susp_defl"], est["susp_defl_vel"], est["rel_vel"]
        ])
        return np.concatenate([body, corner])

    def build(self, est: Dict[str, np.ndarray], preview_obs: np.ndarray,
              conf_obs: np.ndarray, speed: float) -> np.ndarray:
        core = self._core(est)
        hist = np.concatenate(list(self.action_hist))
        obs = np.concatenate([
            core,
            np.asarray(preview_obs, float).reshape(-1),
            np.asarray(conf_obs, float).reshape(-1),
            np.array([speed / self.speed_scale]),
            hist,
        ])
        return obs.astype(np.float32)
