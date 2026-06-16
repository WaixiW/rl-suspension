"""Linear preview-LQR MPC teacher.

Solves an infinite-horizon LQR (DARE) for the feedback gain and augments it with
a finite preview feedforward over the previewed road disturbance. The teacher
outputs *desired forces*; to act on the real (forward-only) actuator or to create
imitation labels, the desired force is mapped to commands with the offline
inverse (Path B). The missing real-time inverse is therefore never on the policy
runtime path - it only feeds the teacher.

Derivation (disturbance-preview LQR):
    x_{k+1} = Ad x_k + Bud u_k + Brd w_k          (w = [z_r; zd_r])
    minimize sum x'Qx + u'Ru
    P  : DARE solution,  M = (R + Bud'P Bud)^{-1} Bud',  K = M P Ad
    Acl = Ad - Bud K
    b_k = Acl'(P Brd w_k + b_{k+1}),  b_N = 0
    u_k = -K x_k - M (P Brd w_k + b_{k+1})
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import solve_discrete_are

from .full_car import FullCar7DOF
from .actuator import OfflineActuatorInverse
from .config import ActuatorParams, N_CORNERS


@dataclass
class MPCWeights:
    w_defl: float = 2e3      # suspension deflection
    w_zu: float = 1e3        # unsprung (tire deflection proxy)
    w_rate: float = 300.0    # body heave/pitch/roll rates (comfort proxy)
    w_u: float = 2e-7        # actuator force effort
    state_reg: float = 1e-4


class MPCTeacher:
    def __init__(
        self,
        car: FullCar7DOF,
        actuator: ActuatorParams,
        control_dt: float,
        preview_steps: int = 25,
        weights: Optional[MPCWeights] = None,
    ):
        self.car = car
        self.dt = control_dt
        self.Np = preview_steps
        self.w = weights or MPCWeights()
        self.inverse = OfflineActuatorInverse(actuator)
        self.ap = actuator
        self._design()

    def _design(self) -> None:
        Ad, Bud, Brd = self.car.discretize(self.dt)
        self.Ad, self.Bud, self.Brd = Ad, Bud, Brd
        nx, n = self.car.nx, self.car.n
        J = self.car.J

        # output selection matrices
        Cdefl = np.zeros((N_CORNERS, nx))
        for i in range(N_CORNERS):
            Cdefl[i, 0:3] = J[i]
            Cdefl[i, 3 + i] = -1.0
        Czu = np.zeros((N_CORNERS, nx))
        for i in range(N_CORNERS):
            Czu[i, 3 + i] = 1.0
        Crate = np.zeros((3, nx))
        Crate[0, n + 0] = 1.0
        Crate[1, n + 1] = 1.0
        Crate[2, n + 2] = 1.0

        w = self.w
        Q = (
            w.w_defl * Cdefl.T @ Cdefl
            + w.w_zu * Czu.T @ Czu
            + w.w_rate * Crate.T @ Crate
            + w.state_reg * np.eye(nx)
        )
        R = w.w_u * np.eye(N_CORNERS)

        P = solve_discrete_are(Ad, Bud, Q, R)
        BtPB_R = R + Bud.T @ P @ Bud
        self.M = np.linalg.solve(BtPB_R, Bud.T)     # (4, nx)
        self.K = self.M @ P @ Ad                     # (4, nx)
        self.Acl = Ad - Bud @ self.K
        self.P = P
        self.Q, self.R = Q, R

    # ------------------------------------------------------- desired force
    def desired_force(self, x: np.ndarray, road_future: np.ndarray) -> np.ndarray:
        """road_future: (Np, 8) sequence of [z_r(4), zd_r(4)] in control steps."""
        Np = min(self.Np, len(road_future))
        # backward pass for b (costate linear term)
        b = np.zeros(self.car.nx)
        PBrd_w = []
        for k in range(Np):
            PBrd_w.append(self.P @ (self.Brd @ road_future[k]))
        for k in reversed(range(Np)):
            b = self.Acl.T @ (PBrd_w[k] + b)
        w0 = road_future[0] if Np > 0 else np.zeros(8)
        u = -self.K @ x - self.M @ (self.P @ (self.Brd @ w0) + b)
        return np.clip(u, -self.ap.force_sat, self.ap.force_sat)

    def act(self, x: np.ndarray, rel_vel: np.ndarray,
            road_future: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (command (4,3), desired_force (4,)) for runtime use."""
        f_des = self.desired_force(x, road_future)
        cmd = self.inverse.invert(f_des, rel_vel)
        return cmd, f_des
