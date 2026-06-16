"""7-DOF full-car suspension model.

Degrees of freedom (generalized coordinates q):
    q = [z_s, theta (pitch), phi (roll), z_u_FL, z_u_FR, z_u_RL, z_u_RR]
State x = [q, qdot]  (14 states).

Sign conventions: z up, x forward, y left.
    Sprung corner displacement:  z_s_i = z_s - x_i * theta + y_i * phi
    (+theta lowers the front, +phi lifts the left side.)

Inputs:
    F_act  : (4,) active actuator forces, positive pushes the body up at the corner
    z_r    : (4,) road height under each tire
    zd_r   : (4,) road vertical velocity under each tire

The model is linear, so we assemble continuous-time A, Bu, Br once and integrate
with an exact zero-order-hold discretization for stability at 1 kHz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.linalg import expm

from .config import VehicleParams, GRAVITY, N_CORNERS


@dataclass
class CarOutputs:
    """Decoded physical outputs at one instant."""

    heave_acc: float            # body vertical acceleration at CG [m/s^2]
    pitch_acc: float            # [rad/s^2]
    roll_acc: float             # [rad/s^2]
    corner_acc: np.ndarray      # (4,) sprung-mass vertical acc at each corner
    susp_defl: np.ndarray       # (4,) z_s_i - z_u_i  (suspension working space) [m]
    susp_defl_vel: np.ndarray   # (4,) d/dt of above [m/s]
    tire_defl: np.ndarray       # (4,) z_u_i - z_r_i [m]
    dyn_tire_load: np.ndarray   # (4,) dynamic tire load [N] (excludes static)
    rel_vel: np.ndarray         # (4,) suspension relative velocity zd_u - zd_s [m/s]


class FullCar7DOF:
    def __init__(self, params: VehicleParams):
        self.p = params
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        p = self.p
        pos = p.corner_positions()        # (4,2)
        x_i = pos[:, 0]
        y_i = pos[:, 1]
        ks = p.k_s()
        cs = p.c_s()
        kt = p.k_t
        ct = p.c_t

        # Jacobian rows mapping corner sprung displacement to generalized coords:
        #   z_s_i = J_i @ q_sprung,  q_sprung = [z_s, theta, phi]
        #   J_i = [1, -x_i, y_i]
        J = np.stack([np.ones(N_CORNERS), -x_i, y_i], axis=1)  # (4,3)

        n = 7
        M = np.zeros((n, n))
        M[0, 0] = p.m_s
        M[1, 1] = p.I_pitch
        M[2, 2] = p.I_roll
        for i in range(N_CORNERS):
            M[3 + i, 3 + i] = p.m_u

        K = np.zeros((n, n))
        C = np.zeros((n, n))
        # Suspension couples sprung (J) and each unsprung.
        for i in range(N_CORNERS):
            Ji = J[i]                       # (3,)
            ui = 3 + i
            # Sprung-sprung block
            K[:3, :3] += ks[i] * np.outer(Ji, Ji)
            C[:3, :3] += cs[i] * np.outer(Ji, Ji)
            # Sprung-unsprung coupling
            K[:3, ui] += -ks[i] * Ji
            K[ui, :3] += -ks[i] * Ji
            C[:3, ui] += -cs[i] * Ji
            C[ui, :3] += -cs[i] * Ji
            # Unsprung-unsprung
            K[ui, ui] += ks[i] + kt
            C[ui, ui] += cs[i] + ct

        # Input maps -------------------------------------------------------
        # Generalized force from actuator F_act_i: Q = sum_i (dz_s_i/dq) F_i for
        # sprung dofs, and -F_i on the matching unsprung dof (reaction).
        Bu_q = np.zeros((n, N_CORNERS))
        for i in range(N_CORNERS):
            Bu_q[:3, i] += J[i]
            Bu_q[3 + i, i] += -1.0

        # Road enters through the tire on the unsprung masses.
        Br_zr = np.zeros((n, N_CORNERS))
        Br_zdr = np.zeros((n, N_CORNERS))
        for i in range(N_CORNERS):
            Br_zr[3 + i, i] = kt
            Br_zdr[3 + i, i] = ct

        Minv = np.linalg.inv(M)

        # Continuous state-space: x = [q, qd]
        A = np.zeros((2 * n, 2 * n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -Minv @ K
        A[n:, n:] = -Minv @ C

        Bu = np.zeros((2 * n, N_CORNERS))
        Bu[n:, :] = Minv @ Bu_q

        Bzr = np.zeros((2 * n, N_CORNERS))
        Bzr[n:, :] = Minv @ Br_zr
        Bzdr = np.zeros((2 * n, N_CORNERS))
        Bzdr[n:, :] = Minv @ Br_zdr

        self.n = n
        self.nx = 2 * n
        self.J = J
        self.M, self.K, self.C, self.Minv = M, K, C, Minv
        self.A, self.Bu, self.Bzr, self.Bzdr = A, Bu, Bzr, Bzdr
        # Combined road input B for [z_r; zd_r]
        self.Br = np.concatenate([Bzr, Bzdr], axis=1)  # (nx, 8)
        self._disc_cache: dict = {}

    # --------------------------------------------------------------- discretize
    def discretize(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact ZOH discretization, cached by dt. Returns (Ad, Bud, Brd)."""
        key = round(dt, 9)
        if key in self._disc_cache:
            return self._disc_cache[key]
        nx = self.nx
        nin = self.Bu.shape[1] + self.Br.shape[1]
        B = np.concatenate([self.Bu, self.Br], axis=1)
        Maug = np.zeros((nx + nin, nx + nin))
        Maug[:nx, :nx] = self.A
        Maug[:nx, nx:] = B
        Md = expm(Maug * dt)
        Ad = Md[:nx, :nx]
        Bd = Md[:nx, nx:]
        Bud = Bd[:, : self.Bu.shape[1]]
        Brd = Bd[:, self.Bu.shape[1]:]
        self._disc_cache[key] = (Ad, Bud, Brd)
        return Ad, Bud, Brd

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.nx)

    # --------------------------------------------------------------------- step
    def step(
        self,
        x: np.ndarray,
        F_act: np.ndarray,
        z_r: np.ndarray,
        zd_r: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Advance one dt with ZOH on (F_act, z_r, zd_r)."""
        Ad, Bud, Brd = self.discretize(dt)
        road = np.concatenate([z_r, zd_r])
        return Ad @ x + Bud @ F_act + Brd @ road

    # ------------------------------------------------------------------ outputs
    def state_deriv(
        self, x: np.ndarray, F_act: np.ndarray, z_r: np.ndarray, zd_r: np.ndarray
    ) -> np.ndarray:
        road = np.concatenate([z_r, zd_r])
        return self.A @ x + self.Bu @ F_act + self.Br @ road

    def outputs(
        self,
        x: np.ndarray,
        F_act: np.ndarray,
        z_r: np.ndarray,
        zd_r: np.ndarray,
    ) -> CarOutputs:
        n = self.n
        q = x[:n]
        qd = x[n:]
        xd = self.state_deriv(x, F_act, z_r, zd_r)
        qdd = xd[n:]

        # Sprung corner displacements / velocities / accelerations
        corner_acc = self.J @ qdd[:3]
        z_s_i = self.J @ q[:3]
        zd_s_i = self.J @ qd[:3]
        z_u_i = q[3:7]
        zd_u_i = qd[3:7]

        susp_defl = z_s_i - z_u_i
        susp_defl_vel = zd_s_i - zd_u_i
        rel_vel = zd_u_i - zd_s_i

        tire_defl = z_u_i - z_r
        dyn_tire_load = self.p.k_t * tire_defl + self.p.c_t * (zd_u_i - zd_r)

        return CarOutputs(
            heave_acc=float(qdd[0]),
            pitch_acc=float(qdd[1]),
            roll_acc=float(qdd[2]),
            corner_acc=corner_acc,
            susp_defl=susp_defl,
            susp_defl_vel=susp_defl_vel,
            tire_defl=tire_defl,
            dyn_tire_load=dyn_tire_load,
            rel_vel=rel_vel,
        )

    def passive_suspension_force(self, x: np.ndarray) -> np.ndarray:
        """Spring + base-damper force on the body at each corner (no actuator).

        Returned for diagnostics; the springs/dampers are already embedded in A.
        """
        n = self.n
        q = x[:n]
        qd = x[n:]
        z_s_i = self.J @ q[:3]
        zd_s_i = self.J @ qd[:3]
        z_u_i = q[3:7]
        zd_u_i = qd[3:7]
        ks = self.p.k_s()
        cs = self.p.c_s()
        return -ks * (z_s_i - z_u_i) - cs * (zd_s_i - zd_u_i)
