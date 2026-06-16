"""Forward-only actuator model and its offline (Path B) inverse.

Per corner the commands are (i1, i2, n_pump):
  * i1, i2 in [0, i_max]  : two damping-valve currents (dissipative)
  * n_pump in [-n_max, n_max] : pump rotation speed (bidirectional active force)

Forward map (per corner), given suspension relative velocity v_rel = zd_u - zd_s:
    c_eff      = c_base + k_curr1 * i1 + k_curr2 * i2          (>= 0)
    F_damp     = -c_eff * sat(v_rel, damp_vel_limit)           (opposes motion)
    F_active   =  k_pump * n_pump
    F_target   = sat(F_damp + F_active, force_sat)

The realized force then follows a first-order lag with a pure transport delay:
    F[k]  = (1 - a) * F[k-d]_target + a * F[k-1]
with a = exp(-dt/tau), d = round(transport_delay_s / dt).

This map is forward-only and redundant (3 inputs -> 1 scalar force per corner),
hence non-invertible in closed form. The RL policy therefore acts directly in the
command space; the offline inverse below is used ONLY to create imitation labels
for the MPC teacher.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .config import ActuatorParams, N_CORNERS


def _sat(x, lim):
    return np.clip(x, -lim, lim)


class ForwardActuator:
    """Stateful forward actuator with first-order lag + transport delay."""

    def __init__(self, params: ActuatorParams, dt: float):
        self.p = params
        self.dt = dt
        self.alpha = float(np.exp(-dt / max(params.tau, 1e-6)))
        self.delay_steps = int(round(params.transport_delay_s / dt))
        self.reset()

    def reset(self) -> None:
        self.force = np.zeros(N_CORNERS)
        # buffer of past *target* forces for transport delay
        self._buf = deque(
            [np.zeros(N_CORNERS) for _ in range(self.delay_steps + 1)],
            maxlen=self.delay_steps + 1,
        )

    # ---------------------------------------------------------------- command
    @staticmethod
    def normalize_action(action: np.ndarray, p: ActuatorParams) -> np.ndarray:
        """Map a (12,) policy action in [-1,1] to physical commands (4,3).

        Currents map clip(a,0,1) -> [0, i_max] (a<=0 is a passive dead zone);
        pump maps [-1,1] -> [-n_max, n_max]. This makes the neutral action 0
        correspond to the passive damper (zero current, zero pump), giving the
        policy a good starting point and an easy passive fallback.
        """
        a = np.asarray(action, dtype=float).reshape(N_CORNERS, 3)
        i1 = np.clip(a[:, 0], 0.0, 1.0) * p.i_max
        i2 = np.clip(a[:, 1], 0.0, 1.0) * p.i_max
        npump = np.clip(a[:, 2], -1.0, 1.0) * p.n_pump_max
        return np.stack([i1, i2, npump], axis=1)

    @staticmethod
    def denormalize_command(cmd: np.ndarray, p: ActuatorParams) -> np.ndarray:
        """Inverse of normalize_action: physical commands (4,3) -> action (12,)."""
        c = np.asarray(cmd, dtype=float).reshape(N_CORNERS, 3)
        a0 = np.clip(c[:, 0] / p.i_max, 0, 1)
        a1 = np.clip(c[:, 1] / p.i_max, 0, 1)
        a2 = np.clip(c[:, 2] / p.n_pump_max, -1, 1)
        return np.stack([a0, a1, a2], axis=1).reshape(-1)

    def target_force(self, cmd: np.ndarray, rel_vel: np.ndarray) -> np.ndarray:
        """Instantaneous target force from commands (no lag). cmd: (4,3)."""
        p = self.p
        cmd = np.asarray(cmd, dtype=float).reshape(N_CORNERS, 3)
        i1 = np.clip(cmd[:, 0], 0, p.i_max)
        i2 = np.clip(cmd[:, 1], 0, p.i_max)
        npump = np.clip(cmd[:, 2], -p.n_pump_max, p.n_pump_max)
        c_eff = p.c_base + p.k_curr1 * i1 + p.k_curr2 * i2
        # rel_vel = zd_u - zd_s. A dissipative damper force on the body opposes
        # body-relative-to-wheel velocity (zd_s - zd_u) = -rel_vel, hence the
        # force on the body is +c_eff * rel_vel.
        f_damp = c_eff * _sat(rel_vel, p.damp_vel_limit)
        f_active = p.k_pump * npump
        return _sat(f_damp + f_active, p.force_sat)

    def step(self, cmd: np.ndarray, rel_vel: np.ndarray) -> np.ndarray:
        """Advance actuator one dt and return the realized force (4,)."""
        f_tgt = self.target_force(cmd, rel_vel)
        self._buf.append(f_tgt.copy())
        delayed_tgt = self._buf[0]  # oldest entry == delayed by delay_steps
        self.force = (1.0 - self.alpha) * delayed_tgt + self.alpha * self.force
        return self.force.copy()


class OfflineActuatorInverse:
    """Path B: approximate command for a desired force, used offline only.

    Resolves the 3->1 redundancy with a minimum-effort criterion: prefer pump
    actuation for the active part and only use damping current when it helps
    dissipate (i.e. when the desired force opposes the relative velocity). This
    is a cheap heuristic; it does not need to be exact since it only seeds
    imitation labels for the teacher.
    """

    def __init__(self, params: ActuatorParams):
        self.p = params

    def invert(self, f_desired: np.ndarray, rel_vel: np.ndarray) -> np.ndarray:
        """Return commands (4,3) approximating f_desired given rel_vel."""
        p = self.p
        f_desired = np.asarray(f_desired, dtype=float).reshape(-1)
        rel_vel = np.asarray(rel_vel, dtype=float).reshape(-1)
        n = f_desired.shape[0]
        cmd = np.zeros((n, 3))
        v = _sat(rel_vel, p.damp_vel_limit)
        c_extra_max = p.k_curr1 * p.i_max + p.k_curr2 * p.i_max
        gsum = p.k_curr1 + p.k_curr2
        for k in range(n):
            f = float(np.clip(f_desired[k], -p.force_sat, p.force_sat))
            vk = float(v[k])
            # Damper force on body = c_eff * v, direction sign(v). Base damping
            # c_base * v is always present.
            f_base = p.c_base * vk
            remainder = f - f_base
            c_needed = 0.0
            if abs(vk) > 1e-4 and np.sign(remainder) == np.sign(vk):
                # extra damping helps reach the target in direction sign(v)
                c_needed = float(np.clip(remainder / vk, 0.0, c_extra_max))
                i1 = np.clip((c_needed * p.k_curr1 / gsum) / p.k_curr1, 0, p.i_max)
                i2 = np.clip((c_needed * p.k_curr2 / gsum) / p.k_curr2, 0, p.i_max)
                cmd[k, 0], cmd[k, 1] = i1, i2
            f_damp_total = (p.c_base + c_needed) * vk
            # Pump covers the remaining active force.
            f_rem = f - f_damp_total
            cmd[k, 2] = np.clip(f_rem / p.k_pump, -p.n_pump_max, p.n_pump_max)
        return cmd
