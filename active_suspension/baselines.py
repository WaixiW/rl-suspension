"""Baseline controllers operating in the raw command space.

These share the same interface as the RL policy and MPC teacher so the
evaluation harness can compare them directly:

    command = controller.act(obs_dict) -> (4,3) physical commands

where obs_dict carries the estimated state and (for skyhook) the relative
velocities. All baselines obey the same forward-only actuator command space.
"""

from __future__ import annotations

import numpy as np

from .config import ActuatorParams, N_CORNERS


class PassiveController:
    """Zero current, zero pump -> only the base damping acts. Reference floor."""

    def __init__(self, actuator: ActuatorParams):
        self.p = actuator

    def reset(self):
        pass

    def act(self, est: dict) -> np.ndarray:
        return np.zeros((N_CORNERS, 3))


class SkyhookController:
    """Semi-active skyhook using the damping currents only (no pump).

    Classic skyhook: raise damping when body velocity and relative velocity have
    the same sign (damper can pull energy out), else minimum damping. The pump
    stays off, so this is a purely dissipative, transfer-safe baseline.
    """

    def __init__(self, actuator: ActuatorParams, gain: float = 0.5,
                 vel_ref: float = 0.5, i_cap_frac: float = 0.35):
        self.p = actuator
        self.gain = gain          # current per unit body velocity scale
        self.vel_ref = vel_ref    # body velocity giving ~full current
        self.i_cap_frac = i_cap_frac

    def reset(self):
        pass

    def act(self, est: dict) -> np.ndarray:
        zd_s_corner = np.asarray(est["corner_body_vel"], float)  # body vel at corner
        rel_vel = np.asarray(est["rel_vel"], float)              # zd_u - zd_s
        cmd = np.zeros((N_CORNERS, 3))
        # Continuous semi-active skyhook: add damping proportional to body
        # velocity, but only when the damper can dissipate body motion (zd_s and
        # suspension velocity aligned). Smooth modulation avoids on/off harshness.
        v_susp = -rel_vel  # zd_s - zd_u
        dissipative = (zd_s_corner * v_susp) > 0
        demand = self.gain * np.abs(zd_s_corner) / self.vel_ref  # in [0, ~1+]
        i_cmd = np.where(dissipative, np.clip(demand, 0, self.i_cap_frac), 0.0) * self.p.i_max
        cmd[:, 0] = i_cmd
        cmd[:, 1] = i_cmd
        return cmd
