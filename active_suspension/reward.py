"""RL reward specification for active suspension.

Derived directly from the evaluation metrics so that "training objective" and
"validation score" stay aligned. The reward intentionally:
  * rewards comfort (low weighted body acceleration, pitch, roll),
  * penalizes proximity to hard constraints (travel, tire load),
  * penalizes command jerk/rate hard (anti bang-bang -> transfer),
  * penalizes energy use, and
  * penalizes safety-filter interventions (so the policy learns to stay feasible).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np

from .config import VehicleParams, ActuatorParams


@dataclass
class RewardWeights:
    # Comfort is the PRIMARY objective and must dominate normal operation.
    w_heave: float = 1.0
    w_pitch: float = 0.1
    w_roll: float = 0.1
    # Constraints act mainly as barriers near the limit, not penalties on
    # normal small motions (otherwise the policy stiffens and hurts comfort).
    w_sws: float = 0.3           # mild quadratic on travel usage
    w_sws_barrier: float = 40.0  # barrier active only past `sws_knee`
    sws_knee: float = 0.7        # fraction of travel limit where barrier starts
    w_dtl: float = 0.5           # hinge penalty only when tire load ratio > 1
    w_cmd_rate: float = 0.02     # command jerk penalty (anti bang-bang)
    w_energy: float = 1e-5
    w_safety: float = 2.0        # per safety-filter intervention
    alive_bonus: float = 0.0
    # normalization scales (typical magnitudes) to keep terms O(1)
    heave_scale: float = 1.0
    pitch_scale: float = 1.0
    roll_scale: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class RewardFunction:
    def __init__(
        self,
        weights: RewardWeights,
        vehicle: VehicleParams,
        actuator: ActuatorParams,
    ):
        self.w = weights
        self.travel_limit = vehicle.susp_travel_limit
        self.static_loads = vehicle.static_corner_loads()
        self.act = actuator

    def __call__(
        self,
        heave_acc: float,
        pitch_acc: float,
        roll_acc: float,
        susp_defl: np.ndarray,
        dyn_tire_load: np.ndarray,
        actuator_force: np.ndarray,
        rel_vel: np.ndarray,
        command_rate_norm: np.ndarray,
        safety_intervened: bool,
    ) -> "RewardInfo":
        w = self.w
        comfort = (
            w.w_heave * (heave_acc / w.heave_scale) ** 2
            + w.w_pitch * (pitch_acc / w.pitch_scale) ** 2
            + w.w_roll * (roll_acc / w.roll_scale) ** 2
        )

        sws = np.asarray(susp_defl, float)
        u = np.abs(sws) / self.travel_limit
        sws_usage = w.w_sws * np.mean(u ** 2)
        # barrier active only past the knee, so normal small motions are free
        over = np.clip(u - w.sws_knee, 0.0, None)
        barrier = w.w_sws_barrier * np.mean(over ** 2)

        ratio = np.abs(dyn_tire_load) / np.maximum(self.static_loads, 1e-6)
        # hinge: only penalize when the tire starts to lose contact (ratio > 1)
        dtl_pen = w.w_dtl * np.mean(np.clip(ratio - 1.0, 0.0, None) ** 2)

        cmd_rate_pen = w.w_cmd_rate * np.mean(np.asarray(command_rate_norm) ** 2)

        energy = w.w_energy * float(np.abs(actuator_force * rel_vel).sum())

        safety_pen = w.w_safety if safety_intervened else 0.0

        cost = comfort + sws_usage + barrier + dtl_pen + cmd_rate_pen + energy + safety_pen
        # Pure cost minimization; comfort dominates. Small alive bonus optional.
        reward = w.alive_bonus - cost
        return RewardInfo(
            reward=float(reward),
            comfort=float(comfort),
            sws=float(sws_usage + barrier),
            dtl=float(dtl_pen),
            cmd_rate=float(cmd_rate_pen),
            energy=float(energy),
            safety=float(safety_pen),
        )


@dataclass
class RewardInfo:
    reward: float
    comfort: float
    sws: float
    dtl: float
    cmd_rate: float
    energy: float
    safety: float
