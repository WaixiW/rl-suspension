"""Convert the spatial ADS profile into wheel-aligned time preview."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rl_suspension.envs.observation import OBSERVATION_SPEC, ObservationSpec


def four_wheel_time_preview(
    observation: NDArray[np.floating],
    horizon: int,
    dt: float,
    spec: ObservationSpec = OBSERVATION_SPEC,
) -> NDArray[np.float64]:
    """Return road heights `[FL, FR, RL, RR]` for each prediction step."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    obs = spec.validate(np.asarray(observation))
    speed = max(float(obs[spec.speed][0]), 0.0)
    profile = spec.road_profile(obs).astype(np.float64)
    offsets = spec.road_offsets()
    travel = speed * dt * np.arange(horizon, dtype=np.float64)
    front_positions = travel
    rear_positions = travel - spec.wheelbase

    def sample(side: int, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.interp(
            positions,
            offsets,
            profile[side],
            left=float(profile[side, 0]),
            right=0.0,
        )

    return np.stack(
        [
            sample(0, front_positions),
            sample(1, front_positions),
            sample(0, rear_positions),
            sample(1, rear_positions),
        ],
        axis=1,
    ).astype(np.float64)
