"""Controller evaluation metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeMetrics:
    rms_vertical_acceleration: float
    peak_vertical_acceleration: float
    pitch_peak: float
    roll_peak: float
    max_suspension_travel: float
    min_tire_load: float
    actuator_energy: float
    mean_action_rate: float
    constraint_violations: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def summarize_episode(infos: list[dict]) -> EpisodeMetrics:
    if not infos:
        raise ValueError("Cannot summarize an empty episode")
    return EpisodeMetrics(
        rms_vertical_acceleration=float(infos[-1]["rms_body_acceleration"]),
        peak_vertical_acceleration=float(max(info["peak_body_acceleration"] for info in infos)),
        pitch_peak=float(max(info["pitch_peak"] for info in infos)),
        roll_peak=float(max(info["roll_peak"] for info in infos)),
        max_suspension_travel=float(max(info["max_suspension_travel"] for info in infos)),
        min_tire_load=float(min(info["tire_load_min"] for info in infos)),
        actuator_energy=float(np.mean([info["actuator_energy"] for info in infos])),
        mean_action_rate=float(np.mean([info["action_rate"] for info in infos])),
        constraint_violations=float(
            sum(sum(info["constraint_violations"].values()) for info in infos)
        ),
    )
