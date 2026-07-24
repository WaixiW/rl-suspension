"""Named layout for the centralized active-suspension observation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ObservationSpec:
    """Stable slices for state, diagnostics, and full road preview.

    Keeping the layout in one place prevents teachers and learned policies from
    silently using stale hard-coded indices when the environment evolves.
    """

    state: slice = field(default_factory=lambda: slice(0, 14))
    suspension_deflections: slice = field(default_factory=lambda: slice(14, 18))
    suspension_velocities: slice = field(default_factory=lambda: slice(18, 22))
    previous_forces: slice = field(default_factory=lambda: slice(22, 26))
    currents: slice = field(default_factory=lambda: slice(26, 34))
    pump_speeds: slice = field(default_factory=lambda: slice(34, 38))
    actual_forces: slice = field(default_factory=lambda: slice(38, 42))
    speed: slice = field(default_factory=lambda: slice(42, 43))
    ads_features: slice = field(default_factory=lambda: slice(43, 50))
    state_feature_dimension: int = 50

    preview_resolution: float = 0.05
    preview_ahead: float = 8.0
    wheelbase: float = 2.8
    preview_points: int = 217
    road_left: slice = field(default_factory=lambda: slice(50, 267))
    road_right: slice = field(default_factory=lambda: slice(267, 484))
    dimension: int = 484

    # Sub-indices inside ads_features.
    ads_peak_distance: int = 0
    ads_peak_height: int = 1
    ads_width: int = 2
    ads_asymmetry: int = 3
    ads_left_slope: int = 4
    ads_right_slope: int = 5
    ads_confidence: int = 6

    @property
    def road_profile_start(self) -> float:
        return -self.wheelbase

    @property
    def road_profile_stop(self) -> float:
        return self.preview_ahead

    def road_offsets(self) -> np.ndarray:
        return np.linspace(
            self.road_profile_start,
            self.road_profile_stop,
            self.preview_points,
            dtype=np.float64,
        )

    def road_profile(self, observation: np.ndarray) -> np.ndarray:
        array = self.validate(observation)
        return np.stack([array[self.road_left], array[self.road_right]], axis=0)

    def validate(self, observation: np.ndarray) -> np.ndarray:
        array = np.asarray(observation)
        if array.shape != (self.dimension,):
            raise ValueError(
                f"observation must have shape ({self.dimension},), got {array.shape}"
            )
        return array


OBSERVATION_SPEC = ObservationSpec()
