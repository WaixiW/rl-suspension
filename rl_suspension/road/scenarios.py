"""Parameterized road profiles for bump-focused training curricula."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from rl_suspension.models.types import FloatArray


ScenarioKind = Literal["single_bump", "double_bump", "asymmetric_bump", "flat"]


@dataclass(frozen=True)
class ScenarioConfig:
    kind: ScenarioKind = "single_bump"
    speed: float = 12.0
    bump_height: float = 0.05
    bump_width: float = 0.6
    bump_start: float = 4.0
    double_spacing: float = 2.0
    asymmetry: float = 0.025
    preview_distance: float = 8.0
    preview_resolution: float = 0.05
    wheelbase: float = 2.8
    episode_time: float = 2.0
    road_noise_std: float = 0.0


class RoadProfile:
    """Left/right road height function with ADS preview extraction."""

    def __init__(self, config: ScenarioConfig, rng: np.random.Generator | None = None) -> None:
        self.config = config
        self.rng = rng or np.random.default_rng()

    def wheel_heights(self, vehicle_x: float) -> FloatArray:
        cfg = self.config
        front_x = vehicle_x
        rear_x = vehicle_x - cfg.wheelbase
        return np.array(
            [
                self.height(front_x, side="left"),
                self.height(front_x, side="right"),
                self.height(rear_x, side="left"),
                self.height(rear_x, side="right"),
            ],
            dtype=np.float64,
        )

    def preview(self, vehicle_x: float) -> FloatArray:
        cfg = self.config
        offsets = np.arange(
            0.0,
            cfg.preview_distance + 0.5 * cfg.preview_resolution,
            cfg.preview_resolution,
            dtype=np.float64,
        )
        xs = vehicle_x + offsets
        left = np.array([self.height(x, side="left") for x in xs], dtype=np.float64)
        right = np.array([self.height(x, side="right") for x in xs], dtype=np.float64)
        if cfg.road_noise_std > 0.0:
            confidence = np.exp(-offsets / max(cfg.preview_distance, 1e-6))
            noise_std = cfg.road_noise_std * (1.0 - confidence)
            left = left + self.rng.normal(0.0, noise_std)
            right = right + self.rng.normal(0.0, noise_std)
        return np.stack([left, right], axis=0)

    def features(self, vehicle_x: float) -> FloatArray:
        cfg = self.config
        preview = self.preview(vehicle_x)
        offsets = np.arange(preview.shape[1], dtype=np.float64) * cfg.preview_resolution
        mean_profile = preview.mean(axis=0)
        peak_idx = int(np.argmax(np.abs(mean_profile)))
        peak_height = float(mean_profile[peak_idx])
        peak_distance = float(offsets[peak_idx])
        threshold = 0.2 * max(abs(peak_height), 1e-9)
        active = np.where(np.abs(mean_profile) >= threshold)[0]
        width = float((active[-1] - active[0] + 1) * cfg.preview_resolution) if active.size else 0.0
        asymmetry = float(preview[0, peak_idx] - preview[1, peak_idx])
        near_count = min(10, preview.shape[1])
        left_slope = float((preview[0, near_count - 1] - preview[0, 0]) / (near_count * cfg.preview_resolution))
        right_slope = float((preview[1, near_count - 1] - preview[1, 0]) / (near_count * cfg.preview_resolution))
        confidence = float(np.exp(-peak_distance / max(cfg.preview_distance, 1e-6)))
        return np.array(
            [peak_distance, peak_height, width, asymmetry, left_slope, right_slope, confidence],
            dtype=np.float64,
        )

    def height(self, x: float, side: Literal["left", "right"]) -> float:
        cfg = self.config
        if cfg.kind == "flat":
            return 0.0

        sign = 1.0
        height = cfg.bump_height
        if cfg.kind == "asymmetric_bump":
            height = cfg.bump_height + (0.5 * cfg.asymmetry if side == "left" else -0.5 * cfg.asymmetry)
            height = max(height, 0.0)

        value = self._cosine_bump(x, cfg.bump_start, cfg.bump_width, height)
        if cfg.kind == "double_bump":
            value += self._cosine_bump(
                x,
                cfg.bump_start + cfg.bump_width + cfg.double_spacing,
                cfg.bump_width,
                height,
            )
        return sign * value

    @staticmethod
    def _cosine_bump(x: float, start: float, width: float, height: float) -> float:
        if width <= 0.0 or x < start or x > start + width:
            return 0.0
        phase = (x - start) / width
        return float(0.5 * height * (1.0 - np.cos(2.0 * np.pi * phase)))


class BumpScenario:
    """Factory for curriculum road scenarios."""

    def __init__(self, config: ScenarioConfig, seed: int | None = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)

    def make_profile(self) -> RoadProfile:
        return RoadProfile(self.config, rng=self.rng)

    def randomized(self, stage: int) -> RoadProfile:
        cfg = self.config
        kind: ScenarioKind = cfg.kind
        if stage <= 1:
            return RoadProfile(cfg, rng=self.rng)
        if stage == 2:
            kind = "single_bump"
        elif stage == 3:
            kind = "double_bump"
        elif stage == 4:
            kind = "asymmetric_bump"
        elif stage >= 5:
            kind = self.rng.choice(["single_bump", "double_bump", "asymmetric_bump"]).item()

        randomized_cfg = ScenarioConfig(
            kind=kind,
            speed=float(self.rng.uniform(8.0, 22.0)),
            bump_height=float(self.rng.uniform(0.02, 0.08)),
            bump_width=float(self.rng.uniform(0.35, 1.0)),
            bump_start=float(self.rng.uniform(2.0, 5.0)),
            double_spacing=float(self.rng.uniform(1.0, 4.0)),
            asymmetry=float(self.rng.uniform(0.0, 0.05)),
            preview_distance=cfg.preview_distance,
            preview_resolution=cfg.preview_resolution,
            wheelbase=cfg.wheelbase,
            episode_time=cfg.episode_time,
            road_noise_std=cfg.road_noise_std if stage < 5 else max(cfg.road_noise_std, 0.005),
        )
        return RoadProfile(randomized_cfg, rng=self.rng)
