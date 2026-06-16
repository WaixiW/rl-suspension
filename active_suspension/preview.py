"""ADS road-preview pipeline.

Responsibilities:
  1. Track per-wheel longitudinal position as the car drives forward.
  2. Provide the *true* road input (z_r, zd_r) at each wheel for the simulator.
  3. Produce the preview vector the policy sees: a confidence-weighted,
     speed/time-aligned, error-corrupted look-ahead of road height per wheel
     path, plus the confidence profile itself.

Geometry: the front axle is at the vehicle reference; the rear axle trails by
the wheelbase (a+b), so the rear wheels see what the front saw (wheelbase / v)
seconds earlier. Lateral offsets pick the left/right road profile.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import PreviewParams, VehicleParams, N_CORNERS
from .road import RoadField


class PreviewSensor:
    def __init__(
        self,
        preview: PreviewParams,
        vehicle: VehicleParams,
        road: RoadField,
        speed: float,
        seed: int = 0,
    ):
        self.pp = preview
        self.vp = vehicle
        self.road = road
        self.speed = speed
        self.rng = np.random.default_rng(seed)

        pos = vehicle.corner_positions()
        self.corner_x = pos[:, 0]   # +a front, -b rear
        self.corner_y = pos[:, 1]
        # distance ahead each preview grid point [m]
        self.grid_d = np.arange(1, preview.n_samples + 1) * preview.resolution_m
        # confidence decays linearly with look-ahead distance
        frac = self.grid_d / preview.horizon_m
        self.confidence = preview.conf_near + (preview.conf_far - preview.conf_near) * frac
        self.confidence = np.clip(self.confidence, 0.0, 1.0)

        # vehicle reference longitudinal position (front axle datum)
        self.s = 0.0
        # start so the first event is a few metres ahead
        self.s0 = 0.0

    def reset(self, s0: float = 0.0) -> None:
        self.s = s0
        self.s0 = s0

    def advance(self, dt: float) -> None:
        self.s += self.speed * dt

    # ------------------------------------------------------- true road input
    def _wheel_long_pos(self) -> np.ndarray:
        """Absolute longitudinal road position under each wheel."""
        # Reference s is the CG datum; corner_x adds the longitudinal offset.
        return self.s + self.corner_x

    def true_road(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (z_r, zd_r) (4,) under the four wheels at the current pose."""
        xw = self._wheel_long_pos()
        yw = self.corner_y
        z_r = self.road.height(xw, yw)
        zd_r = self.speed * self.road.height_dx(xw, yw)
        return np.asarray(z_r, float).reshape(-1), np.asarray(zd_r, float).reshape(-1)

    # ----------------------------------------------------------- preview obs
    def preview_grid(self, corrupt: bool = True) -> np.ndarray:
        """Per-wheel look-ahead road height (4, n_samples).

        With corruption, height noise/dropout/lateral jitter scale with
        (1 - confidence) so the far field is noisier than the near field.
        """
        xw = self._wheel_long_pos()         # (4,)
        yw = self.corner_y                  # (4,)
        # latency shifts the look-ahead reference back in time
        s_lat = self.pp.latency_s * self.speed
        grid_x = xw[:, None] + self.grid_d[None, :] - s_lat   # (4, n)
        grid_y = np.repeat(yw[:, None], self.grid_d.shape[0], axis=1)

        if corrupt and self.pp.lateral_jitter_m > 0:
            jit = self.rng.normal(0, self.pp.lateral_jitter_m, size=grid_y.shape)
            grid_y = grid_y + jit * (1 - self.confidence)[None, :]

        z = self.road.height(grid_x, grid_y)
        z = np.asarray(z, float).reshape(N_CORNERS, -1)

        if corrupt:
            unc = (1 - self.confidence)[None, :]
            noise = self.rng.normal(0, self.pp.height_noise_std, size=z.shape)
            z = z + noise * unc
            drop = self.rng.random(z.shape) < (self.pp.dropout_prob * unc)
            z = np.where(drop, 0.0, z)
        return z

    def preview_obs(self) -> np.ndarray:
        """Downsampled, flattened preview for the policy: (4 * n_ds,)."""
        z = self.preview_grid(corrupt=True)[:, :: self.pp.obs_stride]
        return z.reshape(-1)

    def confidence_obs(self) -> np.ndarray:
        return self.confidence[:: self.pp.obs_stride].copy()

    @property
    def n_preview_obs(self) -> int:
        n_ds = self.grid_d[:: self.pp.obs_stride].shape[0]
        return N_CORNERS * n_ds

    @property
    def n_conf_obs(self) -> int:
        return self.grid_d[:: self.pp.obs_stride].shape[0]
