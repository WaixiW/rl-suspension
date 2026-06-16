"""Road profile generation and a continuous road field for preview sampling.

A RoadField returns road height z_r(x, y) for a longitudinal position x [m] and a
lateral position y [m] (y>0 left). Lateral dependence lets us build asymmetric
(left/right-different) bumps that excite roll and warp in the 7-DOF model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class RoadField:
    """Wraps a height function z_r(x, y) plus a label and finite-diff velocity."""

    name: str
    height_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    length_m: float = 200.0

    def height(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return self.height_fn(x, y)

    def height_dx(self, x, y, eps: float = 1e-3):
        """Spatial slope dz/dx (used to get road velocity = v * dz/dx)."""
        return (self.height(x + eps, y) - self.height(x - eps, y)) / (2 * eps)


def _smooth_bump(x, x0, width, height):
    """A raised-cosine (1 - cos) bump centered at x0 with given width/height."""
    xr = x - x0
    inside = np.abs(xr) <= (width / 2.0)
    val = 0.5 * height * (1.0 + np.cos(2.0 * np.pi * xr / width))
    return np.where(inside, val, 0.0)


def single_bump(height=0.06, width=0.6, x0=10.0) -> RoadField:
    def fn(x, y):
        return _smooth_bump(x, x0, width, height) * np.ones_like(y)
    return RoadField("single_bump", fn)


def double_bump(height=0.06, width=0.5, x0=10.0, gap=1.2) -> RoadField:
    def fn(x, y):
        b = _smooth_bump(x, x0, width, height) + _smooth_bump(
            x, x0 + gap, width, height
        )
        return b * np.ones_like(y)
    return RoadField("double_bump", fn)


def asymmetric_bump(
    height_left=0.07, height_right=0.03, width=0.6, x0=10.0
) -> RoadField:
    """Left and right wheels see different bump heights -> roll/warp excitation."""

    def fn(x, y):
        left = _smooth_bump(x, x0, width, height_left)
        right = _smooth_bump(x, x0, width, height_right)
        w_left = np.clip((np.sign(y) + 1) / 2.0, 0, 1)  # 1 if y>0 else 0
        return left * w_left + right * (1 - w_left)

    return RoadField("asymmetric_bump", fn)


def pothole(depth=0.05, width=0.5, x0=10.0) -> RoadField:
    def fn(x, y):
        return -_smooth_bump(x, x0, width, depth) * np.ones_like(y)
    return RoadField("pothole", fn)


def speed_bump(height=0.10, width=0.9, x0=10.0) -> RoadField:
    def fn(x, y):
        return _smooth_bump(x, x0, width, height) * np.ones_like(y)
    return RoadField("speed_bump", fn)


def washboard(amp=0.012, wavelength=0.5, x0=8.0, length=6.0) -> RoadField:
    def fn(x, y):
        inside = (x >= x0) & (x <= x0 + length)
        wave = amp * np.sin(2 * np.pi * (x - x0) / wavelength)
        return np.where(inside, wave, 0.0) * np.ones_like(y)
    return RoadField("washboard", fn)


def iso_random(
    grade: str = "C", speed: float = 15.0, length: float = 220.0, seed: int = 0
) -> RoadField:
    """Approximate ISO 8608 random road via summed sinusoids.

    grade in {A,B,C,D}. Gd(n0) PSD value at n0=0.1 cycles/m.
    """
    gd0 = {"A": 16e-6, "B": 64e-6, "C": 256e-6, "D": 1024e-6}[grade.upper()]
    rng = np.random.default_rng(seed)
    n0 = 0.1
    # spatial frequency band
    freqs = np.linspace(0.01, 3.0, 200)  # cycles/m
    dn = freqs[1] - freqs[0]
    Gd = gd0 * (freqs / n0) ** (-2.0)
    amps = np.sqrt(2 * Gd * dn)
    phases = rng.uniform(0, 2 * np.pi, size=freqs.shape)
    # small independent left/right variation for realism
    phases_r = phases + rng.normal(0, 0.3, size=freqs.shape)

    def fn(x, y):
        xs = np.asarray(x, float)
        ys = np.asarray(y, float)
        shape = np.broadcast_shapes(xs.shape, ys.shape)
        xf = np.broadcast_to(xs, shape).reshape(-1)
        yf = np.broadcast_to(ys, shape).reshape(-1)
        wl = np.clip((np.sign(yf) + 1) / 2.0, 0, 1)
        ph = 2 * np.pi * np.outer(xf, freqs)
        zl = (np.sin(ph + phases) * amps).sum(axis=1)
        zr = (np.sin(ph + phases_r) * amps).sum(axis=1)
        out = zl * wl + zr * (1 - wl)
        if shape == ():
            return float(out[0])
        return out.reshape(shape)

    return RoadField(f"iso_{grade.upper()}", fn, length_m=length)


def flat() -> RoadField:
    return RoadField("flat", lambda x, y: np.zeros_like(np.asarray(x, float) * y))


# ----------------------------------------------------------------- registry
def scenario_library() -> Dict[str, Callable[[], RoadField]]:
    """Named scenario factories used across training and validation."""
    return {
        "single_bump": lambda: single_bump(),
        "double_bump": lambda: double_bump(),
        "asymmetric_bump": lambda: asymmetric_bump(),
        "pothole": lambda: pothole(),
        "speed_bump": lambda: speed_bump(),
        "washboard": lambda: washboard(),
        "iso_A": lambda: iso_random("A", seed=1),
        "iso_C": lambda: iso_random("C", seed=2),
        "iso_D": lambda: iso_random("D", seed=3),
        "flat": lambda: flat(),
    }


def make_scenario(name: str, **kwargs) -> RoadField:
    lib = scenario_library()
    if name not in lib:
        raise KeyError(f"unknown scenario '{name}'. options: {list(lib)}")
    return lib[name]()
