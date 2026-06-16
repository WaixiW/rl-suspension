"""Domain randomization sampler.

Because there is NO real command-force data (only the nominal forward model
summarized from real tests), robustness must come from wide randomization around
the nominal model rather than system identification. Ranges are deliberately wide
and centered on the nominal config; a later milestone tightens them once real
logs exist. The `width` scalar lets the validation suite sweep randomization
strength.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .config import FullConfig


@dataclass
class DRConfig:
    enabled: bool = True
    width: float = 1.0   # global multiplier on every range (0 -> nominal only)

    # multiplicative ranges (lo, hi) applied to nominal values
    m_s: Tuple[float, float] = (0.85, 1.20)
    I_pitch: Tuple[float, float] = (0.85, 1.20)
    I_roll: Tuple[float, float] = (0.85, 1.20)
    m_u: Tuple[float, float] = (0.85, 1.20)
    k_s: Tuple[float, float] = (0.85, 1.20)
    c_s: Tuple[float, float] = (0.80, 1.25)
    k_t: Tuple[float, float] = (0.90, 1.15)

    # actuator (the most uncertain block)
    act_gain: Tuple[float, float] = (0.75, 1.30)   # scales k_pump, k_curr*
    act_tau: Tuple[float, float] = (0.6, 1.8)      # force lag
    act_delay: Tuple[float, float] = (0.5, 2.0)    # transport delay
    act_c_base: Tuple[float, float] = (0.7, 1.4)
    force_sat: Tuple[float, float] = (0.85, 1.15)

    # speed range [m/s]
    speed: Tuple[float, float] = (8.0, 25.0)


def _scaled(rng: np.random.Generator, lohi: Tuple[float, float], width: float) -> float:
    lo, hi = lohi
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * width
    return float(rng.uniform(mid - half, mid + half))


def sample_config(base: FullConfig, dr: DRConfig,
                  rng: np.random.Generator) -> FullConfig:
    cfg = base.copy()
    if not dr.enabled or dr.width <= 0:
        return cfg
    w = dr.width
    v = cfg.vehicle
    v.m_s *= _scaled(rng, dr.m_s, w)
    v.I_pitch *= _scaled(rng, dr.I_pitch, w)
    v.I_roll *= _scaled(rng, dr.I_roll, w)
    v.m_u *= _scaled(rng, dr.m_u, w)
    ks = _scaled(rng, dr.k_s, w)
    v.k_s_front *= ks
    v.k_s_rear *= ks
    cs = _scaled(rng, dr.c_s, w)
    v.c_s_front *= cs
    v.c_s_rear *= cs
    v.k_t *= _scaled(rng, dr.k_t, w)

    a = cfg.actuator
    g = _scaled(rng, dr.act_gain, w)
    a.k_pump *= g
    a.k_curr1 *= g
    a.k_curr2 *= g
    a.c_base *= _scaled(rng, dr.act_c_base, w)
    a.tau *= _scaled(rng, dr.act_tau, w)
    a.transport_delay_s *= _scaled(rng, dr.act_delay, w)
    a.force_sat *= _scaled(rng, dr.force_sat, w)

    cfg.sim.speed = _scaled(rng, dr.speed, w) if w > 0 else cfg.sim.speed
    return cfg
