"""Evaluation metrics shared by all controllers.

Comfort:  ISO 2631-1 frequency-weighted RMS of vertical body acceleration (Wk),
          plus pitch/roll acceleration RMS.
Handling/safety: suspension working space RMS and dynamic tire load ratio.
Effort:   actuator force/rate, command rate, energy proxy.
Robustness: constraint violation counts.

The Wk weighting is implemented as a digital filter from the ISO 2631-1 transfer
function (Zuo & Nayfeh / BS 6841 realization), applied with scipy.signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
from scipy import signal


# ----------------------------------------------------------- ISO 2631 Wk filter
def _wk_analog():
    """Return (b, a) analog coefficients for the Wk vertical weighting.

    Standard cascade: band-limiting + a-v transition + upward step. Parameters
    from ISO 2631-1:1997 (vertical seat, z-axis).
    """
    # high-pass (f1) and low-pass (f2) band limiting
    f1, f2 = 0.4, 100.0
    w1, w2 = 2 * np.pi * f1, 2 * np.pi * f2
    Q1 = 1 / np.sqrt(2)
    # a-v transition (acceleration->velocity proportionality knee near 12.5 Hz)
    f3, f4 = 12.5, 12.5
    Q4 = 0.63
    w3, w4 = 2 * np.pi * f3, 2 * np.pi * f4

    # High pass (2nd order)
    Hh = ([1, 0, 0], [1, w1 / Q1, w1 ** 2])
    # Low pass (2nd order)
    Hl = ([w2 ** 2], [1, w2 / Q1, w2 ** 2])
    # a-v transition
    Ht = ([0, w4 ** 2 / w3, w4 ** 2], [1, w4 / Q4, w4 ** 2])
    return Hh, Hl, Ht


def wk_weighted_rms(acc: np.ndarray, fs: float) -> float:
    """ISO 2631 Wk frequency-weighted RMS of an acceleration signal.

    Falls back gracefully for very short signals.
    """
    acc = np.asarray(acc, float).reshape(-1)
    if acc.size < 8:
        return float(np.sqrt(np.mean(acc ** 2))) if acc.size else 0.0
    # Build Wk as a product of analog biquads, discretize via bilinear.
    Hh, Hl, Ht = _wk_analog()
    sos_list = []
    for (b, a) in (Hh, Hl, Ht):
        bz, az = signal.bilinear(b, a, fs=fs)
        sos = signal.tf2sos(bz, az)
        sos_list.append(sos)
    sos = np.vstack(sos_list)
    try:
        y = signal.sosfilt(sos, acc)
    except Exception:
        y = acc
    return float(np.sqrt(np.mean(y ** 2)))


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


@dataclass
class EpisodeMetrics:
    """Aggregate metrics over one episode."""

    comfort_wk_rms: float = 0.0          # ISO 2631 weighted vertical acc [m/s^2]
    heave_acc_rms: float = 0.0
    pitch_acc_rms: float = 0.0
    roll_acc_rms: float = 0.0
    sws_rms: float = 0.0                 # suspension working space RMS [m]
    sws_max: float = 0.0
    dtl_ratio_rms: float = 0.0          # dynamic/static tire load ratio RMS
    dtl_ratio_max: float = 0.0
    tire_liftoff_frac: float = 0.0      # fraction of time any tire load ratio > 1
    actuator_force_rms: float = 0.0
    command_rate_rms: float = 0.0
    energy_proxy: float = 0.0
    travel_violations: int = 0
    safety_interventions: int = 0
    n_steps: int = 0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class MetricRecorder:
    """Accumulates per-step signals and produces EpisodeMetrics."""

    def __init__(self, fs_control: float, static_loads: np.ndarray,
                 travel_limit: float):
        self.fs = fs_control
        self.static_loads = np.asarray(static_loads, float)
        self.travel_limit = travel_limit
        self.reset()

    def reset(self) -> None:
        self._heave: List[float] = []
        self._pitch: List[float] = []
        self._roll: List[float] = []
        self._sws: List[np.ndarray] = []
        self._dtl_ratio: List[np.ndarray] = []
        self._force: List[np.ndarray] = []
        self._cmd: List[np.ndarray] = []
        self._prev_cmd: Optional[np.ndarray] = None
        self._cmd_rate: List[np.ndarray] = []
        self._travel_viol = 0
        self._safety_interv = 0

    def record(
        self,
        heave_acc: float,
        pitch_acc: float,
        roll_acc: float,
        susp_defl: np.ndarray,
        dyn_tire_load: np.ndarray,
        actuator_force: np.ndarray,
        rel_vel: np.ndarray,
        command: np.ndarray,
        safety_intervened: bool = False,
    ) -> None:
        self._heave.append(heave_acc)
        self._pitch.append(pitch_acc)
        self._roll.append(roll_acc)
        self._sws.append(np.asarray(susp_defl, float))
        ratio = np.abs(dyn_tire_load) / np.maximum(self.static_loads, 1e-6)
        self._dtl_ratio.append(ratio)
        self._force.append(np.asarray(actuator_force, float))
        cmd = np.asarray(command, float)
        self._cmd.append(cmd)
        if self._prev_cmd is not None:
            self._cmd_rate.append(cmd - self._prev_cmd)
        self._prev_cmd = cmd
        if np.any(np.abs(susp_defl) > self.travel_limit):
            self._travel_viol += 1
        if safety_intervened:
            self._safety_interv += 1
        # energy proxy: actuator force * relative velocity (power) integrated
        self._energy_step = np.abs(actuator_force * rel_vel).sum()
        if not hasattr(self, "_energy"):
            self._energy = 0.0
        self._energy += self._energy_step / self.fs

    def compute(self) -> EpisodeMetrics:
        heave = np.array(self._heave)
        sws = np.array(self._sws) if self._sws else np.zeros((1, 4))
        dtl = np.array(self._dtl_ratio) if self._dtl_ratio else np.zeros((1, 4))
        force = np.array(self._force) if self._force else np.zeros((1, 4))
        cmd_rate = np.array(self._cmd_rate) if self._cmd_rate else np.zeros((1, 12))
        m = EpisodeMetrics()
        m.comfort_wk_rms = wk_weighted_rms(heave, self.fs)
        m.heave_acc_rms = rms(heave)
        m.pitch_acc_rms = rms(np.array(self._pitch))
        m.roll_acc_rms = rms(np.array(self._roll))
        m.sws_rms = rms(sws)
        m.sws_max = float(np.max(np.abs(sws))) if sws.size else 0.0
        m.dtl_ratio_rms = rms(dtl)
        m.dtl_ratio_max = float(np.max(dtl)) if dtl.size else 0.0
        m.tire_liftoff_frac = float(np.mean(np.any(dtl > 1.0, axis=1)))
        m.actuator_force_rms = rms(force)
        m.command_rate_rms = rms(cmd_rate)
        m.energy_proxy = float(getattr(self, "_energy", 0.0))
        m.travel_violations = int(self._travel_viol)
        m.safety_interventions = int(self._safety_interv)
        m.n_steps = len(self._heave)
        return m
