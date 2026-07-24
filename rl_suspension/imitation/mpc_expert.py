"""Adapter exposing preview MPC through imitation and policy interfaces."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rl_suspension.controllers.mpc import MpcWeights, PreviewMPC, PreviewMpcConfig
from rl_suspension.imitation.experts import ExpertResult


def load_mpc_config(path: str | Path | None = None) -> PreviewMpcConfig:
    if path is None:
        return PreviewMpcConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = MpcWeights(**payload.pop("weights", {}))
    return PreviewMpcConfig(weights=weights, **payload)


@dataclass
class MpcExpert:
    controller: PreviewMPC
    name: str = "preview_mpc"

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "MpcExpert":
        return cls(PreviewMPC(load_mpc_config(path)))

    def reset(self) -> None:
        self.controller.reset()

    def predict(self, observation: NDArray[np.floating]) -> ExpertResult:
        result = self.controller.solve(observation)
        valid = bool(
            not result.fallback
            and result.status in {"optimal", "optimal_inaccurate"}
            and np.all(np.isfinite(result.action))
        )
        quality = (
            float(1.0 / (1.0 + max(result.predicted_violation, 0.0)))
            if valid
            else 0.0
        )
        return ExpertResult(
            action=result.action,
            valid=valid,
            quality=quality,
            diagnostics={
                "teacher": self.name,
                "status": result.status,
                "objective": result.objective,
                "iterations": float(result.iterations),
                "latency_ms": 1000.0 * result.solve_time,
                "constraint_margin": result.constraint_margin,
                "predicted_violation": result.predicted_violation,
                "fallback": float(result.fallback),
            },
        )


@dataclass
class MpcPolicy:
    """Gym/SB3-style prediction wrapper used by evaluation."""

    expert: MpcExpert

    def reset(self) -> None:
        self.expert.reset()

    def predict(self, observation, deterministic: bool = True):
        del deterministic
        result = self.expert.predict(observation)
        return result.action, result.diagnostics
