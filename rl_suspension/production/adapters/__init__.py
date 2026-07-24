"""Adapter loading and safe reference implementations."""

from rl_suspension.production.adapters.loader import load_plugin
from rl_suspension.production.adapters.mpc import CallableMpcAdapter
from rl_suspension.production.adapters.rl import CallableRLTrainingAdapter
from rl_suspension.production.adapters.safe_controller import (
    ConstantSafeController,
    SafetyProjector,
)
from rl_suspension.production.adapters.simulator import CallableSimulatorAdapter

__all__ = [
    "CallableMpcAdapter",
    "CallableRLTrainingAdapter",
    "CallableSimulatorAdapter",
    "ConstantSafeController",
    "SafetyProjector",
    "load_plugin",
]
