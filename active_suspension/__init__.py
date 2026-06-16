"""RL-first active suspension control project.

Implements the plan in suspension_control_roadmap:
  - 7-DOF full-car suspension model
  - forward-only (non-invertible) actuator model with force delay
  - ADS road-preview pipeline with confidence + error injection
  - IMU + 4 wheel-height-sensor state estimator and observation builder
  - ISO 2631 ride-comfort / handling metrics and RL reward
  - preview-LQR MPC teacher + command-level predictive safety filter
  - gymnasium environment with raw 12-command action space + domain randomization
"""

from .config import (
    VehicleParams,
    ActuatorParams,
    PreviewParams,
    SimConfig,
    default_config,
)

__all__ = [
    "VehicleParams",
    "ActuatorParams",
    "PreviewParams",
    "SimConfig",
    "default_config",
]

__version__ = "0.1.0"
