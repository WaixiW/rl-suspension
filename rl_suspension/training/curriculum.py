"""Training curriculum definitions for bump-focused active suspension RL."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    stage: int
    name: str
    description: str
    timesteps: int


def default_curriculum() -> list[CurriculumStage]:
    """Return the staged curriculum from the project plan."""

    return [
        CurriculumStage(
            stage=1,
            name="single_fixed",
            description="Single bump, fixed speed, fixed height, no noise.",
            timesteps=25_000,
        ),
        CurriculumStage(
            stage=2,
            name="single_randomized",
            description="Single bump with randomized speed, height, and width.",
            timesteps=50_000,
        ),
        CurriculumStage(
            stage=3,
            name="double_bump",
            description="Double bump with randomized spacing.",
            timesteps=50_000,
        ),
        CurriculumStage(
            stage=4,
            name="asymmetric_bump",
            description="Asymmetric bump with randomized left/right height difference.",
            timesteps=75_000,
        ),
        CurriculumStage(
            stage=5,
            name="mixed_noisy",
            description="Mixed bump scenes with ADS noise and actuator delay.",
            timesteps=100_000,
        ),
        CurriculumStage(
            stage=6,
            name="domain_randomized",
            description="Robustness training with scenario and model randomization hooks.",
            timesteps=150_000,
        ),
    ]
