"""Train a single SAC agent for force-level active suspension control."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_suspension.envs import ActiveSuspensionEnv, EnvConfig
from rl_suspension.training.curriculum import CurriculumStage, default_curriculum


def train_sac(
    output_dir: Path,
    total_timesteps_scale: float = 1.0,
    seed: int | None = None,
    verbose: int = 1,
) -> None:
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required for training. Install the project with "
            "`pip install -e .` from the rl_suspension directory."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(ActiveSuspensionEnv(EnvConfig(curriculum_stage=1)))
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=250_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=verbose,
        seed=seed,
    )

    for stage in default_curriculum():
        _train_stage(model, env, stage, output_dir, total_timesteps_scale)

    model.save(output_dir / "sac_active_suspension_final")


def _train_stage(
    model,
    env,
    stage: CurriculumStage,
    output_dir: Path,
    total_timesteps_scale: float,
) -> None:
    env.unwrapped.config = EnvConfig(curriculum_stage=stage.stage)
    timesteps = max(1, int(stage.timesteps * total_timesteps_scale))
    model.learn(
        total_timesteps=timesteps,
        reset_num_timesteps=False,
        progress_bar=True,
    )
    model.save(output_dir / f"sac_stage_{stage.stage}_{stage.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sac_active_suspension"))
    parser.add_argument("--timesteps-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_sac(
        output_dir=args.output_dir,
        total_timesteps_scale=args.timesteps_scale,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
