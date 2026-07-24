"""Reproducible bounded search for preview-MPC cost weights."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from rl_suspension.baselines import PassivePolicy
from rl_suspension.controllers.mpc import MpcWeights, PreviewMPC, PreviewMpcConfig
from rl_suspension.evaluation.evaluate import evaluate_policy_detailed
from rl_suspension.imitation.mpc_expert import MpcExpert, MpcPolicy


def candidate_weights() -> list[MpcWeights]:
    """Small fixed grid; deterministic ordering makes tuning reproducible."""

    return [
        MpcWeights(),
        MpcWeights(tire_load_variation=1000.0),
        MpcWeights(
            heave_acceleration=4.0,
            tire_load_variation=100.0,
            force_effort=0.5,
            force_rate=50.0,
        ),
        MpcWeights(heave_acceleration=20.0, tire_load_variation=100.0),
        MpcWeights(
            heave_acceleration=30.0,
            pitch_acceleration=3.0,
            roll_acceleration=3.0,
            tire_load_variation=300.0,
            force_rate=20.0,
        ),
        MpcWeights(
            heave_acceleration=60.0,
            suspension_travel=5.0,
            tire_load_variation=500.0,
            force_effort=0.5,
            force_rate=50.0,
        ),
    ]


def tune_mpc(
    output_dir: str | Path,
    *,
    seed: int = 20_000,
    episodes: int = 3,
    curriculum_stage: int = 5,
    horizon: int = 40,
    candidate_limit: int | None = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    passive = evaluate_policy_detailed(
        PassivePolicy(),
        episodes=episodes,
        curriculum_stage=curriculum_stage,
        seed=seed,
        terminate_on_violation=False,
    )

    configurations = candidate_weights()
    if candidate_limit is not None:
        configurations = configurations[:candidate_limit]
    results: list[dict] = []
    for index, weights in enumerate(configurations):
        config = PreviewMpcConfig(horizon=horizon, weights=weights)
        policy = MpcPolicy(MpcExpert(PreviewMPC(config)))
        metrics = evaluate_policy_detailed(
            policy,
            episodes=episodes,
            curriculum_stage=curriculum_stage,
            seed=seed,
            terminate_on_violation=False,
        )
        safety_ok = (
            metrics["metrics"]["constraint_violations"]
            <= passive["metrics"]["constraint_violations"] + 1e-8
        )
        fallback_rate = (
            metrics["solver_quality"]["fallback_rate"]
            if metrics["solver_quality"] is not None
            else 1.0
        )
        results.append(
            {
                "candidate": index,
                "weights": asdict(weights),
                "metrics": metrics,
                "safety_ok": safety_ok,
                "fallback_rate": fallback_rate,
            }
        )

    eligible = [
        item
        for item in results
        if item["safety_ok"] and item["fallback_rate"] <= 0.01
    ]
    pool = eligible or results
    best = max(
        pool,
        key=lambda item: (
            item["metrics"]["mean_episode_return"],
            -item["metrics"]["metrics"]["rms_vertical_acceleration"],
        ),
    )
    required_gain = 0.01 * max(abs(passive["mean_episode_return"]), 1.0)
    qualified = bool(
        best["safety_ok"]
        and best["fallback_rate"] <= 0.01
        and best["metrics"]["mean_episode_return"]
        >= passive["mean_episode_return"] + required_gain
    )
    best_config = replace(
        PreviewMpcConfig(horizon=horizon),
        weights=MpcWeights(**best["weights"]),
    )
    config_path = output / "best_mpc_config.json"
    config_path.write_text(json.dumps(asdict(best_config), indent=2), encoding="utf-8")
    payload = {
        "qualified": qualified,
        "seed_start": seed,
        "episodes": episodes,
        "curriculum_stage": curriculum_stage,
        "passive": passive,
        "best": best,
        "candidates": results,
        "config_path": str(config_path),
    }
    (output / "tuning_results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--curriculum-stage", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--candidate-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = tune_mpc(
        args.output,
        seed=args.seed,
        episodes=args.episodes,
        curriculum_stage=args.curriculum_stage,
        horizon=args.horizon,
        candidate_limit=args.candidate_limit,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
