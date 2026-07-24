"""Held-out passive-versus-MPC benchmark by bump family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl_suspension.baselines import PassivePolicy
from rl_suspension.controllers.mpc import PreviewMPC
from rl_suspension.evaluation.evaluate import evaluate_policy_detailed
from rl_suspension.imitation.mpc_expert import (
    MpcExpert,
    MpcPolicy,
    load_mpc_config,
)


def benchmark_mpc(
    output_path: str | Path,
    *,
    config_path: str | Path,
    episodes: int = 3,
    seed: int = 40_000,
) -> dict:
    scenarios = {
        "single_bump": 2,
        "double_bump": 3,
        "asymmetric_bump": 4,
    }
    results: dict[str, dict] = {}
    for offset, (name, stage) in enumerate(scenarios.items()):
        scenario_seed = seed + 1000 * offset
        passive = evaluate_policy_detailed(
            PassivePolicy(),
            episodes=episodes,
            curriculum_stage=stage,
            seed=scenario_seed,
            terminate_on_violation=False,
        )
        mpc = evaluate_policy_detailed(
            MpcPolicy(MpcExpert(PreviewMPC(load_mpc_config(config_path)))),
            episodes=episodes,
            curriculum_stage=stage,
            seed=scenario_seed,
            terminate_on_violation=False,
        )
        safety_ok = (
            mpc["metrics"]["constraint_violations"]
            <= passive["metrics"]["constraint_violations"] + 1e-8
        )
        return_improvement = (
            mpc["mean_episode_return"] - passive["mean_episode_return"]
        ) / max(abs(passive["mean_episode_return"]), 1.0)
        comfort_improvement = 1.0 - (
            mpc["metrics"]["rms_vertical_acceleration"]
            / max(passive["metrics"]["rms_vertical_acceleration"], 1e-12)
        )
        results[name] = {
            "passive": passive,
            "mpc": mpc,
            "safety_ok": safety_ok,
            "return_improvement": return_improvement,
            "comfort_improvement": comfort_improvement,
        }

    passive_returns = [
        item["passive"]["mean_episode_return"] for item in results.values()
    ]
    mpc_returns = [item["mpc"]["mean_episode_return"] for item in results.values()]
    aggregate_gain = (float(np.mean(mpc_returns)) - float(np.mean(passive_returns))) / max(
        abs(float(np.mean(passive_returns))),
        1.0,
    )
    fallback_rates = [
        item["mpc"]["solver_quality"]["fallback_rate"] for item in results.values()
    ]
    qualified = bool(
        aggregate_gain >= 0.01
        and all(item["safety_ok"] for item in results.values())
        and max(fallback_rates) <= 0.01
    )
    payload = {
        "qualified": qualified,
        "episodes_per_scenario": episodes,
        "seed_start": seed,
        "aggregate_return_improvement": aggregate_gain,
        "maximum_fallback_rate": max(fallback_rates),
        "scenarios": results,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=40_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = benchmark_mpc(
        args.output,
        config_path=args.config,
        episodes=args.episodes,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
