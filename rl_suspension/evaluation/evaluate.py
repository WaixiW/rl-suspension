"""Evaluate baseline or trained active suspension policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl_suspension.baselines import PassivePolicy, PreviewRulePolicy, SkyhookGroundhookPolicy
from rl_suspension.envs import ActiveSuspensionEnv, EnvConfig
from rl_suspension.evaluation.metrics import EpisodeMetrics, summarize_episode


def evaluate_policy(policy, episodes: int = 5, curriculum_stage: int = 1, seed: int = 0) -> EpisodeMetrics:
    env = ActiveSuspensionEnv(EnvConfig(curriculum_stage=curriculum_stage))
    summaries = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode, options={"curriculum_stage": curriculum_stage})
        infos: list[dict] = []
        done = False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            infos.append(info)
            done = terminated or truncated
        summaries.append(summarize_episode(infos))
    return _average_metrics(summaries)


def load_policy(name: str, model_path: Path | None):
    if name == "passive":
        return PassivePolicy()
    if name == "skyhook":
        return SkyhookGroundhookPolicy()
    if name == "preview":
        return PreviewRulePolicy()
    if name == "sac":
        if model_path is None:
            raise ValueError("--model-path is required when --policy sac")
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise RuntimeError("stable-baselines3 is required to evaluate SAC policies") from exc
        return SAC.load(model_path)
    raise ValueError(f"Unknown policy: {name}")


def _average_metrics(metrics: list[EpisodeMetrics]) -> EpisodeMetrics:
    keys = metrics[0].to_dict().keys()
    values = {
        key: float(np.mean([metric.to_dict()[key] for metric in metrics]))
        for key in keys
    }
    return EpisodeMetrics(**values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["passive", "skyhook", "preview", "sac"], default="passive")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--curriculum-stage", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = load_policy(args.policy, args.model_path)
    metrics = evaluate_policy(
        policy=policy,
        episodes=args.episodes,
        curriculum_stage=args.curriculum_stage,
        seed=args.seed,
    )
    payload = metrics.to_dict()
    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
