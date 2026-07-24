"""Evaluate baseline or trained active suspension policies."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rl_suspension.baselines import PassivePolicy, PreviewRulePolicy, SkyhookGroundhookPolicy
from rl_suspension.envs import ActiveSuspensionEnv, EnvConfig
from rl_suspension.evaluation.metrics import EpisodeMetrics, summarize_episode


def evaluate_policy(
    policy,
    episodes: int = 5,
    curriculum_stage: int = 1,
    seed: int = 0,
    terminate_on_violation: bool = True,
) -> EpisodeMetrics:
    result = evaluate_policy_detailed(
        policy=policy,
        episodes=episodes,
        curriculum_stage=curriculum_stage,
        seed=seed,
        terminate_on_violation=terminate_on_violation,
    )
    return EpisodeMetrics(**result["metrics"])


def evaluate_policy_detailed(
    policy,
    episodes: int = 5,
    curriculum_stage: int = 1,
    seed: int = 0,
    expert=None,
    terminate_on_violation: bool = True,
) -> dict:
    env = ActiveSuspensionEnv(
        EnvConfig(
            curriculum_stage=curriculum_stage,
            terminate_on_violation=terminate_on_violation,
        )
    )
    summaries = []
    returns: list[float] = []
    inference_times_ns: list[int] = []
    expert_squared_errors: list[float] = []
    solver_latencies_ms: list[float] = []
    solver_fallbacks: list[float] = []
    for episode in range(episodes):
        reset_policy = getattr(policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        obs, _ = env.reset(seed=seed + episode, options={"curriculum_stage": curriculum_stage})
        infos: list[dict] = []
        episode_return = 0.0
        done = False
        while not done:
            started = time.perf_counter_ns()
            action, metadata = policy.predict(obs, deterministic=True)
            inference_times_ns.append(time.perf_counter_ns() - started)
            if isinstance(metadata, dict) and "latency_ms" in metadata:
                solver_latencies_ms.append(float(metadata["latency_ms"]))
                solver_fallbacks.append(float(metadata.get("fallback", 0.0)))
            if expert is not None:
                expert_action = expert.predict(obs).action
                expert_squared_errors.append(float(np.mean((action - expert_action) ** 2)))
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            infos.append(info)
            done = terminated or truncated
        summaries.append(summarize_episode(infos))
        returns.append(episode_return)
    env.close()
    metrics = _average_metrics(summaries)
    return {
        "metrics": metrics.to_dict(),
        "mean_episode_return": float(np.mean(returns)),
        "mean_inference_ms": float(np.mean(inference_times_ns) / 1e6),
        "expert_action_mse": (
            float(np.mean(expert_squared_errors))
            if expert_squared_errors
            else None
        ),
        "expert_query_rate": (
            float(policy.query_rate)
            if hasattr(policy, "query_rate")
            else None
        ),
        "solver_quality": (
            {
                "mean_latency_ms": float(np.mean(solver_latencies_ms)),
                "p95_latency_ms": float(np.quantile(solver_latencies_ms, 0.95)),
                "fallback_rate": float(np.mean(solver_fallbacks)),
            }
            if solver_latencies_ms
            else None
        ),
    }


def load_policy(
    name: str,
    model_path: Path | None,
    normalization_path: Path | None = None,
    ensemble_models: list[Path] | None = None,
    mpc_config_path: Path | None = None,
):
    if name == "passive":
        return PassivePolicy()
    if name == "skyhook":
        return SkyhookGroundhookPolicy()
    if name == "preview":
        return PreviewRulePolicy()
    if name == "mpc":
        from rl_suspension.imitation.mpc_expert import MpcExpert, MpcPolicy

        return MpcPolicy(MpcExpert.from_config(mpc_config_path))
    if name == "sac":
        if model_path is None:
            raise ValueError("--model-path is required when --policy sac")
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise RuntimeError("stable-baselines3 is required to evaluate SAC policies") from exc
        return SAC.load(model_path)
    if name == "bc":
        if model_path is None or normalization_path is None:
            raise ValueError("--model-path and --normalization-path are required for BC")
        from rl_suspension.imitation.policy import BehaviorClonedPolicy

        return BehaviorClonedPolicy.load(model_path, normalization_path)
    if name == "ensemble":
        if not ensemble_models or normalization_path is None:
            raise ValueError("--ensemble-model and --normalization-path are required")
        from rl_suspension.imitation.policy import BehaviorClonedPolicy, PolicyEnsemble

        return PolicyEnsemble(
            [
                BehaviorClonedPolicy.load(path, normalization_path)
                for path in ensemble_models
            ]
        )
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
    parser.add_argument(
        "--policy",
        choices=["passive", "skyhook", "preview", "mpc", "sac", "bc", "ensemble"],
        default="passive",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--normalization-path", type=Path, default=None)
    parser.add_argument("--ensemble-model", type=Path, action="append", default=[])
    parser.add_argument("--mpc-config", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--curriculum-stage", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--continue-on-violation",
        action="store_true",
        help="Record safety violations but evaluate the complete episode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = load_policy(
        args.policy,
        args.model_path,
        normalization_path=args.normalization_path,
        ensemble_models=args.ensemble_model,
        mpc_config_path=args.mpc_config,
    )
    payload = evaluate_policy_detailed(
        policy=policy,
        episodes=args.episodes,
        curriculum_stage=args.curriculum_stage,
        seed=args.seed,
        terminate_on_violation=not args.continue_on_violation,
    )
    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
