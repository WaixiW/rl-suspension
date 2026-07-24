"""Collect episode-sharded labels from a temporary or future expert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl_suspension.baselines import PreviewRulePolicy, SkyhookGroundhookPolicy
from rl_suspension.envs import ActiveSuspensionEnv, EnvConfig, OBSERVATION_SPEC
from rl_suspension.imitation.dataset import (
    EpisodePhase,
    EpisodeShardWriter,
    compute_normalization,
    load_dataset,
    save_normalization,
)
from rl_suspension.imitation.experts import (
    Expert,
    PolicyExpert,
    QualificationResult,
    qualify_expert_against_passive,
    qualify_temporary_expert,
)


def collect_dataset(
    output_dir: str | Path,
    episodes: int,
    teacher: str = "mpc",
    seed_start: int = 0,
    curriculum_stage: int = 5,
    qualification_episodes: int = 5,
    allow_unqualified: bool = False,
    initial_state_std: float = 0.0,
    observation_noise_std: float = 0.0,
    mpc_config_path: str | Path | None = None,
) -> dict:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    output = Path(output_dir)
    writer = EpisodeShardWriter(output)

    def env_factory() -> ActiveSuspensionEnv:
        return ActiveSuspensionEnv(
            EnvConfig(
                curriculum_stage=curriculum_stage,
                observation_noise_std=observation_noise_std,
                terminate_on_violation=False,
            )
        )

    selected, qualification = _select_teacher(
        teacher=teacher,
        env_factory=env_factory,
        qualification_seeds=list(
            range(seed_start + 50_000, seed_start + 50_000 + qualification_episodes)
        ),
        allow_unqualified=allow_unqualified,
        mpc_config_path=mpc_config_path,
    )

    rng = np.random.default_rng(seed_start)
    phase_counts = {phase.name.lower(): 0 for phase in EpisodePhase}
    total_transitions = 0
    invalid_labels = 0
    expert_latencies_ms: list[float] = []
    expert_fallbacks: list[float] = []

    for local_episode in range(episodes):
        episode_id = len(writer.records)
        scenario_seed = seed_start + local_episode
        env = env_factory()
        observation, info = env.reset(
            seed=scenario_seed,
            options={"curriculum_stage": curriculum_stage},
        )
        reset_teacher = getattr(selected, "reset", None)
        if callable(reset_teacher):
            reset_teacher()
        if initial_state_std > 0.0:
            env.state.q += rng.normal(0.0, initial_state_std, size=7)
            env.state.qd += rng.normal(0.0, initial_state_std, size=7)
            observation = env._build_observation()

        buffers = _empty_episode_buffers()
        done = False
        while not done:
            label = selected.predict(observation)
            phase = classify_phase(env, observation)
            next_observation, reward, terminated, truncated, step_info = env.step(label.action)
            violation = float(sum(step_info["constraint_violations"].values()))

            _append_transition(
                buffers,
                observation=observation,
                action=label.action,
                behavior_action=label.action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
                episode_id=episode_id,
                scenario_seed=scenario_seed,
                phase=phase,
                expert_valid=label.valid,
                expert_quality=label.quality,
                expert_diagnostics=label.diagnostics,
                constraint_violation=violation,
            )
            if "latency_ms" in label.diagnostics:
                expert_latencies_ms.append(float(label.diagnostics["latency_ms"]))
            if "fallback" in label.diagnostics:
                expert_fallbacks.append(float(label.diagnostics["fallback"]))
            phase_counts[phase.name.lower()] += 1
            invalid_labels += int(not label.valid)
            total_transitions += 1
            observation = next_observation
            done = bool(terminated or truncated)

        writer.write_episode(
            episode_id=episode_id,
            scenario_seed=scenario_seed,
            scenario_kind=env.road.config.kind,
            teacher_name=selected.name,
            transitions={key: np.asarray(value) for key, value in buffers.items()},
        )
        env.close()

    train_dataset = load_dataset(output, split="train")
    stats = compute_normalization(train_dataset)
    save_normalization(stats, output / "normalization.json")

    summary = {
        "teacher": selected.name,
        "episodes": episodes,
        "transitions": total_transitions,
        "invalid_labels": invalid_labels,
        "phase_counts": phase_counts,
        "qualification": _qualification_payload(qualification),
        "solver_quality": {
            "mean_latency_ms": (
                float(np.mean(expert_latencies_ms)) if expert_latencies_ms else None
            ),
            "p95_latency_ms": (
                float(np.quantile(expert_latencies_ms, 0.95))
                if expert_latencies_ms
                else None
            ),
            "fallback_rate": (
                float(np.mean(expert_fallbacks)) if expert_fallbacks else None
            ),
        },
    }
    (output / "collection_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def classify_phase(env: ActiveSuspensionEnv, observation: np.ndarray) -> EpisodePhase:
    heights = env.road.wheel_heights(env.vehicle_x)
    contact_threshold = max(1e-5, 3.0 * env.road.config.road_noise_std)
    if np.max(np.abs(heights[:2])) > contact_threshold:
        return EpisodePhase.FRONT_CONTACT
    if np.max(np.abs(heights[2:])) > contact_threshold:
        return EpisodePhase.REAR_CONTACT

    ads = observation[OBSERVATION_SPEC.ads_features]
    peak_height = abs(float(ads[OBSERVATION_SPEC.ads_peak_height]))
    confidence = float(ads[OBSERVATION_SPEC.ads_confidence])
    if peak_height > contact_threshold and confidence > 0.0:
        return EpisodePhase.PREVIEW

    cfg = env.road.config
    last_end = cfg.bump_start + cfg.bump_width
    if cfg.kind == "double_bump":
        last_end += cfg.double_spacing + cfg.bump_width
    if env.vehicle_x - cfg.wheelbase > last_end:
        return EpisodePhase.RECOVERY
    return EpisodePhase.FLAT


def _select_teacher(
    teacher: str,
    env_factory,
    qualification_seeds: list[int],
    allow_unqualified: bool,
    mpc_config_path: str | Path | None = None,
) -> tuple[Expert, QualificationResult | None]:
    if teacher == "preview":
        return PolicyExpert("preview", PreviewRulePolicy()), None
    if teacher == "skyhook":
        return PolicyExpert("skyhook", SkyhookGroundhookPolicy()), None
    if teacher == "mpc":
        from rl_suspension.imitation.mpc_expert import MpcExpert

        expert = MpcExpert.from_config(mpc_config_path)
        result = qualify_expert_against_passive(
            expert,
            env_factory,
            qualification_seeds,
            require_improvement=not allow_unqualified,
        )
        return expert, result
    if teacher != "auto":
        raise ValueError(f"Unknown teacher: {teacher}")
    result = qualify_temporary_expert(
        env_factory=env_factory,
        seeds=qualification_seeds,
        require_improvement=not allow_unqualified,
    )
    return result.expert, result


def _empty_episode_buffers() -> dict[str, list]:
    return {
        "observations": [],
        "actions": [],
        "behavior_actions": [],
        "rewards": [],
        "next_observations": [],
        "terminated": [],
        "truncated": [],
        "episode_ids": [],
        "scenario_seeds": [],
        "phases": [],
        "expert_valid": [],
        "expert_quality": [],
        "expert_status": [],
        "expert_objective": [],
        "expert_iterations": [],
        "expert_latency_ms": [],
        "expert_constraint_margin": [],
        "expert_fallback": [],
        "constraint_violation": [],
    }


def _append_transition(
    buffers: dict[str, list],
    *,
    observation,
    action,
    behavior_action,
    reward,
    next_observation,
    terminated,
    truncated,
    episode_id,
    scenario_seed,
    phase,
    expert_valid,
    expert_quality,
    constraint_violation,
    expert_diagnostics=None,
) -> None:
    diagnostics = expert_diagnostics or {}
    status_codes = {"optimal": 1, "optimal_inaccurate": 2}
    values = {
        "observations": np.asarray(observation, dtype=np.float32),
        "actions": np.asarray(action, dtype=np.float32),
        "behavior_actions": np.asarray(behavior_action, dtype=np.float32),
        "rewards": np.float32(reward),
        "next_observations": np.asarray(next_observation, dtype=np.float32),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "episode_ids": np.int64(episode_id),
        "scenario_seeds": np.int64(scenario_seed),
        "phases": np.uint8(int(phase)),
        "expert_valid": bool(expert_valid),
        "expert_quality": np.float32(expert_quality),
        "expert_status": np.int8(status_codes.get(str(diagnostics.get("status", "")), 0)),
        "expert_objective": np.float64(diagnostics.get("objective", np.nan)),
        "expert_iterations": np.int32(diagnostics.get("iterations", 0)),
        "expert_latency_ms": np.float32(diagnostics.get("latency_ms", np.nan)),
        "expert_constraint_margin": np.float32(
            diagnostics.get("constraint_margin", np.nan)
        ),
        "expert_fallback": bool(diagnostics.get("fallback", False)),
        "constraint_violation": np.float32(constraint_violation),
    }
    for field, value in values.items():
        buffers[field].append(value)


def _qualification_payload(result: QualificationResult | None) -> dict | None:
    if result is None:
        return None
    return {
        "qualified": result.qualified,
        "selected": result.selected.__dict__,
        "passive": result.passive.__dict__,
        "candidates": [item.__dict__ for item in result.candidates],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument(
        "--teacher",
        choices=["mpc", "auto", "preview", "skyhook"],
        default="mpc",
    )
    parser.add_argument("--mpc-config", type=Path, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--curriculum-stage", type=int, default=5)
    parser.add_argument("--qualification-episodes", type=int, default=5)
    parser.add_argument("--allow-unqualified", action="store_true")
    parser.add_argument("--initial-state-std", type=float, default=0.0)
    parser.add_argument("--observation-noise-std", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = collect_dataset(
        output_dir=args.output,
        episodes=args.episodes,
        teacher=args.teacher,
        seed_start=args.seed_start,
        curriculum_stage=args.curriculum_stage,
        qualification_episodes=args.qualification_episodes,
        allow_unqualified=args.allow_unqualified,
        initial_state_std=args.initial_state_std,
        observation_noise_std=args.observation_noise_std,
        mpc_config_path=args.mpc_config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
