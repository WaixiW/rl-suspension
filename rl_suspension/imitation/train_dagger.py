"""Dataset Aggregation and ensemble-gated SafeDAgger training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl_suspension.envs import ActiveSuspensionEnv, EnvConfig
from rl_suspension.imitation.collect import (
    _append_transition,
    _empty_episode_buffers,
    _select_teacher,
    classify_phase,
)
from rl_suspension.imitation.dataset import EpisodeShardWriter, load_dataset
from rl_suspension.imitation.policy import BehaviorClonedPolicy, PolicyEnsemble
from rl_suspension.imitation.train_bc import train_behavior_cloning


def train_dagger(
    dataset_dir: str | Path,
    initial_model_path: str | Path,
    output_dir: str | Path,
    rounds: int = 5,
    episodes_per_round: int = 20,
    betas: list[float] | None = None,
    teacher: str = "mpc",
    allow_unqualified: bool = False,
    bc_loss: str = "huber_smooth",
    bc_epochs: int = 10,
    bc_batches_per_epoch: int = 100,
    seed: int = 0,
    curriculum_stage: int = 5,
    device: str = "auto",
    safe_ensemble_models: list[str | Path] | None = None,
    uncertainty_threshold: float | None = None,
    mpc_config_path: str | Path | None = None,
) -> dict:
    if rounds <= 0 or episodes_per_round <= 0:
        raise ValueError("rounds and episodes_per_round must be positive")
    beta_schedule = betas or np.linspace(1.0, 0.0, rounds).tolist()
    if len(beta_schedule) != rounds:
        raise ValueError("The beta schedule must contain one value per DAgger round")
    if any(beta < 0.0 or beta > 1.0 for beta in beta_schedule):
        raise ValueError("All beta values must be in [0, 1]")

    dataset_root = Path(dataset_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    normalization_path = dataset_root / "normalization.json"

    def env_factory() -> ActiveSuspensionEnv:
        return ActiveSuspensionEnv(
            EnvConfig(
                curriculum_stage=curriculum_stage,
                terminate_on_violation=False,
            )
        )

    selected_teacher, qualification = _select_teacher(
        teacher,
        env_factory,
        list(range(seed + 70_000, seed + 70_005)),
        allow_unqualified,
        mpc_config_path,
    )
    current_model_path = Path(initial_model_path)
    current_policy = BehaviorClonedPolicy.load(current_model_path, normalization_path)

    ensemble: PolicyEnsemble | None = None
    calibrated_threshold = uncertainty_threshold
    if safe_ensemble_models:
        ensemble = PolicyEnsemble(
            [
                BehaviorClonedPolicy.load(model_path, normalization_path)
                for model_path in safe_ensemble_models
            ]
        )
        if calibrated_threshold is None:
            calibrated_threshold = calibrate_uncertainty(
                ensemble,
                dataset_root,
            )

    writer = EpisodeShardWriter(dataset_root)
    rng = np.random.default_rng(seed)
    round_reports: list[dict] = []

    for round_index, beta in enumerate(beta_schedule):
        queried = 0
        visited = 0
        interventions = 0
        collected = 0

        for episode_offset in range(episodes_per_round):
            episode_id = len(writer.records)
            scenario_seed = seed + 100_000 + round_index * episodes_per_round + episode_offset
            env = env_factory()
            observation, _ = env.reset(
                seed=scenario_seed,
                options={"curriculum_stage": curriculum_stage},
            )
            reset_teacher = getattr(selected_teacher, "reset", None)
            if callable(reset_teacher):
                reset_teacher()
            buffers = _empty_episode_buffers()
            done = False

            while not done:
                visited += 1
                phase = classify_phase(env, observation)
                should_query = True
                uncertainty = 0.0

                if ensemble is not None:
                    prediction = ensemble.predict_with_uncertainty(observation)
                    student_action = prediction.action
                    uncertainty = prediction.uncertainty
                    should_query = uncertainty >= float(calibrated_threshold)
                else:
                    student_action = current_policy.predict(
                        observation,
                        deterministic=True,
                    )[0]

                expert_label = selected_teacher.predict(observation) if should_query else None
                queried += int(should_query)

                if ensemble is not None:
                    if should_query and expert_label is not None and expert_label.valid:
                        behavior_action = expert_label.action
                        interventions += 1
                    else:
                        behavior_action = student_action
                else:
                    assert expert_label is not None
                    use_expert = bool(rng.random() < beta)
                    behavior_action = expert_label.action if use_expert else student_action
                    interventions += int(use_expert)

                next_observation, reward, terminated, truncated, info = env.step(
                    behavior_action
                )
                if expert_label is not None:
                    _append_transition(
                        buffers,
                        observation=observation,
                        action=expert_label.action,
                        behavior_action=behavior_action,
                        reward=reward,
                        next_observation=next_observation,
                        terminated=terminated,
                        truncated=truncated,
                        episode_id=episode_id,
                        scenario_seed=scenario_seed,
                        phase=phase,
                        expert_valid=expert_label.valid,
                        expert_quality=expert_label.quality,
                        expert_diagnostics=expert_label.diagnostics,
                        constraint_violation=float(
                            sum(info["constraint_violations"].values())
                        ),
                    )
                    collected += 1

                observation = next_observation
                done = bool(terminated or truncated)

            if buffers["observations"]:
                writer.write_episode(
                    episode_id=episode_id,
                    scenario_seed=scenario_seed,
                    scenario_kind=env.road.config.kind,
                    teacher_name=selected_teacher.name,
                    transitions={
                        key: np.asarray(value)
                        for key, value in buffers.items()
                    },
                )
            env.close()

        round_dir = output / f"round_{round_index + 1:02d}"
        if collected:
            metrics = train_behavior_cloning(
                dataset_dir=dataset_root,
                output_dir=round_dir,
                loss_name=bc_loss,
                epochs=bc_epochs,
                batches_per_epoch=bc_batches_per_epoch,
                seed=seed + round_index + 1,
                device=device,
                curriculum_stage=curriculum_stage,
                initialize_model_path=current_model_path,
            )
            current_model_path = round_dir / "bc_sac_actor.zip"
            current_policy = BehaviorClonedPolicy.load(
                current_model_path,
                normalization_path,
            )
        else:
            metrics = {"skipped_training": True}

        report = {
            "round": round_index + 1,
            "beta": float(beta),
            "visited": visited,
            "queried": queried,
            "collected": collected,
            "interventions": interventions,
            "query_rate": queried / max(visited, 1),
            "intervention_rate": interventions / max(visited, 1),
            "uncertainty_threshold": calibrated_threshold,
            "metrics": metrics,
        }
        round_reports.append(report)
        (round_dir / "dagger_round.json").parent.mkdir(parents=True, exist_ok=True)
        (round_dir / "dagger_round.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

    summary = {
        "teacher": selected_teacher.name,
        "qualification": (
            {
                "qualified": qualification.qualified,
                "selected": qualification.selected.__dict__,
                "passive": qualification.passive.__dict__,
            }
            if qualification is not None
            else None
        ),
        "mode": "safe_dagger" if ensemble is not None else "dagger",
        "final_model_path": str(current_model_path),
        "rounds": round_reports,
    }
    (output / "dagger_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def calibrate_uncertainty(
    ensemble: PolicyEnsemble,
    dataset_dir: str | Path,
    quantile: float = 0.90,
    max_samples: int = 5000,
) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    try:
        dataset = load_dataset(dataset_dir, split="validation")
    except ValueError:
        dataset = load_dataset(dataset_dir, split="train")
    valid = np.flatnonzero(dataset.expert_valid)[:max_samples]
    uncertainties = [
        ensemble.predict_with_uncertainty(dataset.observations[index]).uncertainty
        for index in valid
    ]
    if not uncertainties:
        raise ValueError("Cannot calibrate uncertainty without valid samples")
    return float(np.quantile(uncertainties, quantile))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--episodes-per-round", type=int, default=20)
    parser.add_argument("--beta", default=None, help="Comma-separated beta schedule")
    parser.add_argument(
        "--teacher",
        choices=["mpc", "auto", "preview", "skyhook"],
        default="mpc",
    )
    parser.add_argument("--mpc-config", type=Path, default=None)
    parser.add_argument("--allow-unqualified", action="store_true")
    parser.add_argument("--bc-loss", default="huber_smooth")
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--bc-batches-per-epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curriculum-stage", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ensemble-model", type=Path, action="append", default=[])
    parser.add_argument("--uncertainty-threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    betas = (
        [float(item) for item in args.beta.split(",")]
        if args.beta is not None
        else None
    )
    summary = train_dagger(
        dataset_dir=args.dataset,
        initial_model_path=args.initial_model,
        output_dir=args.output,
        rounds=args.rounds,
        episodes_per_round=args.episodes_per_round,
        betas=betas,
        teacher=args.teacher,
        allow_unqualified=args.allow_unqualified,
        bc_loss=args.bc_loss,
        bc_epochs=args.bc_epochs,
        bc_batches_per_epoch=args.bc_batches_per_epoch,
        seed=args.seed,
        curriculum_stage=args.curriculum_stage,
        device=args.device,
        safe_ensemble_models=args.ensemble_model or None,
        uncertainty_threshold=args.uncertainty_threshold,
        mpc_config_path=args.mpc_config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
