"""Run reproducible BC, DAgger, and SafeDAgger comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from rl_suspension.baselines import PassivePolicy, PreviewRulePolicy, SkyhookGroundhookPolicy
from rl_suspension.envs import ActiveSuspensionEnv, EnvConfig
from rl_suspension.evaluation.evaluate import evaluate_policy_detailed
from rl_suspension.imitation.collect import _select_teacher
from rl_suspension.imitation.policy import (
    BehaviorClonedPolicy,
    PolicyEnsemble,
    SafeDAggerPolicy,
)
from rl_suspension.imitation.train_bc import LOSS_NAMES, train_behavior_cloning
from rl_suspension.imitation.train_dagger import calibrate_uncertainty, train_dagger


def run_benchmark(
    dataset_dir: str | Path,
    output_dir: str | Path,
    episodes: int = 20,
    bc_epochs: int = 30,
    bc_batches_per_epoch: int = 200,
    dagger_rounds: int = 5,
    dagger_episodes_per_round: int = 20,
    seed: int = 0,
    curriculum_stage: int = 5,
    device: str = "auto",
    allow_unqualified: bool = False,
    teacher: str = "mpc",
    mpc_config_path: str | Path | None = None,
) -> dict:
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

    expert, qualification = _select_teacher(
        teacher,
        env_factory,
        list(range(seed + 90_000, seed + 90_005)),
        allow_unqualified,
        mpc_config_path,
    )
    if hasattr(expert, "policy"):
        expert_policy = expert.policy
    else:
        from rl_suspension.imitation.mpc_expert import MpcPolicy

        expert_policy = MpcPolicy(expert)
    results: dict[str, dict] = {}

    policies = {
        "passive": PassivePolicy(),
        "skyhook": SkyhookGroundhookPolicy(),
        "preview": PreviewRulePolicy(),
        f"qualified_teacher:{expert.name}": expert_policy,
    }
    for name, policy in policies.items():
        results[name] = evaluate_policy_detailed(
            policy,
            episodes=episodes,
            curriculum_stage=curriculum_stage,
            seed=seed + 120_000,
            expert=expert,
            terminate_on_violation=False,
        )

    bc_models: dict[str, Path] = {}
    for loss_index, loss_name in enumerate(LOSS_NAMES):
        bc_dir = output / "bc" / loss_name
        train_behavior_cloning(
            dataset_dir=dataset_root,
            output_dir=bc_dir,
            loss_name=loss_name,
            epochs=bc_epochs,
            batches_per_epoch=bc_batches_per_epoch,
            seed=seed + loss_index,
            device=device,
            curriculum_stage=curriculum_stage,
        )
        model_path = bc_dir / "bc_sac_actor.zip"
        bc_models[loss_name] = model_path
        policy = BehaviorClonedPolicy.load(model_path, normalization_path)
        results[f"bc:{loss_name}"] = evaluate_policy_detailed(
            policy,
            episodes=episodes,
            curriculum_stage=curriculum_stage,
            seed=seed + 120_000,
            expert=expert,
            terminate_on_violation=False,
        )

    best_loss = max(
        LOSS_NAMES,
        key=lambda loss: results[f"bc:{loss}"]["mean_episode_return"],
    )
    best_model = bc_models[best_loss]
    dagger_dataset = output / "datasets" / "dagger"
    safe_dataset = output / "datasets" / "safe_dagger"
    shutil.copytree(dataset_root, dagger_dataset, dirs_exist_ok=True)
    shutil.copytree(dataset_root, safe_dataset, dirs_exist_ok=True)
    dagger_summary = train_dagger(
        dataset_dir=dagger_dataset,
        initial_model_path=best_model,
        output_dir=output / "dagger",
        rounds=dagger_rounds,
        episodes_per_round=dagger_episodes_per_round,
        teacher=teacher,
        allow_unqualified=allow_unqualified,
        bc_loss=best_loss,
        bc_epochs=max(2, bc_epochs // 3),
        bc_batches_per_epoch=max(5, bc_batches_per_epoch // 2),
        seed=seed + 100,
        curriculum_stage=curriculum_stage,
        device=device,
        mpc_config_path=mpc_config_path,
    )
    dagger_policy = BehaviorClonedPolicy.load(
        dagger_summary["final_model_path"],
        normalization_path,
    )
    results["dagger"] = evaluate_policy_detailed(
        dagger_policy,
        episodes=episodes,
        curriculum_stage=curriculum_stage,
        seed=seed + 120_000,
        expert=expert,
        terminate_on_violation=False,
    )

    ensemble_models: list[Path] = []
    for member in range(3):
        member_dir = output / "safe_dagger" / "ensemble" / f"member_{member}"
        train_behavior_cloning(
            dataset_dir=dataset_root,
            output_dir=member_dir,
            loss_name=best_loss,
            epochs=bc_epochs,
            batches_per_epoch=bc_batches_per_epoch,
            seed=seed + 200 + member,
            device=device,
            curriculum_stage=curriculum_stage,
        )
        ensemble_models.append(member_dir / "bc_sac_actor.zip")

    ensemble = PolicyEnsemble(
        [
            BehaviorClonedPolicy.load(model, normalization_path)
            for model in ensemble_models
        ]
    )
    threshold = calibrate_uncertainty(ensemble, safe_dataset)
    gated_policy = SafeDAggerPolicy(ensemble, expert, threshold)
    results["safe_dagger:gated_ensemble"] = evaluate_policy_detailed(
        gated_policy,
        episodes=episodes,
        curriculum_stage=curriculum_stage,
        seed=seed + 120_000,
        expert=expert,
        terminate_on_violation=False,
    )

    safe_summary = train_dagger(
        dataset_dir=safe_dataset,
        initial_model_path=ensemble_models[0],
        output_dir=output / "safe_dagger" / "aggregation",
        rounds=dagger_rounds,
        episodes_per_round=dagger_episodes_per_round,
        teacher=teacher,
        allow_unqualified=allow_unqualified,
        bc_loss=best_loss,
        bc_epochs=max(2, bc_epochs // 3),
        bc_batches_per_epoch=max(5, bc_batches_per_epoch // 2),
        seed=seed + 300,
        curriculum_stage=curriculum_stage,
        device=device,
        safe_ensemble_models=ensemble_models,
        uncertainty_threshold=threshold,
        mpc_config_path=mpc_config_path,
    )
    safe_policy = BehaviorClonedPolicy.load(
        safe_summary["final_model_path"],
        normalization_path,
    )
    results["safe_dagger:retrained_actor"] = evaluate_policy_detailed(
        safe_policy,
        episodes=episodes,
        curriculum_stage=curriculum_stage,
        seed=seed + 120_000,
        expert=expert,
        terminate_on_violation=False,
    )

    summary = {
        "qualified_teacher": expert.name,
        "teacher_qualified": qualification.qualified if qualification is not None else None,
        "best_bc_loss": best_loss,
        "uncertainty_threshold": threshold,
        "results": results,
        "dagger": dagger_summary,
        "safe_dagger": safe_summary,
    }
    (output / "benchmark.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _write_csv(results, output / "benchmark.csv")
    return summary


def _write_csv(results: dict[str, dict], path: Path) -> None:
    rows: list[dict] = []
    for name, result in results.items():
        row = {
            "policy": name,
            "mean_episode_return": result["mean_episode_return"],
            "mean_inference_ms": result["mean_inference_ms"],
            "expert_action_mse": result["expert_action_mse"],
            "expert_query_rate": result["expert_query_rate"],
            **result["metrics"],
        }
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--bc-batches-per-epoch", type=int, default=200)
    parser.add_argument("--dagger-rounds", type=int, default=5)
    parser.add_argument("--dagger-episodes-per-round", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curriculum-stage", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-unqualified", action="store_true")
    parser.add_argument(
        "--teacher",
        choices=["mpc", "auto", "preview", "skyhook"],
        default="mpc",
    )
    parser.add_argument("--mpc-config", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_benchmark(
        dataset_dir=args.dataset,
        output_dir=args.output,
        episodes=args.episodes,
        bc_epochs=args.bc_epochs,
        bc_batches_per_epoch=args.bc_batches_per_epoch,
        dagger_rounds=args.dagger_rounds,
        dagger_episodes_per_round=args.dagger_episodes_per_round,
        seed=args.seed,
        curriculum_stage=args.curriculum_stage,
        device=args.device,
        allow_unqualified=args.allow_unqualified,
        teacher=args.teacher,
        mpc_config_path=args.mpc_config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
