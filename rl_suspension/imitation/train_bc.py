"""Behavior-clone the actual Stable-Baselines3 SAC actor."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from rl_suspension.envs import ActiveSuspensionEnv, EnvConfig
from rl_suspension.evaluation.evaluate import evaluate_policy
from rl_suspension.imitation.dataset import (
    PhaseBalancedSampler,
    load_dataset,
    load_normalization,
)
from rl_suspension.imitation.policy import BehaviorClonedPolicy


LOSS_NAMES = ("mse", "huber", "huber_smooth")


def train_behavior_cloning(
    dataset_dir: str | Path,
    output_dir: str | Path,
    loss_name: str = "huber_smooth",
    epochs: int = 30,
    batches_per_epoch: int = 200,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    seed: int = 0,
    device: str = "auto",
    curriculum_stage: int = 5,
    initialize_model_path: str | Path | None = None,
) -> dict:
    if loss_name not in LOSS_NAMES:
        raise ValueError(f"loss_name must be one of {LOSS_NAMES}")
    try:
        import torch
        import torch.nn.functional as functional
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise RuntimeError("PyTorch and stable-baselines3 are required for BC") from exc

    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset_root = Path(dataset_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_data = load_dataset(dataset_root, split="train")
    try:
        validation_data = load_dataset(dataset_root, split="validation")
    except ValueError:
        unique_seeds = np.unique(train_data.scenario_seeds)
        if unique_seeds.size >= 2:
            holdout_seed = int(np.min(unique_seeds))
            validation_mask = train_data.scenario_seeds == holdout_seed
            validation_data = train_data.subset(np.flatnonzero(validation_mask))
            train_data = train_data.subset(np.flatnonzero(~validation_mask))
        else:
            validation_data = train_data
    normalization_path = dataset_root / "normalization.json"
    normalization = load_normalization(normalization_path)
    sampler = PhaseBalancedSampler(train_data, seed=seed)

    env = ActiveSuspensionEnv(EnvConfig(curriculum_stage=curriculum_stage))
    if initialize_model_path is None:
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": [256, 256]},
            learning_rate=learning_rate,
            buffer_size=max(batch_size, 1024),
            learning_starts=0,
            batch_size=batch_size,
            seed=seed,
            device=device,
            verbose=0,
        )
    else:
        model = SAC.load(initialize_model_path, env=env, device=device)
    actor = model.policy.actor
    actor.train()
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    target_log_std = -2.5
    history: list[dict[str, float | int]] = []
    best_validation = _validation_loss(
        actor=actor,
        dataset=validation_data,
        normalization=normalization,
        device=model.device,
        torch_module=torch,
    )
    best_state: dict | None = {
        key: value.detach().cpu().clone()
        for key, value in actor.state_dict().items()
    }

    for epoch in range(epochs):
        batch_losses: list[float] = []
        for _ in range(batches_per_epoch):
            indices = sampler.sample_indices(batch_size)
            obs = normalization.normalize(train_data.observations[indices])
            actions = train_data.actions[indices]
            quality = np.clip(train_data.expert_quality[indices], 0.05, 10.0)

            obs_tensor = torch.as_tensor(obs, device=model.device)
            action_tensor = torch.as_tensor(actions, device=model.device)
            quality_tensor = torch.as_tensor(quality, device=model.device)
            quality_tensor = quality_tensor / quality_tensor.mean().clamp_min(1e-6)

            predicted = actor(obs_tensor, deterministic=True)
            imitation_per_sample = _imitation_loss(
                predicted,
                action_tensor,
                loss_name,
                functional,
            )
            loss = (quality_tensor * imitation_per_sample).mean()

            if loss_name == "huber_smooth":
                previous_indices = _previous_indices(train_data.episode_ids, indices)
                previous_obs = normalization.normalize(
                    train_data.observations[previous_indices]
                )
                previous_actions = train_data.actions[previous_indices]
                previous_predicted = actor(
                    torch.as_tensor(previous_obs, device=model.device),
                    deterministic=True,
                )
                predicted_delta = predicted - previous_predicted
                target_delta = action_tensor - torch.as_tensor(
                    previous_actions,
                    device=model.device,
                )
                smooth_loss = functional.smooth_l1_loss(
                    predicted_delta,
                    target_delta,
                )
                loss = loss + 0.05 * smooth_loss

            _, log_std, _ = actor.get_action_dist_params(obs_tensor)
            variance_loss = functional.mse_loss(
                log_std,
                torch.full_like(log_std, target_log_std),
            )
            loss = loss + 0.01 * variance_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        validation_loss = _validation_loss(
            actor=actor,
            dataset=validation_data,
            normalization=normalization,
            device=model.device,
            torch_module=torch,
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(batch_losses)),
            "validation_mse": validation_loss,
        }
        history.append(record)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in actor.state_dict().items()
            }

    if best_state is not None:
        actor.load_state_dict(best_state)
    actor.eval()
    model_path = output / "bc_sac_actor"
    model.save(model_path)
    shutil.copy2(normalization_path, output / "normalization.json")

    policy = BehaviorClonedPolicy(
        model=model,
        normalization=normalization,
    )
    closed_loop = evaluate_policy(
        policy,
        episodes=5,
        curriculum_stage=curriculum_stage,
        seed=80_000 + seed * 100,
        terminate_on_violation=False,
    )
    metadata = {
        "loss_name": loss_name,
        "epochs": epochs,
        "batches_per_epoch": batches_per_epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "initialize_model_path": (
            str(initialize_model_path) if initialize_model_path is not None else None
        ),
        "best_validation_mse": best_validation,
        "closed_loop": closed_loop.to_dict(),
        "history": history,
        "model_path": str(model_path.with_suffix(".zip")),
    }
    (output / "bc_metrics.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    env.close()
    return metadata


def _imitation_loss(predicted, target, loss_name: str, functional):
    if loss_name == "mse":
        return ((predicted - target) ** 2).mean(dim=1)
    return functional.smooth_l1_loss(
        predicted,
        target,
        reduction="none",
    ).mean(dim=1)


def _previous_indices(
    episode_ids: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    previous = np.maximum(indices - 1, 0)
    same_episode = episode_ids[previous] == episode_ids[indices]
    return np.where(same_episode, previous, indices)


def _validation_loss(
    actor,
    dataset,
    normalization,
    device,
    torch_module,
) -> float:
    valid_indices = np.flatnonzero(dataset.expert_valid)
    if valid_indices.size > 20_000:
        valid_indices = valid_indices[:20_000]
    observations = normalization.normalize(dataset.observations[valid_indices])
    targets = dataset.actions[valid_indices]
    with torch_module.no_grad():
        predicted = actor(
            torch_module.as_tensor(observations, device=device),
            deterministic=True,
        ).cpu().numpy()
    return float(np.mean((predicted - targets) ** 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loss", choices=LOSS_NAMES, default="huber_smooth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batches-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--curriculum-stage", type=int, default=5)
    parser.add_argument("--initialize-model", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_behavior_cloning(
        dataset_dir=args.dataset,
        output_dir=args.output,
        loss_name=args.loss,
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        curriculum_stage=args.curriculum_stage,
        initialize_model_path=args.initialize_model,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
