"""Framework-neutral PyTorch behavior-cloning trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    DEFAULT_OBSERVATION_SCHEMA,
    ActionSchema,
    ObservationSchema,
)
from rl_suspension.production.training.losses import (
    BCLossConfig,
    BehaviorCloningLoss,
)


@dataclass
class BCDatasetArrays:
    """In-memory arrays independent of any environment or RL framework."""

    states: NDArray[np.floating]
    roads: NDArray[np.floating]
    actions: NDArray[np.floating]
    phases: NDArray[np.integer] | None = None
    quality: NDArray[np.floating] | None = None
    episode_ids: NDArray[np.integer] | None = None
    sequence_indices: NDArray[np.integer] | None = None
    valid: NDArray[np.bool_] | None = None
    state_history: NDArray[np.floating] | None = None
    road_history: NDArray[np.floating] | None = None
    actions_normalized: bool = False
    observation_schema: ObservationSchema = field(
        default_factory=lambda: DEFAULT_OBSERVATION_SCHEMA
    )
    action_schema: ActionSchema = field(default_factory=lambda: DEFAULT_ACTION_SCHEMA)

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states, dtype=np.float32)
        self.roads = np.asarray(self.roads, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        count = self.states.shape[0]
        schema = self.observation_schema
        if self.states.shape != (count, schema.state_vector_dim):
            raise ValueError(
                f"states must have shape (N, {schema.state_vector_dim})"
            )
        if self.roads.shape != (count, schema.road_channels, schema.road_points):
            raise ValueError(
                "roads must have shape "
                f"(N, {schema.road_channels}, {schema.road_points})"
            )
        if self.actions.shape != (count, self.action_schema.dimension):
            raise ValueError(
                f"actions must have shape (N, {self.action_schema.dimension})"
            )
        if count == 0:
            raise ValueError("behavior-cloning dataset cannot be empty")

        self.phases = self._default_or_array(self.phases, np.int64, 0)
        self.quality = self._default_or_array(self.quality, np.float32, 1.0)
        self.episode_ids = self._default_or_array(self.episode_ids, np.int64, 0)
        self.sequence_indices = self._default_or_array(
            self.sequence_indices,
            np.int64,
            np.arange(count, dtype=np.int64),
        )
        self.valid = self._default_or_array(self.valid, np.bool_, True)

        if self.state_history is not None or self.road_history is not None:
            if self.state_history is None or self.road_history is None:
                raise ValueError("state_history and road_history must be provided together")
            self.state_history = np.asarray(self.state_history, dtype=np.float32)
            self.road_history = np.asarray(self.road_history, dtype=np.float32)
            if self.state_history.ndim != 3 or self.state_history.shape[0] != count:
                raise ValueError("state_history must have shape (N,T,D)")
            if self.state_history.shape[-1] != schema.state_vector_dim:
                raise ValueError("state_history has an invalid state dimension")
            expected_road = (
                count,
                self.state_history.shape[1],
                schema.road_channels,
                schema.road_points,
            )
            if self.road_history.shape != expected_road:
                raise ValueError(f"road_history must have shape {expected_road}")

        arrays = (self.states, self.roads, self.actions, self.quality)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("dataset contains NaN or Inf")
        if self.actions_normalized and (
            np.any(self.actions < -1e-6) or np.any(self.actions > 1.0 + 1e-6)
        ):
            raise ValueError("normalized actions must lie within [0, 1]")
        if np.any(self.quality < 0.0):
            raise ValueError("quality must be nonnegative")

    def _default_or_array(self, value, dtype, default):
        count = self.states.shape[0]
        if value is None:
            if np.ndim(default) == 0:
                return np.full(count, default, dtype=dtype)
            result = np.asarray(default, dtype=dtype)
        else:
            result = np.asarray(value, dtype=dtype)
        if result.shape != (count,):
            raise ValueError("per-sample metadata must have shape (N,)")
        return result

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def subset(self, indices: NDArray[np.integer]) -> "BCDatasetArrays":
        selected = np.asarray(indices, dtype=np.int64)
        return BCDatasetArrays(
            states=self.states[selected],
            roads=self.roads[selected],
            actions=self.actions[selected],
            phases=self.phases[selected],
            quality=self.quality[selected],
            episode_ids=self.episode_ids[selected],
            sequence_indices=self.sequence_indices[selected],
            valid=self.valid[selected],
            state_history=(
                None if self.state_history is None else self.state_history[selected]
            ),
            road_history=(
                None if self.road_history is None else self.road_history[selected]
            ),
            actions_normalized=self.actions_normalized,
            observation_schema=self.observation_schema,
            action_schema=self.action_schema,
        )

    def fixed_copy(self) -> "BCDatasetArrays":
        """Return a deep copy suitable for a fixed validation holdout."""

        return self.subset(np.arange(len(self), dtype=np.int64))


def concatenate_datasets(
    datasets: Sequence[BCDatasetArrays],
) -> BCDatasetArrays:
    items = list(datasets)
    if not items:
        raise ValueError("at least one dataset is required")
    first = items[0]
    if any(item.actions_normalized != first.actions_normalized for item in items):
        raise ValueError("all datasets must use the same action normalization")
    has_history = first.state_history is not None
    if any((item.state_history is not None) != has_history for item in items):
        raise ValueError("all datasets must agree on temporal history")
    if any(item.observation_schema != first.observation_schema for item in items):
        raise ValueError("all datasets must use the same observation schema")
    if any(item.action_schema != first.action_schema for item in items):
        raise ValueError("all datasets must use the same action schema")

    return BCDatasetArrays(
        states=np.concatenate([item.states for item in items], axis=0),
        roads=np.concatenate([item.roads for item in items], axis=0),
        actions=np.concatenate([item.actions for item in items], axis=0),
        phases=np.concatenate([item.phases for item in items], axis=0),
        quality=np.concatenate([item.quality for item in items], axis=0),
        episode_ids=np.concatenate([item.episode_ids for item in items], axis=0),
        sequence_indices=np.concatenate(
            [item.sequence_indices for item in items], axis=0
        ),
        valid=np.concatenate([item.valid for item in items], axis=0),
        state_history=(
            np.concatenate([item.state_history for item in items], axis=0)
            if has_history
            else None
        ),
        road_history=(
            np.concatenate([item.road_history for item in items], axis=0)
            if has_history
            else None
        ),
        actions_normalized=first.actions_normalized,
        observation_schema=first.observation_schema,
        action_schema=first.action_schema,
    )


@dataclass(frozen=True)
class BCTrainerConfig:
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    gradient_clip_norm: float = 10.0
    seed: int = 0
    device: str = "cpu"
    restore_best: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer settings")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    training_loss: float
    validation_loss: float
    behavior_cloning_loss: float
    action_delta_loss: float


class CheckpointSelector(Protocol):
    def __call__(self, metrics: EpochMetrics) -> bool: ...


class ValidationLossSelector:
    """Select checkpoints when fixed-validation loss improves."""

    def __init__(self, min_delta: float = 0.0) -> None:
        if min_delta < 0.0:
            raise ValueError("min_delta must be nonnegative")
        self.min_delta = float(min_delta)
        self.best = float("inf")

    def __call__(self, metrics: EpochMetrics) -> bool:
        if metrics.validation_loss < self.best - self.min_delta:
            self.best = metrics.validation_loss
            return True
        return False


@dataclass(frozen=True)
class SelectedCheckpoint:
    epoch: int
    validation_loss: float
    model_state: dict[str, Tensor]
    optimizer_state: dict


class CheckpointHook(Protocol):
    def __call__(self, checkpoint: SelectedCheckpoint) -> None: ...


class FileCheckpointHook:
    """Persist checkpoints chosen by a selector without owning selection policy."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __call__(self, checkpoint: SelectedCheckpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": checkpoint.epoch,
                "validation_loss": checkpoint.validation_loss,
                "model_state_dict": checkpoint.model_state,
                "optimizer_state_dict": checkpoint.optimizer_state,
            },
            self.path,
        )


@dataclass(frozen=True)
class BCTrainingResult:
    history: tuple[EpochMetrics, ...]
    best_epoch: int
    best_validation_loss: float


class BCTrainer:
    """Train any compatible PyTorch actor without binding to an RL library."""

    def __init__(
        self,
        model: nn.Module,
        *,
        config: BCTrainerConfig | None = None,
        loss: BehaviorCloningLoss | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.config = config or BCTrainerConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.loss = loss or BehaviorCloningLoss()
        self.loss.to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def fit(
        self,
        training: BCDatasetArrays,
        validation: BCDatasetArrays,
        *,
        checkpoint_selector: CheckpointSelector | None = None,
        checkpoint_hooks: Iterable[CheckpointHook] = (),
    ) -> BCTrainingResult:
        if training.actions_normalized != validation.actions_normalized:
            raise ValueError("training and validation action normalization must match")
        selector = checkpoint_selector or ValidationLossSelector()
        hooks = tuple(checkpoint_hooks)
        generator = np.random.default_rng(self.config.seed)
        torch.manual_seed(self.config.seed)
        history: list[EpochMetrics] = []
        best_checkpoint: SelectedCheckpoint | None = None

        valid_training = np.flatnonzero(training.valid)
        valid_validation = np.flatnonzero(validation.valid)
        if valid_training.size == 0 or valid_validation.size == 0:
            raise ValueError("training and validation require valid expert samples")

        for epoch in range(1, self.config.epochs + 1):
            generator.shuffle(valid_training)
            totals: list[float] = []
            bc_losses: list[float] = []
            delta_losses: list[float] = []
            self.model.train()
            for offset in range(0, valid_training.size, self.config.batch_size):
                # Sorting preserves temporal adjacency when neighboring transitions
                # happen to be sampled into the same minibatch.
                indices = np.sort(
                    valid_training[offset : offset + self.config.batch_size]
                )
                output = self._loss_for_indices(training, indices)
                self.optimizer.zero_grad(set_to_none=True)
                output.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )
                self.optimizer.step()
                totals.append(float(output.total.detach().cpu()))
                bc_losses.append(float(output.behavior_cloning.detach().cpu()))
                delta_losses.append(float(output.action_delta.detach().cpu()))

            validation_loss = self.evaluate(validation, valid_validation)
            metrics = EpochMetrics(
                epoch=epoch,
                training_loss=float(np.mean(totals)),
                validation_loss=validation_loss,
                behavior_cloning_loss=float(np.mean(bc_losses)),
                action_delta_loss=float(np.mean(delta_losses)),
            )
            history.append(metrics)
            if selector(metrics):
                best_checkpoint = self._checkpoint(metrics)
                for hook in hooks:
                    hook(best_checkpoint)

        if best_checkpoint is None:
            raise RuntimeError("checkpoint selector did not select any epoch")
        if self.config.restore_best:
            self.model.load_state_dict(best_checkpoint.model_state)
        self.model.eval()
        return BCTrainingResult(
            history=tuple(history),
            best_epoch=best_checkpoint.epoch,
            best_validation_loss=best_checkpoint.validation_loss,
        )

    def evaluate(
        self,
        dataset: BCDatasetArrays,
        indices: NDArray[np.integer] | None = None,
    ) -> float:
        selected = (
            np.flatnonzero(dataset.valid)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if selected.size == 0:
            raise ValueError("evaluation requires at least one valid sample")
        losses: list[float] = []
        weights: list[int] = []
        self.model.eval()
        with torch.no_grad():
            for offset in range(0, selected.size, self.config.batch_size):
                batch = np.sort(selected[offset : offset + self.config.batch_size])
                output = self._loss_for_indices(dataset, batch)
                losses.append(float(output.total.cpu()))
                weights.append(batch.size)
        return float(np.average(losses, weights=weights))

    def _loss_for_indices(
        self,
        dataset: BCDatasetArrays,
        indices: NDArray[np.integer],
    ):
        states = (
            dataset.states[indices]
            if dataset.state_history is None
            else dataset.state_history[indices]
        )
        roads = (
            dataset.roads[indices]
            if dataset.road_history is None
            else dataset.road_history[indices]
        )
        predicted = self.model(
            torch.as_tensor(states, device=self.device),
            torch.as_tensor(roads, device=self.device),
        )
        target = torch.as_tensor(dataset.actions[indices], device=self.device)
        target = self._target_for_loss(target, dataset.actions_normalized)
        return self.loss(
            predicted,
            target,
            torch.as_tensor(dataset.phases[indices], device=self.device),
            torch.as_tensor(dataset.quality[indices], device=self.device),
            torch.as_tensor(dataset.episode_ids[indices], device=self.device),
            torch.as_tensor(dataset.sequence_indices[indices], device=self.device),
        )

    def _target_for_loss(self, target: Tensor, dataset_normalized: bool) -> Tensor:
        loss_normalized = self.loss.config.target_is_normalized
        if dataset_normalized == loss_normalized:
            return target
        minimum = torch.as_tensor(
            self.loss.action_schema.minimum,
            dtype=target.dtype,
            device=target.device,
        )
        maximum = torch.as_tensor(
            self.loss.action_schema.maximum,
            dtype=target.dtype,
            device=target.device,
        )
        if dataset_normalized:
            return minimum + target * (maximum - minimum)
        return (target - minimum) / (maximum - minimum).clamp_min(1e-12)

    def _checkpoint(self, metrics: EpochMetrics) -> SelectedCheckpoint:
        state = {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }
        return SelectedCheckpoint(
            epoch=metrics.epoch,
            validation_loss=metrics.validation_loss,
            model_state=state,
            optimizer_state=self.optimizer.state_dict(),
        )


def default_bc_loss(
    *,
    target_is_normalized: bool = False,
    action_delta_coefficient: float = 0.05,
) -> BehaviorCloningLoss:
    """Construct the production BC objective with explicit target units."""

    return BehaviorCloningLoss(
        BCLossConfig(
            target_is_normalized=target_is_normalized,
            action_delta_coefficient=action_delta_coefficient,
        )
    )
