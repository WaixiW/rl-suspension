"""Losses and sample weighting for direct-action behavior cloning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    ActionSchema,
)


def _schema_tensor(
    values: Sequence[float],
    reference: Tensor,
) -> Tensor:
    return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)


def normalize_physical_actions(actions: Tensor, schema: ActionSchema) -> Tensor:
    """Normalize physical action tensors channel by channel."""

    if actions.shape[-1] != schema.dimension:
        raise ValueError(f"actions must end in {schema.dimension} channels")
    minimum = _schema_tensor(schema.minimum, actions)
    maximum = _schema_tensor(schema.maximum, actions)
    return (actions - minimum) / (maximum - minimum).clamp_min(1e-12)


def phase_quality_weights(
    phases: Tensor,
    quality: Tensor,
    *,
    phase_weights: Mapping[int, float] | Sequence[float] | None = None,
    minimum_quality: float = 0.05,
    maximum_quality: float = 10.0,
    normalize: bool = True,
) -> Tensor:
    """Combine event-phase and expert-quality weights for each sample."""

    if phases.ndim != 1 or quality.ndim != 1 or phases.shape != quality.shape:
        raise ValueError("phases and quality must be same-length one-dimensional tensors")
    if minimum_quality <= 0.0 or maximum_quality < minimum_quality:
        raise ValueError("invalid quality bounds")

    clipped_quality = quality.clamp(minimum_quality, maximum_quality)
    if phase_weights is None:
        phase_weight = torch.ones_like(clipped_quality)
    elif isinstance(phase_weights, Mapping):
        phase_weight = torch.ones_like(clipped_quality)
        for phase, weight in phase_weights.items():
            if weight < 0.0:
                raise ValueError("phase weights must be nonnegative")
            phase_weight = torch.where(
                phases == int(phase),
                torch.as_tensor(weight, dtype=quality.dtype, device=quality.device),
                phase_weight,
            )
    else:
        phase_values = torch.as_tensor(
            tuple(phase_weights),
            dtype=quality.dtype,
            device=quality.device,
        )
        if phase_values.numel() == 0 or bool(torch.any(phase_values < 0.0)):
            raise ValueError("phase weights must be a nonempty nonnegative sequence")
        if bool(torch.any(phases < 0)) or bool(torch.any(phases >= phase_values.numel())):
            raise ValueError("phase index has no corresponding phase weight")
        phase_weight = phase_values[phases.long()]

    combined = phase_weight * clipped_quality
    if normalize:
        combined = combined / combined.mean().clamp_min(1e-12)
    return combined


def normalized_weighted_huber_loss(
    predicted_normalized: Tensor,
    target: Tensor,
    *,
    action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
    target_is_normalized: bool = False,
    channel_weights: Sequence[float] | Tensor | None = None,
    sample_weights: Tensor | None = None,
    delta: float = 0.05,
) -> Tensor:
    """Compute channel- and sample-weighted Huber loss in normalized units."""

    if predicted_normalized.shape != target.shape:
        raise ValueError("predicted and target action tensors must have identical shapes")
    if predicted_normalized.ndim != 2:
        raise ValueError("action tensors must have shape (batch, channels)")
    if predicted_normalized.shape[-1] != action_schema.dimension:
        raise ValueError(
            f"action tensors must contain {action_schema.dimension} channels"
        )
    if delta <= 0.0:
        raise ValueError("delta must be positive")

    normalized_target = (
        target if target_is_normalized else normalize_physical_actions(target, action_schema)
    )
    error = predicted_normalized - normalized_target
    absolute = error.abs()
    huber = torch.where(
        absolute <= delta,
        0.5 * error.square() / delta,
        absolute - 0.5 * delta,
    )

    if channel_weights is None:
        channels = torch.ones(
            action_schema.dimension,
            dtype=huber.dtype,
            device=huber.device,
        )
    else:
        channels = torch.as_tensor(
            channel_weights,
            dtype=huber.dtype,
            device=huber.device,
        )
        if channels.shape != (action_schema.dimension,):
            raise ValueError(
                f"channel_weights must have shape ({action_schema.dimension},)"
            )
        if bool(torch.any(channels < 0.0)) or float(channels.sum()) <= 0.0:
            raise ValueError("channel_weights must be nonnegative with positive sum")
    channels = channels / channels.mean().clamp_min(1e-12)
    per_sample = (huber * channels).mean(dim=-1)

    if sample_weights is not None:
        if sample_weights.shape != per_sample.shape:
            raise ValueError("sample_weights must have shape (batch,)")
        weights = sample_weights / sample_weights.mean().clamp_min(1e-12)
        per_sample = per_sample * weights
    return per_sample.mean()


def contiguous_action_delta_loss(
    predicted_normalized: Tensor,
    target: Tensor,
    episode_ids: Tensor,
    *,
    sequence_indices: Tensor | None = None,
    action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
    target_is_normalized: bool = False,
    channel_weights: Sequence[float] | Tensor | None = None,
    delta: float = 0.05,
) -> Tensor:
    """Match action changes only across truly contiguous transitions."""

    batch = predicted_normalized.shape[0]
    if target.shape != predicted_normalized.shape:
        raise ValueError("predicted and target action tensors must have identical shapes")
    if episode_ids.shape != (batch,):
        raise ValueError("episode_ids must have shape (batch,)")
    if sequence_indices is not None and sequence_indices.shape != (batch,):
        raise ValueError("sequence_indices must have shape (batch,)")
    if batch < 2:
        return predicted_normalized.sum() * 0.0

    same_episode = episode_ids[1:] == episode_ids[:-1]
    contiguous = same_episode
    if sequence_indices is not None:
        contiguous = contiguous & (sequence_indices[1:] == sequence_indices[:-1] + 1)
    if not bool(torch.any(contiguous)):
        return predicted_normalized.sum() * 0.0

    normalized_target = (
        target if target_is_normalized else normalize_physical_actions(target, action_schema)
    )
    predicted_delta = predicted_normalized[1:] - predicted_normalized[:-1]
    target_delta = normalized_target[1:] - normalized_target[:-1]
    return normalized_weighted_huber_loss(
        predicted_delta[contiguous],
        target_delta[contiguous],
        action_schema=action_schema,
        target_is_normalized=True,
        channel_weights=channel_weights,
        delta=delta,
    )


@dataclass(frozen=True)
class BCLossConfig:
    huber_delta: float = 0.05
    action_delta_coefficient: float = 0.05
    channel_weights: tuple[float, ...] = (1.0,) * 12
    phase_weights: tuple[float, ...] = (0.25, 1.0, 1.0, 1.0, 0.75)
    minimum_quality: float = 0.05
    maximum_quality: float = 10.0
    target_is_normalized: bool = False

    def __post_init__(self) -> None:
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        if self.action_delta_coefficient < 0.0:
            raise ValueError("action_delta_coefficient must be nonnegative")


@dataclass
class BCLossOutput:
    total: Tensor
    behavior_cloning: Tensor
    action_delta: Tensor
    sample_weights: Tensor = field(repr=False)


class BehaviorCloningLoss(nn.Module):
    """Composite weighted Huber and contiguous action-delta objective."""

    def __init__(
        self,
        config: BCLossConfig | None = None,
        *,
        action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
    ) -> None:
        super().__init__()
        self.config = config or BCLossConfig()
        self.action_schema = action_schema
        if len(self.config.channel_weights) != action_schema.dimension:
            raise ValueError("channel_weights do not match the action schema")

    def forward(
        self,
        predicted_normalized: Tensor,
        target: Tensor,
        phases: Tensor,
        quality: Tensor,
        episode_ids: Tensor,
        sequence_indices: Tensor | None = None,
    ) -> BCLossOutput:
        weights = phase_quality_weights(
            phases,
            quality,
            phase_weights=self.config.phase_weights,
            minimum_quality=self.config.minimum_quality,
            maximum_quality=self.config.maximum_quality,
        )
        behavior_cloning = normalized_weighted_huber_loss(
            predicted_normalized,
            target,
            action_schema=self.action_schema,
            target_is_normalized=self.config.target_is_normalized,
            channel_weights=self.config.channel_weights,
            sample_weights=weights,
            delta=self.config.huber_delta,
        )
        action_delta = contiguous_action_delta_loss(
            predicted_normalized,
            target,
            episode_ids,
            sequence_indices=sequence_indices,
            action_schema=self.action_schema,
            target_is_normalized=self.config.target_is_normalized,
            channel_weights=self.config.channel_weights,
            delta=self.config.huber_delta,
        )
        total = (
            behavior_cloning
            + self.config.action_delta_coefficient * action_delta
        )
        return BCLossOutput(
            total=total,
            behavior_cloning=behavior_cloning,
            action_delta=action_delta,
            sample_weights=weights,
        )
