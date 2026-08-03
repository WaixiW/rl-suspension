from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .config import NetworkConfig, ObservationConfig
from .contracts import ObservationBatch


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class SpatialResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size, padding=padding),
            nn.GroupNorm(1, output_channels),
            nn.SiLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size, padding=padding),
            nn.GroupNorm(1, output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, 1)
        )
        self.activation = nn.SiLU()

    def forward(self, values: Tensor) -> Tensor:
        return self.activation(self.body(values) + self.skip(values))


class PreviewEncoder(nn.Module):
    """Shared wheel-path encoder. Output shape is [batch, 4, feature_dim]."""

    def __init__(self, observation: ObservationConfig, network: NetworkConfig):
        super().__init__()
        blocks = []
        input_channels = observation.preview_feature_count
        for output_channels in network.preview_channels:
            blocks.append(
                SpatialResidualBlock(
                    input_channels, output_channels, network.preview_kernel_size
                )
            )
            blocks.append(nn.AvgPool1d(kernel_size=2, stride=2))
            input_channels = output_channels
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = network.preview_channels[-1]

    def forward(self, road: Tensor) -> Tensor:
        if road.ndim != 4 or road.shape[1] != 4:
            raise ValueError("road must have shape [batch, 4, features, points].")
        batch, wheels, features, points = road.shape
        encoded = self.blocks(road.reshape(batch * wheels, features, points))
        encoded = self.pool(encoded).squeeze(-1)
        return encoded.reshape(batch, wheels, self.output_dim)


class FeedbackEncoder(nn.Module):
    def __init__(self, observation: ObservationConfig, network: NetworkConfig):
        super().__init__()
        self.gru = nn.GRU(
            input_size=observation.feedback_dim,
            hidden_size=network.state_hidden_dim,
            num_layers=network.state_layers,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(network.state_hidden_dim)
        self.output_dim = network.state_hidden_dim

    def forward(self, feedback_history: Tensor) -> Tensor:
        if feedback_history.ndim != 3:
            raise ValueError("feedback_history must have shape [batch, history, features].")
        _, hidden = self.gru(feedback_history)
        return self.output_norm(hidden[-1])


@dataclass
class ActorOutput:
    raw_force_n: Tensor
    feedback_force_n: Tensor
    preview_force_n: Tensor
    gate: Tensor
    gate_raw: Tensor
    actuator_authority: Tensor


class ModularActor(nn.Module):
    """Preview residual plus an unattenuated feedback force."""

    def __init__(
        self,
        observation: ObservationConfig,
        network: NetworkConfig,
    ):
        super().__init__()
        self.observation_config = observation
        self.network_config = network
        self.preview_encoder = PreviewEncoder(observation, network)
        self.feedback_encoder = FeedbackEncoder(observation, network)

        self.feedback_head = _mlp(
            self.feedback_encoder.output_dim,
            network.fused_hidden_dim,
            4,
        )
        preview_input = self.preview_encoder.output_dim + 2
        self.preview_head = _mlp(preview_input, network.fused_hidden_dim // 2, 1)
        gate_input = (
            self.preview_encoder.output_dim
            + self.feedback_encoder.output_dim
            + 4
        )
        self.gate_head = _mlp(gate_input, network.fused_hidden_dim // 2, 1)
        self._initialize_preview_residual()

    def _initialize_preview_residual(self) -> None:
        final = self.preview_head[-1]
        nn.init.uniform_(final.weight, -1e-3, 1e-3)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        observation: ObservationBatch,
        gate_override: Optional[Tensor] = None,
        feedback_only: bool = False,
        preview_only: bool = False,
    ) -> ActorOutput:
        if feedback_only and preview_only:
            raise ValueError("feedback_only and preview_only cannot both be true.")
        preview_features = self.preview_encoder(observation.road)
        state_features = self.feedback_encoder(observation.feedback_history)
        batch = observation.batch_size

        authority = (
            1.0
            - observation.actuator_force.abs() / self.network_config.force_limit_n
        ).clamp(0.0, 1.0)
        speed = observation.speed.unsqueeze(1).expand(batch, 4, 1)
        preview_context = torch.cat(
            (preview_features, speed, authority.unsqueeze(-1)), dim=-1
        )
        preview_force = (
            self.network_config.preview_residual_limit_n
            * torch.tanh(self.preview_head(preview_context).squeeze(-1))
        )
        feedback_force = (
            self.network_config.force_limit_n
            * torch.tanh(self.feedback_head(state_features))
        )

        state_per_wheel = state_features.unsqueeze(1).expand(
            batch, 4, state_features.shape[-1]
        )
        effective_preview_time = (
            self.observation_config.preview_range_m
            / observation.speed.abs().clamp_min(self.observation_config.min_speed_mps)
        )
        horizon = effective_preview_time.unsqueeze(1).expand(batch, 4, 1)
        gate_context = torch.cat(
            (
                preview_features,
                state_per_wheel,
                observation.preview_confidence.unsqueeze(-1),
                horizon,
                authority.unsqueeze(-1),
                observation.previous_gate.unsqueeze(-1),
            ),
            dim=-1,
        )
        gate_raw = torch.sigmoid(self.gate_head(gate_context).squeeze(-1))
        if gate_override is not None:
            if gate_override.shape != gate_raw.shape:
                raise ValueError("gate_override must have shape [batch, 4].")
            gate_raw = gate_override.clamp(0.0, 1.0)
        smoothing = self.network_config.gate_smoothing
        gate = smoothing * observation.previous_gate + (1.0 - smoothing) * gate_raw
        gate = gate * observation.preview_confidence.clamp(0.0, 1.0)

        if feedback_only:
            gate = torch.zeros_like(gate)
        if preview_only:
            feedback_force = torch.zeros_like(feedback_force)
        raw_force = feedback_force + gate * preview_force
        return ActorOutput(
            raw_force_n=raw_force,
            feedback_force_n=feedback_force,
            preview_force_n=preview_force,
            gate=gate,
            gate_raw=gate_raw,
            actuator_authority=authority,
        )


class QNetwork(nn.Module):
    def __init__(
        self,
        observation: ObservationConfig,
        network: NetworkConfig,
    ):
        super().__init__()
        self.preview_encoder = PreviewEncoder(observation, network)
        self.feedback_encoder = FeedbackEncoder(observation, network)
        input_dim = (
            self.preview_encoder.output_dim * 4
            + self.feedback_encoder.output_dim
            + 4
            + 4
            + 1
            + 4
        )
        self.value = _mlp(input_dim, network.fused_hidden_dim, 1)
        self.force_scale = network.force_limit_n

    def forward(self, observation: ObservationBatch, raw_force_n: Tensor) -> Tensor:
        road = self.preview_encoder(observation.road).flatten(start_dim=1)
        state = self.feedback_encoder(observation.feedback_history)
        values = torch.cat(
            (
                road,
                state,
                observation.preview_confidence,
                observation.suspension_velocity,
                observation.speed,
                raw_force_n / self.force_scale,
            ),
            dim=-1,
        )
        return self.value(values)


class TwinCritic(nn.Module):
    def __init__(
        self,
        observation: ObservationConfig,
        network: NetworkConfig,
    ):
        super().__init__()
        self.q1 = QNetwork(observation, network)
        self.q2 = QNetwork(observation, network)

    def forward(
        self, observation: ObservationBatch, raw_force_n: Tensor
    ) -> Tuple[Tensor, Tensor]:
        return self.q1(observation, raw_force_n), self.q2(observation, raw_force_n)


class UnifiedActor(nn.Module):
    """Matched interface baseline that fuses modalities before the force head."""

    def __init__(self, observation: ObservationConfig, network: NetworkConfig):
        super().__init__()
        self.preview_encoder = PreviewEncoder(observation, network)
        self.feedback_encoder = FeedbackEncoder(observation, network)
        input_dim = (
            self.preview_encoder.output_dim * 4
            + self.feedback_encoder.output_dim
            + 4
            + 1
        )
        self.force_head = _mlp(input_dim, network.fused_hidden_dim, 4)
        self.force_limit_n = network.force_limit_n

    def forward(self, observation: ObservationBatch) -> Tensor:
        values = torch.cat(
            (
                self.preview_encoder(observation.road).flatten(start_dim=1),
                self.feedback_encoder(observation.feedback_history),
                observation.preview_confidence,
                observation.speed,
            ),
            dim=-1,
        )
        return self.force_limit_n * torch.tanh(self.force_head(values))
