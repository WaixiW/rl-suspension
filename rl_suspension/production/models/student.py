"""Direct twelve-channel student policy for MPC distillation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    DEFAULT_OBSERVATION_SCHEMA,
    ActionSchema,
    ObservationSchema,
    ObservationV1,
)


@dataclass(frozen=True)
class StudentConfig:
    """Architecture parameters for :class:`Direct12Student`."""

    observation_schema: ObservationSchema = field(
        default_factory=lambda: DEFAULT_OBSERVATION_SCHEMA
    )
    state_feature_dim: int = 192
    road_feature_dim: int = 128
    fusion_dim: int = 256
    residual_blocks: int = 3
    dropout: float = 0.0
    use_gru: bool = False
    gru_layers: int = 1
    temporal_history_steps: int = 8

    def __post_init__(self) -> None:
        dimensions = (
            self.state_feature_dim,
            self.road_feature_dim,
            self.fusion_dim,
            self.residual_blocks,
            self.gru_layers,
            self.temporal_history_steps,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("student dimensions and layer counts must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.GELU(),
    )


class StateActuatorActionEncoder(nn.Module):
    """Encode dynamic state, actuator state, and previous action separately."""

    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        schema = config.observation_schema
        dynamic_dim = (
            schema.vehicle_state_dim
            + schema.sensor_feature_dim
            + 1
            + 2
        )
        branch_dim = max(config.state_feature_dim // 3, 16)
        self.dynamic = _mlp(
            dynamic_dim,
            max(branch_dim, 64),
            branch_dim,
            config.dropout,
        )
        self.actuator = _mlp(
            schema.actuator_state_dim,
            max(branch_dim, 64),
            branch_dim,
            config.dropout,
        )
        self.previous_action = _mlp(
            schema.action_dim,
            max(branch_dim, 64),
            branch_dim,
            config.dropout,
        )
        self.output = _mlp(
            3 * branch_dim,
            config.state_feature_dim,
            config.state_feature_dim,
            config.dropout,
        )
        self.schema = schema

    def forward(self, state_vector: Tensor) -> Tensor:
        schema = self.schema
        vehicle_sensor_stop = schema.vehicle_state_dim + schema.sensor_feature_dim
        actuator_stop = vehicle_sensor_stop + schema.actuator_state_dim
        action_stop = actuator_stop + schema.action_dim
        dynamic = torch.cat(
            (state_vector[..., :vehicle_sensor_stop], state_vector[..., action_stop:]),
            dim=-1,
        )
        encoded = torch.cat(
            (
                self.dynamic(dynamic),
                self.actuator(state_vector[..., vehicle_sensor_stop:actuator_stop]),
                self.previous_action(state_vector[..., actuator_stop:action_stop]),
            ),
            dim=-1,
        )
        return self.output(encoded)


class RoadEncoder(nn.Module):
    """Encode all four road-height and validity channels with a 1-D CNN."""

    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        channels = config.observation_schema.road_channels
        self.network = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=9, stride=2, padding=4),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv1d(96, config.road_feature_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
        )

    def forward(self, road: Tensor) -> Tensor:
        return self.network(road)


class ResidualBlock(nn.Module):
    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, 2 * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.network(value)


class Direct12Student(nn.Module):
    """Predict normalized direct actuator commands in ``[0, 1]^12``.

    Inputs may be one observation per batch (``B,D`` and ``B,4,P``), or a
    temporal sequence (``B,T,D`` and ``B,T,4,P``). Sequence inputs are consumed
    by the optional GRU; a non-temporal model uses the final item.
    """

    def __init__(self, config: StudentConfig | None = None) -> None:
        super().__init__()
        self.config = config or StudentConfig()
        schema = self.config.observation_schema
        if schema.action_dim != 12:
            raise ValueError("Direct12Student requires a twelve-dimensional action schema")

        self.state_encoder = StateActuatorActionEncoder(self.config)
        self.road_encoder = RoadEncoder(self.config)
        encoded_dim = self.config.state_feature_dim + self.config.road_feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(encoded_dim, self.config.fusion_dim),
            nn.LayerNorm(self.config.fusion_dim),
            nn.GELU(),
        )
        self.temporal = (
            nn.GRU(
                input_size=self.config.fusion_dim,
                hidden_size=self.config.fusion_dim,
                num_layers=self.config.gru_layers,
                batch_first=True,
                dropout=(
                    self.config.dropout if self.config.gru_layers > 1 else 0.0
                ),
            )
            if self.config.use_gru
            else None
        )
        self.residual = nn.Sequential(
            *[
                ResidualBlock(self.config.fusion_dim, self.config.dropout)
                for _ in range(self.config.residual_blocks)
            ]
        )
        self.action_head = nn.Linear(self.config.fusion_dim, schema.action_dim)

    def _validate_inputs(self, state_vector: Tensor, road: Tensor) -> bool:
        schema = self.config.observation_schema
        if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
            return state_vector.ndim == 3
        if state_vector.ndim not in (2, 3):
            raise ValueError("state_vector must have shape (B,D) or (B,T,D)")
        temporal = state_vector.ndim == 3
        expected_road_rank = 4 if temporal else 3
        if road.ndim != expected_road_rank:
            raise ValueError("state_vector and road must both be temporal or non-temporal")
        if state_vector.shape[-1] != schema.state_vector_dim:
            raise ValueError(
                f"state_vector last dimension must be {schema.state_vector_dim}"
            )
        if road.shape[-2:] != (schema.road_channels, schema.road_points):
            raise ValueError(
                "road trailing dimensions must be "
                f"({schema.road_channels}, {schema.road_points})"
            )
        if state_vector.shape[:2] != road.shape[:2] if temporal else (
            state_vector.shape[0] != road.shape[0]
        ):
            raise ValueError("state_vector and road batch dimensions must match")
        return temporal

    def encode(self, state_vector: Tensor, road: Tensor) -> Tensor:
        temporal_input = self._validate_inputs(state_vector, road)
        if not temporal_input:
            state_vector = state_vector.unsqueeze(1)
            road = road.unsqueeze(1)

        batch, steps = state_vector.shape[:2]
        state_features = self.state_encoder(
            state_vector.reshape(batch * steps, state_vector.shape[-1])
        )
        road_features = self.road_encoder(
            road.reshape(batch * steps, road.shape[-2], road.shape[-1])
        )
        fused = self.fusion(torch.cat((state_features, road_features), dim=-1))
        fused = fused.reshape(batch, steps, self.config.fusion_dim)

        if self.temporal is not None:
            fused, _ = self.temporal(fused)
            representation = fused[:, -1]
        else:
            representation = fused[:, -1]
        return self.residual(representation)

    def forward(self, state_vector: Tensor, road: Tensor) -> Tensor:
        return torch.sigmoid(self.action_head(self.encode(state_vector, road)))


class PhysicalActionExportWrapper(nn.Module):
    """Wrap normalized student outputs as physical direct-12D commands."""

    def __init__(
        self,
        model: Direct12Student,
        action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
    ) -> None:
        super().__init__()
        if action_schema.dimension != model.config.observation_schema.action_dim:
            raise ValueError("student and export action schemas are incompatible")
        self.model = model
        self.register_buffer(
            "action_minimum",
            torch.as_tensor(action_schema.minimum, dtype=torch.float32),
        )
        self.register_buffer(
            "action_range",
            torch.as_tensor(action_schema.maximum, dtype=torch.float32)
            - torch.as_tensor(action_schema.minimum, dtype=torch.float32),
        )

    def forward(self, state_vector: Tensor, road: Tensor) -> Tensor:
        normalized = self.model(state_vector, road)
        return self.action_minimum + normalized * self.action_range


class TorchStudentPolicy:
    """Adapt a trained student to the production ``PolicyAdapter`` contract."""

    def __init__(
        self,
        model: Direct12Student,
        *,
        action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
        device: str | torch.device | None = None,
        name: str = "direct12_student",
    ) -> None:
        self.model = model
        self.action_schema = action_schema
        if action_schema.dimension != model.config.observation_schema.action_dim:
            raise ValueError("student and physical action schemas are incompatible")
        self.name = name
        self.device = torch.device(device) if device is not None else next(
            model.parameters()
        ).device
        history_steps = model.config.temporal_history_steps
        self._state_history: deque[np.ndarray] = deque(maxlen=history_steps)
        self._road_history: deque[np.ndarray] = deque(maxlen=history_steps)

    def reset(self) -> None:
        self._state_history.clear()
        self._road_history.clear()

    def predict_normalized(self, observation: ObservationV1) -> np.ndarray:
        observation.validate(
            self.model.config.observation_schema,
            self.action_schema,
        )
        state = observation.state_vector().astype(np.float32)
        road = observation.road_tensor().astype(np.float32)
        self._state_history.append(state)
        self._road_history.append(road)

        if self.model.config.use_gru:
            states = np.stack(tuple(self._state_history), axis=0)[None]
            roads = np.stack(tuple(self._road_history), axis=0)[None]
        else:
            states = state[None]
            roads = road[None]

        self.model.eval()
        with torch.no_grad():
            normalized = self.model(
                torch.as_tensor(states, device=self.device),
                torch.as_tensor(roads, device=self.device),
            )[0]
        return normalized.cpu().numpy().astype(np.float64)

    def predict(self, observation: ObservationV1) -> np.ndarray:
        return self.action_schema.denormalize(self.predict_normalized(observation))


def stack_observations(
    observations: Sequence[ObservationV1],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert contract observations to model-ready float32 arrays."""

    if not observations:
        raise ValueError("at least one observation is required")
    states = np.stack([item.state_vector() for item in observations]).astype(np.float32)
    roads = np.stack([item.road_tensor() for item in observations]).astype(np.float32)
    return states, roads
