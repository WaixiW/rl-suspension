import torch

from modular_suspension_rl.config import ObservationConfig
from modular_suspension_rl.contracts import ObservationBatch


def make_observation(
    config: ObservationConfig,
    batch_size: int = 2,
    confidence: float = 1.0,
) -> ObservationBatch:
    road = torch.zeros(
        batch_size,
        4,
        config.preview_feature_count,
        config.preview_points,
    )
    road[:, :, 3, :] = confidence
    road[:, :, 5, :] = float(confidence > 0.0)
    return ObservationBatch(
        road=road,
        feedback_history=torch.randn(
            batch_size, config.feedback_history, config.feedback_dim
        ),
        speed=torch.full((batch_size, 1), 12.0),
        preview_confidence=torch.full((batch_size, 4), confidence),
        previous_gate=torch.full((batch_size, 4), 0.4),
        suspension_velocity=torch.zeros(batch_size, 4),
        actuator_force=torch.zeros(batch_size, 4),
        previous_commands=torch.zeros(batch_size, 12),
    )
