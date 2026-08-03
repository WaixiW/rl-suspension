import torch

from conftest import make_observation
from modular_suspension_rl.config import NetworkConfig, ObservationConfig
from modular_suspension_rl.contracts import RoadPreviewProcessor
from modular_suspension_rl.networks import ModularActor, TwinCritic, UnifiedActor


def test_preview_processor_builds_six_features_and_rear_delay():
    config = ObservationConfig(feedback_dim=20, feedback_history=4)
    processor = RoadPreviewProcessor(config)
    heights = torch.zeros(2, 4, 160)
    heights[:, :, 40:60] = 0.05
    confidence = torch.ones_like(heights)
    speed = torch.full((2, 1), 10.0)

    road = processor(heights, confidence, speed)

    assert road.shape == (2, 4, 6, 160)
    assert torch.all(road[:, 2:, processor.ARRIVAL_TIME, 0] > 0.0)
    assert torch.allclose(
        road[:, 0, processor.CONFIDENCE], torch.ones(2, 160)
    )


def test_modular_actor_shapes_residual_and_dropout_fallback():
    observation_config = ObservationConfig(feedback_dim=20, feedback_history=4)
    network_config = NetworkConfig(
        preview_channels=(16, 24),
        state_hidden_dim=24,
        fused_hidden_dim=32,
    )
    actor = ModularActor(observation_config, network_config)
    observation = make_observation(observation_config)
    output = actor(observation)

    assert output.raw_force_n.shape == (2, 4)
    assert output.gate.shape == (2, 4)
    assert torch.allclose(
        output.raw_force_n,
        output.feedback_force_n + output.gate * output.preview_force_n,
    )

    dropout = make_observation(observation_config, confidence=0.0)
    dropout.feedback_history = observation.feedback_history
    full_dropout = actor(dropout)
    feedback_only = actor(dropout, feedback_only=True)
    assert torch.allclose(
        full_dropout.raw_force_n, feedback_only.raw_force_n, atol=1e-5
    )


def test_critic_and_unified_baseline_accept_same_contract():
    observation_config = ObservationConfig(feedback_dim=20, feedback_history=4)
    network_config = NetworkConfig(
        preview_channels=(16, 24),
        state_hidden_dim=24,
        fused_hidden_dim=32,
    )
    observation = make_observation(observation_config)
    critic = TwinCritic(observation_config, network_config)
    unified = UnifiedActor(observation_config, network_config)
    action = unified(observation)
    q1, q2 = critic(observation, action)

    assert action.shape == (2, 4)
    assert q1.shape == (2, 1)
    assert q2.shape == (2, 1)
