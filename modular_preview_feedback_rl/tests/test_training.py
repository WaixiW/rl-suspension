import torch

from conftest import make_observation
from modular_suspension_rl.config import (
    NetworkConfig,
    ObservationConfig,
    TD3Config,
)
from modular_suspension_rl.networks import ModularActor, TwinCritic
from modular_suspension_rl.training import (
    ReplayBuffer,
    TD3Agent,
    TrainingPhase,
    Transition,
)


def make_agent():
    observation_config = ObservationConfig(feedback_dim=12, feedback_history=3)
    network_config = NetworkConfig(
        preview_channels=(8, 12),
        state_hidden_dim=12,
        fused_hidden_dim=16,
    )
    td3_config = TD3Config(
        batch_size=4,
        replay_capacity=16,
        policy_delay=1,
    )
    actor = ModularActor(observation_config, network_config)
    critic = TwinCritic(observation_config, network_config)
    agent = TD3Agent(
        actor, critic, network_config, td3_config, torch.device("cpu")
    )
    return observation_config, agent


def test_training_phase_freezes_expected_actor_parts():
    _, agent = make_agent()

    agent.configure_phase(TrainingPhase.PREVIEW_RESIDUAL)
    assert any(
        parameter.requires_grad
        for parameter in agent.actor.preview_head.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in agent.actor.feedback_head.parameters()
    )

    agent.configure_phase(TrainingPhase.GATE_TRAINING)
    assert any(
        parameter.requires_grad for parameter in agent.actor.gate_head.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in agent.actor.preview_head.parameters()
    )

    agent.configure_phase(TrainingPhase.JOINT_FINE_TUNE)
    assert all(parameter.requires_grad for parameter in agent.actor.parameters())


def test_td3_update_uses_modular_actor_and_shared_twin_critic():
    observation_config, agent = make_agent()
    replay = ReplayBuffer(capacity=16)
    agent.configure_phase(TrainingPhase.FEEDBACK_ONLY)

    for _ in range(4):
        observation = make_observation(
            observation_config, batch_size=1
        )
        next_observation = make_observation(
            observation_config, batch_size=1
        )
        with torch.no_grad():
            output = agent.select_action(observation)
        replay.add(
            Transition(
                observation=observation,
                raw_force_n=output.raw_force_n,
                executed_force_n=output.raw_force_n.clamp(-6000.0, 6000.0),
                commands=torch.zeros(1, 12),
                reward=torch.zeros(1),
                next_observation=next_observation,
                done=torch.zeros(1),
                projection_correction_n=torch.zeros(1, 4),
            )
        )

    metrics = agent.update(replay)

    assert "critic_loss" in metrics
    assert "actor_loss" in metrics
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
