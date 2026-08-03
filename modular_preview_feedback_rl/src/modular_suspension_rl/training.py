import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Protocol, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import NetworkConfig, TD3Config
from .contracts import ObservationBatch
from .networks import ActorOutput, ModularActor, TwinCritic


class TrainingPhase(str, Enum):
    FEEDBACK_ONLY = "feedback_only"
    PREVIEW_RESIDUAL = "preview_residual"
    GATE_TRAINING = "gate_training"
    JOINT_FINE_TUNE = "joint_fine_tune"


@dataclass
class Transition:
    observation: ObservationBatch
    raw_force_n: Tensor
    executed_force_n: Tensor
    commands: Tensor
    reward: Tensor
    next_observation: ObservationBatch
    done: Tensor
    projection_correction_n: Tensor


@dataclass
class ReplayBatch:
    observation: ObservationBatch
    raw_force_n: Tensor
    executed_force_n: Tensor
    commands: Tensor
    reward: Tensor
    next_observation: ObservationBatch
    done: Tensor
    projection_correction_n: Tensor


def _slice_observation(observation: ObservationBatch, index: int) -> ObservationBatch:
    values = {}
    for name, value in observation.__dict__.items():
        item = value[index : index + 1].detach().cpu().clone()
        if name == "road":
            item = item.to(torch.float16)
        values[name] = item
    return ObservationBatch(**values)


def _concatenate_observations(
    observations: List[ObservationBatch], device: torch.device
) -> ObservationBatch:
    values = {}
    for name in observations[0].__dict__:
        value = torch.cat([getattr(item, name) for item in observations], dim=0)
        if name == "road":
            value = value.float()
        values[name] = value.to(device)
    return ObservationBatch(**values)


class ReplayBuffer:
    """Lazy ring buffer; preview tensors are stored as float16 to reduce memory."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self.storage: List[Transition] = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: Transition) -> None:
        batch = transition.raw_force_n.shape[0]
        for index in range(batch):
            item = Transition(
                observation=_slice_observation(transition.observation, index),
                raw_force_n=transition.raw_force_n[index : index + 1].detach().cpu(),
                executed_force_n=transition.executed_force_n[
                    index : index + 1
                ].detach().cpu(),
                commands=transition.commands[index : index + 1].detach().cpu(),
                reward=transition.reward[index : index + 1].detach().cpu(),
                next_observation=_slice_observation(transition.next_observation, index),
                done=transition.done[index : index + 1].detach().cpu(),
                projection_correction_n=transition.projection_correction_n[
                    index : index + 1
                ].detach().cpu(),
            )
            if len(self.storage) < self.capacity:
                self.storage.append(item)
            else:
                self.storage[self.position] = item
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if len(self.storage) < batch_size:
            raise ValueError("Not enough transitions to sample a complete batch.")
        items = random.sample(self.storage, batch_size)

        def concatenate(name: str) -> Tensor:
            return torch.cat([getattr(item, name) for item in items], dim=0).to(device)

        return ReplayBatch(
            observation=_concatenate_observations(
                [item.observation for item in items], device
            ),
            raw_force_n=concatenate("raw_force_n"),
            executed_force_n=concatenate("executed_force_n"),
            commands=concatenate("commands"),
            reward=concatenate("reward").reshape(batch_size, 1),
            next_observation=_concatenate_observations(
                [item.next_observation for item in items], device
            ),
            done=concatenate("done").reshape(batch_size, 1),
            projection_correction_n=concatenate(
                "projection_correction_n"
            ),
        )


class SuspensionTrainingEnvironment(Protocol):
    def reset(self, phase: TrainingPhase) -> ObservationBatch:
        ...

    def step(
        self, raw_force_n: Tensor, actor_output: ActorOutput
    ) -> Tuple[ObservationBatch, Tensor, Tensor, Dict[str, Tensor]]:
        ...


class TD3Agent:
    def __init__(
        self,
        actor: ModularActor,
        critic: TwinCritic,
        network: NetworkConfig,
        config: TD3Config,
        device: torch.device,
    ):
        self.device = device
        self.network = network
        self.config = config
        self.actor = actor.to(device)
        self.actor_target = copy.deepcopy(actor).to(device).eval()
        self.critic = critic.to(device)
        self.critic_target = copy.deepcopy(critic).to(device).eval()
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate
        )
        self.phase = TrainingPhase.FEEDBACK_ONLY
        self.update_count = 0
        self.configure_phase(self.phase)

    def configure_phase(self, phase: TrainingPhase) -> None:
        self.phase = phase
        for parameter in self.actor.parameters():
            parameter.requires_grad = False
        if phase == TrainingPhase.FEEDBACK_ONLY:
            self._set_trainable(self.actor.feedback_encoder, True)
            self._set_trainable(self.actor.feedback_head, True)
        elif phase == TrainingPhase.PREVIEW_RESIDUAL:
            self._set_trainable(self.actor.preview_encoder, True)
            self._set_trainable(self.actor.preview_head, True)
        elif phase == TrainingPhase.GATE_TRAINING:
            self._set_trainable(self.actor.gate_head, True)
        elif phase == TrainingPhase.JOINT_FINE_TUNE:
            for parameter in self.actor.parameters():
                parameter.requires_grad = True
        else:
            raise ValueError("Unsupported training phase: {}".format(phase))

    @staticmethod
    def _set_trainable(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = enabled

    def _actor_output(
        self, actor: ModularActor, observation: ObservationBatch
    ) -> ActorOutput:
        if self.phase == TrainingPhase.FEEDBACK_ONLY:
            return actor(observation, feedback_only=True)
        if self.phase == TrainingPhase.PREVIEW_RESIDUAL:
            return actor(
                observation,
                gate_override=observation.preview_confidence,
            )
        return actor(observation)

    @torch.no_grad()
    def select_action(
        self,
        observation: ObservationBatch,
        exploration: bool = False,
    ) -> ActorOutput:
        output = self._actor_output(self.actor, observation.to(self.device))
        if not exploration:
            return output
        scale = self.network.force_limit_n + self.network.preview_residual_limit_n
        noise = torch.randn_like(output.raw_force_n)
        noisy_force = output.raw_force_n + self.config.exploration_noise_std * scale * noise
        return ActorOutput(
            raw_force_n=noisy_force.clamp(-scale, scale),
            feedback_force_n=output.feedback_force_n,
            preview_force_n=output.preview_force_n,
            gate=output.gate,
            gate_raw=output.gate_raw,
            actuator_authority=output.actuator_authority,
        )

    def update(self, replay: ReplayBuffer) -> Dict[str, float]:
        batch = replay.sample(self.config.batch_size, self.device)
        action_scale = (
            self.network.force_limit_n + self.network.preview_residual_limit_n
        )
        with torch.no_grad():
            target_output = self._actor_output(
                self.actor_target, batch.next_observation
            )
            noise = torch.randn_like(target_output.raw_force_n)
            noise = noise * self.config.target_noise_std * action_scale
            noise = noise.clamp(
                -self.config.target_noise_clip * action_scale,
                self.config.target_noise_clip * action_scale,
            )
            target_action = (target_output.raw_force_n + noise).clamp(
                -action_scale, action_scale
            )
            target_q1, target_q2 = self.critic_target(
                batch.next_observation, target_action
            )
            target_q = batch.reward + (
                1.0 - batch.done
            ) * self.config.discount * torch.minimum(target_q1, target_q2)

        current_q1, current_q2 = self.critic(
            batch.observation, batch.raw_force_n
        )
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics = {"critic_loss": float(critic_loss.detach().cpu())}
        self.update_count += 1
        if self.update_count % self.config.policy_delay == 0:
            for parameter in self.critic.parameters():
                parameter.requires_grad = False
            output = self._actor_output(self.actor, batch.observation)
            actor_loss = -self.critic.q1(
                batch.observation, output.raw_force_n
            ).mean()
            gate_penalty = (
                (output.gate - batch.observation.previous_gate).square().mean()
                + (
                    output.gate
                    * (1.0 - batch.observation.preview_confidence)
                )
                .square()
                .mean()
            )
            projection_excess = torch.relu(
                output.raw_force_n.abs() - self.network.force_limit_n
            )
            projection_penalty = (projection_excess / action_scale).square().mean()
            actor_loss = (
                actor_loss
                + self.config.gate_regularization_weight * gate_penalty
                + self.config.projection_penalty_weight * projection_penalty
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            for parameter in self.critic.parameters():
                parameter.requires_grad = True
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            metrics.update(
                {
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "gate_mean": float(output.gate.mean().detach().cpu()),
                    "projection_penalty": float(
                        projection_penalty.detach().cpu()
                    ),
                }
            )
        return metrics

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters()
            ):
                target_parameter.mul_(1.0 - self.config.target_tau)
                target_parameter.add_(self.config.target_tau * source_parameter)


@dataclass(frozen=True)
class PhaseBudget:
    phase: TrainingPhase
    environment_steps: int
    learning_starts: int
    updates_per_step: int = 1


class StagedTrainer:
    """Orchestrates the four phases while leaving simulator details in an adapter."""

    def __init__(
        self,
        agent: TD3Agent,
        replay: ReplayBuffer,
        environment: SuspensionTrainingEnvironment,
    ):
        self.agent = agent
        self.replay = replay
        self.environment = environment

    def run_phase(self, budget: PhaseBudget) -> Dict[str, float]:
        self.agent.configure_phase(budget.phase)
        observation = self.environment.reset(budget.phase).to(self.agent.device)
        aggregate: Dict[str, float] = {}
        metric_count: Dict[str, int] = {}
        phase_steps = 0

        while phase_steps < budget.environment_steps:
            actor_output = self.agent.select_action(observation, exploration=True)
            next_observation, reward, done, info = self.environment.step(
                actor_output.raw_force_n, actor_output
            )
            next_observation = next_observation.to(self.agent.device)
            transition = Transition(
                observation=observation,
                raw_force_n=actor_output.raw_force_n,
                executed_force_n=info["executed_force_n"],
                commands=info["commands"],
                reward=reward,
                next_observation=next_observation,
                done=done,
                projection_correction_n=info["projection_correction_n"],
            )
            self.replay.add(transition)
            observation = next_observation
            phase_steps += observation.batch_size

            if len(self.replay) >= max(
                budget.learning_starts, self.agent.config.batch_size
            ):
                for _ in range(budget.updates_per_step):
                    metrics = self.agent.update(self.replay)
                    for name, value in metrics.items():
                        aggregate[name] = aggregate.get(name, 0.0) + value
                        metric_count[name] = metric_count.get(name, 0) + 1

            if bool(done.any()):
                observation = self.environment.reset(budget.phase).to(
                    self.agent.device
                )

        return {
            name: value / metric_count[name]
            for name, value in aggregate.items()
        }
