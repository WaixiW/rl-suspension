from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

import torch
from torch import Tensor, nn

from .contracts import ObservationBatch
from .networks import ActorOutput, ModularActor, UnifiedActor
from .randomization import RoadScenario


class EvaluationEnvironment(Protocol):
    sample_time_s: float

    def reset(
        self,
        scenario: RoadScenario,
        seed: int,
        condition: str,
    ) -> ObservationBatch:
        ...

    def step(
        self, raw_force_n: Tensor
    ) -> Tuple[ObservationBatch, "StepMetrics", Tensor]:
        ...


class ForceController(Protocol):
    def reset(self) -> None:
        ...

    def act(self, observation: ObservationBatch) -> Tuple[Tensor, Dict[str, Tensor]]:
        ...


class ModularController:
    def __init__(self, actor: ModularActor, mode: str = "full"):
        if mode not in ("full", "feedback_only", "preview_only", "convex_gate"):
            raise ValueError("Unsupported modular controller mode.")
        self.actor = actor
        self.mode = mode

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def act(self, observation: ObservationBatch) -> Tuple[Tensor, Dict[str, Tensor]]:
        if self.mode == "feedback_only":
            output = self.actor(observation, feedback_only=True)
        elif self.mode == "preview_only":
            output = self.actor(observation, preview_only=True)
        else:
            output = self.actor(observation)
        force = output.raw_force_n
        if self.mode == "convex_gate":
            force = (
                output.gate * output.preview_force_n
                + (1.0 - output.gate) * output.feedback_force_n
            )
        return force, {
            "gate": output.gate,
            "feedback_force_n": output.feedback_force_n,
            "preview_force_n": output.preview_force_n,
        }


class UnifiedController:
    def __init__(self, actor: UnifiedActor):
        self.actor = actor

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def act(self, observation: ObservationBatch) -> Tuple[Tensor, Dict[str, Tensor]]:
        force = self.actor(observation)
        return force, {}


class PassiveController:
    def reset(self) -> None:
        pass

    def act(self, observation: ObservationBatch) -> Tuple[Tensor, Dict[str, Tensor]]:
        return observation.suspension_velocity.new_zeros(
            observation.batch_size, 4
        ), {}


@dataclass
class StepMetrics:
    vertical_acceleration_mps2: Tensor
    pitch_rate_radps: Tensor
    pitch_acceleration_radps2: Tensor
    roll_rate_radps: Tensor
    roll_acceleration_radps2: Tensor
    passenger_acceleration_mps2: Tensor
    suspension_travel_m: Tensor
    tire_load_variation_n: Tensor
    electrical_power_w: Tensor
    saturation: Tensor
    action_slew_normalized: Tensor
    force_tracking_error_n: Tensor
    hard_violation: Tensor


@dataclass
class EpisodeSummary:
    rms_vertical_acceleration: float
    peak_vertical_acceleration: float
    rms_pitch_rate: float
    rms_pitch_acceleration: float
    rms_roll_rate: float
    rms_roll_acceleration: float
    rms_passenger_acceleration: float
    max_suspension_travel: float
    rms_tire_load_variation: float
    energy_j: float
    saturation_fraction: float
    rms_action_slew: float
    rms_force_tracking_error: float
    hard_violation_count: int
    mean_gate: float
    preview_to_feedback_ratio: float


class EpisodeAccumulator:
    def __init__(self, sample_time_s: float):
        self.sample_time_s = sample_time_s
        self.values: Dict[str, List[Tensor]] = {}
        self.gates: List[Tensor] = []
        self.preview_forces: List[Tensor] = []
        self.feedback_forces: List[Tensor] = []

    def add(self, metrics: StepMetrics, auxiliary: Dict[str, Tensor]) -> None:
        for name, value in metrics.__dict__.items():
            self.values.setdefault(name, []).append(value.detach().cpu())
        if "gate" in auxiliary:
            self.gates.append(auxiliary["gate"].detach().cpu())
        if "preview_force_n" in auxiliary:
            self.preview_forces.append(
                auxiliary["preview_force_n"].detach().cpu()
            )
        if "feedback_force_n" in auxiliary:
            self.feedback_forces.append(
                auxiliary["feedback_force_n"].detach().cpu()
            )

    def _stack(self, name: str) -> Tensor:
        return torch.cat(
            [value.reshape(value.shape[0], -1) for value in self.values[name]],
            dim=0,
        )

    @staticmethod
    def _rms(value: Tensor) -> float:
        return float(torch.sqrt(value.float().square().mean()))

    def summarize(self) -> EpisodeSummary:
        vertical = self._stack("vertical_acceleration_mps2")
        power = self._stack("electrical_power_w")
        gate = torch.cat(self.gates, dim=0) if self.gates else torch.zeros(1)
        if self.preview_forces and self.feedback_forces:
            preview = torch.cat(self.preview_forces, dim=0).abs().mean()
            feedback = torch.cat(self.feedback_forces, dim=0).abs().mean()
            ratio = float(preview / feedback.clamp_min(1e-6))
        else:
            ratio = 0.0
        return EpisodeSummary(
            rms_vertical_acceleration=self._rms(vertical),
            peak_vertical_acceleration=float(vertical.abs().max()),
            rms_pitch_rate=self._rms(self._stack("pitch_rate_radps")),
            rms_pitch_acceleration=self._rms(
                self._stack("pitch_acceleration_radps2")
            ),
            rms_roll_rate=self._rms(self._stack("roll_rate_radps")),
            rms_roll_acceleration=self._rms(
                self._stack("roll_acceleration_radps2")
            ),
            rms_passenger_acceleration=self._rms(
                self._stack("passenger_acceleration_mps2")
            ),
            max_suspension_travel=float(
                self._stack("suspension_travel_m").abs().max()
            ),
            rms_tire_load_variation=self._rms(
                self._stack("tire_load_variation_n")
            ),
            energy_j=float(power.clamp_min(0.0).sum() * self.sample_time_s),
            saturation_fraction=float(self._stack("saturation").float().mean()),
            rms_action_slew=self._rms(
                self._stack("action_slew_normalized")
            ),
            rms_force_tracking_error=self._rms(
                self._stack("force_tracking_error_n")
            ),
            hard_violation_count=int(
                self._stack("hard_violation").bool().sum()
            ),
            mean_gate=float(gate.mean()),
            preview_to_feedback_ratio=ratio,
        )


@dataclass
class AblationResult:
    controller: str
    parameter_count: int
    episodes: List[EpisodeSummary] = field(default_factory=list)

    def aggregate(self) -> Dict[str, float]:
        if not self.episodes:
            raise ValueError("No episodes were evaluated.")
        output: Dict[str, float] = {}
        for name in self.episodes[0].__dict__:
            values = [float(getattr(episode, name)) for episode in self.episodes]
            output[name] = sum(values) / len(values)
        return output


def parameter_count(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    return sum(parameter.numel() for parameter in module.parameters())


class AblationSuite:
    """Runs every controller on identical scenario, seed, and condition tuples."""

    def __init__(
        self,
        environment: EvaluationEnvironment,
        controllers: Dict[str, Tuple[ForceController, Optional[nn.Module]]],
    ):
        self.environment = environment
        self.controllers = controllers

    def run(
        self,
        scenarios: Iterable[RoadScenario],
        seeds: Iterable[int],
        conditions: Iterable[str],
        maximum_steps: int,
    ) -> Dict[str, AblationResult]:
        results = {
            name: AblationResult(name, parameter_count(module))
            for name, (_, module) in self.controllers.items()
        }
        cases = [
            (scenario, seed, condition)
            for scenario in scenarios
            for seed in seeds
            for condition in conditions
        ]
        for name, (controller, _) in self.controllers.items():
            for scenario, seed, condition in cases:
                controller.reset()
                observation = self.environment.reset(
                    scenario, seed, condition
                )
                accumulator = EpisodeAccumulator(self.environment.sample_time_s)
                for _ in range(maximum_steps):
                    force, auxiliary = controller.act(observation)
                    observation, metrics, done = self.environment.step(force)
                    accumulator.add(metrics, auxiliary)
                    if bool(done.any()):
                        break
                results[name].episodes.append(accumulator.summarize())
        return results


def default_ablation_controllers(
    modular_actor: ModularActor,
    unified_actor: UnifiedActor,
) -> Dict[str, Tuple[ForceController, Optional[nn.Module]]]:
    return {
        "passive": (PassiveController(), None),
        "feedback_only": (
            ModularController(modular_actor, "feedback_only"),
            modular_actor,
        ),
        "preview_only": (
            ModularController(modular_actor, "preview_only"),
            modular_actor,
        ),
        "unified": (UnifiedController(unified_actor), unified_actor),
        "raw_action_convex_gate": (
            ModularController(modular_actor, "convex_gate"),
            modular_actor,
        ),
        "residual_force": (
            ModularController(modular_actor, "full"),
            modular_actor,
        ),
    }
