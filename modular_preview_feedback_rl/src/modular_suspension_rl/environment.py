from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from .contracts import ObservationBatch
from .networks import ActorOutput
from .objective import RewardSignals, SuspensionReward
from .randomization import (
    ADSCorruptor,
    DomainRandomizer,
    DomainSample,
    RoadScenario,
    RoadScenarioGenerator,
)
from .safety import SafetyDecision, SafetyState, SafetySupervisor
from .training import TrainingPhase


@dataclass
class ResetRequest:
    domain: DomainSample
    clean_road_height_m: Tensor
    measured_road_height_m: Tensor
    road_confidence: Tensor
    road_valid_mask: Tensor
    scenarios: tuple
    phase: TrainingPhase


@dataclass
class SimulationFrame:
    observation: ObservationBatch
    reward_signals: RewardSignals
    safety_state: SafetyState
    done: Tensor


class SevenDOFSimulatorBridge(Protocol):
    """Boundary to the production vehicle, sensor, and actuator simulation."""

    def reset(self, request: ResetRequest) -> SimulationFrame:
        ...

    def step(self, commands: Tensor) -> SimulationFrame:
        ...


class ModularSuspensionEnvironment:
    """Connects RL force actions to safety, allocation, reward, and simulation."""

    def __init__(
        self,
        simulator: SevenDOFSimulatorBridge,
        supervisor: SafetySupervisor,
        reward: SuspensionReward,
        domain_randomizer: DomainRandomizer,
        road_generator: RoadScenarioGenerator,
        ads_corruptor: ADSCorruptor,
        batch_size: int = 1,
        curriculum_stage: int = 0,
    ):
        self.simulator = simulator
        self.supervisor = supervisor
        self.reward_calculator = reward
        self.domain_randomizer = domain_randomizer
        self.road_generator = road_generator
        self.ads_corruptor = ads_corruptor
        self.batch_size = batch_size
        self.curriculum_stage = curriculum_stage
        self.frame = None

    def set_curriculum_stage(self, stage: int) -> None:
        if not 0 <= stage <= self.domain_randomizer.maximum_stage:
            raise ValueError("Invalid curriculum stage.")
        self.curriculum_stage = stage

    def reset(self, phase: TrainingPhase) -> ObservationBatch:
        severity = self.curriculum_stage / max(
            self.domain_randomizer.maximum_stage, 1
        )
        domain = self.domain_randomizer.sample(
            self.batch_size, self.curriculum_stage
        )
        clean_road, scenarios = self.road_generator.generate(
            self.batch_size, severity=max(severity, 0.05)
        )
        measured, confidence, valid = self.ads_corruptor.apply(
            clean_road, domain
        )
        request = ResetRequest(
            domain=domain,
            clean_road_height_m=clean_road,
            measured_road_height_m=measured,
            road_confidence=confidence,
            road_valid_mask=valid,
            scenarios=scenarios,
            phase=phase,
        )
        self.frame = self.simulator.reset(request)
        return self.frame.observation

    def reset_for_evaluation(
        self,
        scenario: RoadScenario,
        seed: int,
        condition: str,
    ) -> ObservationBatch:
        del condition
        self.domain_randomizer.generator.manual_seed(seed)
        self.road_generator.generator.manual_seed(seed)
        domain = self.domain_randomizer.sample(
            self.batch_size, self.curriculum_stage
        )
        clean_road, scenarios = self.road_generator.generate(
            self.batch_size,
            scenario=scenario,
            severity=max(
                self.curriculum_stage
                / max(self.domain_randomizer.maximum_stage, 1),
                0.05,
            ),
        )
        measured, confidence, valid = self.ads_corruptor.apply(
            clean_road, domain
        )
        self.frame = self.simulator.reset(
            ResetRequest(
                domain=domain,
                clean_road_height_m=clean_road,
                measured_road_height_m=measured,
                road_confidence=confidence,
                road_valid_mask=valid,
                scenarios=scenarios,
                phase=TrainingPhase.JOINT_FINE_TUNE,
            )
        )
        return self.frame.observation

    def step(
        self,
        raw_force_n: Tensor,
        actor_output: ActorOutput,
    ):
        if self.frame is None:
            raise RuntimeError("reset must be called before step.")
        if raw_force_n.shape != actor_output.raw_force_n.shape:
            raise ValueError("raw_force_n and actor output must have matching shapes.")
        execution_output = ActorOutput(
            raw_force_n=raw_force_n,
            feedback_force_n=actor_output.feedback_force_n,
            preview_force_n=actor_output.preview_force_n,
            gate=actor_output.gate,
            gate_raw=actor_output.gate_raw,
            actuator_authority=actor_output.actuator_authority,
        )
        decision: SafetyDecision = self.supervisor.execute(
            self.frame.observation,
            execution_output,
            self.frame.safety_state,
        )
        next_frame = self.simulator.step(decision.allocation.commands)
        signals = next_frame.reward_signals
        signals.gate = actor_output.gate
        signals.previous_gate = self.frame.observation.previous_gate
        signals.preview_confidence = self.frame.observation.preview_confidence
        breakdown = self.reward_calculator(signals)
        self.frame = next_frame
        info = {
            "executed_force_n": decision.projected.force_n,
            "commands": decision.allocation.commands,
            "projection_correction_n": decision.projected.correction_n,
            "fallback_active": decision.fallback_active,
            "hard_violation": breakdown.hard_violation,
        }
        return (
            next_frame.observation,
            breakdown.reward,
            next_frame.done,
            info,
        )
