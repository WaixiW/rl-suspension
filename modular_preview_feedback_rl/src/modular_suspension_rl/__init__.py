"""Physics-aligned modular preview-feedback suspension RL reference package."""

from .actuation import (
    DynamicActuatorAllocator,
    ForceProjector,
    NonlinearActuatorModel,
    pack_commands,
    unpack_commands,
)
from .config import (
    ActuatorConfig,
    NetworkConfig,
    ObservationConfig,
    RewardConfig,
    TD3Config,
    WHEEL_ORDER,
)
from .contracts import (
    ObservationBatch,
    PhysicalNormalizer,
    RoadPreviewProcessor,
    concatenate_feedback_state,
)
from .environment import (
    ModularSuspensionEnvironment,
    ResetRequest,
    SevenDOFSimulatorBridge,
    SimulationFrame,
)
from .evaluation import AblationSuite, ModularController, UnifiedController
from .networks import ActorOutput, ModularActor, TwinCritic, UnifiedActor
from .objective import RewardSignals, SuspensionReward
from .randomization import (
    ADSCorruptor,
    DomainRandomizer,
    RoadScenario,
    RoadScenarioGenerator,
    SensorCorruptor,
)
from .safety import (
    SafetyConfig,
    SafetyState,
    SafetySupervisor,
    deployment_go_no_go,
    measure_allocator_step_response,
    verify_policy_safety,
)
from .training import (
    PhaseBudget,
    ReplayBuffer,
    StagedTrainer,
    TD3Agent,
    TrainingPhase,
)

__all__ = [
    "ADSCorruptor",
    "ActuatorConfig",
    "ActorOutput",
    "AblationSuite",
    "DomainRandomizer",
    "DynamicActuatorAllocator",
    "ForceProjector",
    "ModularActor",
    "ModularController",
    "ModularSuspensionEnvironment",
    "NetworkConfig",
    "NonlinearActuatorModel",
    "ObservationBatch",
    "ObservationConfig",
    "PhaseBudget",
    "PhysicalNormalizer",
    "ReplayBuffer",
    "ResetRequest",
    "RewardConfig",
    "RewardSignals",
    "RoadPreviewProcessor",
    "RoadScenario",
    "RoadScenarioGenerator",
    "SafetyConfig",
    "SafetyState",
    "SafetySupervisor",
    "SensorCorruptor",
    "SevenDOFSimulatorBridge",
    "SimulationFrame",
    "StagedTrainer",
    "SuspensionReward",
    "TD3Agent",
    "TD3Config",
    "TrainingPhase",
    "TwinCritic",
    "UnifiedActor",
    "UnifiedController",
    "WHEEL_ORDER",
    "concatenate_feedback_state",
    "deployment_go_no_go",
    "measure_allocator_step_response",
    "pack_commands",
    "unpack_commands",
    "verify_policy_safety",
]
