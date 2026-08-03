import torch

from conftest import make_observation
from modular_suspension_rl.actuation import (
    DynamicActuatorAllocator,
    ForceProjector,
    NonlinearActuatorModel,
    unpack_commands,
)
from modular_suspension_rl.config import (
    ActuatorConfig,
    NetworkConfig,
    ObservationConfig,
)
from modular_suspension_rl.networks import ModularActor
from modular_suspension_rl.safety import (
    SafetyState,
    SafetySupervisor,
    measure_allocator_step_response,
    verify_policy_safety,
)


def make_allocator():
    config = ActuatorConfig(allocator_iterations=8)
    model = NonlinearActuatorModel(config)
    return config, DynamicActuatorAllocator(config, model)


def test_force_projector_respects_force_travel_and_tire_limits():
    projector = ForceProjector((-6000.0, 6000.0))
    requested = torch.tensor([[8000.0, 2000.0, -3000.0, -1000.0]])
    travel = torch.tensor([[0.0, 0.085, 0.0, 0.0]])
    tire_load = torch.tensor([[3500.0, 3500.0, 100.0, 3500.0]])

    result = projector(requested, travel, tire_load)

    assert result.force_n.abs().max() <= 6000.0
    assert result.force_n[0, 1] < requested[0, 1]
    assert result.force_n[0, 2] == 0.0
    assert result.active.any()


def test_allocator_respects_command_and_slew_bounds():
    config, allocator = make_allocator()
    target = torch.full((2, 4), 1500.0)
    velocity = torch.zeros(2, 4)
    force = torch.zeros(2, 4)
    previous = torch.zeros(2, 12)

    result = allocator.allocate(target, velocity, force, previous)
    currents, rpm = unpack_commands(result.commands)

    assert torch.all(currents >= config.current_bounds_a[0])
    assert torch.all(currents <= config.current_bounds_a[1])
    assert torch.all(rpm >= config.rpm_bounds[0])
    assert torch.all(rpm <= config.rpm_bounds[1])
    assert torch.all(
        currents
        <= config.current_slew_a_per_s * config.sample_time_s + 1e-5
    )
    assert torch.all(
        rpm.abs() <= config.rpm_slew_per_s * config.sample_time_s + 1e-5
    )


def test_allocator_step_response_settles_for_reachable_force():
    _, allocator = make_allocator()
    report = measure_allocator_step_response(
        allocator,
        target_force_n=1500.0,
        duration_s=0.5,
    )

    assert report.settled
    assert report.final_tracking_error_n < 100.0


def test_runtime_safety_verifies_dropout_and_limits():
    observation_config = ObservationConfig(feedback_dim=20, feedback_history=4)
    network_config = NetworkConfig(
        preview_channels=(16, 24),
        state_hidden_dim=24,
        fused_hidden_dim=32,
    )
    actor = ModularActor(observation_config, network_config)
    observation = make_observation(observation_config)
    config, allocator = make_allocator()
    projector = ForceProjector(config.force_bounds_n)
    supervisor = SafetySupervisor(projector, allocator)
    safety_state = SafetyState(
        suspension_travel_m=torch.zeros(2, 4),
        tire_load_n=torch.full((2, 4), 3500.0),
        sensor_valid=torch.ones(2, dtype=torch.bool),
        hardware_healthy=torch.ones(2, dtype=torch.bool),
    )

    report = verify_policy_safety(
        actor, observation, supervisor, safety_state
    )

    assert report.passed
    assert report.dropout_force_difference_n == 0.0
