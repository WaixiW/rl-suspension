import numpy as np

from rl_suspension.actuators import ActuatorAllocator
from rl_suspension.models import ActuatorState


def test_allocator_returns_12d_action_and_limited_force():
    allocator = ActuatorAllocator()
    result = allocator.allocate(
        desired_forces=np.array([6000.0, -6000.0, 1000.0, -1000.0]),
        suspension_velocities=np.array([0.2, -0.2, 0.1, -0.1]),
        previous_state=ActuatorState(),
        dt=0.01,
    )

    assert result.action_12d.shape == (12,)
    assert result.actuator_state.currents.shape == (8,)
    assert result.actuator_state.pump_speeds.shape == (4,)
    assert result.actuator_state.forces.shape == (4,)
    assert np.all(np.abs(result.feasible_forces) <= 5000.0)
    assert result.saturated[:2].all()
