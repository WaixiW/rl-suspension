# Modular preview-feedback suspension RL

This is a standalone PyTorch reference implementation of the approved control
plan. It intentionally does not depend on any existing project code.

## Implemented control path

1. `RoadPreviewProcessor` converts four 8 m wheel-path profiles into
   `[height, slope, curvature, confidence, arrival_time, valid]` features.
2. `ModularActor` combines:
   - a shared spatial preview encoder and four-corner preview force residual;
   - a GRU feedback encoder and four-corner stabilizing force;
   - a smooth confidence-aware gate.
3. `ForceProjector` applies force, travel, and tire-load limits.
4. `DynamicActuatorAllocator` converts four forces to eight directional damping
   currents and four pump speeds using the actuator force model.
5. `SuspensionReward` combines comfort, attitude, road holding, power, command
   smoothness, force tracking, gate behavior, and hard violations.
6. `TD3Agent` and `StagedTrainer` implement feedback-only, preview-residual,
   gate, and joint fine-tuning phases with one shared twin critic.
7. Domain randomization, ablation metrics, ADS dropout checks, allocator
   bandwidth checks, and deployment go/no-go checks are included.

## Command and tensor contracts

- Wheel order: `FL, FR, RL, RR`.
- Road tensor: `[batch, 4, 6, 160]`.
- Feedback history: `[batch, history_steps, feedback_dim]`.
- Outer RL action: `[batch, 4]` desired corner forces in newtons.
- Hardware command layout:
  `[FL_I1, FL_I2, FR_I1, FR_I2, RL_I1, RL_I2, RR_I1, RR_I2, FL_rpm, FR_rpm, RL_rpm, RR_rpm]`.

The force projector assumes positive force increases positive suspension travel
and tire normal load. Apply a sign transform at the simulator boundary if the
production model uses another convention.

## Production integration boundary

Implement `SevenDOFSimulatorBridge`:

```python
class ProductionBridge:
    def reset(self, request):
        # Configure vehicle/actuator parameters, road, and sensor corruption.
        # Return SimulationFrame(observation, reward_signals, safety_state, done).
        ...

    def step(self, commands):
        # Advance the actuator and 7-DOF models by one outer-loop interval.
        # Return the next SimulationFrame.
        ...
```

Then create the components:

```python
observation_cfg = ObservationConfig(feedback_dim=YOUR_STATE_DIM)
network_cfg = NetworkConfig()
actuator_cfg = ActuatorConfig()  # replace defaults with identified parameters

actor = ModularActor(observation_cfg, network_cfg)
critic = TwinCritic(observation_cfg, network_cfg)
actuator_model = NonlinearActuatorModel(actuator_cfg)
allocator = DynamicActuatorAllocator(actuator_cfg, actuator_model)
projector = ForceProjector(actuator_cfg.force_bounds_n)
supervisor = SafetySupervisor(projector, allocator, fallback=YOUR_FALLBACK)
```

The numerical actuator defaults are placeholders for interface validation, not
identified production values. Calibrate force curves, delays, bounds, travel
limits, tire-load sign, power, and thermal models before meaningful training.

## Recommended training order

1. Validate force tracking and command limits independently.
2. Train `FEEDBACK_ONLY` with preview disabled.
3. Freeze feedback and train `PREVIEW_RESIDUAL`.
4. Freeze force heads and train `GATE_TRAINING` with ADS corruption/dropout.
5. Run `JOINT_FINE_TUNE` at a reduced learning rate.
6. Evaluate all controllers on identical scenario/seed/condition tuples.
7. Require the safety and go/no-go checks before hardware testing.

## Verification

From this directory:

```powershell
$env:PYTHONPATH = "src"
python -m pytest
```
