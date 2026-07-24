# RL Suspension

Single-agent RL scaffold for active suspension control.

The controller architecture follows the project plan:

```text
vehicle state + ADS preview + actuator memory
  -> single SAC agent
  -> desired corner forces [F_FL, F_FR, F_RL, F_RR]
  -> deterministic actuator allocator
  -> 8 damping currents + 4 pump speeds
  -> nonlinear 7-DOF suspension model
```

## Install

```bash
pip install -e .[dev]
```

## Train SAC

```bash
python -m rl_suspension.training.train_sac --output-dir runs/sac_active_suspension
```

For a quick smoke run:

```bash
python -m rl_suspension.training.train_sac --timesteps-scale 0.001
```

## Evaluate Baselines

```bash
python -m rl_suspension.evaluation.evaluate --policy passive --episodes 5 --curriculum-stage 1
python -m rl_suspension.evaluation.evaluate --policy skyhook --episodes 5 --curriculum-stage 2
python -m rl_suspension.evaluation.evaluate --policy preview --episodes 5 --curriculum-stage 4
```

Evaluate a trained SAC policy:

```bash
python -m rl_suspension.evaluation.evaluate --policy sac --model-path runs/sac_active_suspension/sac_active_suspension_final
```

## Test

```bash
pytest
```

## Adapt Your Numba Car Dynamics Model

If you already have a self-made numba vehicle model, keep it as the source of truth and wrap it with the training-frame interface. The RL environment only needs a small adapter around your model.

Your numba model currently has this shape:

```text
inputs:
  road_profile
  initial_car_states
  actions

outputs:
  14 vehicle states
  suspension acceleration at car body center
```

The training frame expects this shape:

```python
next_state, output = suspension_model.step(
    state=state,
    action_12d=action_12d,
    road_profile=road_profile,
    dt=dt,
    actuator_state=actuator_state,
)
```

So the adapter should convert between the two APIs:

```text
SuspensionState(q, qd)
  -> 14-state vector
  -> numba_model(road_profile, initial_car_states, actions)
  -> 14-state vector + body center acceleration
  -> StepResult(next_state, output)
```

### 1. Preserve the State Order

Use one fixed 14-state convention everywhere. The current scaffold assumes:

```text
q:
  0 body heave
  1 pitch
  2 roll
  3 front-left unsprung vertical displacement
  4 front-right unsprung vertical displacement
  5 rear-left unsprung vertical displacement
  6 rear-right unsprung vertical displacement

qd:
  7 body heave velocity
  8 pitch rate
  9 roll rate
  10 front-left unsprung vertical velocity
  11 front-right unsprung vertical velocity
  12 rear-left unsprung vertical velocity
  13 rear-right unsprung vertical velocity
```

If your numba model uses a different order, do the reorder only inside the adapter.

### 2. Match the Action Convention

The RL policy outputs four desired corner forces:

```text
[F_FL, F_FR, F_RL, F_RR]
```

The actuator allocator converts them to the 12 physical commands:

```text
[I_FL_1, I_FL_2, rpm_FL,
 I_FR_1, I_FR_2, rpm_FR,
 I_RL_1, I_RL_2, rpm_RL,
 I_RR_1, I_RR_2, rpm_RR]
```

Your numba model should receive this 12D `actions` vector. If your model expects a `(12, 1)` column vector, reshape in the adapter:

```python
actions_for_numba = action_12d.reshape(12, 1)
```

### 3. Match the Road Profile

The scaffold passes the road boundary at the four wheels for the current step:

```text
[road_FL, road_FR, road_RL, road_RR]
```

If your numba model expects the full ADS preview instead of four wheel-contact heights, change the environment step to pass `self.road.preview(self.vehicle_x)` into the adapter, or make the adapter reconstruct the model-specific road array from the available road profile.

Recommended rule:

```text
RL observation uses ADS preview.
Plant dynamics step uses the road boundary needed by your numba model.
```

### 4. Implement an Adapter Class

Create a wrapper class such as `rl_suspension/models/numba_adapter.py`:

```python
import numpy as np

from rl_suspension.models.types import (
    ActuatorState,
    StepResult,
    SuspensionOutput,
    SuspensionState,
)


class NumbaCarModelAdapter:
    def __init__(self, numba_step_fn, params):
        self.numba_step_fn = numba_step_fn
        self.params = params

    def step(self, state, action_12d, road_profile, dt, actuator_state=None):
        x0 = state.as_vector()
        actions = np.asarray(action_12d, dtype=np.float64).reshape(12, 1)
        road = np.asarray(road_profile, dtype=np.float64)

        x_next, body_center_acc = self.numba_step_fn(
            road,
            x0,
            actions,
        )

        next_state = SuspensionState.from_vector(np.asarray(x_next, dtype=np.float64).reshape(14))

        output = SuspensionOutput(
            body_acceleration=float(body_center_acc),
            pitch_acceleration=0.0,
            roll_acceleration=0.0,
            suspension_deflections=np.zeros(4, dtype=np.float64),
            suspension_velocities=np.zeros(4, dtype=np.float64),
            tire_deflections=np.zeros(4, dtype=np.float64),
            tire_loads=np.zeros(4, dtype=np.float64),
            actual_corner_forces=(actuator_state.forces.copy() if actuator_state else np.zeros(4)),
            actuator_state=actuator_state or ActuatorState(),
            constraint_violations={
                "suspension_travel": 0.0,
                "tire_lift": 0.0,
            },
        )
        return StepResult(next_state=next_state, output=output)
```

This minimal adapter is enough to train, but the zeros should be replaced with real values from your model when available.

### 5. Add Real Diagnostics When Possible

The reward and evaluation are much better if the adapter can output:

```text
pitch acceleration
roll acceleration
4 suspension deflections
4 suspension velocities
4 tire loads or tire deflections
actual corner forces
constraint violations
```

If your numba model only returns `14 states + body center acceleration`, you can still start training, but the reward should initially focus on:

```text
body vertical acceleration
action force penalty
action rate penalty
estimated suspension travel from state if possible
```

Then extend the numba model outputs later.

### 6. Use the Adapter in the Environment

In `rl_suspension/envs/active_suspension_env.py`, replace:

```python
self.model = SevenDofSuspensionModel(self.config.vehicle_params)
```

with:

```python
self.model = NumbaCarModelAdapter(numba_step_fn=my_numba_step, params=my_params)
```

The rest of the RL frame can stay the same:

```text
SAC policy -> 4 desired forces -> actuator allocator -> 12 actions -> numba car model
```

### 7. Validate Before Training

Before running SAC, check these cases:

```text
flat road + zero action: stable response
single bump + zero action: passive response is reasonable
single bump + fixed force command: action sign is correct
action reshape: model receives the expected 12 command order
state reorder: returned 14 states match the scaffold convention
body acceleration unit: m/s^2, not g
```

Once these pass, run:

```bash
python -m rl_suspension.evaluation.evaluate --policy passive --episodes 1 --curriculum-stage 1
python -m rl_suspension.training.train_sac --timesteps-scale 0.001
```

## Behavior Cloning and DAgger Sandbox

The imitation package is teacher-agnostic.
The deployable observation is now 484D: the original 50 state/diagnostic
features followed by 217 left-road and 217 right-road samples from the rear
axle through 8 m ahead of the front axle. Legacy datasets and checkpoints made
with the 50D observation must be regenerated.

Tune and qualify the OSQP-backed full-car preview MPC:

```bash
python -m rl_suspension.controllers.mpc.tune_mpc \
  --output runs/mpc/tuning \
  --episodes 3 \
  --curriculum-stage 5 \
  --horizon 30

python -m rl_suspension.controllers.mpc.benchmark_mpc \
  --config runs/mpc/tuning/best_mpc_config.json \
  --output runs/mpc/held_out.json \
  --episodes 3
```

MPC is the default imitation teacher. Collection refuses to use it unless its
held-out return and safety gate beat passive control; `--allow-unqualified`
exists only for plumbing tests.

Collect phase-balanced expert episodes:

```bash
python -m rl_suspension.imitation.collect \
  --episodes 500 \
  --teacher mpc \
  --mpc-config runs/mpc/tuning/best_mpc_config.json \
  --output data/mpc_expert
```

Train the actual SAC actor with behavior cloning:

```bash
python -m rl_suspension.imitation.train_bc \
  --dataset data/mpc_expert \
  --loss huber_smooth \
  --output runs/imitation/bc
```

Run five DAgger aggregation rounds:

```bash
python -m rl_suspension.imitation.train_dagger \
  --dataset data/mpc_expert \
  --initial-model runs/imitation/bc/bc_sac_actor.zip \
  --rounds 5 \
  --beta 1,.75,.5,.25,0 \
  --teacher mpc \
  --mpc-config runs/mpc/tuning/best_mpc_config.json \
  --output runs/imitation/dagger
```

Run the complete MSE/Huber/DAgger/SafeDAgger comparison:

```bash
python -m rl_suspension.imitation.run_benchmark \
  --dataset data/mpc_expert \
  --output runs/imitation/benchmark
```

The dataset stores both `actions` (expert labels) and `behavior_actions`
(commands actually executed during DAgger). This distinction is required if
the transitions are later reused for SAC critic training. MPC shards also
store solver status, objective, iterations, latency, constraint margin, and
fallback metadata for every label.

## Private-Server Production Pipeline

The `rl_suspension.production` package is a separate integration boundary for
the business MPC, simulator, and RL framework. Unlike the four-force sandbox,
the production contract clones the MPC's twelve physical actuator commands
directly. Private implementations are loaded as `package.module:factory`
plugins, so proprietary solver and simulator code does not need to be copied
into this repository.

Before data collection, certify the adapter contracts:

```bash
rl-suspension-production certify \
  --mpc-plugin private_suspension.mpc:create_adapter \
  --simulator-plugin private_suspension.simulator:create_adapter \
  --scenario-json config/contract_scenario.json \
  --output runs/production/contract-certification.json
```

For an installation smoke test without private dependencies:

```bash
python -m rl_suspension.production.cli certify \
  --reference \
  --output runs/production/reference-certification.json
```

Start from `config/production.example.json` and
`config/action_schema.example.json`. The physical action order, bounds, safe
values, and slew rates must be reviewed against the actual vehicle interface
before any private-server collection or ECU export.

Operational integration, security, validation, HIL, model-card, and rollback
documents are under `docs/production/`.
