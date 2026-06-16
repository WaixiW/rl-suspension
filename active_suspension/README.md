# RL-First Active Suspension Control

Reference implementation of the project plan: a reinforcement-learning-primary
active-suspension controller with road preview, a 7-DOF full-car model, a
forward-only (non-invertible) actuator, IMU + 4 wheel-height-sensor estimation,
an MPC teacher, and a command-level safety filter.

## Why this design

The actuator model maps commands `(damping_current_1, damping_current_2,
pump_speed)` to a force per corner and is **forward-only / non-invertible**
(3 inputs -> 1 scalar force). So the RL policy outputs the **raw 12 commands**
directly and the forward actuator lives inside the environment (the legged-robot
"actuator-net" pattern). MPC is reused only as (a) an offline teacher and (b) a
runtime safety filter. With no real command-force data, robustness comes from
**wide domain randomization** around the nominal model, not system identification.

## Package layout

```
active_suspension/
  config.py        # vehicle / actuator / preview / sim parameters (corner order FL,FR,RL,RR)
  full_car.py      # 7-DOF full-car model (heave, pitch, roll + 4 unsprung), ZOH discretization
  actuator.py      # forward-only actuator (commands->force + delay) + offline Path-B inverse
  road.py          # bump / pothole / speed-bump / washboard / ISO-8608 random roads
  preview.py       # ADS preview: confidence-weighted, speed-aligned, error-injected look-ahead
  estimator.py     # Kalman filter from IMU + 4 height sensors + observation builder
  metrics.py       # ISO 2631 Wk-weighted RMS comfort + handling / constraint metrics
  reward.py        # comfort-dominant reward with constraint barriers + anti bang-bang penalty
  baselines.py     # passive + semi-active skyhook controllers
  mpc.py           # preview-LQR MPC teacher (desired force) + command labels via Path-B inverse
  safety.py        # command-level predictive safety filter + fallback supervisor
  domain_randomization.py  # wide DR sampler with a `width` knob for the open-risk sweep
  env.py           # gymnasium env: raw 12-command action, forward actuator, DR, safety
  evaluation.py    # shared rollout + scenario evaluation
  policy.py        # SB3 policy inference wrapper (applies VecNormalize stats)
scripts/
  smoke_test.py    # fast end-to-end sanity check
  train_rl.py      # PPO training (domain randomized), optional MPC warm-start
  collect_teacher.py  # MPC teacher -> command labels -> behavior-cloned policy
  evaluate.py      # staged validation suite -> runs/eval/report.md
```

## Environment note

`scipy`/`torch` use BLAS routines that can crash under restrictive sandboxes.
Run scripts in a normal shell. A virtualenv is provided at `.venv-suspension`.

## Quickstart

```bash
pip install -r active_suspension/requirements.txt

# 1. sanity check the whole stack
python scripts/smoke_test.py

# 2. distill the MPC teacher into a behavior-cloned policy (Path B inverse + BC)
python scripts/collect_teacher.py --episodes 36 --epochs 50 --out runs/bc

# 3. train the primary RL policy, warm-started from the teacher
python scripts/train_rl.py --timesteps 500000 --warm_start runs/bc --lr 1e-4 --out runs/ppo_main

# 4. run the staged validation suite (writes runs/eval/report.md + figures)
python scripts/evaluate.py --rl runs/ppo_main --bc runs/bc --seeds 5
```

## Controllers compared

- `passive`  : zero current / zero pump (base damping only) - reference floor.
- `skyhook`  : continuous semi-active baseline.
- `mpc`      : preview-LQR teacher (privileged clean preview).
- `bc`       : teacher-distilled policy (offline, no runtime inverse needed).
- `rl`       : primary RL policy (raw commands, domain randomized, warm-started).

## Validation stages (scripts/evaluate.py)

1. Nominal per-scenario comfort/handling.
2. Monte Carlo under domain randomization (mean +/- std).
3. Preview-error robustness sweep.
4. Domain-randomization width sweep (addresses the open risk: too-narrow is
   fragile, too-wide is over-conservative).
5. Command-level safety-filter effect on a harsh scenario.
6. Fallback-supervisor decision table.
