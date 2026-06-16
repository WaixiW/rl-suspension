# Active Suspension Control (RL-first)

Reinforcement-learning–first active suspension control for a 7-DOF full-car
model with road preview, a forward-only (non-invertible) actuator, an MPC
teacher for warm-starting, and a command-level predictive safety filter.

## Layout

- `active_suspension/` — core package (model, actuator, env, RL, MPC, safety, metrics).
  See [`active_suspension/README.md`](active_suspension/README.md) for the full design overview.
- `scripts/` — training, teacher collection, evaluation, and smoke test.
- `tests/` — component unit tests.
- `runs/` — trained policies (`bc`, `ppo_main`, `ppo_ft`) and the evaluation report (`runs/eval/report.md`).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r active_suspension/requirements.txt
python scripts/smoke_test.py
python scripts/evaluate.py
```

See `active_suspension/README.md` for the controller comparison and validation stages.
