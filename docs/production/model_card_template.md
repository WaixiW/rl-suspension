# Production Model Card Template

## Release identity

- Model name/version and registry rollback ID:
- Source revision and training run:
- Artifact SHA256 and signed-manifest key ID:
- ONNX opset/runtime/target ECU:
- Observation, action, normalization, and envelope versions:
- Owners, reviewers, and approval date:

## Intended operation

- Supported vehicle/configuration:
- Supported road/speed/environment envelope:
- Control objective:
- Explicit exclusions:
- Deterministic safe fallback version:

## Architecture and training

- Framework architecture and parameter count:
- Inputs/outputs and physical units:
- Training/validation datasets:
- Objective, optimization, and seeds:
- Distillation/DAgger/MPC expert versions:
- Quantization or graph transformations:

## Qualification evidence

- Open-loop per-channel and temporal-delta results:
- Paired closed-loop passive/MPC/student results and bootstrap confidence:
- MPC-improvement retention:
- Safety, action-bound, and slew results:
- Golden-vector framework/export/quantized equivalence:
- Target-hardware p50/p95/p99/max latency and deadline misses:
- Shadow, injected-fault, HIL, and rollback evidence locations:

## Limitations and risk

- Known failure modes:
- Sensitivity to preview/sensor degradation:
- OOD behavior:
- Unsafe assumptions:
- Residual risk and compensating supervisor controls:

## Monitoring and rollback

- Runtime counters and alert thresholds:
- Ring-buffer retention/export procedure:
- Fallback-rate and deadline SLO:
- Rollback trigger and approved predecessor:
- Incident/runbook link:

## Approval

Record independent safety, controls, ML, platform, security, and release
approvals. Approval of a model does not approve a changed dataset, runtime,
envelope, normalization, or actuator contract.
