# Production Traceability Matrix

Each requirement has an implementation point, verification evidence, and an
owner-assigned release artifact. Replace `TBD` with the immutable evidence URI
and approval before promotion.

- **PROD-EVAL-001 — Paired baselines.** Passive, MPC, and student execute the
  same scenario IDs and seeds. Implementation:
  `production/evaluation/closed_loop.py`. Evidence: paired report, TBD.
- **PROD-EVAL-002 — Statistical uncertainty.** Controller means and paired
  deltas use deterministic percentile-bootstrap confidence intervals.
  Implementation: `production/evaluation/metrics.py`. Evidence: evaluation
  report, TBD.
- **PROD-EVAL-003 — Open-loop fidelity.** Every action channel reports level
  and temporal-delta errors. Implementation:
  `production/evaluation/metrics.py`. Evidence: open-loop report, TBD.
- **PROD-GATE-001 — Promotion.** MPC improvement retention, confidence option,
  safety, bounds, slew, and p99 latency are non-bypassable checks.
  Implementation: `production/evaluation/gates.py`. Evidence: promotion
  decision, TBD.
- **PROD-ART-001 — Fixed graph.** ECU ONNX inputs/outputs have concrete
  contract dimensions. Implementation: `production/deployment/export.py`.
  Evidence: checked ONNX graph, TBD.
- **PROD-ART-002 — Reproducibility.** Framework/export and quantized candidates
  replay immutable golden vectors within approved tolerances. Implementation:
  `production/deployment/golden.py` and `quantization.py`. Evidence: golden and
  quantization reports, TBD.
- **PROD-ART-003 — Integrity/authenticity.** Artifact SHA256 is mandatory;
  authenticity uses an approved pluggable signer/verifier and never a
  placeholder signature. Implementation: `production/deployment/manifest.py`.
  Evidence: manifest verification, TBD.
- **PROD-TIME-001 — Deadline.** Runtime benchmark and ECU watchdog enforce the
  approved latency limit. Implementation: `benchmark.py` and `runtime.py`.
  Evidence: target benchmark/watchdog logs, TBD.
- **PROD-SAFE-001 — Input safety.** Shape/version, finite values, freshness,
  validity, and engineering envelopes are checked every cycle. Implementation:
  `production/deployment/runtime.py`. Evidence: unit/fault/HIL reports, TBD.
- **PROD-SAFE-002 — Output safety.** Physical bounds and slew limits project
  every primary and fallback command. Implementation:
  `production/deployment/runtime.py`. Evidence: tests/HIL report, TBD.
- **PROD-SAFE-003 — Fail-operational behavior.** Watchdog faults select local
  fallback with latch/dwell/recovery hysteresis and bounded ring logging.
  Implementation: `production/deployment/runtime.py`. Evidence: injected-fault
  report, TBD.
- **PROD-LIFE-001 — Rollback.** Immutable versions are checksum-verified before
  activation and rollback. Implementation: `production/deployment/registry.py`.
  Evidence: registry and rollback drill, TBD.
- **PROD-OPS-001 — Staging.** HIL, non-actuating shadow, and fault campaigns
  have adapter interfaces and auditable reports. Implementation:
  `production/deployment/orchestration.py`. Evidence: campaign reports, TBD.
- **PROD-SEC-001 — Supply chain/access.** SBOM, vulnerability handling,
  least-privilege signing/deployment access, and audit retention are release
  requirements. Guidance: `docs/production/security_sbom_access.md`. Evidence:
  SBOM/access review, TBD.
