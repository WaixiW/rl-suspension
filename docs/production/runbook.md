# Production Controller Runbook

## Pre-deployment

1. Confirm the release bundle is immutable and matches the approved change.
2. Verify artifact SHA256 and the manifest signature with the configured trust
   root. Stop if the manifest is unsigned where signatures are mandatory.
3. Verify schema, normalization, envelope, ECU runtime, and fallback versions.
4. Replay golden vectors and run the target runtime latency benchmark.
5. Confirm promotion gates, SBOM policy, vulnerability exceptions, HIL,
   fault-injection, and rollback evidence are approved.
6. Register the model without activation; resolve it from the registry and
   verify its checksum again.

## Staged rollout

1. Install on a non-driving bench and run startup self-test.
2. Run HIL at nominal and boundary conditions.
3. Deploy in shadow mode. Candidate output must not reach actuators.
4. Review action deltas, p99 latency, invalid observations, watchdog misses,
   fallback entries, and recovery behavior.
5. Enable a limited canary only after written approval. Increase exposure in
   bounded stages with a stop decision at every stage.

## Alerts and immediate actions

- Checksum/signature/startup failure: do not activate; retain current version.
- Elevated fallback or invalid observation rate: keep/force fallback, preserve
  the ring buffer, inspect sensor timing and contract versions.
- Deadline misses: keep fallback, capture target load and latency evidence;
  never increase the deadline without requalification.
- Bound/slew projection spike: keep fallback and inspect model units,
  normalization, channel order, and actuator feedback.
- Safety-envelope excursion: stop the test or vehicle according to site safety
  procedure; do not resume solely because the signal returns in range.
- Suspected compromise: isolate deployment tooling, revoke signing/deployment
  access, preserve evidence, and invoke the security incident process.

## Rollback

1. Latch the deterministic fallback.
2. Preserve the current ring buffer, model identity, counters, and clock state.
3. Ask the registry for the approved predecessor and verify its checksum.
4. Activate the predecessor atomically, run startup/golden checks, then release
   fallback only through normal healthy-cycle recovery.
5. Record trigger, affected versions, evidence, operator, and timestamps.

Do not delete or overwrite the failed artifact. Registry history and manifests
are incident evidence.

## Recovery and closeout

Confirm stable fallback rate, no watchdog misses, no action-limit anomalies,
valid sensor freshness, and expected shadow deltas. Link logs, evaluation,
root-cause analysis, corrective actions, and approval to resume. Any model,
runtime, envelope, normalization, contract, or hardware change requires scoped
requalification.
