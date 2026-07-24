# Production Integration Specification

## Scope

This boundary connects a versioned `observation.preview12.v1` source, a learned
policy, a deterministic safe controller, and a `action.direct12.v1` actuator
sink. Private simulator, MPC, ECU, HSM, and HIL implementations must satisfy the
protocols in `rl_suspension.production`.

## Fixed data contract

- Control period: 10 ms.
- State input: one batch of 53 finite values.
- Road input: one batch of 4 by 217 finite values. Channels are left height,
  right height, left validity, and right validity.
- Output: one batch of 12 physical commands in the ordering defined by
  `ActionSchema.names`.
- Export `PhysicalActionExportWrapper(Direct12Student(...))`; exporting the
  bare student would emit normalized `[0,1]` values instead of physical units.
- Timestamps: monotonic nanoseconds from the same clock domain used by the ECU
  supervisor. Wall-clock timestamps are not accepted at this boundary.
- Schema versions, normalization constants, engineering envelopes, and action
  units are release-controlled artifacts.

ONNX models must have concrete dimensions. Dynamic batch, road, or channel
dimensions are prohibited in the ECU artifact.

## Cycle sequence

1. Acquire one coherent sensor/preview frame and assign its monotonic timestamp.
2. Validate schema, shape, finite values, validity channels, freshness, and
   engineering envelopes.
3. Execute the candidate with the configured watchdog.
4. Select candidate or safe fallback using fault latch/recovery hysteresis.
5. Project the selected command onto physical bounds and per-cycle slew limits.
6. Write the applied command and one supervisor ring-buffer record.

The fallback is local and deterministic. A network service, MPC solver, logger,
or telemetry connection must never be in the real-time fallback path.

## Lifecycle and error behavior

- Startup remains in the safe controller until model checksum, signed manifest
  (when signatures are required), golden vectors, and runtime self-test pass.
- Stale/future observations, NaN/Inf, shape/version mismatch, OOD values,
  policy exceptions, invalid outputs, and deadline misses are faults.
- A fault selects fallback immediately; repeated faults latch fallback.
- Recovery requires both the minimum fallback dwell and consecutive healthy
  candidate cycles. Operators cannot bypass bounds or slew projection.
- Missing telemetry never changes the selected control command.

## Artifact handoff

Every release bundle contains the fixed-shape model, SHA256 manifest, detached
signature block produced by an approved signer, golden vectors, normalization
and envelope versions, evaluation report, latency report, SBOM, model card, and
rollback identifier. An unsigned manifest is explicitly represented as
unsigned; no placeholder or digest-as-signature is permitted.

## Acceptance

Promotion requires paired scenario evaluation against passive, MPC, and the
student; open-loop channel/delta errors; retained MPC improvement; all safety
and action gates; target-hardware p99 latency; golden-vector equivalence; and
successful shadow, fault-injection, and HIL evidence.
