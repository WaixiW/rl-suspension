# Security, SBOM, Signing, and Access Guidance

## Artifact and dependency integrity

- Generate a machine-readable CycloneDX or SPDX SBOM for the training/export
  environment and a separate SBOM for the ECU runtime image.
- Pin and archive resolved direct/transitive dependencies, compiler/runtime
  versions, model opset, base image/firmware identity, and build provenance.
- The server image consumes `deploy/production/constraints.lock`; changes to
  that file or the base-image digest require integration recertification.
- Scan both SBOMs before release. Record severity policy, exploitability,
  remediation, compensating control, owner, expiry, and approval for every
  exception.
- Re-scan supported releases when vulnerability intelligence changes.
- Treat models, normalization, envelopes, golden vectors, manifests, and
  deployment configuration as executable supply-chain artifacts.

## Signing

- Use an approved asymmetric key in an HSM, KMS, or signing service through the
  `ManifestSigner` interface. Private keys must not be stored in this
  repository, build logs, model bundles, or ECU storage.
- Sign the canonical manifest payload containing artifact SHA256 and metadata.
  A SHA256 digest alone is integrity evidence, not a cryptographic signature.
- Verify against a pinned trust root before registration and again before
  activation. Reject unknown algorithms/keys, malformed signatures, checksum
  mismatch, downgrade, and expired/revoked approval.
- Separate build, approval, signing, and deployment identities. Log key ID,
  release ID, requester, approver, result, and timestamp without logging secret
  key material.

## Access control

- Apply least privilege and short-lived credentials to datasets, model
  registry, signer, HIL, telemetry, ECU deployment, and rollback operations.
- Require multi-factor authentication and independent approval for production
  activation, trust-root changes, envelope widening, and emergency access.
- Developers may create candidates but cannot unilaterally sign and deploy
  them. ECU runtime identities are read-only for approved model retrieval.
- Review service and human access at a fixed cadence and immediately after
  role change. Disable dormant credentials and prohibit shared accounts.

## Secrets and data

- Store secrets in the organization secret manager; inject them only for the
  minimum operation. Never place secrets in model metadata, manifests, SBOMs,
  golden files, ring logs, or source control.
- Classify raw vehicle/simulator data and telemetry. Minimize collection,
  encrypt in transit/at rest, enforce retention/deletion, and redact operator
  or vehicle identifiers from engineering exports.

## Audit and incident response

Retain immutable build provenance, SBOMs, scan results, manifests, signature
verification, approvals, registry transitions, deployment results, and
ring-buffer incident exports according to policy. Alert on failed verification,
unexpected key use, repeated rollback, privilege escalation, and artifact
mutation. For suspected compromise, latch fallback, halt promotion, preserve
evidence, revoke affected credentials/keys, identify all releases signed or
built in scope, and reissue only after trusted rebuild and qualification.
