"""Checksummed model manifests with pluggable real signing backends."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class ManifestSigner(Protocol):
    """Adapter for a KMS, HSM, or approved asymmetric signing service."""

    algorithm: str
    key_id: str

    def sign(self, payload: bytes) -> bytes: ...


@runtime_checkable
class ManifestVerifier(Protocol):
    algorithm: str
    key_id: str

    def verify(self, payload: bytes, signature: bytes) -> bool: ...


@dataclass(frozen=True)
class SignatureBlock:
    algorithm: str
    key_id: str
    value_base64: str


@dataclass(frozen=True)
class ModelManifest:
    schema_version: str
    artifact_name: str
    artifact_sha256: str
    artifact_size_bytes: int
    created_utc: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    signature: SignatureBlock | None = None

    def signing_payload(self) -> bytes:
        payload = asdict(self)
        payload.pop("signature", None)
        return _canonical_json(payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManifestVerification:
    passed: bool
    checksum_valid: bool
    signature_required: bool
    signature_valid: bool | None
    errors: tuple[str, ...]

    def __bool__(self) -> bool:
        return self.passed


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    artifact_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    signer: ManifestSigner | None = None,
    created_utc: str | None = None,
) -> ModelManifest:
    """Build a SHA256 manifest and sign it only when a signer is supplied."""

    artifact = Path(artifact_path)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    unsigned = ModelManifest(
        schema_version="model-manifest.v1",
        artifact_name=artifact.name,
        artifact_sha256=sha256_file(artifact),
        artifact_size_bytes=artifact.stat().st_size,
        created_utc=created_utc
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        metadata=dict(metadata or {}),
    )
    if signer is None:
        return unsigned
    signature = signer.sign(unsigned.signing_payload())
    if not isinstance(signature, bytes) or not signature:
        raise ValueError("manifest signer must return nonempty bytes")
    if not signer.algorithm.strip() or not signer.key_id.strip():
        raise ValueError("manifest signer must identify its algorithm and key")
    return ModelManifest(
        **{
            **asdict(unsigned),
            "signature": SignatureBlock(
                algorithm=signer.algorithm,
                key_id=signer.key_id,
                value_base64=base64.b64encode(signature).decode("ascii"),
            ),
        }
    )


def write_manifest(manifest: ModelManifest, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def load_manifest(path: str | Path) -> ModelManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    signature = payload.get("signature")
    payload["signature"] = (
        SignatureBlock(**signature) if signature is not None else None
    )
    return ModelManifest(**payload)


def verify_manifest(
    manifest: ModelManifest,
    artifact_path: str | Path,
    *,
    verifier: ManifestVerifier | None = None,
    require_signature: bool = False,
) -> ManifestVerification:
    """Verify artifact integrity and, when configured, authenticity."""

    artifact = Path(artifact_path)
    errors: list[str] = []
    checksum_valid = bool(
        artifact.is_file()
        and artifact.stat().st_size == manifest.artifact_size_bytes
        and sha256_file(artifact) == manifest.artifact_sha256
    )
    if not checksum_valid:
        errors.append("artifact checksum or size mismatch")

    signature_valid: bool | None = None
    if manifest.signature is None:
        if require_signature:
            errors.append("manifest is unsigned")
    elif verifier is None:
        if require_signature:
            errors.append("signature verifier was not supplied")
    else:
        block = manifest.signature
        if (
            verifier.algorithm != block.algorithm
            or verifier.key_id != block.key_id
        ):
            signature_valid = False
            errors.append("signature verifier identity does not match manifest")
        else:
            try:
                signature = base64.b64decode(block.value_base64, validate=True)
                signature_valid = bool(
                    verifier.verify(manifest.signing_payload(), signature)
                )
            except Exception:
                signature_valid = False
            if not signature_valid:
                errors.append("manifest signature is invalid")

    passed = checksum_valid and not errors
    return ManifestVerification(
        passed=passed,
        checksum_valid=checksum_valid,
        signature_required=require_signature,
        signature_valid=signature_valid,
        errors=tuple(errors),
    )


verify_model_manifest = verify_manifest


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
