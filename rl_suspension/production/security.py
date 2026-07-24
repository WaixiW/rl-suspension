"""Local integrity, redaction, and append-only audit primitives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SENSITIVE_KEYS = {
    "password",
    "secret",
    "token",
    "api_key",
    "license_key",
    "private_key",
}

ROLE_PERMISSIONS = {
    "collector": frozenset({"read_config", "write_dataset", "write_audit"}),
    "trainer": frozenset({"read_dataset", "write_checkpoint", "write_audit"}),
    "evaluator": frozenset({"read_dataset", "read_checkpoint", "write_report"}),
    "reviewer": frozenset({"read_report", "promote_model", "rollback_model"}),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).lower() in SENSITIVE_KEYS
                else redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    return value


@dataclass
class ChainedAuditLog:
    """Append JSON records whose hashes commit to the previous record."""

    path: Path

    def append(self, event: str, payload: dict[str, Any], actor: str) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        previous_hash = self._last_hash()
        record = {
            "event": event,
            "actor": actor,
            "payload": redact(payload),
            "previous_hash": previous_hash,
        }
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
        record_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        record["record_hash"] = record_hash
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record_hash

    def verify(self) -> bool:
        previous_hash = ""
        if not self.path.exists():
            return True
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                record_hash = record.pop("record_hash")
                if record.get("previous_hash") != previous_hash:
                    return False
                canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
                if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != record_hash:
                    return False
                previous_hash = record_hash
        return True

    def _last_hash(self) -> str:
        if not self.path.exists():
            return ""
        last = ""
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = str(json.loads(line)["record_hash"])
        return last


def write_checksum_manifest(
    files: Iterable[str | Path],
    output: str | Path,
) -> dict[str, str]:
    paths = [Path(path) for path in files]
    manifest = {path.as_posix(): sha256_file(path) for path in paths}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def require_permission(role: str, permission: str) -> None:
    allowed = ROLE_PERMISSIONS.get(role)
    if allowed is None:
        raise PermissionError(f"unknown production role: {role}")
    if permission not in allowed:
        raise PermissionError(f"role {role!r} lacks permission {permission!r}")
