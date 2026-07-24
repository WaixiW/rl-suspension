"""Canonical run provenance and append-only artifact directories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


def _json_value(value: Any) -> Any:
    """Convert supported values to a deterministic, strict-JSON representation."""

    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("canonical JSON does not permit NaN or Inf")
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return UTF-8 JSON with stable key ordering and no insignificant whitespace."""

    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RunManifest:
    """Immutable run identity whose embedded payloads are canonical JSON strings.

    Returning decoded payloads through properties prevents callers from mutating
    the data from which the recorded hashes were calculated.
    """

    run_id: str
    created_at_utc: str
    configuration_json: str
    contracts_json: str
    scenarios_json: str
    configuration_hash: str
    contracts_hash: str
    scenarios_hash: str
    code_revision: str = ""
    metadata_json: str = "{}"
    version: str = "production.run-manifest.v1"

    @classmethod
    def create(
        cls,
        *,
        configuration: Mapping[str, Any],
        contracts: Mapping[str, Any],
        scenarios: Sequence[Any],
        run_id: str | None = None,
        code_revision: str = "",
        metadata: Mapping[str, Any] | None = None,
        created_at_utc: str | None = None,
    ) -> "RunManifest":
        configuration_json = canonical_json_bytes(configuration).decode("utf-8")
        contracts_json = canonical_json_bytes(contracts).decode("utf-8")
        scenarios_json = canonical_json_bytes(scenarios).decode("utf-8")
        metadata_json = canonical_json_bytes(metadata or {}).decode("utf-8")
        identity = {
            "configuration_hash": canonical_sha256(configuration),
            "contracts_hash": canonical_sha256(contracts),
            "scenarios_hash": canonical_sha256(scenarios),
            "code_revision": str(code_revision),
        }
        resolved_id = run_id or f"run-{canonical_sha256(identity)[:16]}"
        if not resolved_id or any(character in resolved_id for character in "/\\"):
            raise ValueError("run_id must be a nonempty path component")
        timestamp = created_at_utc or datetime.now(timezone.utc).isoformat()
        manifest = cls(
            run_id=resolved_id,
            created_at_utc=timestamp,
            configuration_json=configuration_json,
            contracts_json=contracts_json,
            scenarios_json=scenarios_json,
            configuration_hash=identity["configuration_hash"],
            contracts_hash=identity["contracts_hash"],
            scenarios_hash=identity["scenarios_hash"],
            code_revision=str(code_revision),
            metadata_json=metadata_json,
        )
        manifest.verify()
        return manifest

    @property
    def configuration(self) -> dict[str, Any]:
        return json.loads(self.configuration_json)

    @property
    def contracts(self) -> dict[str, Any]:
        return json.loads(self.contracts_json)

    @property
    def scenarios(self) -> list[Any]:
        return json.loads(self.scenarios_json)

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_json)

    @property
    def manifest_hash(self) -> str:
        return canonical_sha256(self.to_dict(include_manifest_hash=False))

    def verify(self) -> None:
        expected = (
            ("configuration", self.configuration, self.configuration_hash),
            ("contracts", self.contracts, self.contracts_hash),
            ("scenarios", self.scenarios, self.scenarios_hash),
        )
        for name, payload, recorded in expected:
            actual = canonical_sha256(payload)
            if actual != recorded:
                raise ValueError(f"{name} hash mismatch: {actual} != {recorded}")

    def to_dict(self, *, include_manifest_hash: bool = True) -> dict[str, Any]:
        payload = {
            "version": self.version,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "configuration": self.configuration,
            "contracts": self.contracts,
            "scenarios": self.scenarios,
            "configuration_hash": self.configuration_hash,
            "contracts_hash": self.contracts_hash,
            "scenarios_hash": self.scenarios_hash,
            "code_revision": self.code_revision,
            "metadata": self.metadata,
        }
        if include_manifest_hash:
            payload["manifest_hash"] = canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunManifest":
        raw = dict(payload)
        recorded_manifest_hash = raw.pop("manifest_hash", None)
        if recorded_manifest_hash is not None:
            actual = canonical_sha256(raw)
            if actual != recorded_manifest_hash:
                raise ValueError("run manifest hash mismatch")
        manifest = cls(
            version=str(raw["version"]),
            run_id=str(raw["run_id"]),
            created_at_utc=str(raw["created_at_utc"]),
            configuration_json=canonical_json_bytes(raw["configuration"]).decode("utf-8"),
            contracts_json=canonical_json_bytes(raw["contracts"]).decode("utf-8"),
            scenarios_json=canonical_json_bytes(raw["scenarios"]).decode("utf-8"),
            configuration_hash=str(raw["configuration_hash"]),
            contracts_hash=str(raw["contracts_hash"]),
            scenarios_hash=str(raw["scenarios_hash"]),
            code_revision=str(raw.get("code_revision", "")),
            metadata_json=canonical_json_bytes(raw.get("metadata", {})).decode("utf-8"),
        )
        manifest.verify()
        return manifest


def create_run_manifest(**kwargs: Any) -> RunManifest:
    """Functional alias for callers that do not need the class constructor."""

    return RunManifest.create(**kwargs)


class AppendOnlyRunDirectory:
    """Run directory API that creates files atomically and never overwrites them."""

    MANIFEST_NAME = "manifest.json"

    def __init__(self, path: Path, manifest: RunManifest) -> None:
        self.path = path
        self.manifest = manifest

    @classmethod
    def create(
        cls,
        root: str | Path,
        manifest: RunManifest,
    ) -> "AppendOnlyRunDirectory":
        manifest.verify()
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        run_path = root_path / manifest.run_id
        try:
            run_path.mkdir()
        except FileExistsError as error:
            raise FileExistsError(f"run directory already exists: {run_path}") from error
        instance = cls(run_path, manifest)
        try:
            instance.append_json(cls.MANIFEST_NAME, manifest.to_dict())
        except Exception:
            run_path.rmdir()
            raise
        return instance

    @classmethod
    def open(cls, path: str | Path) -> "AppendOnlyRunDirectory":
        run_path = Path(path)
        payload = json.loads((run_path / cls.MANIFEST_NAME).read_text(encoding="utf-8"))
        manifest = RunManifest.from_dict(payload)
        if run_path.name != manifest.run_id:
            raise ValueError("run directory name does not match manifest run_id")
        return cls(run_path, manifest)

    def append_json(self, relative_path: str | Path, payload: Any) -> str:
        return self.append_bytes(relative_path, canonical_json_bytes(payload))

    def append_bytes(self, relative_path: str | Path, content: bytes) -> str:
        target = self._target(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"append-only artifact already exists: {target}")

        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as stream:
                temporary_name = stream.name
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_name, target)
            except FileExistsError:
                raise
            except OSError:
                descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                with os.fdopen(descriptor, "wb") as output:
                    output.write(content)
                    output.flush()
                    os.fsync(output.fileno())
            return hashlib.sha256(content).hexdigest()
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)

    def verify_artifact(self, relative_path: str | Path, expected_hash: str) -> bool:
        return file_sha256(self._target(relative_path)) == expected_hash

    def _target(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("artifact path must stay within the run directory")
        target = self.path.joinpath(*relative.parts)
        if target == self.path:
            raise ValueError("artifact path must identify a file")
        return target
