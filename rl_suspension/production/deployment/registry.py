"""Checksum-enforced local model registry with explicit rollback history."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

from rl_suspension.production.deployment.manifest import (
    ManifestVerifier,
    ModelManifest,
    load_manifest,
    sha256_file,
    verify_manifest,
    write_manifest,
)


class ModelChecksumError(RuntimeError):
    pass


@dataclass(frozen=True)
class RegisteredModel:
    version: str
    artifact_path: str
    sha256: str
    registered_utc: str
    manifest_path: str | None = None


class ModelRegistry:
    """Store immutable model versions and atomically select an active version."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.models_root = self.root / "models"
        self.state_path = self.root / "registry.json"
        self.models_root.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self._write_state({"schema_version": 1, "active": None, "history": [], "models": {}})

    @property
    def active_version(self) -> str | None:
        return self._read_state()["active"]

    def versions(self) -> tuple[str, ...]:
        return tuple(sorted(self._read_state()["models"]))

    def register(
        self,
        version: str,
        artifact_path: str | Path,
        *,
        manifest: ModelManifest | str | Path | None = None,
        verifier: ManifestVerifier | None = None,
        require_signature: bool = False,
        activate: bool = False,
    ) -> RegisteredModel:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
            raise ValueError("version may contain only letters, digits, dot, underscore, hyphen")
        source = Path(artifact_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        loaded_manifest: ModelManifest | None = None
        source_manifest_path: Path | None = None
        if manifest is not None:
            if isinstance(manifest, ModelManifest):
                loaded_manifest = manifest
            else:
                source_manifest_path = Path(manifest)
                loaded_manifest = load_manifest(source_manifest_path)
            verification = verify_manifest(
                loaded_manifest,
                source,
                verifier=verifier,
                require_signature=require_signature,
            )
            if not verification:
                raise ModelChecksumError("; ".join(verification.errors))
        elif require_signature:
            raise ValueError("a signed manifest is required")

        state = self._read_state()
        if version in state["models"]:
            raise ValueError(f"model version {version!r} is already registered")
        version_root = self.models_root / version
        version_root.mkdir(parents=False, exist_ok=False)
        target = version_root / source.name
        shutil.copy2(source, target)
        digest = sha256_file(target)
        if digest != sha256_file(source):
            shutil.rmtree(version_root)
            raise ModelChecksumError("copied model checksum mismatch")

        target_manifest: Path | None = None
        if source_manifest_path is not None:
            target_manifest = version_root / source_manifest_path.name
            shutil.copy2(source_manifest_path, target_manifest)
        elif loaded_manifest is not None:
            target_manifest = version_root / f"{source.name}.manifest.json"
            write_manifest(loaded_manifest, target_manifest)

        record = RegisteredModel(
            version=version,
            artifact_path=str(target.relative_to(self.root)),
            sha256=digest,
            registered_utc=datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            manifest_path=(
                str(target_manifest.relative_to(self.root))
                if target_manifest is not None
                else None
            ),
        )
        state["models"][version] = asdict(record)
        self._write_state(state)
        if activate:
            self.activate(version)
        return record

    def activate(self, version: str) -> RegisteredModel:
        state = self._read_state()
        record = self._record(state, version)
        self._verify_record(record)
        previous = state["active"]
        if previous is not None and previous != version:
            state["history"].append(previous)
        state["active"] = version
        self._write_state(state)
        return record

    def rollback(self) -> RegisteredModel:
        state = self._read_state()
        if not state["history"]:
            raise RuntimeError("no rollback version is available")
        version = state["history"].pop()
        record = self._record(state, version)
        self._verify_record(record)
        state["active"] = version
        self._write_state(state)
        return record

    def resolve(self, version: str | None = None) -> Path:
        state = self._read_state()
        selected = version or state["active"]
        if selected is None:
            raise RuntimeError("no active model is configured")
        record = self._record(state, selected)
        self._verify_record(record)
        return self.root / record.artifact_path

    def _verify_record(self, record: RegisteredModel) -> None:
        path = self.root / record.artifact_path
        if not path.is_file() or sha256_file(path) != record.sha256:
            raise ModelChecksumError(
                f"registered model {record.version!r} failed checksum verification"
            )

    @staticmethod
    def _record(state: dict, version: str) -> RegisteredModel:
        try:
            return RegisteredModel(**state["models"][version])
        except KeyError as error:
            raise KeyError(f"unknown model version {version!r}") from error

    def _read_state(self) -> dict:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write_state(self, state: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)
