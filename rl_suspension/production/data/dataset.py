"""Atomic, resumable episode shards for direct twelve-dimensional control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator, Mapping

import numpy as np
from numpy.typing import NDArray

from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    DEFAULT_OBSERVATION_SCHEMA,
    ActionSchema,
    ObservationSchema,
    Scenario,
)
from rl_suspension.production.provenance import canonical_sha256, file_sha256


DATASET_VERSION = "direct12.episode-shards.v1"

FIELD_DTYPES: dict[str, np.dtype[Any] | type[np.generic]] = {
    "state_observations": np.float32,
    "road_observations": np.float32,
    "next_state_observations": np.float32,
    "next_road_observations": np.float32,
    "expert_actions_physical": np.float32,
    "expert_actions_normalized": np.float32,
    "behavior_actions_physical": np.float32,
    "behavior_actions_normalized": np.float32,
    "rewards": np.float32,
    "terminated": np.bool_,
    "truncated": np.bool_,
    "timestamps_ns": np.int64,
    "episode_ids": np.str_,
    "scenario_ids": np.str_,
    "phases": np.uint8,
    "expert_valid": np.bool_,
    "solver_status": np.str_,
    "solver_objective": np.float64,
    "solver_iterations": np.int32,
    "solver_solve_time_ms": np.float32,
    "solver_feasibility_margin": np.float32,
    "solver_fallback": np.bool_,
    "solver_timeout": np.bool_,
    "constraint_violation": np.float32,
}

REQUIRED_FIELDS = frozenset(FIELD_DTYPES)


@dataclass(frozen=True)
class EpisodeRecord:
    file: str
    episode_id: str
    scenario_id: str
    scenario_seed: int
    split: str
    bump_family: str
    transitions: int
    sha256: str


@dataclass
class Direct12Dataset:
    arrays: dict[str, NDArray[Any]]
    records: tuple[EpisodeRecord, ...]

    def __len__(self) -> int:
        return int(self.arrays["state_observations"].shape[0])

    def __getattr__(self, name: str) -> NDArray[Any]:
        arrays = self.__dict__.get("arrays", {})
        if name in arrays:
            return arrays[name]
        raise AttributeError(name)

    def subset(self, indices: NDArray[np.integer]) -> "Direct12Dataset":
        selected = np.asarray(indices, dtype=np.int64)
        return Direct12Dataset(
            arrays={name: values[selected] for name, values in self.arrays.items()},
            records=self.records,
        )


class EpisodeShardWriter:
    """Append complete episodes atomically and resume by scenario identifier."""

    def __init__(
        self,
        root: str | Path,
        *,
        action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
        observation_schema: ObservationSchema = DEFAULT_OBSERVATION_SCHEMA,
    ) -> None:
        self.root = Path(root)
        self.shards_dir = self.root / "shards"
        self.manifest_path = self.root / "manifest.json"
        self.action_schema = action_schema
        self.observation_schema = observation_schema
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.records = self._load_records()
        self._check_unique_records()

    @property
    def completed_scenario_ids(self) -> frozenset[str]:
        return frozenset(record.scenario_id for record in self.records)

    def is_completed(self, scenario_id: str) -> bool:
        return scenario_id in self.completed_scenario_ids

    def record_for_scenario(self, scenario_id: str) -> EpisodeRecord | None:
        return next(
            (record for record in self.records if record.scenario_id == scenario_id),
            None,
        )

    def write_episode(
        self,
        *,
        transitions: Mapping[str, NDArray[Any]],
        scenario: Scenario | None = None,
        episode_id: str | int | None = None,
        scenario_id: str | None = None,
        scenario_seed: int | None = None,
        split: str | None = None,
        bump_family: str | None = None,
    ) -> EpisodeRecord:
        if scenario is not None:
            scenario_id = scenario.scenario_id
            scenario_seed = scenario.seed
            split = scenario.split
            bump_family = scenario.bump_family
        if scenario_id is None or scenario_seed is None or split is None or bump_family is None:
            raise ValueError("complete scenario metadata is required")
        existing = self.record_for_scenario(str(scenario_id))
        if existing is not None:
            if not (self.root / existing.file).exists():
                raise ValueError(f"completed shard is missing: {existing.file}")
            if file_sha256(self.root / existing.file) != existing.sha256:
                raise ValueError(f"completed shard checksum mismatch: {existing.file}")
            return existing

        resolved_episode_id = (
            str(episode_id)
            if episode_id is not None
            else f"{len(self.records):06d}"
        )
        if any(record.episode_id == resolved_episode_id for record in self.records):
            raise ValueError(f"duplicate episode_id {resolved_episode_id!r}")
        slug = _safe_slug(resolved_episode_id)
        relative = Path("shards") / f"episode_{slug}.npz"
        target = self.root / relative
        if target.exists():
            raise FileExistsError(f"append-only shard already exists: {target}")

        arrays = validate_episode_arrays(
            transitions,
            episode_id=resolved_episode_id,
            scenario_id=str(scenario_id),
            action_schema=self.action_schema,
            observation_schema=self.observation_schema,
        )
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{target.stem}.",
                suffix=".npz.tmp",
                dir=self.shards_dir,
                delete=False,
            ) as stream:
                temporary_name = stream.name
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            # Hard-link publication is atomic and refuses an existing target,
            # preserving append-only semantics under concurrent writers.
            os.link(temporary_name, target)
            record = EpisodeRecord(
                file=relative.as_posix(),
                episode_id=resolved_episode_id,
                scenario_id=str(scenario_id),
                scenario_seed=int(scenario_seed),
                split=str(split),
                bump_family=str(bump_family),
                transitions=int(arrays["state_observations"].shape[0]),
                sha256=file_sha256(target),
            )
            self.records.append(record)
            try:
                self._write_manifest()
            except Exception:
                self.records.pop()
                raise
            return record
        except Exception:
            target.unlink(missing_ok=True)
            raise
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)

    def _load_records(self) -> list[EpisodeRecord]:
        if not self.manifest_path.exists():
            return []
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        _validate_manifest_header(payload, self.action_schema, self.observation_schema)
        return [EpisodeRecord(**item) for item in payload.get("shards", [])]

    def _check_unique_records(self) -> None:
        episode_ids = [record.episode_id for record in self.records]
        scenario_ids = [record.scenario_id for record in self.records]
        files = [record.file for record in self.records]
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("manifest contains duplicate episode IDs")
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("manifest contains duplicate scenario IDs")
        if len(files) != len(set(files)):
            raise ValueError("manifest contains duplicate shard files")

    def _write_manifest(self) -> None:
        payload = {
            "version": DATASET_VERSION,
            "action_schema_version": self.action_schema.version,
            "observation_schema_version": self.observation_schema.version,
            "action_dim": self.action_schema.dimension,
            "state_observation_dim": self.observation_schema.state_vector_dim,
            "road_shape": [
                self.observation_schema.road_channels,
                self.observation_schema.road_points,
            ],
            "shards": [asdict(record) for record in self.records],
        }
        payload["content_hash"] = canonical_sha256(payload["shards"])
        _atomic_json(self.manifest_path, payload)


def iter_episode_shards(
    root: str | Path,
    *,
    split: str | None = None,
    verify_hashes: bool = True,
) -> Iterator[tuple[EpisodeRecord, dict[str, NDArray[Any]]]]:
    root_path = Path(root)
    payload = json.loads((root_path / "manifest.json").read_text(encoding="utf-8"))
    records = [EpisodeRecord(**item) for item in payload["shards"]]
    for record in records:
        if split is not None and record.split != split:
            continue
        path = root_path / record.file
        if verify_hashes and file_sha256(path) != record.sha256:
            raise ValueError(f"shard checksum mismatch: {record.file}")
        with np.load(path, allow_pickle=False) as archive:
            yield record, {name: np.asarray(archive[name]) for name in archive.files}


def load_dataset(
    root: str | Path,
    *,
    split: str | None = None,
    verify_hashes: bool = True,
) -> Direct12Dataset:
    chunks: dict[str, list[NDArray[Any]]] = {field: [] for field in REQUIRED_FIELDS}
    records: list[EpisodeRecord] = []
    for record, arrays in iter_episode_shards(
        root,
        split=split,
        verify_hashes=verify_hashes,
    ):
        missing = REQUIRED_FIELDS.difference(arrays)
        if missing:
            raise ValueError(f"shard {record.file} is missing fields: {sorted(missing)}")
        records.append(record)
        for field in chunks:
            chunks[field].append(arrays[field])
    if not records:
        raise ValueError(f"no dataset shards found for split={split!r}")
    return Direct12Dataset(
        arrays={
            field: np.concatenate(values, axis=0)
            for field, values in chunks.items()
        },
        records=tuple(records),
    )


def validate_episode_arrays(
    transitions: Mapping[str, NDArray[Any]],
    *,
    episode_id: str,
    scenario_id: str,
    action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
    observation_schema: ObservationSchema = DEFAULT_OBSERVATION_SCHEMA,
) -> dict[str, NDArray[Any]]:
    missing = REQUIRED_FIELDS.difference(transitions)
    # IDs can be supplied by writer metadata to avoid redundant caller plumbing.
    missing_without_ids = missing.difference({"episode_ids", "scenario_ids"})
    if missing_without_ids:
        raise ValueError(f"missing transition fields: {sorted(missing_without_ids)}")

    arrays = {
        field: np.asarray(transitions[field])
        for field in REQUIRED_FIELDS
        if field in transitions
    }
    count = int(arrays["state_observations"].shape[0])
    if count <= 0:
        raise ValueError("cannot write an empty episode")
    arrays.setdefault("episode_ids", np.full(count, episode_id))
    arrays.setdefault("scenario_ids", np.full(count, scenario_id))
    expected_shapes = {
        "state_observations": (count, observation_schema.state_vector_dim),
        "road_observations": (
            count,
            observation_schema.road_channels,
            observation_schema.road_points,
        ),
        "next_state_observations": (count, observation_schema.state_vector_dim),
        "next_road_observations": (
            count,
            observation_schema.road_channels,
            observation_schema.road_points,
        ),
        "expert_actions_physical": (count, action_schema.dimension),
        "expert_actions_normalized": (count, action_schema.dimension),
        "behavior_actions_physical": (count, action_schema.dimension),
        "behavior_actions_normalized": (count, action_schema.dimension),
    }
    for field, expected in expected_shapes.items():
        if arrays[field].shape != expected:
            raise ValueError(f"{field} must have shape {expected}, got {arrays[field].shape}")
    for field, array in arrays.items():
        if array.shape[0] != count:
            raise ValueError(f"{field} has inconsistent episode length")
    if not np.all(np.asarray(arrays["episode_ids"], dtype=str) == episode_id):
        raise ValueError("episode_ids do not match writer metadata")
    if not np.all(np.asarray(arrays["scenario_ids"], dtype=str) == scenario_id):
        raise ValueError("scenario_ids do not match writer metadata")

    converted: dict[str, NDArray[Any]] = {}
    for field, dtype in FIELD_DTYPES.items():
        converted[field] = np.asarray(arrays[field], dtype=dtype)
    return converted


def _validate_manifest_header(
    payload: Mapping[str, Any],
    action_schema: ActionSchema,
    observation_schema: ObservationSchema,
) -> None:
    if payload.get("version") != DATASET_VERSION:
        raise ValueError(f"unsupported dataset version {payload.get('version')!r}")
    if payload.get("action_schema_version") != action_schema.version:
        raise ValueError("dataset action schema version mismatch")
    if payload.get("observation_schema_version") != observation_schema.version:
        raise ValueError("dataset observation schema version mismatch")
    if payload.get("content_hash") != canonical_sha256(payload.get("shards", [])):
        raise ValueError("dataset manifest content hash mismatch")


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not slug:
        raise ValueError("episode_id cannot form a safe shard filename")
    return slug


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
