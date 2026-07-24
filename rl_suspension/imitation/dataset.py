"""Episode-sharded imitation dataset with scenario-safe splits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from rl_suspension.envs.observation import OBSERVATION_SPEC


class EpisodePhase(IntEnum):
    FLAT = 0
    PREVIEW = 1
    FRONT_CONTACT = 2
    REAR_CONTACT = 3
    RECOVERY = 4


PHASE_NAMES = {
    EpisodePhase.FLAT: "flat",
    EpisodePhase.PREVIEW: "preview",
    EpisodePhase.FRONT_CONTACT: "front_contact",
    EpisodePhase.REAR_CONTACT: "rear_contact",
    EpisodePhase.RECOVERY: "recovery",
}


@dataclass(frozen=True)
class ShardRecord:
    file: str
    episode_id: int
    scenario_seed: int
    split: str
    scenario_kind: str
    teacher_name: str
    transitions: int


@dataclass(frozen=True)
class NormalizationStats:
    observation_mean: list[float]
    observation_std: list[float]

    def normalize(self, observations: NDArray[np.floating]) -> NDArray[np.float32]:
        mean = np.asarray(self.observation_mean, dtype=np.float32)
        std = np.asarray(self.observation_std, dtype=np.float32)
        return ((np.asarray(observations, dtype=np.float32) - mean) / std).astype(np.float32)


@dataclass
class TransitionDataset:
    observations: NDArray[np.float32]
    actions: NDArray[np.float32]
    behavior_actions: NDArray[np.float32]
    rewards: NDArray[np.float32]
    next_observations: NDArray[np.float32]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    episode_ids: NDArray[np.int64]
    scenario_seeds: NDArray[np.int64]
    phases: NDArray[np.uint8]
    expert_valid: NDArray[np.bool_]
    expert_quality: NDArray[np.float32]
    expert_status: NDArray[np.int8]
    expert_objective: NDArray[np.float64]
    expert_iterations: NDArray[np.int32]
    expert_latency_ms: NDArray[np.float32]
    expert_constraint_margin: NDArray[np.float32]
    expert_fallback: NDArray[np.bool_]
    constraint_violation: NDArray[np.float32]

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def subset(self, indices: NDArray[np.integer]) -> "TransitionDataset":
        return TransitionDataset(
            **{
                field: np.asarray(getattr(self, field))[indices]
                for field in self.__dataclass_fields__
            }
        )


class EpisodeShardWriter:
    """Write one compressed shard per episode plus a JSON manifest."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.shards_dir = self.root / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.records: list[ShardRecord] = self._load_existing_records()

    def write_episode(
        self,
        *,
        episode_id: int,
        scenario_seed: int,
        scenario_kind: str,
        teacher_name: str,
        transitions: dict[str, NDArray],
    ) -> ShardRecord:
        validated = _validate_episode(transitions)
        split = scenario_split(scenario_seed)
        relative = Path("shards") / f"episode_{episode_id:06d}.npz"
        target = self.root / relative
        np.savez_compressed(target, **validated)

        record = ShardRecord(
            file=relative.as_posix(),
            episode_id=int(episode_id),
            scenario_seed=int(scenario_seed),
            split=split,
            scenario_kind=str(scenario_kind),
            teacher_name=str(teacher_name),
            transitions=int(validated["observations"].shape[0]),
        )
        self.records = [item for item in self.records if item.episode_id != record.episode_id]
        self.records.append(record)
        self.records.sort(key=lambda item: item.episode_id)
        self._write_manifest()
        return record

    def _load_existing_records(self) -> list[ShardRecord]:
        if not self.manifest_path.exists():
            return []
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return [ShardRecord(**item) for item in payload.get("shards", [])]

    def _write_manifest(self) -> None:
        payload = {
            "version": 2,
            "observation_dim": OBSERVATION_SPEC.dimension,
            "action_dim": 4,
            "shards": [asdict(record) for record in self.records],
        }
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.manifest_path)


def scenario_split(seed: int) -> str:
    """Stable 70/15/15 split based on complete scenario seeds."""

    bucket = int(seed) % 20
    if bucket < 14:
        return "train"
    if bucket < 17:
        return "validation"
    return "test"


def load_dataset(root: str | Path, split: str | None = None) -> TransitionDataset:
    root_path = Path(root)
    payload = json.loads((root_path / "manifest.json").read_text(encoding="utf-8"))
    records = [
        ShardRecord(**item)
        for item in payload["shards"]
        if split is None or item["split"] == split
    ]
    if not records:
        raise ValueError(f"No dataset shards found for split={split!r} in {root_path}")

    chunks: dict[str, list[NDArray]] = {
        field: [] for field in TransitionDataset.__dataclass_fields__
    }
    for record in records:
        with np.load(root_path / record.file, allow_pickle=False) as shard:
            for field in chunks:
                chunks[field].append(np.asarray(shard[field]))
    return TransitionDataset(
        **{field: np.concatenate(values, axis=0) for field, values in chunks.items()}
    )


def compute_normalization(dataset: TransitionDataset) -> NormalizationStats:
    valid = dataset.expert_valid
    observations = dataset.observations[valid]
    if observations.size == 0:
        raise ValueError("Cannot compute normalization without valid expert samples")
    mean = observations.mean(axis=0, dtype=np.float64)
    std = observations.std(axis=0, dtype=np.float64)
    # Constant training features should remain on their physical scale rather
    # than being amplified by division through an arbitrarily tiny epsilon.
    std = np.where(std < 1e-6, 1.0, std)
    return NormalizationStats(
        observation_mean=mean.astype(float).tolist(),
        observation_std=std.astype(float).tolist(),
    )


def save_normalization(stats: NormalizationStats, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")


def load_normalization(path: str | Path) -> NormalizationStats:
    return NormalizationStats(**json.loads(Path(path).read_text(encoding="utf-8")))


class PhaseBalancedSampler:
    """Sample event phases without allowing flat-road labels to dominate."""

    def __init__(
        self,
        dataset: TransitionDataset,
        seed: int = 0,
        flat_fraction: float = 0.15,
    ) -> None:
        if not 0.0 <= flat_fraction < 1.0:
            raise ValueError("flat_fraction must be in [0, 1)")
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)
        self.flat_fraction = float(flat_fraction)
        self.indices = {
            phase: np.flatnonzero(
                (dataset.phases == int(phase)) & dataset.expert_valid
            )
            for phase in EpisodePhase
        }
        all_valid = np.flatnonzero(dataset.expert_valid)
        if all_valid.size == 0:
            raise ValueError("Dataset has no valid expert samples")
        self.all_valid = all_valid

    def sample_indices(self, batch_size: int) -> NDArray[np.int64]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        flat_count = int(round(batch_size * self.flat_fraction))
        remaining = batch_size - flat_count
        event_phases = [
            phase
            for phase in EpisodePhase
            if phase is not EpisodePhase.FLAT and self.indices[phase].size > 0
        ]
        selected: list[NDArray[np.int64]] = []
        if flat_count and self.indices[EpisodePhase.FLAT].size:
            selected.append(
                self.rng.choice(
                    self.indices[EpisodePhase.FLAT],
                    size=flat_count,
                    replace=True,
                )
            )
        else:
            remaining += flat_count

        if event_phases:
            counts = _divide_evenly(remaining, len(event_phases))
            for phase, count in zip(event_phases, counts):
                selected.append(
                    self.rng.choice(self.indices[phase], size=count, replace=True)
                )
        elif remaining:
            selected.append(self.rng.choice(self.all_valid, size=remaining, replace=True))

        indices = np.concatenate(selected).astype(np.int64)
        self.rng.shuffle(indices)
        return indices


def concatenate_datasets(datasets: Iterable[TransitionDataset]) -> TransitionDataset:
    items = list(datasets)
    if not items:
        raise ValueError("At least one dataset is required")
    return TransitionDataset(
        **{
            field: np.concatenate([getattr(item, field) for item in items], axis=0)
            for field in TransitionDataset.__dataclass_fields__
        }
    )


def _divide_evenly(total: int, groups: int) -> list[int]:
    base, extra = divmod(total, groups)
    return [base + (index < extra) for index in range(groups)]


def _validate_episode(transitions: dict[str, NDArray]) -> dict[str, NDArray]:
    required = set(TransitionDataset.__dataclass_fields__)
    missing = required.difference(transitions)
    if missing:
        raise ValueError(f"Missing transition fields: {sorted(missing)}")

    arrays = {field: np.asarray(transitions[field]) for field in required}
    count = arrays["observations"].shape[0]
    if count == 0:
        raise ValueError("Cannot write an empty episode")
    if arrays["observations"].shape != (count, OBSERVATION_SPEC.dimension):
        raise ValueError("Invalid observation array shape")
    if arrays["next_observations"].shape != (count, OBSERVATION_SPEC.dimension):
        raise ValueError("Invalid next-observation array shape")
    if arrays["actions"].shape != (count, 4):
        raise ValueError("Invalid action array shape")
    if arrays["behavior_actions"].shape != (count, 4):
        raise ValueError("Invalid behavior-action array shape")
    for field, array in arrays.items():
        if array.shape[0] != count:
            raise ValueError(f"Field {field!r} has inconsistent episode length")

    dtypes = {
        "observations": np.float32,
        "actions": np.float32,
        "behavior_actions": np.float32,
        "rewards": np.float32,
        "next_observations": np.float32,
        "terminated": np.bool_,
        "truncated": np.bool_,
        "episode_ids": np.int64,
        "scenario_seeds": np.int64,
        "phases": np.uint8,
        "expert_valid": np.bool_,
        "expert_quality": np.float32,
        "expert_status": np.int8,
        "expert_objective": np.float64,
        "expert_iterations": np.int32,
        "expert_latency_ms": np.float32,
        "expert_constraint_margin": np.float32,
        "expert_fallback": np.bool_,
        "constraint_violation": np.float32,
    }
    return {field: arrays[field].astype(dtype) for field, dtype in dtypes.items()}
