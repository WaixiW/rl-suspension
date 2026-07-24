"""Semantically grouped normalization fitted only on valid training labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from rl_suspension.production.contracts import (
    DEFAULT_OBSERVATION_SCHEMA,
    ObservationSchema,
)
from rl_suspension.production.data.dataset import Direct12Dataset, load_dataset


@dataclass(frozen=True)
class NormalizationGroup:
    name: str
    source: str
    indices: tuple[int, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]


@dataclass(frozen=True)
class GroupedNormalization:
    groups: tuple[NormalizationGroup, ...]
    split: str
    valid_samples: int
    version: str = "direct12.grouped-normalization.v1"

    def normalize(
        self,
        state_observations: NDArray[np.floating],
        road_observations: NDArray[np.floating],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        state = np.asarray(state_observations, dtype=np.float32).copy()
        road = np.asarray(road_observations, dtype=np.float32).copy()
        for group in self.groups:
            indices = np.asarray(group.indices, dtype=np.int64)
            mean = np.asarray(group.mean, dtype=np.float32)
            std = np.asarray(group.std, dtype=np.float32)
            if group.source == "state":
                state[..., indices] = (state[..., indices] - mean) / std
            elif group.source == "road":
                reshape = (1,) * (road.ndim - 2) + (len(indices), 1)
                road[..., indices, :] = (
                    road[..., indices, :] - mean.reshape(reshape)
                ) / std.reshape(reshape)
            else:
                raise ValueError(f"unknown normalization source {group.source!r}")
        return state, road

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "GroupedNormalization":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["groups"] = tuple(
            NormalizationGroup(
                name=item["name"],
                source=item["source"],
                indices=tuple(item["indices"]),
                mean=tuple(item["mean"]),
                std=tuple(item["std"]),
            )
            for item in payload["groups"]
        )
        return cls(**payload)


def default_group_specs(
    schema: ObservationSchema = DEFAULT_OBSERVATION_SCHEMA,
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    start = 0

    def take(size: int) -> tuple[int, ...]:
        nonlocal start
        indices = tuple(range(start, start + size))
        start += size
        return indices

    specs = (
        ("vehicle_state", "state", take(schema.vehicle_state_dim)),
        ("sensor_features", "state", take(schema.sensor_feature_dim)),
        ("actuator_state", "state", take(schema.actuator_state_dim)),
        ("previous_action", "state", take(schema.action_dim)),
        ("speed", "state", take(1)),
        ("sensor_validity", "state", take(2)),
        ("road_height", "road", (0, 1)),
        ("road_validity", "road", (2, 3)),
    )
    if start != schema.state_vector_dim:
        raise RuntimeError("normalization groups do not cover the state schema")
    return specs


def compute_grouped_normalization(
    dataset_or_root: Direct12Dataset | str | Path,
    *,
    split: str = "train",
    minimum_std: float = 1e-6,
    group_specs: Sequence[tuple[str, str, tuple[int, ...]]] | None = None,
    observation_schema: ObservationSchema = DEFAULT_OBSERVATION_SCHEMA,
) -> GroupedNormalization:
    if minimum_std <= 0.0:
        raise ValueError("minimum_std must be positive")
    dataset = (
        dataset_or_root
        if isinstance(dataset_or_root, Direct12Dataset)
        else load_dataset(dataset_or_root, split=split)
    )
    split_scenario_ids = {
        record.scenario_id for record in dataset.records if record.split == split
    }
    if not split_scenario_ids:
        raise ValueError(f"dataset has no records for split={split!r}")
    split_mask = np.isin(
        np.asarray(dataset.scenario_ids, dtype=str),
        list(split_scenario_ids),
    )
    valid = np.asarray(dataset.expert_valid, dtype=np.bool_) & split_mask
    if not np.any(valid):
        raise ValueError("cannot fit normalization without valid expert labels")
    specs = tuple(group_specs or default_group_specs(observation_schema))
    _validate_group_specs(specs, observation_schema)
    groups: list[NormalizationGroup] = []
    for name, source, raw_indices in specs:
        indices = np.asarray(raw_indices, dtype=np.int64)
        if source == "state":
            values = np.asarray(dataset.state_observations[valid], dtype=np.float64)[
                :, indices
            ]
            mean = values.mean(axis=0)
            std = values.std(axis=0)
        else:
            values = np.asarray(dataset.road_observations[valid], dtype=np.float64)[
                :, indices, :
            ]
            mean = values.mean(axis=(0, 2))
            std = values.std(axis=(0, 2))
        std = np.where(std < minimum_std, 1.0, std)
        groups.append(
            NormalizationGroup(
                name=name,
                source=source,
                indices=tuple(int(index) for index in indices),
                mean=tuple(float(value) for value in mean),
                std=tuple(float(value) for value in std),
            )
        )
    return GroupedNormalization(
        groups=tuple(groups),
        split=split,
        valid_samples=int(np.count_nonzero(valid)),
    )


def _validate_group_specs(
    specs: Iterable[tuple[str, str, tuple[int, ...]]],
    schema: ObservationSchema,
) -> None:
    names: set[str] = set()
    covered: dict[str, set[int]] = {"state": set(), "road": set()}
    limits = {"state": schema.state_vector_dim, "road": schema.road_channels}
    for name, source, indices in specs:
        if name in names:
            raise ValueError(f"duplicate normalization group {name!r}")
        names.add(name)
        if source not in covered or not indices:
            raise ValueError(f"invalid normalization group {name!r}")
        for index in indices:
            if index < 0 or index >= limits[source] or index in covered[source]:
                raise ValueError(f"invalid or repeated index {index} in group {name!r}")
            covered[source].add(index)
    if covered["state"] != set(range(schema.state_vector_dim)):
        raise ValueError("state normalization groups must cover every feature exactly once")
    if covered["road"] != set(range(schema.road_channels)):
        raise ValueError("road normalization groups must cover every channel exactly once")
