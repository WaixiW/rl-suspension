"""Integrity and label validation for direct-12D episode datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    DEFAULT_OBSERVATION_SCHEMA,
    ActionSchema,
    ObservationSchema,
)
from rl_suspension.production.data.collection import EpisodePhase
from rl_suspension.production.data.dataset import (
    EpisodeRecord,
    REQUIRED_FIELDS,
)
from rl_suspension.production.provenance import canonical_sha256, file_sha256


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    count: int
    locations: tuple[str, ...] = ()
    severity: str = "error"


@dataclass(frozen=True)
class DatasetValidationReport:
    passed: bool
    issues: tuple[ValidationIssue, ...]
    shards: int
    transitions: int
    splits: dict[str, int]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        temporary.replace(target)

    def raise_for_errors(self) -> None:
        if not self.passed:
            summary = "; ".join(f"{issue.code}: {issue.count}" for issue in self.issues)
            raise ValueError(f"dataset validation failed ({summary})")


def validate_dataset(
    root: str | Path,
    *,
    action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
    observation_schema: ObservationSchema = DEFAULT_OBSERVATION_SCHEMA,
    allowed_solver_statuses: Iterable[str] = ("optimal", "optimal_inaccurate"),
    check_expert_slew: bool = True,
    raise_on_error: bool = False,
) -> DatasetValidationReport:
    root_path = Path(root)
    manifest_path = root_path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues: list[ValidationIssue] = []
    raw_records = payload.get("shards", [])
    records: list[EpisodeRecord] = []
    try:
        records = [EpisodeRecord(**item) for item in raw_records]
    except (KeyError, TypeError) as error:
        issues.append(ValidationIssue("manifest_record", str(error), 1))

    if payload.get("content_hash") != canonical_sha256(raw_records):
        issues.append(ValidationIssue("manifest_hash", "manifest content hash mismatch", 1))
    _duplicate_record_issues(records, issues)

    split_counts: dict[str, int] = {}
    transitions = 0
    seen_samples: dict[tuple[str, int], str] = {}
    allowed_statuses = set(allowed_solver_statuses)
    for record in records:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
        path = root_path / record.file
        location = record.file
        if not path.exists():
            issues.append(ValidationIssue("missing_shard", "shard file is missing", 1, (location,)))
            continue
        if file_sha256(path) != record.sha256:
            issues.append(
                ValidationIssue("checksum", "shard checksum mismatch", 1, (location,))
            )
        try:
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
        except Exception as error:
            issues.append(ValidationIssue("unreadable_shard", str(error), 1, (location,)))
            continue
        missing = REQUIRED_FIELDS.difference(arrays)
        if missing:
            issues.append(
                ValidationIssue(
                    "missing_fields",
                    f"missing fields: {sorted(missing)}",
                    len(missing),
                    (location,),
                )
            )
            continue
        if arrays["state_observations"].ndim == 0:
            issues.append(
                ValidationIssue(
                    "shape",
                    "state_observations must have a transition dimension",
                    1,
                    (location,),
                )
            )
            continue
        count = arrays["state_observations"].shape[0]
        transitions += int(count)
        if count != record.transitions:
            issues.append(
                ValidationIssue(
                    "transition_count",
                    "recorded transition count does not match shard",
                    1,
                    (location,),
                )
            )
        invalid_shape = _shape_issues(
            arrays,
            count,
            location,
            action_schema,
            observation_schema,
            issues,
        )
        _finite_issues(arrays, location, issues)
        if invalid_shape:
            continue
        _action_issues(
            arrays,
            location,
            action_schema,
            observation_schema,
            check_expert_slew,
            issues,
        )
        _label_issues(arrays, location, allowed_statuses, issues)

        scenario_ids = np.asarray(arrays["scenario_ids"], dtype=str)
        timestamps = np.asarray(arrays["timestamps_ns"], dtype=np.int64)
        if np.any(scenario_ids != record.scenario_id):
            mismatch = int(np.count_nonzero(scenario_ids != record.scenario_id))
            issues.append(
                ValidationIssue(
                    "scenario_label",
                    "row scenario ID does not match shard record",
                    mismatch,
                    (location,),
                )
            )
        duplicates: list[str] = []
        for row, (scenario_id, timestamp) in enumerate(zip(scenario_ids, timestamps)):
            key = (str(scenario_id), int(timestamp))
            current = f"{location}:{row}"
            if key in seen_samples:
                duplicates.extend((seen_samples[key], current))
            else:
                seen_samples[key] = current
        if duplicates:
            issues.append(
                ValidationIssue(
                    "duplicate_transition",
                    "duplicate (scenario_id, timestamp_ns) labels",
                    len(duplicates) // 2,
                    tuple(duplicates[:10]),
                )
            )

    report = DatasetValidationReport(
        passed=not any(issue.severity == "error" for issue in issues),
        issues=tuple(issues),
        shards=len(records),
        transitions=transitions,
        splits=split_counts,
    )
    if raise_on_error:
        report.raise_for_errors()
    return report


def _duplicate_record_issues(
    records: list[EpisodeRecord],
    issues: list[ValidationIssue],
) -> None:
    for attribute, code in (
        ("episode_id", "duplicate_episode"),
        ("scenario_id", "duplicate_scenario"),
        ("file", "duplicate_shard"),
        ("sha256", "duplicate_shard_content"),
    ):
        values = [getattr(record, attribute) for record in records]
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            issues.append(
                ValidationIssue(
                    code,
                    f"duplicate {attribute} values",
                    len(duplicates),
                    tuple(str(item) for item in duplicates[:10]),
                )
            )
    scenario_splits: dict[str, set[str]] = {}
    seed_splits: dict[int, set[str]] = {}
    for record in records:
        scenario_splits.setdefault(record.scenario_id, set()).add(record.split)
        seed_splits.setdefault(record.scenario_seed, set()).add(record.split)
    leaked = [key for key, splits in scenario_splits.items() if len(splits) > 1]
    leaked_seeds = [key for key, splits in seed_splits.items() if len(splits) > 1]
    if leaked or leaked_seeds:
        issues.append(
            ValidationIssue(
                "split_leakage",
                "scenario IDs or seeds occur in more than one split",
                len(leaked) + len(leaked_seeds),
                tuple(
                    [str(value) for value in leaked[:5]]
                    + [f"seed:{value}" for value in leaked_seeds[:5]]
                ),
            )
        )


def _shape_issues(
    arrays: Mapping[str, np.ndarray],
    count: int,
    location: str,
    action_schema: ActionSchema,
    observation_schema: ObservationSchema,
    issues: list[ValidationIssue],
) -> bool:
    expected = {
        "state_observations": (count, observation_schema.state_vector_dim),
        "next_state_observations": (count, observation_schema.state_vector_dim),
        "road_observations": (
            count,
            observation_schema.road_channels,
            observation_schema.road_points,
        ),
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
    invalid = [
        name
        for name, shape in expected.items()
        if np.asarray(arrays[name]).shape != shape
    ]
    invalid.extend(
        name
        for name, array in arrays.items()
        if name not in expected and np.asarray(array).shape[:1] != (count,)
    )
    if invalid:
        issues.append(
            ValidationIssue(
                "shape",
                f"inconsistent shapes: {sorted(set(invalid))}",
                len(set(invalid)),
                (location,),
            )
        )
    return bool(invalid)


def _finite_issues(
    arrays: Mapping[str, np.ndarray],
    location: str,
    issues: list[ValidationIssue],
) -> None:
    invalid = 0
    fields: list[str] = []
    for name, array in arrays.items():
        if np.issubdtype(array.dtype, np.number):
            count = int(np.count_nonzero(~np.isfinite(array)))
            if count:
                invalid += count
                fields.append(name)
    if invalid:
        issues.append(
            ValidationIssue(
                "nonfinite",
                f"NaN or Inf values in {sorted(fields)}",
                invalid,
                (location,),
            )
        )


def _action_issues(
    arrays: Mapping[str, np.ndarray],
    location: str,
    schema: ActionSchema,
    observation_schema: ObservationSchema,
    check_expert_slew: bool,
    issues: list[ValidationIssue],
) -> None:
    low = np.asarray(schema.minimum)
    high = np.asarray(schema.maximum)
    for prefix in ("expert", "behavior"):
        physical = np.asarray(arrays[f"{prefix}_actions_physical"], dtype=np.float64)
        normalized = np.asarray(arrays[f"{prefix}_actions_normalized"], dtype=np.float64)
        bounds = int(
            np.count_nonzero(
                np.any((physical < low - 1e-7) | (physical > high + 1e-7), axis=1)
            )
        )
        normalized_bounds = int(
            np.count_nonzero(
                np.any((normalized < -1e-7) | (normalized > 1.0 + 1e-7), axis=1)
            )
        )
        expected = (physical - low) / np.maximum(high - low, 1e-12)
        inconsistent = int(
            np.count_nonzero(
                np.any(~np.isclose(normalized, expected, atol=2e-5, rtol=2e-5), axis=1)
            )
        )
        if bounds or normalized_bounds:
            issues.append(
                ValidationIssue(
                    "action_bounds",
                    f"{prefix} actions violate physical or normalized bounds",
                    bounds + normalized_bounds,
                    (location,),
                )
            )
        if inconsistent:
            issues.append(
                ValidationIssue(
                    "action_normalization",
                    f"{prefix} physical and normalized actions disagree",
                    inconsistent,
                    (location,),
                )
            )
        if prefix == "expert" and not check_expert_slew:
            continue
        if physical.shape[0] > 0:
            maximum_delta = (
                np.asarray(schema.slew_per_second)
                * observation_schema.control_period_s
            )
            previous = np.vstack(
                (
                    np.asarray(schema.safe_action, dtype=np.float64),
                    physical[:-1],
                )
            )
            violations = int(
                np.count_nonzero(
                    np.any(
                        np.abs(physical - previous) > maximum_delta + 1e-7,
                        axis=1,
                    )
                )
            )
            if violations:
                issues.append(
                    ValidationIssue(
                        "action_slew",
                        f"{prefix} actions violate per-step slew limits",
                        violations,
                        (location,),
                    )
                )


def _label_issues(
    arrays: Mapping[str, np.ndarray],
    location: str,
    allowed_statuses: set[str],
    issues: list[ValidationIssue],
) -> None:
    phases = np.asarray(arrays["phases"], dtype=np.int64)
    allowed_phases = np.asarray([int(phase) for phase in EpisodePhase])
    invalid_phases = int(np.count_nonzero(~np.isin(phases, allowed_phases)))
    expert_valid = np.asarray(arrays["expert_valid"], dtype=np.bool_)
    fallback = np.asarray(arrays["solver_fallback"], dtype=np.bool_)
    timeout = np.asarray(arrays["solver_timeout"], dtype=np.bool_)
    statuses = np.asarray(arrays["solver_status"], dtype=str)
    invalid_labels = int(
        np.count_nonzero(
            (~expert_valid)
            | fallback
            | timeout
            | (~np.isin(statuses, list(allowed_statuses)))
        )
    )
    if invalid_phases:
        issues.append(
            ValidationIssue(
                "invalid_phase",
                "unknown episode phase labels",
                invalid_phases,
                (location,),
            )
        )
    if invalid_labels:
        issues.append(
            ValidationIssue(
                "invalid_expert_label",
                "invalid, fallback, timeout, or disallowed solver labels",
                invalid_labels,
                (location,),
            )
        )
