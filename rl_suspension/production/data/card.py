"""Machine-readable JSON data cards for direct-12D datasets."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from rl_suspension.production.data.collection import EpisodePhase, PHASE_NAMES
from rl_suspension.production.data.dataset import load_dataset
from rl_suspension.production.data.diagnostics import (
    ActionAmbiguityReport,
    diagnose_action_ambiguity,
)
from rl_suspension.production.data.normalization import GroupedNormalization
from rl_suspension.production.data.validation import (
    DatasetValidationReport,
    validate_dataset,
)
from rl_suspension.production.provenance import canonical_sha256


def build_data_card(
    root: str | Path,
    *,
    validation: DatasetValidationReport | None = None,
    normalization: GroupedNormalization | None = None,
    ambiguity: ActionAmbiguityReport | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root_path = Path(root)
    manifest = json.loads((root_path / "manifest.json").read_text(encoding="utf-8"))
    dataset = load_dataset(root_path)
    validation_report = validation or validate_dataset(root_path)
    ambiguity_report = ambiguity or diagnose_action_ambiguity(dataset)

    phases = np.asarray(dataset.phases, dtype=np.int64)
    phase_counts = {
        PHASE_NAMES[phase]: int(np.count_nonzero(phases == int(phase)))
        for phase in EpisodePhase
    }
    split_counts: dict[str, dict[str, int]] = {}
    for record in dataset.records:
        summary = split_counts.setdefault(record.split, {"episodes": 0, "transitions": 0})
        summary["episodes"] += 1
        summary["transitions"] += record.transitions
    card: dict[str, Any] = {
        "version": "direct12.data-card.v1",
        "dataset_version": manifest.get("version"),
        "dataset_content_hash": manifest.get("content_hash"),
        "action_schema_version": manifest.get("action_schema_version"),
        "observation_schema_version": manifest.get("observation_schema_version"),
        "episodes": len(dataset.records),
        "scenarios": len(set(np.asarray(dataset.scenario_ids, dtype=str))),
        "transitions": len(dataset),
        "splits": split_counts,
        "phase_counts": phase_counts,
        "expert_valid": int(np.count_nonzero(dataset.expert_valid)),
        "expert_invalid": int(np.count_nonzero(~dataset.expert_valid)),
        "solver_fallbacks": int(np.count_nonzero(dataset.solver_fallback)),
        "solver_timeouts": int(np.count_nonzero(dataset.solver_timeout)),
        "validation": asdict(validation_report),
        "action_ambiguity": asdict(ambiguity_report),
        "normalization": (
            asdict(normalization) if normalization is not None else None
        ),
    }
    card["card_hash"] = canonical_sha256(card)
    if output_path is not None:
        _atomic_json(Path(output_path), card)
    return card


def write_data_card(
    root: str | Path,
    path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return build_data_card(root, output_path=path, **kwargs)


generate_data_card = build_data_card


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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
