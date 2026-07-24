"""Serializable production-pipeline configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromotionGates:
    minimum_teacher_return_gain: float = 0.01
    minimum_teacher_comfort_gain: float = 0.01
    maximum_solver_invalid_rate: float = 0.001
    minimum_student_mpc_retention: float = 0.80
    maximum_student_safety_regression: float = 0.0
    maximum_p99_inference_ms: float = 2.0


@dataclass(frozen=True)
class CollectionConfig:
    train_episodes: int = 500
    validation_episodes: int = 100
    test_episodes: int = 100
    base_seed: int = 0
    workers: int = 1


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 3e-4
    action_delta_weight: float = 0.1
    bound_weight: float = 0.01
    dagger_betas: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)
    dagger_episodes_per_round: int = 20
    seed: int = 0


@dataclass(frozen=True)
class PipelineConfig:
    output_root: str = "runs/production"
    mpc_plugin: str = ""
    simulator_plugin: str = ""
    safe_controller_plugin: str = ""
    policy_plugin: str = ""
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    gates: PromotionGates = field(default_factory=PromotionGates)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        training_payload = dict(payload.get("training", {}))
        if "dagger_betas" in training_payload:
            training_payload["dagger_betas"] = tuple(training_payload["dagger_betas"])
        return cls(
            output_root=str(payload.get("output_root", "runs/production")),
            mpc_plugin=str(payload.get("mpc_plugin", "")),
            simulator_plugin=str(payload.get("simulator_plugin", "")),
            safe_controller_plugin=str(payload.get("safe_controller_plugin", "")),
            policy_plugin=str(payload.get("policy_plugin", "")),
            collection=CollectionConfig(**payload.get("collection", {})),
            training=TrainingConfig(**training_payload),
            gates=PromotionGates(**payload.get("gates", {})),
            metadata=dict(payload.get("metadata", {})),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
