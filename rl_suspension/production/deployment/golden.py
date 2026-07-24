"""Golden-vector generation and framework/export equivalence checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from rl_suspension.production.deployment.export import OptionalOnnxDependencyError


@dataclass(frozen=True)
class OutputVerification:
    passed: bool
    maximum_absolute_error: float
    maximum_relative_error: float
    absolute_tolerance: float
    relative_tolerance: float


@dataclass(frozen=True)
class GoldenVerificationReport:
    passed: bool
    samples: int
    outputs: Mapping[str, OutputVerification]

    def __bool__(self) -> bool:
        return self.passed

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "samples": self.samples,
            "outputs": {
                name: asdict(result) for name, result in self.outputs.items()
            },
        }


@dataclass(frozen=True)
class GoldenVectorSet:
    inputs: Mapping[str, np.ndarray]
    outputs: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]

    @property
    def samples(self) -> int:
        values = tuple(self.inputs.values())
        return int(values[0].shape[0]) if values else 0

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            **{f"input__{name}": value for name, value in self.inputs.items()},
            **{f"output__{name}": value for name, value in self.outputs.items()},
            "__metadata_json": np.frombuffer(
                json.dumps(
                    dict(self.metadata),
                    sort_keys=True,
                    allow_nan=False,
                ).encode("utf-8"),
                dtype=np.uint8,
            ),
        }
        with target.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "GoldenVectorSet":
        with np.load(Path(path), allow_pickle=False) as archive:
            inputs = {
                key.removeprefix("input__"): np.asarray(archive[key])
                for key in archive.files
                if key.startswith("input__")
            }
            outputs = {
                key.removeprefix("output__"): np.asarray(archive[key])
                for key in archive.files
                if key.startswith("output__")
            }
            metadata_bytes = np.asarray(
                archive["__metadata_json"], dtype=np.uint8
            ).tobytes()
        return cls(
            inputs=inputs,
            outputs=outputs,
            metadata=json.loads(metadata_bytes.decode("utf-8")),
        )


def generate_golden_vectors(
    predictor: Callable[..., Any],
    inputs: Mapping[str, np.ndarray],
    *,
    output_names: Sequence[str] = ("action",),
    metadata: Mapping[str, Any] | None = None,
) -> GoldenVectorSet:
    """Record deterministic framework outputs one fixed-shape case at a time."""

    normalized = _validate_inputs(inputs)
    sample_count = next(iter(normalized.values())).shape[0]
    collected: dict[str, list[np.ndarray]] = {name: [] for name in output_names}
    for index in range(sample_count):
        case = {
            name: value[index : index + 1] for name, value in normalized.items()
        }
        outputs = _normalize_outputs(_invoke(predictor, case), output_names)
        for name, output in outputs.items():
            if output.shape[0] != 1 or not np.all(np.isfinite(output)):
                raise ValueError(
                    f"golden output {name!r} must be finite with batch size one"
                )
            collected[name].append(output)
    return GoldenVectorSet(
        inputs=normalized,
        outputs={
            name: np.concatenate(values, axis=0)
            for name, values in collected.items()
        },
        metadata={
            "schema_version": "golden-vectors.v1",
            **dict(metadata or {}),
        },
    )


create_golden_vectors = generate_golden_vectors


def verify_golden_vectors(
    predictor: Callable[..., Any],
    golden: GoldenVectorSet,
    *,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-4,
) -> GoldenVerificationReport:
    """Replay golden inputs and compare every output independently."""

    candidate = generate_golden_vectors(
        predictor,
        golden.inputs,
        output_names=tuple(golden.outputs),
    )
    results: dict[str, OutputVerification] = {}
    for name, expected in golden.outputs.items():
        actual = candidate.outputs[name]
        if actual.shape != expected.shape:
            results[name] = OutputVerification(
                False,
                float("inf"),
                float("inf"),
                absolute_tolerance,
                relative_tolerance,
            )
            continue
        absolute = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        relative = absolute / np.maximum(np.abs(expected), 1e-12)
        results[name] = OutputVerification(
            passed=bool(
                np.allclose(
                    actual,
                    expected,
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                )
            ),
            maximum_absolute_error=float(np.max(absolute)),
            maximum_relative_error=float(np.max(relative)),
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
    return GoldenVerificationReport(
        passed=all(result.passed for result in results.values()),
        samples=golden.samples,
        outputs=results,
    )


def verify_framework_export(
    framework_predictor: Callable[..., Any],
    exported_predictor: Callable[..., Any],
    inputs: Mapping[str, np.ndarray],
    *,
    output_names: Sequence[str] = ("action",),
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-4,
) -> GoldenVerificationReport:
    golden = generate_golden_vectors(
        framework_predictor,
        inputs,
        output_names=output_names,
    )
    return verify_golden_vectors(
        exported_predictor,
        golden,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


def verify_onnx_export(
    model_path: str | Path,
    golden: GoldenVectorSet,
    *,
    providers: Sequence[str] = ("CPUExecutionProvider",),
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-4,
) -> GoldenVerificationReport:
    if importlib.util.find_spec("onnxruntime") is None:
        raise OptionalOnnxDependencyError(
            "ONNX verification requires optional 'onnxruntime'"
        )
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=list(providers))

    def predict(**inputs):
        return dict(
            zip(
                tuple(golden.outputs),
                session.run(
                    list(golden.outputs),
                    {
                        name: np.asarray(value)
                        for name, value in inputs.items()
                    },
                ),
            )
        )

    return verify_golden_vectors(
        predict,
        golden,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


def _validate_inputs(inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not inputs:
        raise ValueError("golden inputs must not be empty")
    normalized = {
        name: np.asarray(value)
        for name, value in inputs.items()
    }
    sample_counts = {value.shape[0] for value in normalized.values() if value.ndim}
    if len(sample_counts) != 1 or any(value.ndim == 0 for value in normalized.values()):
        raise ValueError("golden inputs must share a nonempty leading sample dimension")
    if next(iter(sample_counts)) == 0:
        raise ValueError("golden inputs must contain at least one sample")
    if any(not np.all(np.isfinite(value)) for value in normalized.values()):
        raise ValueError("golden inputs contain NaN or Inf")
    return normalized


def _invoke(predictor: Callable[..., Any], inputs: Mapping[str, np.ndarray]) -> Any:
    target = getattr(predictor, "predict", predictor)
    try:
        return target(**inputs)
    except TypeError as keyword_error:
        try:
            return target(*inputs.values())
        except TypeError:
            raise keyword_error


def _normalize_outputs(
    outputs: Any,
    output_names: Sequence[str],
) -> dict[str, np.ndarray]:
    if isinstance(outputs, Mapping):
        missing = set(output_names) - set(outputs)
        if missing:
            raise ValueError(f"predictor did not return outputs: {sorted(missing)}")
        return {name: _as_numpy(outputs[name]) for name in output_names}
    if len(output_names) == 1:
        return {output_names[0]: _as_numpy(outputs)}
    if not isinstance(outputs, (tuple, list)) or len(outputs) != len(output_names):
        raise ValueError("predictor output count does not match output_names")
    return {
        name: _as_numpy(value) for name, value in zip(output_names, outputs)
    }


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
