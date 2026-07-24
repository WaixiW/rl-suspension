"""Fixed-shape ONNX export for the production observation contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    DEFAULT_OBSERVATION_SCHEMA,
    ActionSchema,
    ObservationSchema,
)
from rl_suspension.production.deployment.manifest import sha256_file


class OptionalOnnxDependencyError(RuntimeError):
    """Raised when an explicitly requested ONNX operation is unavailable."""


@dataclass(frozen=True)
class FixedShapeOnnxConfig:
    state_shape: tuple[int, int] = (
        1,
        DEFAULT_OBSERVATION_SCHEMA.state_vector_dim,
    )
    road_shape: tuple[int, int, int] = (
        1,
        DEFAULT_OBSERVATION_SCHEMA.road_channels,
        DEFAULT_OBSERVATION_SCHEMA.road_points,
    )
    action_shape: tuple[int, int] = (1, DEFAULT_ACTION_SCHEMA.dimension)
    input_names: tuple[str, str] = ("state", "road")
    output_names: tuple[str, ...] = ("action",)
    opset_version: int = 17


@dataclass(frozen=True)
class OnnxExportResult:
    path: str
    sha256: str
    input_shapes: Mapping[str, tuple[int, ...]]
    output_shapes: Mapping[str, tuple[int, ...]]
    opset_version: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def onnx_dependencies_available(*, runtime: bool = False) -> bool:
    modules = ["torch", "onnx"]
    if runtime:
        modules.append("onnxruntime")
    return all(importlib.util.find_spec(module) is not None for module in modules)


def default_export_inputs(
    observation_schema: ObservationSchema = DEFAULT_OBSERVATION_SCHEMA,
) -> dict[str, np.ndarray]:
    return {
        "state": np.zeros(
            (1, observation_schema.state_vector_dim), dtype=np.float32
        ),
        "road": np.zeros(
            (
                1,
                observation_schema.road_channels,
                observation_schema.road_points,
            ),
            dtype=np.float32,
        ),
    }


def export_fixed_shape_onnx(
    model,
    output_path: str | Path,
    sample_inputs: Mapping[str, np.ndarray] | Sequence[np.ndarray] | None = None,
    *,
    config: FixedShapeOnnxConfig = FixedShapeOnnxConfig(),
) -> OnnxExportResult:
    """Export without dynamic axes and validate concrete graph dimensions."""

    if not onnx_dependencies_available():
        raise OptionalOnnxDependencyError(
            "fixed-shape export requires optional 'torch' and 'onnx' packages"
        )
    import onnx
    import torch

    inputs = default_export_inputs() if sample_inputs is None else sample_inputs
    if isinstance(inputs, Mapping):
        if set(inputs) != set(config.input_names):
            raise ValueError(
                f"sample input names must be {config.input_names!r}"
            )
        arrays = tuple(
            np.asarray(inputs[name], dtype=np.float32)
            for name in config.input_names
        )
    else:
        arrays = tuple(np.asarray(value, dtype=np.float32) for value in inputs)
    expected_shapes = (config.state_shape, config.road_shape)
    if len(arrays) != len(expected_shapes):
        raise ValueError("sample_inputs must contain state and road tensors")
    for name, array, shape in zip(config.input_names, arrays, expected_shapes):
        if array.shape != shape:
            raise ValueError(f"{name} must have fixed shape {shape}, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or Inf")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tensor_inputs = tuple(torch.from_numpy(array) for array in arrays)
    if hasattr(model, "eval"):
        model.eval()
    torch.onnx.export(
        model,
        tensor_inputs,
        str(target),
        input_names=list(config.input_names),
        output_names=list(config.output_names),
        opset_version=config.opset_version,
        dynamic_axes=None,
        do_constant_folding=True,
    )
    graph = onnx.load(str(target))
    onnx.checker.check_model(graph)
    input_shapes, output_shapes = inspect_onnx_shapes(graph)
    required_inputs = dict(zip(config.input_names, expected_shapes))
    for name, shape in required_inputs.items():
        if input_shapes.get(name) != shape:
            raise ValueError(
                f"exported input {name!r} is not fixed at {shape}: "
                f"{input_shapes.get(name)!r}"
            )
    if len(config.output_names) == 1:
        shape = output_shapes.get(config.output_names[0])
        if shape != config.action_shape:
            raise ValueError(
                f"exported action must have fixed shape {config.action_shape}, got {shape}"
            )
    return OnnxExportResult(
        path=str(target),
        sha256=sha256_file(target),
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        opset_version=config.opset_version,
    )


def inspect_onnx_shapes(model_or_path) -> tuple[
    dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]
]:
    if importlib.util.find_spec("onnx") is None:
        raise OptionalOnnxDependencyError("ONNX graph inspection requires 'onnx'")
    import onnx

    graph_model = (
        onnx.load(str(model_or_path))
        if isinstance(model_or_path, (str, Path))
        else model_or_path
    )

    def shapes(values) -> dict[str, tuple[int, ...]]:
        result: dict[str, tuple[int, ...]] = {}
        for value in values:
            dimensions = value.type.tensor_type.shape.dim
            concrete: list[int] = []
            for dimension in dimensions:
                if not dimension.HasField("dim_value"):
                    raise ValueError(f"tensor {value.name!r} has a dynamic dimension")
                concrete.append(int(dimension.dim_value))
            result[value.name] = tuple(concrete)
        return result

    return shapes(graph_model.graph.input), shapes(graph_model.graph.output)


def validate_action_output(
    action: np.ndarray,
    action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA,
) -> np.ndarray:
    """Validate a batched fixed-shape export output in physical units."""

    value = np.asarray(action, dtype=np.float64)
    if value.shape != (1, action_schema.dimension):
        raise ValueError(
            f"action output must have shape (1, {action_schema.dimension})"
        )
    action_schema.validate(value[0])
    return value
