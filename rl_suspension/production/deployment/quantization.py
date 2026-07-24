"""Quantization validation hooks; quantized artifacts are never auto-promoted."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from rl_suspension.production.deployment.golden import (
    GoldenVectorSet,
    GoldenVerificationReport,
    generate_golden_vectors,
    verify_golden_vectors,
)


@dataclass(frozen=True)
class QuantizationHookResult:
    name: str
    passed: bool
    detail: str = ""
    metrics: Mapping[str, float] | None = None


class QuantizationValidationHook(Protocol):
    def __call__(
        self,
        reference: GoldenVectorSet,
        quantized: GoldenVectorSet,
    ) -> QuantizationHookResult | bool: ...


@dataclass(frozen=True)
class QuantizationValidationReport:
    passed: bool
    numerical_equivalence: GoldenVerificationReport
    hooks: tuple[QuantizationHookResult, ...]

    def __bool__(self) -> bool:
        return self.passed

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "numerical_equivalence": self.numerical_equivalence.to_dict(),
            "hooks": [asdict(result) for result in self.hooks],
        }


def validate_quantized_model(
    reference_predictor: Callable[..., Any],
    quantized_predictor: Callable[..., Any],
    inputs: Mapping[str, np.ndarray],
    *,
    output_names: Sequence[str] = ("action",),
    absolute_tolerance: float = 1e-3,
    relative_tolerance: float = 1e-2,
    hooks: Sequence[QuantizationValidationHook] = (),
) -> QuantizationValidationReport:
    """Run numerical equivalence followed by project-specific validation hooks."""

    reference = generate_golden_vectors(
        reference_predictor,
        inputs,
        output_names=output_names,
        metadata={"role": "pre_quantization_reference"},
    )
    equivalence = verify_golden_vectors(
        quantized_predictor,
        reference,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    quantized = generate_golden_vectors(
        quantized_predictor,
        inputs,
        output_names=output_names,
        metadata={"role": "quantized_candidate"},
    )
    results: list[QuantizationHookResult] = []
    for index, hook in enumerate(hooks):
        result = hook(reference, quantized)
        if isinstance(result, QuantizationHookResult):
            results.append(result)
        elif isinstance(result, (bool, np.bool_)):
            results.append(
                QuantizationHookResult(
                    name=getattr(hook, "__name__", f"hook_{index}"),
                    passed=bool(result),
                )
            )
        else:
            raise TypeError(
                "quantization hook must return QuantizationHookResult or bool"
            )
    return QuantizationValidationReport(
        passed=equivalence.passed and all(result.passed for result in results),
        numerical_equivalence=equivalence,
        hooks=tuple(results),
    )


validate_quantization = validate_quantized_model
