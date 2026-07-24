"""Repeatable single-sample inference latency benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter_ns
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class RuntimeBenchmark:
    warmup_iterations: int
    measured_iterations: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    maximum_latency_ms: float
    throughput_hz: float
    deadline_ms: float | None
    deadline_misses: int

    @property
    def deadline_met(self) -> bool:
        return self.deadline_ms is None or self.p99_latency_ms <= self.deadline_ms

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return {**asdict(self), "deadline_met": self.deadline_met}


def benchmark_runtime(
    predictor: Callable[..., Any],
    inputs: Mapping[str, np.ndarray],
    *,
    warmup_iterations: int = 20,
    measured_iterations: int = 200,
    deadline_ms: float | None = None,
    clock_ns: Callable[[], int] = perf_counter_ns,
) -> RuntimeBenchmark:
    """Benchmark the complete predictor call, excluding setup and allocation."""

    if warmup_iterations < 0 or measured_iterations < 1:
        raise ValueError("invalid benchmark iteration count")
    if deadline_ms is not None and deadline_ms <= 0.0:
        raise ValueError("deadline_ms must be positive")
    normalized = {name: np.asarray(value) for name, value in inputs.items()}
    if not normalized:
        raise ValueError("benchmark inputs must not be empty")

    for _ in range(warmup_iterations):
        _invoke(predictor, normalized)

    latencies = np.empty(measured_iterations, dtype=np.float64)
    for index in range(measured_iterations):
        started = clock_ns()
        _invoke(predictor, normalized)
        elapsed = clock_ns() - started
        if elapsed < 0:
            raise ValueError("benchmark clock moved backwards")
        latencies[index] = elapsed / 1e6

    mean = float(np.mean(latencies))
    return RuntimeBenchmark(
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        mean_latency_ms=mean,
        p50_latency_ms=float(np.quantile(latencies, 0.50)),
        p95_latency_ms=float(np.quantile(latencies, 0.95)),
        p99_latency_ms=float(np.quantile(latencies, 0.99)),
        maximum_latency_ms=float(np.max(latencies)),
        throughput_hz=float(1000.0 / mean) if mean > 0.0 else float("inf"),
        deadline_ms=deadline_ms,
        deadline_misses=(
            int(np.count_nonzero(latencies > deadline_ms))
            if deadline_ms is not None
            else 0
        ),
    )


benchmark_inference = benchmark_runtime


def _invoke(predictor: Callable[..., Any], inputs: Mapping[str, np.ndarray]) -> Any:
    target = getattr(predictor, "predict", predictor)
    try:
        return target(**inputs)
    except TypeError as keyword_error:
        try:
            return target(*inputs.values())
        except TypeError:
            raise keyword_error
