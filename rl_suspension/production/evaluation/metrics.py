"""Statistical metrics for production controller qualification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BootstrapInterval:
    """A deterministic percentile-bootstrap confidence interval."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int
    resamples: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def bootstrap_confidence_interval(
    values: Sequence[float] | NDArray[np.floating],
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 0,
    statistic: Callable[[NDArray[np.float64]], float] = np.mean,
) -> BootstrapInterval:
    """Estimate a scalar statistic and its percentile-bootstrap interval.

    Paired comparisons remain paired by passing the element-wise differences
    between matched episodes.
    """

    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size == 0:
        raise ValueError("values must not be empty")
    if not np.all(np.isfinite(data)):
        raise ValueError("values contain NaN or Inf")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")
    if resamples < 1:
        raise ValueError("resamples must be positive")

    estimate = float(statistic(data))
    if data.size == 1:
        lower = upper = estimate
    else:
        generator = np.random.default_rng(seed)
        estimates = np.empty(resamples, dtype=np.float64)
        for index in range(resamples):
            sample = data[generator.integers(0, data.size, size=data.size)]
            estimates[index] = float(statistic(sample))
        alpha = (1.0 - confidence) / 2.0
        lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return BootstrapInterval(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        samples=int(data.size),
        resamples=resamples,
    )


@dataclass(frozen=True)
class ChannelErrorMetrics:
    mae: float
    rmse: float
    bias: float
    maximum_absolute_error: float
    p95_absolute_error: float
    delta_mae: float
    delta_rmse: float
    delta_maximum_absolute_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class OpenLoopReport:
    """Per-channel action imitation errors and temporal-delta errors."""

    samples: int
    channels: int
    per_channel: dict[str, ChannelErrorMetrics]
    aggregate: ChannelErrorMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "channels": self.channels,
            "per_channel": {
                name: metrics.to_dict() for name, metrics in self.per_channel.items()
            },
            "aggregate": self.aggregate.to_dict(),
        }


def open_loop_metrics(
    reference: NDArray[np.floating],
    candidate: NDArray[np.floating],
    *,
    channel_names: Sequence[str] | None = None,
) -> OpenLoopReport:
    """Compare candidate actions with expert actions without simulation."""

    expected = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(candidate, dtype=np.float64)
    if expected.ndim != 2:
        raise ValueError("reference and candidate must have shape (samples, channels)")
    if expected.shape != actual.shape or expected.shape[0] == 0:
        raise ValueError("reference and candidate must have equal, nonempty shapes")
    if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual)):
        raise ValueError("open-loop actions contain NaN or Inf")

    names = (
        tuple(channel_names)
        if channel_names is not None
        else tuple(f"action_{index}" for index in range(expected.shape[1]))
    )
    if len(names) != expected.shape[1] or len(set(names)) != len(names):
        raise ValueError("channel_names must be unique and match the channel dimension")

    error = actual - expected
    delta_error = (
        np.diff(actual, axis=0) - np.diff(expected, axis=0)
        if expected.shape[0] > 1
        else np.zeros((1, expected.shape[1]), dtype=np.float64)
    )
    per_channel = {
        name: _error_metrics(error[:, index], delta_error[:, index])
        for index, name in enumerate(names)
    }
    return OpenLoopReport(
        samples=int(expected.shape[0]),
        channels=int(expected.shape[1]),
        per_channel=per_channel,
        aggregate=_error_metrics(error.reshape(-1), delta_error.reshape(-1)),
    )


def _error_metrics(error: NDArray[np.float64], delta_error: NDArray[np.float64]) -> ChannelErrorMetrics:
    absolute = np.abs(error)
    absolute_delta = np.abs(delta_error)
    return ChannelErrorMetrics(
        mae=float(np.mean(absolute)),
        rmse=float(np.sqrt(np.mean(np.square(error)))),
        bias=float(np.mean(error)),
        maximum_absolute_error=float(np.max(absolute)),
        p95_absolute_error=float(np.quantile(absolute, 0.95)),
        delta_mae=float(np.mean(absolute_delta)),
        delta_rmse=float(np.sqrt(np.mean(np.square(delta_error)))),
        delta_maximum_absolute_error=float(np.max(absolute_delta)),
    )
