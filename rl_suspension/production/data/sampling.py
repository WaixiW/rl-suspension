"""Deterministic phase-balanced sampling over valid expert labels."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rl_suspension.production.data.collection import EpisodePhase
from rl_suspension.production.data.dataset import Direct12Dataset


class PhaseBalancedSampler:
    def __init__(
        self,
        dataset_or_phases: Direct12Dataset | NDArray[np.integer],
        *,
        valid_labels: NDArray[np.bool_] | None = None,
        seed: int = 0,
        flat_fraction: float = 0.15,
    ) -> None:
        if not 0.0 <= flat_fraction < 1.0:
            raise ValueError("flat_fraction must be within [0, 1)")
        if isinstance(dataset_or_phases, Direct12Dataset):
            phases = np.asarray(dataset_or_phases.phases, dtype=np.int64)
            valid = np.asarray(dataset_or_phases.expert_valid, dtype=np.bool_)
        else:
            phases = np.asarray(dataset_or_phases, dtype=np.int64)
            valid = (
                np.ones(phases.shape, dtype=np.bool_)
                if valid_labels is None
                else np.asarray(valid_labels, dtype=np.bool_)
            )
        if phases.ndim != 1 or valid.shape != phases.shape:
            raise ValueError("phases and valid_labels must be equal one-dimensional arrays")
        if not np.any(valid):
            raise ValueError("phase sampler requires at least one valid label")
        allowed = np.asarray([int(phase) for phase in EpisodePhase])
        if np.any(~np.isin(phases[valid], allowed)):
            raise ValueError("valid labels contain unknown episode phases")

        self.phases = phases
        self.valid = valid
        self.flat_fraction = float(flat_fraction)
        self.rng = np.random.default_rng(seed)
        self.indices = {
            phase: np.flatnonzero(valid & (phases == int(phase)))
            for phase in EpisodePhase
        }
        self.all_valid = np.flatnonzero(valid)

    def sample_indices(self, batch_size: int) -> NDArray[np.int64]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        requested_flat = int(round(batch_size * self.flat_fraction))
        selected: list[NDArray[np.int64]] = []
        flat_pool = self.indices[EpisodePhase.FLAT]
        flat_count = requested_flat if flat_pool.size else 0
        if flat_count:
            selected.append(
                self.rng.choice(flat_pool, size=flat_count, replace=True)
            )
        remaining = batch_size - flat_count
        event_phases = [
            phase
            for phase in EpisodePhase
            if phase is not EpisodePhase.FLAT and self.indices[phase].size
        ]
        if event_phases:
            for phase, count in zip(
                event_phases,
                _divide_evenly(remaining, len(event_phases)),
            ):
                if count:
                    selected.append(
                        self.rng.choice(
                            self.indices[phase],
                            size=count,
                            replace=True,
                        )
                    )
        elif remaining:
            selected.append(
                self.rng.choice(self.all_valid, size=remaining, replace=True)
            )
        result = np.concatenate(selected).astype(np.int64, copy=False)
        self.rng.shuffle(result)
        return result


def _divide_evenly(total: int, groups: int) -> list[int]:
    base, remainder = divmod(total, groups)
    return [base + int(index < remainder) for index in range(groups)]
