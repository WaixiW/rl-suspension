"""Diagnostics for contradictory expert actions at similar observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from rl_suspension.production.data.dataset import Direct12Dataset


@dataclass(frozen=True)
class AmbiguousPair:
    first_index: int
    second_index: int
    observation_distance: float
    action_distance: float
    first_scenario_id: str = ""
    second_scenario_id: str = ""


@dataclass(frozen=True)
class ActionAmbiguityReport:
    samples_examined: int
    neighbor_pairs: int
    ambiguous_pairs: int
    ambiguity_rate: float
    maximum_action_distance: float
    observation_radius: float
    action_threshold: float
    examples: tuple[AmbiguousPair, ...]


def diagnose_action_ambiguity(
    observations_or_dataset: Direct12Dataset | NDArray[np.floating],
    actions: NDArray[np.floating] | None = None,
    *,
    road_observations: NDArray[np.floating] | None = None,
    scenario_ids: Sequence[str] | NDArray[np.str_] | None = None,
    valid_labels: NDArray[np.bool_] | None = None,
    observation_radius: float = 0.05,
    action_threshold: float = 0.25,
    cross_scenario_only: bool = True,
    maximum_samples: int = 2000,
    maximum_examples: int = 20,
    seed: int = 0,
) -> ActionAmbiguityReport:
    """Find nearby normalized observations with materially different actions.

    Observation distance is RMS distance after per-feature standardization;
    action distance is maximum absolute normalized-command difference.
    """

    if observation_radius < 0.0 or action_threshold < 0.0:
        raise ValueError("ambiguity thresholds must be nonnegative")
    if maximum_samples <= 1 or maximum_examples < 0:
        raise ValueError("invalid diagnostic sample limits")
    if isinstance(observations_or_dataset, Direct12Dataset):
        dataset = observations_or_dataset
        observations = np.asarray(dataset.state_observations, dtype=np.float64)
        roads = np.asarray(dataset.road_observations, dtype=np.float64)
        action_values = np.asarray(dataset.expert_actions_normalized, dtype=np.float64)
        ids = np.asarray(dataset.scenario_ids, dtype=str)
        valid = np.asarray(dataset.expert_valid, dtype=np.bool_)
    else:
        observations = np.asarray(observations_or_dataset, dtype=np.float64)
        roads = (
            None
            if road_observations is None
            else np.asarray(road_observations, dtype=np.float64)
        )
        if actions is None:
            raise ValueError("actions are required when a dataset is not supplied")
        action_values = np.asarray(actions, dtype=np.float64)
        ids = (
            np.full(observations.shape[0], "", dtype=str)
            if scenario_ids is None
            else np.asarray(scenario_ids, dtype=str)
        )
        valid = (
            np.ones(observations.shape[0], dtype=np.bool_)
            if valid_labels is None
            else np.asarray(valid_labels, dtype=np.bool_)
        )
    count = observations.shape[0]
    if (
        observations.ndim != 2
        or action_values.ndim != 2
        or action_values.shape[0] != count
        or ids.shape != (count,)
        or valid.shape != (count,)
        or (roads is not None and roads.shape[0] != count)
    ):
        raise ValueError("ambiguity diagnostic arrays have inconsistent shapes")
    finite = np.all(np.isfinite(observations), axis=1) & np.all(
        np.isfinite(action_values),
        axis=1,
    )
    indices = np.flatnonzero(valid & finite)
    if indices.size < 2:
        return ActionAmbiguityReport(
            samples_examined=int(indices.size),
            neighbor_pairs=0,
            ambiguous_pairs=0,
            ambiguity_rate=0.0,
            maximum_action_distance=0.0,
            observation_radius=observation_radius,
            action_threshold=action_threshold,
            examples=(),
        )
    if indices.size > maximum_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(
            rng.choice(indices, size=maximum_samples, replace=False)
        )
    features = observations[indices]
    if roads is not None:
        road_features = roads[indices].reshape(indices.size, -1)
        features = np.concatenate((features, road_features), axis=1)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    standardized = (features - mean) / np.where(std < 1e-6, 1.0, std)
    selected_actions = action_values[indices]
    selected_ids = ids[indices]

    euclidean_radius = observation_radius * np.sqrt(standardized.shape[1])
    pairs = cKDTree(standardized).query_pairs(
        euclidean_radius,
        output_type="ndarray",
    )
    if pairs.size:
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    else:
        pairs = np.empty((0, 2), dtype=np.int64)
    if cross_scenario_only and np.any(selected_ids != "") and pairs.size:
        pairs = pairs[selected_ids[pairs[:, 0]] != selected_ids[pairs[:, 1]]]
    if pairs.size:
        observation_distances = np.sqrt(
            np.mean(
                np.square(
                    standardized[pairs[:, 0]] - standardized[pairs[:, 1]]
                ),
                axis=1,
            )
        )
        pairs = pairs[observation_distances <= observation_radius + 1e-12]
        observation_distances = observation_distances[
            observation_distances <= observation_radius + 1e-12
        ]
    else:
        observation_distances = np.empty(0, dtype=np.float64)
    action_distances = (
        np.max(
            np.abs(
                selected_actions[pairs[:, 0]] - selected_actions[pairs[:, 1]]
            ),
            axis=1,
        )
        if pairs.size
        else np.empty(0, dtype=np.float64)
    )
    ambiguous_mask = action_distances >= action_threshold
    ambiguous = pairs[ambiguous_mask]
    ambiguous_observation_distances = observation_distances[ambiguous_mask]
    ambiguous_action_distances = action_distances[ambiguous_mask]
    examples = tuple(
        AmbiguousPair(
            first_index=int(indices[first]),
            second_index=int(indices[second]),
            observation_distance=float(observation_distance),
            action_distance=float(action_distance),
            first_scenario_id=str(selected_ids[first]),
            second_scenario_id=str(selected_ids[second]),
        )
        for (first, second), observation_distance, action_distance in zip(
            ambiguous[:maximum_examples],
            ambiguous_observation_distances[:maximum_examples],
            ambiguous_action_distances[:maximum_examples],
        )
    )
    neighbor_pairs = int(pairs.shape[0])
    ambiguous_pairs = int(ambiguous.shape[0])
    maximum_distance = (
        float(np.max(action_distances)) if action_distances.size else 0.0
    )
    return ActionAmbiguityReport(
        samples_examined=int(indices.size),
        neighbor_pairs=neighbor_pairs,
        ambiguous_pairs=ambiguous_pairs,
        ambiguity_rate=(
            float(ambiguous_pairs / neighbor_pairs) if neighbor_pairs else 0.0
        ),
        maximum_action_distance=maximum_distance,
        observation_radius=observation_radius,
        action_threshold=action_threshold,
        examples=examples,
    )


action_ambiguity_diagnostic = diagnose_action_ambiguity
