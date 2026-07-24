"""Convert validated episode shards into production BC arrays."""

from __future__ import annotations

import numpy as np

from rl_suspension.production.data.dataset import Direct12Dataset
from rl_suspension.production.data.normalization import GroupedNormalization
from rl_suspension.production.training.bc import BCDatasetArrays


def to_bc_dataset(
    dataset: Direct12Dataset,
    normalization: GroupedNormalization,
    *,
    history_steps: int = 1,
) -> BCDatasetArrays:
    if history_steps <= 0:
        raise ValueError("history_steps must be positive")
    states, roads = normalization.normalize(
        dataset.state_observations,
        dataset.road_observations,
    )
    episode_strings = np.asarray(dataset.episode_ids, dtype=str)
    unique_episodes = {
        episode: index for index, episode in enumerate(dict.fromkeys(episode_strings))
    }
    episode_ids = np.asarray(
        [unique_episodes[value] for value in episode_strings],
        dtype=np.int64,
    )
    sequence_indices = np.empty(len(dataset), dtype=np.int64)
    for episode in np.unique(episode_ids):
        selected = np.flatnonzero(episode_ids == episode)
        sequence_indices[selected] = np.arange(selected.size)

    margin = np.asarray(dataset.solver_feasibility_margin, dtype=np.float32)
    quality = np.where(
        np.asarray(dataset.expert_valid, dtype=bool),
        np.maximum(np.nan_to_num(margin, nan=0.0, posinf=1.0, neginf=0.0), 0.05),
        0.0,
    ).astype(np.float32)
    state_history = None
    road_history = None
    if history_steps > 1:
        state_history = _history_windows(states, episode_ids, history_steps)
        road_history = _history_windows(roads, episode_ids, history_steps)

    return BCDatasetArrays(
        states=states,
        roads=roads,
        actions=np.asarray(dataset.expert_actions_normalized, dtype=np.float32),
        phases=np.asarray(dataset.phases, dtype=np.int64),
        quality=quality,
        episode_ids=episode_ids,
        sequence_indices=sequence_indices,
        valid=np.asarray(dataset.expert_valid, dtype=bool),
        state_history=state_history,
        road_history=road_history,
        actions_normalized=True,
    )


def _history_windows(
    values: np.ndarray,
    episode_ids: np.ndarray,
    history_steps: int,
) -> np.ndarray:
    output = np.empty(
        (values.shape[0], history_steps, *values.shape[1:]),
        dtype=values.dtype,
    )
    for index in range(values.shape[0]):
        episode_start = index
        while (
            episode_start > 0
            and episode_ids[episode_start - 1] == episode_ids[index]
            and index - episode_start + 1 < history_steps
        ):
            episode_start -= 1
        selected = values[episode_start : index + 1]
        padding = history_steps - selected.shape[0]
        if padding:
            selected = np.concatenate(
                [np.repeat(selected[:1], padding, axis=0), selected],
                axis=0,
            )
        output[index] = selected
    return output
