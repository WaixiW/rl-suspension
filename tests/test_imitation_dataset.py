import numpy as np

from rl_suspension.envs.observation import OBSERVATION_SPEC
from rl_suspension.imitation.dataset import (
    EpisodePhase,
    EpisodeShardWriter,
    PhaseBalancedSampler,
    compute_normalization,
    load_dataset,
)


def _episode(count=20):
    phases = np.array(
        [EpisodePhase.FLAT] * 10
        + [EpisodePhase.PREVIEW] * 3
        + [EpisodePhase.FRONT_CONTACT] * 3
        + [EpisodePhase.REAR_CONTACT] * 2
        + [EpisodePhase.RECOVERY] * 2,
        dtype=np.uint8,
    )[:count]
    return {
        "observations": np.zeros((count, OBSERVATION_SPEC.dimension), dtype=np.float32),
        "actions": np.zeros((count, 4), dtype=np.float32),
        "behavior_actions": np.zeros((count, 4), dtype=np.float32),
        "rewards": np.zeros(count, dtype=np.float32),
        "next_observations": np.ones((count, OBSERVATION_SPEC.dimension), dtype=np.float32),
        "terminated": np.zeros(count, dtype=np.bool_),
        "truncated": np.zeros(count, dtype=np.bool_),
        "episode_ids": np.zeros(count, dtype=np.int64),
        "scenario_seeds": np.zeros(count, dtype=np.int64),
        "phases": phases,
        "expert_valid": np.ones(count, dtype=np.bool_),
        "expert_quality": np.ones(count, dtype=np.float32),
        "expert_status": np.zeros(count, dtype=np.int8),
        "expert_objective": np.full(count, np.nan, dtype=np.float64),
        "expert_iterations": np.zeros(count, dtype=np.int32),
        "expert_latency_ms": np.full(count, np.nan, dtype=np.float32),
        "expert_constraint_margin": np.full(count, np.nan, dtype=np.float32),
        "expert_fallback": np.zeros(count, dtype=np.bool_),
        "constraint_violation": np.zeros(count, dtype=np.float32),
    }


def test_episode_shard_round_trip_and_normalization(tmp_path):
    writer = EpisodeShardWriter(tmp_path)
    writer.write_episode(
        episode_id=0,
        scenario_seed=0,
        scenario_kind="single_bump",
        teacher_name="test",
        transitions=_episode(),
    )

    dataset = load_dataset(tmp_path, split="train")
    stats = compute_normalization(dataset)

    assert len(dataset) == 20
    assert dataset.behavior_actions.shape == (20, 4)
    assert len(stats.observation_mean) == OBSERVATION_SPEC.dimension
    assert np.all(np.asarray(stats.observation_std) >= 1e-6)


def test_phase_balanced_sampler_caps_flat_samples(tmp_path):
    writer = EpisodeShardWriter(tmp_path)
    writer.write_episode(
        episode_id=0,
        scenario_seed=0,
        scenario_kind="single_bump",
        teacher_name="test",
        transitions=_episode(),
    )
    dataset = load_dataset(tmp_path, split="train")
    sampler = PhaseBalancedSampler(dataset, seed=0, flat_fraction=0.15)
    indices = sampler.sample_indices(100)
    sampled_phases = dataset.phases[indices]

    assert indices.shape == (100,)
    assert np.mean(sampled_phases == EpisodePhase.FLAT) <= 0.16
