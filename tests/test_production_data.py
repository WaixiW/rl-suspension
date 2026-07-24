from dataclasses import replace
import json

import numpy as np
import pytest

from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    contract_payload,
    Scenario,
)
from rl_suspension.production.data import (
    EpisodePhase,
    EpisodeShardWriter,
    PhaseBalancedSampler,
    build_data_card,
    collect_episode,
    compute_grouped_normalization,
    diagnose_action_ambiguity,
    load_dataset,
    to_bc_dataset,
    validate_dataset,
)
from rl_suspension.production.provenance import (
    AppendOnlyRunDirectory,
    RunManifest,
    canonical_sha256,
)
from rl_suspension.production.qualification import (
    QualificationConfig,
    bootstrap_confidence_interval,
    qualify_mpc,
)
from rl_suspension.production.reference import (
    ReferenceDirect12Simulator,
    ReferenceMpcAdapter,
)
from rl_suspension.production.scenarios import (
    ScenarioGenerator,
    scenario_payloads,
    validate_split_safety,
)


def _scenario(
    scenario_id: str = "train-single-001",
    split: str = "train",
    steps: int = 24,
) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        seed=17,
        split=split,
        bump_family="single_bump",
        parameters={
            "speed_mps": 12.0,
            "bump_start_m": 1.5,
            "bump_height_m": 0.05,
            "bump_width_m": 0.6,
            "episode_steps": steps,
        },
    )


def _collect(root, scenario=None):
    selected = scenario or _scenario()
    writer = EpisodeShardWriter(root)
    record = collect_episode(
        ReferenceDirect12Simulator(),
        ReferenceMpcAdapter(),
        selected,
        writer,
    )
    return writer, record


def test_canonical_manifest_and_append_only_run_directory(tmp_path):
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
    manifest = RunManifest.create(
        configuration={"seed": 7, "nested": {"beta": [1, 0]}},
        contracts=contract_payload(
            ReferenceDirect12Simulator().observation_schema,
            DEFAULT_ACTION_SCHEMA,
        ),
        scenarios=scenario_payloads([_scenario()]),
        created_at_utc="2026-01-01T00:00:00+00:00",
        code_revision="abc123",
    )
    run = AppendOnlyRunDirectory.create(tmp_path, manifest)
    artifact_hash = run.append_json("reports/metrics.json", {"loss": 1.25})

    assert run.verify_artifact("reports/metrics.json", artifact_hash)
    assert AppendOnlyRunDirectory.open(run.path).manifest == manifest
    with pytest.raises(FileExistsError):
        run.append_json("reports/metrics.json", {"loss": 0.0})
    with pytest.raises(FileExistsError):
        AppendOnlyRunDirectory.create(tmp_path, manifest)
    with pytest.raises(ValueError):
        run.append_bytes("../escape.bin", b"bad")


def test_scenario_generation_is_deterministic_stratified_and_split_safe():
    counts = {"train": 10, "validation": 7, "test": 5}
    first = ScenarioGenerator(seed=123).generate(counts)
    second = ScenarioGenerator(seed=123).generate(counts)

    assert first == second
    assert len(first) == sum(counts.values())
    assert len({scenario.scenario_id for scenario in first}) == len(first)
    assert len({scenario.seed for scenario in first}) == len(first)
    for split, count in counts.items():
        selected = [scenario for scenario in first if scenario.split == split]
        family_counts = [
            sum(item.bump_family == family for item in selected)
            for family in ScenarioGenerator(seed=123).space.families
        ]
        assert len(selected) == count
        assert max(family_counts) - min(family_counts) <= 1
    validate_split_safety(first)
    with pytest.raises(ValueError, match="duplicate scenario_id"):
        validate_split_safety([first[0], first[0]])


def test_paired_qualification_bootstrap_and_action_gate():
    scenarios = [_scenario(f"qualification-{index}", steps=30) for index in range(3)]
    config = QualificationConfig(
        bootstrap_samples=300,
        bootstrap_seed=99,
        minimum_comfort_improvement_lower_bound=-10.0,
        minimum_return_improvement_lower_bound=-10.0,
        maximum_raw_slew_violation_rate=1.0,
        maximum_safety_violation_rate=1.0,
    )
    report = qualify_mpc(
        ReferenceMpcAdapter(),
        ReferenceDirect12Simulator,
        scenarios,
        config=config,
    )

    assert report.passed
    assert report.solver_calls == sum(item.mpc.steps for item in report.paired_episodes)
    assert report.comfort_improvement == bootstrap_confidence_interval(
        [item.comfort_improvement for item in report.paired_episodes],
        confidence=config.confidence,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed,
    )

    class OutOfBoundsMpc(ReferenceMpcAdapter):
        def solve(self, observation, simulator_snapshot):
            result = super().solve(observation, simulator_snapshot)
            return replace(
                result,
                action_12d=np.asarray(DEFAULT_ACTION_SCHEMA.maximum) * 2.0,
            )

    failed = qualify_mpc(
        OutOfBoundsMpc(),
        ReferenceDirect12Simulator,
        scenarios[:1],
        config=config,
    )
    assert not failed.passed
    assert not failed.gates["action_bounds"]


def test_atomic_resumable_collection_and_round_trip(tmp_path):
    writer, first = _collect(tmp_path)
    second = collect_episode(
        ReferenceDirect12Simulator(),
        ReferenceMpcAdapter(),
        _scenario(),
        writer,
    )
    dataset = load_dataset(tmp_path, split="train")

    assert second == first
    assert len(writer.records) == 1
    assert len(list((tmp_path / "shards").glob("*.npz"))) == 1
    assert not list(tmp_path.rglob("*.tmp"))
    assert dataset.expert_actions_physical.shape == (first.transitions, 12)
    assert dataset.expert_actions_normalized.shape == (first.transitions, 12)
    assert dataset.behavior_actions_physical.shape == (first.transitions, 12)
    assert dataset.road_observations.shape[1:] == (4, 217)
    np.testing.assert_allclose(
        dataset.expert_actions_normalized,
        np.stack(
            [
                DEFAULT_ACTION_SCHEMA.normalize(action)
                for action in dataset.expert_actions_physical
            ]
        ),
        atol=2e-5,
    )
    assert set(dataset.solver_status) == {"optimal"}
    assert set(dataset.scenario_ids) == {_scenario().scenario_id}
    assert validate_dataset(tmp_path).passed


def test_validation_reports_corrupt_actions_labels_and_duplicates(tmp_path):
    clean_root = tmp_path / "clean"
    _collect(clean_root)
    clean = load_dataset(clean_root)
    arrays = {name: value.copy() for name, value in clean.arrays.items()}
    arrays.pop("episode_ids")
    arrays.pop("scenario_ids")
    arrays["state_observations"][0, 0] = np.nan
    arrays["expert_actions_physical"][1] = (
        np.asarray(DEFAULT_ACTION_SCHEMA.maximum) * 2.0
    )
    arrays["expert_actions_normalized"][1] = 2.0
    arrays["behavior_actions_physical"][1] = np.asarray(DEFAULT_ACTION_SCHEMA.maximum)
    arrays["behavior_actions_normalized"][1] = 1.0
    arrays["phases"][0] = 99
    arrays["expert_valid"][0] = False

    corrupt_root = tmp_path / "corrupt"
    writer = EpisodeShardWriter(corrupt_root)
    writer.write_episode(
        transitions=arrays,
        scenario=_scenario("corrupt-scenario"),
    )
    report = validate_dataset(corrupt_root)
    codes = {issue.code for issue in report.issues}
    assert not report.passed
    assert {
        "nonfinite",
        "action_bounds",
        "action_slew",
        "invalid_phase",
        "invalid_expert_label",
    }.issubset(codes)

    manifest_path = corrupt_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    duplicate = dict(manifest["shards"][0])
    duplicate["episode_id"] = "duplicate"
    duplicate["split"] = "test"
    manifest["shards"].append(duplicate)
    manifest["content_hash"] = canonical_sha256(manifest["shards"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    duplicate_report = validate_dataset(corrupt_root)
    duplicate_codes = {issue.code for issue in duplicate_report.issues}
    assert "duplicate_scenario" in duplicate_codes
    assert "split_leakage" in duplicate_codes
    assert "duplicate_transition" in duplicate_codes


def test_normalization_sampling_ambiguity_and_data_card(tmp_path):
    _collect(tmp_path)
    dataset = load_dataset(tmp_path)
    normalization = compute_grouped_normalization(dataset)
    bc_dataset = to_bc_dataset(dataset, normalization, history_steps=3)
    normalized_state, normalized_road = normalization.normalize(
        dataset.state_observations,
        dataset.road_observations,
    )
    assert normalized_state.shape == dataset.state_observations.shape
    assert normalized_road.shape == dataset.road_observations.shape
    assert np.all(np.isfinite(normalized_state))
    assert np.all(np.isfinite(normalized_road))
    assert len(bc_dataset) == len(dataset)
    assert bc_dataset.actions_normalized
    assert bc_dataset.state_history.shape[1] == 3
    np.testing.assert_allclose(bc_dataset.state_history[0, 0], bc_dataset.states[0])

    phases = np.asarray(
        [EpisodePhase.FLAT] * 80
        + [EpisodePhase.PREVIEW] * 5
        + [EpisodePhase.FRONT_CONTACT] * 5
        + [EpisodePhase.REAR_CONTACT] * 5
        + [EpisodePhase.RECOVERY] * 5
    )
    sampler = PhaseBalancedSampler(phases, seed=4, flat_fraction=0.1)
    sampled = phases[sampler.sample_indices(1000)]
    assert np.mean(sampled == EpisodePhase.FLAT) == pytest.approx(0.1)
    for phase in EpisodePhase:
        assert np.count_nonzero(sampled == phase) > 0

    ambiguity = diagnose_action_ambiguity(
        np.asarray([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]]),
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
        scenario_ids=np.asarray(["a", "b", "c"]),
        observation_radius=0.0,
        action_threshold=0.5,
    )
    assert ambiguity.neighbor_pairs == 1
    assert ambiguity.ambiguous_pairs == 1
    assert ambiguity.ambiguity_rate == 1.0

    card_path = tmp_path / "data-card.json"
    card = build_data_card(
        tmp_path,
        normalization=normalization,
        ambiguity=ambiguity,
        output_path=card_path,
    )
    assert card_path.exists()
    assert card["transitions"] == len(dataset)
    assert card["validation"]["passed"]
    assert card["normalization"]["split"] == "train"
