import numpy as np
import pytest

from rl_suspension.envs.observation import OBSERVATION_SPEC
from rl_suspension.road import RoadProfile, ScenarioConfig, four_wheel_time_preview


def test_observation_slices_cover_exactly_all_features():
    covered = np.zeros(OBSERVATION_SPEC.dimension, dtype=np.int64)
    for field in (
        "state",
        "suspension_deflections",
        "suspension_velocities",
        "previous_forces",
        "currents",
        "pump_speeds",
        "actual_forces",
        "speed",
        "ads_features",
        "road_left",
        "road_right",
    ):
        covered[getattr(OBSERVATION_SPEC, field)] += 1

    assert np.all(covered == 1)


def test_flat_road_has_no_detected_bump():
    config = ScenarioConfig(kind="flat")
    features = RoadProfile(config).features(vehicle_x=0.0)

    assert features.shape == (7,)
    assert features[OBSERVATION_SPEC.ads_peak_distance] == config.preview_distance
    assert features[OBSERVATION_SPEC.ads_peak_height] == 0.0
    assert features[OBSERVATION_SPEC.ads_confidence] == 0.0


def test_four_wheel_preview_aligns_front_and_rear_contact():
    config = ScenarioConfig(
        kind="single_bump",
        speed=10.0,
        bump_start=1.0,
        bump_width=0.5,
    )
    road = RoadProfile(config)
    profile, offsets = road.extended_preview(vehicle_x=0.0)
    observation = np.zeros(OBSERVATION_SPEC.dimension, dtype=np.float32)
    observation[OBSERVATION_SPEC.speed] = config.speed
    observation[OBSERVATION_SPEC.road_left] = profile[0]
    observation[OBSERVATION_SPEC.road_right] = profile[1]

    preview = four_wheel_time_preview(observation, horizon=20, dt=0.01)

    assert preview.shape == (20, 4)
    assert np.max(preview[:, :2]) > 0.0
    assert np.max(preview[:, 2:]) == 0.0


@pytest.mark.parametrize("kind", ["single_bump", "double_bump", "asymmetric_bump"])
@pytest.mark.parametrize("speed", [10.0, 15.0])
def test_time_preview_matches_direct_wheel_sampling(kind, speed):
    config = ScenarioConfig(
        kind=kind,
        speed=speed,
        bump_start=1.0,
        bump_width=0.5,
        double_spacing=0.8,
    )
    road = RoadProfile(config)
    profile, _ = road.extended_preview(vehicle_x=0.0)
    observation = np.zeros(OBSERVATION_SPEC.dimension, dtype=np.float32)
    observation[OBSERVATION_SPEC.speed] = speed
    observation[OBSERVATION_SPEC.road_left] = profile[0]
    observation[OBSERVATION_SPEC.road_right] = profile[1]
    horizon = 30

    preview = four_wheel_time_preview(observation, horizon=horizon, dt=0.01)
    expected = np.stack(
        [road.wheel_heights(speed * 0.01 * k) for k in range(horizon)]
    )

    np.testing.assert_allclose(preview, expected, atol=1e-7)
