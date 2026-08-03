import torch

from modular_suspension_rl.config import ObservationConfig, RewardConfig
from modular_suspension_rl.objective import RewardSignals, SuspensionReward
from modular_suspension_rl.randomization import (
    ADSCorruptor,
    DomainRandomizer,
    RoadScenario,
    RoadScenarioGenerator,
)


def make_signals(batch_size=2):
    scalar = torch.zeros(batch_size)
    wheel = torch.zeros(batch_size, 4)
    return RewardSignals(
        vertical_acceleration_mps2=scalar.clone(),
        pitch_rate_radps=scalar.clone(),
        pitch_acceleration_radps2=scalar.clone(),
        roll_rate_radps=scalar.clone(),
        roll_acceleration_radps2=scalar.clone(),
        passenger_acceleration_mps2=torch.zeros(batch_size, 2),
        jerk_mps3=scalar.clone(),
        suspension_travel_m=wheel.clone(),
        tire_load_n=torch.full((batch_size, 4), 3500.0),
        tire_load_variation_n=wheel.clone(),
        wheel_hop_mps=wheel.clone(),
        electrical_power_w=scalar.clone(),
        command_slew_normalized=scalar.clone(),
        force_tracking_error_n=wheel.clone(),
        gate=wheel.clone(),
        previous_gate=wheel.clone(),
        preview_confidence=torch.ones(batch_size, 4),
        end_stop_event=scalar.clone(),
        hardware_violation=scalar.clone(),
    )


def test_reward_penalizes_comfort_and_hard_violations():
    calculator = SuspensionReward(RewardConfig())
    nominal = calculator(make_signals())
    disturbed_signals = make_signals()
    disturbed_signals.vertical_acceleration_mps2[:] = 5.0
    disturbed_signals.end_stop_event[0] = 1.0
    disturbed = calculator(disturbed_signals)

    assert torch.allclose(nominal.reward, torch.zeros(2))
    assert disturbed.reward[0] < disturbed.reward[1] < nominal.reward[1]
    assert disturbed.hard_violation[0] == 1.0


def test_road_and_ads_randomization_cover_asymmetry_and_range_confidence():
    config = ObservationConfig(feedback_dim=20, feedback_history=4)
    generator = RoadScenarioGenerator(config, seed=4)
    randomizer = DomainRandomizer(seed=4)
    corruptor = ADSCorruptor(config, seed=4)
    road, scenarios = generator.generate(
        3, RoadScenario.ASYMMETRIC_BUMP, severity=1.0
    )
    domain = randomizer.sample(3, stage=3)
    measured, confidence, valid = corruptor.apply(road, domain)

    assert road.shape == (3, 4, 160)
    assert all(item == RoadScenario.ASYMMETRIC_BUMP for item in scenarios)
    assert torch.any(road[:, 0] != road[:, 1])
    assert measured.shape == confidence.shape == valid.shape == road.shape
    assert torch.all((confidence >= 0.0) & (confidence <= 1.0))
    assert confidence[..., -1].mean() <= confidence[..., 0].mean()
