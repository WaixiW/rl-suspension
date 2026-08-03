import torch

from modular_suspension_rl.evaluation import EpisodeAccumulator, StepMetrics
from modular_suspension_rl.safety import (
    BandwidthReport,
    SafetyConfig,
    deployment_go_no_go,
)


def make_step(value):
    scalar = torch.tensor([value], dtype=torch.float32)
    wheel = torch.full((1, 4), value)
    return StepMetrics(
        vertical_acceleration_mps2=scalar,
        pitch_rate_radps=scalar,
        pitch_acceleration_radps2=scalar,
        roll_rate_radps=scalar,
        roll_acceleration_radps2=scalar,
        passenger_acceleration_mps2=torch.full((1, 2), value),
        suspension_travel_m=wheel * 0.01,
        tire_load_variation_n=wheel * 100.0,
        electrical_power_w=scalar * 1000.0,
        saturation=torch.zeros(1, 4),
        action_slew_normalized=scalar,
        force_tracking_error_n=wheel * 10.0,
        hard_violation=torch.zeros(1),
    )


def test_episode_metrics_and_go_no_go_are_reproducible():
    accumulator = EpisodeAccumulator(sample_time_s=0.01)
    for value in (1.0, 2.0, 3.0):
        accumulator.add(
            make_step(value),
            {
                "gate": torch.full((1, 4), 0.5),
                "preview_force_n": torch.full((1, 4), 100.0),
                "feedback_force_n": torch.full((1, 4), 200.0),
            },
        )
    summary = accumulator.summarize()

    assert summary.peak_vertical_acceleration == 3.0
    assert summary.mean_gate == 0.5
    assert summary.preview_to_feedback_ratio == 0.5

    feedback = summary.__dict__.copy()
    residual = summary.__dict__.copy()
    residual["rms_vertical_acceleration"] *= 0.9
    residual["max_suspension_travel"] *= 0.9
    report = deployment_go_no_go(
        feedback,
        residual,
        dropout_force_difference_n=0.0,
        policy_safety_passed=True,
        bandwidth=BandwidthReport(0.1, 50.0, True),
        config=SafetyConfig(),
    )
    assert report.approved
