from rl_suspension.baselines import PassivePolicy
from rl_suspension.evaluation.evaluate import evaluate_policy


def test_passive_policy_evaluation_runs_one_episode():
    metrics = evaluate_policy(PassivePolicy(), episodes=1, curriculum_stage=1, seed=0)

    assert metrics.rms_vertical_acceleration >= 0.0
    assert metrics.peak_vertical_acceleration >= 0.0
    assert metrics.max_suspension_travel >= 0.0
