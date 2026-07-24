from rl_suspension.baselines import PassivePolicy
from rl_suspension.controllers.mpc import PreviewMPC, PreviewMpcConfig
from rl_suspension.evaluation.evaluate import evaluate_policy_detailed
from rl_suspension.imitation.mpc_expert import MpcExpert, MpcPolicy


def test_tuned_mpc_improves_comfort_without_worsening_safety_gate():
    passive = evaluate_policy_detailed(
        PassivePolicy(),
        episodes=1,
        curriculum_stage=1,
        seed=123,
        terminate_on_violation=False,
    )
    mpc = evaluate_policy_detailed(
        MpcPolicy(MpcExpert(PreviewMPC(PreviewMpcConfig(horizon=30)))),
        episodes=1,
        curriculum_stage=1,
        seed=123,
        terminate_on_violation=False,
    )

    assert (
        mpc["metrics"]["rms_vertical_acceleration"]
        < 0.99 * passive["metrics"]["rms_vertical_acceleration"]
    )
    assert (
        mpc["metrics"]["constraint_violations"]
        <= passive["metrics"]["constraint_violations"] + 1e-8
    )
    assert mpc["mean_episode_return"] > passive["mean_episode_return"]
    assert mpc["solver_quality"]["fallback_rate"] == 0.0
