import numpy as np

from rl_suspension.envs import ActiveSuspensionEnv, EnvConfig


def test_environment_reset_and_step_shapes():
    env = ActiveSuspensionEnv(EnvConfig())

    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert "scenario" in info

    action = np.zeros(4, dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "rms_body_acceleration" in step_info
    assert "constraint_violations" in step_info


def test_imitation_mode_records_violations_without_early_termination():
    env = ActiveSuspensionEnv(
        EnvConfig(
            curriculum_stage=1,
            randomize_scenario=False,
            terminate_on_violation=False,
        )
    )
    observation, _ = env.reset(seed=0)
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        observation, _, terminated, truncated, _ = env.step(
            np.zeros(4, dtype=np.float32)
        )
        steps += 1

    assert not terminated
    assert truncated
    assert steps == int(env.road.config.episode_time / env.config.dt)
    assert np.all(np.isfinite(observation))
