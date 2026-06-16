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
