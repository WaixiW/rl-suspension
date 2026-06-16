"""Inference wrapper for a trained SB3 policy.

Loads the PPO model and the VecNormalize statistics so the same observation
normalization used in training is applied at evaluation time. Exposes both the
raw action and the decoded (4,3) command so the evaluation harness can treat the
RL policy like any other controller.
"""

from __future__ import annotations

import pickle
from typing import Optional

import numpy as np

from .actuator import ForwardActuator
from .config import ActuatorParams


class RLPolicy:
    def __init__(self, model_path: str, vecnorm_path: Optional[str],
                 actuator: ActuatorParams, deterministic: bool = True):
        from stable_baselines3 import PPO

        self.model = PPO.load(model_path, device="cpu")
        self.actuator = actuator
        self.deterministic = deterministic
        self.obs_rms = None
        self.clip_obs = 10.0
        self.epsilon = 1e-8
        if vecnorm_path:
            with open(vecnorm_path, "rb") as f:
                vn = pickle.load(f)
            self.obs_rms = vn.obs_rms
            self.clip_obs = vn.clip_obs
            self.epsilon = vn.epsilon

    def reset(self):
        pass

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms is None:
            return obs
        norm = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        return np.clip(norm, -self.clip_obs, self.clip_obs)

    def act_action(self, obs: np.ndarray) -> np.ndarray:
        o = self._normalize(np.asarray(obs, float)).astype(np.float32)
        action, _ = self.model.predict(o, deterministic=self.deterministic)
        return action

    def act_obs(self, obs: np.ndarray) -> np.ndarray:
        """Return a (4,3) command from a raw observation vector."""
        a = self.act_action(obs)
        return ForwardActuator.normalize_action(a, self.actuator)
