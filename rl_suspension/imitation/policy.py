"""Inference wrappers for behavior-cloned SAC actors and ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl_suspension.imitation.dataset import NormalizationStats, load_normalization


@dataclass
class BehaviorClonedPolicy:
    """Apply saved observation normalization before SAC actor inference."""

    model: object
    normalization: NormalizationStats

    @classmethod
    def load(cls, model_path: str | Path, normalization_path: str | Path):
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise RuntimeError("stable-baselines3 is required to load a BC policy") from exc
        model = SAC.load(model_path)
        return cls(model=model, normalization=load_normalization(normalization_path))

    def predict(self, observation, deterministic: bool = True):
        obs = np.asarray(observation, dtype=np.float32)
        normalized = self.normalization.normalize(obs)
        action, state = self.model.predict(normalized, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32), state


@dataclass
class EnsemblePrediction:
    action: np.ndarray
    uncertainty: float
    member_actions: np.ndarray


class PolicyEnsemble:
    """Mean action and epistemic uncertainty from independently trained actors."""

    def __init__(self, policies: list[BehaviorClonedPolicy]) -> None:
        if not policies:
            raise ValueError("Policy ensemble cannot be empty")
        self.policies = policies

    def predict_with_uncertainty(self, observation) -> EnsemblePrediction:
        actions = np.stack(
            [
                policy.predict(observation, deterministic=True)[0]
                for policy in self.policies
            ],
            axis=0,
        ).astype(np.float32)
        mean_action = actions.mean(axis=0)
        uncertainty = float(np.mean(np.var(actions, axis=0)))
        return EnsemblePrediction(
            action=mean_action.astype(np.float32),
            uncertainty=uncertainty,
            member_actions=actions,
        )

    def predict(self, observation, deterministic: bool = True):
        prediction = self.predict_with_uncertainty(observation)
        return prediction.action, None


class SafeDAggerPolicy:
    """Use an expert only when ensemble disagreement exceeds a threshold."""

    def __init__(self, ensemble: PolicyEnsemble, expert, threshold: float) -> None:
        self.ensemble = ensemble
        self.expert = expert
        self.threshold = float(threshold)
        self.queries = 0
        self.predictions = 0

    @property
    def query_rate(self) -> float:
        return self.queries / max(self.predictions, 1)

    def reset_statistics(self) -> None:
        self.queries = 0
        self.predictions = 0

    def predict(self, observation, deterministic: bool = True):
        self.predictions += 1
        prediction = self.ensemble.predict_with_uncertainty(observation)
        if prediction.uncertainty >= self.threshold:
            label = self.expert.predict(observation)
            if label.valid:
                self.queries += 1
                return label.action, None
        return prediction.action, None
