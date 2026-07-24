import numpy as np

from rl_suspension.envs.observation import OBSERVATION_SPEC
from rl_suspension.imitation.experts import ExpertResult
from rl_suspension.imitation.policy import PolicyEnsemble, SafeDAggerPolicy


class ConstantPolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def predict(self, observation, deterministic=True):
        return self.action.copy(), None


class ConstantExpert:
    name = "constant"

    def predict(self, observation):
        return ExpertResult(action=np.ones(4, dtype=np.float32))


def test_ensemble_uncertainty_and_safe_dagger_gate():
    ensemble = PolicyEnsemble(
        [
            ConstantPolicy([0.0, 0.0, 0.0, 0.0]),
            ConstantPolicy([0.2, 0.0, 0.0, 0.0]),
            ConstantPolicy([-0.2, 0.0, 0.0, 0.0]),
        ]
    )
    prediction = ensemble.predict_with_uncertainty(
        np.zeros(OBSERVATION_SPEC.dimension)
    )
    gated = SafeDAggerPolicy(ensemble, ConstantExpert(), threshold=0.001)
    action, _ = gated.predict(np.zeros(OBSERVATION_SPEC.dimension))

    assert prediction.uncertainty > 0.0
    assert np.all(action == 1.0)
    assert gated.query_rate == 1.0
