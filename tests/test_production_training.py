import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    DEFAULT_OBSERVATION_SCHEMA,
    Scenario,
)
from rl_suspension.production.models.student import (
    Direct12Student,
    PhysicalActionExportWrapper,
    StudentConfig,
)
from rl_suspension.production.reference import (
    ReferenceDirect12Simulator,
    ReferenceMpcAdapter,
)
from rl_suspension.production.training.bc import (
    BCDatasetArrays,
    BCTrainer,
    BCTrainerConfig,
    default_bc_loss,
)
from rl_suspension.production.training.dagger import (
    DEFAULT_BETA_SCHEDULE,
    DaggerConfig,
    DaggerTrainer,
)
from rl_suspension.production.training.finetune import (
    BCRegularizationConfig,
    DecayingBCRegularizer,
    FineTuneConfig,
    FineTuneOrchestrator,
    PhysicalSafetyFilter,
    intervention_counter,
)
from rl_suspension.production.training.losses import (
    contiguous_action_delta_loss,
    normalized_weighted_huber_loss,
    phase_quality_weights,
)


def _arrays(count=8, *, normalized=True):
    schema = DEFAULT_OBSERVATION_SCHEMA
    rng = np.random.default_rng(3)
    actions = rng.uniform(0.1, 0.9, size=(count, 12)).astype(np.float32)
    if not normalized:
        low = np.asarray(DEFAULT_ACTION_SCHEMA.minimum, dtype=np.float32)
        high = np.asarray(DEFAULT_ACTION_SCHEMA.maximum, dtype=np.float32)
        actions = low + actions * (high - low)
    return BCDatasetArrays(
        states=rng.normal(size=(count, schema.state_vector_dim)),
        roads=rng.normal(size=(count, schema.road_channels, schema.road_points)),
        actions=actions,
        phases=np.arange(count) % 5,
        quality=np.linspace(0.5, 1.5, count),
        episode_ids=np.repeat(np.arange((count + 3) // 4), 4)[:count],
        sequence_indices=np.tile(np.arange(4), (count + 3) // 4)[:count],
        actions_normalized=normalized,
    )


def test_direct12_student_uses_all_road_channels_and_optional_history():
    torch.manual_seed(4)
    schema = DEFAULT_OBSERVATION_SCHEMA
    config = StudentConfig(
        state_feature_dim=24,
        road_feature_dim=16,
        fusion_dim=32,
        residual_blocks=1,
    )
    model = Direct12Student(config)
    states = torch.randn(2, schema.state_vector_dim)
    roads = torch.randn(2, schema.road_channels, schema.road_points)
    actions = model(states, roads)

    assert actions.shape == (2, 12)
    assert torch.all((actions >= 0.0) & (actions <= 1.0))
    actions.sum().backward()
    first_convolution = model.road_encoder.network[0]
    channel_gradients = first_convolution.weight.grad.abs().sum(dim=(0, 2))
    assert torch.all(channel_gradients > 0.0)
    physical_actions = PhysicalActionExportWrapper(model)(states, roads)
    assert physical_actions.shape == (2, 12)
    assert torch.all(
        physical_actions
        <= torch.as_tensor(DEFAULT_ACTION_SCHEMA.maximum) + 1e-6
    )

    temporal = Direct12Student(
        StudentConfig(
            state_feature_dim=24,
            road_feature_dim=16,
            fusion_dim=32,
            residual_blocks=1,
            use_gru=True,
            temporal_history_steps=3,
        )
    )
    temporal_actions = temporal(
        states[:, None].repeat(1, 3, 1),
        roads[:, None].repeat(1, 3, 1, 1),
    )
    assert temporal_actions.shape == (2, 12)


def test_normalized_weighted_losses_and_contiguous_masking():
    schema = DEFAULT_ACTION_SCHEMA
    target_physical = torch.as_tensor(
        np.repeat(np.asarray(schema.maximum)[None], 3, axis=0),
        dtype=torch.float32,
    )
    perfect = torch.ones(3, 12)
    assert normalized_weighted_huber_loss(perfect, target_physical).item() == 0.0
    assert normalized_weighted_huber_loss(
        torch.zeros_like(perfect),
        target_physical,
        channel_weights=[2.0] + [1.0] * 11,
    ).item() > 0.0

    weights = phase_quality_weights(
        torch.tensor([0, 1, 1]),
        torch.tensor([1.0, 1.0, 2.0]),
        phase_weights=(0.5, 2.0),
    )
    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[2] > weights[1] > weights[0]

    predicted = torch.tensor([[0.0] * 12, [1.0] * 12, [0.0] * 12])
    target = torch.zeros_like(predicted)
    ignored = contiguous_action_delta_loss(
        predicted,
        target,
        torch.tensor([0, 1, 1]),
        sequence_indices=torch.tensor([0, 4, 6]),
        target_is_normalized=True,
    )
    assert ignored.item() == 0.0
    included = contiguous_action_delta_loss(
        predicted,
        target,
        torch.tensor([0, 0, 1]),
        sequence_indices=torch.tensor([0, 1, 0]),
        target_is_normalized=True,
    )
    assert included.item() > 0.0


class _TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        state_dim = DEFAULT_OBSERVATION_SCHEMA.state_vector_dim
        self.linear = nn.Linear(state_dim + 4, 12)

    def forward(self, states, roads):
        road_summary = roads.mean(dim=-1)
        return torch.sigmoid(self.linear(torch.cat((states, road_summary), dim=-1)))


def test_framework_neutral_bc_trainer_and_checkpoint_hook():
    training = _arrays()
    selected = []
    trainer = BCTrainer(
        _TinyStudent(),
        config=BCTrainerConfig(epochs=2, batch_size=4, learning_rate=1e-3),
        loss=default_bc_loss(target_is_normalized=True),
    )
    result = trainer.fit(
        training,
        training.fixed_copy(),
        checkpoint_hooks=(selected.append,),
    )

    assert len(result.history) == 2
    assert 1 <= result.best_epoch <= 2
    assert selected
    assert np.isfinite(result.best_validation_loss)


class _ScoredPolicy:
    name = "scored"

    def __init__(self, score):
        self.score = score

    def predict(self, observation):
        del observation
        return np.asarray(DEFAULT_ACTION_SCHEMA.safe_action, dtype=np.float64)


def test_five_round_dagger_rejects_fixed_validation_regression():
    initial = _arrays(2, normalized=False)
    validation = _arrays(2, normalized=False)
    candidate_scores = iter((0.8, 1.2, 0.7, 0.6, 0.5))

    def train_round(aggregate, fixed_validation, round_index):
        del aggregate, fixed_validation, round_index
        return _ScoredPolicy(next(candidate_scores))

    trainer = DaggerTrainer(
        simulator_factory=ReferenceDirect12Simulator,
        expert=ReferenceMpcAdapter(),
        initial_policy=_ScoredPolicy(1.0),
        train_round=train_round,
        validation_evaluator=lambda policy, dataset: policy.score,
        config=DaggerConfig(
            episodes_per_round=1,
            asynchronous_queries=True,
            maximum_steps_per_episode=4,
            seed=5,
        ),
    )
    scenarios = [
        [
            Scenario(
                scenario_id=f"round-{index}",
                seed=100 + index,
                split="train",
                bump_family="single_bump",
                parameters={"episode_steps": 2},
            )
        ]
        for index in range(5)
    ]
    result = trainer.run(initial, validation, scenarios)

    assert tuple(report.beta for report in result.reports) == DEFAULT_BETA_SCHEDULE
    assert len(result.reports) == 5
    assert not result.reports[1].accepted
    assert result.best_validation_score == pytest.approx(0.5)
    assert len(result.aggregate) == len(initial) + 10


class _FakeRLAdapter:
    def __init__(self):
        self.initialized = False
        self.arguments = None

    def initialize_actor(self, student):
        self.initialized = isinstance(student, nn.Module)

    def fine_tune(self, **kwargs):
        self.arguments = kwargs
        return {"steps": kwargs["total_steps"]}


def test_decaying_bc_regularization_and_safe_finetune_hooks():
    regularizer = DecayingBCRegularizer(
        BCRegularizationConfig(
            initial_coefficient=1.0,
            final_coefficient=0.0,
            decay_steps=100,
        )
    )
    assert regularizer.coefficient(0) == pytest.approx(1.0)
    assert regularizer.coefficient(50) == pytest.approx(0.5)
    assert regularizer.coefficient(100) == pytest.approx(0.0)

    callback, count = intervention_counter()
    safety_filter = PhysicalSafetyFilter(callbacks=(callback,))
    projected = safety_filter.project(
        np.asarray(DEFAULT_ACTION_SCHEMA.maximum) * 2.0,
        np.asarray(DEFAULT_ACTION_SCHEMA.safe_action),
        step=0,
    )
    assert count() == 1
    assert np.all(projected <= np.asarray(DEFAULT_ACTION_SCHEMA.maximum))

    adapter = _FakeRLAdapter()
    orchestrator = FineTuneOrchestrator(
        adapter,
        config=FineTuneConfig(total_steps=12),
        regularizer=regularizer,
        safety_filter=safety_filter,
    )
    result = orchestrator.run(_TinyStudent(), _arrays())
    assert adapter.initialized
    assert result.fine_tuned
    assert result.adapter_result == {"steps": 12}
