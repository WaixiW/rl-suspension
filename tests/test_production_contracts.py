import numpy as np

from rl_suspension.production.adapters.mpc import CallableMpcAdapter
from rl_suspension.production.certification import certify_integration
from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    DEFAULT_OBSERVATION_SCHEMA,
    Scenario,
)
from rl_suspension.production.config import PipelineConfig
from rl_suspension.production.reference import (
    ReferenceDirect12Simulator,
    ReferenceMpcAdapter,
)
from rl_suspension.production.security import (
    ChainedAuditLog,
    redact,
    require_permission,
)


def test_action_normalization_projection_and_order():
    schema = DEFAULT_ACTION_SCHEMA
    action = np.asarray(schema.maximum, dtype=np.float64)
    normalized = schema.normalize(action)
    np.testing.assert_allclose(normalized, 1.0)
    np.testing.assert_allclose(schema.denormalize(normalized), action)

    projected = schema.project(
        action,
        np.asarray(schema.safe_action, dtype=np.float64),
        DEFAULT_OBSERVATION_SCHEMA.control_period_s,
    )
    expected = np.minimum(
        action,
        np.asarray(schema.slew_per_second)
        * DEFAULT_OBSERVATION_SCHEMA.control_period_s,
    )
    np.testing.assert_allclose(projected, expected)


def test_reference_adapters_pass_contract_certification():
    scenario = Scenario(
        scenario_id="contract",
        seed=7,
        split="validation",
        bump_family="single_bump",
        parameters={"episode_steps": 5},
    )
    report = certify_integration(
        ReferenceMpcAdapter(),
        ReferenceDirect12Simulator(),
        scenario,
    )

    assert report.passed
    assert report.maximum_replay_error == 0.0


def test_audit_chain_and_secret_redaction(tmp_path):
    audit = ChainedAuditLog(tmp_path / "audit.jsonl")
    audit.append("collect", {"token": "hidden", "episodes": 2}, actor="collector")
    audit.append("train", {"checkpoint": "model.pt"}, actor="trainer")

    assert audit.verify()
    assert redact({"password": "bad", "safe": 1}) == {
        "password": "[REDACTED]",
        "safe": 1,
    }


def test_callable_mpc_adapter_rejects_out_of_bounds_label():
    simulator = ReferenceDirect12Simulator()
    scenario = Scenario("bounds", 0, "train", "flat", {"episode_steps": 1})
    observation = simulator.reset(scenario, 0)

    adapter = CallableMpcAdapter(
        solve_fn=lambda obs, snapshot: {
            "action_12d": np.full(12, 1e6),
            "valid": True,
            "diagnostics": {"status": "optimal"},
        }
    )
    result = adapter.solve(observation, simulator.snapshot())

    assert not result.valid
    np.testing.assert_array_less(
        result.action_12d,
        np.asarray(DEFAULT_ACTION_SCHEMA.maximum) + 1e-9,
    )


def test_pipeline_configuration_round_trip(tmp_path):
    path = tmp_path / "pipeline.json"
    expected = PipelineConfig(mpc_plugin="private.mpc:create")
    expected.save(path)

    actual = PipelineConfig.load(path)

    assert actual.mpc_plugin == expected.mpc_plugin
    assert actual.training.dagger_betas == expected.training.dagger_betas


def test_production_role_permissions_are_least_privilege():
    require_permission("collector", "write_dataset")
    try:
        require_permission("collector", "promote_model")
    except PermissionError:
        pass
    else:
        raise AssertionError("collector unexpectedly promoted a model")
