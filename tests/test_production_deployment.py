from __future__ import annotations

from dataclasses import replace
import hashlib
import hmac

import numpy as np
import pytest

from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    Scenario,
)
from rl_suspension.production.deployment import (
    EngineeringEnvelope,
    GoldenVectorSet,
    ModelRegistry,
    QuantizationHookResult,
    SafetySupervisor,
    ShadowOrchestrator,
    SupervisorConfig,
    benchmark_runtime,
    build_manifest,
    generate_golden_vectors,
    validate_quantized_model,
    verify_golden_vectors,
    verify_manifest,
)
from rl_suspension.production.deployment.export import (
    default_export_inputs,
    export_fixed_shape_onnx,
    onnx_dependencies_available,
)
from rl_suspension.production.deployment.golden import (
    verify_onnx_export,
)
from rl_suspension.production.evaluation import (
    PromotionEvidence,
    PromotionGateConfig,
    PairedClosedLoopEvaluator,
    bootstrap_confidence_interval,
    evaluate_promotion_gates,
    open_loop_metrics,
)
from rl_suspension.production.reference import (
    ReferenceDirect12Simulator,
    ReferenceMpcAdapter,
)
from rl_suspension.production.models import (
    Direct12Student,
    PhysicalActionExportWrapper,
    StudentConfig,
)


class ConstantPolicy:
    def __init__(self, name: str, action: np.ndarray) -> None:
        self.name = name
        self.action = np.asarray(action, dtype=np.float64)

    def predict(self, observation):
        del observation
        return self.action.copy()


class HmacSigner:
    algorithm = "HMAC-SHA256"
    key_id = "test-key"

    def __init__(self, key: bytes) -> None:
        self.key = key

    def sign(self, payload: bytes) -> bytes:
        return hmac.new(self.key, payload, hashlib.sha256).digest()

    def verify(self, payload: bytes, signature: bytes) -> bool:
        return hmac.compare_digest(self.sign(payload), signature)


def _safe_policy(name: str = "student") -> ConstantPolicy:
    return ConstantPolicy(name, np.asarray(DEFAULT_ACTION_SCHEMA.safe_action))


def _observation(timestamp_ns: int = 0):
    scenario = Scenario("runtime", 7, "test", "flat", {"episode_steps": 2})
    observation = ReferenceDirect12Simulator().reset(scenario, scenario.seed)
    return replace(observation, timestamp_ns=timestamp_ns)


def test_bootstrap_and_open_loop_channel_delta_metrics():
    interval = bootstrap_confidence_interval([1.0, 2.0, 3.0], resamples=100, seed=4)
    assert interval.estimate == 2.0
    assert interval.lower <= interval.estimate <= interval.upper

    expert = np.zeros((3, 2))
    student = np.asarray([[0.0, 0.0], [1.0, -1.0], [1.0, -2.0]])
    report = open_loop_metrics(expert, student, channel_names=("left", "right"))
    assert report.per_channel["left"].mae == 2.0 / 3.0
    assert report.per_channel["right"].delta_maximum_absolute_error == 1.0


def test_paired_closed_loop_evaluates_all_three_controllers():
    scenarios = [
        Scenario(
            f"scenario-{seed}",
            seed,
            "test",
            "single_bump",
            {"episode_steps": 4, "speed_mps": 10.0},
        )
        for seed in (1, 2)
    ]
    evaluator = PairedClosedLoopEvaluator(
        ReferenceDirect12Simulator,
        mpc=ReferenceMpcAdapter(),
        student=_safe_policy(),
        bootstrap_resamples=50,
    )
    report = evaluator.evaluate(scenarios)
    assert set(report.episodes) == {"constant_safe", "reference_mpc", "student"}
    assert all(len(episodes) == 2 for episodes in report.episodes.values())
    assert "student_minus_reference_mpc" in report.paired_differences


def test_promotion_gates_include_retention_safety_actions_and_latency():
    evidence = PromotionEvidence(
        passive_metrics={
            "rms_body_acceleration": 10.0,
            "constraint_violation_total": 0.0,
        },
        mpc_metrics={
            "rms_body_acceleration": 6.0,
            "constraint_violation_total": 0.0,
        },
        student_metrics={
            "rms_body_acceleration": 6.5,
            "constraint_violation_total": 0.0,
            "maximum_suspension_travel": 0.05,
            "minimum_tire_load": 500.0,
        },
        student_p99_latency_ms=2.0,
        action_bounds_violations=0,
        action_slew_violations=0,
    )
    decision = evaluate_promotion_gates(
        evidence,
        PromotionGateConfig(minimum_mpc_improvement_retention=0.80),
    )
    assert decision.passed
    assert decision.mpc_improvement_retention == 0.875

    failed = evaluate_promotion_gates(
        replace(evidence, student_p99_latency_ms=11.0),
        PromotionGateConfig(maximum_p99_latency_ms=10.0),
    )
    assert not failed.passed
    assert not failed.checks["p99_latency"].passed


def test_golden_vectors_and_signed_manifest_without_onnx(tmp_path):
    inputs = {
        "state": np.arange(6, dtype=np.float32).reshape(3, 2),
        "road": np.ones((3, 1), dtype=np.float32),
    }

    def predictor(state, road):
        return state[:, :1] + road

    golden = generate_golden_vectors(predictor, inputs)
    path = golden.save(tmp_path / "golden.npz")
    loaded = GoldenVectorSet.load(path)
    assert verify_golden_vectors(predictor, loaded).passed

    signer = HmacSigner(b"unit-test-key")
    manifest = build_manifest(path, signer=signer, created_utc="2026-01-01T00:00:00Z")
    assert manifest.signature is not None
    assert verify_manifest(
        manifest,
        path,
        verifier=signer,
        require_signature=True,
    ).passed
    path.write_bytes(path.read_bytes() + b"tampered")
    assert not verify_manifest(
        manifest,
        path,
        verifier=signer,
        require_signature=True,
    ).passed


def test_fixed_shape_physical_action_onnx_export(tmp_path):
    if not onnx_dependencies_available(runtime=True):
        pytest.skip("optional ONNX dependencies are unavailable")
    import torch

    model = PhysicalActionExportWrapper(
        Direct12Student(
            StudentConfig(
                state_feature_dim=24,
                road_feature_dim=16,
                fusion_dim=32,
                residual_blocks=1,
            )
        )
    ).eval()
    inputs = default_export_inputs()
    target = tmp_path / "student.onnx"
    result = export_fixed_shape_onnx(model, target, inputs)

    def framework_predictor(state, road):
        with torch.no_grad():
            return model(
                torch.as_tensor(state),
                torch.as_tensor(road),
            ).numpy()

    golden = generate_golden_vectors(framework_predictor, inputs)
    verification = verify_onnx_export(result.path, golden)

    assert result.output_shapes["action"] == (1, 12)
    assert verification.passed


def test_runtime_benchmark_and_quantization_hooks_without_onnx():
    inputs = {"value": np.ones((2, 1), dtype=np.float32)}
    clock_values = iter([0, 1_000_000, 1_000_000, 3_000_000, 3_000_000, 6_000_000])
    benchmark = benchmark_runtime(
        lambda value: value,
        inputs,
        warmup_iterations=1,
        measured_iterations=3,
        deadline_ms=2.0,
        clock_ns=lambda: next(clock_values),
    )
    assert benchmark.p50_latency_ms == 2.0
    assert benchmark.deadline_misses == 1
    assert not benchmark.deadline_met

    def hook(reference, quantized):
        error = float(
            np.max(np.abs(reference.outputs["action"] - quantized.outputs["action"]))
        )
        return QuantizationHookResult(
            name="application_tolerance",
            passed=error < 0.01,
            metrics={"maximum_error": error},
        )

    report = validate_quantized_model(
        lambda value: value,
        lambda value: value + 1e-4,
        inputs,
        absolute_tolerance=1e-3,
        hooks=[hook],
    )
    assert report.passed
    assert report.hooks[0].name == "application_tolerance"


def test_supervisor_projects_and_recovers_with_hysteresis():
    safe = np.asarray(DEFAULT_ACTION_SCHEMA.safe_action)
    maximum = np.asarray(DEFAULT_ACTION_SCHEMA.maximum)
    supervisor = SafetySupervisor(
        _safe_policy("primary"),
        _safe_policy("fallback"),
        config=SupervisorConfig(
            faults_to_latch_fallback=1,
            minimum_fallback_cycles=2,
            healthy_cycles_to_recover=2,
            ring_buffer_capacity=3,
        ),
        envelope=EngineeringEnvelope(),
    )
    invalid = replace(
        _observation(10_000_000),
        vehicle_state=np.full(14, np.nan),
    )
    first = supervisor.decide(invalid, now_ns=10_000_000)
    assert first.source == "fallback"
    assert supervisor.fallback_active

    valid = _observation(20_000_000)
    assert supervisor.decide(valid, now_ns=20_000_000).source == "fallback"
    recovered = supervisor.decide(valid, now_ns=20_000_000)
    assert recovered.source == "primary"
    assert not supervisor.fallback_active

    projecting = SafetySupervisor(
        ConstantPolicy("primary", maximum * 2.0),
        _safe_policy("fallback"),
        config=SupervisorConfig(enforce_freshness=False),
    )
    decision = projecting.decide(_observation())
    expected = np.minimum(
        maximum,
        np.asarray(DEFAULT_ACTION_SCHEMA.slew_per_second) * 0.01,
    )
    np.testing.assert_allclose(decision.action, expected)
    assert decision.projected


def test_model_registry_checksum_and_rollback(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"model-one")
    second.write_bytes(b"model-two")
    registry = ModelRegistry(tmp_path / "registry")
    registry.register("1.0", first, activate=True)
    registry.register("2.0", second, activate=True)
    assert registry.active_version == "2.0"
    assert registry.resolve().read_bytes() == b"model-two"
    registry.rollback()
    assert registry.active_version == "1.0"
    assert registry.resolve().read_bytes() == b"model-one"


def test_shadow_candidate_is_never_applied():
    safe = np.asarray(DEFAULT_ACTION_SCHEMA.safe_action)
    candidate = safe.copy()
    candidate[0] = 0.5
    report = ShadowOrchestrator(
        _safe_policy("production"),
        ConstantPolicy("candidate", candidate),
    ).run([_observation()])
    assert report.samples == 1
    assert report.maximum_absolute_delta == 0.5
    assert report.candidate_never_applied
