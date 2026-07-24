import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import kairyu.evaluation.adapters.humanitys_last_exam as hle_module
from kairyu.evaluation.adapters.base import (
    ItemResult,
    ModelConnectorSet,
    ModelRole,
    RunSelection,
)
from kairyu.evaluation.adapters.humanitys_last_exam import HumanitysLastExamAdapter
from kairyu.evaluation.connectors import (
    ConnectorError,
    ConnectorErrorCode,
    ConnectorImagePart,
    ConnectorResponse,
    ConnectorResult,
    ConnectorTextPart,
    ConnectorUsage,
    FakeOpenAIConnector,
    canonical_connector_request_sha256,
)
from kairyu.evaluation.guards import RunGuardError
from kairyu.evaluation.profiles import get_profile_lock, load_profile_resource
from kairyu.evaluation.schemas import RunMode

ROOT = Path(__file__).parents[3]
FIXTURE = (
    ROOT / "kairyu" / "evaluation" / "resources" / "fixtures" / "humanitys-last-exam-smoke.jsonl"
)
FIXTURE_SHA256 = "0f85f6fffc09191181b42ac327a66ed97746f661696d1d24798a8f1b396fb194"
COMPATIBILITY_SHA256 = "37671a5fe9bbc2ed0676a18164797f74f7fad67045e0a2679f72491d3a2f66cb"
HARNESS_COMMIT = "8e53435ff2985b0f32ea7ceb7e92c3a175f2c0f3"
OFFICIAL_REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"


def _selection(**changes):
    values = {
        "profile": "smoke",
        "mode": RunMode.SMOKE,
        "target_model": "offline-target",
    }
    values.update(changes)
    return RunSelection(**values)


def _success(
    request_id,
    content,
    *,
    model,
    attempts=1,
    latency_seconds=0.1,
    prompt_tokens=20,
    completion_tokens=5,
    provider_request_id=None,
):
    return ConnectorResult(
        response=ConnectorResponse(
            request_id=request_id,
            content=content,
            finish_reason="stop",
            provider_request_id=(provider_request_id or f"provider-{request_id}"),
            provider_model=model,
            usage=ConnectorUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_seconds=latency_seconds,
            attempts=attempts,
        )
    )


def _error(
    request_id,
    code=ConnectorErrorCode.TIMEOUT,
    *,
    attempts=1,
    provider_request_id=None,
    latency_seconds=0.0,
):
    return ConnectorResult(
        error=ConnectorError(
            request_id=request_id,
            code=code,
            detail="synthetic connector error",
            retryable=True,
            attempts=attempts,
            provider_request_id=provider_request_id,
            latency_seconds=latency_seconds,
        )
    )


def _judge_response_with_confidence(confidence: str) -> str:
    return (
        "<extracted_final_answer>7</extracted_final_answer>"
        "<reasoning>synthetic</reasoning><correct>yes</correct>"
        f"<confidence>{confidence}</confidence>"
    )


class _SequenceConnector:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.requests = []
        self.max_attempts = []

    def complete(
        self,
        request,
        *,
        cancel_requested=None,
        max_attempts=None,
    ):
        assert max_attempts is not None and max_attempts >= 1
        self.max_attempts.append(max_attempts)
        if cancel_requested is not None and cancel_requested():
            return _error(
                request.request_id,
                ConnectorErrorCode.CANCELLED,
                attempts=0,
            )
        self.requests.append(request)
        if not self._outcomes:
            raise AssertionError("sequence connector exhausted")
        outcome = self._outcomes.pop(0)
        result = outcome.response if outcome.response is not None else outcome.error
        assert result is not None and result.attempts <= max_attempts
        return outcome


def _derived_snapshot(tmp_path, *, count=2500, first_image=None):
    source_rows = [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for ordinal in range(count):
        row = dict(source_rows[ordinal % len(source_rows)])
        row["id"] = f"synthetic-derived-{ordinal:04d}"
        row["image"] = first_image if ordinal == 0 else None
        rows.append(row)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    path = tmp_path / "approved-synthetic-hle.jsonl"
    path.write_text(payload, encoding="utf-8")
    return path, hashlib.sha256(payload.encode()).hexdigest()


def test_metadata_and_profile_locks_pin_current_official_sources():
    adapter = HumanitysLastExamAdapter()
    metadata = adapter.metadata()
    resource = load_profile_resource("humanitys-last-exam")

    assert metadata.implementation_status.value == "available"
    assert metadata.primary_metric == "Accuracy"
    assert metadata.modalities == ("text", "image")
    assert metadata.required_capabilities == (
        "chat completions",
        "inline image input",
        "separate judge connector",
    )
    assert tuple(profile.name for profile in resource.profiles) == (
        "smoke",
        "fugu-2026",
        "official-latest",
    )
    assert all(profile.expected_full_count == 2500 for profile in resource.profiles)
    assert all(profile.harness_commit == HARNESS_COMMIT for profile in resource.profiles)
    assert all(
        profile.compatibility_layer_sha256 == COMPATIBILITY_SHA256 for profile in resource.profiles
    )
    assert (
        hashlib.sha256(
            (ROOT / "kairyu" / "evaluation" / "adapters" / "hle_official_2026.py").read_bytes()
        ).hexdigest()
        == COMPATIBILITY_SHA256
    )
    assert get_profile_lock("humanitys-last-exam", "smoke").dataset_sha256 == (FIXTURE_SHA256)
    assert (
        get_profile_lock("humanitys-last-exam", "official-latest").dataset_revision
        == OFFICIAL_REVISION
    )
    assert get_profile_lock("humanitys-last-exam", "official-latest").unresolved_fields == (
        "generation_parameters",
    )
    assert set(get_profile_lock("humanitys-last-exam", "fugu-2026").unresolved_fields) == {
        "dataset_revision",
        "generation_parameters",
        "harness_commit",
        "judge_model",
    }

    for profile in resource.profiles:
        protocol = profile.to_profile("humanitys-last-exam").protocol
        assert protocol.benchmark_version.startswith("humanitys-last-exam-")
        assert protocol.modalities == ("text", "image")
        assert protocol.tools == ()
        assert protocol.web_access is False
        assert protocol.retries == 4
        assert protocol.judge_prompt_version == "cais-simple-evals-hle-judge-2026.07"
        assert protocol.adapter_configuration["upstream_source_sha256"] == (
            "d276f725ecc5ea2c08f73e161f97760881332c3d33d20b197c2ffbac5f55edfe"
        )
        assert protocol.adapter_configuration["confidence_range_patch"] == ("reject-outside-0-100")
        assert protocol.adapter_configuration["judge_parse_exception_policy"] == (
            "retry-as-judge-parse-error"
        )
        assert protocol.adapter_configuration["request_construction_error_policy"] == (
            "fail-item-attempt-zero-v1"
        )
        assert protocol.dependency_compatibility_patches == (
            "kairyu.evaluation.adapters.hle_official_2026@sha256:" + COMPATIBILITY_SHA256,
        )


def test_smoke_doctor_reports_hash_failure_without_hiding_other_checks(monkeypatch):
    adapter = HumanitysLastExamAdapter()

    def rejected_hash(_path):
        raise ValueError("O_NOFOLLOW is unavailable")

    monkeypatch.setattr(hle_module, "sha256_file", rejected_hash)
    report = adapter.doctor("smoke")
    checks = {check.check_id: check for check in report.checks}

    assert report.runnable is False
    assert checks["compatibility-layer"].status.value == "fail"
    assert checks["synthetic-fixture"].status.value == "fail"
    assert checks["python"].status.value == "pass"
    assert checks["target-capability"].status.value == "warn"
    assert checks["protocol-completeness"].status.value == "pass"


def test_smoke_doctor_prepare_and_plan_are_offline_and_golden():
    adapter = HumanitysLastExamAdapter()

    doctor = adapter.doctor("smoke")
    dry_run = adapter.prepare("smoke", dry_run=True)
    prepared = adapter.prepare("smoke", dry_run=False)
    plan = adapter.build_run_plan(_selection(), environ={})

    assert doctor.runnable
    checks = {check.check_id: check for check in doctor.checks}
    assert set(checks) == {
        "approved-local-dataset",
        "compatibility-layer",
        "cpu",
        "disk",
        "docker",
        "harness-pin",
        "judge-capability",
        "memory",
        "protocol-completeness",
        "python",
        "synthetic-fixture",
        "target-capability",
    } - {"approved-local-dataset"}
    assert checks["target-capability"].status.value == "warn"
    assert checks["judge-capability"].status.value == "warn"
    assert dry_run.item_count == 2 and dry_run.dry_run
    assert prepared.item_count == 2
    assert prepared.dataset_sha256 == FIXTURE_SHA256
    assert plan.official_eligible is False
    assert plan.expected_full_count == 2500
    assert plan.estimated_model_calls == 4
    assert plan.estimate.model_calls == 4
    assert plan.estimate.maximum_model_calls == 20
    assert plan.estimate.estimated_input_tokens is None
    assert plan.estimate.maximum_output_tokens == 61_440
    assert plan.estimate.maximum_duration_seconds == 12_000.0
    assert plan.resources.cpu_cores == 2
    assert plan.resources.ram_bytes == 2 * 1024 * 1024 * 1024
    assert plan.resources.disk_bytes >= 1024 * 1024 * 1024
    assert plan.resources.docker_required is False
    assert plan.execution.command == ("kairyu", "benchmark", "worker", "--once")
    assert plan.selection.judge_model == "gpt-5-mini"
    assert plan.selection.generation_parameters == {
        "max_tokens": 4096,
        "repeats": 1,
        "seed": 42,
        "temperature": 0.0,
        "timeout_seconds": 600.0,
        "top_p": 1.0,
    }
    assert plan.selection.judge_generation_parameters == {
        "max_tokens": 2048,
        "seed": 42,
        "temperature": 0.0,
        "timeout_seconds": 600.0,
        "top_p": 1.0,
    }
    assert plan.item_input_manifest_sha256 == (
        "dea3843aded911fc16c141ea57da35183ed64ff3cbcfa30a6b20048d6ad36064"
    )
    assert plan.protocol_hash == (
        "f5756dcf0093d10e08341280a08a39d1556747cb7b7186c405a096332eac5f20"
    )
    assert [(item.item_id, item.image_data_uri is not None) for item in plan.items] == [
        ("hle-16c93a1db82355a2885bce2c222bb8ea", False),
        ("hle-fd9c0364fb1f01e34d213367daf3de06", True),
    ]
    assert [
        (role.role, role.model_calls, role.maximum_output_tokens) for role in plan.model_roles
    ] == [
        (ModelRole.TARGET, 2, 40_960),
        (ModelRole.JUDGE, 2, 20_480),
    ]
    assert plan.protocol.adapter_configuration["retry_budget_scope"] == (
        "per-model-request-total-attempts"
    )
    assert plan.protocol.adapter_configuration["max_attempts_per_request"] == 5
    assert plan.protocol.adapter_configuration["model_request_policy"] == (
        "target-and-judge-max-5-attempts-v1"
    )
    evidence = plan.protocol.adapter_configuration["reproducibility_evidence"]
    assert {item["status"] for item in evidence.values()} == {"unresolved"}
    assert plan.protocol.adapter_configuration["reproducibility_evidence_complete"] is False


def test_smoke_target_then_judge_multimodal_flow_and_metrics():
    adapter = HumanitysLastExamAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    target = FakeOpenAIConnector(adapter.smoke_connector_results(plan, ModelRole.TARGET))
    judge = FakeOpenAIConnector(adapter.smoke_connector_results(plan, ModelRole.JUDGE))
    connectors = ModelConnectorSet(target=target, judge=judge)

    results = tuple(
        adapter.run(plan, item, connectors, cancel_check=lambda: False) for item in plan.items
    )
    collected = adapter.collect("run-hle-smoke", plan, results)

    assert [result.correct for result in results] == [True, False]
    assert [result.confidence for result in results] == [0.8, 0.6]
    assert [result.extracted_answer for result in results] == ["7", "red"]
    assert all(result.error_class is None for result in results)
    assert all(result.judge_response_text for result in results)
    assert len(target.requests) == 2
    assert len(judge.requests) == 2
    first_user = target.requests[0].messages[1]
    second_user = target.requests[1].messages[1]
    assert not isinstance(first_user.content, str)
    assert [type(part) for part in first_user.content] == [ConnectorTextPart]
    assert not isinstance(second_user.content, str)
    assert [type(part) for part in second_user.content] == [
        ConnectorTextPart,
        ConnectorImagePart,
    ]
    assert target.requests[0].messages[0].role == "system"
    assert "Confidence:" in target.requests[0].messages[0].content
    assert "[correct_answer]: 7" in judge.requests[0].messages[0].content
    assert "[correct_answer]: blue" in judge.requests[1].messages[0].content

    metrics = {metric.name: metric for metric in collected.metrics}
    assert metrics["accuracy"].value == 50.0
    assert metrics["accuracy"].numerator == 1
    assert metrics["accuracy"].denominator == 2
    assert metrics["accuracy"].primary is True
    assert metrics["calibration-error"].value == 20.0
    assert metrics["accuracy-success-only"].value == 50.0
    assert metrics["confidence-interval"].value == 69.3
    assert metrics["confidence-interval-success-only"].value == 69.3
    assert not any(metric.official_eligible for metric in collected.metrics)
    assert collected.completed_count == 2
    assert collected.failed_count == 0
    assert collected.error_counts == {}
    report = adapter.render_report_data(collected)
    assert report["metric_value"] == 50.0
    assert report["calibration_error"] == 20.0


def test_judge_reasoning_effort_changes_protocol_and_reaches_connector_request():
    adapter = HumanitysLastExamAdapter()
    default_plan = adapter.build_run_plan(_selection(), environ={})
    reasoned_plan = adapter.build_run_plan(
        _selection(judge_generation_parameters={"reasoning_effort": "high"}),
        environ={},
    )

    assert default_plan.protocol.judge_reasoning_mode is None
    assert reasoned_plan.protocol.judge_reasoning_mode == "high"
    assert reasoned_plan.selection.judge_generation_parameters["reasoning_effort"] == "high"
    assert reasoned_plan.protocol_hash != default_plan.protocol_hash

    target = FakeOpenAIConnector(adapter.smoke_connector_results(reasoned_plan, ModelRole.TARGET))
    judge = FakeOpenAIConnector(adapter.smoke_connector_results(reasoned_plan, ModelRole.JUDGE))
    result = adapter.run(
        reasoned_plan,
        reasoned_plan.items[0],
        ModelConnectorSet(target=target, judge=judge),
        cancel_check=lambda: False,
    )

    assert result.error_class is None
    assert judge.requests[0].reasoning_effort == "high"
    with pytest.raises(ValueError, match="judge generation parameters are invalid"):
        adapter.build_run_plan(
            _selection(judge_generation_parameters={"reasoning_effort": "extreme"}),
            environ={},
        )


def test_run_preserves_request_evidence_on_early_success_and_exhausted_judge_error():
    adapter = HumanitysLastExamAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    item = plan.items[0]
    target_request_id = hle_module._target_request_id(item)
    judge_request_id = hle_module._judge_request_id(item)
    target_result = adapter.smoke_connector_results(plan, ModelRole.TARGET)[target_request_id]
    judge_result = adapter.smoke_connector_results(plan, ModelRole.JUDGE)[judge_request_id]
    success_target = _SequenceConnector((target_result,))
    success_judge = _SequenceConnector((judge_result,))

    success = adapter.run(
        plan,
        item,
        ModelConnectorSet(target=success_target, judge=success_judge),
        cancel_check=lambda: False,
    )

    assert success.target_attempts == 1
    assert success.target_request_sha256 == canonical_connector_request_sha256(
        success_target.requests[0]
    )
    assert success.judge_attempts == 1
    assert success.judge_request_sha256 == canonical_connector_request_sha256(
        success_judge.requests[0]
    )
    assert success_target.max_attempts == [5]
    assert success_judge.max_attempts == [5]

    error_target = _SequenceConnector((target_result,))
    error_judge = _SequenceConnector(
        (
            _error(
                judge_request_id,
                ConnectorErrorCode.RATE_LIMIT,
                attempts=3,
                provider_request_id="judge-provider-error-01",
                latency_seconds=1.25,
            ),
            _error(
                judge_request_id,
                ConnectorErrorCode.RATE_LIMIT,
                attempts=2,
                provider_request_id="judge-provider-error-02",
                latency_seconds=0.75,
            ),
        )
    )
    failed = adapter.run(
        plan,
        item,
        ModelConnectorSet(target=error_target, judge=error_judge),
        cancel_check=lambda: False,
    )

    assert failed.error_class == "judge_rate_limit"
    assert failed.target_attempts == 1
    assert failed.target_request_sha256 == canonical_connector_request_sha256(
        error_target.requests[0]
    )
    assert failed.judge_provider_request_id == "judge-provider-error-02"
    assert failed.judge_latency_seconds == 2.0
    assert failed.judge_attempts == 5
    assert failed.judge_request_sha256 == canonical_connector_request_sha256(
        error_judge.requests[0]
    )
    assert {canonical_connector_request_sha256(request) for request in error_judge.requests} == {
        failed.judge_request_sha256
    }
    assert error_judge.max_attempts == [5, 2]


def test_target_and_judge_error_then_success_aggregate_all_attempt_evidence():
    adapter = HumanitysLastExamAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    item = plan.items[0]
    target_request_id = hle_module._target_request_id(item)
    judge_request_id = hle_module._judge_request_id(item)
    judge_content = adapter.smoke_connector_results(plan, ModelRole.JUDGE)[
        judge_request_id
    ].response.content
    target = _SequenceConnector(
        (
            _error(
                target_request_id,
                ConnectorErrorCode.TRANSPORT,
                attempts=2,
                provider_request_id="target-error",
                latency_seconds=0.2,
            ),
            _success(
                target_request_id,
                "Answer: 7\nConfidence: 80%",
                model=plan.target_model,
                attempts=1,
                latency_seconds=0.3,
                prompt_tokens=10,
                completion_tokens=4,
                provider_request_id="target-success",
            ),
        )
    )
    judge = _SequenceConnector(
        (
            _error(
                judge_request_id,
                ConnectorErrorCode.HTTP_ERROR,
                attempts=1,
                provider_request_id="judge-error",
                latency_seconds=0.4,
            ),
            _success(
                judge_request_id,
                judge_content,
                model=plan.selection.judge_model,
                attempts=2,
                latency_seconds=0.6,
                prompt_tokens=30,
                completion_tokens=7,
                provider_request_id="judge-success",
            ),
        )
    )

    result = adapter.run(
        plan,
        item,
        ModelConnectorSet(target=target, judge=judge),
        cancel_check=lambda: False,
    )

    assert result.error_class is None
    assert result.target_attempts == 3
    assert result.latency_seconds == pytest.approx(0.5)
    assert result.input_tokens == 10
    assert result.output_tokens == 4
    assert result.provider_request_id == "target-success"
    assert result.judge_attempts == 3
    assert result.judge_latency_seconds == pytest.approx(1.0)
    assert result.judge_input_tokens == 30
    assert result.judge_output_tokens == 7
    assert result.judge_provider_request_id == "judge-success"
    assert target.max_attempts == [5, 3]
    assert judge.max_attempts == [5, 4]
    assert len({canonical_connector_request_sha256(request) for request in target.requests}) == 1
    assert len({canonical_connector_request_sha256(request) for request in judge.requests}) == 1


def test_judge_parse_then_success_aggregates_response_usage_and_latency():
    adapter = HumanitysLastExamAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    item = plan.items[0]
    target_request_id = hle_module._target_request_id(item)
    judge_request_id = hle_module._judge_request_id(item)
    target_result = adapter.smoke_connector_results(plan, ModelRole.TARGET)[target_request_id]
    valid_judge_content = adapter.smoke_connector_results(plan, ModelRole.JUDGE)[
        judge_request_id
    ].response.content
    target = _SequenceConnector((target_result,))
    judge = _SequenceConnector(
        (
            _success(
                judge_request_id,
                "not XML",
                model=plan.selection.judge_model,
                latency_seconds=0.1,
                prompt_tokens=20,
                completion_tokens=5,
                provider_request_id="judge-invalid",
            ),
            _success(
                judge_request_id,
                valid_judge_content,
                model=plan.selection.judge_model,
                attempts=2,
                latency_seconds=0.2,
                prompt_tokens=30,
                completion_tokens=7,
                provider_request_id="judge-valid",
            ),
        )
    )

    result = adapter.run(
        plan,
        item,
        ModelConnectorSet(target=target, judge=judge),
        cancel_check=lambda: False,
    )

    assert result.error_class is None
    assert result.judge_attempts == 3
    assert result.judge_latency_seconds == pytest.approx(0.3)
    assert result.judge_input_tokens == 50
    assert result.judge_output_tokens == 12
    assert result.judge_provider_request_id == "judge-valid"
    assert judge.max_attempts == [5, 4]
    assert len({canonical_connector_request_sha256(request) for request in judge.requests}) == 1


def test_judge_huge_confidence_parse_error_retries_then_succeeds():
    adapter = HumanitysLastExamAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    item = plan.items[0]
    target_request_id = hle_module._target_request_id(item)
    judge_request_id = hle_module._judge_request_id(item)
    target_result = adapter.smoke_connector_results(plan, ModelRole.TARGET)[target_request_id]
    valid_judge_content = adapter.smoke_connector_results(plan, ModelRole.JUDGE)[
        judge_request_id
    ].response.content
    huge_confidence = _judge_response_with_confidence("9" * 5_000)
    target = _SequenceConnector((target_result,))
    judge = _SequenceConnector(
        (
            _success(
                judge_request_id,
                huge_confidence,
                model=plan.selection.judge_model,
                provider_request_id="judge-huge",
            ),
            _success(
                judge_request_id,
                valid_judge_content,
                model=plan.selection.judge_model,
                provider_request_id="judge-valid",
            ),
        )
    )

    result = adapter.run(
        plan,
        item,
        ModelConnectorSet(target=target, judge=judge),
        cancel_check=lambda: False,
    )

    assert result.error_class is None
    assert result.judge_attempts == 2
    assert result.judge_latency_seconds == pytest.approx(0.2)
    assert result.judge_input_tokens == 40
    assert result.judge_output_tokens == 10
    assert result.judge_provider_request_id == "judge-valid"
    assert judge.max_attempts == [5, 4]


def test_judge_huge_confidence_parse_errors_exhaust_as_failed_item():
    adapter = HumanitysLastExamAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    item = plan.items[0]
    target_request_id = hle_module._target_request_id(item)
    judge_request_id = hle_module._judge_request_id(item)
    target_result = adapter.smoke_connector_results(plan, ModelRole.TARGET)[target_request_id]
    huge_confidence = _judge_response_with_confidence("9" * 5_000)
    target = _SequenceConnector((target_result,))
    judge = _SequenceConnector(
        tuple(
            _success(
                judge_request_id,
                huge_confidence,
                model=plan.selection.judge_model,
                provider_request_id=f"judge-huge-{index}",
            )
            for index in range(5)
        )
    )

    result = adapter.run(
        plan,
        item,
        ModelConnectorSet(target=target, judge=judge),
        cancel_check=lambda: False,
    )

    assert result.error_class == "judge_parse_error"
    assert result.judge_attempts == 5
    assert result.judge_latency_seconds == pytest.approx(0.5)
    assert result.judge_input_tokens == 100
    assert result.judge_output_tokens == 25
    assert result.judge_provider_request_id == "judge-huge-4"
    assert result.judge_provider_model == plan.selection.judge_model
    assert judge.max_attempts == [5, 4, 3, 2, 1]
    assert len({canonical_connector_request_sha256(request) for request in judge.requests}) == 1


def test_oversized_dataset_boundary_fails_target_request_with_zero_attempts(monkeypatch):
    adapter = HumanitysLastExamAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    item = plan.items[0]
    target = _SequenceConnector(())
    judge = _SequenceConnector(())

    def reject_oversized_dataset(candidate):
        assert candidate is item
        raise ValueError("target message content exceeds the supported bound")

    monkeypatch.setattr(hle_module, "_target_messages", reject_oversized_dataset)
    result = adapter.run(
        plan,
        item,
        ModelConnectorSet(target=target, judge=judge),
        cancel_check=lambda: False,
    )

    assert result.error_class == "target_request_invalid"
    assert result.target_attempts == 0
    assert result.latency_seconds == 0.0
    assert result.target_request_sha256 is None
    assert target.requests == [] and target.max_attempts == []
    assert judge.requests == [] and judge.max_attempts == []


def test_oversized_provider_boundary_fails_judge_request_with_target_evidence(monkeypatch):
    adapter = HumanitysLastExamAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    item = plan.items[0]
    target_request_id = hle_module._target_request_id(item)
    provider_response = "synthetic oversized provider response"
    target = _SequenceConnector(
        (
            _success(
                target_request_id,
                provider_response,
                model=plan.target_model,
                provider_request_id="target-oversized",
            ),
        )
    )
    judge = _SequenceConnector(())
    real_message = hle_module.ConnectorMessage

    def bounded_message(*args, **kwargs):
        if kwargs.get("role") == "user" and isinstance(kwargs.get("content"), str):
            raise ValueError("judge message content exceeds the supported bound")
        return real_message(*args, **kwargs)

    monkeypatch.setattr(hle_module, "ConnectorMessage", bounded_message)
    result = adapter.run(
        plan,
        item,
        ModelConnectorSet(target=target, judge=judge),
        cancel_check=lambda: False,
    )

    assert result.error_class == "judge_request_invalid"
    assert result.response_text == provider_response
    assert result.provider_request_id == "target-oversized"
    assert result.provider_model == plan.target_model
    assert result.target_attempts == 1
    assert result.target_request_sha256 == canonical_connector_request_sha256(target.requests[0])
    assert result.judge_attempts == 0
    assert result.judge_latency_seconds == 0.0
    assert result.judge_request_sha256 is None
    assert judge.requests == [] and judge.max_attempts == []


def test_judge_parse_then_errors_preserves_observed_provider_model():
    adapter = HumanitysLastExamAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    item = plan.items[0]
    target_request_id = hle_module._target_request_id(item)
    judge_request_id = hle_module._judge_request_id(item)
    target_result = adapter.smoke_connector_results(plan, ModelRole.TARGET)[target_request_id]
    target = _SequenceConnector((target_result,))
    judge = _SequenceConnector(
        (
            _success(
                judge_request_id,
                "not XML",
                model=plan.selection.judge_model,
                latency_seconds=0.1,
                prompt_tokens=20,
                completion_tokens=5,
                provider_request_id="judge-invalid",
            ),
            _error(
                judge_request_id,
                ConnectorErrorCode.HTTP_ERROR,
                attempts=2,
                provider_request_id="judge-error-mid",
                latency_seconds=0.2,
            ),
            _error(
                judge_request_id,
                ConnectorErrorCode.RATE_LIMIT,
                attempts=2,
                provider_request_id="judge-error-final",
                latency_seconds=0.3,
            ),
        )
    )

    result = adapter.run(
        plan,
        item,
        ModelConnectorSet(target=target, judge=judge),
        cancel_check=lambda: False,
    )

    assert result.error_class == "judge_rate_limit"
    assert result.judge_provider_model == plan.selection.judge_model
    assert result.judge_provider_request_id == "judge-error-final"
    assert result.judge_attempts == 5
    assert result.judge_latency_seconds == pytest.approx(0.6)
    assert result.judge_input_tokens == 20
    assert result.judge_output_tokens == 5
    assert judge.max_attempts == [5, 4, 2]


def test_target_errors_exhaust_exact_five_attempt_budget():
    adapter = HumanitysLastExamAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    item = plan.items[0]
    target_request_id = hle_module._target_request_id(item)
    target = _SequenceConnector(
        (
            _error(
                target_request_id,
                ConnectorErrorCode.TRANSPORT,
                attempts=2,
                latency_seconds=0.1,
            ),
            _error(
                target_request_id,
                ConnectorErrorCode.HTTP_ERROR,
                attempts=2,
                latency_seconds=0.2,
            ),
            _error(
                target_request_id,
                ConnectorErrorCode.MALFORMED,
                attempts=1,
                provider_request_id="target-final-error",
                latency_seconds=0.3,
            ),
        )
    )
    judge = _SequenceConnector(())

    result = adapter.run(
        plan,
        item,
        ModelConnectorSet(target=target, judge=judge),
        cancel_check=lambda: False,
    )

    assert result.error_class == "target_malformed"
    assert result.target_attempts == 5
    assert result.latency_seconds == pytest.approx(0.6)
    assert result.provider_request_id == "target-final-error"
    assert target.max_attempts == [5, 3, 1]
    assert len({canonical_connector_request_sha256(request) for request in target.requests}) == 1
    assert judge.requests == []


def test_empty_target_response_reaches_judge_as_official():
    adapter = HumanitysLastExamAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    item = plan.items[0]
    target_request_id = hle_module._target_request_id(item)
    judge_request_id = hle_module._judge_request_id(item)
    target = _SequenceConnector((_success(target_request_id, "", model=plan.target_model),))
    judge_content = (
        "<extracted_final_answer>None</extracted_final_answer>"
        "<reasoning>no answer</reasoning><correct>no</correct>"
        "<confidence>100</confidence>"
    )
    judge = _SequenceConnector(
        (_success(judge_request_id, judge_content, model=plan.selection.judge_model),)
    )

    result = adapter.run(
        plan,
        item,
        ModelConnectorSet(target=target, judge=judge),
        cancel_check=lambda: False,
    )

    assert result.error_class is None
    assert result.response_text == ""
    assert result.correct is False
    assert target.requests[0].allow_empty_content is True
    assert judge.requests[0].allow_empty_content is False
    assert "[response]: \n\n" in judge.requests[0].messages[0].content


def test_missing_judge_and_exhausted_connector_failures_are_denominator_incorrect():
    adapter = HumanitysLastExamAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    item = plan.items[0]
    target_request_id = hle_module._target_request_id(item)
    judge_request_id = hle_module._judge_request_id(item)
    target_result = adapter.smoke_connector_results(plan, ModelRole.TARGET)[target_request_id]
    unused_target = _SequenceConnector((target_result,))

    missing = adapter.run(
        plan,
        item,
        unused_target,
        cancel_check=lambda: False,
    )
    assert missing.error_class == "judge_connector_missing"
    assert unused_target.requests == []

    target_error = _SequenceConnector(
        (
            _error(target_request_id, attempts=2, latency_seconds=0.2),
            _error(target_request_id, attempts=2, latency_seconds=0.3),
            _error(
                target_request_id,
                attempts=1,
                provider_request_id="target-provider-error-final",
                latency_seconds=0.25,
            ),
        )
    )
    unused_judge = _SequenceConnector(())
    failed_target = adapter.run(
        plan,
        item,
        ModelConnectorSet(target=target_error, judge=unused_judge),
        cancel_check=lambda: False,
    )
    assert failed_target.error_class == "target_timeout"
    assert failed_target.latency_seconds == pytest.approx(0.75)
    assert failed_target.provider_request_id == "target-provider-error-final"
    assert failed_target.target_attempts == 5
    assert failed_target.target_request_sha256 == canonical_connector_request_sha256(
        target_error.requests[0]
    )
    assert unused_judge.requests == []

    judge_target = _SequenceConnector((target_result,))
    judge_error = _SequenceConnector(
        (
            _error(
                judge_request_id,
                ConnectorErrorCode.MALFORMED,
                attempts=1,
                latency_seconds=0.1,
            ),
            _error(
                judge_request_id,
                ConnectorErrorCode.EMPTY_RESPONSE,
                attempts=2,
                latency_seconds=0.2,
            ),
            _error(
                judge_request_id,
                attempts=2,
                provider_request_id="judge-provider-error-final",
                latency_seconds=0.3,
            ),
        )
    )
    failed_judge = adapter.run(
        plan,
        item,
        ModelConnectorSet(target=judge_target, judge=judge_error),
        cancel_check=lambda: False,
    )
    assert failed_judge.error_class == "judge_timeout"
    assert failed_judge.response_text
    assert failed_judge.judge_attempts == 5
    assert failed_judge.judge_latency_seconds == pytest.approx(0.6)
    assert failed_judge.judge_provider_request_id == "judge-provider-error-final"
    assert judge_error.max_attempts == [5, 4, 2]

    collected = adapter.collect(
        "run-hle-errors",
        plan,
        (
            failed_target,
            failed_judge,
        ),
    )
    assert collected.metrics[0].value == 0.0
    assert collected.metrics[0].denominator == 2
    assert collected.completed_count == 0
    assert collected.failed_count == 2
    assert collected.error_counts == {"target_timeout": 1, "judge_timeout": 1}


def test_target_and_judge_connector_cancellation_keep_generic_error_class():
    adapter = HumanitysLastExamAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    item = plan.items[0]
    target_request_id = hle_module._target_request_id(item)
    preempted_target = _SequenceConnector(
        (_success(target_request_id, "unused", model=plan.target_model),)
    )
    preempted = adapter.run(
        plan,
        item,
        ModelConnectorSet(target=preempted_target, judge=_SequenceConnector(())),
        cancel_check=lambda: True,
    )

    assert preempted.error_class == "cancelled"
    assert preempted.target_attempts == 0
    assert preempted_target.max_attempts == [5]
    assert preempted_target.requests == []

    target_cancelled = adapter.run(
        plan,
        item,
        ModelConnectorSet(
            target=FakeOpenAIConnector(
                {
                    hle_module._target_request_id(item): _error(
                        hle_module._target_request_id(item),
                        ConnectorErrorCode.CANCELLED,
                    )
                }
            ),
            judge=FakeOpenAIConnector({}),
        ),
        cancel_check=lambda: False,
    )
    judge_cancelled = adapter.run(
        plan,
        item,
        ModelConnectorSet(
            target=FakeOpenAIConnector(
                {
                    hle_module._target_request_id(item): adapter.smoke_connector_results(
                        plan, ModelRole.TARGET
                    )[hle_module._target_request_id(item)]
                }
            ),
            judge=FakeOpenAIConnector(
                {
                    hle_module._judge_request_id(item): _error(
                        hle_module._judge_request_id(item),
                        ConnectorErrorCode.CANCELLED,
                    )
                }
            ),
        ),
        cancel_check=lambda: False,
    )

    assert target_cancelled.error_class == "cancelled"
    assert judge_cancelled.error_class == "cancelled"


@pytest.mark.parametrize(
    ("judge_content", "expected_error"),
    [
        ("not XML", "judge_parse_error"),
        (
            (
                "<extracted_final_answer>7</extracted_final_answer>"
                "<reasoning>same</reasoning><correct>yes</correct>"
                "<confidence>101</confidence>"
            ),
            "judge_confidence_out_of_range",
        ),
    ],
)
def test_malformed_or_out_of_range_judge_output_fails_closed(
    judge_content,
    expected_error,
):
    adapter = HumanitysLastExamAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    item = plan.items[0]
    connectors = ModelConnectorSet(
        target=FakeOpenAIConnector(
            {
                hle_module._target_request_id(item): _success(
                    hle_module._target_request_id(item),
                    "Answer: 7\nConfidence: 80%",
                    model=plan.target_model,
                )
            }
        ),
        judge=FakeOpenAIConnector(
            {
                hle_module._judge_request_id(item): _success(
                    hle_module._judge_request_id(item),
                    judge_content,
                    model=plan.selection.judge_model,
                )
            }
        ),
    )

    result = adapter.run(plan, item, connectors, cancel_check=lambda: False)

    assert result.error_class == expected_error
    assert result.correct is False
    assert result.judge_response_text == judge_content
    assert result.target_attempts == 1
    assert result.target_request_sha256 == canonical_connector_request_sha256(
        connectors.target.requests[0]
    )
    assert result.judge_attempts == 5
    assert result.judge_latency_seconds == pytest.approx(0.5)
    assert result.judge_input_tokens == 100
    assert result.judge_output_tokens == 25
    assert len(connectors.judge.requests) == 5
    assert result.judge_request_sha256 == canonical_connector_request_sha256(
        connectors.judge.requests[0]
    )
    assert {
        canonical_connector_request_sha256(request) for request in connectors.judge.requests
    } == {result.judge_request_sha256}


def test_prepare_real_profile_requires_access_hash_schema_and_exact_count(tmp_path):
    adapter = HumanitysLastExamAdapter()
    needs_action = adapter.prepare("official-latest", dry_run=False)
    path, digest = _derived_snapshot(tmp_path)

    assert needs_action.status.value == "needs_user_action"
    with pytest.raises(ValueError, match="SHA-256"):
        adapter.prepare(
            "official-latest",
            dry_run=False,
            dataset_path=path,
            dataset_sha256="0" * 64,
            accepted_access=True,
        )
    ready = adapter.prepare(
        "official-latest",
        dry_run=False,
        dataset_path=path,
        dataset_sha256=digest,
        accepted_access=True,
    )
    assert ready.status.value == "ready"
    assert ready.item_count == 2500
    assert ready.dataset_revision.startswith(OFFICIAL_REVISION)

    short_path, short_digest = _derived_snapshot(tmp_path, count=2)
    with pytest.raises(ValueError, match="2,500"):
        adapter.prepare(
            "official-latest",
            dry_run=False,
            dataset_path=short_path,
            dataset_sha256=short_digest,
            accepted_access=True,
        )


def test_prepare_validates_local_images_before_items_reach_connector_boundary(tmp_path):
    adapter = HumanitysLastExamAdapter()
    invalid_image = "data:image/png;base64," + base64.b64encode(b"not a png").decode("ascii")
    invalid_path, invalid_sha256 = _derived_snapshot(
        tmp_path,
        first_image=invalid_image,
    )

    with pytest.raises(ValueError, match="declared MIME type"):
        adapter.prepare(
            "official-latest",
            dry_run=False,
            dataset_path=invalid_path,
            dataset_sha256=invalid_sha256,
            accepted_access=True,
        )

    valid_image = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[1])["image"]
    valid_path, valid_sha256 = _derived_snapshot(
        tmp_path,
        first_image=valid_image,
    )
    ready = adapter.prepare(
        "official-latest",
        dry_run=False,
        dataset_path=valid_path,
        dataset_sha256=valid_sha256,
        accepted_access=True,
    )
    plan = adapter.build_run_plan(
        _selection(
            profile="official-latest",
            mode=RunMode.SAMPLE,
            limit=1,
            dataset_path=str(valid_path),
            dataset_sha256=valid_sha256,
            accepted_access=True,
        ),
        environ={},
    )

    assert ready.status.value == "ready"
    assert isinstance(hle_module._target_messages(plan.items[0])[1].content[1], ConnectorImagePart)


def test_sample_selection_and_full_guards_precede_real_data_reads(tmp_path, monkeypatch):
    adapter = HumanitysLastExamAdapter()
    reads = 0

    def forbidden_read(_path):
        nonlocal reads
        reads += 1
        raise AssertionError("dataset read occurred before guard")

    monkeypatch.setattr(hle_module, "load_records_with_sha256", forbidden_read)
    with pytest.raises(RunGuardError, match="confirmation"):
        adapter.build_run_plan(
            _selection(
                profile="official-latest",
                mode=RunMode.FULL,
                judge_model="gpt-5-mini",
                dataset_path=str(tmp_path / "missing.jsonl"),
                dataset_sha256="0" * 64,
                accepted_access=True,
            ),
            environ={"BENCHMARK_ALLOW_FULL_RUN": "1"},
        )
    assert reads == 0


def test_protocol_signature_rejects_foreign_plan_and_profile_judge_change():
    adapter = HumanitysLastExamAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    assert adapter.protocol_signature(plan) is plan.protocol

    with pytest.raises(ValueError, match="foreign"):
        adapter.protocol_signature(replace(plan, benchmark_id="gpqa-diamond"))
    with pytest.raises(ValueError, match="judge model"):
        adapter.build_run_plan(
            _selection(judge_model="different-judge"),
            environ={},
        )
    with pytest.raises(ValueError, match="explicit judge model"):
        adapter.build_run_plan(
            _selection(
                profile="fugu-2026",
                mode=RunMode.SAMPLE,
                limit=1,
            ),
            environ={},
        )


def test_collect_empty_results_reports_all_metrics_as_unmeasured():
    adapter = HumanitysLastExamAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})

    collected = adapter.collect("run-empty", plan, ())
    metrics = {metric.name: metric for metric in collected.metrics}

    assert all(metric.value is None for metric in metrics.values())
    assert metrics["accuracy"].numerator == 0
    assert metrics["accuracy"].denominator == 0
    assert metrics["calibration-error"].numerator is None
    assert metrics["calibration-error"].denominator == 0
    assert metrics["accuracy-success-only"].numerator == 0
    assert metrics["accuracy-success-only"].denominator == 0
    assert metrics["confidence-interval"].numerator is None
    assert metrics["confidence-interval"].denominator == 0
    assert metrics["confidence-interval-success-only"].numerator is None
    assert metrics["confidence-interval-success-only"].denominator == 0
    assert not any(metric.official_eligible for metric in metrics.values())
    assert collected.completed_count == 0
    assert collected.failed_count == 0
    assert collected.error_counts == {}


def test_collect_keeps_failed_items_in_overall_accuracy_denominator():
    adapter = HumanitysLastExamAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    results = (
        ItemResult(
            item_id=plan.items[0].item_id,
            ordinal=0,
            input_sha256=plan.items[0].input_sha256,
            response_text="synthetic",
            extracted_answer="7",
            target="7",
            correct=True,
            confidence=0.8,
            scores={"accuracy": 1.0, "confidence": 0.8},
            provider_model=plan.target_model,
            judge_provider_model=plan.selection.judge_model,
        ),
        ItemResult(
            item_id=plan.items[1].item_id,
            ordinal=1,
            input_sha256=plan.items[1].input_sha256,
            response_text="",
            target="blue",
            correct=False,
            error_class="target_timeout",
        ),
    )

    collected = adapter.collect("run-partial", plan, results)
    metrics = {metric.name: metric for metric in collected.metrics}

    assert metrics["accuracy"].value == 50.0
    assert metrics["accuracy"].numerator == 1
    assert metrics["accuracy"].denominator == 2
    assert metrics["accuracy-success-only"].value == 100.0
    assert metrics["calibration-error"].value == 20.0
    assert collected.completed_count == 1
    assert collected.failed_count == 1
    assert collected.error_counts == {"target_timeout": 1}


def test_collect_preserves_exact_count_ratio_for_nonterminating_accuracy():
    adapter = HumanitysLastExamAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    results = tuple(
        ItemResult(
            item_id=f"synthetic-{ordinal}",
            ordinal=ordinal,
            input_sha256=f"{ordinal + 1:064x}",
            response_text="synthetic",
            target="answer",
            correct=ordinal == 0,
            confidence=0.5,
            scores={
                "accuracy": 1.0 if ordinal == 0 else 0.0,
                "confidence": 0.5,
            },
            provider_model=plan.target_model,
            judge_provider_model=plan.selection.judge_model,
        )
        for ordinal in range(3)
    )

    collected = adapter.collect("run-one-of-three", plan, results)
    metrics = {metric.name: metric for metric in collected.metrics}
    primary = metrics["accuracy"]
    success_only = metrics["accuracy-success-only"]

    assert primary.value == pytest.approx(100.0 / 3.0)
    assert primary.value == primary.numerator / primary.denominator * primary.scale
    assert success_only.value == pytest.approx(100.0 / 3.0)
