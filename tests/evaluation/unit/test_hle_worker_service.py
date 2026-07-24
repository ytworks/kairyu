"""Synthetic HLE coverage across service, worker, artifacts, and reports."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from kairyu.evaluation.adapters.base import (
    AdapterRunPlan,
    ItemResult,
    ModelRole,
    RunSelection,
)
from kairyu.evaluation.adapters.humanitys_last_exam import HumanitysLastExamAdapter
from kairyu.evaluation.artifacts import RUN_ARTIFACT_FILES
from kairyu.evaluation.cli import _preflight_payload
from kairyu.evaluation.connectors import (
    ConnectorError,
    ConnectorErrorCode,
    ConnectorResult,
    FakeOpenAIConnector,
    ModelRequest,
    canonical_connector_request_sha256,
)
from kairyu.evaluation.protocol import protocol_hash
from kairyu.evaluation.schemas import ItemState, ProtocolSignature, RunMode, RunState
from kairyu.evaluation.service import (
    BenchmarkService,
    ConnectorConfig,
    EvaluationRuntime,
    bind_connector_to_plan,
)
from kairyu.evaluation.worker import render_saved_report, run_worker_once

_BENCHMARK_ID = "humanitys-last-exam"
_TARGET_MODEL = "offline-hle-target"
_JUDGE_MODEL = "gpt-5-mini"
_ROOT = Path(__file__).parents[3]
_FIXTURE = (
    _ROOT / "kairyu" / "evaluation" / "resources" / "fixtures" / "humanitys-last-exam-smoke.jsonl"
)


class _RoleAwareHLEConnectorFactory:
    def __init__(self, *, judge_timeout_ordinal: int | None = None) -> None:
        self._judge_timeout_ordinal = judge_timeout_ordinal
        self._connectors: dict[ModelRole, list[FakeOpenAIConnector]] = {}
        self.roles: list[ModelRole] = []

    def __call__(
        self,
        config: ConnectorConfig,
        plan: AdapterRunPlan,
        runtime: EvaluationRuntime,
        role: ModelRole,
    ):
        assert config.kind == "fake"
        assert plan.benchmark_id == _BENCHMARK_ID
        responses = dict(HumanitysLastExamAdapter().smoke_connector_results(plan, role))
        if role is ModelRole.JUDGE and self._judge_timeout_ordinal is not None:
            request_ids = tuple(responses)
            assert 0 <= self._judge_timeout_ordinal < len(request_ids)
            request_id = request_ids[self._judge_timeout_ordinal]
            responses[request_id] = ConnectorResult(
                error=ConnectorError(
                    request_id=request_id,
                    code=ConnectorErrorCode.TIMEOUT,
                    detail="controlled offline HLE judge timeout",
                    retryable=True,
                    attempts=1,
                    provider_request_id=(f"fake-judge-timeout-{self._judge_timeout_ordinal}"),
                    latency_seconds=0.25,
                )
            )
        connector = FakeOpenAIConnector(
            responses,
            secret_registry=runtime.secret_registry,
        )
        self.roles.append(role)
        self._connectors.setdefault(role, []).append(connector)
        return connector, lambda: None

    def requests_for(self, role: ModelRole) -> tuple[ModelRequest, ...]:
        return tuple(
            request
            for connector in self._connectors.get(role, ())
            for request in connector.requests
        )


class _DurableCancelHLEJudgeConnector(FakeOpenAIConnector):
    def __init__(
        self,
        responses: dict[str, ConnectorResult],
        *,
        service: BenchmarkService,
        run_id: str,
        cancel_request_id: str,
        runtime: EvaluationRuntime,
    ) -> None:
        super().__init__(responses, secret_registry=runtime.secret_registry)
        self._service = service
        self._run_id = run_id
        self._cancel_request_id = cancel_request_id

    def complete(
        self,
        request: ModelRequest,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        max_attempts: int | None = None,
    ) -> ConnectorResult:
        if request.request_id != self._cancel_request_id:
            return super().complete(
                request,
                cancel_requested=cancel_requested,
                max_attempts=max_attempts,
            )
        assert cancel_requested is not None
        assert cancel_requested() is False
        recorded = super().complete(
            request,
            cancel_requested=cancel_requested,
            max_attempts=max_attempts,
        )
        assert recorded.response is not None
        cancelled = self._service.cancel(self._run_id)
        assert cancelled.cancel_requested is True
        assert cancel_requested() is True
        return ConnectorResult(
            error=ConnectorError(
                request_id=request.request_id,
                code=ConnectorErrorCode.CANCELLED,
                detail="controlled offline HLE mid-judge cancellation",
                retryable=False,
                attempts=1,
                provider_request_id="fake-judge-cancelled-0",
                latency_seconds=0.5,
            )
        )


class _DurableCancelHLEConnectorFactory(_RoleAwareHLEConnectorFactory):
    def __init__(
        self,
        *,
        service: BenchmarkService,
        run_id: str,
    ) -> None:
        super().__init__()
        self._service = service
        self._run_id = run_id

    def __call__(
        self,
        config: ConnectorConfig,
        plan: AdapterRunPlan,
        runtime: EvaluationRuntime,
        role: ModelRole,
    ):
        if role is not ModelRole.JUDGE:
            return super().__call__(config, plan, runtime, role)
        assert config.kind == "fake"
        assert plan.benchmark_id == _BENCHMARK_ID
        responses = dict(HumanitysLastExamAdapter().smoke_connector_results(plan, role))
        cancel_request_id = tuple(responses)[0]
        connector = _DurableCancelHLEJudgeConnector(
            responses,
            service=self._service,
            run_id=self._run_id,
            cancel_request_id=cancel_request_id,
            runtime=runtime,
        )
        self.roles.append(role)
        self._connectors.setdefault(role, []).append(connector)
        return connector, lambda: None


def _selection() -> RunSelection:
    return RunSelection(
        profile="smoke",
        mode=RunMode.SMOKE,
        target_model=_TARGET_MODEL,
    )


def _submit_fake(runtime: EvaluationRuntime, run_id: str):
    service = BenchmarkService(runtime)
    submitted = service.submit(
        _BENCHMARK_ID,
        _selection(),
        ConnectorConfig(kind="fake"),
        run_id=run_id,
    )
    payload = runtime.store.get_job(submitted.job_id).payload

    assert submitted.estimated_model_calls == submitted.estimate.model_calls == 4
    assert submitted.estimate.maximum_model_calls == 20
    assert submitted.estimate.maximum_duration_seconds == 12_000.0
    assert submitted.run.judge_model == _JUDGE_MODEL
    assert payload["selection"]["judge_model"] == _JUDGE_MODEL
    assert payload["protocol"]["judge_model"] == _JUDGE_MODEL
    assert (
        payload["judge_connector"]
        == payload["connector"]
        == {
            "endpoint": None,
            "kind": "fake",
            "max_response_bytes": 1_048_576,
            "max_retries": 2,
            "secret_env_name": None,
        }
    )
    adapter_configuration = payload["protocol"]["adapter_configuration"]
    assert adapter_configuration["model_calls_by_role"] == {
        "judge": 2,
        "target": 2,
    }
    assert adapter_configuration["retry_budget_scope"] == ("per-model-request-total-attempts")
    assert adapter_configuration["max_attempts_per_request"] == 5
    planning_estimate = adapter_configuration["planning_evidence"]["estimate"]
    assert planning_estimate["maximum_model_calls"] == 20
    assert planning_estimate["maximum_duration_seconds"] == 12_000.0
    assert planning_estimate["maximum_output_tokens"] == submitted.estimate.maximum_output_tokens
    assert planning_estimate["assumptions"][-1] == (
        "Adapter-owned total attempt budgets cap each model request at 5 attempts; "
        "connector-internal retries add 0 seconds of worst-case default backoff "
        "across role calls (judge=0s, target=0s) without increasing model-call or "
        "output-token ceilings."
    )
    return service, submitted


def _read_jsonl(runtime: EvaluationRuntime, run_id: str, relative_path: str):
    return tuple(
        json.loads(line)
        for line in runtime.artifacts.read_bytes(run_id, relative_path).splitlines()
    )


def _report_bytes(runtime: EvaluationRuntime, run_id: str) -> dict[str, bytes]:
    return {
        relative_path: runtime.artifacts.read_bytes(run_id, relative_path)
        for relative_path in ("report.json", "report.md", "report.html")
    }


def _metric_projection(metrics):
    return tuple((metric["name"], metric["value"], metric["denominator"]) for metric in metrics)


def _fixture_sensitive_values() -> tuple[bytes, ...]:
    records = tuple(
        json.loads(line)
        for line in _FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    values = [record["id"] for record in records]
    values.extend(record["question"] for record in records)
    values.extend(record["image"] for record in records if record["image"] is not None)
    return tuple(value.encode("utf-8") for value in values)


def test_hle_smoke_service_worker_publishes_role_separated_evidence(tmp_path: Path):
    runtime = EvaluationRuntime(tmp_path / "state")
    _, submitted = _submit_fake(runtime, "run-hle-smoke")

    assert run_worker_once(runtime, worker_id="worker-hle-smoke") == "run-hle-smoke"

    stored = runtime.store.get_run("run-hle-smoke").run
    assert stored.state is RunState.COMPLETED
    assert (stored.completed_count, stored.failed_count, stored.skipped_count) == (2, 0, 0)
    assert runtime.store.get_job(submitted.job_id).status == "completed"
    stored_items = tuple(item.item for item in runtime.store.list_run_items("run-hle-smoke"))
    assert len(stored_items) == 2
    assert all(item.state.value == "completed" for item in stored_items)
    assert [dict(item.scores) for item in stored_items] == [
        {"accuracy": 1.0, "confidence": 0.8},
        {"accuracy": 0.0, "confidence": 0.6},
    ]

    metrics = runtime.artifacts.read_json("run-hle-smoke", "metrics.json")
    assert [metric["name"] for metric in metrics] == [
        "accuracy",
        "calibration-error",
        "accuracy-success-only",
        "confidence-interval",
        "confidence-interval-success-only",
    ]
    metrics_by_name = {metric["name"]: metric for metric in metrics}
    assert metrics_by_name["accuracy"]["value"] == 50.0
    assert metrics_by_name["accuracy"]["numerator"] == 1
    assert metrics_by_name["calibration-error"]["value"] == 20.0
    assert metrics_by_name["accuracy-success-only"]["value"] == 50.0
    assert metrics_by_name["confidence-interval"]["value"] == 69.3
    assert metrics_by_name["confidence-interval-success-only"]["value"] == 69.3
    assert {metric["denominator"] for metric in metrics} == {2}

    predictions = _read_jsonl(runtime, "run-hle-smoke", "predictions.jsonl")
    assert len(predictions) * 2 == submitted.estimated_model_calls == 4
    assert [prediction["provider_request_id"] for prediction in predictions] == [
        "fake-target-0",
        "fake-target-1",
    ]
    assert [prediction["judge_provider_request_id"] for prediction in predictions] == [
        "fake-judge-0",
        "fake-judge-1",
    ]
    assert {prediction["provider_model"] for prediction in predictions} == {_TARGET_MODEL}
    assert {prediction["judge_provider_model"] for prediction in predictions} == {_JUDGE_MODEL}
    usage = runtime.artifacts.read_json("run-hle-smoke", "usage.json")
    assert usage["input_tokens"] == 40
    assert usage["output_tokens"] == 20
    assert usage["measurement_status"] == "complete"
    manifest = runtime.artifacts.read_json("run-hle-smoke", "manifest.json")
    assert manifest["observed_provider_models"] == sorted((_TARGET_MODEL, _JUDGE_MODEL))

    artifact_paths = {
        artifact.relative_path for artifact in runtime.store.list_artifacts("run-hle-smoke")
    }
    checkpoint_paths = {path for path in artifact_paths if path.startswith("upstream/checkpoints/")}
    assert len(RUN_ARTIFACT_FILES) == 12
    assert len(checkpoint_paths) == 2
    assert artifact_paths == set(RUN_ARTIFACT_FILES) | checkpoint_paths
    checkpoint_results = tuple(
        ItemResult.model_validate_json(runtime.artifacts.read_bytes("run-hle-smoke", relative_path))
        for relative_path in sorted(checkpoint_paths)
    )
    assert [
        (
            result.input_tokens,
            result.output_tokens,
            result.judge_input_tokens,
            result.judge_output_tokens,
        )
        for result in checkpoint_results
    ] == [(10, 5, 10, 5)] * 2
    assert {result.provider_model for result in checkpoint_results} == {_TARGET_MODEL}
    assert {result.judge_provider_model for result in checkpoint_results} == {_JUDGE_MODEL}

    protocol = runtime.artifacts.read_json("run-hle-smoke", "protocol.json")
    references = runtime.artifacts.read_json("run-hle-smoke", "references.json")
    report_before = _report_bytes(runtime, "run-hle-smoke")
    report = json.loads(report_before["report.json"])
    assert protocol["benchmark_id"] == report["benchmark_id"] == _BENCHMARK_ID
    assert protocol["judge_model"] == report["judge_model"] == _JUDGE_MODEL
    assert (
        stored.protocol_hash
        == report["protocol_hash"]
        == protocol_hash(ProtocolSignature.model_validate(protocol))
    )
    assert set(protocol["adapter_configuration"]["model_connectors"]) == {
        "judge",
        "target",
    }
    assert references["benchmark_id"] == _BENCHMARK_ID
    assert references["snapshot_id"] == "sakana-fugu-technical-report-2026-v2-hle"
    assert len(references["results"]) == len(report["references"]) == 5
    assert {reference["reference_id"] for reference in report["references"]} == {
        reference["reference_id"] for reference in references["results"]
    }
    assert _metric_projection(report["metrics"]) == _metric_projection(metrics)
    assert report["counts"]["selected"] == report["counts"]["reported_items"] == 2

    rerendered = render_saved_report(runtime, "run-hle-smoke")

    assert rerendered.metrics[0].value == 50.0
    assert _report_bytes(runtime, "run-hle-smoke") == report_before

    forbidden = _fixture_sensitive_values()
    control_database_paths = tuple(
        runtime.store.database_path.parent.glob(f"{runtime.store.database_path.name}*")
    )
    boundary_paths = (
        *control_database_paths,
        *(runtime.artifacts.path_for("run-hle-smoke", path) for path in RUN_ARTIFACT_FILES[-3:]),
    )
    for path in boundary_paths:
        content = path.read_bytes()
        assert all(value not in content for value in forbidden)


def test_hle_judge_timeout_persists_partial_evidence_and_resume_retries_failed_only(
    tmp_path: Path,
):
    source_run_id = "run-hle-judge-timeout"
    successor_run_id = "run-hle-judge-timeout-resumed"
    runtime = EvaluationRuntime(tmp_path / "state")
    service, submitted = _submit_fake(runtime, source_run_id)
    failing_factory = _RoleAwareHLEConnectorFactory(judge_timeout_ordinal=1)

    assert (
        run_worker_once(
            runtime,
            worker_id="worker-hle-judge-timeout",
            connector_factory=failing_factory,
        )
        == source_run_id
    )

    assert failing_factory.roles == [ModelRole.TARGET, ModelRole.JUDGE]
    target_requests = failing_factory.requests_for(ModelRole.TARGET)
    judge_requests = failing_factory.requests_for(ModelRole.JUDGE)
    assert len(target_requests) == 2
    assert len(judge_requests) == 6
    failed_target_request = target_requests[1]
    failed_judge_requests = judge_requests[1:]
    assert len(failed_judge_requests) == 5
    assert {request.request_id for request in failed_judge_requests} == {
        failed_judge_requests[0].request_id
    }
    failed_judge_request = failed_judge_requests[-1]

    source_run = runtime.store.get_run(source_run_id).run
    assert source_run.state is RunState.PARTIAL
    assert source_run.partial is True
    assert source_run.termination_reason == "one_or_more_items_failed"
    assert (source_run.completed_count, source_run.failed_count) == (1, 1)
    assert runtime.store.get_job(submitted.job_id).status == "failed"
    source_items = tuple(stored.item for stored in runtime.store.list_run_items(source_run_id))
    assert [item.state for item in source_items] == [
        ItemState.COMPLETED,
        ItemState.FAILED,
    ]
    assert source_items[1].error_class == "judge_timeout"
    assert all(item.checkpoint_source_run_id == source_run_id for item in source_items)
    checkpoint_paths = {
        item.checkpoint_relative_path
        for item in source_items
        if item.checkpoint_relative_path is not None
    }
    assert len(checkpoint_paths) == 2
    source_artifact_paths = {
        artifact.relative_path for artifact in runtime.store.list_artifacts(source_run_id)
    }
    assert source_artifact_paths == set(RUN_ARTIFACT_FILES) | checkpoint_paths

    completed_checkpoint_path = source_items[0].checkpoint_relative_path
    failed_checkpoint_path = source_items[1].checkpoint_relative_path
    assert completed_checkpoint_path is not None
    assert failed_checkpoint_path is not None
    failed_checkpoint = ItemResult.model_validate_json(
        runtime.artifacts.read_bytes(source_run_id, failed_checkpoint_path)
    )
    assert failed_checkpoint.model_dump(mode="json") == {
        "item_id": source_items[1].item_id,
        "ordinal": 1,
        "input_sha256": source_items[1].input_sha256,
        "response_text": ("Explanation: synthetic fixture response\nAnswer: red\nConfidence: 60%"),
        "extracted_answer": None,
        "target": "blue",
        "correct": False,
        "error_class": "judge_timeout",
        "latency_seconds": 0.0,
        "finish_reason": "stop",
        "provider_request_id": "fake-target-1",
        "provider_model": _TARGET_MODEL,
        "input_tokens": 10,
        "output_tokens": 5,
        "target_attempts": 1,
        "target_request_sha256": canonical_connector_request_sha256(failed_target_request),
        "scores": {},
        "report_error_class": None,
        "confidence": None,
        "judge_response_text": None,
        "judge_finish_reason": None,
        "judge_provider_request_id": "fake-judge-timeout-1",
        "judge_provider_model": None,
        "judge_input_tokens": None,
        "judge_output_tokens": None,
        "judge_latency_seconds": 1.25,
        "judge_attempts": 5,
        "judge_request_sha256": canonical_connector_request_sha256(failed_judge_request),
    }

    predictions = _read_jsonl(runtime, source_run_id, "predictions.jsonl")
    assert len(predictions) == 2
    assert predictions[1] == {
        "extracted_answer": None,
        "finish_reason": "stop",
        "confidence": None,
        "item_id": source_items[1].item_id,
        "judge_attempts": 5,
        "judge_finish_reason": None,
        "judge_latency_seconds": 1.25,
        "judge_provider_model": None,
        "judge_provider_request_id": "fake-judge-timeout-1",
        "judge_request_sha256": failed_checkpoint.judge_request_sha256,
        "judge_response_text": None,
        "latency_seconds": 0.0,
        "provider_request_id": "fake-target-1",
        "provider_model": _TARGET_MODEL,
        "response_text": failed_checkpoint.response_text,
        "target_attempts": 1,
        "target_request_sha256": failed_checkpoint.target_request_sha256,
    }
    assert _read_jsonl(runtime, source_run_id, "errors.jsonl") == (
        {
            "error_class": "judge_timeout",
            "item_id": source_items[1].item_id,
            "ordinal": 1,
        },
    )

    metrics = runtime.artifacts.read_json(source_run_id, "metrics.json")
    assert [
        (
            metric["name"],
            metric["value"],
            metric["numerator"],
            metric["denominator"],
            metric["official_eligible"],
        )
        for metric in metrics
    ] == [
        ("accuracy", 50.0, 1, 2, False),
        ("calibration-error", 20.0, None, 1, False),
        ("accuracy-success-only", 100.0, 1, 1, False),
        ("confidence-interval", 69.3, None, 2, False),
        ("confidence-interval-success-only", 0.0, None, 1, False),
    ]
    usage = runtime.artifacts.read_json(source_run_id, "usage.json")
    assert usage == {
        "schema_version": 1,
        "input_tokens": 30,
        "output_tokens": 15,
        "total_latency_seconds": 1.25,
        "measurement_status": "partial",
        "measurement_unavailable_reasons": [
            "one or more judged items lack judge input token usage",
            "one or more judged items lack judge output token usage",
            "judge usage may omit earlier connector attempts",
            "usage is unavailable for one or more non-completed items",
        ],
        "actual_cost_usd": None,
        "actual_cost_unavailable_reason": ("item result evidence does not record monetary cost"),
    }
    source_report = runtime.artifacts.read_json(source_run_id, "report.json")
    assert source_report["state"] == "partial"
    assert source_report["counts"]["completed"] == 1
    assert source_report["counts"]["failed"] == 1
    assert _metric_projection(source_report["metrics"]) == _metric_projection(metrics)
    assert source_report["usage"] == usage

    resumed = service.resume(source_run_id, new_run_id=successor_run_id)
    assert resumed.run.run.resumed_from_run_id == source_run_id
    success_factory = _RoleAwareHLEConnectorFactory()

    assert (
        run_worker_once(
            runtime,
            worker_id="worker-hle-judge-timeout-resumed",
            connector_factory=success_factory,
        )
        == successor_run_id
    )

    assert success_factory.roles == [ModelRole.TARGET, ModelRole.JUDGE]
    retried_target_requests = success_factory.requests_for(ModelRole.TARGET)
    retried_judge_requests = success_factory.requests_for(ModelRole.JUDGE)
    assert [request.request_id for request in retried_target_requests] == [
        failed_target_request.request_id
    ]
    assert [request.request_id for request in retried_judge_requests] == [
        failed_judge_request.request_id
    ]
    successor_run = runtime.store.get_run(successor_run_id).run
    assert successor_run.state is RunState.COMPLETED
    assert successor_run.partial is False
    assert successor_run.termination_reason is None
    assert (successor_run.completed_count, successor_run.failed_count) == (2, 0)
    assert runtime.store.get_job(resumed.job.job_id).status == "completed"
    successor_items = tuple(
        stored.item for stored in runtime.store.list_run_items(successor_run_id)
    )
    assert [item.state for item in successor_items] == [
        ItemState.COMPLETED,
        ItemState.COMPLETED,
    ]
    assert successor_items[0].checkpoint_source_run_id == source_run_id
    assert successor_items[0].checkpoint_relative_path == completed_checkpoint_path
    assert successor_items[1].checkpoint_source_run_id == successor_run_id
    assert successor_items[1].checkpoint_relative_path != failed_checkpoint_path
    successor_checkpoint_path = successor_items[1].checkpoint_relative_path
    assert successor_checkpoint_path is not None
    successor_artifact_paths = {
        artifact.relative_path for artifact in runtime.store.list_artifacts(successor_run_id)
    }
    assert successor_artifact_paths == set(RUN_ARTIFACT_FILES) | {successor_checkpoint_path}
    assert {
        artifact.relative_path for artifact in runtime.store.list_artifacts(source_run_id)
    } == source_artifact_paths
    assert runtime.store.get_run(source_run_id).run.state is RunState.PARTIAL
    successor_usage = runtime.artifacts.read_json(successor_run_id, "usage.json")
    assert successor_usage["measurement_status"] == "complete"
    assert successor_usage["measurement_unavailable_reasons"] == []
    assert (successor_usage["input_tokens"], successor_usage["output_tokens"]) == (40, 20)
    successor_report = runtime.artifacts.read_json(successor_run_id, "report.json")
    assert successor_report["state"] == "completed"
    assert successor_report["counts"]["completed"] == 2
    assert successor_report["counts"]["failed"] == 0


def test_hle_mid_judge_cancel_checkpoints_evidence_and_excludes_cancelled_metrics(
    tmp_path: Path,
):
    source_run_id = "run-hle-mid-judge-cancel"
    successor_run_id = "run-hle-mid-judge-cancel-resumed"
    runtime = EvaluationRuntime(tmp_path / "state")
    service, submitted = _submit_fake(runtime, source_run_id)
    cancelling_factory = _DurableCancelHLEConnectorFactory(
        service=service,
        run_id=source_run_id,
    )

    assert (
        run_worker_once(
            runtime,
            worker_id="worker-hle-mid-judge-cancel",
            connector_factory=cancelling_factory,
        )
        == source_run_id
    )

    assert cancelling_factory.roles == [ModelRole.TARGET, ModelRole.JUDGE]
    target_requests = cancelling_factory.requests_for(ModelRole.TARGET)
    judge_requests = cancelling_factory.requests_for(ModelRole.JUDGE)
    assert len(target_requests) == len(judge_requests) == 1
    target_request = target_requests[0]
    judge_request = judge_requests[0]

    source_run = runtime.store.get_run(source_run_id).run
    assert source_run.state is RunState.CANCELLED
    assert source_run.termination_reason == "cancel_requested"
    assert (source_run.completed_count, source_run.failed_count) == (0, 0)
    source_job = runtime.store.get_job(submitted.job_id)
    assert source_job.status == "cancelled"
    assert source_job.cancel_requested is True
    source_items = tuple(stored.item for stored in runtime.store.list_run_items(source_run_id))
    assert [item.state for item in source_items] == [
        ItemState.CANCELLED,
        ItemState.PENDING,
    ]
    cancelled_item = source_items[0]
    assert cancelled_item.error_class == "cancelled"
    assert cancelled_item.checkpoint_source_run_id == source_run_id
    checkpoint_path = cancelled_item.checkpoint_relative_path
    assert checkpoint_path is not None
    assert source_items[1].checkpoint_relative_path is None
    source_artifact_paths = {
        artifact.relative_path for artifact in runtime.store.list_artifacts(source_run_id)
    }
    assert source_artifact_paths == set(RUN_ARTIFACT_FILES) | {checkpoint_path}

    checkpoint = ItemResult.model_validate_json(
        runtime.artifacts.read_bytes(source_run_id, checkpoint_path)
    )
    assert checkpoint.model_dump(mode="json") == {
        "item_id": cancelled_item.item_id,
        "ordinal": 0,
        "input_sha256": cancelled_item.input_sha256,
        "response_text": ("Explanation: synthetic fixture response\nAnswer: 7\nConfidence: 80%"),
        "extracted_answer": None,
        "target": "7",
        "correct": False,
        "error_class": "cancelled",
        "latency_seconds": 0.0,
        "finish_reason": "stop",
        "provider_request_id": "fake-target-0",
        "provider_model": _TARGET_MODEL,
        "input_tokens": 10,
        "output_tokens": 5,
        "target_attempts": 1,
        "target_request_sha256": canonical_connector_request_sha256(target_request),
        "scores": {},
        "report_error_class": None,
        "confidence": None,
        "judge_response_text": None,
        "judge_finish_reason": None,
        "judge_provider_request_id": "fake-judge-cancelled-0",
        "judge_provider_model": None,
        "judge_input_tokens": None,
        "judge_output_tokens": None,
        "judge_latency_seconds": 0.5,
        "judge_attempts": 1,
        "judge_request_sha256": canonical_connector_request_sha256(judge_request),
    }
    predictions = _read_jsonl(runtime, source_run_id, "predictions.jsonl")
    assert predictions == (
        {
            "extracted_answer": None,
            "finish_reason": "stop",
            "confidence": None,
            "item_id": cancelled_item.item_id,
            "judge_attempts": 1,
            "judge_finish_reason": None,
            "judge_latency_seconds": 0.5,
            "judge_provider_model": None,
            "judge_provider_request_id": "fake-judge-cancelled-0",
            "judge_request_sha256": checkpoint.judge_request_sha256,
            "judge_response_text": None,
            "latency_seconds": 0.0,
            "provider_request_id": "fake-target-0",
            "provider_model": _TARGET_MODEL,
            "response_text": checkpoint.response_text,
            "target_attempts": 1,
            "target_request_sha256": checkpoint.target_request_sha256,
        },
    )
    assert _read_jsonl(runtime, source_run_id, "errors.jsonl") == (
        {
            "error_class": "cancelled",
            "item_id": cancelled_item.item_id,
            "ordinal": 0,
        },
    )

    metrics = runtime.artifacts.read_json(source_run_id, "metrics.json")
    assert [metric["name"] for metric in metrics] == [
        "accuracy",
        "calibration-error",
        "accuracy-success-only",
        "confidence-interval",
        "confidence-interval-success-only",
    ]
    assert all(metric["value"] is None for metric in metrics)
    assert all(metric["denominator"] == 0 for metric in metrics)
    usage = runtime.artifacts.read_json(source_run_id, "usage.json")
    assert usage == {
        "schema_version": 1,
        "input_tokens": 10,
        "output_tokens": 5,
        "total_latency_seconds": 0.5,
        "measurement_status": "partial",
        "measurement_unavailable_reasons": [
            "one or more judged items lack judge input token usage",
            "one or more judged items lack judge output token usage",
            "usage is unavailable for one or more non-completed items",
        ],
        "actual_cost_usd": None,
        "actual_cost_unavailable_reason": ("item result evidence does not record monetary cost"),
    }
    report = runtime.artifacts.read_json(source_run_id, "report.json")
    assert report["state"] == "cancelled"
    assert report["counts"]["cancelled"] == 1
    assert report["counts"]["pending"] == 1
    assert report["counts"]["completed"] == report["counts"]["failed"] == 0
    assert _metric_projection(report["metrics"]) == _metric_projection(metrics)
    assert report["usage"] == usage
    manifest = runtime.artifacts.read_json(source_run_id, "manifest.json")
    assert manifest["observed_provider_models"] == [_TARGET_MODEL]

    resumed = service.resume(source_run_id, new_run_id=successor_run_id)
    success_factory = _RoleAwareHLEConnectorFactory()
    assert (
        run_worker_once(
            runtime,
            worker_id="worker-hle-mid-judge-cancel-resumed",
            connector_factory=success_factory,
        )
        == successor_run_id
    )
    assert len(success_factory.requests_for(ModelRole.TARGET)) == 2
    assert len(success_factory.requests_for(ModelRole.JUDGE)) == 2
    assert success_factory.requests_for(ModelRole.TARGET)[0].request_id == target_request.request_id
    assert success_factory.requests_for(ModelRole.JUDGE)[0].request_id == judge_request.request_id
    successor_run = runtime.store.get_run(successor_run_id).run
    assert successor_run.state is RunState.COMPLETED
    assert runtime.store.get_job(resumed.job.job_id).status == "completed"
    successor_items = tuple(
        stored.item for stored in runtime.store.list_run_items(successor_run_id)
    )
    assert all(item.state is ItemState.COMPLETED for item in successor_items)
    assert all(item.checkpoint_source_run_id == successor_run_id for item in successor_items)


def test_hle_queued_cancel_publishes_five_null_metrics_and_three_reports(tmp_path: Path):
    runtime = EvaluationRuntime(tmp_path / "state")
    service, submitted = _submit_fake(runtime, "run-hle-cancelled")

    cancelled = service.cancel("run-hle-cancelled")

    assert cancelled.status == "cancelled"
    assert runtime.store.get_job(submitted.job_id).status == "cancelled"
    stored = runtime.store.get_run("run-hle-cancelled").run
    assert stored.state is RunState.CANCELLED
    assert stored.termination_reason == "cancel_requested"
    artifact_paths = {
        artifact.relative_path for artifact in runtime.store.list_artifacts("run-hle-cancelled")
    }
    assert artifact_paths == set(RUN_ARTIFACT_FILES)
    metrics = runtime.artifacts.read_json("run-hle-cancelled", "metrics.json")
    assert len(metrics) == 5
    assert all(metric["value"] is None for metric in metrics)
    assert all(metric["denominator"] == 0 for metric in metrics)
    assert runtime.artifacts.read_bytes("run-hle-cancelled", "item_results.jsonl") == b""

    reports = _report_bytes(runtime, "run-hle-cancelled")
    report = json.loads(reports["report.json"])
    assert reports["report.md"]
    assert reports["report.html"]
    assert report["state"] == "cancelled"
    assert report["counts"]["selected"] == 2
    assert report["counts"]["reported_items"] == 0
    assert report["judge_model"] == _JUDGE_MODEL
    assert _metric_projection(report["metrics"]) == _metric_projection(metrics)
    assert all(metric["value"] is None for metric in report["metrics"])
    assert all(metric["denominator"] == 0 for metric in report["metrics"])
    assert report["usage"]["input_tokens"] == 0
    assert report["usage"]["output_tokens"] == 0
    assert len(report["references"]) == 5


def test_hle_role_specific_openai_configs_bind_hash_without_persisting_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target_secret = "hle-target-runtime-secret-87c1"
    judge_secret = "hle-judge-runtime-secret-19d4"
    monkeypatch.setenv("HLE_TARGET_API_KEY", target_secret)
    monkeypatch.setenv("HLE_JUDGE_API_KEY", judge_secret)

    def fail_network(*_args, **_kwargs):
        raise AssertionError("HLE submit attempted an external API call")

    monkeypatch.setattr("socket.socket", fail_network)
    runtime = EvaluationRuntime(tmp_path / "state")
    service = BenchmarkService(runtime)
    selection = _selection()

    target_first = ConnectorConfig(
        kind="openai",
        endpoint="https://target-one.example.test/provider/",
        secret_env_name="HLE_TARGET_API_KEY",
        max_retries=1,
    )
    target_second = ConnectorConfig(
        kind="openai",
        endpoint="https://target-two.example.test/provider/v1/",
        secret_env_name="HLE_TARGET_API_KEY",
        max_retries=1,
    )
    judge_first = ConnectorConfig(
        kind="openai",
        endpoint="https://judge-one.example.test/api/",
        secret_env_name="HLE_JUDGE_API_KEY",
        max_retries=2,
    )
    judge_second = ConnectorConfig(
        kind="openai",
        endpoint="https://judge-two.example.test/api/v1/",
        secret_env_name="HLE_JUDGE_API_KEY",
        max_retries=2,
    )

    preview = bind_connector_to_plan(
        HumanitysLastExamAdapter().build_run_plan(selection),
        target_first,
        judge_connector=judge_first,
    )
    expected_backoff_assumption = (
        "Adapter-owned total attempt budgets cap each model request at 5 attempts; "
        "connector-internal retries add 6 seconds of worst-case default backoff "
        "across role calls (judge=4s, target=2s) without increasing model-call or "
        "output-token ceilings."
    )
    assert preview.estimate.maximum_model_calls == 20
    assert preview.estimate.maximum_duration_seconds == 12_006.0
    assert preview.estimate.maximum_output_tokens == 61_440
    assert preview.estimate.assumptions[-1] == expected_backoff_assumption
    planning_estimate = preview.protocol.adapter_configuration["planning_evidence"]["estimate"]
    assert planning_estimate == preview.estimate.model_dump(mode="json")
    preflight = _preflight_payload(preview, HumanitysLastExamAdapter().metadata())
    assert preflight["maximum_api_calls"] == 20
    assert preflight["maximum_duration_seconds"] == 12_006.0
    assert preflight["maximum_output_tokens"] == 61_440
    assert preflight["estimate_assumptions"][-1] == expected_backoff_assumption

    submissions = (
        service.submit(
            _BENCHMARK_ID,
            selection,
            target_first,
            judge_connector=judge_first,
            run_id="run-role-base",
        ),
        service.submit(
            _BENCHMARK_ID,
            selection,
            target_second,
            judge_connector=judge_first,
            run_id="run-role-target-changed",
        ),
        service.submit(
            _BENCHMARK_ID,
            selection,
            target_first,
            judge_connector=judge_second,
            run_id="run-role-judge-changed",
        ),
    )

    assert len({submitted.run.protocol_hash for submitted in submissions}) == 3
    expected_connectors = (
        (target_first, judge_first),
        (target_second, judge_first),
        (target_first, judge_second),
    )
    for submitted, (target, judge) in zip(submissions, expected_connectors, strict=True):
        payload = runtime.store.get_job(submitted.job_id).payload
        target_payload = target.model_dump(mode="json")
        judge_payload = judge.model_dump(mode="json")
        assert payload["connector"] == target_payload
        assert payload["judge_connector"] == judge_payload
        assert payload["connector"] != payload["judge_connector"]
        assert submitted.estimate.maximum_model_calls == 20
        assert submitted.estimate.maximum_duration_seconds == 12_006.0
        assert submitted.estimate.maximum_output_tokens == 61_440
        assert submitted.estimate.assumptions[-1] == expected_backoff_assumption
        assert payload["protocol"]["retries"] == 4
        adapter_configuration = payload["protocol"]["adapter_configuration"]
        assert adapter_configuration["model_connector"] == target_payload
        assert adapter_configuration["model_connectors"] == {
            "judge": judge_payload,
            "target": target_payload,
        }
        assert adapter_configuration["planning_evidence"]["estimate"] == (
            submitted.estimate.model_dump(mode="json")
        )
        assert submitted.run.protocol_hash == protocol_hash(
            ProtocolSignature.model_validate(payload["protocol"])
        )
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        assert target_secret not in serialized
        assert judge_secret not in serialized

    for path in (tmp_path / "state").rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            assert target_secret.encode() not in content
            assert judge_secret.encode() not in content


def _write_approved_sample_snapshot(tmp_path: Path) -> tuple[Path, str]:
    template = json.loads(_FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    rows = []
    for ordinal in range(2_500):
        row = dict(template)
        row["id"] = f"synthetic-approved-{ordinal:04d}"
        row["question"] = f"Synthetic approved sample {ordinal}: answer with 7."
        row["image"] = None
        rows.append(row)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    path = tmp_path / "approved-synthetic-hle.jsonl"
    path.write_text(payload, encoding="utf-8")
    return path, hashlib.sha256(payload.encode()).hexdigest()


def test_hle_three_item_sample_renders_nonterminating_accuracy(tmp_path: Path):
    dataset_path, dataset_sha256 = _write_approved_sample_snapshot(tmp_path)
    runtime = EvaluationRuntime(tmp_path / "state")
    service = BenchmarkService(runtime)
    selection = RunSelection(
        profile="official-latest",
        mode=RunMode.SAMPLE,
        target_model=_TARGET_MODEL,
        judge_model=_JUDGE_MODEL,
        limit=3,
        dataset_path=str(dataset_path),
        dataset_sha256=dataset_sha256,
        accepted_access=True,
    )
    target_config = ConnectorConfig(
        kind="openai",
        endpoint="http://127.0.0.1:9/v1",
        max_retries=0,
    )
    judge_config = ConnectorConfig(
        kind="openai",
        endpoint="http://127.0.0.1:10/v1",
        max_retries=0,
    )
    submitted = service.submit(
        _BENCHMARK_ID,
        selection,
        target_config,
        judge_connector=judge_config,
        run_id="run-hle-one-of-three",
    )
    adapter = HumanitysLastExamAdapter()

    def connector_factory(
        _config: ConnectorConfig,
        plan: AdapterRunPlan,
        worker_runtime: EvaluationRuntime,
        role: ModelRole,
    ):
        return (
            FakeOpenAIConnector(
                adapter.smoke_connector_results(plan, role),
                secret_registry=worker_runtime.secret_registry,
            ),
            lambda: None,
        )

    assert (
        run_worker_once(
            runtime,
            worker_id="worker-hle-one-of-three",
            connector_factory=connector_factory,
        )
        == "run-hle-one-of-three"
    )

    run = runtime.store.get_run("run-hle-one-of-three").run
    assert run.state is RunState.COMPLETED
    assert (run.completed_count, run.failed_count) == (3, 0)
    assert runtime.store.get_job(submitted.job_id).status == "completed"
    metrics = runtime.artifacts.read_json("run-hle-one-of-three", "metrics.json")
    primary = next(metric for metric in metrics if metric["primary"])
    assert primary["numerator"] == 1
    assert primary["denominator"] == 3
    assert primary["value"] == pytest.approx(100.0 / 3.0)
    report = runtime.artifacts.read_json("run-hle-one-of-three", "report.json")
    report_primary = next(metric for metric in report["metrics"] if metric["primary"])
    assert report_primary["value"] == pytest.approx(100.0 / 3.0)
    assert render_saved_report(runtime, run.run_id).metrics[0].value == pytest.approx(100.0 / 3.0)
