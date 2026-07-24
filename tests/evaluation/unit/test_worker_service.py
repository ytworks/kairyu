import hashlib
import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import kairyu.evaluation.worker as worker_module
from kairyu.evaluation.adapters.base import (
    AdapterItem,
    AdapterRunPlan,
    ItemResult,
    ModelRole,
    RunSelection,
)
from kairyu.evaluation.artifacts import RUN_ARTIFACT_FILES
from kairyu.evaluation.connectors import (
    ConnectorError,
    ConnectorErrorCode,
    ConnectorResponse,
    ConnectorResult,
    ConnectorUsage,
    ModelRequest,
)
from kairyu.evaluation.protocol import protocol_hash
from kairyu.evaluation.reporting import RunManifest, UsageEvidence
from kairyu.evaluation.safety import SecretSafetyError, SecretValueRegistry
from kairyu.evaluation.schemas import (
    ItemState,
    ProtocolSignature,
    RunItem,
    RunMode,
    RunState,
)
from kairyu.evaluation.service import (
    BenchmarkService,
    ConnectorConfig,
    EvaluationRuntime,
)
from kairyu.evaluation.worker import render_saved_report, run_worker_once

_RAW_SMOKE_QUESTIONS = (
    "A fictional moon completes one orbit every 12 local days. "
    "How many orbits does it complete in 36 local days?",
    "In a made-up laboratory notation, ZIM means add two. "
    "Starting from five, what does one ZIM operation produce?",
)


class _SimulatedCrash(BaseException):
    pass


class _MutableClock:
    def __init__(self) -> None:
        self.value = datetime.now(UTC) + timedelta(minutes=1)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **delta: float) -> None:
        self.value += timedelta(**delta)


class _ScriptedConnector:
    def __init__(
        self,
        plan: AdapterRunPlan,
        *,
        error_ordinals: frozenset[int] = frozenset(),
        invalid_answer_ordinals: frozenset[int] = frozenset(),
    ) -> None:
        self._targets = {
            f"gpqa-{item.ordinal}-{item.input_sha256[:16]}": item.target for item in plan.items
        }
        self._error_ordinals = error_ordinals
        self._invalid_answer_ordinals = invalid_answer_ordinals
        self.requests: list[ModelRequest] = []

    def complete(
        self,
        request: ModelRequest,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        max_attempts: int | None = None,
    ) -> ConnectorResult:
        if cancel_requested is not None and cancel_requested():
            return _connector_error(request, ConnectorErrorCode.CANCELLED)
        self.requests.append(request)
        ordinal = int(request.request_id.split("-", 2)[1])
        if ordinal in self._error_ordinals:
            return _connector_error(request, ConnectorErrorCode.RATE_LIMIT)
        target = self._targets[request.request_id]
        content = (
            "Offline synthetic response without a labelled answer."
            if ordinal in self._invalid_answer_ordinals
            else f"Offline synthetic response.\nANSWER: {target}"
        )
        return ConnectorResult(
            response=ConnectorResponse(
                request_id=request.request_id,
                content=content,
                finish_reason="stop",
                provider_request_id=f"offline-{ordinal}",
                provider_model=request.model,
                usage=ConnectorUsage(
                    prompt_tokens=7,
                    completion_tokens=3,
                    total_tokens=10,
                ),
                latency_seconds=0.0,
                attempts=1,
            )
        )


class _ConnectorFactory:
    def __init__(
        self,
        *,
        error_ordinals: frozenset[int] = frozenset(),
        invalid_answer_ordinals: frozenset[int] = frozenset(),
    ) -> None:
        self.error_ordinals = error_ordinals
        self.invalid_answer_ordinals = invalid_answer_ordinals
        self.connectors: list[_ScriptedConnector] = []
        self.roles: list[ModelRole] = []

    def __call__(
        self,
        _config: ConnectorConfig,
        plan: AdapterRunPlan,
        _runtime: EvaluationRuntime,
        role: ModelRole,
    ):
        self.roles.append(role)
        connector = _ScriptedConnector(
            plan,
            error_ordinals=self.error_ordinals,
            invalid_answer_ordinals=self.invalid_answer_ordinals,
        )
        self.connectors.append(connector)
        return connector, lambda: None

    @property
    def requests(self) -> tuple[ModelRequest, ...]:
        return tuple(request for connector in self.connectors for request in connector.requests)


class _CrashDuringRequestConnector:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def complete(
        self,
        request: ModelRequest,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        max_attempts: int | None = None,
    ) -> ConnectorResult:
        assert cancel_requested is not None
        assert cancel_requested() is False
        self.requests.append(request)
        raise _SimulatedCrash


class _CancelDuringRequestConnector:
    def __init__(
        self,
        service: BenchmarkService,
        run_id: str,
        *,
        provider_request_id: str | None = None,
    ) -> None:
        self._service = service
        self._run_id = run_id
        self._provider_request_id = provider_request_id
        self.requests: list[ModelRequest] = []

    def complete(
        self,
        request: ModelRequest,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        max_attempts: int | None = None,
    ) -> ConnectorResult:
        self.requests.append(request)
        cancelled = self._service.cancel(self._run_id)
        assert cancelled.cancel_requested is True
        assert cancel_requested is not None
        assert cancel_requested() is True
        return _connector_error(
            request,
            ConnectorErrorCode.CANCELLED,
            provider_request_id=self._provider_request_id,
        )


class _UnexpectedCancelledConnector:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def complete(
        self,
        request: ModelRequest,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        max_attempts: int | None = None,
    ) -> ConnectorResult:
        assert cancel_requested is not None
        assert cancel_requested() is False
        self.requests.append(request)
        return _connector_error(request, ConnectorErrorCode.CANCELLED)


class _ExceptionalCancelAfterCheckpointConnector(_ScriptedConnector):
    def __init__(
        self,
        plan: AdapterRunPlan,
        service: BenchmarkService,
        run_id: str,
    ) -> None:
        super().__init__(plan)
        self._service = service
        self._run_id = run_id

    def complete(
        self,
        request: ModelRequest,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        max_attempts: int | None = None,
    ) -> ConnectorResult:
        ordinal = int(request.request_id.split("-", 2)[1])
        if ordinal == 1:
            self.requests.append(request)
            cancelled = self._service.cancel(self._run_id)
            assert cancelled.cancel_requested is True
            raise RuntimeError("synthetic exception after durable cancellation")
        return super().complete(
            request,
            cancel_requested=cancel_requested,
            max_attempts=max_attempts,
        )


class _ExceptionalCancelAfterFailedCheckpointConnector(_ScriptedConnector):
    def __init__(
        self,
        plan: AdapterRunPlan,
        service: BenchmarkService,
        run_id: str,
    ) -> None:
        super().__init__(plan, error_ordinals=frozenset({0}))
        self._service = service
        self._run_id = run_id

    def complete(
        self,
        request: ModelRequest,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        max_attempts: int | None = None,
    ) -> ConnectorResult:
        ordinal = int(request.request_id.split("-", 2)[1])
        if ordinal == 1:
            self.requests.append(request)
            cancelled = self._service.cancel(self._run_id)
            assert cancelled.cancel_requested is True
            raise RuntimeError("synthetic exception after failed checkpoint and cancellation")
        return super().complete(
            request,
            cancel_requested=cancel_requested,
            max_attempts=max_attempts,
        )


def _connector_error(
    request: ModelRequest,
    code: ConnectorErrorCode,
    *,
    provider_request_id: str | None = None,
) -> ConnectorResult:
    return ConnectorResult(
        error=ConnectorError(
            request_id=request.request_id,
            code=code,
            detail="controlled offline failure",
            retryable=code is ConnectorErrorCode.RATE_LIMIT,
            attempts=1,
            provider_request_id=provider_request_id,
            latency_seconds=0.0,
        )
    )


def _submit(
    runtime: EvaluationRuntime,
    run_id: str,
) -> tuple[BenchmarkService, str]:
    service = BenchmarkService(runtime)
    submitted = service.submit(
        "gpqa-diamond",
        RunSelection(
            profile="smoke",
            mode=RunMode.SMOKE,
            target_model="offline-fake-model",
        ),
        ConnectorConfig(kind="fake"),
        run_id=run_id,
    )
    assert submitted.estimated_model_calls == 2
    assert submitted.official_eligible is False
    return service, submitted.job_id


def _artifact_bytes(
    runtime: EvaluationRuntime,
    run_id: str,
    relative_path: str,
) -> bytes:
    return runtime.artifacts.read_bytes(run_id, relative_path)


def _assert_typed_manifest(
    runtime: EvaluationRuntime,
    run_id: str,
    *,
    expected_input_tokens: int,
    expected_output_tokens: int,
    expected_usage_status: str,
    expected_provider_models: tuple[str, ...],
) -> RunManifest:
    payload = runtime.artifacts.read_json(run_id, "manifest.json")
    assert set(payload) == {
        "benchmark",
        "container",
        "estimates",
        "generation",
        "git",
        "observed_provider_models",
        "resources",
        "run",
        "schema_version",
        "software",
        "upstream",
        "usage",
    }
    manifest = RunManifest.model_validate(payload)
    usage = UsageEvidence.model_validate(runtime.artifacts.read_json(run_id, "usage.json"))
    protocol = runtime.artifacts.read_json(run_id, "protocol.json")
    assert manifest.run == runtime.store.get_run(run_id).run
    assert manifest.usage == usage
    assert usage.input_tokens == expected_input_tokens
    assert usage.output_tokens == expected_output_tokens
    assert usage.total_latency_seconds == 0.0
    assert usage.measurement_status == expected_usage_status
    assert usage.actual_cost_usd is None
    assert usage.actual_cost_unavailable_reason
    assert manifest.benchmark.benchmark_version == protocol["benchmark_version"]
    assert manifest.benchmark.dataset_id == protocol["dataset_id"]
    assert manifest.benchmark.dataset_revision == protocol["dataset_revision"]
    assert manifest.benchmark.protocol_hash == manifest.run.protocol_hash
    assert manifest.upstream.repository.value == "https://github.com/modelscope/evalscope"
    assert manifest.upstream.repository.unavailable_reason is None
    assert manifest.upstream.commit.value == protocol["harness_commit"]
    assert manifest.container.image_digest.value is None
    assert manifest.container.image_digest.unavailable_reason
    assert manifest.generation.temperature.value == 0.0
    assert manifest.generation.top_p.value == 1.0
    assert manifest.generation.max_input_tokens.value is None
    assert manifest.generation.max_input_tokens.unavailable_reason
    assert manifest.generation.max_output_tokens.value == 1_024
    assert manifest.generation.seed.value == 42
    assert manifest.generation.retries == 0
    assert manifest.generation.tools == ()
    assert manifest.generation.scaffold.value is None
    assert manifest.generation.scaffold.unavailable_reason
    assert manifest.generation.max_turns.value is None
    assert manifest.generation.max_turns.unavailable_reason
    assert manifest.generation.timeout_seconds.value == 120.0
    planning = protocol["adapter_configuration"]["planning_evidence"]
    resources = planning["resources"]
    estimate = planning["estimate"]
    execution = planning["execution"]
    assert manifest.resources.status == "known"
    assert manifest.resources.cpu_cores.value == resources["cpu_cores"]
    assert manifest.resources.ram_bytes.value == resources["ram_bytes"]
    assert manifest.resources.disk_bytes.value == resources["disk_bytes"]
    assert manifest.resources.docker_required.value == resources["docker_required"]
    assert manifest.resources.network_policy.value == resources["network_policy"]
    assert manifest.estimates.model_calls.value == estimate["model_calls"]
    assert manifest.estimates.maximum_model_calls.value == estimate["maximum_model_calls"]
    assert manifest.estimates.estimated_input_tokens.value == estimate["estimated_input_tokens"]
    assert manifest.estimates.maximum_output_tokens.value == estimate["maximum_output_tokens"]
    assert manifest.estimates.estimated_duration_seconds.value is None
    assert manifest.estimates.estimated_duration_seconds.unavailable_reason
    assert manifest.estimates.maximum_duration_seconds.value == estimate["maximum_duration_seconds"]
    assert manifest.estimates.estimated_cost_usd.value is None
    assert manifest.estimates.estimated_cost_usd.unavailable_reason
    assert manifest.estimates.assumptions == tuple(estimate["assumptions"])
    assert manifest.container.image_digest.value == execution["container_image_digest"]
    assert manifest.software.status == "partial"
    assert manifest.software.harness_name == protocol["harness_name"]
    assert manifest.software.harness_version == protocol["harness_version"]
    assert manifest.software.dependency_lock_sha256.value == protocol["dependency_lock_sha256"]
    compatibility_sha256 = protocol["adapter_configuration"]["compatibility_layer_sha256"]
    assert manifest.software.compatibility_layer_identity.value == (
        "kairyu.evaluation.adapters.gpqa_v181"
    )
    assert manifest.software.compatibility_layer_sha256.value == compatibility_sha256
    assert manifest.software.runtime_dependency_environment_status.value == "unresolved"
    assert manifest.software.runtime_dependency_environment_reason.value
    assert manifest.git.commit.value is None
    assert manifest.git.commit.unavailable_reason
    assert manifest.git.dirty.value is None
    assert manifest.git.dirty.unavailable_reason
    assert manifest.observed_provider_models == expected_provider_models
    return manifest


def _run_partial(
    runtime: EvaluationRuntime,
    run_id: str,
) -> BenchmarkService:
    service, _ = _submit(runtime, run_id)
    factory = _ConnectorFactory(error_ordinals=frozenset({1}))
    assert (
        run_worker_once(
            runtime,
            worker_id=f"worker-{run_id}",
            connector_factory=factory,
        )
        == run_id
    )
    assert runtime.store.get_run(run_id).run.state is RunState.PARTIAL
    assert [request.request_id.split("-", 2)[1] for request in factory.requests] == [
        "0",
        "1",
    ]
    return service


def _runtime_with_clock(tmp_path: Path) -> tuple[EvaluationRuntime, _MutableClock]:
    runtime = EvaluationRuntime(tmp_path / "state")
    clock = _MutableClock()
    runtime.store._clock = clock
    return runtime, clock


def test_offline_smoke_run_publishes_complete_deterministic_secret_free_evidence(
    tmp_path: Path,
):
    secret = "offline-only-secret-value-7319"
    runtime = EvaluationRuntime(
        tmp_path / "state",
        secret_registry=SecretValueRegistry((secret,)),
    )
    _, job_id = _submit(runtime, "run-smoke")

    assert run_worker_once(runtime, worker_id="worker-smoke") == "run-smoke"

    stored = runtime.store.get_run("run-smoke").run
    assert stored.state is RunState.COMPLETED
    assert (stored.completed_count, stored.failed_count, stored.skipped_count) == (
        2,
        0,
        0,
    )
    assert runtime.store.get_job(job_id).status == "completed"
    manifest = runtime.artifacts.read_json("run-smoke", "manifest.json")
    assert manifest["run"] == stored.model_dump(mode="json")
    _assert_typed_manifest(
        runtime,
        "run-smoke",
        expected_input_tokens=20,
        expected_output_tokens=10,
        expected_usage_status="complete",
        expected_provider_models=("offline-fake-model",),
    )
    artifact_items = tuple(
        RunItem.model_validate_json(line)
        for line in runtime.artifacts.read_bytes("run-smoke", "item_results.jsonl").splitlines()
    )
    assert artifact_items == tuple(
        stored_item.item for stored_item in runtime.store.list_run_items("run-smoke")
    )
    metrics = runtime.artifacts.read_json("run-smoke", "metrics.json")
    assert metrics == [
        {
            "denominator": 2,
            "dimensions": {"aggregation": "mean", "shots": 0},
            "display_name": "Accuracy",
            "higher_is_better": True,
            "name": "accuracy",
            "numerator": 1,
            "official_eligible": False,
            "primary": True,
            "run_id": "run-smoke",
            "scale": 100.0,
            "schema_version": 1,
            "unit": "percent",
            "value": 50.0,
        }
    ]

    artifact_paths = {
        artifact.relative_path for artifact in runtime.store.list_artifacts("run-smoke")
    }
    assert set(RUN_ARTIFACT_FILES) <= artifact_paths
    assert len([path for path in artifact_paths if path.startswith("upstream/checkpoints/")]) == 2
    before = {
        path: _artifact_bytes(runtime, "run-smoke", path)
        for path in ("report.json", "report.md", "report.html")
    }

    regenerated = render_saved_report(runtime, "run-smoke")

    assert regenerated.metrics[0].value == 50.0
    assert before == {
        path: _artifact_bytes(runtime, "run-smoke", path)
        for path in ("report.json", "report.md", "report.html")
    }
    report_payload = json.loads(before["report.json"])
    assert report_payload["unofficial"] is True
    assert report_payload["comparison_eligible"] is False
    assert report_payload["metrics"][0]["value"] == 50.0
    assert report_payload["target_model"] == "offline-fake-model"
    assert report_payload["judge_model"] is None
    assert report_payload["simulator_model"] is None
    assert report_payload["usage"]["input_tokens"] == 20
    assert report_payload["usage"]["output_tokens"] == 10
    assert report_payload["usage"]["actual_cost_usd"] is None
    assert report_payload["usage"]["actual_cost_unavailable_reason"]
    protocol_payload = runtime.artifacts.read_json("run-smoke", "protocol.json")
    assert report_payload["protocol"]["harness_commit"] == protocol_payload["harness_commit"]
    assert report_payload["protocol"]["retries"] == protocol_payload["retries"]
    assert report_payload["protocol"]["timeout_seconds"] == 120.0
    assert report_payload["protocol"]["reasoning_effort"] is None
    assert "Provider APIs may be nondeterministic" in report_payload["reproducibility_notice"]
    assert "## Usage / Cost" in before["report.md"].decode()
    assert "<h2>Usage / Cost</h2>" in before["report.html"].decode()
    assert "## Protocol Details" in before["report.md"].decode()
    assert "Provider APIs may be nondeterministic" in before["report.md"].decode()
    assert "<h2>Protocol Details</h2>" in before["report.html"].decode()
    assert "Provider APIs may be nondeterministic" in before["report.html"].decode()

    forbidden = (
        secret.encode(),
        *[question.encode() for question in _RAW_SMOKE_QUESTIONS],
    )
    for path in (tmp_path / "state").rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            assert all(value not in content for value in forbidden)


def test_invalid_answer_remains_scored_and_is_reported_as_error(tmp_path: Path):
    runtime = EvaluationRuntime(tmp_path / "state")
    _submit(runtime, "run-invalid-answer")
    factory = _ConnectorFactory(invalid_answer_ordinals=frozenset({1}))

    assert (
        run_worker_once(
            runtime,
            worker_id="worker-invalid-answer",
            connector_factory=factory,
        )
        == "run-invalid-answer"
    )

    run = runtime.store.get_run("run-invalid-answer").run
    assert run.state is RunState.COMPLETED
    assert (run.completed_count, run.failed_count) == (2, 0)
    metrics = runtime.artifacts.read_json("run-invalid-answer", "metrics.json")
    assert metrics[0]["numerator"] == 1
    assert metrics[0]["denominator"] == 2
    assert metrics[0]["value"] == 50.0
    errors = [
        json.loads(line)
        for line in runtime.artifacts.read_bytes(
            "run-invalid-answer",
            "errors.jsonl",
        ).splitlines()
    ]
    assert errors == [
        {
            "error_class": "invalid_answer",
            "item_id": run.selected_item_ids[1],
            "ordinal": 1,
        }
    ]
    report = runtime.artifacts.read_json("run-invalid-answer", "report.json")
    assert report["errors"] == [{"count": 1, "error_class": "invalid_answer"}]
    assert [item["state"] for item in report["items"]] == ["completed", "completed"]
    assert [item["score"] for item in report["items"]] == [1.0, 0.0]

    regenerated = render_saved_report(runtime, "run-invalid-answer")
    assert [(error.error_class, error.count) for error in regenerated.errors] == [
        ("invalid_answer", 1)
    ]


def test_queued_cancel_always_publishes_all_three_reports(tmp_path: Path):
    runtime = EvaluationRuntime(tmp_path / "state")
    service, job_id = _submit(runtime, "run-cancelled")

    cancelled = service.cancel("run-cancelled")

    assert cancelled.status == "cancelled"
    assert runtime.store.get_job(job_id).status == "cancelled"
    stored = runtime.store.get_run("run-cancelled").run
    assert stored.state is RunState.CANCELLED
    assert stored.termination_reason == "cancel_requested"
    assert {
        artifact.relative_path for artifact in runtime.store.list_artifacts("run-cancelled")
    } == set(RUN_ARTIFACT_FILES)
    report = runtime.artifacts.read_json("run-cancelled", "report.json")
    manifest = runtime.artifacts.read_json("run-cancelled", "manifest.json")
    assert manifest["run"] == stored.model_dump(mode="json")
    _assert_typed_manifest(
        runtime,
        "run-cancelled",
        expected_input_tokens=0,
        expected_output_tokens=0,
        expected_usage_status="complete",
        expected_provider_models=(),
    )
    assert runtime.artifacts.read_bytes("run-cancelled", "item_results.jsonl") == b""
    assert report["state"] == "cancelled"
    assert report["counts"]["selected"] == 2
    assert report["counts"]["reported_items"] == 0
    assert report["metrics"][0]["value"] is None
    assert report["metrics"][0]["denominator"] == 0
    assert report["usage"]["input_tokens"] == 0
    assert report["usage"]["output_tokens"] == 0
    assert report["usage"]["actual_cost_usd"] is None
    assert report["usage"]["actual_cost_unavailable_reason"]

    before = _artifact_bytes(runtime, "run-cancelled", "report.json")
    service.cancel("run-cancelled")
    assert _artifact_bytes(runtime, "run-cancelled", "report.json") == before


@pytest.mark.parametrize(
    ("error_ordinals", "expected_state", "completed", "failed", "metric_value"),
    (
        (frozenset({1}), RunState.PARTIAL, 1, 1, 100.0),
        (frozenset({0, 1}), RunState.FAILED, 0, 2, None),
    ),
)
def test_structured_connector_errors_produce_partial_or_failed_reports(
    tmp_path: Path,
    error_ordinals: frozenset[int],
    expected_state: RunState,
    completed: int,
    failed: int,
    metric_value: float | None,
):
    runtime = EvaluationRuntime(tmp_path / "state")
    _, job_id = _submit(runtime, "run-errors")
    factory = _ConnectorFactory(error_ordinals=error_ordinals)

    assert (
        run_worker_once(
            runtime,
            worker_id="worker-errors",
            connector_factory=factory,
        )
        == "run-errors"
    )

    stored = runtime.store.get_run("run-errors").run
    assert stored.state is expected_state
    assert stored.partial is (expected_state is RunState.PARTIAL)
    assert (stored.completed_count, stored.failed_count) == (completed, failed)
    assert runtime.store.get_job(job_id).status == "failed"
    report = runtime.artifacts.read_json("run-errors", "report.json")
    assert report["state"] == expected_state.value
    assert report["metrics"][0]["value"] == metric_value
    assert report["errors"] == [{"count": failed, "error_class": "rate_limit"}]
    stored_items = tuple(
        stored_item.item for stored_item in runtime.store.list_run_items("run-errors")
    )
    assert all(item.checkpoint_relative_path is not None for item in stored_items)
    failed_results = tuple(
        ItemResult.model_validate_json(
            runtime.artifacts.read_bytes(
                item.checkpoint_source_run_id,
                item.checkpoint_relative_path,
            )
        )
        for item in stored_items
        if item.state is ItemState.FAILED
    )
    assert {result.error_class for result in failed_results} == {"rate_limit"}
    assert {result.target_attempts for result in failed_results} == {1}
    assert all(result.target_request_sha256 is not None for result in failed_results)
    predictions = tuple(
        json.loads(line)
        for line in runtime.artifacts.read_bytes("run-errors", "predictions.jsonl").splitlines()
    )
    assert len(predictions) == 2
    assert all(prediction["target_attempts"] == 1 for prediction in predictions)
    assert all(prediction["target_request_sha256"] is not None for prediction in predictions)
    assert factory.roles == [ModelRole.TARGET]
    assert {
        artifact.relative_path for artifact in runtime.store.list_artifacts("run-errors")
    } >= set(RUN_ARTIFACT_FILES)


def test_connector_cancel_without_durable_intent_is_a_failed_run(tmp_path: Path):
    runtime = EvaluationRuntime(tmp_path / "state")
    _, job_id = _submit(runtime, "run-unexpected-cancel")
    connector = _UnexpectedCancelledConnector()

    assert (
        run_worker_once(
            runtime,
            worker_id="worker-unexpected-cancel",
            connector_factory=lambda *_args: (connector, lambda: None),
        )
        == "run-unexpected-cancel"
    )

    run = runtime.store.get_run("run-unexpected-cancel").run
    assert run.state is RunState.FAILED
    assert run.termination_reason == "one_or_more_items_failed"
    assert run.failed_count == 2
    assert runtime.store.get_job(job_id).status == "failed"
    items = [stored.item for stored in runtime.store.list_run_items("run-unexpected-cancel")]
    assert [item.state for item in items] == [ItemState.FAILED, ItemState.FAILED]
    assert {item.error_class for item in items} == {"unexpected_connector_cancellation"}
    report = runtime.artifacts.read_json("run-unexpected-cancel", "report.json")
    assert report["state"] == "failed"
    assert report["errors"] == [{"count": 2, "error_class": "unexpected_connector_cancellation"}]


def test_resume_reuses_only_verified_completed_checkpoint(tmp_path: Path):
    runtime = EvaluationRuntime(tmp_path / "state")
    service = _run_partial(runtime, "run-source")
    source_items = tuple(stored.item for stored in runtime.store.list_run_items("run-source"))
    assert [item.state for item in source_items] == [ItemState.COMPLETED, ItemState.FAILED]
    assert all(item.checkpoint_relative_path is not None for item in source_items)

    resumed = service.resume("run-source", new_run_id="run-successor")
    factory = _ConnectorFactory()
    assert resumed.run.run.resumed_from_run_id == "run-source"
    assert (
        run_worker_once(
            runtime,
            worker_id="worker-successor",
            connector_factory=factory,
        )
        == "run-successor"
    )

    assert len(factory.requests) == 1
    assert factory.requests[0].request_id.startswith("gpqa-1-")
    stored = runtime.store.get_run("run-successor").run
    assert stored.state is RunState.COMPLETED
    assert stored.completed_count == 2
    items = [stored_item.item for stored_item in runtime.store.list_run_items("run-successor")]
    assert [item.state for item in items] == [
        ItemState.COMPLETED,
        ItemState.COMPLETED,
    ]
    assert items[0].checkpoint_source_run_id == "run-source"
    assert items[1].checkpoint_source_run_id == "run-successor"
    report = runtime.artifacts.read_json("run-successor", "report.json")
    assert report["metrics"][0]["value"] == 100.0


def test_resume_reexecutes_a_tampered_checkpoint(tmp_path: Path):
    runtime = EvaluationRuntime(tmp_path / "state")
    service = _run_partial(runtime, "run-source")
    source_item = runtime.store.list_run_items("run-source")[0].item
    assert source_item.checkpoint_relative_path is not None
    runtime.artifacts.path_for(
        "run-source",
        source_item.checkpoint_relative_path,
    ).write_bytes(b'{"tampered":true}')

    service.resume("run-source", new_run_id="run-successor")
    factory = _ConnectorFactory()
    assert (
        run_worker_once(
            runtime,
            worker_id="worker-successor",
            connector_factory=factory,
        )
        == "run-successor"
    )

    assert len(factory.requests) == 2
    assert runtime.store.get_run("run-successor").run.state is RunState.COMPLETED
    successor_items = [
        stored_item.item for stored_item in runtime.store.list_run_items("run-successor")
    ]
    assert all(item.checkpoint_source_run_id == "run-successor" for item in successor_items)


def test_preflight_failure_releases_job_for_immediate_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = EvaluationRuntime(tmp_path / "state")
    _, job_id = _submit(runtime, "run-preflight")

    def fail_preflight(_payload, **_kwargs):
        raise RuntimeError("synthetic preflight failure")

    monkeypatch.setattr(
        worker_module,
        "rebuild_plan_from_job",
        fail_preflight,
    )
    with pytest.raises(RuntimeError, match="synthetic preflight failure"):
        run_worker_once(runtime, worker_id="worker-preflight")

    job = runtime.store.get_job(job_id)
    assert job.status == "queued"
    assert job.lease_owner is None
    assert job.lease_expires_at is None
    assert runtime.store.get_run("run-preflight").run.state is RunState.PENDING

    monkeypatch.undo()
    assert run_worker_once(runtime, worker_id="worker-retry") == "run-preflight"
    assert runtime.store.get_run("run-preflight").run.state is RunState.COMPLETED


def test_permanent_preflight_poison_is_blocked_after_three_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = EvaluationRuntime(tmp_path / "state")
    _, job_id = _submit(runtime, "run-poison")

    def permanent_failure(_payload, **_kwargs):
        raise ValueError("synthetic permanent preflight failure")

    monkeypatch.setattr(worker_module, "rebuild_plan_from_job", permanent_failure)
    for attempt in range(1, 4):
        with pytest.raises(ValueError, match="synthetic permanent preflight failure"):
            run_worker_once(runtime, worker_id=f"worker-poison-{attempt}")
        job = runtime.store.get_job(job_id)
        assert job.attempts == attempt
        assert job.status == ("queued" if attempt < 3 else "failed")

    run = runtime.store.get_run("run-poison").run
    assert run.state is RunState.BLOCKED
    assert run.termination_reason == "worker_retry_exhausted"
    assert run.partial is False
    assert run_worker_once(runtime, worker_id="worker-after-poison") is None


@pytest.mark.parametrize(
    ("boundary", "expected_state"),
    (
        ("preparing", RunState.PREPARING),
        ("ready", RunState.READY),
        ("running", RunState.RUNNING),
    ),
)
def test_expired_lease_recovers_from_each_run_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    expected_state: RunState,
):
    runtime, clock = _runtime_with_clock(tmp_path)
    _, job_id = _submit(runtime, "run-crash")

    def crash(*_args, **_kwargs):
        raise _SimulatedCrash

    original_compare = runtime.store.compare_and_set_run_state

    def crash_before_running(*args, **kwargs):
        if kwargs.get("new_state") is RunState.RUNNING:
            raise _SimulatedCrash
        return original_compare(*args, **kwargs)

    connector_factory = None
    with monkeypatch.context() as scoped:
        if boundary == "preparing":
            scoped.setattr(runtime.store, "complete_preparation", crash)
        elif boundary == "ready":
            scoped.setattr(
                runtime.store,
                "compare_and_set_run_state",
                crash_before_running,
            )
        else:
            connector_factory = crash
        with pytest.raises(_SimulatedCrash):
            run_worker_once(
                runtime,
                worker_id="worker-crashed",
                lease_seconds=10,
                connector_factory=connector_factory,
            )

    crashed_job = runtime.store.get_job(job_id)
    assert crashed_job.status == "leased"
    assert crashed_job.lease_owner == "worker-crashed"
    assert crashed_job.attempts == 1
    assert runtime.store.get_run("run-crash").run.state is expected_state

    clock.advance(seconds=11)
    assert (
        run_worker_once(
            runtime,
            worker_id="worker-recovery",
            lease_seconds=10,
        )
        == "run-crash"
    )

    recovered_job = runtime.store.get_job(job_id)
    assert recovered_job.status == "completed"
    assert recovered_job.attempts == 2
    recovered = runtime.store.get_run("run-crash").run
    assert recovered.state is RunState.COMPLETED
    assert recovered.completed_count == 2
    assert runtime.artifacts.path_for("run-crash", "report.json").is_file()


def test_expired_lease_recovers_terminal_report_after_renderer_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime, clock = _runtime_with_clock(tmp_path)
    _, job_id = _submit(runtime, "run-report-crash")

    def crash_renderer(*_args, **_kwargs):
        raise RuntimeError("synthetic report renderer crash")

    with monkeypatch.context() as scoped:
        scoped.setattr(worker_module, "render_saved_report", crash_renderer)
        with pytest.raises(RuntimeError, match="synthetic report renderer crash"):
            run_worker_once(
                runtime,
                worker_id="worker-report-crashed",
                lease_seconds=10,
            )

    crashed_run = runtime.store.get_run("run-report-crash").run
    crashed_job = runtime.store.get_job(job_id)
    assert crashed_run.state is RunState.COMPLETED
    assert crashed_run.completed_count == 2
    assert crashed_job.status == "leased"
    assert crashed_job.lease_owner == "worker-report-crashed"
    artifact_paths = {
        artifact.relative_path for artifact in runtime.store.list_artifacts("run-report-crash")
    }
    assert "manifest.json" in artifact_paths
    assert not {"report.json", "report.md", "report.html"} & artifact_paths

    clock.advance(seconds=11)
    assert (
        run_worker_once(
            runtime,
            worker_id="worker-report-recovery",
            lease_seconds=10,
        )
        == "run-report-crash"
    )

    recovered_job = runtime.store.get_job(job_id)
    assert recovered_job.status == "completed"
    assert recovered_job.attempts == 2
    assert set(RUN_ARTIFACT_FILES) <= {
        artifact.relative_path for artifact in runtime.store.list_artifacts("run-report-crash")
    }
    assert runtime.artifacts.read_json("run-report-crash", "report.json")["state"] == "completed"


def test_expired_lease_reexecutes_running_item_at_least_once(tmp_path: Path):
    runtime, clock = _runtime_with_clock(tmp_path)
    _, job_id = _submit(runtime, "run-item-crash")
    crashed_connector = _CrashDuringRequestConnector()

    with pytest.raises(_SimulatedCrash):
        run_worker_once(
            runtime,
            worker_id="worker-crashed",
            lease_seconds=10,
            connector_factory=lambda *_args: (
                crashed_connector,
                lambda: None,
            ),
        )

    crashed_items = [stored.item for stored in runtime.store.list_run_items("run-item-crash")]
    assert [item.state for item in crashed_items] == [
        ItemState.RUNNING,
        ItemState.PENDING,
    ]
    assert len(crashed_connector.requests) == 1
    assert crashed_connector.requests[0].request_id.startswith("gpqa-0-")
    assert runtime.store.get_job(job_id).status == "leased"

    clock.advance(seconds=11)
    recovery_factory = _ConnectorFactory()
    assert (
        run_worker_once(
            runtime,
            worker_id="worker-recovery",
            lease_seconds=10,
            connector_factory=recovery_factory,
        )
        == "run-item-crash"
    )

    assert [int(request.request_id.split("-", 2)[1]) for request in recovery_factory.requests] == [
        0,
        1,
    ]
    assert runtime.store.get_job(job_id).attempts == 2
    recovered = runtime.store.get_run("run-item-crash").run
    assert recovered.state is RunState.COMPLETED
    assert recovered.completed_count == 2
    assert all(
        stored.item.state is ItemState.COMPLETED
        for stored in runtime.store.list_run_items("run-item-crash")
    )


def test_mid_item_cancel_persists_one_consistent_cancelled_snapshot(tmp_path: Path):
    runtime = EvaluationRuntime(tmp_path / "state")
    service, job_id = _submit(runtime, "run-mid-cancel")
    connector = _CancelDuringRequestConnector(service, "run-mid-cancel")

    assert (
        run_worker_once(
            runtime,
            worker_id="worker-cancel",
            connector_factory=lambda *_args: (connector, lambda: None),
        )
        == "run-mid-cancel"
    )

    stored = runtime.store.get_run("run-mid-cancel").run
    assert stored.state is RunState.CANCELLED
    assert stored.finished_at is not None
    assert stored.termination_reason == "cancel_requested"
    job = runtime.store.get_job(job_id)
    assert job.status == "cancelled"
    assert job.cancel_requested is True
    assert len(connector.requests) == 1
    stored_items = tuple(item.item for item in runtime.store.list_run_items("run-mid-cancel"))
    assert [item.state for item in stored_items] == [
        ItemState.CANCELLED,
        ItemState.PENDING,
    ]
    cancelled_item = stored_items[0]
    assert cancelled_item.error_class == "cancelled"
    assert cancelled_item.checkpoint_relative_path is not None
    assert cancelled_item.checkpoint_sha256 is not None
    assert cancelled_item.checkpoint_source_run_id == "run-mid-cancel"
    checkpoint = runtime.artifacts.read_bytes(
        "run-mid-cancel",
        cancelled_item.checkpoint_relative_path,
    )
    assert hashlib.sha256(checkpoint).hexdigest() == cancelled_item.checkpoint_sha256
    cancelled_result = ItemResult.model_validate_json(checkpoint)
    assert cancelled_result.error_class == "cancelled"
    assert cancelled_result.target_attempts == 1
    assert cancelled_result.target_request_sha256 is not None
    assert worker_module._checkpoint_is_verified(runtime, cancelled_item)
    wrong_ordinal = cancelled_item.model_copy(update={"ordinal": cancelled_item.ordinal + 1})
    assert not worker_module._checkpoint_is_verified(runtime, wrong_ordinal)

    manifest = runtime.artifacts.read_json("run-mid-cancel", "manifest.json")
    report = runtime.artifacts.read_json("run-mid-cancel", "report.json")
    metrics = runtime.artifacts.read_json("run-mid-cancel", "metrics.json")
    assert manifest["run"] == stored.model_dump(mode="json")
    assert manifest["run"]["state"] == "cancelled"
    assert manifest["run"]["finished_at"] == stored.model_dump(mode="json")["finished_at"]
    assert manifest["run"]["termination_reason"] == "cancel_requested"
    assert report["state"] == "cancelled"
    assert (
        datetime.fromisoformat(report["evidence_as_of"].replace("Z", "+00:00"))
        == stored.finished_at
    )
    assert report["counts"]["cancelled"] == 1
    assert report["counts"]["pending"] == 1
    assert metrics[0]["value"] is None
    assert metrics[0]["denominator"] == 0
    predictions = tuple(
        json.loads(line)
        for line in runtime.artifacts.read_bytes("run-mid-cancel", "predictions.jsonl").splitlines()
    )
    assert len(predictions) == 1
    assert predictions[0]["item_id"] == cancelled_item.item_id
    assert predictions[0]["target_attempts"] == 1
    assert predictions[0]["target_request_sha256"] is not None
    errors = tuple(
        json.loads(line)
        for line in runtime.artifacts.read_bytes("run-mid-cancel", "errors.jsonl").splitlines()
    )
    assert errors == (
        {
            "error_class": "cancelled",
            "item_id": cancelled_item.item_id,
            "ordinal": cancelled_item.ordinal,
        },
    )
    usage = UsageEvidence.model_validate(
        runtime.artifacts.read_json("run-mid-cancel", "usage.json")
    )
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.measurement_status == "partial"
    artifact_items = tuple(
        RunItem.model_validate_json(line)
        for line in runtime.artifacts.read_bytes(
            "run-mid-cancel", "item_results.jsonl"
        ).splitlines()
    )
    assert artifact_items == tuple(
        stored_item.item for stored_item in runtime.store.list_run_items("run-mid-cancel")
    )
    assert set(RUN_ARTIFACT_FILES) <= {
        artifact.relative_path for artifact in runtime.store.list_artifacts("run-mid-cancel")
    }

    resumed = service.resume("run-mid-cancel", new_run_id="run-mid-cancel-resumed")
    recovery_factory = _ConnectorFactory()
    assert resumed.run.run.resumed_from_run_id == "run-mid-cancel"
    assert (
        run_worker_once(
            runtime,
            worker_id="worker-cancel-resumed",
            connector_factory=recovery_factory,
        )
        == "run-mid-cancel-resumed"
    )
    assert len(recovery_factory.requests) == 2
    assert runtime.store.get_run("run-mid-cancel-resumed").run.state is RunState.COMPLETED


def test_cancel_checkpoint_rejects_registered_secret_without_persisting_it(
    tmp_path: Path,
):
    secret = "cancel-provider-secret-value-7319"
    runtime = EvaluationRuntime(
        tmp_path / "state",
        secret_registry=SecretValueRegistry((secret,)),
    )
    service, job_id = _submit(runtime, "run-cancel-secret")
    connector = _CancelDuringRequestConnector(
        service,
        "run-cancel-secret",
        provider_request_id=secret,
    )

    with pytest.raises(SecretSafetyError) as raised:
        run_worker_once(
            runtime,
            worker_id="worker-cancel-secret",
            connector_factory=lambda *_args: (connector, lambda: None),
        )

    assert secret not in str(raised.value)
    assert runtime.store.get_job(job_id).status == "cancelled"
    assert runtime.store.get_run("run-cancel-secret").run.state is RunState.CANCELLED
    item = runtime.store.list_run_items("run-cancel-secret")[0].item
    assert item.state is ItemState.RUNNING
    assert item.checkpoint_relative_path is None
    assert secret.encode() not in runtime.store.database_path.read_bytes()
    for path in runtime.artifacts.root.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()


def test_exceptional_cancel_reconstructs_completed_checkpoint_evidence(
    tmp_path: Path,
):
    runtime = EvaluationRuntime(tmp_path / "state")
    service, job_id = _submit(runtime, "run-exceptional-cancel")
    connector: _ExceptionalCancelAfterCheckpointConnector | None = None

    def connector_factory(
        _config: ConnectorConfig,
        plan: AdapterRunPlan,
        _runtime: EvaluationRuntime,
        _role: ModelRole,
    ):
        nonlocal connector
        connector = _ExceptionalCancelAfterCheckpointConnector(
            plan,
            service,
            "run-exceptional-cancel",
        )
        return connector, lambda: None

    with pytest.raises(
        RuntimeError,
        match="synthetic exception after durable cancellation",
    ):
        run_worker_once(
            runtime,
            worker_id="worker-exceptional-cancel",
            connector_factory=connector_factory,
        )

    assert connector is not None
    assert len(connector.requests) == 2
    run = runtime.store.get_run("run-exceptional-cancel").run
    assert run.state is RunState.CANCELLED
    assert run.partial is True
    assert run.completed_count == 1
    assert runtime.store.get_job(job_id).status == "cancelled"
    predictions = [
        json.loads(line)
        for line in runtime.artifacts.read_bytes(
            "run-exceptional-cancel",
            "predictions.jsonl",
        ).splitlines()
    ]
    assert len(predictions) == 1
    assert predictions[0]["item_id"].startswith("gpqa-")
    assert predictions[0]["response_text"].startswith("Offline synthetic response")
    assert predictions[0]["provider_model"] == "offline-fake-model"
    typed_manifest = _assert_typed_manifest(
        runtime,
        "run-exceptional-cancel",
        expected_input_tokens=7,
        expected_output_tokens=3,
        expected_usage_status="partial",
        expected_provider_models=("offline-fake-model",),
    )
    assert typed_manifest.usage.measurement_unavailable_reasons
    metrics = runtime.artifacts.read_json("run-exceptional-cancel", "metrics.json")
    assert metrics[0]["value"] == 100.0
    assert metrics[0]["denominator"] == 1
    report = runtime.artifacts.read_json("run-exceptional-cancel", "report.json")
    assert report["state"] == "cancelled"
    assert report["counts"]["completed"] == 1
    assert report["metrics"][0]["value"] == 100.0
    assert report["usage"]["input_tokens"] == 7
    assert report["usage"]["output_tokens"] == 3
    assert report["usage"]["measurement_status"] == "partial"


def test_cancel_report_collects_failed_checkpoint_but_excludes_running_item(
    tmp_path: Path,
):
    runtime = EvaluationRuntime(tmp_path / "state")
    service, job_id = _submit(runtime, "run-failed-then-cancel")
    connector: _ExceptionalCancelAfterFailedCheckpointConnector | None = None

    def connector_factory(
        _config: ConnectorConfig,
        plan: AdapterRunPlan,
        _runtime: EvaluationRuntime,
        role: ModelRole,
    ):
        nonlocal connector
        assert role is ModelRole.TARGET
        connector = _ExceptionalCancelAfterFailedCheckpointConnector(
            plan,
            service,
            "run-failed-then-cancel",
        )
        return connector, lambda: None

    with pytest.raises(
        RuntimeError,
        match="synthetic exception after failed checkpoint and cancellation",
    ):
        run_worker_once(
            runtime,
            worker_id="worker-failed-then-cancel",
            connector_factory=connector_factory,
        )

    assert connector is not None
    assert len(connector.requests) == 2
    run = runtime.store.get_run("run-failed-then-cancel").run
    assert run.state is RunState.CANCELLED
    assert run.failed_count == 1
    assert runtime.store.get_job(job_id).status == "cancelled"
    items = tuple(stored.item for stored in runtime.store.list_run_items("run-failed-then-cancel"))
    assert [item.state for item in items] == [ItemState.FAILED, ItemState.RUNNING]
    assert items[0].checkpoint_relative_path is not None
    assert items[1].checkpoint_relative_path is None

    predictions = tuple(
        json.loads(line)
        for line in runtime.artifacts.read_bytes(
            "run-failed-then-cancel",
            "predictions.jsonl",
        ).splitlines()
    )
    assert len(predictions) == 1
    assert predictions[0]["item_id"] == items[0].item_id
    assert predictions[0]["target_attempts"] == 1
    assert predictions[0]["target_request_sha256"] is not None
    metrics = runtime.artifacts.read_json("run-failed-then-cancel", "metrics.json")
    assert metrics[0]["denominator"] == 0
    assert metrics[0]["value"] is None
    errors = tuple(
        json.loads(line)
        for line in runtime.artifacts.read_bytes(
            "run-failed-then-cancel",
            "errors.jsonl",
        ).splitlines()
    )
    assert errors == (
        {
            "error_class": "rate_limit",
            "item_id": items[0].item_id,
            "ordinal": 0,
        },
    )


@pytest.mark.parametrize(
    "relative_path",
    RUN_ARTIFACT_FILES,
)
def test_report_regeneration_rejects_tampered_registered_file(
    tmp_path: Path,
    relative_path: str,
):
    runtime = EvaluationRuntime(tmp_path / "state")
    _submit(runtime, "run-tampered")
    assert (
        run_worker_once(
            runtime,
            worker_id="worker-tampered",
        )
        == "run-tampered"
    )
    path = runtime.artifacts.path_for("run-tampered", relative_path)
    original = path.read_bytes()
    replacement = b"X" if original[:1] != b"X" else b"Y"
    path.write_bytes(replacement + original[1:])

    with pytest.raises(RuntimeError, match="failed verification"):
        render_saved_report(runtime, "run-tampered")


def test_report_regeneration_uses_artifacts_not_sqlite_run_or_item_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = EvaluationRuntime(tmp_path / "state")
    _submit(runtime, "run-artifact-report")
    assert run_worker_once(runtime, worker_id="worker-artifact-report") == "run-artifact-report"
    token = runtime.store.finalization_token("run-artifact-report")

    def forbidden_sqlite_snapshot(*_args, **_kwargs):
        raise AssertionError("report regeneration read a mutable SQLite snapshot")

    monkeypatch.setattr(runtime.store, "get_run", forbidden_sqlite_snapshot)
    monkeypatch.setattr(runtime.store, "list_run_items", forbidden_sqlite_snapshot)

    report = render_saved_report(
        runtime,
        "run-artifact-report",
        publication_token=token,
    )

    assert report.run_id == "run-artifact-report"
    assert report.counts.completed == 2


def test_existing_report_json_does_not_skip_missing_sibling_regeneration(
    tmp_path: Path,
):
    runtime = EvaluationRuntime(tmp_path / "state")
    _submit(runtime, "run-report-siblings")
    assert run_worker_once(runtime, worker_id="worker-report-siblings") == "run-report-siblings"
    expected = {
        relative_path: runtime.artifacts.read_bytes("run-report-siblings", relative_path)
        for relative_path in ("report.md", "report.html")
    }
    for relative_path in expected:
        runtime.artifacts.path_for("run-report-siblings", relative_path).unlink()
    with sqlite3.connect(runtime.store.database_path) as connection:
        connection.executemany(
            "DELETE FROM artifacts WHERE run_id = ? AND relative_path = ?",
            [("run-report-siblings", relative_path) for relative_path in expected],
        )

    report = render_saved_report(runtime, "run-report-siblings")

    assert report.run_id == "run-report-siblings"
    assert {
        relative_path: runtime.artifacts.read_bytes("run-report-siblings", relative_path)
        for relative_path in expected
    } == expected


def test_existing_report_json_does_not_bypass_missing_required_evidence(
    tmp_path: Path,
):
    runtime = EvaluationRuntime(tmp_path / "state")
    _submit(runtime, "run-missing-protocol")
    assert run_worker_once(runtime, worker_id="worker-missing-protocol") == "run-missing-protocol"
    assert runtime.artifacts.path_for("run-missing-protocol", "report.json").is_file()
    runtime.artifacts.path_for("run-missing-protocol", "protocol.json").unlink()
    with sqlite3.connect(runtime.store.database_path) as connection:
        connection.execute(
            "DELETE FROM artifacts WHERE run_id = ? AND relative_path = ?",
            ("run-missing-protocol", "protocol.json"),
        )

    with pytest.raises(RuntimeError, match="expected one registered artifact"):
        render_saved_report(runtime, "run-missing-protocol")


def test_default_openai_connector_disables_ambient_proxy_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = EvaluationRuntime(tmp_path / "state")
    _, job_id = _submit(runtime, "run-proxy-policy")
    plan = worker_module.rebuild_plan_from_job(runtime.store.get_job(job_id).payload)
    real_client = worker_module.httpx.Client
    constructor_options: list[dict[str, object]] = []

    class RecordingClient(real_client):
        def __init__(self, *args, **kwargs):
            constructor_options.append(dict(kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:65530")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:65531")
    monkeypatch.setattr(worker_module.httpx, "Client", RecordingClient)
    _, close = worker_module._default_connector_factory(
        ConnectorConfig(
            kind="openai",
            endpoint="http://127.0.0.1:65529",
        ),
        plan,
        runtime,
    )
    close()

    assert constructor_options == [{"trust_env": False}]


def test_submit_file_collision_creates_no_database_run_or_job(tmp_path: Path):
    runtime = EvaluationRuntime(tmp_path / "state")
    service = BenchmarkService(runtime)
    collision = runtime.artifacts.root / "run-collision"
    collision.write_bytes(b"occupied")

    with pytest.raises(FileExistsError):
        service.submit(
            "gpqa-diamond",
            RunSelection(
                profile="smoke",
                mode=RunMode.SMOKE,
                target_model="offline-fake-model",
            ),
            ConnectorConfig(kind="fake"),
            run_id="run-collision",
        )

    assert runtime.store.list_runs() == []
    collision.unlink()
    submitted = service.submit(
        "gpqa-diamond",
        RunSelection(
            profile="smoke",
            mode=RunMode.SMOKE,
            target_model="offline-fake-model",
        ),
        ConnectorConfig(kind="fake"),
        run_id="run-collision",
    )
    assert submitted.run.run_id == "run-collision"
    assert runtime.store.get_job(submitted.job_id).status == "queued"


def test_resume_file_collision_creates_no_successor_run_or_job(tmp_path: Path):
    runtime = EvaluationRuntime(tmp_path / "state")
    service = _run_partial(runtime, "run-source")
    collision = runtime.artifacts.root / "run-successor"
    collision.write_bytes(b"occupied")

    with pytest.raises(FileExistsError):
        service.resume("run-source", new_run_id="run-successor")

    with pytest.raises(KeyError):
        runtime.store.get_run("run-successor")
    collision.unlink()
    resumed = service.resume("run-source", new_run_id="run-successor")
    assert resumed.run.run.run_id == "run-successor"
    assert resumed.job.status == "queued"


@pytest.mark.parametrize(
    "tamper_case",
    (
        "extra-field",
        "missing-field",
        "secret-field",
        "protocol",
        "official-eligibility",
        "connector",
    ),
)
def test_worker_rejects_tampered_durable_job_payload_before_state_or_connector_use(
    tmp_path: Path,
    tamper_case: str,
):
    runtime = EvaluationRuntime(tmp_path / "state")
    _, job_id = _submit(runtime, f"run-payload-{tamper_case}")
    payload = json.loads(json.dumps(runtime.store.get_job(job_id).payload))

    if tamper_case == "extra-field":
        payload["unexpected"] = "not-in-the-job-contract"
    elif tamper_case == "missing-field":
        del payload["protocol"]
    elif tamper_case == "secret-field":
        payload["connector"]["api_key"] = "sk-synthetic-not-a-real-secret-123456"
    elif tamper_case == "protocol":
        payload["protocol"]["prompt_version"] = "tampered-prompt"
    elif tamper_case == "official-eligibility":
        payload["official_eligible"] = not payload["official_eligible"]
    else:
        assert tamper_case == "connector"
        payload["connector"]["max_retries"] = 1

    with sqlite3.connect(runtime.store.database_path) as connection:
        connection.execute(
            "UPDATE jobs SET payload_json = ? WHERE job_id = ?",
            (
                json.dumps(payload, ensure_ascii=False, allow_nan=False),
                job_id,
            ),
        )

    def connector_must_not_be_constructed(*_args):
        raise AssertionError("connector construction preceded durable payload validation")

    with pytest.raises((ValueError, SecretSafetyError)):
        run_worker_once(
            runtime,
            worker_id=f"worker-{tamper_case}",
            connector_factory=connector_must_not_be_constructed,
        )

    job = runtime.store.get_job(job_id)
    assert job.status == "queued"
    assert job.attempts == 1
    assert job.lease_owner is None
    assert job.lease_expires_at is None
    assert runtime.store.get_run(job.run_id).run.state is RunState.PENDING
    assert runtime.store.list_run_items(job.run_id) == []
    assert runtime.store.list_artifacts(job.run_id) == []


def test_worker_revalidates_registered_secret_in_durable_payload_before_plan_use(
    tmp_path: Path,
):
    secret = "ordinary-runtime-secret-7319"
    runtime = EvaluationRuntime(
        tmp_path / "state",
        secret_registry=SecretValueRegistry((secret,)),
    )
    _, job_id = _submit(runtime, "run-registered-secret-tamper")
    payload = json.loads(json.dumps(runtime.store.get_job(job_id).payload))
    payload["connector"]["endpoint"] = f"https://example.test/provider/{secret}"
    with sqlite3.connect(runtime.store.database_path) as connection:
        connection.execute(
            "UPDATE jobs SET payload_json = ? WHERE job_id = ?",
            (json.dumps(payload, ensure_ascii=False, allow_nan=False), job_id),
        )

    def connector_must_not_be_constructed(*_args):
        raise AssertionError("connector construction preceded secret validation")

    with pytest.raises(SecretSafetyError) as raised:
        run_worker_once(
            runtime,
            worker_id="worker-registered-secret-tamper",
            connector_factory=connector_must_not_be_constructed,
        )

    assert secret not in str(raised.value)
    job = runtime.store.get_job(job_id)
    assert job.status == "queued"
    assert runtime.store.get_run(job.run_id).run.state is RunState.PENDING
    assert runtime.store.list_run_items(job.run_id) == []
    assert runtime.store.list_artifacts(job.run_id) == []


def test_connector_configuration_is_bound_into_protocol_identity(tmp_path: Path):
    runtime = EvaluationRuntime(tmp_path / "state")
    service = BenchmarkService(runtime)
    selection = RunSelection(
        profile="smoke",
        mode=RunMode.SMOKE,
        target_model="served-model",
    )

    first = service.submit(
        "gpqa-diamond",
        selection,
        ConnectorConfig(
            kind="openai",
            endpoint="https://first.example.test/provider/",
            max_retries=1,
        ),
        run_id="run-connector-first",
    )
    second = service.submit(
        "gpqa-diamond",
        selection,
        ConnectorConfig(
            kind="openai",
            endpoint="https://second.example.test/provider/v1/",
            max_retries=1,
        ),
        run_id="run-connector-second",
    )

    assert first.run.protocol_hash != second.run.protocol_hash
    for submitted in (first, second):
        payload = runtime.store.get_job(submitted.job_id).payload
        bound_connector = payload["protocol"]["adapter_configuration"]["model_connector"]
        assert bound_connector == payload["connector"]
        assert submitted.run.protocol_hash == protocol_hash(
            ProtocolSignature.model_validate(payload["protocol"])
        )


def test_submit_rejects_runtime_secret_in_endpoint_before_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    secret = "zY8mQ2vL4nP6rT1x"
    monkeypatch.setenv("SYNTHETIC_MODEL_SECRET", secret)
    runtime = EvaluationRuntime(tmp_path / "state")
    service = BenchmarkService(runtime)
    connector = ConnectorConfig(
        kind="openai",
        endpoint=f"https://example.test/provider/{secret}",
        secret_env_name="SYNTHETIC_MODEL_SECRET",
    )

    with pytest.raises(SecretSafetyError) as raised:
        service.submit(
            "gpqa-diamond",
            RunSelection(
                profile="smoke",
                mode=RunMode.SMOKE,
                target_model="served-model",
            ),
            connector,
            run_id="run-secret-endpoint",
        )

    assert secret not in str(raised.value)
    assert runtime.store.list_runs() == []
    assert runtime.store.claim_job("worker-no-secret-job", lease_seconds=10) is None
    assert not (runtime.artifacts.root / "run-secret-endpoint").exists()


@pytest.mark.parametrize(
    "update",
    (
        {"item_id": "other-item"},
        {"ordinal": 1},
        {"input_sha256": "b" * 64},
    ),
)
def test_adapter_result_identity_must_match_planned_item(update):
    plan_item = AdapterItem(
        item_id="item-01",
        ordinal=0,
        input_sha256="a" * 64,
        prompt="synthetic",
        target="answer",
        choice_permutation=(),
    )
    result = ItemResult(
        item_id=plan_item.item_id,
        ordinal=plan_item.ordinal,
        input_sha256=plan_item.input_sha256,
        response_text="synthetic",
        target=plan_item.target,
    ).model_copy(update=update)

    with pytest.raises(ValueError, match="identity"):
        worker_module._validate_item_result_identity(plan_item, result)
