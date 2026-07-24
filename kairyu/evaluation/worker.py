"""Lease-fenced evaluation worker and deterministic report recovery."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from threading import Event, Thread
from typing import Any

import httpx

from kairyu.evaluation.adapters import get_adapter
from kairyu.evaluation.adapters.base import (
    AdapterItem,
    AdapterRunPlan,
    BenchmarkAdapter,
    ItemResult,
    ModelConnectorSet,
    ModelRole,
)
from kairyu.evaluation.artifacts import ArtifactWrite
from kairyu.evaluation.connectors import (
    FakeOpenAIConnector,
    ModelConnector,
    OpenAICompatibleConnector,
)
from kairyu.evaluation.control_store import (
    EventRecord,
    JobClaim,
    LeaseConflictError,
    PublicationToken,
    StoredRunItem,
)
from kairyu.evaluation.references import ReferenceSnapshot, load_reference_snapshot
from kairyu.evaluation.reporting import (
    ReportError,
    ReportInputs,
    RunManifest,
    UsageEvidence,
    build_run_manifest,
    render_report,
)
from kairyu.evaluation.schemas import (
    Artifact,
    ItemState,
    Metric,
    ProtocolSignature,
    RunItem,
    RunState,
)
from kairyu.evaluation.service import (
    BenchmarkJobPayload,
    ConnectorConfig,
    EvaluationRuntime,
    rebuild_plan_from_job,
)

ConnectorFactory = Callable[
    [ConnectorConfig, AdapterRunPlan, EvaluationRuntime, ModelRole],
    tuple[ModelConnector, Callable[[], None]],
]

_MAX_WORKER_ATTEMPTS = 3


class _LeaseHeartbeat:
    """Keep one worker lease alive while a synchronous model call blocks."""

    def __init__(
        self,
        runtime: EvaluationRuntime,
        claim: JobClaim,
        lease_seconds: float,
    ) -> None:
        self._runtime = runtime
        self._claim = claim
        self._lease_seconds = lease_seconds
        self._interval = max(0.05, min(30.0, lease_seconds / 3.0))
        self._stop = Event()
        self._failure: BaseException | None = None
        self._cancel_requested = False
        self._thread = Thread(
            target=self._run,
            name=f"evaluation-lease-{claim.job.job_id}",
            daemon=True,
        )

    @property
    def failed(self) -> bool:
        return self._failure is not None

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    def __enter__(self) -> _LeaseHeartbeat:
        job = self._runtime.store.heartbeat_job(
            self._claim.lease_token,
            lease_seconds=self._lease_seconds,
        )
        self._cancel_requested = job.cancel_requested
        if not self._cancel_requested:
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=max(1.0, self._interval * 2.0))
        if exc is None and self._failure is not None:
            raise self._failure

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                job = self._runtime.store.heartbeat_job(
                    self._claim.lease_token,
                    lease_seconds=self._lease_seconds,
                )
                if job.cancel_requested:
                    self._cancel_requested = True
                    self._stop.set()
                    return
            except BaseException as exc:
                self._failure = exc
                self._stop.set()
                return


def run_worker_once(
    runtime: EvaluationRuntime,
    *,
    worker_id: str,
    lease_seconds: float = 900.0,
    connector_factory: ConnectorFactory | None = None,
) -> str | None:
    """Claim and execute at most one job, returning its run ID."""

    claim = runtime.store.claim_job(worker_id, lease_seconds=lease_seconds)
    if claim is None:
        return None
    try:
        _execute_claim(
            runtime,
            claim,
            lease_seconds=lease_seconds,
            connector_factory=connector_factory,
        )
    except LeaseConflictError:
        raise
    except Exception:
        _handle_failed_claim(runtime, claim)
        raise
    return claim.job.run_id


def _handle_failed_claim(runtime: EvaluationRuntime, claim: JobClaim) -> None:
    stored = runtime.store.get_run(claim.job.run_id)
    if stored.run.state in {
        RunState.CANCELLED,
        RunState.PARTIAL,
        RunState.COMPLETED,
        RunState.FAILED,
    }:
        return
    if claim.job.attempts < _MAX_WORKER_ATTEMPTS:
        released = runtime.store.release_job(claim.lease_token, status="queued")
        if released.status == "cancelled":
            render_cancelled_job_report(
                runtime,
                claim.job.run_id,
                claim.job.payload,
            )
        return

    if stored.run.state is RunState.CANCELLING:
        released = runtime.store.release_job(claim.lease_token, status="cancelled")
        render_cancelled_job_report(runtime, claim.job.run_id, claim.job.payload)
        return
    if stored.run.state is RunState.PENDING:
        preparing = runtime.store.compare_and_set_run_state(
            stored.run.run_id,
            lease_token=claim.lease_token,
            expected_state=RunState.PENDING,
            expected_version=stored.version,
            new_state=RunState.PREPARING,
        )
        if preparing is None:
            raise RuntimeError("run changed while exhausting retries")
        stored = preparing
    if stored.run.state in {RunState.BLOCKED, RunState.NEEDS_USER_ACTION}:
        runtime.store.release_job(claim.lease_token, status="failed")
        return
    blocked = runtime.store.compare_and_set_run_state(
        stored.run.run_id,
        lease_token=claim.lease_token,
        expected_state=stored.run.state,
        expected_version=stored.version,
        new_state=RunState.BLOCKED,
        termination_reason="worker_retry_exhausted",
    )
    if blocked is None:
        raise RuntimeError("run changed while exhausting retries")
    runtime.store.release_job(claim.lease_token, status="failed")


def _execute_claim(
    runtime: EvaluationRuntime,
    claim: JobClaim,
    *,
    lease_seconds: float,
    connector_factory: ConnectorFactory | None,
) -> None:
    plan = rebuild_plan_from_job(
        claim.job.payload,
        secret_registry=runtime.secret_registry,
    )
    adapter = get_adapter(plan.benchmark_id)
    stored = runtime.store.get_run(claim.job.run_id)
    _require_matching_plan(stored.run, plan)

    if stored.run.state in {
        RunState.CANCELLED,
        RunState.COMPLETED,
        RunState.PARTIAL,
        RunState.FAILED,
    }:
        _recover_terminal_claim(runtime, claim, adapter, plan)
        return

    if stored.run.state is RunState.PENDING:
        preparing = runtime.store.compare_and_set_run_state(
            stored.run.run_id,
            lease_token=claim.lease_token,
            expected_state=RunState.PENDING,
            expected_version=stored.version,
            new_state=RunState.PREPARING,
        )
        if preparing is None:
            raise RuntimeError("run changed before preparation")
        stored = preparing

    if stored.run.state is RunState.PREPARING:
        prepared_items = _build_preparation_items(runtime, stored.run, plan)
        ready = runtime.store.complete_preparation(
            stored.run.run_id,
            lease_token=claim.lease_token,
            expected_version=stored.version,
            protocol_hash=plan.protocol_hash,
            item_input_manifest_sha256=plan.item_input_manifest_sha256,
            expected_full_count=plan.expected_full_count,
            items=prepared_items,
        )
        if ready is None:
            raise RuntimeError("run changed while completing preparation")
        stored = ready

    if stored.run.state is RunState.READY:
        running = runtime.store.compare_and_set_run_state(
            stored.run.run_id,
            lease_token=claim.lease_token,
            expected_state=RunState.READY,
            expected_version=stored.version,
            new_state=RunState.RUNNING,
        )
        if running is None:
            raise RuntimeError("run changed before execution")
        stored = running

    if stored.run.state not in {RunState.RUNNING, RunState.CANCELLING}:
        raise RuntimeError(f"worker cannot resume run in {stored.run.state.value!r} state")

    if stored.run.state is RunState.RUNNING:
        if not any(
            event.event_type == "run_started"
            for event in _list_all_events(runtime, stored.run.run_id)
        ):
            runtime.store.append_event(
                stored.run.run_id,
                "run_started",
                {
                    "item_count": len(plan.items),
                    "mode": plan.mode.value,
                    "protocol_hash": plan.protocol_hash,
                },
                lease_token=claim.lease_token,
            )

        payload = BenchmarkJobPayload.model_validate(
            claim.job.payload,
            context={"secret_registry": runtime.secret_registry},
        )
        connectors, close_connectors = _build_model_connectors(
            payload,
            plan,
            runtime,
            connector_factory=connector_factory,
        )
        try:
            _run_pending_items(
                runtime,
                claim,
                adapter,
                plan,
                connectors,
                lease_seconds=lease_seconds,
            )
        finally:
            close_connectors()

    stored = runtime.store.get_run(claim.job.run_id)
    items = tuple(item.item for item in _list_all_run_items(runtime, stored.run.run_id))
    results = _reconstruct_results(runtime, plan, items)
    collected_results = _results_for_collection(results, items)
    collected = adapter.collect(stored.run.run_id, plan, collected_results)
    terminal = _transition_to_terminal(runtime, claim, stored, items)
    _publish_aggregates(
        runtime,
        claim,
        plan,
        terminal.run,
        items,
        results,
        collected_results,
        collected.metrics,
        collected.error_counts,
    )
    if terminal.run.state is RunState.COMPLETED:
        release_status = "completed"
    elif terminal.run.state is RunState.CANCELLED:
        release_status = "cancelled"
    else:
        release_status = "failed"
    render_saved_report(
        runtime,
        claim.job.run_id,
        publication_token=claim.lease_token,
    )
    runtime.store.release_job(claim.lease_token, status=release_status)


def _run_pending_items(
    runtime: EvaluationRuntime,
    claim: JobClaim,
    adapter: BenchmarkAdapter,
    plan: AdapterRunPlan,
    connector: ModelConnectorSet,
    *,
    lease_seconds: float,
) -> None:
    plan_by_id = {item.item_id: item for item in plan.items}
    for stored_item in _list_all_run_items(runtime, claim.job.run_id):
        if stored_item.item.state in {
            ItemState.COMPLETED,
            ItemState.FAILED,
            ItemState.SKIPPED,
            ItemState.CANCELLED,
        }:
            continue
        if _cancel_requested(runtime, claim):
            break
        runtime.store.heartbeat_job(
            claim.lease_token,
            lease_seconds=lease_seconds,
        )
        if stored_item.item.state is ItemState.PENDING:
            started = runtime.store.compare_and_set_run_item(
                claim.job.run_id,
                stored_item.item.item_id,
                lease_token=claim.lease_token,
                expected_state=ItemState.PENDING,
                expected_version=stored_item.version,
                new_state=ItemState.RUNNING,
            )
            if started is None:
                raise RuntimeError("run item changed before execution")
            started_version = started.item.version
        elif stored_item.item.state is ItemState.RUNNING:
            started_version = stored_item.version
        else:
            raise RuntimeError("run item has an unsupported resumable state")
        plan_item = plan_by_id[stored_item.item.item_id]
        with _LeaseHeartbeat(runtime, claim, lease_seconds) as heartbeat:
            result = adapter.run(
                plan,
                plan_item,
                connector,
                cancel_check=lambda: (
                    heartbeat.failed
                    or heartbeat.cancel_requested
                    or _cancel_requested(runtime, claim)
                ),
            )
        cancelled_during_call = heartbeat.cancel_requested or _cancel_requested(runtime, claim)
        if cancelled_during_call:
            checkpoint_path: str | None = None
            checkpoint_sha256: str | None = None
            if _result_has_concrete_evidence(result):
                checkpoint_path, checkpoint_sha256 = _publish_item_checkpoint(
                    runtime,
                    claim,
                    plan_item,
                    result,
                )
            cancelled = runtime.store.compare_and_set_run_item(
                claim.job.run_id,
                plan_item.item_id,
                lease_token=claim.lease_token,
                expected_state=ItemState.RUNNING,
                expected_version=started_version,
                new_state=ItemState.CANCELLED,
                scores=dict(result.scores),
                error_class=result.error_class or "cancelled",
                checkpoint_relative_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
                checkpoint_source_run_id=(
                    claim.job.run_id if checkpoint_path is not None else None
                ),
            )
            if cancelled is None:
                raise RuntimeError("cancelled item changed before persistence")
            break
        error_class = result.error_class
        if error_class == "cancelled":
            error_class = "unexpected_connector_cancellation"
            result = result.model_copy(update={"error_class": error_class})
        checkpoint_path, checkpoint_sha256 = _publish_item_checkpoint(
            runtime,
            claim,
            plan_item,
            result,
        )
        if error_class is not None:
            failed = runtime.store.compare_and_set_run_item(
                claim.job.run_id,
                plan_item.item_id,
                lease_token=claim.lease_token,
                expected_state=ItemState.RUNNING,
                expected_version=started_version,
                new_state=ItemState.FAILED,
                scores=dict(result.scores),
                error_class=error_class,
                checkpoint_relative_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
                checkpoint_source_run_id=claim.job.run_id,
            )
            if failed is None:
                raise RuntimeError("failed item changed before persistence")
            runtime.store.append_event(
                claim.job.run_id,
                "item_failed",
                {
                    "error_class": error_class,
                    "item_id": plan_item.item_id,
                    "ordinal": plan_item.ordinal,
                    "checkpoint_sha256": checkpoint_sha256,
                },
                lease_token=claim.lease_token,
            )
            continue

        completed = runtime.store.compare_and_set_run_item(
            claim.job.run_id,
            plan_item.item_id,
            lease_token=claim.lease_token,
            expected_state=ItemState.RUNNING,
            expected_version=started_version,
            new_state=ItemState.COMPLETED,
            scores=(
                dict(result.scores)
                if result.scores
                else {"accuracy": 1.0 if result.correct else 0.0}
            ),
            checkpoint_relative_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_source_run_id=claim.job.run_id,
        )
        if completed is None:
            raise RuntimeError("completed item changed before persistence")
        runtime.store.append_event(
            claim.job.run_id,
            "item_completed",
            {
                "item_id": plan_item.item_id,
                "report_error_class": result.report_error_class,
                "scores": (
                    dict(result.scores)
                    if result.scores
                    else {"accuracy": 1.0 if result.correct else 0.0}
                ),
                "ordinal": plan_item.ordinal,
            },
            lease_token=claim.lease_token,
        )


def _publish_item_checkpoint(
    runtime: EvaluationRuntime,
    claim: JobClaim,
    plan_item: AdapterItem,
    result: ItemResult,
) -> tuple[str, str]:
    _validate_item_result_identity(plan_item, result)
    checkpoint_content = _canonical_bytes(result.model_dump(mode="json"))
    checkpoint_sha256 = hashlib.sha256(checkpoint_content).hexdigest()
    checkpoint_path = (
        f"upstream/checkpoints/{plan_item.ordinal:06d}-"
        f"{plan_item.input_sha256[:12]}-{checkpoint_sha256[:12]}.json"
    )
    _publish_bytes(
        runtime,
        run_id=claim.job.run_id,
        name=f"checkpoint-{plan_item.ordinal:06d}-{checkpoint_sha256[:12]}",
        relative_path=checkpoint_path,
        media_type="application/json",
        content=checkpoint_content,
        publication_token=claim.lease_token,
    )
    return checkpoint_path, checkpoint_sha256


def _validate_item_result_identity(
    plan_item: AdapterItem,
    result: ItemResult,
) -> None:
    if (
        result.item_id != plan_item.item_id
        or result.ordinal != plan_item.ordinal
        or result.input_sha256 != plan_item.input_sha256
    ):
        raise ValueError("adapter result identity does not match the planned item")


def _result_has_concrete_evidence(result: ItemResult) -> bool:
    """Return whether cancellation produced evidence beyond immutable item identity."""

    return any(
        (
            bool(result.response_text),
            result.extracted_answer is not None,
            result.latency_seconds is not None,
            result.finish_reason is not None,
            result.provider_request_id is not None,
            result.provider_model is not None,
            result.input_tokens is not None,
            result.output_tokens is not None,
            result.target_attempts is not None,
            result.target_request_sha256 is not None,
            bool(result.scores),
            result.report_error_class is not None,
            result.confidence is not None,
            result.judge_response_text is not None,
            result.judge_finish_reason is not None,
            result.judge_provider_request_id is not None,
            result.judge_provider_model is not None,
            result.judge_input_tokens is not None,
            result.judge_output_tokens is not None,
            result.judge_latency_seconds is not None,
            result.judge_attempts is not None,
            result.judge_request_sha256 is not None,
        )
    )


def _build_preparation_items(
    runtime: EvaluationRuntime,
    run,
    plan: AdapterRunPlan,
) -> tuple[RunItem, ...]:
    source_items: dict[str, RunItem] = {}
    if run.resumed_from_run_id is not None:
        source_items = {
            stored.item.item_id: stored.item
            for stored in _list_all_run_items(runtime, run.resumed_from_run_id)
        }

    prepared: list[RunItem] = []
    for item in plan.items:
        source = source_items.get(item.item_id)
        if (
            source is not None
            and source.state is ItemState.COMPLETED
            and source.input_sha256 == item.input_sha256
            and _checkpoint_is_verified(runtime, source)
        ):
            prepared.append(
                source.model_copy(
                    update={
                        "run_id": run.run_id,
                        "ordinal": item.ordinal,
                        "attempt": 1,
                    }
                )
            )
            continue
        prepared.append(
            RunItem(
                run_id=run.run_id,
                item_id=item.item_id,
                ordinal=item.ordinal,
                state=ItemState.PENDING,
                attempt=1,
                input_sha256=item.input_sha256,
            )
        )
    return tuple(prepared)


def _checkpoint_is_verified(
    runtime: EvaluationRuntime,
    item: RunItem,
) -> bool:
    origin = item.checkpoint_source_run_id
    relative_path = item.checkpoint_relative_path
    expected_sha256 = item.checkpoint_sha256
    if origin is None or relative_path is None or expected_sha256 is None:
        return False
    metadata = next(
        (
            artifact
            for artifact in runtime.store.list_artifacts(origin)
            if artifact.relative_path == relative_path
        ),
        None,
    )
    if metadata is None or metadata.sha256 != expected_sha256:
        return False
    try:
        content = runtime.artifacts.read_bytes(origin, relative_path)
    except (OSError, ValueError):
        return False
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        return False
    try:
        result = ItemResult.model_validate_json(
            content,
            context={"secret_registry": runtime.secret_registry},
        )
    except ValueError:
        return False
    if (
        result.item_id != item.item_id
        or result.ordinal != item.ordinal
        or result.input_sha256 != item.input_sha256
    ):
        return False
    if item.state is ItemState.COMPLETED:
        return result.error_class is None
    if item.state is ItemState.FAILED:
        return result.error_class is not None and result.error_class == item.error_class
    if item.state is ItemState.CANCELLED:
        return result.error_class == item.error_class or (
            result.error_class is None and item.error_class == "cancelled"
        )
    return False


def _load_checkpoint_result(
    runtime: EvaluationRuntime,
    item: RunItem,
) -> ItemResult:
    if not _checkpoint_is_verified(runtime, item):
        raise RuntimeError(f"{item.state.value} item checkpoint failed verification")
    origin = item.checkpoint_source_run_id
    relative_path = item.checkpoint_relative_path
    assert origin is not None and relative_path is not None
    return ItemResult.model_validate_json(
        runtime.artifacts.read_bytes(origin, relative_path),
        context={"secret_registry": runtime.secret_registry},
    )


def _reconstruct_results(
    runtime: EvaluationRuntime,
    plan: AdapterRunPlan,
    items: tuple[RunItem, ...],
) -> tuple[ItemResult, ...]:
    plan_by_id = {item.item_id: item for item in plan.items}
    results: list[ItemResult] = []
    for item in sorted(items, key=lambda candidate: candidate.ordinal):
        if item.state is ItemState.COMPLETED:
            results.append(_load_checkpoint_result(runtime, item))
        elif (
            item.state in {ItemState.FAILED, ItemState.CANCELLED}
            and item.checkpoint_relative_path is not None
        ):
            results.append(_load_checkpoint_result(runtime, item))
        elif item.state is ItemState.FAILED:
            plan_item = plan_by_id[item.item_id]
            results.append(
                ItemResult(
                    item_id=item.item_id,
                    ordinal=item.ordinal,
                    input_sha256=item.input_sha256,
                    response_text="",
                    target=plan_item.target,
                    correct=False,
                    error_class=item.error_class or item.state.value,
                )
            )
    return tuple(results)


def _results_for_collection(
    results: tuple[ItemResult, ...],
    items: tuple[RunItem, ...],
) -> tuple[ItemResult, ...]:
    collectible = {
        (item.item_id, item.ordinal)
        for item in items
        if item.state in {ItemState.COMPLETED, ItemState.FAILED}
    }
    return tuple(result for result in results if (result.item_id, result.ordinal) in collectible)


def _transition_to_terminal(
    runtime: EvaluationRuntime,
    claim: JobClaim,
    stored,
    items: tuple[RunItem, ...],
):
    latest = runtime.store.get_run(stored.run.run_id)
    if latest.run.state is RunState.CANCELLING:
        transitioned = runtime.store.compare_and_set_run_state(
            latest.run.run_id,
            lease_token=claim.lease_token,
            expected_state=RunState.CANCELLING,
            expected_version=latest.version,
            new_state=RunState.CANCELLED,
            termination_reason="cancel_requested",
        )
        if transitioned is None:
            raise RuntimeError("run changed before cancellation completed")
        return transitioned
    completed_count = sum(item.state is ItemState.COMPLETED for item in items)
    all_completed = bool(items) and completed_count == len(items)
    if all_completed:
        target = RunState.COMPLETED
        partial = False
        reason = None
    else:
        target = RunState.PARTIAL if completed_count else RunState.FAILED
        partial = completed_count > 0
        reason = (
            "one_or_more_items_failed"
            if any(item.state is ItemState.FAILED for item in items)
            else "one_or_more_items_incomplete"
        )
    transitioned = runtime.store.compare_and_set_run_state(
        latest.run.run_id,
        lease_token=claim.lease_token,
        expected_state=RunState.RUNNING,
        expected_version=latest.version,
        new_state=target,
        partial=partial,
        termination_reason=reason,
    )
    if transitioned is None:
        raise RuntimeError("run changed before terminal transition")
    return transitioned


def _summarize_usage(
    results: tuple[ItemResult, ...],
    items: tuple[RunItem, ...],
) -> UsageEvidence:
    gaps: list[str] = []
    if any(result.input_tokens is None for result in results):
        gaps.append("one or more item results lack target input token usage")
    if any(result.output_tokens is None for result in results):
        gaps.append("one or more item results lack target output token usage")
    if any(result.latency_seconds is None for result in results):
        gaps.append("one or more item results lack target latency measurements")
    if any(result.target_attempts is not None and result.target_attempts > 1 for result in results):
        gaps.append("target usage may omit earlier connector attempts")
    judge_results = tuple(
        result
        for result in results
        if result.judge_response_text is not None
        or result.judge_provider_model is not None
        or result.judge_input_tokens is not None
        or result.judge_output_tokens is not None
        or result.judge_attempts is not None
        or result.judge_request_sha256 is not None
    )
    if any(result.judge_input_tokens is None for result in judge_results):
        gaps.append("one or more judged items lack judge input token usage")
    if any(result.judge_output_tokens is None for result in judge_results):
        gaps.append("one or more judged items lack judge output token usage")
    if any(result.judge_latency_seconds is None for result in judge_results):
        gaps.append("one or more judged items lack judge latency measurements")
    if any(
        result.judge_attempts is not None and result.judge_attempts > 1 for result in judge_results
    ):
        gaps.append("judge usage may omit earlier connector attempts")
    if any(item.state is not ItemState.COMPLETED for item in items):
        gaps.append("usage is unavailable for one or more non-completed items")
    return UsageEvidence(
        input_tokens=sum(
            (result.input_tokens or 0) + (result.judge_input_tokens or 0) for result in results
        ),
        output_tokens=sum(
            (result.output_tokens or 0) + (result.judge_output_tokens or 0) for result in results
        ),
        total_latency_seconds=sum(
            (result.latency_seconds or 0.0) + (result.judge_latency_seconds or 0.0)
            for result in results
        ),
        measurement_status="partial" if gaps else "complete",
        measurement_unavailable_reasons=tuple(dict.fromkeys(gaps)),
        actual_cost_usd=None,
        actual_cost_unavailable_reason=("item result evidence does not record monetary cost"),
    )


def _error_records_from_results(
    results: tuple[ItemResult, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in results:
        error_class = result.error_class or result.report_error_class
        if error_class is not None:
            records.append(
                {
                    "error_class": error_class,
                    "item_id": result.item_id,
                    "ordinal": result.ordinal,
                }
            )
    return records


def _validate_collected_error_counts(
    records: list[dict[str, Any]],
    collected_error_counts: Mapping[str, int],
) -> None:
    declared = Counter(collected_error_counts)
    derived = Counter(str(record["error_class"]) for record in records)
    if declared != derived:
        raise RuntimeError("collected error counts do not match per-item error evidence")


def _publish_aggregates(
    runtime: EvaluationRuntime,
    claim: JobClaim,
    plan: AdapterRunPlan,
    run,
    items: tuple[RunItem, ...],
    results: tuple[ItemResult, ...],
    collected_results: tuple[ItemResult, ...],
    metrics: tuple[Metric, ...],
    collected_error_counts: Mapping[str, int],
) -> None:
    snapshot = load_reference_snapshot(benchmark_id=plan.benchmark_id)
    predictions = [
        {
            "extracted_answer": result.extracted_answer,
            "finish_reason": result.finish_reason,
            "confidence": result.confidence,
            "item_id": result.item_id,
            "judge_attempts": result.judge_attempts,
            "judge_finish_reason": result.judge_finish_reason,
            "judge_latency_seconds": result.judge_latency_seconds,
            "judge_provider_model": result.judge_provider_model,
            "judge_provider_request_id": result.judge_provider_request_id,
            "judge_request_sha256": result.judge_request_sha256,
            "judge_response_text": result.judge_response_text,
            "latency_seconds": result.latency_seconds,
            "provider_request_id": result.provider_request_id,
            "provider_model": result.provider_model,
            "response_text": result.response_text,
            "target_attempts": result.target_attempts,
            "target_request_sha256": result.target_request_sha256,
        }
        for result in results
    ]
    item_results = [
        item.model_dump(mode="json")
        for item in sorted(items, key=lambda candidate: candidate.ordinal)
    ]
    errors = _error_records_from_results(results)
    _validate_collected_error_counts(
        _error_records_from_results(collected_results),
        collected_error_counts,
    )
    represented_items = {(str(record["item_id"]), int(record["ordinal"])) for record in errors}
    errors.extend(
        {
            "error_class": item.error_class or item.state.value,
            "item_id": item.item_id,
            "ordinal": item.ordinal,
        }
        for item in items
        if item.state in {ItemState.FAILED, ItemState.CANCELLED}
        and (item.item_id, item.ordinal) not in represented_items
    )
    usage = _summarize_usage(results, items)
    protocol_payload = plan.protocol.model_dump(mode="json")
    observed_provider_models = sorted(
        {
            model
            for result in results
            for model in (result.provider_model, result.judge_provider_model)
            if model is not None
        }
    )
    manifest = build_run_manifest(
        run,
        plan.protocol,
        usage,
        observed_provider_models=observed_provider_models,
    )
    events = [
        {
            "created_at": event.created_at.isoformat(),
            "event_type": event.event_type,
            "payload": event.payload,
            "sequence": event.sequence,
        }
        for event in _list_all_events(runtime, run.run_id)
    ]
    artifacts = (
        (
            "manifest",
            "manifest.json",
            "application/json",
            _canonical_bytes(manifest.model_dump(mode="json")),
        ),
        (
            "protocol",
            "protocol.json",
            "application/json",
            _canonical_bytes(protocol_payload),
        ),
        (
            "events",
            "events.jsonl",
            "application/x-ndjson",
            _jsonl_bytes(events),
        ),
        (
            "predictions",
            "predictions.jsonl",
            "application/x-ndjson",
            _jsonl_bytes(predictions),
        ),
        (
            "item-results",
            "item_results.jsonl",
            "application/x-ndjson",
            _jsonl_bytes(item_results),
        ),
        (
            "metrics",
            "metrics.json",
            "application/json",
            _canonical_bytes([metric.model_dump(mode="json") for metric in metrics]),
        ),
        (
            "errors",
            "errors.jsonl",
            "application/x-ndjson",
            _jsonl_bytes(errors),
        ),
        (
            "usage",
            "usage.json",
            "application/json",
            _canonical_bytes(usage.model_dump(mode="json")),
        ),
        (
            "references",
            "references.json",
            "application/json",
            _canonical_bytes(snapshot.model_dump(mode="json")),
        ),
    )
    for name, path, media_type, content in artifacts:
        _publish_bytes(
            runtime,
            run_id=run.run_id,
            name=name,
            relative_path=path,
            media_type=media_type,
            content=content,
            publication_token=claim.lease_token,
        )


def _recover_terminal_claim(
    runtime: EvaluationRuntime,
    claim: JobClaim,
    adapter: BenchmarkAdapter,
    plan: AdapterRunPlan,
) -> None:
    stored = runtime.store.get_run(claim.job.run_id)
    items = tuple(item.item for item in _list_all_run_items(runtime, stored.run.run_id))
    results = _reconstruct_results(runtime, plan, items)
    collected_results = _results_for_collection(results, items)
    collected = adapter.collect(stored.run.run_id, plan, collected_results)
    _publish_aggregates(
        runtime,
        claim,
        plan,
        stored.run,
        items,
        results,
        collected_results,
        collected.metrics,
        collected.error_counts,
    )
    if stored.run.state is RunState.COMPLETED:
        status = "completed"
    elif stored.run.state is RunState.CANCELLED:
        status = "cancelled"
    else:
        status = "failed"
    render_saved_report(
        runtime,
        stored.run.run_id,
        publication_token=claim.lease_token,
    )
    runtime.store.release_job(claim.lease_token, status=status)


def render_saved_report(
    runtime: EvaluationRuntime,
    run_id: str,
    *,
    publication_token: PublicationToken | None = None,
):
    """Regenerate three reports from local immutable evidence only."""

    token = publication_token or runtime.store.finalization_token(run_id)
    manifest = RunManifest.model_validate_json(
        _read_verified_artifact(runtime, run_id, "manifest.json")
    )
    run = manifest.run
    if run.run_id != run_id:
        raise ValueError("manifest run ID does not match the requested run")
    protocol = ProtocolSignature.model_validate_json(
        _read_verified_artifact(runtime, run_id, "protocol.json")
    )
    usage = UsageEvidence.model_validate_json(
        _read_verified_artifact(runtime, run_id, "usage.json")
    )
    expected_manifest = build_run_manifest(
        run,
        protocol,
        usage,
        observed_provider_models=manifest.observed_provider_models,
    )
    if manifest != expected_manifest:
        raise ValueError("manifest does not match stored protocol and usage evidence")
    metrics_payload = json.loads(_read_verified_artifact(runtime, run_id, "metrics.json"))
    metrics = tuple(Metric.model_validate(metric) for metric in metrics_payload)
    snapshot = ReferenceSnapshot.model_validate_json(
        _read_verified_artifact(runtime, run_id, "references.json")
    )
    items = _parse_run_items_jsonl(_read_verified_artifact(runtime, run_id, "item_results.jsonl"))
    error_counts = _parse_error_counts_jsonl(
        _read_verified_artifact(runtime, run_id, "errors.jsonl"),
        items,
    )
    for relative_path in ("events.jsonl", "predictions.jsonl"):
        _read_verified_artifact(runtime, run_id, relative_path)
    rendered = render_report(
        ReportInputs(
            run=run,
            protocol=protocol,
            metrics=metrics,
            items=items,
            error_counts=error_counts,
            usage=usage,
            sources=(snapshot.source,),
            references=snapshot.results,
        )
    )
    for name, path, media_type, text in (
        ("report-json", "report.json", "application/json", rendered.json),
        ("report-markdown", "report.md", "text/markdown", rendered.markdown),
        ("report-html", "report.html", "text/html", rendered.html),
    ):
        _publish_bytes(
            runtime,
            run_id=run_id,
            name=name,
            relative_path=path,
            media_type=media_type,
            content=text.encode("utf-8"),
            publication_token=token,
        )
    return rendered.report


def render_cancelled_job_report(
    runtime: EvaluationRuntime,
    run_id: str,
    job_payload: dict[str, Any],
):
    """Publish a zero or partial cancelled report without inventing metrics."""

    stored = runtime.store.get_run(run_id)
    if stored.run.state is not RunState.CANCELLED:
        raise ValueError("cancelled report requires a cancelled run")
    job_snapshot = BenchmarkJobPayload.model_validate(
        job_payload,
        context={"secret_registry": runtime.secret_registry},
    )
    plan = rebuild_plan_from_job(
        job_snapshot.model_dump(mode="json"),
        secret_registry=runtime.secret_registry,
    )
    adapter = get_adapter(plan.benchmark_id)
    items = tuple(item.item for item in _list_all_run_items(runtime, run_id))
    results = _reconstruct_results(runtime, plan, items)
    collected = adapter.collect(run_id, plan, _results_for_collection(results, items))
    reference_snapshot = load_reference_snapshot(benchmark_id=plan.benchmark_id)
    token = runtime.store.finalization_token(run_id)
    _publish_cancelled_evidence(
        runtime,
        stored.run,
        job_snapshot.protocol,
        items,
        results,
        collected.metrics,
        reference_snapshot,
        token,
    )
    return render_saved_report(runtime, run_id, publication_token=token)


def _publish_cancelled_evidence(
    runtime: EvaluationRuntime,
    run,
    protocol: ProtocolSignature,
    items: tuple[RunItem, ...],
    results: tuple[ItemResult, ...],
    metrics: tuple[Metric, ...],
    snapshot: ReferenceSnapshot,
    token: PublicationToken,
) -> None:
    predictions = [
        {
            "extracted_answer": result.extracted_answer,
            "finish_reason": result.finish_reason,
            "confidence": result.confidence,
            "item_id": result.item_id,
            "judge_attempts": result.judge_attempts,
            "judge_finish_reason": result.judge_finish_reason,
            "judge_latency_seconds": result.judge_latency_seconds,
            "judge_provider_model": result.judge_provider_model,
            "judge_provider_request_id": result.judge_provider_request_id,
            "judge_request_sha256": result.judge_request_sha256,
            "judge_response_text": result.judge_response_text,
            "latency_seconds": result.latency_seconds,
            "provider_request_id": result.provider_request_id,
            "provider_model": result.provider_model,
            "response_text": result.response_text,
            "target_attempts": result.target_attempts,
            "target_request_sha256": result.target_request_sha256,
        }
        for result in results
    ]
    usage = _summarize_usage(results, items)
    observed_provider_models = sorted(
        {
            model
            for result in results
            for model in (result.provider_model, result.judge_provider_model)
            if model is not None
        }
    )
    manifest = build_run_manifest(
        run,
        protocol,
        usage,
        observed_provider_models=observed_provider_models,
    )
    events = [
        {
            "created_at": event.created_at.isoformat(),
            "event_type": event.event_type,
            "payload": event.payload,
            "sequence": event.sequence,
        }
        for event in _list_all_events(runtime, run.run_id)
    ]
    item_results = [
        item.model_dump(mode="json")
        for item in sorted(items, key=lambda candidate: candidate.ordinal)
    ]
    errors = _error_records_from_results(results)
    represented_items = {(str(record["item_id"]), int(record["ordinal"])) for record in errors}
    errors.extend(
        {
            "error_class": item.error_class or item.state.value,
            "item_id": item.item_id,
            "ordinal": item.ordinal,
        }
        for item in items
        if item.state in {ItemState.FAILED, ItemState.CANCELLED}
        and (item.item_id, item.ordinal) not in represented_items
    )
    artifacts = (
        (
            "manifest",
            "manifest.json",
            "application/json",
            _canonical_bytes(manifest.model_dump(mode="json")),
        ),
        (
            "protocol",
            "protocol.json",
            "application/json",
            _canonical_bytes(protocol.model_dump(mode="json")),
        ),
        ("events", "events.jsonl", "application/x-ndjson", _jsonl_bytes(events)),
        (
            "predictions",
            "predictions.jsonl",
            "application/x-ndjson",
            _jsonl_bytes(predictions),
        ),
        (
            "item-results",
            "item_results.jsonl",
            "application/x-ndjson",
            _jsonl_bytes(item_results),
        ),
        (
            "metrics",
            "metrics.json",
            "application/json",
            _canonical_bytes([metric.model_dump(mode="json") for metric in metrics]),
        ),
        ("errors", "errors.jsonl", "application/x-ndjson", _jsonl_bytes(errors)),
        (
            "usage",
            "usage.json",
            "application/json",
            _canonical_bytes(usage.model_dump(mode="json")),
        ),
        (
            "references",
            "references.json",
            "application/json",
            _canonical_bytes(snapshot.model_dump(mode="json")),
        ),
    )
    for name, path, media_type, content in artifacts:
        _publish_bytes(
            runtime,
            run_id=run.run_id,
            name=name,
            relative_path=path,
            media_type=media_type,
            content=content,
            publication_token=token,
        )


def _build_model_connectors(
    payload: BenchmarkJobPayload,
    plan: AdapterRunPlan,
    runtime: EvaluationRuntime,
    *,
    connector_factory: ConnectorFactory | None,
) -> tuple[ModelConnectorSet, Callable[[], None]]:
    closers: list[Callable[[], None]] = []

    def build(config: ConnectorConfig | None, role: ModelRole) -> ModelConnector | None:
        if config is None:
            return None
        try:
            if connector_factory is None:
                connector, closer = _default_connector_factory(
                    config,
                    plan,
                    runtime,
                    role=role,
                )
            else:
                connector, closer = connector_factory(config, plan, runtime, role)
        except BaseException:
            for existing_closer in reversed(closers):
                existing_closer()
            raise
        closers.append(closer)
        return connector

    target = build(payload.connector, ModelRole.TARGET)
    assert target is not None
    judge = build(payload.judge_connector, ModelRole.JUDGE)
    simulator = build(payload.simulator_connector, ModelRole.SIMULATOR)
    closed = False

    def close_all() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        for closer in reversed(closers):
            closer()

    return ModelConnectorSet(target=target, judge=judge, simulator=simulator), close_all


def _default_connector_factory(
    config: ConnectorConfig,
    plan: AdapterRunPlan,
    runtime: EvaluationRuntime,
    *,
    role: ModelRole = ModelRole.TARGET,
) -> tuple[ModelConnector, Callable[[], None]]:
    if config.kind == "fake":
        if plan.mode.value != "smoke":
            raise ValueError("fixed fake connectors are smoke-only")
        responses = get_adapter(plan.benchmark_id).smoke_connector_results(plan, role)
        return (
            FakeOpenAIConnector(
                responses,
                secret_registry=runtime.secret_registry,
            ),
            lambda: None,
        )

    assert config.endpoint is not None
    client = httpx.Client(trust_env=False)
    connector = OpenAICompatibleConnector(
        config.endpoint,
        client=client,
        secret_env_name=config.secret_env_name,
        secret_registry=runtime.secret_registry,
        max_response_bytes=config.max_response_bytes,
        max_retries=config.max_retries,
    )
    return connector, client.close


def _require_matching_plan(run, plan: AdapterRunPlan) -> None:
    if (
        run.benchmark_id != plan.benchmark_id
        or run.profile != plan.profile
        or run.mode is not plan.mode
        or run.target_model != plan.target_model
        or run.judge_model != plan.selection.judge_model
        or run.simulator_model != plan.selection.simulator_model
        or run.protocol_hash != plan.protocol_hash
        or run.item_input_manifest_sha256 != plan.item_input_manifest_sha256
        or run.selected_item_ids != tuple(item.item_id for item in plan.items)
    ):
        raise ValueError("reconstructed benchmark plan does not match stored run identity")


def _list_all_run_items(
    runtime: EvaluationRuntime,
    run_id: str,
) -> tuple[StoredRunItem, ...]:
    items: list[StoredRunItem] = []
    after_ordinal = -1
    while True:
        page = runtime.store.list_run_items(
            run_id,
            after_ordinal=after_ordinal,
            limit=1_000,
        )
        if not page:
            return tuple(items)
        items.extend(page)
        after_ordinal = page[-1].item.ordinal


def _list_all_events(
    runtime: EvaluationRuntime,
    run_id: str,
) -> tuple[EventRecord, ...]:
    events: list[EventRecord] = []
    after_sequence = 0
    while True:
        page = runtime.store.list_events(
            run_id,
            after_sequence=after_sequence,
            limit=1_000,
        )
        if not page:
            return tuple(events)
        events.extend(page)
        after_sequence = page[-1].sequence


def _cancel_requested(runtime: EvaluationRuntime, claim: JobClaim) -> bool:
    try:
        with runtime.store.active_lease(claim.lease_token):
            pass
    except LeaseConflictError:
        return True
    return runtime.store.get_job(claim.job.job_id).cancel_requested


def _parse_run_items_jsonl(content: bytes) -> tuple[RunItem, ...]:
    items: list[RunItem] = []
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        if not raw_line:
            raise ValueError(f"item_results.jsonl has a blank line at {line_number}")
        try:
            items.append(RunItem.model_validate_json(raw_line))
        except ValueError as exc:
            raise ValueError(
                f"item_results.jsonl has an invalid item at line {line_number}"
            ) from exc
    return tuple(items)


def _parse_error_counts_jsonl(
    content: bytes,
    items: tuple[RunItem, ...],
) -> tuple[ReportError, ...]:
    item_by_identity = {(item.item_id, item.ordinal): item for item in items}
    occurrences: set[tuple[str, int]] = set()
    counts: Counter[str] = Counter()
    required_fields = {"error_class", "item_id", "ordinal"}
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        if not raw_line:
            raise ValueError(f"errors.jsonl has a blank line at {line_number}")
        try:
            payload = json.loads(raw_line)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"errors.jsonl has invalid JSON at line {line_number}") from exc
        if not isinstance(payload, dict) or set(payload) != required_fields:
            raise ValueError(f"errors.jsonl has an invalid record at line {line_number}")
        error_class = payload["error_class"]
        item_id = payload["item_id"]
        ordinal = payload["ordinal"]
        if (
            not isinstance(error_class, str)
            or not error_class.strip()
            or len(error_class) > 128
            or not isinstance(item_id, str)
            or not item_id.strip()
            or len(item_id) > 512
            or isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 0
        ):
            raise ValueError(f"errors.jsonl has an invalid record at line {line_number}")
        identity = (item_id, ordinal)
        item = item_by_identity.get(identity)
        if item is None:
            raise ValueError(f"errors.jsonl references unknown item evidence at line {line_number}")
        if identity in occurrences:
            raise ValueError(f"errors.jsonl duplicates an item occurrence at line {line_number}")
        if item.state not in {ItemState.COMPLETED, ItemState.FAILED, ItemState.CANCELLED}:
            raise ValueError(f"errors.jsonl references a non-terminal item at line {line_number}")
        if item.error_class is not None and item.error_class != error_class:
            raise ValueError(f"errors.jsonl conflicts with item evidence at line {line_number}")
        occurrences.add(identity)
        counts[error_class] += 1

    for item in items:
        if item.error_class is not None and (item.item_id, item.ordinal) not in occurrences:
            raise ValueError("errors.jsonl omits stored item error evidence")
    return tuple(
        ReportError(error_class=error_class, count=count)
        for error_class, count in sorted(counts.items())
    )


def _read_verified_artifact(
    runtime: EvaluationRuntime,
    run_id: str,
    relative_path: str,
) -> bytes:
    matches = [
        artifact
        for artifact in runtime.store.list_artifacts(run_id)
        if artifact.relative_path == relative_path
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one registered artifact at {relative_path!r}, found {len(matches)}"
        )
    metadata = matches[0]
    try:
        content = runtime.artifacts.read_bytes(run_id, relative_path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"registered artifact {relative_path!r} is unreadable") from exc
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if metadata.size_bytes != len(content) or metadata.sha256 != actual_sha256:
        raise RuntimeError(f"registered artifact {relative_path!r} failed verification")
    return content


def _publish_bytes(
    runtime: EvaluationRuntime,
    *,
    run_id: str,
    name: str,
    relative_path: str,
    media_type: str,
    content: bytes,
    publication_token: PublicationToken,
) -> Artifact:
    existing = next(
        (
            artifact
            for artifact in runtime.store.list_artifacts(run_id)
            if artifact.name == name or artifact.relative_path == relative_path
        ),
        None,
    )
    expected_sha256 = hashlib.sha256(content).hexdigest()
    if existing is not None:
        if (
            existing.name != name
            or existing.relative_path != relative_path
            or existing.media_type != media_type
            or existing.sha256 != expected_sha256
            or existing.size_bytes != len(content)
        ):
            raise RuntimeError("immutable artifact conflicts with regenerated evidence")
        try:
            existing_content = runtime.artifacts.read_bytes(run_id, relative_path)
        except (OSError, ValueError) as exc:
            raise RuntimeError("registered immutable artifact is unreadable") from exc
        if existing_content != content:
            raise RuntimeError("registered immutable artifact failed verification")
        return existing
    write: ArtifactWrite = runtime.artifacts.write_bytes(
        run_id,
        relative_path,
        content,
        publication_token=publication_token,
    )
    artifact = Artifact(
        run_id=run_id,
        name=name,
        relative_path=write.relative_path,
        media_type=media_type,
        sha256=write.sha256,
        size_bytes=write.size_bytes,
    )
    return runtime.store.register_artifact(
        artifact,
        publication_token=publication_token,
    )


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    if not records:
        return b""
    return b"".join(_canonical_bytes(record) + b"\n" for record in records)


__all__ = ["render_cancelled_job_report", "render_saved_report", "run_worker_once"]
