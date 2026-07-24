"""Persistence boundary for the durable benchmark control plane."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol, TypeAlias

from kairyu.evaluation.schemas import Artifact, BenchmarkRun, ItemState, RunItem, RunState

JobStatus = Literal["queued", "leased", "completed", "failed", "cancelled"]
ReleasedJobStatus = Literal["queued", "completed", "failed", "cancelled"]


class StoreConflictError(RuntimeError):
    """A unique record already exists with different contents."""


class LeaseConflictError(RuntimeError):
    """A worker attempted to mutate a lease it no longer owns."""


class InvalidItemStateTransitionError(RuntimeError):
    """An item-state change is not part of the durable item lifecycle."""

    def __init__(self, source: ItemState, target: ItemState) -> None:
        self.source = source
        self.target = target
        super().__init__(f"invalid run-item state transition: {source.value} -> {target.value}")


class InvalidRunStateTransitionError(RuntimeError):
    """A run-state change is not part of the documented lifecycle graph."""

    def __init__(self, source: RunState, target: RunState) -> None:
        self.source = source
        self.target = target
        super().__init__(f"invalid run state transition: {source.value} -> {target.value}")


_ALLOWED_RUN_STATE_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.PENDING: frozenset({RunState.PREPARING, RunState.CANCELLING, RunState.CANCELLED}),
    RunState.PREPARING: frozenset(
        {
            RunState.READY,
            RunState.CANCELLING,
            RunState.CANCELLED,
            RunState.BLOCKED,
            RunState.NEEDS_USER_ACTION,
        }
    ),
    RunState.READY: frozenset(
        {
            RunState.RUNNING,
            RunState.CANCELLING,
            RunState.CANCELLED,
            RunState.BLOCKED,
            RunState.NEEDS_USER_ACTION,
        }
    ),
    RunState.RUNNING: frozenset(
        {
            RunState.CANCELLING,
            RunState.PARTIAL,
            RunState.COMPLETED,
            RunState.FAILED,
            RunState.BLOCKED,
            RunState.NEEDS_USER_ACTION,
        }
    ),
    RunState.CANCELLING: frozenset({RunState.CANCELLED}),
    RunState.CANCELLED: frozenset(),
    RunState.PARTIAL: frozenset(),
    RunState.COMPLETED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.BLOCKED: frozenset({RunState.READY, RunState.CANCELLING, RunState.CANCELLED}),
    RunState.NEEDS_USER_ACTION: frozenset(
        {RunState.READY, RunState.CANCELLING, RunState.CANCELLED}
    ),
}


def validate_run_state_transition(source: RunState, target: RunState) -> None:
    """Raise when ``source`` to ``target`` is outside the run lifecycle graph."""

    if target not in _ALLOWED_RUN_STATE_TRANSITIONS[source]:
        raise InvalidRunStateTransitionError(source, target)


_ALLOWED_ITEM_STATE_TRANSITIONS: dict[ItemState, frozenset[ItemState]] = {
    ItemState.PENDING: frozenset({ItemState.RUNNING, ItemState.SKIPPED, ItemState.CANCELLED}),
    ItemState.RUNNING: frozenset(
        {
            ItemState.COMPLETED,
            ItemState.FAILED,
            ItemState.SKIPPED,
            ItemState.CANCELLED,
        }
    ),
    ItemState.COMPLETED: frozenset(),
    ItemState.FAILED: frozenset(),
    ItemState.SKIPPED: frozenset(),
    ItemState.CANCELLED: frozenset(),
}


def validate_item_state_transition(source: ItemState, target: ItemState) -> None:
    """Raise when an item transition would rewrite terminal evidence."""

    if target not in _ALLOWED_ITEM_STATE_TRANSITIONS[source]:
        raise InvalidItemStateTransitionError(source, target)


@dataclass(frozen=True, slots=True)
class StoredRun:
    run: BenchmarkRun
    version: int
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class StoredRunItem:
    item: RunItem
    version: int
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class RunItemUpdateResult:
    """The item and run versions committed by one atomic item transition."""

    item: StoredRunItem
    run: StoredRun


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    run_id: str
    status: JobStatus
    payload: dict[str, Any]
    attempts: int
    lease_owner: str | None
    lease_expires_at: datetime | None
    cancel_requested: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class RunCreationResult:
    """A first-attempt run and job committed in one transaction."""

    run: StoredRun
    job: JobRecord


@dataclass(frozen=True, slots=True)
class LeaseToken:
    """Fence worker mutations to one specific job-lease generation."""

    job_id: str
    run_id: str
    worker_id: str
    attempt: int


@dataclass(frozen=True, slots=True)
class JobClaim:
    """A claimed job and the generation token required for worker writes."""

    job: JobRecord
    lease_token: LeaseToken


@dataclass(frozen=True, slots=True)
class FinalizationToken:
    """Fence post-run publication to one immutable terminal run version."""

    run_id: str
    run_version: int
    allow_aggregate_artifacts: bool = False


PublicationToken: TypeAlias = LeaseToken | FinalizationToken
FINALIZATION_ARTIFACT_PATHS = frozenset({"report.json", "report.md", "report.html"})
CANCELLED_FINALIZATION_ARTIFACT_PATHS = FINALIZATION_ARTIFACT_PATHS | frozenset(
    {
        "manifest.json",
        "protocol.json",
        "events.jsonl",
        "predictions.jsonl",
        "item_results.jsonl",
        "metrics.json",
        "errors.jsonl",
        "usage.json",
        "references.json",
    }
)


def validate_artifact_publication(
    publication_token: PublicationToken,
    relative_path: str,
) -> None:
    """Restrict post-run writes to immutable reports or cancelled evidence."""

    if not isinstance(publication_token, FinalizationToken):
        return
    allowed_paths = (
        CANCELLED_FINALIZATION_ARTIFACT_PATHS
        if publication_token.allow_aggregate_artifacts
        else FINALIZATION_ARTIFACT_PATHS
    )
    if relative_path not in allowed_paths:
        raise LeaseConflictError("finalization token cannot publish this artifact path")


@dataclass(frozen=True, slots=True)
class ResumeResult:
    """A newly created successor run and its independently queued job."""

    run: StoredRun
    job: JobRecord


@dataclass(frozen=True, slots=True)
class EventRecord:
    run_id: str
    sequence: int
    event_type: str
    payload: dict[str, Any]
    created_at: datetime


class ControlStore(Protocol):
    """Storage operations needed by future controllers and workers."""

    def create_run(self, run: BenchmarkRun) -> StoredRun: ...

    def create_run_and_enqueue(
        self,
        run: BenchmarkRun,
        *,
        payload: Mapping[str, Any] | None = None,
        job_id: str | None = None,
    ) -> RunCreationResult: ...

    def get_run(self, run_id: str) -> StoredRun: ...

    def list_runs(self, *, limit: int = 100) -> list[StoredRun]: ...

    def complete_preparation(
        self,
        run_id: str,
        *,
        lease_token: LeaseToken,
        expected_version: int,
        protocol_hash: str,
        item_input_manifest_sha256: str,
        expected_full_count: int | None,
        items: Sequence[RunItem],
    ) -> StoredRun | None: ...

    def get_run_item(self, run_id: str, item_id: str) -> StoredRunItem: ...

    def list_run_items(
        self,
        run_id: str,
        *,
        after_ordinal: int = -1,
        limit: int = 1_000,
    ) -> list[StoredRunItem]: ...

    def compare_and_set_run_item(
        self,
        run_id: str,
        item_id: str,
        *,
        lease_token: LeaseToken,
        expected_state: ItemState,
        expected_version: int,
        new_state: ItemState,
        scores: Mapping[str, float] | None = None,
        error_class: str | None = None,
        checkpoint_relative_path: str | None = None,
        checkpoint_sha256: str | None = None,
        checkpoint_source_run_id: str | None = None,
    ) -> RunItemUpdateResult | None: ...

    def compare_and_set_run_state(
        self,
        run_id: str,
        *,
        lease_token: LeaseToken,
        expected_state: RunState,
        expected_version: int,
        new_state: RunState,
        partial: bool | None = None,
        termination_reason: str | None = None,
    ) -> StoredRun | None: ...

    def enqueue_job(
        self,
        run_id: str,
        *,
        payload: Mapping[str, Any] | None = None,
        job_id: str | None = None,
    ) -> JobRecord: ...

    def get_job(self, job_id: str) -> JobRecord: ...

    def claim_job(
        self,
        worker_id: str,
        *,
        lease_seconds: float,
    ) -> JobClaim | None: ...

    def heartbeat_job(
        self,
        lease_token: LeaseToken,
        *,
        lease_seconds: float,
    ) -> JobRecord: ...

    def release_job(
        self,
        lease_token: LeaseToken,
        *,
        status: ReleasedJobStatus = "queued",
    ) -> JobRecord: ...

    def request_cancel(self, run_id: str) -> JobRecord: ...

    def resume_run(self, source_run_id: str, new_run_id: str) -> ResumeResult: ...

    def finalization_token(self, run_id: str) -> FinalizationToken: ...

    def publication_guard(
        self,
        publication_token: PublicationToken,
        *,
        run_id: str | None = None,
    ) -> AbstractContextManager[None]: ...

    def active_lease(
        self,
        lease_token: LeaseToken,
        *,
        run_id: str | None = None,
    ) -> AbstractContextManager[None]: ...

    def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        lease_token: LeaseToken,
    ) -> EventRecord: ...

    def list_events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 1_000,
    ) -> list[EventRecord]: ...

    def register_artifact(
        self,
        artifact: Artifact,
        *,
        publication_token: PublicationToken,
    ) -> Artifact: ...

    def list_artifacts(self, run_id: str) -> list[Artifact]: ...
