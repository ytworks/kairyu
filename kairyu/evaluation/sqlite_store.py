"""SQLite WAL implementation of the benchmark control-plane store."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from kairyu.evaluation.control_store import (
    EventRecord,
    FinalizationToken,
    JobClaim,
    JobRecord,
    JobStatus,
    LeaseConflictError,
    LeaseToken,
    PublicationToken,
    ReleasedJobStatus,
    ResumeResult,
    RunCreationResult,
    RunItemUpdateResult,
    StoreConflictError,
    StoredRun,
    StoredRunItem,
    validate_artifact_publication,
    validate_item_state_transition,
    validate_run_state_transition,
)
from kairyu.evaluation.safety import (
    SecretValueRegistry,
    ensure_secret_free_bytes,
    ensure_secret_free_serialized_json,
)
from kairyu.evaluation.schemas import (
    Artifact,
    BenchmarkRun,
    ItemState,
    RunItem,
    RunState,
    thaw_json_value,
)

_SCHEMA_VERSION = 2
_JOB_STATUSES: frozenset[JobStatus] = frozenset(
    {"queued", "leased", "completed", "failed", "cancelled"}
)
_RELEASED_JOB_STATUSES: frozenset[str] = frozenset({"queued", "completed", "failed", "cancelled"})
_FINALIZED_RUN_JOB_STATUSES: dict[RunState, JobStatus] = {
    RunState.CANCELLED: "cancelled",
    RunState.PARTIAL: "failed",
    RunState.COMPLETED: "completed",
    RunState.FAILED: "failed",
}
_RESUMABLE_RUN_JOB_STATUSES: dict[RunState, JobStatus] = {
    RunState.CANCELLED: "cancelled",
    RunState.PARTIAL: "failed",
    RunState.FAILED: "failed",
    RunState.BLOCKED: "failed",
    RunState.NEEDS_USER_ACTION: "failed",
}
_MIGRATION_1 = (
    """
    CREATE TABLE benchmark_runs (
        run_id TEXT PRIMARY KEY,
        resumed_from_run_id TEXT UNIQUE REFERENCES benchmark_runs(run_id),
        benchmark_id TEXT NOT NULL,
        profile TEXT NOT NULL,
        mode TEXT NOT NULL,
        state TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 0 CHECK (version >= 0),
        payload_json TEXT NOT NULL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE jobs (
        job_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL UNIQUE REFERENCES benchmark_runs(run_id) ON DELETE CASCADE,
        status TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        lease_owner TEXT,
        lease_expires_at REAL,
        cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0, 1)),
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        CHECK (status IN ('queued', 'leased', 'completed', 'failed', 'cancelled')),
        CHECK (
            (status = 'leased' AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
            OR
            (status != 'leased' AND lease_owner IS NULL AND lease_expires_at IS NULL)
        )
    )
    """,
    """
    CREATE INDEX jobs_claim_order
    ON jobs(status, lease_expires_at, created_at, job_id)
    """,
    """
    CREATE TABLE events (
        run_id TEXT NOT NULL REFERENCES benchmark_runs(run_id) ON DELETE CASCADE,
        sequence INTEGER NOT NULL CHECK (sequence > 0),
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at REAL NOT NULL,
        PRIMARY KEY (run_id, sequence)
    )
    """,
    """
    CREATE TABLE artifacts (
        run_id TEXT NOT NULL REFERENCES benchmark_runs(run_id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        relative_path TEXT NOT NULL,
        media_type TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
        created_at REAL NOT NULL,
        payload_json TEXT NOT NULL,
        PRIMARY KEY (run_id, name),
        UNIQUE (run_id, relative_path)
    )
    """,
)
_MIGRATION_2 = (
    """
    CREATE TABLE run_items (
        run_id TEXT NOT NULL REFERENCES benchmark_runs(run_id) ON DELETE CASCADE,
        item_id TEXT NOT NULL,
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
        state TEXT NOT NULL CHECK (
            state IN ('pending', 'running', 'completed', 'failed', 'skipped', 'cancelled')
        ),
        version INTEGER NOT NULL DEFAULT 0 CHECK (version >= 0),
        payload_json TEXT NOT NULL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        PRIMARY KEY (run_id, item_id),
        UNIQUE (run_id, ordinal)
    )
    """,
    """
    CREATE INDEX run_items_state_order
    ON run_items(run_id, state, ordinal)
    """,
)


class SqliteControlStore:
    """Process-safe store using short transactions and SQLite worker leases."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        busy_timeout_ms: int = 5_000,
        secret_registry: SecretValueRegistry | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be non-negative")
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._busy_timeout_ms = busy_timeout_ms
        self._secret_registry = secret_registry
        self._clock = clock or _utc_now
        self._migrate()

    @property
    def database_path(self) -> Path:
        return self._database_path

    def _now(self) -> datetime:
        return _normalise_clock_value(self._clock())

    def _scan_persisted_text(self, *values: str) -> None:
        for value in values:
            ensure_secret_free_bytes(
                value.encode("utf-8"),
                secret_registry=self._secret_registry,
            )

    def _scan_lease_token(self, lease_token: LeaseToken) -> None:
        self._scan_persisted_text(
            lease_token.job_id,
            lease_token.run_id,
            lease_token.worker_id,
        )

    def _scan_publication_token(self, publication_token: PublicationToken) -> None:
        if isinstance(publication_token, LeaseToken):
            self._scan_lease_token(publication_token)
            return
        if isinstance(publication_token, FinalizationToken):
            self._scan_persisted_text(publication_token.run_id)
            return
        raise TypeError("publication_token must be a publication fencing token")

    def create_run(self, run: BenchmarkRun) -> StoredRun:
        if run.state is not RunState.PENDING:
            raise ValueError("new runs must be pending")
        if run.attempt != 1 or run.resumed_from_run_id is not None:
            raise ValueError("new runs must be first attempts; use resume_run for successors")
        now = self._now()
        payload = _model_json(run, self._secret_registry)
        state = _state_value(run.state)
        self._scan_persisted_text(
            run.run_id,
            run.benchmark_id,
            run.profile,
            _enum_or_string(run.mode),
            state,
        )
        with closing(self._connect()) as connection:
            try:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO benchmark_runs (
                            run_id, benchmark_id, profile, mode, state, version,
                            payload_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)
                        """,
                        (
                            run.run_id,
                            run.benchmark_id,
                            run.profile,
                            _enum_or_string(run.mode),
                            state,
                            payload,
                            run.created_at.timestamp(),
                            now.timestamp(),
                        ),
                    )
            except sqlite3.IntegrityError as exc:
                raise StoreConflictError("run already exists") from exc
        return StoredRun(run=run, version=0, updated_at=now)

    def create_run_and_enqueue(
        self,
        run: BenchmarkRun,
        *,
        payload: Mapping[str, Any] | None = None,
        job_id: str | None = None,
    ) -> RunCreationResult:
        """Commit a first-attempt run and its queued job atomically."""

        if run.state is not RunState.PENDING:
            raise ValueError("new runs must be pending")
        if run.attempt != 1 or run.resumed_from_run_id is not None:
            raise ValueError("new runs must be first attempts; use resume_run for successors")
        selected_job_id = job_id or f"job-{uuid.uuid4().hex}"
        _validate_identifier(selected_job_id, "job_id")
        timestamp = self._now()
        run_payload = _model_json(run, self._secret_registry)
        state = _state_value(run.state)
        encoded_job_payload = _json(dict(payload or {}), self._secret_registry)
        persisted_job_payload = json.loads(encoded_job_payload)
        self._scan_persisted_text(
            run.run_id,
            run.benchmark_id,
            run.profile,
            _enum_or_string(run.mode),
            state,
            selected_job_id,
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO benchmark_runs (
                            run_id, benchmark_id, profile, mode, state, version,
                            payload_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)
                        """,
                        (
                            run.run_id,
                            run.benchmark_id,
                            run.profile,
                            _enum_or_string(run.mode),
                            state,
                            run_payload,
                            run.created_at.timestamp(),
                            timestamp.timestamp(),
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO jobs (
                            job_id, run_id, status, payload_json, attempts,
                            lease_owner, lease_expires_at, created_at, updated_at
                        ) VALUES (?, ?, 'queued', ?, 0, NULL, NULL, ?, ?)
                        """,
                        (
                            selected_job_id,
                            run.run_id,
                            encoded_job_payload,
                            timestamp.timestamp(),
                            timestamp.timestamp(),
                        ),
                    )
            except sqlite3.IntegrityError as exc:
                raise StoreConflictError("run or job already exists") from exc
        stored_run = StoredRun(run=run, version=0, updated_at=timestamp)
        job = JobRecord(
            job_id=selected_job_id,
            run_id=run.run_id,
            status="queued",
            payload=persisted_job_payload,
            attempts=0,
            lease_owner=None,
            lease_expires_at=None,
            cancel_requested=False,
            created_at=timestamp,
            updated_at=timestamp,
        )
        return RunCreationResult(run=stored_run, job=job)

    def get_run(self, run_id: str) -> StoredRun:
        self._scan_persisted_text(run_id)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload_json, version, updated_at FROM benchmark_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return _stored_run_from_row(row)

    def list_runs(self, *, limit: int = 100) -> list[StoredRun]:
        _validate_limit(limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT payload_json, version, updated_at
                FROM benchmark_runs
                ORDER BY created_at DESC, run_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_stored_run_from_row(row) for row in rows]

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
    ) -> StoredRun | None:
        """Atomically freeze prepared identity, item rows, and the READY state."""

        if expected_version < 0:
            raise ValueError("expected_version must be non-negative")
        prepared_items = _validate_preparation_items(
            run_id,
            items,
            expected_full_count=expected_full_count,
        )
        selected_item_ids = tuple(item.item_id for item in prepared_items)
        prepared_completed_count = sum(item.state is ItemState.COMPLETED for item in prepared_items)
        self._scan_persisted_text(
            run_id,
            protocol_hash,
            item_input_manifest_sha256,
            *(item.item_id for item in prepared_items),
        )
        self._scan_lease_token(lease_token)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                timestamp = self._now()
                _require_active_lease(
                    connection,
                    lease_token,
                    run_id=run_id,
                    now=timestamp,
                )
                row = connection.execute(
                    """
                    SELECT payload_json, version, state, updated_at
                    FROM benchmark_runs
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(run_id)
                stored = _stored_run_from_row(row)
                if (
                    row["state"] == RunState.READY.value
                    and row["version"] == expected_version + 1
                    and _same_prepared_run_identity(
                        stored.run,
                        protocol_hash=protocol_hash,
                        item_input_manifest_sha256=item_input_manifest_sha256,
                        selected_item_ids=selected_item_ids,
                        expected_full_count=expected_full_count,
                        completed_count=prepared_completed_count,
                    )
                    and _stored_items_match(connection, run_id, prepared_items)
                ):
                    connection.commit()
                    return stored
                if row["version"] != expected_version:
                    connection.rollback()
                    return None
                if row["state"] != RunState.PREPARING.value:
                    raise StoreConflictError("preparation completion requires a preparing run")
                run = stored.run
                if run.completed_count or run.failed_count or run.skipped_count:
                    raise StoreConflictError("preparing run already has aggregate item evidence")
                for item in prepared_items:
                    if item.state is not ItemState.COMPLETED:
                        continue
                    source_run_id = item.checkpoint_source_run_id
                    assert source_run_id is not None
                    if source_run_id == run_id or not _run_lineage_contains(
                        connection,
                        run_id,
                        source_run_id,
                    ):
                        raise StoreConflictError(
                            "prepared checkpoint source is not an ancestor run"
                        )
                if run.protocol_hash not in {None, protocol_hash}:
                    raise StoreConflictError("prepared protocol identity conflicts with run intent")
                if run.item_input_manifest_sha256 not in {
                    None,
                    item_input_manifest_sha256,
                }:
                    raise StoreConflictError("prepared item manifest conflicts with run intent")
                if run.selected_item_ids and run.selected_item_ids != selected_item_ids:
                    raise StoreConflictError("prepared item selection conflicts with run intent")
                if (
                    run.expected_full_count is not None
                    and run.expected_full_count != expected_full_count
                ):
                    raise StoreConflictError("prepared expected count conflicts with run intent")
                if (
                    connection.execute(
                        "SELECT 1 FROM run_items WHERE run_id = ? LIMIT 1",
                        (run_id,),
                    ).fetchone()
                    is not None
                ):
                    raise StoreConflictError("preparing run already has item rows")

                validate_run_state_transition(run.state, RunState.READY)
                updated_run = BenchmarkRun.model_validate(
                    {
                        **run.model_dump(),
                        "state": RunState.READY,
                        "protocol_hash": protocol_hash,
                        "item_input_manifest_sha256": item_input_manifest_sha256,
                        "selected_item_ids": selected_item_ids,
                        "expected_full_count": expected_full_count,
                        "completed_count": prepared_completed_count,
                        "failed_count": 0,
                        "skipped_count": 0,
                    }
                )
                for item in prepared_items:
                    payload_json = _model_json(item, self._secret_registry)
                    connection.execute(
                        """
                        INSERT INTO run_items (
                            run_id, item_id, ordinal, state, version, payload_json,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                        """,
                        (
                            run_id,
                            item.item_id,
                            item.ordinal,
                            item.state.value,
                            payload_json,
                            timestamp.timestamp(),
                            timestamp.timestamp(),
                        ),
                    )
                new_version = expected_version + 1
                result = connection.execute(
                    """
                    UPDATE benchmark_runs
                    SET state = ?, version = ?, payload_json = ?, updated_at = ?
                    WHERE run_id = ? AND state = ? AND version = ?
                    """,
                    (
                        RunState.READY.value,
                        new_version,
                        _model_json(updated_run, self._secret_registry),
                        timestamp.timestamp(),
                        run_id,
                        RunState.PREPARING.value,
                        expected_version,
                    ),
                )
                if result.rowcount != 1:
                    raise StoreConflictError("run changed while completing preparation")
                connection.commit()
                return StoredRun(
                    run=updated_run,
                    version=new_version,
                    updated_at=timestamp,
                )
            except sqlite3.IntegrityError as exc:
                if connection.in_transaction:
                    connection.rollback()
                raise StoreConflictError("prepared item rows conflict with stored state") from exc
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise

    def get_run_item(self, run_id: str, item_id: str) -> StoredRunItem:
        self._scan_persisted_text(run_id, item_id)
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT payload_json, version, updated_at
                FROM run_items
                WHERE run_id = ? AND item_id = ?
                """,
                (run_id, item_id),
            ).fetchone()
        if row is None:
            raise KeyError((run_id, item_id))
        return _stored_run_item_from_row(row)

    def list_run_items(
        self,
        run_id: str,
        *,
        after_ordinal: int = -1,
        limit: int = 1_000,
    ) -> list[StoredRunItem]:
        self._scan_persisted_text(run_id)
        if not isinstance(after_ordinal, int) or isinstance(after_ordinal, bool):
            raise TypeError("after_ordinal must be an integer")
        if after_ordinal < -1:
            raise ValueError("after_ordinal must be at least -1")
        _validate_limit(limit)
        with closing(self._connect()) as connection:
            if not _run_exists(connection, run_id):
                raise KeyError(run_id)
            rows = connection.execute(
                """
                SELECT payload_json, version, updated_at
                FROM run_items
                WHERE run_id = ? AND ordinal > ?
                ORDER BY ordinal
                LIMIT ?
                """,
                (run_id, after_ordinal, limit),
            ).fetchall()
        return [_stored_run_item_from_row(row) for row in rows]

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
    ) -> RunItemUpdateResult | None:
        """CAS one item and its aggregate run counters in a single transaction."""

        if expected_version < 0:
            raise ValueError("expected_version must be non-negative")
        expected = ItemState(_enum_or_string(expected_state))
        target = ItemState(_enum_or_string(new_state))
        validate_item_state_transition(expected, target)
        requested_scores = dict(scores or {})
        self._scan_persisted_text(
            run_id,
            item_id,
            expected.value,
            target.value,
            *(
                value
                for value in (
                    error_class,
                    checkpoint_relative_path,
                    checkpoint_sha256,
                    checkpoint_source_run_id,
                )
                if value is not None
            ),
        )
        self._scan_lease_token(lease_token)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                timestamp = self._now()
                _require_active_lease(
                    connection,
                    lease_token,
                    run_id=run_id,
                    now=timestamp,
                )
                row = connection.execute(
                    """
                    SELECT payload_json, version, state, updated_at
                    FROM run_items
                    WHERE run_id = ? AND item_id = ?
                    """,
                    (run_id, item_id),
                ).fetchone()
                if row is None:
                    raise KeyError((run_id, item_id))
                current_item = _stored_run_item_from_row(row)
                run_row = connection.execute(
                    """
                    SELECT payload_json, version, state, updated_at
                    FROM benchmark_runs
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()
                if run_row is None:
                    raise KeyError(run_id)
                current_run = _stored_run_from_row(run_row)
                if row["version"] != expected_version or row["state"] != expected.value:
                    if (
                        row["version"] == expected_version + 1
                        and row["state"] == target.value
                        and _same_item_result(
                            current_item.item,
                            scores=requested_scores,
                            error_class=error_class,
                            checkpoint_relative_path=checkpoint_relative_path,
                            checkpoint_sha256=checkpoint_sha256,
                            checkpoint_source_run_id=checkpoint_source_run_id,
                        )
                    ):
                        connection.commit()
                        return RunItemUpdateResult(item=current_item, run=current_run)
                    connection.rollback()
                    return None
                if current_run.run.state not in {RunState.RUNNING, RunState.CANCELLING}:
                    raise StoreConflictError(
                        "run items may change only while the run is running or cancelling"
                    )
                validate_item_state_transition(current_item.item.state, target)
                if checkpoint_source_run_id is not None and not _run_lineage_contains(
                    connection,
                    run_id,
                    checkpoint_source_run_id,
                ):
                    raise StoreConflictError(
                        "checkpoint source is not the current run or one of its ancestors"
                    )
                mutation_timestamp = max(
                    timestamp,
                    current_item.updated_at,
                    current_run.updated_at,
                )
                updated_item = RunItem.model_validate(
                    {
                        **current_item.item.model_dump(),
                        "state": target,
                        "scores": requested_scores,
                        "error_class": error_class,
                        "checkpoint_relative_path": checkpoint_relative_path,
                        "checkpoint_sha256": checkpoint_sha256,
                        "checkpoint_source_run_id": checkpoint_source_run_id,
                    }
                )
                count_fields = {
                    ItemState.COMPLETED: "completed_count",
                    ItemState.FAILED: "failed_count",
                    ItemState.SKIPPED: "skipped_count",
                }
                run_changes: dict[str, Any] = {}
                count_field = count_fields.get(target)
                if count_field is not None:
                    run_changes[count_field] = getattr(current_run.run, count_field) + 1
                    if current_run.run.state is RunState.CANCELLING:
                        run_changes["partial"] = True
                updated_run = BenchmarkRun.model_validate(
                    {**current_run.run.model_dump(), **run_changes}
                )
                new_item_version = expected_version + 1
                item_result = connection.execute(
                    """
                    UPDATE run_items
                    SET state = ?, version = ?, payload_json = ?, updated_at = ?
                    WHERE run_id = ? AND item_id = ? AND state = ? AND version = ?
                    """,
                    (
                        target.value,
                        new_item_version,
                        _model_json(updated_item, self._secret_registry),
                        mutation_timestamp.timestamp(),
                        run_id,
                        item_id,
                        expected.value,
                        expected_version,
                    ),
                )
                if item_result.rowcount != 1:
                    connection.rollback()
                    return None
                new_run_version = current_run.version + 1
                run_result = connection.execute(
                    """
                    UPDATE benchmark_runs
                    SET version = ?, payload_json = ?, updated_at = ?
                    WHERE run_id = ? AND state = ? AND version = ?
                    """,
                    (
                        new_run_version,
                        _model_json(updated_run, self._secret_registry),
                        mutation_timestamp.timestamp(),
                        run_id,
                        current_run.run.state.value,
                        current_run.version,
                    ),
                )
                if run_result.rowcount != 1:
                    raise StoreConflictError("run changed while updating an item")
                connection.commit()
                return RunItemUpdateResult(
                    item=StoredRunItem(
                        item=updated_item,
                        version=new_item_version,
                        updated_at=mutation_timestamp,
                    ),
                    run=StoredRun(
                        run=updated_run,
                        version=new_run_version,
                        updated_at=mutation_timestamp,
                    ),
                )
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise

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
    ) -> StoredRun | None:
        if expected_version < 0:
            raise ValueError("expected_version must be non-negative")
        self._scan_persisted_text(run_id)
        self._scan_lease_token(lease_token)
        expected = _state_value(expected_state)
        target = _state_value(new_state)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                timestamp = self._now()
                row = connection.execute(
                    """
                    SELECT payload_json, version, state, updated_at
                    FROM benchmark_runs
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(run_id)
                if row["version"] != expected_version or row["state"] != expected:
                    connection.rollback()
                    return None
                _require_active_lease(
                    connection,
                    lease_token,
                    run_id=run_id,
                    now=timestamp,
                )
                validate_run_state_transition(RunState(expected), RunState(target))
                run = BenchmarkRun.model_validate_json(row["payload_json"])
                target_state = RunState(target)
                transition_timestamp = max(
                    timestamp,
                    run.created_at,
                    run.started_at or run.created_at,
                    _from_timestamp(row["updated_at"]),
                )
                terminal_states = {
                    RunState.CANCELLED,
                    RunState.PARTIAL,
                    RunState.COMPLETED,
                    RunState.FAILED,
                }
                changes: dict[str, Any] = {
                    "state": target_state,
                    "partial": _partial_for_transition(run, target_state, partial),
                    "finished_at": (
                        transition_timestamp if target_state in terminal_states else None
                    ),
                }
                if (
                    target_state
                    in {
                        RunState.RUNNING,
                        RunState.CANCELLING,
                        *terminal_states,
                    }
                    and run.started_at is None
                ):
                    changes["started_at"] = transition_timestamp
                if termination_reason is not None:
                    changes["termination_reason"] = termination_reason
                updated_run = BenchmarkRun.model_validate({**run.model_dump(), **changes})
                new_version = expected_version + 1
                result = connection.execute(
                    """
                    UPDATE benchmark_runs
                    SET state = ?, version = ?, payload_json = ?, updated_at = ?
                    WHERE run_id = ? AND state = ? AND version = ?
                    """,
                    (
                        target,
                        new_version,
                        _model_json(updated_run, self._secret_registry),
                        transition_timestamp.timestamp(),
                        run_id,
                        expected,
                        expected_version,
                    ),
                )
                if result.rowcount != 1:
                    connection.rollback()
                    return None
                connection.commit()
                return StoredRun(
                    run=updated_run,
                    version=new_version,
                    updated_at=transition_timestamp,
                )
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise

    def enqueue_job(
        self,
        run_id: str,
        *,
        payload: Mapping[str, Any] | None = None,
        job_id: str | None = None,
    ) -> JobRecord:
        job_id = job_id or f"job-{uuid.uuid4().hex}"
        _validate_identifier(job_id, "job_id")
        self._scan_persisted_text(run_id, job_id)
        timestamp = self._now()
        encoded_payload = _json(dict(payload or {}), self._secret_registry)
        persisted_payload = json.loads(encoded_payload)
        with closing(self._connect()) as connection:
            try:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO jobs (
                            job_id, run_id, status, payload_json, attempts,
                            lease_owner, lease_expires_at, created_at, updated_at
                        ) VALUES (?, ?, 'queued', ?, 0, NULL, NULL, ?, ?)
                        """,
                        (
                            job_id,
                            run_id,
                            encoded_payload,
                            timestamp.timestamp(),
                            timestamp.timestamp(),
                        ),
                    )
            except sqlite3.IntegrityError as exc:
                raise StoreConflictError(
                    "job already exists or run is unknown/already queued"
                ) from exc
        return JobRecord(
            job_id=job_id,
            run_id=run_id,
            status="queued",
            payload=persisted_payload,
            attempts=0,
            lease_owner=None,
            lease_expires_at=None,
            cancel_requested=False,
            created_at=timestamp,
            updated_at=timestamp,
        )

    def get_job(self, job_id: str) -> JobRecord:
        self._scan_persisted_text(job_id)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return _job_from_row(row)

    def claim_job(
        self,
        worker_id: str,
        *,
        lease_seconds: float,
    ) -> JobClaim | None:
        _validate_identifier(worker_id, "worker_id")
        self._scan_persisted_text(worker_id)
        _validate_lease_seconds(lease_seconds)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                timestamp = self._now()
                lease_expires_at = timestamp + timedelta(seconds=lease_seconds)
                row = connection.execute(
                    """
                    SELECT job_id
                    FROM jobs
                    WHERE status = 'queued'
                       OR (status = 'leased' AND lease_expires_at <= ?)
                    ORDER BY
                        attempts,
                        CASE WHEN status = 'leased' THEN 0 ELSE 1 END,
                        created_at,
                        job_id
                    LIMIT 1
                    """,
                    (timestamp.timestamp(),),
                ).fetchone()
                if row is None:
                    connection.commit()
                    return None
                job_id = row["job_id"]
                result = connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'leased',
                        attempts = attempts + 1,
                        lease_owner = ?,
                        lease_expires_at = ?,
                        updated_at = ?
                    WHERE job_id = ?
                      AND (
                          status = 'queued'
                          OR (status = 'leased' AND lease_expires_at <= ?)
                      )
                    """,
                    (
                        worker_id,
                        lease_expires_at.timestamp(),
                        timestamp.timestamp(),
                        job_id,
                        timestamp.timestamp(),
                    ),
                )
                if result.rowcount != 1:
                    connection.rollback()
                    return None
                claimed = connection.execute(
                    "SELECT * FROM jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                connection.commit()
                assert claimed is not None
                job = _job_from_row(claimed)
                return JobClaim(
                    job=job,
                    lease_token=LeaseToken(
                        job_id=job.job_id,
                        run_id=job.run_id,
                        worker_id=worker_id,
                        attempt=job.attempts,
                    ),
                )
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise

    def heartbeat_job(
        self,
        lease_token: LeaseToken,
        *,
        lease_seconds: float,
    ) -> JobRecord:
        self._scan_lease_token(lease_token)
        _validate_lease_seconds(lease_seconds)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                timestamp = self._now()
                lease = _require_active_lease(
                    connection,
                    lease_token,
                    run_id=lease_token.run_id,
                    now=timestamp,
                )
                requested_expiry = timestamp + timedelta(seconds=lease_seconds)
                current_expiry = _from_timestamp(lease["lease_expires_at"])
                lease_expires_at = max(current_expiry, requested_expiry)
                result = connection.execute(
                    """
                    UPDATE jobs
                    SET lease_expires_at = ?, updated_at = ?
                    WHERE job_id = ?
                      AND run_id = ?
                      AND status = 'leased'
                      AND lease_owner = ?
                      AND attempts = ?
                      AND lease_expires_at > ?
                    """,
                    (
                        lease_expires_at.timestamp(),
                        timestamp.timestamp(),
                        lease_token.job_id,
                        lease_token.run_id,
                        lease_token.worker_id,
                        lease_token.attempt,
                        timestamp.timestamp(),
                    ),
                )
                if result.rowcount != 1:
                    raise LeaseConflictError("job lease changed while heartbeating fencing token")
                row = connection.execute(
                    "SELECT * FROM jobs WHERE job_id = ?", (lease_token.job_id,)
                ).fetchone()
                connection.commit()
                assert row is not None
                return _job_from_row(row)
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise

    def release_job(
        self,
        lease_token: LeaseToken,
        *,
        status: ReleasedJobStatus = "queued",
    ) -> JobRecord:
        if status not in _RELEASED_JOB_STATUSES:
            raise ValueError(f"invalid released job status: {status!r}")
        self._scan_lease_token(lease_token)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                timestamp = self._now()
                lease = _require_active_lease(
                    connection,
                    lease_token,
                    run_id=lease_token.run_id,
                    now=timestamp,
                )
                effective_status = "cancelled" if bool(lease["cancel_requested"]) else status
                _validate_release_state(
                    connection,
                    lease_token.run_id,
                    effective_status,
                )
                result = connection.execute(
                    """
                    UPDATE jobs
                    SET status = ?,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        cancel_requested = ?,
                        updated_at = ?
                    WHERE job_id = ?
                      AND run_id = ?
                      AND status = 'leased'
                      AND lease_owner = ?
                      AND attempts = ?
                      AND lease_expires_at > ?
                    """,
                    (
                        effective_status,
                        int(effective_status == "cancelled"),
                        timestamp.timestamp(),
                        lease_token.job_id,
                        lease_token.run_id,
                        lease_token.worker_id,
                        lease_token.attempt,
                        timestamp.timestamp(),
                    ),
                )
                if result.rowcount != 1:
                    raise LeaseConflictError("job lease changed while releasing fencing token")
                if effective_status == "cancelled":
                    _finalise_cancelled_run_in_transaction(
                        connection,
                        lease_token.run_id,
                        timestamp=timestamp,
                        secret_registry=self._secret_registry,
                    )
                row = connection.execute(
                    "SELECT * FROM jobs WHERE job_id = ?", (lease_token.job_id,)
                ).fetchone()
                connection.commit()
                assert row is not None
                return _job_from_row(row)
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise

    def request_cancel(self, run_id: str) -> JobRecord:
        self._scan_persisted_text(run_id)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                timestamp = self._now()
                row = connection.execute(
                    "SELECT * FROM jobs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    raise KeyError("run has no job")
                run_row = connection.execute(
                    "SELECT state FROM benchmark_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                assert run_row is not None
                job = _job_from_row(row)
                run_state = RunState(run_row["state"])
                if job.status == "completed" or run_state in _FINALIZED_RUN_JOB_STATUSES:
                    if job.status == "cancelled" and run_state is RunState.CANCELLED:
                        connection.commit()
                        return job
                    raise StoreConflictError("terminal run or job cannot be cancelled")
                if job.status == "cancelled":
                    connection.commit()
                    return job
                if _successor_exists(connection, run_id):
                    raise StoreConflictError("source run is immutable after resume")
                if job.status == "failed" and run_state not in {
                    RunState.BLOCKED,
                    RunState.NEEDS_USER_ACTION,
                }:
                    raise StoreConflictError("terminal job cannot be cancelled")
                lease_expired = (
                    job.status == "leased"
                    and job.lease_expires_at is not None
                    and job.lease_expires_at <= timestamp
                )
                cancel_immediately = job.status in {"queued", "failed"} or lease_expired
                if cancel_immediately:
                    connection.execute(
                        """
                        UPDATE jobs
                        SET status = 'cancelled', cancel_requested = 1,
                            lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
                        WHERE job_id = ?
                        """,
                        (timestamp.timestamp(), job.job_id),
                    )
                    _finalise_cancelled_run_in_transaction(
                        connection,
                        run_id,
                        timestamp=timestamp,
                        secret_registry=self._secret_registry,
                    )
                else:
                    connection.execute(
                        """
                        UPDATE jobs
                        SET cancel_requested = 1, updated_at = ?
                        WHERE job_id = ? AND status = 'leased'
                        """,
                        (timestamp.timestamp(), job.job_id),
                    )
                    _transition_run_in_transaction(
                        connection,
                        run_id,
                        target=RunState.CANCELLING,
                        timestamp=timestamp,
                        termination_reason="cancel_requested",
                        secret_registry=self._secret_registry,
                    )
                updated = connection.execute(
                    "SELECT * FROM jobs WHERE job_id = ?",
                    (job.job_id,),
                ).fetchone()
                connection.commit()
                assert updated is not None
                return _job_from_row(updated)
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise

    def resume_run(self, source_run_id: str, new_run_id: str) -> ResumeResult:
        self._scan_persisted_text(source_run_id, new_run_id)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                timestamp = self._now()
                source_row = connection.execute(
                    """
                    SELECT payload_json, version, state, updated_at
                    FROM benchmark_runs
                    WHERE run_id = ?
                    """,
                    (source_run_id,),
                ).fetchone()
                if source_row is None:
                    raise KeyError(source_run_id)
                source_job_row = connection.execute(
                    "SELECT * FROM jobs WHERE run_id = ?",
                    (source_run_id,),
                ).fetchone()
                if source_job_row is None:
                    raise StoreConflictError("source run has no job")
                source_run = BenchmarkRun.model_validate_json(source_row["payload_json"])
                source_job = _job_from_row(source_job_row)
                expected_job_status = _RESUMABLE_RUN_JOB_STATUSES.get(source_run.state)
                if expected_job_status is None or source_job.status != expected_job_status:
                    raise StoreConflictError(
                        "source run and job are not a resumable terminal or paused pair"
                    )
                if (
                    source_run.protocol_hash is None
                    or source_run.item_input_manifest_sha256 is None
                ):
                    raise StoreConflictError(
                        "source run lacks protocol or item-input manifest identity"
                    )
                existing_row = connection.execute(
                    """
                    SELECT payload_json, version, state, updated_at, resumed_from_run_id
                    FROM benchmark_runs
                    WHERE run_id = ?
                    """,
                    (new_run_id,),
                ).fetchone()
                if existing_row is not None:
                    existing_run = BenchmarkRun.model_validate_json(existing_row["payload_json"])
                    existing_job_row = connection.execute(
                        "SELECT * FROM jobs WHERE run_id = ?",
                        (new_run_id,),
                    ).fetchone()
                    if (
                        existing_row["resumed_from_run_id"] != source_run_id
                        or existing_run.resumed_from_run_id != source_run_id
                        or existing_run.attempt != source_run.attempt + 1
                        or existing_job_row is None
                        or _job_from_row(existing_job_row).payload != source_job.payload
                    ):
                        raise StoreConflictError("successor run ID already exists")
                    result = ResumeResult(
                        run=_stored_run_from_row(existing_row),
                        job=_job_from_row(existing_job_row),
                    )
                    connection.commit()
                    return result
                if _successor_exists(connection, source_run_id):
                    raise StoreConflictError("source run already has a successor")

                successor = BenchmarkRun.model_validate(
                    {
                        **source_run.model_dump(),
                        "run_id": new_run_id,
                        "state": RunState.PENDING,
                        "partial": False,
                        "termination_reason": None,
                        "completed_count": 0,
                        "failed_count": 0,
                        "skipped_count": 0,
                        "created_at": timestamp,
                        "started_at": None,
                        "finished_at": None,
                        "attempt": source_run.attempt + 1,
                        "resumed_from_run_id": source_run_id,
                    }
                )
                successor_payload = _model_json(successor, self._secret_registry)
                encoded_job_payload = _json(source_job.payload, self._secret_registry)
                persisted_job_payload = json.loads(encoded_job_payload)
                while True:
                    new_job_id = f"job-{uuid.uuid4().hex}"
                    if (
                        connection.execute(
                            "SELECT 1 FROM jobs WHERE job_id = ?",
                            (new_job_id,),
                        ).fetchone()
                        is None
                    ):
                        break
                self._scan_persisted_text(new_job_id)
                connection.execute(
                    """
                    INSERT INTO benchmark_runs (
                        run_id, resumed_from_run_id, benchmark_id, profile, mode,
                        state, version, payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                    """,
                    (
                        successor.run_id,
                        source_run_id,
                        successor.benchmark_id,
                        successor.profile,
                        _enum_or_string(successor.mode),
                        successor.state.value,
                        successor_payload,
                        timestamp.timestamp(),
                        timestamp.timestamp(),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO jobs (
                        job_id, run_id, status, payload_json, attempts,
                        lease_owner, lease_expires_at, cancel_requested,
                        created_at, updated_at
                    ) VALUES (?, ?, 'queued', ?, 0, NULL, NULL, 0, ?, ?)
                    """,
                    (
                        new_job_id,
                        successor.run_id,
                        encoded_job_payload,
                        timestamp.timestamp(),
                        timestamp.timestamp(),
                    ),
                )
                stored_run = StoredRun(run=successor, version=0, updated_at=timestamp)
                job = JobRecord(
                    job_id=new_job_id,
                    run_id=successor.run_id,
                    status="queued",
                    payload=persisted_job_payload,
                    attempts=0,
                    lease_owner=None,
                    lease_expires_at=None,
                    cancel_requested=False,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                connection.commit()
                return ResumeResult(run=stored_run, job=job)
            except sqlite3.IntegrityError as exc:
                if connection.in_transaction:
                    connection.rollback()
                raise StoreConflictError("successor run conflicts with stored state") from exc
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise

    def finalization_token(self, run_id: str) -> FinalizationToken:
        self._scan_persisted_text(run_id)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT benchmark_runs.version, benchmark_runs.state, jobs.status
                    FROM benchmark_runs
                    LEFT JOIN jobs ON jobs.run_id = benchmark_runs.run_id
                    WHERE benchmark_runs.run_id = ?
                    """,
                    (run_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(run_id)
                if not _is_finalized_pair(row["state"], row["status"]):
                    raise StoreConflictError(
                        "finalization token requires a matching terminal run and job"
                    )
                token = FinalizationToken(
                    run_id=run_id,
                    run_version=row["version"],
                    allow_aggregate_artifacts=(RunState(row["state"]) is RunState.CANCELLED),
                )
                connection.commit()
                return token
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise

    @contextmanager
    def publication_guard(
        self,
        publication_token: PublicationToken,
        *,
        run_id: str | None = None,
    ) -> Iterator[None]:
        self._scan_publication_token(publication_token)
        guarded_run_id = run_id or publication_token.run_id
        self._scan_persisted_text(guarded_run_id)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                _require_publication_fence(
                    connection,
                    publication_token,
                    run_id=guarded_run_id,
                    now=self._now(),
                )
                yield
                connection.commit()
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise

    @contextmanager
    def active_lease(
        self,
        lease_token: LeaseToken,
        *,
        run_id: str | None = None,
    ) -> Iterator[None]:
        with self.publication_guard(lease_token, run_id=run_id):
            yield

    def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        lease_token: LeaseToken,
    ) -> EventRecord:
        _validate_identifier(event_type, "event_type")
        self._scan_persisted_text(run_id, event_type)
        self._scan_lease_token(lease_token)
        event_payload = dict(payload or {})
        encoded_event_payload = _json(event_payload, self._secret_registry)
        event_payload = json.loads(encoded_event_payload)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                timestamp = self._now()
                _require_active_lease(
                    connection,
                    lease_token,
                    run_id=run_id,
                    now=timestamp,
                )
                sequence = connection.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0) + 1
                    FROM events
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()[0]
                connection.execute(
                    """
                    INSERT INTO events (
                        run_id, sequence, event_type, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        sequence,
                        event_type,
                        encoded_event_payload,
                        timestamp.timestamp(),
                    ),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                if connection.in_transaction:
                    connection.rollback()
                raise KeyError(run_id) from exc
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise
        return EventRecord(
            run_id=run_id,
            sequence=sequence,
            event_type=event_type,
            payload=event_payload,
            created_at=timestamp,
        )

    def list_events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 1_000,
    ) -> list[EventRecord]:
        self._scan_persisted_text(run_id)
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        _validate_limit(limit)
        with closing(self._connect()) as connection:
            if not _run_exists(connection, run_id):
                raise KeyError(run_id)
            rows = connection.execute(
                """
                SELECT run_id, sequence, event_type, payload_json, created_at
                FROM events
                WHERE run_id = ? AND sequence > ?
                ORDER BY sequence
                LIMIT ?
                """,
                (run_id, after_sequence, limit),
            ).fetchall()
        return [
            EventRecord(
                run_id=row["run_id"],
                sequence=row["sequence"],
                event_type=row["event_type"],
                payload=json.loads(row["payload_json"]),
                created_at=_from_timestamp(row["created_at"]),
            )
            for row in rows
        ]

    def register_artifact(
        self,
        artifact: Artifact,
        *,
        publication_token: PublicationToken,
    ) -> Artifact:
        self._scan_persisted_text(
            artifact.run_id,
            artifact.name,
            artifact.relative_path,
            artifact.media_type,
            artifact.sha256,
        )
        self._scan_publication_token(publication_token)
        if publication_token.run_id != artifact.run_id:
            raise LeaseConflictError("publication token does not match the artifact run")
        validate_artifact_publication(publication_token, artifact.relative_path)
        payload_json = _model_json(artifact, self._secret_registry)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                timestamp = self._now()
                _require_publication_fence(
                    connection,
                    publication_token,
                    run_id=artifact.run_id,
                    now=timestamp,
                )
                row = connection.execute(
                    """
                    SELECT payload_json FROM artifacts
                    WHERE run_id = ? AND (name = ? OR relative_path = ?)
                    """,
                    (artifact.run_id, artifact.name, artifact.relative_path),
                ).fetchone()
                if row is not None:
                    existing = Artifact.model_validate_json(row["payload_json"])
                    if _same_artifact_identity(existing, artifact):
                        connection.commit()
                        return existing
                    raise StoreConflictError("artifact already exists with different contents")
                connection.execute(
                    """
                    INSERT INTO artifacts (
                        run_id, name, relative_path, media_type, sha256,
                        size_bytes, created_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact.run_id,
                        artifact.name,
                        artifact.relative_path,
                        artifact.media_type,
                        artifact.sha256,
                        artifact.size_bytes,
                        artifact.created_at.timestamp(),
                        payload_json,
                    ),
                )
                connection.commit()
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise
        return artifact

    def list_artifacts(self, run_id: str) -> list[Artifact]:
        self._scan_persisted_text(run_id)
        with closing(self._connect()) as connection:
            if not _run_exists(connection, run_id):
                raise KeyError(run_id)
            rows = connection.execute(
                """
                SELECT payload_json
                FROM artifacts
                WHERE run_id = ?
                ORDER BY name
                """,
                (run_id,),
            ).fetchall()
        return [Artifact.model_validate_json(row["payload_json"]) for row in rows]

    def _get_artifact(self, run_id: str, name: str) -> Artifact:
        self._scan_persisted_text(run_id, name)
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM artifacts
                WHERE run_id = ? AND name = ?
                """,
                (run_id, name),
            ).fetchone()
        if row is None:
            raise KeyError((run_id, name))
        return Artifact.model_validate_json(row["payload_json"])

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._database_path,
            timeout=self._busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _migrate(self) -> None:
        with closing(self._connect()) as connection:
            journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(journal_mode).lower() != "wal":
                raise RuntimeError(f"SQLite refused WAL mode: {journal_mode!r}")
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at REAL NOT NULL
                    )
                    """
                )
                applied = {
                    row[0]
                    for row in connection.execute(
                        "SELECT version FROM schema_migrations"
                    ).fetchall()
                }
                unknown = {version for version in applied if version > _SCHEMA_VERSION}
                if unknown:
                    raise RuntimeError(
                        f"database schema is newer than this Kairyu build: {sorted(unknown)}"
                    )
                if 1 not in applied:
                    for statement in _MIGRATION_1:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (1, ?)",
                        (self._now().timestamp(),),
                    )
                if 2 not in applied:
                    for statement in _MIGRATION_2:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (2, ?)",
                        (self._now().timestamp(),),
                    )
                connection.commit()
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise


def _partial_for_transition(
    run: BenchmarkRun,
    target: RunState,
    explicit: bool | None,
) -> bool:
    if explicit is not None and not isinstance(explicit, bool):
        raise TypeError("partial must be a boolean")
    if target is RunState.PARTIAL:
        if explicit is False:
            raise ValueError("partial state requires partial evidence")
        return True
    optional_states = {
        RunState.CANCELLING,
        RunState.CANCELLED,
        RunState.FAILED,
        RunState.BLOCKED,
        RunState.NEEDS_USER_ACTION,
    }
    if target in optional_states:
        if explicit is not None:
            return explicit
        accounted = run.completed_count + run.failed_count + run.skipped_count
        return run.partial or accounted > 0
    if explicit is True:
        raise ValueError("partial evidence is invalid for the target run state")
    return False


def _transition_run_in_transaction(
    connection: sqlite3.Connection,
    run_id: str,
    *,
    target: RunState,
    timestamp: datetime,
    termination_reason: str | None,
    secret_registry: SecretValueRegistry | None,
) -> StoredRun:
    row = connection.execute(
        """
        SELECT payload_json, version, state, updated_at
        FROM benchmark_runs
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        raise KeyError("run does not exist")
    source = RunState(row["state"])
    if source == target:
        return _stored_run_from_row(row)
    validate_run_state_transition(source, target)
    run = BenchmarkRun.model_validate_json(row["payload_json"])
    transition_timestamp = max(
        timestamp,
        run.created_at,
        run.started_at or run.created_at,
        _from_timestamp(row["updated_at"]),
    )
    terminal_states = {
        RunState.CANCELLED,
        RunState.PARTIAL,
        RunState.COMPLETED,
        RunState.FAILED,
    }
    changes: dict[str, Any] = {
        "state": target,
        "termination_reason": termination_reason,
        "partial": _partial_for_transition(run, target, None),
        "finished_at": transition_timestamp if target in terminal_states else None,
    }
    if target in {RunState.RUNNING, RunState.CANCELLING, *terminal_states}:
        changes["started_at"] = run.started_at or transition_timestamp
    updated_run = BenchmarkRun.model_validate({**run.model_dump(), **changes})
    new_version = row["version"] + 1
    result = connection.execute(
        """
        UPDATE benchmark_runs
        SET state = ?, version = ?, payload_json = ?, updated_at = ?
        WHERE run_id = ? AND version = ?
        """,
        (
            target.value,
            new_version,
            _model_json(updated_run, secret_registry),
            transition_timestamp.timestamp(),
            run_id,
            row["version"],
        ),
    )
    if result.rowcount != 1:
        raise StoreConflictError("run changed during controller transition")
    return StoredRun(
        run=updated_run,
        version=new_version,
        updated_at=transition_timestamp,
    )


def _finalise_cancelled_run_in_transaction(
    connection: sqlite3.Connection,
    run_id: str,
    *,
    timestamp: datetime,
    secret_registry: SecretValueRegistry | None,
) -> StoredRun:
    row = connection.execute(
        "SELECT state FROM benchmark_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        raise KeyError("run does not exist")
    state = RunState(row["state"])
    if state not in {RunState.CANCELLING, RunState.CANCELLED}:
        _transition_run_in_transaction(
            connection,
            run_id,
            target=RunState.CANCELLING,
            timestamp=timestamp,
            termination_reason="cancel_requested",
            secret_registry=secret_registry,
        )
    return _transition_run_in_transaction(
        connection,
        run_id,
        target=RunState.CANCELLED,
        timestamp=timestamp,
        termination_reason="cancel_requested",
        secret_registry=secret_registry,
    )


def _validate_preparation_items(
    run_id: str,
    items: Sequence[RunItem],
    *,
    expected_full_count: int | None,
) -> tuple[RunItem, ...]:
    if expected_full_count is not None and (
        not isinstance(expected_full_count, int)
        or isinstance(expected_full_count, bool)
        or expected_full_count < 0
    ):
        raise ValueError("expected_full_count must be a non-negative integer or None")
    prepared_items = tuple(items)
    if not prepared_items:
        raise ValueError("preparation requires at least one run item")
    if expected_full_count is not None and expected_full_count < len(prepared_items):
        raise ValueError("expected_full_count cannot be smaller than selected items")
    for ordinal, item in enumerate(prepared_items):
        if not isinstance(item, RunItem):
            raise TypeError("items must contain RunItem records")
        if item.run_id != run_id:
            raise ValueError("prepared item belongs to a different run")
        if item.ordinal != ordinal:
            raise ValueError("prepared item ordinals must be contiguous and ordered from zero")
        if item.attempt != 1:
            raise ValueError("prepared items must be first-attempt records")
        if item.state is ItemState.PENDING:
            if item.scores or item.error_class is not None:
                raise ValueError("pending prepared items cannot contain result evidence")
        elif item.state is ItemState.COMPLETED:
            if item.error_class is not None or item.checkpoint_relative_path is None:
                raise ValueError(
                    "completed prepared items require checkpoint evidence without an error"
                )
        else:
            raise ValueError("prepared items must be pending or checkpoint-completed")
    item_ids = tuple(item.item_id for item in prepared_items)
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("prepared item IDs must be unique")
    return prepared_items


def _same_prepared_run_identity(
    run: BenchmarkRun,
    *,
    protocol_hash: str,
    item_input_manifest_sha256: str,
    selected_item_ids: tuple[str, ...],
    expected_full_count: int | None,
    completed_count: int,
) -> bool:
    return (
        run.protocol_hash == protocol_hash
        and run.item_input_manifest_sha256 == item_input_manifest_sha256
        and run.selected_item_ids == selected_item_ids
        and run.expected_full_count == expected_full_count
        and run.completed_count == completed_count
        and run.failed_count == 0
        and run.skipped_count == 0
    )


def _stored_items_match(
    connection: sqlite3.Connection,
    run_id: str,
    expected_items: Sequence[RunItem],
) -> bool:
    rows = connection.execute(
        """
        SELECT payload_json, version, updated_at
        FROM run_items
        WHERE run_id = ?
        ORDER BY ordinal
        """,
        (run_id,),
    ).fetchall()
    return len(rows) == len(expected_items) and all(
        row["version"] == 0 and RunItem.model_validate_json(row["payload_json"]) == expected_item
        for row, expected_item in zip(rows, expected_items, strict=True)
    )


def _run_lineage_contains(
    connection: sqlite3.Connection,
    run_id: str,
    source_run_id: str,
) -> bool:
    current_run_id: str | None = run_id
    seen: set[str] = set()
    while current_run_id is not None and current_run_id not in seen:
        if current_run_id == source_run_id:
            return True
        seen.add(current_run_id)
        row = connection.execute(
            "SELECT resumed_from_run_id FROM benchmark_runs WHERE run_id = ?",
            (current_run_id,),
        ).fetchone()
        if row is None:
            return False
        current_run_id = row["resumed_from_run_id"]
    return False


def _same_item_result(
    item: RunItem,
    *,
    scores: Mapping[str, float],
    error_class: str | None,
    checkpoint_relative_path: str | None,
    checkpoint_sha256: str | None,
    checkpoint_source_run_id: str | None,
) -> bool:
    return (
        dict(item.scores) == dict(scores)
        and item.error_class == error_class
        and item.checkpoint_relative_path == checkpoint_relative_path
        and item.checkpoint_sha256 == checkpoint_sha256
        and item.checkpoint_source_run_id == checkpoint_source_run_id
    )


def _stored_run_item_from_row(row: sqlite3.Row) -> StoredRunItem:
    return StoredRunItem(
        item=RunItem.model_validate_json(row["payload_json"]),
        version=row["version"],
        updated_at=_from_timestamp(row["updated_at"]),
    )


def _stored_run_from_row(row: sqlite3.Row) -> StoredRun:
    return StoredRun(
        run=BenchmarkRun.model_validate_json(row["payload_json"]),
        version=row["version"],
        updated_at=_from_timestamp(row["updated_at"]),
    )


def _job_from_row(row: sqlite3.Row) -> JobRecord:
    status = row["status"]
    if status not in _JOB_STATUSES:
        raise RuntimeError(f"invalid persisted job status: {status!r}")
    typed_status: JobStatus = status
    lease_expires_at = row["lease_expires_at"]
    return JobRecord(
        job_id=row["job_id"],
        run_id=row["run_id"],
        status=typed_status,
        payload=json.loads(row["payload_json"]),
        attempts=row["attempts"],
        lease_owner=row["lease_owner"],
        lease_expires_at=(
            _from_timestamp(lease_expires_at) if lease_expires_at is not None else None
        ),
        cancel_requested=bool(row["cancel_requested"]),
        created_at=_from_timestamp(row["created_at"]),
        updated_at=_from_timestamp(row["updated_at"]),
    )


def _run_exists(connection: sqlite3.Connection, run_id: str) -> bool:
    return (
        connection.execute("SELECT 1 FROM benchmark_runs WHERE run_id = ?", (run_id,)).fetchone()
        is not None
    )


def _same_artifact_identity(first: Artifact, second: Artifact) -> bool:
    return (
        first.run_id,
        first.name,
        first.relative_path,
        first.media_type,
        first.sha256,
        first.size_bytes,
    ) == (
        second.run_id,
        second.name,
        second.relative_path,
        second.media_type,
        second.sha256,
        second.size_bytes,
    )


def _validate_release_state(
    connection: sqlite3.Connection,
    run_id: str,
    status: ReleasedJobStatus,
) -> None:
    row = connection.execute(
        "SELECT state FROM benchmark_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        raise KeyError("run does not exist")
    state = RunState(row["state"])
    failed_states = {
        RunState.FAILED,
        RunState.PARTIAL,
        RunState.BLOCKED,
        RunState.NEEDS_USER_ACTION,
    }
    terminal_states = {
        RunState.CANCELLED,
        RunState.PARTIAL,
        RunState.COMPLETED,
        RunState.FAILED,
    }
    if status == "completed" and state is not RunState.COMPLETED:
        raise StoreConflictError("completed job release requires a completed run")
    if status == "failed" and state not in failed_states:
        raise StoreConflictError("failed job release requires a failed or blocked run")
    if status == "queued" and state in terminal_states:
        raise StoreConflictError("terminal run cannot release its job as queued")


def _successor_exists(connection: sqlite3.Connection, source_run_id: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM benchmark_runs WHERE resumed_from_run_id = ?",
            (source_run_id,),
        ).fetchone()
        is not None
    )


def _is_finalized_pair(run_state: str, job_status: str | None) -> bool:
    try:
        state = RunState(run_state)
    except ValueError:
        return False
    return _FINALIZED_RUN_JOB_STATUSES.get(state) == job_status


def _require_finalization_fence(
    connection: sqlite3.Connection,
    finalization_token: FinalizationToken,
    *,
    run_id: str,
) -> sqlite3.Row:
    if finalization_token.run_id != run_id:
        raise LeaseConflictError("finalization token does not match the guarded run")
    row = connection.execute(
        """
        SELECT benchmark_runs.version, benchmark_runs.state, jobs.status
        FROM benchmark_runs
        LEFT JOIN jobs ON jobs.run_id = benchmark_runs.run_id
        WHERE benchmark_runs.run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if (
        row is None
        or row["version"] != finalization_token.run_version
        or not _is_finalized_pair(row["state"], row["status"])
        or (
            finalization_token.allow_aggregate_artifacts
            and RunState(row["state"]) is not RunState.CANCELLED
        )
    ):
        raise LeaseConflictError(
            "finalization token is stale or does not match terminal stored state"
        )
    return row


def _require_publication_fence(
    connection: sqlite3.Connection,
    publication_token: PublicationToken,
    *,
    run_id: str,
    now: datetime,
) -> sqlite3.Row:
    if isinstance(publication_token, LeaseToken):
        return _require_active_lease(
            connection,
            publication_token,
            run_id=run_id,
            now=now,
        )
    if isinstance(publication_token, FinalizationToken):
        return _require_finalization_fence(
            connection,
            publication_token,
            run_id=run_id,
        )
    raise TypeError("publication_token must be a publication fencing token")


def _require_active_lease(
    connection: sqlite3.Connection,
    lease_token: LeaseToken,
    *,
    run_id: str,
    now: datetime,
) -> sqlite3.Row:
    if lease_token.run_id != run_id:
        raise LeaseConflictError("lease token does not match the guarded run")
    row = connection.execute(
        """
        SELECT run_id, status, attempts, lease_owner, lease_expires_at,
               cancel_requested
        FROM jobs
        WHERE job_id = ?
        """,
        (lease_token.job_id,),
    ).fetchone()
    if (
        row is None
        or row["run_id"] != run_id
        or row["status"] != "leased"
        or row["attempts"] != lease_token.attempt
        or row["lease_owner"] != lease_token.worker_id
        or row["lease_expires_at"] is None
        or row["lease_expires_at"] <= now.timestamp()
    ):
        raise LeaseConflictError(
            "job lease is expired, reclaimed, or does not match its fencing token"
        )
    return row


def _enum_or_string(value: Any) -> str:
    enum_value = getattr(value, "value", value)
    if not isinstance(enum_value, str):
        raise TypeError("enum value must be a string")
    return enum_value


def _state_value(state: RunState) -> str:
    return RunState(_enum_or_string(state)).value


def _model_json(
    model: Any,
    secret_registry: SecretValueRegistry | None = None,
) -> str:
    encoded = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    ensure_secret_free_serialized_json(
        encoded,
        secret_registry=secret_registry,
    )
    return encoded.decode("utf-8")


def _json(
    payload: Mapping[str, Any],
    secret_registry: SecretValueRegistry | None = None,
) -> str:
    encoded = json.dumps(
        thaw_json_value(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    ensure_secret_free_serialized_json(
        encoded,
        secret_registry=secret_registry,
    )
    return encoded.decode("utf-8")


def _validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value or len(value) > 255:
        raise ValueError(f"{name} must be a non-empty string of at most 255 characters")


def _validate_lease_seconds(lease_seconds: float) -> None:
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")


def _validate_limit(limit: int) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")


def _normalise_clock_value(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _from_timestamp(value: float) -> datetime:
    return datetime.fromtimestamp(value, tz=UTC)
