"""PostgreSQL-backed shared batch storage and fenced worker claims.

The filesystem :class:`~kairyu.batch.store.BatchStore` intentionally remains the
single-gateway default.  This module provides the multi-gateway implementation:

* file metadata and ordered binary chunks commit in one PostgreSQL transaction;
* upload and JSONL writers use only process-local temporary spool files;
* jobs are visible to every gateway sharing the same durable ``store_id``;
* workers claim jobs with ``FOR UPDATE SKIP LOCKED`` and DB-clock leases; and
* every terminal publication is fenced by the current worker/token/lease.

``psycopg`` is imported lazily so installing core Kairyu does not make the
PostgreSQL deployment extra mandatory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import tempfile
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Literal, Protocol, Self, runtime_checkable

from kairyu.audit_io import BoundedJsonlWriter
from kairyu.batch.store import (
    BatchJob,
    BatchStore,
    FileObject,
    FileTooLargeError,
    JsonlFileWriter,
    StaleBatchClaimError,
    _run_blocking_cancellation_safe,
)

try:  # Optional deployment dependency; checked with a useful error at construction.
    import psycopg
except ModuleNotFoundError:  # pragma: no cover - exercised by core-only installations.
    psycopg = None  # type: ignore[assignment]


_SCHEMA_VERSION = 1
_DB_CHUNK_BYTES = 1024 * 1024
_SUPPORTED_ENDPOINT = "/v1/chat/completions"
_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
_CLAIMABLE_STATES = frozenset({"validating", "in_progress"})
logger = logging.getLogger("kairyu.batch.postgres")


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS batch_store_registry (
        store_id TEXT PRIMARY KEY CHECK (store_id <> ''),
        schema_version INTEGER NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS batch_files (
        store_id TEXT NOT NULL
            REFERENCES batch_store_registry(store_id) ON DELETE CASCADE,
        file_id TEXT NOT NULL,
        owner TEXT NOT NULL,
        created_at BIGINT NOT NULL,
        bytes BIGINT NOT NULL CHECK (bytes >= 0),
        payload JSONB NOT NULL,
        producer_batch_id TEXT,
        producer_fencing_token BIGINT,
        pending BOOLEAN NOT NULL DEFAULT FALSE,
        CONSTRAINT batch_files_producer_shape_ck CHECK (
            (
                producer_batch_id IS NULL
                AND producer_fencing_token IS NULL
                AND pending = FALSE
            )
            OR (
                producer_batch_id IS NOT NULL
                AND producer_fencing_token > 0
            )
        ),
        PRIMARY KEY (store_id, file_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS batch_file_chunks (
        store_id TEXT NOT NULL,
        file_id TEXT NOT NULL,
        chunk_no INTEGER NOT NULL CHECK (chunk_no >= 0),
        data BYTEA NOT NULL,
        PRIMARY KEY (store_id, file_id, chunk_no),
        FOREIGN KEY (store_id, file_id)
            REFERENCES batch_files(store_id, file_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS batch_jobs (
        store_id TEXT NOT NULL
            REFERENCES batch_store_registry(store_id) ON DELETE CASCADE,
        batch_id TEXT NOT NULL,
        owner TEXT NOT NULL,
        created_at BIGINT NOT NULL,
        status TEXT NOT NULL,
        payload JSONB NOT NULL,
        output_file_id TEXT,
        error_file_id TEXT,
        claim_worker TEXT,
        fencing_token BIGINT NOT NULL DEFAULT 0,
        lease_until TIMESTAMPTZ,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        PRIMARY KEY (store_id, batch_id),
        CHECK (
            (claim_worker IS NULL AND lease_until IS NULL)
            OR (claim_worker IS NOT NULL AND lease_until IS NOT NULL)
        ),
        CONSTRAINT batch_jobs_output_file_fk
            FOREIGN KEY (store_id, output_file_id)
            REFERENCES batch_files(store_id, file_id),
        CONSTRAINT batch_jobs_error_file_fk
            FOREIGN KEY (store_id, error_file_id)
            REFERENCES batch_files(store_id, file_id)
    )
    """,
    """
    ALTER TABLE batch_jobs
        ADD COLUMN IF NOT EXISTS output_file_id TEXT,
        ADD COLUMN IF NOT EXISTS error_file_id TEXT
    """,
    """
    ALTER TABLE batch_files
        ADD COLUMN IF NOT EXISTS producer_batch_id TEXT,
        ADD COLUMN IF NOT EXISTS producer_fencing_token BIGINT,
        ADD COLUMN IF NOT EXISTS pending BOOLEAN NOT NULL DEFAULT FALSE
    """,
    """
    UPDATE batch_jobs
    SET output_file_id = NULLIF(payload->>'output_file_id', ''),
        error_file_id = NULLIF(payload->>'error_file_id', '')
    WHERE output_file_id IS DISTINCT FROM NULLIF(payload->>'output_file_id', '')
       OR error_file_id IS DISTINCT FROM NULLIF(payload->>'error_file_id', '')
    """,
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE conrelid = 'batch_jobs'::regclass
              AND conname = 'batch_jobs_output_file_fk'
        ) THEN
            ALTER TABLE batch_jobs
                ADD CONSTRAINT batch_jobs_output_file_fk
                FOREIGN KEY (store_id, output_file_id)
                REFERENCES batch_files(store_id, file_id);
        END IF;
        IF NOT EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE conrelid = 'batch_files'::regclass
              AND conname = 'batch_files_producer_job_fk'
        ) THEN
            ALTER TABLE batch_files
                ADD CONSTRAINT batch_files_producer_job_fk
                FOREIGN KEY (store_id, producer_batch_id)
                REFERENCES batch_jobs(store_id, batch_id);
        END IF;
        IF NOT EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE conrelid = 'batch_files'::regclass
              AND conname = 'batch_files_producer_shape_ck'
        ) THEN
            ALTER TABLE batch_files
                ADD CONSTRAINT batch_files_producer_shape_ck
                CHECK (
                    (
                        producer_batch_id IS NULL
                        AND producer_fencing_token IS NULL
                        AND pending = FALSE
                    )
                    OR (
                        producer_batch_id IS NOT NULL
                        AND producer_fencing_token > 0
                    )
                );
        END IF;
        IF NOT EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE conrelid = 'batch_jobs'::regclass
              AND conname = 'batch_jobs_error_file_fk'
        ) THEN
            ALTER TABLE batch_jobs
                ADD CONSTRAINT batch_jobs_error_file_fk
                FOREIGN KEY (store_id, error_file_id)
                REFERENCES batch_files(store_id, file_id);
        END IF;
    END
    $$
    """,
    """
    CREATE TABLE IF NOT EXISTS batch_claim_audit (
        sequence BIGSERIAL PRIMARY KEY,
        store_id TEXT NOT NULL,
        batch_id TEXT NOT NULL,
        worker_id TEXT,
        fencing_token BIGINT NOT NULL,
        claimed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        event TEXT NOT NULL
            CHECK (event IN ('claim', 'reclaim', 'renew', 'finalize', 'cancel')),
        lease_until TIMESTAMPTZ,
        details JSONB NOT NULL DEFAULT '{}'::jsonb,
        FOREIGN KEY (store_id, batch_id)
            REFERENCES batch_jobs(store_id, batch_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS batch_jobs_claimable_idx
        ON batch_jobs (store_id, status, lease_until, created_at, batch_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS batch_claim_audit_store_batch_idx
        ON batch_claim_audit (store_id, batch_id, sequence)
    """,
    """
    CREATE INDEX IF NOT EXISTS batch_files_pending_producer_idx
        ON batch_files (
            store_id, producer_batch_id, producer_fencing_token, file_id
        )
        WHERE pending
    """,
    """
    CREATE OR REPLACE VIEW kairyu_batch_claim_audit AS
    SELECT
        sequence,
        store_id,
        batch_id,
        worker_id,
        fencing_token,
        claimed_at AS at,
        CASE event
            WHEN 'claim' THEN 'claimed'
            WHEN 'reclaim' THEN 'reclaimed'
            WHEN 'renew' THEN 'renewed'
            WHEN 'finalize' THEN 'terminal_committed'
            ELSE 'cancelled'
        END AS event,
        lease_until,
        details,
        (details->>'previous_lease_until')::timestamptz AS previous_lease_until
    FROM batch_claim_audit
    """,
)


class BatchClaimError(RuntimeError):
    """Base error for invalid shared-worker claim operations."""


@dataclass(frozen=True)
class BatchClaim:
    """One worker's DB-clock lease over a batch job."""

    store_id: str
    job: BatchJob
    worker_id: str
    fencing_token: int
    claimed_at: datetime
    lease_until: datetime

    @property
    def batch_id(self) -> str:
        return self.job.id


@runtime_checkable
class JsonlWriterProtocol(Protocol):
    """Backend-neutral transactional JSONL writer surface used by BatchWorker."""

    @property
    def state(self) -> str: ...

    @property
    def has_content(self) -> bool: ...

    def append(self, payload: dict) -> None: ...

    def commit(self) -> FileObject: ...

    def abort(self) -> None: ...

    def rollback(self) -> None: ...


def _json_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        decoded = json.loads(value)
    else:
        decoded = value
    if not isinstance(decoded, dict):
        raise RuntimeError("corrupt PostgreSQL batch payload: expected an object")
    return decoded


def _validate_identity(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{name} cannot contain NUL")
    return value


def _validate_lease_seconds(value: float) -> float:
    lease_seconds = float(value)
    if not math.isfinite(lease_seconds) or lease_seconds <= 0:
        raise ValueError("lease_seconds must be finite and greater than zero")
    return lease_seconds


class PostgresJsonlFileWriter(JsonlFileWriter):
    """Transactional JSONL writer backed by a process-local binary spool."""

    def __init__(
        self,
        store: PostgresBatchStore,
        filename: str,
        purpose: str,
        owner: str,
        producer_batch_id: str | None = None,
        producer_fencing_token: int | None = None,
    ) -> None:
        # Do not call the filesystem writer constructor: it assumes shared
        # ``_files_dir`` state.  In this backend only the private spool is local.
        self._store = store
        self._filename = filename
        self._purpose = purpose
        self._owner = owner
        self._producer_batch_id = producer_batch_id
        self._producer_fencing_token = producer_fencing_token
        self._file_id = f"file-{uuid.uuid4().hex[:24]}"
        self._temporary_path: Path | None = None
        self._writer: BoundedJsonlWriter | None = None
        self._bytes_written = 0
        self._state: Literal["new", "open", "committed", "aborted"] = "new"

    @property
    def state(self) -> str:
        return self._state

    @property
    def has_content(self) -> bool:
        return self._bytes_written > 0

    def append(self, payload: dict) -> None:
        self._require_writable()
        encoded = json.dumps(payload)
        if self._writer is None:
            self._temporary_path, handle = self._store._new_spool()
            handle.close()
            self._writer = BoundedJsonlWriter(self._temporary_path)
            self._state = "open"
        self._writer.append(encoded)
        self._bytes_written += len(encoded.encode("utf-8")) + 1

    def commit(self) -> FileObject:
        if self._state == "new":
            raise RuntimeError("cannot commit an empty JSONL writer")
        if self._state != "open":
            raise RuntimeError(f"JSONL writer is already {self._state}")
        assert self._temporary_path is not None
        assert self._writer is not None
        try:
            self._close_writer()
            file = self._store._commit_spool_file(
                self._temporary_path,
                file_id=self._file_id,
                bytes_written=self._bytes_written,
                filename=self._filename,
                purpose=self._purpose,
                owner=self._owner,
                producer_batch_id=self._producer_batch_id,
                producer_fencing_token=self._producer_fencing_token,
            )
        except BaseException:
            self._discard_temporary()
            raise
        self._store._remove_spool(self._temporary_path)
        self._temporary_path = None
        self._state = "committed"
        return file

    def abort(self) -> None:
        if self._state == "committed":
            raise RuntimeError("JSONL writer is already committed")
        if self._state == "aborted":
            return
        try:
            self._close_writer()
        finally:
            self._discard_temporary()

    def rollback(self) -> None:
        if self._state == "committed":
            try:
                self._store._discard_file(self._file_id)
            finally:
                self._state = "aborted"
            return
        self.abort()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if self._state != "committed":
            self.abort()
        return False

    def _require_writable(self) -> None:
        if self._state not in ("new", "open"):
            raise RuntimeError(f"JSONL writer is already {self._state}")

    def _close_writer(self) -> None:
        writer = self._writer
        self._writer = None
        if writer is not None:
            writer.close()

    def _discard_temporary(self) -> None:
        self._state = "aborted"
        path = self._temporary_path
        self._temporary_path = None
        if path is not None:
            self._store._remove_spool(path)


class PostgresBatchStore(BatchStore):
    """A PostgreSQL shared store implementing the full batch-store protocol."""

    def __init__(
        self,
        dsn: str,
        *,
        store_id: str,
        spool_dir: str | Path | None = None,
        connect_timeout_s: float = 10.0,
    ) -> None:
        if psycopg is None:
            raise RuntimeError(
                "PostgresBatchStore requires psycopg; install the PostgreSQL deployment dependency"
            )
        self._store_id = _validate_identity(store_id, name="store_id")
        timeout = float(connect_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("connect_timeout_s must be finite and greater than zero")
        self._spool_dir = Path(spool_dir) if spool_dir is not None else None
        self._dsn = dsn
        self._connect_timeout = max(1, math.ceil(timeout))
        if self._spool_dir is not None:
            self._spool_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._lease_lock = threading.RLock()
        self._closed = False
        self._connection = self._connect()
        try:
            self._initialize_schema()
            # File/chunk transactions can be large. Lease heartbeats and fenced
            # terminal publication therefore own a separate connection and
            # lock, so content I/O can never starve renewal.
            self._lease_connection = self._connect()
        except BaseException:
            self._connection.close()
            self._closed = True
            raise

    @property
    def store_id(self) -> str:
        return self._store_id

    def close(self) -> None:
        with self._lock:
            with self._lease_lock:
                if self._closed:
                    return
                self._lease_connection.close()
                self._connection.close()
                self._closed = True

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("PostgresBatchStore is closed")

    def _connect(self):
        assert psycopg is not None
        return psycopg.connect(
            self._dsn,
            autocommit=True,
            connect_timeout=self._connect_timeout,
        )

    def _ensure_general_connection(self) -> None:
        if self._connection.closed or self._connection.broken:
            self._connection.close()
            self._connection = self._connect()

    def _ensure_lease_connection(self) -> None:
        if self._lease_connection.closed or self._lease_connection.broken:
            self._lease_connection.close()
            self._lease_connection = self._connect()

    def _initialize_schema(self) -> None:
        with self._lock:
            self._require_open()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    # ``CREATE TABLE IF NOT EXISTS`` alone still has a catalog
                    # race when three first-start gateways initialize the same
                    # empty database concurrently. Serialize only schema setup;
                    # normal store traffic never takes this advisory lock.
                    cursor.execute(
                        "SELECT pg_advisory_xact_lock(%s, %s)",
                        # Global tables must serialize across schema versions
                        # during a rolling gateway upgrade.
                        (1_261_587_810, 0),
                    )
                    for statement in _SCHEMA_STATEMENTS:
                        cursor.execute(statement)
                    cursor.execute(
                        """
                        INSERT INTO batch_store_registry (store_id, schema_version)
                        VALUES (%s, %s)
                        ON CONFLICT (store_id) DO NOTHING
                        """,
                        (self._store_id, _SCHEMA_VERSION),
                    )
                    cursor.execute(
                        """
                        SELECT schema_version
                        FROM batch_store_registry
                        WHERE store_id = %s
                        """,
                        (self._store_id,),
                    )
                    row = cursor.fetchone()
                    if row is None or row[0] != _SCHEMA_VERSION:
                        observed = None if row is None else row[0]
                        raise RuntimeError(
                            "incompatible PostgreSQL batch schema for "
                            f"store_id {self._store_id!r}: expected "
                            f"{_SCHEMA_VERSION}, got {observed!r}"
                        )

    def _new_spool(self) -> tuple[Path, BinaryIO]:
        handle = tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix="kairyu-postgres-batch-",
            suffix=".spool",
            dir=self._spool_dir,
            delete=False,
        )
        return Path(handle.name), handle

    @staticmethod
    def _remove_spool(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # PostgreSQL publication has already committed when this runs after
            # ``_commit_spool_file``.  A local cleanup failure must not make the
            # caller retry and create a second durable object.
            logger.warning(
                "failed to remove local PostgreSQL batch spool",
                extra={"spool_name": path.name},
                exc_info=True,
            )

    @staticmethod
    def _file_payload(file: FileObject) -> str:
        return json.dumps(file.model_dump(mode="json"))

    @staticmethod
    def _job_payload(job: BatchJob) -> str:
        return json.dumps(job.model_dump(mode="json"))

    # -- files ------------------------------------------------------------

    def save_file(
        self,
        content: bytes,
        filename: str,
        purpose: str,
        owner: str = "default",
    ) -> FileObject:
        file = FileObject(
            id=f"file-{uuid.uuid4().hex[:24]}",
            bytes=len(content),
            created_at=int(time.time()),
            filename=filename,
            purpose=purpose,
            owner=owner,
        )

        def chunks() -> Iterator[bytes]:
            view = memoryview(content)
            for offset in range(0, len(view), _DB_CHUNK_BYTES):
                yield bytes(view[offset : offset + _DB_CHUNK_BYTES])

        self._insert_file(file, chunks())
        return file

    async def save_file_streaming(
        self,
        chunks: AsyncIterator[bytes],
        filename: str,
        purpose: str,
        owner: str = "default",
        max_bytes: int | None = None,
    ) -> FileObject:
        temporary_path, handle = self._new_spool()
        bytes_written = 0
        file_id = f"file-{uuid.uuid4().hex[:24]}"
        try:
            try:
                async for chunk in chunks:
                    prospective_bytes = bytes_written + len(chunk)
                    if max_bytes is not None and prospective_bytes > max_bytes:
                        raise FileTooLargeError
                    await _run_blocking_cancellation_safe(handle.write, chunk)
                    bytes_written = prospective_bytes
            finally:
                await _run_blocking_cancellation_safe(handle.close)

            # Chunk insertion can take seconds for the maximum upload. Keep it
            # off the gateway event loop, but let the underlying thread settle
            # before unlinking its spool if the HTTP task is cancelled.
            commit_task = asyncio.create_task(
                asyncio.to_thread(
                    self._commit_spool_file,
                    temporary_path,
                    file_id=file_id,
                    bytes_written=bytes_written,
                    filename=filename,
                    purpose=purpose,
                    owner=owner,
                )
            )
            cancelled: asyncio.CancelledError | None = None
            while not commit_task.done():
                try:
                    await asyncio.shield(commit_task)
                except asyncio.CancelledError as error:
                    cancelled = error
            try:
                file = commit_task.result()
            except Exception:
                if cancelled is not None:
                    raise cancelled from None
                raise
            if cancelled is not None:
                # The request no longer has a caller that can use this object.
                # Remove the now-complete, still-unreferenced publication.
                try:
                    await _run_blocking_cancellation_safe(
                        self._discard_file,
                        file.id,
                    )
                except Exception:
                    logger.warning(
                        "failed to roll back cancelled PostgreSQL batch upload",
                        extra={"file_id": file.id},
                        exc_info=True,
                    )
                raise cancelled
            return file
        finally:
            await _run_blocking_cancellation_safe(
                self._remove_spool,
                temporary_path,
            )

    def _commit_spool_file(
        self,
        temporary_path: Path,
        *,
        file_id: str,
        bytes_written: int,
        filename: str,
        purpose: str,
        owner: str,
        producer_batch_id: str | None = None,
        producer_fencing_token: int | None = None,
    ) -> FileObject:
        observed_size = temporary_path.stat().st_size
        if observed_size != bytes_written:
            raise RuntimeError("local PostgreSQL batch spool size changed before publication")
        file = FileObject(
            id=file_id,
            bytes=bytes_written,
            created_at=int(time.time()),
            filename=filename,
            purpose=purpose,
            owner=owner,
        )

        def chunks() -> Iterator[bytes]:
            with temporary_path.open("rb") as reader:
                while chunk := reader.read(_DB_CHUNK_BYTES):
                    yield chunk

        self._insert_file(
            file,
            chunks(),
            producer_batch_id=producer_batch_id,
            producer_fencing_token=producer_fencing_token,
        )
        return file

    def _insert_file(
        self,
        file: FileObject,
        chunks: Iterator[bytes],
        *,
        producer_batch_id: str | None = None,
        producer_fencing_token: int | None = None,
    ) -> None:
        if (producer_batch_id is None) != (producer_fencing_token is None):
            raise ValueError("batch output producer identity is incomplete")
        if producer_fencing_token is not None and producer_fencing_token <= 0:
            raise ValueError("batch output fencing token must be positive")
        try:
            with self._lock:
                self._require_open()
                self._ensure_general_connection()
                with self._connection.transaction():
                    with self._connection.cursor() as cursor:
                        if producer_batch_id is not None:
                            cursor.execute(
                                """
                                SELECT status, owner, fencing_token, lease_until,
                                       clock_timestamp()
                                FROM batch_jobs
                                WHERE store_id = %s AND batch_id = %s
                                FOR KEY SHARE
                                """,
                                (self._store_id, producer_batch_id),
                            )
                            producer = cursor.fetchone()
                            if (
                                producer is None
                                or producer[0] != "in_progress"
                                or producer[1] != file.owner
                                or producer[2] != producer_fencing_token
                                or producer[3] is None
                                or producer[3] <= producer[4]
                            ):
                                raise StaleBatchClaimError(
                                    "batch output producer claim is stale"
                                )
                        cursor.execute(
                            """
                            INSERT INTO batch_files (
                                store_id, file_id, owner, created_at, bytes, payload,
                                producer_batch_id, producer_fencing_token, pending
                            )
                            VALUES (
                                %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s
                            )
                            """,
                            (
                                self._store_id,
                                file.id,
                                file.owner,
                                file.created_at,
                                file.bytes,
                                self._file_payload(file),
                                producer_batch_id,
                                producer_fencing_token,
                                producer_batch_id is not None,
                            ),
                        )
                        observed_bytes = 0
                        for chunk_no, chunk in enumerate(chunks):
                            observed_bytes += len(chunk)
                            cursor.execute(
                                """
                                INSERT INTO batch_file_chunks (
                                    store_id, file_id, chunk_no, data
                                )
                                VALUES (%s, %s, %s, %s)
                                """,
                                (self._store_id, file.id, chunk_no, chunk),
                            )
                        if observed_bytes != file.bytes:
                            raise RuntimeError(
                                "PostgreSQL file chunk bytes do not match metadata"
                            )
        except Exception:
            try:
                recovered = self._recover_committed_file(
                    file,
                    producer_batch_id=producer_batch_id,
                    producer_fencing_token=producer_fencing_token,
                )
            except Exception:
                logger.warning(
                    "could not resolve PostgreSQL file commit outcome",
                    extra={"file_id": file.id},
                    exc_info=True,
                )
            else:
                if recovered:
                    logger.warning(
                        "recovered acknowledged PostgreSQL file commit",
                        extra={"file_id": file.id},
                    )
                    return
            raise

    def _recover_committed_file(
        self,
        file: FileObject,
        *,
        producer_batch_id: str | None,
        producer_fencing_token: int | None,
    ) -> bool:
        with self._lease_lock:
            self._require_open()
            self._ensure_lease_connection()
            with self._lease_connection.transaction():
                with self._lease_connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT payload, owner, bytes, producer_batch_id,
                               producer_fencing_token, pending
                        FROM batch_files
                        WHERE store_id = %s AND file_id = %s
                        """,
                        (self._store_id, file.id),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        return False
                    observed = FileObject.model_validate(_json_payload(row[0]))
                    if (
                        observed != file
                        or row[1] != file.owner
                        or int(row[2]) != file.bytes
                        or row[3] != producer_batch_id
                        or row[4] != producer_fencing_token
                        or bool(row[5]) != (producer_batch_id is not None)
                    ):
                        return False
                    cursor.execute(
                        """
                        SELECT count(*), COALESCE(sum(octet_length(data)), 0),
                               min(chunk_no), max(chunk_no)
                        FROM batch_file_chunks
                        WHERE store_id = %s AND file_id = %s
                        """,
                        (self._store_id, file.id),
                    )
                    count, observed_bytes, minimum, maximum = cursor.fetchone()
                    return (
                        int(observed_bytes) == file.bytes
                        and (
                            (count == 0 and minimum is None and maximum is None)
                            or (
                                count > 0
                                and minimum == 0
                                and maximum == count - 1
                            )
                        )
                    )

    def _load_file(
        self,
        cursor: Any,
        file_id: str,
        owner: str | None,
    ) -> FileObject:
        cursor.execute(
            """
            SELECT payload
            FROM batch_files
            WHERE store_id = %s AND file_id = %s AND NOT pending
            """,
            (self._store_id, file_id),
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(file_id)
        file = FileObject.model_validate(_json_payload(row[0]))
        if owner is not None and file.owner != owner:
            raise KeyError(file_id)
        return file

    def get_file(self, file_id: str, owner: str | None = None) -> FileObject:
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    return self._load_file(cursor, file_id, owner)

    def read_file_content(
        self,
        file_id: str,
        owner: str | None = None,
    ) -> bytes:
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    file = self._load_file(cursor, file_id, owner)
                    cursor.execute(
                        """
                        SELECT data
                        FROM batch_file_chunks
                        WHERE store_id = %s AND file_id = %s
                        ORDER BY chunk_no
                        """,
                        (self._store_id, file_id),
                    )
                    content = bytearray()
                    while row := cursor.fetchone():
                        content.extend(bytes(row[0]))
                    if len(content) != file.bytes:
                        raise RuntimeError(
                            f"corrupt PostgreSQL file {file_id!r}: metadata "
                            f"declares {file.bytes} bytes, found {len(content)}"
                        )
                    return bytes(content)

    def iter_file_content(
        self,
        file_id: str,
        owner: str | None = None,
    ) -> Iterator[bytes]:
        file = self.get_file(file_id, owner)

        def chunks() -> Iterator[bytes]:
            chunk_no = 0
            observed_bytes = 0
            while True:
                chunk = self._read_file_chunk(file_id, chunk_no)
                if chunk is None:
                    break
                observed_bytes += len(chunk)
                chunk_no += 1
                yield chunk
            if observed_bytes != file.bytes:
                raise RuntimeError(
                    f"corrupt PostgreSQL file {file_id!r}: metadata "
                    f"declares {file.bytes} bytes, streamed {observed_bytes}"
                )

        return chunks()

    def iter_file_lines(
        self,
        file_id: str,
        owner: str | None = None,
    ) -> Iterator[bytes]:
        # Match the filesystem implementation's immediate existence/ownership
        # check. Each immutable DB chunk is fetched in its own short transaction
        # so a suspended consumer never holds the store connection/RLock and
        # starves the worker's lease-heartbeat thread.
        file = self.get_file(file_id, owner)

        def lines() -> Iterator[bytes]:
            pending = bytearray()
            chunk_no = 0
            observed_bytes = 0
            while True:
                chunk = self._read_file_chunk(file_id, chunk_no)
                if chunk is None:
                    break
                observed_bytes += len(chunk)
                chunk_no += 1
                start = 0
                while True:
                    newline = chunk.find(b"\n", start)
                    if newline < 0:
                        pending.extend(chunk[start:])
                        break
                    pending.extend(chunk[start : newline + 1])
                    yield bytes(pending)
                    pending.clear()
                    start = newline + 1
            if observed_bytes != file.bytes:
                raise RuntimeError(
                    f"corrupt PostgreSQL file {file_id!r}: metadata "
                    f"declares {file.bytes} bytes, streamed {observed_bytes}"
                )
            if pending:
                yield bytes(pending)

        return lines()

    async def iter_file_lines_async(
        self,
        file_id: str,
        owner: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream DB chunks without ever waiting on PostgreSQL in the event loop."""
        file = await asyncio.to_thread(self.get_file, file_id, owner)
        pending = bytearray()
        chunk_no = 0
        observed_bytes = 0
        while True:
            chunk = await asyncio.to_thread(
                self._read_file_chunk,
                file_id,
                chunk_no,
            )
            if chunk is None:
                break
            observed_bytes += len(chunk)
            chunk_no += 1
            start = 0
            while True:
                newline = chunk.find(b"\n", start)
                if newline < 0:
                    pending.extend(chunk[start:])
                    break
                pending.extend(chunk[start : newline + 1])
                yield bytes(pending)
                pending.clear()
                start = newline + 1
        if observed_bytes != file.bytes:
            raise RuntimeError(
                f"corrupt PostgreSQL file {file_id!r}: metadata "
                f"declares {file.bytes} bytes, streamed {observed_bytes}"
            )
        if pending:
            yield bytes(pending)

    def _read_file_chunk(self, file_id: str, chunk_no: int) -> bytes | None:
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT data
                        FROM batch_file_chunks
                        WHERE store_id = %s AND file_id = %s
                          AND chunk_no = %s
                        """,
                        (self._store_id, file_id, chunk_no),
                    )
                    row = cursor.fetchone()
        return None if row is None else bytes(row[0])

    def create_jsonl_writer(
        self,
        filename: str,
        purpose: str,
        owner: str = "default",
    ) -> PostgresJsonlFileWriter:
        return PostgresJsonlFileWriter(self, filename, purpose, owner)

    def create_claim_jsonl_writer(
        self,
        claim: BatchClaim,
        filename: str,
        purpose: str,
        owner: str = "default",
    ) -> PostgresJsonlFileWriter:
        self._validate_claim_identity(claim)
        return PostgresJsonlFileWriter(
            self,
            filename,
            purpose,
            owner,
            producer_batch_id=claim.batch_id,
            producer_fencing_token=claim.fencing_token,
        )

    def _discard_file(self, file_id: str) -> None:
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    cursor.execute(
                        """
                        DELETE FROM batch_files
                        WHERE store_id = %s AND file_id = %s
                        """,
                        (self._store_id, file_id),
                    )

    # -- batches ----------------------------------------------------------

    def create_batch(
        self,
        input_file_id: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: dict | None = None,
        owner: str = "default",
    ) -> BatchJob:
        if endpoint != _SUPPORTED_ENDPOINT:
            raise ValueError(
                f"unsupported endpoint {endpoint!r}; only {_SUPPORTED_ENDPOINT} is supported"
            )
        job = BatchJob(
            id=f"batch_{uuid.uuid4().hex[:24]}",
            endpoint=endpoint,
            input_file_id=input_file_id,
            completion_window=completion_window,
            owner=owner,
            created_at=int(time.time()),
            metadata=metadata,
        )
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    self._load_file(cursor, input_file_id, owner)
                    self._insert_job(cursor, job)
        return job

    def _insert_job(self, cursor: Any, job: BatchJob) -> None:
        cursor.execute(
            """
            INSERT INTO batch_jobs (
                store_id, batch_id, owner, created_at, status, payload,
                output_file_id, error_file_id
            )
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
            """,
            (
                self._store_id,
                job.id,
                job.owner,
                job.created_at,
                job.status,
                self._job_payload(job),
                job.output_file_id,
                job.error_file_id,
            ),
        )

    def _lock_referenced_files(
        self,
        cursor: Any,
        job: BatchJob,
        claim: BatchClaim,
    ) -> set[str]:
        """Validate terminal file ownership and serialize against rollback."""
        seen: set[str] = set()
        for field in ("output_file_id", "error_file_id"):
            file_id = getattr(job, field)
            if file_id is None or file_id in seen:
                continue
            seen.add(file_id)
            cursor.execute(
                """
                SELECT owner, producer_batch_id, producer_fencing_token, pending
                FROM batch_files
                WHERE store_id = %s AND file_id = %s
                FOR KEY SHARE
                """,
                (self._store_id, file_id),
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"{field} {file_id!r} does not exist")
            if row[0] != job.owner:
                raise ValueError(f"{field} {file_id!r} belongs to a different owner")
            if (
                row[1] != claim.batch_id
                or row[2] != claim.fencing_token
                or row[3] is not True
            ):
                raise ValueError(
                    f"{field} {file_id!r} was not produced by the active claim"
                )
        return seen

    def _discard_pending_outputs(
        self,
        cursor: Any,
        batch_id: str,
        *,
        keep_file_ids: set[str] | None = None,
    ) -> None:
        keep = sorted(keep_file_ids or ())
        if keep:
            cursor.execute(
                """
                DELETE FROM batch_files
                WHERE store_id = %s AND producer_batch_id = %s AND pending
                  AND NOT (file_id = ANY(%s))
                """,
                (self._store_id, batch_id, keep),
            )
        else:
            cursor.execute(
                """
                DELETE FROM batch_files
                WHERE store_id = %s AND producer_batch_id = %s AND pending
                """,
                (self._store_id, batch_id),
            )

    def _load_job(
        self,
        cursor: Any,
        batch_id: str,
        owner: str | None,
        *,
        for_update: bool = False,
    ) -> tuple[BatchJob, str | None, int, datetime | None]:
        lock_clause = " FOR UPDATE" if for_update else ""
        cursor.execute(
            f"""
            SELECT payload, claim_worker, fencing_token, lease_until
            FROM batch_jobs
            WHERE store_id = %s AND batch_id = %s
            {lock_clause}
            """,
            (self._store_id, batch_id),
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(batch_id)
        job = BatchJob.model_validate(_json_payload(row[0]))
        if owner is not None and job.owner != owner:
            raise KeyError(batch_id)
        return job, row[1], int(row[2]), row[3]

    def get_batch(
        self,
        batch_id: str,
        owner: str | None = None,
    ) -> BatchJob:
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    job, _worker, _token, _lease = self._load_job(cursor, batch_id, owner)
                    return job

    def list_batches(
        self,
        limit: int = 20,
        owner: str | None = None,
    ) -> list[BatchJob]:
        if limit <= 0:
            return []
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    if owner is None:
                        cursor.execute(
                            """
                            SELECT payload
                            FROM batch_jobs
                            WHERE store_id = %s
                            ORDER BY created_at DESC, batch_id DESC
                            LIMIT %s
                            """,
                            (self._store_id, limit),
                        )
                    else:
                        cursor.execute(
                            """
                            SELECT payload
                            FROM batch_jobs
                            WHERE store_id = %s AND owner = %s
                            ORDER BY created_at DESC, batch_id DESC
                            LIMIT %s
                            """,
                            (self._store_id, owner, limit),
                        )
                    return [BatchJob.model_validate(_json_payload(row[0])) for row in cursor]

    def update_batch(self, job: BatchJob) -> None:
        """Compatibility update, with cancellation fencing for claimed jobs.

        Shared workers must use :meth:`finalize_claim` for terminal publication.
        The existing HTTP cancellation path may still set ``status=cancelled``;
        that transition atomically invalidates any active claim.
        """
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT payload, status, claim_worker, fencing_token,
                               lease_until
                        FROM batch_jobs
                        WHERE store_id = %s AND batch_id = %s
                        FOR UPDATE
                        """,
                        (self._store_id, job.id),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise KeyError(job.id)
                    current = BatchJob.model_validate(_json_payload(row[0]))
                    current_status = str(row[1])
                    claim_worker = row[2]
                    fencing_token = int(row[3])
                    lease_until = row[4]
                    if job.model_dump() == current.model_dump():
                        return
                    if current_status in _TERMINAL_STATES:
                        raise BatchClaimError(
                            f"terminal batch {job.id!r} is immutable"
                        )
                    if job.status != "cancelled":
                        raise BatchClaimError(
                            "shared jobs can only be changed through a fenced "
                            "claim or cancellation"
                        )
                    current_fields = current.model_dump()
                    requested_fields = job.model_dump()
                    for field, value in current_fields.items():
                        if field in {"status", "cancelled_at"}:
                            continue
                        if requested_fields[field] != value:
                            raise BatchClaimError(
                                f"cancellation cannot mutate batch field {field!r}"
                            )
                    cancelled = current.model_copy(deep=True)
                    cancelled.status = "cancelled"
                    cursor.execute("SELECT clock_timestamp()")
                    cancelled.cancelled_at = int(cursor.fetchone()[0].timestamp())
                    next_token = fencing_token + 1
                    cursor.execute(
                        """
                        UPDATE batch_jobs
                        SET status = 'cancelled', payload = %s::jsonb,
                            claim_worker = NULL, fencing_token = %s,
                            lease_until = NULL, updated_at = clock_timestamp()
                        WHERE store_id = %s AND batch_id = %s
                        """,
                        (
                            self._job_payload(cancelled),
                            next_token,
                            self._store_id,
                            job.id,
                        ),
                    )
                    self._discard_pending_outputs(cursor, job.id)
                    self._audit(
                        cursor,
                        batch_id=job.id,
                        worker_id=claim_worker,
                        fencing_token=next_token,
                        event="cancel",
                        lease_until=None,
                        details={
                            "previous_status": current.status,
                            "previous_lease_until": (
                                lease_until.isoformat() if lease_until is not None else None
                            ),
                        },
                    )

    def recover_orphans(self) -> tuple[str, ...]:
        """Shared workers reclaim expired leases; startup never fails live jobs."""
        return ()

    # -- distributed claims ----------------------------------------------

    def claim_next_batch(
        self,
        worker_id: str,
        *,
        lease_seconds: float,
    ) -> BatchClaim | None:
        worker_id = _validate_identity(worker_id, name="worker_id")
        lease_seconds = _validate_lease_seconds(lease_seconds)
        with self._lease_lock:
            self._require_open()
            self._ensure_lease_connection()
            with self._lease_connection.transaction():
                with self._lease_connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT batch_id, payload, status, claim_worker,
                               fencing_token, lease_until
                        FROM batch_jobs
                        WHERE store_id = %s
                          AND (
                              (
                                  status = 'validating'
                                  AND claim_worker IS NULL
                                  AND lease_until IS NULL
                              )
                              OR (
                                  status = 'in_progress'
                                  AND (
                                      lease_until IS NULL
                                      OR lease_until <= clock_timestamp()
                                  )
                              )
                          )
                        ORDER BY created_at, batch_id
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                        """,
                        (self._store_id,),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        return None
                    batch_id = str(row[0])
                    job = BatchJob.model_validate(_json_payload(row[1]))
                    previous_status = str(row[2])
                    previous_worker = row[3]
                    previous_token = int(row[4])
                    previous_lease = row[5]
                    token = previous_token + 1
                    self._discard_pending_outputs(cursor, batch_id)
                    cursor.execute("SELECT clock_timestamp()")
                    claimed_at = cursor.fetchone()[0]
                    job.status = "in_progress"
                    if job.in_progress_at is None:
                        job.in_progress_at = int(claimed_at.timestamp())
                    cursor.execute(
                        """
                        UPDATE batch_jobs
                        SET status = 'in_progress', payload = %s::jsonb,
                            claim_worker = %s, fencing_token = %s,
                            lease_until = %s + (%s * interval '1 second'),
                            updated_at = clock_timestamp()
                        WHERE store_id = %s AND batch_id = %s
                        RETURNING lease_until
                        """,
                        (
                            self._job_payload(job),
                            worker_id,
                            token,
                            claimed_at,
                            lease_seconds,
                            self._store_id,
                            batch_id,
                        ),
                    )
                    lease_until = cursor.fetchone()[0]
                    event = "claim" if previous_status == "validating" else "reclaim"
                    self._audit(
                        cursor,
                        batch_id=batch_id,
                        worker_id=worker_id,
                        fencing_token=token,
                        event=event,
                        claimed_at=claimed_at,
                        lease_until=lease_until,
                        details={
                            "previous_worker_id": previous_worker,
                            "previous_fencing_token": previous_token,
                            "previous_lease_until": (
                                previous_lease.isoformat() if previous_lease is not None else None
                            ),
                        },
                    )
                    return BatchClaim(
                        store_id=self._store_id,
                        job=job,
                        worker_id=worker_id,
                        fencing_token=token,
                        claimed_at=claimed_at,
                        lease_until=lease_until,
                    )

    def renew_claim(
        self,
        claim: BatchClaim,
        *,
        lease_seconds: float,
    ) -> BatchClaim:
        self._validate_claim_identity(claim)
        lease_seconds = _validate_lease_seconds(lease_seconds)
        with self._lease_lock:
            self._require_open()
            self._ensure_lease_connection()
            with self._lease_connection.transaction():
                with self._lease_connection.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE batch_jobs
                        SET lease_until = clock_timestamp()
                                + (%s * interval '1 second'),
                            updated_at = clock_timestamp()
                        WHERE store_id = %s AND batch_id = %s
                          AND status = 'in_progress'
                          AND claim_worker = %s
                          AND fencing_token = %s
                          AND lease_until > clock_timestamp()
                        RETURNING lease_until, updated_at
                        """,
                        (
                            lease_seconds,
                            self._store_id,
                            claim.batch_id,
                            claim.worker_id,
                            claim.fencing_token,
                        ),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise StaleBatchClaimError(f"batch claim for {claim.batch_id!r} is stale")
                    lease_until, renewed_at = row
                    self._audit(
                        cursor,
                        batch_id=claim.batch_id,
                        worker_id=claim.worker_id,
                        fencing_token=claim.fencing_token,
                        event="renew",
                        claimed_at=renewed_at,
                        lease_until=lease_until,
                    )
                    return replace(claim, lease_until=lease_until)

    def finalize_claim(
        self,
        claim: BatchClaim,
        job: BatchJob,
    ) -> BatchJob:
        self._validate_claim_identity(claim)
        if job.id != claim.batch_id:
            raise ValueError("claim and terminal job batch IDs differ")
        if job.status not in _TERMINAL_STATES:
            raise ValueError("finalize_claim requires completed, failed, or cancelled status")
        finalized = job.model_copy(deep=True)
        try:
            return self._finalize_claim_once(claim, finalized)
        except (StaleBatchClaimError, ValueError):
            raise
        except Exception:
            # COMMIT acknowledgement can be lost after PostgreSQL made the
            # terminal row durable. Resolve that ambiguity through the
            # independent general connection before telling the worker to
            # compensate its output files.
            try:
                recovered = self._recover_committed_finalization(claim, finalized)
            except Exception:
                logger.warning(
                    "could not resolve PostgreSQL terminal commit outcome",
                    extra={
                        "batch_id": claim.batch_id,
                        "worker_id": claim.worker_id,
                        "fencing_token": claim.fencing_token,
                    },
                    exc_info=True,
                )
            else:
                if recovered is not None:
                    logger.warning(
                        "recovered acknowledged PostgreSQL terminal commit",
                        extra={
                            "batch_id": claim.batch_id,
                            "worker_id": claim.worker_id,
                            "fencing_token": claim.fencing_token,
                        },
                    )
                    return recovered
            raise

    def _finalize_claim_once(
        self,
        claim: BatchClaim,
        finalized: BatchJob,
    ) -> BatchJob:
        with self._lease_lock:
            self._require_open()
            self._ensure_lease_connection()
            with self._lease_connection.transaction():
                with self._lease_connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT payload, status, claim_worker, fencing_token,
                               lease_until, clock_timestamp()
                        FROM batch_jobs
                        WHERE store_id = %s AND batch_id = %s
                        FOR UPDATE
                        """,
                        (self._store_id, claim.batch_id),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise StaleBatchClaimError(f"batch {claim.batch_id!r} no longer exists")
                    current = BatchJob.model_validate(_json_payload(row[0]))
                    status = str(row[1])
                    worker_id = row[2]
                    fencing_token = int(row[3])
                    lease_until = row[4]
                    now = row[5]
                    if (
                        status != "in_progress"
                        or worker_id != claim.worker_id
                        or fencing_token != claim.fencing_token
                        or lease_until is None
                        or lease_until <= now
                    ):
                        raise StaleBatchClaimError(f"batch claim for {claim.batch_id!r} is stale")
                    immutable_fields = (
                        "object",
                        "owner",
                        "endpoint",
                        "input_file_id",
                        "completion_window",
                        "created_at",
                        "in_progress_at",
                        "metadata",
                    )
                    for field in immutable_fields:
                        if getattr(finalized, field) != getattr(current, field):
                            raise ValueError(f"finalize_claim cannot change batch {field}")
                    timestamp_field = f"{finalized.status}_at"
                    for terminal_state in _TERMINAL_STATES:
                        field = f"{terminal_state}_at"
                        if field == timestamp_field:
                            setattr(finalized, field, int(now.timestamp()))
                        elif getattr(finalized, field) != getattr(current, field):
                            raise ValueError(
                                f"finalize_claim cannot set unrelated batch {field}"
                            )
                    referenced_file_ids = self._lock_referenced_files(
                        cursor,
                        finalized,
                        claim,
                    )
                    self._discard_pending_outputs(
                        cursor,
                        finalized.id,
                        keep_file_ids=referenced_file_ids,
                    )
                    cursor.execute(
                        """
                        UPDATE batch_jobs
                        SET status = %s, payload = %s::jsonb,
                            output_file_id = %s, error_file_id = %s,
                            claim_worker = NULL, lease_until = NULL,
                            updated_at = statement_timestamp()
                        WHERE store_id = %s AND batch_id = %s
                          AND status = 'in_progress'
                          AND claim_worker = %s
                          AND fencing_token = %s
                          AND lease_until > statement_timestamp()
                        RETURNING updated_at
                        """,
                        (
                            finalized.status,
                            self._job_payload(finalized),
                            finalized.output_file_id,
                            finalized.error_file_id,
                            self._store_id,
                            finalized.id,
                            claim.worker_id,
                            claim.fencing_token,
                        ),
                    )
                    committed = cursor.fetchone()
                    if committed is None:
                        raise StaleBatchClaimError(
                            f"batch claim for {claim.batch_id!r} expired "
                            "before terminal publication"
                        )
                    finalized_at = committed[0]
                    if referenced_file_ids:
                        cursor.execute(
                            """
                            UPDATE batch_files
                            SET pending = FALSE
                            WHERE store_id = %s
                              AND producer_batch_id = %s
                              AND producer_fencing_token = %s
                              AND pending
                              AND file_id = ANY(%s)
                            """,
                            (
                                self._store_id,
                                finalized.id,
                                claim.fencing_token,
                                sorted(referenced_file_ids),
                            ),
                        )
                        if cursor.rowcount != len(referenced_file_ids):
                            raise RuntimeError(
                                "not all fenced batch output files were published"
                            )
                    self._audit(
                        cursor,
                        batch_id=finalized.id,
                        worker_id=claim.worker_id,
                        fencing_token=claim.fencing_token,
                        event="finalize",
                        claimed_at=finalized_at,
                        lease_until=None,
                        details={"status": finalized.status},
                    )
                    return finalized

    def _recover_committed_finalization(
        self,
        claim: BatchClaim,
        expected: BatchJob,
    ) -> BatchJob | None:
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT payload, status, claim_worker, fencing_token,
                               lease_until, output_file_id, error_file_id
                        FROM batch_jobs
                        WHERE store_id = %s AND batch_id = %s
                        """,
                        (self._store_id, claim.batch_id),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        return None
                    observed = BatchJob.model_validate(_json_payload(row[0]))
                    if (
                        str(row[1]) != expected.status
                        or row[2] is not None
                        or int(row[3]) != claim.fencing_token
                        or row[4] is not None
                        or row[5] != expected.output_file_id
                        or row[6] != expected.error_file_id
                        or observed.output_file_id != expected.output_file_id
                        or observed.error_file_id != expected.error_file_id
                    ):
                        return None
                    return observed

    def cancel_batch(
        self,
        batch_id: str,
        *,
        owner: str | None = None,
    ) -> BatchJob:
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    job, worker_id, token, previous_lease = self._load_job(
                        cursor,
                        batch_id,
                        owner,
                        for_update=True,
                    )
                    if job.status not in _CLAIMABLE_STATES:
                        return job
                    cursor.execute("SELECT clock_timestamp()")
                    now = cursor.fetchone()[0]
                    job.status = "cancelled"
                    job.cancelled_at = job.cancelled_at or int(now.timestamp())
                    next_token = token + 1
                    cursor.execute(
                        """
                        UPDATE batch_jobs
                        SET status = 'cancelled', payload = %s::jsonb,
                            claim_worker = NULL, fencing_token = %s,
                            lease_until = NULL, updated_at = clock_timestamp()
                        WHERE store_id = %s AND batch_id = %s
                        """,
                        (
                            self._job_payload(job),
                            next_token,
                            self._store_id,
                            batch_id,
                        ),
                    )
                    self._discard_pending_outputs(cursor, batch_id)
                    self._audit(
                        cursor,
                        batch_id=batch_id,
                        worker_id=worker_id,
                        fencing_token=next_token,
                        event="cancel",
                        claimed_at=now,
                        lease_until=None,
                        details={
                            "previous_fencing_token": token,
                            "previous_lease_until": (
                                previous_lease.isoformat() if previous_lease is not None else None
                            ),
                        },
                    )
                    return job

    def _validate_claim_identity(self, claim: BatchClaim) -> None:
        if claim.store_id != self._store_id:
            raise ValueError(
                f"claim belongs to store_id {claim.store_id!r}, not {self._store_id!r}"
            )
        _validate_identity(claim.worker_id, name="claim.worker_id")
        if claim.fencing_token <= 0:
            raise ValueError("claim.fencing_token must be positive")

    def _audit(
        self,
        cursor: Any,
        *,
        batch_id: str,
        worker_id: str | None,
        fencing_token: int,
        event: Literal["claim", "reclaim", "renew", "finalize", "cancel"],
        lease_until: datetime | None,
        claimed_at: datetime | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        cursor.execute(
            """
            INSERT INTO batch_claim_audit (
                store_id, batch_id, worker_id, fencing_token, claimed_at,
                event, lease_until, details
            )
            VALUES (
                %s, %s, %s, %s, COALESCE(%s, clock_timestamp()),
                %s, %s, %s::jsonb
            )
            """,
            (
                self._store_id,
                batch_id,
                worker_id,
                fencing_token,
                claimed_at,
                event,
                lease_until,
                json.dumps(details or {}),
            ),
        )

    def export_claim_audit(
        self,
        batch_id: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        with self._lock:
            self._require_open()
            self._ensure_general_connection()
            with self._connection.transaction():
                with self._connection.cursor() as cursor:
                    if batch_id is None:
                        cursor.execute(
                            """
                            SELECT sequence, store_id, batch_id, worker_id,
                                   fencing_token, claimed_at, event, lease_until,
                                   details
                            FROM batch_claim_audit
                            WHERE store_id = %s
                            ORDER BY sequence
                            """,
                            (self._store_id,),
                        )
                    else:
                        cursor.execute(
                            """
                            SELECT sequence, store_id, batch_id, worker_id,
                                   fencing_token, claimed_at, event, lease_until,
                                   details
                            FROM batch_claim_audit
                            WHERE store_id = %s AND batch_id = %s
                            ORDER BY sequence
                            """,
                            (self._store_id, batch_id),
                        )
                    rows = []
                    for row in cursor:
                        rows.append(
                            {
                                "sequence": int(row[0]),
                                "store_id": str(row[1]),
                                "batch_id": str(row[2]),
                                "worker_id": row[3],
                                "fencing_token": int(row[4]),
                                "claimed_at": row[5].isoformat(),
                                "event": str(row[6]),
                                "lease_until": (row[7].isoformat() if row[7] is not None else None),
                                "details": _json_payload(row[8]),
                            }
                        )
                    return tuple(rows)
