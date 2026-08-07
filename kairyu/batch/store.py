"""Filesystem store for batch files and job state (design m7 D7).

Layout under ``data_dir``: ``files/<id>.bin`` + ``files/<id>.json`` (metadata),
``batches/<id>.json`` (job state, written atomically via rename). No queue
infra; the single in-gateway worker drains jobs. Restart recovery marks
orphaned in-flight jobs failed — honest and simple (single-gateway scope).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from types import TracebackType
from typing import Literal, Protocol, Self

from pydantic import BaseModel, Field

from kairyu.audit_io import BoundedJsonlWriter

_SUPPORTED_ENDPOINT = "/v1/chat/completions"
_FILE_CHUNK_BYTES = 1024 * 1024


async def _run_blocking_cancellation_safe(
    function: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> object:
    """Finish one started thread operation before propagating cancellation."""

    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancelled = error
    try:
        result = task.result()
    except Exception:
        if cancelled is not None:
            raise cancelled from None
        raise
    if cancelled is not None:
        raise cancelled
    return result


class FileTooLargeError(Exception):
    """A streamed file exceeded its configured byte limit."""


class StaleBatchClaimError(RuntimeError):
    """A shared-store worker no longer owns the fenced batch lease."""


class BatchCancelledError(RuntimeError):
    """A filesystem batch was cancelled before terminal publication."""


class FileObject(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    # owning tenant (C3): reads/lists are scoped to it so one tenant can never
    # see another's files; "default" is the keyless / single-tenant owner
    owner: str = "default"


class JsonlFileWriter:
    """Lazy transactional writer for one store-owned JSONL file."""

    def __init__(
        self,
        store: BatchStore,
        filename: str,
        purpose: str,
        owner: str,
    ) -> None:
        self._store = store
        self._filename = filename
        self._purpose = purpose
        self._owner = owner
        self._file_id = f"file-{uuid.uuid4().hex[:24]}"
        self._temporary_path = store._files_dir / f"{self._file_id}.bin.tmp"
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
            self._writer = BoundedJsonlWriter(self._temporary_path)
            self._state = "open"
        self._writer.append(encoded)
        self._bytes_written += len(encoded.encode("utf-8")) + 1

    def commit(self) -> FileObject:
        if self._state == "new":
            raise RuntimeError("cannot commit an empty JSONL writer")
        if self._state != "open":
            raise RuntimeError(f"JSONL writer is already {self._state}")
        assert self._writer is not None
        try:
            self._close_writer()
            file = self._store._commit_file(
                self._temporary_path,
                file_id=self._file_id,
                bytes_written=self._bytes_written,
                filename=self._filename,
                purpose=self._purpose,
                owner=self._owner,
            )
        except Exception:
            self._discard_temporary()
            raise
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
        """Remove this transaction even if it was already committed.

        Batch output finalization may span two files. If publishing the second
        file fails, the worker uses this seam to make the first publication
        invisible again instead of exposing a half-committed result set.
        """
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
        self._temporary_path.unlink(missing_ok=True)


class RequestCounts(BaseModel):
    total: int = 0
    completed: int = 0
    failed: int = 0


class BatchJob(BaseModel):
    id: str
    object: str = "batch"
    endpoint: str
    input_file_id: str
    completion_window: str = "24h"
    owner: str = "default"  # owning tenant (C3)
    status: str = "validating"
    output_file_id: str | None = None
    error_file_id: str | None = None
    created_at: int
    in_progress_at: int | None = None
    completed_at: int | None = None
    failed_at: int | None = None
    cancelled_at: int | None = None
    request_counts: RequestCounts = Field(default_factory=RequestCounts)
    metadata: dict | None = None
    errors: dict | None = None


class BatchStoreProtocol(Protocol):
    """The full store surface (m10a D3/A8) used by workers and routes."""

    def save_file(
        self, content: bytes, filename: str, purpose: str, owner: str = "default"
    ) -> FileObject: ...

    async def save_file_streaming(
        self,
        chunks: AsyncIterator[bytes],
        filename: str,
        purpose: str,
        owner: str = "default",
        max_bytes: int | None = None,
    ) -> FileObject: ...

    def get_file(self, file_id: str, owner: str | None = None) -> FileObject: ...

    def read_file_content(self, file_id: str, owner: str | None = None) -> bytes: ...

    def iter_file_content(
        self, file_id: str, owner: str | None = None
    ) -> Iterator[bytes]: ...

    def iter_file_lines(
        self, file_id: str, owner: str | None = None
    ) -> Iterator[bytes]: ...

    def create_jsonl_writer(
        self, filename: str, purpose: str, owner: str = "default"
    ) -> JsonlFileWriter: ...

    def create_batch(
        self, input_file_id: str, endpoint: str, completion_window: str,
        metadata: dict | None = None, owner: str = "default",
    ) -> BatchJob: ...

    def get_batch(self, batch_id: str, owner: str | None = None) -> BatchJob: ...

    def list_batches(self, limit: int = 20, owner: str | None = None) -> list[BatchJob]: ...

    def update_batch(self, job: BatchJob) -> None: ...

    def recover_orphans(self) -> tuple[str, ...]: ...


class BatchStore:
    def __init__(self, data_dir: str | Path) -> None:
        self._files_dir = Path(data_dir) / "files"
        self._batches_dir = Path(data_dir) / "batches"
        self._files_dir.mkdir(parents=True, exist_ok=True)
        self._batches_dir.mkdir(parents=True, exist_ok=True)
        self._job_lock = threading.RLock()

    # -- files ------------------------------------------------------------

    def save_file(
        self, content: bytes, filename: str, purpose: str, owner: str = "default"
    ) -> FileObject:
        file_id = f"file-{uuid.uuid4().hex[:24]}"
        temporary_path = self._files_dir / f"{file_id}.bin.tmp"
        try:
            temporary_path.write_bytes(content)
            return self._commit_file(
                temporary_path,
                file_id=file_id,
                bytes_written=len(content),
                filename=filename,
                purpose=purpose,
                owner=owner,
            )
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    async def save_file_streaming(
        self,
        chunks: AsyncIterator[bytes],
        filename: str,
        purpose: str,
        owner: str = "default",
        max_bytes: int | None = None,
    ) -> FileObject:
        """Stream chunks into a metadata-last file transaction."""
        file_id = f"file-{uuid.uuid4().hex[:24]}"
        temporary_path = self._files_dir / f"{file_id}.bin.tmp"
        bytes_written = 0
        handle = temporary_path.open("xb")
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

            commit_task = asyncio.create_task(
                asyncio.to_thread(
                    self._commit_file,
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
            file = commit_task.result()
            if cancelled is not None:
                await _run_blocking_cancellation_safe(self._discard_file, file.id)
                raise cancelled
            return file
        finally:
            if not handle.closed:
                await _run_blocking_cancellation_safe(handle.close)
            await _run_blocking_cancellation_safe(
                temporary_path.unlink,
                missing_ok=True,
            )

    def get_file(self, file_id: str, owner: str | None = None) -> FileObject:
        path = self._files_dir / f"{file_id}.json"
        if not path.exists():
            raise KeyError(file_id)
        file = FileObject.model_validate_json(path.read_text(encoding="utf-8"))
        # cross-tenant access reads as not-found so existence never leaks (C3);
        # owner=None is the internal/worker path (no tenant scoping)
        if owner is not None and file.owner != owner:
            raise KeyError(file_id)
        return file

    def read_file_content(self, file_id: str, owner: str | None = None) -> bytes:
        self.get_file(file_id, owner)  # KeyError on missing OR cross-tenant
        return (self._files_dir / f"{file_id}.bin").read_bytes()

    def iter_file_content(
        self, file_id: str, owner: str | None = None
    ) -> Iterator[bytes]:
        self.get_file(file_id, owner)  # validate before response headers
        content_path = self._files_dir / f"{file_id}.bin"

        def chunks() -> Iterator[bytes]:
            with content_path.open("rb") as handle:
                while chunk := handle.read(_FILE_CHUNK_BYTES):
                    yield chunk

        return chunks()

    def iter_file_lines(
        self, file_id: str, owner: str | None = None
    ) -> Iterator[bytes]:
        self.get_file(file_id, owner)  # validate before opening content
        content_path = self._files_dir / f"{file_id}.bin"

        def lines() -> Iterator[bytes]:
            with content_path.open("rb") as handle:
                yield from handle

        return lines()

    def create_jsonl_writer(
        self, filename: str, purpose: str, owner: str = "default"
    ) -> JsonlFileWriter:
        return JsonlFileWriter(self, filename, purpose, owner)

    def _commit_file(
        self,
        temporary_path: Path,
        *,
        file_id: str,
        bytes_written: int,
        filename: str,
        purpose: str,
        owner: str,
    ) -> FileObject:
        file = FileObject(
            id=file_id,
            bytes=bytes_written,
            created_at=int(time.time()),
            filename=filename,
            purpose=purpose,
            owner=owner,
        )
        content_path = self._files_dir / f"{file.id}.bin"
        metadata_path = self._files_dir / f"{file.id}.json"
        temporary_path.replace(content_path)
        try:
            self._write_json(metadata_path, file.model_dump())
        except Exception:
            content_path.unlink(missing_ok=True)
            metadata_path.with_suffix(".tmp").unlink(missing_ok=True)
            raise
        return file

    def _discard_file(self, file_id: str) -> None:
        # Hide metadata first: even if content cleanup then fails, no FileObject
        # remains discoverable through the public store surface.
        (self._files_dir / f"{file_id}.json").unlink(missing_ok=True)
        (self._files_dir / f"{file_id}.bin").unlink(missing_ok=True)
        (self._files_dir / f"{file_id}.tmp").unlink(missing_ok=True)

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
        # the input file must belong to this tenant (KeyError if missing or
        # owned by someone else) — a tenant cannot batch over another's file
        self.get_file(input_file_id, owner)
        job = BatchJob(
            id=f"batch_{uuid.uuid4().hex[:24]}",
            endpoint=endpoint,
            input_file_id=input_file_id,
            completion_window=completion_window,
            owner=owner,
            created_at=int(time.time()),
            metadata=metadata,
        )
        self.update_batch(job)
        return job

    def get_batch(self, batch_id: str, owner: str | None = None) -> BatchJob:
        with self._job_lock:
            path = self._batches_dir / f"{batch_id}.json"
            if not path.exists():
                raise KeyError(batch_id)
            job = BatchJob.model_validate_json(path.read_text(encoding="utf-8"))
            if owner is not None and job.owner != owner:
                raise KeyError(batch_id)  # cross-tenant reads as not-found (C3)
            return job

    def list_batches(self, limit: int = 20, owner: str | None = None) -> list[BatchJob]:
        with self._job_lock:
            jobs = [
                BatchJob.model_validate_json(path.read_text(encoding="utf-8"))
                for path in self._batches_dir.glob("batch_*.json")
            ]
            if owner is not None:
                jobs = [
                    job for job in jobs if job.owner == owner
                ]  # tenant-scoped list (C3)
            jobs.sort(key=lambda job: (job.created_at, job.id), reverse=True)
            return jobs[:limit]

    def update_batch(self, job: BatchJob) -> None:
        with self._job_lock:
            self._write_json(
                self._batches_dir / f"{job.id}.json",
                job.model_dump(),
            )

    def start_batch(self, batch_id: str) -> BatchJob | None:
        """Atomically move a queued filesystem batch into worker ownership."""

        with self._job_lock:
            job = self.get_batch(batch_id)
            if job.status == "cancelled":
                return None
            if job.status != "validating":
                raise RuntimeError(
                    f"cannot start batch {batch_id!r} from state {job.status!r}"
                )
            job.status = "in_progress"
            job.in_progress_at = int(time.time())
            self.update_batch(job)
            return job

    def cancel_batch(
        self,
        batch_id: str,
        owner: str | None = None,
    ) -> BatchJob:
        """Atomically cancel only a non-terminal filesystem batch."""

        with self._job_lock:
            job = self.get_batch(batch_id, owner=owner)
            if job.status in ("validating", "in_progress"):
                job.status = "cancelled"
                job.cancelled_at = int(time.time())
                self.update_batch(job)
            return job

    def finalize_batch(self, job: BatchJob) -> None:
        """Publish one terminal filesystem state unless cancellation won."""

        with self._job_lock:
            current = self.get_batch(job.id)
            if current.status == "cancelled":
                raise BatchCancelledError(job.id)
            if current.status != "in_progress":
                raise RuntimeError(
                    f"cannot finalize batch {job.id!r} from state {current.status!r}"
                )
            self.update_batch(job)

    def recover_orphans(self) -> tuple[str, ...]:
        """Mark jobs left in flight by a previous process as failed (m7 D7)."""
        orphaned = []
        for job in self.list_batches(limit=1_000_000):
            if job.status in ("validating", "in_progress"):
                job.status = "failed"
                job.failed_at = int(time.time())
                job.errors = {
                    "message": "server restarted while the batch was in flight; resubmit"
                }
                self.update_batch(job)
                orphaned.append(job.id)
        return tuple(orphaned)

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
