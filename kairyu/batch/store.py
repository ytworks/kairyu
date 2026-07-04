"""Filesystem store for batch files and job state (design m7 D7).

Layout under ``data_dir``: ``files/<id>.bin`` + ``files/<id>.json`` (metadata),
``batches/<id>.json`` (job state, written atomically via rename). No queue
infra; the single in-gateway worker drains jobs. Restart recovery marks
orphaned in-flight jobs failed — honest and simple (single-gateway scope).
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

_SUPPORTED_ENDPOINT = "/v1/chat/completions"


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
    """The full store surface (m10a D3/A8) — worker, routes and builder use
    exactly these eight methods; M11 tenancy ledgers fake this."""

    def save_file(
        self, content: bytes, filename: str, purpose: str, owner: str = "default"
    ) -> FileObject: ...

    def get_file(self, file_id: str, owner: str | None = None) -> FileObject: ...

    def read_file_content(self, file_id: str, owner: str | None = None) -> bytes: ...

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

    # -- files ------------------------------------------------------------

    def save_file(
        self, content: bytes, filename: str, purpose: str, owner: str = "default"
    ) -> FileObject:
        file = FileObject(
            id=f"file-{uuid.uuid4().hex[:24]}",
            bytes=len(content),
            created_at=int(time.time()),
            filename=filename,
            purpose=purpose,
            owner=owner,
        )
        (self._files_dir / f"{file.id}.bin").write_bytes(content)
        self._write_json(self._files_dir / f"{file.id}.json", file.model_dump())
        return file

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
        path = self._batches_dir / f"{batch_id}.json"
        if not path.exists():
            raise KeyError(batch_id)
        job = BatchJob.model_validate_json(path.read_text(encoding="utf-8"))
        if owner is not None and job.owner != owner:
            raise KeyError(batch_id)  # cross-tenant reads as not-found (C3)
        return job

    def list_batches(self, limit: int = 20, owner: str | None = None) -> list[BatchJob]:
        jobs = [
            BatchJob.model_validate_json(path.read_text(encoding="utf-8"))
            for path in self._batches_dir.glob("batch_*.json")
        ]
        if owner is not None:
            jobs = [job for job in jobs if job.owner == owner]  # tenant-scoped list (C3)
        jobs.sort(key=lambda job: (job.created_at, job.id), reverse=True)
        return jobs[:limit]

    def update_batch(self, job: BatchJob) -> None:
        self._write_json(self._batches_dir / f"{job.id}.json", job.model_dump())

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
