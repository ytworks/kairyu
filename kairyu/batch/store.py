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

    def save_file(self, content: bytes, filename: str, purpose: str) -> FileObject: ...

    def get_file(self, file_id: str) -> FileObject: ...

    def read_file_content(self, file_id: str) -> bytes: ...

    def create_batch(
        self, input_file_id: str, endpoint: str, completion_window: str,
        metadata: dict | None = None,
    ) -> BatchJob: ...

    def get_batch(self, batch_id: str) -> BatchJob: ...

    def list_batches(self, limit: int = 20) -> list[BatchJob]: ...

    def update_batch(self, job: BatchJob) -> None: ...

    def recover_orphans(self) -> tuple[str, ...]: ...


class BatchStore:
    def __init__(self, data_dir: str | Path) -> None:
        self._files_dir = Path(data_dir) / "files"
        self._batches_dir = Path(data_dir) / "batches"
        self._files_dir.mkdir(parents=True, exist_ok=True)
        self._batches_dir.mkdir(parents=True, exist_ok=True)

    # -- files ------------------------------------------------------------

    def save_file(self, content: bytes, filename: str, purpose: str) -> FileObject:
        file = FileObject(
            id=f"file-{uuid.uuid4().hex[:24]}",
            bytes=len(content),
            created_at=int(time.time()),
            filename=filename,
            purpose=purpose,
        )
        (self._files_dir / f"{file.id}.bin").write_bytes(content)
        self._write_json(self._files_dir / f"{file.id}.json", file.model_dump())
        return file

    def get_file(self, file_id: str) -> FileObject:
        path = self._files_dir / f"{file_id}.json"
        if not path.exists():
            raise KeyError(file_id)
        return FileObject.model_validate_json(path.read_text(encoding="utf-8"))

    def read_file_content(self, file_id: str) -> bytes:
        path = self._files_dir / f"{file_id}.bin"
        if not path.exists():
            raise KeyError(file_id)
        return path.read_bytes()

    # -- batches ----------------------------------------------------------

    def create_batch(
        self,
        input_file_id: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: dict | None = None,
    ) -> BatchJob:
        if endpoint != _SUPPORTED_ENDPOINT:
            raise ValueError(
                f"unsupported endpoint {endpoint!r}; only {_SUPPORTED_ENDPOINT} is supported"
            )
        self.get_file(input_file_id)  # KeyError if the input file is missing
        job = BatchJob(
            id=f"batch_{uuid.uuid4().hex[:24]}",
            endpoint=endpoint,
            input_file_id=input_file_id,
            completion_window=completion_window,
            created_at=int(time.time()),
            metadata=metadata,
        )
        self.update_batch(job)
        return job

    def get_batch(self, batch_id: str) -> BatchJob:
        path = self._batches_dir / f"{batch_id}.json"
        if not path.exists():
            raise KeyError(batch_id)
        return BatchJob.model_validate_json(path.read_text(encoding="utf-8"))

    def list_batches(self, limit: int = 20) -> list[BatchJob]:
        jobs = [
            BatchJob.model_validate_json(path.read_text(encoding="utf-8"))
            for path in self._batches_dir.glob("batch_*.json")
        ]
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
