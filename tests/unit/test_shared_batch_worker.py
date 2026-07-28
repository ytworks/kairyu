"""Focused shared-queue ownership tests for the batch worker."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import pytest

from kairyu.batch.store import BatchJob, BatchStore, StaleBatchClaimError
from kairyu.batch.worker import BatchWorker
from kairyu.engine.mock import MockBackend

pytestmark = pytest.mark.asyncio


@dataclass(frozen=True)
class _Claim:
    store_id: str
    job: BatchJob
    worker_id: str
    fencing_token: int
    claimed_at: datetime
    lease_until: datetime

    @property
    def batch_id(self) -> str:
        return self.job.id


class _SharedStore:
    """Filesystem payloads plus the claiming seam exercised by BatchWorker."""

    def __init__(
        self,
        data_dir,
        *,
        lose_on_renew: bool = False,
        finalize_delay_s: float = 0.0,
    ) -> None:
        self.local = BatchStore(data_dir)
        self.lose_on_renew = lose_on_renew
        self.finalize_delay_s = finalize_delay_s
        self.finalize_started = threading.Event()
        self.claimed = False
        self.claim_workers: list[str] = []
        self.finalized: list[tuple[str, int, str]] = []

    def __getattr__(self, name: str):
        return getattr(self.local, name)

    def claim_next_batch(
        self,
        worker_id: str,
        *,
        lease_seconds: float,
    ) -> _Claim | None:
        if self.claimed:
            return None
        jobs = self.local.list_batches(limit=1)
        if not jobs:
            return None
        job = jobs[0]
        job.status = "in_progress"
        job.in_progress_at = int(datetime.now(UTC).timestamp())
        self.local.update_batch(job)
        self.claimed = True
        self.claim_workers.append(worker_id)
        now = datetime.now(UTC)
        return _Claim(
            store_id="test",
            job=job,
            worker_id=worker_id,
            fencing_token=1,
            claimed_at=now,
            lease_until=now + timedelta(seconds=lease_seconds),
        )

    def renew_claim(
        self,
        claim: _Claim,
        *,
        lease_seconds: float,
    ) -> _Claim:
        if self.lose_on_renew:
            raise StaleBatchClaimError("test lease loss")
        return replace(
            claim,
            lease_until=datetime.now(UTC) + timedelta(seconds=lease_seconds),
        )

    def finalize_claim(self, claim: _Claim, job: BatchJob) -> BatchJob:
        self.finalize_started.set()
        time.sleep(self.finalize_delay_s)
        self.local.update_batch(job)
        self.finalized.append((claim.worker_id, claim.fencing_token, job.status))
        return job


def _batch_line() -> bytes:
    return (
        json.dumps(
            {
                "custom_id": "shared-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "m",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            }
        )
        + "\n"
    ).encode()


async def _wait_for_status(
    store: _SharedStore,
    batch_id: str,
    status: str,
) -> BatchJob:
    for _ in range(100):
        job = store.get_batch(batch_id)
        if job.status == status:
            return job
        await asyncio.sleep(0.01)
    raise AssertionError(f"batch {batch_id} never reached {status}")


async def test_shared_worker_discovers_job_without_process_local_submit(tmp_path):
    store = _SharedStore(tmp_path / "batch")
    input_file = store.save_file(_batch_line(), "input.jsonl", "batch")
    job = store.create_batch(input_file.id, "/v1/chat/completions")
    worker = BatchWorker(
        store,
        {"m": MockBackend()},
        worker_id="gateway-pod-uid-a",
        claim_poll_interval_s=0.01,
        claim_lease_seconds=1.0,
    )

    task = asyncio.create_task(worker.run())
    try:
        completed = await _wait_for_status(store, job.id, "completed")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert completed.output_file_id is not None
    assert store.claim_workers == ["gateway-pod-uid-a"]
    assert store.finalized == [("gateway-pod-uid-a", 1, "completed")]


async def test_lost_shared_claim_rolls_back_output_and_cannot_publish_terminal(tmp_path):
    store = _SharedStore(tmp_path / "batch", lose_on_renew=True)
    input_file = store.save_file(_batch_line(), "input.jsonl", "batch")
    job = store.create_batch(input_file.id, "/v1/chat/completions")
    claim = store.claim_next_batch("old-gateway-uid", lease_seconds=0.15)
    assert claim is not None
    worker = BatchWorker(
        store,
        {"m": MockBackend(latency_s=0.12)},
        worker_id="old-gateway-uid",
        claim_poll_interval_s=0.01,
        claim_lease_seconds=0.15,
    )

    await worker.process(job.id, claim=claim)

    durable = store.get_batch(job.id)
    assert durable.status == "in_progress"
    assert durable.output_file_id is None
    assert store.finalized == []
    assert len(tuple(store.local._files_dir.glob("*.json"))) == 1


async def test_cancellation_during_terminal_commit_preserves_published_files(tmp_path):
    store = _SharedStore(tmp_path / "batch", finalize_delay_s=0.1)
    input_file = store.save_file(_batch_line(), "input.jsonl", "batch")
    job = store.create_batch(input_file.id, "/v1/chat/completions")
    claim = store.claim_next_batch("gateway-pod-uid-a", lease_seconds=1.0)
    assert claim is not None
    worker = BatchWorker(
        store,
        {"m": MockBackend()},
        worker_id="gateway-pod-uid-a",
        claim_poll_interval_s=0.01,
        claim_lease_seconds=1.0,
    )

    task = asyncio.create_task(worker.process(job.id, claim=claim))
    assert await asyncio.to_thread(store.finalize_started.wait, 1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    durable = store.get_batch(job.id)
    assert durable.status == "completed"
    assert durable.output_file_id is not None
    assert store.read_file_content(durable.output_file_id)
    assert store.finalized == [("gateway-pod-uid-a", 1, "completed")]
