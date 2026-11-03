"""Focused shared-queue ownership tests for the batch worker."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import Callable
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
        hold_finalize: bool = False,
    ) -> None:
        self.local = BatchStore(data_dir)
        self.lose_on_renew = lose_on_renew
        self.on_renew_lost: Callable[[], None] | None = None
        self.finalize_started = threading.Event()
        # Set by the test; a held finalize stays in progress until then.
        self.release_finalize = threading.Event()
        if not hold_finalize:
            self.release_finalize.set()
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
            if self.on_renew_lost is not None:
                self.on_renew_lost()
            raise StaleBatchClaimError("test lease loss")
        return replace(
            claim,
            lease_until=datetime.now(UTC) + timedelta(seconds=lease_seconds),
        )

    def finalize_claim(self, claim: _Claim, job: BatchJob) -> BatchJob:
        self.finalize_started.set()
        assert self.release_finalize.wait(10)  # hang bound only
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
    async def reached() -> BatchJob:
        while (job := store.get_batch(batch_id)).status != status:
            await asyncio.sleep(0.01)
        return job

    try:
        # The bound only detects a hang; the worker reaches it in milliseconds.
        return await asyncio.wait_for(reached(), timeout=10)
    except TimeoutError:
        raise AssertionError(f"batch {batch_id} never reached {status}") from None


class _HeldBackend(MockBackend):
    """Keeps each generation in flight until the test sets ``release``."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    async def generate(self, request):
        await asyncio.wait_for(self.release.wait(), timeout=10)  # hang bound only
        return await super().generate(request)


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
        legacy_chat_models={"m"},
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
    backend = _HeldBackend()
    loop = asyncio.get_running_loop()
    # The generation stays in flight until the renewal has failed. The release
    # is queued on the loop before the heartbeat learns of the loss, so the
    # worker records the lost claim before it could publish the result.
    store.on_renew_lost = lambda: loop.call_soon_threadsafe(backend.release.set)
    worker = BatchWorker(
        store,
        {"m": backend},
        worker_id="old-gateway-uid",
        claim_poll_interval_s=0.01,
        claim_lease_seconds=0.15,
        legacy_chat_models={"m"},
    )

    await worker.process(job.id, claim=claim)

    durable = store.get_batch(job.id)
    assert durable.status == "in_progress"
    assert durable.output_file_id is None
    assert store.finalized == []
    assert len(tuple(store.local._files_dir.glob("*.json"))) == 1


async def test_cancellation_during_terminal_commit_preserves_published_files(tmp_path):
    store = _SharedStore(tmp_path / "batch", hold_finalize=True)
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
        legacy_chat_models={"m"},
    )

    task = asyncio.create_task(worker.process(job.id, claim=claim))
    assert await asyncio.to_thread(store.finalize_started.wait, 10)
    # The terminal commit is still held, so the cancellation lands during it.
    task.cancel()
    store.release_finalize.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    durable = store.get_batch(job.id)
    assert durable.status == "completed"
    assert durable.output_file_id is not None
    assert store.read_file_content(durable.output_file_id)
    assert store.finalized == [("gateway-pod-uid-a", 1, "completed")]
