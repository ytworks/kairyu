"""Real-PostgreSQL integration tests for the shared BatchStore.

Set ``KAIRYU_TEST_POSTGRES_DSN`` to opt in.  These tests deliberately do not
substitute SQLite or a mock: row locks, DB-clock leases, foreign keys, and
ordered chunk visibility are the behavior under test.
"""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from kairyu.batch.postgres_store import (
    _DB_CHUNK_BYTES,
    BatchClaim,
    BatchClaimError,
    JsonlWriterProtocol,
    PostgresBatchStore,
    StaleBatchClaimError,
)
from kairyu.batch.store import FileTooLargeError

_POSTGRES_DSN = os.environ.get("KAIRYU_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.postgres

if _POSTGRES_DSN:
    psycopg = pytest.importorskip("psycopg")
else:  # Keep core-only test collection importable.
    psycopg = None


@pytest.fixture
def store_factory(tmp_path: Path):
    assert _POSTGRES_DSN is not None
    store_id = f"pytest-{uuid.uuid4().hex}"
    stores: list[PostgresBatchStore] = []

    def create() -> PostgresBatchStore:
        store = PostgresBatchStore(
            _POSTGRES_DSN,
            store_id=store_id,
            spool_dir=tmp_path,
        )
        stores.append(store)
        return store

    yield create, store_id

    for store in reversed(stores):
        store.close()
    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            "DELETE FROM batch_store_registry WHERE store_id = %s",
            (store_id,),
        )


async def _chunks(payload: bytes, *widths: int):
    offset = 0
    for width in widths:
        yield payload[offset : offset + width]
        offset += width
    if offset < len(payload):
        yield payload[offset:]


def _new_job(store: PostgresBatchStore, *, owner: str = "tenant-a"):
    input_file = store.save_file(
        b'{"custom_id":"line-1","method":"POST","url":'
        b'"/v1/chat/completions","body":{"model":"m","messages":[]}}\n',
        filename="input.jsonl",
        purpose="batch",
        owner=owner,
    )
    return store.create_batch(
        input_file.id,
        "/v1/chat/completions",
        owner=owner,
    )


def _terminal_job(claim: BatchClaim, state: str = "completed"):
    job = claim.job.model_copy(deep=True)
    job.status = state
    return job


async def test_cross_instance_streaming_chunks_lines_and_writer(
    store_factory,
    tmp_path: Path,
) -> None:
    create, store_id = store_factory
    first = create()
    second = create()
    middle = b"x" * (_DB_CHUNK_BYTES + 37)
    payload = b"first\n" + middle + b"\nlast"

    file = await first.save_file_streaming(
        _chunks(payload, 2, 17, _DB_CHUNK_BYTES // 2),
        filename="large.jsonl",
        purpose="batch",
        owner="tenant-a",
        max_bytes=len(payload),
    )

    assert second.store_id == first.store_id == store_id
    assert second.get_file(file.id, owner="tenant-a") == file
    assert second.read_file_content(file.id, owner="tenant-a") == payload
    assert b"".join(second.iter_file_content(file.id, owner="tenant-a")) == payload
    assert list(second.iter_file_lines(file.id, owner="tenant-a")) == [
        b"first\n",
        middle + b"\n",
        b"last",
    ]
    assert [
        line
        async for line in second.iter_file_lines_async(
            file.id,
            owner="tenant-a",
        )
    ] == [b"first\n", middle + b"\n", b"last"]
    with pytest.raises(KeyError):
        second.get_file(file.id, owner="tenant-b")
    with pytest.raises(KeyError):
        second.iter_file_lines(file.id, owner="tenant-b")

    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN) as connection:
        row = connection.execute(
            """
            SELECT count(*), sum(octet_length(data))
            FROM batch_file_chunks
            WHERE store_id = %s AND file_id = %s
            """,
            (store_id, file.id),
        ).fetchone()
    assert row == (2, len(payload))
    assert list(tmp_path.glob("*.spool")) == []

    with pytest.raises(FileTooLargeError):
        await first.save_file_streaming(
            _chunks(b"too-large", 3),
            filename="rejected.jsonl",
            purpose="batch",
            owner="tenant-a",
            max_bytes=8,
        )
    assert list(tmp_path.glob("*.spool")) == []

    writer = first.create_jsonl_writer(
        "output.jsonl",
        "batch_output",
        owner="tenant-a",
    )
    assert isinstance(writer, JsonlWriterProtocol)
    writer.append({"line": 1})
    writer.append({"line": 2})
    output = writer.commit()
    assert list(second.iter_file_lines(output.id, owner="tenant-a")) == [
        b'{"line": 1}\n',
        b'{"line": 2}\n',
    ]
    writer.rollback()
    with pytest.raises(KeyError):
        second.get_file(output.id)
    assert list(tmp_path.glob("*.spool")) == []

    job = first.create_batch(
        file.id,
        "/v1/chat/completions",
        owner="tenant-a",
    )
    assert second.get_batch(job.id, owner="tenant-a") == job
    assert [item.id for item in second.list_batches(owner="tenant-a")] == [job.id]


def test_cross_instance_claim_is_atomic_and_recovery_is_non_destructive(
    store_factory,
) -> None:
    create, _store_id = store_factory
    submitter = create()
    worker_a = create()
    worker_b = create()
    job = _new_job(submitter)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                store.claim_next_batch,
                worker_id,
                lease_seconds=5.0,
            )
            for store, worker_id in (
                (worker_a, "gateway-a"),
                (worker_b, "gateway-b"),
            )
        ]
        claims = [future.result() for future in futures]

    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    reset = winners[0].job.model_copy(deep=True)
    reset.status = "validating"
    with pytest.raises(BatchClaimError):
        submitter.update_batch(reset)
    owner_mutation = winners[0].job.model_copy(deep=True)
    owner_mutation.owner = "tenant-b"
    with pytest.raises(BatchClaimError):
        submitter.update_batch(owner_mutation)
    assert worker_a.claim_next_batch("gateway-a", lease_seconds=5.0) is None
    assert worker_b.claim_next_batch("gateway-b", lease_seconds=5.0) is None
    assert submitter.recover_orphans() == ()
    assert submitter.get_batch(job.id).status == "in_progress"
    audit = submitter.export_claim_audit(job.id)
    assert [row["event"] for row in audit] == ["claim"]
    assert audit[0]["worker_id"] == winners[0].worker_id
    assert audit[0]["fencing_token"] == winners[0].fencing_token == 1


def test_expired_lease_is_reclaimed_and_stale_terminal_write_is_fenced(
    store_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create, store_id = store_factory
    submitter = create()
    first_worker = create()
    second_worker = create()
    job = _new_job(submitter)

    first_claim = first_worker.claim_next_batch(
        "gateway-pod-uid-a",
        lease_seconds=5.0,
    )
    assert first_claim is not None
    abandoned_writer = first_worker.create_claim_jsonl_writer(
        first_claim,
        "abandoned.jsonl",
        "batch_output",
        owner="tenant-a",
    )
    abandoned_writer.append({"abandoned": True})
    abandoned_file = abandoned_writer.commit()
    with pytest.raises(KeyError):
        submitter.get_file(abandoned_file.id)
    assert psycopg is not None
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            """
            UPDATE batch_jobs
            SET lease_until = clock_timestamp() - interval '1 second'
            WHERE store_id = %s AND batch_id = %s
            """,
            (store_id, job.id),
        )
    second_claim = second_worker.claim_next_batch(
        "gateway-pod-uid-b",
        lease_seconds=2.0,
    )
    assert second_claim is not None
    assert second_claim.batch_id == first_claim.batch_id == job.id
    assert second_claim.fencing_token == first_claim.fencing_token + 1
    with pytest.raises(KeyError):
        submitter.get_file(abandoned_file.id)
    with psycopg.connect(_POSTGRES_DSN) as connection:
        assert connection.execute(
            """
            SELECT count(*)
            FROM batch_files
            WHERE store_id = %s AND file_id = %s
            """,
            (store_id, abandoned_file.id),
        ).fetchone() == (0,)

    with pytest.raises(StaleBatchClaimError):
        first_worker.finalize_claim(first_claim, _terminal_job(first_claim))

    removed_writer = second_worker.create_jsonl_writer(
        "removed.jsonl",
        "batch_output",
        owner="tenant-a",
    )
    removed_writer.append({"removed": True})
    removed_file = removed_writer.commit()
    removed_writer.rollback()
    missing_reference = _terminal_job(second_claim)
    missing_reference.error_file_id = removed_file.id
    with pytest.raises(ValueError, match="does not exist"):
        second_worker.finalize_claim(second_claim, missing_reference)

    output_writer = second_worker.create_claim_jsonl_writer(
        second_claim,
        "output.jsonl",
        "batch_output",
        owner="tenant-a",
    )
    output_writer.append({"response": "ok"})
    output_file = output_writer.commit()
    terminal = _terminal_job(second_claim)
    terminal.output_file_id = output_file.id

    finalize_once = second_worker._finalize_claim_once

    def commit_then_lose_ack(claim: BatchClaim, terminal_job):
        finalize_once(claim, terminal_job)
        raise RuntimeError("simulated lost COMMIT acknowledgement")

    monkeypatch.setattr(
        second_worker,
        "_finalize_claim_once",
        commit_then_lose_ack,
    )
    completed = second_worker.finalize_claim(
        second_claim,
        terminal,
    )
    assert completed.status == "completed"
    assert completed.completed_at is not None
    assert completed.output_file_id == output_file.id
    assert submitter.get_batch(job.id) == completed
    mutated_terminal = completed.model_copy(deep=True)
    mutated_terminal.metadata = {"forbidden": True}
    with pytest.raises(BatchClaimError, match="immutable"):
        submitter.update_batch(mutated_terminal)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        output_writer.rollback()
    assert submitter.get_file(output_file.id, owner="tenant-a") == output_file

    audit = submitter.export_claim_audit(job.id)
    assert [row["event"] for row in audit] == [
        "claim",
        "reclaim",
        "finalize",
    ]
    assert [row["worker_id"] for row in audit] == [
        "gateway-pod-uid-a",
        "gateway-pod-uid-b",
        "gateway-pod-uid-b",
    ]
    assert [row["fencing_token"] for row in audit] == [1, 2, 2]

    with psycopg.connect(_POSTGRES_DSN) as connection:
        public_rows = connection.execute(
            """
            SELECT event, worker_id, fencing_token, at, lease_until,
                   previous_lease_until
            FROM kairyu_batch_claim_audit
            WHERE store_id = %s AND batch_id = %s
            ORDER BY sequence
            """,
            (store_id, job.id),
        ).fetchall()
    assert [row[:3] for row in public_rows] == [
        ("claimed", "gateway-pod-uid-a", 1),
        ("reclaimed", "gateway-pod-uid-b", 2),
        ("terminal_committed", "gateway-pod-uid-b", 2),
    ]
    assert all(row[3].tzinfo is not None for row in public_rows)
    assert public_rows[1][5] is not None
    assert public_rows[1][3] >= public_rows[1][5]


def test_renew_blocks_takeover_and_cancel_invalidates_claim(
    store_factory,
) -> None:
    create, _store_id = store_factory
    submitter = create()
    worker = create()
    contender = create()
    job = _new_job(submitter, owner="tenant-a")
    claim = worker.claim_next_batch("gateway-a", lease_seconds=5.0)
    assert claim is not None

    # Content/file work uses ``_lock`` and the general connection. A lease
    # heartbeat must still complete while that path is busy.
    with ThreadPoolExecutor(max_workers=1) as executor:
        with worker._lock:
            renewed = executor.submit(
                worker.renew_claim,
                claim,
                lease_seconds=10.0,
            ).result(timeout=2)
    assert renewed.fencing_token == claim.fencing_token
    assert renewed.lease_until > claim.lease_until
    assert contender.claim_next_batch("gateway-b", lease_seconds=1.0) is None

    with pytest.raises(KeyError):
        contender.cancel_batch(job.id, owner="tenant-b")
    cancelled = contender.cancel_batch(job.id, owner="tenant-a")
    assert cancelled.status == "cancelled"
    assert cancelled.cancelled_at is not None
    with pytest.raises(StaleBatchClaimError):
        worker.finalize_claim(renewed, _terminal_job(renewed))

    assert [row["event"] for row in submitter.export_claim_audit(job.id)] == [
        "claim",
        "renew",
        "cancel",
    ]
    assert submitter.get_batch(job.id, owner="tenant-a").status == "cancelled"

    close_probe = create()
    general_connection = close_probe._connection
    lease_connection = close_probe._lease_connection
    close_probe.close()
    close_probe.close()
    assert general_connection.closed
    assert lease_connection.closed
