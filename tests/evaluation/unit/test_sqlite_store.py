import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from kairyu.evaluation.control_store import (
    FinalizationToken,
    InvalidItemStateTransitionError,
    InvalidRunStateTransitionError,
    LeaseConflictError,
    LeaseToken,
    StoreConflictError,
)
from kairyu.evaluation.safety import SecretSafetyError, SecretValueRegistry
from kairyu.evaluation.schemas import Artifact, BenchmarkRun, ItemState, RunItem, RunState
from kairyu.evaluation.sqlite_store import SqliteControlStore

NOW = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)


class MutableClock:
    def __init__(self, value: datetime = NOW) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def set(self, value: datetime) -> None:
        self.value = value

    def advance(self, **delta) -> None:
        self.value += timedelta(**delta)


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock()


def _run(run_id: str = "run-01", **updates) -> BenchmarkRun:
    values = {
        "run_id": run_id,
        "benchmark_id": "gpqa-diamond",
        "profile": "smoke",
        "mode": "smoke",
        "target_model": "fake-model",
        "protocol_hash": "a" * 64,
        "item_input_manifest_sha256": "b" * 64,
        "created_at": NOW,
    }
    values.update(updates)
    return BenchmarkRun(**values)


def _items(run_id: str = "run-01", count: int = 2) -> tuple[RunItem, ...]:
    return tuple(
        RunItem(
            run_id=run_id,
            item_id=f"item-{ordinal + 1}",
            ordinal=ordinal,
            input_sha256=f"{ordinal + 1:064x}",
        )
        for ordinal in range(count)
    )


def _advance_run(store, lease_token, *states: RunState):
    current = store.get_run(lease_token.run_id)
    for state in states:
        updated = store.compare_and_set_run_state(
            lease_token.run_id,
            lease_token=lease_token,
            expected_state=current.run.state,
            expected_version=current.version,
            new_state=state,
        )
        assert updated is not None
        current = updated
    return current


def test_initialises_versioned_schema_in_wal_mode_and_reopens(tmp_path, clock):
    database = tmp_path / "control.sqlite3"
    first = SqliteControlStore(database, clock=clock)
    first.create_run(_run())

    second = SqliteControlStore(database, clock=clock)

    assert second.get_run("run-01").run == _run()
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall() == [(1,), (2,)]
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert {
        "schema_migrations",
        "benchmark_runs",
        "jobs",
        "events",
        "artifacts",
        "run_items",
    } <= tables


def test_rejects_database_from_a_newer_schema_version(tmp_path, clock):
    database = tmp_path / "future.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at REAL)"
        )
        connection.execute("INSERT INTO schema_migrations(version, applied_at) VALUES (999, 0)")

    with pytest.raises(RuntimeError, match="newer"):
        SqliteControlStore(database, clock=clock)


def test_create_get_list_and_duplicate_run(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    first = store.create_run(_run("run-01"))
    store.create_run(_run("run-02", created_at=NOW + timedelta(seconds=1)))

    assert first.version == 0
    assert first.run.state == RunState.PENDING
    assert [record.run.run_id for record in store.list_runs()] == ["run-02", "run-01"]
    with pytest.raises(StoreConflictError, match="already exists"):
        store.create_run(_run("run-01"))
    with pytest.raises(KeyError):
        store.get_run("missing")


def test_run_state_compare_and_set_is_versioned(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None

    updated = store.compare_and_set_run_state(
        "run-01",
        lease_token=claim.lease_token,
        expected_state=RunState.PENDING,
        expected_version=0,
        new_state=RunState.PREPARING,
    )
    stale = store.compare_and_set_run_state(
        "run-01",
        lease_token=claim.lease_token,
        expected_state=RunState.PENDING,
        expected_version=0,
        new_state=RunState.FAILED,
    )

    assert updated is not None
    assert updated.version == 1
    assert updated.run.state == RunState.PREPARING
    assert stale is None
    assert store.get_run("run-01") == updated


def test_only_one_concurrent_state_compare_and_set_wins(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None

    def update(_attempt: int):
        return store.compare_and_set_run_state(
            "run-01",
            lease_token=claim.lease_token,
            expected_state=RunState.PENDING,
            expected_version=0,
            new_state=RunState.PREPARING,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(update, (1, 2)))

    assert sum(result is not None for result in results) == 1
    assert store.get_run("run-01").version == 1


@pytest.mark.parametrize(
    ("path", "target"),
    (
        ((), RunState.COMPLETED),
        (
            (RunState.PREPARING, RunState.READY, RunState.RUNNING),
            RunState.PREPARING,
        ),
        (
            (
                RunState.PREPARING,
                RunState.READY,
                RunState.RUNNING,
                RunState.COMPLETED,
            ),
            RunState.RUNNING,
        ),
    ),
)
def test_invalid_terminal_and_backward_run_state_transitions_are_rejected(
    tmp_path, clock, path, target
):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    current = store.get_run("run-01")
    for state in path:
        updated = store.compare_and_set_run_state(
            "run-01",
            lease_token=claim.lease_token,
            expected_state=current.run.state,
            expected_version=current.version,
            new_state=state,
        )
        assert updated is not None
        current = updated

    with pytest.raises(InvalidRunStateTransitionError) as error:
        store.compare_and_set_run_state(
            "run-01",
            lease_token=claim.lease_token,
            expected_state=current.run.state,
            expected_version=current.version,
            new_state=target,
        )

    assert error.value.source == current.run.state
    assert error.value.target == target
    assert store.get_run("run-01") == current


def test_claim_heartbeat_and_release_enforce_active_lease_owner(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    queued = store.enqueue_job("run-01", payload={"attempt": 1})

    claimed = store.claim_job("worker-a", lease_seconds=30)

    assert claimed is not None
    assert claimed.job.job_id == queued.job_id
    assert claimed.job.status == "leased"
    assert claimed.job.cancel_requested is False
    assert claimed.lease_token.attempt == 1
    clock.advance(seconds=1)
    assert store.claim_job("worker-b", lease_seconds=30) is None
    wrong_token = LeaseToken(
        job_id=claimed.lease_token.job_id,
        run_id=claimed.lease_token.run_id,
        worker_id="worker-b",
        attempt=claimed.lease_token.attempt,
    )
    with pytest.raises(LeaseConflictError):
        store.heartbeat_job(wrong_token, lease_seconds=30)
    heartbeat = store.heartbeat_job(claimed.lease_token, lease_seconds=60)
    assert heartbeat.lease_expires_at == NOW + timedelta(seconds=61)

    clock.advance(seconds=1)
    _advance_run(
        store,
        claimed.lease_token,
        RunState.PREPARING,
        RunState.READY,
        RunState.RUNNING,
        RunState.COMPLETED,
    )
    released = store.release_job(claimed.lease_token, status="completed")
    assert released.status == "completed"
    assert released.lease_owner is None
    clock.advance(minutes=2)
    assert store.claim_job("worker-b", lease_seconds=30) is None


def test_heartbeat_never_shortens_lease_when_store_clock_regresses(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")
    claimed = store.claim_job("worker-a", lease_seconds=30)
    assert claimed is not None
    original_expiry = claimed.job.lease_expires_at
    assert original_expiry == NOW + timedelta(seconds=30)

    clock.set(NOW - timedelta(seconds=10))
    heartbeat = store.heartbeat_job(claimed.lease_token, lease_seconds=30)

    assert heartbeat.lease_expires_at == original_expiry
    clock.set(NOW + timedelta(seconds=21))
    assert store.claim_job("worker-b", lease_seconds=30) is None
    clock.set(original_expiry)
    reclaimed = store.claim_job("worker-b", lease_seconds=30)
    assert reclaimed is not None
    assert reclaimed.lease_token.attempt == 2


def test_expired_lease_is_reclaimed_and_stale_worker_cannot_mutate_it(tmp_path, clock):
    database = tmp_path / "control.sqlite3"
    first_process = SqliteControlStore(database, clock=clock)
    first_process.create_run(_run())
    job = first_process.enqueue_job("run-01")
    original = first_process.claim_job("worker-a", lease_seconds=10)
    assert original is not None

    clock.advance(seconds=10)
    restarted_process = SqliteControlStore(database, clock=clock)
    reclaimed = restarted_process.claim_job("worker-b", lease_seconds=20)

    assert reclaimed is not None
    assert reclaimed.job.job_id == job.job_id
    assert reclaimed.job.lease_owner == "worker-b"
    assert reclaimed.lease_token.attempt == 2
    clock.advance(seconds=1)
    with pytest.raises(LeaseConflictError):
        first_process.heartbeat_job(original.lease_token, lease_seconds=30)
    with pytest.raises(LeaseConflictError):
        first_process.release_job(original.lease_token)


def test_reclaimed_generation_fences_all_stale_worker_mutations(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")
    original = store.claim_job("worker-shared", lease_seconds=10)
    assert original is not None
    clock.advance(seconds=10)
    reclaimed = store.claim_job("worker-shared", lease_seconds=20)
    assert reclaimed is not None
    assert original.lease_token.attempt == 1
    assert reclaimed.lease_token.attempt == 2
    artifact = Artifact(
        run_id="run-01",
        name="manifest",
        relative_path="manifest.json",
        media_type="application/json",
        sha256="a" * 64,
        size_bytes=12,
        created_at=NOW,
    )
    clock.advance(seconds=1)

    with pytest.raises(LeaseConflictError):
        store.compare_and_set_run_state(
            "run-01",
            lease_token=original.lease_token,
            expected_state=RunState.PENDING,
            expected_version=0,
            new_state=RunState.PREPARING,
        )
    with pytest.raises(LeaseConflictError):
        store.append_event(
            "run-01",
            "stale",
            lease_token=original.lease_token,
        )
    with pytest.raises(LeaseConflictError):
        store.register_artifact(artifact, publication_token=original.lease_token)
    with pytest.raises(LeaseConflictError):
        store.heartbeat_job(original.lease_token, lease_seconds=30)
    with pytest.raises(LeaseConflictError):
        store.release_job(original.lease_token)

    assert store.get_run("run-01").version == 0
    assert store.list_events("run-01") == []
    assert store.list_artifacts("run-01") == []

    updated = store.compare_and_set_run_state(
        "run-01",
        lease_token=reclaimed.lease_token,
        expected_state=RunState.PENDING,
        expected_version=0,
        new_state=RunState.PREPARING,
    )
    event = store.append_event(
        "run-01",
        "current",
        lease_token=reclaimed.lease_token,
    )
    registered = store.register_artifact(
        artifact,
        publication_token=reclaimed.lease_token,
    )

    assert updated is not None
    assert event.sequence == 1
    assert registered == artifact


def test_begin_immediate_prevents_double_claim_under_concurrency(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")

    def claim(worker_id: str):
        return store.claim_job(worker_id, lease_seconds=30)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ("worker-a", "worker-b")))

    assert sum(result is not None for result in results) == 1
    winner = next(result for result in results if result)
    assert store.get_job(winner.job.job_id).attempts == 1


def test_lower_attempt_job_is_claimed_before_requeued_poison_job(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run("run-poison"))
    store.enqueue_job("run-poison", job_id="job-poison")
    poison_claim = store.claim_job("worker-a", lease_seconds=30)
    assert poison_claim is not None
    assert poison_claim.job.job_id == "job-poison"
    store.release_job(poison_claim.lease_token, status="queued")

    clock.advance(seconds=1)
    store.create_run(_run("run-healthy", created_at=clock.value))
    store.enqueue_job("run-healthy", job_id="job-healthy")

    healthy_claim = store.claim_job("worker-b", lease_seconds=30)

    assert healthy_claim is not None
    assert healthy_claim.job.job_id == "job-healthy"
    assert healthy_claim.lease_token.attempt == 1
    assert store.get_job("job-poison").attempts == 1


def test_events_are_transactionally_sequenced_and_pageable(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None

    first = store.append_event(
        "run-01",
        "prepared",
        {"count": 2},
        lease_token=claim.lease_token,
    )
    clock.advance(seconds=1)
    second = store.append_event(
        "run-01",
        "started",
        {"worker": "worker-a"},
        lease_token=claim.lease_token,
    )

    assert (first.sequence, second.sequence) == (1, 2)
    assert first.created_at == NOW
    assert second.created_at == NOW + timedelta(seconds=1)
    assert store.list_events("run-01", after_sequence=1) == [second]
    with pytest.raises(LeaseConflictError):
        store.append_event("missing", "started", lease_token=claim.lease_token)
    with pytest.raises(KeyError):
        store.list_events("missing")


def test_artifact_registration_is_immutable_and_idempotent(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    artifact = Artifact(
        run_id="run-01",
        name="manifest",
        relative_path="manifest.json",
        media_type="application/json",
        sha256="a" * 64,
        size_bytes=12,
        created_at=NOW,
    )

    assert store.register_artifact(artifact, publication_token=claim.lease_token) == artifact
    assert store.register_artifact(artifact, publication_token=claim.lease_token) == artifact
    assert store.list_artifacts("run-01") == [artifact]

    changed = artifact.model_copy(update={"sha256": "b" * 64})
    with pytest.raises(StoreConflictError, match="different contents"):
        store.register_artifact(changed, publication_token=claim.lease_token)
    same_path_different_name = artifact.model_copy(update={"name": "manifest-copy"})
    with pytest.raises(StoreConflictError, match="different contents"):
        store.register_artifact(
            same_path_different_name,
            publication_token=claim.lease_token,
        )
    same_name_different_path = artifact.model_copy(update={"relative_path": "manifest-copy.json"})
    with pytest.raises(StoreConflictError, match="different contents"):
        store.register_artifact(
            same_name_different_path,
            publication_token=claim.lease_token,
        )
    assert store.list_artifacts("run-01") == [artifact]


@pytest.mark.parametrize("lease_seconds", (0, -1))
def test_lease_duration_must_be_positive(tmp_path, clock, lease_seconds):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")

    with pytest.raises(ValueError, match="positive"):
        store.claim_job("worker-a", lease_seconds=lease_seconds)


def test_injected_clock_must_be_timezone_aware(tmp_path):
    naive_clock = MutableClock(datetime(2026, 7, 24))

    with pytest.raises(ValueError, match="timezone-aware"):
        SqliteControlStore(tmp_path / "control.sqlite3", clock=naive_clock)


def test_create_run_requires_pending_state(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)

    with pytest.raises(ValueError, match="pending"):
        store.create_run(_run(state=RunState.READY))
    with pytest.raises(ValueError, match="first attempts"):
        store.create_run(_run("run-02", attempt=2, resumed_from_run_id="run-01"))


def test_cancelled_run_resumes_as_immutable_successor(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(
        _run(
            protocol_hash="a" * 64,
            selected_item_ids=("item-1", "item-2", "item-3"),
            expected_full_count=3,
            completed_count=1,
        )
    )
    source_job = store.enqueue_job(
        "run-01",
        job_id="job-01",
        payload={"checkpoint": {"after": "item-1"}},
    )
    clock.advance(seconds=2)
    store.request_cancel("run-01")
    source_run = store.get_run("run-01")
    source_job = store.get_job(source_job.job_id)

    clock.advance(seconds=1)
    result = store.resume_run("run-01", "run-02")

    assert store.get_run("run-01") == source_run
    assert store.get_job(source_job.job_id) == source_job
    assert result.run.run.run_id == "run-02"
    assert result.run.run.state is RunState.PENDING
    assert result.run.run.protocol_hash == source_run.run.protocol_hash
    assert result.run.run.item_input_manifest_sha256 == source_run.run.item_input_manifest_sha256
    assert result.run.run.selected_item_ids == source_run.run.selected_item_ids
    assert result.run.run.expected_full_count == source_run.run.expected_full_count
    assert result.run.run.completed_count == 0
    assert result.run.run.failed_count == 0
    assert result.run.run.skipped_count == 0
    assert result.run.run.partial is False
    assert result.run.run.termination_reason is None
    assert result.run.run.created_at == clock.value
    assert result.run.run.started_at is None
    assert result.run.run.finished_at is None
    assert result.run.run.attempt == 2
    assert result.run.run.resumed_from_run_id == "run-01"
    assert result.run.version == 0
    assert result.job.job_id != source_job.job_id
    assert result.job.run_id == "run-02"
    assert result.job.status == "queued"
    assert result.job.payload == source_job.payload
    assert result.job.attempts == 0
    assert store.claim_job("worker-a", lease_seconds=30) is not None


def test_queued_cancel_derives_partial_from_accounted_items(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(
        _run(
            selected_item_ids=("item-1", "item-2"),
            completed_count=1,
        )
    )
    store.enqueue_job("run-01", job_id="job-01")

    store.request_cancel("run-01")
    cancelled = store.get_run("run-01").run

    assert cancelled.state is RunState.CANCELLED
    assert cancelled.completed_count == 1
    assert cancelled.partial is True


def test_leased_cancel_request_overrides_worker_completion(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    clock.advance(seconds=1)

    cancelling = store.request_cancel("run-01")

    assert cancelling.status == "leased"
    assert cancelling.cancel_requested is True
    assert store.get_run("run-01").run.state is RunState.CANCELLING
    assert store.get_run("run-01").run.started_at == clock.value
    assert store.get_run("run-01").run.finished_at is None

    clock.advance(seconds=1)
    released = store.release_job(claim.lease_token, status="completed")
    cancelled_run = store.get_run("run-01").run

    assert released.status == "cancelled"
    assert released.cancel_requested is True
    assert cancelled_run.state is RunState.CANCELLED
    assert cancelled_run.finished_at == clock.value


def test_blocked_failed_job_resumes_as_successor(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01", payload={"resume": True})
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    _advance_run(store, claim.lease_token, RunState.PREPARING, RunState.BLOCKED)
    store.release_job(claim.lease_token, status="failed")
    source_run = store.get_run("run-01")
    source_job = store.get_job(claim.job.job_id)

    result = store.resume_run("run-01", "run-02")

    assert result.run.run.state is RunState.PENDING
    assert result.job.payload == {"resume": True}
    assert store.get_run("run-01") == source_run
    assert store.get_job(source_job.job_id) == source_job
    with pytest.raises(StoreConflictError, match="immutable"):
        store.request_cancel("run-01")


def test_active_lease_guard_holds_write_fence_across_operation(tmp_path, clock):
    database = tmp_path / "control.sqlite3"
    store = SqliteControlStore(database, clock=clock, busy_timeout_ms=0)
    store.create_run(_run())
    store.enqueue_job("run-01")
    claim = store.claim_job("worker-a", lease_seconds=10)
    assert claim is not None
    clock.advance(seconds=9)

    with store.active_lease(claim.lease_token):
        with sqlite3.connect(database, timeout=0, isolation_level=None) as contender:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                contender.execute("BEGIN IMMEDIATE")
        clock.advance(seconds=2)

    with pytest.raises(LeaseConflictError):
        with store.active_lease(claim.lease_token):
            pass
    reclaimed = store.claim_job("worker-b", lease_seconds=10)
    assert reclaimed is not None
    assert reclaimed.lease_token.attempt == 2


def test_caller_cannot_override_store_clock(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")

    with pytest.raises(TypeError, match="unexpected keyword"):
        store.claim_job(
            "worker-a",
            lease_seconds=30,
            now=NOW + timedelta(days=1),
        )


def test_identifier_secrets_never_reach_sqlite_files(tmp_path, clock):
    secret = "identifier-canary-94d712fef5"
    registry = SecretValueRegistry([secret])
    database = tmp_path / "control.sqlite3"
    store = SqliteControlStore(
        database,
        secret_registry=registry,
        clock=clock,
    )

    with sqlite3.connect(database) as keeper:
        keeper.execute("BEGIN")
        keeper.execute("SELECT * FROM jobs").fetchall()
        store.create_run(_run())

        with pytest.raises(SecretSafetyError) as job_error:
            store.enqueue_job("run-01", job_id=f"job-{secret}")
        assert secret not in str(job_error.value)

        store.enqueue_job("run-01", job_id="job-safe")
        with pytest.raises(SecretSafetyError) as worker_error:
            store.claim_job(f"worker-{secret}", lease_seconds=30)
        assert secret not in str(worker_error.value)

        claim = store.claim_job("worker-safe", lease_seconds=30)
        assert claim is not None
        with pytest.raises(SecretSafetyError) as event_error:
            store.append_event(
                "run-01",
                f"event-{secret}",
                lease_token=claim.lease_token,
            )
        assert secret not in str(event_error.value)

        sqlite_files = (
            database,
            database.with_name(f"{database.name}-wal"),
            database.with_name(f"{database.name}-shm"),
        )
        assert all(path.exists() for path in sqlite_files)
        for sqlite_file in sqlite_files:
            assert secret.encode() not in sqlite_file.read_bytes()


def test_control_store_rejects_registered_secrets_on_every_json_path(tmp_path, clock):
    secret = "sqlite-provider-value-not-shaped-like-a-key"
    registry = SecretValueRegistry([secret])
    database = tmp_path / "control.sqlite3"
    store = SqliteControlStore(database, secret_registry=registry, clock=clock)

    with pytest.raises(SecretSafetyError) as run_error:
        store.create_run(
            _run(
                "run-secret",
                target_model=f"provider/{secret}",
            )
        )
    assert secret not in str(run_error.value)
    with pytest.raises(KeyError):
        store.get_run("run-secret")

    store.create_run(_run())
    with pytest.raises(SecretSafetyError):
        store.enqueue_job(
            "run-01",
            job_id="job-secret",
            payload={"message": f"provider returned {secret}"},
        )
    with pytest.raises(KeyError):
        store.get_job("job-secret")

    store.enqueue_job("run-01", job_id="job-safe")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    with pytest.raises(SecretSafetyError):
        store.append_event(
            "run-01",
            "provider_error",
            {"message": f"provider returned {secret}"},
            lease_token=claim.lease_token,
        )
    assert store.list_events("run-01") == []

    artifact = Artifact(
        run_id="run-01",
        name="provider-output",
        relative_path=f"upstream/{secret}.json",
        media_type="application/json",
        sha256="a" * 64,
        size_bytes=1,
        created_at=NOW,
    )
    with pytest.raises(SecretSafetyError):
        store.register_artifact(
            artifact,
            publication_token=claim.lease_token,
        )
    assert store.list_artifacts("run-01") == []

    for sqlite_file in tmp_path.glob("control.sqlite3*"):
        assert secret.encode() not in sqlite_file.read_bytes()


@pytest.mark.parametrize("paused_state", (RunState.BLOCKED, RunState.NEEDS_USER_ACTION))
def test_paused_failed_job_can_be_cancelled_by_controller(tmp_path, clock, paused_state):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01", job_id="job-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    _advance_run(store, claim.lease_token, RunState.PREPARING, paused_state)
    store.release_job(claim.lease_token, status="failed")
    clock.advance(seconds=1)

    cancelled_job = store.request_cancel("run-01")
    cancelled_run = store.get_run("run-01").run

    assert cancelled_job.status == "cancelled"
    assert cancelled_job.cancel_requested is True
    assert cancelled_run.state is RunState.CANCELLED
    assert cancelled_run.finished_at == clock.value


def test_partial_paused_state_retains_evidence_through_controller_cancel(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01", job_id="job-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    preparing = _advance_run(store, claim.lease_token, RunState.PREPARING)
    blocked = store.compare_and_set_run_state(
        "run-01",
        lease_token=claim.lease_token,
        expected_state=RunState.PREPARING,
        expected_version=preparing.version,
        new_state=RunState.BLOCKED,
        partial=True,
    )
    assert blocked is not None
    assert blocked.run.partial is True
    store.release_job(claim.lease_token, status="failed")

    store.request_cancel("run-01")
    cancelled = store.get_run("run-01").run

    assert cancelled.state is RunState.CANCELLED
    assert cancelled.completed_count == 0
    assert cancelled.failed_count == 0
    assert cancelled.skipped_count == 0
    assert cancelled.partial is True


@pytest.mark.parametrize(
    ("run_path", "released_status"),
    (
        ((), "completed"),
        (
            (
                RunState.PREPARING,
                RunState.READY,
                RunState.RUNNING,
                RunState.COMPLETED,
            ),
            "failed",
        ),
    ),
)
def test_mismatched_terminal_job_release_rolls_back(tmp_path, clock, run_path, released_status):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01", job_id="job-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    if run_path:
        _advance_run(store, claim.lease_token, *run_path)
    before_run = store.get_run("run-01")
    before_job = store.get_job("job-01")

    with pytest.raises(StoreConflictError):
        store.release_job(claim.lease_token, status=released_status)

    assert store.get_run("run-01") == before_run
    assert store.get_job("job-01") == before_job
    store.heartbeat_job(claim.lease_token, lease_seconds=30)


def test_resume_rejects_cancelled_source_without_prepared_identity(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(
        _run(
            protocol_hash=None,
            item_input_manifest_sha256=None,
        )
    )
    store.enqueue_job("run-01", job_id="job-01")
    store.request_cancel("run-01")

    with pytest.raises(StoreConflictError, match="lacks protocol"):
        store.resume_run("run-01", "run-02")

    assert [stored.run.run_id for stored in store.list_runs()] == ["run-01"]


def test_resume_rejects_nonresumable_and_duplicate_successors(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01", job_id="job-01")
    with pytest.raises(StoreConflictError, match="not a resumable"):
        store.resume_run("run-01", "run-02")

    store.request_cancel("run-01")
    first = store.resume_run("run-01", "run-02")
    assert first.run.run.run_id == "run-02"
    with pytest.raises(StoreConflictError, match="already has a successor"):
        store.resume_run("run-01", "run-03")

    store.request_cancel("run-02")
    with pytest.raises(StoreConflictError, match="ID already exists"):
        store.resume_run("run-02", "run-01")


def test_resume_retry_returns_the_existing_exact_successor(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job(
        "run-01",
        job_id="job-01",
        payload={"checkpoint": "item-7"},
    )
    store.request_cancel("run-01")

    first = store.resume_run("run-01", "run-02")
    claimed = store.claim_job("worker-a", lease_seconds=30)
    assert claimed is not None
    clock.advance(seconds=1)
    retry = store.resume_run("run-01", "run-02")

    assert retry.run == first.run
    assert retry.job == claimed.job
    assert retry.job.job_id == first.job.job_id
    assert retry.job.payload == {"checkpoint": "item-7"}
    assert len(store.list_runs()) == 2


def test_concurrent_resume_creates_exactly_one_successor(tmp_path, clock):

    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01", job_id="job-01")
    store.request_cancel("run-01")

    def resume(new_run_id):
        try:
            return store.resume_run("run-01", new_run_id)
        except StoreConflictError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(resume, ("run-02", "run-03")))

    assert sum(result is not None for result in results) == 1
    successor_ids = {stored.run.run_id for stored in store.list_runs()} - {"run-01"}
    assert successor_ids in ({"run-02"}, {"run-03"})
    with sqlite3.connect(store.database_path) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM benchmark_runs WHERE resumed_from_run_id = ?",
                ("run-01",),
            ).fetchone()[0]
            == 1
        )


def test_completed_run_is_not_resumable(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    _advance_run(
        store,
        claim.lease_token,
        RunState.PREPARING,
        RunState.READY,
        RunState.RUNNING,
        RunState.COMPLETED,
    )
    store.release_job(claim.lease_token, status="completed")

    with pytest.raises(StoreConflictError, match="not a resumable"):
        store.resume_run("run-01", "run-02")


def test_finalization_token_requires_matching_terminal_run_and_job(tmp_path, clock):
    database = tmp_path / "control.sqlite3"
    store = SqliteControlStore(database, clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01", job_id="job-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    with pytest.raises(StoreConflictError, match="matching terminal"):
        store.finalization_token("run-01")
    completed = _advance_run(
        store,
        claim.lease_token,
        RunState.PREPARING,
        RunState.READY,
        RunState.RUNNING,
        RunState.COMPLETED,
    )
    with pytest.raises(StoreConflictError, match="matching terminal"):
        store.finalization_token("run-01")
    store.release_job(claim.lease_token, status="completed")

    token = store.finalization_token("run-01")

    assert token == FinalizationToken(run_id="run-01", run_version=completed.version)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE jobs SET status = 'failed' WHERE job_id = 'job-01'")
    with pytest.raises(StoreConflictError, match="matching terminal"):
        store.finalization_token("run-01")
    with pytest.raises(LeaseConflictError, match="stale"):
        with store.publication_guard(token):
            pass


def test_finalization_token_publishes_and_created_at_retry_is_idempotent(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    _advance_run(
        store,
        claim.lease_token,
        RunState.PREPARING,
        RunState.READY,
        RunState.RUNNING,
        RunState.COMPLETED,
    )
    store.release_job(claim.lease_token, status="completed")
    token = store.finalization_token("run-01")
    artifact = Artifact(
        run_id="run-01",
        name="report",
        relative_path="report.json",
        media_type="application/json",
        sha256="a" * 64,
        size_bytes=12,
        created_at=NOW,
    )

    first = store.register_artifact(artifact, publication_token=token)
    retry = store.register_artifact(
        artifact.model_copy(update={"created_at": NOW + timedelta(seconds=5)}),
        publication_token=token,
    )

    assert first == artifact
    assert retry == artifact
    assert store.list_artifacts("run-01") == [artifact]


def test_finalization_token_is_run_and_version_fenced_across_resume(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01", job_id="job-01")
    store.request_cancel("run-01")
    source_token = store.finalization_token("run-01")
    successor = store.resume_run("run-01", "run-02")

    with store.publication_guard(source_token):
        pass
    with pytest.raises(LeaseConflictError, match="does not match"):
        with store.publication_guard(source_token, run_id="run-02"):
            pass
    wrong_run_artifact = Artifact(
        run_id="run-02",
        name="manifest",
        relative_path="manifest.json",
        media_type="application/json",
        sha256="a" * 64,
        size_bytes=1,
        created_at=clock.value,
    )
    with pytest.raises(LeaseConflictError, match="does not match"):
        store.register_artifact(
            wrong_run_artifact,
            publication_token=source_token,
        )
    stale = FinalizationToken(
        run_id="run-01",
        run_version=source_token.run_version + 1,
    )
    with pytest.raises(LeaseConflictError, match="stale"):
        with store.publication_guard(stale):
            pass
    assert store.get_run("run-02") == successor.run


def test_finalization_publication_guard_holds_write_lock(tmp_path, clock):
    database = tmp_path / "control.sqlite3"
    store = SqliteControlStore(database, clock=clock, busy_timeout_ms=0)
    store.create_run(_run())
    store.enqueue_job("run-01")
    store.request_cancel("run-01")
    token = store.finalization_token("run-01")

    with store.publication_guard(token):
        with sqlite3.connect(database, timeout=0, isolation_level=None) as contender:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                contender.execute("BEGIN IMMEDIATE")


def test_migration_two_preserves_a_version_one_database(tmp_path, clock):
    database = tmp_path / "control.sqlite3"
    original = SqliteControlStore(database, clock=clock)
    original.create_run(_run())
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE run_items")
        connection.execute("DELETE FROM schema_migrations WHERE version = 2")

    upgraded = SqliteControlStore(database, clock=clock)

    assert upgraded.get_run("run-01").run == _run()
    assert upgraded.list_run_items("run-01") == []
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall() == [(1,), (2,)]


def test_complete_preparation_atomically_freezes_identity_and_items(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(
        _run(
            protocol_hash=None,
            item_input_manifest_sha256=None,
            expected_full_count=None,
        )
    )
    store.enqueue_job("run-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    preparing = _advance_run(store, claim.lease_token, RunState.PREPARING)
    items = _items()

    ready = store.complete_preparation(
        "run-01",
        lease_token=claim.lease_token,
        expected_version=preparing.version,
        protocol_hash="c" * 64,
        item_input_manifest_sha256="d" * 64,
        expected_full_count=198,
        items=items,
    )
    retry = store.complete_preparation(
        "run-01",
        lease_token=claim.lease_token,
        expected_version=preparing.version,
        protocol_hash="c" * 64,
        item_input_manifest_sha256="d" * 64,
        expected_full_count=198,
        items=items,
    )

    assert ready is not None
    assert ready == retry
    assert ready.version == preparing.version + 1
    assert ready.run.state is RunState.READY
    assert ready.run.protocol_hash == "c" * 64
    assert ready.run.item_input_manifest_sha256 == "d" * 64
    assert ready.run.selected_item_ids == ("item-1", "item-2")
    assert ready.run.expected_full_count == 198
    stored_items = store.list_run_items("run-01")
    assert [record.item for record in stored_items] == list(items)
    assert [record.version for record in stored_items] == [0, 0]
    assert store.get_run_item("run-01", "item-1") == stored_items[0]
    assert store.list_run_items("run-01", after_ordinal=0) == [stored_items[1]]


def test_preparation_failure_rolls_back_all_item_rows_and_run_identity(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run(protocol_hash=None, item_input_manifest_sha256=None))
    store.enqueue_job("run-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    preparing = _advance_run(store, claim.lease_token, RunState.PREPARING)
    invalid_items = (
        _items()[0],
        _items()[1].model_copy(update={"ordinal": 3}),
    )

    with pytest.raises(ValueError, match="contiguous"):
        store.complete_preparation(
            "run-01",
            lease_token=claim.lease_token,
            expected_version=preparing.version,
            protocol_hash="c" * 64,
            item_input_manifest_sha256="d" * 64,
            expected_full_count=198,
            items=invalid_items,
        )

    assert store.get_run("run-01") == preparing
    assert store.list_run_items("run-01") == []
    assert (
        store.complete_preparation(
            "run-01",
            lease_token=claim.lease_token,
            expected_version=preparing.version - 1,
            protocol_hash="c" * 64,
            item_input_manifest_sha256="d" * 64,
            expected_full_count=198,
            items=_items(),
        )
        is None
    )
    assert store.list_run_items("run-01") == []


def test_run_item_transitions_atomically_update_counters_and_are_idempotent(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    preparing = _advance_run(store, claim.lease_token, RunState.PREPARING)
    ready = store.complete_preparation(
        "run-01",
        lease_token=claim.lease_token,
        expected_version=preparing.version,
        protocol_hash="a" * 64,
        item_input_manifest_sha256="b" * 64,
        expected_full_count=198,
        items=_items(),
    )
    assert ready is not None
    running = _advance_run(store, claim.lease_token, RunState.RUNNING)

    started = store.compare_and_set_run_item(
        "run-01",
        "item-1",
        lease_token=claim.lease_token,
        expected_state=ItemState.PENDING,
        expected_version=0,
        new_state=ItemState.RUNNING,
    )
    retry_started = store.compare_and_set_run_item(
        "run-01",
        "item-1",
        lease_token=claim.lease_token,
        expected_state=ItemState.PENDING,
        expected_version=0,
        new_state=ItemState.RUNNING,
    )

    assert started is not None
    assert retry_started == started
    assert started.item.version == 1
    assert started.run.version == running.version + 1
    assert started.run.run.completed_count == 0
    assert store.get_run("run-01") == started.run

    completed = store.compare_and_set_run_item(
        "run-01",
        "item-1",
        lease_token=claim.lease_token,
        expected_state=ItemState.RUNNING,
        expected_version=started.item.version,
        new_state=ItemState.COMPLETED,
        scores={"accuracy": 1.0},
        checkpoint_relative_path="items/item-1.json",
        checkpoint_sha256="e" * 64,
        checkpoint_source_run_id="run-01",
    )
    retry_completed = store.compare_and_set_run_item(
        "run-01",
        "item-1",
        lease_token=claim.lease_token,
        expected_state=ItemState.RUNNING,
        expected_version=started.item.version,
        new_state=ItemState.COMPLETED,
        scores={"accuracy": 1.0},
        checkpoint_relative_path="items/item-1.json",
        checkpoint_sha256="e" * 64,
        checkpoint_source_run_id="run-01",
    )

    assert completed is not None
    assert retry_completed == completed
    assert completed.item.item.state is ItemState.COMPLETED
    assert completed.item.item.checkpoint_source_run_id == "run-01"
    assert completed.run.run.completed_count == 1
    assert completed.run.version == started.run.version + 1
    assert store.get_run("run-01") == completed.run
    assert (
        store.compare_and_set_run_item(
            "run-01",
            "item-1",
            lease_token=claim.lease_token,
            expected_state=ItemState.RUNNING,
            expected_version=started.item.version,
            new_state=ItemState.COMPLETED,
            scores={"accuracy": 0.0},
            checkpoint_relative_path="items/item-1.json",
            checkpoint_sha256="e" * 64,
            checkpoint_source_run_id="run-01",
        )
        is None
    )
    assert store.get_run("run-01") == completed.run

    skipped = store.compare_and_set_run_item(
        "run-01",
        "item-2",
        lease_token=claim.lease_token,
        expected_state=ItemState.PENDING,
        expected_version=0,
        new_state=ItemState.SKIPPED,
    )
    assert skipped is not None
    assert skipped.run.run.completed_count == 1
    assert skipped.run.run.skipped_count == 1
    assert skipped.run.run.failed_count == 0


def test_terminal_timestamp_never_precedes_persisted_item_evidence(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    preparing = _advance_run(store, claim.lease_token, RunState.PREPARING)
    ready = store.complete_preparation(
        "run-01",
        lease_token=claim.lease_token,
        expected_version=preparing.version,
        protocol_hash="a" * 64,
        item_input_manifest_sha256="b" * 64,
        expected_full_count=198,
        items=_items(),
    )
    assert ready is not None
    _advance_run(store, claim.lease_token, RunState.RUNNING)

    first_started = store.compare_and_set_run_item(
        "run-01",
        "item-1",
        lease_token=claim.lease_token,
        expected_state=ItemState.PENDING,
        expected_version=0,
        new_state=ItemState.RUNNING,
    )
    assert first_started is not None
    clock.advance(seconds=20)
    first_completed = store.compare_and_set_run_item(
        "run-01",
        "item-1",
        lease_token=claim.lease_token,
        expected_state=ItemState.RUNNING,
        expected_version=first_started.item.version,
        new_state=ItemState.COMPLETED,
        scores={"accuracy": 1.0},
    )
    assert first_completed is not None

    clock.set(NOW + timedelta(seconds=10))
    second_started = store.compare_and_set_run_item(
        "run-01",
        "item-2",
        lease_token=claim.lease_token,
        expected_state=ItemState.PENDING,
        expected_version=0,
        new_state=ItemState.RUNNING,
    )
    assert second_started is not None
    second_completed = store.compare_and_set_run_item(
        "run-01",
        "item-2",
        lease_token=claim.lease_token,
        expected_state=ItemState.RUNNING,
        expected_version=second_started.item.version,
        new_state=ItemState.COMPLETED,
        scores={"accuracy": 1.0},
    )
    assert second_completed is not None

    terminal = store.compare_and_set_run_state(
        "run-01",
        lease_token=claim.lease_token,
        expected_state=RunState.RUNNING,
        expected_version=second_completed.run.version,
        new_state=RunState.COMPLETED,
    )
    assert terminal is not None
    evidence_timestamp = max(item.updated_at for item in store.list_run_items("run-01"))
    assert terminal.run.finished_at == evidence_timestamp
    assert terminal.updated_at == evidence_timestamp


def test_failed_item_accepts_immutable_checkpoint_provenance(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    preparing = _advance_run(store, claim.lease_token, RunState.PREPARING)
    ready = store.complete_preparation(
        "run-01",
        lease_token=claim.lease_token,
        expected_version=preparing.version,
        protocol_hash="a" * 64,
        item_input_manifest_sha256="b" * 64,
        expected_full_count=198,
        items=_items(count=1),
    )
    assert ready is not None
    _advance_run(store, claim.lease_token, RunState.RUNNING)
    started = store.compare_and_set_run_item(
        "run-01",
        "item-1",
        lease_token=claim.lease_token,
        expected_state=ItemState.PENDING,
        expected_version=0,
        new_state=ItemState.RUNNING,
    )
    assert started is not None

    failed = store.compare_and_set_run_item(
        "run-01",
        "item-1",
        lease_token=claim.lease_token,
        expected_state=ItemState.RUNNING,
        expected_version=started.item.version,
        new_state=ItemState.FAILED,
        error_class="rate_limit",
        checkpoint_relative_path="upstream/checkpoints/item-1.json",
        checkpoint_sha256="e" * 64,
        checkpoint_source_run_id="run-01",
    )
    retry = store.compare_and_set_run_item(
        "run-01",
        "item-1",
        lease_token=claim.lease_token,
        expected_state=ItemState.RUNNING,
        expected_version=started.item.version,
        new_state=ItemState.FAILED,
        error_class="rate_limit",
        checkpoint_relative_path="upstream/checkpoints/item-1.json",
        checkpoint_sha256="e" * 64,
        checkpoint_source_run_id="run-01",
    )

    assert failed is not None
    assert retry == failed
    assert failed.item.item.state is ItemState.FAILED
    assert failed.item.item.error_class == "rate_limit"
    assert failed.item.item.checkpoint_relative_path == "upstream/checkpoints/item-1.json"
    assert failed.item.item.checkpoint_sha256 == "e" * 64
    assert failed.item.item.checkpoint_source_run_id == "run-01"
    assert failed.run.run.failed_count == 1
    assert failed.run.run.completed_count == 0


def test_invalid_item_evidence_rolls_back_item_and_run_together(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    preparing = _advance_run(store, claim.lease_token, RunState.PREPARING)
    ready = store.complete_preparation(
        "run-01",
        lease_token=claim.lease_token,
        expected_version=preparing.version,
        protocol_hash="a" * 64,
        item_input_manifest_sha256="b" * 64,
        expected_full_count=198,
        items=_items(count=1),
    )
    assert ready is not None
    _advance_run(store, claim.lease_token, RunState.RUNNING)
    started = store.compare_and_set_run_item(
        "run-01",
        "item-1",
        lease_token=claim.lease_token,
        expected_state=ItemState.PENDING,
        expected_version=0,
        new_state=ItemState.RUNNING,
    )
    assert started is not None

    with pytest.raises(ValueError, match="all present or absent"):
        store.compare_and_set_run_item(
            "run-01",
            "item-1",
            lease_token=claim.lease_token,
            expected_state=ItemState.RUNNING,
            expected_version=started.item.version,
            new_state=ItemState.COMPLETED,
            checkpoint_relative_path="items/item-1.json",
        )
    with pytest.raises(StoreConflictError, match="current run or one of its ancestors"):
        store.compare_and_set_run_item(
            "run-01",
            "item-1",
            lease_token=claim.lease_token,
            expected_state=ItemState.RUNNING,
            expected_version=started.item.version,
            new_state=ItemState.COMPLETED,
            checkpoint_relative_path="items/item-1.json",
            checkpoint_sha256="e" * 64,
            checkpoint_source_run_id="missing-run",
        )
    with pytest.raises(InvalidItemStateTransitionError):
        store.compare_and_set_run_item(
            "run-01",
            "item-1",
            lease_token=claim.lease_token,
            expected_state=ItemState.RUNNING,
            expected_version=started.item.version,
            new_state=ItemState.RUNNING,
        )

    assert store.get_run_item("run-01", "item-1") == started.item
    assert store.get_run("run-01") == started.run


def test_stale_lease_cannot_prepare_or_mutate_items(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")
    original = store.claim_job("worker-a", lease_seconds=10)
    assert original is not None
    preparing = _advance_run(store, original.lease_token, RunState.PREPARING)
    clock.advance(seconds=10)
    current = store.claim_job("worker-b", lease_seconds=10)
    assert current is not None

    with pytest.raises(LeaseConflictError):
        store.complete_preparation(
            "run-01",
            lease_token=original.lease_token,
            expected_version=preparing.version,
            protocol_hash="a" * 64,
            item_input_manifest_sha256="b" * 64,
            expected_full_count=198,
            items=_items(),
        )
    ready = store.complete_preparation(
        "run-01",
        lease_token=current.lease_token,
        expected_version=preparing.version,
        protocol_hash="a" * 64,
        item_input_manifest_sha256="b" * 64,
        expected_full_count=198,
        items=_items(),
    )
    assert ready is not None
    _advance_run(store, current.lease_token, RunState.RUNNING)
    clock.advance(seconds=10)
    newest = store.claim_job("worker-c", lease_seconds=30)
    assert newest is not None

    with pytest.raises(LeaseConflictError):
        store.compare_and_set_run_item(
            "run-01",
            "item-1",
            lease_token=current.lease_token,
            expected_state=ItemState.PENDING,
            expected_version=0,
            new_state=ItemState.RUNNING,
        )
    updated = store.compare_and_set_run_item(
        "run-01",
        "item-1",
        lease_token=newest.lease_token,
        expected_state=ItemState.PENDING,
        expected_version=0,
        new_state=ItemState.RUNNING,
    )
    assert updated is not None


def test_concurrent_different_item_outcomes_increment_exactly_one_counter(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    preparing = _advance_run(store, claim.lease_token, RunState.PREPARING)
    ready = store.complete_preparation(
        "run-01",
        lease_token=claim.lease_token,
        expected_version=preparing.version,
        protocol_hash="a" * 64,
        item_input_manifest_sha256="b" * 64,
        expected_full_count=198,
        items=_items(count=1),
    )
    assert ready is not None
    _advance_run(store, claim.lease_token, RunState.RUNNING)
    started = store.compare_and_set_run_item(
        "run-01",
        "item-1",
        lease_token=claim.lease_token,
        expected_state=ItemState.PENDING,
        expected_version=0,
        new_state=ItemState.RUNNING,
    )
    assert started is not None

    def finish(state: ItemState):
        return store.compare_and_set_run_item(
            "run-01",
            "item-1",
            lease_token=claim.lease_token,
            expected_state=ItemState.RUNNING,
            expected_version=started.item.version,
            new_state=state,
            error_class="SyntheticError" if state is ItemState.FAILED else None,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(finish, (ItemState.COMPLETED, ItemState.FAILED)))

    assert sum(result is not None for result in results) == 1
    final_run = store.get_run("run-01").run
    assert final_run.completed_count + final_run.failed_count == 1
    assert final_run.skipped_count == 0


def test_resume_keeps_source_items_and_leaves_successor_reconstruction_to_worker(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    store.create_run(_run())
    store.enqueue_job("run-01", job_id="job-01")
    claim = store.claim_job("worker-a", lease_seconds=30)
    assert claim is not None
    preparing = _advance_run(store, claim.lease_token, RunState.PREPARING)
    ready = store.complete_preparation(
        "run-01",
        lease_token=claim.lease_token,
        expected_version=preparing.version,
        protocol_hash="a" * 64,
        item_input_manifest_sha256="b" * 64,
        expected_full_count=198,
        items=_items(),
    )
    assert ready is not None
    _advance_run(store, claim.lease_token, RunState.RUNNING)
    started = store.compare_and_set_run_item(
        "run-01",
        "item-1",
        lease_token=claim.lease_token,
        expected_state=ItemState.PENDING,
        expected_version=0,
        new_state=ItemState.RUNNING,
    )
    assert started is not None
    completed = store.compare_and_set_run_item(
        "run-01",
        "item-1",
        lease_token=claim.lease_token,
        expected_state=ItemState.RUNNING,
        expected_version=started.item.version,
        new_state=ItemState.COMPLETED,
        scores={"accuracy": 1.0},
        checkpoint_relative_path="items/item-1.json",
        checkpoint_sha256="e" * 64,
        checkpoint_source_run_id="run-01",
    )
    assert completed is not None
    store.request_cancel("run-01")
    store.release_job(claim.lease_token, status="failed")
    source_items = store.list_run_items("run-01")

    successor = store.resume_run("run-01", "run-02")

    assert store.list_run_items("run-01") == source_items
    assert successor.run.run.selected_item_ids == ("item-1", "item-2")
    assert store.list_run_items("run-02") == []

    successor_claim = store.claim_job("worker-b", lease_seconds=30)
    assert successor_claim is not None
    successor_preparing = _advance_run(
        store,
        successor_claim.lease_token,
        RunState.PREPARING,
    )
    successor_items = (
        source_items[0].item.model_copy(update={"run_id": "run-02"}),
        source_items[1].item.model_copy(update={"run_id": "run-02"}),
    )
    successor_ready = store.complete_preparation(
        "run-02",
        lease_token=successor_claim.lease_token,
        expected_version=successor_preparing.version,
        protocol_hash="a" * 64,
        item_input_manifest_sha256="b" * 64,
        expected_full_count=198,
        items=successor_items,
    )

    assert successor_ready is not None
    assert successor_ready.run.completed_count == 1
    rebuilt_items = store.list_run_items("run-02")
    assert rebuilt_items[0].item.state is ItemState.COMPLETED
    assert rebuilt_items[0].item.checkpoint_source_run_id == "run-01"
    assert rebuilt_items[1].item.state is ItemState.PENDING


def test_create_run_and_enqueue_commits_run_and_job_together(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)

    result = store.create_run_and_enqueue(
        _run(),
        payload={"adapter": "gpqa-diamond", "mode": "smoke"},
        job_id="job-01",
    )

    assert result.run == store.get_run("run-01")
    assert result.run.version == 0
    assert result.run.updated_at == NOW
    assert result.job == store.get_job("job-01")
    assert result.job.run_id == "run-01"
    assert result.job.status == "queued"
    assert result.job.payload == {"adapter": "gpqa-diamond", "mode": "smoke"}
    assert result.job.attempts == 0
    assert result.job.cancel_requested is False
    assert result.job.created_at == NOW
    assert result.job.updated_at == NOW


def test_create_run_and_enqueue_rolls_back_run_when_job_id_conflicts(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    original = store.create_run_and_enqueue(_run("run-01"), job_id="job-shared")

    with pytest.raises(StoreConflictError, match="run or job already exists"):
        store.create_run_and_enqueue(_run("run-02"), job_id="job-shared")

    assert store.list_runs() == [original.run]
    assert store.get_job("job-shared") == original.job
    with pytest.raises(KeyError):
        store.get_run("run-02")


def test_create_run_and_enqueue_duplicate_run_does_not_insert_job(tmp_path, clock):
    store = SqliteControlStore(tmp_path / "control.sqlite3", clock=clock)
    original = store.create_run_and_enqueue(_run("run-01"), job_id="job-01")

    with pytest.raises(StoreConflictError, match="run or job already exists"):
        store.create_run_and_enqueue(_run("run-01"), job_id="job-02")

    assert store.list_runs() == [original.run]
    assert store.get_job("job-01") == original.job
    with pytest.raises(KeyError):
        store.get_job("job-02")


@pytest.mark.parametrize(
    ("payload", "error"),
    (
        ({"api_key": "plain-looking-value"}, SecretSafetyError),
        ({"nested": {"value": float("nan")}}, ValueError),
        ({"unsupported": object()}, TypeError),
    ),
)
def test_create_run_and_enqueue_invalid_payload_leaves_no_records(
    tmp_path,
    clock,
    payload,
    error,
):
    database = tmp_path / "control.sqlite3"
    store = SqliteControlStore(database, clock=clock)

    with pytest.raises(error):
        store.create_run_and_enqueue(
            _run(),
            payload=payload,
            job_id="job-01",
        )

    assert store.list_runs() == []
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone() == (0,)


def test_create_run_and_enqueue_registered_secret_leaves_no_records(tmp_path, clock):
    database = tmp_path / "control.sqlite3"
    registry = SecretValueRegistry(["known-secret-value"])
    store = SqliteControlStore(database, clock=clock, secret_registry=registry)

    with pytest.raises(SecretSafetyError):
        store.create_run_and_enqueue(
            _run(),
            payload={"metadata": "prefix-known-secret-value-suffix"},
            job_id="job-01",
        )

    assert store.list_runs() == []
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone() == (0,)
