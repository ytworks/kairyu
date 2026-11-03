"""Durable asynchronous Chat Completions API and worker tests."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from datetime import UTC, datetime, timedelta

import httpx
import pytest

import kairyu.async_requests.worker as worker_module
from kairyu.async_requests import (
    AsyncRequestState,
    AsyncRequestSubmission,
    AsyncRequestWorker,
    InMemoryRequestStore,
)
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.async_request_routes import add_async_request_routes
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import TenantConfig, TenantLimits
from tests.server._legacy_chat import create_legacy_app

# Hang bound only: every awaited condition is expected within milliseconds.
_HANG_S = 10.0


class _StoreClock:
    """Store time that moves only when the test advances it."""

    def __init__(self) -> None:
        self.now = datetime.now(UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


async def _until(condition) -> None:
    """Poll ``condition``; the bound only turns a hang into a failure."""

    async with asyncio.timeout(_HANG_S):
        while not condition():
            await asyncio.sleep(0.005)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


def _body(content: str = "hello", *, model: str = "m", **updates) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    body.update(updates)
    return body


def _surface(
    *,
    backend=None,
    settings=None,
    tenant_config=None,
    resolved_keys=None,
    async_request_body_limit=None,
    max_records_per_owner=64,
    request_retention_enabled=False,
    store_clock=None,
):
    backend = backend or MockBackend(responses={"hello": "durable answer"})
    app = create_legacy_app(
        {"m": backend},
        settings=settings,
        tenant_config=tenant_config,
        resolved_api_keys=resolved_keys,
        async_request_body_limit=async_request_body_limit,
    )
    store = InMemoryRequestStore(
        max_records_per_owner=max_records_per_owner,
        clock=store_clock,
    )
    worker = AsyncRequestWorker(
        store,
        {"m": backend},
        legacy_chat_models={"m"},
        tenant_limiter=getattr(app.state, "tenant_limiter", None),
        tenant_config=tenant_config,
    )
    add_async_request_routes(
        app,
        store,
        worker,
        request_retention_enabled=request_retention_enabled,
    )
    return app, store, worker


async def _wait_for_state(
    store: InMemoryRequestStore,
    request_id: str,
    state: AsyncRequestState,
) -> None:
    await _until(lambda: store.get(request_id).state is state)


async def test_submit_replay_list_status_and_result_share_chat_dispatch() -> None:
    app, store, worker = _surface()
    headers = {
        "Idempotency-Key": "same-work",
        "X-Kairyu-Deadline-At": (
            datetime.now(UTC) + timedelta(minutes=5)
        ).isoformat(),
    }
    async with _client(app) as client:
        created = await client.post(
            "/v1/async/chat/completions",
            json=_body(),
            headers=headers,
        )
        replay = await client.post(
            "/v1/async/chat/completions",
            json=_body(),
            headers=headers,
        )

        assert created.status_code == 202
        assert created.headers["location"].startswith("/v1/requests/req-")
        receipt = created.json()
        request_id = receipt["request"]["id"]
        assert replay.json()["request"]["id"] == request_id
        assert receipt["result_url"] == f"/v1/requests/{request_id}/result"
        assert store.get(request_id).priority == 1
        assert "body" not in receipt["request"]
        assert "result" not in receipt["request"]
        pending = await client.get(f"/v1/requests/{request_id}/result")
        assert pending.status_code == 202
        assert pending.headers["retry-after"] == "1"
        listed = await client.get("/v1/requests", params={"state": "queued"})
        assert [item["id"] for item in listed.json()["data"]] == [request_id]
        assert "body" not in listed.json()["data"][0]

        assert await worker.process_next() is True

        status = await client.get(f"/v1/requests/{request_id}")
        result = await client.get(f"/v1/requests/{request_id}/result")
        assert status.json()["state"] == "succeeded"
        assert status.json()["has_result"] is True
        assert "body" not in status.json()
        assert "result" not in status.json()
        assert result.status_code == 200
        assert result.json()["object"] == "chat.completion"
        assert result.json()["choices"][0]["message"]["content"] == "durable answer"

        conflict = await client.post(
            "/v1/async/chat/completions",
            json=_body("different"),
            headers=headers,
        )
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "idempotency_conflict"


async def test_routes_enforce_tenant_scope_for_get_list_result_and_cancel(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ASYNC_REQUEST_KEYS", "key-a,key-b")
    keys = frozenset({"key-a", "key-b"})
    tenant_config = TenantConfig.from_mapping(
        key_tenants={"key-a": "tenant-a", "key-b": "tenant-b"},
        resolved_api_keys=keys,
    )
    app, store, _worker = _surface(
        settings=ServerSettings(api_keys_env="ASYNC_REQUEST_KEYS"),
        tenant_config=tenant_config,
        resolved_keys=keys,
    )
    headers_a = {"Authorization": "Bearer key-a"}
    headers_b = {"Authorization": "Bearer key-b"}
    async with _client(app) as client:
        created = await client.post(
            "/v1/async/chat/completions",
            json=_body(),
            headers=headers_a,
        )
        request_id = created.json()["request"]["id"]
        assert store.get(request_id).owner == "tenant-a"

        assert (
            await client.get(f"/v1/requests/{request_id}", headers=headers_b)
        ).status_code == 404
        assert (
            await client.get(f"/v1/requests/{request_id}/result", headers=headers_b)
        ).status_code == 404
        assert (
            await client.post(f"/v1/requests/{request_id}/cancel", headers=headers_b)
        ).status_code == 404
        assert (await client.get("/v1/requests", headers=headers_b)).json()["data"] == []

        cancelled = await client.post(
            f"/v1/requests/{request_id}/cancel",
            headers=headers_a,
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["state"] == "cancelled"


class _CancellableBackend(MockBackend):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def generate(self, request):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _RecordingBackend(MockBackend):
    def __init__(self) -> None:
        super().__init__()
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return await super().generate(request)


async def test_running_cancel_aborts_local_backend_and_preserves_cancelled_state() -> None:
    backend = _CancellableBackend()
    app, store, worker = _surface(backend=backend)
    worker_task = asyncio.create_task(worker.run())
    try:
        async with _client(app) as client:
            created = await client.post("/v1/async/chat/completions", json=_body())
            request_id = created.json()["request"]["id"]
            await asyncio.wait_for(backend.started.wait(), timeout=_HANG_S)
            await _wait_for_state(store, request_id, AsyncRequestState.RUNNING)

            cancelled = await client.post(f"/v1/requests/{request_id}/cancel")
            assert cancelled.json()["state"] == "cancelled"
            await asyncio.wait_for(backend.cancelled.wait(), timeout=_HANG_S)
            await asyncio.sleep(0)
            assert store.get(request_id).state is AsyncRequestState.CANCELLED
    finally:
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


async def test_cancel_during_mark_running_cannot_escape_local_abort() -> None:
    marked = threading.Event()
    release_mark = threading.Event()

    class DelayedMarkStore(InMemoryRequestStore):
        def mark_running(self, claim):
            running = super().mark_running(claim)
            marked.set()
            assert release_mark.wait(timeout=_HANG_S)
            return running

    backend = _RecordingBackend()
    store = DelayedMarkStore()
    worker = AsyncRequestWorker(store, {"m": backend}, legacy_chat_models={"m"})
    request = store.submit(
        AsyncRequestSubmission(endpoint="/v1/chat/completions", body=_body())
    )
    claim = store.claim_next("worker", lease_seconds=30)
    assert claim is not None
    processing = asyncio.create_task(worker.process(claim))
    assert await asyncio.to_thread(marked.wait, _HANG_S)
    store.cancel(request.id)
    worker.notify_cancel(request.id)
    release_mark.set()
    await processing

    assert backend.requests == []
    assert store.get(request.id).state is AsyncRequestState.CANCELLED


class _ImmediateSleepAsyncio:
    """The worker's asyncio with ``sleep`` reduced to a yield.

    Heartbeat intervals and retry delays then pass at once while the loop clock
    stays real, so a long lease bounds only a hang and never judges speed.
    """

    @staticmethod
    async def sleep(delay: float, result=None):
        await asyncio.sleep(0)
        return result

    def __getattr__(self, name: str):
        return getattr(asyncio, name)


async def test_transient_renewal_error_does_not_abort_inference(monkeypatch) -> None:
    # The verdict is the order "failed renewal, retried renewal, inference
    # completes"; a 60 s lease is far beyond the milliseconds this takes.
    monkeypatch.setattr(worker_module, "asyncio", _ImmediateSleepAsyncio())
    renewals: list[str] = []

    class FlakyRenewStore(InMemoryRequestStore):
        failures = 1

        def renew_claim(self, claim, *, lease_seconds):
            if self.failures:
                self.failures -= 1
                renewals.append("failed")
                raise RuntimeError("connection reset")
            renewed = super().renew_claim(claim, lease_seconds=lease_seconds)
            renewals.append("renewed")
            return renewed

    class RenewalSpanningBackend(MockBackend):
        async def generate(self, request):
            # Inference outlasts a failed renewal and the retry that follows it.
            await _until(lambda: renewals[:2] == ["failed", "renewed"])
            return await super().generate(request)

    backend = RenewalSpanningBackend(responses={"hello": "done"})
    # The store's clock stands still: only the worker's own lease deadline
    # (local loop time) decides whether the retry is still in time.
    store = FlakyRenewStore(clock=_StoreClock())
    worker = AsyncRequestWorker(
        store,
        {"m": backend},
        lease_seconds=60,
        legacy_chat_models={"m"},
    )
    request = store.submit(
        AsyncRequestSubmission(endpoint="/v1/chat/completions", body=_body())
    )

    assert await worker.process_next() is True
    completed = store.get(request.id)
    assert store.failures == 0
    assert completed.state is AsyncRequestState.SUCCEEDED
    assert completed.attempt == 1


async def test_unstorable_result_fails_the_request_instead_of_rerunning_it() -> None:
    class RejectingResultStore(InMemoryRequestStore):
        def succeed(self, claim, result):
            raise RuntimeError("unsupported Unicode escape sequence")

    store = RejectingResultStore()
    worker = AsyncRequestWorker(
        store,
        {"m": MockBackend(responses={"hello": "answer"})},
        legacy_chat_models={"m"},
    )
    request = store.submit(
        AsyncRequestSubmission(endpoint="/v1/chat/completions", body=_body())
    )

    assert await worker.process_next() is True
    failed = store.get(request.id)
    assert failed.state is AsyncRequestState.FAILED
    assert failed.error is not None
    assert failed.error.code == "result_persistence_failed"
    assert store.claim_next("other-gateway", lease_seconds=1) is None


async def test_shutdown_requeues_active_inference_without_waiting_for_lease() -> None:
    backend = _CancellableBackend()
    store = InMemoryRequestStore()
    worker = AsyncRequestWorker(
        store,
        {"m": backend},
        lease_seconds=30,
        legacy_chat_models={"m"},
    )
    request = store.submit(
        AsyncRequestSubmission(endpoint="/v1/chat/completions", body=_body())
    )
    worker_task = asyncio.create_task(worker.run())
    await asyncio.wait_for(backend.started.wait(), timeout=_HANG_S)
    worker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker_task

    assert backend.cancelled.is_set()
    assert store.get(request.id).state is AsyncRequestState.QUEUED
    reclaimed = store.claim_next("next-gateway", lease_seconds=30)
    assert reclaimed is not None
    assert reclaimed.request_id == request.id


async def test_heartbeat_keeps_long_inference_claim_alive() -> None:
    clock = _StoreClock()
    granted: list[datetime] = []

    class RenewalRecordingStore(InMemoryRequestStore):
        def renew_claim(self, claim, *, lease_seconds):
            renewed = super().renew_claim(claim, lease_seconds=lease_seconds)
            granted.append(renewed.lease_until)
            return renewed

    class LongInferenceBackend(MockBackend):
        async def generate(self, request):
            # Inference outlives its first 0.15 s lease on the store's clock.
            # Time moves in 0.1 s steps, each only after a renewal granted a
            # lease beyond it, so only the heartbeat keeps the claim valid.
            for _ in range(3):
                target = clock.now + timedelta(seconds=0.1)
                await _until(
                    lambda target=target: any(until > target for until in granted)
                )
                clock.advance(0.1)
            return await super().generate(request)

    backend = LongInferenceBackend(responses={"hello": "done"})
    store = RenewalRecordingStore(clock=clock)
    worker = AsyncRequestWorker(
        store,
        {"m": backend},
        lease_seconds=0.15,
        legacy_chat_models={"m"},
    )
    request = store.submit(
        AsyncRequestSubmission(
            endpoint="/v1/chat/completions",
            body=_body(),
        )
    )

    assert await worker.process_next() is True
    completed = store.get(request.id)
    assert completed.state is AsyncRequestState.SUCCEEDED
    assert completed.attempt == 1


async def test_deadline_shortens_heartbeat_and_aborts_backend_promptly() -> None:
    clock = _StoreClock()

    class DeadlinePassingBackend(_CancellableBackend):
        async def generate(self, request):
            clock.advance(1)  # the deadline passes while inference runs
            return await super().generate(request)

    backend = DeadlinePassingBackend()
    store = InMemoryRequestStore(clock=clock)
    worker = AsyncRequestWorker(
        store,
        {"m": backend},
        lease_seconds=300,
        legacy_chat_models={"m"},
    )
    request = store.submit(
        AsyncRequestSubmission(
            endpoint="/v1/chat/completions",
            body=_body(),
            deadline_at=clock.now + timedelta(seconds=0.12),
        )
    )

    # The backend never returns, so only a renewal can end the inference. A
    # deadline-paced heartbeat renews after 0.04 s; a lease-paced one would
    # first renew after 100 s, so this hang bound separates the two.
    await asyncio.wait_for(worker.process_next(), timeout=_HANG_S)

    assert backend.cancelled.is_set()
    assert store.get(request.id).state is AsyncRequestState.EXPIRED


async def test_worker_replaces_client_priority_with_tenant_batch_class() -> None:
    backend = _RecordingBackend()
    store = InMemoryRequestStore()
    tenant_config = TenantConfig(
        limits={"tenant-a": TenantLimits(interactive_priority=-10, batch_priority=6)}
    )
    worker = AsyncRequestWorker(
        store,
        {"m": backend},
        tenant_config=tenant_config,
        legacy_chat_models={"m"},
    )
    store.submit(
        AsyncRequestSubmission(
            owner="tenant-a",
            endpoint="/v1/chat/completions",
            body=_body(priority=-999),
        )
    )

    assert await worker.process_next() is True
    assert len(backend.requests) == 1
    assert backend.requests[0].priority == 6
    assert backend.requests[0].scheduling_class == "batch"


async def test_shutdown_requeues_a_claim_returned_during_drain_without_dispatch() -> None:
    claim_started = threading.Event()
    release_claim = threading.Event()

    class DelayedClaimStore(InMemoryRequestStore):
        def claim_next(self, *args, **kwargs):
            claim_started.set()
            assert release_claim.wait(timeout=_HANG_S)
            return super().claim_next(*args, **kwargs)

    backend = _RecordingBackend()
    # The store's clock stands still, so the lease never lapses on its own:
    # the request is claimable again only if the worker returned it. The
    # lease also sets the shutdown drain bound, which only detects a hang.
    store = DelayedClaimStore(clock=_StoreClock())
    worker = AsyncRequestWorker(
        store,
        {"m": backend},
        max_concurrency=1,
        lease_seconds=_HANG_S,
        legacy_chat_models={"m"},
    )
    request = store.submit(
        AsyncRequestSubmission(
            endpoint="/v1/chat/completions",
            body=_body(),
        )
    )
    worker_task = asyncio.create_task(worker.run())
    assert await asyncio.to_thread(claim_started.wait, _HANG_S)
    worker_task.cancel()
    release_claim.set()
    with contextlib.suppress(asyncio.CancelledError):
        await worker_task

    assert backend.requests == []
    assert store.get(request.id).state is AsyncRequestState.QUEUED
    assert store.claim_next("next-gateway", lease_seconds=1) is not None


async def test_worker_validation_failure_is_terminal_and_does_not_echo_payload() -> None:
    store = InMemoryRequestStore()
    worker = AsyncRequestWorker(
        store,
        {"m": MockBackend()},
        legacy_chat_models={"m"},
    )
    request = store.submit(
        AsyncRequestSubmission(
            endpoint="/v1/chat/completions",
            body={"model": "m", "messages": "secret-prompt-canary"},
        )
    )

    assert await worker.process_next() is True
    failed = store.get(request.id)
    assert failed.state is AsyncRequestState.FAILED
    assert failed.error is not None
    assert failed.error.code == "invalid_request"
    assert "secret-prompt-canary" not in failed.error.message


async def test_worker_backend_log_records_type_without_secret(caplog) -> None:
    class SecretErrorBackend(MockBackend):
        async def generate(self, request):
            del request
            raise RuntimeError("secret-prompt-canary")

    store = InMemoryRequestStore()
    worker = AsyncRequestWorker(
        store,
        {"m": SecretErrorBackend()},
        legacy_chat_models={"m"},
    )
    request = store.submit(
        AsyncRequestSubmission(endpoint="/v1/chat/completions", body=_body())
    )

    assert await worker.process_next() is True
    assert store.get(request.id).state is AsyncRequestState.FAILED
    assert "RuntimeError" in caplog.text
    assert "secret-prompt-canary" not in caplog.text


async def test_submit_rejects_stream_unknown_model_and_oversized_body() -> None:
    app, store, _worker = _surface(
        settings=ServerSettings(max_chat_body_bytes=10_000),
        async_request_body_limit=100,
    )
    async with _client(app) as client:
        stream = await client.post(
            "/v1/async/chat/completions",
            json=_body(stream=True),
        )
        unknown = await client.post(
            "/v1/async/chat/completions",
            json=_body(model="unknown"),
        )
        oversized = await client.post(
            "/v1/async/chat/completions",
            json=_body("x" * 200),
        )

    assert stream.status_code == 400
    assert stream.json()["error"]["code"] == "stream_not_supported"
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "model_not_found"
    assert oversized.status_code == 413
    assert store.list(limit=10) == []


async def test_auth_precedes_async_body_limit(monkeypatch) -> None:
    monkeypatch.setenv("ASYNC_BODY_AUTH", "secret")
    app, _store, _worker = _surface(
        settings=ServerSettings(
            api_keys_env="ASYNC_BODY_AUTH",
            max_chat_body_bytes=10_000,
        ),
        resolved_keys=frozenset({"secret"}),
        async_request_body_limit=100,
    )
    body = _body("x" * 200)
    async with _client(app) as client:
        unauthenticated = await client.post(
            "/v1/async/chat/completions",
            json=body,
        )
        authenticated = await client.post(
            "/v1/async/chat/completions",
            json=body,
            headers={"Authorization": "Bearer secret"},
        )

    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 413


async def test_async_control_routes_do_not_consume_worker_request_quota() -> None:
    tenant_config = TenantConfig(
        limits={
            "default": TenantLimits(
                requests_per_minute=1,
                request_burst=1,
                tokens_per_minute=10_000,
            )
        }
    )
    app, store, worker = _surface(tenant_config=tenant_config)
    async with _client(app) as client:
        created = await client.post("/v1/async/chat/completions", json=_body())
        request_id = created.json()["request"]["id"]
        for _ in range(3):
            assert (await client.get(f"/v1/requests/{request_id}")).status_code == 200

        assert await worker.process_next() is True
        assert store.get(request_id).state is AsyncRequestState.SUCCEEDED
        rendered_metrics = app.state.metrics.render()[0].decode()
        assert (
            'kairyu_tenant_in_flight_requests{source="async_submit",tenant="default"} 0.0'
            in rendered_metrics
        )

        second = await client.post("/v1/async/chat/completions", json=_body("later"))
        assert second.status_code == 429
        assert second.json()["error"]["code"] == "tenant_rate_limited"


async def test_metrics_exposes_aggregate_async_queue_without_request_data() -> None:
    app, store, worker = _surface()
    secret_prompt = "private-prompt-not-a-label"
    async with _client(app) as client:
        created = await client.post(
            "/v1/async/chat/completions",
            json=_body(secret_prompt),
        )
        request_id = created.json()["request"]["id"]
        queued_metrics = (await client.get("/metrics")).text
        assert await worker.process_next() is True
        completed_metrics = (await client.get("/metrics")).text

    assert 'kairyu_async_request_queue_depth{store="memory"} 1.0' in queued_metrics
    assert (
        'kairyu_async_request_state{state="queued",store="memory"} 1.0'
        in queued_metrics
    )
    assert 'kairyu_async_request_queue_depth{store="memory"} 0.0' in completed_metrics
    assert (
        'kairyu_async_request_state{state="succeeded",store="memory"} 1.0'
        in completed_metrics
    )
    assert (
        'kairyu_async_request_transitions_total{event="succeed",store="memory"} 1.0'
        in completed_metrics
    )
    assert 'kairyu_async_request_attempts_total{store="memory"} 1.0' in completed_metrics
    assert (
        'kairyu_async_request_metrics_snapshot_success{store="memory"} 1.0'
        in completed_metrics
    )
    assert request_id not in completed_metrics
    assert secret_prompt not in completed_metrics


async def test_worker_defers_blocked_tenant_without_starving_another() -> None:
    tenant_config = TenantConfig(
        limits={"tenant-a": TenantLimits(max_in_flight=1)}
    )
    clock = _StoreClock()
    app, store, worker = _surface(tenant_config=tenant_config, store_clock=clock)
    limiter = app.state.tenant_limiter
    held = limiter.acquire("tenant-a")
    assert held.admitted
    blocked = store.submit(
        AsyncRequestSubmission(
            owner="tenant-a",
            endpoint="/v1/chat/completions",
            body=_body("blocked"),
        )
    )
    runnable = store.submit(
        AsyncRequestSubmission(
            owner="tenant-b",
            endpoint="/v1/chat/completions",
            body=_body("runnable"),
        )
    )

    assert await worker.process_next() is True
    assert store.get(blocked.id).state is AsyncRequestState.QUEUED
    assert await worker.process_next() is True
    assert store.get(runnable.id).state is AsyncRequestState.SUCCEEDED
    held.release()
    clock.advance(1.05)  # past the in-flight rejection's 1 s retry hint
    assert await worker.process_next() is True
    assert store.get(blocked.id).state is AsyncRequestState.SUCCEEDED


async def test_invalid_async_metadata_returns_sanitized_client_error() -> None:
    app, store, _worker = _surface()
    async with _client(app) as client:
        response = await client.post(
            "/v1/async/chat/completions",
            json=_body(),
            headers={"Idempotency-Key": "   "},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_async_request"
    assert store.list(limit=10) == []


@pytest.mark.parametrize(
    ("request_retention_enabled", "retry_after"),
    [(False, None), (True, "60")],
)
async def test_tenant_durable_record_capacity_returns_429(
    request_retention_enabled: bool,
    retry_after: str | None,
) -> None:
    app, _store, _worker = _surface(
        max_records_per_owner=1,
        request_retention_enabled=request_retention_enabled,
    )
    async with _client(app) as client:
        first = await client.post("/v1/async/chat/completions", json=_body())
        second = await client.post("/v1/async/chat/completions", json=_body("two"))

    assert first.status_code == 202
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "request_capacity_exhausted"
    # Without retention the terminal record never leaves, so no retry hint.
    assert second.headers.get("Retry-After") == retry_after
