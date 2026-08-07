"""Global concurrency guard (design m7 D5)."""

import asyncio

import httpx
import pytest

from kairyu.engine.mock import MockBackend
from kairyu.engine.prompt import prompt_text
from kairyu.entrypoints.server.app import _server_sequence_budget
from kairyu.entrypoints.server.middleware import ConcurrencyLimitMiddleware
from kairyu.entrypoints.server.settings import ServerSettings
from tests.server._legacy_chat import create_legacy_app


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _chat_body(content: str) -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": content}]}


class _BudgetedMockBackend(MockBackend):
    def __init__(self, *, sequence_budget: int, latency_s: float) -> None:
        super().__init__(latency_s=latency_s)
        self.sequence_budget = sequence_budget
        self.active = 0
        self.peak_active = 0
        self.started: list[str] = []

    async def generate(self, request):
        text = prompt_text(request.prompt)
        assert text is not None
        self.started.append(text)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            return await super().generate(request)
        finally:
            self.active -= 1


async def test_saturation_returns_429_with_retry_after():
    app = create_legacy_app(
        engines={"m": MockBackend(latency_s=0.2)},
        settings=ServerSettings(max_concurrency=1),
    )
    async with _client(app) as client:
        first = asyncio.create_task(
            client.post("/v1/chat/completions", json=_chat_body("one"))
        )
        await asyncio.sleep(0.05)  # let the first request occupy the only slot
        second = await client.post("/v1/chat/completions", json=_chat_body("two"))
        assert second.status_code == 429
        assert second.headers["retry-after"] == "1"
        assert second.json()["error"]["code"] == "concurrency_exceeded"
        assert (await first).status_code == 200


async def test_backend_budget_moves_burst_waiting_in_front_of_engine():
    backend = _BudgetedMockBackend(sequence_budget=1, latency_s=0.03)
    app = create_legacy_app(
        engines={"m": backend},
        settings=ServerSettings(
            max_concurrency=3,
            admission_wait_timeout_s=0.5,
        ),
    )
    async with _client(app) as client:
        responses = await asyncio.gather(
            *(
                client.post("/v1/chat/completions", json=_chat_body(str(index)))
                for index in range(3)
            )
        )

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert backend.peak_active == 1


async def test_backend_aware_queue_is_explicit_opt_in():
    backend = _BudgetedMockBackend(sequence_budget=1, latency_s=0.03)
    app = create_legacy_app(
        engines={"m": backend},
        settings=ServerSettings(max_concurrency=2),
    )
    async with _client(app) as client:
        responses = await asyncio.gather(
            client.post("/v1/chat/completions", json=_chat_body("one")),
            client.post("/v1/chat/completions", json=_chat_body("two")),
        )

    assert [response.status_code for response in responses] == [200, 200]
    assert backend.peak_active == 2


async def test_admission_queue_hands_slots_to_waiters_in_arrival_order():
    backend = _BudgetedMockBackend(sequence_budget=1, latency_s=0.03)
    app = create_legacy_app(
        engines={"m": backend},
        settings=ServerSettings(
            max_concurrency=3,
            admission_wait_timeout_s=0.5,
        ),
    )
    async with _client(app) as client:
        tasks = []
        for content in ("one", "two", "three"):
            tasks.append(
                asyncio.create_task(
                    client.post("/v1/chat/completions", json=_chat_body(content))
                )
            )
            await asyncio.sleep(0.01)
        responses = await asyncio.gather(*tasks)

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert [
        next(content for content in ("one", "two", "three") if content in prompt)
        for prompt in backend.started
    ] == ["one", "two", "three"]


async def test_admission_queue_rejects_only_beyond_total_bound():
    backend = _BudgetedMockBackend(sequence_budget=1, latency_s=0.1)
    app = create_legacy_app(
        engines={"m": backend},
        settings=ServerSettings(
            max_concurrency=2,
            admission_wait_timeout_s=0.5,
        ),
    )
    async with _client(app) as client:
        first = asyncio.create_task(
            client.post("/v1/chat/completions", json=_chat_body("one"))
        )
        await asyncio.sleep(0.02)
        queued = asyncio.create_task(
            client.post("/v1/chat/completions", json=_chat_body("two"))
        )
        await asyncio.sleep(0.02)
        rejected = await client.post(
            "/v1/chat/completions", json=_chat_body("three")
        )

        assert rejected.status_code == 429
        assert rejected.json()["error"]["code"] == "concurrency_exceeded"
        metrics = (await client.get("/metrics")).text
        assert "kairyu_admission_active_requests 1.0" in metrics
        assert "kairyu_admission_waiting_requests 1.0" in metrics
        assert 'kairyu_admission_rejections_total{reason="overflow"} 1.0' in metrics
        assert (await first).status_code == 200
        assert (await queued).status_code == 200
        drained_metrics = (await client.get("/metrics")).text
        assert "kairyu_admission_active_requests 0.0" in drained_metrics
        assert "kairyu_admission_waiting_requests 0.0" in drained_metrics


async def test_admission_queue_wait_timeout_returns_429_and_releases_waiter():
    backend = _BudgetedMockBackend(sequence_budget=1, latency_s=0.1)
    app = create_legacy_app(
        engines={"m": backend},
        settings=ServerSettings(
            max_concurrency=2,
            admission_wait_timeout_s=0.01,
        ),
    )
    async with _client(app) as client:
        first = asyncio.create_task(
            client.post("/v1/chat/completions", json=_chat_body("one"))
        )
        await asyncio.sleep(0.02)
        timed_out = await client.post(
            "/v1/chat/completions", json=_chat_body("two")
        )

        assert timed_out.status_code == 429
        assert timed_out.headers["retry-after"] == "1"
        metrics = (await client.get("/metrics")).text
        assert 'kairyu_admission_rejections_total{reason="timeout"} 1.0' in metrics
        assert (await first).status_code == 200
        assert (
            await client.post("/v1/chat/completions", json=_chat_body("three"))
        ).status_code == 200


async def test_cancelled_admission_waiter_releases_queue_capacity():
    backend = _BudgetedMockBackend(sequence_budget=1, latency_s=0.1)
    app = create_legacy_app(
        engines={"m": backend},
        settings=ServerSettings(
            max_concurrency=2,
            admission_wait_timeout_s=0.5,
        ),
    )
    async with _client(app) as client:
        first = asyncio.create_task(
            client.post("/v1/chat/completions", json=_chat_body("one"))
        )
        await asyncio.sleep(0.02)
        cancelled = asyncio.create_task(
            client.post("/v1/chat/completions", json=_chat_body("two"))
        )
        await asyncio.sleep(0.02)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        replacement = asyncio.create_task(
            client.post("/v1/chat/completions", json=_chat_body("three"))
        )

        assert (await first).status_code == 200
        assert (await replacement).status_code == 200


async def test_timeout_after_slot_handoff_serves_without_leaking(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = 0

    async def app(scope, receive, send):
        nonlocal calls
        calls += 1

    middleware = ConcurrencyLimitMiddleware(
        app,
        limit=1,
        total_limit=2,
        wait_timeout_s=0.1,
    )
    middleware._active = 1

    async def timeout_after_handoff(waiter, *, timeout):
        assert timeout == 0.1
        middleware._release_slot()
        assert waiter.done() and not waiter.cancelled()
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", timeout_after_handoff)
    await middleware(
        {"type": "http", "path": "/v1/test"},
        lambda: None,
        lambda message: None,
    )

    assert calls == 1
    assert middleware._active == 0
    assert not middleware._waiters


async def test_slot_is_released_after_completion():
    app = create_legacy_app(
        engines={"m": MockBackend()},
        settings=ServerSettings(max_concurrency=1),
    )
    async with _client(app) as client:
        for _ in range(3):  # sequential requests all fit in the single slot
            response = await client.post("/v1/chat/completions", json=_chat_body("hi"))
            assert response.status_code == 200


async def test_health_is_never_guarded():
    app = create_legacy_app(
        engines={"m": MockBackend(latency_s=0.2)},
        settings=ServerSettings(max_concurrency=1),
    )
    async with _client(app) as client:
        task = asyncio.create_task(
            client.post("/v1/chat/completions", json=_chat_body("one"))
        )
        await asyncio.sleep(0.05)
        assert (await client.get("/health")).status_code == 200
        assert (await task).status_code == 200


def test_server_sequence_budget_requires_exactly_one_advertising_backend():
    backend = _BudgetedMockBackend(sequence_budget=2, latency_s=0)

    assert _server_sequence_budget({"m": backend}) == 2
    assert _server_sequence_budget({"a": backend, "b": backend}) is None
    assert _server_sequence_budget({"m": MockBackend()}) is None

    invalid = MockBackend()
    invalid.sequence_budget = True
    with pytest.raises(TypeError, match="sequence_budget"):
        _server_sequence_budget({"m": invalid})


def test_admission_timeout_requires_total_concurrency_bound():
    with pytest.raises(ValueError, match="requires max_concurrency"):
        ServerSettings(admission_wait_timeout_s=1.0)
