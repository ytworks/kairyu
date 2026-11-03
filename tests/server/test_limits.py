"""Global concurrency guard (design m7 D5)."""

import asyncio
import inspect
import re

import httpx
import pytest

from kairyu.engine.mock import MockBackend
from kairyu.engine.prompt import prompt_text
from kairyu.entrypoints.server.app import _server_sequence_budget
from kairyu.entrypoints.server.middleware import ConcurrencyLimitMiddleware
from kairyu.entrypoints.server.settings import ServerSettings
from tests.server._legacy_chat import create_legacy_app

# Hang bound only: every awaited condition is expected within milliseconds.
_HANG_S = 10.0


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _chat_body(content: str) -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": content}]}


class _GatedBackend(MockBackend):
    """Holds every generation in flight until the test sets ``release``."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
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
            await self.release.wait()
            return await super().generate(request)
        finally:
            self.active -= 1


class _BudgetedMockBackend(_GatedBackend):
    def __init__(self, *, sequence_budget: int) -> None:
        super().__init__()
        self.sequence_budget = sequence_budget


async def _until(condition) -> None:
    """Wait for ``condition``; the bound only turns a hang into a failure."""

    async def poll() -> None:
        while True:
            satisfied = condition()
            if inspect.isawaitable(satisfied):
                satisfied = await satisfied
            if satisfied:
                return
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=_HANG_S)


async def _while_held(awaitable):
    """Await a call made while the gate is closed; a broken guard that queues
    it behind the held request would otherwise deadlock the test."""

    return await asyncio.wait_for(awaitable, timeout=_HANG_S)


async def _admission_waiting(client) -> int:
    metrics = (await client.get("/metrics")).text
    match = re.search(
        r"^kairyu_admission_waiting_requests (\S+)$", metrics, re.MULTILINE
    )
    assert match is not None, metrics
    return int(float(match.group(1)))


async def test_saturation_returns_429_with_retry_after():
    backend = _GatedBackend()
    app = create_legacy_app(
        engines={"m": backend},
        settings=ServerSettings(max_concurrency=1),
    )
    async with _client(app) as client:
        first = asyncio.create_task(
            client.post("/v1/chat/completions", json=_chat_body("one"))
        )
        await _until(lambda: backend.active == 1)  # the first holds the only slot
        second = await _while_held(
            client.post("/v1/chat/completions", json=_chat_body("two"))
        )
        assert second.status_code == 429
        assert second.headers["retry-after"] == "1"
        assert second.json()["error"]["code"] == "concurrency_exceeded"
        backend.release.set()
        assert (await first).status_code == 200


async def test_backend_budget_moves_burst_waiting_in_front_of_engine():
    backend = _BudgetedMockBackend(sequence_budget=1)
    app = create_legacy_app(
        engines={"m": backend},
        settings=ServerSettings(
            max_concurrency=3,
            admission_wait_timeout_s=_HANG_S,
        ),
    )
    async with _client(app) as client:
        tasks = [
            asyncio.create_task(
                client.post("/v1/chat/completions", json=_chat_body(str(index)))
            )
            for index in range(3)
        ]

        async def whole_burst_arrived() -> bool:
            return len(backend.started) + await _admission_waiting(client) == 3

        # Every request is inside the server while the engine holds the first.
        await _until(whole_burst_arrived)
        assert backend.peak_active == 1
        backend.release.set()
        responses = await asyncio.gather(*tasks)

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert backend.peak_active == 1


async def test_backend_aware_queue_is_explicit_opt_in():
    backend = _BudgetedMockBackend(sequence_budget=1)
    app = create_legacy_app(
        engines={"m": backend},
        settings=ServerSettings(max_concurrency=2),
    )
    async with _client(app) as client:
        tasks = [
            asyncio.create_task(
                client.post("/v1/chat/completions", json=_chat_body(content))
            )
            for content in ("one", "two")
        ]
        # A held request cannot finish, so a finished task is a rejection.
        await _until(
            lambda: len(backend.started) == 2 or any(task.done() for task in tasks)
        )
        assert backend.peak_active == 2
        backend.release.set()
        responses = await asyncio.gather(*tasks)

    assert [response.status_code for response in responses] == [200, 200]
    assert backend.peak_active == 2


async def test_admission_queue_hands_slots_to_waiters_in_arrival_order():
    backend = _BudgetedMockBackend(sequence_budget=1)
    app = create_legacy_app(
        engines={"m": backend},
        settings=ServerSettings(
            max_concurrency=3,
            admission_wait_timeout_s=_HANG_S,
        ),
    )
    async with _client(app) as client:
        tasks = [
            asyncio.create_task(
                client.post("/v1/chat/completions", json=_chat_body("one"))
            )
        ]
        await _until(lambda: backend.active == 1)
        for waiting, content in enumerate(("two", "three"), start=1):
            tasks.append(
                asyncio.create_task(
                    client.post("/v1/chat/completions", json=_chat_body(content))
                )
            )

            async def queued(expected: int = waiting) -> bool:
                return await _admission_waiting(client) == expected

            # Arrival order is fixed: each request is queued before the next.
            await _until(queued)
        backend.release.set()
        responses = await asyncio.gather(*tasks)

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert [
        next(content for content in ("one", "two", "three") if content in prompt)
        for prompt in backend.started
    ] == ["one", "two", "three"]


async def test_admission_queue_rejects_only_beyond_total_bound():
    backend = _BudgetedMockBackend(sequence_budget=1)
    app = create_legacy_app(
        engines={"m": backend},
        settings=ServerSettings(
            max_concurrency=2,
            admission_wait_timeout_s=_HANG_S,
        ),
    )
    async with _client(app) as client:
        first = asyncio.create_task(
            client.post("/v1/chat/completions", json=_chat_body("one"))
        )
        await _until(lambda: backend.active == 1)
        queued = asyncio.create_task(
            client.post("/v1/chat/completions", json=_chat_body("two"))
        )

        async def one_waiting() -> bool:
            return await _admission_waiting(client) == 1

        await _until(one_waiting)
        rejected = await _while_held(
            client.post("/v1/chat/completions", json=_chat_body("three"))
        )

        assert rejected.status_code == 429
        assert rejected.json()["error"]["code"] == "concurrency_exceeded"
        metrics = (await client.get("/metrics")).text
        assert "kairyu_admission_active_requests 1.0" in metrics
        assert "kairyu_admission_waiting_requests 1.0" in metrics
        assert 'kairyu_admission_rejections_total{reason="overflow"} 1.0' in metrics
        backend.release.set()
        assert (await first).status_code == 200
        assert (await queued).status_code == 200
        drained_metrics = (await client.get("/metrics")).text
        assert "kairyu_admission_active_requests 0.0" in drained_metrics
        assert "kairyu_admission_waiting_requests 0.0" in drained_metrics


async def test_admission_queue_wait_timeout_returns_429_and_releases_waiter():
    backend = _BudgetedMockBackend(sequence_budget=1)
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
        # The slot stays held until release, so the queued wait must time out.
        await _until(lambda: backend.active == 1)
        timed_out = await _while_held(
            client.post("/v1/chat/completions", json=_chat_body("two"))
        )

        assert timed_out.status_code == 429
        assert timed_out.headers["retry-after"] == "1"
        metrics = (await client.get("/metrics")).text
        assert 'kairyu_admission_rejections_total{reason="timeout"} 1.0' in metrics
        backend.release.set()
        assert (await first).status_code == 200
        assert (
            await client.post("/v1/chat/completions", json=_chat_body("three"))
        ).status_code == 200


async def test_cancelled_admission_waiter_releases_queue_capacity():
    backend = _BudgetedMockBackend(sequence_budget=1)
    app = create_legacy_app(
        engines={"m": backend},
        settings=ServerSettings(
            max_concurrency=2,
            admission_wait_timeout_s=_HANG_S,
        ),
    )
    async with _client(app) as client:
        first = asyncio.create_task(
            client.post("/v1/chat/completions", json=_chat_body("one"))
        )
        await _until(lambda: backend.active == 1)
        cancelled = asyncio.create_task(
            client.post("/v1/chat/completions", json=_chat_body("two"))
        )

        async def one_waiting() -> bool:
            return await _admission_waiting(client) == 1

        await _until(one_waiting)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        replacement = asyncio.create_task(
            client.post("/v1/chat/completions", json=_chat_body("three"))
        )

        async def replacement_queued_or_rejected() -> bool:
            return replacement.done() or await one_waiting()

        # The replacement queues in the capacity the cancelled waiter freed.
        await _until(replacement_queued_or_rejected)
        assert not replacement.done()
        backend.release.set()

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
    backend = _GatedBackend()
    app = create_legacy_app(
        engines={"m": backend},
        settings=ServerSettings(max_concurrency=1),
    )
    async with _client(app) as client:
        task = asyncio.create_task(
            client.post("/v1/chat/completions", json=_chat_body("one"))
        )
        await _until(lambda: backend.active == 1)  # the only slot is held
        assert (await _while_held(client.get("/health"))).status_code == 200
        backend.release.set()
        assert (await task).status_code == 200


def test_server_sequence_budget_requires_exactly_one_advertising_backend():
    backend = _BudgetedMockBackend(sequence_budget=2)

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
