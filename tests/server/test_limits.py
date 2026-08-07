"""Global concurrency guard (design m7 D5)."""

import asyncio

import httpx
import pytest

from kairyu.engine.mock import MockBackend
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

    async def generate(self, request):
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
        assert (await first).status_code == 200
        assert (await queued).status_code == 200


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
