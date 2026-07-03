"""Global concurrency guard (design m7 D5)."""

import asyncio

import httpx

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.settings import ServerSettings


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _chat_body(content: str) -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": content}]}


async def test_saturation_returns_429_with_retry_after():
    app = create_app(
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


async def test_slot_is_released_after_completion():
    app = create_app(
        engines={"m": MockBackend()},
        settings=ServerSettings(max_concurrency=1),
    )
    async with _client(app) as client:
        for _ in range(3):  # sequential requests all fit in the single slot
            response = await client.post("/v1/chat/completions", json=_chat_body("hi"))
            assert response.status_code == 200


async def test_health_is_never_guarded():
    app = create_app(
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
