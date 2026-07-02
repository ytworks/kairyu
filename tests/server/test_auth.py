"""API-key auth middleware (goal G3 gate C5)."""

import httpx
import pytest

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.settings import ServerSettings


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _chat_body(content: str) -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": content}]}


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setenv("KAIRYU_API_KEYS", "secret-1, secret-2")
    return create_app(
        engines={"m": MockBackend()},
        settings=ServerSettings(api_keys_env="KAIRYU_API_KEYS"),
    )


async def test_missing_key_is_401(app):
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat_body("hi"))
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"
    assert response.headers["www-authenticate"] == "Bearer"


async def test_wrong_key_is_401(app):
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_body("hi"),
            headers={"Authorization": "Bearer nope"},
        )
    assert response.status_code == 401


async def test_valid_keys_are_admitted(app):
    async with _client(app) as client:
        for key in ("secret-1", "secret-2"):
            response = await client.post(
                "/v1/chat/completions",
                json=_chat_body("hi"),
                headers={"Authorization": f"Bearer {key}"},
            )
            assert response.status_code == 200


async def test_health_readyz_metrics_stay_open(app):
    async with _client(app) as client:
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/readyz")).status_code == 200
        assert (await client.get("/metrics")).status_code == 200


async def test_protect_metrics_requires_key(monkeypatch):
    monkeypatch.setenv("KAIRYU_API_KEYS", "k")
    app = create_app(
        engines={"m": MockBackend()},
        settings=ServerSettings(api_keys_env="KAIRYU_API_KEYS", protect_metrics=True),
    )
    async with _client(app) as client:
        assert (await client.get("/metrics")).status_code == 401
        response = await client.get("/metrics", headers={"Authorization": "Bearer k"})
        assert response.status_code == 200


async def test_empty_key_env_fails_loud(monkeypatch):
    monkeypatch.setenv("KAIRYU_API_KEYS", "  ")
    with pytest.raises(ValueError, match="contains no keys"):
        create_app(
            engines={"m": MockBackend()},
            settings=ServerSettings(api_keys_env="KAIRYU_API_KEYS"),
        )
