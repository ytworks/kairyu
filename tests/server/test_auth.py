"""API-key auth middleware (goal G3 gate C5)."""

import httpx
import pytest

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.orchestration.orchestrator import Orchestrator
from tests.server._legacy_chat import create_legacy_app


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _chat_body(content: str) -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": content}]}


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setenv("KAIRYU_API_KEYS", "secret-1, secret-2")
    return create_legacy_app(
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


def test_non_ascii_bearer_token_is_rejected_not_crash():
    # M5: a non-ASCII token (raw high bytes a client could send) would crash
    # hmac.compare_digest with TypeError; _authorized must return False, not raise.
    from kairyu.entrypoints.server.middleware import AuthMiddleware

    auth = AuthMiddleware(app=None, api_keys=("secret-1",))
    scope = {
        "type": "http",
        "headers": [(b"authorization", b"Bearer s\xe9cr\xe9t")],  # latin-1 high bytes
    }
    assert auth._authorized(scope) is False  # no TypeError


def test_role_auth_compares_every_configured_key(monkeypatch):
    from kairyu.entrypoints.server.middleware import AuthMiddleware

    compared = []

    def compare(left: str, right: str) -> bool:
        compared.append(right)
        return left == right

    monkeypatch.setattr(
        "kairyu.entrypoints.server.middleware.hmac.compare_digest", compare
    )
    auth = AuthMiddleware(
        app=None,
        api_keys=("data", "other-data"),
        admin_keys=("admin", "other-admin"),
    )
    scope = {
        "type": "http",
        "headers": [(b"authorization", b"Bearer data")],
    }

    assert auth._authorized(scope) is True
    assert compared == ["data", "other-data", "admin", "other-admin"]


async def test_valid_keys_are_admitted(app):
    async with _client(app) as client:
        for key in ("secret-1", "secret-2"):
            response = await client.post(
                "/v1/chat/completions",
                json=_chat_body("hi"),
                headers={"Authorization": f"Bearer {key}"},
            )
            assert response.status_code == 200


async def test_route_and_routing_config_require_auth(monkeypatch):
    monkeypatch.setenv("KAIRYU_API_KEYS", "secret")
    app = create_legacy_app(
        engines={"m": MockBackend()},
        orchestrators={"auto": Orchestrator({"tier1": MockBackend()})},
        settings=ServerSettings(api_keys_env="KAIRYU_API_KEYS"),
    )
    route_body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hello"}],
    }
    async with _client(app) as client:
        assert (await client.post("/v1/route", json=route_body)).status_code == 401
        assert (await client.get("/routing")).status_code == 401
        headers = {"Authorization": "Bearer secret"}
        assert (
            await client.post("/v1/route", json=route_body, headers=headers)
        ).status_code == 200
        assert (await client.get("/routing", headers=headers)).status_code == 200


async def test_health_readyz_metrics_stay_open(app):
    async with _client(app) as client:
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/readyz")).status_code == 200
        assert (await client.get("/metrics")).status_code == 200


async def test_protect_metrics_requires_key(monkeypatch):
    monkeypatch.setenv("KAIRYU_API_KEYS", "k")
    app = create_legacy_app(
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
        create_legacy_app(
            engines={"m": MockBackend()},
            settings=ServerSettings(api_keys_env="KAIRYU_API_KEYS"),
        )


async def test_admin_and_data_plane_role_matrix(monkeypatch):
    monkeypatch.setenv("KAIRYU_DATA_KEYS", "data,dual")
    monkeypatch.setenv("KAIRYU_ADMIN_KEYS", "admin,dual")
    app = create_legacy_app(
        {"m": MockBackend()},
        settings=ServerSettings(
            api_keys_env="KAIRYU_DATA_KEYS",
            admin_keys_env="KAIRYU_ADMIN_KEYS",
        ),
    )
    payload = _chat_body("hi")
    async with _client(app) as client:
        admin = {"Authorization": "Bearer admin"}
        data = {"Authorization": "Bearer data"}
        dual = {"Authorization": "Bearer dual"}
        invalid = {"Authorization": "Bearer invalid"}

        assert (await client.post("/admin/drain", headers=admin)).status_code == 200
        assert (await client.post("/admin/undrain", headers=admin)).status_code == 200
        denied = await client.post("/v1/chat/completions", json=payload, headers=admin)
        assert denied.status_code == 403
        assert denied.json()["error"]["code"] == "data_plane_required"
        assert (
            await client.post("/v1/chat/completions", json=payload, headers=data)
        ).status_code == 200
        assert (await client.post("/admin/drain", headers=data)).status_code == 403
        assert (
            await client.post("/v1/chat/completions", json=payload, headers=dual)
        ).status_code == 200
        assert (await client.post("/admin/drain", headers=dual)).status_code == 200
        assert (await client.post("/admin/undrain", headers=dual)).status_code == 200
        assert (await client.post("/admin/drain", headers=invalid)).status_code == 401


async def test_admin_only_configuration_installs_auth(monkeypatch):
    monkeypatch.setenv("KAIRYU_ADMIN_KEYS", "admin")
    app = create_legacy_app(
        {"m": MockBackend()},
        settings=ServerSettings(admin_keys_env="KAIRYU_ADMIN_KEYS"),
    )
    async with _client(app) as client:
        assert (await client.post("/admin/drain")).status_code == 401
        assert (
            await client.post(
                "/admin/drain", headers={"Authorization": "Bearer admin"}
            )
        ).status_code == 200


def _messages_body() -> dict:
    return {
        "model": "m",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    }


async def test_x_api_key_is_accepted_on_both_dialects(app):
    # Claude Code sends the gateway credential in x-api-key (issue #508); it
    # feeds the same key set and tenant policy as the Bearer header.
    async with _client(app) as client:
        chat = await client.post(
            "/v1/chat/completions",
            json=_chat_body("hi"),
            headers={"x-api-key": "secret-1"},
        )
        messages = await client.post(
            "/v1/messages", json=_messages_body(), headers={"x-api-key": "secret-1"}
        )
    assert chat.status_code == 200
    assert messages.status_code == 200


async def test_wrong_x_api_key_is_401_and_bearer_takes_precedence(app):
    async with _client(app) as client:
        wrong = await client.post(
            "/v1/chat/completions",
            json=_chat_body("hi"),
            headers={"x-api-key": "nope"},
        )
        both_good_bearer = await client.post(
            "/v1/chat/completions",
            json=_chat_body("hi"),
            headers={"Authorization": "Bearer secret-1", "x-api-key": "nope"},
        )
        bad_bearer_wins = await client.post(
            "/v1/chat/completions",
            json=_chat_body("hi"),
            headers={"Authorization": "Bearer nope", "x-api-key": "secret-1"},
        )
    assert wrong.status_code == 401
    assert both_good_bearer.status_code == 200
    assert bad_bearer_wins.status_code == 401


async def test_messages_401_uses_anthropic_envelope(app):
    # /v1/messages must never receive the OpenAI error envelope; the OpenAI
    # routes keep theirs unchanged.
    async with _client(app) as client:
        messages = await client.post("/v1/messages", json=_messages_body())
        chat = await client.post("/v1/chat/completions", json=_chat_body("hi"))
    assert messages.status_code == 401
    payload = messages.json()
    assert payload["type"] == "error"
    assert payload["error"] == {
        "type": "authentication_error",
        "message": "missing or invalid API key",
    }
    assert messages.headers["www-authenticate"] == "Bearer"
    assert chat.status_code == 401
    assert chat.json()["error"]["code"] == "invalid_api_key"
