import httpx

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.chat_template import ChatTemplate
from kairyu.entrypoints.server.app import create_app
from kairyu.orchestration.orchestrator import Orchestrator


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


def _body(query: str, model: str = "auto") -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": query}],
    }


async def test_preview_uses_rendered_chat_prompt_without_dispatch():
    tier1 = MockBackend()
    tier2 = MockBackend()
    app = create_app(
        engines={"direct": MockBackend()},
        orchestrators={"auto": Orchestrator({"tier1": tier1, "tier2": tier2})},
    )
    # Raw input is below the 600-char threshold. The default chat renderer adds
    # "user: " and "\nassistant:", taking the routed prompt over the boundary.
    query = "x" * 590
    async with _client(app) as client:
        preview = await client.post("/v1/route", json=_body(query))
        actual = await client.post(
            "/v1/chat/completions",
            headers={"X-Kairyu-Trace": "1"},
            json=_body(query),
        )

    assert preview.status_code == 200
    assert preview.json()["target"] == "tier2"
    assert preview.json()["features"]["char_len"] == len(query) + 17
    assert preview.json()["binding"] is False
    assert tier1.prompts_seen == ()
    assert len(tier2.prompts_seen) == 1  # actual request only
    assert actual.json()["kairyu_route"]["target"] == preview.json()["target"]


async def test_preview_uses_the_configured_model_chat_template():
    template = ChatTemplate("PREFIX:{{ messages[0].content }}:ASSISTANT")
    tier1 = MockBackend()
    app = create_app(
        engines={},
        orchestrators={"auto": Orchestrator({"tier1": tier1})},
        chat_templates={"auto": template},
    )
    async with _client(app) as client:
        preview = await client.post("/v1/route", json=_body("hello"))
        actual = await client.post(
            "/v1/chat/completions",
            headers={"X-Kairyu-Trace": "1"},
            json=_body("hello"),
        )

    expected_prompt = "PREFIX:hello:ASSISTANT"
    assert preview.status_code == 200
    assert preview.json()["features"]["char_len"] == len(expected_prompt)
    assert tier1.prompts_seen == (expected_prompt,)
    assert actual.json()["kairyu_route"]["features"] == preview.json()["features"]


async def test_preview_routes_on_rendered_prompt_at_multi_agent_boundary():
    app = create_app(
        engines={},
        orchestrators={"auto": Orchestrator({"tier1": MockBackend()})},
    )
    # The raw query is below 2000 characters; the default renderer adds 17.
    query = "x" * 1983
    async with _client(app) as client:
        response = await client.post("/v1/route", json=_body(query))

    assert response.status_code == 200
    assert response.json()["features"]["char_len"] == 2000
    assert response.json()["target"] == "multi_agent"


async def test_preview_direct_and_unknown_models():
    app = create_app(
        engines={"direct": MockBackend()},
        orchestrators={"auto": Orchestrator({"tier1": MockBackend()})},
    )
    async with _client(app) as client:
        direct = await client.post("/v1/route", json=_body("hello", "direct"))
        missing = await client.post("/v1/route", json=_body("hello", "missing"))

    assert direct.status_code == 200
    assert direct.json() == {
        "model": "direct",
        "orchestrated": False,
        "binding": False,
        "router_type": None,
        "target": None,
        "confidence": None,
        "reason": None,
        "features": None,
    }
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "model_not_found"


async def test_preview_rejects_router_without_non_mutating_contract():
    class RouteOnly:
        def route(self, query, context=None):
            raise AssertionError("route must not be called by preview")

    app = create_app(
        engines={},
        orchestrators={"auto": Orchestrator({"tier1": MockBackend()}, router=RouteOnly())},
    )
    async with _client(app) as client:
        response = await client.post("/v1/route", json=_body("hello"))

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "preview_not_supported"


async def test_route_field_is_absent_without_trace_opt_in():
    app = create_app(
        engines={},
        orchestrators={"auto": Orchestrator({"tier1": MockBackend()})},
    )
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_body("hello"))

    assert response.status_code == 200
    assert "kairyu_route" not in response.json()
