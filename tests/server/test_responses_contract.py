"""Responses wire contract: typed errors, unrouted paths, stored-response endpoints."""

from __future__ import annotations

import asyncio

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import TenantConfig, TenantLimits
from kairyu.orchestration.orchestrator import Orchestrator
from tests.server._legacy_chat import create_legacy_app

_ENVELOPE_KEYS = {"message", "type", "param", "code"}


def _app(tmp_path, backend=None, **kwargs):
    return create_legacy_app(
        {"m": backend or MockBackend({"hello": "stored hello"})},
        settings=kwargs.pop(
            "settings", ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl"))
        ),
        **kwargs,
    )


def _sdk(http: TestClient, api_key: str = "sk-local") -> openai.OpenAI:
    return openai.OpenAI(
        base_url=str(http.base_url) + "/v1", api_key=api_key, http_client=http
    )


def test_malformed_bodies_are_typed_400s_naming_the_parameter(tmp_path):
    backend = MockBackend()
    with TestClient(_app(tmp_path, backend)) as http:
        with pytest.raises(openai.BadRequestError) as wrong_value:
            _sdk(http).responses.create(model="m", input="hi", max_output_tokens="many")
        missing = http.post("/v1/responses", json={"input": "hi"})
        not_json = http.post(
            "/v1/responses",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
    assert wrong_value.value.param == "max_output_tokens"
    assert missing.status_code == 400
    assert missing.json()["error"]["code"] == "missing_required_parameter"
    assert missing.json()["error"]["param"] == "model"
    assert not_json.status_code == 400
    assert set(not_json.json()["error"]) == _ENVELOPE_KEYS
    assert backend.prompts_seen == ()


def test_unrouted_paths_and_methods_answer_in_the_openai_envelope(tmp_path):
    with TestClient(_app(tmp_path)) as http:
        unknown_path = http.get("/v1/responses/resp_x/unknown")
        retrieve_put = http.put("/v1/responses/resp_x")
        create_patch = http.patch("/v1/responses")

    assert unknown_path.status_code == 404
    assert set(unknown_path.json()["error"]) == _ENVELOPE_KEYS
    assert retrieve_put.status_code == 405
    assert retrieve_put.headers["allow"] == "DELETE, GET"
    assert retrieve_put.json()["error"]["code"] == "method_not_allowed"
    assert create_patch.status_code == 405
    assert create_patch.headers["allow"] == "GET, POST"


def test_stored_response_lifecycle_through_the_sdk(tmp_path):
    with TestClient(_app(tmp_path)) as http:
        sdk = _sdk(http)
        created = sdk.responses.create(
            model="m",
            input=[
                {"role": "user", "content": "hello"},
                {"role": "user", "content": [{"type": "input_text", "text": "again"}]},
            ],
        )
        retrieved = sdk.responses.retrieve(created.id, include=["reasoning.encrypted_content"])
        first_page = sdk.responses.input_items.list(created.id, limit=1, order="asc")
        every_item = list(sdk.responses.input_items.list(created.id, limit=1, order="asc"))
        with pytest.raises(openai.BadRequestError):
            sdk.responses.cancel(created.id)
        sdk.responses.delete(created.id)
        with pytest.raises(openai.NotFoundError):
            sdk.responses.retrieve(created.id)
        unstored = sdk.responses.create(model="m", input="hello", store=False)
        with pytest.raises(openai.NotFoundError):
            sdk.responses.retrieve(unstored.id)

    assert retrieved.model_dump() == created.model_dump()
    assert first_page.has_more
    assert [item.content[0].text for item in every_item] == ["hello", "again"]
    assert len({item.id for item in every_item}) == 2


def test_stored_responses_are_invisible_to_other_tenants(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIRYU_RESPONSES_KEYS", "key-a,key-b")
    app = _app(
        tmp_path,
        settings=ServerSettings(
            api_keys_env="KAIRYU_RESPONSES_KEYS",
            usage_ledger_path=str(tmp_path / "usage.jsonl"),
        ),
        tenant_config=TenantConfig(key_tenants={"key-a": "tenant-a", "key-b": "tenant-b"}),
    )
    with TestClient(app) as http:
        created = _sdk(http, "key-a").responses.create(model="m", input="hello")
        other = _sdk(http, "key-b")
        with pytest.raises(openai.NotFoundError):
            other.responses.retrieve(created.id)
        with pytest.raises(openai.NotFoundError):
            other.responses.input_items.list(created.id)
        with pytest.raises(openai.NotFoundError):
            other.responses.delete(created.id)
        assert _sdk(http, "key-a").responses.retrieve(created.id).id == created.id


def test_input_tokens_counts_the_rendered_prompt(tmp_path):
    class CountingBackend(MockBackend):
        def __init__(self):
            super().__init__()
            self.counted: list[str] = []

        async def count_prompt_tokens_async(self, prompt: str) -> int:
            self.counted.append(prompt)
            return len(prompt)

    backend = CountingBackend()
    app = _app(
        tmp_path,
        backend,
        orchestrators={"kairyu-auto": Orchestrator({"tier1": backend, "tier2": backend})},
    )
    with TestClient(app) as http:
        counted = _sdk(http).responses.input_tokens.count(
            model="m", input="count these words", instructions="be brief"
        )
        auto = http.post(
            "/v1/responses/input_tokens", json={"model": "kairyu-auto", "input": "x"}
        )

    assert counted.input_tokens == len(backend.counted[-1])
    assert "count these words" in backend.counted[-1]
    assert "be brief" in backend.counted[-1]
    assert auto.status_code == 400
    assert auto.json()["error"]["param"] == "model"


async def _overloaded_second_request(app, path: str, body: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.create_task(client.post(path, json=body))
        await asyncio.sleep(0.05)  # the first request holds the only slot
        second = await client.post(path, json=body)
        assert (await first).status_code == 200
    return second


@pytest.mark.parametrize(
    "limits",
    [
        {"settings": ServerSettings(max_concurrency=1)},
        {
            "tenant_config": TenantConfig(
                limits={"default": TenantLimits(request_burst=10, max_in_flight=1)}
            )
        },
    ],
    ids=["server-concurrency", "tenant-in-flight"],
)
async def test_transient_overload_is_a_retryable_slow_down_only_on_responses(limits):
    def app():
        return create_legacy_app({"m": MockBackend(latency_s=0.2)}, **limits)

    responses = await _overloaded_second_request(
        app(), "/v1/responses", {"model": "m", "input": "hello"}
    )
    chat = await _overloaded_second_request(
        app(),
        "/v1/chat/completions",
        {"model": "m", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert responses.status_code == 503
    assert responses.headers["retry-after"] == "1"
    assert responses.json()["error"]["code"] == "slow_down"
    assert set(responses.json()["error"]) == _ENVELOPE_KEYS
    assert chat.status_code == 429
