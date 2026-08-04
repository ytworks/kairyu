"""Request omission plumbing for model-owned generation defaults (#351)."""

from __future__ import annotations

import httpx
import pytest

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app
from kairyu.sampling_params import GENERATION_CONFIG_SAMPLING_FIELDS


class _RecordingBackend(MockBackend):
    def __init__(self) -> None:
        super().__init__()
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return await super().generate(request)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


@pytest.mark.parametrize(
    ("path", "base"),
    [
        (
            "/v1/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": "hello"}]},
        ),
        ("/v1/completions", {"model": "m", "prompt": "hello"}),
    ],
)
async def test_openai_endpoints_preserve_omitted_and_explicit_sampling(path, base):
    backend = _RecordingBackend()
    app = create_app({"m": backend})
    explicit = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }

    async with _client(app) as client:
        omitted_response = await client.post(path, json=base)
        explicit_response = await client.post(path, json={**base, **explicit})

    assert omitted_response.status_code == 200
    assert explicit_response.status_code == 200
    omitted, supplied = (request.sampling_params for request in backend.requests)
    assert omitted.generation_config_omitted == GENERATION_CONFIG_SAMPLING_FIELDS
    assert supplied.generation_config_omitted == frozenset()
    assert {
        field: getattr(supplied, field)
        for field in GENERATION_CONFIG_SAMPLING_FIELDS
    } == explicit


async def test_responses_endpoint_preserves_temperature_and_top_p_omission():
    backend = _RecordingBackend()
    app = create_app({"m": backend})

    async with _client(app) as client:
        omitted_response = await client.post(
            "/v1/responses",
            json={"model": "m", "input": "hello"},
        )
        explicit_response = await client.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "hello",
                "temperature": 1.0,
                "top_p": 1.0,
            },
        )

    assert omitted_response.status_code == 200
    assert explicit_response.status_code == 200
    omitted, supplied = (request.sampling_params for request in backend.requests)
    assert omitted.generation_config_omitted == GENERATION_CONFIG_SAMPLING_FIELDS
    assert supplied.generation_config_omitted == (
        GENERATION_CONFIG_SAMPLING_FIELDS - {"temperature", "top_p"}
    )


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            "/v1/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": "hello"}]},
        ),
        ("/v1/completions", {"model": "m", "prompt": "hello"}),
    ],
)
async def test_openai_endpoints_still_reject_explicit_null_sampling(path, body):
    app = create_app({"m": MockBackend()})

    async with _client(app) as client:
        response = await client.post(path, json={**body, "temperature": None})

    assert response.status_code == 422
