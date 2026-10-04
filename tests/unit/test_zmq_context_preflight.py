"""Process-split parent preflight of the context window (kairyu-proc).

Moved from ``test_zmq_backend.py`` (M20 WP-04) so the shared output-budget
contract (``resolve_output_budget``) is owned in one place: the parent rejects
exactly what the child engine would, before any child process starts.
"""

from __future__ import annotations

import json

import httpx
import pytest

from kairyu import SamplingParams
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.prompt import PromptInput, TokensPrompt
from kairyu.engine.request_errors import ContextLengthExceededError
from kairyu.engine.zmq_backend import ZmqEngineBackend
from kairyu.entrypoints.server.app import create_app


def _request(request_id: str, prompt: PromptInput, **sampling) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt=prompt,
        sampling_params=SamplingParams(**sampling),
    )


async def test_parent_preflight_enforces_context_limit_without_starting_child():
    backend = ZmqEngineBackend(num_pages=64, max_model_len=5)
    oversized = _request(
        "context-retry",
        TokensPrompt((1, 2, 3, 4)),
        max_tokens=2,
    )

    try:
        with pytest.raises(ValueError, match="exceed max_model_len"):
            backend.validate_request(oversized)
        with pytest.raises(ValueError, match="exceed max_model_len"):
            await backend.prepare_request(oversized)

        assert backend._process is None
        assert backend._active_request_ids == set()
        assert backend._queues == {}
        assert backend._prepared_requests == {}

        boundary = await backend.generate(
            _request(
                "context-retry",
                TokensPrompt((1, 2, 3)),
                max_tokens=2,
            )
        )
        assert boundary.finished
        assert boundary.usage is not None
        assert boundary.usage.prompt_tokens + boundary.usage.completion_tokens == 5
    finally:
        await backend.shutdown()


@pytest.mark.parametrize("limit_source", ["configured", "model-config"])
async def test_parent_preflight_resolves_the_child_output_budget(tmp_path, limit_source):
    # With max_tokens omitted the child generates up to the remaining context
    # (#496). The parent preflight once assumed 16 output tokens, rejecting
    # prompts the child accepts and reporting overflow as an untyped error.
    # Without a configured limit the child derives it from the model's
    # max_position_embeddings; the parent must enforce that same limit.
    if limit_source == "configured":
        backend = ZmqEngineBackend(num_pages=64, max_model_len=8)
    else:
        (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 8}))
        backend = ZmqEngineBackend(num_pages=64, model_path=str(tmp_path))
    try:
        backend.validate_request(
            _request("fits", TokensPrompt(tuple(range(1, 8))), max_tokens=None)
        )
        with pytest.raises(ContextLengthExceededError) as raised:
            backend.validate_request(
                _request("full", TokensPrompt(tuple(range(1, 9))), max_tokens=None)
            )

        error = raised.value
        assert (error.prompt_tokens, error.max_tokens, error.max_model_len) == (
            8,
            None,
            8,
        )
        assert error.code == "context_length_exceeded"
        assert backend._process is None
    finally:
        await backend.shutdown()


@pytest.mark.parametrize(
    "wire_prompt",
    [
        "one two",
        [1, 2],
    ],
)
async def test_streaming_context_rejection_is_http_400_before_sse_headers(
    wire_prompt,
):
    backend = ZmqEngineBackend(num_pages=64, max_model_len=3)
    app = create_app(engines={"limited-proc": backend})
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/completions",
                json={
                    "model": "limited-proc",
                    "prompt": wire_prompt,
                    "max_tokens": 2,
                    "stream": True,
                },
            )

        assert response.status_code == 400
        assert "exceed max_model_len" in response.json()["error"]["message"]
        assert not response.headers["content-type"].startswith("text/event-stream")
        assert backend._process is None
        assert backend._prepared_requests == {}
    finally:
        await backend.shutdown()
