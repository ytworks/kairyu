import json

import pytest
from fastapi.testclient import TestClient

from kairyu.engine.backend import GenerationResult, GenerationUsage
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.outputs import CompletionOutput, TokenLogprob
from tests.server._legacy_chat import create_legacy_app


class ParityBackend:
    def __init__(self, *, satisfy_tools: bool = True, supports_n: bool = True) -> None:
        self.satisfy_tools = satisfy_tools
        self.supports_n = supports_n
        self.requests = []

    def _result(self, request):
        completions = []
        for index in range(request.sampling_params.n):
            if request.tools and self.satisfy_tools:
                text = f'<tool_call>{{"name":"lookup","arguments":{{"index":{index}}}}}</tool_call>'
                finish_reason = "tool_calls"
            elif request.sampling_params.extra_args.get("response_format"):
                text = json.dumps({"answer": index}, separators=(",", ":"))
                finish_reason = "stop"
            else:
                text = f"answer-{index}"
                finish_reason = "stop"
            logprob_content = (
                (
                    TokenLogprob(
                        token=text,
                        token_id=index,
                        logprob=-0.25,
                        bytes_=tuple(text.encode()),
                        top=(
                            TokenLogprob(
                                token=text,
                                token_id=index,
                                logprob=-0.25,
                                bytes_=tuple(text.encode()),
                            ),
                        ),
                    ),
                )
                if request.sampling_params.logprobs is not None
                else None
            )
            completions.append(
                CompletionOutput(
                    index=index,
                    text=text,
                    token_ids=(index,),
                    finish_reason=finish_reason,
                    logprob_content=logprob_content,
                )
            )
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=tuple(completions),
            usage=GenerationUsage(
                prompt_tokens=5,
                completion_tokens=len(completions),
                cached_tokens=1,
            ),
        )

    async def generate(self, request):
        self.requests.append(request)
        return self._result(request)

    async def stream(self, request):
        yield await self.generate(request)

    async def shutdown(self):
        return None


def _app(backend):
    return create_legacy_app(
        {"direct": backend},
        orchestrators={
            "auto": Orchestrator({"tier1": backend, "tier2": backend}),
        },
    )


def _sampling_body(model, *, stream=False):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.25,
        "top_p": 0.8,
        "n": 2,
        "max_tokens": 17,
        "presence_penalty": 0.2,
        "frequency_penalty": -0.3,
        "stop": ["END"],
        "seed": 41,
        "logprobs": True,
        "top_logprobs": 1,
        "response_format": {"type": "json_object"},
        "stream": stream,
        **({"stream_options": {"include_usage": True}} if stream else {}),
    }


def _sse_payloads(response):
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]


def test_direct_and_auto_sampling_n_logprobs_and_schema_are_wire_equivalent():
    backend = ParityBackend()
    with TestClient(_app(backend)) as client:
        direct = client.post(
            "/v1/chat/completions",
            json=_sampling_body("direct"),
        )
        auto = client.post(
            "/v1/chat/completions",
            json=_sampling_body("auto"),
        )

    assert direct.status_code == auto.status_code == 200
    direct_body = direct.json()
    auto_body = auto.json()
    assert auto_body["choices"] == direct_body["choices"]
    assert auto_body["usage"]["prompt_tokens"] == direct_body["usage"]["prompt_tokens"]
    assert auto_body["usage"]["completion_tokens"] == direct_body["usage"]["completion_tokens"]
    assert auto_body["usage"]["orchestration_input_tokens"] == 5
    assert auto_body["usage"]["orchestration_output_tokens"] == 2
    assert [json.loads(choice["message"]["content"]) for choice in auto_body["choices"]] == [
        {"answer": 0},
        {"answer": 1},
    ]


def test_auto_stream_preserves_choice_indices_and_logprobs_once():
    backend = ParityBackend()
    with TestClient(_app(backend)) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_sampling_body("auto", stream=True),
        )

    assert response.status_code == 200
    payloads = _sse_payloads(response)
    choices = [choice for payload in payloads for choice in payload["choices"]]
    assert {choice["index"] for choice in choices} == {0, 1}
    assert sorted(choice["index"] for choice in choices if choice.get("finish_reason")) == [0, 1]
    logprob_entries = {
        index: [
            entry
            for choice in choices
            if choice["index"] == index and choice.get("logprobs")
            for entry in choice["logprobs"]["content"]
        ]
        for index in (0, 1)
    }
    assert [len(logprob_entries[index]) for index in (0, 1)] == [1, 1]
    usage = next(payload["usage"] for payload in payloads if payload.get("usage"))
    assert usage["orchestration_input_tokens"] == 5
    assert usage["orchestration_output_tokens"] == 2


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("stream", [False, True])
def test_auto_output_limit_alias_reaches_final_backend(field, stream):
    backend = ParityBackend()
    body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
        field: 7,
    }

    with TestClient(_app(backend)) as client:
        response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert len(backend.requests) == 1
    assert backend.requests[0].sampling_params.max_tokens == 7


def _tool_body(*, stream=False, n=2):
    return {
        "model": "auto",
        "messages": [{"role": "user", "content": "look it up"}],
        "n": n,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "required",
        "stream": stream,
        **({"stream_options": {"include_usage": True}} if stream else {}),
    }


def test_auto_tools_are_preserved_and_required_for_every_choice():
    backend = ParityBackend()
    with TestClient(_app(backend)) as client:
        response = client.post("/v1/chat/completions", json=_tool_body())

    assert response.status_code == 200
    assert len(backend.requests) == 1
    assert backend.requests[0].tools
    assert backend.requests[0].tool_choice == "required"
    for index, choice in enumerate(response.json()["choices"]):
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] is None
        call = choice["message"]["tool_calls"][0]
        assert call["function"]["name"] == "lookup"
        assert json.loads(call["function"]["arguments"]) == {"index": index}


def test_auto_tool_stream_buffers_validation_then_emits_structured_calls():
    backend = ParityBackend()
    with TestClient(_app(backend)) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_tool_body(stream=True),
        )

    assert response.status_code == 200
    payloads = _sse_payloads(response)
    choices = [choice for payload in payloads for choice in payload["choices"]]
    starts = [choice for choice in choices if choice["delta"].get("tool_calls")]
    assert [choice["index"] for choice in starts] == [0, 1]
    assert all(choice["delta"]["content"] is None for choice in starts)
    assert sorted(choice["index"] for choice in choices if choice.get("finish_reason")) == [0, 1]


def test_required_tool_failure_is_502_before_stream_starts():
    backend = ParityBackend(satisfy_tools=False)
    with TestClient(_app(backend)) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_tool_body(stream=True, n=1),
        )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "tool_choice_not_satisfied"


def test_auto_n_capability_mismatch_is_a_predispatch_400():
    backend = ParityBackend(supports_n=False)
    with TestClient(_app(backend)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
                "n": 2,
            },
        )

    assert response.status_code == 400
    assert backend.requests == []
    assert "not supported" in response.json()["error"]["message"]
