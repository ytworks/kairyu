import asyncio
import json
import threading

import httpx
import pytest

from kairyu import SamplingParams
from kairyu.engine import vision as vision_module
from kairyu.engine.backend import (
    GenerationRequest,
    GenerationStageMetric,
    UpstreamClientError,
)
from kairyu.engine.openai_backend import OpenAICompatBackend, _stage_metrics_from_trace
from kairyu.engine.openai_capabilities import (
    OpenAIRequestCapabilities,
    known_openai_upstreams,
)
from kairyu.engine.prompt import (
    MultimodalItem,
    MultimodalMessage,
    MultimodalMessagePart,
    MultimodalPrompt,
    PromptInput,
    TemplatedPrompt,
    TokensPrompt,
)
from kairyu.engine.vision import ImageInputPolicy
from kairyu.outputs import TokenLogprob
from kairyu.sampling_params import GENERATION_CONFIG_SAMPLING_FIELDS

_RED_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP8z8DAwMDA"
    "xMDAwMDAAAANHQEDasKb6QAAAABJRU5ErkJggg=="
)


_GENERATION_NEUTRALS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
}


def _request(
    prompt: PromptInput = "hi",
    sampling_params: SamplingParams | None = None,
    *,
    explicit_generation_neutrals: bool = False,
    trace_requested: bool = False,
) -> GenerationRequest:
    params = sampling_params or SamplingParams(temperature=0.2, max_tokens=64)
    if not explicit_generation_neutrals:
        params = params.with_generation_config_omitted(
            {
                field
                for field, neutral in _GENERATION_NEUTRALS.items()
                if getattr(params, field) == neutral
            }
        )
    return GenerationRequest(
        request_id="r1",
        prompt=prompt,
        sampling_params=params,
        trace_requested=trace_requested,
    )


def _stage_trace(*metrics: tuple[str, int, int]) -> dict[str, object]:
    return {
        "trace_version": "2.0",
        "request_id": "chatcmpl-upstream",
        "started_at": "2026-08-05T00:00:00.000Z",
        "completed_at": "2026-08-05T00:00:00.001Z",
        "events": [
            {
                "seq": seq,
                "node": stage,
                "role": "engine",
                "kind": "stage",
                "status": "success",
                "attempt": 0,
                "timing": None,
                "detail": {
                    "stage": stage,
                    "duration_ns": duration_ns,
                    "occurrences": occurrences,
                    "aggregation": "sum",
                    "scope": "request-observed",
                },
            }
            for seq, (stage, duration_ns, occurrences) in enumerate(
                metrics, start=1
            )
        ],
    }


def _multimodal_prompt(image_url: str = _RED_PNG_DATA_URL) -> MultimodalPrompt:
    return MultimodalPrompt(
        base="display only",
        items=(MultimodalItem("image", "uri", image_url),),
        messages=(
            MultimodalMessage(
                "system",
                (MultimodalMessagePart("text", text="Answer exactly."),),
            ),
            MultimodalMessage(
                "user",
                (
                    MultimodalMessagePart("text", text="before "),
                    MultimodalMessagePart(
                        "item",
                        item_index=0,
                        detail="high",
                    ),
                    MultimodalMessagePart("text", text=" after"),
                ),
            ),
        ),
    )


def test_prerendered_chat_prompt_is_rejected_before_upstream_dispatch():
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
    )

    with pytest.raises(ValueError, match="pre-rendered chat prompt.*template boundary"):
        backend.validate_request(_request(TemplatedPrompt("<BOS>hello")))


def _ok_transport(captured: dict) -> httpx.MockTransport:
    def handler(http_request: httpx.Request) -> httpx.Response:
        captured["url"] = str(http_request.url)
        captured["auth"] = http_request.headers.get("authorization")
        captured["body"] = json.loads(http_request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello from api"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    return httpx.MockTransport(handler)


async def test_generate_maps_openai_response(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-test")
    captured: dict = {}
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="gpt-x",
        api_key_env="TEST_API_KEY",
        transport=_ok_transport(captured),
    )
    result = await backend.generate(_request("say hello"))
    assert result.completions[0].text == "hello from api"
    assert result.completions[0].finish_reason == "stop"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["model"] == "gpt-x"
    assert captured["body"]["temperature"] == 0.2
    assert captured["body"]["max_tokens"] == 64
    assert captured["body"]["messages"][-1]["content"] == "say hello"


async def test_generate_propagates_and_parses_kairyu_stage_trace():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["trace_header"] = request.headers.get("x-kairyu-trace")
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-upstream",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "kairyu_trace_v2": _stage_trace(
                    ("tokenize", 101, 1),
                    ("decode_step", 303, 3),
                ),
            },
        )

    backend = OpenAICompatBackend(
        base_url="http://replica:8000/v1",
        model="m",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
        upstream="kairyu",
    )

    result = await backend.generate(_request(trace_requested=True))

    assert captured["trace_header"] == "1"
    assert [
        (metric.stage, metric.duration_ns, metric.occurrences)
        for metric in result.stage_metrics
    ] == [("tokenize", 101, 1), ("decode_step", 303, 3)]


async def test_generate_allows_missing_opted_in_trace_during_rolling_upgrade():
    captured: dict[str, object] = {}
    backend = OpenAICompatBackend(
        base_url="http://replica:8000/v1",
        model="m",
        api_key_env=None,
        transport=_ok_transport(captured),
        upstream="kairyu",
    )

    result = await backend.generate(_request(trace_requested=True))

    assert result.stage_metrics == ()


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda trace: trace["events"][0]["detail"].__setitem__(
                "duration_ns", True
            ),
            "invalid stage duration",
        ),
        (
            lambda trace: trace.__setitem__("request_id", "chatcmpl-stale"),
            "request id does not match",
        ),
        (
            lambda trace: trace["events"][0].__setitem__("node", "wrong"),
            "invalid stage trace name",
        ),
        (
            lambda trace: trace.__setitem__(
                "started_at", "2026-08-05T00:00:00Z"
            ),
            "invalid started_at",
        ),
        (
            lambda trace: trace["events"][0].__setitem__("timing", {}),
            "invalid stage trace name",
        ),
        (
            lambda trace: trace["events"][0].__setitem__("seq", True),
            "invalid stage trace sequence",
        ),
    ),
)
async def test_generate_rejects_malformed_kairyu_stage_trace(mutate, match):
    trace = _stage_trace(("tokenize", 101, 1))
    mutate(trace)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-upstream",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "kairyu_trace_v2": trace,
            },
        )

    backend = OpenAICompatBackend(
        base_url="http://replica:8000/v1",
        model="m",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
        upstream="kairyu",
    )

    with pytest.raises(RuntimeError, match=match):
        await backend.generate(_request(trace_requested=True))


def test_kairyu_stage_trace_accepts_future_stage_and_ignores_unknown_status():
    trace = _stage_trace(
        ("future_stage", 101, 1),
        ("tokenize", 202, 1),
    )
    trace["events"][1]["status"] = "future-status"

    metrics = _stage_metrics_from_trace(
        trace,
        response_id="chatcmpl-upstream",
    )

    assert metrics == (GenerationStageMetric("future_stage", 101),)


async def test_generate_maps_upstream_logprobs():
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": "Hello"},
        "finish_reason": "length",
        "logprobs": {
            "content": [
                {
                    "token": "Hel",
                    "logprob": -0.25,
                    "bytes": [72, 101, 108],
                    "top_logprobs": [
                        {
                            "token": "Hel",
                            "logprob": -0.25,
                            "bytes": [72, 101, 108],
                        },
                        {
                            "token": "Help",
                            "logprob": -1.5,
                            "bytes": [72, 101, 108, 112],
                            "top_logprobs": [
                                {
                                    "token": "help",
                                    "logprob": -2.0,
                                    "bytes": None,
                                }
                            ],
                        },
                    ],
                },
                {
                    "token": "lo",
                    "logprob": -0.75,
                    "bytes": None,
                    "top_logprobs": [
                        {"token": "lo", "logprob": -0.75, "bytes": None},
                        {
                            "token": " low",
                            "logprob": -1.25,
                            "bytes": [32, 108, 111, 119],
                        },
                    ],
                },
            ]
        },
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [choice]}))
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=transport,
    )

    result = await backend.generate(_request())

    output = result.completions[0]
    assert output.text == "Hello"
    assert output.finish_reason == "length"
    assert output.logprob_content == (
        TokenLogprob(
            token="Hel",
            token_id=-1,
            logprob=-0.25,
            bytes_=(72, 101, 108),
            top=(
                TokenLogprob(
                    token="Hel",
                    token_id=-1,
                    logprob=-0.25,
                    bytes_=(72, 101, 108),
                ),
                TokenLogprob(
                    token="Help",
                    token_id=-1,
                    logprob=-1.5,
                    bytes_=(72, 101, 108, 112),
                    top=(
                        TokenLogprob(
                            token="help",
                            token_id=-1,
                            logprob=-2.0,
                            bytes_=None,
                        ),
                    ),
                ),
            ),
        ),
        TokenLogprob(
            token="lo",
            token_id=-1,
            logprob=-0.75,
            bytes_=None,
            top=(
                TokenLogprob(token="lo", token_id=-1, logprob=-0.75, bytes_=None),
                TokenLogprob(
                    token=" low",
                    token_id=-1,
                    logprob=-1.25,
                    bytes_=(32, 108, 111, 119),
                ),
            ),
        ),
    )
    assert output.cumulative_logprob == -1.0
    assert output.logprobs is None
    await backend.shutdown()


@pytest.mark.parametrize("content_state", ["logprobs-absent", "content-absent", "null"])
async def test_generate_no_logprobs_content_remains_none(content_state):
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": "plain"},
        "finish_reason": "stop",
    }
    if content_state == "content-absent":
        choice["logprobs"] = {}
    elif content_state == "null":
        choice["logprobs"] = {"content": None}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [choice]}))
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=transport,
    )

    output = (await backend.generate(_request())).completions[0]

    assert output.text == "plain"
    assert output.finish_reason == "stop"
    assert output.logprob_content is None
    assert output.cumulative_logprob is None
    assert output.logprobs is None
    await backend.shutdown()


async def test_generate_maps_upstream_logprobs_empty_content():
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": ""},
        "finish_reason": "stop",
        "logprobs": {"content": []},
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [choice]}))
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=transport,
    )

    output = (await backend.generate(_request())).completions[0]

    assert output.logprob_content == ()
    assert output.cumulative_logprob == 0.0
    assert output.logprobs is None
    await backend.shutdown()


async def test_generate_forwards_and_preserves_native_tool_calls():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_upstream",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"city":"Tokyo"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    tools = (
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object"},
            },
        },
    )
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
    )
    request = GenerationRequest(
        request_id="tools",
        prompt="weather",
        sampling_params=SamplingParams().with_generation_config_omitted(
            GENERATION_CONFIG_SAMPLING_FIELDS
        ),
        tools=tools,
        tool_choice="required",
    )

    result = await backend.generate(request)

    assert captured["body"]["tools"] == list(tools)
    assert captured["body"]["tool_choice"] == "required"
    assert result.completions[0].finish_reason == "tool_calls"
    assert result.text == ('<tool_call>{"name":"lookup","arguments":{"city":"Tokyo"}}</tool_call>')
    await backend.shutdown()


async def test_generate_forwards_representable_sampling_payload():
    captured: dict = {}
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=_ok_transport(captured),
    )
    params = SamplingParams(
        presence_penalty=0.4,
        frequency_penalty=-0.3,
        top_k=8,
        min_p=0.15,
        logprobs=3,
        extra_args={"response_format": {"type": "json_object"}},
    )

    await backend.generate(_request(sampling_params=params))

    assert captured["body"]["presence_penalty"] == 0.4
    assert captured["body"]["frequency_penalty"] == -0.3
    assert captured["body"]["top_k"] == 8
    assert captured["body"]["min_p"] == 0.15
    assert captured["body"]["logprobs"] is True
    assert captured["body"]["top_logprobs"] == 3
    assert captured["body"]["response_format"] == {"type": "json_object"}
    await backend.shutdown()


async def test_generate_forwards_logprobs_zero():
    captured: dict = {}
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=_ok_transport(captured),
    )

    await backend.generate(_request(sampling_params=SamplingParams(logprobs=0)))

    assert captured["body"]["logprobs"] is True
    assert captured["body"]["top_logprobs"] == 0
    await backend.shutdown()


async def test_generate_marked_generation_omissions_stay_absent_upstream():
    captured: dict = {}
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=_ok_transport(captured),
    )

    params = SamplingParams().with_generation_config_omitted(
        GENERATION_CONFIG_SAMPLING_FIELDS
    )
    await backend.generate(_request(sampling_params=params))

    assert captured["body"]["presence_penalty"] == 0.0
    assert captured["body"]["frequency_penalty"] == 0.0
    for field in (
        "top_k",
        "min_p",
        "repetition_penalty",
        "temperature",
        "top_p",
        "logprobs",
        "top_logprobs",
        "response_format",
    ):
        assert field not in captured["body"]
    await backend.shutdown()


@pytest.mark.parametrize("upstream", ["vllm", "kairyu"])
async def test_generate_forwards_explicit_neutral_generation_values(upstream):
    captured: dict = {}
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=_ok_transport(captured),
        upstream=upstream,
    )

    await backend.generate(
        _request(
            sampling_params=SamplingParams(),
            explicit_generation_neutrals=True,
        )
    )

    assert {
        field: captured["body"][field]
        for field in GENERATION_CONFIG_SAMPLING_FIELDS
    } == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }
    await backend.shutdown()


@pytest.mark.parametrize(
    ("field", "params"),
    [
        ("best_of", SamplingParams(best_of=2)),
        ("repetition_penalty", SamplingParams(repetition_penalty=1.1)),
        ("stop_token_ids", SamplingParams(stop_token_ids=[7])),
        ("min_tokens", SamplingParams(min_tokens=1)),
        ("prompt_logprobs", SamplingParams(prompt_logprobs=1)),
        ("ignore_eos", SamplingParams(ignore_eos=True)),
        ("skip_special_tokens", SamplingParams(skip_special_tokens=False)),
        (
            "extra_args.unknown_control",
            SamplingParams(extra_args={"unknown_control": True}),
        ),
        ("logprobs", SamplingParams(logprobs=-1)),
        ("extra_args", SamplingParams(extra_args=[])),
    ],
)
async def test_generate_rejects_unsupported_intent_before_client_or_transport(field, params):
    transport_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        transport_calls.append(request)
        return httpx.Response(200, json={"choices": []})

    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(UpstreamClientError) as exc_info:
        await backend.generate(_request(sampling_params=params))

    assert exc_info.value.status_code == 400
    assert field in str(exc_info.value)
    assert transport_calls == []
    assert backend._client is None


@pytest.mark.parametrize(
    ("upstream", "field", "params"),
    [
        ("openai", "top_k", SamplingParams(top_k=-1)),
        ("gemini", "temperature", SamplingParams(temperature=1.0)),
    ],
)
async def test_explicit_neutral_generation_value_is_not_silently_dropped(
    upstream,
    field,
    params,
):
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        upstream=upstream,
        api_key_env=None,
    )

    with pytest.raises(UpstreamClientError, match=field):
        await backend.generate(
            _request(
                sampling_params=params,
                explicit_generation_neutrals=True,
            )
        )


@pytest.mark.parametrize(
    ("upstream", "params", "expected"),
    [
        (
            "openai",
            SamplingParams(
                temperature=0.4,
                top_p=0.8,
                n=2,
                presence_penalty=0.2,
                frequency_penalty=-0.1,
                max_tokens=31,
                stop=["END"],
                seed=9,
                logprobs=2,
                extra_args={
                    "response_format": {"type": "json_object"},
                    "reasoning_effort": "low",
                },
            ),
            {
                "temperature": 0.4,
                "top_p": 0.8,
                "n": 2,
                "presence_penalty": 0.2,
                "frequency_penalty": -0.1,
                "max_completion_tokens": 31,
                "stop": ["END"],
                "seed": 9,
                "logprobs": True,
                "top_logprobs": 2,
                "response_format": {"type": "json_object"},
                "reasoning_effort": "low",
            },
        ),
        (
            "vllm",
            SamplingParams(
                max_tokens=23,
                top_k=7,
                min_p=0.2,
                repetition_penalty=1.1,
                stop_token_ids=[3, 4],
                min_tokens=2,
                ignore_eos=True,
                skip_special_tokens=False,
            ),
                {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "max_tokens": 23,
                "top_k": 7,
                "min_p": 0.2,
                "repetition_penalty": 1.1,
                "stop_token_ids": [3, 4],
                "min_tokens": 2,
                "ignore_eos": True,
                "skip_special_tokens": False,
                "priority": 0,
            },
        ),
        (
            "kairyu",
            SamplingParams(
                max_tokens=19,
                top_k=5,
                min_p=0.1,
                repetition_penalty=1.2,
                stop_token_ids=[8],
                min_tokens=4,
                ignore_eos=True,
            ),
                {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "max_completion_tokens": 19,
                "top_k": 5,
                "min_p": 0.1,
                "repetition_penalty": 1.2,
                "stop_token_ids": [8],
                "min_tokens": 4,
                "ignore_eos": True,
                "priority": 0,
            },
        ),
        (
            "anthropic",
            SamplingParams(
                max_tokens=17,
                temperature=0.7,
                top_p=0.9,
                stop=["DONE"],
            ),
            {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_completion_tokens": 17,
                "stop": ["DONE"],
            },
        ),
        (
            "gemini",
            SamplingParams(
                max_tokens=13,
                extra_args={
                    "response_format": {"type": "json_object"},
                    "extra_body": {
                        "google": {"thinking_config": {"thinking_budget": 64}}
                    },
                },
            ),
            {
                "max_completion_tokens": 13,
                "response_format": {"type": "json_object"},
                "extra_body": {
                    "google": {"thinking_config": {"thinking_budget": 64}}
                },
            },
        ),
    ],
)
async def test_upstream_profiles_forward_exact_supported_body(upstream, params, expected):
    captured: dict = {}
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=_ok_transport(captured),
        upstream=upstream,
    )

    await backend.generate(
        _request(
            sampling_params=params,
            explicit_generation_neutrals=upstream in {"vllm", "kairyu"},
        )
    )

    body = captured["body"]
    assert {key: body[key] for key in expected} == expected
    assert set(body) == {"model", "messages", *expected}
    await backend.shutdown()


@pytest.mark.parametrize(
    ("upstream", "field", "params"),
    [
        ("openai", "top_k", SamplingParams(top_k=8)),
        ("anthropic", "n", SamplingParams(n=2)),
        ("anthropic", "temperature", SamplingParams(temperature=1.1)),
        ("anthropic", "logprobs", SamplingParams(logprobs=1)),
        (
            "anthropic",
            "response_format",
            SamplingParams(extra_args={"response_format": {"type": "json_object"}}),
        ),
        ("gemini", "logprobs", SamplingParams(logprobs=1)),
        ("vllm", "prompt_logprobs", SamplingParams(prompt_logprobs=1)),
        ("kairyu", "prompt_logprobs", SamplingParams(prompt_logprobs=1)),
        (
            "openai",
            "extra_args.extra_body",
            SamplingParams(extra_args={"extra_body": {"google": {}}}),
        ),
        (
            "gemini",
            "temperature",
            SamplingParams(temperature=0.5),
        ),
    ],
)
async def test_upstream_profile_mismatch_is_400_before_transport(upstream, field, params):
    transport_calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        transport_calls.append(request)
        return httpx.Response(200, json={"choices": []})

    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
        upstream=upstream,
    )

    with pytest.raises(UpstreamClientError, match=field) as exc_info:
        await backend.generate(_request(sampling_params=params))

    assert exc_info.value.status_code == 400
    assert transport_calls == []
    assert backend._client is None


@pytest.mark.parametrize(
    "prompt",
    [
        TokensPrompt((1, 2, 3)),
        MultimodalPrompt(
            base="describe",
            items=(
                MultimodalItem(
                    modality="image",
                    encoding="uri",
                    data="https://example.test/image.png",
                ),
            ),
        ),
    ],
)
async def test_non_text_prompt_is_rejected_before_http_client_or_transport(prompt):
    transport_calls: list[httpx.Request] = []
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=httpx.MockTransport(
            lambda request: (
                transport_calls.append(request)
                or httpx.Response(200, json={"choices": []})
            )
        ),
        upstream="vllm",
    )

    with pytest.raises(UpstreamClientError, match="prompt kind") as exc_info:
        await backend.generate(_request(prompt=prompt))

    assert exc_info.value.status_code == 400
    assert transport_calls == []
    assert backend._client is None


async def test_anthropic_strict_tool_schema_is_rejected_before_transport():
    transport_calls: list[httpx.Request] = []
    backend = OpenAICompatBackend(
        base_url="https://api.anthropic.example/v1",
        model="claude",
        api_key_env=None,
        transport=httpx.MockTransport(
            lambda request: (
                transport_calls.append(request)
                or httpx.Response(200, json={"choices": []})
            )
        ),
        upstream="anthropic",
    )
    request = GenerationRequest(
        request_id="strict",
        prompt="call",
        sampling_params=SamplingParams(),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                    "strict": True,
                },
            },
        ),
    )

    with pytest.raises(UpstreamClientError, match=r"tools\[0\]\.function\.strict"):
        await backend.generate(request)

    assert transport_calls == []
    assert backend._client is None


async def test_custom_capability_allows_reviewed_sampling_and_vendor_extensions():
    captured: dict = {}
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=_ok_transport(captured),
        upstream="generic",
        capabilities={
            "allow_sampling_fields": ["best_of"],
            "allow_extra_args": ["vendor_cache"],
        },
    )
    params = SamplingParams(
        n=2,
        best_of=3,
        extra_args={"vendor_cache": {"mode": "ephemeral"}},
    )

    await backend.generate(_request(sampling_params=params))

    assert captured["body"]["best_of"] == 3
    assert captured["body"]["vendor_cache"] == {"mode": "ephemeral"}
    await backend.shutdown()


@pytest.mark.parametrize(
    "reserved",
    [
        "model",
        "prompt",
        "prompt_token_ids",
        "prompt_embeds",
        "multi_modal_data",
        "mm_processor_kwargs",
        "stream",
        "temperature",
        "response_format",
    ],
)
def test_capability_configuration_cannot_allow_core_overrides(reserved):
    with pytest.raises(ValueError, match="may not override"):
        OpenAICompatBackend(
            base_url="https://api.example.com/v1",
            model="m",
            capabilities={"allow_extra_args": [reserved]},
        )


def test_named_provider_capability_configuration_cannot_broaden_sampling_contract():
    with pytest.raises(ValueError, match="require upstream='generic'"):
        OpenAICompatBackend(
            base_url="https://api.example.com/v1",
            model="m",
            upstream="anthropic",
            capabilities={"allow_sampling_fields": ["frequency_penalty"]},
        )


def test_unrepresentable_prompt_logprobs_cannot_be_enabled_by_custom_policy():
    with pytest.raises(ValueError, match="unknown OpenAI sampling capability fields"):
        OpenAICompatBackend(
            base_url="https://verified.example/v1",
            model="m",
            upstream="generic",
            capabilities={"allow_sampling_fields": ["prompt_logprobs"]},
        )
    with pytest.raises(ValueError, match="may not override"):
        OpenAICompatBackend(
            base_url="https://verified.example/v1",
            model="m",
            upstream="generic",
            capabilities={"allow_extra_args": ["prompt_logprobs"]},
        )
    with pytest.raises(ValueError, match="unknown OpenAI sampling capability fields"):
        OpenAIRequestCapabilities(
            upstream="custom",
            sampling_fields={"prompt_logprobs"},
        )


def test_upstream_profile_names_are_explicit_and_unknown_fails_at_construction():
    assert known_openai_upstreams() == (
        "anthropic",
        "gemini",
        "generic",
        "kairyu",
        "openai",
        "vllm",
    )
    with pytest.raises(ValueError, match="unknown OpenAI-compatible upstream"):
        OpenAICompatBackend(
            base_url="https://api.example.com/v1",
            model="m",
            upstream="url-guessed-provider",
        )


def test_resolved_capability_policy_is_deeply_immutable():
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        upstream="openai",
    )
    assert isinstance(backend.capabilities.sampling_fields, frozenset)
    assert isinstance(backend.capabilities.extra_args, frozenset)
    assert backend.capabilities.prompt_kinds == frozenset({"text"})


def test_adapter_rejects_a_capability_contract_it_cannot_execute():
    capabilities = OpenAIRequestCapabilities(
        upstream="custom",
        sampling_fields={"max_tokens"},
        prompt_kinds={"text", "tokens"},
    )

    with pytest.raises(ValueError, match="text and multimodal"):
        OpenAICompatBackend(
            base_url="https://api.example.com/v1",
            model="m",
            capabilities=capabilities,
        )


async def test_multimodal_chat_forwards_roles_part_order_and_exact_usage():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "RED"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 37,
                    "completion_tokens": 1,
                    "total_tokens": 38,
                },
            },
        )

    backend = OpenAICompatBackend(
        base_url="http://vlm:8000/v1",
        model="vlm",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
        upstream="vllm",
        capabilities={"allow_prompt_kinds": ["multimodal"]},
        image_input_policy={
            "max_images": 1,
            "max_processed_prompt_tokens": 512,
        },
    )
    request = _request(_multimodal_prompt())

    result = await backend.generate(request)

    assert captured["body"]["messages"] == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Answer exactly."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before "},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _RED_PNG_DATA_URL,
                        "detail": "high",
                    },
                },
                {"type": "text", "text": " after"},
            ],
        },
    ]
    assert result.text == "RED"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 37
    assert result.usage.completion_tokens == 1
    bound = backend.admission_upper_bound(request)
    assert bound.tokens == 512 + 64
    assert bound.refundable_on_exact_usage is True


def test_multimodal_backend_requires_stream_usage_reporting():
    with pytest.raises(ValueError, match="request_stream_usage=true"):
        OpenAICompatBackend(
            base_url="http://vlm:8000/v1",
            model="vlm",
            api_key_env=None,
            upstream="vllm",
            capabilities={"allow_prompt_kinds": ["multimodal"]},
            image_input_policy={"max_images": 1},
            request_stream_usage=False,
        )


async def test_token_based_multimodal_prompt_is_rejected_before_media_or_transport(
    monkeypatch,
):
    decode_calls = 0
    original_decode = ImageInputPolicy._decode_item

    def recording_decode(self, item, *, index):
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(self, item, index=index)

    monkeypatch.setattr(ImageInputPolicy, "_decode_item", recording_decode)
    transport_calls: list[httpx.Request] = []
    backend = OpenAICompatBackend(
        base_url="http://vlm:8000/v1",
        model="vlm",
        api_key_env=None,
        transport=httpx.MockTransport(
            lambda request: (
                transport_calls.append(request)
                or httpx.Response(200, json={"choices": []})
            )
        ),
        upstream="vllm",
        capabilities={"allow_prompt_kinds": ["multimodal"]},
        image_input_policy={"max_images": 1},
    )
    original = _multimodal_prompt()
    prompt = MultimodalPrompt(
        base=TokensPrompt((123, 456), prompt="display"),
        items=original.items,
        messages=original.messages,
    )

    with pytest.raises(UpstreamClientError, match="token-based multimodal"):
        await backend.generate(_request(prompt))

    assert decode_calls == 0
    assert transport_calls == []
    assert backend._client is None


async def test_multimodal_full_decode_runs_once_off_the_event_loop(monkeypatch):
    vision_module._clear_verified_raster_cache()
    decode_threads: list[int] = []
    original_verify = ImageInputPolicy._verify_raster

    def recording_verify(data, *, mime, index):
        decode_threads.append(threading.get_ident())
        original_verify(data, mime=mime, index=index)

    monkeypatch.setattr(
        ImageInputPolicy,
        "_verify_raster",
        staticmethod(recording_verify),
    )
    backend = OpenAICompatBackend(
        base_url="http://vlm:8000/v1",
        model="vlm",
        api_key_env=None,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "RED",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 37,
                        "completion_tokens": 1,
                    },
                },
            )
        ),
        upstream="vllm",
        capabilities={"allow_prompt_kinds": ["multimodal"]},
        image_input_policy={"max_images": 1},
    )
    request = _request(_multimodal_prompt())

    backend.validate_request(request)
    assert decode_threads == []

    await backend.generate(request)

    assert len(decode_threads) == 1
    assert decode_threads[0] != threading.get_ident()


async def test_multimodal_remote_url_is_400_before_http_client_or_transport():
    transport_calls: list[httpx.Request] = []
    backend = OpenAICompatBackend(
        base_url="http://vlm:8000/v1",
        model="vlm",
        api_key_env=None,
        transport=httpx.MockTransport(
            lambda request: (
                transport_calls.append(request)
                or httpx.Response(200, json={"choices": []})
            )
        ),
        upstream="vllm",
        capabilities={"allow_prompt_kinds": ["multimodal"]},
        image_input_policy={"max_images": 1},
    )

    with pytest.raises(UpstreamClientError, match="remote and local") as exc_info:
        await backend.generate(
            _request(_multimodal_prompt("https://example.test/image.png"))
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == "invalid_image"
    assert transport_calls == []
    assert backend._client is None


async def test_multimodal_unary_response_must_report_processed_prompt_usage():
    backend = OpenAICompatBackend(
        base_url="http://vlm:8000/v1",
        model="vlm",
        api_key_env=None,
        transport=_ok_transport({}),
        upstream="vllm",
        capabilities={"allow_prompt_kinds": ["multimodal"]},
        image_input_policy={"max_images": 1},
    )

    with pytest.raises(RuntimeError, match="omitted exact processed prompt usage"):
        await backend.generate(_request(_multimodal_prompt()))


async def test_stream_rejects_unsupported_intent_before_client_or_transport():
    transport_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        transport_calls.append(request)
        return httpx.Response(200, content=_SSE_BODY)

    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(UpstreamClientError, match="best_of"):
        async for _ in backend.stream(_request(sampling_params=SamplingParams(best_of=2))):
            pass

    assert transport_calls == []
    assert backend._client is None


async def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1", model="m", api_key_env="MISSING_KEY"
    )
    with pytest.raises(RuntimeError, match="MISSING_KEY"):
        await backend.generate(_request())


async def test_client_error_surfaces_as_upstream_client_error(monkeypatch):
    # O1: a 4xx is the client's fault, raised as UpstreamClientError so the
    # ReplicaPool does not count it against the replica's health.
    monkeypatch.setenv("TEST_API_KEY", "sk-test")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(400, json={"error": {"message": "bad request"}})
    )
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env="TEST_API_KEY",
        transport=transport,
    )
    with pytest.raises(UpstreamClientError) as excinfo:
        await backend.generate(_request())
    assert excinfo.value.status_code == 400


async def test_server_error_surfaces_as_runtime_error(monkeypatch):
    # 5xx is a replica/transport failure the pool SHOULD count.
    monkeypatch.setenv("TEST_API_KEY", "sk-test")
    transport = httpx.MockTransport(lambda request: httpx.Response(503, text="unavailable"))
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env="TEST_API_KEY",
        transport=transport,
    )
    with pytest.raises(RuntimeError, match="503"):
        await backend.generate(_request())


# --- m6 D2 fixes: real streaming, pooled client, optional auth, token counts ---

_SSE_BODY = (
    b'data: {"choices":[{"index":0,"delta":{"content":"hel"}}]}\n\n'
    b'data: {"choices":[{"index":0,"delta":{"content":"lo"}}]}\n\n'
    b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    b"data: [DONE]\n\n"
)


class _ChunkedAsyncStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


def _chunked_sse_transport(*chunks: bytes) -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            stream=_ChunkedAsyncStream(*chunks),
            headers={"content-type": "text/event-stream"},
        )
    )


async def test_multimodal_stream_requires_and_returns_exact_final_usage():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        body = (
            b'data: {"choices":[{"index":0,"delta":{"content":"RED"}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":41,'
            b'"completion_tokens":1,"total_tokens":42}}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    backend = OpenAICompatBackend(
        base_url="http://vlm:8000/v1",
        model="vlm",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
        upstream="vllm",
        capabilities={"allow_prompt_kinds": ["multimodal"]},
        image_input_policy={"max_images": 1},
    )

    results = [
        result
        async for result in backend.stream(_request(_multimodal_prompt()))
    ]

    assert captured["body"]["stream_options"] == {"include_usage": True}
    assert captured["body"]["messages"][1]["content"][1]["type"] == "image_url"
    assert results[-1].finished is True
    assert results[-1].text == "RED"
    assert results[-1].usage is not None
    assert results[-1].usage.prompt_tokens == 41
    assert results[-1].usage.completion_tokens == 1


def _sse_transport(captured: dict) -> httpx.MockTransport:
    def handler(http_request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(http_request.content)
        return httpx.Response(200, content=_SSE_BODY, headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


def _sse_chunks_transport(*chunks: dict) -> httpx.MockTransport:
    body = (
        b"".join(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks)
        + b"data: [DONE]\n\n"
    )
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )
    )


async def test_stream_parses_sse_into_cumulative_partials(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-test")
    captured: dict = {}
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env="TEST_API_KEY",
        transport=_sse_transport(captured),
    )

    results = [result async for result in backend.stream(_request("stream it"))]

    assert captured["body"]["stream"] is True
    texts = [result.completions[0].text for result in results]
    assert texts == ["hel", "hello", "hello"]
    assert [result.finished for result in results] == [False, False, True]
    assert results[-1].completions[0].finish_reason == "stop"
    await backend.shutdown()


async def test_stream_propagates_and_parses_terminal_kairyu_stage_trace():
    captured: dict[str, object] = {}
    trace = _stage_trace(
        ("queue_wait", 202, 1),
        ("sse_write", 404, 4),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        captured["trace_header"] = request.headers.get("x-kairyu-trace")
        metadata = json.dumps(
            {
                "id": "chatcmpl-upstream",
                "choices": [],
                "kairyu_trace_v2": trace,
            }
        )
        body = (
            b'data: {"id":"chatcmpl-upstream","choices":'
            b'[{"index":0,"delta":{"content":"ok"}}]}\n\n'
            b'data: {"id":"chatcmpl-upstream","choices":'
            b'[{"index":0,"delta":{},'
            b'"finish_reason":"stop"}]}\n\n'
            + f"data: {metadata}\n\n".encode()
            + b"data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    backend = OpenAICompatBackend(
        base_url="http://replica:8000/v1",
        model="m",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
        upstream="kairyu",
    )

    results = [
        result
        async for result in backend.stream(_request(trace_requested=True))
    ]

    assert captured["trace_header"] == "1"
    assert len(results) == 2
    assert results[0].stage_metrics == ()
    assert [
        (metric.stage, metric.duration_ns, metric.occurrences)
        for metric in results[-1].stage_metrics
    ] == [("queue_wait", 202, 1), ("sse_write", 404, 4)]


@pytest.mark.parametrize(
    ("trailer", "match"),
    (
        (
            b'data: {"id":"chatcmpl-upstream","choices":[]}\n\n'
            b"data: [DONE]\n\n",
            "not terminal",
        ),
        (b"", "omitted SSE"),
    ),
)
async def test_stream_rejects_nonterminal_or_unfinished_kairyu_stage_trace(
    trailer,
    match,
):
    trace = _stage_trace(("tokenize", 101, 1))
    metadata = json.dumps(
        {
            "id": "chatcmpl-upstream",
            "choices": [],
            "kairyu_trace_v2": trace,
        }
    )
    body = (
        b'data: {"id":"chatcmpl-upstream","choices":'
        b'[{"index":0,"delta":{"content":"ok"}}]}\n\n'
        + f"data: {metadata}\n\n".encode()
        + trailer
    )
    backend = OpenAICompatBackend(
        base_url="http://replica:8000/v1",
        model="m",
        api_key_env=None,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=body,
                headers={"content-type": "text/event-stream"},
            )
        ),
        upstream="kairyu",
    )

    with pytest.raises(RuntimeError, match=match):
        _ = [
            result
            async for result in backend.stream(_request(trace_requested=True))
        ]


async def test_stream_preserves_kairyu_stage_metrics_before_sanitized_error():
    trace = _stage_trace(("decode_step", 303, 2))
    metadata = json.dumps(
        {
            "id": "chatcmpl-upstream",
            "choices": [],
            "kairyu_trace_v2": trace,
        }
    )
    body = (
        b'data: {"id":"chatcmpl-upstream","choices":'
        b'[{"index":0,"delta":{"content":"ok"}}]}\n\n'
        + f"data: {metadata}\n\n".encode()
        + b'data: {"error":{"message":"secret upstream detail",'
        b'"type":"upstream_error"}}\n\n'
        + b"data: [DONE]\n\n"
    )
    backend = OpenAICompatBackend(
        base_url="http://replica:8000/v1",
        model="m",
        api_key_env=None,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=body,
                headers={"content-type": "text/event-stream"},
            )
        ),
        upstream="kairyu",
    )
    yielded = []

    with pytest.raises(RuntimeError, match="reported an error") as caught:
        async for result in backend.stream(_request(trace_requested=True)):
            yielded.append(result)

    assert "secret upstream detail" not in str(caught.value)
    assert yielded[-1].stage_metrics == (
        GenerationStageMetric("decode_step", 303, 2),
    )


async def test_stream_preserves_unicode_line_separators_inside_json():
    separators = "\u0085\u2028\u2029"
    content_event = (
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": separators},
                    }
                ]
            },
            ensure_ascii=False,
        )
        + "\n\n"
    ).encode()
    finish = (
        b'data: {"choices":[{"index":0,"delta":{},'
        b'"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    separator_start = content_event.index("\u2028".encode())
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=_chunked_sse_transport(
            content_event[: separator_start + 1],
            content_event[separator_start + 1 :],
            finish,
        ),
    )

    results = [result async for result in backend.stream(_request("stream it"))]

    assert [result.text for result in results] == [separators, separators]
    assert results[-1].finished is True


async def test_stream_parses_fragmented_multiline_sse_data_event():
    body = (
        b": keepalive\r\n"
        b"event: ignored\r\n"
        b'data: {"choices":[\r\n'
        b'data: {"index":0,"delta":{"content":"hello"}}\r\n'
        b"data: ]}\r\n"
        b"\r\n"
        b'data: {"choices":[{"index":0,"delta":{},'
        b'"finish_reason":"stop"}]}\r\n'
        b"\r\n"
        b"data: [DONE]\r\n"
        b"\r\n"
    )
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=_chunked_sse_transport(
            body[:17],
            body[17:31],
            body[31:92],
            body[92:-1],
            body[-1:],
        ),
    )

    results = [result async for result in backend.stream(_request("stream it"))]

    assert [result.text for result in results] == ["hello", "hello"]
    assert results[-1].completions[0].finish_reason == "stop"


async def test_stream_rejects_complete_malformed_sse_json():
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=_chunked_sse_transport(
            b'data: {"choices":[}\n\n',
            b"data: [DONE]\n\n",
        ),
    )

    with pytest.raises(json.JSONDecodeError):
        _ = [result async for result in backend.stream(_request("stream it"))]


async def test_stream_returns_empty_text_for_valid_empty_single_choice():
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=_sse_chunks_transport(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ),
    )

    results = [result async for result in backend.stream(_request())]

    assert len(results) == 1
    assert results[0].finished is True
    assert len(results[0].completions) == 1
    completion = results[0].completions[0]
    assert completion.index == 0
    assert completion.text == ""
    assert completion.token_ids == ()
    assert completion.finish_reason == "stop"
    await backend.shutdown()


async def test_stream_preserves_empty_choice_alongside_nonempty_with_n_gt_1():
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=_sse_chunks_transport(
            {
                "choices": [
                    {"index": 0, "delta": {"content": "hello"}},
                    {"index": 1, "delta": {"role": "assistant"}},
                ]
            },
            {
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"},
                    {"index": 1, "delta": {}, "finish_reason": "stop"},
                ]
            },
        ),
    )

    results = [
        result async for result in backend.stream(_request(sampling_params=SamplingParams(n=2)))
    ]

    final = results[-1]
    assert final.finished is True
    assert [completion.index for completion in final.completions] == [0, 1]
    assert [completion.text for completion in final.completions] == ["hello", ""]
    assert [completion.finish_reason for completion in final.completions] == [
        "stop",
        "stop",
    ]
    await backend.shutdown()


async def test_stream_preserves_choice_scoped_logprobs_cumulatively():
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=_sse_chunks_transport(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "a"},
                        "logprobs": {
                            "content": [
                                {
                                    "token": "a",
                                    "logprob": -0.1,
                                    "bytes": [97],
                                    "top_logprobs": [],
                                }
                            ]
                        },
                    },
                    {
                        "index": 1,
                        "delta": {"content": "b"},
                        "logprobs": {
                            "content": [
                                {
                                    "token": "b",
                                    "logprob": -0.2,
                                    "bytes": [98],
                                    "top_logprobs": [],
                                }
                            ]
                        },
                    },
                ]
            },
            {
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"},
                    {"index": 1, "delta": {}, "finish_reason": "stop"},
                ]
            },
        ),
    )

    results = [
        result
        async for result in backend.stream(
            _request(sampling_params=SamplingParams(n=2, logprobs=1))
        )
    ]

    final = results[-1]
    assert [
        [entry.token for entry in completion.logprob_content] for completion in final.completions
    ] == [["a"], ["b"]]
    assert [completion.cumulative_logprob for completion in final.completions] == [-0.1, -0.2]
    await backend.shutdown()


async def test_stream_raises_when_no_choices_observed():
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=_sse_chunks_transport(
            {
                "choices": [],
                "usage": {"prompt_tokens": 2, "completion_tokens": 0},
            }
        ),
    )

    with pytest.raises(RuntimeError, match="streamed no choices"):
        async for _ in backend.stream(_request()):
            pass
    await backend.shutdown()


async def test_keyless_backend_omits_auth_header():
    captured: dict = {}
    backend = OpenAICompatBackend(
        base_url="http://node-b:8000/v1",
        model="m",
        api_key_env=None,  # node-to-node replica: no auth (design m6 D2)
        transport=_ok_transport(captured),
    )
    result = await backend.generate(_request())
    assert result.completions[0].text == "hello from api"
    assert captured["auth"] is None
    await backend.shutdown()


async def test_async_client_is_reused_across_requests(monkeypatch):
    instances = []
    real_client = httpx.AsyncClient

    class CountingClient(real_client):
        def __init__(self, *args, **kwargs):
            instances.append(self)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", CountingClient)
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=_ok_transport({}),
    )
    await backend.generate(_request())
    await backend.generate(_request())
    assert len(instances) == 1  # persistent pooled client, no per-request handshake
    await backend.shutdown()
    assert backend._client is None


async def test_owned_client_factory_creates_once_and_closes_with_backend():
    captured: dict = {}
    clients: list[httpx.AsyncClient] = []

    def client_factory() -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=_ok_transport(captured))
        clients.append(client)
        return client

    backend = OpenAICompatBackend(
        base_url="http://replica-a:8000/v1",
        model="m",
        api_key_env=None,
        client_factory=client_factory,
    )
    assert backend._client is None

    await backend.generate(_request())
    await backend.generate(_request())

    assert len(clients) == 1
    assert backend._client is clients[0]
    await backend.shutdown()
    assert clients[0].is_closed is True
    assert backend._client is None


def test_validation_key_is_exactly_the_immutable_capability_contract():
    first = OpenAICompatBackend(
        base_url="http://replica-a:8000/v1",
        model="m",
        api_key_env=None,
        upstream="kairyu",
    )
    second = OpenAICompatBackend(
        base_url="http://replica-b:8000/v1",
        model="m",
        api_key_env=None,
        upstream="kairyu",
    )
    different = OpenAICompatBackend(
        base_url="http://replica-c:8000/v1",
        model="m",
        api_key_env=None,
        upstream="generic",
    )

    assert first.request_validation_key == (first.capabilities, None)
    assert first.request_validation_key == second.request_validation_key
    assert first.request_validation_key != different.request_validation_key
    assert isinstance(hash(first.request_validation_key), int)


def test_validation_key_requires_subclasses_to_explicitly_opt_in():
    class StatefulValidationBackend(OpenAICompatBackend):
        def validate_request(self, request):
            return None

    backend = StatefulValidationBackend(
        base_url="http://replica-a:8000/v1",
        model="m",
        api_key_env=None,
        upstream="kairyu",
    )

    assert backend.request_validation_key is None


async def test_client_factory_rejects_a_closed_client():
    client = httpx.AsyncClient(transport=_ok_transport({}))
    await client.aclose()
    backend = OpenAICompatBackend(
        base_url="http://replica-a:8000/v1",
        model="m",
        api_key_env=None,
        client_factory=lambda: client,
    )

    with pytest.raises(RuntimeError, match="client_factory returned a closed"):
        await backend.generate(_request())

    assert backend._client is None


async def test_owned_client_shutdown_is_terminal_exactly_once_and_cancellation_safe():
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingClient:
        def __init__(self) -> None:
            self.is_closed = False
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1
            started.set()
            await release.wait()
            self.is_closed = True

    client = BlockingClient()
    backend = OpenAICompatBackend(
        base_url="http://replica-a:8000/v1",
        model="m",
        api_key_env=None,
        client_factory=lambda: client,
    )
    assert backend._get_client() is client

    first = asyncio.create_task(backend.shutdown())
    await asyncio.wait_for(started.wait(), timeout=0.2)
    second = asyncio.create_task(backend.shutdown())
    await asyncio.sleep(0)
    assert second.done() is False

    first.cancel()
    await asyncio.sleep(0)
    assert first.done() is False
    assert second.done() is False
    with pytest.raises(RuntimeError, match="is shut down"):
        backend._get_client()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second

    assert client.is_closed is True
    assert client.close_calls == 1
    assert backend._client is None
    await backend.shutdown()
    assert client.close_calls == 1


async def test_external_client_is_shared_and_not_closed_by_replica_shutdown():
    captured: dict = {}
    client = httpx.AsyncClient(transport=_ok_transport(captured))
    first = OpenAICompatBackend(
        base_url="http://replica-a:8000/v1",
        model="m",
        api_key_env=None,
        client=client,
    )
    second = OpenAICompatBackend(
        base_url="http://replica-b:8000/v1",
        model="m",
        api_key_env=None,
        client=client,
    )

    await first.generate(_request())
    await first.shutdown()
    assert client.is_closed is False
    assert first._client is client
    with pytest.raises(RuntimeError, match="is shut down"):
        await first.generate(_request())

    await second.generate(_request())
    assert second._client is first._client
    await second.shutdown()
    assert client.is_closed is False
    await client.aclose()


async def test_stream_uses_backend_timeout_with_external_client():
    observed: dict[str, float] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(request.extensions["timeout"])
        return httpx.Response(
            200,
            content=_SSE_BODY,
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(
        timeout=0.125,
        transport=httpx.MockTransport(handler),
    )
    backend = OpenAICompatBackend(
        base_url="http://replica:8000/v1",
        model="m",
        api_key_env=None,
        timeout_s=17.0,
        client=client,
    )

    results = [result async for result in backend.stream(_request())]

    assert results[-1].finished is True
    assert observed == {
        "connect": 17.0,
        "read": 17.0,
        "write": 17.0,
        "pool": 17.0,
    }
    await backend.shutdown()
    assert client.is_closed is False
    await client.aclose()


async def test_closed_external_client_is_not_silently_replaced():
    client = httpx.AsyncClient(transport=_ok_transport({}))
    backend = OpenAICompatBackend(
        base_url="http://replica:8000/v1",
        model="m",
        api_key_env=None,
        client=client,
    )
    await client.aclose()

    with pytest.raises(RuntimeError, match="externally owned HTTP client is closed"):
        await backend.generate(_request())

    assert backend._client is client


def test_external_client_and_transport_are_mutually_exclusive():
    client = httpx.AsyncClient(transport=_ok_transport({}))
    with pytest.raises(ValueError, match="client and transport are mutually exclusive"):
        OpenAICompatBackend(
            base_url="http://replica:8000/v1",
            model="m",
            api_key_env=None,
            client=client,
            transport=_ok_transport({}),
        )


def test_client_factory_is_mutually_exclusive_with_client_and_transport():
    client = httpx.AsyncClient(transport=_ok_transport({}))

    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=_ok_transport({}))

    with pytest.raises(
        ValueError,
        match="client and client_factory are mutually exclusive",
    ):
        OpenAICompatBackend(
            base_url="http://replica:8000/v1",
            model="m",
            api_key_env=None,
            client=client,
            client_factory=factory,
        )
    with pytest.raises(
        ValueError,
        match="client_factory and transport are mutually exclusive",
    ):
        OpenAICompatBackend(
            base_url="http://replica:8000/v1",
            model="m",
            api_key_env=None,
            client_factory=factory,
            transport=_ok_transport({}),
        )


async def test_usage_completion_tokens_populate_token_ids():
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "five tokens here"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 5},
            },
        )

    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
    )
    result = await backend.generate(_request())
    assert len(result.completions[0].token_ids) == 5
    await backend.shutdown()
