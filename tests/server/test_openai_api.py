import json

import httpx
import pytest

from kairyu.engine.backend import GenerationUsage
from kairyu.engine.kairyu_backend import KairyuBackend
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import UsageLedger
from kairyu.orchestration.orchestrator import Orchestrator

TOOL_CALL_TEXT = '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'


class _EmptyTokenizer:
    eos_token_id = None

    def encode(self, text: str) -> tuple[int, ...]:
        return ()

    def decode(self, token_ids) -> str:
        return ""

    def vocab(self) -> list[str]:
        return []


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture()
def app():
    engines = {
        "kairyu-mock": MockBackend(responses={"weather": TOOL_CALL_TEXT}),
    }
    orchestrator = Orchestrator(engines={"tier1": MockBackend(), "tier2": MockBackend()})
    return create_app(engines=engines, orchestrator=orchestrator)


def _chat_body(content: str, **extra) -> dict:
    return {
        "model": "kairyu-mock",
        "messages": [{"role": "user", "content": content}],
        **extra,
    }


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "completions"])
async def test_zero_token_prompt_returns_400_before_streaming(endpoint, stream):
    backend = KairyuBackend(tokenizer=_EmptyTokenizer())
    app = create_app(engines={"empty": backend})
    if endpoint == "chat":
        path = "/v1/chat/completions"
        body = {
            "model": "empty",
            "messages": [{"role": "user", "content": "normal-looking input"}],
            "stream": stream,
        }
    else:
        path = "/v1/completions"
        body = {
            "model": "empty",
            "prompt": "normal-looking input",
            "stream": stream,
        }

    async with _client(app) as client:
        response = await client.post(path, json=body)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "at least one token" in response.json()["error"]["message"]
    assert not response.headers["content-type"].startswith("text/event-stream")


async def test_backend_without_validate_request_still_serves_requests():
    app = create_app(engines={"mock": MockBackend()})
    async with _client(app) as client:
        response = await client.post(
            "/v1/completions", json={"model": "mock", "prompt": "hello"}
        )
    assert response.status_code == 200


async def test_models_endpoint_lists_engines_and_auto(app):
    async with _client(app) as client:
        response = await client.get("/v1/models")
    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert "kairyu-mock" in ids
    assert "kairyu-auto" in ids


async def test_chat_completion_happy_path(app):
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat_body("hello"))
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "kairyu-mock"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"]
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["total_tokens"] >= 0


async def test_streaming_reassembles_to_full_answer(app):
    async with _client(app) as client:
        full = (await client.post("/v1/chat/completions", json=_chat_body("hello"))).json()
        full_text = full["choices"][0]["message"]["content"]
        response = await client.post(
            "/v1/chat/completions", json=_chat_body("hello", stream=True)
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    deltas = []
    for line in lines[:-1]:
        chunk = json.loads(line[len("data: "):])
        assert chunk["object"] == "chat.completion.chunk"
        delta = chunk["choices"][0]["delta"]
        deltas.append(delta.get("content") or "")
    assert "".join(deltas) == full_text


async def test_tool_call_is_parsed_into_openai_schema(app):
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions", json=_chat_body("what is the weather?", tools=tools)
        )
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Tokyo"}


async def test_streamed_tool_calls_carry_required_index(app):
    # S6: each streamed delta.tool_calls[] item must include `index` so OpenAI
    # SDK stream accumulators can merge the fragments.
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]
    async with _client(app) as client:
        async with client.stream(
            "POST", "/v1/chat/completions",
            json=_chat_body("what is the weather?", tools=tools, stream=True),
        ) as response:
            saw_tool_call = False
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[len("data: "):])
                for tc in chunk["choices"][0]["delta"].get("tool_calls") or []:
                    assert "index" in tc
                    saw_tool_call = True
    assert saw_tool_call  # a tool-call delta was actually emitted


async def test_kairyu_auto_routes_through_orchestrator(app):
    body = _chat_body("What is the capital of France?")
    body["model"] = "kairyu-auto"
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"]


async def test_unknown_model_returns_404(app):
    body = _chat_body("hi")
    body["model"] = "ghost"
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 404
    assert "ghost" in response.json()["error"]["message"]


async def test_invalid_body_returns_422(app):
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json={"model": "kairyu-mock"})
    assert response.status_code == 422


async def test_invalid_sampling_params_return_400_not_500(app):
    # S2: out-of-range sampling values must be a client error (400), not an
    # unhandled ValueError surfacing as 500 (chat) or a mislabeled 502.
    async with _client(app) as client:
        for bad in ({"top_p": 0}, {"n": 0}, {"temperature": -1}):
            response = await client.post("/v1/chat/completions", json=_chat_body("hi", **bad))
            assert response.status_code == 400, (bad, response.status_code)
            assert response.json()["error"]["type"] == "invalid_request_error"


async def test_completions_invalid_sampling_returns_400(app):
    async with _client(app) as client:
        response = await client.post(
            "/v1/completions", json={"model": "kairyu-mock", "prompt": "hi", "top_p": 0}
        )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


class StubBackend:
    """Fixed completions / optional failure, for finish_reason and error tests."""

    def __init__(
        self, text="stub answer", finish_reason="length", error=None, usage=None
    ):
        self._text = text
        self._finish_reason = finish_reason
        self._error = error
        self._usage = usage
        self.calls = 0

    async def generate(self, request):
        from kairyu.engine.backend import GenerationResult
        from kairyu.outputs import CompletionOutput

        self.calls += 1
        if self._error is not None:
            raise self._error
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0, text=self._text, token_ids=(), finish_reason=self._finish_reason
                ),
            ),
            usage=self._usage,
        )

    async def stream(self, request):
        yield await self.generate(request)

    async def shutdown(self):
        return None


_WEATHER_TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object"}},
}


@pytest.mark.parametrize(
    ("tools", "tool_choice"),
    [
        (None, None),
        ([], None),
        ([_WEATHER_TOOL], None),
        (None, "auto"),
        ([], "auto"),
        ([_WEATHER_TOOL], "auto"),
        (None, "none"),
        ([], "none"),
        ([_WEATHER_TOOL], "none"),
        ([_WEATHER_TOOL], "required"),
        (
            [_WEATHER_TOOL],
            {"type": "function", "function": {"name": "get_weather"}},
        ),
    ],
)
async def test_coherent_tool_choice_is_accepted(tools, tool_choice):
    engine = StubBackend(text=TOOL_CALL_TEXT, finish_reason="stop")
    app = create_app(engines={"stub": engine})
    body = _chat_body("weather", tools=tools, tool_choice=tool_choice)
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert engine.calls == 1


@pytest.mark.parametrize(
    ("tools", "tool_choice"),
    [
        (None, "required"),
        ([], "required"),
        ([_WEATHER_TOOL], "any"),
        ([_WEATHER_TOOL], {}),
        ([_WEATHER_TOOL], {"type": "function"}),
        (
            [_WEATHER_TOOL],
            {"type": "other", "function": {"name": "get_weather"}},
        ),
        ([_WEATHER_TOOL], {"type": "function", "function": {}}),
        (
            [_WEATHER_TOOL],
            {"type": "function", "function": {"name": ""}},
        ),
        (
            [_WEATHER_TOOL],
            {"type": "function", "function": {"name": 7}},
        ),
        (
            [_WEATHER_TOOL],
            {"type": "function", "function": {"name": "other"}},
        ),
        ([{"type": "function", "function": {}}], "auto"),
        ([{"type": "function", "function": {"name": ""}}], "none"),
        ([{"type": "function", "function": {"name": "   "}}], None),
        ([{"type": "function", "function": {"name": 7}}], "auto"),
        ([{"type": "other", "function": {"name": "get_weather"}}], "auto"),
        ([{"type": "function", "function": "get_weather"}], "auto"),
        ([_WEATHER_TOOL, _WEATHER_TOOL], "required"),
    ],
)
async def test_invalid_tool_choice_returns_400_before_backend_dispatch(
    tools, tool_choice
):
    engine = StubBackend(text=TOOL_CALL_TEXT, finish_reason="stop")
    app = create_app(engines={"stub": engine})
    body = _chat_body("weather", tools=tools, tool_choice=tool_choice)
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "invalid_request"
    assert error["message"]
    assert engine.calls == 0


def _tool_response_contract(response, stream: bool):
    if not stream:
        choice = response.json()["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        return (
            choice["message"].get("content"),
            [call["function"]["name"] for call in calls],
            choice["finish_reason"],
        )

    chunks = [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    choices = [choice for chunk in chunks for choice in chunk["choices"]]
    return (
        "".join(choice["delta"].get("content") or "" for choice in choices) or None,
        [
            call["function"]["name"]
            for choice in choices
            for call in choice["delta"].get("tool_calls") or []
        ],
        next(
            choice["finish_reason"]
            for choice in reversed(choices)
            if choice["finish_reason"] is not None
        ),
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("upstream_finish_reason", "expected_finish_reason"),
    [("stop", "stop"), ("tool_calls", "stop"), ("length", "length")],
)
async def test_tool_choice_none_suppresses_calls_and_keeps_content(
    stream, upstream_finish_reason, expected_finish_reason
):
    engine = StubBackend(text=TOOL_CALL_TEXT, finish_reason=upstream_finish_reason)
    app = create_app(engines={"stub": engine})
    body = _chat_body(
        "weather", tools=[_WEATHER_TOOL], tool_choice="none", stream=stream
    )
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert _tool_response_contract(response, stream) == (
        TOOL_CALL_TEXT,
        [],
        expected_finish_reason,
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_choice", [None, "auto"])
async def test_auto_and_omitted_emit_only_declared_calls(stream, tool_choice):
    text = "".join(
        (
            '<tool_call>{"name":"undeclared","arguments":{}}</tool_call>',
            TOOL_CALL_TEXT,
        )
    )
    engine = StubBackend(text=text, finish_reason="stop")
    app = create_app(engines={"stub": engine})
    body = _chat_body("weather", tools=[_WEATHER_TOOL], stream=stream)
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert _tool_response_contract(response, stream) == (
        None,
        ["get_weather"],
        "tool_calls",
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_auto_suppresses_undeclared_model_function_names(stream):
    text = '<tool_call>{"name":"undeclared","arguments":{}}</tool_call>'
    engine = StubBackend(text=text, finish_reason="tool_calls")
    app = create_app(engines={"stub": engine})
    body = _chat_body("weather", tools=[_WEATHER_TOOL], stream=stream)
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert _tool_response_contract(response, stream) == (text, [], "stop")


@pytest.mark.parametrize("stream", [False, True])
async def test_tool_choice_required_emits_declared_calls(stream):
    engine = StubBackend(text=TOOL_CALL_TEXT, finish_reason="stop")
    app = create_app(engines={"stub": engine})
    body = _chat_body(
        "weather", tools=[_WEATHER_TOOL], tool_choice="required", stream=stream
    )
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert _tool_response_contract(response, stream) == (
        None,
        ["get_weather"],
        "tool_calls",
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_named_tool_choice_filters_multi_call_output(stream):
    text = "".join(
        (
            TOOL_CALL_TEXT,
            '<tool_call>{"name":"search","arguments":{"q":"rain"}}</tool_call>',
        )
    )
    search_tool = {
        "type": "function",
        "function": {"name": "search", "parameters": {"type": "object"}},
    }
    engine = StubBackend(text=text, finish_reason="stop")
    app = create_app(engines={"stub": engine})
    body = _chat_body(
        "weather",
        tools=[_WEATHER_TOOL, search_tool],
        tool_choice={"type": "function", "function": {"name": "search"}},
        stream=stream,
    )
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert _tool_response_contract(response, stream) == (
        None,
        ["search"],
        "tool_calls",
    )


@pytest.mark.parametrize(
    ("text", "tool_choice"),
    [
        ("no tool call", "required"),
        (
            '<tool_call>{"name":"other","arguments":{}}</tool_call>',
            {"type": "function", "function": {"name": "get_weather"}},
        ),
    ],
)
async def test_unsatisfied_tool_choice_is_controlled_upstream_failure(
    text, tool_choice
):
    other_tool = {
        "type": "function",
        "function": {"name": "other", "parameters": {"type": "object"}},
    }
    engine = StubBackend(text=text, finish_reason="stop")
    app = create_app(engines={"stub": engine})
    body = _chat_body(
        "weather", tools=[_WEATHER_TOOL, other_tool], tool_choice=tool_choice
    )
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"
    assert response.json()["error"]["code"] == "tool_choice_not_satisfied"
    assert engine.calls == 1


async def test_malformed_auto_tool_output_records_actual_backend_usage(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    text = "before <tool_call>[]</tool_call> after"
    engine = StubBackend(
        text=text,
        finish_reason="stop",
        usage=GenerationUsage(prompt_tokens=7, completion_tokens=3),
    )
    app = create_app(
        engines={"stub": engine},
        settings=ServerSettings(usage_ledger_path=str(ledger_path)),
    )
    body = _chat_body("weather", tools=[_WEATHER_TOOL], tool_choice="auto")
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["tool_calls"] is None
    assert UsageLedger(ledger_path).totals()["default"] == {
        "requests": 1,
        "prompt_tokens": 7,
        "completion_tokens": 3,
    }


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("text", "tool_choice"),
    [
        ("no tool call", "required"),
        (
            '<tool_call>{"name":"other","arguments":{}}</tool_call>',
            {"type": "function", "function": {"name": "get_weather"}},
        ),
    ],
)
async def test_unsatisfied_tool_choice_is_metered_once(
    tmp_path, stream, text, tool_choice
):
    ledger_path = tmp_path / "usage.jsonl"
    engine = StubBackend(
        text=text,
        finish_reason="stop",
        usage=GenerationUsage(prompt_tokens=11, completion_tokens=5),
    )
    app = create_app(
        engines={"stub": engine},
        settings=ServerSettings(usage_ledger_path=str(ledger_path)),
    )
    body = _chat_body(
        "weather",
        tools=[_WEATHER_TOOL],
        tool_choice=tool_choice,
        stream=stream,
    )
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "tool_choice_not_satisfied"
    assert engine.calls == 1
    assert UsageLedger(ledger_path).totals()["default"] == {
        "requests": 1,
        "prompt_tokens": 11,
        "completion_tokens": 5,
    }


async def test_named_tool_stream_includes_per_call_index():
    text = '<tool_call>{"name":"get_weather","arguments":{}}</tool_call>'
    app = create_app(engines={"stub": StubBackend(text=text, finish_reason="stop")})
    body = _chat_body(
        "weather",
        tools=[_WEATHER_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        stream=True,
    )
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    chunks = [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    calls = [
        call
        for chunk in chunks
        for choice in chunk["choices"]
        for call in choice["delta"].get("tool_calls") or []
    ]
    assert [call["index"] for call in calls] == [0]


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        "[]",
        "{}",
        '{"arguments": {}}',
        '{"name": 7, "arguments": {}}',
        '{"name": "", "arguments": {}}',
        '{"name": "get_weather", "arguments": []}',
        '{"name": "get_weather", "arguments": 7}',
        '{"name": "get_weather", "arguments": true}',
        pytest.param(
            '{"name": "get_weather", "arguments": ' + ("9" * 5000) + "}",
            id="oversized-json-integer",
        ),
        pytest.param(
            '{"name": "get_weather", "arguments": '
            + ("[" * 1100)
            + "0"
            + ("]" * 1100)
            + "}",
            id="excessive-json-nesting",
        ),
    ],
)
async def test_malformed_generated_tool_payload_stays_content(payload):
    text = f"before <tool_call>{payload}</tool_call> after"
    app = create_app(engines={"stub": StubBackend(text=text, finish_reason="stop")})
    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    ]
    body = _chat_body("weather", tools=tools)
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == text
    assert choice["message"]["tool_calls"] is None


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ('{"city": "Tokyo"}', '{"city": "Tokyo"}'),
        ('"{\\"city\\":\\"Tokyo\\"}"', '{"city":"Tokyo"}'),
    ],
)
async def test_generated_tool_arguments_are_serialized_once(arguments, expected):
    text = (
        '<tool_call>{"name": "get_weather", "arguments": '
        f"{arguments}}}</tool_call>"
    )
    app = create_app(engines={"stub": StubBackend(text=text, finish_reason="stop")})
    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    ]
    body = _chat_body("weather", tools=tools)
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    call = response.json()["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert call["function"]["arguments"] == expected


async def test_generated_tool_calls_keep_order_while_skipping_only_malformed_entries():
    text = "".join(
        (
            '<tool_call>{"name":"first","arguments":{}}</tool_call>',
            '<tool_call>{"name":"broken","arguments":[]}</tool_call>',
            '<tool_call>{"name":"second"}</tool_call>',
        )
    )
    app = create_app(engines={"stub": StubBackend(text=text, finish_reason="stop")})
    tools = [
        {"type": "function", "function": {"name": name, "parameters": {}}}
        for name in ("first", "broken", "second")
    ]
    body = _chat_body("tools", tools=tools)
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    calls = response.json()["choices"][0]["message"]["tool_calls"]
    assert [call["function"]["name"] for call in calls] == ["first", "second"]
    assert [call["function"]["arguments"] for call in calls] == ["{}", "{}"]


async def test_zero_token_prompt_array_is_validated_before_any_generation_starts():
    class ArrayValidationBackend(StubBackend):
        def __init__(self):
            super().__init__()
            self.validated: list[str] = []
            self.started: list[str] = []

        def validate_request(self, request):
            self.validated.append(request.prompt)
            if request.prompt == "invalid":
                raise ValueError("prompt must tokenize to at least one token")

        async def generate(self, request):
            self.started.append(request.prompt)
            return await super().generate(request)

    engine = ArrayValidationBackend()
    app = create_app(engines={"stub": engine})
    async with _client(app) as client:
        response = await client.post(
            "/v1/completions",
            json={"model": "stub", "prompt": ["valid", "invalid"]},
        )

    assert response.status_code == 400
    assert engine.validated == ["valid", "invalid"]
    assert engine.started == []


async def test_runtime_value_error_remains_an_upstream_error():
    app = create_app(engines={"stub": StubBackend(error=ValueError("runtime failure"))})
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "stub",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"


async def test_streaming_with_n_gt_1_emits_all_choice_indexes(app):
    async with _client(app) as client:
        full = (
            await client.post("/v1/chat/completions", json=_chat_body("hello", n=2))
        ).json()
        response = await client.post(
            "/v1/chat/completions", json=_chat_body("hello", n=2, stream=True)
        )
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    texts: dict[int, str] = {}
    for line in lines[:-1]:
        chunk = json.loads(line[len("data: "):])
        for choice in chunk["choices"]:
            texts[choice["index"]] = texts.get(choice["index"], "") + (
                choice["delta"].get("content") or ""
            )
    expected = {c["index"]: c["message"]["content"] for c in full["choices"]}
    assert texts == expected
    assert set(texts) == {0, 1}


async def test_streaming_with_tools_signals_tool_calls(app):
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_body("what is the weather?", tools=tools, stream=True),
        )
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    chunks = [json.loads(line[len("data: "):]) for line in lines[:-1]]
    finish_reasons = [
        choice["finish_reason"] for chunk in chunks for choice in chunk["choices"]
    ]
    assert "tool_calls" in finish_reasons
    tool_deltas = [
        choice["delta"].get("tool_calls")
        for chunk in chunks
        for choice in chunk["choices"]
        if choice["delta"].get("tool_calls")
    ]
    assert tool_deltas[0][0]["function"]["name"] == "get_weather"


async def test_backend_finish_reason_passes_through():
    app = create_app(engines={"stub": StubBackend(finish_reason="length")})
    async with _client(app) as client:
        body = _chat_body("hi")
        body["model"] = "stub"
        response = await client.post("/v1/chat/completions", json=body)
    assert response.json()["choices"][0]["finish_reason"] == "length"


async def test_backend_error_returns_openai_error_envelope():
    app = create_app(
        engines={"stub": StubBackend(error=RuntimeError("secret://host:5432 key missing"))}
    )
    async with _client(app) as client:
        body = _chat_body("hi")
        body["model"] = "stub"
        response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 502
    error = response.json()["error"]
    assert error["type"] == "upstream_error"
    # M3: the backend's raw message (which may carry secrets/hosts) must NOT leak
    assert "secret://host:5432" not in error["message"]
    assert "RuntimeError" in error["message"]  # only the class name is disclosed


async def test_response_format_passes_through_to_engine():
    class CapturingBackend(StubBackend):
        def __init__(self):
            super().__init__()
            self.seen = None

        async def generate(self, request):
            self.seen = request
            return await super().generate(request)

    engine = CapturingBackend()
    app = create_app(engines={"stub": engine})
    body = _chat_body("give me json")
    body["model"] = "stub"
    body["response_format"] = {"type": "json_object"}
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert engine.seen.sampling_params.extra_args["response_format"] == {
        "type": "json_object"
    }
