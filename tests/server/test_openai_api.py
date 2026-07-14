import asyncio
import json
import logging

import httpx
import pytest

from kairyu.engine.backend import GenerationResult, GenerationUsage
from kairyu.engine.kairyu_backend import KairyuBackend
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.metering import resolve_usage_counts
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import UsageLedger
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.outputs import CompletionOutput

TOOL_CALL_TEXT = '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'


class _EmptyTokenizer:
    eos_token_id = None

    def encode(self, text: str) -> tuple[int, ...]:
        return ()

    def decode(self, token_ids) -> str:
        return ""

    def vocab(self) -> list[str]:
        return []


def _client(app, *, raise_app_exceptions: bool = True) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(
        app=app, raise_app_exceptions=raise_app_exceptions
    )
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
        self.requests = []

    async def generate(self, request):
        from kairyu.engine.backend import GenerationResult
        from kairyu.outputs import CompletionOutput

        self.calls += 1
        self.requests.append(request)
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


class CompletionUsageBackend:
    """Legacy-completion backend with per-prompt optional usage."""

    def __init__(self, reported_usage):
        self._reported_usage = reported_usage

    async def generate(self, request):
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0,
                    text=f"answer for {request.prompt}",
                    token_ids=(),
                    finish_reason="stop",
                ),
            ),
            usage=self._reported_usage.get(request.prompt),
        )

    async def stream(self, request):
        yield await self.generate(request)

    async def shutdown(self):
        return None


class MeteringStreamBackend:
    """Cumulative two-part stream with controllable finalization behavior."""

    def __init__(self, case):
        self.case = case
        self.requests = []
        self.cancelled = False

    async def generate(self, request):
        raise AssertionError("streaming test reached generate")

    async def stream(self, request):
        self.requests.append(request)
        yield GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0,
                    text="partial output",
                    token_ids=(101, 102),
                    finish_reason=None,
                ),
            ),
            finished=False,
            usage=(
                GenerationUsage(prompt_tokens=17, completion_tokens=9)
                if self.case == "reported-then-none"
                else None
            ),
        )
        if self.case == "client-close":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if self.case == "upstream-error":
            raise RuntimeError("stream failed after partial output")
        yield GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0,
                    text="partial output final",
                    token_ids=(101, 102, 103),
                    finish_reason="stop",
                ),
            ),
            usage=(
                GenerationUsage(prompt_tokens=17, completion_tokens=9)
                if self.case == "reported"
                else None
            ),
        )

    async def shutdown(self):
        return None


def _metered_stream_app(kind, backend, ledger_path):
    settings = ServerSettings(usage_ledger_path=str(ledger_path))
    if kind == "orchestrated-chat":
        app = create_app(
            engines={},
            orchestrators={"auto": Orchestrator({"tier1": backend})},
            settings=settings,
        )
        return app, "/v1/chat/completions", {
            "model": "auto",
            "messages": [{"role": "user", "content": "metering prompt"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    app = create_app(engines={"stream": backend}, settings=settings)
    if kind == "engine-chat":
        return app, "/v1/chat/completions", {
            "model": "stream",
            "messages": [{"role": "user", "content": "metering prompt"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    return app, "/v1/completions", {
        "model": "stream",
        "prompt": "metering prompt",
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def _disconnect_after_partial(app, path, body):
    """Drive the ASGI endpoint until one partial body, then disconnect."""
    request_sent = False
    disconnect = asyncio.Event()
    response_messages = []
    encoded = json.dumps(body).encode()

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": encoded, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        response_messages.append(message)
        if (
            message["type"] == "http.response.body"
            and b"partial output" in message.get("body", b"")
        ):
            disconnect.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"test"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(encoded)).encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=2)
    assert disconnect.is_set()
    return response_messages


class DispatchCountingOrchestrator:
    """Failing spy: invalid requests must never reach either dispatch seam."""

    def __init__(self):
        self.run_calls = 0
        self.run_chat_calls = 0

    async def run(self, prompt):
        self.run_calls += 1
        raise AssertionError("invalid request reached orchestrator.run")

    async def run_chat(self, prompt, *, stream):
        self.run_chat_calls += 1
        raise AssertionError("invalid request reached orchestrator.run_chat")


_WEATHER_TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object"}},
}


_OUTPUT_LIMIT_CASES = [
    pytest.param(
        "/v1/chat/completions",
        "max_tokens",
        {"model": "stub", "messages": [{"role": "user", "content": "hello"}]},
        id="chat-max-tokens",
    ),
    pytest.param(
        "/v1/chat/completions",
        "max_completion_tokens",
        {"model": "stub", "messages": [{"role": "user", "content": "hello"}]},
        id="chat-max-completion-tokens",
    ),
    pytest.param(
        "/v1/completions",
        "max_tokens",
        {"model": "stub", "prompt": "hello"},
        id="legacy-completions-max-tokens",
    ),
    pytest.param(
        "/v1/responses",
        "max_output_tokens",
        {"model": "stub", "input": "hello"},
        id="responses-max-output-tokens",
    ),
]


@pytest.mark.parametrize(("path", "field", "body"), _OUTPUT_LIMIT_CASES)
@pytest.mark.parametrize("limit", [0, -1])
async def test_nonpositive_output_token_limits_are_predispatch_400(
    path, field, body, limit
):
    engine = StubBackend(text="done", finish_reason="stop")
    app = create_app(engines={"stub": engine})

    async with _client(app, raise_app_exceptions=False) as client:
        response = await client.post(path, json={**body, field: limit})

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["code"] == "invalid_request"
    assert engine.calls == 0


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("limit", [0, -1])
@pytest.mark.parametrize("stream", [False, True])
async def test_nonpositive_auto_output_limits_are_predispatch_400(
    field, limit, stream
):
    orchestrator = DispatchCountingOrchestrator()
    app = create_app(engines={}, orchestrators={"auto": orchestrator})
    body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
        field: limit,
    }

    async with _client(app, raise_app_exceptions=False) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": f"max_tokens must be >= 1, got {limit}",
        "type": "invalid_request_error",
        "code": "invalid_request",
    }
    assert orchestrator.run_calls == 0
    assert orchestrator.run_chat_calls == 0


@pytest.mark.parametrize(("path", "field", "body"), _OUTPUT_LIMIT_CASES)
async def test_positive_output_token_limits_reach_backend(path, field, body):
    engine = StubBackend(text="done", finish_reason="stop")
    app = create_app(engines={"stub": engine})

    async with _client(app) as client:
        response = await client.post(path, json={**body, field: 7})

    assert response.status_code == 200
    assert engine.calls == 1
    assert engine.requests[0].sampling_params.max_tokens == 7


async def test_chat_max_tokens_precedes_modern_alias_when_both_are_present():
    engine = StubBackend(text="done", finish_reason="stop")
    app = create_app(engines={"stub": engine})
    body = {
        "model": "stub",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 5,
        "max_completion_tokens": 9,
    }

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert engine.requests[0].sampling_params.max_tokens == 5


async def test_responses_default_output_token_limit_is_1024():
    engine = StubBackend(text="done", finish_reason="stop")
    app = create_app(engines={"stub": engine})

    async with _client(app) as client:
        response = await client.post(
            "/v1/responses", json={"model": "stub", "input": "hello"}
        )

    assert response.status_code == 200
    assert engine.requests[0].sampling_params.max_tokens == 1024


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


async def test_sync_usage_none_records_wire_derived_counts(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    engine = StubBackend(text="derived completion words", finish_reason="stop")
    app = create_app(
        engines={"stub": engine},
        settings=ServerSettings(usage_ledger_path=str(ledger_path)),
    )

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "stub",
                "messages": [{"role": "user", "content": "rendered prompt words"}],
            },
        )

    assert response.status_code == 200
    wire_usage = response.json()["usage"]
    assert UsageLedger(ledger_path).totals()["default"] == {
        "requests": 1,
        "prompt_tokens": wire_usage["prompt_tokens"],
        "completion_tokens": wire_usage["completion_tokens"],
    }


async def test_sync_orchestrator_zero_usage_derives_wire_and_ledger(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    engine = StubBackend(text="derived orchestrated answer", finish_reason="stop")
    app = create_app(
        engines={},
        orchestrators={"auto": Orchestrator({"tier1": engine})},
        settings=ServerSettings(usage_ledger_path=str(ledger_path)),
    )

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "brief prompt"}],
            },
        )

    assert response.status_code == 200
    expected = resolve_usage_counts(
        None,
        prompt=engine.requests[0].prompt,
        completions=(
            CompletionOutput(
                index=0,
                text="derived orchestrated answer",
                token_ids=(),
            ),
        ),
    )
    wire_usage = response.json()["usage"]
    assert wire_usage["prompt_tokens"] == expected[0]
    assert wire_usage["completion_tokens"] == expected[1]
    assert UsageLedger(ledger_path).totals()["default"] == {
        "requests": 1,
        "prompt_tokens": expected[0],
        "completion_tokens": expected[1],
    }


@pytest.mark.parametrize(
    ("prompt", "reported_usage", "expected"),
    [
        pytest.param("single prompt", {}, (2, 4), id="single-missing"),
        pytest.param(
            ["first prompt", "second"],
            {},
            (3, 7),
            id="array-missing",
        ),
        pytest.param(
            ["derived prompt", "reported prompt"],
            {
                "reported prompt": GenerationUsage(
                    prompt_tokens=17,
                    completion_tokens=9,
                )
            },
            (19, 13),
            id="array-mixed",
        ),
    ],
)
async def test_sync_completions_derives_each_missing_usage(
    tmp_path, prompt, reported_usage, expected
):
    ledger_path = tmp_path / "usage.jsonl"
    app = create_app(
        engines={"stub": CompletionUsageBackend(reported_usage)},
        settings=ServerSettings(usage_ledger_path=str(ledger_path)),
    )

    async with _client(app) as client:
        response = await client.post(
            "/v1/completions",
            json={"model": "stub", "prompt": prompt},
        )

    assert response.status_code == 200
    wire_usage = response.json()["usage"]
    assert wire_usage["prompt_tokens"] == expected[0]
    assert wire_usage["completion_tokens"] == expected[1]
    assert wire_usage["total_tokens"] == sum(expected)
    assert UsageLedger(ledger_path).totals()["default"] == {
        "requests": 1,
        "prompt_tokens": expected[0],
        "completion_tokens": expected[1],
    }


@pytest.mark.parametrize(
    "kind", ["engine-chat", "orchestrated-chat", "legacy-completions"]
)
@pytest.mark.parametrize(
    "case", ["reported", "usage-none", "client-close", "upstream-error"]
)
async def test_every_dispatched_stream_is_metered_exactly_once(
    tmp_path, kind, case
):
    ledger_path = tmp_path / "usage.jsonl"
    backend = MeteringStreamBackend(case)
    app, path, body = _metered_stream_app(kind, backend, ledger_path)

    if case == "client-close":
        await _disconnect_after_partial(app, path, body)
        assert backend.cancelled
        response_text = ""
    else:
        async with _client(app) as client:
            response = await client.post(path, json=body)
        assert response.status_code == 200
        response_text = response.text
        assert "data: [DONE]" in response_text

    assert len(backend.requests) == 1
    completion_text = (
        "partial output final" if case in {"reported", "usage-none"}
        else "partial output"
    )
    expected = (
        (17, 9)
        if case == "reported"
        else resolve_usage_counts(
            None,
            prompt=backend.requests[0].prompt,
            completions=(
                CompletionOutput(index=0, text=completion_text, token_ids=()),
            ),
        )
    )
    assert UsageLedger(ledger_path).totals()["default"] == {
        "requests": 1,
        "prompt_tokens": expected[0],
        "completion_tokens": expected[1],
    }

    if case in {"reported", "usage-none"}:
        usage_chunks = [
            json.loads(line[len("data: "):])["usage"]
            for line in response_text.splitlines()
            if line.startswith("data: {")
            and json.loads(line[len("data: "):]).get("usage") is not None
        ]
        assert usage_chunks[-1]["prompt_tokens"] == expected[0]
        assert usage_chunks[-1]["completion_tokens"] == expected[1]
    elif case == "upstream-error":
        error_chunks = [
            json.loads(line[len("data: "):])["error"]
            for line in response_text.splitlines()
            if line.startswith("data: {")
            and "error" in json.loads(line[len("data: "):])
        ]
        assert len(error_chunks) == 1
        assert "RuntimeError" in error_chunks[0]["message"]


@pytest.mark.parametrize(
    "kind", ["engine-chat", "orchestrated-chat", "legacy-completions"]
)
async def test_stream_retains_reported_usage_when_final_partial_has_none(
    tmp_path, kind
):
    ledger_path = tmp_path / "usage.jsonl"
    backend = MeteringStreamBackend("reported-then-none")
    app, path, body = _metered_stream_app(kind, backend, ledger_path)

    async with _client(app) as client:
        response = await client.post(path, json=body)

    assert response.status_code == 200
    usage_chunks = [
        json.loads(line[len("data: "):])["usage"]
        for line in response.text.splitlines()
        if line.startswith("data: {")
        and json.loads(line[len("data: "):]).get("usage") is not None
    ]
    assert usage_chunks[-1]["prompt_tokens"] == 17
    assert usage_chunks[-1]["completion_tokens"] == 9
    assert UsageLedger(ledger_path).totals()["default"] == {
        "requests": 1,
        "prompt_tokens": 17,
        "completion_tokens": 9,
    }


@pytest.mark.parametrize(
    "kind", ["engine-chat", "orchestrated-chat", "legacy-completions"]
)
async def test_closing_unstarted_stream_does_not_dispatch_or_meter(
    tmp_path, kind
):
    from starlette.requests import Request

    from kairyu.engine.backend import GenerationRequest
    from kairyu.entrypoints.server.app import (
        _stream_completions,
        _stream_engine,
        _stream_orchestrator,
    )
    from kairyu.entrypoints.server.protocol import (
        ChatCompletionRequest,
        CompletionRequest,
    )
    from kairyu.sampling_params import SamplingParams

    ledger_path = tmp_path / "usage.jsonl"
    backend = MeteringStreamBackend("client-close")
    app, _, _ = _metered_stream_app(kind, backend, ledger_path)
    http_request = Request({"type": "http", "app": app, "state": {}})
    generation_request = GenerationRequest(
        request_id="not-started",
        prompt="unstarted prompt",
        sampling_params=SamplingParams(),
    )
    if kind == "engine-chat":
        stream = _stream_engine(
            backend,
            generation_request,
            "stream",
            ChatCompletionRequest(
                model="stream",
                messages=[{"role": "user", "content": "unstarted"}],
                stream=True,
            ),
            http_request,
        )
    elif kind == "orchestrated-chat":
        stream = _stream_orchestrator(
            Orchestrator({"tier1": backend}),
            generation_request.prompt,
            ChatCompletionRequest(
                model="auto",
                messages=[{"role": "user", "content": "unstarted"}],
                stream=True,
            ),
            False,
            False,
            http_request,
        )
    else:
        stream = _stream_completions(
            backend,
            generation_request,
            CompletionRequest(model="stream", prompt="unstarted", stream=True),
            http_request,
        )

    await stream.aclose()

    assert backend.requests == []
    assert not ledger_path.exists()


async def test_generate_fully_tool_stream_keeps_single_sync_metering_owner(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    engine = StubBackend(
        text=TOOL_CALL_TEXT,
        finish_reason="tool_calls",
        usage=GenerationUsage(prompt_tokens=13, completion_tokens=8),
    )
    app = create_app(
        engines={"stub": engine},
        settings=ServerSettings(usage_ledger_path=str(ledger_path)),
    )
    body = _chat_body(
        "weather",
        tools=[_WEATHER_TOOL],
        tool_choice="auto",
        stream=True,
    )
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert UsageLedger(ledger_path).totals()["default"] == {
        "requests": 1,
        "prompt_tokens": 13,
        "completion_tokens": 8,
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


async def test_backend_error_returns_openai_error_envelope(caplog):
    app = create_app(
        engines={"stub": StubBackend(error=RuntimeError("secret://host:5432 key missing"))}
    )
    with caplog.at_level(logging.ERROR, logger="kairyu.entrypoints.server.app"):
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
    # ...but the full traceback IS logged server-side (observability): a replica
    # backend failure must be diagnosable from the logs even though the client
    # only sees the exception class name.
    assert any(
        record.exc_info and record.levelno == logging.ERROR
        for record in caplog.records
    ), "expected the backend traceback to be logged server-side"


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
