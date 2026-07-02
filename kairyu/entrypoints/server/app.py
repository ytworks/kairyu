"""OpenAI-compatible FastAPI app: /v1/models, /v1/chat/completions (+SSE, tools).

Model name ``kairyu-auto`` routes the request through the Orchestrator behind
the same endpoint (design doc D6).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Mapping

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from kairyu.engine.backend import EngineBackend, GenerationRequest
from kairyu.entrypoints.chat_template import render_chat
from kairyu.entrypoints.server.health import add_health_routes
from kairyu.entrypoints.server.metrics import ServerMetrics
from kairyu.entrypoints.server.middleware import (
    AccessLogMiddleware,
    AuthMiddleware,
    ConcurrencyLimitMiddleware,
    MetricsMiddleware,
)
from kairyu.entrypoints.server.protocol import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChunkChoice,
    ChunkDelta,
    FunctionCall,
    ModelCard,
    ModelList,
    ResponseMessage,
    ToolCall,
    Usage,
)
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.orchestration.replica import ReplicaPool
from kairyu.sampling_params import SamplingParams

AUTO_MODEL = "kairyu-auto"
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _sampling_params_from(request: ChatCompletionRequest) -> SamplingParams:
    extra_args = (
        {"response_format": request.response_format} if request.response_format else {}
    )
    return SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        n=request.n,
        max_tokens=request.max_tokens,
        stop=request.stop,
        seed=request.seed,
        extra_args=extra_args,
    )


def _approx_tokens(text: str) -> int:
    return len(text.split())


def _parse_tool_calls(text: str) -> list[ToolCall]:
    calls = []
    for match in _TOOL_CALL_PATTERN.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        arguments = payload.get("arguments", {})
        calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                function=FunctionCall(
                    name=payload.get("name", ""),
                    arguments=(
                        arguments if isinstance(arguments, str) else json.dumps(arguments)
                    ),
                ),
            )
        )
    return calls


def _model_not_found(model: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": f"model {model!r} not found",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        },
    )


def _upstream_error(error: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={
            "error": {
                "message": str(error),
                "type": "upstream_error",
                "code": "backend_error",
            }
        },
    )


def _build_choice(
    index: int, text: str, request: ChatCompletionRequest, finish_reason: str | None
) -> Choice:
    tool_calls = _parse_tool_calls(text) if request.tools else []
    if tool_calls:
        message = ResponseMessage(content=None, tool_calls=tool_calls)
        return Choice(index=index, message=message, finish_reason="tool_calls")
    return Choice(
        index=index, message=ResponseMessage(content=text), finish_reason=finish_reason or "stop"
    )


def _completion_response(
    request: ChatCompletionRequest, prompt: str, texts: list[tuple[str, str | None]]
) -> ChatCompletionResponse:
    choices = [
        _build_choice(i, text, request, finish_reason)
        for i, (text, finish_reason) in enumerate(texts)
    ]
    completion_tokens = sum(_approx_tokens(text) for text, _ in texts)
    prompt_tokens = _approx_tokens(prompt)
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:16]}",
        created=int(time.time()),
        model=request.model,
        choices=choices,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _sse_chunk(
    response_id: str, created: int, model: str, index: int, delta: ChunkDelta,
    finish_reason: str | None = None,
) -> str:
    payload = ChatCompletionChunk(
        id=response_id,
        created=created,
        model=model,
        choices=[ChunkChoice(index=index, delta=delta, finish_reason=finish_reason)],
    )
    return f"data: {payload.model_dump_json()}\n\n"


async def _stream_engine(
    engine: EngineBackend, generation_request: GenerationRequest, model: str
) -> AsyncIterator[str]:
    response_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    sent: dict[int, int] = {}
    last = None
    try:
        async for partial in engine.stream(generation_request):
            last = partial
            for completion in partial.completions:
                delta_text = completion.text[sent.get(completion.index, 0):]
                if not delta_text and not partial.finished:
                    continue
                is_first = completion.index not in sent
                sent[completion.index] = len(completion.text)
                yield _sse_chunk(
                    response_id, created, model, completion.index,
                    ChunkDelta(role="assistant" if is_first else None, content=delta_text),
                )
    except Exception as error:  # surface backend failures inside the SSE stream
        payload = {"error": {"message": str(error), "type": "upstream_error"}}
        yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"
        return
    for completion in last.completions if last else ():
        yield _sse_chunk(
            response_id, created, model, completion.index, ChunkDelta(),
            finish_reason=completion.finish_reason or "stop",
        )
    yield "data: [DONE]\n\n"


async def _stream_choices(choices: list[Choice], model: str) -> AsyncIterator[str]:
    """Stream already-final choices (orchestrated or tool-call responses)."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    for choice in choices:
        yield _sse_chunk(
            response_id, created, model, choice.index,
            ChunkDelta(
                role="assistant",
                content=choice.message.content,
                tool_calls=choice.message.tool_calls,
            ),
        )
        yield _sse_chunk(
            response_id, created, model, choice.index, ChunkDelta(),
            finish_reason=choice.finish_reason,
        )
    yield "data: [DONE]\n\n"


def create_app(
    engines: Mapping[str, EngineBackend],
    orchestrator: Orchestrator | None = None,
    settings: ServerSettings | None = None,
) -> FastAPI:
    settings = settings or ServerSettings()
    app = FastAPI(title="kairyu", version="0.1.0")
    served_engines = dict(engines)

    metrics = ServerMetrics() if settings.metrics else None
    app.state.metrics = metrics
    if metrics is not None:
        for name, engine in served_engines.items():
            if isinstance(engine, ReplicaPool):
                metrics.track_pool(name, engine)
    add_health_routes(app, served_engines, metrics)

    # add_middleware prepends, so add innermost first: metrics -> concurrency
    # guard -> auth -> access log (outermost).
    if metrics is not None:
        app.add_middleware(MetricsMiddleware, metrics=metrics)
    if settings.max_concurrency is not None:
        app.add_middleware(ConcurrencyLimitMiddleware, limit=settings.max_concurrency)
    api_keys = settings.resolve_api_keys()
    if api_keys:
        app.add_middleware(
            AuthMiddleware, api_keys=api_keys, protect_metrics=settings.protect_metrics
        )
    if settings.access_log:
        app.add_middleware(AccessLogMiddleware)

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        names = list(served_engines)
        if orchestrator is not None:
            names.append(AUTO_MODEL)
        return ModelList(data=[ModelCard(id=name) for name in names])

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, http_request: Request):
        http_request.state.model = request.model  # label for the metrics middleware
        prompt = render_chat([message.model_dump() for message in request.messages])
        if request.model == AUTO_MODEL and orchestrator is not None:
            try:
                result = await orchestrator.run(prompt)
            except Exception as error:
                return _upstream_error(error)
            texts = [(result.text, "stop")]
            if request.stream:
                response = _completion_response(request, prompt, texts)
                return StreamingResponse(
                    _stream_choices(response.choices, request.model),
                    media_type="text/event-stream",
                )
            return _completion_response(request, prompt, texts)

        engine = served_engines.get(request.model)
        if engine is None:
            return _model_not_found(request.model)
        generation_request = GenerationRequest(
            request_id=f"http-{uuid.uuid4().hex[:12]}",
            prompt=prompt,
            sampling_params=_sampling_params_from(request),
        )
        if request.stream and not request.tools:
            return StreamingResponse(
                _stream_engine(engine, generation_request, request.model),
                media_type="text/event-stream",
            )
        try:
            result = await engine.generate(generation_request)
        except Exception as error:
            return _upstream_error(error)
        texts = [
            (completion.text, completion.finish_reason) for completion in result.completions
        ]
        response = _completion_response(request, prompt, texts)
        if request.stream:
            # Tool calling + streaming: generate fully, then emit structured chunks so
            # tool_calls and finish_reason stay correct.
            return StreamingResponse(
                _stream_choices(response.choices, request.model),
                media_type="text/event-stream",
            )
        return response

    return app
