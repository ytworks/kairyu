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

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from kairyu.engine.backend import EngineBackend, GenerationRequest
from kairyu.entrypoints.chat_template import render_chat
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
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.sampling_params import SamplingParams

AUTO_MODEL = "kairyu-auto"
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _sampling_params_from(request: ChatCompletionRequest) -> SamplingParams:
    return SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        n=request.n,
        max_tokens=request.max_tokens,
        stop=request.stop,
        seed=request.seed,
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


def _build_choice(index: int, text: str, request: ChatCompletionRequest) -> Choice:
    tool_calls = _parse_tool_calls(text) if request.tools else []
    if tool_calls:
        message = ResponseMessage(content=None, tool_calls=tool_calls)
        return Choice(index=index, message=message, finish_reason="tool_calls")
    return Choice(index=index, message=ResponseMessage(content=text), finish_reason="stop")


def _completion_response(
    request: ChatCompletionRequest, prompt: str, texts: list[str]
) -> ChatCompletionResponse:
    choices = [_build_choice(i, text, request) for i, text in enumerate(texts)]
    completion_tokens = sum(_approx_tokens(text) for text in texts)
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


async def _stream_engine(
    engine: EngineBackend, generation_request: GenerationRequest, model: str
) -> AsyncIterator[str]:
    response_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())

    def chunk(delta: ChunkDelta, finish_reason: str | None = None) -> str:
        payload = ChatCompletionChunk(
            id=response_id,
            created=created,
            model=model,
            choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
        )
        return f"data: {payload.model_dump_json()}\n\n"

    sent = 0
    first = True
    async for partial in engine.stream(generation_request):
        text = partial.completions[0].text
        delta_text = text[sent:]
        sent = len(text)
        if not delta_text and not partial.finished:
            continue
        yield chunk(ChunkDelta(role="assistant" if first else None, content=delta_text))
        first = False
    yield chunk(ChunkDelta(), finish_reason="stop")
    yield "data: [DONE]\n\n"


def create_app(
    engines: Mapping[str, EngineBackend],
    orchestrator: Orchestrator | None = None,
) -> FastAPI:
    app = FastAPI(title="kairyu", version="0.1.0")
    served_engines = dict(engines)

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        names = list(served_engines)
        if orchestrator is not None:
            names.append(AUTO_MODEL)
        return ModelList(data=[ModelCard(id=name) for name in names])

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        prompt = render_chat([message.model_dump() for message in request.messages])
        if request.model == AUTO_MODEL and orchestrator is not None:
            result = await orchestrator.run(prompt)
            texts = [result.text]
            if request.stream:
                return StreamingResponse(
                    _stream_text(texts[0], request.model), media_type="text/event-stream"
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
        if request.stream:
            return StreamingResponse(
                _stream_engine(engine, generation_request, request.model),
                media_type="text/event-stream",
            )
        result = await engine.generate(generation_request)
        texts = [completion.text for completion in result.completions]
        return _completion_response(request, prompt, texts)

    return app


async def _stream_text(text: str, model: str) -> AsyncIterator[str]:
    """Stream an already-final text as a single-delta SSE sequence (orchestrated path)."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    head = ChatCompletionChunk(
        id=response_id,
        created=created,
        model=model,
        choices=[ChunkChoice(index=0, delta=ChunkDelta(role="assistant", content=text))],
    )
    tail = ChatCompletionChunk(
        id=response_id,
        created=created,
        model=model,
        choices=[ChunkChoice(index=0, delta=ChunkDelta(), finish_reason="stop")],
    )
    yield f"data: {head.model_dump_json()}\n\n"
    yield f"data: {tail.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
