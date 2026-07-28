"""OpenAI-compatible FastAPI app: /v1/models, /v1/chat/completions (+SSE, tools).

Model name ``kairyu-auto`` routes the request through the Orchestrator behind
the same endpoint (design doc D6).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from kairyu.engine.backend import (
    CacheHint,
    EngineBackend,
    GenerationRequest,
    GenerationUsage,
)
from kairyu.entrypoints.chat_template import ChatTemplate
from kairyu.entrypoints.server.chat_service import (
    ChatRequestError,
    _logprob_entries,
    _wire_usage,
    completion_response,
    execute_chat,
    sampling_params_from,
    tool_choice_is_satisfied,
    validate_chat_input,
    validate_chat_request,
)
from kairyu.entrypoints.server.chat_service import (
    render_prompt as render_prompt,
)
from kairyu.entrypoints.server.errors import (
    invalid_request,
    model_not_found,
    sanitize_backend_error,
    upstream_error,
)
from kairyu.entrypoints.server.health import add_health_routes
from kairyu.entrypoints.server.metering import (
    StreamUsageOwner,
    record_state_usage,
    resolve_cached_tokens,
    resolve_usage_counts,
    stream_usage_owner_from_state,
)
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
    ChoiceLogprobs,
    ChunkChoice,
    ChunkDelta,
    ChunkToolCall,
    CompletionChoice,
    CompletionChunk,
    CompletionLogprobs,
    CompletionRequest,
    CompletionResponse,
    ModelCard,
    ModelList,
    PromptTokensDetails,
    RouteDecisionPayload,
    RoutePreviewRequest,
    RoutePreviewResponse,
    RoutingResponse,
    Usage,
)
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.orchestration.orchestrator import (
    Orchestrator,
    OrchestratorExecutionError,
    PreviewNotSupportedError,
)
from kairyu.orchestration.replica import ReplicaPool
from kairyu.orchestration.request import OrchestrationRequest
from kairyu.outputs import CompletionOutput
from kairyu.pricing import InvoiceExportError, PriceSheet, export_invoice_csv
from kairyu.sampling_params import SamplingParams

if TYPE_CHECKING:
    from kairyu.entrypoints.server.extra_routes import EmbeddingBackend

logger = logging.getLogger(__name__)

AUTO_MODEL = "kairyu-auto"
_KAIRYU_CHAT_EXTENSION_FIELDS = {
    "kairyu_trace",
    "kairyu_trace_v2",
    "kairyu_route",
}


def _route_payload(decision) -> RouteDecisionPayload:
    return RouteDecisionPayload(
        target=decision.target,
        confidence=decision.confidence,
        reason=decision.reason,
        features=decision.features.as_dict(),
    )


def _chat_response_payload(response: ChatCompletionResponse) -> dict:
    """Preserve the OpenAI wire shape while omitting unset Kairyu extensions."""
    return response.model_dump(
        mode="json",
        exclude=_KAIRYU_CHAT_EXTENSION_FIELDS,
    )


def _orchestration_wire_usage(
    prompt: str,
    completions: Sequence[CompletionOutput],
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    approximate_missing_standard_usage: bool = False,
) -> Usage:
    """Build OpenAI usage plus exact cumulative AUTO internal-call totals."""

    reported = (
        None
        if approximate_missing_standard_usage and prompt_tokens == 0 and completion_tokens == 0
        else GenerationUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )
    )
    usage = _wire_usage(prompt, completions, reported)
    usage.orchestration_input_tokens = prompt_tokens
    usage.orchestration_output_tokens = completion_tokens
    return usage


def _with_usage_ledger_cleanup(lifespan):
    """Make the app-created ledger the outermost lifespan-owned resource."""

    @contextlib.asynccontextmanager
    async def wrapped(app: FastAPI):
        try:
            if lifespan is None:
                yield
            else:
                async with lifespan(app):
                    yield
        finally:
            ledger = getattr(app.state, "usage_ledger", None)
            if ledger is not None:
                ledger.close()

    return wrapped


def _validate_generation_request(
    engine: EngineBackend, request: GenerationRequest
) -> JSONResponse | None:
    validate = getattr(engine, "validate_request", None)
    if validate is None:
        return None
    try:
        validate(request)
    except ValueError as error:
        return invalid_request(str(error))
    return None


def _stream_usage_owner(http_request: Request, model: str, prompt: str) -> StreamUsageOwner:
    tenant = getattr(http_request.state, "tenant", None) or "default"
    return stream_usage_owner_from_state(
        http_request.app.state,
        tenant=tenant,
        model=model,
        prompt=prompt,
    )


def _sse_chunk(
    response_id: str,
    created: int,
    model: str,
    index: int,
    delta: ChunkDelta,
    finish_reason: str | None = None,
    include_usage: bool = False,
    usage: Usage | None = None,
    logprobs: ChoiceLogprobs | None = None,
) -> str:
    payload = ChatCompletionChunk(
        id=response_id,
        created=created,
        model=model,
        choices=[
            ChunkChoice(index=index, delta=delta, finish_reason=finish_reason, logprobs=logprobs)
        ],
        usage=usage,
    )
    # OpenAI contract: usage key omitted unless include_usage; then explicit
    # null on non-final chunks, populated on the final choices-less chunk
    exclude = set(_KAIRYU_CHAT_EXTENSION_FIELDS)
    if not include_usage:
        exclude.add("usage")
    return f"data: {payload.model_dump_json(exclude=exclude)}\n\n"


def _usage_chunk(response_id: str, created: int, model: str, usage: Usage) -> str:
    payload = ChatCompletionChunk(
        id=response_id, created=created, model=model, choices=[], usage=usage
    )
    return f"data: {payload.model_dump_json(exclude=_KAIRYU_CHAT_EXTENSION_FIELDS)}\n\n"


def _orchestrator_metadata_chunk(
    response_id: str,
    created: int,
    model: str,
    *,
    usage: Usage | None,
    result,
    include_usage: bool,
    want_trace: bool,
) -> str | None:
    """One terminal AUTO chunk carrying requested usage and/or trace data."""

    if not include_usage and not want_trace:
        return None
    trace = (
        result.structured_trace.as_dict(request_id=response_id)
        if want_trace and result.structured_trace is not None
        else None
    )
    payload = ChatCompletionChunk(
        id=response_id,
        created=created,
        model=model,
        choices=[],
        usage=usage if include_usage else None,
        kairyu_trace=list(result.trace) if want_trace else None,
        kairyu_trace_v2=trace,
        kairyu_route=_route_payload(result.route) if want_trace else None,
    )
    exclude: set[str] = set()
    if not include_usage:
        exclude.add("usage")
    if not want_trace:
        exclude.update(_KAIRYU_CHAT_EXTENSION_FIELDS)
    elif trace is None:
        exclude.add("kairyu_trace_v2")
    return f"data: {payload.model_dump_json(exclude=exclude)}\n\n"


async def _stream_engine(
    engine: EngineBackend,
    generation_request: GenerationRequest,
    model: str,
    request: ChatCompletionRequest,
    http_request: Request,
) -> AsyncIterator[str]:
    response_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    include_usage = bool(request.stream_options and request.stream_options.include_usage)
    sent: dict[int, int] = {}
    logprobs_sent: dict[int, int] = {}
    last = None
    owner = _stream_usage_owner(http_request, model, generation_request.prompt)
    try:
        try:
            owner.mark_dispatched()
            async for partial in engine.stream(generation_request):
                last = partial
                owner.observe(partial.usage, partial.completions)
                for completion in partial.completions:
                    delta_text = completion.text[sent.get(completion.index, 0) :]
                    if not delta_text and not partial.finished:
                        continue
                    is_first = completion.index not in sent
                    sent[completion.index] = len(completion.text)
                    chunk_logprobs = None
                    if request.logprobs and completion.logprob_content is not None:
                        seen = logprobs_sent.get(completion.index, 0)
                        fresh = completion.logprob_content[seen:]
                        logprobs_sent[completion.index] = len(completion.logprob_content)
                        if fresh:
                            chunk_logprobs = ChoiceLogprobs(content=_logprob_entries(fresh))
                    yield _sse_chunk(
                        response_id,
                        created,
                        model,
                        completion.index,
                        ChunkDelta(
                            role="assistant" if is_first else None,
                            content=delta_text,
                        ),
                        include_usage=include_usage,
                        logprobs=chunk_logprobs,
                    )
        except Exception as error:  # surface backend failures inside the SSE stream
            logger.exception("upstream backend error")
            payload = {  # M3: only the class name, no raw backend message
                "error": {
                    "message": f"upstream backend error ({type(error).__name__})",
                    "type": "upstream_error",
                }
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
            return
        for completion in last.completions if last else ():
            yield _sse_chunk(
                response_id,
                created,
                model,
                completion.index,
                ChunkDelta(),
                finish_reason=completion.finish_reason or "stop",
                include_usage=include_usage,
            )
        if include_usage and last is not None:
            yield _usage_chunk(
                response_id,
                created,
                model,
                _wire_usage(
                    generation_request.prompt,
                    last.completions,
                    owner.latest_usage,
                ),
            )
        yield "data: [DONE]\n\n"
    finally:
        owner.finalize()


async def _stream_orchestrator(
    orchestrator,
    call: OrchestrationRequest | str,
    request: ChatCompletionRequest,
    include_usage: bool,
    want_trace: bool,
    http_request: Request,
) -> AsyncIterator[str]:
    """AUTO-model SSE (m11 D1/A2): status keep-alives ride SSE COMMENT lines
    (the OpenAI SDK parses every data: payload as a chunk), deltas and the
    final chunk are standard chat chunks."""
    if isinstance(call, str):
        call = OrchestrationRequest(
            prompt=call,
            sampling_params=sampling_params_from(request),
            tools=tuple(request.tools or ()),
            tool_choice=request.tool_choice,
            response_format=request.response_format,
        )
    response_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    first = True
    sent: dict[int, int] = {}
    logprobs_sent: dict[int, int] = {}
    final_result = None
    completion_text = ""
    completions: tuple[CompletionOutput, ...] = ()
    reported_usage: GenerationUsage | None = None
    terminal_error_type: str | None = None
    prompt = call.prompt
    owner = _stream_usage_owner(http_request, request.model, prompt)

    def observe_internal_usage(usage: GenerationUsage) -> None:
        owner.observe(usage, completions)

    def fresh_chunks(
        snapshot: Sequence[CompletionOutput],
    ):
        for completion in sorted(snapshot, key=lambda item: item.index):
            seen_text = sent.get(completion.index, 0)
            delta_text = completion.text[seen_text:]
            seen_logprobs = logprobs_sent.get(completion.index, 0)
            fresh_logprobs = (
                completion.logprob_content[seen_logprobs:]
                if completion.logprob_content is not None
                else ()
            )
            if not delta_text and not fresh_logprobs and completion.index in sent:
                continue
            is_first = completion.index not in sent
            sent[completion.index] = len(completion.text)
            if completion.logprob_content is not None:
                logprobs_sent[completion.index] = len(completion.logprob_content)
            yield (
                completion.index,
                ChunkDelta(
                    role="assistant" if is_first else None,
                    content=delta_text,
                ),
                (
                    ChoiceLogprobs(content=_logprob_entries(fresh_logprobs))
                    if fresh_logprobs
                    else None
                ),
            )

    try:
        try:
            owner.mark_dispatched()
            stream = await orchestrator.run_chat(
                call,
                stream=True,
                usage_observer=observe_internal_usage,
            )
            async for event in stream:
                if event.kind == "status":
                    yield f": status {event.text}\n\n"  # SSE comment (A2)
                elif event.kind == "delta":
                    if event.completions:
                        completions = event.completions
                        owner.observe(None, completions)
                        for index, delta, chunk_logprobs in fresh_chunks(completions):
                            yield _sse_chunk(
                                response_id,
                                created,
                                request.model,
                                index,
                                delta,
                                include_usage=include_usage,
                                logprobs=chunk_logprobs,
                            )
                    else:
                        completion_text += event.text
                        completions = (
                            CompletionOutput(
                                index=0,
                                text=completion_text,
                                token_ids=(),
                                finish_reason=None,
                            ),
                        )
                        owner.observe(None, completions)
                        delta = (
                            ChunkDelta(role="assistant", content=event.text)
                            if first
                            else ChunkDelta(content=event.text)
                        )
                        first = False
                        yield _sse_chunk(
                            response_id,
                            created,
                            request.model,
                            0,
                            delta,
                            include_usage=include_usage,
                        )
                elif event.kind in {"result", "error"}:
                    final_result = event.result
                    if final_result is not None:
                        completion_text = final_result.text or completion_text
                        completions = final_result.completions or (
                            CompletionOutput(
                                index=0,
                                text=completion_text,
                                token_ids=(),
                                finish_reason="stop",
                            ),
                        )
                        for index, delta, chunk_logprobs in fresh_chunks(completions):
                            yield _sse_chunk(
                                response_id,
                                created,
                                request.model,
                                index,
                                delta,
                                include_usage=include_usage,
                                logprobs=chunk_logprobs,
                            )
                        if final_result.prompt_tokens or final_result.completion_tokens:
                            reported_usage = GenerationUsage(
                                prompt_tokens=final_result.prompt_tokens,
                                completion_tokens=final_result.completion_tokens,
                                cached_tokens=final_result.cached_tokens,
                            )
                        owner.observe(reported_usage, completions)
                    if event.kind == "error":
                        terminal_error_type = event.error_type or "RuntimeError"
                        break
        except Exception as error:  # surface as an SSE error event, then close
            logger.exception("orchestrator stream error")
            yield f'data: {{"error": {{"message": "{type(error).__name__}"}}}}\n\n'
            yield "data: [DONE]\n\n"
            return
        if terminal_error_type is not None and final_result is not None:
            usage = _orchestration_wire_usage(
                prompt,
                completions,
                prompt_tokens=final_result.prompt_tokens,
                completion_tokens=final_result.completion_tokens,
                cached_tokens=final_result.cached_tokens,
            )
            metadata = _orchestrator_metadata_chunk(
                response_id,
                created,
                request.model,
                usage=usage,
                result=final_result,
                include_usage=include_usage,
                want_trace=want_trace,
            )
            if metadata is not None:
                yield metadata
            if want_trace:
                yield f": trace {' | '.join(final_result.trace)}\n\n"
            safe_type = (
                terminal_error_type if terminal_error_type.isidentifier() else "RuntimeError"
            )
            payload = {
                "error": {
                    "message": f"upstream backend error ({safe_type})",
                    "type": "upstream_error",
                    "code": "backend_error",
                }
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
            return
        final_completions = completions or (
            CompletionOutput(index=0, text="", token_ids=(), finish_reason="stop"),
        )
        for completion in sorted(final_completions, key=lambda item: item.index):
            yield _sse_chunk(
                response_id,
                created,
                request.model,
                completion.index,
                ChunkDelta(),
                finish_reason=completion.finish_reason or "stop",
                include_usage=include_usage,
            )
        if final_result is not None:
            usage = _orchestration_wire_usage(
                prompt,
                completions,
                prompt_tokens=final_result.prompt_tokens,
                completion_tokens=final_result.completion_tokens,
                cached_tokens=final_result.cached_tokens,
                approximate_missing_standard_usage=True,
            )
            metadata = _orchestrator_metadata_chunk(
                response_id,
                created,
                request.model,
                usage=usage,
                result=final_result,
                include_usage=include_usage,
                want_trace=want_trace,
            )
            if metadata is not None:
                yield metadata
            if want_trace:
                yield f": trace {' | '.join(final_result.trace)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        owner.finalize()


async def _stream_choices(
    choices: list[Choice],
    model: str,
    usage: Usage | None = None,
    *,
    orchestration_result=None,
    want_trace: bool = False,
) -> AsyncIterator[str]:
    """Stream already-final choices (orchestrated or tool-call responses)."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    include_usage = usage is not None
    for choice in choices:
        tool_calls = None
        if choice.message.tool_calls:
            # attach the required per-item index so SDK stream accumulators merge
            # the tool-call fragments correctly (S6)
            tool_calls = [
                ChunkToolCall(index=i, id=tc.id, type=tc.type, function=tc.function)
                for i, tc in enumerate(choice.message.tool_calls)
            ]
        yield _sse_chunk(
            response_id,
            created,
            model,
            choice.index,
            ChunkDelta(
                role="assistant",
                content=choice.message.content,
                tool_calls=tool_calls,
            ),
            include_usage=include_usage,
            logprobs=choice.logprobs,
        )
        yield _sse_chunk(
            response_id,
            created,
            model,
            choice.index,
            ChunkDelta(),
            finish_reason=choice.finish_reason,
            include_usage=include_usage,
        )
    if usage is not None:
        yield _usage_chunk(response_id, created, model, usage)
    if orchestration_result is not None and want_trace:
        metadata = _orchestrator_metadata_chunk(
            response_id,
            created,
            model,
            usage=None,
            result=orchestration_result,
            include_usage=False,
            want_trace=True,
        )
        if metadata is not None:
            yield metadata
        yield f": trace {' | '.join(orchestration_result.trace)}\n\n"
    yield "data: [DONE]\n\n"


def _completion_logprobs(completion: CompletionOutput) -> CompletionLogprobs | None:
    """Legacy four-parallel-array shape (m9 D3); offsets from 0 within text."""
    if completion.logprob_content is None:
        return None
    tokens: list[str] = []
    token_logprobs: list[float] = []
    top_logprobs: list[dict[str, float]] = []
    text_offset: list[int] = []
    offset = 0
    has_top = False
    for entry in completion.logprob_content:
        tokens.append(entry.token)
        token_logprobs.append(entry.logprob)
        top_logprobs.append({top.token: top.logprob for top in entry.top})
        has_top = has_top or bool(entry.top)
        text_offset.append(offset)
        offset += len(entry.token)
    return CompletionLogprobs(
        tokens=tokens,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs if has_top else None,
        text_offset=text_offset,
    )


def _completion_choice(index: int, completion: CompletionOutput) -> CompletionChoice:
    return CompletionChoice(
        index=index,
        text=completion.text,
        logprobs=_completion_logprobs(completion),
        finish_reason=completion.finish_reason or "stop",
    )


async def _stream_completions(
    engine: EngineBackend,
    generation_request: GenerationRequest,
    request: CompletionRequest,
    http_request: Request,
) -> AsyncIterator[str]:
    """Legacy text_completion stream: cumulative text deltas, not delta objects."""
    response_id = f"cmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    include_usage = bool(request.stream_options and request.stream_options.include_usage)
    sent: dict[int, int] = {}
    last = None
    owner = _stream_usage_owner(http_request, request.model, generation_request.prompt)

    def _chunk(choices: list[CompletionChoice], usage: Usage | None = None) -> str:
        payload = CompletionChunk(
            id=response_id,
            created=created,
            model=request.model,
            choices=choices,
            usage=usage,
        )
        exclude = None if include_usage else {"usage"}
        return f"data: {payload.model_dump_json(exclude=exclude)}\n\n"

    try:
        try:
            owner.mark_dispatched()
            async for partial in engine.stream(generation_request):
                last = partial
                owner.observe(partial.usage, partial.completions)
                for completion in partial.completions:
                    delta = completion.text[sent.get(completion.index, 0) :]
                    if not delta and not partial.finished:
                        continue
                    sent[completion.index] = len(completion.text)
                    finish = (completion.finish_reason or "stop") if partial.finished else None
                    yield _chunk(
                        [
                            CompletionChoice(
                                index=completion.index,
                                text=delta,
                                finish_reason=finish,
                            )
                        ]
                    )
        except Exception as error:
            logger.exception("upstream backend error")
            payload = {  # M3: only the class name, no raw backend message
                "error": {
                    "message": f"upstream backend error ({type(error).__name__})",
                    "type": "upstream_error",
                }
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
            return
        if include_usage and last is not None:
            yield _chunk(
                [],
                usage=_wire_usage(
                    generation_request.prompt,
                    last.completions,
                    owner.latest_usage,
                ),
            )
        yield "data: [DONE]\n\n"
    finally:
        owner.finalize()


def _record_usage(
    http_request: Request,
    model: str,
    usage: GenerationUsage | Usage | None,
    *,
    prompt: str,
    completions: Sequence[CompletionOutput],
) -> None:
    """m11 D3/A7: metering happens in handlers (middleware can't see tokens)."""
    prompt_tokens, completion_tokens = resolve_usage_counts(
        usage, prompt=prompt, completions=completions
    )
    tenant = getattr(http_request.state, "tenant", None) or "default"
    record_state_usage(
        http_request.app.state,
        tenant=tenant,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=resolve_cached_tokens(usage),
    )


def _session_id(request: ChatCompletionRequest, http_request: Request) -> str | None:
    """Session for ReplicaPool affinity: X-Session-ID header, else the OpenAI user field."""
    return http_request.headers.get("x-session-id") or request.user


def _scheduling_class(http_request: Request) -> str:
    """Resolve trusted gateway state, then the Kairyu replica transport hint."""

    request_class = getattr(http_request.state, "scheduling_class", None)
    if request_class in {"interactive", "batch"}:
        return request_class
    transported = http_request.headers.get("x-kairyu-scheduling-class")
    return transported if transported in {"interactive", "batch"} else "interactive"


def create_app(
    engines: Mapping[str, EngineBackend],
    orchestrator: Orchestrator | None = None,
    settings: ServerSettings | None = None,
    lifespan=None,
    chat_templates: Mapping[str, ChatTemplate] | None = None,
    orchestrators: Mapping[str, Orchestrator] | None = None,
    tenant_config=None,
    embedding_backends: Mapping[str, EmbeddingBackend] | None = None,
    resolved_api_keys: frozenset[str] | None = None,
    resolved_admin_keys: frozenset[str] | None = None,
    price_sheet: PriceSheet | None = None,
) -> FastAPI:
    settings = settings or ServerSettings()
    if price_sheet is not None and settings.usage_ledger_path is None:
        raise ValueError("price_sheet requires usage_ledger_path")
    if price_sheet is not None:
        known_tenants = {"default"}
        if tenant_config is not None:
            known_tenants = {
                tenant_config.default_tenant,
                *tenant_config.key_tenants.values(),
            }
        unknown_discounts = price_sheet.tenant_discounts.keys() - known_tenants
        if unknown_discounts:
            raise ValueError(
                f"price_sheet discounts reference unknown tenants {sorted(unknown_discounts)}"
            )
    app = FastAPI(
        title="kairyu",
        version="0.1.0",
        lifespan=_with_usage_ledger_cleanup(lifespan),
    )
    served_engines = dict(engines)
    # m11 D2: tiered auto models; the single-orchestrator kwarg is a shim
    auto_models: dict[str, Orchestrator] = dict(orchestrators or {})
    if orchestrator is not None:
        auto_models.setdefault(AUTO_MODEL, orchestrator)
    served_embedding_backends = dict(embedding_backends or {})
    collisions = (
        (set(auto_models) & set(served_engines))
        | (set(served_embedding_backends) & set(served_engines))
        | (set(served_embedding_backends) & set(auto_models))
    )
    if collisions:
        raise ValueError(
            "served model names collide across engines, orchestrators, and embeddings: "
            f"{sorted(collisions)}"
        )

    metrics = ServerMetrics() if settings.metrics else None
    app.state.metrics = metrics
    if metrics is not None:
        for name, engine in served_engines.items():
            if isinstance(engine, ReplicaPool):
                metrics.track_pool(name, engine)
            metrics.track_scheduler(name, engine)
    api_keys = settings.resolve_api_keys() if resolved_api_keys is None else resolved_api_keys
    admin_keys = (
        settings.resolve_admin_keys() if resolved_admin_keys is None else resolved_admin_keys
    )
    add_health_routes(app, served_engines, metrics, admin_keys=admin_keys)
    from kairyu.entrypoints.server.extra_routes import add_extra_routes

    add_extra_routes(
        app,
        served_engines,
        embedding_backends=served_embedding_backends,
        chat_templates=chat_templates,
    )

    # add_middleware prepends, so add innermost first: metrics -> concurrency
    # guard -> auth -> access log (outermost).
    if metrics is not None:
        app.add_middleware(MetricsMiddleware, metrics=metrics)
    if settings.max_concurrency is not None:
        app.add_middleware(ConcurrencyLimitMiddleware, limit=settings.max_concurrency)
    if tenant_config is not None:
        from kairyu.entrypoints.server.tenancy import (
            TenantLimiter,
            TenantLimitMiddleware,
        )

        # added BEFORE auth => auth wraps it: 401 wins over 429 and
        # unauthenticated requests never drain buckets (m11 A6)
        limiter = TenantLimiter(tenant_config)
        app.state.tenant_limiter = limiter  # handlers charge tokens post-response (S4)
        app.add_middleware(
            TenantLimitMiddleware,
            config=tenant_config,
            limiter=limiter,
        )
    if api_keys or admin_keys:
        app.add_middleware(
            AuthMiddleware,
            api_keys=api_keys,
            admin_keys=admin_keys,
            protect_metrics=settings.protect_metrics,
        )
    ledger = None
    if settings.usage_ledger_path:
        from kairyu.entrypoints.server.tenancy import UsageLedger

        ledger = UsageLedger(settings.usage_ledger_path)
        app.state.usage_ledger = ledger
        if metrics is not None:
            metrics.restore_usage_totals(ledger.totals())

        @app.get("/admin/usage")
        async def admin_usage(http_request: Request, tenant: str | None = None):
            """Scoped to the CALLER's tenant when tenancy is configured
            (security review: no cross-tenant disclosure); single-tenant
            deployments (no tenant_config) see everything behind auth."""
            state = http_request.scope.get("state", {})
            if state.get("is_admin"):
                return {
                    "usage": await asyncio.to_thread(
                        ledger.totals,
                        tenant,
                    )
                }
            if tenant_config is not None:
                caller = getattr(http_request.state, "tenant", None)
                if caller is None:
                    caller = tenant_config.default_tenant
                if tenant is not None and tenant != caller:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": {
                                "message": "cannot query another tenant's usage",
                                "type": "invalid_request_error",
                                "code": "tenant_forbidden",
                            }
                        },
                    )
                return {
                    "usage": await asyncio.to_thread(
                        ledger.totals,
                        caller,
                    )
                }
            return {
                "usage": await asyncio.to_thread(
                    ledger.totals,
                    tenant,
                )
            }

        if price_sheet is not None:
            app.state.price_sheet = price_sheet

            @app.get("/admin/usage.csv")
            async def admin_usage_csv(
                http_request: Request,
                tenant: str | None = None,
                start_ts: float | None = None,
                end_ts: float | None = None,
            ):
                """Export a deterministic, caller-scoped invoice snapshot."""
                state = http_request.scope.get("state", {})
                scoped_tenant = tenant
                if not state.get("is_admin") and tenant_config is not None:
                    caller = getattr(http_request.state, "tenant", None)
                    if caller is None:
                        caller = tenant_config.default_tenant
                    if tenant is not None and tenant != caller:
                        return JSONResponse(
                            status_code=403,
                            content={
                                "error": {
                                    "message": ("cannot export another tenant's invoice"),
                                    "type": "invalid_request_error",
                                    "code": "tenant_forbidden",
                                }
                            },
                        )
                    scoped_tenant = caller
                try:
                    snapshot = await asyncio.to_thread(ledger.snapshot_bytes)
                    payload = await asyncio.to_thread(
                        export_invoice_csv,
                        {"gateway-local": snapshot},
                        price_sheet,
                        tenant=scoped_tenant,
                        start_ts=start_ts,
                        end_ts=end_ts,
                    )
                except InvoiceExportError as error:
                    return JSONResponse(
                        status_code=409,
                        content={
                            "error": {
                                "message": str(error),
                                "type": "invoice_export_error",
                                "code": "invoice_ledger_invalid",
                            }
                        },
                    )
                return Response(
                    content=payload,
                    media_type="text/csv",
                    headers={
                        "content-disposition": ('attachment; filename="kairyu-usage-invoice.csv"')
                    },
                )

    if settings.tracing:
        from kairyu.entrypoints.server.middleware import TracingMiddleware
        from kairyu.telemetry import configure_tracing

        configure_tracing(True)
        app.add_middleware(TracingMiddleware)
    if settings.access_log:
        app.add_middleware(AccessLogMiddleware)

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        names = list(served_engines) + list(auto_models) + list(served_embedding_backends)
        return ModelList(data=[ModelCard(id=name) for name in names])

    @app.post("/v1/route", response_model=RoutePreviewResponse)
    async def preview_route(request: RoutePreviewRequest, http_request: Request):
        http_request.state.model = request.model
        selected = auto_models.get(request.model)
        if selected is not None:
            chat_request = ChatCompletionRequest(
                model=request.model,
                messages=request.messages,
            )
            try:
                prompt = validate_chat_input(chat_request, chat_templates).prompt
                decision = selected.preview_route(prompt)
            except ChatRequestError as error:
                return JSONResponse(
                    status_code=error.status_code,
                    content={"error": error.payload()},
                )
            except PreviewNotSupportedError as error:
                return JSONResponse(
                    status_code=409,
                    content={
                        "error": {
                            "message": str(error),
                            "type": "invalid_request_error",
                            "code": "preview_not_supported",
                        }
                    },
                )
            payload = _route_payload(decision)
            descriptor = selected.describe_routing()
            return RoutePreviewResponse(
                model=request.model,
                orchestrated=True,
                router_type=descriptor["router"]["router_type"],
                **payload.model_dump(),
            )
        if request.model in served_engines:
            return RoutePreviewResponse(model=request.model, orchestrated=False)
        return model_not_found(request.model)

    @app.get("/routing", response_model=RoutingResponse)
    async def routing_config() -> RoutingResponse:
        return RoutingResponse(
            models={name: selected.describe_routing() for name, selected in auto_models.items()}
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(request: ChatCompletionRequest, http_request: Request):
        http_request.state.model = request.model  # label for the metrics middleware
        if request.model in auto_models:
            try:
                validated_input = validate_chat_input(request, chat_templates)
            except ChatRequestError as error:
                return JSONResponse(
                    status_code=error.status_code, content={"error": error.payload()}
                )
            prompt = validated_input.prompt
            normalized_tool_choice = validated_input.normalized_tool_choice
            include_usage = validated_input.include_usage
            try:
                sampling = sampling_params_from(request)
            except ValueError as error:
                return invalid_request(str(error))
            orchestration_request = OrchestrationRequest(
                prompt=prompt,
                sampling_params=sampling,
                tools=tuple(request.tools or ()),
                tool_choice=request.tool_choice,
                tools_in_prompt=validated_input.tools_in_prompt,
                response_format=request.response_format,
            )
            selected = auto_models[request.model]
            try:
                selected.validate_request(orchestration_request)
            except ValueError as error:
                return invalid_request(str(error))
            want_trace = http_request.headers.get("x-kairyu-trace") == "1"
            # Tool choices must be validated across every final choice before
            # any bytes become irrevocable SSE output. Indexed alternatives
            # and logprobs otherwise stay on the low-latency pull-through path.
            buffered_stream = bool(request.tools)
            if request.stream and not buffered_stream:
                return StreamingResponse(
                    _stream_orchestrator(
                        selected,
                        orchestration_request,
                        request,
                        include_usage,
                        want_trace,
                        http_request,
                    ),
                    media_type="text/event-stream",
                )
            try:
                result = await selected.run(orchestration_request)
            except OrchestratorExecutionError as error:
                logger.exception("orchestrator backend error")
                result = error.result
                completions = result.completions or (
                    CompletionOutput(
                        index=0,
                        text=result.text,
                        token_ids=(),
                        finish_reason=None,
                    ),
                )
                usage = _orchestration_wire_usage(
                    prompt,
                    completions,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    cached_tokens=result.cached_tokens,
                )
                _record_usage(
                    http_request,
                    request.model,
                    usage,
                    prompt=prompt,
                    completions=completions,
                )
                payload = {
                    "error": sanitize_backend_error(error.cause),
                    "usage": usage.model_dump(mode="json"),
                }
                if want_trace:
                    payload["kairyu_trace"] = list(result.trace)
                    if result.structured_trace is not None:
                        payload["kairyu_trace_v2"] = result.structured_trace.as_dict()
                    payload["kairyu_route"] = _route_payload(result.route).model_dump(mode="json")
                return JSONResponse(status_code=502, content=payload)
            except Exception as error:
                return upstream_error(error)
            completions = result.completions or (
                CompletionOutput(index=0, text=result.text, token_ids=(), finish_reason="stop"),
            )
            # OrchestratorResult uses 0/0 when its backend did not report usage.
            # Keep that state missing so the same wire approximation can derive it.
            usage = (
                GenerationUsage(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    cached_tokens=result.cached_tokens,
                )
                if result.prompt_tokens or result.completion_tokens
                else None
            )
            response = completion_response(
                request,
                prompt,
                completions,
                usage=usage,
                normalized_tool_choice=normalized_tool_choice,
            )
            response.usage.orchestration_input_tokens = result.prompt_tokens
            response.usage.orchestration_output_tokens = result.completion_tokens
            if not tool_choice_is_satisfied(
                response.choices,
                normalized_tool_choice,
            ):
                _record_usage(
                    http_request,
                    request.model,
                    response.usage,
                    prompt=prompt,
                    completions=completions,
                )
                error = ChatRequestError(
                    "upstream model did not satisfy tool_choice",
                    status_code=502,
                    code="tool_choice_not_satisfied",
                    error_type="upstream_error",
                )
                return JSONResponse(
                    status_code=error.status_code,
                    content={"error": error.payload()},
                )
            _record_usage(
                http_request,
                request.model,
                response.usage,
                prompt=prompt,
                completions=completions,
            )
            if request.stream:
                return StreamingResponse(
                    _stream_choices(
                        response.choices,
                        request.model,
                        usage=response.usage if include_usage else None,
                        orchestration_result=result,
                        want_trace=want_trace,
                    ),
                    media_type="text/event-stream",
                )
            if want_trace:
                payload = _chat_response_payload(response)
                payload["kairyu_trace"] = list(result.trace)
                if result.structured_trace is not None:
                    payload["kairyu_trace_v2"] = result.structured_trace.as_dict(
                        request_id=response.id
                    )
                payload["kairyu_route"] = _route_payload(result.route).model_dump(mode="json")
                return JSONResponse(content=payload)
            return JSONResponse(content=_chat_response_payload(response))

        session_id = _session_id(request, http_request)
        try:
            validated = validate_chat_request(
                request,
                served_engines,
                chat_templates,
                request_id=f"http-{uuid.uuid4().hex[:12]}",
                # Affinity glue (m7 D6): keeps a session's turns on the replica
                # holding its warm radix-KV prefix.
                cache_hint=CacheHint(session_id=session_id) if session_id else None,
                priority=getattr(http_request.state, "priority", None),
                scheduling_class=_scheduling_class(http_request),
            )
        except ChatRequestError as error:
            return JSONResponse(status_code=error.status_code, content={"error": error.payload()})
        if metrics is not None:
            metrics.record_priority(
                validated.generation_request.scheduling_class,
                source="http",
            )
        if request.stream and not request.tools:
            return StreamingResponse(
                _stream_engine(
                    validated.engine,
                    validated.generation_request,
                    request.model,
                    request,
                    http_request,
                ),
                media_type="text/event-stream",
            )
        try:
            executed = await execute_chat(validated)
        except ChatRequestError as error:
            if error.execution is not None:
                _record_usage(
                    http_request,
                    request.model,
                    error.execution.result.usage,
                    prompt=error.execution.result.prompt,
                    completions=error.execution.result.completions,
                )
            return JSONResponse(status_code=error.status_code, content={"error": error.payload()})
        except Exception as error:
            return upstream_error(error)
        response = executed.response
        _record_usage(
            http_request,
            request.model,
            executed.result.usage,
            prompt=executed.result.prompt,
            completions=executed.result.completions,
        )
        if request.stream:
            # Tool calling + streaming: generate fully, then emit structured chunks so
            # tool_calls and finish_reason stay correct.
            return StreamingResponse(
                _stream_choices(
                    response.choices,
                    request.model,
                    usage=(response.usage if validated.input.include_usage else None),
                ),
                media_type="text/event-stream",
            )
        return JSONResponse(content=_chat_response_payload(response))

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest, http_request: Request):
        http_request.state.model = request.model
        extra = request.model_extra or {}
        for unsupported in ("echo", "suffix", "best_of"):
            if extra.get(unsupported) is not None:
                return invalid_request(f"{unsupported} is not supported")
        if request.logprobs is not None and not 0 <= request.logprobs <= 5:
            return invalid_request("logprobs must be between 0 and 5")
        if request.stream_options is not None and not request.stream:
            return invalid_request("stream_options is only allowed when stream is true")
        if isinstance(request.prompt, list) and request.stream:
            return invalid_request("streaming with a prompt array is not supported")
        engine = served_engines.get(request.model)
        if engine is None:
            return model_not_found(request.model)
        if request.n > 1 and getattr(engine, "supports_n", True) is False:
            return invalid_request(f"model {request.model!r} does not support n > 1")
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        try:
            sampling = SamplingParams(  # invalid params are a client error, not a 502
                temperature=request.temperature,
                top_p=request.top_p,
                n=request.n,
                max_tokens=request.max_tokens,
                stop=request.stop,
                seed=request.seed,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                logprobs=request.logprobs,
            )
        except ValueError as error:
            return invalid_request(str(error))

        def _generation_request(prompt: str) -> GenerationRequest:
            return GenerationRequest(
                request_id=f"http-{uuid.uuid4().hex[:12]}",
                prompt=prompt,
                sampling_params=sampling,
                priority=getattr(http_request.state, "priority", request.priority),
                scheduling_class=_scheduling_class(http_request),
            )

        generation_requests = [_generation_request(prompt) for prompt in prompts]
        for generation_request in generation_requests:
            validation_error = _validate_generation_request(engine, generation_request)
            if validation_error is not None:
                return validation_error
        if metrics is not None:
            for generation_request in generation_requests:
                metrics.record_priority(
                    generation_request.scheduling_class,
                    source="http",
                )
        if request.stream:
            return StreamingResponse(
                _stream_completions(
                    engine,
                    generation_requests[0],
                    request,
                    http_request,
                ),
                media_type="text/event-stream",
            )
        choices: list[CompletionChoice] = []
        usage_totals = [0, 0, 0]  # prompt, completion, cached
        try:
            # run the prompt array concurrently (latency = max, not sum); order is
            # restored by prompt_index below so the response is unchanged (P-perf)
            results = await asyncio.gather(*(engine.generate(item) for item in generation_requests))
        except Exception as error:
            return upstream_error(error)
        for prompt_index, (prompt, result) in enumerate(zip(prompts, results, strict=True)):
            for completion in result.completions:
                choices.append(
                    _completion_choice(prompt_index * request.n + completion.index, completion)
                )
            prompt_tokens, completion_tokens = resolve_usage_counts(
                result.usage,
                prompt=prompt,
                completions=result.completions,
            )
            usage_totals[0] += prompt_tokens
            usage_totals[1] += completion_tokens
            if result.usage is not None:
                usage_totals[2] += result.usage.cached_tokens
        details = PromptTokensDetails(cached_tokens=usage_totals[2]) if usage_totals[2] else None
        _record_usage(  # S3: /v1/completions was never metered
            http_request,
            request.model,
            GenerationUsage(
                prompt_tokens=usage_totals[0],
                completion_tokens=usage_totals[1],
                cached_tokens=usage_totals[2],
            ),
            prompt="",
            completions=(),
        )
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:16]}",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage=Usage(
                prompt_tokens=usage_totals[0],
                completion_tokens=usage_totals[1],
                total_tokens=usage_totals[0] + usage_totals[1],
                prompt_tokens_details=details,
            ),
        )

    return app
