"""``/v1/responses`` route registration and request handling.

Responses wire items are normalized into a ``ChatCompletionRequest`` first, so
engine models share the Chat Completions validation and admission contract,
and AUTO models delegate to the Chat Completions handler itself.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from kairyu.engine.backend import (
    CacheHint,
    EngineBackend,
    UpstreamClientError,
    backend_admission_upper_bound_async,
    prepare_backend_request,
)
from kairyu.entrypoints.chat_template import ChatTemplate
from kairyu.entrypoints.server.chat_service import (
    ChatRequestError,
    ValidatedChatRequest,
    chat_error_from_upstream_client_error,
    validate_chat_request_async,
)
from kairyu.entrypoints.server.errors import upstream_error
from kairyu.entrypoints.server.protocol import ChatCompletionRequest
from kairyu.entrypoints.server.responses.canonical import (
    canonical_input,
    validate_function_outputs,
)
from kairyu.entrypoints.server.responses.compaction import (
    CompactionCodec,
    compaction_prompt_items,
    extract_compaction_trigger,
)
from kairyu.entrypoints.server.responses.deps import ChatDispatch, ResponsesDeps
from kairyu.entrypoints.server.responses.errors import (
    chat_error,
    request_error,
    request_failure,
)
from kairyu.entrypoints.server.responses.paths_legacy_buffered import (
    engine_buffered_response,
)
from kairyu.entrypoints.server.responses.paths_legacy_live import live_text_events
from kairyu.entrypoints.server.responses.paths_legacy_relay import orchestrated_response
from kairyu.entrypoints.server.responses.request import (
    ResponsesRequest,
    validate_request_surface,
)
from kairyu.entrypoints.server.responses.store import ResponseStore
from kairyu.entrypoints.server.responses.to_chat import to_chat_request
from kairyu.entrypoints.server.sse_response import sse_response

if TYPE_CHECKING:
    from kairyu.entrypoints.server.metrics import ServerMetrics
    from kairyu.entrypoints.server.tenancy import TenantAdmission

_SCHEDULING_CLASSES = frozenset({"interactive", "batch"})


@dataclass(frozen=True)
class _Turn:
    """The prompt of one turn and what a successful turn stores."""

    chat_request: ChatCompletionRequest
    stored_items: list[dict]
    compaction_request: bool


def add_responses_route(
    app: FastAPI,
    engines: Mapping[str, EngineBackend],
    *,
    chat_templates: Mapping[str, ChatTemplate] | None = None,
    legacy_chat_models: AbstractSet[str] | None = None,
    orchestrated_models: AbstractSet[str] | None = None,
    chat_dispatch: ChatDispatch | None = None,
    compaction_key: bytes,
) -> ResponseStore:
    store = ResponseStore()
    app.state.response_store = store
    deps = ResponsesDeps(
        engines=engines,
        store=store,
        compaction_codec=CompactionCodec(compaction_key),
        chat_templates=chat_templates,
        legacy_chat_models=legacy_chat_models,
        orchestrated_models=orchestrated_models,
        chat_dispatch=chat_dispatch,
    )

    @app.get("/v1/responses")
    async def responses_upgrade_required() -> JSONResponse:
        # Codex tries a WebSocket upgrade first when it targets the built-in
        # openai provider (the Harbor/Terminal-Bench shape). The upgrade
        # arrives as a GET; 426 makes Codex fall back to HTTPS immediately
        # and silently instead of burning its stream-retry budget.
        return JSONResponse(
            status_code=426,
            content={
                "error": {
                    "message": (
                        "WebSocket transport is not supported; "
                        "retry over HTTPS"
                    ),
                    "type": "invalid_request_error",
                    "code": "upgrade_required",
                }
            },
        )

    @app.post("/v1/responses")
    async def responses(request: ResponsesRequest, http_request: Request):
        return await _create_response(deps, request, http_request)

    return store


async def _create_response(
    deps: ResponsesDeps, request: ResponsesRequest, http_request: Request
) -> Response:
    http_request.state.model = request.model
    metrics = getattr(http_request.app.state, "metrics", None)
    ingress_ns = getattr(http_request.state, "placement_started_ns", None)
    if metrics is not None and type(ingress_ns) is int:
        metrics.record_preplacement_phase(
            "responses",
            "ingress_to_handler",
            max(0, time.perf_counter_ns() - ingress_ns),
        )
    try:
        validate_request_surface(request)
    except ChatRequestError as error:
        return chat_error(error)
    engine = deps.engines.get(request.model)
    orchestrated = (
        engine is None
        and deps.chat_dispatch is not None
        and request.model in (deps.orchestrated_models or ())
    )
    if engine is None and not orchestrated:
        return request_error(
            f"model {request.model!r} not found",
            status_code=404,
            code="model_not_found",
        )
    owner = getattr(http_request.state, "tenant", None) or "default"
    context: list[dict] = []
    if request.previous_response_id:
        previous = deps.store.get(request.previous_response_id, owner=owner)
        if previous is None:
            return request_error("previous response not found", status_code=404)
        context.extend(previous)
    validation_started_ns = time.perf_counter_ns()
    try:
        turn = _prepare_turn(deps, request, context, owner=owner)
        if not orchestrated:
            validated = await _validate_engine_request(deps, request, turn, http_request)
    except ChatRequestError as error:
        return request_failure(request, error) or chat_error(error)
    finally:
        if metrics is not None:
            metrics.record_preplacement_phase(
                "responses",
                "request_validation",
                max(0, time.perf_counter_ns() - validation_started_ns),
            )
    if orchestrated:
        return await orchestrated_response(
            request,
            turn.chat_request,
            http_request,
            deps.chat_dispatch,
            response_id=f"resp_{uuid.uuid4().hex}",
            created_at=int(time.time()),
            stored_items=turn.stored_items,
            store=deps.store,
            owner=owner,
            compaction_codec=deps.compaction_codec,
            compaction_request=turn.compaction_request,
        )
    return await _engine_response(
        deps, request, validated, turn, http_request, owner=owner, metrics=metrics
    )


def _prepare_turn(
    deps: ResponsesDeps, request: ResponsesRequest, context: list[dict], *, owner: str
) -> _Turn:
    current_items = canonical_input(
        request.input, compaction_codec=deps.compaction_codec, owner=owner
    )
    all_items = context + current_items
    compaction_request = extract_compaction_trigger(all_items)
    work_items = (
        [item for item in all_items if item["type"] != "compaction_trigger"]
        if compaction_request
        else all_items
    )
    # A successful compaction replaces the continuation context. Keeping
    # ``work_items`` here would store the full pre-compaction history next
    # to its summary, so a later previous_response_id request would grow
    # the prompt instead of compacting it.
    stored_items = [] if compaction_request else work_items
    validate_function_outputs(work_items)
    prompt_items = (
        compaction_prompt_items(work_items) if compaction_request else work_items
    )
    chat_request = to_chat_request(
        request, prompt_items, compaction_codec=deps.compaction_codec, owner=owner
    )
    if compaction_request:
        # The summary is a plain text turn: tools cannot help it and a
        # tool call in its place would break the client's compaction
        # collection, so the summarization call runs tool-free.
        chat_request = chat_request.model_copy(
            update={"tools": None, "tool_choice": None}
        )
    return _Turn(chat_request, stored_items, compaction_request)


async def _validate_engine_request(
    deps: ResponsesDeps, request: ResponsesRequest, turn: _Turn, http_request: Request
) -> ValidatedChatRequest:
    cache_key = request.prompt_cache_key or request.previous_response_id
    scheduling_class = getattr(http_request.state, "scheduling_class", None)
    if scheduling_class not in _SCHEDULING_CLASSES:
        transported = http_request.headers.get("x-kairyu-scheduling-class")
        scheduling_class = (
            transported if transported in _SCHEDULING_CLASSES else "interactive"
        )
    return await validate_chat_request_async(
        turn.chat_request,
        deps.engines,
        deps.chat_templates,
        request_id=(
            getattr(http_request.state, "request_id", None)
            or f"resp-{uuid.uuid4().hex[:12]}"
        ),
        cache_hint=CacheHint(session_id=cache_key) if cache_key else None,
        priority=getattr(http_request.state, "priority", None),
        scheduling_class=scheduling_class,
        placement_started_ns=getattr(http_request.state, "placement_started_ns", None),
        legacy_chat_models=deps.legacy_chat_models,
    )


async def _engine_response(
    deps: ResponsesDeps,
    request: ResponsesRequest,
    validated: ValidatedChatRequest,
    turn: _Turn,
    http_request: Request,
    *,
    owner: str,
    metrics: ServerMetrics | None,
) -> Response:
    failure = await _prepare_backend(request, validated, metrics)
    if failure is not None:
        return failure
    admission = getattr(http_request.state, "tenant_admission", None)
    failure = await _admit(validated, http_request, admission, metrics, owner=owner)
    if failure is not None:
        return failure
    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())
    if request.stream and not request.tools and not turn.compaction_request:
        return sse_response(
            live_text_events(
                request,
                validated,
                response_id=response_id,
                created_at=created_at,
                stored_items=turn.stored_items,
                store=deps.store,
                owner=owner,
                http_request=http_request,
            )
        )
    return await engine_buffered_response(
        request,
        validated,
        http_request,
        admission=admission,
        response_id=response_id,
        created_at=created_at,
        stored_items=turn.stored_items,
        store=deps.store,
        owner=owner,
        compaction_codec=deps.compaction_codec,
        compaction_request=turn.compaction_request,
    )


async def _prepare_backend(
    request: ResponsesRequest,
    validated: ValidatedChatRequest,
    metrics: ServerMetrics | None,
) -> Response | None:
    prepare_started_ns = time.perf_counter_ns()
    try:
        await prepare_backend_request(
            validated.engine,
            validated.generation_request,
        )
    except UpstreamClientError as error:
        return chat_error(chat_error_from_upstream_client_error(error))
    except ValueError as error:
        return request_failure(request, error) or request_error(str(error))
    except RuntimeError as error:
        return upstream_error(error)
    finally:
        if metrics is not None:
            metrics.record_preplacement_phase(
                "responses",
                "backend_prepare",
                max(0, time.perf_counter_ns() - prepare_started_ns),
            )
    return None


async def _admit(
    validated: ValidatedChatRequest,
    http_request: Request,
    admission: TenantAdmission | None,
    metrics: ServerMetrics | None,
    *,
    owner: str,
) -> Response | None:
    admission_started_ns = time.perf_counter_ns()
    try:
        bound = await backend_admission_upper_bound_async(
            validated.engine,
            validated.generation_request,
        )
    except ValueError as error:
        return request_error(str(error))
    except RuntimeError as error:
        return upstream_error(error)
    admission_ns = max(0, time.perf_counter_ns() - admission_started_ns)
    reserve_started_ns = time.perf_counter_ns()
    if admission is not None:
        admitted = admission.reserve_tokens(
            bound.tokens,
            refundable_on_exact_usage=bound.refundable_on_exact_usage,
        )
        if metrics is not None:
            metrics.record_tenant_admission(
                owner,
                source="http",
                admitted=admitted,
                reason=admission.reason,
            )
        if admitted:
            http_request.state.tenant_metric_admitted = True
        if not admitted:
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": "1"},
                content={
                    "error": {
                        "message": (
                            f"tenant {owner!r} admission limit exceeded "
                            f"({admission.reason})"
                        ),
                        "type": "rate_limit_error",
                        "code": "tenant_rate_limited",
                    }
                },
            )
    admission_ns += max(0, time.perf_counter_ns() - reserve_started_ns)
    if metrics is not None:
        metrics.record_preplacement_phase(
            "responses",
            "admission",
            admission_ns,
        )
        metrics.record_priority(
            validated.generation_request.scheduling_class,
            source="http",
        )
    return None
