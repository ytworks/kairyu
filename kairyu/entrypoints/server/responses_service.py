"""OpenAI Responses API adapter with typed streaming and function tools.

The engine contract is intentionally shared with Chat Completions.  Responses
wire items are normalized into ``ChatCompletionRequest`` first, so chat
templates, tool-choice validation, and upstream capability preflight remain one
source of truth.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Mapping
from collections.abc import Set as AbstractSet

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from kairyu.engine.backend import (
    CacheHint,
    UpstreamClientError,
    backend_admission_upper_bound_async,
    prepare_backend_request,
)
from kairyu.entrypoints.server.chat_service import (
    ChatRequestError,
    chat_error_from_upstream_client_error,
    execute_chat,
    validate_chat_request_async,
)
from kairyu.entrypoints.server.errors import upstream_error
from kairyu.entrypoints.server.responses_auto import _orchestrated_response
from kairyu.entrypoints.server.responses_codec import (
    _COMPACTION_INSTRUCTION_ITEM,
    _COMPACTION_MAX_OUTPUT_TOKENS,
    _compaction_output_from_message,
    _CompactionCodec,
    _extract_compaction_trigger,
)
from kairyu.entrypoints.server.responses_engine import (
    _live_text_events,
    _record_execution,
    _validate_parallel_tool_calls,
)
from kairyu.entrypoints.server.responses_events import (
    _apply_terminal_item_status,
    _buffered_events,
    _terminal_status,
)
from kairyu.entrypoints.server.responses_items import (
    _canonical_input,
    _output_items,
    _to_chat_request,
    _validate_function_outputs,
)
from kairyu.entrypoints.server.responses_protocol import (
    ResponsesRequest,
    _BufferedFailure,
    _chat_error,
    _request_error,
    _response_envelope,
    _usage_payload,
    _validate_request_surface,
)
from kairyu.entrypoints.server.responses_store import ResponseStore
from kairyu.entrypoints.server.sse_response import sse_response

logger = logging.getLogger(__name__)


def add_responses_route(
    app: FastAPI,
    engines: Mapping,
    *,
    chat_templates=None,
    legacy_chat_models: AbstractSet[str] | None = None,
    orchestrated_models: AbstractSet[str] | None = None,
    chat_dispatch=None,
    compaction_key: bytes,
) -> ResponseStore:
    store = ResponseStore()
    app.state.response_store = store
    compaction_codec = _CompactionCodec(compaction_key)

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
            _validate_request_surface(request)
        except ChatRequestError as error:
            return _chat_error(error)
        engine = engines.get(request.model)
        orchestrated = (
            engine is None
            and chat_dispatch is not None
            and request.model in (orchestrated_models or ())
        )
        if engine is None and not orchestrated:
            return _request_error(
                f"model {request.model!r} not found",
                status_code=404,
                code="model_not_found",
            )
        owner = getattr(http_request.state, "tenant", None) or "default"
        context: list[dict] = []
        if request.previous_response_id:
            previous = store.get(request.previous_response_id, owner=owner)
            if previous is None:
                return _request_error("previous response not found", status_code=404)
            context.extend(previous)
        validation_started_ns = time.perf_counter_ns()
        try:
            current_items = _canonical_input(
                request.input, compaction_codec=compaction_codec, owner=owner
            )
            all_items = context + current_items
            compaction_request = _extract_compaction_trigger(all_items)
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
            _validate_function_outputs(work_items)
            prompt_items = (
                work_items + [_COMPACTION_INSTRUCTION_ITEM]
                if compaction_request
                else work_items
            )
            chat_request = _to_chat_request(
                request, prompt_items, compaction_codec=compaction_codec, owner=owner
            )
            if compaction_request:
                # The summary is a plain text turn: tools cannot help it and a
                # tool call in its place would break the client's compaction
                # collection, so the summarization call runs tool-free with
                # enough room for a useful summary.
                chat_request = chat_request.model_copy(
                    update={
                        "tools": None,
                        "tool_choice": None,
                        "max_completion_tokens": (
                            request.max_output_tokens
                            or _COMPACTION_MAX_OUTPUT_TOKENS
                        ),
                    }
                )
            if not orchestrated:
                cache_key = request.prompt_cache_key or request.previous_response_id
                scheduling_class = getattr(
                    http_request.state, "scheduling_class", None
                )
                if scheduling_class not in {"interactive", "batch"}:
                    transported = http_request.headers.get(
                        "x-kairyu-scheduling-class"
                    )
                    scheduling_class = (
                        transported
                        if transported in {"interactive", "batch"}
                        else "interactive"
                    )
                validated = await validate_chat_request_async(
                    chat_request,
                    engines,
                    chat_templates,
                    request_id=(
                        getattr(http_request.state, "request_id", None)
                        or f"resp-{uuid.uuid4().hex[:12]}"
                    ),
                    cache_hint=CacheHint(session_id=cache_key) if cache_key else None,
                    priority=getattr(http_request.state, "priority", None),
                    scheduling_class=scheduling_class,
                    placement_started_ns=getattr(
                        http_request.state, "placement_started_ns", None
                    ),
                    legacy_chat_models=legacy_chat_models,
                )
        except ChatRequestError as error:
            return _chat_error(error)
        finally:
            if metrics is not None:
                metrics.record_preplacement_phase(
                    "responses",
                    "request_validation",
                    max(0, time.perf_counter_ns() - validation_started_ns),
                )
        if orchestrated:
            return await _orchestrated_response(
                request,
                chat_request,
                http_request,
                chat_dispatch,
                response_id=f"resp_{uuid.uuid4().hex}",
                created_at=int(time.time()),
                stored_items=stored_items,
                store=store,
                owner=owner,
                compaction_codec=compaction_codec,
                compaction_request=compaction_request,
            )
        prepare_started_ns = time.perf_counter_ns()
        try:
            await prepare_backend_request(
                validated.engine,
                validated.generation_request,
            )
        except UpstreamClientError as error:
            return _chat_error(chat_error_from_upstream_client_error(error))
        except ValueError as error:
            return _request_error(str(error))
        except RuntimeError as error:
            return upstream_error(error)
        finally:
            if metrics is not None:
                metrics.record_preplacement_phase(
                    "responses",
                    "backend_prepare",
                    max(0, time.perf_counter_ns() - prepare_started_ns),
                )
        admission_started_ns = time.perf_counter_ns()
        try:
            bound = await backend_admission_upper_bound_async(
                validated.engine,
                validated.generation_request,
            )
        except ValueError as error:
            return _request_error(str(error))
        except RuntimeError as error:
            return upstream_error(error)
        admission_ns = max(0, time.perf_counter_ns() - admission_started_ns)
        reserve_started_ns = time.perf_counter_ns()
        admission = getattr(http_request.state, "tenant_admission", None)
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

        response_id = f"resp_{uuid.uuid4().hex}"
        created_at = int(time.time())
        if request.stream and not request.tools and not compaction_request:
            return sse_response(
                _live_text_events(
                    request,
                    validated,
                    response_id=response_id,
                    created_at=created_at,
                    stored_items=stored_items,
                    store=store,
                    owner=owner,
                    http_request=http_request,
                )
            )

        async def produce() -> tuple[list[dict], dict, str, dict | None]:
            try:
                if admission is not None:
                    admission.mark_dispatched()
                execution = await execute_chat(validated)
            except ChatRequestError as error:
                if error.execution is not None:
                    _record_execution(http_request, request, error.execution)
                raise _BufferedFailure.from_chat_error(error) from error
            except Exception as error:
                logger.exception("Responses API upstream generation failed")
                raise _BufferedFailure(
                    {
                        "message": f"upstream backend error ({type(error).__name__})",
                        "type": "upstream_error",
                        "code": "backend_error",
                    },
                    502,
                ) from error
            try:
                _validate_parallel_tool_calls(request, execution)
            except ChatRequestError as error:
                _record_execution(http_request, request, execution)
                raise _BufferedFailure.from_chat_error(error) from error
            _record_execution(http_request, request, execution)
            usage = _usage_payload(
                execution.result.prompt,
                execution.result.completions,
                execution.result.usage,
            )
            status, incomplete_details = _terminal_status(execution)
            if compaction_request:
                if status != "completed":
                    return [], usage, status, incomplete_details
                message = (
                    execution.response.choices[0].message.model_dump(mode="json")
                    if execution.response.choices
                    else {}
                )
                output = _compaction_output_from_message(
                    message, compaction_codec=compaction_codec, owner=owner
                )
                return output, usage, status, None
            output = _output_items(request, execution)
            _apply_terminal_item_status(output, status)
            return output, usage, status, incomplete_details

        if request.stream:
            return sse_response(
                _buffered_events(
                    request,
                    produce,
                    response_id=response_id,
                    created_at=created_at,
                    stored_items=stored_items,
                    store=store,
                    owner=owner,
                    compaction_request=compaction_request,
                )
            )
        try:
            output, usage, status, incomplete_details = await produce()
        except _BufferedFailure as failure:
            return failure.json_response()
        response = _response_envelope(
            request,
            response_id=response_id,
            created_at=created_at,
            status=status,
            output=output,
            usage=usage,
            incomplete_details=incomplete_details,
        )
        if request.store and not (compaction_request and not output):
            store.save(response_id, stored_items + output, owner=owner)
        return JSONResponse(content=response)

    return store
