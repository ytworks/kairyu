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
from fastapi.routing import APIRoute
from starlette.routing import Match

from kairyu.engine.backend import (
    CacheHint,
    UpstreamClientError,
    backend_admission_upper_bound_async,
    backend_count_prompt_tokens_async,
    prepare_backend_request,
    render_tool_intent,
)
from kairyu.engine.prompt import prompt_text
from kairyu.entrypoints.server.chat_service import (
    ChatRequestError,
    chat_error_from_upstream_client_error,
    validate_chat_input_async,
    validate_chat_request_async,
)
from kairyu.entrypoints.server.errors import (
    openai_error_payload,
    sanitize_backend_error,
    wants_responses_envelope,
)
from kairyu.entrypoints.server.responses_auto import auto_response
from kairyu.entrypoints.server.responses_codec import (
    _COMPACTION_INSTRUCTION_ITEM,
    _CompactionCodec,
    _extract_compaction_trigger,
    reasoning_codec,
)
from kairyu.entrypoints.server.responses_engine import (
    engine_compaction,
    engine_stream,
    engine_unary,
)
from kairyu.entrypoints.server.responses_events import (
    ResponseEmitter,
    buffered_stream,
    failed_stream,
)
from kairyu.entrypoints.server.responses_items import (
    _canonical_input,
    _to_chat_request,
    _validate_function_outputs,
    input_listing,
)
from kairyu.entrypoints.server.responses_protocol import (
    ResponsesError,
    ResponsesInputTokensRequest,
    ResponsesRequest,
    _BufferedFailure,
    _validate_request_surface,
    context_overflow_error,
    is_context_overflow,
    not_found,
    parse_json_object,
    responses_error_response,
    validate_include,
    validate_model,
)
from kairyu.entrypoints.server.responses_store import (
    PendingSave,
    ResponseStore,
    list_page,
)
from kairyu.entrypoints.server.sse_response import sse_response

logger = logging.getLogger(__name__)
_FALLBACK_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]


def _owner(http_request: Request) -> str:
    return getattr(http_request.state, "tenant", None) or "default"


def _upstream_failure(error: BaseException) -> JSONResponse:
    # The full traceback stays server-side; the wire gets the sanitized class.
    logger.exception("Responses API upstream backend error")
    return _BufferedFailure(sanitize_backend_error(error), 502).json_response()


def _query_include(http_request: Request) -> list[str]:
    query = http_request.query_params
    include = [*query.getlist("include[]"), *query.getlist("include")]
    validate_include(include)
    return include


def _query_flag(http_request: Request, name: str) -> bool:
    value = http_request.query_params.get(name)
    if value in (None, "false", "0"):
        return False
    if value in ("true", "1"):
        return True
    raise ResponsesError(f"{name} must be true or false", param=name, code="invalid_value")


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
    reasoning_tokens = reasoning_codec(compaction_key)

    def reasoning_sealer(request: ResponsesRequest, owner: str):
        if "reasoning.encrypted_content" not in (request.include or ()):
            return None
        return lambda text: reasoning_tokens.encode(text, owner=owner)

    def continuation(previous_response_id: str | None, owner: str) -> list[dict]:
        if not previous_response_id:
            return []
        previous = store.get(previous_response_id, owner=owner)
        if previous is None:
            raise ResponsesError(
                f"Previous response with id '{previous_response_id}' not found.",
                param="previous_response_id",
                code="previous_response_not_found",
            )
        return previous

    @app.get("/v1/responses")
    async def responses_upgrade_required() -> JSONResponse:
        # Codex tries a WebSocket upgrade first when it targets the built-in
        # openai provider (the Harbor/Terminal-Bench shape). The upgrade
        # arrives as a GET; 426 makes Codex fall back to HTTPS immediately
        # and silently instead of burning its stream-retry budget.
        return JSONResponse(
            status_code=426,
            content={
                "error": openai_error_payload(
                    "WebSocket transport is not supported; retry over HTTPS",
                    code="upgrade_required",
                )
            },
        )

    @app.post("/v1/responses")
    async def responses(http_request: Request):
        try:
            request = validate_model(ResponsesRequest, await parse_json_object(http_request))
        except ResponsesError as error:
            return responses_error_response(error)
        http_request.state.model = request.model
        metrics = getattr(http_request.app.state, "metrics", None)
        ingress_ns = getattr(http_request.state, "placement_started_ns", None)
        if metrics is not None and type(ingress_ns) is int:
            metrics.record_preplacement_phase(
                "responses",
                "ingress_to_handler",
                max(0, time.perf_counter_ns() - ingress_ns),
            )
        owner = _owner(http_request)
        try:
            _validate_request_surface(request)
            engine = engines.get(request.model)
            orchestrated = (
                engine is None
                and chat_dispatch is not None
                and request.model in (orchestrated_models or ())
            )
            if engine is None and not orchestrated:
                raise ResponsesError(
                    f"model {request.model!r} not found",
                    param="model",
                    code="model_not_found",
                    status_code=404,
                )
            context = continuation(request.previous_response_id, owner)
        except ChatRequestError as error:
            return responses_error_response(error)
        validation_started_ns = time.perf_counter_ns()
        try:
            current_items = _canonical_input(
                request.input,
                compaction_codec=compaction_codec,
                owner=owner,
                reasoning_codec=reasoning_tokens,
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
            saver = PendingSave(
                store if request.store else None,
                owner,
                [] if compaction_request else work_items,
                input_listing(current_items),
            )
            _validate_function_outputs(work_items)
            prompt_items = (
                work_items + [_COMPACTION_INSTRUCTION_ITEM]
                if compaction_request
                else work_items
            )
            chat_request = _to_chat_request(
                request,
                prompt_items,
                compaction_codec=compaction_codec,
                owner=owner,
                # AUTO reasoning is intermediate stage output; replaying it into
                # the L2 conversation would grow every later prompt.
                replay_reasoning=not orchestrated,
            )
            if compaction_request:
                # The summary is a plain text turn: tools cannot help it and a
                # tool call in its place would break the client's compaction
                # collection, so the summarization call runs tool-free. Its
                # length follows max_output_tokens like any turn (omitted means
                # the remaining context).
                chat_request = chat_request.model_copy(
                    update={"tools": None, "tool_choice": None}
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
            return responses_error_response(error)
        finally:
            if metrics is not None:
                metrics.record_preplacement_phase(
                    "responses",
                    "request_validation",
                    max(0, time.perf_counter_ns() - validation_started_ns),
                )
        emitter = ResponseEmitter(
            request,
            response_id=f"resp_{uuid.uuid4().hex}",
            created_at=int(time.time()),
            seal_reasoning=reasoning_sealer(request, owner),
        )
        if orchestrated:
            return await auto_response(
                request,
                chat_request,
                http_request,
                chat_dispatch,
                emitter=emitter,
                saver=saver,
                owner=owner,
                compaction_codec=compaction_codec,
                compaction_request=compaction_request,
            )

        def overflowed() -> object:
            # Codex compacts only on an in-band context_length_exceeded; a
            # stream therefore opens even when dispatch never happened.
            if request.stream:
                return sse_response(failed_stream(emitter, context_overflow_error().payload()))
            return responses_error_response(context_overflow_error())

        prepare_started_ns = time.perf_counter_ns()
        try:
            await prepare_backend_request(
                validated.engine,
                validated.generation_request,
            )
        except UpstreamClientError as error:
            if is_context_overflow(error):
                return overflowed()
            return responses_error_response(chat_error_from_upstream_client_error(error))
        except ValueError as error:
            if is_context_overflow(error):
                return overflowed()
            return responses_error_response(ResponsesError(str(error)))
        except RuntimeError as error:
            return _upstream_failure(error)
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
            if is_context_overflow(error):
                return overflowed()
            return responses_error_response(ResponsesError(str(error)))
        except RuntimeError as error:
            return _upstream_failure(error)
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
                return responses_error_response(
                    ResponsesError(
                        f"tenant {owner!r} admission limit exceeded ({admission.reason})",
                        status_code=429,
                        error_type="rate_limit_error",
                        code="tenant_rate_limited",
                        headers={"Retry-After": "1"},
                    )
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

        if compaction_request:

            async def produce() -> tuple[list[dict], dict, str, dict | None]:
                return await engine_compaction(
                    request,
                    validated,
                    http_request=http_request,
                    compaction_codec=compaction_codec,
                    owner=owner,
                )

            if request.stream:
                return sse_response(buffered_stream(emitter, produce, saver))
            try:
                output, usage, status, incomplete_details = await produce()
            except _BufferedFailure as failure:
                return failure.json_response()
            for item in output:
                emitter.add_item(item)
            response, _frames = emitter.complete(status, usage, incomplete_details)
            if output:
                saver.commit(response)
            return JSONResponse(content=response)
        if request.stream:
            return sse_response(
                engine_stream(
                    request,
                    validated,
                    emitter=emitter,
                    saver=saver,
                    owner=owner,
                    http_request=http_request,
                )
            )
        try:
            response = await engine_unary(
                request, validated, emitter=emitter, http_request=http_request
            )
        except _BufferedFailure as failure:
            return failure.json_response()
        saver.commit(response)
        return JSONResponse(content=response)

    @app.post("/v1/responses/input_tokens")
    async def responses_input_tokens(http_request: Request) -> JSONResponse:
        owner = _owner(http_request)
        try:
            body = validate_model(
                ResponsesInputTokensRequest, await parse_json_object(http_request)
            )
            request = body.as_responses_request()
            engine = engines.get(request.model)
            if engine is None:
                if chat_dispatch is not None and request.model in (orchestrated_models or ()):
                    raise ResponsesError(
                        "input token counting is not available for orchestrated models",
                        param="model",
                        code="unsupported_value",
                    )
                raise ResponsesError(
                    f"model {request.model!r} not found",
                    param="model",
                    code="model_not_found",
                    status_code=404,
                )
            items = continuation(request.previous_response_id, owner) + _canonical_input(
                request.input,
                compaction_codec=compaction_codec,
                owner=owner,
                reasoning_codec=reasoning_tokens,
            )
            _validate_function_outputs(items)
            chat_request = _to_chat_request(
                request, items, compaction_codec=compaction_codec, owner=owner
            )
            validated_input = await validate_chat_input_async(
                chat_request,
                chat_templates,
                allow_multimodal=True,
                legacy_chat_models=legacy_chat_models,
            )
            text = prompt_text(
                render_tool_intent(
                    validated_input.prompt,
                    tools=tuple(chat_request.tools or ()),
                    tool_choice=chat_request.tool_choice,
                    tools_in_prompt=validated_input.tools_in_prompt,
                )
            )
            if text is None:
                raise ResponsesError(
                    "input token counting is not available for image input",
                    param="input",
                    code="unsupported_value",
                )
            input_tokens = await backend_count_prompt_tokens_async(engine, text)
            if input_tokens is None:
                raise ResponsesError(
                    f"model {request.model!r} does not support token counting",
                    param="model",
                    code="unsupported_value",
                )
        except ChatRequestError as error:
            return responses_error_response(error)
        except ValueError as error:
            return responses_error_response(ResponsesError(str(error)))
        return JSONResponse(
            content={"object": "response.input_tokens", "input_tokens": input_tokens}
        )

    @app.get("/v1/responses/{response_id}")
    async def retrieve_response(response_id: str, http_request: Request) -> JSONResponse:
        owner = _owner(http_request)
        try:
            include = _query_include(http_request)
            if _query_flag(http_request, "stream"):
                raise ResponsesError(
                    "streaming retrieval needs background responses, which are not supported",
                    param="stream",
                    code="unsupported_value",
                )
            response = store.response(response_id, owner=owner)
            if response is None:
                raise not_found(response_id)
        except ResponsesError as error:
            return responses_error_response(error)
        if "reasoning.encrypted_content" in include:
            for item in response["output"]:
                if item.get("type") == "reasoning":
                    text = "".join(part.get("text", "") for part in item.get("content") or ())
                    item["encrypted_content"] = reasoning_tokens.encode(text, owner=owner)
        return JSONResponse(content=response)

    @app.delete("/v1/responses/{response_id}")
    async def delete_response(response_id: str, http_request: Request) -> JSONResponse:
        if not store.delete(response_id, owner=_owner(http_request)):
            return responses_error_response(not_found(response_id))
        return JSONResponse(
            content={"id": response_id, "object": "response.deleted", "deleted": True}
        )

    @app.get("/v1/responses/{response_id}/input_items")
    async def list_input_items(response_id: str, http_request: Request) -> JSONResponse:
        try:
            _query_include(http_request)
            items = store.input_items(response_id, owner=_owner(http_request))
            if items is None:
                raise not_found(response_id)
            page = list_page(items, http_request.query_params)
        except ResponsesError as error:
            return responses_error_response(error)
        return JSONResponse(content=page)

    @app.post("/v1/responses/{response_id}/cancel")
    async def cancel_response(response_id: str, http_request: Request) -> JSONResponse:
        if not store.has(response_id, owner=_owner(http_request)):
            return responses_error_response(not_found(response_id))
        return responses_error_response(
            ResponsesError(
                "Only background responses can be cancelled, and this server does "
                "not run background responses.",
                code="unsupported_value",
            )
        )

    _add_unrouted_fallbacks(app)
    return store


def _add_unrouted_fallbacks(app: FastAPI) -> None:
    """OpenAI-shaped 404/405 for unrouted ``/v1/responses`` paths and methods.

    Registered after every Responses route: Starlette tries routes in order,
    so these catch-alls only see requests no Responses route fully matched.
    """

    routed = [
        route
        for route in app.router.routes
        if isinstance(route, APIRoute) and wants_responses_envelope(route.path)
    ]

    async def unrouted(http_request: Request) -> JSONResponse:
        method = http_request.method
        allowed = sorted(
            {
                allowed_method
                for route in routed
                if route.matches(http_request.scope)[0] is Match.PARTIAL
                for allowed_method in route.methods or ()
            }
        )
        if allowed:
            return JSONResponse(
                status_code=405,
                headers={"Allow": ", ".join(allowed)},
                content={
                    "error": openai_error_payload(
                        f"Method {method} is not allowed for {http_request.url.path}.",
                        code="method_not_allowed",
                    )
                },
            )
        return JSONResponse(
            status_code=404,
            content={
                "error": openai_error_payload(f"Invalid URL ({method} {http_request.url.path})")
            },
        )

    app.add_api_route(
        "/v1/responses",
        unrouted,
        methods=["PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    app.add_api_route(
        "/v1/responses/{unrouted_path:path}",
        unrouted,
        methods=_FALLBACK_METHODS,
        include_in_schema=False,
    )
