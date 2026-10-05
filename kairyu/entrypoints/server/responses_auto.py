"""AUTO-model Responses paths delegated to the Chat Completions handler.

The chat handler owns the entire orchestration contract (validation,
judge-first tenant reservation, admission, execution, public usage and
metering), so these paths never run the engine pipeline and never record usage
themselves. Text streams relay the orchestrator's chat chunks; tool-bearing
streams use the in-process raw tool-stream sentinel (#573) and enforce the tool
gates here, like the Messages adapter; unary turns map the chat JSON, whose
content keeps the prose beside its tool calls (#619).
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import AsyncIterator

from fastapi import Request
from fastapi.responses import JSONResponse

from kairyu.entrypoints.chat_template import ToolCallProtocol
from kairyu.entrypoints.server.chat_service import ChatRequestError, _normalize_tool_choice
from kairyu.entrypoints.server.errors import openai_error_payload
from kairyu.entrypoints.server.middleware import _ANTHROPIC_INTERNAL_TOOL_STREAM_STATE_KEY
from kairyu.entrypoints.server.protocol import ChatCompletionRequest, StreamOptions
from kairyu.entrypoints.server.responses_codec import (
    _compaction_output_from_message,
    _CompactionCodec,
)
from kairyu.entrypoints.server.responses_engine import terminal_status
from kairyu.entrypoints.server.responses_events import (
    OutputAssembler,
    ReasoningSplit,
    ResponseEmitter,
    buffered_stream,
    failed_stream,
    heartbeat_tick,
)
from kairyu.entrypoints.server.responses_protocol import (
    ResponsesRequest,
    _BufferedFailure,
    _usage_payload_from_wire,
    context_overflow_error,
    overflow_text,
    responses_error_response,
)
from kairyu.entrypoints.server.responses_store import PendingSave
from kairyu.entrypoints.server.responses_tools import _namespace_names
from kairyu.entrypoints.server.sse_keepalive import iter_with_idle_markers, sse_frames
from kairyu.entrypoints.server.sse_response import sse_response
from kairyu.entrypoints.server.tool_stream import tool_stream_scanner_for

logger = logging.getLogger(__name__)


def _status(finish_reason: str | None) -> tuple[str, dict | None]:
    # An orchestrated turn's length stop may be a stage cap, not the context:
    # it stays incomplete (documented).
    return terminal_status(finish_reason, cap_omitted=False, context_bounded=False)


def _error_of(response: JSONResponse) -> dict:
    try:
        return json.loads(bytes(response.body)).get("error") or {}
    except (AttributeError, ValueError):
        return {}


def _rerendered_chat_error(response: JSONResponse) -> JSONResponse:
    """Re-render a delegated chat error in the Responses envelope (with param)."""

    error = _error_of(response)
    message = error.get("message") or "upstream backend error"
    retry_after = response.headers.get("retry-after")
    code = error.get("code")
    payload = openai_error_payload(
        message,
        error_type=error.get("type") or "invalid_request_error",
        code=code,
        param="model" if code == "model_not_found" else None,
    )
    return JSONResponse(
        status_code=response.status_code,
        content={"error": payload},
        headers={"Retry-After": retry_after} if retry_after else None,
    )


def _delegated_failure(request: ResponsesRequest, emitter: ResponseEmitter, response):
    """A delegated chat error: in-band when it is an overflow on a stream."""

    if overflow_text(_error_of(response).get("message")):
        if request.stream:
            return sse_response(failed_stream(emitter, context_overflow_error().payload()))
        return responses_error_response(context_overflow_error())
    return _rerendered_chat_error(response)


async def _relay(
    upstream,
    *,
    emitter: ResponseEmitter,
    assembler: OutputAssembler,
    saver: PendingSave,
) -> AsyncIterator[str | bytes]:
    """Re-encode this process's AUTO chat-chunk stream as Responses SSE.

    The frames are the internal wire format: ``: status`` comments (forwarded;
    they do not count as data for Codex, so data heartbeats still run),
    ``data: {chat chunk}``, ``data: {"error": ...}``, and ``data: [DONE]``.
    """

    split = ReasoningSplit()
    finish_reason: str | None = None
    wire_usage: dict | None = None
    error_payload: dict | None = None
    try:
        for frame in emitter.start():
            yield frame
        frames = iter_with_idle_markers(sse_frames(upstream), heartbeat_tick())
        async with contextlib.aclosing(frames):
            async for frame in frames:
                if frame is not None and frame.startswith(":"):
                    yield f"{frame}\n\n"
                elif frame is not None and frame.startswith("data:"):
                    payload_text = frame[len("data:") :].strip()
                    if payload_text == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload_text)
                    except ValueError:
                        continue
                    if "error" in chunk and "choices" not in chunk:
                        error_payload = chunk["error"] or {}
                        break
                    if isinstance(chunk.get("usage"), dict):
                        wire_usage = chunk["usage"]
                    for choice in chunk.get("choices") or ():
                        if choice.get("index", 0) != 0:
                            continue
                        finish_reason = choice.get("finish_reason") or finish_reason
                        delta = choice.get("delta") or {}
                        remote = delta.get("reasoning_content")
                        if isinstance(remote, str):
                            for out in assembler.reasoning(remote):
                                yield out
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            reasoning, text = split.feed(None, content, final=False)
                            for out in assembler.reasoning(reasoning) + assembler.content(text):
                                yield out
                    if assembler.failure is not None:
                        break
                if emitter.heartbeat_due():
                    yield emitter.heartbeat()
    except Exception as error:
        logger.exception("Responses API orchestrated relay failed")
        error_payload = {"message": f"upstream backend error ({type(error).__name__})"}
    finally:
        aclose = getattr(upstream, "aclose", None)
        if aclose is not None:
            await aclose()

    usage = _usage_payload_from_wire(wire_usage)
    if error_payload is not None:
        message = error_payload.get("message") or "upstream backend error"
        payload = (
            context_overflow_error().payload()
            if overflow_text(message)
            else {"message": message, "type": "server_error", "code": "server_error"}
        )
        _envelope, out = emitter.fail(payload, usage)
        for frame in out:
            yield frame
        return
    if assembler.failure is None:
        reasoning, text = split.feed(None, "", final=True)
        for out in assembler.reasoning(reasoning) + assembler.content(text, final=True):
            yield out
    status, details = _status(finish_reason)
    closing = assembler.finish(incomplete=status == "incomplete")
    if assembler.failure is not None:
        _envelope, out = emitter.fail(assembler.failure, usage)
        for frame in out:
            yield frame
        return
    envelope, terminal = emitter.complete(status, usage, details)
    for frame in closing + terminal:
        yield frame
    saver.commit(envelope)


async def auto_response(
    request: ResponsesRequest,
    chat_request: ChatCompletionRequest,
    http_request: Request,
    chat_dispatch,
    *,
    emitter: ResponseEmitter,
    saver: PendingSave,
    owner: str,
    compaction_codec: _CompactionCodec,
    compaction_request: bool = False,
):
    """Serve an AUTO model by delegating to the Chat Completions handler."""

    if compaction_request:
        return await _auto_compaction(
            request,
            chat_request,
            http_request,
            chat_dispatch,
            emitter=emitter,
            saver=saver,
            owner=owner,
            compaction_codec=compaction_codec,
        )
    if request.stream:
        tools_active = bool(chat_request.tools) and chat_request.tool_choice != "none"
        try:
            choice = _normalize_tool_choice(chat_request) if tools_active else None
        except ChatRequestError as error:
            return responses_error_response(error)
        if chat_request.tools:
            # Without the sentinel the chat handler buffers every tool-bearing
            # stream to run its pre-SSE gates; this adapter gates the raw
            # stream itself (the flag is unreachable from the wire).
            setattr(http_request.state, _ANTHROPIC_INTERNAL_TOOL_STREAM_STATE_KEY, True)
        delegated = await chat_dispatch(
            chat_request.model_copy(
                update={"stream": True, "stream_options": StreamOptions(include_usage=True)}
            ),
            http_request,
        )
        if isinstance(delegated, JSONResponse):
            return _delegated_failure(request, emitter, delegated)
        assembler = OutputAssembler(
            emitter,
            scanner=(
                tool_stream_scanner_for(ToolCallProtocol.GENERIC, chat_request.tools, choice)
                if choice is not None
                else None
            ),
            tool_choice=choice,
            parallel=False if chat_request.parallel_tool_calls is False else None,
            namespaces=_namespace_names(request.tools),
        )
        return sse_response(
            _relay(
                delegated.body_iterator,
                emitter=emitter,
                assembler=assembler,
                saver=saver,
            )
        )
    delegated = await chat_dispatch(chat_request.model_copy(update={"stream": False}), http_request)
    if not isinstance(delegated, JSONResponse):
        logger.error("unexpected non-JSON chat dispatch reply for a unary request")
        return _BufferedFailure(
            {
                "message": "unexpected non-JSON chat dispatch reply",
                "type": "upstream_error",
                "code": "backend_error",
            },
            502,
        ).json_response()
    if delegated.status_code != 200:
        return _delegated_failure(request, emitter, delegated)
    payload = json.loads(bytes(delegated.body))
    choices = payload.get("choices") or []
    message = (choices[0].get("message") if choices else None) or {}
    assembler = OutputAssembler(emitter, namespaces=_namespace_names(request.tools))
    reasoning, content = ReasoningSplit().feed(None, message.get("content") or "", final=True)
    remote = message.get("reasoning_content")
    assembler.reasoning((remote if isinstance(remote, str) else "") + reasoning)
    assembler.content(content)
    assembler.parsed_calls(message.get("tool_calls") or [])
    status, details = _status(choices[0].get("finish_reason") if choices else None)
    assembler.finish(incomplete=status == "incomplete")
    envelope, _frames = emitter.complete(
        status, _usage_payload_from_wire(payload.get("usage")), details
    )
    saver.commit(envelope)
    return JSONResponse(content=envelope)


async def _auto_compaction(
    request: ResponsesRequest,
    chat_request: ChatCompletionRequest,
    http_request: Request,
    chat_dispatch,
    *,
    emitter: ResponseEmitter,
    saver: PendingSave,
    owner: str,
    compaction_codec: _CompactionCodec,
):
    buffered_request = chat_request.model_copy(update={"stream": False})

    async def produce() -> tuple[list[dict], dict, str, dict | None]:
        delegated = await chat_dispatch(buffered_request, http_request)
        if not isinstance(delegated, JSONResponse):
            raise _BufferedFailure(
                {
                    "message": "unexpected non-JSON chat dispatch reply",
                    "type": "upstream_error",
                    "code": "backend_error",
                },
                502,
            )
        payload = json.loads(bytes(delegated.body))
        if delegated.status_code != 200:
            error = payload.get("error") or {}
            if overflow_text(error.get("message")):
                raise _BufferedFailure(context_overflow_error().payload(), 400)
            raise _BufferedFailure(
                error
                or {
                    "message": "upstream backend error",
                    "type": "upstream_error",
                    "code": "backend_error",
                },
                delegated.status_code,
            )
        choices = payload.get("choices") or []
        usage = _usage_payload_from_wire(payload.get("usage"))
        status, details = _status(choices[0].get("finish_reason") if choices else None)
        if status != "completed":
            return [], usage, status, details
        message = (choices[0].get("message") if choices else None) or {}
        _reasoning, summary = ReasoningSplit().feed(None, message.get("content") or "", final=True)
        output = _compaction_output_from_message(
            {"content": summary}, compaction_codec=compaction_codec, owner=owner
        )
        return output, usage, status, None

    if request.stream:
        return sse_response(buffered_stream(emitter, produce, saver))
    try:
        output, usage, status, details = await produce()
    except _BufferedFailure as failure:
        return failure.json_response()
    for item in output:
        emitter.add_item(item)
    envelope, _frames = emitter.complete(status, usage, details)
    if output:
        saver.commit(envelope)
    return JSONResponse(content=envelope)
