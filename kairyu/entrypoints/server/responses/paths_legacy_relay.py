"""AUTO (orchestrated) models served through the Chat Completions handler.

The live text relay re-encodes the orchestrated chat stream; every other AUTO
turn is buffered. Temporary: WP-18 replaces both with the single emitter
pipeline.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import Request
from fastapi.responses import JSONResponse

from kairyu.entrypoints.server import stream_util
from kairyu.entrypoints.server.errors import upstream_error
from kairyu.entrypoints.server.protocol import ChatCompletionRequest, StreamOptions
from kairyu.entrypoints.server.responses import events
from kairyu.entrypoints.server.responses.compaction import (
    CompactionCodec,
    compaction_output_from_message,
)
from kairyu.entrypoints.server.responses.envelope import (
    message_item,
    open_message_snapshot,
    response_envelope,
    usage_payload_from_wire,
)
from kairyu.entrypoints.server.responses.errors import (
    BufferedFailure,
    delegated_error,
    delegated_failure,
)
from kairyu.entrypoints.server.responses.events import responses_sse as _sse
from kairyu.entrypoints.server.responses.output import (
    apply_terminal_item_status,
    output_items_from_message,
    terminal_status_for,
)
from kairyu.entrypoints.server.responses.paths_legacy_buffered import buffered_events
from kairyu.entrypoints.server.responses.request import ResponsesRequest
from kairyu.entrypoints.server.responses.store import ResponseStore
from kairyu.entrypoints.server.sse_encode import ResponsesTextDeltaSSEEncoder
from kairyu.entrypoints.server.sse_response import sse_response

logger = logging.getLogger(__name__)


async def _relay_auto_chat_stream(
    request: ResponsesRequest,
    upstream,
    *,
    response_id: str,
    created_at: int,
    stored_items: list[dict],
    store: ResponseStore,
    owner: str,
) -> AsyncIterator[str | bytes]:
    """Re-encode the orchestrated Chat Completions SSE stream as Responses SSE.

    ``upstream`` is this process's own chat-chunk stream for an AUTO model, so
    the frames are the internal wire format this release emits: ``: status``
    keep-alive comments, ``data: {chat chunk}`` frames, ``data: {"error": ...}``
    frames, and a final ``data: [DONE]``. Status comments are not relayed:
    liveness is the repeated ``response.in_progress`` data event.
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    text_delta_encoder = ResponsesTextDeltaSSEEncoder(message_id)
    in_progress = response_envelope(
        request,
        response_id=response_id,
        created_at=created_at,
        status="in_progress",
        output=[],
        usage=None,
    )
    sequence = 0
    text_parts: list[str] = []
    finish_reason: str | None = None
    wire_usage: dict | None = None
    error_payload: dict | None = None
    heartbeat = stream_util.DataHeartbeat(events.HEARTBEAT_SECONDS)
    frames = stream_util.iter_with_idle_markers(
        stream_util.sse_frames(upstream), idle_seconds=events.HEARTBEAT_SECONDS
    )
    try:
        yield _sse("response.created", sequence, response=in_progress)
        sequence += 1
        yield _sse("response.in_progress", sequence, response=in_progress)
        sequence += 1
        added_item = {**message_item(message_id, "in_progress", ""), "content": []}
        yield _sse(
            "response.output_item.added",
            sequence,
            output_index=0,
            item=added_item,
        )
        sequence += 1
        yield _sse(
            "response.content_part.added",
            sequence,
            item_id=message_id,
            output_index=0,
            content_index=0,
            part={"type": "output_text", "text": "", "annotations": [], "logprobs": []},
        )
        sequence += 1
        heartbeat.mark()
        async for frame in frames:
            if frame is not stream_util.IDLE_MARKER and frame.startswith("data:"):
                payload_text = frame[len("data:"):].strip()
                if payload_text == "[DONE]":
                    break
                try:
                    chunk_payload = json.loads(payload_text)
                except ValueError:
                    chunk_payload = {}
                if "error" in chunk_payload and "choices" not in chunk_payload:
                    error_payload = chunk_payload["error"]
                    break
                if isinstance(chunk_payload.get("usage"), dict):
                    wire_usage = chunk_payload["usage"]
                content, reason = stream_util.primary_chat_delta(chunk_payload)
                finish_reason = reason or finish_reason
                if content:
                    yield text_delta_encoder.encode(sequence, content)
                    sequence += 1
                    heartbeat.mark()
                    text_parts.append(content)
            if heartbeat.due():
                snapshot = open_message_snapshot(
                    request,
                    response_id=response_id,
                    created_at=created_at,
                    message_id=message_id,
                    text="".join(text_parts),
                )
                yield _sse("response.in_progress", sequence, response=snapshot)
                sequence += 1
                heartbeat.mark()
    except Exception as error:
        logger.exception("Responses API orchestrated relay failed")
        error_payload = {"message": f"upstream backend error ({type(error).__name__})"}
    finally:
        # Stop the frame pump first: closing a generator that is running raises.
        await frames.aclose()
        aclose = getattr(upstream, "aclose", None)
        if aclose is not None:
            await aclose()

    text = "".join(text_parts)
    if error_payload is not None:
        message = error_payload.get("message") or "upstream backend error"
        yield _sse("error", sequence, code="server_error", message=message, param=None)
        sequence += 1
        failed_output = [message_item(message_id, "incomplete", text)] if text else []
        failed = response_envelope(
            request,
            response_id=response_id,
            created_at=created_at,
            status="failed",
            output=failed_output,
            usage=usage_payload_from_wire(wire_usage),
            error={"code": "server_error", "message": message},
        )
        yield _sse("response.failed", sequence, response=failed)
        return

    status, incomplete_details = terminal_status_for((finish_reason,))
    output_item = message_item(message_id, status, text)
    yield _sse(
        "response.output_text.done",
        sequence,
        item_id=message_id,
        output_index=0,
        content_index=0,
        text=text,
        logprobs=[],
    )
    sequence += 1
    yield _sse(
        "response.content_part.done",
        sequence,
        item_id=message_id,
        output_index=0,
        content_index=0,
        part=output_item["content"][0],
    )
    sequence += 1
    yield _sse(
        "response.output_item.done",
        sequence,
        output_index=0,
        item=output_item,
    )
    sequence += 1
    final_response = response_envelope(
        request,
        response_id=response_id,
        created_at=created_at,
        status=status,
        output=[output_item],
        usage=usage_payload_from_wire(wire_usage),
        incomplete_details=incomplete_details,
    )
    if request.store:
        store.save(response_id, stored_items + [output_item], owner=owner)
    terminal_type = "response.completed" if status == "completed" else "response.incomplete"
    yield _sse(terminal_type, sequence, response=final_response)


async def orchestrated_response(
    request: ResponsesRequest,
    chat_request: ChatCompletionRequest,
    http_request: Request,
    chat_dispatch,
    *,
    response_id: str,
    created_at: int,
    stored_items: list[dict],
    store: ResponseStore,
    owner: str,
    compaction_codec: CompactionCodec,
    compaction_request: bool = False,
):
    """Serve an AUTO model by delegating to the Chat Completions handler.

    The chat handler owns the entire orchestration contract (validation,
    judge-first tenant reservation, admission, execution, public usage and
    metering), so this branch never runs the engine-only pipeline below and
    never records usage itself.
    """
    admission = getattr(http_request.state, "tenant_admission", None)
    if request.stream and not request.tools and not compaction_request:
        live_request = chat_request.model_copy(
            update={"stream": True, "stream_options": StreamOptions(include_usage=True)}
        )
        delegated = await chat_dispatch(live_request, http_request)
        if isinstance(delegated, JSONResponse):
            return delegated_failure(request, delegated, admission)
        return sse_response(
            _relay_auto_chat_stream(
                request,
                delegated.body_iterator,
                response_id=response_id,
                created_at=created_at,
                stored_items=stored_items,
                store=store,
                owner=owner,
            )
        )
    buffered_request = chat_request.model_copy(update={"stream": False})

    def outcome_from_payload(payload: dict) -> tuple[list[dict], dict, str, dict | None]:
        choices = payload.get("choices") or []
        message = (choices[0].get("message") if choices else None) or {}
        usage = usage_payload_from_wire(payload.get("usage"))
        status, incomplete_details = terminal_status_for(
            choice.get("finish_reason") for choice in choices
        )
        if compaction_request:
            if status != "completed":
                return [], usage, status, incomplete_details
            output = compaction_output_from_message(
                message, compaction_codec=compaction_codec, owner=owner
            )
            return output, usage, status, None
        output = output_items_from_message(request, message) if choices else []
        apply_terminal_item_status(output, status)
        return output, usage, status, incomplete_details

    if request.stream:

        async def produce() -> tuple[list[dict], dict, str, dict | None]:
            delegated = await chat_dispatch(buffered_request, http_request)
            if not isinstance(delegated, JSONResponse):
                raise BufferedFailure.from_payload(
                    {
                        "message": "unexpected non-JSON chat dispatch reply",
                        "type": "upstream_error",
                        "code": "backend_error",
                    },
                    502,
                )
            payload = json.loads(bytes(delegated.body))
            if delegated.status_code != 200:
                raise BufferedFailure(
                    delegated_error(delegated.status_code, payload, admission)
                )
            return outcome_from_payload(payload)

        return sse_response(
            buffered_events(
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
    delegated = await chat_dispatch(buffered_request, http_request)
    if not isinstance(delegated, JSONResponse):
        return upstream_error(RuntimeError("unexpected non-JSON chat dispatch reply"))
    if delegated.status_code != 200:
        return delegated_failure(request, delegated, admission)
    try:
        output, usage, status, incomplete_details = outcome_from_payload(
            json.loads(bytes(delegated.body))
        )
    except BufferedFailure as failure:
        return failure.json_response()
    response = response_envelope(
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
