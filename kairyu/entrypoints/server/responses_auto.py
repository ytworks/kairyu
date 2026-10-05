"""AUTO-model Responses paths delegated to the Chat Completions handler."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import Request
from fastapi.responses import JSONResponse

from kairyu.entrypoints.server.errors import (
    openai_error_payload,
    responses_slow_down_payload,
)
from kairyu.entrypoints.server.protocol import ChatCompletionRequest, StreamOptions
from kairyu.entrypoints.server.responses_codec import (
    _compaction_output_from_message,
    _CompactionCodec,
)
from kairyu.entrypoints.server.responses_events import (
    _apply_terminal_item_status,
    _buffered_events,
    _sse,
    _terminal_status_for,
)
from kairyu.entrypoints.server.responses_items import _output_items_from_message
from kairyu.entrypoints.server.responses_protocol import (
    ResponsesRequest,
    _BufferedFailure,
    _response_envelope,
    _usage_payload_from_wire,
)
from kairyu.entrypoints.server.responses_store import PendingSave
from kairyu.entrypoints.server.sse_encode import ResponsesTextDeltaSSEEncoder
from kairyu.entrypoints.server.sse_response import sse_response

logger = logging.getLogger(__name__)


async def _relay_auto_chat_stream(
    request: ResponsesRequest,
    upstream,
    *,
    response_id: str,
    created_at: int,
    saver: PendingSave,
) -> AsyncIterator[str | bytes]:
    """Re-encode the orchestrated Chat Completions SSE stream as Responses SSE.

    ``upstream`` is this process's own chat-chunk stream for an AUTO model, so
    the frames are the internal wire format this release emits: ``: status``
    keep-alive comments, ``data: {chat chunk}`` frames, ``data: {"error": ...}``
    frames, and a final ``data: [DONE]``.
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    text_delta_encoder = ResponsesTextDeltaSSEEncoder(message_id)
    in_progress = _response_envelope(
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

    async def frames() -> AsyncIterator[str]:
        buffer = ""
        async for chunk in upstream:
            buffer += chunk.decode() if isinstance(chunk, bytes) else chunk
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                if frame:
                    yield frame
        if buffer:
            yield buffer

    try:
        yield _sse("response.created", sequence, response=in_progress)
        sequence += 1
        yield _sse("response.in_progress", sequence, response=in_progress)
        sequence += 1
        added_item = {
            "type": "message",
            "id": message_id,
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        }
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
        async for frame in frames():
            if frame.startswith(":"):
                # Orchestrator keep-alive comments stay comments on the
                # Responses stream (invisible to SDKs, reset idle timers).
                yield f"{frame}\n\n"
                continue
            if not frame.startswith("data:"):
                continue
            payload_text = frame[len("data:"):].strip()
            if payload_text == "[DONE]":
                break
            try:
                chunk_payload = json.loads(payload_text)
            except ValueError:
                continue
            if "error" in chunk_payload and "choices" not in chunk_payload:
                error_payload = chunk_payload["error"]
                break
            usage_value = chunk_payload.get("usage")
            if isinstance(usage_value, dict):
                wire_usage = usage_value
            for choice in chunk_payload.get("choices") or ():
                if choice.get("index", 0) != 0:
                    continue
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    yield text_delta_encoder.encode(sequence, content)
                    sequence += 1
                    text_parts.append(content)
    except Exception as error:
        logger.exception("Responses API orchestrated relay failed")
        error_payload = {"message": f"upstream backend error ({type(error).__name__})"}
    finally:
        aclose = getattr(upstream, "aclose", None)
        if aclose is not None:
            await aclose()

    text = "".join(text_parts)
    if error_payload is not None:
        message = error_payload.get("message") or "upstream backend error"
        yield _sse("error", sequence, code="server_error", message=message, param=None)
        sequence += 1
        failed_output = []
        if text:
            failed_output.append(
                {
                    "type": "message",
                    "id": message_id,
                    "role": "assistant",
                    "status": "incomplete",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                }
            )
        failed = _response_envelope(
            request,
            response_id=response_id,
            created_at=created_at,
            status="failed",
            output=failed_output,
            usage=_usage_payload_from_wire(wire_usage),
            error={"code": "server_error", "message": message},
        )
        yield _sse("response.failed", sequence, response=failed)
        return

    status, incomplete_details = _terminal_status_for((finish_reason,))
    output_item = {
        "type": "message",
        "id": message_id,
        "role": "assistant",
        "status": status,
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
                "logprobs": [],
            }
        ],
    }
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
    final_response = _response_envelope(
        request,
        response_id=response_id,
        created_at=created_at,
        status=status,
        output=[output_item],
        usage=_usage_payload_from_wire(wire_usage),
        incomplete_details=incomplete_details,
    )
    saver.commit(final_response)
    terminal_type = "response.completed" if status == "completed" else "response.incomplete"
    yield _sse(terminal_type, sequence, response=final_response)


async def _orchestrated_response(
    request: ResponsesRequest,
    chat_request: ChatCompletionRequest,
    http_request: Request,
    chat_dispatch,
    *,
    response_id: str,
    created_at: int,
    saver: PendingSave,
    owner: str,
    compaction_codec: _CompactionCodec,
    compaction_request: bool = False,
):
    """Serve an AUTO model by delegating to the Chat Completions handler.

    The chat handler owns the entire orchestration contract (validation,
    judge-first tenant reservation, admission, execution, public usage and
    metering), so this branch never runs the engine-only pipeline below and
    never records usage itself.
    """
    if request.stream and not request.tools and not compaction_request:
        live_request = chat_request.model_copy(
            update={"stream": True, "stream_options": StreamOptions(include_usage=True)}
        )
        delegated = await chat_dispatch(live_request, http_request)
        if isinstance(delegated, JSONResponse):
            return _rerendered_chat_error(delegated)
        return sse_response(
            _relay_auto_chat_stream(
                request,
                delegated.body_iterator,
                response_id=response_id,
                created_at=created_at,
                saver=saver,
            )
        )
    buffered_request = chat_request.model_copy(update={"stream": False})

    def outcome_from_payload(payload: dict) -> tuple[list[dict], dict, str, dict | None]:
        choices = payload.get("choices") or []
        message = (choices[0].get("message") if choices else None) or {}
        usage = _usage_payload_from_wire(payload.get("usage"))
        status, incomplete_details = _terminal_status_for(
            choice.get("finish_reason") for choice in choices
        )
        if compaction_request:
            if status != "completed":
                return [], usage, status, incomplete_details
            output = _compaction_output_from_message(
                message, compaction_codec=compaction_codec, owner=owner
            )
            return output, usage, status, None
        output = _output_items_from_message(request, message) if choices else []
        _apply_terminal_item_status(output, status)
        return output, usage, status, incomplete_details

    if request.stream:

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
                raise _BufferedFailure(
                    payload.get("error")
                    or {
                        "message": "upstream backend error",
                        "type": "upstream_error",
                        "code": "backend_error",
                    },
                    delegated.status_code,
                )
            return outcome_from_payload(payload)

        return sse_response(
            _buffered_events(
                request,
                produce,
                response_id=response_id,
                created_at=created_at,
                saver=saver,
                compaction_request=compaction_request,
            )
        )
    delegated = await chat_dispatch(buffered_request, http_request)
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
        return _rerendered_chat_error(delegated)
    try:
        output, usage, status, incomplete_details = outcome_from_payload(
            json.loads(bytes(delegated.body))
        )
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
    if not (compaction_request and not output):
        saver.commit(response)
    return JSONResponse(content=response)


def _rerendered_chat_error(response: JSONResponse) -> JSONResponse:
    """Re-render a delegated chat error in the Responses envelope.

    A per-tenant concurrency rejection (``in_flight``) is transient, so it
    becomes the 503 ``slow_down`` that Codex retries; other errors keep their
    status and gain ``param``.
    """

    try:
        error = json.loads(bytes(response.body)).get("error") or {}
    except (AttributeError, ValueError):
        error = {}
    message = error.get("message") or "upstream backend error"
    retry_after = response.headers.get("retry-after")
    if response.status_code == 429 and message.endswith("(in_flight)"):
        return JSONResponse(
            status_code=503,
            content=responses_slow_down_payload(message),
            headers={"Retry-After": retry_after or "1"},
        )
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
