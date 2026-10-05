"""Responses SSE event encoding and the buffered event stream."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator

from kairyu.entrypoints.server.chat_service import ExecutedChat
from kairyu.entrypoints.server.responses_protocol import (
    ResponsesRequest,
    _BufferedFailure,
    _response_envelope,
    _usage_payload_from_wire,
)
from kairyu.entrypoints.server.responses_store import ResponseStore
from kairyu.sse import escape_json_line_separators

# Buffered generation can run for minutes on orchestrated models while the
# strictest known client budget (Codex) closes idle SSE streams at 300s.
_BUFFERED_KEEPALIVE_SECONDS = 15.0


def _sse(event_type: str, sequence_number: int, **payload) -> str:
    event = {"type": event_type, "sequence_number": sequence_number, **payload}
    serialized = escape_json_line_separators(
        json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    )
    return (
        f"event: {event_type}\n"
        f"data: {serialized}\n\n"
    )


def _terminal_status(execution: ExecutedChat) -> tuple[str, dict | None]:
    return _terminal_status_for(
        completion.finish_reason for completion in execution.result.completions
    )


def _terminal_status_for(finish_reasons) -> tuple[str, dict | None]:
    if any(reason in {"length", "max_tokens"} for reason in finish_reasons):
        return "incomplete", {"reason": "max_output_tokens"}
    return "completed", None


def _apply_terminal_item_status(output: list[dict], status: str) -> None:
    if status != "incomplete":
        return
    for item in output:
        if item["type"] == "message":
            item["status"] = "incomplete"


async def _buffered_events(
    request: ResponsesRequest,
    produce,
    *,
    response_id: str,
    created_at: int,
    stored_items: list[dict],
    store: ResponseStore,
    owner: str,
    compaction_request: bool = False,
) -> AsyncIterator[str]:
    """Stream buffered generation as Responses SSE without going silent.

    ``produce`` runs the whole generation and returns ``(output, usage,
    status, incomplete_details)`` or raises ``_BufferedFailure``. The opening
    events flush immediately and keep-alive comments cover the generation
    window, so long orchestrated turns never trip client idle timeouts.
    """
    in_progress = _response_envelope(
        request,
        response_id=response_id,
        created_at=created_at,
        status="in_progress",
        output=[],
        usage=None,
    )
    sequence = 0
    yield _sse("response.created", sequence, response=in_progress)
    sequence += 1
    yield _sse("response.in_progress", sequence, response=in_progress)
    sequence += 1
    task = asyncio.ensure_future(produce())
    try:
        while True:
            done, _pending = await asyncio.wait(
                {task}, timeout=_BUFFERED_KEEPALIVE_SECONDS
            )
            if done:
                break
            yield ": keep-alive\n\n"
    except BaseException:
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
        raise
    try:
        output, usage, status, incomplete_details = task.result()
    except _BufferedFailure as failure:
        message = failure.payload.get("message") or "generation failed"
        yield _sse(
            "error",
            sequence,
            code=failure.payload.get("code") or "server_error",
            message=message,
            param=None,
        )
        sequence += 1
        failed = _response_envelope(
            request,
            response_id=response_id,
            created_at=created_at,
            status="failed",
            output=[],
            usage=_usage_payload_from_wire(None),
            error={
                "code": failure.payload.get("code") or "server_error",
                "message": message,
            },
        )
        yield _sse("response.failed", sequence, response=failed)
        return
    for output_index, final_item in enumerate(output):
        if final_item["type"] == "message":
            text = final_item["content"][0]["text"]
            added_item = {
                **final_item,
                "status": "in_progress",
                "content": [],
            }
            yield _sse(
                "response.output_item.added",
                sequence,
                output_index=output_index,
                item=added_item,
            )
            sequence += 1
            empty_part = {
                "type": "output_text",
                "text": "",
                "annotations": [],
                "logprobs": [],
            }
            yield _sse(
                "response.content_part.added",
                sequence,
                item_id=final_item["id"],
                output_index=output_index,
                content_index=0,
                part=empty_part,
            )
            sequence += 1
            if text:
                yield _sse(
                    "response.output_text.delta",
                    sequence,
                    item_id=final_item["id"],
                    output_index=output_index,
                    content_index=0,
                    delta=text,
                    logprobs=[],
                )
                sequence += 1
            yield _sse(
                "response.output_text.done",
                sequence,
                item_id=final_item["id"],
                output_index=output_index,
                content_index=0,
                text=text,
                logprobs=[],
            )
            sequence += 1
            yield _sse(
                "response.content_part.done",
                sequence,
                item_id=final_item["id"],
                output_index=output_index,
                content_index=0,
                part=final_item["content"][0],
            )
            sequence += 1
        elif final_item["type"] != "function_call":
            # Non-message, non-call output (e.g. a compaction item): the item
            # lifecycle alone, no content-part or argument events.
            yield _sse(
                "response.output_item.added",
                sequence,
                output_index=output_index,
                item={**final_item, "status": "in_progress"},
            )
            sequence += 1
        else:
            added_item = {**final_item, "arguments": "", "status": "in_progress"}
            yield _sse(
                "response.output_item.added",
                sequence,
                output_index=output_index,
                item=added_item,
            )
            sequence += 1
            if final_item["arguments"]:
                yield _sse(
                    "response.function_call_arguments.delta",
                    sequence,
                    item_id=final_item["id"],
                    output_index=output_index,
                    delta=final_item["arguments"],
                )
                sequence += 1
            yield _sse(
                "response.function_call_arguments.done",
                sequence,
                item_id=final_item["id"],
                output_index=output_index,
                arguments=final_item["arguments"],
                name=final_item["name"],
            )
            sequence += 1
        yield _sse(
            "response.output_item.done",
            sequence,
            output_index=output_index,
            item=final_item,
        )
        sequence += 1
    final_response = _response_envelope(
        request,
        response_id=response_id,
        created_at=created_at,
        status=status,
        output=output,
        usage=usage,
        incomplete_details=incomplete_details,
    )
    if request.store and not (compaction_request and not output):
        # An incomplete compaction produced no replacement context; storing an
        # empty continuation entry would silently blank a thread.
        store.save(response_id, stored_items + output, owner=owner)
    terminal_type = "response.completed" if status == "completed" else "response.incomplete"
    yield _sse(terminal_type, sequence, response=final_response)
