"""Buffered generation served unary or as Responses SSE.

Engine turns that are not a live text stream (unary requests, tool turns,
compaction) run to completion here; ``buffered_events`` also streams buffered
AUTO turns. Temporary: WP-18 replaces it with the single emitter pipeline.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from kairyu.entrypoints.server.chat_service import (
    ChatRequestError,
    ValidatedChatRequest,
    execute_chat,
)
from kairyu.entrypoints.server.responses import events
from kairyu.entrypoints.server.responses.compaction import (
    CompactionCodec,
    compaction_output_from_message,
)
from kairyu.entrypoints.server.responses.envelope import (
    response_envelope,
    usage_payload,
    usage_payload_from_wire,
)
from kairyu.entrypoints.server.responses.errors import BufferedFailure
from kairyu.entrypoints.server.responses.events import responses_sse as _sse
from kairyu.entrypoints.server.responses.output import (
    apply_terminal_item_status,
    output_items,
    record_execution,
    terminal_status,
    validate_parallel_tool_calls,
)
from kairyu.entrypoints.server.responses.request import ResponsesRequest
from kairyu.entrypoints.server.responses.store import ResponseStore
from kairyu.entrypoints.server.sse_response import sse_response

if TYPE_CHECKING:
    from kairyu.entrypoints.server.tenancy import TenantAdmission

logger = logging.getLogger(__name__)


async def buffered_events(
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
    status, incomplete_details)`` or raises ``BufferedFailure``. The opening
    events flush immediately and repeated ``response.in_progress`` data events
    cover the generation window, so long orchestrated turns never trip client
    idle timeouts.
    """
    in_progress = response_envelope(
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
            done, _pending = await asyncio.wait({task}, timeout=events.HEARTBEAT_SECONDS)
            if done:
                break
            yield _sse("response.in_progress", sequence, response=in_progress)
            sequence += 1
    except BaseException:
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
        raise
    try:
        output, usage, status, incomplete_details = task.result()
    except BufferedFailure as failure:
        message = failure.payload.get("message") or "generation failed"
        yield _sse(
            "error",
            sequence,
            code=failure.payload.get("code") or "server_error",
            message=message,
            param=failure.payload.get("param"),
        )
        sequence += 1
        failed = response_envelope(
            request,
            response_id=response_id,
            created_at=created_at,
            status="failed",
            output=[],
            usage=usage_payload_from_wire(None),
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
    final_response = response_envelope(
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


async def engine_buffered_response(
    request: ResponsesRequest,
    validated: ValidatedChatRequest,
    http_request: Request,
    *,
    admission: TenantAdmission | None,
    response_id: str,
    created_at: int,
    stored_items: list[dict],
    store: ResponseStore,
    owner: str,
    compaction_codec: CompactionCodec,
    compaction_request: bool,
) -> Response:
    """Run one engine generation to completion; serve it unary or as buffered SSE."""

    async def produce() -> tuple[list[dict], dict, str, dict | None]:
        try:
            if admission is not None:
                admission.mark_dispatched()
            execution = await execute_chat(validated)
        except ChatRequestError as error:
            if error.execution is not None:
                record_execution(http_request, request, error.execution)
            raise BufferedFailure.from_chat_error(error) from error
        except Exception as error:
            logger.exception("Responses API upstream generation failed")
            raise BufferedFailure(
                {
                    "message": f"upstream backend error ({type(error).__name__})",
                    "type": "upstream_error",
                    "code": "backend_error",
                },
                502,
            ) from error
        try:
            validate_parallel_tool_calls(request, execution)
        except ChatRequestError as error:
            record_execution(http_request, request, execution)
            raise BufferedFailure.from_chat_error(error) from error
        record_execution(http_request, request, execution)
        usage = usage_payload(
            execution.result.prompt,
            execution.result.completions,
            execution.result.usage,
        )
        status, incomplete_details = terminal_status(execution)
        if compaction_request:
            if status != "completed":
                return [], usage, status, incomplete_details
            message = (
                execution.response.choices[0].message.model_dump(mode="json")
                if execution.response.choices
                else {}
            )
            output = compaction_output_from_message(
                message, compaction_codec=compaction_codec, owner=owner
            )
            return output, usage, status, None
        output = output_items(request, execution)
        apply_terminal_item_status(output, status)
        return output, usage, status, incomplete_details

    if request.stream:
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
    try:
        output, usage, status, incomplete_details = await produce()
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
