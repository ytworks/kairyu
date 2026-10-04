"""Live engine text stream (no tools, no compaction) re-encoded as Responses SSE.

Temporary: WP-17a replaces it with the single emitter pipeline.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import Request

from kairyu.engine.backend import GenerationResult
from kairyu.entrypoints.server import stream_util
from kairyu.entrypoints.server.chat_service import ValidatedChatRequest
from kairyu.entrypoints.server.error_classifier import classify_request_error
from kairyu.entrypoints.server.metering import stream_usage_owner_from_state
from kairyu.entrypoints.server.responses import events
from kairyu.entrypoints.server.responses.envelope import (
    message_item,
    open_message_snapshot,
    response_envelope,
    usage_payload,
)
from kairyu.entrypoints.server.responses.events import responses_sse as _sse
from kairyu.entrypoints.server.responses.request import ResponsesRequest
from kairyu.entrypoints.server.responses.store import ResponseStore
from kairyu.entrypoints.server.sse_encode import ResponsesTextDeltaSSEEncoder

logger = logging.getLogger(__name__)


async def live_text_events(
    request: ResponsesRequest,
    validated: ValidatedChatRequest,
    *,
    response_id: str,
    created_at: int,
    stored_items: list[dict],
    store: ResponseStore,
    owner: str,
    http_request: Request,
) -> AsyncIterator[str | bytes]:
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
    sent = 0
    streamed: list[str] = []
    last: GenerationResult | None = None
    usage_owner = stream_usage_owner_from_state(
        http_request.app.state,
        tenant=owner,
        model=request.model,
        prompt=validated.generation_request.prompt,
        reservation=getattr(http_request.state, "tenant_admission", None),
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
        empty_part = message_item(message_id, "in_progress", "")["content"][0]
        yield _sse(
            "response.content_part.added",
            sequence,
            item_id=message_id,
            output_index=0,
            content_index=0,
            part=empty_part,
        )
        sequence += 1
        heartbeat = stream_util.DataHeartbeat(events.HEARTBEAT_SECONDS)
        partials = stream_util.iter_with_idle_markers(
            validated.engine.stream(validated.generation_request),
            idle_seconds=events.HEARTBEAT_SECONDS,
        )
        try:
            usage_owner.mark_dispatched()
            async for partial in partials:
                if partial is not stream_util.IDLE_MARKER:
                    last = partial
                    usage_owner.observe(partial.usage, partial.completions)
                    completion = min(
                        partial.completions, key=lambda item: item.index, default=None
                    )
                    delta = ""
                    if completion is not None:
                        delta, sent = completion.delta_after(sent)
                    if delta:
                        if type(delta) is str:
                            yield text_delta_encoder.encode(sequence, delta)
                        else:
                            yield _sse(
                                "response.output_text.delta",
                                sequence,
                                item_id=message_id,
                                output_index=0,
                                content_index=0,
                                delta=delta,
                                logprobs=[],
                            )
                        sequence += 1
                        streamed.append(delta)
                        heartbeat.mark()
                        continue
                if heartbeat.due():
                    snapshot = open_message_snapshot(
                        request,
                        response_id=response_id,
                        created_at=created_at,
                        message_id=message_id,
                        text="".join(streamed),
                    )
                    yield _sse("response.in_progress", sequence, response=snapshot)
                    sequence += 1
                    heartbeat.mark()
        except Exception as error:
            logger.exception("Responses API upstream stream failed")
            classified = classify_request_error(error, "responses")
            code = classified.code if classified else "server_error"
            safe_message = (
                classified.message
                if classified
                else f"upstream backend error ({type(error).__name__})"
            )
            failed_completions = last.completions if last is not None else ()
            failed_completion = min(
                failed_completions,
                key=lambda item: item.index,
                default=None,
            )
            failed_output = []
            if failed_completion is not None:
                failed_output.append(
                    {
                        "type": "message",
                        "id": message_id,
                        "role": "assistant",
                        "status": "incomplete",
                        "content": [
                            {
                                "type": "output_text",
                                "text": failed_completion.text,
                                "annotations": [],
                                "logprobs": [],
                            }
                        ],
                    }
                )
            failed_usage = usage_payload(
                validated.generation_request.prompt,
                failed_completions,
                usage_owner.latest_usage,
            )
            yield _sse(
                "error",
                sequence,
                code=code,
                message=safe_message,
                param=classified.param if classified else None,
            )
            sequence += 1
            failed = response_envelope(
                request,
                response_id=response_id,
                created_at=created_at,
                status="failed",
                output=failed_output,
                usage=failed_usage,
                error={"code": code, "message": safe_message},
            )
            yield _sse("response.failed", sequence, response=failed)
            return
        finally:
            await partials.aclose()

        completions = last.completions if last is not None else ()
        usage_owner.mark_completed()
        completion = min(completions, key=lambda item: item.index, default=None)
        text = completion.text if completion is not None else ""
        status = (
            "incomplete"
            if completion is not None
            and completion.finish_reason in {"length", "max_tokens"}
            else "completed"
        )
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
        usage = usage_payload(
            validated.generation_request.prompt,
            completions,
            usage_owner.latest_usage,
        )
        final_response = response_envelope(
            request,
            response_id=response_id,
            created_at=created_at,
            status=status,
            output=[output_item],
            usage=usage,
            incomplete_details=(
                {"reason": "max_output_tokens"} if status == "incomplete" else None
            ),
        )
        if request.store:
            store.save(response_id, stored_items + [output_item], owner=owner)
        terminal_type = "response.completed" if status == "completed" else "response.incomplete"
        yield _sse(terminal_type, sequence, response=final_response)
    finally:
        usage_owner.finalize()
