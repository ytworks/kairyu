"""Direct-engine Responses execution: usage recording and live text streams."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import Request

from kairyu.engine.backend import GenerationResult
from kairyu.entrypoints.server.chat_service import (
    ChatRequestError,
    ExecutedChat,
    ValidatedChatRequest,
)
from kairyu.entrypoints.server.metering import record_state_usage, stream_usage_owner_from_state
from kairyu.entrypoints.server.responses_events import _sse
from kairyu.entrypoints.server.responses_protocol import (
    ResponsesRequest,
    _response_envelope,
    _usage_payload,
)
from kairyu.entrypoints.server.responses_store import ResponseStore
from kairyu.entrypoints.server.sse_encode import ResponsesTextDeltaSSEEncoder

logger = logging.getLogger(__name__)


def _record_execution(
    http_request: Request,
    request: ResponsesRequest,
    execution: ExecutedChat,
) -> None:
    usage = _usage_payload(
        execution.result.prompt,
        execution.result.completions,
        execution.result.usage,
    )
    owner = getattr(http_request.state, "tenant", None) or "default"
    record_state_usage(
        http_request.app.state,
        tenant=owner,
        model=request.model,
        prompt_tokens=usage["input_tokens"],
        completion_tokens=usage["output_tokens"],
        cached_tokens=usage["input_tokens_details"]["cached_tokens"],
        reservation=getattr(http_request.state, "tenant_admission", None),
        usage_exact=execution.result.usage is not None,
    )


def _validate_parallel_tool_calls(
    request: ResponsesRequest,
    execution: ExecutedChat,
) -> None:
    if request.parallel_tool_calls:
        return
    calls = sum(
        len(choice.message.tool_calls or ())
        for choice in execution.response.choices
    )
    if calls > 1:
        raise ChatRequestError(
            "upstream model emitted multiple calls while parallel_tool_calls=false",
            status_code=502,
            code="parallel_tool_calls_not_satisfied",
            error_type="upstream_error",
            execution=execution,
        )


async def _live_text_events(
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
    in_progress = _response_envelope(
        request,
        response_id=response_id,
        created_at=created_at,
        status="in_progress",
        output=[],
        usage=None,
    )
    sequence = 0
    sent = 0
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
        empty_part = {
            "type": "output_text",
            "text": "",
            "annotations": [],
            "logprobs": [],
        }
        yield _sse(
            "response.content_part.added",
            sequence,
            item_id=message_id,
            output_index=0,
            content_index=0,
            part=empty_part,
        )
        sequence += 1
        try:
            usage_owner.mark_dispatched()
            async for partial in validated.engine.stream(validated.generation_request):
                last = partial
                usage_owner.observe(partial.usage, partial.completions)
                completion = min(partial.completions, key=lambda item: item.index, default=None)
                if completion is None:
                    continue
                delta, sent = completion.delta_after(sent)
                if not delta:
                    continue
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
        except Exception as error:
            logger.exception("Responses API upstream stream failed")
            safe_message = f"upstream backend error ({type(error).__name__})"
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
            failed_usage = _usage_payload(
                validated.generation_request.prompt,
                failed_completions,
                usage_owner.latest_usage,
            )
            yield _sse(
                "error",
                sequence,
                code="server_error",
                message=safe_message,
                param=None,
            )
            sequence += 1
            failed = _response_envelope(
                request,
                response_id=response_id,
                created_at=created_at,
                status="failed",
                output=failed_output,
                usage=failed_usage,
                error={"code": "server_error", "message": safe_message},
            )
            yield _sse("response.failed", sequence, response=failed)
            return

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
        usage = _usage_payload(
            validated.generation_request.prompt,
            completions,
            usage_owner.latest_usage,
        )
        final_response = _response_envelope(
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
