"""The echoed Response object and its usage shapes.

Shared by every Responses stream path, unary replies, and the error renderer,
so a failed, incomplete, or completed response always echoes the request the
same way. WP-08b replaces it with the full spec envelope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kairyu.entrypoints.server.metering import resolve_cached_tokens, resolve_usage_counts

if TYPE_CHECKING:
    from kairyu.entrypoints.server.responses.request import ResponsesRequest


def response_envelope(
    request: ResponsesRequest,
    *,
    response_id: str,
    created_at: int,
    status: str,
    output: list[dict],
    usage: dict | None,
    error: dict | None = None,
    incomplete_details: dict | None = None,
) -> dict:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": error,
        "incomplete_details": incomplete_details,
        "instructions": request.instructions,
        "model": request.model,
        "output": output,
        "parallel_tool_calls": request.parallel_tool_calls,
        "tool_choice": request.tool_choice or "auto",
        "tools": request.tools or [],
        "temperature": request.temperature,
        "top_p": request.top_p,
        "previous_response_id": request.previous_response_id,
        "metadata": request.metadata,
        "max_output_tokens": request.max_output_tokens,
        "reasoning": None,
        "service_tier": "default" if request.service_tier == "auto" else None,
        "text": request.text,
        "usage": usage,
    }


def usage_payload(
    prompt: str,
    completions,
    usage,
) -> dict:
    """Map an engine generation's usage onto the Responses usage shape."""
    input_tokens, output_tokens = resolve_usage_counts(
        usage, prompt=prompt, completions=completions
    )
    return _usage(input_tokens, resolve_cached_tokens(usage), output_tokens)


def usage_payload_from_wire(usage: dict | None) -> dict:
    """Map public Chat Completions usage onto the Responses usage shape."""
    usage = usage or {}
    details = usage.get("prompt_tokens_details") or {}
    return _usage(
        int(usage.get("prompt_tokens") or 0),
        int(details.get("cached_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
    )


def _usage(input_tokens: int, cached_tokens: int, output_tokens: int) -> dict:
    return {
        "input_tokens": input_tokens,
        # Kairyu never bills a prompt-cache write (m20 D11); openai-python 3.x
        # requires the field.
        "input_tokens_details": {"cached_tokens": cached_tokens, "cache_write_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }


def message_item(message_id: str, status: str, text: str) -> dict:
    return {
        "type": "message",
        "id": message_id,
        "role": "assistant",
        "status": status,
        "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
    }


def open_message_snapshot(
    request: ResponsesRequest, *, response_id: str, created_at: int, message_id: str, text: str
) -> dict:
    """The in-progress envelope matching every event already sent.

    Clients that rebuild their snapshot from lifecycle events (openai-node's
    ResponseStream) replace it with a heartbeat's response, so a heartbeat
    after the message opened must carry it with the text streamed so far.
    """
    return response_envelope(
        request,
        response_id=response_id,
        created_at=created_at,
        status="in_progress",
        output=[message_item(message_id, "in_progress", text)],
        usage=None,
    )
