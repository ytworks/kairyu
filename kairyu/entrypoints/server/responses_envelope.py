"""The echoed Response object and its usage shape (moved from responses_service).

Shared by every Responses stream path, unary replies, and the error renderer,
so a failed, incomplete, or completed response always echoes the request the
same way. WP-08b replaces it with the full spec envelope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kairyu.entrypoints.server.responses_service import ResponsesRequest


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


def usage_payload_from_wire(usage: dict | None) -> dict:
    """Map public Chat Completions usage onto the Responses usage shape."""
    usage = usage or {}
    details = usage.get("prompt_tokens_details") or {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": int(details.get("cached_tokens") or 0)},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }
