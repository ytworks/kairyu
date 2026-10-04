"""Translation of canonical Responses items into a ``ChatCompletionRequest``.

The engine contract is shared with Chat Completions, so chat templates,
tool-choice validation, and upstream capability preflight remain one source of
truth.
"""

from __future__ import annotations

from collections.abc import Sequence

from kairyu.entrypoints.server.chat_service import ChatRequestError
from kairyu.entrypoints.server.protocol import (
    ChatCompletionRequest,
    normalize_reasoning_effort,
)
from kairyu.entrypoints.server.responses.canonical import content_text
from kairyu.entrypoints.server.responses.compaction import CompactionCodec
from kairyu.entrypoints.server.responses.request import ResponsesRequest
from kairyu.entrypoints.server.responses.tools import (
    chat_tool_choice,
    chat_tools,
    namespaced_name,
)


def _items_to_messages(
    items: Sequence[dict],
    *,
    compaction_codec: CompactionCodec,
    owner: str,
) -> list[dict]:
    messages: list[dict] = []
    buffered_calls: list[dict] = []

    def flush_calls() -> None:
        if buffered_calls:
            messages.append(
                {"role": "assistant", "content": None, "tool_calls": list(buffered_calls)}
            )
            buffered_calls.clear()

    for item in items:
        kind = item["type"]
        if kind == "function_call":
            name = item["name"]
            if item.get("namespace"):
                name = namespaced_name(item["namespace"], name)
            buffered_calls.append(
                {
                    "id": item["call_id"],
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": item["arguments"],
                    },
                }
            )
            continue
        flush_calls()
        if kind == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "content": item["output"],
                    "tool_call_id": item["call_id"],
                }
            )
        elif kind == "compaction":
            # Same construction as Codex's own local compaction: the summary
            # rides a user bridge message the conversation continues from.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Summary of the earlier conversation (compacted):\n"
                        + compaction_codec.decode(
                            item["encrypted_content"], owner=owner
                        )
                    ),
                }
            )
        elif kind == "compaction_trigger":
            # Request-level control item; the handler consumes it and it is
            # never rendered into the prompt.
            pass
        else:
            messages.append(
                {
                    "role": item["role"],
                    "content": content_text(item.get("content", ""), path="message.content"),
                }
            )
    flush_calls()
    return messages


def _response_format(text: dict | None) -> dict | None:
    if text is None:
        return None
    if set(text) - {"format", "verbosity"}:
        raise ChatRequestError("text has unsupported fields")
    verbosity = text.get("verbosity")
    if verbosity not in (None, "low", "medium", "high"):
        raise ChatRequestError("text.verbosity must be low, medium, or high")
    fmt = text.get("format")
    if fmt is None:
        return None
    if not isinstance(fmt, dict):
        raise ChatRequestError("text.format must be an object")
    kind = fmt.get("type")
    if kind == "text":
        return {"type": "text"}
    if kind == "json_object":
        return {"type": "json_object"}
    if kind == "json_schema":
        schema = fmt.get("schema")
        if not isinstance(schema, dict):
            raise ChatRequestError("text.format.schema must be a JSON schema object")
        return {
            "type": "json_schema",
            "json_schema": {
                "name": fmt.get("name") or "response",
                "schema": schema,
                "strict": bool(fmt.get("strict", False)),
            },
        }
    raise ChatRequestError(f"text.format.type {kind!r} is not supported")


def _reasoning_effort(reasoning: dict | None) -> str | None:
    if not reasoning:
        return None
    effort = reasoning.get("effort")
    try:
        return normalize_reasoning_effort(effort)
    except ValueError:
        raise ChatRequestError(
            f"reasoning.effort {effort!r} is not supported"
        ) from None


def to_chat_request(
    request: ResponsesRequest,
    items: Sequence[dict],
    *,
    compaction_codec: CompactionCodec,
    owner: str,
) -> ChatCompletionRequest:
    messages = _items_to_messages(
        items, compaction_codec=compaction_codec, owner=owner
    )
    if request.instructions:
        messages.insert(0, {"role": "system", "content": request.instructions})
    verbosity = request.text.get("verbosity") if request.text else None
    if verbosity == "low":
        messages.insert(0, {"role": "system", "content": "Keep the response concise."})
    elif verbosity == "high":
        messages.insert(
            0,
            {"role": "system", "content": "Provide a detailed, thorough response."},
        )
    if not messages:
        messages.append({"role": "user", "content": ""})
    values: dict[str, object] = {}
    if request.temperature is not None:
        values["temperature"] = request.temperature
    if request.top_p is not None:
        values["top_p"] = request.top_p
    reasoning_effort = _reasoning_effort(request.reasoning)
    if reasoning_effort is not None:
        values["reasoning_effort"] = reasoning_effort
    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        # None follows the OpenAI contract: output is bounded by the model's
        # remaining context (issue #496), as on /v1/chat/completions.
        max_completion_tokens=request.max_output_tokens,
        stream=request.stream,
        tools=chat_tools(request.tools),
        tool_choice=chat_tool_choice(request.tool_choice, request.tools),
        parallel_tool_calls=request.parallel_tool_calls,
        response_format=_response_format(request.text),
        user=request.user,
        priority=request.priority,
        **values,
    )
