"""Responses input-item canonicalization and chat message mapping."""

from __future__ import annotations

import copy
import uuid
from collections.abc import Sequence

from kairyu.entrypoints.server.chat_service import ChatRequestError, ExecutedChat
from kairyu.entrypoints.server.protocol import ChatCompletionRequest
from kairyu.entrypoints.server.responses_codec import _CompactionCodec
from kairyu.entrypoints.server.responses_protocol import ResponsesRequest
from kairyu.entrypoints.server.responses_tools import (
    _chat_tool_choice,
    _chat_tools,
    _namespace_names,
    _namespaced_name,
    _reasoning_effort,
    _response_format,
)

_SUPPORTED_ROLES = {"user", "assistant", "system", "developer"}
_CODEX_INTERNAL_ITEM_FIELDS = {
    "internal_chat_message_metadata_passthrough",
    "encrypted_function_args",
}


def _canonical_input(
    payload: str | list[dict],
    *,
    compaction_codec: _CompactionCodec,
    owner: str,
) -> list[dict]:
    if isinstance(payload, str):
        return [{"type": "message", "role": "user", "content": payload}]
    if not isinstance(payload, list):
        raise ChatRequestError("input must be a string or an array of input items")
    items: list[dict] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ChatRequestError(f"input[{index}] must be an object")
        kind = item.get("type")
        if kind is None and "role" in item:
            kind = "message"
        # Codex-internal passthrough metadata rides input items verbatim when
        # Codex addresses an OpenAI-shaped base URL (the Harbor/Terminal-Bench
        # setup); it scrubs them for custom providers. Accepted and dropped on
        # every item kind.
        item = {
            key: value
            for key, value in item.items()
            if key not in _CODEX_INTERNAL_ITEM_FIELDS
        }
        if kind == "message":
            unknown = set(item) - {
                "type",
                "role",
                "content",
                "status",
                "id",
                "phase",
            }
            if unknown:
                raise ChatRequestError(
                    f"input[{index}] has unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            role = item.get("role")
            if role not in _SUPPORTED_ROLES:
                raise ChatRequestError(
                    f"input[{index}].role must be user, assistant, system, or developer"
                )
            content = item.get("content", "")
            _content_text(content, path=f"input[{index}].content")
            phase = item.get("phase")
            if phase not in (None, "commentary", "final_answer"):
                raise ChatRequestError(
                    f"input[{index}].phase must be commentary or final_answer"
                )
            canonical = {
                "type": "message",
                "role": role,
                "content": copy.deepcopy(content),
            }
            if phase is not None:
                canonical["phase"] = phase
            items.append(canonical)
            continue
        if kind == "function_call":
            unknown = set(item) - {
                "type",
                "id",
                "call_id",
                "name",
                "namespace",
                "arguments",
                "status",
            }
            if unknown:
                raise ChatRequestError(
                    f"input[{index}] has unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            name = item.get("name")
            call_id = item.get("call_id")
            arguments = item.get("arguments")
            namespace = item.get("namespace")
            if not isinstance(name, str) or not name:
                raise ChatRequestError(f"input[{index}].name must be a non-empty string")
            if not isinstance(call_id, str) or not call_id:
                raise ChatRequestError(f"input[{index}].call_id must be a non-empty string")
            if not isinstance(arguments, str):
                raise ChatRequestError(f"input[{index}].arguments must be a JSON string")
            if namespace is not None and (
                not isinstance(namespace, str) or not namespace
            ):
                raise ChatRequestError(
                    f"input[{index}].namespace must be a non-empty string"
                )
            items.append(
                {
                    "type": "function_call",
                    "id": item.get("id") or f"fc_{uuid.uuid4().hex[:24]}",
                    "call_id": call_id,
                    "name": name,
                    "namespace": namespace,
                    "arguments": arguments,
                    "status": item.get("status") or "completed",
                }
            )
            continue
        if kind == "function_call_output":
            unknown = set(item) - {"type", "id", "call_id", "output", "status"}
            if unknown:
                raise ChatRequestError(
                    f"input[{index}] has unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            call_id = item.get("call_id")
            output = item.get("output")
            if not isinstance(call_id, str) or not call_id:
                raise ChatRequestError(f"input[{index}].call_id must be a non-empty string")
            if isinstance(output, list):
                # Codex sends structured tool output (e.g. view_image) as an
                # array of content items instead of a plain string.
                output = _function_output_text(
                    output, path=f"input[{index}].output"
                )
            elif not isinstance(output, str):
                raise ChatRequestError(
                    f"input[{index}].output must be a string or an array of output parts"
                )
            items.append(
                {"type": "function_call_output", "call_id": call_id, "output": output}
            )
            continue
        if kind == "reasoning":
            # Codex echoes prior reasoning items back with the history
            # (encrypted_content may be null or foreign). Kairyu emits no
            # reasoning output item and renders none into the prompt, so the
            # echo is accepted for wire compatibility and dropped.
            unknown = set(item) - {
                "type",
                "id",
                "summary",
                "content",
                "encrypted_content",
                "status",
            }
            if unknown:
                raise ChatRequestError(
                    f"input[{index}] has unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            continue
        if kind in {"compaction", "compaction_summary"}:
            unknown = set(item) - {"type", "id", "encrypted_content", "status"}
            if unknown:
                raise ChatRequestError(
                    f"input[{index}] has unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            # Fail-fast on foreign or cross-tenant tokens; render re-decodes it.
            compaction_codec.decode(item.get("encrypted_content"), owner=owner)
            items.append(
                {
                    "type": "compaction",
                    "encrypted_content": item["encrypted_content"],
                }
            )
            continue
        if kind == "compaction_trigger":
            unknown = set(item) - {"type", "id", "status"}
            if unknown:
                raise ChatRequestError(
                    f"input[{index}] has unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            items.append({"type": "compaction_trigger"})
            continue
        raise ChatRequestError(
            f"input[{index}].type {kind!r} is not supported; "
            "use message, function_call, or function_call_output"
        )
    return items


def _function_output_text(parts: list, *, path: str) -> str:
    texts: list[str] = []
    for index, part in enumerate(parts):
        if not isinstance(part, dict):
            raise ChatRequestError(f"{path}[{index}] must be an object")
        kind = part.get("type")
        if kind in {"input_text", "output_text", "text"}:
            text = part.get("text")
            if not isinstance(text, str):
                raise ChatRequestError(f"{path}[{index}].text must be a string")
            texts.append(text)
            continue
        raise ChatRequestError(
            f"{path}[{index}].type {kind!r} is not supported; "
            "this model accepts text tool output only"
        )
    return "".join(texts)


def _content_text(content: object, *, path: str) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ChatRequestError(f"{path} must be a string or an array of text parts")
    parts: list[str] = []
    for index, part in enumerate(content):
        if not isinstance(part, dict):
            raise ChatRequestError(f"{path}[{index}] must be an object")
        kind = part.get("type")
        if kind not in {"input_text", "output_text", "text"}:
            raise ChatRequestError(f"{path}[{index}].type {kind!r} is not supported")
        allowed = (
            {"type", "text", "annotations", "logprobs"}
            if kind == "output_text"
            else {"type", "text"}
        )
        unknown = set(part) - allowed
        if unknown:
            raise ChatRequestError(
                f"{path}[{index}] has unsupported fields: "
                + ", ".join(sorted(unknown))
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise ChatRequestError(f"{path}[{index}].text must be a string")
        parts.append(text)
    return "".join(parts)


def _validate_function_outputs(items: Sequence[dict]) -> None:
    pending: set[str] = set()
    consumed: set[str] = set()
    for item in items:
        if item["type"] == "function_call":
            call_id = item["call_id"]
            if call_id in pending:
                raise ChatRequestError(f"duplicate function call_id {call_id!r}")
            pending.add(call_id)
        elif item["type"] == "function_call_output":
            call_id = item["call_id"]
            if call_id not in pending:
                raise ChatRequestError(
                    f"input function_call_output references unknown call_id {call_id!r}"
                )
            if call_id in consumed:
                raise ChatRequestError(
                    f"input function_call_output repeats call_id {call_id!r}"
                )
            consumed.add(call_id)


def _items_to_messages(
    items: Sequence[dict],
    *,
    compaction_codec: _CompactionCodec,
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
                name = _namespaced_name(item["namespace"], name)
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
                    "content": _content_text(item.get("content", ""), path="message.content"),
                }
            )
    flush_calls()
    return messages


def _to_chat_request(
    request: ResponsesRequest,
    items: Sequence[dict],
    *,
    compaction_codec: _CompactionCodec,
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
        max_completion_tokens=(
            request.max_output_tokens if request.max_output_tokens is not None else 1024
        ),
        stream=request.stream,
        tools=_chat_tools(request.tools),
        tool_choice=_chat_tool_choice(request.tool_choice, request.tools),
        parallel_tool_calls=request.parallel_tool_calls,
        response_format=_response_format(request.text),
        user=request.user,
        priority=request.priority,
        **values,
    )


def _output_items(
    request: ResponsesRequest,
    execution: ExecutedChat,
) -> list[dict]:
    if not execution.response.choices:
        return []
    message = execution.response.choices[0].message.model_dump(mode="json")
    return _output_items_from_message(request, message)


def _output_items_from_message(request: ResponsesRequest, message: dict) -> list[dict]:
    calls = message.get("tool_calls") or []
    if calls:
        namespaces = _namespace_names(request.tools)
        items = []
        for call in calls:
            function = call.get("function") or {}
            call_name = function.get("name") or ""
            namespace_name = namespaces.get(call_name)
            item = {
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "call_id": call.get("id") or "",
                "type": "function_call",
                "name": (
                    namespace_name[1] if namespace_name is not None else call_name
                ),
                "arguments": function.get("arguments") or "",
                "status": "completed",
            }
            if namespace_name is not None:
                item["namespace"] = namespace_name[0]
            items.append(item)
        return items
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return [
        {
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": content,
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        }
    ]
