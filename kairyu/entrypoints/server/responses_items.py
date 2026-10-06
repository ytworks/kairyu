"""Responses input-item canonicalization and chat message mapping."""

from __future__ import annotations

import copy
import uuid
from collections.abc import Sequence

from kairyu.entrypoints.server.chat_service import ChatRequestError
from kairyu.entrypoints.server.protocol import ChatCompletionRequest
from kairyu.entrypoints.server.responses_codec import _CompactionCodec
from kairyu.entrypoints.server.responses_protocol import ResponsesError, ResponsesRequest
from kairyu.entrypoints.server.responses_tools import (
    _chat_tool_choice,
    _chat_tools,
    _namespaced_name,
    _reasoning_effort,
    _response_format,
)

_SUPPORTED_ROLES = {"user", "assistant", "system", "developer"}
_CODEX_INTERNAL_ITEM_FIELDS = {
    "internal_chat_message_metadata_passthrough",
    "encrypted_function_args",
}
_TEXT_PART_TYPES = {"input_text", "output_text", "text"}
_LISTING_PREFIXES = {
    "message": "msg",
    "function_call": "fc",
    "function_call_output": "fco",
    "reasoning": "rs",
    "compaction": "cmp",
}


def _invalid(param: str, message: str, *, code: str = "invalid_value") -> ResponsesError:
    return ResponsesError(message, param=param, code=code)


def _string_field(item: dict, key: str, path: str) -> str | None:
    """A field compared against known names; any other JSON type is a 400."""

    value = item.get(key)
    if value is not None and not isinstance(value, str):
        raise _invalid(f"{path}.{key}", f"{path}.{key} must be a string")
    return value


def _reject_unknown(path: str, item: dict, allowed: set[str]) -> None:
    unknown = sorted(set(item) - allowed)
    if unknown:
        raise ResponsesError(
            f"{path} has unsupported fields: " + ", ".join(unknown),
            param=f"{path}.{unknown[0]}",
            code="unknown_parameter",
        )


def _item_id(item: dict, prefix: str) -> str:
    supplied = item.get("id")
    if isinstance(supplied, str) and supplied:
        return supplied
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _reasoning_text(
    item: dict, *, path: str, codec: _CompactionCodec | None, owner: str
) -> str:
    """Recover replayed reasoning: a token this server sealed, else visible text.

    Decoding is best-effort on purpose: a token sealed under another key (a
    gateway without the shared secret, another provider) must not fail a
    stateless client's turn, so it falls back to the item's reasoning_text.
    """

    token = item.get("encrypted_content")
    if codec is not None and codec.issued(token):
        try:
            return codec.decode(token, owner=owner)
        except ChatRequestError:
            pass
    content = item.get("content")
    if content is None:
        return ""
    if not isinstance(content, list):
        raise _invalid(f"{path}.content", f"{path}.content must be an array")
    return "".join(
        part["text"]
        for part in content
        if isinstance(part, dict)
        and part.get("type") in ("reasoning_text", "text")
        and isinstance(part.get("text"), str)
    )


def _canonical_input(
    payload: str | list[dict],
    *,
    compaction_codec: _CompactionCodec,
    owner: str,
    reasoning_codec: _CompactionCodec | None = None,
) -> list[dict]:
    if isinstance(payload, str):
        return [
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "role": "user",
                "content": payload,
            }
        ]
    if not isinstance(payload, list):
        raise _invalid("input", "input must be a string or an array of input items")
    items: list[dict] = []
    for index, item in enumerate(payload):
        path = f"input[{index}]"
        if not isinstance(item, dict):
            raise _invalid(path, f"{path} must be an object")
        kind = _string_field(item, "type", path)
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
            _reject_unknown(path, item, {"type", "role", "content", "status", "id", "phase"})
            role = _string_field(item, "role", path)
            if role not in _SUPPORTED_ROLES:
                raise _invalid(
                    f"{path}.role",
                    f"{path}.role must be user, assistant, system, or developer",
                )
            content = item.get("content", "")
            _content_text(content, path=f"{path}.content", images=role != "assistant")
            phase = item.get("phase")
            if phase not in (None, "commentary", "final_answer"):
                raise _invalid(f"{path}.phase", f"{path}.phase must be commentary or final_answer")
            canonical = {
                "type": "message",
                "id": _item_id(item, "msg"),
                "role": role,
                "content": copy.deepcopy(content),
            }
            if phase is not None:
                canonical["phase"] = phase
            items.append(canonical)
            continue
        if kind == "function_call":
            _reject_unknown(
                path,
                item,
                {"type", "id", "call_id", "name", "namespace", "arguments", "status"},
            )
            name = item.get("name")
            call_id = item.get("call_id")
            arguments = item.get("arguments")
            namespace = item.get("namespace")
            if not isinstance(name, str) or not name:
                raise _invalid(f"{path}.name", f"{path}.name must be a non-empty string")
            if not isinstance(call_id, str) or not call_id:
                raise _invalid(f"{path}.call_id", f"{path}.call_id must be a non-empty string")
            if not isinstance(arguments, str):
                raise _invalid(f"{path}.arguments", f"{path}.arguments must be a JSON string")
            if namespace is not None and (
                not isinstance(namespace, str) or not namespace
            ):
                raise _invalid(
                    f"{path}.namespace", f"{path}.namespace must be a non-empty string"
                )
            items.append(
                {
                    "type": "function_call",
                    "id": _item_id(item, "fc"),
                    "call_id": call_id,
                    "name": name,
                    "namespace": namespace,
                    "arguments": arguments,
                    "status": item.get("status") or "completed",
                }
            )
            continue
        if kind == "function_call_output":
            _reject_unknown(path, item, {"type", "id", "call_id", "output", "status"})
            call_id = item.get("call_id")
            output = item.get("output")
            if not isinstance(call_id, str) or not call_id:
                raise _invalid(f"{path}.call_id", f"{path}.call_id must be a non-empty string")
            if isinstance(output, list):
                # Codex sends structured tool output (e.g. view_image) as an
                # array of content items instead of a plain string.
                output = _function_output_text(output, path=f"{path}.output")
            elif not isinstance(output, str):
                raise _invalid(
                    f"{path}.output",
                    f"{path}.output must be a string or an array of output parts",
                )
            items.append(
                {
                    "type": "function_call_output",
                    "id": _item_id(item, "fco"),
                    "call_id": call_id,
                    "output": output,
                }
            )
            continue
        if kind == "reasoning":
            _reject_unknown(
                path,
                item,
                {"type", "id", "summary", "content", "encrypted_content", "status"},
            )
            text = _reasoning_text(item, path=path, codec=reasoning_codec, owner=owner)
            if text:
                items.append(
                    {
                        "type": "reasoning",
                        "id": _item_id(item, "rs"),
                        "summary": [],
                        "content": [{"type": "reasoning_text", "text": text}],
                        "status": "completed",
                    }
                )
            continue
        if kind in {"compaction", "compaction_summary"}:
            _reject_unknown(path, item, {"type", "id", "encrypted_content", "status"})
            # Fail-fast on foreign or cross-tenant tokens; render re-decodes it.
            try:
                compaction_codec.decode(item.get("encrypted_content"), owner=owner)
            except ChatRequestError as error:
                raise _invalid(f"{path}.encrypted_content", str(error)) from None
            items.append(
                {
                    "type": "compaction",
                    "id": _item_id(item, "cmp"),
                    "encrypted_content": item["encrypted_content"],
                }
            )
            continue
        if kind == "compaction_trigger":
            _reject_unknown(path, item, {"type", "id", "status"})
            items.append({"type": "compaction_trigger", "id": _item_id(item, "cmpt")})
            continue
        raise ResponsesError(
            f"{path}.type {kind!r} is not supported; "
            "use message, function_call, or function_call_output",
            param=f"{path}.type",
            code="unsupported_value",
        )
    return items


def _function_output_text(parts: list, *, path: str) -> str:
    texts: list[str] = []
    for index, part in enumerate(parts):
        part_path = f"{path}[{index}]"
        if not isinstance(part, dict):
            raise _invalid(part_path, f"{part_path} must be an object")
        kind = _string_field(part, "type", part_path)
        if kind in _TEXT_PART_TYPES:
            text = part.get("text")
            if not isinstance(text, str):
                raise _invalid(f"{part_path}.text", f"{part_path}.text must be a string")
            texts.append(text)
            continue
        raise ResponsesError(
            f"{part_path}.type {kind!r} is not supported; tool output must be text "
            "(the engine's multimodal prompt carries no tool transcript)",
            param=f"{part_path}.type",
            code="unsupported_value",
        )
    return "".join(texts)


def _content_text(content: object, *, path: str, images: bool = False) -> str:
    """Validate message content and return its text (image parts excluded)."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise _invalid(path, f"{path} must be a string or an array of text parts")
    parts: list[str] = []
    for index, part in enumerate(content):
        part_path = f"{path}[{index}]"
        if not isinstance(part, dict):
            raise _invalid(part_path, f"{part_path} must be an object")
        kind = _string_field(part, "type", part_path)
        if kind == "input_image" and images:
            _validate_image(part, path=part_path)
            continue
        if kind not in _TEXT_PART_TYPES:
            raise ResponsesError(
                f"{part_path}.type {kind!r} is not supported",
                param=f"{part_path}.type",
                code="unsupported_value",
            )
        # prompt_cache_breakpoint marks an explicit OpenAI cache breakpoint;
        # Kairyu's prefix cache is automatic, so the marker is vacuous here.
        allowed = (
            {"type", "text", "annotations", "logprobs"}
            if kind == "output_text"
            else {"type", "text", "prompt_cache_breakpoint"}
        )
        _reject_unknown(part_path, part, allowed)
        text = part.get("text")
        if not isinstance(text, str):
            raise _invalid(f"{part_path}.text", f"{part_path}.text must be a string")
        parts.append(text)
    return "".join(parts)


def _validate_image(part: dict, *, path: str) -> None:
    _reject_unknown(
        path, part, {"type", "image_url", "file_id", "detail", "prompt_cache_breakpoint"}
    )
    if part.get("file_id") is not None:
        raise ResponsesError(
            "file_id images are not supported; send image_url as a data or http(s) URL",
            param=f"{path}.file_id",
            code="unsupported_value",
        )
    if not isinstance(part.get("image_url"), str) or not part["image_url"]:
        raise _invalid(f"{path}.image_url", f"{path}.image_url must be a non-empty URL")
    if part.get("detail") not in (None, "auto", "low", "high", "original"):
        raise _invalid(f"{path}.detail", f"{path}.detail must be auto, low, high, or original")


def _chat_content(content: str | list) -> str | list[dict]:
    """Chat content: plain text, or ordered text/image_url parts with images."""

    if isinstance(content, str):
        return content
    if not any(part.get("type") == "input_image" for part in content):
        return _content_text(content, path="message.content")
    parts: list[dict] = []
    for part in content:
        if part.get("type") != "input_image":
            parts.append({"type": "text", "text": part["text"]})
            continue
        image = {"url": part["image_url"]}
        # "original" (no downscaling) has no chat equivalent: the backend
        # default applies.
        if part.get("detail") in ("auto", "low", "high"):
            image["detail"] = part["detail"]
        parts.append({"type": "image_url", "image_url": image})
    return parts


def _image_paths(payload: object) -> list[str]:
    if not isinstance(payload, list):
        return []
    return [
        f"input[{index}].content[{part_index}]"
        for index, item in enumerate(payload)
        if isinstance(item, dict) and isinstance(item.get("content"), list)
        for part_index, part in enumerate(item["content"])
        if isinstance(part, dict) and part.get("type") == "input_image"
    ]


def reject_images_beside_tools(context: Sequence[dict], payload: object) -> None:
    """Images and tool history cannot share one request (an L1 limit).

    The engine's multimodal prompt carries role and content parts only, with
    no tool calls, tool results, or reasoning, so a request mixing them would
    silently lose the transcript.
    """

    tool_kinds = ("function_call", "function_call_output")
    raw = payload if isinstance(payload, list) else []
    has_tools = any(item["type"] in tool_kinds for item in context) or any(
        isinstance(item, dict) and item.get("type") in tool_kinds for item in raw
    )
    if not has_tools:
        return
    paths = _image_paths(payload)
    context_images = any(
        item["type"] == "message"
        and isinstance(item["content"], list)
        and any(part.get("type") == "input_image" for part in item["content"])
        for item in context
    )
    if paths or context_images:
        raise ResponsesError(
            "images cannot be combined with tool-call history in one request: the "
            "engine's multimodal prompt carries no tool transcript",
            param=paths[0] if paths else "previous_response_id",
            code="unsupported_value",
        )


def first_image_path(payload: object) -> str | None:
    paths = _image_paths(payload)
    return paths[0] if paths else None


def _validate_function_outputs(items: Sequence[dict]) -> None:
    pending: set[str] = set()
    consumed: set[str] = set()
    for item in items:
        if item["type"] == "function_call":
            call_id = item["call_id"]
            if call_id in pending:
                raise _invalid("input", f"duplicate function call_id {call_id!r}")
            pending.add(call_id)
        elif item["type"] == "function_call_output":
            call_id = item["call_id"]
            if call_id not in pending:
                raise _invalid(
                    "input",
                    f"input function_call_output references unknown call_id {call_id!r}",
                )
            if call_id in consumed:
                raise _invalid(
                    "input", f"input function_call_output repeats call_id {call_id!r}"
                )
            consumed.add(call_id)


def input_listing(items: Sequence[dict]) -> list[dict]:
    """Wire view of canonical input items for ``GET .../input_items``."""

    listed: list[dict] = []
    seen: set[str] = set()
    for item in items:
        kind = item["type"]
        if kind == "message":
            content = copy.deepcopy(item["content"])
            if isinstance(content, str):
                if item["role"] == "assistant":
                    content = [{"type": "output_text", "text": content, "annotations": []}]
                else:
                    content = [{"type": "input_text", "text": content}]
            entry = {
                "type": "message",
                "id": item["id"],
                "role": item["role"],
                "status": "completed",
                "content": content,
            }
            if "phase" in item:
                entry["phase"] = item["phase"]
        elif kind == "function_call_output":
            entry = {
                "type": kind,
                "id": item["id"],
                "call_id": item["call_id"],
                "output": item["output"],
                "status": "completed",
            }
        else:
            entry = {key: value for key, value in item.items() if value is not None}
        if entry.get("id") in seen:
            # Client ids can repeat; a cursor (``after``) needs unique ones.
            entry["id"] = f"{_LISTING_PREFIXES.get(kind, 'item')}_{uuid.uuid4().hex[:24]}"
        seen.add(entry["id"])
        listed.append(copy.deepcopy(entry))
    return listed


def _items_to_messages(
    items: Sequence[dict],
    *,
    compaction_codec: _CompactionCodec,
    owner: str,
    replay_reasoning: bool = True,
) -> list[dict]:
    """Render canonical items as chat messages.

    An assistant message and the function calls right after it (prose before
    calls, one Responses turn) become one chat assistant turn, and reasoning
    items ride the next assistant turn as ``reasoning_content`` (the chat
    templates decide what to render). Reasoning not followed by an assistant
    turn is dropped.
    """

    messages: list[dict] = []
    buffered_calls: list[dict] = []
    reasoning: list[str] = []
    assistant_open = False

    def attach_reasoning(message: dict) -> None:
        if reasoning:
            # Reasoning can arrive on both sides of a preamble; keep both.
            earlier = message.get("reasoning_content")
            message["reasoning_content"] = "\n\n".join(
                [earlier, *reasoning] if earlier else reasoning
            )
            reasoning.clear()

    def flush_calls() -> None:
        nonlocal assistant_open
        if not buffered_calls:
            return
        if assistant_open:
            messages[-1]["tool_calls"] = list(buffered_calls)
            attach_reasoning(messages[-1])
        else:
            message = {"role": "assistant", "content": None, "tool_calls": list(buffered_calls)}
            attach_reasoning(message)
            messages.append(message)
        buffered_calls.clear()
        assistant_open = False

    for item in items:
        kind = item["type"]
        if kind == "reasoning":
            if replay_reasoning:
                reasoning.append(
                    "".join(part.get("text", "") for part in item.get("content") or ())
                )
            continue
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
        if kind == "compaction_trigger":
            # Request-level control item; the handler consumes it and it is
            # never rendered into the prompt.
            continue
        if kind == "message" and item["role"] == "assistant":
            message = {
                "role": "assistant",
                "content": _content_text(item.get("content", ""), path="message.content"),
            }
            attach_reasoning(message)
            messages.append(message)
            assistant_open = True
            continue
        reasoning.clear()
        assistant_open = False
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
        else:
            messages.append(
                {"role": item["role"], "content": _chat_content(item.get("content", ""))}
            )
    flush_calls()
    return messages


def _to_chat_request(
    request: ResponsesRequest,
    items: Sequence[dict],
    *,
    compaction_codec: _CompactionCodec,
    owner: str,
    replay_reasoning: bool = True,
) -> ChatCompletionRequest:
    messages = _items_to_messages(
        items,
        compaction_codec=compaction_codec,
        owner=owner,
        replay_reasoning=replay_reasoning,
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
        # Omitted means the remaining context, exactly like Chat Completions
        # (#496): Codex never sends a cap and retries every incomplete turn.
        max_completion_tokens=request.max_output_tokens,
        stream=request.stream,
        tools=_chat_tools(request.tools),
        tool_choice=_chat_tool_choice(request.tool_choice, request.tools),
        parallel_tool_calls=request.parallel_tool_calls,
        response_format=_response_format(request.text),
        user=request.user,
        priority=request.priority,
        **values,
    )
