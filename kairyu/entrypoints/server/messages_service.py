"""Anthropic Messages API adapter for Claude Code gateways (issue #508).

The engine contract is intentionally shared with Chat Completions: Anthropic
wire messages are normalized into ``ChatCompletionRequest`` first, so chat
templates, tool-choice validation, admission, and usage computation remain one
source of truth. AUTO (orchestrated) models delegate to the Chat Completions
handler through the same late-bound ``chat_dispatch`` closure the Responses
adapter uses (issue #530 pattern).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import replace

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from kairyu.engine.backend import (
    CacheHint,
    GenerationResult,
    UpstreamClientError,
    backend_admission_upper_bound_async,
    backend_count_prompt_tokens_async,
    backend_supports_slo_defer,
    prepare_backend_request,
    validate_backend_request_before_prepare,
)
from kairyu.engine.prompt import prompt_text
from kairyu.entrypoints.chat_template import ToolCallProtocol
from kairyu.entrypoints.server.chat_service import (
    ChatRequestError,
    ExecutedChat,
    NormalizedToolChoice,
    ReasoningDeltaParser,
    ValidatedChatRequest,
    _normalize_tool_choice,
    chat_error_from_upstream_client_error,
    execute_chat,
    validate_chat_request_async,
    validate_orchestration_chat_input_async,
)
from kairyu.entrypoints.server.messages_protocol import (
    ANTHROPIC_VERSION,
    MessagesCountTokensRequest,
    MessagesRequest,
    anthropic_error_payload,
    anthropic_error_response,
    anthropic_error_type_for_status,
    new_message_id,
)
from kairyu.entrypoints.server.metering import (
    _approx_tokens,
    metered_prompt,
    record_state_usage,
    resolve_cached_tokens,
    resolve_usage_counts,
    stream_usage_owner_from_state,
)
from kairyu.entrypoints.server.middleware import (
    _ANTHROPIC_INTERNAL_TOOL_STREAM_STATE_KEY,
    _CALLER_API_STATE_KEY,
    _SLO_ADMISSION_LEASE_STATE_KEY,
)
from kairyu.entrypoints.server.protocol import (
    ChatCompletionRequest,
    StreamOptions,
    normalize_reasoning_effort,
)
from kairyu.entrypoints.server.sse_encode import AnthropicTextDeltaSSEEncoder
from kairyu.entrypoints.server.sse_keepalive import iter_with_idle_markers, sse_frames
from kairyu.entrypoints.server.sse_response import sse_response
from kairyu.entrypoints.server.tool_stream import (
    FoldedToolStream,
    StreamInvalid,
    TextDelta,
    ToolArgsDelta,
    ToolStart,
    ToolStop,
    ToolStreamScanner,
    fold_tool_stream,
    tool_stream_scanner_for,
)
from kairyu.sse import escape_json_line_separators

logger = logging.getLogger(__name__)

_LOWEST_SCHEDULER_PRIORITY = 2**63 - 1
_SLO_INTERACTIVE_PRIORITY_CEILING = _LOWEST_SCHEDULER_PRIORITY - 1

# Claude Code counts every relayed byte (including ping events) and aborts a
# stream that goes silent for 300 seconds; protocol-valid pings every 15s keep
# long orchestrated turns comfortably inside that watchdog (same cadence as
# the Responses adapter's keep-alive comments).
_KEEPALIVE_SECONDS = 15.0

_PING_EVENT = 'event: ping\ndata: {"type": "ping"}\n\n'

_SUPPORTED_BLOCKS_HINT = (
    "this endpoint currently supports text, thinking, tool_use, and tool_result blocks"
)
_TOOL_RESULT_ERROR_MARKER = "[tool_result_error]"

# finish_reason (OpenAI) -> stop_reason (Anthropic). Which stop sequence
# matched is not observable on the chat wire, so stop-sequence hits report
# end_turn with stop_sequence null (documented deviation).
_FINISH_TO_STOP = {
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "max_tokens": "max_tokens",
}


class _MessagesFailure(Exception):
    """Generation failure carrying the Anthropic-shaped error content."""

    def __init__(
        self,
        message: str,
        status_code: int,
        error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type or anthropic_error_type_for_status(status_code)

    @classmethod
    def from_chat_error(cls, error: ChatRequestError) -> _MessagesFailure:
        return cls(str(error), error.status_code)

    def json_response(self, request_id: str | None) -> JSONResponse:
        return anthropic_error_response(
            self.message,
            status_code=self.status_code,
            error_type=self.error_type,
            request_id=request_id,
        )


def _slo_shed_response(request_id: str | None) -> JSONResponse:
    return anthropic_error_response(
        "predicted TTFT exceeds the configured SLO",
        status_code=429,
        request_id=request_id,
        headers={"Retry-After": "1"},
    )


def _slo_defer_to_shed_response(
    http_request: Request,
    lease,
    request_id: str | None,
) -> JSONResponse:
    state = http_request.scope.setdefault("state", {})
    if state.get(_SLO_ADMISSION_LEASE_STATE_KEY) is lease:
        state.pop(_SLO_ADMISSION_LEASE_STATE_KEY)
    if lease.active:
        lease.completed()
    return _slo_shed_response(request_id)


# ---------------------------------------------------------------------------
# Request conversion (Anthropic wire -> ChatCompletionRequest)


def _validate_surface(request: MessagesRequest) -> None:
    if request.context_management is not None:
        raise ChatRequestError("context_management is not supported")
    if request.mcp_servers is not None:
        raise ChatRequestError("mcp_servers is not supported")
    if request.container is not None:
        raise ChatRequestError("container is not supported")
    if request.betas is not None:
        raise ChatRequestError(
            "betas must be requested via the anthropic-beta header, "
            "not the request body"
        )
    if request.service_tier is not None:
        raise ChatRequestError("service_tier is not supported; omit it")
    if request.model_extra:
        # Unknown extras are tolerated by design (Claude Code gains fields
        # over releases); known semantics-changing fields are declared on
        # MessagesRequest and handled above.
        logger.debug(
            "ignoring unknown /v1/messages fields: %s",
            ", ".join(sorted(request.model_extra)),
        )


def _system_messages(system: str | list | None) -> list[dict]:
    if system is None:
        return []
    if isinstance(system, str):
        return [{"role": "system", "content": system}]
    messages: list[dict] = []
    for index, block in enumerate(system):
        if not isinstance(block, dict):
            raise ChatRequestError(f"system[{index}] must be an object")
        if block.get("type") != "text":
            raise ChatRequestError(
                f"system[{index}].type {block.get('type')!r} is not supported; "
                "system blocks must be text"
            )
        unknown = set(block) - {"type", "text", "cache_control"}
        if unknown:
            raise ChatRequestError(
                f"system[{index}] has unsupported fields: "
                + ", ".join(sorted(unknown))
            )
        text = block.get("text")
        if not isinstance(text, str):
            raise ChatRequestError(f"system[{index}].text must be a string")
        # cache_control is accepted and ignored (prompt-cache billing is out
        # of scope for this adapter).
        messages.append({"role": "system", "content": text})
    return messages


def _tool_result_text(content: object, *, path: str) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ChatRequestError(
            f"{path}.content must be a string or an array of text blocks"
        )
    parts: list[str] = []
    for index, block in enumerate(content):
        if not isinstance(block, dict):
            raise ChatRequestError(f"{path}.content[{index}] must be an object")
        if block.get("type") != "text":
            raise ChatRequestError(
                f"{path}.content[{index}].type {block.get('type')!r} is not "
                "supported inside tool_result; only text blocks are supported"
            )
        text = block.get("text")
        if not isinstance(text, str):
            raise ChatRequestError(f"{path}.content[{index}].text must be a string")
        parts.append(text)
    return "".join(parts)


def _messages_to_chat(messages: list, *, keep_reasoning: bool) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ChatRequestError("messages must be a non-empty array")
    chat_messages: list[dict] = []
    for m_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ChatRequestError(f"messages[{m_index}] must be an object")
        role = message.get("role")
        if role not in {"user", "assistant", "system"}:
            raise ChatRequestError(
                f"messages[{m_index}].role must be 'user', 'assistant', or 'system'"
            )
        content = message.get("content")
        if isinstance(content, str):
            chat_messages.append({"role": role, "content": content})
            continue
        if role == "system":
            # Claude Code appends mid-conversation {"role": "system"} entries
            # (the operator channel); they render as ordered system messages.
            if not isinstance(content, list):
                raise ChatRequestError(
                    f"messages[{m_index}].content must be a string or an "
                    "array of text blocks"
                )
            chat_messages.extend(_system_messages(content))
            continue
        if not isinstance(content, list):
            raise ChatRequestError(
                f"messages[{m_index}].content must be a string or an array of blocks"
            )
        if role == "user":
            chat_messages.extend(_user_blocks(content, m_index))
        else:
            chat_messages.append(_assistant_blocks(content, m_index, keep_reasoning=keep_reasoning))
    return chat_messages


def _user_blocks(blocks: list, m_index: int) -> list[dict]:
    """Map one user block array: tool_result blocks become tool-role messages
    (in block order), remaining text blocks form one user message."""

    out: list[dict] = []
    text_parts: list[dict] = []
    for b_index, block in enumerate(blocks):
        path = f"messages[{m_index}].content[{b_index}]"
        if not isinstance(block, dict):
            raise ChatRequestError(f"{path} must be an object")
        kind = block.get("type")
        if kind == "tool_result":
            unknown = set(block) - {
                "type",
                "tool_use_id",
                "content",
                "is_error",
                "cache_control",
            }
            if unknown:
                raise ChatRequestError(
                    f"{path} has unsupported fields: " + ", ".join(sorted(unknown))
                )
            tool_use_id = block.get("tool_use_id")
            if not isinstance(tool_use_id, str) or not tool_use_id:
                raise ChatRequestError(f"{path}.tool_use_id must be a non-empty string")
            is_error = block.get("is_error")
            if is_error is not None and not isinstance(is_error, bool):
                raise ChatRequestError(f"{path}.is_error must be a boolean")
            content = _tool_result_text(block.get("content"), path=path)
            if is_error:
                content = (
                    f"{_TOOL_RESULT_ERROR_MARKER}\n{content}"
                    if content
                    else _TOOL_RESULT_ERROR_MARKER
                )
            out.append(
                {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": tool_use_id,
                }
            )
        elif kind == "text":
            unknown = set(block) - {"type", "text", "cache_control"}
            if unknown:
                raise ChatRequestError(
                    f"{path} has unsupported fields: " + ", ".join(sorted(unknown))
                )
            text = block.get("text")
            if not isinstance(text, str):
                raise ChatRequestError(f"{path}.text must be a string")
            text_parts.append({"type": "text", "text": text})
        else:
            raise ChatRequestError(
                f"{path}.type {kind!r} is not supported; {_SUPPORTED_BLOCKS_HINT}"
            )
    if text_parts:
        if len(text_parts) == 1:
            out.append({"role": "user", "content": text_parts[0]["text"]})
        else:
            out.append({"role": "user", "content": text_parts})
    return out


def _assistant_blocks(blocks: list, m_index: int, *, keep_reasoning: bool) -> dict:
    """Map one assistant block array onto a chat assistant message.

    ``thinking`` blocks are the model's own replayed reasoning: a direct engine
    receives them as ``reasoning_content`` (as Responses replays reasoning
    items), an orchestrated model drops them (its stage reasoning is private).
    Signatures are accepted and not verified; Kairyu does not sign.
    """

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    for b_index, block in enumerate(blocks):
        path = f"messages[{m_index}].content[{b_index}]"
        if not isinstance(block, dict):
            raise ChatRequestError(f"{path} must be an object")
        kind = block.get("type")
        if kind == "text":
            unknown = set(block) - {"type", "text", "cache_control"}
            if unknown:
                raise ChatRequestError(
                    f"{path} has unsupported fields: " + ", ".join(sorted(unknown))
                )
            text = block.get("text")
            if not isinstance(text, str):
                raise ChatRequestError(f"{path}.text must be a string")
            text_parts.append(text)
        elif kind == "thinking":
            unknown = set(block) - {"type", "thinking", "signature", "cache_control"}
            if unknown:
                raise ChatRequestError(
                    f"{path} has unsupported fields: " + ", ".join(sorted(unknown))
                )
            thinking = block.get("thinking")
            if not isinstance(thinking, str):
                raise ChatRequestError(f"{path}.thinking must be a string")
            signature = block.get("signature")
            if signature is not None and not isinstance(signature, str):
                raise ChatRequestError(f"{path}.signature must be a string")
            if keep_reasoning and thinking:
                reasoning_parts.append(thinking)
        elif kind == "tool_use":
            unknown = set(block) - {"type", "id", "name", "input", "cache_control"}
            if unknown:
                raise ChatRequestError(
                    f"{path} has unsupported fields: " + ", ".join(sorted(unknown))
                )
            call_id = block.get("id")
            name = block.get("name")
            arguments = block.get("input")
            if not isinstance(call_id, str) or not call_id:
                raise ChatRequestError(f"{path}.id must be a non-empty string")
            if not isinstance(name, str) or not name:
                raise ChatRequestError(f"{path}.name must be a non-empty string")
            if not isinstance(arguments, dict):
                raise ChatRequestError(f"{path}.input must be an object")
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
        else:
            raise ChatRequestError(
                f"{path}.type {kind!r} is not supported; {_SUPPORTED_BLOCKS_HINT}"
            )
    chat_message: dict = {
        "role": "assistant",
        "content": "".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        chat_message["tool_calls"] = tool_calls
    if reasoning_parts:
        chat_message["reasoning_content"] = "".join(reasoning_parts)
    return chat_message


def _chat_tools(tools: list | None) -> list[dict] | None:
    if not tools:
        return None
    converted: list[dict] = []
    names: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ChatRequestError(f"tools[{index}] must be an object")
        kind = tool.get("type")
        if kind not in (None, "custom"):
            raise ChatRequestError(
                f"tools[{index}].type {kind!r} is not supported; "
                "only custom function tools are supported"
            )
        unknown = set(tool) - {
            "type",
            "name",
            "description",
            "input_schema",
            "cache_control",
            "strict",
        }
        if unknown:
            # eager_input_streaming, defer_loading, and other beta tool schema
            # fields are rejected explicitly rather than silently dropped.
            raise ChatRequestError(
                f"tools[{index}] has unsupported fields: "
                + ", ".join(sorted(unknown))
            )
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise ChatRequestError(f"tools[{index}].name must be a non-empty string")
        if name in names:
            raise ChatRequestError(f"tools[{index}].name {name!r} is duplicated")
        names.add(name)
        input_schema = tool.get("input_schema")
        if input_schema is None:
            input_schema = {"type": "object"}
        if not isinstance(input_schema, dict):
            raise ChatRequestError(f"tools[{index}].input_schema must be an object")
        function: dict = {"name": name, "parameters": input_schema}
        description = tool.get("description")
        if description is not None:
            if not isinstance(description, str):
                raise ChatRequestError(f"tools[{index}].description must be a string")
            function["description"] = description
        strict = tool.get("strict")
        if strict is not None:
            if not isinstance(strict, bool):
                raise ChatRequestError(f"tools[{index}].strict must be a boolean")
            function["strict"] = strict
        converted.append({"type": "function", "function": function})
    return converted


def _chat_tool_choice(choice: dict | None) -> tuple[str | dict | None, bool | None]:
    """Return the chat tool_choice and the parallel_tool_calls override."""

    if choice is None:
        return None, None
    if not isinstance(choice, dict):
        raise ChatRequestError("tool_choice must be an object")
    kind = choice.get("type")
    disable = choice.get("disable_parallel_tool_use")
    if disable is not None and not isinstance(disable, bool):
        raise ChatRequestError("tool_choice.disable_parallel_tool_use must be a boolean")
    parallel = False if disable else None
    allowed = {"type", "disable_parallel_tool_use"}
    if kind == "tool":
        allowed = allowed | {"name"}
    unknown = set(choice) - allowed
    if unknown:
        raise ChatRequestError(
            "tool_choice has unsupported fields: " + ", ".join(sorted(unknown))
        )
    if kind == "auto":
        return "auto", parallel
    if kind == "any":
        return "required", parallel
    if kind == "none":
        return "none", parallel
    if kind == "tool":
        name = choice.get("name")
        if not isinstance(name, str) or not name:
            raise ChatRequestError("tool_choice.name must be a non-empty string")
        return {"type": "function", "function": {"name": name}}, parallel
    raise ChatRequestError(
        f"tool_choice.type {kind!r} is not supported; "
        "expected auto, any, tool, or none"
    )


def _reasoning_effort_from(request: MessagesRequest, *, direct: bool) -> str | None:
    """Kairyu's effort for the request; ``"none"`` turns a direct engine's thinking off.

    ``thinking.type: disabled`` reaches a direct engine as thinking off (the
    ``none`` effort, set only here); an orchestrated model keeps its stage
    policy, so it stays a no-op there.
    """

    thinking = request.thinking
    thinking_kind: str | None = None
    if thinking is not None:
        if not isinstance(thinking, dict):
            raise ChatRequestError("thinking must be an object")
        kind = thinking.get("type")
        if kind == "enabled":
            if "budget_tokens" in thinking:
                raise ChatRequestError(
                    "thinking.type 'enabled' with budget_tokens is not supported; "
                    'use {"type": "adaptive"} or output_config.effort'
                )
            # Without a budget, enabled means thinking on with the effort of
            # output_config (DeepSeek Harness sends it this way): adaptive.
            kind = "adaptive"
        if kind not in {"adaptive", "disabled"}:
            raise ChatRequestError(
                f"thinking.type {kind!r} is not supported; "
                "expected adaptive or disabled"
            )
        unknown = set(thinking) - {"type", "display"}
        if unknown:
            raise ChatRequestError(
                "thinking has unsupported fields: " + ", ".join(sorted(unknown))
            )
        if thinking.get("display") not in {None, "summarized", "omitted"}:
            raise ChatRequestError(
                "thinking.display must be 'summarized' or 'omitted'"
            )
        thinking_kind = kind
        # Anthropic's current adaptive semantics are also the default when
        # thinking is omitted, so adaptive keeps the model's own default.
    thinking_off = "none" if thinking_kind == "disabled" and direct else None
    output_config = request.output_config
    if output_config is None:
        return thinking_off
    if not isinstance(output_config, dict):
        raise ChatRequestError("output_config must be an object")
    unknown = set(output_config) - {"effort"}
    if unknown:
        # format (structured outputs) and task_budget are not implemented
        # end-to-end; reject rather than silently ignore.
        raise ChatRequestError(
            "output_config has unsupported fields: " + ", ".join(sorted(unknown))
        )
    effort = output_config.get("effort")
    if effort is None:
        return thinking_off
    if thinking_kind == "disabled":
        raise ChatRequestError(
            "thinking.type 'disabled' conflicts with output_config.effort"
        )
    try:
        return normalize_reasoning_effort(effort)
    except ValueError:
        raise ChatRequestError(
            f"output_config.effort {effort!r} is not supported"
        ) from None


def _to_chat_request(request: MessagesRequest, *, direct: bool) -> ChatCompletionRequest:
    """The chat request for one Messages request; ``direct`` is a non-AUTO model."""

    messages = _system_messages(request.system) + _messages_to_chat(
        request.messages, keep_reasoning=direct
    )
    tool_choice, parallel_tool_calls = _chat_tool_choice(request.tool_choice)
    values: dict[str, object] = {}
    if request.temperature is not None:
        values["temperature"] = request.temperature
    if request.top_p is not None:
        values["top_p"] = request.top_p
    if request.top_k is not None:
        values["top_k"] = request.top_k
    if request.stop_sequences is not None:
        if not isinstance(request.stop_sequences, list) or not all(
            isinstance(item, str) and item for item in request.stop_sequences
        ):
            raise ChatRequestError(
                "stop_sequences must be an array of non-empty strings"
            )
        values["stop"] = list(request.stop_sequences)
    reasoning_effort = _reasoning_effort_from(request, direct=direct)
    if reasoning_effort not in {None, "none"}:
        values["reasoning_effort"] = reasoning_effort
    if request.metadata is not None:
        if not isinstance(request.metadata, dict):
            raise ChatRequestError("metadata must be an object")
        user_id = request.metadata.get("user_id")
        if user_id is not None:
            if not isinstance(user_id, str):
                raise ChatRequestError("metadata.user_id must be a string")
            values["user"] = user_id
    chat_request = ChatCompletionRequest(
        model=request.model,
        messages=messages,
        max_completion_tokens=request.max_tokens,
        stream=request.stream,
        tools=_chat_tools(request.tools),
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        **values,
    )
    if reasoning_effort == "none":
        # Thinking off is internal: the public Chat and Responses parsers
        # keep rejecting "none" (m11 D4 Messages thinking amendment).
        chat_request = chat_request.model_copy(update={"reasoning_effort": "none"})
    return chat_request


# ---------------------------------------------------------------------------
# Response conversion (chat -> Anthropic Message)


def _stop_reason_for(finish_reason: str | None) -> str:
    return _FINISH_TO_STOP.get(finish_reason or "stop", "end_turn")


def _content_blocks_from_message(message: Mapping) -> list[dict]:
    """Build Anthropic content blocks from one chat assistant message.

    ``reasoning_content`` is deliberately never read: Kairyu's private
    orchestration reasoning must not leak into Anthropic text blocks.
    """

    blocks: list[dict] = []
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    if content:
        blocks.append({"type": "text", "text": content})
    for call in message.get("tool_calls") or ():
        function = call.get("function") or {}
        arguments = function.get("arguments") or ""
        try:
            parsed = json.loads(arguments) if arguments else {}
        except ValueError:
            raise _MessagesFailure(
                "upstream model produced malformed tool arguments", 502
            ) from None
        if not isinstance(parsed, dict):
            raise _MessagesFailure(
                "upstream model produced non-object tool arguments", 502
            )
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": function.get("name") or "",
                "input": parsed,
            }
        )
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


def _usage_payload(prompt, completions, usage) -> dict:
    input_tokens, output_tokens = resolve_usage_counts(
        usage, prompt=prompt, completions=completions
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": resolve_cached_tokens(usage),
    }


def _usage_from_wire(usage: Mapping | None) -> dict:
    """Map public Chat Completions usage onto the Anthropic usage shape.

    Reuses the corrected #496 public-contract semantics — no third accounting
    definition; Kairyu's orchestration extension fields are not surfaced here.
    """

    usage = usage or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": int(details.get("cached_tokens") or 0),
    }


def _message_envelope(
    request: MessagesRequest,
    *,
    message_id: str,
    blocks: list[dict],
    stop_reason: str,
    usage: dict,
) -> dict:
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": request.model,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


def _tools_active(chat_request: ChatCompletionRequest) -> bool:
    return bool(chat_request.tools) and chat_request.tool_choice != "none"


def _split_inline_reasoning(text: str) -> tuple[str, str]:
    """Mirror ``_build_choice``'s inline ``<think>`` split for raw text."""

    candidate = text
    has_opening = candidate.startswith("<think>")
    if has_opening:
        candidate = candidate[len("<think>") :]
    if "</think>" in candidate:
        reasoning, candidate = candidate.split("</think>", 1)
        return reasoning, candidate
    if has_opening:
        return candidate, ""
    return "", text


def _thinks(reasoning_effort: str | None) -> bool:
    """Whether reasoning may lead the output (an effort that is not thinking off)."""

    return reasoning_effort not in {None, "none"}


def _thinking_omitted(request: MessagesRequest) -> bool:
    """``thinking.display: "omitted"``: thinking blocks carry no text."""

    thinking = request.thinking
    return isinstance(thinking, dict) and thinking.get("display") == "omitted"


def _thinking_block(reasoning: str | None, *, omitted: bool) -> list[dict]:
    """A direct engine's own reasoning as one Anthropic thinking block.

    Kairyu does not sign thinking; the signature is empty, and an omitted
    display leaves the text empty too. Orchestrated models never reach this
    (their stage reasoning stays private).
    """

    if not reasoning:
        return []
    return [{"type": "thinking", "thinking": "" if omitted else reasoning, "signature": ""}]


def _enforce_tool_gates(
    folded: FoldedToolStream,
    choice_spec: NormalizedToolChoice,
    parallel_tool_calls: bool | None,
) -> None:
    """Re-home the pre-SSE 502 gates onto the scanner outcome (unary side)."""

    if (
        not folded.invalidated
        and parallel_tool_calls is False
        and folded.committed_calls > 1
    ):
        raise _MessagesFailure(
            "upstream model emitted multiple calls while parallel_tool_calls=false",
            502,
        )
    if choice_spec.mode in {"required", "named"} and folded.committed_calls == 0:
        raise _MessagesFailure("upstream model did not satisfy tool_choice", 502)


def _outcome_from_execution(
    execution: ExecutedChat,
    validated: ValidatedChatRequest,
    *,
    omit_thinking: bool,
) -> tuple[list[dict], str, dict]:
    usage = _usage_payload(
        execution.metered_prompt,
        execution.result.completions,
        execution.result.usage,
    )
    chat_request = validated.input.request
    if _tools_active(chat_request):
        # The shared scanner is the single source of truth for /v1/messages:
        # the unary body is a one-shot fold of the same parse the stream
        # emits, so stream/unary reconstruction is equivalent by construction
        # (text and tool_use blocks coexist, the Anthropic contract).
        completion = min(
            execution.result.completions, key=lambda item: item.index, default=None
        )
        raw = completion.text if completion is not None else ""
        finish_reason = (
            completion.finish_reason if completion is not None else None
        )
        reasoning = (
            completion.reasoning_content if completion is not None else None
        )
        if (
            _thinks(chat_request.reasoning_effort)
            and completion is not None
            and completion.reasoning_content is None
        ):
            reasoning, raw = _split_inline_reasoning(raw)
        choice_spec = validated.input.normalized_tool_choice
        scanner = tool_stream_scanner_for(
            validated.input.tool_call_protocol, chat_request.tools, choice_spec
        )
        events = scanner.feed(raw, final=True)
        folded = fold_tool_stream(events, raw_text=scanner.raw_text)
        _enforce_tool_gates(
            folded, choice_spec, validated.input.parallel_tool_calls
        )
        stop_reason = (
            "tool_use"
            if folded.committed_calls
            else _stop_reason_for(finish_reason)
        )
        return (
            _thinking_block(reasoning, omitted=omit_thinking) + folded.blocks,
            stop_reason,
            usage,
        )
    choice = execution.response.choices[0] if execution.response.choices else None
    message = choice.message.model_dump(mode="json") if choice is not None else {}
    blocks = _thinking_block(
        message.get("reasoning_content"), omitted=omit_thinking
    ) + _content_blocks_from_message(message)
    stop_reason = _stop_reason_for(choice.finish_reason if choice is not None else None)
    return blocks, stop_reason, usage


def _record_execution(
    http_request: Request,
    model: str,
    execution: ExecutedChat,
) -> None:
    usage = _usage_payload(
        execution.metered_prompt,
        execution.result.completions,
        execution.result.usage,
    )
    owner = getattr(http_request.state, "tenant", None) or "default"
    record_state_usage(
        http_request.app.state,
        tenant=owner,
        model=model,
        prompt_tokens=usage["input_tokens"],
        completion_tokens=usage["output_tokens"],
        cached_tokens=usage["cache_read_input_tokens"],
        reservation=getattr(http_request.state, "tenant_admission", None),
        usage_exact=execution.result.usage is not None,
    )


# ---------------------------------------------------------------------------
# SSE event emission


def _sse(event_type: str, payload: dict) -> str:
    serialized = escape_json_line_separators(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return f"event: {event_type}\ndata: {serialized}\n\n"


def _message_start_event(
    request: MessagesRequest, message_id: str, *, input_tokens: int = 0
) -> str:
    return _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": request.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        },
    )


def _content_block_start_event(index: int, content_block: dict) -> str:
    return _sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": content_block,
        },
    )


def _content_block_stop_event(index: int) -> str:
    return _sse(
        "content_block_stop", {"type": "content_block_stop", "index": index}
    )


def _message_delta_event(stop_reason: str, usage: dict) -> str:
    return _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": usage,
        },
    )


def _message_stop_event() -> str:
    return _sse("message_stop", {"type": "message_stop"})


def _error_event(message: str, error_type: str = "api_error") -> str:
    return _sse(
        "error", anthropic_error_payload(message, error_type=error_type)
    )


def _thinking_start_event(index: int) -> str:
    return _content_block_start_event(index, {"type": "thinking", "thinking": "", "signature": ""})


def _thinking_delta_event(index: int, text: str) -> str:
    return _sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "thinking_delta", "thinking": text},
        },
    )


def _thinking_stop_events(index: int) -> list[str]:
    # Kairyu does not sign thinking: an empty signature closes the block.
    return [
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "signature_delta", "signature": ""},
            },
        ),
        _content_block_stop_event(index),
    ]


class _AnthropicBlockEmitter:
    """Render scanner events as Anthropic SSE with stable block indices.

    The text block opens lazily on the first non-whitespace text so a
    pure-call output keeps tool indices 0..n-1; whitespace between blocks is
    held and dropped when a tool block follows (or flushed with later text).
    New text after a tool block opens a fresh text index — Anthropic-legal
    interleaving. A direct engine's reasoning is a ``thinking`` block that
    closes before the next text or tool block opens.
    """

    def __init__(self, *, omit_thinking: bool = False) -> None:
        self._encoder = AnthropicTextDeltaSSEEncoder()
        self._omit_thinking = omit_thinking
        self._next_index = 0
        self._text_index: int | None = None
        self._thinking_index: int | None = None
        self._tool_index = 0
        self._pending_ws = ""
        self._has_content = False
        self.deltas_emitted = 0

    def thinking(self, text: str) -> list[str | bytes]:
        """Reasoning text: opens (or continues) the current thinking block."""

        if not text:
            return []
        out: list[str | bytes] = []
        if self._thinking_index is None:
            out.extend(self._close_text())
            self._thinking_index = self._next_index
            self._next_index += 1
            out.append(_thinking_start_event(self._thinking_index))
        if not self._omit_thinking:
            out.append(_thinking_delta_event(self._thinking_index, text))
        return out

    def _close_thinking(self) -> list[str]:
        if self._thinking_index is None:
            return []
        index, self._thinking_index = self._thinking_index, None
        return _thinking_stop_events(index)

    def events(self, scanned) -> list[str | bytes]:
        out: list[str | bytes] = []
        for event in scanned:
            if isinstance(event, TextDelta):
                if self._text_index is None and not event.text.strip():
                    self._pending_ws += event.text
                    continue
                if self._text_index is None:
                    out.extend(self._close_thinking())
                    self._text_index = self._next_index
                    self._next_index += 1
                    self._has_content = True
                    out.append(
                        _content_block_start_event(
                            self._text_index, {"type": "text", "text": ""}
                        )
                    )
                text = self._pending_ws + event.text
                self._pending_ws = ""
                out.append(self._encoder.encode(self._text_index, text))
                self.deltas_emitted += 1
            elif isinstance(event, ToolStart):
                out.extend(self._close_thinking())
                out.extend(self._close_text())
                self._pending_ws = ""
                self._tool_index = self._next_index
                self._next_index += 1
                self._has_content = True
                out.append(
                    _content_block_start_event(
                        self._tool_index,
                        {
                            "type": "tool_use",
                            "id": event.id,
                            "name": event.name,
                            "input": {},
                        },
                    )
                )
            elif isinstance(event, ToolArgsDelta):
                out.append(
                    _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self._tool_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": event.partial_json,
                            },
                        },
                    )
                )
                self.deltas_emitted += 1
            elif isinstance(event, ToolStop):
                out.append(_content_block_stop_event(self._tool_index))
        return out

    def _close_text(self) -> list[str]:
        if self._text_index is None:
            return []
        index, self._text_index = self._text_index, None
        return [_content_block_stop_event(index)]

    def finish(self) -> list[str | bytes]:
        out: list[str | bytes] = [*self._close_thinking(), *self._close_text()]
        if self._has_content:
            return out
        index = self._next_index
        self._next_index += 1
        out.append(
            _content_block_start_event(
                index, {"type": "text", "text": ""}
            )
        )
        if self._pending_ws:
            out.append(self._encoder.encode(index, self._pending_ws))
            self._pending_ws = ""
            self.deltas_emitted += 1
        out.append(_content_block_stop_event(index))
        return out


async def _live_tool_events(
    request: MessagesRequest,
    validated: ValidatedChatRequest,
    *,
    message_id: str,
    http_request: Request,
    owner: str,
) -> AsyncIterator[str | bytes]:
    """Stream a direct-engine turn with executable tools incrementally (#573).

    Tool calls commit as each envelope closes, so ``input_json_delta``
    fragments reach the client while the model is still generating. The
    pre-SSE 502 gates are re-homed here: a ``parallel_tool_calls=false``
    violation fails the moment a second call would open, an unsatisfied
    required/named ``tool_choice`` fails at end of stream — both as an
    Anthropic ``error`` event with no ``message_stop``.
    """

    chat_request = validated.input.request
    choice_spec = validated.input.normalized_tool_choice
    parallel = validated.input.parallel_tool_calls
    scanner = tool_stream_scanner_for(
        validated.input.tool_call_protocol, chat_request.tools, choice_spec
    )
    emitter = _AnthropicBlockEmitter(omit_thinking=_thinking_omitted(request))
    reasoning_parser = (
        ReasoningDeltaParser() if _thinks(chat_request.reasoning_effort) else None
    )
    sent = 0
    reasoning_sent = 0
    started = 0
    saw_final = False
    last: GenerationResult | None = None
    slo_lease = getattr(http_request.state, _SLO_ADMISSION_LEASE_STATE_KEY, None)
    first_token_observed = False
    usage_owner = stream_usage_owner_from_state(
        http_request.app.state,
        tenant=owner,
        model=request.model,
        prompt=metered_prompt(validated.generation_request),
        reservation=getattr(http_request.state, "tenant_admission", None),
    )

    def process(scanned) -> tuple[list[str | bytes], str | None]:
        nonlocal started
        rendered: list[str | bytes] = []
        for event in scanned:
            if isinstance(event, StreamInvalid):
                return rendered, event.message
            if isinstance(event, ToolStart):
                if parallel is False and started >= 1:
                    return rendered, (
                        "upstream model emitted multiple calls while "
                        "parallel_tool_calls=false"
                    )
                started += 1
            rendered.extend(emitter.events([event]))
        return rendered, None

    try:
        yield _message_start_event(request, message_id)
        try:
            usage_owner.mark_dispatched()
            async for item in iter_with_idle_markers(
                validated.engine.stream(validated.generation_request), _KEEPALIVE_SECONDS
            ):
                if item is None:
                    yield _PING_EVENT
                    continue
                partial = item
                last = partial
                usage_owner.observe(partial.usage, partial.completions)
                completion = min(
                    partial.completions, key=lambda entry: entry.index, default=None
                )
                if completion is None:
                    continue
                reasoning, reasoning_sent = completion.reasoning_delta_after(
                    reasoning_sent
                )
                delta, sent = completion.delta_after(sent)
                if reasoning_parser is not None and type(delta) is str:
                    # <think> reasoning is split off before the scanner.
                    inline_reasoning, delta = reasoning_parser.feed(
                        delta, final=partial.finished
                    )
                    reasoning += inline_reasoning
                for chunk in emitter.thinking(reasoning):
                    yield chunk
                if reasoning and slo_lease is not None and not first_token_observed:
                    slo_lease.finished_first_token()
                    first_token_observed = True
                if partial.finished:
                    saw_final = True
                if type(delta) is not str:
                    delta = ""
                if not delta and not partial.finished:
                    continue
                rendered, fatal = process(
                    scanner.feed(delta, final=partial.finished)
                )
                for chunk in rendered:
                    yield chunk
                if (
                    slo_lease is not None
                    and not first_token_observed
                    and emitter.deltas_emitted
                ):
                    slo_lease.finished_first_token()
                    first_token_observed = True
                if fatal is not None:
                    yield _error_event(fatal)
                    return
        except Exception as error:
            logger.exception("Anthropic Messages upstream tool stream failed")
            yield _error_event(f"upstream backend error ({type(error).__name__})")
            return
        if not saw_final:
            rendered, fatal = process(scanner.feed("", final=True))
            for chunk in rendered:
                yield chunk
            if fatal is not None:
                yield _error_event(fatal)
                return
        completions = last.completions if last is not None else ()
        usage_owner.mark_completed()
        completion = min(completions, key=lambda entry: entry.index, default=None)
        finish_reason = completion.finish_reason if completion is not None else None
        if (
            choice_spec.mode in {"required", "named"}
            and scanner.committed_calls == 0
        ):
            yield _error_event("upstream model did not satisfy tool_choice")
            return
        for chunk in emitter.finish():
            yield chunk
        usage = _usage_payload(
            metered_prompt(validated.generation_request),
            completions,
            usage_owner.latest_usage,
        )
        stop_reason = (
            "tool_use" if scanner.committed_calls else _stop_reason_for(finish_reason)
        )
        yield _message_delta_event(stop_reason, usage)
        yield _message_stop_event()
    finally:
        usage_owner.finalize()


async def _relay_auto_tool_stream(
    request: MessagesRequest,
    upstream,
    *,
    message_id: str,
    scanner: ToolStreamScanner,
    choice_spec: NormalizedToolChoice,
    parallel: bool | None,
) -> AsyncIterator[str | bytes]:
    """Re-encode the internal raw AUTO stream with incremental tool parsing.

    The upstream carries the final unit's verbatim text (tool tags included,
    #573 sentinel); ``: status`` comments are forwarded (they feed the byte
    watchdog) and ``delta.reasoning_content`` is dropped entirely. The chat
    handler's pre-SSE tool gates are re-homed here as SSE ``error`` events.
    """

    emitter = _AnthropicBlockEmitter()
    started = 0
    finish_reason: str | None = None
    wire_usage: dict | None = None
    error_payload: dict | None = None
    fatal: str | None = None

    def process(scanned) -> tuple[list[str | bytes], str | None]:
        nonlocal started
        rendered: list[str | bytes] = []
        for event in scanned:
            if isinstance(event, StreamInvalid):
                return rendered, event.message
            if isinstance(event, ToolStart):
                if parallel is False and started >= 1:
                    return rendered, (
                        "upstream model emitted multiple calls while "
                        "parallel_tool_calls=false"
                    )
                started += 1
            rendered.extend(emitter.events([event]))
        return rendered, None

    try:
        yield _message_start_event(request, message_id)
        try:
            async for frame in sse_frames(upstream):
                if frame.startswith(":"):
                    yield f"{frame}\n\n"
                    continue
                if not frame.startswith("data:"):
                    continue
                payload_text = frame[len("data:") :].strip()
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
                        rendered, fatal = process(scanner.feed(content))
                        for chunk in rendered:
                            yield chunk
                        if fatal is not None:
                            break
                if fatal is not None:
                    break
        except Exception as error:
            logger.exception("Anthropic Messages orchestrated tool relay failed")
            error_payload = {
                "message": f"upstream backend error ({type(error).__name__})"
            }
    finally:
        aclose = getattr(upstream, "aclose", None)
        if aclose is not None:
            await aclose()

    if fatal is not None:
        yield _error_event(fatal)
        return
    if error_payload is not None:
        yield _error_event(error_payload.get("message") or "upstream backend error")
        return
    rendered, fatal = process(scanner.feed("", final=True))
    for chunk in rendered:
        yield chunk
    if fatal is not None:
        yield _error_event(fatal)
        return
    if choice_spec.mode in {"required", "named"} and scanner.committed_calls == 0:
        yield _error_event("upstream model did not satisfy tool_choice")
        return
    for chunk in emitter.finish():
        yield chunk
    stop_reason = (
        "tool_use" if scanner.committed_calls else _stop_reason_for(finish_reason)
    )
    yield _message_delta_event(stop_reason, _usage_from_wire(wire_usage))
    yield _message_stop_event()


async def _consume_auto_tool_stream(
    upstream, *, scanner: ToolStreamScanner
) -> tuple[list, str | None, dict | None]:
    """Drain the internal AUTO stream for the unary tool path (no bytes sent)."""

    events: list = []
    finish_reason: str | None = None
    wire_usage: dict | None = None
    try:
        async for frame in sse_frames(upstream):
            if not frame.startswith("data:"):
                continue
            payload_text = frame[len("data:") :].strip()
            if payload_text == "[DONE]":
                break
            try:
                chunk_payload = json.loads(payload_text)
            except ValueError:
                continue
            if "error" in chunk_payload and "choices" not in chunk_payload:
                error = chunk_payload["error"] or {}
                raise _MessagesFailure(
                    error.get("message") or "upstream backend error", 502
                )
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
                    events.extend(scanner.feed(content))
    finally:
        aclose = getattr(upstream, "aclose", None)
        if aclose is not None:
            await aclose()
    events.extend(scanner.feed("", final=True))
    return events, finish_reason, wire_usage


async def _live_text_events(
    request: MessagesRequest,
    validated: ValidatedChatRequest,
    *,
    message_id: str,
    http_request: Request,
    owner: str,
) -> AsyncIterator[str | bytes]:
    """Stream a direct-engine text turn live (no tools declared)."""

    encoder = AnthropicTextDeltaSSEEncoder()
    omit_thinking = _thinking_omitted(request)
    reasoning_parser = (
        ReasoningDeltaParser()
        if _thinks(validated.input.request.reasoning_effort)
        else None
    )
    sent = 0
    reasoning_sent = 0
    next_index = 0
    text_index: int | None = None
    thinking_index: int | None = None
    last: GenerationResult | None = None
    last_activity = time.monotonic()
    slo_lease = getattr(
        http_request.state, _SLO_ADMISSION_LEASE_STATE_KEY, None
    )
    first_token_observed = False
    usage_owner = stream_usage_owner_from_state(
        http_request.app.state,
        tenant=owner,
        model=request.model,
        prompt=metered_prompt(validated.generation_request),
        reservation=getattr(http_request.state, "tenant_admission", None),
    )
    try:
        yield _message_start_event(request, message_id)
        try:
            usage_owner.mark_dispatched()
            async for partial in validated.engine.stream(validated.generation_request):
                last = partial
                usage_owner.observe(partial.usage, partial.completions)
                completion = min(
                    partial.completions, key=lambda item: item.index, default=None
                )
                if completion is None:
                    continue
                reasoning, reasoning_sent = completion.reasoning_delta_after(
                    reasoning_sent
                )
                delta, sent = completion.delta_after(sent)
                if reasoning_parser is not None and type(delta) is str:
                    # <think> reasoning is split off, never surfaced as text.
                    inline_reasoning, delta = reasoning_parser.feed(
                        delta, final=partial.finished
                    )
                    reasoning += inline_reasoning
                if reasoning:
                    # The engine's own reasoning is one thinking block that
                    # closes before the next text block opens.
                    if thinking_index is None:
                        if text_index is not None:
                            yield _content_block_stop_event(text_index)
                            text_index = None
                        thinking_index = next_index
                        next_index += 1
                        yield _thinking_start_event(thinking_index)
                        last_activity = time.monotonic()
                    if not omit_thinking:
                        yield _thinking_delta_event(thinking_index, reasoning)
                        last_activity = time.monotonic()
                    if slo_lease is not None and not first_token_observed:
                        slo_lease.finished_first_token()
                        first_token_observed = True
                if not delta:
                    # Reasoning-only progress: keep the byte watchdog fed.
                    if time.monotonic() - last_activity >= _KEEPALIVE_SECONDS:
                        yield _PING_EVENT
                        last_activity = time.monotonic()
                    continue
                if thinking_index is not None:
                    for chunk in _thinking_stop_events(thinking_index):
                        yield chunk
                    thinking_index = None
                if text_index is None:
                    text_index = next_index
                    next_index += 1
                    yield _content_block_start_event(
                        text_index, {"type": "text", "text": ""}
                    )
                if type(delta) is str:
                    yield encoder.encode(text_index, delta)
                else:
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": text_index,
                            "delta": {"type": "text_delta", "text": delta},
                        },
                    )
                if slo_lease is not None and not first_token_observed:
                    slo_lease.finished_first_token()
                    first_token_observed = True
                last_activity = time.monotonic()
        except Exception as error:
            logger.exception("Anthropic Messages upstream stream failed")
            yield _error_event(f"upstream backend error ({type(error).__name__})")
            return
        completions = last.completions if last is not None else ()
        usage_owner.mark_completed()
        completion = min(completions, key=lambda item: item.index, default=None)
        finish_reason = completion.finish_reason if completion is not None else None
        if thinking_index is not None:
            for chunk in _thinking_stop_events(thinking_index):
                yield chunk
        if text_index is None:
            # Every message carries a text block, empty when nothing was said.
            text_index = next_index
            yield _content_block_start_event(text_index, {"type": "text", "text": ""})
        yield _content_block_stop_event(text_index)
        usage = _usage_payload(
            metered_prompt(validated.generation_request),
            completions,
            usage_owner.latest_usage,
        )
        yield _message_delta_event(_stop_reason_for(finish_reason), usage)
        yield _message_stop_event()
    finally:
        usage_owner.finalize()


async def _relay_auto_stream(
    request: MessagesRequest,
    upstream,
    *,
    message_id: str,
) -> AsyncIterator[str | bytes]:
    """Re-encode the orchestrated Chat Completions SSE stream as Anthropic SSE.

    ``upstream`` is this process's own chat-chunk stream for an AUTO model:
    ``: status`` keep-alive comments (forwarded verbatim — they reset the
    byte watchdog), ``data: {chat chunk}`` frames, ``data: {"error": ...}``
    frames, and a final ``data: [DONE]``. ``delta.reasoning_content`` is
    dropped entirely: private orchestration reasoning never becomes Anthropic
    text content.
    """

    encoder = AnthropicTextDeltaSSEEncoder()
    finish_reason: str | None = None
    wire_usage: dict | None = None
    error_payload: dict | None = None

    try:
        yield _message_start_event(request, message_id)
        yield _content_block_start_event(0, {"type": "text", "text": ""})
        async for frame in sse_frames(upstream):
            if frame.startswith(":"):
                yield f"{frame}\n\n"
                continue
            if not frame.startswith("data:"):
                continue
            payload_text = frame[len("data:") :].strip()
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
                    yield encoder.encode(0, content)
    except Exception as error:
        logger.exception("Anthropic Messages orchestrated relay failed")
        error_payload = {
            "message": f"upstream backend error ({type(error).__name__})"
        }
    finally:
        aclose = getattr(upstream, "aclose", None)
        if aclose is not None:
            await aclose()

    if error_payload is not None:
        yield _error_event(error_payload.get("message") or "upstream backend error")
        return
    yield _content_block_stop_event(0)
    yield _message_delta_event(
        _stop_reason_for(finish_reason), _usage_from_wire(wire_usage)
    )
    yield _message_stop_event()


# ---------------------------------------------------------------------------
# Route


def _validation_error_message(error: ValidationError) -> str:
    first = error.errors()[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    message = first.get("msg", "invalid value")
    return f"{location}: {message}" if location else message


def _rerendered_chat_json_error(
    delegated: JSONResponse, request_id: str | None
) -> JSONResponse:
    """Re-render a chat-handler OpenAI error body in the Anthropic envelope."""

    try:
        payload = json.loads(bytes(delegated.body))
    except ValueError:
        payload = {}
    error = payload.get("error") or {}
    return anthropic_error_response(
        error.get("message") or "upstream backend error",
        status_code=delegated.status_code,
        request_id=request_id,
    )


async def _orchestrated_message(
    request: MessagesRequest,
    chat_request: ChatCompletionRequest,
    http_request: Request,
    chat_dispatch,
    *,
    message_id: str,
    request_id: str | None,
):
    """Serve an AUTO model by delegating to the Chat Completions handler.

    The chat handler owns the entire orchestration contract (validation,
    tenant reservation, admission, execution, public usage and metering), so
    this branch never runs the engine-only pipeline and never records usage
    itself.
    """

    setattr(http_request.state, _CALLER_API_STATE_KEY, "messages")
    if _tools_active(chat_request):
        # #573: tool-bearing AUTO requests consume the in-process raw stream
        # (the state sentinel is unreachable from the wire) for BOTH stream
        # and unary modes — the delegated non-stream chat JSON nulls content
        # when calls exist, so the raw text needed for coexisting text/tool
        # blocks is only recoverable from the stream. AUTO is pinned GENERIC,
        # and this adapter enforces the tool gates itself.
        try:
            choice_spec = _normalize_tool_choice(chat_request)
        except ChatRequestError as error:
            return _MessagesFailure.from_chat_error(error).json_response(
                request_id
            )
        parallel = False if chat_request.parallel_tool_calls is False else None
        scanner = tool_stream_scanner_for(
            ToolCallProtocol.GENERIC, chat_request.tools, choice_spec
        )
        setattr(
            http_request.state, _ANTHROPIC_INTERNAL_TOOL_STREAM_STATE_KEY, True
        )
        live_request = chat_request.model_copy(
            update={"stream": True, "stream_options": StreamOptions(include_usage=True)}
        )
        delegated = await chat_dispatch(live_request, http_request)
        if isinstance(delegated, JSONResponse):
            return _rerendered_chat_json_error(delegated, request_id)
        if not isinstance(delegated, StreamingResponse):
            # The client left while the profile judge ran (the chat handler's
            # 499); there is no stream to read and nobody to answer.
            return delegated
        if request.stream:
            return sse_response(
                _relay_auto_tool_stream(
                    request,
                    delegated.body_iterator,
                    message_id=message_id,
                    scanner=scanner,
                    choice_spec=choice_spec,
                    parallel=parallel,
                )
            )
        try:
            events, finish_reason, wire_usage = await _consume_auto_tool_stream(
                delegated.body_iterator, scanner=scanner
            )
            folded = fold_tool_stream(events, raw_text=scanner.raw_text)
            _enforce_tool_gates(folded, choice_spec, parallel)
        except _MessagesFailure as failure:
            return failure.json_response(request_id)
        stop_reason = (
            "tool_use"
            if folded.committed_calls
            else _stop_reason_for(finish_reason)
        )
        return JSONResponse(
            content=_message_envelope(
                request,
                message_id=message_id,
                blocks=folded.blocks,
                stop_reason=stop_reason,
                usage=_usage_from_wire(wire_usage),
            )
        )
    if request.stream:
        if chat_request.tools:
            # tools declared but tool_choice=none: the calls can never run, so
            # the plain text relay is correct — but the sentinel is still
            # needed or the chat handler would buffer the whole generation.
            setattr(
                http_request.state,
                _ANTHROPIC_INTERNAL_TOOL_STREAM_STATE_KEY,
                True,
            )
        live_request = chat_request.model_copy(
            update={"stream": True, "stream_options": StreamOptions(include_usage=True)}
        )
        delegated = await chat_dispatch(live_request, http_request)
        if isinstance(delegated, JSONResponse):
            return _rerendered_chat_json_error(delegated, request_id)
        if not isinstance(delegated, StreamingResponse):
            return delegated  # the client left during the judge (499), as above
        return sse_response(
            _relay_auto_stream(
                request, delegated.body_iterator, message_id=message_id
            )
        )
    buffered_request = chat_request.model_copy(update={"stream": False})

    async def produce() -> tuple[list[dict], str, dict]:
        delegated = await chat_dispatch(buffered_request, http_request)
        if not isinstance(delegated, JSONResponse):
            raise _MessagesFailure("unexpected non-JSON chat dispatch reply", 502)
        payload = json.loads(bytes(delegated.body))
        if delegated.status_code != 200:
            error = payload.get("error") or {}
            raise _MessagesFailure(
                error.get("message") or "upstream backend error",
                delegated.status_code,
            )
        choices = payload.get("choices") or []
        message = (choices[0].get("message") if choices else None) or {}
        blocks = _content_blocks_from_message(message)
        finish_reason = choices[0].get("finish_reason") if choices else None
        return (
            blocks,
            _stop_reason_for(finish_reason),
            _usage_from_wire(payload.get("usage")),
        )

    try:
        blocks, stop_reason, usage = await produce()
    except _MessagesFailure as failure:
        return failure.json_response(request_id)
    return JSONResponse(
        content=_message_envelope(
            request,
            message_id=message_id,
            blocks=blocks,
            stop_reason=stop_reason,
            usage=usage,
        )
    )


def add_messages_route(
    app: FastAPI,
    engines: Mapping,
    *,
    chat_templates=None,
    legacy_chat_models: AbstractSet[str] | None = None,
    upstream_chat_models: AbstractSet[str] | None = None,
    orchestrated_models: AbstractSet[str] | None = None,
    chat_dispatch=None,
) -> None:
    @app.post("/v1/messages/count_tokens")
    async def messages_count_tokens(http_request: Request) -> JSONResponse:
        """Anthropic token counting, consistent with billed usage.

        Tier resolution: orchestrated models count the L2-rendered prompt with
        the same word-split billing uses for multi-stage routes; direct
        engines count the generation request exactly, through the backend
        that would send it. A backend that cannot count authoritatively
        declines with the Anthropic-shaped 404 that Claude Code treats as an
        absent endpoint.
        """

        request_id = getattr(http_request.state, "request_id", None)

        def request_error(
            message: str, *, status_code: int = 400
        ) -> JSONResponse:
            return anthropic_error_response(
                message, status_code=status_code, request_id=request_id
            )

        version = http_request.headers.get("anthropic-version")
        if version is not None and version != ANTHROPIC_VERSION:
            return request_error(
                f"anthropic-version {version!r} is not supported; "
                f"use {ANTHROPIC_VERSION!r}"
            )
        try:
            payload = await http_request.json()
        except ValueError:
            return request_error("request body must be valid JSON")
        try:
            request = MessagesCountTokensRequest.model_validate(payload)
        except ValidationError as error:
            return request_error(_validation_error_message(error))
        http_request.state.model = request.model
        engine = engines.get(request.model)
        orchestrated = (
            engine is None
            and chat_dispatch is not None
            and request.model in (orchestrated_models or ())
        )
        if engine is None and not orchestrated:
            return request_error(
                f"model {request.model!r} not found", status_code=404
            )
        try:
            _validate_surface(request)
            chat_request = _to_chat_request(request, direct=not orchestrated)
            if orchestrated:
                validated_input = await validate_orchestration_chat_input_async(
                    chat_request
                )
                text = prompt_text(validated_input.prompt) or ""
                input_tokens = _approx_tokens(text)
            else:
                validated = await validate_chat_request_async(
                    chat_request,
                    engines,
                    chat_templates,
                    request_id=request_id or f"msg-{uuid.uuid4().hex[:12]}",
                    legacy_chat_models=legacy_chat_models,
                    upstream_chat_models=upstream_chat_models,
                )
                try:
                    counted = await backend_count_prompt_tokens_async(
                        validated.engine, validated.generation_request
                    )
                except UpstreamClientError as error:
                    raise chat_error_from_upstream_client_error(error) from error
                if counted is None:
                    return request_error(
                        f"model {request.model!r} does not support "
                        "authoritative token counting",
                        status_code=404,
                    )
                input_tokens = counted
        except ChatRequestError as error:
            return _MessagesFailure.from_chat_error(error).json_response(
                request_id
            )
        except ValueError as error:
            return request_error(str(error))
        except _MessagesFailure as failure:
            return failure.json_response(request_id)
        return JSONResponse(content={"input_tokens": input_tokens})

    @app.post("/v1/messages")
    async def messages(http_request: Request):
        request_id = getattr(http_request.state, "request_id", None)

        def request_error(
            message: str, *, status_code: int = 400
        ) -> JSONResponse:
            return anthropic_error_response(
                message, status_code=status_code, request_id=request_id
            )

        def chat_error(error: ChatRequestError) -> JSONResponse:
            return _MessagesFailure.from_chat_error(error).json_response(request_id)

        version = http_request.headers.get("anthropic-version")
        if version is not None and version != ANTHROPIC_VERSION:
            return request_error(
                f"anthropic-version {version!r} is not supported; "
                f"use {ANTHROPIC_VERSION!r}"
            )
        # anthropic-beta is an open, comma-separated capability list; it is
        # accepted verbatim (never an allowlist) and unused capabilities are
        # governed by explicit body-field policy instead.
        try:
            payload = await http_request.json()
        except ValueError:
            return request_error("request body must be valid JSON")
        try:
            request = MessagesRequest.model_validate(payload)
        except ValidationError as error:
            return request_error(_validation_error_message(error))

        http_request.state.model = request.model
        metrics = getattr(http_request.app.state, "metrics", None)
        ingress_ns = getattr(http_request.state, "placement_started_ns", None)
        if metrics is not None and type(ingress_ns) is int:
            metrics.record_preplacement_phase(
                "messages",
                "ingress_to_handler",
                max(0, time.perf_counter_ns() - ingress_ns),
            )
        engine = engines.get(request.model)
        orchestrated = (
            engine is None
            and chat_dispatch is not None
            and request.model in (orchestrated_models or ())
        )
        if engine is None and not orchestrated:
            return request_error(
                f"model {request.model!r} not found", status_code=404
            )
        owner = getattr(http_request.state, "tenant", None) or "default"
        message_id = new_message_id()
        validation_started_ns = time.perf_counter_ns()
        try:
            _validate_surface(request)
            chat_request = _to_chat_request(request, direct=not orchestrated)
            if not orchestrated:
                # The Claude Code session id keeps one session's turns on one
                # replica (prefix-cache affinity); it is tracing metadata and
                # never becomes a metric label or a logged credential.
                session_id = http_request.headers.get("x-claude-code-session-id")
                scheduling_class = getattr(
                    http_request.state, "scheduling_class", None
                )
                if scheduling_class not in {"interactive", "batch"}:
                    transported = http_request.headers.get(
                        "x-kairyu-scheduling-class"
                    )
                    scheduling_class = (
                        transported
                        if transported in {"interactive", "batch"}
                        else "interactive"
                    )
                validated = await validate_chat_request_async(
                    chat_request,
                    engines,
                    chat_templates,
                    request_id=request_id or f"msg-{uuid.uuid4().hex[:12]}",
                    cache_hint=(
                        CacheHint(session_id=session_id) if session_id else None
                    ),
                    priority=getattr(http_request.state, "priority", None),
                    scheduling_class=scheduling_class,
                    placement_started_ns=getattr(
                        http_request.state, "placement_started_ns", None
                    ),
                    legacy_chat_models=legacy_chat_models,
                    upstream_chat_models=upstream_chat_models,
                    caller_api="messages",
                )
        except ChatRequestError as error:
            return chat_error(error)
        except _MessagesFailure as failure:
            return failure.json_response(request_id)
        finally:
            if metrics is not None:
                metrics.record_preplacement_phase(
                    "messages",
                    "request_validation",
                    max(0, time.perf_counter_ns() - validation_started_ns),
                )
        if orchestrated:
            return await _orchestrated_message(
                request,
                chat_request,
                http_request,
                chat_dispatch,
                message_id=message_id,
                request_id=request_id,
            )
        slo_lease = None
        slo_admission = getattr(http_request.app.state, "slo_admission", None)
        if (
            slo_admission is not None
            and validated.generation_request.scheduling_class == "interactive"
        ):
            ingress_ns = getattr(http_request.state, "placement_started_ns", None)
            elapsed_s = (
                max(0, time.perf_counter_ns() - ingress_ns) / 1_000_000_000
                if type(ingress_ns) is int
                else 0.0
            )
            lease = slo_admission.begin(elapsed_s=elapsed_s)
            if lease.decision.action == "shed":
                return _slo_shed_response(request_id)
            slo_lease = lease
            http_request.scope.setdefault("state", {})[
                _SLO_ADMISSION_LEASE_STATE_KEY
            ] = lease
            generation_request = validated.generation_request
            if lease.decision.action == "defer":
                admission_request = replace(
                    generation_request,
                    priority=_LOWEST_SCHEDULER_PRIORITY,
                    scheduling_class="batch",
                )
                if not backend_supports_slo_defer(
                    validated.engine,
                    admission_request,
                ):
                    return _slo_defer_to_shed_response(
                        http_request, lease, request_id
                    )
            elif generation_request.priority > _SLO_INTERACTIVE_PRIORITY_CEILING:
                admission_request = replace(
                    generation_request,
                    priority=_SLO_INTERACTIVE_PRIORITY_CEILING,
                )
            else:
                admission_request = generation_request
            if admission_request is not generation_request:
                try:
                    validate_backend_request_before_prepare(
                        validated.engine,
                        admission_request,
                    )
                except ValueError as error:
                    return request_error(str(error))
                validated = replace(
                    validated,
                    generation_request=admission_request,
                )

        prepare_started_ns = time.perf_counter_ns()
        try:
            await prepare_backend_request(
                validated.engine,
                validated.generation_request,
            )
        except UpstreamClientError as error:
            return chat_error(chat_error_from_upstream_client_error(error))
        except ValueError as error:
            return request_error(str(error))
        except RuntimeError as error:
            logger.exception("Anthropic Messages backend prepare failed")
            return request_error(
                f"upstream backend error ({type(error).__name__})",
                status_code=502,
            )
        finally:
            if metrics is not None:
                metrics.record_preplacement_phase(
                    "messages",
                    "backend_prepare",
                    max(0, time.perf_counter_ns() - prepare_started_ns),
                )
        if (
            slo_lease is not None
            and slo_lease.decision.action == "defer"
            and not backend_supports_slo_defer(
                validated.engine,
                validated.generation_request,
            )
        ):
            return _slo_defer_to_shed_response(
                http_request, slo_lease, request_id
            )

        admission_started_ns = time.perf_counter_ns()
        try:
            bound = await backend_admission_upper_bound_async(
                validated.engine,
                validated.generation_request,
            )
        except ValueError as error:
            return request_error(str(error))
        except RuntimeError as error:
            logger.exception("Anthropic Messages admission bound failed")
            return request_error(
                f"upstream backend error ({type(error).__name__})",
                status_code=502,
            )
        admission_ns = max(0, time.perf_counter_ns() - admission_started_ns)
        reserve_started_ns = time.perf_counter_ns()
        admission = getattr(http_request.state, "tenant_admission", None)
        if admission is not None:
            admitted = admission.reserve_tokens(
                bound.tokens,
                refundable_on_exact_usage=bound.refundable_on_exact_usage,
            )
            if metrics is not None:
                metrics.record_tenant_admission(
                    owner,
                    source="http",
                    admitted=admitted,
                    reason=admission.reason,
                )
            if admitted:
                http_request.state.tenant_metric_admitted = True
            if not admitted:
                return anthropic_error_response(
                    f"tenant {owner!r} admission limit exceeded "
                    f"({admission.reason})",
                    status_code=429,
                    request_id=request_id,
                    headers={"Retry-After": "1"},
                )
        admission_ns += max(0, time.perf_counter_ns() - reserve_started_ns)
        if metrics is not None:
            metrics.record_preplacement_phase("messages", "admission", admission_ns)
            metrics.record_priority(
                validated.generation_request.scheduling_class,
                source="http",
            )

        if request.stream:
            if _tools_active(chat_request):
                return sse_response(
                    _live_tool_events(
                        request,
                        validated,
                        message_id=message_id,
                        http_request=http_request,
                        owner=owner,
                    )
                )
            return sse_response(
                _live_text_events(
                    request,
                    validated,
                    message_id=message_id,
                    http_request=http_request,
                    owner=owner,
                )
            )

        async def produce() -> tuple[list[dict], str, dict]:
            try:
                if admission is not None:
                    admission.mark_dispatched()
                execution = await execute_chat(validated)
            except ChatRequestError as error:
                if error.execution is not None:
                    # The OpenAI-shaped 502 gates re-evaluate on the shared
                    # scanner so the unary and stream verdicts cannot diverge.
                    _record_execution(http_request, request.model, error.execution)
                    return _outcome_from_execution(
                        error.execution,
                        validated,
                        omit_thinking=_thinking_omitted(request),
                    )
                raise _MessagesFailure.from_chat_error(error) from error
            except Exception as error:
                logger.exception("Anthropic Messages upstream generation failed")
                raise _MessagesFailure(
                    f"upstream backend error ({type(error).__name__})", 502
                ) from error
            _record_execution(http_request, request.model, execution)
            blocks, stop_reason, usage = _outcome_from_execution(
                execution, validated, omit_thinking=_thinking_omitted(request)
            )
            if slo_lease is not None and any(
                block["type"] in {"tool_use", "thinking"} or block.get("text")
                for block in blocks
            ):
                slo_lease.finished_first_token()
            return blocks, stop_reason, usage

        try:
            blocks, stop_reason, usage = await produce()
        except _MessagesFailure as failure:
            return failure.json_response(request_id)
        return JSONResponse(
            content=_message_envelope(
                request,
                message_id=message_id,
                blocks=blocks,
                stop_reason=stop_reason,
                usage=usage,
            )
        )


def add_hello_probe(app: FastAPI) -> None:
    """Claude Code's best-effort connection-warming probe.

    ``/api/hello`` sits outside the ``/v1/`` auth guard by design: the probe
    carries no request body and a cheap success avoids burning the client's
    startup budget. Its disclosure level matches the already-open ``/health``.
    """

    @app.head("/api/hello")
    async def api_hello() -> Response:
        return Response(status_code=200)
