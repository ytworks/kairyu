"""OpenAI Responses API adapter with typed streaming and function tools.

The engine contract is intentionally shared with Chat Completions.  Responses
wire items are normalized into ``ChatCompletionRequest`` first, so chat
templates, tool-choice validation, and upstream capability preflight remain one
source of truth.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import json
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping, Sequence
from collections.abc import Set as AbstractSet

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from kairyu.engine.backend import (
    CacheHint,
    GenerationResult,
    UpstreamClientError,
    backend_admission_upper_bound_async,
    prepare_backend_request,
)
from kairyu.entrypoints.server.chat_service import (
    ChatRequestError,
    ExecutedChat,
    ValidatedChatRequest,
    chat_error_from_upstream_client_error,
    execute_chat,
    validate_chat_request_async,
)
from kairyu.entrypoints.server.errors import upstream_error
from kairyu.entrypoints.server.metering import (
    record_state_usage,
    resolve_cached_tokens,
    resolve_usage_counts,
    stream_usage_owner_from_state,
)
from kairyu.entrypoints.server.protocol import ChatCompletionRequest, StreamOptions
from kairyu.entrypoints.server.sse_encode import ResponsesTextDeltaSSEEncoder
from kairyu.entrypoints.server.sse_response import sse_response
from kairyu.sse import escape_json_line_separators

logger = logging.getLogger(__name__)

_ALLOWED_INCLUDE = {"reasoning.encrypted_content"}
_SUPPORTED_ROLES = {"user", "assistant", "system", "developer"}
_NAMESPACE_SEPARATOR = "__"
_CODEX_INTERNAL_ITEM_FIELDS = {
    "internal_chat_message_metadata_passthrough",
    "encrypted_function_args",
}
# Buffered generation can run for minutes on orchestrated models while the
# strictest known client budget (Codex) closes idle SSE streams at 300s.
_BUFFERED_KEEPALIVE_SECONDS = 15.0
# Remote compaction v2 (Codex against an OpenAI-shaped provider): a terminal
# compaction_trigger input item requests exactly one compaction output item
# whose encrypted_content is opaque to the client and echoed back verbatim.
_COMPACTION_MARKER = "kairyu.compaction.v1"
_COMPACTION_MAX_OUTPUT_TOKENS = 4096
_COMPACTION_INSTRUCTION_ITEM = {
    "type": "message",
    "role": "user",
    "content": (
        "Summarize the conversation above for a compacted continuation. "
        "Capture the goals, decisions, constraints, tool results, and any "
        "unfinished work so the conversation can continue from this summary "
        "alone."
    ),
}


def _encode_compaction(summary: str) -> str:
    payload = json.dumps(
        {"k": _COMPACTION_MARKER, "summary": summary}, ensure_ascii=False
    )
    return base64.urlsafe_b64encode(payload.encode()).decode()


def _decode_compaction(token: object) -> str:
    if not isinstance(token, str) or not token:
        raise ChatRequestError(
            "compaction encrypted_content must be a non-empty string"
        )
    try:
        payload = json.loads(base64.urlsafe_b64decode(token.encode()).decode())
    except Exception:
        raise ChatRequestError(
            "compaction encrypted_content was not issued by this server"
        ) from None
    if (
        not isinstance(payload, dict)
        or payload.get("k") != _COMPACTION_MARKER
        or not isinstance(payload.get("summary"), str)
    ):
        raise ChatRequestError(
            "compaction encrypted_content was not issued by this server"
        )
    return payload["summary"]


def _extract_compaction_trigger(items: Sequence[dict]) -> bool:
    for index, item in enumerate(items):
        if item["type"] == "compaction_trigger" and index != len(items) - 1:
            raise ChatRequestError("compaction_trigger must be the final input item")
    return bool(items) and items[-1]["type"] == "compaction_trigger"


def _compaction_output_from_message(message: dict) -> list[dict]:
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    if not isinstance(content, str) or not content.strip():
        raise _BufferedFailure(
            {
                "message": "upstream model returned an empty compaction summary",
                "type": "upstream_error",
                "code": "compaction_failed",
            },
            502,
        )
    return [
        {
            "type": "compaction",
            "id": f"cmp_{uuid.uuid4().hex[:24]}",
            "encrypted_content": _encode_compaction(content),
            "status": "completed",
        }
    ]


class _BufferedFailure(Exception):
    """Generation failure carrying the exact wire error payload.

    The buffered path surfaces it as an HTTP error before streaming starts
    (non-stream requests) or as ``error`` + ``response.failed`` SSE events
    once the stream is already open.
    """

    def __init__(self, payload: dict, status_code: int) -> None:
        super().__init__(payload.get("message") or "generation failed")
        self.payload = payload
        self.status_code = status_code

    @classmethod
    def from_chat_error(cls, error: ChatRequestError) -> _BufferedFailure:
        return cls(error.payload(), error.status_code)

    def json_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code, content={"error": self.payload}
        )


class ResponsesRequest(BaseModel):
    """Supported Responses request surface used by the OpenAI SDK and Codex."""

    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[dict] = ""
    instructions: str | None = None
    previous_response_id: str | None = None
    store: bool = True
    max_output_tokens: int | None = None
    stream: bool = False
    metadata: dict = Field(default_factory=dict)
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool = True
    temperature: float | None = None
    top_p: float | None = None
    text: dict | None = None
    include: list[str] | None = None
    reasoning: dict | None = None
    service_tier: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    stream_options: dict | None = None
    client_metadata: dict | None = None
    context_management: list[dict] | None = None
    safety_identifier: str | None = None
    user: str | None = None
    top_logprobs: int | None = None
    truncation: str | None = None
    background: bool | None = None
    conversation: str | dict | None = None
    prompt: dict | None = None
    max_tool_calls: int | None = None
    moderation: dict | None = None
    priority: int = Field(default=0, ge=-(2**63), le=2**63 - 1)


class ResponseStore:
    """Bounded, tenant-scoped state for ``previous_response_id``."""

    def __init__(self, max_items: int = 4096) -> None:
        self._items: OrderedDict[str, tuple[str, list[dict]]] = OrderedDict()
        self._max = max_items

    def save(self, response_id: str, items: list[dict], owner: str = "default") -> None:
        self._items[response_id] = (owner, copy.deepcopy(items))
        self._items.move_to_end(response_id)
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    def get(self, response_id: str, owner: str = "default") -> list[dict] | None:
        entry = self._items.get(response_id)
        if entry is None or entry[0] != owner:
            return None
        self._items.move_to_end(response_id)
        return copy.deepcopy(entry[1])


def _request_error(message: str, *, status_code: int = 400, code: str | None = None):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": code,
            }
        },
    )


def _chat_error(error: ChatRequestError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content={"error": error.payload()})


def _validate_request_surface(request: ResponsesRequest) -> None:
    if request.model_extra:
        raise ChatRequestError(
            "unsupported request fields: " + ", ".join(sorted(request.model_extra))
        )
    if request.background:
        raise ChatRequestError("background responses are not supported")
    if request.conversation is not None:
        raise ChatRequestError("conversation is not supported; use previous_response_id")
    if request.prompt is not None:
        raise ChatRequestError("prompt templates are not supported")
    if request.max_tool_calls is not None:
        raise ChatRequestError("max_tool_calls is not supported")
    if request.moderation is not None:
        raise ChatRequestError("moderation is not supported")
    if request.top_logprobs is not None:
        raise ChatRequestError("top_logprobs is not supported by the Responses adapter")
    if request.service_tier not in (None, "auto"):
        raise ChatRequestError("service_tier is not supported; omit it or use 'auto'")
    if request.truncation not in (None, "disabled"):
        raise ChatRequestError("truncation='auto' is not supported")
    unsupported_includes = set(request.include or ()) - _ALLOWED_INCLUDE
    if unsupported_includes:
        raise ChatRequestError(
            "unsupported include values: " + ", ".join(sorted(unsupported_includes))
        )
    if request.prompt_cache_retention not in (None, "in_memory", "24h"):
        raise ChatRequestError("prompt_cache_retention must be 'in_memory' or '24h'")
    if request.context_management:
        raise ChatRequestError("context_management is not supported")
    if request.stream_options is not None:
        if not isinstance(request.stream_options, dict):
            raise ChatRequestError("stream_options must be an object")
        unknown_stream_options = set(request.stream_options) - {"include_obfuscation"}
        if unknown_stream_options:
            raise ChatRequestError(
                "unsupported stream_options fields: "
                + ", ".join(sorted(unknown_stream_options))
            )
        if request.stream_options.get("include_obfuscation") not in (None, False):
            raise ChatRequestError("stream obfuscation is not supported")


def _canonical_input(payload: str | list[dict]) -> list[dict]:
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
            # Fail-fast on foreign tokens; the summary is re-decoded at render.
            _decode_compaction(item.get("encrypted_content"))
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


def _items_to_messages(items: Sequence[dict]) -> list[dict]:
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
                        + _decode_compaction(item["encrypted_content"])
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


def _namespaced_name(namespace: str, name: str) -> str:
    return f"{namespace}{_NAMESPACE_SEPARATOR}{name}"


def _function_tool(
    tool: dict,
    *,
    path: str,
    name_override: str | None = None,
    description_prefix: str | None = None,
) -> dict:
    if tool.get("type") != "function":
        raise ChatRequestError(f"{path}.type must be 'function'")
    name = tool.get("name")
    parameters = tool.get("parameters", {})
    if not isinstance(name, str) or not name:
        raise ChatRequestError(f"{path}.name must be a non-empty string")
    if not isinstance(parameters, dict):
        raise ChatRequestError(f"{path}.parameters must be an object")
    unknown = set(tool) - {
        "type",
        "name",
        "description",
        "parameters",
        "strict",
        "defer_loading",
    }
    if unknown:
        raise ChatRequestError(
            f"{path} has unsupported fields: " + ", ".join(sorted(unknown))
        )
    function = {"name": name_override or name, "parameters": parameters}
    description = tool.get("description")
    if description_prefix:
        description = (
            f"{description_prefix} {description}" if description else description_prefix
        )
    if description is not None:
        if not isinstance(description, str):
            raise ChatRequestError(f"{path}.description must be a string")
        function["description"] = description
    if tool.get("strict") is not None:
        if not isinstance(tool["strict"], bool):
            raise ChatRequestError(f"{path}.strict must be a boolean")
        function["strict"] = tool["strict"]
    if tool.get("defer_loading") is not None and not isinstance(
        tool["defer_loading"], bool
    ):
        raise ChatRequestError(f"{path}.defer_loading must be a boolean")
    return {"type": "function", "function": function}


def _chat_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    converted: list[dict] = []
    names: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ChatRequestError(f"tools[{index}] must be an object")
        kind = tool.get("type")
        if kind == "namespace":
            unknown = set(tool) - {"type", "name", "description", "tools"}
            if unknown:
                raise ChatRequestError(
                    f"tools[{index}] has unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            namespace = tool.get("name")
            nested = tool.get("tools")
            if not isinstance(namespace, str) or not namespace:
                raise ChatRequestError(
                    f"tools[{index}].name must be a non-empty string"
                )
            if not isinstance(nested, list) or not nested:
                raise ChatRequestError(
                    f"tools[{index}].tools must be a non-empty array"
                )
            namespace_description = tool.get("description")
            if namespace_description is not None and not isinstance(
                namespace_description, str
            ):
                raise ChatRequestError(
                    f"tools[{index}].description must be a string"
                )
            for nested_index, nested_tool in enumerate(nested):
                if not isinstance(nested_tool, dict):
                    raise ChatRequestError(
                        f"tools[{index}].tools[{nested_index}] must be an object"
                    )
                nested_name = nested_tool.get("name")
                if not isinstance(nested_name, str) or not nested_name:
                    raise ChatRequestError(
                        f"tools[{index}].tools[{nested_index}].name "
                        "must be a non-empty string"
                    )
                encoded = _namespaced_name(namespace, nested_name)
                if encoded in names:
                    raise ChatRequestError(
                        f"tools[{index}].tools[{nested_index}] resolves to duplicate "
                        f"function name {encoded!r}"
                    )
                names.add(encoded)
                converted.append(
                    _function_tool(
                        nested_tool,
                        path=f"tools[{index}].tools[{nested_index}]",
                        name_override=encoded,
                        description_prefix=(
                            f"Namespace {namespace!r}."
                            + (
                                f" {namespace_description}"
                                if namespace_description
                                else ""
                            )
                        ),
                    )
                )
            continue
        if kind == "web_search" and tool.get("external_web_access") is False:
            # Codex attaches search configuration (filters, location, context
            # size) even when external access is disabled; the settings of a
            # tool that can never run are accepted and ignored.
            unknown = set(tool) - {
                "type",
                "external_web_access",
                "filters",
                "user_location",
                "search_context_size",
                "search_content_types",
            }
            if unknown:
                raise ChatRequestError(
                    f"tools[{index}] has unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            # Codex declares its built-in web tool even when the current run
            # explicitly disables external access.  Keeping it in the response
            # envelope but omitting it from model-visible callable functions is
            # truthful: the disabled tool cannot be selected or executed.
            continue
        if kind != "function":
            keys = sorted(tool)
            raise ChatRequestError(
                f"tools[{index}].type {kind!r} is not supported; expected 'function' "
                f"(fields: {', '.join(keys)})"
            )
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise ChatRequestError(f"tools[{index}].name must be a non-empty string")
        if name in names:
            raise ChatRequestError(f"tools[{index}].name {name!r} is duplicated")
        names.add(name)
        converted.append(_function_tool(tool, path=f"tools[{index}]"))
    return converted


def _chat_tool_choice(
    choice: str | dict | None,
    tools: list[dict] | None,
) -> str | dict | None:
    if not isinstance(choice, dict):
        return choice
    if choice.get("type") != "function":
        raise ChatRequestError("named tool_choice.type must be 'function'")
    name = choice.get("name")
    if not isinstance(name, str) or not name:
        raise ChatRequestError("named tool_choice.name must be a non-empty string")
    namespace = choice.get("namespace")
    if namespace is not None and (
        not isinstance(namespace, str) or not namespace
    ):
        raise ChatRequestError("named tool_choice.namespace must be a non-empty string")
    if set(choice) - {"type", "name", "namespace"}:
        raise ChatRequestError("named tool_choice has unsupported fields")
    if namespace is not None:
        selected = _namespaced_name(namespace, name)
        namespace_names = _namespace_names(tools)
        if selected not in namespace_names:
            raise ChatRequestError(
                f"named tool_choice references unknown namespace function {namespace!r}.{name!r}"
            )
        name = selected
    return {"type": "function", "function": {"name": name}}


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


def _to_chat_request(
    request: ResponsesRequest,
    items: Sequence[dict],
) -> ChatCompletionRequest:
    messages = _items_to_messages(items)
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


def _usage_payload(
    prompt: str,
    completions,
    usage,
) -> dict:
    input_tokens, output_tokens = resolve_usage_counts(
        usage, prompt=prompt, completions=completions
    )
    cached_tokens = resolve_cached_tokens(usage)
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }


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


def _namespace_names(tools: list[dict] | None) -> dict[str, tuple[str, str]]:
    names: dict[str, tuple[str, str]] = {}
    for tool in tools or ():
        if not isinstance(tool, dict) or tool.get("type") != "namespace":
            continue
        namespace = tool.get("name")
        if not isinstance(namespace, str):
            continue
        for nested in tool.get("tools", ()):
            if not isinstance(nested, dict):
                continue
            name = nested.get("name")
            if isinstance(name, str):
                names[_namespaced_name(namespace, name)] = (namespace, name)
    return names


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


def _response_envelope(
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


def _usage_payload_from_wire(usage: dict | None) -> dict:
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
    if request.store:
        store.save(response_id, stored_items + output, owner=owner)
    terminal_type = "response.completed" if status == "completed" else "response.incomplete"
    yield _sse(terminal_type, sequence, response=final_response)


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


async def _relay_auto_chat_stream(
    request: ResponsesRequest,
    upstream,
    *,
    response_id: str,
    created_at: int,
    stored_items: list[dict],
    store: ResponseStore,
    owner: str,
) -> AsyncIterator[str | bytes]:
    """Re-encode the orchestrated Chat Completions SSE stream as Responses SSE.

    ``upstream`` is this process's own chat-chunk stream for an AUTO model, so
    the frames are the internal wire format this release emits: ``: status``
    keep-alive comments, ``data: {chat chunk}`` frames, ``data: {"error": ...}``
    frames, and a final ``data: [DONE]``.
    """
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
    text_parts: list[str] = []
    finish_reason: str | None = None
    wire_usage: dict | None = None
    error_payload: dict | None = None

    async def frames() -> AsyncIterator[str]:
        buffer = ""
        async for chunk in upstream:
            buffer += chunk.decode() if isinstance(chunk, bytes) else chunk
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                if frame:
                    yield frame
        if buffer:
            yield buffer

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
        yield _sse(
            "response.content_part.added",
            sequence,
            item_id=message_id,
            output_index=0,
            content_index=0,
            part={"type": "output_text", "text": "", "annotations": [], "logprobs": []},
        )
        sequence += 1
        async for frame in frames():
            if frame.startswith(":"):
                # Orchestrator keep-alive comments stay comments on the
                # Responses stream (invisible to SDKs, reset idle timers).
                yield f"{frame}\n\n"
                continue
            if not frame.startswith("data:"):
                continue
            payload_text = frame[len("data:"):].strip()
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
                    yield text_delta_encoder.encode(sequence, content)
                    sequence += 1
                    text_parts.append(content)
    except Exception as error:
        logger.exception("Responses API orchestrated relay failed")
        error_payload = {"message": f"upstream backend error ({type(error).__name__})"}
    finally:
        aclose = getattr(upstream, "aclose", None)
        if aclose is not None:
            await aclose()

    text = "".join(text_parts)
    if error_payload is not None:
        message = error_payload.get("message") or "upstream backend error"
        yield _sse("error", sequence, code="server_error", message=message, param=None)
        sequence += 1
        failed_output = []
        if text:
            failed_output.append(
                {
                    "type": "message",
                    "id": message_id,
                    "role": "assistant",
                    "status": "incomplete",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                }
            )
        failed = _response_envelope(
            request,
            response_id=response_id,
            created_at=created_at,
            status="failed",
            output=failed_output,
            usage=_usage_payload_from_wire(wire_usage),
            error={"code": "server_error", "message": message},
        )
        yield _sse("response.failed", sequence, response=failed)
        return

    status, incomplete_details = _terminal_status_for((finish_reason,))
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
    final_response = _response_envelope(
        request,
        response_id=response_id,
        created_at=created_at,
        status=status,
        output=[output_item],
        usage=_usage_payload_from_wire(wire_usage),
        incomplete_details=incomplete_details,
    )
    if request.store:
        store.save(response_id, stored_items + [output_item], owner=owner)
    terminal_type = "response.completed" if status == "completed" else "response.incomplete"
    yield _sse(terminal_type, sequence, response=final_response)


async def _orchestrated_response(
    request: ResponsesRequest,
    chat_request: ChatCompletionRequest,
    http_request: Request,
    chat_dispatch,
    *,
    response_id: str,
    created_at: int,
    stored_items: list[dict],
    store: ResponseStore,
    owner: str,
    compaction_request: bool = False,
):
    """Serve an AUTO model by delegating to the Chat Completions handler.

    The chat handler owns the entire orchestration contract (validation,
    judge-first tenant reservation, admission, execution, public usage and
    metering), so this branch never runs the engine-only pipeline below and
    never records usage itself.
    """
    if request.stream and not request.tools and not compaction_request:
        live_request = chat_request.model_copy(
            update={"stream": True, "stream_options": StreamOptions(include_usage=True)}
        )
        delegated = await chat_dispatch(live_request, http_request)
        if isinstance(delegated, JSONResponse):
            return delegated
        return sse_response(
            _relay_auto_chat_stream(
                request,
                delegated.body_iterator,
                response_id=response_id,
                created_at=created_at,
                stored_items=stored_items,
                store=store,
                owner=owner,
            )
        )
    buffered_request = chat_request.model_copy(update={"stream": False})

    def outcome_from_payload(payload: dict) -> tuple[list[dict], dict, str, dict | None]:
        choices = payload.get("choices") or []
        message = (choices[0].get("message") if choices else None) or {}
        usage = _usage_payload_from_wire(payload.get("usage"))
        status, incomplete_details = _terminal_status_for(
            choice.get("finish_reason") for choice in choices
        )
        if compaction_request:
            if status != "completed":
                return [], usage, status, incomplete_details
            return _compaction_output_from_message(message), usage, status, None
        output = _output_items_from_message(request, message) if choices else []
        _apply_terminal_item_status(output, status)
        return output, usage, status, incomplete_details

    if request.stream:

        async def produce() -> tuple[list[dict], dict, str, dict | None]:
            delegated = await chat_dispatch(buffered_request, http_request)
            if not isinstance(delegated, JSONResponse):
                raise _BufferedFailure(
                    {
                        "message": "unexpected non-JSON chat dispatch reply",
                        "type": "upstream_error",
                        "code": "backend_error",
                    },
                    502,
                )
            payload = json.loads(bytes(delegated.body))
            if delegated.status_code != 200:
                raise _BufferedFailure(
                    payload.get("error")
                    or {
                        "message": "upstream backend error",
                        "type": "upstream_error",
                        "code": "backend_error",
                    },
                    delegated.status_code,
                )
            return outcome_from_payload(payload)

        return sse_response(
            _buffered_events(
                request,
                produce,
                response_id=response_id,
                created_at=created_at,
                stored_items=stored_items,
                store=store,
                owner=owner,
            )
        )
    delegated = await chat_dispatch(buffered_request, http_request)
    if not isinstance(delegated, JSONResponse):
        return upstream_error(RuntimeError("unexpected non-JSON chat dispatch reply"))
    if delegated.status_code != 200:
        return delegated
    try:
        output, usage, status, incomplete_details = outcome_from_payload(
            json.loads(bytes(delegated.body))
        )
    except _BufferedFailure as failure:
        return failure.json_response()
    response = _response_envelope(
        request,
        response_id=response_id,
        created_at=created_at,
        status=status,
        output=output,
        usage=usage,
        incomplete_details=incomplete_details,
    )
    if request.store:
        store.save(response_id, stored_items + output, owner=owner)
    return JSONResponse(content=response)


def add_responses_route(
    app: FastAPI,
    engines: Mapping,
    *,
    chat_templates=None,
    legacy_chat_models: AbstractSet[str] | None = None,
    orchestrated_models: AbstractSet[str] | None = None,
    chat_dispatch=None,
) -> ResponseStore:
    store = ResponseStore()
    app.state.response_store = store

    @app.get("/v1/responses")
    async def responses_upgrade_required() -> JSONResponse:
        # Codex tries a WebSocket upgrade first when it targets the built-in
        # openai provider (the Harbor/Terminal-Bench shape). The upgrade
        # arrives as a GET; 426 makes Codex fall back to HTTPS immediately
        # and silently instead of burning its stream-retry budget.
        return JSONResponse(
            status_code=426,
            content={
                "error": {
                    "message": (
                        "WebSocket transport is not supported; "
                        "retry over HTTPS"
                    ),
                    "type": "invalid_request_error",
                    "code": "upgrade_required",
                }
            },
        )

    @app.post("/v1/responses")
    async def responses(request: ResponsesRequest, http_request: Request):
        http_request.state.model = request.model
        metrics = getattr(http_request.app.state, "metrics", None)
        ingress_ns = getattr(http_request.state, "placement_started_ns", None)
        if metrics is not None and type(ingress_ns) is int:
            metrics.record_preplacement_phase(
                "responses",
                "ingress_to_handler",
                max(0, time.perf_counter_ns() - ingress_ns),
            )
        try:
            _validate_request_surface(request)
        except ChatRequestError as error:
            return _chat_error(error)
        engine = engines.get(request.model)
        orchestrated = (
            engine is None
            and chat_dispatch is not None
            and request.model in (orchestrated_models or ())
        )
        if engine is None and not orchestrated:
            return _request_error(
                f"model {request.model!r} not found",
                status_code=404,
                code="model_not_found",
            )
        owner = getattr(http_request.state, "tenant", None) or "default"
        context: list[dict] = []
        if request.previous_response_id:
            previous = store.get(request.previous_response_id, owner=owner)
            if previous is None:
                return _request_error("previous response not found", status_code=404)
            context.extend(previous)
        validation_started_ns = time.perf_counter_ns()
        try:
            current_items = _canonical_input(request.input)
            all_items = context + current_items
            compaction_request = _extract_compaction_trigger(all_items)
            work_items = (
                [item for item in all_items if item["type"] != "compaction_trigger"]
                if compaction_request
                else all_items
            )
            # A successful compaction replaces the continuation context. Keeping
            # ``work_items`` here would store the full pre-compaction history next
            # to its summary, so a later previous_response_id request would grow
            # the prompt instead of compacting it.
            stored_items = [] if compaction_request else work_items
            _validate_function_outputs(work_items)
            prompt_items = (
                work_items + [_COMPACTION_INSTRUCTION_ITEM]
                if compaction_request
                else work_items
            )
            chat_request = _to_chat_request(request, prompt_items)
            if compaction_request:
                # The summary is a plain text turn: tools cannot help it and a
                # tool call in its place would break the client's compaction
                # collection, so the summarization call runs tool-free with
                # enough room for a useful summary.
                chat_request = chat_request.model_copy(
                    update={
                        "tools": None,
                        "tool_choice": None,
                        "max_completion_tokens": (
                            request.max_output_tokens
                            or _COMPACTION_MAX_OUTPUT_TOKENS
                        ),
                    }
                )
            if not orchestrated:
                cache_key = request.prompt_cache_key or request.previous_response_id
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
                    request_id=(
                        getattr(http_request.state, "request_id", None)
                        or f"resp-{uuid.uuid4().hex[:12]}"
                    ),
                    cache_hint=CacheHint(session_id=cache_key) if cache_key else None,
                    priority=getattr(http_request.state, "priority", None),
                    scheduling_class=scheduling_class,
                    placement_started_ns=getattr(
                        http_request.state, "placement_started_ns", None
                    ),
                    legacy_chat_models=legacy_chat_models,
                )
        except ChatRequestError as error:
            return _chat_error(error)
        finally:
            if metrics is not None:
                metrics.record_preplacement_phase(
                    "responses",
                    "request_validation",
                    max(0, time.perf_counter_ns() - validation_started_ns),
                )
        if orchestrated:
            return await _orchestrated_response(
                request,
                chat_request,
                http_request,
                chat_dispatch,
                response_id=f"resp_{uuid.uuid4().hex}",
                created_at=int(time.time()),
                stored_items=stored_items,
                store=store,
                owner=owner,
                compaction_request=compaction_request,
            )
        prepare_started_ns = time.perf_counter_ns()
        try:
            await prepare_backend_request(
                validated.engine,
                validated.generation_request,
            )
        except UpstreamClientError as error:
            return _chat_error(chat_error_from_upstream_client_error(error))
        except ValueError as error:
            return _request_error(str(error))
        except RuntimeError as error:
            return upstream_error(error)
        finally:
            if metrics is not None:
                metrics.record_preplacement_phase(
                    "responses",
                    "backend_prepare",
                    max(0, time.perf_counter_ns() - prepare_started_ns),
                )
        admission_started_ns = time.perf_counter_ns()
        try:
            bound = await backend_admission_upper_bound_async(
                validated.engine,
                validated.generation_request,
            )
        except ValueError as error:
            return _request_error(str(error))
        except RuntimeError as error:
            return upstream_error(error)
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
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": "1"},
                    content={
                        "error": {
                            "message": (
                                f"tenant {owner!r} admission limit exceeded "
                                f"({admission.reason})"
                            ),
                            "type": "rate_limit_error",
                            "code": "tenant_rate_limited",
                        }
                    },
                )
        admission_ns += max(0, time.perf_counter_ns() - reserve_started_ns)
        if metrics is not None:
            metrics.record_preplacement_phase(
                "responses",
                "admission",
                admission_ns,
            )
            metrics.record_priority(
                validated.generation_request.scheduling_class,
                source="http",
            )

        response_id = f"resp_{uuid.uuid4().hex}"
        created_at = int(time.time())
        if request.stream and not request.tools and not compaction_request:
            return sse_response(
                _live_text_events(
                    request,
                    validated,
                    response_id=response_id,
                    created_at=created_at,
                    stored_items=stored_items,
                    store=store,
                    owner=owner,
                    http_request=http_request,
                )
            )

        async def produce() -> tuple[list[dict], dict, str, dict | None]:
            try:
                if admission is not None:
                    admission.mark_dispatched()
                execution = await execute_chat(validated)
            except ChatRequestError as error:
                if error.execution is not None:
                    _record_execution(http_request, request, error.execution)
                raise _BufferedFailure.from_chat_error(error) from error
            except Exception as error:
                logger.exception("Responses API upstream generation failed")
                raise _BufferedFailure(
                    {
                        "message": f"upstream backend error ({type(error).__name__})",
                        "type": "upstream_error",
                        "code": "backend_error",
                    },
                    502,
                ) from error
            try:
                _validate_parallel_tool_calls(request, execution)
            except ChatRequestError as error:
                _record_execution(http_request, request, execution)
                raise _BufferedFailure.from_chat_error(error) from error
            _record_execution(http_request, request, execution)
            usage = _usage_payload(
                execution.result.prompt,
                execution.result.completions,
                execution.result.usage,
            )
            status, incomplete_details = _terminal_status(execution)
            if compaction_request:
                if status != "completed":
                    return [], usage, status, incomplete_details
                message = (
                    execution.response.choices[0].message.model_dump(mode="json")
                    if execution.response.choices
                    else {}
                )
                return _compaction_output_from_message(message), usage, status, None
            output = _output_items(request, execution)
            _apply_terminal_item_status(output, status)
            return output, usage, status, incomplete_details

        if request.stream:
            return sse_response(
                _buffered_events(
                    request,
                    produce,
                    response_id=response_id,
                    created_at=created_at,
                    stored_items=stored_items,
                    store=store,
                    owner=owner,
                )
            )
        try:
            output, usage, status, incomplete_details = await produce()
        except _BufferedFailure as failure:
            return failure.json_response()
        response = _response_envelope(
            request,
            response_id=response_id,
            created_at=created_at,
            status=status,
            output=output,
            usage=usage,
            incomplete_details=incomplete_details,
        )
        if request.store:
            store.save(response_id, stored_items + output, owner=owner)
        return JSONResponse(content=response)

    return store
