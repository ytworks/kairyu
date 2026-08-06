"""OpenAI Responses API adapter with typed streaming and function tools.

The engine contract is intentionally shared with Chat Completions.  Responses
wire items are normalized into ``ChatCompletionRequest`` first, so chat
templates, tool-choice validation, and upstream capability preflight remain one
source of truth.
"""

from __future__ import annotations

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

from kairyu.engine.backend import CacheHint, GenerationResult, admission_upper_bound
from kairyu.entrypoints.server.chat_service import (
    ChatRequestError,
    ExecutedChat,
    ValidatedChatRequest,
    execute_chat,
    validate_chat_request,
)
from kairyu.entrypoints.server.metering import (
    record_state_usage,
    resolve_cached_tokens,
    resolve_usage_counts,
    stream_usage_owner_from_state,
)
from kairyu.entrypoints.server.protocol import ChatCompletionRequest
from kairyu.entrypoints.server.sse_encode import ResponsesTextDeltaSSEEncoder
from kairyu.entrypoints.server.sse_response import sse_response
from kairyu.sse import escape_json_line_separators

logger = logging.getLogger(__name__)

_ALLOWED_INCLUDE = {"reasoning.encrypted_content"}
_SUPPORTED_ROLES = {"user", "assistant", "system", "developer"}
_NAMESPACE_SEPARATOR = "__"


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
            if not isinstance(output, str):
                raise ChatRequestError(f"input[{index}].output must be a string")
            items.append(
                {"type": "function_call_output", "call_id": call_id, "output": output}
            )
            continue
        raise ChatRequestError(
            f"input[{index}].type {kind!r} is not supported; "
            "use message, function_call, or function_call_output"
        )
    return items


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
            unknown = set(tool) - {"type", "external_web_access"}
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
    choice = execution.response.choices[0]
    calls = choice.message.tool_calls or []
    if calls:
        namespaces = _namespace_names(request.tools)
        items = []
        for call in calls:
            namespace_name = namespaces.get(call.function.name)
            item = {
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "call_id": call.id,
                "type": "function_call",
                "name": (
                    namespace_name[1]
                    if namespace_name is not None
                    else call.function.name
                ),
                "arguments": call.function.arguments,
                "status": "completed",
            }
            if namespace_name is not None:
                item["namespace"] = namespace_name[0]
            items.append(item)
        return items
    return [
        {
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": choice.message.content or "",
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
    if any(
        completion.finish_reason in {"length", "max_tokens"}
        for completion in execution.result.completions
    ):
        return "incomplete", {"reason": "max_output_tokens"}
    return "completed", None


def _apply_terminal_item_status(output: list[dict], status: str) -> None:
    if status != "incomplete":
        return
    for item in output:
        if item["type"] == "message":
            item["status"] = "incomplete"


async def _buffered_events(
    request: ResponsesRequest,
    execution: ExecutedChat,
    *,
    response_id: str,
    created_at: int,
    stored_items: list[dict],
    store: ResponseStore,
    owner: str,
) -> AsyncIterator[str]:
    output = _output_items(request, execution)
    usage = _usage_payload(
        execution.result.prompt,
        execution.result.completions,
        execution.result.usage,
    )
    status, incomplete_details = _terminal_status(execution)
    _apply_terminal_item_status(output, status)
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
                delta = completion.text[sent:]
                sent = len(completion.text)
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


def add_responses_route(
    app: FastAPI,
    engines: Mapping,
    *,
    chat_templates=None,
    legacy_chat_models: AbstractSet[str] | None = None,
) -> ResponseStore:
    store = ResponseStore()
    app.state.response_store = store

    @app.post("/v1/responses")
    async def responses(request: ResponsesRequest, http_request: Request):
        http_request.state.model = request.model
        try:
            _validate_request_surface(request)
        except ChatRequestError as error:
            return _chat_error(error)
        engine = engines.get(request.model)
        if engine is None:
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
        try:
            current_items = _canonical_input(request.input)
            all_items = context + current_items
            _validate_function_outputs(all_items)
            chat_request = _to_chat_request(request, all_items)
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
            validated = validate_chat_request(
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
        try:
            bound = admission_upper_bound(validated.generation_request)
        except ValueError as error:
            return _request_error(str(error))
        admission = getattr(http_request.state, "tenant_admission", None)
        if admission is not None:
            admitted = admission.reserve_tokens(
                bound.tokens,
                refundable_on_exact_usage=bound.refundable_on_exact_usage,
            )
            metrics = getattr(http_request.app.state, "metrics", None)
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
        metrics = getattr(http_request.app.state, "metrics", None)
        if metrics is not None:
            metrics.record_priority(
                validated.generation_request.scheduling_class,
                source="http",
            )

        response_id = f"resp_{uuid.uuid4().hex}"
        created_at = int(time.time())
        if request.stream and not request.tools:
            return sse_response(
                _live_text_events(
                    request,
                    validated,
                    response_id=response_id,
                    created_at=created_at,
                    stored_items=all_items,
                    store=store,
                    owner=owner,
                    http_request=http_request,
                )
            )
        try:
            if admission is not None:
                admission.mark_dispatched()
            execution = await execute_chat(validated)
        except ChatRequestError as error:
            if error.execution is not None:
                _record_execution(http_request, request, error.execution)
            return _chat_error(error)
        except Exception as error:
            logger.exception("Responses API upstream generation failed")
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"upstream backend error ({type(error).__name__})",
                        "type": "upstream_error",
                        "code": "backend_error",
                    }
                },
            )
        try:
            _validate_parallel_tool_calls(request, execution)
        except ChatRequestError as error:
            _record_execution(http_request, request, execution)
            return _chat_error(error)
        _record_execution(http_request, request, execution)
        if request.stream:
            return sse_response(
                _buffered_events(
                    request,
                    execution,
                    response_id=response_id,
                    created_at=created_at,
                    stored_items=all_items,
                    store=store,
                    owner=owner,
                )
            )
        output = _output_items(request, execution)
        usage = _usage_payload(
            execution.result.prompt,
            execution.result.completions,
            execution.result.usage,
        )
        status, incomplete_details = _terminal_status(execution)
        _apply_terminal_item_status(output, status)
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
            store.save(response_id, all_items + output, owner=owner)
        return JSONResponse(content=response)

    return store
