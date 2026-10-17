"""Native upstream wires: OpenAI Responses and Anthropic Messages.

A request reaches an OpenAI-compatible upstream in the API its caller used at
L3 when the deployment declares that the upstream serves that API
(``upstream_responses_models`` / ``upstream_messages_models``, m9 D2 amendment
2026-10-17). Kairyu's internal request stays backend-neutral: the caller's
conversation travels as role-tagged chat messages (``conversation_prefix`` or
a ``MultimodalPrompt``), and this module expresses those messages, the tools
and the sampling intent on the selected wire, then reads the upstream's reply
back into Kairyu's text/reasoning/usage form. Tool calls keep Kairyu's
backend-neutral ``<tool_call>`` text form, exactly as on the chat wire.

Selection order: a Messages caller prefers Messages, then Responses, then
chat; a Responses caller prefers Responses, then chat; a chat caller always
uses chat. A wire is skipped when the upstream does not serve it or cannot
express the request (for example ``n > 1`` or token logprobs).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet

from kairyu.engine.backend import CALLER_APIS, GenerationUsage

NATIVE_WIRES = frozenset({"responses", "messages"})
_PREFERENCE: dict[str, tuple[str, ...]] = {
    "chat": (),
    "responses": ("responses",),
    "messages": ("messages", "responses"),
}
# OpenAI's Responses enum has no ``max``; upstream templates map ``xhigh``
# back to their top effort level (m11 D4 amendment 2026-10-17).
_RESPONSES_EFFORT = {"low": "low", "high": "high", "max": "xhigh"}
_RESPONSES_ENDPOINT = "responses"
_MESSAGES_ENDPOINT = "messages"


class WireError(ValueError):
    """The request cannot be expressed on the selected native wire."""


def wire_preference(caller_api: str) -> tuple[str, ...]:
    """Native wires to try, best first, for a caller API (chat is implicit)."""

    if caller_api not in CALLER_APIS:
        raise ValueError(f"caller_api must be one of {sorted(CALLER_APIS)}")
    return _PREFERENCE[caller_api]


def wire_endpoint(wire: str) -> str:
    if wire == "responses":
        return _RESPONSES_ENDPOINT
    if wire == "messages":
        return _MESSAGES_ENDPOINT
    raise ValueError(f"unknown native wire {wire!r}")


def validate_upstream_apis(apis: Iterable[str]) -> frozenset[str]:
    declared = frozenset(apis)
    unknown = declared - NATIVE_WIRES
    if unknown:
        raise ValueError(
            "upstream_apis may contain only responses and messages; got "
            + ", ".join(sorted(unknown))
        )
    return declared


def _text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, Mapping) and part.get("type") == "text"
        )
    return str(content)


def _arguments_text(arguments: object) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)


def _arguments_object(arguments: object) -> object:
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}
        try:
            return json.loads(arguments)
        except json.JSONDecodeError as error:
            raise WireError(
                "a tool call's arguments are not JSON and cannot be sent as an "
                "Anthropic tool_use input"
            ) from error
    return arguments if arguments is not None else {}


def _reasoning(message: Mapping[str, object]) -> str | None:
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def tool_call_text(name: str, arguments: object) -> str:
    """Kairyu's backend-neutral text form of one native tool call."""

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            pass
    return (
        "<tool_call>"
        + json.dumps(
            {"name": name, "arguments": arguments},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "</tool_call>"
    )


# --- OpenAI Responses -------------------------------------------------------


def _responses_content(content: object) -> object:
    if not isinstance(content, list):
        return _text(content)
    parts: list[dict[str, object]] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        if part.get("type") == "text":
            parts.append({"type": "input_text", "text": part.get("text", "")})
        elif part.get("type") == "image_url":
            image = part.get("image_url")
            image = image if isinstance(image, Mapping) else {"url": image}
            converted: dict[str, object] = {
                "type": "input_image",
                "image_url": image.get("url"),
                "detail": image.get("detail") or "auto",
            }
            parts.append(converted)
    return parts


def responses_input(
    messages: list[Mapping[str, object]],
    *,
    continue_final: bool = False,
) -> list[dict[str, object]]:
    """The chat messages as Responses input items, turn for turn.

    System and developer turns stay message items in place (the upstream
    renders ``instructions`` as the same leading system message), assistant
    reasoning becomes a reasoning item before its turn, tool calls become
    ``function_call`` items and tool results ``function_call_output`` items.
    A continued final assistant turn is marked ``incomplete`` so the upstream
    continues it instead of opening a new turn.
    """

    items: list[dict[str, object]] = []
    last = len(messages) - 1
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id") or ""),
                    "output": _text(content),
                }
            )
            continue
        if role == "assistant":
            reasoning = _reasoning(message)
            if reasoning is not None:
                items.append(
                    {
                        "type": "reasoning",
                        "summary": [],
                        "content": [{"type": "reasoning_text", "text": reasoning}],
                    }
                )
            text = _text(content)
            continued = continue_final and index == last
            if text or continued or not message.get("tool_calls"):
                item: dict[str, object] = {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
                if continued:
                    item["status"] = "incomplete"
                items.append(item)
            for call in message.get("tool_calls") or ():
                if not isinstance(call, Mapping):
                    continue
                function = call.get("function")
                if not isinstance(function, Mapping) or not function.get("name"):
                    continue
                items.append(
                    {
                        "type": "function_call",
                        "call_id": str(call.get("id") or ""),
                        "name": function["name"],
                        "arguments": _arguments_text(function.get("arguments")),
                    }
                )
            continue
        items.append(
            {
                "type": "message",
                "role": role,
                "content": _responses_content(content),
            }
        )
    return items


def responses_tools(tools: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for tool in tools:
        function = tool.get("function")
        if tool.get("type", "function") != "function" or not isinstance(function, Mapping):
            raise WireError("only function tools can be sent on the Responses wire")
        item: dict[str, object] = {"type": "function", "name": function.get("name")}
        for key in ("description", "parameters", "strict"):
            if key in function:
                item[key] = function[key]
        converted.append(item)
    return converted


def responses_tool_choice(choice: object) -> object:
    if choice is None or isinstance(choice, str):
        return choice
    if isinstance(choice, Mapping):
        function = choice.get("function")
        name = function.get("name") if isinstance(function, Mapping) else choice.get("name")
        if isinstance(name, str) and name:
            return {"type": "function", "name": name}
    raise WireError("tool_choice cannot be expressed on the Responses wire")


def responses_text_format(response_format: object) -> dict[str, object] | None:
    if response_format is None:
        return None
    if not isinstance(response_format, Mapping):
        raise WireError("response_format must be an object")
    kind = response_format.get("type")
    if kind in {"text", "json_object"}:
        return {"format": {"type": kind}}
    if kind == "json_schema":
        schema = response_format.get("json_schema")
        if not isinstance(schema, Mapping):
            raise WireError("response_format.json_schema must be an object")
        converted = {"type": "json_schema", "name": schema.get("name") or "response"}
        for key in ("schema", "strict", "description"):
            if key in schema:
                converted[key] = schema[key]
        return {"format": converted}
    raise WireError(f"response_format type {kind!r} cannot be sent on the Responses wire")


# Chat-wire sampling fields the Responses wire expresses, by upstream: vLLM's
# ResponsesRequest executes these extensions; a Kairyu replica's Responses
# boundary is the strict OpenAI surface.
_RESPONSES_FIELDS = {
    "vllm": frozenset(
        {
            "temperature",
            "top_p",
            "top_k",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "seed",
            "stop",
            "ignore_eos",
            "max_tokens",
            "max_completion_tokens",
            "priority",
            "reasoning_effort",
            "chat_template_kwargs",
            "response_format",
        }
    ),
    "kairyu": frozenset(
        {
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "parallel_tool_calls",
            "priority",
            "reasoning_effort",
            "response_format",
        }
    ),
}
# A chat field a wire cannot carry is still expressible when its value is the
# one the upstream applies without it, by (upstream, wire): vLLM's
# ResponsesRequest never reads min_p and always samples with 0.0; vLLM builds
# an Anthropic request with the default priority 0; a Kairyu replica's
# Messages boundary takes priority only from its own tenancy, else 0. vLLM
# keeps a single tool call only on its chat path (v0.23.0 and v0.31.0): its
# Responses serving never applies parallel_tool_calls=false and v0.23.0's
# Anthropic tool_choice drops disable_parallel_tool_use, so only the default
# true is expressible there. Any other value would be silently dropped, so the
# request takes the next wire.
_NEUTRAL_FIELDS: dict[tuple[str, str], dict[str, object]] = {
    ("vllm", "responses"): {"n": 1, "min_p": 0.0, "parallel_tool_calls": True},
    ("vllm", "messages"): {"n": 1, "priority": 0, "parallel_tool_calls": True},
    ("kairyu", "responses"): {"n": 1},
    ("kairyu", "messages"): {"n": 1, "priority": 0},
}


def _unsupported(
    fields: AbstractSet[str],
    validated: Mapping[str, object],
    neutral: Mapping[str, object],
) -> list[str]:
    return sorted(
        key
        for key, value in validated.items()
        if key not in fields and neutral.get(key, object()) != value
    )


def responses_expressible(upstream: str, validated: Mapping[str, object]) -> bool:
    fields = _RESPONSES_FIELDS.get(upstream)
    if fields is None:
        return False
    if _unsupported(fields, validated, _NEUTRAL_FIELDS[(upstream, "responses")]):
        return False
    response_format = validated.get("response_format")
    if response_format is not None:
        try:
            responses_text_format(response_format)
        except WireError:
            return False
    return True


def responses_body(
    *,
    model: str,
    upstream: str,
    messages: list[Mapping[str, object]],
    validated: Mapping[str, object],
    tools: list[Mapping[str, object]] | None,
    tool_choice: object,
    continue_final: bool,
) -> dict[str, object]:
    """A complete stateless Responses request for the same generation."""

    unsupported = _unsupported(
        _RESPONSES_FIELDS[upstream], validated, _NEUTRAL_FIELDS[(upstream, "responses")]
    )
    if unsupported:
        raise WireError(
            "request fields cannot be sent on the Responses wire: " + ", ".join(unsupported)
        )
    body: dict[str, object] = {
        "model": model,
        "input": responses_input(messages, continue_final=continue_final),
        # Kairyu owns conversation state; the upstream stays stateless.
        "store": False,
    }
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "seed",
        "stop",
        "ignore_eos",
        "parallel_tool_calls",
        "priority",
        "chat_template_kwargs",
    ):
        if key in validated:
            body[key] = validated[key]
    for key in ("max_tokens", "max_completion_tokens"):
        if key in validated:
            body["max_output_tokens"] = validated[key]
    effort = validated.get("reasoning_effort")
    if effort is not None:
        body["reasoning"] = {"effort": _RESPONSES_EFFORT[str(effort)]}
    text = responses_text_format(validated.get("response_format"))
    if text is not None:
        body["text"] = text
    if tools:
        body["tools"] = responses_tools(tools)
        if tool_choice is not None:
            body["tool_choice"] = responses_tool_choice(tool_choice)
    return body


_INCOMPLETE_FINISH = {"max_output_tokens": "length", "content_filter": "content_filter"}


def _responses_usage(usage: object) -> GenerationUsage | None:
    if not isinstance(usage, Mapping):
        return None
    details = usage.get("input_tokens_details") or {}
    return GenerationUsage(
        prompt_tokens=int(usage.get("input_tokens") or 0),
        completion_tokens=int(usage.get("output_tokens") or 0),
        cached_tokens=int(details.get("cached_tokens") or 0)
        if isinstance(details, Mapping)
        else 0,
    )


def _responses_finish(response: Mapping[str, object], calls: int) -> str:
    status = response.get("status")
    if status == "incomplete":
        details = response.get("incomplete_details") or {}
        reason = details.get("reason") if isinstance(details, Mapping) else None
        return _INCOMPLETE_FINISH.get(str(reason), "length")
    if status not in {"completed", None}:
        raise RuntimeError(f"upstream Responses status {status!r}")
    return "tool_calls" if calls else "stop"


def _responses_error(payload: Mapping[str, object]) -> RuntimeError:
    error = payload.get("error")
    if not isinstance(error, Mapping):
        response = payload.get("response")
        error = response.get("error") if isinstance(response, Mapping) else None
    message = error.get("message") if isinstance(error, Mapping) else None
    return RuntimeError(f"upstream Responses stream failed: {message or 'unknown error'}")


def parse_responses_result(
    data: Mapping[str, object],
) -> tuple[str, str | None, str, GenerationUsage | None]:
    """(text, reasoning, finish_reason, usage) of a non-streaming response."""

    text: list[str] = []
    reasoning: list[str] = []
    calls = 0
    for item in data.get("output") or ():
        if not isinstance(item, Mapping):
            continue
        kind = item.get("type")
        if kind == "message":
            for part in item.get("content") or ():
                if isinstance(part, Mapping) and part.get("type") == "output_text":
                    text.append(str(part.get("text") or ""))
        elif kind == "reasoning":
            parts = item.get("content") or item.get("summary") or ()
            for part in parts:
                if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                    reasoning.append(part["text"])
        elif kind == "function_call":
            calls += 1
            text.append(tool_call_text(str(item.get("name")), item.get("arguments")))
    return (
        "".join(text),
        "".join(reasoning) if reasoning else None,
        _responses_finish(data, calls),
        _responses_usage(data.get("usage")),
    )


class ResponsesStreamReader:
    """Folds a Responses event stream into text/reasoning deltas.

    Returns ``(text_delta, reasoning_delta)`` per event. Function calls are
    emitted as one ``<tool_call>`` text delta when their item finishes, the
    same form the chat wire produces at its finish chunk.
    """

    def __init__(self) -> None:
        self.finish_reason: str | None = None
        self.usage: GenerationUsage | None = None
        self.finished = False
        # The terminal response object (carries Kairyu extensions when the
        # upstream is a Kairyu replica).
        self.response: Mapping[str, object] | None = None
        self._calls: dict[object, dict[str, object]] = {}
        self._call_count = 0

    def feed(self, event: Mapping[str, object]) -> tuple[str, str]:
        kind = event.get("type")
        if kind == "response.output_text.delta":
            return str(event.get("delta") or ""), ""
        if kind in {"response.reasoning_text.delta", "response.reasoning_summary_text.delta"}:
            return "", str(event.get("delta") or "")
        if kind == "response.output_item.added":
            item = event.get("item")
            if isinstance(item, Mapping) and item.get("type") == "function_call":
                self._calls[event.get("output_index")] = {
                    "name": item.get("name"),
                    "arguments": [str(item.get("arguments") or "")],
                }
            return "", ""
        if kind == "response.function_call_arguments.delta":
            call = self._calls.get(event.get("output_index"))
            if call is not None:
                call["arguments"].append(str(event.get("delta") or ""))  # type: ignore[union-attr]
            return "", ""
        if kind == "response.output_item.done":
            item = event.get("item")
            if not isinstance(item, Mapping) or item.get("type") != "function_call":
                return "", ""
            pending = self._calls.pop(event.get("output_index"), None)
            name = item.get("name") or (pending or {}).get("name")
            arguments = item.get("arguments")
            if arguments is None and pending is not None:
                arguments = "".join(pending["arguments"])  # type: ignore[arg-type]
            self._call_count += 1
            return tool_call_text(str(name), arguments), ""
        if kind in {"response.completed", "response.incomplete"}:
            response = event.get("response")
            if not isinstance(response, Mapping):
                raise RuntimeError("upstream Responses terminal event has no response")
            self.response = response
            self.usage = _responses_usage(response.get("usage"))
            self.finish_reason = _responses_finish(response, self._call_count)
            self.finished = True
            return "", ""
        if kind in {"response.failed", "error"}:
            raise _responses_error(event)
        return "", ""


# --- Anthropic Messages -----------------------------------------------------


def _anthropic_image(part: Mapping[str, object]) -> dict[str, object]:
    image = part.get("image_url")
    url = image.get("url") if isinstance(image, Mapping) else image
    if not isinstance(url, str):
        raise WireError("an image part has no URL")
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = header[len("data:") :].split(";", 1)[0]
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _anthropic_user_content(content: object) -> object:
    if not isinstance(content, list):
        return _text(content)
    blocks: list[dict[str, object]] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        if part.get("type") == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif part.get("type") == "image_url":
            blocks.append(_anthropic_image(part))
    return blocks


def anthropic_conversation(
    messages: list[Mapping[str, object]],
    *,
    thinking: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """(system blocks, messages) for the same conversation.

    Leading system/developer turns become the top-level system blocks; one
    later in the conversation has no Anthropic representation. Consecutive
    tool results share one user turn, as Anthropic tool_result blocks.
    Replayed assistant reasoning becomes a thinking block only for upstreams
    that accept one (a Kairyu replica's Messages boundary does not).
    """

    system: list[dict[str, object]] = []
    converted: list[dict[str, object]] = []
    started = False
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role in {"system", "developer"}:
            if started:
                raise WireError(
                    "a system message after the first user or assistant turn cannot "
                    "be sent on the Anthropic Messages wire"
                )
            text = _text(content)
            if text:
                system.append({"type": "text", "text": text})
            continue
        started = True
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": str(message.get("tool_call_id") or ""),
                "content": _text(content),
            }
            previous = converted[-1] if converted else None
            if (
                previous is not None
                and previous["role"] == "user"
                and isinstance(previous["content"], list)
                and previous["content"]
                and all(item.get("type") == "tool_result" for item in previous["content"])
            ):
                previous["content"].append(block)
            else:
                converted.append({"role": "user", "content": [block]})
            continue
        if role == "assistant":
            blocks: list[dict[str, object]] = []
            reasoning = _reasoning(message) if thinking else None
            if reasoning is not None:
                blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
            text = _text(content)
            if text:
                blocks.append({"type": "text", "text": text})
            for call in message.get("tool_calls") or ():
                if not isinstance(call, Mapping):
                    continue
                function = call.get("function")
                if not isinstance(function, Mapping) or not function.get("name"):
                    continue
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(call.get("id") or ""),
                        "name": function["name"],
                        "input": _arguments_object(function.get("arguments")),
                    }
                )
            converted.append({"role": "assistant", "content": blocks or text})
            continue
        if role != "user":
            raise WireError(f"message role {role!r} cannot be sent on the Anthropic Messages wire")
        converted.append({"role": "user", "content": _anthropic_user_content(content)})
    return system, converted


def anthropic_tools(tools: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for tool in tools:
        function = tool.get("function")
        if tool.get("type", "function") != "function" or not isinstance(function, Mapping):
            raise WireError("only function tools can be sent on the Anthropic Messages wire")
        item: dict[str, object] = {
            "name": function.get("name"),
            "input_schema": function.get("parameters") or {"type": "object"},
        }
        if "description" in function:
            item["description"] = function["description"]
        converted.append(item)
    return converted


def anthropic_tool_choice(choice: object, parallel_tool_calls: object) -> dict | None:
    converted: dict[str, object] | None
    if choice is None:
        converted = None
    elif choice == "auto":
        converted = {"type": "auto"}
    elif choice == "required":
        converted = {"type": "any"}
    elif choice == "none":
        converted = {"type": "none"}
    elif isinstance(choice, Mapping):
        function = choice.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(name, str) or not name:
            raise WireError("tool_choice cannot be expressed on the Anthropic Messages wire")
        converted = {"type": "tool", "name": name}
    else:
        raise WireError("tool_choice cannot be expressed on the Anthropic Messages wire")
    if parallel_tool_calls is False:
        converted = {**(converted or {"type": "auto"}), "disable_parallel_tool_use": True}
    return converted


_MESSAGES_FIELDS = {
    "vllm": frozenset(
        {
            "temperature",
            "top_p",
            "top_k",
            "stop",
            "max_tokens",
            "max_completion_tokens",
            "reasoning_effort",
            "chat_template_kwargs",
            "response_format",
        }
    ),
    "kairyu": frozenset(
        {
            "temperature",
            "top_p",
            "top_k",
            "stop",
            "max_tokens",
            "max_completion_tokens",
            "parallel_tool_calls",
            "reasoning_effort",
        }
    ),
}


def messages_expressible(
    upstream: str,
    validated: Mapping[str, object],
    messages: list[Mapping[str, object]],
    *,
    max_model_len: int | None,
) -> bool:
    fields = _MESSAGES_FIELDS.get(upstream)
    if fields is None or _unsupported(
        fields, validated, _NEUTRAL_FIELDS[(upstream, "messages")]
    ):
        return False
    response_format = validated.get("response_format")
    if response_format is not None and (
        not isinstance(response_format, Mapping)
        or response_format.get("type") != "json_schema"
    ):
        return False
    if (
        "max_tokens" not in validated
        and "max_completion_tokens" not in validated
        and max_model_len is None
    ):
        return False
    try:
        anthropic_conversation(messages, thinking=upstream != "kairyu")
    except WireError:
        return False
    return True


def messages_body(
    *,
    model: str,
    upstream: str,
    messages: list[Mapping[str, object]],
    validated: Mapping[str, object],
    tools: list[Mapping[str, object]] | None,
    tool_choice: object,
    max_model_len: int | None,
) -> dict[str, object]:
    """A complete Anthropic Messages request for the same generation."""

    unsupported = _unsupported(
        _MESSAGES_FIELDS[upstream], validated, _NEUTRAL_FIELDS[(upstream, "messages")]
    )
    if unsupported:
        raise WireError(
            "request fields cannot be sent on the Anthropic Messages wire: "
            + ", ".join(unsupported)
        )
    system, conversation = anthropic_conversation(messages, thinking=upstream != "kairyu")
    max_tokens = validated.get("max_tokens", validated.get("max_completion_tokens"))
    if max_tokens is None:
        # Anthropic requires max_tokens; an omitted limit means the remaining
        # context, which the configured context length bounds.
        max_tokens = max_model_len
    if max_tokens is None:
        raise WireError("the Anthropic Messages wire requires max_tokens")
    body: dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": conversation,
    }
    if system:
        body["system"] = system
    for key in ("temperature", "top_p", "top_k", "chat_template_kwargs"):
        if key in validated:
            body[key] = validated[key]
    if "stop" in validated:
        body["stop_sequences"] = validated["stop"]
    output_config: dict[str, object] = {}
    effort = validated.get("reasoning_effort")
    if effort is not None:
        output_config["effort"] = effort
    response_format = validated.get("response_format")
    if isinstance(response_format, Mapping):
        schema = response_format.get("json_schema")
        if not isinstance(schema, Mapping) or not isinstance(schema.get("schema"), Mapping):
            raise WireError("response_format.json_schema.schema must be an object")
        output_config["format"] = {"type": "json_schema", "schema": schema["schema"]}
    if output_config:
        body["output_config"] = output_config
    if tools:
        body["tools"] = anthropic_tools(tools)
    choice = anthropic_tool_choice(
        tool_choice if tools else None,
        validated.get("parallel_tool_calls") if tools else None,
    )
    if choice is not None:
        body["tool_choice"] = choice
    return body


_STOP_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "refusal": "content_filter",
}


def _anthropic_usage(usage: object, *, input_includes_cache: bool) -> GenerationUsage | None:
    if not isinstance(usage, Mapping):
        return None
    cached = int(usage.get("cache_read_input_tokens") or 0)
    prompt = int(usage.get("input_tokens") or 0)
    if not input_includes_cache:
        # Anthropic counts cache reads and writes outside input_tokens.
        prompt += cached + int(usage.get("cache_creation_input_tokens") or 0)
    return GenerationUsage(
        prompt_tokens=prompt,
        completion_tokens=int(usage.get("output_tokens") or 0),
        cached_tokens=cached,
    )


def parse_messages_result(
    data: Mapping[str, object],
    *,
    input_includes_cache: bool,
) -> tuple[str, str | None, str, GenerationUsage | None]:
    text: list[str] = []
    reasoning: list[str] = []
    for block in data.get("content") or ():
        if not isinstance(block, Mapping):
            continue
        kind = block.get("type")
        if kind == "text":
            text.append(str(block.get("text") or ""))
        elif kind == "thinking":
            reasoning.append(str(block.get("thinking") or ""))
        elif kind == "tool_use":
            text.append(tool_call_text(str(block.get("name")), block.get("input")))
    finish = _STOP_FINISH.get(str(data.get("stop_reason")), "stop")
    return (
        "".join(text),
        "".join(reasoning) if reasoning else None,
        finish,
        _anthropic_usage(data.get("usage"), input_includes_cache=input_includes_cache),
    )


class MessagesStreamReader:
    """Folds an Anthropic Messages event stream into text/reasoning deltas."""

    def __init__(self, *, input_includes_cache: bool) -> None:
        self._input_includes_cache = input_includes_cache
        self.finish_reason: str | None = None
        self.usage: GenerationUsage | None = None
        self.finished = False
        self._usage: dict[str, object] = {}
        self._tools: dict[object, dict[str, object]] = {}

    def feed(self, event: Mapping[str, object]) -> tuple[str, str]:
        kind = event.get("type")
        if kind == "message_start":
            message = event.get("message")
            if isinstance(message, Mapping) and isinstance(message.get("usage"), Mapping):
                self._usage.update(message["usage"])
            return "", ""
        if kind == "content_block_start":
            block = event.get("content_block")
            if isinstance(block, Mapping) and block.get("type") == "tool_use":
                self._tools[event.get("index")] = {"name": block.get("name"), "json": []}
            return "", ""
        if kind == "content_block_delta":
            delta = event.get("delta")
            if not isinstance(delta, Mapping):
                return "", ""
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                return str(delta.get("text") or ""), ""
            if delta_type == "thinking_delta":
                return "", str(delta.get("thinking") or "")
            if delta_type == "input_json_delta":
                tool = self._tools.get(event.get("index"))
                if tool is not None:
                    tool["json"].append(str(delta.get("partial_json") or ""))  # type: ignore[union-attr]
            return "", ""
        if kind == "content_block_stop":
            tool = self._tools.pop(event.get("index"), None)
            if tool is None:
                return "", ""
            raw = "".join(tool["json"])  # type: ignore[arg-type]
            return tool_call_text(str(tool["name"]), raw or "{}"), ""
        if kind == "message_delta":
            delta = event.get("delta")
            if isinstance(delta, Mapping) and delta.get("stop_reason") is not None:
                self.finish_reason = _STOP_FINISH.get(str(delta["stop_reason"]), "stop")
            if isinstance(event.get("usage"), Mapping):
                self._usage.update(event["usage"])
            return "", ""
        if kind == "message_stop":
            self.usage = _anthropic_usage(
                self._usage,
                input_includes_cache=self._input_includes_cache,
            )
            self.finish_reason = self.finish_reason or "stop"
            self.finished = True
            return "", ""
        if kind == "error":
            error = event.get("error")
            message = error.get("message") if isinstance(error, Mapping) else None
            raise RuntimeError(
                f"upstream Messages stream failed: {message or 'unknown error'}"
            )
        return "", ""
