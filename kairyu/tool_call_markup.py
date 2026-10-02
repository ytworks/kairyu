"""Kairyu's backend-neutral tool-call markup and the rules for reading it.

A backend that returns native tool calls renders each as
``<tool_call>{"name": ..., "arguments": ...}</tool_call>`` inside the text
(``kairyu.engine.openai_backend._message_text``). The public API publishes
such a payload as an OpenAI tool call only when it passes the rules here, and
a checklist judge reads exactly the calls the API would publish.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence

GENERIC_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value!r}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validate_finite_json(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if isinstance(value, Mapping):
        for nested in value.values():
            _validate_finite_json(nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_finite_json(nested)


def strict_json_loads(value: str) -> object:
    """JSON without NaN/Infinity or duplicate keys."""

    parsed = json.loads(
        value,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    _validate_finite_json(parsed)
    return parsed


def tool_call_payload(payload: object) -> tuple[str, str] | None:
    """``(name, arguments JSON)`` of a publishable call payload, else None.

    The name must be a non-empty string; the arguments (``arguments``, or the
    ``parameters`` alias) must be an object or a JSON string holding one.
    """

    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    arguments = payload.get("arguments", payload.get("parameters", {}))
    if not isinstance(arguments, (dict, str)):
        return None
    try:
        if isinstance(arguments, dict):
            return name, json.dumps(arguments, allow_nan=False)
        parsed_arguments = strict_json_loads(arguments)
        if not isinstance(parsed_arguments, dict):
            return None
        return name, arguments
    except (TypeError, ValueError, RecursionError):
        return None


def call_is_selected(
    name: str,
    mode: str,
    allowed_names: frozenset[str],
    named: str | None,
) -> bool:
    """Whether the caller's tool choice lets a call named ``name`` be published.

    ``mode`` is the normalized tool choice ("auto", "none", "required" or
    "named"); ``allowed_names`` the declared function names.
    """

    return mode != "none" and name in allowed_names and (named is None or name == named)


def tool_selection(
    tools: Sequence[Mapping[str, object]],
    tool_choice: object,
) -> tuple[str, frozenset[str], str | None]:
    """``(mode, allowed_names, named)`` of an already validated request."""

    allowed: set[str] = set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            allowed.add(function["name"])
    if tool_choice is None:
        return "auto", frozenset(allowed), None
    if isinstance(tool_choice, str):
        return tool_choice, frozenset(allowed), None
    function = tool_choice.get("function") if isinstance(tool_choice, Mapping) else None
    name = function.get("name") if isinstance(function, Mapping) else None
    return "named", frozenset(allowed), name if isinstance(name, str) else None


def generic_tool_calls(
    text: str,
    selection: tuple[str, frozenset[str], str | None] | None = None,
) -> tuple[str, list[dict[str, object]]]:
    """The text without its publishable calls, and those calls.

    Markup the API would not publish as a call stays in the text, as it would
    in the API's content; with a ``selection`` (:func:`tool_selection`), so
    does a call the caller's tool choice excludes.
    """

    calls: list[dict[str, object]] = []
    kept: list[str] = []
    cursor = 0
    for match in GENERIC_TOOL_CALL.finditer(text):
        try:
            payload = strict_json_loads(match.group(1))
        except (TypeError, ValueError, RecursionError):
            continue
        call = tool_call_payload(payload)
        if call is None:
            continue
        name, arguments = call
        if selection is not None and not call_is_selected(name, *selection):
            continue
        calls.append({"name": name, "arguments": json.loads(arguments)})
        kept.append(text[cursor : match.start()])
        cursor = match.end()
    kept.append(text[cursor:])
    return "".join(kept), calls


__all__ = [
    "GENERIC_TOOL_CALL",
    "call_is_selected",
    "tool_selection",
    "generic_tool_calls",
    "strict_json_loads",
    "tool_call_payload",
]
