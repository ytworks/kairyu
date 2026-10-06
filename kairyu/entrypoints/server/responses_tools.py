"""Responses tools, tool_choice, text.format, and reasoning mapped onto chat."""

from __future__ import annotations

from kairyu.entrypoints.server.protocol import normalize_reasoning_effort
from kairyu.entrypoints.server.responses_protocol import ResponsesError

_NAMESPACE_SEPARATOR = "__"
# Hosted web search is declared, never executed: Codex auto-declares it (live
# or indexed) in full-access runs, so the declaration and its search settings
# are accepted, echoed in ``response.tools``, and kept out of the callable
# functions the model sees — no ``web_search_call`` item can ever be emitted.
_WEB_SEARCH_TYPES = frozenset(
    {
        "web_search",
        "web_search_2025_08_26",
        "web_search_preview",
        "web_search_preview_2025_03_11",
    }
)
_WEB_SEARCH_FIELDS = frozenset(
    {
        "type",
        "external_web_access",
        "filters",
        "user_location",
        "search_context_size",
        "search_content_types",
    }
)


def _unknown_fields(path: str, item: dict, allowed: set[str] | frozenset[str]) -> None:
    unknown = sorted(set(item) - allowed)
    if unknown:
        raise ResponsesError(
            f"{path} has unsupported fields: " + ", ".join(unknown),
            param=f"{path}.{unknown[0]}",
            code="unknown_parameter",
        )


def _invalid(param: str, message: str) -> ResponsesError:
    return ResponsesError(message, param=param, code="invalid_value")


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
        raise _invalid(f"{path}.type", f"{path}.type must be 'function'")
    name = tool.get("name")
    parameters = tool.get("parameters", {})
    if not isinstance(name, str) or not name:
        raise _invalid(f"{path}.name", f"{path}.name must be a non-empty string")
    if not isinstance(parameters, dict):
        raise _invalid(f"{path}.parameters", f"{path}.parameters must be an object")
    _unknown_fields(
        path,
        tool,
        {"type", "name", "description", "parameters", "strict", "defer_loading"},
    )
    function = {"name": name_override or name, "parameters": parameters}
    description = tool.get("description")
    if description_prefix:
        description = (
            f"{description_prefix} {description}" if description else description_prefix
        )
    if description is not None:
        if not isinstance(description, str):
            raise _invalid(f"{path}.description", f"{path}.description must be a string")
        function["description"] = description
    if tool.get("strict") is not None:
        if not isinstance(tool["strict"], bool):
            raise _invalid(f"{path}.strict", f"{path}.strict must be a boolean")
        function["strict"] = tool["strict"]
    if tool.get("defer_loading") is not None and not isinstance(
        tool["defer_loading"], bool
    ):
        raise _invalid(f"{path}.defer_loading", f"{path}.defer_loading must be a boolean")
    return {"type": "function", "function": function}


def _chat_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    converted: list[dict] = []
    names: set[str] = set()
    for index, tool in enumerate(tools):
        path = f"tools[{index}]"
        if not isinstance(tool, dict):
            raise _invalid(path, f"{path} must be an object")
        kind = tool.get("type")
        if not isinstance(kind, str):
            raise _invalid(f"{path}.type", f"{path}.type must be a string")
        if kind == "namespace":
            _unknown_fields(path, tool, {"type", "name", "description", "tools"})
            namespace = tool.get("name")
            nested = tool.get("tools")
            if not isinstance(namespace, str) or not namespace:
                raise _invalid(f"{path}.name", f"{path}.name must be a non-empty string")
            if not isinstance(nested, list) or not nested:
                raise _invalid(f"{path}.tools", f"{path}.tools must be a non-empty array")
            namespace_description = tool.get("description")
            if namespace_description is not None and not isinstance(
                namespace_description, str
            ):
                raise _invalid(f"{path}.description", f"{path}.description must be a string")
            for nested_index, nested_tool in enumerate(nested):
                nested_path = f"{path}.tools[{nested_index}]"
                if not isinstance(nested_tool, dict):
                    raise _invalid(nested_path, f"{nested_path} must be an object")
                nested_name = nested_tool.get("name")
                if not isinstance(nested_name, str) or not nested_name:
                    raise _invalid(
                        f"{nested_path}.name", f"{nested_path}.name must be a non-empty string"
                    )
                encoded = _namespaced_name(namespace, nested_name)
                if encoded in names:
                    raise _invalid(
                        f"{nested_path}.name",
                        f"{nested_path} resolves to duplicate function name {encoded!r}",
                    )
                names.add(encoded)
                converted.append(
                    _function_tool(
                        nested_tool,
                        path=nested_path,
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
        if kind in _WEB_SEARCH_TYPES:
            _unknown_fields(path, tool, _WEB_SEARCH_FIELDS)
            continue
        if kind != "function":
            raise ResponsesError(
                f"{path}.type {kind!r} is not supported; Kairyu executes no hosted "
                "or built-in tools and accepts 'function', 'namespace', and web "
                "search declarations",
                param=f"{path}.type",
                code="unsupported_value",
            )
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise _invalid(f"{path}.name", f"{path}.name must be a non-empty string")
        if name in names:
            raise _invalid(f"{path}.name", f"{path}.name {name!r} is duplicated")
        names.add(name)
        converted.append(_function_tool(tool, path=path))
    return converted


def _chat_tool_choice(
    choice: str | dict | None,
    tools: list[dict] | None,
) -> str | dict | None:
    if not isinstance(choice, dict):
        return choice
    kind = choice.get("type")
    if kind != "function":
        if isinstance(kind, str):
            raise ResponsesError(
                f"tool_choice.type {kind!r} is not supported; use none, auto, "
                "required, or a named function",
                param="tool_choice.type",
                code="unsupported_value",
            )
        raise _invalid("tool_choice.type", "named tool_choice.type must be 'function'")
    name = choice.get("name")
    if not isinstance(name, str) or not name:
        raise _invalid("tool_choice.name", "named tool_choice.name must be a non-empty string")
    namespace = choice.get("namespace")
    if namespace is not None and (
        not isinstance(namespace, str) or not namespace
    ):
        raise _invalid(
            "tool_choice.namespace", "named tool_choice.namespace must be a non-empty string"
        )
    _unknown_fields("tool_choice", choice, {"type", "name", "namespace"})
    if namespace is not None:
        selected = _namespaced_name(namespace, name)
        namespace_names = _namespace_names(tools)
        if selected not in namespace_names:
            raise _invalid(
                "tool_choice.name",
                f"named tool_choice references unknown namespace function {namespace!r}.{name!r}",
            )
        name = selected
    return {"type": "function", "function": {"name": name}}


def _response_format(text: dict | None) -> dict | None:
    if text is None:
        return None
    _unknown_fields("text", text, {"format", "verbosity"})
    verbosity = text.get("verbosity")
    if verbosity not in (None, "low", "medium", "high"):
        raise _invalid("text.verbosity", "text.verbosity must be low, medium, or high")
    fmt = text.get("format")
    if fmt is None:
        return None
    if not isinstance(fmt, dict):
        raise _invalid("text.format", "text.format must be an object")
    kind = fmt.get("type")
    if kind == "text":
        return {"type": "text"}
    if kind == "json_object":
        return {"type": "json_object"}
    if kind == "json_schema":
        schema = fmt.get("schema")
        if not isinstance(schema, dict):
            raise _invalid("text.format.schema", "text.format.schema must be a JSON schema object")
        return {
            "type": "json_schema",
            "json_schema": {
                "name": fmt.get("name") or "response",
                "schema": schema,
                "strict": bool(fmt.get("strict", False)),
            },
        }
    raise _invalid("text.format.type", f"text.format.type {kind!r} is not supported")


def _reasoning_effort(reasoning: dict | None) -> str | None:
    if not reasoning:
        return None
    effort = reasoning.get("effort")
    try:
        return normalize_reasoning_effort(effort)
    except ValueError:
        raise _invalid(
            "reasoning.effort", f"reasoning.effort {effort!r} is not supported"
        ) from None


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
