"""Responses tool declarations mapped onto Chat Completions function tools.

Namespace tools flatten to ``<namespace>__<name>`` functions; hosted
``web_search`` has no executor, so it is accepted, echoed, and never offered to
the model.
"""

from __future__ import annotations

from kairyu.entrypoints.server.chat_service import ChatRequestError

_NAMESPACE_SEPARATOR = "__"


def namespaced_name(namespace: str, name: str) -> str:
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


def chat_tools(tools: list[dict] | None) -> list[dict] | None:
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
                encoded = namespaced_name(namespace, nested_name)
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
        if kind == "web_search":
            # No server-side executor: accept and drop. Codex declares hosted
            # web search on every turn (live/indexed under full-access
            # sandboxes); the declaration is echoed but never offered to the
            # model, so it can be neither selected nor executed.
            unknown = set(tool) - {
                "type",
                "external_web_access",
                "indexed_web_access",
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
            for flag in ("external_web_access", "indexed_web_access"):
                if tool.get(flag) is not None and not isinstance(tool[flag], bool):
                    raise ChatRequestError(f"tools[{index}].{flag} must be a boolean")
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
    # Templates gate on `tools is not none`: dropped hosted tools leave none.
    return converted or None


def chat_tool_choice(
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
        selected = namespaced_name(namespace, name)
        known_names = namespace_names(tools)
        if selected not in known_names:
            raise ChatRequestError(
                f"named tool_choice references unknown namespace function {namespace!r}.{name!r}"
            )
        name = selected
    return {"type": "function", "function": {"name": name}}


def namespace_names(tools: list[dict] | None) -> dict[str, tuple[str, str]]:
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
                names[namespaced_name(namespace, name)] = (namespace, name)
    return names
