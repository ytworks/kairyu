"""Fail-closed parsing for server-attested model-native tool protocols."""

from __future__ import annotations

import json

import pytest

from kairyu.entrypoints.chat_template import ChatTemplate, ToolCallProtocol
from kairyu.entrypoints.server.chat_service import (
    NormalizedToolChoice,
    ValidatedChatInput,
    _parse_tool_calls,
)
from kairyu.entrypoints.server.protocol import ChatCompletionRequest
from kairyu.entrypoints.server.tool_stream import (
    StreamInvalid,
    ToolArgsDelta,
    ToolStart,
    ToolStop,
    ToolStreamScanner,
)

_GENERIC_TEMPLATE = "{{ messages[0].content }}"
_LLAMA_TEMPLATE = "{{ '<|python_tag|>' }}{{ messages[0].content }}"
_QWEN_TEMPLATE = (
    "{{ '<function=name><parameter=value></parameter></function>' }}{{ messages[0].content }}"
)
_DEEPSEEK_TEMPLATE = "{{ '<｜DSML｜tool_calls></｜DSML｜tool_calls>' }}"


def _tool(
    properties: dict[str, object] | None = None,
    *,
    required: list[str] | None = None,
    additional: bool | dict[str, object] = True,
) -> tuple[dict[str, object], ...]:
    parameters: dict[str, object] = {
        "type": "object",
        "properties": properties or {},
        "additionalProperties": additional,
    }
    if required is not None:
        parameters["required"] = required
    return (
        {
            "type": "function",
            "function": {"name": "run", "parameters": parameters},
        },
    )


def _qwen(*parameters: tuple[str, str], name: str = "run") -> str:
    body = "".join(
        f"<parameter={parameter}>\n{value}\n</parameter>\n" for parameter, value in parameters
    )
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def _arguments(text: str, tools: tuple[dict[str, object], ...]) -> dict[str, object]:
    calls = _parse_tool_calls(text, tools, ToolCallProtocol.QWEN)
    assert len(calls) == 1
    return json.loads(calls[0].function.arguments)


def test_chat_template_attests_selected_protocol_not_model_name_or_unused_template():
    assert ChatTemplate(_GENERIC_TEMPLATE).tool_call_protocol is ToolCallProtocol.GENERIC
    assert ChatTemplate(_LLAMA_TEMPLATE).tool_call_protocol is ToolCallProtocol.LLAMA
    assert ChatTemplate(_QWEN_TEMPLATE).tool_call_protocol is ToolCallProtocol.QWEN
    assert (
        ChatTemplate(_DEEPSEEK_TEMPLATE).tool_call_protocol
        is ToolCallProtocol.DEEPSEEK_V4
    )

    mixed = ChatTemplate(
        {
            "default": _GENERIC_TEMPLATE,
            "tool_use": _QWEN_TEMPLATE,
            "unused_llama": _LLAMA_TEMPLATE,
        }
    )
    assert mixed.tool_call_protocol_for_tools(has_tools=False) is ToolCallProtocol.GENERIC
    assert mixed.tool_call_protocol_for_tools(has_tools=True) is ToolCallProtocol.QWEN


def test_validated_chat_input_preserves_existing_positional_constructor():
    request = ChatCompletionRequest(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
    )
    normalized = NormalizedToolChoice("auto", frozenset())
    validated = ValidatedChatInput(request, "prompt", normalized, False, False)
    assert validated.tool_call_protocol is ToolCallProtocol.GENERIC
    assert validated.parallel_tool_calls is None


def test_inert_jinja_comments_do_not_attest_a_native_protocol():
    commented = ChatTemplate(
        "{# <|python_tag|> <function=x><parameter=y></parameter></function> #}" + _GENERIC_TEMPLATE
    )
    assert commented.tool_call_protocol is ToolCallProtocol.GENERIC


def test_ambiguous_active_template_does_not_attest_either_native_protocol():
    ambiguous = ChatTemplate(_LLAMA_TEMPLATE + _QWEN_TEMPLATE)
    assert ambiguous.tool_call_protocol is ToolCallProtocol.GENERIC


def test_model_native_syntax_is_bound_to_attested_protocol():
    bare = '{"name":"run","parameters":{"value":1}}'
    qwen = _qwen(("value", "1"))
    tools = _tool({"value": {"type": "integer"}})

    assert _parse_tool_calls(bare, tools, ToolCallProtocol.GENERIC) == []
    assert len(_parse_tool_calls(bare, tools, ToolCallProtocol.LLAMA)) == 1
    assert _parse_tool_calls(bare, tools, ToolCallProtocol.QWEN) == []
    assert _parse_tool_calls(qwen, tools, ToolCallProtocol.GENERIC) == []
    assert _parse_tool_calls(qwen, tools, ToolCallProtocol.LLAMA) == []
    assert len(_parse_tool_calls(qwen, tools, ToolCallProtocol.QWEN)) == 1


def test_deepseek_v4_dsml_preserves_strings_and_decodes_typed_json() -> None:
    text = (
        '<｜DSML｜tool_calls><｜DSML｜invoke name="run">'
        '<｜DSML｜parameter name="path" string="true">資料</｜DSML｜parameter>'
        '<｜DSML｜parameter name="count" string="false">2</｜DSML｜parameter>'
        "</｜DSML｜invoke></｜DSML｜tool_calls>"
    )
    tools = _tool(
        {"path": {"type": "string"}, "count": {"type": "integer"}},
        required=["path", "count"],
    )

    calls = _parse_tool_calls(text, tools, ToolCallProtocol.DEEPSEEK_V4)

    assert len(calls) == 1
    assert json.loads(calls[0].function.arguments) == {"path": "資料", "count": 2}


@pytest.mark.parametrize("protocol", list(ToolCallProtocol))
def test_explicit_json_tool_tag_remains_portable_across_protocols(protocol):
    text = '<tool_call>{"name":"run","arguments":{"value":"ok"}}</tool_call>'
    calls = _parse_tool_calls(text, _tool(), protocol)
    assert len(calls) == 1
    assert json.loads(calls[0].function.arguments) == {"value": "ok"}


def test_generic_tool_arguments_stream_across_every_character_boundary():
    arguments = {
        "text": 'braces stay in strings: } { and quote: "',
        "nested": {"items": [{"value": 1}, {"value": 2}]},
    }
    body = json.dumps(
        {"name": "run", "arguments": arguments},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    scanner = ToolStreamScanner(
        _tool(),
        NormalizedToolChoice("auto", frozenset({"run"})),
    )

    events = []
    for character in f"<tool_call>{body}</tool_call>":
        events.extend(scanner.feed(character))
    events.extend(scanner.feed("", final=True))

    assert sum(isinstance(event, ToolStart) for event in events) == 1
    assert sum(isinstance(event, ToolStop) for event in events) == 1
    assert not any(isinstance(event, StreamInvalid) for event in events)
    partial_json = "".join(
        event.partial_json for event in events if isinstance(event, ToolArgsDelta)
    )
    assert json.loads(partial_json) == arguments


@pytest.mark.parametrize("suffix", ["", "<|eom_id|>", "<|eot_id|>"])
def test_attested_llama_accepts_only_exact_json_with_protocol_delimiters(suffix):
    text = '<|python_tag|>{"name":"run","parameters":{"path":"資料"}}' + suffix
    calls = _parse_tool_calls(text, _tool(), ToolCallProtocol.LLAMA)
    assert len(calls) == 1
    assert json.loads(calls[0].function.arguments) == {"path": "資料"}


@pytest.mark.parametrize(
    "arguments",
    [
        "null",
        "[]",
        '"text"',
        "NaN",
        "Infinity",
        "-Infinity",
        "1e999",
        '{"value":NaN}',
        '{"value":1e999}',
        '{"value":1,"value":2}',
    ],
)
def test_explicit_json_tool_arguments_require_one_finite_json_object(arguments):
    text = f'<tool_call>{{"name":"run","arguments":{json.dumps(arguments)}}}</tool_call>'
    assert _parse_tool_calls(text, _tool(), ToolCallProtocol.GENERIC) == []


def test_explicit_json_tool_arguments_string_round_trips_after_strict_validation():
    arguments = '{"path":"資料","nested":{"ok":true}}'
    text = f'<tool_call>{{"name":"run","arguments":{json.dumps(arguments)}}}</tool_call>'
    calls = _parse_tool_calls(text, _tool(), ToolCallProtocol.GENERIC)
    assert len(calls) == 1
    assert calls[0].function.arguments == arguments


@pytest.mark.parametrize(
    "number",
    ["nan", "NaN", "inf", "+inf", "-inf", "Infinity", "+Infinity", "-Infinity", "1e999"],
)
def test_qwen_rejects_non_finite_numeric_parameters(number):
    tools = _tool({"value": {"type": "number"}}, required=["value"])
    assert _parse_tool_calls(_qwen(("value", number)), tools, ToolCallProtocol.QWEN) == []


@pytest.mark.parametrize(
    ("schema", "value"),
    [
        ({"type": "array"}, "[NaN]"),
        ({"type": "array"}, "[1e999]"),
        ({"type": "object"}, '{"nested":Infinity}'),
        ({"type": "object"}, '{"nested":1e999}'),
    ],
)
def test_qwen_rejects_nested_non_finite_values(schema, value):
    tools = _tool({"value": schema}, required=["value"])
    assert _parse_tool_calls(_qwen(("value", value)), tools, ToolCallProtocol.QWEN) == []


def test_qwen_preserves_string_null_and_converts_only_nullable_null():
    tools = _tool(
        {
            "string_value": {"type": "string"},
            "nullable_value": {"type": ["number", "null"]},
        },
        required=["string_value", "nullable_value"],
    )
    assert _arguments(_qwen(("string_value", "null"), ("nullable_value", "null")), tools) == {
        "string_value": "null",
        "nullable_value": None,
    }


def test_qwen_accepts_valid_typed_scalars_unicode_and_nested_json():
    tools = _tool(
        {
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "enabled": {"type": "boolean"},
            "path": {"type": "string"},
            "items": {"type": "array"},
            "options": {"type": "object"},
        },
        required=["count", "ratio", "enabled", "path", "items", "options"],
        additional=False,
    )
    assert _arguments(
        _qwen(
            ("count", "3"),
            ("ratio", "0.25"),
            ("enabled", "true"),
            ("path", "資料/東京.md"),
            ("items", '["a", "β"]'),
            ("options", '{"force":false}'),
        ),
        tools,
    ) == {
        "count": 3,
        "ratio": 0.25,
        "enabled": True,
        "path": "資料/東京.md",
        "items": ["a", "β"],
        "options": {"force": False},
    }


def test_qwen_unconstrained_additional_parameter_remains_text():
    tools = _tool()
    assert _arguments(_qwen(("freeform", "null")), tools) == {"freeform": "null"}


@pytest.mark.parametrize(
    "text",
    [
        _qwen(("value", "one"), ("value", "two")),
        _qwen(("unknown", "one")),
        _qwen(),
        "junk" + _qwen(("value", "one")),
        _qwen(("value", "one")) + "junk",
        "<tool_call><function=run>junk<parameter=value>one</parameter></function></tool_call>",
        "<tool_call><function=run><parameter=value>one</function></tool_call>",
        (
            "<tool_call><function=run><parameter=value>one</parameter></function>"
            "<function=run></function></tool_call>"
        ),
    ],
)
def test_qwen_rejects_duplicate_unknown_missing_unbalanced_or_residual_text(text):
    tools = _tool(
        {"value": {"type": "string"}},
        required=["value"],
        additional=False,
    )
    assert _parse_tool_calls(text, tools, ToolCallProtocol.QWEN) == []
