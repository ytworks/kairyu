import pytest

from kairyu import SamplingParams
from kairyu.engine.backend import GenerationRequest, native_sampling_params
from kairyu.engine.core.structured import XGrammarEnforcer
from kairyu.engine.engine_loop import engine_sampling_from
from kairyu.entrypoints.chat_template import ToolCallProtocol
from kairyu.entrypoints.server.chat_service import _parse_tool_calls

pytest.importorskip("xgrammar")

_VOCAB = [chr(codepoint) for codepoint in range(32, 127)] + ["\n", "｜", "<eos>"]


def _tool(name: str, *, strict: bool) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "strict": strict,
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }


@pytest.mark.parametrize(
    ("protocol", "begin", "style", "stop_after_first"),
    [
        (
            "generic",
            '<tool_call>{"name":"strict_tool","arguments":',
            "json",
            False,
        ),
        (
            "llama",
            '<|python_tag|>{"name": "strict_tool", "parameters": ',
            "json",
            True,
        ),
        (
            "qwen",
            "<tool_call>\n<function=strict_tool>\n",
            "qwen_xml",
            False,
        ),
    ],
)
def test_native_strict_tool_builds_parser_matched_structural_tag(
    protocol,
    begin,
    style,
    stop_after_first,
):
    request = GenerationRequest(
        "strict",
        "prompt",
        SamplingParams(),
        tools=(_tool("strict_tool", strict=True),),
        tool_choice="required",
        tool_call_protocol=protocol,
    )

    params = native_sampling_params(request)
    response_format = params.extra_args["response_format"]
    grammar_format = response_format["format"]
    tag = grammar_format["tags"][0]

    assert response_format["type"] == "structural_tag"
    assert grammar_format["at_least_one"] is True
    assert grammar_format["stop_after_first"] is stop_after_first
    assert tag["begin"] == begin
    assert tag["content"]["style"] == style
    assert tag["content"]["json_schema"] == _tool(
        "strict_tool", strict=True
    )["function"]["parameters"]
    assert engine_sampling_from(params).needs_grammar is True
    XGrammarEnforcer(
        _VOCAB,
        structural_tag=response_format,
        stop_token_id=len(_VOCAB) - 1,
    )


def test_deepseek_v4_strict_tool_builds_nested_dsml_structural_tag():
    request = GenerationRequest(
        "strict-deepseek",
        "prompt",
        SamplingParams(),
        tools=(_tool("strict_tool", strict=True),),
        tool_choice="required",
        tool_call_protocol="deepseek_v4",
    )

    response_format = native_sampling_params(request).extra_args[
        "response_format"
    ]
    grammar_format = response_format["format"]
    outer = grammar_format["tags"][0]
    invoke = outer["content"]["tags"][0]

    assert grammar_format["triggers"] == ["<｜DSML｜tool_calls>"]
    assert outer["begin"] == "<｜DSML｜tool_calls>\n"
    assert outer["end"] == "</｜DSML｜tool_calls>"
    assert invoke["begin"] == '<｜DSML｜invoke name="strict_tool">\n'
    assert invoke["content"]["style"] == "deepseek_xml"
    XGrammarEnforcer(
        _VOCAB,
        structural_tag=response_format,
        stop_token_id=len(_VOCAB) - 1,
    )


@pytest.mark.parametrize("protocol", ["generic", "llama", "qwen"])
def test_non_strict_tool_arguments_remain_unconstrained_in_mixed_request(
    protocol,
):
    request = GenerationRequest(
        "mixed",
        "prompt",
        SamplingParams(),
        tools=(
            _tool("strict_tool", strict=True),
            _tool("ordinary_tool", strict=False),
        ),
        tool_call_protocol=protocol,
    )

    response_format = native_sampling_params(request).extra_args[
        "response_format"
    ]
    grammar_format = response_format["format"]

    assert grammar_format["tags"][0]["content"]["json_schema"] is not True
    assert grammar_format["tags"][1]["content"]["json_schema"] is True
    XGrammarEnforcer(
        _VOCAB,
        structural_tag=response_format,
        stop_token_id=len(_VOCAB) - 1,
    )


@pytest.mark.parametrize(
    ("protocol", "output", "parser_protocol"),
    [
        (
            "generic",
            '<tool_call>{"name":"strict_tool","arguments":{"value":1}}'
            "</tool_call>",
            ToolCallProtocol.GENERIC,
        ),
        (
            "llama",
            '<|python_tag|>{"name": "strict_tool", "parameters": '
            '{"value":1}}',
            ToolCallProtocol.LLAMA,
        ),
        (
            "qwen",
            "<tool_call>\n<function=strict_tool>\n<parameter=value>\n1\n"
            "</parameter>\n</function>\n</tool_call>",
            ToolCallProtocol.QWEN,
        ),
        (
            "deepseek_v4",
            '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="strict_tool">\n'
            '<｜DSML｜parameter name="value" string="false">1'
            '</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
            ToolCallProtocol.DEEPSEEK_V4,
        ),
    ],
)
def test_strict_tool_grammar_output_round_trips_through_attested_parser(
    protocol,
    output,
    parser_protocol,
):
    tool = _tool("strict_tool", strict=True)
    request = GenerationRequest(
        "roundtrip",
        "prompt",
        SamplingParams(),
        tools=(tool,),
        tool_choice="required",
        tool_call_protocol=protocol,
    )
    response_format = native_sampling_params(request).extra_args[
        "response_format"
    ]
    enforcer = XGrammarEnforcer(
        _VOCAB,
        structural_tag=response_format,
        stop_token_id=len(_VOCAB) - 1,
    )

    for character in output:
        assert enforcer.accept(_VOCAB.index(character))
    assert enforcer.accept(len(_VOCAB) - 1)
    assert enforcer.is_terminated()

    calls = _parse_tool_calls(output, (tool,), parser_protocol)
    assert len(calls) == 1
    assert calls[0].function.name == "strict_tool"


def test_named_non_strict_tool_does_not_activate_unselected_strict_grammar():
    request = GenerationRequest(
        "named",
        "prompt",
        SamplingParams(),
        tools=(
            _tool("strict_tool", strict=True),
            _tool("ordinary_tool", strict=False),
        ),
        tool_choice={
            "type": "function",
            "function": {"name": "ordinary_tool"},
        },
    )

    assert native_sampling_params(request) is request.sampling_params


@pytest.mark.parametrize(
    ("strict", "tool_choice"),
    [(False, None), (True, "none")],
)
def test_non_enforced_legacy_tool_name_does_not_activate_new_validation(
    strict,
    tool_choice,
):
    request = GenerationRequest(
        "legacy-name",
        "prompt",
        SamplingParams(),
        tools=(_tool("legacy.name", strict=strict),),
        tool_choice=tool_choice,
    )

    assert native_sampling_params(request) is request.sampling_params


def test_strict_tools_reject_conflicting_structured_response_format():
    request = GenerationRequest(
        "conflict",
        "prompt",
        SamplingParams(
            extra_args={"response_format": {"type": "json_object"}}
        ),
        tools=(_tool("strict_tool", strict=True),),
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        native_sampling_params(request)


def test_strict_tool_requires_parameter_schema():
    request = GenerationRequest(
        "missing-schema",
        "prompt",
        SamplingParams(),
        tools=(
            {
                "type": "function",
                "function": {"name": "strict_tool", "strict": True},
            },
        ),
    )

    with pytest.raises(ValueError, match="requires a JSON-schema"):
        native_sampling_params(request)
