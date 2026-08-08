import pytest

from kairyu import SamplingParams
from kairyu.engine.backend import GenerationRequest, native_sampling_params
from kairyu.engine.engine_loop import engine_sampling_from


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
    ("protocol", "begin", "style"),
    [
        (
            "generic",
            '<tool_call>{"name":"strict_tool","arguments":',
            "json",
        ),
        (
            "llama",
            '<|python_tag|>{"name": "strict_tool", "parameters": ',
            "json",
        ),
        (
            "qwen",
            "<tool_call>\n<function=strict_tool>\n",
            "qwen_xml",
        ),
    ],
)
def test_native_strict_tool_builds_parser_matched_structural_tag(
    protocol,
    begin,
    style,
):
    request = GenerationRequest(
        "strict",
        "prompt",
        SamplingParams(),
        tools=(_tool("strict_tool", strict=True),),
        tool_choice="required",
        parallel_tool_calls=False,
        tool_call_protocol=protocol,
    )

    params = native_sampling_params(request)
    response_format = params.extra_args["response_format"]
    grammar_format = response_format["format"]
    tag = grammar_format["tags"][0]

    assert response_format["type"] == "structural_tag"
    assert grammar_format["at_least_one"] is True
    assert grammar_format["stop_after_first"] is True
    assert tag["begin"] == begin
    assert tag["content"]["style"] == style
    assert tag["content"]["json_schema"] == _tool(
        "strict_tool", strict=True
    )["function"]["parameters"]
    assert engine_sampling_from(params).needs_grammar is True


def test_non_strict_tool_arguments_remain_unconstrained_in_mixed_request():
    request = GenerationRequest(
        "mixed",
        "prompt",
        SamplingParams(),
        tools=(
            _tool("strict_tool", strict=True),
            _tool("ordinary_tool", strict=False),
        ),
    )

    grammar_format = native_sampling_params(request).extra_args[
        "response_format"
    ]["format"]

    assert grammar_format["tags"][0]["content"]["json_schema"] is not True
    assert grammar_format["tags"][1]["content"]["json_schema"] is True


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
