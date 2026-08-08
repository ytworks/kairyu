import pytest

from kairyu import SamplingParams
from kairyu.engine.engine_loop import engine_sampling_from
from kairyu.sampling_params import (
    GENERATION_CONFIG_SAMPLING_FIELDS,
    PROMPT_OWNED_EXTRA_ARGS,
)


def test_defaults_match_vllm():
    params = SamplingParams()
    assert params.n == 1
    assert params.temperature == 1.0
    assert params.top_p == 1.0
    assert params.top_k == -1
    assert params.min_p == 0.0
    assert params.max_tokens == 16
    assert params.min_tokens == 0
    assert params.presence_penalty == 0.0
    assert params.frequency_penalty == 0.0
    assert params.repetition_penalty == 1.0
    assert params.seed is None
    assert params.stop == ()
    assert params.stop_token_ids == ()
    assert params.forced_token_ids is None
    assert params.logprobs is None
    assert params.ignore_eos is False
    assert params.skip_special_tokens is True


def test_vllm_style_kwargs():
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256, n=2, seed=42)
    assert params.temperature == 0.8
    assert params.top_p == 0.95
    assert params.max_tokens == 256
    assert params.n == 2
    assert params.seed == 42


def test_stop_string_normalized_to_tuple():
    assert SamplingParams(stop="END").stop == ("END",)
    assert SamplingParams(stop=["a", "b"]).stop == ("a", "b")
    assert SamplingParams(stop=None).stop == ()


def test_forced_token_ids_are_normalized_and_mapped_to_engine_sampling():
    params = SamplingParams(
        max_tokens=3,
        forced_token_ids=[7, 3, 11],
        ignore_eos=True,
    )

    assert params.forced_token_ids == (7, 3, 11)
    assert engine_sampling_from(params).forced_token_ids == (7, 3, 11)


@pytest.mark.parametrize(
    "response_format",
    [
        pytest.param({"type": "json_object"}, id="json-object"),
        pytest.param(
            {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}},
            },
            id="json-schema",
        ),
    ],
)
def test_forced_token_ids_reject_structured_output_intent(response_format):
    with pytest.raises(ValueError, match="structured-output.*response_format"):
        SamplingParams(
            max_tokens=1,
            forced_token_ids=(7,),
            ignore_eos=True,
            extra_args={"response_format": response_format},
        )


def test_forced_token_ids_leave_unrecognized_extra_args_unchanged():
    params = SamplingParams(
        max_tokens=1,
        forced_token_ids=(7,),
        ignore_eos=True,
        extra_args={"vendor_cache": {"mode": "ephemeral"}},
    )

    sampling = engine_sampling_from(params)
    assert params.extra_args == {"vendor_cache": {"mode": "ephemeral"}}
    assert sampling.forced_token_ids == (7,)
    assert sampling.needs_grammar is False


@pytest.mark.parametrize(
    ("response_format", "field", "value"),
    [
        ({"type": "regex", "pattern": "a+"}, "regex", "a+"),
        (
            {"type": "grammar", "grammar": 'root ::= "a"'},
            "grammar",
            'root ::= "a"',
        ),
        (
            {
                "type": "structural_tag",
                "format": {"type": "const_string", "value": "a"},
            },
            "structural_tag",
            {
                "type": "structural_tag",
                "format": {"type": "const_string", "value": "a"},
            },
        ),
    ],
)
def test_extended_structured_formats_map_to_engine_sampling(
    response_format,
    field,
    value,
):
    sampling = engine_sampling_from(
        SamplingParams(extra_args={"response_format": response_format})
    )

    assert getattr(sampling, field) == value
    assert sampling.needs_grammar is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -0.1},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"n": 0},
        {"max_tokens": 0},
        {"min_p": -0.5},
        {"repetition_penalty": 0.0},
        {"min_tokens": -1},
        {"max_tokens": 1, "min_tokens": 2},
        {"forced_token_ids": []},
        {"forced_token_ids": [-1]},
        {"forced_token_ids": [True]},
        {"forced_token_ids": ["7"]},
        {"forced_token_ids": 7},
        {"max_tokens": 1, "forced_token_ids": [7, 3]},
        {"max_tokens": None, "forced_token_ids": [7]},
        {"max_tokens": 1, "forced_token_ids": [7]},
        {"max_tokens": 1, "forced_token_ids": [7], "stop": "END", "ignore_eos": True},
        {
            "max_tokens": 1,
            "forced_token_ids": [7],
            "stop_token_ids": [9],
            "ignore_eos": True,
        },
        {
            "max_tokens": 1,
            "forced_token_ids": [7],
            "min_tokens": 1,
            "ignore_eos": True,
        },
    ],
)
def test_invalid_values_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_error_message_names_field():
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(temperature=-1.0)


def test_clone_returns_new_object_and_leaves_original_unchanged():
    original = SamplingParams(temperature=0.5)
    changed = original.clone(temperature=0.9, max_tokens=64)
    assert changed is not original
    assert changed.temperature == 0.9
    assert changed.max_tokens == 64
    assert original.temperature == 0.5
    assert original.max_tokens == 16


def test_generation_config_omissions_survive_clone_until_explicit_override():
    original = SamplingParams().with_generation_config_omitted(
        GENERATION_CONFIG_SAMPLING_FIELDS
    )

    cloned = original.clone(max_tokens=64)
    explicit = cloned.clone(temperature=1.0)

    assert cloned.generation_config_omitted == GENERATION_CONFIG_SAMPLING_FIELDS
    assert explicit.generation_config_omitted == (
        GENERATION_CONFIG_SAMPLING_FIELDS - {"temperature"}
    )


def test_unknown_generation_config_omission_is_rejected():
    with pytest.raises(ValueError, match="unknown generation-config"):
        SamplingParams().with_generation_config_omitted({"max_tokens"})


def test_params_are_immutable():
    params = SamplingParams()
    with pytest.raises(AttributeError):
        params.temperature = 0.2


@pytest.mark.parametrize("prompt_owned", sorted(PROMPT_OWNED_EXTRA_ARGS))
def test_prompt_owned_fields_cannot_enter_through_extra_args(prompt_owned):
    with pytest.raises(ValueError, match="prompt-owned fields"):
        SamplingParams(extra_args={prompt_owned: "opaque"})


def test_non_prompt_sampling_extensions_remain_supported():
    params = SamplingParams(
        extra_args={
            "response_format": {"type": "json_object"},
            "vendor_cache": {"mode": "ephemeral"},
        }
    )

    assert params.extra_args["response_format"] == {"type": "json_object"}
    assert params.extra_args["vendor_cache"] == {"mode": "ephemeral"}
    assert engine_sampling_from(params).json_mode is True


def test_legacy_parallel_tool_calls_extra_arg_remains_supported():
    params = SamplingParams(extra_args={"parallel_tool_calls": False})

    assert params.extra_args["parallel_tool_calls"] is False


def test_extra_args_are_defensively_copied_and_top_level_immutable():
    source = {"vendor_cache": {"mode": "ephemeral"}}
    params = SamplingParams(extra_args=source)
    source["prompt_token_ids"] = [1, 2, 3]

    assert "prompt_token_ids" not in params.extra_args
    with pytest.raises(TypeError):
        params.extra_args["prompt_token_ids"] = [1, 2, 3]
