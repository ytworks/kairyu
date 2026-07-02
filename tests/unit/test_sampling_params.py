import pytest

from kairyu import SamplingParams


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


def test_params_are_immutable():
    params = SamplingParams()
    with pytest.raises(AttributeError):
        params.temperature = 0.2
