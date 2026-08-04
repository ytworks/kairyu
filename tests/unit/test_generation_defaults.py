import json

import pytest

from kairyu.models.generation import GenerationDefaults, parse_generation_defaults
from kairyu.sampling_params import (
    GENERATION_CONFIG_SAMPLING_FIELDS,
    SamplingParams,
)


def _write_generation_config(path, payload) -> None:
    (path / "generation_config.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_auto_parses_qwen_sampling_defaults_and_applies_only_omissions(tmp_path):
    _write_generation_config(
        tmp_path,
        {
            "eos_token_id": [151645, 151643],
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0,
            "repetition_penalty": 1.05,
        },
    )
    defaults = parse_generation_defaults(
        tmp_path,
        {"eos_token_id": 2},
        "auto",
    )

    params = SamplingParams(
        temperature=1.0,
        top_p=0.8,
    ).with_generation_config_omitted(
        GENERATION_CONFIG_SAMPLING_FIELDS - {"top_p"}
    )
    resolved = defaults.apply(params)

    assert defaults.eos_token_id == 151645
    assert defaults.stop_token_ids == (151643,)
    assert defaults.mode == "auto"
    assert defaults.source == "generation_config.json"
    assert defaults.sampling_defaults() == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.05,
    }
    assert resolved.temperature == 0.6
    assert resolved.top_p == 0.8  # explicit request value wins
    assert resolved.top_k == 20
    assert resolved.min_p == 0.0
    assert resolved.repetition_penalty == 1.05
    assert resolved.generation_config_omitted == frozenset()


def test_vllm_keeps_model_stop_metadata_but_uses_neutral_sampling(tmp_path):
    _write_generation_config(
        tmp_path,
        {
            "eos_token_id": [7, 9],
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
    )

    defaults = parse_generation_defaults(
        tmp_path,
        {"eos_token_id": 2},
        "vllm",
    )

    assert defaults.eos_token_id == 7
    assert defaults.stop_token_ids == (9,)
    assert defaults.source == "vllm"
    assert defaults.sampling_defaults() == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }


def test_none_ignores_generation_file_including_malformed_json(tmp_path):
    (tmp_path / "generation_config.json").write_text("{bad", encoding="utf-8")

    defaults = parse_generation_defaults(
        tmp_path,
        {"eos_token_id": 2},
        "none",
    )

    assert defaults.eos_token_id == 2
    assert defaults.stop_token_ids == ()
    assert defaults.mode == "none"
    assert defaults.source == "disabled"


def test_hf_zero_top_k_is_normalized_to_kairyu_disabled_value(tmp_path):
    _write_generation_config(tmp_path, {"top_k": 0})

    defaults = parse_generation_defaults(tmp_path, {}, "auto")

    assert defaults.top_k == -1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", -0.1),
        ("temperature", True),
        ("temperature", 10**1000),
        ("top_p", 0.0),
        ("top_p", float("nan")),
        ("top_k", 1.5),
        ("top_k", -2),
        ("min_p", 1.1),
        ("repetition_penalty", 0.0),
    ],
)
def test_invalid_generation_sampling_default_fails_closed(tmp_path, field, value):
    _write_generation_config(tmp_path, {field: value})

    with pytest.raises(ValueError, match=field):
        parse_generation_defaults(tmp_path, {}, "auto")


def test_invalid_generation_config_mode_fails_without_reading_files(tmp_path):
    with pytest.raises(ValueError, match="generation_config"):
        parse_generation_defaults(tmp_path, {}, "model")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": True},
        {"top_p": float("inf")},
        {"top_k": True},
        {"source": ""},
        {"mode": "model"},
    ],
)
def test_generation_defaults_enforce_the_in_memory_invariant(kwargs):
    with pytest.raises(ValueError):
        GenerationDefaults(**kwargs)
