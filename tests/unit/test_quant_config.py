import json

import pytest

from kairyu.engine.core.quant_config import (
    QuantConfig,
    QuantMethod,
    detect_quantization,
    load_checkpoint_quantization,
)


def test_no_quantization_config_means_none():
    config = detect_quantization({"model_type": "llama"})
    assert config == QuantConfig(method=QuantMethod.NONE)


def test_fp8_via_compressed_tensors():
    hf_config = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 8, "type": "float"},
                    "input_activations": {"num_bits": 8, "type": "float"},
                }
            },
        }
    }
    config = detect_quantization(hf_config)
    assert config.method is QuantMethod.FP8
    assert config.weight_bits == 8
    assert config.activation_bits == 8
    assert config.activation_dynamic is False


def test_fp8_direct_method():
    config = detect_quantization({"quantization_config": {"quant_method": "fp8"}})
    assert config.method is QuantMethod.FP8
    assert config.activation_dynamic is True


def test_fp8_static_activation_scheme_is_preserved():
    config = detect_quantization(
        {
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": "static",
            }
        }
    )
    assert config.activation_dynamic is False


def test_fp8_unknown_activation_scheme_fails_loudly():
    with pytest.raises(ValueError, match="activation_scheme"):
        detect_quantization(
            {
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "block",
                }
            }
        )


def test_modelopt_fp8_unknown_activation_scheme_fails_loudly():
    with pytest.raises(ValueError, match="modelopt FP8 activation_scheme"):
        detect_quantization(
            {
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "FP8",
                    "activation_scheme": "block",
                }
            }
        )


@pytest.mark.parametrize("group_size", [8, 32, "16", 16.0, True])
def test_modelopt_nvfp4_rejects_non_group16_layout(group_size):
    with pytest.raises(ValueError, match="group_size"):
        detect_quantization(
            {
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "NVFP4",
                    "group_size": group_size,
                }
            }
        )


def test_awq_with_group_size():
    hf_config = {"quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128}}
    config = detect_quantization(hf_config)
    assert config.method is QuantMethod.AWQ
    assert config.weight_bits == 4
    assert config.group_size == 128


def test_gptq():
    hf_config = {"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128}}
    config = detect_quantization(hf_config)
    assert config.method is QuantMethod.GPTQ
    assert config.weight_bits == 4


def test_checkpoint_ignore_metadata_is_preserved():
    config = detect_quantization(
        {
            "quantization_config": {
                "quant_method": "fp8",
                "ignore": ["lm_head", r"re:model\.layers\.0\..*"],
            }
        }
    )
    assert config.ignored_layers == (
        "lm_head",
        r"re:model\.layers\.0\..*",
    )


@pytest.mark.parametrize("ignore", ["lm_head", [""]])
def test_malformed_checkpoint_ignore_metadata_fails(ignore):
    with pytest.raises(ValueError, match="ignore"):
        detect_quantization(
            {
                "quantization_config": {
                    "quant_method": "fp8",
                    "ignore": ignore,
                }
            }
        )


def test_unsupported_method_raises_with_supported_list():
    with pytest.raises(ValueError, match="awq"):
        detect_quantization({"quantization_config": {"quant_method": "bitsandbytes"}})


def _write_external_modelopt(path, **quantization):
    (path / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {"name": "modelopt", "version": "0.33.0"},
                "quantization": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                    "exclude_modules": ["model.layers.0.mlp.gate", "lm_head"],
                    **quantization,
                },
            }
        )
    )


def test_external_modelopt_metadata_is_not_mistaken_for_dense(tmp_path):
    _write_external_modelopt(tmp_path, kv_cache_quant_algo="FP8")

    resolved = load_checkpoint_quantization(tmp_path, {"model_type": "qwen3_moe"})

    assert resolved.weights == QuantConfig(
        method=QuantMethod.NVFP4,
        weight_bits=4,
        group_size=16,
        ignored_layers=("model.layers.0.mlp.gate", "lm_head"),
    )
    assert resolved.source == "hf_quant_config.json"
    assert resolved.kv_cache_quant_algo == "FP8"
    assert resolved.producer_name == "modelopt"
    assert resolved.producer_version == "0.33.0"


@pytest.mark.parametrize(
    "contents",
    [
        '{"producer":{"name":"modelopt","name":"other",'
        '"version":"0.33.0"},"quantization":{}}',
        '{"producer":{"name":"modelopt","version":"0.33.0"},'
        '"quantization":{"quant_algo":"NVFP4","quant_algo":"FP8"}}',
    ],
)
def test_external_modelopt_metadata_rejects_duplicate_json_keys(
    tmp_path,
    contents,
):
    (tmp_path / "hf_quant_config.json").write_text(contents)

    with pytest.raises(ValueError, match="invalid external quantization metadata"):
        load_checkpoint_quantization(tmp_path, {})


@pytest.mark.parametrize(
    "producer",
    [
        {"name": " ", "version": "0.33.0"},
        {"name": "modelopt", "version": " \t"},
        {"name": "modelopt", "version": " 0.33.0"},
        {"name": "modelopt", "version": "0.33.0 "},
    ],
)
def test_external_modelopt_metadata_rejects_whitespace_producer_fields(
    tmp_path,
    producer,
):
    _write_external_modelopt(tmp_path)
    payload = json.loads((tmp_path / "hf_quant_config.json").read_text())
    payload["producer"] = producer
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="producer"):
        load_checkpoint_quantization(tmp_path, {})


def test_external_and_embedded_quantization_must_agree(tmp_path):
    _write_external_modelopt(tmp_path)
    embedded = {
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "FP8",
        }
    }

    with pytest.raises(ValueError, match="disagree"):
        load_checkpoint_quantization(tmp_path, embedded)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"producer": {"name": "other", "version": "1"}}, "producer"),
        ({"quantization": {"quant_algo": "NVFP4", "exclude_modules": "lm_head"}}, "exclude"),
        (
            {
                "quantization": {
                    "quant_algo": "NVFP4",
                    "exclude_modules": [],
                    "kv_cache_quant_algo": "INT8",
                }
            },
            "kv_cache_quant_algo",
        ),
    ],
)
def test_external_modelopt_metadata_fails_closed(tmp_path, mutation, message):
    _write_external_modelopt(tmp_path)
    payload = json.loads((tmp_path / "hf_quant_config.json").read_text())
    payload.update(mutation)
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        load_checkpoint_quantization(tmp_path, {})
