import pytest

from kairyu.engine.core.quant_config import QuantConfig, QuantMethod, detect_quantization


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
