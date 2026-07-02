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


def test_fp8_direct_method():
    config = detect_quantization({"quantization_config": {"quant_method": "fp8"}})
    assert config.method is QuantMethod.FP8


def test_awq_with_group_size():
    hf_config = {
        "quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128}
    }
    config = detect_quantization(hf_config)
    assert config.method is QuantMethod.AWQ
    assert config.weight_bits == 4
    assert config.group_size == 128


def test_gptq():
    hf_config = {
        "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128}
    }
    config = detect_quantization(hf_config)
    assert config.method is QuantMethod.GPTQ
    assert config.weight_bits == 4


def test_unsupported_method_raises_with_supported_list():
    with pytest.raises(ValueError, match="awq"):
        detect_quantization({"quantization_config": {"quant_method": "bitsandbytes"}})
