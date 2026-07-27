"""Shared synthetic full-model checkpoints for quantization tests."""

from __future__ import annotations

import json

import torch

from kairyu.quant import awq, fp8, gptq, int8, nvfp4

TINY = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=128,
    vocab_size=256,
    max_position_embeddings=512,
)

QUANT_CONFIGS = {
    "fp8": {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "weights": {"type": "float", "num_bits": 8, "strategy": "channel"},
                "input_activations": {"type": "float", "num_bits": 8, "dynamic": True},
            }
        },
        "ignore": ["lm_head"],
    },
    "int8": {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "weights": {"type": "int", "num_bits": 8, "strategy": "channel"},
                "input_activations": {"type": "int", "num_bits": 8, "dynamic": True},
            }
        },
        "ignore": ["lm_head"],
    },
    "awq": {
        "quant_method": "awq",
        "bits": 4,
        "group_size": 16,
        "version": "gemm",
        "zero_point": True,
    },
    "gptq": {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 16,
        "desc_act": False,
        "sym": True,
    },
    "nvfp4": {"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16},
}

_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _quantize_projection(
    scheme: str, weight: torch.Tensor
) -> dict[str, torch.Tensor]:
    if scheme == "fp8":
        qweight, scale = fp8.quantize_fp8(weight)
        return {"weight": qweight, "weight_scale": scale}
    if scheme == "int8":
        qweight, scale = int8.quantize_int8_weight(weight)
        return {"weight": qweight, "weight_scale": scale}
    if scheme == "awq":
        qweight, qzeros, scales = awq.quantize_awq(weight, group_size=16)
        return {"qweight": qweight, "qzeros": qzeros, "scales": scales}
    if scheme == "gptq":
        qweight, qzeros, scales, g_idx = gptq.quantize_gptq(
            weight, group_size=16
        )
        return {
            "qweight": qweight,
            "qzeros": qzeros,
            "scales": scales,
            "g_idx": g_idx,
        }
    qweight, scale, global_scale = nvfp4.quantize_nvfp4(weight)
    return {
        "weight": qweight,
        "weight_scale": scale,
        "weight_scale_2": global_scale,
        # Static ModelOpt activation multiplier. Random tiny-model activations
        # are RMS-normalized around one, so calibrate for an amax near 4.
        "input_scale": torch.tensor(4.0 / (6.0 * 448.0)),
    }


def write_quantized_checkpoint(
    scheme: str, source_path, hf_model, target
) -> None:
    from safetensors.torch import save_file

    quantized: dict[str, torch.Tensor] = {}
    for name, tensor in hf_model.state_dict().items():
        if any(f".{projection}.weight" in name for projection in _PROJECTIONS):
            prefix = name.rsplit(".", 1)[0]
            for suffix, packed in _quantize_projection(scheme, tensor).items():
                quantized[f"{prefix}.{suffix}"] = packed.contiguous()
        else:
            quantized[name] = tensor.contiguous()
    save_file(quantized, target / "model.safetensors")
    config = json.loads((source_path / "config.json").read_text())
    config["quantization_config"] = QUANT_CONFIGS[scheme]
    (target / "config.json").write_text(json.dumps(config))
