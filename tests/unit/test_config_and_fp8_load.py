"""Phase 4 parity fixes: rope-scaling validation (M3) + FP8 scale-shape load (M4)."""

import pytest
import torch

from kairyu.models.config import parse_model_config
from kairyu.quant.linear import Fp8Linear


def _llama_config(**extra) -> dict:
    return {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "vocab_size": 256,
        **extra,
    }


def test_unsupported_rope_scaling_raises():
    # M3: a linear/dynamic/longrope rope must fail fast, not silently drop to
    # None and then generate confidently-wrong tokens vs hf.generate.
    with pytest.raises(ValueError, match="unsupported rope_scaling"):
        parse_model_config(
            _llama_config(rope_scaling={"type": "linear", "factor": 4.0})
        )


def test_plain_and_default_rope_are_allowed():
    assert parse_model_config(_llama_config()).rope_scaling is None
    cfg = _llama_config(rope_scaling={"rope_type": "default"})
    assert parse_model_config(cfg).rope_scaling is None


def test_llama3_and_yarn_still_parse():
    llama3 = parse_model_config(
        _llama_config(
            rope_scaling={
                "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                "high_freq_factor": 4.0, "original_max_position_embeddings": 8192,
            }
        )
    )
    assert llama3.rope_scaling.kind == "llama3"


def test_fp8_linear_loads_per_tensor_scale():
    # M4: a static (per-tensor) FP8 checkpoint ships a (1,) weight_scale; it must
    # load, not raise a size mismatch against the default [out,1] buffer.
    layer = Fp8Linear(in_features=8, out_features=4, bias=False)
    state = {
        "weight": torch.zeros(4, 8, dtype=torch.float8_e4m3fn),
        "weight_scale": torch.tensor([0.5]),  # per-tensor (1,)
    }
    missing, unexpected = layer.load_state_dict(state, strict=False, assign=True)
    assert not unexpected
    assert layer.weight_scale.shape == (1,)
    assert layer.dequantize().shape == (4, 8)


def test_fp8_linear_still_loads_per_channel_scale():
    layer = Fp8Linear(in_features=8, out_features=4, bias=False)
    state = {
        "weight": torch.zeros(4, 8, dtype=torch.float8_e4m3fn),
        "weight_scale": torch.ones(4, 1),  # per-channel [out,1]
    }
    layer.load_state_dict(state, strict=False, assign=True)
    assert layer.weight_scale.shape == (4, 1)


def test_deepseek_moe_defaults_match_hf_when_keys_omitted():
    # M2: a trimmed DeepSeek config must fall back to HF defaults, not the
    # Qwen-flavored ones (norm_topk_prob True, routed_scaling 2.5, etc.).
    cfg = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
        "intermediate_size": 128, "vocab_size": 256,
        "num_experts_per_tok": 2, "moe_intermediate_size": 32, "n_routed_experts": 8,
        "kv_lora_rank": 16, "qk_nope_head_dim": 8, "qk_rope_head_dim": 8, "v_head_dim": 8,
    }
    moe = parse_model_config(cfg).moe
    assert moe.norm_topk_prob is True
    assert moe.routed_scaling_factor == 2.5
    assert moe.first_k_dense_replace == 3
    assert moe.n_group == 8 and moe.topk_group == 4


def test_group_size_minus_one_is_a_single_group():
    # M3-models: AutoGPTQ/AWQ group_size=-1 (one whole-input group) must not
    # crash the buffer allocation with a negative count.
    from kairyu.quant.linear import AwqLinear, GptqLinear

    awq = AwqLinear(in_features=32, out_features=16, bias=False, group_size=-1)
    assert awq.group_size == 32 and awq.dequantize().shape == (16, 32)
    gptq = GptqLinear(in_features=32, out_features=16, bias=False, group_size=-1)
    assert gptq.group_size == 32 and gptq.dequantize().shape == (16, 32)


def test_tp_view_rejects_moe_fast():
    # M4-models: TP over a MoE config must fail fast (no dense down_proj to
    # row-parallelize), not load every shard and then AttributeError.
    from kairyu.models.parallel import tp_view

    cfg = parse_model_config({
        "architectures": ["Qwen3MoeForCausalLM"],
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 4, "intermediate_size": 128, "vocab_size": 256,
        "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
    })
    with pytest.raises(ValueError, match="MoE"):
        tp_view(cfg, tp=2, rank=0)


def test_block_fp8_is_rejected_loudly():
    # M5-models: block-wise FP8 (weight_block_size) must not silently route to
    # the per-channel FP8 path.
    from kairyu.engine.core.quant_config import detect_quantization

    with pytest.raises(ValueError, match="block-wise FP8"):
        detect_quantization(
            {"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}
        )
