"""Real-CUDA gates for packed FP8 EAGLE/MTP draft heads."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from kairyu.engine.core.attention.mla_torch import mla_absorbed, mla_decompress, mla_scale
from kairyu.models.config import parse_model_config
from kairyu.models.draft_quant import (
    MTP_DRAFT_QUANTIZATION_KEY,
    draft_linear_factory,
    pack_dynamic_fp8_draft_state,
    parse_draft_quantization,
)
from kairyu.models.eagle import EagleConfig, EagleDraftHead, load_eagle_head
from kairyu.models.layers import RotaryEmbedding
from kairyu.models.mtp import MtpDraftHead, load_mtp_head
from kairyu.quant.linear import Fp8Linear, LinearKernel, ModelScope

pytestmark = pytest.mark.gpu

_FP8_CONFIG = {
    "quant_method": "compressed-tensors",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "type": "float",
                "num_bits": 8,
                "strategy": "channel",
                "symmetric": True,
            },
            "input_activations": {
                "type": "float",
                "num_bits": 8,
                "dynamic": True,
                "strategy": "token",
                "symmetric": True,
            },
        }
    },
}

_EAGLE = EagleConfig(
    hidden_size=32,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    intermediate_size=64,
    draft_vocab_size=48,
)

_DSV3_RAW = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 64,
    "vocab_size": 128,
    "max_position_embeddings": 256,
    "kv_lora_rank": 8,
    "q_lora_rank": None,
    "qk_nope_head_dim": 4,
    "qk_rope_head_dim": 4,
    "v_head_dim": 6,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "n_group": 2,
    "topk_group": 1,
    "n_shared_experts": 1,
    "first_k_dense_replace": 1,
    "rms_norm_eps": 1e-6,
}


def _mtp_fp8_config() -> dict:
    layer = _DSV3_RAW["num_hidden_layers"]
    return {
        **_FP8_CONFIG,
        "ignore": [
            f"model.layers.{layer}.mlp.gate",
            f"model.layers.{layer}.self_attn.kv_b_proj",
        ],
    }


@pytest.fixture(scope="module")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():  # pragma: no cover - deploy-day only
        pytest.skip("CUDA required")
    return torch.device("cuda:0")


def _eagle_raw() -> dict:
    return {
        "hidden_size": _EAGLE.hidden_size,
        "intermediate_size": _EAGLE.intermediate_size,
        "draft_vocab_size": _EAGLE.draft_vocab_size,
        "num_attention_heads": _EAGLE.num_attention_heads,
        "num_key_value_heads": _EAGLE.num_key_value_heads,
        "head_dim": _EAGLE.head_dim,
        "rope_theta": _EAGLE.rope_theta,
        "rms_norm_eps": _EAGLE.rms_norm_eps,
        "quantization_config": _FP8_CONFIG,
    }


def _mtp_checkpoint_state(config, local: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{config.num_hidden_layers}."
    checkpoint: dict[str, torch.Tensor] = {}
    for name, tensor in local.items():
        if name.startswith("head."):
            target = prefix + "shared_head.head." + name[len("head.") :]
        elif name.startswith("decoder."):
            target = prefix + name[len("decoder.") :]
        else:
            target = prefix + name
        checkpoint[target] = tensor
    return checkpoint


def test_packed_eagle_runs_fused_fp8_forward(cuda_device, tmp_path):
    raw = _eagle_raw()
    metadata = parse_draft_quantization(
        raw,
        expected_scope=ModelScope.EAGLE_DRAFT,
    )
    dense = EagleDraftHead(_EAGLE).eval()
    packed_module = EagleDraftHead(
        _EAGLE,
        linear_factory=draft_linear_factory(metadata),
    ).eval()
    packed = pack_dynamic_fp8_draft_state(dense.state_dict(), packed_module)
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    save_file(packed, tmp_path / "model.safetensors")

    loaded = load_eagle_head(
        tmp_path,
        _EAGLE,
        dtype=torch.bfloat16,
        target_device=cuda_device,
    ).to(cuda_device)
    assert isinstance(loaded.fc, Fp8Linear)
    assert loaded.fc.linear_selection.kernel is LinearKernel.CUDA_FP8_AUTO
    assert loaded.fc.weight.dtype is torch.float8_e4m3fn

    embeds = torch.randn(5, 32, dtype=torch.bfloat16, device=cuda_device)
    aux = torch.randn(5, 96, dtype=torch.bfloat16, device=cuda_device)
    hidden = loaded.midlayer(embeds, loaded.fuse(aux))
    logits = loaded.lm_head(loaded.norm(hidden[-1]))

    assert hidden.shape == (5, 32)
    assert logits.shape == (48,)
    assert hidden.device == logits.device == cuda_device
    assert torch.isfinite(hidden).all()
    assert torch.isfinite(logits).all()


def test_packed_mtp_runs_fused_fp8_forward_with_cuda_local_kv(cuda_device, tmp_path):
    config = parse_model_config(_DSV3_RAW)
    raw = {
        **_DSV3_RAW,
        MTP_DRAFT_QUANTIZATION_KEY: _mtp_fp8_config(),
    }
    metadata = parse_draft_quantization(
        raw,
        expected_scope=ModelScope.MTP_DRAFT,
    )
    dense = MtpDraftHead(config).eval()
    packed_module = MtpDraftHead(
        config,
        linear_factory=draft_linear_factory(metadata),
    ).eval()
    local = pack_dynamic_fp8_draft_state(dense.state_dict(), packed_module)
    checkpoint = _mtp_checkpoint_state(config, local)
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    save_file(checkpoint, tmp_path / "model.safetensors")

    loaded = load_mtp_head(
        tmp_path,
        config,
        dtype=torch.bfloat16,
        target_device=cuda_device,
    ).to(cuda_device)
    assert isinstance(loaded.eh_proj, Fp8Linear)
    assert loaded.eh_proj.linear_selection.kernel is LinearKernel.CUDA_FP8_AUTO
    assert loaded.eh_proj.weight.dtype is torch.float8_e4m3fn

    rotary = RotaryEmbedding(config).to(cuda_device)
    token_ids = torch.arange(5, dtype=torch.long, device=cuda_device)
    target_hidden = torch.randn(
        5,
        config.hidden_size,
        dtype=torch.bfloat16,
        device=cuda_device,
    )
    hidden = loaded.forward_chain(token_ids, target_hidden, rotary)
    logits = loaded.logits(hidden[-1])

    assert hidden.shape == target_hidden.shape
    assert logits.shape == (config.vocab_size,)
    assert hidden.device == logits.device == cuda_device
    assert torch.isfinite(hidden).all()
    assert torch.isfinite(logits).all()


def test_mla_causal_mask_follows_cuda_scores_device(cuda_device):
    torch.manual_seed(23)
    chunk, heads, d_nope, d_rope, rank, seq, d_value = 3, 2, 8, 4, 16, 5, 6
    tensors = {
        "q_nope": torch.randn(chunk, heads, d_nope, device=cuda_device),
        "q_pe": torch.randn(chunk, heads, d_rope, device=cuda_device),
        "c_kv": torch.randn(seq, rank, device=cuda_device),
        "k_pe": torch.randn(seq, d_rope, device=cuda_device),
        "w_uk": torch.randn(heads, rank, d_nope, device=cuda_device),
        "w_uv": torch.randn(heads, rank, d_value, device=cuda_device),
    }
    scale = mla_scale(d_nope, d_rope)

    decompressed = mla_decompress(**tensors, scale=scale, causal_offset=2)
    absorbed = mla_absorbed(**tensors, scale=scale, causal_offset=2)

    assert decompressed.device == absorbed.device == cuda_device
    assert torch.allclose(decompressed, absorbed, atol=2e-5, rtol=2e-5)
