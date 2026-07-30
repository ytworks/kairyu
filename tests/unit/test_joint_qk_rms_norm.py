"""Joint Q/K RMSNorm routing and safe-fallback contracts."""

from __future__ import annotations

import torch

import kairyu.kernels.rms_norm_gpu as rms_norm_gpu
from kairyu.kernels.rms_norm_gpu import try_joint_qk_rms_norm
from kairyu.models.config import parse_model_config
from kairyu.models.llama import DenseDecoder

_TINY_QWEN = {
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "intermediate_size": 128,
    "vocab_size": 256,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
}


def _packed_qk_views(
    batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed_qkv = torch.randn(batch, 96)
    query = packed_qkv[:, :64].view(batch, 4, 16)
    keys = packed_qkv[:, 64:80].view(batch, 1, 16)
    assert query.stride() == (96, 16, 1)
    assert keys.stride() == (96, 16, 1)
    return query, keys


def test_non_cuda_inputs_decline_joint_kernel_and_keep_exact_model_math() -> None:
    torch.manual_seed(156)
    model = DenseDecoder(parse_model_config(_TINY_QWEN))
    attention = model.model.layers[0].self_attn
    query, keys = _packed_qk_views(batch=3)

    assert try_joint_qk_rms_norm(
        query,
        keys,
        attention.q_norm.weight,
        attention.k_norm.weight,
        attention.q_norm.eps,
    ) is None

    actual_query, actual_keys = attention._normalize_qk(query, keys)
    expected_query = attention.q_norm(query)
    expected_keys = attention.k_norm(keys)
    torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(actual_keys, expected_keys, rtol=0, atol=0)


def test_exact_norm_modules_use_supported_joint_result(monkeypatch) -> None:
    torch.manual_seed(157)
    model = DenseDecoder(parse_model_config(_TINY_QWEN))
    attention = model.model.layers[0].self_attn
    query, keys = _packed_qk_views(batch=3)
    expected = (torch.randn_like(query), torch.randn_like(keys))
    calls: list[tuple] = []

    def fake_joint(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(rms_norm_gpu, "try_joint_qk_rms_norm", fake_joint)
    actual = attention._normalize_qk(query, keys)

    assert actual is expected
    assert len(calls) == 1
    called_query, called_keys, query_weight, key_weight, eps = calls[0]
    assert called_query is query
    assert called_keys is keys
    assert query_weight is attention.q_norm.weight
    assert key_weight is attention.k_norm.weight
    assert eps == attention.q_norm.eps


def test_norm_hooks_retain_separate_module_semantics() -> None:
    torch.manual_seed(158)
    model = DenseDecoder(parse_model_config(_TINY_QWEN))
    attention = model.model.layers[0].self_attn
    query, keys = _packed_qk_views(batch=3)
    calls: list[str] = []
    handle = attention.q_norm.register_forward_hook(
        lambda _module, _inputs, _output: calls.append("query")
    )
    try:
        actual_query, actual_keys = attention._normalize_qk(query, keys)
    finally:
        handle.remove()

    assert calls == ["query"]
    expected_query = attention.q_norm(query)
    expected_keys = attention.k_norm(keys)
    torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(actual_keys, expected_keys, rtol=0, atol=0)


def test_instance_forward_override_is_not_bypassed() -> None:
    torch.manual_seed(159)
    model = DenseDecoder(parse_model_config(_TINY_QWEN))
    attention = model.model.layers[0].self_attn
    query, keys = _packed_qk_views(batch=3)
    attention.q_norm.forward = lambda hidden: torch.zeros_like(hidden)

    actual_query, actual_keys = attention._normalize_qk(query, keys)

    torch.testing.assert_close(
        actual_query,
        torch.zeros_like(query),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_keys,
        attention.k_norm(keys),
        rtol=0,
        atol=0,
    )
