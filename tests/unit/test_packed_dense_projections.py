"""Canonical-state and fallback contracts for dense QKV packing."""

from __future__ import annotations

import torch

from kairyu.engine.core.quant_config import QuantConfig, QuantMethod
from kairyu.models.config import parse_model_config
from kairyu.models.llama import DenseDecoder
from kairyu.models.packed_linear import DenseLinearPack
from kairyu.models.parallel import checkpoint_member_specs, tp_view
from kairyu.quant.linear import linear_factory

_TINY = {
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 128,
    "vocab_size": 256,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}


def _storage_layout(
    tensors: tuple[torch.Tensor, ...],
) -> tuple[int, tuple[int, ...]]:
    return (
        tensors[0].untyped_storage().data_ptr(),
        tuple(tensor.storage_offset() for tensor in tensors),
    )


def test_dense_model_packs_qkv_without_changing_canonical_names() -> None:
    torch.manual_seed(17)
    model = DenseDecoder(parse_model_config(_TINY))
    names_before = tuple(model.state_dict())
    layer = model.model.layers[0]
    attention = layer.self_attn

    qkv = (attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight)
    qkv_storage, qkv_offsets = _storage_layout(qkv)
    assert all(weight.untyped_storage().data_ptr() == qkv_storage for weight in qkv)
    assert qkv_offsets == (
        0,
        attention.q_proj.weight.numel(),
        attention.q_proj.weight.numel() + attention.k_proj.weight.numel(),
    )
    assert qkv[0].untyped_storage().nbytes() == sum(
        weight.numel() * weight.element_size()
        for weight in qkv
    )
    assert attention._qkv_pack.is_current()
    assert not hasattr(layer.mlp, "_gate_up_pack")

    assert tuple(model.state_dict()) == names_before
    assert "model.layers.0.self_attn.qkv_proj.weight" not in names_before
    assert set(name for name, _ in model.named_parameters()) == set(names_before)


def test_packed_projection_math_matches_separate_canonical_linears() -> None:
    torch.manual_seed(31)
    model = DenseDecoder(parse_model_config(_TINY))
    layer = model.model.layers[0]
    hidden = torch.randn(11, _TINY["hidden_size"])

    expected_qkv = (
        layer.self_attn.q_proj(hidden),
        layer.self_attn.k_proj(hidden),
        layer.self_attn.v_proj(hidden),
    )
    actual_qkv = layer.self_attn._project_qkv(hidden)
    for actual, expected in zip(actual_qkv, expected_qkv, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_assign_load_and_dtype_move_rebuild_packs_from_canonical_members() -> None:
    torch.manual_seed(47)
    source = DenseDecoder(parse_model_config(_TINY))
    snapshot = {
        name: tensor.detach().clone()
        for name, tensor in source.state_dict().items()
    }
    destination = DenseDecoder(parse_model_config(_TINY))
    result = destination.load_state_dict(snapshot, strict=True, assign=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    attention = destination.model.layers[0].self_attn
    assert attention._qkv_pack.is_current()
    for name, expected in snapshot.items():
        torch.testing.assert_close(destination.state_dict()[name], expected)

    destination.to(dtype=torch.float64)
    attention = destination.model.layers[0].self_attn
    assert attention._qkv_pack.is_current()
    assert attention.q_proj.weight.dtype is torch.float64


def test_tp_pack_keeps_each_checkpoint_member_and_local_output_partition() -> None:
    config = parse_model_config(_TINY)
    local = tp_view(config, tp=2, rank=1)
    model = DenseDecoder(
        local,
        linear_factory=linear_factory(
            QuantConfig(QuantMethod.NONE),
            tensor_parallel_size=2,
            tensor_parallel_rank=1,
        ),
    )
    attention = model.model.layers[0].self_attn
    assert attention._qkv_pack.out_features == (32, 16, 16)

    specs = checkpoint_member_specs(model)
    for projection in ("q_proj", "k_proj", "v_proj"):
        name = f"model.layers.0.self_attn.{projection}.weight"
        assert specs[name].source_name == name
        assert specs[name].shard_dim == 0
    for projection in ("gate_proj", "up_proj"):
        name = f"model.layers.0.mlp.{projection}.weight"
        assert specs[name].source_name == name
        assert specs[name].shard_dim == 0


def test_projection_hook_or_replaced_parameter_uses_safe_separate_fallback() -> None:
    torch.manual_seed(53)
    model = DenseDecoder(parse_model_config(_TINY))
    attention = model.model.layers[0].self_attn
    hidden = torch.randn(3, _TINY["hidden_size"])
    calls = []
    handle = attention.q_proj.register_forward_hook(
        lambda _module, _inputs, _output: calls.append("q")
    )
    try:
        actual = attention._project_qkv(hidden)
    finally:
        handle.remove()
    assert calls == ["q"]
    expected = (
        attention.q_proj(hidden),
        attention.k_proj(hidden),
        attention.v_proj(hidden),
    )
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference)

    attention.k_proj.weight = torch.nn.Parameter(
        attention.k_proj.weight.detach().clone()
    )
    assert not attention._qkv_pack.is_current()
    actual = attention._project_qkv(hidden)
    expected = (
        attention.q_proj(hidden),
        attention.k_proj(hidden),
        attention.v_proj(hidden),
    )
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference)


def test_replaced_projection_module_uses_the_canonical_module() -> None:
    class ReplacementLinear(torch.nn.Linear):
        pass

    torch.manual_seed(59)
    model = DenseDecoder(parse_model_config(_TINY))
    attention = model.model.layers[0].self_attn
    hidden = torch.randn(3, _TINY["hidden_size"])
    old_pack = attention._qkv_pack
    replacement = ReplacementLinear(
        _TINY["hidden_size"],
        attention.q_proj.out_features,
        bias=attention.q_proj.bias is not None,
    )
    with torch.no_grad():
        replacement.weight.fill_(0.125)
        if replacement.bias is not None:
            replacement.bias.fill_(0.25)
    attention.q_proj = replacement

    assert not old_pack.matches(
        (attention.q_proj, attention.k_proj, attention.v_proj)
    )
    actual = attention._project_qkv(hidden)
    expected = (
        attention.q_proj(hidden),
        attention.k_proj(hidden),
        attention.v_proj(hidden),
    )
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_non_dense_and_mixed_bias_groups_decline_packing() -> None:
    class CustomLinear(torch.nn.Linear):
        pass

    assert DenseLinearPack.create(
        (torch.nn.Linear(8, 4), CustomLinear(8, 4))
    ) is None
    assert DenseLinearPack.create(
        (torch.nn.Linear(8, 4, bias=False), torch.nn.Linear(8, 4, bias=True))
    ) is None

    model = DenseDecoder(
        parse_model_config(_TINY),
        linear_factory=linear_factory(QuantConfig(QuantMethod.FP8)),
    )
    layer = model.model.layers[0]
    assert layer.self_attn._qkv_pack is None
    assert not hasattr(layer.mlp, "_gate_up_pack")
