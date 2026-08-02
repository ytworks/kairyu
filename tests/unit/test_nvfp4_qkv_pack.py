"""Packed NVFP4 Q/K/V execution and load-time operand preparation."""

from __future__ import annotations

import torch

from kairyu.kernels.quant_gemm_gpu import prepare_nvfp4_linear
from kairyu.models.packed_linear import NvFp4LinearPack
from kairyu.quant import nvfp4
from kairyu.quant.linear import NvFp4Linear


def _compatible_qkv(
    *,
    in_features: int = 64,
    out_features: tuple[int, int, int] = (64, 32, 32),
) -> tuple[NvFp4Linear, NvFp4Linear, NvFp4Linear]:
    modules = tuple(
        NvFp4Linear(in_features, rows, bias=False) for rows in out_features
    )
    full_weight = torch.randn(sum(out_features), in_features)
    packed, block_scale, global_scale = nvfp4.quantize_nvfp4(full_weight)
    row = 0
    for module in modules:
        end = row + module.out_features
        module._buffers["weight"] = packed[row:end].clone()
        module._buffers["weight_scale"] = block_scale[row:end].clone()
        module._buffers["weight_scale_2"] = global_scale.clone()
        module.input_scale.fill_(0.125)
        row = end
    return modules


def test_nvfp4_qkv_pack_quantizes_activation_once_and_matches_canonical_math(
    monkeypatch,
) -> None:
    torch.manual_seed(17)
    projections = _compatible_qkv()
    pack = NvFp4LinearPack.create(projections)
    assert pack is not None
    hidden = torch.randn(5, 64, dtype=torch.bfloat16)
    expected = tuple(module.forward_reference(hidden) for module in projections)

    calls = 0
    original = nvfp4.quantize_nvfp4_with_scale

    def observed_quantize(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(nvfp4, "quantize_nvfp4_with_scale", observed_quantize)
    actual = pack(hidden)

    assert calls == 1
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_nvfp4_qkv_pack_preserves_qwen3_235b_geometry_without_weight_alias_copy() -> None:
    projections = tuple(
        NvFp4Linear(4096, rows, bias=False)
        for rows in (8192, 512, 512)
    )
    pack = NvFp4LinearPack.create(projections)

    assert pack is not None
    assert pack.in_features == 4096
    assert pack.out_features == 9216
    assert pack.split_features == (8192, 512, 512)
    assert pack.weight.shape == (9216, 2048)
    assert pack.weight_scale.shape == (9216, 256)
    weight_storage = pack.weight.untyped_storage()
    scale_storage = pack.weight_scale.untyped_storage()
    assert all(
        module.weight.untyped_storage().data_ptr() == weight_storage.data_ptr()
        for module in projections
    )
    assert all(
        module.weight_scale.untyped_storage().data_ptr() == scale_storage.data_ptr()
        for module in projections
    )
    assert weight_storage.nbytes() == sum(
        module.weight.numel() * module.weight.element_size()
        for module in projections
    )
    assert scale_storage.nbytes() == sum(
        module.weight_scale.numel() * module.weight_scale.element_size()
        for module in projections
    )


def test_nvfp4_qkv_pack_keeps_canonical_state_dict_paths() -> None:
    owner = torch.nn.Module()
    owner.q_proj, owner.k_proj, owner.v_proj = _compatible_qkv()
    names_before = tuple(owner.state_dict())

    pack = NvFp4LinearPack.create((owner.q_proj, owner.k_proj, owner.v_proj))

    assert pack is not None
    assert tuple(owner.state_dict()) == names_before
    assert set(names_before) == {
        f"{projection}.{member}"
        for projection in ("q_proj", "k_proj", "v_proj")
        for member in ("weight", "weight_scale", "weight_scale_2", "input_scale")
    }
    assert not any("qkv" in name for name in names_before)


def test_nvfp4_qkv_pack_declines_scale_mismatch_and_hooks_invalidate_execution() -> None:
    projections = _compatible_qkv()
    projections[1].input_scale.mul_(2)
    assert NvFp4LinearPack.create(projections) is None

    projections = _compatible_qkv()
    pack = NvFp4LinearPack.create(projections)
    assert pack is not None and pack.is_current()
    handle = projections[0].register_forward_hook(
        lambda _module, _inputs, output: output
    )
    try:
        assert not pack.is_current()
    finally:
        handle.remove()


def test_nvfp4_linear_prepares_scalar_operands_once_and_invalidates_by_version() -> None:
    module = NvFp4Linear(in_features=64, out_features=32, bias=False)
    module.input_scale.fill_(0.25)
    module.weight_scale_2.fill_(0.5)

    first = prepare_nvfp4_linear(module)
    second = prepare_nvfp4_linear(module)

    assert first[0] is module.weight
    assert first[0] is second[0]
    assert first[1] is second[1]
    assert first[2] is second[2]
    assert first[3] is second[3]
    torch.testing.assert_close(first[2], torch.tensor([4.0]))
    torch.testing.assert_close(first[3], torch.tensor(0.125))

    module.input_scale.fill_(0.5)
    refreshed = prepare_nvfp4_linear(module)
    assert refreshed[1] is first[1]
    assert refreshed[2] is not first[2]
    assert refreshed[3] is not first[3]
    torch.testing.assert_close(refreshed[2], torch.tensor([2.0]))
    torch.testing.assert_close(refreshed[3], torch.tensor(0.25))
