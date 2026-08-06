"""CUDA RMSNorm preserves its historical BF16 rounding boundary."""

from __future__ import annotations

import pytest
import torch

from kairyu.bench.profiling import profile_scope
from kairyu.kernels.rms_norm_gpu import try_joint_qk_rms_norm
from kairyu.models.layers import RMSNorm

pytestmark = pytest.mark.gpu


def _reference(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = hidden.dtype
    normalized = hidden.float()
    normalized = normalized * torch.rsqrt(
        normalized.pow(2).mean(-1, keepdim=True) + eps
    )
    return (weight * normalized.to(dtype)).to(dtype)


@pytest.mark.parametrize(
    "shape",
    (
        (1, 5120),
        (16, 5120),
        (1, 16, 128),
        (16, 16, 128),
        (1, 2, 128),
        (16, 2, 128),
    ),
)
def test_cuda_rms_norm_is_bit_exact_to_explicit_reference(shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(156)
    layer = RMSNorm(shape[-1], 1e-6).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(
            torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
        )
    hidden = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    # Exclude one-time dispatcher/kernel loading from the structural profile.
    layer(hidden)
    torch.cuda.synchronize()
    with profile_scope(
        enabled=True,
        activities=("cpu", "cuda"),
        acc_events=False,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profile:
        actual = layer(hidden)
    expected = _reference(hidden, layer.weight, layer.eps)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    cuda_events = [
        event
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    # Bind the actual device work, independently of whether the installed CUDA
    # runtime labels its CPU launch event cudaLaunchKernel or cuLaunchKernelEx.
    assert [event.name for event in cuda_events] == ["_rms_norm_bf16_kernel"]


def _qwen_tp4_packed_qk(
    batch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed_qkv = torch.randn(
        batch,
        2560,
        device="cuda",
        dtype=torch.bfloat16,
    )
    query = packed_qkv[:, :2048].view(batch, 16, 128)
    keys = packed_qkv[:, 2048:2304].view(batch, 2, 128)
    assert query.stride()[1:] == (128, 1)
    assert keys.stride()[1:] == (128, 1)
    if batch > 1:
        assert query.stride(0) == 2560
        assert keys.stride(0) == 2560
    return packed_qkv, query, keys


@pytest.mark.parametrize("batch", (1, 16))
def test_joint_qk_rms_norm_is_one_bit_exact_launch_for_packed_tp4_views(
    batch,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(158 + batch)
    _packed, query, keys = _qwen_tp4_packed_qk(batch)
    query_weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    key_weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)

    # Compile before profiling so the assertion describes steady device work.
    assert try_joint_qk_rms_norm(
        query,
        keys,
        query_weight,
        key_weight,
        1e-6,
    ) is not None
    torch.cuda.synchronize()
    with profile_scope(
        enabled=True,
        activities=("cpu", "cuda"),
        acc_events=False,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profile:
        result = try_joint_qk_rms_norm(
            query,
            keys,
            query_weight,
            key_weight,
            1e-6,
        )
        assert result is not None
        actual_query, actual_keys = result
        torch.cuda.synchronize()

    expected_query = _reference(query, query_weight, 1e-6)
    expected_keys = _reference(keys, key_weight, 1e-6)
    torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(actual_keys, expected_keys, rtol=0, atol=0)
    assert actual_query.is_contiguous()
    assert actual_keys.is_contiguous()
    cuda_events = [
        event
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert [event.name for event in cuda_events] == [
        "_joint_qk_rms_norm_bf16_kernel"
    ]


def test_joint_qk_rms_norm_declines_unsupported_layout_and_dtype():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    query = torch.randn(
        16,
        128,
        16,
        device="cuda",
        dtype=torch.bfloat16,
    ).transpose(1, 2)
    keys = torch.randn(
        16,
        128,
        2,
        device="cuda",
        dtype=torch.bfloat16,
    ).transpose(1, 2)
    query_weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    key_weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    assert query.shape == (16, 16, 128)
    assert keys.shape == (16, 2, 128)
    assert query.stride(-1) != 1
    assert keys.stride(-1) != 1
    assert try_joint_qk_rms_norm(
        query,
        keys,
        query_weight,
        key_weight,
        1e-6,
    ) is None

    query = query.contiguous()
    keys = keys.contiguous()
    assert try_joint_qk_rms_norm(
        query,
        keys,
        query_weight.float(),
        key_weight,
        1e-6,
    ) is None
    assert try_joint_qk_rms_norm(
        query,
        keys,
        query_weight.view(1, 128),
        key_weight,
        1e-6,
    ) is None


def test_joint_qk_rms_norm_is_cuda_graph_capturable_for_packed_views():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(175)
    packed, query, keys = _qwen_tp4_packed_qk(batch=16)
    query_weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    key_weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            try_joint_qk_rms_norm(
                query,
                keys,
                query_weight,
                key_weight,
                1e-6,
            )
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = try_joint_qk_rms_norm(
            query,
            keys,
            query_weight,
            key_weight,
            1e-6,
        )
        assert captured is not None
    replacement = torch.randn_like(packed)
    packed.copy_(replacement)
    graph.replay()
    torch.cuda.synchronize()
    replayed_query = captured[0].clone()
    replayed_keys = captured[1].clone()
    expected_query = _reference(query, query_weight, 1e-6)
    expected_keys = _reference(keys, key_weight, 1e-6)
    torch.testing.assert_close(replayed_query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(replayed_keys, expected_keys, rtol=0, atol=0)
