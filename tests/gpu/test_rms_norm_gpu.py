"""CUDA RMSNorm preserves its historical BF16 rounding boundary."""

from __future__ import annotations

import pytest
import torch

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
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
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
