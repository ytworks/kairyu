"""Numerical, dispatch, and graph-replay gates for fused in-place RoPE."""

import pytest
import torch

pytestmark = pytest.mark.gpu


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():  # pragma: no cover - CPU CI
        pytest.skip("fused RoPE needs CUDA")
    pytest.importorskip("triton")
    return torch.device("cuda:0")


def _cos_sin(
    positions: torch.Tensor,
    *,
    dim: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, device=positions.device).float()
            / dim
        )
    )
    frequencies = positions.float()[:, None] * inv_freq[None, :]
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return embedding.cos(), embedding.sin()


def _assert_within_bf16_ulps(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    max_ulps: int,
) -> None:
    """Bind every element to the local BF16 spacing around the reference."""

    positive = torch.full_like(reference, float("inf"))
    negative = torch.full_like(reference, float("-inf"))
    ulp = torch.maximum(
        (torch.nextafter(reference, positive) - reference).abs(),
        (reference - torch.nextafter(reference, negative)).abs(),
    ).float()
    error = (actual.float() - reference.float()).abs()
    violating = error > max_ulps * ulp
    assert not bool(violating.any()), (
        f"{int(violating.sum())}/{reference.numel()} values exceed "
        f"{max_ulps} BF16 ULP; max_error={float(error.max())}"
    )


def test_default_rope_is_inplace_and_exact_to_reference(monkeypatch):
    from kairyu.kernels import rope_gpu
    from kairyu.models.layers import apply_rope

    device = _require_cuda()
    torch.manual_seed(229)
    theta = 1_000_000.0
    positions = torch.tensor(
        [0, 1, 127, 40959], dtype=torch.int64, device=device
    )
    q = torch.randn(4, 16, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(4, 2, 128, device=device, dtype=torch.bfloat16)
    cos, sin = _cos_sin(positions, dim=128, theta=theta)
    reference_q, reference_k = apply_rope(
        q.clone(), k.clone(), cos, sin
    )
    q_pointer, k_pointer = q.data_ptr(), k.data_ptr()
    fused_results = []
    fused_rope = rope_gpu.try_apply_rope_inplace

    def spy_fused_rope(*args, **kwargs):
        result = fused_rope(*args, **kwargs)
        fused_results.append(result)
        return result

    monkeypatch.setattr(
        rope_gpu, "try_apply_rope_inplace", spy_fused_rope
    )

    actual_q, actual_k = apply_rope(
        q,
        k,
        cos,
        sin,
        positions=positions,
        rope_theta=theta,
    )
    torch.cuda.synchronize()

    assert fused_results == [True]
    assert actual_q.data_ptr() == q_pointer
    assert actual_k.data_ptr() == k_pointer
    _assert_within_bf16_ulps(actual_q, reference_q, max_ulps=0)
    _assert_within_bf16_ulps(actual_k, reference_k, max_ulps=0)


def test_default_rope_cuda_graph_replay_reads_current_cos_sin(monkeypatch):
    from kairyu.kernels import rope_gpu
    from kairyu.models.layers import apply_rope

    device = _require_cuda()
    torch.manual_seed(233)
    theta = 1_000_000.0
    positions = torch.tensor([1, 2], dtype=torch.int32, device=device)
    q = torch.randn(2, 16, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(2, 2, 128, device=device, dtype=torch.bfloat16)
    cos, sin = _cos_sin(positions, dim=128, theta=theta)
    fused_results = []
    fused_rope = rope_gpu.try_apply_rope_inplace

    def spy_fused_rope(*args, **kwargs):
        result = fused_rope(*args, **kwargs)
        fused_results.append(result)
        return result

    monkeypatch.setattr(
        rope_gpu, "try_apply_rope_inplace", spy_fused_rope
    )

    # Load/compile the AOT wrapper before capture.
    apply_rope(
        q,
        k,
        cos,
        sin,
        positions=positions,
        rope_theta=theta,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        apply_rope(
            q,
            k,
            cos,
            sin,
            positions=positions,
            rope_theta=theta,
        )
    assert fused_results == [True, True]

    q.copy_(torch.randn_like(q))
    k.copy_(torch.randn_like(k))
    positions.copy_(torch.tensor([511, 40959], dtype=torch.int32, device=device))
    replay_cos, replay_sin = _cos_sin(positions, dim=128, theta=theta)
    cos.copy_(replay_cos)
    sin.copy_(replay_sin)
    reference_q, reference_k = apply_rope(
        q.clone(), k.clone(), replay_cos, replay_sin
    )
    graph.replay()
    torch.cuda.synchronize()

    assert fused_results == [True, True]
    _assert_within_bf16_ulps(q, reference_q, max_ulps=0)
    _assert_within_bf16_ulps(k, reference_k, max_ulps=0)
