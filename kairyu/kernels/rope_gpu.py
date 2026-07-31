"""One-launch in-place RoPE for the dense CUDA path.

The kernel consumes Kairyu's already-computed fp32 cosine/sine rows.  Besides
supporting CUDA graph replay, this preserves the established numerical
contract: cosine/sine and each product are rounded to the activation dtype
before the final add, exactly like the portable PyTorch expression.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - caller retains the torch fallback
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _round_bf16_to_fp32(value):
        """Round fp32 to BF16 while retaining an fp32 register value.

        The integer bit step is intentional: it prevents LLVM from folding the
        two BF16 product roundings into one fp32 fused multiply-add.
        """

        bits = value.to(tl.int32, bitcast=True)
        bias = 0x7FFF + ((bits >> 16) & 1)
        rounded = (bits + bias) & -65536
        return rounded.to(tl.float32, bitcast=True)

    @triton.jit
    def _rope_inplace_kernel(
        query,
        key,
        cos,
        sin,
        NUM_QUERY_HEADS: tl.constexpr,
        NUM_KEY_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        columns = tl.arange(0, BLOCK)
        half = HEAD_DIM // 2
        payload = columns < half
        upper_columns = columns + half

        lower_cosine = tl.load(
            cos + token * HEAD_DIM + columns,
            mask=payload,
            other=0.0,
        )
        upper_cosine = tl.load(
            cos + token * HEAD_DIM + upper_columns,
            mask=payload,
            other=0.0,
        )
        lower_sine = tl.load(
            sin + token * HEAD_DIM + columns,
            mask=payload,
            other=0.0,
        )
        upper_sine = tl.load(
            sin + token * HEAD_DIM + upper_columns,
            mask=payload,
            other=0.0,
        )
        lower_cosine = _round_bf16_to_fp32(lower_cosine)
        upper_cosine = _round_bf16_to_fp32(upper_cosine)
        lower_sine = _round_bf16_to_fp32(lower_sine)
        upper_sine = _round_bf16_to_fp32(upper_sine)

        query_mask = payload & (head < NUM_QUERY_HEADS)
        query_base = (token * NUM_QUERY_HEADS + head) * HEAD_DIM
        query_lower = tl.load(
            query + query_base + columns,
            mask=query_mask,
            other=0.0,
        )
        query_upper = tl.load(
            query + query_base + upper_columns,
            mask=query_mask,
            other=0.0,
        )
        query_lower_left = _round_bf16_to_fp32(
            query_lower.to(tl.float32) * lower_cosine
        )
        query_lower_right = _round_bf16_to_fp32(
            -query_upper.to(tl.float32) * lower_sine
        )
        query_upper_left = _round_bf16_to_fp32(
            query_upper.to(tl.float32) * upper_cosine
        )
        query_upper_right = _round_bf16_to_fp32(
            query_lower.to(tl.float32) * upper_sine
        )
        tl.store(
            query + query_base + columns,
            query_lower_left + query_lower_right,
            mask=query_mask,
        )
        tl.store(
            query + query_base + upper_columns,
            query_upper_left + query_upper_right,
            mask=query_mask,
        )

        key_mask = payload & (head < NUM_KEY_HEADS)
        key_base = (token * NUM_KEY_HEADS + head) * HEAD_DIM
        key_lower = tl.load(
            key + key_base + columns,
            mask=key_mask,
            other=0.0,
        )
        key_upper = tl.load(
            key + key_base + upper_columns,
            mask=key_mask,
            other=0.0,
        )
        key_lower_left = _round_bf16_to_fp32(
            key_lower.to(tl.float32) * lower_cosine
        )
        key_lower_right = _round_bf16_to_fp32(
            -key_upper.to(tl.float32) * lower_sine
        )
        key_upper_left = _round_bf16_to_fp32(
            key_upper.to(tl.float32) * upper_cosine
        )
        key_upper_right = _round_bf16_to_fp32(
            key_lower.to(tl.float32) * upper_sine
        )
        tl.store(
            key + key_base + columns,
            key_lower_left + key_lower_right,
            mask=key_mask,
        )
        tl.store(
            key + key_base + upper_columns,
            key_upper_left + key_upper_right,
            mask=key_mask,
        )


def try_apply_rope_inplace(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    """Apply dense half-split RoPE and report whether the fused path ran."""

    if (
        triton is None
        or query.device.type != "cuda"
        or not (query.device == key.device == cos.device == sin.device)
        or query.dtype != key.dtype
        or query.dtype is not torch.bfloat16
        or query.ndim != 3
        or key.ndim != 3
        or cos.ndim != 2
        or sin.ndim != 2
        or query.shape[0] != key.shape[0]
        or query.shape[0] != cos.shape[0]
        or cos.shape != sin.shape
        or query.shape[2] != key.shape[2]
        or cos.shape[1] != query.shape[2]
        or query.shape[2] not in (64, 128, 256)
        or not query.is_contiguous()
        or not key.is_contiguous()
        or not cos.is_contiguous()
        or not sin.is_contiguous()
        or cos.dtype != torch.float32
        or sin.dtype != torch.float32
        or query.requires_grad
        or key.requires_grad
    ):
        return False
    tokens = query.shape[0]
    if tokens == 0:
        return True
    block = triton.next_power_of_2(query.shape[2] // 2)
    num_warps = max(1, min(4, block // 32))
    _rope_inplace_kernel[
        (tokens, max(query.shape[1], key.shape[1]))
    ](
        query,
        key,
        cos,
        sin,
        NUM_QUERY_HEADS=query.shape[1],
        NUM_KEY_HEADS=key.shape[1],
        HEAD_DIM=query.shape[2],
        BLOCK=block,
        num_warps=num_warps,
    )
    return True
