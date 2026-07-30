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
        payload = columns < HEAD_DIM
        half = HEAD_DIM // 2
        rotated_columns = tl.where(
            columns < half, columns + half, columns - half
        )
        sign = tl.where(columns < half, -1.0, 1.0)

        cosine = tl.load(
            cos + token * HEAD_DIM + columns,
            mask=payload,
            other=0.0,
        )
        sine = tl.load(
            sin + token * HEAD_DIM + columns,
            mask=payload,
            other=0.0,
        )
        cosine = _round_bf16_to_fp32(cosine)
        sine = _round_bf16_to_fp32(sine)

        query_mask = payload & (head < NUM_QUERY_HEADS)
        query_base = (token * NUM_QUERY_HEADS + head) * HEAD_DIM
        query_value = tl.load(
            query + query_base + columns,
            mask=query_mask,
            other=0.0,
        )
        query_rotated = sign * tl.load(
            query + query_base + rotated_columns,
            mask=query_mask,
            other=0.0,
        )
        query_left = _round_bf16_to_fp32(
            query_value.to(tl.float32) * cosine
        )
        query_right = _round_bf16_to_fp32(
            query_rotated.to(tl.float32) * sine
        )
        query_output = query_left + query_right
        tl.store(
            query + query_base + columns,
            query_output,
            mask=query_mask,
        )

        key_mask = payload & (head < NUM_KEY_HEADS)
        key_base = (token * NUM_KEY_HEADS + head) * HEAD_DIM
        key_value = tl.load(
            key + key_base + columns,
            mask=key_mask,
            other=0.0,
        )
        key_rotated = sign * tl.load(
            key + key_base + rotated_columns,
            mask=key_mask,
            other=0.0,
        )
        key_left = _round_bf16_to_fp32(
            key_value.to(tl.float32) * cosine
        )
        key_right = _round_bf16_to_fp32(
            key_rotated.to(tl.float32) * sine
        )
        key_output = key_left + key_right
        tl.store(key + key_base + columns, key_output, mask=key_mask)


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
    block = triton.next_power_of_2(query.shape[2])
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
        num_warps=4,
    )
    return True
