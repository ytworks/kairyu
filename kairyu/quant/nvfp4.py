"""NVFP4 (modelopt) pack/unpack (m14 D1, review A8 — verified vs vLLM + live headers).

Storage: ``weight`` uint8 [out, in//2], two e2m1 values per byte packed along
the INPUT axis — LOW nibble = even element, HIGH nibble = odd. Nibble: bit 3 =
sign, bits 0-2 = magnitude index into LUT [0, .5, 1, 1.5, 2, 3, 4, 6].
``weight_scale`` fp8-e4m3 [out, in//16] (row-major; the cutlass swizzle is a
runtime transform); ``weight_scale_2`` fp32 scalar = global_amax / (6 * 448).
Dequant: ``w = lut[q] * fp8_scale * weight_scale_2`` (MULTIPLY).

NOTE (review A8): compressed-tensors FP4 checkpoints use DIFFERENT names
(``weight_packed``) and an INVERTED global scale — they are rejected loudly
upstream; this module implements the modelopt convention only.
"""

from __future__ import annotations

import torch

from kairyu.quant.fp8 import FP8_MAX

E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_BLOCK = 16
FP4_MAX = 6.0


def _cast_to_fp4_indices(values: torch.Tensor) -> torch.Tensor:
    """fp32 magnitudes (clamped to [0, 6]) -> LUT indices, round-to-nearest-even.

    Each boundary is the midpoint between two adjacent LUT levels; a value that
    lands exactly on one resolves to the neighbor with the EVEN LUT INDEX (M1).
    e.g. 0.75 = mid(0.5@1, 1.0@2) -> index 2 (=1.0); 2.5 = mid(2.0@4, 3.0@5) ->
    index 4 (=2.0). (The prior table applied LUT *values* as indices and dropped
    two boundaries, corrupting the packing oracle by up to 60%.)
    """
    boundaries = values.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    indices = torch.bucketize(values, boundaries, right=False)
    # override every exact-boundary value to its even LUT index (idempotent where
    # bucketize already lands even; corrective where it lands odd)
    ties = ((0.25, 0), (0.75, 2), (1.25, 2), (1.75, 4), (2.5, 4), (3.5, 6), (5.0, 6))
    for boundary, even_index in ties:
        indices = torch.where(
            values == boundary, indices.new_tensor(even_index), indices
        )
    return indices


def quantize_nvfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """weight [out, in] (in % 16 == 0) -> (packed uint8 [out, in//2],
    block scales fp8 [out, in//16], global scale fp32 scalar)."""
    w = weight.to(torch.float32)
    global_scale = (w.abs().amax() / (FP4_MAX * FP8_MAX)).clamp(min=1e-12)
    packed, fp8_scale = quantize_nvfp4_with_scale(w, global_scale)
    return packed, fp8_scale, global_scale.reshape(())


def quantize_nvfp4_with_scale(
    values: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor using a checkpoint-provided global multiplier."""
    out_features, in_features = values.shape
    if in_features % _BLOCK:
        raise ValueError("NVFP4 last dimension must be divisible by 16")
    w = values.to(torch.float32)
    global_scale = global_scale.to(device=w.device, dtype=torch.float32)
    if global_scale.numel() != 1 or global_scale.item() <= 0:
        raise ValueError("NVFP4 global scale must be one positive scalar")
    blocks = w.reshape(out_features, in_features // _BLOCK, _BLOCK)
    block_amax = blocks.abs().amax(dim=-1)
    fp8_scale = (
        (block_amax / FP4_MAX / global_scale).clamp(-FP8_MAX, FP8_MAX)
    ).to(torch.float8_e4m3fn)
    effective = fp8_scale.to(torch.float32) * global_scale
    scaled = (blocks / effective.clamp(min=1e-12)[:, :, None]).clamp(-FP4_MAX, FP4_MAX)
    signs = (scaled < 0).to(torch.uint8) << 3
    indices = _cast_to_fp4_indices(scaled.abs()).to(torch.uint8)
    nibbles = (signs | indices).reshape(out_features, in_features)
    packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)  # low nibble = even elem
    return packed, fp8_scale


def quantize_nvfp4_per_token(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference per-token global scaling used by the dynamic CUDA path."""
    if values.ndim != 2 or values.shape[1] % _BLOCK:
        raise ValueError("dynamic NVFP4 input must be 2D with K divisible by 16")
    source = values.to(torch.float32)
    row_amax = source.abs().amax(dim=1, keepdim=True)
    global_scale = (row_amax / (FP4_MAX * FP8_MAX)).clamp(min=1e-12)
    blocks = source.reshape(source.shape[0], source.shape[1] // _BLOCK, _BLOCK)
    block_amax = blocks.abs().amax(dim=-1)
    block_scale = (
        block_amax / FP4_MAX / global_scale
    ).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    effective = block_scale.to(torch.float32) * global_scale
    scaled = (blocks / effective.clamp(min=1e-12)[..., None]).clamp(
        -FP4_MAX, FP4_MAX
    )
    signs = (scaled < 0).to(torch.uint8) << 3
    indices = _cast_to_fp4_indices(scaled.abs()).to(torch.uint8)
    nibbles = (signs | indices).reshape_as(source)
    packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    return packed, block_scale, global_scale


def activation_saturation_counts(
    values: torch.Tensor,
    static_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return device-resident saturated and total group-16 block counts."""
    source = values.reshape(-1, values.shape[-1]).to(torch.float32)
    if source.shape[1] % _BLOCK:
        raise ValueError("NVFP4 activation width must be divisible by 16")
    blocks = source.reshape(source.shape[0], source.shape[1] // _BLOCK, _BLOCK)
    block_amax = blocks.abs().amax(dim=-1)
    if static_scale is None:
        row_amax = source.abs().amax(dim=1, keepdim=True)
        scale = (row_amax / (FP4_MAX * FP8_MAX)).clamp(min=1e-12)
    else:
        if static_scale.numel() != 1:
            raise ValueError("static NVFP4 input scale must be scalar")
        scale = static_scale.to(device=source.device, dtype=torch.float32).reshape(1, 1)
    saturated = (block_amax > scale * FP4_MAX * FP8_MAX).sum(dtype=torch.int64)
    total = torch.full(
        (), block_amax.numel(), dtype=torch.int64, device=source.device
    )
    return saturated, total


def unpack_nvfp4(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [out, in//2] -> fp32 LUT values with sign, [out, in]."""
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    nibbles = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    magnitudes = E2M1_LUT.to(packed.device)[(nibbles & 0x7).to(torch.long)]
    signs = torch.where(nibbles & 0x8 > 0, -1.0, 1.0)
    return magnitudes * signs


def dequantize_nvfp4(
    packed: torch.Tensor, fp8_scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    values = unpack_nvfp4(packed)  # [out, in]
    out_features, in_features = values.shape
    blocks = values.reshape(out_features, in_features // _BLOCK, _BLOCK)
    global_scale = global_scale.to(torch.float32)
    if global_scale.numel() == 1:
        multiplier = global_scale.reshape(1, 1, 1)
    elif global_scale.numel() == out_features:
        multiplier = global_scale.reshape(out_features, 1, 1)
    else:
        raise ValueError("NVFP4 global scale must be scalar or per row")
    scaled = blocks * fp8_scale.to(torch.float32)[:, :, None] * multiplier
    return scaled.reshape(out_features, in_features)
