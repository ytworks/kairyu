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
    """fp32 magnitudes (clamped to [0, 6]) -> LUT indices, round-to-nearest-even
    at the boundaries (0.25 -> 0, 0.75 -> 1, 2.5 -> 2, 3.5 -> 4, 5.0 -> 4)."""
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    indices = torch.bucketize(values, boundaries, right=False)
    # bucketize(right=False) puts boundary values in the HIGHER bucket; RNE
    # boundary cases resolve to the EVEN LUT index — fix the five boundaries
    for boundary, even_index in ((0.25, 0), (0.75, 1), (2.5, 2), (3.5, 4), (5.0, 4)):
        indices = torch.where(values == boundary, torch.tensor(even_index), indices)
    return indices


def quantize_nvfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """weight [out, in] (in % 16 == 0) -> (packed uint8 [out, in//2],
    block scales fp8 [out, in//16], global scale fp32 scalar)."""
    out_features, in_features = weight.shape
    w = weight.to(torch.float32)
    global_scale = (w.abs().amax() / (FP4_MAX * FP8_MAX)).clamp(min=1e-12)
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
    return packed, fp8_scale, global_scale.reshape(())


def unpack_nvfp4(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [out, in//2] -> fp32 LUT values with sign, [out, in]."""
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    nibbles = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    magnitudes = E2M1_LUT[(nibbles & 0x7).to(torch.long)]
    signs = torch.where(nibbles & 0x8 > 0, -1.0, 1.0)
    return magnitudes * signs


def dequantize_nvfp4(
    packed: torch.Tensor, fp8_scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    values = unpack_nvfp4(packed)  # [out, in]
    out_features, in_features = values.shape
    blocks = values.reshape(out_features, in_features // _BLOCK, _BLOCK)
    scaled = blocks * fp8_scale.to(torch.float32)[:, :, None] * global_scale.to(
        torch.float32
    )
    return scaled.reshape(out_features, in_features)
