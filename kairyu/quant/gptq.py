"""GPTQ W4A16 pack/unpack (m14 D1, review A5 — verified vs AutoGPTQ + live headers).

Storage: qweight int32 [in//8, out] SEQUENTIAL LSB-first along IN (bits 0-3 =
row 8k — no AWQ-style reorder); qzeros int32 [ceil(in/g), out//8] sequential
LSB-first along OUT, stored as ``z - 1`` with +1 restored at dequant (the
infamous offset); scales fp16 [ceil(in/g), out]; g_idx int32 [in] present even
when desc_act=False. Dequant ``w = (q - (z_stored + 1)) * s`` via g_idx.
"""

from __future__ import annotations

import torch


def pack_gptq_rows(values: torch.Tensor) -> torch.Tensor:
    """int32 [rows, cols] (rows % 8 == 0) -> int32 [rows//8, cols], LSB-first."""
    rows, cols = values.shape
    grouped = values.reshape(rows // 8, 8, cols).to(torch.int64)
    packed = torch.zeros(rows // 8, cols, dtype=torch.int64)
    for nibble in range(8):
        packed |= (grouped[:, nibble, :] & 0xF) << (4 * nibble)
    return packed.to(torch.int32)


def unpack_gptq_rows(packed: torch.Tensor) -> torch.Tensor:
    wide = packed.to(torch.int64) & 0xFFFFFFFF
    nibbles = torch.stack([(wide >> (4 * i)) & 0xF for i in range(8)], dim=1)
    rows8, _, cols = nibbles.shape
    return nibbles.reshape(rows8 * 8, cols).to(torch.int32)


def pack_gptq_cols(values: torch.Tensor) -> torch.Tensor:
    """int32 [rows, cols] (cols % 8 == 0) -> int32 [rows, cols//8], LSB-first."""
    rows, cols = values.shape
    grouped = values.reshape(rows, cols // 8, 8).to(torch.int64)
    packed = torch.zeros(rows, cols // 8, dtype=torch.int64)
    for nibble in range(8):
        packed |= (grouped[:, :, nibble] & 0xF) << (4 * nibble)
    return packed.to(torch.int32)


def unpack_gptq_cols(packed: torch.Tensor) -> torch.Tensor:
    wide = packed.to(torch.int64) & 0xFFFFFFFF
    nibbles = torch.stack([(wide >> (4 * i)) & 0xF for i in range(8)], dim=-1)
    rows, packed_cols, _ = nibbles.shape
    return nibbles.reshape(rows, packed_cols * 8).to(torch.int32)


def quantize_gptq(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """weight [out, in] -> (qweight, qzeros_stored, scales fp16, g_idx)."""
    out_features, in_features = weight.shape
    w = weight.to(torch.float32).t().contiguous()  # [in, out]
    groups = -(-in_features // group_size)
    g_idx = (torch.arange(in_features) // group_size).to(torch.int32)
    scales = torch.zeros(groups, out_features)
    zeros = torch.zeros(groups, out_features)
    q = torch.zeros(in_features, out_features)
    for group in range(groups):
        rows = slice(group * group_size, min((group + 1) * group_size, in_features))
        w_min = w[rows].amin(dim=0)
        w_max = w[rows].amax(dim=0)
        scale = ((w_max - w_min).clamp(min=1e-5) / 15.0)
        zero = torch.clamp(torch.round(-w_min / scale), 0, 15)
        scales[group] = scale
        zeros[group] = zero
        q[rows] = torch.clamp(torch.round(w[rows] / scale) + zero, 0, 15)
    stored_zeros = (zeros - 1).to(torch.int64) & 0xF  # the GPTQ -1 storage offset
    return (
        pack_gptq_rows(q.to(torch.int32)),
        pack_gptq_cols(stored_zeros.to(torch.int32)),
        scales.to(torch.float16),
        g_idx,
    )


def dequantize_gptq(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """-> weight fp32 [out, in]."""
    q = unpack_gptq_rows(qweight).to(torch.float32)  # [in, out]
    zeros = (unpack_gptq_cols(qzeros) + 1).to(torch.float32)  # +1 restored
    groups = g_idx.to(torch.long)
    w = (q - zeros[groups]) * scales.to(torch.float32)[groups]
    return w.t().contiguous()
