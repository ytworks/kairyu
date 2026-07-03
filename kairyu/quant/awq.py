"""AWQ W4A16 pack/unpack (m14 D1, review A4 — verified vs AutoAWQ + live headers).

Storage: qweight int32 [in, out//8] packed along OUT — nibble i (bits 4i..4i+3)
of packed column j holds original column ``8j + ORDER[i]`` with
ORDER = [0, 2, 4, 6, 1, 3, 5, 7]; qzeros int32 [in//g, out//8] same packing;
scales fp16 [in//g, out]. Dequant ``w = (q - z) * s`` — NO +1 offset (that is
GPTQ's convention, not AWQ's).
"""

from __future__ import annotations

import torch

AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def pack_awq(values: torch.Tensor) -> torch.Tensor:
    """values int32 [rows, cols] (cols % 8 == 0) -> int32 [rows, cols//8]."""
    rows, cols = values.shape
    grouped = values.reshape(rows, cols // 8, 8)
    packed = torch.zeros(rows, cols // 8, dtype=torch.int64)
    for nibble, source in enumerate(AWQ_ORDER):
        packed |= (grouped[:, :, source].to(torch.int64) & 0xF) << (4 * nibble)
    return packed.to(torch.int32)


def unpack_awq(packed: torch.Tensor) -> torch.Tensor:
    """int32 [rows, cols//8] -> int32 [rows, cols] in original column order."""
    rows, packed_cols = packed.shape
    wide = packed.to(torch.int64) & 0xFFFFFFFF
    nibbles = torch.stack(
        [(wide >> (4 * i)) & 0xF for i in range(8)], dim=-1
    )  # order [0,2,4,6,1,3,5,7]
    restored = nibbles[:, :, list(AWQ_REVERSE_ORDER)]
    return restored.reshape(rows, packed_cols * 8).to(torch.int32)


def quantize_awq(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """weight [out, in] -> (qweight [in, out//8], qzeros [in//g, out//8],
    scales fp16 [in//g, out]) — checkpoint layout."""
    out_features, in_features = weight.shape
    w = weight.to(torch.float32).t().contiguous()  # [in, out]
    groups = in_features // group_size
    w_grouped = w.reshape(groups, group_size, out_features)
    w_min = w_grouped.amin(dim=1)  # [groups, out]
    w_max = w_grouped.amax(dim=1)
    scales = ((w_max - w_min).clamp(min=1e-5) / 15.0).to(torch.float32)
    zeros = torch.clamp(torch.round(-w_min / scales), 0, 15)
    q = torch.clamp(
        torch.round(w_grouped / scales[:, None, :]) + zeros[:, None, :], 0, 15
    ).reshape(in_features, out_features).to(torch.int32)
    return (
        pack_awq(q),
        pack_awq(zeros.to(torch.int32)),
        scales.to(torch.float16),
    )


def dequantize_awq(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """-> weight fp32 [out, in] (transposed back to nn.Linear layout)."""
    q = unpack_awq(qweight).to(torch.float32)  # [in, out]
    zeros = unpack_awq(qzeros).to(torch.float32)  # [in//g, out]
    in_features = q.shape[0]
    group_index = torch.arange(in_features) // group_size
    w = (q - zeros[group_index]) * scales.to(torch.float32)[group_index]
    return w.t().contiguous()
