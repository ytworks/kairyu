"""Fail-closed fix for SM120 invalid sparse-KV rows containing NaN payloads.

Copies from an aligned zero device row for invalid indices. Copy lengths and
mbarrier transaction counts are unchanged; valid KV addressing is unchanged.
"""

# Source anchors intentionally retain pinned C++ line boundaries.
# ruff: noqa: E501
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/attention/sparse_mla_sm120"
)
SOURCE_SHA256 = {
    "common/kv_cache_io.cuh": "8cd5f05c2ce08627aca6e878aab2e336ff11cf77e606233d8675554b0d9f1c2d",
    "decode_dsv4_kernel.cuh": "e9d939bd6cc23cdfb1c8a9be0811e0d8ee3307a930a67ff02c30c4450299ab7d",
    "prefill_common.cuh": "7ca57dea73afa2f3866bad50b029bb612e4a0b295cbe181a3f095273ec197081",
}
ZERO_ROW = """// Invalid candidates must not read recycled/unwritten slot 0: 0 * NaN is NaN.
// Global device storage permits the same cp.async.bulk transaction and barrier
// accounting as valid rows. Never written by kernels. Covers every row ABI here.
static __device__ __align__(16) uint8_t kKairyuInvalidKVZeroRow[2048] = {};

"""


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one source anchor: {old!r}")
    return text.replace(old, new, 1)


def transform(name, text):
    if name == "common/kv_cache_io.cuh":
        text = replace_once(
            text,
            "// KV cache IO: gather BI entries",
            ZERO_ROW + "// KV cache IO: gather BI entries",
        )
        text = replace_once(
            text,
            "  idx = (idx >= 0) ? idx : 0;\n\n  const uint8_t* src;",
            "  const bool valid_idx = idx >= 0;\n  idx = valid_idx ? idx : 0;\n\n  const uint8_t* src;",
        )
        text = replace_once(
            text,
            "  if constexpr (USE_L2_HINT)\n    cp_async_bulk_g2s_l2hint",
            "  static_assert(COPY_BYTES <= sizeof(kKairyuInvalidKVZeroRow));\n"
            "  if (!valid_idx) src = kKairyuInvalidKVZeroRow;\n"
            "  if constexpr (USE_L2_HINT)\n    cp_async_bulk_g2s_l2hint",
        )
        text = replace_once(
            text,
            "  idx = (idx >= 0) ? idx : 0;\n\n  int block_idx",
            "  if (idx < 0) {\n"
            "    *reinterpret_cast<uint64_t*>(scale_dst + io_tid * SCALE_BYTES) = 0;\n"
            "    return;\n  }\n\n  int block_idx",
        )
    elif name == "decode_dsv4_kernel.cuh":
        # Decode is a separate translation unit and does not include common KV IO.
        text = replace_once(
            text,
            "namespace flashinfer::sparse_mla_sm120 {\n",
            "namespace flashinfer::sparse_mla_sm120 {\n\n"
            + ZERO_ROW.replace("kKairyuInvalidKVZeroRow", "kKairyuDecodeInvalidKVZeroRow"),
        )
        text = replace_once(
            text,
            "          section_kv + (size_t)block_idx_g * section_stride + (size_t)local_idx_g * IO_STRIDE;\n      cp_async_bulk_g2s(kv_fp8_dst",
            "          idx_raw[e] >= 0\n"
            "              ? section_kv + (size_t)block_idx_g * section_stride + (size_t)local_idx_g * IO_STRIDE\n"
            "              : kKairyuDecodeInvalidKVZeroRow;\n"
            "      static_assert(D_NOPE + DSV4_BULK_ROPE_BYTES <= sizeof(kKairyuDecodeInvalidKVZeroRow));\n"
            "      cp_async_bulk_g2s(kv_fp8_dst",
        )
    elif name == "prefill_common.cuh":
        text = replace_once(
            text, "  idx = (idx >= 0) ? idx : 0;", "  if (idx < 0) return kKairyuInvalidKVZeroRow;"
        )
    else:
        raise ValueError(f"Unsupported source file: {name}")
    return text


def patch(root=ROOT):
    updates = {}
    for name, expected in SOURCE_SHA256.items():
        raw = (root / name).read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Unrecognized FlashInfer source {name}: {actual}, expected {expected}"
            )
        updates[name] = transform(name, raw.decode())
    # All source hashes and anchors must pass before writing any file.
    for name, text in updates.items():
        (root / name).write_text(text)
        print(f"{name} {hashlib.sha256(text.encode()).hexdigest()}")


if __name__ == "__main__":
    patch()
