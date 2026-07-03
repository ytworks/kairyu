"""PagedKVPool ⇄ PageFrame serde (m18 D1).

Fragments are the layer-major slices — ``2 * num_layers`` per page
(``k[layer, page]`` then ``v[layer, page]``; both contiguous by layout,
verified). MLA pools (v width 0) produce EMPTY v fragments: inject asserts
``b""`` and skips (``torch.frombuffer`` rejects empty buffers — reviewed).
Serde goes through numpy, so dtypes are pinned to numpy-compatible ones
(fp32/fp16/int8/uint8); bf16 pools route through a uint8 view on deploy day
(recorded, not implemented — CPU pools are fp32).

``pool_fingerprint`` validates sender/receiver pool compatibility at
rendezvous time (the transport has no handshake hook by design).
"""

from __future__ import annotations

import numpy as np
import torch

from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.engine.core.kv_transport import KVTransportError, PageFrame


def pool_fingerprint(pool: PagedKVPool) -> str:
    return (
        f"L{pool.num_layers}-P{pool.page_size}-H{pool.num_kv_heads}"
        f"-D{pool.head_dim}-V{pool.v_head_dim}-{pool.k.dtype}"
    )


def extract_page(pool: PagedKVPool, page_id: int) -> PageFrame:
    """One logical page -> its per-layer fragments (k then v, per layer)."""
    fragments: list[bytes] = []
    for layer in range(pool.num_layers):
        fragments.append(pool.k[layer, page_id].contiguous().numpy().tobytes())
        if pool.v_head_dim:
            fragments.append(pool.v[layer, page_id].contiguous().numpy().tobytes())
        else:
            fragments.append(b"")  # MLA: latent lives in k (m15 contract)
    return PageFrame(page_id=page_id, fragments=tuple(fragments))


def inject_page(pool: PagedKVPool, page_id: int, frame: PageFrame) -> None:
    """Write a received frame into LOCAL ``page_id`` (sender ids are
    meaningless here — the receiver remaps, m18 amendment 3)."""
    expected = 2 * pool.num_layers
    if len(frame.fragments) != expected:
        raise KVTransportError(
            f"frame has {len(frame.fragments)} fragments, pool needs {expected}"
        )
    numpy_dtype = torch.empty(0, dtype=pool.k.dtype).numpy().dtype
    for layer in range(pool.num_layers):
        k_bytes = frame.fragments[2 * layer]
        v_bytes = frame.fragments[2 * layer + 1]
        k_target = pool.k[layer, page_id]
        if len(k_bytes) != k_target.numel() * k_target.element_size():
            raise KVTransportError(
                f"layer {layer} k fragment is {len(k_bytes)} bytes, "
                f"expected {k_target.numel() * k_target.element_size()}"
            )
        k_array = np.frombuffer(k_bytes, dtype=numpy_dtype).copy()
        k_target.copy_(torch.from_numpy(k_array).reshape(k_target.shape))
        if pool.v_head_dim:
            v_target = pool.v[layer, page_id]
            if len(v_bytes) != v_target.numel() * v_target.element_size():
                raise KVTransportError(f"layer {layer} v fragment length mismatch")
            v_array = np.frombuffer(v_bytes, dtype=numpy_dtype).copy()
            v_target.copy_(torch.from_numpy(v_array).reshape(v_target.shape))
        elif v_bytes:
            raise KVTransportError("MLA pool received a non-empty v fragment")


def extract_pages(pool: PagedKVPool, page_ids: tuple[int, ...]) -> tuple[PageFrame, ...]:
    return tuple(extract_page(pool, page_id) for page_id in page_ids)
