"""AttentionBackend seam (m13 D1).

Backends are plain objects satisfying the protocol — NEVER nn.Module (a
stateful backend like FlashInfer's workspace must not register as a submodule
or checkpoints grow bogus keys). ``query`` positions are contiguous from
``chunk_start`` (documented invariant all call sites satisfy).
"""

from typing import Protocol

import torch

from kairyu.engine.core.kv_pool import PagedKVPool


class AttentionBackend(Protocol):
    def attend(
        self,
        query: torch.Tensor,
        kv_pool: PagedKVPool,
        layer: int,
        page_table: list[int],
        seq_len: int,
        chunk_start: int,
    ) -> torch.Tensor:
        """query [T, heads, head_dim] -> context [T, heads * head_dim]."""
        ...

    def attend_batched(
        self,
        queries: list[torch.Tensor],
        kv_pool: PagedKVPool,
        layer: int,
        page_tables: list[list[int]],
        seq_lens: list[int],
        chunk_starts: list[int],
    ) -> list[torch.Tensor]:
        """Per-sequence contexts, identical to per-sequence ``attend`` (C4).

        The batched seam the GPU runner needs: one call per layer per step over N
        sequences instead of N calls. Backends may batch the kernel internally."""
        ...


from kairyu.engine.core.attention.selector import (  # noqa: E402
    select_backend,
    select_backend_name,
)
from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend  # noqa: E402

__all__ = [
    "AttentionBackend",
    "TorchAttentionBackend",
    "select_backend",
    "select_backend_name",
]
