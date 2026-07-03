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


from kairyu.engine.core.attention.selector import select_backend  # noqa: E402
from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend  # noqa: E402

__all__ = ["AttentionBackend", "TorchAttentionBackend", "select_backend"]
