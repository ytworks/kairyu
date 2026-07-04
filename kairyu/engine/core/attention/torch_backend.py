"""Pure-torch paged attention backend (m13 D2) — device-agnostic.

M12's ``paged_attention`` moved verbatim: rectangular chunk mask (a query at
absolute position ``chunk_start + i`` attends keys ``[0, chunk_start + i]``;
SDPA's ``is_causal`` is top-left aligned and wrong over cached prefixes) and
``enable_gqa``. The same code runs CUDA tensors on deploy.
"""

from __future__ import annotations

import torch
from torch import nn

from kairyu.engine.core.kv_pool import PagedKVPool


class TorchAttentionBackend:
    def attend(
        self,
        query: torch.Tensor,
        kv_pool: PagedKVPool,
        layer: int,
        page_table: list[int],
        seq_len: int,
        chunk_start: int,
    ) -> torch.Tensor:
        keys, values = kv_pool.gather(layer, page_table, seq_len)
        chunk_len = query.shape[0]
        positions = torch.arange(chunk_len)[:, None] + chunk_start
        mask = torch.arange(seq_len)[None, :] <= positions  # [T, S] rectangular causal
        out = nn.functional.scaled_dot_product_attention(
            query.transpose(0, 1)[None],  # [1, heads, T, d]
            keys.transpose(0, 1)[None],  # [1, kv_heads, S, d]
            values.transpose(0, 1)[None],
            attn_mask=mask[None, None],
            enable_gqa=True,
        )
        return out[0].transpose(0, 1).reshape(chunk_len, -1)

    def attend_batched(
        self,
        queries: list[torch.Tensor],
        kv_pool: PagedKVPool,
        layer: int,
        page_tables: list[list[int]],
        seq_lens: list[int],
        chunk_starts: list[int],
    ) -> list[torch.Tensor]:
        """Batched decode/prefill attention (C4): per-sequence contexts.

        The CPU reference dispatches each sequence through ``attend`` — so the
        result is per-sequence IDENTICAL to calling ``attend`` one at a time. On
        GPU the FlashInfer backend replaces the loop with ONE batched kernel over
        indptr/indices arrays behind this same signature (no batch = N launches).
        """
        return [
            self.attend(queries[i], kv_pool, layer, page_tables[i], seq_lens[i], chunk_starts[i])
            for i in range(len(queries))
        ]
