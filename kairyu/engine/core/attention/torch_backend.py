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
    device_agnostic = True
    # ``attend_batched`` below is a correctness fallback around per-row SDPA.
    # Keep the runner on its sequential path rather than claiming launch
    # reduction that this backend does not provide.
    supports_batched_prefill = False

    #: ``GraphDecodeBackend`` (m17 D1): ``attend_decode`` below is pure tensor
    #: algebra — ``gather_batched``, an ``arange`` compare and SDPA. Nothing
    #: reads a device value into Python, so a capture of it is sound.
    supports_graph_capture = True

    def plan_decode(
        self,
        kv_pool: PagedKVPool,
        page_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        num_qo_heads: int,
        q_dtype: torch.dtype,
    ) -> None:
        """No host phase: nothing to plan (``GraphDecodeBackend``).

        ``attend_decode`` reads the page-table and length BUFFERS at run time,
        so there is no schedule to refresh between replays. Implemented anyway
        so the executor's step-boundary hook is unconditionally callable.
        """
        return None

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
        # build index tensors on the query's device so SDPA does not trip on a
        # cpu/cuda mismatch when real GPU tensors flow through this backend
        device = query.device
        positions = torch.arange(chunk_len, device=device)[:, None] + chunk_start
        mask = torch.arange(seq_len, device=device)[None, :] <= positions  # [T, S] causal
        out = nn.functional.scaled_dot_product_attention(
            query.transpose(0, 1)[None],  # [1, heads, T, d]
            keys.transpose(0, 1)[None],  # [1, kv_heads, S, d]
            values.transpose(0, 1)[None],
            attn_mask=mask[None, None],
            enable_gqa=True,
        )
        return out[0].transpose(0, 1).reshape(chunk_len, -1)

    def attend_decode(
        self,
        query: torch.Tensor,  # [B, heads, head_dim] — one new token per sequence
        kv_pool: PagedKVPool,
        layer: int,
        page_tables: torch.Tensor,  # [B, P] int
        seq_lens: torch.Tensor,  # [B] int — tokens valid so far, this one included
    ) -> torch.Tensor:
        """Decode attention with NO host values (m17 D1 static-buffer rule).

        `attend_batched` takes Python page lists and int lengths, so a captured
        graph would replay against whatever those were AT CAPTURE TIME. Here the
        page table and lengths are tensors: the same kernels read wherever the
        buffers currently point, which is what makes capture/replay correct.

        Every page is gathered and the tail masked, rather than slicing to
        `seq_len` — a data-dependent shape cannot be captured.
        """
        keys, values = kv_pool.gather_batched(layer, page_tables)  # [B, S, KVH, D]
        span = keys.shape[1]
        valid = (
            torch.arange(span, device=query.device)[None, :] < seq_lens[:, None]
        )  # [B, S]
        context = nn.functional.scaled_dot_product_attention(
            query.unsqueeze(2),  # [B, heads, 1, d]
            keys.transpose(1, 2),  # [B, kv_heads, S, d]
            values.transpose(1, 2),
            attn_mask=valid[:, None, None, :],
            enable_gqa=True,
        )
        return context.squeeze(2).reshape(query.shape[0], -1)

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
