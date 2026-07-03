"""Paged GQA attention (m12 D2).

``paged_attention`` is the ONE function M13 extracts into the
``AttentionBackend`` seam — it takes the pool + page table (not pre-gathered
K/V). The chunk mask is rectangular: a query at absolute position
``chunk_start + i`` attends keys ``[0, chunk_start + i]`` — SDPA's
``is_causal=True`` is top-left aligned and WRONG for chunks over a cached
prefix (reviewed, measured).
"""

from __future__ import annotations

import torch
from torch import nn

from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.models.config import ModelConfig
from kairyu.models.layers import RMSNorm, apply_rope


def paged_attention(
    query: torch.Tensor,
    kv_pool: PagedKVPool,
    layer: int,
    page_table: list[int],
    seq_len: int,
    chunk_start: int,
) -> torch.Tensor:
    """query: [T, num_heads, head_dim] -> context [T, num_heads * head_dim]."""
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


class Attention(nn.Module):
    """GQA attention over the paged pool; HF module names (m12 D2)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        heads, kv_heads, dim = (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.num_heads = heads
        self.num_kv_heads = kv_heads
        self.head_dim = dim
        self.q_proj = nn.Linear(config.hidden_size, heads * dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_heads * dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_heads * dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(heads * dim, config.hidden_size, bias=config.o_bias)
        if config.qk_norm:  # Qwen3: per-head RMSNorm over head_dim, BEFORE RoPE
            self.q_norm = RMSNorm(dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(dim, config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_pool: PagedKVPool,
        layer: int,
        page_table: list[int],
        positions: torch.Tensor,
        seq_len: int,
        write_from: int,
    ) -> torch.Tensor:
        chunk_len = hidden.shape[0]
        query = self.q_proj(hidden).view(chunk_len, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden).view(chunk_len, self.num_kv_heads, self.head_dim)
        values = self.v_proj(hidden).view(chunk_len, self.num_kv_heads, self.head_dim)
        if self.q_norm is not None:
            query = self.q_norm(query)
            keys = self.k_norm(keys)
        query, keys = apply_rope(query, keys, cos, sin)
        # KV-write skip (m12 D4, BLOCKING amendment): positions below
        # num_cached_tokens already hold valid (possibly SHARED) KV — never
        # rewrite them; recomputing their Q is enough.
        writable = positions >= write_from
        if bool(writable.any()):
            kv_pool.write(
                layer, page_table, positions[writable], keys[writable], values[writable]
            )
        chunk_start = int(positions[0].item())
        context = paged_attention(query, kv_pool, layer, page_table, seq_len, chunk_start)
        return self.o_proj(context)
