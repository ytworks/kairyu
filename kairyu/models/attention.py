"""Paged GQA attention module (m12 D2, m13 D2).

Score computation is delegated to the ``AttentionBackend`` seam (m13):
backends are plain objects, NEVER nn.Module (a stateful backend's buffers
must not register into state_dict), and the same backend INSTANCE is shared
across all layers (FlashInfer's workspace/plan cache depends on it).
"""

from __future__ import annotations

import torch
from torch import nn

from kairyu.engine.core.attention import AttentionBackend, TorchAttentionBackend
from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.models.config import ModelConfig
from kairyu.models.layers import RMSNorm, apply_rope


class Attention(nn.Module):
    """GQA attention over the paged pool; HF module names (m12 D2)."""

    def __init__(self, config: ModelConfig, backend: AttentionBackend | None = None) -> None:
        super().__init__()
        # plain attribute, not a submodule; None default avoids a shared
        # import-time instance (m13 review C2)
        self.backend = backend or TorchAttentionBackend()
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
        context = self.backend.attend(
            query, kv_pool, layer, page_table, seq_len, chunk_start
        )
        return self.o_proj(context)
