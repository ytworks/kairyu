"""DenseDecoder: config-switched Llama-3.x / Qwen2 / Qwen3 decoder (m12 D2).

Module tree mirrors HF names exactly (``model.layers.N.self_attn.q_proj`` …)
so loading is a 1:1 name map — ``load_state_dict`` from an HF checkpoint (or
a transformers model's state_dict) works with zero renaming tables.
"""

from __future__ import annotations

import torch
from torch import nn

from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.models.attention import Attention
from kairyu.models.config import ModelConfig
from kairyu.models.layers import RMSNorm, RotaryEmbedding, SwiGluMlp


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, backend=None, linear_factory=None) -> None:
        super().__init__()
        self.self_attn = Attention(config, backend=backend, linear_factory=linear_factory)
        self.mlp = SwiGluMlp(config, linear_factory=linear_factory)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

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
        hidden = hidden + self.self_attn(
            self.input_layernorm(hidden),
            cos,
            sin,
            kv_pool,
            layer,
            page_table,
            positions,
            seq_len,
            write_from,
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class _Backbone(nn.Module):
    """The HF ``model.*`` subtree."""

    def __init__(self, config: ModelConfig, backend=None, linear_factory=None) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # ONE backend instance shared across layers (m13: FlashInfer workspace
        # and plan cache are per-instance — sharing is load-bearing)
        self.layers = nn.ModuleList(
            DecoderLayer(config, backend=backend, linear_factory=linear_factory)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)


class DenseDecoder(nn.Module):
    """Paged incremental decoder; covers the m12 dense family via config."""

    def __init__(self, config: ModelConfig, attention_backend=None, linear_factory=None) -> None:
        super().__init__()
        self.config = config
        # projections only — embeddings, norms and lm_head stay unquantized
        self.model = _Backbone(config, backend=attention_backend, linear_factory=linear_factory)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @torch.no_grad()
    def forward_tokens(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_pool: PagedKVPool,
        page_table: list[int],
        seq_len: int,
        write_from: int = 0,
    ) -> torch.Tensor:
        """Write this chunk's KV (skipping cached positions), attend, and return
        post-final-norm hidden states for ALL chunk positions (M17's tap)."""
        hidden = self.model.embed_tokens(token_ids)
        cos, sin = self.model.rotary_emb(positions)  # once per forward, fp32
        for index, layer in enumerate(self.model.layers):
            hidden = layer(
                hidden, cos, sin, kv_pool, index, page_table, positions, seq_len, write_from
            )
        return self.model.norm(hidden)

    @torch.no_grad()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)
