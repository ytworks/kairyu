"""DenseDecoder: config-switched Llama-3.x / Qwen2 / Qwen3 decoder (m12 D2).

Module tree mirrors HF names exactly (``model.layers.N.self_attn.q_proj`` …)
so loading is a 1:1 name map — ``load_state_dict`` from an HF checkpoint (or
a transformers model's state_dict) works with zero renaming tables.
"""

from __future__ import annotations

import torch
from torch import nn

from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.engine.core.prefill import PrefillBatch
from kairyu.models.attention import Attention
from kairyu.models.config import ModelConfig
from kairyu.models.layers import RMSNorm, RotaryEmbedding
from kairyu.models.mla import MlaAttention
from kairyu.models.moe import build_mlp
from kairyu.quant.linear import LinearRole, ModelScope, make_linear


def _refresh_dense_packs_after_load(
    module: nn.Module,
    _incompatible_keys,
) -> None:
    """Root post-load hook: ``assign=True`` replaces canonical parameters."""

    assert isinstance(module, DenseDecoder)
    module.refresh_dense_packs()


class DecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_index: int = 0,
        backend=None,
        linear_factory=None,
        *,
        prefix: str | None = None,
        model_scope: ModelScope = ModelScope.TARGET,
    ) -> None:
        super().__init__()
        prefix = prefix or f"model.layers.{layer_index}"
        if config.is_mla:
            self.self_attn = MlaAttention(
                config,
                linear_factory=linear_factory,
                prefix=f"{prefix}.self_attn",
                layer_index=layer_index,
                model_scope=model_scope,
            )
        else:
            self.self_attn = Attention(
                config,
                backend=backend,
                linear_factory=linear_factory,
                prefix=f"{prefix}.self_attn",
                layer_index=layer_index,
                model_scope=model_scope,
            )
        self.mlp = build_mlp(
            config,
            layer_index,
            linear_factory=linear_factory,
            prefix=f"{prefix}.mlp",
            model_scope=model_scope,
        )
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

    def plan_decode_tensors(
        self,
        kv_pool,
        page_tables,
        seq_lens,
        *,
        q_dtype,
        replay: bool = False,
        host_seq_lens: tuple[int, ...] | None = None,
    ):
        """Step-boundary host phase for this layer's attention backend."""
        self.self_attn.plan_decode_tensors(
            kv_pool,
            page_tables,
            seq_lens,
            q_dtype=q_dtype,
            replay=replay,
            host_seq_lens=host_seq_lens,
        )

    def forward_decode_tensors(
        self,
        hidden,
        cos,
        sin,
        kv_pool,
        layer,
        page_tables,
        positions,
        seq_lens,
        write_from=None,
    ):
        """Tensor-only decode layer (m17 D1): no Python page lists."""
        hidden = hidden + self.self_attn.forward_decode_tensors(
            self.input_layernorm(hidden),
            cos,
            sin,
            kv_pool,
            layer,
            page_tables,
            positions,
            seq_lens,
            write_from,
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))

    def forward_prefill_batch(
        self,
        hidden,
        cos,
        sin,
        kv_pool,
        layer,
        batch: PrefillBatch,
    ):
        """Residual layer over the flat ragged-prefill token axis."""
        hidden = hidden + self.self_attn.forward_prefill_batch(
            self.input_layernorm(hidden),
            cos,
            sin,
            kv_pool,
            layer,
            batch,
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))

    def forward_decode_batch(
        self,
        hidden,
        cos,
        sin,
        kv_pool,
        layer,
        page_tables,
        positions,
        seq_lens,
        write_from,
        position_values=None,
    ):
        """Batched decode (C4): row-batched residual around the batched attention."""
        hidden = hidden + self.self_attn.forward_decode_batch(
            self.input_layernorm(hidden),
            cos,
            sin,
            kv_pool,
            layer,
            page_tables,
            positions,
            seq_lens,
            write_from,
            position_values,
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class _Backbone(nn.Module):
    """The HF ``model.*`` subtree."""

    def __init__(
        self,
        config: ModelConfig,
        backend=None,
        linear_factory=None,
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # Optional backend preflight: experimental kernels must reject model
        # shapes they cannot serve before weights are loaded or a request is
        # admitted.  Stable backends without this hook retain their existing
        # construction path.
        validate_model_config = getattr(backend, "validate_model_config", None)
        if callable(validate_model_config):
            validate_model_config(config, dtype=dtype)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # ONE backend instance shared across layers (m13: FlashInfer workspace
        # and plan cache are per-instance — sharing is load-bearing)
        self.layers = nn.ModuleList(
            DecoderLayer(
                config,
                layer_index=index,
                backend=backend,
                linear_factory=linear_factory,
                prefix=f"model.layers.{index}",
            )
            for index in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)


class DenseDecoder(nn.Module):
    """Paged incremental decoder; covers the m12 dense family via config."""

    # The runner may pass replay-only scheduler metadata through this model
    # seam. Models implementing only the original tensor-decode contract omit
    # this capability and retain stock planning on every replay.
    supports_fast_replay_plan = True

    def __init__(
        self,
        config: ModelConfig,
        attention_backend=None,
        linear_factory=None,
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = _Backbone(
            config,
            backend=attention_backend,
            linear_factory=linear_factory,
            dtype=dtype,
        )
        self.lm_head = make_linear(
            linear_factory,
            config.hidden_size,
            config.vocab_size,
            False,
            qualified_name="lm_head",
            role=LinearRole.OUTPUT_HEAD,
            # A tied checkpoint has no independent packed head payload.  This
            # is a hard format constraint; untied heads remain a policy choice.
            allow_quantization=not config.tie_word_embeddings,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        # Packing at construction gives direct models the optimized path;
        # assign-load and device/dtype conversion rebuild it below.
        self.refresh_dense_packs()
        self.register_load_state_dict_post_hook(_refresh_dense_packs_after_load)

    def refresh_dense_packs(self) -> None:
        """Pack compatible dense QKV projections in every layer."""

        for layer in self.model.layers:
            refresh_attention = getattr(layer.self_attn, "refresh_dense_pack", None)
            if callable(refresh_attention):
                refresh_attention()

    def _apply(self, fn, recurse: bool = True):
        """Keep packed storage coherent across ``to()``, ``cuda()``, and casts."""

        result = super()._apply(fn, recurse=recurse)
        self.refresh_dense_packs()
        return result

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
    def forward_tokens_with_aux(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_pool: PagedKVPool,
        page_table: list[int],
        seq_len: int,
        aux_layer_ids: tuple[int, ...],
        write_from: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one target pass and retain selected pre-final-norm residuals.

        ``aux_layer_ids`` names decoder outputs using zero-based layer ids.
        The returned auxiliary tensor concatenates those residual-added layer
        outputs in ascending order along the hidden dimension.  This is the
        EAGLE-3 target feature contract: capture happens in the same pass that
        writes target KV, and the ordinary post-final-norm output is unchanged.
        """

        if not aux_layer_ids:
            raise ValueError("aux_layer_ids must contain at least one layer")
        if tuple(sorted(set(aux_layer_ids))) != aux_layer_ids:
            raise ValueError("aux_layer_ids must be unique and strictly increasing")
        if aux_layer_ids[0] < 0 or aux_layer_ids[-1] >= len(self.model.layers):
            raise ValueError(
                "aux_layer_ids must be within "
                f"[0, {len(self.model.layers) - 1}], got {aux_layer_ids}"
            )

        hidden = self.model.embed_tokens(token_ids)
        cos, sin = self.model.rotary_emb(positions)
        captures: list[torch.Tensor] = []
        capture_ids = frozenset(aux_layer_ids)
        for index, layer in enumerate(self.model.layers):
            hidden = layer(
                hidden,
                cos,
                sin,
                kv_pool,
                index,
                page_table,
                positions,
                seq_len,
                write_from,
            )
            if index in capture_ids:
                captures.append(hidden)
        return self.model.norm(hidden), torch.cat(captures, dim=-1)

    @torch.no_grad()
    def forward_decode_batch(
        self,
        token_ids: torch.Tensor,  # [B] one new token per sequence
        positions: torch.Tensor,  # [B] each sequence's decode position
        kv_pool: PagedKVPool,
        page_tables: list[list[int]],
        seq_lens: list[int],
        write_from: list[int],
        position_values: list[int] | None = None,
    ) -> torch.Tensor:
        """Batched single-token decode over B sequences (C4): ONE forward pass
        instead of B. Row i of the output equals ``forward_tokens`` for sequence
        i's decode step — batching is a performance transform, not a numeric one
        (pinned by test_batched_decode_equals_sequential)."""
        hidden = self.model.embed_tokens(token_ids)  # [B, H]
        cos, sin = self.model.rotary_emb(positions)  # [B, head_dim]
        for index, layer in enumerate(self.model.layers):
            hidden = layer.forward_decode_batch(
                hidden,
                cos,
                sin,
                kv_pool,
                index,
                page_tables,
                positions,
                seq_lens,
                write_from,
                position_values,
            )
        return self.model.norm(hidden)  # [B, H]

    @torch.no_grad()
    def forward_prefill_batch(
        self,
        batch: PrefillBatch,
        kv_pool: PagedKVPool,
    ) -> torch.Tensor:
        """One model chain over a ragged cross-request prefill batch."""
        hidden = self.model.embed_tokens(batch.token_ids)
        cos, sin = self.model.rotary_emb(batch.positions)
        for index, layer in enumerate(self.model.layers):
            hidden = layer.forward_prefill_batch(
                hidden,
                cos,
                sin,
                kv_pool,
                index,
                batch,
            )
        return self.model.norm(hidden)

    @torch.no_grad()
    def forward_decode_tensors(
        self,
        token_ids: torch.Tensor,  # [B]
        positions: torch.Tensor,  # [B]
        kv_pool: PagedKVPool,
        page_tables: torch.Tensor,  # [B, P]
        seq_lens: torch.Tensor,  # [B]
        write_from: torch.Tensor | None = None,  # [B]
    ) -> torch.Tensor:
        """Batched decode with every input a tensor — the CUDA-graph-capturable
        form of ``forward_decode_batch`` (m17 D1). Numerically identical; pinned
        by test_tensor_decode_matches_list_decode."""
        hidden = self.model.embed_tokens(token_ids)
        cos, sin = self.model.rotary_emb(positions)
        for index, layer in enumerate(self.model.layers):
            hidden = layer.forward_decode_tensors(
                hidden,
                cos,
                sin,
                kv_pool,
                index,
                page_tables,
                positions,
                seq_lens,
                write_from,
            )
        return self.model.norm(hidden)

    @torch.no_grad()
    def plan_decode_tensors(
        self,
        kv_pool: PagedKVPool,
        page_tables: torch.Tensor,  # [B, P]
        seq_lens: torch.Tensor,  # [B]
        *,
        replay: bool = False,
        host_seq_lens: tuple[int, ...] | None = None,
    ) -> None:
        """Host phase of ONE tensor decode step (``GraphDecodeBackend``).

        The model-level end of the hook the step executor calls before capture
        and before every replay. Backends are shared across layers BY DESIGN
        (m13: one FlashInfer instance owns the workspace and the plan), and the
        plan depends only on the page tables, the lengths and the head/dtype
        metadata — never on which layer is running. So planning is per BACKEND
        INSTANCE, not per layer: calling it once per layer would re-do the whole
        host schedule ``num_hidden_layers`` times a step, which is precisely the
        per-layer host sync the tensor decode contract exists to remove.
        """
        # the dtype the plan must describe is the one the QUERY will have, and
        # the query is a projection of `hidden` — which starts here, at the
        # embedding, and keeps this dtype through every layer
        q_dtype = self.model.embed_tokens.weight.dtype
        planned: set[int] = set()
        for layer in self.model.layers:
            attention = getattr(layer, "self_attn", None)
            backend = getattr(attention, "backend", None)
            if backend is None or id(backend) in planned:
                continue
            planned.add(id(backend))
            layer.plan_decode_tensors(
                kv_pool,
                page_tables,
                seq_lens,
                q_dtype=q_dtype,
                replay=replay,
                host_seq_lens=host_seq_lens,
            )

    @torch.no_grad()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)
