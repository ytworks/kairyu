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
from kairyu.engine.core.prefill import PrefillBatch
from kairyu.kernels import rms_norm_gpu
from kairyu.models.config import ModelConfig
from kairyu.models.layers import RMSNorm, apply_rope
from kairyu.models.packed_linear import DenseLinearPack, NvFp4LinearPack
from kairyu.quant.linear import LinearRole, ModelScope, make_linear


class Attention(nn.Module):
    """GQA attention over the paged pool; HF module names (m12 D2)."""

    def __init__(
        self,
        config: ModelConfig,
        backend: AttentionBackend | None = None,
        linear_factory=None,
        *,
        prefix: str = "self_attn",
        layer_index: int | None = None,
        model_scope: ModelScope = ModelScope.TARGET,
    ) -> None:
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
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.q_proj = make_linear(
            linear_factory,
            config.hidden_size,
            heads * dim,
            config.qkv_bias,
            qualified_name=f"{prefix}.q_proj",
            model_scope=model_scope,
            role=LinearRole.ATTENTION_QUERY,
            layer_index=layer_index,
            shard_dim=0,
        )
        self.k_proj = make_linear(
            linear_factory,
            config.hidden_size,
            kv_heads * dim,
            config.qkv_bias,
            qualified_name=f"{prefix}.k_proj",
            model_scope=model_scope,
            role=LinearRole.ATTENTION_KEY,
            layer_index=layer_index,
            shard_dim=0,
        )
        self.v_proj = make_linear(
            linear_factory,
            config.hidden_size,
            kv_heads * dim,
            config.qkv_bias,
            qualified_name=f"{prefix}.v_proj",
            model_scope=model_scope,
            role=LinearRole.ATTENTION_VALUE,
            layer_index=layer_index,
            shard_dim=0,
        )
        self.o_proj = make_linear(
            linear_factory,
            heads * dim,
            config.hidden_size,
            config.o_bias,
            qualified_name=f"{prefix}.o_proj",
            model_scope=model_scope,
            role=LinearRole.ATTENTION_OUTPUT,
            layer_index=layer_index,
            shard_dim=1,
        )
        if config.qk_norm:  # Qwen3: per-head RMSNorm over head_dim, BEFORE RoPE
            self.q_norm = RMSNorm(dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(dim, config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        # Preserve canonical q/k/v modules and checkpoint members while using
        # one GEMM for compatible dense or NVFP4 projections.
        self.refresh_dense_pack()

    def refresh_dense_pack(self) -> None:
        """Rebuild QKV execution views and invariant NVFP4 operands."""

        projections = (self.q_proj, self.k_proj, self.v_proj)
        packed = DenseLinearPack.create(projections)
        if packed is None:
            packed = NvFp4LinearPack.create(projections)
        object.__setattr__(
            self,
            "_qkv_pack",
            packed,
        )
        if packed is None:
            for projection in projections:
                self._prepare_nvfp4_projection(projection)
        self._prepare_nvfp4_projection(self.o_proj)
        q_norm = self.q_norm
        k_norm = self.k_norm
        object.__setattr__(
            self,
            "_joint_qk_norm_modules",
            (
                (q_norm, k_norm)
                if type(q_norm) is RMSNorm
                and type(k_norm) is RMSNorm
                and q_norm.eps == k_norm.eps
                else None
            ),
        )

    @staticmethod
    def _prepare_nvfp4_projection(projection: nn.Module) -> None:
        """Prepare a resident dense NVFP4 projection at model load/move time."""

        from kairyu.quant.linear import NvFp4Linear

        if type(projection) is not NvFp4Linear or projection.weight.device.type != "cuda":
            return
        from kairyu.kernels.quant_gemm_gpu import prepare_nvfp4_linear

        prepare_nvfp4_linear(projection, projection.weight.device)

    def _project_qkv(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packed = getattr(self, "_qkv_pack", None)
        projections = (self.q_proj, self.k_proj, self.v_proj)
        if (
            packed is not None
            and packed.matches(projections)
            and packed.is_current()
        ):
            query, keys, values = packed(hidden)
            return query, keys, values
        return tuple(projection(hidden) for projection in projections)

    def _normalize_qk(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.q_norm is None:
            return query, keys
        assert self.k_norm is not None
        # Exact modules with unobserved forwards can use the joint CUDA
        # execution path. Replacements and hooks retain normal nn.Module
        # semantics through the separate fallback below.
        cached_modules = self._joint_qk_norm_modules
        can_fuse = (
            cached_modules is not None
            and self.q_norm is cached_modules[0]
            and self.k_norm is cached_modules[1]
            and "forward" not in self.q_norm.__dict__
            and "forward" not in self.k_norm.__dict__
            and not self.q_norm._forward_hooks
            and not self.q_norm._forward_pre_hooks
            and not self.q_norm._backward_hooks
            and not self.q_norm._backward_pre_hooks
            and not self.k_norm._forward_hooks
            and not self.k_norm._forward_pre_hooks
            and not self.k_norm._backward_hooks
            and not self.k_norm._backward_pre_hooks
        )
        if can_fuse:
            joint = rms_norm_gpu.try_joint_qk_rms_norm(
                query,
                keys,
                self.q_norm.weight,
                self.k_norm.weight,
                self.q_norm.eps,
            )
            if joint is not None:
                return joint
        return self.q_norm(query), self.k_norm(keys)

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
        *,
        chunk_start: int | None = None,
        has_writable: bool | None = None,
    ) -> torch.Tensor:
        chunk_len = hidden.shape[0]
        query, keys, values = self._project_qkv(hidden)
        query = query.view(chunk_len, self.num_heads, self.head_dim)
        keys = keys.view(chunk_len, self.num_kv_heads, self.head_dim)
        values = values.view(chunk_len, self.num_kv_heads, self.head_dim)
        query, keys = self._normalize_qk(query, keys)
        query, keys = apply_rope(
            query,
            keys,
            cos,
            sin,
            positions=positions,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )
        # KV-write skip (m12 D4, BLOCKING amendment): positions below
        # num_cached_tokens already hold valid (possibly SHARED) KV — never
        # rewrite them; recomputing their Q is enough.
        if (chunk_start is None) != (has_writable is None):
            raise ValueError(
                "chunk_start and has_writable must be provided together"
            )
        if chunk_start is None:
            # Compatibility path for the m12 public contract, which accepts
            # arbitrary explicit positions rather than only a contiguous
            # scheduler chunk.
            writable = positions >= write_from
            if bool(writable.any()):
                kv_pool.write(
                    layer,
                    page_table,
                    positions[writable],
                    keys[writable],
                    values[writable],
                )
            chunk_start = int(positions[0].item())
        else:
            # Boolean indexing a CUDA tensor performs a dynamic-shape
            # ``nonzero`` and synchronizes the stream. Scheduler chunks are
            # contiguous, so host metadata identifies the same suffix as a
            # fixed-shape slice.
            write_offset = min(max(write_from - chunk_start, 0), chunk_len)
            if has_writable != (write_offset < chunk_len):
                raise ValueError(
                    "has_writable does not match chunk_start/write_from metadata"
                )
            if has_writable:
                kv_pool.write(
                    layer,
                    page_table,
                    positions[write_offset:],
                    keys[write_offset:],
                    values[write_offset:],
                )
        context = self.backend.attend(query, kv_pool, layer, page_table, seq_len, chunk_start)
        return self.o_proj(context)

    def plan_decode_tensors(
        self,
        kv_pool,
        page_tables: torch.Tensor,  # [B, P]
        seq_lens: torch.Tensor,  # [B]
        *,
        q_dtype: torch.dtype,
        replay: bool = False,
        host_seq_lens: tuple[int, ...] | None = None,
    ) -> None:
        """Run the backend's step-boundary host phase (``GraphDecodeBackend``).

        ``forward_decode_tensors`` is the captured half and may not touch the
        host, so whatever the backend needs to plan on the CPU has to happen
        here — outside any capture, once per step, over the buffers the step is
        about to attend over. The plan must describe the query
        ``forward_decode_tensors`` will build, so the head count comes from this
        module and ``q_dtype`` from the caller: it is the ACTIVATION dtype, not
        ``q_proj``'s. A quantized ``q_proj`` dequantizes to the input's dtype
        (and AWQ/GPTQ have no ``.weight`` to ask in the first place), so reading
        it off the projection would be wrong for exactly the deployments that
        most want a captured decode.
        """
        plan = getattr(self.backend, "plan_decode", None)
        if plan is None:  # a backend outside the graph contract; nothing to do
            return
        kwargs = {
            "num_qo_heads": self.num_heads,
            "q_dtype": q_dtype,
        }
        if getattr(self.backend, "supports_fast_replay_plan", False) and (
            replay or host_seq_lens is not None
        ):
            # Scheduler-owned host lengths let FlashInfer use the same fast
            # planning extension for graph replay and eager tensor decode.
            # Backends not declaring the extension retain the stock signature.
            kwargs.update(
                replay=replay,
                host_seq_lens=host_seq_lens,
            )
        plan(kv_pool, page_tables, seq_lens, **kwargs)

    def forward_decode_tensors(
        self,
        hidden: torch.Tensor,  # [B, H]
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_pool,
        layer: int,
        page_tables: torch.Tensor,  # [B, P]
        positions: torch.Tensor,  # [B]
        seq_lens: torch.Tensor,  # [B]
        write_from: torch.Tensor | None = None,  # [B]
    ) -> torch.Tensor:
        """Decode attention with no host values anywhere (m17 D1).

        Same math as ``forward_decode_batch``; the difference is that the page
        table, positions and lengths stay tensors, so a captured graph replays
        against the buffers' CURRENT contents instead of the values that happened
        to be there at capture time.
        """
        batch = hidden.shape[0]
        query, keys, values = self._project_qkv(hidden)
        query = query.view(batch, self.num_heads, self.head_dim)
        keys = keys.view(batch, self.num_kv_heads, self.head_dim)
        values = values.view(batch, self.num_kv_heads, self.head_dim)
        query, keys = self._normalize_qk(query, keys)
        query, keys = apply_rope(
            query,
            keys,
            cos,
            sin,
            positions=positions,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )
        kv_pool.write_batched(layer, page_tables, positions, keys, values, write_from=write_from)
        context = self.backend.attend_decode(query, kv_pool, layer, page_tables, seq_lens)
        return self.o_proj(context)

    def forward_prefill_batch(
        self,
        hidden: torch.Tensor,  # [sum(T_i), H]
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_pool: PagedKVPool,
        layer: int,
        batch: PrefillBatch,
    ) -> torch.Tensor:
        """Native ragged prefill over independent paged sequences.

        Projection, RoPE, KV slot mapping, and the backend plan all consume the
        same flat ScheduledChunk order.  ``build_prefill_batch`` has already
        proven that writable pages are private; ``write_ragged`` then filters
        cached positions so shared radix pages remain byte-identical.
        """
        tokens = hidden.shape[0]
        query, keys, values = self._project_qkv(hidden)
        query = query.view(tokens, self.num_heads, self.head_dim)
        keys = keys.view(tokens, self.num_kv_heads, self.head_dim)
        values = values.view(tokens, self.num_kv_heads, self.head_dim)
        query, keys = self._normalize_qk(query, keys)
        query, keys = apply_rope(
            query,
            keys,
            cos,
            sin,
            positions=batch.positions,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )
        kv_pool.write_ragged(
            layer,
            batch.page_tables,
            batch.row_ids,
            batch.positions,
            keys,
            values,
            batch.write_from,
        )
        context = self.backend.attend_prefill(
            query,
            kv_pool,
            layer,
            batch.page_lists,
            batch.seq_lens,
            batch.chunk_starts,
            batch.qo_indptr,
        )
        return self.o_proj(context)

    def forward_decode_batch(
        self,
        hidden: torch.Tensor,  # [B, H] — one new token per sequence
        cos: torch.Tensor,  # [B, head_dim] — per-sequence rotary
        sin: torch.Tensor,
        kv_pool: PagedKVPool,
        layer: int,
        page_tables: list[list[int]],
        positions: torch.Tensor,  # [B]
        seq_lens: list[int],
        write_from: list[int],
        position_values: list[int] | None = None,
    ) -> torch.Tensor:
        """Batched decode attention (C4): projections/RoPE are row-batched, each
        sequence's single-token KV is written to its OWN pages, and attention is
        per-sequence via ``attend_batched`` — output is IDENTICAL, row by row, to
        running each sequence through ``forward``."""
        batch = hidden.shape[0]
        query, keys, values = self._project_qkv(hidden)
        query = query.view(batch, self.num_heads, self.head_dim)
        keys = keys.view(batch, self.num_kv_heads, self.head_dim)
        values = values.view(batch, self.num_kv_heads, self.head_dim)
        query, keys = self._normalize_qk(query, keys)
        query, keys = apply_rope(
            query,
            keys,
            cos,
            sin,
            positions=positions,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )
        if position_values is None:
            if positions.device.type != "cpu":
                raise ValueError(
                    "GPU list-metadata decode needs host position_values; "
                    "reading positions row by row would synchronize the device"
                )
            position_values = positions.tolist()
        for i in range(batch):
            if position_values[i] >= write_from[i]:
                kv_pool.write(
                    layer, page_tables[i], positions[i : i + 1], keys[i : i + 1], values[i : i + 1]
                )
        contexts = self.backend.attend_batched(
            [query[i : i + 1] for i in range(batch)],  # each [1, heads, d]
            kv_pool,
            layer,
            page_tables,
            seq_lens,
            position_values,
        )
        return self.o_proj(torch.cat(contexts, dim=0))  # [B, heads*d] -> [B, H]
