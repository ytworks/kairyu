"""MLA attention module: DeepSeek-V3 class over the latent KV pool (m15 D2/A4).

Wires M13's verified reference (``mla_torch``, matched to 3.7e-9 against
``DeepseekV3Attention``) into a module. The pool caches the per-token
``[post-kv_a_layernorm c_kv ‖ roped k_pe]`` as ONE kv head of width
``kv_lora_rank + qk_rope_head_dim`` (the v tensor is width 0 — recorded for
M18's serde). Prefill chunks use the decompress form; decode uses the
absorbed (memory-bound) form — verified equal.

Pins (m15 §6): q split nope-first; kv_a output c_kv-first; interleaved rope
(``rope_interleave`` default True); q_a/kv_a RMSNorm eps hardcoded 1e-6 in
HF; softmax scale = qk_head_dim^-0.5 × yarn mscale_all_dim² when applicable.
"""

from __future__ import annotations

import torch
from torch import nn

from kairyu.engine.core.attention.mla_torch import mla_absorbed, mla_decompress
from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.models.config import ModelConfig
from kairyu.models.layers import RMSNorm, apply_rope_interleave, mla_softmax_scale

_HF_MLA_NORM_EPS = 1e-6  # hardcoded in DeepseekV3RMSNorm construction (A4)


class MlaAttention(nn.Module):
    def __init__(self, config: ModelConfig, linear_factory=None) -> None:
        super().__init__()
        mla = config.mla
        assert mla is not None
        make = linear_factory or (lambda i, o, b: nn.Linear(i, o, bias=b))
        heads = config.num_attention_heads
        self.num_heads = heads
        self.mla = mla
        self.scale = mla_softmax_scale(mla.qk_head_dim, config.rope_scaling)
        bias = config._attention_bias
        if mla.q_lora_rank is not None:
            self.q_a_proj = make(config.hidden_size, mla.q_lora_rank, bias)
            self.q_a_layernorm = RMSNorm(mla.q_lora_rank, _HF_MLA_NORM_EPS)
            self.q_b_proj = make(mla.q_lora_rank, heads * mla.qk_head_dim, False)
            self.q_proj = None
        else:
            self.q_proj = make(config.hidden_size, heads * mla.qk_head_dim, False)
        self.kv_a_proj_with_mqa = make(
            config.hidden_size, mla.kv_lora_rank + mla.qk_rope_head_dim, bias
        )
        self.kv_a_layernorm = RMSNorm(mla.kv_lora_rank, _HF_MLA_NORM_EPS)
        self.kv_b_proj = make(
            mla.kv_lora_rank, heads * (mla.qk_nope_head_dim + mla.v_head_dim), False
        )
        self.o_proj = make(heads * mla.v_head_dim, config.hidden_size, bias)

    def _uk_uv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """kv_b_proj.weight [(H*(d_nope+d_v)), r] -> w_uk [H,r,d_nope], w_uv [H,r,d_v]."""
        mla = self.mla
        weight = self.kv_b_proj.weight.view(
            self.num_heads, mla.qk_nope_head_dim + mla.v_head_dim, mla.kv_lora_rank
        )
        w_uk = weight[:, : mla.qk_nope_head_dim, :].transpose(1, 2)
        w_uv = weight[:, mla.qk_nope_head_dim :, :].transpose(1, 2)
        return w_uk, w_uv

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
        mla = self.mla
        chunk_len = hidden.shape[0]
        if self.q_proj is not None:
            q = self.q_proj(hidden)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden)))
        q = q.view(chunk_len, self.num_heads, mla.qk_head_dim)
        q_nope, q_pe = q.split([mla.qk_nope_head_dim, mla.qk_rope_head_dim], dim=-1)

        kv_a = self.kv_a_proj_with_mqa(hidden)
        c_kv, k_pe = kv_a.split([mla.kv_lora_rank, mla.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_a_layernorm(c_kv)

        q_pe = apply_rope_interleave(q_pe, cos, sin)
        k_pe = apply_rope_interleave(k_pe[:, None, :], cos, sin)[:, 0]

        # cache [c_kv ‖ roped k_pe] as one kv head; skip shared cached slots
        latent = torch.cat([c_kv, k_pe], dim=-1)[:, None, :]
        writable = positions >= write_from
        if bool(writable.any()):
            kv_pool.write(
                layer,
                page_table,
                positions[writable],
                latent[writable],
                latent[writable][:, :, :0],  # v tensor has width 0 for MLA
            )
        cached, _ = kv_pool.gather(layer, page_table, seq_len)
        cached = cached[:, 0, :]  # [S, r + d_rope]
        c_all, kpe_all = cached.split([mla.kv_lora_rank, mla.qk_rope_head_dim], dim=-1)

        w_uk, w_uv = self._uk_uv()
        chunk_start = int(positions[0].item())
        form = mla_absorbed if chunk_len == 1 else mla_decompress
        context = form(
            q_nope, q_pe, c_all, kpe_all, w_uk, w_uv, self.scale,
            causal_offset=chunk_start,
        )
        return self.o_proj(context.reshape(chunk_len, self.num_heads * mla.v_head_dim))
