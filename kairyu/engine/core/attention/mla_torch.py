"""MLA reference math (m13 D3) — the trusted oracle M15 wires into DeepSeek.

DeepSeek-V2/V3 MLA (arXiv:2405.04434), reviewed pins:
- the cache stores the per-token compressed latent ``c_kv`` (kv_lora_rank)
  concatenated with the decoupled rope key ``k_pe`` — ``k_pe`` is a SINGLE
  shared head (q_pe is per-head) stored POST-RoPE; the pool variant is a
  1-kv-head pool of width ``kv_lora_rank + qk_rope_head_dim``.
- softmax scale is ``(qk_nope_head_dim + qk_rope_head_dim) ** -0.5`` in BOTH
  forms and the oracle — the absorbed form's latent-width layout makes SDPA
  defaults silently wrong. The scale is a parameter (M15's YaRN mscale folds
  into it).
- inputs are already-projected, already-roped ``q_nope [T, H, d_nope]`` and
  ``q_pe [T, H, d_rope]``; q-side LoRA compression is M15 wiring.

Two algebraically equivalent forms, cross-checked by test:
``decompress`` (materialize per-head K/V — the prefill-ish form) and
``absorbed`` (attend in latent space — the memory-bound decode form real
serving uses).
"""

from __future__ import annotations

import torch


def mla_scale(qk_nope_head_dim: int, qk_rope_head_dim: int) -> float:
    return (qk_nope_head_dim + qk_rope_head_dim) ** -0.5


def _masked_softmax(scores: torch.Tensor, causal_offset: int) -> torch.Tensor:
    """scores [H, T, S]; query i attends keys [0, causal_offset + i]."""
    chunk_len, seq_len = scores.shape[-2], scores.shape[-1]
    positions = torch.arange(chunk_len)[:, None] + causal_offset
    mask = torch.arange(seq_len)[None, :] <= positions
    scores = scores.masked_fill(~mask[None], float("-inf"))
    return torch.softmax(scores, dim=-1)


def mla_decompress(
    q_nope: torch.Tensor,  # [T, H, d_nope]
    q_pe: torch.Tensor,  # [T, H, d_rope]
    c_kv: torch.Tensor,  # [S, kv_lora_rank]
    k_pe: torch.Tensor,  # [S, d_rope] — ONE shared head, post-RoPE
    w_uk: torch.Tensor,  # [H, kv_lora_rank, d_nope]
    w_uv: torch.Tensor,  # [H, kv_lora_rank, d_v]
    scale: float,
    causal_offset: int = 0,
) -> torch.Tensor:
    """Materialize per-head K/V from the latent, then standard attention.

    Returns [T, H, d_v].
    """
    k_nope = torch.einsum("sr,hrd->hsd", c_kv, w_uk)  # [H, S, d_nope]
    values = torch.einsum("sr,hrd->hsd", c_kv, w_uv)  # [H, S, d_v]
    # shared k_pe broadcast into every head
    scores = (
        torch.einsum("thd,hsd->hts", q_nope, k_nope)
        + torch.einsum("thd,sd->hts", q_pe, k_pe)
    ) * scale
    attention = _masked_softmax(scores, causal_offset)
    return torch.einsum("hts,hsd->thd", attention, values)


def mla_absorbed(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    c_kv: torch.Tensor,
    k_pe: torch.Tensor,
    w_uk: torch.Tensor,
    w_uv: torch.Tensor,
    scale: float,
    causal_offset: int = 0,
) -> torch.Tensor:
    """Matrix-absorption form: W_UK folds into the query, W_UV into the output;
    attention runs in latent space (MQA-shaped: the kv side has one head)."""
    q_latent = torch.einsum("thd,hrd->thr", q_nope, w_uk)  # [T, H, kv_lora_rank]
    scores = (
        torch.einsum("thr,sr->hts", q_latent, c_kv)
        + torch.einsum("thd,sd->hts", q_pe, k_pe)
    ) * scale
    attention = _masked_softmax(scores, causal_offset)
    context_latent = torch.einsum("hts,sr->thr", attention, c_kv)
    return torch.einsum("thr,hrd->thd", context_latent, w_uv)
