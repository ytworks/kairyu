"""Shared dense-decoder layers: RMSNorm, RoPE, SwiGLU (m12 D2).

Numerics mirror HF transformers exactly (reviewed, empirically verified):
``rotate_half`` half-split convention, cos/sin computed in fp32 from
``arange(0, dim, 2, int64)/dim`` inverse frequencies, llama3 rope scaling per
``_compute_llama3_parameters`` (whose ``attention_scaling`` is 1.0 — omitted).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from kairyu.models.config import ModelConfig, RopeScaling


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        hidden = hidden.to(torch.float32)
        variance = hidden.pow(2).mean(-1, keepdim=True)
        hidden = hidden * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden.to(dtype)).to(dtype)


def _llama3_inv_freq(inv_freq: torch.Tensor, scaling: RopeScaling) -> torch.Tensor:
    """HF ``_compute_llama3_parameters`` (attention_factor is 1.0 for llama3)."""
    low_freq_wavelen = scaling.original_max_position_embeddings / scaling.low_freq_factor
    high_freq_wavelen = scaling.original_max_position_embeddings / scaling.high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / scaling.factor, inv_freq)
    smooth = (
        scaling.original_max_position_embeddings / wavelen - scaling.low_freq_factor
    ) / (scaling.high_freq_factor - scaling.low_freq_factor)
    smoothed = (1 - smooth) * scaled / scaling.factor + smooth * scaled
    is_medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    return torch.where(is_medium, smoothed, scaled)


class RotaryEmbedding(nn.Module):
    """cos/sin computed once per forward at model level, fp32 (m12 D2)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.head_dim
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.int64).to(torch.float32) / dim)
        )
        if config.rope_scaling is not None:
            inv_freq = _llama3_inv_freq(inv_freq, config.rope_scaling)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """q/k: [T, heads, head_dim]; cos/sin: [T, head_dim] (broadcast over heads)."""
    cos = cos[:, None, :].to(q.dtype)
    sin = sin[:, None, :].to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class SwiGluMlp(nn.Module):
    def __init__(self, config: ModelConfig, linear_factory=None) -> None:
        super().__init__()
        make = linear_factory or (lambda i, o, b: nn.Linear(i, o, bias=b))
        self.gate_proj = make(config.hidden_size, config.intermediate_size, False)
        self.up_proj = make(config.hidden_size, config.intermediate_size, False)
        self.down_proj = make(config.intermediate_size, config.hidden_size, False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden))
