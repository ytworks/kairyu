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


def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_inv_freq_and_factor(
    dim: int, base: float, scaling: RopeScaling
) -> tuple[torch.Tensor, float]:
    """HF ``_compute_yarn_parameters`` (m15 A5): interp/extrapolation ramp +
    the cos/sin attention factor (1.0 for real DeepSeek-V3 configs)."""
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
    inv_extra = 1.0 / pos_freqs
    inv_inter = 1.0 / (scaling.factor * pos_freqs)
    original_max = scaling.original_max_position_embeddings

    def correction_dim(num_rotations: float) -> float:
        return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = max(math.floor(correction_dim(scaling.beta_fast)), 0)
    high = min(math.ceil(correction_dim(scaling.beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2).float() - low) / (high - low)).clamp(0, 1)
    extrapolation_factor = 1.0 - ramp
    inv_freq = inv_inter * (1.0 - extrapolation_factor) + inv_extra * extrapolation_factor
    if scaling.mscale is not None and scaling.mscale_all_dim is not None:
        factor = _yarn_get_mscale(scaling.factor, scaling.mscale) / _yarn_get_mscale(
            scaling.factor, scaling.mscale_all_dim
        )
    else:
        factor = _yarn_get_mscale(scaling.factor)
    return inv_freq, factor


def mla_softmax_scale(qk_head_dim: int, scaling: RopeScaling | None) -> float:
    """DeepSeek attention scale (m15 A5): base qk_head_dim^-0.5; yarn configs
    with mscale_all_dim multiply by yarn_get_mscale(factor, mscale_all_dim)^2."""
    scale = qk_head_dim**-0.5
    if scaling is not None and scaling.kind != "default" and scaling.mscale_all_dim:
        scale *= _yarn_get_mscale(scaling.factor, scaling.mscale_all_dim) ** 2
    return scale


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
    """cos/sin computed once per forward at model level, fp32 (m12 D2).

    ``attention_scaling`` multiplies cos AND sin (HF convention; 1.0 for
    default/llama3 and for real DeepSeek-V3 yarn configs — m15 A5)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.rope_dim  # MLA ropes only the decoupled qk_rope dims
        self.attention_scaling = 1.0
        scaling = config.rope_scaling
        if scaling is not None and scaling.kind == "yarn":
            inv_freq, self.attention_scaling = _yarn_inv_freq_and_factor(
                dim, config.rope_theta, scaling
            )
        else:
            inv_freq = 1.0 / (
                config.rope_theta
                ** (torch.arange(0, dim, 2, dtype=torch.int64).to(torch.float32) / dim)
            )
            if scaling is not None and scaling.kind == "llama3":
                inv_freq = _llama3_inv_freq(inv_freq, scaling)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling


def apply_rope_interleave(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """DeepSeek interleaved rope (m15 A2): even/odd pairs, cos/sin truncated
    to d/2; output is [rotated_evens ‖ rotated_odds] (NOT re-interleaved)."""
    half = x.shape[-1] // 2
    cos = cos[:, None, :half].to(x.dtype)
    sin = sin[:, None, :half].to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.cat([even * cos - odd * sin, odd * cos + even * sin], dim=-1)


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
