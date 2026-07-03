"""Sparse MoE blocks: Qwen3-MoE and DeepSeek-V3 routing (m15 D1, A6/A8).

Token-loop reference implementations — M16's EP dispatch/combine replaces the
loop, never the math. Expert projections build via the m14 ``linear_factory``
so quantized experts come free.
"""

from __future__ import annotations

import torch
from torch import nn

from kairyu.models.config import ModelConfig
from kairyu.models.layers import SwiGluMlp


class _ExpertMlp(nn.Module):
    """One expert: SwiGLU at moe_intermediate_size, HF names."""

    def __init__(self, config: ModelConfig, intermediate_size: int, linear_factory=None):
        super().__init__()
        make = linear_factory or (lambda i, o, b: nn.Linear(i, o, bias=b))
        self.gate_proj = make(config.hidden_size, intermediate_size, False)
        self.up_proj = make(config.hidden_size, intermediate_size, False)
        self.down_proj = make(intermediate_size, config.hidden_size, False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
        )


def _mix_experts(
    hidden: torch.Tensor,
    experts: nn.ModuleList,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Reference combine: for each selected expert, run its tokens and add."""
    out = torch.zeros_like(hidden)
    for expert_id in topk_indices.unique():
        token_mask, slot = (topk_indices == expert_id).nonzero(as_tuple=True)
        expert_out = experts[int(expert_id)](hidden[token_mask])
        out.index_add_(
            0, token_mask, expert_out * topk_weights[token_mask, slot][:, None]
        )
    return out


class Qwen3MoeSparseBlock(nn.Module):
    """Softmax top-k routing (A8: fp32 softmax BEFORE top-k; renorm no eps)."""

    def __init__(self, config: ModelConfig, linear_factory=None) -> None:
        super().__init__()
        moe = config.moe
        assert moe is not None
        self.top_k = moe.num_experts_per_tok
        self.norm_topk_prob = moe.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, moe.num_experts, bias=False)
        self.experts = nn.ModuleList(
            _ExpertMlp(config, moe.moe_intermediate_size, linear_factory)
            for _ in range(moe.num_experts)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.gate(hidden)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = probs.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden.dtype)
        return _mix_experts(hidden, self.experts, topk_indices, topk_weights)


class DeepseekV3MoeBlock(nn.Module):
    """Sigmoid + correction-bias grouped routing (A6, matched exactly vs HF).

    The correction bias affects SELECTION only; mixing weights are the
    uncorrected sigmoid scores. Group score = sum of the top-2 corrected
    scores per group (the 2 is hardcoded in HF). routed_scaling applies to
    routed outputs only; shared experts add unscaled.
    """

    def __init__(self, config: ModelConfig, linear_factory=None) -> None:
        super().__init__()
        moe = config.moe
        assert moe is not None and moe.n_group and moe.topk_group
        self.top_k = moe.num_experts_per_tok
        self.n_group = moe.n_group
        self.topk_group = moe.topk_group
        self.norm_topk_prob = moe.norm_topk_prob
        self.routed_scaling_factor = moe.routed_scaling_factor
        self.gate = nn.Linear(config.hidden_size, moe.num_experts, bias=False)
        self.gate.register_buffer(
            "e_score_correction_bias", torch.zeros(moe.num_experts)
        )
        self.experts = nn.ModuleList(
            _ExpertMlp(config, moe.moe_intermediate_size, linear_factory)
            for _ in range(moe.num_experts)
        )
        if moe.n_shared_experts:
            self.shared_experts = _ExpertMlp(
                config, moe.moe_intermediate_size * moe.n_shared_experts, linear_factory
            )
        else:
            self.shared_experts = None

    def _route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # routing is entirely fp32 (A6)
        logits = nn.functional.linear(
            hidden.to(torch.float32), self.gate.weight.to(torch.float32)
        )
        scores = logits.sigmoid()
        corrected = scores + self.gate.e_score_correction_bias
        tokens = corrected.shape[0]
        grouped = corrected.view(tokens, self.n_group, -1)
        group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)  # top-2 hardcoded
        group_keep = group_scores.topk(self.topk_group, dim=-1).indices
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_keep, 1.0)
        masked = corrected.masked_fill(
            (group_mask[:, :, None].expand_as(grouped) == 0).reshape(tokens, -1),
            float("-inf"),
        )
        topk_indices = masked.topk(self.top_k, dim=-1).indices
        topk_weights = scores.gather(1, topk_indices)  # UNCORRECTED scores
        if self.norm_topk_prob:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights.to(hidden.dtype)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        topk_indices, topk_weights = self._route(hidden)
        out = _mix_experts(hidden, self.experts, topk_indices, topk_weights)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden)
        return out


def build_mlp(config: ModelConfig, layer_index: int, linear_factory=None) -> nn.Module:
    """Per-layer mlp choice: dense SwiGLU or the architecture's sparse block."""
    moe = config.moe
    if moe is None or not moe.is_sparse_layer(layer_index):
        return SwiGluMlp(config, linear_factory=linear_factory)
    if config.architecture == "DeepseekV3ForCausalLM":
        return DeepseekV3MoeBlock(config, linear_factory=linear_factory)
    return Qwen3MoeSparseBlock(config, linear_factory=linear_factory)
