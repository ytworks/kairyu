"""EAGLE-3 draft head (m17 D4, structure per review A2 — verified against
vLLM llama_eagle3, SpecForge llama3_eagle, and a live SpecForge safetensors
header).

Corrected pins:
- ``lm_head`` is TRAINED over a reduced draft vocab (not target-tied);
  ``d2t`` is an int64 OFFSET map: ``target_id = draft_id + d2t[draft_id]``
  (``t2d`` exists in checkpoints but is skippable at inference).
- The single decoder layer ("midlayer") takes ``cat([input_layernorm(embeds),
  hidden_norm(hidden)])`` — q/k/v in_features are 2H; the residual is the
  PRE-norm hidden.
- ``fc`` is [H, 3H], applied ONCE per verify cycle to the target's three aux
  hidden states (residual-added, pre-final-norm); subsequent draft steps feed
  back the midlayer's own hidden.
- ``embed_tokens`` is absent from SpecForge checkpoints (aliased from the
  target model); the loader handles both presence and absence.

CPU reference runs the midlayer densely per call (the draft-KV paging is a
deploy-day optimization; dense recompute sidesteps rejection bookkeeping).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from kairyu.models.layers import RMSNorm


@dataclass(frozen=True)
class EagleConfig:
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    draft_vocab_size: int
    rms_norm_eps: float = 1e-5


class _EagleMidLayer(nn.Module):
    """The 2H-input decoder layer (checkpoint prefix ``midlayer.``)."""

    def __init__(self, config: EagleConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = hidden // config.num_attention_heads
        self.input_layernorm = RMSNorm(hidden, config.rms_norm_eps)
        self.hidden_norm = RMSNorm(hidden, config.rms_norm_eps)
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(2 * hidden, hidden, bias=False),
                "k_proj": nn.Linear(2 * hidden, hidden, bias=False),
                "v_proj": nn.Linear(2 * hidden, hidden, bias=False),
                "o_proj": nn.Linear(hidden, hidden, bias=False),
            }
        )
        self.post_attention_layernorm = RMSNorm(hidden, config.rms_norm_eps)
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": nn.Linear(hidden, config.intermediate_size, bias=False),
                "up_proj": nn.Linear(hidden, config.intermediate_size, bias=False),
                "down_proj": nn.Linear(config.intermediate_size, hidden, bias=False),
            }
        )

    def forward(self, embeds: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """embeds/hidden: [T, H]; dense causal attention over the T positions."""
        fused = torch.cat([self.input_layernorm(embeds), self.hidden_norm(hidden)], -1)
        chunk_len = fused.shape[0]

        def heads(projection: torch.Tensor) -> torch.Tensor:
            return projection.view(chunk_len, self.num_heads, self.head_dim).transpose(0, 1)

        query = heads(self.self_attn["q_proj"](fused))
        key = heads(self.self_attn["k_proj"](fused))
        value = heads(self.self_attn["v_proj"](fused))
        context = nn.functional.scaled_dot_product_attention(
            query[None], key[None], value[None], is_causal=chunk_len > 1
        )[0]
        attn_out = self.self_attn["o_proj"](
            context.transpose(0, 1).reshape(chunk_len, -1)
        )
        hidden = hidden + attn_out  # residual = PRE-norm hidden (A2)
        normed = self.post_attention_layernorm(hidden)
        mlp_out = self.mlp["down_proj"](
            nn.functional.silu(self.mlp["gate_proj"](normed)) * self.mlp["up_proj"](normed)
        )
        return hidden + mlp_out


class EagleDraftHead(nn.Module):
    """SpecForge-shaped EAGLE-3 head: fc fusion + midlayer + reduced-vocab head."""

    def __init__(self, config: EagleConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.config = config
        self.fc = nn.Linear(3 * hidden, hidden, bias=False)
        self.midlayer = _EagleMidLayer(config)
        self.norm = RMSNorm(hidden, config.rms_norm_eps)
        self.lm_head = nn.Linear(hidden, config.draft_vocab_size, bias=False)
        self.register_buffer(
            "d2t", torch.zeros(config.draft_vocab_size, dtype=torch.int64)
        )

    def fuse(self, aux_hidden: torch.Tensor) -> torch.Tensor:
        """Target aux hiddens [T, 3H] -> the draft chain's initial hidden [T, H].

        Applied ONCE per verify cycle (A2)."""
        return self.fc(aux_hidden)

    @torch.no_grad()
    def rollout(
        self, embeds: torch.Tensor, initial_hidden: torch.Tensor, embed_fn, k: int
    ) -> list[int]:
        """Greedy k-token draft chain in TARGET ids.

        ``embeds`` [T, H] = target-embedded context; ``initial_hidden`` [T, H]
        = fuse(aux) rows; ``embed_fn(token_id) -> [H]`` embeds drafted tokens
        (the target's embedding — SpecForge ships none).
        """
        embeds = embeds.clone()
        hidden = initial_hidden.clone()
        drafted: list[int] = []
        for _ in range(k):
            hidden = self.midlayer(embeds, hidden)
            logits = self.lm_head(self.norm(hidden[-1]))
            draft_id = int(torch.argmax(logits).item())
            target_id = draft_id + int(self.d2t[draft_id].item())  # offset map
            drafted.append(target_id)
            embeds = torch.cat([embeds, embed_fn(target_id)[None]], dim=0)
            hidden = torch.cat([hidden, hidden[-1:]], dim=0)  # feed back own hidden
        return drafted


def load_eagle_head(path, config: EagleConfig) -> EagleDraftHead:
    """SpecForge checkpoint -> EagleDraftHead; unknown tensors fail loudly.

    ``embed_tokens``/``t2d`` are accepted-and-ignored (embeds come from the
    target model; t2d is training-time only).
    """
    from kairyu.engine.core.weights import CheckpointReader

    head = EagleDraftHead(config)
    reader = CheckpointReader(path)
    expected = dict(head.state_dict())
    ignored = ("embed_tokens.weight", "t2d")
    state: dict[str, torch.Tensor] = {}
    for name in reader.names():
        if any(name.endswith(suffix) for suffix in ignored):
            continue
        if name not in expected:
            raise KeyError(f"unexpected EAGLE tensor {name!r} — format drift")
        state[name] = reader.tensor(name)
    missing = set(expected) - set(state)
    if missing:
        raise KeyError(f"EAGLE checkpoint missing tensors: {sorted(missing)}")
    head.load_state_dict(state)
    head.eval()
    return head
