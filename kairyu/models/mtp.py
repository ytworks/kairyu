"""DeepSeek-V3 MTP draft head (m17 D5, pins per review A8 — verified against
vLLM deepseek_mtp, SGLang deepseek_nextn, and the DeepSeek-V3 weight README).

Checkpoint layout: the MTP layer lives at ``model.layers.{num_hidden_layers}``
with ``enorm``/``hnorm``/``eh_proj``/``embed_tokens``/``shared_head.{norm,
head}`` plus one FULL MLA+MoE decoder block. ``eh_proj`` input is
``cat([enorm(embedding), hnorm(hidden)])`` — EMBEDDING FIRST (the paper's
equation writes it the other way; the implementations win). ``shared_head.
head`` and ``embed_tokens`` are separate physical tensors (tied to the target
only by training) — loaded, never assumed tied. k>1 MTP = reapplying the same
module on its own output.
"""

from __future__ import annotations

import torch
from torch import nn

from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.models.config import ModelConfig
from kairyu.models.layers import RMSNorm
from kairyu.models.llama import DecoderLayer


class MtpDraftHead(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden)
        self.enorm = RMSNorm(hidden, config.rms_norm_eps)
        self.hnorm = RMSNorm(hidden, config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * hidden, hidden, bias=False)
        # layer_index = num_hidden_layers: past first_k_dense_replace, so the
        # block correctly comes out MoE (A8)
        self.decoder = DecoderLayer(config, layer_index=config.num_hidden_layers)
        self.shared_head = nn.ModuleDict(
            {"norm": RMSNorm(hidden, config.rms_norm_eps)}
        )
        self.head = nn.Linear(hidden, config.vocab_size, bias=False)

    def fresh_pool(self, num_pages: int = 64, page_size: int = 4) -> PagedKVPool:
        """The head's own 1-layer KV (dense per-proposal recompute on CPU)."""
        return PagedKVPool(
            num_layers=1,
            num_pages=num_pages,
            page_size=page_size,
            num_kv_heads=self.config.kv_cache_num_heads,
            head_dim=self.config.kv_cache_head_dim,
            v_head_dim=self.config.kv_cache_v_head_dim,
        )

    @torch.no_grad()
    def forward_chain(
        self,
        token_ids: torch.Tensor,
        target_hidden: torch.Tensor,
        rotary_emb,
    ) -> torch.Tensor:
        """One MTP application over the context: [T] tokens + [T, H] target
        hidden -> [T, H] draft hidden (last row feeds ``logits``)."""
        embedding = self.embed_tokens(token_ids)
        fused = self.eh_proj(
            torch.cat([self.enorm(embedding), self.hnorm(target_hidden)], dim=-1)
        )
        length = token_ids.shape[0]
        pool = self.fresh_pool(num_pages=-(-length // 4) + 1)
        positions = torch.arange(length)
        cos, sin = rotary_emb(positions)
        page_table = list(range(pool.num_pages))
        return self.decoder(
            fused, cos, sin, pool, 0, page_table, positions, length, 0
        )

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(self.shared_head["norm"](hidden))


def load_mtp_head(path, config: ModelConfig) -> MtpDraftHead:
    """DeepSeek checkpoint MTP-extra layer -> MtpDraftHead.

    Maps ``model.layers.{L}.`` names (L = num_hidden_layers); the decoder
    block reuses the standard in-layer names.
    """
    from kairyu.engine.core.weights import CheckpointReader

    head = MtpDraftHead(config)
    reader = CheckpointReader(path)
    prefix = f"model.layers.{config.num_hidden_layers}."
    rename = {
        "embed_tokens.weight": "embed_tokens.weight",
        "enorm.weight": "enorm.weight",
        "hnorm.weight": "hnorm.weight",
        "eh_proj.weight": "eh_proj.weight",
        "shared_head.norm.weight": "shared_head.norm.weight",
        "shared_head.head.weight": "head.weight",
    }
    state: dict[str, torch.Tensor] = {}
    for name in reader.names():
        if not name.startswith(prefix):
            continue
        local = name[len(prefix):]
        if local in rename:
            state[rename[local]] = reader.tensor(name)
        else:  # decoder block tensors keep their in-layer names
            state[f"decoder.{local}"] = reader.tensor(name)
    missing = set(head.state_dict()) - set(state)
    if missing:
        raise KeyError(f"MTP layer missing tensors: {sorted(missing)[:5]}")
    head.load_state_dict(state)
    head.eval()
    return head
