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
from kairyu.models.config import ModelConfig, parse_model_config
from kairyu.models.layers import RMSNorm
from kairyu.models.llama import DecoderLayer
from kairyu.quant.linear import LinearRole, ModelScope, make_linear


class MtpDraftHead(nn.Module):
    def __init__(self, config: ModelConfig, linear_factory=None) -> None:
        super().__init__()
        hidden = config.hidden_size
        layer_index = config.num_hidden_layers
        prefix = f"model.layers.{layer_index}"
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden)
        self.enorm = RMSNorm(hidden, config.rms_norm_eps)
        self.hnorm = RMSNorm(hidden, config.rms_norm_eps)
        self.eh_proj = make_linear(
            linear_factory,
            2 * hidden,
            hidden,
            False,
            qualified_name=f"{prefix}.eh_proj",
            model_scope=ModelScope.MTP_DRAFT,
            role=LinearRole.DRAFT_FUSION,
            layer_index=layer_index,
        )
        # layer_index = num_hidden_layers: past first_k_dense_replace, so the
        # block correctly comes out MoE (A8)
        self.decoder = DecoderLayer(
            config,
            layer_index=layer_index,
            linear_factory=linear_factory,
            prefix=prefix,
            model_scope=ModelScope.MTP_DRAFT,
        )
        self.shared_head = nn.ModuleDict({"norm": RMSNorm(hidden, config.rms_norm_eps)})
        self.head = make_linear(
            linear_factory,
            hidden,
            config.vocab_size,
            False,
            qualified_name=f"{prefix}.shared_head.head",
            model_scope=ModelScope.MTP_DRAFT,
            role=LinearRole.OUTPUT_HEAD,
            layer_index=layer_index,
        )

    def fresh_pool(
        self,
        num_pages: int = 64,
        page_size: int = 4,
        *,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ) -> PagedKVPool:
        """The head's own 1-layer KV for one dense proposal recompute."""

        reference = self.embed_tokens.weight
        return PagedKVPool(
            num_layers=1,
            num_pages=num_pages,
            page_size=page_size,
            num_kv_heads=self.config.kv_cache_num_heads,
            head_dim=self.config.kv_cache_head_dim,
            v_head_dim=self.config.kv_cache_v_head_dim,
            dtype=reference.dtype if dtype is None else dtype,
            device=reference.device if device is None else device,
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
        if token_ids.device != target_hidden.device:
            raise ValueError(
                "MTP token ids and target hidden states must share a device: "
                f"got {token_ids.device} and {target_hidden.device}"
            )
        embedding = self.embed_tokens(token_ids)
        if embedding.device != target_hidden.device:
            raise ValueError(
                "MTP token embeddings and target hidden states must share a device: "
                f"got {embedding.device} and {target_hidden.device}"
            )
        fused = self.eh_proj(torch.cat([self.enorm(embedding), self.hnorm(target_hidden)], dim=-1))
        length = token_ids.shape[0]
        pool = self.fresh_pool(
            num_pages=-(-length // 4) + 1,
            dtype=target_hidden.dtype,
            device=target_hidden.device,
        )
        positions = torch.arange(length, device=target_hidden.device)
        cos, sin = rotary_emb(positions)
        page_table = list(range(pool.num_pages))
        return self.decoder(
            fused,
            cos,
            sin,
            pool,
            0,
            page_table,
            positions,
            length,
            0,
            chunk_start=0,
            has_writable=length > 0,
        )

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(self.shared_head["norm"](hidden))

    @torch.no_grad()
    def rollout_cached(
        self,
        token_ids: torch.Tensor,
        target_hidden: torch.Tensor,
        rotary_emb,
        k: int,
    ) -> list[int]:
        """Greedy MTP rollout with one context pass and one persistent KV pool."""

        if type(k) is not int or k < 1:
            raise ValueError("MTP rollout length must be a positive integer")
        if token_ids.ndim != 1 or token_ids.shape[0] < 1:
            raise ValueError("MTP token ids must be a nonempty rank-1 tensor")
        expected = (token_ids.shape[0], self.config.hidden_size)
        if target_hidden.ndim != 2 or tuple(target_hidden.shape) != expected:
            raise ValueError(
                "MTP target hidden must pair one-for-one with token ids and have "
                f"shape {expected}, got {tuple(target_hidden.shape)}"
            )
        if token_ids.device != target_hidden.device:
            raise ValueError(
                "MTP token ids and target hidden states must share a device: "
                f"got {token_ids.device} and {target_hidden.device}"
            )

        page_size = 16
        total = token_ids.shape[0] + k - 1
        pool = self.fresh_pool(
            num_pages=-(-total // page_size),
            page_size=page_size,
            dtype=target_hidden.dtype,
            device=target_hidden.device,
        )
        page_table = list(range(pool.num_pages))
        embeddings = self.embed_tokens(token_ids)
        fused = self.eh_proj(
            torch.cat(
                [self.enorm(embeddings), self.hnorm(target_hidden)], dim=-1
            )
        )
        length = token_ids.shape[0]
        positions = torch.arange(length, device=target_hidden.device)
        cos, sin = rotary_emb(positions)
        output = self.decoder(
            fused,
            cos,
            sin,
            pool,
            0,
            page_table,
            positions,
            length,
            0,
            chunk_start=0,
            has_writable=True,
        )[-1]

        drafted: list[int] = []
        for index in range(k):
            # DeepSeek recycles the post-shared-head-norm state into the next
            # MTP step; logits consume that same once-normalized state.
            recycled = self.shared_head["norm"](output)
            token_id = int(torch.argmax(self.head(recycled)).item())
            drafted.append(token_id)
            if index + 1 == k:
                break
            position = length + index
            token = torch.tensor(
                [token_id], dtype=torch.long, device=target_hidden.device
            )
            embedding = self.embed_tokens(token)
            fused = self.eh_proj(
                torch.cat(
                    [self.enorm(embedding), self.hnorm(recycled[None])], dim=-1
                )
            )
            position_tensor = torch.tensor(
                [position], dtype=torch.long, device=target_hidden.device
            )
            cos, sin = rotary_emb(position_tensor)
            output = self.decoder(
                fused,
                cos,
                sin,
                pool,
                0,
                page_table,
                position_tensor,
                position + 1,
                position,
                chunk_start=position,
                has_writable=True,
            )[0]
        return drafted


def load_mtp_head(
    path,
    config: ModelConfig,
    *,
    linear_factory=None,
    target_quantization=None,
    dtype: torch.dtype = torch.float32,
    target_device: str | torch.device = "cpu",
    linear_capabilities=None,
) -> MtpDraftHead:
    """DeepSeek checkpoint MTP-extra layer -> MtpDraftHead.

    Maps ``model.layers.{L}.`` names (L = num_hidden_layers); the decoder
    block reuses the standard in-layer names.

    ``target_device`` is the logical execution target used for linear-kernel
    selection.  Checkpoint tensors remain CPU-resident until the runtime caller
    explicitly moves the returned head, matching ``load_model``.
    """
    import json
    from pathlib import Path

    from kairyu.engine.core.quant_config import load_checkpoint_quantization
    from kairyu.engine.core.weights import CheckpointReader
    from kairyu.models.draft_quant import (
        coerce_draft_state_for_load,
        draft_linear_factory,
        load_draft_quantization,
    )

    checkpoint = Path(path)
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise ValueError(f"MTP checkpoint has no config.json at {config_path}")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError("MTP config.json must contain a JSON object")
    checkpoint_config = parse_model_config(raw_config)
    if checkpoint_config != config:
        mismatches = [
            f"{name}: caller={getattr(config, name)!r}, "
            f"checkpoint={getattr(checkpoint_config, name)!r}"
            for name in config.__dataclass_fields__
            if getattr(config, name) != getattr(checkpoint_config, name)
        ]
        raise ValueError(
            "MTP caller config does not match checkpoint semantics: " + "; ".join(mismatches)
        )
    if target_quantization is None:
        target_quantization = load_checkpoint_quantization(
            checkpoint,
            raw_config,
        ).weights
    metadata = load_draft_quantization(
        checkpoint,
        expected_scope=ModelScope.MTP_DRAFT,
        target_quantization=target_quantization,
    )
    if metadata is not None and metadata.is_quantized:
        if linear_factory is not None:
            raise ValueError(
                "quantized MTP checkpoint metadata cannot be combined with "
                "an explicit linear_factory"
            )
        linear_factory = draft_linear_factory(
            metadata,
            device=target_device,
            dtype=dtype,
            capabilities=linear_capabilities,
        )
    head = MtpDraftHead(config, linear_factory=linear_factory)
    reader = CheckpointReader(checkpoint)
    prefix = f"model.layers.{config.num_hidden_layers}."
    direct = {
        "embed_tokens.weight": "embed_tokens.weight",
        "enorm.weight": "enorm.weight",
        "hnorm.weight": "hnorm.weight",
        "shared_head.norm.weight": "shared_head.norm.weight",
    }
    state: dict[str, torch.Tensor] = {}
    for name in reader.names():
        if not name.startswith(prefix):
            continue
        local = name[len(prefix) :]
        if local in direct:
            target = direct[local]
        elif local.startswith("eh_proj."):
            # Dense ``weight`` and every packed-format payload retain the
            # local projection prefix.
            target = local
        elif local.startswith("shared_head.head."):
            # The module's established local attribute is ``head`` even though
            # the checkpoint identity is ``shared_head.head``.
            target = f"head.{local[len('shared_head.head.') :]}"
        else:  # decoder block tensors keep their in-layer names
            target = f"decoder.{local}"
        state[target] = reader.tensor(name)
    missing = set(head.state_dict()) - set(state)
    if missing:
        raise KeyError(f"MTP layer missing tensors: {sorted(missing)[:5]}")
    head.load_state_dict(
        coerce_draft_state_for_load(head, state, dtype=dtype),
        assign=True,
    )
    head.eval()
    return head
