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

The public ``rollout`` remains the dense CPU reference.  Native serving uses
``rollout_cached``: it computes the context K/V once, then appends one draft
row at a time without re-running the context for every proposed token.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from kairyu.models.layers import RMSNorm, apply_rope
from kairyu.quant.linear import LinearRole, ModelScope, make_linear


def default_eagle3_aux_layer_ids(num_layers: int) -> tuple[int, int, int]:
    """Translate SGLang's default before-layer taps to Kairyu after-layer ids."""

    if type(num_layers) is not int or num_layers < 7:
        raise ValueError("EAGLE-3 auxiliary layer selection requires at least 7 layers")
    layer_ids = (1, num_layers // 2 - 1, num_layers - 4)
    if tuple(sorted(set(layer_ids))) != layer_ids or layer_ids[-1] >= num_layers:
        raise ValueError(f"EAGLE-3 auxiliary layer selection is invalid for {num_layers} layers")
    return layer_ids


@dataclass(frozen=True)
class EagleConfig:
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    draft_vocab_size: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0  # SpecForge/vLLM EAGLE-3 midlayers are RoPE-trained (H1)
    # Older Kairyu fixtures modelled the draft layer as ordinary MHA and
    # derived the head width from ``hidden_size``.  Public EAGLE checkpoints
    # may instead retain the target model's explicit GQA geometry (for
    # example Qwen3-32B uses 64 query heads, 8 KV heads, and head_dim=128 even
    # though hidden_size=5120).  Resolve omitted values to the former MHA
    # contract so existing callers remain source- and behavior-compatible.
    num_key_value_heads: int | None = None
    head_dim: int | None = None

    @classmethod
    def from_checkpoint_config(cls, raw: Mapping[str, object]) -> EagleConfig:
        """Parse the semantic EAGLE geometry declared by ``config.json``."""

        if not isinstance(raw, Mapping):
            raise ValueError("EAGLE config.json must contain a JSON object")

        def required_int(name: str) -> int:
            value = raw.get(name)
            if type(value) is not int or value < 1:
                raise ValueError(f"EAGLE config field {name!r} must be a positive integer")
            return value

        def optional_int(name: str, default: int | None) -> int | None:
            if name not in raw:
                return default
            return required_int(name)

        def required_float(name: str) -> float:
            value = raw.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"EAGLE config field {name!r} must be a positive number")
            parsed = float(value)
            if not math.isfinite(parsed) or parsed <= 0:
                raise ValueError(f"EAGLE config field {name!r} must be a positive number")
            return parsed

        query_heads = required_int("num_attention_heads")
        return cls(
            hidden_size=required_int("hidden_size"),
            num_attention_heads=query_heads,
            intermediate_size=required_int("intermediate_size"),
            draft_vocab_size=required_int("draft_vocab_size"),
            rms_norm_eps=required_float("rms_norm_eps"),
            rope_theta=required_float("rope_theta"),
            num_key_value_heads=optional_int("num_key_value_heads", query_heads),
            head_dim=optional_int("head_dim", None),
        )

    def __post_init__(self) -> None:
        if self.num_attention_heads < 1:
            raise ValueError("num_attention_heads must be positive")
        kv_heads = (
            self.num_attention_heads
            if self.num_key_value_heads is None
            else self.num_key_value_heads
        )
        if kv_heads < 1:
            raise ValueError("num_key_value_heads must be positive")
        if self.num_attention_heads % kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    "hidden_size must be divisible by num_attention_heads when head_dim is omitted"
                )
            head_dim = self.hidden_size // self.num_attention_heads
        else:
            head_dim = self.head_dim
        if head_dim < 1:
            raise ValueError("head_dim must be positive")
        if head_dim % 2:
            raise ValueError("head_dim must be even for rotary embedding")
        object.__setattr__(self, "num_key_value_heads", kv_heads)
        object.__setattr__(self, "head_dim", head_dim)


class _EagleMidLayer(nn.Module):
    """The 2H-input decoder layer (checkpoint prefix ``midlayer.``)."""

    def __init__(self, config: EagleConfig, linear_factory=None) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        assert config.num_key_value_heads is not None
        assert config.head_dim is not None
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        query_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        # RoPE over the head dim (H1): SpecForge/vLLM EAGLE-3 midlayers are Llama
        # attention layers trained WITH rotary embeddings — omitting it makes the
        # draft position-blind and collapses acceptance at any context length
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.int64).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.input_layernorm = RMSNorm(hidden, config.rms_norm_eps)
        self.hidden_norm = RMSNorm(hidden, config.rms_norm_eps)
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": make_linear(
                    linear_factory,
                    2 * hidden,
                    query_width,
                    False,
                    qualified_name="midlayer.self_attn.q_proj",
                    model_scope=ModelScope.EAGLE_DRAFT,
                    role=LinearRole.ATTENTION_QUERY,
                    layer_index=0,
                    shard_dim=0,
                ),
                "k_proj": make_linear(
                    linear_factory,
                    2 * hidden,
                    kv_width,
                    False,
                    qualified_name="midlayer.self_attn.k_proj",
                    model_scope=ModelScope.EAGLE_DRAFT,
                    role=LinearRole.ATTENTION_KEY,
                    layer_index=0,
                    shard_dim=0,
                ),
                "v_proj": make_linear(
                    linear_factory,
                    2 * hidden,
                    kv_width,
                    False,
                    qualified_name="midlayer.self_attn.v_proj",
                    model_scope=ModelScope.EAGLE_DRAFT,
                    role=LinearRole.ATTENTION_VALUE,
                    layer_index=0,
                    shard_dim=0,
                ),
                "o_proj": make_linear(
                    linear_factory,
                    query_width,
                    hidden,
                    False,
                    qualified_name="midlayer.self_attn.o_proj",
                    model_scope=ModelScope.EAGLE_DRAFT,
                    role=LinearRole.ATTENTION_OUTPUT,
                    layer_index=0,
                    shard_dim=1,
                ),
            }
        )
        self.post_attention_layernorm = RMSNorm(hidden, config.rms_norm_eps)
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": make_linear(
                    linear_factory,
                    hidden,
                    config.intermediate_size,
                    False,
                    qualified_name="midlayer.mlp.gate_proj",
                    model_scope=ModelScope.EAGLE_DRAFT,
                    role=LinearRole.MLP_GATE,
                    layer_index=0,
                    shard_dim=0,
                ),
                "up_proj": make_linear(
                    linear_factory,
                    hidden,
                    config.intermediate_size,
                    False,
                    qualified_name="midlayer.mlp.up_proj",
                    model_scope=ModelScope.EAGLE_DRAFT,
                    role=LinearRole.MLP_UP,
                    layer_index=0,
                    shard_dim=0,
                ),
                "down_proj": make_linear(
                    linear_factory,
                    config.intermediate_size,
                    hidden,
                    False,
                    qualified_name="midlayer.mlp.down_proj",
                    model_scope=ModelScope.EAGLE_DRAFT,
                    role=LinearRole.MLP_DOWN,
                    layer_index=0,
                    shard_dim=1,
                ),
            }
        )

    def forward(
        self,
        embeds: torch.Tensor,
        hidden: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """embeds/hidden: [T, H]; dense causal attention over the T positions.

        ``positions`` (default 0..T-1) drives RoPE on q/k (H1)."""
        fused = torch.cat([self.input_layernorm(embeds), self.hidden_norm(hidden)], -1)
        chunk_len = fused.shape[0]
        if positions is None:
            positions = torch.arange(chunk_len, device=fused.device)

        def heads_thd(projection: torch.Tensor, num_heads: int) -> torch.Tensor:
            return projection.view(chunk_len, num_heads, self.head_dim)

        query = heads_thd(self.self_attn["q_proj"](fused), self.num_heads)
        key = heads_thd(self.self_attn["k_proj"](fused), self.num_kv_heads)
        value = heads_thd(self.self_attn["v_proj"](fused), self.num_kv_heads)
        freqs = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        query, key = apply_rope(query, key, emb.cos(), emb.sin())  # [T, heads, d]
        query = query.transpose(0, 1)  # [heads, T, d]
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        context = nn.functional.scaled_dot_product_attention(
            query[None],
            key[None],
            value[None],
            is_causal=chunk_len > 1,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )[0]
        attn_out = self.self_attn["o_proj"](context.transpose(0, 1).reshape(chunk_len, -1))
        hidden = hidden + attn_out  # residual = PRE-norm hidden (A2)
        normed = self.post_attention_layernorm(hidden)
        mlp_out = self.mlp["down_proj"](
            nn.functional.silu(self.mlp["gate_proj"](normed)) * self.mlp["up_proj"](normed)
        )
        return hidden + mlp_out

    def _project(
        self,
        embeds: torch.Tensor,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project and rotate one contiguous draft sequence."""

        fused = torch.cat(
            [self.input_layernorm(embeds), self.hidden_norm(hidden)], -1
        )
        length = fused.shape[0]

        def heads_thd(projection: torch.Tensor, num_heads: int) -> torch.Tensor:
            return projection.view(length, num_heads, self.head_dim)

        query = heads_thd(self.self_attn["q_proj"](fused), self.num_heads)
        key = heads_thd(self.self_attn["k_proj"](fused), self.num_kv_heads)
        value = heads_thd(self.self_attn["v_proj"](fused), self.num_kv_heads)
        freqs = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
        rope = torch.cat((freqs, freqs), dim=-1)
        query, key = apply_rope(query, key, rope.cos(), rope.sin())
        return (
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
        )

    def _finish(self, context: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        length = context.shape[1]
        attn_out = self.self_attn["o_proj"](
            context.transpose(0, 1).reshape(length, -1)
        )
        residual = hidden + attn_out
        normed = self.post_attention_layernorm(residual)
        mlp_out = self.mlp["down_proj"](
            nn.functional.silu(self.mlp["gate_proj"](normed))
            * self.mlp["up_proj"](normed)
        )
        return residual + mlp_out

    def prefill_cached(
        self,
        embeds: torch.Tensor,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the context once and return its last output plus reusable K/V."""

        query, key, value = self._project(embeds, hidden, positions)
        context = nn.functional.scaled_dot_product_attention(
            query[None],
            key[None],
            value[None],
            is_causal=query.shape[1] > 1,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )[0]
        return self._finish(context, hidden)[-1], key, value

    def decode_cached(
        self,
        embed: torch.Tensor,
        hidden: torch.Tensor,
        position: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append one feedback row to an existing linear-chain draft cache."""

        positions = torch.tensor([position], dtype=torch.long, device=embed.device)
        query, next_key, next_value = self._project(
            embed[None], hidden[None], positions
        )
        key[:, position : position + 1].copy_(next_key)
        value[:, position : position + 1].copy_(next_value)
        # The sole query is the newest absolute position and may attend to every
        # cached key.  ``is_causal=True`` would apply a length-1 local mask and
        # incorrectly expose only the first key.
        context = nn.functional.scaled_dot_product_attention(
            query[None],
            key[None, :, : position + 1],
            value[None, :, : position + 1],
            is_causal=False,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )[0]
        return self._finish(context, hidden[None])[0], key, value


class EagleDraftHead(nn.Module):
    """SpecForge-shaped EAGLE-3 head: fc fusion + midlayer + reduced-vocab head."""

    def __init__(self, config: EagleConfig, linear_factory=None) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.config = config
        self.fc = make_linear(
            linear_factory,
            3 * hidden,
            hidden,
            False,
            qualified_name="fc",
            model_scope=ModelScope.EAGLE_DRAFT,
            role=LinearRole.DRAFT_FUSION,
        )
        self.midlayer = _EagleMidLayer(config, linear_factory=linear_factory)
        self.norm = RMSNorm(hidden, config.rms_norm_eps)
        self.lm_head = make_linear(
            linear_factory,
            hidden,
            config.draft_vocab_size,
            False,
            qualified_name="lm_head",
            model_scope=ModelScope.EAGLE_DRAFT,
            role=LinearRole.OUTPUT_HEAD,
        )
        self.register_buffer("d2t", torch.zeros(config.draft_vocab_size, dtype=torch.int64))

    def fuse(self, aux_hidden: torch.Tensor) -> torch.Tensor:
        """Target aux hiddens [T, 3H] -> the draft chain's initial hidden [T, H].

        Applied ONCE per verify cycle (A2)."""
        return self.fc(aux_hidden)

    @torch.no_grad()
    def rollout(
        self,
        shifted_embeds: torch.Tensor,
        initial_hidden: torch.Tensor,
        embed_fn,
        k: int,
    ) -> list[int]:
        """Greedy k-token draft chain in TARGET ids.

        ``shifted_embeds`` [T, H] is the EAGLE-shifted target embedding sequence:
        token rows ``1..T`` ending in the target-produced root, while
        ``initial_hidden`` [T, H] is fused from target auxiliary rows
        ``0..T-1``. ``embed_fn(token_id) -> [H]`` embeds subsequent drafted
        tokens (the target's embedding — SpecForge ships none).
        """
        if type(k) is not int or k < 1:
            raise ValueError("EAGLE rollout length must be a positive integer")
        if (
            initial_hidden.ndim != 2
            or initial_hidden.shape[0] < 1
            or initial_hidden.shape[1] != self.config.hidden_size
        ):
            raise ValueError(
                "EAGLE initial hidden must be a nonempty rank-2 tensor with width "
                f"{self.config.hidden_size}, got {tuple(initial_hidden.shape)}"
            )
        expected = (initial_hidden.shape[0], self.config.hidden_size)
        if shifted_embeds.ndim != 2 or tuple(shifted_embeds.shape) != expected:
            raise ValueError(
                "EAGLE shifted embeddings must pair one-for-one with auxiliary "
                f"hidden rows and have shape {expected}, got {tuple(shifted_embeds.shape)}"
            )
        if shifted_embeds.device != initial_hidden.device:
            raise ValueError(
                "EAGLE shifted embeddings and initial hidden must share a device: "
                f"got {shifted_embeds.device} and {initial_hidden.device}"
            )
        embeds = shifted_embeds.clone()
        # `hidden` accumulates the midlayer's INPUT hidden per position: the
        # context rows stay the original fuse(aux) inputs (their K/V are computed
        # once and frozen in a KV-cached EAGLE), and each draft step appends ONE
        # feedback row = the previous step's output for the last position (H2).
        # The old code overwrote the whole tensor with the midlayer OUTPUT, so
        # every context position's K/V was recomputed from mutated inputs.
        hidden = initial_hidden.clone()
        drafted: list[int] = []
        for _ in range(k):
            positions = torch.arange(embeds.shape[0], device=embeds.device)
            out = self.midlayer(embeds, hidden, positions)
            logits = self.lm_head(self.norm(out[-1]))
            draft_id = int(torch.argmax(logits).item())
            target_id = draft_id + int(self.d2t[draft_id].item())  # offset map
            drafted.append(target_id)
            next_embed = embed_fn(target_id)
            if tuple(next_embed.shape) != (self.config.hidden_size,):
                raise ValueError(
                    "EAGLE target embedding callback must return one hidden row with "
                    f"shape ({self.config.hidden_size},), got {tuple(next_embed.shape)}"
                )
            if next_embed.device != embeds.device:
                raise ValueError(
                    "EAGLE target embedding callback returned the wrong device: "
                    f"expected {embeds.device}, got {next_embed.device}"
                )
            embeds = torch.cat([embeds, next_embed[None]], dim=0)
            hidden = torch.cat([hidden, out[-1:]], dim=0)  # append feedback only
        return drafted

    @torch.no_grad()
    def rollout_cached(
        self,
        shifted_embeds: torch.Tensor,
        initial_hidden: torch.Tensor,
        embed_fn,
        k: int,
    ) -> list[int]:
        """Greedy rollout with one context pass and incremental draft K/V."""

        if type(k) is not int or k < 1:
            raise ValueError("EAGLE rollout length must be a positive integer")
        if (
            initial_hidden.ndim != 2
            or initial_hidden.shape[0] < 1
            or initial_hidden.shape[1] != self.config.hidden_size
        ):
            raise ValueError(
                "EAGLE initial hidden must be a nonempty rank-2 tensor with width "
                f"{self.config.hidden_size}, got {tuple(initial_hidden.shape)}"
            )
        expected = (initial_hidden.shape[0], self.config.hidden_size)
        if shifted_embeds.ndim != 2 or tuple(shifted_embeds.shape) != expected:
            raise ValueError(
                "EAGLE shifted embeddings must pair one-for-one with auxiliary "
                f"hidden rows and have shape {expected}, got {tuple(shifted_embeds.shape)}"
            )
        if shifted_embeds.device != initial_hidden.device:
            raise ValueError(
                "EAGLE shifted embeddings and initial hidden must share a device: "
                f"got {shifted_embeds.device} and {initial_hidden.device}"
            )

        length = shifted_embeds.shape[0]
        positions = torch.arange(length, device=shifted_embeds.device)
        output, key, value = self.midlayer.prefill_cached(
            shifted_embeds, initial_hidden, positions
        )
        capacity = length + k - 1
        key_cache = key.new_empty((key.shape[0], capacity, key.shape[2]))
        value_cache = value.new_empty((value.shape[0], capacity, value.shape[2]))
        key_cache[:, :length].copy_(key)
        value_cache[:, :length].copy_(value)
        drafted: list[int] = []
        for index in range(k):
            logits = self.lm_head(self.norm(output))
            draft_id = int(torch.argmax(logits).item())
            target_id = draft_id + int(self.d2t[draft_id].item())
            drafted.append(target_id)
            if index + 1 == k:
                break
            next_embed = embed_fn(target_id)
            if tuple(next_embed.shape) != (self.config.hidden_size,):
                raise ValueError(
                    "EAGLE target embedding callback must return one hidden row with "
                    f"shape ({self.config.hidden_size},), got {tuple(next_embed.shape)}"
                )
            if next_embed.device != shifted_embeds.device:
                raise ValueError(
                    "EAGLE target embedding callback returned the wrong device: "
                    f"expected {shifted_embeds.device}, got {next_embed.device}"
                )
            output, key_cache, value_cache = self.midlayer.decode_cached(
                next_embed,
                output,
                length + index,
                key_cache,
                value_cache,
            )
        return drafted


def load_eagle_head(
    path,
    config: EagleConfig,
    *,
    linear_factory=None,
    target_quantization=None,
    dtype: torch.dtype = torch.float32,
    target_device: str | torch.device = "cpu",
    linear_capabilities=None,
) -> EagleDraftHead:
    """SpecForge checkpoint -> EagleDraftHead; unknown tensors fail loudly.

    ``embed_tokens``/``t2d`` are accepted-and-ignored (embeds come from the
    target model; t2d is training-time only).

    ``target_device`` is the logical execution target used for linear-kernel
    selection.  As with ``load_model``, checkpoint allocation remains on CPU;
    the runtime caller explicitly moves the returned head to that device.
    """
    import json
    from pathlib import Path

    from kairyu.engine.core.weights import CheckpointReader
    from kairyu.models.draft_quant import (
        coerce_draft_state_for_load,
        draft_linear_factory,
        parse_draft_quantization,
    )

    checkpoint = Path(path)
    metadata = None
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise ValueError(f"EAGLE checkpoint has no config.json at {config_path}")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint_config = EagleConfig.from_checkpoint_config(raw_config)
    geometry_fields = (
        "hidden_size",
        "intermediate_size",
        "draft_vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rope_theta",
        "rms_norm_eps",
    )
    mismatches = [
        f"{name}: caller={getattr(config, name)!r}, checkpoint={getattr(checkpoint_config, name)!r}"
        for name in geometry_fields
        if getattr(config, name) != getattr(checkpoint_config, name)
    ]
    if mismatches:
        raise ValueError(
            "EAGLE caller config does not match checkpoint semantics: " + "; ".join(mismatches)
        )
    metadata = parse_draft_quantization(
        raw_config,
        expected_scope=ModelScope.EAGLE_DRAFT,
        target_quantization=target_quantization,
    )
    if metadata is not None and metadata.is_quantized:
        if linear_factory is not None:
            raise ValueError(
                "quantized EAGLE checkpoint metadata cannot be combined with "
                "an explicit linear_factory"
            )
        linear_factory = draft_linear_factory(
            metadata,
            device=target_device,
            dtype=dtype,
            capabilities=linear_capabilities,
        )
    head = EagleDraftHead(config, linear_factory=linear_factory)
    reader = CheckpointReader(checkpoint)
    expected = dict(head.state_dict())
    ignored = ("embed_tokens.weight", "t2d")
    state: dict[str, torch.Tensor] = {}
    for name in reader.names():
        if name in ignored:
            continue
        if name not in expected:
            raise KeyError(f"unexpected EAGLE tensor {name!r} — format drift")
        tensor = reader.tensor(name)
        if name == "d2t":
            if tensor.dtype not in (torch.int32, torch.int64):
                raise TypeError(
                    "EAGLE d2t must contain signed 32-bit or 64-bit integer offsets, "
                    f"got {tensor.dtype}"
                )
            # Public SpecJAX/SpecForge checkpoints store compact I32 offsets;
            # runtime indexing uses an int64 buffer.  Widen explicitly rather
            # than relying on load_state_dict's implicit parameter copy cast.
            tensor = tensor.to(dtype=torch.int64)
        state[name] = tensor
    missing = set(expected) - set(state)
    if missing:
        raise KeyError(f"EAGLE checkpoint missing tensors: {sorted(missing)}")
    head.load_state_dict(
        coerce_draft_state_for_load(head, state, dtype=dtype),
        assign=True,
    )
    head.eval()
    return head
