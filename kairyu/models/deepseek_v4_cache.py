"""Memory-bounded HCA/CSA cache extensions for DeepSeek V4.

The outer compressed K/V remains BF16 because it is consumed by attention.
The Lightning Indexer history is stored as block-32 E2M1 with UE8M0 scales and
scored in bounded blocks, avoiding both a BF16 duplicate and an S x T score
matrix at native context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MethodType

import torch
import torch.nn.functional as F

from kairyu.kernels.deepseek_v4_moe_gpu import _fp4_linear

_FP4_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_INDEX_QUERY_BLOCK = 16
_ATTENTION_KEY_BLOCK = 512
_INDEX_PACKED_CHUNK_ROWS = 8192


@dataclass
class PackedFP4IndexerCache:
    batch_size: int
    head_dim: int
    packed_chunks: list[torch.Tensor] = field(default_factory=list)
    scale_chunks: list[torch.Tensor] = field(default_factory=list)
    entry_count: int = 0

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.batch_size, self.entry_count, self.head_dim)

    def append(self, values: torch.Tensor) -> None:
        if values.ndim != 3 or values.shape[0] != self.batch_size:
            raise ValueError("DeepSeek V4 indexer cache append has invalid geometry")
        if values.shape[0] != 1:
            raise ValueError("DeepSeek V4 request-owned cache requires batch size one")
        if values.shape[-1] != self.head_dim or self.head_dim % 32:
            raise ValueError("DeepSeek V4 FP4 indexer head dim must divide into 32")
        if values.shape[1] == 0:
            return
        rows = values.reshape(-1, self.head_dim).float()
        blocks = rows.reshape(rows.shape[0], self.head_dim // 32, 32)
        amax = blocks.abs().amax(dim=-1)
        dequant_scale = torch.where(
            amax > 0,
            torch.pow(
                2.0,
                torch.ceil(torch.log2((amax / 6.0).clamp_min(torch.finfo(torch.float32).tiny))),
            ),
            torch.ones_like(amax),
        )
        normalized = (blocks / dequant_scale.unsqueeze(-1)).clamp(-6.0, 6.0)
        thresholds = normalized.new_tensor(_FP4_THRESHOLDS)
        magnitude = torch.bucketize(normalized.abs().contiguous(), thresholds)
        nibble = magnitude.to(torch.uint8) | ((normalized < 0).to(torch.uint8) << 3)
        nibble = nibble.reshape(rows.shape[0], self.head_dim)
        packed = nibble[:, 0::2] | (nibble[:, 1::2] << 4)
        ue8m0 = getattr(torch, "float8_e8m0fnu", None)
        if ue8m0 is None:
            raise RuntimeError("DeepSeek V4 FP4 indexer cache requires UE8M0 support")
        scales = dequant_scale.to(ue8m0)
        packed = packed.view(torch.int8).contiguous()
        scales = scales.contiguous()
        if (
            self.packed_chunks
            and self.packed_chunks[-1].shape[0] + packed.shape[0]
            <= _INDEX_PACKED_CHUNK_ROWS
        ):
            self.packed_chunks[-1] = torch.cat((self.packed_chunks[-1], packed), dim=0)
            self.scale_chunks[-1] = torch.cat((self.scale_chunks[-1], scales), dim=0)
        else:
            self.packed_chunks.append(packed)
            self.scale_chunks.append(scales)
        self.entry_count += values.shape[1]


def _register_cache_layer():
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4CSACache

    class DeepseekV4FP4CSACache(DeepseekV4CSACache):
        layer_type = "compressed_sparse_attention"

        def __init__(self, config):
            super().__init__(config)
            self.fp4_indexer = PackedFP4IndexerCache(1, config.index_head_dim)

        def update_compressor_states(self, name: str, compressed: torch.Tensor):
            if name != "indexer":
                return super().update_compressor_states(name, compressed)
            if self.fp4_indexer.batch_size != compressed.shape[0]:
                if self.fp4_indexer.entry_count:
                    raise RuntimeError("DeepSeek V4 indexer cache batch size changed")
                self.fp4_indexer.batch_size = compressed.shape[0]
            self.fp4_indexer.append(compressed)
            self.entry_count[name] += compressed.shape[1]
            self.compressed_kv[name] = self.fp4_indexer
            return self.fp4_indexer

    return DeepseekV4FP4CSACache


DeepseekV4FP4CSACache = _register_cache_layer()


def make_deepseek_v4_cache(config):
    """Build one custom cache without changing Transformers process globals."""

    from transformers.cache_utils import LAYER_TYPE_CACHE_MAPPING, DynamicCache

    layer_type = DeepseekV4FP4CSACache.layer_type
    previous = LAYER_TYPE_CACHE_MAPPING.get(layer_type)
    LAYER_TYPE_CACHE_MAPPING[layer_type] = DeepseekV4FP4CSACache
    try:
        return DynamicCache(config=config)
    finally:
        if previous is None:
            LAYER_TYPE_CACHE_MAPPING.pop(layer_type, None)
        else:
            LAYER_TYPE_CACHE_MAPPING[layer_type] = previous


@dataclass(frozen=True)
class CompressedAttentionState:
    keys: torch.Tensor
    compress_rate: int
    selected_indices: torch.Tensor | None = None


def _fp4_index_topk(
    scorer,
    q: torch.Tensor,
    cache: PackedFP4IndexerCache,
    hidden_states: torch.Tensor,
    causal_threshold: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    batch, sequence, heads, head_dim = q.shape
    if batch != cache.batch_size or head_dim != cache.head_dim:
        raise RuntimeError("DeepSeek V4 FP4 index scorer/cache geometry mismatch")
    weights = scorer.weights_proj(hidden_states).float() * scorer.weights_scaling
    result = torch.empty((batch, sequence, top_k), dtype=torch.long, device=q.device)
    for query_start in range(0, sequence, _INDEX_QUERY_BLOCK):
        query_stop = min(sequence, query_start + _INDEX_QUERY_BLOCK)
        query = q[:, query_start:query_stop].reshape(-1, head_dim).contiguous()
        query_weights = weights[:, query_start:query_stop]
        threshold = causal_threshold[:, query_start:query_stop]
        best_values = torch.empty(
            (batch, query_stop - query_start, 0),
            dtype=torch.float32,
            device=q.device,
        )
        best_indices = torch.empty_like(best_values, dtype=torch.long)
        key_start = 0
        for packed, scales in zip(cache.packed_chunks, cache.scale_chunks, strict=True):
            key_count = packed.shape[0] // batch
            if packed.shape[0] != batch * key_count:
                raise RuntimeError("DeepSeek V4 FP4 index cache chunk is malformed")
            scores = _fp4_linear(query, packed, scales)
            scores = scores.view(batch, query_stop - query_start, heads, key_count)
            scores = F.relu(scores.float()) * scorer.softmax_scale
            scores = (scores * query_weights.unsqueeze(-1)).sum(dim=2)
            indices = torch.arange(key_start, key_start + key_count, device=q.device)
            scores.masked_fill_(indices.view(1, 1, -1) >= threshold.unsqueeze(-1), float("-inf"))
            indices = indices.view(1, 1, -1).expand(batch, query_stop - query_start, -1)
            candidates = torch.cat((best_values, scores), dim=-1)
            candidate_indices = torch.cat((best_indices, indices), dim=-1)
            retain = min(top_k, candidates.shape[-1])
            best_values, selected = candidates.topk(retain, dim=-1)
            best_indices = candidate_indices.gather(-1, selected)
            key_start += key_count
        invalid = best_indices >= threshold.unsqueeze(-1)
        result[:, query_start:query_stop] = best_indices.masked_fill(invalid, -1)
    return result


def _indexer_forward_fp4(
    self,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values,
    layer_idx: int,
) -> torch.LongTensor:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb

    batch, seq_len, _ = hidden_states.shape
    cache_layer = past_key_values.layers[layer_idx] if past_key_values is not None else None
    if cache_layer is None or not isinstance(cache_layer, DeepseekV4FP4CSACache):
        raise RuntimeError("DeepSeek V4 native indexer requires its FP4 CSA cache")
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)
    chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights(
        "indexer", kv, gate
    )
    if chunk_kv.shape[1] > 0:
        windows = chunk_kv.shape[1] // self.compress_rate
        ratio = self.compress_rate
        chunk_kv = chunk_kv.view(batch, windows, ratio, -1)
        chunk_gate = chunk_gate.view(batch, windows, ratio, -1) + self.position_bias
        new_kv = chunk_kv.new_zeros((batch, windows, 2 * ratio, self.head_dim))
        new_gate = chunk_gate.new_full(
            (batch, windows, 2 * ratio, self.head_dim), float("-inf")
        )
        new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
        new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
        if windows > 1:
            new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
            new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
        prior_kv, prior_gate = cache_layer.update_overlap_state(
            "indexer", chunk_kv, chunk_gate, self.head_dim
        )
        if prior_kv is not None:
            new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
            new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)
        compressed = self.kv_norm(
            (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(
                dim=2
            )
        )
        positions = torch.arange(windows, device=compressed.device)
        positions = positions * ratio + first_window_position
        positions = positions.unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary_emb(
            compressed, position_ids=positions, layer_type=self.rope_layer_type
        )
        compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
    else:
        compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))
    packed_cache = cache_layer.update_compressor_states("indexer", compressed)
    total = packed_cache.entry_count
    top_k = min(self.index_topk, total)
    if top_k == 0:
        return torch.empty((batch, seq_len, 0), dtype=torch.long, device=hidden_states.device)
    cos_q, sin_q = self.rotary_emb(
        hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type
    )
    q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
    q = apply_rotary_pos_emb(q, cos_q, sin_q).transpose(1, 2)
    causal_threshold = (position_ids + 1) // self.compress_rate
    return _fp4_index_topk(
        self.scorer,
        q,
        packed_cache,
        hidden_states,
        causal_threshold,
        top_k,
    )


def _compress_hca(
    self,
    hidden_states: torch.Tensor,
    _q_residual: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values,
    layer_idx: int,
) -> CompressedAttentionState:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb

    batch = hidden_states.shape[0]
    cache_layer = past_key_values.layers[layer_idx]
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)
    chunk_kv, chunk_gate, first = cache_layer.store_compression_weights(
        "compressor", kv, gate
    )
    if chunk_kv.shape[1]:
        windows = chunk_kv.shape[1] // self.compress_rate
        chunk_kv = chunk_kv.view(batch, windows, self.compress_rate, -1)
        chunk_gate = (
            chunk_gate.view(batch, windows, self.compress_rate, -1) + self.position_bias
        )
        compressed = self.kv_norm(
            (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(
                dim=2
            )
        )
        positions = torch.arange(windows, device=compressed.device) * self.compress_rate + first
        positions = positions.unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary_emb(
            compressed, position_ids=positions, layer_type=self.rope_layer_type
        )
        compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
    else:
        compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))
    keys = cache_layer.update_compressor_states("compressor", compressed)
    return CompressedAttentionState(keys, self.compress_rate)


def _compress_csa(
    self,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values,
    layer_idx: int,
) -> CompressedAttentionState:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb

    batch = hidden_states.shape[0]
    cache_layer = past_key_values.layers[layer_idx]
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)
    chunk_kv, chunk_gate, first = cache_layer.store_compression_weights(
        "compressor", kv, gate
    )
    if chunk_kv.shape[1]:
        windows = chunk_kv.shape[1] // self.compress_rate
        ratio = self.compress_rate
        chunk_kv = chunk_kv.view(batch, windows, ratio, -1)
        chunk_gate = chunk_gate.view(batch, windows, ratio, -1) + self.position_bias
        new_kv = chunk_kv.new_zeros((batch, windows, 2 * ratio, self.head_dim))
        new_gate = chunk_gate.new_full(
            (batch, windows, 2 * ratio, self.head_dim), float("-inf")
        )
        new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
        new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
        if windows > 1:
            new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
            new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
        prior_kv, prior_gate = cache_layer.update_overlap_state(
            "compressor", chunk_kv, chunk_gate, self.head_dim
        )
        if prior_kv is not None:
            new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
            new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)
        compressed = self.kv_norm(
            (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(
                dim=2
            )
        )
        positions = torch.arange(windows, device=compressed.device) * ratio + first
        positions = positions.unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary_emb(
            compressed, position_ids=positions, layer_type=self.rope_layer_type
        )
        compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
    else:
        compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))
    keys = cache_layer.update_compressor_states("compressor", compressed)
    selected = self.indexer(
        hidden_states, q_residual, position_ids, past_key_values, layer_idx
    )
    return CompressedAttentionState(keys, self.compress_rate, selected)


def _online_attention_update(
    query: torch.Tensor,
    keys: torch.Tensor,
    logits: torch.Tensor,
    running_max: torch.Tensor,
    running_sum: torch.Tensor,
    running_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block_max = logits.amax(dim=-1)
    next_max = torch.maximum(running_max, block_max)
    old_scale = torch.exp(running_max - next_max)
    block_exp = torch.exp(logits - next_max.unsqueeze(-1))
    block_exp = torch.nan_to_num(block_exp, nan=0.0, posinf=0.0, neginf=0.0)
    next_sum = running_sum * old_scale + block_exp.sum(dim=-1)
    if keys.ndim == 3:
        contribution = torch.einsum("bhqk,bkd->bhqd", block_exp, keys.float())
    else:
        contribution = torch.einsum("bhqk,bqkd->bhqd", block_exp, keys.float())
    next_value = running_value * old_scale.unsqueeze(-1) + contribution
    return next_max, next_sum, next_value


def _compressed_attention(
    module,
    query: torch.Tensor,
    sliding_kv: torch.Tensor,
    compressed: CompressedAttentionState,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    batch, heads, sequence, head_dim = query.shape
    outputs: list[torch.Tensor] = []
    for query_start in range(0, sequence, _INDEX_QUERY_BLOCK):
        query_stop = min(sequence, query_start + _INDEX_QUERY_BLOCK)
        q = query[:, :, query_start:query_stop].float()
        q_count = query_stop - query_start
        sink = module.sinks.float().view(1, heads, 1).expand(batch, heads, q_count)
        running_max = sink
        running_sum = torch.ones_like(sink)
        running_value = torch.zeros(
            (batch, heads, q_count, head_dim), dtype=torch.float32, device=q.device
        )
        sliding_logits = torch.einsum(
            "bhqd,bkd->bhqk", q, sliding_kv[:, 0].float()
        ) * module.scaling
        if attention_mask is not None:
            sliding_logits = sliding_logits + attention_mask[
                :, :, query_start:query_stop, : sliding_kv.shape[2]
            ].float()
        running_max, running_sum, running_value = _online_attention_update(
            q,
            sliding_kv[:, 0],
            sliding_logits,
            running_max,
            running_sum,
            running_value,
        )
        if compressed.selected_indices is None:
            threshold = (
                position_ids[:, query_start:query_stop] + 1
            ) // compressed.compress_rate
            for key_start in range(0, compressed.keys.shape[1], _ATTENTION_KEY_BLOCK):
                key_stop = min(compressed.keys.shape[1], key_start + _ATTENTION_KEY_BLOCK)
                keys = compressed.keys[:, key_start:key_stop]
                logits = torch.einsum("bhqd,bkd->bhqk", q, keys.float()) * module.scaling
                positions = torch.arange(key_start, key_stop, device=q.device)
                logits.masked_fill_(
                    positions.view(1, 1, 1, -1)
                    >= threshold.unsqueeze(1).unsqueeze(-1),
                    float("-inf"),
                )
                running_max, running_sum, running_value = _online_attention_update(
                    q,
                    keys,
                    logits,
                    running_max,
                    running_sum,
                    running_value,
                )
        else:
            selected = compressed.selected_indices[:, query_start:query_stop]
            valid = selected >= 0
            safe = selected.clamp_min(0)
            batch_index = torch.arange(batch, device=q.device).view(batch, 1, 1)
            keys = compressed.keys[batch_index, safe]
            logits = torch.einsum("bhqd,bqkd->bhqk", q, keys.float()) * module.scaling
            logits.masked_fill_(~valid.unsqueeze(1), float("-inf"))
            running_max, running_sum, running_value = _online_attention_update(
                q,
                keys,
                logits,
                running_max,
                running_sum,
                running_value,
            )
        outputs.append((running_value / running_sum.unsqueeze(-1)).to(query.dtype))
    return torch.cat(outputs, dim=2).transpose(1, 2).contiguous()


def _attention_forward_native(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    **_kwargs,
):
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings[self.rope_layer_type]
    q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
    q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
    q = apply_rotary_pos_emb(self.q_b_norm(q), cos, sin)
    kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
    kv = apply_rotary_pos_emb(kv, cos, sin)
    if past_key_values is not None:
        kv = past_key_values.update(kv, kv, self.layer_idx)[0]
    if self.compressor is None:
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import eager_attention_forward

        output, weights = eager_attention_forward(
            self,
            q,
            kv,
            kv,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0,
        )
    else:
        compressed = self.compressor(
            hidden_states, q_residual, position_ids, past_key_values, self.layer_idx
        )
        output = _compressed_attention(
            self, q, kv, compressed, attention_mask, position_ids
        )
        weights = None
    output = apply_rotary_pos_emb(output.transpose(1, 2), cos, -sin).transpose(1, 2)
    grouped = output.reshape(*input_shape, self.config.o_groups, -1)
    return self.o_b_proj(self.o_a_proj(grouped).flatten(2)), weights


def install_deepseek_v4_fp4_indexer(model: torch.nn.Module) -> int:
    installed = 0
    for module in model.modules():
        if (
            type(module).__name__ == "DeepseekV4Indexer"
            and hasattr(module, "scorer")
            and hasattr(module, "q_b_proj")
        ):
            module.forward = MethodType(_indexer_forward_fp4, module)
            installed += 1
        elif type(module).__name__ == "DeepseekV4HCACompressor":
            module.forward = MethodType(_compress_hca, module)
        elif type(module).__name__ == "DeepseekV4CSACompressor":
            module.forward = MethodType(_compress_csa, module)
        elif type(module).__name__ == "DeepseekV4Attention":
            module.forward = MethodType(_attention_forward_native, module)
    if installed < 1:
        raise RuntimeError("DeepSeek V4 model contains no CSA Lightning Indexer")
    return installed


__all__ = [
    "DeepseekV4FP4CSACache",
    "PackedFP4IndexerCache",
    "install_deepseek_v4_fp4_indexer",
    "make_deepseek_v4_cache",
]
