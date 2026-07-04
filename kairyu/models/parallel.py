"""Tensor-parallel sharding: pre-sharded config + communication wrappers (m16 D2).

Shard ownership (review A2): the rank model is built from ``tp_view(config,
tp, rank)`` — heads/kv-heads/intermediate divided — so every module comes out
rank-local for free (Attention's reshapes, the kv pool sizing). The parallel
wrappers only ADD communication: ``RowParallelLinear`` all_reduces its output
(bias added ONCE, after the reduce — A4); the TP logits head all_gathers vocab
shards (gloo rejects unequal shapes → ``vocab_size % tp == 0`` fail-fast, A3).
Shard-loading bounds come from the FULL config so ``get_slice`` rows align to
whole heads. Embeddings and lm_head are REPLICATED in M16 (every rank holds
full logits → every rank samples identically, keeping the m5 D1 invariant
with zero gather traffic); vocab-parallel sharding is a deploy-day memory
optimization behind the same seam.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from kairyu.engine.core.quant_config import QuantMethod, detect_quantization
from kairyu.engine.core.tp_runner import validate_tp_degree
from kairyu.models.config import ModelConfig


def shard_bounds(total: int, world_size: int, rank: int) -> tuple[int, int]:
    """Contiguous equal shards; fail-fast on indivisibility (gloo all_gather
    rejects unequal shapes)."""
    if total % world_size != 0:
        raise ValueError(f"{total} does not divide evenly across {world_size} ranks")
    span = total // world_size
    return rank * span, (rank + 1) * span


def tp_view(config: ModelConfig, tp: int, rank: int) -> ModelConfig:
    """The rank-local config (A2): the whole model tree sizes itself from it."""
    validate_tp_degree(tp, num_kv_heads=config.num_key_value_heads)
    if config.num_attention_heads % tp != 0:
        raise ValueError(
            f"num_attention_heads={config.num_attention_heads} not divisible by tp={tp}"
        )
    if config.intermediate_size % tp != 0:
        raise ValueError(
            f"intermediate_size={config.intermediate_size} not divisible by tp={tp}"
        )
    if config.vocab_size % tp != 0:
        raise ValueError(f"vocab_size={config.vocab_size} not divisible by tp={tp}")
    if config.is_mla:
        raise ValueError("TP for MLA models is not supported (attention-DP, m16 §3)")
    if config.moe is not None:
        # sparse MoE layers have no `mlp.down_proj` to row-parallelize (M4); MoE
        # is distributed by expert parallelism, not this dense-MLP TP path — fail
        # fast instead of loading every shard and then AttributeError-ing
        raise ValueError(
            "TP for MoE models is not supported; use expert parallelism (m16 EP)"
        )
    return dataclasses.replace(
        config,
        num_attention_heads=config.num_attention_heads // tp,
        num_key_value_heads=config.num_key_value_heads // tp,
        intermediate_size=config.intermediate_size // tp,
    )


# name -> (shard dim, sizing) rules against the FULL config; None = replicated
def tp_shard_spec(config: ModelConfig) -> dict[str, int | None]:
    """Parameter-name suffix -> shard dim (0 = out/vocab rows, 1 = in columns)."""
    return {
        "self_attn.q_proj.weight": 0,
        "self_attn.q_proj.bias": 0,
        "self_attn.k_proj.weight": 0,
        "self_attn.k_proj.bias": 0,
        "self_attn.v_proj.weight": 0,
        "self_attn.v_proj.bias": 0,
        "self_attn.o_proj.weight": 1,  # row-parallel: shard in_features
        # o_proj.bias replicated: added once after the all_reduce (A4)
        "mlp.gate_proj.weight": 0,
        "mlp.up_proj.weight": 0,
        "mlp.down_proj.weight": 1,
        # embed_tokens / lm_head replicated (full logits on every rank)
    }


def shard_dim_for(name: str, spec: dict[str, int | None]) -> int | None:
    for suffix, dim in spec.items():
        if name.endswith(suffix):
            return dim
    return None


class RowParallelLinear(nn.Module):
    """Wraps a rank-local Linear: all_reduce the partial output, bias once."""

    def __init__(self, local: nn.Linear, comm) -> None:
        super().__init__()
        self.local = local
        self._comm = comm
        # detach bias from the local matmul: it must be added AFTER the reduce
        self._bias = local.bias
        local.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        partial = self.local(x)
        reduced = self._comm.tensor_all_reduce(partial.contiguous())
        if self._bias is not None:
            reduced = reduced + self._bias
        return reduced


def load_tp_shard(model: nn.Module, config: ModelConfig, reader, tp: int, rank: int,
                  dtype: torch.dtype = torch.float32) -> None:
    """Per-rank weights via CheckpointReader.get_slice (the m8 seam).

    Bounds computed from the FULL config; embeddings/lm_head shard vocab rows;
    norms and biasless row-parallel load replicated. Quantized checkpoints are
    rejected upstream (A10).
    """
    spec = tp_shard_spec(config)
    state: dict[str, torch.Tensor] = {}
    expected = model.state_dict()
    for name, current in expected.items():
        if name == "lm_head.weight" and config.tie_word_embeddings:
            continue  # re-tied to the LOCAL embed shard after load (A3)
        source = name
        if source not in reader:
            raise KeyError(f"checkpoint missing tensor {source!r}")
        dim = shard_dim_for(name, spec)
        if dim is None:
            tensor = reader.tensor(source)
        else:
            # current is rank-local; the FULL size along dim is local * tp
            total = current.shape[dim] * tp
            start, end = shard_bounds(total, tp, rank)
            tensor = reader.get_slice(source, dim=dim, start=start, end=end)
        if current.dtype == torch.float32 and tensor.is_floating_point():
            tensor = tensor.to(dtype)
        state[name] = tensor
    model.load_state_dict(state, strict=False, assign=True)
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.eval()


def build_tp_model(model_dir: str, tp: int, rank: int, comm):
    """Rank-sharded DenseDecoder: tp_view config + row-parallel/gathered wrappers."""
    import json
    from pathlib import Path

    from kairyu.engine.core.weights import CheckpointReader
    from kairyu.models.config import parse_model_config
    from kairyu.models.llama import DenseDecoder

    raw = json.loads((Path(model_dir) / "config.json").read_text())
    if detect_quantization(raw).method is not QuantMethod.NONE:
        raise ValueError("quantized checkpoints with tensor parallelism arrive later (m16 A10)")
    full_config = parse_model_config(raw)
    local_config = tp_view(full_config, tp, rank)
    model = DenseDecoder(local_config)
    reader = CheckpointReader(model_dir)
    load_tp_shard(model, full_config, reader, tp, rank)
    # add communication: o_proj/down_proj partial sums (lm_head replicated)
    for layer in model.model.layers:
        layer.self_attn.o_proj = RowParallelLinear(layer.self_attn.o_proj, comm)
        layer.mlp.down_proj = RowParallelLinear(layer.mlp.down_proj, comm)
    return model, local_config, full_config
