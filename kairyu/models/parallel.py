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

from kairyu.engine.core.quant_config import (
    QuantConfig,
    QuantMethod,
    detect_quantization,
    validate_tensor_parallel_quantization,
)
from kairyu.models.config import ModelConfig, validate_tensor_parallel_config


def shard_bounds(total: int, world_size: int, rank: int) -> tuple[int, int]:
    """Contiguous equal shards; fail-fast on indivisibility (gloo all_gather
    rejects unequal shapes)."""
    if total % world_size != 0:
        raise ValueError(f"{total} does not divide evenly across {world_size} ranks")
    span = total // world_size
    return rank * span, (rank + 1) * span


def tp_view(config: ModelConfig, tp: int, rank: int) -> ModelConfig:
    """The rank-local config (A2): the whole model tree sizes itself from it."""
    validate_tensor_parallel_config(config, tp)
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


class SequenceParallelContext:
    """Shared state for one model's sequence-parallel region (Megatron TP+SP).

    The residual stream between blocks is sharded along TOKENS, so the norms
    hold S/tp rows instead of S. Attention and the MLP still need every token,
    so a norm's output is all_gathered on the way in and the row-parallel output
    is reduce_scattered on the way out. Traffic is unchanged — all_gather +
    reduce_scatter moves what one all_reduce does, and `bench/reduce_scatter_bench.py`
    measures the pair at ~0.96x an all_reduce here. The gain is ACTIVATION
    MEMORY, not latency, and enabling this for speed would be a mistake.

    ``padding`` tracks the rows added to make the token count divisible, so the
    final gather can trim back to the real sequence.
    """

    def __init__(self, comm) -> None:
        self.comm = comm
        self.padding = 0

    @property
    def world_size(self) -> int:
        return self.comm.world_size

    def scatter(self, x: torch.Tensor) -> torch.Tensor:
        """Full [S, H] -> this rank's [ceil(S/tp), H], padding when ragged."""
        world = self.world_size
        remainder = x.shape[0] % world
        self.padding = (world - remainder) % world
        if self.padding:
            x = torch.cat([x, x.new_zeros((self.padding, *x.shape[1:]))], dim=0)
        span = x.shape[0] // world
        start = self.comm.rank * span
        return x[start : start + span].contiguous()

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's shard -> the REAL sequence, padding removed.

        The TP region has to see the true token count: attention builds its mask
        from it, and the residual add downstream is against a real-length
        tensor. The padding only exists to make the shard split even, so it is
        added on the way in and removed on the way out.
        """
        full = self.comm.tensor_all_gather(x.contiguous())
        return full[: full.shape[0] - self.padding] if self.padding else full

    def reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        """Full [S, H] partial sums -> this rank's reduced token shard.

        Re-pads first: reduce_scatter needs a count divisible by the world size,
        and S is whatever the caller's real sequence is.
        """
        if self.padding:
            x = torch.cat([x, x.new_zeros((self.padding, *x.shape[1:]))], dim=0)
        return self.comm.tensor_reduce_scatter(x.contiguous())


class SequenceParallelNorm(nn.Module):
    """Norm on the token shard, then all_gather for the TP region.

    RMSNorm/LayerNorm are per-token, so running them on a shard is exactly the
    same arithmetic — that is what makes the sharded residual stream valid.
    """

    def __init__(self, norm: nn.Module, context: SequenceParallelContext) -> None:
        super().__init__()
        self.norm = norm
        self._context = context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._context.gather(self.norm(x))


class ScatterAfterEmbedding(nn.Module):
    """Entry to the sequence-parallel region: full embedding -> token shard."""

    def __init__(self, embedding: nn.Module, context: SequenceParallelContext) -> None:
        super().__init__()
        self.embedding = embedding
        self._context = context

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._context.scatter(self.embedding(token_ids))

    @property
    def weight(self) -> torch.Tensor:  # tie_word_embeddings reads this
        return self.embedding.weight


class GatherBeforeNorm(nn.Module):
    """Exit: the final norm sees the whole sequence again, for the lm_head."""

    def __init__(self, norm: nn.Module, context: SequenceParallelContext) -> None:
        super().__init__()
        self.norm = norm
        self._context = context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._context.gather(self.norm(x))


class RowParallelLinear(nn.Module):
    """Wraps a rank-local Linear: reduce the partial output, bias once.

    ``context`` switches the reduction from all_reduce to reduce_scatter, which
    both sums across ranks AND re-shards along tokens — the exit from the TP
    region back into the sequence-parallel one.
    """

    def __init__(
        self, local: nn.Module, comm, context: SequenceParallelContext | None = None
    ) -> None:
        super().__init__()
        self.local = local
        self._comm = comm
        self._context = context
        # detach bias from the local matmul: it must be added AFTER the reduce
        self._bias = local.bias
        local.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from kairyu.quant.linear import Fp8Linear

        if isinstance(self.local, Fp8Linear) and self.local.activation_dynamic:
            # A row-parallel rank sees only its input-feature shard. Dynamic
            # W8A8 quantization must nevertheless use the same per-token scale
            # the unsharded activation would use; rank-local scales change the
            # numeric contract with TP degree. Synchronize only the M amax
            # values, then quantize each local feature shard with that global
            # scale (the extra collective is tiny relative to the output sum).
            local_amax = self.local.activation_amax(x).contiguous()
            global_amax = self._comm.tensor_all_reduce_max(local_amax)
            activation_scale = (global_amax / 448.0).clamp(min=1e-12)
            partial = self.local.forward_with_activation_scale(
                x, activation_scale
            )
        else:
            partial = self.local(x)
        if self._context is None:
            reduced = self._comm.tensor_all_reduce(partial.contiguous())
        else:
            reduced = self._context.reduce_scatter(partial)
        if self._bias is not None:
            reduced = reduced + self._bias
        return reduced


def _checkpoint_shard_dim(
    name: str,
    spec: dict[str, int | None],
    quant: QuantConfig,
) -> int | None:
    """Resolve the checkpoint tensor axis owned by one TP rank.

    FP8 weights follow the dense projection axis. Their per-output-channel
    scale follows a column-parallel weight (dim 0), but is replicated for a
    row-parallel weight (dim 1) because every rank still produces every output
    channel.
    """
    dim = shard_dim_for(name, spec)
    if quant.method is QuantMethod.FP8 and name.endswith(".weight_scale"):
        weight_dim = shard_dim_for(name.removesuffix("_scale"), spec)
        return 0 if weight_dim == 0 else None
    return dim


def load_tp_shard(
    model: nn.Module,
    config: ModelConfig,
    reader,
    tp: int,
    rank: int,
    dtype: torch.dtype = torch.float32,
    quant: QuantConfig | None = None,
) -> None:
    """Per-rank weights via CheckpointReader.get_slice (the m8 seam).

    Bounds are computed from the FULL config. Dense and FP8 projection weights
    shard on their projection axis; FP8 scales shard only with output-channel
    (column-parallel) weights. Embeddings, lm_head, norms, row-parallel scales,
    and biases are replicated.
    """
    quant = quant or QuantConfig(QuantMethod.NONE)
    spec = tp_shard_spec(config)
    state: dict[str, torch.Tensor] = {}
    expected = model.state_dict()
    quantized_buffer_dtypes = {
        (
            f"{module_name}.{buffer_name}" if module_name else buffer_name
        ): buffer.dtype
        for module_name, module in model.named_modules()
        if getattr(module, "is_quantized", False)
        for buffer_name, buffer in module.named_buffers(recurse=False)
    }
    for name, current in expected.items():
        if name == "lm_head.weight" and config.tie_word_embeddings:
            continue  # re-tied to the LOCAL embed shard after load (A3)
        source = name
        if source not in reader:
            raise KeyError(f"checkpoint missing tensor {source!r}")
        dim = _checkpoint_shard_dim(name, spec, quant)
        if dim is None:
            tensor = reader.tensor(source)
        else:
            # current is rank-local; the FULL size along dim is local * tp
            total = current.shape[dim] * tp
            start, end = shard_bounds(total, tp, rank)
            tensor = reader.get_slice(source, dim=dim, start=start, end=end)
        quantized_dtype = quantized_buffer_dtypes.get(name)
        if quantized_dtype is not None and tensor.dtype != quantized_dtype:
            # Match the quantized module's fused-kernel ABI rather than the
            # model compute dtype. Real FP8 checkpoints commonly store scale
            # values as bf16; scaled_mm requires the same values in fp32.
            tensor = tensor.to(quantized_dtype)
        elif (
            name not in quantized_buffer_dtypes
            and current.dtype == torch.float32
            and tensor.is_floating_point()
        ):
            tensor = tensor.to(dtype)
        state[name] = tensor
    model.load_state_dict(state, strict=False, assign=True)
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.eval()


def build_tp_model(
    model_dir: str,
    tp: int,
    rank: int,
    comm,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    attention_backend=None,
    sequence_parallel: bool = False,
):
    """Rank-sharded DenseDecoder: tp_view config + row-parallel/gathered wrappers.

    ``dtype``/``device`` place the shard exactly like the single-process path
    places the whole model (bf16 on-device for GPU, fp32 on host for CPU); the
    defaults keep every CPU test byte-for-byte unchanged.
    """
    import json
    from pathlib import Path

    from kairyu.engine.core.weights import CheckpointReader
    from kairyu.models.config import parse_model_config
    from kairyu.models.llama import DenseDecoder

    raw = json.loads((Path(model_dir) / "config.json").read_text())
    quant = detect_quantization(raw)
    validate_tensor_parallel_quantization(quant)
    full_config = parse_model_config(raw)
    local_config = tp_view(full_config, tp, rank)
    if sequence_parallel and tp < 2:
        raise ValueError("sequence parallelism needs tp >= 2")
    from kairyu.quant.linear import linear_factory

    model = DenseDecoder(
        local_config,
        attention_backend=attention_backend,
        linear_factory=linear_factory(quant),
        dtype=dtype,
    )
    reader = CheckpointReader(model_dir)
    # dtype is applied while loading, so a bf16 target never materializes the
    # fp32 shard on the host first
    load_tp_shard(
        model,
        full_config,
        reader,
        tp,
        rank,
        dtype=dtype,
        quant=quant,
    )
    # add communication: o_proj/down_proj partial sums (lm_head replicated)
    context = SequenceParallelContext(comm) if sequence_parallel else None
    for layer in model.model.layers:
        layer.self_attn.o_proj = RowParallelLinear(layer.self_attn.o_proj, comm, context)
        layer.mlp.down_proj = RowParallelLinear(layer.mlp.down_proj, comm, context)
        if context is not None:
            # norms run on the shard and gather on the way into the TP region
            layer.input_layernorm = SequenceParallelNorm(layer.input_layernorm, context)
            layer.post_attention_layernorm = SequenceParallelNorm(
                layer.post_attention_layernorm, context
            )
    if context is not None:
        model.model.embed_tokens = ScatterAfterEmbedding(model.model.embed_tokens, context)
        model.model.norm = GatherBeforeNorm(model.model.norm, context)
    # after the wrappers, so RowParallelLinear's detached bias moves too
    return model.to(device), local_config, full_config
