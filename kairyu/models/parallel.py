"""Tensor-parallel sharding: pre-sharded config + canonical bindings (m16 D2).

Shard ownership (review A2): the rank model is built from ``tp_view(config,
tp, rank)`` — heads/kv-heads/intermediate divided — so every module comes out
rank-local for free (Attention's reshapes, the kv pool sizing). The parallel
bindings only ADD communication on the original parameter-owning modules:
``RowParallelLinear`` all_reduces an unbiased local output and then adds the
canonical bias exactly once (A4); the TP logits head
all_gathers vocab shards (gloo rejects unequal shapes →
``vocab_size % tp == 0`` fail-fast, A3).
Shard-loading bounds come from the FULL config so ``get_slice`` rows align to
whole heads. Embeddings and lm_head are REPLICATED in M16 (every rank holds
full logits, but rank 0 alone runs the stateful sampler and broadcasts the
selected device-token packet); vocab-parallel sharding is a deploy-day memory
optimization behind the same single-owner seam.
"""

from __future__ import annotations

import dataclasses
from types import MethodType

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


@dataclasses.dataclass(frozen=True)
class ParallelShardInfo:
    """Rank-local checkpoint ownership attached to parallelized modules."""

    kind: str
    rank: int
    world_size: int

    def __post_init__(self) -> None:
        if self.kind not in {"tensor", "expert"}:
            raise ValueError(f"unknown parallel shard kind {self.kind!r}")
        if self.world_size < 1:
            raise ValueError("parallel shard world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"parallel shard rank {self.rank} is outside world_size {self.world_size}"
            )


class UnsupportedParallelSerializationError(RuntimeError):
    """A rank-local tensor set was requested as a complete HF checkpoint."""

    def __init__(
        self,
        mode: str,
        topology: tuple[tuple[str, ParallelShardInfo], ...],
    ) -> None:
        self.mode = mode
        self.topology = topology
        details = ", ".join(
            f"{info.kind}@{name}:rank={info.rank}/{info.world_size}"
            for name, info in topology
        )
        super().__init__(
            f"{mode} serialization is unsupported from one parallel rank "
            f"({details}); use mode='rank-local' for a same-topology "
            "round-trip or gather canonical shards from every rank"
        )


def parallel_shard_info(
    model: nn.Module,
) -> tuple[tuple[str, ParallelShardInfo], ...]:
    """Return deterministic rank-local ownership metadata for introspection."""

    found = []
    for name, module in model.named_modules():
        info = getattr(module, "_kairyu_parallel_shard", None)
        if isinstance(info, ParallelShardInfo):
            found.append((name or "<root>", info))
    return tuple(found)


def parallel_state_dict(
    model: nn.Module,
    *,
    mode: str = "rank-local",
    keep_vars: bool = False,
) -> dict[str, torch.Tensor]:
    """Enumerate canonical state with an explicit serialization boundary.

    ``rank-local`` is loadable only into the same rank/topology and preserves
    native HF names. A complete ``full-hf`` export needs TP concatenation, EP
    union, replicated-value checks, and tied-weight restoration; until that
    collective exporter exists, returning one rank's shards would be data
    corruption and therefore fails deterministically.
    """

    if mode not in {"rank-local", "full-hf"}:
        raise ValueError("parallel state mode must be 'rank-local' or 'full-hf'")
    topology = parallel_shard_info(model)
    sharded = tuple(item for item in topology if item[1].world_size > 1)
    if mode == "full-hf" and sharded:
        raise UnsupportedParallelSerializationError(mode, sharded)
    return model.state_dict(keep_vars=keep_vars)


@dataclasses.dataclass(frozen=True)
class CheckpointMemberSpec:
    """Canonical source and physical shard axis for one rank-local tensor."""

    source_name: str
    shard_dim: int | None


def checkpoint_member_specs(model: nn.Module) -> dict[str, CheckpointMemberSpec]:
    """Derive checkpoint slicing from each projection's typed #228 context.

    No layer-name suffix table participates. The canonical projection identity
    and TP placement live on the parameter-owning module itself, so arbitrary
    model prefixes, quantization metadata, post-bind loading, and adapters all
    agree on one native module tree. Unknown persistent members on a
    contextual projection fail closed instead of being silently replicated.
    """

    from kairyu.quant.linear import QuantizedLinearBase

    expected = model.state_dict()
    specs = {
        name: CheckpointMemberSpec(name, None)
        for name in expected
    }
    for module_name, module in model.named_modules():
        context = getattr(module, "linear_context", None)
        if context is None:
            continue
        canonical = context.qualified_name
        if module_name != canonical:
            raise RuntimeError(
                "parallel projection path disagrees with its checkpoint identity: "
                f"module={module_name!r}, canonical={canonical!r}"
            )
        prefix = f"{module_name}." if module_name else ""
        direct_members = {
            name[len(prefix) :]
            for name in expected
            if name.startswith(prefix) and "." not in name[len(prefix) :]
        }
        if isinstance(module, QuantizedLinearBase):
            layouts = module.checkpoint_member_layouts()
        else:
            from kairyu.quant.linear import CheckpointMemberLayout

            layouts = {
                "weight": CheckpointMemberLayout(column_dim=0, row_dim=1),
                "bias": CheckpointMemberLayout(column_dim=0, row_dim=None),
            }
        layouts = {
            **layouts,
            **getattr(module, "_kairyu_checkpoint_member_layouts", {}),
        }
        unknown = sorted(direct_members - layouts.keys())
        if unknown:
            raise RuntimeError(
                f"contextual projection {canonical!r} has no checkpoint "
                f"shard contract for persistent members {unknown}"
            )

        placement = context.tensor_parallel
        for member in direct_members:
            shard_dim = layouts[member].shard_dim(placement.mode)
            state_name = f"{prefix}{member}"
            specs[state_name] = CheckpointMemberSpec(
                f"{canonical}.{member}",
                shard_dim,
            )
    return specs


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


class _SequenceOutputBinding:
    """Forward behavior attached to a canonical embedding/norm module."""

    def __init__(self, context: SequenceParallelContext, *, scatter: bool) -> None:
        self.context = context
        self.scatter = scatter

    def __call__(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self.context.scatter(output) if self.scatter else self.context.gather(output)


class _SequenceInputBinding:
    """Gather a sequence shard before the canonical final norm runs."""

    def __init__(self, context: SequenceParallelContext) -> None:
        self.context = context

    def __call__(
        self,
        _module: nn.Module,
        inputs: tuple[object, ...],
    ) -> tuple[object, ...]:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise TypeError("sequence-parallel norm expects a tensor input")
        return (self.context.gather(inputs[0]), *inputs[1:])


def _bind_sequence_output(
    module: nn.Module,
    context: SequenceParallelContext,
    *,
    scatter: bool,
) -> nn.Module:
    marker = "_kairyu_sequence_scatter" if scatter else "_kairyu_sequence_gather"
    if getattr(module, marker, None) is not None:
        raise ValueError(f"{type(module).__name__} already has {marker} behavior")
    binding = _SequenceOutputBinding(context, scatter=scatter)
    module.register_forward_hook(binding)
    object.__setattr__(module, marker, binding)
    return module


def SequenceParallelNorm(
    norm: nn.Module,
    context: SequenceParallelContext,
) -> nn.Module:
    """Bind post-norm all-gather without inserting a synthetic child path."""

    return _bind_sequence_output(norm, context, scatter=False)


def ScatterAfterEmbedding(
    embedding: nn.Module,
    context: SequenceParallelContext,
) -> nn.Module:
    """Bind embedding scatter while keeping ``embed_tokens.weight`` canonical."""

    return _bind_sequence_output(embedding, context, scatter=True)


def GatherBeforeNorm(
    norm: nn.Module,
    context: SequenceParallelContext,
) -> nn.Module:
    """Bind pre-norm gather while keeping the final norm at its HF path."""

    marker = "_kairyu_sequence_gather_input"
    if getattr(norm, marker, None) is not None:
        raise ValueError(f"{type(norm).__name__} already has {marker} behavior")
    binding = _SequenceInputBinding(context)
    norm.register_forward_pre_hook(binding)
    object.__setattr__(norm, marker, binding)
    return norm


class _RowParallelBinding:
    """Reduce a projection output while the projection keeps tensor ownership."""

    def __init__(
        self,
        comm,
        context: SequenceParallelContext | None,
    ) -> None:
        self.comm = comm
        self.context = context

    def __call__(
        self,
        module: nn.Module,
        _inputs: tuple[object, ...],
        partial: torch.Tensor,
    ) -> torch.Tensor:
        if self.context is None:
            reduced = self.comm.tensor_all_reduce(partial.contiguous())
        else:
            reduced = self.context.reduce_scatter(partial)
        # The canonical bias stays registered on the projection so state_dict,
        # named_parameters, and adapters all see ``...proj.bias``. Its local
        # forward omits bias, preserving the established numerical contract:
        # reduce unbiased partials, then add one bias copy.
        bias = getattr(module, "bias", None)
        if bias is not None:
            reduced = reduced + bias
        return reduced


def _dense_row_parallel_forward(
    module: nn.Linear,
    x: torch.Tensor,
) -> torch.Tensor:
    """Dense local GEMM without bias; the reduction binding owns bias add."""

    return nn.functional.linear(x, module.weight, None)


def RowParallelLinear(
    local: nn.Module,
    comm,
    context: SequenceParallelContext | None = None,
) -> nn.Module:
    """Bind row-parallel execution to the canonical projection module.

    This compatibility constructor intentionally returns ``local`` itself.
    Registering it as a wrapper child would change every checkpoint,
    ``named_*`` and adapter path to ``...proj.local.*``. Dynamic FP8 keeps its
    global per-token scale through the communicator bound here.
    """

    if getattr(local, "_kairyu_row_parallel", None) is not None:
        raise ValueError(f"{type(local).__name__} is already row-parallel")

    bias = getattr(local, "bias", None)
    if bias is not None:
        from kairyu.quant.linear import QuantizedLinearBase

        if type(local) is nn.Linear:
            object.__setattr__(
                local,
                "forward",
                MethodType(_dense_row_parallel_forward, local),
            )
        elif isinstance(local, QuantizedLinearBase):
            object.__setattr__(local, "_kairyu_row_parallel_omit_bias", True)
        else:
            raise TypeError(
                f"{type(local).__name__} cannot omit its local bias for "
                "row-parallel reduction"
            )
    binding = _RowParallelBinding(comm, context)
    local.register_forward_hook(binding)
    object.__setattr__(local, "_kairyu_row_parallel", binding)

    from kairyu.quant.linear import Fp8Linear

    if isinstance(local, Fp8Linear) and local.activation_dynamic:
        object.__setattr__(local, "_row_parallel_comm", comm)
    return local


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
    state: dict[str, torch.Tensor] = {}
    expected = model.state_dict()
    member_specs = checkpoint_member_specs(model)
    quantized_buffer_dtypes = {
        (f"{module_name}.{buffer_name}" if module_name else buffer_name): buffer.dtype
        for module_name, module in model.named_modules()
        if getattr(module, "is_quantized", False)
        for buffer_name, buffer in module.named_buffers(recurse=False)
    }
    for name, current in expected.items():
        if name == "lm_head.weight" and config.tie_word_embeddings:
            continue  # re-tied to the LOCAL embed shard after load (A3)
        member = member_specs[name]
        source = member.source_name
        if source not in reader:
            raise KeyError(f"checkpoint missing tensor {source!r}")
        dim = member.shard_dim
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
    linear_capabilities=None,
    linear_selection_policy=None,
):
    """Rank-sharded DenseDecoder: tp_view config + canonical TP/SP bindings.

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
        linear_factory=linear_factory(
            quant,
            device=device,
            dtype=dtype,
            tensor_parallel_size=tp,
            tensor_parallel_rank=rank,
            capabilities=linear_capabilities,
            selection_policy=linear_selection_policy,
        ),
        dtype=dtype,
    )
    # Bind execution behavior before loading. Parameter-owning modules stay at
    # their HF/checkpoint paths, so post-bind state enumeration and canonical
    # checkpoint slicing use exactly the same tree.
    context = SequenceParallelContext(comm) if sequence_parallel else None
    for layer in model.model.layers:
        RowParallelLinear(layer.self_attn.o_proj, comm, context)
        RowParallelLinear(layer.mlp.down_proj, comm, context)
        if context is not None:
            # Norms run on the token shard and gather on the way into TP.
            SequenceParallelNorm(layer.input_layernorm, context)
            SequenceParallelNorm(layer.post_attention_layernorm, context)
    if context is not None:
        ScatterAfterEmbedding(model.model.embed_tokens, context)
        GatherBeforeNorm(model.model.norm, context)
    object.__setattr__(
        model,
        "_kairyu_parallel_shard",
        ParallelShardInfo("tensor", rank, tp),
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
    return model.to(device), local_config, full_config
