"""Expert-parallel dispatch/combine over all_to_all (m16 D3).

Routing runs REPLICATED (fp32, deterministic on CPU/gloo; a deploy-day debug
guard hashes topk_indices across ranks — m16 A8). Tokens permute to
expert-owning ranks via ``tensor_all_to_all_single`` (counts exchange first,
then payload), local experts compute, reverse all_to_all, weighted combine
locally. Contiguous expert blocks per rank; the math is the m15 token-loop's,
algebraically identical (accumulation order differs — parity gates use token
equality, m16 A7). gloo and NCCL share this code path; DeepEP/UCCL is the
deploy-day fast path behind the same block interface.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from kairyu.models.moe import (
    apply_nvfp4_moe_global_input_scales,
    route_experts,
)
from kairyu.models.parallel import ParallelShardInfo
from kairyu.quant.linear import (
    ExpertScope,
    LinearRole,
    ModelScope,
    NvFp4Linear,
)


class RemoteExpertOwnershipError(LookupError):
    """A canonical expert exists, but its tensors belong to another rank."""


class _RemoteExpertProjection(nn.Module):
    """Allocation-free construction placeholder removed before execution."""

    def __init__(self, qualified_name: str, owner_rank: int) -> None:
        super().__init__()
        self.qualified_name = qualified_name
        self.owner_rank = owner_rank

    def forward(self, _hidden: torch.Tensor) -> torch.Tensor:
        raise RemoteExpertOwnershipError(
            f"projection {self.qualified_name!r} belongs to expert rank "
            f"{self.owner_rank}"
        )


class _ExpertShardedLinearFactory:
    """Construct checkpoint modules only for the current rank's experts."""

    def __init__(
        self,
        base,
        *,
        num_experts: int,
        ep_rank: int,
        ep_size: int,
    ) -> None:
        if num_experts % ep_size:
            raise ValueError(
                f"{num_experts} experts do not divide across {ep_size} ranks"
            )
        self._base = base
        self._num_experts = num_experts
        self._ep_rank = ep_rank
        self._experts_per_rank = num_experts // ep_size

    def __call__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        *,
        qualified_name: str | None = None,
        model_scope: ModelScope = ModelScope.TARGET,
        role: LinearRole | None = None,
        layer_index: int | None = None,
        expert_index: int | None = None,
        expert_scope: ExpertScope = ExpertScope.NONE,
        shard_dim: int | None = None,
        allow_quantization: bool = True,
    ) -> nn.Module:
        if expert_scope is ExpertScope.ROUTED:
            if expert_index is None or not 0 <= expert_index < self._num_experts:
                raise ValueError("routed expert construction requires a valid global index")
            owner = expert_index // self._experts_per_rank
            if owner != self._ep_rank:
                if qualified_name is None:
                    raise ValueError("remote expert projection requires a checkpoint name")
                return _RemoteExpertProjection(qualified_name, owner)
        return self._base(
            in_features,
            out_features,
            bias,
            qualified_name=qualified_name,
            model_scope=model_scope,
            role=role,
            layer_index=layer_index,
            expert_index=expert_index,
            expert_scope=expert_scope,
            shard_dim=shard_dim,
            allow_quantization=allow_quantization,
        )


@dataclass(frozen=True)
class ExpertParallelLoadInfo:
    """Fail-closed provenance from one rank-local expert load."""

    ep_rank: int
    ep_size: int
    owned_expert_indices: tuple[int, ...]
    quantization_source: str
    quantization_method: str
    kv_cache_quant_algo: str | None
    producer_name: str | None
    producer_version: str | None
    checkpoint_tensor_count: int
    rank_loaded_tensor_count: int
    auxiliary_kv_scale_count: int


def _assert_collective_device(
    expected: torch.device,
    **tensors: torch.Tensor,
) -> None:
    if any(tensor.device != expected for tensor in tensors.values()):
        details = ", ".join(
            f"{name}={tensor.device}" for name, tensor in tensors.items()
        )
        raise ValueError(
            f"expert-parallel collective device mismatch: expected {expected}; {details}"
        )


class EpMoeBlock(nn.Module):
    """Canonical MoE tree with rank-local experts + all-to-all execution.

    Router, shared expert, and owned routed experts are registered directly at
    their HF paths. ``experts`` keeps the global length and uses ``None`` for
    remote owners, which PyTorch naturally omits from state/named enumeration
    while retaining global expert indices.
    """

    def __init__(self, block: nn.Module, comm, ep_rank: int, ep_size: int) -> None:
        super().__init__()
        num_experts = len(block.experts)
        if ep_size < 1:
            raise ValueError("expert-parallel size must be positive")
        if not 0 <= ep_rank < ep_size:
            raise ValueError(
                f"expert-parallel rank {ep_rank} is outside size {ep_size}"
            )
        if num_experts % ep_size != 0:
            raise ValueError(f"{num_experts} experts do not divide across {ep_size} ranks")
        self._comm = comm
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.experts_per_rank = num_experts // ep_size
        start = ep_rank * self.experts_per_rank
        end = (ep_rank + 1) * self.experts_per_rank
        self.owned_expert_indices = tuple(range(start, end))

        # Register the canonical tree directly. Replacing the original block's
        # list too releases non-local tensors even if a caller retains a
        # reference to that pre-parallel block.
        if hasattr(block, "gate"):
            self.gate = block.gate
            route_override = None
        elif hasattr(block, "_route"):
            # Small/custom blocks may own routing logic without parameters.
            # The original expert list is replaced below before this bound
            # method is retained, so it cannot keep remote expert tensors live.
            self.gate = None
            route_override = block._route
        else:
            raise TypeError(
                f"{type(block).__name__} must expose gate or _route for EP"
            )
        experts = nn.ModuleList(
            expert if index in self.owned_expert_indices else None
            for index, expert in enumerate(block.experts)
        )
        self.experts = experts
        block.experts = experts
        self.shared_experts = getattr(block, "shared_experts", None)
        object.__setattr__(self, "_route_override", route_override)

        # Routing metadata is scalar/plain state, never checkpoint state.
        if route_override is None:
            self.top_k = block.top_k
            self.norm_topk_prob = block.norm_topk_prob
            if hasattr(block, "n_group"):
                self.n_group = block.n_group
                self.topk_group = block.topk_group
                self.routed_scaling_factor = block.routed_scaling_factor
        object.__setattr__(
            self,
            "_kairyu_parallel_shard",
            ParallelShardInfo("expert", ep_rank, ep_size),
        )

    @property
    def local_experts(self) -> tuple[nn.Module, ...]:
        """Compatibility view; modules stay registered by global index."""

        return tuple(
            self.local_expert(index)
            for index in self.owned_expert_indices
        )

    def owner_rank(self, expert_index: int) -> int:
        if not 0 <= expert_index < len(self.experts):
            raise IndexError(
                f"expert index {expert_index} is outside [0, {len(self.experts)})"
            )
        return expert_index // self.experts_per_rank

    def local_expert(self, expert_index: int) -> nn.Module:
        owner = self.owner_rank(expert_index)
        expert = self.experts[expert_index]
        if expert is None:
            raise RemoteExpertOwnershipError(
                f"expert {expert_index} is owned by rank {owner}, "
                f"not rank {self.ep_rank}"
            )
        return expert

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        device = hidden.device
        topk_indices, topk_weights = self._route(hidden)
        tokens, k = topk_indices.shape
        flat_expert = topk_indices.reshape(-1)  # [tokens*k]
        owner = flat_expert // self.experts_per_rank
        order = torch.argsort(owner, stable=True)
        send_counts = torch.bincount(owner, minlength=self.ep_size)
        payload = hidden.repeat_interleave(k, dim=0)[order]

        recv_counts = torch.empty(
            self.ep_size, dtype=send_counts.dtype, device=device
        )
        _assert_collective_device(
            device, send_counts=send_counts, recv_counts=recv_counts
        )
        self._comm.tensor_all_to_all_single(
            recv_counts, send_counts.contiguous(), [1] * self.ep_size, [1] * self.ep_size
        )
        recv_total = int(recv_counts.sum().item())
        received = torch.empty(
            recv_total, hidden.shape[-1], dtype=hidden.dtype, device=device
        )
        _assert_collective_device(device, payload=payload, received=received)
        self._comm.tensor_all_to_all_single(
            received,
            payload.contiguous(),
            recv_counts.tolist(),
            send_counts.tolist(),
        )
        # which local expert each received row wants: exchange expert ids too
        expert_ids_out = flat_expert[order].to(torch.int64)
        expert_ids_in = torch.empty(recv_total, dtype=torch.int64, device=device)
        _assert_collective_device(
            device, expert_ids_out=expert_ids_out, expert_ids_in=expert_ids_in
        )
        self._comm.tensor_all_to_all_single(
            expert_ids_in,
            expert_ids_out.contiguous(),
            recv_counts.tolist(),
            send_counts.tolist(),
        )
        computed = torch.zeros_like(received)
        for global_index in expert_ids_in.unique():
            expert_index = int(global_index)
            mask = expert_ids_in == global_index
            computed[mask] = self.local_expert(expert_index)(received[mask])

        returned = torch.empty(
            tokens * k, hidden.shape[-1], dtype=hidden.dtype, device=device
        )
        _assert_collective_device(device, computed=computed, returned=returned)
        self._comm.tensor_all_to_all_single(
            returned,
            computed.contiguous(),
            send_counts.tolist(),
            recv_counts.tolist(),
        )
        # undo the permutation, weight, and combine per token
        unsorted = torch.empty_like(returned)
        unsorted[order] = returned
        combine_dtype = torch.promote_types(unsorted.dtype, topk_weights.dtype)
        weighted = unsorted.reshape(tokens, k, -1).to(combine_dtype)
        weighted = weighted * topk_weights.to(combine_dtype)[:, :, None]
        out = weighted.sum(dim=1).to(hidden.dtype)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden)
        return out

    def _route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._route_override is not None:
            return self._route_override(hidden)
        return route_experts(self, hidden)


def _ep_checkpoint_shapes(
    model: nn.Module,
    *,
    tie_word_embeddings: bool,
) -> tuple[
    dict[str, tuple[int, ...]],
    dict[str, tuple[int, ...]],
    tuple[tuple[str, EpMoeBlock], ...],
]:
    """Return rank-local and reconstructed full-checkpoint shape contracts."""

    local = {
        name: tuple(tensor.shape)
        for name, tensor in model.state_dict().items()
    }
    if tie_word_embeddings:
        local.pop("lm_head.weight", None)
    expected = dict(local)
    blocks = tuple(
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, EpMoeBlock)
    )
    for block_name, block in blocks:
        template_index = block.owned_expert_indices[0]
        template_prefix = f"{block_name}.experts.{template_index}."
        template = {
            name.removeprefix(template_prefix): shape
            for name, shape in local.items()
            if name.startswith(template_prefix)
        }
        if not template:
            raise RuntimeError(
                f"owned expert {template_index} at {block_name!r} has no checkpoint state"
            )
        for expert_index in range(len(block.experts)):
            for suffix, shape in template.items():
                expected[
                    f"{block_name}.experts.{expert_index}.{suffix}"
                ] = shape
    return local, expected, blocks


def _kv_scale_names(model: nn.Module) -> tuple[str, ...]:
    names = []
    for layer_index, _layer in enumerate(model.model.layers):
        names.extend(
            (
                f"model.layers.{layer_index}.self_attn.k_proj.k_scale",
                f"model.layers.{layer_index}.self_attn.v_proj.v_scale",
            )
        )
    return tuple(names)


_NVFP4_SAFETENSORS_DTYPES = {
    torch.uint8: "U8",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float32: "F32",
}


def _global_nvfp4_dtype_contract(
    model: nn.Module,
    blocks: tuple[tuple[str, EpMoeBlock], ...],
) -> dict[str, str]:
    """Reconstruct the packed ABI for owned and remote expert projections."""

    local = {
        (f"{module_name}.{buffer_name}" if module_name else buffer_name): (
            _NVFP4_SAFETENSORS_DTYPES[buffer.dtype]
        )
        for module_name, module in model.named_modules()
        if isinstance(module, NvFp4Linear)
        for buffer_name, buffer in module.named_buffers(recurse=False)
    }
    expected = dict(local)
    for block_name, block in blocks:
        template_index = block.owned_expert_indices[0]
        template_prefix = f"{block_name}.experts.{template_index}."
        template = {
            name.removeprefix(template_prefix): dtype
            for name, dtype in local.items()
            if name.startswith(template_prefix)
        }
        if not template:
            raise RuntimeError(
                f"owned NVFP4 expert {template_index} at {block_name!r} "
                "has no packed checkpoint members"
            )
        for expert_index in range(len(block.experts)):
            for suffix, dtype in template.items():
                expected[f"{block_name}.experts.{expert_index}.{suffix}"] = dtype
    return expected


def _checkpoint_nvfp4_moe_global_input_scales(
    reader,
    blocks: tuple[tuple[str, EpMoeBlock], ...],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Read layer-global FC1/FC2 activation scales, including remote experts."""

    requested: dict[str, tuple[str, str]] = {}
    for block_name, block in blocks:
        for expert_index in range(len(block.experts)):
            prefix = f"{block_name}.experts.{expert_index}"
            for projection in ("gate_proj", "up_proj"):
                requested[f"{prefix}.{projection}.input_scale"] = (
                    block_name,
                    "w13",
                )
            requested[f"{prefix}.down_proj.input_scale"] = (block_name, "w2")

    grouped: dict[tuple[str, str], list[torch.Tensor]] = {}
    for name, scale in reader.selected_items(requested):
        value = scale.detach().to(dtype=torch.float32).reshape(-1)
        if (
            value.numel() != 1
            or not bool(torch.isfinite(value).all())
            or not bool((value > 0).all())
        ):
            raise ValueError(
                f"checkpoint NVFP4 MoE input scale {name!r} must be one "
                "finite positive FP32 value"
            )
        grouped.setdefault(requested[name], []).append(value.reshape(()))

    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for block_name, block in blocks:
        expected_experts = len(block.experts)
        w13 = grouped.get((block_name, "w13"), [])
        w2 = grouped.get((block_name, "w2"), [])
        if len(w13) != expected_experts * 2 or len(w2) != expected_experts:
            raise RuntimeError(
                f"checkpoint NVFP4 MoE scales for {block_name!r} are incomplete"
            )
        result[block_name] = (torch.stack(w13).max(), torch.stack(w2).max())
    return result


def _validate_ep_checkpoint_contract(
    reader,
    model: nn.Module,
    *,
    tie_word_embeddings: bool,
    kv_cache_quant_algo: str | None,
) -> tuple[
    dict[str, tuple[int, ...]],
    tuple[tuple[str, EpMoeBlock], ...],
    tuple[str, ...],
]:
    local, expected, blocks = _ep_checkpoint_shapes(
        model,
        tie_word_embeddings=tie_word_embeddings,
    )
    kv_scale_names = (
        _kv_scale_names(model)
        if kv_cache_quant_algo == "FP8"
        else ()
    )
    required = set(expected) | set(kv_scale_names)
    actual = set(reader.names())
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if missing or unexpected:
        raise ValueError(
            "expert-parallel checkpoint tensor layout mismatch: "
            f"missing={missing[:8]} ({len(missing)} total), "
            f"unexpected={unexpected[:8]} ({len(unexpected)} total)"
        )

    actual_specs = reader.specs(expected)
    mismatches = [
        (name, expected[name], actual_specs[name][0])
        for name in expected
        if actual_specs[name][0] != expected[name]
    ]
    if mismatches:
        name, wanted, found = mismatches[0]
        raise ValueError(
            f"checkpoint tensor {name!r} shape {found} != expected {wanted}; "
            f"{len(mismatches)} shape mismatch(es)"
        )
    expected_dtypes = _global_nvfp4_dtype_contract(model, blocks)
    dtype_mismatches = [
        (name, expected_dtype, actual_specs[name][1])
        for name, expected_dtype in expected_dtypes.items()
        if actual_specs[name][1] != expected_dtype
    ]
    if dtype_mismatches:
        name, wanted, found = dtype_mismatches[0]
        raise ValueError(
            f"NVFP4 checkpoint tensor {name!r} dtype {found} != "
            f"required ABI dtype {wanted}; "
            f"{len(dtype_mismatches)} dtype mismatch(es)"
        )
    for name, scale in reader.selected_items(kv_scale_names):
        if (
            scale.numel() != 1
            or not scale.is_floating_point()
            or not torch.isfinite(scale).all()
            or not bool((scale > 0).all())
        ):
            raise ValueError(
                f"checkpoint FP8-KV calibration tensor {name!r} must be one "
                "finite positive floating-point value"
            )
    return local, blocks, kv_scale_names


def _assign_checkpoint_tensor(
    model: nn.Module,
    name: str,
    tensor: torch.Tensor,
) -> None:
    module_name, separator, member = name.rpartition(".")
    module = model.get_submodule(module_name) if separator else model
    if member in module._parameters:
        current = module._parameters[member]
        if current is None:
            raise RuntimeError(f"checkpoint parameter {name!r} is disabled")
        module._parameters[member] = nn.Parameter(
            tensor,
            requires_grad=current.requires_grad,
        )
        return
    if member in module._buffers:
        module._buffers[member] = tensor
        return
    raise KeyError(f"checkpoint member {name!r} is not registered on the model")


def build_ep_model(
    model_dir: str | Path,
    ep_size: int,
    ep_rank: int,
    comm,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    attention_backend=None,
    linear_capabilities=None,
    linear_selection_policy=None,
) -> tuple[nn.Module, object, ExpertParallelLoadInfo]:
    """Build one real Qwen3-MoE rank without materializing remote experts.

    Attention, router, embeddings, and the output head are replicated. Routed
    expert modules retain their global HF names, but only this rank's contiguous
    block is constructed and loaded. The complete checkpoint name/shape
    contract is validated on every rank before any request can execute.
    """

    from kairyu.engine.core.quant_config import (
        QuantMethod,
        load_checkpoint_quantization,
        validate_model_quantization,
    )
    from kairyu.engine.core.weights import CheckpointReader
    from kairyu.models.config import parse_model_config
    from kairyu.models.layers import RotaryEmbedding
    from kairyu.models.llama import DenseDecoder
    from kairyu.quant.linear import (
        QuantizedLinearBase,
        linear_factory,
    )

    directory = Path(model_dir)
    raw = json.loads((directory / "config.json").read_text())
    config = parse_model_config(raw)
    if config.architecture != "Qwen3MoeForCausalLM":
        raise ValueError(
            "expert-parallel real-checkpoint loading currently supports "
            f"Qwen3MoeForCausalLM, got {config.architecture!r}"
        )
    if config.moe is None:
        raise ValueError("expert-parallel loading requires a MoE model config")
    if ep_size < 1:
        raise ValueError("expert-parallel size must be positive")
    if not 0 <= ep_rank < ep_size:
        raise ValueError(
            f"expert-parallel rank {ep_rank} is outside size {ep_size}"
        )
    if config.moe.num_experts % ep_size:
        raise ValueError(
            f"{config.moe.num_experts} experts do not divide across {ep_size} ranks"
        )

    resolved = load_checkpoint_quantization(directory, raw)
    quant = resolved.weights
    validate_model_quantization(
        quant,
        is_mla=config.is_mla,
        architecture=config.architecture,
    )
    base_factory = linear_factory(
        quant,
        device=device,
        dtype=dtype,
        capabilities=linear_capabilities,
        selection_policy=linear_selection_policy,
    )
    factory = _ExpertShardedLinearFactory(
        base_factory,
        num_experts=config.moe.num_experts,
        ep_rank=ep_rank,
        ep_size=ep_size,
    )
    # The official checkpoint is 134 GB. Even rank-local EP2 dummy buffers
    # would allocate ~70 GB on the host before loading useful values. Construct
    # shape-only modules, drop remote experts, and assign only selected tensors.
    with torch.device("meta"):
        model = DenseDecoder(
            config,
            attention_backend=attention_backend,
            linear_factory=factory,
            dtype=dtype,
        )
    for layer in model.model.layers:
        if hasattr(layer.mlp, "experts"):
            layer.mlp = EpMoeBlock(
                layer.mlp,
                comm,
                ep_rank=ep_rank,
                ep_size=ep_size,
            )
    reader = CheckpointReader(directory)
    local_shapes, blocks, kv_scale_names = _validate_ep_checkpoint_contract(
        reader,
        model,
        tie_word_embeddings=config.tie_word_embeddings,
        kv_cache_quant_algo=resolved.kv_cache_quant_algo,
    )
    if not blocks:
        raise ValueError("Qwen3-MoE checkpoint has no sparse expert layers")
    nvfp4_moe_input_scales = (
        _checkpoint_nvfp4_moe_global_input_scales(reader, blocks)
        if quant.method is QuantMethod.NVFP4
        else {}
    )

    quantized_dtypes = {
        (f"{module_name}.{buffer_name}" if module_name else buffer_name): buffer.dtype
        for module_name, module in model.named_modules()
        if isinstance(module, QuantizedLinearBase)
        for buffer_name, buffer in module.named_buffers(recurse=False)
    }
    strict_nvfp4_dtypes = {
        (f"{module_name}.{buffer_name}" if module_name else buffer_name): buffer.dtype
        for module_name, module in model.named_modules()
        if isinstance(module, NvFp4Linear)
        for buffer_name, buffer in module.named_buffers(recurse=False)
    }
    loaded: set[str] = set()
    for name, tensor in reader.selected_items(local_shapes):
        expected_shape = local_shapes[name]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"rank-local checkpoint tensor {name!r} shape "
                f"{tuple(tensor.shape)} != expected {expected_shape}"
            )
        quantized_dtype = quantized_dtypes.get(name)
        if name in strict_nvfp4_dtypes and tensor.dtype != strict_nvfp4_dtypes[name]:
            raise ValueError(
                f"NVFP4 checkpoint tensor {name!r} dtype {tensor.dtype} != "
                f"required ABI dtype {strict_nvfp4_dtypes[name]}"
            )
        if quantized_dtype is not None and tensor.dtype != quantized_dtype:
            tensor = tensor.to(quantized_dtype)
        elif (
            quantized_dtype is None
            and tensor.is_floating_point()
            and tensor.dtype != dtype
        ):
            tensor = tensor.to(dtype)
        _assign_checkpoint_tensor(model, name, tensor)
        loaded.add(name)
    if loaded != set(local_shapes):
        absent = sorted(set(local_shapes) - loaded)
        raise RuntimeError(
            f"rank-local expert load omitted {absent[:8]} ({len(absent)} total)"
        )
    for block_name, block in blocks:
        if block_name in nvfp4_moe_input_scales:
            w13_input_scale, w2_input_scale = nvfp4_moe_input_scales[block_name]
            apply_nvfp4_moe_global_input_scales(
                block.experts,
                w13_input_scale=w13_input_scale,
                w2_input_scale=w2_input_scale,
            )
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight

    fresh_rope = RotaryEmbedding(config)
    model.model.rotary_emb._buffers["inv_freq"] = fresh_rope.inv_freq
    model.model.rotary_emb.attention_scaling = fresh_rope.attention_scaling
    remaining_meta = [
        name
        for name, tensor in (
            *model.named_parameters(),
            *model.named_buffers(),
        )
        if tensor.device.type == "meta"
    ]
    if remaining_meta:
        raise RuntimeError(
            "expert-parallel load left meta tensors unresolved: "
            f"{remaining_meta[:8]} ({len(remaining_meta)} total)"
        )
    model.eval()
    model.refresh_dense_packs()
    gc.collect()
    model = model.to(device)
    owned = blocks[0][1].owned_expert_indices
    if any(block.owned_expert_indices != owned for _name, block in blocks):
        raise RuntimeError("expert ownership differs across sparse layers")
    info = ExpertParallelLoadInfo(
        ep_rank=ep_rank,
        ep_size=ep_size,
        owned_expert_indices=owned,
        quantization_source=resolved.source,
        quantization_method=quant.method.value,
        kv_cache_quant_algo=resolved.kv_cache_quant_algo,
        producer_name=resolved.producer_name,
        producer_version=resolved.producer_version,
        checkpoint_tensor_count=len(reader.names()),
        rank_loaded_tensor_count=len(loaded),
        auxiliary_kv_scale_count=len(kv_scale_names),
    )
    object.__setattr__(model, "_kairyu_ep_load_info", info)
    return model, config, info
