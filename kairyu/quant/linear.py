"""QuantizedLinear modules + the loader's linear factory (m14 D2).

Each module holds the PACKED tensors as persistent buffers under the
checkpoint's own names (our convention — review A9), so the loader assigns by
name with zero renaming. CPU ``forward`` uses a correctness oracle; CUDA
``forward`` dispatches to a fused kernel that never materializes the full
floating-point weight.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from importlib.util import find_spec
from typing import Protocol

import torch
from torch import nn

from kairyu.engine.core.quant_config import QuantConfig, QuantMethod
from kairyu.quant import awq, fp8, gptq, int8, nvfp4


class LinearRole(Enum):
    """Stable semantic roles used by construction policy.

    The qualified module name remains the checkpoint identity.  The role is
    deliberately orthogonal: policy can distinguish two projections without
    parsing a string, while changing policy never changes state-dict keys.
    """

    ATTENTION_QUERY = "attention_query"
    ATTENTION_KEY = "attention_key"
    ATTENTION_VALUE = "attention_value"
    ATTENTION_OUTPUT = "attention_output"
    MLA_QUERY_A = "mla_query_a"
    MLA_QUERY_B = "mla_query_b"
    MLA_KV_A = "mla_kv_a"
    MLA_KV_B = "mla_kv_b"
    MLP_GATE = "mlp_gate"
    MLP_UP = "mlp_up"
    MLP_DOWN = "mlp_down"
    MOE_ROUTER = "moe_router"
    DRAFT_FUSION = "draft_fusion"
    OUTPUT_HEAD = "output_head"


class ExpertScope(Enum):
    """Whether a projection belongs to a routed or shared MoE expert."""

    NONE = "none"
    ROUTED = "routed"
    SHARED = "shared"


class ModelScope(Enum):
    """Model tree that owns the checkpoint projection."""

    TARGET = "target"
    EAGLE_DRAFT = "eagle_draft"
    MTP_DRAFT = "mtp_draft"


class TensorParallelMode(Enum):
    REPLICATED = "replicated"
    COLUMN = "column"
    ROW = "row"


@dataclass(frozen=True)
class CheckpointMemberLayout:
    """Physical checkpoint axes for one direct persistent linear member.

    Quantized formats do not all store their logical ``[out, in]`` weight in
    that order.  Keeping this exact-member contract on the concrete module
    avoids both suffix parsing and assumptions that packed tensors share the
    dense weight's physical axes.
    """

    column_dim: int | None
    row_dim: int | None

    def __post_init__(self) -> None:
        if self.column_dim not in (None, 0, 1):
            raise ValueError("column checkpoint shard dim must be 0, 1, or None")
        if self.row_dim not in (None, 0, 1):
            raise ValueError("row checkpoint shard dim must be 0, 1, or None")

    def shard_dim(self, mode: TensorParallelMode) -> int | None:
        return {
            TensorParallelMode.REPLICATED: None,
            TensorParallelMode.COLUMN: self.column_dim,
            TensorParallelMode.ROW: self.row_dim,
        }[mode]


def bind_checkpoint_member_layout(
    module: nn.Module,
    member: str,
    layout: CheckpointMemberLayout,
) -> None:
    """Attach an exact auxiliary-state layout to a contextual projection."""

    if not member or "." in member:
        raise ValueError("checkpoint member layout requires a direct member name")
    if member not in module._parameters and member not in module._buffers:
        raise ValueError(
            f"checkpoint member {member!r} is not registered directly on "
            f"{type(module).__name__}"
        )
    layouts = dict(getattr(module, "_kairyu_checkpoint_member_layouts", {}))
    if member in layouts:
        raise ValueError(f"checkpoint member {member!r} already has a layout")
    layouts[member] = layout
    object.__setattr__(module, "_kairyu_checkpoint_member_layouts", layouts)


@dataclass(frozen=True)
class TensorParallelPlacement:
    """Logical TP ownership of one projection at construction time."""

    mode: TensorParallelMode
    world_size: int
    rank: int
    shard_dim: int | None
    global_in_features: int
    global_out_features: int
    local_in_features: int
    local_out_features: int

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("tensor-parallel world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"tensor-parallel rank {self.rank} is outside world_size {self.world_size}"
            )
        if self.shard_dim not in (None, 0, 1):
            raise ValueError("tensor-parallel shard_dim must be 0, 1, or None")
        expected_dim = {
            TensorParallelMode.REPLICATED: None,
            TensorParallelMode.COLUMN: 0,
            TensorParallelMode.ROW: 1,
        }[self.mode]
        if self.shard_dim != expected_dim:
            raise ValueError(
                f"tensor-parallel mode {self.mode.value} requires shard_dim={expected_dim}"
            )
        if (
            min(
                self.global_in_features,
                self.global_out_features,
                self.local_in_features,
                self.local_out_features,
            )
            < 1
        ):
            raise ValueError("tensor-parallel projection dimensions must be positive")


class LinearKernel(Enum):
    """Production kernel family bound to one constructed projection."""

    TORCH_DENSE = "torch_dense"
    CPU_REFERENCE = "cpu_reference"
    CUDA_FP8_AUTO = "cuda_fp8_auto"
    TRITON_INT8 = "triton_int8"
    TRITON_W4A16 = "triton_w4a16"
    FLASHINFER_NVFP4 = "flashinfer_nvfp4"


_KERNEL_METHODS: dict[LinearKernel, frozenset[QuantMethod]] = {
    LinearKernel.TORCH_DENSE: frozenset({QuantMethod.NONE}),
    LinearKernel.CPU_REFERENCE: frozenset(
        {
            QuantMethod.FP8,
            QuantMethod.INT8,
            QuantMethod.AWQ,
            QuantMethod.GPTQ,
            QuantMethod.NVFP4,
        }
    ),
    LinearKernel.CUDA_FP8_AUTO: frozenset({QuantMethod.FP8}),
    LinearKernel.TRITON_INT8: frozenset({QuantMethod.INT8}),
    LinearKernel.TRITON_W4A16: frozenset(
        {
            QuantMethod.AWQ,
            QuantMethod.GPTQ,
        }
    ),
    LinearKernel.FLASHINFER_NVFP4: frozenset({QuantMethod.NVFP4}),
}


@dataclass(frozen=True)
class LinearKernelCapabilities:
    """Quantized formats accepted by the target kernel profile.

    Dense construction is always required.  Quantized checkpoint formats never
    silently fall back to dense when a target lacks a kernel: their checkpoint
    tensors are ABI-incompatible with ``nn.Linear``.
    """

    available_kernels: frozenset[LinearKernel]
    source: str

    def __post_init__(self) -> None:
        if LinearKernel.TORCH_DENSE not in self.available_kernels:
            raise ValueError("linear capabilities must include torch_dense")
        if not self.source:
            raise ValueError("linear capability source must be non-empty")

    @property
    def supported_quant_methods(self) -> frozenset[QuantMethod]:
        methods: set[QuantMethod] = set()
        for kernel in self.available_kernels:
            methods.update(_KERNEL_METHODS[kernel])
        return frozenset(methods)

    def preferred_kernel(self, method: QuantMethod) -> LinearKernel | None:
        order = (
            LinearKernel.TORCH_DENSE,
            LinearKernel.CUDA_FP8_AUTO,
            LinearKernel.TRITON_INT8,
            LinearKernel.TRITON_W4A16,
            LinearKernel.FLASHINFER_NVFP4,
            LinearKernel.CPU_REFERENCE,
        )
        return next(
            (
                kernel
                for kernel in order
                if kernel in self.available_kernels and method in _KERNEL_METHODS[kernel]
            ),
            None,
        )

    @classmethod
    def for_device(cls, device: str | torch.device) -> LinearKernelCapabilities:
        target = torch.device(device)
        if target.type == "cpu":
            # Every quantized module has a CPU correctness oracle.
            return cls(
                frozenset(
                    {
                        LinearKernel.TORCH_DENSE,
                        LinearKernel.CPU_REFERENCE,
                    }
                ),
                "cpu-reference",
            )
        if target.type != "cuda":
            return cls(
                frozenset({LinearKernel.TORCH_DENSE}),
                f"{target.type}-dense-only",
            )
        if not torch.cuda.is_available():
            raise RuntimeError(f"cannot probe linear kernels for {target}: CUDA is unavailable")
        major, minor = torch.cuda.get_device_capability(target)
        kernels = {LinearKernel.TORCH_DENSE}
        has_triton, triton_status = _probe_python_backend("triton")
        has_flashinfer, flashinfer_status = _probe_python_backend(
            "flashinfer",
            ("SfLayout", "mm_fp4", "nvfp4_quantize"),
        )
        if (major, minor) >= (8, 0) and has_triton:
            kernels.update(
                {
                    LinearKernel.TRITON_INT8,
                    LinearKernel.TRITON_W4A16,
                }
            )
        if (major, minor) >= (8, 9) and has_triton:
            kernels.add(LinearKernel.CUDA_FP8_AUTO)
        if (major, minor) >= (10, 0) and has_flashinfer:
            kernels.add(LinearKernel.FLASHINFER_NVFP4)
        return cls(
            frozenset(kernels),
            (f"cuda-sm{major}{minor};triton={triton_status};flashinfer={flashinfer_status}"),
        )


def _probe_python_backend(
    module_name: str,
    required_symbols: tuple[str, ...] = (),
) -> tuple[bool, str]:
    """Import one optional backend once and retain an actionable probe result."""

    if find_spec(module_name) is None:
        return False, "missing"
    try:
        module = import_module(module_name)
    except Exception as error:  # optional binary packages can raise OSError too
        return False, f"import-{type(error).__name__}"
    missing = tuple(symbol for symbol in required_symbols if not hasattr(module, symbol))
    if missing:
        return False, f"missing-symbol:{','.join(missing)}"
    return True, "available"


@dataclass(frozen=True)
class LinearBuildContext:
    """Complete immutable input to contextual projection selection."""

    qualified_name: str
    model_scope: ModelScope
    role: LinearRole
    device: torch.device
    dtype: torch.dtype
    tensor_parallel: TensorParallelPlacement
    capabilities: LinearKernelCapabilities
    layer_index: int | None = None
    expert_index: int | None = None
    expert_scope: ExpertScope = ExpertScope.NONE
    allow_quantization: bool = True

    def __post_init__(self) -> None:
        if not self.qualified_name:
            raise ValueError("linear qualified_name must be non-empty")
        if self.layer_index is not None and self.layer_index < 0:
            raise ValueError("linear layer_index must be non-negative")
        if self.expert_index is not None and self.expert_index < 0:
            raise ValueError("linear expert_index must be non-negative")
        if self.expert_scope is ExpertScope.ROUTED and self.expert_index is None:
            raise ValueError("routed expert context requires a global expert_index")
        if self.expert_scope is not ExpertScope.ROUTED and self.expert_index is not None:
            raise ValueError(f"{self.expert_scope.value} expert context cannot carry expert_index")


@dataclass(frozen=True)
class LinearSelection:
    quantization: QuantConfig
    kernel: LinearKernel | None
    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("linear selection reason must be non-empty")


LinearSelectionPolicy = Callable[[QuantConfig, LinearBuildContext], LinearSelection]


class LinearFactory(Protocol):
    """Construction protocol consumed by model modules."""

    def __call__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        *,
        qualified_name: str,
        model_scope: ModelScope = ModelScope.TARGET,
        role: LinearRole,
        layer_index: int | None = None,
        expert_index: int | None = None,
        expert_scope: ExpertScope = ExpertScope.NONE,
        shard_dim: int | None = None,
        allow_quantization: bool = True,
    ) -> nn.Module: ...


class QuantizedLinearBase(nn.Module):
    """Common shape bookkeeping; subclasses define buffers + dequantize()."""

    is_quantized = True  # the loader's dtype-cast skip flag (review A2)
    quant_scheme: str

    def __init__(self, in_features: int, out_features: int, bias: bool) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("quantized linear dimensions must be positive")
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def dequantize(self) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def checkpoint_member_layouts(self) -> dict[str, CheckpointMemberLayout]:
        """Exact direct-member layouts; concrete packed formats extend this."""

        return {
            "bias": CheckpointMemberLayout(column_dim=0, row_dim=None),
        }

    def _add_local_bias(self) -> bool:
        return not bool(getattr(self, "_kairyu_row_parallel_omit_bias", False))

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize().to(x.dtype)
        bias = self.bias if self._add_local_bias() else None
        return nn.functional.linear(x, weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cuda":
            return self.forward_fused(x)
        return self.forward_reference(x)

    def forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Production CUDA dispatch. Unsupported paths fail; none dequantize."""
        bound = getattr(self, "_fused_forward", None)
        if bound is not None:
            return bound(x, self, add_bias=self._add_local_bias())
        from kairyu.kernels.quant_gemm_gpu import linear_forward

        return linear_forward(x, self, add_bias=self._add_local_bias())


class Fp8Linear(QuantizedLinearBase):
    """compressed-tensors FP8: per-channel [out,1] or per-tensor (1,) scale."""

    quant_scheme = "fp8"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        activation_dynamic: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias)
        self.activation_dynamic = activation_dynamic
        self.register_buffer(
            "weight", torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn)
        )
        self.register_buffer("weight_scale", torch.ones(out_features, 1))
        if not activation_dynamic:
            self.register_buffer("input_scale", torch.ones(()))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # static (per-tensor) FP8 checkpoints ship a (1,) scale, dynamic ones a
        # per-channel [out,1] scale. Adopt the checkpoint's shape so both load —
        # a fixed [out,1] buffer rejects the (1,) form even with assign=True (M4).
        scale_key = prefix + "weight_scale"
        incoming = state_dict.get(scale_key)
        if incoming is not None and incoming.shape != self.weight_scale.shape:
            self.register_buffer("weight_scale", torch.empty_like(incoming))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def dequantize(self) -> torch.Tensor:
        return fp8.dequantize_fp8(self.weight, self.weight_scale)

    def checkpoint_member_layouts(self) -> dict[str, CheckpointMemberLayout]:
        return {
            **super().checkpoint_member_layouts(),
            "weight": CheckpointMemberLayout(column_dim=0, row_dim=1),
            "weight_scale": CheckpointMemberLayout(column_dim=0, row_dim=None),
            "input_scale": CheckpointMemberLayout(column_dim=None, row_dim=None),
        }

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        input_scale = None if self.activation_dynamic else self.input_scale
        x_q, x_scale = fp8.quantize_fp8_activation(x.to(torch.float32), scale=input_scale)
        out = fp8.fp8_w8a8_matmul(x_q, x_scale, self.weight, self.weight_scale)
        if self.bias is not None and self._add_local_bias():
            out = out + self.bias
        return out.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Honor an optional row-parallel activation-scale owner.

        The communicator is bound to this same projection object by the TP
        builder.  Keeping the projection at its checkpoint-canonical module
        path preserves parameter names, adapter targeting, and quantization
        metadata while retaining the global per-token scale required by
        dynamic FP8 row sharding.
        """
        comm = getattr(self, "_row_parallel_comm", None)
        if comm is not None and self.activation_dynamic:
            local_amax = self.activation_amax(x).contiguous()
            global_amax = comm.tensor_all_reduce_max(local_amax)
            activation_scale = (global_amax / fp8.FP8_MAX).clamp(min=1e-12)
            return self.forward_with_activation_scale(x, activation_scale)
        return super().forward(x)

    def activation_amax(self, x: torch.Tensor) -> torch.Tensor:
        """Per-token amax used to synchronize row-parallel quantization."""
        if x.device.type == "cuda":
            from kairyu.kernels.quant_gemm_gpu import fp8_activation_amax

            return fp8_activation_amax(x)
        return x.to(torch.float32).abs().amax(dim=-1, keepdim=True)

    def forward_with_activation_scale(
        self, x: torch.Tensor, activation_scale: torch.Tensor
    ) -> torch.Tensor:
        """Run dynamic FP8 with a precomputed per-token global TP scale."""
        if not self.activation_dynamic:
            raise ValueError("a global per-token scale requires dynamic FP8 activation")
        activation_scale = activation_scale.to(device=x.device, dtype=torch.float32).clamp(
            min=1e-12
        )
        if x.device.type == "cuda":
            bound = getattr(self, "_fused_forward", None)
            if bound is None:
                # Compatibility for a pre-#228 CPU factory whose model is
                # explicitly moved to CUDA after construction.
                from kairyu.kernels.quant_gemm_gpu import fp8_linear_forward

                bound = fp8_linear_forward
            return bound(
                x,
                self,
                activation_scale=activation_scale,
                add_bias=self._add_local_bias(),
            )
        scaled = (x.to(torch.float32) / activation_scale).clamp(-fp8.FP8_MAX, fp8.FP8_MAX)
        x_q = scaled.to(torch.float8_e4m3fn)
        out = fp8.fp8_w8a8_matmul(x_q, activation_scale, self.weight, self.weight_scale)
        if self.bias is not None and self._add_local_bias():
            out = out + self.bias
        return out.to(x.dtype)


class Int8Linear(QuantizedLinearBase):
    """compressed-tensors INT8 W8A8: dynamic per-token activations."""

    quant_scheme = "int8"

    def __init__(self, in_features: int, out_features: int, bias: bool) -> None:
        super().__init__(in_features, out_features, bias)
        self.register_buffer("weight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.ones(out_features, 1))

    def dequantize(self) -> torch.Tensor:
        return int8.dequantize_int8(self.weight, self.weight_scale)

    def checkpoint_member_layouts(self) -> dict[str, CheckpointMemberLayout]:
        return {
            **super().checkpoint_member_layouts(),
            "weight": CheckpointMemberLayout(column_dim=0, row_dim=1),
            "weight_scale": CheckpointMemberLayout(column_dim=0, row_dim=None),
        }

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        # exact W8A8 reference: int32 accumulation (the GPU kernel's oracle)
        x_q, x_scale = int8.quantize_int8_activation(x.to(torch.float32))
        out = int8.int8_w8a8_matmul(x_q, x_scale, self.weight, self.weight_scale)
        if self.bias is not None and self._add_local_bias():
            out = out + self.bias
        return out.to(x.dtype)


def _resolve_group_size(group_size: int, in_features: int) -> int:
    """group_size == -1 means one group over the whole input axis (AutoGPTQ/AWQ
    convention). Normalize it to in_features so the group count is 1, not a
    negative number that crashes the buffer allocation."""
    return in_features if group_size <= 0 else group_size


class AwqLinear(QuantizedLinearBase):
    quant_scheme = "awq"

    def __init__(
        self, in_features: int, out_features: int, bias: bool, group_size: int = 128
    ) -> None:
        super().__init__(in_features, out_features, bias)
        group_size = _resolve_group_size(group_size, in_features)
        if out_features % 8:
            raise ValueError("AWQ out_features must be divisible by 8")
        if in_features % group_size:
            raise ValueError("AWQ in_features must be divisible by group_size")
        self.group_size = group_size
        groups = in_features // group_size
        self.register_buffer(
            "qweight", torch.zeros(in_features, out_features // 8, dtype=torch.int32)
        )
        self.register_buffer("qzeros", torch.zeros(groups, out_features // 8, dtype=torch.int32))
        self.register_buffer("scales", torch.ones(groups, out_features, dtype=torch.float16))

    def dequantize(self) -> torch.Tensor:
        return awq.dequantize_awq(self.qweight, self.qzeros, self.scales, self.group_size)

    def checkpoint_member_layouts(self) -> dict[str, CheckpointMemberLayout]:
        packed = CheckpointMemberLayout(column_dim=1, row_dim=0)
        return {
            **super().checkpoint_member_layouts(),
            "qweight": packed,
            "qzeros": packed,
            "scales": packed,
        }


class GptqLinear(QuantizedLinearBase):
    quant_scheme = "gptq"

    def __init__(
        self, in_features: int, out_features: int, bias: bool, group_size: int = 128
    ) -> None:
        super().__init__(in_features, out_features, bias)
        group_size = _resolve_group_size(group_size, in_features)
        if in_features % 8 or out_features % 8:
            raise ValueError("GPTQ in_features and out_features must be divisible by 8")
        self.group_size = group_size
        groups = -(-in_features // group_size)
        self.register_buffer(
            "qweight", torch.zeros(in_features // 8, out_features, dtype=torch.int32)
        )
        self.register_buffer("qzeros", torch.zeros(groups, out_features // 8, dtype=torch.int32))
        self.register_buffer("scales", torch.ones(groups, out_features, dtype=torch.float16))
        self.register_buffer("g_idx", torch.zeros(in_features, dtype=torch.int32))

    def dequantize(self) -> torch.Tensor:
        return gptq.dequantize_gptq(
            self.qweight, self.qzeros, self.scales, self.g_idx, self.group_size
        )

    def checkpoint_member_layouts(self) -> dict[str, CheckpointMemberLayout]:
        packed = CheckpointMemberLayout(column_dim=1, row_dim=0)
        return {
            **super().checkpoint_member_layouts(),
            "qweight": packed,
            "qzeros": packed,
            "scales": packed,
            "g_idx": CheckpointMemberLayout(column_dim=None, row_dim=0),
        }


class NvFp4Linear(QuantizedLinearBase):
    """modelopt NVFP4 (block-16 fp8 scales + fp32 global scale)."""

    quant_scheme = "nvfp4"

    def __init__(self, in_features: int, out_features: int, bias: bool) -> None:
        super().__init__(in_features, out_features, bias)
        if in_features % 16:
            raise ValueError("NVFP4 in_features must be divisible by 16")
        self.register_buffer(
            "weight", torch.zeros(out_features, in_features // 2, dtype=torch.uint8)
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, in_features // 16, dtype=torch.float8_e4m3fn),
        )
        self.register_buffer("weight_scale_2", torch.ones(()))
        # real checkpoints ship an input_scale scalar (activation quant arrives
        # with the GPU kernels); accepted so loads don't reject it
        self.register_buffer("input_scale", torch.ones(()))

    def dequantize(self) -> torch.Tensor:
        return nvfp4.dequantize_nvfp4(self.weight, self.weight_scale, self.weight_scale_2)

    def checkpoint_member_layouts(self) -> dict[str, CheckpointMemberLayout]:
        packed = CheckpointMemberLayout(column_dim=0, row_dim=1)
        replicated = CheckpointMemberLayout(column_dim=None, row_dim=None)
        return {
            **super().checkpoint_member_layouts(),
            "weight": packed,
            "weight_scale": packed,
            "weight_scale_2": replicated,
            "input_scale": replicated,
        }

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
        packed, scale = nvfp4.quantize_nvfp4_with_scale(flat, self.input_scale)
        activation = nvfp4.dequantize_nvfp4(packed, scale, self.input_scale)
        weight = self.dequantize()
        bias = self.bias if self._add_local_bias() else None
        out = nn.functional.linear(activation, weight, bias)
        return out.reshape(*x.shape[:-1], self.out_features).to(x.dtype)


def _construct_linear(
    quant: QuantConfig,
    in_features: int,
    out_features: int,
    bias: bool,
) -> nn.Module:
    if quant.method is QuantMethod.NONE:
        return nn.Linear(in_features, out_features, bias=bias)
    if quant.method is QuantMethod.FP8:
        dynamic = quant.activation_dynamic is not False
        return Fp8Linear(
            in_features,
            out_features,
            bias,
            activation_dynamic=dynamic,
        )
    if quant.method is QuantMethod.INT8:
        return Int8Linear(in_features, out_features, bias)
    if quant.method is QuantMethod.AWQ:
        return AwqLinear(
            in_features,
            out_features,
            bias,
            group_size=quant.group_size or 128,
        )
    if quant.method is QuantMethod.GPTQ:
        return GptqLinear(
            in_features,
            out_features,
            bias,
            group_size=quant.group_size or 128,
        )
    if quant.method is QuantMethod.NVFP4:
        return NvFp4Linear(in_features, out_features, bias)
    raise ValueError(f"no QuantizedLinear for {quant.method}")  # pragma: no cover


def _default_selection(quant: QuantConfig, context: LinearBuildContext) -> LinearSelection:
    if not context.allow_quantization:
        return LinearSelection(
            QuantConfig(method=QuantMethod.NONE),
            None,
            "architectural-exclusion",
        )
    for pattern in quant.ignored_layers:
        try:
            matched = (
                re.fullmatch(pattern[3:], context.qualified_name) is not None
                if pattern.startswith("re:")
                else (
                    context.qualified_name == pattern
                    or context.qualified_name.startswith(f"{pattern}.")
                )
            )
        except re.error as error:
            raise ValueError(f"invalid quantization ignore pattern {pattern!r}") from error
        if matched:
            return LinearSelection(
                QuantConfig(method=QuantMethod.NONE),
                None,
                f"checkpoint-ignore:{pattern}",
            )
    # These projections were dense before contextual construction was added.
    # Keeping that as the default *policy* (rather than an architectural ban)
    # preserves existing checkpoint behavior while allowing an explicit
    # specialized policy to opt into a compatible packed representation.
    if context.role in {
        LinearRole.MOE_ROUTER,
        LinearRole.DRAFT_FUSION,
        LinearRole.OUTPUT_HEAD,
    }:
        return LinearSelection(
            QuantConfig(method=QuantMethod.NONE),
            None,
            f"default-dense-role:{context.role.value}",
        )
    return LinearSelection(quant, None, "checkpoint-format")


def _context_description(context: LinearBuildContext) -> str:
    tp = context.tensor_parallel
    return (
        f"name={context.qualified_name!r}, scope={context.model_scope.value}, "
        f"role={context.role.value}, device={context.device}, dtype={context.dtype}, "
        f"tp={tp.mode.value}:{tp.rank}/{tp.world_size}:dim={tp.shard_dim}, "
        f"local={tp.local_in_features}x{tp.local_out_features}, "
        f"global={tp.global_in_features}x{tp.global_out_features}, "
        f"capabilities={context.capabilities.source}"
    )


def _bind_fused_forward(
    module: nn.Module,
    selection: LinearSelection,
    device: torch.device,
) -> None:
    if device.type != "cuda" or selection.quantization.method is QuantMethod.NONE:
        return
    from kairyu.kernels import quant_gemm_gpu

    functions = {
        (
            LinearKernel.CUDA_FP8_AUTO,
            QuantMethod.FP8,
        ): quant_gemm_gpu.fp8_linear_forward,
        (
            LinearKernel.TRITON_INT8,
            QuantMethod.INT8,
        ): quant_gemm_gpu.int8_linear_forward,
        (
            LinearKernel.TRITON_W4A16,
            QuantMethod.AWQ,
        ): quant_gemm_gpu.awq_linear_forward,
        (
            LinearKernel.TRITON_W4A16,
            QuantMethod.GPTQ,
        ): quant_gemm_gpu.gptq_linear_forward,
        (
            LinearKernel.FLASHINFER_NVFP4,
            QuantMethod.NVFP4,
        ): quant_gemm_gpu.nvfp4_linear_forward,
    }
    key = (selection.kernel, selection.quantization.method)
    try:
        module._fused_forward = functions[key]
    except KeyError as error:  # construction validation should make this unreachable
        raise RuntimeError(
            "no bound forward implementation for "
            f"{selection.kernel.value}/{selection.quantization.method.value}"
        ) from error


class ContextualLinearFactory:
    """Projection factory with target and parallel placement bound once.

    The model supplies only projection-local identity.  The selection policy
    receives the complete immutable context and may explicitly choose a
    heterogeneous checkpoint format.  Unsupported quantized selections fail at
    construction; implicit dense fallback would misread packed checkpoint
    tensors and is therefore forbidden.
    """

    def __init__(
        self,
        quant: QuantConfig,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        tensor_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
        capabilities: LinearKernelCapabilities | None = None,
        selection_policy: LinearSelectionPolicy | None = None,
    ) -> None:
        if not isinstance(dtype, torch.dtype):
            raise TypeError("linear target dtype must be a torch.dtype")
        if tensor_parallel_size < 1:
            raise ValueError("tensor-parallel world_size must be positive")
        if not 0 <= tensor_parallel_rank < tensor_parallel_size:
            raise ValueError(
                f"tensor-parallel rank {tensor_parallel_rank} is outside "
                f"world_size {tensor_parallel_size}"
            )
        self._quant = quant
        self._device = torch.device(device)
        self._dtype = dtype
        self._world_size = tensor_parallel_size
        self._rank = tensor_parallel_rank
        self._capabilities = capabilities or LinearKernelCapabilities.for_device(self._device)
        self._selection_policy = selection_policy or _default_selection

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
        if qualified_name is None and role is None:
            # Preserve the public pre-#228
            # ``linear_factory(quant)(in, out, bias)`` contract exactly.  With
            # no identity there is intentionally no contextual policy or
            # selection metadata.
            return _construct_linear(self._quant, in_features, out_features, bias)
        if qualified_name is None or role is None:
            raise TypeError("contextual linear construction requires both qualified_name and role")
        mode = {
            None: TensorParallelMode.REPLICATED,
            0: TensorParallelMode.COLUMN,
            1: TensorParallelMode.ROW,
        }.get(shard_dim)
        if mode is None:
            raise ValueError("tensor-parallel shard_dim must be 0, 1, or None")
        global_in = (
            in_features * self._world_size if mode is TensorParallelMode.ROW else in_features
        )
        global_out = (
            out_features * self._world_size if mode is TensorParallelMode.COLUMN else out_features
        )
        context = LinearBuildContext(
            qualified_name=qualified_name,
            model_scope=model_scope,
            role=role,
            device=self._device,
            dtype=self._dtype,
            tensor_parallel=TensorParallelPlacement(
                mode=mode,
                world_size=self._world_size,
                rank=self._rank,
                shard_dim=shard_dim,
                global_in_features=global_in,
                global_out_features=global_out,
                local_in_features=in_features,
                local_out_features=out_features,
            ),
            capabilities=self._capabilities,
            layer_index=layer_index,
            expert_index=expert_index,
            expert_scope=expert_scope,
            allow_quantization=allow_quantization,
        )
        selected = self._selection_policy(self._quant, context)
        if not isinstance(selected, LinearSelection):
            raise TypeError(
                f"linear selection policy must return LinearSelection for {qualified_name!r}"
            )
        selected_quant = selected.quantization
        if not allow_quantization and selected_quant.method is not QuantMethod.NONE:
            raise ValueError(
                f"linear {qualified_name!r} ({role.value}) excludes quantization, "
                f"but policy selected {selected_quant.method.value}"
            )
        kernel = selected.kernel or self._capabilities.preferred_kernel(selected_quant.method)
        if kernel is None:
            raise RuntimeError(
                f"no production kernel for {selected_quant.method.value}; "
                f"{_context_description(context)}"
            )
        if (
            kernel not in self._capabilities.available_kernels
            or selected_quant.method not in _KERNEL_METHODS[kernel]
        ):
            raise RuntimeError(
                f"kernel {kernel.value} cannot serve {selected_quant.method.value}; "
                f"{_context_description(context)}"
            )
        if (self._device.type == "cuda" and kernel is LinearKernel.CPU_REFERENCE) or (
            self._device.type != "cuda"
            and kernel not in {LinearKernel.TORCH_DENSE, LinearKernel.CPU_REFERENCE}
        ):
            raise RuntimeError(
                f"kernel {kernel.value} is incompatible with target "
                f"{self._device.type}; {_context_description(context)}"
            )
        if (
            self._device.type == "cuda"
            and selected_quant.method is not QuantMethod.NONE
            and self._dtype not in (torch.float16, torch.bfloat16)
        ):
            raise ValueError(
                "quantized CUDA compute requires float16 or bfloat16; "
                f"{_context_description(context)}"
            )
        resolved = LinearSelection(selected_quant, kernel, selected.reason)
        module = _construct_linear(selected_quant, in_features, out_features, bias)
        _bind_fused_forward(module, resolved, self._device)
        # Plain metadata: it is intentionally absent from state_dict and cannot
        # change checkpoint names or tensor ownership.
        module.linear_context = context
        module.linear_selection = resolved
        return module


def linear_factory(
    quant: QuantConfig,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    tensor_parallel_size: int = 1,
    tensor_parallel_rank: int = 0,
    capabilities: LinearKernelCapabilities | None = None,
    selection_policy: LinearSelectionPolicy | None = None,
) -> LinearFactory:
    """Build the contextual constructor used by model projection sites."""

    return ContextualLinearFactory(
        quant,
        device=device,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
        capabilities=capabilities,
        selection_policy=selection_policy,
    )


def make_linear(
    factory: LinearFactory | None,
    in_features: int,
    out_features: int,
    bias: bool,
    *,
    qualified_name: str,
    model_scope: ModelScope = ModelScope.TARGET,
    role: LinearRole,
    layer_index: int | None = None,
    expert_index: int | None = None,
    expert_scope: ExpertScope = ExpertScope.NONE,
    shard_dim: int | None = None,
    allow_quantization: bool = True,
) -> nn.Module:
    """Construct one projection while preserving direct-model dense defaults."""

    if factory is None:
        return nn.Linear(in_features, out_features, bias=bias)
    contextual_kwargs = {
        "qualified_name": qualified_name,
        "model_scope": model_scope,
        "role": role,
        "layer_index": layer_index,
        "expert_index": expert_index,
        "expert_scope": expert_scope,
        "shard_dim": shard_dim,
        "allow_quantization": allow_quantization,
    }
    if isinstance(factory, ContextualLinearFactory):
        return factory(
            in_features,
            out_features,
            bias,
            **contextual_kwargs,
        )
    try:
        inspect.signature(factory).bind(
            in_features,
            out_features,
            bias,
            **contextual_kwargs,
        )
    except (TypeError, ValueError):
        # Backward compatibility for the public pre-#228 three-argument
        # factory contract. Signature binding, rather than catching invocation
        # TypeError, never masks an error raised inside the legacy constructor.
        return factory(in_features, out_features, bias)
    return factory(
        in_features,
        out_features,
        bias,
        **contextual_kwargs,
    )
