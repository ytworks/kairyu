"""SM120 FlashInfer CUTLASS NVFP4 fused-MoE dispatch.

Packing and ownership live above this L1 adapter.  The tensors accepted here
are already in FlashInfer's physical ABI: FP4 weights are viewed as ``int64``
and their 128x4-interleaved E4M3 block scales are viewed as ``int32``.  Keeping
the adapter allocation-free (apart from FlashInfer's internal workspace) lets
the caller retain one preallocated output per execution bucket.

Expert-parallel ranks pass global router IDs and only their contiguous local
expert tensors.  FlashInfer therefore writes a rank-local weighted partial;
L2 owns the subsequent all-reduce.  A rank with no selected local expert is a
valid request and still dispatches so that the output buffer is overwritten
with zeros by the kernel.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PreparedNvFp4Moe:
    """Prepared rank-local tensors for one fused NVFP4 MoE layer.

    ``fc1_weight`` has logical projection order ``[up; gate]``.  The integer
    dtypes are storage views, not numerical INT64/INT32 operands.
    """

    fc1_weight: torch.Tensor
    fc2_weight: torch.Tensor
    fc1_act_scale: torch.Tensor
    fc1_weight_scale: torch.Tensor
    fc1_dequant_scale: torch.Tensor
    fc2_act_scale: torch.Tensor
    fc2_weight_scale: torch.Tensor
    fc2_dequant_scale: torch.Tensor
    num_global_experts: int
    local_expert_start: int

    def quant_scales(self) -> list[torch.Tensor]:
        """Return FlashInfer's six-element NVFP4 scale ABI in fixed order."""
        return [
            self.fc1_act_scale,
            self.fc1_weight_scale,
            self.fc1_dequant_scale,
            self.fc2_act_scale,
            self.fc2_weight_scale,
            self.fc2_dequant_scale,
        ]


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _require_sm120(hidden: torch.Tensor) -> None:
    if hidden.device.type != "cuda":
        raise RuntimeError(
            "NVFP4 fused MoE requires CUDA tensors on SM120; "
            f"hidden is on {hidden.device}"
        )
    capability = torch.cuda.get_device_capability(hidden.device)
    if capability != (12, 0):
        raise RuntimeError(
            "NVFP4 fused MoE is verified only on SM120; "
            f"got SM{capability[0]}{capability[1]} on {hidden.device}"
        )


def _require_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} shape must be {shape}, got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise RuntimeError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_request(
    hidden: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    prepared: PreparedNvFp4Moe,
    output: torch.Tensor,
    *,
    ep_size: int,
    ep_rank: int,
) -> None:
    _require_sm120(hidden)
    if hidden.ndim != 2:
        raise ValueError(f"hidden must be 2-D [tokens, hidden_size], got {tuple(hidden.shape)}")
    if hidden.dtype != torch.bfloat16:
        raise TypeError(f"hidden must have dtype torch.bfloat16, got {hidden.dtype}")
    if not hidden.is_contiguous():
        raise ValueError("hidden must be contiguous")

    if isinstance(ep_size, bool) or not isinstance(ep_size, int) or ep_size < 1:
        raise ValueError(f"ep_size must be a positive integer, got {ep_size!r}")
    if isinstance(ep_rank, bool) or not isinstance(ep_rank, int) or not 0 <= ep_rank < ep_size:
        raise ValueError(f"ep_rank must be in [0, {ep_size}), got {ep_rank!r}")
    if (
        isinstance(prepared.num_global_experts, bool)
        or not isinstance(prepared.num_global_experts, int)
        or prepared.num_global_experts < 1
    ):
        raise ValueError(
            "num_global_experts must be a positive integer, got "
            f"{prepared.num_global_experts!r}"
        )
    if (
        isinstance(prepared.local_expert_start, bool)
        or not isinstance(prepared.local_expert_start, int)
        or prepared.local_expert_start < 0
    ):
        raise ValueError(
            "local_expert_start must be a non-negative integer, got "
            f"{prepared.local_expert_start!r}"
        )

    if prepared.fc1_weight.ndim != 3 or prepared.fc2_weight.ndim != 3:
        raise ValueError("fc1_weight and fc2_weight must both be 3-D")
    local_experts = prepared.fc2_weight.shape[0]
    if local_experts < 1:
        raise ValueError("prepared tensors must contain at least one local expert")
    if prepared.fc1_weight.shape[0] != local_experts:
        raise ValueError(
            "fc1_weight and fc2_weight must contain the same number of local experts"
        )
    if prepared.num_global_experts != local_experts * ep_size:
        raise ValueError(
            "num_global_experts must equal local_experts * ep_size: "
            f"{prepared.num_global_experts} != {local_experts} * {ep_size}"
        )
    expected_start = ep_rank * local_experts
    if prepared.local_expert_start != expected_start:
        raise ValueError(
            "local experts must be the contiguous global range owned by ep_rank: "
            f"start={prepared.local_expert_start}, expected {expected_start}"
        )
    if prepared.local_expert_start + local_experts > prepared.num_global_experts:
        raise ValueError("local expert range exceeds num_global_experts")

    tokens, hidden_size = hidden.shape
    if tokens < 1 or hidden_size < 1:
        raise ValueError("hidden must contain at least one token and one hidden element")
    if prepared.fc2_weight.shape[1] != hidden_size:
        raise ValueError(
            "fc2_weight hidden dimension must match hidden: "
            f"{prepared.fc2_weight.shape[1]} != {hidden_size}"
        )
    if prepared.fc2_weight.shape[2] < 1:
        raise ValueError("fc2_weight must contain a non-empty packed intermediate dimension")
    intermediate_size = prepared.fc2_weight.shape[2] * 16
    expected_fc1_shape = (local_experts, 2 * intermediate_size, hidden_size // 16)
    if hidden_size % 16 != 0 or tuple(prepared.fc1_weight.shape) != expected_fc1_shape:
        raise ValueError(
            "fc1_weight must be packed [local_experts, 2 * intermediate_size, "
            f"hidden_size / 16]; expected {expected_fc1_shape}, "
            f"got {tuple(prepared.fc1_weight.shape)}"
        )

    device = hidden.device
    _require_tensor(
        prepared.fc1_weight,
        name="fc1_weight",
        dtype=torch.int64,
        shape=expected_fc1_shape,
        device=device,
    )
    _require_tensor(
        prepared.fc2_weight,
        name="fc2_weight",
        dtype=torch.int64,
        shape=(local_experts, hidden_size, intermediate_size // 16),
        device=device,
    )

    fc1_scale_shape = (
        local_experts,
        _ceil_div(2 * intermediate_size, 128) * 128,
        _ceil_div(_ceil_div(hidden_size, 16), 4),
    )
    fc2_scale_shape = (
        local_experts,
        _ceil_div(hidden_size, 128) * 128,
        _ceil_div(_ceil_div(intermediate_size, 16), 4),
    )
    for tensor, name, dtype, shape in (
        (prepared.fc1_act_scale, "fc1_act_scale", torch.float32, ()),
        (prepared.fc1_weight_scale, "fc1_weight_scale", torch.int32, fc1_scale_shape),
        (
            prepared.fc1_dequant_scale,
            "fc1_dequant_scale",
            torch.float32,
            (local_experts,),
        ),
        (prepared.fc2_act_scale, "fc2_act_scale", torch.float32, ()),
        (prepared.fc2_weight_scale, "fc2_weight_scale", torch.int32, fc2_scale_shape),
        (
            prepared.fc2_dequant_scale,
            "fc2_dequant_scale",
            torch.float32,
            (local_experts,),
        ),
    ):
        _require_tensor(tensor, name=name, dtype=dtype, shape=shape, device=device)

    if selected_experts.ndim != 2 or selected_experts.shape[0] != tokens:
        raise ValueError(
            "selected_experts must be 2-D [tokens, top_k] with the same token count as hidden"
        )
    top_k = selected_experts.shape[1]
    if not 1 <= top_k <= prepared.num_global_experts:
        raise ValueError(
            f"top_k must be in [1, {prepared.num_global_experts}], got {top_k}"
        )
    route_shape = (tokens, top_k)
    _require_tensor(
        selected_experts,
        name="selected_experts",
        dtype=torch.int32,
        shape=route_shape,
        device=device,
    )
    _require_tensor(
        routing_weights,
        name="routing_weights",
        dtype=torch.float32,
        shape=route_shape,
        device=device,
    )
    # Device-side assertion avoids a host scalar sync in the steady-state hot
    # path while still failing closed before invalid IDs can index the kernel.
    torch._assert_async(
        ((selected_experts >= 0) & (selected_experts < prepared.num_global_experts)).all(),
        "selected_experts must contain global IDs in "
        f"[0, {prepared.num_global_experts})",
    )

    _require_tensor(
        output,
        name="output",
        dtype=torch.bfloat16,
        shape=(tokens, hidden_size),
        device=device,
    )
    if output.data_ptr() == hidden.data_ptr():
        raise ValueError("output must not alias hidden")


def _load_flashinfer() -> tuple[object, object]:
    try:
        module = importlib.import_module("flashinfer")
    except ImportError as error:
        raise RuntimeError(
            "SM120 NVFP4 fused MoE requires FlashInfer with cutlass_fused_moe; "
            "install the Kairyu CUDA image's GPU dependencies"
        ) from error

    function = getattr(module, "cutlass_fused_moe", None)
    activation_type = getattr(module, "ActivationType", None)
    swiglu = getattr(activation_type, "Swiglu", None)
    if not callable(function) or swiglu is None:
        version = getattr(module, "__version__", "unknown")
        raise RuntimeError(
            "FlashInfer fused-MoE API mismatch "
            f"(version={version}): expected cutlass_fused_moe and ActivationType.Swiglu"
        )
    return function, swiglu


def fused_moe_forward(
    hidden: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    prepared: PreparedNvFp4Moe,
    *,
    output: torch.Tensor,
    ep_size: int,
    ep_rank: int,
) -> torch.Tensor:
    """Write one rank-local NVFP4 MoE partial into ``output`` and return it."""
    _validate_request(
        hidden,
        selected_experts,
        routing_weights,
        prepared,
        output,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )
    cutlass_fused_moe, swiglu = _load_flashinfer()

    # FlashInfer 0.6.14 resolves its AOT module from the current device rather
    # than the input tensor, so bind the context explicitly on multi-GPU ranks.
    with torch.cuda.device(hidden.device):
        result = cutlass_fused_moe(
            hidden,
            selected_experts,
            routing_weights,
            prepared.fc1_weight,
            prepared.fc2_weight,
            torch.bfloat16,
            quant_scales=prepared.quant_scales(),
            fc1_expert_biases=None,
            fc2_expert_biases=None,
            input_sf=None,
            tp_size=1,
            tp_rank=0,
            ep_size=ep_size,
            ep_rank=ep_rank,
            cluster_size=1,
            cluster_rank=0,
            output=output,
            enable_alltoall=False,
            use_deepseek_fp8_block_scale=False,
            use_w4_group_scaling=False,
            use_mxfp8_act_scaling=False,
            min_latency_mode=False,
            use_packed_weights=False,
            activation_type=swiglu,
            swizzled_input_sf=True,
        )

    # The 0.6.14 implementation returns a list even though its public
    # annotation says Tensor.  Accept both documented and observed forms, but
    # require that the supplied output buffer is the one actually returned.
    returned = result[0] if isinstance(result, (list, tuple)) and result else result
    if not isinstance(returned, torch.Tensor) or returned.data_ptr() != output.data_ptr():
        raise RuntimeError("FlashInfer did not return the preallocated fused-MoE output buffer")
    return output
