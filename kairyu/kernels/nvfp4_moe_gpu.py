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
    input_sf: torch.Tensor | None = None,
) -> None:
    _require_sm120(hidden)
    if hidden.ndim != 2:
        raise ValueError(f"hidden must be 2-D [tokens, hidden_size], got {tuple(hidden.shape)}")
    if not hidden.is_contiguous():
        raise ValueError("hidden must be contiguous")
    if hidden.dtype is torch.bfloat16:
        if input_sf is not None:
            raise ValueError("BF16 hidden must not provide a prequantized input_sf")
        hidden_size = hidden.shape[1]
    elif hidden.dtype is torch.uint8:
        if input_sf is None:
            raise ValueError("packed NVFP4 hidden requires a swizzled input_sf")
        hidden_size = hidden.shape[1] * 2
    else:
        raise TypeError(
            "hidden must have dtype torch.bfloat16 or torch.uint8 packed NVFP4, "
            f"got {hidden.dtype}"
        )

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

    tokens = hidden.shape[0]
    if tokens < 1 or hidden_size < 1:
        raise ValueError("hidden must contain at least one token and one hidden element")
    if input_sf is not None:
        expected_scale_elements = (
            _ceil_div(tokens, 128) * 128 * _ceil_div(hidden_size, 16)
        )
        _require_tensor(
            input_sf,
            name="input_sf",
            dtype=torch.uint8,
            shape=(expected_scale_elements,),
            device=hidden.device,
        )
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


def _load_flashinfer_fp4_quantization() -> tuple[object, object]:
    """Load the two public activation helpers without importing at module load."""

    try:
        module = importlib.import_module("flashinfer")
    except ImportError as error:
        raise RuntimeError(
            "SM120 NVFP4 attention-DP requires FlashInfer fp4_quantize and "
            "nvfp4_block_scale_interleave"
        ) from error
    quantize = getattr(module, "fp4_quantize", None)
    interleave = getattr(module, "nvfp4_block_scale_interleave", None)
    if not callable(quantize) or not callable(interleave):
        version = getattr(module, "__version__", "unknown")
        raise RuntimeError(
            "FlashInfer NVFP4 activation API mismatch "
            f"(version={version}): expected fp4_quantize and "
            "nvfp4_block_scale_interleave"
        )
    return quantize, interleave


def quantize_moe_input(
    hidden: torch.Tensor,
    input_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 rows before EP communication.

    The returned scale factors deliberately remain in FlashInfer's ordinary
    ``[tokens, hidden/16]`` layout. Callers gather those compact rows first,
    then invoke :func:`interleave_moe_input_scales` exactly once on the global
    rank-major tensor. This is the same FP4-before-communication contract used
    by SGLang's stable FlashInfer CUTLASS dispatcher.
    """

    _require_sm120(hidden)
    if hidden.ndim != 2 or hidden.dtype is not torch.bfloat16:
        raise TypeError(
            "NVFP4 MoE input quantization requires 2-D torch.bfloat16 hidden"
        )
    if not hidden.is_contiguous():
        raise ValueError("hidden must be contiguous")
    tokens, hidden_size = hidden.shape
    if tokens < 1 or hidden_size < 1 or hidden_size % 16:
        raise ValueError(
            "hidden must contain at least one row and a hidden size divisible by 16"
        )
    _require_tensor(
        input_global_scale,
        name="input_global_scale",
        dtype=torch.float32,
        shape=(),
        device=hidden.device,
    )
    quantize, _ = _load_flashinfer_fp4_quantization()
    with torch.cuda.device(hidden.device):
        result = quantize(
            hidden,
            input_global_scale,
            is_sf_swizzled_layout=False,
        )
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError("FlashInfer fp4_quantize returned a malformed result")
    packed, scales = result
    _require_tensor(
        packed,
        name="packed_hidden",
        dtype=torch.uint8,
        shape=(tokens, hidden_size // 2),
        device=hidden.device,
    )
    _require_tensor(
        scales,
        name="input_scale_rows",
        dtype=torch.uint8,
        shape=(tokens, hidden_size // 16),
        device=hidden.device,
    )
    return packed, scales


def interleave_moe_input_scales(scale_rows: torch.Tensor) -> torch.Tensor:
    """Convert gathered FP4 scale rows to CUTLASS's 128-row swizzled ABI."""

    _require_sm120(scale_rows)
    if scale_rows.ndim != 2 or scale_rows.dtype is not torch.uint8:
        raise TypeError(
            "NVFP4 MoE input scale rows must be a 2-D torch.uint8 tensor"
        )
    if not scale_rows.is_contiguous():
        raise ValueError("input scale rows must be contiguous")
    tokens, scale_columns = scale_rows.shape
    if tokens < 1 or scale_columns < 1:
        raise ValueError("input scale rows must be non-empty")
    _, interleave = _load_flashinfer_fp4_quantization()
    with torch.cuda.device(scale_rows.device):
        swizzled = interleave(scale_rows)
    expected = _ceil_div(tokens, 128) * 128 * scale_columns
    _require_tensor(
        swizzled,
        name="swizzled_input_sf",
        dtype=torch.uint8,
        shape=(expected,),
        device=scale_rows.device,
    )
    return swizzled


def _call_fused_moe(
    hidden: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    prepared: PreparedNvFp4Moe,
    *,
    output: torch.Tensor,
    input_sf: torch.Tensor | None,
    ep_size: int,
    ep_rank: int,
) -> torch.Tensor:
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
            input_sf=input_sf,
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
    # annotation says Tensor. Accept both documented and observed forms, but
    # require that the supplied output buffer is the one actually returned.
    returned = result[0] if isinstance(result, (list, tuple)) and result else result
    if not isinstance(returned, torch.Tensor) or returned.data_ptr() != output.data_ptr():
        raise RuntimeError("FlashInfer did not return the preallocated fused-MoE output buffer")
    return output


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
    return _call_fused_moe(
        hidden,
        selected_experts,
        routing_weights,
        prepared,
        output=output,
        input_sf=None,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )


def fused_moe_forward_quantized(
    packed_hidden: torch.Tensor,
    input_sf: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    prepared: PreparedNvFp4Moe,
    *,
    output: torch.Tensor,
    ep_size: int,
    ep_rank: int,
) -> torch.Tensor:
    """Run local experts from a gathered packed-NVFP4 activation tensor."""

    _validate_request(
        packed_hidden,
        selected_experts,
        routing_weights,
        prepared,
        output,
        ep_size=ep_size,
        ep_rank=ep_rank,
        input_sf=input_sf,
    )
    return _call_fused_moe(
        packed_hidden,
        selected_experts,
        routing_weights,
        prepared,
        output=output,
        input_sf=input_sf,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )
