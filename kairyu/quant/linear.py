"""QuantizedLinear modules + the loader's linear factory (m14 D2).

Each module holds the PACKED tensors as persistent buffers under the
checkpoint's own names (our convention — review A9), so the loader assigns by
name with zero renaming. CPU ``forward`` uses a correctness oracle; CUDA
``forward`` dispatches to a fused kernel that never materializes the full
floating-point weight.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from kairyu.engine.core.quant_config import QuantConfig, QuantMethod
from kairyu.quant import awq, fp8, gptq, int8, nvfp4


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

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize().to(x.dtype)
        return nn.functional.linear(x, weight, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cuda":
            return self.forward_fused(x)
        return self.forward_reference(x)

    def forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Production CUDA dispatch. Unsupported paths fail; none dequantize."""
        from kairyu.kernels.quant_gemm_gpu import linear_forward

        return linear_forward(x, self)


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

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        input_scale = None if self.activation_dynamic else self.input_scale
        x_q, x_scale = fp8.quantize_fp8_activation(
            x.to(torch.float32), scale=input_scale
        )
        out = fp8.fp8_w8a8_matmul(
            x_q, x_scale, self.weight, self.weight_scale
        )
        if self.bias is not None:
            out = out + self.bias
        return out.to(x.dtype)


class Int8Linear(QuantizedLinearBase):
    """compressed-tensors INT8 W8A8: dynamic per-token activations."""

    quant_scheme = "int8"

    def __init__(self, in_features: int, out_features: int, bias: bool) -> None:
        super().__init__(in_features, out_features, bias)
        self.register_buffer(
            "weight", torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer("weight_scale", torch.ones(out_features, 1))

    def dequantize(self) -> torch.Tensor:
        return int8.dequantize_int8(self.weight, self.weight_scale)

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        # exact W8A8 reference: int32 accumulation (the GPU kernel's oracle)
        x_q, x_scale = int8.quantize_int8_activation(x.to(torch.float32))
        out = int8.int8_w8a8_matmul(x_q, x_scale, self.weight, self.weight_scale)
        if self.bias is not None:
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
        self.register_buffer(
            "qzeros", torch.zeros(groups, out_features // 8, dtype=torch.int32)
        )
        self.register_buffer(
            "scales", torch.ones(groups, out_features, dtype=torch.float16)
        )

    def dequantize(self) -> torch.Tensor:
        return awq.dequantize_awq(self.qweight, self.qzeros, self.scales, self.group_size)


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
        self.register_buffer(
            "qzeros", torch.zeros(groups, out_features // 8, dtype=torch.int32)
        )
        self.register_buffer(
            "scales", torch.ones(groups, out_features, dtype=torch.float16)
        )
        self.register_buffer("g_idx", torch.zeros(in_features, dtype=torch.int32))

    def dequantize(self) -> torch.Tensor:
        return gptq.dequantize_gptq(
            self.qweight, self.qzeros, self.scales, self.g_idx, self.group_size
        )


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

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
        packed, scale = nvfp4.quantize_nvfp4_with_scale(flat, self.input_scale)
        activation = nvfp4.dequantize_nvfp4(packed, scale, self.input_scale)
        weight = self.dequantize()
        out = nn.functional.linear(activation, weight, self.bias)
        return out.reshape(*x.shape[:-1], self.out_features).to(x.dtype)


LinearFactory = Callable[[int, int, bool], nn.Module]


def linear_factory(quant: QuantConfig) -> LinearFactory:
    """QuantConfig -> constructor for the model's projection layers (m12 hook)."""
    if quant.method is QuantMethod.NONE:
        return lambda i, o, b: nn.Linear(i, o, bias=b)
    if quant.method is QuantMethod.FP8:
        dynamic = quant.activation_dynamic is not False
        return lambda i, o, b: Fp8Linear(i, o, b, activation_dynamic=dynamic)
    if quant.method is QuantMethod.INT8:
        return lambda i, o, b: Int8Linear(i, o, b)
    if quant.method is QuantMethod.AWQ:
        group = quant.group_size or 128
        return lambda i, o, b: AwqLinear(i, o, b, group_size=group)
    if quant.method is QuantMethod.GPTQ:
        group = quant.group_size or 128
        return lambda i, o, b: GptqLinear(i, o, b, group_size=group)
    if quant.method is QuantMethod.NVFP4:
        return lambda i, o, b: NvFp4Linear(i, o, b)
    raise ValueError(f"no QuantizedLinear for {quant.method}")  # pragma: no cover
