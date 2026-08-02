"""Fused tiled GPU GEMMs for Kairyu's production quantized linear formats.

No path in this module constructs an ``[out, in]`` floating weight tensor. FP8
and INT8 quantize the much smaller activation matrix and use tensor-core dot
products; AWQ/GPTQ keep each unpacked/dequantized tile in registers only for
one Triton program, while NVFP4 dispatches to FlashInfer's native W4A4 kernels.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - reported loudly by the CUDA dispatcher
    triton = None
    tl = None
    libdevice = None


if triton is not None:

    @triton.jit
    def _scaled_activation_quant_kernel(
        x_ptr,
        quant_ptr,
        input_scale_ptr,
        output_scale_ptr,
        k_size,
        stride_xm,
        stride_xk,
        stride_qm,
        stride_qk,
        QUANT_MAX: tl.constexpr,
        DYNAMIC: tl.constexpr,
        ROW_SCALE: tl.constexpr,
        ROUND_TO_INT: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K)
        mask = offsets < k_size
        values = tl.load(
            x_ptr + row * stride_xm + offsets * stride_xk,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if DYNAMIC:
            amax = tl.max(tl.abs(values), axis=0)
            scale = tl.maximum(amax / QUANT_MAX, 1e-12)
        elif ROW_SCALE:
            scale = tl.load(input_scale_ptr + row).to(tl.float32)
        else:
            scale = tl.load(input_scale_ptr).to(tl.float32)
        scaled = tl.maximum(tl.minimum(values / scale, QUANT_MAX), -QUANT_MAX)
        if ROUND_TO_INT:
            scaled = libdevice.rint(scaled)
        tl.store(
            quant_ptr + row * stride_qm + offsets * stride_qk,
            scaled,
            mask=mask,
        )
        tl.store(output_scale_ptr + row, scale)

    @triton.jit
    def _row_amax_kernel(
        x_ptr,
        output_ptr,
        k_size,
        stride_xm,
        stride_xk,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K)
        values = tl.load(
            x_ptr + row * stride_xm + offsets * stride_xk,
            mask=offsets < k_size,
            other=0.0,
        ).to(tl.float32)
        tl.store(output_ptr + row, tl.max(tl.abs(values), axis=0))

    @triton.jit
    def _fp8_w8a8_kernel(
        x_ptr,
        weight_ptr,
        x_scale_ptr,
        weight_scale_ptr,
        bias_ptr,
        out_ptr,
        m_size,
        n_size,
        k_size,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        HAS_BIAS: tl.constexpr,
        PER_CHANNEL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        blocks_n = tl.cdiv(n_size, BLOCK_N)
        pid_m = pid // blocks_n
        pid_n = pid % blocks_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_block in range(0, tl.cdiv(k_size, BLOCK_K)):
            current_k = k_block * BLOCK_K + offs_k
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + current_k[None, :] * stride_xk,
                mask=(offs_m[:, None] < m_size) & (current_k[None, :] < k_size),
                other=0.0,
            )
            weight = tl.load(
                weight_ptr
                + offs_n[None, :] * stride_wn
                + current_k[:, None] * stride_wk,
                mask=(offs_n[None, :] < n_size) & (current_k[:, None] < k_size),
                other=0.0,
            )
            accumulator += tl.dot(x, weight)
        x_scale = tl.load(x_scale_ptr + offs_m, mask=offs_m < m_size, other=0.0)
        if PER_CHANNEL:
            weight_scale = tl.load(
                weight_scale_ptr + offs_n, mask=offs_n < n_size, other=0.0
            )
        else:
            weight_scale = tl.load(weight_scale_ptr)
        output = accumulator * x_scale[:, None] * weight_scale[None, :]
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < n_size, other=0.0)
            output += bias[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            output,
            mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size),
        )

    @triton.jit
    def _int8_w8a8_kernel(
        x_ptr,
        weight_ptr,
        x_scale_ptr,
        weight_scale_ptr,
        bias_ptr,
        out_ptr,
        m_size,
        n_size,
        k_size,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        blocks_n = tl.cdiv(n_size, BLOCK_N)
        pid_m = pid // blocks_n
        pid_n = pid % blocks_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k_block in range(0, tl.cdiv(k_size, BLOCK_K)):
            current_k = k_block * BLOCK_K + offs_k
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + current_k[None, :] * stride_xk,
                mask=(offs_m[:, None] < m_size) & (current_k[None, :] < k_size),
                other=0,
            )
            weight = tl.load(
                weight_ptr
                + offs_n[None, :] * stride_wn
                + current_k[:, None] * stride_wk,
                mask=(offs_n[None, :] < n_size) & (current_k[:, None] < k_size),
                other=0,
            )
            accumulator += tl.dot(x, weight)
        x_scale = tl.load(x_scale_ptr + offs_m, mask=offs_m < m_size, other=0.0)
        weight_scale = tl.load(
            weight_scale_ptr + offs_n, mask=offs_n < n_size, other=0.0
        )
        output = accumulator.to(tl.float32) * x_scale[:, None] * weight_scale[None, :]
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < n_size, other=0.0)
            output += bias[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            output,
            mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size),
        )

    @triton.jit
    def _awq_w4a16_kernel(
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        bias_ptr,
        out_ptr,
        m_size,
        n_size,
        k_size,
        stride_xm,
        stride_xk,
        stride_qwk,
        stride_qwn,
        stride_qzg,
        stride_qzn,
        stride_sg,
        stride_sn,
        stride_om,
        stride_on,
        GROUP_SIZE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        X_IS_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        blocks_n = tl.cdiv(n_size, BLOCK_N)
        pid_m = pid // blocks_n
        pid_n = pid % blocks_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        packed_n = offs_n // 8
        nibble_n = offs_n % 8
        # AWQ packing order [0,2,4,6,1,3,5,7], inverted without a lookup.
        packed_position = nibble_n // 2 + (nibble_n % 2) * 4
        shift_n = packed_position * 4
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_block in range(0, tl.cdiv(k_size, BLOCK_K)):
            current_k = k_block * BLOCK_K + offs_k
            groups = current_k // GROUP_SIZE
            qword = tl.load(
                qweight_ptr
                + current_k[:, None] * stride_qwk
                + packed_n[None, :] * stride_qwn,
                mask=(current_k[:, None] < k_size) & (offs_n[None, :] < n_size),
                other=0,
            )
            zword = tl.load(
                qzeros_ptr
                + groups[:, None] * stride_qzg
                + packed_n[None, :] * stride_qzn,
                mask=(current_k[:, None] < k_size) & (offs_n[None, :] < n_size),
                other=0,
            )
            quant = (qword >> shift_n[None, :]) & 0xF
            zero = (zword >> shift_n[None, :]) & 0xF
            scales = tl.load(
                scales_ptr
                + groups[:, None] * stride_sg
                + offs_n[None, :] * stride_sn,
                mask=(current_k[:, None] < k_size) & (offs_n[None, :] < n_size),
                other=0.0,
            )
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + current_k[None, :] * stride_xk,
                mask=(offs_m[:, None] < m_size) & (current_k[None, :] < k_size),
                other=0.0,
            )
            weight = (quant - zero).to(tl.float32) * scales
            weight = weight.to(tl.bfloat16) if X_IS_BF16 else weight.to(tl.float16)
            accumulator += tl.dot(x, weight)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < n_size, other=0.0)
            accumulator += bias[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            accumulator,
            mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size),
        )

    @triton.jit
    def _gptq_w4a16_kernel(
        x_ptr,
        qweight_ptr,
        qzeros_ptr,
        scales_ptr,
        g_idx_ptr,
        bias_ptr,
        out_ptr,
        m_size,
        n_size,
        k_size,
        stride_xm,
        stride_xk,
        stride_qwk,
        stride_qwn,
        stride_qzg,
        stride_qzn,
        stride_sg,
        stride_sn,
        stride_om,
        stride_on,
        HAS_BIAS: tl.constexpr,
        X_IS_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        blocks_n = tl.cdiv(n_size, BLOCK_N)
        pid_m = pid // blocks_n
        pid_n = pid % blocks_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_block in range(0, tl.cdiv(k_size, BLOCK_K)):
            current_k = k_block * BLOCK_K + offs_k
            groups = tl.load(
                g_idx_ptr + current_k, mask=current_k < k_size, other=0
            )
            qword = tl.load(
                qweight_ptr
                + (current_k[:, None] // 8) * stride_qwk
                + offs_n[None, :] * stride_qwn,
                mask=(current_k[:, None] < k_size) & (offs_n[None, :] < n_size),
                other=0,
            )
            zword = tl.load(
                qzeros_ptr
                + groups[:, None] * stride_qzg
                + (offs_n[None, :] // 8) * stride_qzn,
                mask=(current_k[:, None] < k_size) & (offs_n[None, :] < n_size),
                other=0,
            )
            quant = (qword >> ((current_k[:, None] % 8) * 4)) & 0xF
            # GPTQ stores zero - 1, packed sequentially along output.
            zero = ((zword >> ((offs_n[None, :] % 8) * 4)) & 0xF) + 1
            scales = tl.load(
                scales_ptr
                + groups[:, None] * stride_sg
                + offs_n[None, :] * stride_sn,
                mask=(current_k[:, None] < k_size) & (offs_n[None, :] < n_size),
                other=0.0,
            )
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + current_k[None, :] * stride_xk,
                mask=(offs_m[:, None] < m_size) & (current_k[None, :] < k_size),
                other=0.0,
            )
            weight = (quant - zero).to(tl.float32) * scales
            weight = weight.to(tl.bfloat16) if X_IS_BF16 else weight.to(tl.float16)
            accumulator += tl.dot(x, weight)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < n_size, other=0.0)
            accumulator += bias[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            accumulator,
            mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size),
        )

_CAPABILITY_FLOORS = {
    "fp8": (8, 9),
    "int8": (8, 0),
    "awq": (8, 0),
    "gptq": (8, 0),
    "nvfp4": (10, 0),
}


def _prepare(x: torch.Tensor, module, scheme: str) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None and scheme in ("fp8", "int8", "awq", "gptq"):
        raise RuntimeError(
            f"{scheme} CUDA execution requires Triton; install the gpu extra"
        )
    if not x.is_cuda:
        raise RuntimeError(f"{scheme} fused kernel requires a CUDA activation")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"{scheme} fused kernel supports float16/bfloat16 activations, got {x.dtype}"
        )
    capability = torch.cuda.get_device_capability(x.device)
    floor = _CAPABILITY_FLOORS[scheme]
    if capability < floor:
        raise RuntimeError(
            f"{scheme} fused kernel requires SM{floor[0]}{floor[1]}+, "
            f"got SM{capability[0]}{capability[1]}; no dequantized GPU fallback exists"
        )
    if x.shape[-1] != module.in_features:
        raise ValueError(
            f"{scheme} linear expected K={module.in_features}, got {x.shape[-1]}"
        )
    payload = list(module.buffers(recurse=False))
    if module.bias is not None:
        payload.append(module.bias)
    misplaced = [tensor.device for tensor in payload if tensor.device != x.device]
    if misplaced:
        raise RuntimeError(
            f"{scheme} fused kernel requires module payloads on {x.device}; "
            f"found {misplaced[0]}"
        )
    _validate_layout(module, scheme)
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    output = torch.empty(
        (flat.shape[0], module.out_features), device=x.device, dtype=x.dtype
    )
    return flat, output


def _validate_layout(module, scheme: str) -> None:
    n_size, k_size = module.out_features, module.in_features
    if scheme == "fp8":
        if module.weight.shape != (n_size, k_size):
            raise ValueError(f"FP8 weight shape {module.weight.shape} != {(n_size, k_size)}")
        if module.weight_scale.numel() not in (1, n_size):
            raise ValueError(
                "FP8 weight_scale must be scalar or per-output-channel, got "
                f"{module.weight_scale.shape}"
            )
        if not module.activation_dynamic and module.input_scale.numel() != 1:
            raise ValueError("static FP8 input_scale must be scalar")
    elif scheme == "int8":
        if module.weight.shape != (n_size, k_size):
            raise ValueError(
                f"INT8 weight shape {module.weight.shape} != {(n_size, k_size)}"
            )
        if module.weight_scale.numel() != n_size:
            raise ValueError("INT8 weight_scale must be per-output-channel")
    elif scheme == "awq":
        expected = (
            (k_size, n_size // 8),
            (k_size // module.group_size, n_size // 8),
            (k_size // module.group_size, n_size),
        )
        actual = (module.qweight.shape, module.qzeros.shape, module.scales.shape)
        if actual != expected:
            raise ValueError(f"AWQ packed layout {actual} != {expected}")
    elif scheme == "gptq":
        groups = -(-k_size // module.group_size)
        expected = (
            (k_size // 8, n_size),
            (groups, n_size // 8),
            (groups, n_size),
            (k_size,),
        )
        actual = (
            module.qweight.shape,
            module.qzeros.shape,
            module.scales.shape,
            module.g_idx.shape,
        )
        if actual != expected:
            raise ValueError(f"GPTQ packed layout {actual} != {expected}")
    else:
        expected = (
            (n_size, k_size // 2),
            (n_size, k_size // 16),
        )
        actual = (module.weight.shape, module.weight_scale.shape)
        if actual != expected:
            raise ValueError(f"NVFP4 packed layout {actual} != {expected}")


def _grid(m_size: int, n_size: int):
    return lambda meta: (
        triton.cdiv(m_size, meta["BLOCK_M"])
        * triton.cdiv(n_size, meta["BLOCK_N"]),
    )


def _bias(module, output: torch.Tensor, add_bias: bool):
    return module.bias if add_bias and module.bias is not None else output


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _quantize_activation(
    flat: torch.Tensor,
    dtype: torch.dtype,
    quant_max: float,
    static_scale: torch.Tensor | None = None,
    row_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if static_scale is not None and row_scale is not None:
        raise ValueError("activation quantization accepts one scale source")
    quantized = torch.empty_like(flat, dtype=dtype)
    scales = torch.empty((flat.shape[0], 1), device=flat.device, dtype=torch.float32)
    if row_scale is not None:
        row_scale = row_scale.reshape(flat.shape[0], 1).to(
            device=flat.device, dtype=torch.float32
        ).contiguous()
    input_scale = row_scale if row_scale is not None else static_scale
    block_k = triton.next_power_of_2(flat.shape[1])
    _scaled_activation_quant_kernel[(flat.shape[0],)](
        flat,
        quantized,
        scales if input_scale is None else input_scale,
        scales,
        flat.shape[1],
        flat.stride(0),
        flat.stride(1),
        quantized.stride(0),
        quantized.stride(1),
        QUANT_MAX=quant_max,
        DYNAMIC=input_scale is None,
        ROW_SCALE=row_scale is not None,
        ROUND_TO_INT=dtype is torch.int8,
        BLOCK_K=block_k,
        num_warps=8 if block_k >= 2048 else 4,
    )
    return quantized, scales


def fp8_activation_amax(x: torch.Tensor) -> torch.Tensor:
    """Return per-token FP32 amax without a full FP32 activation temporary."""
    if triton is None:
        raise RuntimeError("Triton is required for FP8 CUDA activation scaling")
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    output = torch.empty(
        (flat.shape[0], 1), device=flat.device, dtype=torch.float32
    )
    block_k = triton.next_power_of_2(flat.shape[1])
    _row_amax_kernel[(flat.shape[0],)](
        flat,
        output,
        flat.shape[1],
        flat.stride(0),
        flat.stride(1),
        BLOCK_K=block_k,
        num_warps=8 if block_k >= 2048 else 4,
    )
    return output.reshape(*x.shape[:-1], 1)


def fp8_linear_forward(
    x: torch.Tensor,
    module,
    activation_scale: torch.Tensor | None = None,
    *,
    add_bias: bool = True,
) -> torch.Tensor:
    flat, output = _prepare(x, module, "fp8")
    # Dynamic per-token W8A8 activation quantization. Materializing this [M,K]
    # tensor is bounded by the live activation; the forbidden [N,K] floating
    # dequantized weight is never created.
    input_scale = None if module.activation_dynamic else module.input_scale
    x_quant, x_scale = _quantize_activation(
        flat,
        torch.float8_e4m3fn,
        448.0,
        static_scale=input_scale,
        row_scale=activation_scale,
    )
    per_channel = module.weight_scale.numel() != 1
    if module.in_features % 16 == 0 and module.out_features % 16 == 0:
        weight_scale = module.weight_scale.reshape(-1)
        if not per_channel:
            weight_scale = weight_scale.expand(module.out_features)
        output = torch._scaled_mm(
            x_quant,
            module.weight.T,
            x_scale.reshape(-1, 1).contiguous(),
            weight_scale.reshape(1, -1).contiguous(),
            out_dtype=x.dtype,
            use_fast_accum=False,
        )
        if add_bias and module.bias is not None:
            output = output + module.bias
        return output.reshape(*x.shape[:-1], module.out_features)
    _fp8_w8a8_kernel[_grid(flat.shape[0], module.out_features)](
        x_quant,
        module.weight,
        x_scale,
        module.weight_scale,
        _bias(module, output, add_bias),
        output,
        flat.shape[0],
        module.out_features,
        module.in_features,
        x_quant.stride(0),
        x_quant.stride(1),
        module.weight.stride(0),
        module.weight.stride(1),
        output.stride(0),
        output.stride(1),
        HAS_BIAS=add_bias and module.bias is not None,
        PER_CHANNEL=per_channel,
        BLOCK_M=16,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    return output.reshape(*x.shape[:-1], module.out_features)


def int8_linear_forward(
    x: torch.Tensor,
    module,
    *,
    add_bias: bool = True,
) -> torch.Tensor:
    flat, output = _prepare(x, module, "int8")
    x_quant, x_scale = _quantize_activation(flat, torch.int8, 127.0)
    _int8_w8a8_kernel[_grid(flat.shape[0], module.out_features)](
        x_quant,
        module.weight,
        x_scale,
        module.weight_scale,
        _bias(module, output, add_bias),
        output,
        flat.shape[0],
        module.out_features,
        module.in_features,
        x_quant.stride(0),
        x_quant.stride(1),
        module.weight.stride(0),
        module.weight.stride(1),
        output.stride(0),
        output.stride(1),
        HAS_BIAS=add_bias and module.bias is not None,
        BLOCK_M=16,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    return output.reshape(*x.shape[:-1], module.out_features)


def awq_linear_forward(
    x: torch.Tensor,
    module,
    *,
    add_bias: bool = True,
) -> torch.Tensor:
    flat, output = _prepare(x, module, "awq")
    _awq_w4a16_kernel[_grid(flat.shape[0], module.out_features)](
        flat,
        module.qweight,
        module.qzeros,
        module.scales,
        _bias(module, output, add_bias),
        output,
        flat.shape[0],
        module.out_features,
        module.in_features,
        flat.stride(0),
        flat.stride(1),
        module.qweight.stride(0),
        module.qweight.stride(1),
        module.qzeros.stride(0),
        module.qzeros.stride(1),
        module.scales.stride(0),
        module.scales.stride(1),
        output.stride(0),
        output.stride(1),
        GROUP_SIZE=module.group_size,
        HAS_BIAS=add_bias and module.bias is not None,
        X_IS_BF16=x.dtype is torch.bfloat16,
        BLOCK_M=16,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    return output.reshape(*x.shape[:-1], module.out_features)


def gptq_linear_forward(
    x: torch.Tensor,
    module,
    *,
    add_bias: bool = True,
) -> torch.Tensor:
    flat, output = _prepare(x, module, "gptq")
    _gptq_w4a16_kernel[_grid(flat.shape[0], module.out_features)](
        flat,
        module.qweight,
        module.qzeros,
        module.scales,
        module.g_idx,
        _bias(module, output, add_bias),
        output,
        flat.shape[0],
        module.out_features,
        module.in_features,
        flat.stride(0),
        flat.stride(1),
        module.qweight.stride(0),
        module.qweight.stride(1),
        module.qzeros.stride(0),
        module.qzeros.stride(1),
        module.scales.stride(0),
        module.scales.stride(1),
        output.stride(0),
        output.stride(1),
        HAS_BIAS=add_bias and module.bias is not None,
        X_IS_BF16=x.dtype is torch.bfloat16,
        BLOCK_M=16,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    return output.reshape(*x.shape[:-1], module.out_features)


def _prepare_nvfp4_weight(module, device: torch.device):
    source_key = (
        id(module.weight),
        module.weight._version,
        id(module.weight_scale),
        module.weight_scale._version,
        device.type,
        device.index,
    )
    cached_source_key = getattr(module, "_nvfp4_native_source_key", None)
    cached_weight = getattr(module, "_weight_native", None)
    cached_scale = getattr(module, "_weight_scale_swizzled", None)
    if cached_source_key == source_key and cached_scale is not None:
        if cached_weight is None:
            if module.weight.device == device and cached_scale.device == device:
                return module.weight, cached_scale
        elif cached_weight.device == device and cached_scale.device == device:
            return cached_weight, cached_scale

    n_size, packed_k = module.weight.shape
    n_padded = _ceil_div(n_size, 32) * 32
    k_size = packed_k * 2
    k_padded = _ceil_div(k_size, 32) * 32
    pad_k = k_padded // 2 - packed_k
    pad_n = n_padded - n_size
    if pad_k == 0 and pad_n == 0 and module.weight.is_contiguous():
        # Real Qwen3 NVFP4 expert matrices already satisfy FlashInfer's N/K
        # alignment.  ``F.pad(..., zeros)`` still allocates and copies the full
        # packed weight; warming every expert in a 235B EP2 model would retain a
        # second ~64 GiB weight set and eventually OOM.  Keep the checkpoint
        # buffer itself as the native operand when no padding is required.
        weight = module.weight
    else:
        weight = torch.nn.functional.pad(
            module.weight,
            (0, pad_k, 0, pad_n),
        ).contiguous()

    scale = module.weight_scale
    scale_n_padded = _ceil_div(scale.shape[0], 128) * 128
    scale_k_padded = _ceil_div(k_padded // 16, 4) * 4
    scale = torch.nn.functional.pad(
        scale,
        (0, scale_k_padded - scale.shape[1], 0, scale_n_padded - scale.shape[0]),
    )
    scale = scale.reshape(1, scale_n_padded // 128, 4, 32, scale_k_padded // 4, 4)
    scale = scale.permute(0, 1, 4, 3, 2, 5).contiguous()
    scale = scale.reshape(scale_n_padded, scale_k_padded)
    if weight is module.weight:
        # Do not retain a plain tensor alias: unlike a registered buffer it is
        # invisible to ``Module._apply`` and would keep the old GPU allocation
        # alive after a device move. A source-key cache hit returns
        # ``module.weight`` directly.
        module._buffers.pop("_weight_native", None)
        if "_weight_native" in module.__dict__:
            object.__delattr__(module, "_weight_native")
    else:
        module.register_buffer("_weight_native", weight, persistent=False)
    module.register_buffer("_weight_scale_swizzled", scale, persistent=False)
    object.__setattr__(module, "_nvfp4_native_source_key", source_key)
    return weight, scale


def nvfp4_linear_forward(
    x: torch.Tensor,
    module,
    *,
    add_bias: bool = True,
) -> torch.Tensor:
    flat, _ = _prepare(x, module, "nvfp4")
    try:
        from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
    except ImportError as error:
        raise RuntimeError(
            "NVFP4 CUDA execution requires FlashInfer's native FP4 kernels; "
            "no W4A16 or dequantized fallback is enabled"
        ) from error

    native_weight, native_scale = _prepare_nvfp4_weight(module, flat.device)
    k_padded = native_weight.shape[1] * 2
    if flat.shape[1] != k_padded:
        flat = torch.nn.functional.pad(flat, (0, k_padded - flat.shape[1]))
    input_scale_inv = module.input_scale.float().reciprocal().reshape(1)
    x_fp4, x_scale = nvfp4_quantize(
        flat,
        input_scale_inv,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    alpha = (module.input_scale.float() * module.weight_scale_2.float()).reshape(())
    output = mm_fp4(
        x_fp4,
        native_weight.T,
        x_scale,
        native_scale,
        alpha=alpha,
        out_dtype=x.dtype,
        backend="auto",
    )
    if add_bias and module.bias is not None:
        output = output[:, : module.out_features] + module.bias
    else:
        output = output[:, : module.out_features]
    return output.reshape(*x.shape[:-1], module.out_features)


def linear_forward(
    x: torch.Tensor,
    module,
    *,
    add_bias: bool = True,
) -> torch.Tensor:
    scheme = getattr(module, "quant_scheme", None)
    dispatch = {
        "fp8": fp8_linear_forward,
        "int8": int8_linear_forward,
        "awq": awq_linear_forward,
        "gptq": gptq_linear_forward,
        "nvfp4": nvfp4_linear_forward,
    }
    if scheme not in dispatch:
        raise RuntimeError(f"no fused GPU kernel registered for {scheme!r}")
    return dispatch[scheme](x, module, add_bias=add_bias)
