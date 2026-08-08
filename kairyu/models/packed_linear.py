"""Inference-only dense projection packing without checkpoint path changes.

Hugging Face stores Q, K, and V as separate tensors. Executing those tensors
as separate ``nn.Linear`` modules needlessly launches three GEMMs.
``DenseLinearPack`` gives the canonical projection parameters views into one
output-row-contiguous allocation and executes that allocation with one
``functional.linear`` call.

The pack is deliberately a plain Python object, not an ``nn.Module``:

* ``state_dict`` / ``named_parameters`` retain the exact HF projection paths;
* contextual quantization and TP metadata remain on the original modules;
* unsupported/custom projections and incompatible quantized groups keep their
  existing forwards.

Kairyu is an inference framework.  The unregistered packed base tensor is
therefore an execution view, not an optimizer parameter; callers that install
projection hooks or replace one canonical parameter automatically take the
separate-projection compatibility path.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class DenseLinearPack:
    """One contiguous dense weight and its canonical projection views."""

    __slots__ = (
        "_bias",
        "_bias_offsets",
        "_out_features",
        "_projections",
        "_weight",
        "_weight_offsets",
    )

    def __init__(
        self,
        projections: tuple[nn.Linear, ...],
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: tuple[int, ...],
        weight_offsets: tuple[int, ...],
        bias_offsets: tuple[int, ...] | None,
    ) -> None:
        self._projections = projections
        self._weight = weight
        self._bias = bias
        self._out_features = out_features
        self._weight_offsets = weight_offsets
        self._bias_offsets = bias_offsets

    @classmethod
    def create(
        cls,
        projections: Sequence[nn.Module],
    ) -> DenseLinearPack | None:
        """Pack compatible plain dense projections, otherwise return ``None``.

        Exact ``nn.Linear`` matching is intentional.  Quantized modules,
        subclasses, parametrizations, mixed bias layouts, and a row-parallel
        linear with an overridden forward all preserve their established
        behavior rather than being guessed at here.
        """

        if len(projections) < 2 or any(type(module) is not nn.Linear for module in projections):
            return None
        linears = tuple(projections)
        first = linears[0]
        if first.weight.device.type == "meta":
            return None
        if any(
            module.in_features != first.in_features
            or module.weight.device != first.weight.device
            or module.weight.dtype != first.weight.dtype
            or module.weight.layout is not torch.strided
            or type(module.weight) is not nn.Parameter
            or "forward" in module.__dict__
            or module._forward_hooks
            or module._forward_pre_hooks
            or module._backward_hooks
            for module in linears
        ):
            return None
        requires_grad = first.weight.requires_grad
        if any(module.weight.requires_grad != requires_grad for module in linears):
            return None
        has_bias = first.bias is not None
        if any((module.bias is not None) != has_bias for module in linears):
            return None
        if has_bias and any(
            type(module.bias) is not nn.Parameter
            or module.bias.device != first.bias.device
            or module.bias.dtype != first.bias.dtype
            or module.bias.requires_grad != first.bias.requires_grad
            for module in linears
        ):
            return None

        out_features = tuple(module.out_features for module in linears)
        weight_offsets = tuple(
            sum(out_features[:index]) * first.in_features
            for index in range(len(linears))
        )
        with torch.no_grad():
            weight = torch.cat(
                tuple(module.weight.detach() for module in linears),
                dim=0,
            ).contiguous()
            bias = (
                torch.cat(
                    tuple(module.bias.detach() for module in linears),
                    dim=0,
                ).contiguous()
                if has_bias
                else None
            )

        row_offset = 0
        for module, rows in zip(linears, out_features, strict=True):
            module.weight = nn.Parameter(
                weight.narrow(0, row_offset, rows),
                requires_grad=requires_grad,
            )
            if bias is not None:
                assert module.bias is not None
                module.bias = nn.Parameter(
                    bias.narrow(0, row_offset, rows),
                    requires_grad=module.bias.requires_grad,
                )
            row_offset += rows
        bias_offsets = (
            tuple(sum(out_features[:index]) for index in range(len(linears)))
            if bias is not None
            else None
        )
        return cls(
            linears,
            weight,
            bias,
            out_features,
            weight_offsets,
            bias_offsets,
        )

    @property
    def out_features(self) -> tuple[int, ...]:
        return self._out_features

    def matches(self, projections: Sequence[nn.Module]) -> bool:
        """Whether the owner's canonical module attributes are still ours."""

        return len(projections) == len(self._projections) and all(
            current is packed
            for current, packed in zip(
                projections,
                self._projections,
                strict=True,
            )
        )

    def is_current(self) -> bool:
        """Whether canonical parameters still describe this packed storage."""

        weight_storage = self._weight.untyped_storage().data_ptr()
        base_weight_offset = self._weight.storage_offset()
        for module, rows, offset in zip(
            self._projections,
            self._out_features,
            self._weight_offsets,
            strict=True,
        ):
            parameter = module.weight
            if (
                type(module) is not nn.Linear
                or "forward" in module.__dict__
                or module._forward_hooks
                or module._forward_pre_hooks
                or module._backward_hooks
                or tuple(parameter.shape) != (rows, module.in_features)
                or parameter.stride() != (module.in_features, 1)
                or parameter.untyped_storage().data_ptr() != weight_storage
                or parameter.storage_offset() != base_weight_offset + offset
            ):
                return False

        if self._bias is None:
            return all(module.bias is None for module in self._projections)
        assert self._bias_offsets is not None
        bias_storage = self._bias.untyped_storage().data_ptr()
        base_bias_offset = self._bias.storage_offset()
        for module, rows, offset in zip(
            self._projections,
            self._out_features,
            self._bias_offsets,
            strict=True,
        ):
            parameter = module.bias
            if (
                parameter is None
                or tuple(parameter.shape) != (rows,)
                or parameter.stride() != (1,)
                or parameter.untyped_storage().data_ptr() != bias_storage
                or parameter.storage_offset() != base_bias_offset + offset
            ):
                return False
        return True

    def __call__(self, hidden: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not self.is_current():
            raise RuntimeError("canonical projection parameters no longer match packed storage")
        output = nn.functional.linear(hidden, self._weight, self._bias)
        return output.split(self._out_features, dim=-1)


class NvFp4LinearPack:
    """One FlashInfer W4A4 call over compatible canonical NVFP4 projections.

    Qwen3 NVFP4 checkpoints give Q, K, and V identical activation and global
    weight scales.  Their output-row tensors can therefore share both one
    activation quantization and one concatenated GEMM without changing the
    checkpoint's canonical module paths.  A mismatch in type, hooks, layout,
    placement, or either scalar scale declines the optimization.
    """

    __slots__ = (
        "_bias",
        "_bias_offsets",
        "_alpha",
        "_input_scale",
        "_input_scale_inv",
        "_native_weight",
        "_out_features",
        "_projections",
        "_source_key",
        "_weight",
        "_weight_offsets",
        "_weight_scale",
        "_weight_scale_2",
        "_weight_scale_offsets",
        "_weight_scale_swizzled",
    )

    def __init__(
        self,
        projections: tuple[nn.Module, ...],
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor,
        input_scale: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: tuple[int, ...],
        weight_offsets: tuple[int, ...],
        weight_scale_offsets: tuple[int, ...],
        bias_offsets: tuple[int, ...] | None,
    ) -> None:
        self._projections = projections
        self._weight = weight
        self._weight_scale = weight_scale
        self._weight_scale_2 = weight_scale_2
        self._input_scale = input_scale
        self._input_scale_inv = None
        self._alpha = None
        self._bias = bias
        self._out_features = out_features
        self._weight_offsets = weight_offsets
        self._weight_scale_offsets = weight_scale_offsets
        self._bias_offsets = bias_offsets
        self._native_weight = None
        self._weight_scale_swizzled = None
        self._source_key = self._canonical_source_key()
        if weight.device.type == "cuda":
            self.prepare_native_operands(weight.device)

    @classmethod
    def create(
        cls,
        projections: Sequence[nn.Module],
    ) -> NvFp4LinearPack | None:
        """Pack exact compatible ``NvFp4Linear`` modules or return ``None``."""

        from kairyu.quant.linear import NvFp4Linear

        if len(projections) < 2 or any(
            type(module) is not NvFp4Linear for module in projections
        ):
            return None
        modules = tuple(projections)
        first = modules[0]
        if first.weight.device.type == "meta":
            return None
        target = getattr(getattr(first, "linear_context", None), "device", None)
        if (
            target is not None
            and target.type == "cuda"
            and target.index is None
            and first.weight.device.type == "cuda"
        ):
            target = first.weight.device
        if target is not None and first.weight.device != target:
            # EP builds checkpoint buffers on CPU before one bulk model.to().
            # Packing that transient copy doubles the replicated QKV payload.
            return None
        if any(
            module.in_features != first.in_features
            or bool(getattr(module, "activation_dynamic", False))
            or bool(getattr(module, "observe_activation_saturation", False))
            or module.weight.device != first.weight.device
            or module.weight.dtype is not torch.uint8
            or module.weight.layout is not torch.strided
            or not module.weight.is_contiguous()
            or module.weight_scale.device != first.weight.device
            or module.weight_scale.dtype is not torch.float8_e4m3fn
            or module.weight_scale.layout is not torch.strided
            or not module.weight_scale.is_contiguous()
            or module.input_scale.device != first.weight.device
            or module.input_scale.numel() != 1
            or module.weight_scale_2.device != first.weight.device
            or module.weight_scale_2.numel() != 1
            or "forward" in module.__dict__
            or module._forward_hooks
            or module._forward_pre_hooks
            or module._backward_hooks
            or module._backward_pre_hooks
            or bool(getattr(module, "_kairyu_row_parallel_omit_bias", False))
            for module in modules
        ):
            return None
        if any(
            module.weight.shape
            != (module.out_features, module.in_features // 2)
            or module.weight_scale.shape
            != (module.out_features, module.in_features // 16)
            for module in modules
        ):
            return None
        if any(
            not torch.equal(module.input_scale, first.input_scale)
            or not torch.equal(module.weight_scale_2, first.weight_scale_2)
            for module in modules[1:]
        ):
            return None
        if first.weight.device.type == "cuda":
            from kairyu.kernels.quant_gemm_gpu import nvfp4_linear_forward

            if any(
                getattr(module, "_fused_forward", None)
                not in (None, nvfp4_linear_forward)
                for module in modules
            ):
                return None

        has_bias = first.bias is not None
        if any((module.bias is not None) != has_bias for module in modules):
            return None
        if has_bias and any(
            type(module.bias) is not nn.Parameter
            or module.bias.device != first.bias.device
            or module.bias.dtype != first.bias.dtype
            or module.bias.requires_grad != first.bias.requires_grad
            for module in modules
        ):
            return None

        out_features = tuple(module.out_features for module in modules)
        packed_k = first.in_features // 2
        scale_k = first.in_features // 16
        weight_offsets = tuple(
            sum(out_features[:index]) * packed_k
            for index in range(len(modules))
        )
        weight_scale_offsets = tuple(
            sum(out_features[:index]) * scale_k
            for index in range(len(modules))
        )
        weight = torch.cat(tuple(module.weight for module in modules), dim=0).contiguous()
        weight_scale = torch.cat(
            tuple(module.weight_scale for module in modules),
            dim=0,
        ).contiguous()
        bias = (
            torch.cat(tuple(module.bias.detach() for module in modules), dim=0).contiguous()
            if has_bias
            else None
        )

        from kairyu.kernels.quant_gemm_gpu import invalidate_nvfp4_linear_cache

        row_offset = 0
        for module, rows in zip(modules, out_features, strict=True):
            module._buffers["weight"] = weight.narrow(0, row_offset, rows)
            module._buffers["weight_scale"] = weight_scale.narrow(
                0,
                row_offset,
                rows,
            )
            if bias is not None:
                assert module.bias is not None
                module.bias = nn.Parameter(
                    bias.narrow(0, row_offset, rows),
                    requires_grad=module.bias.requires_grad,
            )
            row_offset += rows
            invalidate_nvfp4_linear_cache(module)
        bias_offsets = (
            tuple(sum(out_features[:index]) for index in range(len(modules)))
            if bias is not None
            else None
        )
        return cls(
            modules,
            weight,
            weight_scale,
            first.weight_scale_2,
            first.input_scale,
            bias,
            out_features,
            weight_offsets,
            weight_scale_offsets,
            bias_offsets,
        )

    @property
    def in_features(self) -> int:
        return self._projections[0].in_features

    @property
    def out_features(self) -> int:
        return sum(self._out_features)

    @property
    def split_features(self) -> tuple[int, ...]:
        return self._out_features

    @property
    def weight(self) -> torch.Tensor:
        return self._weight

    @property
    def weight_scale(self) -> torch.Tensor:
        return self._weight_scale

    @property
    def weight_scale_2(self) -> torch.Tensor:
        return self._weight_scale_2

    @property
    def input_scale(self) -> torch.Tensor:
        return self._input_scale

    @property
    def bias(self) -> torch.Tensor | None:
        return self._bias

    def buffers(self, recurse: bool = False):
        """Small ``nn.Module``-compatible payload iterator for kernel checks."""

        del recurse
        yield self._weight
        yield self._weight_scale
        yield self._weight_scale_2
        yield self._input_scale

    def _canonical_source_key(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            (
                id(module.weight),
                module.weight._version,
                id(module.weight_scale),
                module.weight_scale._version,
                id(module.input_scale),
                module.input_scale._version,
            )
            + (
                id(module.weight_scale_2),
                module.weight_scale_2._version,
            )
            for module in self._projections
        )

    def matches(self, projections: Sequence[nn.Module]) -> bool:
        return len(projections) == len(self._projections) and all(
            current is packed
            for current, packed in zip(projections, self._projections, strict=True)
        )

    def is_current(self) -> bool:
        from kairyu.quant.linear import NvFp4Linear

        if self._canonical_source_key() != self._source_key:
            return False
        weight_storage = self._weight.untyped_storage().data_ptr()
        scale_storage = self._weight_scale.untyped_storage().data_ptr()
        base_weight_offset = self._weight.storage_offset()
        base_scale_offset = self._weight_scale.storage_offset()
        for module, rows, weight_offset, scale_offset in zip(
            self._projections,
            self._out_features,
            self._weight_offsets,
            self._weight_scale_offsets,
            strict=True,
        ):
            if (
                type(module) is not NvFp4Linear
                or "forward" in module.__dict__
                or module._forward_hooks
                or module._forward_pre_hooks
                or module._backward_hooks
                or module._backward_pre_hooks
                or tuple(module.weight.shape) != (rows, self.in_features // 2)
                or module.weight.stride() != (self.in_features // 2, 1)
                or module.weight.untyped_storage().data_ptr() != weight_storage
                or module.weight.storage_offset() != base_weight_offset + weight_offset
                or tuple(module.weight_scale.shape) != (rows, self.in_features // 16)
                or module.weight_scale.stride() != (self.in_features // 16, 1)
                or module.weight_scale.untyped_storage().data_ptr() != scale_storage
                or module.weight_scale.storage_offset() != base_scale_offset + scale_offset
            ):
                return False
        if self._bias is None:
            return all(module.bias is None for module in self._projections)
        assert self._bias_offsets is not None
        bias_storage = self._bias.untyped_storage().data_ptr()
        base_bias_offset = self._bias.storage_offset()
        return all(
            module.bias is not None
            and tuple(module.bias.shape) == (rows,)
            and module.bias.stride() == (1,)
            and module.bias.untyped_storage().data_ptr() == bias_storage
            and module.bias.storage_offset() == base_bias_offset + offset
            for module, rows, offset in zip(
                self._projections,
                self._out_features,
                self._bias_offsets,
                strict=True,
            )
        )

    def prepare_native_operands(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return all load-time invariant FlashInfer QKV operands."""

        target = torch.device(device)
        if target.type == "cuda" and target.index is None:
            target = self._weight.device
        if self._weight.device != target or self._weight_scale.device != target:
            raise RuntimeError(
                f"packed NVFP4 QKV payload is on {self._weight.device}, not {target}"
            )
        if self._native_weight is None or self._weight_scale_swizzled is None:
            from kairyu.kernels.quant_gemm_gpu import _native_nvfp4_weight

            self._native_weight, self._weight_scale_swizzled = _native_nvfp4_weight(
                self._weight,
                self._weight_scale,
            )
        if self._input_scale_inv is None or self._alpha is None:
            self._input_scale_inv = self._input_scale.float().reciprocal().reshape(1)
            self._alpha = (
                self._input_scale.float() * self._weight_scale_2.float()
            ).reshape(())
        return (
            self._native_weight,
            self._weight_scale_swizzled,
            self._input_scale_inv,
            self._alpha,
        )

    def __call__(self, hidden: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not self.is_current():
            raise RuntimeError("canonical NVFP4 projections no longer match packed storage")
        if hidden.device.type == "cuda":
            from kairyu.kernels.quant_gemm_gpu import nvfp4_packed_linear_forward

            return nvfp4_packed_linear_forward(hidden, self)

        from kairyu.quant import nvfp4

        flat = hidden.reshape(-1, hidden.shape[-1]).to(torch.float32)
        packed, scale = nvfp4.quantize_nvfp4_with_scale(flat, self._input_scale)
        activation = nvfp4.dequantize_nvfp4(packed, scale, self._input_scale)
        weight = nvfp4.dequantize_nvfp4(
            self._weight,
            self._weight_scale,
            self._weight_scale_2,
        )
        output = nn.functional.linear(activation, weight, self._bias)
        output = output.reshape(*hidden.shape[:-1], self.out_features).to(hidden.dtype)
        return output.split(self._out_features, dim=-1)
