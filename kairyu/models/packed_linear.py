"""Inference-only dense projection packing without checkpoint path changes.

Hugging Face stores Q, K, and V as separate tensors. Executing those tensors
as separate ``nn.Linear`` modules needlessly launches three GEMMs.
``DenseLinearPack`` gives the canonical projection parameters views into one
output-row-contiguous allocation and executes that allocation with one
``functional.linear`` call.

The pack is deliberately a plain Python object, not an ``nn.Module``:

* ``state_dict`` / ``named_parameters`` retain the exact HF projection paths;
* contextual quantization and TP metadata remain on the original modules;
* unsupported/custom/quantized projections keep their existing forwards.

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
