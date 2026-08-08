"""Packed BF16 experts executed with FlashInfer's cuDNN grouped GEMM."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

import torch
from torch import nn

_GROUPED_MOE_MAX_ROWS = 8192
_SUPPORTED_COMPUTE_CAPABILITIES = {80, 86, 87, 89, 90, 100, 103, 110, 120, 121}
_WARMED_GROUPED_GEOMETRIES: set[tuple[str, int, int, int, int]] = set()


@cache
def _grouped_moe_capable(device_type: str, device_index: int) -> bool:
    """Probe FlashInfer/cuDNN support before selecting the optimized path."""

    if device_type != "cuda":
        return False
    try:
        import cudnn
        import flashinfer

        properties = torch.cuda.get_device_properties(device_index)
        capability = properties.major * 10 + properties.minor
        return (
            hasattr(flashinfer, "grouped_mm_bf16")
            and cudnn.backend_version() >= 91800
            and hasattr(cudnn, "moe_grouped_matmul_mode")
            and capability in _SUPPORTED_COMPUTE_CAPABILITIES
        )
    except (ImportError, OSError, RuntimeError):
        return False


def _grouped_rows_bucket(rows: int) -> int | None:
    """Bound grouped plans to power-of-two eager/decode row buckets."""

    if rows < 1 or rows > _GROUPED_MOE_MAX_ROWS:
        return None
    return 1 << (rows - 1).bit_length()


class GroupedExpertPack:
    """Contiguous gate/up/down weights behind canonical per-expert parameters.

    The packed tensors are plain Python-owned execution storage.  Each expert's
    canonical ``nn.Linear.weight`` remains registered at its Hugging Face path
    as a view, so checkpoint names do not change and no duplicate model-wide
    weight allocation remains after packing.
    """

    __slots__ = ("_down", "_experts", "_gate_up")

    def __init__(
        self,
        experts: tuple[nn.Module, ...],
        gate_up: torch.Tensor,
        down: torch.Tensor,
    ) -> None:
        self._experts = experts
        self._gate_up = gate_up
        self._down = down

    @classmethod
    def create(cls, experts: Sequence[nn.Module]) -> GroupedExpertPack | None:
        """Pack homogeneous bias-free dense SwiGLU experts when compatible."""

        if not experts:
            return None
        packed_experts = tuple(experts)
        projections: list[tuple[nn.Linear, nn.Linear, nn.Linear]] = []
        for expert in packed_experts:
            if (
                type(expert).__dict__.get("_kairyu_grouped_swiglu") is not True
                or "forward" in expert.__dict__
                or expert._forward_hooks
                or expert._forward_pre_hooks
                or expert._backward_hooks
            ):
                return None
            candidate = (
                getattr(expert, "gate_proj", None),
                getattr(expert, "up_proj", None),
                getattr(expert, "down_proj", None),
            )
            if any(type(module) is not nn.Linear for module in candidate):
                return None
            gate, up, down = candidate
            if any(module.bias is not None for module in candidate):
                return None
            projections.append((gate, up, down))

        first_gate, first_up, first_down = projections[0]
        if first_gate.weight.device.type == "meta":
            return None
        hidden = first_gate.in_features
        intermediate = first_gate.out_features
        if (
            first_up.in_features != hidden
            or first_up.out_features != intermediate
            or first_down.in_features != intermediate
            or first_down.out_features != hidden
        ):
            return None
        device = first_gate.weight.device
        dtype = first_gate.weight.dtype
        if dtype is not torch.bfloat16:
            return None
        if device.type == "cuda" and not _grouped_moe_capable(
            device.type,
            device.index if device.index is not None else torch.cuda.current_device(),
        ):
            return None
        requires_grad = first_gate.weight.requires_grad
        if any(
            module.weight.device != device
            or module.weight.dtype != dtype
            or module.weight.layout is not torch.strided
            or type(module.weight) is not nn.Parameter
            or module.weight.requires_grad != requires_grad
            or "forward" in module.__dict__
            or module._forward_hooks
            or module._forward_pre_hooks
            or module._backward_hooks
            for triple in projections
            for module in triple
        ):
            return None
        if any(
            gate.in_features != hidden
            or gate.out_features != intermediate
            or up.in_features != hidden
            or up.out_features != intermediate
            or down.in_features != intermediate
            or down.out_features != hidden
            for gate, up, down in projections
        ):
            return None

        with torch.no_grad():
            gate_up = torch.empty(
                len(projections),
                intermediate * 2,
                hidden,
                dtype=dtype,
                device=device,
            )
            down = torch.empty(
                len(projections),
                hidden,
                intermediate,
                dtype=dtype,
                device=device,
            )
            for index, (gate, up, down_proj) in enumerate(projections):
                gate_up[index, :intermediate].copy_(gate.weight)
                gate_up[index, intermediate:].copy_(up.weight)
                down[index].copy_(down_proj.weight)

        for index, (gate, up, down_proj) in enumerate(projections):
            gate.weight = nn.Parameter(
                gate_up[index, :intermediate],
                requires_grad=requires_grad,
            )
            up.weight = nn.Parameter(
                gate_up[index, intermediate:],
                requires_grad=requires_grad,
            )
            down_proj.weight = nn.Parameter(
                down[index],
                requires_grad=requires_grad,
            )
        pack = cls(packed_experts, gate_up, down)
        if device.type == "cuda":
            try:
                pack._warm_grouped_plans()
            except (ImportError, OSError, RuntimeError):
                return None
        return pack

    def is_current(self) -> bool:
        """Whether canonical parameters still point into the packed storage."""

        gate_up_storage = self._gate_up.untyped_storage().data_ptr()
        down_storage = self._down.untyped_storage().data_ptr()
        experts = len(self._experts)
        intermediate = self._gate_up.shape[1] // 2
        hidden = self._gate_up.shape[2]
        gate_up_stride = intermediate * 2 * hidden
        down_stride = hidden * intermediate
        for index, expert in enumerate(self._experts):
            gate = getattr(expert, "gate_proj", None)
            up = getattr(expert, "up_proj", None)
            down = getattr(expert, "down_proj", None)
            if any(type(module) is not nn.Linear for module in (gate, up, down)):
                return False
            if any(
                "forward" in module.__dict__
                or module._forward_hooks
                or module._forward_pre_hooks
                or module._backward_hooks
                for module in (gate, up, down)
            ):
                return False
            if (
                gate.weight.shape != (intermediate, hidden)
                or up.weight.shape != (intermediate, hidden)
                or down.weight.shape != (hidden, intermediate)
                or gate.weight.stride() != (hidden, 1)
                or up.weight.stride() != (hidden, 1)
                or down.weight.stride() != (intermediate, 1)
                or gate.weight.untyped_storage().data_ptr() != gate_up_storage
                or up.weight.untyped_storage().data_ptr() != gate_up_storage
                or down.weight.untyped_storage().data_ptr() != down_storage
                or gate.weight.storage_offset() != index * gate_up_stride
                or up.weight.storage_offset()
                != index * gate_up_stride + intermediate * hidden
                or down.weight.storage_offset() != index * down_stride
            ):
                return False
        return experts == self._gate_up.shape[0] == self._down.shape[0]

    def supports(self, hidden: torch.Tensor, rows: int | None = None) -> bool:
        """Whether the CUDA BF16 grouped operator can execute this input."""

        return (
            hidden.device.type == "cuda"
            and hidden.dtype is torch.bfloat16
            and self._gate_up.dtype is torch.bfloat16
            and self._down.dtype is torch.bfloat16
            and self._gate_up.device == hidden.device
            and self._down.device == hidden.device
            and (rows is None or _grouped_rows_bucket(rows) is not None)
        )

    def _execute_sorted(
        self,
        hidden: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        from flashinfer import grouped_mm_bf16

        indptr = torch.cat((offsets.new_zeros(1), offsets))
        gate_up = grouped_mm_bf16(hidden.contiguous(), self._gate_up, indptr)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = nn.functional.silu(gate) * up
        return grouped_mm_bf16(activated.contiguous(), self._down, indptr)

    def _warm_grouped_plans(self) -> None:
        """Compile every bounded row bucket and reserve the workspace high-water."""

        device = self._gate_up.device
        key = (
            str(device),
            self._gate_up.shape[0],
            self._gate_up.shape[1],
            self._gate_up.shape[2],
            self._down.shape[1],
        )
        if key in _WARMED_GROUPED_GEOMETRIES:
            return
        experts = len(self._experts)
        rows = 1
        while rows <= _GROUPED_MOE_MAX_ROWS:
            hidden = torch.zeros(
                rows,
                self._gate_up.shape[2],
                dtype=torch.bfloat16,
                device=device,
            )
            offsets = torch.zeros(experts, dtype=torch.int32, device=device)
            offsets[-1] = rows
            self._execute_sorted(hidden, offsets)
            rows *= 2
        _WARMED_GROUPED_GEOMETRIES.add(key)

    def forward_sorted(
        self,
        hidden: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Run rows already sorted by local expert with device-side offsets."""

        if not self.supports(hidden, hidden.shape[0]):
            raise RuntimeError("grouped expert pack is not current CUDA BF16 storage")
        if offsets.shape != (len(self._experts),) or offsets.dtype is not torch.int32:
            raise ValueError("grouped expert offsets must be one int32 value per expert")
        if hidden.shape[0] == 0:
            return hidden
        return self._execute_sorted(hidden, offsets)

    def forward(
        self,
        hidden: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Sort rows by expert, execute two grouped GEMMs, and restore order."""

        if expert_indices.ndim != 1 or expert_indices.shape[0] != hidden.shape[0]:
            raise ValueError("one expert index is required per grouped-MoE row")
        bucket = _grouped_rows_bucket(hidden.shape[0])
        if bucket is None:
            raise ValueError(
                f"grouped-MoE rows must be in [1, {_GROUPED_MOE_MAX_ROWS}]"
            )
        order = torch.argsort(expert_indices, stable=True)
        sorted_indices = expert_indices[order]
        counts = torch.zeros(
            len(self._experts),
            dtype=torch.int32,
            device=expert_indices.device,
        )
        counts.scatter_add_(
            0,
            sorted_indices,
            torch.ones_like(sorted_indices, dtype=torch.int32),
        )
        padding = bucket - hidden.shape[0]
        sorted_hidden = hidden[order]
        if padding:
            sorted_hidden = nn.functional.pad(sorted_hidden, (0, 0, 0, padding))
            counts[-1] += padding
        offsets = counts.cumsum(0, dtype=torch.int32)
        sorted_output = self.forward_sorted(sorted_hidden, offsets)[: hidden.shape[0]]
        output = torch.empty_like(sorted_output)
        output[order] = sorted_output
        return output
