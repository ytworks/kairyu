"""torch.distributed-backed Communicator: gloo locally, NCCL by constructor (m16 D1).

Satisfies the m5 object-level ``Communicator`` protocol AND the tensor
extension real parallelism needs. gloo has no reduce_scatter (verified), so
`tensor_reduce_scatter` is all_reduce + a local slice there and the real
collective under NCCL. It is NOT a drop-in for the RowParallelLinear all_reduce:
measured 2026-07-25, rs+ag is ~4% SLOWER than ar at that call site, and the 1.9x
win from rs alone requires sequence parallelism (m16 D1 amendment).
"""

from __future__ import annotations

from datetime import timedelta

import torch
import torch.distributed as dist

_DEFAULT_TIMEOUT_S = 120.0  # gloo's 30-min default turns deadlocks into CI killers


def init_distributed(
    rank: int,
    world_size: int,
    init_method: str,
    backend: str = "gloo",
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> None:
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=timedelta(seconds=timeout_s),
    )


class TorchDistCommunicator:
    """One per process; ``device`` is the rank's compute device under NCCL.

    NCCL rejects host tensors, so the collectives this class builds itself (the
    float tuple all_reduce, the barrier) must be staged on that device. It
    stays ``None`` for gloo, where host tensors are the only correct choice.
    """

    def __init__(
        self,
        group: dist.ProcessGroup | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("call init_distributed() before TorchDistCommunicator")
        self._group = group
        self._device = (
            torch.device(device) if device is not None and str(device) != "cpu" else None
        )
        self._attention_dp_direct_nccl = None
        self._attention_dp_collective_metadata: dict[str, object] | None = None

    @property
    def group(self) -> dist.ProcessGroup | None:
        """The group these collectives run on; None means the default group."""
        return self._group

    @property
    def tensor_device(self) -> torch.device | None:
        """Device for caller-created tensors on this communicator."""

        return self._device

    @property
    def supports_step_tensor_transport(self) -> bool:
        """Whether this group can carry the per-step CUDA control packet."""

        return (
            self._device is not None
            and self._device.type == "cuda"
            and dist.get_backend(self._group) == "nccl"
        )

    @property
    def rank(self) -> int:
        return dist.get_rank(self._group)

    @property
    def world_size(self) -> int:
        return dist.get_world_size(self._group)

    # -- object-level Communicator protocol (m5 comm.py) --------------------

    def broadcast(self, payload: object, src: int) -> object:
        box = [payload]
        dist.broadcast_object_list(box, src=src, group=self._group)
        return box[0]

    def all_reduce(self, values: tuple[float, ...]) -> tuple[float, ...]:
        tensor = torch.tensor(values, dtype=torch.float64, device=self._device)
        dist.all_reduce(tensor, group=self._group)
        return tuple(tensor.tolist())

    def all_gather(self, payload: object) -> tuple[object, ...]:
        box: list[object] = [None] * self.world_size
        dist.all_gather_object(box, payload, group=self._group)
        return tuple(box)

    def barrier(self) -> None:
        if self._device is not None and self._device.type == "cuda":
            # without device_ids NCCL picks a device by guesswork and can pair
            # two ranks onto one GPU
            dist.barrier(group=self._group, device_ids=[self._device.index])
            return
        dist.barrier(group=self._group)

    def send(self, dst: int, payload: object) -> None:
        dist.send_object_list([payload], dst=dst, group=self._group)

    def recv(self, src: int) -> object:
        box: list[object] = [None]
        dist.recv_object_list(box, src=src, group=self._group)
        return box[0]

    # -- tensor extension (m16 D1) -------------------------------------------

    def tensor_broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        dist.broadcast(tensor, src=src, group=self._group)
        return tensor

    def tensor_all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor, group=self._group)
        return tensor

    def tensor_all_reduce_max(self, tensor: torch.Tensor) -> torch.Tensor:
        """Elementwise maximum, used by row-parallel activation scaling."""
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=self._group)
        return tensor

    def tensor_reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sum across ranks, keep only this rank's contiguous shard of dim 0.

        The primitive m16 D1 recorded as missing. gloo genuinely has no
        reduce_scatter, so there it is all_reduce + a local slice — identical
        result, identical traffic. Under NCCL it is the real collective, which
        moves (N-1)/N of the bytes an all_reduce does.

        NOT a drop-in for the `RowParallelLinear` all_reduce: that call site
        needs the FULL sum, and handing it a shard changes what the next layer
        reads. Trading one for the other is sequence parallelism — the shard has
        to survive the norm and be re-gathered by the next column-parallel
        matmul — which m16 does not specify. See `bench/reduce_scatter_bench.py`
        for what the swap is actually worth on a given fabric.
        """
        world = self.world_size
        if tensor.shape[0] % world != 0:
            raise ValueError(
                f"reduce_scatter needs dim 0 ({tensor.shape[0]}) divisible by "
                f"world_size ({world})"
            )
        direct = getattr(self, "_attention_dp_direct_nccl", None)
        if direct is not None and tensor.device.type == "cuda":
            return direct.reduce_scatter(tensor)
        if dist.get_backend(self._group) == "gloo":
            reduced = tensor.clone()
            dist.all_reduce(reduced, group=self._group)
            span = tensor.shape[0] // world
            start = self.rank * span
            return reduced[start : start + span].contiguous()
        output = torch.empty(
            (tensor.shape[0] // world, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dist.reduce_scatter_tensor(output, tensor.contiguous(), group=self._group)
        return output

    def tensor_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Equal-shard gather, concatenated along dim 0 in rank order (gloo
        rejects unequal shapes — callers fail-fast on divisibility)."""
        shards = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(shards, tensor, group=self._group)
        return torch.cat(shards, dim=0)

    def tensor_all_gather_many(
        self,
        tensors: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Gather several equal-shape-per-rank tensors into direct outputs.

        The result for each input is concatenated along dim 0 in rank order,
        matching :meth:`tensor_all_gather`.  NCCL groups the operations into one
        launch boundary; every backend writes directly into the final output so
        callers avoid one shard list plus ``torch.cat`` per tensor.
        """

        if type(tensors) is not tuple or not tensors:
            raise ValueError("tensor_all_gather_many requires a non-empty tuple")
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("tensor_all_gather_many inputs must be tensors")
        if any(tensor.ndim < 1 for tensor in tensors):
            raise ValueError("tensor_all_gather_many inputs must have at least one dim")

        world = self.world_size
        inputs = tuple(tensor.contiguous() for tensor in tensors)
        direct = getattr(self, "_attention_dp_direct_nccl", None)
        if direct is not None and all(tensor.device.type == "cuda" for tensor in inputs):
            return direct.all_gather_many(inputs)
        outputs = tuple(
            torch.empty(
                (world * tensor.shape[0], *tensor.shape[1:]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            for tensor in inputs
        )

        def gather_all() -> None:
            for output, tensor in zip(outputs, inputs, strict=True):
                dist.all_gather_into_tensor(
                    output,
                    tensor,
                    group=self._group,
                )

        if dist.get_backend(self._group) == "nccl":
            with dist._coalescing_manager(group=self._group):
                gather_all()
        else:
            gather_all()
        return outputs

    def tensor_all_gatherv_many(
        self,
        tensors: tuple[torch.Tensor, ...],
        row_sizes: tuple[int, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Rank-ragged grouped gather for the direct attention-DP transport.

        torch.distributed exposes neither the grouped variable-count primitive
        nor a safe equivalent on this model process group. Callers must retain
        their fixed-padding path when direct NCCL was unavailable at startup.
        """

        direct = getattr(self, "_attention_dp_direct_nccl", None)
        if direct is None or not direct.active:
            raise RuntimeError(
                "tensor_all_gatherv_many requires active attention-DP direct "
                "NCCL; torch.distributed fallback is not implemented"
            )
        if any(
            not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda"
            for tensor in tensors
        ):
            raise RuntimeError(
                "tensor_all_gatherv_many requires CUDA tensors on the active "
                "direct NCCL communicator"
            )
        return direct.all_gatherv_many(tensors, row_sizes)

    def tensor_reduce_scatterv(
        self,
        tensor: torch.Tensor,
        row_sizes: tuple[int, ...],
    ) -> torch.Tensor:
        """Rank-ragged SUM reduce-to-owner for direct attention-DP NCCL."""

        direct = getattr(self, "_attention_dp_direct_nccl", None)
        if direct is None or not direct.active:
            raise RuntimeError(
                "tensor_reduce_scatterv requires active attention-DP direct "
                "NCCL; torch.distributed fallback is not implemented"
            )
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
            raise RuntimeError(
                "tensor_reduce_scatterv requires a CUDA tensor on the active "
                "direct NCCL communicator"
            )
        return direct.reduce_scatterv(tensor, row_sizes)

    def synchronize_attention_dp_ragged_layouts(
        self,
        layouts: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...]:
        """Agree one worker step's ragged layouts before model collectives."""

        direct = getattr(self, "_attention_dp_direct_nccl", None)
        if direct is None or not direct.active:
            raise RuntimeError(
                "attention-DP ragged layout synchronization requires active "
                "direct NCCL"
            )
        return direct.synchronize_ragged_layouts(layouts)

    def enable_attention_dp_direct_nccl(
        self,
        control_comm,
        *,
        library_path: str | None = None,
        library_factory=None,
    ) -> bool:
        """Collectively select direct NCCL for the two attention-DP primitives.

        Construction is deliberately explicit and topology-scoped. TP, normal
        EP, single-tensor collectives, and every gloo/CPU operation continue to
        use this communicator's torch.distributed process group.
        """

        from kairyu.engine.core.direct_nccl import DirectNcclCommunicator, NcclLibrary

        backend = str(dist.get_backend(self._group))
        if backend != "nccl" or self._device is None or self._device.type != "cuda":
            self._attention_dp_collective_metadata = {
                "selected_backend": f"torch.distributed:{backend}",
                "fallback_backend": f"torch.distributed:{backend}",
                "direct_nccl_active": False,
                "direct_nccl_library": None,
                "direct_nccl_version": None,
                "selection_reason": (
                    "direct NCCL requires an attention-DP CUDA/NCCL model group"
                ),
            }
            return False
        factory = library_factory if library_factory is not None else NcclLibrary
        direct, metadata = DirectNcclCommunicator.try_create(
            control_comm,
            self._device,
            library_path=library_path,
            library_factory=factory,
        )
        self._attention_dp_direct_nccl = direct
        self._attention_dp_collective_metadata = metadata
        return direct is not None

    @property
    def attention_dp_direct_nccl_active(self) -> bool:
        direct = getattr(self, "_attention_dp_direct_nccl", None)
        return direct is not None and direct.active

    def attention_dp_collective_metadata(self) -> dict[str, object]:
        metadata = getattr(self, "_attention_dp_collective_metadata", None)
        if metadata is not None:
            direct = getattr(self, "_attention_dp_direct_nccl", None)
            if metadata.get("direct_nccl_active") is True and (
                direct is None or not direct.active
            ):
                return {
                    "selected_backend": "unavailable",
                    "fallback_backend": "torch.distributed:nccl",
                    "direct_nccl_active": False,
                    "direct_nccl_library": None,
                    "direct_nccl_version": None,
                    "selection_reason": (
                        "direct NCCL communicator was closed or aborted; "
                        "runtime collective fallback is unsafe"
                    ),
                }
            return dict(metadata)
        backend = str(dist.get_backend(self._group))
        return {
            "selected_backend": f"torch.distributed:{backend}",
            "fallback_backend": f"torch.distributed:{backend}",
            "direct_nccl_active": False,
            "direct_nccl_library": None,
            "direct_nccl_version": None,
            "selection_reason": "direct NCCL was not requested for this topology",
        }

    def close_direct_nccl(self) -> None:
        direct = getattr(self, "_attention_dp_direct_nccl", None)
        self._attention_dp_direct_nccl = None
        if direct is not None:
            direct.close()

    def abort_direct_nccl(self) -> None:
        direct = getattr(self, "_attention_dp_direct_nccl", None)
        self._attention_dp_direct_nccl = None
        if direct is not None:
            direct.abort()

    def tensor_all_to_all_single(
        self,
        output: torch.Tensor,
        tensor: torch.Tensor,
        output_split_sizes: list[int] | None = None,
        input_split_sizes: list[int] | None = None,
    ) -> torch.Tensor:
        dist.all_to_all_single(
            output,
            tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=self._group,
        )
        return output

    def tensor_send(self, tensor: torch.Tensor, dst: int) -> None:
        dist.send(tensor, dst=dst, group=self._group)

    def tensor_recv(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        dist.recv(tensor, src=src, group=self._group)
        return tensor
