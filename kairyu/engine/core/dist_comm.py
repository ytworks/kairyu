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

    @property
    def group(self) -> dist.ProcessGroup | None:
        """The group these collectives run on; None means the default group."""
        return self._group

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
