"""torch.distributed-backed Communicator: gloo locally, NCCL by constructor (m16 D1).

Satisfies the m5 object-level ``Communicator`` protocol AND the tensor
extension real parallelism needs. gloo has no reduce_scatter (verified) — all
call sites use all_reduce; NCCL's reduce_scatter is a same-call-site
optimization recorded for deploy day.
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
    """One per process; ``backend='nccl'`` + device is the deploy-day change."""

    def __init__(self, group: dist.ProcessGroup | None = None) -> None:
        if not dist.is_initialized():
            raise RuntimeError("call init_distributed() before TorchDistCommunicator")
        self._group = group

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
        tensor = torch.tensor(values, dtype=torch.float64)
        dist.all_reduce(tensor, group=self._group)
        return tuple(tensor.tolist())

    def all_gather(self, payload: object) -> tuple[object, ...]:
        box: list[object] = [None] * self.world_size
        dist.all_gather_object(box, payload, group=self._group)
        return tuple(box)

    def barrier(self) -> None:
        dist.barrier(group=self._group)

    def send(self, dst: int, payload: object) -> None:
        dist.send_object_list([payload], dst=dst, group=self._group)

    def recv(self, src: int) -> object:
        box: list[object] = [None]
        dist.recv_object_list(box, src=src, group=self._group)
        return box[0]

    # -- tensor extension (m16 D1) -------------------------------------------

    def tensor_all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor, group=self._group)
        return tensor

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
