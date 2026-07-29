"""Expert-parallel dispatch/combine over all_to_all (m16 D3).

Routing runs REPLICATED (fp32, deterministic on CPU/gloo; a deploy-day debug
guard hashes topk_indices across ranks — m16 A8). Tokens permute to
expert-owning ranks via ``tensor_all_to_all_single`` (counts exchange first,
then payload), local experts compute, reverse all_to_all, weighted combine
locally. Contiguous expert blocks per rank; the math is the m15 token-loop's,
algebraically identical (accumulation order differs — parity gates use token
equality, m16 A7). gloo and NCCL share this code path; DeepEP/UCCL is the
deploy-day fast path behind the same block interface.
"""

from __future__ import annotations

import torch
from torch import nn

from kairyu.models.moe import route_experts
from kairyu.models.parallel import ParallelShardInfo


class RemoteExpertOwnershipError(LookupError):
    """A canonical expert exists, but its tensors belong to another rank."""


def _assert_collective_device(
    expected: torch.device,
    **tensors: torch.Tensor,
) -> None:
    if any(tensor.device != expected for tensor in tensors.values()):
        details = ", ".join(
            f"{name}={tensor.device}" for name, tensor in tensors.items()
        )
        raise ValueError(
            f"expert-parallel collective device mismatch: expected {expected}; {details}"
        )


class EpMoeBlock(nn.Module):
    """Canonical MoE tree with rank-local experts + all-to-all execution.

    Router, shared expert, and owned routed experts are registered directly at
    their HF paths. ``experts`` keeps the global length and uses ``None`` for
    remote owners, which PyTorch naturally omits from state/named enumeration
    while retaining global expert indices.
    """

    def __init__(self, block: nn.Module, comm, ep_rank: int, ep_size: int) -> None:
        super().__init__()
        num_experts = len(block.experts)
        if ep_size < 1:
            raise ValueError("expert-parallel size must be positive")
        if not 0 <= ep_rank < ep_size:
            raise ValueError(
                f"expert-parallel rank {ep_rank} is outside size {ep_size}"
            )
        if num_experts % ep_size != 0:
            raise ValueError(f"{num_experts} experts do not divide across {ep_size} ranks")
        self._comm = comm
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.experts_per_rank = num_experts // ep_size
        start = ep_rank * self.experts_per_rank
        end = (ep_rank + 1) * self.experts_per_rank
        self.owned_expert_indices = tuple(range(start, end))

        # Register the canonical tree directly. Replacing the original block's
        # list too releases non-local tensors even if a caller retains a
        # reference to that pre-parallel block.
        if hasattr(block, "gate"):
            self.gate = block.gate
            route_override = None
        elif hasattr(block, "_route"):
            # Small/custom blocks may own routing logic without parameters.
            # The original expert list is replaced below before this bound
            # method is retained, so it cannot keep remote expert tensors live.
            self.gate = None
            route_override = block._route
        else:
            raise TypeError(
                f"{type(block).__name__} must expose gate or _route for EP"
            )
        experts = nn.ModuleList(
            expert if index in self.owned_expert_indices else None
            for index, expert in enumerate(block.experts)
        )
        self.experts = experts
        block.experts = experts
        self.shared_experts = getattr(block, "shared_experts", None)
        object.__setattr__(self, "_route_override", route_override)

        # Routing metadata is scalar/plain state, never checkpoint state.
        if route_override is None:
            self.top_k = block.top_k
            self.norm_topk_prob = block.norm_topk_prob
            if hasattr(block, "n_group"):
                self.n_group = block.n_group
                self.topk_group = block.topk_group
                self.routed_scaling_factor = block.routed_scaling_factor
        object.__setattr__(
            self,
            "_kairyu_parallel_shard",
            ParallelShardInfo("expert", ep_rank, ep_size),
        )

    @property
    def local_experts(self) -> tuple[nn.Module, ...]:
        """Compatibility view; modules stay registered by global index."""

        return tuple(
            self.local_expert(index)
            for index in self.owned_expert_indices
        )

    def owner_rank(self, expert_index: int) -> int:
        if not 0 <= expert_index < len(self.experts):
            raise IndexError(
                f"expert index {expert_index} is outside [0, {len(self.experts)})"
            )
        return expert_index // self.experts_per_rank

    def local_expert(self, expert_index: int) -> nn.Module:
        owner = self.owner_rank(expert_index)
        expert = self.experts[expert_index]
        if expert is None:
            raise RemoteExpertOwnershipError(
                f"expert {expert_index} is owned by rank {owner}, "
                f"not rank {self.ep_rank}"
            )
        return expert

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        device = hidden.device
        topk_indices, topk_weights = self._route(hidden)
        tokens, k = topk_indices.shape
        flat_expert = topk_indices.reshape(-1)  # [tokens*k]
        owner = flat_expert // self.experts_per_rank
        order = torch.argsort(owner, stable=True)
        send_counts = torch.bincount(owner, minlength=self.ep_size)
        payload = hidden.repeat_interleave(k, dim=0)[order]

        recv_counts = torch.empty(
            self.ep_size, dtype=send_counts.dtype, device=device
        )
        _assert_collective_device(
            device, send_counts=send_counts, recv_counts=recv_counts
        )
        self._comm.tensor_all_to_all_single(
            recv_counts, send_counts.contiguous(), [1] * self.ep_size, [1] * self.ep_size
        )
        recv_total = int(recv_counts.sum().item())
        received = torch.empty(
            recv_total, hidden.shape[-1], dtype=hidden.dtype, device=device
        )
        _assert_collective_device(device, payload=payload, received=received)
        self._comm.tensor_all_to_all_single(
            received,
            payload.contiguous(),
            recv_counts.tolist(),
            send_counts.tolist(),
        )
        # which local expert each received row wants: exchange expert ids too
        expert_ids_out = flat_expert[order].to(torch.int64)
        expert_ids_in = torch.empty(recv_total, dtype=torch.int64, device=device)
        _assert_collective_device(
            device, expert_ids_out=expert_ids_out, expert_ids_in=expert_ids_in
        )
        self._comm.tensor_all_to_all_single(
            expert_ids_in,
            expert_ids_out.contiguous(),
            recv_counts.tolist(),
            send_counts.tolist(),
        )
        computed = torch.zeros_like(received)
        for global_index in expert_ids_in.unique():
            expert_index = int(global_index)
            mask = expert_ids_in == global_index
            computed[mask] = self.local_expert(expert_index)(received[mask])

        returned = torch.empty(
            tokens * k, hidden.shape[-1], dtype=hidden.dtype, device=device
        )
        _assert_collective_device(device, computed=computed, returned=returned)
        self._comm.tensor_all_to_all_single(
            returned,
            computed.contiguous(),
            send_counts.tolist(),
            recv_counts.tolist(),
        )
        # undo the permutation, weight, and combine per token
        unsorted = torch.empty_like(returned)
        unsorted[order] = returned
        weighted = unsorted.reshape(tokens, k, -1) * topk_weights[:, :, None]
        out = weighted.sum(dim=1)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden)
        return out

    def _route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._route_override is not None:
            return self._route_override(hidden)
        return route_experts(self, hidden)
