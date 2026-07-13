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
    """Wraps an m15 MoE block: local experts + all_to_all token exchange."""

    def __init__(self, block: nn.Module, comm, ep_rank: int, ep_size: int) -> None:
        super().__init__()
        num_experts = len(block.experts)
        if num_experts % ep_size != 0:
            raise ValueError(f"{num_experts} experts do not divide across {ep_size} ranks")
        self.block = block
        self._comm = comm
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.experts_per_rank = num_experts // ep_size
        # drop non-local experts so per-rank memory actually shrinks; keep
        # module indices stable via an offset at dispatch time
        local = range(
            ep_rank * self.experts_per_rank, (ep_rank + 1) * self.experts_per_rank
        )
        self.local_experts = nn.ModuleList(block.experts[i] for i in local)
        block.experts = nn.ModuleList()  # weights now owned by local_experts

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
        local_ids = expert_ids_in - self.ep_rank * self.experts_per_rank

        computed = torch.zeros_like(received)
        for local_index in local_ids.unique():
            mask = local_ids == local_index
            computed[mask] = self.local_experts[int(local_index)](received[mask])

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
        shared = getattr(self.block, "shared_experts", None)
        if shared is not None:
            out = out + shared(hidden)
        return out

    def _route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        block = self.block
        if hasattr(block, "_route"):  # DeepseekV3MoeBlock
            return block._route(hidden)
        # Qwen3MoeSparseBlock routing (m15 A8)
        logits = block.gate(hidden)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = probs.topk(block.top_k, dim=-1)
        if block.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_indices, topk_weights.to(hidden.dtype)
