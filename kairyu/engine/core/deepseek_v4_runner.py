"""Request-owned Attention-DP runner for DeepSeek V4 native state."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.distributed as dist

from kairyu.engine.core.frontier_runner import NativeFrontierModelRunner
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import ScheduledChunk


class DeepseekV4DistributedRunner(NativeFrontierModelRunner):
    """Synchronize per-request HCA/CSA forwards across EP ranks.

    Attention and cache state are request-owned.  Expert collectives still
    require an identical forward count on every rank, so each prefill/decode
    phase is count-agreed and padded with a real one-token cache forward.  The
    model's packed-FP4 dispatcher independently agrees the largest row tile.
    """

    execution_mode = "deepseek-v4-native-attention-dp"

    def __init__(
        self,
        model,
        sampler,
        *,
        process_group,
        prefix_state_capacity_bytes: int = 0,
    ) -> None:
        super().__init__(
            model,
            sampler=sampler,
            prefix_state_capacity_bytes=prefix_state_capacity_bytes,
        )
        self._process_group = process_group
        self._batched_prefill_enabled = True
        self._unified_mixed_enabled = False
        self._dummy_token_id = 0

    @property
    def unified_mixed_enabled(self) -> bool:
        return self._unified_mixed_enabled

    def set_unified_mixed_enabled(self, enabled: bool) -> None:
        if enabled:
            raise ValueError("DeepSeek V4 Attention-DP keeps prefill/decode phases split")
        self._unified_mixed_enabled = False

    def set_batched_prefill_enabled(self, enabled: bool) -> None:
        if not enabled:
            raise ValueError("DeepSeek V4 Attention-DP cannot disable native prefill")
        self._batched_prefill_enabled = True

    def prefill_execution_stats(self, *, reset: bool = False) -> dict[str, object]:
        del reset
        return {
            "enabled": True,
            "capability_gap": None,
            "implementation": "deepseek-v4-sequential-cache-count-agreed",
        }

    def decode_graph_metadata(self) -> dict[str, object]:
        return {
            "decode_mode": "eager",
            "cuda_graph_decode": False,
            "cuda_graph_buckets": (),
            "cuda_graph_captures": 0,
            "cuda_graph_replays": 0,
            "cuda_graph_eager_fallbacks": 0,
        }

    def _phase_forward_count(self, local_count: int) -> int:
        value = torch.tensor(local_count, dtype=torch.int64, device=self._device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=self._process_group)
        return int(value.item())

    def _padding_forward(self) -> None:
        token = torch.tensor([self._dummy_token_id], dtype=torch.long, device=self._device)
        self._model.forward_cached(token, past_key_values=None, position=0)

    def _execute_phase(
        self,
        chunks: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
        sampled: dict[str, tuple[SampledToken, ...]],
        *,
        prefill: bool,
    ) -> None:
        agreed = self._phase_forward_count(len(chunks))
        if agreed == 0:
            return
        self._model._kairyu_begin_attention_dp_phase()
        if agreed < len(chunks):
            raise RuntimeError("DeepSeek V4 Attention-DP forward-count agreement regressed")
        for chunk in chunks:
            state = states[chunk.request_id]
            if prefill:
                token, forwarded = self._advance_prefill(state)
            else:
                if chunk.num_tokens != 1:
                    raise ValueError("DeepSeek V4 native EP requires single-token decode")
                desired = tuple(state.request.prompt_token_ids) + self._history(
                    state, chunk.position
                )
                handle, forwarded = self._ensure_sequence_with_result(state, desired)
                self._forget_committed(chunk.request_id, len(state.outputs))
                token = self._sample(state, handle.last_logits, chunk.position)
                self._future_tokens.setdefault(chunk.request_id, {})[
                    chunk.position
                ] = token.token_id
            if not forwarded:
                self._padding_forward()
            if token is not None:
                sampled[chunk.request_id] = (token,)
        for _ in range(agreed - len(chunks)):
            self._padding_forward()

    def execute(
        self,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
    ) -> Mapping[str, tuple[SampledToken, ...]]:
        sampled: dict[str, tuple[SampledToken, ...]] = {}
        prefills = tuple(chunk for chunk in scheduled if chunk.is_prefill)
        decodes = tuple(chunk for chunk in scheduled if not chunk.is_prefill)
        self._execute_phase(prefills, states, sampled, prefill=True)
        self._execute_phase(decodes, states, sampled, prefill=False)
        return sampled


__all__ = ["DeepseekV4DistributedRunner"]
