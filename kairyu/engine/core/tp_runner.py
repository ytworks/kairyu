"""TP driver facade over per-rank ModelRunners (design m5 D1/D2/D3).

``TPModelRunner`` implements the existing ``ModelRunner`` protocol, so the
Scheduler / RadixKV / step loop above it are unchanged (design D1: KV
accounting is rank-invariant and stays on the driver). Per step, the driver
builds one immutable ``StepInput`` snapshot, broadcasts it through the
``Communicator`` seam, runs every rank on the snapshot, gathers each rank's
sampled ids over send/recv, and enforces rank agreement — TP ranks execute
the same step and must sample identically.

The CPU-testable configuration uses deterministic rank runners and a
``FakeCommunicator`` group; the GPU phase swaps in ``NcclCommunicator`` and
sharded model processes behind the same seams.
"""

from __future__ import annotations

from collections.abc import Mapping

from kairyu.engine.core.comm import Communicator
from kairyu.engine.core.engine_core import ModelRunner
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.engine.core.step_input import snapshot_step

# Both G2 contract models (Llama-3.1-8B, Llama-3.3-70B) have 8 KV heads (GQA).
_CONTRACT_NUM_KV_HEADS = 8
_DRIVER_RANK = 0


def validate_tp_degree(
    tensor_parallel_size: int, num_kv_heads: int = _CONTRACT_NUM_KV_HEADS
) -> None:
    """Reject TP degrees that cannot shard the KV heads evenly (design m5 D3)."""
    if num_kv_heads < 1:
        raise ValueError(f"num_kv_heads must be >= 1, got {num_kv_heads}")
    if tensor_parallel_size < 1:
        raise ValueError(f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}")
    if num_kv_heads % tensor_parallel_size != 0:
        raise ValueError(
            f"tensor_parallel_size={tensor_parallel_size} does not divide "
            f"num_kv_heads={num_kv_heads}; TP degree must shard KV heads evenly "
            "(design m5 D3)"
        )


class TPModelRunner:
    """Driver over N rank runners; one Communicator per rank from the same group."""

    def __init__(
        self,
        rank_runners: tuple[ModelRunner, ...],
        comms: tuple[Communicator, ...],
    ) -> None:
        if not rank_runners:
            raise ValueError("rank_runners must not be empty")
        if len(rank_runners) != len(comms):
            raise ValueError(
                f"rank_runners length {len(rank_runners)} and comms length "
                f"{len(comms)} must match"
            )
        for expected_rank, comm in enumerate(comms):
            if comm.world_size != len(comms):
                raise ValueError(
                    f"comm at index {expected_rank} has world_size {comm.world_size}, "
                    f"expected {len(comms)}"
                )
            if comm.rank != expected_rank:
                raise ValueError(
                    f"comm at index {expected_rank} has rank {comm.rank}; "
                    "comms must be ordered by rank"
                )
        self._rank_runners = tuple(rank_runners)
        self._comms = tuple(comms)

    def release(self, request_id: str) -> None:
        """Forward request-scoped cleanup to every local TP rank."""
        for runner in self._rank_runners:
            release = getattr(runner, "release", None)
            if release is not None:
                release(request_id)

    def execute(
        self, scheduled: tuple[ScheduledChunk, ...], states: Mapping[str, object]
    ) -> dict[str, tuple]:
        """Snapshot once, broadcast, run every rank, gather, and check agreement.

        Agreement compares token ids only (m8 D2 review): the m5 D1 invariant
        is about tokens; logprob float equality would be brittle on GPU. Rank
        0's full StepOutput (with logprobs) is what the driver returns.
        """
        step_input = snapshot_step(scheduled, states)
        driver_comm = self._comms[_DRIVER_RANK]
        sent = driver_comm.broadcast(step_input, src=_DRIVER_RANK)
        for rank, (runner, comm) in enumerate(
            zip(self._rank_runners, self._comms, strict=True)
        ):
            received = sent if rank == _DRIVER_RANK else comm.broadcast(None, src=_DRIVER_RANK)
            sampled = runner.execute(received.chunks, received.states_view())
            comm.send(_DRIVER_RANK, dict(sampled))
        reference: dict[str, tuple] = driver_comm.recv(_DRIVER_RANK)
        reference_ids = _agreement_view(reference)
        for rank in range(1, len(self._comms)):
            candidate = driver_comm.recv(rank)
            if _agreement_view(candidate) != reference_ids:
                raise RuntimeError(
                    f"TP rank {rank} sampled {candidate!r} but rank 0 sampled "
                    f"{reference!r}; TP ranks must agree (design m5 D1)"
                )
        return reference


def _agreement_view(step_output: dict[str, tuple]) -> dict[str, tuple[int, ...]]:
    return {
        request_id: tuple(token.token_id for token in tokens)
        for request_id, tokens in step_output.items()
    }
