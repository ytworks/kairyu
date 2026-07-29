"""In-process TP facade over per-rank ModelRunners (design m5 D1/D2/D3).

``TPModelRunner`` implements the existing ``ModelRunner`` protocol, so the
Scheduler / RadixKV / step loop above it are unchanged (design D1: KV
accounting is rank-invariant and stays on the driver). Per step, the driver
builds one immutable ``StepInput`` snapshot, broadcasts it through the
``Communicator`` seam, lets rank 0 execute and sample, and runs every follower
through its passive model/KV path. Rank 0 then broadcasts one fixed-layout
device-token packet; every rank adopts those authoritative ids before the
next step. No follower produces a public result or advances sampling state.

This legacy CPU-testable facade uses deterministic rank runners and a
``FakeCommunicator`` group. Real sharded model processes use
``DistTPModelRunner`` but share this sampling-ownership protocol.
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
_RANK_RUNNER_METHODS = (
    "execute",
    "execute_passive",
    "make_sampling_token_packet",
    "adopt_sampling_token_packet",
)


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
        identities: dict[int, int] = {}
        for rank, runner in enumerate(rank_runners):
            previous_rank = identities.setdefault(id(runner), rank)
            if previous_rank != rank:
                raise ValueError(
                    "rank_runners must contain one distinct runner instance per "
                    f"rank; ranks {previous_rank} and {rank} share an instance"
                )
            missing = [
                method
                for method in _RANK_RUNNER_METHODS
                if not callable(getattr(runner, method, None))
            ]
            if missing:
                raise TypeError(
                    f"rank runner {rank} does not support rank-0-authoritative TP "
                    f"sampling; missing callable methods: {', '.join(missing)}. "
                    "Use tensor_parallel_size=1 or implement the passive "
                    "execution and sampling-token packet protocol."
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
        """Sample on rank 0, broadcast its fixed token packet, and adopt it."""
        step_input = snapshot_step(scheduled, states)
        driver_comm = self._comms[_DRIVER_RANK]
        sent = driver_comm.broadcast(step_input, src=_DRIVER_RANK)
        received = [
            sent
            if rank == _DRIVER_RANK
            else comm.broadcast(None, src=_DRIVER_RANK)
            for rank, comm in enumerate(self._comms)
        ]

        owner = self._rank_runners[_DRIVER_RANK]
        owner_input = received[_DRIVER_RANK]
        owner_states = owner_input.states_view()
        sampled = owner.execute(owner_input.chunks, owner_states)
        packet = owner.make_sampling_token_packet(
            owner_input.chunks, owner_states, sampled=sampled
        )
        packet = driver_comm.tensor_broadcast(packet, src=_DRIVER_RANK)
        owner.adopt_sampling_token_packet(owner_input.chunks, owner_states, packet)

        for rank in range(1, len(self._rank_runners)):
            runner = self._rank_runners[rank]
            rank_input = received[rank]
            rank_states = rank_input.states_view()
            runner.execute_passive(rank_input.chunks, rank_states)
            receive_packet = runner.make_sampling_token_packet(
                rank_input.chunks, rank_states, sampled=None
            )
            receive_packet = self._comms[rank].tensor_broadcast(
                receive_packet, src=_DRIVER_RANK
            )
            runner.adopt_sampling_token_packet(
                rank_input.chunks, rank_states, receive_packet
            )
        return sampled
