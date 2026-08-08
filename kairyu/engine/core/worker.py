"""SPMD TP execution: driver-side runner + worker main (m16 D4).

Rank 0 owns the scheduler/EngineCore, broadcasts each immutable state delta,
and is the sole sampling authority. Every rank executes the same model/KV step,
then rank 0 broadcasts one fixed-layout device token packet over the model
communicator. All ranks adopt that packet before advancing. Finished-request
cleanup is piggybacked on the next step (or shutdown) so the steady path does
not spend a separate control collective per request.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from threading import Lock

from kairyu.engine.core.sampling_types import EngineSampling, SampledToken
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.engine.core.step_executor import (
    DecodeGraphCapturePlan,
    GraphDecodeDecision,
)
from kairyu.engine.core.step_input import RequestSnapshot, StateSync, StepDelta
from kairyu.engine.tokenizer import GrammarVocabulary

_SHUTDOWN = None

#: Ranks load their shard between the rendezvous and the handshake, so the
#: CI-tuned 120s default would fire on a cold multi-GB read long before anything
#: is actually deadlocked. This covers ONLY startup.
_STARTUP_TIMEOUT_S = 1800.0
#: Model collectives once the group is serving. `init_process_group(timeout=)`
#: bounds every operation on that group, so a wedged rank must not hold an
#: in-flight generation for the startup allowance.
_SERVE_OP_TIMEOUT_S = 120.0
#: Non-zero ranks intentionally sit inside the next control broadcast while the
#: server is idle. A short collective timeout therefore kills a healthy TP group
#: after exactly that much idle time. Keep the control receive effectively
#: process-lifetime while model work retains the fail-fast bound above.
_CONTROL_IDLE_TIMEOUT_S = 365 * 24 * 60 * 60.0
# Diagnostic rows are tiny, but a fixed packet keeps the collective shape
# identical on every rank and lets it run on the bounded model communicator.
_PREFILL_STATS_PACKET_BYTES = 4096
_VERIFICATION_STATS_PACKET_BYTES = 4096
_PAGE_TABLE_STATS_PACKET_BYTES = 4096
_EP_KV_ACCOUNTING_MAX_OBSERVATIONS = 4096
_EP_KV_ACCOUNTING_MAX_REQUEST_ID_BYTES = 256


@dataclass(frozen=True)
class _ShutdownRequest:
    """Terminal control payload carrying cleanup not consumed by another step."""

    released_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _EPAttentionDPGraphPlan:
    """One globally coordinated decode shape and immutable dispatch branch."""

    cuda_graph_decode: bool
    batch_size: int
    max_pages: int
    decision: GraphDecodeDecision


@dataclass(frozen=True)
class _EPAttentionDPStep:
    """One request-owned attention-DP transaction.

    ``delta`` remains the canonical scheduler snapshot. ``owners`` is limited
    to requests present in this step and is rank ordered by first chunk
    appearance, so every process derives the same local work and collective
    layout without another control exchange.
    """

    delta: StepDelta
    owners: tuple[tuple[str, int], ...]
    graph_plan: _EPAttentionDPGraphPlan | None = None


@dataclass(frozen=True)
class _EPAttentionDPReply:
    """Pickle-safe owner-local sampling result or bounded rank failure."""

    rank: int
    sampled: dict[str, tuple[SampledToken, ...]] | None = None
    error: str | None = None


@dataclass(frozen=True)
class _EPAttentionDPLocalResult:
    """Rank-local status plus the device packet kept off the gloo path."""

    reply: _EPAttentionDPReply
    gathered_token_packet: object | None = None


@dataclass(frozen=True)
class _EPAttentionDPGraphValidationReply:
    """Bounded pre-model validation result gathered on the control group."""

    rank: int
    error: str | None = None


@dataclass(frozen=True)
class _SamplingOwnershipProbe:
    """Out-of-band request for rank-local TP sampling metadata."""


@dataclass(frozen=True)
class _SamplingOwnershipReply:
    """Pickle-safe sampling probe reply that keeps every rank in the gather."""

    rank: int
    row: dict[str, object] | None = None
    error: str | None = None


@dataclass(frozen=True)
class _CaptureDecodeGraphs:
    """Startup-only command that warms every rank's configured graph buckets."""

    attention_dp: bool = False


@dataclass(frozen=True)
class _DecodeGraphCapturePlanReply:
    """Rank-local graph identity gathered before model collectives begin."""

    rank: int
    plan: DecodeGraphCapturePlan | None = None
    error: str | None = None


@dataclass(frozen=True)
class _DecodeGraphCapturePreparationReply:
    """Rank-local ACK after any attention-DP layout queue is armed."""

    rank: int
    error: str | None = None


@dataclass(frozen=True)
class _CaptureDecodeGraphsReply:
    """Rank-local graph warmup result kept on the host control group."""

    rank: int
    buckets: tuple[int, ...] | None = None
    error: str | None = None


@dataclass(frozen=True)
class _EPKernelInventoryProbe:
    """Out-of-band request for rank-local EP NVFP4 kernel metadata."""


@dataclass(frozen=True)
class _EPKernelInventoryReply:
    """Pickle-safe EP probe reply that keeps every rank in the gather."""

    rank: int
    row: dict[str, object] | None = None
    error: str | None = None


@dataclass(frozen=True)
class _EPKVAccountingControl:
    """Start or finish one opt-in all-rank EP KV-accounting capture."""

    action: str


@dataclass(frozen=True)
class _EPKVAccountingReply:
    """Pickle-safe EP KV-accounting reply with a bounded error envelope."""

    rank: int
    row: dict[str, object] | None = None
    error: str | None = None


@dataclass(frozen=True)
class _BatchedPrefillMode:
    """Out-of-band, all-rank optimization toggle for rollback/matched A-B."""

    enabled: bool


@dataclass(frozen=True)
class _PrefillStatsProbe:
    """Out-of-band request for rank-local structural prefill counters."""

    reset: bool = False


@dataclass(frozen=True)
class _BatchedVerificationMode:
    """Out-of-band, all-rank speculative-verification toggle."""

    enabled: bool


@dataclass(frozen=True)
class _VerificationStatsProbe:
    """Out-of-band request for rank-local speculative-verification counters."""

    reset: bool = False


@dataclass(frozen=True)
class _DecodePageTableCacheMode:
    """Out-of-band, all-rank decode page-table cache toggle."""

    enabled: bool


@dataclass(frozen=True)
class _DecodePageTableCacheStatsProbe:
    """Out-of-band request for rank-local page-table cache counters."""

    reset: bool = False


@dataclass(frozen=True)
class _DramKVAvailablePrefix:
    """Query the rank-local DRAM tier without crossing a model step."""

    keys: tuple[str, ...]
    min_pages: int


@dataclass(frozen=True)
class _DramKVOffload:
    """Copy the same logical KV pages to every rank-local DRAM tier."""

    keys: tuple[str, ...]
    page_ids: tuple[int, ...]


@dataclass(frozen=True)
class _DramKVRestore:
    """Restore the same logical KV pages on every rank."""

    keys: tuple[str, ...]
    page_ids: tuple[int, ...]


@dataclass(frozen=True)
class _DramKVDiscard:
    """Atomically discard rank-local host copies for the supplied keys."""

    keys: tuple[str, ...]


_DramKVControl = (
    _DramKVAvailablePrefix | _DramKVOffload | _DramKVRestore | _DramKVDiscard
)


@dataclass(frozen=True)
class _DramKVAck:
    """Pickle-safe completion record; every rank contributes exactly one."""

    rank: int
    operation: str
    ok: bool
    value: object = None
    error: str | None = None
    base_exception: bool = False
    source_reusable: bool = False
    destination_reusable: bool = False
    destination_publishable: bool = False


class DramKVTierTransactionError(RuntimeError):
    """An all-rank KV tier transaction could not establish safe ownership."""

    def __init__(
        self,
        message: str,
        *,
        source_reusable: bool = False,
        destination_reusable: bool = False,
        destination_publishable: bool = False,
    ) -> None:
        super().__init__(message)
        self.source_reusable = source_reusable
        self.destination_reusable = destination_reusable
        self.destination_publishable = destination_publishable


_SAMPLING_OWNERSHIP_FIELDS = frozenset(
    {
        "rank",
        "control_world_size",
        "control_backend",
        "model_world_size",
        "model_backend",
        "sampling_owner",
        "sampler_present",
        "device",
    }
)


def _communicator_backend(comm) -> str:
    """Return the real process-group backend without labeling it externally."""
    from kairyu.engine.core.comm import FakeCommunicator

    if isinstance(comm, FakeCommunicator):
        return "fake"

    import torch.distributed as dist

    try:
        backend = str(dist.get_backend(comm.group))
    except Exception as error:
        raise RuntimeError(
            "cannot inspect communicator backend for "
            f"{type(comm).__name__}; expected FakeCommunicator or a "
            "torch.distributed communicator with a process group"
        ) from error
    if not backend:
        raise RuntimeError(f"communicator backend for {type(comm).__name__} is empty")
    return backend


def _sampling_ownership_row(control_comm, model_comm, local_runner) -> dict[str, object]:
    """Build one rank's JSON-serializable ownership/topology observation."""
    control_rank = control_comm.rank
    model_rank = model_comm.rank
    if type(control_rank) is not int or type(model_rank) is not int:
        raise RuntimeError(
            "sampling ownership rank metadata must be integers: "
            f"control={control_rank!r}, model={model_rank!r}"
        )
    if control_rank != model_rank:
        raise RuntimeError(
            "sampling ownership communicator rank mismatch: "
            f"control rank={control_rank}, model rank={model_rank}"
        )

    sampling_owner = getattr(local_runner, "sampling_owner", None)
    if type(sampling_owner) is not bool:
        raise RuntimeError(
            f"rank {control_rank} runner has malformed sampling_owner="
            f"{sampling_owner!r}; expected bool"
        )
    if not hasattr(local_runner, "_sampler"):
        raise RuntimeError(f"rank {control_rank} runner does not expose sampler ownership")
    if not hasattr(local_runner, "_device"):
        raise RuntimeError(f"rank {control_rank} runner does not expose its compute device")

    control_world_size = control_comm.world_size
    model_world_size = model_comm.world_size
    if (
        type(control_world_size) is not int
        or control_world_size < 1
        or type(model_world_size) is not int
        or model_world_size < 1
    ):
        raise RuntimeError(
            f"rank {control_rank} has malformed communicator world sizes: "
            f"control={control_world_size!r}, model={model_world_size!r}"
        )
    device = str(local_runner._device)
    if not device:
        raise RuntimeError(f"rank {control_rank} runner device is empty")

    return {
        "rank": control_rank,
        "control_world_size": control_world_size,
        "control_backend": _communicator_backend(control_comm),
        "model_world_size": model_world_size,
        "model_backend": _communicator_backend(model_comm),
        "sampling_owner": sampling_owner,
        "sampler_present": local_runner._sampler is not None,
        "device": device,
    }


def _sampling_ownership_reply(
    control_comm,
    model_comm,
    local_runner,
) -> _SamplingOwnershipReply:
    """Encode rank-local failures so every peer still enters the gather."""

    rank = control_comm.rank
    try:
        return _SamplingOwnershipReply(
            rank=rank,
            row=_sampling_ownership_row(control_comm, model_comm, local_runner),
        )
    except BaseException as error:
        return _SamplingOwnershipReply(
            rank=rank,
            error=(f"{type(error).__name__}: {error}")[:1024],
        )


def _validate_decode_graph_capture_plan(plan: object) -> DecodeGraphCapturePlan:
    if not isinstance(plan, DecodeGraphCapturePlan):
        raise RuntimeError(
            f"runner returned malformed decode-graph capture plan: {plan!r}"
        )
    buckets = plan.buckets
    pending = plan.pending_buckets
    if (
        not buckets
        or any(type(bucket) is not int or bucket < 1 for bucket in buckets)
        or tuple(sorted(set(buckets))) != buckets
        or any(type(bucket) is not int for bucket in pending)
        or tuple(bucket for bucket in buckets if bucket in set(pending)) != pending
        or type(plan.max_pages) is not int
        or plan.max_pages < 1
        or type(plan.scratch_page) is not int
        or plan.scratch_page < 0
        or type(plan.capture_model_forward_count) is not int
        or plan.capture_model_forward_count < 1
    ):
        raise RuntimeError(
            f"runner returned malformed decode-graph capture plan: {plan!r}"
        )
    return plan


def _decode_graph_capture_plan_reply(
    control_comm,
    local_runner,
) -> _DecodeGraphCapturePlanReply:
    """Read one rank's immutable graph identity without touching CUDA state."""

    rank = control_comm.rank
    try:
        getter = getattr(local_runner, "decode_graph_capture_plan", None)
        if not callable(getter):
            raise RuntimeError("runner exposes no decode-graph startup plan")
        plan = _validate_decode_graph_capture_plan(getter())
        return _DecodeGraphCapturePlanReply(rank=rank, plan=plan)
    except BaseException as error:
        return _DecodeGraphCapturePlanReply(
            rank=rank,
            error=(f"{type(error).__name__}: {error}")[:1024],
        )


def _validate_decode_graph_capture_plan_replies(
    gathered: object,
    *,
    world_size: int,
) -> DecodeGraphCapturePlan:
    """Reject rank skew before any process enters graph model collectives."""

    if not isinstance(gathered, (tuple, list)) or len(gathered) != world_size:
        raise RuntimeError(
            "decode-graph startup preflight gathered malformed rank count: "
            f"expected {world_size}, got "
            f"{len(gathered) if isinstance(gathered, (tuple, list)) else type(gathered).__name__}"
        )
    by_rank: dict[int, _DecodeGraphCapturePlanReply] = {}
    for slot, reply in enumerate(gathered):
        if (
            not isinstance(reply, _DecodeGraphCapturePlanReply)
            or type(reply.rank) is not int
            or not 0 <= reply.rank < world_size
            or reply.rank in by_rank
            or (reply.plan is None) == (reply.error is None)
        ):
            raise RuntimeError(
                "decode-graph startup preflight slot "
                f"{slot} returned a malformed reply: {reply!r}"
            )
        by_rank[reply.rank] = reply
    if set(by_rank) != set(range(world_size)):
        missing = sorted(set(range(world_size)) - set(by_rank))
        raise RuntimeError(
            f"decode-graph startup preflight is incomplete: missing={missing}"
        )
    failures = tuple(
        f"rank {rank}: {by_rank[rank].error}"
        for rank in range(world_size)
        if by_rank[rank].error is not None
    )
    if failures:
        raise RuntimeError(
            "decode-graph startup preflight failures: " + "; ".join(failures)
        )
    plans = tuple(by_rank[rank].plan for rank in range(world_size))
    if len(set(plans)) != 1:
        raise RuntimeError(
            "decode-graph startup configuration differs across ranks: "
            f"{plans!r}"
        )
    plan = plans[0]
    assert plan is not None
    return plan


def _prepare_decode_graph_capture_reply(
    control_comm,
    local_runner,
    plan: DecodeGraphCapturePlan,
    *,
    attention_dp: bool,
) -> _DecodeGraphCapturePreparationReply:
    """Arm all capture-time MoE layouts, reporting failure before NCCL work."""

    rank = control_comm.rank
    try:
        local = _decode_graph_capture_plan_reply(control_comm, local_runner)
        if local.error is not None or local.plan != plan:
            raise RuntimeError(
                "decode-graph startup plan changed after preflight: "
                f"expected={plan!r}, local={local!r}"
            )
        if attention_dp:
            from kairyu.models.moe_parallel import (
                assert_attention_dp_layouts_idle,
                prepare_attention_dp_layouts,
            )

            model = getattr(local_runner, "_model", None)
            if model is None:
                raise RuntimeError(
                    "attention-DP graph runner exposes no rank-local model"
                )
            layouts = tuple(
                (bucket,) * control_comm.world_size
                for bucket in plan.pending_buckets
                for _ in range(plan.capture_model_forward_count)
            )
            if layouts:
                prepare_attention_dp_layouts(model, layouts)
            else:
                assert_attention_dp_layouts_idle(model)
        return _DecodeGraphCapturePreparationReply(rank=rank)
    except BaseException as error:
        return _DecodeGraphCapturePreparationReply(
            rank=rank,
            error=(f"{type(error).__name__}: {error}")[:1024],
        )


def _validate_decode_graph_capture_preparation_replies(
    gathered: object,
    *,
    world_size: int,
) -> None:
    if not isinstance(gathered, (tuple, list)) or len(gathered) != world_size:
        raise RuntimeError(
            "decode-graph startup preparation gathered malformed rank count"
        )
    by_rank: dict[int, _DecodeGraphCapturePreparationReply] = {}
    for reply in gathered:
        if (
            not isinstance(reply, _DecodeGraphCapturePreparationReply)
            or type(reply.rank) is not int
            or not 0 <= reply.rank < world_size
            or reply.rank in by_rank
            or (
                reply.error is not None
                and (not isinstance(reply.error, str) or not reply.error)
            )
        ):
            raise RuntimeError(
                "decode-graph startup preparation returned a malformed reply"
            )
        by_rank[reply.rank] = reply
    if set(by_rank) != set(range(world_size)):
        raise RuntimeError("decode-graph startup preparation ranks are incomplete")
    failures = tuple(
        f"rank {rank}: {by_rank[rank].error}"
        for rank in range(world_size)
        if by_rank[rank].error is not None
    )
    if failures:
        raise RuntimeError(
            "decode-graph startup preparation failures: " + "; ".join(failures)
        )


def _capture_decode_graphs_reply(
    control_comm,
    local_runner,
    plan: DecodeGraphCapturePlan,
    *,
    attention_dp: bool,
) -> _CaptureDecodeGraphsReply:
    """Warm one rank and preserve convergence through the status gather."""

    rank = control_comm.rank
    try:
        capture = getattr(local_runner, "capture_decode_graphs", None)
        if not callable(capture):
            raise RuntimeError("runner exposes no decode-graph startup warmup")
        buckets = capture()
        if (
            not isinstance(buckets, tuple)
            or not buckets
            or any(type(bucket) is not int or bucket < 1 for bucket in buckets)
            or tuple(sorted(set(buckets))) != buckets
        ):
            raise RuntimeError(
                f"runner returned malformed captured buckets: {buckets!r}"
            )
        if buckets != plan.buckets:
            raise RuntimeError(
                "runner captured buckets that disagree with preflight: "
                f"expected={plan.buckets!r}, got={buckets!r}"
            )
        after = _decode_graph_capture_plan_reply(control_comm, local_runner)
        expected_after = DecodeGraphCapturePlan(
            buckets=plan.buckets,
            pending_buckets=(),
            max_pages=plan.max_pages,
            scratch_page=plan.scratch_page,
            capture_model_forward_count=plan.capture_model_forward_count,
        )
        if after.error is not None or after.plan != expected_after:
            raise RuntimeError(
                "decode-graph startup did not finish every planned bucket: "
                f"expected={expected_after!r}, local={after!r}"
            )
        if attention_dp:
            from kairyu.models.moe_parallel import (
                assert_attention_dp_layouts_consumed,
                assert_attention_dp_layouts_idle,
            )

            model = getattr(local_runner, "_model", None)
            if plan.pending_buckets:
                assert_attention_dp_layouts_consumed(model)
            else:
                assert_attention_dp_layouts_idle(model)
        synchronize = getattr(
            local_runner,
            "synchronize_decode_graph_capture",
            None,
        )
        if not callable(synchronize):
            raise RuntimeError(
                "runner exposes no decode-graph startup synchronization"
            )
        synchronize()
        return _CaptureDecodeGraphsReply(rank=rank, buckets=buckets)
    except BaseException as error:
        return _CaptureDecodeGraphsReply(
            rank=rank,
            error=(f"{type(error).__name__}: {error}")[:1024],
        )


def _validate_capture_decode_graphs_replies(
    gathered: object,
    *,
    world_size: int,
) -> tuple[int, ...]:
    """Require complete, successful, identical startup capture on every rank."""

    if not isinstance(gathered, (tuple, list)) or len(gathered) != world_size:
        raise RuntimeError(
            "decode-graph startup gathered malformed rank count: "
            f"expected {world_size}, got "
            f"{len(gathered) if isinstance(gathered, (tuple, list)) else type(gathered).__name__}"
        )
    by_rank: dict[int, _CaptureDecodeGraphsReply] = {}
    for slot, reply in enumerate(gathered):
        if (
            not isinstance(reply, _CaptureDecodeGraphsReply)
            or type(reply.rank) is not int
            or not 0 <= reply.rank < world_size
            or reply.rank in by_rank
            or (reply.buckets is None) == (reply.error is None)
        ):
            raise RuntimeError(
                "decode-graph startup gather slot "
                f"{slot} returned a malformed reply: {reply!r}"
            )
        by_rank[reply.rank] = reply
    missing = sorted(set(range(world_size)) - set(by_rank))
    if missing:
        raise RuntimeError(
            f"decode-graph startup replies are incomplete: missing={missing}"
        )
    failures = tuple(
        f"rank {rank}: {by_rank[rank].error}"
        for rank in range(world_size)
        if by_rank[rank].error is not None
    )
    if failures:
        raise RuntimeError(
            "decode-graph startup rank failures: " + "; ".join(failures)
        )
    bucket_sets = tuple(by_rank[rank].buckets for rank in range(world_size))
    if len(set(bucket_sets)) != 1:
        raise RuntimeError(
            "decode-graph startup captured different buckets across ranks: "
            f"{bucket_sets!r}"
        )
    buckets = bucket_sets[0]
    assert buckets is not None
    return buckets


def _capture_decode_graphs_transaction(
    control_comm,
    local_runner,
    *,
    attention_dp: bool,
) -> tuple[int, ...]:
    """Run the rank-symmetric startup transaction on a bounded host group."""

    plan_reply = _decode_graph_capture_plan_reply(control_comm, local_runner)
    plan = _validate_decode_graph_capture_plan_replies(
        control_comm.all_gather(plan_reply),
        world_size=control_comm.world_size,
    )
    preparation = _prepare_decode_graph_capture_reply(
        control_comm,
        local_runner,
        plan,
        attention_dp=attention_dp,
    )
    _validate_decode_graph_capture_preparation_replies(
        control_comm.all_gather(preparation),
        world_size=control_comm.world_size,
    )
    local = _capture_decode_graphs_reply(
        control_comm,
        local_runner,
        plan,
        attention_dp=attention_dp,
    )
    return _validate_capture_decode_graphs_replies(
        control_comm.all_gather(local),
        world_size=control_comm.world_size,
    )


def _bind_ep_model_communicator(
    deferred_comm,
    model_comm,
    startup_control_comm,
    *,
    attention_dp: bool,
) -> None:
    """Publish model-group ownership before bounded direct-NCCL setup.

    Direct-NCCL construction performs several host collectives before the
    launcher can report ready, and every live ragged-layout agreement reuses the
    supplied communicator.  Neither operation may run on the ordinary control
    group: workers legitimately wait inside that group's receive while idle, so
    its timeout spans the server lifetime rather than one bounded transaction.
    Binding first also makes a partially initialized serving communicator
    reachable by the launcher's non-collective abandon path if setup raises.
    """

    deferred_comm.bind(model_comm)
    if attention_dp:
        model_comm.enable_attention_dp_direct_nccl(startup_control_comm)


def _abort_ep_worker_model_communicator(comm, device: str) -> None:
    """Best-effort non-collective cleanup after an asymmetric worker failure."""

    import contextlib

    import torch

    with contextlib.suppress(Exception):
        comm.abort_direct_nccl()
    group = getattr(comm, "group", None)
    if group is None:
        return
    with contextlib.suppress(Exception):
        backend = group._get_backend(torch.device(device))
        abort = getattr(backend, "abort", None) or getattr(backend, "_abort", None)
        if abort is not None:
            abort()


_EP_KERNEL_INVENTORY_FIELDS = frozenset(
    {
        "rank",
        "projection_count",
        "projection_inventory_sha256",
        "kernel_counts",
        "attention_output",
        "fused_moe",
        "meta_count",
        "load_info",
    }
)
_EP_ATTENTION_OUTPUT_FIELDS = frozenset(
    {
        "backend",
        "placement",
        "parallel_size",
        "rank",
        "shard_axis",
        "global_in_features",
        "local_in_features",
        "out_features",
        "partial_dtype",
        "block_count",
        "successful_forward_block_count",
    }
)
_EP_ATTENTION_OUTPUT_BLOCK_COUNT = 94
_EP_FUSED_MOE_BLOCK_FIELDS = frozenset(
    {
        "backend",
        "global_expert_count",
        "local_expert_count",
        "local_expert_start",
        "weight_order",
        "scale_layout",
        "output_dtype",
        "ep_partial",
        "enable_alltoall",
        "successful_forward",
    }
)
_EP_ATTENTION_DP_FUSED_MOE_BLOCK_FIELDS = _EP_FUSED_MOE_BLOCK_FIELDS | {
    "attention_dp",
    "attention_placement",
    "attention_output_placement",
    "dispatcher",
    "row_layout",
    "activation_collective",
    "activation_collective_dtype",
    "scale_collective_dtype",
    "combine_collective",
    "post_combine_all_reduce",
    "layout_queue_depth",
    "layout_queue_consumed",
    "completed_layout_steps",
    "last_rank_row_layout",
    "last_local_real_rows",
    "last_padded_rows_per_rank",
    "total_collective_rows",
}
_EP_FUSED_MOE_SUMMARY_FIELDS = (
    _EP_FUSED_MOE_BLOCK_FIELDS - {"successful_forward"}
) | {"block_count", "successful_forward_block_count"}
_EP_ATTENTION_DP_FUSED_MOE_SUMMARY_FIELDS = (
    _EP_ATTENTION_DP_FUSED_MOE_BLOCK_FIELDS - {"successful_forward"}
) | {"block_count", "successful_forward_block_count"}
_EP_KERNEL_LOAD_INFO_FIELDS = frozenset(
    {
        "ep_rank",
        "ep_size",
        "owned_expert_count",
        "owned_expert_indices_sha256",
        "first_owned_expert",
        "last_owned_expert",
        "quantization_source",
        "quantization_method",
        "kv_cache_quant_algo",
        "producer_name",
        "producer_version",
        "checkpoint_tensor_count",
        "rank_loaded_tensor_count",
        "auxiliary_kv_scale_count",
    }
)
_LOWER_HEX = frozenset("0123456789abcdef")


def _json_sha256(value: object) -> str:
    """Hash the same canonical JSON representation used by formal gates."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


_EP_KV_ACCOUNTING_OBSERVATION_FIELDS = frozenset(
    {
        "sequence",
        "request_id",
        "prompt_tokens",
        "cached_tokens",
        "prefill_start",
        "first_num_tokens",
        "first_is_prefill",
        "prompt_token_ids_sha256",
        "page_count",
        "page_ids_sha256",
    }
)
_EP_KV_ACCOUNTING_ROW_FIELDS = frozenset(
    {
        "rank",
        "world_size",
        "capture_action",
        "observation_limit",
        "overflow",
        "capture_error",
        "request_count",
        "prompt_tokens",
        "cached_tokens",
        "observations_sha256",
        "observations",
    }
)


class _EPKVAccountingRecorder:
    """Record first rank-local views only while a formal probe is armed.

    The steady serving path pays one boolean branch and retains no request
    history. M-A2 explicitly arms the recorder before its fixed trace, then
    gathers the rank-local observations once after all requests complete.
    """

    def __init__(self) -> None:
        self._active = False
        self._overflow = False
        self._error: str | None = None
        self._seen: set[str] = set()
        self._observations: list[dict[str, object]] = []

    def start(self) -> None:
        if self._active:
            raise RuntimeError("EP KV accounting capture is already active")
        self._active = True
        self._overflow = False
        self._error = None
        self._seen.clear()
        self._observations.clear()

    def observe(self, chunks, states) -> None:
        if not self._active or self._overflow or self._error is not None:
            return
        try:
            for chunk in chunks:
                request_id = getattr(chunk, "request_id", None)
                if (
                    not isinstance(request_id, str)
                    or not request_id
                    or len(request_id.encode("utf-8")) > _EP_KV_ACCOUNTING_MAX_REQUEST_ID_BYTES
                ):
                    raise RuntimeError(
                        "EP KV accounting observed a malformed request ID"
                    )
                if request_id in self._seen:
                    continue
                state = states.get(request_id)
                if state is None:
                    raise RuntimeError(
                        f"EP KV accounting cannot resolve request {request_id!r}"
                    )
                prompt_token_ids = tuple(state.request.prompt_token_ids)
                allocation = state.allocation
                if allocation is None:
                    raise RuntimeError(
                        f"EP KV accounting observed no allocation for {request_id!r}"
                    )
                page_ids = tuple(allocation.pages)
                cached_tokens = allocation.num_cached_tokens
                prefill_start = state.computed_prompt - chunk.num_tokens
                expected_prefill_start = min(
                    cached_tokens,
                    len(prompt_token_ids) - 1,
                )
                if (
                    not prompt_token_ids
                    or any(type(token_id) is not int for token_id in prompt_token_ids)
                    or not page_ids
                    or any(
                        type(page_id) is not int or page_id < 0
                        for page_id in page_ids
                    )
                    or type(cached_tokens) is not int
                    or not 0 <= cached_tokens <= len(prompt_token_ids)
                    or chunk.is_prefill is not True
                    or type(chunk.num_tokens) is not int
                    or chunk.num_tokens < 1
                    or type(prefill_start) is not int
                    or prefill_start != expected_prefill_start
                ):
                    raise RuntimeError(
                        "EP KV accounting observed malformed first-prefill "
                        f"allocation for {request_id!r}"
                    )
                if len(self._observations) >= _EP_KV_ACCOUNTING_MAX_OBSERVATIONS:
                    self._overflow = True
                    return
                self._seen.add(request_id)
                self._observations.append(
                    {
                        "sequence": len(self._observations),
                        "request_id": request_id,
                        "prompt_tokens": len(prompt_token_ids),
                        "cached_tokens": cached_tokens,
                        "prefill_start": prefill_start,
                        "first_num_tokens": chunk.num_tokens,
                        "first_is_prefill": True,
                        "prompt_token_ids_sha256": _json_sha256(
                            list(prompt_token_ids)
                        ),
                        "page_count": len(page_ids),
                        "page_ids_sha256": _json_sha256(list(page_ids)),
                    }
                )
        except BaseException as error:
            # This diagnostic must never break the inference collective order.
            # Retain the bounded failure and let every rank complete the step;
            # finish() reports it through the normal all-gather boundary.
            self._error = (f"{type(error).__name__}: {error}")[:1024]

    def snapshot(self, *, rank: int, world_size: int, action: str) -> dict[str, object]:
        if not self._active:
            raise RuntimeError("EP KV accounting capture is not active")
        observations = tuple(dict(row) for row in self._observations)
        row = {
            "rank": rank,
            "world_size": world_size,
            "capture_action": action,
            "observation_limit": _EP_KV_ACCOUNTING_MAX_OBSERVATIONS,
            "overflow": self._overflow,
            "capture_error": self._error,
            "request_count": len(observations),
            "prompt_tokens": sum(
                int(observation["prompt_tokens"]) for observation in observations
            ),
            "cached_tokens": sum(
                int(observation["cached_tokens"]) for observation in observations
            ),
            "observations_sha256": _json_sha256(list(observations)),
            "observations": observations,
        }
        if action == "finish":
            self._active = False
            self._overflow = False
            self._error = None
            self._seen.clear()
            self._observations.clear()
        return row


def _ep_kv_accounting_reply(
    control_comm,
    recorder: _EPKVAccountingRecorder,
    *,
    action: str,
) -> _EPKVAccountingReply:
    """Apply one capture boundary and keep every rank in the gather."""

    rank = control_comm.rank
    try:
        if action == "start":
            recorder.start()
        elif action != "finish":
            raise RuntimeError(f"unknown EP KV accounting action {action!r}")
        return _EPKVAccountingReply(
            rank=rank,
            row=recorder.snapshot(
                rank=rank,
                world_size=control_comm.world_size,
                action=action,
            ),
        )
    except BaseException as error:
        return _EPKVAccountingReply(
            rank=rank,
            error=(f"{type(error).__name__}: {error}")[:1024],
        )


def _validate_ep_kv_accounting_rows(
    rows: object,
    *,
    world_size: int,
    action: str,
) -> tuple[dict[str, object], ...]:
    """Validate each rank envelope without hiding a substantive rank drift."""

    if not isinstance(rows, (tuple, list)) or len(rows) != world_size:
        actual = len(rows) if isinstance(rows, (tuple, list)) else type(rows).__name__
        raise RuntimeError(
            "EP KV accounting reply count mismatch: "
            f"expected {world_size}, got {actual}"
        )
    by_rank: dict[int, dict[str, object]] = {}
    for slot, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != _EP_KV_ACCOUNTING_ROW_FIELDS:
            raise RuntimeError(
                f"EP KV accounting reply in gather slot {slot} is malformed"
            )
        rank = row["rank"]
        observations = row["observations"]
        if (
            type(rank) is not int
            or not 0 <= rank < world_size
            or rank in by_rank
            or row["world_size"] != world_size
            or row["capture_action"] != action
            or row["observation_limit"] != _EP_KV_ACCOUNTING_MAX_OBSERVATIONS
            or type(row["overflow"]) is not bool
            or (
                row["capture_error"] is not None
                and (
                    not isinstance(row["capture_error"], str)
                    or not row["capture_error"]
                    or len(row["capture_error"]) > 1024
                )
            )
            or not isinstance(observations, (tuple, list))
            or type(row["request_count"]) is not int
            or row["request_count"] != len(observations)
            or row["request_count"] > _EP_KV_ACCOUNTING_MAX_OBSERVATIONS
            or (
                row["overflow"]
                and row["request_count"] != _EP_KV_ACCOUNTING_MAX_OBSERVATIONS
            )
            or type(row["prompt_tokens"]) is not int
            or type(row["cached_tokens"]) is not int
            or not _is_lower_sha256(row["observations_sha256"])
            or row["observations_sha256"] != _json_sha256(list(observations))
        ):
            raise RuntimeError(
                f"EP KV accounting reply in gather slot {slot} has malformed summary"
            )
        request_ids: set[str] = set()
        prompt_tokens = 0
        cached_tokens = 0
        for sequence, observation in enumerate(observations):
            if (
                not isinstance(observation, dict)
                or set(observation) != _EP_KV_ACCOUNTING_OBSERVATION_FIELDS
                or type(observation["sequence"]) is not int
                or observation["sequence"] != sequence
                or not isinstance(observation["request_id"], str)
                or not observation["request_id"]
                or len(observation["request_id"].encode("utf-8"))
                > _EP_KV_ACCOUNTING_MAX_REQUEST_ID_BYTES
                or observation["request_id"] in request_ids
                or type(observation["prompt_tokens"]) is not int
                or observation["prompt_tokens"] < 1
                or type(observation["cached_tokens"]) is not int
                or not 0
                <= observation["cached_tokens"]
                <= observation["prompt_tokens"]
                or type(observation["prefill_start"]) is not int
                or observation["prefill_start"]
                != min(
                    observation["cached_tokens"],
                    observation["prompt_tokens"] - 1,
                )
                or type(observation["first_num_tokens"]) is not int
                or observation["first_num_tokens"] < 1
                or observation["first_is_prefill"] is not True
                or not _is_lower_sha256(
                    observation["prompt_token_ids_sha256"]
                )
                or type(observation["page_count"]) is not int
                or observation["page_count"] < 1
                or not _is_lower_sha256(observation["page_ids_sha256"])
            ):
                raise RuntimeError(
                    f"EP KV accounting rank {rank} observation {sequence} is malformed"
                )
            request_ids.add(observation["request_id"])
            prompt_tokens += observation["prompt_tokens"]
            cached_tokens += observation["cached_tokens"]
        if (
            row["prompt_tokens"] != prompt_tokens
            or row["cached_tokens"] != cached_tokens
            or (
                action == "start"
                and (
                    observations
                    or row["overflow"]
                    or row["capture_error"] is not None
                )
            )
        ):
            raise RuntimeError(
                f"EP KV accounting rank {rank} summary disagrees with observations"
            )
        by_rank[rank] = row
    if set(by_rank) != set(range(world_size)):
        raise RuntimeError("EP KV accounting ranks are incomplete")
    return tuple(by_rank[rank] for rank in range(world_size))


def _validate_ep_kv_accounting_replies(
    gathered: object,
    *,
    world_size: int,
    action: str,
) -> tuple[dict[str, object], ...]:
    """Validate bounded transport envelopes before exposing rank observations."""

    if not isinstance(gathered, (tuple, list)) or len(gathered) != world_size:
        actual = (
            len(gathered)
            if isinstance(gathered, (tuple, list))
            else type(gathered).__name__
        )
        raise RuntimeError(
            "EP KV accounting gathered malformed rank count: "
            f"expected {world_size}, got {actual}"
        )
    replies: dict[int, _EPKVAccountingReply] = {}
    for slot, reply in enumerate(gathered):
        if not isinstance(reply, _EPKVAccountingReply):
            raise RuntimeError(
                f"EP KV accounting gather slot {slot} returned malformed "
                f"{type(reply).__name__}"
            )
        if (
            type(reply.rank) is not int
            or not 0 <= reply.rank < world_size
            or reply.rank in replies
            or (reply.row is None) == (reply.error is None)
            or (
                reply.error is not None
                and (not isinstance(reply.error, str) or not reply.error)
            )
        ):
            raise RuntimeError(
                f"EP KV accounting gather slot {slot} has malformed envelope"
            )
        replies[reply.rank] = reply
    missing = sorted(set(range(world_size)) - set(replies))
    if missing:
        raise RuntimeError(f"EP KV accounting ranks are incomplete: missing={missing}")
    failures = tuple(
        f"rank {rank}: {replies[rank].error}"
        for rank in range(world_size)
        if replies[rank].error is not None
    )
    if failures:
        raise RuntimeError("EP KV accounting rank failures: " + "; ".join(failures))
    return _validate_ep_kv_accounting_rows(
        tuple(replies[rank].row for rank in range(world_size)),
        world_size=world_size,
        action=action,
    )


def _ep_load_info_summary(
    local_runner,
    *,
    rank: int,
    world_size: int,
) -> dict[str, object]:
    """Reduce rank-local loader evidence without sending the expert list."""

    load_info = getattr(local_runner, "expert_parallel_load_info", None)
    if load_info is None:
        raise RuntimeError(f"rank {rank} runner has no expert_parallel_load_info")
    ep_rank = getattr(load_info, "ep_rank", None)
    ep_size = getattr(load_info, "ep_size", None)
    owned = getattr(load_info, "owned_expert_indices", None)
    if type(ep_rank) is not int or ep_rank != rank:
        raise RuntimeError(
            f"rank {rank} load info has malformed ep_rank={ep_rank!r}"
        )
    if type(ep_size) is not int or ep_size != world_size:
        raise RuntimeError(
            f"rank {rank} load info has malformed ep_size={ep_size!r}; "
            f"expected {world_size}"
        )
    if (
        not isinstance(owned, tuple)
        or not owned
        or any(type(index) is not int or index < 0 for index in owned)
        or tuple(sorted(set(owned))) != owned
        or owned[-1] - owned[0] + 1 != len(owned)
    ):
        raise RuntimeError(
            f"rank {rank} load info has malformed contiguous expert ownership"
        )

    string_fields = (
        "quantization_source",
        "quantization_method",
    )
    for name in string_fields:
        value = getattr(load_info, name, None)
        if not isinstance(value, str) or not value:
            raise RuntimeError(
                f"rank {rank} load info has malformed {name}={value!r}"
            )
    optional_string_fields = (
        "kv_cache_quant_algo",
        "producer_name",
        "producer_version",
    )
    for name in optional_string_fields:
        value = getattr(load_info, name, None)
        if value is not None and (not isinstance(value, str) or not value):
            raise RuntimeError(
                f"rank {rank} load info has malformed {name}={value!r}"
            )
    count_fields = (
        "checkpoint_tensor_count",
        "rank_loaded_tensor_count",
        "auxiliary_kv_scale_count",
    )
    for name in count_fields:
        value = getattr(load_info, name, None)
        if type(value) is not int or value < 0:
            raise RuntimeError(
                f"rank {rank} load info has malformed {name}={value!r}"
            )

    return {
        "ep_rank": ep_rank,
        "ep_size": ep_size,
        "owned_expert_count": len(owned),
        "owned_expert_indices_sha256": _json_sha256(list(owned)),
        "first_owned_expert": owned[0],
        "last_owned_expert": owned[-1],
        "quantization_source": load_info.quantization_source,
        "quantization_method": load_info.quantization_method,
        "kv_cache_quant_algo": load_info.kv_cache_quant_algo,
        "producer_name": load_info.producer_name,
        "producer_version": load_info.producer_version,
        "checkpoint_tensor_count": load_info.checkpoint_tensor_count,
        "rank_loaded_tensor_count": load_info.rank_loaded_tensor_count,
        "auxiliary_kv_scale_count": load_info.auxiliary_kv_scale_count,
    }


def _ep_attention_output_summary(
    modules: tuple[tuple[str, object], ...],
    *,
    rank: int,
    world_size: int,
    attention_dp: bool = False,
) -> dict[str, object]:
    """Validate every canonical NVFP4 ``o_proj`` and reduce its live geometry."""

    if type(attention_dp) is not bool:
        raise TypeError("attention_dp must be a boolean")

    import torch

    from kairyu.quant.linear import (
        LinearKernel,
        LinearRole,
        NvFp4Linear,
        TensorParallelMode,
    )

    expected_names = tuple(
        f"model.layers.{layer}.self_attn.o_proj"
        for layer in range(_EP_ATTENTION_OUTPUT_BLOCK_COUNT)
    )
    by_name = dict(modules)
    actual_names = tuple(
        sorted(name for name, _module in modules if name.endswith(".self_attn.o_proj"))
    )
    if set(actual_names) != set(expected_names) or len(actual_names) != len(
        expected_names
    ):
        missing = sorted(set(expected_names) - set(actual_names))
        unexpected = sorted(set(actual_names) - set(expected_names))
        raise RuntimeError(
            f"rank {rank} attention output inventory mismatch: "
            f"missing={missing[:4]} ({len(missing)} total), "
            f"unexpected={unexpected[:4]} ({len(unexpected)} total)"
        )

    geometry: tuple[int, int, int] | None = None
    successful: int | None = None if attention_dp else 0
    for layer, name in enumerate(expected_names):
        module = by_name.get(name)
        if not isinstance(module, NvFp4Linear):
            raise RuntimeError(
                f"rank {rank} attention output {name!r} is not NvFp4Linear"
            )
        context = getattr(module, "linear_context", None)
        placement = getattr(context, "tensor_parallel", None)
        common_malformed = (
            context is None
            or context.qualified_name != name
            or context.role is not LinearRole.ATTENTION_OUTPUT
            or context.layer_index != layer
            or context.dtype is not torch.bfloat16
            or placement is None
            or module.in_features != placement.local_in_features
            or module.out_features != placement.local_out_features
        )
        if attention_dp:
            malformed_placement = (
                common_malformed
                or placement.mode is not TensorParallelMode.REPLICATED
                or placement.rank != 0
                or placement.world_size != 1
                or placement.shard_dim is not None
                or placement.global_in_features != placement.local_in_features
                or placement.global_out_features != placement.local_out_features
            )
            placement_name = "replicated"
        else:
            malformed_placement = (
                common_malformed
                or placement.mode is not TensorParallelMode.ROW
                or placement.rank != rank
                or placement.world_size != world_size
                or placement.shard_dim != 1
                or placement.global_in_features
                != placement.local_in_features * world_size
                or placement.global_out_features != placement.local_out_features
            )
            placement_name = "row_parallel"
        if malformed_placement:
            raise RuntimeError(
                f"rank {rank} attention output {name!r} has malformed "
                f"{placement_name} placement"
            )
        if (
            tuple(module.weight.shape)
            != (placement.local_out_features, placement.local_in_features // 2)
            or tuple(module.weight_scale.shape)
            != (placement.local_out_features, placement.local_in_features // 16)
        ):
            raise RuntimeError(
                f"rank {rank} attention output {name!r} has malformed NVFP4 weights"
            )
        selection = getattr(module, "linear_selection", None)
        if getattr(selection, "kernel", None) is not LinearKernel.FLASHINFER_NVFP4:
            raise RuntimeError(
                f"rank {rank} attention output {name!r} does not select "
                "FlashInfer NVFP4"
            )
        if attention_dp:
            forbidden_bindings = (
                "_kairyu_replicated_input_shard",
                "_kairyu_row_parallel",
                "_kairyu_row_parallel_executed",
                "_kairyu_row_parallel_partial_dtype",
            )
            present = tuple(
                binding for binding in forbidden_bindings if hasattr(module, binding)
            )
            if present:
                raise RuntimeError(
                    f"rank {rank} attention output {name!r} unexpectedly has "
                    f"row-parallel bindings {present}"
                )
        else:
            input_binding = getattr(module, "_kairyu_replicated_input_shard", None)
            reduce_binding = getattr(module, "_kairyu_row_parallel", None)
            reduce_comm = getattr(reduce_binding, "comm", None)
            if (
                input_binding is None
                or input_binding.rank != rank
                or input_binding.world_size != world_size
                or input_binding.local_features != placement.local_in_features
                or reduce_binding is None
                or reduce_binding.context is not None
                or getattr(reduce_comm, "rank", None) != rank
                or getattr(reduce_comm, "world_size", None) != world_size
            ):
                raise RuntimeError(
                    f"rank {rank} attention output {name!r} lacks its EP row bindings"
                )
            executed = getattr(module, "_kairyu_row_parallel_executed", None)
            partial_dtype = getattr(
                module,
                "_kairyu_row_parallel_partial_dtype",
                None,
            )
            if (
                type(executed) is not bool
                or (executed and partial_dtype is not torch.bfloat16)
                or (not executed and partial_dtype is not None)
            ):
                raise RuntimeError(
                    f"rank {rank} attention output {name!r} has malformed "
                    "execution/dtype markers"
                )
            assert successful is not None
            successful += int(executed)
        current = (
            placement.global_in_features,
            placement.local_in_features,
            placement.global_out_features,
        )
        if geometry is None:
            geometry = current
        elif current != geometry:
            raise RuntimeError(
                f"rank {rank} attention output geometry differs across layers"
            )

    if geometry is None:  # guarded by the exact inventory check above
        raise RuntimeError(f"rank {rank} has no attention output projections")
    global_in_features, local_in_features, out_features = geometry
    return {
        "backend": "flashinfer.mm_fp4",
        "placement": "replicated" if attention_dp else "row_parallel",
        "parallel_size": 1 if attention_dp else world_size,
        "rank": rank,
        "shard_axis": None if attention_dp else 1,
        "global_in_features": global_in_features,
        "local_in_features": local_in_features,
        "out_features": out_features,
        "partial_dtype": None if attention_dp else "bfloat16",
        "block_count": _EP_ATTENTION_OUTPUT_BLOCK_COUNT,
        "successful_forward_block_count": successful,
    }


def _ep_kernel_inventory_row(
    control_comm,
    local_runner,
) -> dict[str, object]:
    """Measure one local model while retaining only a names digest."""

    from kairyu.models.moe_parallel import EpMoeBlock
    from kairyu.quant.linear import NvFp4Linear

    rank = control_comm.rank
    world_size = control_comm.world_size
    if type(rank) is not int or type(world_size) is not int or world_size < 1:
        raise RuntimeError(
            "EP kernel inventory communicator metadata must be positive integers"
        )
    execution_mode = getattr(
        local_runner,
        "execution_mode",
        _EP_CORRECTNESS_MODE,
    )
    if execution_mode not in {_EP_CORRECTNESS_MODE, _EP_ATTENTION_DP_MODE}:
        raise RuntimeError(
            f"rank {rank} runner has unknown EP execution_mode={execution_mode!r}"
        )
    attention_dp = execution_mode == _EP_ATTENTION_DP_MODE
    if attention_dp and (
        world_size != _EP_ATTENTION_DP_SIZE
        or getattr(local_runner, "moe_dispatcher", None)
        != _EP_ATTENTION_DP_DISPATCHER
    ):
        raise RuntimeError(
            f"rank {rank} runner has malformed attention-DP dispatcher topology"
        )
    model = getattr(local_runner, "_model", None)
    if model is None or not callable(getattr(model, "named_modules", None)):
        raise RuntimeError(f"rank {rank} runner does not expose its local model")

    modules = tuple(model.named_modules())
    fused_blocks = tuple(
        (name, module.fused_nvfp4_metadata())
        for name, module in modules
        if isinstance(module, EpMoeBlock)
        and module.fused_nvfp4_metadata() is not None
    )
    fused_prefixes = tuple(f"{name}.experts." for name, _metadata in fused_blocks)
    names: list[str] = []
    kernel_counts: dict[str, int] = {}
    for name, module in modules:
        if not isinstance(module, NvFp4Linear):
            continue
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"rank {rank} has an unnamed NVFP4 projection")
        selection = getattr(module, "linear_selection", None)
        quantization = getattr(selection, "quantization", None)
        method = getattr(getattr(quantization, "method", None), "value", None)
        kernel = getattr(getattr(selection, "kernel", None), "value", None)
        if not isinstance(method, str) or not method:
            raise RuntimeError(
                f"rank {rank} NVFP4 projection {name!r} has no quant method"
            )
        if not isinstance(kernel, str) or not kernel:
            raise RuntimeError(
                f"rank {rank} NVFP4 projection {name!r} has no kernel"
            )
        names.append(name)
        fused_expert = any(name.startswith(prefix) for prefix in fused_prefixes)
        key = (
            f"{method}:flashinfer_cutlass_fused_moe"
            if fused_expert
            else f"{method}:{kernel}"
        )
        kernel_counts[key] = kernel_counts.get(key, 0) + 1
    names.sort()
    if not names:
        raise RuntimeError(f"rank {rank} local model has no NVFP4 projections")
    if len(set(names)) != len(names):
        raise RuntimeError(f"rank {rank} local model has duplicate module names")

    named_parameters = getattr(model, "named_parameters", None)
    named_buffers = getattr(model, "named_buffers", None)
    if not callable(named_parameters) or not callable(named_buffers):
        raise RuntimeError(
            f"rank {rank} local model cannot enumerate parameters and buffers"
        )
    meta_parameters = sum(
        tensor.device.type == "meta" for _name, tensor in named_parameters()
    )
    meta_buffers = sum(
        tensor.device.type == "meta" for _name, tensor in named_buffers()
    )
    fused_moe = None
    if fused_blocks:
        expected_fields = (
            _EP_ATTENTION_DP_FUSED_MOE_BLOCK_FIELDS
            if attention_dp
            else _EP_FUSED_MOE_BLOCK_FIELDS
        )
        metadata_rows = [metadata for _name, metadata in fused_blocks]
        if any(
            not isinstance(metadata, dict)
            or set(metadata) != expected_fields
            or type(metadata["successful_forward"]) is not bool
            for metadata in metadata_rows
        ):
            raise RuntimeError(f"rank {rank} has malformed fused-MoE metadata")
        invariant_fields = expected_fields - {"successful_forward"}
        first = metadata_rows[0]
        if any(
            any(metadata[field] != first[field] for field in invariant_fields)
            for metadata in metadata_rows[1:]
        ):
            raise RuntimeError(f"rank {rank} fused-MoE block metadata disagrees")
        fused_moe = {
            **{field: first[field] for field in invariant_fields},
            "block_count": len(fused_blocks),
            "successful_forward_block_count": sum(
                metadata["successful_forward"] is True
                for metadata in metadata_rows
            ),
        }
    if attention_dp and (
        fused_moe is None
        or fused_moe["block_count"] != _EP_ATTENTION_OUTPUT_BLOCK_COUNT
    ):
        raise RuntimeError(
            f"rank {rank} attention-DP requires fused NVFP4 metadata for all "
            f"{_EP_ATTENTION_OUTPUT_BLOCK_COUNT} sparse layers"
        )
    return {
        "rank": rank,
        "projection_count": len(names),
        "projection_inventory_sha256": _json_sha256(names),
        "kernel_counts": dict(sorted(kernel_counts.items())),
        "attention_output": _ep_attention_output_summary(
            modules,
            rank=rank,
            world_size=world_size,
            attention_dp=attention_dp,
        ),
        "fused_moe": fused_moe,
        "meta_count": {
            "parameters": meta_parameters,
            "buffers": meta_buffers,
            "total": meta_parameters + meta_buffers,
        },
        "load_info": _ep_load_info_summary(
            local_runner,
            rank=rank,
            world_size=world_size,
        ),
    }


def _ep_kernel_inventory_reply(control_comm, local_runner) -> _EPKernelInventoryReply:
    """Encode rank-local failures so all peers still enter the gloo gather."""

    rank = control_comm.rank
    try:
        return _EPKernelInventoryReply(
            rank=rank,
            row=_ep_kernel_inventory_row(control_comm, local_runner),
        )
    except BaseException as error:
        return _EPKernelInventoryReply(
            rank=rank,
            error=(f"{type(error).__name__}: {error}")[:1024],
        )


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _LOWER_HEX for character in value)
    )


def _validate_ep_kernel_inventory_rows(
    rows: object,
    *,
    world_size: int,
) -> tuple[dict[str, object], ...]:
    """Reject missing ranks and every untyped or internally inconsistent row."""

    if not isinstance(rows, (tuple, list)) or len(rows) != world_size:
        actual = len(rows) if isinstance(rows, (tuple, list)) else type(rows).__name__
        raise RuntimeError(
            "EP kernel inventory reply count mismatch: "
            f"expected {world_size}, got {actual}"
        )
    by_rank: dict[int, dict[str, object]] = {}
    for slot, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != _EP_KERNEL_INVENTORY_FIELDS:
            raise RuntimeError(
                f"EP kernel inventory reply in gather slot {slot} is malformed"
            )
        rank = row["rank"]
        if type(rank) is not int or not 0 <= rank < world_size or rank in by_rank:
            raise RuntimeError(
                f"EP kernel inventory reply in gather slot {slot} has invalid "
                f"rank={rank!r}"
            )
        projection_count = row["projection_count"]
        kernel_counts = row["kernel_counts"]
        if type(projection_count) is not int or projection_count < 1:
            raise RuntimeError(
                f"EP kernel inventory rank {rank} has malformed projection_count"
            )
        if not _is_lower_sha256(row["projection_inventory_sha256"]):
            raise RuntimeError(
                f"EP kernel inventory rank {rank} has malformed inventory SHA-256"
            )
        if (
            not isinstance(kernel_counts, dict)
            or not kernel_counts
            or any(
                not isinstance(key, str)
                or key.count(":") != 1
                or any(not part for part in key.split(":"))
                or type(count) is not int
                or count < 1
                for key, count in kernel_counts.items()
            )
            or sum(kernel_counts.values()) != projection_count
        ):
            raise RuntimeError(
                f"EP kernel inventory rank {rank} has malformed kernel_counts"
            )
        attention_output = row["attention_output"]
        if not isinstance(attention_output, dict) or set(
            attention_output
        ) != _EP_ATTENTION_OUTPUT_FIELDS:
            raise RuntimeError(
                f"EP kernel inventory rank {rank} has malformed attention_output"
            )
        attention_dp = attention_output["placement"] == "replicated"
        common_attention_malformed = (
            attention_output["backend"] != "flashinfer.mm_fp4"
            or attention_output["placement"] not in {"row_parallel", "replicated"}
            or type(attention_output["rank"]) is not int
            or attention_output["rank"] != rank
            or type(attention_output["global_in_features"]) is not int
            or type(attention_output["local_in_features"]) is not int
            or type(attention_output["out_features"]) is not int
            or attention_output["global_in_features"] < 1
            or attention_output["local_in_features"] < 1
            or attention_output["out_features"] < 1
            or type(attention_output["block_count"]) is not int
            or attention_output["block_count"]
            != _EP_ATTENTION_OUTPUT_BLOCK_COUNT
        )
        if attention_dp:
            mode_attention_malformed = (
                type(attention_output["parallel_size"]) is not int
                or attention_output["parallel_size"] != 1
                or attention_output["shard_axis"] is not None
                or attention_output["global_in_features"]
                != attention_output["local_in_features"]
                or attention_output["partial_dtype"] is not None
                or attention_output["successful_forward_block_count"] is not None
            )
        else:
            mode_attention_malformed = (
                type(attention_output["parallel_size"]) is not int
                or attention_output["parallel_size"] != world_size
                or type(attention_output["shard_axis"]) is not int
                or attention_output["shard_axis"] != 1
                or attention_output["global_in_features"]
                != attention_output["local_in_features"] * world_size
                or attention_output["partial_dtype"] != "bfloat16"
                or type(attention_output["successful_forward_block_count"])
                is not int
                or not 0
                <= attention_output["successful_forward_block_count"]
                <= _EP_ATTENTION_OUTPUT_BLOCK_COUNT
            )
        if common_attention_malformed or mode_attention_malformed:
            raise RuntimeError(
                f"EP kernel inventory rank {rank} has malformed attention_output"
            )
        fused_moe = row["fused_moe"]
        if fused_moe is not None:
            expected_fused_fields = (
                _EP_ATTENTION_DP_FUSED_MOE_SUMMARY_FIELDS
                if attention_dp
                else _EP_FUSED_MOE_SUMMARY_FIELDS
            )
            common_fused_malformed = (
                not isinstance(fused_moe, dict)
                or set(fused_moe) != expected_fused_fields
                or fused_moe["backend"] != "flashinfer.cutlass_fused_moe"
                or type(fused_moe["global_expert_count"]) is not int
                or fused_moe["global_expert_count"] < 1
                or type(fused_moe["local_expert_count"]) is not int
                or fused_moe["local_expert_count"] < 1
                or type(fused_moe["local_expert_start"]) is not int
                or fused_moe["local_expert_start"] < 0
                or fused_moe["weight_order"] != "up_gate"
                or fused_moe["scale_layout"] != "128x4"
                or fused_moe["output_dtype"] != "bfloat16"
                or fused_moe["ep_partial"] is not True
                or fused_moe["enable_alltoall"] is not False
                or type(fused_moe["block_count"]) is not int
                or fused_moe["block_count"] < 1
                or type(fused_moe["successful_forward_block_count"]) is not int
                or not 0
                <= fused_moe["successful_forward_block_count"]
                <= fused_moe["block_count"]
            )
            mode_fused_malformed = False
            if not common_fused_malformed and attention_dp:
                last_layout = fused_moe["last_rank_row_layout"]
                completed_steps = fused_moe["completed_layout_steps"]
                no_completed_layout = (
                    completed_steps == 0
                    and last_layout is None
                    and fused_moe["last_local_real_rows"] is None
                    and fused_moe["last_padded_rows_per_rank"] is None
                    and fused_moe["total_collective_rows"] is None
                    and fused_moe["successful_forward_block_count"] == 0
                )
                idle_layout = (
                    no_completed_layout
                    and fused_moe["dispatcher"] == _EP_ATTENTION_DP_DISPATCHER
                    and fused_moe["row_layout"] == "rank_major_zero_padded"
                    and fused_moe["activation_collective"]
                    == "tensor_all_gather_many"
                    and fused_moe["combine_collective"]
                    == "tensor_reduce_scatter"
                )
                completed_layout_common = (
                    type(completed_steps) is int
                    and completed_steps > 0
                    and isinstance(last_layout, tuple)
                    and len(last_layout) == world_size
                    and all(type(count) is int and count > 0 for count in last_layout)
                    and fused_moe["last_local_real_rows"] == last_layout[rank]
                    and fused_moe["successful_forward_block_count"]
                    == fused_moe["block_count"]
                )
                padded_layout = (
                    completed_layout_common
                    and fused_moe["dispatcher"] == _EP_ATTENTION_DP_DISPATCHER
                    and fused_moe["row_layout"] == "rank_major_zero_padded"
                    and fused_moe["activation_collective"]
                    == "tensor_all_gather_many"
                    and fused_moe["combine_collective"]
                    == "tensor_reduce_scatter"
                    and fused_moe["last_padded_rows_per_rank"]
                    == max(last_layout)
                    and fused_moe["total_collective_rows"]
                    == max(last_layout) * world_size
                )
                ragged_layout = (
                    completed_layout_common
                    and fused_moe["dispatcher"]
                    == "nvfp4_allgatherv_reduce_scatterv"
                    and fused_moe["row_layout"] == "rank_major_ragged"
                    and fused_moe["activation_collective"]
                    == "tensor_all_gatherv_many"
                    and fused_moe["combine_collective"]
                    == "tensor_reduce_scatterv"
                    and fused_moe["last_padded_rows_per_rank"] is None
                    and fused_moe["total_collective_rows"] == sum(last_layout)
                )
                mode_fused_malformed = (
                    fused_moe["attention_dp"] is not True
                    or fused_moe["attention_placement"]
                    != "request_owned_data_parallel"
                    or fused_moe["attention_output_placement"] != "replicated"
                    or fused_moe["activation_collective_dtype"]
                    != "nvfp4_packed_uint8"
                    or fused_moe["scale_collective_dtype"] != "uint8"
                    or fused_moe["post_combine_all_reduce"] is not False
                    or type(fused_moe["layout_queue_depth"]) is not int
                    or fused_moe["layout_queue_depth"] != 0
                    or type(fused_moe["layout_queue_consumed"]) is not int
                    or fused_moe["layout_queue_consumed"] != 0
                    or type(completed_steps) is not int
                    or completed_steps < 0
                    or fused_moe["block_count"]
                    != _EP_ATTENTION_OUTPUT_BLOCK_COUNT
                    or not (
                        idle_layout
                        or padded_layout
                        or ragged_layout
                    )
                )
            if common_fused_malformed or mode_fused_malformed:
                raise RuntimeError(
                    f"EP kernel inventory rank {rank} has malformed fused_moe"
                )
        elif attention_dp:
            raise RuntimeError(
                f"EP kernel inventory rank {rank} attention-DP has no fused_moe"
            )
        meta_count = row["meta_count"]
        if (
            not isinstance(meta_count, dict)
            or set(meta_count) != {"parameters", "buffers", "total"}
            or any(type(value) is not int or value < 0 for value in meta_count.values())
            or meta_count["total"]
            != meta_count["parameters"] + meta_count["buffers"]
        ):
            raise RuntimeError(
                f"EP kernel inventory rank {rank} has malformed meta_count"
            )
        load_info = row["load_info"]
        if not isinstance(load_info, dict) or set(load_info) != _EP_KERNEL_LOAD_INFO_FIELDS:
            raise RuntimeError(
                f"EP kernel inventory rank {rank} has malformed load_info fields"
            )
        integer_fields = {
            "ep_rank",
            "ep_size",
            "owned_expert_count",
            "first_owned_expert",
            "last_owned_expert",
            "checkpoint_tensor_count",
            "rank_loaded_tensor_count",
            "auxiliary_kv_scale_count",
        }
        if any(type(load_info[name]) is not int for name in integer_fields):
            raise RuntimeError(
                f"EP kernel inventory rank {rank} has untyped load_info counts"
            )
        if (
            load_info["ep_rank"] != rank
            or load_info["ep_size"] != world_size
            or load_info["owned_expert_count"] < 1
            or load_info["first_owned_expert"] < 0
            or load_info["last_owned_expert"] - load_info["first_owned_expert"] + 1
            != load_info["owned_expert_count"]
            or load_info["checkpoint_tensor_count"] < 1
            or load_info["rank_loaded_tensor_count"] < 1
            or load_info["auxiliary_kv_scale_count"] < 0
            or not _is_lower_sha256(load_info["owned_expert_indices_sha256"])
        ):
            raise RuntimeError(
                f"EP kernel inventory rank {rank} has inconsistent load_info"
            )
        for name in ("quantization_source", "quantization_method"):
            value = load_info[name]
            if not isinstance(value, str) or not value:
                raise RuntimeError(
                    f"EP kernel inventory rank {rank} has malformed load_info {name}"
                )
        for name in ("kv_cache_quant_algo", "producer_name", "producer_version"):
            value = load_info[name]
            if value is not None and (not isinstance(value, str) or not value):
                raise RuntimeError(
                    f"EP kernel inventory rank {rank} has malformed load_info {name}"
                )
        if fused_moe is not None and (
            fused_moe["global_expert_count"]
            != fused_moe["local_expert_count"] * world_size
            or fused_moe["local_expert_count"] != load_info["owned_expert_count"]
            or fused_moe["local_expert_start"] != load_info["first_owned_expert"]
        ):
            raise RuntimeError(
                f"EP kernel inventory rank {rank} fused_moe disagrees with load_info"
            )
        by_rank[rank] = row
    expected = set(range(world_size))
    if set(by_rank) != expected:
        raise RuntimeError(
            "EP kernel inventory ranks are incomplete: "
            f"expected={sorted(expected)}, got={sorted(by_rank)}"
        )
    invariant_attention_fields = _EP_ATTENTION_OUTPUT_FIELDS - {"rank"}
    first_attention = by_rank[0]["attention_output"]
    if any(
        any(
            row["attention_output"][field] != first_attention[field]
            for field in invariant_attention_fields
        )
        for row in by_rank.values()
    ):
        raise RuntimeError(
            "EP kernel inventory attention_output geometry differs across ranks"
        )
    if first_attention["placement"] == "replicated":
        invariant_fused_fields = (
            _EP_ATTENTION_DP_FUSED_MOE_SUMMARY_FIELDS
            - {"local_expert_start", "last_local_real_rows"}
        )
        first_fused = by_rank[0]["fused_moe"]
        if any(
            any(
                row["fused_moe"][field] != first_fused[field]
                for field in invariant_fused_fields
            )
            for row in by_rank.values()
        ):
            raise RuntimeError(
                "EP kernel inventory attention-DP fused_moe differs across ranks"
            )
    return tuple(by_rank[rank] for rank in range(world_size))


def _validate_ep_kernel_inventory_replies(
    gathered: object,
    *,
    world_size: int,
) -> tuple[dict[str, object], ...]:
    """Validate transport envelopes before exposing their measured rows."""

    if not isinstance(gathered, (tuple, list)) or len(gathered) != world_size:
        actual = (
            len(gathered)
            if isinstance(gathered, (tuple, list))
            else type(gathered).__name__
        )
        raise RuntimeError(
            "EP kernel inventory gathered malformed rank count: "
            f"expected {world_size}, got {actual}"
        )
    replies: dict[int, _EPKernelInventoryReply] = {}
    for slot, reply in enumerate(gathered):
        if not isinstance(reply, _EPKernelInventoryReply):
            raise RuntimeError(
                f"EP kernel inventory gather slot {slot} returned malformed "
                f"{type(reply).__name__}"
            )
        if (
            type(reply.rank) is not int
            or not 0 <= reply.rank < world_size
            or reply.rank in replies
        ):
            raise RuntimeError(
                f"EP kernel inventory gather slot {slot} has invalid rank="
                f"{reply.rank!r}"
            )
        if (reply.row is None) == (reply.error is None):
            raise RuntimeError(
                f"EP kernel inventory rank {reply.rank} reply must contain exactly "
                "one of row or error"
            )
        if reply.error is not None and (
            not isinstance(reply.error, str) or not reply.error
        ):
            raise RuntimeError(
                f"EP kernel inventory rank {reply.rank} has malformed error"
            )
        replies[reply.rank] = reply
    missing = sorted(set(range(world_size)) - set(replies))
    if missing:
        raise RuntimeError(f"EP kernel inventory ranks are incomplete: missing={missing}")
    failures = tuple(
        f"rank {rank}: {replies[rank].error}"
        for rank in range(world_size)
        if replies[rank].error is not None
    )
    if failures:
        raise RuntimeError("EP kernel inventory rank failures: " + "; ".join(failures))
    return _validate_ep_kernel_inventory_rows(
        tuple(replies[rank].row for rank in range(world_size)),
        world_size=world_size,
    )


def _prefill_stats_row(control_comm, local_runner, *, reset: bool) -> dict[str, object]:
    getter = getattr(local_runner, "prefill_execution_stats", None)
    if not callable(getter):
        raise RuntimeError(f"rank {control_comm.rank} runner has no prefill_execution_stats")
    stats = getter(reset=reset)
    if not isinstance(stats, dict):
        raise RuntimeError(f"rank {control_comm.rank} prefill stats must be a dict")
    device = str(getattr(local_runner, "_device", ""))
    if not device:
        raise RuntimeError(f"rank {control_comm.rank} runner has no compute device")
    return {
        "rank": control_comm.rank,
        "world_size": control_comm.world_size,
        "device": device,
        "stats": stats,
    }


def _prefill_stats_packet(control_comm, model_comm, local_runner, *, reset: bool):
    """Serialize one stats row without letting a rank miss the collective.

    The control group intentionally has a process-lifetime timeout because
    workers idle inside its broadcast.  Diagnostics instead use the model
    group's 120-second timeout.  A rank-local getter/serialization failure is
    encoded into the same fixed-size packet so peers still enter the gather.
    """
    import torch

    try:
        envelope: dict[str, object] = {
            "ok": True,
            "row": _prefill_stats_row(control_comm, local_runner, reset=reset),
        }
        raw = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    except Exception as error:
        envelope = {
            "ok": False,
            "rank": control_comm.rank,
            "error": (f"{type(error).__name__}: {error}")[:1024],
        }
        raw = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    capacity = _PREFILL_STATS_PACKET_BYTES - 2
    if len(raw) > capacity:
        raw = json.dumps(
            {
                "ok": False,
                "rank": control_comm.rank,
                "error": (f"serialized prefill stats exceed {capacity} bytes"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    device = getattr(model_comm, "_device", None)
    packet = torch.zeros(
        _PREFILL_STATS_PACKET_BYTES,
        dtype=torch.uint8,
        device=device,
    )
    packet[0] = len(raw) & 0xFF
    packet[1] = len(raw) >> 8
    packet[2 : 2 + len(raw)].copy_(torch.tensor(tuple(raw), dtype=torch.uint8, device=device))
    return packet


def _decode_prefill_stats_packets(gathered, *, world_size: int) -> tuple[dict[str, object], ...]:
    """Decode a bounded tensor gather and surface every rank-local error."""
    import torch

    if (
        not isinstance(gathered, torch.Tensor)
        or gathered.dtype != torch.uint8
        or gathered.ndim != 1
        or gathered.numel() != world_size * _PREFILL_STATS_PACKET_BYTES
    ):
        raise RuntimeError("prefill stats tensor gather has malformed shape or dtype")
    packets = gathered.reshape(world_size, _PREFILL_STATS_PACKET_BYTES).cpu()
    rows: list[object] = []
    failures: list[str] = []
    for rank, packet in enumerate(packets):
        length = int(packet[0]) | (int(packet[1]) << 8)
        if not 0 < length <= _PREFILL_STATS_PACKET_BYTES - 2:
            failures.append(f"rank {rank}: malformed packet length {length}")
            continue
        try:
            envelope = json.loads(bytes(packet[2 : 2 + length].tolist()))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            failures.append(f"rank {rank}: invalid JSON: {error}")
            continue
        if not isinstance(envelope, dict):
            failures.append(f"rank {rank}: envelope is not an object")
        elif envelope.get("ok") is True:
            rows.append(envelope.get("row"))
        else:
            failures.append(f"rank {rank}: {envelope.get('error', 'unknown error')}")
    if failures:
        raise RuntimeError("prefill stats rank failures: " + "; ".join(failures))
    return _validate_prefill_stats_rows(rows, world_size=world_size)


def _validate_prefill_stats_rows(rows: object, *, world_size: int) -> tuple[dict[str, object], ...]:
    if not isinstance(rows, (tuple, list)) or len(rows) != world_size:
        raise RuntimeError(
            "prefill stats reply count mismatch: "
            f"expected {world_size}, got "
            f"{len(rows) if isinstance(rows, (tuple, list)) else type(rows).__name__}"
        )
    by_rank: dict[int, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "rank",
            "world_size",
            "device",
            "stats",
        }:
            raise RuntimeError("prefill stats reply is malformed")
        rank = row["rank"]
        if (
            type(rank) is not int
            or rank in by_rank
            or row["world_size"] != world_size
            or not isinstance(row["device"], str)
            or not isinstance(row["stats"], dict)
        ):
            raise RuntimeError(f"prefill stats reply has invalid rank data: {row!r}")
        by_rank[rank] = row
    expected = set(range(world_size))
    if set(by_rank) != expected:
        raise RuntimeError(
            "prefill stats ranks are incomplete: "
            f"expected={sorted(expected)}, got={sorted(by_rank)}"
        )
    return tuple(by_rank[rank] for rank in range(world_size))


def _verification_stats_row(control_comm, local_runner, *, reset: bool) -> dict[str, object]:
    getter = getattr(local_runner, "verification_execution_stats", None)
    if not callable(getter):
        raise RuntimeError(f"rank {control_comm.rank} runner has no verification_execution_stats")
    stats = getter(reset=reset)
    if not isinstance(stats, dict):
        raise RuntimeError(f"rank {control_comm.rank} verification stats must be a dict")
    device = str(getattr(local_runner, "_device", ""))
    if not device:
        raise RuntimeError(f"rank {control_comm.rank} runner has no compute device")
    return {
        "rank": control_comm.rank,
        "world_size": control_comm.world_size,
        "device": device,
        "stats": stats,
    }


def _verification_stats_packet(control_comm, model_comm, local_runner, *, reset: bool):
    """Serialize one verification-stats row while every rank enters the gather."""
    import torch

    try:
        envelope: dict[str, object] = {
            "ok": True,
            "row": _verification_stats_row(control_comm, local_runner, reset=reset),
        }
        raw = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    except Exception as error:
        envelope = {
            "ok": False,
            "rank": control_comm.rank,
            "error": (f"{type(error).__name__}: {error}")[:1024],
        }
        raw = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    capacity = _VERIFICATION_STATS_PACKET_BYTES - 2
    if len(raw) > capacity:
        raw = json.dumps(
            {
                "ok": False,
                "rank": control_comm.rank,
                "error": (f"serialized verification stats exceed {capacity} bytes"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    device = getattr(model_comm, "_device", None)
    packet = torch.zeros(
        _VERIFICATION_STATS_PACKET_BYTES,
        dtype=torch.uint8,
        device=device,
    )
    packet[0] = len(raw) & 0xFF
    packet[1] = len(raw) >> 8
    packet[2 : 2 + len(raw)].copy_(torch.tensor(tuple(raw), dtype=torch.uint8, device=device))
    return packet


def _decode_verification_stats_packets(
    gathered,
    *,
    world_size: int,
) -> tuple[dict[str, object], ...]:
    """Decode a bounded tensor gather and surface every rank-local error."""
    import torch

    if (
        not isinstance(gathered, torch.Tensor)
        or gathered.dtype != torch.uint8
        or gathered.ndim != 1
        or gathered.numel() != world_size * _VERIFICATION_STATS_PACKET_BYTES
    ):
        raise RuntimeError("verification stats tensor gather has malformed shape or dtype")
    packets = gathered.reshape(world_size, _VERIFICATION_STATS_PACKET_BYTES).cpu()
    rows: list[object] = []
    failures: list[str] = []
    for rank, packet in enumerate(packets):
        length = int(packet[0]) | (int(packet[1]) << 8)
        if not 0 < length <= _VERIFICATION_STATS_PACKET_BYTES - 2:
            failures.append(f"rank {rank}: malformed packet length {length}")
            continue
        try:
            envelope = json.loads(bytes(packet[2 : 2 + length].tolist()))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            failures.append(f"rank {rank}: invalid JSON: {error}")
            continue
        if not isinstance(envelope, dict):
            failures.append(f"rank {rank}: envelope is not an object")
        elif envelope.get("ok") is True:
            rows.append(envelope.get("row"))
        else:
            failures.append(f"rank {rank}: {envelope.get('error', 'unknown error')}")
    if failures:
        raise RuntimeError("verification stats rank failures: " + "; ".join(failures))
    return _validate_verification_stats_rows(rows, world_size=world_size)


def _validate_verification_stats_rows(
    rows: object,
    *,
    world_size: int,
) -> tuple[dict[str, object], ...]:
    if not isinstance(rows, (tuple, list)) or len(rows) != world_size:
        raise RuntimeError(
            "verification stats reply count mismatch: "
            f"expected {world_size}, got "
            f"{len(rows) if isinstance(rows, (tuple, list)) else type(rows).__name__}"
        )
    by_rank: dict[int, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "rank",
            "world_size",
            "device",
            "stats",
        }:
            raise RuntimeError("verification stats reply is malformed")
        rank = row["rank"]
        if (
            type(rank) is not int
            or rank in by_rank
            or row["world_size"] != world_size
            or not isinstance(row["device"], str)
            or not isinstance(row["stats"], dict)
        ):
            raise RuntimeError(f"verification stats reply has invalid rank data: {row!r}")
        by_rank[rank] = row
    expected = set(range(world_size))
    if set(by_rank) != expected:
        raise RuntimeError(
            "verification stats ranks are incomplete: "
            f"expected={sorted(expected)}, got={sorted(by_rank)}"
        )
    return tuple(by_rank[rank] for rank in range(world_size))


def _page_table_stats_row(
    control_comm,
    local_runner,
    *,
    reset: bool,
) -> dict[str, object]:
    getter = getattr(local_runner, "decode_page_table_cache_stats", None)
    if not callable(getter):
        raise RuntimeError(f"rank {control_comm.rank} runner has no decode_page_table_cache_stats")
    stats = getter(reset=reset)
    if not isinstance(stats, dict):
        raise RuntimeError(f"rank {control_comm.rank} decode page-table stats must be a dict")
    device = str(getattr(local_runner, "_device", ""))
    if not device:
        raise RuntimeError(f"rank {control_comm.rank} runner has no compute device")
    return {
        "rank": control_comm.rank,
        "world_size": control_comm.world_size,
        "device": device,
        "stats": stats,
    }


def _page_table_stats_packet(
    control_comm,
    model_comm,
    local_runner,
    *,
    reset: bool,
):
    """Serialize page-table stats while every rank enters the bounded gather."""
    import torch

    try:
        envelope: dict[str, object] = {
            "ok": True,
            "row": _page_table_stats_row(control_comm, local_runner, reset=reset),
        }
        raw = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    except Exception as error:
        envelope = {
            "ok": False,
            "rank": control_comm.rank,
            "error": (f"{type(error).__name__}: {error}")[:1024],
        }
        raw = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    capacity = _PAGE_TABLE_STATS_PACKET_BYTES - 2
    if len(raw) > capacity:
        raw = json.dumps(
            {
                "ok": False,
                "rank": control_comm.rank,
                "error": (f"serialized decode page-table stats exceed {capacity} bytes"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    device = getattr(model_comm, "_device", None)
    packet = torch.zeros(
        _PAGE_TABLE_STATS_PACKET_BYTES,
        dtype=torch.uint8,
        device=device,
    )
    packet[0] = len(raw) & 0xFF
    packet[1] = len(raw) >> 8
    packet[2 : 2 + len(raw)].copy_(torch.tensor(tuple(raw), dtype=torch.uint8, device=device))
    return packet


def _decode_page_table_stats_packets(
    gathered,
    *,
    world_size: int,
) -> tuple[dict[str, object], ...]:
    """Decode page-table stats and reject any missing or failed rank."""
    import torch

    if (
        not isinstance(gathered, torch.Tensor)
        or gathered.dtype != torch.uint8
        or gathered.ndim != 1
        or gathered.numel() != world_size * _PAGE_TABLE_STATS_PACKET_BYTES
    ):
        raise RuntimeError("decode page-table stats tensor gather has malformed shape or dtype")
    packets = gathered.reshape(world_size, _PAGE_TABLE_STATS_PACKET_BYTES).cpu()
    rows: list[object] = []
    failures: list[str] = []
    for rank, packet in enumerate(packets):
        length = int(packet[0]) | (int(packet[1]) << 8)
        if not 0 < length <= _PAGE_TABLE_STATS_PACKET_BYTES - 2:
            failures.append(f"rank {rank}: malformed packet length {length}")
            continue
        try:
            envelope = json.loads(bytes(packet[2 : 2 + length].tolist()))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            failures.append(f"rank {rank}: invalid JSON: {error}")
            continue
        if not isinstance(envelope, dict):
            failures.append(f"rank {rank}: envelope is not an object")
        elif envelope.get("ok") is True:
            rows.append(envelope.get("row"))
        else:
            failures.append(f"rank {rank}: {envelope.get('error', 'unknown error')}")
    if failures:
        raise RuntimeError("decode page-table stats rank failures: " + "; ".join(failures))
    return _validate_page_table_stats_rows(rows, world_size=world_size)


def _validate_page_table_stats_rows(
    rows: object,
    *,
    world_size: int,
) -> tuple[dict[str, object], ...]:
    if not isinstance(rows, (tuple, list)) or len(rows) != world_size:
        raise RuntimeError(
            "decode page-table stats reply count mismatch: "
            f"expected {world_size}, got "
            f"{len(rows) if isinstance(rows, (tuple, list)) else type(rows).__name__}"
        )
    by_rank: dict[int, dict[str, object]] = {}
    expected_fields = {"rank", "world_size", "device", "stats"}
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise RuntimeError("decode page-table stats reply is malformed")
        rank = row["rank"]
        if (
            type(rank) is not int
            or rank in by_rank
            or row["world_size"] != world_size
            or not isinstance(row["device"], str)
            or not isinstance(row["stats"], dict)
        ):
            raise RuntimeError(f"decode page-table stats reply has invalid rank data: {row!r}")
        by_rank[rank] = row
    expected_ranks = set(range(world_size))
    if set(by_rank) != expected_ranks:
        raise RuntimeError(
            "decode page-table stats ranks are incomplete: "
            f"expected={sorted(expected_ranks)}, got={sorted(by_rank)}"
        )
    return tuple(by_rank[rank] for rank in range(world_size))


def _validate_sampling_ownership_rows(
    rows: object,
    *,
    control_world_size: int,
    model_world_size: int,
    control_backend: str,
    model_backend: str,
    request_owner_sampling: bool = False,
) -> tuple[dict[str, object], ...]:
    """Reject incomplete or forged topology evidence with rank-specific errors."""
    if type(request_owner_sampling) is not bool:
        raise TypeError("request_owner_sampling must be a boolean")
    if not isinstance(rows, (tuple, list)):
        raise RuntimeError(
            "sampling ownership replies are malformed: expected a rank sequence, "
            f"got {type(rows).__name__}"
        )
    if len(rows) != control_world_size:
        reported = {
            row.get("rank")
            for row in rows
            if isinstance(row, dict) and type(row.get("rank")) is int
        }
        missing = sorted(set(range(control_world_size)) - reported)
        raise RuntimeError(
            "sampling ownership reply count mismatch: "
            f"expected {control_world_size}, received {len(rows)}; "
            f"missing ranks={missing}"
        )

    by_rank: dict[int, dict[str, object]] = {}
    for slot, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(
                f"sampling ownership reply in gather slot {slot} is malformed: "
                f"expected dict, got {type(row).__name__}"
            )
        fields = set(row)
        if fields != _SAMPLING_OWNERSHIP_FIELDS:
            missing_fields = sorted(_SAMPLING_OWNERSHIP_FIELDS - fields)
            extra_fields = sorted(fields - _SAMPLING_OWNERSHIP_FIELDS)
            raise RuntimeError(
                f"sampling ownership reply in gather slot {slot} has malformed "
                f"fields: missing={missing_fields}, extra={extra_fields}"
            )
        rank = row["rank"]
        if type(rank) is not int:
            raise RuntimeError(
                f"sampling ownership reply in gather slot {slot} has non-integer rank={rank!r}"
            )
        if rank in by_rank:
            raise RuntimeError(
                f"sampling ownership replies contain duplicate rank {rank} (gather slot {slot})"
            )
        if not 0 <= rank < control_world_size:
            raise RuntimeError(
                f"sampling ownership reply in gather slot {slot} has rank {rank}; "
                f"expected [0, {control_world_size})"
            )
        typed_fields = {
            "control_world_size": int,
            "control_backend": str,
            "model_world_size": int,
            "model_backend": str,
            "sampling_owner": bool,
            "sampler_present": bool,
            "device": str,
        }
        for name, field_type in typed_fields.items():
            if type(row[name]) is not field_type:
                raise RuntimeError(
                    f"sampling ownership reply for rank {rank} has malformed "
                    f"{name}={row[name]!r}; expected {field_type.__name__}"
                )
        expected = {
            "control_world_size": control_world_size,
            "control_backend": control_backend,
            "model_world_size": model_world_size,
            "model_backend": model_backend,
        }
        mismatches = {
            name: (row[name], value) for name, value in expected.items() if row[name] != value
        }
        if mismatches:
            raise RuntimeError(
                f"sampling ownership reply for rank {rank} disagrees with rank 0 "
                f"topology (reported, expected)={mismatches}"
            )
        expected_owner = True if request_owner_sampling else rank == 0
        if (
            row["sampling_owner"] is not expected_owner
            or row["sampler_present"] is not expected_owner
        ):
            contract = (
                "all-rank request-owner sampler"
                if request_owner_sampling
                else "rank-0-only sampler"
            )
            raise RuntimeError(
                f"sampling ownership reply for rank {rank} violates the "
                f"{contract} contract: "
                f"sampling_owner={row['sampling_owner']!r}, "
                f"sampler_present={row['sampler_present']!r}, "
                f"expected={expected_owner!r}"
            )
        if not row["device"]:
            raise RuntimeError(f"sampling ownership reply for rank {rank} has an empty device")
        by_rank[rank] = row

    missing = sorted(set(range(control_world_size)) - set(by_rank))
    if missing:
        raise RuntimeError(f"sampling ownership replies are missing ranks={missing}")
    return tuple(by_rank[rank] for rank in sorted(by_rank))


def _validate_sampling_ownership_replies(
    gathered: object,
    *,
    control_world_size: int,
    model_world_size: int,
    control_backend: str,
    model_backend: str,
    request_owner_sampling: bool = False,
) -> tuple[dict[str, object], ...]:
    """Validate all-rank envelopes before exposing sampling topology rows."""

    if not isinstance(gathered, (tuple, list)) or len(gathered) != control_world_size:
        actual = (
            len(gathered)
            if isinstance(gathered, (tuple, list))
            else type(gathered).__name__
        )
        raise RuntimeError(
            "sampling ownership gathered malformed rank count: "
            f"expected {control_world_size}, got {actual}"
        )
    replies: dict[int, _SamplingOwnershipReply] = {}
    for slot, reply in enumerate(gathered):
        if not isinstance(reply, _SamplingOwnershipReply):
            raise RuntimeError(
                f"sampling ownership gather slot {slot} returned malformed "
                f"{type(reply).__name__}"
            )
        if (
            type(reply.rank) is not int
            or not 0 <= reply.rank < control_world_size
            or reply.rank in replies
        ):
            raise RuntimeError(
                f"sampling ownership gather slot {slot} has invalid rank="
                f"{reply.rank!r}"
            )
        if (reply.row is None) == (reply.error is None):
            raise RuntimeError(
                f"sampling ownership rank {reply.rank} reply must contain "
                "exactly one of row or error"
            )
        if reply.error is not None and (
            not isinstance(reply.error, str) or not reply.error
        ):
            raise RuntimeError(
                f"sampling ownership rank {reply.rank} has malformed error"
            )
        replies[reply.rank] = reply
    missing = sorted(set(range(control_world_size)) - set(replies))
    if missing:
        raise RuntimeError(f"sampling ownership ranks are incomplete: missing={missing}")
    failures = tuple(
        f"rank {rank}: {replies[rank].error}"
        for rank in range(control_world_size)
        if replies[rank].error is not None
    )
    if failures:
        raise RuntimeError("sampling ownership rank failures: " + "; ".join(failures))
    return _validate_sampling_ownership_rows(
        tuple(replies[rank].row for rank in range(control_world_size)),
        control_world_size=control_world_size,
        model_world_size=model_world_size,
        control_backend=control_backend,
        model_backend=model_backend,
        request_owner_sampling=request_owner_sampling,
    )


def _config_fingerprint(model_dir: str) -> str:
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:16]


def make_handshake(
    model_dir: str,
    num_pages: int,
    page_size: int,
    attention_backend_identity: str | None = None,
    kv_cache_dtype_requested: str | None = None,
    kv_cache_dtype_resolved: str | None = None,
    dram_kv_tier_identity: str | None = None,
) -> dict:
    """Rank 0 broadcasts this before the step loop; workers validate (A11)."""
    handshake = {
        "num_pages": num_pages,
        "page_size": page_size,
        "config": _config_fingerprint(model_dir),
    }
    if attention_backend_identity is not None:
        handshake["attention_backend_identity"] = attention_backend_identity
    if kv_cache_dtype_requested is not None:
        handshake["kv_cache_dtype_requested"] = kv_cache_dtype_requested
    if kv_cache_dtype_resolved is not None:
        handshake["kv_cache_dtype_resolved"] = kv_cache_dtype_resolved
    if dram_kv_tier_identity is not None:
        handshake["dram_kv_tier_identity"] = dram_kv_tier_identity
    return handshake


def validate_handshake(
    handshake: dict,
    model_dir: str,
    num_pages: int,
    page_size: int,
    attention_backend_identity: str | None = None,
    kv_cache_dtype_requested: str | None = None,
    kv_cache_dtype_resolved: str | None = None,
    dram_kv_tier_identity: str | None = None,
) -> None:
    expected = make_handshake(
        model_dir,
        num_pages,
        page_size,
        attention_backend_identity,
        kv_cache_dtype_requested,
        kv_cache_dtype_resolved,
        dram_kv_tier_identity,
    )
    if handshake != expected:
        raise RuntimeError(
            f"TP worker mismatch: driver={handshake} worker={expected} — "
            "pool sizing/config, backend identity, KV dtype identity, and "
            "DRAM tier identity must be identical on every rank"
        )


_EP_CORRECTNESS_MODE = "replicated-attention-correctness"
_EP_ATTENTION_DP_MODE = "request-owned-attention-dp"
_EP_SUPPORTED_SIZES = frozenset({2, 4})
_EP_ATTENTION_DP_SIZE = 4
_EP_ATTENTION_DP_SCRATCH_PAGES = 2
_EP_ATTENTION_DP_DISPATCHER = "nvfp4_allgather_reduce_scatter"
_EP_ATTENTION_DP_ASSIGNMENT_LOG_LIMIT = 4096
_EP_ATTENTION_DP_ASSIGNMENT_FIELDS = frozenset(
    {"sequence", "request_id", "owner_rank", "cache_affine"}
)
_EP_ATTENTION_DP_PACKET_OK = 0
_EP_ATTENTION_DP_PACKET_FAILED = 1
_EP_METADATA_FILES = (
    "config.json",
    "hf_quant_config.json",
    "model.safetensors.index.json",
)
_EP_OFFICIAL_SHARD_COUNT = 27


def _validate_ep_correctness_mode(
    *,
    expert_parallel_size: object,
    pipeline_depth: object = 1,
    decode_mode: object = "eager",
    kv_cache_dtype: object = "bfloat16",
    pd_separation: object = False,
    graph_scratch_page: object = None,
    graph_max_batch: object = 0,
    graph_max_pages: object = 0,
    graph_warmup_iters: object = 3,
    dram_kv_tier_capacity_pages: object = 0,
    dram_kv_tier_profile: object = None,
    speculative: object = None,
) -> None:
    """Fail closed outside the first production EP correctness envelope.

    This path intentionally replicates QKV projection, attention core, and KV
    state on every rank. NVFP4 attention output is row-parallel over the EP
    communicator, and routed experts retain contiguous rank ownership. It
    establishes correctness for EP2/EP4; it is not the attention-DP path.
    """

    if (
        type(expert_parallel_size) is not int
        or expert_parallel_size not in _EP_SUPPORTED_SIZES
    ):
        supported = ", ".join(str(size) for size in sorted(_EP_SUPPORTED_SIZES))
        raise ValueError(
            "replicated-QKV/KV EP correctness mode supports only "
            f"expert_parallel_size in {{{supported}}}; got {expert_parallel_size!r}"
        )
    if type(pipeline_depth) is not int or pipeline_depth != 1:
        raise ValueError(
            "replicated-QKV/KV EP correctness mode requires pipeline_depth=1"
        )
    if decode_mode != "eager":
        raise ValueError(
            "replicated-QKV/KV EP correctness mode requires decode_mode='eager'; "
            "CUDA graph decode is not supported"
        )
    if kv_cache_dtype != "bfloat16":
        raise ValueError(
            "replicated-QKV/KV EP correctness mode requires "
            "kv_cache_dtype='bfloat16'"
        )
    if type(pd_separation) is not bool:
        raise TypeError("pd_separation must be a boolean")
    if pd_separation:
        raise ValueError(
            "replicated-QKV/KV EP correctness mode does not support P-D separation"
        )
    if graph_scratch_page is not None:
        raise ValueError(
            "replicated-QKV/KV EP correctness mode does not support CUDA graphs"
        )
    if (
        type(dram_kv_tier_capacity_pages) is not int
        or dram_kv_tier_capacity_pages != 0
        or dram_kv_tier_profile is not None
    ):
        raise ValueError(
            "replicated-QKV/KV EP correctness mode does not support a DRAM KV tier"
        )
    if speculative is not None:
        raise ValueError(
            "replicated-QKV/KV EP correctness mode does not support speculative decoding"
        )


def _validate_ep_attention_dp_mode(
    *,
    expert_parallel_size: object,
    pipeline_depth: object = 1,
    decode_mode: object = "eager",
    kv_cache_dtype: object = "bfloat16",
    pd_separation: object = False,
    graph_scratch_page: object = None,
    graph_max_batch: object = 0,
    graph_max_pages: object = 0,
    graph_warmup_iters: object = 3,
    dram_kv_tier_capacity_pages: object = 0,
    dram_kv_tier_profile: object = None,
    speculative: object = None,
) -> None:
    """Fail closed outside the first measured Qwen3-235B attention-DP path."""

    if (
        type(expert_parallel_size) is not int
        or expert_parallel_size != _EP_ATTENTION_DP_SIZE
    ):
        raise ValueError(
            "request-owned attention-DP currently requires "
            f"expert_parallel_size={_EP_ATTENTION_DP_SIZE}; "
            f"got {expert_parallel_size!r}"
        )
    if type(pipeline_depth) is not int or pipeline_depth < 1:
        raise ValueError(
            "request-owned attention-DP requires pipeline_depth to be a "
            "positive integer"
        )
    if decode_mode not in {"eager", "cuda_graph"}:
        raise ValueError(
            "request-owned attention-DP decode_mode must be 'eager' or "
            "'cuda_graph'"
        )
    if kv_cache_dtype != "bfloat16":
        raise ValueError(
            "request-owned attention-DP requires kv_cache_dtype='bfloat16'"
        )
    if type(pd_separation) is not bool:
        raise TypeError("pd_separation must be a boolean")
    if pd_separation:
        raise ValueError(
            "request-owned attention-DP does not support P-D separation"
        )
    if decode_mode == "cuda_graph":
        if type(graph_scratch_page) is not int or graph_scratch_page < 0:
            raise ValueError(
                "request-owned attention-DP CUDA graph decode requires a "
                "reserved graph_scratch_page"
            )
        if type(graph_max_batch) is not int or graph_max_batch < 1:
            raise ValueError(
                "request-owned attention-DP graph_max_batch must be >= 1"
            )
        if type(graph_max_pages) is not int or graph_max_pages < 1:
            raise ValueError(
                "request-owned attention-DP graph_max_pages must be >= 1"
            )
        if type(graph_warmup_iters) is not int or graph_warmup_iters < 0:
            raise ValueError(
                "request-owned attention-DP graph_warmup_iters must be >= 0"
            )
    elif graph_scratch_page is not None:
        raise ValueError(
            "request-owned attention-DP eager decode cannot reserve a CUDA "
            "graph scratch page"
        )
    if (
        type(dram_kv_tier_capacity_pages) is not int
        or dram_kv_tier_capacity_pages != 0
        or dram_kv_tier_profile is not None
    ):
        raise ValueError(
            "request-owned attention-DP does not support a DRAM KV tier"
        )
    if speculative is not None:
        raise ValueError(
            "request-owned attention-DP does not support speculative decoding"
        )


def _validate_attention_dp_batched_prefill_runner(local_runner) -> None:
    """Require the single-forward prefill contract used by row-layout queues."""

    getter = getattr(local_runner, "prefill_execution_stats", None)
    if not callable(getter):
        raise RuntimeError(
            "request-owned attention-DP runner exposes no batched-prefill "
            "capability state"
        )
    stats = getter(reset=False)
    if not isinstance(stats, dict):
        raise RuntimeError(
            "request-owned attention-DP batched-prefill capability state is "
            "malformed"
        )
    if stats.get("enabled") is not True:
        raise RuntimeError(
            "request-owned attention-DP requires batched prefill to remain enabled"
        )
    if stats.get("unified_mixed_enabled") is not False:
        raise RuntimeError(
            "request-owned attention-DP requires mixed steps to remain split"
        )
    gap = stats.get("capability_gap")
    if gap is not None:
        raise RuntimeError(
            "request-owned attention-DP requires native batched prefill: "
            f"{gap}"
        )


def _validate_ep_mode(
    *,
    attention_dp: bool,
    **options: object,
) -> None:
    if type(attention_dp) is not bool:
        raise TypeError("expert_parallel_attention_dp must be a boolean")
    validator = (
        _validate_ep_attention_dp_mode
        if attention_dp
        else _validate_ep_correctness_mode
    )
    validator(**options)


def _ep_checkpoint_identity(model_dir: str) -> dict[str, object]:
    """Hash lightweight metadata and stat, but never read, indexed shards."""

    directory = Path(model_dir).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError(f"EP model path is not a directory: {directory}")
    payloads: dict[str, bytes] = {}
    metadata_sha256: list[tuple[str, str]] = []
    for name in _EP_METADATA_FILES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(
                f"EP checkpoint is missing required metadata file {name!r}"
            )
        payload = path.read_bytes()
        payloads[name] = payload
        metadata_sha256.append((name, hashlib.sha256(payload).hexdigest()))

    try:
        index = json.loads(payloads["model.safetensors.index.json"])
    except (TypeError, ValueError) as error:
        raise ValueError("EP checkpoint safetensors index is not valid JSON") from error
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError("EP checkpoint safetensors index must contain a weight_map object")
    weight_map = index["weight_map"]
    if not weight_map or any(
        not isinstance(tensor_name, str)
        or not tensor_name
        or not isinstance(shard_name, str)
        or not shard_name
        for tensor_name, shard_name in weight_map.items()
    ):
        raise ValueError("EP checkpoint safetensors weight_map is malformed")
    shard_names = sorted(set(weight_map.values()))
    if len(shard_names) != _EP_OFFICIAL_SHARD_COUNT:
        raise ValueError(
            "official Qwen3-235B NVFP4 checkpoint must reference exactly "
            f"{_EP_OFFICIAL_SHARD_COUNT} shards; found {len(shard_names)}"
        )

    shards: list[tuple[str, int]] = []
    for name in shard_names:
        relative = Path(name)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.suffix != ".safetensors"
        ):
            raise ValueError(f"EP checkpoint index has an unsafe shard name {name!r}")
        path = directory / relative
        if not path.is_file():
            raise FileNotFoundError(f"EP checkpoint shard {name!r} is missing")
        resolved = path.resolve(strict=True)
        if resolved.parent != directory:
            raise ValueError(
                f"EP checkpoint shard {name!r} resolves outside the shared model path"
            )
        size = resolved.stat().st_size
        if size < 1:
            raise ValueError(f"EP checkpoint shard {name!r} is empty")
        shards.append((name, size))
    return {
        "resolved_model_path": str(directory),
        "metadata_sha256": tuple(metadata_sha256),
        "indexed_shards": tuple(shards),
    }


def _make_ep_handshake(
    model_dir: str,
    expert_parallel_size: int,
    num_pages: int,
    page_size: int,
    attention_backend_identity: str,
    kv_cache_dtype_resolved: str,
    *,
    attention_dp: bool = False,
    pipeline_depth: int = 1,
    decode_mode: str = "eager",
    graph_max_batch: int = 0,
    graph_max_pages: int = 0,
    graph_warmup_iters: int = 3,
) -> dict[str, object]:
    """Exact all-rank identity for either explicit EP execution mode."""

    _validate_ep_mode(
        attention_dp=attention_dp,
        expert_parallel_size=expert_parallel_size,
        pipeline_depth=pipeline_depth,
        decode_mode=decode_mode,
        graph_scratch_page=(
            num_pages + _EP_ATTENTION_DP_SCRATCH_PAGES + graph_max_batch
            if attention_dp and decode_mode == "cuda_graph"
            else None
        ),
        graph_max_batch=graph_max_batch,
        graph_max_pages=graph_max_pages,
        graph_warmup_iters=graph_warmup_iters,
    )
    handshake = make_handshake(
        model_dir,
        num_pages,
        page_size,
        attention_backend_identity=attention_backend_identity,
        kv_cache_dtype_requested="bfloat16",
        kv_cache_dtype_resolved=kv_cache_dtype_resolved,
    )
    if attention_dp:
        graph_decode = decode_mode == "cuda_graph"
        decode_scratch_pages = graph_max_batch if graph_decode else 1
        graph_scratch_pages = int(graph_decode)
        scratch_pages = (
            _EP_ATTENTION_DP_SCRATCH_PAGES
            + decode_scratch_pages
            + graph_scratch_pages
        )
        handshake.update(
            {
                "parallelism": "expert_parallel",
                "expert_parallel_size": expert_parallel_size,
                "attention_placement": "request_owned_data_parallel",
                "attention_data_parallel_size": expert_parallel_size,
                "attention_tensor_parallel_size": 1,
                "attention_output_placement": "replicated",
                "attention_output_parallel_size": 1,
                "attention_output_partial_dtype": None,
                "moe_dispatcher": _EP_ATTENTION_DP_DISPATCHER,
                "sampling_ownership": "request_owner",
                "kv_cache_ownership": "request_owner",
                "execution_mode": _EP_ATTENTION_DP_MODE,
                "decode_mode": decode_mode,
                "cuda_graph_decode": graph_decode,
                "cuda_graph_max_batch": graph_max_batch if graph_decode else 0,
                "cuda_graph_max_pages": graph_max_pages if graph_decode else 0,
                "cuda_graph_warmup_iters": (
                    graph_warmup_iters if graph_decode else 0
                ),
                "pipeline_depth": pipeline_depth,
                "usable_kv_pages": num_pages,
                "physical_kv_pages": num_pages + scratch_pages,
                "attention_dp_prefill_scratch_pages": (
                    _EP_ATTENTION_DP_SCRATCH_PAGES
                ),
                "attention_dp_decode_scratch_pages": decode_scratch_pages,
                "attention_dp_graph_scratch_pages": graph_scratch_pages,
                "attention_dp_scratch_pages": scratch_pages,
                "checkpoint_identity": _ep_checkpoint_identity(model_dir),
            }
        )
    else:
        handshake.update(
            {
                "parallelism": "expert_parallel",
                "expert_parallel_size": expert_parallel_size,
                "attention_placement": "replicated",
                "attention_output_placement": "row_parallel",
                "attention_output_parallel_size": expert_parallel_size,
                "attention_output_partial_dtype": "bfloat16",
                "execution_mode": _EP_CORRECTNESS_MODE,
                "decode_mode": "eager",
                "pipeline_depth": 1,
                "checkpoint_identity": _ep_checkpoint_identity(model_dir),
            }
        )
    return handshake


def _validate_ep_handshake(
    handshake: dict[str, object],
    model_dir: str,
    expert_parallel_size: int,
    num_pages: int,
    page_size: int,
    attention_backend_identity: str,
    kv_cache_dtype_resolved: str,
    *,
    attention_dp: bool = False,
    pipeline_depth: int = 1,
    decode_mode: str = "eager",
    graph_max_batch: int = 0,
    graph_max_pages: int = 0,
    graph_warmup_iters: int = 3,
) -> None:
    expected = _make_ep_handshake(
        model_dir,
        expert_parallel_size,
        num_pages,
        page_size,
        attention_backend_identity,
        kv_cache_dtype_resolved,
        attention_dp=attention_dp,
        pipeline_depth=pipeline_depth,
        decode_mode=decode_mode,
        graph_max_batch=graph_max_batch,
        graph_max_pages=graph_max_pages,
        graph_warmup_iters=graph_warmup_iters,
    )
    if handshake != expected:
        raise RuntimeError(
            f"EP worker mismatch: driver={handshake} worker={expected} — "
            "expert topology, attention/KV ownership, MoE dispatcher, eager "
            "decode/graph execution, BF16 KV, pool sizing/config, and backend identity "
            "must match on every rank"
        )


def _release_runner_requests(local_runner, request_ids: tuple[str, ...]) -> None:
    """Apply one ordered release batch to a rank-local runner."""
    release = getattr(local_runner, "release", None)
    if release is None:
        return
    for request_id in request_ids:
        release(request_id)


def _dram_kv_operation(payload: _DramKVControl) -> str:
    if isinstance(payload, _DramKVAvailablePrefix):
        return "available_prefix"
    if isinstance(payload, _DramKVOffload):
        return "offload"
    if isinstance(payload, _DramKVRestore):
        return "restore"
    if isinstance(payload, _DramKVDiscard):
        return "discard"
    raise TypeError(f"unsupported DRAM KV control payload: {type(payload).__name__}")


def _dram_kv_local_ack(control_comm, local_runner, payload: _DramKVControl) -> _DramKVAck:
    """Run one blocking rank-local operation and always produce a gather ACK.

    Rank-local exceptions cannot escape before the all-gather: doing so would
    strand the other TP ranks in the collective.  Ownership flags are copied
    by duck typing so this seam stays independent of a particular tier class.
    Missing tiers are an explicit safe miss, which lets a mixed rollout fall
    back to recompute instead of publishing a partial multi-rank prefix.
    """
    operation = _dram_kv_operation(payload)
    tier = getattr(local_runner, "dram_kv_tier", None)
    if tier is None:
        value: object = 0 if operation == "available_prefix" else True
        if operation in {"offload", "restore"}:
            value = False
        return _DramKVAck(
            rank=control_comm.rank,
            operation=operation,
            ok=True,
            value=value,
            source_reusable=True,
            destination_reusable=True,
        )

    try:
        if isinstance(payload, _DramKVAvailablePrefix):
            value = tier.available_prefix(payload.keys, min_pages=payload.min_pages)
        elif isinstance(payload, _DramKVOffload):
            value = tier.offload(payload.keys, payload.page_ids)
        elif isinstance(payload, _DramKVRestore):
            value = tier.restore(payload.keys, payload.page_ids)
        else:
            value = tier.discard(payload.keys)
        return _DramKVAck(
            rank=control_comm.rank,
            operation=operation,
            ok=True,
            value=value,
            source_reusable=True,
            destination_reusable=True,
            destination_publishable=operation == "restore" and value is True,
        )
    except BaseException as error:
        return _DramKVAck(
            rank=control_comm.rank,
            operation=operation,
            ok=False,
            error=f"{type(error).__name__}: {error}"[:1024],
            base_exception=not isinstance(error, Exception),
            source_reusable=getattr(error, "source_reusable", None) is True,
            destination_reusable=getattr(error, "destination_reusable", None) is True,
            destination_publishable=(
                getattr(error, "destination_publishable", None) is True
            ),
        )


def _validate_dram_kv_acks(
    gathered,
    *,
    operation: str,
    world_size: int,
) -> tuple[_DramKVAck, ...]:
    if not isinstance(gathered, (tuple, list)) or len(gathered) != world_size:
        actual = (
            len(gathered)
            if isinstance(gathered, (tuple, list))
            else type(gathered).__name__
        )
        raise RuntimeError(
            f"DRAM KV {operation} gathered malformed rank count: "
            f"expected {world_size}, got {actual}"
        )
    rows: list[_DramKVAck] = []
    for rank, row in enumerate(gathered):
        if not isinstance(row, _DramKVAck):
            raise RuntimeError(
                f"DRAM KV {operation} rank {rank} returned malformed ACK "
                f"{type(row).__name__}"
            )
        if row.rank != rank or row.operation != operation:
            raise RuntimeError(
                f"DRAM KV {operation} ACK identity mismatch at rank {rank}: "
                f"rank={row.rank!r}, operation={row.operation!r}"
            )
        rows.append(row)
    return tuple(rows)


def _dram_kv_failure_summary(rows: tuple[_DramKVAck, ...]) -> str:
    failures: list[str] = []
    for row in rows:
        if row.ok:
            failures.append(f"rank {row.rank}: returned {row.value!r}")
        else:
            failures.append(f"rank {row.rank}: {row.error or 'unknown error'}")
    return "; ".join(failures)


def _raise_dram_kv_base_exceptions(
    rows: tuple[_DramKVAck, ...],
    *,
    operation: str,
) -> None:
    fatal = tuple(row for row in rows if row.base_exception)
    if fatal:
        raise DramKVTierTransactionError(
            f"TP DRAM KV {operation} intercepted rank-local process control: "
            f"{_dram_kv_failure_summary(fatal)}"
        )


def _validate_dram_kv_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(keys, tuple):
        raise TypeError("DRAM KV keys must be a tuple")
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("DRAM KV keys must be non-empty strings")
    return keys


def _validate_dram_kv_pages(
    keys: tuple[str, ...],
    page_ids: tuple[int, ...],
) -> tuple[int, ...]:
    if not isinstance(page_ids, tuple):
        raise TypeError("DRAM KV page ids must be a tuple")
    if len(page_ids) != len(keys):
        raise ValueError(
            "DRAM KV key/page cardinality mismatch: "
            f"{len(keys)} keys for {len(page_ids)} pages"
        )
    if any(type(page_id) is not int or page_id < 0 for page_id in page_ids):
        raise ValueError("DRAM KV page ids must be non-negative integers")
    if len(set(page_ids)) != len(page_ids):
        raise ValueError("DRAM KV page ids must be unique")
    return page_ids


def _serialized_protocol(method):
    """Hold one rank-0 transaction across all of its ordered collectives."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._protocol_lock:
            return method(self, *args, **kwargs)

    return wrapped


class DistTPModelRunner:
    """Driver-side ModelRunner with rank-0-authoritative sampling.

    Drops in where ``TPModelRunner`` sits: the driver's own rank-0 shard runs
    inside this call and samples once. Non-zero ranks execute the passive
    model/KV path, then all ranks adopt rank 0's device token packet.
    """

    def __init__(
        self,
        control_comm,
        local_runner,
        model_comm=None,
        *,
        parallelism_display: str = "tensor-parallel",
        parallelism_prefix: str = "TP",
    ) -> None:
        self._control_comm = control_comm
        self._model_comm = model_comm if model_comm is not None else control_comm
        self._local = local_runner
        self._parallelism_display = parallelism_display
        self._parallelism_prefix = parallelism_prefix
        self._fatal_error: Exception | None = None
        # delta-broadcast state (F4): only new/finished requests + committed
        # tokens cross the wire each step, not a full pickled snapshot of every
        # active request's (growing) prompt/outputs
        self._sync = StateSync()
        # A control broadcast is only the first operation in a TP transaction:
        # execute continues through model collectives and packet adoption.  A
        # concurrent release/shutdown must not insert another control payload
        # before that transaction is complete.
        self._protocol_lock = Lock()
        # ``release`` is called by the EngineLoop thread while a later device
        # step may already be running. Queueing is the only safe non-blocking
        # operation here; the next device transaction snapshots the ordered,
        # deduped batch and carries it in its existing StepDelta broadcast.
        self._pending_release_lock = Lock()
        self._pending_releases: dict[str, int] = {}
        self._release_generation = 0
        self._shutdown_started = False
        self._ep_kv_accounting = (
            _EPKVAccountingRecorder()
            if parallelism_prefix == "EP"
            else None
        )

    def _broadcast_control(self, payload: object) -> object:
        """Broadcast one rank-0 control payload."""

        return self._control_comm.broadcast(payload, src=0)

    def _snapshot_pending_releases(self) -> tuple[tuple[str, int], ...]:
        with self._pending_release_lock:
            if self._shutdown_started:
                raise RuntimeError(f"{self._parallelism_display} runner is shutting down")
            return tuple(self._pending_releases.items())

    def _ack_pending_releases(
        self,
        released: tuple[tuple[str, int], ...],
    ) -> None:
        """Remove only the exact release generations applied by this transaction."""
        with self._pending_release_lock:
            for request_id, generation in released:
                if self._pending_releases.get(request_id) == generation:
                    del self._pending_releases[request_id]

    @property
    def supports_batched_verification(self) -> bool:
        """Advertise the capability only when the rank-local runner has it."""
        return getattr(self._local, "supports_batched_verification", False) is True

    def execute(self, scheduled, states) -> dict:
        with self._protocol_lock:
            if self._fatal_error is not None:
                raise RuntimeError(
                    f"{self._parallelism_display} runner is unavailable after a fatal "
                    "step failure"
                ) from self._fatal_error
            chunks = tuple(scheduled)
            pending_releases = self._snapshot_pending_releases()
            released_ids = tuple(request_id for request_id, _ in pending_releases)
            try:
                delta = self._sync.diff(
                    chunks,
                    states,
                    released_ids=released_ids,
                )
                self._broadcast_control(delta)
                # Match the worker ordering: stale per-request local state and
                # StateSync identity are gone before a new snapshot with the
                # same request id is installed or executed.
                _release_runner_requests(self._local, released_ids)
                view = self._sync.apply(delta)  # reconstructs snapshot_step() exactly
                # Keep pending entries durable across a failed broadcast/local
                # cleanup. Acknowledge only after both rank protocols have
                # accepted the batch and rank 0 has applied it.
                self._ack_pending_releases(pending_releases)
                if self._ep_kv_accounting is not None:
                    self._ep_kv_accounting.observe(chunks, view)
                sampled = self._local.execute(chunks, view)
                packet = self._local.make_sampling_token_packet(chunks, view, sampled=sampled)
                packet = self._model_comm.tensor_broadcast(packet, src=0)
                self._local.adopt_sampling_token_packet(chunks, view, packet)
                return sampled
            except Exception as error:
                # Once one TP rank misses a step, retrying on the same groups can only
                # diverge their collective sequences further. Surface a fatal health
                # state and require process replacement.
                self._fatal_error = error
                raise

    @property
    def fatal_error(self) -> Exception | None:
        return self._fatal_error

    @_serialized_protocol
    def capture_decode_graphs(
        self,
        *,
        attention_dp: bool = False,
    ) -> tuple[int, ...]:
        """Warm every rank before the launcher can publish readiness."""

        if type(attention_dp) is not bool:
            raise TypeError("attention_dp graph warmup flag must be bool")
        if self._fatal_error is not None:
            raise RuntimeError(
                f"{self._parallelism_display} runner is unavailable after a fatal "
                "step failure"
            ) from self._fatal_error
        try:
            payload = _CaptureDecodeGraphs(attention_dp=attention_dp)
            delivered = self._broadcast_control(payload)
            if delivered != payload:
                raise RuntimeError(
                    "decode-graph startup broadcast returned a malformed payload"
                )
            return _capture_decode_graphs_transaction(
                self._control_comm,
                self._local,
                attention_dp=attention_dp,
            )
        except Exception as error:
            self._fatal_error = error
            raise

    def decode_graph_metadata(self) -> dict[str, object]:
        """Read rank 0 counters without a collective or protocol-lock wait.

        The in-process runner exposes the same lock-free diagnostic snapshot.
        Startup capture guarantees the bucket set no longer mutates while
        serving; integer dispatch counters are safe to sample between Python
        bytecodes.  Taking ``_protocol_lock`` here would let a metrics scrape or
        process-wire event wait behind later pipelined inference and delay TTFT.
        """

        getter = getattr(self._local, "decode_graph_metadata", None)
        if not callable(getter):
            raise RuntimeError(
                f"{self._parallelism_display} runner exposes no graph metadata"
            )
        metadata = getter()
        if not isinstance(metadata, Mapping):
            raise RuntimeError(
                f"{self._parallelism_display} runner returned malformed graph metadata"
            )
        return dict(metadata)

    def _dram_kv_collect(self, payload: _DramKVControl) -> tuple[_DramKVAck, ...]:
        """Run one already-serialized all-rank tier transaction."""
        if self._fatal_error is not None:
            raise RuntimeError(
                f"{self._parallelism_display} runner is unavailable after a fatal "
                "step failure"
            ) from self._fatal_error
        operation = _dram_kv_operation(payload)
        try:
            delivered = self._broadcast_control(payload)
            if delivered != payload:
                raise RuntimeError(
                    f"DRAM KV {operation} broadcast returned a malformed payload"
                )
            local = _dram_kv_local_ack(self._control_comm, self._local, payload)
            gathered = self._control_comm.all_gather(local)
            return _validate_dram_kv_acks(
                gathered,
                operation=operation,
                world_size=self._control_comm.world_size,
            )
        except Exception as error:
            # A transport/shape failure can leave ranks on different collective
            # rounds. Local tier failures are encoded in ACKs and never enter
            # this branch, so only protocol failures poison the TP runner.
            failure = RuntimeError(
                f"{self._parallelism_prefix} DRAM KV {operation} protocol failed: {error}"
            )
            self._fatal_error = failure
            raise failure from error

    def _dram_kv_discard_locked(self, keys: tuple[str, ...]) -> None:
        rows = self._dram_kv_collect(_DramKVDiscard(keys))
        _raise_dram_kv_base_exceptions(rows, operation="discard")
        failed = tuple(row for row in rows if not row.ok or row.value is not True)
        if failed:
            raise DramKVTierTransactionError(
                f"{self._parallelism_prefix} DRAM KV discard could not converge "
                "every rank: "
                f"{_dram_kv_failure_summary(failed)}"
            )

    @_serialized_protocol
    def available_prefix(self, keys: tuple[str, ...], *, min_pages: int = 1) -> int:
        """Return only a DRAM prefix that every TP rank can restore."""
        keys = _validate_dram_kv_keys(keys)
        if type(min_pages) is not int or min_pages < 1:
            raise ValueError("DRAM KV min_pages must be a positive integer")
        if not keys:
            return 0
        rows = self._dram_kv_collect(_DramKVAvailablePrefix(keys, min_pages))
        _raise_dram_kv_base_exceptions(rows, operation="available_prefix")
        values = tuple(row.value for row in rows if row.ok and type(row.value) is int)
        agreed = (
            len(values) == len(rows)
            and len(set(values)) == 1
            and 0 <= values[0] <= len(keys)
        )
        if agreed:
            return values[0] if values[0] >= min_pages else 0
        self._dram_kv_discard_locked(keys)
        return 0

    @_serialized_protocol
    def offload(self, keys: tuple[str, ...], page_ids: tuple[int, ...]) -> bool:
        """Offload all rank shards before allowing source-page reuse."""
        keys = _validate_dram_kv_keys(keys)
        page_ids = _validate_dram_kv_pages(keys, page_ids)
        if not keys:
            return True
        rows = self._dram_kv_collect(_DramKVOffload(keys, page_ids))
        _raise_dram_kv_base_exceptions(rows, operation="offload")
        if all(row.ok and row.value is True for row in rows):
            return True

        unsafe = tuple(
            row
            for row in rows
            if (not row.ok and not row.source_reusable)
            or (row.ok and row.value is not True and row.value is not False)
        )
        if unsafe:
            raise DramKVTierTransactionError(
                f"{self._parallelism_prefix} DRAM KV offload left source ownership "
                "unknown: "
                f"{_dram_kv_failure_summary(unsafe)}",
                source_reusable=False,
            )
        self._dram_kv_discard_locked(keys)
        return False

    @_serialized_protocol
    def restore(self, keys: tuple[str, ...], page_ids: tuple[int, ...]) -> bool:
        """Publish success only after every rank completed its blocking restore."""
        keys = _validate_dram_kv_keys(keys)
        page_ids = _validate_dram_kv_pages(keys, page_ids)
        if not keys:
            return True
        rows = self._dram_kv_collect(_DramKVRestore(keys, page_ids))
        _raise_dram_kv_base_exceptions(rows, operation="restore")
        if all(
            row.ok and row.value is True and row.destination_publishable
            for row in rows
        ):
            return True

        unsafe = tuple(
            row
            for row in rows
            if (not row.ok and not row.destination_reusable)
            or (row.ok and row.value is not True and row.value is not False)
        )
        if unsafe:
            raise DramKVTierTransactionError(
                f"{self._parallelism_prefix} DRAM KV restore left destination "
                "ownership unknown: "
                f"{_dram_kv_failure_summary(unsafe)}",
                destination_reusable=False,
                destination_publishable=False,
            )
        self._dram_kv_discard_locked(keys)
        return False

    @_serialized_protocol
    def discard(self, keys: tuple[str, ...]) -> bool:
        """Discard host copies on every rank as one serialized transaction."""
        keys = _validate_dram_kv_keys(keys)
        if not keys:
            return True
        self._dram_kv_discard_locked(keys)
        return True

    @_serialized_protocol
    def sampling_ownership_metadata(self) -> tuple[dict[str, object], ...]:
        """Collect rank-observed ownership/topology outside the inference path.

        This explicit diagnostic is the only place the metadata all-gather runs;
        normal steps retain only their state and sampled-token broadcasts.
        """
        if self._fatal_error is not None:
            raise RuntimeError(
                f"{self._parallelism_display} runner is unavailable after a fatal "
                "step failure"
            ) from self._fatal_error
        try:
            control_world_size = self._control_comm.world_size
            model_world_size = self._model_comm.world_size
            control_backend = _communicator_backend(self._control_comm)
            model_backend = _communicator_backend(self._model_comm)
            delivered = self._broadcast_control(_SamplingOwnershipProbe())
            if not isinstance(delivered, _SamplingOwnershipProbe):
                raise RuntimeError(
                    "sampling ownership probe broadcast returned a malformed "
                    f"payload: {type(delivered).__name__}"
                )
            local = _sampling_ownership_reply(
                self._control_comm,
                self._model_comm,
                self._local,
            )
            rows = self._control_comm.all_gather(local)
            return _validate_sampling_ownership_replies(
                rows,
                control_world_size=control_world_size,
                model_world_size=model_world_size,
                control_backend=control_backend,
                model_backend=model_backend,
            )
        except Exception as error:
            failure = RuntimeError(
                f"{self._parallelism_prefix} sampling ownership metadata probe "
                f"failed: {error}"
            )
            self._fatal_error = failure
            raise failure from error

    @_serialized_protocol
    def set_batched_prefill_enabled(self, enabled: bool) -> None:
        """Apply the matched-A/B/rollback switch to every TP rank."""
        if type(enabled) is not bool:
            raise TypeError("batched prefill enabled flag must be bool")
        if self._fatal_error is not None:
            raise RuntimeError(
                f"{self._parallelism_display} runner is unavailable after a fatal "
                "step failure"
            ) from self._fatal_error
        try:
            payload = _BatchedPrefillMode(enabled)
            delivered = self._broadcast_control(payload)
            if delivered != payload:
                raise RuntimeError("batched prefill mode broadcast returned a malformed payload")
            self._local.set_batched_prefill_enabled(enabled)
        except Exception as error:
            self._fatal_error = error
            raise

    @_serialized_protocol
    def prefill_execution_stats(self, *, reset: bool = False) -> tuple[dict[str, object], ...]:
        """Gather structural counters from every TP rank outside inference."""
        if type(reset) is not bool:
            raise TypeError("prefill stats reset flag must be bool")
        if self._fatal_error is not None:
            raise RuntimeError(
                f"{self._parallelism_display} runner is unavailable after a fatal "
                "step failure"
            ) from self._fatal_error
        try:
            probe = _PrefillStatsProbe(reset)
            delivered = self._broadcast_control(probe)
            if delivered != probe:
                raise RuntimeError("prefill stats probe broadcast returned a malformed payload")
            packet = _prefill_stats_packet(
                self._control_comm,
                self._model_comm,
                self._local,
                reset=reset,
            )
            gathered = self._model_comm.tensor_all_gather(packet)
            return _decode_prefill_stats_packets(
                gathered,
                world_size=self._control_comm.world_size,
            )
        except Exception as error:
            failure = RuntimeError(
                f"{self._parallelism_prefix} prefill stats probe failed: {error}"
            )
            self._fatal_error = failure
            raise failure from error

    @_serialized_protocol
    def set_batched_verification_enabled(self, enabled: bool) -> None:
        """Apply the speculative-verification mode switch to every TP rank."""
        if type(enabled) is not bool:
            raise TypeError("batched verification enabled flag must be bool")
        if self._fatal_error is not None:
            raise RuntimeError(
                f"{self._parallelism_display} runner is unavailable after a fatal "
                "step failure"
            ) from self._fatal_error
        try:
            payload = _BatchedVerificationMode(enabled)
            delivered = self._broadcast_control(payload)
            if delivered != payload:
                raise RuntimeError(
                    "batched verification mode broadcast returned a malformed payload"
                )
            self._local.set_batched_verification_enabled(enabled)
        except Exception as error:
            self._fatal_error = error
            raise

    @_serialized_protocol
    def verification_execution_stats(
        self,
        *,
        reset: bool = False,
    ) -> tuple[dict[str, object], ...]:
        """Gather speculative-verification counters from every TP rank."""
        if type(reset) is not bool:
            raise TypeError("verification stats reset flag must be bool")
        if self._fatal_error is not None:
            raise RuntimeError(
                f"{self._parallelism_display} runner is unavailable after a fatal "
                "step failure"
            ) from self._fatal_error
        try:
            probe = _VerificationStatsProbe(reset)
            delivered = self._broadcast_control(probe)
            if delivered != probe:
                raise RuntimeError(
                    "verification stats probe broadcast returned a malformed payload"
                )
            packet = _verification_stats_packet(
                self._control_comm,
                self._model_comm,
                self._local,
                reset=reset,
            )
            gathered = self._model_comm.tensor_all_gather(packet)
            return _decode_verification_stats_packets(
                gathered,
                world_size=self._control_comm.world_size,
            )
        except Exception as error:
            failure = RuntimeError(
                f"{self._parallelism_prefix} verification stats probe failed: {error}"
            )
            self._fatal_error = failure
            raise failure from error

    @_serialized_protocol
    def set_decode_page_table_cache_enabled(self, enabled: bool) -> None:
        """Apply the decode page-table cache mode to every TP rank."""
        if type(enabled) is not bool:
            raise TypeError("decode page-table cache enabled flag must be bool")
        if self._fatal_error is not None:
            raise RuntimeError(
                f"{self._parallelism_display} runner is unavailable after a fatal "
                "step failure"
            ) from self._fatal_error
        try:
            payload = _DecodePageTableCacheMode(enabled)
            delivered = self._broadcast_control(payload)
            if delivered != payload:
                raise RuntimeError(
                    "decode page-table cache mode broadcast returned a malformed payload"
                )
            self._local.set_decode_page_table_cache_enabled(enabled)
        except Exception as error:
            self._fatal_error = error
            raise

    @_serialized_protocol
    def decode_page_table_cache_stats(
        self,
        *,
        reset: bool = False,
    ) -> tuple[dict[str, object], ...]:
        """Gather decode page-table counters from every rank."""
        if type(reset) is not bool:
            raise TypeError("decode page-table cache stats reset flag must be bool")
        if self._fatal_error is not None:
            raise RuntimeError(
                f"{self._parallelism_display} runner is unavailable after a fatal "
                "step failure"
            ) from self._fatal_error
        try:
            probe = _DecodePageTableCacheStatsProbe(reset)
            delivered = self._broadcast_control(probe)
            if delivered != probe:
                raise RuntimeError(
                    "decode page-table stats probe broadcast returned a malformed payload"
                )
            packet = _page_table_stats_packet(
                self._control_comm,
                self._model_comm,
                self._local,
                reset=reset,
            )
            gathered = self._model_comm.tensor_all_gather(packet)
            return _decode_page_table_stats_packets(
                gathered,
                world_size=self._control_comm.world_size,
            )
        except Exception as error:
            failure = RuntimeError(
                f"{self._parallelism_prefix} decode page-table stats probe failed: {error}"
            )
            self._fatal_error = failure
            raise failure from error

    def release(self, request_id: str) -> None:
        """Queue cleanup for the next all-rank transaction without blocking it."""
        with self._pending_release_lock:
            if self._shutdown_started:
                raise RuntimeError(f"{self._parallelism_display} runner is shutting down")
            self._release_generation += 1
            # Reinsert duplicates at their newest position. The generation
            # prevents an execute transaction from acknowledging a same-id
            # release that arrived after its snapshot.
            self._pending_releases.pop(request_id, None)
            self._pending_releases[request_id] = self._release_generation

    def shutdown(self) -> None:
        with self._protocol_lock:
            with self._pending_release_lock:
                if self._shutdown_started:
                    return
                self._shutdown_started = True
                pending_releases = tuple(self._pending_releases.items())
            released_ids = tuple(request_id for request_id, _ in pending_releases)
            self._sync.discard(released_ids)
            try:
                if self._fatal_error is None:
                    self._broadcast_control(_ShutdownRequest(released_ids))
            finally:
                _release_runner_requests(self._local, released_ids)
                self._ack_pending_releases(pending_releases)

    def invalidate_graphs(self) -> None:
        invalidate = getattr(self._local, "invalidate_graphs", None)
        if invalidate is not None:
            invalidate()


_EP_ATTENTION_DP_SCRATCH_PREFIX = "__kairyu_attention_dp_scratch__"


def _ep_attention_dp_owner_map(
    owners: tuple[tuple[str, int], ...],
    chunks: tuple[ScheduledChunk, ...],
    *,
    world_size: int,
) -> dict[str, int]:
    if not isinstance(owners, tuple):
        raise RuntimeError("attention-DP owners must be a tuple")
    result: dict[str, int] = {}
    for item in owners:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or type(item[1]) is not int
            or not 0 <= item[1] < world_size
        ):
            raise RuntimeError(f"attention-DP owner entry is malformed: {item!r}")
        request_id, rank = item
        if request_id.startswith(_EP_ATTENTION_DP_SCRATCH_PREFIX):
            raise RuntimeError(
                "request id uses the reserved attention-DP scratch namespace"
            )
        if request_id in result:
            raise RuntimeError(
                f"attention-DP owners contain duplicate request {request_id!r}"
            )
        result[request_id] = rank
    expected = tuple(dict.fromkeys(chunk.request_id for chunk in chunks))
    if tuple(result) != expected:
        raise RuntimeError(
            "attention-DP owners do not match scheduled request order: "
            f"expected={expected!r}, got={tuple(result)!r}"
        )
    return result


def _ep_attention_dp_dummy_state(
    request_id: str,
    *,
    page_id: int,
    prefill: bool,
) -> RequestSnapshot:
    prompt = (0, 0) if prefill else (0,)
    return RequestSnapshot(
        request_id=request_id,
        prompt_token_ids=prompt,
        computed_prompt=1,
        outputs=(),
        in_flight=0,
        page_ids=(page_id,),
        decode_page_ids=(),
        eos_token_id=None,
        max_new_tokens=1,
        num_cached_tokens=0,
        sampling=EngineSampling(),
    )


def _ep_attention_dp_rank_chunks(
    chunks: tuple[ScheduledChunk, ...],
    owner_map: dict[str, int],
    rank: int,
) -> tuple[ScheduledChunk, ...]:
    return tuple(chunk for chunk in chunks if owner_map[chunk.request_id] == rank)


def _ep_attention_dp_decode_shape(
    chunks: tuple[ScheduledChunk, ...],
    owners: tuple[tuple[str, int], ...],
    states: Mapping[str, RequestSnapshot],
    *,
    world_size: int,
) -> tuple[int, int] | None:
    """Return global ``(max rows/rank, max page width)`` for decode only."""

    decodes = tuple(chunk for chunk in chunks if not chunk.is_prefill)
    if not decodes:
        return None
    if any(chunk.num_tokens != 1 for chunk in decodes):
        raise RuntimeError(
            "attention-DP does not support multi-token verification chunks"
        )
    owner_map = _ep_attention_dp_owner_map(
        owners,
        chunks,
        world_size=world_size,
    )
    counts = [0] * world_size
    max_pages = 0
    for chunk in decodes:
        counts[owner_map[chunk.request_id]] += 1
        state = states[chunk.request_id]
        allocation = getattr(state, "allocation", None)
        if allocation is None:
            raise RuntimeError(
                f"decode request {chunk.request_id!r} has no KV allocation"
            )
        width = len(tuple(allocation.pages)) + len(tuple(state.decode_pages))
        if width < 1:
            raise RuntimeError(
                f"decode request {chunk.request_id!r} has an empty KV page table"
            )
        max_pages = max(max_pages, width)
    return max(counts), max_pages


def _ep_attention_dp_local_graph_decision(
    local_runner,
    *,
    cuda_graph_decode: bool,
    batch_size: int,
    max_pages: int,
) -> GraphDecodeDecision:
    getter = getattr(local_runner, "decode_graph_decision_for_shape", None)
    if callable(getter):
        decision = getter(batch_size=batch_size, max_pages=max_pages)
    elif cuda_graph_decode:
        raise RuntimeError(
            "attention-DP CUDA graph runner has no prospective decision API"
        )
    else:
        decision = GraphDecodeDecision(
            kind="eager_fallback",
            bucket_size=batch_size,
            capture_model_forward_count=1,
        )
    if not isinstance(decision, GraphDecodeDecision):
        raise RuntimeError(
            "attention-DP prospective graph decision has a malformed type"
        )
    return decision


def _ep_attention_dp_make_graph_plan(
    local_runner,
    chunks: tuple[ScheduledChunk, ...],
    owners: tuple[tuple[str, int], ...],
    states: Mapping[str, RequestSnapshot],
    *,
    world_size: int,
    cuda_graph_decode: bool,
) -> _EPAttentionDPGraphPlan | None:
    shape = _ep_attention_dp_decode_shape(
        chunks,
        owners,
        states,
        world_size=world_size,
    )
    if shape is None:
        return None
    batch_size, max_pages = shape
    decision = _ep_attention_dp_local_graph_decision(
        local_runner,
        cuda_graph_decode=cuda_graph_decode,
        batch_size=batch_size,
        max_pages=max_pages,
    )
    if not cuda_graph_decode and decision.kind != "eager_fallback":
        raise RuntimeError("eager attention-DP produced a CUDA graph decision")
    return _EPAttentionDPGraphPlan(
        cuda_graph_decode=cuda_graph_decode,
        batch_size=batch_size,
        max_pages=max_pages,
        decision=decision,
    )


def _ep_attention_dp_validate_graph_plan(
    local_runner,
    payload: _EPAttentionDPStep,
    states: Mapping[str, RequestSnapshot],
    *,
    world_size: int,
) -> None:
    """Validate payload geometry and rank-local graph state without CUDA work."""

    expected_shape = _ep_attention_dp_decode_shape(
        payload.delta.chunks,
        payload.owners,
        states,
        world_size=world_size,
    )
    plan = payload.graph_plan
    if expected_shape is None:
        if plan is not None:
            raise RuntimeError("attention-DP prefill-only step has a graph plan")
        return
    if not isinstance(plan, _EPAttentionDPGraphPlan):
        raise RuntimeError("attention-DP decode step is missing its graph plan")
    if (plan.batch_size, plan.max_pages) != expected_shape:
        raise RuntimeError(
            "attention-DP graph plan shape disagrees with request ownership/KV "
            f"state: payload={(plan.batch_size, plan.max_pages)!r}, "
            f"local={expected_shape!r}"
        )
    if type(plan.cuda_graph_decode) is not bool:
        raise RuntimeError("attention-DP graph enable flag is malformed")
    local = _ep_attention_dp_local_graph_decision(
        local_runner,
        cuda_graph_decode=plan.cuda_graph_decode,
        batch_size=plan.batch_size,
        max_pages=plan.max_pages,
    )
    if local != plan.decision:
        raise RuntimeError(
            "attention-DP graph decision differs across ranks: "
            f"payload={plan.decision!r}, local={local!r}"
        )
    decision = plan.decision
    if decision.kind == "eager_fallback":
        valid = (
            decision.bucket_size == plan.batch_size
            and decision.capture_model_forward_count == 1
        )
    elif decision.kind == "capture":
        valid = (
            plan.cuda_graph_decode
            and decision.bucket_size >= plan.batch_size
            and decision.capture_model_forward_count >= 1
        )
    else:
        valid = (
            decision.kind == "replay"
            and plan.cuda_graph_decode
            and decision.bucket_size >= plan.batch_size
            and decision.capture_model_forward_count == 0
        )
    if not valid:
        raise RuntimeError(
            f"attention-DP graph plan has malformed decision {decision!r}"
        )


def _ep_attention_dp_validate_graph_replies(
    replies: object,
    *,
    world_size: int,
) -> None:
    if not isinstance(replies, (tuple, list)) or len(replies) != world_size:
        raise RuntimeError(
            "attention-DP graph validation reply count does not match EP size"
        )
    by_rank: dict[int, _EPAttentionDPGraphValidationReply] = {}
    for reply in replies:
        if (
            not isinstance(reply, _EPAttentionDPGraphValidationReply)
            or type(reply.rank) is not int
            or not 0 <= reply.rank < world_size
            or reply.rank in by_rank
            or (
                reply.error is not None
                and (not isinstance(reply.error, str) or not reply.error)
            )
        ):
            raise RuntimeError("attention-DP graph validation reply is malformed")
        by_rank[reply.rank] = reply
    if set(by_rank) != set(range(world_size)):
        raise RuntimeError("attention-DP graph validation ranks are incomplete")
    failures = tuple(
        f"rank {rank}: {by_rank[rank].error}"
        for rank in range(world_size)
        if by_rank[rank].error is not None
    )
    if failures:
        raise RuntimeError(
            "attention-DP graph preflight failed before model collectives: "
            + "; ".join(failures)
        )


def _ep_attention_dp_graph_preflight(
    control_comm,
    local_runner,
    payload: _EPAttentionDPStep,
    states: Mapping[str, RequestSnapshot],
) -> None:
    """Agree stateful graph transitions; trust an already-certified replay.

    Capture remains an all-rank object preflight because it mutates the local
    graph cache.  A successful capture is then certified by the existing
    all-rank execution status before any later payload is processed.  Captured
    buckets are grow-only while serving, and request-owned attention-DP rejects
    live invalidation.  Steady replay therefore needs only a fixed-size CPU
    status reduction; the diagnostic object gather runs only after every rank
    has learned that at least one validation failed.  This preserves symmetric
    failure without adding a CUDA-to-host synchronization.
    """

    rank = control_comm.rank
    try:
        _ep_attention_dp_validate_graph_plan(
            local_runner,
            payload,
            states,
            world_size=control_comm.world_size,
        )
        validation = _EPAttentionDPGraphValidationReply(rank=rank)
    except BaseException as error:
        validation = _EPAttentionDPGraphValidationReply(
            rank=rank,
            error=(f"{type(error).__name__}: {error}")[:2048],
        )

    plan = payload.graph_plan
    certified_replay = (
        isinstance(plan, _EPAttentionDPGraphPlan)
        and plan.cuda_graph_decode is True
        and isinstance(plan.decision, GraphDecodeDecision)
        and plan.decision.kind == "replay"
    )
    if certified_replay:
        failure_counts = control_comm.all_reduce(
            (float(validation.error is not None),)
        )
        valid_counts = tuple(
            float(count) for count in range(control_comm.world_size + 1)
        )
        if (
            not isinstance(failure_counts, tuple)
            or len(failure_counts) != 1
            or failure_counts[0] not in valid_counts
        ):
            raise RuntimeError(
                "attention-DP certified graph replay status reduction is malformed"
            )
        if failure_counts[0] == 0.0:
            return

    validation_replies = control_comm.all_gather(validation)
    _ep_attention_dp_validate_graph_replies(
        validation_replies,
        world_size=control_comm.world_size,
    )


def _ep_attention_dp_plan(
    payload: _EPAttentionDPStep,
    states: dict[str, RequestSnapshot],
    *,
    rank: int,
    world_size: int,
    prefill_scratch_pages: tuple[int, int],
    decode_scratch_pages: tuple[int, ...],
) -> tuple[
    tuple[ScheduledChunk, ...],
    dict[str, RequestSnapshot],
    tuple[tuple[int, ...], ...],
    tuple[str, ...],
]:
    """Derive one rank's work plus identical per-forward collective layouts."""

    chunks = payload.delta.chunks
    if not chunks:
        raise RuntimeError("attention-DP step must schedule at least one chunk")
    if (
        not isinstance(prefill_scratch_pages, tuple)
        or len(prefill_scratch_pages) != _EP_ATTENTION_DP_SCRATCH_PAGES
        or any(type(page) is not int or page < 0 for page in prefill_scratch_pages)
        or len(set(prefill_scratch_pages)) != len(prefill_scratch_pages)
    ):
        raise RuntimeError("attention-DP prefill scratch page contract is malformed")
    if (
        not isinstance(decode_scratch_pages, tuple)
        or not decode_scratch_pages
        or any(type(page) is not int or page < 0 for page in decode_scratch_pages)
        or len(set(decode_scratch_pages)) != len(decode_scratch_pages)
        or set(decode_scratch_pages) & set(prefill_scratch_pages)
    ):
        raise RuntimeError("attention-DP decode scratch page contract is malformed")
    owner_map = _ep_attention_dp_owner_map(
        payload.owners,
        chunks,
        world_size=world_size,
    )
    if set(states) != set(owner_map):
        raise RuntimeError(
            "attention-DP state view does not match scheduled requests"
        )
    if any(not chunk.is_prefill and chunk.num_tokens != 1 for chunk in chunks):
        raise RuntimeError(
            "attention-DP does not support multi-token verification chunks"
        )
    global_prefill = any(chunk.is_prefill for chunk in chunks)
    global_decode = any(not chunk.is_prefill for chunk in chunks)
    layouts: list[tuple[int, ...]] = []
    per_rank_chunks = tuple(
        _ep_attention_dp_rank_chunks(chunks, owner_map, owner)
        for owner in range(world_size)
    )
    if global_prefill:
        prefill_rows: list[int] = []
        for owned in per_rank_chunks:
            prefills = tuple(chunk for chunk in owned if chunk.is_prefill)
            dummy_count = max(0, 2 - len(prefills))
            prefill_rows.append(
                sum(chunk.num_tokens for chunk in prefills) + dummy_count
            )
        layouts.append(tuple(prefill_rows))
    if global_decode:
        graph_plan = payload.graph_plan
        if not isinstance(graph_plan, _EPAttentionDPGraphPlan):
            raise RuntimeError("attention-DP decode step has no graph plan")
        decision = graph_plan.decision
        if decision.kind in {"capture", "replay"}:
            graph_layout = (decision.bucket_size,) * world_size
            layouts.extend(
                graph_layout
                for _ in range(decision.capture_model_forward_count)
            )
        else:
            decode_rows = tuple(
                max(1, sum(not chunk.is_prefill for chunk in owned))
                for owned in per_rank_chunks
            )
            layouts.append(decode_rows)

    owned = list(per_rank_chunks[rank])
    local_states = {
        chunk.request_id: states[chunk.request_id]
        for chunk in owned
    }
    dummy_ids: list[str] = []
    if global_prefill:
        local_prefills = sum(chunk.is_prefill for chunk in owned)
        for slot in range(max(0, 2 - local_prefills)):
            request_id = (
                f"{_EP_ATTENTION_DP_SCRATCH_PREFIX}:rank={rank}:prefill={slot}"
            )
            owned.append(ScheduledChunk(request_id, 1, True, 0))
            local_states[request_id] = _ep_attention_dp_dummy_state(
                request_id,
                page_id=prefill_scratch_pages[slot],
                prefill=True,
            )
            dummy_ids.append(request_id)
    if global_decode:
        graph_plan = payload.graph_plan
        assert graph_plan is not None
        local_decode_count = sum(not chunk.is_prefill for chunk in owned)
        if graph_plan.decision.kind in {"capture", "replay"}:
            dummy_count = graph_plan.decision.bucket_size - local_decode_count
        else:
            dummy_count = int(local_decode_count == 0)
        if dummy_count < 0 or dummy_count > len(decode_scratch_pages):
            raise RuntimeError(
                "attention-DP decode scratch capacity cannot pad the shared bucket"
            )
        for slot in range(dummy_count):
            request_id = (
                f"{_EP_ATTENTION_DP_SCRATCH_PREFIX}:rank={rank}:decode={slot}"
            )
            owned.append(ScheduledChunk(request_id, 1, False, 0))
            local_states[request_id] = _ep_attention_dp_dummy_state(
                request_id,
                page_id=decode_scratch_pages[slot],
                prefill=False,
            )
            dummy_ids.append(request_id)
    return tuple(owned), local_states, tuple(layouts), tuple(dummy_ids)


def _ep_attention_dp_local_reply(
    control_comm,
    model_comm,
    local_runner,
    payload: _EPAttentionDPStep,
    states: dict[str, RequestSnapshot],
) -> _EPAttentionDPLocalResult:
    """Execute owner-local work and retain pure-greedy tokens on the device."""

    rank = control_comm.rank
    dummy_ids: tuple[str, ...] = ()
    fast_path = _ep_attention_dp_fast_sampling(
        payload.delta.chunks,
        states,
    )
    failure: BaseException | None = None
    sampled = None
    try:
        _ep_attention_dp_graph_preflight(
            control_comm,
            local_runner,
            payload,
            states,
        )

        prefill_scratch_pages = getattr(
            local_runner,
            "attention_dp_scratch_pages",
            None,
        )
        decode_scratch_pages = getattr(
            local_runner,
            "attention_dp_decode_scratch_pages",
            None,
        )
        if decode_scratch_pages is None and isinstance(
            prefill_scratch_pages,
            tuple,
        ):
            # Compatibility for eager-only fake runners. Production runners
            # expose a disjoint decode reservation.
            decode_scratch_pages = (max(prefill_scratch_pages) + 1,)
        local_chunks, local_states, layouts, dummy_ids = _ep_attention_dp_plan(
            payload,
            states,
            rank=rank,
            world_size=control_comm.world_size,
            prefill_scratch_pages=prefill_scratch_pages,
            decode_scratch_pages=decode_scratch_pages,
        )
        ragged_layouts = tuple(
            dict.fromkeys(
                layout for layout in layouts if len(set(layout)) > 1
            )
        )
        if (
            ragged_layouts
            and getattr(
                model_comm,
                "attention_dp_direct_nccl_active",
                False,
            )
            is True
        ):
            synchronize = getattr(
                model_comm,
                "synchronize_attention_dp_ragged_layouts",
                None,
            )
            if not callable(synchronize):
                raise RuntimeError(
                    "active attention-DP direct NCCL communicator cannot "
                    "synchronize ragged layouts"
                )
            synchronized = synchronize(ragged_layouts)
            if synchronized != ragged_layouts:
                raise RuntimeError(
                    "attention-DP ragged layout synchronization changed the "
                    "worker plan"
                )
        from kairyu.models.moe_parallel import (
            assert_attention_dp_layouts_consumed,
            assert_attention_dp_layouts_idle,
            prepare_attention_dp_layouts,
        )

        model = getattr(local_runner, "_model", None)
        if model is None:
            raise RuntimeError("attention-DP local runner exposes no model")
        if layouts:
            prepare_attention_dp_layouts(model, layouts)
        else:
            assert_attention_dp_layouts_idle(model)
        graph_plan = payload.graph_plan
        if graph_plan is not None and graph_plan.cuda_graph_decode:
            coordinate = getattr(
                local_runner,
                "coordinate_decode_graph_decision",
                None,
            )
            if not callable(coordinate):
                raise RuntimeError(
                    "attention-DP CUDA graph runner has no coordinated override"
                )
            coordinate(graph_plan.decision)
        sampled = local_runner.execute(local_chunks, local_states)
        if graph_plan is not None and graph_plan.cuda_graph_decode:
            consumed = getattr(
                local_runner,
                "assert_coordinated_decode_graph_decision_consumed",
                None,
            )
            if not callable(consumed):
                raise RuntimeError(
                    "attention-DP CUDA graph runner cannot verify override consumption"
                )
            consumed()
        if layouts:
            assert_attention_dp_layouts_consumed(model)
        else:
            assert_attention_dp_layouts_idle(model)
    except BaseException as error:
        cancel = getattr(
            local_runner,
            "cancel_coordinated_decode_graph_decision",
            None,
        )
        if callable(cancel):
            cancel()
        failure = error

    try:
        if fast_path:
            try:
                packet = _ep_attention_dp_local_token_packet(
                    model_comm,
                    local_runner,
                    payload,
                    states,
                    sampled,
                    dummy_ids=dummy_ids,
                    failed=failure is not None,
                )
            except BaseException as error:
                if failure is None:
                    failure = error
                packet = _ep_attention_dp_failed_token_packet(
                    model_comm,
                    local_runner,
                    chunk_count=len(payload.delta.chunks),
                )
            gathered = model_comm.tensor_all_gather(packet)
            reply = _EPAttentionDPReply(
                rank=rank,
                sampled={} if failure is None else None,
                error=(
                    None
                    if failure is None
                    else (
                        "local execution or token-packet encoding failed: "
                        f"{type(failure).__name__}: {failure}"
                    )[:2048]
                ),
            )
            return _EPAttentionDPLocalResult(reply, gathered)

        if failure is not None:
            raise failure
        assert sampled is not None
        resolved = {
            request_id: tuple(tokens)
            for request_id, tokens in sampled.items()
            if request_id not in set(dummy_ids)
        }
        if any(
            not isinstance(token, SampledToken)
            for tokens in resolved.values()
            for token in tokens
        ):
            raise RuntimeError(
                "attention-DP sampler returned a malformed token record"
            )
        return _EPAttentionDPLocalResult(
            _EPAttentionDPReply(rank=rank, sampled=resolved)
        )
    except BaseException as error:
        return _EPAttentionDPLocalResult(
            _EPAttentionDPReply(
                rank=rank,
                error=(f"{type(error).__name__}: {error}")[:2048],
            )
        )
    finally:
        release = getattr(local_runner, "release", None)
        if callable(release):
            for request_id in dummy_ids:
                release(request_id)


def _ep_attention_dp_chunk_emits(
    chunk: ScheduledChunk,
    state: RequestSnapshot,
) -> bool:
    return (
        chunk.num_tokens > 0
        if not chunk.is_prefill
        else bool(
            state.prefill_done
            and state.computed_prompt == len(state.prompt_token_ids)
        )
    )


def _ep_attention_dp_fast_sampling(
    chunks: tuple[ScheduledChunk, ...],
    states: dict[str, RequestSnapshot],
) -> bool:
    """Use the future-token device path only when no metadata can be lost."""

    emitting = tuple(
        chunk
        for chunk in chunks
        if _ep_attention_dp_chunk_emits(chunk, states[chunk.request_id])
    )
    return bool(emitting) and all(
        states[chunk.request_id].sampling.is_greedy_pure
        and states[chunk.request_id].sampling.logprobs is None
        and not states[chunk.request_id].sampling.needs_grammar
        for chunk in emitting
    )


def _ep_attention_dp_raw_records(sampled):
    raw_records = getattr(sampled, "raw_records", None)
    return raw_records() if callable(raw_records) else sampled


def _ep_attention_dp_failed_token_packet(
    model_comm,
    local_runner,
    *,
    chunk_count: int,
):
    """Build the minimal fixed packet after model or encoder failure."""

    import torch

    from kairyu.engine.core.comm import FakeCommunicator

    packet_device = (
        torch.device("cpu")
        if isinstance(model_comm, FakeCommunicator)
        else torch.device(local_runner._device)
    )
    packet = torch.full(
        (chunk_count + 1,),
        -1,
        dtype=torch.int64,
        device=packet_device,
    )
    packet[0].fill_(_EP_ATTENTION_DP_PACKET_FAILED)
    return packet


def _ep_attention_dp_local_token_packet(
    model_comm,
    local_runner,
    payload: _EPAttentionDPStep,
    states: dict[str, RequestSnapshot],
    sampled,
    *,
    dummy_ids: tuple[str, ...],
    failed: bool,
):
    """Encode rank status plus owner-local greedy ids in one device packet."""

    chunks = payload.delta.chunks
    if failed:
        return _ep_attention_dp_failed_token_packet(
            model_comm,
            local_runner,
            chunk_count=len(chunks),
        )

    import torch

    from kairyu.engine.core.comm import FakeCommunicator

    packet_device = (
        torch.device("cpu")
        if isinstance(model_comm, FakeCommunicator)
        else torch.device(local_runner._device)
    )
    owner_map = _ep_attention_dp_owner_map(
        payload.owners,
        chunks,
        world_size=model_comm.world_size,
    )
    records = {} if sampled is None else _ep_attention_dp_raw_records(sampled)
    if not isinstance(records, Mapping):
        raise RuntimeError("attention-DP sampler records are not a mapping")
    dummy = set(dummy_ids)
    public_records = {
        request_id: values
        for request_id, values in records.items()
        if request_id not in dummy
    }
    expected = {
        chunk.request_id
        for chunk in chunks
        if owner_map[chunk.request_id] == model_comm.rank
        and _ep_attention_dp_chunk_emits(chunk, states[chunk.request_id])
    }
    if not failed and set(public_records) != expected:
        raise RuntimeError(
            "attention-DP owner-local sampled request set mismatch: "
            f"expected={sorted(expected)}, got={sorted(public_records)}"
        )

    sampled_device = None
    for values in records.values():
        for record in values:
            sample = getattr(record, "sample", None)
            token_id = getattr(sample, "token_id", None)
            if isinstance(token_id, torch.Tensor):
                sampled_device = token_id.device
                break
        if sampled_device is not None:
            break
    if sampled_device is not None:
        packet_device = sampled_device
    packet = torch.full(
        (len(chunks) + 1,),
        -1,
        dtype=torch.int64,
        device=packet_device,
    )
    packet[0].fill_(_EP_ATTENTION_DP_PACKET_OK)
    for index, chunk in enumerate(chunks):
        if (
            owner_map[chunk.request_id] != model_comm.rank
            or not _ep_attention_dp_chunk_emits(
                chunk,
                states[chunk.request_id],
            )
        ):
            continue
        values = public_records[chunk.request_id]
        if not isinstance(values, tuple) or len(values) != 1:
            raise RuntimeError(
                "attention-DP pure-greedy packet requires exactly one token for "
                f"{chunk.request_id!r}"
            )
        record = values[0]
        if isinstance(record, SampledToken):
            if record.token_id < 0:
                raise RuntimeError("attention-DP sampler returned a negative token id")
            packet[index + 1] = record.token_id
            continue
        sample = getattr(record, "sample", None)
        token_id = getattr(sample, "token_id", None)
        if (
            not isinstance(token_id, torch.Tensor)
            or token_id.numel() != 1
            or token_id.dtype is not torch.int64
            or token_id.device != packet.device
        ):
            raise RuntimeError(
                "attention-DP sampler returned a malformed device token"
            )
        packet[index + 1].copy_(token_id)
    return packet


def _ep_attention_dp_packet_views(
    gathered,
    *,
    chunk_count: int,
    world_size: int,
):
    """Return the device-resident rank-major token packet view."""

    import torch

    packet_width = chunk_count + 1
    if (
        not isinstance(gathered, torch.Tensor)
        or gathered.dtype is not torch.int64
        or tuple(gathered.shape) != (world_size * packet_width,)
    ):
        raise RuntimeError(
            "attention-DP gathered token packet has malformed shape or dtype"
        )
    matrix = gathered.reshape(world_size, packet_width)
    return matrix[:, 1:]


def _ep_attention_dp_validate_fast_status_replies(
    replies: object,
    *,
    world_size: int,
) -> None:
    """Raise one deterministic all-rank diagnostic for a fast-step failure."""

    if not isinstance(replies, (tuple, list)) or len(replies) != world_size:
        raise RuntimeError(
            "attention-DP fast status gathered malformed rank count: "
            f"expected {world_size}, got "
            f"{len(replies) if isinstance(replies, (tuple, list)) else type(replies).__name__}"
        )
    by_rank: dict[int, _EPAttentionDPReply] = {}
    failures: list[str] = []
    for reply in replies:
        if (
            not isinstance(reply, _EPAttentionDPReply)
            or type(reply.rank) is not int
            or not 0 <= reply.rank < world_size
            or reply.rank in by_rank
        ):
            failures.append(f"malformed reply: {reply!r}")
            continue
        by_rank[reply.rank] = reply
        if reply.error is not None:
            failures.append(f"rank {reply.rank}: {reply.error}")
        elif reply.sampled != {}:
            failures.append(
                f"rank {reply.rank}: fast status sampled payload is malformed"
            )
    missing_ranks = sorted(set(range(world_size)) - set(by_rank))
    if missing_ranks:
        failures.append(f"missing ranks={missing_ranks}")
    if not failures:
        failures.append(
            "fixed failure reduction disagrees with all-rank diagnostics"
        )
    raise RuntimeError("attention-DP fast rank failures: " + "; ".join(failures))


def _ep_attention_dp_fast_status(
    control_comm,
    reply: _EPAttentionDPReply,
) -> None:
    """Agree fast-step host failures without waiting for the CUDA packet.

    The caller invokes this only after every rank has enqueued the fixed token
    packet all-gather.  The one-scalar gloo reduction is unconditional and can
    overlap the already-enqueued CUDA work.  Python objects cross the control
    group only on failure, when they retain the bounded rank-local diagnostic.
    """

    local_failed = (
        not isinstance(reply, _EPAttentionDPReply)
        or type(reply.rank) is not int
        or reply.rank != control_comm.rank
        or (reply.error is None and reply.sampled != {})
        or (reply.error is not None and not isinstance(reply.error, str))
    )
    if isinstance(reply, _EPAttentionDPReply) and reply.error is not None:
        local_failed = True
    failure_counts = control_comm.all_reduce((float(local_failed),))
    valid_counts = tuple(float(count) for count in range(control_comm.world_size + 1))
    if (
        not isinstance(failure_counts, tuple)
        or len(failure_counts) != 1
        or type(failure_counts[0]) is not float
        or failure_counts[0] not in valid_counts
    ):
        raise RuntimeError(
            "attention-DP fast status reduction is malformed"
        )
    if failure_counts[0] == 0.0:
        return
    replies = control_comm.all_gather(reply)
    _ep_attention_dp_validate_fast_status_replies(
        replies,
        world_size=control_comm.world_size,
    )


def _ep_attention_dp_deferred_packet_output(
    gathered,
    *,
    chunks: tuple[ScheduledChunk, ...],
    states: dict[str, RequestSnapshot],
    owners: tuple[tuple[str, int], ...],
    world_size: int,
    local_runner,
):
    """Select owner slots D2D and defer their one vector D2H to commit."""

    from kairyu.engine.core.model_runner import (
        _DeferredStepOutput,
        _PendingDeviceToken,
    )
    from kairyu.engine.core.sampler import DeviceSample

    matrix = _ep_attention_dp_packet_views(
        gathered,
        chunk_count=len(chunks),
        world_size=world_size,
    )
    owner_map = _ep_attention_dp_owner_map(
        owners,
        chunks,
        world_size=world_size,
    )
    records: dict[str, tuple[object, ...]] = {}
    for index, chunk in enumerate(chunks):
        if not _ep_attention_dp_chunk_emits(
            chunk,
            states[chunk.request_id],
        ):
            continue
        if chunk.request_id in records:
            raise RuntimeError(
                "attention-DP device packet cannot emit twice for one request"
            )

        def validate_token(
            token: SampledToken,
            request_id: str = chunk.request_id,
        ) -> None:
            if token.token_id < 0:
                raise RuntimeError(
                    "attention-DP owner packet omitted token for "
                    f"{request_id!r}"
                )

        records[chunk.request_id] = (
            _PendingDeviceToken(
                sample=DeviceSample(
                    matrix[owner_map[chunk.request_id], index]
                ),
                on_resolve=validate_token,
            ),
        )
    return _DeferredStepOutput(
        records,
        getattr(local_runner, "_output_copy_stream", None),
    )


def _validate_ep_attention_dp_replies(
    replies: object,
    *,
    chunks: tuple[ScheduledChunk, ...],
    states: dict[str, RequestSnapshot],
    world_size: int,
) -> dict[str, tuple[SampledToken, ...]]:
    if not isinstance(replies, (tuple, list)) or len(replies) != world_size:
        raise RuntimeError(
            "attention-DP gathered malformed rank count: "
            f"expected {world_size}, got "
            f"{len(replies) if isinstance(replies, (tuple, list)) else type(replies).__name__}"
        )
    by_rank: dict[int, _EPAttentionDPReply] = {}
    failures: list[str] = []
    merged: dict[str, tuple[SampledToken, ...]] = {}
    for reply in replies:
        if (
            not isinstance(reply, _EPAttentionDPReply)
            or type(reply.rank) is not int
            or not 0 <= reply.rank < world_size
            or reply.rank in by_rank
        ):
            raise RuntimeError(
                f"attention-DP gathered malformed reply: {reply!r}"
            )
        by_rank[reply.rank] = reply
        if reply.error is not None:
            failures.append(f"rank {reply.rank}: {reply.error}")
            continue
        if not isinstance(reply.sampled, dict):
            failures.append(f"rank {reply.rank}: sampled payload is not a dict")
            continue
        for request_id, tokens in reply.sampled.items():
            if request_id in merged:
                failures.append(
                    f"request {request_id!r} was sampled by more than one rank"
                )
                continue
            if (
                not isinstance(request_id, str)
                or not isinstance(tokens, tuple)
                or any(not isinstance(token, SampledToken) for token in tokens)
            ):
                failures.append(
                    f"rank {reply.rank}: malformed sampled row {request_id!r}"
                )
                continue
            merged[request_id] = tokens
    missing_ranks = sorted(set(range(world_size)) - set(by_rank))
    if missing_ranks:
        failures.append(f"missing ranks={missing_ranks}")
    expected: list[str] = []
    for chunk in chunks:
        state = states[chunk.request_id]
        emits = _ep_attention_dp_chunk_emits(chunk, state)
        if emits:
            expected.append(chunk.request_id)
    if len(expected) != len(set(expected)):
        failures.append(
            "scheduled step emits more than one sampled row for a request"
        )
    if set(merged) != set(expected):
        failures.append(
            "sampled request set mismatch: "
            f"expected={sorted(set(expected))}, got={sorted(merged)}"
        )
    if failures:
        raise RuntimeError("attention-DP rank failures: " + "; ".join(failures))
    return merged


class DistEPModelRunner:
    """Driver facade for explicit replicated or request-owned EP execution.

    The default retains M-A1/M-A2's replicated-attention correctness protocol.
    The opt-in M-A3 path fixes one owner rank per request; only that rank runs
    attention, owns KV, and samples, while every rank enters the same packed
    NVFP4 MoE all-gather/reduce-scatter sequence.
    """

    def __init__(
        self,
        control_comm,
        local_runner,
        model_comm=None,
        *,
        expert_parallel_size: int,
        attention_dp: bool = False,
        pipeline_depth: int = 1,
        page_size: int = 16,
        decode_mode: str = "eager",
        graph_scratch_page: int | None = None,
        graph_max_batch: int = 0,
        graph_max_pages: int = 0,
        graph_warmup_iters: int = 3,
    ) -> None:
        _validate_ep_mode(
            attention_dp=attention_dp,
            expert_parallel_size=expert_parallel_size,
            pipeline_depth=pipeline_depth,
            decode_mode=decode_mode,
            graph_scratch_page=graph_scratch_page,
            graph_max_batch=graph_max_batch,
            graph_max_pages=graph_max_pages,
            graph_warmup_iters=graph_warmup_iters,
        )
        if control_comm.world_size != expert_parallel_size:
            raise ValueError(
                "EP control communicator world size does not match "
                f"expert_parallel_size={expert_parallel_size}"
            )
        if model_comm is not None and model_comm.world_size != expert_parallel_size:
            raise ValueError(
                "EP model communicator world size does not match "
                f"expert_parallel_size={expert_parallel_size}"
            )
        self._delegate = DistTPModelRunner(
            control_comm,
            local_runner,
            model_comm,
            parallelism_display=(
                "request-owned expert-parallel attention-DP"
                if attention_dp
                else "expert-parallel correctness-mode"
            ),
            parallelism_prefix="EP",
        )
        if type(page_size) is not int or page_size < 1:
            raise ValueError("EP page_size must be a positive integer")
        self._attention_dp = attention_dp
        self._page_size = page_size
        self._request_owners: dict[str, int] = {}
        self._kv_page_owners: dict[int, int] = {}
        self._next_owner = 0
        self._owner_assignment_sequence = 0
        self._owner_assignment_log: list[dict[str, object]] = []
        self._owner_assignment_dropped = 0
        self.expert_parallel_size = expert_parallel_size
        self.parallelism = "expert_parallel"
        self.execution_mode = (
            _EP_ATTENTION_DP_MODE if attention_dp else _EP_CORRECTNESS_MODE
        )
        self.attention_placement = (
            "request_owned_data_parallel" if attention_dp else "replicated"
        )
        self.attention_output_placement = (
            "replicated" if attention_dp else "row_parallel"
        )
        self.attention_output_parallel_size = (
            1 if attention_dp else expert_parallel_size
        )
        self.attention_output_partial_dtype = (
            None if attention_dp else "bfloat16"
        )
        self.pipeline_depth = pipeline_depth
        self.decode_mode = decode_mode

    def __getattr__(self, name: str):
        """Delegate the proven SPMD control protocol without becoming a TP type."""

        return getattr(object.__getattribute__(self, "_delegate"), name)

    def set_batched_prefill_enabled(self, enabled: bool) -> None:
        """Keep attention-DP's one-layout-per-forward invariant fail-closed."""

        if type(enabled) is not bool:
            raise TypeError("batched prefill enabled flag must be bool")
        if self._attention_dp and not enabled:
            raise ValueError(
                "request-owned attention-DP cannot disable batched prefill"
            )
        self._delegate.set_batched_prefill_enabled(enabled)

    def invalidate_graphs(self) -> None:
        """Prevent one rank from dropping a graph while peers keep replaying."""

        delegate = object.__getattribute__(self, "_delegate")
        with delegate._protocol_lock:
            if self._attention_dp and not delegate._shutdown_started:
                raise RuntimeError(
                    "request-owned attention-DP cannot invalidate CUDA graphs "
                    "while serving; shut down every rank first"
                )
            delegate.invalidate_graphs()

    def parallelism_metadata(self) -> dict[str, object]:
        """Unambiguous topology metadata for gates and serving surfaces."""

        metadata: dict[str, object] = {
            "parallelism": self.parallelism,
            "expert_parallel_size": self.expert_parallel_size,
            "attention_placement": self.attention_placement,
            "attention_output_placement": self.attention_output_placement,
            "attention_output_parallel_size": self.attention_output_parallel_size,
            "attention_output_partial_dtype": self.attention_output_partial_dtype,
            "execution_mode": self.execution_mode,
            "pipeline_depth": self.pipeline_depth,
            "decode_mode": self.decode_mode,
            "kv_cache_dtype": "bfloat16",
        }
        if self._attention_dp:
            delegate = object.__getattribute__(self, "_delegate")
            graph_probe = getattr(
                delegate._local,
                "decode_graph_metadata",
                None,
            )
            if callable(graph_probe):
                graph_metadata = graph_probe()
            elif self.decode_mode == "eager":
                graph_metadata = {
                    "decode_mode": "eager",
                    "cuda_graph_decode": False,
                    "cuda_graph_buckets": (),
                    "cuda_graph_captures": 0,
                    "cuda_graph_replays": 0,
                    "cuda_graph_eager_fallbacks": 0,
                }
            else:
                raise RuntimeError(
                    "attention-DP CUDA graph runner exposes no graph metadata"
                )
            if graph_metadata.get("decode_mode") != self.decode_mode:
                raise RuntimeError(
                    "attention-DP runner graph metadata disagrees with its "
                    "configured decode mode"
                )
            transport_probe = getattr(
                delegate._model_comm,
                "attention_dp_collective_metadata",
                None,
            )
            if callable(transport_probe):
                collective_transport = transport_probe()
            else:
                backend = _communicator_backend(delegate._model_comm)
                collective_transport = {
                    "selected_backend": backend,
                    "fallback_backend": backend,
                    "direct_nccl_active": False,
                    "direct_nccl_library": None,
                    "direct_nccl_version": None,
                    "selection_reason": (
                        "communicator does not expose a direct NCCL transport"
                    ),
                }
            metadata.update(
                {
                    "attention_data_parallel_size": self.expert_parallel_size,
                    "attention_tensor_parallel_size": 1,
                    "moe_dispatcher": _EP_ATTENTION_DP_DISPATCHER,
                    "sampling_ownership": "request_owner",
                    "kv_cache_ownership": "request_owner",
                    **graph_metadata,
                    "attention_dp_prefill_scratch_pages": (
                        _EP_ATTENTION_DP_SCRATCH_PAGES
                    ),
                    "attention_dp_decode_scratch_pages": len(
                        getattr(
                            delegate._local,
                            "attention_dp_decode_scratch_pages",
                            (0,),
                        )
                    ),
                    "attention_dp_graph_scratch_pages": int(
                        graph_metadata["cuda_graph_decode"]
                    ),
                    "attention_dp_scratch_pages": getattr(
                        delegate._local,
                        "attention_dp_reserved_page_count",
                        _EP_ATTENTION_DP_SCRATCH_PAGES + 1,
                    )
                    + int(graph_metadata["cuda_graph_decode"]),
                    "moe_collective_transport": collective_transport,
                }
            )
        return metadata

    @staticmethod
    def _state_page_contract(
        state: object,
    ) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        allocation = getattr(state, "allocation", None)
        if allocation is None:
            raise RuntimeError("scheduled attention-DP request has no KV allocation")
        pages = tuple(getattr(allocation, "pages", ()))
        decode_pages = tuple(getattr(state, "decode_pages", ()))
        cached_tokens = getattr(allocation, "num_cached_tokens", None)
        if (
            any(type(page) is not int or page < 0 for page in pages + decode_pages)
            or type(cached_tokens) is not int
            or cached_tokens < 0
        ):
            raise RuntimeError("scheduled attention-DP request has malformed KV state")
        return pages, decode_pages, cached_tokens

    def _stage_attention_dp_owners(
        self,
        chunks: tuple[ScheduledChunk, ...],
        states,
        *,
        released_ids: tuple[str, ...],
    ) -> tuple[
        dict[str, int],
        dict[int, int],
        int,
        tuple[tuple[str, int], ...],
        tuple[dict[str, object], ...],
    ]:
        owners = dict(self._request_owners)
        page_owners = dict(self._kv_page_owners)
        for request_id in released_ids:
            owners.pop(request_id, None)
        next_owner = self._next_owner
        active_counts = [
            sum(owner == rank for owner in owners.values())
            for rank in range(self.expert_parallel_size)
        ]
        ordered_ids = tuple(dict.fromkeys(chunk.request_id for chunk in chunks))
        new_log: list[dict[str, object]] = []
        for request_id in ordered_ids:
            pages, decode_pages, cached_tokens = self._state_page_contract(
                states[request_id]
            )
            cached_page_count = min(
                len(pages),
                (cached_tokens + self._page_size - 1) // self._page_size,
            )
            cached_pages = pages[:cached_page_count]
            cached_owners = {
                page_owners[page]
                for page in cached_pages
                if page in page_owners
            }
            if cached_pages and len(cached_owners) != 1:
                raise RuntimeError(
                    "attention-DP cached prefix has missing or mixed KV owners "
                    f"for request {request_id!r}"
                )
            cached_owner = next(iter(cached_owners)) if cached_owners else None
            owner = owners.get(request_id)
            if owner is None:
                if cached_owner is not None:
                    owner = cached_owner
                else:
                    minimum = min(active_counts)
                    candidates = {
                        rank
                        for rank, count in enumerate(active_counts)
                        if count == minimum
                    }
                    owner = next(
                        (next_owner + offset) % self.expert_parallel_size
                        for offset in range(self.expert_parallel_size)
                        if (next_owner + offset) % self.expert_parallel_size
                        in candidates
                    )
                    next_owner = (owner + 1) % self.expert_parallel_size
                owners[request_id] = owner
                active_counts[owner] += 1
                new_log.append(
                    {
                        "sequence": self._owner_assignment_sequence
                        + len(new_log),
                        "request_id": request_id,
                        "owner_rank": owner,
                        "cache_affine": cached_owner is not None,
                    }
                )
            elif cached_owner is not None and cached_owner != owner:
                raise RuntimeError(
                    "attention-DP request owner disagrees with cached KV owner "
                    f"for request {request_id!r}: request={owner}, cache={cached_owner}"
                )
            for page in pages + decode_pages:
                page_owners[page] = owner
        owner_pairs = tuple((request_id, owners[request_id]) for request_id in ordered_ids)
        return owners, page_owners, next_owner, owner_pairs, tuple(new_log)

    def _retain_attention_dp_assignments(
        self,
        assignments: tuple[dict[str, object], ...],
    ) -> None:
        """Commit a contiguous sequence while retaining only the newest bound."""

        if not isinstance(assignments, tuple):
            raise RuntimeError("attention-DP assignment evidence must be a tuple")
        for offset, assignment in enumerate(assignments):
            expected_sequence = self._owner_assignment_sequence + offset
            if (
                not isinstance(assignment, dict)
                or set(assignment) != _EP_ATTENTION_DP_ASSIGNMENT_FIELDS
                or type(assignment["sequence"]) is not int
                or assignment["sequence"] != expected_sequence
                or not isinstance(assignment["request_id"], str)
                or not assignment["request_id"]
                or type(assignment["owner_rank"]) is not int
                or not 0
                <= assignment["owner_rank"]
                < self.expert_parallel_size
                or type(assignment["cache_affine"]) is not bool
            ):
                raise RuntimeError(
                    "attention-DP assignment evidence is malformed at sequence "
                    f"{expected_sequence}"
                )
        self._owner_assignment_log.extend(assignments)
        self._owner_assignment_sequence += len(assignments)
        overflow = max(
            0,
            len(self._owner_assignment_log)
            - _EP_ATTENTION_DP_ASSIGNMENT_LOG_LIMIT,
        )
        if overflow:
            del self._owner_assignment_log[:overflow]
            self._owner_assignment_dropped += overflow

    def execute(self, scheduled, states) -> dict:
        if not self._attention_dp:
            return self._delegate.execute(scheduled, states)
        delegate = object.__getattribute__(self, "_delegate")
        with delegate._protocol_lock:
            if delegate._fatal_error is not None:
                raise RuntimeError(
                    "request-owned attention-DP runner is unavailable after a "
                    "fatal step failure"
                ) from delegate._fatal_error
            chunks = tuple(scheduled)
            pending_releases = delegate._snapshot_pending_releases()
            released_ids = tuple(request_id for request_id, _ in pending_releases)
            try:
                delta = delegate._sync.diff(
                    chunks,
                    states,
                    released_ids=released_ids,
                )
                (
                    staged_owners,
                    staged_page_owners,
                    staged_next_owner,
                    owner_pairs,
                    new_log,
                ) = self._stage_attention_dp_owners(
                    chunks,
                    states,
                    released_ids=released_ids,
                )
                graph_plan = _ep_attention_dp_make_graph_plan(
                    delegate._local,
                    chunks,
                    owner_pairs,
                    states,
                    world_size=self.expert_parallel_size,
                    cuda_graph_decode=self.decode_mode == "cuda_graph",
                )
                payload = _EPAttentionDPStep(
                    delta,
                    owner_pairs,
                    graph_plan,
                )
                delivered = delegate._broadcast_control(payload)
                if delivered != payload:
                    raise RuntimeError(
                        "attention-DP control broadcast returned a malformed payload"
                    )
                _release_runner_requests(delegate._local, released_ids)
                view = delegate._sync.apply(delta)
                delegate._ack_pending_releases(pending_releases)
                fast_path = _ep_attention_dp_fast_sampling(chunks, view)
                local = _ep_attention_dp_local_reply(
                    delegate._control_comm,
                    delegate._model_comm,
                    delegate._local,
                    payload,
                    view,
                )
                if fast_path:
                    _ep_attention_dp_fast_status(
                        delegate._control_comm,
                        local.reply,
                    )
                    sampled = _ep_attention_dp_deferred_packet_output(
                        local.gathered_token_packet,
                        chunks=chunks,
                        states=view,
                        owners=owner_pairs,
                        world_size=self.expert_parallel_size,
                        local_runner=delegate._local,
                    )
                else:
                    gathered = delegate._control_comm.all_gather(local.reply)
                    sampled = _validate_ep_attention_dp_replies(
                        gathered,
                        chunks=chunks,
                        states=view,
                        world_size=self.expert_parallel_size,
                    )
                self._request_owners = staged_owners
                self._kv_page_owners = staged_page_owners
                self._next_owner = staged_next_owner
                self._retain_attention_dp_assignments(new_log)
                return sampled
            except Exception as error:
                delegate._fatal_error = error
                raise

    @_serialized_protocol
    def sampling_ownership_metadata(self) -> tuple[dict[str, object], ...]:
        """Validate the sampler contract for the selected EP execution mode."""

        delegate = object.__getattribute__(self, "_delegate")
        if delegate._fatal_error is not None:
            raise RuntimeError(
                "expert-parallel runner is unavailable after a fatal step failure"
            ) from delegate._fatal_error
        try:
            control_world_size = delegate._control_comm.world_size
            model_world_size = delegate._model_comm.world_size
            control_backend = _communicator_backend(delegate._control_comm)
            model_backend = _communicator_backend(delegate._model_comm)
            delivered = delegate._broadcast_control(
                _SamplingOwnershipProbe()
            )
            if not isinstance(delivered, _SamplingOwnershipProbe):
                raise RuntimeError(
                    "sampling ownership probe broadcast returned a malformed payload"
                )
            local = _sampling_ownership_reply(
                delegate._control_comm,
                delegate._model_comm,
                delegate._local,
            )
            gathered = delegate._control_comm.all_gather(local)
            return _validate_sampling_ownership_replies(
                gathered,
                control_world_size=control_world_size,
                model_world_size=model_world_size,
                control_backend=control_backend,
                model_backend=model_backend,
                request_owner_sampling=self._attention_dp,
            )
        except Exception as error:
            failure = RuntimeError(
                f"EP sampling ownership metadata probe failed: {error}"
            )
            delegate._fatal_error = failure
            raise failure from error

    def attention_dp_ownership_metadata(self) -> dict[str, object]:
        """Return bounded exact ownership evidence outside the hot path."""

        if not self._attention_dp:
            raise RuntimeError(
                "attention-DP ownership metadata requires request-owned mode"
            )
        delegate = object.__getattribute__(self, "_delegate")
        with delegate._protocol_lock:
            retained = len(self._owner_assignment_log)
            if (
                retained > _EP_ATTENTION_DP_ASSIGNMENT_LOG_LIMIT
                or self._owner_assignment_dropped + retained
                != self._owner_assignment_sequence
                or any(
                    assignment["sequence"]
                    != self._owner_assignment_dropped + offset
                    for offset, assignment in enumerate(self._owner_assignment_log)
                )
            ):
                raise RuntimeError(
                    "attention-DP assignment retention accounting is inconsistent"
                )
            owner_counts = {
                str(rank): sum(
                    owner == rank for owner in self._request_owners.values()
                )
                for rank in range(self.expert_parallel_size)
            }
            return {
                "execution_mode": self.execution_mode,
                "assignment_count": self._owner_assignment_sequence,
                "assignment_retention_limit": (
                    _EP_ATTENTION_DP_ASSIGNMENT_LOG_LIMIT
                ),
                "retained_assignment_count": retained,
                "dropped_assignment_count": self._owner_assignment_dropped,
                "active_request_count": len(self._request_owners),
                "tracked_kv_page_count": len(self._kv_page_owners),
                "owner_counts": owner_counts,
                "assignments": tuple(self._owner_assignment_log),
            }

    @_serialized_protocol
    def _ep_kv_accounting_collect(
        self,
        action: str,
    ) -> tuple[dict[str, object], ...]:
        if self._attention_dp:
            raise RuntimeError(
                "M-A2 replicated-rank KV accounting is not valid for "
                "request-owned attention-DP"
            )
        delegate = object.__getattribute__(self, "_delegate")
        if action not in {"start", "finish"}:
            raise ValueError("EP KV accounting action must be start or finish")
        if delegate._fatal_error is not None:
            raise RuntimeError(
                "expert-parallel correctness-mode runner is unavailable after "
                "a fatal step failure"
            ) from delegate._fatal_error
        try:
            payload = _EPKVAccountingControl(action)
            delivered = delegate._broadcast_control(payload)
            if not isinstance(delivered, _EPKVAccountingControl):
                raise RuntimeError(
                    "EP KV accounting control returned a malformed payload"
                )
            recorder = delegate._ep_kv_accounting
            if recorder is None:
                raise RuntimeError("EP KV accounting recorder is unavailable")
            local = _ep_kv_accounting_reply(
                delegate._control_comm,
                recorder,
                action=delivered.action,
            )
            gathered = delegate._control_comm.all_gather(local)
            return _validate_ep_kv_accounting_replies(
                gathered,
                world_size=self.expert_parallel_size,
                action=action,
            )
        except Exception as error:
            failure = RuntimeError(
                f"EP KV accounting {action} probe failed: {error}"
            )
            delegate._fatal_error = failure
            raise failure from error

    def start_ep_kv_accounting(self) -> tuple[dict[str, object], ...]:
        """Arm a bounded rank-local capture and prove every EP rank started."""

        return self._ep_kv_accounting_collect("start")

    def finish_ep_kv_accounting(self) -> tuple[dict[str, object], ...]:
        """Gather and disarm the rank-local observations after a fixed trace."""

        return self._ep_kv_accounting_collect("finish")

    @_serialized_protocol
    def ep_kernel_inventory_metadata(self) -> tuple[dict[str, object], ...]:
        """Gather bounded all-rank NVFP4 runtime evidence over gloo.

        Projection names never leave their owning rank. Each rank sends only a
        count, canonical names SHA-256, quant/kernel counts, live attention
        output geometry, unresolved-meta counts, and a reduced loader summary.
        The returned tuple is rank ordered.
        """

        delegate = object.__getattribute__(self, "_delegate")
        if delegate._fatal_error is not None:
            raise RuntimeError(
                "expert-parallel runner is unavailable after a fatal step failure"
            ) from delegate._fatal_error
        try:
            backend = _communicator_backend(delegate._control_comm)
            if backend not in {"gloo", "fake"}:
                raise RuntimeError(
                    "EP kernel inventory requires the bounded control communicator "
                    f"(gloo), got {backend!r}"
                )
            delivered = delegate._broadcast_control(
                _EPKernelInventoryProbe()
            )
            if not isinstance(delivered, _EPKernelInventoryProbe):
                raise RuntimeError(
                    "EP kernel inventory probe broadcast returned a malformed payload"
                )
            local = _ep_kernel_inventory_reply(
                delegate._control_comm,
                delegate._local,
            )
            gathered = delegate._control_comm.all_gather(local)
            return _validate_ep_kernel_inventory_replies(
                gathered,
                world_size=self.expert_parallel_size,
            )
        except Exception as error:
            failure = RuntimeError(f"EP kernel inventory metadata probe failed: {error}")
            delegate._fatal_error = failure
            raise failure from error

    @staticmethod
    def _reject_dram_kv() -> None:
        raise RuntimeError(
            "replicated-QKV/KV EP correctness mode does not support a DRAM KV tier"
        )

    def available_prefix(self, keys: tuple[str, ...], *, min_pages: int = 1) -> int:
        self._reject_dram_kv()

    def offload(self, keys: tuple[str, ...], page_ids: tuple[int, ...]) -> bool:
        self._reject_dram_kv()

    def restore(self, keys: tuple[str, ...], page_ids: tuple[int, ...]) -> bool:
        self._reject_dram_kv()

    def discard(self, keys: tuple[str, ...]) -> bool:
        self._reject_dram_kv()


def worker_step_loop(
    control_comm,
    local_runner,
    model_comm=None,
    *,
    parallelism_prefix: str = "TP",
) -> int:
    """Non-zero-rank main loop: execute passive steps until shutdown.

    Returns the number of steps executed (spawn tests assert on it).
    """
    if model_comm is None:
        model_comm = control_comm
    steps = 0
    sync = StateSync()
    ep_kv_accounting = (
        _EPKVAccountingRecorder()
        if parallelism_prefix == "EP"
        else None
    )
    while True:
        payload = control_comm.broadcast(_SHUTDOWN, src=0)
        if isinstance(payload, _EPAttentionDPStep):
            _release_runner_requests(
                local_runner,
                payload.delta.released_ids,
            )
            view = sync.apply(payload.delta)
            fast_path = _ep_attention_dp_fast_sampling(
                payload.delta.chunks,
                view,
            )
            local = _ep_attention_dp_local_reply(
                control_comm,
                model_comm,
                local_runner,
                payload,
                view,
            )
            if fast_path:
                _ep_attention_dp_fast_status(
                    control_comm,
                    local.reply,
                )
            else:
                gathered = control_comm.all_gather(local.reply)
                _validate_ep_attention_dp_replies(
                    gathered,
                    chunks=payload.delta.chunks,
                    states=view,
                    world_size=control_comm.world_size,
                )
            steps += 1
            continue
        if not isinstance(payload, StepDelta):
            if payload is _SHUTDOWN or payload is None:
                return steps
            if isinstance(payload, _ShutdownRequest):
                _release_runner_requests(local_runner, payload.released_ids)
                sync.discard(payload.released_ids)
                return steps
            if isinstance(payload, _SamplingOwnershipProbe):
                local = _sampling_ownership_reply(
                    control_comm,
                    model_comm,
                    local_runner,
                )
                control_comm.all_gather(local)
                continue
            if isinstance(payload, _CaptureDecodeGraphs):
                _capture_decode_graphs_transaction(
                    control_comm,
                    local_runner,
                    attention_dp=payload.attention_dp,
                )
                continue
            if isinstance(payload, _EPKernelInventoryProbe):
                local = _ep_kernel_inventory_reply(control_comm, local_runner)
                control_comm.all_gather(local)
                continue
            if isinstance(payload, _EPKVAccountingControl):
                if ep_kv_accounting is None:
                    raise RuntimeError(
                        "EP KV accounting control reached a non-EP worker"
                    )
                local = _ep_kv_accounting_reply(
                    control_comm,
                    ep_kv_accounting,
                    action=payload.action,
                )
                control_comm.all_gather(local)
                continue
            if isinstance(payload, _BatchedPrefillMode):
                local_runner.set_batched_prefill_enabled(payload.enabled)
                continue
            if isinstance(payload, _PrefillStatsProbe):
                packet = _prefill_stats_packet(
                    control_comm,
                    model_comm,
                    local_runner,
                    reset=payload.reset,
                )
                model_comm.tensor_all_gather(packet)
                continue
            if isinstance(payload, _BatchedVerificationMode):
                local_runner.set_batched_verification_enabled(payload.enabled)
                continue
            if isinstance(payload, _VerificationStatsProbe):
                packet = _verification_stats_packet(
                    control_comm,
                    model_comm,
                    local_runner,
                    reset=payload.reset,
                )
                model_comm.tensor_all_gather(packet)
                continue
            if isinstance(payload, _DecodePageTableCacheMode):
                local_runner.set_decode_page_table_cache_enabled(payload.enabled)
                continue
            if isinstance(payload, _DecodePageTableCacheStatsProbe):
                packet = _page_table_stats_packet(
                    control_comm,
                    model_comm,
                    local_runner,
                    reset=payload.reset,
                )
                model_comm.tensor_all_gather(packet)
                continue
            if isinstance(
                payload,
                (_DramKVAvailablePrefix, _DramKVOffload, _DramKVRestore, _DramKVDiscard),
            ):
                local = _dram_kv_local_ack(control_comm, local_runner, payload)
                control_comm.all_gather(local)
                continue
            raise RuntimeError(
                f"{parallelism_prefix} worker received an unsupported control payload: "
                f"{type(payload).__name__}"
            )
        _release_runner_requests(local_runner, payload.released_ids)
        view = sync.apply(payload)  # same delta -> same reconstructed states
        if ep_kv_accounting is not None:
            ep_kv_accounting.observe(payload.chunks, view)
        local_runner.execute_passive(payload.chunks, view)
        packet = local_runner.make_sampling_token_packet(payload.chunks, view, sampled=None)
        packet = model_comm.tensor_broadcast(packet, src=0)
        local_runner.adopt_sampling_token_packet(payload.chunks, view, packet)
        steps += 1


@dataclass(frozen=True)
class TPPlacement:
    """Where one TP rank computes: the multi-process twin of the single-process
    ``probe()`` block in ``kairyu_backend.build_engine_loop``.

    That block never runs for ``model_path`` + ``tp > 1`` — ``build_engine_loop``
    returns into ``_build_dist_tp_loop`` before it — so without this the spawned
    ranks silently kept the CPU/fp32 defaults of ``DenseDecoder`` and
    ``PagedKVPool`` on a machine full of GPUs.
    """

    device: str
    dtype: object  # torch.dtype; annotated loosely to keep this import-light
    backend: str  # torch.distributed backend matching the device


@dataclass(frozen=True)
class _EPPlacement:
    """One rank per device for replicated-QKV/KV expert parallelism."""

    device: str
    dtype: object
    backend: str


def tp_placement(tp: int, rank: int, force_cpu: bool = False) -> TPPlacement:
    """Rank-local placement: one GPU per rank, else CPU (m8 D5 probe rules).

    ``force_cpu`` is for the CPU parity tests, which compare TP output against a
    single-process fp32 host reference and so must not follow the probe onto a
    GPU. ``KAIRYU_TP_FORCE_CPU`` is the same switch for callers that cannot pass
    the argument — notably `build_engine_loop`, whose spawned ranks read it from
    the inherited environment, so rank 0 and the workers cannot end up on
    different backends and deadlock the first collective. Deployment sets neither.
    """
    import os

    import torch

    from kairyu.engine.core.hw_profile import probe

    force_cpu = force_cpu or bool(os.environ.get("KAIRYU_TP_FORCE_CPU"))
    profile = probe()
    if force_cpu or profile.arch != "cuda":
        return TPPlacement("cpu", torch.float32, "gloo")
    if profile.device_count < tp:
        # one rank per device: overcommitting would put two shards on one GPU
        # and silently halve the memory each expects
        raise RuntimeError(
            f"tensor_parallel_size={tp} needs {tp} CUDA devices; found {profile.device_count}"
        )
    # gloo would move every RowParallelLinear all_reduce through host memory
    return TPPlacement(f"cuda:{rank}", torch.bfloat16, "nccl")


def _ep_placement(
    expert_parallel_size: int,
    rank: int,
) -> _EPPlacement:
    """Rank-local placement for the EP2/EP4 correctness path."""

    import torch

    from kairyu.engine.core.hw_profile import probe

    _validate_ep_correctness_mode(
        expert_parallel_size=expert_parallel_size,
    )
    if not 0 <= rank < expert_parallel_size:
        raise ValueError(
            f"expert-parallel rank {rank} is outside size {expert_parallel_size}"
        )
    profile = probe()
    if profile.arch != "cuda":
        raise RuntimeError(
            "replicated-QKV/KV EP correctness mode requires CUDA; "
            f"probed architecture {profile.arch!r}"
        )
    if profile.device_count < expert_parallel_size:
        raise RuntimeError(
            f"expert_parallel_size={expert_parallel_size} needs "
            f"{expert_parallel_size} CUDA devices; found {profile.device_count}"
        )
    return _EPPlacement(f"cuda:{rank}", torch.bfloat16, "nccl")


class _DeferredComm:
    """Forwards to whichever communicator is bound to it.

    The model's `RowParallelLinear` wrappers capture a communicator at build
    time, but the serving groups must not exist until every failure-prone startup
    step has succeeded — otherwise a rank-local failure leaves an UNaborted
    subgroup whose teardown waits on peers that will never arrive (review [P1] on
    #129). Building against this proxy and binding afterwards keeps both: the
    load happens on the startup group, the model runs on the serving tensor one.
    """

    def __init__(self, target) -> None:
        self._target = target

    def bind(self, target) -> None:
        self._target = target

    @property
    def group(self):
        return self._target.group

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_target"), name)


@dataclass(frozen=True)
class ServingGroups:
    """Operational groups with idle, readiness, and model work disjoint.

    ``broadcast_object_list`` is not one NCCL operation: it broadcasts metadata
    and payload tensors, then receivers copy the payload back to the host before
    deserializing it.  A source rank can therefore enqueue the following model
    all-reduce while peers are still completing the object broadcast.  Keeping
    the Python control protocol on gloo makes that hand-off blocking and leaves
    the NCCL group with tensor collectives only.  Startup graph status uses a
    second Gloo group with the same fail-fast bound as model work; the ordinary
    control group must remain able to wait for traffic indefinitely.
    """

    control: object
    startup_control: object
    model: object


def serving_group(backend: str, *, timeout_s: float = _SERVE_OP_TIMEOUT_S):
    """One process group carrying the OPERATIONAL collective timeout.

    Every rank must call this at the same point — it is itself a collective — so
    it is created right after the rendezvous, before the slow shard load, while
    the ranks are still in lockstep.
    """
    from datetime import timedelta

    import torch.distributed as dist

    return dist.new_group(timeout=timedelta(seconds=timeout_s), backend=backend)


def serving_groups(
    model_backend: str,
    *,
    control_timeout_s: float = _CONTROL_IDLE_TIMEOUT_S,
    model_timeout_s: float = _SERVE_OP_TIMEOUT_S,
) -> ServingGroups:
    """Create idle-control/readiness/model groups in the same rank order.

    The control timeout must cover the server's idle lifetime because workers
    wait *inside* its receive. Readiness and model groups have no pending
    operation while idle, so both keep the short fail-fast bound.
    """
    return ServingGroups(
        control=serving_group("gloo", timeout_s=control_timeout_s),
        startup_control=serving_group(
            "gloo",
            timeout_s=model_timeout_s,
        ),
        model=serving_group(model_backend, timeout_s=model_timeout_s),
    )


def build_tp_runner(
    model_dir: str,
    tp: int,
    rank: int,
    comm,
    num_pages: int,
    page_size: int,
    vocab: list[str] | GrammarVocabulary,
    placement: TPPlacement | None = None,
    graph_scratch_page: int | None = None,
    graph_max_batch: int = 0,
    graph_max_pages: int = 0,
    graph_warmup_iters: int = 3,
    kv_cache_dtype: str = "auto",
    dram_kv_tier_capacity_pages: int = 0,
    dram_kv_tier_profile: str | Path | None = None,
    max_num_batched_tokens: int = 2048,
):
    """The per-rank sharded PagedModelRunner (pool sized from the tp_view config).

    ``placement`` defaults to CPU/fp32 so the CPU-only callers (tests, gloo
    parity targets) are unchanged.
    """
    import torch

    from kairyu.engine.core.attention import select_backend
    from kairyu.engine.core.attention_selector import (
        attention_backend_execution_identity,
        attention_backend_identity,
    )
    from kairyu.engine.core.hw_profile import probe
    from kairyu.engine.core.kv_cache_dtype import (
        kv_cache_dtype_name,
        resolve_kv_cache_dtype,
    )
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.sampler import Sampler
    from kairyu.models.parallel import build_tp_model

    if placement is None:
        placement = TPPlacement("cpu", torch.float32, "gloo")
    profile = probe(placement.device)
    attention_backend = select_backend(
        profile,
        device=placement.device,
    )
    selected_attention_backend_identity = attention_backend_identity(
        attention_backend.selection_decision
    )
    model, local_config, full_config = build_tp_model(
        model_dir,
        tp,
        rank,
        comm,
        dtype=placement.dtype,
        device=placement.device,
        # keyed off the PLACEMENT, not the raw probe: a CPU-placed rank on a GPU
        # box would otherwise get the flashinfer kernel and hand it fp32 tensors
        attention_backend=attention_backend,
    )
    resolved_kv_cache_dtype = resolve_kv_cache_dtype(
        kv_cache_dtype,
        placement.dtype,
        profile,
        attention_backend,
        full_config,
    )
    pool = PagedKVPool(
        num_layers=local_config.num_hidden_layers,
        num_pages=num_pages,
        page_size=page_size,
        num_kv_heads=local_config.kv_cache_num_heads,
        head_dim=local_config.kv_cache_head_dim,
        dtype=resolved_kv_cache_dtype,
        device=placement.device,
    )
    dram_kv_binding = None
    if dram_kv_tier_profile is not None:
        from kairyu.engine.core.kv_tier_policy import build_dram_kv_tier

        dram_attention_backend_identity = attention_backend_execution_identity(
            attention_backend.selection_decision,
            attention_backend,
        )
        dram_kv_binding = build_dram_kv_tier(
            pool,
            model_path=model_dir,
            tensor_parallel_size=tp,
            tensor_parallel_rank=rank,
            capacity_pages=dram_kv_tier_capacity_pages,
            profile_path=dram_kv_tier_profile,
            attention_backend_identity=dram_attention_backend_identity,
            max_num_batched_tokens=max_num_batched_tokens,
        )
    grammar_vocab = (
        vocab if isinstance(vocab, GrammarVocabulary) else GrammarVocabulary(list(vocab))
    )
    graph_options = {}
    if graph_scratch_page is not None:
        from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend

        graph_options = {
            "graph_backend": CudaGraphBackend(warmup_iters=graph_warmup_iters),
            "graph_max_batch": graph_max_batch,
            "graph_max_pages": graph_max_pages,
            "graph_scratch_page": graph_scratch_page,
        }
    runner = PagedModelRunner(
        model,
        pool,
        sampler=(Sampler(vocab_provider=lambda: grammar_vocab) if rank == 0 else None),
        sampling_owner=rank == 0,
        **graph_options,
    )
    runner.attention_backend_decision = attention_backend.selection_decision
    runner.attention_backend_identity = selected_attention_backend_identity
    runner.kv_cache_dtype_requested = kv_cache_dtype
    runner.kv_cache_dtype_resolved = kv_cache_dtype_name(
        resolved_kv_cache_dtype
    )
    runner.dram_kv_binding = dram_kv_binding
    runner.dram_kv_tier = (
        dram_kv_binding.tier if dram_kv_binding is not None else None
    )
    runner.dram_kv_policy = (
        dram_kv_binding.policy if dram_kv_binding is not None else None
    )
    runner.dram_kv_tier_identity = (
        dram_kv_binding.handshake_identity
        if dram_kv_binding is not None
        else None
    )
    return runner, full_config


def build_ep_runner(
    model_dir: str,
    expert_parallel_size: int,
    rank: int,
    comm,
    num_pages: int,
    page_size: int,
    vocab: list[str] | GrammarVocabulary,
    placement: _EPPlacement | None = None,
    *,
    attention_dp: bool = False,
    pipeline_depth: int = 1,
    decode_mode: str = "eager",
    kv_cache_dtype: str = "bfloat16",
    pd_separation: bool = False,
    graph_scratch_page: int | None = None,
    graph_max_batch: int = 0,
    graph_max_pages: int = 0,
    graph_warmup_iters: int = 3,
    dram_kv_tier_capacity_pages: int = 0,
    dram_kv_tier_profile: str | Path | None = None,
    speculative: str | None = None,
):
    """Build one explicit replicated-attention or request-owned EP rank."""

    import torch

    from kairyu.engine.core.attention import select_backend
    from kairyu.engine.core.attention_selector import attention_backend_identity
    from kairyu.engine.core.hw_profile import probe
    from kairyu.engine.core.kv_cache_dtype import (
        kv_cache_dtype_name,
        resolve_kv_cache_dtype,
    )
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.sampler import Sampler
    from kairyu.models.moe_parallel import build_ep_model

    _validate_ep_mode(
        attention_dp=attention_dp,
        expert_parallel_size=expert_parallel_size,
        pipeline_depth=pipeline_depth,
        decode_mode=decode_mode,
        kv_cache_dtype=kv_cache_dtype,
        pd_separation=pd_separation,
        graph_scratch_page=graph_scratch_page,
        graph_max_batch=graph_max_batch,
        graph_max_pages=graph_max_pages,
        graph_warmup_iters=graph_warmup_iters,
        dram_kv_tier_capacity_pages=dram_kv_tier_capacity_pages,
        dram_kv_tier_profile=dram_kv_tier_profile,
        speculative=speculative,
    )
    if not 0 <= rank < expert_parallel_size:
        raise ValueError(
            f"expert-parallel rank {rank} is outside size {expert_parallel_size}"
        )
    if placement is None:
        placement = _ep_placement(expert_parallel_size, rank)
    if (
        placement.backend != "nccl"
        or not str(placement.device).startswith("cuda:")
        or placement.dtype is not torch.bfloat16
    ):
        raise ValueError(
            "expert parallelism requires a CUDA/NCCL BF16 placement"
        )
    profile = probe(placement.device)
    attention_backend = select_backend(profile, device=placement.device)
    selected_attention_backend_identity = attention_backend_identity(
        attention_backend.selection_decision
    )
    model, full_config, load_info = build_ep_model(
        model_dir,
        expert_parallel_size,
        rank,
        comm,
        dtype=placement.dtype,
        device=placement.device,
        attention_backend=attention_backend,
        attention_dp=attention_dp,
    )
    resolved_kv_cache_dtype = resolve_kv_cache_dtype(
        kv_cache_dtype,
        placement.dtype,
        profile,
        attention_backend,
        full_config,
    )
    if resolved_kv_cache_dtype is not torch.bfloat16:
        raise RuntimeError(
            "expert-parallel execution resolved a non-BF16 KV cache"
        )
    graph_decode = decode_mode == "cuda_graph"
    decode_scratch_capacity = (
        graph_max_batch if graph_decode else 1
    ) if attention_dp else 0
    attention_dp_reserved_pages = (
        _EP_ATTENTION_DP_SCRATCH_PAGES + decode_scratch_capacity
        if attention_dp
        else 0
    )
    expected_graph_scratch_page = (
        num_pages + attention_dp_reserved_pages
        if attention_dp and graph_decode
        else None
    )
    if graph_scratch_page != expected_graph_scratch_page:
        raise ValueError(
            "attention-DP graph scratch page does not match the reserved "
            f"physical layout: got {graph_scratch_page!r}, "
            f"expected {expected_graph_scratch_page!r}"
        )
    physical_num_pages = (
        num_pages
        + attention_dp_reserved_pages
        + int(expected_graph_scratch_page is not None)
    )
    pool = PagedKVPool(
        num_layers=full_config.num_hidden_layers,
        num_pages=physical_num_pages,
        page_size=page_size,
        num_kv_heads=full_config.kv_cache_num_heads,
        head_dim=full_config.kv_cache_head_dim,
        dtype=resolved_kv_cache_dtype,
        device=placement.device,
    )
    grammar_vocab = (
        vocab if isinstance(vocab, GrammarVocabulary) else GrammarVocabulary(list(vocab))
    )
    graph_options: dict[str, object] = {}
    if graph_decode:
        from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend

        graph_options = {
            "graph_backend": CudaGraphBackend(warmup_iters=graph_warmup_iters),
            "graph_max_batch": graph_max_batch,
            "graph_max_pages": graph_max_pages,
            "graph_scratch_page": graph_scratch_page,
        }
    runner = PagedModelRunner(
        model,
        pool,
        sampler=(
            Sampler(vocab_provider=lambda: grammar_vocab)
            if attention_dp or rank == 0
            else None
        ),
        sampling_owner=attention_dp or rank == 0,
        **graph_options,
    )
    if attention_dp:
        runner.set_unified_mixed_enabled(False)
        _validate_attention_dp_batched_prefill_runner(runner)
    runner.attention_backend_decision = attention_backend.selection_decision
    runner.attention_backend_identity = selected_attention_backend_identity
    runner.kv_cache_dtype_requested = "bfloat16"
    runner.kv_cache_dtype_resolved = kv_cache_dtype_name(
        resolved_kv_cache_dtype
    )
    runner.dram_kv_binding = None
    runner.dram_kv_tier = None
    runner.dram_kv_policy = None
    runner.dram_kv_tier_identity = None
    runner.parallelism = "expert_parallel"
    runner.expert_parallel_size = expert_parallel_size
    runner.expert_parallel_rank = rank
    runner.attention_placement = (
        "request_owned_data_parallel" if attention_dp else "replicated"
    )
    runner.attention_output_placement = (
        "replicated" if attention_dp else "row_parallel"
    )
    runner.attention_output_parallel_size = (
        1 if attention_dp else expert_parallel_size
    )
    runner.attention_output_partial_dtype = (
        None if attention_dp else "bfloat16"
    )
    runner.execution_mode = (
        _EP_ATTENTION_DP_MODE if attention_dp else _EP_CORRECTNESS_MODE
    )
    runner.pipeline_depth = pipeline_depth
    runner.decode_mode = decode_mode
    runner.attention_data_parallel_size = (
        expert_parallel_size if attention_dp else 1
    )
    runner.attention_tensor_parallel_size = 1
    runner.moe_dispatcher = (
        _EP_ATTENTION_DP_DISPATCHER if attention_dp else "replicated_allreduce"
    )
    runner.attention_dp_scratch_pages = (
        (num_pages, num_pages + 1) if attention_dp else None
    )
    runner.attention_dp_decode_scratch_pages = (
        tuple(
            range(
                num_pages + _EP_ATTENTION_DP_SCRATCH_PAGES,
                num_pages
                + _EP_ATTENTION_DP_SCRATCH_PAGES
                + decode_scratch_capacity,
            )
        )
        if attention_dp
        else None
    )
    runner.attention_dp_graph_scratch_page = expected_graph_scratch_page
    runner.attention_dp_reserved_page_count = attention_dp_reserved_pages
    runner.usable_kv_pages = num_pages
    runner.physical_kv_pages = physical_num_pages
    runner.expert_parallel_load_info = load_info
    return runner, full_config, load_info


def _tp_worker_entry(
    spawn_index: int,
    world_size: int,
    init_file: str,
    model_dir: str,
    num_pages: int,
    page_size: int,
    vocab: list[str] | GrammarVocabulary,
    force_cpu: bool = False,
    graph_scratch_page: int | None = None,
    graph_max_batch: int = 0,
    graph_max_pages: int = 0,
    graph_warmup_iters: int = 3,
    kv_cache_dtype: str = "auto",
    dram_kv_tier_capacity_pages: int = 0,
    dram_kv_tier_profile: str | Path | None = None,
    max_num_batched_tokens: int = 2048,
) -> None:
    """Spawned worker (rank = spawn_index + 1; rank 0 is the driver process).

    Module-level and side-effect-free at import (m16 A6) so torch spawn can
    pickle it. Joins the group, validates the handshake, runs the step loop
    until rank 0 broadcasts shutdown, then tears the group down."""
    import torch

    from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed

    rank = spawn_index + 1
    torch.set_num_threads(1)
    placement = tp_placement(world_size, rank, force_cpu)
    if placement.backend == "nccl":
        # must precede init_process_group: NCCL binds the rank to the current
        # device, and object collectives stage their buffers on it
        torch.cuda.set_device(rank)
    init_distributed(
        rank,
        world_size,
        f"file://{init_file}",
        backend=placement.backend,
        timeout_s=_STARTUP_TIMEOUT_S,
    )
    startup_comm = TorchDistCommunicator(device=placement.device)
    comm = _DeferredComm(startup_comm)
    runner, _ = build_tp_runner(
        model_dir,
        world_size,
        rank,
        comm,
        num_pages,
        page_size,
        vocab,
        placement,
        graph_scratch_page,
        graph_max_batch,
        graph_max_pages,
        graph_warmup_iters,
        kv_cache_dtype,
        dram_kv_tier_capacity_pages,
        dram_kv_tier_profile,
        max_num_batched_tokens,
    )
    # the handshake is the collective that absorbs load skew, so it — and only it
    # — runs on the long-timeout startup group
    handshake = startup_comm.broadcast(None, src=0)
    validate_handshake(
        handshake,
        model_dir,
        num_pages,
        page_size,
        runner.attention_backend_identity,
        runner.kv_cache_dtype_requested,
        runner.kv_cache_dtype_resolved,
        runner.dram_kv_tier_identity,
    )
    groups = serving_groups(placement.backend)
    comm.bind(TorchDistCommunicator(group=groups.model, device=placement.device))
    control_comm = TorchDistCommunicator(group=groups.control)
    startup_control_comm = TorchDistCommunicator(group=groups.startup_control)
    try:
        if graph_scratch_page is not None:
            _capture_decode_graphs_transaction(
                startup_control_comm,
                runner,
                attention_dp=False,
            )
        worker_step_loop(control_comm, runner, comm)
    finally:
        import torch.distributed as dist

        invalidate = getattr(runner, "invalidate_graphs", None)
        if invalidate is not None:
            invalidate()
        if placement.backend == "nccl":
            # Captured NCCL collectives retain graph-owned communicator work
            # until the graph objects are released and their stream is drained.
            # Drain it, then rendezvous on the model communicator so no rank
            # destroys that communicator while a peer is still releasing its
            # captured work.
            torch.cuda.synchronize()
            comm.barrier()
        dist.destroy_process_group(comm.group)
        dist.destroy_process_group(startup_control_comm.group)
        dist.destroy_process_group(control_comm.group)
        dist.destroy_process_group()


def _ep_worker_entry(
    spawn_index: int,
    expert_parallel_size: int,
    init_file: str,
    model_dir: str,
    num_pages: int,
    page_size: int,
    vocab: list[str] | GrammarVocabulary,
    attention_dp: bool = False,
    pipeline_depth: int = 1,
    decode_mode: str = "eager",
    graph_scratch_page: int | None = None,
    graph_max_batch: int = 0,
    graph_max_pages: int = 0,
    graph_warmup_iters: int = 3,
) -> None:
    """Spawned EP worker; rank 0 remains in the driver process."""

    import torch

    from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed

    _validate_ep_mode(
        attention_dp=attention_dp,
        expert_parallel_size=expert_parallel_size,
        pipeline_depth=pipeline_depth,
        decode_mode=decode_mode,
        graph_scratch_page=graph_scratch_page,
        graph_max_batch=graph_max_batch,
        graph_max_pages=graph_max_pages,
        graph_warmup_iters=graph_warmup_iters,
    )
    rank = spawn_index + 1
    torch.set_num_threads(1)
    placement = _ep_placement(expert_parallel_size, rank)
    if placement.backend == "nccl":
        torch.cuda.set_device(rank)
    init_distributed(
        rank,
        expert_parallel_size,
        f"file://{init_file}",
        backend=placement.backend,
        timeout_s=_STARTUP_TIMEOUT_S,
    )
    startup_comm = TorchDistCommunicator(device=placement.device)
    comm = _DeferredComm(startup_comm)
    runner, _, _ = build_ep_runner(
        model_dir,
        expert_parallel_size,
        rank,
        comm,
        num_pages,
        page_size,
        vocab,
        placement,
        attention_dp=attention_dp,
        pipeline_depth=pipeline_depth,
        decode_mode=decode_mode,
        graph_scratch_page=graph_scratch_page,
        graph_max_batch=graph_max_batch,
        graph_max_pages=graph_max_pages,
        graph_warmup_iters=graph_warmup_iters,
    )
    if (
        runner.expert_parallel_size != expert_parallel_size
        or runner.expert_parallel_rank != rank
    ):
        raise RuntimeError(
            "EP rank-local runner ownership does not match its process identity"
        )
    handshake = startup_comm.broadcast(None, src=0)
    _validate_ep_handshake(
        handshake,
        model_dir,
        expert_parallel_size,
        num_pages,
        page_size,
        runner.attention_backend_identity,
        runner.kv_cache_dtype_resolved,
        attention_dp=attention_dp,
        pipeline_depth=pipeline_depth,
        decode_mode=decode_mode,
        graph_max_batch=graph_max_batch,
        graph_max_pages=graph_max_pages,
        graph_warmup_iters=graph_warmup_iters,
    )
    groups = serving_groups(placement.backend)
    control_comm = TorchDistCommunicator(group=groups.control)
    startup_control_comm = TorchDistCommunicator(group=groups.startup_control)
    model_comm = TorchDistCommunicator(group=groups.model, device=placement.device)
    clean_shutdown = False
    try:
        _bind_ep_model_communicator(
            comm,
            model_comm,
            startup_control_comm,
            attention_dp=attention_dp,
        )
        if decode_mode == "cuda_graph":
            _capture_decode_graphs_transaction(
                startup_control_comm,
                runner,
                attention_dp=attention_dp,
            )
        worker_step_loop(
            control_comm,
            runner,
            comm,
            parallelism_prefix="EP",
        )
        clean_shutdown = True
    finally:
        import contextlib

        import torch.distributed as dist

        invalidate = getattr(runner, "invalidate_graphs", None)
        invalidation_error: Exception | None = None
        if callable(invalidate):
            try:
                invalidate()
            except Exception as error:
                if clean_shutdown:
                    invalidation_error = error
        if not clean_shutdown:
            _abort_ep_worker_model_communicator(comm, placement.device)
        else:
            if placement.backend == "nccl":
                torch.cuda.synchronize()
                comm.barrier()
            with contextlib.suppress(Exception):
                comm.close_direct_nccl()
            dist.destroy_process_group(comm.group)
            dist.destroy_process_group(startup_control_comm.group)
            dist.destroy_process_group(control_comm.group)
            dist.destroy_process_group()
            if invalidation_error is not None:
                raise invalidation_error


class DistTPLauncher:
    """Owns the spawned worker processes + the rank-0 DistTPModelRunner.

    Wires real multi-process TP into a single-process serve path: rank 0 lives in
    THIS process, ranks 1..tp-1 are spawned workers. ``shutdown()`` broadcasts the
    terminating None (worker_step_loop returns), joins the workers, and destroys
    the rank-0 group — so ``kairyu serve --tp N`` starts and stops cleanly."""

    def __init__(
        self,
        model_dir: str,
        tp: int,
        num_pages: int,
        page_size: int,
        vocab: list[str] | GrammarVocabulary,
        force_cpu: bool = False,
        graph_scratch_page: int | None = None,
        graph_max_batch: int = 0,
        graph_max_pages: int = 0,
        graph_warmup_iters: int = 3,
        kv_cache_dtype: str = "auto",
        dram_kv_tier_capacity_pages: int = 0,
        dram_kv_tier_profile: str | Path | None = None,
        max_num_batched_tokens: int = 2048,
    ) -> None:
        import tempfile

        import torch
        import torch.multiprocessing as mp

        from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed
        from kairyu.engine.core.kv_cache_dtype import validate_kv_cache_dtype

        # Retain the constructed degree for process-boundary startup
        # attestation.  In the in-process backend the parent already owns this
        # value; kairyu-proc must receive it back from the child after all ranks
        # have actually started before it may advertise TP readiness.
        self.tensor_parallel_size = tp

        kv_cache_dtype = validate_kv_cache_dtype(kv_cache_dtype)
        if kv_cache_dtype != "auto" and force_cpu:
            raise ValueError(
                "explicit kv_cache_dtype cannot use forced CPU TP placement"
            )
        if (
            type(dram_kv_tier_capacity_pages) is not int
            or dram_kv_tier_capacity_pages < 0
        ):
            raise ValueError(
                "dram_kv_tier_capacity_pages must be a non-negative integer"
            )
        if dram_kv_tier_profile is not None and not isinstance(
            dram_kv_tier_profile, (str, Path)
        ):
            raise ValueError("dram_kv_tier_profile must be a local path or null")
        if dram_kv_tier_profile is not None and not str(
            dram_kv_tier_profile
        ):
            raise ValueError(
                "dram_kv_tier_profile must be a non-empty local path or null"
            )
        if (dram_kv_tier_capacity_pages > 0) != (
            dram_kv_tier_profile is not None
        ):
            raise ValueError(
                "DRAM KV tier requires both a positive capacity and profile"
            )

        # a fresh, not-yet-created path is the file:// rendezvous point
        self._init_file = tempfile.mktemp(prefix="kairyu-tp-")  # noqa: S306
        placement = tp_placement(tp, 0, force_cpu)
        self._placement_backend = placement.backend
        if graph_scratch_page is not None and placement.backend != "nccl":
            raise ValueError("CUDA graph decode needs CUDA/NCCL TP placement")
        # force_cpu travels to the workers: rank 0 on host memory while the
        # spawned ranks probed their way onto GPUs would deadlock the first
        # all_reduce on mismatched backends
        self._ctx = mp.spawn(
            _tp_worker_entry,
            args=(
                tp,
                self._init_file,
                model_dir,
                num_pages,
                page_size,
                vocab,
                force_cpu,
                graph_scratch_page,
                graph_max_batch,
                graph_max_pages,
                graph_warmup_iters,
                kv_cache_dtype,
                dram_kv_tier_capacity_pages,
                dram_kv_tier_profile,
                max_num_batched_tokens,
            ),
            nprocs=tp - 1,
            join=False,
        )
        # Everything past the spawn can raise — an indivisible TP degree, a
        # missing tensor, not enough GPUs. Without this the workers and the
        # process group outlive the failed constructor: the caller sees the real
        # error, then a "destroy_process_group() was not called" warning, and the
        # next launcher in the same process cannot rendezvous at all.
        try:
            if placement.backend == "nccl":
                torch.cuda.set_device(0)
            init_distributed(
                0,
                tp,
                f"file://{self._init_file}",
                backend=placement.backend,
                timeout_s=_STARTUP_TIMEOUT_S,
            )
            startup_comm = TorchDistCommunicator(device=placement.device)
            # the model is built against the startup group; nothing that can fail
            # may leave a serving subgroup behind for _abandon_start to miss
            self._comm = _DeferredComm(startup_comm)
            runner, self.full_config = build_tp_runner(
                model_dir,
                tp,
                0,
                self._comm,
                num_pages,
                page_size,
                vocab,
                placement,
                graph_scratch_page,
                graph_max_batch,
                graph_max_pages,
                graph_warmup_iters,
                kv_cache_dtype,
                dram_kv_tier_capacity_pages,
                dram_kv_tier_profile,
                max_num_batched_tokens,
            )
            self.attention_backend_decision = runner.attention_backend_decision
            self.attention_backend_identity = runner.attention_backend_identity
            self.kv_cache_dtype_requested = runner.kv_cache_dtype_requested
            self.kv_cache_dtype_resolved = runner.kv_cache_dtype_resolved
            self.dram_kv_binding = runner.dram_kv_binding
            self.dram_kv_tier_identity = runner.dram_kv_tier_identity
            # the one collective that legitimately absorbs load skew
            startup_comm.broadcast(
                make_handshake(
                    model_dir,
                    num_pages,
                    page_size,
                    self.attention_backend_identity,
                    self.kv_cache_dtype_requested,
                    self.kv_cache_dtype_resolved,
                    self.dram_kv_tier_identity,
                ),
                src=0,
            )
            # Every failure-prone step is done: now the step loop gets bounded
            # operational groups.  Python state deltas stay on gloo; the model
            # wrappers use only the tensor/NCCL group.
            groups = serving_groups(placement.backend)
            self._comm.bind(TorchDistCommunicator(group=groups.model, device=placement.device))
            self._control_comm = TorchDistCommunicator(group=groups.control)
            self._startup_control_comm = TorchDistCommunicator(
                group=groups.startup_control
            )
            self.runner = DistTPModelRunner(self._control_comm, runner, self._comm)
            if graph_scratch_page is not None:
                _capture_decode_graphs_transaction(
                    self._startup_control_comm,
                    runner,
                    attention_dp=False,
                )
        except BaseException:
            self._abandon_start()
            raise

    def _abandon_start(self) -> None:
        """Tear down a half-built group without waiting on it.

        Order is the OPPOSITE of the normal shutdown's. There the ranks are
        healthy and destroy is a rendezvous; here they are dead of the same error
        or stuck in a collective nobody will complete, so `destroy_process_group`
        — which every rank must reach — can BLOCK rather than return the original
        error. `contextlib.suppress` catches an exception but cannot bound that
        (review [P1] on #129).

        So the communicator is aborted first, which is non-collective; after that
        nothing can block, and the workers are terminated and reaped. Every step
        is best-effort: this runs while an exception is in flight and must not
        replace it with one of its own.
        """
        import contextlib
        import os

        import torch.distributed as dist

        # A rank can finish startup capture successfully while a peer reports
        # a final capture/sync/layout failure.  Drop rank 0's graph objects
        # before aborting their NCCL communicator: graph executables retain
        # communicator registrations even though the distributed runner has
        # already been marked fatal.  Reach through the EP wrapper deliberately
        # here; its public invalidation guard protects live serving, whereas an
        # abandoned launcher can never serve another step.
        distributed_runner = getattr(self, "runner", None)
        local_runner = getattr(distributed_runner, "_local", None)
        invalidate = getattr(local_runner, "invalidate_graphs", None)
        if callable(invalidate):
            with contextlib.suppress(Exception):
                invalidate()
        if dist.is_initialized():
            with contextlib.suppress(Exception):
                self._abort_communicator()
        # Reap peers before any potentially rendezvous-like process-group
        # destruction. This ordering remains bounded even if an older backend
        # lacks the private abort hook or abort itself raises.
        for process in self._ctx.processes:
            if process.is_alive():
                process.terminate()
        for process in self._ctx.processes:
            with contextlib.suppress(Exception):
                process.join(timeout=10)
            if process.is_alive():  # pragma: no cover - terminate was ignored
                with contextlib.suppress(Exception):
                    process.kill()
                    process.join(timeout=5)
        if dist.is_initialized():
            with contextlib.suppress(Exception):
                dist.destroy_process_group()
        with contextlib.suppress(OSError):
            os.unlink(self._init_file)

    def _abort_communicator(self) -> None:
        """NCCL only: drop every live communicator without waiting for peers.

        The operational model subgroup is a distinct NCCL communicator from the
        startup/default group. Aborting only the latter leaves a worker stuck in
        the sampling-token broadcast when rank 0 fails after model execution.
        Gloo needs no abort, and the hook is absent on older torch — both are
        "nothing to abort" rather than an error.
        """
        import contextlib

        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available():
            return
        groups = []
        model_comm = getattr(self, "_comm", None)
        if model_comm is not None:
            with contextlib.suppress(Exception):
                model_comm.abort_direct_nccl()
            groups.append(model_comm.group)
        groups.append(dist.distributed_c10d._get_default_group())
        seen: set[int] = set()
        device = torch.device("cuda", torch.cuda.current_device())
        for group in groups:
            if id(group) in seen:
                continue
            seen.add(id(group))
            try:
                backend = group._get_backend(device)
            except Exception:
                continue
            abort = getattr(backend, "abort", None) or getattr(backend, "_abort", None)
            if abort is not None:
                abort()

    def dead_ranks(self) -> tuple[int, ...]:
        """Spawned ranks that are no longer running (rank 0 is this process).

        A dead rank leaves the group unable to complete a single collective, but
        rank 0 stays up and keeps answering health checks — on hardware this
        presented as a served model that accepted requests and never returned a
        token. Cheap enough for `/readyz`: `is_alive()` is a waitpid, no IPC.
        """
        return tuple(
            index + 1 for index, process in enumerate(self._ctx.processes) if not process.is_alive()
        )

    def failure_type(self) -> str | None:
        """Fatal distributed step failure, sanitized for health reporting."""
        error = self.runner.fatal_error
        return type(error).__name__ if error is not None else None

    def shutdown(self) -> None:
        import contextlib
        import os

        import torch
        import torch.distributed as dist

        if self.runner.fatal_error is not None:
            # The collective sequence is already untrustworthy. A graceful
            # broadcast/barrier can only hang; abort and reap like failed startup.
            self._abandon_start()
            return
        self.runner.shutdown()  # broadcasts None -> workers leave worker_step_loop
        self.runner.invalidate_graphs()
        if self._placement_backend == "nccl":
            torch.cuda.synchronize()
            # Match the worker-side rendezvous after every rank has dropped its
            # CUDA graphs.  Without this, TP graph serving can complete all
            # inference steps and then hang forever in process-group teardown.
            self._comm.barrier()
        with contextlib.suppress(Exception):
            self._comm.close_direct_nccl()
        # BEFORE the join, not after: NCCL's destroy_process_group waits for every
        # rank to reach it, so joining first deadlocks rank 0 against workers that
        # are already sitting in their own destroy. gloo never blocks here, which
        # is why the CPU parity gates could not see this.
        if dist.is_initialized():
            # Multiple process groups must be destroyed explicitly in the same
            # order on every rank.  Reverse creation order keeps the graph-owning
            # NCCL subgroup ahead of the gloo control and startup groups.
            dist.destroy_process_group(self._comm.group)
            dist.destroy_process_group(self._startup_control_comm.group)
            dist.destroy_process_group(self._control_comm.group)
            dist.destroy_process_group()
        self._ctx.join()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self._init_file)


class _DistLauncherLifecycle:
    """Share proven process-group cleanup without making EP a TP subtype."""

    _abandon_start = DistTPLauncher._abandon_start
    _abort_communicator = DistTPLauncher._abort_communicator
    dead_ranks = DistTPLauncher.dead_ranks
    failure_type = DistTPLauncher.failure_type
    shutdown = DistTPLauncher.shutdown


class DistEPLauncher(_DistLauncherLifecycle):
    """Own one explicit EP2/EP4 replicated or attention-DP process group.

    This is a distinct public topology from :class:`DistTPLauncher`. Every rank
    holds complete QKV, attention-core, and KV state. ``build_ep_model`` loads
    its rank's attention-output K shard and routed experts. The constructor
    validates the deliberately narrow execution envelope before spawning.
    """

    def __init__(
        self,
        model_dir: str,
        expert_parallel_size: int,
        num_pages: int,
        page_size: int,
        vocab: list[str] | GrammarVocabulary,
        *,
        attention_dp: bool = False,
        pipeline_depth: int = 1,
        decode_mode: str = "eager",
        kv_cache_dtype: str = "bfloat16",
        pd_separation: bool = False,
        graph_scratch_page: int | None = None,
        graph_max_batch: int = 0,
        graph_max_pages: int = 0,
        graph_warmup_iters: int = 3,
        dram_kv_tier_capacity_pages: int = 0,
        dram_kv_tier_profile: str | Path | None = None,
        speculative: str | None = None,
    ) -> None:
        import tempfile

        import torch
        import torch.multiprocessing as mp

        from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed

        _validate_ep_mode(
            attention_dp=attention_dp,
            expert_parallel_size=expert_parallel_size,
            pipeline_depth=pipeline_depth,
            decode_mode=decode_mode,
            kv_cache_dtype=kv_cache_dtype,
            pd_separation=pd_separation,
            graph_scratch_page=graph_scratch_page,
            graph_max_batch=graph_max_batch,
            graph_max_pages=graph_max_pages,
            graph_warmup_iters=graph_warmup_iters,
            dram_kv_tier_capacity_pages=dram_kv_tier_capacity_pages,
            dram_kv_tier_profile=dram_kv_tier_profile,
            speculative=speculative,
        )
        if attention_dp and decode_mode == "cuda_graph":
            expected_graph_scratch = (
                num_pages + _EP_ATTENTION_DP_SCRATCH_PAGES + graph_max_batch
            )
            if graph_scratch_page != expected_graph_scratch:
                raise ValueError(
                    "request-owned attention-DP graph_scratch_page must follow "
                    "the scheduler-invisible prefill/decode reservations: "
                    f"got {graph_scratch_page!r}, expected "
                    f"{expected_graph_scratch}"
                )
            if graph_max_pages > num_pages:
                raise ValueError(
                    "request-owned attention-DP graph_max_pages must not exceed "
                    "scheduler-visible num_pages"
                )
        self.expert_parallel_size = expert_parallel_size
        self.attention_dp = attention_dp
        self.parallelism = "expert_parallel"
        self.execution_mode = (
            _EP_ATTENTION_DP_MODE if attention_dp else _EP_CORRECTNESS_MODE
        )
        self.attention_placement = (
            "request_owned_data_parallel" if attention_dp else "replicated"
        )
        self.attention_output_placement = (
            "replicated" if attention_dp else "row_parallel"
        )
        self.attention_output_parallel_size = (
            1 if attention_dp else expert_parallel_size
        )
        self.attention_output_partial_dtype = (
            None if attention_dp else "bfloat16"
        )
        self.pipeline_depth = pipeline_depth
        self.decode_mode = decode_mode
        self._init_file = tempfile.mktemp(prefix="kairyu-ep-")  # noqa: S306
        placement = _ep_placement(expert_parallel_size, 0)
        self._placement_backend = placement.backend
        self._ctx = mp.spawn(
            _ep_worker_entry,
            args=(
                expert_parallel_size,
                self._init_file,
                model_dir,
                num_pages,
                page_size,
                vocab,
                attention_dp,
                pipeline_depth,
                decode_mode,
                graph_scratch_page,
                graph_max_batch,
                graph_max_pages,
                graph_warmup_iters,
            ),
            nprocs=expert_parallel_size - 1,
            join=False,
        )
        try:
            if placement.backend == "nccl":
                torch.cuda.set_device(0)
            init_distributed(
                0,
                expert_parallel_size,
                f"file://{self._init_file}",
                backend=placement.backend,
                timeout_s=_STARTUP_TIMEOUT_S,
            )
            startup_comm = TorchDistCommunicator(device=placement.device)
            self._comm = _DeferredComm(startup_comm)
            runner, self.full_config, self.expert_parallel_load_info = build_ep_runner(
                model_dir,
                expert_parallel_size,
                0,
                self._comm,
                num_pages,
                page_size,
                vocab,
                placement,
                attention_dp=attention_dp,
                pipeline_depth=pipeline_depth,
                decode_mode=decode_mode,
                kv_cache_dtype=kv_cache_dtype,
                pd_separation=pd_separation,
                graph_scratch_page=graph_scratch_page,
                graph_max_batch=graph_max_batch,
                graph_max_pages=graph_max_pages,
                graph_warmup_iters=graph_warmup_iters,
                dram_kv_tier_capacity_pages=dram_kv_tier_capacity_pages,
                dram_kv_tier_profile=dram_kv_tier_profile,
                speculative=speculative,
            )
            self.attention_backend_decision = runner.attention_backend_decision
            self.attention_backend_identity = runner.attention_backend_identity
            self.kv_cache_dtype_requested = runner.kv_cache_dtype_requested
            self.kv_cache_dtype_resolved = runner.kv_cache_dtype_resolved
            self.dram_kv_binding = None
            self.dram_kv_tier_identity = None
            startup_comm.broadcast(
                _make_ep_handshake(
                    model_dir,
                    expert_parallel_size,
                    num_pages,
                    page_size,
                    self.attention_backend_identity,
                    self.kv_cache_dtype_resolved,
                    attention_dp=attention_dp,
                    pipeline_depth=pipeline_depth,
                    decode_mode=decode_mode,
                    graph_max_batch=graph_max_batch,
                    graph_max_pages=graph_max_pages,
                    graph_warmup_iters=graph_warmup_iters,
                ),
                src=0,
            )
            groups = serving_groups(placement.backend)
            self._control_comm = TorchDistCommunicator(group=groups.control)
            self._startup_control_comm = TorchDistCommunicator(
                group=groups.startup_control
            )
            model_comm = TorchDistCommunicator(
                group=groups.model,
                device=placement.device,
            )
            _bind_ep_model_communicator(
                self._comm,
                model_comm,
                self._startup_control_comm,
                attention_dp=attention_dp,
            )
            self.runner = DistEPModelRunner(
                self._control_comm,
                runner,
                self._comm,
                expert_parallel_size=expert_parallel_size,
                attention_dp=attention_dp,
                pipeline_depth=pipeline_depth,
                page_size=page_size,
                decode_mode=decode_mode,
                graph_scratch_page=graph_scratch_page,
                graph_max_batch=graph_max_batch,
                graph_max_pages=graph_max_pages,
                graph_warmup_iters=graph_warmup_iters,
            )
            if decode_mode == "cuda_graph":
                _capture_decode_graphs_transaction(
                    self._startup_control_comm,
                    runner,
                    attention_dp=attention_dp,
                )
        except BaseException:
            self._abandon_start()
            raise

    def parallelism_metadata(self) -> dict[str, object]:
        return self.runner.parallelism_metadata()

    def ep_kernel_inventory_metadata(self) -> tuple[dict[str, object], ...]:
        """Return the rank-complete runtime inventory from the EP runner."""

        return self.runner.ep_kernel_inventory_metadata()
