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
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from threading import Lock

from kairyu.engine.core.step_input import StateSync, StepDelta
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


@dataclass(frozen=True)
class _ShutdownRequest:
    """Terminal control payload carrying cleanup not consumed by another step."""

    released_ids: tuple[str, ...] = ()


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
class _EPKernelInventoryProbe:
    """Out-of-band request for rank-local EP NVFP4 kernel metadata."""


@dataclass(frozen=True)
class _EPKernelInventoryReply:
    """Pickle-safe EP probe reply that keeps every rank in the gather."""

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


_EP_KERNEL_INVENTORY_FIELDS = frozenset(
    {
        "rank",
        "projection_count",
        "projection_inventory_sha256",
        "kernel_counts",
        "meta_count",
        "load_info",
    }
)
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


def _ep_kernel_inventory_row(
    control_comm,
    local_runner,
) -> dict[str, object]:
    """Measure one local model while retaining only a names digest."""

    from kairyu.quant.linear import NvFp4Linear

    rank = control_comm.rank
    world_size = control_comm.world_size
    if type(rank) is not int or type(world_size) is not int or world_size < 1:
        raise RuntimeError(
            "EP kernel inventory communicator metadata must be positive integers"
        )
    model = getattr(local_runner, "_model", None)
    if model is None or not callable(getattr(model, "named_modules", None)):
        raise RuntimeError(f"rank {rank} runner does not expose its local model")

    names: list[str] = []
    kernel_counts: dict[str, int] = {}
    for name, module in model.named_modules():
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
        key = f"{method}:{kernel}"
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
    return {
        "rank": rank,
        "projection_count": len(names),
        "projection_inventory_sha256": _json_sha256(names),
        "kernel_counts": dict(sorted(kernel_counts.items())),
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
        by_rank[rank] = row
    expected = set(range(world_size))
    if set(by_rank) != expected:
        raise RuntimeError(
            "EP kernel inventory ranks are incomplete: "
            f"expected={sorted(expected)}, got={sorted(by_rank)}"
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
) -> tuple[dict[str, object], ...]:
    """Reject incomplete or forged topology evidence with rank-specific errors."""
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
        expected_owner = rank == 0
        if (
            row["sampling_owner"] is not expected_owner
            or row["sampler_present"] is not expected_owner
        ):
            raise RuntimeError(
                f"sampling ownership reply for rank {rank} violates the "
                "rank-0-only sampler contract: "
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
_EP_SUPPORTED_SIZES = frozenset({2, 4})
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
    dram_kv_tier_capacity_pages: object = 0,
    dram_kv_tier_profile: object = None,
    speculative: object = None,
) -> None:
    """Fail closed outside the first production EP correctness envelope.

    This path intentionally replicates attention/KV state on every rank and
    shards only routed experts. It establishes correctness for EP2/EP4; it is
    not the attention-DP or grouped-expert throughput path.
    """

    if (
        type(expert_parallel_size) is not int
        or expert_parallel_size not in _EP_SUPPORTED_SIZES
    ):
        supported = ", ".join(str(size) for size in sorted(_EP_SUPPORTED_SIZES))
        raise ValueError(
            "replicated-attention EP correctness mode supports only "
            f"expert_parallel_size in {{{supported}}}; got {expert_parallel_size!r}"
        )
    if type(pipeline_depth) is not int or pipeline_depth != 1:
        raise ValueError(
            "replicated-attention EP correctness mode requires pipeline_depth=1"
        )
    if decode_mode != "eager":
        raise ValueError(
            "replicated-attention EP correctness mode requires decode_mode='eager'; "
            "CUDA graph decode is not supported"
        )
    if kv_cache_dtype != "bfloat16":
        raise ValueError(
            "replicated-attention EP correctness mode requires "
            "kv_cache_dtype='bfloat16'"
        )
    if type(pd_separation) is not bool:
        raise TypeError("pd_separation must be a boolean")
    if pd_separation:
        raise ValueError(
            "replicated-attention EP correctness mode does not support P-D separation"
        )
    if graph_scratch_page is not None:
        raise ValueError(
            "replicated-attention EP correctness mode does not support CUDA graphs"
        )
    if (
        type(dram_kv_tier_capacity_pages) is not int
        or dram_kv_tier_capacity_pages != 0
        or dram_kv_tier_profile is not None
    ):
        raise ValueError(
            "replicated-attention EP correctness mode does not support a DRAM KV tier"
        )
    if speculative is not None:
        raise ValueError(
            "replicated-attention EP correctness mode does not support speculative decoding"
        )


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
) -> dict[str, object]:
    """Exact all-rank identity for replicated-attention EP correctness mode."""

    _validate_ep_correctness_mode(expert_parallel_size=expert_parallel_size)
    handshake = make_handshake(
        model_dir,
        num_pages,
        page_size,
        attention_backend_identity=attention_backend_identity,
        kv_cache_dtype_requested="bfloat16",
        kv_cache_dtype_resolved=kv_cache_dtype_resolved,
    )
    handshake.update(
        {
            "parallelism": "expert_parallel",
            "expert_parallel_size": expert_parallel_size,
            "attention_placement": "replicated",
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
) -> None:
    expected = _make_ep_handshake(
        model_dir,
        expert_parallel_size,
        num_pages,
        page_size,
        attention_backend_identity,
        kv_cache_dtype_resolved,
    )
    if handshake != expected:
        raise RuntimeError(
            f"EP worker mismatch: driver={handshake} worker={expected} — "
            "expert topology, replicated attention, eager execution, BF16 KV, "
            "pool sizing/config, and backend identity must match on every rank"
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
                self._control_comm.broadcast(delta, src=0)
                # Match the worker ordering: stale per-request local state and
                # StateSync identity are gone before a new snapshot with the
                # same request id is installed or executed.
                _release_runner_requests(self._local, released_ids)
                view = self._sync.apply(delta)  # reconstructs snapshot_step() exactly
                # Keep pending entries durable across a failed broadcast/local
                # cleanup. Acknowledge only after both rank protocols have
                # accepted the batch and rank 0 has applied it.
                self._ack_pending_releases(pending_releases)
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

    def _dram_kv_collect(self, payload: _DramKVControl) -> tuple[_DramKVAck, ...]:
        """Run one already-serialized all-rank tier transaction."""
        if self._fatal_error is not None:
            raise RuntimeError(
                f"{self._parallelism_display} runner is unavailable after a fatal "
                "step failure"
            ) from self._fatal_error
        operation = _dram_kv_operation(payload)
        try:
            delivered = self._control_comm.broadcast(payload, src=0)
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
            delivered = self._control_comm.broadcast(_SamplingOwnershipProbe(), src=0)
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
            delivered = self._control_comm.broadcast(payload, src=0)
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
            delivered = self._control_comm.broadcast(probe, src=0)
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
            delivered = self._control_comm.broadcast(payload, src=0)
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
            delivered = self._control_comm.broadcast(probe, src=0)
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
            delivered = self._control_comm.broadcast(payload, src=0)
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
            delivered = self._control_comm.broadcast(probe, src=0)
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
                    self._control_comm.broadcast(
                        _ShutdownRequest(released_ids),
                        src=0,
                    )
            finally:
                _release_runner_requests(self._local, released_ids)
                self._ack_pending_releases(pending_releases)

    def invalidate_graphs(self) -> None:
        invalidate = getattr(self._local, "invalidate_graphs", None)
        if invalidate is not None:
            invalidate()


class DistEPModelRunner:
    """Driver facade for replicated-attention expert-parallel correctness.

    The rank protocol is intentionally the same SPMD/single-sampling-owner
    protocol as TP, but the public topology is not: attention and KV are full
    replicas and only routed experts are sharded.
    """

    def __init__(
        self,
        control_comm,
        local_runner,
        model_comm=None,
        *,
        expert_parallel_size: int,
    ) -> None:
        _validate_ep_correctness_mode(
            expert_parallel_size=expert_parallel_size,
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
            parallelism_display="expert-parallel correctness-mode",
            parallelism_prefix="EP",
        )
        self.expert_parallel_size = expert_parallel_size
        self.parallelism = "expert_parallel"
        self.execution_mode = _EP_CORRECTNESS_MODE
        self.attention_placement = "replicated"
        self.pipeline_depth = 1
        self.decode_mode = "eager"

    def __getattr__(self, name: str):
        """Delegate the proven SPMD control protocol without becoming a TP type."""

        return getattr(object.__getattribute__(self, "_delegate"), name)

    def parallelism_metadata(self) -> dict[str, object]:
        """Unambiguous topology metadata for gates and serving surfaces."""

        return {
            "parallelism": self.parallelism,
            "expert_parallel_size": self.expert_parallel_size,
            "attention_placement": self.attention_placement,
            "execution_mode": self.execution_mode,
            "pipeline_depth": self.pipeline_depth,
            "decode_mode": self.decode_mode,
            "kv_cache_dtype": "bfloat16",
        }

    @_serialized_protocol
    def ep_kernel_inventory_metadata(self) -> tuple[dict[str, object], ...]:
        """Gather bounded all-rank NVFP4 runtime evidence over gloo.

        Projection names never leave their owning rank. Each rank sends only a
        count, canonical names SHA-256, quant/kernel counts, unresolved-meta
        counts, and a reduced loader summary. The returned tuple is rank ordered.
        """

        delegate = object.__getattribute__(self, "_delegate")
        if delegate._fatal_error is not None:
            raise RuntimeError(
                "expert-parallel correctness-mode runner is unavailable after "
                "a fatal step failure"
            ) from delegate._fatal_error
        try:
            backend = _communicator_backend(delegate._control_comm)
            if backend not in {"gloo", "fake"}:
                raise RuntimeError(
                    "EP kernel inventory requires the bounded control communicator "
                    f"(gloo), got {backend!r}"
                )
            delivered = delegate._control_comm.broadcast(
                _EPKernelInventoryProbe(),
                src=0,
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
            "replicated-attention EP correctness mode does not support a DRAM KV tier"
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
    while True:
        payload = control_comm.broadcast(_SHUTDOWN, src=0)
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
            if isinstance(payload, _EPKernelInventoryProbe):
                local = _ep_kernel_inventory_reply(control_comm, local_runner)
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
    """One rank per device for replicated-attention expert parallelism."""

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
            "replicated-attention EP correctness mode requires CUDA; "
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
    """Operational groups with control and model collectives kept disjoint.

    ``broadcast_object_list`` is not one NCCL operation: it broadcasts metadata
    and payload tensors, then receivers copy the payload back to the host before
    deserializing it.  A source rank can therefore enqueue the following model
    all-reduce while peers are still completing the object broadcast.  Keeping
    the Python control protocol on gloo makes that hand-off blocking and leaves
    the NCCL group with tensor collectives only.
    """

    control: object
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
    """Create control/model groups in the same order on every rank.

    The control timeout must cover the server's idle lifetime because workers
    wait *inside* its receive. The model group has no pending operation while
    idle, so it keeps the short fail-fast bound.
    """
    return ServingGroups(
        control=serving_group("gloo", timeout_s=control_timeout_s),
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
    pipeline_depth: int = 1,
    decode_mode: str = "eager",
    kv_cache_dtype: str = "bfloat16",
    pd_separation: bool = False,
    graph_scratch_page: int | None = None,
    dram_kv_tier_capacity_pages: int = 0,
    dram_kv_tier_profile: str | Path | None = None,
    speculative: str | None = None,
):
    """Build one rank of the replicated-attention EP2/EP4 correctness path."""

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

    _validate_ep_correctness_mode(
        expert_parallel_size=expert_parallel_size,
        pipeline_depth=pipeline_depth,
        decode_mode=decode_mode,
        kv_cache_dtype=kv_cache_dtype,
        pd_separation=pd_separation,
        graph_scratch_page=graph_scratch_page,
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
            "replicated-attention EP correctness mode requires a CUDA/NCCL "
            "BF16 placement"
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
            "replicated-attention EP correctness mode resolved a non-BF16 KV cache"
        )
    pool = PagedKVPool(
        num_layers=full_config.num_hidden_layers,
        num_pages=num_pages,
        page_size=page_size,
        num_kv_heads=full_config.kv_cache_num_heads,
        head_dim=full_config.kv_cache_head_dim,
        dtype=resolved_kv_cache_dtype,
        device=placement.device,
    )
    grammar_vocab = (
        vocab if isinstance(vocab, GrammarVocabulary) else GrammarVocabulary(list(vocab))
    )
    runner = PagedModelRunner(
        model,
        pool,
        sampler=(
            Sampler(vocab_provider=lambda: grammar_vocab)
            if rank == 0
            else None
        ),
        sampling_owner=rank == 0,
    )
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
    runner.attention_placement = "replicated"
    runner.execution_mode = _EP_CORRECTNESS_MODE
    runner.pipeline_depth = 1
    runner.decode_mode = "eager"
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
    try:
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
) -> None:
    """Spawned EP correctness worker; rank 0 remains in the driver process."""

    import torch

    from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed

    _validate_ep_correctness_mode(
        expert_parallel_size=expert_parallel_size,
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
    )
    groups = serving_groups(placement.backend)
    comm.bind(TorchDistCommunicator(group=groups.model, device=placement.device))
    control_comm = TorchDistCommunicator(group=groups.control)
    try:
        worker_step_loop(
            control_comm,
            runner,
            comm,
            parallelism_prefix="EP",
        )
    finally:
        import torch.distributed as dist

        if placement.backend == "nccl":
            torch.cuda.synchronize()
            comm.barrier()
        dist.destroy_process_group(comm.group)
        dist.destroy_process_group(control_comm.group)
        dist.destroy_process_group()


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
            self.runner = DistTPModelRunner(self._control_comm, runner, self._comm)
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
        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available():
            return
        groups = []
        model_comm = getattr(self, "_comm", None)
        if model_comm is not None:
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
        # BEFORE the join, not after: NCCL's destroy_process_group waits for every
        # rank to reach it, so joining first deadlocks rank 0 against workers that
        # are already sitting in their own destroy. gloo never blocks here, which
        # is why the CPU parity gates could not see this.
        if dist.is_initialized():
            # Multiple process groups must be destroyed explicitly in the same
            # order on every rank.  Reverse creation order keeps the graph-owning
            # NCCL subgroup ahead of the gloo control and startup groups.
            dist.destroy_process_group(self._comm.group)
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
    """Own one EP2/EP4 replicated-attention correctness-mode process group.

    This is a distinct public topology from :class:`DistTPLauncher`. Every rank
    holds complete attention/KV state, while ``build_ep_model`` loads only that
    rank's routed experts. The constructor validates the deliberately narrow
    execution envelope before spawning any process.
    """

    def __init__(
        self,
        model_dir: str,
        expert_parallel_size: int,
        num_pages: int,
        page_size: int,
        vocab: list[str] | GrammarVocabulary,
        *,
        pipeline_depth: int = 1,
        decode_mode: str = "eager",
        kv_cache_dtype: str = "bfloat16",
        pd_separation: bool = False,
        graph_scratch_page: int | None = None,
        dram_kv_tier_capacity_pages: int = 0,
        dram_kv_tier_profile: str | Path | None = None,
        speculative: str | None = None,
    ) -> None:
        import tempfile

        import torch
        import torch.multiprocessing as mp

        from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed

        _validate_ep_correctness_mode(
            expert_parallel_size=expert_parallel_size,
            pipeline_depth=pipeline_depth,
            decode_mode=decode_mode,
            kv_cache_dtype=kv_cache_dtype,
            pd_separation=pd_separation,
            graph_scratch_page=graph_scratch_page,
            dram_kv_tier_capacity_pages=dram_kv_tier_capacity_pages,
            dram_kv_tier_profile=dram_kv_tier_profile,
            speculative=speculative,
        )
        self.expert_parallel_size = expert_parallel_size
        self.parallelism = "expert_parallel"
        self.execution_mode = _EP_CORRECTNESS_MODE
        self.attention_placement = "replicated"
        self.pipeline_depth = 1
        self.decode_mode = "eager"
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
                pipeline_depth=pipeline_depth,
                decode_mode=decode_mode,
                kv_cache_dtype=kv_cache_dtype,
                pd_separation=pd_separation,
                graph_scratch_page=graph_scratch_page,
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
                ),
                src=0,
            )
            groups = serving_groups(placement.backend)
            self._comm.bind(
                TorchDistCommunicator(
                    group=groups.model,
                    device=placement.device,
                )
            )
            self._control_comm = TorchDistCommunicator(group=groups.control)
            self.runner = DistEPModelRunner(
                self._control_comm,
                runner,
                self._comm,
                expert_parallel_size=expert_parallel_size,
            )
        except BaseException:
            self._abandon_start()
            raise

    def parallelism_metadata(self) -> dict[str, object]:
        return self.runner.parallelism_metadata()

    def ep_kernel_inventory_metadata(self) -> tuple[dict[str, object], ...]:
        """Return the rank-complete runtime inventory from the EP runner."""

        return self.runner.ep_kernel_inventory_metadata()
