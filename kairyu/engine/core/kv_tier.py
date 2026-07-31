"""Bounded pinned-DRAM tier for page-granular KV cache spill and restore.

The tier deliberately separates physical copy completion from ownership
publication.  ``begin_offload`` and ``begin_restore`` reserve every host and
device page until their completion handle has both observed the backend event
and finalized the state transition.  A failed or cancelled transfer quarantines
its slots and device pages; ambiguous DMA is never converted into reusable
memory, even through a different logical key.

The default backend uses a dedicated CUDA stream and one startup-allocated
pinned host slab.  Tests and CPU-only users can inject another backend behind
the same completion protocol without weakening the state machine.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

import torch

from kairyu.engine.core.kv_pool import PagedKVPool

_FULL_SHA256 = re.compile(r"[0-9a-f]{64}")
_LAYOUT = "layer-major:[layer,page,token,kv-head,head-dim];fragments:k,v-per-layer"
FRAGMENT_MAJOR_HOST_LAYOUT = "fragment-major:[fragment,slot,bytes]-v1"
CUDA_CONTIGUOUS_BACKEND = "cuda-pinned-dram-fragment-major-torch-copy-v1"
_CHECKSUM_BATCH_PAGES = 32


class PageState(StrEnum):
    """Host-page ownership states visible to policy and contract tests."""

    RESERVED = "reserved"
    OFFLOADING = "offloading"
    RESIDENT = "resident"
    RESTORING = "restoring"


class TransferDirection(StrEnum):
    OFFLOAD = "offload"
    RESTORE = "restore"


class _TransferTerminal(StrEnum):
    ACTIVE = "active"
    FINALIZED = "finalized"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class KVPageFingerprint:
    """Complete identity of one TP rank's physical KV page layout."""

    model_id: str
    tp_size: int
    tp_rank: int
    layout: str
    num_layers: int
    page_size: int
    num_kv_heads: int
    k_head_dim: int
    v_head_dim: int
    k_dtype: str
    v_dtype: str
    page_nbytes: int
    schema_version: int = 1

    @classmethod
    def from_pool(
        cls,
        pool: PagedKVPool,
        *,
        model_id: str,
        tp_size: int,
        tp_rank: int,
    ) -> KVPageFingerprint:
        if not model_id:
            raise ValueError("model_id must be a non-empty pinned model identity")
        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp_size}")
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"tp_rank {tp_rank} is outside [0, {tp_size})")
        page_nbytes = sum(
            fragment.numel() * fragment.element_size()
            for fragment in _page_fragments(pool, 0)
        )
        return cls(
            model_id=model_id,
            tp_size=tp_size,
            tp_rank=tp_rank,
            layout=_LAYOUT,
            num_layers=pool.num_layers,
            page_size=pool.page_size,
            num_kv_heads=pool.num_kv_heads,
            k_head_dim=pool.head_dim,
            v_head_dim=pool.v_head_dim,
            k_dtype=str(pool.k.dtype),
            v_dtype=str(pool.v.dtype),
            page_nbytes=page_nbytes,
        )

    @property
    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()


class KVPageTransferError(RuntimeError):
    """Failure with explicit post-error ownership safety.

    The three booleans describe what the caller may do *after catching* the
    exception.  In particular, an unpublished restore destination can still be
    unsafe to reuse when physical H2D completion is unknown.
    """

    def __init__(
        self,
        message: str,
        *,
        source_reusable: bool,
        destination_reusable: bool,
        destination_publishable: bool,
    ) -> None:
        super().__init__(message)
        self.source_reusable = source_reusable
        self.destination_reusable = destination_reusable
        self.destination_publishable = destination_publishable


class KVPageCapacityError(KVPageTransferError):
    """The bounded host slab cannot reserve the requested batch."""


class KVPageIdentityError(KVPageTransferError):
    """A key or physical-layout identity does not match this tier."""


class KVPageChecksumError(KVPageTransferError):
    """Resident bytes no longer match their commit-time checksum."""


class KVPageStateError(KVPageTransferError):
    """The requested transition is not legal from the current state."""


class BackendCompletion(Protocol):
    """Physical completion returned by an injected transfer backend."""

    def query(self) -> bool: ...

    def synchronize(self) -> None: ...


class PageTransferBackend(Protocol):
    """Allocation and copy seam used by the fail-closed state machine."""

    def allocate_host_pages(self, capacity_pages: int, page_nbytes: int) -> torch.Tensor:
        ...

    def begin_offload(
        self,
        sources: tuple[tuple[torch.Tensor, ...], ...],
        destinations: tuple[torch.Tensor, ...],
    ) -> BackendCompletion: ...

    def begin_restore(
        self,
        sources: tuple[torch.Tensor, ...],
        destinations: tuple[tuple[torch.Tensor, ...], ...],
    ) -> BackendCompletion: ...


class _ImmediateCompletion:
    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None

    def cancel(self) -> None:
        return None


class _CudaCompletion:
    """Keep tensors alive until the side-stream event has physically fired."""

    def __init__(
        self,
        start_event: torch.cuda.Event,
        end_event: torch.cuda.Event,
        *,
        stream_id: int,
        keepalive: tuple[object, ...],
    ) -> None:
        self.start_event = start_event
        self.event = end_event
        self.stream_id = stream_id
        self._keepalive = keepalive

    def query(self) -> bool:
        return bool(self.event.query())

    def synchronize(self) -> None:
        self.event.synchronize()

    def cancel(self) -> None:
        # CUDA work already submitted to a stream cannot be withdrawn.  The
        # tier quarantines ownership; this method intentionally does not lie.
        return None

    def elapsed_ns(self) -> int:
        return max(1, round(self.start_event.elapsed_time(self.event) * 1_000_000))


class TorchPageTransferBackend:
    """Torch CPU copy or asynchronous CUDA D2H/H2D copy backend."""

    def __init__(self, device: torch.device | str) -> None:
        requested = torch.device(device)
        if requested.type == "cuda":
            index = torch.cuda.current_device() if requested.index is None else requested.index
            requested = torch.device("cuda", index)
        self.device = requested
        self._stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self.transfer_backend = (
            CUDA_CONTIGUOUS_BACKEND
            if self.device.type == "cuda"
            else "torch-cpu-page-copy-v1"
        )
        self.host_layout = (
            FRAGMENT_MAJOR_HOST_LAYOUT if self.device.type == "cuda" else None
        )
        self.last_fragment_copy_count = 0

    def allocate_host_pages(self, capacity_pages: int, page_nbytes: int) -> torch.Tensor:
        return torch.empty(
            (capacity_pages, page_nbytes),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=self.device.type == "cuda",
        )

    @staticmethod
    def _validate_fragment(fragment: torch.Tensor) -> torch.Tensor:
        if not fragment.is_contiguous():
            raise ValueError("KV page fragments must be contiguous views")
        return fragment.detach().view(torch.uint8).reshape(-1)

    @classmethod
    def _copy_offload_page(
        cls,
        source: tuple[torch.Tensor, ...],
        destination: torch.Tensor,
        *,
        non_blocking: bool,
    ) -> None:
        offset = 0
        for fragment in source:
            raw = cls._validate_fragment(fragment)
            destination[offset : offset + raw.numel()].copy_(
                raw, non_blocking=non_blocking
            )
            offset += raw.numel()
        if offset != destination.numel():
            raise ValueError(
                f"page fragments contain {offset} bytes, host page has "
                f"{destination.numel()}"
            )

    @classmethod
    def _copy_restore_page(
        cls,
        source: torch.Tensor,
        destination: tuple[torch.Tensor, ...],
        *,
        non_blocking: bool,
    ) -> None:
        offset = 0
        for fragment in destination:
            raw = cls._validate_fragment(fragment)
            raw.copy_(source[offset : offset + raw.numel()], non_blocking=non_blocking)
            offset += raw.numel()
        if offset != source.numel():
            raise ValueError(
                f"destination fragments contain {offset} bytes, host page has "
                f"{source.numel()}"
            )

    @staticmethod
    def _jointly_contiguous_runs(
        host_slots: tuple[int, ...],
        device_page_ids: tuple[int, ...],
    ) -> tuple[tuple[int, int, int], ...]:
        """Return ``(host slot, device page, count)`` for matching extents."""

        if not host_slots or len(host_slots) != len(device_page_ids):
            raise ValueError("host/device KV index batches must be non-empty and equal")
        starts = [0]
        for index in range(1, len(host_slots)):
            if (
                host_slots[index] != host_slots[index - 1] + 1
                or device_page_ids[index] != device_page_ids[index - 1] + 1
            ):
                starts.append(index)
        return tuple(
            (
                host_slots[start],
                device_page_ids[start],
                end - start,
            )
            for start, end in zip(
                starts,
                (*starts[1:], len(host_slots)),
                strict=True,
            )
        )

    @classmethod
    def _copy_matching_fragments(
        cls,
        source: tuple[torch.Tensor, ...],
        destination: tuple[torch.Tensor, ...],
        *,
        non_blocking: bool,
    ) -> None:
        if len(source) != len(destination):
            raise ValueError("host/device KV fragment counts differ")
        for source_fragment, destination_fragment in zip(
            source,
            destination,
            strict=True,
        ):
            raw_source = cls._validate_fragment(source_fragment)
            raw_destination = cls._validate_fragment(destination_fragment)
            if raw_source.numel() != raw_destination.numel():
                raise ValueError("host/device KV fragment byte sizes differ")
            raw_destination.copy_(raw_source, non_blocking=non_blocking)

    def _validated_fragment_planes(
        self,
        host_planes: tuple[torch.Tensor, ...],
        device_planes: tuple[torch.Tensor, ...],
        host_slots: tuple[int, ...],
        device_page_ids: tuple[int, ...],
    ) -> tuple[tuple[int, int, int], ...]:
        """Validate plane owners once, without constructing views per page."""

        if not host_planes or len(host_planes) != len(device_planes):
            raise ValueError("host/device KV fragment planes must be non-empty and equal")
        runs = self._jointly_contiguous_runs(host_slots, device_page_ids)
        if any(
            plane.ndim != 2
            or plane.device.type != "cpu"
            or plane.dtype != torch.uint8
            or not plane.is_contiguous()
            or (self.device.type == "cuda" and not plane.is_pinned())
            for plane in host_planes
        ):
            raise ValueError(
                "fragment-major transfer requires contiguous CPU uint8 host planes"
            )
        if any(
            plane.ndim < 2
            or plane.device != self.device
            or not plane.is_contiguous()
            for plane in device_planes
        ):
            raise ValueError(
                "fragment-major transfer requires contiguous device planes on "
                f"{self.device}"
            )
        host_capacity = host_planes[0].shape[0]
        device_capacity = device_planes[0].shape[0]
        if any(plane.shape[0] != host_capacity for plane in host_planes):
            raise ValueError("host KV fragment planes have different slot capacities")
        if any(plane.shape[0] != device_capacity for plane in device_planes):
            raise ValueError("device KV fragment planes have different page capacities")
        fragment_sizes = tuple(
            plane[0].numel() * plane.element_size() for plane in device_planes
        )
        if any(
            plane.shape[1] != fragment_nbytes
            for plane, fragment_nbytes in zip(
                host_planes,
                fragment_sizes,
                strict=True,
            )
        ):
            raise ValueError("host/device KV fragment plane byte sizes differ")
        if any(
            host_slot < 0 or host_slot >= host_capacity
            for host_slot in host_slots
        ):
            raise ValueError("host KV slot is outside the fragment-major slab")
        if any(
            page_id < 0 or page_id >= device_capacity
            for page_id in device_page_ids
        ):
            raise ValueError("device KV page is outside the fragment planes")
        return runs

    def _copy_fragment_planes(
        self,
        host_planes: tuple[torch.Tensor, ...],
        device_planes: tuple[torch.Tensor, ...],
        runs: tuple[tuple[int, int, int], ...],
        *,
        restore: bool,
        non_blocking: bool,
    ) -> int:
        copy_count = 0
        for host_plane, device_plane in zip(
            host_planes,
            device_planes,
            strict=True,
        ):
            for host_start, device_start, count in runs:
                host_run = host_plane[host_start : host_start + count].reshape(-1)
                device_run = (
                    device_plane[device_start : device_start + count]
                    .detach()
                    .view(torch.uint8)
                    .reshape(-1)
                )
                destination, source = (
                    (device_run, host_run)
                    if restore
                    else (host_run, device_run)
                )
                destination.copy_(source, non_blocking=non_blocking)
                copy_count += 1
        return copy_count

    def begin_offload(
        self,
        sources: tuple[tuple[torch.Tensor, ...], ...],
        destinations: tuple[torch.Tensor, ...],
    ) -> BackendCompletion:
        if self._stream is None:
            for source, destination in zip(sources, destinations, strict=True):
                self._copy_offload_page(source, destination, non_blocking=False)
            return _ImmediateCompletion()

        if any(not destination.is_pinned() for destination in destinations):
            raise ValueError("asynchronous D2H requires pinned host pages")
        current = torch.cuda.current_stream(device=self.device)
        self._stream.wait_stream(current)
        with torch.cuda.stream(self._stream):
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(self._stream)
            for source, destination in zip(sources, destinations, strict=True):
                self._copy_offload_page(source, destination, non_blocking=True)
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(self._stream)
        return _CudaCompletion(
            start_event,
            end_event,
            stream_id=int(self._stream.cuda_stream),
            keepalive=(sources, destinations),
        )

    def begin_restore(
        self,
        sources: tuple[torch.Tensor, ...],
        destinations: tuple[tuple[torch.Tensor, ...], ...],
    ) -> BackendCompletion:
        if self._stream is None:
            for source, destination in zip(sources, destinations, strict=True):
                self._copy_restore_page(source, destination, non_blocking=False)
            return _ImmediateCompletion()

        if any(not source.is_pinned() for source in sources):
            raise ValueError("asynchronous H2D requires pinned host pages")
        current = torch.cuda.current_stream(device=self.device)
        self._stream.wait_stream(current)
        with torch.cuda.stream(self._stream):
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(self._stream)
            for source, destination in zip(sources, destinations, strict=True):
                self._copy_restore_page(source, destination, non_blocking=True)
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(self._stream)
        return _CudaCompletion(
            start_event,
            end_event,
            stream_id=int(self._stream.cuda_stream),
            keepalive=(sources, destinations),
        )

    def _begin_fragment_transfer(
        self,
        host_planes: tuple[torch.Tensor, ...],
        device_planes: tuple[torch.Tensor, ...],
        host_slots: tuple[int, ...],
        device_page_ids: tuple[int, ...],
        *,
        restore: bool,
    ) -> BackendCompletion:
        runs = self._validated_fragment_planes(
            host_planes,
            device_planes,
            host_slots,
            device_page_ids,
        )
        if self._stream is None:
            self.last_fragment_copy_count = self._copy_fragment_planes(
                host_planes,
                device_planes,
                runs,
                restore=restore,
                non_blocking=False,
            )
            return _ImmediateCompletion()

        current = torch.cuda.current_stream(device=self.device)
        self._stream.wait_stream(current)
        with torch.cuda.stream(self._stream):
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(self._stream)
            self.last_fragment_copy_count = self._copy_fragment_planes(
                host_planes,
                device_planes,
                runs,
                restore=restore,
                non_blocking=True,
            )
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(self._stream)
        return _CudaCompletion(
            start_event,
            end_event,
            stream_id=int(self._stream.cuda_stream),
            keepalive=(host_planes, device_planes),
        )

    def begin_offload_fragments(
        self,
        sources: tuple[torch.Tensor, ...],
        source_page_ids: tuple[int, ...],
        destinations: tuple[torch.Tensor, ...],
        destination_slots: tuple[int, ...],
    ) -> BackendCompletion:
        return self._begin_fragment_transfer(
            destinations,
            sources,
            destination_slots,
            source_page_ids,
            restore=False,
        )

    def begin_restore_fragments(
        self,
        sources: tuple[torch.Tensor, ...],
        source_slots: tuple[int, ...],
        destinations: tuple[torch.Tensor, ...],
        destination_page_ids: tuple[int, ...],
    ) -> BackendCompletion:
        return self._begin_fragment_transfer(
            sources,
            destinations,
            source_slots,
            destination_page_ids,
            restore=True,
        )


def resolve_transfer_backend_identity(
    device: torch.device | str,
) -> str:
    """Resolve the copy path before allocating a potentially large host slab."""

    requested = torch.device(device)
    if requested.type != "cuda":
        return "torch-cpu-page-copy-v1"
    return CUDA_CONTIGUOUS_BACKEND


@dataclass(frozen=True)
class ResidentPageMetadata:
    key: str
    checksum: str
    fingerprint: str
    page_nbytes: int
    last_touch: int


@dataclass(frozen=True)
class CompletedTransferMetrics:
    """Immutable telemetry for the most recently published transfer."""

    sequence: int
    direction: TransferDirection
    page_count: int
    bytes_transferred: int
    stream_id: int | None
    cuda_elapsed_ns: int | None


@dataclass
class _SlotEntry:
    key: str
    slot: int
    generation: int
    state: PageState
    fingerprint: str
    checksum: str | None = None
    last_touch: int = 0


def _page_fragments(pool: PagedKVPool, page_id: int) -> tuple[torch.Tensor, ...]:
    fragments: list[torch.Tensor] = []
    for layer in range(pool.num_layers):
        fragments.append(pool.k[layer, page_id])
        if pool.v_head_dim:
            fragments.append(pool.v[layer, page_id])
    return tuple(fragments)


def _device_fragment_planes(pool: PagedKVPool) -> tuple[torch.Tensor, ...]:
    """Return one contiguous page-indexed device plane per logical fragment."""

    planes: list[torch.Tensor] = []
    for layer in range(pool.num_layers):
        planes.append(pool.k[layer])
        if pool.v_head_dim:
            planes.append(pool.v[layer])
    return tuple(planes)


def _fragment_major_planes(
    storage: torch.Tensor,
    pool: PagedKVPool,
) -> tuple[torch.Tensor, ...]:
    """Carve one row-compatible owner into fragment-major slot planes."""

    capacity_pages = storage.shape[0]
    flat = storage.reshape(-1)
    offset = 0
    planes: list[torch.Tensor] = []
    for fragment in _page_fragments(pool, 0):
        fragment_nbytes = fragment.numel() * fragment.element_size()
        end = offset + capacity_pages * fragment_nbytes
        if end > flat.numel():
            raise ValueError("fragment-major host layout exceeds its owner slab")
        planes.append(flat[offset:end].view(capacity_pages, fragment_nbytes))
        offset = end
    if offset != flat.numel():
        raise ValueError(
            f"fragment-major host layout covers {offset} bytes, owner has "
            f"{flat.numel()}"
        )
    return tuple(planes)


def _fragment_major_page(
    planes: tuple[torch.Tensor, ...],
    slot: int,
) -> tuple[torch.Tensor, ...]:
    return tuple(plane[slot] for plane in planes)


def _checksum(page: torch.Tensor | tuple[torch.Tensor, ...]) -> str:
    fragments = (page,) if isinstance(page, torch.Tensor) else page
    if not fragments:
        raise ValueError("host page checksum requires at least one fragment")
    digest = hashlib.sha256()
    for fragment in fragments:
        if (
            fragment.device.type != "cpu"
            or fragment.dtype != torch.uint8
            or not fragment.is_contiguous()
        ):
            raise ValueError(
                "host page checksum requires contiguous CPU uint8 fragments"
            )
        # NumPy exposes each contiguous fragment through the buffer protocol,
        # so SHA consumes it without allocating another Python ``bytes`` copy.
        digest.update(fragment.numpy())
    return digest.hexdigest()


def _fragment_major_checksums(
    storage: torch.Tensor,
    planes: tuple[torch.Tensor, ...],
    slots: tuple[int, ...],
    *,
    page_nbytes: int,
) -> tuple[str, ...]:
    """Hash logical pages in bounded page-major batches without changing ABI."""

    if not slots:
        return ()
    fragment_sizes = tuple(plane.shape[1] for plane in planes)
    if len(set(fragment_sizes)) != 1:
        return tuple(
            _checksum(_fragment_major_page(planes, slot)) for slot in slots
        )
    fragment_nbytes = fragment_sizes[0]
    grouped = storage.reshape(
        len(planes),
        storage.shape[0],
        fragment_nbytes,
    )
    result: list[str] = []
    for start in range(0, len(slots), _CHECKSUM_BATCH_PAGES):
        chunk_slots = slots[start : start + _CHECKSUM_BATCH_PAGES]
        first = chunk_slots[0]
        if chunk_slots == tuple(range(first, first + len(chunk_slots))):
            selected = grouped[:, first : first + len(chunk_slots)]
        else:
            indices = torch.tensor(chunk_slots, dtype=torch.long)
            selected = grouped.index_select(1, indices)
        page_major = (
            selected.permute(1, 0, 2)
            .contiguous()
            .view(len(chunk_slots), page_nbytes)
        )
        result.extend(_checksum(page) for page in page_major)
    return tuple(result)


def _keys(value: Iterable[str]) -> tuple[str, ...]:
    keys = tuple(value)
    if not keys:
        raise KVPageIdentityError(
            "a KV tier operation requires at least one key",
            source_reusable=True,
            destination_reusable=True,
            destination_publishable=False,
        )
    if len(set(keys)) != len(keys):
        raise KVPageIdentityError(
            "one KV tier operation cannot contain duplicate keys",
            source_reusable=True,
            destination_reusable=True,
            destination_publishable=False,
        )
    for key in keys:
        if _FULL_SHA256.fullmatch(key) is None:
            raise KVPageIdentityError(
                f"KV page key must be a full lowercase SHA-256, got {key!r}",
                source_reusable=True,
                destination_reusable=True,
                destination_publishable=False,
            )
    return keys


class TierTransfer:
    """Opaque ownership token for one batch of D2H or H2D copies."""

    def __init__(
        self,
        tier: PinnedKVPageTier,
        *,
        sequence: int,
        direction: TransferDirection,
        keys: tuple[str, ...],
        page_ids: tuple[int, ...],
        entries: tuple[_SlotEntry, ...],
        previous: tuple[_SlotEntry | None, ...],
        completion: BackendCompletion,
        device_page_owner: object,
        checksums: tuple[str, ...] = (),
    ) -> None:
        self._tier = tier
        self.sequence = sequence
        self.direction = direction
        self.keys = keys
        self.page_ids = page_ids
        self.bytes_transferred = len(keys) * tier.page_nbytes
        self._entries = entries
        self._previous = previous
        self._completion = completion
        self._device_page_owner = device_page_owner
        self._checksums = checksums
        self._physical_complete = False
        self._terminal = _TransferTerminal.ACTIVE
        self._failure: KVPageTransferError | None = None
        # A transfer handle is a linearizable ownership token.  Its backend
        # completion and terminal cleanup may each run at most once even when
        # callers race ready/wait/finalize/cancel from different threads.
        self._lock = threading.RLock()

    @property
    def checksums(self) -> tuple[str, ...]:
        with self._lock:
            return self._checksums

    @property
    def event(self) -> object | None:
        """The backend CUDA event when present; opaque for other backends."""
        return getattr(self._completion, "event", None)

    @property
    def stream_id(self) -> int | None:
        return getattr(self._completion, "stream_id", None)

    @property
    def cuda_elapsed_ns(self) -> int | None:
        with self._lock:
            if (
                not self._physical_complete
                and self._terminal is not _TransferTerminal.FINALIZED
            ):
                return None
            elapsed = getattr(self._completion, "elapsed_ns", None)
            return int(elapsed()) if callable(elapsed) else None

    @property
    def source_reusable(self) -> bool:
        with self._lock:
            if self._failure is not None:
                return self._failure.source_reusable
            return self._terminal is _TransferTerminal.FINALIZED or (
                self.direction is TransferDirection.OFFLOAD
                and self._physical_complete
            )

    @property
    def destination_reusable(self) -> bool:
        with self._lock:
            if self._failure is not None:
                return self._failure.destination_reusable
            return self._terminal is _TransferTerminal.FINALIZED or (
                self.direction is TransferDirection.RESTORE
                and self._physical_complete
            )

    @property
    def destination_publishable(self) -> bool:
        with self._lock:
            if self._failure is not None:
                return self._failure.destination_publishable
            return (
                self._terminal is _TransferTerminal.FINALIZED
                and self.direction is TransferDirection.RESTORE
            )

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._terminal in {
            _TransferTerminal.CANCELLED,
            _TransferTerminal.FAILED,
        }:
            raise RuntimeError("terminal KV transfer is missing its failure record")

    def _ready_locked(self) -> bool:
        self._raise_failure()
        if (
            self._terminal is _TransferTerminal.FINALIZED
            or self._physical_complete
        ):
            return True
        try:
            self._physical_complete = bool(self._completion.query())
        except BaseException as error:
            self._failure = self._tier._fail_transfer(self, error)
            self._terminal = _TransferTerminal.FAILED
            raise self._failure from error
        return self._physical_complete

    def ready(self) -> bool:
        with self._lock:
            return self._ready_locked()

    def wait(self) -> bool:
        with self._lock:
            self._raise_failure()
            if (
                self._terminal is _TransferTerminal.FINALIZED
                or self._physical_complete
            ):
                return True
            try:
                self._completion.synchronize()
            except BaseException as error:
                self._failure = self._tier._fail_transfer(self, error)
                self._terminal = _TransferTerminal.FAILED
                raise self._failure from error
            self._physical_complete = True
            return True

    def finalize(self) -> bool:
        with self._lock:
            self._raise_failure()
            if self._terminal is _TransferTerminal.FINALIZED:
                return True
            if not self._ready_locked():
                raise KVPageStateError(
                    f"{self.direction} transfer {self.sequence} is still in flight",
                    source_reusable=False,
                    destination_reusable=False,
                    destination_publishable=False,
                )
            try:
                self._tier._finalize_transfer(self)
            except BaseException as error:
                self._failure = self._tier._fail_finalization(self, error)
                self._terminal = _TransferTerminal.FAILED
                raise self._failure from error
            self._terminal = _TransferTerminal.FINALIZED
            return True

    def cancel(self) -> None:
        with self._lock:
            self._raise_failure()
            if self._terminal is _TransferTerminal.FINALIZED:
                raise KVPageStateError(
                    f"{self.direction} transfer {self.sequence} is already finalized",
                    source_reusable=True,
                    destination_reusable=True,
                    destination_publishable=self.direction
                    is TransferDirection.RESTORE,
                )
            cancel = getattr(self._completion, "cancel", None)
            cancel_error: BaseException | None = None
            try:
                if cancel is not None:
                    cancel()
            except BaseException as error:
                cancel_error = error
            self._terminal = _TransferTerminal.CANCELLED
            failure = self._tier._cancel_transfer(self, cancel_error)
            self._failure = failure
            if cancel_error is not None:
                self._terminal = _TransferTerminal.FAILED
                raise failure from cancel_error


class PinnedKVPageTier:
    """Bounded host page store with transactional offload/restore ownership."""

    def __init__(
        self,
        pool: PagedKVPool,
        *,
        model_id: str,
        tp_size: int,
        tp_rank: int,
        capacity_pages: int,
        backend: PageTransferBackend | None = None,
    ) -> None:
        if capacity_pages < 1:
            raise ValueError(f"capacity_pages must be >= 1, got {capacity_pages}")
        if pool.num_pages < 1:
            raise ValueError("the device KV pool must contain at least one page")
        self.pool = pool
        self.fingerprint = KVPageFingerprint.from_pool(
            pool, model_id=model_id, tp_size=tp_size, tp_rank=tp_rank
        )
        self.fingerprint_digest = self.fingerprint.digest
        self.capacity_pages = capacity_pages
        self.page_nbytes = self.fingerprint.page_nbytes
        self._numa_plan = None
        self._numa_attestation = None
        if pool.k.device.type == "cuda":
            from kairyu.engine.core.numa import resolve_cuda_numa_plan

            self._numa_plan = resolve_cuda_numa_plan(
                pool.k.device,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
        affinity_scope = nullcontext()
        if self._numa_plan is not None:
            from kairyu.engine.core.numa import scoped_current_thread_affinity

            affinity_scope = scoped_current_thread_affinity(self._numa_plan)
        with affinity_scope:
            # CUDA stream creation, pinned allocation, and complete first-touch
            # all use the rank-local CPU set. The scope restores its caller on
            # success and on every constructor failure path.
            self.backend = backend or TorchPageTransferBackend(pool.k.device)
            self._host_pages = self.backend.allocate_host_pages(
                capacity_pages, self.page_nbytes
            )
            if (
                self._host_pages.shape != (capacity_pages, self.page_nbytes)
                or self._host_pages.dtype != torch.uint8
                or self._host_pages.device.type != "cpu"
                or not self._host_pages.is_contiguous()
            ):
                raise ValueError(
                    "transfer backend must allocate contiguous CPU uint8 "
                    f"[{capacity_pages}, {self.page_nbytes}] storage"
                )
            if pool.k.device.type == "cuda" and not self._host_pages.is_pinned():
                raise ValueError("CUDA KV tier storage must be pinned host memory")
            host_layout = getattr(self.backend, "host_layout", None)
            if host_layout is None:
                self._host_fragment_planes: tuple[torch.Tensor, ...] | None = None
                self._device_fragment_planes: tuple[torch.Tensor, ...] | None = None
            elif host_layout == FRAGMENT_MAJOR_HOST_LAYOUT:
                if not callable(
                    getattr(self.backend, "begin_offload_fragments", None)
                ) or not callable(
                    getattr(self.backend, "begin_restore_fragments", None)
                ):
                    raise ValueError(
                        "fragment-major backend is missing fragment transfer methods"
                    )
                self._host_fragment_planes = _fragment_major_planes(
                    self._host_pages,
                    pool,
                )
                self._device_fragment_planes = _device_fragment_planes(pool)
            else:
                raise ValueError(f"unsupported transfer backend host layout: {host_layout}")
            if self._numa_plan is not None:
                from kairyu.engine.core.numa import first_touch_and_attest_pinned_host

                self._numa_attestation = first_touch_and_attest_pinned_host(
                    self._host_pages,
                    self._numa_plan,
                )
        self._host_numa_page_counts: Mapping[int, int] = MappingProxyType(
            dict(
                self._numa_attestation.page_counts
                if self._numa_attestation is not None
                else ()
            )
        )

        self._lock = threading.RLock()
        self._slots: list[_SlotEntry | None] = [None] * capacity_pages
        self._slot_generations = [0] * capacity_pages
        self._free_slots = list(reversed(range(capacity_pages)))
        self._published: dict[str, _SlotEntry] = {}
        self._in_flight: set[str] = set()
        self._quarantined: dict[int, tuple[str, BaseException]] = {}
        self._quarantined_keys: set[str] = set()
        self._device_page_owners: dict[int, object] = {}
        self._quarantined_device_pages: set[int] = set()
        self._last_completed_transfer: CompletedTransferMetrics | None = None
        self._sequence = 0
        self._clock = 0

    def _host_checksums(self, entries: Iterable[_SlotEntry]) -> tuple[str, ...]:
        """Hash one host batch on GPU-local CPUs without leaking affinity."""

        entries = tuple(entries)
        slots = tuple(entry.slot for entry in entries)

        def checksums() -> tuple[str, ...]:
            if self._host_fragment_planes is None:
                return tuple(_checksum(self._host_pages[slot]) for slot in slots)
            return _fragment_major_checksums(
                self._host_pages,
                self._host_fragment_planes,
                slots,
                page_nbytes=self.page_nbytes,
            )

        if self._numa_plan is None:
            return checksums()
        from kairyu.engine.core.numa import scoped_current_thread_affinity

        with scoped_current_thread_affinity(self._numa_plan):
            return checksums()

    def _host_page(
        self,
        slot: int,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if self._host_fragment_planes is None:
            return self._host_pages[slot]
        return _fragment_major_page(self._host_fragment_planes, slot)

    @property
    def host_is_pinned(self) -> bool:
        return bool(self._host_pages.is_pinned())

    @property
    def transfer_backend(self) -> str:
        """Exact physical copy implementation used by this rank-local tier."""

        value = getattr(self.backend, "transfer_backend", None)
        return (
            value
            if isinstance(value, str) and value
            else f"injected:{type(self.backend).__module__}.{type(self.backend).__qualname__}"
        )

    @property
    def host_data_ptr(self) -> int:
        """Base address for read-only residency attestation (for example move_pages)."""

        return int(self._host_pages.data_ptr())

    @property
    def host_nbytes(self) -> int:
        return self._host_pages.numel() * self._host_pages.element_size()

    @property
    def host_numa_policy(self) -> str | None:
        """Placement contract applied to the pinned slab, if it is a CUDA tier."""

        return (
            None
            if self._numa_attestation is None
            else self._numa_attestation.policy_name
        )

    @property
    def host_numa_node(self) -> int | None:
        """GPU-local NUMA node proven for every resident host-slab page."""

        return (
            None
            if self._numa_attestation is None
            else self._numa_attestation.numa_node
        )

    @property
    def host_cpu_affinity(self) -> tuple[int, ...]:
        """Calling-thread CPU set used to allocate and first-touch the slab."""

        return (
            ()
            if self._numa_attestation is None
            else self._numa_attestation.cpu_affinity
        )

    @property
    def host_numa_page_counts(self) -> Mapping[int, int]:
        """Read-only kernel page counts by NUMA node for the slab's VMAs."""

        return self._host_numa_page_counts

    @property
    def resident_pages(self) -> int:
        with self._lock:
            return sum(entry.state is PageState.RESIDENT for entry in self._published.values())

    @property
    def quarantined_pages(self) -> int:
        with self._lock:
            return len(self._quarantined)

    @property
    def free_pages(self) -> int:
        with self._lock:
            return len(self._free_slots)

    @property
    def quarantined_keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._quarantined_keys))

    @property
    def quarantined_device_pages(self) -> tuple[int, ...]:
        """Device pages retained after an ambiguously completed DMA."""

        with self._lock:
            return tuple(sorted(self._quarantined_device_pages))

    @property
    def last_completed_transfer(self) -> CompletedTransferMetrics | None:
        """Return exact backend DMA telemetry without retaining its live handle."""

        with self._lock:
            return self._last_completed_transfer

    def state(self, key: str) -> PageState | None:
        with self._lock:
            entry = self._published.get(key)
            return None if entry is None else entry.state

    def validate_fingerprint(self, expected: KVPageFingerprint | str) -> None:
        digest = expected.digest if isinstance(expected, KVPageFingerprint) else expected
        if digest != self.fingerprint_digest:
            raise KVPageIdentityError(
                "KV page fingerprint mismatch: "
                f"expected {digest}, local {self.fingerprint_digest}",
                source_reusable=True,
                destination_reusable=True,
                destination_publishable=False,
            )

    def available_prefix(self, keys: tuple[str, ...], *, min_pages: int = 1) -> int:
        if min_pages < 1:
            raise ValueError(f"min_pages must be >= 1, got {min_pages}")
        wanted = _keys(keys)
        with self._lock:
            hits: list[_SlotEntry] = []
            for key in wanted:
                entry = self._published.get(key)
                if entry is None or entry.state is not PageState.RESIDENT:
                    break
                hits.append(entry)
            available = len(hits)
            if available >= min_pages:
                # A policy probe that can actually be restored is a real cache
                # hit.  Protect its whole returned prefix from the next LRU
                # reservation, rather than touching only the first page.
                for entry in hits:
                    self._clock += 1
                    entry.last_touch = self._clock
            return available if available >= min_pages else 0

    def resident_metadata(self, key: str) -> ResidentPageMetadata:
        _keys((key,))
        with self._lock:
            entry = self._published.get(key)
            if entry is None or entry.state is not PageState.RESIDENT or entry.checksum is None:
                raise KVPageStateError(
                    f"KV page {key!r} is not publishable resident data",
                    source_reusable=False,
                    destination_reusable=False,
                    destination_publishable=False,
                )
            return ResidentPageMetadata(
                key=key,
                checksum=entry.checksum,
                fingerprint=entry.fingerprint,
                page_nbytes=self.page_nbytes,
                last_touch=entry.last_touch,
            )

    def _page_ids(self, page_ids: tuple[int, ...], count: int) -> tuple[int, ...]:
        pages = tuple(page_ids)
        if len(pages) != count:
            raise KVPageIdentityError(
                f"received {len(pages)} page ids for {count} keys",
                source_reusable=True,
                destination_reusable=True,
                destination_publishable=False,
            )
        if len(set(pages)) != len(pages):
            raise KVPageIdentityError(
                "one KV tier operation cannot contain duplicate page ids",
                source_reusable=True,
                destination_reusable=True,
                destination_publishable=False,
            )
        for page_id in pages:
            if not 0 <= page_id < self.pool.num_pages:
                raise KVPageIdentityError(
                    f"device page {page_id} is outside [0, {self.pool.num_pages})",
                    source_reusable=True,
                    destination_reusable=True,
                    destination_publishable=False,
                )
        return pages

    def _evict_lru_for(
        self,
        count: int,
        *,
        protected_keys: set[str],
    ) -> None:
        shortfall = count - len(self._free_slots)
        if shortfall <= 0:
            return
        protected = protected_keys | self._in_flight | self._quarantined_keys
        candidates = sorted(
            (
                entry
                for key, entry in self._published.items()
                if key not in protected and entry.state is PageState.RESIDENT
            ),
            key=lambda entry: (entry.last_touch, entry.slot),
        )
        if len(candidates) < shortfall:
            raise KVPageCapacityError(
                f"KV DRAM tier needs {count} reservable pages but has "
                f"{len(self._free_slots)} free and {len(candidates)} safe LRU victims",
                source_reusable=True,
                destination_reusable=True,
                destination_publishable=False,
            )
        # Candidate sufficiency is checked before mutation, so capacity failure
        # never partially discards a useful resident prefix.
        for entry in candidates[:shortfall]:
            del self._published[entry.key]
            self._release(entry)

    def _reserve(self, keys: tuple[str, ...]) -> tuple[_SlotEntry, ...]:
        self._evict_lru_for(len(keys), protected_keys=set(keys))
        entries: list[_SlotEntry] = []
        # Stable ascending slot assignment preserves long jointly contiguous
        # host/device runs after earlier restores return slots in arbitrary
        # order. The physical slot order is not part of logical LRU identity.
        slots = sorted(self._free_slots.pop() for _ in keys)
        for key, slot in zip(keys, slots, strict=True):
            self._slot_generations[slot] += 1
            entry = _SlotEntry(
                key=key,
                slot=slot,
                generation=self._slot_generations[slot],
                state=PageState.RESERVED,
                fingerprint=self.fingerprint_digest,
            )
            self._slots[slot] = entry
            entries.append(entry)
        return tuple(entries)

    def _release(self, entry: _SlotEntry) -> None:
        if self._slots[entry.slot] is not entry:
            raise RuntimeError("stale KV tier slot release")
        self._slots[entry.slot] = None
        self._free_slots.append(entry.slot)

    def _validate_device_pages_available(
        self,
        page_ids: tuple[int, ...],
        direction: TransferDirection,
    ) -> None:
        retained = tuple(
            page_id
            for page_id in page_ids
            if page_id in self._device_page_owners
            or page_id in self._quarantined_device_pages
        )
        if not retained:
            return
        raise KVPageStateError(
            f"device KV pages {retained} already have retained transfer ownership",
            source_reusable=direction is TransferDirection.RESTORE,
            destination_reusable=direction is TransferDirection.OFFLOAD,
            destination_publishable=False,
        )

    def _reserve_device_pages(self, page_ids: tuple[int, ...]) -> object:
        # Availability is checked before any host-slot mutation.  The caller
        # still holds ``_lock``, so no competing transfer can claim a page
        # between that check and this reservation.
        owner = object()
        for page_id in page_ids:
            if (
                page_id in self._device_page_owners
                or page_id in self._quarantined_device_pages
            ):
                raise RuntimeError("device KV page reservation raced its validation")
            self._device_page_owners[page_id] = owner
        return owner

    def _release_device_pages(
        self,
        page_ids: tuple[int, ...],
        owner: object,
    ) -> None:
        for page_id in page_ids:
            if self._device_page_owners.get(page_id) is owner:
                del self._device_page_owners[page_id]

    def _quarantine_device_pages(
        self,
        page_ids: tuple[int, ...],
        owner: object,
    ) -> None:
        for page_id in page_ids:
            if self._device_page_owners.get(page_id) is owner:
                del self._device_page_owners[page_id]
                self._quarantined_device_pages.add(page_id)

    def _quarantine(
        self,
        entries: tuple[_SlotEntry, ...],
        error: BaseException,
        *,
        retain_keys: bool = False,
    ) -> None:
        for entry in entries:
            # A stale terminal callback must not quarantine a host slot that a
            # later transfer acquired under a newer slot generation.
            if self._slots[entry.slot] is not entry:
                continue
            if self._published.get(entry.key) is entry:
                del self._published[entry.key]
            self._quarantined[entry.slot] = (entry.key, error)
            # A replacement transfer leaves the previous committed entry
            # published.  Unknown D2H completion must nevertheless retain the
            # logical key: that DMA may still be reading the HBM page even
            # though an older host copy remains valid for lookup.
            if retain_keys or entry.key not in self._published:
                self._quarantined_keys.add(entry.key)
            self._in_flight.discard(entry.key)

    def begin_offload(
        self, keys: tuple[str, ...], page_ids: tuple[int, ...]
    ) -> TierTransfer:
        wanted = _keys(keys)
        pages = self._page_ids(page_ids, len(wanted))
        with self._lock:
            for key in wanted:
                if key in self._in_flight or key in self._quarantined_keys:
                    raise KVPageStateError(
                        f"KV page {key!r} already has retained in-flight ownership",
                        # This is not a fresh pre-submission rejection.  An
                        # earlier DMA may still be reading the same physical
                        # HBM page, so a retry must not launder that unknown
                        # completion into an ordinary safe eviction.
                        source_reusable=False,
                        destination_reusable=False,
                        destination_publishable=False,
                    )
                existing = self._published.get(key)
                if existing is not None and existing.state is not PageState.RESIDENT:
                    raise KVPageStateError(
                        f"KV page {key!r} is {existing.state}, not resident",
                        source_reusable=True,
                        destination_reusable=True,
                        destination_publishable=False,
                    )
            self._validate_device_pages_available(pages, TransferDirection.OFFLOAD)
            previous = tuple(self._published.get(key) for key in wanted)
            entries = self._reserve(wanted)
            for key, entry, old in zip(wanted, entries, previous, strict=True):
                entry.state = PageState.OFFLOADING
                self._in_flight.add(key)
                if old is None:
                    self._published[key] = entry
            device_page_owner = self._reserve_device_pages(pages)
            try:
                if self._host_fragment_planes is None:
                    sources = tuple(
                        _page_fragments(self.pool, page_id) for page_id in pages
                    )
                    destinations = tuple(
                        self._host_page(entry.slot) for entry in entries
                    )
                    completion = self.backend.begin_offload(sources, destinations)
                else:
                    if self._device_fragment_planes is None:
                        raise RuntimeError("fragment-major device planes are missing")
                    completion = self.backend.begin_offload_fragments(
                        self._device_fragment_planes,
                        pages,
                        self._host_fragment_planes,
                        tuple(entry.slot for entry in entries),
                    )
            except BaseException as error:
                self._quarantine(entries, error, retain_keys=True)
                self._quarantine_device_pages(pages, device_page_owner)
                raise KVPageTransferError(
                    "D2H submission failed with ambiguous completion; host slots "
                    "were quarantined",
                    source_reusable=False,
                    destination_reusable=False,
                    destination_publishable=False,
                ) from error
            self._sequence += 1
            return TierTransfer(
                self,
                sequence=self._sequence,
                direction=TransferDirection.OFFLOAD,
                keys=wanted,
                page_ids=pages,
                entries=entries,
                previous=previous,
                completion=completion,
                device_page_owner=device_page_owner,
            )

    def offload(self, keys: tuple[str, ...], page_ids: tuple[int, ...]) -> bool:
        transfer = self.begin_offload(keys, page_ids)
        transfer.wait()
        return transfer.finalize()

    def begin_restore(
        self,
        keys: tuple[str, ...],
        dest_page_ids: tuple[int, ...],
        *,
        expected_fingerprint: KVPageFingerprint | str | None = None,
    ) -> TierTransfer:
        wanted = _keys(keys)
        pages = self._page_ids(dest_page_ids, len(wanted))
        if expected_fingerprint is not None:
            self.validate_fingerprint(expected_fingerprint)
        with self._lock:
            self._validate_device_pages_available(pages, TransferDirection.RESTORE)
            entries: list[_SlotEntry] = []
            for key in wanted:
                entry = self._published.get(key)
                if entry is None or entry.state is not PageState.RESIDENT:
                    raise KVPageStateError(
                        f"KV page {key!r} is not resident",
                        source_reusable=True,
                        destination_reusable=True,
                        destination_publishable=False,
                    )
                if key in self._in_flight:
                    raise KVPageStateError(
                        f"KV page {key!r} already has an in-flight transfer",
                        source_reusable=True,
                        destination_reusable=True,
                        destination_publishable=False,
                    )
                if entry.fingerprint != self.fingerprint_digest:
                    raise KVPageIdentityError(
                        f"resident KV page {key!r} has a different layout fingerprint",
                        source_reusable=False,
                        destination_reusable=True,
                        destination_publishable=False,
                    )
                entries.append(entry)
            checksums = self._host_checksums(entries)
            mismatches = [
                entry
                for entry, actual in zip(entries, checksums, strict=True)
                if entry.checksum != actual
            ]
            if mismatches:
                error = KVPageChecksumError(
                    "resident KV page checksum mismatch; corrupted slots were quarantined",
                    source_reusable=False,
                    destination_reusable=True,
                    destination_publishable=False,
                )
                self._quarantine(tuple(mismatches), error)
                raise error
            for entry in entries:
                entry.state = PageState.RESTORING
                self._in_flight.add(entry.key)
            device_page_owner = self._reserve_device_pages(pages)
            try:
                if self._host_fragment_planes is None:
                    sources = tuple(self._host_page(entry.slot) for entry in entries)
                    destinations = tuple(
                        _page_fragments(self.pool, page_id) for page_id in pages
                    )
                    completion = self.backend.begin_restore(sources, destinations)
                else:
                    if self._device_fragment_planes is None:
                        raise RuntimeError("fragment-major device planes are missing")
                    completion = self.backend.begin_restore_fragments(
                        self._host_fragment_planes,
                        tuple(entry.slot for entry in entries),
                        self._device_fragment_planes,
                        pages,
                    )
            except BaseException as error:
                self._quarantine(tuple(entries), error)
                self._quarantine_device_pages(pages, device_page_owner)
                raise KVPageTransferError(
                    "H2D submission failed with ambiguous completion; source and "
                    "destination ownership were retained",
                    source_reusable=False,
                    destination_reusable=False,
                    destination_publishable=False,
                ) from error
            self._sequence += 1
            return TierTransfer(
                self,
                sequence=self._sequence,
                direction=TransferDirection.RESTORE,
                keys=wanted,
                page_ids=pages,
                entries=tuple(entries),
                previous=tuple(None for _ in entries),
                completion=completion,
                device_page_owner=device_page_owner,
                checksums=tuple(checksums),
            )

    def restore(
        self,
        keys: tuple[str, ...],
        dest_page_ids: tuple[int, ...],
        *,
        expected_fingerprint: KVPageFingerprint | str | None = None,
    ) -> bool:
        transfer = self.begin_restore(
            keys, dest_page_ids, expected_fingerprint=expected_fingerprint
        )
        transfer.wait()
        return transfer.finalize()

    def _fail_transfer(
        self, transfer: TierTransfer, error: BaseException
    ) -> KVPageTransferError:
        with self._lock:
            self._quarantine(
                transfer._entries,
                error,
                retain_keys=transfer.direction is TransferDirection.OFFLOAD,
            )
            self._quarantine_device_pages(
                transfer.page_ids,
                transfer._device_page_owner,
            )
            return KVPageTransferError(
                f"{transfer.direction} completion failed; ownership was quarantined",
                source_reusable=False,
                destination_reusable=False,
                destination_publishable=False,
            )

    def _cancel_transfer(
        self,
        transfer: TierTransfer,
        cause: BaseException | None = None,
    ) -> KVPageTransferError:
        with self._lock:
            error = KVPageTransferError(
                f"{transfer.direction} transfer {transfer.sequence} cancellation "
                + ("failed" if cause is not None else "retained ownership"),
                source_reusable=False,
                destination_reusable=False,
                destination_publishable=False,
            )
            self._quarantine(
                transfer._entries,
                error,
                retain_keys=transfer.direction is TransferDirection.OFFLOAD,
            )
            self._quarantine_device_pages(
                transfer.page_ids,
                transfer._device_page_owner,
            )
            return error

    def _fail_finalization(
        self,
        transfer: TierTransfer,
        error: BaseException,
    ) -> KVPageTransferError:
        """Quarantine after a known-complete copy whose host commit failed."""
        with self._lock:
            self._quarantine(transfer._entries, error)
            self._release_device_pages(
                transfer.page_ids,
                transfer._device_page_owner,
            )
            return KVPageTransferError(
                f"{transfer.direction} copy completed but ownership finalization failed",
                source_reusable=transfer.direction is TransferDirection.OFFLOAD,
                destination_reusable=transfer.direction is TransferDirection.RESTORE,
                destination_publishable=False,
            )

    def _finalize_transfer(self, transfer: TierTransfer) -> None:
        with self._lock:
            for entry in transfer._entries:
                if self._slots[entry.slot] is not entry:
                    raise KVPageStateError(
                        "stale KV tier completion cannot mutate a reused slot",
                        source_reusable=True,
                        destination_reusable=True,
                        destination_publishable=False,
                    )
            if transfer.direction is TransferDirection.OFFLOAD:
                checksums = self._host_checksums(transfer._entries)
                transfer._checksums = checksums
                for entry, old, checksum in zip(
                    transfer._entries, transfer._previous, checksums, strict=True
                ):
                    self._clock += 1
                    entry.checksum = checksum
                    entry.last_touch = self._clock
                    if old is not None and old.checksum == checksum:
                        old.last_touch = self._clock
                        self._release(entry)
                    else:
                        entry.state = PageState.RESIDENT
                        self._published[entry.key] = entry
                        if old is not None:
                            self._release(old)
                    self._in_flight.discard(entry.key)
                self._record_completed_transfer(transfer)
                self._release_device_pages(
                    transfer.page_ids,
                    transfer._device_page_owner,
                )
                return

            for entry in transfer._entries:
                if entry.state is not PageState.RESTORING:
                    raise KVPageStateError(
                        f"restore completion found page {entry.key!r} in {entry.state}",
                        source_reusable=False,
                        destination_reusable=True,
                        destination_publishable=False,
                    )
            for entry in transfer._entries:
                if self._published.get(entry.key) is entry:
                    del self._published[entry.key]
                self._in_flight.discard(entry.key)
                self._release(entry)
            self._record_completed_transfer(transfer)
            self._release_device_pages(
                transfer.page_ids,
                transfer._device_page_owner,
            )

    def _record_completed_transfer(self, transfer: TierTransfer) -> None:
        self._last_completed_transfer = CompletedTransferMetrics(
            sequence=transfer.sequence,
            direction=transfer.direction,
            page_count=len(transfer.page_ids),
            bytes_transferred=transfer.bytes_transferred,
            stream_id=transfer.stream_id,
            cuda_elapsed_ns=transfer.cuda_elapsed_ns,
        )

    def discard(self, keys: tuple[str, ...]) -> bool:
        wanted = _keys(keys)
        with self._lock:
            entries: list[_SlotEntry] = []
            for key in wanted:
                if key in self._in_flight:
                    raise KVPageStateError(
                        f"KV page {key!r} cannot be discarded while transfer is in flight",
                        source_reusable=False,
                        destination_reusable=False,
                        destination_publishable=False,
                    )
                entry = self._published.get(key)
                if entry is not None and entry.state is not PageState.RESIDENT:
                    raise KVPageStateError(
                        f"KV page {key!r} cannot be discarded from {entry.state}",
                        source_reusable=False,
                        destination_reusable=False,
                        destination_publishable=False,
                    )
                if entry is not None:
                    entries.append(entry)
            for entry in entries:
                del self._published[entry.key]
                self._release(entry)
            return True


# Short integration-facing name; keep the descriptive class public for docs.
DRAMKVTier = PinnedKVPageTier
