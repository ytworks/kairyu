"""Linux NUMA placement for CUDA page-locked host memory.

The DRAM KV tier needs a stronger contract than an allocation hint: every
page in its pinned slab must actually reside on the NUMA node local to the
rank's GPU.  This module keeps that contract independent of ``libnuma`` and
privileged NUMA syscalls:

* resolve the physical PCI device behind a CUDA-visible index through Torch;
* bind the calling thread to a disjoint share of GPU-local CPUs;
* first-touch the complete pinned allocation on that thread; and
* reconcile its VMAs with Linux's ``/proc/self/numa_maps`` accounting.

Nothing here runs for a CPU tier.  A CUDA caller on a platform without the
required Linux sysfs/procfs interfaces fails closed instead of silently using
remote or unknown memory.
"""

from __future__ import annotations

import ctypes
import os
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


GPU_LOCAL_FIRST_TOUCH_POLICY = "gpu-local-first-touch-attested-v1"
_NODE_PAGE = re.compile(r"N([0-9]+)=([0-9]+)")
_KERNEL_PAGE_KIB = re.compile(r"kernelpagesize_kB=([0-9]+)")
_AFFINITY_LOCK = threading.Lock()
_INITIAL_AFFINITY_BY_PID: dict[int, frozenset[int]] = {}


class NumaPlacementError(RuntimeError):
    """The requested GPU-local placement cannot be proven."""


@dataclass(frozen=True)
class CudaNumaPlan:
    """Rank-local CPU placement resolved before pinned allocation."""

    policy_name: str
    device_index: int
    tp_size: int
    tp_rank: int
    pci_bus_id: str
    numa_node: int
    cpu_affinity: tuple[int, ...]


@dataclass(frozen=True)
class NumaRangeResidency:
    """Kernel-reported residency for one completely faulted address range."""

    policies: tuple[str, ...]
    page_counts: tuple[tuple[int, int], ...]
    resident_pages: int
    resident_bytes: int
    os_page_size: int


@dataclass(frozen=True)
class PinnedHostNumaAttestation:
    """Immutable placement receipt retained by the DRAM KV tier."""

    policy_name: str
    numa_node: int
    cpu_affinity: tuple[int, ...]
    memory_policies: tuple[str, ...]
    page_counts: tuple[tuple[int, int], ...]
    resident_pages: int
    os_page_size: int


@dataclass(frozen=True)
class _VirtualMemoryArea:
    start: int
    end: int


@dataclass(frozen=True)
class _NumaMapRow:
    start: int
    policy: str
    page_counts: tuple[tuple[int, int], ...]
    kernel_page_size: int


def _parse_cpu_list(value: str) -> tuple[int, ...]:
    cpus: list[int] = []
    for raw_item in value.strip().split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "-" in item:
            lower_text, upper_text = item.split("-", 1)
            try:
                lower, upper = int(lower_text), int(upper_text)
            except ValueError as error:
                raise NumaPlacementError(f"invalid CPU range {item!r}") from error
            if lower < 0 or upper < lower:
                raise NumaPlacementError(f"invalid CPU range {item!r}")
            cpus.extend(range(lower, upper + 1))
        else:
            try:
                cpu = int(item)
            except ValueError as error:
                raise NumaPlacementError(f"invalid CPU id {item!r}") from error
            if cpu < 0:
                raise NumaPlacementError(f"invalid CPU id {item!r}")
            cpus.append(cpu)
    if not cpus or len(cpus) != len(set(cpus)):
        raise NumaPlacementError("CPU list is empty or contains duplicates")
    return tuple(cpus)


def _initial_process_affinity() -> frozenset[int]:
    pid = os.getpid()
    with _AFFINITY_LOCK:
        cached = _INITIAL_AFFINITY_BY_PID.get(pid)
        if cached is None:
            try:
                cached = frozenset(os.sched_getaffinity(0))
            except (AttributeError, OSError) as error:
                raise NumaPlacementError(
                    "CUDA DRAM KV placement requires Linux sched_getaffinity"
                ) from error
            if not cached:
                raise NumaPlacementError("the process has no allowed CPUs")
            _INITIAL_AFFINITY_BY_PID[pid] = cached
        return cached


def _cuda_pci_bus_id(device_index: int) -> str:
    import torch

    try:
        properties = torch.cuda.get_device_properties(device_index)
        domain = int(properties.pci_domain_id)
        bus = int(properties.pci_bus_id)
        device = int(properties.pci_device_id)
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise NumaPlacementError(
            f"cannot resolve PCI identity for CUDA device {device_index}"
        ) from error
    if domain < 0 or bus < 0 or device < 0:
        raise NumaPlacementError(
            f"CUDA device {device_index} returned an invalid PCI identity"
        )
    return f"{domain:04x}:{bus:02x}:{device:02x}.0"


def _read_numa_node(system_root: Path, pci_bus_id: str) -> int:
    path = system_root / "sys/bus/pci/devices" / pci_bus_id / "numa_node"
    try:
        node = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise NumaPlacementError(
            f"cannot read NUMA node for PCI device {pci_bus_id}: {path}"
        ) from error
    if node < 0:
        raise NumaPlacementError(
            f"PCI device {pci_bus_id} has no GPU-local NUMA node"
        )
    return node


def resolve_cuda_numa_plan(
    device: torch.device | str,
    *,
    tp_size: int,
    tp_rank: int,
    system_root: Path = Path("/"),
    allowed_cpus: frozenset[int] | None = None,
) -> CudaNumaPlan:
    """Resolve one rank's GPU-local, peer-partitioned CPU set.

    ``allowed_cpus`` and ``system_root`` are explicit seams for pure tests.  A
    production call snapshots its process affinity before the first plan is
    applied, so constructing a second tier in the same rank is idempotent.
    """

    import torch

    requested = torch.device(device)
    if requested.type != "cuda":
        raise ValueError("CUDA NUMA placement requires a CUDA device")
    if type(tp_size) is not int or tp_size < 1:
        raise ValueError("tp_size must be a positive integer")
    if type(tp_rank) is not int or not 0 <= tp_rank < tp_size:
        raise ValueError("tp_rank must be inside its TP degree")
    device_index = (
        torch.cuda.current_device()
        if requested.index is None
        else requested.index
    )
    base_index = device_index - tp_rank
    if base_index < 0:
        raise NumaPlacementError(
            "CUDA device index and TP rank do not describe a contiguous TP group"
        )

    pci_by_rank = tuple(
        _cuda_pci_bus_id(base_index + rank) for rank in range(tp_size)
    )
    node_by_rank = tuple(
        _read_numa_node(system_root, pci_bus_id)
        for pci_bus_id in pci_by_rank
    )
    local_node = node_by_rank[tp_rank]
    cpu_path = (
        system_root
        / "sys/devices/system/node"
        / f"node{local_node}"
        / "cpulist"
    )
    try:
        node_cpus = _parse_cpu_list(cpu_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise NumaPlacementError(
            f"cannot read CPU list for NUMA node {local_node}: {cpu_path}"
        ) from error
    permitted = allowed_cpus if allowed_cpus is not None else _initial_process_affinity()
    local_cpus = tuple(cpu for cpu in node_cpus if cpu in permitted)
    if not local_cpus:
        raise NumaPlacementError(
            f"TP rank {tp_rank} has no allowed CPU on GPU-local NUMA node {local_node}"
        )
    local_peers = tuple(
        rank for rank, node in enumerate(node_by_rank) if node == local_node
    )
    peer_ordinal = local_peers.index(tp_rank)
    rank_cpus = local_cpus[peer_ordinal :: len(local_peers)]
    if not rank_cpus:
        raise NumaPlacementError(
            f"cannot partition NUMA node {local_node} CPUs among TP ranks {local_peers}"
        )
    return CudaNumaPlan(
        policy_name=GPU_LOCAL_FIRST_TOUCH_POLICY,
        device_index=device_index,
        tp_size=tp_size,
        tp_rank=tp_rank,
        pci_bus_id=pci_by_rank[tp_rank],
        numa_node=local_node,
        cpu_affinity=rank_cpus,
    )


def apply_current_thread_affinity(plan: CudaNumaPlan) -> None:
    """Bind and verify the calling Linux task without affecting peer ranks."""

    try:
        os.sched_setaffinity(0, plan.cpu_affinity)
        observed = tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError) as error:
        raise NumaPlacementError(
            f"cannot bind TP rank {plan.tp_rank} to GPU-local CPUs"
        ) from error
    if observed != tuple(sorted(plan.cpu_affinity)):
        raise NumaPlacementError(
            f"TP rank {plan.tp_rank} CPU affinity differs after binding: {observed}"
        )


@contextmanager
def scoped_current_thread_affinity(plan: CudaNumaPlan) -> Iterator[None]:
    """Temporarily bind the calling Linux task and restore its exact mask.

    Thread-pool workers are shared infrastructure: retaining one tier's NUMA
    mask after a host-memory operation can route unrelated engines to the wrong
    socket. Snapshotting on every entry also makes nested scopes restore their
    immediate caller rather than a process-global mask.
    """

    try:
        previous = tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError) as error:
        raise NumaPlacementError(
            "cannot snapshot calling-thread CPU affinity before GPU-local work"
        ) from error
    if not previous:
        raise NumaPlacementError("the calling thread has no allowed CPUs")
    try:
        apply_current_thread_affinity(plan)
        yield
    finally:
        try:
            os.sched_setaffinity(0, previous)
            restored = tuple(sorted(os.sched_getaffinity(0)))
        except (AttributeError, OSError) as error:
            raise NumaPlacementError(
                "cannot restore calling-thread CPU affinity after GPU-local work"
            ) from error
        if restored != previous:
            raise NumaPlacementError(
                "calling-thread CPU affinity differs after restore: "
                f"expected {previous}, got {restored}"
            )


def _parse_proc_maps(value: str) -> tuple[_VirtualMemoryArea, ...]:
    result: list[_VirtualMemoryArea] = []
    for line_number, line in enumerate(value.splitlines(), 1):
        if not line.strip():
            continue
        interval = line.split(maxsplit=1)[0]
        try:
            lower_text, upper_text = interval.split("-", 1)
            lower, upper = int(lower_text, 16), int(upper_text, 16)
        except (ValueError, TypeError) as error:
            raise NumaPlacementError(
                f"invalid /proc/self/maps row {line_number}"
            ) from error
        if lower < 0 or upper <= lower:
            raise NumaPlacementError(
                f"invalid /proc/self/maps interval at row {line_number}"
            )
        result.append(_VirtualMemoryArea(lower, upper))
    if not result:
        raise NumaPlacementError("/proc/self/maps is empty")
    return tuple(sorted(result, key=lambda area: area.start))


def _parse_proc_numa_maps(value: str, os_page_size: int) -> dict[int, _NumaMapRow]:
    result: dict[int, _NumaMapRow] = {}
    for line_number, line in enumerate(value.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            start = int(parts[0], 16)
        except ValueError as error:
            raise NumaPlacementError(
                f"invalid /proc/self/numa_maps row {line_number}"
            ) from error
        if start in result:
            raise NumaPlacementError(
                f"duplicate /proc/self/numa_maps address at row {line_number}"
            )
        policy = parts[1] if len(parts) > 1 else "unknown"
        page_counts = tuple(
            sorted(
                (int(match.group(1)), int(match.group(2)))
                for item in parts[1:]
                if (match := _NODE_PAGE.fullmatch(item)) is not None
                and int(match.group(2)) > 0
            )
        )
        page_match = next(
            (
                match
                for item in parts[1:]
                if (match := _KERNEL_PAGE_KIB.fullmatch(item)) is not None
            ),
            None,
        )
        kernel_page_size = (
            int(page_match.group(1)) * 1024
            if page_match is not None
            else os_page_size
        )
        if kernel_page_size < 1:
            raise NumaPlacementError(
                f"invalid kernel page size at numa_maps row {line_number}"
            )
        result[start] = _NumaMapRow(
            start=start,
            policy=policy,
            page_counts=page_counts,
            kernel_page_size=kernel_page_size,
        )
    if not result:
        raise NumaPlacementError("/proc/self/numa_maps is empty")
    return result


def attest_numa_maps_range(
    address: int,
    nbytes: int,
    *,
    expected_node: int,
    maps_text: str,
    numa_maps_text: str,
    os_page_size: int,
) -> NumaRangeResidency:
    """Prove that every resident VMA page covering a range is local.

    The caller must first-touch the complete range.  Requiring every non-zero
    node count in every overlapping VMA to be the expected node is deliberately
    conservative when an allocator VMA is larger than the requested tensor.
    """

    if address <= 0 or nbytes <= 0:
        raise ValueError("NUMA attestation needs a non-empty address range")
    if expected_node < 0 or os_page_size < 1:
        raise ValueError("NUMA attestation node and OS page size must be valid")
    end = address + nbytes
    areas = _parse_proc_maps(maps_text)
    rows = _parse_proc_numa_maps(numa_maps_text, os_page_size)
    cursor = address
    policies: list[str] = []
    page_counts: dict[int, int] = {}
    resident_bytes = 0
    for area in areas:
        if area.end <= cursor:
            continue
        if area.start > cursor:
            break
        overlap_end = min(end, area.end)
        if overlap_end <= cursor:
            continue
        row = rows.get(area.start)
        if row is None:
            raise NumaPlacementError(
                f"numa_maps omits overlapping VMA at 0x{area.start:x}"
            )
        if not row.page_counts:
            raise NumaPlacementError(
                f"overlapping VMA at 0x{area.start:x} has no resident pages"
            )
        remote_nodes = tuple(
            node for node, count in row.page_counts if count and node != expected_node
        )
        if remote_nodes:
            raise NumaPlacementError(
                "pinned host slab has pages on remote NUMA nodes "
                f"{remote_nodes}; expected node {expected_node}"
            )
        policies.append(row.policy)
        for node, count in row.page_counts:
            page_counts[node] = page_counts.get(node, 0) + count
            resident_bytes += count * row.kernel_page_size
        cursor = overlap_end
        if cursor == end:
            break
    if cursor != end:
        raise NumaPlacementError(
            f"process maps do not cover pinned slab range through 0x{end:x}"
        )
    aligned_start = address - address % os_page_size
    aligned_end = ((end + os_page_size - 1) // os_page_size) * os_page_size
    required_bytes = aligned_end - aligned_start
    if resident_bytes < required_bytes:
        raise NumaPlacementError(
            "numa_maps resident bytes do not cover the completely touched pinned slab"
        )
    resident_pages = sum(page_counts.values())
    return NumaRangeResidency(
        policies=tuple(policies),
        page_counts=tuple(sorted(page_counts.items())),
        resident_pages=resident_pages,
        resident_bytes=resident_bytes,
        os_page_size=os_page_size,
    )


def first_touch_and_attest_pinned_host(
    tensor: torch.Tensor,
    plan: CudaNumaPlan,
    *,
    proc_root: Path = Path("/proc/self"),
) -> PinnedHostNumaAttestation:
    """First-touch and attest one contiguous CUDA-pinned CPU tensor."""

    if (
        tensor.device.type != "cpu"
        or not tensor.is_contiguous()
        or not tensor.is_pinned()
    ):
        raise NumaPlacementError(
            "NUMA attestation requires contiguous CUDA-pinned CPU storage"
        )
    observed_affinity = tuple(sorted(os.sched_getaffinity(0)))
    if observed_affinity != tuple(sorted(plan.cpu_affinity)):
        raise NumaPlacementError(
            "calling thread left its GPU-local CPU affinity before first touch"
        )
    address = int(tensor.data_ptr())
    nbytes = tensor.numel() * tensor.element_size()
    if address <= 0 or nbytes <= 0:
        raise NumaPlacementError("pinned host storage has an empty address range")
    ctypes.memset(address, 0, nbytes)
    try:
        maps_text = (proc_root / "maps").read_text(encoding="utf-8")
        numa_maps_text = (proc_root / "numa_maps").read_text(encoding="utf-8")
    except OSError as error:
        raise NumaPlacementError(
            "CUDA DRAM KV placement requires readable /proc/self maps"
        ) from error
    os_page_size = os.sysconf("SC_PAGE_SIZE")
    residency = attest_numa_maps_range(
        address,
        nbytes,
        expected_node=plan.numa_node,
        maps_text=maps_text,
        numa_maps_text=numa_maps_text,
        os_page_size=os_page_size,
    )
    return PinnedHostNumaAttestation(
        policy_name=plan.policy_name,
        numa_node=plan.numa_node,
        cpu_affinity=plan.cpu_affinity,
        memory_policies=residency.policies,
        page_counts=residency.page_counts,
        resident_pages=residency.resident_pages,
        os_page_size=residency.os_page_size,
    )


__all__ = [
    "CudaNumaPlan",
    "GPU_LOCAL_FIRST_TOUCH_POLICY",
    "NumaPlacementError",
    "NumaRangeResidency",
    "PinnedHostNumaAttestation",
    "apply_current_thread_affinity",
    "attest_numa_maps_range",
    "first_touch_and_attest_pinned_host",
    "resolve_cuda_numa_plan",
    "scoped_current_thread_affinity",
]
