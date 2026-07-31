from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from kairyu.engine.core import numa
from kairyu.engine.core.numa import (
    GPU_LOCAL_FIRST_TOUCH_POLICY,
    CudaNumaPlan,
    NumaPlacementError,
    apply_current_thread_affinity,
    attest_numa_maps_range,
    first_touch_and_attest_pinned_host,
    resolve_cuda_numa_plan,
    scoped_current_thread_affinity,
)


def _write_topology(
    root: Path,
    pci_nodes: dict[str, int],
    cpu_lists: dict[int, str],
) -> None:
    for pci_bus_id, node in pci_nodes.items():
        path = root / "sys/bus/pci/devices" / pci_bus_id / "numa_node"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{node}\n", encoding="utf-8")
    for node, cpu_list in cpu_lists.items():
        path = root / "sys/devices/system/node" / f"node{node}" / "cpulist"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{cpu_list}\n", encoding="utf-8")


def _plan(cpu_affinity: tuple[int, ...], *, tp_rank: int = 0) -> CudaNumaPlan:
    return CudaNumaPlan(
        policy_name=GPU_LOCAL_FIRST_TOUCH_POLICY,
        device_index=tp_rank,
        tp_size=2,
        tp_rank=tp_rank,
        pci_bus_id=f"0000:{16 + tp_rank:02x}:00.0",
        numa_node=tp_rank,
        cpu_affinity=cpu_affinity,
    )


def test_resolve_cuda_plan_partitions_only_gpu_local_allowed_cpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pci = {
        0: "0000:10:00.0",
        1: "0000:11:00.0",
        2: "0000:20:00.0",
        3: "0000:21:00.0",
    }
    _write_topology(
        tmp_path,
        {
            pci[0]: 0,
            pci[1]: 0,
            pci[2]: 1,
            pci[3]: 1,
        },
        {0: "0-3,8-11", 1: "4-7,12-15"},
    )
    monkeypatch.setattr(numa, "_cuda_pci_bus_id", pci.__getitem__)

    plan = resolve_cuda_numa_plan(
        "cuda:1",
        tp_size=4,
        tp_rank=1,
        system_root=tmp_path,
        allowed_cpus=frozenset(range(1, 16)),
    )

    assert plan.policy_name == GPU_LOCAL_FIRST_TOUCH_POLICY
    assert plan.device_index == 1
    assert plan.pci_bus_id == pci[1]
    assert plan.numa_node == 0
    assert plan.cpu_affinity == (2, 8, 10)


def test_resolve_cuda_plan_rejects_missing_local_allowed_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pci = "0000:10:00.0"
    _write_topology(tmp_path, {pci: 0}, {0: "0-3"})
    monkeypatch.setattr(numa, "_cuda_pci_bus_id", lambda _index: pci)

    with pytest.raises(NumaPlacementError, match="no allowed CPU"):
        resolve_cuda_numa_plan(
            "cuda:0",
            tp_size=1,
            tp_rank=0,
            system_root=tmp_path,
            allowed_cpus=frozenset({8, 9}),
        )


def test_apply_affinity_verifies_exact_calling_thread_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current: set[int] = {0, 1, 2, 3}

    def set_affinity(_pid: int, cpus) -> None:
        current.clear()
        current.update(cpus)

    monkeypatch.setattr(os, "sched_setaffinity", set_affinity)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: current)
    plan = CudaNumaPlan(
        policy_name=GPU_LOCAL_FIRST_TOUCH_POLICY,
        device_index=0,
        tp_size=1,
        tp_rank=0,
        pci_bus_id="0000:10:00.0",
        numa_node=0,
        cpu_affinity=(1, 3),
    )

    apply_current_thread_affinity(plan)

    assert current == {1, 3}


def test_scoped_affinity_is_nested_and_restores_each_callers_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current: set[int] = {0, 1, 2, 3}

    def set_affinity(_pid: int, cpus) -> None:
        current.clear()
        current.update(cpus)

    monkeypatch.setattr(os, "sched_setaffinity", set_affinity)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(current))

    with scoped_current_thread_affinity(_plan((0, 2))):
        assert current == {0, 2}
        with scoped_current_thread_affinity(_plan((2,), tp_rank=1)):
            assert current == {2}
        assert current == {0, 2}

    assert current == {0, 1, 2, 3}


def test_scoped_affinity_restores_after_body_base_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current: set[int] = {0, 1, 2, 3}

    def set_affinity(_pid: int, cpus) -> None:
        current.clear()
        current.update(cpus)

    monkeypatch.setattr(os, "sched_setaffinity", set_affinity)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(current))

    with pytest.raises(KeyboardInterrupt, match="injected process control"):
        with scoped_current_thread_affinity(_plan((1, 3))):
            assert current == {1, 3}
            raise KeyboardInterrupt("injected process control")

    assert current == {0, 1, 2, 3}


def test_attest_numa_maps_range_accepts_complete_local_multi_vma_range() -> None:
    residency = attest_numa_maps_range(
        0x1800,
        0x2000,
        expected_node=2,
        maps_text=(
            "00001000-00003000 rw-p 00000000 00:00 0\n"
            "00003000-00005000 rw-p 00000000 00:00 0\n"
        ),
        numa_maps_text=(
            "00001000 default anon=2 dirty=2 N2=2 kernelpagesize_kB=4\n"
            "00003000 bind:2 anon=2 dirty=2 N2=2 kernelpagesize_kB=4\n"
        ),
        os_page_size=4096,
    )

    assert residency.policies == ("default", "bind:2")
    assert residency.page_counts == ((2, 4),)
    assert residency.resident_pages == 4
    assert residency.resident_bytes == 16_384


@pytest.mark.parametrize(
    ("maps_text", "numa_maps_text", "message"),
    [
        (
            "00001000-00003000 rw-p 00000000 00:00 0\n",
            "00001000 default anon=2 N0=1 N1=1 kernelpagesize_kB=4\n",
            "remote NUMA nodes",
        ),
        (
            "00001000-00002000 rw-p 00000000 00:00 0\n"
            "00003000-00004000 rw-p 00000000 00:00 0\n",
            "00001000 default N0=1 kernelpagesize_kB=4\n"
            "00003000 default N0=1 kernelpagesize_kB=4\n",
            "do not cover",
        ),
        (
            "00001000-00003000 rw-p 00000000 00:00 0\n",
            "00002000 default N0=1 kernelpagesize_kB=4\n",
            "omits overlapping VMA",
        ),
        (
            "00001000-00003000 rw-p 00000000 00:00 0\n",
            "00001000 default anon=0 kernelpagesize_kB=4\n",
            "no resident pages",
        ),
        (
            "00001000-00003000 rw-p 00000000 00:00 0\n",
            "00001000 default N0=1 kernelpagesize_kB=4\n",
            "resident bytes do not cover",
        ),
    ],
)
def test_attest_numa_maps_range_fails_closed(
    maps_text: str,
    numa_maps_text: str,
    message: str,
) -> None:
    with pytest.raises(NumaPlacementError, match=message):
        attest_numa_maps_range(
            0x1000,
            0x2000,
            expected_node=0,
            maps_text=maps_text,
            numa_maps_text=numa_maps_text,
            os_page_size=4096,
        )


class _PinnedBuffer:
    def __init__(self, nbytes: int) -> None:
        self.storage = ctypes.create_string_buffer(b"x" * nbytes, nbytes)
        self.device = SimpleNamespace(type="cpu")

    def is_contiguous(self) -> bool:
        return True

    def is_pinned(self) -> bool:
        return True

    def data_ptr(self) -> int:
        return ctypes.addressof(self.storage)

    def numel(self) -> int:
        return len(self.storage)

    def element_size(self) -> int:
        return 1


def test_first_touch_zeroes_complete_buffer_and_retains_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    size = 8192
    tensor = _PinnedBuffer(size)
    address = tensor.data_ptr()
    page_size = os.sysconf("SC_PAGE_SIZE")
    vma_start = address - address % page_size
    vma_end = math.ceil((address + size) / page_size) * page_size
    pages = (vma_end - vma_start) // page_size
    (tmp_path / "maps").write_text(
        f"{vma_start:x}-{vma_end:x} rw-p 00000000 00:00 0\n",
        encoding="utf-8",
    )
    (tmp_path / "numa_maps").write_text(
        f"{vma_start:x} default anon={pages} N3={pages} kernelpagesize_kB=4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {2, 4})
    plan = CudaNumaPlan(
        policy_name=GPU_LOCAL_FIRST_TOUCH_POLICY,
        device_index=0,
        tp_size=1,
        tp_rank=0,
        pci_bus_id="0000:10:00.0",
        numa_node=3,
        cpu_affinity=(2, 4),
    )

    receipt = first_touch_and_attest_pinned_host(
        tensor,  # type: ignore[arg-type]
        plan,
        proc_root=tmp_path,
    )

    assert bytes(tensor.storage) == b"\x00" * size
    assert receipt.policy_name == GPU_LOCAL_FIRST_TOUCH_POLICY
    assert receipt.numa_node == 3
    assert receipt.cpu_affinity == (2, 4)
    assert receipt.page_counts == ((3, pages),)
