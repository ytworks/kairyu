from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
import torch

import kairyu.engine.core.kv_tier as kv_tier_module
from kairyu.engine.core import numa as numa_module
from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.engine.core.kv_tier import (
    FRAGMENT_MAJOR_HOST_LAYOUT,
    DRAMKVTier,
    KVPageCapacityError,
    KVPageChecksumError,
    KVPageIdentityError,
    KVPageStateError,
    KVPageTransferError,
    PageState,
    TorchPageTransferBackend,
    TransferDirection,
)
from kairyu.engine.core.numa import (
    GPU_LOCAL_FIRST_TOUCH_POLICY,
    CudaNumaPlan,
)


def _key(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _pool(
    *,
    num_pages: int = 4,
    page_size: int = 2,
    dtype: torch.dtype = torch.float32,
) -> PagedKVPool:
    return PagedKVPool(
        num_layers=2,
        num_pages=num_pages,
        page_size=page_size,
        num_kv_heads=1,
        head_dim=3,
        v_head_dim=2,
        dtype=dtype,
    )


def _fill(pool: PagedKVPool, page_id: int, seed: int) -> None:
    start = seed * 100
    for layer in range(pool.num_layers):
        k = torch.arange(pool.k[layer, page_id].numel(), dtype=torch.float32)
        v = torch.arange(pool.v[layer, page_id].numel(), dtype=torch.float32)
        pool.k[layer, page_id].copy_((k + start + layer * 10).reshape_as(pool.k[0, 0]))
        pool.v[layer, page_id].copy_((v - start - layer * 10).reshape_as(pool.v[0, 0]))


def _snapshot(pool: PagedKVPool, page_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    return pool.k[:, page_id].clone(), pool.v[:, page_id].clone()


class _ManualCompletion:
    def __init__(self, work) -> None:
        self._work = work
        self.done = False
        self.query_error: BaseException | None = None
        self.sync_error: BaseException | None = None
        self.cancel_error: BaseException | None = None
        self.cancel_called = False

    def complete(self) -> None:
        if not self.done:
            self._work()
            self.done = True

    def query(self) -> bool:
        if self.query_error is not None:
            raise self.query_error
        return self.done

    def synchronize(self) -> None:
        if self.sync_error is not None:
            raise self.sync_error
        self.complete()

    def cancel(self) -> None:
        self.cancel_called = True
        if self.cancel_error is not None:
            raise self.cancel_error


class _ManualBackend:
    def __init__(self) -> None:
        self.storage: torch.Tensor | None = None
        self.completions: list[_ManualCompletion] = []
        self.submission_count = 0
        self.fail_submit: BaseException | None = None
        self.corrupt_offload = False

    def allocate_host_pages(self, capacity_pages: int, page_nbytes: int) -> torch.Tensor:
        self.storage = torch.empty((capacity_pages, page_nbytes), dtype=torch.uint8)
        return self.storage

    def begin_offload(self, sources, destinations) -> _ManualCompletion:
        self.submission_count += 1
        if self.fail_submit is not None:
            raise self.fail_submit

        def work() -> None:
            for source, destination in zip(sources, destinations, strict=True):
                TorchPageTransferBackend._copy_offload_page(
                    source, destination, non_blocking=False
                )
            if self.corrupt_offload:
                destinations[0][0] ^= 1

        completion = _ManualCompletion(work)
        self.completions.append(completion)
        return completion

    def begin_restore(self, sources, destinations) -> _ManualCompletion:
        self.submission_count += 1
        if self.fail_submit is not None:
            raise self.fail_submit

        def work() -> None:
            for source, destination in zip(sources, destinations, strict=True):
                TorchPageTransferBackend._copy_restore_page(
                    source, destination, non_blocking=False
                )

        completion = _ManualCompletion(work)
        self.completions.append(completion)
        return completion


class _PartialFragmentBackend(TorchPageTransferBackend):
    """CPU fault injector for the production fragment-plane dispatch seam."""

    def __init__(self) -> None:
        super().__init__("cpu")
        self.host_layout = FRAGMENT_MAJOR_HOST_LAYOUT
        self.fail_offload = False
        self.fail_restore = False

    def _copy_first_run(
        self,
        host_planes,
        device_planes,
        host_slots,
        device_page_ids,
        *,
        restore: bool,
    ) -> None:
        runs = self._validated_fragment_planes(
            host_planes,
            device_planes,
            host_slots,
            device_page_ids,
        )
        self._copy_fragment_planes(
            host_planes[:1],
            device_planes[:1],
            runs[:1],
            restore=restore,
            non_blocking=False,
        )

    def begin_offload_fragments(
        self,
        sources,
        source_page_ids,
        destinations,
        destination_slots,
    ):
        if self.fail_offload:
            self._copy_first_run(
                destinations,
                sources,
                destination_slots,
                source_page_ids,
                restore=False,
            )
            raise RuntimeError("partial fragment offload submission")
        return super().begin_offload_fragments(
            sources,
            source_page_ids,
            destinations,
            destination_slots,
        )

    def begin_restore_fragments(
        self,
        sources,
        source_slots,
        destinations,
        destination_page_ids,
    ):
        if self.fail_restore:
            self._copy_first_run(
                sources,
                destinations,
                source_slots,
                destination_page_ids,
                restore=True,
            )
            raise RuntimeError("partial fragment restore submission")
        return super().begin_restore_fragments(
            sources,
            source_slots,
            destinations,
            destination_page_ids,
        )


def _tier(
    pool: PagedKVPool,
    *,
    capacity: int = 2,
    backend: _ManualBackend | None = None,
    model_id: str = "org/model@0123456789abcdef",
    tp_size: int = 2,
    tp_rank: int = 0,
) -> DRAMKVTier:
    return DRAMKVTier(
        pool,
        model_id=model_id,
        tp_size=tp_size,
        tp_rank=tp_rank,
        capacity_pages=capacity,
        backend=backend,
    )


def _numa_plan(cpu_affinity: tuple[int, ...]) -> CudaNumaPlan:
    return CudaNumaPlan(
        policy_name=GPU_LOCAL_FIRST_TOUCH_POLICY,
        device_index=0,
        tp_size=1,
        tp_rank=0,
        pci_bus_id="0000:10:00.0",
        numa_node=0,
        cpu_affinity=cpu_affinity,
    )


def test_cuda_constructor_failure_restores_calling_thread_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current: set[int] = {0, 1, 2, 3}
    plan = _numa_plan((1, 3))

    def set_affinity(_pid: int, cpus) -> None:
        current.clear()
        current.update(cpus)

    monkeypatch.setattr(numa_module.os, "sched_setaffinity", set_affinity)
    monkeypatch.setattr(
        numa_module.os,
        "sched_getaffinity",
        lambda _pid: set(current),
    )
    monkeypatch.setattr(
        numa_module,
        "resolve_cuda_numa_plan",
        lambda *_args, **_kwargs: plan,
    )
    fingerprint = SimpleNamespace(digest="f" * 64, page_nbytes=8)
    monkeypatch.setattr(
        kv_tier_module.KVPageFingerprint,
        "from_pool",
        classmethod(lambda _cls, *_args, **_kwargs: fingerprint),
    )

    host_pages = SimpleNamespace(
        shape=(1, 8),
        dtype=torch.uint8,
        device=SimpleNamespace(type="cpu"),
        is_contiguous=lambda: True,
        is_pinned=lambda: True,
    )

    class FailingCudaBackend:
        def __init__(self, _device) -> None:
            assert current == {1, 3}

        def allocate_host_pages(self, capacity_pages, page_nbytes):
            assert current == {1, 3}
            assert (capacity_pages, page_nbytes) == (1, 8)
            return host_pages

    monkeypatch.setattr(
        kv_tier_module,
        "TorchPageTransferBackend",
        FailingCudaBackend,
    )

    def fail_first_touch(_host_pages, actual_plan) -> None:
        assert current == {1, 3}
        assert actual_plan is plan
        raise RuntimeError("injected first-touch attestation failure")

    monkeypatch.setattr(
        numa_module,
        "first_touch_and_attest_pinned_host",
        fail_first_touch,
    )
    fake_pool = SimpleNamespace(
        num_pages=1,
        k=SimpleNamespace(device=SimpleNamespace(type="cuda")),
    )

    with pytest.raises(RuntimeError, match="first-touch attestation failure"):
        DRAMKVTier(
            fake_pool,
            model_id="model",
            tp_size=1,
            tp_rank=0,
            capacity_pages=1,
        )

    assert current == {0, 1, 2, 3}


def test_cpu_tier_never_enters_a_numa_affinity_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_scope(_plan):
        raise AssertionError("CPU tier entered a NUMA affinity scope")

    monkeypatch.setattr(
        numa_module,
        "scoped_current_thread_affinity",
        unexpected_scope,
    )
    pool = _pool(num_pages=1)
    _fill(pool, 0, 2)
    tier = _tier(pool, capacity=1, backend=_ManualBackend())

    assert tier.offload((_key("cpu-no-numa"),), (0,))


def test_host_checksum_batches_scope_and_restore_the_worker_thread_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = threading.local()
    original_affinity = (0, 1, 2, 3)
    plan = _numa_plan((1, 3))

    def current_affinity() -> set[int]:
        if not hasattr(state, "affinity"):
            state.affinity = set(original_affinity)
        return state.affinity

    def set_affinity(_pid: int, cpus) -> None:
        state.affinity = set(cpus)

    monkeypatch.setattr(numa_module.os, "sched_setaffinity", set_affinity)
    monkeypatch.setattr(
        numa_module.os,
        "sched_getaffinity",
        lambda _pid: set(current_affinity()),
    )
    observed: list[tuple[int, ...]] = []
    real_checksum = kv_tier_module._checksum

    def checking_checksum(page: torch.Tensor) -> str:
        observed.append(tuple(sorted(current_affinity())))
        return real_checksum(page)

    monkeypatch.setattr(kv_tier_module, "_checksum", checking_checksum)
    pool = _pool(num_pages=2)
    _fill(pool, 0, 4)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=1, backend=backend)
    tier._numa_plan = plan
    key = _key("worker-affinity")

    def offload_and_report() -> tuple[int, ...]:
        assert tier.offload((key,), (0,))
        return tuple(sorted(current_affinity()))

    def restore_and_report() -> tuple[int, ...]:
        assert tier.restore((key,), (1,))
        return tuple(sorted(current_affinity()))

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(offload_and_report).result() == original_affinity
        assert executor.submit(restore_and_report).result() == original_affinity

    assert observed == [plan.cpu_affinity, plan.cpu_affinity]


def test_fingerprint_binds_model_layout_tp_rank_dtype_and_page_size() -> None:
    base = _tier(_pool())
    variants = [
        _tier(_pool(), model_id="org/other@0123456789abcdef"),
        _tier(_pool(), tp_size=4, tp_rank=0),
        _tier(_pool(), tp_size=2, tp_rank=1),
        _tier(_pool(dtype=torch.float16)),
        _tier(_pool(page_size=4)),
    ]

    assert len(base.fingerprint.digest) == 64
    assert base.fingerprint.layout.startswith("layer-major:")
    assert base.fingerprint.page_nbytes == base.page_nbytes
    assert all(item.fingerprint.digest != base.fingerprint.digest for item in variants)
    assert "org/model@0123456789abcdef" in base.fingerprint.canonical_json
    assert base.host_numa_policy is None
    assert base.host_numa_node is None
    assert base.host_cpu_affinity == ()
    assert dict(base.host_numa_page_counts) == {}

    with pytest.raises(KVPageIdentityError) as mismatch:
        base.validate_fingerprint(variants[-1].fingerprint)
    assert mismatch.value.source_reusable
    assert mismatch.value.destination_reusable
    assert not mismatch.value.destination_publishable


def test_fragment_plane_copy_groups_only_jointly_contiguous_index_runs() -> None:
    contiguous = TorchPageTransferBackend._jointly_contiguous_runs(
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    )
    fragmented_host = TorchPageTransferBackend._jointly_contiguous_runs(
        (0, 1, 3, 4),
        (4, 5, 6, 7),
    )
    fragmented_device = TorchPageTransferBackend._jointly_contiguous_runs(
        (0, 1, 2, 3),
        (4, 5, 7, 8),
    )

    assert contiguous == ((0, 4, 4),)
    assert fragmented_host == ((0, 4, 2), (3, 6, 2))
    assert fragmented_device == ((0, 4, 2), (2, 7, 2))


def test_fragment_major_checksum_preserves_the_logical_page_byte_abi() -> None:
    pool = _pool(num_pages=2)
    _fill(pool, 0, 3)
    _fill(pool, 1, 7)
    page_nbytes = kv_tier_module.KVPageFingerprint.from_pool(
        pool,
        model_id="checksum-layout",
        tp_size=1,
        tp_rank=0,
    ).page_nbytes
    storage = torch.empty((2, page_nbytes), dtype=torch.uint8)
    planes = kv_tier_module._fragment_major_planes(storage, pool)
    expected = []
    for slot in range(2):
        source = kv_tier_module._page_fragments(pool, slot)
        destination = kv_tier_module._fragment_major_page(planes, slot)
        TorchPageTransferBackend._copy_matching_fragments(
            source,
            destination,
            non_blocking=False,
        )
        packed = torch.empty(page_nbytes, dtype=torch.uint8)
        TorchPageTransferBackend._copy_offload_page(
            source,
            packed,
            non_blocking=False,
        )
        expected.append(kv_tier_module._checksum(packed))
        assert kv_tier_module._checksum(destination) == expected[-1]

    assert kv_tier_module._fragment_major_checksums(
        storage,
        planes,
        (0, 1),
        page_nbytes=page_nbytes,
    ) == tuple(expected)
    planes[0][0, 0] ^= 1
    corrupted = kv_tier_module._fragment_major_checksums(
        storage,
        planes,
        (0, 1),
        page_nbytes=page_nbytes,
    )
    assert corrupted[0] != expected[0]
    assert corrupted[1] == expected[1]


@pytest.mark.parametrize(
    "slots",
    [
        (0, 1, 2, 3),
        (0, 2, 3),
        (3, 1, 0),
    ],
)
def test_equal_fragment_batched_checksums_preserve_legacy_page_abi(
    slots: tuple[int, ...],
) -> None:
    pool = PagedKVPool(
        num_layers=3,
        num_pages=4,
        page_size=2,
        num_kv_heads=1,
        head_dim=4,
        v_head_dim=4,
        dtype=torch.float32,
    )
    for page_id in range(4):
        _fill(pool, page_id, page_id + 1)
    page_nbytes = kv_tier_module.KVPageFingerprint.from_pool(
        pool,
        model_id="equal-fragment-checksum-layout",
        tp_size=1,
        tp_rank=0,
    ).page_nbytes
    storage = torch.empty((4, page_nbytes), dtype=torch.uint8)
    planes = kv_tier_module._fragment_major_planes(storage, pool)
    expected_by_slot: list[str] = []
    for slot in range(4):
        source = kv_tier_module._page_fragments(pool, slot)
        TorchPageTransferBackend._copy_matching_fragments(
            source,
            kv_tier_module._fragment_major_page(planes, slot),
            non_blocking=False,
        )
        legacy_page = torch.empty(page_nbytes, dtype=torch.uint8)
        TorchPageTransferBackend._copy_offload_page(
            source,
            legacy_page,
            non_blocking=False,
        )
        expected_by_slot.append(kv_tier_module._checksum(legacy_page))

    assert kv_tier_module._fragment_major_checksums(
        storage,
        planes,
        slots,
        page_nbytes=page_nbytes,
    ) == tuple(expected_by_slot[slot] for slot in slots)


def test_released_host_slots_are_reassigned_in_ascending_contiguous_order() -> None:
    pool = _pool(num_pages=8)
    for page_id in range(4):
        _fill(pool, page_id, page_id + 1)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=4, backend=backend)
    keys = tuple(_key(f"slot-{index}") for index in range(4))

    offload = tier.begin_offload(keys, (0, 1, 2, 3))
    backend.completions[-1].complete()
    assert offload.finalize()
    restore = tier.begin_restore(keys, (4, 5, 6, 7))
    backend.completions[-1].complete()
    assert restore.finalize()

    next_keys = tuple(_key(f"slot-next-{index}") for index in range(4))
    next_offload = tier.begin_offload(next_keys, (0, 1, 2, 3))
    assert [entry.slot for entry in next_offload._entries] == [0, 1, 2, 3]
    backend.completions[-1].complete()
    assert next_offload.finalize()


def test_async_state_machine_never_publishes_or_reuses_before_finalize() -> None:
    pool = _pool(num_pages=3)
    _fill(pool, 0, 7)
    expected = _snapshot(pool, 0)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=1, backend=backend)
    key = _key("page-0")

    offload = tier.begin_offload((key,), (0,))
    assert offload.direction is TransferDirection.OFFLOAD
    assert tier.state(key) is PageState.OFFLOADING
    assert tier.available_prefix((key,)) == 0
    assert not offload.source_reusable
    with pytest.raises(KVPageStateError):
        tier.resident_metadata(key)
    with pytest.raises(KVPageCapacityError) as full:
        tier.begin_offload((_key("other"),), (1,))
    assert full.value.source_reusable
    assert not offload.ready()

    backend.completions[-1].complete()
    assert offload.ready()
    assert offload.source_reusable
    assert tier.state(key) is PageState.OFFLOADING
    with pytest.raises(KVPageStateError):
        tier.resident_metadata(key)
    assert offload.finalize()
    assert tier.state(key) is PageState.RESIDENT
    assert tier.available_prefix((key,)) == 1
    assert tier.resident_metadata(key).checksum == offload.checksums[0]

    pool.k[:, 1].zero_()
    pool.v[:, 1].zero_()
    restore = tier.begin_restore((key,), (1,))
    assert restore.direction is TransferDirection.RESTORE
    assert tier.state(key) is PageState.RESTORING
    assert tier.available_prefix((key,)) == 0
    assert not restore.destination_reusable
    assert not restore.destination_publishable
    with pytest.raises(KVPageStateError):
        tier.discard((key,))

    backend.completions[-1].complete()
    assert restore.ready()
    assert restore.destination_reusable
    assert not restore.destination_publishable
    assert restore.finalize()
    assert restore.destination_publishable
    assert tier.state(key) is None
    assert tier.free_pages == 1
    assert torch.equal(pool.k[:, 1], expected[0])
    assert torch.equal(pool.v[:, 1], expected[1])
    followup = tier.begin_offload((_key("restored-page-reuse"),), (1,))
    assert backend.submission_count == 3
    followup.cancel()


def test_device_page_stays_reserved_until_successful_finalize() -> None:
    pool = _pool(num_pages=2)
    _fill(pool, 0, 3)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=2, backend=backend)

    transfer = tier.begin_offload((_key("reserved-first"),), (0,))
    backend.completions[0].complete()
    assert transfer.ready()

    # Physical completion alone does not release the tier's page reservation:
    # logical ownership publication must also succeed first.
    with pytest.raises(KVPageStateError) as retained:
        tier.begin_offload((_key("reserved-second"),), (0,))
    assert not retained.value.source_reusable
    assert retained.value.destination_reusable
    assert backend.submission_count == 1

    assert transfer.finalize()
    replacement = tier.begin_offload((_key("reserved-second"),), (0,))
    assert backend.submission_count == 2
    replacement.cancel()


def test_in_flight_offload_page_blocks_different_keys_in_both_directions() -> None:
    pool = _pool(num_pages=3)
    _fill(pool, 0, 4)
    _fill(pool, 1, 5)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=3, backend=backend)
    restore_key = _key("offload-overlap-resident")
    assert tier.offload((restore_key,), (1,))

    transfer = tier.begin_offload((_key("offload-overlap-active"),), (0,))
    submissions = backend.submission_count

    with pytest.raises(KVPageStateError) as offload_conflict:
        tier.begin_offload((_key("offload-overlap-other"),), (0,))
    assert not offload_conflict.value.source_reusable
    assert offload_conflict.value.destination_reusable

    with pytest.raises(KVPageStateError) as restore_conflict:
        tier.begin_restore((restore_key,), (0,))
    assert restore_conflict.value.source_reusable
    assert not restore_conflict.value.destination_reusable
    assert backend.submission_count == submissions
    transfer.cancel()


def test_in_flight_restore_page_blocks_different_keys_in_both_directions() -> None:
    pool = _pool(num_pages=4)
    _fill(pool, 0, 6)
    _fill(pool, 1, 7)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=3, backend=backend)
    first = _key("restore-overlap-first")
    second = _key("restore-overlap-second")
    assert tier.offload((first, second), (0, 1))

    transfer = tier.begin_restore((first,), (2,))
    submissions = backend.submission_count

    with pytest.raises(KVPageStateError) as offload_conflict:
        tier.begin_offload((_key("restore-overlap-other"),), (2,))
    assert not offload_conflict.value.source_reusable
    assert offload_conflict.value.destination_reusable

    with pytest.raises(KVPageStateError) as restore_conflict:
        tier.begin_restore((second,), (2,))
    assert restore_conflict.value.source_reusable
    assert not restore_conflict.value.destination_reusable
    assert backend.submission_count == submissions
    transfer.cancel()


def _ready_restore_transfer(
    pool: PagedKVPool,
    backend: _ManualBackend,
    tier: DRAMKVTier,
):
    key = _key("terminal-race-resident")
    _fill(pool, 0, 11)
    assert tier.offload((key,), (0,))
    transfer = tier.begin_restore((key,), (1,))
    backend.completions[-1].complete()
    assert transfer.ready()
    return transfer


def test_concurrent_double_finalize_cannot_release_a_new_transfer_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _pool(num_pages=3)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=3, backend=backend)
    transfer = _ready_restore_transfer(pool, backend, tier)
    finalized = Event()
    release = Event()
    original_finalize = tier._finalize_transfer

    def blocking_finalize(candidate) -> None:
        original_finalize(candidate)
        finalized.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr(tier, "_finalize_transfer", blocking_finalize)
    second_started = Event()

    def finalize_again() -> bool:
        second_started.set()
        return transfer.finalize()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(transfer.finalize)
        assert finalized.wait(timeout=2)
        replacement = tier.begin_offload((_key("double-finalize-new-owner"),), (1,))
        submissions = backend.submission_count
        second = executor.submit(finalize_again)
        assert second_started.wait(timeout=2)
        assert not second.done()
        with pytest.raises(KVPageStateError):
            tier.begin_offload((_key("double-finalize-overlap"),), (1,))
        assert backend.submission_count == submissions
        release.set()
        assert first.result(timeout=2) is True
        assert second.result(timeout=2) is True

    with pytest.raises(KVPageStateError):
        tier.begin_offload((_key("double-finalize-after"),), (1,))
    assert backend.submission_count == submissions
    replacement.cancel()


def test_concurrent_finalize_then_cancel_cannot_quarantine_a_new_transfer_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _pool(num_pages=3)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=3, backend=backend)
    transfer = _ready_restore_transfer(pool, backend, tier)
    finalized = Event()
    release = Event()
    original_finalize = tier._finalize_transfer

    def blocking_finalize(candidate) -> None:
        original_finalize(candidate)
        finalized.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr(tier, "_finalize_transfer", blocking_finalize)
    cancel_started = Event()

    def cancel_after_finalize() -> None:
        cancel_started.set()
        transfer.cancel()

    with ThreadPoolExecutor(max_workers=2) as executor:
        finalizing = executor.submit(transfer.finalize)
        assert finalized.wait(timeout=2)
        replacement = tier.begin_offload((_key("finalize-cancel-new-owner"),), (1,))
        submissions = backend.submission_count
        cancelling = executor.submit(cancel_after_finalize)
        assert cancel_started.wait(timeout=2)
        assert not cancelling.done()
        release.set()
        assert finalizing.result(timeout=2) is True
        with pytest.raises(KVPageStateError, match="already finalized"):
            cancelling.result(timeout=2)

    with pytest.raises(KVPageStateError):
        tier.begin_offload((_key("finalize-cancel-overlap"),), (1,))
    assert backend.submission_count == submissions
    assert tier.quarantined_device_pages == ()
    replacement.cancel()


def test_concurrent_cancel_then_finalize_has_one_fail_closed_terminal_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _pool(num_pages=3)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=3, backend=backend)
    transfer = _ready_restore_transfer(pool, backend, tier)
    cancelled = Event()
    release = Event()
    original_cancel = tier._cancel_transfer

    def blocking_cancel(candidate, cause=None):
        failure = original_cancel(candidate, cause)
        cancelled.set()
        assert release.wait(timeout=5)
        return failure

    monkeypatch.setattr(tier, "_cancel_transfer", blocking_cancel)
    finalize_started = Event()

    def finalize_after_cancel() -> bool:
        finalize_started.set()
        return transfer.finalize()

    submissions = backend.submission_count
    with ThreadPoolExecutor(max_workers=2) as executor:
        cancelling = executor.submit(transfer.cancel)
        assert cancelled.wait(timeout=2)
        finalizing = executor.submit(finalize_after_cancel)
        assert finalize_started.wait(timeout=2)
        assert not finalizing.done()
        with pytest.raises(KVPageStateError):
            tier.begin_offload((_key("cancel-finalize-overlap"),), (1,))
        assert backend.submission_count == submissions
        release.set()
        assert cancelling.result(timeout=2) is None
        with pytest.raises(KVPageTransferError) as failed:
            finalizing.result(timeout=2)

    assert not failed.value.source_reusable
    assert not failed.value.destination_reusable
    assert tier.quarantined_device_pages == (1,)


def test_cancel_after_ready_overrides_preterminal_reusability() -> None:
    pool = _pool(num_pages=2)
    _fill(pool, 0, 12)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=2, backend=backend)
    transfer = tier.begin_offload((_key("ready-then-cancel"),), (0,))
    backend.completions[0].complete()

    assert transfer.ready()
    assert transfer.source_reusable
    transfer.cancel()

    assert not transfer.source_reusable
    assert not transfer.destination_reusable
    assert not transfer.destination_publishable
    assert tier.quarantined_device_pages == (0,)


def test_blocking_multi_page_round_trip_consumes_host_entries() -> None:
    pool = _pool(num_pages=4)
    _fill(pool, 0, 2)
    _fill(pool, 1, 3)
    expected = (_snapshot(pool, 0), _snapshot(pool, 1))
    tier = _tier(pool, capacity=2)
    keys = (_key("a"), _key("b"))

    assert tier.offload(keys, (0, 1))
    offload_metrics = tier.last_completed_transfer
    assert offload_metrics is not None
    assert offload_metrics.direction is TransferDirection.OFFLOAD
    assert offload_metrics.page_count == 2
    assert offload_metrics.bytes_transferred == 2 * tier.page_nbytes
    assert tier.available_prefix(keys, min_pages=2) == 2
    assert tier.available_prefix((keys[0], _key("missing")), min_pages=2) == 0

    pool.k[:, 2:].zero_()
    pool.v[:, 2:].zero_()
    assert tier.restore(keys, (2, 3), expected_fingerprint=tier.fingerprint)
    restore_metrics = tier.last_completed_transfer
    assert restore_metrics is not None
    assert restore_metrics.direction is TransferDirection.RESTORE
    assert restore_metrics.page_count == 2
    assert restore_metrics.bytes_transferred == 2 * tier.page_nbytes
    assert tier.resident_pages == 0
    assert tier.free_pages == 2
    assert torch.equal(pool.k[:, 2], expected[0][0])
    assert torch.equal(pool.v[:, 2], expected[0][1])
    assert torch.equal(pool.k[:, 3], expected[1][0])
    assert torch.equal(pool.v[:, 3], expected[1][1])


def test_cancel_quarantines_capacity_and_completion_cannot_reuse_it() -> None:
    pool = _pool(num_pages=2)
    _fill(pool, 0, 1)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=1, backend=backend)
    key = _key("cancel")

    transfer = tier.begin_offload((key,), (0,))
    transfer.cancel()

    assert backend.completions[0].cancel_called
    assert tier.quarantined_pages == 1
    assert tier.quarantined_keys == (key,)
    assert tier.quarantined_device_pages == (0,)
    assert tier.free_pages == 0
    assert tier.state(key) is None
    backend.completions[0].complete()
    with pytest.raises(KVPageTransferError) as cancelled:
        transfer.finalize()
    assert not cancelled.value.source_reusable
    assert not cancelled.value.destination_reusable
    assert not cancelled.value.destination_publishable
    submissions = backend.submission_count
    with pytest.raises(KVPageStateError) as retained:
        tier.begin_offload((_key("different-key-same-device-page"),), (0,))
    assert not retained.value.source_reusable
    assert retained.value.destination_reusable
    assert backend.submission_count == submissions
    with pytest.raises(KVPageCapacityError):
        tier.begin_offload((_key("replacement"),), (1,))


def test_cancel_backend_fault_is_typed_and_still_quarantines() -> None:
    pool = _pool(num_pages=1)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=1, backend=backend)
    transfer = tier.begin_offload((_key("cancel-fault"),), (0,))
    backend.completions[0].cancel_error = RuntimeError("cancel backend fault")

    with pytest.raises(KVPageTransferError) as failed:
        transfer.cancel()

    assert not failed.value.source_reusable
    assert not failed.value.destination_reusable
    assert tier.quarantined_pages == 1
    assert tier.quarantined_device_pages == (0,)
    assert tier.free_pages == 0


@pytest.mark.parametrize("failure_site", ["query", "synchronize", "submit"])
def test_transfer_faults_fail_closed_with_explicit_ownership(failure_site: str) -> None:
    pool = _pool(num_pages=2)
    _fill(pool, 0, 1)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=1, backend=backend)

    if failure_site == "submit":
        backend.fail_submit = RuntimeError("submission fault")
        with pytest.raises(KVPageTransferError) as failed:
            tier.begin_offload((_key("fault"),), (0,))
    else:
        transfer = tier.begin_offload((_key("fault"),), (0,))
        if failure_site == "query":
            backend.completions[0].query_error = RuntimeError("query fault")
            operation = transfer.ready
        else:
            backend.completions[0].sync_error = RuntimeError("sync fault")
            operation = transfer.wait
        with pytest.raises(KVPageTransferError) as failed:
            operation()

    assert not failed.value.source_reusable
    assert not failed.value.destination_reusable
    assert not failed.value.destination_publishable
    assert tier.quarantined_pages == 1
    assert tier.quarantined_device_pages == (0,)
    assert tier.free_pages == 0

    backend.fail_submit = None
    submissions = backend.submission_count
    with pytest.raises(KVPageStateError) as retained:
        tier.begin_offload((_key(f"different-key-after-{failure_site}"),), (0,))
    assert not retained.value.source_reusable
    assert retained.value.destination_reusable
    assert backend.submission_count == submissions


@pytest.mark.parametrize("direction", ["offload", "restore"])
def test_fragment_plane_partial_submission_fault_quarantines_ownership(
    direction: str,
) -> None:
    pool = _pool(num_pages=3)
    _fill(pool, 0, 5)
    backend = _PartialFragmentBackend()
    tier = _tier(pool, capacity=1, backend=backend)
    key = _key(f"fragment-partial-{direction}")

    if direction == "offload":
        assert tier._host_fragment_planes is not None
        tier._host_pages.zero_()
        expected_fragment = pool.k[0, 0].detach().view(torch.uint8).reshape(-1).clone()
        backend.fail_offload = True
        retained_page = 0
        with pytest.raises(KVPageTransferError) as failed:
            tier.begin_offload((key,), (retained_page,))
        assert torch.equal(tier._host_fragment_planes[0][0], expected_fragment)
    else:
        assert tier.offload((key,), (0,))
        pool.k[:, 1].zero_()
        pool.v[:, 1].zero_()
        backend.fail_restore = True
        retained_page = 1
        with pytest.raises(KVPageTransferError) as failed:
            tier.begin_restore((key,), (retained_page,))
        assert torch.equal(pool.k[0, 1], pool.k[0, 0])

    assert not failed.value.source_reusable
    assert not failed.value.destination_reusable
    assert not failed.value.destination_publishable
    assert tier.quarantined_pages == 1
    assert tier.quarantined_keys == (key,)
    assert tier.quarantined_device_pages == (retained_page,)
    assert tier.free_pages == 0
    with pytest.raises(KVPageStateError):
        tier.begin_offload(
            (_key(f"fragment-partial-reuse-{direction}"),),
            (retained_page,),
        )


def test_known_complete_finalization_failure_releases_device_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _pool(num_pages=2)
    _fill(pool, 0, 8)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=2, backend=backend)
    transfer = tier.begin_offload((_key("finalize-fault"),), (0,))
    backend.completions[0].complete()

    def raise_commit_fault(_page: torch.Tensor) -> str:
        raise RuntimeError("commit fault")

    with monkeypatch.context() as patch:
        patch.setattr(kv_tier_module, "_checksum", raise_commit_fault)
        with pytest.raises(KVPageTransferError) as failed:
            transfer.finalize()

    assert failed.value.source_reusable
    assert not failed.value.destination_reusable
    assert tier.quarantined_device_pages == ()
    replacement = tier.begin_offload((_key("after-finalize-fault"),), (0,))
    assert backend.submission_count == 2
    replacement.cancel()


def test_unknown_offload_completion_stays_unsafe_across_same_key_retry() -> None:
    pool = _pool(num_pages=1)
    _fill(pool, 0, 1)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=1, backend=backend)
    key = _key("unknown-offload-retry")

    transfer = tier.begin_offload((key,), (0,))
    backend.completions[0].query_error = RuntimeError("lost D2H event")
    with pytest.raises(KVPageTransferError) as first:
        transfer.ready()
    assert first.value.source_reusable is False

    with pytest.raises(KVPageStateError) as retry:
        tier.begin_offload((key,), (0,))
    assert retry.value.source_reusable is False
    assert retry.value.destination_reusable is False
    assert tier.quarantined_keys == (key,)
    assert tier.free_pages == 0
    assert len(backend.completions) == 1


def test_checksum_mismatch_is_detected_before_h2d_and_destination_is_reusable() -> None:
    pool = _pool(num_pages=2)
    _fill(pool, 0, 4)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=1, backend=backend)
    key = _key("checksum")
    assert tier.offload((key,), (0,))
    assert backend.storage is not None
    backend.storage[0, 0] ^= 1
    destination_before = _snapshot(pool, 1)

    with pytest.raises(KVPageChecksumError) as corrupt:
        tier.begin_restore((key,), (1,))

    assert not corrupt.value.source_reusable
    assert corrupt.value.destination_reusable
    assert not corrupt.value.destination_publishable
    assert tier.quarantined_pages == 1
    assert tier.state(key) is None
    assert torch.equal(pool.k[:, 1], destination_before[0])
    assert torch.equal(pool.v[:, 1], destination_before[1])
    # No restore was submitted after the pre-copy checksum rejected the page.
    assert len(backend.completions) == 1


def test_same_key_retry_touches_or_atomically_replaces_committed_page() -> None:
    pool = _pool(num_pages=2)
    _fill(pool, 0, 5)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=3, backend=backend)
    key = _key("retry")
    assert tier.offload((key,), (0,))
    original = tier.resident_metadata(key)

    same = tier.begin_offload((key,), (0,))
    assert tier.available_prefix((key,)) == 1
    assert tier.resident_metadata(key).checksum == original.checksum
    same.wait()
    same.finalize()
    touched = tier.resident_metadata(key)
    assert touched.checksum == original.checksum
    assert touched.last_touch > original.last_touch
    assert tier.resident_pages == 1
    assert tier.free_pages == 2

    _fill(pool, 0, 9)
    replacement = tier.begin_offload((key,), (0,))
    assert tier.resident_metadata(key).checksum == original.checksum
    replacement.wait()
    replacement.finalize()
    replaced = tier.resident_metadata(key)
    assert replaced.checksum != original.checksum
    assert tier.resident_pages == 1
    assert tier.free_pages == 2


def test_unknown_same_key_replacement_keeps_old_resident_but_blocks_retry() -> None:
    pool = _pool(num_pages=2)
    _fill(pool, 0, 6)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=3, backend=backend)
    key = _key("replacement-fault")
    assert tier.offload((key,), (0,))
    committed = tier.resident_metadata(key)
    _fill(pool, 0, 8)

    failed = tier.begin_offload((key,), (0,))
    backend.completions[-1].query_error = RuntimeError("event lost")
    with pytest.raises(KVPageTransferError):
        failed.ready()

    assert tier.available_prefix((key,)) == 1
    assert tier.resident_metadata(key).checksum == committed.checksum
    assert key in tier.quarantined_keys
    assert tier.quarantined_pages == 1
    # The older host copy is still readable, but the lost D2H event may still
    # own the HBM source. A later successful copy must not launder that source
    # into reusable ownership.
    with pytest.raises(KVPageStateError) as retry:
        tier.begin_offload((key,), (0,))
    assert retry.value.source_reusable is False
    assert len(backend.completions) == 2


def test_bounded_reservation_evicts_only_untouched_resident_lru() -> None:
    pool = _pool(num_pages=3)
    for page_id in range(3):
        _fill(pool, page_id, page_id + 1)
    tier = _tier(pool, capacity=2)
    first, second, third = _key("lru-first"), _key("lru-second"), _key("lru-third")
    assert tier.offload((first, second), (0, 1))

    before = tier.resident_metadata(first).last_touch
    assert tier.available_prefix((first,)) == 1
    assert tier.resident_metadata(first).last_touch > before
    assert tier.offload((third,), (2,))

    assert tier.state(first) is PageState.RESIDENT
    assert tier.state(second) is None
    assert tier.state(third) is PageState.RESIDENT
    assert tier.free_pages == 0


def test_same_key_old_resident_is_protected_when_no_replacement_slot_exists() -> None:
    pool = _pool(num_pages=2)
    _fill(pool, 0, 2)
    tier = _tier(pool, capacity=1)
    key = _key("protected-current")
    assert tier.offload((key,), (0,))
    committed = tier.resident_metadata(key)

    _fill(pool, 0, 7)
    with pytest.raises(KVPageCapacityError) as full:
        tier.begin_offload((key,), (0,))

    assert full.value.source_reusable
    assert tier.state(key) is PageState.RESIDENT
    assert tier.resident_metadata(key) == committed
    assert tier.quarantined_pages == 0


def test_capacity_failure_does_not_partially_evict_when_safe_lru_is_insufficient() -> None:
    pool = _pool(num_pages=4)
    _fill(pool, 0, 1)
    _fill(pool, 1, 2)
    tier = _tier(pool, capacity=2)
    first, second = _key("atomic-first"), _key("atomic-second")
    assert tier.offload((first, second), (0, 1))

    restoring = tier.begin_restore((first,), (2,))
    with pytest.raises(KVPageCapacityError):
        tier.begin_offload((_key("new-a"), _key("new-b")), (0, 3))

    # second is a potential LRU victim, but one page cannot satisfy the batch;
    # it must remain published when the reservation fails atomically.
    assert tier.state(first) is PageState.RESTORING
    assert tier.state(second) is PageState.RESIDENT
    restoring.cancel()


def test_restore_completion_fault_never_publishes_or_reuses_destination() -> None:
    pool = _pool(num_pages=2)
    _fill(pool, 0, 3)
    backend = _ManualBackend()
    tier = _tier(pool, capacity=1, backend=backend)
    key = _key("restore-fault")
    assert tier.offload((key,), (0,))
    pool.k[:, 1].zero_()
    pool.v[:, 1].zero_()

    transfer = tier.begin_restore((key,), (1,))
    backend.completions[-1].sync_error = RuntimeError("lost H2D event")
    with pytest.raises(KVPageTransferError) as failed:
        transfer.wait()

    assert not failed.value.destination_reusable
    assert not failed.value.destination_publishable
    assert tier.state(key) is None
    assert tier.quarantined_pages == 1
    assert tier.quarantined_device_pages == (1,)
    submissions = backend.submission_count
    with pytest.raises(KVPageStateError) as retained:
        tier.begin_offload((_key("restore-fault-different-key"),), (1,))
    assert not retained.value.source_reusable
    assert retained.value.destination_reusable
    assert backend.submission_count == submissions


def test_discard_is_atomic_idempotent_and_releases_only_resident_pages() -> None:
    pool = _pool(num_pages=3)
    _fill(pool, 0, 1)
    _fill(pool, 1, 2)
    tier = _tier(pool, capacity=2)
    first, second, missing = _key("first"), _key("second"), _key("missing")
    assert tier.offload((first, second), (0, 1))

    assert tier.discard((first, missing))
    assert tier.state(first) is None
    assert tier.state(second) is PageState.RESIDENT
    assert tier.free_pages == 1
    assert tier.discard((first, missing))
    assert tier.free_pages == 1

    assert tier.offload((_key("reused"),), (2,))
    assert tier.free_pages == 0


def test_invalid_keys_and_batches_fail_before_any_transfer() -> None:
    pool = _pool(num_pages=2)
    backend = _ManualBackend()
    tier = _tier(pool, backend=backend)

    for keys, pages in [(("short",), (0,)), ((_key("a"), _key("a")), (0, 1))]:
        with pytest.raises(KVPageIdentityError) as invalid:
            tier.begin_offload(keys, pages)
        assert invalid.value.source_reusable
    with pytest.raises(KVPageIdentityError):
        tier.begin_offload((_key("a"),), (0, 1))
    with pytest.raises(KVPageIdentityError):
        tier.begin_offload((_key("a"),), (9,))
    assert backend.completions == []
