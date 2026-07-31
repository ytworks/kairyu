"""Real pinned-memory and CUDA-event contract for the DRAM KV tier."""

from __future__ import annotations

import hashlib
import os

import pytest
import torch

from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.engine.core.kv_tier import DRAMKVTier, KVPageStateError, PageState
from kairyu.engine.core.numa import GPU_LOCAL_FIRST_TOUCH_POLICY

pytestmark = pytest.mark.gpu


def _key(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def test_cuda_d2h_h2d_uses_pinned_slab_and_physical_completion_events() -> None:
    pool = PagedKVPool(
        num_layers=2,
        num_pages=2,
        page_size=16,
        num_kv_heads=2,
        head_dim=32,
        dtype=torch.float16,
        device="cuda:0",
    )
    torch.manual_seed(187)
    pool.k[:, 0].normal_()
    pool.v[:, 0].normal_()
    expected_k = pool.k[:, 0].clone()
    expected_v = pool.v[:, 0].clone()
    caller_affinity = tuple(sorted(os.sched_getaffinity(0)))
    tier = DRAMKVTier(
        pool,
        model_id="gpu-test/model@0123456789abcdef",
        tp_size=1,
        tp_rank=0,
        capacity_pages=1,
    )
    key = _key("cuda-page")

    assert tier.host_is_pinned
    assert tier.host_numa_policy == GPU_LOCAL_FIRST_TOUCH_POLICY
    assert tier.host_numa_node is not None
    assert tier.host_cpu_affinity
    assert tuple(sorted(os.sched_getaffinity(0))) == caller_affinity
    assert set(tier.host_numa_page_counts) == {tier.host_numa_node}
    assert sum(tier.host_numa_page_counts.values()) * os.sysconf(
        "SC_PAGE_SIZE"
    ) >= tier.host_nbytes
    # The copy stream waits for this producer tail.  begin_offload must return
    # an event instead of synchronizing it on the host.
    torch.cuda._sleep(300_000_000)
    offload = tier.begin_offload((key,), (0,))
    assert isinstance(offload.event, torch.cuda.Event)
    assert isinstance(offload.stream_id, int)
    assert offload.cuda_elapsed_ns is None
    assert tier.state(key) is PageState.OFFLOADING
    assert not offload.ready()
    with pytest.raises(KVPageStateError):
        tier.resident_metadata(key)
    offload.wait()
    assert tier.state(key) is PageState.OFFLOADING
    assert offload.finalize()
    assert tuple(sorted(os.sched_getaffinity(0))) == caller_affinity
    assert offload.cuda_elapsed_ns is not None
    assert offload.cuda_elapsed_ns > 0
    assert tier.state(key) is PageState.RESIDENT
    assert tier.host_data_ptr > 0
    assert tier.host_nbytes == tier.page_nbytes

    pool.k[:, 1].zero_()
    pool.v[:, 1].zero_()
    restore = tier.begin_restore((key,), (1,))
    assert tuple(sorted(os.sched_getaffinity(0))) == caller_affinity
    assert isinstance(restore.event, torch.cuda.Event)
    assert tier.state(key) is PageState.RESTORING
    assert not restore.destination_publishable
    restore.wait()
    assert restore.destination_reusable
    assert not restore.destination_publishable
    assert restore.finalize()
    assert tuple(sorted(os.sched_getaffinity(0))) == caller_affinity
    assert restore.cuda_elapsed_ns is not None
    assert restore.cuda_elapsed_ns > 0

    assert restore.destination_publishable
    assert tier.state(key) is None
    assert torch.equal(pool.k[:, 1], expected_k)
    assert torch.equal(pool.v[:, 1], expected_v)
