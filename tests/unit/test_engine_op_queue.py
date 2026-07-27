"""EngineLoop producer batching, coalescing and failure safety (issue #218)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from kairyu import SamplingParams
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import EngineRequest, Scheduler
from kairyu.engine.engine_loop import (
    EngineLoop,
    StreamUpdate,
    _AbortBatch,
    _AddBatch,
)


class _Tokenizer:
    eos_token_id = None

    def encode(self, text: str) -> tuple[int, ...]:
        return (len(text) % 31 + 1,)

    def decode(self, token_ids) -> str:
        return "".join(chr(ord("a") + token_id % 26) for token_id in token_ids)

    def vocab(self) -> list[str]:
        return [chr(ord("a") + index % 26) for index in range(2048)]


class _Runner:
    def __init__(self) -> None:
        self.released: list[str] = []

    def execute(self, scheduled, states):
        return {
            chunk.request_id: (SampledToken(1000 + chunk.position),)
            for chunk in scheduled
            if not chunk.is_prefill or states[chunk.request_id].prefill_done
        }

    def release(self, request_id: str) -> None:
        self.released.append(request_id)


def _scheduler() -> Scheduler:
    return Scheduler(
        RadixKVCache(num_pages=1024, page_size=4),
        max_num_batched_tokens=256,
        max_num_seqs=256,
        page_size=4,
    )


def _loop(scheduler: object | None = None) -> tuple[EngineLoop, _Runner]:
    runner = _Runner()
    return EngineLoop(_Tokenizer(), scheduler or _scheduler(), runner), runner


def _drive(loop: EngineLoop, limit: int = 1000) -> list[tuple[str, StreamUpdate]]:
    updates: list[tuple[str, StreamUpdate]] = []
    for _ in range(limit):
        if not loop.has_work():
            return updates
        updates.extend(loop.step())
    raise AssertionError("engine loop did not drain")


def test_add_abort_add_batches_preserve_order_and_one_terminal() -> None:
    class _RecordingScheduler(Scheduler):
        def __init__(self):
            super().__init__(
                RadixKVCache(num_pages=128, page_size=4),
                max_num_batched_tokens=16,
                page_size=4,
            )
            self.bulk_sizes: list[int] = []

        def add_requests_atomic(self, requests):
            self.bulk_sizes.append(len(requests))
            super().add_requests_atomic(requests)

    scheduler = _RecordingScheduler()
    loop, runner = _loop(scheduler)
    params = SamplingParams(max_tokens=2)
    loop.submit("a", "first", params)
    loop.submit("b", "second", params)
    for _ in range(100):
        loop.abort("a")
    loop.submit("c", "third", params)

    with loop._ops_lock:
        assert [type(batch) for batch in loop._ops] == [
            _AddBatch,
            _AbortBatch,
            _AddBatch,
        ]
        assert loop._ops[1].request_ids == ["a"]

    updates = loop.step()
    updates.extend(_drive(loop))
    finals = {
        request_id: update for request_id, update in updates if update.finished
    }

    assert scheduler.bulk_sizes == [2]
    assert finals["a"].finish_reason == "abort"
    assert finals["b"].finish_reason == "length"
    assert finals["c"].finish_reason == "length"
    terminal_a = [
        request_id
        for request_id, update in updates
        if request_id == "a" and update.finished
    ]
    assert terminal_a == ["a"]
    assert sorted(runner.released) == ["a", "b", "c"]
    loop.close()


def test_concurrent_duplicate_submit_reserves_exactly_once() -> None:
    loop, _runner = _loop()
    workers = 32
    barrier = Barrier(workers)

    def submit() -> str:
        barrier.wait()
        try:
            loop.submit("same", "prompt", SamplingParams(max_tokens=1))
        except ValueError:
            return "duplicate"
        return "accepted"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(lambda _index: submit(), range(workers)))

    assert outcomes.count("accepted") == 1
    assert outcomes.count("duplicate") == workers - 1
    with loop._ops_lock:
        assert len(loop._ops) == 1
        assert isinstance(loop._ops[0], _AddBatch)
        assert len(loop._ops[0].requests) == 1
    loop.purge(("same",))
    loop.close()


def test_concurrent_unique_producers_share_one_add_batch() -> None:
    loop, _runner = _loop()
    workers = 8
    per_worker = 500
    barrier = Barrier(workers)

    def submit_partition(worker: int) -> None:
        barrier.wait()
        for index in range(per_worker):
            request_id = f"worker-{worker}-{index}"
            loop.submit(request_id, "p", SamplingParams(max_tokens=1))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(submit_partition, range(workers)))

    with loop._ops_lock:
        assert len(loop._ops) == 1
        assert isinstance(loop._ops[0], _AddBatch)
        assert len(loop._ops[0].requests) == workers * per_worker

    request_ids = tuple(loop._active_request_ids)
    loop.purge(request_ids)
    loop.close()


def test_concurrent_repeated_aborts_enqueue_and_finish_once() -> None:
    loop, _runner = _loop()
    workers = 16
    per_worker = 500
    barrier = Barrier(workers)
    loop.submit("cancel", "p", SamplingParams(max_tokens=8))

    def abort_repeatedly(_worker: int) -> None:
        barrier.wait()
        for _ in range(per_worker):
            loop.abort("cancel")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(abort_repeatedly, range(workers)))

    with loop._ops_lock:
        assert len(loop._ops) == 2
        assert isinstance(loop._ops[1], _AbortBatch)
        assert loop._ops[1].request_ids == ["cancel"]

    updates = _drive(loop)
    terminals = [
        update
        for request_id, update in updates
        if request_id == "cancel" and update.finished
    ]
    assert len(terminals) == 1
    assert terminals[0].finish_reason == "abort"
    loop.close()


def test_ten_thousand_bursty_operations_coalesce_to_two_batches() -> None:
    loop, _runner = _loop()
    params = SamplingParams(max_tokens=1)
    request_ids = tuple(f"burst-{index}" for index in range(10_000))

    for request_id in request_ids:
        loop.submit(request_id, "p", params)
    for request_id in request_ids[::2]:
        loop.abort(request_id)
        loop.abort(request_id)

    with loop._ops_lock:
        assert len(loop._ops) == 2
        assert isinstance(loop._ops[0], _AddBatch)
        assert len(loop._ops[0].requests) == 10_000
        assert isinstance(loop._ops[1], _AbortBatch)
        assert len(loop._ops[1].request_ids) == 5_000

    loop.purge(request_ids)
    assert not loop.has_work()
    loop.close()


def test_scheduler_bulk_add_is_atomic_on_duplicate() -> None:
    scheduler = _scheduler()
    scheduler.add_request(EngineRequest("existing", (1,)))

    with pytest.raises(ValueError, match="duplicate request_id"):
        scheduler.add_requests_atomic(
            (
                EngineRequest("before", (2,)),
                EngineRequest("existing", (3,)),
                EngineRequest("after", (4,)),
            )
        )

    assert scheduler.waiting_ids == ("existing",)
    assert set(scheduler.states) == {"existing"}


def test_partial_add_failure_restores_unprocessed_batch_in_order() -> None:
    class _FailingAdapter:
        add_requests_atomic = None

        def __init__(self, scheduler: Scheduler) -> None:
            self._scheduler = scheduler
            self.failed = False

        def add_request(self, request: EngineRequest) -> None:
            if request.request_id == "bad" and not self.failed:
                self.failed = True
                raise RuntimeError("injected add failure")
            self._scheduler.add_request(request)

        def __getattr__(self, name):
            return getattr(self._scheduler, name)

    scheduler = _scheduler()
    adapter = _FailingAdapter(scheduler)
    loop, _runner = _loop(adapter)
    params = SamplingParams(max_tokens=1)
    loop.submit("before", "p", params)
    loop.submit("bad", "p", params)
    loop.submit("after", "p", params)

    with pytest.raises(RuntimeError, match="injected add failure"):
        loop.step()

    with loop._ops_lock:
        assert len(loop._ops) == 1
        assert isinstance(loop._ops[0], _AddBatch)
        assert [request.request_id for request in loop._ops[0].requests] == ["after"]

    loop.purge(("before", "bad"))
    updates = _drive(loop)
    final = next(
        update
        for request_id, update in updates
        if request_id == "after" and update.finished
    )
    assert final.finish_reason == "length"
    loop.close()


def test_close_rejects_new_submissions_and_drops_queued_ops() -> None:
    loop, _runner = _loop()
    loop.submit("queued", "p", SamplingParams(max_tokens=1))

    loop.close()

    assert not loop.has_work()
    with pytest.raises(RuntimeError, match="closed"):
        loop.submit("late", "p", SamplingParams(max_tokens=1))
