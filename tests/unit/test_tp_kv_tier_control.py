"""All-rank ownership tests for the TP DRAM KV tier coordinator."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from kairyu.engine.core.comm import FakeCommunicator
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.engine.core.step_input import RequestSnapshot
from kairyu.engine.core.worker import (
    DistTPModelRunner,
    DramKVTierTransactionError,
    worker_step_loop,
)


class _TransferError(RuntimeError):
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


class _FakeTier:
    def __init__(
        self,
        *,
        available: object = 2,
        offload: object = True,
        restore: object = True,
        discard: object = True,
    ) -> None:
        self.available_result = available
        self.offload_result = offload
        self.restore_result = restore
        self.discard_result = discard
        self.calls: list[tuple[object, ...]] = []
        self.offload_entered = Event()

    @staticmethod
    def _resolve(result, *args):
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            return result(*args)
        return result

    def available_prefix(self, keys, *, min_pages=1):
        self.calls.append(("available_prefix", keys, min_pages))
        return self._resolve(self.available_result, keys, min_pages)

    def offload(self, keys, page_ids):
        self.calls.append(("offload", keys, page_ids))
        self.offload_entered.set()
        return self._resolve(self.offload_result, keys, page_ids)

    def restore(self, keys, page_ids):
        self.calls.append(("restore", keys, page_ids))
        return self._resolve(self.restore_result, keys, page_ids)

    def discard(self, keys):
        self.calls.append(("discard", keys))
        return self._resolve(self.discard_result, keys)


class _TPHarness:
    def __init__(self, tiers: tuple[_FakeTier, ...]) -> None:
        self.control = FakeCommunicator.create_group(len(tiers))
        self.runners = tuple(SimpleNamespace(dram_kv_tier=tier) for tier in tiers)
        self.driver = DistTPModelRunner(self.control[0], self.runners[0])
        self.pool = ThreadPoolExecutor(max_workers=len(tiers) - 1)
        self.workers = [
            self.pool.submit(worker_step_loop, self.control[rank], self.runners[rank])
            for rank in range(1, len(tiers))
        ]

    def close(self) -> None:
        self.driver.shutdown()
        assert [worker.result(timeout=2) for worker in self.workers] == [
            0
        ] * len(self.workers)
        self.pool.shutdown()


def _discard_count(tier: _FakeTier) -> int:
    return sum(call[0] == "discard" for call in tier.calls)


def test_tp_kv_tier_all_rank_success():
    keys = ("prefix-1", "prefix-2")
    page_ids = (3, 7)
    tiers = (_FakeTier(), _FakeTier(), _FakeTier())
    harness = _TPHarness(tiers)
    try:
        assert harness.driver.available_prefix(keys, min_pages=2) == 2
        assert harness.driver.offload(keys, page_ids) is True
        assert harness.driver.restore(keys, page_ids) is True
        assert harness.driver.discard(keys) is True
    finally:
        harness.close()

    expected = [
        ("available_prefix", keys, 2),
        ("offload", keys, page_ids),
        ("restore", keys, page_ids),
        ("discard", keys),
    ]
    assert all(tier.calls == expected for tier in tiers)


def test_tp_kv_tier_rank_mismatch_discards_every_rank_and_falls_back():
    keys = ("prefix-1", "prefix-2")
    page_ids = (3, 7)
    tiers = (
        _FakeTier(available=2, offload=True),
        _FakeTier(available=1, offload=False),
    )
    harness = _TPHarness(tiers)
    try:
        assert harness.driver.available_prefix(keys, min_pages=1) == 0
        assert harness.driver.offload(keys, page_ids) is False
    finally:
        harness.close()

    assert [_discard_count(tier) for tier in tiers] == [2, 2]


@pytest.mark.parametrize("operation", ["offload", "restore"])
def test_tp_kv_tier_safe_rank_exception_discards_every_rank(operation):
    keys = ("prefix",)
    page_ids = (4,)
    error = _TransferError(
        "safe failure",
        source_reusable=True,
        destination_reusable=True,
    )
    kwargs = {operation: error}
    tiers = (_FakeTier(), _FakeTier(**kwargs))
    harness = _TPHarness(tiers)
    try:
        method = getattr(harness.driver, operation)
        assert method(keys, page_ids) is False
    finally:
        harness.close()

    assert [_discard_count(tier) for tier in tiers] == [1, 1]


@pytest.mark.parametrize("operation", ["offload", "restore"])
def test_tp_kv_tier_unknown_page_ownership_raises_aggregated_error(operation):
    keys = ("prefix",)
    page_ids = (4,)
    kwargs = {operation: _TransferError("ownership unknown")}
    tiers = (_FakeTier(), _FakeTier(**kwargs))
    harness = _TPHarness(tiers)
    try:
        method = getattr(harness.driver, operation)
        with pytest.raises(DramKVTierTransactionError, match="rank 1") as raised:
            method(keys, page_ids)
    finally:
        harness.close()

    assert raised.value.source_reusable is False
    assert raised.value.destination_reusable is False
    assert raised.value.destination_publishable is False
    assert [_discard_count(tier) for tier in tiers] == [0, 0]


@pytest.mark.parametrize(
    ("operation", "failure"),
    [
        ("available_prefix", KeyboardInterrupt()),
        ("offload", SystemExit("stop")),
        ("restore", KeyboardInterrupt()),
        ("discard", SystemExit("stop")),
    ],
)
def test_tp_kv_tier_base_exception_completes_collective_and_fails_closed(
    operation,
    failure,
):
    keys = ("prefix",)
    page_ids = (4,)
    kwargs = {"available" if operation == "available_prefix" else operation: failure}
    tiers = (_FakeTier(), _FakeTier(**kwargs))
    harness = _TPHarness(tiers)
    try:
        method = getattr(harness.driver, operation)
        with pytest.raises(DramKVTierTransactionError, match="rank 1"):
            if operation == "available_prefix":
                method(keys, min_pages=1)
            elif operation == "discard":
                method(keys)
            else:
                method(keys, page_ids)

        # The worker reached the ACK gather and returned to the next control
        # broadcast; a subsequent transaction therefore still completes.
        tiers[1].discard_result = True
        assert harness.driver.discard(keys) is True
        assert harness.driver.fatal_error is None
    finally:
        harness.close()


def test_tp_restore_cannot_publish_before_every_rank_finishes():
    entered = Event()
    release = Event()

    def blocking_restore(keys, page_ids):  # noqa: ARG001 - synchronization probe
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test did not release the blocking restore")
        return True

    tiers = (_FakeTier(), _FakeTier(restore=blocking_restore))
    harness = _TPHarness(tiers)
    try:
        with ThreadPoolExecutor(max_workers=1) as calls:
            restored = calls.submit(harness.driver.restore, ("prefix",), (8,))
            assert entered.wait(timeout=2)
            assert restored.done() is False
            release.set()
            assert restored.result(timeout=2) is True
    finally:
        release.set()
        harness.close()


class _BlockingStepRunner:
    def __init__(self, tier: _FakeTier, entered: Event | None, release: Event | None) -> None:
        self.dram_kv_tier = tier
        self.entered = entered
        self.release_step = release

    def execute(self, scheduled, states):  # noqa: ARG002 - protocol-only fake
        assert self.entered is not None
        assert self.release_step is not None
        self.entered.set()
        if not self.release_step.wait(timeout=5):
            raise RuntimeError("test did not release the blocking model step")
        return {scheduled[0].request_id: (17,)}

    def execute_passive(self, scheduled, states):  # noqa: ARG002
        return {}

    def make_sampling_token_packet(self, scheduled, states, sampled=None):  # noqa: ARG002
        value = 17 if sampled is not None else -1
        return torch.full((len(scheduled),), value, dtype=torch.int64)

    def adopt_sampling_token_packet(self, scheduled, states, packet):  # noqa: ARG002
        return None


def _snapshot() -> RequestSnapshot:
    return RequestSnapshot(
        request_id="request",
        prompt_token_ids=(1,),
        computed_prompt=1,
        outputs=(),
        in_flight=0,
        page_ids=(0,),
        decode_page_ids=(0,),
        eos_token_id=None,
        max_new_tokens=2,
    )


def test_tp_kv_tier_transaction_does_not_intersect_a_model_step():
    step_entered = Event()
    release_step = Event()
    tiers = (_FakeTier(), _FakeTier())
    control = FakeCommunicator.create_group(2)
    runners = (
        _BlockingStepRunner(tiers[0], step_entered, release_step),
        _BlockingStepRunner(tiers[1], None, None),
    )
    driver = DistTPModelRunner(control[0], runners[0])
    chunk = ScheduledChunk("request", 1, True, 0)

    with ThreadPoolExecutor(max_workers=3) as pool:
        worker = pool.submit(worker_step_loop, control[1], runners[1])
        step = pool.submit(driver.execute, (chunk,), {"request": _snapshot()})
        assert step_entered.wait(timeout=2)
        offload = pool.submit(driver.offload, ("prefix",), (4,))
        try:
            assert tiers[0].offload_entered.wait(timeout=0.1) is False
        finally:
            release_step.set()
        assert step.result(timeout=2) == {"request": (17,)}
        assert offload.result(timeout=2) is True
        driver.shutdown()
        assert worker.result(timeout=2) == 1
