"""P-D disaggregation: resume_with_kv + PDCoordinator (design m5 D5).

The CPU half pins the protocol ordering (copy-before-commit, no re-sample of
token 0, preemption shield) and greedy-equivalence vs a single combined core;
the GPU phase swaps LocalKVHandoff for a device copy behind the same seam.
"""

from __future__ import annotations

import pytest

from kairyu.engine.core.engine_core import EngineCore
from kairyu.engine.core.pd import KVHandoffError, LocalKVHandoff, PDCoordinator
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import EngineRequest, Scheduler

_VOCAB = 50_000


class _ToyRunner:
    """Deterministic runner matching kairyu_backend's toy forward."""

    def execute(self, scheduled, states):
        sampled = {}
        for chunk in scheduled:
            state = states[chunk.request_id]
            if not chunk.is_prefill or state.prefill_done:
                seed = sum(state.request.prompt_token_ids)
                sampled[chunk.request_id] = (SampledToken((seed + 31 * chunk.position) % _VOCAB),)
        return sampled


def _make_pair(num_pages: int = 64, budget: int = 32) -> tuple[Scheduler, RadixKVCache]:
    kv = RadixKVCache(num_pages=num_pages, page_size=4)
    return Scheduler(kv, max_num_batched_tokens=budget, page_size=4), kv


def _make_coordinator(
    *,
    prefill_pages: int = 64,
    decode_pages: int = 64,
    handoff=None,
    max_transfer_retries: int = 1,
) -> tuple[PDCoordinator, RadixKVCache, RadixKVCache]:
    prefill_sched, prefill_kv = _make_pair(num_pages=prefill_pages)
    decode_sched, decode_kv = _make_pair(num_pages=decode_pages)
    coordinator = PDCoordinator(
        prefill_scheduler=prefill_sched,
        prefill_runner=_ToyRunner(),
        decode_scheduler=decode_sched,
        decode_runner=_ToyRunner(),
        handoff=handoff or LocalKVHandoff(decode_kv),
        max_transfer_retries=max_transfer_retries,
    )
    return coordinator, prefill_kv, decode_kv


def _single_core_reference(requests: list[EngineRequest]) -> dict[str, tuple[int, ...]]:
    scheduler, _ = _make_pair()
    core = EngineCore(scheduler, _ToyRunner())
    for request in requests:
        core.add_request(request)
    return core.run_to_completion()


# --- Scheduler.resume_with_kv -------------------------------------------------


def test_resume_with_kv_decodes_to_completion_without_resampling_token0() -> None:
    # Arrange: a decode core adopts prompt KV plus the already-sampled token 0
    scheduler, kv = _make_pair()
    request = EngineRequest("r1", prompt_token_ids=(1, 2, 3, 4, 5), max_new_tokens=4)
    allocation = kv.allocate(request.prompt_token_ids)
    kv.mark_computed(allocation)
    seed = sum(request.prompt_token_ids)
    token0 = seed % _VOCAB

    # Act
    finished = scheduler.resume_with_kv(request, allocation, first_token=token0)
    core = EngineCore(scheduler, _ToyRunner())
    outputs = core.run_to_completion()

    # Assert: token 0 was adopted, not re-sampled, and decode continued from position 1
    assert finished is False
    reference = _single_core_reference([request])
    assert scheduler.output_tokens("r1") == reference["r1"]
    assert outputs == {} or outputs["r1"] == reference["r1"]


def test_resume_with_kv_rejects_duplicate_and_mismatched_allocation() -> None:
    scheduler, kv = _make_pair()
    request = EngineRequest("r1", prompt_token_ids=(1, 2, 3, 4), max_new_tokens=2)
    allocation = kv.allocate(request.prompt_token_ids)
    scheduler.resume_with_kv(request, allocation, first_token=7)
    with pytest.raises(ValueError):
        scheduler.resume_with_kv(request, allocation, first_token=7)

    other_kv_alloc = kv.allocate((9, 9, 9, 9))
    other = EngineRequest("r2", prompt_token_ids=(1, 2, 3, 4), max_new_tokens=2)
    with pytest.raises(ValueError):
        scheduler.resume_with_kv(other, other_kv_alloc, first_token=7)


def test_resume_with_kv_finishes_immediately_at_max_or_eos() -> None:
    scheduler, kv = _make_pair()
    one = EngineRequest("one", prompt_token_ids=(1, 2, 3, 4), max_new_tokens=1)
    finished = scheduler.resume_with_kv(one, kv.allocate(one.prompt_token_ids), first_token=5)
    assert finished is True
    assert scheduler.output_tokens("one") == (5,)
    assert not scheduler.has_unfinished()

    eos = EngineRequest("eos", prompt_token_ids=(5, 6, 7, 8), max_new_tokens=8, eos_token_id=42)
    finished = scheduler.resume_with_kv(eos, kv.allocate(eos.prompt_token_ids), first_token=42)
    assert finished is True
    assert scheduler.output_tokens("eos") == (42,)
    assert scheduler.finish_reason("eos") == "stop"  # reason set like the normal path


def test_resume_with_kv_honors_ignore_eos_and_min_tokens() -> None:
    # The P-D adoption path must respect ignore_eos / min_tokens exactly like the
    # normal decode terminal check, not finish on a bare EOS match.
    scheduler, kv = _make_pair()
    ignored = EngineRequest(
        "ig", prompt_token_ids=(1, 2, 3, 4), max_new_tokens=8, eos_token_id=42, ignore_eos=True
    )
    finished = scheduler.resume_with_kv(
        ignored, kv.allocate(ignored.prompt_token_ids), first_token=42
    )
    assert finished is False  # ignore_eos -> the EOS-valued first token is kept

    held = EngineRequest(
        "mt", prompt_token_ids=(5, 6, 7, 8), max_new_tokens=8, eos_token_id=42, min_tokens=3
    )
    finished = scheduler.resume_with_kv(held, kv.allocate(held.prompt_token_ids), first_token=42)
    assert finished is False  # min_tokens=3 not yet met, so EOS does not terminate


def test_resumed_request_is_shielded_from_preemption() -> None:
    # Arrange: pool sized so the resumed request's decode growth collides with a
    # mid-prefill victim — preemption must pick the victim, never the resumed one
    kv = RadixKVCache(num_pages=6, page_size=4)
    scheduler = Scheduler(kv, max_num_batched_tokens=8, page_size=4)
    resumed = EngineRequest("resumed", prompt_token_ids=(1, 2, 3, 4), max_new_tokens=8)
    allocation = kv.allocate(resumed.prompt_token_ids)
    kv.mark_computed(allocation)
    scheduler.resume_with_kv(resumed, allocation, first_token=10)
    victim = EngineRequest("victim", prompt_token_ids=tuple(range(20, 37)), max_new_tokens=2)
    scheduler.add_request(victim)

    # Act: step until the resumed request finishes, tracking who gets requeued
    core = EngineCore(scheduler, _ToyRunner())
    victim_was_preempted = False
    for _ in range(40):
        if "resumed" not in scheduler.states or not scheduler.has_unfinished():
            break
        assert "resumed" not in scheduler.waiting_ids  # shield: never requeued
        if "victim" in scheduler.waiting_ids:
            victim_was_preempted = True
        core.step()

    # Assert
    assert victim_was_preempted
    assert len(scheduler.output_tokens("resumed")) == 8
    assert scheduler.output_tokens("resumed")[0] == 10


# --- PDCoordinator ------------------------------------------------------------


def test_pd_coordinator_matches_single_core_greedy() -> None:
    requests = [
        EngineRequest("a", prompt_token_ids=tuple(range(1, 6)), max_new_tokens=4),
        # long prompt: chunked prefill spans multiple steps
        EngineRequest("b", prompt_token_ids=tuple(range(10, 90)), max_new_tokens=3),
        EngineRequest("c", prompt_token_ids=(3, 1, 4, 1, 5), max_new_tokens=1),
    ]
    coordinator, _, _ = _make_coordinator()
    for request in requests:
        coordinator.add_request(request)

    outputs = coordinator.run_to_completion()

    assert outputs == _single_core_reference(requests)


def test_pd_handoff_reuses_decode_side_cached_prefix() -> None:
    shared = tuple(range(100, 140))
    first = EngineRequest("s1", prompt_token_ids=shared + (1,), max_new_tokens=2)
    second = EngineRequest("s2", prompt_token_ids=shared + (2,), max_new_tokens=2)
    coordinator, _, decode_kv = _make_coordinator()
    coordinator.add_request(first)
    coordinator.run_to_completion()
    hits_before = decode_kv.hit_rate

    coordinator.add_request(second)
    coordinator.run_to_completion()

    assert decode_kv.hit_rate > hits_before  # adopt path hit the shared prefix


def test_prefill_core_retains_prefix_for_cross_request_reuse() -> None:
    shared = tuple(range(200, 240))
    coordinator, prefill_kv, _ = _make_coordinator()
    coordinator.add_request(EngineRequest("p1", prompt_token_ids=shared + (1,), max_new_tokens=2))
    coordinator.run_to_completion()

    coordinator.add_request(EngineRequest("p2", prompt_token_ids=shared + (2,), max_new_tokens=2))
    coordinator.run_to_completion()

    assert prefill_kv.hit_rate > 0.0  # commit_and_release folded p1's prompt


class _FlakyHandoff:
    """Fails the first N transfers, then delegates to a real handoff."""

    def __init__(self, delegate: LocalKVHandoff, failures: int) -> None:
        self._delegate = delegate
        self._failures = failures
        self.attempts = 0

    def transfer(self, tokens, first_token, pages=()):
        self.attempts += 1
        if self.attempts <= self._failures:
            raise KVHandoffError("injected transfer failure")
        return self._delegate.transfer(tokens, first_token)

    def discard(self, allocation):
        self._delegate.discard(allocation)


def test_transfer_failure_requeues_and_retries_once() -> None:
    decode_sched, decode_kv = _make_pair()
    flaky = _FlakyHandoff(LocalKVHandoff(decode_kv), failures=1)
    prefill_sched, _ = _make_pair()
    coordinator = PDCoordinator(
        prefill_scheduler=prefill_sched,
        prefill_runner=_ToyRunner(),
        decode_scheduler=decode_sched,
        decode_runner=_ToyRunner(),
        handoff=flaky,
        max_transfer_retries=1,
    )
    request = EngineRequest("r", prompt_token_ids=(1, 2, 3, 4, 5, 6), max_new_tokens=3)
    coordinator.add_request(request)

    outputs = coordinator.run_to_completion()

    assert flaky.attempts == 2
    assert outputs == _single_core_reference([request])
    assert coordinator.failed_requests == ()


def test_transfer_failure_exhausts_retries_and_reports() -> None:
    decode_sched, decode_kv = _make_pair()
    flaky = _FlakyHandoff(LocalKVHandoff(decode_kv), failures=10)
    prefill_sched, _ = _make_pair()
    coordinator = PDCoordinator(
        prefill_scheduler=prefill_sched,
        prefill_runner=_ToyRunner(),
        decode_scheduler=decode_sched,
        decode_runner=_ToyRunner(),
        handoff=flaky,
        max_transfer_retries=1,
    )
    coordinator.add_request(EngineRequest("r", prompt_token_ids=(1, 2, 3, 4), max_new_tokens=2))

    outputs = coordinator.run_to_completion()

    assert outputs == {}
    assert coordinator.failed_requests == ("r",)


def test_raw_deferred_failure_is_normalized_and_retries_cleanly() -> None:
    from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff

    decode_sched, decode_kv = _make_pair()
    prefill_sched, _ = _make_pair()

    class _RawFailure:
        def __init__(self):
            self.delegate = LocalKVHandoff(decode_kv)
            self.attempts = 0
            self.allocations = []

        def transfer(self, tokens, first_token, pages=()):
            self.attempts += 1
            allocation = self.delegate.transfer(tokens, first_token)
            self.allocations.append(allocation)
            if self.attempts == 1:
                error = RuntimeError("raw deferred failure")
                error.allocation = allocation
                raise error
            return allocation

        def discard(self, allocation):
            self.delegate.discard(allocation)

    inner = _RawFailure()
    coordinator = PDCoordinator(
        prefill_scheduler=prefill_sched,
        prefill_runner=_ToyRunner(),
        decode_scheduler=decode_sched,
        decode_runner=_ToyRunner(),
        handoff=StreamCopyKVHandoff(inner, CpuNoopStream(), defer=True),
        max_transfer_retries=1,
    )
    request = EngineRequest(
        "r",
        prompt_token_ids=(1, 2, 3, 4, 5, 6),
        max_new_tokens=3,
    )
    coordinator.add_request(request)

    assert coordinator.run_to_completion() == _single_core_reference([request])
    assert inner.attempts == 2
    assert inner.allocations[0]._freed is True
    assert coordinator._handover is None


def test_retry_enqueue_is_reenterable_after_success_then_raise() -> None:
    coordinator, _ = _deferred_coordinator(failures=1, max_transfer_retries=1)
    request = EngineRequest(
        "r",
        prompt_token_ids=(1, 2, 3, 4, 5, 6),
        max_new_tokens=3,
    )
    coordinator.add_request(request)
    enqueue = coordinator._enqueue
    failed = False

    def enqueue_then_raise(candidate, attempt):
        nonlocal failed
        result = enqueue(candidate, attempt)
        if attempt == 1 and not failed:
            failed = True
            raise RuntimeError("injected post-enqueue failure")
        return result

    coordinator._enqueue = enqueue_then_raise
    coordinator.step_prefill()

    with pytest.raises(RuntimeError, match="post-enqueue failure"):
        coordinator.step_prefill()

    assert coordinator.internal_id_for("r") == "r#p1"
    assert "r#p1" in coordinator.prefill_scheduler.states
    assert coordinator.run_to_completion() == _single_core_reference([request])
    assert coordinator.failed_requests == ()
    assert coordinator._handover is None
    assert coordinator._handoff._settled == {}


def test_deferred_retry_reconciles_discard_success_then_raise() -> None:
    """Finalizer re-entry must replay the transfer error, not discard twice."""
    from kairyu.engine.core.handoff_stream import (
        CpuNoopStream,
        StreamCopyKVHandoff,
    )

    decode_sched, decode_kv = _make_pair()
    prefill_sched, _ = _make_pair()
    delegate = LocalKVHandoff(decode_kv)
    attempts = 0
    fail_after_release = True

    class _AllocatedFailure:
        def transfer(self, tokens, first_token, pages=()):
            nonlocal attempts
            attempts += 1
            allocation = delegate.transfer(tokens, first_token)
            if attempts == 1:
                raise KVHandoffError(
                    "injected transfer failure",
                    allocation=allocation,
                )
            return allocation

        def discard(self, allocation):
            nonlocal fail_after_release
            delegate.discard(allocation)
            if fail_after_release:
                fail_after_release = False
                raise RuntimeError("injected post-release cleanup failure")

    handoff = StreamCopyKVHandoff(
        _AllocatedFailure(),
        CpuNoopStream(),
        defer=True,
    )
    coordinator = PDCoordinator(
        prefill_scheduler=prefill_sched,
        prefill_runner=_ToyRunner(),
        decode_scheduler=decode_sched,
        decode_runner=_ToyRunner(),
        handoff=handoff,
        max_transfer_retries=1,
    )
    request = EngineRequest(
        "r",
        prompt_token_ids=(1, 2, 3, 4, 5, 6),
        max_new_tokens=3,
    )
    coordinator.add_request(request)
    coordinator.step_prefill()

    with pytest.raises(RuntimeError, match="post-release cleanup failure"):
        coordinator.step_prefill()

    assert coordinator.run_to_completion() == _single_core_reference([request])
    assert attempts == 2
    assert coordinator.failed_requests == ()
    assert handoff._settled == {}


def test_blocking_handoff_adoption_failure_discards_and_retries() -> None:
    decode_sched, decode_kv = _make_pair()
    prefill_sched, _ = _make_pair()
    handoff = LocalKVHandoff(decode_kv)
    allocations = []
    transfer = handoff.transfer

    def capture_transfer(*args, **kwargs):
        allocation = transfer(*args, **kwargs)
        allocations.append(allocation)
        return allocation

    handoff.transfer = capture_transfer
    resume = decode_sched.resume_with_kv
    fail_once = True

    def resume_once(*args, **kwargs):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise RuntimeError("injected decode adoption failure")
        return resume(*args, **kwargs)

    decode_sched.resume_with_kv = resume_once
    coordinator = PDCoordinator(
        prefill_scheduler=prefill_sched,
        prefill_runner=_ToyRunner(),
        decode_scheduler=decode_sched,
        decode_runner=_ToyRunner(),
        handoff=handoff,
        max_transfer_retries=1,
    )
    request = EngineRequest(
        "r",
        prompt_token_ids=(1, 2, 3, 4, 5, 6),
        max_new_tokens=3,
    )
    coordinator.add_request(request)

    assert coordinator.run_to_completion() == _single_core_reference([request])
    assert len(allocations) == 2
    assert allocations[0]._freed is True
    assert coordinator.failed_requests == ()


def test_adoption_failure_reconciles_discard_that_raises_after_release() -> None:
    """A post-discard error must not make a freed allocation adoptable again."""
    decode_sched, decode_kv = _make_pair()
    prefill_sched, _ = _make_pair()
    handoff = LocalKVHandoff(decode_kv)
    allocations = []
    transfer = handoff.transfer

    def capture_transfer(*args, **kwargs):
        allocation = transfer(*args, **kwargs)
        allocations.append(allocation)
        return allocation

    handoff.transfer = capture_transfer
    resume = decode_sched.resume_with_kv
    fail_resume = True

    def resume_once(*args, **kwargs):
        nonlocal fail_resume
        if fail_resume:
            fail_resume = False
            raise RuntimeError("injected decode adoption failure")
        return resume(*args, **kwargs)

    decode_sched.resume_with_kv = resume_once
    discard = handoff.discard
    fail_discard = True

    def discard_then_raise(allocation):
        nonlocal fail_discard
        discard(allocation)
        if fail_discard:
            fail_discard = False
            raise RuntimeError("injected post-discard failure")

    handoff.discard = discard_then_raise
    coordinator = PDCoordinator(
        prefill_scheduler=prefill_sched,
        prefill_runner=_ToyRunner(),
        decode_scheduler=decode_sched,
        decode_runner=_ToyRunner(),
        handoff=handoff,
        max_transfer_retries=1,
    )
    request = EngineRequest(
        "r",
        prompt_token_ids=(1, 2, 3, 4, 5, 6),
        max_new_tokens=3,
    )
    coordinator.add_request(request)

    with pytest.raises(RuntimeError, match="post-discard failure"):
        coordinator.step_prefill()

    assert allocations[0]._freed is True
    assert "r" not in decode_sched.states
    assert coordinator.run_to_completion() == _single_core_reference([request])
    assert len(allocations) == 2
    assert coordinator.failed_requests == ()


# --- deferred handoff: the prefill-side lease (m18 D3 under m6 D4) ------------


class _WatchedKVCache(RadixKVCache):
    """Records every point at which prefill-side pages go back to the pool.

    Sharing the stream provider's event list puts releases and stream calls on
    ONE timeline, which is the only way to see the ordering bug: a release that
    lands while a deferred copy is still reading the page it hands back.
    """

    def __init__(self, log: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._log = log

    def commit_and_release(self, *args, **kwargs):
        self._log.append("release")
        return super().commit_and_release(*args, **kwargs)

    def release_preempted(self, *args, **kwargs):
        self._log.append("release")
        return super().release_preempted(*args, **kwargs)

    def free(self, *args, **kwargs):
        self._log.append("release")
        return super().free(*args, **kwargs)


class _LoggingRunner(_ToyRunner):
    """A runner that puts its forward on the same timeline as the stream calls.

    Without it the timeline shows only copies and releases, and a gate fenced
    directly in front of every kernel looks exactly like a gate that lets one
    through — which is how the un-overlapped version passed review twice.
    """

    def __init__(self, log: list[str], name: str) -> None:
        self._log = log
        self._name = name

    def execute(self, scheduled, states):
        self._log.append(f"execute:{self._name}")
        return super().execute(scheduled, states)


def _deferred_coordinator(
    *,
    failures: int = 0,
    max_transfer_retries: int = 1,
    provider=None,
):
    """A PDCoordinator whose handoff returns before its copy has finished."""
    from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff

    provider = provider or CpuNoopStream()
    log = provider.events
    prefill_kv = _WatchedKVCache(log, num_pages=64, page_size=4)
    prefill_sched = Scheduler(prefill_kv, max_num_batched_tokens=32, page_size=4)
    decode_sched, decode_kv = _make_pair()
    inner = LocalKVHandoff(decode_kv)
    coordinator = PDCoordinator(
        prefill_scheduler=prefill_sched,
        prefill_runner=_LoggingRunner(log, "prefill"),
        decode_scheduler=decode_sched,
        decode_runner=_LoggingRunner(log, "decode"),
        handoff=StreamCopyKVHandoff(
            _FlakyHandoff(inner, failures) if failures else inner, provider, defer=True
        ),
        max_transfer_retries=max_transfer_retries,
    )
    return coordinator, log


class _TransientQueryEvent:
    """A completion whose first non-blocking status query fails transiently."""

    def __init__(self, log: list[str]) -> None:
        self._log = log
        self._queries = 0
        self._completed = False

    def query(self) -> bool:
        self._queries += 1
        if self._queries == 1:
            self._log.append("query-error")
            raise RuntimeError("transient completion query failure")
        if not self._completed:
            self._completed = True
            self._log.append("complete")
        return True

    def synchronize(self) -> None:
        self._queries = max(self._queries, 1)
        self._log.append("wait")

    def wait(self, stream=None) -> None:  # noqa: ARG002 - mirrors torch.cuda.Event
        self._log.append("gate")


class _TransientQueryProvider:
    def __init__(self) -> None:
        self.events: list[str] = []

    def begin(self) -> None:
        self.events.append("begin")

    def synchronize(self) -> None:
        self.events.append("synchronize")

    def record(self) -> _TransientQueryEvent:
        self.events.append("record")
        return _TransientQueryEvent(self.events)

    def gate(self, event: _TransientQueryEvent) -> None:
        event.wait()


class _HeldEvent:
    def __init__(self, log: list[str]) -> None:
        self._log = log
        self.ready = False
        self._logged = False

    def _complete(self) -> None:
        self.ready = True
        if not self._logged:
            self._logged = True
            self._log.append("complete")

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        self._complete()

    def wait(self, stream=None) -> None:  # noqa: ARG002 - mirrors torch.cuda.Event
        self._log.append("gate")


class _HeldProvider:
    def __init__(self, *, fail_record: int | None = None) -> None:
        self.events: list[str] = []
        self.recorded: list[_HeldEvent] = []
        self._records = 0
        self._fail_record = fail_record

    def begin(self) -> None:
        self.events.append("begin")

    def synchronize(self) -> None:
        self.events.append("stream-sync")
        for event in self.recorded:
            event._complete()

    def record(self) -> _HeldEvent:
        self._records += 1
        if self._records == self._fail_record:
            self.events.append("record-error")
            raise RuntimeError("injected event record failure")
        self.events.append("record")
        event = _HeldEvent(self.events)
        self.recorded.append(event)
        return event

    def gate(self, event: _HeldEvent) -> None:
        event.wait()


def _assert_no_release_under_a_running_copy(log: list[str]) -> None:
    """The invariant: no prefill-side page is handed back while a copy is live.

    ``record`` opens a copy whose source pages are still being read; ``complete``
    is a successful event query. A ``release`` in between returns a page the copy
    has not finished reading, and the next prefill step may overwrite it.
    """
    outstanding = 0
    for index, event in enumerate(log):
        if event == "record":
            outstanding += 1
        elif event == "complete":
            outstanding -= 1
        elif event == "release":
            assert outstanding == 0, (
                f"released prefill pages at {index} with {outstanding} copy(s) "
                f"still reading them: {log}"
            )
    assert outstanding == 0, f"a deferred copy was never settled: {log}"
    assert "record" in log, "the handoff never deferred; the test proves nothing"
    assert "release" in log, "no prefill-side release happened; the test proves nothing"


def test_the_deferred_copy_is_settled_before_the_source_pages_are_released() -> None:
    coordinator, log = _deferred_coordinator()
    request = EngineRequest("r", prompt_token_ids=(1, 2, 3, 4, 5, 6), max_new_tokens=3)
    coordinator.add_request(request)

    outputs = coordinator.run_to_completion()

    assert outputs == _single_core_reference([request])
    _assert_no_release_under_a_running_copy(log)


def test_a_failed_deferred_transfer_settles_before_aborting_the_source() -> None:
    """The failure path releases too, and a raising transfer may already have
    queued part of the copy — so it has to be settled just the same."""
    coordinator, log = _deferred_coordinator(failures=1)
    request = EngineRequest("r", prompt_token_ids=(1, 2, 3, 4, 5, 6), max_new_tokens=3)
    coordinator.add_request(request)

    outputs = coordinator.run_to_completion()

    assert outputs == _single_core_reference([request])
    assert coordinator.failed_requests == ()
    _assert_no_release_under_a_running_copy(log)


def test_a_transient_completion_query_error_retains_both_page_leases() -> None:
    """A failed poll is not evidence that the transfer failed or completed."""
    provider = _TransientQueryProvider()
    coordinator, log = _deferred_coordinator(provider=provider)
    request = EngineRequest("r", prompt_token_ids=(1, 2, 3, 4, 5, 6), max_new_tokens=3)
    coordinator.add_request(request)
    coordinator.step_prefill()
    handover = coordinator._handover

    with pytest.raises(RuntimeError, match="transient completion query failure"):
        coordinator.step_prefill()

    assert coordinator._handover is handover
    assert coordinator._handoff.pending_events
    assert "r" not in coordinator.decode_scheduler.states
    assert "release" not in log

    outputs = coordinator.run_to_completion()

    assert outputs == _single_core_reference([request])
    assert coordinator._handover is None
    _assert_no_release_under_a_running_copy(log)


def test_a_batched_record_failure_keeps_earlier_completion_reachable() -> None:
    """Starting item B must not detach item A's still-unfinalized ownership."""
    provider = _HeldProvider(fail_record=2)
    coordinator, log = _deferred_coordinator(provider=provider)
    requests = [
        EngineRequest(name, prompt_token_ids=prompt, max_new_tokens=2)
        for name, prompt in (("a", (1, 2, 3, 4, 5, 6)), ("b", (7, 8, 9, 10, 11, 12)))
    ]
    for request in requests:
        coordinator.add_request(request)

    with pytest.raises(RuntimeError, match="event record failure"):
        coordinator.step_prefill()

    assert coordinator._handover is not None
    assert coordinator._handoff.pending_events
    assert "release" not in log
    coordinator.abort("a")
    coordinator.abort("b")
    assert coordinator._handoff.pending_events == ()
    assert log.index("complete") < log.index("release")


def test_partial_finalize_reentry_does_not_republish_the_first_item() -> None:
    coordinator, _ = _deferred_coordinator()
    requests = [
        EngineRequest(name, prompt_token_ids=prompt, max_new_tokens=2)
        for name, prompt in (("a", (1, 2, 3, 4)), ("b", (5, 6, 7, 8)))
    ]
    for request in requests:
        coordinator.add_request(request)
    original_finalize = coordinator._finalize_completion
    calls: dict[int, int] = {}
    failed = False

    def fail_second_once(completion):
        nonlocal failed
        calls[completion.sequence] = calls.get(completion.sequence, 0) + 1
        if completion.sequence == 2 and not failed:
            failed = True
            raise RuntimeError("injected second finalize failure")
        return original_finalize(completion)

    coordinator._finalize_completion = fail_second_once
    coordinator.step_prefill()
    with pytest.raises(RuntimeError, match="second finalize failure"):
        coordinator.step_prefill()

    outputs = coordinator.run_to_completion()

    assert outputs == _single_core_reference(requests)
    assert calls[1] == 1, "the already-finalized first allocation was published twice"


@pytest.mark.parametrize("failed_transfer", [False, True], ids=["adopt", "retry"])
def test_raise_after_completion_finalize_is_reenterable(failed_transfer) -> None:
    """Consuming a completion is a durable phase even if its caller then raises."""
    coordinator, _ = _deferred_coordinator(failures=int(failed_transfer))
    request = EngineRequest(
        "r",
        prompt_token_ids=(1, 2, 3, 4, 5, 6),
        max_new_tokens=3,
    )
    coordinator.add_request(request)
    original_finalize = coordinator._finalize_completion
    failed = False

    def finalize_then_raise(completion):
        nonlocal failed
        try:
            result = original_finalize(completion)
        except KVHandoffError as transfer_error:
            if failed_transfer and not failed:
                failed = True
                raise RuntimeError("injected post-finalize failure") from transfer_error
            raise
        if not failed_transfer and not failed:
            failed = True
            raise RuntimeError("injected post-finalize failure")
        return result

    coordinator._finalize_completion = finalize_then_raise
    coordinator.step_prefill()

    with pytest.raises(RuntimeError, match="post-finalize failure"):
        coordinator.step_prefill()

    assert coordinator._handoff.pending_events == ()
    assert coordinator.run_to_completion() == _single_core_reference([request])
    assert coordinator.failed_requests == ()
    assert coordinator._handover is None
    assert coordinator._handoff._settled == {}


def test_raw_deferred_publication_failure_is_normalized_and_retried() -> None:
    coordinator, _ = _deferred_coordinator(max_transfer_retries=1)
    request = EngineRequest(
        "r",
        prompt_token_ids=(1, 2, 3, 4, 5, 6),
        max_new_tokens=3,
    )
    coordinator.add_request(request)
    publish = coordinator._handoff.publish
    allocations = []
    failed = False

    def publish_once(allocation):
        nonlocal failed
        allocations.append(allocation)
        if not failed:
            failed = True
            raise RuntimeError("injected raw publication failure")
        return publish(allocation)

    coordinator._handoff.publish = publish_once

    assert coordinator.run_to_completion() == _single_core_reference([request])
    assert allocations[0]._freed is True
    assert len(allocations) == 2
    assert coordinator.failed_requests == ()
    assert coordinator._handoff._settled == {}


def test_interrupted_deferred_publication_retries_after_original_interrupt() -> None:
    coordinator, _ = _deferred_coordinator(max_transfer_retries=1)
    request = EngineRequest(
        "r",
        prompt_token_ids=(1, 2, 3, 4, 5, 6),
        max_new_tokens=3,
    )
    coordinator.add_request(request)
    publish = coordinator._handoff.publish
    allocations = []
    failed = False

    def interrupt_publish_once(allocation):
        nonlocal failed
        allocations.append(allocation)
        if not failed:
            failed = True
            raise KeyboardInterrupt
        return publish(allocation)

    coordinator._handoff.publish = interrupt_publish_once
    coordinator.step_prefill()

    with pytest.raises(KeyboardInterrupt):
        coordinator.step_prefill()

    assert allocations[0]._freed is True
    assert coordinator.run_to_completion() == _single_core_reference([request])
    assert len(allocations) == 2
    assert coordinator.failed_requests == ()
    assert coordinator._handoff._settled == {}


def test_completion_lookup_interruption_keeps_both_allocations_fail_closed() -> None:
    """Losing the token after registration must not detach the live copy."""
    coordinator, log = _deferred_coordinator()
    request = EngineRequest(
        "r",
        prompt_token_ids=(1, 2, 3, 4, 5, 6),
        max_new_tokens=3,
    )
    coordinator.add_request(request)

    def interrupt_completion_lookup(_allocation):
        raise KeyboardInterrupt

    coordinator._completion_for = interrupt_completion_lookup
    with pytest.raises(KeyboardInterrupt):
        coordinator.step_prefill()

    handover = coordinator._handover
    source = coordinator.prefill_scheduler.states["r#p0"].allocation
    assert handover is not None
    assert "r#p0" in handover.poisoned
    assert coordinator._handoff.pending_events
    assert source is not None and source._freed is False
    assert "release" not in log
    with pytest.raises(RuntimeError, match="completion was lost"):
        coordinator.abort("r")
    assert source._freed is False


@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, KeyboardInterrupt],
    ids=["exception", "base-exception"],
)
def test_raise_after_completion_registration_keeps_the_copy_owned(error_type) -> None:
    """A registered event cannot become an orphan before PD receives its token."""
    provider = _HeldProvider()
    coordinator, log = _deferred_coordinator(provider=provider)
    request = EngineRequest(
        "r",
        prompt_token_ids=(1, 2, 3, 4, 5, 6),
        max_new_tokens=3,
    )
    coordinator.add_request(request)
    register = coordinator._handoff._register
    failed = False

    def register_then_raise(*args):
        nonlocal failed
        completion = register(*args)
        if not failed:
            failed = True
            raise error_type("injected post-registration failure")
        return completion

    coordinator._handoff._register = register_then_raise
    if issubclass(error_type, Exception):
        coordinator.step_prefill()
    else:
        with pytest.raises(error_type, match="post-registration failure"):
            coordinator.step_prefill()

    handover = coordinator._handover
    source = coordinator.prefill_scheduler.states["r#p0"].allocation
    assert handover is not None
    assert source is not None and source._freed is False
    assert len(coordinator._handoff.pending_events) == 1
    assert tuple(coordinator._staged_sampled or ()) == ()

    provider.recorded[0]._complete()
    coordinator.step_prefill()
    assert len(provider.recorded) == 2
    provider.recorded[1]._complete()

    assert coordinator.run_to_completion() == _single_core_reference([request])
    assert coordinator._handoff.pending_events == ()
    assert coordinator._handoff._settled == {}
    _assert_no_release_under_a_running_copy(log)


def test_raise_after_decode_adoption_resumes_without_discarding_live_kv() -> None:
    """A post-adoption exception is not an adoption failure or retry signal."""
    coordinator, _ = _deferred_coordinator()
    request = EngineRequest(
        "r",
        prompt_token_ids=(1, 2, 3, 4, 5, 6),
        max_new_tokens=3,
    )
    coordinator.add_request(request)
    original_resume = coordinator.decode_scheduler.resume_with_kv
    failed = False

    def adopt_then_raise(*args, **kwargs):
        nonlocal failed
        result = original_resume(*args, **kwargs)
        if not failed:
            failed = True
            raise RuntimeError("injected post-adoption failure")
        return result

    coordinator.decode_scheduler.resume_with_kv = adopt_then_raise
    coordinator.step_prefill()
    with pytest.raises(RuntimeError, match="post-adoption failure"):
        coordinator.step_prefill()

    handover = coordinator._handover
    allocation = coordinator.decode_scheduler.states["r"].allocation
    assert handover is not None and "r#p0" in handover.resumed
    assert allocation is not None and allocation._freed is False

    assert coordinator.run_to_completion() == _single_core_reference([request])
    assert coordinator.failed_requests == ()
    assert coordinator._handover is None


def test_raise_after_source_commit_does_not_commit_token_zero_twice() -> None:
    """A successful source release must be reconciled before re-entry."""
    coordinator, _ = _deferred_coordinator()
    request = EngineRequest(
        "r",
        prompt_token_ids=(1, 2, 3, 4, 5, 6),
        max_new_tokens=3,
    )
    coordinator.add_request(request)
    original_update = coordinator.prefill_scheduler.update
    failed = False

    def commit_then_raise(*args, **kwargs):
        nonlocal failed
        result = original_update(*args, **kwargs)
        if not failed:
            failed = True
            raise RuntimeError("injected post-commit failure")
        return result

    coordinator.prefill_scheduler.update = commit_then_raise
    coordinator.step_prefill()
    with pytest.raises(RuntimeError, match="post-commit failure"):
        coordinator.step_prefill()

    source = coordinator.prefill_scheduler.states["r#p0"].allocation
    assert source is not None and source._freed is True
    assert coordinator._handover is not None
    assert "r#p0" not in coordinator._handover.commit

    assert coordinator.run_to_completion() == _single_core_reference([request])
    assert coordinator._handover is None


def test_owned_current_item_is_not_staged_for_a_second_transfer() -> None:
    """A caller error after registration may stage later items, not the owner."""
    coordinator, _ = _deferred_coordinator()
    requests = [
        EngineRequest(name, prompt_token_ids=prompt, max_new_tokens=2)
        for name, prompt in (
            ("a", (1, 2, 3, 4, 5, 6)),
            ("b", (7, 8, 9, 10, 11, 12)),
        )
    ]
    for request in requests:
        coordinator.add_request(request)
    original_handoff = coordinator._handoff_or_retry
    failed = False

    def register_then_raise(internal_id, token0, handover):
        nonlocal failed
        result = original_handoff(internal_id, token0, handover)
        if not failed:
            failed = True
            raise RuntimeError("injected post-registration failure")
        return result

    coordinator._handoff_or_retry = register_then_raise

    with pytest.raises(RuntimeError, match="post-registration failure"):
        coordinator.step_prefill()

    assert coordinator._handover is not None
    assert [item[0] for item in coordinator._handover.adopt] == ["a#p0"]
    assert tuple(coordinator._staged_sampled or ()) == ("b#p0",)
    assert coordinator.run_to_completion() == _single_core_reference(requests)
    assert coordinator._handover is None


def test_abort_during_handoff_never_adopts_the_unreported_token_zero() -> None:
    provider = _HeldProvider()
    coordinator, _ = _deferred_coordinator(provider=provider)
    coordinator.add_request(
        EngineRequest("r", prompt_token_ids=(1, 2, 3, 4, 5, 6), max_new_tokens=3)
    )
    coordinator.step_prefill()
    assert coordinator._handover is not None

    coordinator.abort("r")

    assert "r" not in coordinator.decode_scheduler.states
    assert coordinator.prefill_scheduler.output_tokens("r#p0") == ()
    assert coordinator.prefill_scheduler.finish_reason("r#p0") == "abort"
    assert coordinator.drain_carried_tokens() == {}


def test_cancel_reconciles_discard_that_raises_after_release() -> None:
    provider = _HeldProvider()
    coordinator, _ = _deferred_coordinator(provider=provider)
    coordinator.add_request(
        EngineRequest(
            "r",
            prompt_token_ids=(1, 2, 3, 4, 5, 6),
            max_new_tokens=3,
        )
    )
    coordinator.step_prefill()
    assert coordinator._handover is not None
    allocation = coordinator._handover.adopt[0][2]
    discard = coordinator._handoff.discard
    failed = False

    def discard_then_raise(candidate):
        nonlocal failed
        discard(candidate)
        if not failed:
            failed = True
            raise RuntimeError("injected post-discard cancellation failure")

    coordinator._handoff.discard = discard_then_raise

    with pytest.raises(RuntimeError, match="post-discard cancellation failure"):
        coordinator.abort("r")

    assert allocation._freed is True
    coordinator.abort("r")
    assert coordinator._handover is None
    assert "r" not in coordinator.decode_scheduler.states
    assert coordinator.prefill_scheduler.finish_reason("r#p0") == "abort"


@pytest.mark.parametrize(
    "boundary",
    ["decode_abort", "decode_forget", "prefill_abort"],
)
def test_cancel_mutations_are_reenterable_after_success_then_raise(boundary) -> None:
    coordinator, _ = _deferred_coordinator()
    coordinator.add_request(
        EngineRequest(
            "r",
            prompt_token_ids=(1, 2, 3, 4, 5, 6),
            max_new_tokens=3,
        )
    )

    def stop_after_resume(_request_id):
        raise RuntimeError("injected sampler carry failure")

    coordinator._carry_sampler_state = stop_after_resume
    coordinator.step_prefill()
    with pytest.raises(RuntimeError, match="sampler carry failure"):
        coordinator.step_prefill()

    if boundary == "decode_abort":
        owner, method_name = coordinator.decode_scheduler, "abort"
    elif boundary == "decode_forget":
        owner, method_name = coordinator.decode_scheduler, "forget"
    else:
        owner, method_name = coordinator.prefill_scheduler, "abort"
    mutation = getattr(owner, method_name)
    failed = False

    def mutate_then_raise(request_id):
        nonlocal failed
        mutation(request_id)
        if not failed:
            failed = True
            raise RuntimeError(f"injected post-{boundary} failure")

    setattr(owner, method_name, mutate_then_raise)

    with pytest.raises(RuntimeError, match=f"post-{boundary} failure"):
        coordinator.abort("r")

    coordinator.abort("r")
    assert coordinator._handover is None
    assert "r" not in coordinator.decode_scheduler.states
    assert coordinator.prefill_scheduler.finish_reason("r#p0") == "abort"
    assert coordinator.drain_carried_tokens() == {}


def test_abort_after_partial_adoption_drops_the_unreported_token_zero() -> None:
    coordinator, _ = _deferred_coordinator()
    coordinator.add_request(
        EngineRequest("r", prompt_token_ids=(1, 2, 3, 4, 5, 6), max_new_tokens=3)
    )
    fail_once = True

    def fail_carry_once(request_id):  # noqa: ARG001 - failure injection
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise RuntimeError("injected sampler carry failure")

    coordinator._carry_sampler_state = fail_carry_once
    coordinator.step_prefill()
    with pytest.raises(RuntimeError, match="sampler carry failure"):
        coordinator.step_prefill()

    assert coordinator.decode_scheduler.output_tokens("r") != ()
    coordinator.abort("r")

    assert "r" not in coordinator.decode_scheduler.states
    assert coordinator.prefill_scheduler.output_tokens("r#p0") == ()
    assert coordinator.prefill_scheduler.finish_reason("r#p0") == "abort"
    assert coordinator.drain_carried_tokens() == {}


def test_every_copy_in_a_batched_prefill_step_is_settled() -> None:
    """One prefill step transfers every prompt that completed in it; keeping only
    the last event would leave the earlier copies unordered."""
    coordinator, log = _deferred_coordinator()
    requests = [
        EngineRequest(name, prompt_token_ids=prompt, max_new_tokens=2)
        for name, prompt in (("a", (1, 2, 3, 4, 5, 6)), ("b", (7, 8, 9, 10, 11, 12)))
    ]
    for request in requests:
        coordinator.add_request(request)

    outputs = coordinator.run_to_completion()

    assert outputs == _single_core_reference(requests)
    assert log.count("record") == 2, f"expected one copy per prompt: {log}"
    _assert_no_release_under_a_running_copy(log)


def _assert_every_copy_overlaps_engine_work(log: list[str]) -> None:
    """The point of deferring: a copy must have engine work queued ALONGSIDE it.

    ``record`` starts a copy on the side stream; the first successful completion
    query closes its ownership window. Engine work queued in between runs against
    the copy. A query with nothing in between leaves the copy on the critical
    path even though the host was never blocked.
    """
    records = [index for index, event in enumerate(log) if event == "record"]
    assert records, "the handoff never deferred; the test proves nothing"
    for index in records:
        complete = next(
            (j for j in range(index + 1, len(log)) if log[j] == "complete"),
            None,
        )
        assert complete is not None, f"a deferred copy was never settled: {log}"
        overlapped = [event for event in log[index + 1 : complete] if event.startswith("execute")]
        assert overlapped, (
            f"the copy recorded at {index} completed at {complete} with no engine "
            f"work queued in between: nothing overlapped it, so deferring bought "
            f"only a non-blocking host. timeline: {log}"
        )


def test_a_deferred_copy_has_engine_work_queued_alongside_it() -> None:
    """[P1] the gate used to sit at the end of the step that started the copy.

    Both prompts here transfer in different steps, so each copy has a later
    prefill forward or a decode step available to overlap with — if the gate is
    positioned to let them through.
    """
    coordinator, log = _deferred_coordinator()
    requests = [
        # 24 + 24 tokens against a 32-token budget: `a` prefills in one step,
        # `b` spills into the next, so step 2 has a forward to queue while a's
        # copy is still running
        EngineRequest("a", prompt_token_ids=tuple(range(1, 25)), max_new_tokens=2),
        EngineRequest("b", prompt_token_ids=tuple(range(50, 74)), max_new_tokens=2),
    ]
    for request in requests:
        coordinator.add_request(request)

    outputs = coordinator.run_to_completion()

    assert outputs == _single_core_reference(requests)
    _assert_every_copy_overlaps_engine_work(log)
    _assert_no_release_under_a_running_copy(log)


def test_a_blocking_handoff_settles_inside_its_own_step() -> None:
    """The pipeline is the deferring path's alone: a blocking transfer already
    finished its copy, so holding the request back a step would buy nothing."""
    coordinator, _, _ = _make_coordinator()
    request = EngineRequest("r", prompt_token_ids=(1, 2, 3, 4, 5, 6), max_new_tokens=3)
    coordinator.add_request(request)

    coordinator.step_prefill()

    assert coordinator._handover is None
    assert "r" in coordinator._decode.states, "adoption was deferred for no reason"


def test_a_deferring_handoff_without_completion_polling_is_refused() -> None:
    """A stream gate alone cannot authorize host-visible page publication."""

    class _UngatedDeferring:
        defers = True

        def transfer(self, tokens, first_token, pages=()):  # pragma: no cover
            raise AssertionError("must not be reachable")

    prefill_sched, _ = _make_pair()
    decode_sched, decode_kv = _make_pair()
    with pytest.raises(ValueError, match="completion polling"):
        PDCoordinator(
            prefill_scheduler=prefill_sched,
            prefill_runner=_ToyRunner(),
            decode_scheduler=decode_sched,
            decode_runner=_ToyRunner(),
            handoff=_UngatedDeferring(),
        )


# --- PDLoopAdapter: the EngineLoop-facing seam --------------------------------


def _adapter(**kwargs):
    from kairyu.engine.core.pd_loop import PDLoopAdapter

    coordinator, prefill_kv, decode_kv = _make_coordinator(**kwargs)
    return PDLoopAdapter(coordinator), coordinator, prefill_kv, decode_kv


def test_submissions_enter_at_prefill_not_at_the_decode_scheduler() -> None:
    # EngineLoop._drain_ops calls add_request on the scheduler it was given, so
    # a bare coordinator wired as that scheduler admits requests into DECODE
    # with no prompt KV — the request could never produce a token.
    adapter, coordinator, _, _ = _adapter()

    adapter.add_request(EngineRequest("r", prompt_token_ids=(1, 2, 3, 4), max_new_tokens=2))

    assert coordinator.prefill_scheduler.waiting_ids == ("r#p0",)
    assert coordinator.decode_scheduler.states == {}
    assert adapter.has_unfinished()


def test_one_adapter_step_prefills_transfers_and_starts_decoding() -> None:
    adapter, coordinator, _, _ = _adapter()
    request = EngineRequest("r", prompt_token_ids=(1, 2, 3, 4), max_new_tokens=3)
    adapter.add_request(request)

    plan = adapter.schedule()

    # the handoff ran inside schedule(), so decode plans the same step
    assert [chunk.request_id for chunk in plan.scheduled] == ["r"]
    assert adapter.output_tokens("r") == (sum(request.prompt_token_ids) % _VOCAB,)
    assert adapter.states["r"] is coordinator.decode_scheduler.states["r"]


def test_chunked_prefill_reports_control_progress_with_an_empty_decode_plan() -> None:
    adapter, _, _, _ = _adapter()
    adapter.add_request(EngineRequest("r", prompt_token_ids=tuple(range(40)), max_new_tokens=2))

    plan = adapter.schedule()

    assert plan.scheduled == ()
    assert adapter.has_unfinished()
    assert adapter.reject_waiting_head() is None
    assert adapter.made_control_progress()


def test_a_request_still_at_prefill_is_visible_under_its_public_id() -> None:
    """EngineLoop reads per-request state by the id it submitted; while the
    request is prefill-side that state lives under the clone id."""
    adapter, coordinator, _, _ = _adapter()
    adapter.add_request(EngineRequest("r", prompt_token_ids=(1, 2, 3, 4), max_new_tokens=2))

    assert "r" in adapter.states
    assert adapter.states["r"] is coordinator.prefill_scheduler.states["r#p0"]
    assert adapter.output_tokens("r") == ()
    assert adapter.finish_reason("r") is None


def test_a_prompt_prefill_can_never_admit_finishes_instead_of_hanging() -> None:
    """C2, through the adapter: a prompt larger than the prefill cache is
    rejected there, and the public request must still reach a terminal state —
    it never enters the decode scheduler at all."""
    adapter, coordinator, _, _ = _adapter(prefill_pages=2)
    adapter.add_request(EngineRequest("r", prompt_token_ids=tuple(range(40)), max_new_tokens=2))

    plan = adapter.schedule()
    adapter.drain_rejected()

    assert plan.scheduled == ()
    assert not adapter.has_unfinished()
    assert adapter.states["r"].status.value == "finished"
    assert adapter.finish_reason("r") == "length"
    assert adapter.output_tokens("r") == ()


def test_a_request_that_exhausts_its_transfer_retries_finishes_as_abort() -> None:
    decode_sched, decode_kv = _make_pair()
    prefill_sched, _ = _make_pair()
    from kairyu.engine.core.pd_loop import PDLoopAdapter

    coordinator = PDCoordinator(
        prefill_scheduler=prefill_sched,
        prefill_runner=_ToyRunner(),
        decode_scheduler=decode_sched,
        decode_runner=_ToyRunner(),
        handoff=_FlakyHandoff(LocalKVHandoff(decode_kv), failures=10),
        max_transfer_retries=1,
    )
    adapter = PDLoopAdapter(coordinator)
    adapter.add_request(EngineRequest("r", prompt_token_ids=(1, 2, 3, 4), max_new_tokens=2))

    for _ in range(4):
        adapter.schedule()

    assert coordinator.failed_requests == ("r",)
    assert adapter.states["r"].status.value == "finished"
    assert adapter.finish_reason("r") == "abort"


def test_abort_reaches_whichever_half_holds_the_request() -> None:
    adapter, coordinator, _, _ = _adapter()
    adapter.add_request(EngineRequest("p", prompt_token_ids=(1, 2, 3, 4), max_new_tokens=4))
    adapter.add_request(EngineRequest("d", prompt_token_ids=(5, 6, 7, 8), max_new_tokens=4))
    adapter.schedule()  # both hand off to decode

    adapter.abort("d")
    adapter.forget("d")

    assert "d" not in coordinator.decode_scheduler.states
    assert "d#p0" not in coordinator.prefill_scheduler.states
    assert coordinator.internal_id_for("d") is None
    assert adapter.has_unfinished()  # "p" is untouched


def test_driving_the_adapter_like_engine_loop_generates_and_reclaims() -> None:
    """schedule -> execute -> update, the loop's own cycle, plus the E2 sweep: a
    long-running loop must not retain the prefill clone's state either."""
    adapter, coordinator, _, _ = _adapter()
    request = EngineRequest("r", prompt_token_ids=(1, 2, 3, 4), max_new_tokens=3)
    adapter.add_request(request)

    while adapter.has_unfinished():
        plan = adapter.schedule()
        if not plan.scheduled:
            adapter.reject_waiting_head()
            continue
        sampled = adapter.execute(plan.scheduled, adapter.states)
        adapter.update({rid: tokens[0].token_id for rid, tokens in sampled.items()})
    outputs = adapter.output_tokens("r")
    adapter.forget("r")

    assert outputs == _single_core_reference([request])["r"]
    assert coordinator.decode_scheduler.states == {}
    assert coordinator.prefill_scheduler.states == {}
    assert adapter.states == {}


# --- token 0 reaches the driver at all (m5 D5 under m8 D2) --------------------


class _MetaRunner(_ToyRunner):
    """A runner whose token 0 carries real sampling metadata."""

    def __init__(self, *, terminates: bool = False) -> None:
        self._terminates = terminates

    def execute(self, scheduled, states):
        sampled = super().execute(scheduled, states)
        return {
            request_id: (
                SampledToken(
                    tokens[0].token_id,
                    logprob=-0.5,
                    top_logprobs=((tokens[0].token_id, -0.5),),
                    grammar_terminated=self._terminates,
                ),
            )
            for request_id, tokens in sampled.items()
        }


def _meta_adapter(**runner_kwargs):
    from kairyu.engine.core.pd_loop import PDLoopAdapter

    prefill_sched, _ = _make_pair()
    decode_sched, decode_kv = _make_pair()
    coordinator = PDCoordinator(
        prefill_scheduler=prefill_sched,
        prefill_runner=_MetaRunner(**runner_kwargs),
        decode_scheduler=decode_sched,
        decode_runner=_ToyRunner(),
        handoff=LocalKVHandoff(decode_kv),
    )
    return PDLoopAdapter(coordinator), coordinator


def test_token0_is_carried_out_to_the_driver_with_its_metadata() -> None:
    """`resume_with_kv` commits token 0 straight into the decode outputs, so no
    `execute()` ever reports it — the driver has to be handed it separately."""
    adapter, _ = _meta_adapter()
    adapter.add_request(EngineRequest("r", prompt_token_ids=(1, 2, 3, 4), max_new_tokens=3))

    adapter.schedule()
    carried = adapter.drain_carried_tokens()

    assert set(carried) == {"r"}
    assert carried["r"].logprob == -0.5
    assert carried["r"].token_id == adapter.output_tokens("r")[0]
    # and draining is exactly once: a second step must not re-report it
    assert adapter.drain_carried_tokens() == {}


def test_a_grammar_terminating_token0_finishes_the_request_in_the_loop() -> None:
    """m8 D1's grammar finish has to survive the handoff: token 0 completing the
    grammar is the whole completion, and no decode step exists to notice it."""
    from kairyu.engine.engine_loop import EngineLoop
    from kairyu.engine.tokenizer import ToyTokenizer
    from kairyu.sampling_params import SamplingParams

    adapter, _ = _meta_adapter(terminates=True)
    loop = EngineLoop(ToyTokenizer(), adapter, adapter)
    loop.submit("r", "one two three four", SamplingParams(max_tokens=8, temperature=0.0))

    last = None
    for _ in range(10):
        if not loop.has_work():
            break
        for request_id, update in loop.step():
            if request_id == "r":
                last = update

    assert last is not None and last.finished
    assert len(last.outputs) == 1, "decode kept generating past a completed grammar"
