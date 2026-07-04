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
