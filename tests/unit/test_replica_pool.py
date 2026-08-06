"""Tests for the DP ReplicaPool (design doc m5 D4)."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator

import pytest

from kairyu.engine.backend import (
    AdmissionUpperBound,
    CacheHint,
    EngineBackend,
    GenerationRequest,
    GenerationResult,
    UpstreamClientError,
)
from kairyu.engine.mock import MockBackend
from kairyu.engine.prompt import TokensPrompt
from kairyu.orchestration.replica import ReplicaPool
from kairyu.orchestration.router import JsonlRouterLog
from kairyu.sampling_params import SamplingParams


class FlakyBackend:
    """MockBackend wrapper whose calls can be toggled to fail; counts shutdowns."""

    def __init__(self, latency_s: float = 0.0) -> None:
        self._inner = MockBackend(latency_s=latency_s)
        self.failing = False
        self.client_failing = False  # raises a 4xx-style client error
        self.shutdown_count = 0
        self.shutdown_error = False

    @property
    def prompts_seen(self) -> tuple[str, ...]:
        return self._inner.prompts_seen

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if self.client_failing:
            raise UpstreamClientError("bad request", status_code=400)
        if self.failing:
            raise RuntimeError("injected failure")
        return await self._inner.generate(request)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        if self.failing:
            raise RuntimeError("injected failure")
        async for chunk in self._inner.stream(request):
            yield chunk

    async def shutdown(self) -> None:
        self.shutdown_count += 1
        if self.shutdown_error:
            raise RuntimeError("shutdown failed")


class KeyedValidationBackend(FlakyBackend):
    def __init__(
        self,
        validation_key: object,
        *,
        rejection: str | None = None,
    ) -> None:
        super().__init__()
        self._validation_key = validation_key
        self.rejection = rejection
        self.validation_calls = 0

    @property
    def request_validation_key(self) -> object:
        return self._validation_key

    def validate_request(self, request: GenerationRequest) -> None:
        self.validation_calls += 1
        if self.rejection is not None:
            raise ValueError(self.rejection)


class UnkeyedValidationBackend(FlakyBackend):
    def __init__(self) -> None:
        super().__init__()
        self.validation_calls = 0

    def validate_request(self, request: GenerationRequest) -> None:
        self.validation_calls += 1


class PreparingBackend(FlakyBackend):
    def __init__(
        self,
        name: str,
        events: list[tuple[str, str]],
        *,
        prepare_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.events = events
        self.prepare_error = prepare_error

    async def prepare_request(self, request: GenerationRequest) -> None:
        self.events.append(("prepare", self.name))
        if self.prepare_error is not None:
            raise self.prepare_error

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.events.append(("generate", self.name))
        return await super().generate(request)

    async def stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationResult]:
        self.events.append(("stream", self.name))
        async for chunk in super().stream(request):
            yield chunk


class KeyedPreparingBackend(PreparingBackend):
    @property
    def request_validation_key(self) -> object:
        return ("shared-validation",)


class PreparingAdmissionBackend(PreparingBackend):
    def __init__(self, name, events, tokens):
        super().__init__(name, events)
        self.tokens = tokens

    def admission_upper_bound(self, request):
        return AdmissionUpperBound(self.tokens, True)


class AdmissionBackend(FlakyBackend):
    def __init__(self, tokens: int, *, refundable: bool = True) -> None:
        super().__init__()
        self._bound = AdmissionUpperBound(
            tokens=tokens,
            refundable_on_exact_usage=refundable,
        )
        self.admission_calls = 0

    def admission_upper_bound(
        self,
        request: GenerationRequest,
    ) -> AdmissionUpperBound:
        self.admission_calls += 1
        return self._bound


class KeyedAdmissionBackend(AdmissionBackend):
    @property
    def admission_upper_bound_key(self):
        return (self._bound.tokens, self._bound.refundable_on_exact_usage)


class BlockingShutdownBackend(FlakyBackend):
    def __init__(self) -> None:
        super().__init__()
        self.shutdown_started = asyncio.Event()
        self.shutdown_release = asyncio.Event()
        self.shutdown_cancelled = False
        self.closed = False

    async def shutdown(self) -> None:
        self.shutdown_count += 1
        self.shutdown_started.set()
        try:
            await self.shutdown_release.wait()
        except asyncio.CancelledError:
            self.shutdown_cancelled = True
            raise
        self.closed = True


class SelfCancellingShutdownBackend(FlakyBackend):
    async def shutdown(self) -> None:
        self.shutdown_count += 1
        raise asyncio.CancelledError


class OrderedOutcomeBackend:
    """Let a successful in-flight call finish after a sibling ejects the pool."""

    def __init__(self) -> None:
        self._inner = MockBackend()
        self.success_started = asyncio.Event()
        self.release_success = asyncio.Event()

    async def _outcome(self, request: GenerationRequest) -> GenerationResult:
        if request.prompt == "late-success":
            self.success_started.set()
            await self.release_success.wait()
            return await self._inner.generate(request)
        await self.success_started.wait()
        raise RuntimeError("ordered failure")

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        return await self._outcome(request)

    async def stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationResult]:
        yield await self._outcome(request)

    async def shutdown(self) -> None:
        return None


def make_request(prompt: str, session_id: str | None = None) -> GenerationRequest:
    hint = CacheHint(session_id=session_id) if session_id is not None else None
    return GenerationRequest(
        request_id=f"req-{prompt}",
        prompt=prompt,
        sampling_params=SamplingParams(),
        cache_hint=hint,
    )


async def place_session(pool: ReplicaPool, backends: list[FlakyBackend], session: str) -> int:
    """Send one request for ``session`` and return the index of the replica that got it."""
    prompt = f"probe:{session}:{sum(len(b.prompts_seen) for b in backends)}"
    await pool.generate(make_request(prompt, session_id=session))
    return next(i for i, b in enumerate(backends) if prompt in b.prompts_seen)


def test_pool_satisfies_engine_backend_protocol():
    pool = ReplicaPool([MockBackend()])
    assert isinstance(pool, EngineBackend)


def test_constructor_rejects_empty_replicas():
    with pytest.raises(ValueError, match="at least 1 replica"):
        ReplicaPool([])


def test_constructor_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="unhealthy_after"):
        ReplicaPool([MockBackend()], unhealthy_after=0)
    with pytest.raises(ValueError, match="queue_depth_threshold"):
        ReplicaPool([MockBackend()], queue_depth_threshold=-1)


def test_admission_uses_safe_maximum_across_every_eligible_replica() -> None:
    smaller = AdmissionBackend(128)
    larger = AdmissionBackend(512, refundable=False)
    pool = ReplicaPool({"smaller": smaller, "larger": larger})

    bound = pool.admission_upper_bound(make_request("media"))

    assert bound == AdmissionUpperBound(
        tokens=512,
        refundable_on_exact_usage=False,
    )
    assert smaller.admission_calls == 1
    assert larger.admission_calls == 1


def test_admission_excludes_draining_replicas() -> None:
    active = AdmissionBackend(128)
    draining = AdmissionBackend(4096)
    pool = ReplicaPool({"active": active, "draining": draining})
    pool.drain("draining")

    assert pool.admission_upper_bound(make_request("media")).tokens == 128
    assert active.admission_calls == 1
    assert draining.admission_calls == 0


async def test_cached_ring_preserves_exact_hrw_and_membership_order() -> None:
    replica_ids = ("replica:one", "レプリカ-二", "replica-three")
    pool = ReplicaPool(
        {replica_id: MockBackend() for replica_id in replica_ids}
    )
    sessions = ("plain", "セッション", "contains:colon", " ")

    def expected(session_id: str, candidates: tuple[str, ...]) -> str:
        return max(
            candidates,
            key=lambda replica_id: hashlib.sha256(
                f"{session_id}:{replica_id}".encode()
            ).digest(),
        )

    for session_id in sessions:
        assert pool._select(make_request("select", session_id))[0] == expected(
            session_id,
            replica_ids,
        )

    pool.drain("レプリカ-二")
    eligible = ("replica:one", "replica-three")
    assert pool.eligible_ids == eligible
    for session_id in sessions:
        assert pool._select(make_request("drained", session_id))[0] == expected(
            session_id,
            eligible,
        )

    await pool.remove_replica("レプリカ-二")
    pool.add_replica(
        "replacement",
        MockBackend(),
        health_url="http://replacement/readyz",
    )
    assert pool.eligible_ids == eligible
    await pool.probe("replacement")
    eligible = (*eligible, "replacement")
    assert pool.eligible_ids == eligible
    for session_id in sessions:
        assert pool._select(make_request("replaced", session_id))[0] == expected(
            session_id,
            eligible,
        )

    replica_snapshot, healthy, snapshot_eligible, generations = (
        pool.membership_snapshot()
    )
    assert replica_snapshot == eligible
    assert healthy == eligible
    assert snapshot_eligible == eligible
    assert tuple(generations) == eligible


def test_validation_deduplicates_only_explicit_equivalent_backend_contracts():
    first = KeyedValidationBackend(("openai", "same"))
    equivalent = KeyedValidationBackend(("openai", "same"))
    distinct = KeyedValidationBackend(("openai", "different"))
    unhashable = KeyedValidationBackend(["malformed"])
    unkeyed_first = UnkeyedValidationBackend()
    unkeyed_second = UnkeyedValidationBackend()
    pool = ReplicaPool(
        {
            "first": first,
            "equivalent": equivalent,
            "distinct": distinct,
            "unhashable": unhashable,
            "unkeyed-first": unkeyed_first,
            "unkeyed-second": unkeyed_second,
        }
    )

    pool.validate_request(make_request("validate"))

    assert first.validation_calls == 1
    assert equivalent.validation_calls == 0
    assert distinct.validation_calls == 1
    assert unhashable.validation_calls == 1
    assert unkeyed_first.validation_calls == 1
    assert unkeyed_second.validation_calls == 1


def test_equivalent_validation_failure_rejects_before_dispatch():
    rejecting = KeyedValidationBackend("same", rejection="unsupported")
    equivalent = KeyedValidationBackend("same", rejection="unsupported")
    pool = ReplicaPool({"rejecting": rejecting, "equivalent": equivalent})

    with pytest.raises(
        ValueError,
        match="every eligible replica: unsupported",
    ):
        pool.validate_request(make_request("invalid"))

    assert rejecting.validation_calls == 1
    assert equivalent.validation_calls == 0


async def test_generate_delegates_and_returns_result():
    backend = MockBackend(responses={"hello": "world"})
    pool = ReplicaPool([backend])
    result = await pool.generate(make_request("hello"))
    assert result.text == "world"
    assert backend.prompts_seen == ("hello",)


@pytest.mark.parametrize("stream", [False, True])
async def test_dispatch_prepares_every_eligible_contract_before_placement(stream):
    events: list[tuple[str, str]] = []
    first = PreparingBackend("first", events)
    second = PreparingBackend("second", events)
    pool = ReplicaPool({"first": first, "second": second})
    request = make_request("prepared")

    if stream:
        results = [chunk async for chunk in pool.stream(request)]
        dispatch = "stream"
    else:
        results = [await pool.generate(request)]
        dispatch = "generate"

    assert results[-1].finished is True
    assert events[:2] == [("prepare", "first"), ("prepare", "second")]
    assert len(events) == 3
    assert events[2][0] == dispatch
    assert pool.outstanding == (0, 0)


@pytest.mark.parametrize("stream", [False, True])
async def test_prepare_failure_preserves_error_before_placement_or_dispatch(stream):
    events: list[tuple[str, str]] = []
    failure = RuntimeError("second preparation failed")
    first = PreparingBackend("first", events)
    second = PreparingBackend("second", events, prepare_error=failure)
    pool = ReplicaPool({"first": first, "second": second})
    request = make_request("rejected")

    with pytest.raises(RuntimeError, match="second preparation failed") as raised:
        if stream:
            async for _chunk in pool.stream(request):
                raise AssertionError("preparation failure dispatched a stream")
        else:
            await pool.generate(request)

    assert raised.value is failure
    assert events == [("prepare", "first"), ("prepare", "second")]
    assert first.prompts_seen == ()
    assert second.prompts_seen == ()
    assert pool.outstanding == (0, 0)
    assert sum(pool.decision_counts.values()) == 0


async def test_equal_validation_keys_do_not_deduplicate_backend_preparation():
    events: list[tuple[str, str]] = []
    first = KeyedPreparingBackend("first", events)
    second = KeyedPreparingBackend("second", events)
    pool = ReplicaPool({"first": first, "second": second})
    pool._entries["first"].outstanding = 1

    result = await pool.generate(make_request("instance-preparation"))

    assert result.finished is True
    assert events == [
        ("prepare", "first"),
        ("prepare", "second"),
        ("generate", "second"),
    ]


async def test_removed_replica_prepare_failure_does_not_reject_survivor():
    events: list[tuple[str, str]] = []

    class RemovedDuringPrepare(PreparingBackend):
        def __init__(self) -> None:
            super().__init__("removed", events)
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def prepare_request(self, request: GenerationRequest) -> None:
            self.events.append(("prepare", self.name))
            self.entered.set()
            await self.release.wait()
            raise RuntimeError("detached backend stopped")

    removed = RemovedDuringPrepare()
    survivor = PreparingBackend("survivor", events)
    pool = ReplicaPool({"removed": removed, "survivor": survivor})
    request = make_request("rolling-removal")
    preparation = asyncio.create_task(pool.prepare_request(request))

    await removed.entered.wait()
    removal = asyncio.create_task(pool.remove_replica("removed"))
    removed.release.set()
    await removal

    assert await preparation == ("survivor",)
    result = await pool.generate(request)
    assert result.finished is True
    assert events == [
        ("prepare", "removed"),
        ("prepare", "survivor"),
        ("generate", "survivor"),
    ]


async def test_prepare_skips_later_snapshot_member_removed_during_prior_await():
    events: list[tuple[str, str]] = []

    class GatedSurvivor(PreparingBackend):
        def __init__(self) -> None:
            super().__init__("survivor", events)
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def prepare_request(self, request: GenerationRequest) -> None:
            self.events.append(("prepare", self.name))
            self.entered.set()
            await self.release.wait()

    class RemovedBeforeTurn(PreparingBackend):
        def __init__(self) -> None:
            super().__init__("removed", events)
            self.closed = False
            self.prepare_after_close = False

        async def prepare_request(self, request: GenerationRequest) -> None:
            self.prepare_after_close = self.closed
            self.events.append(("prepare", self.name))

        async def shutdown(self) -> None:
            self.closed = True

    survivor = GatedSurvivor()
    removed = RemovedBeforeTurn()
    pool = ReplicaPool({"survivor": survivor, "removed": removed})
    request = make_request("stale-later-prepare")
    preparation = asyncio.create_task(pool.prepare_request(request))

    await survivor.entered.wait()
    await pool.remove_replica("removed")
    survivor.release.set()

    assert await preparation == ("survivor",)
    assert removed.prepare_after_close is False
    assert events == [("prepare", "survivor")]


async def test_prepared_lease_cannot_reexpand_after_admission_shrinks_it():
    events: list[tuple[str, str]] = []
    first = PreparingAdmissionBackend("first", events, 10)
    second = PreparingAdmissionBackend("second", events, 100)
    pool = ReplicaPool({"first": first, "second": second})
    request = make_request("monotonic-lease")

    await pool.prepare_request(request)
    pool.drain("second")
    bound = await pool.admission_upper_bound_async(request)
    pool.cancel_drain("second")
    pool._entries["first"].outstanding = 1
    result = await pool.generate(request)

    assert bound.tokens == 10
    assert result.finished is True
    assert events == [
        ("prepare", "first"),
        ("prepare", "second"),
        ("generate", "first"),
    ]


async def test_concurrent_prepare_is_single_flight_and_cannot_publish_newcomer():
    events: list[tuple[str, str]] = []

    class GatedBackend(PreparingAdmissionBackend):
        def __init__(self, name: str, tokens: int) -> None:
            super().__init__(name, events, tokens)
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def prepare_request(self, request: GenerationRequest) -> None:
            self.events.append(("prepare", self.name))
            self.entered.set()
            await self.release.wait()

    original = GatedBackend("original", 10)
    newcomer = GatedBackend("newcomer", 100)
    pool = ReplicaPool({"original": original})
    request = make_request("concurrent-preflight")

    first = asyncio.create_task(pool.prepare_request(request))
    await original.entered.wait()
    pool.add_replica("newcomer", newcomer)
    second = asyncio.create_task(pool.prepare_request(request))
    await asyncio.sleep(0)

    assert newcomer.entered.is_set() is False
    original.release.set()
    assert await asyncio.gather(first, second) == [
        ("original",),
        ("original",),
    ]
    bound = await pool.admission_upper_bound_async(request)
    pool._entries["original"].outstanding = 1
    result = await pool.generate(request)

    assert bound.tokens == 10
    assert result.finished is True
    assert events == [
        ("prepare", "original"),
        ("generate", "original"),
    ]


async def test_cancelling_one_prepare_waiter_does_not_cancel_shared_flight():
    events: list[tuple[str, str]] = []

    class GatedBackend(PreparingBackend):
        def __init__(self) -> None:
            super().__init__("only", events)
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def prepare_request(self, request: GenerationRequest) -> None:
            self.events.append(("prepare", self.name))
            self.entered.set()
            await self.release.wait()

    backend = GatedBackend()
    pool = ReplicaPool({"only": backend})
    request = make_request("cancelled-waiter")
    cancelled = asyncio.create_task(pool.prepare_request(request))
    survivor = asyncio.create_task(pool.prepare_request(request))

    await backend.entered.wait()
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    backend.release.set()

    assert await survivor == ("only",)
    assert events == [("prepare", "only")]


async def test_async_admission_ignores_removed_failing_member_and_rebounds_survivor():
    events: list[tuple[str, str]] = []

    class RemovedDuringBound(PreparingBackend):
        def __init__(self) -> None:
            super().__init__("removed", events)
            self.bound_entered = asyncio.Event()
            self.bound_release = asyncio.Event()
            self.closed = False

        async def admission_upper_bound_async(self, request: GenerationRequest):
            self.bound_entered.set()
            await self.bound_release.wait()
            if self.closed:
                raise RuntimeError("detached backend stopped")
            return AdmissionUpperBound(100, True)

        async def shutdown(self) -> None:
            self.closed = True
            self.bound_release.set()

    removed = RemovedDuringBound()
    survivor = PreparingAdmissionBackend("survivor", events, 10)
    pool = ReplicaPool({"removed": removed, "survivor": survivor})
    request = make_request("admission-removal")
    await pool.prepare_request(request)
    admission = asyncio.create_task(pool.admission_upper_bound_async(request))

    await removed.bound_entered.wait()
    await pool.remove_replica("removed")
    bound = await admission
    result = await pool.generate(request)

    assert bound == AdmissionUpperBound(10, True)
    assert result.finished is True
    assert events == [
        ("prepare", "removed"),
        ("prepare", "survivor"),
        ("generate", "survivor"),
    ]


async def test_async_admission_skips_later_member_removed_during_prior_bound():
    class GatedSurvivor(FlakyBackend):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0

        async def admission_upper_bound_async(self, request: GenerationRequest):
            self.calls += 1
            self.entered.set()
            await self.release.wait()
            return AdmissionUpperBound(10, True)

    class RemovedBeforeTurn(FlakyBackend):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False
            self.bound_after_close = False

        async def admission_upper_bound_async(self, request: GenerationRequest):
            self.bound_after_close = self.closed
            return AdmissionUpperBound(100, True)

        async def shutdown(self) -> None:
            self.closed = True

    survivor = GatedSurvivor()
    removed = RemovedBeforeTurn()
    pool = ReplicaPool({"survivor": survivor, "removed": removed})
    request = make_request("stale-later-bound")
    await pool.prepare_request(request)
    admission = asyncio.create_task(pool.admission_upper_bound_async(request))

    await survivor.entered.wait()
    await pool.remove_replica("removed")
    survivor.release.set()

    assert await admission == AdmissionUpperBound(10, True)
    assert survivor.calls == 2
    assert removed.bound_after_close is False


async def test_failed_preparation_cannot_reenter_lease_after_undrain():
    events: list[tuple[str, str]] = []

    class GatedFailure(PreparingBackend):
        def __init__(self):
            super().__init__("bad", events)
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def prepare_request(self, request):
            self.events.append(("prepare", self.name))
            self.entered.set()
            await self.release.wait()
            raise RuntimeError("not prepared")

    class GatedSuccess(PreparingBackend):
        def __init__(self):
            super().__init__("good", events)
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def prepare_request(self, request):
            self.events.append(("prepare", self.name))
            self.entered.set()
            await self.release.wait()

    bad = GatedFailure()
    good = GatedSuccess()
    pool = ReplicaPool({"bad": bad, "good": good})
    request = make_request("failed-lease-member")
    preparation = asyncio.create_task(pool.prepare_request(request))

    await bad.entered.wait()
    pool.drain("bad")
    bad.release.set()
    await good.entered.wait()
    pool.cancel_drain("bad")
    good.release.set()

    assert await preparation == ("good",)
    result = await pool.generate(request)
    assert result.finished is True
    assert events == [
        ("prepare", "bad"),
        ("prepare", "good"),
        ("generate", "good"),
    ]


async def test_typed_prepared_lease_ignores_new_unvalidated_candidate():
    original = KeyedValidationBackend("original")
    newcomer = KeyedValidationBackend("newcomer", rejection="tokens rejected")
    pool = ReplicaPool({"original": original})
    request = GenerationRequest(
        request_id="typed-membership-lease",
        prompt=TokensPrompt((1, 2)),
        sampling_params=SamplingParams(max_tokens=1),
    )

    assert await pool.prepare_request(request) == ("original",)
    pool.add_replica("newcomer", newcomer)
    result = await pool.generate(request)

    assert result.finished is True
    assert original.validation_calls == 1
    assert newcomer.validation_calls == 0


async def test_text_prepare_revalidates_new_snapshot_member_before_leasing():
    original = KeyedValidationBackend("original")
    rejecting = KeyedValidationBackend("rejecting", rejection="text rejected")
    pool = ReplicaPool({"original": original})
    request = make_request("text-membership-validation")

    pool.validate_request(request)
    preparation = asyncio.create_task(pool.prepare_request(request))
    pool.add_replica("rejecting", rejecting)

    with pytest.raises(ValueError, match="text rejected"):
        await preparation
    assert original.validation_calls == 2
    assert rejecting.validation_calls == 1


@pytest.mark.parametrize("stream", [False, True])
async def test_preflight_lease_excludes_replica_added_while_prepare_awaits(stream):
    events: list[tuple[str, str]] = []

    class GatedPreparingBackend(PreparingBackend):
        def __init__(self):
            super().__init__("original", events)
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def prepare_request(self, request: GenerationRequest) -> None:
            self.events.append(("prepare", self.name))
            self.entered.set()
            await self.release.wait()

        def admission_upper_bound(self, request):
            return AdmissionUpperBound(10, True)

    class NewcomerBackend(PreparingBackend):
        def admission_upper_bound(self, request):
            return AdmissionUpperBound(100, True)

    original = GatedPreparingBackend()
    newcomer = NewcomerBackend("newcomer", events)
    pool = ReplicaPool({"original": original})
    request = make_request("membership-race")
    preparation = asyncio.create_task(pool.prepare_request(request))

    await original.entered.wait()
    pool.add_replica("newcomer", newcomer)
    # A fresh least-outstanding placement would prefer the newcomer.
    pool._entries["original"].outstanding = 1
    original.release.set()
    await preparation
    bound = await pool.admission_upper_bound_async(request)

    if stream:
        result = [chunk async for chunk in pool.stream(request)][-1]
        dispatch = "stream"
    else:
        result = await pool.generate(request)
        dispatch = "generate"

    assert result.finished is True
    assert bound.tokens == 10
    assert events == [
        ("prepare", "original"),
        (dispatch, "original"),
    ]
    assert pool.outstanding == (1, 0)


async def test_async_admission_deduplicates_large_homogeneous_contract_pool():
    backends = [KeyedAdmissionBackend(128) for _ in range(181)]
    pool = ReplicaPool(backends)

    bound = await pool.admission_upper_bound_async(make_request("bounded"))

    assert bound.tokens == 128
    assert sum(backend.admission_calls for backend in backends) == 1


async def test_cancelled_remove_finishes_detached_backend_shutdown() -> None:
    backend = BlockingShutdownBackend()
    pool = ReplicaPool({"detached": backend})
    task = asyncio.create_task(pool.remove_replica("detached"))
    await asyncio.wait_for(backend.shutdown_started.wait(), timeout=1)

    task.cancel()
    await asyncio.sleep(0)
    assert pool.replica_ids == ()
    assert not task.done()
    backend.shutdown_release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend.closed is True
    assert backend.shutdown_cancelled is False
    assert backend.shutdown_count == 1


async def test_membership_observer_failure_cannot_break_backend_ownership() -> None:
    def broken_observer(_event) -> None:
        raise RuntimeError("observer failed")

    pool = ReplicaPool(
        {},
        allow_empty=True,
        membership_event_sink=broken_observer,
    )
    backend = FlakyBackend()

    pool.add_replica("replica", backend)
    assert pool.replica_ids == ("replica",)

    await pool.remove_replica("replica")
    assert pool.replica_ids == ()
    assert backend.shutdown_count == 1


async def test_self_cancelled_detached_backend_shutdown_is_not_silent() -> None:
    backend = SelfCancellingShutdownBackend()
    pool = ReplicaPool({"detached": backend})

    with pytest.raises(
        ExceptionGroup,
        match="replica 'detached' shutdown failed",
    ) as raised:
        await pool.remove_replica("detached")

    assert pool.replica_ids == ()
    assert backend.shutdown_count == 1
    assert len(raised.value.exceptions) == 1
    error = raised.value.exceptions[0]
    assert isinstance(error, RuntimeError)
    assert "SelfCancellingShutdownBackend.shutdown raised CancelledError" in str(
        error
    )
    assert isinstance(error.__cause__, asyncio.CancelledError)


async def test_session_affinity_sticks_across_calls():
    backends = [FlakyBackend() for _ in range(3)]
    pool = ReplicaPool(backends)
    first = await place_session(pool, backends, "session-A")
    for _ in range(5):
        assert await place_session(pool, backends, "session-A") == first


async def test_distinct_sessions_spread_over_replicas():
    backends = [FlakyBackend() for _ in range(4)]
    pool = ReplicaPool(backends)
    placements = {
        await place_session(pool, backends, f"session-{i}") for i in range(32)
    }
    assert len(placements) > 1  # rendezvous hashing does not funnel all sessions to one


async def test_rendezvous_remap_only_moves_ejected_sessions():
    backends = [FlakyBackend() for _ in range(3)]
    pool = ReplicaPool(backends, unhealthy_after=2)
    sessions = [f"session-{i}" for i in range(30)]
    before = {s: await place_session(pool, backends, s) for s in sessions}
    ejected = before[sessions[0]]

    # Eject that replica: fail `unhealthy_after` consecutive requests on it.
    backends[ejected].failing = True
    for _ in range(2):
        with pytest.raises(RuntimeError, match="injected failure"):
            await pool.generate(make_request("fail-me", session_id=sessions[0]))
    backends[ejected].failing = False  # healthy again, but still off the ring

    after = {s: await place_session(pool, backends, s) for s in sessions}
    for session in sessions:
        if before[session] == ejected:
            assert after[session] != ejected  # remapped off the ejected replica
        else:
            assert after[session] == before[session]  # everyone else stays put


async def test_client_errors_do_not_eject_the_replica():
    # O1: a bad client request (4xx) repeated on a session-pinned replica must
    # NOT count as a replica failure — otherwise one misbehaving client could
    # cascade-eject the whole pool.
    backends = [FlakyBackend() for _ in range(3)]
    pool = ReplicaPool(backends, unhealthy_after=2)
    target = await place_session(pool, backends, "sticky")
    backends[target].client_failing = True
    for _ in range(5):  # far more than unhealthy_after
        with pytest.raises(UpstreamClientError):
            await pool.generate(make_request("bad", session_id="sticky"))
    backends[target].client_failing = False
    # the replica is still healthy and still serves the session
    assert await place_session(pool, backends, "sticky") == target


async def test_sessionless_goes_to_least_outstanding_with_lowest_index_ties():
    backends = [MockBackend(latency_s=0.05) for _ in range(2)]
    pool = ReplicaPool(backends)
    # All idle: first sessionless request must pick the lowest index.
    task = asyncio.ensure_future(pool.generate(make_request("first")))
    await asyncio.sleep(0.01)
    assert backends[0].prompts_seen == ()  # still in flight; check dispatch below
    # While replica 0 is busy, the next sessionless request goes to replica 1.
    await pool.generate(make_request("second"))
    await task
    assert backends[0].prompts_seen == ("first",)
    assert backends[1].prompts_seen == ("second",)


async def test_queue_depth_fallback_overflows_affine_replica():
    backends = [FlakyBackend(latency_s=0.1) for _ in range(2)]
    pool = ReplicaPool(backends, queue_depth_threshold=0)
    session = "hot-session"
    fast_pool = ReplicaPool(backends)  # shares backends only to discover affinity
    affine = await place_session(fast_pool, backends, session)
    other = 1 - affine

    in_flight = asyncio.ensure_future(
        pool.generate(make_request("occupy", session_id=session))
    )
    await asyncio.sleep(0.02)  # let it dispatch to the affine replica
    await pool.generate(make_request("overflow", session_id=session))
    await in_flight
    assert any("occupy" in p for p in backends[affine].prompts_seen)
    assert any("overflow" in p for p in backends[other].prompts_seen)


async def test_health_ejection_after_consecutive_failures_and_probe_recovery():
    backends = [FlakyBackend() for _ in range(2)]
    pool = ReplicaPool(backends, unhealthy_after=2)
    session = "sticky"
    affine = await place_session(pool, backends, session)
    other = 1 - affine

    backends[affine].failing = True
    for _ in range(2):
        with pytest.raises(RuntimeError, match="injected failure"):
            await pool.generate(make_request("boom", session_id=session))
    backends[affine].failing = False

    # Ejected: the affine session now lands on the surviving replica.
    assert await place_session(pool, backends, session) == other

    # A successful probe returns the replica to the ring; affinity is restored.
    await pool.probe(affine)
    assert await place_session(pool, backends, session) == affine


async def test_remote_replica_requires_probe_then_ejects_and_restores():
    trusted = FlakyBackend()
    remote = FlakyBackend()
    pool = ReplicaPool({"trusted": trusted}, unhealthy_after=2)
    pool.add_replica("remote", remote, health_url="http://remote/readyz")
    await pool.remove_replica("trusted")

    assert pool.validated_by_id() == {"remote": False}
    assert pool.healthy_by_id() == {"remote": False}
    with pytest.raises(RuntimeError, match="eligible"):
        await pool.generate(make_request("before-probe"))

    lease = pool.acquire_drain("remote")
    pool.drain("remote")
    await pool.probe("remote")
    assert pool.validated_by_id() == {"remote": True}
    assert pool.healthy_by_id() == {"remote": True}
    assert pool.is_draining("remote") is True
    assert pool.is_manually_draining("remote") is True
    with pytest.raises(RuntimeError, match="eligible"):
        await pool.generate(make_request("still-draining"))

    pool.cancel_drain("remote")
    assert pool.is_draining("remote") is True
    pool.release_drain("remote", lease)
    assert pool.is_draining("remote") is False
    assert (await pool.generate(make_request("validated"))).finished
    remote.failing = True
    with pytest.raises(RuntimeError, match="injected failure"):
        await pool.generate(make_request("failure-1"))
    assert pool.healthy_by_id() == {"remote": True}
    with pytest.raises(RuntimeError, match="injected failure"):
        await pool.generate(make_request("failure-2"))
    assert pool.validated_by_id() == {"remote": True}
    assert pool.healthy_by_id() == {"remote": False}

    remote.failing = False
    await pool.probe("remote")
    assert pool.healthy_by_id() == {"remote": True}
    assert (await pool.generate(make_request("restored"))).finished


async def test_local_replica_starts_validated_and_can_require_probe_by_id():
    pool = ReplicaPool({"local": MockBackend()})

    assert pool.validated_by_id() == {"local": True}
    assert pool.healthy_by_id() == {"local": True}
    assert (await pool.generate(make_request("trusted"))).finished

    pool.require_probe("local")
    assert pool.validated_by_id() == {"local": False}
    assert pool.healthy_by_id() == {"local": False}
    with pytest.raises(RuntimeError, match="eligible"):
        await pool.generate(make_request("unknown"))

    await pool.probe("local")
    assert pool.validated_by_id() == {"local": True}
    assert (await pool.generate(make_request("revalidated"))).finished


async def test_same_id_remote_readd_does_not_inherit_validation():
    pool = ReplicaPool({"local": MockBackend()})
    pool.add_replica("remote", MockBackend(), health_url="http://old/readyz")
    await pool.probe("remote")
    old_generation = pool.entry_generation("remote")
    assert pool.validated_by_id()["remote"] is True

    await pool.remove_replica("remote")
    pool.add_replica("remote", MockBackend(), health_url="http://new/readyz")

    assert pool.entry_generation("remote") != old_generation
    assert pool.validated_by_id()["remote"] is False
    assert pool.healthy_by_id()["remote"] is False


async def test_failure_count_resets_on_any_success():
    backends = [FlakyBackend()]
    pool = ReplicaPool(backends, unhealthy_after=3)
    for _ in range(2):
        backends[0].failing = True
        with pytest.raises(RuntimeError):
            await pool.generate(make_request("x"))
        backends[0].failing = False
    await pool.generate(make_request("ok"))  # resets the consecutive-failure count
    backends[0].failing = True
    for _ in range(2):
        with pytest.raises(RuntimeError, match="injected failure"):
            await pool.generate(make_request("y"))
    backends[0].failing = False
    # Still healthy (2 failures < 3 after the reset), so it serves again.
    result = await pool.generate(make_request("still-alive"))
    assert result.finished


@pytest.mark.parametrize("surface", ["generate", "stream"])
async def test_late_inflight_success_cannot_silently_restore_ejected_replica(
    surface: str,
) -> None:
    backend = OrderedOutcomeBackend()
    events: list[dict[str, object]] = []
    pool = ReplicaPool(
        {"replica": backend},
        unhealthy_after=1,
        membership_event_sink=events.append,
    )

    async def invoke(prompt: str) -> None:
        request = make_request(prompt)
        if surface == "generate":
            await pool.generate(request)
        else:
            async for _chunk in pool.stream(request):
                pass

    successful = asyncio.create_task(invoke("late-success"))
    await asyncio.wait_for(backend.success_started.wait(), timeout=1)
    with pytest.raises(RuntimeError, match="ordered failure"):
        await invoke("failure")
    assert pool.eligible_ids == ()
    assert [event["reason"] for event in events] == ["health_ejected"]

    backend.release_success.set()
    await successful
    assert pool.eligible_ids == ()
    assert [event["reason"] for event in events] == ["health_ejected"]

    await pool.probe("replica")
    assert pool.eligible_ids == ("replica",)
    assert [event["reason"] for event in events] == [
        "health_ejected",
        "probe_succeeded",
    ]


async def test_all_unhealthy_raises_clear_runtime_error():
    backends = [FlakyBackend()]
    pool = ReplicaPool(backends, unhealthy_after=1)
    backends[0].failing = True
    with pytest.raises(RuntimeError, match="injected failure"):
        await pool.generate(make_request("boom"))
    with pytest.raises(RuntimeError, match="none of the 1 replicas are eligible"):
        await pool.generate(make_request("nobody-home"))


async def test_probe_rejects_invalid_index():
    pool = ReplicaPool([MockBackend()])
    with pytest.raises(ValueError, match="replica index"):
        await pool.probe(5)


async def test_placement_logged_with_hashed_session(tmp_path):
    log_path = tmp_path / "router.jsonl"
    log = JsonlRouterLog(log_path)
    pool = ReplicaPool([MockBackend(), MockBackend()], log=log)
    await pool.generate(make_request("hello", session_id="secret-session"))
    await pool.generate(make_request("anon"))
    log.flush()
    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [line["kind"] for line in lines] == ["replica", "replica"]
    assert lines[0]["session_sha256"] == hashlib.sha256(b"secret-session").hexdigest()
    assert lines[0]["reason"] == "session_affinity"
    assert lines[1]["session_sha256"] is None
    assert lines[1]["reason"] == "least_outstanding"
    assert all(isinstance(line["replica"], int) for line in lines)
    assert "secret-session" not in log_path.read_text()  # raw session id never stored
    log.close()


async def test_placement_logged_before_dispatch_even_on_failure(tmp_path):
    log_path = tmp_path / "router.jsonl"
    backends = [FlakyBackend()]
    backends[0].failing = True
    log = JsonlRouterLog(log_path)
    pool = ReplicaPool(backends, log=log)
    with pytest.raises(RuntimeError, match="injected failure"):
        await pool.generate(make_request("boom", session_id="s"))
    log.flush()
    entry = json.loads(log_path.read_text().splitlines()[0])
    assert entry["kind"] == "replica"
    log.close()


async def test_pool_shutdown_drains_owned_router_log(tmp_path):
    log_path = tmp_path / "router.jsonl"
    log = JsonlRouterLog(log_path)
    pool = ReplicaPool([MockBackend()], log=log)
    await pool.generate(make_request("hello"))

    await pool.shutdown()

    entry = json.loads(log_path.read_text().splitlines()[0])
    assert entry["kind"] == "replica"


async def test_outstanding_accounting_under_concurrent_load():
    backends = [MockBackend(latency_s=0.05) for _ in range(2)]
    pool = ReplicaPool(backends)
    requests = [make_request(f"load-{i}") for i in range(8)]

    async def snapshot_later() -> tuple[int, ...]:
        await asyncio.sleep(0.02)  # all 8 dispatched, none finished
        return pool.outstanding

    mid_flight, *_ = await asyncio.gather(
        snapshot_later(), *(pool.generate(r) for r in requests)
    )
    assert sum(mid_flight) == 8
    assert mid_flight == (4, 4)  # least-outstanding spreads evenly
    assert pool.outstanding == (0, 0)  # decremented on completion
    assert len(backends[0].prompts_seen) == 4
    assert len(backends[1].prompts_seen) == 4


async def test_outstanding_decremented_on_error():
    backends = [FlakyBackend()]
    backends[0].failing = True
    pool = ReplicaPool(backends)
    with pytest.raises(RuntimeError, match="injected failure"):
        await pool.generate(make_request("boom"))
    assert pool.outstanding == (0,)


async def test_stream_delegates_with_affinity_and_accounting(tmp_path):
    log_path = tmp_path / "router.jsonl"
    backends = [FlakyBackend() for _ in range(2)]
    log = JsonlRouterLog(log_path)
    pool = ReplicaPool(backends, log=log)
    session = "stream-session"
    affine = await place_session(pool, backends, session)

    chunks = []
    async for chunk in pool.stream(make_request("stream me a story", session_id=session)):
        chunks.append(chunk)
        assert sum(pool.outstanding) == 1  # counted while streaming
    assert chunks[-1].finished
    assert len(chunks) > 1  # partials + final, per MockBackend chunking
    assert pool.outstanding == (0, 0)
    assert any("stream me a story" in p for p in backends[affine].prompts_seen)
    log.flush()
    last_entry = json.loads(log_path.read_text().splitlines()[-1])
    assert last_entry["kind"] == "replica"
    log.close()


async def test_stream_failure_counts_toward_health_and_decrements():
    backends = [FlakyBackend()]
    pool = ReplicaPool(backends, unhealthy_after=1)
    backends[0].failing = True
    with pytest.raises(RuntimeError, match="injected failure"):
        async for _ in pool.stream(make_request("boom")):
            pass
    assert pool.outstanding == (0,)
    with pytest.raises(RuntimeError, match="unhealthy"):
        await pool.generate(make_request("next"))


async def test_shutdown_shuts_down_all_members():
    backends = [FlakyBackend() for _ in range(3)]
    pool = ReplicaPool(backends)
    await pool.shutdown()
    assert [b.shutdown_count for b in backends] == [1, 1, 1]


async def test_remove_replica_closes_backend_exactly_once():
    removed = FlakyBackend()
    survivor = FlakyBackend()
    pool = ReplicaPool({"removed": removed, "survivor": survivor})

    await pool.remove_replica("removed")

    assert removed.shutdown_count == 1
    assert survivor.shutdown_count == 0
    assert pool.replica_ids == ("survivor",)
    await pool.shutdown()
    assert removed.shutdown_count == 1
    assert survivor.shutdown_count == 1


async def test_forced_remove_closes_inflight_backend_immediately():
    backend = FlakyBackend(latency_s=0.1)
    pool = ReplicaPool({"busy": backend})
    task = asyncio.create_task(pool.generate(make_request("busy")))
    await asyncio.sleep(0.01)

    await pool.remove_replica("busy", force=True)

    assert backend.shutdown_count == 1
    await task


async def test_shutdown_attempts_every_unique_backend_and_aggregates_errors():
    shared = FlakyBackend()
    failing = FlakyBackend()
    failing.shutdown_error = True
    pool = ReplicaPool({"shared-a": shared, "shared-b": shared, "bad": failing})

    with pytest.raises(ExceptionGroup, match="ReplicaPool shutdown") as caught:
        await pool.shutdown()

    assert shared.shutdown_count == 1
    assert failing.shutdown_count == 1
    assert len(caught.value.exceptions) == 1
