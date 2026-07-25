"""P-D disaggregation coordinator: copy-on-handoff between two cores (design m5 D5).

The prefill core runs each request with ``max_new_tokens=1`` so no decode chunk
is ever scheduled there; the coordinator intercepts at execute-completion —
after the KV is written, before ``update()`` commits — transfers the prompt KV
plus token 0 to the decode core, and only then commits (copy-before-commit:
``commit_and_release`` must not pool-free the tail page under the copy). The
decode core adopts via ``Scheduler.resume_with_kv``. On CPU the "copy" is page
adoption in the destination cache; the GPU phase swaps in a device copy behind
the same ``KVHandoff`` seam, and M6 swaps in a network transport.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Protocol

from kairyu.engine.core.engine_core import ModelRunner, token_ids
from kairyu.engine.core.radix_kv import KVAllocation, KVCacheFull, RadixKVCache
from kairyu.engine.core.scheduler import EngineRequest, Scheduler

_PREFILL_ID_SEPARATOR = "#p"


class KVHandoffError(RuntimeError):
    """A KV transfer failed; the request may be retried on the prefill core."""


class KVHandoff(Protocol):
    def transfer(
        self, tokens: tuple[int, ...], first_token: int, pages: tuple[int, ...] = ()
    ) -> KVAllocation:
        """Move one prompt's KV to the destination; return its allocation there.

        ``pages`` are the SOURCE-side page ids holding the prompt's KV (m18:
        byte-extracting handoffs need them; the accounting-only LocalKVHandoff
        ignores them).
        """
        ...


class LocalKVHandoff:
    """Same-process handoff: adopt the prompt's pages in the destination cache.

    Page *contents* exist only on devices (GPU phase); CPU-side the transfer is
    the accounting half — allocate in the destination (skipping pages already
    cached there, the receiver-side dedup of design m6 D4) and mark computed.
    """

    def __init__(self, dest_kv: RadixKVCache) -> None:
        self._dest = dest_kv

    def transfer(
        self, tokens: tuple[int, ...], first_token: int, pages: tuple[int, ...] = ()
    ) -> KVAllocation:
        try:
            allocation = self._dest.allocate(tuple(tokens))
        except KVCacheFull as error:
            raise KVHandoffError(f"destination cache full: {error}") from error
        self._dest.mark_computed(allocation)
        return allocation


class PDCoordinator:
    """Drives a prefill Scheduler and a decode Scheduler as one P-D engine."""

    def __init__(
        self,
        *,
        prefill_scheduler: Scheduler,
        prefill_runner: ModelRunner,
        decode_scheduler: Scheduler,
        decode_runner: ModelRunner,
        handoff: KVHandoff,
        max_transfer_retries: int = 1,
    ) -> None:
        if max_transfer_retries < 0:
            raise ValueError(f"max_transfer_retries must be >= 0, got {max_transfer_retries}")
        self._prefill = prefill_scheduler
        self._prefill_runner = prefill_runner
        self._decode = decode_scheduler
        self._decode_runner = decode_runner
        self._handoff = handoff
        self._max_retries = max_transfer_retries
        self._pending: dict[str, tuple[EngineRequest, int]] = {}
        self._outputs: dict[str, tuple[int, ...]] = {}
        self._failed: list[str] = []
        # public request id -> the prefill clone id currently standing for it.
        # A long-lived driver (the serving loop) needs this both ways: to find
        # the request while it is still prefill-side, and to reclaim the clone's
        # scheduler/runner state when the request is forgotten.
        self._internal: dict[str, str] = {}

    @property
    def decode_scheduler(self) -> Scheduler:
        """The scheduler whose state the serving loop observes.

        Requests enter at prefill and finish under decode, so decode's is the one
        that answers "is this request done" for `EngineLoop`.
        """
        return self._decode

    @property
    def prefill_scheduler(self) -> Scheduler:
        """The scheduler submissions actually enter."""
        return self._prefill

    @property
    def decode_runner(self) -> ModelRunner:
        """The runner that executes what ``decode_scheduler`` plans."""
        return self._decode_runner

    @property
    def decode_cache(self) -> RadixKVCache:
        """The cache the served request's KV ends up in."""
        return self._decode.kv_cache

    @property
    def internal_ids(self) -> Mapping[str, str]:
        """Live read view of public request id -> current prefill clone id."""
        return MappingProxyType(self._internal)

    def internal_id_for(self, request_id: str) -> str | None:
        return self._internal.get(request_id)

    @property
    def failed_requests(self) -> tuple[str, ...]:
        """Requests dropped after exhausting transfer retries."""
        return tuple(self._failed)

    def add_request(self, request: EngineRequest) -> None:
        self._enqueue(request, attempt=0)

    def _enqueue(self, request: EngineRequest, attempt: int) -> None:
        internal_id = f"{request.request_id}{_PREFILL_ID_SEPARATOR}{attempt}"
        clone = replace(request, request_id=internal_id, max_new_tokens=1)
        self._pending[internal_id] = (request, attempt)
        self._internal[request.request_id] = internal_id
        self._prefill.add_request(clone)

    def _discard_clone(self, internal_id: str) -> None:
        """Reclaim a prefill clone's scheduler and runner state (E2)."""
        self._pending.pop(internal_id, None)
        self._prefill.forget(internal_id)
        release = getattr(self._prefill_runner, "release", None)
        if release is not None:
            release(internal_id)

    def abort(self, request_id: str) -> None:
        """Cancel a request wherever it currently is (client disconnect).

        The public id is only ever in ONE half: prefill under its clone id until
        the handoff commits, decode after it.
        """
        internal_id = self._internal.get(request_id)
        if internal_id is not None:
            self._pending.pop(internal_id, None)
            self._prefill.abort(internal_id)
        self._decode.abort(request_id)

    def forget(self, request_id: str) -> None:
        """Drop every trace of a finished request across both halves (E2)."""
        internal_id = self._internal.pop(request_id, None)
        if internal_id is not None:
            self._discard_clone(internal_id)
        self._decode.forget(request_id)
        self._outputs.pop(request_id, None)

    def _handoff_or_retry(self, internal_id: str, token0: int) -> bool:
        """Transfer one prompt's KV; return True if the commit may proceed."""
        original, attempt = self._pending.pop(internal_id)
        state = self._prefill.states.get(internal_id)
        source_pages: tuple[int, ...] = ()
        if state is not None and state.allocation is not None:
            source_pages = tuple(state.allocation.pages)
        try:
            allocation = self._handoff.transfer(
                original.prompt_token_ids, token0, source_pages
            )
        except KVHandoffError:
            # copy failed before commit: the prefill-side KV is released
            # un-marked and the request recomputes from scratch (design m6 D4)
            self._prefill.abort(internal_id)
            if attempt < self._max_retries:
                # the superseded clone's state is dead weight once its successor
                # is queued; the surviving one stays visible so a driver can see
                # a request that exhausted its retries finish as "abort"
                self._discard_clone(internal_id)
                self._enqueue(original, attempt + 1)
            else:
                self._failed.append(original.request_id)
            return False
        finished = self._decode.resume_with_kv(original, allocation, token0)
        if finished:
            self._outputs[original.request_id] = self._decode.output_tokens(
                original.request_id
            )
        return True

    def step_prefill(self, *, reject_on_stall: bool = False) -> None:
        """One prefill step: schedule, execute, hand the KV off, then commit.

        ``reject_on_stall`` is the engine-loop backstop (``EngineLoop.step``
        does the same for its own scheduler): an empty plan while requests are
        unfinished means nothing is running to free pages, so the waiting head
        is finished rather than the whole engine — and every concurrent request
        with it — dying on a stall.
        """
        plan = self._prefill.schedule()
        if not plan.scheduled:
            if not self._prefill.has_unfinished():
                return
            if reject_on_stall and self._prefill.reject_waiting_head() is not None:
                return
            raise RuntimeError("P-D prefill stall: nothing schedulable")
        sampled = self._prefill_runner.execute(plan.scheduled, self._prefill.states)
        # explicit SampledToken -> int unwrap (m8 D2): KVHandoff.transfer and
        # resume_with_kv keep their int-typed first_token contracts
        commit = {
            internal_id: token0
            for internal_id, tokens in sampled.items()
            if self._handoff_or_retry(internal_id, token0 := tokens[0].token_id)
        }
        if commit:
            self._prefill.update(commit)

    def _step_decode(self) -> None:
        plan = self._decode.schedule()
        if not plan.scheduled:
            if self._decode.has_unfinished():
                raise RuntimeError("P-D decode stall: nothing schedulable")
            return
        sampled = self._decode_runner.execute(plan.scheduled, self._decode.states)
        finished = self._decode.update(token_ids(sampled)) if sampled else ()
        for request_id in finished:
            self._outputs[request_id] = self._decode.output_tokens(request_id)

    def run_to_completion(self) -> dict[str, tuple[int, ...]]:
        while self._prefill.has_unfinished() or self._decode.has_unfinished():
            if self._prefill.has_unfinished():
                self.step_prefill()
            if self._decode.has_unfinished():
                self._step_decode()
        return dict(self._outputs)
