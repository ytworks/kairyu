"""P-D disaggregation coordinator: copy-on-handoff between two cores (design m5 D5).

The prefill core runs each request with ``max_new_tokens=1`` so no decode chunk
is ever scheduled there; the coordinator intercepts at execute-completion —
after the KV is written, before ``update()`` commits — transfers the prompt KV
plus token 0 to the decode core, and only then commits (copy-before-commit:
``commit_and_release`` must not pool-free the tail page under the copy). The
decode core adopts via ``Scheduler.resume_with_kv``.

Three handoffs sit behind the ``KVHandoff`` seam: ``LocalCopyKVHandoff`` (same
process, real page bytes — what a deployment gets, on host or device),
``RemoteKVHandoff`` (m18, over a transport), and ``LocalKVHandoff``, which does
the accounting and nothing else and is a test double, not a deployment option.

A handoff that DEFERS (m18 D3, ``StreamCopyKVHandoff(defer=True)``) returns while
its copy is still running. The coordinator then holds the prefill-side lease
itself: no commit, no abort, no release, no decode-side adoption happens until
``_settle_handover`` has gated on the copy's completion event. Without that, the
released source page is re-allocated by the next prefill step and overwritten on
the caller's stream while the side stream is still reading it.

That settlement is deliberately PIPELINED one producer step. A gate placed at the
end of the step that started the copy is a fence in front of every later kernel:
the copy is still the only thing on the device, so nothing overlaps it. Holding
the handover until the NEXT prefill step's forward has been queued is what makes
the overlap real — the copy runs against that forward (and against the decode
step in between), and only then is the caller's stream ordered behind it. The
source pages stay leased for that whole window, so the extra step costs prefill
capacity, never correctness.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from kairyu.engine.core.engine_core import ModelRunner, token_ids
from kairyu.engine.core.radix_kv import KVAllocation, KVCacheFull, RadixKVCache
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import EngineRequest, Scheduler

if TYPE_CHECKING:  # pragma: no cover
    from kairyu.engine.core.kv_pool import PagedKVPool

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
    """Accounting-only handoff: adopt the prompt's pages in the destination cache.

    NOT a production handoff. It allocates in the destination (skipping pages
    already cached there, the receiver-side dedup of design m6 D4) and marks
    computed WITHOUT touching either pool, so it is correct only when both
    halves index the same ``PagedKVPool`` — which a real prefill/decode pair
    never does. Deployments build ``LocalCopyKVHandoff`` (same process, real
    bytes) or ``RemoteKVHandoff`` (over a transport); this one exists so the
    protocol tests can drive ordering with a runner that has no pool at all.
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


def _geometry(pool: PagedKVPool) -> tuple:
    """Everything a page copy has to agree on; device deliberately excluded."""
    return (
        pool.num_layers,
        pool.page_size,
        pool.num_kv_heads,
        pool.head_dim,
        pool.v_head_dim,
        pool.k.dtype,
    )


class LocalCopyKVHandoff:
    """Same-process handoff that moves the KV BYTES, not just the accounting.

    The failure this exists to prevent: with ``LocalKVHandoff`` between two
    engines that own separate pools, ``transfer()`` publishes the destination
    allocation as *computed* while its pages still hold whatever was in them —
    zeros on a fresh pool. Decode then continues from KV that was never
    written, and the radix tree hands that wrong prefix to every later request
    that matches it. Nothing raises; the tokens are simply wrong.

    Same shape as ``RemoteKVReceiver`` minus the wire: allocate in the
    destination, skip the leading ``len(cached_pages)`` source pages (radix
    matches are prefix-only, so receiver-side dedup skips the COPY, not the
    pages — design m6 D4), copy the rest page-for-page, and only then mark
    computed. A failed copy releases the allocation instead of publishing a
    half-written prefix.

    The copy is a direct pool-to-pool tensor copy rather than the serde
    round-trip ``RemoteKVHandoff`` uses: within one process there is no wire, so
    a device pair stays device-to-device instead of paying D2H + H2D. That copy
    is exactly the work ``StreamCopyKVHandoff`` puts on a side stream.
    """

    def __init__(
        self, dest_kv: RadixKVCache, source_pool: PagedKVPool, dest_pool: PagedKVPool
    ) -> None:
        if _geometry(source_pool) != _geometry(dest_pool):
            raise ValueError(
                "prefill and decode pools must have the same geometry: "
                f"{_geometry(source_pool)} != {_geometry(dest_pool)}"
            )
        if dest_pool.page_size != dest_kv.page_size:
            raise ValueError(
                f"pool page_size {dest_pool.page_size} != cache page_size "
                f"{dest_kv.page_size}"
            )
        self._dest = dest_kv
        self._source_pool = source_pool
        self._dest_pool = dest_pool

    def transfer(
        self, tokens: tuple[int, ...], first_token: int, pages: tuple[int, ...] = ()
    ) -> KVAllocation:
        tokens = tuple(tokens)
        if not tokens:
            raise KVHandoffError("local copy handoff requires a non-empty prompt")
        page_size = self._dest_pool.page_size
        expected = -(-len(tokens) // page_size)
        if len(pages) != expected:
            raise KVHandoffError(
                f"expected {expected} source pages for {len(tokens)} tokens, "
                f"got {len(pages)} — the source allocation is not this prompt's"
            )
        try:
            allocation = self._dest.allocate(tokens)
        except KVCacheFull as error:
            raise KVHandoffError(f"destination cache full: {error}") from error
        try:
            targets = tuple(
                page
                for page in (*allocation.new_full_pages, allocation.tail_page)
                if page is not None
            )
            incoming = tuple(pages)[len(allocation.cached_pages) :]
            if len(incoming) != len(targets):
                raise KVHandoffError(
                    f"{len(incoming)} source pages for {len(targets)} destination slots"
                )
            for source_page, dest_page in zip(incoming, targets, strict=True):
                self._copy_page(source_page, dest_page)
            self._dest.mark_computed(allocation)
            return allocation
        except Exception:
            self._dest.release_preempted(allocation)
            raise

    def _copy_page(self, source_page: int, dest_page: int) -> None:
        """All layers of one logical page; ``k[:, page]`` is a strided view."""
        self._dest_pool.k[:, dest_page].copy_(self._source_pool.k[:, source_page])
        if self._dest_pool.v_head_dim:  # MLA keeps the latent in k (m15 A7)
            self._dest_pool.v[:, dest_page].copy_(self._source_pool.v[:, source_page])


@dataclass
class _Handover:
    """One prefill step's transfers, held until their copies have been settled.

    Everything here is an action the m6 D4 ordering rule forbids until the copy
    has completed: ``commit``/``retries`` hand the SOURCE pages back, ``adopt``
    lets the decode core READ the destination pages. Keeping them together is
    what lets the whole set be deferred past the next producer step.
    """

    commit: dict[str, int] = field(default_factory=dict)
    adopt: list[tuple[EngineRequest, KVAllocation, SampledToken]] = field(default_factory=list)
    retries: list[tuple[str, EngineRequest, int]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.commit or self.adopt or self.retries)


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
        # A deferring handoff (m18 D3) returns while the copy is still reading the
        # prefill-side pages, so this coordinator owns the lease on them until the
        # copy's completion event says otherwise. Refuse one that cannot be gated
        # rather than releasing pages under a copy we have no way to order against.
        self._gate_pending = None
        if getattr(handoff, "defers", False):
            gate = getattr(handoff, "gate_pending", None)
            if gate is None:
                raise ValueError(
                    "a deferring KVHandoff must expose gate_pending(); without it "
                    "the prefill-side pages would be released under a running copy"
                )
            self._gate_pending = gate
        # the one prefill step's worth of transfers whose copies are still in
        # flight; None with a blocking handoff, which settles inside its own step
        self._handover: _Handover | None = None
        # token 0s adopted since the driver last drained them: committed by
        # resume_with_kv, so no runner ever reports them
        self._carried: dict[str, SampledToken] = {}

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
        # request_id is the prefill scheduler's bookkeeping name; sampling_id
        # keeps the PUBLIC id as the sampling identity, so token 0 comes off the
        # same RNG stream (and the same grammar state) as token 1 onward
        clone = replace(
            request,
            request_id=internal_id,
            max_new_tokens=1,
            sampling_id=request.request_id,
        )
        self._pending[internal_id] = (request, attempt)
        self._internal[request.request_id] = internal_id
        self._prefill.add_request(clone)

    def _release_sampling_state(self, public_id: str) -> None:
        """Drop the prefill half's sampler state for a request (E2).

        Separate from the clone id: the clone SAMPLES under the public id (so
        token 0 shares token 1's seed and grammar state), so releasing the
        scheduler-side clone id alone would leak the seed and the enforcer.
        """
        release = getattr(self._prefill_runner, "release", None)
        if release is not None:
            release(public_id)

    def _discard_clone(self, internal_id: str, public_id: str | None = None) -> None:
        """Reclaim a prefill clone's scheduler and runner state (E2)."""
        self._pending.pop(internal_id, None)
        self._prefill.forget(internal_id)
        release = getattr(self._prefill_runner, "release", None)
        if release is not None:
            release(internal_id)
        if public_id is not None:
            self._release_sampling_state(public_id)

    def _settle_if_held(self, request_id: str) -> None:
        """Land an outstanding handover before touching a request it carries.

        A deferred settlement spans a step boundary, which is exactly where the
        serving loop drains abort/forget ops. Cancelling a request whose copy is
        still in flight would otherwise leave the settlement to adopt it into
        decode afterwards — resurrecting an aborted request, and leaking the
        destination allocation the transfer already made. Settling first costs
        this one request its overlap and keeps every path single-meaning.
        """
        if self._handover is None:
            return
        internal_id = self._internal.get(request_id)
        held = any(original.request_id == request_id for original, _, _ in self._handover.adopt)
        held = held or any(held_id == internal_id for held_id, _, _ in self._handover.retries)
        if held:
            self._settle_handover()

    def abort(self, request_id: str) -> None:
        """Cancel a request wherever it currently is (client disconnect).

        The public id is only ever in ONE half: prefill under its clone id until
        the handoff commits, decode after it.
        """
        self._settle_if_held(request_id)
        internal_id = self._internal.get(request_id)
        if internal_id is not None:
            self._pending.pop(internal_id, None)
            self._prefill.abort(internal_id)
        self._decode.abort(request_id)

    def forget(self, request_id: str) -> None:
        """Drop every trace of a finished request across both halves (E2)."""
        self._settle_if_held(request_id)
        internal_id = self._internal.pop(request_id, None)
        if internal_id is not None:
            self._discard_clone(internal_id, request_id)
        self._decode.forget(request_id)
        self._outputs.pop(request_id, None)
        self._carried.pop(request_id, None)

    def _handoff_or_retry(
        self, internal_id: str, token0: SampledToken, handover: _Handover
    ) -> None:
        """Start one prompt's KV transfer; record what may happen once it lands.

        Nothing is acted on here. Success owes the decode core an adoption and
        the prefill core a commit; failure owes it an abort — and all three are
        forbidden until the copy has completed, because the abort frees the
        prefill-side pages the copy is still reading (a transfer that raised may
        already have queued part of it) and the adoption lets decode read the
        destination pages. ``_settle_handover`` is the single place that gates
        first and performs them second.
        """
        original, attempt = self._pending.pop(internal_id)
        state = self._prefill.states.get(internal_id)
        source_pages: tuple[int, ...] = ()
        if state is not None and state.allocation is not None:
            source_pages = tuple(state.allocation.pages)
        try:
            allocation = self._handoff.transfer(
                # explicit SampledToken -> int unwrap (m8 D2): KVHandoff.transfer
                # keeps its int-typed first_token contract
                original.prompt_token_ids, token0.token_id, source_pages
            )
        except KVHandoffError:
            handover.retries.append((internal_id, original, attempt))
            return
        handover.commit[internal_id] = token0.token_id
        handover.adopt.append((original, allocation, token0))

    def _settle_handover(self) -> None:
        """Complete one step's transfers — never before their copies have landed.

        Every release exits through here: ``update()`` finishes the
        max_new_tokens=1 clone and ``commit_and_release`` pool-frees its tail
        page, ``abort()`` release-preempts the whole allocation. Either way the
        pages return to the prefill pool and the next prefill step can allocate
        and overwrite them on the caller's stream. Adoption is gated for the
        mirror-image reason: it is what puts the destination pages in front of
        the decode runner.

        With the blocking handoff that is free — ``transfer()`` did not return
        until the copy finished, so the caller settles in its own step. With a
        deferring one the gate is stream-ordered (``event.wait(current_stream)``)
        rather than a host block, and the CALLER decides when to apply it. Doing
        it one producer step later is what buys the overlap: everything queued in
        between — the decode step, then the next prefill forward — runs against
        the copy instead of behind it (m6 D4 under m18 D3's defer).
        """
        handover, self._handover = self._handover, None
        if handover is None:
            return
        if self._gate_pending is not None:
            self._gate_pending()
        for internal_id, original, attempt in handover.retries:
            # copy failed before commit: the prefill-side KV is released
            # un-marked and the request recomputes from scratch (design m6 D4)
            self._prefill.abort(internal_id)
            if attempt < self._max_retries:
                # the superseded clone's state is dead weight once its successor
                # is queued; the surviving one stays visible so a driver can see
                # a request that exhausted its retries finish as "abort".
                # The sampling state goes with it: the retry re-samples token 0
                # at position 0 from a state built under the same public id, so
                # a fresh one is identical to the one being dropped.
                self._discard_clone(internal_id, original.request_id)
                self._enqueue(original, attempt + 1)
            else:
                # the clone itself stays visible so a driver can see the request
                # finish as "abort"; only its sampling state is dead
                self._release_sampling_state(original.request_id)
                self._failed.append(original.request_id)
        if handover.commit:
            self._prefill.update(handover.commit)
        for original, allocation, token0 in handover.adopt:
            self._carry_sampler_state(original.request_id)
            # token 0's FULL metadata, not just its id: it is the first token of
            # the completion, so its logprob/top_logprobs belong in the stream
            # and its grammar_terminated flag can finish the request outright
            self._carried[original.request_id] = token0
            finished = self._decode.resume_with_kv(original, allocation, token0.token_id)
            if not finished and token0.grammar_terminated:
                # the grammar completed AT token 0 (m8 D1). The decode half never
                # sampled that token, so nothing there would ever notice — and it
                # has to be noticed BEFORE decode plans, or the request generates
                # one token past a finished grammar.
                self._decode.finish_early(original.request_id)
                finished = True
            if finished:
                self._outputs[original.request_id] = self._decode.output_tokens(
                    original.request_id
                )

    def _carry_sampler_state(self, request_id: str) -> None:
        """Move the request's sampling state from the prefill half to decode.

        Both halves key it under the PUBLIC id now, so the base seed already
        agrees — but the grammar enforcer does not travel with a seed. Left
        behind, the decode half builds a matcher that has never accepted token 0
        and masks token 1 against the wrong grammar position (m8 D2 under m5 D5).
        """
        source = getattr(self._prefill_runner, "sampler", None)
        destination = getattr(self._decode_runner, "sampler", None)
        if source is None or destination is None or source is destination:
            return
        source.hand_over(request_id, destination)

    def drain_carried_tokens(self) -> dict[str, SampledToken]:
        """Token 0s committed by the ADOPTION rather than by a runner this step.

        ``resume_with_kv`` appends token 0 to the decode request's outputs
        directly, so it never appears in any ``execute()`` return — a driver
        reading only runner output loses its logprobs entirely, and with
        ``max_tokens=1`` there is no decode step at all to lose them in.
        """
        carried, self._carried = self._carried, {}
        return carried

    def step_prefill(self, *, reject_on_stall: bool = False) -> None:
        """One prefill step: schedule, execute, hand the KV off, then commit.

        ``reject_on_stall`` is the engine-loop backstop (``EngineLoop.step``
        does the same for its own scheduler): an empty plan while requests are
        unfinished means nothing is running to free pages, so the waiting head
        is finished rather than the whole engine — and every concurrent request
        with it — dying on a stall.
        """
        plan = self._prefill.schedule()
        if not plan.scheduled and self._handover is not None:
            # nothing to queue alongside the outstanding copies — and their still
            # -leased source pages are part of why. There is no overlap left to
            # win, so settle and re-plan with the pages back.
            self._settle_handover()
            plan = self._prefill.schedule()
        if not plan.scheduled:
            if not self._prefill.has_unfinished():
                return
            if reject_on_stall and self._prefill.reject_waiting_head() is not None:
                return
            raise RuntimeError("P-D prefill stall: nothing schedulable")
        sampled = self._prefill_runner.execute(plan.scheduled, self._prefill.states)
        # The gate for the PREVIOUS step's copies lands here, with this step's
        # forward already queued in front of it — that forward, and the decode
        # step queued before it, are what those copies overlap.
        self._settle_handover()
        handover = _Handover()
        for internal_id, tokens in sampled.items():
            self._handoff_or_retry(internal_id, tokens[0], handover)
        self._handover = handover or None
        if self._gate_pending is None:
            # a blocking handoff already finished its copy inside transfer();
            # deferring the settlement would only delay the request for nothing
            self._settle_handover()

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

    def has_prefill_work(self) -> bool:
        """Prefill-side work, INCLUDING a handover whose copies are still in
        flight — its clones stay unfinished until the settlement commits them,
        but saying so explicitly keeps a driver from ever dropping one."""
        return self._prefill.has_unfinished() or self._handover is not None

    def run_to_completion(self) -> dict[str, tuple[int, ...]]:
        while self.has_prefill_work() or self._decode.has_unfinished():
            if self.has_prefill_work():
                self.step_prefill()
            if self._decode.has_unfinished():
                self._step_decode()
        return dict(self._outputs)
