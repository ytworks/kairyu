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
``_settle_handover`` polls the completion event as physically done and finalizes
publication. A stream wait alone is not completion. Without the retained lease,
the released source page could be re-allocated and overwritten while the copy
still reads it.

That settlement is deliberately PIPELINED one producer step. The next eligible
prefill/decode forward is queued while the completion remains pending; only a
successful later poll admits decode and releases source ownership. Transient
failures retain re-enterable state. Lost completion or cleanup ownership poisons
the handover and fails closed, so partial KV is never published or recycled.
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

    def __init__(
        self,
        message: str,
        *,
        allocation: KVAllocation | None = None,
    ) -> None:
        super().__init__(message)
        self.allocation = allocation
        self.handoff_completion: object | None = None


class HandoffCompletionLostError(RuntimeError):
    """Safe completion or cleanup cannot be proven; retain KV ownership."""

    handoff_completion_lost = True

    def __init__(self, message: str, *, allocation: object | None) -> None:
        super().__init__(message)
        self.allocation = allocation


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

    def discard(self, allocation: KVAllocation) -> None:
        """Release a destination allocation that decode could not adopt."""
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

    def discard(self, allocation: KVAllocation) -> None:
        """Release the destination lease when coordinator adoption fails."""
        self._dest.release_preempted(allocation)


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
        self,
        dest_kv: RadixKVCache,
        source_pool: PagedKVPool,
        dest_pool: PagedKVPool,
        *,
        defer_publish: bool = False,
    ) -> None:
        if _geometry(source_pool) != _geometry(dest_pool):
            raise ValueError(
                "prefill and decode pools must have the same geometry: "
                f"{_geometry(source_pool)} != {_geometry(dest_pool)}"
            )
        if dest_pool.page_size != dest_kv.page_size:
            raise ValueError(
                f"pool page_size {dest_pool.page_size} != cache page_size {dest_kv.page_size}"
            )
        self._dest = dest_kv
        self._source_pool = source_pool
        self._dest_pool = dest_pool
        self._defer_publish = defer_publish

    def manage_completion_externally(self) -> None:
        """Let a stream wrapper own publication and failed-copy cleanup."""
        self._defer_publish = True

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
            self._copy_pages(incoming, targets)
            if not self._defer_publish:
                # A bare handoff has no event owner.  Device copies can already
                # be queued when a later operation raises, so physical
                # completion must precede publication or destination reuse.
                self._synchronize_copy_devices()
                self._dest.mark_computed(allocation)
            return allocation
        except Exception as error:
            if self._defer_publish:
                if isinstance(error, KVHandoffError):
                    error.allocation = allocation
                    raise
                raise KVHandoffError(
                    f"destination KV copy failed: {type(error).__name__}",
                    allocation=allocation,
                ) from error
            self._settle_and_discard(
                allocation,
                "failed local KV copy could not prove completion or release "
                "its destination allocation",
            )
            if isinstance(error, KVHandoffError):
                raise
            raise KVHandoffError(f"destination KV copy failed: {type(error).__name__}") from error
        except BaseException as interruption:
            # StreamCopyKVHandoff must be able to reclaim an allocation created
            # before cancellation/termination interrupted the copy. Preserve
            # the original BaseException while attaching that ownership.
            try:
                interruption.allocation = allocation  # type: ignore[attr-defined]
            except Exception:
                pass
            if not self._defer_publish:
                self._settle_and_discard(
                    allocation,
                    "interrupted local KV copy could not prove completion or "
                    "release its destination allocation",
                )
            raise

    def publish(self, allocation: KVAllocation) -> None:
        """Expose copied pages only after their physical completion."""
        try:
            self._dest.mark_computed(allocation)
        except BaseException as error:
            if not self._defer_publish:
                try:
                    self._dest.release_preempted(allocation)
                except BaseException as cleanup_error:
                    if not allocation._freed:
                        raise HandoffCompletionLostError(
                            "failed local KV publication could not release its "
                            "destination allocation",
                            allocation=allocation,
                        ) from cleanup_error
            if not isinstance(error, Exception):
                try:
                    error.allocation = allocation  # type: ignore[attr-defined]
                except Exception:
                    pass
                raise
            raise KVHandoffError(f"destination cache publication failed: {error}") from error

    def discard(self, allocation: KVAllocation) -> None:
        """Release a failed transfer after all queued writes have completed."""
        self._dest.release_preempted(allocation)

    def _copy_page(self, source_page: int, dest_page: int) -> None:
        """All layers of one logical page; ``k[:, page]`` is a strided view."""
        self._dest_pool.k[:, dest_page].copy_(self._source_pool.k[:, source_page])
        if self._dest_pool.v_head_dim:  # MLA keeps the latent in k (m15 A7)
            self._dest_pool.v[:, dest_page].copy_(self._source_pool.v[:, source_page])

    def _synchronize_copy_devices(self) -> None:
        """Prove every possibly queued device copy is physically complete."""
        source_device = self._source_pool.k.device
        dest_device = self._dest_pool.k.device
        cuda_devices = tuple(
            device
            for device in dict.fromkeys((source_device, dest_device))
            if device.type == "cuda"
        )
        if not cuda_devices:
            return
        import torch

        for device in cuda_devices:
            torch.cuda.synchronize(device)

    def _settle_and_discard(
        self,
        allocation: KVAllocation,
        message: str,
    ) -> None:
        """Make a bare-copy failure safe, or retain its only known owner."""
        try:
            self._synchronize_copy_devices()
        except BaseException as sync_error:
            raise HandoffCompletionLostError(
                message,
                allocation=allocation,
            ) from sync_error
        try:
            self._dest.release_preempted(allocation)
        except BaseException as cleanup_error:
            if not allocation._freed:
                raise HandoffCompletionLostError(
                    message,
                    allocation=allocation,
                ) from cleanup_error

    def _copy_pages(self, source_pages: tuple[int, ...], dest_pages: tuple[int, ...]) -> None:
        source_device = self._source_pool.k.device
        dest_device = self._dest_pool.k.device
        if source_device == dest_device:
            for source_page, dest_page in zip(source_pages, dest_pages, strict=True):
                self._copy_page(source_page, dest_page)
            return

        # ``pool[:, page]`` is strided across layers.  PyTorch's cross-device
        # copy of that view synchronizes the host (measured on the production
        # layout), defeating the completion event before it is even recorded.
        # A layer's consecutive page slice is contiguous, so coalesce matching
        # source/destination runs and enqueue truly asynchronous P2P copies.
        if not self._defer_publish and source_device.type == "cuda":
            # The side-stream wrapper normally orders this with a source event.
            # Its explicitly declined/bare blocking control has no such event,
            # so wait for the producer before reading across devices.
            import torch

            torch.cuda.synchronize(source_device)
        runs: list[tuple[int, int, int]] = []
        for source_page, dest_page in zip(source_pages, dest_pages, strict=True):
            if (
                runs
                and source_page == runs[-1][0] + runs[-1][2]
                and dest_page == runs[-1][1] + runs[-1][2]
            ):
                source_start, dest_start, length = runs[-1]
                runs[-1] = (source_start, dest_start, length + 1)
            else:
                runs.append((source_page, dest_page, 1))
        for layer in range(self._source_pool.num_layers):
            for source_start, dest_start, length in runs:
                self._dest_pool.k[layer, dest_start : dest_start + length].copy_(
                    self._source_pool.k[layer, source_start : source_start + length],
                    non_blocking=True,
                )
                if self._dest_pool.v_head_dim:
                    self._dest_pool.v[layer, dest_start : dest_start + length].copy_(
                        self._source_pool.v[layer, source_start : source_start + length],
                        non_blocking=True,
                    )


@dataclass
class _Handover:
    """One prefill step's transfers, held until their copies have been settled.

    Everything here is an action the m6 D4 ordering rule forbids until the copy
    has completed: ``commit``/``retries`` hand the SOURCE pages back, ``adopt``
    lets the decode core READ the destination pages. Keeping them together is
    what lets the whole set be deferred past the next producer step.
    """

    commit: dict[str, int] = field(default_factory=dict)
    adopt: list[tuple[str, EngineRequest, KVAllocation, SampledToken, int, object | None]] = field(
        default_factory=list
    )
    retries: list[tuple[str, EngineRequest, int, KVAllocation | None, object | None]] = field(
        default_factory=list
    )
    resumed: set[str] = field(default_factory=set)
    adopted: set[str] = field(default_factory=set)
    discarded: set[str] = field(default_factory=set)
    cancelled: set[str] = field(default_factory=set)
    poisoned: dict[str, BaseException] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.commit or self.adopt or self.retries or self.poisoned)


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
        # copy's completion event says otherwise. A stream wait is not completion:
        # host-visible cache publication and page release require a successful
        # event query, as does surfacing an asynchronous transfer failure.
        self._completion_for = None
        self._completion_ready = None
        self._finalize_completion = None
        self._acknowledge_completion = None
        self._wait_completion = None
        if getattr(handoff, "defers", False):
            completion_for = getattr(handoff, "completion_for", None)
            completion_ready = getattr(handoff, "completion_ready", None)
            finalize_completion = getattr(handoff, "finalize_completion", None)
            acknowledge_completion = getattr(
                handoff,
                "acknowledge_completion",
                None,
            )
            wait_completion = getattr(handoff, "wait_completion", None)
            if any(
                item is None
                for item in (
                    completion_for,
                    completion_ready,
                    finalize_completion,
                    acknowledge_completion,
                    wait_completion,
                )
            ):
                raise ValueError(
                    "a deferring KVHandoff must expose completion polling and "
                    "finalization; stream waits alone cannot publish or release KV"
                )
            self._completion_for = completion_for
            self._completion_ready = completion_ready
            self._finalize_completion = finalize_completion
            self._acknowledge_completion = acknowledge_completion
            self._wait_completion = wait_completion
            self._prefill.enable_deferred_handoff_overlap()
        # the one prefill step's worth of transfers whose copies are still in
        # flight; None with a blocking handoff, which settles inside its own step
        self._handover: _Handover | None = None
        # If the overlap forward finishes before the previous copy does, retain
        # its result rather than blocking the host or scheduling its clones twice.
        self._staged_sampled: dict[str, tuple[SampledToken, ...]] | None = None
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
        if (
            request.request_id in self._internal
            or request.request_id in self._decode.states
        ):
            raise ValueError(f"duplicate request_id {request.request_id!r}")
        self._enqueue(request, attempt=0)

    def _enqueue(
        self,
        request: EngineRequest,
        attempt: int,
        *,
        allow_existing: bool | None = None,
    ) -> None:
        if allow_existing is None:
            allow_existing = attempt > 0
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
        existing = self._prefill.states.get(internal_id)
        if existing is not None:
            if (
                allow_existing
                and existing.request == clone
                and self._pending.get(internal_id) == (request, attempt)
                and self._internal.get(request.request_id) == internal_id
            ):
                return
            raise ValueError(f"duplicate request_id {internal_id!r}")
        try:
            self._prefill.add_request(clone)
        except BaseException:
            # A wrapper may raise after Scheduler.add_request installed the
            # clone. Reconcile that exact state, but never overwrite unrelated
            # bookkeeping on a pre-mutation failure.
            installed = self._prefill.states.get(internal_id)
            if installed is not None and installed.request == clone:
                self._pending[internal_id] = (request, attempt)
                self._internal[request.request_id] = internal_id
            raise
        self._pending[internal_id] = (request, attempt)
        self._internal[request.request_id] = internal_id

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
        held = any(
            original.request_id == request_id for _, original, _, _, _, _ in self._handover.adopt
        )
        held = held or any(held_id == internal_id for held_id, _, _, _, _ in self._handover.retries)
        held = held or (internal_id is not None and internal_id in self._handover.poisoned)
        if held:
            if internal_id is not None:
                self._handover.cancelled.add(internal_id)
            if internal_id is not None and internal_id in self._handover.poisoned:
                raise RuntimeError(
                    "cannot release a request whose handoff completion was lost"
                ) from self._handover.poisoned[internal_id]
            # Abort/forget are exceptional paths: wait only for this handover's
            # completion events so its pages can be reclaimed without a race.
            if self._wait_completion is not None:
                for completion in self._handover_completions(self._handover):
                    self._wait_completion(completion)
            if not self._settle_handover():  # pragma: no cover - wait made it ready
                raise RuntimeError("waited handoff did not become ready")

    @staticmethod
    def _handover_completions(handover: _Handover) -> tuple[object, ...]:
        completions = [completion for *_, completion in handover.adopt if completion is not None]
        completions.extend(
            completion for *_, completion in handover.retries if completion is not None
        )
        return tuple(completions)

    def _discard_or_reconcile(
        self,
        handover: _Handover,
        internal_id: str,
        allocation: KVAllocation,
    ) -> BaseException | None:
        """Release destination ownership and detect success-then-raise hooks.

        ``discard`` is intentionally non-idempotent: calling it twice can free
        pages now owned by another request.  A production ``KVAllocation``
        records terminal release, so a raised hook can be reconciled only when
        that flag proves the mutation completed.  Anything else poisons the
        handover and fails closed.
        """
        try:
            self._handoff.discard(allocation)
        except BaseException as cleanup_error:
            if allocation._freed:
                return cleanup_error
            handover.poisoned[internal_id] = cleanup_error
            raise
        return None

    def _drop_staged(self, internal_id: str | None) -> None:
        if internal_id is not None and self._staged_sampled is not None:
            self._staged_sampled.pop(internal_id, None)

    def abort(self, request_id: str) -> None:
        """Cancel a request wherever it currently is (client disconnect).

        The public id is only ever in ONE half: prefill under its clone id until
        the handoff commits, decode after it.
        """
        self._settle_if_held(request_id)
        internal_id = self._internal.get(request_id)
        self._drop_staged(internal_id)
        if internal_id is not None:
            self._pending.pop(internal_id, None)
            self._prefill.abort(internal_id)
        self._release_sampling_state(request_id)
        self._decode.abort(request_id)

    def forget(self, request_id: str) -> None:
        """Drop every trace of a finished request across both halves (E2)."""
        self._settle_if_held(request_id)
        internal_id = self._internal.pop(request_id, None)
        self._drop_staged(internal_id)
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
        original, attempt = self._pending[internal_id]
        state = self._prefill.states.get(internal_id)
        source_pages: tuple[int, ...] = ()
        if state is not None and state.allocation is not None:
            source_pages = tuple(state.allocation.pages)
        try:
            allocation = self._handoff.transfer(
                # explicit SampledToken -> int unwrap (m8 D2): KVHandoff.transfer
                # keeps its int-typed first_token contract
                original.prompt_token_ids,
                token0.token_id,
                source_pages,
            )
        except BaseException as error:
            completion = getattr(error, "handoff_completion", None)
            if getattr(error, "handoff_completion_lost", False):
                handover.poisoned[internal_id] = error
                raise
            retryable = isinstance(error, KVHandoffError)
            if retryable or completion is not None:
                self._pending.pop(internal_id)
                handover.retries.append(
                    (
                        internal_id,
                        original,
                        attempt,
                        getattr(error, "allocation", None),
                        completion,
                    )
                )
                if isinstance(error, Exception):
                    return
            raise
        try:
            completion = (
                self._completion_for(allocation) if self._completion_for is not None else None
            )
        except BaseException as error:
            handover.poisoned[internal_id] = error
            raise
        self._pending.pop(internal_id)
        handover.commit[internal_id] = token0.token_id
        handover.adopt.append((internal_id, original, allocation, token0, attempt, completion))

    def _settle_handover(self) -> bool:
        """Poll and settle one step's transfers after physical completion.

        Every release exits through here: ``update()`` frees source ownership,
        retry aborts source KV, and ``resume_with_kv`` admits destination pages.
        A stream wait only orders future work on that stream; it does not prove
        completion or authorize host-visible radix publication.  The normal path
        therefore returns ``False`` while any completion query is pending.
        Unrelated decode and one next prefill forward can run during those polls.
        """
        handover = self._handover
        if handover is None:
            return True
        if handover.poisoned:
            internal_id, error = next(iter(handover.poisoned.items()))
            raise RuntimeError(
                f"handoff completion was lost for {internal_id}; "
                "the engine must fail closed"
            ) from error
        if self._completion_ready is not None:
            for completion in self._handover_completions(handover):
                if not self._completion_ready(completion):
                    return False

        index = 0
        while index < len(handover.adopt):
            (
                internal_id,
                original,
                allocation,
                token0,
                attempt,
                completion,
            ) = handover.adopt[index]
            try:
                if completion is not None and not self._finalize_completion(completion):
                    raise RuntimeError("ready handoff completion became incomplete")
            except KVHandoffError:
                handover.commit.pop(internal_id, None)
                handover.retries.append((internal_id, original, attempt, None, None))
                handover.adopt.pop(index)
                if completion is not None:
                    self._acknowledge_completion(completion)
            else:
                handover.adopt[index] = (
                    internal_id,
                    original,
                    allocation,
                    token0,
                    attempt,
                    None,
                )
                if completion is not None:
                    self._acknowledge_completion(completion)
                index += 1

        for index, (
            internal_id,
            original,
            attempt,
            _allocation,
            completion,
        ) in enumerate(tuple(handover.retries)):
            if completion is not None:
                try:
                    if not self._finalize_completion(completion):
                        raise RuntimeError("ready failed handoff became incomplete")
                except KVHandoffError:
                    # Expected transfer failure, surfaced only after partially
                    # queued writes completed and destination cleanup was safe.
                    pass
            handover.retries[index] = (internal_id, original, attempt, None, None)
            if completion is not None:
                self._acknowledge_completion(completion)

        # Activate and source-commit one adoption at a time. The handover stays
        # installed until each external mutation succeeds, so an exception can
        # resume without double-adopting an earlier request or losing its lock.
        index = 0
        while index < len(handover.adopt):
            (
                internal_id,
                original,
                allocation,
                token0,
                attempt,
                _completion,
            ) = handover.adopt[index]
            if internal_id in handover.cancelled:
                if internal_id in handover.resumed:
                    self._decode.abort(original.request_id)
                    # Token 0 was adopted internally but never surfaced to the
                    # EngineLoop. Remove that decode state completely so the
                    # abort result routes through the still-owned prefill clone
                    # and remains an empty completion.
                    self._decode.forget(original.request_id)
                    self._carried.pop(original.request_id, None)
                    self._outputs.pop(original.request_id, None)
                elif internal_id not in handover.discarded:
                    cleanup_error = self._discard_or_reconcile(
                        handover,
                        internal_id,
                        allocation,
                    )
                    handover.discarded.add(internal_id)
                    if cleanup_error is not None:
                        raise cleanup_error
                self._prefill.abort(internal_id)
                handover.commit.pop(internal_id, None)
                handover.resumed.discard(internal_id)
                handover.adopted.discard(internal_id)
                handover.discarded.discard(internal_id)
                handover.adopt.pop(index)
                continue

            if internal_id not in handover.resumed:
                try:
                    self._decode.resume_with_kv(original, allocation, token0.token_id)
                except BaseException as adoption_error:
                    # ``resume_with_kv`` installs the decode state before its
                    # terminal checks. An outer hook can therefore raise after
                    # adoption already succeeded. Detect that ownership
                    # transition before deciding whether discard is legal.
                    state = self._decode.states.get(original.request_id)
                    if state is not None and state.allocation is allocation:
                        handover.resumed.add(internal_id)
                        raise
                    if not isinstance(adoption_error, Exception):
                        raise
                    # Completion already made this allocation safe to release.
                    # Keep source ownership and turn the request into the normal
                    # recompute retry path instead of leaking a destination lock.
                    cleanup_error = self._discard_or_reconcile(
                        handover,
                        internal_id,
                        allocation,
                    )
                    handover.commit.pop(internal_id, None)
                    handover.retries.append((internal_id, original, attempt, None, None))
                    handover.adopt.pop(index)
                    if cleanup_error is not None:
                        raise cleanup_error from adoption_error
                    continue
                handover.resumed.add(internal_id)

            if internal_id not in handover.adopted:
                self._carry_sampler_state(original.request_id)
                self._carried[original.request_id] = token0
                state = self._decode.states[original.request_id]
                if state.finish_reason is None and token0.grammar_terminated:
                    self._decode.finish_early(original.request_id)
                if self._decode.finish_reason(original.request_id) is not None:
                    self._outputs[original.request_id] = self._decode.output_tokens(
                        original.request_id
                    )
                handover.adopted.add(internal_id)

            if internal_id in handover.commit:
                try:
                    self._prefill.update({internal_id: handover.commit[internal_id]})
                except BaseException as commit_error:
                    source_state = self._prefill.states.get(internal_id)
                    source_allocation = (
                        source_state.allocation if source_state is not None else None
                    )
                    if (
                        source_state is not None
                        and source_state.finish_reason is not None
                        and source_allocation is not None
                        and source_allocation._freed
                    ):
                        # An outer hook raised after ``update`` committed and
                        # released the clone. Record that completed phase before
                        # propagating so re-entry cannot commit token 0 twice.
                        handover.commit.pop(internal_id)
                        handover.resumed.discard(internal_id)
                        handover.adopted.discard(internal_id)
                        handover.adopt.pop(index)
                        raise
                    if (
                        source_state is not None
                        and source_state.finish_reason is not None
                    ):
                        # The scheduler crossed its terminal mutation but could
                        # not prove allocation release. Retrying update is not
                        # legal and releasing either side could race.
                        handover.poisoned[internal_id] = commit_error
                    raise
                handover.commit.pop(internal_id)
            handover.resumed.discard(internal_id)
            handover.adopted.discard(internal_id)
            handover.adopt.pop(index)

        while handover.retries:
            internal_id, original, attempt, _, _ = handover.retries[0]
            self._prefill.abort(internal_id)
            if internal_id not in handover.cancelled:
                if attempt < self._max_retries:
                    self._discard_clone(internal_id, original.request_id)
                    self._enqueue(original, attempt + 1)
                else:
                    self._release_sampling_state(original.request_id)
                    if original.request_id not in self._failed:
                        self._failed.append(original.request_id)
            handover.retries.pop(0)

        if handover.commit:  # pragma: no cover - state-machine invariant
            raise RuntimeError(f"unsettled P-D source commits: {tuple(handover.commit)}")
        self._handover = None
        return True

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

    def _start_handover(self, sampled: Mapping[str, tuple[SampledToken, ...]]) -> None:
        if self._handover is not None:  # pragma: no cover - state-machine guard
            raise RuntimeError("cannot start a second P-D handover batch")
        handover = _Handover()
        self._handover = handover
        items = tuple(sampled.items())
        for index, (internal_id, tokens) in enumerate(items):
            try:
                self._handoff_or_retry(internal_id, tokens[0], handover)
            except BaseException as error:
                if getattr(error, "handoff_completion_lost", False):
                    handover.poisoned[internal_id] = error
                elif internal_id not in handover.poisoned:
                    # A wrapper may raise after the current transfer registered
                    # its adoption/retry. Stage it again only when handover does
                    # not already own it; later sampled items are always staged.
                    owned = (
                        internal_id in handover.commit
                        or any(
                            held_id == internal_id
                            for held_id, *_ in handover.adopt
                        )
                        or any(
                            held_id == internal_id
                            for held_id, *_ in handover.retries
                        )
                    )
                    remaining = items[index + int(owned) :]
                    self._staged_sampled = dict(remaining) if remaining else None
                if not handover:
                    self._handover = None
                raise
        if not handover:
            self._handover = None
        if self._completion_ready is None:
            # Blocking handoffs synchronized and published inside transfer().
            self._settle_handover()

    def step_prefill(self, *, reject_on_stall: bool = False) -> None:
        """One prefill step: schedule, execute, hand the KV off, then commit.

        ``reject_on_stall`` is the engine-loop backstop (``EngineLoop.step``
        does the same for its own scheduler): an empty plan while requests are
        unfinished means nothing is running to free pages, so the waiting head
        is finished rather than the whole engine — and every concurrent request
        with it — dying on a stall.
        """
        if self._staged_sampled is not None:
            # One overlap forward already ran.  If the older copy is still
            # pending, return to the driver so existing decode work can progress;
            # never schedule these completed clones a second time.
            if not self._settle_handover():
                return
            sampled, self._staged_sampled = self._staged_sampled, None
            self._start_handover(sampled)
            return

        plan = self._prefill.schedule()
        if not plan.scheduled and self._handover is not None:
            # Nothing remains to overlap. Poll rather than blocking the host; the
            # driver can run decode and call us again when the event is ready.
            if not self._settle_handover():
                return
            plan = self._prefill.schedule()
        if not plan.scheduled:
            if not self._prefill.has_unfinished():
                return
            if reject_on_stall and self._prefill.reject_waiting_head() is not None:
                return
            raise RuntimeError("P-D prefill stall: nothing schedulable")
        sampled = self._prefill_runner.execute(plan.scheduled, self._prefill.states)
        # The previous copy ran alongside this forward and the decode step queued
        # before it. If it is still pending, retain this result and poll on later
        # host turns: no host synchronize and no duplicate scheduling.
        if not self._settle_handover():
            self._staged_sampled = dict(sampled)
            return
        self._start_handover(sampled)

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
        return (
            self._prefill.has_unfinished()
            or self._handover is not None
            or self._staged_sampled is not None
        )

    def run_to_completion(self) -> dict[str, tuple[int, ...]]:
        while self.has_prefill_work() or self._decode.has_unfinished():
            if self.has_prefill_work():
                self.step_prefill()
            if self._decode.has_unfinished():
                self._step_decode()
        return dict(self._outputs)
