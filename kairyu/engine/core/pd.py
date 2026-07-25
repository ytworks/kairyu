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
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from kairyu.engine.core.engine_core import ModelRunner, token_ids
from kairyu.engine.core.radix_kv import KVAllocation, KVCacheFull, RadixKVCache
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
        self._prefill.add_request(clone)

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

    def _step_prefill(self) -> None:
        plan = self._prefill.schedule()
        if not plan.scheduled:
            if self._prefill.has_unfinished():
                raise RuntimeError("P-D prefill stall: nothing schedulable")
            return
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
                self._step_prefill()
            if self._decode.has_unfinished():
                self._step_decode()
        return dict(self._outputs)
