"""Remote P-D handoff over a KVTransport (m18 D2, amended).

Two artifacts (review amendment 1):

- ``RemoteKVHandoff`` — a SINGLE-process ``KVHandoff``: extracts the prompt's
  page bytes from the prefill pool, sends them over the transport, and hands
  the receiver half (``RemoteKVReceiver``) the decode-side adoption. Wire
  ordering IS copy-before-commit: PDCoordinator calls transfer() between
  execute() and update(), while every source page is still locked and valid
  (the tail page is freed by commit_and_release — extraction after commit
  would read reallocated memory, review amendment 2).
- ``RemoteKVReceiver`` — decode-side: allocate → skip the leading
  ``len(cached_pages)`` frames (radix matches are prefix-only — receiver-side
  dedup skips INJECTION, not wire bytes; amendment 4) → inject the rest into
  ``new_full_pages + (tail_page,)`` in order → mark_computed (publication is
  conditional on no uncomputed-sibling collision — amendment 8c).

The two-process E2E (tests/dist) drives these halves directly in separate
processes; ``PDCoordinator`` stays single-process with either handoff.
"""

from __future__ import annotations

import asyncio

from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.engine.core.kv_serde import extract_pages, inject_page
from kairyu.engine.core.kv_transport import KVTransportError, SequenceMeta
from kairyu.engine.core.pd import KVHandoffError
from kairyu.engine.core.radix_kv import KVAllocation, KVCacheFull, RadixKVCache


class RemoteKVReceiver:
    """Decode-side adoption: frames + meta -> local allocation with real bytes."""

    def __init__(self, cache: RadixKVCache, pool: PagedKVPool) -> None:
        self._cache = cache
        self._pool = pool
        self.injected_pages = 0  # observability: dedup gates assert on this

    def adopt(self, frames, meta: SequenceMeta) -> KVAllocation:
        try:
            allocation = self._cache.allocate(tuple(meta.token_ids))
        except KVCacheFull as error:
            raise KVHandoffError(f"decode cache full: {error}") from error
        cached = len(allocation.cached_pages)
        targets = tuple(allocation.new_full_pages) + (allocation.tail_page,)
        incoming = frames[cached:]
        usable = [t for t in targets if t is not None]
        if len(incoming) > len(usable):
            raise KVTransportError(
                f"received {len(incoming)} non-cached frames for {len(usable)} slots"
            )
        for frame, local_page in zip(incoming, usable, strict=False):
            inject_page(self._pool, local_page, frame)
            self.injected_pages += 1
        self._cache.mark_computed(allocation)
        return allocation


class RemoteKVHandoff:
    """Prefill-side KVHandoff over an async transport + a local receiver hook.

    For the single-process (LocalFabric) topology the decode half is invoked
    inline; the transport still carries the REAL bytes, so serde and ordering
    are exercised end-to-end.
    """

    def __init__(
        self,
        transport,
        peer: str,
        prefill_pool: PagedKVPool,
        receiver: RemoteKVReceiver,
        receiver_transport,
        sender_name: str,
    ) -> None:
        self._transport = transport
        self._peer = peer
        self._pool = prefill_pool
        self._receiver = receiver
        self._receiver_transport = receiver_transport
        # the receiver reads the channel keyed by the SENDER's name
        self._sender_name = sender_name

    def transfer(
        self, tokens: tuple[int, ...], first_token: int, pages: tuple[int, ...] = ()
    ) -> KVAllocation:
        if not pages:
            raise KVHandoffError("remote handoff needs the source page ids")
        frames = extract_pages(self._pool, pages)
        meta = SequenceMeta(token_ids=tuple(tokens), first_token=first_token)

        async def _round_trip() -> KVAllocation:
            await self._transport.send(self._peer, frames, meta)
            received, received_meta = await self._receiver_transport.recv(self._sender_name)
            return self._receiver.adopt(received, received_meta)

        try:
            return asyncio.run(_round_trip())
        except (KVTransportError, OSError) as error:
            raise KVHandoffError(f"kv transfer failed: {error}") from error
