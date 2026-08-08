"""Request-local arbitration between precise and approximate KV routing.

``KvEventIndex`` and ``PrefixIndex`` intentionally track different key spaces:
engine block hashes versus gateway text chunks.  Their overlap counts therefore
must never be mixed in one scoring pass.  ``KvRoutingIndex`` prepares the
existing approximate keys once, then chooses one mode for the whole request:

* precise only while every eligible replica's event feed is live;
* otherwise the existing ``PrefixIndex`` path for every replica.

The exact index is duck typed so its transport/state-machine implementation can
evolve independently.  It may expose ``all_live()``, per-replica ``is_live()``,
or only the historical ``overlap(...)->None`` stale signal.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from kairyu.orchestration.prefix_index import PrefixIndex, PreparedPrefixKeys

ApproximateKeys = str | PreparedPrefixKeys
BlockHashProvider = Callable[[str], Sequence[str] | None]


class ExactKvIndex(Protocol):
    def route_overlaps(
        self,
        replica_ids: Sequence[str],
        block_hashes: Sequence[str],
    ) -> tuple[int, ...] | None: ...


logger = logging.getLogger(__name__)

_EXACT = "exact"
_PROVIDER_UNAVAILABLE = "approximate_provider_unavailable"
_PROVIDER_ERROR = "approximate_provider_error"
_INVALID_HASHES = "approximate_invalid_hashes"
_FEED_NOT_LIVE = "approximate_feed_not_live"
_FEED_CHANGED = "approximate_feed_changed"
_MODES = (
    _EXACT,
    _PROVIDER_UNAVAILABLE,
    _PROVIDER_ERROR,
    _INVALID_HASHES,
    _FEED_NOT_LIVE,
    _FEED_CHANGED,
)


@dataclass(frozen=True)
class PreparedKvRouting:
    """Request-owned work shared by selection and successful observation."""

    prompt: str
    approximate: ApproximateKeys


@dataclass(frozen=True)
class KvRoutingView:
    """Immutable result of choosing one routing mode for one request."""

    mode: Literal["exact", "approximate"]
    reason: str
    # Aligned with the eligible replica tuple passed to ``route_view``.
    overlaps: tuple[int, ...] = ()


class KvRoutingIndex:
    """Combine a precise event index with the existing approximate index.

    The adapter owns no transport lifecycle.  ``exact`` must provide atomic
    ``route_overlaps(replica_ids, block_hashes)`` and may provide
    ``all_live``, ``is_live``, ``register_replica``, and ``forget_replica``.
    Optional exact routing failures degrade to the approximate index rather
    than failing an otherwise valid generation request.
    """

    def __init__(
        self,
        approximate: PrefixIndex,
        exact: ExactKvIndex,
        block_hash_provider: BlockHashProvider | None,
    ) -> None:
        self.approximate = approximate
        self.exact = exact
        self._block_hash_provider = block_hash_provider
        self._mode_counts = {mode: 0 for mode in _MODES}
        # An exact lifecycle call may fail after partially mutating its index.
        # Keep the replica fail-closed until a complete forget/register reset
        # succeeds; a register retry alone cannot prove stale state was erased.
        self._lifecycle_lock = threading.RLock()
        self._lifecycle_quarantine: set[str] = set()
        self._lifecycle_reset_ready: set[str] = set()
        self._lifecycle_revision = 0

    @property
    def mode_counts(self) -> dict[str, int]:
        """Cumulative request-mode choices; returned as a defensive copy."""

        return dict(self._mode_counts)

    def prepare(
        self,
        prompt: str,
        first_key: str | None = None,
    ) -> PreparedKvRouting:
        return PreparedKvRouting(
            prompt=prompt,
            approximate=self.approximate.prepare(prompt, first_key),
        )

    def route_view(
        self,
        eligible: Sequence[str],
        prepared: PreparedKvRouting,
    ) -> KvRoutingView:
        """Choose exact or approximate once without mixing their score units."""

        replica_ids = tuple(eligible)
        route_overlaps = getattr(self.exact, "route_overlaps", None)
        if not callable(route_overlaps):
            return self._fallback(_FEED_NOT_LIVE)
        provider = self._block_hash_provider
        if provider is None:
            return self._fallback(_PROVIDER_UNAVAILABLE)
        with self._lifecycle_lock:
            if not self._lifecycle_quarantine.isdisjoint(replica_ids):
                return self._fallback(_FEED_CHANGED)
            lifecycle_revision = self._lifecycle_revision
        try:
            feeds_live = self._all_live(replica_ids)
        except Exception:
            return self._fallback(_FEED_CHANGED)
        if not feeds_live:
            # Avoid gateway tokenization/hashing while the result cannot be used.
            return self._fallback(_FEED_NOT_LIVE)
        try:
            supplied = provider(prepared.prompt)
            if supplied is None:
                return self._fallback(_PROVIDER_UNAVAILABLE)
            if isinstance(supplied, (str, bytes)):
                return self._fallback(_INVALID_HASHES)
            block_hashes = tuple(supplied)
        except Exception:
            return self._fallback(_PROVIDER_ERROR)
        if any(not isinstance(block_hash, str) for block_hash in block_hashes):
            return self._fallback(_INVALID_HASHES)

        try:
            with self._lifecycle_lock:
                if (
                    self._lifecycle_revision != lifecycle_revision
                    or not self._lifecycle_quarantine.isdisjoint(replica_ids)
                ):
                    return self._fallback(_FEED_CHANGED)
            supplied_overlaps = route_overlaps(
                replica_ids,
                block_hashes,
            )
            with self._lifecycle_lock:
                if (
                    self._lifecycle_revision != lifecycle_revision
                    or not self._lifecycle_quarantine.isdisjoint(replica_ids)
                ):
                    return self._fallback(_FEED_CHANGED)
            overlaps = None if supplied_overlaps is None else tuple(supplied_overlaps)
        except Exception:
            return self._fallback(_FEED_CHANGED)
        if (
            overlaps is None
            or len(overlaps) != len(replica_ids)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > len(block_hashes)
                for value in overlaps
            )
        ):
            return self._fallback(_FEED_CHANGED)
        self._mode_counts[_EXACT] += 1
        return KvRoutingView(
            mode="exact",
            reason=_EXACT,
            overlaps=overlaps,
        )

    def _fallback(self, reason: str) -> KvRoutingView:
        self._mode_counts[reason] += 1
        return KvRoutingView(mode="approximate", reason=reason)

    def _all_live(self, replica_ids: Sequence[str]) -> bool:
        all_live = getattr(self.exact, "all_live", None)
        if callable(all_live):
            return bool(all_live(tuple(replica_ids)))
        is_live = getattr(self.exact, "is_live", None)
        if callable(is_live):
            return all(bool(is_live(replica_id)) for replica_id in replica_ids)
        # Atomic bulk validation remains authoritative when no cheap precheck
        # is offered.
        return True

    # PrefixIndex-compatible lifecycle and successful-observation surface.
    def register_replica(self, replica_id: str) -> None:
        self.approximate.register_replica(replica_id)
        register = getattr(self.exact, "register_replica", None)
        if callable(register):
            with self._lifecycle_lock:
                self._lifecycle_revision += 1
                try:
                    register(replica_id)
                except Exception:
                    self._lifecycle_quarantine.add(replica_id)
                    self._lifecycle_reset_ready.discard(replica_id)
                    logger.exception(
                        "exact KV index failed to register replica %r; "
                        "exact routing quarantined until a successful "
                        "forget/register reset",
                        replica_id,
                    )
                else:
                    if replica_id in self._lifecycle_reset_ready:
                        self._lifecycle_reset_ready.remove(replica_id)
                        self._lifecycle_quarantine.discard(replica_id)

    def forget_replica(self, replica_id: str) -> None:
        self.approximate.forget_replica(replica_id)
        forget = getattr(self.exact, "forget_replica", None)
        if callable(forget):
            with self._lifecycle_lock:
                self._lifecycle_revision += 1
                try:
                    forget(replica_id)
                except Exception:
                    self._lifecycle_quarantine.add(replica_id)
                    self._lifecycle_reset_ready.discard(replica_id)
                    logger.exception(
                        "exact KV index failed to forget replica %r; "
                        "exact routing quarantined until a successful "
                        "forget/register reset",
                        replica_id,
                    )
                else:
                    if replica_id in self._lifecycle_quarantine:
                        self._lifecycle_reset_ready.add(replica_id)

    def observe_prepared(
        self,
        replica_id: str,
        prepared: PreparedKvRouting,
        promote: bool,
    ) -> None:
        self.approximate.observe_prepared(
            replica_id,
            prepared.approximate,
            promote,
        )

    def observe_root(self, replica_id: str, key: str) -> None:
        self.approximate.observe_root(replica_id, key)

    def observe(self, replica_id: str, prompt: str) -> None:
        self.approximate.observe(replica_id, prompt)

    def chunk_keys(
        self,
        prompt: str,
        first_key: str | None = None,
    ) -> tuple[str, ...]:
        return self.approximate.chunk_keys(prompt, first_key)

    def overlap_keys(self, replica_id: str, keys: Sequence[str]) -> int:
        return self.approximate.overlap_keys(replica_id, keys)

    def candidate_ids(self, keys: Sequence[str]) -> tuple[str, ...]:
        return self.approximate.candidate_ids(keys)
