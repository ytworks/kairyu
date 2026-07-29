"""Recoverable per-replica KV-block index (m10b D7/A31).

The approximate text ``PrefixIndex`` remains the always-available routing
fallback.  This module maintains the optional exact block-hash view.  Exact
state is usable only while its source stream is contiguous, synchronized, and
inside a short route-time freshness lease.

The wire keeps vLLM's useful PUB sequence + replay shape, but adds the pieces a
consumer needs to fail closed:

* a cache-lifetime epoch, so sequence reset cannot revive an old index;
* explicit replay completeness;
* an authoritative snapshot when the bounded journal cannot cover a gap; and
* a heartbeat high-water mark that detects a dropped tail without making a
  gapped index trusted.

Legacy direct ``apply`` and two-frame PUB/SUB messages are retained for callers
that do not opt into the sequenced protocol.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

_DEFAULT_STALENESS_S = 0.4
_DEFAULT_HEARTBEAT_S = 0.1
_WIRE_VERSION = 1
_STATE_UNKNOWN = "unknown"
_STATE_LIVE = "live"
_STATE_GAP = "gap"
_STATE_SYNCING = "syncing"
_STATE_INACTIVE = "inactive"

logger = logging.getLogger("kairyu.kv_index")


@dataclass
class _ReplicaBlocks:
    hashes: set[str] = field(default_factory=set)
    last_event: float = 0.0
    stream_epoch: str | None = None
    last_sequence: int = -1
    trusted: bool = False
    synchronized: bool = False
    state: str = _STATE_UNKNOWN
    # A membership generation may receive arbitrarily delayed frames. Keep
    # every retired cache epoch for the process lifetime; a bounded FIFO lets
    # sufficiently many newer epochs revive an old cache after replica-ID
    # reuse.
    retired_epochs: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class KvEventStateView:
    """One atomic, immutable replica-state observation."""

    replica_id: str
    state: str
    stream_epoch: str | None
    last_sequence: int
    trusted: bool
    synchronized: bool
    age_s: float | None
    block_hashes: tuple[str, ...]


def _validate_event(event: object) -> tuple[str, tuple[str, ...]]:
    if not isinstance(event, dict):
        raise ValueError("KV event must be a JSON object")

    kind = event.get("type")
    if not isinstance(kind, str):
        raise ValueError("KV event type must be a string")
    if kind not in ("BlockStored", "BlockRemoved", "AllBlocksCleared"):
        raise ValueError(f"unknown KV event type {kind!r}")

    if "block_size" in event:
        block_size = event["block_size"]
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("KV event block_size must be a positive integer")

    if kind == "AllBlocksCleared" and "block_hashes" not in event:
        return kind, ()

    block_hashes = event.get("block_hashes")
    if not isinstance(block_hashes, list):
        raise ValueError(f"{kind} block_hashes must be a list")
    if any(not isinstance(block_hash, str) for block_hash in block_hashes):
        raise ValueError(f"{kind} block_hashes members must be strings")
    return kind, tuple(block_hashes)


def _validate_stream_identity(stream_epoch: object, sequence: object) -> tuple[str, int]:
    if not isinstance(stream_epoch, str) or not stream_epoch:
        raise ValueError("KV event stream_epoch must be a non-empty string")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("KV event sequence must be a non-negative integer")
    return stream_epoch, sequence


def _apply_validated(entry: _ReplicaBlocks, kind: str, hashes: Sequence[str]) -> None:
    if kind == "BlockStored":
        entry.hashes.update(hashes)
    elif kind == "BlockRemoved":
        entry.hashes.difference_update(hashes)
    else:
        entry.hashes.clear()


class KvEventIndex:
    """Exact block truth with a route-time fail-closed freshness lease."""

    def __init__(
        self,
        staleness_s: float = _DEFAULT_STALENESS_S,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if staleness_s <= 0:
            raise ValueError("staleness_s must be positive")
        self._staleness_s = staleness_s
        self._now = now
        self._replicas: dict[str, _ReplicaBlocks] = {}
        self._inactive_replicas: set[str] = set()
        self._lock = threading.RLock()

    @property
    def staleness_s(self) -> float:
        return self._staleness_s

    def register_replica(self, replica_id: str) -> None:
        with self._lock:
            entry = self._replicas.setdefault(replica_id, _ReplicaBlocks())
            if replica_id in self._inactive_replicas:
                self._inactive_replicas.remove(replica_id)
                entry.hashes.clear()
                entry.last_event = 0.0
                entry.stream_epoch = None
                entry.last_sequence = -1
                entry.trusted = False
                entry.synchronized = False
                entry.state = _STATE_UNKNOWN

    @staticmethod
    def _adopt_epoch(entry: _ReplicaBlocks, stream_epoch: str) -> None:
        previous = entry.stream_epoch
        if previous is not None and previous != stream_epoch:
            entry.retired_epochs.add(previous)
        entry.hashes.clear()
        entry.stream_epoch = stream_epoch
        entry.last_sequence = -1
        entry.trusted = False
        entry.synchronized = False
        entry.state = _STATE_SYNCING

    def _is_live_unlocked(self, entry: _ReplicaBlocks, now: float) -> bool:
        return bool(
            entry.trusted
            and entry.state == _STATE_LIVE
            and now - entry.last_event < self._staleness_s
        )

    def _view_unlocked(
        self,
        replica_id: str,
        now: float,
    ) -> KvEventStateView:
        entry = self._replicas.get(replica_id)
        if entry is None:
            return KvEventStateView(
                replica_id=replica_id,
                state=_STATE_UNKNOWN,
                stream_epoch=None,
                last_sequence=-1,
                trusted=False,
                synchronized=False,
                age_s=None,
                block_hashes=(),
            )
        age_s = (
            max(0.0, now - entry.last_event)
            if entry.state not in {_STATE_UNKNOWN, _STATE_INACTIVE}
            else None
        )
        state = entry.state
        if (
            entry.trusted
            and entry.state == _STATE_LIVE
            and age_s is not None
            and age_s >= self._staleness_s
        ):
            state = "stale"
        return KvEventStateView(
            replica_id=replica_id,
            state=state,
            stream_epoch=entry.stream_epoch,
            last_sequence=entry.last_sequence,
            trusted=entry.trusted,
            synchronized=entry.synchronized,
            age_s=age_s,
            block_hashes=tuple(sorted(entry.hashes)),
        )

    def view(self, replica_id: str) -> KvEventStateView:
        """Return epoch, sequence, trust, age, and hashes from one lock epoch."""

        with self._lock:
            return self._view_unlocked(replica_id, self._now())

    def has_sequenced_stream(self, replica_id: str) -> bool:
        """Check protocol mode without sampling the freshness clock."""

        with self._lock:
            entry = self._replicas.get(replica_id)
            return bool(entry is not None and entry.stream_epoch is not None)

    def apply(self, replica_id: str, event: object) -> None:
        """Apply one legacy, unsequenced event atomically.

        This compatibility path cannot detect loss.  Sequenced transports use
        ``apply_sequenced`` instead.
        """

        kind, hashes = _validate_event(event)
        with self._lock:
            if replica_id in self._inactive_replicas:
                raise ValueError("KV event replica is inactive")
            entry = self._replicas.setdefault(replica_id, _ReplicaBlocks())
            if entry.stream_epoch is not None:
                # Never let a legacy frame downgrade a sequenced feed.  The
                # mutation is deliberately fail-closed even though the caller
                # receives a controlled validation error.
                entry.trusted = False
                entry.synchronized = False
                entry.state = _STATE_GAP
                raise ValueError("legacy KV event cannot replace a sequenced stream")
            _apply_validated(entry, kind, hashes)
            entry.last_event = self._now()
            entry.last_sequence = -1
            entry.trusted = True
            # Legacy feeds remain available to legacy overlap callers, but
            # cannot enter the loss-detecting composite bulk route.
            entry.synchronized = False
            entry.state = _STATE_LIVE

    def apply_sequenced(
        self,
        replica_id: str,
        stream_epoch: str,
        sequence: int,
        event: object,
    ) -> Literal[
        "applied",
        "duplicate",
        "gap",
        "stale_epoch",
        "inactive",
    ]:
        """Apply only the next contiguous delta.

        A gap mutates no block truth and immediately makes the whole replica
        unusable for exact routing.  Replayed deltas may advance the staged
        state, but a previously gapped entry remains untrusted until
        ``mark_synchronized`` verifies the source high-water mark.
        """

        stream_epoch, sequence = _validate_stream_identity(stream_epoch, sequence)
        kind, hashes = _validate_event(event)
        with self._lock:
            if replica_id in self._inactive_replicas:
                return "inactive"
            entry = self._replicas.setdefault(replica_id, _ReplicaBlocks())
            if stream_epoch in entry.retired_epochs:
                return "stale_epoch"
            changed_epoch = entry.stream_epoch != stream_epoch
            now = self._now()
            was_live = not changed_epoch and self._is_live_unlocked(entry, now)
            if changed_epoch:
                self._adopt_epoch(entry, stream_epoch)

            if sequence <= entry.last_sequence:
                return "duplicate"
            expected = entry.last_sequence + 1
            if sequence != expected:
                entry.trusted = False
                entry.synchronized = False
                entry.state = _STATE_GAP
                return "gap"

            _apply_validated(entry, kind, hashes)
            entry.last_sequence = sequence
            entry.last_event = now
            # Sequence zero is a complete empty-cache baseline only when the
            # epoch is genuinely new.  A delayed seq=0 in an existing GAP must
            # not revive incomplete state.  Likewise, a delta arriving after
            # the lease expired remains staged until a matching heartbeat,
            # complete replay, or snapshot proves the high-water mark.
            if was_live or (changed_epoch and sequence == 0):
                entry.trusted = True
                entry.state = _STATE_LIVE
            else:
                entry.trusted = False
                entry.state = _STATE_SYNCING
            if not was_live:
                entry.synchronized = False
            return "applied"

    def mark_gap(
        self,
        replica_id: str,
        *,
        stream_epoch: str | None = None,
    ) -> bool:
        if stream_epoch is not None and (not isinstance(stream_epoch, str) or not stream_epoch):
            raise ValueError("gap stream_epoch must be a non-empty string")
        with self._lock:
            if replica_id in self._inactive_replicas:
                return False
            entry = self._replicas.setdefault(replica_id, _ReplicaBlocks())
            if stream_epoch is not None and stream_epoch in entry.retired_epochs:
                return False
            if stream_epoch is not None and entry.stream_epoch != stream_epoch:
                self._adopt_epoch(entry, stream_epoch)
            entry.trusted = False
            entry.synchronized = False
            entry.state = _STATE_GAP
            return True

    def install_snapshot(
        self,
        replica_id: str,
        stream_epoch: str,
        sequence: int,
        block_hashes: Sequence[str],
    ) -> bool:
        """Atomically replace one replica with a source-authoritative snapshot."""

        if not isinstance(stream_epoch, str) or not stream_epoch:
            raise ValueError("snapshot stream_epoch must be a non-empty string")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < -1:
            raise ValueError("snapshot sequence must be >= -1")
        if isinstance(block_hashes, (str, bytes)) or any(
            not isinstance(block_hash, str) for block_hash in block_hashes
        ):
            raise ValueError("snapshot block_hashes must contain only strings")
        with self._lock:
            if replica_id in self._inactive_replicas:
                return False
            entry = self._replicas.setdefault(replica_id, _ReplicaBlocks())
            if stream_epoch in entry.retired_epochs:
                return False
            if entry.stream_epoch != stream_epoch:
                self._adopt_epoch(entry, stream_epoch)
            entry.hashes = set(block_hashes)
            entry.last_sequence = sequence
            entry.last_event = self._now()
            entry.trusted = True
            entry.synchronized = True
            entry.state = _STATE_LIVE
            return True

    def mark_synchronized(
        self,
        replica_id: str,
        stream_epoch: str,
        high_sequence: int,
    ) -> bool:
        if not isinstance(stream_epoch, str) or not stream_epoch:
            raise ValueError("synchronized stream_epoch must be a non-empty string")
        if (
            isinstance(high_sequence, bool)
            or not isinstance(high_sequence, int)
            or high_sequence < -1
        ):
            raise ValueError("synchronized high_sequence must be >= -1")
        with self._lock:
            if replica_id in self._inactive_replicas:
                return False
            entry = self._replicas.get(replica_id)
            if (
                entry is None
                or stream_epoch in entry.retired_epochs
                or entry.stream_epoch != stream_epoch
                or entry.last_sequence != high_sequence
            ):
                if entry is not None and stream_epoch not in entry.retired_epochs:
                    entry.trusted = False
                    entry.synchronized = False
                    entry.state = _STATE_GAP
                return False
            entry.last_event = self._now()
            entry.trusted = True
            entry.synchronized = True
            entry.state = _STATE_LIVE
            return True

    def heartbeat(
        self,
        replica_id: str,
        *,
        stream_epoch: str | None = None,
        high_sequence: int | None = None,
    ) -> bool:
        """Refresh only a stream whose high-water mark is already synchronized."""

        if (stream_epoch is None) != (high_sequence is None):
            raise ValueError("heartbeat stream_epoch and high_sequence must be paired")
        if stream_epoch is not None:
            if not isinstance(stream_epoch, str) or not stream_epoch:
                raise ValueError("heartbeat stream_epoch must be a non-empty string")
            if (
                isinstance(high_sequence, bool)
                or not isinstance(high_sequence, int)
                or high_sequence < -1
            ):
                raise ValueError("heartbeat high_sequence must be >= -1")
        with self._lock:
            if replica_id in self._inactive_replicas:
                return False
            entry = self._replicas.setdefault(replica_id, _ReplicaBlocks())
            if stream_epoch is None:
                # Legacy compatibility: never use this to revive a sequenced
                # feed or an already quarantined legacy feed.
                if entry.stream_epoch is not None or not entry.trusted:
                    return False
                entry.last_event = self._now()
                entry.state = _STATE_LIVE
                return True
            assert high_sequence is not None
            if stream_epoch in entry.retired_epochs:
                return False
            if entry.stream_epoch != stream_epoch:
                if high_sequence == -1:
                    # An empty high-water mark is itself an authoritative empty
                    # baseline for a genuinely new source epoch.
                    self._adopt_epoch(entry, stream_epoch)
                    entry.last_event = self._now()
                    entry.trusted = True
                    entry.synchronized = True
                    entry.state = _STATE_LIVE
                    return True
                self._adopt_epoch(entry, stream_epoch)
                entry.state = _STATE_GAP
                return False
            if (
                high_sequence != entry.last_sequence
                or not entry.trusted
                or entry.state != _STATE_LIVE
            ):
                entry.trusted = False
                entry.synchronized = False
                entry.state = _STATE_GAP
                return False
            entry.last_event = self._now()
            entry.synchronized = True
            entry.state = _STATE_LIVE
            return True

    def state(self, replica_id: str) -> str:
        return self.view(replica_id).state

    def stream_state(self, replica_id: str) -> dict[str, object]:
        view = self.view(replica_id)
        return {
            "state": view.state,
            "stream_epoch": view.stream_epoch,
            "last_sequence": view.last_sequence,
            "trusted": view.trusted,
            "synchronized": view.synchronized,
            "age_s": view.age_s,
            "block_count": len(view.block_hashes),
        }

    def expected_sequence(
        self,
        replica_id: str,
        stream_epoch: str,
    ) -> int:
        with self._lock:
            if replica_id in self._inactive_replicas:
                return 0
            entry = self._replicas.get(replica_id)
            if entry is None or entry.stream_epoch != stream_epoch:
                return 0
            return entry.last_sequence + 1

    def is_stale(self, replica_id: str) -> bool:
        return not self.is_live(replica_id)

    def is_live(self, replica_id: str) -> bool:
        with self._lock:
            entry = self._replicas.get(replica_id)
            return bool(entry is not None and self._is_live_unlocked(entry, self._now()))

    def all_live(self, replica_ids: Sequence[str]) -> bool:
        with self._lock:
            now = self._now()
            return all(
                replica_id not in self._inactive_replicas
                and (entry := self._replicas.get(replica_id)) is not None
                and entry.stream_epoch is not None
                and entry.synchronized
                and self._is_live_unlocked(entry, now)
                for replica_id in replica_ids
            )

    def route_overlaps(
        self,
        replica_ids: Sequence[str],
        block_hashes: Sequence[str],
    ) -> tuple[int, ...] | None:
        """Return one atomic exact-score vector, or request global fallback.

        Unlike the legacy per-replica ``overlap`` API, this production routing
        surface admits only sequenced feeds whose high-water mark was confirmed
        by a heartbeat, complete replay, or authoritative snapshot.  One lock
        and one clock sample cover the entire eligible fleet, so churn cannot
        construct a score vector that never existed at any instant.
        """

        with self._lock:
            now = self._now()
            entries: list[_ReplicaBlocks] = []
            for replica_id in replica_ids:
                entry = self._replicas.get(replica_id)
                if (
                    replica_id in self._inactive_replicas
                    or entry is None
                    or entry.stream_epoch is None
                    or not entry.synchronized
                    or not self._is_live_unlocked(entry, now)
                ):
                    return None
                entries.append(entry)
            overlaps: list[int] = []
            for entry in entries:
                count = 0
                for block_hash in block_hashes:
                    if block_hash not in entry.hashes:
                        break
                    count += 1
                overlaps.append(count)
            return tuple(overlaps)

    def overlap(
        self,
        replica_id: str,
        block_hashes: Sequence[str],
    ) -> int | None:
        """Longest known prefix, or ``None`` for exact-routing fallback."""

        with self._lock:
            entry = self._replicas.get(replica_id)
            if (
                replica_id in self._inactive_replicas
                or entry is None
                or not self._is_live_unlocked(entry, self._now())
            ):
                return None
            count = 0
            for block_hash in block_hashes:
                if block_hash not in entry.hashes:
                    break
                count += 1
            return count

    def snapshot_hashes(self, replica_id: str) -> tuple[str, ...]:
        return self.view(replica_id).block_hashes

    def forget_replica(self, replica_id: str) -> None:
        with self._lock:
            entry = self._replicas.setdefault(replica_id, _ReplicaBlocks())
            if entry.stream_epoch is not None:
                entry.retired_epochs.add(entry.stream_epoch)
            entry.hashes.clear()
            entry.last_event = 0.0
            entry.stream_epoch = None
            entry.last_sequence = -1
            entry.trusted = False
            entry.synchronized = False
            entry.state = _STATE_INACTIVE
            self._inactive_replicas.add(replica_id)


@dataclass(frozen=True)
class _QueuedWireEvent:
    sequence: int
    payload: bytes


class ZmqKvEventPublisher:
    """Engine-side background PUB plus bounded reliable replay/snapshot."""

    def __init__(
        self,
        endpoint: str,
        replica_id: str,
        *,
        replay_endpoint: str | None = None,
        buffer_events: int = 10_000,
        queue_size: int = 100_000,
        heartbeat_s: float = _DEFAULT_HEARTBEAT_S,
        stream_epoch: str | None = None,
    ) -> None:
        if buffer_events < 1 or queue_size < 1:
            raise ValueError("buffer_events and queue_size must be positive")
        if heartbeat_s <= 0:
            raise ValueError("heartbeat_s must be positive")
        if stream_epoch is not None and (not isinstance(stream_epoch, str) or not stream_epoch):
            raise ValueError("stream_epoch must be a non-empty string")
        self._endpoint = endpoint
        self._replay_endpoint = replay_endpoint
        self._replica_id = replica_id
        self._stream_epoch = stream_epoch if stream_epoch is not None else uuid.uuid4().hex
        self._heartbeat_s = heartbeat_s
        self._queue: queue.Queue[_QueuedWireEvent] = queue.Queue(maxsize=queue_size)
        self._journal: deque[tuple[int, bytes]] = deque(maxlen=buffer_events)
        self._hashes: set[str] = set()
        self._next_sequence = 0
        self._dropped_live = 0
        # A callback belongs to one cache generation, not merely to this
        # long-lived publisher object.  The integer token prevents an old
        # cache callback from being accepted even if an epoch string is later
        # reused.
        self._generation = 0
        self._direct_generation = self._generation
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"kairyu-kv-events-{replica_id}",
        )
        self._thread.start()
        if not self._ready.wait(timeout=5):
            with self._lock:
                self._stop.set()
            self._thread.join(timeout=2)
            raise RuntimeError("KV event publisher socket setup timed out")
        if self._startup_error is not None:
            raise RuntimeError("KV event publisher socket setup failed") from self._startup_error

    @property
    def stream_epoch(self) -> str:
        with self._lock:
            return self._stream_epoch

    @property
    def high_sequence(self) -> int:
        with self._lock:
            return self._next_sequence - 1

    @property
    def dropped_live_events(self) -> int:
        with self._lock:
            return self._dropped_live

    def authoritative_hashes(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._hashes))

    def event_sink(self) -> Callable[[dict], None]:
        """Return a callback bound to the current cache generation.

        A replacement cache must obtain a new callback after
        :meth:`rotate_epoch`.  Calls through an older callback fail closed
        instead of publishing old-cache state under the replacement epoch.
        """

        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("KV event publisher is closed")
            generation = self._generation

        def publish(event: dict) -> None:
            self._publish(event, expected_generation=generation)

        return publish

    def rotate_epoch(
        self,
        *,
        stream_epoch: str | None = None,
    ) -> str:
        """Start a new cache-lifetime stream without restarting the publisher.

        Dynamic membership may replace a cache behind the same replica ID.
        Rotation preserves the publisher object, socket thread, and endpoints,
        while atomically resetting sequence, replay journal, and authoritative
        mirror for the replacement cache.
        """

        next_epoch = uuid.uuid4().hex if stream_epoch is None else stream_epoch
        if not isinstance(next_epoch, str) or not next_epoch:
            raise ValueError("rotated stream_epoch must be a non-empty string")
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("KV event publisher is closed")
            if next_epoch == self._stream_epoch:
                raise ValueError("rotated stream_epoch must be new")
            self._stream_epoch = next_epoch
            self._generation += 1
            self._next_sequence = 0
            self._journal.clear()
            self._hashes.clear()
            # Old queued live frames carry their old epoch and are rejected by
            # the subscriber tombstone.  Drop all frames not already owned by
            # the sender thread so the replacement feed converges promptly.
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._queue.task_done()
            return next_epoch

    def __call__(self, event: dict) -> None:
        """Publish through the construction-time cache generation.

        This preserves the original callable API for publishers that never
        rotate.  Rotation-aware caches use :meth:`event_sink`; direct calls
        after a rotation are rejected because their cache ownership is
        ambiguous.
        """

        self._publish(event, expected_generation=self._direct_generation)

    def _publish(
        self,
        event: dict,
        *,
        expected_generation: int,
    ) -> None:
        kind, hashes = _validate_event(event)
        # Freeze caller-owned values before returning from the cache mutation.
        event_copy = json.loads(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
        emitted_ns = time.monotonic_ns()
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("KV event publisher is closed")
            if expected_generation != self._generation:
                raise RuntimeError("KV event sink belongs to a retired cache epoch")
            sequence = self._next_sequence
            self._next_sequence += 1
            if kind == "BlockStored":
                self._hashes.update(hashes)
            elif kind == "BlockRemoved":
                self._hashes.difference_update(hashes)
            else:
                self._hashes.clear()
            envelope = {
                "wire_version": _WIRE_VERSION,
                "kind": "event",
                "stream_epoch": self._stream_epoch,
                "sequence": sequence,
                "emitted_ns": emitted_ns,
                "event": event_copy,
            }
            payload = json.dumps(
                envelope,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            self._journal.append((sequence, payload))
            queued = _QueuedWireEvent(sequence=sequence, payload=payload)
            try:
                self._queue.put_nowait(queued)
            except queue.Full:
                # The authoritative journal already owns this delta.  A later
                # heartbeat exposes the high-water gap and replay/snapshot
                # repairs it without blocking RadixKV's mutation path.
                self._dropped_live += 1

    def _replay_response(self, request: object) -> dict[str, object]:
        if not isinstance(request, dict):
            raise ValueError("replay request must be an object")
        requested_epoch = request.get("stream_epoch")
        after = request.get("after_sequence")
        if requested_epoch is not None and not isinstance(requested_epoch, str):
            raise ValueError("replay stream_epoch must be a string or null")
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise ValueError("replay after_sequence must be non-negative")
        with self._lock:
            high = self._next_sequence - 1
            journal = tuple(self._journal)
            first_available = journal[0][0] if journal else high + 1
            same_epoch = requested_epoch == self._stream_epoch
            replay_complete = same_epoch and after >= first_available
            if replay_complete:
                events = [json.loads(payload) for sequence, payload in journal if sequence >= after]
                return {
                    "wire_version": _WIRE_VERSION,
                    "mode": "events",
                    "complete": True,
                    "stream_epoch": self._stream_epoch,
                    "requested_from": after,
                    "first_available": first_available,
                    "last_available": high,
                    "high_sequence": high,
                    "events": events,
                }
            return {
                "wire_version": _WIRE_VERSION,
                "mode": "snapshot",
                "complete": True,
                "stream_epoch": self._stream_epoch,
                "requested_from": after,
                "first_available": first_available,
                "last_available": high,
                "high_sequence": high,
                "block_hashes": sorted(self._hashes),
                "reason": ("epoch_mismatch" if not same_epoch else "journal_too_old"),
            }

    def _send_heartbeat(self, socket: Any) -> None:
        with self._lock:
            high = self._next_sequence - 1
            envelope = {
                "wire_version": _WIRE_VERSION,
                "kind": "heartbeat",
                "stream_epoch": self._stream_epoch,
                "high_sequence": high,
                "emitted_ns": time.monotonic_ns(),
            }
        socket.send_multipart(
            (
                self._replica_id.encode(),
                high.to_bytes(8, "big", signed=True),
                json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode(),
            )
        )

    def _service_replay(self, socket: Any) -> None:
        frames = socket.recv_multipart()
        if len(frames) != 2:
            return
        identity, payload = frames
        try:
            request = json.loads(payload.decode())
            response = self._replay_response(request)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            response = {
                "wire_version": _WIRE_VERSION,
                "complete": False,
                "error": type(error).__name__,
            }
        socket.send_multipart(
            (
                identity,
                json.dumps(response, sort_keys=True, separators=(",", ":")).encode(),
            )
        )

    def _run(self) -> None:
        publisher = None
        replay = None
        try:
            import zmq

            context = zmq.Context.instance()
            publisher = context.socket(zmq.PUB)
            publisher.bind(self._endpoint)
            if self._replay_endpoint is not None:
                replay = context.socket(zmq.ROUTER)
                replay.bind(self._replay_endpoint)
        except BaseException as error:
            self._startup_error = error
            self._ready.set()
            if publisher is not None:
                publisher.close(linger=0)
            if replay is not None:
                replay.close(linger=0)
            return
        self._ready.set()
        next_heartbeat = time.monotonic() + self._heartbeat_s
        try:
            while not self._stop.is_set() or not self._queue.empty():
                if replay is not None:
                    while replay.poll(timeout=0):
                        self._service_replay(replay)
                timeout = max(0.0, min(0.02, next_heartbeat - time.monotonic()))
                try:
                    queued = self._queue.get(timeout=timeout)
                except queue.Empty:
                    queued = None
                if queued is not None:
                    publisher.send_multipart(
                        (
                            self._replica_id.encode(),
                            queued.sequence.to_bytes(8, "big", signed=True),
                            queued.payload,
                        )
                    )
                    self._queue.task_done()
                now = time.monotonic()
                if now >= next_heartbeat:
                    self._send_heartbeat(publisher)
                    next_heartbeat = now + self._heartbeat_s
        finally:
            publisher.close(linger=0)
            if replay is not None:
                replay.close(linger=0)

    def close(self) -> None:
        with self._lock:
            self._stop.set()
        self._thread.join(timeout=2)
        if self._thread.is_alive():
            raise RuntimeError("KV event publisher thread did not stop")


class ZmqKvEventSubscriber:
    """Gateway subscriber with gap detection and synchronous recovery."""

    def __init__(
        self,
        endpoints: list[str],
        index: KvEventIndex,
        *,
        replay_endpoints: Mapping[str, str] | None = None,
        replay_timeout_s: float = 0.5,
        on_delivery: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        if replay_timeout_s <= 0:
            raise ValueError("replay_timeout_s must be positive")
        import zmq

        self._context = zmq.Context.instance()
        self._endpoints = tuple(endpoints)
        self._owner_thread_id = threading.get_ident()
        self._socket = self._new_subscriber()
        self._index = index
        self._replay_endpoints = dict(replay_endpoints or {})
        self._replay_timeout_s = replay_timeout_s
        self._on_delivery = on_delivery
        self._paused = False
        self._closed = False
        self.recovery_history: list[dict[str, object]] = []

    def _assert_owner(self) -> None:
        # Legacy unit doubles use object.__new__ and inject only the socket and
        # index.  Real ZMQ sockets remain strictly on their creating thread.
        owner = getattr(self, "_owner_thread_id", None)
        if owner is not None and owner != threading.get_ident():
            raise RuntimeError("KV event subscriber socket used from a non-owner thread")

    def _new_subscriber(self):
        self._assert_owner()
        import zmq

        socket = self._context.socket(zmq.SUB)
        for endpoint in self._endpoints:
            socket.connect(endpoint)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        return socket

    def pause(self) -> None:
        self._assert_owner()
        if self._closed or self._paused:
            return
        self._socket.close(linger=0)
        self._paused = True

    def resume(self) -> None:
        self._assert_owner()
        if self._closed:
            raise RuntimeError("KV event subscriber is closed")
        if not self._paused:
            return
        self._socket = self._new_subscriber()
        self._paused = False

    def _notify(
        self,
        *,
        replica_id: str,
        sequence: int,
        envelope: Mapping[str, object],
        received_ns: int,
        replayed: bool,
        result: str,
    ) -> None:
        if self._on_delivery is None:
            return
        self._on_delivery(
            {
                "replica_id": replica_id,
                "sequence": sequence,
                "stream_epoch": envelope.get("stream_epoch"),
                "emitted_ns": envelope.get("emitted_ns"),
                "received_ns": received_ns,
                "applied_ns": time.monotonic_ns(),
                "event_kind": (
                    envelope.get("event", {}).get("type")
                    if isinstance(envelope.get("event"), dict)
                    else envelope.get("kind")
                ),
                "event": envelope.get("event"),
                "replayed": replayed,
                "result": result,
            }
        )

    def _apply_envelope(
        self,
        replica_id: str,
        sequence: int,
        envelope: object,
        *,
        received_ns: int,
        replayed: bool,
    ) -> str:
        if not isinstance(envelope, dict):
            raise ValueError("sequenced KV envelope must be an object")
        if envelope.get("wire_version") != _WIRE_VERSION:
            raise ValueError("unsupported KV event wire version")
        if envelope.get("kind") != "event":
            raise ValueError("replayed KV envelope must contain an event")
        if envelope.get("sequence") != sequence:
            raise ValueError("KV event frame and payload sequence differ")
        stream_epoch = envelope.get("stream_epoch")
        event = envelope.get("event")
        if not isinstance(stream_epoch, str):
            raise ValueError("KV event envelope stream_epoch is invalid")
        result = self._index.apply_sequenced(
            replica_id,
            stream_epoch,
            sequence,
            event,
        )
        self._notify(
            replica_id=replica_id,
            sequence=sequence,
            envelope=envelope,
            received_ns=received_ns,
            replayed=replayed,
            result=result,
        )
        return result

    def _recover(
        self,
        replica_id: str,
        stream_epoch: str,
    ) -> bool:
        endpoint = self._replay_endpoints.get(replica_id)
        if endpoint is None:
            return False
        import zmq

        requested_from = self._index.expected_sequence(replica_id, stream_epoch)
        socket = self._context.socket(zmq.DEALER)
        socket.connect(endpoint)
        requested_ns = time.monotonic_ns()
        try:
            socket.send_json(
                {
                    "stream_epoch": stream_epoch,
                    "after_sequence": requested_from,
                }
            )
            if not socket.poll(timeout=int(self._replay_timeout_s * 1000)):
                return False
            response = socket.recv_json()
            response_received_ns = time.monotonic_ns()
        finally:
            socket.close(linger=0)
        if (
            not isinstance(response, dict)
            or response.get("wire_version") != _WIRE_VERSION
            or response.get("complete") is not True
        ):
            return False
        response_epoch = response.get("stream_epoch")
        high = response.get("high_sequence")
        if (
            not isinstance(response_epoch, str)
            or isinstance(high, bool)
            or not isinstance(high, int)
            or high < -1
        ):
            return False
        mode = response.get("mode")
        applied_count = 0
        if mode == "snapshot":
            hashes = response.get("block_hashes")
            if not isinstance(hashes, list):
                return False
            if not self._index.install_snapshot(
                replica_id,
                response_epoch,
                high,
                hashes,
            ):
                return False
        elif mode == "events":
            events = response.get("events")
            if not isinstance(events, list):
                return False
            expected = requested_from
            for envelope in events:
                if not isinstance(envelope, dict):
                    return False
                sequence = envelope.get("sequence")
                if sequence != expected:
                    return False
                result = self._apply_envelope(
                    replica_id,
                    expected,
                    envelope,
                    received_ns=response_received_ns,
                    replayed=True,
                )
                if result not in {"applied", "duplicate"}:
                    return False
                expected += 1
                applied_count += 1
            if expected != high + 1:
                return False
            if not self._index.mark_synchronized(replica_id, response_epoch, high):
                return False
        else:
            return False
        record = {
            "replica_id": replica_id,
            "mode": mode,
            "stream_epoch": response_epoch,
            "requested_from": requested_from,
            "first_available": response.get("first_available"),
            "last_available": response.get("last_available"),
            "high_sequence": high,
            "event_count": applied_count,
            "reason": response.get("reason"),
            "requested_ns": requested_ns,
            "response_received_ns": response_received_ns,
            "completed_ns": time.monotonic_ns(),
        }
        self.recovery_history.append(record)
        return True

    def _handle_sequenced(
        self,
        replica_id: str,
        sequence: int,
        envelope: object,
        received_ns: int,
    ) -> None:
        if not isinstance(envelope, dict):
            raise ValueError("sequenced KV envelope must be an object")
        if envelope.get("wire_version") != _WIRE_VERSION:
            raise ValueError("unsupported KV event wire version")
        stream_epoch = envelope.get("stream_epoch")
        if not isinstance(stream_epoch, str):
            raise ValueError("KV event envelope stream_epoch is invalid")
        if envelope.get("kind") == "heartbeat":
            high = envelope.get("high_sequence")
            if isinstance(high, bool) or not isinstance(high, int) or high != sequence:
                raise ValueError("heartbeat high-water mark is invalid")
            refreshed = self._index.heartbeat(
                replica_id,
                stream_epoch=stream_epoch,
                high_sequence=high,
            )
            result = "refreshed"
            if not refreshed:
                accepted = self._index.mark_gap(replica_id, stream_epoch=stream_epoch)
                if not accepted:
                    result = "stale_epoch"
                else:
                    result = "recovered" if self._recover(replica_id, stream_epoch) else "gap"
            self._notify(
                replica_id=replica_id,
                sequence=sequence,
                envelope=envelope,
                received_ns=received_ns,
                replayed=False,
                result=result,
            )
            return
        result = self._apply_envelope(
            replica_id,
            sequence,
            envelope,
            received_ns=received_ns,
            replayed=False,
        )
        if result == "gap":
            self._recover(replica_id, stream_epoch)

    def drain(self, max_events: int = 1000) -> int:
        import zmq

        self._assert_owner()
        # A few focused legacy tests construct the subscriber with
        # ``object.__new__`` and inject only a fake socket plus index.
        if getattr(self, "_closed", False) or getattr(self, "_paused", False):
            return 0
        drained = 0
        while drained < max_events:
            try:
                frames = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            received_ns = time.monotonic_ns()
            drained += 1
            replica_id_for_error: str | None = None
            try:
                if len(frames) not in (2, 3):
                    logger.warning(
                        "dropped a malformed KV event (frame_count=%d)",
                        len(frames),
                    )
                    continue
                replica_bytes = frames[0]
                replica_id = replica_bytes.decode()
                replica_id_for_error = replica_id
                if len(frames) == 2:
                    if self._index.has_sequenced_stream(replica_id):
                        self._index.mark_gap(replica_id)
                        raise ValueError("legacy frame cannot downgrade a sequenced stream")
                    self._index.apply(replica_id, json.loads(frames[1].decode()))
                    continue
                if len(frames[1]) != 8:
                    logger.warning(
                        "dropped a malformed KV event (frame_count=%d)",
                        len(frames),
                    )
                    self._index.mark_gap(replica_id)
                    continue
                sequence = int.from_bytes(frames[1], "big", signed=True)
                envelope = json.loads(frames[2].decode())
                self._handle_sequenced(
                    replica_id,
                    sequence,
                    envelope,
                    received_ns,
                )
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
                if replica_id_for_error is not None and len(frames) == 3:
                    self._index.mark_gap(replica_id_for_error)
                logger.warning(
                    "dropped a malformed KV event (%s)",
                    type(error).__name__,
                )
        return drained

    def close(self) -> None:
        self._assert_owner()
        if self._closed:
            return
        self._socket.close(linger=0)
        self._closed = True
