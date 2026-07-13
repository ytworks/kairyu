"""Precise per-replica KV-block index fed by engine events (m10b D7/A12).

Separate from the text-chunk ``PrefixIndex`` (the gateway cannot compute
engine block hashes — A12): this structure tracks which BLOCK HASHES each
replica reported via BlockStored/BlockRemoved, plus per-replica staleness.
When a replica's feed goes stale (> ``staleness_s`` since its last event or
heartbeat), ``overlap`` reports None so the caller falls back to the
approximate trie (the m10b chaos contract).

Transport: ZMQ PUB on the engine side, SUB here — both thin wrappers around
``apply``; the wire schema is the vLLM-compatible event dict from radix_kv.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

_DEFAULT_STALENESS_S = 0.5


@dataclass
class _ReplicaBlocks:
    hashes: set[str] = field(default_factory=set)
    last_event: float = 0.0


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
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or block_size <= 0
        ):
            raise ValueError("KV event block_size must be a positive integer")

    if kind == "AllBlocksCleared" and "block_hashes" not in event:
        return kind, ()

    block_hashes = event.get("block_hashes")
    if not isinstance(block_hashes, list):
        raise ValueError(f"{kind} block_hashes must be a list")
    if any(not isinstance(block_hash, str) for block_hash in block_hashes):
        raise ValueError(f"{kind} block_hashes members must be strings")
    return kind, tuple(block_hashes)


class KvEventIndex:
    def __init__(
        self,
        staleness_s: float = _DEFAULT_STALENESS_S,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._staleness_s = staleness_s
        self._now = now
        self._replicas: dict[str, _ReplicaBlocks] = {}

    def apply(self, replica_id: str, event: object) -> None:
        kind, hashes = _validate_event(event)
        entry = self._replicas.setdefault(replica_id, _ReplicaBlocks())
        if kind == "BlockStored":
            entry.hashes.update(hashes)
        elif kind == "BlockRemoved":
            entry.hashes.difference_update(hashes)
        else:
            entry.hashes.clear()  # vLLM emits this on a cache reset
        # stamp freshness only AFTER a valid apply (M4): a stream of garbage
        # events must NOT keep the replica "fresh" while its index rots — an
        # unrecognized event lets staleness fall back to the trie instead
        entry.last_event = self._now()

    def heartbeat(self, replica_id: str) -> None:
        self._replicas.setdefault(replica_id, _ReplicaBlocks()).last_event = self._now()

    def is_stale(self, replica_id: str) -> bool:
        entry = self._replicas.get(replica_id)
        if entry is None:
            return True
        return self._now() - entry.last_event > self._staleness_s

    def overlap(self, replica_id: str, block_hashes: list[str]) -> int | None:
        """Longest known prefix of ``block_hashes``; None = stale -> caller
        falls back to the approximate trie (graceful degradation)."""
        if self.is_stale(replica_id):
            return None
        entry = self._replicas[replica_id]
        count = 0
        for block_hash in block_hashes:
            if block_hash not in entry.hashes:
                break
            count += 1
        return count

    def forget_replica(self, replica_id: str) -> None:
        self._replicas.pop(replica_id, None)


class ZmqKvEventPublisher:
    """Engine side: RadixKVCache(event_sink=publisher) -> ZMQ PUB."""

    def __init__(self, endpoint: str, replica_id: str) -> None:
        import zmq  # deferred: fleet extra

        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(endpoint)
        self._replica_id = replica_id

    def __call__(self, event: dict) -> None:
        import json

        self._socket.send_multipart(
            [self._replica_id.encode(), json.dumps(event).encode()]
        )

    def close(self) -> None:
        self._socket.close(linger=0)


class ZmqKvEventSubscriber:
    """Gateway side: drain pending events into a KvEventIndex."""

    def __init__(self, endpoints: list[str], index: KvEventIndex) -> None:
        import zmq

        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        for endpoint in endpoints:
            self._socket.connect(endpoint)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._index = index

    def drain(self, max_events: int = 1000) -> int:
        import json
        import logging

        import zmq

        drained = 0
        while drained < max_events:
            try:
                replica_id, payload = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            # a single malformed frame or unknown event type must not abort the
            # drain mid-queue and desync the whole index (M4): drop it and continue
            try:
                self._index.apply(replica_id.decode(), json.loads(payload))
            except (ValueError, KeyError, UnicodeDecodeError) as error:
                logging.getLogger("kairyu.kv_index").warning(
                    "dropped a malformed KV event: %r", error
                )
            drained += 1
        return drained

    def close(self) -> None:
        self._socket.close(linger=0)
