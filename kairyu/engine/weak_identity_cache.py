"""Per-object caches keyed by the identity of a live object.

Prepared request payloads are cached for exactly one request object and
released when that object dies. The release runs from a weakref callback,
and garbage collection may fire that callback on any thread at almost any
bytecode boundary, including inside a critical section of this very cache on
the thread that already holds its lock. A callback that takes the lock can
therefore deadlock (a plain ``Lock``) or mutate the dict under a caller that
is midway through a lookup (a re-entrant ``RLock``).

The callback here never takes the lock and never touches the dict: it only
appends ``(key, reference)`` to a lock-free deque. Every operation drains that
deque under the lock before it reads, the same scheme CPython's
``WeakValueDictionary`` uses for its pending removals. A drained removal only
deletes the entry that still holds the exact dead reference, so a later entry
that reused the object's ``id`` survives.
"""

from __future__ import annotations

import threading
import weakref
from collections import deque
from collections.abc import Hashable
from typing import Generic, TypeVar

T = TypeVar("T")
V = TypeVar("V")
S = TypeVar("S")

_Removal = tuple[int, "weakref.ReferenceType[object]"]


class _WeakIdentityStore(Generic[T, V]):
    def __init__(self) -> None:
        self._entries: dict[int, tuple[weakref.ReferenceType[T], V]] = {}
        # Re-entrancy is defence in depth: no GC callback takes this lock.
        self._lock = threading.RLock()
        self._dead: deque[_Removal] = deque()

    def _reference(self, obj: T) -> weakref.ReferenceType[T]:
        key = id(obj)
        # Capture only the deque: the callback must not keep the owner alive
        # and must not block, allocate a lock, or touch the entries.
        dead = self._dead

        def discard(reference: weakref.ReferenceType[T]) -> None:
            dead.append((key, reference))

        return weakref.ref(obj, discard)

    def _drain(self) -> None:
        """Apply pending removals; the caller holds ``self._lock``."""

        dead = self._dead
        while True:
            try:
                key, reference = dead.popleft()
            except IndexError:
                return
            current = self._entries.get(key)
            if current is not None and current[0] is reference:
                del self._entries[key]

    def _live(self, obj: T) -> tuple[weakref.ReferenceType[T], V] | None:
        """The entry for ``obj`` or ``None``; drops a stale same-id entry."""

        key = id(obj)
        cached = self._entries.get(key)
        if cached is None:
            return None
        if cached[0]() is obj:
            return cached
        del self._entries[key]
        return None

    def clear(self) -> None:
        with self._lock:
            self._dead.clear()
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            self._drain()
            return len(self._entries)


class WeakIdentityCache(_WeakIdentityStore[T, V]):
    """One value per live object."""

    def peek(self, obj: T) -> V | None:
        with self._lock:
            self._drain()
            cached = self._live(obj)
            return None if cached is None else cached[1]

    def retain(self, obj: T, value: V) -> V:
        """Store ``value`` unless ``obj`` already has one; return the kept value."""

        reference = self._reference(obj)
        with self._lock:
            self._drain()
            cached = self._live(obj)
            if cached is not None:
                return cached[1]
            self._entries[id(obj)] = (reference, value)
            return value

    def take(self, obj: T) -> V | None:
        """Remove and return the value of ``obj``."""

        with self._lock:
            self._drain()
            cached = self._live(obj)
            if cached is None:
                return None
            del self._entries[id(obj)]
            return cached[1]


class WeakIdentityBuckets(_WeakIdentityStore[T, dict[Hashable, S]]):
    """Several values per live object, each under a hashable sub-key."""

    def get(self, obj: T, subkey: Hashable) -> S | None:
        with self._lock:
            self._drain()
            cached = self._live(obj)
            return None if cached is None else cached[1].get(subkey)

    def _bucket(self, obj: T) -> dict[Hashable, S]:
        cached = self._live(obj)
        if cached is not None:
            return cached[1]
        bucket: dict[Hashable, S] = {}
        self._entries[id(obj)] = (self._reference(obj), bucket)
        return bucket

    def setdefault(self, obj: T, subkey: Hashable, value: S) -> S:
        with self._lock:
            self._drain()
            return self._bucket(obj).setdefault(subkey, value)

    def set(self, obj: T, subkey: Hashable, value: S) -> None:
        with self._lock:
            self._drain()
            self._bucket(obj)[subkey] = value


__all__ = ["WeakIdentityBuckets", "WeakIdentityCache"]
