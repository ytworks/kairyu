"""Approximate prefix index for KV-aware placement (m10b D6/A12).

The gateway has NO token ids (prompts are strings; tokenizers live in the
optional hf extra), so the trie keys on fixed-size TEXT chunks of the prompt
— an approximation of the engine-side token pages. Key unification via
gateway tokenization is a deploy-time option (install tokenizers in the
gateway image). ``observe`` is called only after successful generation (the
replica now holds that prefix); ``overlap`` scores candidates at placement
time.

Bounded: per-replica chunk sets are LRU-capped so a long-running gateway
cannot grow without bound. A reverse map for the first cumulative key is
bounded by those same stores and avoids scanning cold replicas.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from itertools import islice

_DEFAULT_CHUNK_CHARS = 256
_DEFAULT_MAX_CHUNKS_PER_REPLICA = 4096


class PreparedPrefixKeys(Sequence[str]):
    """Request-local, lazily extended cumulative prefix hashes.

    The first key is enough to prove that a request is cold.  Deeper keys are
    generated only if that root has a candidate, and the same object is then
    consumed by successful publication.  Each complete prompt chunk is hashed
    at most once; there is no process-global prompt cache to coordinate or
    invalidate between concurrent requests.
    """

    __slots__ = (
        "_chunk_chars",
        "candidate_ids",
        "first_key",
        "_hasher",
        "_keys",
        "_length",
        "_prompt",
    )

    def __init__(
        self,
        prompt: str,
        chunk_chars: int,
        max_chunks: int,
        hasher,
        first_key: str,
        candidate_ids: tuple[str, ...],
    ) -> None:
        self._prompt = prompt
        self._chunk_chars = chunk_chars
        self._length = min(len(prompt) // chunk_chars, max_chunks)
        self._hasher = hasher
        self.first_key = first_key
        self._keys: list[str] | None = None
        self.candidate_ids = candidate_ids

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int | slice) -> str | tuple[str, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(self._length)))
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        if index == 0:
            assert self.first_key is not None
            return self.first_key
        self._ensure(index + 1)
        assert self._keys is not None
        return self._keys[index]

    def __iter__(self):
        if self._length:
            assert self.first_key is not None
            yield self.first_key
        for index in range(1, self._length):
            yield self[index]

    def _ensure(self, count: int) -> None:
        if count <= 1:
            return
        if self._hasher is None:
            return
        if self._keys is None:
            assert self.first_key is not None
            self._keys = [self.first_key]
        while len(self._keys) < count:
            start = len(self._keys) * self._chunk_chars
            end = start + self._chunk_chars
            self._hasher.update(self._prompt[start:end].encode())
            self._keys.append(self._hasher.hexdigest()[:16])


def prompt_chunks(prompt: str, chunk_chars: int = _DEFAULT_CHUNK_CHARS) -> tuple[str, ...]:
    """Prefix-chained chunk keys: chunk i hashes chars [0 : (i+1)*chunk).

    Fed incrementally into ONE streaming sha256 (P5): key i is the digest after
    feeding chunks 0..i. This is byte-identical to ``sha256(prompt[:end])`` —
    sha256 is a streaming hash and str slicing never splits a character — but
    O(L) total instead of re-hashing the whole prefix per chunk (O(L²)).
    """
    keys: list[str] = []
    hasher = hashlib.sha256()
    pos = 0
    for end in range(chunk_chars, len(prompt) + 1, chunk_chars):
        hasher.update(prompt[pos:end].encode())
        keys.append(hasher.hexdigest()[:16])
        pos = end
    return tuple(keys)


class PrefixIndex:
    """Per-replica sets of prefix-chunk keys with LRU capping."""

    def __init__(
        self,
        chunk_chars: int = _DEFAULT_CHUNK_CHARS,
        max_chunks_per_replica: int = _DEFAULT_MAX_CHUNKS_PER_REPLICA,
    ) -> None:
        if chunk_chars < 1:
            raise ValueError(f"chunk_chars must be >= 1, got {chunk_chars}")
        if max_chunks_per_replica < 1:
            raise ValueError(
                "max_chunks_per_replica must be >= 1, "
                f"got {max_chunks_per_replica}"
            )
        self.chunk_chars = chunk_chars
        self._max_chunks = max_chunks_per_replica
        self._chunks: dict[str, dict[str, None]] = {}
        # Most roots live on one replica. Keep that common case allocation-free
        # and promote to an ordered candidate dict only on the second replica.
        self._replicas_by_first_key: dict[
            str,
            str | dict[str, None],
        ] = {}

    def observe(self, replica_id: str, prompt: str) -> None:
        """Publish every usable prefix key after a successful generation."""
        # Preserve the original subclass seam: custom ``chunk_keys`` and
        # ``observe_keys`` overrides must remain authoritative.
        self.observe_keys(replica_id, self.chunk_keys(prompt))

    def observe_keys(
        self,
        replica_id: str,
        keys: Sequence[str],
        max_keys: int | None = None,
    ) -> None:
        """Publish caller-prepared keys, optionally admitting only the root.

        ``max_keys=1`` is used for a cold placement: the root is sufficient to
        route a later shared-prefix request back to the backend that now owns
        the complete KV state.  A successful warm route promotes the entry by
        publishing every usable key.
        """
        store = self._chunks.setdefault(replica_id, {})
        # Prefix keys are cumulative: a retained child is unusable when an
        # earlier key in the same chain was evicted because overlap stops at
        # the first miss.  Admit at most one store's worth from the root side
        # so a single long prompt leaves a reachable prefix, not unreachable
        # tail keys.
        admitted = self._max_chunks if max_keys is None else min(max_keys, self._max_chunks)
        for index, key in enumerate(islice(keys, admitted)):
            if index == 0:
                candidates = self._replicas_by_first_key.get(key)
                if candidates is None:
                    self._replicas_by_first_key[key] = replica_id
                elif isinstance(candidates, str):
                    if candidates != replica_id:
                        self._replicas_by_first_key[key] = {
                            candidates: None,
                            replica_id: None,
                        }
                else:
                    candidates[replica_id] = None
            store.pop(key, None)  # refresh recency
            store[key] = None
        while len(store) > self._max_chunks:
            evicted = next(iter(store))
            store.pop(evicted)
            self._discard_candidate(evicted, replica_id)

    def prepare(self, prompt: str) -> str | PreparedPrefixKeys:
        """Prepare one request's root lookup and lazy deeper hash chain."""
        if len(prompt) < self.chunk_chars:
            return ""
        hasher = hashlib.sha256(prompt[: self.chunk_chars].encode())
        first_key = hasher.hexdigest()[:16]
        candidates = self._replicas_by_first_key.get(first_key)
        if candidates is None:
            # The common cold path needs only its root at successful completion.
            return first_key
        candidate_ids = (
            (candidates,)
            if isinstance(candidates, str)
            else tuple(candidates)
        )
        return PreparedPrefixKeys(
            prompt,
            self.chunk_chars,
            self._max_chunks,
            hasher,
            first_key,
            candidate_ids,
        )

    def observe_prepared(
        self,
        replica_id: str,
        keys: str | PreparedPrefixKeys,
        promote: bool,
    ) -> None:
        """Publish native prepared keys, specializing the common cold miss."""
        if promote and isinstance(keys, PreparedPrefixKeys):
            self.observe_keys(replica_id, keys)
            return
        key = keys if isinstance(keys, str) else keys.first_key
        if not key:
            return
        if isinstance(keys, PreparedPrefixKeys) and keys.candidate_ids:
            # A negative warm score can still fall through to cold placement.
            # Preserve refresh/eviction behavior when the root already exists.
            self.observe_keys(replica_id, keys, 1)
            return
        store = self._chunks.get(replica_id)
        if store is None:
            store = {}
            self._chunks[replica_id] = store
        candidates = self._replicas_by_first_key.get(key)
        if candidates is None:
            self._replicas_by_first_key[key] = replica_id
            store[key] = None
        elif isinstance(candidates, str):
            if candidates != replica_id:
                self._replicas_by_first_key[key] = {
                    candidates: None,
                    replica_id: None,
                }
            store.pop(key, None)
            store[key] = None
        else:
            # Another same-prefix request may have completed while this request
            # awaited its backend. Merge instead of overwriting that candidate.
            candidates[replica_id] = None
            store.pop(key, None)
            store[key] = None
        if len(store) > self._max_chunks:
            evicted = next(iter(store))
            store.pop(evicted)
            self._discard_candidate(evicted, replica_id)

    def chunk_keys(self, prompt: str) -> tuple[str, ...]:
        """Return the immutable prefix keys for this index's chunk size."""
        return prompt_chunks(prompt, self.chunk_chars)

    def overlap(self, replica_id: str, prompt: str) -> int:
        """Longest known prefix, in chunks (prefix-chained: stop at first miss)."""
        return self.overlap_keys(replica_id, self.chunk_keys(prompt))

    def overlap_keys(self, replica_id: str, keys: Sequence[str]) -> int:
        """Longest known prefix from a caller-owned, precomputed key sequence."""
        store = self._chunks.get(replica_id)
        if not store:
            return 0
        count = 0
        for key in keys:
            if key not in store:
                break
            count += 1
        return count

    def candidate_ids(self, keys: Sequence[str]) -> tuple[str, ...]:
        """Replica IDs retaining the first cumulative key, without a fleet scan."""
        if not keys:
            return ()
        candidates = self._replicas_by_first_key.get(keys[0])
        if isinstance(candidates, str):
            return (candidates,)
        return tuple(candidates) if candidates is not None else ()

    def forget_replica(self, replica_id: str) -> None:
        store = self._chunks.pop(replica_id, None)
        if store is None:
            return
        for key in store:
            self._discard_candidate(key, replica_id)

    def _discard_candidate(self, key: str, replica_id: str) -> None:
        candidates = self._replicas_by_first_key.get(key)
        if candidates is None:
            return
        if isinstance(candidates, str):
            if candidates == replica_id:
                del self._replicas_by_first_key[key]
            return
        candidates.pop(replica_id, None)
        if not candidates:
            del self._replicas_by_first_key[key]
        elif len(candidates) == 1:
            self._replicas_by_first_key[key] = next(iter(candidates))
