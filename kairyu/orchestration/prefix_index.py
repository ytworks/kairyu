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
from collections import OrderedDict
from collections.abc import Sequence

_DEFAULT_CHUNK_CHARS = 256
_DEFAULT_MAX_CHUNKS_PER_REPLICA = 4096


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
        self._chunks: dict[str, OrderedDict[str, None]] = {}
        self._replicas_by_first_key: dict[str, dict[str, None]] = {}

    def observe(self, replica_id: str, prompt: str) -> None:
        store = self._chunks.setdefault(replica_id, OrderedDict())
        # Prefix keys are cumulative: a retained child is unusable when an
        # earlier key in the same chain was evicted because overlap stops at
        # the first miss.  Admit at most one store's worth from the root side
        # so a single long prompt leaves a reachable prefix, not unreachable
        # tail keys.
        keys = self.chunk_keys(prompt)[: self._max_chunks]
        if keys:
            self._replicas_by_first_key.setdefault(keys[0], {})[replica_id] = None
        for key in keys:
            store.pop(key, None)  # refresh recency
            store[key] = None
        while len(store) > self._max_chunks:
            evicted, _value = store.popitem(last=False)
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
        candidates.pop(replica_id, None)
        if not candidates:
            del self._replicas_by_first_key[key]
