"""Deterministic scaling contract for prefix-aware large-fleet placement."""

from collections.abc import Callable, Sequence
from types import SimpleNamespace

import kairyu.orchestration.prefix_index as prefix_module
from kairyu.orchestration.prefix_index import PrefixIndex
from kairyu.orchestration.replica import ReplicaPool


def _legacy_hash_every_candidate(
    replica_ids: Sequence[str], prompt: str, chunk_keys: Callable[[str], object]
) -> None:
    """Model the previous candidate loop: one complete prompt pass per replica."""
    for _replica_id in replica_ids:
        chunk_keys(prompt)


def test_large_fleet_cold_decision_hashes_only_one_root_chunk(monkeypatch):
    replica_ids = tuple(f"replica-{index}" for index in range(1_000))
    prompt = "x" * (32 * 1024)
    index = PrefixIndex(chunk_chars=256)
    pool = ReplicaPool(
        {replica_id: object() for replica_id in replica_ids}, prefix_index=index
    )
    real_xxh3 = prefix_module.xxhash.xxh3_64
    hash_instances = 0
    chunk_updates = 0

    class CountingHash:
        def __init__(self, payload=b""):
            self._inner = real_xxh3(payload)
            if payload:
                self._record_update()

        def _record_update(self):
            nonlocal chunk_updates
            chunk_updates += 1

        def update(self, payload):
            self._record_update()
            return self._inner.update(payload)

        def hexdigest(self):
            return self._inner.hexdigest()

    def counted_xxh3(payload=b""):
        nonlocal hash_instances
        hash_instances += 1
        return CountingHash(payload)

    monkeypatch.setattr(
        prefix_module,
        "xxhash",
        SimpleNamespace(xxh3_64=counted_xxh3),
    )

    assert pool._prefix_select(replica_ids, prompt) is None
    assert hash_instances == 1
    assert chunk_updates == 1

    hash_instances = 0
    chunk_updates = 0
    _legacy_hash_every_candidate(replica_ids, prompt, index.chunk_keys)
    assert hash_instances == len(replica_ids) == 1_000
    assert chunk_updates == len(replica_ids) * (len(prompt) // index.chunk_chars)
