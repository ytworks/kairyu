"""Deterministic scaling contract for prefix-aware large-fleet placement."""

from collections.abc import Callable, Sequence

import kairyu.orchestration.prefix_index as prefix_module
from kairyu.orchestration.prefix_index import PrefixIndex
from kairyu.orchestration.replica import ReplicaPool


def _legacy_hash_every_candidate(
    replica_ids: Sequence[str], prompt: str, chunk_keys: Callable[[str], object]
) -> None:
    """Model the previous candidate loop: one complete prompt pass per replica."""
    for _replica_id in replica_ids:
        chunk_keys(prompt)


def test_large_fleet_prefix_decision_hashes_prompt_once(monkeypatch):
    replica_ids = tuple(f"replica-{index}" for index in range(1_000))
    prompt = "x" * (32 * 1024)
    index = PrefixIndex(chunk_chars=256)
    pool = ReplicaPool(
        {replica_id: object() for replica_id in replica_ids}, prefix_index=index
    )
    real_prompt_chunks = prefix_module.prompt_chunks
    hash_passes = 0

    def counted_prompt_chunks(text: str, chunk_chars: int = 256):
        nonlocal hash_passes
        hash_passes += 1
        return real_prompt_chunks(text, chunk_chars)

    monkeypatch.setattr(prefix_module, "prompt_chunks", counted_prompt_chunks)

    assert pool._prefix_select(replica_ids, prompt) is None
    assert hash_passes == 1

    hash_passes = 0
    _legacy_hash_every_candidate(replica_ids, prompt, index.chunk_keys)
    assert hash_passes == len(replica_ids) == 1_000
