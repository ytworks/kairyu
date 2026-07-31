"""Contract coverage for the standalone KV event hash benchmark."""

from bench.kv_event_hash_bench import _incremental_hashes, _legacy_hashes


def test_incremental_benchmark_consumes_the_four_value_hash_chain_contract() -> None:
    tokens = tuple(range(-17, 47))

    assert _incremental_hashes(tokens, 4) == _legacy_hashes(tokens, 4)
