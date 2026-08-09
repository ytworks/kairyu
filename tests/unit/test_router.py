import hashlib
import json
import threading
import time

import pytest

import kairyu.orchestration.router as router_module
from kairyu.orchestration.features import extract_features
from kairyu.orchestration.router import (
    JsonlRouterLog,
    RuleRouter,
    load_calibrated_router,
)

SIMPLE_QUERY = "What is the capital of France?"
REASONING_QUERY = (
    "Prove that the sum of two even numbers is even. "
    "Explain your reasoning step by step and derive the general case."
)
CODE_QUERY = "Fix this bug:\n```python\ndef f(x):\n    return x +\n```\nwhy does it fail?"
MULTI_STEP_QUERY = (
    "First, research the top five vector databases and summarize their trade-offs. "
    "Then design a benchmark plan comparing them on our workload. "
    "After that, draft the implementation outline. "
    "Finally, write a risk assessment and a rollout plan for the migration. " * 3
)


def test_extract_features_is_pure_and_counts_signals():
    features = extract_features(CODE_QUERY)
    assert features.has_code_fence is True
    assert features.char_len == len(CODE_QUERY)
    assert features.question_count == 1
    again = extract_features(CODE_QUERY)
    assert again == features


def test_simple_query_routes_to_tier1():
    decision = RuleRouter().route(SIMPLE_QUERY)
    assert decision.target == "tier1"
    assert decision.reason


def test_rule_router_preview_matches_route_and_describes_thresholds():
    router = RuleRouter()
    assert router.preview(REASONING_QUERY) == router.route(REASONING_QUERY)
    descriptor = router.describe()
    assert descriptor["router_type"] == "RuleRouter"
    assert descriptor["thresholds"]["tier2_min_chars"] == 600


def test_reasoning_query_routes_to_tier2():
    assert RuleRouter().route(REASONING_QUERY).target == "tier2"


def test_code_query_routes_to_tier2():
    assert RuleRouter().route(CODE_QUERY).target == "tier2"


def test_multi_step_query_routes_to_multi_agent():
    assert RuleRouter().route(MULTI_STEP_QUERY).target == "multi_agent"


def test_calibrated_router_rejects_hash_and_quality_gate(tmp_path) -> None:
    path = tmp_path / "router.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "kairyu-calibrated-rule-router",
                "artifact_id": "unsafe",
                "quality_ci_lower": 0.98,
                "tier1_max_input_tokens": 262_144,
                "tier1_max_input_chars": 262_144,
                "train_split_sha256": "a" * 64,
                "holdout_split_sha256": "b" * 64,
                "thresholds": {},
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_calibrated_router(path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="0.99 quality"):
        load_calibrated_router(path, expected_sha256=digest)


def test_routing_p99_latency_under_10ms():
    router = RuleRouter()
    queries = [SIMPLE_QUERY, REASONING_QUERY, CODE_QUERY, MULTI_STEP_QUERY] * 250
    durations = []
    for query in queries:
        start = time.perf_counter()
        router.route(query)
        durations.append(time.perf_counter() - start)
    durations.sort()
    p99 = durations[int(len(durations) * 0.99)]
    assert p99 < 0.010, f"router p99 {p99 * 1000:.3f}ms exceeds 10ms"


def test_jsonl_router_log_records_decision(tmp_path):
    log_path = tmp_path / "router.jsonl"
    log = JsonlRouterLog(log_path)
    decision = RuleRouter().route(SIMPLE_QUERY)
    log.record(SIMPLE_QUERY, decision)
    log.flush()
    line = json.loads(log_path.read_text().splitlines()[0])
    assert line["target"] == "tier1"
    assert line["features"]["char_len"] == len(SIMPLE_QUERY)
    assert "query_sha256" in line
    assert SIMPLE_QUERY not in json.dumps(line)  # raw text is not logged
    log.close()


def test_jsonl_router_log_defers_encoding_to_writer_thread(tmp_path, monkeypatch):
    producer_thread = threading.get_ident()
    encoder_threads: list[int] = []
    original_dumps = router_module.json.dumps

    def tracked_dumps(value):
        encoder_threads.append(threading.get_ident())
        return original_dumps(value)

    monkeypatch.setattr(router_module.json, "dumps", tracked_dumps)
    log = JsonlRouterLog(tmp_path / "router.jsonl")
    log.record(SIMPLE_QUERY, RuleRouter().route(SIMPLE_QUERY))
    log.close()

    assert encoder_threads
    assert all(thread_id != producer_thread for thread_id in encoder_threads)


def test_record_replica_hashes_session_id(tmp_path):
    import hashlib

    log_path = tmp_path / "router.jsonl"
    log = JsonlRouterLog(log_path)
    log.record_replica("session-42", 1, "session_affinity")
    log.flush()
    line = json.loads(log_path.read_text().splitlines()[0])
    assert line["kind"] == "replica"
    assert line["session_sha256"] == hashlib.sha256(b"session-42").hexdigest()
    assert line["replica"] == 1
    assert line["reason"] == "session_affinity"
    assert "session-42" not in log_path.read_text()  # raw session id is never stored
    log.close()


def test_record_replica_without_session_logs_null_hash(tmp_path):
    log_path = tmp_path / "router.jsonl"
    log = JsonlRouterLog(log_path)
    log.record_replica(None, 0, "least_outstanding")
    log.flush()
    line = json.loads(log_path.read_text().splitlines()[0])
    assert line["kind"] == "replica"
    assert line["session_sha256"] is None
    assert line["replica"] == 0
    log.close()


def test_record_replica_entries_are_ignored_by_dataset_builder(tmp_path):
    from kairyu.orchestration.learning.dataset import build_dataset

    log_path = tmp_path / "router.jsonl"
    log = JsonlRouterLog(log_path)
    log.record_replica("session-42", 1, "session_affinity")
    log.flush()
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert build_dataset(records) == ()  # kind filter keeps the corpus clean (m5 D4)
    log.close()
