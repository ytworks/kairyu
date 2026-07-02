import json
import time

from kairyu.orchestration.features import extract_features
from kairyu.orchestration.router import JsonlRouterLog, RuleRouter

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


def test_reasoning_query_routes_to_tier2():
    assert RuleRouter().route(REASONING_QUERY).target == "tier2"


def test_code_query_routes_to_tier2():
    assert RuleRouter().route(CODE_QUERY).target == "tier2"


def test_multi_step_query_routes_to_multi_agent():
    assert RuleRouter().route(MULTI_STEP_QUERY).target == "multi_agent"


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
    line = json.loads(log_path.read_text().splitlines()[0])
    assert line["target"] == "tier1"
    assert line["features"]["char_len"] == len(SIMPLE_QUERY)
    assert "query_sha256" in line
    assert SIMPLE_QUERY not in json.dumps(line)  # raw text is not logged
