import json
import random
import time

from kairyu.orchestration.features import extract_features
from kairyu.orchestration.learning.bandit import GreedyLinearBandit
from kairyu.orchestration.learning.classifier import LearnedRouter, RouterModel, train_model
from kairyu.orchestration.learning.dataset import LabeledExample, build_dataset
from kairyu.orchestration.router import JsonlRouterLog, RuleRouter

SHORT = "What is the capital of France?"
CODE = "Fix this bug:\n```python\nx = [\n```\nwhy does it fail?"
MULTI = "First, research options. Then design a plan. After that, implement. Finally verify."


def _decision_record(query: str, target: str) -> dict:
    decision = RuleRouter().route(query)
    return {
        "kind": "decision",
        "query_sha256": query,  # tests use the raw query as a stand-in hash
        "target": target,
        "features": decision.features.as_dict(),
    }


def _outcome_record(query: str, target: str, quality: float, cost: float) -> dict:
    return {
        "kind": "outcome",
        "query_sha256": query,
        "target": target,
        "quality": quality,
        "cost_usd": cost,
    }


def test_build_dataset_picks_highest_utility_target():
    records = [
        _decision_record(SHORT, "tier1"),
        _outcome_record(SHORT, "tier1", quality=0.9, cost=0.001),
        _outcome_record(SHORT, "tier2", quality=0.95, cost=0.05),  # better quality, high cost
        _decision_record(CODE, "tier2"),
        _outcome_record(CODE, "tier2", quality=0.9, cost=0.05),
        _outcome_record(CODE, "tier1", quality=0.3, cost=0.001),
    ]
    dataset = build_dataset(records, cost_weight=10.0)
    labels = {example.query_hash: example.label for example in dataset}
    # tier1: 0.9 - 10*0.001 = 0.89 beats tier2: 0.95 - 10*0.05 = 0.45
    assert labels[SHORT] == "tier1"
    # tier2: 0.9 - 0.5 = 0.4 beats tier1: 0.3 - 0.01 = 0.29
    assert labels[CODE] == "tier2"


def _synthetic_examples(n: int = 300) -> list[LabeledExample]:
    rng = random.Random(7)
    examples = []
    for i in range(n):
        kind = rng.choice(["short", "code", "multi"])
        if kind == "short":
            query, label = f"What is {i}?", "tier1"
        elif kind == "code":
            query, label = f"Fix bug {i}:\n```python\nx = {i}\n```\nwhy?", "tier2"
        else:
            query, label = (
                f"First, research topic {i}. Then design. After that, build. Finally verify.",
                "multi_agent",
            )
        examples.append(
            LabeledExample(
                query_hash=str(i), features=extract_features(query).as_dict(), label=label
            )
        )
    return examples


def test_classifier_learns_separable_data_and_round_trips(tmp_path):
    examples = _synthetic_examples()
    train, held_out = examples[:240], examples[240:]
    model = train_model(train, epochs=200, seed=3)
    accuracy = sum(model.predict(e.features)[0] == e.label for e in held_out) / len(held_out)
    assert accuracy >= 0.9
    path = tmp_path / "router-model.json"
    model.save(path)
    loaded = RouterModel.load(path)
    assert loaded.predict(held_out[0].features) == model.predict(held_out[0].features)


def test_learned_router_implements_protocol_with_fallback():
    model = train_model(_synthetic_examples(), epochs=200, seed=3)
    router = LearnedRouter(model=model, fallback=RuleRouter(), min_confidence=0.34)
    decision = router.route(SHORT)
    assert decision.target == "tier1"
    assert 0.0 <= decision.confidence <= 1.0
    # an impossible confidence bar always defers to the fallback rule router
    strict = LearnedRouter(model=model, fallback=RuleRouter(), min_confidence=1.01)
    assert strict.route(CODE).target == RuleRouter().route(CODE).target
    assert "fallback" in strict.route(CODE).reason
    assert router.preview(SHORT) == router.route(SHORT)
    assert router.describe() == {
        "router_type": "LearnedRouter",
        "min_confidence": 0.34,
        "fallback_type": "RuleRouter",
    }


def test_learned_router_p99_latency_under_10ms():
    model = train_model(_synthetic_examples(120), epochs=50, seed=3)
    router = LearnedRouter(model=model, fallback=RuleRouter())
    durations = []
    for _ in range(500):
        start = time.perf_counter()
        router.route(CODE)
        durations.append(time.perf_counter() - start)
    durations.sort()
    assert durations[int(len(durations) * 0.99)] < 0.010


def test_learned_preview_rejects_route_only_fallback():
    class RouteOnly:
        def route(self, query, context=None):
            raise AssertionError("preview must not call route")

    model = train_model(_synthetic_examples(60), epochs=20, seed=3)
    router = LearnedRouter(model=model, fallback=RouteOnly(), min_confidence=1.01)
    import pytest as _pytest

    with _pytest.raises(NotImplementedError, match="does not support preview"):
        router.preview(CODE)


def test_bandit_converges_to_context_optimal_arm():
    bandit = GreedyLinearBandit(epsilon=0.1, seed=11)
    rng = random.Random(5)

    def reward(query: str, target: str) -> float:
        features = extract_features(query)
        best = "tier2" if features.has_code_fence else "tier1"
        return 1.0 if target == best else 0.0

    for i in range(600):
        query = CODE if rng.random() < 0.5 else f"What is {i}?"
        features = extract_features(query)
        target = bandit.select(features)
        bandit.update(features, target, reward(query, target))

    exploit = GreedyLinearBandit(epsilon=0.0, seed=11, weights=bandit.weights)
    assert exploit.select(extract_features(CODE)) == "tier2"
    assert exploit.select(extract_features("What is 2?")) == "tier1"


def test_outcome_logging_joins_with_decisions(tmp_path):
    log_path = tmp_path / "router.jsonl"
    log = JsonlRouterLog(log_path)
    decision = RuleRouter().route(SHORT)
    log.record(SHORT, decision)
    log.record_outcome(SHORT, target=decision.target, quality=0.9, cost_usd=0.001)
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records[0]["kind"] == "decision"
    assert records[1]["kind"] == "outcome"
    assert records[0]["query_sha256"] == records[1]["query_sha256"]
    dataset = build_dataset(records, cost_weight=1.0)
    assert dataset[0].label == decision.target


def test_bandit_exploration_rate_is_honored():
    bandit = GreedyLinearBandit(epsilon=0.3, seed=1)
    features = extract_features("a plain query with no signals")
    picks = [bandit.select(features) for _ in range(2000)]
    # untrained weights tie at 0 -> exploitation always picks the first arm;
    # only exploration (eps * 2/3) produces non-tier1 picks
    explore_fraction = sum(1 for pick in picks if pick != "tier1") / len(picks)
    assert 0.10 < explore_fraction < 0.30


def test_bandit_rejects_malformed_saved_weights():
    import pytest as _pytest

    with _pytest.raises(ValueError, match="missing arms"):
        GreedyLinearBandit(weights={"tier1": [0.0] * 8})
    with _pytest.raises(ValueError, match="dimension"):
        GreedyLinearBandit(
            weights={"tier1": [0.0] * 3, "tier2": [0.0] * 8, "multi_agent": [0.0] * 8}
        )


def test_bandit_router_defers_to_base_until_warm():
    from kairyu.orchestration.learning.bandit import BanditRouter

    router = BanditRouter(
        GreedyLinearBandit(epsilon=0.0, seed=2), base=RuleRouter(), min_updates_per_arm=3
    )
    cold = router.route(SHORT)
    assert cold.target == RuleRouter().route(SHORT).target
    assert "cold_start" in cold.reason
    for arm in ("tier1", "tier2", "multi_agent"):
        for _ in range(3):
            router.record_reward(SHORT, arm, reward=1.0 if arm == "tier1" else 0.0)
    warm = router.route(SHORT)
    assert warm.reason == "bandit"
    assert warm.target == "tier1"


def test_bandit_preview_does_not_advance_rng_or_mutate_learning_state():
    from kairyu.orchestration.learning.bandit import BanditRouter

    bandit = GreedyLinearBandit(epsilon=1.0, seed=23)
    router = BanditRouter(bandit, base=RuleRouter(), min_updates_per_arm=0)
    rng_before = bandit._rng.getstate()
    weights_before = bandit.weights
    counts_before = dict(router._update_counts)

    preview = router.preview(CODE)

    assert preview.reason == "bandit:preview"
    assert bandit._rng.getstate() == rng_before
    assert bandit.weights == weights_before
    assert router._update_counts == counts_before
    assert router.route(CODE).target == preview.target
    assert router.describe() == {
        "router_type": "BanditRouter",
        "epsilon": 1.0,
        "is_warm": True,
        "min_updates_per_arm": 0,
        "fallback_type": "RuleRouter",
    }


def test_record_outcome_validates_ranges(tmp_path):
    import pytest as _pytest

    log = JsonlRouterLog(tmp_path / "router.jsonl")
    with _pytest.raises(ValueError, match="quality"):
        log.record_outcome("q", target="tier1", quality=1.5, cost_usd=0.1)
    with _pytest.raises(ValueError, match="cost_usd"):
        log.record_outcome("q", target="tier1", quality=0.5, cost_usd=-0.1)
