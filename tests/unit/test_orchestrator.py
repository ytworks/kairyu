import pytest

from kairyu.engine.mock import MockBackend
from kairyu.orchestration.orchestrator import Orchestrator

SIMPLE = "What is 2?"
COMPLEX = (
    "First, research the options and summarize trade-offs. Then design a plan. "
    "After that, implement it. Finally, verify everything works end to end."
)


def _orchestrator(**kwargs) -> Orchestrator:
    engines = kwargs.pop(
        "engines",
        {"tier1": MockBackend(), "tier2": MockBackend()},
    )
    return Orchestrator(engines=engines, **kwargs)


async def test_simple_query_goes_to_tier1_engine():
    tier1 = MockBackend(responses={SIMPLE: "two"})
    orchestrator = _orchestrator(engines={"tier1": tier1, "tier2": MockBackend()})
    result = await orchestrator.run(SIMPLE)
    assert result.route.target == "tier1"
    assert result.text == "two"
    assert len(tier1.prompts_seen) == 1


async def test_complex_query_uses_default_conductor_dag():
    tier1 = MockBackend()
    tier2 = MockBackend(responses={"[verifier]": "PASS"})
    orchestrator = _orchestrator(engines={"tier1": tier1, "tier2": tier2})
    result = await orchestrator.run(COMPLEX)
    assert result.route.target == "multi_agent"
    assert result.text
    assert len(tier1.prompts_seen) + len(tier2.prompts_seen) >= 3  # planner/worker/verifier/synth


async def test_missing_tier_falls_back_with_trace_note():
    only_engine = MockBackend()
    orchestrator = _orchestrator(engines={"tier1": only_engine})
    result = await orchestrator.run(
        "Prove the theorem and explain your reasoning step by step, derive it."
    )
    assert result.route.target == "tier2"
    assert result.text
    assert any("fallback" in note for note in result.trace)


def test_run_sync_wrapper():
    orchestrator = _orchestrator()
    result = orchestrator.run_sync(SIMPLE)
    assert result.text


def test_requires_at_least_one_engine():
    with pytest.raises(ValueError, match="engine"):
        Orchestrator(engines={})
