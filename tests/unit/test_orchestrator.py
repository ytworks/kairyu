import pytest

from kairyu.engine.mock import MockBackend
from kairyu.orchestration.orchestrator import Orchestrator

SIMPLE = "What is 2?"
COMPLEX = (
    "First, research the options and summarize trade-offs. Then design a plan. "
    "After that, implement it. Finally, verify everything works end to end."
)


class _ShutdownBackend(MockBackend):
    def __init__(self) -> None:
        super().__init__()
        self.shutdown_count = 0

    async def shutdown(self) -> None:
        self.shutdown_count += 1


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


async def test_moa_tier_charges_cost_model_and_reports_budget(tmp_path):
    # M3: the deep MoA tier must invoke the cost model and surface a budget
    # overrun in the trace, instead of being invisible to max_cost_usd.
    from kairyu.orchestration.budget import Budget
    from kairyu.orchestration.conductor import chars_cost_model

    orchestrator = _orchestrator(
        moa_samples=2,
        cost_model=chars_cost_model(usd_per_1k_chars=1000.0),  # huge -> exceeds
        budget=Budget(max_cost_usd=0.001),
    )
    result = await orchestrator.run(COMPLEX)
    assert any("moa:" in line and "cost=" in line for line in result.trace)
    assert any("budget exceeded" in line for line in result.trace)


async def test_multistage_stream_emits_periodic_keepalives(monkeypatch):
    # M8: a long multi-stage run must emit PERIODIC status keep-alives, not one
    # status then silence (which a proxy idle timeout would sever).
    import kairyu.orchestration.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "_KEEPALIVE_INTERVAL_S", 0.01)
    # backends with latency so the multi-agent run outlasts several keepalives
    slow = {"tier1": MockBackend(latency_s=0.03), "tier2": MockBackend(latency_s=0.03)}
    orchestrator = _orchestrator(engines=slow)
    events = [event async for event in await orchestrator.run_chat(COMPLEX, stream=True)]
    kinds = [e.kind for e in events]
    assert kinds.count("status") >= 2  # routing + at least one "working" keepalive
    assert kinds[-1] == "result"
    assert any(e.kind == "delta" for e in events)


async def test_missing_tier_falls_back_with_trace_note():
    only_engine = MockBackend()
    orchestrator = _orchestrator(engines={"tier1": only_engine})
    result = await orchestrator.run(
        "Prove the theorem and explain your reasoning step by step, derive it."
    )
    assert result.route.target == "tier2"
    assert result.text
    assert any("fallback" in note for note in result.trace)


async def test_shutdown_closes_each_owned_engine_once():
    shared = _ShutdownBackend()
    orchestrator = Orchestrator(engines={"tier1": shared, "tier2": shared})
    await orchestrator.shutdown()
    assert shared.shutdown_count == 1


def test_run_sync_wrapper():
    orchestrator = _orchestrator()
    result = orchestrator.run_sync(SIMPLE)
    assert result.text


def test_requires_at_least_one_engine():
    with pytest.raises(ValueError, match="engine"):
        Orchestrator(engines={})
