import asyncio

import pytest

from kairyu.engine.mock import MockBackend
from kairyu.orchestration.budget import Budget, BudgetState
from kairyu.orchestration.moa import MoAResult
from kairyu.orchestration.orchestrator import (
    EngineDescriptor,
    Orchestrator,
    PreviewNotSupportedError,
)

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


def test_preview_route_is_non_dispatching_and_describes_effective_fallback():
    only = MockBackend()
    orchestrator = Orchestrator(
        engines={"worker": only},
        engine_descriptors={
            "worker": EngineDescriptor(backend_type="openai", model="small")
        },
    )

    decision = orchestrator.preview_route(SIMPLE)
    descriptor = orchestrator.describe_routing()

    assert decision.target == "tier1"
    assert only.prompts_seen == ()
    assert descriptor["configured_engines"] == {
        "worker": {"backend_type": "openai", "model": "small"}
    }
    assert descriptor["target_resolution"]["tier1"] == {
        "configured": False,
        "engine": "worker",
        "fallback": True,
    }
    assert descriptor["router"]["thresholds"]["multi_agent_min_chars"] == 2000
    assert all("prompt" not in role for role in descriptor["roles"])


def test_preview_route_rejects_router_without_preview():
    class RouteOnly:
        def route(self, query, context=None):
            return _orchestrator().preview_route(query)

    orchestrator = _orchestrator(router=RouteOnly())
    with pytest.raises(PreviewNotSupportedError, match="does not support preview"):
        orchestrator.preview_route(SIMPLE)


def _track_budget_releases(monkeypatch):
    releases = []
    original_release = BudgetState.release

    def tracked_release(self, steps=1, *, unknown_cost=False):
        released = original_release(
            self,
            steps=steps,
            unknown_cost=unknown_cost,
        )
        releases.append((self, steps, unknown_cost, released))
        return released

    monkeypatch.setattr(BudgetState, "release", tracked_release)
    return releases


async def test_simple_query_goes_to_tier1_engine():
    tier1 = MockBackend(responses={SIMPLE: "two"})
    orchestrator = _orchestrator(engines={"tier1": tier1, "tier2": MockBackend()})
    result = await orchestrator.run(SIMPLE)
    assert result.route.target == "tier1"
    assert result.text == "two"
    assert len(tier1.prompts_seen) == 1
    assert result.structured_trace is not None
    payload = result.structured_trace.as_dict()
    assert payload["trace_version"] == "2.0"
    assert [event["kind"] for event in payload["events"]] == [
        "routing",
        "generation",
    ]
    assert payload["events"][0]["detail"]["target"] == "tier1"
    assert payload["events"][1]["engine"] == "tier1"
    assert payload["events"][1]["status"] == "success"
    assert SIMPLE not in str(payload)


async def test_complex_query_uses_default_conductor_dag():
    tier1 = MockBackend()
    tier2 = MockBackend(responses={"[verifier]": "PASS"})
    orchestrator = _orchestrator(engines={"tier1": tier1, "tier2": tier2})
    result = await orchestrator.run(COMPLEX)
    assert result.route.target == "multi_agent"
    assert result.text
    assert len(tier1.prompts_seen) + len(tier2.prompts_seen) >= 3  # planner/worker/verifier/synth
    assert result.structured_trace is not None
    payload = result.structured_trace.as_dict()
    events = payload["events"]
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert events[0]["kind"] == "routing"
    role_events = [event for event in events if event["role"] != "router"]
    assert {event["node"] for event in role_events} >= {
        "planner",
        "worker",
        "verifier",
        "synthesizer",
    }
    assert all(event["timing"]["completed_at"] for event in role_events)
    assert COMPLEX not in str(payload)


async def test_moa_tier_charges_cost_model_and_reports_budget(tmp_path):
    # M3: the deep MoA tier must invoke the cost model and surface a budget
    # overrun in the trace, instead of being invisible to max_cost_usd.
    from kairyu.orchestration.conductor import chars_cost_model

    orchestrator = _orchestrator(
        moa_samples=2,
        cost_model=chars_cost_model(usd_per_1k_chars=1000.0),  # huge -> exceeds
        budget=Budget(max_cost_usd=0.001),
    )
    result = await orchestrator.run(COMPLEX)
    assert any("moa:" in line and "cost=" in line for line in result.trace)
    assert any("budget exceeded" in line for line in result.trace)


@pytest.mark.parametrize(
    "budget",
    (
        Budget(max_steps=2),
        Budget(max_steps=3, max_cost_usd=0.0),
    ),
    ids=("insufficient-steps", "cost-slot-unavailable"),
)
async def test_moa_budget_refusal_skips_without_dispatch(monkeypatch, budget):
    import kairyu.orchestration.moa as moa_module

    async def unexpected_run_moa(*args, **kwargs):
        pytest.fail("run_moa must not be called after budget refusal")

    monkeypatch.setattr(moa_module, "run_moa", unexpected_run_moa)
    orchestrator = _orchestrator(moa_samples=2, budget=budget)

    result = await orchestrator.run(COMPLEX)

    assert result.text == ""
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.trace[-1] == "moa: skipped:budget"


async def test_moa_success_reconciles_full_reservation_once(monkeypatch):
    import kairyu.orchestration.moa as moa_module

    events = []
    reconciled_states = []
    original_try_reserve = BudgetState.try_reserve
    original_reconcile_success = BudgetState.reconcile_success

    def tracked_try_reserve(self, steps=1, *, unknown_cost=False):
        reserved = original_try_reserve(
            self,
            steps=steps,
            unknown_cost=unknown_cost,
        )
        events.append(("reserve", steps, unknown_cost, reserved))
        return reserved

    def tracked_reconcile_success(
        self,
        steps=1,
        cost=0.0,
        *,
        unknown_cost=False,
    ):
        events.append(("reconcile", steps, unknown_cost, self))
        reconciled = original_reconcile_success(
            self,
            steps=steps,
            cost=cost,
            unknown_cost=unknown_cost,
        )
        reconciled_states.append(reconciled)
        return reconciled

    async def fake_run_moa(*args, **kwargs):
        events.append(("dispatch",))
        return MoAResult(
            final_text="synthesized",
            proposals=("proposal one", "proposal two"),
            usage=(7, 3),
        )

    monkeypatch.setattr(BudgetState, "try_reserve", tracked_try_reserve)
    monkeypatch.setattr(BudgetState, "reconcile_success", tracked_reconcile_success)
    monkeypatch.setattr(moa_module, "run_moa", fake_run_moa)
    orchestrator = _orchestrator(
        moa_samples=2,
        budget=Budget(max_steps=4, max_cost_usd=1.0),
        cost_model=lambda request, result: 0.25,
    )

    result = await orchestrator.run(COMPLEX)

    assert [event[0] for event in events] == ["reserve", "dispatch", "reconcile"]
    assert events[0][1:3] == (3, True)
    reserved = events[0][3]
    assert reserved is not None
    assert reserved.steps_reserved == 3
    assert reserved.unknown_cost_reserved is True
    assert events[2][1:3] == (3, True)
    assert reconciled_states == [
        BudgetState(
            budget=Budget(max_steps=4, max_cost_usd=1.0),
            steps_used=3,
            cost_used=0.25,
        )
    ]
    assert result.text == "synthesized"
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 3
    assert "moa: 2 proposals synthesized (cost=0.2500)" in result.trace
    assert "moa: budget exceeded" not in result.trace
    assert result.structured_trace is not None
    moa_event = result.structured_trace.as_dict()["events"][-1]
    assert moa_event["engine"] == "tier1,tier2"
    assert moa_event["detail"]["proposal_engine"] == "tier1"
    assert moa_event["detail"]["synthesizer_engine"] == "tier2"


async def test_moa_trace_records_resolved_fallback_engine_and_model(monkeypatch):
    import kairyu.orchestration.moa as moa_module

    shared = MockBackend()

    async def fake_run_moa(proposal_engine, *args, **kwargs):
        assert proposal_engine is shared
        assert kwargs["synthesizer"] is shared
        return MoAResult(
            final_text="shared result",
            proposals=("one", "two"),
            usage=(5, 2),
        )

    monkeypatch.setattr(moa_module, "run_moa", fake_run_moa)
    orchestrator = _orchestrator(
        engines={"shared": shared},
        engine_descriptors={
            "shared": EngineDescriptor(
                backend_type="mock",
                model="shared-model",
            )
        },
        moa_samples=2,
    )

    result = await orchestrator.run(COMPLEX)

    assert result.structured_trace is not None
    moa_event = result.structured_trace.as_dict()["events"][-1]
    assert moa_event["engine"] == "shared"
    assert moa_event["model"] == "shared-model"
    detail = moa_event["detail"]
    assert (
        detail["proposal_engine"],
        detail["proposal_model"],
        detail["synthesizer_engine"],
        detail["synthesizer_model"],
    ) == ("shared", "shared-model", "shared", "shared-model")


async def test_moa_failure_releases_full_reservation_and_reraises(monkeypatch):
    import kairyu.orchestration.moa as moa_module

    async def failing_run_moa(*args, **kwargs):
        raise RuntimeError("moa failed")

    releases = _track_budget_releases(monkeypatch)
    monkeypatch.setattr(moa_module, "run_moa", failing_run_moa)
    orchestrator = _orchestrator(
        moa_samples=2,
        budget=Budget(max_steps=3, max_cost_usd=1.0),
    )

    with pytest.raises(RuntimeError, match="moa failed"):
        await orchestrator.run(COMPLEX)

    assert len(releases) == 1
    reserved, steps, unknown_cost, released = releases[0]
    assert (steps, unknown_cost) == (3, True)
    assert (reserved.steps_reserved, reserved.unknown_cost_reserved) == (3, True)
    assert (released.steps_reserved, released.unknown_cost_reserved) == (0, False)


async def test_moa_cost_model_failure_releases_full_reservation_and_reraises(
    monkeypatch,
):
    import kairyu.orchestration.moa as moa_module

    async def fake_run_moa(*args, **kwargs):
        return MoAResult(
            final_text="synthesized",
            proposals=("proposal one", "proposal two"),
        )

    def failing_cost_model(request, result):
        raise ValueError("cost unavailable")

    releases = _track_budget_releases(monkeypatch)
    monkeypatch.setattr(moa_module, "run_moa", fake_run_moa)
    orchestrator = _orchestrator(
        moa_samples=2,
        budget=Budget(max_steps=3, max_cost_usd=1.0),
        cost_model=failing_cost_model,
    )

    with pytest.raises(ValueError, match="cost unavailable"):
        await orchestrator.run(COMPLEX)

    assert len(releases) == 1
    reserved, steps, unknown_cost, released = releases[0]
    assert (steps, unknown_cost) == (3, True)
    assert (reserved.steps_reserved, reserved.unknown_cost_reserved) == (3, True)
    assert (released.steps_reserved, released.unknown_cost_reserved) == (0, False)


async def test_moa_cancellation_releases_full_reservation_and_propagates(monkeypatch):
    import kairyu.orchestration.moa as moa_module

    started = asyncio.Event()
    blocked = asyncio.Event()

    async def blocked_run_moa(*args, **kwargs):
        started.set()
        await blocked.wait()
        raise AssertionError("unreachable")

    releases = _track_budget_releases(monkeypatch)
    monkeypatch.setattr(moa_module, "run_moa", blocked_run_moa)
    orchestrator = _orchestrator(
        moa_samples=2,
        budget=Budget(max_steps=3, max_cost_usd=1.0),
    )

    task = asyncio.create_task(orchestrator.run(COMPLEX))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(releases) == 1
    reserved, steps, unknown_cost, released = releases[0]
    assert (steps, unknown_cost) == (3, True)
    assert (reserved.steps_reserved, reserved.unknown_cost_reserved) == (3, True)
    assert (released.steps_reserved, released.unknown_cost_reserved) == (0, False)


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
