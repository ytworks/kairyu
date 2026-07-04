import asyncio
import time

import pytest

from kairyu.engine.backend import GenerationRequest, GenerationResult
from kairyu.engine.mock import MockBackend
from kairyu.orchestration.budget import Budget
from kairyu.orchestration.conductor import Conductor, RoleSpec
from kairyu.outputs import CompletionOutput


class ScriptedBackend:
    """Returns queued responses in order; used to script verifier verdicts."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts_seen: list[str] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.prompts_seen.append(request.prompt)
        text = self._responses.pop(0)
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(CompletionOutput(index=0, text=text, token_ids=()),),
        )

    async def stream(self, request):  # pragma: no cover - unused
        yield await self.generate(request)

    async def shutdown(self) -> None:
        return None


def _linear_roles() -> tuple[RoleSpec, ...]:
    return (
        RoleSpec(name="planner", worker="w", prompt="[planner] plan for: {query}"),
        RoleSpec(
            name="worker",
            worker="w",
            prompt="[worker] execute: {planner}",
            depends_on=("planner",),
        ),
    )


async def test_linear_dag_passes_upstream_output_downstream():
    backend = MockBackend()
    conductor = Conductor(roles=_linear_roles(), workers={"w": backend})
    result = await conductor.run("build a cli")
    planner_output = result.outputs["planner"]
    assert "build a cli" in backend.prompts_seen[0]
    assert planner_output[-20:] in backend.prompts_seen[1]
    assert result.final_text == result.outputs["worker"]


async def test_unit_backend_failure_does_not_destroy_the_run():
    # O4: a transient backend error on one unit must not raise out of run() and
    # discard every completed output — the Conductor returns best-so-far.
    class _FlakyWorker:
        async def generate(self, request):
            if "planner" in request.prompt:
                return GenerationResult(
                    request_id=request.request_id, prompt=request.prompt,
                    completions=(CompletionOutput(index=0, text="a plan", token_ids=()),),
                )
            raise RuntimeError("worker backend down")

        async def stream(self, request):  # pragma: no cover
            yield await self.generate(request)

        async def shutdown(self):
            return None

    conductor = Conductor(roles=_linear_roles(), workers={"w": _FlakyWorker()})
    result = await conductor.run("build a cli")  # must not raise
    assert result.outputs["planner"] == "a plan"  # completed work survives
    assert "worker" not in result.outputs  # the failed unit produced nothing
    assert any(event.kind == "failed" for event in result.trace)


async def test_diamond_dag_runs_middle_wave_concurrently():
    latency = 0.05
    backend = MockBackend(latency_s=latency)
    roles = (
        RoleSpec(name="root", worker="w", prompt="root: {query}"),
        RoleSpec(name="a", worker="w", prompt="a: {root}", depends_on=("root",)),
        RoleSpec(name="b", worker="w", prompt="b: {root}", depends_on=("root",)),
        RoleSpec(
            name="synth",
            worker="w",
            prompt="synth: {a} | {b}",
            role_type="synthesizer",
            depends_on=("a", "b"),
        ),
    )
    conductor = Conductor(roles=roles, workers={"w": backend})
    start = time.perf_counter()
    result = await conductor.run("q")
    elapsed = time.perf_counter() - start
    assert elapsed < latency * 4  # a and b overlapped: 3 waves, not 4 sequential calls
    assert result.outputs["root"][-10:] in result.outputs["a"]
    assert result.final_text == result.outputs["synth"]


def test_verifier_with_unavailable_dependency_rejected_at_init():
    # M1: a verifier runs inline after its target, so a dependency the target
    # doesn't have (here "planner", scheduled in a parallel wave) would render
    # as "" at verify time — reject it loudly instead of a silent wrong verdict.
    roles = (
        RoleSpec(name="planner", worker="w", prompt="plan: {query}"),
        RoleSpec(name="worker", worker="w", prompt="do: {query}"),
        RoleSpec(
            name="checker", worker="w", role_type="verifier", verifies="worker",
            prompt="check {worker} against {planner}",
            depends_on=("worker", "planner"),  # planner is NOT a dep of worker
        ),
    )
    with pytest.raises(ValueError, match="not.*available when it runs inline"):
        Conductor(roles=roles, workers={"w": MockBackend()})


def test_cycle_is_rejected_at_init():
    roles = (
        RoleSpec(name="a", worker="w", prompt="{b}", depends_on=("b",)),
        RoleSpec(name="b", worker="w", prompt="{a}", depends_on=("a",)),
    )
    with pytest.raises(ValueError, match="cycle"):
        Conductor(roles=roles, workers={"w": MockBackend()})


def test_unknown_dependency_and_worker_rejected():
    with pytest.raises(ValueError, match="unknown dependency"):
        Conductor(
            roles=(RoleSpec(name="a", worker="w", prompt="x", depends_on=("ghost",)),),
            workers={"w": MockBackend()},
        )
    with pytest.raises(ValueError, match="unknown worker"):
        Conductor(roles=(RoleSpec(name="a", worker="ghost", prompt="x"),), workers={})


def test_verifier_must_depend_on_its_target():
    roles = (
        RoleSpec(name="worker", worker="w", prompt="do: {query}"),
        RoleSpec(
            name="check", worker="w", prompt="verify", role_type="verifier", verifies="worker"
        ),
    )
    with pytest.raises(ValueError, match="depend on"):
        Conductor(roles=roles, workers={"w": MockBackend()})


async def test_verifier_fail_triggers_refinement_until_pass():
    backend = ScriptedBackend(
        ["draft v1", "FAIL: missing edge case", "draft v2", "PASS: good"]
    )
    roles = (
        RoleSpec(name="worker", worker="w", prompt="do: {query}"),
        RoleSpec(
            name="check",
            worker="w",
            prompt="verify: {worker}",
            role_type="verifier",
            verifies="worker",
            depends_on=("worker",),
        ),
    )
    conductor = Conductor(roles=roles, workers={"w": backend})
    result = await conductor.run("task", budget=Budget(max_refine_depth=2))
    assert result.outputs["worker"] == "draft v2"
    assert result.outputs["check"] == "PASS: good"
    assert "missing edge case" in backend.prompts_seen[2]  # feedback fed to retry
    assert result.budget_state.steps_used == 4


async def test_refinement_depth_is_bounded():
    backend = ScriptedBackend(["d1", "FAIL 1", "d2", "FAIL 2", "d3", "FAIL 3"])
    roles = (
        RoleSpec(name="worker", worker="w", prompt="do: {query}"),
        RoleSpec(
            name="check",
            worker="w",
            prompt="verify: {worker}",
            role_type="verifier",
            verifies="worker",
            depends_on=("worker",),
        ),
    )
    conductor = Conductor(roles=roles, workers={"w": backend})
    result = await conductor.run("task", budget=Budget(max_refine_depth=2))
    assert result.outputs["worker"] == "d3"  # 1 initial + 2 refinements, then stop
    assert result.outputs["check"] == "FAIL 3"


async def test_budget_exhaustion_returns_best_so_far():
    backend = MockBackend()
    conductor = Conductor(roles=_linear_roles(), workers={"w": backend})
    result = await conductor.run("q", budget=Budget(max_steps=1))
    assert result.budget_state.is_exhausted is True
    assert result.outputs["planner"]
    assert "worker" not in result.outputs
    assert result.final_text == result.outputs["planner"]
    assert any(event.kind == "skipped:budget" for event in result.trace)


async def test_all_prompts_share_prefix_for_kv_affinity():
    backend = MockBackend()
    prefix = "SYSTEM: you are kairyu.\n\n"
    conductor = Conductor(roles=_linear_roles(), workers={"w": backend}, shared_prefix=prefix)
    await conductor.run("q")
    assert all(p.startswith(prefix) for p in backend.prompts_seen)


async def test_concurrent_runs_do_not_share_state():
    backend = MockBackend()
    conductor = Conductor(roles=_linear_roles(), workers={"w": backend})
    results = await asyncio.gather(conductor.run("one"), conductor.run("two"))
    assert results[0].final_text != results[1].final_text


async def test_cost_model_charges_budget_and_trips_cost_cap():
    backend = MockBackend()
    conductor = Conductor(
        roles=_linear_roles(),
        workers={"w": backend},
        cost_model=lambda request, result: 1.0,
    )
    result = await conductor.run("q", budget=Budget(max_cost_usd=0.5))
    assert result.budget_state.cost_used == 1.0
    assert result.budget_state.is_exhausted is True
    assert "worker" not in result.outputs
    assert result.final_text == result.outputs["planner"]
