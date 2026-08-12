import asyncio
import threading
import time
from dataclasses import replace

import pytest

import kairyu.orchestration.conductor as conductor_module
from kairyu.engine.backend import (
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
)
from kairyu.engine.mock import MockBackend
from kairyu.engine.prompt import (
    MultimodalItem,
    MultimodalMessage,
    MultimodalMessagePart,
    MultimodalPrompt,
    PromptInput,
    TemplatedPrompt,
    TextPrompt,
)
from kairyu.orchestration.budget import Budget, BudgetState
from kairyu.orchestration.conductor import Conductor, RoleSpec
from kairyu.orchestration.prefix_index import prefix_root_fingerprint
from kairyu.orchestration.trace import WorkerTraceIdentity
from kairyu.outputs import CompletionOutput


class ScriptedBackend:
    """Returns queued responses in order; used to script verifier verdicts."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts_seen: list[PromptInput] = []

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


class GateBackend:
    """Records calls and holds them in flight until the test opens the gate."""

    def __init__(self) -> None:
        self.calls: list[GenerationRequest] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        self.started.set()
        await self.release.wait()
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(index=0, text=f"result: {request.prompt}", token_ids=()),
            ),
        )

    async def stream(self, request):  # pragma: no cover - unused
        yield await self.generate(request)

    async def shutdown(self) -> None:
        return None


class PullThroughBackend:
    """Holds a final stream open after its first cumulative chunk."""

    def __init__(self, *, fail_after_first: bool = False) -> None:
        self.stream_started = asyncio.Event()
        self.release = asyncio.Event()
        self.stream_closed = False
        self.fail_after_first = fail_after_first

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(index=0, text="draft", token_ids=(1,)),
            ),
            usage=GenerationUsage(
                prompt_tokens=5,
                completion_tokens=1,
                cached_tokens=2,
            ),
        )

    async def stream(self, request):
        self.stream_started.set()
        try:
            yield GenerationResult(
                request_id=request.request_id,
                prompt=request.prompt,
                completions=(
                    CompletionOutput(index=0, text="first ", token_ids=(1,)),
                ),
                finished=False,
            )
            await self.release.wait()
            if self.fail_after_first:
                raise RuntimeError("final stream failed")
            yield GenerationResult(
                request_id=request.request_id,
                prompt=request.prompt,
                completions=(
                    CompletionOutput(
                        index=0,
                        text="first second",
                        token_ids=(1, 2),
                        finish_reason="stop",
                    ),
                ),
                usage=GenerationUsage(
                    prompt_tokens=7,
                    completion_tokens=2,
                    cached_tokens=4,
                ),
            )
        finally:
            self.stream_closed = True

    async def shutdown(self) -> None:
        return None


async def test_role_prompt_render_runs_off_event_loop(monkeypatch):
    backend = MockBackend()
    conductor = Conductor(
        roles=(RoleSpec(name="worker", worker="w", prompt="answer: {query}"),),
        workers={"w": backend},
    )
    event_loop_thread = threading.get_ident()
    render_threads = []
    original = conductor._render

    def tracked_render(*args):
        render_threads.append(threading.get_ident())
        return original(*args)

    monkeypatch.setattr(conductor, "_render", tracked_render)

    await conductor.run("q")

    assert render_threads
    assert all(thread_id != event_loop_thread for thread_id in render_threads)

@pytest.mark.parametrize("streaming", [False, True])
async def test_conductor_rejects_templated_query_before_derivation(streaming):
    backend = MockBackend()
    conductor = Conductor(roles=_linear_roles(), workers={"w": backend})
    query = TemplatedPrompt("<BOS>user hello<ASSISTANT>")

    with pytest.raises(ValueError, match="cannot derive role prompts.*pre-rendered"):
        if streaming:
            _ = [event async for event in conductor.stream(query)]
        else:
            await conductor.run(query)

    assert backend.prompts_seen == ()


def test_conductor_rejects_templated_role_or_shared_prefix():
    templated = TemplatedPrompt("<BOS>already rendered")

    with pytest.raises(ValueError, match="role templates.*pre-rendered"):
        RoleSpec(name="role", worker="w", prompt=templated)
    with pytest.raises(ValueError, match="shared_prefix.*pre-rendered"):
        Conductor(
            roles=_linear_roles(),
            workers={"w": MockBackend()},
            shared_prefix=templated,
        )


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


def _image_prompt() -> MultimodalPrompt:
    return MultimodalPrompt(
        base=TextPrompt("user: inspect <image:0>"),
        items=(
            MultimodalItem(
                modality="image",
                encoding="uri",
                data="data:image/png;base64,opaque",
            ),
        ),
        messages=(
            MultimodalMessage(
                "user",
                (
                    MultimodalMessagePart("text", text="inspect "),
                    MultimodalMessagePart("item", item_index=0),
                ),
            ),
        ),
    )


async def test_multimodal_input_runs_vision_role_then_text_verifier_and_publisher():
    backend = ScriptedBackend(
        ["red square", "draft from red square", "PASS grounded", "final red"]
    )
    roles = (
        RoleSpec(
            name="vision",
            worker="w",
            role_type="vision",
            prompt="ground the image for {query}",
            extra_args={
                "chat_template_kwargs": {"enable_thinking": False}
            },
        ),
        RoleSpec(
            name="draft",
            worker="w",
            prompt="draft using {vision}",
            depends_on=("vision",),
        ),
        RoleSpec(
            name="verify",
            worker="w",
            role_type="verifier",
            verifies="draft",
            prompt="verify {draft} against {vision}",
            depends_on=("draft", "vision"),
        ),
        RoleSpec(
            name="publisher",
            worker="w",
            role_type="publisher",
            prompt="publish {draft}; {verify}; evidence {vision}",
            depends_on=("vision", "draft", "verify"),
        ),
    )
    conductor = Conductor(roles=roles, workers={"w": backend})
    initial = conductor.initial_requests(
        "chart question",
        request_id_suffix="vision",
        multimodal_prompt=_image_prompt(),
    )

    assert len(initial) == 1
    assert initial[0][1].sampling_params.extra_args == {
        "chat_template_kwargs": {"enable_thinking": False}
    }

    result = await conductor.run(
        "chart question",
        multimodal_prompt=_image_prompt(),
    )

    assert result.final_text == "final red"
    assert [event.node for event in result.trace] == [
        "vision",
        "draft",
        "verify",
        "publisher",
    ]
    vision_prompt = backend.prompts_seen[0]
    assert isinstance(vision_prompt, MultimodalPrompt)
    assert vision_prompt.items == _image_prompt().items
    assert vision_prompt.messages[-1].content[-1].text is not None
    assert "KAIRYU VISION ROLE" in vision_prompt.messages[-1].content[-1].text
    assert backend.prompts_seen[1] == "draft using red square"
    assert backend.prompts_seen[2] == (
        "verify draft from red square against red square"
    )
    assert backend.prompts_seen[3] == (
        "publish draft from red square; PASS grounded; evidence red square"
    )


async def test_text_input_skips_conditional_vision_role():
    backend = ScriptedBackend(["text draft"])
    roles = (
        RoleSpec(
            name="vision",
            worker="w",
            role_type="vision",
            prompt="ground image",
        ),
        RoleSpec(
            name="draft",
            worker="w",
            prompt="answer {query} {vision}",
            depends_on=("vision",),
        ),
    )

    result = await Conductor(roles=roles, workers={"w": backend}).run("plain")

    assert result.final_text == "text draft"
    assert backend.prompts_seen == ["answer plain "]


async def test_multimodal_input_fails_closed_when_vision_role_fails():
    roles = (
        RoleSpec(
            name="vision",
            worker="w",
            role_type="vision",
            prompt="ground image",
        ),
        RoleSpec(
            name="draft",
            worker="w",
            prompt="answer from {vision}",
            depends_on=("vision",),
        ),
    )

    with pytest.raises(conductor_module._ObservedGenerationError):
        await Conductor(
            roles=roles,
            workers={"w": ScriptedBackend([])},
        ).run("image task", multimodal_prompt=_image_prompt())


def test_multimodal_input_requires_a_dependency_free_vision_role():
    with pytest.raises(ValueError, match="vision role.*dependency-free"):
        Conductor(
            roles=(
                RoleSpec(name="root", worker="w", prompt="root"),
                RoleSpec(
                    name="vision",
                    worker="w",
                    role_type="vision",
                    prompt="vision",
                    depends_on=("root",),
                ),
            ),
            workers={"w": MockBackend()},
        )

    conductor = Conductor(roles=_linear_roles(), workers={"w": MockBackend()})
    with pytest.raises(ValueError, match="requires a vision role"):
        conductor.initial_requests(
            "image task",
            request_id_suffix="test",
            multimodal_prompt=_image_prompt(),
        )


def test_cache_hint_carries_only_an_exact_complete_shared_root() -> None:
    worker = MockBackend()
    empty = Conductor(roles=_linear_roles(), workers={"w": worker})
    short = Conductor(
        roles=_linear_roles(),
        workers={"w": worker},
        shared_prefix="s" * 255,
    )
    shared_prefix = "s" * 256 + "unused suffix"
    complete = Conductor(
        roles=_linear_roles(),
        workers={"w": worker},
        shared_prefix=shared_prefix,
    )

    assert empty._cache_hint("session").prefix_fingerprint == ""
    assert short._cache_hint("session").prefix_fingerprint == ""
    assert (
        complete._cache_hint("session").prefix_fingerprint
        == prefix_root_fingerprint(shared_prefix)
    )


async def test_linear_dag_passes_upstream_output_downstream():
    backend = MockBackend()
    conductor = Conductor(roles=_linear_roles(), workers={"w": backend})
    result = await conductor.run("build a cli")
    planner_output = result.outputs["planner"]
    assert "build a cli" in backend.prompts_seen[0]
    assert planner_output[-20:] in backend.prompts_seen[1]
    assert result.final_text == result.outputs["worker"]


async def test_conductor_sums_cached_tokens_across_units():
    class CachedBackend(MockBackend):
        async def generate(self, request):
            result = await super().generate(request)
            return replace(
                result,
                usage=GenerationUsage(
                    prompt_tokens=10,
                    completion_tokens=2,
                    cached_tokens=4,
                ),
            )

    conductor = Conductor(
        roles=_linear_roles(),
        workers={"w": CachedBackend()},
    )

    result = await conductor.run("build a cli")

    assert result.usage == (20, 4)
    assert result.cached_tokens == 8


async def test_final_synthesizer_deltas_pull_through_before_completion():
    backend = PullThroughBackend()
    roles = (
        RoleSpec(name="draft", worker="w", prompt="draft: {query}"),
        RoleSpec(
            name="final",
            worker="w",
            prompt="final: {draft}",
            role_type="synthesizer",
            depends_on=("draft",),
        ),
    )
    conductor = Conductor(roles=roles, workers={"w": backend})
    stream = conductor.stream("q")

    first = await anext(stream)

    assert first.kind == "delta"
    assert first.text == "first "
    assert not backend.release.is_set()

    backend.release.set()
    remaining = [event async for event in stream]
    assert [event.kind for event in remaining] == ["delta", "result"]
    assert remaining[0].text == "second"
    result = remaining[-1].result
    assert result is not None
    assert result.final_text == "first second"
    assert result.usage == (12, 3)
    assert result.cached_tokens == 6
    final_trace = result.trace[-1]
    assert final_trace.node == "final"
    assert final_trace.metadata["streamed"] is True


async def test_final_stream_cancellation_closes_backend_and_releases_budget():
    backend = PullThroughBackend()
    spec = RoleSpec(name="final", worker="w", prompt="{query}")
    conductor = Conductor(roles=(spec,), workers={"w": backend})
    run = conductor_module._RunState(
        budget=BudgetState(Budget(max_steps=1, max_cost_usd=1.0))
    )
    stream = conductor._stream_unit(run, "session", "q", spec)

    assert (await anext(stream)).text == "first "
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert backend.stream_closed is True
    assert run.budget.steps_used == 0
    assert run.budget.steps_reserved == 0
    assert run.budget.unknown_cost_reserved is False


async def test_final_stream_failure_is_not_converted_to_false_success():
    backend = PullThroughBackend(fail_after_first=True)
    roles = (RoleSpec(name="final", worker="w", prompt="{query}"),)
    conductor = Conductor(roles=roles, workers={"w": backend})
    stream = conductor.stream("q")

    assert (await anext(stream)).text == "first "
    backend.release.set()
    with pytest.raises(RuntimeError, match="final stream failed"):
        await anext(stream)


async def test_stream_rejects_provisional_final_with_post_verifier():
    backend = MockBackend()
    roles = (
        RoleSpec(name="final", worker="w", prompt="{query}"),
        RoleSpec(
            name="check",
            worker="w",
            prompt="{final}",
            role_type="verifier",
            verifies="final",
            depends_on=("final",),
        ),
    )
    conductor = Conductor(roles=roles, workers={"w": backend})

    with pytest.raises(ValueError, match="cannot have a post-generation verifier"):
        await anext(conductor.stream("q"))


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
    verifier_events = [event for event in result.trace if event.role == "verifier"]
    assert verifier_events[-1].metadata["refinement_exhausted"] is True


async def test_visible_intermediates_are_attributed_and_separate_from_final():
    backend = ScriptedBackend(
        ["draft v1", "FAIL: revise", "draft v2", "PASS", "published answer"]
    )
    roles = (
        RoleSpec(name="draft", worker="tier1", prompt="draft: {query}"),
        RoleSpec(
            name="check",
            worker="tier2",
            prompt="verify: {draft}",
            role_type="verifier",
            verifies="draft",
            depends_on=("draft",),
        ),
        RoleSpec(
            name="publisher",
            worker="tier2",
            prompt="publish: {draft} {check}",
            role_type="publisher",
            depends_on=("draft", "check"),
        ),
    )
    conductor = Conductor(
        roles=roles,
        workers={"tier1": backend, "tier2": backend},
        worker_trace={
            "tier1": WorkerTraceIdentity(engine="qwen-pool", model="Qwen3.6-27B"),
            "tier2": WorkerTraceIdentity(engine="deepseek-pool", model="DeepSeek-V4"),
        },
        expose_intermediate_outputs=True,
    )

    events = [
        event
        async for event in conductor.stream(
            "task",
            budget=Budget(max_steps=5, max_refine_depth=1),
        )
    ]

    reasoning = "".join(event.text for event in events if event.kind == "reasoning")
    final = events[-1].result
    assert final is not None
    assert final.final_text == "published answer"
    assert final.reasoning_content is not None
    assert reasoning.removesuffix("\n\n---\n\n") == final.reasoning_content
    assert reasoning.index("draft v1") < reasoning.index("FAIL: revise")
    assert reasoning.index("draft v2") < reasoning.index("PASS")
    assert "L2 role: `worker`" in reasoning
    assert "L1 worker: `tier1`" in reasoning
    assert "Engine: `qwen-pool`" in reasoning
    assert "Model: `Qwen3.6-27B`" in reasoning
    assert "Model: `DeepSeek-V4`" in reasoning
    assert "published answer" not in reasoning


async def test_unary_reasoning_starts_with_actual_final_model_attribution():
    backend = ScriptedBackend(["draft", "published answer"])
    roles = (
        RoleSpec(name="draft", worker="tier1", prompt="draft: {query}"),
        RoleSpec(
            name="publisher",
            worker="tier2",
            prompt="publish: {draft}",
            role_type="publisher",
            depends_on=("draft",),
        ),
    )
    conductor = Conductor(
        roles=roles,
        workers={"tier1": backend, "tier2": backend},
        worker_trace={
            "tier1": WorkerTraceIdentity(engine="qwen-pool", model="Qwen3.6-27B"),
            "tier2": WorkerTraceIdentity(engine="deepseek-pool", model="DeepSeek-V4"),
        },
        expose_intermediate_outputs=True,
    )

    result = await conductor.run("task")

    assert result.reasoning_content is not None
    assert result.reasoning_content.startswith("### Final answer attribution")
    assert "L2 role: `publisher`" in result.reasoning_content
    assert "L1 worker: `tier2`" in result.reasoning_content
    assert "Engine: `deepseek-pool`" in result.reasoning_content
    assert "Model: `DeepSeek-V4`" in result.reasoning_content
    assert result.reasoning_content.index("Final answer attribution") < (
        result.reasoning_content.index("### draft")
    )


async def test_intermediates_remain_hidden_by_default():
    backend = ScriptedBackend(["draft", "published answer"])
    roles = (
        RoleSpec(name="draft", worker="w", prompt="draft: {query}"),
        RoleSpec(
            name="publisher",
            worker="w",
            prompt="publish: {draft}",
            depends_on=("draft",),
        ),
    )
    conductor = Conductor(roles=roles, workers={"w": backend})

    result = await conductor.run("task")

    assert result.reasoning_content is None


async def test_budget_exhaustion_returns_best_so_far():
    backend = MockBackend()
    conductor = Conductor(roles=_linear_roles(), workers={"w": backend})
    result = await conductor.run("q", budget=Budget(max_steps=1))
    assert result.budget_state.is_exhausted is True
    assert result.outputs["planner"]
    assert "worker" not in result.outputs
    assert result.final_text == result.outputs["planner"]
    assert any(event.kind == "skipped:budget" for event in result.trace)


async def test_stream_budget_exhaustion_emits_best_so_far_fallback():
    backend = MockBackend()
    conductor = Conductor(roles=_linear_roles(), workers={"w": backend})

    events = [
        event
        async for event in conductor.stream("q", budget=Budget(max_steps=1))
    ]

    result = events[-1].result
    assert result is not None
    assert result.final_text == result.outputs["planner"]
    assert "".join(event.text for event in events if event.kind == "delta") == (
        result.final_text
    )


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


async def test_parallel_ready_roots_reserve_step_before_backend_dispatch():
    backend = GateBackend()
    roles = (
        RoleSpec(name="a", worker="w", prompt="a: {query}"),
        RoleSpec(name="b", worker="w", prompt="b: {query}"),
    )
    conductor = Conductor(roles=roles, workers={"w": backend})

    task = asyncio.create_task(conductor.run("q", budget=Budget(max_steps=1)))
    await backend.started.wait()
    await asyncio.sleep(0)
    backend.release.set()
    result = await task

    assert len(backend.calls) == 1
    assert result.budget_state.steps_used == 1
    assert result.budget_state.steps_reserved == 0
    assert sum(event.kind == "skipped:budget" for event in result.trace) == 1
    assert not any(event.kind == "failed" for event in result.trace)


async def test_failed_generation_releases_reservation_for_later_unit():
    class _FailThenSucceedBackend:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, request: GenerationRequest) -> GenerationResult:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary backend failure")
            return GenerationResult(
                request_id=request.request_id,
                prompt=request.prompt,
                completions=(
                    CompletionOutput(index=0, text="recovered", token_ids=()),
                ),
            )

        async def stream(self, request):  # pragma: no cover - unused
            yield await self.generate(request)

        async def shutdown(self) -> None:
            return None

    backend = _FailThenSucceedBackend()
    roles = (
        RoleSpec(name="first", worker="w", prompt="first: {query}"),
        RoleSpec(
            name="second",
            worker="w",
            prompt="second: {first}",
            depends_on=("first",),
        ),
    )
    conductor = Conductor(
        roles=roles,
        workers={"w": backend},
        cost_model=lambda request, result: 0.25,
    )

    result = await conductor.run(
        "q", budget=Budget(max_steps=1, max_cost_usd=1.0)
    )

    assert backend.calls == 2
    assert result.outputs["second"] == "recovered"
    assert result.budget_state.steps_used == 1
    assert result.budget_state.steps_reserved == 0
    assert result.budget_state.cost_used == 0.25
    assert result.budget_state.unknown_cost_reserved is False
    assert any(
        event.node == "first" and event.kind == "failed" for event in result.trace
    )
    failed = next(event for event in result.trace if event.node == "first")
    assert failed.operation == "generation"
    assert failed.status == "failed"
    assert failed.error is not None
    assert failed.error.type == "RuntimeError"
    assert failed.timing is not None
    assert failed.timing.completed_at is not None


async def test_cancelled_generation_releases_reservation_and_propagates():
    backend = GateBackend()
    conductor = Conductor(
        roles=(RoleSpec(name="worker", worker="w", prompt="{query}"),),
        workers={"w": backend},
    )
    run = conductor_module._RunState(
        budget=BudgetState(Budget(max_steps=1, max_cost_usd=1.0))
    )

    task = asyncio.create_task(
        conductor._generate(run, "session", "worker", "w", "prompt", 0)
    )
    await backend.started.wait()
    reserved = (run.budget.steps_reserved, run.budget.unknown_cost_reserved)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert reserved == (1, True)
    assert run.budget.steps_used == 0
    assert run.budget.steps_reserved == 0
    assert run.budget.unknown_cost_reserved is False


async def test_cost_cap_allows_only_one_unknown_cost_dispatch_and_shows_overrun():
    backend = GateBackend()
    roles = (
        RoleSpec(name="a", worker="w", prompt="a: {query}"),
        RoleSpec(name="b", worker="w", prompt="b: {query}"),
    )
    conductor = Conductor(
        roles=roles,
        workers={"w": backend},
        cost_model=lambda request, result: 1.0,
    )

    task = asyncio.create_task(
        conductor.run("q", budget=Budget(max_steps=2, max_cost_usd=0.5))
    )
    await backend.started.wait()
    await asyncio.sleep(0)
    backend.release.set()
    result = await task

    assert len(backend.calls) == 1
    assert result.budget_state.steps_used == 1
    assert result.budget_state.cost_used == 1.0
    assert result.budget_state.is_exhausted is True
    assert result.budget_state.unknown_cost_reserved is False
    assert sum(event.kind == "skipped:budget" for event in result.trace) == 1


async def test_verifier_budget_refusal_preserves_completed_target():
    backend = ScriptedBackend(["best completed draft"])
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
    run = conductor_module._RunState(budget=BudgetState(Budget(max_steps=1)))

    await conductor._run_unit(run, "session", "task", roles[0])

    assert backend.prompts_seen == ["do: task"]
    assert run.outputs == {"worker": "best completed draft"}
    assert run.completion_order == ["worker"]
    assert [(event.node, event.kind, event.detail) for event in run.trace] == [
        ("worker", "generated", "attempt=0"),
        ("check", "skipped:budget", ""),
    ]
    assert conductor._final_text(run) == "best completed draft"
    generated, skipped = run.trace
    assert generated.operation == "generation"
    assert generated.status == "success"
    assert generated.timing is not None
    assert generated.budget is not None
    assert generated.budget.steps_consumed == 1
    assert skipped.operation == "verification"
    assert skipped.status == "skipped"
    assert skipped.metadata == {"reason": "budget"}
