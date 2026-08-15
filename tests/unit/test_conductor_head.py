"""Fast-head streaming, continuation dedup, and per-role sampling (EO-D7/D9)."""

import asyncio

import pytest

from kairyu.engine.backend import (
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
)
from kairyu.orchestration.budget import Budget
from kairyu.orchestration.conductor import (
    Conductor,
    ConductorStreamError,
    RoleSamplingOverrides,
    RoleSpec,
)
from kairyu.outputs import CompletionOutput
from kairyu.sampling_params import SamplingParams


class StreamScriptedBackend:
    """Streams cumulative chunks; generate() returns the full text."""

    def __init__(self, chunks: list[str], *, fail_after: int | None = None) -> None:
        self._chunks = list(chunks)
        self._fail_after = fail_after
        self.requests_seen: list[GenerationRequest] = []

    def _result(self, request, text, *, finished):
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0,
                    text=text,
                    token_ids=(),
                    finish_reason="stop" if finished else None,
                ),
            ),
            usage=GenerationUsage(prompt_tokens=3, completion_tokens=len(text)),
            finished=finished,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests_seen.append(request)
        return self._result(request, self._chunks[-1], finished=True)

    async def stream(self, request: GenerationRequest):
        self.requests_seen.append(request)
        for index, cumulative in enumerate(self._chunks):
            if self._fail_after is not None and index >= self._fail_after:
                raise RuntimeError("scripted stream failure")
            yield self._result(
                request,
                cumulative,
                finished=index == len(self._chunks) - 1,
            )

    async def shutdown(self) -> None:
        return None


class GatedScriptedBackend:
    """Holds generate() until released so head concurrency is observable."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.started.set()
        await self.release.wait()
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(CompletionOutput(index=0, text=self._text, token_ids=()),),
        )

    async def stream(self, request):  # pragma: no cover - unused
        yield await self.generate(request)

    async def shutdown(self) -> None:
        return None


def _head_roles() -> tuple[RoleSpec, ...]:
    return (
        RoleSpec(name="head", worker="hw", role_type="head", prompt="[head] {query}"),
        RoleSpec(
            name="draft",
            worker="dw",
            prompt="[draft] {query} :: {head}",
            depends_on=("head",),
        ),
        RoleSpec(
            name="continuation",
            worker="cw",
            role_type="publisher",
            prompt="[cont] {head} :: {draft}",
            depends_on=("head", "draft"),
        ),
    )


def _conductor(head_backend, draft_backend, continuation_backend, **kwargs):
    return Conductor(
        _head_roles(),
        {"hw": head_backend, "dw": draft_backend, "cw": continuation_backend},
        **kwargs,
    )


async def _collect(stream):
    events = []
    async for event in stream:
        events.append(event)
    return events


async def test_head_streams_public_deltas_before_dag_completes():
    head = StreamScriptedBackend(["Intro ", "Intro done. "])
    draft = GatedScriptedBackend("draft body")
    continuation = StreamScriptedBackend(["Final answer."])
    conductor = _conductor(head, draft, continuation)

    events = []
    saw_delta_while_draft_pending = False
    async for event in conductor.stream("task"):
        if event.kind == "delta" and not draft.release.is_set():
            saw_delta_while_draft_pending = True
            draft.release.set()
        events.append(event)
    assert saw_delta_while_draft_pending, "head must stream before the DAG finishes"
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Intro done. Final answer."
    assert "".join(e.text for e in events if e.kind == "delta") == result.final_text
    assert result.completions[0].text == result.final_text


async def test_reasoning_events_emit_incrementally_while_units_pending():
    head = StreamScriptedBackend(["Intro. "])
    fast_unit = StreamScriptedBackend(["fast unit output"])
    slow_unit = GatedScriptedBackend("slow unit output")
    continuation = StreamScriptedBackend(["Final."])
    roles = (
        RoleSpec(name="head", worker="hw", role_type="head", prompt="[head] {query}"),
        RoleSpec(name="fast", worker="fw", prompt="[fast] {query} {head}",
                 depends_on=("head",)),
        RoleSpec(name="slow", worker="sw", prompt="[slow] {query} {head}",
                 depends_on=("head",)),
        RoleSpec(
            name="continuation",
            worker="cw",
            role_type="publisher",
            prompt="[cont] {head} {fast} {slow}",
            depends_on=("head", "fast", "slow"),
        ),
    )
    conductor = Conductor(
        roles,
        {"hw": head, "fw": fast_unit, "sw": slow_unit, "cw": continuation},
        expose_intermediate_outputs=True,
    )

    saw_reasoning_while_slow_pending = False
    events = []
    async for event in conductor.stream("task"):
        if event.kind == "reasoning" and not slow_unit.release.is_set():
            saw_reasoning_while_slow_pending = True
        if event.kind in {"delta", "reasoning"} and not slow_unit.release.is_set():
            # Only release once the fast unit's section proved incremental.
            if saw_reasoning_while_slow_pending:
                slow_unit.release.set()
        events.append(event)
    assert saw_reasoning_while_slow_pending, (
        "a completed unit's reasoning section must stream while siblings run"
    )
    result = events[-1].result
    assert result is not None
    assert "fast unit output" in (result.reasoning_content or "")
    # The committed head is public content and must not be duplicated into
    # the reasoning sections.
    assert "Intro." not in (result.reasoning_content or "")


async def test_continuation_dedupes_exact_head_repetition():
    head = StreamScriptedBackend(["Intro. "])
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["Intro. ", "Intro. The rest."])
    conductor = _conductor(head, draft, continuation)

    events = await _collect(conductor.stream("task"))
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Intro. The rest."
    assert "".join(e.text for e in events if e.kind == "delta") == result.final_text


async def test_continuation_divergent_text_flushes_verbatim():
    head = StreamScriptedBackend(["Intro. "])
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["Int", "Interesting tail."])
    conductor = _conductor(head, draft, continuation)

    events = await _collect(conductor.stream("task"))
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Intro. Interesting tail."


async def test_head_zero_byte_failure_degrades_to_continuation_only():
    head = StreamScriptedBackend(["never"], fail_after=0)
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["Standalone answer."])
    conductor = _conductor(head, draft, continuation)

    events = await _collect(conductor.stream("task"))
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Standalone answer."
    failed = [e for e in result.trace if e.node == "head" and e.kind == "failed"]
    assert failed and failed[0].metadata.get("partial") is False


async def test_head_failure_after_bytes_commits_partial_prefix():
    head = StreamScriptedBackend(["Intro ", "never"], fail_after=1)
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["continues."])
    conductor = _conductor(head, draft, continuation)

    events = await _collect(conductor.stream("task"))
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Intro continues."
    assert result.outputs["head"] == "Intro "
    # Downstream prompts saw the committed partial prefix.
    assert ":: Intro " in draft.requests_seen[0].prompt
    failed = [e for e in result.trace if e.node == "head" and e.kind == "failed"]
    assert failed and failed[0].metadata.get("partial") is True


async def test_continuation_failure_carries_committed_text_in_stream_error():
    head = StreamScriptedBackend(["Intro. "])
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["Part ", "never"], fail_after=1)
    conductor = _conductor(head, draft, continuation)

    emitted = ""
    with pytest.raises(ConductorStreamError) as excinfo:
        async for event in conductor.stream("task"):
            if event.kind == "delta":
                emitted += event.text
    assert excinfo.value.result.final_text == emitted == "Intro. Part "
    # The failed stream released its reservation.
    state = excinfo.value.result.budget_state
    assert state.steps_reserved == 0


async def test_head_skipped_for_incompatible_public_intent():
    head = StreamScriptedBackend(["Intro. "])
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["Full answer."])
    conductor = _conductor(
        head,
        draft,
        continuation,
        final_sampling_params=SamplingParams(max_tokens=64, n=2),
    )

    events = await _collect(conductor.stream("task"))
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Full answer."
    assert "head" not in result.outputs
    skipped = [e for e in result.trace if e.kind == "skipped:intent"]
    assert skipped and skipped[0].node == "head"
    # Non-streaming parity: the head stays disabled in run() too.
    run_result = await conductor.run("task")
    assert run_result.final_text == "Full answer."
    assert "head" not in run_result.outputs


async def test_run_and_stream_concatenation_parity():
    def build():
        return _conductor(
            StreamScriptedBackend(["Intro. "]),
            StreamScriptedBackend(["draft"]),
            StreamScriptedBackend(["Body."]),
        )

    run_result = await build().run("task")
    events = await _collect(build().stream("task"))
    stream_result = events[-1].result
    assert stream_result is not None
    assert run_result.final_text == stream_result.final_text == "Intro. Body."
    assert run_result.completions[0].text == "Intro. Body."


async def test_head_reserves_step_within_max_steps():
    head = StreamScriptedBackend(["Intro. "])
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["Body."])
    conductor = _conductor(head, draft, continuation)

    events = await _collect(
        conductor.stream("task", budget=Budget(max_steps=3))
    )
    result = events[-1].result
    assert result is not None
    assert result.budget_state.steps_used == 3
    # One step fewer and the continuation is refused, best-so-far = head text.
    events = await _collect(
        _conductor(
            StreamScriptedBackend(["Intro. "]),
            StreamScriptedBackend(["draft"]),
            StreamScriptedBackend(["Body."]),
        ).stream("task", budget=Budget(max_steps=2))
    )
    result = events[-1].result
    assert result is not None
    assert result.final_text == "Intro. "


async def test_per_role_sampling_overrides_reach_backend_requests():
    head = StreamScriptedBackend(["Intro. "])
    draft = StreamScriptedBackend(["draft"])
    continuation = StreamScriptedBackend(["Body."])
    roles = (
        RoleSpec(
            name="head",
            worker="hw",
            role_type="head",
            prompt="[head] {query}",
            sampling=RoleSamplingOverrides(
                temperature=0.3,
                max_tokens=256,
                stop=("\n## ",),
            ),
        ),
        RoleSpec(
            name="draft",
            worker="dw",
            prompt="[draft] {query} {head}",
            depends_on=("head",),
            sampling=RoleSamplingOverrides(
                temperature=0.9,
                max_tokens=9000,
                seed_offset=2,
            ),
        ),
        RoleSpec(
            name="continuation",
            worker="cw",
            role_type="publisher",
            prompt="[cont] {head} {draft}",
            depends_on=("head", "draft"),
        ),
    )
    conductor = Conductor(
        roles,
        {"hw": head, "dw": draft, "cw": continuation},
        sampling_params=SamplingParams(max_tokens=1024, seed=7),
        final_sampling_params=SamplingParams(max_tokens=4096, temperature=0.6),
    )
    await _collect(conductor.stream("task"))

    head_params = head.requests_seen[0].sampling_params
    assert head_params.temperature == 0.3
    assert head_params.max_tokens == 256
    assert head_params.stop == ("\n## ",)
    assert head_params.n == 1
    draft_params = draft.requests_seen[0].sampling_params
    assert draft_params.temperature == 0.9
    # Internal ceiling caps the override (m11 #208).
    assert draft_params.max_tokens == 1024
    assert draft_params.seed == 9
    continuation_params = continuation.requests_seen[0].sampling_params
    assert continuation_params.max_tokens == 4096
    assert continuation_params.temperature == 0.6


def test_final_role_sampling_override_rejected():
    roles = (
        RoleSpec(name="solo", worker="w", prompt="{query}",
                 sampling=RoleSamplingOverrides(temperature=0.1)),
    )
    with pytest.raises(ValueError, match="public"):
        Conductor(roles, {"w": StreamScriptedBackend(["x"])})


def test_head_dag_shape_validation():
    backend = StreamScriptedBackend(["x"])
    with pytest.raises(ValueError, match="dependencies"):
        Conductor(
            (
                RoleSpec(name="head", worker="w", role_type="head",
                         prompt="{query}", depends_on=("other",)),
                RoleSpec(name="other", worker="w", prompt="{query}"),
            ),
            {"w": backend},
        )
    with pytest.raises(ValueError, match="at most one"):
        Conductor(
            (
                RoleSpec(name="h1", worker="w", role_type="head", prompt="{query}"),
                RoleSpec(name="h2", worker="w", role_type="head", prompt="{query}"),
                RoleSpec(name="final", worker="w", prompt="{h1}{h2}",
                         depends_on=("h1", "h2")),
            ),
            {"w": backend},
        )
    with pytest.raises(ValueError, match="depend on"):
        # The selected final unit (the synthesizer) does not transitively
        # depend on the head, so it could not continue the committed prefix.
        Conductor(
            (
                RoleSpec(name="head", worker="w", role_type="head", prompt="{query}"),
                RoleSpec(name="uses", worker="w", prompt="{head}",
                         depends_on=("head",)),
                RoleSpec(name="final", worker="w", role_type="synthesizer",
                         prompt="{query}"),
            ),
            {"w": backend},
        )


def test_initial_requests_include_enabled_head_and_exclude_disabled():
    def build(**kwargs):
        return _conductor(
            StreamScriptedBackend(["Intro. "]),
            StreamScriptedBackend(["draft"]),
            StreamScriptedBackend(["Body."]),
            **kwargs,
        )

    enabled = build()
    names = [spec.name for spec, _ in enabled.initial_requests("q", request_id_suffix="s")]
    assert names == ["head"]
    disabled = build(final_sampling_params=SamplingParams(max_tokens=64, n=2))
    assert [
        spec.name for spec, _ in disabled.initial_requests("q", request_id_suffix="s")
    ] == []


async def test_prompt_suffix_survives_refinement_attempts():
    # A worker-native scaffold (e.g. a chat assistant marker) must terminate
    # EVERY attempt's prompt; refinement feedback goes before it, or the
    # refined generation runs against a malformed prompt.
    class Recorder:
        def __init__(self, responses):
            self._responses = list(responses)
            self.prompts = []

        async def generate(self, request):
            from kairyu.engine.backend import GenerationResult

            self.prompts.append(request.prompt)
            return GenerationResult(
                request_id=request.request_id,
                prompt=request.prompt,
                completions=(
                    CompletionOutput(
                        index=0, text=self._responses.pop(0), token_ids=()
                    ),
                ),
            )

        async def stream(self, request):  # pragma: no cover - unused
            yield await self.generate(request)

        async def shutdown(self):
            return None

    backend = Recorder(["draft-1", "FAIL: broken", "draft-2", "PASS", "final"])
    roles = (
        RoleSpec(
            name="draft",
            worker="w",
            prompt="[draft] {query}",
            prompt_suffix="<ASSISTANT>",
        ),
        RoleSpec(
            name="check",
            worker="w",
            role_type="verifier",
            verifies="draft",
            prompt="[check] {draft}",
            prompt_suffix="<THINK>",
            depends_on=("draft",),
        ),
        RoleSpec(
            name="final",
            worker="w",
            role_type="synthesizer",
            prompt="[final] {draft} {check}",
            depends_on=("draft", "check"),
        ),
    )
    conductor = Conductor(roles, {"w": backend})
    result = await conductor.run("task")

    assert result.final_text == "final"
    draft_prompts = [p for p in backend.prompts if p.startswith("[draft]")]
    assert len(draft_prompts) == 2
    assert all(p.endswith("<ASSISTANT>") for p in draft_prompts)
    assert "Verifier feedback:" in draft_prompts[1]
    assert draft_prompts[1].index("Verifier feedback:") < draft_prompts[1].index(
        "<ASSISTANT>"
    )
    check_prompts = [p for p in backend.prompts if p.startswith("[check]")]
    assert all(p.endswith("<THINK>") for p in check_prompts)
