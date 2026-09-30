"""Internal roles whose reasoning consumes their whole budget still produce
output: the first attempt thinks within max_tokens - output_floor, and an
attempt that ends inside its reasoning span is continued once after a forced
close. Without it a thinking planner or verifier left its dependents an empty
slot after ~950 s of deliberation (PR #602, V4.1 six-GPU runs).
"""

from __future__ import annotations

from kairyu.engine.backend import GenerationRequest, GenerationResult, GenerationUsage
from kairyu.engine.prompt import TemplatedPrompt
from kairyu.orchestration.budget import Budget
from kairyu.orchestration.conductor import Conductor, RoleSamplingOverrides, RoleSpec
from kairyu.outputs import CompletionOutput


class DryFirstBackend:
    """Tagged roles spend their first attempt inside the reasoning span."""

    def __init__(self, dry: tuple[str, ...], rendered: str | None = None) -> None:
        self.dry = set(dry)
        self.rendered = rendered
        self.requests: list[GenerationRequest] = []
        self.render_requests: list[GenerationRequest] = []

    @staticmethod
    def _tag(request: GenerationRequest) -> str:
        return str(request.prompt).split("]", 1)[0].rsplit("[", 1)[-1]

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        tag = self._tag(request)
        if tag in self.dry:
            self.dry.discard(tag)
            completion = CompletionOutput(
                index=0,
                text="",
                token_ids=(),
                finish_reason="length",
                reasoning_content=f"{tag} deliberation",
            )
        else:
            text = "PASS" if tag == "check" else f"{tag}-output"
            completion = CompletionOutput(index=0, text=text, token_ids=(), finish_reason="stop")
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(completion,),
            usage=GenerationUsage(prompt_tokens=5, completion_tokens=7),
            finished=True,
        )

    async def stream(self, request):  # pragma: no cover - unused
        yield await self.generate(request)

    async def shutdown(self) -> None:
        return None


class RenderingBackend(DryFirstBackend):
    async def render_generation_prompt_async(self, request):
        self.render_requests.append(request)
        return None if self.rendered is None else TemplatedPrompt(self.rendered)


def _roles(continuation: str, *, floor: int = 64) -> tuple[RoleSpec, ...]:
    open_tag = "<think>" if continuation == "chat" else ""
    return (
        RoleSpec(
            name="plan",
            worker="w",
            prompt="[plan] {query}",
            role_type="planner",
            reasoning_effort="high",
            reasoning_close_tag="</think>",
            reasoning_open_tag=open_tag,
            reasoning_continuation=continuation,
            output_floor=floor,
            sampling=RoleSamplingOverrides(max_tokens=1024),
        ),
        RoleSpec(
            name="final",
            worker="w",
            prompt="[final] {plan}",
            role_type="synthesizer",
            depends_on=("plan",),
        ),
    )


async def test_chat_role_continues_after_a_forced_close():
    backend = RenderingBackend(dry=("plan",))
    conductor = Conductor(roles=_roles("chat"), workers={"w": backend})
    result = await conductor.run("task")

    first, retry, final = backend.requests
    assert first.sampling_params.max_tokens == 1024 - 64
    assert retry.sampling_params.max_tokens == 64
    assert retry.assistant_prefill == "<think>\nplan deliberation\n</think>\n\n"
    # The dependent reads the continued output, not an empty slot.
    assert "plan-output" in str(final.prompt)
    retry_event = next(e for e in result.trace if e.kind == "retry:empty_output")
    assert retry_event.node == "plan"
    assert retry_event.metadata["continuation"] == "think_close"
    assert retry_event.usage is not None and retry_event.usage.completion_tokens == 7


async def test_rendered_role_extends_the_upstream_generation_prompt():
    backend = RenderingBackend(dry=("plan",), rendered="<sys>[plan] task<asst><think>")
    conductor = Conductor(roles=_roles("rendered"), workers={"w": backend})
    result = await conductor.run("task")

    _, retry, final = backend.requests
    assert isinstance(retry.prompt, TemplatedPrompt)
    assert retry.prompt == "<sys>[plan] task<asst><think>plan deliberation</think>"
    assert retry.sampling_params.max_tokens == 64
    # The render request is the role's own chat call (effort included).
    assert backend.render_requests[0].reasoning_effort == "high"
    assert "plan-output" in str(final.prompt)
    assert result.final_text == "final-output"


async def test_rendered_role_without_an_upstream_render_keeps_its_empty_attempt():
    backend = RenderingBackend(dry=("plan",), rendered=None)
    conductor = Conductor(roles=_roles("rendered"), workers={"w": backend})
    result = await conductor.run("task")

    assert [DryFirstBackend._tag(r) for r in backend.requests] == ["plan", "final"]
    assert result.outputs["plan"] == ""


async def test_verifier_states_the_verdict_it_deliberated_to():
    backend = RenderingBackend(dry=("check",))
    roles = (
        RoleSpec(name="draft", worker="w", prompt="[draft] {query}", role_type="publisher"),
        RoleSpec(
            name="check",
            worker="w",
            role_type="verifier",
            verifies="draft",
            depends_on=("draft",),
            prompt="[check] {draft}",
            reasoning_effort="high",
            reasoning_close_tag="</think>",
            reasoning_open_tag="<think>",
            reasoning_continuation="chat",
            output_floor=32,
            sampling=RoleSamplingOverrides(max_tokens=512),
        ),
    )
    conductor = Conductor(roles=roles, workers={"w": backend})
    result = await conductor.run("task", budget=Budget(max_refine_depth=2))

    draft, verdict, continued = backend.requests
    assert verdict.sampling_params.max_tokens == 512 - 32
    assert continued.assistant_prefill == "<think>\ncheck deliberation\n</think>\n\n"
    assert continued.sampling_params.max_tokens == 32
    assert "Answer immediately" not in str(continued.prompt)
    assert result.final_text == "draft-output"
    assert result.outputs["check"] == "PASS"
