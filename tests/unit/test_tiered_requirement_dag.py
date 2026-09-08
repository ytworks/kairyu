"""Exercise the shipped example's checklist flow with scripted model responses."""

import asyncio
import re
from dataclasses import fields
from pathlib import Path

import pytest

from kairyu.dsl.loader import load_spec
from kairyu.engine.backend import GenerationResult
from kairyu.engine.prompt import MultimodalItem, MultimodalPrompt, prompt_text
from kairyu.entrypoints.chat_template import ChatTemplate
from kairyu.orchestration.budget import Budget
from kairyu.orchestration.conductor import Conductor, RoleSamplingOverrides, RoleSpec
from kairyu.outputs import CompletionOutput
from kairyu.sampling_params import SamplingParams

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/qwen3.8-deepseek-v4-8gpu"
CHECKLIST = (
    "R1 | priority: minimum | requirement: Compare both options | "
    "acceptance_criterion: Discuss A and B | source: Compare A and B"
)
DEFECT = "FAIL\nR1 | unsatisfied | evidence: B is absent | correction: Add B"
ANSWER = "A is inexpensive. B is faster."


class ExampleBackend:
    def __init__(self, *, with_image: bool, fail_audits: int):
        self.with_image = with_image
        self.fail_audits = fail_audits
        self.requests: dict[str, list] = {}
        self.started = {name: asyncio.Event() for name in ("requirements", "image_description")}

    def supports_prompt_kind(self, kind):
        return kind in {"text", "multimodal"}

    def supports_chat_template_kwargs(self, keys):
        return keys <= {"enable_thinking"}

    async def generate(self, request):
        prompt = request.prompt
        text = prompt_text(prompt.base if isinstance(prompt, MultimodalPrompt) else prompt)
        # The role marker precedes any embedded candidate content.
        role = re.search(r"\[(\w+)\]", text).group(1)
        self.requests.setdefault(role, []).append((request, text))
        if role in self.started:
            self.started[role].set()
            if self.with_image:
                other = "image_description" if role == "requirements" else "requirements"
                # A sequential schedule deadlocks here and fails the bounded run.
                await self.started[other].wait()
        if role == "requirements":
            output = CHECKLIST
        elif role == "image_description":
            output = "The chart labels A and B."
        elif role == "audit":
            output = DEFECT if len(self.requests[role]) <= self.fail_audits else (
                "PASS\nR1 | satisfied | evidence: Both A and B are compared"
            )
        elif role == "synthesis":
            output = "A is inexpensive." if len(self.requests[role]) == 1 else ANSWER
        elif role == "head":
            output = "Comparison:"
        else:
            output = "Candidate data."
        return GenerationResult(
            request_id=request.request_id,
            prompt=prompt,
            completions=(CompletionOutput(index=0, text=output, token_ids=()),),
        )

    async def stream(self, request):
        yield await self.generate(request)

    async def shutdown(self):
        pass


@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("head_enabled", [False, True])
@pytest.mark.parametrize("fail_audits", [1, 3])
async def test_example_checklist_survives_parallel_roots_and_refinement(
    with_image, head_enabled, fail_audits,
):
    spec = load_spec(EXAMPLE / "auto-max.yaml")
    backend = ExampleBackend(with_image=with_image, fail_audits=fail_audits)
    roles = []
    for role in spec.roles:
        values = {field.name: getattr(role, field.name) for field in fields(RoleSpec)}
        values["sampling"] = (
            RoleSamplingOverrides(**role.sampling.model_dump()) if role.sampling else None
        )
        roles.append(RoleSpec(**values))
    media = (
        MultimodalPrompt(
            "Compare A and B",
            (MultimodalItem("image", "uri", "data:image/png;base64,AAAA"),),
        )
        if with_image else None
    )
    conductor = Conductor(
        roles=tuple(roles),
        workers={worker.name: backend for worker in spec.workers},
        sampling_params=SamplingParams(max_tokens=65536),
        final_structured_format_in_prompt=not head_enabled,
        multimodal_prompt=media,
        chat_template_kwargs={"enable_thinking": False} if with_image else None,
        reasoning_effort="low",
    )
    result = await asyncio.wait_for(
        conductor.run(
            "Compare A and B",
            Budget(max_steps=spec.budget.max_steps, max_refine_depth=spec.budget.max_refine_depth),
        ),
        timeout=5,
    )

    assert result.final_unit_ok, [(e.node, e.detail) for e in result.trace]
    assert len(backend.requests["requirements"]) == 1
    assert ("image_description" in backend.requests) is with_image
    for name in ("requirements", "image_description") if with_image else ("requirements",):
        request, _ = backend.requests[name][0]
        assert request.reasoning_effort == "high"  # Fixed medium alias, independent of caller.
        assert request.sampling_params.max_tokens == (8192 if name == "requirements" else 4096)
        if with_image:
            assert request.chat_template_kwargs == {"enable_thinking": True}
    for name in (
        "policies", "critique", "synthesis", "audit",
        *(f"answer_{i}" for i in range(1, 5)),
    ):
        assert all(CHECKLIST in text for _, text in backend.requests[name])
    assert "{requirements}" not in backend.requests["audit"][0][1]
    assert DEFECT in backend.requests["synthesis"][1][1]
    expected_attempts = 2 if fail_audits == 1 else 3
    assert len(backend.requests["audit"]) == expected_attempts
    assert len(backend.requests["synthesis"]) == expected_attempts
    assert ANSWER in result.final_text
    assert CHECKLIST not in result.final_text and DEFECT not in result.final_text
    # Document the existing framework contract: exhausted FAIL is observable,
    # but the last answer is still returned. This config adds no hard gate.
    assert result.outputs["audit"].startswith("PASS" if fail_audits == 1 else "FAIL")


def test_parallel_roots_render_the_existing_medium_thinking_template():
    spec = load_spec(EXAMPLE / "auto-max.yaml")
    template = ChatTemplate.load(str(EXAMPLE / "qwen3.8-chat.jinja"))
    for role in spec.roles:
        if role.name not in {"requirements", "image_description"}:
            continue
        rendered = template.render(
            [{"role": "user", "content": "Compare A and B"}],
            template_kwargs={"reasoning_effort": role.reasoning_effort},
        )
        assert "Reason privately and carefully before answering." in rendered
        assert rendered.endswith("<|im_start|>assistant\n<think>\n")
