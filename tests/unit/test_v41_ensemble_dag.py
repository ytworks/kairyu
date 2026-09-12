"""Exercise the shipped example's checklist flow with scripted model responses."""

import asyncio
import json
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

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/qwen3.8-deepseek-v4.1-8gpu"
CHECKLIST = json.dumps(
    [
        {
            "id": "R1",
            "priority": "minimum",
            "requirement": "Compare both options",
            "acceptance_criterion": "Discuss A and B",
            "source": "Compare A and B",
        }
    ]
)
DEFECT = "FAIL\nR1 | unsatisfied | evidence: B is absent | correction: Add B"
ANSWER = "A is inexpensive. B is faster."


class ExampleBackend:
    def __init__(self, *, with_image: bool, fail_audits: int):
        self.with_image = with_image
        self.fail_audits = fail_audits
        self.requests: dict[str, list] = {}
        self.roots = {name: asyncio.Event() for name in ("draft", "requirements")}

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
        if role in self.roots:
            self.roots[role].set()
            other = "draft" if role == "requirements" else "requirements"
            await self.roots[other].wait()
        if role == "requirements":
            output = CHECKLIST
        elif role == "audit":
            output = (
                DEFECT
                if len(self.requests[role]) <= self.fail_audits
                else (
                    "PASS\nR1 | satisfied | evidence: Both A and B are compared | correction: none"
                )
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


@pytest.mark.parametrize("effort", [None, "low", "high", "max"])
@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("head_enabled", [False, True])
@pytest.mark.parametrize("fail_audits", [0, 1, 3])
async def test_example_checklist_survives_parallel_roots_and_refinement(
    with_image,
    head_enabled,
    fail_audits,
    effort,
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
        if with_image
        else None
    )
    conductor = Conductor(
        roles=tuple(roles),
        workers={worker.name: backend for worker in spec.workers},
        sampling_params=SamplingParams(max_tokens=65536),
        final_structured_format_in_prompt=not head_enabled,
        multimodal_prompt=media,
        chat_template_kwargs={"enable_thinking": False} if with_image else None,
        reasoning_effort=effort or spec.default_reasoning_effort,
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
    assert "image_description" not in backend.requests
    assert not {"answer_3", "answer_4"} & backend.requests.keys()
    for name in ("requirements",):
        request, _ = backend.requests[name][0]
        assert (
            request.reasoning_effort == "high"
        )  # Fixed native DeepSeek high, independent of caller.
        assert request.sampling_params.max_tokens == (8192 if name == "requirements" else 4096)
        if with_image:
            assert request.chat_template_kwargs == {"enable_thinking": True}
    for name in (
        "policies",
        "critique",
        "synthesis",
        "audit",
        *(f"answer_{i}" for i in range(1, 3)),
    ):
        assert all(CHECKLIST in text for _, text in backend.requests[name])
    for name in ("requirements", "policies", "critique", "synthesis"):
        request, text = backend.requests[name][0]
        assert "<｜User｜>" not in text
        if name != "requirements":
            assert request.reasoning_effort == (effort or "high")
        if with_image:
            assert isinstance(request.prompt, MultimodalPrompt)
            assert request.prompt.items == media.items
    assert "{requirements}" not in backend.requests["audit"][0][1]
    if fail_audits:
        assert DEFECT in backend.requests["synthesis"][1][1]
    expected_attempts = min(fail_audits + 1, 3)
    assert len(backend.requests["audit"]) == expected_attempts
    assert len(backend.requests["synthesis"]) == expected_attempts
    assert (ANSWER if fail_audits else "A is inexpensive.") in result.final_text
    assert CHECKLIST not in result.final_text and DEFECT not in result.final_text
    # Document the existing framework contract: exhausted FAIL is observable,
    # but the last answer is still returned. This config adds no hard gate.
    assert result.outputs["audit"].startswith("FAIL" if fail_audits == 3 else "PASS")
    verdicts = [
        event for event in result.trace if event.node == "audit" and "pass" in event.metadata
    ]
    assert len(verdicts) == expected_attempts
    assert verdicts[-1].metadata["pass"] is (fail_audits != 3)
    assert verdicts[-1].metadata["refinement_exhausted"] is (fail_audits == 3)
    # Initial PASS stops immediately even though the scripted answer omits B:
    # this characterizes control flow, not the model judgment's correctness.


def test_qwen_keeps_medium_alias_and_requirements_uses_deepseek():
    spec = load_spec(EXAMPLE / "auto-max.yaml")
    assert next(r for r in spec.roles if r.name == "requirements").worker == "tier2"
    template = ChatTemplate.load(str(EXAMPLE / "qwen3.8-chat.jinja"))
    rendered = template.render(
        [{"role": "user", "content": "Compare A and B"}],
        template_kwargs={"reasoning_effort": "high"},
    )
    assert "Reason privately and carefully before answering." in rendered


@pytest.mark.parametrize(
    "profile_name", ["qwen_direct", "qwen_think_medium", "deepseek_direct", "deepseek_think"]
)
async def test_direct_profiles_bypass_checklist_and_preserve_raw_images(profile_name):
    spec = load_spec(EXAMPLE / "auto-max.yaml")
    profile = next(p for p in spec.profiles if p.name == profile_name)
    backend = ExampleBackend(with_image=True, fail_audits=0)
    role = profile.roles[0]
    values = {field.name: getattr(role, field.name) for field in fields(RoleSpec)}
    values["sampling"] = RoleSamplingOverrides(**role.sampling.model_dump())
    media = MultimodalPrompt(
        "Compare A and B", (MultimodalItem("image", "uri", "data:image/png;base64,AAAA"),)
    )
    result = await Conductor(
        roles=(RoleSpec(**values),),
        workers={w.name: backend for w in spec.workers},
        sampling_params=SamplingParams(max_tokens=65536),
        reasoning_effort="max",
        multimodal_prompt=media,
    ).run("Compare A and B", Budget(max_steps=18, max_refine_depth=2))
    assert result.final_unit_ok
    assert set(backend.requests) == {role.name}
    request, _ = backend.requests[role.name][0]
    assert request.prompt.items == media.items
    expected = (
        "max"
        if profile_name == "deepseek_think"
        else ("high" if profile_name == "qwen_think_medium" else None)
    )
    assert request.reasoning_effort == expected


async def test_shipped_native_deepseek_backend_forwards_tools_and_image():
    import httpx
    import yaml

    from kairyu.engine.openai_backend import OpenAICompatBackend

    config = yaml.safe_load((EXAMPLE / "kairyu.yaml").read_text())
    options = config["pools"]["deepseek-v4.1-flash"]["replicas"][0]["options"]
    captured = []

    def upstream(request):
        captured.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 8, "total_tokens": 108},
            },
        )

    backend = OpenAICompatBackend(**options, transport=httpx.MockTransport(upstream))
    spec = load_spec(EXAMPLE / "auto-max.yaml")
    role = next(p for p in spec.profiles if p.name == "deepseek_direct").roles[0]
    values = {field.name: getattr(role, field.name) for field in fields(RoleSpec)}
    values["sampling"] = RoleSamplingOverrides(**role.sampling.model_dump())
    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP8z8DAwMDA"
        "xMDAwMDAAAANHQEDasKb6QAAAABJRU5ErkJggg=="
    )
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    try:
        result = await Conductor(
            roles=(RoleSpec(**values),),
            workers={"tier2-direct": backend},
            sampling_params=SamplingParams(max_tokens=512),
            final_tools=(tool,),
            final_tool_choice="required",
            multimodal_prompt=MultimodalPrompt(
                "Inspect image", (MultimodalItem("image", "uri", image),)
            ),
        ).run("Inspect image", Budget(max_steps=18, max_refine_depth=2))
        assert result.final_unit_ok, [(e.node, e.detail) for e in result.trace]
        assert len(captured) == 1
        path, body = captured[0]
        assert path == "/v1/chat/completions"
        assert body["tools"] == [tool] and body["tool_choice"] == "required"
        assert any(p["type"] == "image_url" for p in body["messages"][0]["content"])
        assert "prompt" not in body
        assert "<tool_call>" in result.final_text
    finally:
        await backend.shutdown()
