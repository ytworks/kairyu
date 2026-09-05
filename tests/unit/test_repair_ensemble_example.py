"""Behavior checks for the Example's additive configuration, using existing L2."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import httpx
import pytest
import yaml
from jinja2 import Environment

from kairyu.dsl.loader import build_orchestrator, load_spec
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.orchestration.request import OrchestrationRequest
from kairyu.sampling_params import SamplingParams

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/qwen3.8-deepseek-v4-8gpu"
# Canonical semantic digest of the entire pre-addition config at 097affb1.
ORIGINAL = "a9603c853c7170693fe4fb717cdb31c4f3fbf6ef629b33c0514053bac1a9ed71"
TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
)


def raw_config():
    return yaml.safe_load((EXAMPLE / "auto-max.yaml").read_text())


def test_only_one_profile_and_choice_added_to_existing_config():
    config = copy.deepcopy(raw_config())
    assert config["profiles"].pop()["name"] == "swe_evidence"
    assert config["profile_judge"]["choices"].pop()["profile"] == "swe_evidence"
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == ORIGINAL


def test_repair_caps_and_medium_qwen_template():
    spec = load_spec(EXAMPLE / "auto-max.yaml")
    roles = next(p.roles for p in spec.profiles if p.name == "swe_evidence")
    regular = [r for r in roles if r.requires != "image"]
    assert len(regular) == 6
    assert sum(r.sampling.max_tokens for r in regular) == 30720
    assert not any(r.role_type in {"head", "verifier", "executor"} for r in roles)
    template = Environment().from_string((EXAMPLE / "qwen3.8-chat.jinja").read_text())
    for role in roles:
        if role.worker == "tier1" and role.name != "swe_final":
            assert role.reasoning_effort == "high"  # DTO-D14's fixed medium alias
            rendered = template.render(
                messages=[{"role": "user", "content": "probe"}],
                reasoning_effort=role.reasoning_effort,
                add_generation_prompt=True,
            )
            assert "Keep the private scratchpad proportional to the difficulty" in rendered
            assert rendered.endswith("<think>\n")
    final = next(r for r in roles if r.name == "swe_final")
    assert final.worker == "tier1" and final.reasoning_effort is None
    perception = next(r for r in roles if r.name == "image_description")
    original_perception = next(r for r in spec.roles if r.name == "image_description")
    assert perception == original_perception
    for name in ("swe_plan", "swe_decision"):
        role = next(r for r in roles if r.name == name)
        assert "image_description" in role.depends_on
        assert "{image_description}" in role.prompt


def build_transport(label, *, empty_finals=0):
    captures = []
    final_count = 0

    def handler(request):
        nonlocal final_count
        body = json.loads(request.content)
        captures.append((request.url.path, body))
        text = body["messages"][0]["content"]
        match = re.search(r"\[(swe_[a-z]+)\]", text)
        if "You route requests inside an inference product" in text:
            answer = label
        elif match:
            role = match.group(1)
            if role == "swe_plan":
                assert "evidence:swe_hypothesis" in text and "evidence:swe_alternative" in text
            if role == "swe_critic":
                assert "evidence:swe_plan" in text
            if role == "swe_decision":
                assert "evidence:swe_plan" in text and "evidence:swe_critic" in text
            if role == "swe_final":
                assert "evidence:swe_decision" in text
            answer = "evidence:" + role
        elif "[audit]" in text:
            answer = "PASS"
        else:
            answer = "legacy answer"
        if match and match.group(1) == "swe_final":
            final_count += 1
            if final_count <= empty_finals:
                answer = ""
        message = {"role": "assistant", "content": answer}
        finish = "stop"
        if body.get("tools"):
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_test",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"pytest -q"}'},
                    }
                ],
            }
            finish = "tool_calls"
        usage = {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13}
        if body.get("stream"):
            chunks = [
                {"choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
            ]
            content = (
                "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": usage,
            },
        )

    transport = httpx.MockTransport(handler)
    qwen = OpenAICompatBackend(
        "https://qwen.test/v1", "qwen", api_key_env=None, upstream="vllm", transport=transport
    )
    deepseek = OpenAICompatBackend(
        "https://deepseek.test/v1",
        "deepseek",
        api_key_env=None,
        upstream="vllm",
        transport=transport,
        allow_templated_chat_passthrough=True,
    )
    data = raw_config()
    data["router"]["artifact"] = str(EXAMPLE / "router.json")
    orchestrator = build_orchestrator(
        load_spec(yaml.safe_dump(data)),
        engine_refs={
            "qwen3.8-27b": qwen,
            "deepseek-v4-flash-0731-thinking": deepseek,
            "deepseek-v4-flash-0731": deepseek,
        },
    )
    return orchestrator, captures, (qwen, deepseek)


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["low", "max"])
async def test_actual_chat_transport_publishes_tools_only_from_qwen(effort):
    orchestrator, captures, engines = build_transport("SWE")
    call = OrchestrationRequest(
        prompt="Repair the repository; conflicting failures span files.",
        sampling_params=SamplingParams(max_tokens=8192),
        tools=TOOLS,
        tool_choice="required",
        reasoning_effort=effort,
    )
    try:
        call = await orchestrator.judge_role_profile(call)
        result = await orchestrator.run(call)
        assert call.role_profile_judgment == "swe_evidence"
        assert "<tool_call>" in result.text and "pytest -q" in result.text
        assert len(captures) == 7  # one existing judge + six text roles
        assert all(path == "/v1/chat/completions" for path, _ in captures)
        publishers = [body for _, body in captures if body.get("tools")]
        assert len(publishers) == 1
        assert publishers[0]["model"] == "qwen"
        assert publishers[0]["tools"] == list(TOOLS)
        assert publishers[0]["tool_choice"] == "required"
        assert result.completion_tokens == 14  # all seven billed calls
        for _, body in captures:
            text = body["messages"][0]["content"]
            if any(
                f"[{name}]" in text for name in ("swe_hypothesis", "swe_alternative", "swe_critic")
            ):
                assert body["reasoning_effort"] == "high"
                assert body["max_tokens"] == 2048
    finally:
        for engine in engines:
            await engine.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "profile"),
    [
        ("QWEN", "qwen_direct"),
        ("QWEN_THINK", "qwen_think_medium"),
        ("DEEPSEEK", "deepseek_direct"),
        ("DEEPSEEK_THINK", "deepseek_think"),
        ("ENSEMBLE", "primary"),
        ("invalid", "primary"),
    ],
)
async def test_retained_routes_and_fallback(label, profile):
    orchestrator, captures, engines = build_transport(label)
    call = OrchestrationRequest(
        prompt="Explain this problem.", sampling_params=SamplingParams(max_tokens=1024)
    )
    try:
        call = await orchestrator.judge_role_profile(call)
        assert orchestrator._role_profile(call) == profile
        result = await orchestrator.run(call)
        assert result.text
        assert not any("[swe_" in body["messages"][0]["content"] for _, body in captures)
        if profile != "primary":
            assert len(captures) == 2  # same judge and one final call
    finally:
        for engine in engines:
            await engine.shutdown()


@pytest.mark.asyncio
async def test_new_route_stream_has_only_final_public_content():
    orchestrator, captures, engines = build_transport("SWE")
    call = OrchestrationRequest(
        prompt="Repair conflicting repository behavior.",
        sampling_params=SamplingParams(max_tokens=8192),
    )
    try:
        call = await orchestrator.judge_role_profile(call)
        stream = await orchestrator.run_chat(call, stream=True)
        events = [event async for event in stream]
        assert "".join(e.text for e in events if e.kind == "delta") == "evidence:swe_final"
        assert len(captures) == 7
        assert events[-1].kind == "result"
    finally:
        for engine in engines:
            await engine.shutdown()


def test_verification_rejects_missing_new_internal_stage(tmp_path):
    from tests.unit.test_tiered_frontier_examplectl import _write_product_serving_result

    module_spec = importlib.util.spec_from_file_location(
        "repair_verification", EXAMPLE / "verification.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    roles = [r for r in raw_config()["profiles"][-1]["roles"] if r.get("requires") != "image"]
    stages = [{"node": "profile_judge", "kind": "classification", "status": "success"}] + [
        {"node": r["name"], "role": r["role_type"], "kind": "generation", "status": "success"}
        for r in roles
    ]

    def validate():
        return module._validate_serving_row(
            tmp_path,
            1,
            32,
            expected_route="synthesis",
            expected_role="publisher",
            judged_routes=True,
        )

    _write_product_serving_result(tmp_path, stages)
    assert validate() == 0
    for role in roles:
        _write_product_serving_result(tmp_path, [s for s in stages if s["node"] != role["name"]])
        assert validate() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_finals", [1, 2])
async def test_final_empty_recovery_is_bounded_and_respects_caller_cap(empty_finals):
    orchestrator, captures, engines = build_transport("SWE", empty_finals=empty_finals)
    call = OrchestrationRequest(
        prompt="Repair inconsistent repository behavior.",
        sampling_params=SamplingParams(max_tokens=512),
    )
    try:
        call = await orchestrator.judge_role_profile(call)
        if empty_finals == 1:
            result = await orchestrator.run(call)
            assert result.text == "evidence:swe_final"
        else:
            from kairyu.orchestration.orchestrator import OrchestratorExecutionError

            with pytest.raises(OrchestratorExecutionError):
                await orchestrator.run(call)
        finals = [b for _, b in captures if "[swe_final]" in b["messages"][0]["content"]]
        assert len(finals) == 2
        assert all(b["max_tokens"] <= 512 for b in finals)
        assert len(captures) == 8
    finally:
        for engine in engines:
            await engine.shutdown()
