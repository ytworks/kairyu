"""Routed answers: DeepSeek-V4.1 six-GPU + two Quyet replicas example (VCO-D21, VCO-D22).

The example's own kairyu.yaml / verified-tool.yaml drive the production loaders, the
real OpenAI backend (against a fake vLLM) and the real System One backend
(against fake Quyet adapters that, like the real one, refuse more than 32
questions per request), so the tests observe what the deployed L1 services would
receive and which route answers.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import httpx
import pytest
import yaml

from kairyu.deploy.builder import build_app_from_spec
from kairyu.deploy.spec import load_deployment_spec
from kairyu.dsl.loader import build_orchestrator, load_spec
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.engine.systemone import HTTPSystemOneBackend
from kairyu.entrypoints.server.chat_service import validate_orchestration_chat_input
from kairyu.entrypoints.server.protocol import ChatCompletionRequest
from kairyu.orchestration.request import OrchestrationRequest
from kairyu.sampling_params import SamplingParams

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/deepseek-v4.1-quyet-8gpu"

# The real adapter's QUYET_MAX_QUESTIONS (example.json).
QUYET_MAX_QUESTIONS = 32
CANDIDATES = {
    "candidates": [
        {
            "id": f"C{k}",
            "viewpoint": f"view {k}",
            "purpose": f"step {k}",
            "name": "bash",
            "arguments": json.dumps({"command": f"step-{k}"}),
        }
        for k in range(1, 11)
    ]
}
BASH = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}
# An agent turn: the conversation ends with a tool result.
AGENT_TURN = [
    {"role": "system", "content": "You can run shell commands."},
    {"role": "user", "content": "Fix the failing test in /app."},
    {
        "role": "assistant",
        "content": "Running the tests first.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "pytest -q"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "1 failed: test_parse"},
]
# The move the fake DeepSeek makes: short text and one structured call.
MOVE = {
    "role": "assistant",
    "content": "Reading the failing test.",
    "tool_calls": [
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "cat tests/test_parse.py"}'},
        }
    ],
}


def _deepseek(seen: list[dict]):
    """The fake DeepSeek vLLM service: candidates by role tag, else the move."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        prompt = body["messages"][-1]["content"]
        if prompt.startswith("[candidates]"):
            message = {"role": "assistant", "content": json.dumps(CANDIDATES)}
        elif body.get("tools"):
            message = MOVE
        else:
            message = {"role": "assistant", "content": "Paris"}
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-fake",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 16, "completion_tokens": 4, "total_tokens": 20},
            },
        )

    return handler


def _quyet(reads: list[tuple[str, dict]], name: str, *, route: str, down: bool, form):
    """One fake Quyet adapter; ``reads`` records which replica read what.

    ``form(n)`` is the probability every form question reads on the n-th form
    check (0-based); judgments read 0.7.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if down:
            raise httpx.ConnectError("Quyet is down", request=request)
        body = json.loads(request.content)
        if len(body["questions"]) > QUYET_MAX_QUESTIONS:
            return httpx.Response(
                400, json={"detail": f"at most {QUYET_MAX_QUESTIONS} questions per request"}
            )
        reads.append((name, body))
        if "route" in body["questions"]:
            labels = list(body["questions"]["route"]["criteria"])
            answers = {
                "route": {
                    "type": "choice",
                    "probabilities": {label: 0.9 if label == route else 0.1 for label in labels},
                }
            }
        else:
            checks = sum(1 for _, seen in reads if "reply" in seen["state"]) - 1
            p = form(checks) if "reply" in body["state"] else 0.7
            answers = {key: {"type": "noul", "noul": p} for key in body["questions"]}
        return httpx.Response(
            200, json={"answers": answers, "usage": {"input_tokens": 50, "output_tokens": 0}}
        )

    return handler


def _orchestrator(
    seen: list[dict], reads: list, *, route: str = "TOOL", down=frozenset(), form=lambda n: 0.9
):
    deployment = load_deployment_spec(
        (EXAMPLE / "kairyu.yaml").read_text(), resolve_credentials=False
    )
    engines = {
        name: OpenAICompatBackend(
            **pool.replicas[0].options, transport=httpx.MockTransport(_deepseek(seen))
        )
        for name, pool in deployment.pools.items()
    }
    judges = {
        name: HTTPSystemOneBackend(
            base_url=section.base_url,
            upstream_model=section.upstream_model,
            transport=httpx.MockTransport(
                _quyet(reads, name, route=route, down=name in down, form=form)
            ),
        )
        for name, section in deployment.systemone.items()
    }
    return build_orchestrator(
        load_spec(EXAMPLE / "verified-tool.yaml"), engine_refs=engines, systemone_refs=judges
    )


def _call(content: str) -> OrchestrationRequest:
    """The orchestration call the chat route makes for one user message."""

    chat = ChatCompletionRequest(
        model="kairyu-verified-tool", messages=[{"role": "user", "content": content}]
    )
    validated = validate_orchestration_chat_input(chat)
    return OrchestrationRequest(
        prompt=validated.prompt,
        sampling_params=SamplingParams(max_tokens=4096),
        conversation=validated.conversation_messages,
    )


def _agent_call() -> OrchestrationRequest:
    """The orchestration call for an agent turn with the caller's bash tool."""

    chat = ChatCompletionRequest(model="kairyu-verified-tool", messages=AGENT_TURN, tools=[BASH])
    validated = validate_orchestration_chat_input(chat)
    return OrchestrationRequest(
        prompt=validated.prompt,
        sampling_params=SamplingParams(max_tokens=4096),
        tools=(BASH,),
        conversation=validated.conversation_messages,
    )


def test_gateway_builds_from_the_example_configs(tmp_path: Path) -> None:
    raw = (
        (EXAMPLE / "kairyu.yaml")
        .read_text()
        .replace("/etc/kairyu/verified-tool.yaml", str(EXAMPLE / "verified-tool.yaml"))
        .replace("/var/lib/kairyu/placement", str(tmp_path))
    )
    build_app_from_spec(load_deployment_spec(raw, resolve_credentials=False), EXAMPLE)


def test_compose_gpus_match_the_allocation() -> None:
    spec = json.loads((EXAMPLE / "example.json").read_text())
    services = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())["services"]

    def gpus(service: str) -> list[int]:
        devices = services[service]["deploy"]["resources"]["reservations"]["devices"]
        return [int(index) for index in devices[0]["device_ids"]]

    assert gpus("deepseek") == spec["allocation"]["deepseek"]["gpu_ids"]
    for replica in spec["allocation"]["quyet"]["replicas"]:
        assert gpus(replica["vllm_service"]) == [replica["gpu_id"]]
        adapter = services[replica["service"]]["environment"]
        assert adapter["QUYET_UPSTREAM"] == f"http://{replica['vllm_service']}:8000"


@pytest.mark.parametrize("effort", [None, "low", "max"])
async def test_think_is_one_deepseek_answer_at_the_callers_effort(effort) -> None:
    seen: list[dict] = []
    reads: list[dict] = []
    orchestrator = _orchestrator(seen, reads, route="THINK")
    call = _call("Name the capital of France in one word.")
    if effort is not None:
        call = dataclasses.replace(call, reasoning_effort=effort)

    call = await orchestrator.judge_role_profile(call)
    result = await orchestrator.run(call)

    ((judge, read),) = reads
    assert judge == "quyet-route-systemone"
    assert set(read["questions"]["route"]["criteria"]) == {"THINK", "TOOL"}
    assert read["state"]["conversation"][-1]["role"] == "user"
    assert read["state"]["tool_calling"] is False
    assert result.text == "Paris"
    (body,) = seen
    assert body["reasoning_effort"] == (effort or "high")
    # The caller's conversation reaches DeepSeek as it was sent.
    assert body["messages"] == [
        {"role": "user", "content": "Name the capital of France in one word."}
    ]


@pytest.mark.parametrize("effort", [None, "low", "max"])
async def test_tool_route_judges_candidates_then_checks_the_replys_form(effort) -> None:
    seen: list[dict] = []
    reads: list[dict] = []
    orchestrator = _orchestrator(seen, reads, route="TOOL")
    call = _agent_call()
    if effort is not None:
        call = dataclasses.replace(call, reasoning_effort=effort)

    call = await orchestrator.judge_role_profile(call)
    result = await orchestrator.run(call)

    # The published move carries its structured call.
    assert '<tool_call>{"name":"bash"' in result.text
    # Stage reports are not replayed through reasoning_content.
    assert not result.reasoning_content
    # The candidates: one DeepSeek call at the caller's effort that reads the
    # caller's tools.
    candidates, answer = seen
    prompt = candidates["messages"][-1]["content"]
    assert prompt.startswith("[candidates]") and candidates["model"] == "deepseek-v4.1-flash"
    assert candidates["reasoning_effort"] == (effort or "high")
    assert '"name": "bash"' in prompt.split("--- TOOLS ---")[1]
    # The route on its own replica (told the caller declared tools); six
    # questions per candidate on the judge replica, at most 32 per read, each
    # quoting its candidate's call, over the conversation and the tools; then
    # seven form questions over the reply, the tools and the conversation.
    (route_judge, route), *judged, (form_judge, form) = reads
    assert route_judge == "quyet-route-systemone" and route["state"]["tool_calling"] is True
    assert {name for name, _ in judged} | {form_judge} == {"quyet-judge-systemone"}
    questions = [q for _, read in judged for q in read["questions"].values()]
    assert len(questions) == 6 * len(CANDIDATES["candidates"])
    assert all(len(read["questions"]) <= QUYET_MAX_QUESTIONS for _, read in judged)
    assert questions[0]["instructions"]["candidate_arguments"] == '{"command": "step-1"}'
    for _, read in judged:
        assert list(read["state"]) == ["conversation", "tools"]
        assert read["state"]["conversation"][-1]["content"] == "1 failed: test_parse"
        assert read["state"]["tools"] == [BASH]
    assert list(form["state"])[:3] == ["reply", "tools", "conversation"]
    assert '<tool_call>{"name":"bash"' in form["state"]["reply"]
    assert len(form["questions"]) == 7
    # The answer at the caller's effort gets the caller's turn as native
    # messages, then one message with the candidates and every judgment, and
    # publishes with the caller's tools.
    assert answer["reasoning_effort"] == (effort or "high")
    assert answer["tools"][0]["function"]["name"] == "bash"
    assert answer["messages"][:-1] == AGENT_TURN
    material = answer["messages"][-1]["content"]
    assert answer["messages"][-1]["role"] == "user" and material.startswith("Before replying")
    assert '"viewpoint": "view 10"' in material
    assert "- [C1-now] p=0.70" in material and "- [C10-safe] p=0.70" in material


@pytest.mark.parametrize(
    ("form", "answers"),
    [(lambda n: 0.2 if n == 0 else 0.9, 2), (lambda n: 0.2, 3)],
    ids=["fixed", "exhausted"],
)
async def test_a_reply_failing_the_form_check_is_fixed_and_checked_again(form, answers) -> None:
    seen: list[dict] = []
    reads: list[dict] = []
    orchestrator = _orchestrator(seen, reads, route="TOOL", form=form)

    result = await orchestrator.run(await orchestrator.judge_role_profile(_agent_call()))

    # One fix and a passing check, or two fixes and the last reply published.
    _, *replies = seen
    assert len(replies) == answers
    assert sum(1 for _, read in reads if "reply" in read["state"]) == answers
    for fix in replies[1:]:
        # The fix is DeepSeek again, with the caller's turn, tools and effort,
        # the previous reply (its call included) and the unmet requirements.
        assert fix["messages"][:-1] == AGENT_TURN
        assert fix["tools"] == replies[0]["tools"]
        assert fix["reasoning_effort"] == "high"
        material = fix["messages"][-1]["content"]
        draft = material.split("--- DRAFT ---")[1].split("--- END DRAFT ---")[0]
        assert '<tool_call>{"name":"bash"' in draft
        unmet = material.split("--- UNMET REQUIREMENTS ---")[1]
        assert "- [F1] p=0.20" in unmet and "- [F7] p=0.20" in unmet
    assert '<tool_call>{"name":"bash"' in result.text


async def test_an_unavailable_route_judge_routes_to_the_think_answer() -> None:
    seen: list[dict] = []
    orchestrator = _orchestrator(seen, [], down={"quyet-route-systemone"})

    call = await orchestrator.judge_role_profile(_call("Name the capital of France."))
    await orchestrator.run(call)

    assert call.role_profile_judgment is None
    (body,) = seen
    assert body["messages"] == [{"role": "user", "content": "Name the capital of France."}]
