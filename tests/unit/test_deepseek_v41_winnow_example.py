"""Routed answers: DeepSeek-V4.1 six-GPU + two Winnow replicas example (VCO-D18..D20).

The example's own kairyu.yaml / verified-tool.yaml drive the production loaders, the
real OpenAI backend (against a fake vLLM) and the real System One backend
(against fake Winnow replicas), so the tests observe what the deployed L1 services
would receive and which route answers.
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

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/deepseek-v4.1-winnow-8gpu"


DRAFTS = {
    f"D{k}": {"viewpoint": f"view {k}", "answer": "Paris", "tool_calls": []} for k in range(1, 6)
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
POINTS = {"points": [{"id": "R1", "point": "names the capital"}, {"id": "R2", "point": "one word"}]}


def _deepseek(seen: list[dict]):
    """The fake DeepSeek vLLM service, answering by role tag."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        prompt = body["messages"][-1]["content"]
        text = (
            json.dumps(DRAFTS)
            if prompt.startswith("[drafts]")
            else json.dumps(POINTS)
            if prompt.startswith("[requirements]")
            else "Paris"
        )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-fake",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 16, "completion_tokens": 4, "total_tokens": 20},
            },
        )

    return handler


def _winnow(reads: list[tuple[str, dict]], name: str, *, route: str, down: bool):
    """One fake Winnow replica; ``reads`` records which replica read what."""

    def handler(request: httpx.Request) -> httpx.Response:
        if down:
            raise httpx.ConnectError("Winnow is down", request=request)
        body = json.loads(request.content)
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
            answers = {key: {"type": "noul", "noul": 0.7} for key in body["questions"]}
        return httpx.Response(
            200, json={"answers": answers, "usage": {"input_tokens": 50, "output_tokens": 0}}
        )

    return handler


def _orchestrator(seen: list[dict], reads: list, *, route: str = "TOOL", down=frozenset()):
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
            transport=httpx.MockTransport(_winnow(reads, name, route=route, down=name in down)),
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
    return OrchestrationRequest(
        prompt=validate_orchestration_chat_input(chat).prompt,
        sampling_params=SamplingParams(max_tokens=4096),
    )


def _agent_call() -> OrchestrationRequest:
    """The orchestration call for an agent turn with the caller's bash tool."""

    chat = ChatCompletionRequest(model="kairyu-verified-tool", messages=AGENT_TURN, tools=[BASH])
    return OrchestrationRequest(
        prompt=validate_orchestration_chat_input(chat).prompt,
        sampling_params=SamplingParams(max_tokens=4096),
        tools=(BASH,),
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
    for replica in spec["allocation"]["winnow"]["replicas"]:
        assert gpus(replica["service"]) == [replica["gpu_id"]]


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
    assert judge == "winnow-route-systemone"
    assert set(read["questions"]["route"]["criteria"]) == {"THINK", "TOOL"}
    assert read["state"]["conversation"][-1]["role"] == "user"
    assert read["state"]["tool_calling"] is False
    assert result.text == "Paris"
    (body,) = seen
    assert body["reasoning_effort"] == (effort or "high")
    assert body["messages"][-1]["content"].startswith("[deepseek_think_answer]")


@pytest.mark.parametrize("effort", [None, "low", "max"])
async def test_tool_route_runs_three_waves_into_one_critical_answer(effort) -> None:
    seen: list[dict] = []
    reads: list[dict] = []
    orchestrator = _orchestrator(seen, reads, route="TOOL")
    call = _agent_call()
    if effort is not None:
        call = dataclasses.replace(call, reasoning_effort=effort)

    call = await orchestrator.judge_role_profile(call)
    result = await orchestrator.run(call)

    assert result.text == "Paris"
    # Stage reports are not replayed through reasoning_content.
    assert not result.reasoning_content
    # Wave 1: five DeepSeek drafts in one call at the caller's effort, its
    # requirements at max whatever the caller sent; both read the caller's
    # tools.
    drafts, requirements = sorted(seen[:2], key=lambda body: body["messages"][-1]["content"])
    assert drafts["messages"][-1]["content"].startswith("[drafts]")
    assert drafts["reasoning_effort"] == (effort or "high")
    assert requirements["messages"][-1]["content"].startswith("[requirements]")
    assert requirements["reasoning_effort"] == "max"
    assert {drafts["model"], requirements["model"]} == {"deepseek-v4.1-flash"}
    for body in (drafts, requirements):
        assert '"name": "bash"' in body["messages"][-1]["content"].split("--- TOOLS ---")[1]
    # Wave 2: one read on the judgments replica (the route on its own, told
    # that the caller declared tools): 5 adoptions + 5 x 2 points, each on a
    # draft's text and tool calls, with where the work stands (the tool
    # result) and the tools the drafted calls must fit.
    (route_judge, route), (judge, judgment) = reads
    assert (route_judge, judge) == ("winnow-route-systemone", "winnow-judge-systemone")
    assert route["state"]["tool_calling"] is True
    assert judgment["state"]["drafts"] == DRAFTS
    assert judgment["state"]["request"][-1]["role"] == "user"
    assert judgment["state"]["conversation"][-1]["content"] == "1 failed: test_parse"
    assert judgment["state"]["tools"] == [BASH]
    assert len(judgment["questions"]) == 5 + 5 * 2
    assert all(
        "tool_calls" in json.dumps(q["instructions"]) for q in judgment["questions"].values()
    )
    # Wave 3: the answer at the caller's effort reads the drafts, requirements
    # and judgments, and publishes with the caller's tools.
    answer = seen[2]
    assert len(seen) == 3 and answer["reasoning_effort"] == (effort or "high")
    assert answer["tools"][0]["function"]["name"] == "bash"
    prompt = answer["messages"][-1]["content"]
    assert '"viewpoint": "view 5"' in prompt and '"names the capital"' in prompt
    assert "- [D1] p=0.70" in prompt and "- [D5-R2] p=0.70 one word" in prompt


async def test_an_unavailable_route_judge_routes_to_the_think_answer() -> None:
    seen: list[dict] = []
    orchestrator = _orchestrator(seen, [], down={"winnow-route-systemone"})

    call = await orchestrator.judge_role_profile(_call("Name the capital of France."))
    await orchestrator.run(call)

    assert call.role_profile_judgment is None
    (body,) = seen
    assert body["messages"][-1]["content"].startswith("[deepseek_think_answer]")
