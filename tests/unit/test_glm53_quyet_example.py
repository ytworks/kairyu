"""Routed answers: GLM-5.3-Flash six-GPU + two Quyet replicas example (VCO-D23, VCO-D24).

The example's own kairyu.yaml / verified-tool.yaml drive the production loaders, the
real OpenAI backend (against a fake vLLM) and the real System One backend
(against fake Quyet adapters that, like the real one, refuse more than 32
questions per request, any question whose instructions are not one non-empty
string, and, under the strict name, a state too long to read whole), so the tests
observe what the deployed L1 services would receive and which route answers.
"""

from __future__ import annotations

import dataclasses
import importlib.util
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

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/glm-5.3-flash-quyet-8gpu"
# verified-tool.yaml's default_reasoning_effort: GLM's official default.
DEFAULT_EFFORT = "max"

# The real adapter's QUYET_MAX_QUESTIONS (example.json).
QUYET_MAX_QUESTIONS = 32
# The decision's strict name (QUYET_STRICT_ALIASES) and, for the fake, the
# state size Quyet's 6,000 tokens stand for.
STRICT_MODEL = "quyet-strict"
STRICT_STATE_CHARS = 20000
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
# The move the fake GLM makes: short text and one structured call.
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


def _glm(seen: list[dict], move: dict = MOVE):
    """The fake GLM vLLM service: the move when tools are declared."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if body.get("tools"):
            message = move
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


def _quyet(reads: list[tuple[str, dict]], name: str, *, route: str, down: bool, decision):
    """One fake Quyet adapter; ``reads`` records which replica read what.

    ``decision(n)`` is the probability every question reads on the n-th
    decision (0-based).
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if down:
            raise httpx.ConnectError("Quyet is down", request=request)
        body = json.loads(request.content)
        if len(body["questions"]) > QUYET_MAX_QUESTIONS:
            return httpx.Response(
                400, json={"detail": f"at most {QUYET_MAX_QUESTIONS} questions per request"}
            )
        for key, question in body["questions"].items():
            # The quyet package's own question check (quyet.questions).
            instructions = question.get("instructions")
            if not isinstance(instructions, str) or not instructions.strip():
                return httpx.Response(
                    400,
                    json={"detail": f"question {key!r}: instructions must be a non-empty string"},
                )
        if body["model"] == STRICT_MODEL and len(json.dumps(body["state"])) > STRICT_STATE_CHARS:
            return httpx.Response(400, json={"detail": "state over the limit (strict)"})
        reads.append((name, body))
        if "route" in body["questions"]:
            labels = list(body["questions"]["route"]["criteria"])
            answers = {
                "route": {
                    "type": "choice",
                    "probabilities": {label: 0.95 if label == route else 0.05 for label in labels},
                }
            }
        else:
            p = decision(sum(1 for _, seen in reads if "reply" in seen["state"]) - 1)
            answers = {key: {"type": "noul", "noul": p} for key in body["questions"]}
        return httpx.Response(
            200, json={"answers": answers, "usage": {"input_tokens": 50, "output_tokens": 0}}
        )

    return handler


def _orchestrator(
    seen: list[dict],
    reads: list,
    *,
    route: str = "TOOL",
    down=frozenset(),
    decision=lambda n: 0.9,
    move: dict = MOVE,
):
    deployment = load_deployment_spec(
        (EXAMPLE / "kairyu.yaml").read_text(), resolve_credentials=False
    )
    engines = {
        name: OpenAICompatBackend(
            **pool.replicas[0].options, transport=httpx.MockTransport(_glm(seen, move))
        )
        for name, pool in deployment.pools.items()
    }
    judges = {
        name: HTTPSystemOneBackend(
            base_url=section.base_url,
            upstream_model=section.upstream_model,
            transport=httpx.MockTransport(
                _quyet(reads, name, route=route, down=name in down, decision=decision)
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

    assert gpus("glm") == spec["allocation"]["glm"]["gpu_ids"]
    for replica in spec["allocation"]["quyet"]["replicas"]:
        assert gpus(replica["vllm_service"]) == [replica["gpu_id"]]
        adapter = services[replica["service"]]["environment"]
        assert adapter["QUYET_UPSTREAM"] == f"http://{replica['vllm_service']}:8000"


@pytest.mark.parametrize("effort", [None, "low", "max"])
async def test_think_is_one_glm_answer_at_the_callers_effort(effort) -> None:
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
    assert body["reasoning_effort"] == (effort or DEFAULT_EFFORT)
    # GLM always thinks: the effort is its only switch, no template kwargs.
    assert "chat_template_kwargs" not in body
    # The caller's conversation reaches GLM as it was sent.
    assert body["messages"] == [
        {"role": "user", "content": "Name the capital of France in one word."}
    ]


@pytest.mark.parametrize("effort", [None, "low", "max"])
async def test_tool_route_writes_one_reply_then_quyet_decides_it(effort) -> None:
    seen: list[dict] = []
    reads: list[dict] = []
    orchestrator = _orchestrator(seen, reads, route="TOOL")
    call = _agent_call()
    if effort is not None:
        call = dataclasses.replace(call, reasoning_effort=effort)

    call = await orchestrator.judge_role_profile(call)
    result = await orchestrator.run(call)

    # The published move carries its structured call, and its decision passed.
    assert '<tool_call>{"name":"bash"' in result.text
    assert result.verification is not None and result.verification.guaranteed is True
    # Stage reports are not replayed through reasoning_content.
    assert not result.reasoning_content
    # One GLM call at the caller's effort, without template kwargs: the
    # caller's turn as native messages, then one note, published with the
    # caller's tools.
    (answer,) = seen
    assert "chat_template_kwargs" not in answer
    assert answer["model"] == "glm-5.3-flash"
    assert answer["reasoning_effort"] == (effort or DEFAULT_EFFORT)
    assert answer["tools"][0]["function"]["name"] == "bash"
    assert answer["messages"][:-1] == AGENT_TURN
    note = answer["messages"][-1]
    assert note["role"] == "user" and note["content"].startswith("Before replying")
    # The route on its own replica (told the caller declared tools), then one
    # strict read of nine questions over the reply, the tools and the
    # conversation.
    (route_judge, route), (decision_judge, decision) = reads
    assert route_judge == "quyet-route-systemone" and route["state"]["tool_calling"] is True
    assert decision_judge == "quyet-form-systemone" and decision["model"] == STRICT_MODEL
    assert len(decision["questions"]) == 9
    assert list(decision["state"])[:3] == ["reply", "tools", "conversation"]
    assert '<tool_call>{"name":"bash"' in decision["state"]["reply"]
    assert decision["state"]["tools"] == [BASH]
    assert decision["state"]["conversation"][-1]["content"] == "1 failed: test_parse"


@pytest.mark.parametrize(
    ("decision", "guaranteed"),
    [(lambda n: 0.2 if n == 0 else 0.9, True), (lambda n: 0.2, False)],
    ids=["fixed", "still-failing"],
)
async def test_a_failing_reply_is_fixed_once_with_the_decision_shown(decision, guaranteed) -> None:
    seen: list[dict] = []
    reads: list[dict] = []
    orchestrator = _orchestrator(seen, reads, route="TOOL", decision=decision)

    result = await orchestrator.run(await orchestrator.judge_role_profile(_agent_call()))

    # One fix, decided again; the fixed reply is published either way.
    draft, fix = seen
    assert sum(1 for _, read in reads if "reply" in read["state"]) == 2
    assert result.verification is not None
    assert result.verification.guaranteed is guaranteed
    assert result.verification.attempts == 2
    # The fix is GLM again, with the caller's turn, tools and effort, the
    # draft (its call included) and Quyet's unmet requirements with their
    # probabilities.
    assert fix["messages"][:-1] == AGENT_TURN
    assert fix["tools"] == draft["tools"]
    assert fix["reasoning_effort"] == DEFAULT_EFFORT
    material = fix["messages"][-1]["content"]
    previous = material.split("--- DRAFT ---")[1].split("--- END DRAFT ---")[0]
    assert '<tool_call>{"name":"bash"' in previous
    findings = material.split("--- JUDGE'S FINDINGS ---")[1]
    assert "- [R1] p=0.20" in findings and "- [S1] p=0.20" in findings
    assert "- [F5] p=0.20" in findings
    assert '<tool_call>{"name":"bash"' in result.text


async def test_a_reply_too_long_to_check_whole_is_published_unverified() -> None:
    """The decision never passes a reply on a cut state (PR #645 review): a
    state over Quyet's limit is refused under the strict name, and the reply,
    call included, is published unverified without a fix."""

    seen: list[dict] = []
    reads: list[dict] = []
    fixture = "\n".join(f"line {n}: value" for n in range(2000))
    write = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_3",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps(
                        {"command": f"cat > /app/fixture.txt <<'EOF'\n{fixture}\nEOF"}
                    ),
                },
            }
        ],
    }
    orchestrator = _orchestrator(seen, reads, route="TOOL", move=write)

    result = await orchestrator.run(await orchestrator.judge_role_profile(_agent_call()))

    assert result.verification is not None
    assert result.verification.guaranteed is False
    assert result.verification.reason == "judge_unavailable"
    assert "fixture.txt" in result.text and '<tool_call>{"name":"bash"' in result.text
    # One answer, no fix: the refused check never sends the reply back.
    assert sum(1 for body in seen if body.get("tools")) == 1


async def test_an_unavailable_route_judge_routes_to_the_think_answer() -> None:
    seen: list[dict] = []
    orchestrator = _orchestrator(seen, [], down={"quyet-route-systemone"})

    call = await orchestrator.judge_role_profile(_call("Name the capital of France."))
    await orchestrator.run(call)

    assert call.role_profile_judgment is None
    (body,) = seen
    assert body["messages"] == [{"role": "user", "content": "Name the capital of France."}]


def _control():
    module_spec = importlib.util.spec_from_file_location(
        "glm_quyet_control", EXAMPLE / "control.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("glm_present", "quyet_present", "required"),
    [(False, False, 480), (True, False, 80), (True, True, 0)],
)
def test_free_space_counts_only_missing_checkpoints(glm_present, quyet_present, required) -> None:
    # One check before any download (PR #648 review): a fetched GLM must not
    # leave the Quyet download facing the whole example's requirement again.
    control = _control()
    assert (
        control.required_free_gib(glm_present=glm_present, quyet_present=quyet_present) == required
    )


def test_chat_ui_check_follows_the_bind_address() -> None:
    control = _control()
    env = {"CHAT_UI_PORT": "3017"}
    assert control.ui_check_url({**env, "CHAT_UI_BIND_ADDRESS": "0.0.0.0"}) == (
        "http://127.0.0.1:3017"
    )
    assert control.ui_check_url({**env, "CHAT_UI_BIND_ADDRESS": "192.0.2.10"}) == (
        "http://192.0.2.10:3017"
    )


def test_free_space_counts_a_leftover_glm_directory_as_missing(tmp_path, monkeypatch) -> None:
    # A directory left by a failed run stops _seed_model from linking, and the
    # model program then downloads the whole checkpoint (PR #648 review): GLM
    # counts as present only once its config.json is there, seed or not.
    control = _control()
    monkeypatch.setenv("GLM_MODEL_SEED", str(tmp_path / "seed"))
    leftover = tmp_path / "glm-5.3-flash"
    leftover.mkdir()
    assert not control.glm_present(leftover)
    (leftover / "config.json").write_text("{}")
    assert control.glm_present(leftover)
