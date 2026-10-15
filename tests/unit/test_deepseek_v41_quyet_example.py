"""DeepSeek-V4.1 answers decided by Quyet: six-GPU DeepSeek + two Quyet replicas (VCO-D27).

The example's own kairyu.yaml / verified-tool.yaml drive the production loaders, the
real OpenAI backend (against a fake vLLM) and the real System One backend
(against fake Quyet adapters that, like the real one, refuse more than 32
questions per request, any question whose instructions are not one non-empty
string, and, under the strict name, a state too long to read whole), so the tests
observe what the deployed L1 services would receive and what is published.
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


def _deepseek(seen: list[dict], move: dict = MOVE):
    """The fake DeepSeek vLLM service: the move when tools are declared."""

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


def _quyet(reads: list[tuple[str, dict]], name: str, *, decision):
    """One fake Quyet adapter; ``reads`` records which replica read what.

    ``decision(n)`` is the probability every question reads on the n-th
    decision (0-based).
    """

    def handler(request: httpx.Request) -> httpx.Response:
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
        p = decision(len(reads) - 1)
        answers = {key: {"type": "noul", "noul": p} for key in body["questions"]}
        return httpx.Response(
            200, json={"answers": answers, "usage": {"input_tokens": 50, "output_tokens": 0}}
        )

    return handler


def _orchestrator(
    seen: list[dict],
    reads: list,
    *,
    decision=lambda n: 0.9,
    move: dict = MOVE,
):
    deployment = load_deployment_spec(
        (EXAMPLE / "kairyu.yaml").read_text(), resolve_credentials=False
    )
    engines = {
        name: OpenAICompatBackend(
            **pool.replicas[0].options, transport=httpx.MockTransport(_deepseek(seen, move))
        )
        for name, pool in deployment.pools.items()
    }
    judges = {
        name: HTTPSystemOneBackend(
            base_url=section.base_url,
            upstream_model=section.upstream_model,
            transport=httpx.MockTransport(_quyet(reads, name, decision=decision)),
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


@pytest.mark.parametrize(("kind", "effort"), [("agent", None), ("agent", "max"), ("text", "low")])
async def test_a_passing_answer_is_deepseeks_own_published_as_written(kind, effort) -> None:
    seen: list[dict] = []
    reads: list = []
    orchestrator = _orchestrator(seen, reads)
    call = _agent_call() if kind == "agent" else _call("Name the capital of France in one word.")
    if effort is not None:
        call = dataclasses.replace(call, reasoning_effort=effort)

    result = await orchestrator.run(call)

    # One DeepSeek call at the caller's effort: the caller's conversation as it
    # was sent, nothing added, with the caller's tools.
    (answer,) = seen
    assert answer["model"] == "deepseek-v4.1-flash"
    assert answer["reasoning_effort"] == (effort or "high")
    # Then one strict read of the five questions over the answer, the tools
    # and the conversation; all pass, so the answer is published as written.
    ((judge, decision),) = reads
    assert judge == "quyet-form-systemone" and decision["model"] == STRICT_MODEL
    assert len(decision["questions"]) == 5
    assert list(decision["state"])[:3] == ["reply", "tools", "conversation"]
    assert result.verification is not None and result.verification.guaranteed is True
    assert result.verification.attempts == 1
    # Stage reports are not replayed through reasoning_content.
    assert not result.reasoning_content
    if kind == "agent":
        assert answer["messages"] == AGENT_TURN
        assert answer["tools"][0]["function"]["name"] == "bash"
        assert '<tool_call>{"name":"bash"' in decision["state"]["reply"]
        assert decision["state"]["tools"] == [BASH]
        assert decision["state"]["conversation"][-1]["content"] == "1 failed: test_parse"
        assert '<tool_call>{"name":"bash"' in result.text
    else:
        assert answer["messages"] == [
            {"role": "user", "content": "Name the capital of France in one word."}
        ]
        assert decision["state"]["tools"] == "none"
        assert result.text == "Paris"


@pytest.mark.parametrize(
    ("decision", "guaranteed"),
    [(lambda n: 0.2 if n == 0 else 0.9, True), (lambda n: 0.2, False)],
    ids=["fixed", "still-failing"],
)
async def test_a_failing_answer_is_fixed_once_with_the_decision_shown(decision, guaranteed) -> None:
    seen: list[dict] = []
    reads: list = []
    orchestrator = _orchestrator(seen, reads, decision=decision)

    result = await orchestrator.run(_agent_call())

    # One fix, decided again; the fix is published either way.
    draft, fix = seen
    assert len(reads) == 2
    assert result.verification is not None
    assert result.verification.guaranteed is guaranteed
    assert result.verification.attempts == 2
    # The fix is DeepSeek again with the caller's conversation, tools and
    # effort, then one message: the draft (its call included) and Quyet's
    # unmet questions with their probabilities.
    assert fix["messages"][:-1] == AGENT_TURN
    assert fix["tools"] == draft["tools"]
    assert fix["reasoning_effort"] == "high"
    material = fix["messages"][-1]["content"]
    previous = material.split("--- DRAFT ---")[1].split("--- END DRAFT ---")[0]
    assert '<tool_call>{"name":"bash"' in previous
    findings = material.split("--- JUDGE'S FINDINGS ---")[1]
    for item in ("E1", "E2", "E3", "E4", "E5"):
        assert f"- [{item}] p=0.20" in findings
    assert '<tool_call>{"name":"bash"' in result.text


async def test_a_reply_too_long_to_check_whole_is_published_unverified() -> None:
    """The decision never passes an answer on a cut state (PR #645 review): a
    state over Quyet's limit is refused under the strict name, and the answer,
    call included, is published unverified without a fix (as when quyet-judge
    is down)."""

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
    orchestrator = _orchestrator(seen, reads, move=write)

    result = await orchestrator.run(_agent_call())

    assert result.verification is not None
    assert result.verification.guaranteed is False
    assert result.verification.reason == "judge_unavailable"
    assert "fixture.txt" in result.text and '<tool_call>{"name":"bash"' in result.text
    # One answer, no fix: the refused decision never sends the answer back.
    assert sum(1 for body in seen if body.get("tools")) == 1
