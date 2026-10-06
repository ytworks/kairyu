"""Routed answers: DeepSeek-V4.1 six-GPU + Qwen3.8 + Winnow example (VCO-D18, D19).

The example's own kairyu.yaml / verified.yaml drive the production loaders, the
real OpenAI backend (against a fake vLLM) and the real System One backend
(against a fake Winnow), so the tests observe what the deployed L1 services
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

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/deepseek-v4.1-qwen3.8-winnow-8gpu"


DRAFTS = {f"D{k}": {"viewpoint": f"view {k}", "answer": "Paris"} for k in range(1, 6)}
POINTS = {"points": [{"id": "R1", "point": "names the capital"}, {"id": "R2", "point": "one word"}]}


def _deepseek(seen: list[dict]):
    """The fake vLLM services: DeepSeek and Qwen, answering by role tag."""

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


def _winnow(reads: list[dict], *, route: str, down: bool):
    def handler(request: httpx.Request) -> httpx.Response:
        if down:
            raise httpx.ConnectError("Winnow is down", request=request)
        body = json.loads(request.content)
        reads.append(body)
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


def _orchestrator(seen: list[dict], reads: list[dict], *, route: str = "VERIFIED", down=False):
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
            transport=httpx.MockTransport(_winnow(reads, route=route, down=down)),
        )
        for name, section in deployment.systemone.items()
    }
    return build_orchestrator(
        load_spec(EXAMPLE / "verified.yaml"), engine_refs=engines, systemone_refs=judges
    )


def _call(content: str) -> OrchestrationRequest:
    """The orchestration call the chat route makes for one user message."""

    chat = ChatCompletionRequest(
        model="kairyu-verified", messages=[{"role": "user", "content": content}]
    )
    return OrchestrationRequest(
        prompt=validate_orchestration_chat_input(chat).prompt,
        sampling_params=SamplingParams(max_tokens=4096),
    )


def test_gateway_builds_from_the_example_configs(tmp_path: Path) -> None:
    raw = (
        (EXAMPLE / "kairyu.yaml")
        .read_text()
        .replace("/etc/kairyu/verified.yaml", str(EXAMPLE / "verified.yaml"))
        .replace("/etc/kairyu/verified-always.yaml", str(EXAMPLE / "verified-always.yaml"))
        .replace("/var/lib/kairyu/placement", str(tmp_path))
    )
    build_app_from_spec(load_deployment_spec(raw, resolve_credentials=False), EXAMPLE)


def test_compose_gpus_match_the_allocation() -> None:
    spec = json.loads((EXAMPLE / "example.json").read_text())
    services = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())["services"]

    def gpus(service: str) -> list[int]:
        devices = services[service]["deploy"]["resources"]["reservations"]["devices"]
        return [int(index) for index in devices[0]["device_ids"]]

    for service in ("deepseek", "qwen", "winnow"):
        assert gpus(service) == spec["allocation"][service]["gpu_ids"]


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

    (read,) = reads
    assert set(read["questions"]["route"]["criteria"]) == {"THINK", "VERIFIED"}
    assert read["state"]["conversation"][-1]["role"] == "user"
    assert result.text == "Paris"
    (body,) = seen
    assert body["reasoning_effort"] == (effort or "high")
    assert body["messages"][-1]["content"].startswith("[deepseek_think_answer]")


@pytest.mark.parametrize("effort", [None, "low", "max"])
async def test_verified_runs_three_waves_into_one_critical_answer(effort) -> None:
    seen: list[dict] = []
    reads: list[dict] = []
    orchestrator = _orchestrator(seen, reads, route="VERIFIED")
    call = _call("Name the capital of France in one word.")
    if effort is not None:
        call = dataclasses.replace(call, reasoning_effort=effort)

    call = await orchestrator.judge_role_profile(call)
    result = await orchestrator.run(call)

    assert result.text == "Paris"
    # Wave 1: five DeepSeek drafts in one call at the caller's effort, Qwen's
    # requirements at low whatever the caller sent.
    drafts, requirements = sorted(seen[:2], key=lambda body: body["model"])
    assert drafts["model"] == "deepseek-v4.1-flash"
    assert drafts["reasoning_effort"] == (effort or "high")
    assert drafts["messages"][-1]["content"].startswith("[drafts]")
    assert requirements["model"] == "qwen3.8-27b" and requirements["reasoning_effort"] == "low"
    assert requirements["messages"][-1]["content"].startswith("[requirements]")
    # Wave 2: one Winnow read, with the request: 5 adoptions + 5 x 2 points.
    _route, judgment = reads
    assert judgment["state"]["drafts"] == DRAFTS
    assert judgment["state"]["request"][-1]["role"] == "user"
    assert len(judgment["questions"]) == 5 + 5 * 2
    # Wave 3: the answer at the caller's effort reads the drafts, requirements
    # and judgments.
    answer = seen[2]
    assert len(seen) == 3 and answer["reasoning_effort"] == (effort or "high")
    prompt = answer["messages"][-1]["content"]
    assert '"viewpoint": "view 5"' in prompt and '"names the capital"' in prompt
    assert "- [D1] p=0.70" in prompt and "- [D5-R2] p=0.70 one word" in prompt


async def test_an_unavailable_winnow_routes_to_the_think_answer() -> None:
    seen: list[dict] = []
    orchestrator = _orchestrator(seen, [], down=True)

    call = await orchestrator.judge_role_profile(_call("Name the capital of France."))
    await orchestrator.run(call)

    assert call.role_profile_judgment is None
    (body,) = seen
    assert body["messages"][-1]["content"].startswith("[deepseek_think_answer]")
