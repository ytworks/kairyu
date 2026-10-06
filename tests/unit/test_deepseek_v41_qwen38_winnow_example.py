"""Routed answers: DeepSeek-V4.1 six-GPU + Qwen3.8 + Winnow example (VCO-D18).

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


def _deepseek(seen: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-fake",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Paris"},
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
        labels = list(body["questions"]["route"]["criteria"])
        return httpx.Response(
            200,
            json={
                "answers": {
                    "route": {
                        "type": "choice",
                        "probabilities": {
                            label: 0.9 if label == route else 0.1 for label in labels
                        },
                    }
                },
                "usage": {"input_tokens": 50, "output_tokens": 0},
            },
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


@pytest.mark.parametrize(("route", "served"), [("VERIFIED", "verified"), ("THINK", "think")])
@pytest.mark.parametrize("effort", [None, "low", "max"])
async def test_winnow_routes_to_one_deepseek_answer_at_the_routes_effort(
    route, served, effort
) -> None:
    seen: list[dict] = []
    reads: list[dict] = []
    orchestrator = _orchestrator(seen, reads, route=route)
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
    prompt = body["messages"][-1]["content"]
    if served == "verified":
        # The verified route is one DeepSeek call at max, whatever the caller asked.
        assert body["reasoning_effort"] == "max"
        assert not prompt.startswith("[deepseek_think_answer]")
    else:
        assert body["reasoning_effort"] == (effort or "high")
        assert prompt.startswith("[deepseek_think_answer]")


async def test_an_unavailable_winnow_routes_to_the_think_answer() -> None:
    seen: list[dict] = []
    orchestrator = _orchestrator(seen, [], down=True)

    call = await orchestrator.judge_role_profile(_call("Name the capital of France."))
    await orchestrator.run(call)

    assert call.role_profile_judgment is None
    (body,) = seen
    assert body["messages"][-1]["content"].startswith("[deepseek_think_answer]")
