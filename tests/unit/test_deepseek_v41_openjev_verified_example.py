"""Checklist-verified answers: DeepSeek-V4.1 six-GPU + OpenJev x 2 example (VCO-D15).

The example's own kairyu.yaml / verified.yaml drive the production loaders, the
real OpenAI backend (against a fake vLLM) and the real System One backend
(against two fake OpenJev replicas), so the tests observe what the deployed
L1 services would receive and what the caller gets back.
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

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/deepseek-v4.1-openjev-verified-8gpu"

EXPLICIT = [
    {"id": "E1", "point": "names Paris"},
    {"id": "E2", "point": "answers in one word"},
]
IMPLICIT = [{"id": "I1", "point": "the answer is a proper noun"}]


def _text(body: dict) -> str:
    return "\n".join(
        message["content"] if isinstance(message["content"], str) else ""
        for message in body["messages"]
    )


def _deepseek(
    seen: list[dict],
    *,
    draft: str,
    explicit: list[dict] | None = None,
    implicit: list[dict] | None = None,
    repair: str = "Paris",
    draft_finish: str = "stop",
):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        text = _text(body)
        if text.startswith("[extract]"):
            answer = json.dumps({"points": EXPLICIT if explicit is None else explicit})
        elif text.startswith("[implicit]"):
            answer = json.dumps({"points": IMPLICIT if implicit is None else implicit})
        elif text.startswith("[history]"):
            answer = "none"
        elif text.startswith("[repair]"):
            answer = repair
        else:
            answer = draft
        finish = draft_finish if answer == draft else "stop"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-fake",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": finish,
                    }
                ],
                "usage": {"prompt_tokens": 16, "completion_tokens": 4, "total_tokens": 20},
            },
        )

    return handler


def _openjev(
    reads: list[dict],
    *,
    route: str = "VERIFIED",
    down: bool = False,
    unneeded: str | None = None,
    covered_by: dict[str, str] | None = None,
):
    """Necessity is low only for ``unneeded``; a point in ``covered_by`` is
    contained only in an answer holding the mapped text."""

    def answer(question: dict, state: dict) -> dict:
        text = json.dumps(question)
        if question["type"] == "choice":
            other = next(label for label in question["criteria"] if label != route)
            return {"type": "choice", "probabilities": {route: 0.9, other: 0.1}}
        if "adopted as the reply the user expects" in text:
            return {"noul": 0.9999 if all(i["passed"] for i in state["checklist"]) else 0.0}
        if "Must the reply the assistant gives now meet this point" in text:
            return {"noul": 0.1 if unneeded is not None and unneeded in text else 0.9999}
        for point, needed in (covered_by or {}).items():
            if point in text:
                return {"noul": 0.9999 if needed in state["answer"] else 0.0}
        return {"noul": 0.9999}

    def handler(request: httpx.Request) -> httpx.Response:
        if down:
            raise httpx.ConnectError("both OpenJev replicas are down", request=request)
        body = json.loads(request.content)
        reads.append({"replica": request.url.host, **body})
        answers = {
            key: answer(question, body["state"]) for key, question in body["questions"].items()
        }
        return httpx.Response(
            200, json={"answers": answers, "usage": {"input_tokens": 50, "output_tokens": 0}}
        )

    return handler


def _orchestrator(
    seen: list[dict],
    reads: list[dict],
    *,
    draft: str,
    spec: str = "verified-always.yaml",
    route: str = "VERIFIED",
    explicit: list[dict] | None = None,
    implicit: list[dict] | None = None,
    repair: str = "Paris",
    jev_down: bool = False,
    unneeded: str | None = None,
    covered_by: dict[str, str] | None = None,
    draft_finish: str = "stop",
):
    deployment = load_deployment_spec(
        (EXAMPLE / "kairyu.yaml").read_text(), resolve_credentials=False
    )
    engines = {
        name: OpenAICompatBackend(
            **pool.replicas[0].options,
            transport=httpx.MockTransport(
                _deepseek(
                    seen,
                    draft=draft,
                    explicit=explicit,
                    implicit=implicit,
                    repair=repair,
                    draft_finish=draft_finish,
                )
            ),
        )
        for name, pool in deployment.pools.items()
    }
    judges = {
        name: HTTPSystemOneBackend(
            base_urls=section.base_urls,
            upstream_model=section.upstream_model,
            transport=httpx.MockTransport(
                _openjev(
                    reads,
                    route=route,
                    down=jev_down,
                    unneeded=unneeded,
                    covered_by=covered_by,
                )
            ),
        )
        for name, section in deployment.systemone.items()
    }
    return build_orchestrator(load_spec(EXAMPLE / spec), engine_refs=engines, systemone_refs=judges)


def _call(content: str, **sampling) -> OrchestrationRequest:
    """The orchestration call the chat route makes for one user message."""

    params = SamplingParams(max_tokens=4096, **sampling)
    chat = ChatCompletionRequest(
        model="kairyu-verified", messages=[{"role": "user", "content": content}]
    )
    return OrchestrationRequest(
        prompt=validate_orchestration_chat_input(chat).prompt,
        sampling_params=params,
        response_format=params.extra_args.get("response_format"),
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


async def test_a_draft_covering_every_adopted_point_is_published_with_a_guarantee() -> None:
    seen: list[dict] = []
    reads: list[dict] = []
    orchestrator = _orchestrator(seen, reads, draft="Paris")

    result = await orchestrator.run(_call("Name the capital of France in one word."))

    assert result.text == "Paris"
    report = result.verification.as_dict()
    assert report["guaranteed"] is True
    assert [(item["id"], item["tags"]["origin"]) for item in report["requirements"]] == [
        ("E1", "explicit"),
        ("E2", "explicit"),
        ("I1", "implicit"),
    ]
    by_role = {_text(body).split("]", 1)[0].lstrip("["): body for body in seen}
    # The draft is published as-is (no repair call).
    assert "repair" not in by_role
    # The extractors analyse the user's request, not Kairyu's answer contract.
    extract_text = _text(by_role["extract"])
    assert "Name the capital of France in one word." in extract_text
    assert "Return only the assistant response body" not in extract_text
    assert by_role["extract"]["response_format"]["type"] == "json_schema"
    assert by_role["implicit"]["response_format"]["type"] == "json_schema"
    # Two System One requests, one per stage, on the OpenJev replicas.
    adopt, coverage, acceptance = reads
    assert {read["replica"] for read in reads} <= {"openjev-0", "openjev-1"}
    # Adoption: every point of both lists, judged against the request
    # verbatim (system/developer + latest user) and the history summary.
    assert len(adopt["questions"]) == 3
    assert adopt["state"]["request"] == [
        {"role": "user", "content": "Name the capital of France in one word."}
    ]
    assert adopt["state"]["history"] == "none"
    # Without tools the reply needed now is the answer to the request.
    assert adopt["state"]["tools"] == "none"
    # Coverage: every adopted point against the answer as it will be sent,
    # in the context of the request and the history summary.
    assert coverage["state"] == {
        "request": adopt["state"]["request"],
        "history": "none",
        "answer": "Paris",
    }
    assert len(coverage["questions"]) == 3
    # Acceptance: the point results, the answer and the original prompt.
    assert set(acceptance["state"]) == {"prompt", "answer", "checklist"}
    assert acceptance["state"]["prompt"] == adopt["state"]["request"]


async def test_a_missing_point_is_repaired_then_guaranteed() -> None:
    seen: list[dict] = []
    reads: list[dict] = []
    orchestrator = _orchestrator(
        seen,
        reads,
        draft="It is the city on the Seine.",
        covered_by={"names Paris": "Paris"},
    )

    result = await orchestrator.run(_call("Name the capital of France in one word."))

    assert result.text == "Paris"
    assert result.verification.guaranteed and result.verification.attempts == 2
    repair = next(_text(body) for body in seen if _text(body).startswith("[repair]"))
    assert "[E1] names Paris" in repair and "[E2]" not in repair
    assert "It is the city on the Seine." in repair
    assert "--- ORIGINAL PROMPT ---" in repair


async def test_a_point_the_request_does_not_need_is_never_judged() -> None:
    # Both lists are adopted in the same request; an unnecessary point,
    # explicit or implicit, leaves its list before the answer is judged.
    unneeded = "the answer is a proper noun"
    seen: list[dict] = []
    reads: list[dict] = []
    orchestrator = _orchestrator(seen, reads, draft="Paris", unneeded=unneeded)

    result = await orchestrator.run(_call("Name the capital of France in one word."))

    judged = {item.proposition for item in result.verification.items}
    assert unneeded not in judged and len(judged) == 2
    assert unneeded not in json.dumps(reads[-1]["questions"])
    # The generator writes after adoption, to meet exactly the kept points.
    generator = _text(next(body for body in seen if _text(body).startswith("Kairyu L2")))
    assert "names Paris" in generator and unneeded not in generator
    assert result.verification.guaranteed


async def test_the_callers_response_format_constrains_draft_and_extraction() -> None:
    seen: list[dict] = []
    reads: list[dict] = []
    orchestrator = _orchestrator(seen, reads, draft="Paris")
    schema = {"type": "json_schema", "json_schema": {"name": "a", "schema": {"type": "string"}}}

    await orchestrator.run(
        _call("Name the capital of France in one word.", extra_args={"response_format": schema})
    )

    generator = next(body for body in seen if _text(body).startswith("Kairyu L2"))
    assert generator["response_format"] == schema
    # The extractor reads the request within the caller's format, so it never
    # demands content the format cannot hold.
    extract = next(_text(body) for body in seen if _text(body).startswith("[extract]"))
    assert json.dumps(schema) in extract


def test_compose_gpus_match_the_allocation() -> None:
    spec = json.loads((EXAMPLE / "example.json").read_text())
    services = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())["services"]

    def gpus(service: str) -> list[int]:
        devices = services[service]["deploy"]["resources"]["reservations"]["devices"]
        return [int(index) for index in devices[0]["device_ids"]]

    assert gpus("deepseek") == spec["allocation"]["deepseek"]["gpu_ids"]
    assert [*gpus("openjev-0"), *gpus("openjev-1")] == spec["allocation"]["openjev"]["gpu_ids"]


def test_both_models_share_one_verified_dag() -> None:
    routed = yaml.safe_load((EXAMPLE / "verified.yaml").read_text())
    always = yaml.safe_load((EXAMPLE / "verified-always.yaml").read_text())
    assert routed["roles"] == always["roles"]
    assert "profile_judge" not in always and not always.get("profiles")


@pytest.mark.parametrize(("route", "served"), [("VERIFIED", "verified"), ("THINK", "think")])
@pytest.mark.parametrize("effort", [None, "low", "max"])
async def test_jev_routes_and_every_deepseek_call_thinks_at_the_callers_effort(
    route, served, effort
) -> None:
    seen: list[dict] = []
    reads: list[dict] = []
    orchestrator = _orchestrator(seen, reads, draft="Paris", spec="verified.yaml", route=route)
    call = _call("Name the capital of France in one word.")
    if effort is not None:
        call = dataclasses.replace(call, reasoning_effort=effort)

    call = await orchestrator.judge_role_profile(call)
    result = await orchestrator.run(call)

    route_read = next(read for read in reads if "route" in read["questions"])
    assert route_read["state"]["conversation"][-1]["role"] == "user"
    if served == "verified":
        assert result.verification is not None and result.verification.guaranteed
    else:
        assert result.verification is None
        assert [_text(body).split("]")[0] for body in seen] == ["[deepseek_think_answer"]
    # One effort for every thinking DeepSeek step, default high (VCO-D9);
    # the history summary never thinks.
    thinking = [body for body in seen if not _text(body).startswith("[history]")]
    assert {body.get("reasoning_effort") for body in thinking} == {effort or "high"}
    history = [body for body in seen if _text(body).startswith("[history]")]
    assert all(body.get("reasoning_effort") is None for body in history)


async def test_an_unavailable_jev_routes_to_the_think_answer() -> None:
    seen: list[dict] = []
    orchestrator = _orchestrator(seen, [], draft="Paris", spec="verified.yaml", jev_down=True)

    call = await orchestrator.judge_role_profile(_call("Name the capital of France."))
    await orchestrator.run(call)

    assert call.role_profile_judgment is None
    assert [_text(body).split("]")[0] for body in seen] == ["[deepseek_think_answer"]


async def test_the_longest_path_fits_the_step_budget() -> None:
    # Two failed repairs are the longest path; it must end refinement_limit,
    # never unverified for lack of steps (reason: budget).
    orchestrator = _orchestrator(
        [], [], draft="Lyon", repair="Lyon", covered_by={"names Paris": "Paris"}
    )

    result = await orchestrator.run(_call("Name the capital of France in one word."))

    assert result.verification.attempts == 3
    assert result.verification.reason == "refinement_limit"


async def test_n_greater_than_one_is_refused_on_the_real_dag() -> None:
    # Review P2 (round 2): the final unit publishes one seeded draft.
    orchestrator = _orchestrator([], [], draft="Paris")

    with pytest.raises(ValueError, match="n > 1"):
        await orchestrator.run(_call("Name the capital of France.", n=2))


async def test_an_unverified_draft_keeps_its_finish_reason() -> None:
    # Review P2 (round 2): the judge is down; the published draft was cut.
    orchestrator = _orchestrator([], [], draft="Paris", jev_down=True, draft_finish="length")

    result = await orchestrator.run(_call("Name the capital of France."))

    assert result.verification.reason == "judge_unavailable"
    assert result.completions[0].finish_reason == "length"
