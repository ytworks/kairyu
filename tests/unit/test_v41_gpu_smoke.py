"""Validate smoke verdicts and request coverage without a live service."""

import importlib.util
import json
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/qwen3.8-deepseek-v4.1-8gpu"


def harness():
    path = EXAMPLE / "gpu_smoke.py"
    assert path.exists(), "standalone smoke harness must exist"
    spec = importlib.util.spec_from_file_location("v41_gpu_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def response(content="323", finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


def test_length_is_never_a_pass_even_with_correct_body():
    result = harness().validate_response("arithmetic", response(finish="length"))
    assert result["passed"] is False and result["truncated"] is True


@pytest.mark.parametrize("body", [{}, {"choices": []}, response(content="wrong")])
def test_malformed_or_wrong_arithmetic_fails(body):
    assert not harness().validate_response("arithmetic", body)["passed"]


def test_checklist_contract_rejects_empty_and_nonconsecutive_ids():
    import json

    module = harness()
    assert not module.validate_response("requirements", response("[]"))["passed"]
    row = dict(
        id="R2",
        priority="minimum",
        requirement="Compare",
        acceptance_criterion="Compare A and B",
        source="Compare A and B",
    )
    assert not module.validate_response("requirements", response(json.dumps([row])))["passed"]
    row["id"] = "R1"
    assert module.validate_response("requirements", response(json.dumps([row])))["passed"]


def test_requirements_cases_use_exact_shipped_prompt_without_client_override():
    import yaml

    module = harness()
    cases = module.build_cases(EXAMPLE, "deepseek-v4.1-flash")
    requirements = [c for c in cases if c["kind"] == "requirements"]
    assert len(requirements) == 12
    prompt = next(
        r["prompt"]
        for r in yaml.safe_load((EXAMPLE / "auto-max.yaml").read_text())["roles"]
        if r["name"] == "requirements"
    ).format(query=module.REQUIREMENTS_QUERY)
    assert {c["payload"].get("reasoning_effort") for c in requirements} == {
        None,
        "low",
        "high",
        "max",
    }
    for case in requirements:
        assert case["payload"]["messages"] == [{"role": "user", "content": prompt}]
        assert "structured_outputs" not in case["payload"]
        assert "thinking_token_budget" not in case["payload"]


def test_tool_arguments_and_stream_completion_are_checked():
    module = harness()
    tool = response(None, "tool_calls")
    tool["choices"][0]["message"]["tool_calls"] = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"key":"probe"}'},
        }
    ]
    assert module.validate_response("tool", tool)["passed"]
    tool["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{}"
    assert not module.validate_response("tool", tool)["passed"]
    stream = (
        'data: {"choices":[{"delta":{"content":"323"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    )
    body, done = module.parse_stream(stream)
    assert done and module.validate_response("arithmetic", body)["passed"]
    _, done = module.parse_stream(stream.replace("data: [DONE]", ""))
    assert not done


def test_execute_case_persists_partial_stream_as_failure(tmp_path, monkeypatch):
    import io
    import json

    module = harness()

    class Response(io.BytesIO):
        status = 200
        headers = {"content-type": "text/event-stream"}

    raw = b'data: {"choices":[{"delta":{"content":"323"},"finish_reason":"stop"}]}\n\n'
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: Response(raw))
    case = {"name": "stream", "kind": "arithmetic", "payload": {"stream": True}}
    result = module.execute_case(
        case, endpoint="http://unused/v1/chat/completions", output=tmp_path, timeout=10
    )
    assert result["passed"] is False
    assert result["stream_done"] is False
    assert (tmp_path / "stream" / "response.raw").read_bytes() == raw
    assert json.loads((tmp_path / "stream" / "result.json").read_text()) == result


def test_execute_case_records_http_error_body(tmp_path, monkeypatch):
    import io

    module = harness()

    def fail(*args, **kwargs):
        raise module.urllib.error.HTTPError(
            "http://unused", 400, "bad request", {}, io.BytesIO(b'{"error":"budget rejected"}')
        )

    monkeypatch.setattr(module.urllib.request, "urlopen", fail)
    result = module.execute_case(
        {"name": "bad", "kind": "requirements", "payload": {}},
        endpoint="http://unused/v1/chat/completions",
        output=tmp_path,
        timeout=10,
    )
    assert result["passed"] is False and result["http_status"] == 400
    assert (tmp_path / "bad" / "response.raw").read_bytes() == b'{"error":"budget rejected"}'


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("top", [False, True])
def test_nonfinite_logprob_rejects_correct_answer(value, top):
    module = harness()
    body = response()
    token = {"token": "323", "logprob": -0.1, "top_logprobs": []}
    if top:
        token["top_logprobs"] = [{"token": "323", "logprob": value}]
    else:
        token["logprob"] = value
    body["choices"][0]["logprobs"] = {"content": [token]}
    result = module.validate_response("arithmetic", body)
    assert not result["passed"]
    assert "nonfinite" in result["detail"]


def test_rank_header_and_logprob_request_are_recorded(tmp_path, monkeypatch):
    import io
    import json

    module = harness()
    sent = []

    class Response(io.BytesIO):
        status = 200
        headers = {"content-type": "application/json"}

    def send(request, **kwargs):
        sent.append(request)
        return Response(json.dumps(response()).encode())

    monkeypatch.setattr(module.urllib.request, "urlopen", send)
    case = {
        "name": "rank",
        "kind": "arithmetic",
        "payload": {"messages": [{"role": "user", "content": "test"}]},
    }
    result = module.execute_case(
        case,
        endpoint="http://unused/v1/chat/completions",
        output=tmp_path,
        timeout=10,
        data_parallel_rank=2,
        logprobs=True,
    )
    headers = {key.lower(): value for key, value in sent[0].header_items()}
    assert headers["x-data-parallel-rank"] == "2"
    body = json.loads(sent[0].data)
    assert body["logprobs"] is True and body["top_logprobs"] == 2
    assert body["messages"] == case["payload"]["messages"]
    assert result["data_parallel_rank"] == 2
    assert json.loads((tmp_path / "rank" / "request.json").read_text()) == body
    recorded = json.loads((tmp_path / "rank" / "request-headers.json").read_text())
    assert recorded["X-data-parallel-rank"] == "2"


def test_stream_rejects_nonfinite_logprob():
    module = harness()
    stream = (
        'data: {"choices":[{"delta":{"content":"323"},"finish_reason":"stop",'
        '"logprobs":{"content":[{"token":"323","logprob":NaN}]}}]}\n\n'
        "data: [DONE]\n\n"
    )
    with pytest.raises(ValueError, match="nonfinite"):
        module.parse_stream(stream)


def test_finite_logprobs_allow_correct_answer():
    body = response()
    body["choices"][0]["logprobs"] = {
        "content": [
            {
                "token": "323",
                "logprob": -0.125,
                "top_logprobs": [
                    {"token": "323", "logprob": -0.125},
                    {"token": "324", "logprob": -9999.0},
                ],
            }
        ]
    }
    assert harness().validate_response("arithmetic", body)["passed"]


def test_l2_suite_covers_five_routes_and_headless_tools_images():
    module = harness()
    cases = module.build_l2_cases(EXAMPLE)
    assert {c.get("expected_profile") for c in cases} >= {
        "primary",
        "qwen_direct",
        "qwen_think_medium",
        "deepseek_direct",
        "deepseek_think",
    }
    assert all(c["payload"]["model"] == "kairyu-auto-max" for c in cases)
    assert any(c["payload"].get("response_format") for c in cases)
    assert any(c["payload"].get("tools") for c in cases)
    assert any(isinstance(c["payload"]["messages"][0]["content"], list) for c in cases)


def test_l2_trace_records_observed_route_and_reports_coverage_gap():
    module = harness()
    body = response("answer")
    body["kairyu_trace_v2"] = {
        "events": [
            {"node": "profile_judge", "status": "success"},
            {"node": "deepseek_answer", "status": "success"},
        ]
    }
    result = module.validate_l2_trace(body, expected_profile="deepseek_direct")
    assert result["passed"] and result["observed_profile"] == "deepseek_direct"
    result = module.validate_l2_trace(body, expected_profile="primary")
    assert not result["passed"] and result["coverage_gap"]
    body["kairyu_trace_v2"]["events"].append({"node": "image_description", "status": "success"})
    assert not module.validate_l2_trace(body)["passed"]


def test_stream_preserves_kairyu_trace_and_reasoning_alias():
    module = harness()
    raw = (
        'data: {"choices":[{"delta":{"reasoning":"private"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"323"},"finish_reason":"stop"}]}\n\n'
        'data: {"kairyu_trace_v2":{"events":[{"node":"qwen_answer","status":"success"}]}}\n\n'
        "data: [DONE]\n\n"
    )
    body, done = module.parse_stream(raw)
    assert done
    assert body["choices"][0]["message"]["reasoning_content"] == "private"
    assert body["kairyu_trace_v2"]["events"][0]["node"] == "qwen_answer"


def test_l2_effort_matrix_preserves_primary_prompt_and_covers_api_values():
    cases = harness().build_l2_cases(EXAMPLE, effort_matrix=True)
    primary = [c for c in cases if c.get("expected_profile") == "primary"]
    assert len(primary) == 4
    assert {c["payload"].get("reasoning_effort") for c in primary} == {None, "low", "high", "max"}
    assert all(c["payload"]["messages"] == primary[0]["payload"]["messages"] for c in primary)


def test_cli_enables_l2_effort_matrix(tmp_path, monkeypatch):
    import sys

    module = harness()
    seen = []

    def execute(case, **kwargs):
        seen.append(case)
        return {"passed": True, "elapsed_seconds": 0.0, "name": case["name"]}

    monkeypatch.setattr(module, "execute_case", execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu_smoke.py",
            "--suite",
            "l2",
            "--l2-effort-matrix",
            "--case",
            "^l2-route-primary",
            "--base-url",
            "http://unused/v1",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert module.main() == 0
    assert len(seen) == 4


def budget_response(tokens=16, content="437", finish="stop"):
    body = response(content, finish)
    body["choices"][0]["message"]["reasoning"] = "Independent private calculation"
    body["usage"] = {
        "completion_tokens": 20,
        "completion_tokens_details": {"reasoning_tokens": tokens},
    }
    return body


def test_budget_probe_reaches_limit_and_finishes_public_answer():
    module = harness()
    case = next(
        c
        for c in module.build_cases(EXAMPLE, "deepseek-v4.1-flash")
        if c["kind"] == "thinking-budget"
    )
    assert case["payload"]["thinking_token_budget"] == 16
    assert case["payload"]["max_tokens"] == 512
    assert case["payload"]["reasoning_effort"] == "high"
    assert case["payload"]["chat_template_kwargs"]["thinking"] is True
    verdict = module.validate_response("thinking-budget", budget_response())
    assert verdict["passed"] and verdict["budget_reached"]
    assert verdict["reasoning_tokens_upper_bound"] == 16
    assert verdict["marker_allowance_tokens"] == 0


@pytest.mark.parametrize("tokens", [None, 0, 15])
def test_budget_probe_below_limit_is_not_exercised(tokens):
    verdict = harness().validate_response("thinking-budget", budget_response(tokens))
    assert not verdict["passed"] and verdict["outcome"] == "not_exercised"


@pytest.mark.parametrize(
    "body",
    [
        budget_response(17),
        budget_response(16, finish="length"),
        budget_response(16, content="wrong"),
    ],
)
def test_budget_probe_excess_truncation_or_wrong_answer_fails(body):
    assert not harness().validate_response("thinking-budget", body)["passed"]


def primary_response(*, empty=None, draft_tokens=512):
    body = response("Final answer")
    nodes = [
        "profile_judge",
        "head",
        "draft",
        "requirements",
        "policies",
        "answer_1",
        "answer_2",
        "critique",
        "synthesis",
        "audit",
    ]
    body["kairyu_trace_v2"] = {
        "events": [
            {
                "node": node,
                "status": "success",
                "usage": {"completion_tokens": draft_tokens if node == "draft" else 100},
            }
            for node in nodes
        ]
    }
    sections = []
    for node in ["draft", "answer_1", "answer_2", "critique"]:
        text = "" if node == empty else "A complete candidate answer."
        sections.append(
            f"### {node} — attempt 1\n\n- L2 role: `proposal`\n\n"
            f"#### Model reasoning\n\nPrivate reasoning.\n\n#### Stage output\n{text}\n---\n"
        )
    body["choices"][0]["message"]["reasoning_content"] = "\n".join(sections)
    return body


def test_primary_requires_nonempty_three_peer_bodies_not_just_success_events():
    module = harness()
    assert module.validate_l2_trace(primary_response(), expected_profile="primary")["passed"]
    verdict = module.validate_l2_trace(
        primary_response(empty="answer_1"), expected_profile="primary"
    )
    assert not verdict["passed"] and "answer_1" in verdict["detail"]
    assert "empty" in verdict["detail"]


def test_primary_rejects_candidate_at_cap_without_complete_output_evidence():
    verdict = harness().validate_l2_trace(
        primary_response(draft_tokens=2048), expected_profile="primary"
    )
    assert not verdict["passed"] and "draft" in verdict["detail"] and "cap" in verdict["detail"]


def test_headed_primary_rejects_observed_sentence_heading_seam():
    body = primary_response()
    body["choices"][0]["message"]["content"] = "Both options violate a constraint.**Facts**"
    assert not harness().validate_l2_trace(body, expect_headless=False)["passed"]
    body["choices"][0]["message"]["content"] = "Both options fail.Facts: A is slower."
    assert not harness().validate_l2_trace(body, expect_headless=False)["passed"]
    body["choices"][0]["message"]["content"] = "Both options violate a constraint.\n\n**Facts**"
    assert harness().validate_l2_trace(body, expect_headless=False)["passed"]
    body["choices"][0]["message"]["content"] = (
        "The requested literal follows.\n\n```text\nx.**A\n```"
    )
    assert harness().validate_l2_trace(body, expect_headless=False)["passed"]


def test_requirement_literal_must_be_in_acceptance_not_only_source():
    row = {
        "id": "R1",
        "priority": "minimum",
        "requirement": "End with the required sentence.",
        "acceptance_criterion": "Ends with exactly the literal ",
        "source": "End with exactly: Ready.",
    }
    body = response(json.dumps([row]))
    module = harness()
    assert not module.validate_response(
        "requirements", body, expected_literals=["Ready."]
    )["passed"]
    row["acceptance_criterion"] = "Ends with exactly the literal Ready. and nothing after it."
    assert module.validate_response(
        "requirements", response(json.dumps([row])), expected_literals=["Ready."]
    )["passed"]
