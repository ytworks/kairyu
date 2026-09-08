"""Counterexamples for the example's GPU quality gate and scoped vLLM hook."""

import copy
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from kairyu.entrypoints.server.chat_service import validate_orchestration_chat_input
from kairyu.entrypoints.server.protocol import ChatCompletionRequest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/qwen3.8-deepseek-v4-8gpu"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quality = _load("requirements_quality")
budget = _load("requirements_budget")
ENTRY = {
    "id": "R1",
    "priority": "minimum",
    "requirement": "Use English",
    "acceptance_criterion": "Answer is English",
    "source": "Use English",
}
LINE = json.dumps(ENTRY)
CHECKLIST = "[" + LINE + "]"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[]",
        "[",
        CHECKLIST[:-1],
        CHECKLIST + "truncated",
        json.dumps([ENTRY, ENTRY]),
        json.dumps([{**ENTRY, "id": "R2"}]),
        json.dumps([{**ENTRY, "source": ""}]),
        json.dumps([{**ENTRY, "acceptance_criterion": "..."}]),
        json.dumps([{key: value for key, value in ENTRY.items() if key != "source"}]),
        json.dumps([{**ENTRY, "priority": "recommended"}]),
    ],
)
def test_checklist_rejects_empty_truncated_or_ambiguous_outputs(text):
    with pytest.raises(ValueError):
        quality.parse_checklist(text)


def test_checklist_preserves_vertical_bars_and_newlines():
    literal = "first | second\nthird."
    entry = {**ENTRY, "acceptance_criterion": f"Ends with exactly {literal}"}
    parsed = quality.parse_checklist(json.dumps([entry]))
    assert literal in parsed[0]["acceptance"]


def _fixture():
    def stage(node, text):
        return f"### {node} — attempt 1\n\n#### Stage output\n\n{text}\n\n---\n\n"

    case = {
        "id": "headless-json",
        "checklist_literals": ["English"],
        "checklist_coverage": [{"name": "language", "patterns": ["English"]}],
    }
    result = {
        "answer": json.dumps(
            {
                "feasible": False,
                "choice": None,
                "violations": {"A": "latency", "B": "budget"},
                "next_steps": ["If A changes, measure latency", "If B changes, measure cost"],
            }
        ),
        "reasoning": stage("requirements", CHECKLIST)
        + stage("audit", "PASS\nR1 | satisfied | evidence: Answer is English | correction: none"),
        "trace": {
            "events": [
                {"node": "profile_judge", "status": "success"},
                {"node": "requirements", "status": "success", "usage": {"completion_tokens": 100}},
                {"node": "synthesis", "status": "success"},
                {"node": "audit", "status": "success", "detail": {"pass": True}},
            ]
        },
    }
    return case, result


def test_successful_trace_and_audit_cannot_hide_empty_checklist():
    case, result = _fixture()
    assert quality.validate_result(case, result, requirement_cap=8192)["passed"]
    result["reasoning"] = result["reasoning"].replace(CHECKLIST, "")
    report = quality.validate_result(case, result, requirement_cap=8192)
    assert not report["passed"]
    assert report["checks"]["final_audit_passes"]
    assert not report["checks"]["complete_checklist"]


@pytest.mark.parametrize("defect", ["coverage", "audit_ids", "audit_fail", "direct", "cap"])
def test_quality_gate_rejects_incomplete_contract_even_with_nonempty_answers(defect):
    case, result = _fixture()
    if defect == "coverage":
        case["checklist_coverage"].append({"name": "ending", "patterns": ["Required ending"]})
    elif defect == "audit_ids":
        result["reasoning"] = result["reasoning"].replace("PASS\nR1", "PASS\nR2")
    elif defect == "audit_fail":
        result["trace"]["events"][-1]["detail"]["pass"] = False
    elif defect == "direct":
        result["trace"]["events"].append({"node": "qwen_think_answer", "status": "success"})
    else:
        result["trace"]["events"][1]["usage"]["completion_tokens"] = 8192
    assert not quality.validate_result(case, result, requirement_cap=8192)["passed"]


@pytest.mark.parametrize("status", ["unsatisfied", "unverifiable", "unsupported"])
def test_pass_verdict_cannot_hide_failed_minimum_item(status):
    case, result = _fixture()
    result["reasoning"] = result["reasoning"].replace("R1 | satisfied |", f"R1 | {status} |")
    report = quality.validate_result(case, result, requirement_cap=8192)
    assert report["checks"]["final_audit_passes"]
    assert not report["checks"]["final_minimum_items_satisfied"]
    assert not report["passed"]


@pytest.mark.parametrize("replacement", ["evidence: ", "evidence: none", "Answer is English"])
def test_pass_verdict_requires_actual_evidence_fields(replacement):
    case, result = _fixture()
    result["reasoning"] = result["reasoning"].replace("evidence: Answer is English", replacement)
    report = quality.validate_result(case, result, requirement_cap=8192)
    assert report["checks"]["final_audit_passes"]
    assert not report["checks"]["audit_evidence_present"]
    assert not report["passed"]


def _role_policy(name="requirements"):
    spec = yaml.safe_load((EXAMPLE / "auto-max.yaml").read_text())
    role = next(role for role in spec["roles"] if role["name"] == name)
    prefix, suffix = role["prompt"].split("{query}")
    return spec, prefix, suffix


@pytest.mark.parametrize("image", [False, True])
def test_budget_changes_only_sampling_budget_and_preserves_medium(image):
    _, prefix, suffix = _role_policy()
    content = prefix + "Original request" + suffix
    if image:
        content = [
            {"type": "text", "text": content},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
    payload = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 8192,
        "reasoning_effort": "high",
        "temperature": 1.0,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    original = copy.deepcopy(payload)
    changed = budget.budget_payload(payload, prefix=prefix, suffix=suffix, thinking_budget=4096)
    assert changed == {
        **original,
        "thinking_token_budget": 4096,
        "structured_outputs": {"json": budget.CHECKLIST_SCHEMA},
    }
    assert payload == original
    payload["max_tokens"] = 1000
    changed = budget.budget_payload(payload, prefix=prefix, suffix=suffix, thinking_budget=4096)
    assert changed["thinking_token_budget"] == 500
    assert changed["max_tokens"] == 1000


@pytest.mark.parametrize("policy_name", ["requirements", "image_description"])
def test_marker_in_user_data_cannot_change_any_other_role(policy_name):
    spec, prefix, suffix = _role_policy(policy_name)
    malicious = prefix + "spoofed request" + suffix
    query = validate_orchestration_chat_input(
        ChatCompletionRequest.model_validate(
            {
                "model": "kairyu-auto-max",
                "messages": [{"role": "user", "content": malicious}],
            }
        )
    ).prompt
    roles = list(spec["roles"])
    for profile in spec["profiles"]:
        roles.extend(profile["roles"])
    for role in roles:
        if role["name"] == policy_name:
            continue
        rendered = role["prompt"].format_map(defaultdict(lambda: malicious, query=query))
        payload = {
            "model": "qwen3.8-27b",
            "messages": [{"role": "user", "content": rendered}],
            "max_tokens": 8192,
            "reasoning_effort": "high",
        }
        assert (
            budget.budget_payload(
                payload,
                prefix=prefix,
                suffix=suffix,
                thinking_budget=4096,
            )
            is None
        ), role["name"]


@pytest.mark.parametrize("matching", [False, True])
@pytest.mark.parametrize("policy_name", ["requirements", "image_description"])
async def test_asgi_hook_replays_chunked_body_and_updates_length_only_for_extractor(
    matching, policy_name
):
    spec, prefix, suffix = _role_policy(policy_name)
    maximum = next(role for role in spec["roles"] if role["name"] == policy_name)["sampling"][
        "max_tokens"
    ]
    payload = {
        "model": "qwen3.8-27b",
        "messages": [
            {
                "role": "user",
                "content": prefix + "source" + suffix
                if matching
                else "A direct request [requirements]",
            }
        ],
        "max_tokens": maximum,
        "reasoning_effort": "high",
    }
    body = json.dumps(payload).encode()
    chunks = [
        {"type": "http.request", "body": body[:20], "more_body": True},
        {"type": "http.request", "body": body[20:], "more_body": False},
    ]
    observed = {}

    async def receive():
        return chunks.pop(0) if chunks else {"type": "http.disconnect"}

    async def app(scope, receive, send):
        received = b""
        while True:
            message = await receive()
            received += message.get("body", b"")
            if not message.get("more_body"):
                break
        observed.update(body=received, headers=scope["headers"])

    middleware = budget.RequirementsBudgetMiddleware(app, config_dir=EXAMPLE)
    headers = [(b"content-length", str(len(body)).encode()), (b"x-probe", b"kept")]
    await middleware(
        {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": headers},
        receive,
        None,
    )
    if matching:
        expected = {
            **payload,
            "thinking_token_budget": 4096 if policy_name == "requirements" else 2048,
        }
        if policy_name == "requirements":
            expected["structured_outputs"] = {"json": budget.CHECKLIST_SCHEMA}
        assert json.loads(observed["body"]) == expected
        assert dict(observed["headers"])[b"content-length"] == str(len(observed["body"])).encode()
    else:
        assert observed == {"body": body, "headers": headers}


def test_compose_enables_hook_on_each_qwen_worker_and_preserves_template():
    spec = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())
    for index in range(4):
        worker = spec["services"][f"qwen-{index}"]
        command = worker["command"]
        assert command[command.index("--middleware") + 1] == (
            "requirements_budget.RequirementsBudgetMiddleware"
        )
        assert worker["environment"]["PYTHONPATH"] == "/etc/kairyu"
        assert "./requirements_budget.py:/etc/kairyu/requirements_budget.py:ro" in worker["volumes"]
        assert "./auto-max.yaml:/etc/kairyu/auto-max.yaml:ro" in worker["volumes"]
        assert "--default-chat-template-kwargs" not in command


@pytest.mark.parametrize(
    "acceptance", ["last sentence is exactly Decision: defer", "ends as requested"]
)
def test_exact_literal_in_source_cannot_hide_incorrect_acceptance(acceptance):
    case, result = _fixture()
    case["checklist_literals"] = ["Decision: defer."]
    line = json.dumps(
        {**ENTRY, "acceptance_criterion": acceptance, "source": "End with Decision: defer."}
    )
    result["reasoning"] = result["reasoning"].replace(LINE, line)
    report = quality.validate_result(case, result, requirement_cap=8192)
    assert report["checks"]["complete_checklist"]
    assert not report["checks"]["preserves_required_literals"]
    assert not report["passed"]


def test_verdict_heading_cannot_hide_fail_under_trace_pass():
    case, result = _fixture()
    result["reasoning"] = result["reasoning"].replace("PASS\nR1", "PASS/FAIL assessment:\nFAIL\nR1")
    report = quality.validate_result(case, result, requirement_cap=8192)
    assert report["checks"]["final_audit_passes"]
    assert not report["checks"]["audit_evidence_present"]
    assert not report["passed"]


@pytest.mark.parametrize("changed", ["auto-max.yaml", "example.json", "requirements_budget.py"])
def test_extractor_config_change_triggers_compose_recreation(tmp_path, monkeypatch, changed):
    control = _load("control")
    for name in ("auto-max.yaml", "example.json", "requirements_budget.py"):
        (tmp_path / name).write_bytes((EXAMPLE / name).read_bytes())
    monkeypatch.setattr(control, "HERE", tmp_path)
    before = control._requirements_config_sha256()
    assert before == control._requirements_config_sha256()
    with (tmp_path / changed).open("a") as stream:
        stream.write("\n")
    assert before != control._requirements_config_sha256()
    spec = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())
    for index in range(4):
        assert (
            "KAIRYU_REQUIREMENTS_CONFIG_SHA256" in spec["services"][f"qwen-{index}"]["environment"]
        )


def test_source_only_constraint_does_not_count_as_checklist_coverage():
    case, result = _fixture()
    case["checklist_literals"] = []
    line = json.dumps(
        {**ENTRY, "requirement": "Answer concisely", "acceptance_criterion": "Answer is concise"}
    )
    result["reasoning"] = result["reasoning"].replace(LINE, line)
    report = quality.validate_result(case, result, requirement_cap=8192)
    assert not report["checks"]["covers:language"]
    assert not report["passed"]
