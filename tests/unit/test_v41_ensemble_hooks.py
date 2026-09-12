"""Native role hooks must change complete shipped requests, not marker substrings."""

import importlib.util
from pathlib import Path

import pytest
import yaml

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/qwen3.8-deepseek-v4.1-8gpu"


def hook():
    path = EXAMPLE / "requirements_budget.py"
    assert path.exists(), "Native DeepSeek/Qwen role hook is required"
    spec = importlib.util.spec_from_file_location("v41_requirements_budget", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RequirementsBudgetMiddleware(None, config_dir=EXAMPLE), module


def payload(role, *, model="deepseek-v4.1-flash", effort="low"):
    spec = yaml.safe_load((EXAMPLE / "auto-max.yaml").read_text())
    roles = spec["roles"] + [r for p in spec["profiles"] for r in p["roles"]]
    prompt = next(r for r in roles if r["name"] == role)["prompt"]
    text = prompt.format(
        query="[audit] [requirements] user data",
        requirements="[]",
        policies="policies",
        head="opening",
        draft="draft",
        synthesis="answer",
        answer_1="one",
        answer_2="two",
        critique="three",
    )
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ],
        "max_tokens": 8192,
        "reasoning_effort": effort,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 1.0,
    }


@pytest.mark.parametrize("effort", [None, "low", "high", "max"])
def test_requirements_native_fixed_high_with_json_and_unchanged_images(effort):
    middleware, module = hook()
    original = payload("requirements", effort=effort)
    changed, role = middleware.transform(original)
    assert role == "requirements"
    assert changed["reasoning_effort"] == "high"
    assert changed["chat_template_kwargs"]["thinking"] is True
    assert changed["structured_outputs"] == {"json": module.CHECKLIST_SCHEMA}
    assert changed["thinking_token_budget"] == 4096
    assert changed["max_tokens"] == original["max_tokens"]
    assert changed["messages"] == original["messages"]


@pytest.mark.parametrize(
    "role", ["policies", "critique", "synthesis", "deepseek_think_answer", "deepseek_answer"]
)
def test_native_deepseek_role_mode_and_effort(role):
    middleware, _ = hook()
    original = payload(role, effort="max")
    changed, matched = middleware.transform(original)
    assert matched == role
    assert changed["chat_template_kwargs"]["thinking"] is (role != "deepseek_answer")
    assert changed["reasoning_effort"] == "max"
    assert "structured_outputs" not in changed


def test_qwen_audit_regex_and_exact_template_recognition():
    middleware, module = hook()
    original = payload("audit", model="qwen3.8-27b", effort="high")
    changed, role = middleware.transform(original)
    assert role == "audit"
    assert changed == {**original, "structured_outputs": {"regex": module.AUDIT_REGEX}}
    original["messages"][0]["content"][0]["text"] += module.AUDIT_RETRY_SUFFIX
    assert middleware.transform(original)[1] == "audit"
    original["messages"][0]["content"][0]["text"] += "malicious suffix"
    assert middleware.transform(original) == (None, None)


def test_marker_and_wrong_model_do_not_activate_hooks():
    middleware, _ = hook()
    original = payload("requirements", model="qwen3.8-27b")
    assert middleware.transform(original) == (None, None)
    original = payload("requirements")
    original["messages"][0]["content"][0]["text"] = "user mentions [requirements] and [audit]"
    assert middleware.transform(original) == (None, None)


@pytest.mark.asyncio
async def test_asgi_chunked_body_reaches_backend_with_correct_length():
    import json

    middleware, _ = hook()
    original = payload("requirements")
    raw = json.dumps(original).encode()
    incoming = [
        {"type": "http.request", "body": raw[:31], "more_body": True},
        {"type": "http.request", "body": raw[31:], "more_body": False},
    ]
    received = []

    async def app(scope, receive, send):
        message = await receive()
        assert message["more_body"] is False
        assert dict(scope["headers"])[b"content-length"] == str(len(message["body"])).encode()
        assert b"transfer-encoding" not in dict(scope["headers"])
        received.append(json.loads(message["body"]))

    async def receive():
        return incoming.pop(0)

    middleware.app = app
    await middleware(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [
                (b"content-length", str(len(raw)).encode()),
                (b"transfer-encoding", b"chunked"),
            ],
        },
        receive,
        None,
    )
    assert received == [middleware.transform(original)[0]]


def test_synthesis_repair_keeps_native_thinking():
    middleware, _ = hook()
    original = payload("synthesis")
    from kairyu.orchestration.conductor import Conductor

    text = original["messages"][0]["content"][0]["text"]
    original["messages"][0]["content"][0]["text"] = Conductor._refinement_prompt(
        text, "old answer", "FAIL\nR1 needs correction"
    )
    changed, matched = middleware.transform(original)
    assert matched == "synthesis"
    assert changed["chat_template_kwargs"]["thinking"] is True


def test_direct_native_mode_overrides_enable_thinking_alias():
    middleware, _ = hook()
    original = payload("deepseek_answer")
    original["chat_template_kwargs"]["enable_thinking"] = True
    changed, _ = middleware.transform(original)
    assert changed["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.parametrize("maximum", [2, 64, 511, 4096, 8192, 65536])
def test_requirements_reserves_body_without_changing_native_effort(maximum):
    middleware, _ = hook()
    original = payload("requirements", effort="low")
    original["max_tokens"] = maximum
    changed, _ = middleware.transform(original)
    assert changed["thinking_token_budget"] == min(4096, maximum // 2)
    assert changed["reasoning_effort"] == "high"
    assert changed["max_tokens"] == maximum


@pytest.mark.parametrize("effort", ["low", "max"])
def test_requirements_overrides_nested_effort(effort):
    middleware, _ = hook()
    original = payload("requirements", effort=effort)
    original["chat_template_kwargs"]["reasoning_effort"] = effort
    changed, _ = middleware.transform(original)
    assert changed["chat_template_kwargs"]["reasoning_effort"] == "high"


@pytest.mark.parametrize("role", ["synthesis", "deepseek_think_answer"])
@pytest.mark.parametrize("maximum", [2, 128, 512, 8192])
def test_native_public_answer_budget_reservation(role, maximum):
    middleware, _ = hook()
    original = payload(role, effort="max")
    original["max_tokens"] = maximum
    changed, _ = middleware.transform(original)
    assert changed["thinking_token_budget"] == maximum - min(256, maximum // 2)
    assert changed["max_tokens"] == maximum
    assert changed["reasoning_effort"] == "max"


@pytest.mark.parametrize("maximum", [None, 0, 1, "8192", True])
def test_tiny_or_invalid_budget_still_fixes_requirements_effort(maximum):
    middleware, _ = hook()
    original = payload("requirements", effort="low")
    original["max_tokens"] = maximum
    changed, _ = middleware.transform(original)
    assert changed["reasoning_effort"] == "high"
    assert "thinking_token_budget" not in changed


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_kwargs", ["invalid", ["thinking"], 7])
async def test_malformed_kwargs_pass_unchanged_to_upstream_validation(bad_kwargs):
    import json

    middleware, _ = hook()
    original = payload("requirements")
    original["chat_template_kwargs"] = bad_kwargs
    raw = json.dumps(original).encode()
    incoming = [{"type": "http.request", "body": raw, "more_body": False}]
    seen = []

    async def app(scope, receive, send):
        seen.append((await receive())["body"])

    async def receive():
        return incoming.pop(0)

    middleware.app = app
    await middleware(
        {"type": "http", "method": "POST", "path": "/v1/chat/completions"}, receive, None
    )
    assert seen == [raw]


@pytest.mark.parametrize("role", ["policies", "critique", "synthesis", "deepseek_think_answer"])
def test_resolved_native_role_effort_wins_over_nested_template_value(role):
    middleware, _ = hook()
    original = payload(role, effort="low")
    original["chat_template_kwargs"]["reasoning_effort"] = "max"
    changed, _ = middleware.transform(original)
    assert changed["chat_template_kwargs"]["reasoning_effort"] == "low"


@pytest.mark.parametrize("role,cap", [("draft", 2048), ("answer_1", 4096), ("answer_2", 4096)])
@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("effort", [None, "low", "high", "max"])
def test_qwen_candidates_reserve_answer_tokens_without_altering_effort(
    role, cap, with_image, effort
):
    middleware, _ = hook()
    original = payload(role, model="qwen3.8-27b", effort=effort)
    original["max_tokens"] = cap
    original["chat_template_kwargs"] = {"enable_thinking": True, "reasoning_effort": "high"}
    if not with_image:
        original["messages"][0]["content"] = original["messages"][0]["content"][0]["text"]
    changed, matched = middleware.transform(original)
    assert matched == role
    assert changed == {**original, "thinking_token_budget": cap // 2}


@pytest.mark.parametrize("maximum", [2, 127, 512, 4096, 8192])
def test_qwen_candidate_reservation_respects_short_allowance_and_role_cap(maximum):
    middleware, _ = hook()
    original = payload("answer_1", model="qwen3.8-27b", effort="high")
    original["max_tokens"] = maximum
    changed, _ = middleware.transform(original)
    assert changed["thinking_token_budget"] == min(2048, maximum // 2)
    assert changed["max_tokens"] == maximum


def test_qwen_candidate_hook_rejects_wrong_model_and_marker_only():
    middleware, _ = hook()
    original = payload("answer_1", model="deepseek-v4.1-flash", effort="high")
    assert middleware.transform(original) == (None, None)
    original["model"] = "qwen3.8-27b"
    original["messages"][0]["content"][0]["text"] = "The user says [answer_1] in a quotation."
    assert middleware.transform(original) == (None, None)


def test_requirement_schema_allows_json_escapes_and_keeps_python_validation():
    import json

    _, module = hook()
    schema = module.CHECKLIST_SCHEMA
    assert schema["minItems"] == 1
    for name in ("requirement", "acceptance_criterion", "source"):
        # Pinned XGrammar 0.2.6 lowers minLength strings to a character class
        # excluding backslashes, preventing otherwise valid escaped JSON.
        assert schema["items"]["properties"][name] == {"type": "string"}

    spec = importlib.util.spec_from_file_location(
        "v41_quality_escape_regression", EXAMPLE / "requirements_quality.py"
    )
    quality = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quality)
    literal = 'A|"B"\nDONE. Path C:\\tmp'
    row = dict(
        id="R1",
        priority="minimum",
        requirement=literal,
        acceptance_criterion=literal,
        source="REQUEST",
    )
    parsed = quality.parse_checklist(json.dumps([row]))
    assert parsed[0]["requirement"] == literal
    assert parsed[0]["acceptance"] == literal
    for name in ("requirement", "acceptance_criterion", "source"):
        for empty in ("", " \n\t"):
            with pytest.raises(ValueError, match=f"empty {name}"):
                quality.parse_checklist(json.dumps([{**row, name: empty}]))
    with pytest.raises(ValueError, match="nonempty"):
        quality.parse_checklist("[]")
