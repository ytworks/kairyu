"""JudgeClient verdict parsing/degradation + judged adapters (HLE free-form, CharXiv)."""

import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from conftest import make_config, make_target

from evals.adapters.charxiv import CharXivAdapter
from evals.adapters.hle import HleAdapter
from evals.judge import (
    JudgeClient,
    MajorityJudgeClient,
    build_judge_client,
    parse_verdict,
)
from evals.judge_prompts import judge_template
from evals.runner import SuiteRunner
from evals.store import ResultStore
from evals.types import BenchItem, JudgeConfig, JudgeEndpointConfig


def test_parse_verdict():
    assert parse_verdict("reasoning: fine\ncorrect: yes") is True
    assert parse_verdict("correct: no") is False
    assert parse_verdict("CORRECT: Yes") is True
    assert parse_verdict("correct: **no**") is False
    assert parse_verdict("the answer looks correct to me") is None
    # the last verdict field wins (models sometimes restate)
    assert parse_verdict("correct: no\n...revised...\ncorrect: yes") is True
    # M10: markdown-emphasized labels must still parse (bold before the label)
    assert parse_verdict("**correct:** yes") is True
    assert parse_verdict("Correct: **No**") is False
    assert parse_verdict("**Correct**: yes") is True


def test_charxiv_preserves_target_output_budget_for_reasoning_models():
    request = CharXivAdapter().build_request(
        BenchItem(
            id="charxiv",
            payload={
                "question": "What is shown?",
                "answer": "one",
                "image": "data:image/png;base64,AAAA",
            },
        ),
        make_target(max_output_tokens=8192),
        SimpleNamespace(),
    )

    assert request.max_tokens == 8192


@pytest.mark.parametrize(
    ("name", "sha256"),
    [
        (
            "hle",
            "0befab1400eba0c4d4d4e16cc018b4d988d775bbf4ad554fed5fe5f76fede74d",
        ),
        (
            "charxiv-reasoning",
            "e8bf1acd527e025da041a43acd729924abc44610194cbf696f8e228c50b2319d",
        ),
    ],
)
def test_production_judge_template_bytes_remain_frozen(name, sha256):
    template = judge_template(name)
    assert hashlib.sha256(template.encode("utf-8")).hexdigest() == sha256


@pytest.mark.parametrize(
    ("name", "production_first", "production_second"),
    [
        (
            "hle",
            "[response]: {response}",
            "[correct_answer]: {expected}",
        ),
        (
            "charxiv-reasoning",
            "Ground-truth answer: {expected}",
            "Model response: {response}",
        ),
    ],
)
def test_counterbalanced_template_only_reverses_reference_response_blocks(
    name,
    production_first,
    production_second,
):
    production = judge_template(name)
    counterbalanced = judge_template(name, counterbalanced=True)
    production_pair = f"{production_first}\n\n{production_second}"
    reversed_pair = f"{production_second}\n\n{production_first}"
    sentinel = "<reference-response-block-order>"

    assert production.count(production_pair) == 1
    assert counterbalanced.count(reversed_pair) == 1
    assert production.replace(production_pair, sentinel) == counterbalanced.replace(
        reversed_pair, sentinel
    )


def _canned_factory(reply_for_model):
    """MockTransport routing by requested model name."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        reply = reply_for_model(body["model"], body)
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": reply}}
                ],
            },
        )

    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return factory


def _judge(factory, **overrides) -> JudgeClient:
    config = JudgeConfig(base_url="http://judge/v1", model="judge-m", **overrides)
    return JudgeClient(config, http_factory=factory)


async def test_judge_grade_yes_no_and_unparseable():
    replies = iter(["correct: yes", "correct: no", "hmm hard to say"])
    factory = _canned_factory(lambda model, body: next(replies))
    judge = _judge(factory)
    yes = await judge.grade(question="q", expected="a", response="r")
    no = await judge.grade(question="q", expected="a", response="r")
    unknown = await judge.grade(question="q", expected="a", response="r")
    assert yes.correct is True
    assert no.correct is False
    assert unknown.correct is None
    assert yes.as_dict()["model"] == "judge-m"
    assert yes.as_dict() == {
        "model": "judge-m",
        "correct": True,
        "raw_excerpt": "correct: yes",
    }


async def test_two_judge_panel_requires_unanimous_parseable_votes():
    config = JudgeConfig(
        base_url="http://judge/v1",
        model="primary",
        additional_judges=(
            JudgeEndpointConfig(base_url="http://judge/v1", model="secondary"),
        ),
    )
    panel = build_judge_client(
        config,
        http_factory=_canned_factory(
            lambda model, body: "correct: yes" if model == "primary" else "correct: no"
        ),
    )
    assert isinstance(panel, MajorityJudgeClient)
    verdict = await panel.grade(question="q", expected="e", response="r")
    assert verdict.correct is None
    assert verdict.failure_reason == "no strict majority (1 yes, 1 no)"
    evidence = verdict.as_dict()
    assert evidence["aggregation"] == "strict-majority"
    assert [vote["model"] for vote in evidence["votes"]] == [
        "primary",
        "secondary",
    ]
    assert [vote["correct"] for vote in evidence["votes"]] == [True, False]


async def test_two_judge_panel_agrees_and_unavailable_member_fails_closed():
    config = JudgeConfig(
        base_url="http://judge/v1",
        model="primary",
        additional_judges=(
            JudgeEndpointConfig(base_url="http://judge/v1", model="secondary"),
        ),
    )
    agreed = build_judge_client(
        config,
        http_factory=_canned_factory(lambda model, body: "correct: no"),
    )
    assert (await agreed.grade(question="q", expected="e", response="r")).correct is False

    unavailable = build_judge_client(
        config,
        http_factory=_canned_factory(
            lambda model, body: "correct: yes" if model == "primary" else "not sure"
        ),
    )
    verdict = await unavailable.grade(question="q", expected="e", response="r")
    assert verdict.correct is None
    assert verdict.failure_reason == "unavailable vote(s): secondary"


async def test_three_judge_panel_uses_true_strict_majority():
    config = JudgeConfig(
        base_url="http://judge/v1",
        model="primary",
        additional_judges=(
            JudgeEndpointConfig(base_url="http://judge/v1", model="second"),
            JudgeEndpointConfig(base_url="http://judge/v1", model="third"),
        ),
    )
    panel = build_judge_client(
        config,
        http_factory=_canned_factory(
            lambda model, body: "correct: no" if model == "primary" else "correct: yes"
        ),
    )
    verdict = await panel.grade(question="q", expected="e", response="r")
    assert verdict.correct is True
    assert verdict.model == "strict-majority-panel"
    assert verdict.failure_reason is None
    assert "threshold 2" in verdict.raw_excerpt
    evidence = verdict.as_dict()
    assert evidence["model"] == "strict-majority-panel"
    assert evidence["votes"][0]["model"] == "primary"
    assert evidence["votes"][0]["correct"] is False


async def test_judged_adapters_pass_their_fingerprinted_template_selector():
    seen = []

    class RecordingJudge:
        async def grade(self, **kwargs):
            seen.append(kwargs["template_name"])
            from evals.judge import JudgeVerdict

            return JudgeVerdict(correct=True, raw_excerpt="correct: yes", model="j")

    ctx = SimpleNamespace(judge=RecordingJudge())
    await HleAdapter().score(
        BenchItem(
            id="hle",
            payload={"question": "q", "answer": "a", "answer_type": "exactMatch"},
        ),
        "r",
        ctx,
    )
    await CharXivAdapter().score(
        BenchItem(id="charxiv", payload={"question": "q", "answer": "a"}),
        "r",
        ctx,
    )
    assert seen == [
        HleAdapter.info.judge_template_name,
        CharXivAdapter.info.judge_template_name,
    ]


async def test_judge_http_failure_degrades_to_none():
    factory = _canned_factory(
        lambda model, body: httpx.Response(500, json={"error": "boom"})
    )
    judge = _judge(factory, max_retries=0)
    verdict = await judge.grade(question="q", expected="a", response="r")
    assert verdict.correct is None
    assert "judge request failed" in verdict.raw_excerpt


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(httpx.Response(200, text="not-json"), id="malformed-json"),
        pytest.param(httpx.Response(200, json={"choices": []}), id="missing-choice"),
    ],
)
async def test_judge_malformed_success_response_degrades_to_none(response):
    judge = _judge(_canned_factory(lambda model, body: response), max_retries=0)
    verdict = await judge.grade(question="q", expected="a", response="r")
    assert verdict.correct is None
    assert "judge request raised" in verdict.raw_excerpt


async def test_judge_prompt_carries_question_expected_response():
    seen = {}

    def reply(model, body):
        seen["prompt"] = body["messages"][0]["content"]
        return "correct: yes"

    judge = _judge(_canned_factory(reply))
    await judge.grade(question="THE-Q", expected="THE-A", response="THE-R")
    assert "THE-Q" in seen["prompt"]
    assert "THE-A" in seen["prompt"]
    assert "THE-R" in seen["prompt"]


async def test_raw_judge_template_override_remains_backward_compatible():
    seen = {}

    def reply(model, body):
        seen["prompt"] = body["messages"][0]["content"]
        return "correct: yes"

    judge = _judge(_canned_factory(reply))
    verdict = await judge.grade(
        question="THE-Q",
        expected="THE-A",
        response="THE-R",
        # A legacy direct-library caller may still supply raw template text.
        # It must take precedence without requiring a registered selector.
        template_name="not-registered",
        template="legacy question={question}; answer={expected}; reply={response}",
    )

    assert verdict.correct is True
    assert seen["prompt"] == "legacy question=THE-Q; answer=THE-A; reply=THE-R"


def _suite_factory(judge_reply="correct: yes"):
    """Targets answer with a fixed string; the judge model answers the verdict."""

    def reply(model, body):
        if model == "judge-m":
            return judge_reply
        return "Final answer: photosynthesis"

    return _canned_factory(reply)


async def test_hle_freeform_judged_end_to_end(tmp_path):
    config = make_config(
        tmp_path,
        models=("m",),
        only=("hle",),
        judge=JudgeConfig(base_url="http://judge/v1", model="judge-m").model_dump(),
    )
    runner = SuiteRunner(
        config,
        http_factory=_suite_factory(),
        probe_docker=lambda: (False, "t"),
    )
    assert await runner.run() == 0
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("hle", "m")
    assert pair.status == "completed"
    assert pair.metrics["n_unjudged"] == 0
    freeform = next(item for item in pair.items if item.item_id == "fixture-hle-0003")
    assert freeform.judge["correct"] is True
    assert freeform.score == 1.0


async def test_hle_unparseable_judge_verdict_becomes_unjudged(tmp_path):
    config = make_config(
        tmp_path,
        models=("m",),
        only=("hle",),
        judge=JudgeConfig(base_url="http://judge/v1", model="judge-m").model_dump(),
    )
    runner = SuiteRunner(
        config,
        http_factory=_suite_factory(judge_reply="no idea"),
        probe_docker=lambda: (False, "t"),
    )
    await runner.run()
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("hle", "m")
    assert pair.status == "partial"
    assert pair.metrics["n_unjudged"] == 1


async def test_hle_panel_disagreement_becomes_unjudged_with_vote_evidence(tmp_path):
    config = make_config(
        tmp_path,
        models=("m",),
        only=("hle",),
        judge=JudgeConfig(
            base_url="http://judge/v1",
            model="judge-m",
            additional_judges=(
                JudgeEndpointConfig(
                    base_url="http://judge/v1", model="judge-secondary"
                ),
            ),
        ).model_dump(),
    )

    def reply(model, body):
        if model == "judge-m":
            return "correct: yes"
        if model == "judge-secondary":
            return "correct: no"
        return "Final answer: photosynthesis"

    runner = SuiteRunner(
        config,
        http_factory=_canned_factory(reply),
        probe_docker=lambda: (False, "t"),
    )
    assert await runner.run() == 0
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("hle", "m")
    assert pair.status == "partial"
    freeform = next(item for item in pair.items if item.item_id == "fixture-hle-0003")
    assert freeform.status == "unjudged"
    assert freeform.judge["aggregation"] == "strict-majority"
    assert [vote["correct"] for vote in freeform.judge["votes"]] == [True, False]


async def test_charxiv_without_judge_is_skipped(tmp_path, http_factory):
    config = make_config(tmp_path, models=("m",), only=("charxiv-reasoning",))
    runner = SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: (False, "t"))
    assert await runner.run() == 0
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("charxiv-reasoning", "m")
    assert pair.status == "skipped"
    assert "judge" in pair.reason


async def test_charxiv_skips_non_vision_target(tmp_path, http_factory):
    config = make_config(
        tmp_path,
        models=(),
        targets=(make_target(model="m", supports_vision=False),),
        only=("charxiv-reasoning",),
        judge=JudgeConfig(base_url="http://judge/v1", model="judge-m").model_dump(),
    )
    runner = SuiteRunner(
        config, http_factory=_suite_factory(), probe_docker=lambda: (False, "t")
    )
    await runner.run()
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("charxiv-reasoning", "m")
    assert pair.status == "skipped"
    assert "vision" in pair.reason


async def test_charxiv_judged_end_to_end(tmp_path):
    config = make_config(
        tmp_path,
        models=("m",),
        only=("charxiv-reasoning",),
        judge=JudgeConfig(base_url="http://judge/v1", model="judge-m").model_dump(),
    )
    runner = SuiteRunner(
        config, http_factory=_suite_factory(), probe_docker=lambda: (False, "t")
    )
    assert await runner.run() == 0
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("charxiv-reasoning", "m")
    assert pair.status == "completed"
    assert pair.score == 1.0
    assert all(item.judge is not None for item in pair.items)
