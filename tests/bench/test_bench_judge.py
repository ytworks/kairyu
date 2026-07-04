"""JudgeClient verdict parsing/degradation + judged adapters (HLE free-form, CharXiv)."""

import json

import httpx
from conftest import make_config, make_target

from kairyu.bench.judge import JudgeClient, parse_verdict
from kairyu.bench.runner import SuiteRunner
from kairyu.bench.store import ResultStore
from kairyu.bench.types import JudgeConfig


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


async def test_judge_http_failure_degrades_to_none():
    factory = _canned_factory(
        lambda model, body: httpx.Response(500, json={"error": "boom"})
    )
    judge = _judge(factory, max_retries=0)
    verdict = await judge.grade(question="q", expected="a", response="r")
    assert verdict.correct is None
    assert "judge request failed" in verdict.raw_excerpt


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
