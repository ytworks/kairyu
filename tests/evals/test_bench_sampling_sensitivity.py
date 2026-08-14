"""Chat seed sweeps retain grouped evidence and honest sampling statistics."""

from __future__ import annotations

import json

import httpx
import pytest

from evals.adapters.base import (
    AdapterInfo,
    DownloadContext,
    GenerativeAdapter,
    RunContext,
    call_chat,
)
from evals.aggregate import build_scoreboard, render_markdown
from evals.cache import BenchCache
from evals.sampling import (
    PROTOCOL_ID as SAMPLING_PROTOCOL_ID,
)
from evals.sampling import (
    pass_at_k,
    sampling_summary,
    seed_schedule,
)
from evals.types import (
    BenchItem,
    BenchTarget,
    ChatRequestSpec,
    ItemResult,
    PairResult,
    pair_result_evidence_error,
)


class _BinaryAdapter(GenerativeAdapter):
    info = AdapterInfo(
        name="gsm8k",
        display_name="GSM8K",
        metric="accuracy",
        binary_outcomes=True,
    )

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        del ctx
        return []

    def load_items(self, ctx: RunContext) -> list[BenchItem]:
        del ctx
        return [
            BenchItem(id="a", payload={"prompt": "a"}),
            BenchItem(id="b", payload={"prompt": "b"}),
        ]

    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec:
        del target, ctx
        return ChatRequestSpec(
            messages=({"role": "user", "content": item.payload["prompt"]},)
        )

    async def score(
        self, item: BenchItem, response_text: str, ctx: RunContext
    ) -> ItemResult:
        del ctx
        return ItemResult(
            item_id=item.id,
            status="completed",
            score=1.0 if response_text == "pass" else 0.0,
        )


def _context(tmp_path, handler, *, attempts=4) -> RunContext:
    return RunContext(
        cache=BenchCache(tmp_path / "cache"),
        http_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ),
        attempts=attempts,
        concurrency=8,
        retries=0,
        offline_fixtures=True,
    )


def _scoreboard(pair, target):
    sampling = pair.methodology.get("sampling")
    attempts = sampling.get("attempts", 1) if isinstance(sampling, dict) else 1
    config = {"attempts": attempts}
    if attempts > 1:
        config["sampling_protocol"] = SAMPLING_PROTOCOL_ID
    return build_scoreboard(
        run_id="sampling",
        suite="core",
        config=config,
        environment={},
        pairs=[pair],
        targets=[target.label()],
        target_configs=[target],
    )


def test_seed_schedule_preserves_single_attempt_omission_and_sweeps_from_base():
    assert seed_schedule(None, 1) == (None,)
    assert seed_schedule(7, 1) == (7,)
    assert seed_schedule(None, 4) == (0, 1, 2, 3)
    assert seed_schedule(7, 4) == (7, 8, 9, 10)


@pytest.mark.parametrize(
    ("successes", "k", "expected"),
    [(1, 1, 0.25), (1, 2, 0.5), (1, 4, 1.0)],
)
def test_pass_at_k_uses_the_unbiased_without_replacement_estimator(
    successes, k, expected
):
    assert pass_at_k(successes, 4, k) == expected


async def test_recommended_sampling_omits_defaults_and_explicit_temperature_wins():
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await call_chat(
            client,
            BenchTarget(
                base_url="http://gateway",
                model="m",
                sampling_mode="recommended",
            ),
            ChatRequestSpec(messages=({"role": "user", "content": "q"},)),
            retries=0,
            timeout_s=1,
        )
        await call_chat(
            client,
            BenchTarget(base_url="http://gateway", model="m", temperature=0.7),
            ChatRequestSpec(
                messages=({"role": "user", "content": "q"},),
                temperature=0.0,
            ),
            retries=0,
            timeout_s=1,
        )

    assert "temperature" not in bodies[0]
    assert "top_p" not in bodies[0]
    assert bodies[1]["temperature"] == 0.7


async def test_seed_override_is_stable_across_request_retries():
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(500, text="retry")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await call_chat(
            client,
            BenchTarget(base_url="http://gateway", model="m", seed=99),
            ChatRequestSpec(messages=({"role": "user", "content": "q"},)),
            retries=1,
            timeout_s=1,
            sampling_seed=7,
        )

    assert [body["seed"] for body in bodies] == [7, 7]


async def test_chat_attempts_group_evidence_and_report_seed_spread_and_pass_at_k(
    tmp_path,
):
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        item_id = body["messages"][0]["content"]
        passed = item_id == "b" or body["seed"] == 10
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "pass" if passed else "fail"}}]},
        )

    target = BenchTarget(
        base_url="http://gateway",
        model="m",
        seed=10,
        sampling_mode="recommended",
    )
    pair = await _BinaryAdapter().run(target, _context(tmp_path, handler))

    assert pair.status == "completed"
    assert pair.score == 0.625
    assert len(pair.items) == 2
    assert all(item.sampling_attempts is not None for item in pair.items)
    assert [attempt.seed for attempt in pair.items[0].sampling_attempts or ()] == [
        10,
        11,
        12,
        13,
    ]
    assert len(bodies) == 8
    assert sorted({body["seed"] for body in bodies}) == [10, 11, 12, 13]
    assert all("temperature" not in body for body in bodies)
    assert pair.metrics["sampling_seed_score_mean"] == 0.625
    assert pair.metrics["sampling_seed_score_sample_sd"] == 0.25
    assert pair.metrics["sampling_seed_score_min"] == 0.5
    assert pair.metrics["sampling_seed_score_max"] == 1.0
    assert pair.metrics["pass_at_1"] == 0.625
    assert pair.metrics["pass_at_2"] == 0.75
    assert pair.metrics["pass_at_4"] == 1.0
    assert PairResult.model_validate_json(pair.model_dump_json()) == pair
    expected_sampling = {
        "attempts": 4,
        "seed": 10,
        "mode": "recommended",
        "temperature": None,
        "top_p": None,
        "required": True,
    }
    assert pair_result_evidence_error(pair, expected_sampling=expected_sampling) is None
    unsafe_schedule_error = pair_result_evidence_error(
        pair,
        expected_sampling={
            **expected_sampling,
            "attempts": 2,
            "seed": (1 << 53) - 1,
        },
    )
    assert unsafe_schedule_error is not None
    assert "seed schedule exceeds JSON-safe integers" in unsafe_schedule_error
    assert "attempt budget" in (
        pair_result_evidence_error(
            pair,
            expected_sampling={**expected_sampling, "attempts": 2},
        )
        or ""
    )
    assert "seeds disagree" in (
        pair_result_evidence_error(
            pair,
            expected_sampling={**expected_sampling, "seed": 99},
        )
        or ""
    )
    assert "mode disagrees" in (
        pair_result_evidence_error(
            pair,
            expected_sampling={**expected_sampling, "mode": "adapter"},
        )
        or ""
    )
    mismatched_target = target.model_copy(update={"seed": 99})
    mismatched_cell = _scoreboard(pair, mismatched_target)["cells"]["gsm8k"][
        target.label()
    ]
    assert mismatched_cell["status"] == "failed"
    assert "sampling_sensitivity" not in mismatched_cell
    assert "seeds disagree" in (mismatched_cell["reason"] or "")
    assert "sampling_attempts" not in ItemResult(
        item_id="legacy", status="completed", score=1.0
    ).model_dump()

    board = _scoreboard(pair, target)
    cell = board["cells"]["gsm8k"][target.label()]
    sensitivity = cell["sampling_sensitivity"]
    assert sensitivity["seed_scores"] == [
        {"seed": 10, "score": 1.0},
        {"seed": 11, "score": 0.5},
        {"seed": 12, "score": 0.5},
        {"seed": 13, "score": 0.5},
    ]
    assert sensitivity["item_successes"] == [1, 4]
    assert cell["confidence_interval"] is None
    markdown = render_markdown(board)
    assert "62.5 ± 25.0 SD [50.0–100.0]" in markdown
    assert "pass@2 75.0" in markdown
    assert "not remotely attested" in markdown


async def test_incomplete_seed_matrix_withholds_sampling_statistics(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = body["messages"][0]["content"]
        if content == "a" and body["seed"] == 2:
            content = ""
        else:
            content = "pass"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    target = BenchTarget(
        base_url="http://gateway",
        model="m",
        sampling_mode="recommended",
    )
    pair = await _BinaryAdapter().run(target, _context(tmp_path, handler))
    assert pair.status == "partial"
    assert pair_result_evidence_error(pair) is None
    assert pair.metrics["sampling_complete"] == 0
    assert "sampling_seed_score_mean" not in pair.metrics

    board = _scoreboard(pair, target)
    cell = board["cells"]["gsm8k"][target.label()]
    assert "sampling_sensitivity" not in cell
    assert cell["confidence_interval"] is None
    assert any(
        "sampling sensitivity withheld" in board["footnotes"][number - 1]
        for number in cell["footnotes"]
    )

    forged_status = pair.model_copy(update={"status": "completed", "reason": None})
    assert "pair status" in (pair_result_evidence_error(forged_status) or "")


def test_sampling_parent_status_is_derived_from_child_outcomes():
    from evals.types import SamplingAttemptResult

    complete = SamplingAttemptResult(
        attempt=1,
        seed=0,
        result=ItemResult(item_id="x", status="completed", score=1.0),
    )
    failed = SamplingAttemptResult(
        attempt=2,
        seed=1,
        result=ItemResult(item_id="x", status="failed", error="boom"),
    )
    with pytest.raises(ValueError, match="incomplete attempt outcomes"):
        ItemResult(
            item_id="x",
            status="skipped",
            sampling_attempts=(complete, failed),
        )

    second_complete = SamplingAttemptResult(
        attempt=2,
        seed=1,
        result=ItemResult(item_id="x", status="completed", score=0.0),
    )
    with pytest.raises(ValueError, match="require a completed parent"):
        ItemResult(
            item_id="x",
            status="skipped",
            sampling_attempts=(complete, second_complete),
        )


async def test_malformed_stored_sampling_summary_never_breaks_markdown(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"choices": [{"message": {"content": "pass"}}]})

    target = BenchTarget(base_url="http://gateway", model="m", temperature=0.6)
    pair = await _BinaryAdapter().run(
        target,
        _context(tmp_path, handler, attempts=2),
    )
    board = _scoreboard(pair, target)
    board["cells"]["gsm8k"][target.label()]["sampling_sensitivity"][
        "sample_sd"
    ] = "forged"

    markdown = render_markdown(board)
    assert "100.0 (n=2)" in markdown
    assert "Sampling sweeps show" not in markdown


async def test_stored_sampling_summary_is_bound_to_binary_contract_and_counts(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"choices": [{"message": {"content": "pass"}}]})

    target = BenchTarget(base_url="http://gateway", model="m")
    pair = await _BinaryAdapter().run(target, _context(tmp_path, handler, attempts=2))
    board = _scoreboard(pair, target)
    cell = board["cells"]["gsm8k"][target.label()]

    cell["n"] = 999
    assert "Sampling sweeps show" not in render_markdown(board)
    cell["n"] = 2

    board["benchmarks"] = ["custom-score"]
    board["display_names"] = {"custom-score": "Custom Score"}
    board["cells"] = {"custom-score": board["cells"].pop("gsm8k")}
    markdown = render_markdown(board)
    assert "Sampling sweeps show" not in markdown
    assert "pass@" not in markdown


def test_stored_binary_summary_rejects_impossible_item_seed_margins():
    target = BenchTarget(base_url="http://gateway", model="m")
    pair = PairResult(
        benchmark="gsm8k",
        target=target.label(),
        status="completed",
        metrics={"score": 0.5, "n_total": 2, "n_scored": 2},
    )
    board = _scoreboard(pair, target)
    board["cells"]["gsm8k"][target.label()]["sampling_sensitivity"] = {
        "attempts": 2,
        "seeds": [0, 1],
        "n_items": 2,
        "seed_scores": [{"seed": 0, "score": 1.0}, {"seed": 1, "score": 0.0}],
        "mean": 0.5,
        "sample_sd": 2**-0.5,
        "min": 0.0,
        "max": 1.0,
        # These row sums and the [2, 0] column sums have the same total but
        # cannot belong to one binary item-by-seed matrix.
        "item_successes": [2, 0],
        "pass_at_k": [{"k": 1, "estimate": 0.5}, {"k": 2, "estimate": 0.5}],
        "mode": "adapter",
        "wire_omitted": [],
        "recommended_defaults_attested": False,
    }

    markdown = render_markdown(board)
    assert "Sampling sweeps show" not in markdown
    assert "pass@" not in markdown


@pytest.mark.parametrize(
    "mutation",
    [
        lambda summary: summary["item_successes"].__setitem__(0, 2),
        lambda summary: summary["pass_at_k"][1].__setitem__("estimate", 0.125),
        lambda summary: summary["pass_at_k"].pop(),
    ],
)
async def test_stored_pass_at_k_must_recompute_from_item_successes(
    tmp_path, mutation
):
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"choices": [{"message": {"content": "pass"}}]})

    target = BenchTarget(base_url="http://gateway", model="m")
    pair = await _BinaryAdapter().run(target, _context(tmp_path, handler))
    board = _scoreboard(pair, target)
    summary = board["cells"]["gsm8k"][target.label()]["sampling_sensitivity"]
    mutation(summary)

    markdown = render_markdown(board)
    assert "Sampling sweeps show" not in markdown
    assert "pass@" not in markdown


async def test_grouped_pair_counts_and_source_ids_are_bound_to_raw_items(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"choices": [{"message": {"content": "pass"}}]})

    target = BenchTarget(base_url="http://gateway", model="m")
    pair = await _BinaryAdapter().run(target, _context(tmp_path, handler))

    forged_counts = pair.model_copy(update={"metrics": {**pair.metrics, "n_total": 999}})
    assert "metrics.n_total" in (pair_result_evidence_error(forged_counts) or "")
    assert "sampling_sensitivity" not in _scoreboard(forged_counts, target)["cells"][
        "gsm8k"
    ][target.label()]

    duplicate = pair.model_copy(update={"items": (pair.items[0], pair.items[0])})
    assert "item ids must be unique" in (pair_result_evidence_error(duplicate) or "")


def test_sampling_summary_rejects_nonbinary_evidence_for_binary_adapter():
    child = ItemResult(item_id="x", status="completed", score=0.5)
    from evals.sampling import combine_attempts
    from evals.types import SamplingAttemptResult

    parent = combine_attempts(
        "x",
        (
            SamplingAttemptResult(attempt=1, seed=0, result=child),
            SamplingAttemptResult(attempt=2, seed=1, result=child),
        ),
    )
    with pytest.raises(ValueError, match="non-binary"):
        sampling_summary([parent], seeds=[0, 1], declared_binary=True)


def test_sampling_helpers_reject_boolean_seed_and_binary_contract():
    with pytest.raises(ValueError, match="target seed must be an integer"):
        seed_schedule(True, 2)
    with pytest.raises(ValueError, match="declared_binary must be a boolean"):
        sampling_summary([], seeds=[0, 1], declared_binary=1)
    with pytest.raises(ValueError, match="seeds must be integers"):
        sampling_summary([], seeds=[0, True], declared_binary=False)


def test_sampling_helpers_reject_json_unsafe_seed_schedule():
    maximum = (1 << 53) - 1
    with pytest.raises(ValueError, match="seed schedule.*JSON-safe integers"):
        seed_schedule(maximum, 2)
    with pytest.raises(ValueError, match="seeds must be JSON-safe integers"):
        sampling_summary(
            [],
            seeds=[0, maximum + 1],
            declared_binary=False,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("attempt", True, "sampling attempt must be an integer"),
        ("seed", "0", "sampling seed must be an integer"),
        ("seed", 1 << 53, "sampling seed must be a JSON-safe integer"),
    ],
)
def test_sampling_attempt_identity_rejects_lossy_integer_coercion(field, value, message):
    from evals.types import SamplingAttemptResult

    data = {
        "attempt": 1,
        "seed": 0,
        "result": ItemResult(item_id="x", status="completed", score=1.0),
    }
    data[field] = value
    with pytest.raises(ValueError, match=message):
        SamplingAttemptResult(**data)
