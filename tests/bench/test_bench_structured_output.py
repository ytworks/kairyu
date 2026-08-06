"""Paired JSON-Schema conformance is wire-identical and evidence-replayable."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import httpx
import pytest

from kairyu.bench import history
from kairyu.bench.adapters.base import DownloadContext, RunContext, cache_pins
from kairyu.bench.adapters.structured_output import StructuredOutputAdapter
from kairyu.bench.aggregate import build_scoreboard, render_markdown
from kairyu.bench.cache import BenchCache
from kairyu.bench.runner import (
    _adapter_identity,
    _recordable_config,
    _run_fingerprint,
    _run_identity,
)
from kairyu.bench.structured import (
    CATEGORY_ORDER,
    CORPUS_ID,
    CORPUS_REVISION,
    StrictJsonError,
    arm_order,
    evaluate_arm,
    load_packaged_corpus,
    response_format,
    strict_json_loads,
    structured_pair_evidence_error,
    structured_run_expectation,
    structured_summary,
    structured_summary_error,
    validate_corpus_rows,
)
from kairyu.bench.types import (
    JSON_SAFE_INTEGER_MAX,
    BenchConfig,
    BenchTarget,
    PairResult,
    StructuredArmEvidence,
    pair_result_evidence_error,
)


def _context(
    tmp_path,
    handler,
    *,
    attempts: int = 1,
    limit: int | None = None,
    offline_fixtures: bool = True,
) -> RunContext:
    return RunContext(
        cache=BenchCache(tmp_path / "cache"),
        http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        attempts=attempts,
        limit=limit,
        concurrency=16,
        retries=0,
        request_timeout_s=1,
        offline_fixtures=offline_fixtures,
    )


def _body_row(body: dict, rows: list[dict]) -> dict:
    prompt = body["messages"][0]["content"]
    matching = [row for row in rows if prompt.startswith(row["prompt"])]
    assert len(matching) == 1
    return matching[0]


def _completion(
    content: str,
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 4,
    include_usage: bool = True,
) -> httpx.Response:
    payload = {
        "choices": [
            {
                "index": 0,
                # Some compatible servers serialize an empty tool_calls list
                # on ordinary content responses; it is not a call payload.
                "message": {"content": content, "tool_calls": []},
                "finish_reason": "stop",
            }
        ]
    }
    if include_usage:
        payload["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            # Real OpenAI-compatible endpoints may add detail objects. They do
            # not alter the three retained counts.
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
    return httpx.Response(200, json=payload)


def _refusal_completion(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {"content": None, "refusal": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        },
    )


def _non_text_completion(message: dict, finish_reason: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 0,
                "total_tokens": 10,
            },
        },
    )


def _schema_rejection(status_code: int = 422) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={
            "error": {
                "message": "unsupported response_format json_schema",
                "type": "invalid_request_error",
                "code": "invalid_json_schema",
            }
        },
    )


def _expected_sampling(target: BenchTarget, attempts: int) -> dict[str, object]:
    return {
        "attempts": attempts,
        "seed": target.seed,
        "mode": target.sampling_mode,
        "temperature": target.temperature,
        "top_p": target.top_p,
        "required": True,
    }


def _completed_arm(
    arm: str,
    content: str,
    *,
    latency_s: float = 0.1,
) -> StructuredArmEvidence:
    return StructuredArmEvidence(
        arm=arm,
        outcome="completed",
        http_status=200,
        content=content,
        finish_reason="stop",
        latency_s=latency_s,
    )


def test_fixed_corpus_covers_each_required_schema_class_and_rejects_drift():
    rows = load_packaged_corpus()

    assert [row["category"] for row in rows] == list(CATEGORY_ORDER)
    assert [row["id"] for row in rows] == [
        "nested-object",
        "recursive-tree",
        "enum-signal",
        "pattern-code",
        "union-value",
    ]
    assert CORPUS_ID == "package:kairyu.bench.fixtures/structured-output.jsonl"
    assert CORPUS_REVISION.startswith("sha256:")
    assert len(CORPUS_REVISION.removeprefix("sha256:")) == 64
    assert rows[1]["schema"]["$defs"]["node"]["properties"]["children"]["items"] == {
        "$ref": "#/$defs/node"
    }
    assert rows[2]["schema"]["properties"]["signal"]["enum"] == [
        "red",
        "yellow",
        "green",
    ]
    assert rows[3]["schema"]["properties"]["code"]["pattern"] == "^KR-[0-9]{4}$"
    assert "anyOf" in rows[4]["schema"]["properties"]["value"]

    for row in rows:
        result = evaluate_arm(
            _completed_arm("constrained", json.dumps(row["expected"])),
            row,
        )
        assert result == {
            "accepted": True,
            "schema_rejected": False,
            "json_valid": True,
            "schema_valid": True,
            "task_correct": True,
            "malformed_json": False,
        }

    external = deepcopy(rows)
    external[1]["schema"]["$defs"]["node"]["properties"]["children"]["items"][
        "$ref"
    ] = "https://schemas.example/node.json"
    with pytest.raises(ValueError, match=r"external \$ref"):
        validate_corpus_rows(external)

    wrong_expected = deepcopy(rows)
    wrong_expected[4]["expected"] = {"value": True}
    with pytest.raises(ValueError, match="expected answer violates"):
        validate_corpus_rows(wrong_expected)


def test_fixed_cache_comparison_is_type_sensitive(tmp_path):
    adapter = StructuredOutputAdapter()
    cache = BenchCache(tmp_path / "cache")
    assert adapter.download(DownloadContext(cache=cache)).status == "ok"
    rows = cache.read_rows(adapter.info.name)
    scores = rows[0]["schema"]["properties"]["user"]["properties"]["scores"]
    scores["minItems"] = 2.0
    cache.write_rows(adapter.info.name, rows, cache_pins(adapter.info))

    assert adapter.additional_cache_ready(cache) is False
    with pytest.raises(ValueError, match="differs from the fixed packaged corpus"):
        adapter.load_items(
            _context(
                tmp_path,
                lambda request: _completion("{}"),
                offline_fixtures=False,
            )
        )


def test_every_corpus_schema_compiles_and_enforces_keywords_with_xgrammar():
    import xgrammar

    rows = load_packaged_corpus()
    tokenizer = xgrammar.TokenizerInfo([bytes([value]) for value in range(256)], vocab_size=256)
    compiler = xgrammar.GrammarCompiler(tokenizer)
    invalid = {
        "nested-object": {"user": {"name": "Ada"}},
        "recursive-tree": {"label": "root", "children": [{"label": 1, "children": []}]},
        "enum-signal": {"signal": "blue"},
        "pattern-code": {"code": "XX-9999"},
        "union-value": {"value": True},
    }
    for row in rows:
        compiled = compiler.compile_json_schema(
            json.dumps(row["schema"]),
            any_whitespace=False,
        )
        accepted = xgrammar.GrammarMatcher(compiled)
        rejected = xgrammar.GrammarMatcher(compiled)
        assert accepted.accept_string(json.dumps(row["expected"])) is True
        assert rejected.accept_string(json.dumps(invalid[row["id"]])) is False


@pytest.mark.parametrize(
    "text",
    [
        '{"value": 1, "value": 2}',
        '{"nested": {"x": 1, "x": 2}}',
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": -Infinity}',
        '{"value": 1e999}',
        '{"value": 1} trailing',
    ],
)
def test_strict_json_parser_rejects_ambiguous_or_non_rfc_payloads(text):
    with pytest.raises(StrictJsonError):
        strict_json_loads(text)


def test_strict_json_parser_contains_integer_and_nesting_resource_failures():
    with pytest.raises(StrictJsonError):
        strict_json_loads("1" * 5_000)
    with pytest.raises(StrictJsonError):
        strict_json_loads("[" * 2_000 + "]" * 2_000)


def test_task_oracle_is_schema_checked_and_type_sensitive():
    row = load_packaged_corpus()[-1]

    correct = evaluate_arm(_completed_arm("constrained", ' {"value": 42}\n'), row)
    schema_valid_but_wrong = evaluate_arm(
        _completed_arm("constrained", '{"value": "fallback"}'), row
    )
    boolean_is_not_integer = evaluate_arm(
        _completed_arm("constrained", '{"value": true}'), row
    )
    duplicate_key = evaluate_arm(
        _completed_arm("constrained", '{"value": 42, "value": 41}'), row
    )

    assert correct["task_correct"] is True
    assert schema_valid_but_wrong["schema_valid"] is True
    assert schema_valid_but_wrong["task_correct"] is False
    assert boolean_is_not_integer["schema_valid"] is False
    assert duplicate_key["json_valid"] is False
    assert duplicate_key["malformed_json"] is True


@pytest.mark.parametrize(
    "auxiliary",
    ['{"tool_calls":[]}', '{"function_call":{}}'],
)
def test_auxiliary_call_evidence_requires_a_meaningful_nonempty_payload(auxiliary):
    with pytest.raises(ValueError, match="auxiliary message payload"):
        StructuredArmEvidence(
            arm="constrained",
            outcome="completed",
            http_status=200,
            content=None,
            message_auxiliary=auxiliary,
            finish_reason="tool_calls",
            latency_s=0.1,
        )


def test_structured_suite_rejects_target_response_format_without_breaking_other_suites(
    tmp_path,
):
    target = BenchTarget(
        base_url="http://gateway/v1",
        model="model",
        extra_body_json='{"response_format":{"type":"text"}}',
    )
    reason = StructuredOutputAdapter().check_preconditions(
        target,
        _context(tmp_path, lambda request: _completion("{}")),
    )

    assert "rejects extra_body_json" in (reason or "")


def test_structured_pairing_rejects_hidden_control_constraints_and_unseeded_randomness(
    tmp_path,
):
    adapter = StructuredOutputAdapter()
    ctx = _context(tmp_path, lambda request: _completion("{}"))
    hidden_constraint = BenchTarget(
        base_url="http://gateway/v1",
        model="model",
        extra_body_json='{"guided_json":{"type":"object"}}',
    )
    assert "could constrain the control arm" in (
        adapter.check_preconditions(hidden_constraint, ctx) or ""
    )
    unseeded = BenchTarget(
        base_url="http://gateway/v1",
        model="model",
        sampling_mode="recommended",
    )
    assert "requires target.seed" in (adapter.check_preconditions(unseeded, ctx) or "")


async def test_run_config_binds_exact_subset_and_rejected_preconditions(tmp_path):
    rows = load_packaged_corpus()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        row = _body_row(body, rows)
        return _completion(json.dumps(row["expected"], separators=(",", ":")))

    adapter = StructuredOutputAdapter()
    target = BenchTarget(base_url="http://gateway/v1", model="model")
    pair = await adapter.run(target, _context(tmp_path, handler, limit=1))
    expected_subset = structured_run_expectation(
        limit=1,
        smoke=False,
        selection_seed=0,
        extra_body_present=False,
    )

    assert "may select a subset" in pair.methodology["split"]
    assert (
        pair_result_evidence_error(
            pair,
            expected_sampling=_expected_sampling(target, 1),
            expected_structured_run=expected_subset,
        )
        is None
    )
    full_run = structured_run_expectation(
        limit=None,
        smoke=False,
        selection_seed=0,
        extra_body_present=False,
    )
    assert "selected item ids disagree" in (
        pair_result_evidence_error(
            pair,
            expected_sampling=_expected_sampling(target, 1),
            expected_structured_run=full_run,
        )
        or ""
    )
    forbidden_body = {**expected_subset, "extra_body_present": True}
    assert "rejected extra_body_json" in (
        pair_result_evidence_error(
            pair,
            expected_sampling=_expected_sampling(target, 1),
            expected_structured_run=forbidden_body,
        )
        or ""
    )
    durable_config = BenchConfig(
        suite="structured",
        targets=(target,),
        limit=1,
        progress=False,
    ).model_dump(mode="json")
    durable_config["targets"][0]["extra_body_json"] = (
        "sha256:" + hashlib.sha256(b"opaque-extra-body").hexdigest()
    )
    rebuilt = build_scoreboard(
        run_id="structured-report",
        suite="structured",
        config=durable_config,
        environment={},
        pairs=[pair],
        targets=[target.label()],
        target_configs=[target],
    )
    rebuilt_cell = rebuilt["cells"]["structured-output"][target.label()]
    assert rebuilt_cell["status"] == "failed"
    assert "rejected extra_body_json" in rebuilt_cell["reason"]
    assert "structured_conformance" not in rebuilt_cell

    unseeded = BenchTarget(
        base_url="http://gateway/v1",
        model="model",
        sampling_mode="recommended",
    )
    skipped = await adapter.run(unseeded, _context(tmp_path, handler, limit=1))
    assert skipped.status == "skipped"
    assert (
        pair_result_evidence_error(
            skipped,
            expected_sampling=_expected_sampling(unseeded, 1),
            expected_structured_run=expected_subset,
        )
        is None
    )
    forged_methodology = {
        **pair.methodology,
        "temperature": None,
        "sampling": {
            "mode": "recommended",
            "attempts": 1,
            "seeds": [None],
            "seed_source": "wire omission",
            "wire_omitted": [
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "repetition_penalty",
            ],
            "recommended_defaults_attested": False,
        },
    }
    forged_pair = pair.model_copy(update={"methodology": forged_methodology})
    assert "requires target.seed" in (
        pair_result_evidence_error(
            forged_pair,
            expected_sampling=_expected_sampling(unseeded, 1),
            expected_structured_run=expected_subset,
        )
        or ""
    )


async def test_paired_run_changes_only_response_format_and_retains_usage(tmp_path):
    rows = load_packaged_corpus()
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        row = _body_row(body, rows)
        constrained = "response_format" in body
        return _completion(
            json.dumps(row["expected"], separators=(",", ":")),
            prompt_tokens=20,
            completion_tokens=4 if constrained else 6,
        )

    target = BenchTarget(
        base_url="http://gateway/v1",
        model="model",
        seed=11,
        temperature=0.2,
        top_p=0.9,
        reasoning_effort="high",
    )
    pair = await StructuredOutputAdapter().run(target, _context(tmp_path, handler))

    assert pair.status == "completed"
    assert pair.score == 1.0
    assert pair.metrics["structured_complete"] == 1
    assert pair.metrics["structured_observations"] == len(rows)
    assert pair.metrics["structured_constrained_task_accuracy"] == 1.0
    assert pair.metrics["structured_unconstrained_task_accuracy"] == 1.0
    assert pair.metrics["structured_constrained_mean_total_tokens"] == 24
    assert pair.metrics["structured_unconstrained_mean_total_tokens"] == 26
    assert pair.metrics["structured_paired_cost_mean_prompt_token_delta"] == 0
    assert pair.metrics["structured_paired_cost_mean_completion_token_delta"] == -2
    assert pair.metrics["structured_paired_cost_mean_total_token_delta"] == -2
    assert "structured_paired_cost_currency_cost" not in pair.metrics
    assert len(bodies) == len(rows) * 2

    for source_index, row in enumerate(rows):
        matching = [body for body in bodies if _body_row(body, rows)["id"] == row["id"]]
        assert len(matching) == 2
        constrained = next(body for body in matching if "response_format" in body)
        unconstrained = next(body for body in matching if "response_format" not in body)
        assert {key: value for key, value in constrained.items() if key != "response_format"} == (
            unconstrained
        )
        assert constrained["response_format"] == response_format(row)
        assert constrained["seed"] == unconstrained["seed"] == 11
        assert constrained["temperature"] == unconstrained["temperature"] == 0.2
        assert constrained["top_p"] == unconstrained["top_p"] == 0.9
        assert constrained["reasoning_effort"] == unconstrained["reasoning_effort"] == "high"
        evidence = pair.items[source_index].structured_output
        assert evidence is not None
        assert evidence.order == arm_order(source_index)
        assert evidence.constrained.usage is not None
        assert evidence.unconstrained.usage is not None

    assert pair_result_evidence_error(
        pair,
        expected_sampling=_expected_sampling(target, 1),
    ) is None
    assert PairResult.model_validate_json(pair.model_dump_json()) == pair


async def test_summary_uses_declared_denominators_and_no_fake_currency_cost(tmp_path):
    rows = load_packaged_corpus()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        row = _body_row(body, rows)
        constrained = "response_format" in body
        if constrained and row["category"] == "union":
            return _schema_rejection()
        if constrained and row["category"] == "recursive":
            content = "not json"
        elif constrained and row["category"] == "enum":
            content = '{"signal":"blue"}'
        elif constrained and row["category"] == "pattern":
            content = '{"code":"KR-9999"}'
        else:
            content = json.dumps(row["expected"], separators=(",", ":"))
        return _completion(
            content,
            prompt_tokens=10 if constrained else 8,
            completion_tokens=4 if constrained else 6,
        )

    target = BenchTarget(base_url="http://gateway/v1", model="model")
    pair = await StructuredOutputAdapter().run(target, _context(tmp_path, handler))
    summary = structured_summary(pair)

    assert pair.status == "completed"
    assert pair.score == 0.2
    assert summary is not None
    assert summary["categories"] == list(CATEGORY_ORDER)
    assert summary["n_source_items"] == 5
    assert summary["attempts"] == 1
    assert summary["observations"] == 5
    assert summary["arms"]["constrained"] == {
        "observations": 5,
        "accepted": 4,
        "schema_rejected": 1,
        "json_valid": 3,
        "schema_valid": 2,
        "task_correct": 1,
        "malformed_json": 1,
        "acceptance_rate": 0.8,
        "json_valid_rate": 0.75,
        "schema_conformance_rate": 0.4,
        "task_accuracy": 0.2,
        "malformed_json_rate": 0.25,
        "usage_observations": 4,
        "usage_coverage": 1.0,
        "prompt_tokens_total": 40,
        "completion_tokens_total": 16,
        "total_tokens_total": 56,
        "mean_prompt_tokens": 10,
        "mean_completion_tokens": 4,
        "mean_total_tokens": 14,
        "mean_latency_s": summary["arms"]["constrained"]["mean_latency_s"],
    }
    assert summary["arms"]["unconstrained"]["accepted"] == 5
    assert summary["arms"]["unconstrained"]["task_correct"] == 5
    assert summary["arms"]["unconstrained"]["task_accuracy"] == 1.0
    assert summary["arms"]["unconstrained"]["malformed_json_rate"] == 0.0
    assert summary["quality_deltas"] == {
        "schema_conformance_rate": pytest.approx(-0.6),
        "task_accuracy": -0.8,
    }
    assert summary["paired_cost"] == {
        "usage_observations": 4,
        "usage_coverage": 0.8,
        "mean_prompt_token_delta": 2,
        "mean_completion_token_delta": -2,
        "mean_total_token_delta": 0,
        "currency_cost": None,
    }
    assert summary["latency_diagnostic"]["observations"] == 5
    assert "dimensions disagree" in (
        structured_summary_error(
            summary,
            expected_score=pair.score,
            expected_total=5.0,
            expected_status=pair.status,
        )
        or ""
    )
    float_latency_count = deepcopy(summary)
    float_latency_count["latency_diagnostic"]["observations"] = 5.0
    assert "latency diagnostic is invalid" in (
        structured_summary_error(
            float_latency_count,
            expected_score=pair.score,
            expected_total=5,
            expected_status=pair.status,
        )
        or ""
    )
    assert pair_result_evidence_error(pair) is None


@pytest.mark.parametrize(
    ("failed_arm", "status_code", "body", "expected_outcome"),
    [
        (
            "constrained",
            400,
            {
                "error": {
                    "message": "response_format json_schema is unsupported",
                    "type": "invalid_request_error",
                    "code": "unsupported_schema",
                }
            },
            "schema_rejected",
        ),
        (
            "constrained",
            400,
            {"error": {"message": "context is too long", "type": "invalid_request_error"}},
            "failed",
        ),
        (
            "constrained",
            422,
            (
                '{"error":{"message":"response_format json_schema is unsupported",'
                '"type":"invalid_request_error"},"overflow":1e999}'
            ),
            "failed",
        ),
        (
            "constrained",
            401,
            {"error": {"message": "invalid credential", "type": "authentication_error"}},
            "failed",
        ),
        (
            "unconstrained",
            422,
            {
                "error": {
                    "message": "response_format json_schema is unsupported",
                    "type": "invalid_request_error",
                }
            },
            "failed",
        ),
        (
            "unconstrained",
            503,
            {"error": {"message": "temporarily unavailable", "type": "server_error"}},
            "failed",
        ),
    ],
)
async def test_only_constrained_schema_rejection_is_a_scored_observation(
    tmp_path,
    failed_arm,
    status_code,
    body,
    expected_outcome,
):
    rows = load_packaged_corpus()

    def handler(request: httpx.Request) -> httpx.Response:
        request_body = json.loads(request.content)
        row = _body_row(request_body, rows)
        arm = "constrained" if "response_format" in request_body else "unconstrained"
        if arm == failed_arm:
            return (
                httpx.Response(status_code, content=body.encode())
                if isinstance(body, str)
                else httpx.Response(status_code, json=body)
            )
        return _completion(json.dumps(row["expected"], separators=(",", ":")))

    pair = await StructuredOutputAdapter().run(
        BenchTarget(base_url="http://gateway/v1", model="model"),
        _context(tmp_path, handler, limit=1),
    )
    item = pair.items[0]
    evidence = item.structured_output
    assert evidence is not None
    affected = getattr(evidence, failed_arm)

    assert affected.outcome == expected_outcome
    assert affected.http_status == status_code
    if expected_outcome == "schema_rejected":
        assert item.status == "completed"
        assert item.score == 0.0
        assert pair.status == "completed"
        assert pair.metrics["structured_complete"] == 1
        assert pair.metrics["structured_constrained_schema_rejected"] == 1
    else:
        assert item.status == "failed"
        assert item.score is None
        assert pair.status == "failed"
        assert pair.metrics["structured_complete"] == 0
        assert pair.metrics["structured_observations"] == 1
        assert structured_summary(pair) is None
    assert pair_result_evidence_error(pair) is None


async def test_missing_usage_is_measured_as_zero_coverage_not_fabricated_cost(tmp_path):
    rows = load_packaged_corpus()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        row = _body_row(body, rows)
        return _completion(
            json.dumps(row["expected"], separators=(",", ":")),
            include_usage=False,
        )

    pair = await StructuredOutputAdapter().run(
        BenchTarget(base_url="http://gateway/v1", model="model"),
        _context(tmp_path, handler, limit=1),
    )
    summary = structured_summary(pair)

    assert pair.status == "completed"
    assert summary is not None
    for arm in ("constrained", "unconstrained"):
        arm_summary = summary["arms"][arm]
        assert arm_summary["usage_observations"] == 0
        assert arm_summary["usage_coverage"] == 0.0
        assert arm_summary["mean_prompt_tokens"] is None
        assert arm_summary["mean_completion_tokens"] is None
        assert arm_summary["mean_total_tokens"] is None
    assert summary["paired_cost"]["usage_observations"] == 0
    assert summary["paired_cost"]["usage_coverage"] == 0.0
    assert summary["paired_cost"]["mean_total_token_delta"] is None
    assert summary["paired_cost"]["currency_cost"] is None
    assert pair_result_evidence_error(pair) is None


async def test_http_200_refusal_is_retained_as_accepted_non_json_quality_failure(
    tmp_path,
):
    rows = load_packaged_corpus()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        row = _body_row(body, rows)
        if "response_format" in body:
            return _refusal_completion("I cannot provide that response.")
        return _completion(json.dumps(row["expected"], separators=(",", ":")))

    pair = await StructuredOutputAdapter().run(
        BenchTarget(base_url="http://gateway/v1", model="model"),
        _context(tmp_path, handler, limit=1),
    )
    evidence = pair.items[0].structured_output
    summary = structured_summary(pair)

    assert evidence is not None
    assert evidence.constrained.content is None
    assert evidence.constrained.refusal == "I cannot provide that response."
    assert evidence.constrained.outcome == "completed"
    assert pair.status == "completed"
    assert pair.score == 0.0
    assert summary is not None
    assert summary["arms"]["constrained"]["accepted"] == 1
    assert summary["arms"]["constrained"]["malformed_json"] == 1
    assert summary["arms"]["constrained"]["task_correct"] == 0
    assert pair_result_evidence_error(pair) is None


@pytest.mark.parametrize(
    ("message", "finish_reason", "expected_auxiliary"),
    [
        ({"content": None, "refusal": None}, "content_filter", None),
        (
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "unexpected", "arguments": "{}"},
                    }
                ],
            },
            "tool_calls",
            (
                '{"tool_calls":[{"function":{"arguments":"{}",'
                '"name":"unexpected"},"id":"call-1","type":"function"}]}'
            ),
        ),
    ],
)
async def test_other_valid_non_text_completions_are_quality_failures(
    tmp_path,
    message,
    finish_reason,
    expected_auxiliary,
):
    rows = load_packaged_corpus()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        row = _body_row(body, rows)
        if "response_format" in body:
            return _non_text_completion(message, finish_reason)
        return _completion(json.dumps(row["expected"], separators=(",", ":")))

    pair = await StructuredOutputAdapter().run(
        BenchTarget(base_url="http://gateway/v1", model="model"),
        _context(tmp_path, handler, limit=1),
    )
    evidence = pair.items[0].structured_output
    summary = structured_summary(pair)

    assert evidence is not None
    assert evidence.constrained.outcome == "completed"
    assert evidence.constrained.message_auxiliary == expected_auxiliary
    assert pair.status == "completed"
    assert pair.score == 0.0
    assert summary is not None
    assert summary["arms"]["constrained"]["malformed_json"] == 1
    assert pair_result_evidence_error(pair) is None


def _invalid_completion(kind: str, content: str) -> httpx.Response:
    choice = {
        "index": 0,
        "message": {"content": content},
        "finish_reason": "stop",
    }
    if kind == "usage-total":
        return httpx.Response(
            200,
            json={
                "choices": [choice],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 99,
                },
            },
        )
    if kind == "usage-boolean":
        return httpx.Response(
            200,
            json={
                "choices": [choice],
                "usage": {
                    "prompt_tokens": True,
                    "completion_tokens": 3,
                    "total_tokens": 4,
                },
            },
        )
    if kind == "usage-list":
        return httpx.Response(200, json={"choices": [choice], "usage": [2, 3, 5]})
    if kind == "usage-overflow":
        return httpx.Response(
            200,
            json={
                "choices": [choice],
                "usage": {
                    "prompt_tokens": JSON_SAFE_INTEGER_MAX + 1,
                    "completion_tokens": 0,
                    "total_tokens": JSON_SAFE_INTEGER_MAX + 1,
                },
            },
        )
    if kind == "finish-reason":
        choice.pop("finish_reason")
        return httpx.Response(200, json={"choices": [choice]})
    if kind == "numeric-overflow":
        encoded = json.dumps({"choices": [choice]}, separators=(",", ":"))
        return httpx.Response(200, content=(encoded[:-1] + ',"overflow":1e999}').encode())
    assert kind == "duplicate-top-level"
    encoded = json.dumps({"choices": [choice]}, separators=(",", ":"))
    return httpx.Response(200, content=(encoded[:-1] + ',"choices":[]}').encode())


@pytest.mark.parametrize(
    "kind",
    [
        "usage-total",
        "usage-boolean",
        "usage-list",
        "usage-overflow",
        "finish-reason",
        "numeric-overflow",
        "duplicate-top-level",
    ],
)
async def test_malformed_success_envelope_is_an_environment_failure(tmp_path, kind):
    rows = load_packaged_corpus()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        row = _body_row(body, rows)
        content = json.dumps(row["expected"], separators=(",", ":"))
        if "response_format" in body:
            return _invalid_completion(kind, content)
        return _completion(content)

    pair = await StructuredOutputAdapter().run(
        BenchTarget(base_url="http://gateway/v1", model="model"),
        _context(tmp_path, handler, limit=1),
    )
    item = pair.items[0]
    evidence = item.structured_output

    assert evidence is not None
    assert evidence.constrained.outcome == "failed"
    assert evidence.constrained.http_status == 200
    assert evidence.constrained.error == "structured request invalid HTTP 200 envelope"
    assert item.status == "failed"
    assert pair.status == "failed"
    assert pair.metrics["structured_complete"] == 0
    assert structured_summary(pair) is None
    assert pair_result_evidence_error(pair) is None


async def test_multi_attempts_nest_seed_then_paired_arms_without_flattening(tmp_path):
    rows = load_packaged_corpus()
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        row = _body_row(body, rows)
        return _completion(json.dumps(row["expected"], separators=(",", ":")))

    target = BenchTarget(
        base_url="http://gateway/v1",
        model="model",
        seed=7,
        sampling_mode="recommended",
    )
    pair = await StructuredOutputAdapter().run(
        target,
        _context(tmp_path, handler, attempts=3, limit=2),
    )

    assert pair.status == "completed"
    assert pair.score == 1.0
    assert len(pair.items) == 2
    assert len(bodies) == 2 * 3 * 2
    observation_index = 0
    for item in pair.items:
        assert item.structured_output is None
        assert item.sampling_attempts is not None
        assert [attempt.seed for attempt in item.sampling_attempts] == [7, 8, 9]
        for attempt_index, attempt in enumerate(item.sampling_attempts, start=1):
            assert attempt.attempt == attempt_index
            assert attempt.result.item_id == item.item_id
            evidence = attempt.result.structured_output
            assert evidence is not None
            assert evidence.seed == attempt.seed
            assert evidence.order == arm_order(observation_index)
            observation_index += 1

    for row in rows:
        row_bodies = [body for body in bodies if _body_row(body, rows)["id"] == row["id"]]
        if not row_bodies:
            continue
        assert sorted(body["seed"] for body in row_bodies) == [7, 7, 8, 8, 9, 9]
        assert all("temperature" not in body for body in row_bodies)
        for seed in (7, 8, 9):
            seed_bodies = [body for body in row_bodies if body["seed"] == seed]
            constrained = next(body for body in seed_bodies if "response_format" in body)
            unconstrained = next(body for body in seed_bodies if "response_format" not in body)
            assert {
                key: value for key, value in constrained.items() if key != "response_format"
            } == unconstrained

    summary = structured_summary(pair)
    assert summary is not None
    assert summary["n_source_items"] == 2
    assert summary["attempts"] == 3
    assert summary["observations"] == 6
    assert pair.metrics["structured_observations"] == 6
    assert pair.metrics["sampling_complete"] == 1
    assert pair.metrics["sampling_attempts"] == 3
    assert pair.metrics["sampling_base_items"] == 2
    assert pair_result_evidence_error(
        pair,
        expected_sampling=_expected_sampling(target, 3),
    ) is None


async def test_replay_validator_rejects_raw_derived_and_methodology_tampering(tmp_path):
    rows = load_packaged_corpus()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        row = _body_row(body, rows)
        return _completion(json.dumps(row["expected"], separators=(",", ":")))

    pair = await StructuredOutputAdapter().run(
        BenchTarget(base_url="http://gateway/v1", model="model", seed=3),
        _context(tmp_path, handler, limit=1),
    )
    assert structured_pair_evidence_error(pair) is None
    assert pair_result_evidence_error(pair) is None

    item = pair.items[0]
    evidence = item.structured_output
    assert evidence is not None
    row = next(value for value in rows if value["id"] == item.item_id)
    wrong = deepcopy(row["expected"])
    first_key = next(iter(wrong))
    wrong[first_key] = "forged"
    forged_content = json.dumps(wrong, separators=(",", ":"))
    forged_arm = evidence.constrained.model_copy(update={"content": forged_content})
    forged_evidence = evidence.model_copy(update={"constrained": forged_arm})
    forged_item = item.model_copy(
        update={
            "structured_output": forged_evidence,
            "response_excerpt": forged_content,
        }
    )
    forged_raw = pair.model_copy(update={"items": (forged_item,)})
    assert "score/status" in (structured_pair_evidence_error(forged_raw) or "")

    forged_metrics = dict(pair.metrics)
    forged_metrics["structured_constrained_task_accuracy"] = 0.0
    forged_derived = pair.model_copy(update={"metrics": forged_metrics})
    assert "disagrees with structured arm evidence" in (
        structured_pair_evidence_error(forged_derived) or ""
    )

    forged_methodology = deepcopy(pair.methodology)
    forged_methodology["structured_output"]["corpus_revision"] = "sha256:" + "0" * 64
    forged_claim = pair.model_copy(update={"methodology": forged_methodology})
    assert "methodology does not match" in (
        structured_pair_evidence_error(forged_claim) or ""
    )

    forged_excerpt = pair.model_copy(
        update={"items": (item.model_copy(update={"response_excerpt": "forged"}),)}
    )
    assert "excerpt" in (structured_pair_evidence_error(forged_excerpt) or "")

    wrong_seed = evidence.model_copy(update={"seed": 4})
    seed_pair = pair.model_copy(
        update={"items": (item.model_copy(update={"structured_output": wrong_seed}),)}
    )
    assert "seed" in (structured_pair_evidence_error(seed_pair) or "")

    wrong_order = evidence.model_copy(update={"order": tuple(reversed(evidence.order))})
    order_pair = pair.model_copy(
        update={"items": (item.model_copy(update={"structured_output": wrong_order}),)}
    )
    assert "arm order" in (structured_pair_evidence_error(order_pair) or "")

    wrong_latency_arm = evidence.constrained.model_copy(
        update={"latency_s": evidence.constrained.latency_s + 1.0}
    )
    wrong_latency = evidence.model_copy(update={"constrained": wrong_latency_arm})
    latency_pair = pair.model_copy(
        update={"items": (item.model_copy(update={"structured_output": wrong_latency}),)}
    )
    assert "latency" in (structured_pair_evidence_error(latency_pair) or "")

    usage = evidence.constrained.usage
    assert usage is not None
    wrong_usage = usage.model_copy(
        update={
            "prompt_tokens": usage.prompt_tokens + 1,
            "total_tokens": usage.total_tokens + 1,
        }
    )
    wrong_usage_arm = evidence.constrained.model_copy(update={"usage": wrong_usage})
    usage_evidence = evidence.model_copy(update={"constrained": wrong_usage_arm})
    usage_pair = pair.model_copy(
        update={"items": (item.model_copy(update={"structured_output": usage_evidence}),)}
    )
    assert "disagrees with structured arm evidence" in (
        structured_pair_evidence_error(usage_pair) or ""
    )


async def test_scoreboard_and_schema1_history_bind_then_strip_structured_claim(tmp_path):
    rows = load_packaged_corpus()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        row = _body_row(body, rows)
        return _completion(json.dumps(row["expected"], separators=(",", ":")))

    adapter = StructuredOutputAdapter()
    cache = BenchCache(tmp_path / "cache")
    report = adapter.download(DownloadContext(cache=cache))
    assert report.status == "ok"
    target = BenchTarget(
        name="target-a",
        base_url="http://gateway/v1",
        model="model",
        seed=17,
    )
    config = BenchConfig(
        suite="structured",
        targets=(target,),
        limit=1,
        cache_dir=str(cache.root),
        results_dir=str(tmp_path / "results"),
        run_id="structured-history",
        download=False,
        progress=False,
    )
    adapter_identity = _adapter_identity(adapter, cache, offline_fixtures=False)
    identity = _run_identity(config, [adapter_identity])
    fingerprint = _run_fingerprint(identity)
    recorded_config = _recordable_config(config)
    environment = {
        "git_commit": "a" * 40,
        "git_commit_role": "local_benchmark_harness",
        "source_kind": "git-checkout",
        "source_tree_clean": True,
        "source_tree_sha256": hashlib.sha256(b"structured-tree").hexdigest(),
        "kairyu_version": "0.1.0",
        "python": "3.12.3",
        "created_at": "2026-08-06T00:00:00+00:00",
        "execution": {
            "runner": "local",
            "available": True,
            "availability_detail": "available",
        },
    }
    metadata = {
        "fingerprint": fingerprint,
        "identity": identity,
        "config": recorded_config,
        "environment": environment,
        "run_id": "structured-history",
    }
    pair = await adapter.run(
        target,
        _context(tmp_path, handler, limit=1, offline_fixtures=False),
    )
    pair = pair.model_copy(
        update={
            "run_fingerprint": fingerprint,
            "source_identity": history.source_identity(environment),
            "comparable": False,
            "incomparable_reasons": (
                "subset run: at most 1 items per benchmark, not the full set",
            ),
        }
    )
    board = build_scoreboard(
        run_id="structured-history",
        suite="structured",
        config=recorded_config,
        environment=environment,
        fingerprint=fingerprint,
        pairs=[pair],
        targets=["target-a"],
        target_configs=[target],
    )
    cell = board["cells"]["structured-output"]["target-a"]
    assert cell["structured_conformance"] == structured_summary(pair)
    markdown = render_markdown(board)
    assert "Structured-output conformance details" in markdown
    assert "not calculated" in markdown

    falsely_comparable = pair.model_copy(
        update={"comparable": True, "incomparable_reasons": ()}
    )
    falsely_comparable_board = build_scoreboard(
        run_id="structured-history",
        suite="structured",
        config=recorded_config,
        environment=environment,
        fingerprint=fingerprint,
        pairs=[falsely_comparable],
        targets=["target-a"],
        target_configs=[target],
    )
    with pytest.raises(ValueError, match="omits run-level incomparability"):
        history.build_entry(
            falsely_comparable_board,
            metadata,
            pairs=[falsely_comparable],
            previous_record_sha256=None,
        )

    entry = history.build_entry(
        board,
        metadata,
        pairs=[pair],
        previous_record_sha256=None,
    )
    assert "structured_conformance" not in entry["scoreboard"]["cells"][
        "structured-output"
    ]["target-a"]
    history._validate_entry(entry, previous=None)

    forged_fresh = deepcopy(board)
    forged_fresh["cells"]["structured-output"]["target-a"][
        "structured_conformance"
    ]["arms"]["constrained"]["task_correct"] = 0
    assert "Structured-output conformance details" not in render_markdown(forged_fresh)
    with pytest.raises(ValueError, match="structured conformance disagrees"):
        history.build_entry(
            forged_fresh,
            metadata,
            pairs=[pair],
            previous_record_sha256=None,
        )

    huge_detached = deepcopy(board)
    huge_detached["cells"]["structured-output"]["target-a"][
        "structured_conformance"
    ]["arms"]["constrained"]["accepted"] = 10**1_000
    assert "Structured-output conformance details" not in render_markdown(
        huge_detached
    )

    forged_stored = deepcopy(entry)
    forged_stored["scoreboard"]["cells"]["structured-output"]["target-a"][
        "structured_conformance"
    ] = cell["structured_conformance"]
    forged_stored["record_sha256"] = history._record_digest(forged_stored)
    with pytest.raises(ValueError, match="must not carry structured conformance"):
        history._validate_entry(forged_stored, previous=None)

    missing_resource = deepcopy(metadata)
    resources = missing_resource["identity"]["adapters"][0]["evaluation_protocol"][
        "resources"
    ]
    resources[:] = [
        resource
        for resource in resources
        if (resource["package"], resource["resource"])
        != ("kairyu.bench", "structured.py")
    ]
    missing_resource["fingerprint"] = _run_fingerprint(missing_resource["identity"])
    missing_board = deepcopy(board)
    missing_board["fingerprint"] = missing_resource["fingerprint"]
    missing_pair = pair.model_copy(
        update={"run_fingerprint": missing_resource["fingerprint"]}
    )
    with pytest.raises(ValueError, match="omits required resources"):
        history.build_entry(
            missing_board,
            missing_resource,
            pairs=[missing_pair],
            previous_record_sha256=None,
        )

    missing_distribution = deepcopy(metadata)
    distributions = missing_distribution["identity"]["adapters"][0][
        "evaluation_protocol"
    ]["distributions"]
    distributions[:] = [
        distribution
        for distribution in distributions
        if distribution["name"] != "jsonschema"
    ]
    missing_distribution["fingerprint"] = _run_fingerprint(
        missing_distribution["identity"]
    )
    missing_board = deepcopy(board)
    missing_board["fingerprint"] = missing_distribution["fingerprint"]
    missing_pair = pair.model_copy(
        update={"run_fingerprint": missing_distribution["fingerprint"]}
    )
    with pytest.raises(ValueError, match="omits required distributions"):
        history.build_entry(
            missing_board,
            missing_distribution,
            pairs=[missing_pair],
            previous_record_sha256=None,
        )

    absent_distributions = deepcopy(metadata)
    distributions = absent_distributions["identity"]["adapters"][0][
        "evaluation_protocol"
    ]["distributions"]
    distributions[:] = [
        {"name": distribution["name"], "installed": False}
        for distribution in distributions
    ]
    absent_distributions["fingerprint"] = _run_fingerprint(
        absent_distributions["identity"]
    )
    absent_board = deepcopy(board)
    absent_board["fingerprint"] = absent_distributions["fingerprint"]
    absent_pair = pair.model_copy(
        update={"run_fingerprint": absent_distributions["fingerprint"]}
    )
    with pytest.raises(ValueError, match="requires installed distributions"):
        history.build_entry(
            absent_board,
            absent_distributions,
            pairs=[absent_pair],
            previous_record_sha256=None,
        )

    skipped_metadata = deepcopy(absent_distributions)
    skipped_metadata["fingerprint"] = _run_fingerprint(skipped_metadata["identity"])
    skipped_pair = PairResult(
        benchmark="structured-output",
        target="target-a",
        status="skipped",
        reason="structured-output scoring requires the bench extra",
        metrics={"score": None, "n_total": 0},
        comparable=False,
        incomparable_reasons=(
            "subset run: at most 1 items per benchmark, not the full set",
        ),
        run_fingerprint=skipped_metadata["fingerprint"],
        source_identity=history.source_identity(environment),
    )
    skipped_board = build_scoreboard(
        run_id="structured-history",
        suite="structured",
        config=skipped_metadata["config"],
        environment=environment,
        fingerprint=skipped_metadata["fingerprint"],
        pairs=[skipped_pair],
        targets=["target-a"],
        target_configs=[target],
    )
    history.build_entry(
        skipped_board,
        skipped_metadata,
        pairs=[skipped_pair],
        previous_record_sha256=None,
    )

    for field, tampered, message in (
        ("dataset", "attacker/corpus", "fixed corpus"),
        ("revision", "attacker-revision", "fixed corpus"),
        ("sha256", "0" * 64, "exact fixed cache rows"),
    ):
        wrong_source = deepcopy(metadata)
        wrong_source["identity"]["adapters"][0][field] = tampered
        wrong_source["fingerprint"] = _run_fingerprint(wrong_source["identity"])
        wrong_board = deepcopy(board)
        wrong_board["fingerprint"] = wrong_source["fingerprint"]
        wrong_pair = pair.model_copy(
            update={"run_fingerprint": wrong_source["fingerprint"]}
        )
        with pytest.raises(ValueError, match=message):
            history.build_entry(
                wrong_board,
                wrong_source,
                pairs=[wrong_pair],
                previous_record_sha256=None,
            )

    forbidden_config = deepcopy(metadata)
    extra_body_digest = "sha256:" + hashlib.sha256(b"response-format").hexdigest()
    forbidden_config["config"]["targets"][0]["extra_body_json"] = extra_body_digest
    forbidden_config["identity"]["config"]["targets"][0][
        "extra_body_json"
    ] = extra_body_digest
    forbidden_config["fingerprint"] = _run_fingerprint(forbidden_config["identity"])
    forbidden_board = deepcopy(board)
    forbidden_board["config"] = deepcopy(forbidden_config["config"])
    forbidden_board["fingerprint"] = forbidden_config["fingerprint"]
    forbidden_pair = pair.model_copy(
        update={"run_fingerprint": forbidden_config["fingerprint"]}
    )
    with pytest.raises(ValueError, match="rejected extra_body_json"):
        history.build_entry(
            forbidden_board,
            forbidden_config,
            pairs=[forbidden_pair],
            previous_record_sha256=None,
        )

    forged_full_run = deepcopy(metadata)
    forged_full_run["config"]["limit"] = None
    forged_full_run["identity"]["config"]["limit"] = None
    forged_full_run["fingerprint"] = _run_fingerprint(forged_full_run["identity"])
    forged_full_board = deepcopy(board)
    forged_full_board["config"] = deepcopy(forged_full_run["config"])
    forged_full_board["fingerprint"] = forged_full_run["fingerprint"]
    forged_full_pair = pair.model_copy(
        update={"run_fingerprint": forged_full_run["fingerprint"]}
    )
    with pytest.raises(ValueError, match="fixed-corpus selection"):
        history.build_entry(
            forged_full_board,
            forged_full_run,
            pairs=[forged_full_pair],
            previous_record_sha256=None,
        )
