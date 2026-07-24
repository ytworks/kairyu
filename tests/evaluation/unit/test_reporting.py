import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from kairyu.evaluation.protocol import protocol_hash
from kairyu.evaluation.references import load_reference_snapshot
from kairyu.evaluation.reporting import (
    ReportError,
    ReportInputs,
    UsageEvidence,
    build_report,
    build_run_manifest,
    render_report,
)
from kairyu.evaluation.schemas import (
    BenchmarkRun,
    ItemState,
    Metric,
    ProtocolSignature,
    RunItem,
    RunMode,
    RunState,
)

CREATED = datetime(2026, 7, 20, 1, 0, tzinfo=UTC)
STARTED = datetime(2026, 7, 20, 1, 1, tzinfo=UTC)
FINISHED = datetime(2026, 7, 20, 1, 2, tzinfo=UTC)


def _protocol() -> ProtocolSignature:
    return ProtocolSignature(
        benchmark_id="gpqa-diamond",
        benchmark_version="synthetic-v1",
        dataset_revision="sha256:synthetic-fixture",
        split="smoke",
        harness_name="kairyu-gpqa-smoke",
        harness_version="1",
        harness_commit="c" * 40,
        prompt_version="evalscope-1.8.1-shaped-v1",
        retries=2,
        timeout_seconds=90.0,
        generation_parameters={"temperature": 0.0, "seed": 42},
        reasoning_effort="medium",
        judge_model="judge-model",
        adapter_configuration={
            "judge_generation_parameters": {
                "max_tokens": 2048,
                "reasoning_effort": "high",
                "temperature": 0.0,
            }
        },
        simulator_model="simulator-model",
        metric_implementation="accuracy-v1",
    )


def _run(
    *,
    mode: RunMode = RunMode.SMOKE,
    state: RunState = RunState.COMPLETED,
    partial: bool = False,
    selected_item_ids: tuple[str, ...] = ("synthetic-1", "synthetic-2"),
    completed_count: int = 2,
    failed_count: int = 0,
    skipped_count: int = 0,
    target_model: str = "fake-model",
    judge_model: str | None = "judge-model",
    simulator_model: str | None = "simulator-model",
) -> BenchmarkRun:
    protocol = _protocol()
    return BenchmarkRun(
        run_id="run-gpqa-01",
        benchmark_id="gpqa-diamond",
        profile="smoke",
        mode=mode,
        state=state,
        partial=partial,
        protocol_hash=protocol_hash(protocol),
        item_input_manifest_sha256="b" * 64,
        selected_item_ids=selected_item_ids,
        expected_full_count=198,
        completed_count=completed_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        target_model=target_model,
        judge_model=judge_model,
        simulator_model=simulator_model,
        created_at=CREATED,
        started_at=STARTED,
        finished_at=FINISHED,
    )


def _items() -> tuple[RunItem, ...]:
    return (
        RunItem(
            run_id="run-gpqa-01",
            item_id="synthetic-1",
            ordinal=0,
            state=ItemState.COMPLETED,
            input_sha256="1" * 64,
            scores={"accuracy": 1.0},
        ),
        RunItem(
            run_id="run-gpqa-01",
            item_id="synthetic-2",
            ordinal=1,
            state=ItemState.COMPLETED,
            input_sha256="2" * 64,
            scores={"accuracy": 0.0},
        ),
    )


def _metric(
    *,
    value: float | None = 50.0,
    numerator: int | None = 1,
    denominator: int | None = 2,
    official_eligible: bool = False,
) -> Metric:
    return Metric(
        run_id="run-gpqa-01",
        name="accuracy",
        display_name="Accuracy",
        value=value,
        numerator=numerator,
        denominator=denominator,
        primary=True,
        official_eligible=official_eligible,
    )


def _inputs(
    *,
    run: BenchmarkRun | None = None,
    metric: Metric | None = None,
    items: tuple[RunItem, ...] | None = None,
    protocol: ProtocolSignature | None = None,
    error_counts: tuple[ReportError, ...] = (),
) -> ReportInputs:
    snapshot = load_reference_snapshot()
    return ReportInputs(
        run=run or _run(),
        protocol=protocol or _protocol(),
        metrics=(metric or _metric(),),
        items=_items() if items is None else items,
        error_counts=error_counts,
        usage=UsageEvidence(
            input_tokens=20,
            output_tokens=10,
            total_latency_seconds=1.25,
            measurement_status="complete",
            actual_cost_usd=0.125,
        ),
        sources=(snapshot.source,),
        references=snapshot.results,
    )


def test_manifest_uses_planning_evidence_and_reports_runtime_attestation_gaps():
    compatibility_sha256 = "a" * 64
    protocol = _protocol().model_copy(
        update={
            "timeout_seconds": None,
            "generation_parameters": {
                "temperature": 0.0,
                "max_tokens": 1_024,
                "timeout_seconds": 45.0,
            },
            "dependency_lock_sha256": "b" * 64,
            "dependency_compatibility_patches": (
                f"kairyu.evaluation.adapters.gpqa_v181@sha256:{compatibility_sha256}",
            ),
            "adapter_configuration": {
                "compatibility_layer_sha256": compatibility_sha256,
                "planning_evidence": {
                    "estimate": {
                        "selected_item_count": 2,
                        "model_calls": 2,
                        "maximum_model_calls": 6,
                        "estimated_input_tokens": 123,
                        "maximum_output_tokens": 2_048,
                        "estimated_duration_seconds": None,
                        "maximum_duration_seconds": 360.0,
                        "estimated_cost_usd": None,
                        "assumptions": ["provider pricing is not configured"],
                    },
                    "resources": {
                        "cpu_cores": 1,
                        "ram_bytes": 536_870_912,
                        "disk_bytes": 67_108_864,
                        "docker_required": False,
                        "network_policy": "configured model endpoint only",
                    },
                    "execution": {
                        "kind": "in-process-adapter",
                        "command": ["kairyu", "benchmark", "worker", "--once"],
                        "container_image_digest": None,
                    },
                },
                "reproducibility_evidence": {
                    "runtime_dependency_environment": {
                        "status": "unresolved",
                        "reason": "active environment is not attested",
                    }
                },
            },
        }
    )
    run = _run().model_copy(update={"protocol_hash": protocol_hash(protocol)})
    usage = _inputs().usage

    manifest = build_run_manifest(run, protocol, usage)
    report = build_report(_inputs(run=run, protocol=protocol))

    assert report.protocol.timeout_seconds == 45.0
    assert manifest.resources.status == "known"
    assert manifest.resources.cpu_cores.value == 1
    assert manifest.resources.ram_bytes.value == 536_870_912
    assert manifest.resources.disk_bytes.value == 67_108_864
    assert manifest.resources.docker_required.value is False
    assert manifest.resources.network_policy.value == "configured model endpoint only"
    assert manifest.estimates.model_calls.value == 2
    assert manifest.estimates.maximum_model_calls.value == 6
    assert manifest.estimates.estimated_input_tokens.value == 123
    assert manifest.estimates.maximum_output_tokens.value == 2_048
    assert manifest.estimates.estimated_duration_seconds.value is None
    assert manifest.estimates.estimated_duration_seconds.unavailable_reason
    assert manifest.estimates.maximum_duration_seconds.value == 360.0
    assert manifest.estimates.estimated_cost_usd.value is None
    assert manifest.estimates.estimated_cost_usd.unavailable_reason
    assert manifest.estimates.assumptions == ("provider pricing is not configured",)
    assert manifest.container.image_digest.value is None
    assert manifest.container.image_digest.unavailable_reason
    assert manifest.software.status == "partial"
    assert manifest.software.compatibility_layer_identity.value == (
        "kairyu.evaluation.adapters.gpqa_v181"
    )
    assert manifest.software.compatibility_layer_sha256.value == compatibility_sha256
    assert manifest.software.runtime_dependency_environment_status.value == "unresolved"
    assert (
        manifest.software.runtime_dependency_environment_reason.value
        == "active environment is not attested"
    )


def test_rendered_reports_are_deterministic_complete_and_evidence_only():
    first = render_report(_inputs())
    second = render_report(_inputs())

    assert first.json == second.json
    assert first.markdown == second.markdown
    assert first.html == second.html
    assert first.json.startswith('{\n  "notice":')
    payload = json.loads(first.json)
    assert payload["schema_version"] == 3
    assert payload["report_date"] == "2026-07-20"
    assert payload["evidence_as_of"] == "2026-07-20T01:02:00Z"
    assert payload["protocol_hash"] == protocol_hash(_protocol())
    assert payload["protocol"]["harness_commit"] == "c" * 40
    assert payload["protocol"]["retries"] == 2
    assert payload["protocol"]["timeout_seconds"] == 90.0
    assert payload["protocol"]["reasoning_effort"] == "medium"
    assert payload["protocol"]["judge_generation_parameters"] == {
        "max_tokens": 2048,
        "reasoning_effort": "high",
        "temperature": 0.0,
    }
    assert "Provider APIs may be nondeterministic" in payload["reproducibility_notice"]
    assert "seed does not guarantee identical responses" in payload["reproducibility_notice"]
    assert payload["target_model"] == "fake-model"
    assert payload["judge_model"] == "judge-model"
    assert payload["simulator_model"] == "simulator-model"
    assert payload["usage"] == {
        "actual_cost_unavailable_reason": None,
        "actual_cost_usd": 0.125,
        "input_tokens": 20,
        "measurement_status": "complete",
        "measurement_unavailable_reasons": [],
        "output_tokens": 10,
        "schema_version": 1,
        "total_latency_seconds": 1.25,
    }
    assert payload["counts"] == {
        "cancelled": 0,
        "completed": 2,
        "expected_full": 198,
        "failed": 0,
        "pending": 0,
        "reported_items": 2,
        "running": 0,
        "selected": 2,
        "skipped": 0,
    }
    assert payload["errors"] == []
    assert payload["items"] == [
        {"item_id": "synthetic-1", "score": 1.0, "state": "completed"},
        {"item_id": "synthetic-2", "score": 0.0, "state": "completed"},
    ]
    assert len(payload["references"]) == 5
    assert {reference["comparability"] for reference in payload["references"]} == {"incompatible"}
    assert all(reference["protocol_hash"] is None for reference in payload["references"])
    assert all(reference["delta"] is None for reference in payload["references"])
    assert all(reference["rank"] is None for reference in payload["references"])
    assert payload["sources"][0]["url"] == "https://arxiv.org/pdf/2606.21228"
    assert payload["sources"][0]["retrieved_at"] == "2026-07-24T00:00:00Z"
    assert "notes" not in payload["sources"][0]
    assert "question" not in first.json.casefold()
    assert "raw prompt" not in first.markdown.casefold()
    assert "answer choices" not in first.html.casefold()
    assert "- Judge model: `judge-model`" in first.markdown
    assert "- Simulator model: `simulator-model`" in first.markdown
    assert "## Usage / Cost" in first.markdown
    assert "| 20 | 10 | 1.25 | complete | 0.125 | null |" in first.markdown
    assert "<dt>Judge model</dt><dd>judge-model</dd>" in first.html
    assert "<dt>Simulator model</dt><dd>simulator-model</dd>" in first.html
    assert "<h2>Usage / Cost</h2>" in first.html
    assert "<dt>Actual cost (USD)</dt><dd>0.125</dd>" in first.html
    assert "## Protocol Details" in first.markdown
    assert (
        "- Judge generation parameters: "
        '{"max\\_tokens":2048,"reasoning\\_effort":"high",'
        '"temperature":0.0}'
    ) in first.markdown
    assert f"| kairyu-gpqa-smoke | 1 | {'c' * 40} | 2 | 90.0 | medium |" in first.markdown
    assert "Provider APIs may be nondeterministic" in first.markdown
    assert "<h2>Protocol Details</h2>" in first.html
    assert f"<dt>Upstream commit</dt><dd>{'c' * 40}</dd>" in first.html
    assert "<dt>Retries</dt><dd>2</dd>" in first.html
    assert "<dt>Timeout (seconds)</dt><dd>90.0</dd>" in first.html
    assert "<dt>Reasoning effort</dt><dd>medium</dd>" in first.html
    assert "<dt>Judge generation parameters</dt>" in first.html
    assert "Provider APIs may be nondeterministic" in first.html


@pytest.mark.parametrize(
    ("mode", "state", "partial"),
    (
        (RunMode.SMOKE, RunState.COMPLETED, False),
        (RunMode.SAMPLE, RunState.COMPLETED, False),
        (RunMode.FULL, RunState.PARTIAL, True),
        (RunMode.FULL, RunState.CANCELLED, False),
        (RunMode.FULL, RunState.FAILED, False),
    ),
)
def test_nonofficial_or_interrupted_reports_never_compute_deltas_or_ranks(mode, state, partial):
    run = _run(
        mode=mode,
        state=state,
        partial=partial,
        selected_item_ids=(),
        completed_count=0,
    )
    metric = _metric(value=None, numerator=0, denominator=0, official_eligible=True)
    report = build_report(_inputs(run=run, metric=metric, items=()))

    assert report.notice.startswith("UNOFFICIAL:")
    assert "not full-suite accuracy" in report.notice
    assert "ranking" in report.notice
    assert report.unofficial is True
    assert report.comparison_eligible is False
    assert report.rank is None
    assert all(reference.delta is None for reference in report.references)
    assert all(reference.rank is None for reference in report.references)


def test_zero_item_report_uses_null_metric_and_never_nan():
    run = _run(selected_item_ids=(), completed_count=0)
    metric = _metric(value=None, numerator=0, denominator=0)

    rendered = render_report(_inputs(run=run, metric=metric, items=()))
    payload = json.loads(rendered.json)

    assert payload["metrics"][0]["value"] is None
    assert payload["metrics"][0]["denominator"] == 0
    assert "NaN" not in rendered.json
    assert "nan" not in rendered.markdown.casefold()


def test_failed_item_errors_are_aggregated_without_leaking_item_evidence():
    run = _run(
        state=RunState.FAILED,
        partial=True,
        selected_item_ids=("synthetic-1",),
        completed_count=0,
        failed_count=1,
    )
    item = RunItem(
        run_id=run.run_id,
        item_id="synthetic-1",
        ordinal=0,
        state=ItemState.FAILED,
        input_sha256="1" * 64,
        error_class="rate_limit",
    )
    metric = _metric(value=0.0, numerator=0, denominator=1)

    payload = json.loads(
        render_report(
            _inputs(
                run=run,
                metric=metric,
                items=(item,),
                error_counts=(ReportError(error_class="rate_limit", count=1),),
            )
        ).json
    )

    assert payload["errors"] == [{"count": 1, "error_class": "rate_limit"}]
    assert payload["metrics"][0]["value"] == 0.0
    assert payload["metrics"][0]["denominator"] == 1
    assert payload["items"] == [{"item_id": "synthetic-1", "score": None, "state": "failed"}]
    assert set(payload["items"][0]) == {"item_id", "state", "score"}


def test_report_input_rejects_raw_prompt_or_choice_payloads():
    tainted = _metric().model_copy(
        update={"dimensions": {"raw_prompt": "official benchmark content"}}
    )

    with pytest.raises(ValidationError, match="raw benchmark"):
        _inputs(metric=tainted)

    snapshot = load_reference_snapshot()
    with pytest.raises(ValidationError, match="raw benchmark"):
        ReportInputs.model_validate(
            {
                "run": _run(),
                "protocol": _protocol(),
                "metrics": [_metric()],
                "items": _items(),
                "sources": [snapshot.source],
                "references": snapshot.results,
                "choices": ["forbidden"],
            }
        )


def test_html_and_markdown_escape_untrusted_stored_labels_and_are_self_contained():
    run = _run(target_model='<img src=x onerror="alert(1)">')

    rendered = render_report(_inputs(run=run))

    assert "<img" not in rendered.html
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in rendered.html
    assert '\\<img src=x onerror="alert\\(1\\)"\\>' in rendered.markdown
    assert rendered.html.startswith("<!doctype html>")
    assert "<style>" in rendered.html
    assert "<script" not in rendered.html.casefold()
    assert "<link" not in rendered.html.casefold()


def test_full_complete_run_with_only_incompatible_references_stays_unranked():
    run = _run(mode=RunMode.FULL)
    metric = _metric(official_eligible=True)

    report = build_report(_inputs(run=run, metric=metric))

    assert report.unofficial is False
    assert report.notice.startswith("COMPARISON UNAVAILABLE:")
    assert report.comparison_eligible is False
    assert report.rank is None
    assert all(reference.delta is None for reference in report.references)


def test_report_rejects_mismatched_run_counts_and_metric_math():
    bad_run = _run(completed_count=1)
    with pytest.raises(ValidationError, match="completed item evidence"):
        _inputs(run=bad_run)

    bad_metric = _metric(value=49.0)
    with pytest.raises(ValidationError, match="does not match"):
        _inputs(metric=bad_metric)


@pytest.mark.parametrize(
    ("role", "run_model", "protocol_model"),
    (
        ("judge", None, "judge-model"),
        ("judge", "judge-model", None),
        ("judge", "run-judge", "protocol-judge"),
        ("simulator", None, "simulator-model"),
        ("simulator", "simulator-model", None),
        ("simulator", "run-simulator", "protocol-simulator"),
    ),
)
def test_report_requires_exact_optional_role_model_equality(
    role: str,
    run_model: str | None,
    protocol_model: str | None,
):
    field = f"{role}_model"
    protocol = _protocol().model_copy(update={field: protocol_model})
    run = _run().model_copy(
        update={
            field: run_model,
            "protocol_hash": protocol_hash(protocol),
        }
    )

    with pytest.raises(ValidationError, match=f"run and protocol {role} models must match"):
        _inputs(run=run, protocol=protocol)


def test_report_accepts_matching_absent_optional_role_models():
    protocol = _protocol().model_copy(
        update={
            "judge_model": None,
            "simulator_model": None,
        }
    )
    run = _run(judge_model=None, simulator_model=None).model_copy(
        update={"protocol_hash": protocol_hash(protocol)}
    )

    assert _inputs(run=run, protocol=protocol).run == run
