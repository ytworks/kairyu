import json
from pathlib import Path

import pytest

from bench.tiered_auto_bench import (
    LIVE_CODE_BENCH_DATASET,
    LIVE_CODE_BENCH_REVISION,
    QualityMeasurement,
    count_internal_calls,
    estimated_cost_usd,
    load_fixed_items,
    summarize_quality,
)

RESULT = (
    Path(__file__).parents[2]
    / "bench/results/tiered-auto-qwen3-32b-tp8-2026-07-27.json"
)


def _measurement(**overrides):
    values = {
        "item_id": "x",
        "model": "auto",
        "order": 0,
        "latency_s": 2.0,
        "score": 1.0,
        "error": None,
        "route": "multi_agent",
        "internal_calls": 4,
        "orchestration_input_tokens": 100,
        "orchestration_output_tokens": 20,
        "cached_tokens": 5,
        "allocated_gpu_seconds": 16.0,
        "estimated_cost_usd": None,
        "response_sha256": "a" * 64,
        "response_excerpt": "ok",
    }
    values.update(overrides)
    return QualityMeasurement(**values)


def test_count_internal_calls_uses_moa_proposal_count():
    trace = {
        "events": [
            {"node": "router", "operation": "routing"},
            {
                "node": "moa",
                "kind": "synthesis",
                "usage": {"prompt_tokens": 10},
                "detail": {"proposals": 3},
            },
        ]
    }
    assert count_internal_calls(trace) == 4


def test_count_internal_calls_counts_conductor_usage_events():
    trace = {
        "events": [
            {"node": "router", "kind": "routing", "usage": None},
            {"node": "planner", "kind": "generation", "usage": {}},
            {"node": "worker", "kind": "generation", "usage": {}},
            {"node": "verifier", "kind": "verification", "usage": {}},
            {"node": "synthesizer", "kind": "generation", "usage": {}},
        ]
    }
    assert count_internal_calls(trace) == 4


def test_cost_is_optional_or_uses_declared_rates():
    assert (
        estimated_cost_usd(
            100,
            50,
            input_usd_per_mtok=0,
            output_usd_per_mtok=0,
        )
        is None
    )
    assert estimated_cost_usd(
        1_000_000,
        500_000,
        input_usd_per_mtok=2,
        output_usd_per_mtok=4,
    ) == pytest.approx(4.0)


def test_quality_summary_keeps_resource_costs_and_score():
    summary = summarize_quality(
        [
            _measurement(),
            _measurement(
                item_id="y",
                score=0.0,
                latency_s=4.0,
                internal_calls=5,
                orchestration_input_tokens=200,
                orchestration_output_tokens=40,
                allocated_gpu_seconds=32.0,
            ),
        ]
    )
    assert summary == {
        "items": 2,
        "passed_items": 1,
        "score": 0.5,
        "latency_p50_s": 3.0,
        "latency_total_s": 6.0,
        "internal_calls": 9,
        "orchestration_input_tokens": 300,
        "orchestration_output_tokens": 60,
        "cached_tokens": 10,
        "allocated_gpu_seconds": 48.0,
        "estimated_cost_usd": None,
    }


def test_fixed_item_loader_requires_pin_and_preserves_requested_order(tmp_path):
    root = tmp_path / "livecodebench"
    root.mkdir()
    manifest = {
        "dataset": LIVE_CODE_BENCH_DATASET,
        "revision": LIVE_CODE_BENCH_REVISION,
        "rows": 2,
        "sha256": "fixture",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [
        {
            "id": "b",
            "question": "b",
            "starter_code": "",
            "fn_name": None,
            "tests": [],
        },
        {
            "id": "a",
            "question": "a",
            "starter_code": "",
            "fn_name": None,
            "tests": [],
        },
    ]
    (root / "data.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    items, loaded_manifest = load_fixed_items(tmp_path, item_ids=("a", "b"))

    assert [item.id for item in items] == ["a", "b"]
    assert loaded_manifest == manifest


def test_fixed_item_loader_rejects_revision_drift(tmp_path):
    root = tmp_path / "livecodebench"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"dataset": LIVE_CODE_BENCH_DATASET, "revision": "wrong"}),
        encoding="utf-8",
    )
    (root / "data.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="revision"):
        load_fixed_items(tmp_path, item_ids=("x",))


def test_committed_qwen32b_evidence_closes_every_gate():
    evidence = json.loads(RESULT.read_text(encoding="utf-8"))
    latency = evidence["latency"]
    quality = evidence["quality"]
    standard = quality["models"]["kairyu-auto"]
    maximum = quality["models"]["kairyu-auto-max"]

    assert evidence["passed"] is True
    assert evidence["served_models"] == [
        "kairyu-auto",
        "kairyu-auto-max",
        "qwen3-32b",
    ]
    assert max(latency["ratios"].values()) <= latency["config"]["max_ttft_ratio"]
    assert quality["all_routes_multi_agent"] is True
    assert maximum["summary"]["score"] > standard["summary"]["score"]
    assert maximum["summary"]["internal_calls"] == sum(
        sample["internal_calls"] for sample in maximum["samples"]
    )
    assert standard["summary"]["internal_calls"] == sum(
        sample["internal_calls"] for sample in standard["samples"]
    )
    assert all(sample["internal_calls"] == 4 for sample in maximum["samples"])
