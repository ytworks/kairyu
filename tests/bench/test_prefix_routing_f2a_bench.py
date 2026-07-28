"""Focused tests for the replayable F2a prefix-routing evidence gate."""

from __future__ import annotations

import itertools
import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from bench import prefix_routing_f2a_bench as f2a


@pytest.fixture(scope="module")
def smoke_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Path]:
    """Run the real 500-entry smoke path under a deterministic evidence clock."""

    output_dir = tmp_path_factory.mktemp("prefix-routing-f2a")
    ticks = itertools.count(start=1_000_000_000, step=10_000)
    original_clock = f2a.time.perf_counter_ns
    original_source = f2a.source_descriptor
    original_source_end = f2a.source_end_descriptor
    original_ancestor = f2a._git_is_ancestor
    original_blob_hash = f2a._git_blob_sha256
    source_hashes = {
        relative_path: f2a._sha256_file(f2a._REPO_ROOT / relative_path)
        for relative_path in f2a._SOURCE_PATHS
    }
    frozen_source = {
        "commit": "a" * 40,
        "expected_commit": None,
        "expected_commit_matches": True,
        "git_clean_at_start": True,
        "files": [
            {"path": path, "sha256": source_hashes[path]}
            for path in f2a._SOURCE_PATHS
        ],
    }
    f2a.time.perf_counter_ns = lambda: next(ticks)
    f2a.source_descriptor = lambda: frozen_source
    f2a.source_end_descriptor = lambda: {
        key: frozen_source[key]
        for key in (
            "commit",
            "expected_commit",
            "expected_commit_matches",
            "files",
        )
    }
    f2a._git_is_ancestor = lambda _commit: True
    f2a._git_blob_sha256 = (
        lambda _commit, relative_path: source_hashes[relative_path]
    )
    try:
        yield f2a.run_benchmark(output_dir, profile="smoke")
    finally:
        f2a.time.perf_counter_ns = original_clock
        f2a.source_descriptor = original_source
        f2a.source_end_descriptor = original_source_end
        f2a._git_is_ancestor = original_ancestor
        f2a._git_blob_sha256 = original_blob_hash


def test_trace_and_initial_cache_are_deterministic() -> None:
    first = f2a.profile_config("smoke")
    second = f2a.profile_config("smoke")

    assert first == second
    assert first.replicas == 500
    assert f2a.profile_config("formal").uniform_rounds == 21
    assert f2a.profile_config("formal").uniform_sign_min_rounds == 15
    assert f2a.cache_seed_rows(first) == f2a.cache_seed_rows(second)
    assert [
        (item.request_id, item.family_id, item.prompt_sha256, item.session_sha256)
        for item in f2a.shared_trace(first)
    ] == [
        (item.request_id, item.family_id, item.prompt_sha256, item.session_sha256)
        for item in f2a.shared_trace(second)
    ]
    assert [
        (item.request_id, item.family_id, item.prompt_sha256)
        for item in f2a.uniform_trace(first, 1)
    ] == [
        (item.request_id, item.family_id, item.prompt_sha256)
        for item in f2a.uniform_trace(second, 1)
    ]


def test_formal_context_requires_typed_github_identity() -> None:
    formal = f2a.profile_config("formal")
    source = {
        "commit": "a" * 40,
        "expected_commit": "a" * 40,
        "expected_commit_matches": True,
    }
    github = {
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_SHA": "B" * 40,
    }

    assert f2a._formal_context_complete(
        formal, source, {"github": github}
    )
    assert f2a._formal_context_complete(
        f2a.profile_config("smoke"), {}, {}
    )

    invalid_values = {
        "GITHUB_RUN_ID": ("0", "abc"),
        "GITHUB_RUN_ATTEMPT": ("0", "-1"),
        "GITHUB_EVENT_NAME": ("push", ""),
        "GITHUB_SHA": ("a" * 39, "g" * 40),
    }
    for field, values in invalid_values.items():
        for value in values:
            invalid_github = {**github, field: value}
            assert not f2a._formal_context_complete(
                formal,
                source,
                {"github": invalid_github},
            )


def test_independent_replay_accepts_the_frozen_smoke_artifact(
    smoke_artifact: Path,
) -> None:
    result = f2a.verify_artifact(smoke_artifact)

    assert result["passed"], result["errors"]
    assert result["checks"]["every_actual_selection_and_cache_hit_replays"]
    assert result["checks"]["shared_trace_is_identical_between_policies"]
    assert result["checks"]["uniform_traces_and_sessions_match_between_policies"]
    assert result["checks"]["paired_round_order_alternates"]
    assert result["checks"]["round_summaries_follow_emitted_arm_and_round_order"]
    assert result["checks"]["prompt_bodies_are_absent"]
    shared = result["metrics"]["shared"]
    assert shared["hit_rate_ratio"] >= 2.0
    assert (
        shared["prefix_prompt_chunks"]
        == shared["request_count_per_policy"] * 5
    )
    assert (
        shared["prefix_cached_prefix_chunks"]
        == shared["prefix_request_hits"] * 4
    )
    assert (
        shared["hrw_cached_prefix_chunks"]
        == shared["hrw_request_hits"] * 4
    )
    assert (
        result["metrics"]["uniform"]["paired_ratio_lcb95"]
        >= 0.99
    )


def test_replay_rejects_raw_tampering_even_if_attacker_updates_the_hash(
    smoke_artifact: Path, tmp_path: Path
) -> None:
    tampered_dir = tmp_path / "raw-tamper"
    shutil.copytree(smoke_artifact.parent, tampered_dir)
    raw_path = tampered_dir / f2a.RAW_NAME
    rows = f2a._read_jsonl(raw_path)
    placement = next(row for row in rows if row.get("type") == "placement")
    placement["selected_replica_id"] = "replica-499"
    f2a._write_jsonl(raw_path, rows)
    manifest_path = tampered_dir / f2a.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["raw"]["sha256"] = f2a._sha256_file(raw_path)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    result = f2a.verify_artifact(manifest_path)

    assert not result["passed"]
    assert not result["checks"]["every_actual_selection_and_cache_hit_replays"]


def test_replay_rejects_manifest_metric_tampering(
    smoke_artifact: Path, tmp_path: Path
) -> None:
    tampered_dir = tmp_path / "manifest-tamper"
    shutil.copytree(smoke_artifact.parent, tampered_dir)
    manifest_path = tampered_dir / f2a.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metrics"]["shared"]["prefix_hit_rate"] = 0.0
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    result = f2a.verify_artifact(manifest_path)

    assert not result["passed"]
    assert not result["checks"]["manifest_metrics_recomputed_exactly"]


@pytest.mark.parametrize(
    "tamper_kind",
    (
        "summary_request_count",
        "summary_wall",
        "summary_dispatch_sum",
        "summary_emitted_order",
        "global_summary_overlap",
        "arm_order",
        "source_drift",
    ),
)
def test_replay_rejects_summary_order_and_source_drift_tampering(
    smoke_artifact: Path,
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    tampered_dir = tmp_path / tamper_kind
    shutil.copytree(smoke_artifact.parent, tampered_dir)
    raw_path = tampered_dir / f2a.RAW_NAME
    manifest_path = tampered_dir / f2a.MANIFEST_NAME
    rows = f2a._read_jsonl(raw_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if tamper_kind == "summary_request_count":
        summary = next(row for row in rows if row.get("type") == "round_summary")
        summary["request_count"] = int(summary["request_count"]) + 1
    elif tamper_kind == "summary_wall":
        summary = next(row for row in rows if row.get("type") == "round_summary")
        summary["wall_started_ns"] = int(summary["wall_finished_ns"]) + 1
        summary["wall_ns"] = -1
    elif tamper_kind == "summary_dispatch_sum":
        summary = next(row for row in rows if row.get("type") == "round_summary")
        summary["dispatch_sum_ns"] = int(summary["dispatch_sum_ns"]) + 1
    elif tamper_kind == "summary_emitted_order":
        summary_indices = [
            index
            for index, row in enumerate(rows)
            if row.get("type") == "round_summary"
        ]
        first, second = summary_indices[:2]
        rows[first], rows[second] = rows[second], rows[first]
        for row_index, row in enumerate(rows):
            row["row_index"] = row_index
    elif tamper_kind == "global_summary_overlap":
        summaries = [
            row for row in rows if row.get("type") == "round_summary"
        ]
        previous, current = summaries[1:3]
        current["wall_started_ns"] = int(previous["wall_finished_ns"]) - 1
        current["wall_ns"] = (
            int(current["wall_finished_ns"])
            - int(current["wall_started_ns"])
        )
    elif tamper_kind == "arm_order":
        paired = next(row for row in rows if row.get("type") == "paired_round")
        paired["policy_order"] = list(reversed(paired["policy_order"]))
    else:
        run = next(row for row in rows if row.get("type") == "run")
        run["source_end"]["files"][0]["sha256"] = "f" * 64
        manifest["source_end"] = run["source_end"]

    f2a._write_jsonl(raw_path, rows)
    manifest["raw"]["sha256"] = f2a._sha256_file(raw_path)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    result = f2a.verify_artifact(manifest_path)

    assert not result["passed"]


def test_coordinated_round_wall_change_cannot_change_goodput(
    smoke_artifact: Path,
    tmp_path: Path,
) -> None:
    original = f2a.verify_artifact(smoke_artifact)
    tampered_dir = tmp_path / "audit-wall-only"
    shutil.copytree(smoke_artifact.parent, tampered_dir)
    raw_path = tampered_dir / f2a.RAW_NAME
    manifest_path = tampered_dir / f2a.MANIFEST_NAME
    rows = f2a._read_jsonl(raw_path)
    summary = next(
        row
        for row in rows
        if row.get("type") == "round_summary"
        and row.get("trace") == "uniform"
    )
    summary["wall_started_ns"] = int(summary["wall_started_ns"]) - 1
    summary["wall_ns"] = int(summary["wall_ns"]) + 1
    f2a._write_jsonl(raw_path, rows)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["raw"]["sha256"] = f2a._sha256_file(raw_path)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    result = f2a.verify_artifact(manifest_path)

    assert result["passed"], result["errors"]
    assert result["metrics"] == original["metrics"]


def test_slo_miss_stays_in_dispatch_goodput_denominator(
    smoke_artifact: Path,
) -> None:
    config = f2a.profile_config("smoke")
    rows = json.loads(
        json.dumps(f2a._read_jsonl(smoke_artifact.parent / f2a.RAW_NAME))
    )
    target = next(
        row
        for row in rows
        if row.get("type") == "placement"
        and row.get("trace") == "uniform"
        and row.get("round_index") == 0
        and row.get("policy") == f2a.PREFIX_POLICY
    )
    summary = next(
        row
        for row in rows
        if row.get("type") == "round_summary"
        and row.get("trace") == "uniform"
        and row.get("round_index") == 0
        and row.get("policy") == f2a.PREFIX_POLICY
    )
    old_latency_ns = int(target["placement_latency_ns"])
    target["placement_latency_ns"] = int(
        config.placement_p99_limit_ms * 1_000_000
    )
    summary["placement_sum_ns"] = (
        int(summary["placement_sum_ns"])
        + int(target["placement_latency_ns"])
        - old_latency_ns
    )
    summary["placement_slo_success_count"] = (
        int(summary["placement_slo_success_count"]) - 1
    )

    metrics = f2a.derive_metrics(rows, config)
    round_zero = metrics["uniform"]["paired_rounds"][0]
    successes = int(round_zero["prefix_slo_success_count"])
    all_dispatch_ns = int(round_zero["prefix_dispatch_sum_ns"])
    qualifying_dispatch_ns = sum(
        int(row["dispatch_latency_ns"])
        for row in rows
        if row.get("type") == "placement"
        and row.get("trace") == "uniform"
        and row.get("round_index") == 0
        and row.get("policy") == f2a.PREFIX_POLICY
        and int(row["placement_latency_ns"])
        < int(config.placement_p99_limit_ms * 1_000_000)
    )

    assert all_dispatch_ns > qualifying_dispatch_ns
    assert round_zero["prefix_placement_slo_goodput_rps"] == (
        successes * 1_000_000_000 / all_dispatch_ns
    )


def test_thresholds_use_exact_ratio_p99_and_one_percent_noninferiority() -> None:
    config = f2a.profile_config("smoke")
    passing = {
        "shared": {
            "hrw_request_hits": 1,
            "hit_rate_ratio": 2.0,
            "hit_rate_ratio_infinite": False,
        },
        "placement": {"prefix_worst_trace_p99_ms": 9.999999},
        "uniform": {
            "paired_ratio_lcb95": 0.99,
            "rounds_at_or_above_noninferiority_floor": 2,
        },
    }
    failing = {
        "shared": {
            "hrw_request_hits": 0,
            "hit_rate_ratio": 1.999999,
            "hit_rate_ratio_infinite": False,
        },
        "placement": {"prefix_worst_trace_p99_ms": 10.0},
        "uniform": {
            "paired_ratio_lcb95": 0.989999,
            "rounds_at_or_above_noninferiority_floor": 1,
        },
    }

    assert all(f2a.threshold_checks(passing, config).values())
    assert not any(f2a.threshold_checks(failing, config).values())
    assert config.uniform_equivalence_margin == 0.01
