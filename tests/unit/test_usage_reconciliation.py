import importlib.util
import json
from pathlib import Path

import pytest

from kairyu.usage_reconciliation import (
    UsageReconciliationError,
    aggregate_gateway_ledgers,
    aggregate_request_logs,
    reconcile_usage,
)


def _jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _load_replay_module():
    path = Path("bench/fleet_usage_replay.py")
    spec = importlib.util.spec_from_file_location("fleet_usage_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multi_gateway_ledger_aggregation_exports_cached_tokens(tmp_path):
    gateway_a = _jsonl(
        tmp_path / "a.jsonl",
        [
            {
                "tenant": "tenant-a",
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "cached_tokens": 8,
            },
            {
                "tenant": "tenant-b",
                "prompt_tokens": 5,
                "completion_tokens": 1,
                # Old rows without cached_tokens remain readable.
            },
        ],
    )
    gateway_b = _jsonl(
        tmp_path / "b.jsonl",
        [
            {
                "tenant": "tenant-a",
                "prompt_tokens": 20,
                "completion_tokens": 4,
                "cached_tokens": 16,
            }
        ],
    )

    assert aggregate_gateway_ledgers(
        {"gateway-a": gateway_a, "gateway-b": gateway_b}
    ) == {
        "tenant-a": {
            "requests": 2,
            "prompt_tokens": 30,
            "completion_tokens": 7,
            "cached_tokens": 24,
        },
        "tenant-b": {
            "requests": 1,
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "cached_tokens": 0,
        },
    }


def test_request_log_aggregation_counts_only_metered_execution_outcomes(tmp_path):
    path = _jsonl(
        tmp_path / "requests.jsonl",
        [
            {
                "gateway": "a",
                "event_id": "disconnect",
                "metered": True,
                "tenant": "tenant-a",
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "cached_tokens": 8,
            },
            {
                "gateway": "a",
                "event_id": "upstream-before-dispatch",
                "metered": False,
                "tenant": "tenant-a",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
            },
            {
                "gateway": "b",
                "event_id": "batch-output-rollback",
                "metered": True,
                "tenant": "tenant-b",
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "cached_tokens": 0,
            },
        ],
    )

    assert aggregate_request_logs([path]) == {
        "tenant-a": {
            "requests": 1,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "cached_tokens": 8,
        },
        "tenant-b": {
            "requests": 1,
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "cached_tokens": 0,
        },
    }


def test_reconciliation_reports_any_metric_outside_point_one_percent(tmp_path):
    ledger = _jsonl(
        tmp_path / "ledger.jsonl",
        [
            {
                "tenant": "tenant-a",
                "prompt_tokens": 999,
                "completion_tokens": 100,
                "cached_tokens": 200,
            }
        ],
    )
    requests = _jsonl(
        tmp_path / "requests.jsonl",
        [
            {
                "gateway": "a",
                "event_id": "one",
                "metered": True,
                "tenant": "tenant-a",
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "cached_tokens": 200,
            }
        ],
    )

    report = reconcile_usage(ledgers={"a": ledger}, request_logs=[requests])

    assert report["max_relative_error_percent"] == 0.1
    assert not report["passed"]  # The gate is strictly less than 0.1%.
    assert (
        report["tenants"]["tenant-a"]["relative_error_percent"]["prompt_tokens"]
        == 0.1
    )


def test_duplicate_gateway_events_and_shared_ledger_paths_fail_closed(tmp_path):
    request = {
        "gateway": "a",
        "event_id": "duplicate",
        "metered": False,
    }
    requests = _jsonl(tmp_path / "requests.jsonl", [request, request])
    with pytest.raises(UsageReconciliationError, match="duplicate gateway event"):
        aggregate_request_logs([requests])

    ledger = _jsonl(tmp_path / "ledger.jsonl", [])
    with pytest.raises(UsageReconciliationError, match="more than one gateway"):
        aggregate_gateway_ledgers({"a": ledger, "b": ledger})

    empty_requests = _jsonl(
        tmp_path / "empty-requests.jsonl",
        [{"gateway": "a", "event_id": "unmetered", "metered": False}],
    )
    with pytest.raises(UsageReconciliationError, match="no metered usage"):
        reconcile_usage(ledgers={"a": ledger}, request_logs=[empty_requests])


def test_fixed_replay_is_byte_reproducible_and_passes(tmp_path):
    replay = _load_replay_module()
    first = replay.generate_replay(tmp_path / "first", "2026-07-27")
    second = replay.generate_replay(tmp_path / "second", "2026-07-27")

    assert set(first) == set(second)
    for key in first:
        assert first[key].read_bytes() == second[key].read_bytes()
    report = json.loads(first["report"].read_text(encoding="utf-8"))
    assert report["passed"]
    assert report["max_relative_error_percent"] == 0
    assert report["gateway_count"] == 3
    assert report["replay"]["metered_events"] == 7
    assert report["tenants"]["tenant-a"]["ledger"]["cached_tokens"] == 96
    assert report["tenants"]["tenant-b"]["ledger"]["cached_tokens"] == 176
    assert report["tenants"]["tenant-c"]["ledger"]["cached_tokens"] == 192

    committed = Path("bench/results")
    for path in first.values():
        assert path.read_bytes() == (committed / path.name).read_bytes()
