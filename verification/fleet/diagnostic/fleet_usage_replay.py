#!/usr/bin/env python3
"""Generate and reconcile the fixed G5 F5e fleet usage replay.

The request log is the independent observation source.  Each gateway ledger is
written separately, matching production ownership; aggregation happens only
after every raw input is closed.

Run:
    uv run python verification/fleet/diagnostic/fleet_usage_replay.py \
      --date 2026-07-27 --output-dir bench/results
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kairyu.usage_reconciliation import reconcile_usage

_GATEWAYS = ("gateway-a", "gateway-b", "gateway-c")
_TRACE: tuple[dict[str, Any], ...] = (
    {
        "gateway": "gateway-a",
        "event_id": "chat-sync-001",
        "endpoint": "/v1/chat/completions",
        "outcome": "success",
        "tenant": "tenant-a",
        "model": "qwen",
        "metered": True,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "cached_tokens": 0,
    },
    {
        "gateway": "gateway-a",
        "event_id": "chat-stream-002",
        "endpoint": "/v1/chat/completions",
        "outcome": "client_disconnect_after_dispatch",
        "tenant": "tenant-a",
        "model": "qwen",
        "metered": True,
        "prompt_tokens": 80,
        "completion_tokens": 5,
        "cached_tokens": 64,
    },
    {
        "gateway": "gateway-a",
        "event_id": "responses-003",
        "endpoint": "/v1/responses",
        "outcome": "upstream_failure_before_result",
        "tenant": "tenant-b",
        "model": "qwen",
        "metered": False,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
    },
    {
        "gateway": "gateway-b",
        "event_id": "completions-stream-004",
        "endpoint": "/v1/completions",
        "outcome": "upstream_failure_after_dispatch",
        "tenant": "tenant-a",
        "model": "qwen",
        "metered": True,
        "prompt_tokens": 60,
        "completion_tokens": 3,
        "cached_tokens": 32,
    },
    {
        "gateway": "gateway-b",
        "event_id": "embeddings-005",
        "endpoint": "/v1/embeddings",
        "outcome": "success",
        "tenant": "tenant-b",
        "model": "embed",
        "metered": True,
        "prompt_tokens": 12,
        "completion_tokens": 0,
        "cached_tokens": 0,
    },
    {
        "gateway": "gateway-b",
        "event_id": "batch-006",
        "endpoint": "/v1/batches",
        "outcome": "output_rollback_after_execution",
        "tenant": "tenant-b",
        "model": "qwen",
        "metered": True,
        "prompt_tokens": 200,
        "completion_tokens": 40,
        "cached_tokens": 128,
    },
    {
        "gateway": "gateway-c",
        "event_id": "responses-007",
        "endpoint": "/v1/responses",
        "outcome": "success",
        "tenant": "tenant-b",
        "model": "qwen",
        "metered": True,
        "prompt_tokens": 90,
        "completion_tokens": 10,
        "cached_tokens": 48,
    },
    {
        "gateway": "gateway-c",
        "event_id": "orchestrated-chat-008",
        "endpoint": "/v1/chat/completions",
        "outcome": "success",
        "tenant": "tenant-c",
        "model": "kairyu-auto",
        "metered": True,
        "prompt_tokens": 300,
        "completion_tokens": 50,
        "cached_tokens": 192,
    },
    {
        "gateway": "gateway-c",
        "event_id": "batch-009",
        "endpoint": "/v1/batches",
        "outcome": "input_rollback_before_execution",
        "tenant": "tenant-c",
        "model": "qwen",
        "metered": False,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
    },
)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    path.write_text(payload, encoding="utf-8")


def generate_replay(output_dir: Path, date: str) -> dict[str, Path]:
    """Write deterministic raw logs and the reconciliation report."""
    replay_epoch = int(
        datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC).timestamp()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    request_log = output_dir / f"fleet-usage-request-log-{date}.jsonl"
    request_records = [
        {
            **record,
            "uncached_tokens": (
                record["prompt_tokens"] - record["cached_tokens"]
            ),
            "observed_at": f"{date}T00:00:{index:02d}Z",
        }
        for index, record in enumerate(_TRACE, start=1)
    ]
    _write_jsonl(request_log, request_records)

    ledgers: dict[str, Path] = {}
    for gateway in _GATEWAYS:
        path = output_dir / f"fleet-usage-ledger-{gateway}-{date}.jsonl"
        records = [
            {
                "tenant": record["tenant"],
                "model": record["model"],
                "prompt_tokens": record["prompt_tokens"],
                "completion_tokens": record["completion_tokens"],
                "cached_tokens": record["cached_tokens"],
                "uncached_tokens": (
                    record["prompt_tokens"] - record["cached_tokens"]
                ),
                "ts": replay_epoch + index,
            }
            for index, record in enumerate(_TRACE, start=1)
            if record["gateway"] == gateway and record["metered"]
        ]
        _write_jsonl(path, records)
        ledgers[gateway] = path

    report = reconcile_usage(ledgers=ledgers, request_logs=[request_log])
    report["replay"] = {
        "date": date,
        "events": len(_TRACE),
        "metered_events": sum(record["metered"] for record in _TRACE),
        "coverage": [
            "batch_rollback",
            "cached_tokens",
            "client_disconnect",
            "cross_endpoint",
            "multi_gateway",
            "upstream_failure",
        ],
    }
    report_path = output_dir / f"fleet-usage-reconciliation-{date}.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "request_log": request_log,
        "report": report_path,
        **{gateway: path for gateway, path in ledgers.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    args = parser.parse_args()
    paths = generate_replay(args.output_dir, args.date)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    print(json.dumps({"paths": {key: str(value) for key, value in paths.items()}, **report}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
