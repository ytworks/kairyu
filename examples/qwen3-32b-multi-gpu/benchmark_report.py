"""Aggregate Qwen3-32B serving benchmark JSON files into a Markdown report."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _display(value: Any) -> str:
    if value is None:
        return "-"
    return str(value).replace("|", r"\|")


def _load_runs(
    results_dir: Path,
) -> tuple[list[tuple[Path, dict[str, Any], dict[str, Any]]], list[str]]:
    runs = []
    rejected = []
    for path in sorted(results_dir.glob("*-serving.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            config = payload.get("config")
            summary = payload.get("summary")
            if not isinstance(config, dict) or not isinstance(summary, dict):
                raise ValueError("expected object fields 'config' and 'summary'")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            rejected.append(f"{path.name}: {error}")
            continue
        runs.append((path, config, summary))
    if not runs:
        raise ValueError(f"no *-serving.json files found in {results_dir}")
    return runs, rejected


def build_report(results_dir: Path) -> str:
    runs, rejected = _load_runs(results_dir)
    lines = [
        "# Qwen3-32B serving benchmark",
        "",
        f"Generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "",
        (
            "| Run | GPUs / TP | Dataset | Requests | Concurrency | Max tokens | "
            "TTFT SLO (s) | Wall (s) | "
            "TTFT p50 (ms) | TTFT p99 (ms) | TPOT mean (ms/token) | "
            "TPOT method | Throughput (req/s) | Goodput (req/s) |"
        ),
        (
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---|---:|---:|"
        ),
    ]
    for path, config, summary in runs:
        lines.append(
            "| "
            + " | ".join(
                _display(value)
                for value in (
                    path.stem.removesuffix("-serving"),
                    config.get("tensor_parallel"),
                    summary.get("dataset"),
                    summary.get("requests"),
                    config.get("concurrency"),
                    config.get("max_tokens"),
                    config.get("ttft_slo_s"),
                    summary.get("wall_s"),
                    summary.get("ttft_p50_ms"),
                    summary.get("ttft_p99_ms"),
                    summary.get("tpot_mean_ms"),
                    summary.get("tpot_method"),
                    summary.get("throughput_rps"),
                    summary.get("goodput_rps"),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "Goodput counts requests whose TTFT met the run's SLO. "
                "Source: the timestamped `*-serving.json` files in this directory."
            ),
            "",
        ]
    )
    if rejected:
        lines.extend(
            [
                "## Skipped result files",
                "",
                *(f"- `{message}`" for message in rejected),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
