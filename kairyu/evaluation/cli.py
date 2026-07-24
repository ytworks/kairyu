"""Non-destructive ``kairyu benchmark`` foundation commands."""

from __future__ import annotations

import argparse
import json

from kairyu.evaluation.registry import benchmark_catalog


def add_benchmark_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the evaluation CLI without importing benchmark adapters."""

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Inspect the reproducible evaluation-platform catalog.",
    )
    commands = benchmark.add_subparsers(dest="benchmark_command", required=True)

    list_command = commands.add_parser(
        "list",
        help="List the supported benchmark catalog and implementation status.",
    )
    list_command.add_argument(
        "--format",
        choices=("human", "json"),
        default="human",
        help="Output format (default: human).",
    )


def handle(args: argparse.Namespace) -> int:
    """Dispatch a parsed evaluation command."""

    if args.benchmark_command == "list":
        return _handle_list(args.format)
    raise ValueError(f"unknown benchmark command {args.benchmark_command!r}")


def _handle_list(output_format: str) -> int:
    entries = benchmark_catalog()
    if output_format == "json":
        payload = [entry.model_dump(mode="json") for entry in entries]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"evaluation benchmark catalog ({len(entries)} entries)")
    for entry in entries:
        print(
            f"  {entry.benchmark_id:25s} {entry.display_name} "
            f"— {entry.primary_metric} [{entry.implementation_status.value}]"
        )
    return 0
