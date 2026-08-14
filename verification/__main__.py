"""Registry-driven checkout CLI for Kairyu verification gates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from evidence.formal import load_correctness_reference
from evidence.reporting import atomic_write_json
from verification.registry import (
    VerificationGate,
    load_registry,
    registry_payload,
    validate_repository,
)

ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m verification",
        description="List, validate, and run checkout-only verification gates.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="list registered gates")
    list_parser.add_argument("--json", action="store_true", help="emit registry JSON")
    list_parser.add_argument(
        "--scope", choices=("l1", "orchestration", "fleet", "product")
    )
    list_parser.add_argument(
        "--kind",
        choices=("correctness", "performance", "resilience", "diagnostic"),
    )

    check_parser = commands.add_parser("check", help="validate the source checkout")
    check_parser.add_argument("--root", type=Path, default=ROOT)

    run_parser = commands.add_parser("run", help="run one registered gate")
    run_parser.add_argument("gate_id")
    run_parser.add_argument(
        "--correctness-artifact",
        type=Path,
        help="accepted correctness record required by formal performance gates",
    )
    run_parser.add_argument(
        "--record",
        type=Path,
        help="execution record path (default: verification/results/executions/...)",
    )
    run_parser.add_argument(
        "gate_args",
        nargs="*",
        help="arguments passed to the gate; place them after --",
    )
    return parser


def _select_gate(gate_id: str) -> VerificationGate:
    by_id = {entry.id: entry for entry in load_registry()}
    try:
        return by_id[gate_id]
    except KeyError as error:
        raise ValueError(f"unknown verification gate: {gate_id}") from error


def _list(args: argparse.Namespace) -> int:
    entries = [
        entry
        for entry in load_registry()
        if (args.scope is None or entry.scope == args.scope)
        and (args.kind is None or entry.kind == args.kind)
    ]
    if args.json:
        payload = registry_payload()
        selected = {entry.id for entry in entries}
        payload["gates"] = [
            gate for gate in payload["gates"] if gate["id"] in selected
        ]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    for entry in entries:
        formal = "formal" if entry.formal else entry.entrypoint_class
        print(f"{entry.id}\t{entry.scope}\t{entry.kind}\t{formal}\t{entry.path}")
    return 0


def _check(args: argparse.Namespace) -> int:
    errors = validate_repository(args.root)
    if not errors:
        print("verification registry: OK")
        return 0
    print("verification registry: FAILED", file=sys.stderr)
    for error in errors:
        print(f"- {error}", file=sys.stderr)
    return 1


def _record_path(entry: VerificationGate, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return ROOT / "verification/results/executions" / f"{timestamp}-{entry.id}.json"


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        entry = _select_gate(args.gate_id)
    except ValueError as error:
        parser.error(str(error))
    correctness = None
    if entry.formal and entry.kind == "performance":
        if args.correctness_artifact is None:
            parser.error(
                "formal performance gates require --correctness-artifact pointing "
                "to accepted correctness evidence"
            )
        try:
            correctness = load_correctness_reference(args.correctness_artifact)
        except ValueError as error:
            parser.error(str(error))
    elif args.correctness_artifact is not None:
        parser.error("--correctness-artifact is valid only for formal performance gates")

    gate_args = list(args.gate_args)
    if gate_args[:1] == ["--"]:
        gate_args = gate_args[1:]
    command = [sys.executable, "-m", entry.module, *gate_args]
    started_at = datetime.now(tz=UTC).isoformat()
    environment = os.environ.copy()
    if correctness is not None:
        environment["KAIRYU_CORRECTNESS_GATE_ID"] = correctness.gate_id
        environment["KAIRYU_CORRECTNESS_SHA256"] = correctness.sha256
        environment["KAIRYU_CORRECTNESS_ARTIFACT"] = correctness.artifact_path
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    finished_at = datetime.now(tz=UTC).isoformat()

    if entry.formal:
        record: dict[str, object] = {
            "schema_version": 1,
            "gate_id": entry.id,
            "scope": entry.scope,
            "kind": entry.kind,
            "verdict": "pass" if completed.returncode == 0 else "fail",
            "started_at": started_at,
            "finished_at": finished_at,
            "command": command,
            "returncode": completed.returncode,
            "output_schema": entry.output_schema,
        }
        if correctness is not None:
            record["correctness_reference"] = correctness.as_mapping()
        path = _record_path(entry, args.record)
        atomic_write_json(
            path,
            record,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        print(f"verification execution record: {path}")
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        return _list(args)
    if args.command == "check":
        return _check(args)
    if args.command == "run":
        return _run(args, parser)
    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
