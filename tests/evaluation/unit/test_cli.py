"""Offline tests for the non-destructive ``kairyu benchmark list`` command."""

import argparse
import json

import pytest

from kairyu.entrypoints import cli
from kairyu.evaluation.cli import add_benchmark_parser, handle
from kairyu.evaluation.registry import BENCHMARK_IDS


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_benchmark_parser(subparsers)
    return parser.parse_args(["benchmark", *argv])


def test_human_list_is_ordered_and_marks_every_entry_planned(capsys):
    assert handle(_parse(["list"])) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "evaluation benchmark catalog (11 entries)"
    assert [line.split()[0] for line in lines[1:]] == list(BENCHMARK_IDS)
    assert all(line.endswith("[planned]") for line in lines[1:])


def test_json_list_is_stable_and_machine_readable(capsys):
    assert handle(_parse(["list", "--format", "json"])) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [entry["benchmark_id"] for entry in payload] == list(BENCHMARK_IDS)
    assert all(entry["implementation_status"] == "planned" for entry in payload)
    assert payload[0]["benchmark_id"] == "swe-bench-pro"
    assert payload[0]["display_name"] == "SWE-Bench Pro"
    assert payload[0]["primary_metric"] == "resolved rate"


def test_console_entrypoint_exposes_benchmark_list(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["benchmark", "list", "--format", "json"])

    assert exc.value.code == 0
    assert len(json.loads(capsys.readouterr().out)) == 11


def test_unimplemented_lifecycle_commands_are_not_exposed():
    with pytest.raises(SystemExit) as exc:
        _parse(["run"])

    assert exc.value.code == 2
