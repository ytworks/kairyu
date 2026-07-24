"""Offline tests for the non-destructive ``kairyu benchmark list`` command."""

import argparse
import json
from dataclasses import replace

import pytest

import kairyu.evaluation.cli as evaluation_cli
from kairyu.entrypoints import cli
from kairyu.evaluation.adapters.gpqa_diamond import GPQADiamondAdapter
from kairyu.evaluation.cli import add_benchmark_parser, handle
from kairyu.evaluation.registry import BENCHMARK_IDS
from kairyu.evaluation.schemas import RunMode


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_benchmark_parser(subparsers)
    return parser.parse_args(["benchmark", *argv])


def test_human_list_is_ordered_and_marks_landed_adapter_available(capsys):
    assert handle(_parse(["list"])) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "evaluation benchmark catalog (11 entries)"
    assert [line.split()[0] for line in lines[1:]] == list(BENCHMARK_IDS)
    statuses = {line.split()[0]: line.rsplit("[", 1)[1].rstrip("]") for line in lines[1:]}
    assert statuses["gpqa-diamond"] == "available"
    assert statuses["humanitys-last-exam"] == "available"
    assert {
        status
        for benchmark_id, status in statuses.items()
        if benchmark_id not in {"gpqa-diamond", "humanitys-last-exam"}
    } == {"planned"}


def test_json_list_is_stable_and_machine_readable(capsys):
    assert handle(_parse(["list", "--format", "json"])) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [entry["benchmark_id"] for entry in payload] == list(BENCHMARK_IDS)
    statuses = {entry["benchmark_id"]: entry["implementation_status"] for entry in payload}
    assert statuses["gpqa-diamond"] == "available"
    assert statuses["humanitys-last-exam"] == "available"
    assert {
        status
        for benchmark_id, status in statuses.items()
        if benchmark_id not in {"gpqa-diamond", "humanitys-last-exam"}
    } == {"planned"}
    assert payload[0]["benchmark_id"] == "swe-bench-pro"
    assert payload[0]["display_name"] == "SWE-Bench Pro"
    assert payload[0]["primary_metric"] == "resolved rate"


def test_plan_exposes_truthful_estimates_resources_and_unresolved_evidence(capsys):
    assert handle(_parse(["plan", "gpqa-diamond", "--format", "json"])) == 0

    payload = json.loads(capsys.readouterr().out)
    preflight = payload["preflight"]
    assert payload["estimate"]["estimated_cost_usd"] is None
    assert payload["estimate"]["estimated_duration_seconds"] is None
    assert payload["estimate"]["maximum_duration_seconds"] == 240.0
    assert payload["effective_retries"] == 0
    assert payload["maximum_model_calls"] == 2
    assert payload["estimate"]["maximum_model_calls"] == 2
    assert payload["required_resources"]["cpu_cores"] == 1
    assert payload["required_resources"]["docker_required"] is False
    assert payload["execution"]["command"] == [
        "kairyu",
        "benchmark",
        "worker",
        "--once",
    ]
    assert preflight["problem_count"] == 2
    assert preflight["estimated_api_calls"] == 2
    assert preflight["effective_retries"] == 0
    assert preflight["maximum_api_calls"] == 2
    assert preflight["models"] == {
        "target": "kairyu-synthetic-model",
        "judge": None,
        "simulator": None,
    }
    assert preflight["cancellation_supported"] is True
    assert preflight["resume_supported"] is True
    assert preflight["official_eligible"] is False
    assert set(preflight["unresolved_reproducibility_evidence"]) == {
        "hardware_conditions",
        "provider_api_version",
        "runtime_dependency_environment",
        "source_retrieval_date",
    }


def test_hle_plan_cli_judge_reasoning_effort_changes_protocol_hash(capsys):
    assert (
        handle(
            _parse(
                [
                    "plan",
                    "humanitys-last-exam",
                    "--format",
                    "json",
                ]
            )
        )
        == 0
    )
    default_plan = json.loads(capsys.readouterr().out)

    assert (
        handle(
            _parse(
                [
                    "plan",
                    "humanitys-last-exam",
                    "--judge-reasoning-effort",
                    "high",
                    "--format",
                    "json",
                ]
            )
        )
        == 0
    )
    reasoned_plan = json.loads(capsys.readouterr().out)

    assert reasoned_plan["selected_item_ids"] == default_plan["selected_item_ids"]
    assert (
        reasoned_plan["item_input_manifest_sha256"] == (default_plan["item_input_manifest_sha256"])
    )
    assert reasoned_plan["protocol_hash"] != default_plan["protocol_hash"]


def test_plan_binds_openai_retries_and_exact_default_backoff(capsys):
    assert (
        handle(
            _parse(
                [
                    "plan",
                    "gpqa-diamond",
                    "--connector",
                    "openai",
                    "--endpoint",
                    "http://127.0.0.1:18080",
                    "--max-retries",
                    "2",
                    "--format",
                    "json",
                ]
            )
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["estimated_model_calls"] == 2
    assert payload["effective_retries"] == 2
    assert payload["maximum_model_calls"] == 6
    assert payload["estimate"]["maximum_model_calls"] == 6
    assert payload["estimate"]["maximum_duration_seconds"] == 723.0
    assert payload["preflight"]["estimated_api_calls"] == 2
    assert payload["preflight"]["maximum_api_calls"] == 6


def test_full_run_prints_preflight_before_service_submit(
    tmp_path,
    monkeypatch,
    capsys,
):
    smoke_plan = GPQADiamondAdapter().build_run_plan(
        evaluation_cli.RunSelection(
            profile="smoke",
            mode=RunMode.SMOKE,
            target_model="preflight-model",
        ),
        environ={},
    )
    full_preview = replace(
        smoke_plan,
        profile="official-latest",
        mode=RunMode.FULL,
    )
    real_metadata = GPQADiamondAdapter().metadata()

    class FakeAdapter:
        def build_run_plan(self, _selection):
            return full_preview

        def metadata(self):
            return real_metadata

    observed = {}

    class StopBeforeEnqueueService:
        def __init__(self, _runtime):
            pass

        def submit(self, *_args, **_kwargs):
            observed["stderr_before_submit"] = capsys.readouterr().err
            raise RuntimeError("stop before enqueue")

    import kairyu.evaluation.adapters as adapters

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("BENCHMARK_ALLOW_FULL_RUN", "1")
    monkeypatch.setattr(adapters, "get_adapter", lambda _benchmark_id: FakeAdapter())
    monkeypatch.setattr(evaluation_cli, "BenchmarkService", StopBeforeEnqueueService)

    with pytest.raises(RuntimeError, match="stop before enqueue"):
        handle(
            _parse(
                [
                    "run",
                    "gpqa-diamond",
                    "--profile",
                    "official-latest",
                    "--mode",
                    "full",
                    "--confirm-full-run",
                    "--connector",
                    "openai",
                    "--endpoint",
                    "http://127.0.0.1:18080",
                    "--max-retries",
                    "2",
                    "--state-dir",
                    str(tmp_path / "state"),
                    "--format",
                    "json",
                ]
            )
        )

    envelope = json.loads(observed["stderr_before_submit"])
    preflight = envelope["full_run_preflight"]
    assert preflight["mode"] == "full"
    assert preflight["problem_count"] == 2
    assert preflight["effective_retries"] == 2
    assert preflight["estimated_api_calls"] == 2
    assert preflight["maximum_api_calls"] == 6
    assert preflight["maximum_duration_seconds"] == 723.0
    assert preflight["estimated_cost_usd"] is None
    assert preflight["data_licenses"]
    assert preflight["score_claim"].startswith("Unofficial")


def test_console_entrypoint_exposes_benchmark_list(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["benchmark", "list", "--format", "json"])

    assert exc.value.code == 0
    assert len(json.loads(capsys.readouterr().out)) == 11


def test_unimplemented_lifecycle_commands_are_not_exposed():
    with pytest.raises(SystemExit) as exc:
        _parse(["run"])

    assert exc.value.code == 2
