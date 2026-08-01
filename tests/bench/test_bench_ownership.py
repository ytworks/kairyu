"""Package-owned benchmark entrypoint and checkout ownership contract."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest

from kairyu.bench import cli as bench_cli
from kairyu.bench.ownership import (
    MANIFEST_NAME,
    _top_level_bench_import_names,
    _top_level_bench_imports,
    entrypoints_payload,
    load_compatibility_imports,
    load_entrypoints,
    validate_repository,
)
from kairyu.entrypoints import cli

ROOT = Path(__file__).resolve().parents[2]


def test_package_manifest_is_complete_and_sorted() -> None:
    manifest = resources.files("kairyu.bench").joinpath(MANIFEST_NAME)
    assert manifest.is_file()
    entries = load_entrypoints()
    assert len(entries) == 57
    assert [entry.path for entry in entries] == sorted(
        entry.path for entry in entries
    )
    assert all(not entry.installed for entry in entries)
    assert all(set(entry.invocations) == {"path", "module"} for entry in entries)


def test_source_checkout_satisfies_ownership_contract() -> None:
    assert validate_repository(ROOT) == ()


def test_retained_cross_wrapper_imports_are_an_exact_allowlist() -> None:
    assert load_compatibility_imports() == {
        "bench.batched_spec_verify_qwen": ("bench.batched_prefill_qwen",),
        "bench.decode_page_table_cache_qwen": ("bench.batched_prefill_qwen",),
        "bench.fleet_rollout_bench": ("bench.fleet_churn_bench",),
        "bench.g2_a6_vllm_bench": (
            "bench.multiturn_prefix",
            "bench.tp_kv_hit_g2_a7_bench",
        ),
        "bench.g2_a9_dp_tp_crossover_bench": (
            "bench.dp_scaling_g2_a8_bench",
        ),
        "bench.gate_a1": ("bench.parity_tp",),
        "bench.gate_a2": ("bench.parity_hf", "bench.parity_tp"),
        "bench.global_kv_pool_decision": ("bench.kv_aware_ttft_f2c_bench",),
        "bench.kv_event_f2b_bench": ("bench.fleet_churn_bench",),
        "bench.parity_hf": ("bench.parity_tp",),
        "bench.pd_overlap_qwen": ("bench.parity_tp",),
        "bench.prefix_weight_f2d_bench": ("bench.prefix_routing_f2a_bench",),
        "bench.tiered_auto_bench": ("bench.orchestration_stream_bench",),
        "bench.tp_kv_hit_g2_a7_bench": ("bench.multiturn_prefix",),
        "bench.tp_sampling_owner_qwen": ("bench.parity_tp",),
    }


def test_bare_sibling_import_cannot_bypass_wrapper_allowlist(tmp_path: Path) -> None:
    source = tmp_path / "wrapper.py"
    source.write_text(
        "import parity_tp\nfrom fleet_churn_bench import workload\n",
        encoding="utf-8",
    )

    assert _top_level_bench_import_names(
        source,
        wrapper_stems=frozenset({"fleet_churn_bench", "parity_tp"}),
    ) == ["bench.fleet_churn_bench", "bench.parity_tp"]
    assert _top_level_bench_imports(
        source,
        wrapper_stems=frozenset({"fleet_churn_bench", "parity_tp"}),
    ) == [
        f"{source.as_posix()} -> bench.fleet_churn_bench",
        f"{source.as_posix()} -> bench.parity_tp",
    ]


def test_entrypoints_json_is_stable_and_explicit() -> None:
    payload = json.loads(json.dumps(entrypoints_payload(), sort_keys=True))
    assert payload["schema_version"] == 1
    assert payload["ownership"] == {
        "installed_cli": "kairyu bench",
        "installed_fixtures": "kairyu.bench.fixtures",
        "repository_entrypoints": "bench",
        "repository_results": "bench/results",
        "reusable_code": "kairyu.bench",
    }
    assert payload["compatibility_imports"] == {
        source: list(targets)
        for source, targets in load_compatibility_imports().items()
    }
    assert len(payload["entrypoints"]) == 57


def test_bench_entrypoints_parser_dispatches(capsys) -> None:
    args = cli._build_parser().parse_args(["bench", "entrypoints", "--json"])
    assert args.command == "bench"
    assert args.bench_command == "entrypoints"
    assert bench_cli.handle(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["entrypoints"]) == 57


def test_bench_entrypoints_check_failure_is_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "kairyu.bench.ownership.validate_repository",
        lambda root: ("unregistered benchmark entrypoint: bench/new.py",),
    )
    args = cli._build_parser().parse_args(
        ["bench", "entrypoints", "--check-repo", str(ROOT)]
    )
    assert bench_cli.handle(args) == 1
    output = capsys.readouterr().out
    assert "repository check: FAILED" in output
    assert "bench/new.py" in output
