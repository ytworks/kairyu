"""Verification registry, boundary, and runner contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verification import __main__ as cli
from verification.registry import (
    VerificationGate,
    load_compatibility_imports,
    load_registry,
    registry_payload,
    validate_repository,
)

ROOT = Path(__file__).resolve().parents[2]


def _gate_mapping(**overrides):
    value = {
        "id": "l1.correctness.example",
        "scope": "l1",
        "kind": "correctness",
        "path": "verification/l1/correctness/example.py",
        "module": "verification.l1.correctness.example",
        "output_schema": "entrypoint-defined",
        "entrypoint_class": "operator",
        "formal": False,
        "requires": ["core"],
        "documentation": ["verification/README.md"],
        "installed": False,
        "invocations": ["path", "module"],
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"scope": "evaluation"}, "unsupported scope"),
        ({"kind": "accuracy"}, "unsupported kind"),
        ({"installed": True}, "not installed"),
        ({"formal": True}, "formal must match"),
        ({"invocations": ["module"]}, "path, optionally followed"),
        ({"unknown": "field"}, "unknown unknown"),
    ),
)
def test_gate_mapping_is_closed_and_classified(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        VerificationGate.from_mapping(_gate_mapping(**overrides))


def test_registry_inventories_all_checkout_entrypoints() -> None:
    entries = load_registry()
    assert len(entries) == 68
    assert len({entry.id for entry in entries}) == len(entries)
    assert len({entry.path for entry in entries}) == len(entries)
    assert {entry.scope for entry in entries} == {"l1", "orchestration", "fleet", "product"}
    assert {entry.kind for entry in entries} == {
        "correctness",
        "performance",
        "resilience",
        "diagnostic",
    }
    assert any(entry.scope == "l1" and entry.kind == "correctness" for entry in entries)
    assert any(entry.scope == "l1" and entry.kind == "performance" for entry in entries)
    assert all(entry.installed is False for entry in entries)


def test_source_checkout_satisfies_registry_and_import_boundaries() -> None:
    assert validate_repository(ROOT) == ()


def test_registry_payload_is_explicit_and_json_safe() -> None:
    payload = json.loads(json.dumps(registry_payload(), sort_keys=True))
    assert payload["schema_version"] == 1
    assert payload["ownership"] == {
        "installed": False,
        "legacy_results": "bench/results",
        "verification": "verification",
    }
    assert len(payload["gates"]) == 68
    assert payload["compatibility_imports"] == {
        source: list(targets) for source, targets in load_compatibility_imports().items()
    }


def test_formal_performance_requires_accepted_correctness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0)
    )
    with pytest.raises(SystemExit) as error:
        cli.main(["run", "l1.performance.dp_scaling_g2_a8_bench"])
    assert error.value.code == 2

    correctness = tmp_path / "correctness.json"
    correctness.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gate_id": "l1.correctness.anchor",
                "kind": "correctness",
                "verdict": "pass",
            }
        ),
        encoding="utf-8",
    )
    record = tmp_path / "performance.json"
    assert (
        cli.main(
            [
                "run",
                "l1.performance.dp_scaling_g2_a8_bench",
                "--correctness-artifact",
                str(correctness),
                "--record",
                str(record),
            ]
        )
        == 0
    )
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["kind"] == "performance"
    assert payload["verdict"] == "pass"
    assert payload["correctness_reference"]["gate_id"] == "l1.correctness.anchor"
    assert len(payload["correctness_reference"]["sha256"]) == 64


def test_list_json_filters_without_losing_registry_identity(capsys) -> None:
    assert cli.main(["list", "--scope", "l1", "--kind", "correctness", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gates"]
    assert all(gate["scope"] == "l1" for gate in payload["gates"])
    assert all(gate["kind"] == "correctness" for gate in payload["gates"])
