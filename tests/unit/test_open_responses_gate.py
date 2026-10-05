"""Verdict of the Open Responses gate (scripts/open_responses_gate/verdict.py).

The live gate (bun, the pinned suite, Kairyu) runs only in CI and today
produces no failure beyond its listed ones, so it cannot show that a listed
scenario which fails for an additional reason turns the gate red. This test
feeds the verdict a list file and a synthetic suite report instead.
"""

from __future__ import annotations

from pathlib import Path

from scripts.open_responses_gate.verdict import evaluate, load_gate_file

_ENVELOPE = ["completed_at: Invalid input", "store: Required"]
_GATE_FILE = """
[suite]
repository = "openresponses/openresponses"
commit = "92c12d96d7b61d6d15e2214daa5e9c6000ab6e1c"
spec = "2026-04-24"
{entries}
"""
_ENTRY = """
[[expected_failure]]
id = "{id}"
gap_id = "G-output-object-1"
owner_wp = "WP-08b"
errors = {errors}
{body}reason = "listed"
"""


def _entry(scenario: str, errors: list[str], body: str = "") -> str:
    body_line = f'body_contains = "{body}"\n' if body else ""
    quoted = "[" + ", ".join(f'"{error}"' for error in errors) + "]"
    return _ENTRY.format(id=scenario, errors=quoted, body=body_line)


def _failed(scenario: str, errors: list[str], response: object = None) -> dict[str, object]:
    return {"id": scenario, "status": "failed", "errors": errors, "response": response}


def test_listed_failure_is_xfail_only_with_exactly_its_listed_errors(tmp_path: Path) -> None:
    # Arrange
    rejection = ["HTTP 400: [object Object]"]
    listed = {
        "same": _ENVELOPE,
        "extra-error": _ENVELOPE,
        "fewer-errors": _ENVELOPE,
        "same-body": rejection,
        "other-body": rejection,
    }
    entries = "".join(
        _entry(scenario, errors, "'input_image' is not supported" if "body" in scenario else "")
        for scenario, errors in listed.items()
    )
    gate_file = tmp_path / "expected-failures.toml"
    gate_file.write_text(_GATE_FILE.format(entries=entries), encoding="utf-8")
    image_error = {"error": {"message": "'input_image' is not supported"}}
    results = [
        _failed("same", list(reversed(_ENVELOPE))),
        _failed("extra-error", [*_ENVELOPE, "output.0.call_id: Required"]),
        _failed("fewer-errors", _ENVELOPE[:1]),
        _failed("same-body", rejection, image_error),
        _failed("other-body", rejection, {"error": {"message": "input: Required"}}),
    ]
    report = {
        "summary": {"passed": 0, "failed": len(results), "skipped": 0, "total": len(results)},
        "results": results,
    }

    # Act
    verdicts = evaluate(report, load_gate_file(gate_file).expected)

    # Assert
    assert {verdict.id: verdict.outcome for verdict in verdicts} == {
        "same": "XFAIL",
        "extra-error": "FAIL",
        "fewer-errors": "FAIL",
        "same-body": "XFAIL",
        "other-body": "FAIL",
    }
