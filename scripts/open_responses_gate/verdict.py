"""Expected failures and verdicts of the Open Responses gate (M20 WP-14, D1).

``tests/contracts/openresponses/expected-failures.toml`` pins the suite
(``[suite]``: repository, full commit, spec version) and lists every scenario
expected to fail at that pin, each with the gap it tracks, the work package
that makes it pass, the reason, and a ``signature``: text that the scenario's
errors or error body contain while it fails for that reason.

Per scenario of a ``bin/compliance-test.ts --json`` report:

- ``PASS``: passed and not listed;
- ``XFAIL``: failed, listed, and its failure contains the signature;
- ``FAIL``: failed unlisted, failed listed for another reason, or skipped;
- ``XPASS``: passed although listed, so the owning WP must delete the entry;
- ``MISSING``: listed but absent from the report (the pin moved).

Only ``PASS`` and ``XFAIL`` keep the gate green, so the list stays truthful.
The file is validated when loaded; a malformed list fails the gate.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.contracts.divergences import GAP_ID_PATTERN, OWNER_WP_PATTERN

EXPECTED_FILE = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "contracts"
    / "openresponses"
    / "expected-failures.toml"
)
GREEN = frozenset({"PASS", "XFAIL"})
_ENTRY_FIELDS = ("id", "gap_id", "owner_wp", "signature", "reason")
_SUITE_FIELDS = ("repository", "commit", "spec")
_SCENARIO_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9-]+/[A-Za-z0-9._-]+$")
_SPEC_VERSION = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DETAIL_CHARS = 300


class GateFileError(ValueError):
    """``expected-failures.toml`` or a suite report violates its contract."""


@dataclass(frozen=True)
class Suite:
    repository: str
    commit: str
    spec: str


@dataclass(frozen=True)
class ExpectedFailure:
    id: str
    gap_id: str
    owner_wp: str
    signature: str
    reason: str


@dataclass(frozen=True)
class GateFile:
    suite: Suite
    expected: tuple[ExpectedFailure, ...]


@dataclass(frozen=True)
class Verdict:
    id: str
    outcome: str
    detail: str


def _fields(raw: Mapping[str, Any], names: Sequence[str], where: str) -> dict[str, str]:
    missing = [name for name in names if name not in raw]
    unknown = sorted(set(raw) - set(names))
    blank = [name for name in names if not str(raw.get(name, "")).strip()]
    if missing or unknown or blank:
        raise GateFileError(f"{where}: missing {missing}, unknown {unknown}, empty {blank}")
    if not all(isinstance(raw[name], str) for name in names):
        raise GateFileError(f"{where}: every field must be a string")
    return {name: raw[name] for name in names}


def _suite(raw: Any) -> Suite:
    if not isinstance(raw, Mapping):
        raise GateFileError("[suite] table is required")
    fields = _fields(raw, _SUITE_FIELDS, "[suite]")
    checks = {
        "repository": _REPOSITORY.match(fields["repository"]),
        "commit": _FULL_SHA.match(fields["commit"]),
        "spec": _SPEC_VERSION.match(fields["spec"]),
    }
    invalid = [name for name, ok in checks.items() if not ok]
    if invalid:
        raise GateFileError(f"[suite]: invalid {invalid}")
    return Suite(**fields)


def _entry(raw: Any, index: int) -> ExpectedFailure:
    if not isinstance(raw, Mapping):
        raise GateFileError(f"expected_failure #{index} must be a table")
    where = f"expected_failure #{index} ({raw.get('id', '?')})"
    fields = _fields(raw, _ENTRY_FIELDS, where)
    checks = {
        "id": _SCENARIO_ID.match(fields["id"]),
        "gap_id": GAP_ID_PATTERN.match(fields["gap_id"]),
        "owner_wp": OWNER_WP_PATTERN.match(fields["owner_wp"]),
    }
    invalid = [name for name, ok in checks.items() if not ok]
    if invalid:
        raise GateFileError(f"{where}: invalid {invalid}")
    return ExpectedFailure(**fields)


def load_gate_file(path: Path = EXPECTED_FILE) -> GateFile:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(raw) - {"suite", "expected_failure"})
    if unknown:
        raise GateFileError(f"{path.name}: unknown top-level keys {unknown}")
    listed = raw.get("expected_failure", [])
    entries = tuple(_entry(item, index) for index, item in enumerate(listed))
    ids = [entry.id for entry in entries]
    duplicates = sorted({entry_id for entry_id in ids if ids.count(entry_id) > 1})
    if duplicates:
        raise GateFileError(f"duplicate expected_failure ids: {duplicates}")
    return GateFile(suite=_suite(raw.get("suite")), expected=entries)


def _results(report: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """The report's results, checked against its own summary."""

    results = report.get("results")
    summary = report.get("summary")
    if not isinstance(results, list) or not isinstance(summary, Mapping):
        raise GateFileError("suite report lacks results or summary")
    if not all(isinstance(item, Mapping) and isinstance(item.get("id"), str) for item in results):
        raise GateFileError("every suite result needs a string id")
    ids = [item["id"] for item in results]
    if len(set(ids)) != len(ids):
        raise GateFileError(f"suite report repeats scenario ids: {sorted(ids)}")
    counted = {
        status: sum(item.get("status") == status for item in results)
        for status in ("passed", "failed", "skipped")
    }
    expected_summary = {**counted, "total": len(results)}
    if {name: summary.get(name) for name in expected_summary} != expected_summary:
        raise GateFileError(f"suite summary {dict(summary)} disagrees with its results")
    return tuple(results)


def failure_text(result: Mapping[str, Any]) -> str:
    """A failed scenario's errors and response body, the text signatures match."""

    errors = [str(error) for error in result.get("errors") or ()]
    response = result.get("response")
    if response is None:
        return "\n".join(errors)
    body = response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)
    return "\n".join([*errors, body])


def _verdict(result: Mapping[str, Any], expected: ExpectedFailure | None) -> Verdict:
    scenario, status = result["id"], result.get("status")
    if status == "passed":
        if expected is None:
            return Verdict(scenario, "PASS", "")
        return Verdict(scenario, "XPASS", f"passes; delete its entry ({expected.owner_wp})")
    text = failure_text(result)
    detail = " ".join(text.split())[:_DETAIL_CHARS]
    if status != "failed":
        return Verdict(scenario, "FAIL", f"status {status!r}: {detail}")
    if expected is not None and expected.signature in text:
        return Verdict(scenario, "XFAIL", f"{expected.gap_id} → {expected.owner_wp}")
    if expected is not None:
        return Verdict(scenario, "FAIL", f"lacks signature {expected.signature!r}: {detail}")
    return Verdict(scenario, "FAIL", detail)


def evaluate(report: Mapping[str, Any], expected: Sequence[ExpectedFailure]) -> tuple[Verdict, ...]:
    by_id = {entry.id: entry for entry in expected}
    results = _results(report)
    reported = tuple(_verdict(result, by_id.get(result["id"])) for result in results)
    reported_ids = {result["id"] for result in results}
    missing = tuple(
        Verdict(entry.id, "MISSING", f"not in the suite report; delete it ({entry.owner_wp})")
        for entry in expected
        if entry.id not in reported_ids
    )
    return tuple(sorted((*reported, *missing), key=lambda verdict: verdict.id))


def is_green(verdicts: Sequence[Verdict]) -> bool:
    return bool(verdicts) and all(verdict.outcome in GREEN for verdict in verdicts)


def render(verdicts: Sequence[Verdict], repository: str, commit: str) -> str:
    """A Markdown summary of one run."""

    counts = {
        outcome: sum(verdict.outcome == outcome for verdict in verdicts)
        for outcome in ("PASS", "XFAIL", "FAIL", "XPASS", "MISSING")
    }
    tally = ", ".join(f"{count} {outcome}" for outcome, count in counts.items() if count)
    status = "green" if is_green(verdicts) else "RED"
    lines = [
        f"### Open Responses compliance ({repository}@{commit[:10]}): {status}",
        "",
        tally,
        "",
        "| Scenario | Outcome | Detail |",
        "|---|---|---|",
    ]
    lines += [
        f"| `{verdict.id}` | {verdict.outcome} | {verdict.detail.replace('|', '/')} |"
        for verdict in verdicts
    ]
    return "\n".join([*lines, ""])
