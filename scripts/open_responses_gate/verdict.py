"""Expected failures and verdicts of the Open Responses gate (M20 WP-14, D1).

``tests/contracts/openresponses/expected-failures.toml`` pins the suite
(``[suite]``: repository, full commit, spec version) and lists every scenario
expected to fail at that pin, each with the gap it tracks, the work package
that makes it pass, the reason, the exact set of errors the scenario reports
while it fails for that reason (its own ``errors`` plus the named
``[error_sets]`` it shares with other entries), and optionally
``body_contains``: text its response body contains (an HTTP status error
reports only ``HTTP 400: [object Object]``; the reason is in the body).

Per scenario of a ``bin/compliance-test.ts --json`` report:

- ``PASS``: passed and not listed;
- ``XFAIL``: failed, listed, with exactly its listed errors (and body text);
- ``FAIL``: failed unlisted, failed listed with an unlisted error, a listed
  error no longer seen or another body, or skipped;
- ``XPASS``: passed although listed, so the owning WP must delete the entry;
- ``MISSING``: listed but absent from the report (the pin moved).

Only ``PASS`` and ``XFAIL`` keep the gate green, so the list stays truthful
and a new error in a listed scenario is not masked by its listed ones.
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
_ENTRY_TEXTS = ("id", "gap_id", "owner_wp", "reason")
_ENTRY_OPTIONAL = ("errors", "error_sets", "body_contains")
_TOP_LEVEL = frozenset({"suite", "error_sets", "expected_failure"})
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
    errors: tuple[str, ...]
    reason: str
    body_contains: str | None = None


@dataclass(frozen=True)
class GateFile:
    suite: Suite
    expected: tuple[ExpectedFailure, ...]


@dataclass(frozen=True)
class Verdict:
    id: str
    outcome: str
    detail: str


def _check_keys(
    raw: Mapping[str, Any], required: Sequence[str], optional: Sequence[str], where: str
) -> None:
    missing = [name for name in required if name not in raw]
    unknown = sorted(set(raw) - {*required, *optional})
    if missing or unknown:
        raise GateFileError(f"{where}: missing {missing}, unknown {unknown}")


def _text(value: Any, name: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GateFileError(f"{where}: {name} must be a non-empty string")
    return value


def _fields(raw: Mapping[str, Any], names: Sequence[str], where: str) -> dict[str, str]:
    _check_keys(raw, names, (), where)
    return {name: _text(raw[name], name, where) for name in names}


def _texts(raw: Any, name: str, where: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise GateFileError(f"{where}: {name} must be a non-empty list")
    texts = tuple(_text(item, name, where) for item in raw)
    if len(set(texts)) != len(texts):
        raise GateFileError(f"{where}: {name} repeats an item")
    return texts


def _error_sets(raw: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, Mapping):
        raise GateFileError("[error_sets] must be a table of lists")
    return {name: _texts(errors, "errors", f"[error_sets].{name}") for name, errors in raw.items()}


def _expected_errors(
    raw: Mapping[str, Any], error_sets: Mapping[str, tuple[str, ...]], where: str
) -> tuple[str, ...]:
    """The entry's own ``errors`` and those of its ``error_sets``, without repeats."""

    names = _texts(raw["error_sets"], "error_sets", where) if "error_sets" in raw else ()
    unknown = sorted(set(names) - set(error_sets))
    if unknown:
        raise GateFileError(f"{where}: unknown error_sets {unknown}")
    own = _texts(raw["errors"], "errors", where) if "errors" in raw else ()
    errors = tuple(dict.fromkeys([*(e for name in names for e in error_sets[name]), *own]))
    if not errors:
        raise GateFileError(f"{where}: lists no errors or error_sets")
    return errors


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


def _entry(raw: Any, index: int, error_sets: Mapping[str, tuple[str, ...]]) -> ExpectedFailure:
    if not isinstance(raw, Mapping):
        raise GateFileError(f"expected_failure #{index} must be a table")
    where = f"expected_failure #{index} ({raw.get('id', '?')})"
    _check_keys(raw, _ENTRY_TEXTS, _ENTRY_OPTIONAL, where)
    fields = {name: _text(raw[name], name, where) for name in _ENTRY_TEXTS}
    checks = {
        "id": _SCENARIO_ID.match(fields["id"]),
        "gap_id": GAP_ID_PATTERN.match(fields["gap_id"]),
        "owner_wp": OWNER_WP_PATTERN.match(fields["owner_wp"]),
    }
    invalid = [name for name, ok in checks.items() if not ok]
    if invalid:
        raise GateFileError(f"{where}: invalid {invalid}")
    body = raw.get("body_contains")
    return ExpectedFailure(
        **fields,
        errors=_expected_errors(raw, error_sets, where),
        body_contains=None if body is None else _text(body, "body_contains", where),
    )


def load_gate_file(path: Path = EXPECTED_FILE) -> GateFile:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(raw) - _TOP_LEVEL)
    if unknown:
        raise GateFileError(f"{path.name}: unknown top-level keys {unknown}")
    error_sets = _error_sets(raw.get("error_sets", {}))
    listed = raw.get("expected_failure", [])
    entries = tuple(_entry(item, index, error_sets) for index, item in enumerate(listed))
    ids = [entry.id for entry in entries]
    duplicates = sorted({entry_id for entry_id in ids if ids.count(entry_id) > 1})
    if duplicates:
        raise GateFileError(f"duplicate expected_failure ids: {duplicates}")
    used = {name for item in listed for name in item.get("error_sets", ())}
    if unused := sorted(set(error_sets) - used):
        raise GateFileError(f"[error_sets] no entry uses: {unused}")
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


def _reported_errors(result: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(error) for error in result.get("errors") or ())


def _body(result: Mapping[str, Any]) -> str:
    response = result.get("response")
    if response is None:
        return ""
    return response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)


def _mismatches(result: Mapping[str, Any], expected: ExpectedFailure) -> list[str]:
    """How a listed scenario's failure differs from its entry; empty when it matches."""

    observed, listed = set(_reported_errors(result)), set(expected.errors)
    problems = []
    if unlisted := sorted(observed - listed):
        problems.append(f"unlisted errors {unlisted}")
    if unseen := sorted(listed - observed):
        problems.append(f"listed errors no longer seen {unseen}, update the entry")
    if expected.body_contains is not None and expected.body_contains not in _body(result):
        problems.append(f"body lacks {expected.body_contains!r}")
    return problems


def _verdict(result: Mapping[str, Any], expected: ExpectedFailure | None) -> Verdict:
    scenario, status = result["id"], result.get("status")
    if status == "passed":
        if expected is None:
            return Verdict(scenario, "PASS", "")
        return Verdict(scenario, "XPASS", f"passes; delete its entry ({expected.owner_wp})")
    text = "\n".join([*_reported_errors(result), _body(result)])
    detail = " ".join(text.split())[:_DETAIL_CHARS]
    if status != "failed":
        return Verdict(scenario, "FAIL", f"status {status!r}: {detail}")
    if expected is None:
        return Verdict(scenario, "FAIL", detail)
    if problems := _mismatches(result, expected):
        return Verdict(scenario, "FAIL", "; ".join(problems)[:_DETAIL_CHARS])
    return Verdict(scenario, "XFAIL", f"{expected.gap_id} → {expected.owner_wp}")


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
