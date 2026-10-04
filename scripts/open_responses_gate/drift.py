#!/usr/bin/env python3
"""Weekly drift check of the pinned Open Responses suite (M20 WP-14, D1).

::

    python -m scripts.open_responses_gate.drift --out DIR --title FILE --report FILE

Resolves upstream ``main`` of the pinned repository. When it is not the pin,
runs that suite against the same Kairyu gate server (``run.run_gate``) with
the pinned expected failures and writes an issue title to ``--title`` and a
Markdown body to ``--report``: the compare link, the spec and suite files that
changed, and the verdict at ``main``. Both files are empty when ``main`` is the
pin. The exit status is 0 either way: the job reports drift, it gates nothing.
The title names upstream's commit, so the workflow opens one issue per state.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from scripts.open_responses_gate import run, verdict

UPSTREAM_BRANCH = "main"
TITLE_PREFIX = "M20 Open Responses drift"
# What the gate depends on: the published spec, the suite, its schemas and CLI.
TRACKED_PATHS = (
    "CHANGELOG.md",
    "bin/compliance-test.ts",
    "public/openapi",
    "src/generated",
    "src/lib/compliance-tests.ts",
    "src/lib/sse-parser.ts",
)
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
PROCEDURE = (
    "Procedure (m20 D1, docs/design/m20-open-responses-ci.md): review the upstream "
    "changes; to move the pin, set `[suite].commit` (and `spec` on a new spec release) in "
    "`tests/contracts/openresponses/expected-failures.toml`, run "
    "`python -m scripts.open_responses_gate.run --out <dir>` and, in the same PR, add or "
    "delete the expected failures its FAIL, XPASS and MISSING rows name."
)


def upstream_head(repository: str, cwd: Path) -> str:
    url = f"https://github.com/{repository}.git"
    fields = run.run_git("ls-remote", url, f"refs/heads/{UPSTREAM_BRANCH}", cwd=cwd).split()
    if not fields or not _FULL_SHA.match(fields[0]):
        raise SystemExit(f"cannot resolve {repository}@{UPSTREAM_BRANCH}: {fields!r}")
    return fields[0]


def changed_files(checkout: Path, repository: str, pin: str, head: str) -> list[str]:
    """``git diff --numstat`` lines of ``TRACKED_PATHS`` between the pin and ``head``."""

    # A shallow fetch adds the pin's tree next to ``head``; the diff needs no history.
    url = f"https://github.com/{repository}.git"
    run.run_git("fetch", "-q", "--depth=1", url, pin, cwd=checkout)
    numstat = run.run_git("diff", "--numstat", pin, head, "--", *TRACKED_PATHS, cwd=checkout)
    return [line for line in numstat.splitlines() if line.strip()]


def _changes_section(numstat: Sequence[str]) -> list[str]:
    if not numstat:
        return ["No tracked spec or suite file changed.", ""]
    rows = []
    for line in numstat:
        added, removed, path = line.split("\t", 2)
        rows.append(f"- `{path}` (+{added} −{removed})")
    return ["Changed spec and suite files:", *rows, ""]


def report(gate: verdict.GateFile, head: str, numstat: Sequence[str], summary: str) -> str:
    pin = gate.suite.commit
    compare = f"https://github.com/{gate.suite.repository}/compare/{pin}...{head}"
    return "\n".join(
        [
            "Automated weekly check of the pinned Open Responses suite.",
            "",
            f"### `{pin[:10]}` (pin) → `{head[:10]}` ({UPSTREAM_BRANCH})",
            f"Compare: {compare}",
            "",
            *_changes_section(numstat),
            summary,
            PROCEDURE,
            "",
        ]
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="suite checkout parent (default: --out)")
    parser.add_argument("--bun", help="bun executable (default: bun on PATH)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    gate = verdict.load_gate_file()
    for path in (args.title, args.report):
        path.write_text("", encoding="utf-8")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    head = upstream_head(gate.suite.repository, out)
    if head == gate.suite.commit:
        print(f"no drift: {UPSTREAM_BRANCH} is the pin {head[:10]}")
        return 0
    bun = run.resolve_bun(args.bun)
    checkout = (args.cache_dir or out) / "openresponses"
    commit = run.prepare_suite(gate.suite.repository, head, checkout, bun)
    numstat = changed_files(checkout, gate.suite.repository, gate.suite.commit, commit)
    verdicts = run.run_gate(gate, checkout, commit, bun, out)
    summary = verdict.render(verdicts, gate.suite.repository, commit)
    args.title.write_text(f"{TITLE_PREFIX}: {UPSTREAM_BRANCH} {commit[:10]}", encoding="utf-8")
    args.report.write_text(report(gate, commit, numstat, summary), encoding="utf-8")
    print(f"drift: {UPSTREAM_BRANCH} {commit[:10]} differs from the pin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
