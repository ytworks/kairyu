#!/usr/bin/env python3
"""Run the pinned Open Responses compliance suite against Kairyu (M20 WP-14).

::

    python -m scripts.open_responses_gate.run --out DIR [--cache-dir DIR] [--bun BUN]

fetches ``[suite]`` of ``tests/contracts/openresponses/expected-failures.toml``
at its pinned commit, installs it with ``bun install --frozen-lockfile
--ignore-scripts``, starts ``launcher`` (Kairyu on ScenarioBackend) and runs
``bin/compliance-test.ts --json`` against it. The suite's own exit status is
ignored: ``verdict.evaluate`` checks every scenario against the expected
failures, and the exit status is 1 unless all are ``PASS`` or ``XFAIL``.
``--out`` keeps ``results.json`` (the suite report), ``verdict.md`` and
``server.log``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.codex_gate.run_matrix import background, free_port
from scripts.open_responses_gate import verdict
from scripts.open_responses_gate.launcher import MODEL

READY_TIMEOUT_S = 120.0
INSTALL_TIMEOUT_S = 600.0
SUITE_TIMEOUT_S = 600.0
GIT_TIMEOUT_S = 300.0
# The gate deployment has no API keys; the suite requires one to send.
DUMMY_KEY = "kairyu-open-responses-gate"


def run_git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=GIT_TIMEOUT_S
    ).stdout.strip()


def prepare_suite(repository: str, ref: str, checkout: Path, bun: str) -> str:
    """Check ``repository@ref`` out (depth 1) and install it; return the commit."""

    checkout.mkdir(parents=True, exist_ok=True)
    if not (checkout / ".git").exists():
        run_git("init", "-q", cwd=checkout)
    run_git("fetch", "-q", "--depth=1", f"https://github.com/{repository}.git", ref, cwd=checkout)
    run_git("checkout", "-q", "--force", "--detach", "FETCH_HEAD", cwd=checkout)
    # --ignore-scripts: the suite needs only its committed zod schemas, so no
    # lifecycle script of its dependency tree runs on the runner.
    subprocess.run(
        [bun, "install", "--frozen-lockfile", "--ignore-scripts"],
        cwd=checkout,
        check=True,
        timeout=INSTALL_TIMEOUT_S,
    )
    return run_git("rev-parse", "HEAD", cwd=checkout)


def _wait_healthy(url: str, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Kairyu launcher exited with status {server.returncode}")
        with contextlib.suppress(OSError), urllib.request.urlopen(url, timeout=2) as response:
            if response.status == 200:
                return
        time.sleep(0.2)
    raise RuntimeError(f"Kairyu launcher not ready after {READY_TIMEOUT_S:.0f}s")


@contextlib.contextmanager
def kairyu_server(out: Path) -> Iterator[str]:
    """A gate launcher on a free port; yields its ``/v1`` base URL."""

    port = free_port()
    command = ["scripts.open_responses_gate.launcher", "--port", str(port)]
    command += ["--workdir", str(out / "server")]
    with background(command, out / "server.log") as server:
        _wait_healthy(f"http://127.0.0.1:{port}/health", server)
        yield f"http://127.0.0.1:{port}/v1"


def run_suite(checkout: Path, bun: str, base_url: str) -> dict[str, Any]:
    """The suite's ``--json`` report against ``base_url``."""

    command = [bun, "run", "bin/compliance-test.ts", "--base-url", base_url]
    command += ["--api-key", DUMMY_KEY, "--model", MODEL, "--json"]
    completed = subprocess.run(
        command,
        cwd=checkout,
        capture_output=True,
        text=True,
        timeout=SUITE_TIMEOUT_S,
        env={**os.environ, "NO_COLOR": "1"},
    )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"suite exited {completed.returncode} without a JSON report: {error}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        ) from error
    if not isinstance(report, Mapping):
        raise RuntimeError("suite report is not a JSON object")
    return dict(report)


def run_gate(
    gate: verdict.GateFile, checkout: Path, commit: str, bun: str, out: Path
) -> tuple[verdict.Verdict, ...]:
    """Run the suite at ``checkout``; write ``results.json`` and ``verdict.md``."""

    with kairyu_server(out) as base_url:
        report = run_suite(checkout, bun, base_url)
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    verdicts = verdict.evaluate(report, gate.expected)
    summary = verdict.render(verdicts, gate.suite.repository, commit)
    (out / "verdict.md").write_text(summary, encoding="utf-8")
    print(summary)
    return verdicts


def resolve_bun(explicit: str | None) -> str:
    bun = explicit or shutil.which("bun")
    if bun is None:
        raise SystemExit("bun is required (install it or pass --bun)")
    return bun


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="suite checkout parent (default: --out)")
    parser.add_argument("--bun", help="bun executable (default: bun on PATH)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    gate = verdict.load_gate_file()
    bun = resolve_bun(args.bun)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkout = (args.cache_dir or out) / "openresponses"
    commit = prepare_suite(gate.suite.repository, gate.suite.commit, checkout, bun)
    if commit != gate.suite.commit:
        raise SystemExit(f"fetched {commit}, expected the pin {gate.suite.commit}")
    verdicts = run_gate(gate, checkout, commit, bun, out)
    return 0 if verdict.is_green(verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
