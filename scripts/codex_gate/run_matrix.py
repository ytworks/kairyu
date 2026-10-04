#!/usr/bin/env python3
"""Codex live matrix: real Codex binaries against Kairyu (M20 WP-05, D1/D22).

For every ``--codex`` version, scenario (``scenarios.py``) and provider shape::

    codex exec  ->  record_proxy --exchange-log  ->  launcher (ScenarioBackend)

Codex runs with an isolated ``HOME``/``CODEX_HOME``, a dummy key, default
retries and a ``model_catalog_json`` generated from the server's ``/v1/models``
(``scripts/codex_model_catalog.py``). A run passes when the Codex turn
completes as scripted, the exchange log meets the scenario's wire expectation,
Codex used the catalog (no fallback-metadata warning; its traced
auto-compaction limit equals the catalog's) and no HTTP response was >= 400
except the expected WebSocket-upgrade 426 of the ``openai_base_url`` shape
(D-e).

``--live --base-url URL --model ID`` runs ``LIVE_SCENARIOS`` against a real
deployment instead (it replaces ``scripts/codex_responses_smoke.sh``)::

    python -m scripts.codex_gate.run_matrix --codex 0.160.0 \\
        --live --base-url http://kairyu:8000/v1 --model qwen3-32b

A scenario marked ``xfail`` must fail (``XFAIL``); an unexpected pass is
``XPASS`` and, like ``FAIL``, makes the exit status 1. Artifacts (Codex
events and logs, captures, exchange log, catalog, ``report.json``) are kept in
``--out``.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts import codex_model_catalog as catalog_gen
from scripts.codex_gate.codex_cli import (
    CUSTOM,
    OPENAI_BASE_URL,
    CodexBinary,
    Provider,
    TurnResult,
    prepare_home,
    resolve_codex,
    run_turn,
)
from scripts.codex_gate.exchange_log import read_exchanges
from scripts.codex_gate.scenarios import LIVE_SCENARIOS, SCENARIOS, MatrixScenario

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CODEX = catalog_gen.CODEX_TAG.removeprefix("rust-v")
READY_TIMEOUT_S = 120.0
TURN_TIMEOUT_S = 600.0
# Expected >= 400 per shape: (method, path suffix, status). The built-in openai
# provider tries a WebSocket upgrade first; 426 sends it to HTTPS (D-e, WP-47).
EXPECTED_ERRORS = {OPENAI_BASE_URL: (("GET", "/responses", 426),)}
AUTO_COMPACT_KEY = "model_auto_compact_token_limit"
IMAGE_NAME = "pixel.png"
PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@dataclass(frozen=True)
class RunResult:
    codex: str
    shape: str
    scenario: str
    outcome: str  # PASS | FAIL | XFAIL | XPASS
    failures: tuple[str, ...]
    xfail: str | None
    seconds: float
    run_dir: str


@dataclass(frozen=True)
class Options:
    out: Path
    cache_dir: Path
    live_base_url: str | None
    live_model: str | None
    api_key: str
    catalog: Path | None
    turn_timeout_s: float


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait(ready: Callable[[], bool], what: str, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{what} exited with status {process.returncode}")
        with contextlib.suppress(OSError):
            if ready():
                return
        time.sleep(0.2)
    raise RuntimeError(f"{what} not ready after {READY_TIMEOUT_S:.0f}s")


def _http_ok(url: str) -> bool:
    with urllib.request.urlopen(url, timeout=2) as response:
        return response.status == 200


def _port_open(port: int) -> bool:
    with socket.create_connection(("127.0.0.1", port), timeout=1):
        return True


@contextlib.contextmanager
def background(command: Sequence[str], log_path: Path) -> Iterator[subprocess.Popen]:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", *command],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        try:
            yield process
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def base_instructions(cache_dir: Path) -> str:
    path = cache_dir / f"base-instructions-{catalog_gen.CODEX_TAG}.md"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(catalog_gen.load_base_instructions(None, catalog_gen.CODEX_TAG), "utf-8")
    return path.read_text(encoding="utf-8")


def write_catalog(
    path: Path, base_url: str, model: str, api_key: str | None, instructions: str
) -> int:
    """Generate the run's catalog from ``/models``; return its auto-compact limit."""

    served = catalog_gen.served_models(catalog_gen.fetch_models(base_url, api_key))
    if model not in served:
        raise RuntimeError(f"{base_url}/models does not list {model!r}")
    specs = catalog_gen.specs_from_served({model: served[model]}, {})
    catalog = catalog_gen.build_catalog(specs, instructions)
    path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    return catalog["models"][0]["auto_compact_token_limit"]


def _expected_error(exchange: Mapping[str, Any], shape: str) -> bool:
    return any(
        exchange["method"] == method
        and exchange["path"].endswith(suffix)
        and exchange["status"] == status
        for method, suffix, status in EXPECTED_ERRORS.get(shape, ())
    )


def _turn_failures(scenario: MatrixScenario, turn: TurnResult) -> list[str]:
    failures = []
    if turn.exit_code is None:
        failures.append("Codex timed out")
    elif not (turn.completed and turn.exit_code == 0):
        failures.append(f"turn did not complete ({turn.failure})")
    if turn.fallback_metadata:
        failures.append("Codex used fallback model metadata, not the catalog")
    outcome, message = scenario.outcome, turn.final_message
    if outcome.final_text not in message:
        failures.append(f"final message lacks {outcome.final_text!r}: {message[:80]!r}")
    if len(message.split()) < outcome.min_final_words:
        failures.append(f"final message has {len(message.split())} words")
    if outcome.command is not None and not any(
        outcome.command in run.command and run.exit_code == 0 for run in turn.commands
    ):
        failures.append(f"no successful command containing {outcome.command!r}")
    return failures


def _wire_failures(
    scenario: MatrixScenario, shape: str, exchanges: Sequence[Mapping[str, Any]]
) -> list[str]:
    wire = scenario.wire
    failures = [
        f"unexpected HTTP {e['status']} {e['method']} {e['path']} ({e.get('error_code')})"
        for e in exchanges
        if e["status"] >= 400 and not _expected_error(e, shape)
    ]
    codes = [e["failed_code"] for e in exchanges if e.get("failed_code")]
    failures += [f"unexpected in-band {code}" for code in codes if code not in wire.in_band_codes]
    if shape in wire.in_band_required:
        failures += [f"no in-band {code}" for code in wire.in_band_codes if code not in codes]
    posts = [e for e in exchanges if e["method"] == "POST" and e["path"].endswith("/responses")]
    if wire.max_posts is not None and len(posts) > wire.max_posts:
        failures.append(f"{len(posts)} POST /responses (retried turn?), at most {wire.max_posts}")
    heartbeats = max((e["events"].get("response.in_progress", 1) - 1 for e in posts), default=0)
    if heartbeats < wire.min_heartbeats:
        failures.append(f"{heartbeats} data heartbeats in a stream, need {wire.min_heartbeats}")
    if wire.compaction and not any(
        e.get("tools") == 0 or e.get("compaction_trigger") for e in posts
    ):
        failures.append("no compaction request (tools: [] or compaction_trigger)")
    if wire.live_web_search and not any(e.get("live_web_search") for e in posts):
        failures.append("no request declared live web_search")
    return failures


def evaluate(
    scenario: MatrixScenario,
    shape: str,
    turn: TurnResult,
    exchanges: Sequence[Mapping[str, Any]],
    auto_compact_limit: int | None,
) -> list[str]:
    """Every way the run differs from the scenario (empty: it passed)."""

    failures = _turn_failures(scenario, turn) + _wire_failures(scenario, shape, exchanges)
    if auto_compact_limit is not None:
        override = dict(scenario.codex_config).get(AUTO_COMPACT_KEY, auto_compact_limit)
        expected = min(int(override), auto_compact_limit)
        if set(turn.auto_compact_limits) - {expected}:
            failures.append(
                f"Codex auto-compact limit {sorted(set(turn.auto_compact_limits))}, "
                f"catalog gives {expected}"
            )
    return failures


def _start_server(
    stack: contextlib.ExitStack, scenario: MatrixScenario, run_dir: Path, options: Options
) -> tuple[str, str]:
    """``(upstream root URL, model)``: a scenario launcher, or the live deployment."""

    if options.live_base_url is not None:
        upstream = options.live_base_url.rstrip("/").removesuffix("/v1")
        return upstream, options.live_model or scenario.served_model
    port = free_port()
    command = ["scripts.codex_gate.launcher", "--scenario", scenario.name, "--port", str(port)]
    command += ["--workdir", str(run_dir / "server")]
    server = stack.enter_context(background(command, run_dir / "server.log"))
    upstream = f"http://127.0.0.1:{port}"
    # /health is outside tenant admission, which some scenarios configure.
    _wait(lambda: _http_ok(f"{upstream}/health"), "Kairyu launcher", server)
    return upstream, scenario.served_model


def _start_proxy(
    stack: contextlib.ExitStack,
    upstream: str,
    binary: CodexBinary,
    scenario: MatrixScenario,
    shape: str,
    run_dir: Path,
) -> int:
    port = free_port()
    command = ["scripts.codex_gate.record_proxy", "serve", "--upstream", upstream]
    command += ["--port", str(port), "--out", str(run_dir / "captures")]
    command += ["--codex-version", binary.version, "--provider-shape", shape]
    command += ["--scenario", scenario.name, "--exchange-log", str(run_dir / "exchanges.jsonl")]
    proxy = stack.enter_context(background(command, run_dir / "proxy.log"))
    _wait(lambda: _port_open(port), "record proxy", proxy)
    return port


def _codex_turn(
    binary: CodexBinary, scenario: MatrixScenario, run_dir: Path, options: Options
) -> TurnResult:
    exec_args = scenario.codex_args
    if scenario.attach_image:
        (run_dir / "work" / IMAGE_NAME).write_bytes(PIXEL_PNG)
        exec_args = (*exec_args, f"--image={IMAGE_NAME}")
    return run_turn(
        binary,
        run_dir,
        prompt=scenario.prompt,
        api_key=options.api_key,
        exec_args=exec_args,
        config=scenario.codex_config,
        timeout_s=options.turn_timeout_s,
    )


def run_one(
    binary: CodexBinary, scenario: MatrixScenario, shape: str, options: Options
) -> RunResult:
    started = time.monotonic()
    run_dir = options.out / binary.version / shape / scenario.name
    run_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        upstream, model = _start_server(stack, scenario, run_dir, options)
        proxy_port = _start_proxy(stack, upstream, binary, scenario, shape, run_dir)
        catalog, limit = options.catalog, None
        if catalog is None:
            catalog = run_dir / "catalog.json"
            instructions = base_instructions(options.cache_dir)
            limit = write_catalog(catalog, f"{upstream}/v1", model, options.api_key, instructions)
        provider = Provider(
            shape=shape,
            base_url=f"http://127.0.0.1:{proxy_port}/v1",
            model=model,
            catalog=catalog,
            idle_timeout_ms=scenario.idle_timeout_ms,
        )
        prepare_home(run_dir, provider)
        turn = _codex_turn(binary, scenario, run_dir, options)
    exchanges = read_exchanges(run_dir / "exchanges.jsonl")
    failures = evaluate(scenario, shape, turn, exchanges, limit)
    xfail = scenario.xfail
    if xfail is None:
        outcome = "FAIL" if failures else "PASS"
    else:
        outcome = "XFAIL" if failures else "XPASS"
    return RunResult(
        codex=binary.version,
        shape=shape,
        scenario=scenario.name,
        outcome=outcome,
        failures=tuple(failures),
        xfail=None if xfail is None else f"{xfail.gap} / {xfail.wp}: {xfail.reason}",
        seconds=round(time.monotonic() - started, 1),
        run_dir=str(run_dir),
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--codex", action="append", default=[], metavar="VERSION|TAG|PATH")
    parser.add_argument("--scenario", action="append", default=[], metavar="NAME")
    parser.add_argument("--shape", action="append", default=[], choices=(CUSTOM, OPENAI_BASE_URL))
    parser.add_argument("--out", type=Path, help="artifact directory (default: a temp dir)")
    default_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    parser.add_argument("--cache-dir", type=Path, default=default_cache / "kairyu-codex-gate")
    parser.add_argument("--live", action="store_true", help="target --base-url, not a launcher")
    parser.add_argument("--base-url", help="live Kairyu OpenAI base URL (with /v1)")
    parser.add_argument("--model", help="live model id")
    parser.add_argument("--api-key-env", default="KAIRYU_API_KEY")
    parser.add_argument("--catalog", type=Path, help="use this catalog instead of generating one")
    parser.add_argument("--turn-timeout", type=float, default=TURN_TIMEOUT_S)
    args = parser.parse_args(argv)
    if args.live and not (args.base_url and args.model):
        parser.error("--live needs --base-url and --model")
    if not args.live and (args.base_url or args.model):
        parser.error("--base-url and --model apply only with --live")
    return args


def _select(args: argparse.Namespace) -> list[MatrixScenario]:
    pool = LIVE_SCENARIOS if args.live else SCENARIOS
    names = {scenario.name for scenario in pool}
    unknown = set(args.scenario) - names
    if unknown:
        raise SystemExit(f"unknown scenarios {sorted(unknown)}; choose from {sorted(names)}")
    return [s for s in pool if not args.scenario or s.name in args.scenario]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    scenarios = _select(args)
    out = args.out or Path(tempfile.mkdtemp(prefix="codex-gate-"))
    options = Options(
        out=out,
        cache_dir=args.cache_dir,
        live_base_url=args.base_url if args.live else None,
        live_model=args.model,
        api_key=os.environ.get(args.api_key_env) or "kairyu-codex-gate",
        catalog=args.catalog.resolve() if args.catalog else None,
        turn_timeout_s=args.turn_timeout,
    )
    binaries = [resolve_codex(spec, args.cache_dir) for spec in args.codex or [DEFAULT_CODEX]]
    results = []
    for binary in binaries:
        for scenario in scenarios:
            for shape in scenario.shapes:
                if args.shape and shape not in args.shape:
                    continue
                result = run_one(binary, scenario, shape, options)
                results.append(result)
                print(
                    f"{result.outcome:5} codex {result.codex} {shape:16} {scenario.name}"
                    f" ({result.seconds}s)",
                    flush=True,
                )
                for failure in result.failures:
                    print(f"      {failure}", flush=True)
    report = {"codex": [b.version for b in binaries], "results": [asdict(r) for r in results]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report: {out / 'report.json'}")
    return 1 if any(r.outcome in {"FAIL", "XPASS"} for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
