"""``kairyu bench`` subcommands and benchmark ownership inspection."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def add_bench_parser(subparsers) -> None:
    bench = subparsers.add_parser(
        "bench",
        help="Fugu-suite quality benchmarks against a deployed OpenAI-compatible "
        "gateway (single models and orchestrations are both just model names).",
    )
    commands = bench.add_subparsers(dest="bench_command", required=True)

    run = commands.add_parser(
        "run", help="Download missing datasets, run the suite, write the scoreboard."
    )
    run.add_argument("--config", default=None, help="bench.yaml (CLI flags override it)")
    run.add_argument("--base-url", default=None, help="Gateway URL for --model targets")
    run.add_argument(
        "--model",
        action="append",
        default=None,
        help="Target model name (repeatable); orchestrations are model names too",
    )
    run.add_argument(
        "--target",
        action="append",
        default=None,
        help="Full target: name=base_url=model[=api_key_env] (repeatable)",
    )
    run.add_argument("--api-key-env", default=None, help="Env VAR holding the API key")
    run.add_argument(
        "--no-vision",
        action="store_true",
        help="Declare the target text-only, so vision slots skip instead of being "
        "measured on prompts with the image silently dropped",
    )
    run.add_argument(
        "--reasoning-effort",
        default=None,
        help="reasoning_effort sent to every target (Fugu reports max effort)",
    )
    run.add_argument("--top-p", type=float, default=None, help="top_p sent to every target")
    run.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="seed sent to every target (distinct from --seed, which samples items)",
    )
    run.add_argument(
        "--extra-body",
        default=None,
        help='JSON object merged into every target request, e.g. \'{"chat_template_kwargs":'
        ' {"enable_thinking": true}}\'',
    )
    run.add_argument("--suite", default=None, help="Benchmark suite (default: fugu)")
    run.add_argument("--only", action="append", default=None, help="Comma-separated names")
    run.add_argument("--exclude", action="append", default=None, help="Comma-separated names")
    run.add_argument("--limit", type=int, default=None, help="Max items per benchmark")
    run.add_argument("--smoke", action="store_true", help="Small deterministic subset")
    run.add_argument(
        "--offline-fixtures",
        action="store_true",
        help="Use the committed tiny fixtures (no network, no cache)",
    )
    run.add_argument("--seed", type=int, default=None)
    run.add_argument(
        "--attempts",
        type=int,
        default=None,
        help="Trials per task for the agentic harnesses (Fugu reports tau-3 as pass@4)",
    )
    run.add_argument("--judge-base-url", default=None)
    run.add_argument("--judge-model", default=None)
    run.add_argument("--judge-api-key-env", default=None)
    run.add_argument(
        "--judge-reasoning-effort",
        default=None,
        help="reasoning_effort for the judge / tau user simulator (Fugu used low)",
    )
    run.add_argument("--judge-extra-body", default=None, help="JSON object for judge requests")
    run.add_argument("--concurrency", type=int, default=None)
    run.add_argument("--results-dir", default=None)
    run.add_argument("--run-id", default=None, help="Reuse an id to resume a run")
    run.add_argument("--rerun", action="store_true", help="Ignore stored pair results")
    run.add_argument("--cache-dir", default=None)
    run.add_argument("--no-download", action="store_true")
    run.add_argument(
        "--exec-runner",
        choices=("local", "docker"),
        default=None,
        help="Generated-code runner (default: local; use docker for untrusted code)",
    )
    run.add_argument(
        "--exec-image",
        default=None,
        help="Immutable sha256 image ID or repository@sha256 digest for Docker execution",
    )
    run.add_argument(
        "--exec-cpus",
        type=float,
        default=None,
        help="CPU limit for each execution container (default: 1.0)",
    )
    run.add_argument(
        "--exec-pids-limit",
        type=int,
        default=None,
        help="PID limit for each execution container (default: 64)",
    )
    run.add_argument(
        "--exec-disk-mb",
        type=int,
        default=None,
        help="Writable container /work limit in MiB (default: 256)",
    )
    run.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the live progress display (scoreboard output is unchanged)",
    )

    download = commands.add_parser(
        "download", help="Fetch and normalize the suite's datasets into the cache."
    )
    download.add_argument("--suite", default="fugu")
    download.add_argument("--only", action="append", default=None)
    download.add_argument("--exclude", action="append", default=None)
    download.add_argument("--cache-dir", default=None)
    download.add_argument("--force", action="store_true", help="Re-download cached datasets")
    download.add_argument(
        "--strict", action="store_true", help="Exit 1 if any dataset failed to download"
    )

    report = commands.add_parser(
        "report",
        help="Rebuild the scoreboard and the accuracy report vs published Fugu scores.",
    )
    report.add_argument("run", help="Run id (under bench/results/fugu) or a run directory")
    report.add_argument("--results-dir", default="bench/results/fugu")
    report.add_argument(
        "--no-comparison",
        action="store_true",
        help="Print only the scoreboard, without the published-score comparison",
    )

    commands.add_parser("list", help="List benchmarks, requirements, and cache status.")

    entrypoints = commands.add_parser(
        "entrypoints",
        help="List installed and repository-only benchmark ownership.",
    )
    entrypoints.add_argument(
        "--json",
        action="store_true",
        help="Emit the package-owned entrypoint manifest as JSON.",
    )
    entrypoints.add_argument(
        "--check-repo",
        type=Path,
        default=None,
        metavar="PATH",
        help="Validate a Kairyu source checkout against the ownership manifest.",
    )


def handle(args: argparse.Namespace) -> int:
    if args.bench_command == "run":
        return _handle_run(args)
    if args.bench_command == "download":
        return _handle_download(args)
    if args.bench_command == "report":
        return _handle_report(args)
    if args.bench_command == "list":
        return _handle_list(args)
    if args.bench_command == "entrypoints":
        return _handle_entrypoints(args)
    raise ValueError(f"unknown bench command {args.bench_command!r}")


def _handle_run(args) -> int:
    from kairyu.bench.config import build_config
    from kairyu.bench.runner import SuiteRunner

    config = build_config(args)
    return asyncio.run(SuiteRunner(config).run())


def _handle_download(args) -> int:
    from kairyu.bench.adapters import suite_adapters
    from kairyu.bench.adapters.base import DownloadContext
    from kairyu.bench.cache import BenchCache, resolve_cache_root
    from kairyu.bench.config import _split_csv

    cache = BenchCache(resolve_cache_root(args.cache_dir))
    adapters = suite_adapters(
        args.suite, only=_split_csv(args.only), exclude=_split_csv(args.exclude)
    )
    ctx = DownloadContext(cache=cache, force=args.force)
    failed = 0
    for adapter in adapters:
        report = adapter.download(ctx)
        line = f"{report.adapter}: {report.status}"
        if report.detail:
            line += f" — {report.detail}"
        print(line)
        if report.status in ("gated", "unavailable", "extras_missing"):
            failed += 1
    print(f"cache: {cache.root}")
    return 1 if (failed and args.strict) else 0


def _handle_report(args) -> int:
    from kairyu.bench.aggregate import build_scoreboard, render_markdown
    from kairyu.bench.compare import build_comparison, render_comparison_markdown
    from kairyu.bench.store import ResultStore
    from kairyu.bench.types import BenchTarget, JudgeConfig, PairResult

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        run_dir = Path(args.results_dir) / args.run
    if not run_dir.is_dir():
        print(f"no such run: {args.run}")
        return 1

    run_meta = {}
    run_json = run_dir / "run.json"
    if run_json.exists():
        run_meta = json.loads(run_json.read_text(encoding="utf-8"))
    pairs = [
        PairResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(run_dir.glob("*/*.json"))
    ]
    targets = list(dict.fromkeys(pair.target for pair in pairs))
    config = run_meta.get("config", {})
    configured: list[str] = []
    target_configs: list[BenchTarget] = []
    for raw_target in config.get("targets", []):
        if not isinstance(raw_target, dict):
            continue
        label = raw_target.get("name") or raw_target.get("model")
        if isinstance(label, str) and label:
            configured.append(label)
        try:
            target_configs.append(BenchTarget.model_validate(raw_target))
        except ValueError:
            # Legacy/incomplete target records remain displayable, but aggregate
            # marks their judge-independence identity unknown.
            continue
    raw_judge = config.get("judge")
    judge = None
    if isinstance(raw_judge, dict):
        explicitly_disabled = (
            "base_url" in raw_judge
            and "model" in raw_judge
            and raw_judge["base_url"] is None
            and raw_judge["model"] is None
        )
        if raw_judge and not explicitly_disabled:
            base_url = raw_judge.get("base_url")
            model = raw_judge.get("model")
            if not isinstance(base_url, str):
                base_url = None
            if not isinstance(model, str):
                model = None
            if base_url is None and model is None:
                model = "identity-unavailable"
            judge = JudgeConfig(base_url=base_url, model=model)
    scoreboard = build_scoreboard(
        run_id=run_meta.get("run_id", run_dir.name),
        suite=config.get("suite", "fugu"),
        config=config,
        environment=run_meta.get("environment", {}),
        pairs=pairs,
        targets=configured or targets,
        target_configs=target_configs,
        judge=judge,
    )
    markdown = render_markdown(scoreboard)
    store = ResultStore(run_dir.parent, run_dir.name)
    store.save_scoreboard(scoreboard, markdown)
    print(markdown)

    if not getattr(args, "no_comparison", False):
        comparison = build_comparison(scoreboard)
        comparison_markdown = render_comparison_markdown(comparison)
        store.save_comparison(comparison, comparison_markdown)
        print(comparison_markdown)
    return 0


def _handle_list(args) -> int:  # noqa: ARG001 - argparse handler signature
    from kairyu.bench.adapters import FUGU_ROW_ORDER, all_adapters
    from kairyu.bench.cache import BenchCache, resolve_cache_root

    cache = BenchCache(resolve_cache_root())
    registry = all_adapters()
    print(f"suite fugu ({len(FUGU_ROW_ORDER)} slots), cache: {cache.root}")
    for name in FUGU_ROW_ORDER:
        adapter = registry.get(name)
        if adapter is None:
            print(f"  {name:24s} (not implemented)")
            continue
        info = adapter.info
        needs = [
            label
            for label, flag in (
                ("gated", info.gated),
                ("docker", info.needs_docker),
                ("exec", info.needs_execution),
                ("vision", info.needs_vision),
                ("judge", info.judge_preferred),
                ("agentic", info.agentic),
            )
            if flag
        ]
        state = "cached" if cache.is_ready(name) else "not downloaded"
        extras = f" [{', '.join(needs)}]" if needs else ""
        print(f"  {name:24s} {info.display_name} — {info.metric}{extras} ({state})")
    return 0


def _handle_entrypoints(args) -> int:
    from kairyu.bench.ownership import (
        entrypoints_payload,
        load_entrypoints,
        validate_repository,
    )

    errors = (
        validate_repository(args.check_repo)
        if args.check_repo is not None
        else ()
    )
    if args.json:
        payload = entrypoints_payload()
        if args.check_repo is not None:
            payload["repository_check"] = {
                "path": str(args.check_repo),
                "valid": not errors,
                "errors": list(errors),
            }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1 if errors else 0

    print("installed: kairyu.bench library, kairyu bench CLI, package fixtures")
    print("repository-only: bench/*.py entrypoints and bench/results artifacts")
    for entry in load_entrypoints():
        requirements = ",".join(entry.requires)
        print(f"  {entry.path:48s} {entry.kind:16s} [{requirements}]")
    if args.check_repo is not None:
        if errors:
            print(f"repository check: FAILED ({args.check_repo})")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"repository check: OK ({args.check_repo})")
    return 1 if errors else 0
