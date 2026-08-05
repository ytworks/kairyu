"""``kairyu bench`` subcommands and benchmark ownership inspection."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def add_bench_parser(subparsers) -> None:
    from kairyu.bench.adapters import suite_names

    suites = suite_names()
    bench = subparsers.add_parser(
        "bench",
        help="Quality benchmark suites against a deployed OpenAI-compatible "
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
        "--served-config-label",
        default=None,
        help="Operator label for the immutable served deployment config (all targets)",
    )
    run.add_argument(
        "--served-config-sha256",
        default=None,
        help="Lowercase SHA-256 of the served deployment config (all targets)",
    )
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
    run.add_argument(
        "--suite", choices=suites, default=None, help="Benchmark suite (default: fugu)"
    )
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
    run.add_argument(
        "--judge-secondary",
        action="append",
        default=None,
        metavar="BASE_URL=MODEL[=API_KEY_ENV]",
        help=(
            "Additional pointwise judge (repeatable). Two total judges require "
            "unanimity; configure per-member sampling in YAML."
        ),
    )
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
    download.add_argument("--suite", choices=suites, default="fugu")
    download.add_argument("--only", action="append", default=None)
    download.add_argument("--exclude", action="append", default=None)
    download.add_argument("--cache-dir", default=None)
    download.add_argument("--force", action="store_true", help="Re-download cached datasets")
    download.add_argument(
        "--strict", action="store_true", help="Exit 1 if any dataset failed to download"
    )

    report = commands.add_parser(
        "report",
        help="Rebuild a suite scoreboard and its published comparison when available.",
    )
    report.add_argument("run", help="Run id (under the suite results dir) or a run directory")
    report.add_argument("--suite", choices=suites, default="fugu")
    report.add_argument(
        "--results-dir",
        default=None,
        help="Results root (default: bench/results/<suite>)",
    )
    report.add_argument(
        "--no-comparison",
        action="store_true",
        help="Print only the scoreboard, without the published-score comparison",
    )

    compare_runs = commands.add_parser(
        "compare-runs",
        help="Compare two immutable indexed scoreboards (candidate minus baseline).",
    )
    compare_runs.add_argument("baseline", help="Baseline run id")
    compare_runs.add_argument("candidate", help="Candidate run id")
    compare_runs.add_argument("--suite", choices=suites, default="fugu")
    compare_runs.add_argument(
        "--results-dir",
        default=None,
        help="Results root (default: bench/results/<suite>)",
    )

    compare = commands.add_parser(
        "compare",
        help=(
            "Gate candidate-vs-baseline served configurations using paired, "
            "item-bound quality evidence."
        ),
    )
    compare.add_argument("--baseline", required=True, help="Indexed baseline run id")
    compare.add_argument("--candidate", required=True, help="Indexed candidate run id")
    compare.add_argument(
        "--baseline-target",
        default=None,
        help="Baseline target label (required when its run has multiple targets)",
    )
    compare.add_argument(
        "--candidate-target",
        default=None,
        help="Candidate target label (required when its run has multiple targets)",
    )
    compare.add_argument(
        "--tolerance",
        action="append",
        required=True,
        metavar="BENCHMARK=PP",
        help=(
            "Allowed score loss in percentage points for one benchmark "
            "(repeatable; every configured row must pass)"
        ),
    )
    compare.add_argument("--suite", choices=suites, default="fugu")
    compare.add_argument(
        "--results-dir",
        default=None,
        help="Results root (default: bench/results/<suite>)",
    )
    compare.add_argument(
        "--json",
        action="store_true",
        help="Print the machine-readable artifact instead of Markdown",
    )

    calibrate = commands.add_parser(
        "calibrate-judge",
        help="Measure judge agreement on the committed published-gold set.",
    )
    calibrate.add_argument(
        "--run",
        default=None,
        help=(
            "Bind headline calibration to a benchmark run id (under --results-dir) or run directory"
        ),
    )
    calibrate.add_argument("--results-dir", default="bench/results/fugu")
    calibrate.add_argument("--config", default=None, help="bench.yaml judge block")
    calibrate.add_argument("--judge-base-url", default=None)
    calibrate.add_argument("--judge-model", default=None)
    calibrate.add_argument("--judge-api-key-env", default=None)
    calibrate.add_argument("--judge-reasoning-effort", default=None)
    calibrate.add_argument("--judge-extra-body", default=None)
    calibrate.add_argument(
        "--judge-secondary",
        action="append",
        default=None,
        metavar="BASE_URL=MODEL[=API_KEY_ENV]",
    )
    calibrate.add_argument(
        "--calibration-set",
        default=None,
        help="Custom labelled JSONL (default: committed published-gold set)",
    )
    calibrate.add_argument("--output", default=None, help="Write the JSON artifact here")
    calibrate.add_argument("--min-agreement", type=float, default=11 / 12)
    calibrate.add_argument("--min-coverage", type=float, default=1.0)
    calibrate.add_argument("--max-position-flip", type=float, default=0.0)
    calibrate.add_argument("--max-self-preference-gap", type=float, default=0.2)
    calibrate.add_argument(
        "--require-self-preference",
        action="store_true",
        help="Fail unless every judge has balanced, provenance-backed bias evidence",
    )

    listing = commands.add_parser("list", help="List benchmarks, requirements, and cache status.")
    listing.add_argument("--suite", choices=suites, default="fugu")

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
    if args.bench_command == "compare-runs":
        return _handle_compare_runs(args)
    if args.bench_command == "compare":
        return _handle_compare(args)
    if args.bench_command == "calibrate-judge":
        return _handle_calibrate_judge(args)
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
    from kairyu.bench.adapters import suite_info
    from kairyu.bench.aggregate import build_scoreboard, render_markdown
    from kairyu.bench.compare import build_comparison, render_comparison_markdown
    from kairyu.bench.store import ResultStore
    from kairyu.bench.types import (
        BenchTarget,
        JudgeConfig,
        JudgeEndpointConfig,
        PairResult,
    )

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        results_dir = getattr(args, "results_dir", None) or (
            f"bench/results/{getattr(args, 'suite', 'fugu')}"
        )
        run_dir = Path(results_dir) / args.run
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
            # Durable run metadata hashes opaque request bodies. They are not
            # executable config anymore, but removing only that sentinel keeps
            # the endpoint/model identity available for self-judge disclosure.
            without_opaque_body = {
                key: value for key, value in raw_target.items() if key != "extra_body_json"
            }
            try:
                target_configs.append(BenchTarget.model_validate(without_opaque_body))
            except ValueError:
                # Legacy/incomplete target records remain displayable, but
                # aggregate marks their judge-independence identity unknown.
                continue
    raw_judge = config.get("judge")
    judge = None
    judge_identity_incomplete = False
    if isinstance(raw_judge, dict):
        explicitly_disabled = (
            "base_url" in raw_judge
            and "model" in raw_judge
            and raw_judge["base_url"] is None
            and raw_judge["model"] is None
            and not raw_judge.get("additional_judges")
        )
        if raw_judge and not explicitly_disabled:

            def without_opaque_body(endpoint):
                if not isinstance(endpoint, dict):
                    return endpoint
                marker = endpoint.get("extra_body_json")
                if isinstance(marker, str) and marker.startswith("sha256:") and len(marker) == 71:
                    return {
                        key: value for key, value in endpoint.items() if key != "extra_body_json"
                    }
                return endpoint

            judge_for_validation = dict(without_opaque_body(raw_judge))
            if isinstance(raw_judge.get("additional_judges"), list | tuple):
                judge_for_validation["additional_judges"] = [
                    without_opaque_body(member) for member in raw_judge.get("additional_judges", [])
                ]
            try:
                candidate = JudgeConfig.model_validate(judge_for_validation)
            except ValueError:
                judge_identity_incomplete = True
                candidate = None

                def salvage_endpoint(raw):
                    if not isinstance(raw, dict):
                        return None
                    base_url = raw.get("base_url")
                    model = raw.get("model")
                    if (
                        not isinstance(base_url, str)
                        or not isinstance(model, str)
                        or not model.strip()
                    ):
                        return None
                    try:
                        return JudgeEndpointConfig(base_url=base_url, model=model)
                    except ValueError:
                        return None

                primary = salvage_endpoint(raw_judge)
                if primary is not None:
                    members: list[JudgeEndpointConfig] = []
                    known = {primary.resolved_identity()}
                    raw_members = raw_judge.get("additional_judges", [])
                    if not isinstance(raw_members, list | tuple):
                        raw_members = []
                    for raw_member in raw_members:
                        member = salvage_endpoint(raw_member)
                        if member is None:
                            continue
                        identity = member.resolved_identity()
                        if identity in known:
                            continue
                        known.add(identity)
                        members.append(member)
                    candidate = JudgeConfig(
                        base_url=primary.base_url,
                        model=primary.model,
                        additional_judges=tuple(members),
                    )
                if candidate is None:
                    candidate = JudgeConfig(model="identity-unavailable")
            if not candidate.enabled:
                judge_identity_incomplete = True
                candidate = JudgeConfig(model="identity-unavailable")
            judge = candidate
    elif raw_judge is not None:
        judge = JudgeConfig(model="identity-unavailable")
        judge_identity_incomplete = True
    suite = config.get("suite", getattr(args, "suite", "fugu"))
    scoreboard = build_scoreboard(
        run_id=run_meta.get("run_id", run_dir.name),
        suite=suite,
        config=config,
        environment=run_meta.get("environment", {}),
        fingerprint=run_meta.get("fingerprint"),
        pairs=pairs,
        targets=configured or targets,
        target_configs=target_configs,
        judge=judge,
        judge_identity_incomplete=judge_identity_incomplete,
    )
    markdown = render_markdown(scoreboard)
    store = ResultStore(run_dir.parent, run_dir.name)
    store.save_scoreboard(scoreboard, markdown)
    print(markdown)

    if suite_info(suite).published_comparison and not getattr(args, "no_comparison", False):
        comparison = build_comparison(scoreboard)
        comparison_markdown = render_comparison_markdown(comparison)
        store.save_comparison(comparison, comparison_markdown)
        print(comparison_markdown)
    return 0


def _handle_compare_runs(args) -> int:
    from kairyu.bench.compare import (
        build_run_comparison,
        render_run_comparison_markdown,
    )
    from kairyu.bench.store import ResultStore

    results_dir = Path(
        getattr(args, "results_dir", None) or f"bench/results/{getattr(args, 'suite', 'fugu')}"
    )
    try:
        store = ResultStore(results_dir, args.baseline)
        # Validate both user-supplied ids before touching the shared index.
        ResultStore(results_dir, args.candidate)
        entries = store.load_scoreboard_index()
        by_run_id = {entry["run_id"]: entry for entry in entries}
        missing = [run_id for run_id in (args.baseline, args.candidate) if run_id not in by_run_id]
        if missing:
            names = ", ".join(repr(run_id) for run_id in missing)
            raise ValueError(
                f"run id(s) {names} are not present in {results_dir / 'scoreboards.jsonl'}"
            )
        comparison = build_run_comparison(
            by_run_id[args.baseline],
            by_run_id[args.candidate],
        )
        if comparison["suite"] != args.suite:
            raise ValueError(
                f"indexed runs belong to suite {comparison['suite']!r}, "
                f"not selected suite {args.suite!r}"
            )
        markdown = render_run_comparison_markdown(comparison)
    except (OSError, ValueError) as error:
        print(f"compare-runs: {error}")
        return 1
    print(markdown)
    return 0


def _config_tolerances(values: list[str]) -> dict[str, float]:
    import math

    tolerances: dict[str, float] = {}
    for value in values:
        if not isinstance(value, str) or value.count("=") != 1:
            raise ValueError("--tolerance must use BENCHMARK=PP")
        benchmark, raw_points = (part.strip() for part in value.split("=", 1))
        if not benchmark or not raw_points:
            raise ValueError("--tolerance must use BENCHMARK=PP")
        if benchmark in tolerances:
            raise ValueError(f"duplicate --tolerance for benchmark {benchmark!r}")
        try:
            points = float(raw_points)
        except ValueError as error:
            raise ValueError(
                f"--tolerance for {benchmark!r} must be a number of percentage points"
            ) from error
        if not math.isfinite(points) or not 0.0 <= points <= 100.0:
            raise ValueError(
                f"--tolerance for {benchmark!r} must be finite and within [0, 100] "
                "percentage points"
            )
        tolerances[benchmark] = points / 100.0
    return tolerances


def _config_target(entry: dict, requested: str | None, role: str) -> str:
    scoreboard = entry.get("scoreboard")
    targets = scoreboard.get("targets") if isinstance(scoreboard, dict) else None
    if not isinstance(targets, list) or any(
        not isinstance(target, str) or not target for target in targets
    ):
        raise ValueError(f"{role} indexed target layout is malformed")
    if requested is None:
        if len(targets) != 1:
            raise ValueError(
                f"{role} run has {len(targets)} targets; select --{role}-target"
            )
        return targets[0]
    if requested not in targets:
        raise ValueError(f"{role} target {requested!r} is not in the indexed run")
    return requested


def _handle_compare(args) -> int:
    from kairyu.bench.config_ab import (
        build_config_comparison,
        render_config_comparison_markdown,
    )
    from kairyu.bench.store import ResultStore

    results_dir = Path(
        getattr(args, "results_dir", None) or f"bench/results/{getattr(args, 'suite', 'fugu')}"
    )
    try:
        tolerances = _config_tolerances(args.tolerance)
        baseline_store = ResultStore(results_dir, args.baseline)
        candidate_store = ResultStore(results_dir, args.candidate)
        entries = baseline_store.load_scoreboard_index()
        by_run_id = {entry["run_id"]: entry for entry in entries}
        missing = [
            run_id
            for run_id in (args.baseline, args.candidate)
            if run_id not in by_run_id
        ]
        if missing:
            names = ", ".join(repr(run_id) for run_id in missing)
            raise ValueError(
                f"run id(s) {names} are not present in {results_dir / 'scoreboards.jsonl'}"
            )
        baseline_entry = by_run_id[args.baseline]
        candidate_entry = by_run_id[args.candidate]
        if baseline_entry.get("suite") != args.suite or candidate_entry.get("suite") != args.suite:
            raise ValueError(f"indexed runs do not both belong to selected suite {args.suite!r}")
        baseline_target = _config_target(
            baseline_entry, getattr(args, "baseline_target", None), "baseline"
        )
        candidate_target = _config_target(
            candidate_entry, getattr(args, "candidate_target", None), "candidate"
        )
        baseline_pairs = {
            benchmark: baseline_store.load_indexed_pair(
                baseline_entry, benchmark, baseline_target
            )
            for benchmark in tolerances
        }
        candidate_pairs = {
            benchmark: candidate_store.load_indexed_pair(
                candidate_entry, benchmark, candidate_target
            )
            for benchmark in tolerances
        }
        comparison = build_config_comparison(
            baseline_entry,
            candidate_entry,
            baseline_pairs=baseline_pairs,
            candidate_pairs=candidate_pairs,
            tolerances=tolerances,
            baseline_target=baseline_target,
            candidate_target=candidate_target,
        )
        markdown = render_config_comparison_markdown(comparison)
        candidate_store.save_config_comparison(comparison, markdown)
    except (OSError, ValueError) as error:
        print(f"compare: {error}")
        return 2
    if getattr(args, "json", False):
        print(json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(markdown)
    return 0 if comparison["overall_verdict"] == "pass" else 1


def _handle_calibrate_judge(args) -> int:
    from kairyu.bench.calibration import run_calibration_cli

    return asyncio.run(run_calibration_cli(args))


def _handle_list(args) -> int:
    from kairyu.bench.adapters import all_adapters, suite_info
    from kairyu.bench.adapters.base import adapter_cache_ready
    from kairyu.bench.cache import BenchCache, resolve_cache_root

    cache = BenchCache(resolve_cache_root())
    registry = all_adapters()
    definition = suite_info(getattr(args, "suite", "fugu"))
    print(f"suite {definition.name} ({len(definition.row_order)} slots), cache: {cache.root}")
    for name in definition.row_order:
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
        state = "cached" if adapter_cache_ready(adapter, cache) else "not downloaded"
        extras = f" [{', '.join(needs)}]" if needs else ""
        print(f"  {name:24s} {info.display_name} — {info.metric}{extras} ({state})")
    return 0


def _handle_entrypoints(args) -> int:
    from kairyu.bench.ownership import (
        entrypoints_payload,
        load_entrypoints,
        validate_repository,
    )

    errors = validate_repository(args.check_repo) if args.check_repo is not None else ()
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
