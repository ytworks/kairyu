"""BenchConfig assembly: bench.yaml (optional) + CLI overrides.

Precedence: CLI flags beat YAML; targets come from --target/--model+--base-url
if given, else the YAML. Keys never appear anywhere — only env var names.
"""

from __future__ import annotations

from pathlib import Path

from kairyu.bench.types import BenchConfig, BenchTarget, JudgeConfig


def parse_target_flag(spec: str) -> BenchTarget:
    """`name=base_url=model[=api_key_env]` (frontier_compare.py precedent)."""
    parts = spec.split("=")
    if len(parts) not in (3, 4):
        raise ValueError(
            f"--target {spec!r}: expected name=base_url=model[=api_key_env]"
        )
    name, base_url, model = parts[:3]
    api_key_env = parts[3] if len(parts) == 4 else None
    return BenchTarget(name=name, base_url=base_url, model=model, api_key_env=api_key_env)


def _cli_targets(args) -> tuple[BenchTarget, ...]:
    targets: list[BenchTarget] = [parse_target_flag(spec) for spec in args.target or []]
    if args.model:
        if not args.base_url:
            raise ValueError("--model requires --base-url (or use --target)")
        targets += [
            BenchTarget(base_url=args.base_url, model=model, api_key_env=args.api_key_env)
            for model in args.model
        ]
    return tuple(targets)


def _split_csv(values: list[str] | None) -> tuple[str, ...]:
    names: list[str] = []
    for value in values or []:
        names += [part.strip() for part in value.split(",") if part.strip()]
    return tuple(names)


def build_config(args) -> BenchConfig:
    data: dict = {}
    if args.config is not None:
        import yaml

        loaded = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("bench config YAML must be a mapping at the top level")
        data = loaded

    cli_targets = _cli_targets(args)
    if cli_targets:
        data["targets"] = [target.model_dump() for target in cli_targets]

    judge = dict(data.get("judge") or {})
    if getattr(args, "judge_base_url", None):
        judge["base_url"] = args.judge_base_url
    if getattr(args, "judge_model", None):
        judge["model"] = args.judge_model
    if getattr(args, "judge_api_key_env", None):
        judge["api_key_env"] = args.judge_api_key_env
    if judge:
        data["judge"] = JudgeConfig(**judge).model_dump()

    overrides = {
        "suite": args.suite,
        "limit": args.limit,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "results_dir": args.results_dir,
        "run_id": args.run_id,
        "cache_dir": args.cache_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    if args.smoke:
        data["smoke"] = True
    if args.offline_fixtures:
        data["offline_fixtures"] = True
    if args.rerun:
        data["rerun"] = True
    if args.no_download:
        data["download"] = False
    only = _split_csv(args.only)
    exclude = _split_csv(args.exclude)
    if only:
        data["only"] = only
    if exclude:
        data["exclude"] = exclude

    return BenchConfig.model_validate(data)
