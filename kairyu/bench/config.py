"""BenchConfig assembly: bench.yaml (optional) + CLI overrides.

Precedence: CLI flags beat YAML; targets come from --target/--model+--base-url
if given, else the YAML. Keys never appear anywhere — only env var names.
"""

from __future__ import annotations

from pathlib import Path

from kairyu.bench.types import BenchConfig, BenchTarget, JudgeConfig


def parse_target_flag(spec: str, **sampling) -> BenchTarget:
    """`name=base_url=model[=api_key_env]` (frontier_compare.py precedent)."""
    parts = spec.split("=")
    if len(parts) not in (3, 4):
        raise ValueError(
            f"--target {spec!r}: expected name=base_url=model[=api_key_env]"
        )
    name, base_url, model = parts[:3]
    api_key_env = parts[3] if len(parts) == 4 else None
    return BenchTarget(
        name=name, base_url=base_url, model=model, api_key_env=api_key_env, **sampling
    )


def _cli_sampling(args) -> dict:
    """CLI sampling knobs, applied to every CLI-declared target."""
    options = {
        "reasoning_effort": getattr(args, "reasoning_effort", None),
        "top_p": getattr(args, "top_p", None),
        "seed": getattr(args, "sampling_seed", None),
        "extra_body_json": getattr(args, "extra_body", None),
    }
    return {key: value for key, value in options.items() if value is not None}


def _cli_targets(args) -> tuple[BenchTarget, ...]:
    sampling = _cli_sampling(args)
    targets: list[BenchTarget] = [
        parse_target_flag(spec, **sampling) for spec in args.target or []
    ]
    if args.model:
        if not args.base_url:
            raise ValueError("--model requires --base-url (or use --target)")
        targets += [
            BenchTarget(
                base_url=args.base_url,
                model=model,
                api_key_env=args.api_key_env,
                **sampling,
            )
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
    else:
        # CLI beats YAML: sampling flags also apply to YAML-declared targets.
        sampling = _cli_sampling(args)
        if sampling:
            data["targets"] = [
                {**target, **sampling} if isinstance(target, dict) else target
                for target in data.get("targets") or []
            ]

    judge = dict(data.get("judge") or {})
    if getattr(args, "judge_base_url", None):
        judge["base_url"] = args.judge_base_url
    if getattr(args, "judge_model", None):
        judge["model"] = args.judge_model
    if getattr(args, "judge_api_key_env", None):
        judge["api_key_env"] = args.judge_api_key_env
    if getattr(args, "judge_reasoning_effort", None):
        judge["reasoning_effort"] = args.judge_reasoning_effort
    if getattr(args, "judge_extra_body", None):
        judge["extra_body_json"] = args.judge_extra_body
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
