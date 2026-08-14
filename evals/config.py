"""BenchConfig assembly: bench.yaml (optional) + CLI overrides.

Precedence: CLI flags beat YAML; targets come from --target/--model+--base-url
if given, else the YAML. Keys never appear anywhere — only env var names.
"""

from __future__ import annotations

from pathlib import Path

from evals.targets import parse_target_spec
from evals.types import (
    BenchConfig,
    BenchTarget,
    JudgeConfig,
    JudgeEndpointConfig,
    ServedConfigIdentity,
)


def parse_target_flag(spec: str, **sampling) -> BenchTarget:
    """Compatibility alias for the package-owned target parser."""
    return parse_target_spec(spec, **sampling)


def _cli_sampling(args) -> dict:
    """CLI target overrides, applied to every CLI-declared target."""
    recommended = bool(getattr(args, "recommended_sampling", False))
    temperature = getattr(args, "temperature", None)
    top_p = getattr(args, "top_p", None)
    if recommended and (temperature is not None or top_p is not None):
        raise ValueError(
            "--recommended-sampling cannot be combined with --temperature or --top-p"
        )
    options = {
        "reasoning_effort": getattr(args, "reasoning_effort", None),
        "top_p": top_p,
        "seed": getattr(args, "sampling_seed", None),
        "extra_body_json": getattr(args, "extra_body", None),
        "temperature": temperature,
        "sampling_mode": (
            "recommended"
            if recommended
            else ("adapter" if temperature is not None or top_p is not None else None)
        ),
        # only ever narrows the default; `--no-vision` absent means "unspecified"
        "supports_vision": False if getattr(args, "no_vision", False) else None,
        "max_context_tokens": getattr(args, "max_context_tokens", None),
        "max_output_tokens": getattr(args, "max_output_tokens", None),
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


def _cli_served_config(args) -> ServedConfigIdentity | None:
    label = getattr(args, "served_config_label", None)
    sha256 = getattr(args, "served_config_sha256", None)
    if (label is None) != (sha256 is None):
        raise ValueError(
            "--served-config-label and --served-config-sha256 must be provided together"
        )
    if label is None:
        return None
    return ServedConfigIdentity(label=label, sha256=sha256)


def _split_csv(values: list[str] | None) -> tuple[str, ...]:
    names: list[str] = []
    for value in values or []:
        names += [part.strip() for part in value.split(",") if part.strip()]
    return tuple(names)


def parse_additional_judge_flag(spec: str) -> JudgeEndpointConfig:
    """Parse ``base_url=model[=api_key_env]`` for a panel member."""
    if not isinstance(spec, str):
        raise ValueError(
            "--judge-secondary: expected base_url=model[=api_key_env]"
        )
    parts = [part.strip() for part in spec.split("=")]
    if len(parts) not in (2, 3) or any(not part for part in parts):
        raise ValueError(
            "--judge-secondary: expected base_url=model[=api_key_env]"
        )
    values = {"base_url": parts[0], "model": parts[1]}
    if len(parts) == 3:
        values["api_key_env"] = parts[2]
    return JudgeEndpointConfig(**values)


def _load_config_mapping(path: str | None) -> dict:
    if path is None:
        return {}
    import yaml

    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("bench config YAML must be a mapping at the top level")
    return loaded


def _judge_data(data: dict, args) -> dict:
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
    secondary = getattr(args, "judge_secondary", None)
    if secondary:
        judge["additional_judges"] = [
            parse_additional_judge_flag(spec).model_dump() for spec in secondary
        ]
    return judge


def build_judge_config(args) -> JudgeConfig:
    """Build just the judge block for the calibration subcommand."""
    data = _load_config_mapping(getattr(args, "config", None))
    return JudgeConfig(**_judge_data(data, args))


def build_config(args) -> BenchConfig:
    data = _load_config_mapping(args.config)
    served_config = _cli_served_config(args)

    cli_targets = _cli_targets(args)
    if cli_targets:
        data["targets"] = [target.model_dump() for target in cli_targets]
    else:
        # CLI beats YAML: sampling flags also apply to YAML-declared targets.
        sampling = _cli_sampling(args)
        if sampling:
            targets = []
            for target in data.get("targets") or []:
                if not isinstance(target, dict):
                    targets.append(target)
                    continue
                target = dict(target)
                if sampling.get("sampling_mode") == "recommended":
                    target.pop("temperature", None)
                    target.pop("top_p", None)
                targets.append({**target, **sampling})
            data["targets"] = targets

    if served_config is not None:
        served_config_data = served_config.model_dump()
        data["targets"] = [
            {**target, "served_config": served_config_data}
            if isinstance(target, dict)
            else target
            for target in data.get("targets") or []
        ]

    judge = _judge_data(data, args)
    if judge:
        data["judge"] = JudgeConfig(**judge).model_dump()

    execution = dict(data.get("execution") or {})
    exec_runner = getattr(args, "exec_runner", None)
    if exec_runner == "local":
        # Selecting local on the CLI replaces a Docker block from YAML instead
        # of retaining an image that is invalid (and meaningless) for local
        # execution.
        execution = {"runner": "local"}
    elif exec_runner is not None:
        execution["runner"] = exec_runner
    execution_overrides = {
        "image": getattr(args, "exec_image", None),
        "cpus": getattr(args, "exec_cpus", None),
        "pids_limit": getattr(args, "exec_pids_limit", None),
        "disk_mb": getattr(args, "exec_disk_mb", None),
    }
    for key, value in execution_overrides.items():
        if value is not None:
            execution[key] = value
    if execution:
        data["execution"] = execution

    overrides = {
        "suite": args.suite,
        "limit": args.limit,
        "seed": args.seed,
        "attempts": getattr(args, "attempts", None),
        "concurrency": args.concurrency,
        "request_timeout_s": getattr(args, "request_timeout_s", None),
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
    if getattr(args, "no_progress", False):
        data["progress"] = False
    only = _split_csv(args.only)
    exclude = _split_csv(args.exclude)
    if only:
        data["only"] = only
    if exclude:
        data["exclude"] = exclude

    return BenchConfig.model_validate(data)
