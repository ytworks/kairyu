#!/usr/bin/env python3
"""Verifier-gated L2 product serving verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = json.loads((HERE / "example.json").read_text(encoding="utf-8"))


def _nvme_root() -> Path:
    configured = Path(os.environ.get("NVME_STORAGE_ROOT", SPEC["storage"]["root"]))
    if not configured.is_absolute():
        raise SystemExit("NVME_STORAGE_ROOT must be an absolute path below /mnt/nvme")
    root = configured.resolve()
    nvme = Path("/mnt/nvme")
    if root != nvme and nvme not in root.parents:
        raise SystemExit("NVME_STORAGE_ROOT must be /mnt/nvme or one of its descendants")
    return root


STORAGE_ROOT = _nvme_root()
ENVIRONMENT_STORAGE = STORAGE_ROOT / "model-volumes" / SPEC["environment"]
RESULTS_ROOT = Path(
    os.environ.get(
        "VERIFICATION_RESULTS_ROOT",
        ENVIRONMENT_STORAGE / "verification-results",
    )
)


def _run(
    command: list[str],
    *,
    log: Path | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> int:
    print("+ " + " ".join(command), flush=True)
    if log is None:
        return subprocess.run(command, cwd=ROOT, check=check, env=env).returncode
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
        )
    if check and process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return process.returncode


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _ensure_environment(no_start: bool) -> None:
    if not no_start:
        _run([sys.executable, str(HERE / "control.py"), "up"])



def _serving_dataset(
    path: Path,
    requests: int,
    approximate_tokens: int,
    *,
    namespace: str,
    response_instruction: str = "",
) -> None:
    vocabulary = (
        "code",
        "review",
        "function",
        "module",
        "request",
        "result",
        "verify",
        "runtime",
        "system",
        "design",
        "state",
        "input",
        "output",
        "stream",
        "cache",
        "token",
    )
    rows = []
    for request in range(requests):
        words = [
            vocabulary[(request * 7 + position * 11) % len(vocabulary)]
            for position in range(approximate_tokens)
        ]
        # Put the run/row identity before the repeated body so another
        # concurrency row cannot become a full-prefix-cache microbenchmark.
        prompt = f"Run {namespace}, case {request}: " + " ".join(words)
        if response_instruction:
            prompt += "\n\n" + response_instruction
        rows.append({"conversations": [{"from": "human", "value": prompt}]})
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def _validate_serving_row(
    row_dir: Path,
    requests: int,
    output_tokens: int,
    *,
    expected_route: str | None = None,
    expected_role: str = "direct",
    expected_kind: str = "generation",
    public_tokens: bool = False,
) -> int:
    artifacts = list(row_dir.glob("*-serving.json"))
    if len(artifacts) != 1:
        print(f"serving row produced {len(artifacts)} result files", file=sys.stderr)
        return 1
    try:
        result = json.loads(artifacts[0].read_text(encoding="utf-8"))
        summary = result["summary"]
        samples = result["samples"]
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"invalid serving result: {error}", file=sys.stderr)
        return 1
    if public_tokens:
        complete = (
            summary.get("requests") == requests
            and isinstance(summary.get("public_completion_tokens_total"), int)
            and summary["public_completion_tokens_total"] > 0
            and isinstance(summary.get("public_output_tokens_per_s"), (int, float))
            and summary["public_output_tokens_per_s"] > 0
            and len(samples) == requests
            and all(
                isinstance(sample.get("public_completion_tokens"), int)
                and sample["public_completion_tokens"] > 0
                for sample in samples
            )
        )
    else:
        expected_total = requests * output_tokens
        complete = (
            summary.get("requests") == requests
            and summary.get("completion_tokens_total") == expected_total
            and isinstance(summary.get("output_tokens_per_s"), (int, float))
            and summary["output_tokens_per_s"] > 0
            and len(samples) == requests
            and all(sample.get("completion_tokens") == output_tokens for sample in samples)
        )
    if complete and expected_route is not None:
        complete = all(
            sample.get("trace", {}).get("status") == "valid"
            and any(
                stage.get("node") == expected_route
                and stage.get("role") == expected_role
                and stage.get("kind") == expected_kind
                and stage.get("status") == "success"
                for stage in sample.get("trace", {}).get("stages", [])
            )
            for sample in samples
        )
    if not complete:
        print("serving row did not produce complete fixed-token evidence", file=sys.stderr)
        return 1
    return 0


def _serving(
    model: str,
    run_dir: Path,
    *,
    tensor_parallel: int,
    replicas: int,
    expected_route: str | None = None,
    expected_role: str = "direct",
    expected_kind: str = "generation",
    warmup_requests: int | None = None,
    natural_completion: bool = False,
) -> int:
    config = SPEC["verification"]["serving"]
    requests = int(config["requests_per_concurrency"])
    output_tokens = int(config["output_tokens"])
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_tokens = int(config["prompt_tokens_approx"])
    warmup_requests = warmup_requests or max(1, replicas)
    warmup_dataset = run_dir / "warmup-8k.json"
    response_instruction = (
        "Synthesize a useful final answer of approximately 256 output tokens. "
        "Return only that answer; do not expose candidates or private reasoning."
        if natural_completion
        else ""
    )
    _serving_dataset(
        warmup_dataset,
        warmup_requests,
        prompt_tokens,
        namespace=f"{run_dir.parent.name}-{run_dir.name}-warmup",
        response_instruction=response_instruction,
    )
    request_max_tokens = (
        int(config["auto_max_combined_max_tokens"])
        if natural_completion
        else output_tokens
    )
    fixed_output_args = (
        []
        if natural_completion
        else ["--min-tokens", "32", "--ignore-eos"]
    )
    public_tokenizer_args = (
        [
            "--public-tokenizer-url",
            f"http://127.0.0.1:{os.environ.get('DEEPSEEK_L1_PORT', 8005)}/tokenize",
            "--public-tokenizer-model",
            "deepseek-v4-flash-0731",
        ]
        if natural_completion
        else []
    )
    warmup_code = _run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "verification/l1/performance/serving_bench.py"),
            "--base-url",
            f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}/v1",
            "--model",
            model,
            "--dataset",
            str(warmup_dataset),
            "--num-requests",
            str(warmup_requests),
            "--concurrency",
            str(warmup_requests),
            "--max-tokens",
            str(request_max_tokens if natural_completion else 32),
            *fixed_output_args,
            "--temperature",
            "0.0" if natural_completion else "1.0",
            "--seed",
            "0",
            "--timeout",
            "86400",
            "--results-dir",
            str(run_dir / "warmup"),
            "--tensor-parallel",
            str(tensor_parallel),
            "--dp-replicas",
            str(replicas),
            *public_tokenizer_args,
            *(["--stage-trace"] if expected_route is not None else []),
        ],
        log=run_dir / "warmup.log",
        check=False,
    )
    if warmup_code:
        return warmup_code
    for concurrency in config["concurrency"]:
        dataset = run_dir / f"serving-c{concurrency}-8k.json"
        _serving_dataset(
            dataset,
            requests,
            prompt_tokens,
            namespace=f"{run_dir.parent.name}-{run_dir.name}-c{concurrency}",
            response_instruction=response_instruction,
        )
        row_dir = run_dir / f"serving-c{concurrency}"
        code = _run(
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "verification/l1/performance/serving_bench.py"),
                "--base-url",
                f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}/v1",
                "--model",
                model,
                "--dataset",
                str(dataset),
                "--num-requests",
                str(requests),
                "--concurrency",
                str(concurrency),
                "--max-tokens",
                str(request_max_tokens),
                *(
                    []
                    if natural_completion
                    else ["--min-tokens", str(output_tokens), "--ignore-eos"]
                ),
                "--temperature",
                "0.0" if natural_completion else "1.0",
                "--seed",
                "0",
                "--timeout",
                "86400",
                "--results-dir",
                str(row_dir),
                "--tensor-parallel",
                str(tensor_parallel),
                "--dp-replicas",
                str(replicas),
                *public_tokenizer_args,
                *(["--stage-trace"] if expected_route is not None else []),
            ],
            log=run_dir / f"serving-c{concurrency}.log",
            check=False,
        )
        if code == 0:
            code = _validate_serving_row(
                row_dir,
                requests,
                output_tokens,
                expected_route=expected_route,
                expected_role=expected_role,
                expected_kind=expected_kind,
                public_tokens=natural_completion,
            )
        if code:
            return code
    return 0


def serving_auto_max(run_dir: Path) -> int:
    return _serving(
        SPEC["orchestration"]["auto_max_model"],
        run_dir,
        tensor_parallel=4,
        replicas=1,
        expected_route="publisher",
        expected_role="publisher",
        expected_kind="generation",
        warmup_requests=4,
        natural_completion=True,
    )



def _served_config_sha256() -> str:
    digest = hashlib.sha256()
    for name in (
        "example.json",
        "compose.yaml",
        "kairyu.yaml",
        "auto-max.yaml",
        "router.json",
        "deepseek-thinking.jinja",
    ):
        path = HERE / name
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verification", choices=("serving-auto-max", "list"))
    parser.add_argument("--run-id")
    parser.add_argument("--no-start", action="store_true")
    args = parser.parse_args()
    if args.verification == "list":
        print("serving-auto-max  verifier-gated product DAG serving matrix")
        return

    _ensure_environment(args.no_start)
    run_dir = RESULTS_ROOT / (args.run_id or _run_id())
    run_dir.mkdir(parents=True, exist_ok=True)
    target_dir = run_dir / "serving-auto-max"
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "started_at": datetime.now(UTC).isoformat(),
        "requested": args.verification,
        "served_config_sha256": _served_config_sha256(),
        "spec": SPEC,
    }
    (run_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    try:
        code = serving_auto_max(target_dir)
    except Exception as error:
        print(f"serving-auto-max failed: {error}", file=sys.stderr)
        code = 1
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["exit_codes"] = {"serving-auto-max": code}
    (run_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"artifacts: {run_dir}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
