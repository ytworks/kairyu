#!/usr/bin/env python3
"""Serving, LiveCodeBench, and CharXiv runner for this environment."""

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
RESULTS_ROOT = ROOT / "bench/results/examples" / SPEC["environment"]


def _run(command: list[str], *, log: Path | None = None, check: bool = True) -> int:
    print("+ " + " ".join(command), flush=True)
    if log is None:
        return subprocess.run(command, cwd=ROOT, check=check).returncode
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if check and process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return process.returncode


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _ensure_environment(no_start: bool) -> None:
    if not no_start:
        _run([sys.executable, str(HERE / "control.py"), "up"])


def _serving_dataset(path: Path, requests: int, approximate_tokens: int) -> None:
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
        # Put the unique case first so prefix caching cannot turn the matrix
        # into a shared-prefix microbenchmark. Server-reported usage remains
        # the source of truth for the actual token count.
        words = [
            vocabulary[(request * 7 + position * 11) % len(vocabulary)]
            for position in range(approximate_tokens)
        ]
        prompt = f"Case {request}: " + " ".join(words)
        rows.append({"conversations": [{"from": "human", "value": prompt}]})
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def serving(run_dir: Path) -> int:
    config = SPEC["benchmarks"]["serving"]
    requests = int(config["requests_per_concurrency"])
    dataset = run_dir / "serving-8k.json"
    _serving_dataset(dataset, requests, int(config["prompt_tokens_approx"]))
    failures = 0
    for concurrency in config["concurrency"]:
        row_dir = run_dir / f"serving-c{concurrency}"
        code = _run(
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "bench/serving_bench.py"),
                "--base-url",
                f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}/v1",
                "--model",
                SPEC["model"]["served_name"],
                "--dataset",
                str(dataset),
                "--num-requests",
                str(requests),
                "--concurrency",
                str(concurrency),
                "--max-tokens",
                str(config["output_tokens"]),
                "--min-tokens",
                str(config["output_tokens"]),
                "--ignore-eos",
                "--temperature",
                "1.0",
                "--seed",
                "0",
                "--timeout",
                "86400",
                "--results-dir",
                str(row_dir),
                "--tensor-parallel",
                "1",
                "--dp-replicas",
                "1",
            ],
            log=run_dir / f"serving-c{concurrency}.log",
            check=False,
        )
        if code == 0:
            code = _validate_serving_row(row_dir, requests, int(config["output_tokens"]))
        failures += code != 0
        if code:
            break
    return 1 if failures else 0


def _validate_serving_row(row_dir: Path, requests: int, output_tokens: int) -> int:
    """Reject partial streams and nominally successful zero-token runs."""

    artifacts = list(row_dir.glob("*-serving.json"))
    if len(artifacts) != 1:
        print(
            f"serving row produced {len(artifacts)} result files; expected exactly one",
            file=sys.stderr,
        )
        return 1
    try:
        result = json.loads(artifacts[0].read_text(encoding="utf-8"))
        summary = result["summary"]
        samples = result["samples"]
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"invalid serving result: {error}", file=sys.stderr)
        return 1
    expected_total = requests * output_tokens
    complete = (
        summary.get("requests") == requests
        and summary.get("completion_tokens_total") == expected_total
        and isinstance(summary.get("output_tokens_per_s"), (int, float))
        and summary["output_tokens_per_s"] > 0
        and len(samples) == requests
        and all(sample.get("completion_tokens") == output_tokens for sample in samples)
    )
    if not complete:
        print(
            "serving row did not produce complete evidence: "
            f"requests={summary.get('requests')!r}, "
            f"completion_tokens_total={summary.get('completion_tokens_total')!r}, "
            f"output_tokens_per_s={summary.get('output_tokens_per_s')!r}, "
            f"samples={len(samples)!r}",
            file=sys.stderr,
        )
        return 1
    return 0


def _execution_image() -> str:
    tag = os.environ.get("BENCH_EXEC_IMAGE", "kairyu-bench-exec:local")
    inspect = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", tag],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if inspect.returncode:
        _run(
            [
                "docker",
                "build",
                "--file",
                "deploy/bench/Dockerfile.exec",
                "--tag",
                tag,
                "deploy/bench",
            ]
        )
        inspect = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", tag],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
    image_id = inspect.stdout.strip()
    if not image_id.startswith("sha256:"):
        raise SystemExit("benchmark execution image is not content-addressed")
    return image_id


def livecodebench(run_dir: Path) -> int:
    config = SPEC["benchmarks"]["livecodebench"]
    command = [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "kairyu.entrypoints.cli",
        "bench",
        "run",
        "--base-url",
        f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}/v1",
        "--model",
        SPEC["model"]["served_name"],
        "--served-config-label",
        _served_config_label(),
        "--served-config-sha256",
        _served_config_sha256(),
        "--no-vision",
        "--max-context-tokens",
        str(SPEC["model"]["max_context_tokens"]),
        "--max-output-tokens",
        str(config["max_output_tokens"]),
        "--request-timeout-s",
        "86400",
        "--reasoning-effort",
        config["reasoning_effort"],
        "--temperature",
        str(config["temperature"]),
        "--top-p",
        str(config["top_p"]),
        "--sampling-seed",
        str(config["seed"]),
        "--suite",
        "accuracy",
        "--only",
        "livecodebench",
        "--limit",
        str(config["problems"]),
        "--seed",
        str(config["seed"]),
        "--attempts",
        str(config["attempts"]),
        "--concurrency",
        str(config["concurrency"]),
        "--results-dir",
        str(run_dir),
        "--run-id",
        "livecodebench-20",
        "--exec-runner",
        "docker",
        "--exec-image",
        _execution_image(),
        "--no-progress",
    ]
    code = _run(command, log=run_dir / "livecodebench.log", check=False)
    if code:
        return code
    return _validate_livecodebench(run_dir)


def charxiv(run_dir: Path) -> int:
    config = SPEC["benchmarks"]["charxiv"]
    base_url = f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}/v1"
    model = SPEC["model"]["served_name"]
    command = [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "kairyu.entrypoints.cli",
        "bench",
        "run",
        "--base-url",
        base_url,
        "--model",
        model,
        "--served-config-label",
        _served_config_label(),
        "--served-config-sha256",
        _served_config_sha256(),
        "--max-context-tokens",
        str(SPEC["model"]["max_context_tokens"]),
        "--max-output-tokens",
        str(config["max_output_tokens"]),
        "--request-timeout-s",
        "86400",
        "--temperature",
        str(config["temperature"]),
        "--top-p",
        str(config["top_p"]),
        "--sampling-seed",
        str(config["seed"]),
        "--extra-body",
        json.dumps(config["extra_body"], separators=(",", ":")),
        "--judge-base-url",
        os.environ.get("JUDGE_BASE_URL", base_url),
        "--judge-model",
        os.environ.get("JUDGE_MODEL", model),
        "--suite",
        "accuracy",
        "--only",
        "charxiv-reasoning",
        "--limit",
        str(config["problems"]),
        "--seed",
        str(config["seed"]),
        "--attempts",
        str(config["attempts"]),
        "--concurrency",
        str(config["concurrency"]),
        "--results-dir",
        str(run_dir),
        "--run-id",
        "charxiv-10",
        "--no-progress",
    ]
    code = _run(command, log=run_dir / "charxiv.log", check=False)
    if code:
        return code
    return _validate_charxiv(run_dir, model=model)


def _served_config_label() -> str:
    prefix = "rtx-pro-6000-blackwell-1x-vllm-fp8"
    mtp_tokens = int(SPEC["vllm"]["mtp_speculative_tokens"])
    return f"{prefix}-mtp{mtp_tokens}" if mtp_tokens else f"{prefix}-no-mtp"


def _validate_livecodebench(run_dir: Path) -> int:
    """Fail closed unless all 20 deterministic rows are measured and scored."""

    scoreboard_path = run_dir / "livecodebench-20" / "scoreboard.json"
    try:
        scoreboard = json.loads(scoreboard_path.read_text(encoding="utf-8"))
        cell = scoreboard["cells"]["livecodebench"][SPEC["model"]["served_name"]]
        performance = cell["performance"]
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"invalid LiveCodeBench scoreboard: {error}", file=sys.stderr)
        return 1

    expected = int(SPEC["benchmarks"]["livecodebench"]["problems"])
    complete = (
        cell.get("status") == "completed"
        and cell.get("n") == expected
        and cell.get("n_scored") == expected
        and performance.get("requests") == expected
        and performance.get("errors") == 0
        and performance.get("unmeasured_requests") == 0
    )
    if not complete:
        print(
            "LiveCodeBench did not produce complete evidence: "
            f"status={cell.get('status')!r}, n={cell.get('n')!r}, "
            f"n_scored={cell.get('n_scored')!r}, "
            f"requests={performance.get('requests')!r}, "
            f"errors={performance.get('errors')!r}, "
            f"unmeasured={performance.get('unmeasured_requests')!r}",
            file=sys.stderr,
        )
        return 1
    return 0


def _validate_charxiv(run_dir: Path, *, model: str) -> int:
    scoreboard_path = run_dir / "charxiv-10" / "scoreboard.json"
    try:
        scoreboard = json.loads(scoreboard_path.read_text(encoding="utf-8"))
        cell = scoreboard["cells"]["charxiv-reasoning"][model]
        performance = cell["performance"]
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"invalid CharXiv scoreboard: {error}", file=sys.stderr)
        return 1

    expected = int(SPEC["benchmarks"]["charxiv"]["problems"])
    complete = (
        cell.get("status") == "completed"
        and cell.get("n") == expected
        and cell.get("n_scored") == expected
        and performance.get("requests") == expected
        and performance.get("errors") == 0
        and performance.get("unmeasured_requests") == 0
    )
    if not complete:
        print(
            "CharXiv did not produce complete 10-item evidence: "
            f"status={cell.get('status')!r}, n={cell.get('n')!r}, "
            f"n_scored={cell.get('n_scored')!r}, "
            f"requests={performance.get('requests')!r}, "
            f"errors={performance.get('errors')!r}, "
            f"unmeasured={performance.get('unmeasured_requests')!r}",
            file=sys.stderr,
        )
        return 1
    return 0


def _served_config_sha256() -> str:
    digest = hashlib.sha256()
    for name in ("compose.yaml", "kairyu.yaml"):
        path = HERE / name
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark",
        choices=("serving", "livecodebench", "charxiv", "all", "list"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--no-start", action="store_true")
    args = parser.parse_args()
    if args.benchmark == "list":
        print("serving       fixed 8K-input/256-output TTFT and throughput at c=1,8,16,32")
        print("livecodebench deterministic 20-item release_v6 pass@1 run")
        print("charxiv       deterministic 10-item vision and judge run")
        print("all           serving, LiveCodeBench-20, then CharXiv-10")
        return

    _ensure_environment(args.no_start)
    run_dir = RESULTS_ROOT / (args.run_id or _run_id())
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "started_at": datetime.now(UTC).isoformat(),
        "requested": args.benchmark,
        "served_config_sha256": _served_config_sha256(),
        "spec": SPEC,
    }
    (run_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    selected = (
        ("serving", "livecodebench", "charxiv")
        if args.benchmark == "all"
        else (args.benchmark,)
    )
    results: dict[str, int] = {}
    for name in selected:
        try:
            results[name] = globals()[name](run_dir)
        except Exception as error:
            print(f"{name} failed: {error}", file=sys.stderr)
            results[name] = 1
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["exit_codes"] = results
    (run_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"artifacts: {run_dir}")
    raise SystemExit(1 if any(results.values()) else 0)


if __name__ == "__main__":
    main()
