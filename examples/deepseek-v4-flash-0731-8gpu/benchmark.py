#!/usr/bin/env python3
"""Measured serving and full LiveCodeBench runner for this environment."""

from __future__ import annotations

import argparse
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
    # DeepSeek's tokenizer maps this stable ASCII pair close to one token per
    # repetition. The server-reported usage remains the source of truth.
    prompt = ("code " * approximate_tokens).strip()
    rows = [
        {"conversations": [{"from": "human", "value": f"{prompt}\nCase {index}."}]}
        for index in range(requests)
    ]
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
                str(ROOT / "verification/l1/performance/serving_bench.py"),
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
                str(SPEC["vllm"]["tensor_parallel_size"]),
                "--dp-replicas",
                "1",
            ],
            log=run_dir / f"serving-c{concurrency}.log",
            check=False,
        )
        failures += code != 0
    return 1 if failures else 0


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
        "rtx-pro-6000-blackwell-8x-vllm-tp8-ep8-dspark5",
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
        "--seed",
        str(config["seed"]),
        "--attempts",
        str(config["attempts"]),
        "--concurrency",
        str(config["concurrency"]),
        "--results-dir",
        str(run_dir),
        "--run-id",
        "livecodebench-full",
        "--exec-runner",
        "docker",
        "--exec-image",
        _execution_image(),
        "--no-progress",
    ]
    # Deliberately no --limit or --smoke: the adapter requires all 1,055 rows.
    code = _run(command, log=run_dir / "livecodebench.log", check=False)
    if code:
        return code
    return _validate_livecodebench(run_dir)


def _validate_livecodebench(run_dir: Path) -> int:
    """Fail closed unless the full, error-free 1,055-row pair was scored."""

    scoreboard_path = run_dir / "livecodebench-full" / "scoreboard.json"
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


def _served_config_sha256() -> str:
    import hashlib

    digest = hashlib.sha256()
    for name in ("compose.yaml", "kairyu.yaml", "deepseek-v4-0731.jinja"):
        path = HERE / name
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("serving", "livecodebench", "all", "list"))
    parser.add_argument("--run-id")
    parser.add_argument("--no-start", action="store_true")
    args = parser.parse_args()
    if args.benchmark == "list":
        print("serving       fixed 8K-input/256-output TTFT and throughput at c=1,8,16,32")
        print("livecodebench full pinned release_v6 (1,055 problems), pass@1")
        print("all           serving followed by LiveCodeBench")
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
    selected = ("serving", "livecodebench") if args.benchmark == "all" else (args.benchmark,)
    results: dict[str, int] = {}
    for name in selected:
        try:
            results[name] = serving(run_dir) if name == "serving" else livecodebench(run_dir)
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
