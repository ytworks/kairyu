#!/usr/bin/env python3
"""Measured serving verification for this environment."""

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
RESULTS_ROOT = ROOT / "verification/results/examples" / SPEC["environment"]


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
    config = SPEC["verification"]["serving"]
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
    parser.add_argument("verification", choices=("serving", "list"))
    parser.add_argument("--run-id")
    parser.add_argument("--no-start", action="store_true")
    args = parser.parse_args()
    if args.verification == "list":
        print("serving  fixed 8K-input/256-output TTFT and throughput at c=1,8,16,32")
        return

    _ensure_environment(args.no_start)
    run_dir = RESULTS_ROOT / (args.run_id or _run_id())
    run_dir.mkdir(parents=True, exist_ok=True)
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
        code = serving(run_dir)
    except Exception as error:
        print(f"serving failed: {error}", file=sys.stderr)
        code = 1
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["exit_codes"] = {"serving": code}
    (run_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"artifacts: {run_dir}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
