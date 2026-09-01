#!/usr/bin/env python3
"""Measured serving verification: fixed-token matrix plus the replica placement gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = json.loads((HERE / "example.json").read_text(encoding="utf-8"))
REPLICAS = int(SPEC["allocation"]["replicas"])
TENSOR_PARALLEL = int(SPEC["allocation"]["tensor_parallel_size"])
SERVED_MODEL = SPEC["model"]["served_name"]
# Files whose bytes define the served configuration (the Qwen chat template
# is shared with the single-GPU example and mounted from there).
SERVED_CONFIG_FILES = (
    HERE / "example.json",
    HERE / "compose.yaml",
    HERE / "kairyu.yaml",
    HERE / "../qwen3.8-27b-1gpu/chat_template.jinja",
)


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
    os.environ.get("VERIFICATION_RESULTS_ROOT", ENVIRONMENT_STORAGE / "verification-results")
)
# Host-side view of the pool's placement_log_path bind mount (compose.yaml).
PLACEMENT_LOG = ENVIRONMENT_STORAGE / "placement-log" / Path(SPEC["pool"]["placement_log"]).name


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


def _serving_dataset(
    path: Path, requests: int, approximate_tokens: int, *, namespace: str
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
        # The run/row identity and the unique case come first, so neither
        # vLLM prefix caching nor Kairyu's prefix-aware placement can turn a
        # row into a shared-prefix microbenchmark. Server-reported usage
        # remains the source of truth for the actual token count.
        prompt = f"Run {namespace}, case {request}: " + " ".join(words)
        rows.append({"conversations": [{"from": "human", "value": prompt}]})
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def _bench(
    dataset: Path,
    *,
    requests: int,
    concurrency: int,
    max_tokens: int,
    results_dir: Path,
    log: Path,
) -> int:
    return _run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "verification/l1/performance/serving_bench.py"),
            "--base-url",
            f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}/v1",
            "--model",
            SERVED_MODEL,
            "--dataset",
            str(dataset),
            "--num-requests",
            str(requests),
            "--concurrency",
            str(concurrency),
            "--max-tokens",
            str(max_tokens),
            "--min-tokens",
            str(max_tokens),
            "--ignore-eos",
            "--temperature",
            "1.0",
            "--seed",
            "0",
            "--timeout",
            "86400",
            "--results-dir",
            str(results_dir),
            "--tensor-parallel",
            str(TENSOR_PARALLEL),
            "--dp-replicas",
            str(REPLICAS),
        ],
        log=log,
        check=False,
    )


def _placement_offset(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _placement_counts(path: Path, offset: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8") as stream:
        stream.seek(offset)
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("kind") != "replica":
                continue
            replica = row.get("replica_id", row.get("replica"))
            counts[str(replica)] += 1
    return counts


def _placement_report(
    path: Path,
    offset: int,
    *,
    expected_requests: int,
    replicas: int,
    gated: bool,
    max_share_of_mean: float,
    settle_s: float = 10.0,
) -> dict:
    """Per-replica placement counts for one row, from the pool's JSONL log.

    The log is written asynchronously, so poll until the row's placements
    have landed (or the settle window expires). The gate requires exactly
    ``expected_requests`` placements, every replica to receive traffic, and
    no replica to exceed ``max_share_of_mean``
    times the ideal even share; a serial row (c1) is reported only, because
    least-outstanding ties resolve to the lowest replica id by design.
    """

    deadline = time.monotonic() + settle_s
    counts = _placement_counts(path, offset)
    while sum(counts.values()) < expected_requests and time.monotonic() < deadline:
        time.sleep(0.5)
        counts = _placement_counts(path, offset)
    total = sum(counts.values())
    mean = expected_requests / replicas if replicas else 0.0
    largest = max(counts.values(), default=0)
    even = (
        total == expected_requests
        and len(counts) == replicas
        and largest <= max_share_of_mean * mean
    )
    return {
        "schema_version": 1,
        "placements": total,
        "expected_requests": expected_requests,
        "replicas": replicas,
        "per_replica": dict(sorted(counts.items())),
        "largest_share_of_mean": round(largest / mean, 3) if mean else None,
        "max_share_of_mean": max_share_of_mean,
        "gated": gated,
        "passed": even if gated else None,
    }


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


def serving(run_dir: Path) -> int:
    config = SPEC["verification"]["serving"]
    gate = config["placement_gate"]
    requests = int(config["requests_per_concurrency"])
    output_tokens = int(config["output_tokens"])
    prompt_tokens = int(config["prompt_tokens_approx"])
    run_dir.mkdir(parents=True, exist_ok=True)

    # One short request per replica so every replica's first-request work
    # (autotune, graph capture) is done before the measured rows.
    warmup_dataset = run_dir / "warmup-8k.json"
    _serving_dataset(warmup_dataset, REPLICAS, prompt_tokens, namespace=f"{run_dir.name}-warmup")
    if _bench(
        warmup_dataset,
        requests=REPLICAS,
        concurrency=REPLICAS,
        max_tokens=32,
        results_dir=run_dir / "warmup",
        log=run_dir / "warmup.log",
    ):
        print("warm-up row failed", file=sys.stderr)
        return 1

    failures = 0
    for concurrency in config["concurrency"]:
        row_dir = run_dir / f"serving-c{concurrency}"
        dataset = run_dir / f"serving-8k-c{concurrency}.json"
        _serving_dataset(
            dataset, requests, prompt_tokens, namespace=f"{run_dir.name}-c{concurrency}"
        )
        offset = _placement_offset(PLACEMENT_LOG)
        code = _bench(
            dataset,
            requests=requests,
            concurrency=concurrency,
            max_tokens=output_tokens,
            results_dir=row_dir,
            log=run_dir / f"serving-c{concurrency}.log",
        )
        if code == 0:
            code = _validate_serving_row(row_dir, requests, output_tokens)
        if code == 0:
            report = _placement_report(
                PLACEMENT_LOG,
                offset,
                expected_requests=requests,
                replicas=REPLICAS,
                gated=concurrency >= int(gate["min_concurrency"]),
                max_share_of_mean=float(gate["max_share_of_mean"]),
            )
            (row_dir / "placement.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"{row_dir.name}: placements {json.dumps(report['per_replica'])}")
            if report["passed"] is False:
                print(
                    f"{row_dir.name}: replica placement is not even "
                    f"(largest share {report['largest_share_of_mean']}x mean)",
                    file=sys.stderr,
                )
                code = 1
        failures += code != 0
        if code:
            break
    return 1 if failures else 0


def _served_config_sha256() -> str:
    digest = hashlib.sha256()
    for path in SERVED_CONFIG_FILES:
        digest.update(path.name.encode())
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
        print(
            "serving  fixed 8K-input/256-output TTFT and throughput at c=1,8,16,32,64 "
            "plus the per-row replica placement gate"
        )
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
