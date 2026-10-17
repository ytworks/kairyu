#!/usr/bin/env python3
"""Measured serving and L1/L3 Responses-equivalence verification for this environment."""

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


# Gate results stay on the environment's NVMe storage, beside its model and cache.
ENVIRONMENT_STORAGE = _nvme_root() / "model-volumes" / SPEC["environment"]
RESULTS_ROOT = Path(
    os.environ.get("VERIFICATION_RESULTS_ROOT", ENVIRONMENT_STORAGE / "verification-results")
)


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


def _api_url() -> str:
    return f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}"


def serving(run_dir: Path) -> int:
    config = SPEC["verification"]["serving"]
    requests = int(config["requests_per_concurrency"])
    max_output_tokens = int(config["max_output_tokens"])
    dataset = run_dir / "serving-8k.json"
    _serving_dataset(dataset, requests, int(config["prompt_tokens_approx"]))
    failures = 0
    for concurrency in config["concurrency"]:
        row_dir = run_dir / f"serving-c{concurrency}"
        # Rows run on the public Responses API (m11 D7 amendment 2026-10-17).
        # It has no ignore_eos/min_tokens/seed, so each request ends naturally
        # within max_output_tokens; rows are not comparable with the
        # fixed-length chat rows recorded before 2026-10-17.
        code = _run(
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "verification/l1/performance/serving_bench.py"),
                "--base-url",
                f"{_api_url()}/v1",
                "--model",
                SPEC["model"]["served_name"],
                "--api",
                config["api"],
                "--dataset",
                str(dataset),
                "--num-requests",
                str(requests),
                "--concurrency",
                str(concurrency),
                "--max-tokens",
                str(max_output_tokens),
                "--temperature",
                "1.0",
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
            code = _validate_serving_row(row_dir, requests, max_output_tokens)
        failures += code != 0
        if code:
            break
    return 1 if failures else 0


def _validate_serving_row(row_dir: Path, requests: int, max_output_tokens: int) -> int:
    """Reject partial streams and empty answers; each request ends naturally."""

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
    tokens = [sample.get("completion_tokens") for sample in samples]
    complete = (
        summary.get("requests") == requests
        and len(samples) == requests
        and all(isinstance(count, int) and 1 <= count <= max_output_tokens for count in tokens)
        and summary.get("completion_tokens_total") == sum(tokens)
        and isinstance(summary.get("output_tokens_per_s"), (int, float))
        and summary["output_tokens_per_s"] > 0
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


def _l1_container() -> str:
    """The running vLLM container of this Compose project (no host port)."""

    project = SPEC["environment"].replace(".", "-")
    listed = subprocess.run(
        [
            "docker",
            "ps",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--filter",
            "label=com.docker.compose.service=vllm",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    if len(listed) != 1:
        raise RuntimeError(f"expected one running vllm container in {project}, found {listed}")
    return listed[0]


def api_equivalence(run_dir: Path) -> int:
    """The same Responses requests to vLLM L1 directly and to Kairyu L3."""

    container = _l1_container()
    served = SPEC["model"]["served_name"]
    return _run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "verification/product/correctness/upstream_api_equivalence.py"),
            "--l1",
            "http://127.0.0.1:8000/v1",
            "--l1-docker",
            container,
            "--l1-model",
            served,
            "--l1-metrics",
            f"{container}=http://127.0.0.1:8000/metrics",
            "--l3",
            f"{_api_url()}/v1",
            "--l3-model",
            served,
            "--cases",
            "text,stream,effort,tools,image,cap,openwebui",
            # The template clamps every effort to low; each still thinks.
            "--efforts",
            "low,high,xhigh",
            "--expect-reasoning",
            "--exact-input-tokens",
            "--out",
            str(run_dir / "api-equivalence.json"),
        ],
        log=run_dir / "api-equivalence.log",
        check=False,
    )


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
    parser.add_argument("verification", choices=("serving", "api-equivalence", "list"))
    parser.add_argument("--run-id")
    parser.add_argument("--no-start", action="store_true")
    args = parser.parse_args()
    if args.verification == "list":
        print(
            "serving          8K input, up to 256 output (natural completion) TTFT and "
            "throughput at c=1,8,16,32 on /v1/responses"
        )
        print(
            "api-equivalence  the same Responses requests to vLLM L1 and Kairyu L3 agree, "
            "and L3 reached L1 on /v1/responses"
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
    target = serving if args.verification == "serving" else api_equivalence
    try:
        code = target(run_dir)
    except Exception as error:
        print(f"{args.verification} failed: {error}", file=sys.stderr)
        code = 1
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["exit_codes"] = {args.verification: code}
    (run_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"artifacts: {run_dir}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
