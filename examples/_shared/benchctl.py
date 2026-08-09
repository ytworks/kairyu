#!/usr/bin/env python3
"""Report-owning benchmark controller shared by frontier examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

CONCURRENCY = (1, 2, 4, 8, 16, 32, 64)
SHAREGPT_SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"


def _root(environment_dir: Path) -> Path:
    return environment_dir.resolve().parents[1]


def _rows(root: Path, suite: str) -> tuple[str, ...]:
    sys.path.insert(0, str(root))
    from kairyu.bench.adapters import suite_info

    return suite_info(suite).row_order


def _ids(root: Path, orchestration: bool) -> tuple[str, ...]:
    ids = ["accuracy", *[f"accuracy:{row}" for row in _rows(root, "accuracy")]]
    ids += ["structured", "long-context"]
    ids += [f"long-context:{row}" for row in _rows(root, "long-context")]
    ids.append("serving")
    if orchestration:
        ids.append("orchestration")
    return tuple(ids)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_url(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def _write_report(directory: Path, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {payload['benchmark_id']}",
        "",
        f"- Status: **{payload['status']}**",
        f"- Backend: `{payload['backend']}`",
        f"- Exit code: `{payload['exit_code']}`",
        f"- Started: `{payload['started_at']}`",
        f"- Completed: `{payload['completed_at']}`",
    ]
    if payload.get("error"):
        lines += ["", "## Error", "", str(payload["error"])]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evidence(spec_dir: Path, spec: dict, backend: str, output: Path, port: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    commands = {
        "gpu.txt": [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,compute_cap,pci.bus_id",
            "--format=csv,noheader",
        ],
        "topology.txt": ["nvidia-smi", "topo", "-m"],
        "git.txt": ["git", "status", "--short"],
    }
    for filename, command in commands.items():
        result = subprocess.run(command, cwd=_root(spec_dir), text=True, capture_output=True)
        (output / filename).write_text(result.stdout + result.stderr, encoding="utf-8")
    (output / "example.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for endpoint, filename in (("/backends", "backends.json"), ("/v1/models", "models.json")):
        payload = _json_url(f"http://127.0.0.1:{port}{endpoint}")
        (output / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    attestation = {
        "schema_version": 1,
        "environment": spec["environment"],
        "backend": backend,
        "models": spec["models"],
        "compose_sha256": _sha256(spec_dir / "compose.yaml"),
        "credentials_recorded": False,
    }
    (output / "attestation.json").write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _models(spec: dict, benchmark_id: str) -> list[str]:
    direct = [
        "qwen3.6-27b" if model["repo"].startswith("Qwen/") else "deepseek-v4-flash-0731"
        for model in spec["models"]
    ]
    if benchmark_id == "orchestration":
        return ["qwen3.6-27b", "deepseek-v4-flash-0731", "kairyu-auto", "kairyu-auto-max"]
    if spec["environment"] == "qwen3.6-deepseek-v4-8gpu":
        return [*direct, "kairyu-auto", "kairyu-auto-max"]
    return direct


def _quality_command(
    root: Path,
    spec: dict,
    benchmark_id: str,
    port: int,
    output: Path,
    *,
    models: list[str] | None = None,
    max_context_tokens: int | None = None,
) -> list[str]:
    suite, _, row = benchmark_id.partition(":")
    command = [
        str(root / ".venv/bin/python"),
        "-m",
        "kairyu.entrypoints.cli",
        "bench",
        "run",
        "--suite",
        suite,
        "--base-url",
        f"http://127.0.0.1:{port}/v1",
        "--no-vision",
        "--reasoning-effort",
        "max",
        "--results-dir",
        str(output / "suite-results"),
        "--run-id",
        benchmark_id.replace(":", "-"),
        "--no-progress",
    ]
    for model in models or _models(spec, benchmark_id):
        command.extend(["--model", model])
    if suite == "accuracy":
        judge_model = (
            "deepseek-v4-flash-0731"
            if len(spec["models"]) > 1
            or spec["models"][0]["repo"].startswith("deepseek-ai/")
            else "qwen3.6-27b"
        )
        command.extend(
            [
                "--judge-base-url",
                f"http://127.0.0.1:{port}/v1",
                "--judge-model",
                judge_model,
                "--judge-reasoning-effort",
                "low",
            ]
        )
    if row:
        command.extend(["--only", row])
    max_context = max_context_tokens or min(
        model["max_context_tokens"] for model in spec["models"]
    )
    command.extend(["--max-context-tokens", str(max_context)])
    return command


def _serving_commands(
    root: Path,
    spec: dict,
    port: int,
    output: Path,
) -> list[list[str]]:
    dataset = os.environ.get("SHAREGPT_DATASET")
    if not dataset:
        raise RuntimeError("SHAREGPT_DATASET must point to the pinned ShareGPT JSON")
    dataset_path = Path(dataset)
    if _sha256(dataset_path) != SHAREGPT_SHA256:
        raise RuntimeError("SHAREGPT_DATASET SHA-256 does not match the fixed corpus")
    commands = []
    for item in spec["models"]:
        model = (
            "qwen3.6-27b"
            if item["repo"].startswith("Qwen/")
            else "deepseek-v4-flash-0731"
        )
        commands.append(
            [
                str(root / ".venv/bin/python"),
                str(root / "examples/_shared/serving_matrix.py"),
                "--base-url",
                f"http://127.0.0.1:{port}/v1",
                "--model",
                model,
                "--repo",
                item["repo"],
                "--revision",
                item["revision"],
                "--native-max",
                str(item["max_context_tokens"]),
                "--dataset",
                str(dataset_path),
                "--output",
                str(output / f"serving-{model}.json"),
            ]
        )
    return commands


def _orchestration_command(root: Path, port: int, output: Path) -> list[str]:
    return [
        str(root / ".venv/bin/python"),
        str(root / "bench/tiered_auto_bench.py"),
        "--base-url",
        f"http://127.0.0.1:{port}/v1",
        "--direct-model",
        "qwen3.6-27b",
        "--auto-model",
        "kairyu-auto",
        "--auto-max-model",
        "kairyu-auto-max",
        "--gpu-count",
        "8",
        "--result",
        str(output / "tiered-auto.json"),
    ]


def _run_one(
    spec_dir: Path,
    spec: dict,
    backend: str,
    benchmark_id: str,
    run_root: Path,
) -> bool:
    root = _root(spec_dir)
    output = run_root / benchmark_id.replace(":", "_")
    output.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", spec["port"]))
    started = datetime.now(UTC).isoformat()
    exit_code = 0
    error = None
    commands: list[list[str]] = []
    try:
        _evidence(spec_dir, spec, backend, output / "evidence", port)
        if benchmark_id.startswith(("accuracy", "structured", "long-context")):
            if benchmark_id.startswith("long-context") and len(spec["models"]) > 1:
                commands = [
                    _quality_command(
                        root,
                        spec,
                        benchmark_id,
                        port,
                        output / model["slug"],
                        models=[
                            "qwen3.6-27b"
                            if model["repo"].startswith("Qwen/")
                            else "deepseek-v4-flash-0731"
                        ],
                        max_context_tokens=model["max_context_tokens"],
                    )
                    for model in spec["models"]
                ]
            else:
                commands = [_quality_command(root, spec, benchmark_id, port, output)]
        elif benchmark_id == "serving":
            commands = _serving_commands(root, spec, port, output)
        elif benchmark_id == "orchestration":
            commands = [_orchestration_command(root, port, output)]
        else:
            raise RuntimeError(f"unsupported benchmark id: {benchmark_id}")
        with (output / "raw.log").open("w", encoding="utf-8") as log:
            for command in commands:
                result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    exit_code = result.returncode
                    break
    except Exception as caught:
        exit_code = 1
        error = f"{type(caught).__name__}: {caught}"
    payload = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "backend": backend,
        "status": "passed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "started_at": started,
        "completed_at": datetime.now(UTC).isoformat(),
        "commands": commands,
        "error": error,
    }
    _write_report(output, payload)
    return exit_code == 0


def _start(spec_dir: Path, backend: str) -> None:
    subprocess.run([str(spec_dir / "run.sh"), backend, "up"], check=True)


def _stop(spec_dir: Path, backend: str) -> None:
    subprocess.run([str(spec_dir / "run.sh"), backend, "down"], check=False)


def _backend_run(spec_dir: Path, spec: dict, backend: str, requested: str) -> tuple[Path, bool]:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = (
        _root(spec_dir)
        / "bench/results/examples"
        / spec["environment"]
        / backend
        / run_id
    )
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        _start(spec_dir, backend)
    except Exception as error:
        now = datetime.now(UTC).isoformat()
        failure = {
            "schema_version": 1,
            "benchmark_id": "startup",
            "backend": backend,
            "status": "failed",
            "exit_code": 1,
            "started_at": now,
            "completed_at": now,
            "commands": [],
            "error": f"{type(error).__name__}: {error}",
        }
        _write_report(run_root / "startup", failure)
        integrated = {
            "schema_version": 1,
            "environment": spec["environment"],
            "backend": backend,
            "run_id": run_id,
            "passed": False,
            "benchmarks": {"startup": False},
            "error": failure["error"],
        }
        (run_root / "integrated.json").write_text(
            json.dumps(integrated, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (run_root / "integrated.md").write_text(
            "# Integrated report\n\nStartup failed; see `startup/report.md`.\n",
            encoding="utf-8",
        )
        return run_root, False
    orchestration = len(spec["models"]) > 1
    ids = _ids(_root(spec_dir), orchestration)
    selected = (
        tuple(benchmark_id for benchmark_id in ids if ":" not in benchmark_id)
        if requested == "all"
        else (requested,)
    )
    results = {
        benchmark_id: _run_one(spec_dir, spec, backend, benchmark_id, run_root)
        for benchmark_id in selected
    }
    integrated = {
        "schema_version": 1,
        "environment": spec["environment"],
        "backend": backend,
        "run_id": run_id,
        "passed": all(results.values()),
        "benchmarks": results,
    }
    (run_root / "integrated.json").write_text(
        json.dumps(integrated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    table = ["# Integrated report", "", "| Benchmark | Passed |", "|---|---:|"]
    table += [f"| {name} | {'yes' if passed else 'no'} |" for name, passed in results.items()]
    (run_root / "integrated.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    return run_root, all(results.values())


def _compare(spec_dir: Path, spec: dict, requested: str) -> bool:
    roots: dict[str, Path] = {}
    passed = True
    for backend in ("vllm", "kairyu"):
        try:
            roots[backend], ok = _backend_run(spec_dir, spec, backend, requested)
            passed = passed and ok
        finally:
            _stop(spec_dir, backend)
    compare_root = _root(spec_dir) / "bench/results/examples" / spec["environment"] / "compare"
    compare_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "environment": spec["environment"],
        "passed": passed,
        "quality_publication_allowed": passed,
        "vllm": str(roots.get("vllm", "")),
        "kairyu": str(roots.get("kairyu", "")),
        "note": "Performance is publishable only when every paired quality gate passes.",
    }
    (compare_root / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (compare_root / "comparison.md").write_text(
        "# Backend comparison\n\n"
        f"Overall gate: **{'PASS' if passed else 'FAIL'}**\n\n"
        "Kairyu performance is not a formal result unless paired quality is non-inferior.\n",
        encoding="utf-8",
    )
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("environment_dir", type=Path)
    parser.add_argument("backend", nargs="?")
    parser.add_argument("benchmark_id", nargs="?")
    args = parser.parse_args()
    spec_dir = args.environment_dir.resolve()
    spec = json.loads((spec_dir / "example.json").read_text(encoding="utf-8"))
    valid_ids = _ids(_root(spec_dir), len(spec["models"]) > 1)
    if args.backend == "list" and args.benchmark_id is None:
        print("\n".join(valid_ids))
        return
    if args.backend not in {"vllm", "kairyu", "compare"} or args.benchmark_id is None:
        raise SystemExit(
            "usage: ./bench.sh <vllm|kairyu|compare> <benchmark-id|all> | "
            "./bench.sh list"
        )
    if args.benchmark_id != "all" and args.benchmark_id not in valid_ids:
        raise SystemExit(f"unknown benchmark id: {args.benchmark_id}")
    if args.backend == "compare":
        passed = _compare(spec_dir, spec, args.benchmark_id)
    else:
        _, passed = _backend_run(spec_dir, spec, args.backend, args.benchmark_id)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
