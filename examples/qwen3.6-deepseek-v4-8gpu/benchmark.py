#!/usr/bin/env python3
"""Independent/all serving, L2 orchestration, and Terminal-Bench 2.1 runner."""

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
        "BENCH_RESULTS_ROOT",
        ENVIRONMENT_STORAGE / "bench-results",
    )
)
TERMINAL_BENCH_DATASET = ENVIRONMENT_STORAGE / "bench-data/terminal-bench-2-1"
BENCHMARKS = (
    "serving-qwen",
    "serving-deepseek",
    "serving-auto",
    "serving-auto-max",
    "serving-auto-max-chat",
    "serving-auto-max-moa1",
    "serving-auto-max-moa2",
    "serving-auto-max-moa3",
    "serving-auto-max-moa4",
    "orchestration",
    "terminalbench-pilot",
    "terminalbench",
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


def _terminalbench_dataset(run_dir: Path) -> Path:
    expected = int(SPEC["benchmarks"]["terminalbench"]["dataset_tasks"])

    def task_count() -> int:
        if not TERMINAL_BENCH_DATASET.is_dir():
            return 0
        return sum(
            (path / "task.toml").is_file()
            for path in TERMINAL_BENCH_DATASET.iterdir()
            if path.is_dir()
        )

    if task_count() != expected:
        TERMINAL_BENCH_DATASET.parent.mkdir(parents=True, exist_ok=True)
        code = _run(
            [
                "harbor",
                "dataset",
                "download",
                SPEC["benchmarks"]["terminalbench"]["dataset"],
                "--output-dir",
                str(TERMINAL_BENCH_DATASET.parent),
                "--export",
                "--overwrite",
            ],
            log=run_dir / "terminalbench-dataset-download.log",
            check=False,
        )
        if code:
            raise RuntimeError(f"Harbor dataset download failed with exit code {code}")
    observed = task_count()
    if observed != expected:
        raise RuntimeError(
            f"Terminal-Bench dataset has {observed} tasks; expected exactly {expected}"
        )
    return TERMINAL_BENCH_DATASET


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
    expected_moa_samples: int | None = None,
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
                and (
                    expected_moa_samples is None
                    or stage.get("proposals") == expected_moa_samples
                )
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
    expected_moa_samples: int | None = None,
    warmup_requests: int | None = None,
    natural_completion: bool = False,
) -> int:
    config = SPEC["benchmarks"]["serving"]
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
            str(ROOT / "bench/serving_bench.py"),
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
                str(ROOT / "bench/serving_bench.py"),
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
                expected_moa_samples=expected_moa_samples,
                public_tokens=natural_completion,
            )
        if code:
            return code
    return 0


def serving_qwen(run_dir: Path) -> int:
    tier1 = SPEC["allocation"]["tier1"]
    return _serving(
        "qwen3.6-27b",
        run_dir,
        tensor_parallel=int(tier1["tensor_parallel_size"]),
        replicas=int(tier1["replicas"]),
    )


def serving_deepseek(run_dir: Path) -> int:
    return _serving("deepseek-v4-flash-0731", run_dir, tensor_parallel=4, replicas=1)


def serving_auto(run_dir: Path) -> int:
    tier1 = SPEC["allocation"]["tier1"]
    return _serving(
        "kairyu-auto",
        run_dir,
        tensor_parallel=int(tier1["tensor_parallel_size"]),
        replicas=int(tier1["replicas"]),
        expected_route="tier1",
    )


def serving_auto_max(run_dir: Path) -> int:
    return _serving_auto_max_candidate("kairyu-auto-max", run_dir, samples=3)


def serving_auto_max_chat(run_dir: Path) -> int:
    return _serving_auto_max_candidate("kairyu-auto-max-chat", run_dir, samples=3)


def _serving_auto_max_candidate(model: str, run_dir: Path, *, samples: int) -> int:
    return _serving(
        model,
        run_dir,
        tensor_parallel=4,
        replicas=1,
        expected_route="moa",
        expected_role="moa",
        expected_kind="synthesis",
        expected_moa_samples=samples,
        warmup_requests=4,
        natural_completion=True,
    )


def serving_auto_max_moa1(run_dir: Path) -> int:
    return _serving_auto_max_candidate("kairyu-auto-max-moa1", run_dir, samples=1)


def serving_auto_max_moa2(run_dir: Path) -> int:
    return _serving_auto_max_candidate("kairyu-auto-max-moa2", run_dir, samples=2)


def serving_auto_max_moa3(run_dir: Path) -> int:
    return _serving_auto_max_candidate("kairyu-auto-max-moa3", run_dir, samples=3)


def serving_auto_max_moa4(run_dir: Path) -> int:
    return _serving_auto_max_candidate("kairyu-auto-max-moa4", run_dir, samples=4)


def orchestration(run_dir: Path) -> int:
    return _run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "bench/tiered_auto_bench.py"),
            "--base-url",
            f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}/v1",
            "--direct-model",
            "qwen3.6-27b",
            "--auto-model",
            "kairyu-auto",
            "--auto-max-model",
            "kairyu-auto-max",
            "--gpu-count",
            "8",
            "--concurrency",
            "16",
            "--timeout",
            "86400",
            "--hardware",
            "8x RTX PRO 6000 Blackwell; Qwen FP8 TP1x4 + DeepSeek TP4/EP4",
            "--result",
            str(run_dir / "tiered-auto.json"),
        ],
        log=run_dir / "orchestration.log",
        check=False,
    )


def _terminalbench_run(
    run_dir: Path,
    *,
    models: tuple[str, ...],
    run_id: str,
    served_label: str,
    tasks: tuple[str, ...] = (),
) -> int:
    config = SPEC["benchmarks"]["terminalbench"]
    dataset = _terminalbench_dataset(run_dir)
    command = [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "kairyu.entrypoints.cli",
        "bench",
        "run",
        "--base-url",
        f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}/v1",
        *[part for model in models for part in ("--model", model)],
        "--served-config-label",
        served_label,
        "--served-config-sha256",
        _served_config_sha256(),
        "--no-vision",
        "--max-context-tokens",
        str(SPEC["models"]["tier1"]["max_context_tokens"]),
        "--max-output-tokens",
        str(config["max_output_tokens"]),
        "--request-timeout-s",
        "86400",
        "--suite",
        "accuracy",
        "--only",
        "terminal-bench",
        "--attempts",
        str(config["attempts"]),
        "--concurrency",
        str(config["concurrency"]),
        "--results-dir",
        str(run_dir),
        "--run-id",
        run_id,
        *(["--limit", str(len(tasks))] if tasks else []),
        "--no-progress",
    ]
    env = dict(os.environ)
    env["KAIRYU_TERMINAL_BENCH_PATH"] = str(dataset)
    if tasks:
        env["KAIRYU_TERMINAL_BENCH_TASKS"] = ",".join(tasks)
    else:
        env.pop("KAIRYU_TERMINAL_BENCH_TASKS", None)
    code = _run(
        command,
        log=run_dir / "terminalbench.log",
        check=False,
        env=env,
    )
    if code:
        return code
    return _validate_terminalbench(
        run_dir,
        run_id=run_id,
        models=models,
        expected_tasks=(len(tasks) if tasks else int(config["dataset_tasks"])),
    )


def terminalbench_pilot(run_dir: Path) -> int:
    config = SPEC["benchmarks"]["terminalbench"]
    return _terminalbench_run(
        run_dir,
        models=tuple(config["pilot_models"]),
        run_id="terminalbench-2.1-pilot",
        served_label=(
            "rtx-pro-6000-blackwell-8x-vllm-qwen-tp1x4-"
            "deepseek-tp4-ep4-quality-pilot"
        ),
        tasks=tuple(config["pilot_tasks"]),
    )


def terminalbench(run_dir: Path) -> int:
    # No task filter or limit: Harbor receives all 89 locally pinned tasks.
    # Unsupported sampling knobs are deliberately omitted at this boundary.
    return _terminalbench_run(
        run_dir,
        models=(SPEC["orchestration"]["auto_max_model"],),
        run_id="terminalbench-2.1",
        served_label=(
            "rtx-pro-6000-blackwell-8x-vllm-qwen-tp1x4-"
            "deepseek-tp4-ep4-auto-max-selected"
        ),
    )


def _validate_terminalbench(
    run_dir: Path,
    *,
    run_id: str,
    models: tuple[str, ...],
    expected_tasks: int,
) -> int:
    scoreboard_path = run_dir / run_id / "scoreboard.json"
    try:
        scoreboard = json.loads(scoreboard_path.read_text(encoding="utf-8"))
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"invalid Terminal-Bench scoreboard: {error}", file=sys.stderr)
        return 1
    for model in models:
        try:
            cell = scoreboard["cells"]["terminal-bench"][model]
        except (KeyError, TypeError) as error:
            print(f"Terminal-Bench has no cell for {model}: {error}", file=sys.stderr)
            return 1
        complete = (
            cell.get("status") == "completed"
            and isinstance(cell.get("score"), (int, float))
            and cell.get("n") == expected_tasks
            and cell.get("n_scored") == expected_tasks
        )
        if not complete:
            print(
                f"Terminal-Bench did not produce complete evidence for {model}: "
                f"status={cell.get('status')!r}, n={cell.get('n')!r}, "
                f"n_scored={cell.get('n_scored')!r}, score={cell.get('score')!r}",
                file=sys.stderr,
            )
            return 1
    return 0


def _served_config_sha256() -> str:
    digest = hashlib.sha256()
    for name in (
        "example.json",
        "compose.yaml",
        "kairyu.yaml",
        "auto.yaml",
        "auto-max.yaml",
        "auto-max-chat.yaml",
        "auto-max-moa1.yaml",
        "auto-max-moa2.yaml",
        "auto-max-moa3.yaml",
        "auto-max-moa4.yaml",
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
    parser.add_argument("benchmark", choices=(*BENCHMARKS, "all", "list"))
    parser.add_argument("--run-id")
    parser.add_argument("--no-start", action="store_true")
    args = parser.parse_args()
    if args.benchmark == "list":
        print("serving-qwen      Tier1 TP1 x 4 replica TTFT/throughput matrix")
        print("serving-deepseek  Tier2 TP4+EP4 TTFT/throughput matrix")
        print("serving-auto      routed Tier1/Tier2/MoA-2 matrix")
        print("serving-auto-max  forced MoA-3 + Tier2 synthesis matrix")
        print("serving-auto-max-chat  MoA-3 + non-thinking Tier2 synthesis matrix")
        print("serving-auto-max-moa1..4  fixed-fanout quality-candidate matrices")
        print("orchestration     fixed L2 direct/auto/auto-max latency and quality")
        print(
            "terminalbench-pilot  same four tasks on direct Qwen/DeepSeek, "
            "MoA3-chat, and MoA1..4-thinking"
        )
        print("terminalbench     complete Terminal-Bench 2.1, terminus-2, 500 turns")
        print("all               every benchmark above, continuing after failures")
        return

    _ensure_environment(args.no_start)
    run_dir = RESULTS_ROOT / (args.run_id or _run_id())
    run_dir.mkdir(parents=True, exist_ok=True)
    selected = BENCHMARKS if args.benchmark == "all" else (args.benchmark,)
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "started_at": datetime.now(UTC).isoformat(),
        "requested": args.benchmark,
        "selected": list(selected),
        "served_config_sha256": _served_config_sha256(),
        "spec": SPEC,
    }
    (run_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    results: dict[str, int] = {}
    for name in selected:
        try:
            function = globals()[name.replace("-", "_")]
            results[name] = function(run_dir / name)
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
