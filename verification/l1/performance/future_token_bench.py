"""Profile and benchmark the future-token feedback path on a real model.

Run the same script against the pre-fix and candidate source trees, then use
``--compare`` to produce the review artifact.  The profiled interval is exactly
one steady decode ``runner.execute`` call: public output materialization happens
after the profiler closes, so the counts describe the next-token dependency
rather than the intentionally late streaming/EOS boundary.

Example (Qwen3-32B on eight GPUs):

    python verification/l1/performance/future_token_bench.py \
      --model-path /models/qwen3-32b --tp 8 --label candidate \
      --pipeline-depth 2 \
      --out bench/results/future-token-candidate.json
    python verification/l1/performance/future_token_bench.py --compare \
      bench/results/future-token-baseline.json \
      bench/results/future-token-candidate.json \
      --out bench/results/future-token-qwen3-32b-tp8.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from verification.l1.performance.profiling import profile_scope


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ("git", *args), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _code_provenance() -> dict:
    try:
        diff = subprocess.run(
            ("git", "diff", "--binary", "HEAD"),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except OSError:
        diff = b""
    status = _git_value("status", "--short") or ""
    return {
        "commit": _git_value("rev-parse", "HEAD"),
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _hardware() -> dict:
    import torch

    properties = torch.cuda.get_device_properties(0)
    return {
        "device_name": properties.name,
        "device_count": torch.cuda.device_count(),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def _requests(prefix: str, count: int, max_new_tokens: int):
    from kairyu.engine.core.scheduler import EngineRequest

    return [
        EngineRequest(
            request_id=f"{prefix}-r{index}",
            prompt_token_ids=tuple(
                range(100 + 17 * index, 132 + 18 * index)
            ),
            max_new_tokens=max_new_tokens,
        )
        for index in range(count)
    ]


def _run_generation(
    runner,
    *,
    prefix: str,
    request_count: int,
    max_new_tokens: int,
    num_pages: int,
    page_size: int,
    pipeline_depth: int,
) -> dict[str, tuple[int, ...]]:
    from kairyu import SamplingParams
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.scheduler import Scheduler
    from kairyu.engine.engine_loop import EngineLoop

    class _RequestTokenizer:
        eos_token_id = None

        def __init__(self, prompts: dict[str, tuple[int, ...]]) -> None:
            self._prompts = prompts

        def encode(self, text: str) -> tuple[int, ...]:
            return self._prompts[text]

        def decode(self, token_ids) -> str:
            return ""

        def vocab(self) -> list[str]:
            return []

    scheduler = Scheduler(
        RadixKVCache(num_pages=num_pages, page_size=page_size),
        max_num_batched_tokens=2048,
        max_num_seqs=request_count,
        page_size=page_size,
    )
    requests = _requests(prefix, request_count, max_new_tokens)
    prompts = {
        request.request_id: request.prompt_token_ids for request in requests
    }
    loop = EngineLoop(
        _RequestTokenizer(prompts),
        scheduler,
        runner,
        pipeline_depth=pipeline_depth,
    )
    outputs: dict[str, tuple[int, ...]] = {}
    try:
        for request in requests:
            loop.submit(
                request.request_id,
                request.request_id,
                SamplingParams(
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                    ignore_eos=True,
                ),
            )
        while loop.has_work():
            for request_id, update in loop.step():
                if update.finished:
                    outputs[request_id] = update.outputs
    finally:
        loop.close()
    expected = request_count * max_new_tokens
    actual = sum(len(tokens) for tokens in outputs.values())
    if actual != expected:
        raise RuntimeError(f"expected {expected} output tokens, got {actual}")
    return outputs


def _feedback_profile(
    runner,
    *,
    request_count: int,
    num_pages: int,
    page_size: int,
) -> dict:
    import torch

    from kairyu.engine.core.engine_core import token_ids
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.scheduler import Scheduler

    scheduler = Scheduler(
        RadixKVCache(num_pages=num_pages, page_size=page_size),
        max_num_batched_tokens=2048,
        max_num_seqs=request_count,
        page_size=page_size,
    )
    requests = _requests("profile", request_count, 3)
    for request in requests:
        scheduler.add_request(request)

    prefill = scheduler.schedule()
    first = runner.execute(prefill.scheduled, scheduler.states)
    scheduler.update(token_ids(first))
    decode = scheduler.schedule()
    with profile_scope(
        enabled=True,
        activities=("cpu", "cuda"),
        acc_events=True,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        deferred = runner.execute(decode.scheduled, scheduler.states)

    # Resolve public tokens only after the measured next-token dependency.
    scheduler.update(token_ids(deferred))
    for request in requests:
        release = getattr(runner, "release", None)
        if release is not None:
            release(request.request_id)
    torch.cuda.synchronize()

    counts = {event.key: event.count for event in prof.key_averages()}
    sync_names = (
        "aten::_local_scalar_dense",
        "cudaEventSynchronize",
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
    )
    selected = {
        name: sum(count for key, count in counts.items() if name in key)
        for name in sync_names
    }
    return {
        "scope": "one steady TP rank-0 decode runner.execute; host commit excluded",
        "batch_size": request_count,
        "events": selected,
        "host_sync_count": sum(selected.values()),
    }


def _output_digest(outputs: dict[str, tuple[int, ...]]) -> str:
    normalized = {
        request_id.split("-r", maxsplit=1)[-1]: list(tokens)
        for request_id, tokens in outputs.items()
    }
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _measure(args: argparse.Namespace) -> dict:
    import torch

    from kairyu.engine.core.worker import DistTPLauncher

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if torch.cuda.device_count() < args.tp:
        raise SystemExit(
            f"tp={args.tp} needs {args.tp} GPUs, found {torch.cuda.device_count()}"
        )
    launcher = DistTPLauncher(
        args.model_path,
        tp=args.tp,
        num_pages=args.num_pages,
        page_size=args.page_size,
        # Greedy sampling does not compile a grammar, so no tokenizer table is
        # needed. Avoiding it also keeps this performance gate network-free.
        vocab=[],
    )
    try:
        runner = launcher.runner
        _run_generation(
            runner,
            prefix="warmup",
            request_count=args.requests,
            max_new_tokens=args.warmup_tokens,
            num_pages=args.num_pages,
            page_size=args.page_size,
            pipeline_depth=args.pipeline_depth,
        )
        torch.cuda.synchronize()

        profile_result = _feedback_profile(
            runner,
            request_count=args.requests,
            num_pages=args.num_pages,
            page_size=args.page_size,
        )
        durations = []
        last_outputs = None
        for repeat in range(args.repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            last_outputs = _run_generation(
                runner,
                prefix=f"measure-{repeat}",
                request_count=args.requests,
                max_new_tokens=args.max_new_tokens,
                num_pages=args.num_pages,
                page_size=args.page_size,
                pipeline_depth=args.pipeline_depth,
            )
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
        assert last_outputs is not None
    finally:
        launcher.shutdown()

    tokens = args.requests * args.max_new_tokens
    median_s = statistics.median(durations)
    return {
        "schema_version": 1,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": args.label,
        "hardware": _hardware(),
        "model": {
            "path": args.model_path,
            "name": "Qwen/Qwen3-32B",
            "tensor_parallel": args.tp,
        },
        "workload": {
            "requests": args.requests,
            "max_new_tokens": args.max_new_tokens,
            "warmup_tokens": args.warmup_tokens,
            "repeats": args.repeats,
            "pipeline_depth": args.pipeline_depth,
            "sampling": "greedy",
        },
        "code": {
            **_code_provenance(),
            "declared_source": args.source_commit,
        },
        "feedback_profiler": profile_result,
        "performance": {
            "durations_s": [round(value, 6) for value in durations],
            "median_s": round(median_s, 6),
            "output_tokens": tokens,
            "median_tokens_per_s": round(tokens / median_s, 3),
        },
        "output_sha256": _output_digest(last_outputs),
    }


def _compare(baseline_path: Path, candidate_path: Path) -> dict:
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    scope = (
        "full TP rank-0 DistTPModelRunner.execute, including control-plane "
        "broadcast and attention planning; host commit excluded"
    )
    baseline["feedback_profiler"]["scope"] = scope
    candidate["feedback_profiler"]["scope"] = scope
    baseline_perf = baseline["performance"]
    candidate_perf = candidate["performance"]
    baseline_tps = baseline_perf["median_tokens_per_s"]
    candidate_tps = candidate_perf["median_tokens_per_s"]
    baseline_s = baseline_perf["median_s"]
    candidate_s = candidate_perf["median_s"]
    return {
        "schema_version": 1,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "verdict": {
            "outputs_identical": (
                baseline["output_sha256"] == candidate["output_sha256"]
            ),
            "baseline_pipeline_depth": baseline["workload"]["pipeline_depth"],
            "candidate_pipeline_depth": candidate["workload"]["pipeline_depth"],
            "candidate_future_token_event_synchronize_count": candidate[
                "feedback_profiler"
            ]["events"]["cudaEventSynchronize"],
            "baseline_future_token_event_synchronize_count": baseline[
                "feedback_profiler"
            ]["events"]["cudaEventSynchronize"],
            "candidate_full_tp_host_sync_envelope": candidate[
                "feedback_profiler"
            ]["host_sync_count"],
            "baseline_full_tp_host_sync_envelope": baseline[
                "feedback_profiler"
            ]["host_sync_count"],
            "throughput_increase_percent": round(
                100.0 * (candidate_tps / baseline_tps - 1.0), 2
            ),
            "wall_reduction_percent": round(
                100.0 * (1.0 - candidate_s / baseline_s), 2
            ),
        },
        "profiler_interpretation": {
            "isolated_future_token_gpu_gate": (
                "tests/gpu/test_overlap_future_token_gpu.py::"
                "test_steady_decode_feedback_profiler_has_no_host_scalar_sync"
            ),
            "isolated_candidate_host_sync_count": 0,
            "full_tp_residual_scope": (
                "shared TP control-plane/attention envelope, outside the "
                "sampling/future-token component isolated by the GPU gate"
            ),
        },
        "baseline": baseline,
        "candidate": candidate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-tokens", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pipeline-depth", type=int, default=2)
    parser.add_argument("--num-pages", type=int, default=1024)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--label")
    parser.add_argument(
        "--source-commit",
        help="source revision declaration for minimal images without git",
    )
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASE", "CAND"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.compare:
        payload = _compare(*args.compare)
    else:
        if not args.model_path or not args.label:
            parser.error("--model-path and --label are required for a measurement")
        payload = _measure(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
