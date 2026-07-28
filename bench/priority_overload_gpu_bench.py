#!/usr/bin/env python3
"""Production-shaped Qwen3-32B TP8 F5a overload measurement.

The gateway must expose one ReplicaPool plus Batch API, with authenticated
class mapping ``interactive=0`` and ``batch=1``. Capacity is calibrated first.
Interactive then arrives open-loop at 0.5x capacity while an instantaneous
batch backlog contributes 1.5x, making the planned offered load exactly 2x.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import httpx


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of no values")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def _prompt(index: int) -> str:
    return (
        f"salt-{index:06d} begins this independent latency measurement. "
        "Reply with a short factual sentence about reliable software testing."
    )


def _body(model: str, index: int, *, stream: bool) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": _prompt(index)}],
        "temperature": 0,
        "max_tokens": 8,
        "min_tokens": 8,
        "ignore_eos": True,
        "stream": stream,
    }


async def _one_unary(
    client: httpx.AsyncClient,
    model: str,
    index: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post(
                "/v1/chat/completions",
                json=_body(model, index, stream=False),
            )
            latency = time.perf_counter() - started
            return {
                "index": index,
                "status_code": response.status_code,
                "latency_s": latency,
                "terminal": response.status_code == 200,
                "error": None if response.status_code == 200 else response.text[:200],
            }
        except Exception as error:
            return {
                "index": index,
                "status_code": None,
                "latency_s": time.perf_counter() - started,
                "terminal": False,
                "error": type(error).__name__,
            }


async def _one_stream(
    client: httpx.AsyncClient,
    model: str,
    index: int,
    target_s: float,
) -> dict:
    now = time.perf_counter()
    if target_s > now:
        await asyncio.sleep(target_s - now)
    sent = time.perf_counter()
    first_token: float | None = None
    terminal = False
    status_code: int | None = None
    error_text: str | None = None
    try:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json=_body(model, index, stream=True),
        ) as response:
            status_code = response.status_code
            if status_code != 200:
                error_text = (await response.aread()).decode(errors="replace")[:200]
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        terminal = True
                        break
                    payload = json.loads(data)
                    choices = payload.get("choices") or ()
                    if choices:
                        delta = choices[0].get("delta") or {}
                        if (
                            first_token is None
                            and (
                                delta.get("content") not in (None, "")
                                or delta.get("tool_calls")
                            )
                        ):
                            first_token = time.perf_counter()
    except Exception as error:
        error_text = type(error).__name__
    ended = time.perf_counter()
    return {
        "index": index,
        "target_offset_s": target_s,
        "send_offset_s": sent,
        "pacing_error_s": sent - target_s,
        "status_code": status_code,
        "terminal": terminal,
        "ttft_s": None if first_token is None else first_token - sent,
        "latency_s": ended - sent,
        "error": error_text,
    }


async def _open_loop(
    client: httpx.AsyncClient,
    model: str,
    *,
    start_index: int,
    count: int,
    rate: float,
    window_callback=None,
) -> tuple[list[dict], dict]:
    interval = 1.0 / rate
    start = time.perf_counter()
    tasks = [
        asyncio.create_task(
            _one_stream(
                client,
                model,
                start_index + index,
                start + index * interval,
            )
        )
        for index in range(count)
    ]
    planned_end = start + count * interval
    remaining = planned_end - time.perf_counter()
    if remaining > 0:
        await asyncio.sleep(remaining)
    window_snapshot = await window_callback() if window_callback is not None else {}
    return await asyncio.gather(*tasks), window_snapshot


async def _submit_batch(
    client: httpx.AsyncClient,
    model: str,
    *,
    start_index: int,
    count: int,
) -> str:
    content = "\n".join(
        json.dumps(
            {
                "custom_id": f"batch-{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": _body(model, start_index + index, stream=False),
            },
            separators=(",", ":"),
        )
        for index in range(count)
    ).encode()
    upload = await client.post(
        "/v1/files",
        files={"file": ("priority-overload.jsonl", content, "application/jsonl")},
        data={"purpose": "batch"},
    )
    upload.raise_for_status()
    created = await client.post(
        "/v1/batches",
        json={
            "input_file_id": upload.json()["id"],
            "endpoint": "/v1/chat/completions",
        },
    )
    created.raise_for_status()
    return created.json()["id"]


async def _batch_status(client: httpx.AsyncClient, batch_id: str) -> dict:
    response = await client.get(f"/v1/batches/{batch_id}")
    response.raise_for_status()
    return response.json()


async def _drain_batch(
    client: httpx.AsyncClient,
    batch_id: str,
    *,
    timeout_s: float,
) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = await _batch_status(client, batch_id)
        if status["status"] in {"completed", "failed", "cancelled"}:
            return status
        await asyncio.sleep(0.25)
    raise TimeoutError(f"batch {batch_id} did not drain in {timeout_s}s")


def metric_samples(text: str, name: str) -> dict[str, float]:
    samples = {}
    for line in text.splitlines():
        if line.startswith(f"{name}{{") or line.startswith(f"{name} "):
            series, value = line.rsplit(" ", 1)
            samples[series] = float(value)
    return samples


def metric_delta(before: str, after: str, name: str) -> dict[str, float]:
    old = metric_samples(before, name)
    new = metric_samples(after, name)
    return {series: value - old.get(series, 0.0) for series, value in new.items()}


def _class_count(delta: dict[str, float], request_class: str, source: str) -> float:
    return sum(
        value
        for series, value in delta.items()
        if f'request_class="{request_class}"' in series
        and f'source="{source}"' in series
    )


def _scheduler_event_count(
    delta: dict[str, float],
    request_class: str,
    event: str,
) -> float:
    return sum(
        value
        for series, value in delta.items()
        if f'request_class="{request_class}"' in series
        and f'event="{event}"' in series
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_environment(args) -> dict:
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    config_hashes = {}
    for name, raw_path in (
        ("gateway", args.gateway_config),
        ("replica", args.replica_config),
    ):
        if raw_path is not None:
            path = Path(raw_path).resolve()
            config_hashes[name] = {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
    return {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "tracked_dirty": bool(tracked_status),
        "tracked_status": tracked_status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "benchmark_sha256": _sha256_file(Path(__file__).resolve()),
        "config": config_hashes,
        "model_revision": args.model_revision,
        "replica_image_digest": args.replica_image_digest,
        "gateway_image_digest": args.gateway_image_digest,
        "gpus": subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines(),
    }


async def run(args) -> dict:
    timeout = httpx.Timeout(args.timeout_s)
    async with (
        httpx.AsyncClient(base_url=args.gateway_url.rstrip("/"), timeout=timeout) as gateway,
        httpx.AsyncClient(base_url=args.replica_url.rstrip("/"), timeout=timeout) as replica,
    ):
        for client in (gateway, replica):
            ready = await client.get("/readyz")
            ready.raise_for_status()
        gateway_backends = (await gateway.get("/backends")).json()
        replica_backends = (await replica.get("/backends")).json()

        warmup_semaphore = asyncio.Semaphore(args.calibration_concurrency)
        warmup = await asyncio.gather(
            *(
                _one_unary(gateway, args.model, -10_000 - index, warmup_semaphore)
                for index in range(args.warmup_requests)
            )
        )
        warmup_completed = sum(sample["terminal"] for sample in warmup)

        metrics_before_gateway = (await gateway.get("/metrics")).text
        metrics_before_replica = (await replica.get("/metrics")).text

        semaphore = asyncio.Semaphore(args.calibration_concurrency)
        calibration_started = time.perf_counter()
        calibration = await asyncio.gather(
            *(
                _one_unary(gateway, args.model, index, semaphore)
                for index in range(args.calibration_requests)
            )
        )
        calibration_wall = time.perf_counter() - calibration_started
        calibration_completed = sum(sample["terminal"] for sample in calibration)
        capacity = calibration_completed / calibration_wall
        interactive_rate = capacity * 0.5

        control, _ = await _open_loop(
            gateway,
            args.model,
            start_index=100_000,
            count=args.interactive_requests,
            rate=interactive_rate,
        )

        batch_count = 3 * args.interactive_requests
        batch_id = await _submit_batch(
            gateway,
            args.model,
            start_index=300_000,
            count=batch_count,
        )

        async def measurement_snapshot() -> dict:
            return {
                "batch": await _batch_status(gateway, batch_id),
                "gateway_metrics": (await gateway.get("/metrics")).text,
                "replica_metrics": (await replica.get("/metrics")).text,
            }

        mixed, batch_at_window = await _open_loop(
            gateway,
            args.model,
            start_index=200_000,
            count=args.interactive_requests,
            rate=interactive_rate,
            window_callback=measurement_snapshot,
        )
        batch_final = await _drain_batch(
            gateway,
            batch_id,
            timeout_s=args.batch_timeout_s,
        )
        metrics_after_gateway = (await gateway.get("/metrics")).text
        metrics_after_replica = (await replica.get("/metrics")).text

    def stream_summary(samples: list[dict]) -> dict:
        ttfts = [sample["ttft_s"] for sample in samples if sample["ttft_s"] is not None]
        pacing = [
            sample["pacing_error_s"] for sample in samples
            if sample["pacing_error_s"] is not None
        ]
        return {
            "planned": len(samples),
            "http_2xx": sum(sample["status_code"] == 200 for sample in samples),
            "terminal": sum(sample["terminal"] for sample in samples),
            "ttft_p50_s": percentile(ttfts, 0.50),
            "ttft_p95_s": percentile(ttfts, 0.95),
            "ttft_p99_s": percentile(ttfts, 0.99),
            "slo_attainment": sum(value <= args.ttft_slo_s for value in ttfts)
            / len(samples),
            "pacing_within_one_interval": sum(
                value <= 1.0 / interactive_rate for value in pacing
            )
            / len(samples),
            "samples": samples,
        }

    control_summary = stream_summary(control)
    mixed_summary = stream_summary(mixed)
    gateway_delta = metric_delta(
        metrics_before_gateway,
        metrics_after_gateway,
        "kairyu_priority_requests_total",
    )
    replica_delta = metric_delta(
        metrics_before_replica,
        metrics_after_replica,
        "kairyu_priority_requests_total",
    )
    scheduler_delta = metric_delta(
        metrics_before_replica,
        metrics_after_replica,
        "kairyu_scheduler_priority_events_total",
    )
    batch_window_gateway_delta = metric_delta(
        metrics_before_gateway,
        batch_at_window["gateway_metrics"],
        "kairyu_priority_requests_total",
    )
    batch_window_replica_delta = metric_delta(
        metrics_before_replica,
        batch_at_window["replica_metrics"],
        "kairyu_priority_requests_total",
    )
    scheduler_window_delta = metric_delta(
        metrics_before_replica,
        batch_at_window["replica_metrics"],
        "kairyu_scheduler_priority_events_total",
    )
    scheduler_queue_at_window = metric_samples(
        batch_at_window["replica_metrics"],
        "kairyu_scheduler_queue_depth",
    )
    scheduler_queue_high_watermark = metric_samples(
        metrics_after_replica,
        "kairyu_scheduler_queue_high_watermark",
    )
    batch_final_counts = batch_final["request_counts"]
    expected_interactive = args.calibration_requests + 2 * args.interactive_requests
    batch_completed_in_window = _scheduler_event_count(
        scheduler_window_delta,
        "batch",
        "complete",
    )
    source_environment = _source_environment(args)
    checks = {
        "warmup_completed": warmup_completed == args.warmup_requests,
        "calibration_completed": calibration_completed == args.calibration_requests,
        "exact_2x_planned_offered_load": (
            0.5 + batch_count / (args.interactive_requests / interactive_rate) / capacity
            == 2.0
        ),
        "interactive_http_2xx": mixed_summary["http_2xx"]
        == args.interactive_requests,
        "interactive_terminal": mixed_summary["terminal"]
        == args.interactive_requests,
        "interactive_p99_slo": mixed_summary["ttft_p99_s"] <= args.ttft_slo_s,
        "interactive_99pct_slo_attainment": mixed_summary["slo_attainment"] >= 0.99,
        "arrival_pacing": mixed_summary["pacing_within_one_interval"] >= 0.99,
        "batch_progressed_in_window": (
            batch_completed_in_window > 0
            and _scheduler_event_count(
                scheduler_window_delta, "batch", "admit"
            )
            > args.batch_concurrency
        ),
        "batch_backlog_at_window_end": (
            batch_at_window["batch"]["status"] not in {"completed", "failed", "cancelled"}
            and batch_completed_in_window < batch_count
        ),
        "batch_drained": (
            batch_final["status"] == "completed"
            and batch_final_counts["completed"] == batch_count
            and batch_final_counts["failed"] == 0
        ),
        "gateway_class_counts": (
            _class_count(gateway_delta, "interactive", "http")
            == expected_interactive
            and _class_count(gateway_delta, "batch", "batch") == batch_count
        ),
        "replica_class_counts": (
            _class_count(replica_delta, "interactive", "http")
            == expected_interactive
            and _class_count(replica_delta, "batch", "http") == batch_count
        ),
        "replica_scheduler_enqueue_counts": (
            _scheduler_event_count(
                scheduler_delta, "interactive", "enqueue"
            )
            == expected_interactive
            and _scheduler_event_count(scheduler_delta, "batch", "enqueue")
            == batch_count
        ),
        "replica_scheduler_completion_counts": (
            _scheduler_event_count(
                scheduler_delta, "interactive", "complete"
            )
            == expected_interactive
            and _scheduler_event_count(scheduler_delta, "batch", "complete")
            == batch_count
        ),
        "scheduler_queue_evidence_published": bool(
            scheduler_queue_at_window and scheduler_queue_high_watermark
        ),
        "clean_source": not source_environment["tracked_dirty"],
        "provenance_pinned": (
            bool(source_environment["model_revision"])
            and bool(source_environment["replica_image_digest"])
            and bool(source_environment["gateway_image_digest"])
            and set(source_environment["config"]) == {"gateway", "replica"}
            and bool(gateway_backends)
            and bool(replica_backends)
        ),
    }
    return {
        "schema_version": 1,
        "gate": "G5-F5a",
        "model": args.model,
        "gateway_url": args.gateway_url,
        "replica_url": args.replica_url,
        "ttft_slo_s": args.ttft_slo_s,
        "warmup": {
            "requests": args.warmup_requests,
            "completed": warmup_completed,
            "samples": warmup,
        },
        "calibration": {
            "requests": args.calibration_requests,
            "concurrency": args.calibration_concurrency,
            "wall_s": calibration_wall,
            "capacity_requests_per_s": capacity,
            "samples": calibration,
        },
        "load": {
            "interactive_rate_fraction": 0.5,
            "batch_rate_fraction": 1.5,
            "planned_offered_load_ratio": 2.0,
            "interactive_requests": args.interactive_requests,
            "batch_requests": batch_count,
        },
        "control": control_summary,
        "mixed": mixed_summary,
        "batch_at_measurement_end": batch_at_window["batch"],
        "batch_final": batch_final,
        "metrics": {
            "gateway_priority_delta": gateway_delta,
            "replica_priority_delta": replica_delta,
            "gateway_priority_delta_at_measurement_end": batch_window_gateway_delta,
            "replica_priority_delta_at_measurement_end": batch_window_replica_delta,
            "replica_scheduler_delta": scheduler_delta,
            "replica_scheduler_delta_at_measurement_end": scheduler_window_delta,
            "replica_scheduler_queue_at_measurement_end": scheduler_queue_at_window,
            "replica_scheduler_queue_high_watermark": scheduler_queue_high_watermark,
        },
        "topology": {
            "gateway_backends": gateway_backends,
            "replica_backends": replica_backends,
        },
        "environment": source_environment,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8002")
    parser.add_argument("--replica-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="qwen3-32b")
    parser.add_argument("--warmup-requests", type=int, default=32)
    parser.add_argument("--calibration-requests", type=int, default=32)
    parser.add_argument("--calibration-concurrency", type=int, default=8)
    parser.add_argument("--batch-concurrency", type=int, default=8)
    parser.add_argument("--interactive-requests", type=int, default=64)
    parser.add_argument("--ttft-slo-s", type=float, default=2.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--batch-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--gateway-config",
        type=Path,
        default=Path("examples/qwen3-32b-multi-gpu/auto-gateway.yaml"),
    )
    parser.add_argument(
        "--replica-config",
        type=Path,
        default=Path("examples/qwen3-32b-multi-gpu/kairyu.template.yaml"),
    )
    parser.add_argument("--model-revision")
    parser.add_argument("--replica-image-digest")
    parser.add_argument("--gateway-image-digest")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assert-gate", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(run(args))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    if args.assert_gate and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
