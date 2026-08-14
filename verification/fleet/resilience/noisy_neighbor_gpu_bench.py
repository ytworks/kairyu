#!/usr/bin/env python3
"""Qwen3-32B TP8 G5-F5b noisy-neighbor isolation gate (issue #191)."""

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
import yaml


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of no values")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def _prompt(phase: str, index: int) -> str:
    return (
        f"{phase}-salt-{index:06d} is an independent latency sample. "
        "Reply with a short factual sentence about reliable software testing."
    )


def _body(model: str, phase: str, index: int, *, stream: bool) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": _prompt(phase, index)}],
        "temperature": 0,
        "max_tokens": 8,
        "min_tokens": 8,
        "ignore_eos": True,
        "stream": stream,
    }


def _oversize_body(model: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "pre-dispatch quota probe"}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }


async def _one_unary(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    model: str,
    phase: str,
    index: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json=_body(model, phase, index, stream=False),
            )
            latency = time.perf_counter() - started
            error_code = None
            if response.status_code != 200:
                try:
                    error_code = response.json()["error"]["code"]
                except (KeyError, TypeError, ValueError):
                    error_code = "invalid_error_body"
            return {
                "index": index,
                "status_code": response.status_code,
                "latency_s": latency,
                "terminal": response.status_code == 200,
                "error_code": error_code,
            }
        except Exception as error:
            return {
                "index": index,
                "status_code": None,
                "latency_s": time.perf_counter() - started,
                "terminal": False,
                "error_code": type(error).__name__,
            }


async def _one_stream(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    model: str,
    phase: str,
    index: int,
    target_s: float,
    origin_s: float,
) -> dict:
    now = time.perf_counter()
    if target_s > now:
        await asyncio.sleep(target_s - now)
    sent = time.perf_counter()
    first_token: float | None = None
    terminal = False
    status_code: int | None = None
    error_code: str | None = None
    try:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=headers,
            json=_body(model, phase, index, stream=True),
        ) as response:
            status_code = response.status_code
            if status_code != 200:
                try:
                    payload = json.loads((await response.aread()).decode())
                    error_code = payload["error"]["code"]
                except (KeyError, TypeError, ValueError):
                    error_code = "invalid_error_body"
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
        error_code = type(error).__name__
    ended = time.perf_counter()
    return {
        "index": index,
        "target_offset_s": target_s - origin_s,
        "send_offset_s": sent - origin_s,
        "pacing_error_s": sent - target_s,
        "status_code": status_code,
        "terminal": terminal,
        "ttft_s": None if first_token is None else first_token - sent,
        "latency_s": ended - sent,
        "error_code": error_code,
    }


async def _paced_streams(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    model: str,
    phase: str,
    *,
    count: int,
    rate: float,
) -> list[dict]:
    origin = time.perf_counter()
    return await asyncio.gather(
        *(
            _one_stream(
                client,
                headers,
                model,
                phase,
                index,
                origin + index / rate,
                origin,
            )
            for index in range(count)
        )
    )


async def _two_tenant_window(
    client: httpx.AsyncClient,
    good_headers: dict[str, str],
    noisy_headers: dict[str, str],
    model: str,
    *,
    phase: str,
    good_count: int,
    good_rate: float,
    noisy_rate: float,
) -> tuple[list[dict], list[dict], float]:
    duration = good_count / good_rate
    noisy_count = math.ceil(duration * noisy_rate)
    origin = time.perf_counter()
    good_tasks = [
        asyncio.create_task(
            _one_stream(
                client,
                good_headers,
                model,
                f"{phase}-good",
                index,
                origin + index / good_rate,
                origin,
            )
        )
        for index in range(good_count)
    ]
    noisy_tasks = [
        asyncio.create_task(
            _one_stream(
                client,
                noisy_headers,
                model,
                f"{phase}-noisy",
                index,
                origin + index / noisy_rate,
                origin,
            )
        )
        for index in range(noisy_count)
    ]
    good, noisy = await asyncio.gather(
        asyncio.gather(*good_tasks),
        asyncio.gather(*noisy_tasks),
    )
    return list(good), list(noisy), duration


def stream_summary(samples: list[dict], *, rate: float) -> dict:
    ttfts = [sample["ttft_s"] for sample in samples if sample["ttft_s"] is not None]
    return {
        "requests": len(samples),
        "http_2xx": sum(sample["status_code"] == 200 for sample in samples),
        "terminal": sum(sample["terminal"] for sample in samples),
        "ttft_p50_s": percentile(ttfts, 0.50),
        "ttft_p95_s": percentile(ttfts, 0.95),
        "ttft_p99_s": percentile(ttfts, 0.99),
        "pacing_within_one_interval": (
            sum(
                sample["pacing_error_s"] <= 1.0 / rate
                for sample in samples
            )
            / len(samples)
        ),
        "samples": samples,
    }


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


def _labeled_count(delta: dict[str, float], **labels: str) -> float:
    return sum(
        value
        for series, value in delta.items()
        if all(f'{name}="{label}"' in series for name, label in labels.items())
    )


def _single_labeled_value(
    samples: dict[str, float],
    **labels: str,
) -> float | None:
    matched = [
        value
        for series, value in samples.items()
        if all(f'{name}="{label}"' in series for name, label in labels.items())
    ]
    return matched[0] if len(matched) == 1 else None


async def _text(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url)
    response.raise_for_status()
    return response.text


async def _json(client: httpx.AsyncClient, url: str) -> object:
    response = await client.get(url)
    response.raise_for_status()
    return response.json()


def _gateway_contract(args) -> dict:
    raw = yaml.safe_load(args.gateway_config.read_text())
    tenants = raw["tenants"]
    limits = tenants["limits"]
    key_tenants = tenants["key_tenants"]
    return {
        "key_tenants": key_tenants,
        "limits": limits,
        "checks": {
            "keys_map_to_distinct_expected_tenants": (
                key_tenants.get(args.good_key) == "good"
                and key_tenants.get(args.noisy_key) == "noisy"
            ),
            "same_interactive_priority": (
                limits["good"]["interactive_priority"]
                == limits["noisy"]["interactive_priority"]
            ),
            "noisy_rpm_matches_cli": (
                limits["noisy"]["requests_per_minute"]
                == args.noisy_quota_per_minute
            ),
            "explicit_burst_and_inflight": (
                limits["good"]["request_burst"] >= 1
                and limits["good"]["token_burst"] >= 1
                and limits["good"]["max_in_flight"] >= 1
                and limits["noisy"]["request_burst"] >= 1
                and limits["noisy"]["token_burst"] >= 1
                and limits["noisy"]["max_in_flight"] >= 1
            ),
        },
    }


def _contains_tensor_parallel_size(value: object, expected: int) -> bool:
    if isinstance(value, dict):
        if value.get("tensor_parallel_size") == expected:
            return True
        return any(
            _contains_tensor_parallel_size(item, expected)
            for item in value.values()
        )
    if isinstance(value, list):
        return any(
            _contains_tensor_parallel_size(item, expected)
            for item in value
        )
    return False


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_environment(args) -> dict:
    status = subprocess.run(
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
    return {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "tracked_dirty": bool(status),
        "tracked_status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "benchmark_sha256": _sha256_file(Path(__file__)),
        "config": {
            "gateway": {
                "path": str(args.gateway_config.resolve()),
                "sha256": _sha256_file(args.gateway_config),
            },
            "replica": {
                "path": str(args.replica_config.resolve()),
                "sha256": _sha256_file(args.replica_config),
            },
        },
        "model_revision": args.model_revision,
        "gateway_image_digest": args.gateway_image_digest,
        "replica_image_digest": args.replica_image_digest,
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
    good_headers = {"Authorization": f"Bearer {args.good_key}"}
    noisy_headers = {"Authorization": f"Bearer {args.noisy_key}"}
    contract = _gateway_contract(args)
    async with httpx.AsyncClient(
        base_url=args.gateway_url,
        timeout=timeout,
    ) as gateway, httpx.AsyncClient(timeout=timeout) as observer:
        ready = await _json(observer, f"{args.gateway_url}/readyz")
        replica_ready = await _json(observer, f"{args.replica_url}/readyz")
        warmup_semaphore = asyncio.Semaphore(args.calibration_concurrency)
        warmup = await asyncio.gather(
            *(
                _one_unary(
                    gateway,
                    good_headers,
                    args.model,
                    "warmup",
                    index,
                    warmup_semaphore,
                )
                for index in range(args.warmup_requests)
            )
        )
        calibration_started = time.perf_counter()
        semaphore = asyncio.Semaphore(args.calibration_concurrency)
        calibration = await asyncio.gather(
            *(
                _one_unary(
                    gateway,
                    good_headers,
                    args.model,
                    "calibration",
                    index,
                    semaphore,
                )
                for index in range(args.calibration_requests)
            )
        )
        calibration_wall = time.perf_counter() - calibration_started
        capacity = args.calibration_requests / calibration_wall
        good_rate = capacity * args.good_rate_fraction
        quota_per_second = args.noisy_quota_per_minute / 60.0
        control_noisy_rate = (
            quota_per_second * args.control_noisy_rate_fraction
        )
        arm_cooldown_s = (
            max(
                limits["request_burst"]
                / (limits["requests_per_minute"] / 60.0)
                for limits in contract["limits"].values()
            )
            + 0.25
        )
        await asyncio.sleep(arm_cooldown_s)
        good_only_control = await _paced_streams(
            gateway,
            good_headers,
            args.model,
            "good-only-control",
            count=args.good_requests,
            rate=good_rate,
        )
        oversize_limit = contract["limits"]["noisy"]["token_burst"] + 1
        oversize = await gateway.post(
            "/v1/chat/completions",
            headers=noisy_headers,
            json=_oversize_body(args.model, oversize_limit),
        )
        try:
            oversize_error_code = oversize.json()["error"]["code"]
        except (KeyError, TypeError, ValueError):
            oversize_error_code = "invalid_error_body"
        # Every arm starts with full request buckets and no in-flight work.
        # The controls include the noisy tenant just within its legitimate
        # quota; the treatment changes only offered excess (0.9x -> 10x).
        await asyncio.sleep(arm_cooldown_s)
        control_before, control_before_noisy, _ = await _two_tenant_window(
            gateway,
            good_headers,
            noisy_headers,
            args.model,
            phase="control-before",
            good_count=args.good_requests,
            good_rate=good_rate,
            noisy_rate=control_noisy_rate,
        )
        await asyncio.sleep(arm_cooldown_s)
        gateway_metrics_before = await _text(observer, f"{args.gateway_url}/metrics")
        replica_metrics_before = await _text(observer, f"{args.replica_url}/metrics")
        mixed_good, mixed_noisy, mixed_duration = await _two_tenant_window(
            gateway,
            good_headers,
            noisy_headers,
            args.model,
            phase="mixed",
            good_count=args.good_requests,
            good_rate=good_rate,
            noisy_rate=args.noisy_rate,
        )
        gateway_metrics_after = await _text(observer, f"{args.gateway_url}/metrics")
        replica_metrics_after = await _text(observer, f"{args.replica_url}/metrics")
        await asyncio.sleep(arm_cooldown_s)
        control_after, control_after_noisy, _ = await _two_tenant_window(
            gateway,
            good_headers,
            noisy_headers,
            args.model,
            phase="control-after",
            good_count=args.good_requests,
            good_rate=good_rate,
            noisy_rate=control_noisy_rate,
        )
        gateway_backends = await _json(observer, f"{args.gateway_url}/backends")
        replica_backends = await _json(observer, f"{args.replica_url}/backends")

    good_only_summary = stream_summary(good_only_control, rate=good_rate)
    control_before_summary = stream_summary(control_before, rate=good_rate)
    control_after_summary = stream_summary(control_after, rate=good_rate)
    control_before_noisy_summary = stream_summary(
        control_before_noisy,
        rate=control_noisy_rate,
    )
    control_after_noisy_summary = stream_summary(
        control_after_noisy,
        rate=control_noisy_rate,
    )
    mixed_summary = stream_summary(mixed_good, rate=good_rate)
    noisy_2xx = sum(sample["status_code"] == 200 for sample in mixed_noisy)
    noisy_429 = sum(sample["status_code"] == 429 for sample in mixed_noisy)
    bad_statuses_valid = all(
        sample["status_code"] == 200
        or (
            sample["status_code"] == 429
            and sample["error_code"] == "tenant_rate_limited"
        )
        for sample in mixed_noisy
    )
    noisy_pacing = sum(
        sample["pacing_error_s"] <= 1.0 / args.noisy_rate
        for sample in mixed_noisy
    ) / len(mixed_noisy)
    admission_delta = metric_delta(
        gateway_metrics_before,
        gateway_metrics_after,
        "kairyu_tenant_admission_total",
    )
    gateway_priority_delta = metric_delta(
        gateway_metrics_before,
        gateway_metrics_after,
        "kairyu_priority_requests_total",
    )
    gateway_usage_delta = metric_delta(
        gateway_metrics_before,
        gateway_metrics_after,
        "kairyu_usage_requests_total",
    )
    replica_priority_delta = metric_delta(
        replica_metrics_before,
        replica_metrics_after,
        "kairyu_priority_requests_total",
    )
    scheduler_delta = metric_delta(
        replica_metrics_before,
        replica_metrics_after,
        "kairyu_scheduler_priority_events_total",
    )
    expected_dispatches = args.good_requests + noisy_2xx
    noisy_send_offsets = [sample["send_offset_s"] for sample in mixed_noisy]
    noisy_send_span = max(noisy_send_offsets) - min(noisy_send_offsets)
    noisy_burst = contract["limits"]["noisy"]["request_burst"]
    noisy_admit_bound = math.floor(
        noisy_burst + quota_per_second * noisy_send_span + 1e-9
    )
    noisy_admit_floor = max(
        1,
        math.floor(0.5 * noisy_admit_bound),
    )
    baseline_noisy_accepted = (
        control_before_noisy_summary["http_2xx"]
        + control_after_noisy_summary["http_2xx"]
    ) / 2
    accepted_work_ratio = (
        noisy_2xx / baseline_noisy_accepted
        if baseline_noisy_accepted
        else None
    )
    source_environment = _source_environment(args)
    bracket_p99 = max(
        control_before_summary["ttft_p99_s"],
        control_after_summary["ttft_p99_s"],
    )
    noninferiority_margin = max(0.05, 0.10 * bracket_p99)
    final_in_flight = metric_samples(
        gateway_metrics_after,
        "kairyu_tenant_in_flight_requests",
    )
    final_reserved = metric_samples(
        gateway_metrics_after,
        "kairyu_tenant_reserved_tokens",
    )
    final_bound_violations = metric_samples(
        gateway_metrics_after,
        "kairyu_tenant_reservation_bound_violations_total",
    )
    good_in_flight = _single_labeled_value(
        final_in_flight,
        tenant="good",
        source="http",
    )
    noisy_in_flight = _single_labeled_value(
        final_in_flight,
        tenant="noisy",
        source="http",
    )
    good_reserved = _single_labeled_value(final_reserved, tenant="good")
    noisy_reserved = _single_labeled_value(final_reserved, tenant="noisy")
    good_bound_violations = _single_labeled_value(
        final_bound_violations,
        tenant="good",
    )
    noisy_bound_violations = _single_labeled_value(
        final_bound_violations,
        tenant="noisy",
    )
    checks = {
        "gateway_ready": ready.get("status") == "ready",
        "replica_ready": replica_ready.get("status") == "ready",
        "warmup_completed": sum(sample["terminal"] for sample in warmup)
        == args.warmup_requests,
        "calibration_completed": sum(sample["terminal"] for sample in calibration)
        == args.calibration_requests,
        "good_only_control_http_2xx": (
            good_only_summary["http_2xx"] == args.good_requests
        ),
        "good_only_control_terminal": (
            good_only_summary["terminal"] == args.good_requests
        ),
        "good_only_control_arrival_pacing": (
            good_only_summary["pacing_within_one_interval"] == 1.0
        ),
        "control_before_http_2xx": (
            control_before_summary["http_2xx"] == args.good_requests
        ),
        "control_before_terminal": (
            control_before_summary["terminal"] == args.good_requests
        ),
        "control_before_arrival_pacing": (
            control_before_summary["pacing_within_one_interval"] == 1.0
        ),
        "control_before_noisy_http_2xx": (
            control_before_noisy_summary["http_2xx"]
            == control_before_noisy_summary["requests"]
        ),
        "control_before_noisy_terminal": (
            control_before_noisy_summary["terminal"]
            == control_before_noisy_summary["requests"]
        ),
        "control_before_noisy_arrival_pacing": (
            control_before_noisy_summary["pacing_within_one_interval"] == 1.0
        ),
        "control_after_http_2xx": (
            control_after_summary["http_2xx"] == args.good_requests
        ),
        "control_after_terminal": (
            control_after_summary["terminal"] == args.good_requests
        ),
        "control_after_arrival_pacing": (
            control_after_summary["pacing_within_one_interval"] == 1.0
        ),
        "control_after_noisy_http_2xx": (
            control_after_noisy_summary["http_2xx"]
            == control_after_noisy_summary["requests"]
        ),
        "control_after_noisy_terminal": (
            control_after_noisy_summary["terminal"]
            == control_after_noisy_summary["requests"]
        ),
        "control_after_noisy_arrival_pacing": (
            control_after_noisy_summary["pacing_within_one_interval"] == 1.0
        ),
        "control_noisy_offer_is_compliant": (
            0.9 <= args.control_noisy_rate_fraction <= 1.0
        ),
        "mixed_accepted_work_matches_compliant_control": (
            accepted_work_ratio is not None
            and 0.9 <= accepted_work_ratio <= 1.1
        ),
        "oversize_token_work_rejected_predispatch": (
            oversize.status_code == 429
            and oversize_error_code == "tenant_rate_limited"
        ),
        "noisy_offers_exactly_10x_quota": math.isclose(
            args.noisy_rate / quota_per_second,
            10.0,
        ),
        "good_http_2xx": mixed_summary["http_2xx"] == args.good_requests,
        "good_terminal": mixed_summary["terminal"] == args.good_requests,
        "good_arrival_pacing": mixed_summary["pacing_within_one_interval"] == 1.0,
        "good_fixed_ttft_slo": mixed_summary["ttft_p99_s"] <= args.ttft_slo_s,
        "good_p99_noninferior_to_compliant_control": (
            mixed_summary["ttft_p99_s"]
            <= bracket_p99 + noninferiority_margin
        ),
        "noisy_status_accounting": (
            noisy_2xx + noisy_429 == len(mixed_noisy) and bad_statuses_valid
        ),
        "noisy_accepted_terminal": (
            sum(sample["terminal"] for sample in mixed_noisy) == noisy_2xx
        ),
        "noisy_arrival_pacing": noisy_pacing == 1.0,
        "noisy_quota_bound": (
            noisy_admit_floor <= noisy_2xx <= noisy_admit_bound
        ),
        "noisy_majority_rejected": noisy_429 / len(mixed_noisy) >= 0.80,
        "tenant_admission_reconciles": (
            _labeled_count(
                admission_delta,
                tenant="good",
                source="http",
                decision="admitted",
            )
            == args.good_requests
            and _labeled_count(
                admission_delta,
                tenant="noisy",
                source="http",
                decision="admitted",
            )
            == noisy_2xx
            and _labeled_count(
                admission_delta,
                tenant="noisy",
                source="http",
                decision="rejected",
            )
            == noisy_429
        ),
        "gateway_dispatches_match_admission": (
            _labeled_count(
                gateway_priority_delta,
                request_class="interactive",
                source="http",
            )
            == expected_dispatches
        ),
        "replica_dispatches_match_admission": (
            _labeled_count(
                replica_priority_delta,
                request_class="interactive",
                source="http",
            )
            == expected_dispatches
        ),
        "scheduler_enqueue_matches_admission": (
            _labeled_count(
                scheduler_delta,
                request_class="interactive",
                event="enqueue",
            )
            == expected_dispatches
        ),
        "scheduler_completion_matches_admission": (
            _labeled_count(
                scheduler_delta,
                request_class="interactive",
                event="complete",
            )
            == expected_dispatches
        ),
        "usage_matches_admission": (
            _labeled_count(gateway_usage_delta, tenant="good")
            == args.good_requests
            and _labeled_count(gateway_usage_delta, tenant="noisy") == noisy_2xx
        ),
        "tenant_in_flight_drains": (
            good_in_flight is not None
            and good_in_flight == 0
            and noisy_in_flight is not None
            and noisy_in_flight == 0
        ),
        "tenant_reservations_drain": (
            good_reserved is not None
            and good_reserved == 0
            and noisy_reserved is not None
            and noisy_reserved == 0
        ),
        "zero_reservation_bound_violations": (
            good_bound_violations is not None
            and good_bound_violations == 0
            and noisy_bound_violations is not None
            and noisy_bound_violations == 0
        ),
        "gateway_config_matches_gate": all(contract["checks"].values()),
        "exactly_eight_gpus": len(source_environment["gpus"]) == 8,
        "replica_reports_tp8": _contains_tensor_parallel_size(
            replica_backends,
            8,
        ),
        "clean_source": not source_environment["tracked_dirty"],
        "provenance_pinned": (
            bool(source_environment["model_revision"])
            and bool(source_environment["replica_image_digest"])
            and bool(source_environment["gateway_image_digest"])
            and bool(gateway_backends)
            and bool(replica_backends)
        ),
    }
    return {
        "schema_version": 1,
        "gate": "G5-F5b",
        "model": args.model,
        "gateway_url": args.gateway_url,
        "replica_url": args.replica_url,
        "tenant_contract": contract,
        "warmup": {"samples": warmup},
        "calibration": {
            "samples": calibration,
            "wall_s": calibration_wall,
            "capacity_requests_per_s": capacity,
        },
        "load": {
            "good_rate_fraction": args.good_rate_fraction,
            "good_rate_requests_per_s": good_rate,
            "control_noisy_rate_fraction": args.control_noisy_rate_fraction,
            "control_noisy_rate_requests_per_s": control_noisy_rate,
            "noisy_rate_requests_per_s": args.noisy_rate,
            "noisy_quota_requests_per_s": quota_per_second,
            "noisy_offered_to_quota_ratio": args.noisy_rate / quota_per_second,
            "mixed_duration_s": mixed_duration,
            "arm_cooldown_s": arm_cooldown_s,
            "mixed_to_control_noisy_accepted_work_ratio": accepted_work_ratio,
        },
        "good_only_control": good_only_summary,
        "control_before": control_before_summary,
        "control_before_noisy": control_before_noisy_summary,
        "control_after": control_after_summary,
        "control_after_noisy": control_after_noisy_summary,
        "mixed_good": mixed_summary,
        "mixed_noisy": {
            "requests": len(mixed_noisy),
            "http_2xx": noisy_2xx,
            "http_429": noisy_429,
            "pacing_within_one_interval": noisy_pacing,
            "actual_send_span_s": noisy_send_span,
            "admit_lower_bound": noisy_admit_floor,
            "admit_upper_bound": noisy_admit_bound,
            "samples": mixed_noisy,
        },
        "metrics": {
            "gateway_admission_delta": admission_delta,
            "gateway_priority_delta": gateway_priority_delta,
            "gateway_usage_delta": gateway_usage_delta,
            "gateway_final_in_flight": final_in_flight,
            "gateway_final_reserved_tokens": final_reserved,
            "gateway_final_bound_violations": final_bound_violations,
            "replica_priority_delta": replica_priority_delta,
            "replica_scheduler_delta": scheduler_delta,
        },
        "topology": {
            "gateway_backends": gateway_backends,
            "replica_backends": replica_backends,
        },
        "environment": source_environment,
        "noninferiority_margin_s": noninferiority_margin,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8003")
    parser.add_argument("--replica-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="qwen3-32b")
    parser.add_argument("--good-key", default="good-key")
    parser.add_argument("--noisy-key", default="noisy-key")
    parser.add_argument("--warmup-requests", type=int, default=32)
    parser.add_argument("--calibration-requests", type=int, default=32)
    parser.add_argument("--calibration-concurrency", type=int, default=8)
    parser.add_argument("--good-requests", type=int, default=256)
    parser.add_argument("--good-rate-fraction", type=float, default=0.5)
    parser.add_argument(
        "--control-noisy-rate-fraction",
        type=float,
        default=0.9,
        help=(
            "compliant-control noisy offer as a fraction of quota; 0.9 "
            "avoids token-boundary jitter while matching admitted treatment work"
        ),
    )
    parser.add_argument("--noisy-rate", type=float, default=10.0)
    parser.add_argument("--noisy-quota-per-minute", type=int, default=60)
    parser.add_argument("--ttft-slo-s", type=float, default=2.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--gateway-config",
        type=Path,
        default=Path(
            "deploy/verification/qwen3-32b-multi-gpu/noisy-neighbor-gateway.yaml"
        ),
    )
    parser.add_argument(
        "--replica-config",
        type=Path,
        default=Path("deploy/verification/qwen3-32b-multi-gpu/kairyu.template.yaml"),
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
