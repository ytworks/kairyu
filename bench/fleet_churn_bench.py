#!/usr/bin/env python3
"""Scheduled kind F1a replica-churn gate (issue #175).

The formal profile is intentionally frozen: 200 StatefulSet ordinals are
permuted by one fixed seed into ten disjoint batches of twenty, one batch is
deleted at each absolute 60-second boundary, and valid OpenAI requests continue
open-loop without retries throughout the ten-minute measurement.

The gateway contract used by this black-box gate is explicit.  It assigns and
echoes ``X-Request-ID``; the same id, selected replica UID, and
request-receipt-to-selection latency are written to its placement JSONL.  The
mock's completion text identifies the serving pod UID, so all three records can
be joined without adding benchmark-only HTTP headers.  Kubernetes assumptions
are likewise explicit:

* mock replicas are ``<statefulset>-<ordinal>`` pods;
* EndpointSlice ``targetRef.uid`` is the upstream pod identity;
* terminating/not-ready endpoints are absent from gateway discovery;
* a recreated ordinal has a new pod UID and normally a new pod IP.

Raw requests, placements, churn joins, EndpointSlice snapshots, pod snapshots,
and kubelet resource summaries are stored as JSONL sidecars.
``--verify-artifact`` hashes and replays those sidecars instead of trusting the
manifest's pass bit.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import hashlib
import json
import math
import random
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import httpx

SCHEMA_VERSION = 1
GATE = "G5-F1a"
_COMMAND_TERMINATE_GRACE_SECONDS = 2.0


@dataclass(frozen=True)
class GateConfig:
    profile: str
    replica_count: int
    churn_batch_size: int
    churn_epochs: int
    epoch_seconds: float
    seed: int
    request_rate: float
    warmup_seconds: float
    cooldown_seconds: float
    max_concurrency: int
    request_timeout_seconds: float
    recovery_timeout_seconds: float
    churn_window_seconds: float
    evidence_interval_seconds: float
    resource_interval_seconds: float
    placement_p99_limit_ms: float
    schedule_lateness_limit_ms: float
    endpoint_withdrawal_limit_seconds: float
    pacing_success_fraction: float

    @property
    def measurement_seconds(self) -> float:
        return self.churn_epochs * self.epoch_seconds

    @property
    def traffic_seconds(self) -> float:
        return (
            self.warmup_seconds
            + self.measurement_seconds
            + self.cooldown_seconds
        )

    @property
    def expected_requests(self) -> int:
        return math.floor(self.traffic_seconds * self.request_rate)

    def validate(self) -> None:
        if self.profile not in {"formal", "smoke"}:
            raise ValueError("profile must be 'formal' or 'smoke'")
        integer_values = (
            self.replica_count,
            self.churn_batch_size,
            self.churn_epochs,
            self.seed,
            self.max_concurrency,
        )
        if any(type(value) is not int for value in integer_values):
            raise TypeError("integer gate fields must be integers")
        if self.replica_count < 1 or self.churn_batch_size < 1:
            raise ValueError("replica_count and churn_batch_size must be positive")
        if self.churn_epochs < 1 or self.max_concurrency < 1:
            raise ValueError("churn_epochs and max_concurrency must be positive")
        if self.replica_count != self.churn_batch_size * self.churn_epochs:
            raise ValueError(
                "replica_count must equal churn_batch_size * churn_epochs "
                "so every ordinal is replaced exactly once"
            )
        positive = (
            self.epoch_seconds,
            self.request_rate,
            self.request_timeout_seconds,
            self.recovery_timeout_seconds,
            self.churn_window_seconds,
            self.evidence_interval_seconds,
            self.resource_interval_seconds,
            self.placement_p99_limit_ms,
            self.schedule_lateness_limit_ms,
            self.endpoint_withdrawal_limit_seconds,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("gate duration/rate/limit fields must be positive")
        if self.warmup_seconds < 0 or self.cooldown_seconds < 0:
            raise ValueError("warmup/cooldown must be non-negative")
        if not 0 < self.pacing_success_fraction <= 1:
            raise ValueError("pacing_success_fraction must be in (0, 1]")
        if self.churn_window_seconds > self.epoch_seconds:
            raise ValueError("churn_window_seconds must not exceed epoch_seconds")


FORMAL_CONFIG = GateConfig(
    profile="formal",
    replica_count=200,
    churn_batch_size=20,
    churn_epochs=10,
    epoch_seconds=60.0,
    seed=175,
    request_rate=50.0,
    warmup_seconds=30.0,
    cooldown_seconds=30.0,
    max_concurrency=256,
    request_timeout_seconds=10.0,
    recovery_timeout_seconds=60.0,
    churn_window_seconds=10.0,
    evidence_interval_seconds=1.0,
    resource_interval_seconds=5.0,
    placement_p99_limit_ms=10.0,
    schedule_lateness_limit_ms=1_000.0,
    endpoint_withdrawal_limit_seconds=5.0,
    pacing_success_fraction=0.99,
)

SMOKE_CONFIG = GateConfig(
    profile="smoke",
    replica_count=4,
    churn_batch_size=2,
    churn_epochs=2,
    epoch_seconds=20.0,
    seed=175,
    request_rate=10.0,
    warmup_seconds=1.0,
    cooldown_seconds=1.0,
    max_concurrency=32,
    request_timeout_seconds=5.0,
    recovery_timeout_seconds=20.0,
    churn_window_seconds=5.0,
    evidence_interval_seconds=0.5,
    resource_interval_seconds=1.0,
    placement_p99_limit_ms=10.0,
    schedule_lateness_limit_ms=1_000.0,
    endpoint_withdrawal_limit_seconds=5.0,
    pacing_success_fraction=0.95,
)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of no values")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def churn_plan(config: GateConfig) -> list[dict[str, Any]]:
    """Return the frozen disjoint schedule without consulting wall time."""

    config.validate()
    ordinals = list(range(config.replica_count))
    random.Random(config.seed).shuffle(ordinals)
    return [
        {
            "epoch": epoch,
            "deadline_offset_s": epoch * config.epoch_seconds,
            "ordinals": ordinals[
                epoch * config.churn_batch_size : (epoch + 1)
                * config.churn_batch_size
            ],
        }
        for epoch in range(config.churn_epochs)
    ]


def resolve_config(args: argparse.Namespace) -> GateConfig:
    base = FORMAL_CONFIG if args.profile == "formal" else SMOKE_CONFIG
    overrides = {}
    for field in fields(GateConfig):
        if field.name == "profile":
            continue
        value = getattr(args, field.name, None)
        if value is not None:
            if args.profile == "formal" and value != getattr(base, field.name):
                raise ValueError(
                    f"formal setting {field.name} is frozen at "
                    f"{getattr(base, field.name)!r}, got {value!r}"
                )
            overrides[field.name] = value
    config = replace(base, **overrides)
    config.validate()
    return config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*command: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *command],
        check=True,
        capture_output=True,
        text=not binary,
    )
    return completed.stdout


def source_environment(args: argparse.Namespace) -> dict[str, Any]:
    status = _git_output("status", "--porcelain", "--untracked-files=no")
    diff = _git_output("diff", "--binary", "HEAD", binary=True)
    repository = Path(
        str(_git_output("rev-parse", "--show-toplevel")).strip()
    ).resolve()
    benchmark_path = Path(__file__).resolve()
    driver_path = Path(args.driver_path).resolve()

    def repository_path(path: Path) -> str:
        try:
            return str(path.relative_to(repository))
        except ValueError:
            return str(path)

    gate_input_scopes = [
        ".dockerignore",
        "Dockerfile",
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "kairyu",
        repository_path(benchmark_path),
        repository_path(driver_path),
        "deploy/kind/f1a",
    ]
    untracked = _git_output(
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        *gate_input_scopes,
    )
    assert isinstance(status, str)
    assert isinstance(diff, bytes)
    assert isinstance(untracked, str)
    untracked_gate_inputs = sorted(
        line for line in untracked.splitlines() if line
    )
    git_commit = str(_git_output("rev-parse", "HEAD")).strip()
    frozen_inputs = []
    for raw in args.frozen_input:
        path = Path(raw).resolve()
        frozen_inputs.append(
            {"path": str(path), "sha256": _sha256_file(path)}
        )
    return {
        "git_commit": git_commit,
        "expected_git_commit": args.expected_git_commit,
        "git_commit_matches_expected": (
            isinstance(args.expected_git_commit, str)
            and git_commit == args.expected_git_commit
        ),
        "tracked_dirty": bool(status),
        "tracked_status": status.splitlines(),
        "untracked_gate_input_dirty": bool(untracked_gate_inputs),
        "untracked_gate_inputs": untracked_gate_inputs,
        "gate_input_scopes": gate_input_scopes,
        "source_dirty": bool(status) or bool(untracked_gate_inputs),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "benchmark_sha256": _sha256_file(Path(__file__).resolve()),
        "driver_sha256": _sha256_file(Path(args.driver_path).resolve()),
        "frozen_inputs": frozen_inputs,
        "gateway_image_digest": args.gateway_image_digest,
        "mock_image_digest": args.mock_image_digest,
        "kind_node_image": args.kind_node_image,
        "tool_versions": {
            name: value
            for name, value in (
                ("kind", args.kind_version),
                ("kubectl", args.kubectl_version),
                ("docker", args.docker_version),
            )
            if value
        },
    }


async def _run_command(
    *command: str,
    check: bool = True,
) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await process.communicate()
    except asyncio.CancelledError:
        if process.returncode is None:
            process.terminate()
        reap = asyncio.create_task(process.communicate())
        try:
            await asyncio.wait_for(
                asyncio.shield(reap),
                timeout=_COMMAND_TERMINATE_GRACE_SECONDS,
            )
        except TimeoutError:
            if process.returncode is None:
                process.kill()
            await reap
        raise
    output = stdout.decode(errors="replace")
    error = stderr.decode(errors="replace")
    if check and process.returncode:
        raise RuntimeError(
            f"command {command!r} exited {process.returncode}: {error[:500]}"
        )
    return process.returncode or 0, output, error


async def _kubectl_json(
    kubectl: str,
    namespace: str,
    *arguments: str,
) -> dict[str, Any]:
    _, stdout, _ = await _run_command(
        kubectl,
        "-n",
        namespace,
        *arguments,
        "-o",
        "json",
    )
    payload = json.loads(stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("kubectl JSON output must be an object")
    return payload


def _ready(item: dict[str, Any]) -> bool:
    return any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in item.get("status", {}).get("conditions", ())
    )


def _pod_identity(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata", {})
    status = item.get("status", {})
    spec_containers = {
        value.get("name"): value.get("image")
        for value in item.get("spec", {}).get("containers", ())
        if isinstance(value, dict)
    }
    status_containers = {
        value.get("name"): value
        for value in status.get("containerStatuses", ())
        if isinstance(value, dict)
    }
    return {
        "name": metadata.get("name"),
        "uid": metadata.get("uid"),
        "resource_version": metadata.get("resourceVersion"),
        "ip": status.get("podIP"),
        "node": item.get("spec", {}).get("nodeName"),
        "ready": _ready(item),
        "phase": status.get("phase"),
        "restart_count": sum(
            value.get("restartCount", 0)
            for value in status.get("containerStatuses", ())
        ),
        "containers": [
            {
                "name": name,
                "spec_image": spec_containers.get(name),
                "runtime_image": value.get("image"),
                "image_id": value.get("imageID"),
            }
            for name, value in sorted(status_containers.items())
            if isinstance(name, str)
        ],
    }


def _endpoint_uids(snapshot: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for item in snapshot.get("items", ()):
        for endpoint in item.get("endpoints", ()):
            conditions = endpoint.get("conditions") or {}
            if (
                conditions.get("ready") is not True
                or conditions.get("terminating") is True
            ):
                continue
            target = endpoint.get("targetRef") or {}
            uid = target.get("uid")
            if isinstance(uid, str):
                values.add(uid)
    return values


class JsonlSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = path.open("w", encoding="utf-8")

    def write(self, value: dict[str, Any]) -> None:
        self._handle.write(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


@dataclass(frozen=True)
class SurfaceContract:
    request_id_header: str


_MOCK_IDENTITY = re.compile(
    r"\bkairyu-f1a-mock pod_name=(?P<pod_name>\S+) "
    r"pod_uid=(?P<pod_uid>\S+)"
)


async def _one_request(
    client: httpx.AsyncClient,
    *,
    sequence: int,
    target_ns: int,
    measurement_origin_ns: int,
    model: str,
    contract: SurfaceContract,
) -> dict[str, Any]:
    sent_ns = time.monotonic_ns()
    request_id = f"f1a-{sequence:08d}-{uuid.uuid4().hex[:8]}"
    headers = {
        "X-Request-ID": request_id,
        "X-Session-ID": f"f1a-session-{sequence % 4096:04d}",
    }
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "request_id": request_id,
        "scheduled_ns": target_ns,
        "send_ns": sent_ns,
        "measurement_offset_s": (
            sent_ns - measurement_origin_ns
        )
        / 1_000_000_000,
        "pacing_error_ms": (sent_ns - target_ns) / 1_000_000,
        "not_sent": False,
        "status_code": None,
        "transport_error": None,
        "response_valid": False,
        "request_id_echo": None,
        "pod_name": None,
        "pod_uid": None,
    }
    try:
        response = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"f1a request {sequence}",
                    }
                ],
                "max_tokens": 1,
                "temperature": 0,
            },
        )
        record["status_code"] = response.status_code
        record["request_id_echo"] = response.headers.get(
            contract.request_id_header
        )
        try:
            payload = response.json()
            choices = payload.get("choices")
            content = (
                (choices[0].get("message") or {}).get("content")
                if isinstance(choices, list)
                and choices
                and isinstance(choices[0], dict)
                else None
            )
            identity = (
                _MOCK_IDENTITY.fullmatch(content)
                if isinstance(content, str)
                else None
            )
            if identity is not None:
                record.update(identity.groupdict())
            record["response_valid"] = (
                response.status_code == 200
                and isinstance(choices, list)
                and bool(choices)
            )
        except (TypeError, ValueError):
            record["response_valid"] = False
    except Exception as error:
        record["transport_error"] = type(error).__name__
    record["end_ns"] = time.monotonic_ns()
    record["latency_ms"] = (record["end_ns"] - sent_ns) / 1_000_000
    return record


async def _traffic_loop(
    args: argparse.Namespace,
    config: GateConfig,
    request_sink: JsonlSink,
    measurement_origin_ns: int,
) -> list[dict[str, Any]]:
    timeout = httpx.Timeout(config.request_timeout_seconds)
    limits = httpx.Limits(
        max_connections=config.max_concurrency,
        max_keepalive_connections=config.max_concurrency,
    )
    contract = SurfaceContract(
        request_id_header=args.request_id_header,
    )
    traffic_origin_ns = measurement_origin_ns - int(
        config.warmup_seconds * 1_000_000_000
    )
    interval_ns = int(1_000_000_000 / config.request_rate)
    pending: set[asyncio.Task] = set()
    records: list[dict[str, Any]] = []

    def complete(task: asyncio.Task) -> None:
        pending.discard(task)
        if task.cancelled():
            return
        record = task.result()
        records.append(record)
        request_sink.write(record)

    try:
        async with httpx.AsyncClient(
            base_url=args.gateway_url.rstrip("/"),
            timeout=timeout,
            limits=limits,
        ) as client:
            for sequence in range(config.expected_requests):
                target_ns = traffic_origin_ns + sequence * interval_ns
                remaining = (
                    target_ns - time.monotonic_ns()
                ) / 1_000_000_000
                if remaining > 0:
                    await asyncio.sleep(remaining)
                if len(pending) >= config.max_concurrency:
                    now_ns = time.monotonic_ns()
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "sequence": sequence,
                        "request_id": f"f1a-{sequence:08d}-not-sent",
                        "scheduled_ns": target_ns,
                        "send_ns": None,
                        "end_ns": now_ns,
                        "measurement_offset_s": (
                            now_ns - measurement_origin_ns
                        )
                        / 1_000_000_000,
                        "pacing_error_ms": (
                            now_ns - target_ns
                        )
                        / 1_000_000,
                        "not_sent": True,
                        "status_code": None,
                        "transport_error": "max_concurrency",
                        "response_valid": False,
                        "request_id_echo": None,
                        "pod_name": None,
                        "pod_uid": None,
                        "latency_ms": None,
                    }
                    records.append(record)
                    request_sink.write(record)
                    continue
                task = asyncio.create_task(
                    _one_request(
                        client,
                        sequence=sequence,
                        target_ns=target_ns,
                        measurement_origin_ns=measurement_origin_ns,
                        model=args.model,
                        contract=contract,
                    )
                )
                pending.add(task)
                task.add_done_callback(complete)
            if pending:
                await asyncio.gather(*tuple(pending))
    finally:
        remaining_tasks = tuple(pending)
        for task in remaining_tasks:
            if not task.done():
                task.cancel()
        if remaining_tasks:
            await asyncio.gather(*remaining_tasks, return_exceptions=True)
    return sorted(records, key=lambda item: item["sequence"])


async def _fetch_gateway_placements(
    args: argparse.Namespace,
    *,
    expected_request_ids: set[str],
    initial_uids: set[str],
    final_uids: set[str],
    sink: JsonlSink,
) -> list[dict[str, Any]]:
    """Read the gateway-owned audit file after its async writer has drained."""

    deadline = time.monotonic() + args.placement_log_timeout_seconds
    records: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        try:
            _, stdout, _ = await _run_command(
                args.kubectl,
                "-n",
                args.namespace,
                "exec",
                "deployment/f1a-gateway",
                "--",
                "python",
                "-c",
                (
                    "from pathlib import Path; import sys; "
                    "sys.stdout.write(Path(sys.argv[1]).read_text())"
                ),
                args.gateway_placement_log,
            )
            candidate = [
                json.loads(line)
                for line in stdout.splitlines()
                if line.strip()
            ]
            if all(isinstance(row, dict) for row in candidate):
                records = candidate
            observed_ids = {
                row.get("request_id")
                for row in records
                if row.get("kind") == "replica"
                and isinstance(row.get("request_id"), str)
            }
            memberships = [
                row for row in records if row.get("kind") == "membership"
            ]
            initial_seen = any(
                set(row.get("replica_ids", ())) == initial_uids
                and set(row.get("healthy_ids", ())) == initial_uids
                and set(row.get("eligible_ids", ())) == initial_uids
                for row in memberships
            )
            final_seen = bool(memberships) and all(
                set(memberships[-1].get(key, ())) == final_uids
                for key in ("replica_ids", "healthy_ids", "eligible_ids")
            )
            if (
                expected_request_ids <= observed_ids
                and initial_seen
                and final_seen
            ):
                break
        except (json.JSONDecodeError, RuntimeError):
            pass
        await asyncio.sleep(0.1)
    for record in records:
        sink.write(record)
    return records


async def _snapshot_loop(
    *,
    kind: str,
    interval_s: float,
    stop: asyncio.Event,
    sink: JsonlSink,
    fetch,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    origin_ns = time.monotonic_ns()
    index = 0
    while not stop.is_set():
        target_ns = origin_ns + int(index * interval_s * 1_000_000_000)
        remaining = (target_ns - time.monotonic_ns()) / 1_000_000_000
        if remaining > 0:
            try:
                await asyncio.wait_for(stop.wait(), timeout=remaining)
                break
            except TimeoutError:
                pass
        fetch_started_ns = time.monotonic_ns()
        try:
            payload = await fetch()
            record = {
                "schema_version": SCHEMA_VERSION,
                "kind": kind,
                "fetch_started_ns": fetch_started_ns,
                "observed_ns": time.monotonic_ns(),
                "error": None,
                "payload": payload,
            }
        except Exception as error:
            record = {
                "schema_version": SCHEMA_VERSION,
                "kind": kind,
                "fetch_started_ns": fetch_started_ns,
                "observed_ns": time.monotonic_ns(),
                "error": type(error).__name__,
                "payload": None,
            }
        records.append(record)
        sink.write(record)
        index += 1
    return records


async def _initial_inventory(args: argparse.Namespace, config: GateConfig) -> list[dict]:
    names = [
        f"{args.statefulset}-{ordinal}"
        for ordinal in range(config.replica_count)
    ]
    snapshot = await _kubectl_json(
        args.kubectl,
        args.namespace,
        "get",
        "pods",
        *names,
    )
    inventory = [_pod_identity(item) for item in snapshot.get("items", ())]
    if len(inventory) != config.replica_count:
        raise RuntimeError(
            f"expected {config.replica_count} mock pods, found {len(inventory)}"
        )
    if not all(item["ready"] and item["uid"] and item["ip"] for item in inventory):
        raise RuntimeError("all initial mock pods must be Ready with UID and IP")
    return inventory


async def _recover_epoch(
    args: argparse.Namespace,
    config: GateConfig,
    *,
    plan_entry: dict[str, Any],
    old: list[dict[str, Any]],
    scheduled_ns: int,
    api_started_ns: int,
    api_completed_ns: int,
    delete_returncode: int,
    delete_stderr: str,
    endpoint_sink: JsonlSink,
    sink: JsonlSink,
) -> dict[str, Any]:
    deadline = time.monotonic() + config.recovery_timeout_seconds
    names = [item["name"] for item in old]
    old_uids = {str(item["uid"]) for item in old}
    new: list[dict[str, Any]] = []
    endpoint_uids: set[str] = set()
    old_withdrawn_ns: int | None = None
    new_ready_ns: int | None = None
    error: str | None = None
    while time.monotonic() < deadline:
        try:
            endpoints = await _kubectl_json(
                args.kubectl,
                args.namespace,
                "get",
                "endpointslices",
                "-l",
                f"kubernetes.io/service-name={args.endpoint_service}",
            )
        except Exception as endpoint_error:
            endpoint_sink.write(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "endpointslices",
                    "capture": "churn_recovery",
                    "epoch": plan_entry["epoch"],
                    "observed_ns": time.monotonic_ns(),
                    "error": type(endpoint_error).__name__,
                    "payload": None,
                }
            )
        else:
            endpoint_uids = _endpoint_uids(endpoints)
            observed_ns = time.monotonic_ns()
            endpoint_sink.write(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "endpointslices",
                    "capture": "churn_recovery",
                    "epoch": plan_entry["epoch"],
                    "observed_ns": observed_ns,
                    "error": None,
                    "payload": endpoints,
                }
            )
            if old_withdrawn_ns is None and old_uids.isdisjoint(endpoint_uids):
                old_withdrawn_ns = observed_ns
        try:
            pods = await _kubectl_json(
                args.kubectl,
                args.namespace,
                "get",
                "pods",
                *names,
            )
            new = [_pod_identity(item) for item in pods.get("items", ())]
            ready_new = (
                len(new) == len(old)
                and all(
                    item["ready"]
                    and item["uid"] not in old_uids
                    and item["uid"] in endpoint_uids
                    for item in new
                )
            )
            if ready_new and new_ready_ns is None:
                new_ready_ns = time.monotonic_ns()
            if (
                old_withdrawn_ns is not None
                and ready_new
                and len(endpoint_uids) == config.replica_count
            ):
                break
        except Exception:
            pass
        await asyncio.sleep(0.25)
    else:
        error = "recovery_timeout"
    recovered_ns = time.monotonic_ns()
    record = {
        "schema_version": SCHEMA_VERSION,
        "epoch": plan_entry["epoch"],
        "deadline_offset_s": plan_entry["deadline_offset_s"],
        "ordinals": plan_entry["ordinals"],
        "scheduled_ns": scheduled_ns,
        "api_started_ns": api_started_ns,
        "api_completed_ns": api_completed_ns,
        "schedule_lateness_ms": (
            api_started_ns - scheduled_ns
        )
        / 1_000_000,
        "delete_returncode": delete_returncode,
        "delete_stderr": delete_stderr[:500],
        "old": old,
        "new": new,
        "endpoint_uids_at_recovery": sorted(endpoint_uids),
        "ready_endpoint_count_at_recovery": len(endpoint_uids),
        "old_withdrawn_ns": old_withdrawn_ns,
        "old_withdrawal_seconds": (
            None
            if old_withdrawn_ns is None
            else (old_withdrawn_ns - api_started_ns) / 1_000_000_000
        ),
        "old_withdrawal_margin_seconds": (
            None
            if old_withdrawn_ns is None
            else config.endpoint_withdrawal_limit_seconds
            - (old_withdrawn_ns - api_started_ns) / 1_000_000_000
        ),
        "new_ready_ns": new_ready_ns,
        "new_ready_seconds": (
            None
            if new_ready_ns is None
            else (new_ready_ns - api_started_ns) / 1_000_000_000
        ),
        "recovered_ns": recovered_ns,
        "recovery_seconds": (
            recovered_ns - api_started_ns
        )
        / 1_000_000_000,
        "error": error,
    }
    sink.write(record)
    return record


async def _churn_loop(
    args: argparse.Namespace,
    config: GateConfig,
    initial_by_name: dict[str, dict[str, Any]],
    measurement_origin_ns: int,
    endpoint_sink: JsonlSink,
    sink: JsonlSink,
) -> list[dict[str, Any]]:
    recoveries: list[asyncio.Task] = []
    try:
        for entry in churn_plan(config):
            scheduled_ns = measurement_origin_ns + int(
                entry["deadline_offset_s"] * 1_000_000_000
            )
            remaining = (
                scheduled_ns - time.monotonic_ns()
            ) / 1_000_000_000
            if remaining > 0:
                await asyncio.sleep(remaining)
            names = [
                f"{args.statefulset}-{ordinal}"
                for ordinal in entry["ordinals"]
            ]
            old = [dict(initial_by_name[name]) for name in names]
            predelete_endpoints = await _kubectl_json(
                args.kubectl,
                args.namespace,
                "get",
                "endpointslices",
                "-l",
                f"kubernetes.io/service-name={args.endpoint_service}",
            )
            endpoint_sink.write(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "endpointslices",
                    "capture": "churn_predelete",
                    "epoch": entry["epoch"],
                    "observed_ns": time.monotonic_ns(),
                    "error": None,
                    "payload": predelete_endpoints,
                }
            )
            api_started_ns = time.monotonic_ns()
            returncode, _, stderr = await _run_command(
                args.kubectl,
                "-n",
                args.namespace,
                "delete",
                "pods",
                *names,
                "--wait=false",
                check=False,
            )
            api_completed_ns = time.monotonic_ns()
            recoveries.append(
                asyncio.create_task(
                    _recover_epoch(
                        args,
                        config,
                        plan_entry=entry,
                        old=old,
                        scheduled_ns=scheduled_ns,
                        api_started_ns=api_started_ns,
                        api_completed_ns=api_completed_ns,
                        delete_returncode=returncode,
                        delete_stderr=stderr,
                        endpoint_sink=endpoint_sink,
                        sink=sink,
                    )
                )
            )
        if not recoveries:
            return []
        return list(await asyncio.gather(*recoveries))
    finally:
        for recovery in recoveries:
            if not recovery.done():
                recovery.cancel()
        if recoveries:
            await asyncio.gather(*recoveries, return_exceptions=True)


def _resource_summary(payload: dict[str, Any], namespace: str) -> dict[str, Any]:
    node = payload.get("node", {})
    pods = []
    for pod in payload.get("pods", ()):
        reference = pod.get("podRef", {})
        if reference.get("namespace") != namespace:
            continue
        pods.append(
            {
                "name": reference.get("name"),
                "uid": reference.get("uid"),
                "cpu_usage_nano_cores": pod.get("cpu", {}).get(
                    "usageNanoCores"
                ),
                "memory_working_set_bytes": pod.get("memory", {}).get(
                    "workingSetBytes"
                ),
                "network": pod.get("network"),
            }
        )
    return {
        "node": {
            "name": node.get("nodeName"),
            "cpu_usage_nano_cores": node.get("cpu", {}).get("usageNanoCores"),
            "memory_working_set_bytes": node.get("memory", {}).get(
                "workingSetBytes"
            ),
        },
        "pods": pods,
    }


def _placement_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(record["placement_us"]) / 1_000
        for record in records
        if record.get("placement_us") is not None
    ]
    if not values:
        return {"samples": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    return {
        "samples": len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values),
    }


def _windowed_placement(
    requests: list[dict[str, Any]],
    config: GateConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    measurement = [
        record
        for record in requests
        if 0 <= record["measurement_offset_s"] < config.measurement_seconds
    ]
    overall = _placement_summary(measurement)
    epochs = []
    churn_windows = []
    for epoch in range(config.churn_epochs):
        start = epoch * config.epoch_seconds
        end = start + config.epoch_seconds
        epoch_rows = [
            record
            for record in measurement
            if start <= record["measurement_offset_s"] < end
        ]
        epochs.append(
            {
                "epoch": epoch,
                "start_s": start,
                "end_s": end,
                **_placement_summary(epoch_rows),
            }
        )
        window_end = start + config.churn_window_seconds
        window_rows = [
            record
            for record in measurement
            if start <= record["measurement_offset_s"] < window_end
        ]
        churn_windows.append(
            {
                "epoch": epoch,
                "start_s": start,
                "end_s": window_end,
                **_placement_summary(window_rows),
            }
        )
    return overall, epochs, churn_windows


def _runtime_mock_image_summary(
    pods: list[dict[str, Any]],
    container_name: str,
) -> dict[str, Any]:
    spec_images: set[str] = set()
    runtime_images: set[str] = set()
    image_ids: set[str] = set()
    for record in pods:
        if record.get("error"):
            continue
        for identity in record.get("payload", ()):
            for container in identity.get("containers", ()):
                if container.get("name") != container_name:
                    continue
                for key, destination in (
                    ("spec_image", spec_images),
                    ("runtime_image", runtime_images),
                    ("image_id", image_ids),
                ):
                    value = container.get(key)
                    if isinstance(value, str) and value:
                        destination.add(value)
    return {
        "container_name": container_name,
        "spec_images": sorted(spec_images),
        "runtime_images": sorted(runtime_images),
        "image_ids": sorted(image_ids),
    }


def _sha256_digest(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"sha256:[0-9a-fA-F]{64}", value)
    return match.group(0).lower() if match is not None else None


def _normalized_image_reference(value: str) -> str:
    for prefix in ("docker.io/library/", "docker.io/", "library/"):
        if value.startswith(prefix):
            return value.removeprefix(prefix)
    return value


def _derived_close(value: Any, expected: float) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
        and math.isclose(
            float(value),
            expected,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    )


def _provenance_valid(environment: dict[str, Any]) -> bool:
    sha256 = re.compile(r"[0-9a-f]{64}")
    commit = re.compile(r"[0-9a-f]{40}")

    def digest(value: Any) -> bool:
        return isinstance(value, str) and sha256.fullmatch(value) is not None

    def image_digest(value: Any) -> bool:
        return (
            isinstance(value, str)
            and re.search(r"sha256:([0-9a-f]{64})(?:$|[^0-9a-f])", value)
            is not None
        )

    frozen_inputs = environment.get("frozen_inputs")
    tool_versions = environment.get("tool_versions")
    git_commit = environment.get("git_commit")
    expected_git_commit = environment.get("expected_git_commit")
    return (
        isinstance(git_commit, str)
        and commit.fullmatch(git_commit) is not None
        and isinstance(expected_git_commit, str)
        and commit.fullmatch(expected_git_commit) is not None
        and git_commit == expected_git_commit
        and environment.get("git_commit_matches_expected") is True
        and digest(environment.get("tracked_diff_sha256"))
        and digest(environment.get("benchmark_sha256"))
        and digest(environment.get("driver_sha256"))
        and image_digest(environment.get("gateway_image_digest"))
        and image_digest(environment.get("mock_image_digest"))
        and image_digest(environment.get("kind_node_image"))
        and isinstance(tool_versions, dict)
        and set(tool_versions) == {"kind", "kubectl", "docker"}
        and all(
            isinstance(value, str) and bool(value.strip())
            for value in tool_versions.values()
        )
        and isinstance(frozen_inputs, list)
        and bool(frozen_inputs)
        and all(
            isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and bool(item["path"])
            and digest(item.get("sha256"))
            for item in frozen_inputs
        )
    )


def evaluate_gate(
    *,
    config: GateConfig,
    requests: list[dict[str, Any]],
    placements: list[dict[str, Any]],
    churn: list[dict[str, Any]],
    endpoints: list[dict[str, Any]],
    pods: list[dict[str, Any]],
    resources: list[dict[str, Any]],
    environment: dict[str, Any],
) -> dict[str, Any]:
    config.validate()
    raw_request_sequences = [record.get("sequence") for record in requests]
    request_sequence_coverage_ok = (
        len(requests) == config.expected_requests
        and all(type(sequence) is int for sequence in raw_request_sequences)
        and len(set(raw_request_sequences)) == len(raw_request_sequences)
        and set(raw_request_sequences) == set(range(config.expected_requests))
    )
    if request_sequence_coverage_ok:
        # Requests are durably appended when each concurrent HTTP call
        # completes, so JSONL row order is intentionally not schedule order.
        # Canonicalize only after proving exact unique sequence coverage.
        requests = sorted(requests, key=lambda record: record["sequence"])
    plan = churn_plan(config)
    all_targets = [
        ordinal for entry in plan for ordinal in entry["ordinals"]
    ]
    raw_placements = [
        record for record in placements if record.get("kind") == "replica"
    ]
    placement_by_request: dict[str, list[dict[str, Any]]] = {}
    for record in raw_placements:
        request_id = record.get("request_id")
        if isinstance(request_id, str) and request_id:
            placement_by_request.setdefault(request_id, []).append(record)
    joined_requests = []
    for request in requests:
        matches = placement_by_request.get(request.get("request_id_echo"), [])
        placement = matches[0] if len(matches) == 1 else {}
        latency_ns = placement.get("placement_latency_ns")
        placement_started_ns = placement.get("placement_started_ns")
        selected_at_ns = placement.get("selected_at_ns")
        send_ns = request.get("send_ns")
        end_ns = request.get("end_ns")
        raw_latency_valid = (
            type(latency_ns) is int
            and type(placement_started_ns) is int
            and type(selected_at_ns) is int
            and type(send_ns) is int
            and type(end_ns) is int
            and send_ns
            <= placement_started_ns
            <= selected_at_ns
            <= end_ns
            and latency_ns == selected_at_ns - placement_started_ns
        )
        joined_requests.append(
            {
                **request,
                "placement_latency_ns": latency_ns,
                "placement_us": (
                    float(selected_at_ns - placement_started_ns) / 1_000
                    if raw_latency_valid
                    else None
                ),
                "replica_id": placement.get("replica_id"),
                "replica_generation": placement.get("replica_generation"),
                "placement_started_ns": placement_started_ns,
                "selected_at_ns": selected_at_ns,
            }
        )
    overall, epochs, churn_windows = _windowed_placement(
        joined_requests,
        config,
    )
    sent = [record for record in requests if not record.get("not_sent")]
    pacing_ok = [
        record.get("pacing_error_ms", math.inf)
        <= 1_000 / config.request_rate
        for record in sent
    ]
    status_2xx = sum(
        isinstance(record.get("status_code"), int)
        and 200 <= record["status_code"] < 300
        for record in requests
    )
    status_429 = sum(record.get("status_code") == 429 for record in requests)
    status_5xx = sum(
        isinstance(record.get("status_code"), int)
        and record["status_code"] >= 500
        for record in requests
    )
    status_other = len(requests) - status_2xx - status_429 - status_5xx
    transport_errors = sum(
        bool(record.get("transport_error")) for record in requests
    )
    not_sent = sum(bool(record.get("not_sent")) for record in requests)
    endpoint_uids = set()
    endpoint_uid_snapshots: list[set[str]] = []
    endpoint_timed_snapshots: list[dict[str, Any]] = []
    endpoint_errors = 0
    for record in endpoints:
        if record.get("error"):
            endpoint_errors += 1
            continue
        snapshot_uids = _endpoint_uids(record.get("payload") or {})
        endpoint_uid_snapshots.append(snapshot_uids)
        if type(record.get("observed_ns")) is int:
            endpoint_timed_snapshots.append(
                {
                    "observed_ns": record["observed_ns"],
                    "fetch_started_ns": (
                        record["fetch_started_ns"]
                        if type(record.get("fetch_started_ns")) is int
                        else record["observed_ns"]
                    ),
                    "uids": snapshot_uids,
                    "capture": record.get("capture"),
                    "epoch": record.get("epoch"),
                }
            )
        endpoint_uids.update(snapshot_uids)
    endpoint_timed_snapshots.sort(key=lambda record: record["observed_ns"])
    pod_uids: set[str] = set()
    pod_name_by_uid: dict[str, str] = {}
    pod_uid_name_consistent = True
    for record in pods:
        if record.get("error"):
            continue
        for identity in record.get("payload", ()):
            uid = identity.get("uid")
            name = identity.get("name")
            if not isinstance(uid, str) or not uid:
                continue
            pod_uids.add(uid)
            if not isinstance(name, str) or not name:
                pod_uid_name_consistent = False
                continue
            previous_name = pod_name_by_uid.setdefault(uid, name)
            if previous_name != name:
                pod_uid_name_consistent = False
    old_uids = {
        uid
        for record in churn
        for identity in record.get("old", ())
        if isinstance((uid := identity.get("uid")), str) and uid
    }
    new_uids = {
        uid
        for record in churn
        for identity in record.get("new", ())
        if isinstance((uid := identity.get("uid")), str) and uid
    }
    response_ids = [
        record.get("request_id_echo")
        for record in requests
        if isinstance(record.get("request_id_echo"), str)
        and record.get("request_id_echo")
    ]
    placement_ids = [
        record.get("request_id")
        for record in raw_placements
        if isinstance(record.get("request_id"), str)
        and record.get("request_id")
    ]
    response_id_set = set(response_ids)
    matched_placement_ids = [
        request_id
        for request_id in placement_ids
        if request_id in response_id_set
    ]
    ordered_churn = sorted(
        churn,
        key=lambda record: record.get("epoch", math.inf),
    )
    measurement_origin_ns = (
        ordered_churn[0].get("scheduled_ns") if ordered_churn else None
    )
    request_sequences = [record.get("sequence") for record in requests]
    request_timeline_ok = (
        request_sequence_coverage_ok
        and
        type(measurement_origin_ns) is int
        and request_sequences == list(range(config.expected_requests))
    )
    if type(measurement_origin_ns) is int:
        traffic_origin_ns = measurement_origin_ns - int(
            config.warmup_seconds * 1_000_000_000
        )
        interval_ns = int(1_000_000_000 / config.request_rate)
        for sequence, record in enumerate(requests):
            scheduled_ns = record.get("scheduled_ns")
            send_ns = record.get("send_ns")
            end_ns = record.get("end_ns")
            expected_scheduled_ns = traffic_origin_ns + sequence * interval_ns
            record_ok = (
                type(scheduled_ns) is int
                and scheduled_ns == expected_scheduled_ns
                and type(send_ns) is int
                and type(end_ns) is int
                and scheduled_ns <= send_ns <= end_ns
                and _derived_close(
                    record.get("measurement_offset_s"),
                    (send_ns - measurement_origin_ns) / 1_000_000_000,
                )
                and _derived_close(
                    record.get("pacing_error_ms"),
                    (send_ns - scheduled_ns) / 1_000_000,
                )
            )
            request_timeline_ok = request_timeline_ok and record_ok
    else:
        request_timeline_ok = False

    churn_timeline_ok = (
        type(measurement_origin_ns) is int
        and len(ordered_churn) == config.churn_epochs
    )
    if type(measurement_origin_ns) is int:
        for epoch, record in enumerate(ordered_churn):
            if epoch >= len(plan):
                churn_timeline_ok = False
                continue
            plan_entry = plan[epoch]
            expected_scheduled_ns = measurement_origin_ns + int(
                float(plan_entry.get("deadline_offset_s", math.inf))
                * 1_000_000_000
            )
            scheduled_ns = record.get("scheduled_ns")
            api_started_ns = record.get("api_started_ns")
            api_completed_ns = record.get("api_completed_ns")
            old_withdrawn_ns = record.get("old_withdrawn_ns")
            new_ready_ns = record.get("new_ready_ns")
            recovered_ns = record.get("recovered_ns")
            raw_times_valid = all(
                type(value) is int
                for value in (
                    scheduled_ns,
                    api_started_ns,
                    api_completed_ns,
                    old_withdrawn_ns,
                    new_ready_ns,
                    recovered_ns,
                )
            )
            causal = (
                raw_times_valid
                and scheduled_ns == expected_scheduled_ns
                and scheduled_ns <= api_started_ns <= api_completed_ns
                and api_started_ns <= old_withdrawn_ns <= recovered_ns
                and api_started_ns <= new_ready_ns <= recovered_ns
                and api_completed_ns <= recovered_ns
            )
            derived = (
                causal
                and record.get("epoch") == epoch
                and record.get("ordinals") == plan_entry.get("ordinals")
                and _derived_close(
                    record.get("deadline_offset_s"),
                    float(plan_entry.get("deadline_offset_s", math.inf)),
                )
                and _derived_close(
                    record.get("schedule_lateness_ms"),
                    (api_started_ns - scheduled_ns) / 1_000_000,
                )
                and _derived_close(
                    record.get("old_withdrawal_seconds"),
                    (old_withdrawn_ns - api_started_ns) / 1_000_000_000,
                )
                and _derived_close(
                    record.get("old_withdrawal_margin_seconds"),
                    config.endpoint_withdrawal_limit_seconds
                    - (
                        old_withdrawn_ns - api_started_ns
                    )
                    / 1_000_000_000,
                )
                and _derived_close(
                    record.get("new_ready_seconds"),
                    (new_ready_ns - api_started_ns) / 1_000_000_000,
                )
                and _derived_close(
                    record.get("recovery_seconds"),
                    (recovered_ns - api_started_ns) / 1_000_000_000,
                )
            )
            churn_timeline_ok = churn_timeline_ok and derived
    else:
        churn_timeline_ok = False

    statefulset = environment.get("statefulset")
    expected_initial_names = {
        f"{statefulset}-{ordinal}"
        for ordinal in range(config.replica_count)
    } if isinstance(statefulset, str) and statefulset else set()
    all_old_entries = [
        item
        for record in ordered_churn
        for item in record.get("old", ())
        if isinstance(item, dict)
    ]
    initial_state_by_name = {
        item.get("name"): item.get("uid") for item in all_old_entries
    }
    initial_pod_payload = (
        pods[0].get("payload", ())
        if pods and not pods[0].get("error")
        else ()
    )
    initial_pod_state = {
        item.get("name"): item.get("uid")
        for item in initial_pod_payload
        if isinstance(item, dict)
    }
    state_mapping_valid = (
        bool(expected_initial_names)
        and len(all_old_entries) == config.replica_count
        and len(initial_state_by_name) == config.replica_count
        and set(initial_state_by_name) == expected_initial_names
        and initial_state_by_name == initial_pod_state
        and set(initial_state_by_name.values()) == old_uids
    )
    current_state_by_name = dict(initial_state_by_name)
    current_fleet = set(current_state_by_name.values())
    ever_seen_uids = set(current_fleet)
    expected_fleet_before_epoch: list[set[str]] = []
    expected_fleet_by_epoch: list[set[str]] = []
    lifecycle_valid = len(current_fleet) == config.replica_count
    exact_identity = True
    for expected_epoch, record in enumerate(ordered_churn):
        expected_fleet_before_epoch.append(set(current_fleet))
        old_entries = record.get("old", ())
        new_entries = record.get("new", ())
        epoch_old_uids = {
            uid
            for item in old_entries
            if isinstance((uid := item.get("uid")), str) and uid
        }
        epoch_new_uids = {
            uid
            for item in new_entries
            if isinstance((uid := item.get("uid")), str) and uid
        }
        plan_entry = plan[expected_epoch] if expected_epoch < len(plan) else {}
        expected_names = [
            f"{statefulset}-{ordinal}"
            for ordinal in plan_entry.get("ordinals", ())
        ]
        epoch_identity_valid = (
            state_mapping_valid
            and
            record.get("epoch") == expected_epoch
            and record.get("ordinals") == plan_entry.get("ordinals")
            and len(old_entries) == config.churn_batch_size
            and len(new_entries) == config.churn_batch_size
            and len(epoch_old_uids) == config.churn_batch_size
            and len(epoch_new_uids) == config.churn_batch_size
            and {
                item.get("name") for item in old_entries
            }
            == {item.get("name") for item in new_entries}
            and [
                item.get("name") for item in old_entries
            ] == expected_names
            and [
                item.get("name") for item in new_entries
            ] == expected_names
            and all(
                current_state_by_name.get(expected_name)
                == old_item.get("uid")
                for expected_name, old_item in zip(
                    expected_names,
                    old_entries,
                    strict=True,
                )
            )
            and epoch_old_uids <= current_fleet
            and epoch_new_uids.isdisjoint(ever_seen_uids)
        )
        exact_identity = exact_identity and epoch_identity_valid
        lifecycle_valid = lifecycle_valid and epoch_identity_valid
        if epoch_identity_valid:
            for expected_name, new_item in zip(
                expected_names,
                new_entries,
                strict=True,
            ):
                current_state_by_name[expected_name] = new_item["uid"]
            current_fleet = set(current_state_by_name.values())
            ever_seen_uids.update(epoch_new_uids)
        expected_fleet_by_epoch.append(set(current_fleet))
    lifecycle_valid = (
        lifecycle_valid
        and len(ordered_churn) == config.churn_epochs
        and len(expected_fleet_by_epoch) == config.churn_epochs
        and len(current_fleet) == config.replica_count
        and current_fleet == new_uids
    )
    endpoint_epoch_recoveries = []
    for epoch, record in enumerate(ordered_churn):
        expected_fleet = (
            expected_fleet_by_epoch[epoch]
            if epoch < len(expected_fleet_by_epoch)
            else set()
        )
        recovery_uids = {
            uid
            for uid in record.get("endpoint_uids_at_recovery", ())
            if isinstance(uid, str) and uid
        }
        api_started_ns = record.get("api_started_ns")
        recovered_ns = record.get("recovered_ns")
        periodic_exact_observed_ns = [
            snapshot["observed_ns"]
            for snapshot in endpoint_timed_snapshots
            if type(api_started_ns) is int
            and type(recovered_ns) is int
            and api_started_ns
            <= snapshot["observed_ns"]
            and snapshot["fetch_started_ns"]
            <= recovered_ns
            + int(config.evidence_interval_seconds * 1_000_000_000)
            and snapshot["capture"] is None
            and snapshot["uids"] == expected_fleet
        ]
        exact = (
            lifecycle_valid
            and recovery_uids == expected_fleet
            and record.get("ready_endpoint_count_at_recovery")
            == config.replica_count
            and record.get("new_ready_ns") is not None
            and bool(periodic_exact_observed_ns)
        )
        endpoint_epoch_recoveries.append(
            {
                "epoch": record.get("epoch"),
                "expected_uids": sorted(expected_fleet),
                "observed_uids": sorted(recovery_uids),
                "periodic_exact_observed_ns": periodic_exact_observed_ns,
                "exact": exact,
            }
        )
    endpoint_recovery_exact = (
        len(endpoint_epoch_recoveries) == config.churn_epochs
        and all(record["exact"] for record in endpoint_epoch_recoveries)
    )
    endpoint_epoch_causality = []
    endpoint_interval_ns = int(
        config.evidence_interval_seconds * 1_000_000_000
    )
    for epoch, record in enumerate(ordered_churn):
        expected_before = (
            expected_fleet_before_epoch[epoch]
            if epoch < len(expected_fleet_before_epoch)
            else set()
        )
        old_epoch_uids = {
            uid
            for item in record.get("old", ())
            if isinstance((uid := item.get("uid")), str) and uid
        }
        api_started_ns = record.get("api_started_ns")
        old_withdrawn_ns = record.get("old_withdrawn_ns")
        predelete_snapshots = []
        postdelete_snapshots = []
        if type(api_started_ns) is int:
            predelete_snapshots = [
                snapshot
                for snapshot in endpoint_timed_snapshots
                if api_started_ns - endpoint_interval_ns
                <= snapshot["observed_ns"]
                <= api_started_ns
                and snapshot["capture"] == "churn_predelete"
                and snapshot["epoch"] == epoch
            ]
            postdelete_snapshots = [
                snapshot
                for snapshot in endpoint_timed_snapshots
                if snapshot["observed_ns"] >= api_started_ns
            ]
        first_disjoint = next(
            (
                snapshot
                for snapshot in postdelete_snapshots
                if old_epoch_uids.isdisjoint(snapshot["uids"])
            ),
            None,
        )
        last_with_old = None
        if first_disjoint is not None:
            last_with_old = next(
                (
                    snapshot
                    for snapshot in reversed(endpoint_timed_snapshots)
                    if snapshot["observed_ns"]
                    < first_disjoint["observed_ns"]
                    and not old_epoch_uids.isdisjoint(snapshot["uids"])
                ),
                None,
            )
        raw_claim_snapshot = next(
            (
                snapshot
                for snapshot in postdelete_snapshots
                if type(old_withdrawn_ns) is int
                and snapshot["observed_ns"] == old_withdrawn_ns
                and old_epoch_uids.isdisjoint(snapshot["uids"])
            ),
            None,
        )
        predelete_exact = (
            len(predelete_snapshots) == 1
            and predelete_snapshots[0]["uids"] == expected_before
        )
        withdrawal_bracketed = (
            first_disjoint is not None
            and last_with_old is not None
            and raw_claim_snapshot is not None
            and type(old_withdrawn_ns) is int
            and first_disjoint["observed_ns"]
            <= old_withdrawn_ns
            <= first_disjoint["observed_ns"] + endpoint_interval_ns
            and first_disjoint["observed_ns"]
            - last_with_old["observed_ns"]
            <= endpoint_interval_ns
        )
        no_reappearance = (
            type(old_withdrawn_ns) is int
            and all(
                old_epoch_uids.isdisjoint(snapshot["uids"])
                for snapshot in endpoint_timed_snapshots
                if snapshot["observed_ns"] >= old_withdrawn_ns
            )
        )
        endpoint_epoch_causality.append(
            {
                "epoch": record.get("epoch"),
                "predelete_exact": predelete_exact,
                "first_disjoint_ns": (
                    first_disjoint["observed_ns"]
                    if first_disjoint is not None
                    else None
                ),
                "last_with_old_ns": (
                    last_with_old["observed_ns"]
                    if last_with_old is not None
                    else None
                ),
                "claim_has_raw_snapshot": raw_claim_snapshot is not None,
                "withdrawal_bracketed": withdrawal_bracketed,
                "no_reappearance": no_reappearance,
            }
        )
    endpoint_predelete_exact = (
        len(endpoint_epoch_causality) == config.churn_epochs
        and all(record["predelete_exact"] for record in endpoint_epoch_causality)
    )
    endpoint_raw_withdrawal_causal = (
        len(endpoint_epoch_causality) == config.churn_epochs
        and all(
            record["withdrawal_bracketed"] and record["no_reappearance"]
            for record in endpoint_epoch_causality
        )
    )
    endpoint_withdrawal_ok = all(
        isinstance(record.get("old_withdrawal_seconds"), (int, float))
        and 0 <= record["old_withdrawal_seconds"]
        < config.endpoint_withdrawal_limit_seconds
        and isinstance(
            record.get("old_withdrawal_margin_seconds"),
            (int, float),
        )
        and record["old_withdrawal_margin_seconds"] > 0
        and math.isclose(
            record["old_withdrawal_margin_seconds"],
            config.endpoint_withdrawal_limit_seconds
            - record["old_withdrawal_seconds"],
            rel_tol=0,
            abs_tol=1e-9,
        )
        and record.get("old_withdrawn_ns") is not None
        for record in churn
    )
    memberships = [
        record for record in placements if record.get("kind") == "membership"
    ]

    def membership_ids(record: dict[str, Any], key: str) -> set[str]:
        raw = record.get(key)
        if not isinstance(raw, list):
            return set()
        return {value for value in raw if isinstance(value, str) and value}

    membership_identity_lists_unique = (
        bool(memberships)
        and all(
            isinstance(record.get(key), list)
            and all(
                isinstance(value, str) and bool(value)
                for value in record[key]
            )
            and len(record[key]) == len(set(record[key]))
            for record in memberships
            for key in ("replica_ids", "healthy_ids", "eligible_ids")
        )
    )
    membership_set_invariants_ok = (
        bool(memberships)
        and all(
            membership_ids(record, "eligible_ids")
            <= membership_ids(record, "healthy_ids")
            <= membership_ids(record, "replica_ids")
            for record in memberships
        )
    )
    membership_initial = any(
        all(
            membership_ids(record, key) == old_uids
            for key in ("replica_ids", "healthy_ids", "eligible_ids")
        )
        for record in memberships
    )
    expected_final_fleet = (
        expected_fleet_by_epoch[-1] if expected_fleet_by_epoch else set()
    )
    membership_final = bool(memberships) and all(
        membership_ids(memberships[-1], key) == expected_final_fleet
        for key in ("replica_ids", "healthy_ids", "eligible_ids")
    )
    membership_sources = [
        record.get("event_source_id") for record in memberships
    ]
    membership_sequences = [record.get("sequence") for record in memberships]
    membership_observed_ns = [
        record.get("observed_ns") for record in memberships
    ]
    event_source = (
        membership_sources[0]
        if membership_sources
        and isinstance(membership_sources[0], str)
        and membership_sources[0]
        else None
    )
    membership_event_stream_ok = (
        bool(memberships)
        and event_source is not None
        and all(source == event_source for source in membership_sources)
        and all(type(sequence) is int for sequence in membership_sequences)
        and membership_sequences == list(range(1, len(memberships) + 1))
        and all(type(value) is int for value in membership_observed_ns)
        and all(
            previous < current
            for previous, current in zip(
                membership_observed_ns,
                membership_observed_ns[1:],
                strict=False,
            )
        )
        and len(
            {
                record.get("event_id")
                for record in memberships
                if isinstance(record.get("event_id"), str)
            }
        )
        == len(memberships)
        and all(
            record.get("event_id")
            == f"{event_source}:{sequence:020d}"
            for sequence, record in enumerate(memberships, 1)
        )
    )
    membership_generation_snapshots_ok = (
        bool(memberships)
        and all(
            isinstance(record.get("generation_by_id"), dict)
            and set(record["generation_by_id"]) == membership_ids(
                record,
                "replica_ids",
            )
            and all(
                isinstance(generation, str) and bool(generation)
                for generation in record["generation_by_id"].values()
            )
            for record in memberships
        )
    )
    membership_epoch_recoveries = []
    for epoch, churn_record in enumerate(ordered_churn):
        api_started_ns = churn_record.get("api_started_ns")
        expected_fleet = (
            expected_fleet_by_epoch[epoch]
            if epoch < len(expected_fleet_by_epoch)
            else set()
        )
        matching = []
        if isinstance(api_started_ns, int):
            deadline_ns = api_started_ns + int(
                config.recovery_timeout_seconds * 1_000_000_000
            )
            for membership in memberships:
                observed_ns = membership.get("observed_ns")
                if (
                    not isinstance(observed_ns, int)
                    or not api_started_ns <= observed_ns <= deadline_ns
                ):
                    continue
                snapshots = [
                    membership_ids(membership, key)
                    for key in (
                        "replica_ids",
                        "healthy_ids",
                        "eligible_ids",
                    )
                ]
                if (
                    all(snapshot == expected_fleet for snapshot in snapshots)
                ):
                    matching.append(membership)
        recovered_membership = (
            min(matching, key=lambda record: record["observed_ns"])
            if matching
            else None
        )
        membership_epoch_recoveries.append(
            {
                "epoch": churn_record.get("epoch"),
                "expected_uids": sorted(expected_fleet),
                "event_id": (
                    recovered_membership.get("event_id")
                    if recovered_membership is not None
                    else None
                ),
                "sequence": (
                    recovered_membership.get("sequence")
                    if recovered_membership is not None
                    else None
                ),
                "observed_ns": (
                    recovered_membership.get("observed_ns")
                    if recovered_membership is not None
                    else None
                ),
                "recovery_seconds": (
                    (
                        recovered_membership["observed_ns"]
                        - api_started_ns
                    )
                    / 1_000_000_000
                    if recovered_membership is not None
                    and isinstance(api_started_ns, int)
                    else None
                ),
            }
        )
    membership_epoch_recovery_ok = (
        lifecycle_valid
        and len(membership_epoch_recoveries) == config.churn_epochs
        and all(
            record["event_id"] is not None
            for record in membership_epoch_recoveries
        )
    )
    membership_epoch_causality = []
    for epoch, churn_record in enumerate(ordered_churn):
        api_started_ns = churn_record.get("api_started_ns")
        old_withdrawn_ns = churn_record.get("old_withdrawn_ns")
        expected_before = (
            expected_fleet_before_epoch[epoch]
            if epoch < len(expected_fleet_before_epoch)
            else set()
        )
        old_epoch_uids = {
            uid
            for item in churn_record.get("old", ())
            if isinstance((uid := item.get("uid")), str) and uid
        }
        predelete_membership = None
        deadline_membership = None
        withdrawal_deadline_ns = None
        if type(api_started_ns) is int:
            predelete_membership = next(
                (
                    membership
                    for membership in reversed(memberships)
                    if type(membership.get("observed_ns")) is int
                    and membership["observed_ns"] < api_started_ns
                ),
                None,
            )
        if type(old_withdrawn_ns) is int:
            withdrawal_deadline_ns = old_withdrawn_ns + endpoint_interval_ns
            deadline_membership = next(
                (
                    membership
                    for membership in reversed(memberships)
                    if type(membership.get("observed_ns")) is int
                    and membership["observed_ns"] <= withdrawal_deadline_ns
                ),
                None,
            )
        predelete_exact = (
            predelete_membership is not None
            and all(
                membership_ids(predelete_membership, key) == expected_before
                for key in ("replica_ids", "healthy_ids", "eligible_ids")
            )
        )
        withdrawn_by_deadline = (
            deadline_membership is not None
            and old_epoch_uids.isdisjoint(
                membership_ids(deadline_membership, "eligible_ids")
            )
        )
        no_eligible_reappearance = (
            withdrawal_deadline_ns is not None
            and all(
                old_epoch_uids.isdisjoint(
                    membership_ids(membership, "eligible_ids")
                )
                for membership in memberships
                if type(membership.get("observed_ns")) is int
                and membership["observed_ns"] >= withdrawal_deadline_ns
            )
        )
        no_late_placement = (
            withdrawal_deadline_ns is not None
            and all(
                placement.get("replica_id") not in old_epoch_uids
                for placement in raw_placements
                if type(placement.get("selected_at_ns")) is int
                and placement["selected_at_ns"] >= withdrawal_deadline_ns
            )
        )
        membership_epoch_causality.append(
            {
                "epoch": churn_record.get("epoch"),
                "predelete_exact": predelete_exact,
                "withdrawal_deadline_ns": withdrawal_deadline_ns,
                "withdrawn_by_deadline": withdrawn_by_deadline,
                "no_eligible_reappearance": no_eligible_reappearance,
                "no_late_placement": no_late_placement,
            }
        )
    membership_withdrawal_causal = (
        len(membership_epoch_causality) == config.churn_epochs
        and all(
            record["predelete_exact"]
            and record["withdrawn_by_deadline"]
            and record["no_eligible_reappearance"]
            and record["no_late_placement"]
            for record in membership_epoch_causality
        )
    )
    placement_membership_failures = []
    placement_membership_matches = 0
    placement_latency_raw_consistent = True
    membership_timeline_ready = (
        membership_event_stream_ok
        and membership_identity_lists_unique
        and membership_set_invariants_ok
        and membership_generation_snapshots_ok
    )
    for request in joined_requests:
        reasons = []
        send_ns = request.get("send_ns")
        placement_started_ns = request.get("placement_started_ns")
        selected_at_ns = request.get("selected_at_ns")
        end_ns = request.get("end_ns")
        placement_latency_ns = request.get("placement_latency_ns")
        if not (
            type(send_ns) is int
            and type(placement_started_ns) is int
            and type(selected_at_ns) is int
            and type(end_ns) is int
            and send_ns <= placement_started_ns <= selected_at_ns <= end_ns
        ):
            reasons.append("request_selection_timestamp_order")
        if not (
            type(placement_latency_ns) is int
            and type(placement_started_ns) is int
            and type(selected_at_ns) is int
            and placement_latency_ns
            == selected_at_ns - placement_started_ns
        ):
            placement_latency_raw_consistent = False
            reasons.append("placement_latency_out_of_range")
        latest_membership = None
        if membership_timeline_ready and type(selected_at_ns) is int:
            membership_index = bisect.bisect_right(
                membership_observed_ns,
                selected_at_ns,
            ) - 1
            if membership_index >= 0:
                latest_membership = memberships[membership_index]
        if latest_membership is None:
            reasons.append("no_membership_at_selection")
        else:
            replica_id = request.get("replica_id")
            replica_ids = membership_ids(
                latest_membership,
                "replica_ids",
            )
            eligible_ids = membership_ids(
                latest_membership,
                "eligible_ids",
            )
            placement = placement_by_request.get(
                request.get("request_id_echo"),
                [{}],
            )[0]
            if placement.get("pool_size") != len(replica_ids):
                reasons.append("pool_size_mismatch")
            if placement.get("eligible_size") != len(eligible_ids):
                reasons.append("eligible_size_mismatch")
            if replica_id not in eligible_ids:
                reasons.append("replica_not_eligible")
            generation_by_id = latest_membership.get("generation_by_id")
            expected_generation = (
                generation_by_id.get(replica_id)
                if isinstance(generation_by_id, dict)
                else None
            )
            if (
                not isinstance(expected_generation, str)
                or request.get("replica_generation") != expected_generation
            ):
                reasons.append("replica_generation_mismatch")
        if reasons:
            if len(placement_membership_failures) < 20:
                placement_membership_failures.append(
                    {
                        "request_id": request.get("request_id_echo"),
                        "selected_at_ns": selected_at_ns,
                        "reasons": reasons,
                    }
                )
        else:
            placement_membership_matches += 1
    placement_membership_join_ok = (
        len(joined_requests) == len(requests)
        and placement_membership_matches == len(requests)
    )
    mock_container_name = environment.get("mock_container_name")
    runtime_mock_image = _runtime_mock_image_summary(
        pods,
        (
            mock_container_name
            if isinstance(mock_container_name, str)
            else ""
        ),
    )
    expected_mock_image = environment.get("expected_mock_image")
    expected_mock_digest = _sha256_digest(
        environment.get("mock_image_digest")
    )
    runtime_image_digests = {
        digest
        for raw in runtime_mock_image["image_ids"]
        if (digest := _sha256_digest(raw)) is not None
    }
    expected_lifecycle_uids = old_uids | new_uids
    mock_pod_uids = set()
    image_evidence_uids = set()
    for record in pods:
        if record.get("error"):
            continue
        for identity in record.get("payload", ()):
            uid = identity.get("uid")
            if not isinstance(uid, str) or not uid:
                continue
            for container in identity.get("containers", ()):
                if container.get("name") == mock_container_name:
                    mock_pod_uids.add(uid)
                if (
                    container.get("name") == mock_container_name
                    and container.get("spec_image") == expected_mock_image
                    and container.get("runtime_image")
                    in runtime_mock_image["runtime_images"]
                    and container.get("image_id")
                    in runtime_mock_image["image_ids"]
                ):
                    image_evidence_uids.add(uid)
                    break
    runtime_mock_per_uid_ok = (
        mock_pod_uids == expected_lifecycle_uids
        and image_evidence_uids == expected_lifecycle_uids
    )
    runtime_mock_image_ok = (
        isinstance(mock_container_name, str)
        and bool(mock_container_name)
        and isinstance(expected_mock_image, str)
        and runtime_mock_image == environment.get("runtime_mock_image")
        and runtime_mock_image["spec_images"] == [expected_mock_image]
        and {
            _normalized_image_reference(value)
            for value in runtime_mock_image["runtime_images"]
        }
        == {_normalized_image_reference(expected_mock_image)}
        and expected_mock_digest is not None
        and runtime_image_digests == {expected_mock_digest}
        and runtime_mock_per_uid_ok
    )
    gateway_container_name = environment.get("gateway_container_name")
    expected_gateway_image = environment.get("expected_gateway_image")
    expected_gateway_digest = _sha256_digest(
        environment.get("gateway_image_digest")
    )
    runtime_gateway_image = _runtime_mock_image_summary(
        pods,
        (
            gateway_container_name
            if isinstance(gateway_container_name, str)
            else ""
        ),
    )
    runtime_gateway_digests = {
        digest
        for raw in runtime_gateway_image["image_ids"]
        if (digest := _sha256_digest(raw)) is not None
    }
    gateway_instances = [
        identity
        for record in pods
        if not record.get("error")
        for identity in record.get("payload", ())
        if any(
            container.get("name") == gateway_container_name
            for container in identity.get("containers", ())
        )
    ]
    runtime_gateway_image_ok = (
        isinstance(gateway_container_name, str)
        and bool(gateway_container_name)
        and isinstance(expected_gateway_image, str)
        and bool(expected_gateway_image)
        and len(
            {
                identity.get("uid")
                for identity in gateway_instances
                if isinstance(identity.get("uid"), str)
                and identity.get("uid")
            }
        )
        == 1
        and runtime_gateway_image
        == environment.get("runtime_gateway_image")
        and runtime_gateway_image["spec_images"] == [expected_gateway_image]
        and {
            _normalized_image_reference(value)
            for value in runtime_gateway_image["runtime_images"]
        }
        == {_normalized_image_reference(expected_gateway_image)}
        and expected_gateway_digest is not None
        and runtime_gateway_digests == {expected_gateway_digest}
    )
    placement_windows = [overall, *epochs, *churn_windows]
    placement_ok = all(
        item["samples"] > 0
        and item["p99_ms"] is not None
        and item["p99_ms"] < config.placement_p99_limit_ms
        for item in placement_windows
    )
    required_provenance = _provenance_valid(environment)
    response_identity_matches_pods = (
        pod_uid_name_consistent
        and all(
            record.get("response_valid") is True
            and isinstance(record.get("pod_name"), str)
            and bool(record["pod_name"])
            and isinstance(record.get("pod_uid"), str)
            and bool(record["pod_uid"])
            and pod_name_by_uid.get(record["pod_uid"]) == record["pod_name"]
            for record in requests
        )
    )
    checks = {
        "formal_configuration_frozen": (
            config.profile != "formal" or config == FORMAL_CONFIG
        ),
        "schedule_exact_disjoint_coverage": (
            len(plan) == config.churn_epochs
            and len(all_targets) == config.replica_count
            and sorted(all_targets) == list(range(config.replica_count))
        ),
        "request_count_matches_schedule": (
            len(requests) == config.expected_requests
        ),
        "request_raw_timeline_consistent": request_timeline_ok,
        "all_requests_valid_2xx": (
            status_2xx == len(requests)
            and status_429 == 0
            and status_5xx == 0
            and status_other == 0
        ),
        "zero_transport_or_not_sent": (
            transport_errors == 0 and not_sent == 0
        ),
        "request_id_correlation_complete": (
            len(response_ids) == len(requests)
            and len(response_ids) == len(set(response_ids))
            and len(raw_placements) == len(requests)
            and set(placement_by_request) == set(response_ids)
            and len(matched_placement_ids) == len(requests)
            and all(
                len(placement_by_request[request_id]) == 1
                for request_id in response_ids
            )
        ),
        "response_and_identity_complete": response_identity_matches_pods,
        "pod_uid_name_mapping_consistent": pod_uid_name_consistent,
        "placement_to_upstream_uid_join_complete": all(
            record.get("replica_id")
            and record.get("replica_id") == record.get("pod_uid")
            for record in joined_requests
        ),
        "placement_membership_temporal_join_complete": (
            placement_membership_join_ok
        ),
        "placement_latency_raw_consistent": placement_latency_raw_consistent,
        "placement_samples_complete": (
            placement_latency_raw_consistent
            and all(
                record.get("placement_us") is not None
                for record in joined_requests
            )
            and overall["samples"]
            == sum(
                0 <= record["measurement_offset_s"]
                < config.measurement_seconds
                for record in joined_requests
            )
        ),
        "placement_p99_all_windows": placement_ok,
        "open_loop_pacing": (
            bool(pacing_ok)
            and sum(pacing_ok) / len(pacing_ok)
            >= config.pacing_success_fraction
        ),
        "churn_epochs_complete": (
            len(churn) == config.churn_epochs
            and {record.get("epoch") for record in churn}
            == set(range(config.churn_epochs))
        ),
        "churn_delete_and_recovery_success": all(
            record.get("delete_returncode") == 0
            and record.get("error") is None
            and record.get("recovery_seconds", math.inf)
            <= config.recovery_timeout_seconds
            for record in churn
        ),
        "churn_deadlines_met": all(
            abs(record.get("schedule_lateness_ms", math.inf))
            <= config.schedule_lateness_limit_ms
            for record in churn
        ),
        "churn_raw_timeline_consistent": churn_timeline_ok,
        "old_endpoints_withdrawn_before_grace": endpoint_withdrawal_ok,
        "endpoint_predelete_snapshots_exact": endpoint_predelete_exact,
        "endpoint_raw_withdrawal_causal": endpoint_raw_withdrawal_causal,
        "new_endpoints_ready_with_exact_fleet_size": endpoint_recovery_exact,
        "pod_uid_replaced_exactly_once": (
            exact_identity
            and lifecycle_valid
            and len(old_uids) == config.replica_count
            and len(new_uids) == config.replica_count
            and not old_uids & new_uids
        ),
        "endpoint_uid_join_complete": (
            endpoint_errors == 0
            and old_uids <= endpoint_uids
            and new_uids <= endpoint_uids
        ),
        "initial_endpoints_exact_old_fleet": (
            endpoint_errors == 0
            and any(snapshot == old_uids for snapshot in endpoint_uid_snapshots)
        ),
        "final_endpoints_exact_new_fleet": (
            endpoint_errors == 0
            and any(snapshot == new_uids for snapshot in endpoint_uid_snapshots)
        ),
        "gateway_initial_membership_all_eligible": membership_initial,
        "gateway_final_membership_all_replacements_eligible": membership_final,
        "membership_event_stream_consistent": membership_event_stream_ok,
        "membership_identity_lists_unique": (
            membership_identity_lists_unique
        ),
        "membership_set_invariants_valid": membership_set_invariants_ok,
        "membership_generation_snapshots_complete": (
            membership_generation_snapshots_ok
        ),
        "gateway_membership_recovers_each_epoch": (
            membership_epoch_recovery_ok
        ),
        "gateway_membership_withdrawal_causal": (
            membership_withdrawal_causal
        ),
        "pod_uid_evidence_complete": (
            pod_uid_name_consistent
            and old_uids <= pod_uids
            and new_uids <= pod_uids
        ),
        "runtime_mock_image_pinned": runtime_mock_image_ok,
        "runtime_gateway_image_pinned": runtime_gateway_image_ok,
        "resource_evidence_complete": (
            bool(resources)
            and all(record.get("error") is None for record in resources)
        ),
        "clean_source": not environment.get(
            "source_dirty",
            environment.get("tracked_dirty", True),
        ),
        "source_stable_during_run": (
            environment.get("source_end_match") is True
        ),
        "provenance_pinned": required_provenance,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "traffic": {
            "expected": config.expected_requests,
            "observed": len(requests),
            "http_2xx": status_2xx,
            "http_429": status_429,
            "http_5xx": status_5xx,
            "http_other_or_missing": status_other,
            "transport_errors": transport_errors,
            "not_sent": not_sent,
            "pacing_within_one_interval_fraction": (
                sum(pacing_ok) / len(pacing_ok) if pacing_ok else 0.0
            ),
        },
        "placement": {
            "overall": overall,
            "epochs": epochs,
            "churn_windows": churn_windows,
            "raw_replica_rows": len(raw_placements),
            "matched_request_rows": len(matched_placement_ids),
            "unrelated_replica_rows": (
                len(raw_placements) - len(matched_placement_ids)
            ),
            "membership_join_matches": placement_membership_matches,
            "membership_join_failures": placement_membership_failures,
        },
        "churn": {
            "planned_epochs": len(plan),
            "observed_epochs": len(churn),
            "old_uid_count": len(old_uids),
            "new_uid_count": len(new_uids),
            "endpoint_uid_count": len(endpoint_uids),
            "pod_uid_count": len(pod_uids),
            "lifecycle_exact": lifecycle_valid,
            "endpoint_epoch_recoveries": endpoint_epoch_recoveries,
            "endpoint_epoch_causality": endpoint_epoch_causality,
            "max_old_withdrawal_seconds": max(
                (
                    float(record["old_withdrawal_seconds"])
                    for record in churn
                    if isinstance(
                        record.get("old_withdrawal_seconds"),
                        (int, float),
                    )
                ),
                default=None,
            ),
            "min_old_withdrawal_margin_seconds": min(
                (
                    float(record["old_withdrawal_margin_seconds"])
                    for record in churn
                    if isinstance(
                        record.get("old_withdrawal_margin_seconds"),
                        (int, float),
                    )
                ),
                default=None,
            ),
        },
        "membership": {
            "events": len(memberships),
            "event_source_id": event_source,
            "first_sequence": (
                membership_sequences[0] if membership_sequences else None
            ),
            "last_sequence": (
                membership_sequences[-1] if membership_sequences else None
            ),
            "initial_all_eligible_seen": membership_initial,
            "final_all_eligible": membership_final,
            "epoch_recoveries": membership_epoch_recoveries,
            "epoch_causality": membership_epoch_causality,
        },
        "runtime_mock_image": runtime_mock_image,
        "runtime_gateway_image": runtime_gateway_image,
    }


def _sidecar_descriptor(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed = [
        value
        for row in rows
        for key in (
            "observed_ns",
            "send_ns",
            "api_started_ns",
            "selected_at_ns",
        )
        if isinstance((value := row.get(key)), int)
    ]
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "rows": len(rows),
        "min_monotonic_ns": min(observed) if observed else None,
        "max_monotonic_ns": max(observed) if observed else None,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _safe_sidecar(manifest_path: Path, raw: str) -> Path:
    if not raw or Path(raw).is_absolute() or ".." in Path(raw).parts:
        raise ValueError(f"unsafe sidecar path {raw!r}")
    path = (manifest_path.parent / raw).resolve()
    if path.parent != manifest_path.parent.resolve():
        raise ValueError(f"sidecar escapes artifact directory: {raw!r}")
    return path


def verify_artifact(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("artifact manifest must be a JSON object")
    checks: dict[str, bool] = {
        "schema_version": manifest.get("schema_version") == SCHEMA_VERSION,
        "gate": manifest.get("gate") == GATE,
    }
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name in (
        "requests",
        "placements",
        "churn",
        "endpointslices",
        "pods",
        "resources",
    ):
        descriptor = (manifest.get("sidecars") or {}).get(name)
        if not isinstance(descriptor, dict):
            checks[f"sidecar:{name}:descriptor"] = False
            loaded[name] = []
            continue
        path = _safe_sidecar(manifest_path, str(descriptor.get("path", "")))
        exists = path.is_file()
        checks[f"sidecar:{name}:exists"] = exists
        if not exists:
            loaded[name] = []
            continue
        rows = _load_jsonl(path)
        loaded[name] = rows
        checks[f"sidecar:{name}:hash"] = (
            _sha256_file(path) == descriptor.get("sha256")
        )
        checks[f"sidecar:{name}:rows"] = len(rows) == descriptor.get("rows")
    try:
        config = GateConfig(**manifest["configuration"])
        checks["published_schedule_matches_frozen_plan"] = (
            manifest.get("schedule") == churn_plan(config)
        )
        replay = evaluate_gate(
            config=config,
            requests=loaded["requests"],
            placements=loaded["placements"],
            churn=loaded["churn"],
            endpoints=loaded["endpointslices"],
            pods=loaded["pods"],
            resources=loaded["resources"],
            environment=manifest.get("environment") or {},
        )
    except (KeyError, TypeError, ValueError) as error:
        checks["published_schedule_matches_frozen_plan"] = False
        replay = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    checks["replayed_gate_passed"] = replay.get("passed") is True
    for key in (
        "checks",
        "passed",
        "traffic",
        "placement",
        "churn",
        "membership",
        "runtime_mock_image",
        "runtime_gateway_image",
    ):
        checks[f"published_{key}_matches_replay"] = (
            key in replay and manifest.get(key) == replay[key]
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": f"{GATE}-artifact-verifier",
        "manifest": str(manifest_path.resolve()),
        "checks": checks,
        "replayed": replay,
        "verified": all(checks.values()),
    }


async def run(args: argparse.Namespace, config: GateConfig) -> dict[str, Any]:
    source_start = source_environment(args)
    artifact_dir = Path(args.output).resolve().parent
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: artifact_dir / f"{name}.jsonl"
        for name in (
            "requests",
            "placements",
            "churn",
            "endpointslices",
            "pods",
            "resources",
        )
    }
    sinks = {name: JsonlSink(path) for name, path in paths.items()}
    stop = asyncio.Event()
    owned_tasks: list[asyncio.Task] = []
    try:
        initial = await _initial_inventory(args, config)
        initial_by_name = {item["name"]: item for item in initial}
        initial_pod_record = {
            "schema_version": SCHEMA_VERSION,
            "kind": "pods",
            "observed_ns": time.monotonic_ns(),
            "error": None,
            "payload": initial,
        }
        sinks["pods"].write(initial_pod_record)
        measurement_origin_ns = time.monotonic_ns() + int(
            config.warmup_seconds * 1_000_000_000
        )

        async def fetch_endpoints() -> dict[str, Any]:
            return await _kubectl_json(
                args.kubectl,
                args.namespace,
                "get",
                "endpointslices",
                "-l",
                f"kubernetes.io/service-name={args.endpoint_service}",
            )

        async def fetch_pods() -> list[dict[str, Any]]:
            payload = await _kubectl_json(
                args.kubectl,
                args.namespace,
                "get",
                "pods",
                "-l",
                args.pod_selector,
            )
            return [_pod_identity(item) for item in payload.get("items", ())]

        node = str(initial[0]["node"])

        async def fetch_resources() -> dict[str, Any]:
            _, stdout, _ = await _run_command(
                args.kubectl,
                "get",
                "--raw",
                f"/api/v1/nodes/{node}/proxy/stats/summary",
            )
            return _resource_summary(json.loads(stdout), args.namespace)

        endpoint_task = asyncio.create_task(
            _snapshot_loop(
                kind="endpointslices",
                interval_s=config.evidence_interval_seconds,
                stop=stop,
                sink=sinks["endpointslices"],
                fetch=fetch_endpoints,
            )
        )
        owned_tasks.append(endpoint_task)
        pod_task = asyncio.create_task(
            _snapshot_loop(
                kind="pods",
                interval_s=config.evidence_interval_seconds,
                stop=stop,
                sink=sinks["pods"],
                fetch=fetch_pods,
            )
        )
        owned_tasks.append(pod_task)
        resource_task = asyncio.create_task(
            _snapshot_loop(
                kind="resources",
                interval_s=config.resource_interval_seconds,
                stop=stop,
                sink=sinks["resources"],
                fetch=fetch_resources,
            )
        )
        owned_tasks.append(resource_task)
        traffic_task = asyncio.create_task(
            _traffic_loop(
                args,
                config,
                sinks["requests"],
                measurement_origin_ns,
            )
        )
        owned_tasks.append(traffic_task)
        churn_task = asyncio.create_task(
            _churn_loop(
                args,
                config,
                initial_by_name,
                measurement_origin_ns,
                sinks["endpointslices"],
                sinks["churn"],
            )
        )
        owned_tasks.append(churn_task)
        requests, churn = await asyncio.gather(traffic_task, churn_task)
        placements = await _fetch_gateway_placements(
            args,
            expected_request_ids={
                request_id
                for request in requests
                if isinstance(
                    (request_id := request.get("request_id_echo")),
                    str,
                )
                and request_id
            },
            initial_uids={
                str(identity["uid"])
                for identity in initial
                if identity.get("uid")
            },
            final_uids={
                str(identity["uid"])
                for record in churn
                for identity in record.get("new", ())
                if identity.get("uid")
            },
            sink=sinks["placements"],
        )
        stop.set()
        endpoints, pod_samples, resources = await asyncio.gather(
            endpoint_task,
            pod_task,
            resource_task,
        )
        pods = [initial_pod_record, *pod_samples]
        gateway_snapshot = await _kubectl_json(
            args.kubectl,
            args.namespace,
            "get",
            "pods",
            "-l",
            args.gateway_pod_selector,
        )
        gateway_pod_record = {
            "schema_version": SCHEMA_VERSION,
            "kind": "gateway_pods",
            "observed_ns": time.monotonic_ns(),
            "error": None,
            "payload": [
                _pod_identity(item)
                for item in gateway_snapshot.get("items", ())
            ],
        }
        sinks["pods"].write(gateway_pod_record)
        pods.append(gateway_pod_record)
    finally:
        stop.set()
        for task in owned_tasks:
            if not task.done():
                task.cancel()
        if owned_tasks:
            await asyncio.gather(*owned_tasks, return_exceptions=True)
        for sink in sinks.values():
            sink.close()

    # The churn recovery loop writes causally joined EndpointSlice snapshots to
    # the same sidecar as the periodic sampler. Reload the closed file so the
    # in-process evaluation and the independent artifact replay see identical
    # raw rows.
    endpoints = _load_jsonl(paths["endpointslices"])
    source_end = source_environment(args)
    environment = dict(source_start)
    environment.update(
        {
            "source_end_match": source_end == source_start,
            "statefulset": args.statefulset,
            "mock_container_name": args.mock_container,
            "expected_mock_image": args.mock_image,
            "runtime_mock_image": _runtime_mock_image_summary(
                pods,
                args.mock_container,
            ),
            "gateway_container_name": args.gateway_container,
            "expected_gateway_image": args.gateway_image,
            "runtime_gateway_image": _runtime_mock_image_summary(
                pods,
                args.gateway_container,
            ),
        }
    )
    replay = evaluate_gate(
        config=config,
        requests=requests,
        placements=placements,
        churn=churn,
        endpoints=endpoints,
        pods=pods,
        resources=resources,
        environment=environment,
    )
    sidecars = {
        name: _sidecar_descriptor(paths[name], rows)
        for name, rows in (
            ("requests", requests),
            ("placements", placements),
            ("churn", churn),
            ("endpointslices", endpoints),
            ("pods", pods),
            ("resources", resources),
        )
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "methodology": {
            "traffic": (
                "absolute-deadline open-loop requests; retry=0; 429, 5xx, "
                "transport errors, and unsent arrivals are failures"
            ),
            "churn": (
                "fixed-seed permutation of every StatefulSet ordinal into "
                "disjoint batches at absolute monotonic deadlines"
            ),
            "placement": (
                "gateway request receipt through replica selection, joined "
                "from the echoed X-Request-ID to one raw gateway audit record"
            ),
            "percentile": "nearest-rank over raw samples",
            "windows": "overall measurement, every epoch, and every post-delete window",
            "endpoint_identity": "EndpointSlice targetRef.uid joined to old/new pod UIDs",
            "endpoint_readiness": (
                "only ready=true and terminating!=true endpoints count; "
                "old withdrawal must precede the five-second preStop grace"
            ),
            "membership": (
                "gateway audit snapshots must begin with the full old fleet "
                "eligible and end with the full replacement fleet eligible"
            ),
            "clock_join": (
                "Linux/CPython time.monotonic_ns and time.perf_counter_ns "
                "share the same monotonic clock for churn-to-membership joins"
            ),
        },
        "assumptions": {
            "namespace": args.namespace,
            "statefulset": args.statefulset,
            "pod_selector": args.pod_selector,
            "endpoint_service": args.endpoint_service,
            "gateway_url": args.gateway_url,
            "model": args.model,
            "gateway_placement_log": args.gateway_placement_log,
            "mock_container": args.mock_container,
            "mock_image": args.mock_image,
            "headers": asdict(
                SurfaceContract(request_id_header=args.request_id_header)
            ),
        },
        "configuration": asdict(config),
        "schedule": churn_plan(config),
        "environment": environment,
        "sidecars": sidecars,
        **replay,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="f1a")
    parser.add_argument("--namespace", default="kairyu-f1a")
    parser.add_argument("--statefulset", default="f1a-replica")
    parser.add_argument(
        "--pod-selector",
        default="app.kubernetes.io/name=f1a-replica",
    )
    parser.add_argument("--endpoint-service", default="f1a-replicas")
    parser.add_argument("--mock-container", default="mock")
    parser.add_argument("--mock-image", default="kairyu-f1a-mock:dev")
    parser.add_argument(
        "--gateway-pod-selector",
        default="app.kubernetes.io/name=f1a-gateway",
    )
    parser.add_argument("--gateway-container", default="gateway")
    parser.add_argument("--gateway-image", default="kairyu:dev")
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument("--request-id-header", default="X-Request-ID")
    parser.add_argument(
        "--gateway-placement-log",
        default="/evidence/placements.jsonl",
    )
    parser.add_argument(
        "--placement-log-timeout-seconds",
        type=float,
        default=10.0,
    )
    for field in fields(GateConfig):
        if field.name == "profile":
            continue
        option = "--" + field.name.replace("_", "-")
        value_type = type(getattr(FORMAL_CONFIG, field.name))
        parser.add_argument(option, type=value_type, default=None)
    parser.add_argument(
        "--frozen-input",
        action="append",
        default=[],
        help="manifest/config input whose content hash is pinned (repeatable)",
    )
    parser.add_argument("--gateway-image-digest")
    parser.add_argument("--mock-image-digest")
    parser.add_argument("--kind-node-image")
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--kind-version")
    parser.add_argument("--kubectl-version")
    parser.add_argument("--docker-version")
    parser.add_argument(
        "--driver-path",
        default="scripts/kind_churn_gate.sh",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bench/results/f1a-fleet-churn/manifest.json"),
    )
    parser.add_argument("--verify-artifact", type=Path)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--assert-gate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.verify_artifact is not None:
        result = verify_artifact(args.verify_artifact.resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
        return int(args.assert_gate and not result["verified"])
    try:
        config = resolve_config(args)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if args.print_plan:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "gate": GATE,
                    "configuration": asdict(config),
                    "schedule": churn_plan(config),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = asyncio.run(run(args, config))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
    print(rendered)
    if args.assert_gate and not result["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
