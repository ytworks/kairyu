#!/usr/bin/env python3
"""Automated kind F1b zero-failure StatefulSet rollout gate (issue #176).

The formal profile freezes one hundred mock replicas and retry-free open-loop
traffic.  One ``kubectl rollout restart`` creates the target ControllerRevision
while ``partition=100`` holds every old Pod.  The driver then walks ordinals
99 down to 0.  For each ordinal it explicitly drains the old Pod through the
Kubernetes API proxy, observes Pod readiness and EndpointSlice withdrawal,
waits until the production gateway has made that UID ineligible with no
outstanding work, lowers the partition by exactly one, and waits for the new
UID to be Ready and eligible before continuing.

Requests, gateway placement and membership records, rollout transitions, Pod
snapshots, and EndpointSlice snapshots are stored as hashed JSONL sidecars.
``--verify-artifact`` independently replays those raw records.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

import verification.fleet.resilience.fleet_churn_bench as fleet_churn
from verification.fleet.resilience.fleet_churn_bench import (
    JsonlSink,
    KubernetesProxyReader,
    _containerd_metadata_file_matches,
    _cri_metadata_file_matches,
    _fetch_gateway_placements,
    _git_output,
    _load_jsonl,
    _pod_identity,
    _pod_identity_matches_image,
    _provenance_valid,
    _read_endpoint_slices,
    _read_pods,
    _run_command,
    _runtime_mock_image_summary,
    _safe_sidecar,
    _sha256_file,
    _sidecar_descriptor,
)

SCHEMA_VERSION = 1
GATE = "G5-F1b"
_POLL_SECONDS = 0.25
_MOCK_IDENTITY = re.compile(
    r"\bkairyu-f1a-mock pod_name=(?P<pod_name>\S+) "
    r"pod_uid=(?P<pod_uid>\S+)"
)
_PROMETHEUS_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+(?P<value>\S+)"
    r"(?:\s+\d+)?$"
)
_PROMETHEUS_LABEL = re.compile(
    r'(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)="'
    r'(?P<value>(?:\\.|[^"])*)"(?:,|$)'
)


@dataclass(frozen=True)
class RolloutGateConfig:
    profile: str
    replica_count: int
    request_rate: float
    warmup_seconds: float
    cooldown_seconds: float
    max_concurrency: int
    request_timeout_seconds: float
    evidence_interval_seconds: float
    withdrawal_timeout_seconds: float
    replacement_timeout_seconds: float
    rollout_timeout_seconds: float
    schedule_lateness_limit_ms: float
    pacing_success_fraction: float

    def validate(self) -> None:
        if self.profile not in {"formal", "smoke"}:
            raise ValueError("profile must be 'formal' or 'smoke'")
        if type(self.replica_count) is not int or self.replica_count < 1:
            raise ValueError("replica_count must be a positive integer")
        if type(self.max_concurrency) is not int or self.max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        positive = (
            self.request_rate,
            self.request_timeout_seconds,
            self.evidence_interval_seconds,
            self.withdrawal_timeout_seconds,
            self.replacement_timeout_seconds,
            self.rollout_timeout_seconds,
            self.schedule_lateness_limit_ms,
        )
        if any(
            type(value) not in (int, float) or not math.isfinite(float(value)) or value <= 0
            for value in positive
        ):
            raise ValueError("gate rates, durations, and limits must be positive")
        if self.warmup_seconds < 0 or self.cooldown_seconds < 0:
            raise ValueError("warmup/cooldown must be non-negative")
        if not 0 < self.pacing_success_fraction <= 1:
            raise ValueError("pacing_success_fraction must be in (0, 1]")
        if self.rollout_timeout_seconds <= self.replacement_timeout_seconds:
            raise ValueError("rollout timeout must exceed a single replacement timeout")


FORMAL_CONFIG = RolloutGateConfig(
    profile="formal",
    replica_count=100,
    request_rate=50.0,
    warmup_seconds=30.0,
    cooldown_seconds=30.0,
    max_concurrency=256,
    request_timeout_seconds=10.0,
    evidence_interval_seconds=0.25,
    withdrawal_timeout_seconds=5.0,
    replacement_timeout_seconds=60.0,
    rollout_timeout_seconds=1_500.0,
    schedule_lateness_limit_ms=1_000.0,
    pacing_success_fraction=0.99,
)

SMOKE_CONFIG = RolloutGateConfig(
    profile="smoke",
    replica_count=4,
    request_rate=10.0,
    warmup_seconds=2.0,
    cooldown_seconds=2.0,
    max_concurrency=32,
    request_timeout_seconds=5.0,
    evidence_interval_seconds=0.25,
    withdrawal_timeout_seconds=5.0,
    replacement_timeout_seconds=30.0,
    rollout_timeout_seconds=180.0,
    schedule_lateness_limit_ms=1_000.0,
    pacing_success_fraction=0.95,
)


def rollout_plan(config: RolloutGateConfig) -> list[dict[str, int]]:
    """Return the frozen StatefulSet controller order."""

    config.validate()
    return [
        {
            "step": step,
            "ordinal": ordinal,
            "partition": ordinal,
        }
        for step, ordinal in enumerate(range(config.replica_count - 1, -1, -1))
    ]


def resolve_config(args: argparse.Namespace) -> RolloutGateConfig:
    base = FORMAL_CONFIG if args.profile == "formal" else SMOKE_CONFIG
    overrides: dict[str, Any] = {}
    for field in fields(RolloutGateConfig):
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


def source_environment(args: argparse.Namespace) -> dict[str, Any]:
    """Reuse the audited F1a provenance chain with F1b-owned source scopes."""

    environment = fleet_churn.source_environment(args)
    repository = Path(str(_git_output("rev-parse", "--show-toplevel")).strip()).resolve()
    scopes = [
        ".dockerignore",
        "Dockerfile",
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "kairyu",
        "verification/fleet/resilience/fleet_churn_bench.py",
        "verification/fleet/resilience/fleet_rollout_bench.py",
        "scripts/kind_rollout_gate.sh",
        ".github/workflows/f1b-rollout.yml",
        "deploy/kind/f1a",
        "deploy/kind/f1b",
    ]
    untracked = _git_output(
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        *scopes,
    )
    assert isinstance(untracked, str)
    untracked_inputs = sorted(line for line in untracked.splitlines() if line)
    tracked_dirty = bool(environment.get("tracked_dirty"))
    environment.update(
        {
            "benchmark_sha256": _sha256_file(Path(__file__).resolve()),
            "gate_input_scopes": scopes,
            "untracked_gate_input_dirty": bool(untracked_inputs),
            "untracked_gate_inputs": untracked_inputs,
            "source_dirty": tracked_dirty or bool(untracked_inputs),
            "repository": str(repository),
        }
    )
    return environment


class RolloutKubernetesProxy(KubernetesProxyReader):
    """F1a's lifecycle-owned reader plus bounded POST/PATCH support."""

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        content_type: str | None = None,
        raise_for_status: bool = True,
    ) -> tuple[int, dict[str, Any] | None, str]:
        client = self._client
        if client is None:
            raise RuntimeError("kubectl proxy reader is not running")
        headers = {"Content-Type": content_type} if content_type else None
        response = await client.request(
            method,
            path,
            json=body,
            headers=headers,
        )
        text = response.text
        if raise_for_status:
            response.raise_for_status()
        try:
            payload = response.json()
        except (TypeError, ValueError):
            payload = None
        if payload is not None and not isinstance(payload, dict):
            payload = None
        return response.status_code, payload, text


def _statefulset_path(args: argparse.Namespace) -> str:
    namespace = quote(str(args.namespace), safe="")
    name = quote(str(args.statefulset), safe="")
    return f"/apis/apps/v1/namespaces/{namespace}/statefulsets/{name}"


async def _read_statefulset(args: argparse.Namespace) -> dict[str, Any]:
    reader = getattr(args, "_kubernetes_proxy_reader", None)
    if isinstance(reader, KubernetesProxyReader):
        return await reader.get_json(_statefulset_path(args))
    _, stdout, _ = await _run_command(
        args.kubectl,
        "-n",
        args.namespace,
        "get",
        "statefulset",
        args.statefulset,
        "-o",
        "json",
    )
    payload = json.loads(stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("StatefulSet JSON output must be an object")
    return payload


def _statefulset_identity(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata") or {}
    spec = payload.get("spec") or {}
    status = payload.get("status") or {}
    strategy = spec.get("updateStrategy") or {}
    rolling = strategy.get("rollingUpdate") or {}
    return {
        "name": metadata.get("name"),
        "namespace": metadata.get("namespace"),
        "uid": metadata.get("uid"),
        "generation": metadata.get("generation"),
        "observed_generation": status.get("observedGeneration"),
        "replicas": spec.get("replicas"),
        "strategy": strategy.get("type"),
        "partition": rolling.get("partition"),
        "status_replicas": status.get("replicas", 0),
        "ready_replicas": status.get("readyReplicas", 0),
        "current_replicas": status.get("currentReplicas", 0),
        "updated_replicas": status.get("updatedReplicas", 0),
        "available_replicas": status.get("availableReplicas", 0),
        "current_revision": status.get("currentRevision"),
        "update_revision": status.get("updateRevision"),
    }


async def _patch_partition(
    args: argparse.Namespace,
    partition: int,
) -> dict[str, Any]:
    reader = getattr(args, "_kubernetes_proxy_reader", None)
    body = {
        "spec": {
            "updateStrategy": {
                "type": "RollingUpdate",
                "rollingUpdate": {"partition": partition},
            }
        }
    }
    if isinstance(reader, RolloutKubernetesProxy):
        _, payload, _ = await reader.request_json(
            "PATCH",
            _statefulset_path(args),
            body=body,
            content_type="application/merge-patch+json",
        )
        if payload is None:
            raise RuntimeError("StatefulSet partition PATCH returned no JSON")
        return payload
    _, stdout, _ = await _run_command(
        args.kubectl,
        "-n",
        args.namespace,
        "patch",
        "statefulset",
        args.statefulset,
        "--type=merge",
        "-p",
        json.dumps(body, separators=(",", ":")),
        "-o",
        "json",
    )
    payload = json.loads(stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("StatefulSet partition PATCH output must be an object")
    return payload


def _rollout_pod_identity(item: dict[str, Any]) -> dict[str, Any]:
    identity = _pod_identity(item)
    metadata = item.get("metadata") or {}
    labels = metadata.get("labels") or {}
    observed_container_names = {
        container.get("name")
        for container in identity.get("containers", ())
        if isinstance(container, dict)
    }
    for container in item.get("spec", {}).get("containers", ()):
        name = container.get("name") if isinstance(container, dict) else None
        image = container.get("image") if isinstance(container, dict) else None
        if isinstance(name, str) and name not in observed_container_names:
            identity["containers"].append(
                {
                    "name": name,
                    "spec_image": image,
                    "runtime_image": image,
                    "image_id": "",
                }
            )
    identity["containers"].sort(key=lambda container: str(container.get("name")))
    identity.update(
        {
            "creation_timestamp": metadata.get("creationTimestamp"),
            "deletion_timestamp": metadata.get("deletionTimestamp"),
            "revision": labels.get("controller-revision-hash"),
        }
    )
    return identity


def _ordinal(name: Any, statefulset: str) -> int | None:
    if not isinstance(name, str):
        return None
    prefix = f"{statefulset}-"
    if not name.startswith(prefix):
        return None
    raw = name.removeprefix(prefix)
    if not raw.isdigit():
        return None
    return int(raw)


def _pod_map(
    payload: Any,
    *,
    statefulset: str,
) -> tuple[dict[int, dict[str, Any]], bool]:
    if not isinstance(payload, list):
        return {}, False
    result: dict[int, dict[str, Any]] = {}
    uids: set[str] = set()
    for identity in payload:
        if not isinstance(identity, dict):
            return {}, False
        ordinal = _ordinal(identity.get("name"), statefulset)
        uid = identity.get("uid")
        if (
            ordinal is None
            or ordinal in result
            or not isinstance(uid, str)
            or not uid
            or uid in uids
        ):
            return {}, False
        result[ordinal] = identity
        uids.add(uid)
    return result, True


def _endpoint_targets(
    snapshot: dict[str, Any],
    *,
    namespace: str,
) -> tuple[dict[str, str], bool]:
    return fleet_churn._ready_endpoint_target_map(
        snapshot,
        namespace=namespace,
    )


def _parse_prometheus_labels(raw: str | None) -> dict[str, str] | None:
    if raw is None:
        return {}
    labels: dict[str, str] = {}
    position = 0
    while position < len(raw):
        match = _PROMETHEUS_LABEL.match(raw, position)
        if match is None or match.group("name") in labels:
            return None
        try:
            value = json.loads(f'"{match.group("value")}"')
        except json.JSONDecodeError:
            return None
        if not isinstance(value, str):
            return None
        labels[match.group("name")] = value
        position = match.end()
    return labels


def _gateway_metrics_snapshot(text: str, *, pool: str) -> dict[str, Any]:
    pool_counts: dict[str, int] = {}
    outstanding_by_uid: dict[str, int] = {}
    healthy_by_uid: dict[str, bool] = {}
    parse_errors = 0
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        sample = _PROMETHEUS_SAMPLE.fullmatch(line)
        if sample is None:
            continue
        name = sample.group("name")
        if name not in {
            "kairyu_pool_replicas",
            "kairyu_replica_outstanding",
            "kairyu_replica_healthy",
        }:
            continue
        labels = _parse_prometheus_labels(sample.group("labels"))
        try:
            value = float(sample.group("value"))
        except ValueError:
            value = math.nan
        if labels is None or not math.isfinite(value):
            parse_errors += 1
            continue
        if labels.get("pool") != pool:
            continue
        if name == "kairyu_pool_replicas":
            state = labels.get("state")
            if state in {"configured", "healthy", "draining"}:
                pool_counts[state] = int(value)
            continue
        uid = labels.get("replica")
        if not uid:
            parse_errors += 1
            continue
        if name == "kairyu_replica_outstanding":
            outstanding_by_uid[uid] = int(value)
        else:
            healthy_by_uid[uid] = value == 1.0
    uids = set(outstanding_by_uid) | set(healthy_by_uid)
    return {
        "configured": pool_counts.get("configured"),
        "healthy": pool_counts.get("healthy"),
        "draining": pool_counts.get("draining"),
        "outstanding_by_uid": dict(sorted(outstanding_by_uid.items())),
        "healthy_by_uid": dict(sorted(healthy_by_uid.items())),
        "uids": sorted(uids),
        "parse_errors": parse_errors,
    }


async def _read_gateway_metrics(
    client: httpx.AsyncClient,
    *,
    pool: str,
) -> dict[str, Any]:
    response = await client.get("/metrics")
    response.raise_for_status()
    return _gateway_metrics_snapshot(response.text, pool=pool)


async def _drain_pod(
    args: argparse.Namespace,
    pod_name: str,
) -> tuple[int, dict[str, Any] | None, str]:
    reader = getattr(args, "_kubernetes_proxy_reader", None)
    namespace = quote(str(args.namespace), safe="")
    pod = quote(pod_name, safe="")
    path = f"/api/v1/namespaces/{namespace}/pods/{pod}:8080/proxy/admin/drain"
    if isinstance(reader, RolloutKubernetesProxy):
        return await reader.request_json(
            "POST",
            path,
            body={},
            raise_for_status=False,
        )
    returncode, stdout, stderr = await _run_command(
        args.kubectl,
        "-n",
        args.namespace,
        "exec",
        pod_name,
        "--",
        "wget",
        "-qO-",
        "--post-data={}",
        "http://127.0.0.1:8080/admin/drain",
        check=False,
    )
    try:
        payload = json.loads(stdout)
    except (TypeError, ValueError):
        payload = None
    return (200 if returncode == 0 else 500), payload, stderr


async def _one_request(
    client: httpx.AsyncClient,
    *,
    sequence: int,
    target_ns: int,
    origin_ns: int,
    model: str,
    request_id_header: str,
) -> dict[str, Any]:
    sent_ns = time.monotonic_ns()
    request_id = f"f1b-{sequence:08d}-{uuid.uuid4().hex[:8]}"
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "request_id": request_id,
        "scheduled_ns": target_ns,
        "send_ns": sent_ns,
        "traffic_offset_s": (sent_ns - origin_ns) / 1_000_000_000,
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
            headers={
                "X-Request-ID": request_id,
                "X-Session-ID": f"f1b-session-{sequence % 4096:04d}",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"f1b request {sequence}",
                    }
                ],
                "max_tokens": 1,
                "temperature": 0,
            },
        )
        record["status_code"] = response.status_code
        record["request_id_echo"] = response.headers.get(request_id_header)
        try:
            payload = response.json()
            choices = payload.get("choices")
            content = (
                (choices[0].get("message") or {}).get("content")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict)
                else None
            )
            identity = _MOCK_IDENTITY.fullmatch(content) if isinstance(content, str) else None
            if identity is not None:
                record.update(identity.groupdict())
            record["response_valid"] = (
                response.status_code == 200
                and isinstance(record["request_id_echo"], str)
                and bool(record["request_id_echo"])
                and identity is not None
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
    config: RolloutGateConfig,
    sink: JsonlSink,
    stop: asyncio.Event,
    origin_ns: int,
) -> list[dict[str, Any]]:
    interval_ns = int(1_000_000_000 / config.request_rate)
    records: list[dict[str, Any]] = []
    pending: set[asyncio.Task[dict[str, Any]]] = set()
    sequence = 0

    def complete(task: asyncio.Task[dict[str, Any]]) -> None:
        pending.discard(task)
        if task.cancelled():
            return
        record = task.result()
        records.append(record)
        sink.write(record)

    try:
        async with httpx.AsyncClient(
            base_url=args.gateway_url.rstrip("/"),
            timeout=httpx.Timeout(config.request_timeout_seconds),
            limits=httpx.Limits(
                max_connections=config.max_concurrency,
                max_keepalive_connections=config.max_concurrency,
            ),
            trust_env=False,
        ) as client:
            while not stop.is_set():
                target_ns = origin_ns + sequence * interval_ns
                remaining = (target_ns - time.monotonic_ns()) / 1_000_000_000
                if remaining > 0:
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=remaining)
                        break
                    except TimeoutError:
                        pass
                if stop.is_set():
                    break
                if len(pending) >= config.max_concurrency:
                    now_ns = time.monotonic_ns()
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "sequence": sequence,
                        "request_id": f"f1b-{sequence:08d}-not-sent",
                        "scheduled_ns": target_ns,
                        "send_ns": None,
                        "end_ns": now_ns,
                        "traffic_offset_s": (now_ns - origin_ns) / 1_000_000_000,
                        "pacing_error_ms": (now_ns - target_ns) / 1_000_000,
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
                    sink.write(record)
                else:
                    task = asyncio.create_task(
                        _one_request(
                            client,
                            sequence=sequence,
                            target_ns=target_ns,
                            origin_ns=origin_ns,
                            model=args.model,
                            request_id_header=args.request_id_header,
                        )
                    )
                    pending.add(task)
                    task.add_done_callback(complete)
                sequence += 1
            if pending:
                await asyncio.gather(*tuple(pending))
    finally:
        remaining_tasks = tuple(pending)
        for task in remaining_tasks:
            if not task.done():
                task.cancel()
        if remaining_tasks:
            await asyncio.gather(*remaining_tasks, return_exceptions=True)
    return sorted(records, key=lambda row: row["sequence"])


def _pod_record(
    *,
    capture: str,
    observed_ns: int,
    snapshot: dict[str, Any],
    step: int | None = None,
    ordinal: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "pods",
        "capture": capture,
        "step": step,
        "ordinal": ordinal,
        "observed_ns": observed_ns,
        "error": None,
        "payload": [
            _rollout_pod_identity(item)
            for item in snapshot.get("items", ())
            if isinstance(item, dict)
        ],
    }


def _endpoint_record(
    *,
    capture: str,
    observed_ns: int,
    snapshot: dict[str, Any],
    step: int | None = None,
    ordinal: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "endpointslices",
        "capture": capture,
        "step": step,
        "ordinal": ordinal,
        "observed_ns": observed_ns,
        "error": None,
        "payload": snapshot,
    }


async def _capture_inventory(
    args: argparse.Namespace,
    *,
    pod_sink: JsonlSink,
    endpoint_sink: JsonlSink,
    capture: str,
    step: int | None = None,
    ordinal: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    pods, endpoints = await asyncio.gather(
        _read_pods(args, selector=args.pod_selector),
        _read_endpoint_slices(args),
    )
    observed_ns = time.monotonic_ns()
    pod_sink.write(
        _pod_record(
            capture=capture,
            observed_ns=observed_ns,
            snapshot=pods,
            step=step,
            ordinal=ordinal,
        )
    )
    endpoint_sink.write(
        _endpoint_record(
            capture=capture,
            observed_ns=observed_ns,
            snapshot=endpoints,
            step=step,
            ordinal=ordinal,
        )
    )
    return pods, endpoints, observed_ns


def _gateway_withdrawn(
    metrics: dict[str, Any],
    *,
    old_uid: str,
    replica_count: int,
    expected_uids: set[str] | None = None,
) -> bool:
    raw_uids = metrics.get("uids")
    outstanding = metrics.get("outstanding_by_uid")
    if (
        not isinstance(raw_uids, list)
        or not all(isinstance(uid, str) for uid in raw_uids)
        or not isinstance(outstanding, dict)
        or metrics.get("parse_errors") != 0
    ):
        return False
    uids = set(raw_uids)
    if old_uid not in uids:
        return (
            metrics.get("configured") == replica_count - 1
            and metrics.get("healthy") == replica_count - 1
            and metrics.get("draining") == 0
            and (expected_uids is None or uids == expected_uids - {old_uid})
        )
    return (
        metrics.get("configured") == replica_count
        and metrics.get("healthy") == replica_count
        and metrics.get("draining") == 1
        and outstanding.get(old_uid) == 0
        and (expected_uids is None or uids == expected_uids)
    )


async def _wait_withdrawn(
    args: argparse.Namespace,
    config: RolloutGateConfig,
    *,
    gateway_client: httpx.AsyncClient,
    pod_sink: JsonlSink,
    endpoint_sink: JsonlSink,
    step: int,
    ordinal: int,
    old_uid: str,
) -> dict[str, Any]:
    deadline_ns = time.monotonic_ns() + int(config.withdrawal_timeout_seconds * 1_000_000_000)
    deadline = deadline_ns / 1_000_000_000
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            async with asyncio.timeout(remaining):
                pods, endpoints, observed_ns = await _capture_inventory(
                    args,
                    pod_sink=pod_sink,
                    endpoint_sink=endpoint_sink,
                    capture="withdrawal_poll",
                    step=step,
                    ordinal=ordinal,
                )
                metrics = await _read_gateway_metrics(
                    gateway_client,
                    pool=args.model,
                )
        except TimeoutError:
            break
        identities = [
            _rollout_pod_identity(item) for item in pods.get("items", ()) if isinstance(item, dict)
        ]
        pod_by_ordinal, pods_valid = _pod_map(
            identities,
            statefulset=args.statefulset,
        )
        target = pod_by_ordinal.get(ordinal)
        endpoint_by_name, endpoints_valid = _endpoint_targets(
            endpoints,
            namespace=args.namespace,
        )
        target_name = f"{args.statefulset}-{ordinal}"
        pod_uids = {
            identity.get("uid") for identity in identities if isinstance(identity.get("uid"), str)
        }
        expected_ready_endpoints = {
            identity["name"]: identity["uid"]
            for identity in identities
            if identity.get("ready") is True
            and isinstance(identity.get("name"), str)
            and isinstance(identity.get("uid"), str)
        }
        barrier_observed_ns = time.monotonic_ns()
        last = {
            "observed_ns": barrier_observed_ns,
            "snapshot_observed_ns": observed_ns,
            "pod_ready_false": (
                pods_valid
                and target is not None
                and target.get("uid") == old_uid
                and target.get("ready") is False
            ),
            "endpoint_absent": (
                endpoints_valid
                and endpoint_by_name == expected_ready_endpoints
                and endpoint_by_name.get(target_name) != old_uid
            ),
            "gateway_withdrawn": _gateway_withdrawn(
                metrics,
                old_uid=old_uid,
                replica_count=config.replica_count,
                expected_uids=pod_uids,
            ),
            "gateway": metrics,
        }
        if (
            all(
                last[key]
                for key in (
                    "pod_ready_false",
                    "endpoint_absent",
                    "gateway_withdrawn",
                )
            )
            and barrier_observed_ns <= deadline_ns
        ):
            return last
        if barrier_observed_ns > deadline_ns:
            break
        remaining = deadline - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(min(config.evidence_interval_seconds, remaining))
    raise TimeoutError(f"ordinal {ordinal} UID {old_uid} did not withdraw: {last}")


async def _wait_replacement(
    args: argparse.Namespace,
    config: RolloutGateConfig,
    *,
    gateway_client: httpx.AsyncClient,
    pod_sink: JsonlSink,
    endpoint_sink: JsonlSink,
    step: int,
    ordinal: int,
    old_uid: str,
    update_revision: str,
) -> dict[str, Any]:
    deadline_ns = time.monotonic_ns() + int(config.replacement_timeout_seconds * 1_000_000_000)
    deadline = deadline_ns / 1_000_000_000
    expected_updated = config.replica_count - ordinal
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            async with asyncio.timeout(remaining):
                (
                    pods,
                    endpoints,
                    observed_ns,
                ) = await _capture_inventory(
                    args,
                    pod_sink=pod_sink,
                    endpoint_sink=endpoint_sink,
                    capture="replacement_poll",
                    step=step,
                    ordinal=ordinal,
                )
                metrics, statefulset = await asyncio.gather(
                    _read_gateway_metrics(gateway_client, pool=args.model),
                    _read_statefulset(args),
                )
        except TimeoutError:
            break
        identities = [
            _rollout_pod_identity(item) for item in pods.get("items", ()) if isinstance(item, dict)
        ]
        pod_by_ordinal, pods_valid = _pod_map(
            identities,
            statefulset=args.statefulset,
        )
        target = pod_by_ordinal.get(ordinal)
        new_uid = target.get("uid") if target is not None else None
        endpoint_by_name, endpoints_valid = _endpoint_targets(
            endpoints,
            namespace=args.namespace,
        )
        state = _statefulset_identity(statefulset)
        all_ready = (
            pods_valid
            and len(pod_by_ordinal) == config.replica_count
            and all(identity.get("ready") is True for identity in identities)
        )
        endpoint_uids = set(endpoint_by_name.values())
        pod_uids = {
            identity.get("uid") for identity in identities if isinstance(identity.get("uid"), str)
        }
        pod_name_to_uid = {
            identity["name"]: identity["uid"]
            for identity in identities
            if isinstance(identity.get("name"), str) and isinstance(identity.get("uid"), str)
        }
        gateway_uids = metrics.get("uids")
        barrier_observed_ns = time.monotonic_ns()
        last = {
            "observed_ns": barrier_observed_ns,
            "snapshot_observed_ns": observed_ns,
            "new_uid": new_uid,
            "pod_replaced": (
                isinstance(new_uid, str)
                and new_uid != old_uid
                and target is not None
                and target.get("ready") is True
                and target.get("revision") == update_revision
            ),
            "all_pods_ready": all_ready,
            "endpoints_exact": (
                endpoints_valid
                and len(endpoint_by_name) == config.replica_count
                and endpoint_uids == pod_uids
                and endpoint_by_name == pod_name_to_uid
                and endpoint_by_name.get(f"{args.statefulset}-{ordinal}") == new_uid
            ),
            "gateway_exact": (
                metrics.get("parse_errors") == 0
                and metrics.get("configured") == config.replica_count
                and metrics.get("healthy") == config.replica_count
                and metrics.get("draining") == 0
                and isinstance(new_uid, str)
                and isinstance(gateway_uids, list)
                and all(isinstance(uid, str) for uid in gateway_uids)
                and set(gateway_uids) == pod_uids
                and new_uid in gateway_uids
                and old_uid not in gateway_uids
            ),
            "statefulset_advanced": (
                state.get("strategy") == "RollingUpdate"
                and state.get("partition") == ordinal
                and state.get("update_revision") == update_revision
                and state.get("updated_replicas") == expected_updated
                and state.get("ready_replicas") == config.replica_count
                and isinstance(state.get("generation"), int)
                and isinstance(state.get("observed_generation"), int)
                and state["observed_generation"] >= state["generation"]
            ),
            "gateway": metrics,
            "statefulset": state,
        }
        if (
            all(
                last[key]
                for key in (
                    "pod_replaced",
                    "all_pods_ready",
                    "endpoints_exact",
                    "gateway_exact",
                    "statefulset_advanced",
                )
            )
            and barrier_observed_ns <= deadline_ns
        ):
            return last
        if barrier_observed_ns > deadline_ns:
            break
        remaining = deadline - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(min(config.evidence_interval_seconds, remaining))
    raise TimeoutError(f"ordinal {ordinal} UID {old_uid} was not replaced: {last}")


def _membership_rows(
    placements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [row for row in placements if row.get("kind") == "membership"]


def _replica_placements(
    placements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [row for row in placements if row.get("kind") == "replica"]


def _membership_stream_valid(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    source_ids: set[str] = set()
    event_ids: set[str] = set()
    previous_observed: int | None = None
    previous_sequence: int | None = None
    for row in rows:
        observed = row.get("observed_ns")
        source = row.get("event_source_id")
        sequence = row.get("sequence")
        event_id = row.get("event_id")
        replica_ids = row.get("replica_ids")
        healthy_ids = row.get("healthy_ids")
        eligible_ids = row.get("eligible_ids")
        generation_by_id = row.get("generation_by_id")
        if (
            type(observed) is not int
            or not isinstance(source, str)
            or not source
            or type(sequence) is not int
            or sequence < 1
            or not isinstance(event_id, str)
            or not event_id
            or not isinstance(replica_ids, list)
            or not isinstance(healthy_ids, list)
            or not isinstance(eligible_ids, list)
            or not isinstance(generation_by_id, dict)
            or not all(isinstance(value, str) for value in replica_ids)
            or not all(isinstance(value, str) for value in healthy_ids)
            or not all(isinstance(value, str) for value in eligible_ids)
            or len(replica_ids) != len(set(replica_ids))
            or len(healthy_ids) != len(set(healthy_ids))
            or len(eligible_ids) != len(set(eligible_ids))
            or not set(healthy_ids) <= set(replica_ids)
            or not set(eligible_ids) <= set(healthy_ids)
            or set(generation_by_id) != set(replica_ids)
            or not all(isinstance(value, str) and value for value in generation_by_id.values())
            or event_id in event_ids
        ):
            return False
        if previous_observed is not None and observed < previous_observed:
            return False
        if previous_sequence is not None and sequence != previous_sequence + 1:
            return False
        source_ids.add(source)
        event_ids.add(event_id)
        previous_observed = observed
        previous_sequence = sequence
    return len(source_ids) == 1


def _latest_membership(
    memberships: list[dict[str, Any]],
    selected_at_ns: int,
) -> dict[str, Any] | None:
    latest = None
    for row in memberships:
        observed_ns = row.get("observed_ns")
        if type(observed_ns) is not int or observed_ns > selected_at_ns:
            break
        latest = row
    return latest


def _record_by_capture(
    records: list[dict[str, Any]],
    capture: str,
) -> list[dict[str, Any]]:
    return [row for row in records if row.get("capture") == capture]


def _replica_pod_map(
    record: dict[str, Any] | None,
    *,
    statefulset: str,
) -> tuple[dict[int, dict[str, Any]], bool]:
    if record is None:
        return {}, False
    return _pod_map(record.get("payload"), statefulset=statefulset)


def _endpoint_map_from_record(
    record: dict[str, Any] | None,
    *,
    namespace: str,
) -> tuple[dict[str, str], bool]:
    if record is None or not isinstance(record.get("payload"), dict):
        return {}, False
    return _endpoint_targets(record["payload"], namespace=namespace)


def _runtime_images_valid(
    records: list[dict[str, Any]],
    *,
    environment: dict[str, Any],
    capture: str,
    container_name: str,
    expected_image: str,
    expected_count: int,
) -> bool:
    matching = _record_by_capture(records, capture)
    if len(matching) != 1:
        return False
    payload = matching[0].get("payload")
    metadata = environment.get(
        "gateway_cri_image_metadata" if capture == "gateway" else "mock_cri_image_metadata"
    )
    if not isinstance(payload, list) or not isinstance(metadata, dict):
        return False
    allowed_digests = metadata.get("runtime_digests")
    if not isinstance(allowed_digests, list) or not all(
        isinstance(value, str) for value in allowed_digests
    ):
        return False
    return len(payload) == expected_count and all(
        isinstance(identity, dict)
        and _pod_identity_matches_image(
            identity,
            container_name=container_name,
            expected_image=expected_image,
            allowed_digests=set(allowed_digests),
        )
        for identity in payload
    )


def _all_replica_runtime_images_valid(
    records: list[dict[str, Any]],
    *,
    environment: dict[str, Any],
    container_name: str,
    expected_image: str,
) -> bool:
    metadata = environment.get("mock_cri_image_metadata")
    if not isinstance(metadata, dict):
        return False
    allowed_digests = metadata.get("runtime_digests")
    if not isinstance(allowed_digests, list) or not all(
        isinstance(value, str) for value in allowed_digests
    ):
        return False
    allowed = set(allowed_digests)
    if not allowed:
        return False
    seen_uids: set[str] = set()
    pinned_uids: set[str] = set()
    for record in records:
        if record.get("capture") == "gateway":
            continue
        payload = record.get("payload")
        if not isinstance(payload, list):
            return False
        for identity in payload:
            if not isinstance(identity, dict):
                return False
            uid = identity.get("uid")
            containers = identity.get("containers")
            if not isinstance(uid, str) or not uid or not isinstance(containers, list):
                return False
            seen_uids.add(uid)
            if _pod_identity_matches_image(
                identity,
                container_name=container_name,
                expected_image=expected_image,
                allowed_digests=allowed,
            ):
                pinned_uids.add(uid)
                continue
            matches = [
                container
                for container in containers
                if isinstance(container, dict) and container.get("name") == container_name
            ]
            if (
                not matches
                and not containers
                and identity.get("phase") == "Pending"
                and identity.get("ready") is False
            ):
                # Kubernetes may publish a new immutable Pod UID before its
                # first ContainerStatus. The same UID must still become
                # digest-attested below; otherwise seen_uids != pinned_uids.
                continue
            if len(matches) != 1:
                return False
            container = matches[0]
            runtime_image = container.get("runtime_image")
            if not (
                identity.get("phase") == "Pending"
                and identity.get("ready") is False
                and container.get("image_id") in (None, "")
                and container.get("spec_image") == expected_image
                and isinstance(runtime_image, str)
                and fleet_churn._normalized_image_reference(runtime_image)
                == fleet_churn._normalized_image_reference(expected_image)
            ):
                return False
    return bool(seen_uids) and seen_uids == pinned_uids


def evaluate_gate(
    *,
    config: RolloutGateConfig,
    requests: list[dict[str, Any]],
    placements: list[dict[str, Any]],
    rollout: list[dict[str, Any]],
    pods: list[dict[str, Any]],
    endpoints: list[dict[str, Any]],
    environment: dict[str, Any],
    statefulset: str = "f1a-replica",
    namespace: str = "kairyu-f1a",
    mock_container: str = "mock",
    mock_image: str = "kairyu-f1a-mock:dev",
    gateway_container: str = "gateway",
    gateway_image: str = "kairyu:dev",
    kubectl: str = "kubectl",
) -> dict[str, Any]:
    """Replay F1b solely from raw sidecars and pinned environment metadata."""

    config.validate()
    plan = rollout_plan(config)
    initial_rows = [row for row in rollout if row.get("kind") == "initial"]
    restart_rows = [row for row in rollout if row.get("kind") == "restart"]
    step_rows = [row for row in rollout if row.get("kind") == "step"]
    final_rows = [row for row in rollout if row.get("kind") == "final"]
    initial = initial_rows[0] if len(initial_rows) == 1 else {}
    restart = restart_rows[0] if len(restart_rows) == 1 else {}
    final = final_rows[0] if len(final_rows) == 1 else {}

    initial_pod_rows = _record_by_capture(pods, "initial")
    final_pod_rows = _record_by_capture(pods, "final")
    initial_endpoint_rows = _record_by_capture(endpoints, "initial")
    final_endpoint_rows = _record_by_capture(endpoints, "final")
    initial_pods, initial_pods_valid = _replica_pod_map(
        initial_pod_rows[0] if len(initial_pod_rows) == 1 else None,
        statefulset=statefulset,
    )
    final_pods, final_pods_valid = _replica_pod_map(
        final_pod_rows[0] if len(final_pod_rows) == 1 else None,
        statefulset=statefulset,
    )
    initial_endpoints, initial_endpoints_valid = _endpoint_map_from_record(
        initial_endpoint_rows[0] if len(initial_endpoint_rows) == 1 else None,
        namespace=namespace,
    )
    final_endpoints, final_endpoints_valid = _endpoint_map_from_record(
        final_endpoint_rows[0] if len(final_endpoint_rows) == 1 else None,
        namespace=namespace,
    )
    ordinals = set(range(config.replica_count))
    initial_uid_by_ordinal = {
        ordinal: identity.get("uid") for ordinal, identity in initial_pods.items()
    }
    final_uid_by_ordinal = {
        ordinal: identity.get("uid") for ordinal, identity in final_pods.items()
    }
    initial_uids = {uid for uid in initial_uid_by_ordinal.values() if isinstance(uid, str)}
    final_uids = {uid for uid in final_uid_by_ordinal.values() if isinstance(uid, str)}
    pod_name_by_uid = {
        identity.get("uid"): identity.get("name")
        for identity in (*initial_pods.values(), *final_pods.values())
        if isinstance(identity.get("uid"), str) and isinstance(identity.get("name"), str)
    }
    initial_name_to_uid = {
        identity["name"]: identity["uid"]
        for identity in initial_pods.values()
        if isinstance(identity.get("name"), str) and isinstance(identity.get("uid"), str)
    }
    final_name_to_uid = {
        identity["name"]: identity["uid"]
        for identity in final_pods.values()
        if isinstance(identity.get("name"), str) and isinstance(identity.get("uid"), str)
    }
    initial_state = initial.get("statefulset") or {}
    final_state = final.get("statefulset") or {}
    update_revision = restart.get("update_revision")

    request_sequences = [row.get("sequence") for row in requests]
    request_sequence_coverage = (
        bool(requests)
        and all(type(value) is int for value in request_sequences)
        and len(set(request_sequences)) == len(request_sequences)
        and set(request_sequences) == set(range(len(requests)))
    )
    if request_sequence_coverage:
        requests = sorted(requests, key=lambda row: row["sequence"])
    interval_ns = int(1_000_000_000 / config.request_rate)
    first_scheduled_ns = requests[0].get("scheduled_ns") if requests else None
    request_schedule_exact = (
        request_sequence_coverage
        and type(first_scheduled_ns) is int
        and all(
            type(row.get("scheduled_ns")) is int
            and type(row.get("send_ns")) is int
            and row["scheduled_ns"] <= row["send_ns"]
            and row["scheduled_ns"] == first_scheduled_ns + index * interval_ns
            and type(row.get("pacing_error_ms")) in (int, float)
            and math.isfinite(float(row["pacing_error_ms"]))
            and math.isclose(
                float(row["pacing_error_ms"]),
                (row["send_ns"] - row["scheduled_ns"]) / 1_000_000,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            for index, row in enumerate(requests)
        )
    )
    sent_rows = [row for row in requests if row.get("not_sent") is False]
    zero_failures = len(sent_rows) == len(requests) and all(
        row.get("status_code") == 200
        and row.get("transport_error") is None
        and row.get("response_valid") is True
        and isinstance(row.get("request_id_echo"), str)
        and bool(row["request_id_echo"])
        and isinstance(row.get("pod_uid"), str)
        and isinstance(row.get("pod_name"), str)
        and pod_name_by_uid.get(row["pod_uid"]) == row["pod_name"]
        for row in sent_rows
    )
    on_time = sum(
        type(row.get("pacing_error_ms")) in (int, float)
        and row["pacing_error_ms"] <= config.schedule_lateness_limit_ms
        for row in requests
    )
    pacing_fraction = on_time / len(requests) if requests else 0.0
    traffic_started_ns = initial.get("traffic_started_ns")
    traffic_stop_requested_ns = final.get("traffic_stop_requested_ns")
    traffic_stopped_ns = final.get("traffic_stopped_ns")
    rollout_started_ns = restart.get("started_ns")
    rollout_completed_ns = final.get("rollout_completed_ns")
    traffic_encloses_rollout = (
        type(traffic_started_ns) is int
        and type(traffic_stop_requested_ns) is int
        and type(rollout_started_ns) is int
        and type(rollout_completed_ns) is int
        and traffic_started_ns
        < rollout_started_ns
        < rollout_completed_ns
        < traffic_stop_requested_ns
        and rollout_started_ns - traffic_started_ns
        >= int(config.warmup_seconds * 1_000_000_000 * 0.95)
        and traffic_stop_requested_ns - rollout_completed_ns
        >= int(config.cooldown_seconds * 1_000_000_000 * 0.95)
    )
    traffic_raw_coverage = (
        request_schedule_exact
        and type(traffic_started_ns) is int
        and first_scheduled_ns == traffic_started_ns
        and type(traffic_stop_requested_ns) is int
        and traffic_stop_requested_ns > traffic_started_ns
        and len(requests)
        == (traffic_stop_requested_ns - traffic_started_ns + interval_ns - 1) // interval_ns
        and type(traffic_stopped_ns) is int
        and traffic_stop_requested_ns <= traffic_stopped_ns
        and type(requests[-1].get("scheduled_ns")) is int
        and requests[-1]["scheduled_ns"] <= traffic_stopped_ns
        and traffic_stopped_ns - requests[-1]["scheduled_ns"]
        <= interval_ns + int(config.request_timeout_seconds * 1_000_000_000)
    )

    memberships = _membership_rows(placements)
    replica_placements = _replica_placements(placements)
    membership_stream_valid = _membership_stream_valid(memberships)
    pre_rollout_memberships = [
        row
        for row in memberships
        if type(row.get("observed_ns")) is int
        and type(rollout_started_ns) is int
        and row["observed_ns"] <= rollout_started_ns
    ]
    membership_initial = (
        membership_stream_valid
        and bool(pre_rollout_memberships)
        and all(
            set(pre_rollout_memberships[-1][key]) == initial_uids
            for key in ("replica_ids", "healthy_ids", "eligible_ids")
        )
    )
    membership_final = membership_stream_valid and all(
        set(memberships[-1].get(key, ())) == final_uids
        for key in ("replica_ids", "healthy_ids", "eligible_ids")
    )

    request_by_id = {
        row.get("request_id"): row for row in requests if isinstance(row.get("request_id"), str)
    }
    request_by_echo = {
        row.get("request_id_echo"): row
        for row in requests
        if isinstance(row.get("request_id_echo"), str) and row.get("request_id_echo")
    }
    placements_by_request: dict[str, list[dict[str, Any]]] = {}
    for row in replica_placements:
        request_id = row.get("request_id")
        if isinstance(request_id, str):
            placements_by_request.setdefault(request_id, []).append(row)

    def placement_matches_request(
        request_id: str,
        request: dict[str, Any],
    ) -> bool:
        candidates = placements_by_request.get(request_id, ())
        if len(candidates) != 1:
            return False
        placement = candidates[0]
        placement_started_ns = placement.get("placement_started_ns")
        selected_at_ns = placement.get("selected_at_ns")
        request_send_ns = request.get("send_ns")
        request_end_ns = request.get("end_ns")
        return (
            placement.get("replica_id") == request.get("pod_uid")
            and type(placement_started_ns) is int
            and type(selected_at_ns) is int
            and type(request_send_ns) is int
            and type(request_end_ns) is int
            and request_send_ns <= placement_started_ns <= selected_at_ns <= request_end_ns
        )

    request_placement_join = (
        len(request_by_id) == len(requests)
        and len(request_by_echo) == len(requests)
        and set(placements_by_request) == set(request_by_echo)
        and all(
            placement_matches_request(request_id, request)
            for request_id, request in request_by_echo.items()
        )
    )
    placement_membership_join = membership_stream_valid and all(
        type(row.get("selected_at_ns")) is int
        and (
            latest := _latest_membership(
                memberships,
                row["selected_at_ns"],
            )
        )
        is not None
        and row.get("replica_id") in latest.get("eligible_ids", ())
        and row.get("replica_generation")
        == latest.get("generation_by_id", {}).get(row.get("replica_id"))
        for row in replica_placements
    )

    step_order_exact = [
        {
            "step": row.get("step"),
            "ordinal": row.get("ordinal"),
            "partition": row.get("partition"),
        }
        for row in step_rows
    ] == plan
    step_lifecycle_valid = step_order_exact
    withdrawal_events: list[dict[str, Any]] = []
    step_traffic_overlap = True
    previous_replacement_ns: int | None = None
    no_old_placement_after_withdrawal = True
    no_old_placement_after_partition = True
    for step, row in enumerate(step_rows):
        ordinal = row.get("ordinal")
        old_uid = row.get("old_uid")
        new_uid = row.get("new_uid")
        drain_started_ns = row.get("drain_started_ns")
        drain_completed_ns = row.get("drain_completed_ns")
        withdrawal_snapshot_ns = row.get("withdrawal_snapshot_observed_ns")
        withdrawn_ns = row.get("withdrawn_observed_ns")
        partition_started_ns = row.get("partition_patch_started_ns")
        partition_completed_ns = row.get("partition_patch_completed_ns")
        replacement_snapshot_ns = row.get("replacement_snapshot_observed_ns")
        replacement_ns = row.get("replacement_observed_ns")
        expected_old = initial_uid_by_ordinal.get(ordinal)
        expected_new = final_uid_by_ordinal.get(ordinal)
        expected_updated = config.replica_count - ordinal if type(ordinal) is int else None
        timestamps = (
            drain_started_ns,
            drain_completed_ns,
            withdrawal_snapshot_ns,
            withdrawn_ns,
            partition_started_ns,
            partition_completed_ns,
            replacement_snapshot_ns,
            replacement_ns,
        )
        causal = (
            all(type(value) is int for value in timestamps)
            and drain_started_ns
            <= drain_completed_ns
            <= withdrawal_snapshot_ns
            <= withdrawn_ns
            <= partition_started_ns
            <= partition_completed_ns
            <= replacement_snapshot_ns
            <= replacement_ns
            and (previous_replacement_ns is None or previous_replacement_ns <= drain_started_ns)
        )
        duration_bounded = (
            causal
            and withdrawn_ns - drain_completed_ns
            <= int(config.withdrawal_timeout_seconds * 1_000_000_000)
            and replacement_ns - partition_completed_ns
            <= int(config.replacement_timeout_seconds * 1_000_000_000)
        )
        withdrawal_matches = [
            membership
            for membership in memberships
            if membership.get("reason") == "reconciler_drain"
            and membership.get("replica_id") == old_uid
            and isinstance(membership.get("generation_by_id"), dict)
            and membership.get("replica_generation") == membership["generation_by_id"].get(old_uid)
            and isinstance(membership.get("ineligible"), list)
            and old_uid in membership["ineligible"]
            and type(membership.get("observed_ns")) is int
            and type(drain_completed_ns) is int
            and type(partition_started_ns) is int
            and drain_completed_ns <= membership["observed_ns"] <= partition_started_ns
        ]
        withdrawal_event = withdrawal_matches[0] if len(withdrawal_matches) == 1 else None
        withdrawn_pod_rows = [
            record
            for record in pods
            if record.get("capture") == "withdrawal_poll"
            and record.get("step") == step
            and record.get("ordinal") == ordinal
            and record.get("observed_ns") == withdrawal_snapshot_ns
        ]
        withdrawn_endpoint_rows = [
            record
            for record in endpoints
            if record.get("capture") == "withdrawal_poll"
            and record.get("step") == step
            and record.get("ordinal") == ordinal
            and record.get("observed_ns") == withdrawal_snapshot_ns
        ]
        replacement_pod_rows = [
            record
            for record in pods
            if record.get("capture") == "replacement_poll"
            and record.get("step") == step
            and record.get("ordinal") == ordinal
            and record.get("observed_ns") == replacement_snapshot_ns
        ]
        replacement_endpoint_rows = [
            record
            for record in endpoints
            if record.get("capture") == "replacement_poll"
            and record.get("step") == step
            and record.get("ordinal") == ordinal
            and record.get("observed_ns") == replacement_snapshot_ns
        ]
        withdrawn_pods, withdrawn_pods_valid = _replica_pod_map(
            withdrawn_pod_rows[0] if len(withdrawn_pod_rows) == 1 else None,
            statefulset=statefulset,
        )
        withdrawn_endpoints, withdrawn_endpoints_valid = _endpoint_map_from_record(
            (withdrawn_endpoint_rows[0] if len(withdrawn_endpoint_rows) == 1 else None),
            namespace=namespace,
        )
        replacement_pods, replacement_pods_valid = _replica_pod_map(
            replacement_pod_rows[0] if len(replacement_pod_rows) == 1 else None,
            statefulset=statefulset,
        )
        replacement_endpoints, replacement_endpoints_valid = _endpoint_map_from_record(
            (replacement_endpoint_rows[0] if len(replacement_endpoint_rows) == 1 else None),
            namespace=namespace,
        )
        target_name = f"{statefulset}-{ordinal}" if type(ordinal) is int else ""
        withdrawn_target = withdrawn_pods.get(ordinal)
        replacement_target = replacement_pods.get(ordinal)
        withdrawn_pod_uids = {
            identity.get("uid")
            for identity in withdrawn_pods.values()
            if isinstance(identity.get("uid"), str)
        }
        withdrawn_ready_name_to_uid = {
            identity["name"]: identity["uid"]
            for identity in withdrawn_pods.values()
            if identity.get("ready") is True
            and isinstance(identity.get("name"), str)
            and isinstance(identity.get("uid"), str)
        }
        withdrawal_gateway = row.get("gateway_at_withdrawal") or {}
        withdrawal_gateway_uids = withdrawal_gateway.get("uids")
        withdrawal_gateway_set_exact = (
            isinstance(withdrawal_gateway_uids, list)
            and all(isinstance(uid, str) for uid in withdrawal_gateway_uids)
            and frozenset(withdrawal_gateway_uids)
            in {
                frozenset(withdrawn_pod_uids),
                frozenset(withdrawn_pod_uids - {old_uid}),
            }
        )
        raw_withdrawal_valid = (
            withdrawn_pods_valid
            and len(withdrawn_pods) == config.replica_count
            and withdrawn_target is not None
            and withdrawn_target.get("uid") == old_uid
            and withdrawn_target.get("ready") is False
            and withdrawn_endpoints_valid
            and len(withdrawn_endpoints) == config.replica_count - 1
            and withdrawn_endpoints == withdrawn_ready_name_to_uid
            and withdrawn_endpoints.get(target_name) != old_uid
            and withdrawal_gateway_set_exact
            and _gateway_withdrawn(
                withdrawal_gateway,
                old_uid=old_uid if isinstance(old_uid, str) else "",
                replica_count=config.replica_count,
            )
        )
        replacement_pod_uids = {
            identity.get("uid")
            for identity in replacement_pods.values()
            if isinstance(identity.get("uid"), str)
        }
        replacement_name_to_uid = {
            identity["name"]: identity["uid"]
            for identity in replacement_pods.values()
            if isinstance(identity.get("name"), str) and isinstance(identity.get("uid"), str)
        }
        replacement_gateway = row.get("gateway_at_replacement") or {}
        replacement_gateway_uids = replacement_gateway.get("uids")
        replacement_state = row.get("statefulset") or {}
        raw_replacement_valid = (
            replacement_pods_valid
            and len(replacement_pods) == config.replica_count
            and replacement_target is not None
            and replacement_target.get("uid") == new_uid
            and replacement_target.get("ready") is True
            and replacement_target.get("revision") == update_revision
            and all(identity.get("ready") is True for identity in replacement_pods.values())
            and replacement_endpoints_valid
            and len(replacement_endpoints) == config.replica_count
            and replacement_endpoints == replacement_name_to_uid
            and replacement_endpoints.get(target_name) == new_uid
            and replacement_gateway.get("parse_errors") == 0
            and replacement_gateway.get("configured") == config.replica_count
            and replacement_gateway.get("healthy") == config.replica_count
            and replacement_gateway.get("draining") == 0
            and isinstance(replacement_gateway_uids, list)
            and all(isinstance(uid, str) for uid in replacement_gateway_uids)
            and new_uid in replacement_gateway_uids
            and old_uid not in replacement_gateway_uids
            and set(replacement_gateway_uids) == replacement_pod_uids
            and replacement_state.get("strategy") == "RollingUpdate"
            and replacement_state.get("partition") == ordinal
            and replacement_state.get("update_revision") == update_revision
            and replacement_state.get("updated_replicas") == expected_updated
            and replacement_state.get("ready_replicas") == config.replica_count
            and isinstance(replacement_state.get("generation"), int)
            and isinstance(
                replacement_state.get("observed_generation"),
                int,
            )
            and replacement_state["observed_generation"] >= replacement_state["generation"]
        )
        withdrawal_events.append(
            {
                "step": step,
                "ordinal": ordinal,
                "old_uid": old_uid,
                "event_id": (
                    withdrawal_event.get("event_id") if withdrawal_event is not None else None
                ),
                "observed_ns": (
                    withdrawal_event.get("observed_ns") if withdrawal_event is not None else None
                ),
            }
        )
        if withdrawal_event is None:
            no_old_placement_after_withdrawal = False
        else:
            no_old_placement_after_withdrawal &= not any(
                placement.get("replica_id") == old_uid
                and type(placement.get("selected_at_ns")) is int
                and placement["selected_at_ns"] >= withdrawal_event["observed_ns"]
                for placement in replica_placements
            )
        if type(partition_started_ns) is not int:
            no_old_placement_after_partition = False
        else:
            no_old_placement_after_partition &= not any(
                placement.get("replica_id") == old_uid
                and type(placement.get("selected_at_ns")) is int
                and placement["selected_at_ns"] >= partition_started_ns
                for placement in replica_placements
            )
        step_lifecycle_valid &= (
            causal
            and duration_bounded
            and old_uid == expected_old
            and new_uid == expected_new
            and isinstance(old_uid, str)
            and isinstance(new_uid, str)
            and old_uid != new_uid
            and row.get("drain_status_code") in range(200, 300)
            and row.get("drain_body_uid") == old_uid
            and row.get("drain_error") is None
            and row.get("pod_ready_false") is True
            and row.get("endpoint_absent") is True
            and row.get("gateway_withdrawn") is True
            and row.get("partition") == ordinal
            and row.get("pod_replaced") is True
            and row.get("all_pods_ready") is True
            and row.get("endpoints_exact") is True
            and row.get("gateway_exact") is True
            and row.get("statefulset_advanced") is True
            and withdrawal_event is not None
            and raw_withdrawal_valid
            and raw_replacement_valid
        )
        if all(type(value) is int for value in (drain_started_ns, replacement_ns)):
            step_traffic_overlap &= any(
                type(request.get("send_ns")) is int
                and drain_started_ns <= request["send_ns"] <= replacement_ns
                for request in requests
            )
        else:
            step_traffic_overlap = False
        if type(replacement_ns) is int:
            previous_replacement_ns = replacement_ns

    pod_snapshot_shapes = True
    minimum_ready = config.replica_count
    observed_uids_by_ordinal = {ordinal: set() for ordinal in range(config.replica_count)}
    container_restarts_zero = True
    for row in pods:
        if row.get("capture") == "gateway":
            continue
        pod_map, valid = _replica_pod_map(row, statefulset=statefulset)
        ready_count = sum(identity.get("ready") is True for identity in pod_map.values())
        minimum_ready = min(minimum_ready, ready_count)
        for ordinal, identity in pod_map.items():
            uid = identity.get("uid")
            if isinstance(uid, str) and ordinal in observed_uids_by_ordinal:
                observed_uids_by_ordinal[ordinal].add(uid)
            container_restarts_zero &= identity.get("restart_count") == 0
        pod_snapshot_shapes &= (
            valid
            and set(pod_map) <= ordinals
            and len(pod_map) >= config.replica_count - 1
            and len(pod_map) <= config.replica_count
        )
    pod_uid_lineage_exact = all(
        observed_uids_by_ordinal[ordinal]
        == {
            initial_uid_by_ordinal.get(ordinal),
            final_uid_by_ordinal.get(ordinal),
        }
        for ordinal in range(config.replica_count)
    )
    endpoint_snapshot_shapes = True
    minimum_ready_endpoints = config.replica_count
    for row in endpoints:
        endpoint_map, valid = _endpoint_map_from_record(
            row,
            namespace=namespace,
        )
        minimum_ready_endpoints = min(
            minimum_ready_endpoints,
            len(endpoint_map),
        )
        endpoint_snapshot_shapes &= (
            valid and config.replica_count - 1 <= len(endpoint_map) <= config.replica_count
        )

    restart_command = restart.get("command")
    final_command = final.get("rollout_status_command")
    restart_state = restart.get("statefulset")
    first_drain_started_ns = step_rows[0].get("drain_started_ns") if step_rows else None
    restart_time_chain_valid = (
        type(restart.get("started_ns")) is int
        and type(restart.get("completed_ns")) is int
        and type(restart.get("observed_ns")) is int
        and type(first_drain_started_ns) is int
        and restart["started_ns"]
        <= restart["completed_ns"]
        <= restart["observed_ns"]
        <= first_drain_started_ns
    )
    restart_state_valid = (
        isinstance(restart_state, dict)
        and restart_state.get("strategy") == "RollingUpdate"
        and restart_state.get("partition") == config.replica_count
        and restart_state.get("replicas") == config.replica_count
        and restart_state.get("ready_replicas") == config.replica_count
        and restart_state.get("updated_replicas") == 0
        and restart_state.get("current_revision") == initial_state.get("current_revision")
        and restart_state.get("update_revision") == update_revision
        and isinstance(restart_state.get("generation"), int)
        and isinstance(restart_state.get("observed_generation"), int)
        and restart_state["observed_generation"] >= restart_state["generation"]
    )
    frozen_sidecar_paths = {
        item.get("path")
        for item in environment.get("frozen_sidecars", ())
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    frozen_input_paths = [
        item.get("path")
        for item in environment.get("frozen_inputs", ())
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    ]
    required_frozen_inputs = {
        "verification/fleet/resilience/fleet_churn_bench.py",
        "verification/fleet/resilience/fleet_rollout_bench.py",
        "scripts/kind_rollout_gate.sh",
        ".github/workflows/f1b-rollout.yml",
        "deploy/kind/f1a/kind-config.yaml",
        "deploy/kind/f1a/base/replicas.yaml",
        "deploy/kind/f1a/base/gateway.yaml",
        "deploy/kind/f1a/mock/main.go",
        "deploy/kind/f1b/overlays/formal/kustomization.yaml",
        "deploy/kind/f1b/overlays/formal/replicas-patch.yaml",
        "deploy/kind/f1b/overlays/smoke/kustomization.yaml",
        "deploy/kind/f1b/overlays/smoke/replicas-patch.yaml",
    }
    checks = {
        "configuration_frozen": (
            config == (FORMAL_CONFIG if config.profile == "formal" else SMOKE_CONFIG)
        ),
        "rollout_rows_complete": (
            len(initial_rows) == 1
            and len(restart_rows) == 1
            and len(final_rows) == 1
            and len(step_rows) == config.replica_count
            and len(rollout) == config.replica_count + 3
        ),
        "raw_schema_versions_and_kinds_valid": (
            all(row.get("schema_version") == SCHEMA_VERSION for row in requests)
            and all(row.get("kind") in {"replica", "membership"} for row in placements)
            and all(row.get("schema_version") == SCHEMA_VERSION for row in rollout)
            and all(
                row.get("schema_version") == SCHEMA_VERSION and row.get("kind") == "pods"
                for row in pods
            )
            and all(
                row.get("schema_version") == SCHEMA_VERSION and row.get("kind") == "endpointslices"
                for row in endpoints
            )
        ),
        "initial_exact_replicas_ready": (
            initial_pods_valid
            and set(initial_pods) == ordinals
            and len(initial_uids) == config.replica_count
            and all(
                identity.get("ready") is True
                and identity.get("revision") == initial_state.get("current_revision")
                for identity in initial_pods.values()
            )
        ),
        "initial_endpoints_exact": (
            initial_endpoints_valid and initial_endpoints == initial_name_to_uid
        ),
        "initial_statefulset_partition_holds_old_fleet": (
            initial_state.get("strategy") == "RollingUpdate"
            and initial_state.get("partition") == config.replica_count
            and initial_state.get("replicas") == config.replica_count
            and initial_state.get("ready_replicas") == config.replica_count
            and initial_state.get("current_revision") == initial_state.get("update_revision")
        ),
        "initial_gateway_metrics_exact": (
            isinstance(initial.get("gateway"), dict)
            and initial["gateway"].get("parse_errors") == 0
            and initial["gateway"].get("configured") == config.replica_count
            and initial["gateway"].get("healthy") == config.replica_count
            and initial["gateway"].get("draining") == 0
            and isinstance(initial["gateway"].get("uids"), list)
            and all(isinstance(uid, str) for uid in initial["gateway"]["uids"])
            and set(initial["gateway"]["uids"]) == initial_uids
        ),
        "single_rollout_restart": (
            restart.get("returncode") == 0
            and isinstance(restart_command, list)
            and restart_command
            == [
                kubectl,
                "-n",
                namespace,
                "rollout",
                "restart",
                f"statefulset/{statefulset}",
            ]
            and isinstance(update_revision, str)
            and update_revision
            and update_revision != initial_state.get("current_revision")
            and restart.get("initial_revision") == initial_state.get("current_revision")
            and restart_time_chain_valid
            and restart_state_valid
        ),
        "step_order_exact": step_order_exact,
        "step_lifecycle_causal": step_lifecycle_valid,
        "traffic_sequence_and_schedule_exact": (
            request_sequence_coverage and request_schedule_exact and traffic_raw_coverage
        ),
        "traffic_retry_free_zero_failures": zero_failures,
        "traffic_pacing_within_bound": (pacing_fraction >= config.pacing_success_fraction),
        "traffic_encloses_rollout": traffic_encloses_rollout,
        "whole_rollout_within_deadline": (
            type(rollout_started_ns) is int
            and type(rollout_completed_ns) is int
            and rollout_completed_ns - rollout_started_ns
            <= int(config.rollout_timeout_seconds * 1_000_000_000)
        ),
        "traffic_overlaps_every_step": step_traffic_overlap,
        "request_placement_response_join_exact": request_placement_join,
        "membership_event_stream_consistent": membership_stream_valid,
        "placement_membership_temporal_join_complete": (placement_membership_join),
        "gateway_initial_membership_all_eligible": membership_initial,
        "gateway_final_membership_all_eligible": membership_final,
        "no_old_placement_after_gateway_withdrawal": (no_old_placement_after_withdrawal),
        "no_old_placement_after_partition_patch": (no_old_placement_after_partition),
        "pod_snapshots_valid_and_availability_floor_n_minus_one": (
            pod_snapshot_shapes and minimum_ready >= config.replica_count - 1
        ),
        "pod_uid_lineage_exactly_one_replacement_each": (pod_uid_lineage_exact),
        "replica_container_restart_count_zero": container_restarts_zero,
        "endpoint_snapshots_valid_and_floor_n_minus_one": (
            endpoint_snapshot_shapes and minimum_ready_endpoints >= config.replica_count - 1
        ),
        "final_exact_replacement_fleet": (
            final_pods_valid
            and set(final_pods) == ordinals
            and len(final_uids) == config.replica_count
            and initial_uids.isdisjoint(final_uids)
            and all(
                identity.get("ready") is True and identity.get("revision") == update_revision
                for identity in final_pods.values()
            )
        ),
        "final_endpoints_exact": (final_endpoints_valid and final_endpoints == final_name_to_uid),
        "final_statefulset_revision_complete": (
            final_state.get("strategy") == "RollingUpdate"
            and final_state.get("partition") == 0
            and final_state.get("replicas") == config.replica_count
            and final_state.get("ready_replicas") == config.replica_count
            and final_state.get("updated_replicas") == config.replica_count
            and final_state.get("current_revision") == update_revision
            and final_state.get("update_revision") == update_revision
        ),
        "final_gateway_metrics_exact": (
            isinstance(final.get("gateway"), dict)
            and final["gateway"].get("parse_errors") == 0
            and final["gateway"].get("configured") == config.replica_count
            and final["gateway"].get("healthy") == config.replica_count
            and final["gateway"].get("draining") == 0
            and isinstance(final["gateway"].get("uids"), list)
            and all(isinstance(uid, str) for uid in final["gateway"]["uids"])
            and set(final["gateway"]["uids"]) == final_uids
        ),
        "kubectl_rollout_status_succeeded": (
            final.get("rollout_status_returncode") == 0
            and isinstance(final_command, list)
            and len(final_command) >= 6
            and final_command[:5]
            == [
                kubectl,
                "-n",
                namespace,
                "rollout",
                "status",
            ]
            and final_command[5] == f"statefulset/{statefulset}"
        ),
        "source_and_image_provenance_valid": _provenance_valid(environment),
        "frozen_runtime_manifests_exact": (
            frozen_sidecar_paths
            == {
                "rendered-kind-config.yaml",
                "rendered-manifest.yaml",
            }
        ),
        "critical_frozen_source_inventory_complete": (
            len(frozen_input_paths) == len(set(frozen_input_paths))
            and all(
                any(
                    path == required or path.endswith(f"/{required}") for path in frozen_input_paths
                )
                for required in required_frozen_inputs
            )
        ),
        "source_immutable_for_entire_run": (
            environment.get("source_dirty") is False
            and environment.get("tracked_dirty") is False
            and environment.get("untracked_gate_input_dirty") is False
            and environment.get("source_end_match") is True
        ),
        "mock_runtime_images_pinned": _runtime_images_valid(
            pods,
            environment=environment,
            capture="final",
            container_name=mock_container,
            expected_image=mock_image,
            expected_count=config.replica_count,
        ),
        "all_replica_lifecycle_images_pinned": (
            _all_replica_runtime_images_valid(
                pods,
                environment=environment,
                container_name=mock_container,
                expected_image=mock_image,
            )
        ),
        "gateway_runtime_image_pinned": _runtime_images_valid(
            pods,
            environment=environment,
            capture="gateway",
            container_name=gateway_container,
            expected_image=gateway_image,
            expected_count=1,
        ),
    }
    traffic = {
        "requests": len(requests),
        "responses_2xx": sum(
            isinstance(row.get("status_code"), int) and 200 <= row["status_code"] < 300
            for row in requests
        ),
        "http_429": sum(row.get("status_code") == 429 for row in requests),
        "http_5xx": sum(
            isinstance(row.get("status_code"), int) and 500 <= row["status_code"] < 600
            for row in requests
        ),
        "other_http": sum(
            isinstance(row.get("status_code"), int)
            and not 200 <= row["status_code"] < 300
            and row.get("status_code") != 429
            and not 500 <= row["status_code"] < 600
            for row in requests
        ),
        "transport_errors": sum(row.get("transport_error") is not None for row in requests),
        "unsent": sum(row.get("not_sent") is True for row in requests),
        "pacing_success_fraction": pacing_fraction,
    }
    rollout_summary = {
        "replicas": config.replica_count,
        "steps": len(step_rows),
        "old_uids": sorted(initial_uids),
        "new_uids": sorted(final_uids),
        "initial_revision": initial_state.get("current_revision"),
        "update_revision": update_revision,
        "final_current_revision": final_state.get("current_revision"),
        "minimum_ready_pods_observed": minimum_ready,
        "minimum_ready_endpoints_observed": minimum_ready_endpoints,
        "withdrawals": withdrawal_events,
        "operator_interventions": 0,
        "rollout_restart_commands": len(restart_rows),
        "direct_pod_delete_commands": 0,
    }
    membership_summary = {
        "events": len(memberships),
        "placements": len(replica_placements),
        "initial_uids": sorted(initial_uids),
        "final_uids": sorted(final_uids),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "traffic": traffic,
        "rollout": rollout_summary,
        "membership": membership_summary,
        "runtime_mock_image": _runtime_mock_image_summary(
            pods,
            mock_container,
        ),
        "runtime_gateway_image": _runtime_mock_image_summary(
            pods,
            gateway_container,
        ),
    }


async def _wait_for_update_revision(
    args: argparse.Namespace,
    *,
    old_revision: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            async with asyncio.timeout(remaining):
                last = _statefulset_identity(await _read_statefulset(args))
        except TimeoutError:
            break
        if (
            isinstance(last.get("update_revision"), str)
            and last["update_revision"] != old_revision
            and last.get("current_revision") == old_revision
            and last.get("partition") == last.get("replicas")
            and isinstance(last.get("generation"), int)
            and isinstance(last.get("observed_generation"), int)
            and last["observed_generation"] >= last["generation"]
        ):
            return last
        remaining = deadline - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(min(_POLL_SECONDS, remaining))
    raise TimeoutError(f"StatefulSet update revision did not appear: {last}")


async def run(
    args: argparse.Namespace,
    config: RolloutGateConfig,
) -> dict[str, Any]:
    config.validate()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        name: output.parent / filename
        for name, filename in (
            ("requests", "requests.jsonl"),
            ("placements", "placements.jsonl"),
            ("rollout", "rollout.jsonl"),
            ("pods", "pods.jsonl"),
            ("endpointslices", "endpointslices.jsonl"),
        )
    }
    sinks = {name: JsonlSink(path) for name, path in paths.items()}
    stop_traffic = asyncio.Event()
    traffic_task: asyncio.Task[list[dict[str, Any]]] | None = None
    source_start = source_environment(args)
    try:
        async with httpx.AsyncClient(
            base_url=args.gateway_url.rstrip("/"),
            timeout=httpx.Timeout(config.request_timeout_seconds),
            trust_env=False,
        ) as gateway_client:
            initial_pods_raw, initial_endpoints_raw, initial_observed_ns = await _capture_inventory(
                args,
                pod_sink=sinks["pods"],
                endpoint_sink=sinks["endpointslices"],
                capture="initial",
            )
            initial_state = _statefulset_identity(await _read_statefulset(args))
            initial_gateway = await _read_gateway_metrics(
                gateway_client,
                pool=args.model,
            )
            initial_identities = [
                _rollout_pod_identity(item)
                for item in initial_pods_raw.get("items", ())
                if isinstance(item, dict)
            ]
            initial_by_ordinal, initial_valid = _pod_map(
                initial_identities,
                statefulset=args.statefulset,
            )
            initial_endpoint_map, initial_endpoints_valid = _endpoint_targets(
                initial_endpoints_raw,
                namespace=args.namespace,
            )
            initial_uids = {
                identity.get("uid")
                for identity in initial_by_ordinal.values()
                if isinstance(identity.get("uid"), str)
            }
            initial_name_to_uid = {
                identity["name"]: identity["uid"]
                for identity in initial_by_ordinal.values()
                if isinstance(identity.get("name"), str) and isinstance(identity.get("uid"), str)
            }
            initial_gateway_uids = initial_gateway.get("uids")
            if not (
                initial_valid
                and initial_endpoints_valid
                and set(initial_by_ordinal) == set(range(config.replica_count))
                and len(initial_uids) == config.replica_count
                and all(identity.get("ready") is True for identity in initial_by_ordinal.values())
                and initial_endpoint_map == initial_name_to_uid
                and all(
                    identity.get("revision") == initial_state.get("current_revision")
                    for identity in initial_by_ordinal.values()
                )
                and initial_state.get("strategy") == "RollingUpdate"
                and initial_state.get("partition") == config.replica_count
                and initial_state.get("ready_replicas") == config.replica_count
                and initial_state.get("current_revision") == initial_state.get("update_revision")
                and initial_gateway.get("configured") == config.replica_count
                and initial_gateway.get("healthy") == config.replica_count
                and initial_gateway.get("draining") == 0
                and isinstance(initial_gateway_uids, list)
                and all(isinstance(uid, str) for uid in initial_gateway_uids)
                and set(initial_gateway_uids) == initial_uids
            ):
                raise RuntimeError(
                    "initial F1b inventory did not converge exactly: "
                    f"pods={len(initial_by_ordinal)} "
                    f"endpoints={len(initial_endpoint_map)} "
                    f"statefulset={initial_state} gateway={initial_gateway}"
                )

            traffic_started_ns = time.monotonic_ns()
            initial_record = {
                "schema_version": SCHEMA_VERSION,
                "kind": "initial",
                "observed_ns": initial_observed_ns,
                "traffic_started_ns": traffic_started_ns,
                "statefulset": initial_state,
                "gateway": initial_gateway,
                "old_uid_by_ordinal": {
                    str(ordinal): identity["uid"]
                    for ordinal, identity in sorted(initial_by_ordinal.items())
                },
            }
            sinks["rollout"].write(initial_record)
            traffic_task = asyncio.create_task(
                _traffic_loop(
                    args,
                    config,
                    sinks["requests"],
                    stop_traffic,
                    traffic_started_ns,
                )
            )
            await asyncio.sleep(config.warmup_seconds)

            restart_command = [
                args.kubectl,
                "-n",
                args.namespace,
                "rollout",
                "restart",
                f"statefulset/{args.statefulset}",
            ]
            restart_started_ns = time.monotonic_ns()
            rollout_deadline = restart_started_ns / 1_000_000_000 + config.rollout_timeout_seconds
            async with asyncio.timeout_at(rollout_deadline):
                returncode, stdout, stderr = await _run_command(
                    *restart_command,
                    check=False,
                )
            restart_completed_ns = time.monotonic_ns()
            if returncode != 0:
                raise RuntimeError(f"rollout restart failed ({returncode}): {stderr[:500]}")
            old_revision = initial_state.get("current_revision")
            if not isinstance(old_revision, str) or not old_revision:
                raise RuntimeError("initial StatefulSet revision is missing")
            async with asyncio.timeout_at(rollout_deadline):
                restart_state = await _wait_for_update_revision(
                    args,
                    old_revision=old_revision,
                    timeout_seconds=config.replacement_timeout_seconds,
                )
            update_revision = restart_state.get("update_revision")
            assert isinstance(update_revision, str)
            sinks["rollout"].write(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "restart",
                    "started_ns": restart_started_ns,
                    "completed_ns": restart_completed_ns,
                    "observed_ns": time.monotonic_ns(),
                    "command": restart_command,
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "initial_revision": old_revision,
                    "update_revision": update_revision,
                    "statefulset": restart_state,
                }
            )

            current_by_ordinal = dict(initial_by_ordinal)
            for planned in rollout_plan(config):
                if time.monotonic() >= rollout_deadline:
                    raise TimeoutError("whole F1b rollout exceeded its deadline")
                step = planned["step"]
                ordinal = planned["ordinal"]
                old_identity = current_by_ordinal[ordinal]
                old_uid = old_identity.get("uid")
                pod_name = old_identity.get("name")
                if not isinstance(old_uid, str) or not isinstance(pod_name, str):
                    raise RuntimeError(
                        f"invalid old identity for ordinal {ordinal}: {old_identity}"
                    )

                drain_started_ns = time.monotonic_ns()
                async with asyncio.timeout_at(rollout_deadline):
                    drain_status, drain_body, drain_text = await _drain_pod(
                        args,
                        pod_name,
                    )
                drain_completed_ns = time.monotonic_ns()
                drain_body_uid = drain_body.get("pod_uid") if isinstance(drain_body, dict) else None
                drain_error = None
                if not 200 <= drain_status < 300:
                    drain_error = f"HTTP {drain_status}: {drain_text[:500]}"
                elif drain_body_uid != old_uid:
                    drain_error = f"drain UID {drain_body_uid!r} != {old_uid!r}"
                if drain_error is not None:
                    raise RuntimeError(drain_error)

                async with asyncio.timeout_at(rollout_deadline):
                    withdrawn = await _wait_withdrawn(
                        args,
                        config,
                        gateway_client=gateway_client,
                        pod_sink=sinks["pods"],
                        endpoint_sink=sinks["endpointslices"],
                        step=step,
                        ordinal=ordinal,
                        old_uid=old_uid,
                    )
                partition_started_ns = time.monotonic_ns()
                async with asyncio.timeout_at(rollout_deadline):
                    partition_payload = await _patch_partition(args, ordinal)
                partition_completed_ns = time.monotonic_ns()
                patched_state = _statefulset_identity(partition_payload)
                if patched_state.get("partition") != ordinal:
                    raise RuntimeError(f"partition PATCH did not set {ordinal}: {patched_state}")
                async with asyncio.timeout_at(rollout_deadline):
                    replacement = await _wait_replacement(
                        args,
                        config,
                        gateway_client=gateway_client,
                        pod_sink=sinks["pods"],
                        endpoint_sink=sinks["endpointslices"],
                        step=step,
                        ordinal=ordinal,
                        old_uid=old_uid,
                        update_revision=update_revision,
                    )
                if time.monotonic() > rollout_deadline:
                    raise TimeoutError("whole F1b rollout exceeded its deadline")
                new_uid = replacement.get("new_uid")
                if not isinstance(new_uid, str):
                    raise RuntimeError(f"replacement UID missing for ordinal {ordinal}")
                current_by_ordinal[ordinal] = {
                    **old_identity,
                    "uid": new_uid,
                    "ready": True,
                    "revision": update_revision,
                }
                sinks["rollout"].write(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "step",
                        "step": step,
                        "ordinal": ordinal,
                        "old_uid": old_uid,
                        "new_uid": new_uid,
                        "drain_started_ns": drain_started_ns,
                        "drain_completed_ns": drain_completed_ns,
                        "drain_status_code": drain_status,
                        "drain_body_uid": drain_body_uid,
                        "drain_error": drain_error,
                        "withdrawal_snapshot_observed_ns": (withdrawn["snapshot_observed_ns"]),
                        "withdrawn_observed_ns": withdrawn["observed_ns"],
                        "pod_ready_false": withdrawn["pod_ready_false"],
                        "endpoint_absent": withdrawn["endpoint_absent"],
                        "gateway_withdrawn": withdrawn["gateway_withdrawn"],
                        "gateway_at_withdrawal": withdrawn["gateway"],
                        "partition_patch_started_ns": partition_started_ns,
                        "partition_patch_completed_ns": (partition_completed_ns),
                        "partition": ordinal,
                        "replacement_snapshot_observed_ns": (replacement["snapshot_observed_ns"]),
                        "replacement_observed_ns": replacement["observed_ns"],
                        "pod_replaced": replacement["pod_replaced"],
                        "all_pods_ready": replacement["all_pods_ready"],
                        "endpoints_exact": replacement["endpoints_exact"],
                        "gateway_exact": replacement["gateway_exact"],
                        "statefulset_advanced": (replacement["statefulset_advanced"]),
                        "gateway_at_replacement": replacement["gateway"],
                        "statefulset": replacement["statefulset"],
                    }
                )

            rollout_status_command = [
                args.kubectl,
                "-n",
                args.namespace,
                "rollout",
                "status",
                f"statefulset/{args.statefulset}",
                f"--timeout={int(config.replacement_timeout_seconds)}s",
            ]
            async with asyncio.timeout_at(rollout_deadline):
                (
                    rollout_status_returncode,
                    rollout_status_stdout,
                    rollout_status_stderr,
                ) = await _run_command(*rollout_status_command, check=False)
            rollout_completed_ns = time.monotonic_ns()
            if rollout_status_returncode != 0:
                raise RuntimeError(
                    "rollout status failed "
                    f"({rollout_status_returncode}): "
                    f"{rollout_status_stderr[:500]}"
                )

            await asyncio.sleep(config.cooldown_seconds)
            (
                final_pods_raw,
                final_endpoints_raw,
                final_observed_ns,
            ) = await _capture_inventory(
                args,
                pod_sink=sinks["pods"],
                endpoint_sink=sinks["endpointslices"],
                capture="final",
            )
            final_state = _statefulset_identity(await _read_statefulset(args))
            final_gateway = await _read_gateway_metrics(
                gateway_client,
                pool=args.model,
            )
            gateway_pods_raw = await _read_pods(
                args,
                selector=args.gateway_pod_selector,
            )
            gateway_pod_record = _pod_record(
                capture="gateway",
                observed_ns=time.monotonic_ns(),
                snapshot=gateway_pods_raw,
            )
            sinks["pods"].write(gateway_pod_record)

            traffic_stop_requested_ns = time.monotonic_ns()
            stop_traffic.set()
            requests = await traffic_task
            traffic_task = None
            traffic_stopped_ns = time.monotonic_ns()
            final_identities = [
                _rollout_pod_identity(item)
                for item in final_pods_raw.get("items", ())
                if isinstance(item, dict)
            ]
            final_by_ordinal, final_valid = _pod_map(
                final_identities,
                statefulset=args.statefulset,
            )
            final_endpoint_map, final_endpoints_valid = _endpoint_targets(
                final_endpoints_raw,
                namespace=args.namespace,
            )
            final_uids = {
                identity.get("uid")
                for identity in final_by_ordinal.values()
                if isinstance(identity.get("uid"), str)
            }
            final_name_to_uid = {
                identity["name"]: identity["uid"]
                for identity in final_by_ordinal.values()
                if isinstance(identity.get("name"), str) and isinstance(identity.get("uid"), str)
            }
            final_gateway_uids = final_gateway.get("uids")
            if not (
                final_valid
                and final_endpoints_valid
                and set(final_by_ordinal) == set(range(config.replica_count))
                and len(final_uids) == config.replica_count
                and initial_uids.isdisjoint(final_uids)
                and all(
                    identity.get("ready") is True and identity.get("revision") == update_revision
                    for identity in final_by_ordinal.values()
                )
                and final_endpoint_map == final_name_to_uid
                and final_state.get("partition") == 0
                and final_state.get("current_revision") == update_revision
                and final_state.get("update_revision") == update_revision
                and final_state.get("ready_replicas") == config.replica_count
                and final_gateway.get("configured") == config.replica_count
                and final_gateway.get("healthy") == config.replica_count
                and final_gateway.get("draining") == 0
                and isinstance(final_gateway_uids, list)
                and all(isinstance(uid, str) for uid in final_gateway_uids)
                and set(final_gateway_uids) == final_uids
            ):
                raise RuntimeError(
                    "final F1b inventory did not converge exactly: "
                    f"pods={len(final_by_ordinal)} "
                    f"endpoints={len(final_endpoint_map)} "
                    f"statefulset={final_state} gateway={final_gateway}"
                )
            sinks["rollout"].write(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "final",
                    "observed_ns": final_observed_ns,
                    "rollout_completed_ns": rollout_completed_ns,
                    "traffic_stop_requested_ns": (traffic_stop_requested_ns),
                    "traffic_stopped_ns": traffic_stopped_ns,
                    "rollout_status_command": rollout_status_command,
                    "rollout_status_returncode": rollout_status_returncode,
                    "rollout_status_stdout": rollout_status_stdout,
                    "rollout_status_stderr": rollout_status_stderr,
                    "statefulset": final_state,
                    "gateway": final_gateway,
                    "new_uid_by_ordinal": {
                        str(ordinal): identity["uid"]
                        for ordinal, identity in sorted(final_by_ordinal.items())
                    },
                }
            )
            await _fetch_gateway_placements(
                args,
                expected_request_ids={
                    row["request_id_echo"]
                    for row in requests
                    if isinstance(row.get("request_id_echo"), str) and row["request_id_echo"]
                },
                initial_uids=initial_uids,
                final_uids=final_uids,
                sink=sinks["placements"],
            )
    finally:
        stop_traffic.set()
        if traffic_task is not None:
            traffic_task.cancel()
            await asyncio.gather(traffic_task, return_exceptions=True)
        for sink in sinks.values():
            sink.close()

    loaded = {name: _load_jsonl(path) for name, path in paths.items()}
    source_end = source_environment(args)
    environment = dict(source_start)
    environment.update(
        {
            "source_end_match": source_end == source_start,
            "namespace": args.namespace,
            "statefulset": args.statefulset,
            "mock_container_name": args.mock_container,
            "expected_mock_image": args.mock_image,
            "runtime_mock_image": _runtime_mock_image_summary(
                loaded["pods"],
                args.mock_container,
            ),
            "gateway_container_name": args.gateway_container,
            "expected_gateway_image": args.gateway_image,
            "runtime_gateway_image": _runtime_mock_image_summary(
                loaded["pods"],
                args.gateway_container,
            ),
        }
    )
    replay = evaluate_gate(
        config=config,
        requests=loaded["requests"],
        placements=loaded["placements"],
        rollout=loaded["rollout"],
        pods=loaded["pods"],
        endpoints=loaded["endpointslices"],
        environment=environment,
        statefulset=args.statefulset,
        namespace=args.namespace,
        mock_container=args.mock_container,
        mock_image=args.mock_image,
        gateway_container=args.gateway_container,
        gateway_image=args.gateway_image,
        kubectl=args.kubectl,
    )
    sidecars = {name: _sidecar_descriptor(paths[name], loaded[name]) for name in paths}
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "methodology": {
            "traffic": (
                "absolute-deadline open-loop requests; retry=0; every non-2xx, "
                "transport error, or unsent arrival fails"
            ),
            "rollout": (
                "one kubectl rollout restart held at partition=N, followed by "
                "automatic ordinal N-1..0 drain-withdraw-partition-replace "
                "steps"
            ),
            "drain_boundary": (
                "gateway reconciler_drain membership event; new placement to "
                "that UID is forbidden at and after the event while earlier "
                "in-flight requests must finish successfully"
            ),
            "identity": (
                "StatefulSet ordinal, Pod UID, EndpointSlice targetRef.uid, "
                "gateway placement replica_id, and response Pod UID are joined"
            ),
            "availability": (
                "every observed Pod and EndpointSlice snapshot retains at "
                "least N-1 Ready identities; each step returns to exact N "
                "before the next begins"
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
            "gateway_container": args.gateway_container,
            "gateway_image": args.gateway_image,
            "request_id_header": args.request_id_header,
            "kubectl": args.kubectl,
        },
        "configuration": asdict(config),
        "schedule": rollout_plan(config),
        "environment": environment,
        "sidecars": sidecars,
        **replay,
    }


async def _run_with_kubernetes_proxy(
    args: argparse.Namespace,
    config: RolloutGateConfig,
) -> dict[str, Any]:
    missing = object()
    previous_reader = getattr(args, "_kubernetes_proxy_reader", missing)
    async with RolloutKubernetesProxy(args.kubectl) as reader:
        args._kubernetes_proxy_reader = reader
        try:
            return await run(args, config)
        finally:
            if previous_reader is missing:
                del args._kubernetes_proxy_reader
            else:
                args._kubernetes_proxy_reader = previous_reader


def verify_artifact(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("artifact manifest must be a JSON object")
    checks: dict[str, bool] = {
        "schema_version": manifest.get("schema_version") == SCHEMA_VERSION,
        "gate": manifest.get("gate") == GATE,
    }
    raw_environment = manifest.get("environment")
    environment = raw_environment if isinstance(raw_environment, dict) else {}
    checks["environment:shape"] = isinstance(raw_environment, dict)
    frozen_sidecars = environment.get("frozen_sidecars")
    frozen_valid = (
        isinstance(frozen_sidecars, list)
        and bool(frozen_sidecars)
        and all(isinstance(item, dict) for item in frozen_sidecars)
    )
    checks["frozen_sidecars:descriptor"] = frozen_valid
    if frozen_valid:
        for index, descriptor in enumerate(frozen_sidecars):
            try:
                path = _safe_sidecar(
                    manifest_path,
                    str(descriptor.get("path", "")),
                )
            except (OSError, TypeError, ValueError):
                checks[f"frozen_sidecar:{index}:safe"] = False
                continue
            checks[f"frozen_sidecar:{index}:safe"] = True
            exists = path.is_file()
            checks[f"frozen_sidecar:{index}:exists"] = exists
            checks[f"frozen_sidecar:{index}:hash"] = exists and _sha256_file(
                path
            ) == descriptor.get("sha256")
    for role in ("gateway", "mock"):
        try:
            checks[f"{role}_containerd_image_metadata_raw"] = _containerd_metadata_file_matches(
                manifest_path,
                environment.get(f"{role}_containerd_image_metadata"),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            checks[f"{role}_containerd_image_metadata_raw"] = False
        try:
            checks[f"{role}_cri_image_metadata_raw"] = _cri_metadata_file_matches(
                manifest_path,
                environment.get(f"{role}_cri_image_metadata"),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            checks[f"{role}_cri_image_metadata_raw"] = False

    raw_sidecars = manifest.get("sidecars")
    sidecars = raw_sidecars if isinstance(raw_sidecars, dict) else {}
    checks["sidecars:shape"] = isinstance(raw_sidecars, dict)
    expected_sidecars = {
        "requests",
        "placements",
        "rollout",
        "pods",
        "endpointslices",
    }
    checks["sidecars:exact_keys"] = (
        isinstance(raw_sidecars, dict) and set(raw_sidecars) == expected_sidecars
    )
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(expected_sidecars):
        descriptor = sidecars.get(name)
        if not isinstance(descriptor, dict):
            checks[f"sidecar:{name}:descriptor"] = False
            loaded[name] = []
            continue
        try:
            path = _safe_sidecar(
                manifest_path,
                str(descriptor.get("path", "")),
            )
        except (OSError, TypeError, ValueError):
            checks[f"sidecar:{name}:safe"] = False
            loaded[name] = []
            continue
        checks[f"sidecar:{name}:safe"] = True
        exists = path.is_file()
        checks[f"sidecar:{name}:exists"] = exists
        if not exists:
            loaded[name] = []
            continue
        try:
            rows = _load_jsonl(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            checks[f"sidecar:{name}:parse"] = False
            loaded[name] = []
            continue
        loaded[name] = rows
        checks[f"sidecar:{name}:hash"] = _sha256_file(path) == descriptor.get("sha256")
        checks[f"sidecar:{name}:rows"] = len(rows) == descriptor.get("rows")
        checks[f"sidecar:{name}:descriptor_exact"] = descriptor == _sidecar_descriptor(path, rows)

    raw_assumptions = manifest.get("assumptions")
    assumptions = raw_assumptions if isinstance(raw_assumptions, dict) else {}
    checks["assumptions:shape"] = isinstance(raw_assumptions, dict)
    try:
        config = RolloutGateConfig(**manifest["configuration"])
        checks["published_schedule_matches_frozen_plan"] = manifest.get("schedule") == rollout_plan(
            config
        )
        replay = evaluate_gate(
            config=config,
            requests=loaded["requests"],
            placements=loaded["placements"],
            rollout=loaded["rollout"],
            pods=loaded["pods"],
            endpoints=loaded["endpointslices"],
            environment=environment,
            statefulset=assumptions.get("statefulset", ""),
            namespace=assumptions.get("namespace", ""),
            mock_container=assumptions.get("mock_container", ""),
            mock_image=assumptions.get("mock_image", ""),
            gateway_container=assumptions.get("gateway_container", ""),
            gateway_image=assumptions.get("gateway_image", ""),
            kubectl=assumptions.get("kubectl", ""),
        )
    except Exception as error:
        checks["published_schedule_matches_frozen_plan"] = False
        replay = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    checks["replayed_gate_passed"] = replay.get("passed") is True
    for key in (
        "checks",
        "passed",
        "traffic",
        "rollout",
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
    for field in fields(RolloutGateConfig):
        if field.name == "profile":
            continue
        option = "--" + field.name.replace("_", "-")
        value_type = type(getattr(FORMAL_CONFIG, field.name))
        parser.add_argument(option, type=value_type, default=None)
    parser.add_argument("--frozen-input", action="append", default=[])
    parser.add_argument("--frozen-sidecar", action="append", default=[])
    parser.add_argument("--gateway-image-digest")
    parser.add_argument("--mock-image-digest")
    parser.add_argument("--gateway-image-config-digest")
    parser.add_argument("--mock-image-config-digest")
    parser.add_argument("--gateway-containerd-image-digest")
    parser.add_argument("--mock-containerd-image-digest")
    parser.add_argument("--gateway-containerd-config-digest")
    parser.add_argument("--mock-containerd-config-digest")
    parser.add_argument("--gateway-containerd-image-metadata")
    parser.add_argument("--mock-containerd-image-metadata")
    parser.add_argument("--gateway-cri-image-metadata")
    parser.add_argument("--mock-cri-image-metadata")
    parser.add_argument("--kind-node-image")
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--kind-version")
    parser.add_argument("--kubectl-version")
    parser.add_argument("--docker-version")
    parser.add_argument(
        "--driver-path",
        default="scripts/kind_rollout_gate.sh",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bench/results/f1b-fleet-rollout/manifest.json"),
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
                    "schedule": rollout_plan(config),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = asyncio.run(_run_with_kubernetes_proxy(args, config))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
    print(rendered)
    if args.assert_gate and not result["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
