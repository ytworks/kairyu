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
from urllib.parse import quote

import httpx

from kairyu.audit_io import AuditQueueFull, BoundedJsonlWriter
from kairyu.bench.evidence import sha256_file as _sha256_file

SCHEMA_VERSION = 1
GATE = "G5-F1a"
_COMMAND_TERMINATE_GRACE_SECONDS = 2.0
_KUBECTL_PROXY_START_TIMEOUT_SECONDS = 10.0
_KUBERNETES_READ_TIMEOUT_SECONDS = 10.0
_KUBERNETES_DELETE_MAX_CONCURRENCY = 8
_ENDPOINT_WITHDRAWAL_POLL_SECONDS = 0.25
_ENDPOINT_WITHDRAWAL_CAUSAL_BRACKET_SECONDS = 1.0
_RECOVERY_POLL_SECONDS = 1.0
_KUBERNETES_OBSERVATION_TRANSPORT = (
    "one lifecycle-owned kubectl proxy and persistent HTTP client carry all "
    "EndpointSlice, Pod, resource, and Pod DELETE requests; DELETE uses "
    "bounded concurrency, background propagation, and the Pod's declared "
    "grace period"
)
_KUBECTL_PROXY_ADDRESS = re.compile(
    r"Starting to serve on 127\.0\.0\.1:(?P<port>[0-9]+)"
)


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
    gateway_membership_withdrawal_limit_seconds: float = 1.0

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
            self.gateway_membership_withdrawal_limit_seconds,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("gate duration/rate/limit fields must be positive")
        if self.warmup_seconds < 0 or self.cooldown_seconds < 0:
            raise ValueError("warmup/cooldown must be non-negative")
        if not 0 < self.pacing_success_fraction <= 1:
            raise ValueError("pacing_success_fraction must be in (0, 1]")
        if self.churn_window_seconds > self.epoch_seconds:
            raise ValueError("churn_window_seconds must not exceed epoch_seconds")
        if (
            self.gateway_membership_withdrawal_limit_seconds
            > self.endpoint_withdrawal_limit_seconds
        ):
            raise ValueError(
                "gateway membership withdrawal limit must not exceed the "
                "endpoint withdrawal limit"
            )


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
    gateway_membership_withdrawal_limit_seconds=1.0,
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
    gateway_membership_withdrawal_limit_seconds=1.0,
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


def _cri_image_metadata(path: Path, artifact_dir: Path) -> dict[str, Any]:
    path = path.resolve()
    artifact_dir = artifact_dir.resolve()
    if path.parent != artifact_dir:
        raise ValueError(
            f"CRI image metadata must be beside the manifest: {path}"
        )
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"CRI image metadata must be an object: {path}")
    status = payload.get("status")
    info = payload.get("info")
    image_spec = info.get("imageSpec") if isinstance(info, dict) else None
    if not isinstance(status, dict) or not isinstance(image_spec, dict):
        raise ValueError(f"incomplete CRI image metadata: {path}")
    config = image_spec.get("config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    if not isinstance(labels, dict):
        raise ValueError(f"CRI image labels are missing: {path}")

    digest_pattern = re.compile(r"sha256:[0-9a-f]{64}")

    def exact_digest(value: Any) -> str:
        if not isinstance(value, str) or digest_pattern.fullmatch(value) is None:
            raise ValueError(f"invalid CRI image digest in {path}: {value!r}")
        return value

    status_id = exact_digest(status.get("id"))
    raw_repo_digests = status.get("repoDigests")
    raw_repo_tags = status.get("repoTags")
    if not isinstance(raw_repo_digests, list) or not all(
        isinstance(value, str) for value in raw_repo_digests
    ):
        raise ValueError(f"invalid CRI repoDigests in {path}")
    if not isinstance(raw_repo_tags, list) or not raw_repo_tags or not all(
        isinstance(value, str) and value for value in raw_repo_tags
    ):
        raise ValueError(f"invalid CRI repoTags in {path}")
    repo_digests = []
    for value in raw_repo_digests:
        _, separator, digest = value.rpartition("@")
        if not separator:
            raise ValueError(f"invalid CRI repoDigest in {path}: {value!r}")
        repo_digests.append(exact_digest(digest))
    revision = labels.get("org.opencontainers.image.revision")
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "status_id": status_id,
        "repo_digests": sorted(raw_repo_digests),
        "repo_tags": sorted(raw_repo_tags),
        "runtime_digests": sorted({status_id, *repo_digests}),
        "revision": revision,
    }


def _containerd_image_metadata(
    path: Path,
    artifact_dir: Path,
    target_digest: str,
) -> dict[str, Any]:
    path = path.resolve()
    artifact_dir = artifact_dir.resolve()
    if path.parent != artifact_dir:
        raise ValueError(
            f"containerd manifest must be beside the artifact: {path}"
        )
    digest_pattern = re.compile(r"sha256:[0-9a-f]{64}")
    if digest_pattern.fullmatch(target_digest) is None:
        raise ValueError(f"invalid containerd target digest: {target_digest}")
    raw_sha256 = _sha256_file(path)
    if f"sha256:{raw_sha256}" != target_digest:
        raise ValueError(
            f"containerd manifest hash does not match target: {path}"
        )
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict) or payload.get("schemaVersion") != 2:
        raise ValueError(f"invalid containerd image manifest: {path}")
    allowed_media_types = {
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    }
    media_type = payload.get("mediaType")
    if media_type not in allowed_media_types:
        raise ValueError(
            f"unsupported containerd manifest media type: {media_type!r}"
        )
    config = payload.get("config")
    layers = payload.get("layers")
    if not isinstance(config, dict) or not isinstance(layers, list) or not layers:
        raise ValueError(f"incomplete containerd image manifest: {path}")
    config_digest = config.get("digest")
    layer_digests = [
        layer.get("digest") if isinstance(layer, dict) else None
        for layer in layers
    ]
    if (
        not isinstance(config_digest, str)
        or digest_pattern.fullmatch(config_digest) is None
        or not all(
            isinstance(digest, str)
            and digest_pattern.fullmatch(digest) is not None
            for digest in layer_digests
        )
    ):
        raise ValueError(f"invalid containerd content digest: {path}")
    return {
        "path": path.name,
        "sha256": raw_sha256,
        "target_digest": target_digest,
        "media_type": media_type,
        "config_digest": config_digest,
        "layer_digests": layer_digests,
    }


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
    artifact_dir = Path(args.output).resolve().parent
    frozen_inputs = []
    for raw in args.frozen_input:
        path = Path(raw).resolve()
        frozen_inputs.append(
            {"path": str(path), "sha256": _sha256_file(path)}
        )
    frozen_sidecars = []
    for raw in getattr(args, "frozen_sidecar", ()):
        path = Path(raw).resolve()
        try:
            relative = path.relative_to(artifact_dir)
        except ValueError as error:
            raise ValueError(
                f"frozen sidecar must be inside {artifact_dir}: {path}"
            ) from error
        relative_path = relative.as_posix()
        if _safe_sidecar(Path(args.output).resolve(), relative_path) != path:
            raise ValueError(
                "frozen sidecar must be a direct child of the artifact "
                f"directory: {path}"
            )
        frozen_sidecars.append(
            {"path": relative_path, "sha256": _sha256_file(path)}
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
        "frozen_sidecars": frozen_sidecars,
        "gateway_image_digest": args.gateway_image_digest,
        "mock_image_digest": args.mock_image_digest,
        "gateway_image_config_digest": args.gateway_image_config_digest,
        "mock_image_config_digest": args.mock_image_config_digest,
        "gateway_containerd_image_digest": (
            args.gateway_containerd_image_digest
        ),
        "mock_containerd_image_digest": args.mock_containerd_image_digest,
        "gateway_containerd_config_digest": (
            args.gateway_containerd_config_digest
        ),
        "mock_containerd_config_digest": (
            args.mock_containerd_config_digest
        ),
        "gateway_containerd_image_metadata": _containerd_image_metadata(
            Path(args.gateway_containerd_image_metadata),
            artifact_dir,
            args.gateway_containerd_image_digest,
        ),
        "mock_containerd_image_metadata": _containerd_image_metadata(
            Path(args.mock_containerd_image_metadata),
            artifact_dir,
            args.mock_containerd_image_digest,
        ),
        "gateway_cri_image_metadata": _cri_image_metadata(
            Path(args.gateway_cri_image_metadata),
            artifact_dir,
        ),
        "mock_cri_image_metadata": _cri_image_metadata(
            Path(args.mock_cri_image_metadata),
            artifact_dir,
        ),
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


class KubernetesProxyReader:
    """One lifecycle-owned kubectl proxy and persistent HTTP client."""

    def __init__(self, kubectl: str) -> None:
        self._kubectl = kubectl
        self._process: asyncio.subprocess.Process | None = None
        self._client: httpx.AsyncClient | None = None
        self._output_drain: asyncio.Task | None = None

    async def __aenter__(self) -> KubernetesProxyReader:
        self._process = await asyncio.create_subprocess_exec(
            self._kubectl,
            "proxy",
            "--address=127.0.0.1",
            r"--accept-hosts=^127\.0\.0\.1$",
            "--port=0",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            port = await asyncio.wait_for(
                self._read_port(),
                timeout=_KUBECTL_PROXY_START_TIMEOUT_SECONDS,
            )
            self._output_drain = asyncio.create_task(self._drain_output())
            self._client = httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
                timeout=_KUBERNETES_READ_TIMEOUT_SECONDS,
                limits=httpx.Limits(
                    max_connections=16,
                    max_keepalive_connections=16,
                ),
                trust_env=False,
            )
        except BaseException:
            await self.aclose()
            raise
        return self

    async def __aexit__(self, *_exc_info) -> None:
        await self.aclose()

    async def _read_port(self) -> int:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("kubectl proxy stdout is unavailable")
        output: list[str] = []
        while True:
            line = await process.stdout.readline()
            if not line:
                await process.wait()
                rendered = "".join(output)[-500:]
                raise RuntimeError(
                    "kubectl proxy exited before serving"
                    + (f": {rendered}" if rendered else "")
                )
            text = line.decode(errors="replace")
            output.append(text)
            match = _KUBECTL_PROXY_ADDRESS.search(text)
            if match is not None:
                return int(match.group("port"))

    async def _drain_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        while await process.stdout.readline():
            pass

    async def get_json(
        self,
        path: str,
        *,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        client = self._client
        if client is None:
            raise RuntimeError("kubectl proxy reader is not running")
        response = await client.get(path, params=params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Kubernetes API JSON output must be an object")
        return payload

    async def delete_json(
        self,
        path: str,
        *,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        client = self._client
        if client is None:
            raise RuntimeError("kubectl proxy reader is not running")
        response = await client.request("DELETE", path, json=body)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Kubernetes API JSON output must be an object")
        return payload

    async def aclose(self) -> None:
        client, self._client = self._client, None
        try:
            if client is not None:
                await client.aclose()
        finally:
            await self._stop_process()

    async def _stop_process(self) -> None:
        process, self._process = self._process, None
        output_drain, self._output_drain = self._output_drain, None
        if process is not None:
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    # The child can exit between the returncode check and signal.
                    pass
            try:
                await asyncio.wait_for(
                    process.wait(),
                    timeout=_COMMAND_TERMINATE_GRACE_SECONDS,
                )
            except TimeoutError:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                await process.wait()
        if output_drain is not None:
            if not output_drain.done():
                output_drain.cancel()
            await asyncio.gather(output_drain, return_exceptions=True)


def _proxy_reader(args: argparse.Namespace) -> KubernetesProxyReader | None:
    return getattr(args, "_kubernetes_proxy_reader", None)


async def _read_endpoint_slices(args: argparse.Namespace) -> dict[str, Any]:
    reader = _proxy_reader(args)
    if reader is not None:
        namespace = quote(str(args.namespace), safe="")
        return await reader.get_json(
            (
                "/apis/discovery.k8s.io/v1/namespaces/"
                f"{namespace}/endpointslices"
            ),
            params={
                "labelSelector": (
                    f"kubernetes.io/service-name={args.endpoint_service}"
                )
            },
        )
    return await _kubectl_json(
        args.kubectl,
        args.namespace,
        "get",
        "endpointslices",
        "-l",
        f"kubernetes.io/service-name={args.endpoint_service}",
    )


async def _read_pods(
    args: argparse.Namespace,
    *,
    selector: str | None = None,
    names: list[str] | None = None,
) -> dict[str, Any]:
    reader = _proxy_reader(args)
    if reader is None:
        if names is not None:
            return await _kubectl_json(
                args.kubectl,
                args.namespace,
                "get",
                "pods",
                *names,
            )
        if selector is None:
            raise ValueError("pod selector is required")
        return await _kubectl_json(
            args.kubectl,
            args.namespace,
            "get",
            "pods",
            "-l",
            selector,
        )

    effective_selector = selector
    if effective_selector is None and names is not None:
        effective_selector = args.pod_selector
    namespace = quote(str(args.namespace), safe="")
    payload = await reader.get_json(
        f"/api/v1/namespaces/{namespace}/pods",
        params=(
            {"labelSelector": effective_selector}
            if effective_selector is not None
            else None
        ),
    )
    if names is None:
        return payload
    requested = set(names)
    items = payload.get("items")
    if not isinstance(items, list):
        return payload
    by_name = {
        name: item
        for item in items
        if isinstance(item, dict)
        and isinstance(
            name := item.get("metadata", {}).get("name"),
            str,
        )
        and name in requested
    }
    return {
        **payload,
        "items": [by_name[name] for name in names if name in by_name],
    }


async def _read_node_summary(
    args: argparse.Namespace,
    node: str,
) -> dict[str, Any]:
    path = f"/api/v1/nodes/{quote(node, safe='')}/proxy/stats/summary"
    reader = _proxy_reader(args)
    if reader is not None:
        return await reader.get_json(
            path,
            params={"only_cpu_and_memory": "true"},
        )
    path = f"{path}?only_cpu_and_memory=true"
    _, stdout, _ = await _run_command(
        args.kubectl,
        "get",
        "--raw",
        path,
    )
    payload = json.loads(stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("kubectl raw JSON output must be an object")
    return payload


async def _delete_pods(
    args: argparse.Namespace,
    names: list[str],
) -> tuple[int, str]:
    reader = _proxy_reader(args)
    if reader is None:
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
        return returncode, stderr

    namespace = quote(str(args.namespace), safe="")
    semaphore = asyncio.Semaphore(_KUBERNETES_DELETE_MAX_CONCURRENCY)

    async def delete_one(name: str) -> str | None:
        try:
            async with semaphore:
                await reader.delete_json(
                    (
                        f"/api/v1/namespaces/{namespace}/pods/"
                        f"{quote(name, safe='')}"
                    ),
                    # Match kubectl's default background cascading while
                    # leaving gracePeriodSeconds unset. The Pod's declared
                    # grace period and preStop lifecycle therefore remain in
                    # force.
                    body={"propagationPolicy": "Background"},
                )
        except Exception as error:
            return f"{name}: {type(error).__name__}: {error}"
        return None

    errors = [
        error
        for error in await asyncio.gather(
            *(delete_one(name) for name in names)
        )
        if error is not None
    ]
    if errors:
        return 1, "\n".join(errors)
    return 0, ""


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


def _pod_identity_map(
    payload: Any,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Return an exact Pod name-to-identity map, rejecting ambiguity."""

    if not isinstance(payload, list):
        return {}, False
    identities: dict[str, dict[str, Any]] = {}
    names_by_uid: dict[str, str] = {}
    for identity in payload:
        if not isinstance(identity, dict):
            return {}, False
        name = identity.get("name")
        uid = identity.get("uid")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(uid, str)
            or not uid
            or name in identities
            or uid in names_by_uid
        ):
            return {}, False
        identities[name] = identity
        names_by_uid[uid] = name
    return identities, True


def _pod_payload_identities(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not all(
        isinstance(identity, dict) for identity in payload
    ):
        return []
    return payload


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


def _ready_endpoint_target_map(
    snapshot: dict[str, Any],
    *,
    namespace: str,
) -> tuple[dict[str, str], bool]:
    """Return ready, non-terminating EndpointSlice pod names and UIDs."""

    values: dict[str, str] = {}
    names_by_uid: dict[str, str] = {}
    pairs: set[tuple[str, str]] = set()
    for item in snapshot.get("items", ()):
        for endpoint in item.get("endpoints", ()):
            conditions = endpoint.get("conditions") or {}
            if (
                conditions.get("ready") is not True
                or conditions.get("terminating") is True
            ):
                continue
            target = endpoint.get("targetRef") or {}
            name = target.get("name")
            uid = target.get("uid")
            if (
                target.get("kind") != "Pod"
                or target.get("namespace") != namespace
                or not isinstance(name, str)
                or not name
                or not isinstance(uid, str)
                or not uid
                or (name in values and values[name] != uid)
                or (uid in names_by_uid and names_by_uid[uid] != name)
                or (name, uid) in pairs
            ):
                return {}, False
            values[name] = uid
            names_by_uid[uid] = name
            pairs.add((name, uid))
    return values, True


def _ready_endpoint_targets(
    snapshot: dict[str, Any],
    *,
    namespace: str,
) -> dict[str, str]:
    values, valid = _ready_endpoint_target_map(
        snapshot,
        namespace=namespace,
    )
    return values if valid else {}


def _endpoint_payload_shape_valid(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    items = payload.get("items")
    return (
        isinstance(items, list)
        and all(
            isinstance(item, dict)
            and isinstance(item.get("endpoints"), list)
            and all(
                isinstance(endpoint, dict)
                and (
                    endpoint.get("conditions") is None
                    or isinstance(endpoint.get("conditions"), dict)
                )
                and (
                    endpoint.get("targetRef") is None
                    or isinstance(endpoint.get("targetRef"), dict)
                )
                for endpoint in item["endpoints"]
            )
            for item in items
        )
    )


def _resource_payload_shape_valid(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    node = payload.get("node")
    pods = payload.get("pods")

    def metric(value: Any) -> bool:
        return value is None or type(value) is int

    return (
        isinstance(node, dict)
        and isinstance(node.get("name"), str)
        and bool(node["name"])
        and metric(node.get("cpu_usage_nano_cores"))
        and metric(node.get("memory_working_set_bytes"))
        and {
            "name",
            "cpu_usage_nano_cores",
            "memory_working_set_bytes",
        }
        <= set(node)
        and isinstance(pods, list)
        and all(
            isinstance(pod, dict)
            and isinstance(pod.get("name"), str)
            and bool(pod["name"])
            and isinstance(pod.get("uid"), str)
            and bool(pod["uid"])
            and metric(pod.get("cpu_usage_nano_cores"))
            and metric(pod.get("memory_working_set_bytes"))
            and (
                pod.get("network") is None
                or isinstance(pod.get("network"), dict)
            )
            and {
                "name",
                "uid",
                "cpu_usage_nano_cores",
                "memory_working_set_bytes",
                "network",
            }
            <= set(pod)
            for pod in pods
        )
    )


class JsonlSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        # A gate rerun into the same directory must never append to stale
        # evidence. Encoding and filesystem writes run on the existing
        # lifecycle-owned bounded writer so large Kubernetes snapshots cannot
        # stall the traffic generator's event loop.
        self.path.unlink(missing_ok=True)
        self._writer = BoundedJsonlWriter(path)

    def write(self, value: dict[str, Any]) -> None:
        self._writer.append_deferred(
            lambda value=value: json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def write_many(self, values: list[dict[str, Any]]) -> None:
        """Write a bulk capture without weakening bounded admission.

        A post-measurement gateway audit can contain tens of thousands of
        records. When its bounded queue fills, drain the already accepted
        prefix and retry the current row instead of dropping evidence or
        failing the completed measurement.
        """

        for value in values:
            while True:
                try:
                    self.write(value)
                    break
                except AuditQueueFull:
                    self._writer.flush()

    def close(self) -> None:
        self._writer.close()


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
    # This is a post-traffic bulk copy. Keep JSON encoding and bounded-writer
    # backpressure off the asyncio loop so the remaining evidence samplers stay
    # responsive while all gateway-owned placement rows are preserved.
    await asyncio.to_thread(sink.write_many, records)
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
    interval_ns = int(interval_s * 1_000_000_000)
    index = 0
    skipped_intervals_before = 0
    while not stop.is_set():
        target_ns = origin_ns + index * interval_ns
        remaining = (target_ns - time.monotonic_ns()) / 1_000_000_000
        if remaining > 0:
            try:
                await asyncio.wait_for(stop.wait(), timeout=remaining)
                break
            except TimeoutError:
                pass
        # A busy event loop can resume the absolute-deadline wait several
        # intervals late. Skip every fully expired slot before issuing the
        # fetch; otherwise this loop performs one stale catch-up fetch before
        # its existing post-fetch skip can take effect.
        while True:
            fetch_started_ns = time.monotonic_ns()
            if fetch_started_ns < target_ns + interval_ns:
                break
            first_unexpired_index = (
                fetch_started_ns - origin_ns
            ) // interval_ns
            skipped_intervals_before += first_unexpired_index - index
            index = first_unexpired_index
            target_ns = origin_ns + index * interval_ns
        try:
            payload = await fetch()
            record = {
                "schema_version": SCHEMA_VERSION,
                "kind": kind,
                "scheduled_ns": target_ns,
                "skipped_intervals_before": skipped_intervals_before,
                "fetch_started_ns": fetch_started_ns,
                "observed_ns": time.monotonic_ns(),
                "error": None,
                "payload": payload,
            }
        except Exception as error:
            record = {
                "schema_version": SCHEMA_VERSION,
                "kind": kind,
                "scheduled_ns": target_ns,
                "skipped_intervals_before": skipped_intervals_before,
                "fetch_started_ns": fetch_started_ns,
                "observed_ns": time.monotonic_ns(),
                "error": type(error).__name__,
                "payload": None,
            }
        records.append(record)
        sink.write(record)
        completed_index = index
        next_scheduled_index = completed_index + 1
        first_future_index = (
            math.floor(
                (time.monotonic_ns() - origin_ns)
                / interval_ns
            )
            + 1
        )
        index = max(next_scheduled_index, first_future_index)
        skipped_intervals_before = index - next_scheduled_index
    return records


async def _initial_inventory(args: argparse.Namespace, config: GateConfig) -> list[dict]:
    names = [
        f"{args.statefulset}-{ordinal}"
        for ordinal in range(config.replica_count)
    ]
    snapshot = await _read_pods(args, names=names)
    inventory = [_pod_identity(item) for item in snapshot.get("items", ())]
    if len(inventory) != config.replica_count:
        raise RuntimeError(
            f"expected {config.replica_count} mock pods, found {len(inventory)}"
        )
    if not all(item["ready"] and item["uid"] and item["ip"] for item in inventory):
        raise RuntimeError("all initial mock pods must be Ready with UID and IP")
    return inventory


async def _observe_endpoint_withdrawal(
    args: argparse.Namespace,
    config: GateConfig,
    *,
    epoch: int,
    old_uids: set[str],
    api_started_ns: int,
    endpoint_sink: JsonlSink,
) -> tuple[int | None, set[str]]:
    """Capture the causal EndpointSlice transition independently of pod recovery.

    A 20-pod ``kubectl delete`` can take more than two seconds to return, and a
    multi-pod readiness fetch can take similarly long. Neither may leave the
    fixed one-second raw-evidence bracket dependent on the phase of the global
    sampler, so this observer starts with the delete and polls only the much
    cheaper EndpointSlice surface at an absolute 250 ms cadence.
    """

    deadline_ns = api_started_ns + int(
        config.endpoint_withdrawal_limit_seconds * 1_000_000_000
    )
    interval_ns = int(
        _ENDPOINT_WITHDRAWAL_POLL_SECONDS * 1_000_000_000
    )
    endpoint_uids: set[str] = set()
    index = 0
    while time.monotonic_ns() <= deadline_ns:
        scheduled_ns = api_started_ns + index * interval_ns
        fetch_started_ns = time.monotonic_ns()
        try:
            endpoints = await _read_endpoint_slices(args)
        except Exception as endpoint_error:
            observed_ns = time.monotonic_ns()
            endpoint_sink.write(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "endpointslices",
                    "capture": "churn_withdrawal",
                    "epoch": epoch,
                    "observer_sequence": index,
                    "scheduled_ns": scheduled_ns,
                    "fetch_started_ns": fetch_started_ns,
                    "observed_ns": observed_ns,
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
                    "capture": "churn_withdrawal",
                    "epoch": epoch,
                    "observer_sequence": index,
                    "scheduled_ns": scheduled_ns,
                    "fetch_started_ns": fetch_started_ns,
                    "observed_ns": observed_ns,
                    "error": None,
                    "payload": endpoints,
                }
            )
            if old_uids.isdisjoint(endpoint_uids):
                return observed_ns, endpoint_uids
        index += 1
        target_ns = api_started_ns + index * interval_ns
        remaining = (target_ns - time.monotonic_ns()) / 1_000_000_000
        if remaining > 0:
            await asyncio.sleep(remaining)
    return None, endpoint_uids


async def _delete_with_endpoint_observation(
    args: argparse.Namespace,
    config: GateConfig,
    *,
    epoch: int,
    names: list[str],
    old_uids: set[str],
    api_started_ns: int,
    endpoint_sink: JsonlSink,
) -> tuple[int, str, int, int | None, set[str]]:
    observer = asyncio.create_task(
        _observe_endpoint_withdrawal(
            args,
            config,
            epoch=epoch,
            old_uids=old_uids,
            api_started_ns=api_started_ns,
            endpoint_sink=endpoint_sink,
        )
    )
    try:
        returncode, stderr = await _delete_pods(args, names)
        api_completed_ns = time.monotonic_ns()
        old_withdrawn_ns, endpoint_uids = await observer
        return (
            returncode,
            stderr,
            api_completed_ns,
            old_withdrawn_ns,
            endpoint_uids,
        )
    finally:
        if not observer.done():
            observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)


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
    initial_old_withdrawn_ns: int | None,
    initial_endpoint_uids: set[str],
    endpoint_sink: JsonlSink,
    sink: JsonlSink,
) -> dict[str, Any]:
    deadline = time.monotonic() + config.recovery_timeout_seconds
    names = [item["name"] for item in old]
    old_uids = {str(item["uid"]) for item in old}
    new: list[dict[str, Any]] = []
    endpoint_uids = set(initial_endpoint_uids)
    endpoint_targets: dict[str, str] = {}
    old_withdrawn_ns = initial_old_withdrawn_ns
    endpoint_recovered_ns: int | None = None
    new_pods_fetch_started_ns: int | None = None
    new_pods_observed_ns: int | None = None
    new_ready_ns: int | None = None
    error: str | None = None
    while time.monotonic() < deadline:
        endpoint_fetch_started_ns = time.monotonic_ns()
        try:
            endpoints = await _read_endpoint_slices(args)
        except Exception as endpoint_error:
            endpoint_sink.write(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "endpointslices",
                    "capture": "churn_recovery",
                    "epoch": plan_entry["epoch"],
                    "fetch_started_ns": endpoint_fetch_started_ns,
                    "observed_ns": time.monotonic_ns(),
                    "error": type(endpoint_error).__name__,
                    "payload": None,
                }
            )
        else:
            endpoint_uids = _endpoint_uids(endpoints)
            endpoint_targets = _ready_endpoint_targets(
                endpoints,
                namespace=args.namespace,
            )
            observed_ns = time.monotonic_ns()
            endpoint_sink.write(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "endpointslices",
                    "capture": "churn_recovery",
                    "epoch": plan_entry["epoch"],
                    "fetch_started_ns": endpoint_fetch_started_ns,
                    "observed_ns": observed_ns,
                    "error": None,
                    "payload": endpoints,
                }
            )
            if old_withdrawn_ns is None and old_uids.isdisjoint(endpoint_uids):
                old_withdrawn_ns = observed_ns
        replacement_uids = {
            name: endpoint_targets.get(name)
            for name in names
        }
        endpoints_recovered = (
            old_withdrawn_ns is not None
            and len(endpoint_uids) == config.replica_count
            and len(endpoint_targets) == config.replica_count
            and all(
                isinstance(uid, str) and uid not in old_uids
                for uid in replacement_uids.values()
            )
        )
        if endpoints_recovered:
            try:
                # EndpointSlice readiness is the serving contract. Fetch only
                # the twenty replaced Pods, and only after that contract is
                # satisfied, to publish full Pod/image identity without
                # polling all 200 Pod objects on every recovery iteration.
                candidate_pods_fetch_started_ns = time.monotonic_ns()
                pods = await _read_pods(
                    args,
                    selector=(
                        f"{args.pod_selector},"
                        "statefulset.kubernetes.io/pod-name in "
                        f"({','.join(names)})"
                    ),
                )
                candidate_pods_observed_ns = time.monotonic_ns()
                identities_by_name = {
                    identity["name"]: identity
                    for item in pods.get("items", ())
                    if isinstance(
                        (identity := _pod_identity(item)).get("name"),
                        str,
                    )
                }
                new = [
                    identities_by_name[name]
                    for name in names
                    if name in identities_by_name
                ]
                ready_new = (
                    len(new) == len(old)
                    and all(
                        item["ready"]
                        and item["uid"] == replacement_uids[item["name"]]
                        for item in new
                    )
                )
                if ready_new:
                    endpoint_recovered_ns = observed_ns
                    new_pods_fetch_started_ns = (
                        candidate_pods_fetch_started_ns
                    )
                    new_pods_observed_ns = candidate_pods_observed_ns
                    new_ready_ns = candidate_pods_observed_ns
                    break
            except Exception:
                pass
        await asyncio.sleep(_RECOVERY_POLL_SECONDS)
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
        "endpoint_recovered_ns": endpoint_recovered_ns,
        "new_pods_fetch_started_ns": new_pods_fetch_started_ns,
        "new_pods_observed_ns": new_pods_observed_ns,
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
            predelete_fetch_started_ns = time.monotonic_ns()
            predelete_endpoints = await _read_endpoint_slices(args)
            endpoint_sink.write(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "endpointslices",
                    "capture": "churn_predelete",
                    "epoch": entry["epoch"],
                    "fetch_started_ns": predelete_fetch_started_ns,
                    "observed_ns": time.monotonic_ns(),
                    "error": None,
                    "payload": predelete_endpoints,
                }
            )
            api_started_ns = time.monotonic_ns()
            old_uids = {str(item["uid"]) for item in old}
            (
                returncode,
                stderr,
                api_completed_ns,
                old_withdrawn_ns,
                endpoint_uids,
            ) = await _delete_with_endpoint_observation(
                args,
                config,
                epoch=entry["epoch"],
                names=names,
                old_uids=old_uids,
                api_started_ns=api_started_ns,
                endpoint_sink=endpoint_sink,
            )
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
                        initial_old_withdrawn_ns=old_withdrawn_ns,
                        initial_endpoint_uids=endpoint_uids,
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
        for identity in _pod_payload_identities(record.get("payload")):
            containers = identity.get("containers")
            if not isinstance(containers, list):
                continue
            for container in containers:
                if not isinstance(container, dict):
                    continue
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


def _pod_identity_matches_image(
    identity: dict[str, Any],
    *,
    container_name: Any,
    expected_image: Any,
    allowed_digests: set[str],
) -> bool:
    """Verify one Pod's exact container image against pinned CRI metadata."""

    containers = identity.get("containers")
    if (
        not isinstance(container_name, str)
        or not container_name
        or not isinstance(expected_image, str)
        or not expected_image
        or not isinstance(containers, list)
    ):
        return False
    matches = [
        container
        for container in containers
        if isinstance(container, dict)
        and container.get("name") == container_name
    ]
    if len(matches) != 1:
        return False
    container = matches[0]
    runtime_image = container.get("runtime_image")
    digest = _sha256_digest(container.get("image_id"))
    return (
        container.get("spec_image") == expected_image
        and isinstance(runtime_image, str)
        and bool(runtime_image)
        and _normalized_image_reference(runtime_image)
        == _normalized_image_reference(expected_image)
        and digest is not None
        and digest in allowed_digests
    )


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


def _periodic_schedule_consistent(
    records: list[dict[str, Any]],
    *,
    interval_seconds: float,
    coverage_start_ns: int | None,
    coverage_end_ns: int | None,
) -> bool:
    """Replay one absolute-deadline sampler without trusting its summary."""

    periodic = [
        record for record in records if record.get("capture") is None
    ]
    interval_ns = int(interval_seconds * 1_000_000_000)
    if (
        not periodic
        or interval_ns <= 0
        or type(coverage_start_ns) is not int
        or type(coverage_end_ns) is not int
        or coverage_start_ns > coverage_end_ns
    ):
        return False
    previous_scheduled_ns: int | None = None
    previous_observed_ns: int | None = None
    for record in periodic:
        scheduled_ns = record.get("scheduled_ns")
        skipped = record.get("skipped_intervals_before")
        fetch_started_ns = record.get("fetch_started_ns")
        observed_ns = record.get("observed_ns")
        valid = (
            type(scheduled_ns) is int
            and type(skipped) is int
            and skipped >= 0
            and type(fetch_started_ns) is int
            and type(observed_ns) is int
            and scheduled_ns <= fetch_started_ns <= observed_ns
            and fetch_started_ns < scheduled_ns + interval_ns
        )
        if previous_scheduled_ns is not None:
            valid = (
                valid
                and scheduled_ns - previous_scheduled_ns
                == (skipped + 1) * interval_ns
                and previous_observed_ns is not None
                and previous_observed_ns < scheduled_ns
            )
        else:
            valid = valid and skipped == 0
        if not valid:
            return False
        previous_scheduled_ns = scheduled_ns
        previous_observed_ns = observed_ns
    first_scheduled_ns = periodic[0]["scheduled_ns"]
    last_scheduled_ns = periodic[-1]["scheduled_ns"]
    last_observed_ns = periodic[-1]["observed_ns"]
    return (
        first_scheduled_ns <= coverage_start_ns
        < first_scheduled_ns + interval_ns
        and max(
            last_scheduled_ns + interval_ns,
            last_observed_ns,
        )
        > coverage_end_ns
    )


def _provenance_valid(environment: dict[str, Any]) -> bool:
    sha256 = re.compile(r"[0-9a-f]{64}")
    commit = re.compile(r"[0-9a-f]{40}")
    exact_image_sha256 = re.compile(r"sha256:[0-9a-f]{64}")

    def digest(value: Any) -> bool:
        return isinstance(value, str) and sha256.fullmatch(value) is not None

    def image_digest(value: Any) -> bool:
        return (
            isinstance(value, str)
            and re.search(r"sha256:([0-9a-f]{64})(?:$|[^0-9a-f])", value)
            is not None
        )

    git_commit = environment.get("git_commit")

    def cri_metadata(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        path = value.get("path")
        status_id = value.get("status_id")
        repo_digests = value.get("repo_digests")
        repo_tags = value.get("repo_tags")
        runtime_digests = value.get("runtime_digests")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not digest(value.get("sha256"))
            or not isinstance(status_id, str)
            or exact_image_sha256.fullmatch(status_id) is None
            or not isinstance(repo_digests, list)
            or not all(isinstance(item, str) for item in repo_digests)
            or not isinstance(repo_tags, list)
            or not repo_tags
            or not all(isinstance(item, str) and item for item in repo_tags)
            or not isinstance(runtime_digests, list)
            or value.get("revision") != git_commit
        ):
            return False
        derived_digests = {status_id}
        for repo_digest in repo_digests:
            _, separator, candidate = repo_digest.rpartition("@")
            if (
                not separator
                or exact_image_sha256.fullmatch(candidate) is None
            ):
                return False
            derived_digests.add(candidate)
        return (
            repo_digests == sorted(set(repo_digests))
            and repo_tags == sorted(set(repo_tags))
            and runtime_digests == sorted(derived_digests)
        )

    def image_chain(role: str) -> bool:
        image_digest_value = environment.get(f"{role}_image_digest")
        image_config_digest = environment.get(
            f"{role}_image_config_digest"
        )
        containerd_image_digest = environment.get(
            f"{role}_containerd_image_digest"
        )
        containerd_config_digest = environment.get(
            f"{role}_containerd_config_digest"
        )
        containerd_metadata = environment.get(
            f"{role}_containerd_image_metadata"
        )
        metadata = environment.get(f"{role}_cri_image_metadata")
        containerd_metadata_valid = (
            isinstance(containerd_metadata, dict)
            and isinstance(containerd_metadata.get("path"), str)
            and bool(containerd_metadata["path"])
            and not Path(containerd_metadata["path"]).is_absolute()
            and ".." not in Path(containerd_metadata["path"]).parts
            and digest(containerd_metadata.get("sha256"))
            and containerd_metadata.get("target_digest")
            == containerd_image_digest
            and containerd_metadata.get("media_type")
            in {
                "application/vnd.oci.image.manifest.v1+json",
                "application/vnd.docker.distribution.manifest.v2+json",
            }
            and containerd_metadata.get("config_digest")
            == containerd_config_digest
            and isinstance(containerd_metadata.get("layer_digests"), list)
            and bool(containerd_metadata["layer_digests"])
            and all(
                isinstance(item, str)
                and exact_image_sha256.fullmatch(item) is not None
                for item in containerd_metadata["layer_digests"]
            )
        )
        return (
            isinstance(image_digest_value, str)
            and exact_image_sha256.fullmatch(image_digest_value) is not None
            and isinstance(image_config_digest, str)
            and exact_image_sha256.fullmatch(image_config_digest) is not None
            and isinstance(containerd_image_digest, str)
            and exact_image_sha256.fullmatch(containerd_image_digest)
            is not None
            and isinstance(containerd_config_digest, str)
            and exact_image_sha256.fullmatch(containerd_config_digest)
            is not None
            and image_digest_value
            in {image_config_digest, containerd_image_digest}
            and image_config_digest == containerd_config_digest
            and containerd_metadata_valid
            and cri_metadata(metadata)
            and metadata.get("status_id") == image_config_digest
        )

    frozen_inputs = environment.get("frozen_inputs")
    frozen_sidecars = environment.get("frozen_sidecars")
    frozen_sidecar_paths = (
        [item.get("path") for item in frozen_sidecars]
        if isinstance(frozen_sidecars, list)
        and all(
            isinstance(item, dict)
            and isinstance(item.get("path"), str)
            for item in frozen_sidecars
        )
        else []
    )
    tool_versions = environment.get("tool_versions")
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
        and image_chain("gateway")
        and image_chain("mock")
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
        and isinstance(frozen_sidecars, list)
        and bool(frozen_sidecars)
        and len(frozen_sidecar_paths) == len(set(frozen_sidecar_paths))
        and all(
            isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and bool(item["path"])
            and not Path(item["path"]).is_absolute()
            and ".." not in Path(item["path"]).parts
            and digest(item.get("sha256"))
            for item in frozen_sidecars
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
    namespace = environment.get("namespace")
    endpoint_uids = set()
    endpoint_timed_snapshots: list[dict[str, Any]] = []
    endpoint_errors = 0
    endpoint_evidence_schema_valid = bool(endpoints)
    endpoint_namespace_valid = (
        isinstance(namespace, str) and bool(namespace)
    )
    all_endpoint_target_refs_valid = endpoint_namespace_valid
    for record in endpoints:
        record_shape_valid = (
            type(record.get("schema_version")) is int
            and record["schema_version"] == SCHEMA_VERSION
            and record.get("kind") == "endpointslices"
            and "error" in record
            and (
                record.get("error") is not None
                or _endpoint_payload_shape_valid(record.get("payload"))
            )
        )
        endpoint_evidence_schema_valid = (
            endpoint_evidence_schema_valid and record_shape_valid
        )
        if record.get("error"):
            endpoint_errors += 1
            continue
        payload = (
            record["payload"]
            if _endpoint_payload_shape_valid(record.get("payload"))
            else {}
        )
        snapshot_uids = _endpoint_uids(payload)
        snapshot_targets, target_refs_valid = _ready_endpoint_target_map(
            payload,
            namespace=namespace if isinstance(namespace, str) else "",
        )
        all_endpoint_target_refs_valid = (
            all_endpoint_target_refs_valid and target_refs_valid
        )
        if type(record.get("observed_ns")) is int:
            endpoint_timed_snapshots.append(
                {
                    "observed_ns": record["observed_ns"],
                    "fetch_started_ns": (
                        record["fetch_started_ns"]
                        if type(record.get("fetch_started_ns")) is int
                        else record["observed_ns"]
                    ),
                    "fetch_started_recorded": (
                        type(record.get("fetch_started_ns")) is int
                    ),
                    "uids": snapshot_uids,
                    "targets": snapshot_targets,
                    "target_refs_valid": target_refs_valid,
                    "capture": record.get("capture"),
                    "epoch": record.get("epoch"),
                    "observer_sequence": record.get("observer_sequence"),
                    "scheduled_ns": record.get("scheduled_ns"),
                    "skipped_intervals_before": record.get(
                        "skipped_intervals_before"
                    ),
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
        raw_payload = record.get("payload")
        identities = _pod_payload_identities(raw_payload)
        if not isinstance(raw_payload, list) or len(identities) != len(
            raw_payload
        ):
            pod_uid_name_consistent = False
        for identity in identities:
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
            endpoint_recovered_ns = record.get("endpoint_recovered_ns")
            new_pods_fetch_started_ns = record.get(
                "new_pods_fetch_started_ns"
            )
            new_pods_observed_ns = record.get("new_pods_observed_ns")
            new_ready_ns = record.get("new_ready_ns")
            recovered_ns = record.get("recovered_ns")
            raw_times_valid = all(
                type(value) is int
                for value in (
                    scheduled_ns,
                    api_started_ns,
                    api_completed_ns,
                    old_withdrawn_ns,
                    endpoint_recovered_ns,
                    new_pods_fetch_started_ns,
                    new_pods_observed_ns,
                    new_ready_ns,
                    recovered_ns,
                )
            )
            causal = (
                raw_times_valid
                and scheduled_ns == expected_scheduled_ns
                and scheduled_ns <= api_started_ns <= api_completed_ns
                and api_started_ns
                <= old_withdrawn_ns
                <= endpoint_recovered_ns
                <= new_pods_fetch_started_ns
                <= new_pods_observed_ns
                == new_ready_ns
                <= recovered_ns
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

    pod_initial_records = [
        record
        for record in pods
        if record.get("kind") == "pods"
        and record.get("capture") == "initial"
    ]
    pod_recovery_records = [
        record
        for record in pods
        if record.get("kind") == "pods"
        and record.get("capture") == "churn_recovery"
    ]
    pod_final_records = [
        record
        for record in pods
        if record.get("kind") == "pods"
        and record.get("capture") == "final"
    ]
    gateway_pod_records = [
        record
        for record in pods
        if record.get("kind") == "gateway_pods"
        and record.get("capture") == "gateway"
    ]
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
        pod_initial_records[0].get("payload", ())
        if len(pod_initial_records) == 1
        and not pod_initial_records[0].get("error")
        else ()
    )
    initial_pod_identities, initial_pod_map_valid = _pod_identity_map(
        initial_pod_payload
    )
    initial_pod_state = {
        name: identity["uid"]
        for name, identity in initial_pod_identities.items()
    }
    state_mapping_valid = (
        bool(expected_initial_names)
        and initial_pod_map_valid
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
    expected_state_before_epoch: list[dict[str, str]] = []
    expected_state_by_epoch: list[dict[str, str]] = []
    lifecycle_valid = len(current_fleet) == config.replica_count
    exact_identity = True
    for expected_epoch, record in enumerate(ordered_churn):
        expected_fleet_before_epoch.append(set(current_fleet))
        expected_state_before_epoch.append(dict(current_state_by_name))
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
        expected_state_by_epoch.append(dict(current_state_by_name))
    lifecycle_valid = (
        lifecycle_valid
        and len(ordered_churn) == config.churn_epochs
        and len(expected_fleet_by_epoch) == config.churn_epochs
        and len(current_fleet) == config.replica_count
        and current_fleet == new_uids
    )
    pod_recovery_epochs = [
        record.get("epoch") for record in pod_recovery_records
    ]
    pod_capture_structure_complete = (
        len(pods) == config.churn_epochs + 3
        and len(pod_initial_records) == 1
        and len(pod_recovery_records) == config.churn_epochs
        and pod_recovery_epochs == list(range(config.churn_epochs))
        and len(pod_final_records) == 1
        and len(gateway_pod_records) == 1
        and all(record.get("error") is None for record in pods)
        and all(
            type(record.get("schema_version")) is int
            and record["schema_version"] == SCHEMA_VERSION
            and "error" in record
            for record in pods
        )
    )

    def pod_record_identities(
        record: dict[str, Any],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str], bool]:
        identities, valid = _pod_identity_map(record.get("payload"))
        return (
            identities,
            {
                name: identity["uid"]
                for name, identity in identities.items()
            },
            valid,
        )

    initial_capture_identities: dict[str, dict[str, Any]] = {}
    initial_capture_state: dict[str, str] = {}
    initial_capture_valid = False
    if len(pod_initial_records) == 1:
        (
            initial_capture_identities,
            initial_capture_state,
            initial_capture_valid,
        ) = pod_record_identities(pod_initial_records[0])
    final_capture_identities: dict[str, dict[str, Any]] = {}
    final_capture_state: dict[str, str] = {}
    final_capture_valid = False
    if len(pod_final_records) == 1:
        (
            final_capture_identities,
            final_capture_state,
            final_capture_valid,
        ) = pod_record_identities(pod_final_records[0])
    gateway_capture_identities: dict[str, dict[str, Any]] = {}
    gateway_capture_valid = False
    if len(gateway_pod_records) == 1:
        (
            gateway_capture_identities,
            _,
            gateway_capture_valid,
        ) = pod_record_identities(gateway_pod_records[0])

    recovery_capture_details = []
    required_mock_identities = list(initial_capture_identities.values())
    for epoch, churn_record in enumerate(ordered_churn):
        captures = [
            record
            for record in pod_recovery_records
            if record.get("epoch") == epoch
        ]
        identities: dict[str, dict[str, Any]] = {}
        state: dict[str, str] = {}
        capture_valid = False
        if len(captures) == 1:
            identities, state, capture_valid = pod_record_identities(
                captures[0]
            )
        expected_state = {
            identity.get("name"): identity.get("uid")
            for identity in churn_record.get("new", ())
            if isinstance(identity, dict)
        }
        identity_exact = (
            capture_valid
            and len(expected_state) == config.churn_batch_size
            and state == expected_state
        )
        timeline_consistent = (
            len(captures) == 1
            and type(churn_record.get("endpoint_recovered_ns")) is int
            and type(churn_record.get("new_pods_fetch_started_ns")) is int
            and type(churn_record.get("new_pods_observed_ns")) is int
            and type(churn_record.get("new_ready_ns")) is int
            and type(churn_record.get("recovered_ns")) is int
            and captures[0].get("fetch_started_ns")
            == churn_record["new_pods_fetch_started_ns"]
            and captures[0].get("observed_ns")
            == churn_record["new_pods_observed_ns"]
            and churn_record["endpoint_recovered_ns"]
            <= churn_record["new_pods_fetch_started_ns"]
            <= churn_record["new_pods_observed_ns"]
            == churn_record["new_ready_ns"]
            <= churn_record["recovered_ns"]
        )
        ready = (
            identity_exact
            and all(
                identity.get("ready") is True
                for identity in identities.values()
            )
        )
        recovery_capture_details.append(
            {
                "epoch": churn_record.get("epoch"),
                "identity_exact": identity_exact,
                "timeline_consistent": timeline_consistent,
                "ready": ready,
            }
        )
        required_mock_identities.extend(identities.values())
    required_mock_identities.extend(final_capture_identities.values())

    initial_identity_exact = (
        initial_capture_valid
        and len(initial_capture_state) == config.replica_count
        and initial_capture_state == initial_state_by_name
    )
    final_identity_exact = (
        final_capture_valid
        and len(final_capture_state) == config.replica_count
        and final_capture_state == current_state_by_name
    )
    gateway_identity_exact = (
        gateway_capture_valid
        and len(gateway_capture_identities) == 1
        and {
            name: identity["uid"]
            for name, identity in gateway_capture_identities.items()
        }
        == {
            environment.get("gateway_pod_name"): environment.get(
                "gateway_pod_uid"
            )
        }
        and isinstance(environment.get("gateway_pod_name"), str)
        and bool(environment["gateway_pod_name"])
        and isinstance(environment.get("gateway_pod_uid"), str)
        and bool(environment["gateway_pod_uid"])
    )
    pod_capture_identity_exact = (
        lifecycle_valid
        and initial_identity_exact
        and len(recovery_capture_details) == config.churn_epochs
        and all(
            detail["identity_exact"] for detail in recovery_capture_details
        )
        and final_identity_exact
        and gateway_identity_exact
    )

    def record_fetch_precedes_observation(record: dict[str, Any]) -> bool:
        return (
            type(record.get("fetch_started_ns")) is int
            and type(record.get("observed_ns")) is int
            and record["fetch_started_ns"] <= record["observed_ns"]
        )

    first_api_started_ns = min(
        (
            record["api_started_ns"]
            for record in ordered_churn
            if type(record.get("api_started_ns")) is int
        ),
        default=None,
    )
    last_recovered_ns = max(
        (
            record["recovered_ns"]
            for record in ordered_churn
            if type(record.get("recovered_ns")) is int
        ),
        default=None,
    )
    traffic_started_ns = min(
        (
            record["send_ns"]
            for record in requests
            if type(record.get("send_ns")) is int
        ),
        default=None,
    )
    traffic_completed_ns = max(
        (
            record["end_ns"]
            for record in requests
            if type(record.get("end_ns")) is int
        ),
        default=None,
    )
    pod_capture_timeline_consistent = (
        pod_capture_structure_complete
        and record_fetch_precedes_observation(pod_initial_records[0])
        and type(first_api_started_ns) is int
        and type(traffic_started_ns) is int
        and pod_initial_records[0]["observed_ns"] <= first_api_started_ns
        and pod_initial_records[0]["observed_ns"] <= traffic_started_ns
        and all(
            detail["timeline_consistent"]
            for detail in recovery_capture_details
        )
        and record_fetch_precedes_observation(pod_final_records[0])
        and type(last_recovered_ns) is int
        and type(traffic_completed_ns) is int
        and last_recovered_ns <= pod_final_records[0]["fetch_started_ns"]
        and traffic_completed_ns <= pod_final_records[0]["fetch_started_ns"]
        and record_fetch_precedes_observation(gateway_pod_records[0])
        and pod_final_records[0]["observed_ns"]
        <= gateway_pod_records[0]["fetch_started_ns"]
    )
    pod_capture_readiness_complete = (
        pod_capture_identity_exact
        and all(
            identity.get("ready") is True
            for identity in (
                *initial_capture_identities.values(),
                *final_capture_identities.values(),
                *gateway_capture_identities.values(),
            )
        )
        and all(
            detail["ready"] for detail in recovery_capture_details
        )
    )
    endpoint_epoch_recoveries = []
    for epoch, record in enumerate(ordered_churn):
        expected_fleet = (
            expected_fleet_by_epoch[epoch]
            if epoch < len(expected_fleet_by_epoch)
            else set()
        )
        expected_state = (
            expected_state_by_epoch[epoch]
            if epoch < len(expected_state_by_epoch)
            else {}
        )
        recovery_uids = {
            uid
            for uid in record.get("endpoint_uids_at_recovery", ())
            if isinstance(uid, str) and uid
        }
        api_started_ns = record.get("api_started_ns")
        endpoint_recovered_ns = record.get("endpoint_recovered_ns")
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
            and snapshot["target_refs_valid"]
            and snapshot["targets"] == expected_state
        ]
        claimed_recovery_snapshots = [
            snapshot
            for snapshot in endpoint_timed_snapshots
            if snapshot["capture"] == "churn_recovery"
            and snapshot["epoch"] == epoch
            and type(endpoint_recovered_ns) is int
            and snapshot["observed_ns"] == endpoint_recovered_ns
        ]
        claimed_recovery_exact = (
            len(claimed_recovery_snapshots) == 1
            and claimed_recovery_snapshots[0]["target_refs_valid"]
            and claimed_recovery_snapshots[0]["targets"] == expected_state
        )
        exact = (
            lifecycle_valid
            and recovery_uids == expected_fleet
            and record.get("ready_endpoint_count_at_recovery")
            == config.replica_count
            and claimed_recovery_exact
            and bool(periodic_exact_observed_ns)
        )
        endpoint_epoch_recoveries.append(
            {
                "epoch": record.get("epoch"),
                "expected_uids": sorted(expected_fleet),
                "observed_uids": sorted(recovery_uids),
                "endpoint_recovered_ns": endpoint_recovered_ns,
                "claimed_recovery_snapshot_count": len(
                    claimed_recovery_snapshots
                ),
                "claimed_recovery_exact": claimed_recovery_exact,
                "periodic_exact_observed_ns": periodic_exact_observed_ns,
                "exact": exact,
            }
        )
    endpoint_recovery_exact = (
        len(endpoint_epoch_recoveries) == config.churn_epochs
        and all(record["exact"] for record in endpoint_epoch_recoveries)
    )
    endpoint_epoch_causality = []
    endpoint_withdrawal_observers = []
    endpoint_interval_ns = int(
        config.evidence_interval_seconds * 1_000_000_000
    )
    gateway_withdrawal_limit_ns = int(
        config.gateway_membership_withdrawal_limit_seconds
        * 1_000_000_000
    )
    endpoint_causal_bracket_ns = int(
        _ENDPOINT_WITHDRAWAL_CAUSAL_BRACKET_SECONDS * 1_000_000_000
    )
    withdrawal_poll_interval_ns = int(
        _ENDPOINT_WITHDRAWAL_POLL_SECONDS * 1_000_000_000
    )
    for epoch, record in enumerate(ordered_churn):
        old_epoch_uids = {
            uid
            for item in record.get("old", ())
            if isinstance((uid := item.get("uid")), str) and uid
        }
        api_started_ns = record.get("api_started_ns")
        old_withdrawn_ns = record.get("old_withdrawn_ns")
        observer_snapshots = [
            snapshot
            for snapshot in endpoint_timed_snapshots
            if snapshot["capture"] == "churn_withdrawal"
            and snapshot["epoch"] == epoch
        ]
        observer_sequences = [
            snapshot["observer_sequence"]
            for snapshot in observer_snapshots
        ]
        observer_first_disjoint = next(
            (
                snapshot
                for snapshot in observer_snapshots
                if old_epoch_uids.isdisjoint(snapshot["uids"])
            ),
            None,
        )
        observer_consistent = (
            type(api_started_ns) is int
            and bool(observer_snapshots)
            and observer_sequences == list(range(len(observer_snapshots)))
            and all(
                type(snapshot["observer_sequence"]) is int
                and type(snapshot["scheduled_ns"]) is int
                and snapshot["fetch_started_recorded"]
                and snapshot["scheduled_ns"]
                == api_started_ns
                + snapshot["observer_sequence"]
                * withdrawal_poll_interval_ns
                and snapshot["scheduled_ns"]
                <= snapshot["fetch_started_ns"]
                <= snapshot["observed_ns"]
                for snapshot in observer_snapshots
            )
            and observer_first_disjoint is observer_snapshots[-1]
            and type(old_withdrawn_ns) is int
            and observer_first_disjoint["observed_ns"] == old_withdrawn_ns
        )
        endpoint_withdrawal_observers.append(
            {
                "epoch": record.get("epoch"),
                "samples": len(observer_snapshots),
                "first_sequence": (
                    observer_sequences[0] if observer_sequences else None
                ),
                "last_sequence": (
                    observer_sequences[-1] if observer_sequences else None
                ),
                "first_disjoint_ns": (
                    observer_first_disjoint["observed_ns"]
                    if observer_first_disjoint is not None
                    else None
                ),
                "consistent": observer_consistent,
            }
        )
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
            and predelete_snapshots[0]["target_refs_valid"]
            and epoch < len(expected_state_before_epoch)
            and predelete_snapshots[0]["targets"]
            == expected_state_before_epoch[epoch]
        )
        withdrawal_bracketed = (
            first_disjoint is not None
            and last_with_old is not None
            and raw_claim_snapshot is not None
            and first_disjoint["fetch_started_recorded"]
            and last_with_old["fetch_started_recorded"]
            and type(old_withdrawn_ns) is int
            and first_disjoint["observed_ns"]
            <= old_withdrawn_ns
            <= first_disjoint["observed_ns"] + endpoint_causal_bracket_ns
            and first_disjoint["observed_ns"]
            - last_with_old["fetch_started_ns"]
            <= endpoint_causal_bracket_ns
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
    endpoint_withdrawal_observer_consistent = (
        len(endpoint_withdrawal_observers) == config.churn_epochs
        and all(
            record["consistent"]
            for record in endpoint_withdrawal_observers
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
        if type(old_withdrawn_ns) is int and type(api_started_ns) is int:
            # Endpoint evidence and gateway discovery are independent polling
            # loops.  The former's capture cadence is not a convergence SLO.
            # Bound the gateway by its explicit two-interval budget and, even
            # when the EndpointSlice observer is late, by the five-second
            # graceful withdrawal contract measured from deletion.
            withdrawal_deadline_ns = min(
                old_withdrawn_ns + gateway_withdrawal_limit_ns,
                api_started_ns
                + int(
                    config.endpoint_withdrawal_limit_seconds
                    * 1_000_000_000
                ),
            )
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
    mock_cri_metadata = environment.get("mock_cri_image_metadata")
    expected_mock_runtime_digests = {
        digest
        for raw in (
            mock_cri_metadata.get("runtime_digests", ())
            if isinstance(mock_cri_metadata, dict)
            else ()
        )
        if (digest := _sha256_digest(raw)) is not None
    }
    expected_mock_runtime_tags = {
        _normalized_image_reference(value)
        for value in (
            mock_cri_metadata.get("repo_tags", ())
            if isinstance(mock_cri_metadata, dict)
            else ()
        )
        if isinstance(value, str)
    }
    required_mock_images_pinned = (
        pod_capture_identity_exact
        and bool(expected_mock_runtime_digests)
        and all(
            _pod_identity_matches_image(
                identity,
                container_name=mock_container_name,
                expected_image=expected_mock_image,
                allowed_digests=expected_mock_runtime_digests,
            )
            for identity in required_mock_identities
        )
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
        for identity in _pod_payload_identities(record.get("payload")):
            uid = identity.get("uid")
            if not isinstance(uid, str) or not uid:
                continue
            containers = identity.get("containers")
            if not isinstance(containers, list):
                continue
            for container in containers:
                if not isinstance(container, dict):
                    continue
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
        and expected_mock_runtime_tags
        == {_normalized_image_reference(expected_mock_image)}
        and bool(runtime_image_digests)
        and runtime_image_digests <= expected_mock_runtime_digests
        and runtime_mock_per_uid_ok
        and required_mock_images_pinned
    )
    gateway_container_name = environment.get("gateway_container_name")
    expected_gateway_image = environment.get("expected_gateway_image")
    gateway_cri_metadata = environment.get("gateway_cri_image_metadata")
    expected_gateway_runtime_digests = {
        digest
        for raw in (
            gateway_cri_metadata.get("runtime_digests", ())
            if isinstance(gateway_cri_metadata, dict)
            else ()
        )
        if (digest := _sha256_digest(raw)) is not None
    }
    expected_gateway_runtime_tags = {
        _normalized_image_reference(value)
        for value in (
            gateway_cri_metadata.get("repo_tags", ())
            if isinstance(gateway_cri_metadata, dict)
            else ()
        )
        if isinstance(value, str)
    }
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
        for identity in _pod_payload_identities(record.get("payload"))
        if any(
            isinstance(container, dict)
            and container.get("name") == gateway_container_name
            for container in (
                identity.get("containers")
                if isinstance(identity.get("containers"), list)
                else ()
            )
        )
    ]
    required_gateway_images_pinned = (
        gateway_identity_exact
        and bool(expected_gateway_runtime_digests)
        and all(
            _pod_identity_matches_image(
                identity,
                container_name=gateway_container_name,
                expected_image=expected_gateway_image,
                allowed_digests=expected_gateway_runtime_digests,
            )
            for identity in gateway_capture_identities.values()
        )
    )
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
        and expected_gateway_runtime_tags
        == {_normalized_image_reference(expected_gateway_image)}
        and bool(runtime_gateway_digests)
        and runtime_gateway_digests <= expected_gateway_runtime_digests
        and required_gateway_images_pinned
    )
    pod_capture_image_identity_pinned = (
        required_mock_images_pinned and required_gateway_images_pinned
    )
    placement_ok = (
        overall["samples"] > 0
        and overall["p99_ms"] is not None
        and overall["p99_ms"] < config.placement_p99_limit_ms
    )
    placement_diagnostics_complete = all(
        item["samples"] > 0 and item["p99_ms"] is not None
        for item in [*epochs, *churn_windows]
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
    allowed_endpoint_captures = {
        None,
        "churn_predelete",
        "churn_withdrawal",
        "churn_recovery",
    }
    endpoint_capture_labels_valid = all(
        record.get("capture") in allowed_endpoint_captures
        for record in endpoints
    )
    initial_region_snapshots = [
        snapshot
        for snapshot in endpoint_timed_snapshots
        if type(traffic_started_ns) is int
        and type(first_api_started_ns) is int
        and traffic_started_ns
        <= snapshot["observed_ns"]
        <= first_api_started_ns
    ]
    final_region_snapshots = [
        snapshot
        for snapshot in endpoint_timed_snapshots
        if type(last_recovered_ns) is int
        and snapshot["observed_ns"] >= last_recovered_ns
    ]
    initial_endpoints_exact = (
        type(first_api_started_ns) is int
        and bool(initial_region_snapshots)
        and all(
            snapshot["target_refs_valid"]
            and snapshot["targets"] == initial_state_by_name
            for snapshot in initial_region_snapshots
        )
    )
    final_endpoints_exact = (
        type(last_recovered_ns) is int
        and bool(final_region_snapshots)
        and all(
            snapshot["target_refs_valid"]
            and snapshot["targets"] == current_state_by_name
            for snapshot in final_region_snapshots
        )
    )
    endpoint_target_refs_valid = (
        endpoint_namespace_valid
        and all_endpoint_target_refs_valid
        and endpoint_capture_labels_valid
        and initial_endpoints_exact
        and final_endpoints_exact
        and endpoint_predelete_exact
        and len(endpoint_epoch_recoveries) == config.churn_epochs
        and all(
            recovery["claimed_recovery_exact"]
            for recovery in endpoint_epoch_recoveries
        )
    )
    endpoint_periodic_schedule_ok = _periodic_schedule_consistent(
        endpoints,
        interval_seconds=config.evidence_interval_seconds,
        coverage_start_ns=traffic_started_ns,
        coverage_end_ns=traffic_completed_ns,
    )
    resource_periodic_schedule_ok = _periodic_schedule_consistent(
        resources,
        interval_seconds=config.resource_interval_seconds,
        coverage_start_ns=traffic_started_ns,
        coverage_end_ns=traffic_completed_ns,
    )
    resource_evidence_schema_valid = (
        bool(resources)
        and all(
            type(record.get("schema_version")) is int
            and record["schema_version"] == SCHEMA_VERSION
            and record.get("kind") == "resources"
            and record.get("capture") is None
            and "error" in record
            and record.get("error") is None
            and _resource_payload_shape_valid(record.get("payload"))
            for record in resources
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
        "placement_p99_measurement": placement_ok,
        "placement_window_diagnostics_complete": (
            placement_diagnostics_complete
        ),
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
        "endpoint_withdrawal_observer_consistent": (
            endpoint_withdrawal_observer_consistent
        ),
        "endpoint_raw_withdrawal_causal": endpoint_raw_withdrawal_causal,
        "new_endpoints_ready_with_exact_fleet_size": endpoint_recovery_exact,
        "endpoint_evidence_schema_valid": endpoint_evidence_schema_valid,
        "endpoint_capture_labels_valid": endpoint_capture_labels_valid,
        "endpoint_target_refs_valid": endpoint_target_refs_valid,
        "periodic_sampler_schedule_consistent": (
            endpoint_periodic_schedule_ok
            and resource_periodic_schedule_ok
        ),
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
            and initial_endpoints_exact
        ),
        "final_endpoints_exact_new_fleet": (
            endpoint_errors == 0
            and final_endpoints_exact
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
        "pod_capture_structure_complete": pod_capture_structure_complete,
        "pod_capture_identity_exact": pod_capture_identity_exact,
        "pod_capture_timeline_consistent": (
            pod_capture_timeline_consistent
        ),
        "pod_capture_readiness_complete": (
            pod_capture_readiness_complete
        ),
        "pod_capture_image_identity_pinned": (
            pod_capture_image_identity_pinned
        ),
        "runtime_mock_image_pinned": runtime_mock_image_ok,
        "runtime_gateway_image_pinned": runtime_gateway_image_ok,
        "resource_evidence_complete": (
            resource_evidence_schema_valid
            and resource_periodic_schedule_ok
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
            "endpoint_withdrawal_observers": (
                endpoint_withdrawal_observers
            ),
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


def _cri_metadata_file_matches(
    manifest_path: Path,
    descriptor: Any,
) -> bool:
    if not isinstance(descriptor, dict):
        return False
    path = _safe_sidecar(manifest_path, str(descriptor.get("path", "")))
    return path.is_file() and _cri_image_metadata(path, path.parent) == descriptor


def _containerd_metadata_file_matches(
    manifest_path: Path,
    descriptor: Any,
) -> bool:
    if not isinstance(descriptor, dict):
        return False
    path = _safe_sidecar(manifest_path, str(descriptor.get("path", "")))
    return path.is_file() and _containerd_image_metadata(
        path,
        path.parent,
        str(descriptor.get("target_digest", "")),
    ) == descriptor


def verify_artifact(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("artifact manifest must be a JSON object")
    checks: dict[str, bool] = {
        "schema_version": manifest.get("schema_version") == SCHEMA_VERSION,
        "gate": manifest.get("gate") == GATE,
    }
    environment = manifest.get("environment") or {}
    frozen_sidecars = environment.get("frozen_sidecars")
    frozen_sidecar_descriptors_valid = (
        isinstance(frozen_sidecars, list)
        and bool(frozen_sidecars)
        and all(isinstance(item, dict) for item in frozen_sidecars)
    )
    checks["frozen_sidecars:descriptor"] = frozen_sidecar_descriptors_valid
    if frozen_sidecar_descriptors_valid:
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
            checks[f"frozen_sidecar:{index}:hash"] = (
                exists and _sha256_file(path) == descriptor.get("sha256")
            )
    for role in ("gateway", "mock"):
        try:
            checks[f"{role}_containerd_image_metadata_raw"] = (
                _containerd_metadata_file_matches(
                    manifest_path,
                    environment.get(f"{role}_containerd_image_metadata"),
                )
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            checks[f"{role}_containerd_image_metadata_raw"] = False
        try:
            checks[f"{role}_cri_image_metadata_raw"] = (
                _cri_metadata_file_matches(
                    manifest_path,
                    environment.get(f"{role}_cri_image_metadata"),
                )
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            checks[f"{role}_cri_image_metadata_raw"] = False
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
        initial_pods_fetch_started_ns = time.monotonic_ns()
        initial = await _initial_inventory(args, config)
        initial_pods_observed_ns = time.monotonic_ns()
        initial_by_name = {item["name"]: item for item in initial}
        initial_pod_record = {
            "schema_version": SCHEMA_VERSION,
            "kind": "pods",
            "capture": "initial",
            "fetch_started_ns": initial_pods_fetch_started_ns,
            "observed_ns": initial_pods_observed_ns,
            "error": None,
            "payload": initial,
        }
        sinks["pods"].write(initial_pod_record)
        measurement_origin_ns = time.monotonic_ns() + int(
            config.warmup_seconds * 1_000_000_000
        )

        async def fetch_endpoints() -> dict[str, Any]:
            return await _read_endpoint_slices(args)

        node = str(initial[0]["node"])

        async def fetch_resources() -> dict[str, Any]:
            return _resource_summary(
                await _read_node_summary(args, node),
                args.namespace,
            )

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
        endpoints, resources = await asyncio.gather(
            endpoint_task,
            resource_task,
        )
        churn_pod_records = [
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "pods",
                "capture": "churn_recovery",
                "epoch": record["epoch"],
                "fetch_started_ns": record["new_pods_fetch_started_ns"],
                "observed_ns": record["new_pods_observed_ns"],
                "error": record.get("error"),
                "payload": record.get("new", ()),
            }
            for record in churn
        ]
        for record in churn_pod_records:
            sinks["pods"].write(record)
        final_mock_fetch_started_ns = time.monotonic_ns()
        final_mock_snapshot = await _read_pods(
            args,
            selector=args.pod_selector,
        )
        final_mock_observed_ns = time.monotonic_ns()
        final_mock_pod_record = {
            "schema_version": SCHEMA_VERSION,
            "kind": "pods",
            "capture": "final",
            "fetch_started_ns": final_mock_fetch_started_ns,
            "observed_ns": final_mock_observed_ns,
            "error": None,
            "payload": [
                _pod_identity(item)
                for item in final_mock_snapshot.get("items", ())
            ],
        }
        sinks["pods"].write(final_mock_pod_record)
        pods = [
            initial_pod_record,
            *churn_pod_records,
            final_mock_pod_record,
        ]
        gateway_pods_fetch_started_ns = time.monotonic_ns()
        gateway_snapshot = await _read_pods(
            args,
            selector=args.gateway_pod_selector,
        )
        gateway_pods_observed_ns = time.monotonic_ns()
        gateway_pod_record = {
            "schema_version": SCHEMA_VERSION,
            "kind": "gateway_pods",
            "capture": "gateway",
            "fetch_started_ns": gateway_pods_fetch_started_ns,
            "observed_ns": gateway_pods_observed_ns,
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
            "namespace": args.namespace,
            "statefulset": args.statefulset,
            "mock_container_name": args.mock_container,
            "expected_mock_image": args.mock_image,
            "runtime_mock_image": _runtime_mock_image_summary(
                pods,
                args.mock_container,
            ),
            "gateway_container_name": args.gateway_container,
            "expected_gateway_image": args.gateway_image,
            "gateway_pod_name": (
                gateway_pod_record["payload"][0].get("name")
                if len(gateway_pod_record["payload"]) == 1
                else None
            ),
            "gateway_pod_uid": (
                gateway_pod_record["payload"][0].get("uid")
                if len(gateway_pod_record["payload"]) == 1
                else None
            ),
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
            "windows": (
                "the overall measurement p99 is the F1a acceptance metric; "
                "every epoch and post-delete window remains a replayed "
                "diagnostic"
            ),
            "endpoint_identity": "EndpointSlice targetRef.uid joined to old/new pod UIDs",
            "endpoint_readiness": (
                "only ready=true and terminating!=true endpoints count; "
                "old withdrawal must precede the five-second preStop grace"
            ),
            "endpoint_withdrawal_observation": (
                "an endpoint-only absolute-250ms observer starts with each "
                "delete; raw sequence/schedule/fetch/observation timestamps "
                "must replay before slower pod readiness recovery"
            ),
            "kubernetes_observation_transport": (
                _KUBERNETES_OBSERVATION_TRANSPORT
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


async def _run_with_kubernetes_proxy(
    args: argparse.Namespace,
    config: GateConfig,
) -> dict[str, Any]:
    """Run a parsed CLI benchmark with one owned Kubernetes read channel."""
    missing = object()
    previous_reader = getattr(args, "_kubernetes_proxy_reader", missing)
    async with KubernetesProxyReader(args.kubectl) as reader:
        args._kubernetes_proxy_reader = reader
        try:
            return await run(args, config)
        finally:
            if previous_reader is missing:
                del args._kubernetes_proxy_reader
            else:
                args._kubernetes_proxy_reader = previous_reader


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
    parser.add_argument(
        "--frozen-sidecar",
        action="append",
        default=[],
        help=(
            "artifact-local runtime input whose safe relative path and content "
            "hash are independently verified"
        ),
    )
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
