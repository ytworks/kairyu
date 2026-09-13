#!/usr/bin/env python3
"""Measured verification for the 6 + 2 GPU DeepSeek V4.1 / Qwen3.8 tiered stack.

Every gate reads what the running stack actually did (traces, usage, finish
reasons, placement logs, L1 gauges) and writes it under the run directory.
Nothing here changes how a request is executed: routing, the ensemble DAG,
audit and refinement are Kairyu's, configured in auto-max.yaml.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = json.loads((HERE / "example.json").read_text(encoding="utf-8"))
PRODUCT_MODEL = SPEC["orchestration"]["auto_max_model"]
ENSEMBLE_MODEL = SPEC["orchestration"]["ensemble_model"]
DEEPSEEK_MODEL = SPEC["models"]["tier2"]["served_name"]
QWEN_MODEL = SPEC["models"]["tier1"]["served_name"]
QWEN_REPLICAS = int(SPEC["allocation"]["tier1"]["replicas"])
PRIMARY_ROLES: tuple[str, ...] = tuple(SPEC["orchestration"]["roles"])
PRIMARY_GENERATION_ROLES = tuple(role for role in PRIMARY_ROLES if role != "audit")
PRIMARY_VERIFICATION_ROLES = ("audit",)
ROUTE_FINAL_NODES: dict[str, str] = dict(SPEC["orchestration"]["profile_final_roles"])
TTFT_GATED_PROFILES = tuple(SPEC["orchestration"]["ttft_gated_profiles"])
JUDGE_NODE = "profile_judge"
PROJECT = SPEC["environment"].replace(".", "-")
# Files whose bytes define the served configuration.
SERVED_CONFIG_FILES = (
    "example.json",
    "compose.yaml",
    "kairyu.yaml",
    "auto-max.yaml",
    "ensemble-max.yaml",
    "router.json",
    "l1-qwen3.8-27b-vllm-chat-template.jinja",
    "webui-reasoning-effort-filter.py",
    "benchmark.py",
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
PLACEMENT_LOG_DIR = ENVIRONMENT_STORAGE / "placement-log"
QWEN_PLACEMENT_LOG = PLACEMENT_LOG_DIR / f"{QWEN_MODEL}.jsonl"
REQUEST_LOG: Path | None = None


def _api_port() -> str:
    return os.environ.get("API_PORT", str(SPEC["api_port"]))


def _l1_port() -> str:
    return os.environ.get("DEEPSEEK_L1_PORT", str(SPEC["deepseek_l1_loopback_port"]))


def _api_url() -> str:
    return f"http://127.0.0.1:{_api_port()}"


def _l1_url() -> str:
    return f"http://127.0.0.1:{_l1_port()}"


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


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


# --- HTTP helpers ------------------------------------------------------------


def _post_json(
    url: str,
    payload: dict,
    *,
    timeout_s: float,
    headers: dict[str, str] | None = None,
) -> tuple[int, object, str | None]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            status = response.status
            request_id = response.headers.get("x-request-id")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        status = error.code
        request_id = error.headers.get("x-request-id") if error.headers else None
    parsed: object = body
    if not payload.get("stream"):
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = body
    if REQUEST_LOG is not None:
        with REQUEST_LOG.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(
                    {
                        "url": url,
                        "request": payload,
                        "status": status,
                        "response": parsed,
                        "request_id": request_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return status, parsed, request_id


def _post_chat(
    payload: dict, *, timeout_s: float = 1800.0, trace: bool = True, base: str | None = None
):
    return _post_json(
        f"{base or _api_url()}/v1/chat/completions",
        payload,
        timeout_s=timeout_s,
        headers={"X-Kairyu-Trace": "1"} if trace else None,
    )


def _get_text(url: str, timeout_s: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read().decode("utf-8")


def _container(service: str) -> str:
    return f"{PROJECT}-{service}-1"


def _container_metrics(service: str) -> str:
    return subprocess.check_output(
        [
            "docker",
            "exec",
            _container(service),
            "python3",
            "-c",
            "import urllib.request; print(urllib.request.urlopen("
            '"http://127.0.0.1:8000/metrics", timeout=5).read().decode())',
        ],
        text=True,
        timeout=20,
    )


def _upstream_active_requests(service: str) -> float:
    metrics = _container_metrics(service)
    total = 0.0
    for gauge in ("running", "waiting"):
        values = re.findall(
            rf"^vllm:num_requests_{gauge}\{{[^\n]*\}} ([0-9.eE+-]+)$", metrics, re.M
        )
        if not values:
            raise ValueError(f"missing {service} {gauge} request gauge")
        total += sum(float(value) for value in values)
    return total


def _all_upstreams_active() -> dict[str, float]:
    return {
        service: _upstream_active_requests(service)
        for service in ("deepseek", *(f"qwen-{i}" for i in range(QWEN_REPLICAS)))
    }


def _gateway_outstanding() -> float:
    metrics = _get_text(f"{_api_url()}/metrics")
    values = re.findall(r"^kairyu_replica_outstanding\{[^\n]*\} ([0-9.]+)$", metrics, re.M)
    if not values:
        raise ValueError("missing kairyu_replica_outstanding gauge")
    return sum(float(value) for value in values)


# --- memory ----------------------------------------------------------------------


def memory_snapshot() -> dict:
    """GPU memory per device plus container resident/pinned host memory."""

    gpus = []
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for raw in query.splitlines():
        if raw.strip():
            index, used, total = [part.strip() for part in raw.split(",")]
            gpus.append({"gpu": int(index), "used_mib": int(used), "total_mib": int(total)})
    containers = {}
    for service in ("deepseek", *(f"qwen-{i}" for i in range(QWEN_REPLICAS)), "kairyu"):
        name = _container(service)
        try:
            stats = json.loads(
                subprocess.check_output(
                    ["docker", "stats", "--no-stream", "--format", "{{json .}}", name],
                    text=True,
                    timeout=30,
                )
            )
            status = subprocess.check_output(
                [
                    "docker",
                    "exec",
                    name,
                    "sh",
                    "-c",
                    "for p in /proc/[0-9]*; do cat $p/status 2>/dev/null; done"
                    " | grep -E '^(VmRSS|VmLck|VmPin):' ",
                ],
                text=True,
                timeout=60,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            containers[service] = {"error": str(error)}
            continue
        totals: Counter[str] = Counter()
        for line in status.splitlines():
            key, _, rest = line.partition(":")
            digits = re.findall(r"\d+", rest)
            if digits:
                totals[key] += int(digits[0])
        containers[service] = {
            "docker_mem_usage": stats.get("MemUsage"),
            "rss_kib_sum": totals.get("VmRSS", 0),
            "locked_kib_sum": totals.get("VmLck", 0),
            "pinned_kib_sum": totals.get("VmPin", 0),
        }
    return {"at": datetime.now(UTC).isoformat(), "gpus": gpus, "containers": containers}


class MemoryWatch:
    """Sample memory every ``interval_s`` during a row and keep the peaks."""

    def __init__(self, interval_s: float = 10.0):
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(memory_snapshot())
            except Exception as error:  # noqa: BLE001 - keep sampling
                self.samples.append({"error": str(error)})
            self._stop.wait(self.interval_s)

    def __enter__(self) -> MemoryWatch:
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_s + 5)

    def peaks(self) -> dict:
        gpu_peak: dict[int, int] = {}
        container_peak: dict[str, int] = {}
        for sample in self.samples:
            for row in sample.get("gpus", []):
                gpu_peak[row["gpu"]] = max(gpu_peak.get(row["gpu"], 0), row["used_mib"])
            for name, row in sample.get("containers", {}).items():
                if isinstance(row.get("rss_kib_sum"), int):
                    container_peak[name] = max(container_peak.get(name, 0), row["rss_kib_sum"])
        return {
            "samples": len(self.samples),
            "gpu_used_mib_peak": dict(sorted(gpu_peak.items())),
            "container_rss_kib_peak": dict(sorted(container_peak.items())),
        }


# --- datasets --------------------------------------------------------------------


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
        # Row identity first so another concurrency row cannot become a
        # full-prefix-cache microbenchmark.
        prompt = f"Run {namespace}, case {request}: " + " ".join(words)
        if response_instruction:
            prompt += "\n\n" + response_instruction
        rows.append({"conversations": [{"from": "human", "value": prompt}]})
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


_CODING_TASKS = (
    "Implement `parse_duration(text: str) -> int` converting strings such as "
    "'2h30m15s' (any subset and order of h/m/s units, each at most once) to "
    "total seconds. Reject empty input, unknown units, repeated units, and "
    "negative or non-integer amounts with ValueError.",
    "Implement `rle_encode(text: str) -> str` and `rle_decode(data: str) -> str` "
    "run-length coding using digits+character pairs (e.g. 'aaab' -> '3a1b'). "
    "Round-trips must be exact for any printable ASCII input without digits; "
    "rle_decode must raise ValueError on malformed input.",
    "Implement `is_balanced(text: str, pairs: dict[str, str]) -> bool` checking "
    "whether every opener in `pairs` closes with its matching closer in the "
    "correct nesting order; characters outside `pairs` are ignored.",
    "Implement class `LRUCache` with `__init__(self, capacity: int)`, "
    "`get(self, key)` returning None on miss, and `put(self, key, value)` "
    "evicting the least recently used entry beyond capacity; `get` counts as "
    "a use. Reject capacity < 1 with ValueError.",
    "Implement `to_roman(value: int) -> str` and `from_roman(text: str) -> int` "
    "for 1..3999 using standard subtractive notation; raise ValueError on "
    "out-of-range values and invalid numerals.",
    "Implement `merge_intervals(intervals: list[tuple[int, int]]) -> "
    "list[tuple[int, int]]` merging overlapping or touching [start, end] "
    "intervals and returning them sorted by start; raise ValueError when any "
    "interval has end < start.",
    "Implement `top_k_words(text: str, k: int) -> list[str]` returning the k "
    "most frequent lowercase words (split on non-letters), ties broken "
    "alphabetically; raise ValueError when k < 1.",
    "Implement `evaluate_rpn(tokens: list[str]) -> float` evaluating reverse "
    "Polish notation with + - * / and raising ValueError on malformed "
    "expressions or ZeroDivisionError on division by zero.",
)


def _coding_dataset(path: Path, requests: int, *, namespace: str) -> None:
    """Deterministic self-contained Python tasks (~1.5K prompt tokens)."""

    config = SPEC["verification"]["coding"]
    approximate_tokens = int(config["prompt_tokens_approx"])
    vocabulary = (
        "service",
        "module",
        "pipeline",
        "review",
        "constraint",
        "interface",
        "contract",
        "release",
        "quality",
        "coverage",
        "boundary",
        "failure",
        "latency",
        "budget",
        "design",
        "ticket",
    )
    filler_words = max(0, approximate_tokens - 220)
    rows = []
    for request in range(requests):
        context = " ".join(
            vocabulary[(request * 5 + position * 13) % len(vocabulary)]
            for position in range(filler_words)
        )
        task = _CODING_TASKS[request % len(_CODING_TASKS)]
        prompt = (
            f"Run {namespace}, case {request}. Project context (background "
            f"only, no action needed): {context}\n\n"
            f"Task: {task}\n\n"
            "Write a self-contained Python module named `solution` and return "
            "it in one fenced ```python code block with a brief explanation."
        )
        rows.append({"conversations": [{"from": "human", "value": prompt}]})
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


# --- benchmark rows ------------------------------------------------------------


def _bench(
    *,
    base_url: str,
    model: str,
    dataset: Path,
    requests: int,
    concurrency: int,
    max_tokens: int,
    results_dir: Path,
    log: Path,
    fixed_output: bool = False,
    reasoning_effort: str | None = None,
    stage_trace: bool = False,
    label: str = "",
) -> int:
    generation = SPEC["deepseek_generation"]
    command = [
        str(ROOT / ".venv/bin/python"),
        str(HERE / "benchmark.py"),
        "--base-url",
        base_url,
        "--model",
        model,
        "--dataset",
        str(dataset),
        "--num-requests",
        str(requests),
        "--concurrency",
        str(concurrency),
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(generation["temperature"]),
        "--top-p",
        str(generation["top_p"]),
        "--seed",
        "0",
        "--timeout",
        "14400",
        "--results-dir",
        str(results_dir),
        "--tensor-parallel",
        str(SPEC["allocation"]["tier2"]["tensor_parallel_size"]),
        "--dp-replicas",
        str(SPEC["allocation"]["tier2"]["attention_data_parallel_size"]),
        "--label",
        label,
    ]
    if fixed_output:
        command.extend(["--min-tokens", str(max_tokens), "--ignore-eos"])
    if reasoning_effort:
        command.extend(["--reasoning-effort", reasoning_effort])
    if stage_trace:
        command.append("--stage-trace")
    return _run(command, log=log, check=False)


def _row_report(row_dir: Path) -> dict | None:
    path = row_dir / "row-serving.json"
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return report if isinstance(report, dict) else None


# --- trace analysis --------------------------------------------------------------


def _events(sample: dict) -> list[dict]:
    trace = sample.get("trace")
    if not isinstance(trace, dict):
        return []
    events = trace.get("events")
    return (
        [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []
    )


def _sample_route(sample: dict) -> str | None:
    """The profile a traced sample ran: the unique route whose final unit
    succeeded as a publisher generation stage; None when ambiguous."""

    events = _events(sample)
    routes = [
        profile
        for profile, node in ROUTE_FINAL_NODES.items()
        if any(
            event.get("node") == node
            and event.get("role") == "publisher"
            and event.get("kind") == "generation"
            and event.get("status") == "success"
            for event in events
        )
    ]
    return routes[0] if len(routes) == 1 else None


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _ms(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start).total_seconds() * 1000


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, int(-(-len(ordered) * fraction // 1)) - 1)
    return round(ordered[index], 2)


def role_caps(effort: str, combined_max_tokens: int) -> dict[str, int]:
    """Per-role completion-token caps the served spec gives an ensemble
    request at ``effort`` with the caller cap ``combined_max_tokens``; used to
    flag internal roles that ended exactly at their cap (a cut-off proxy,
    because trace v2 carries no per-stage finish reason)."""

    from kairyu.dsl.loader import load_spec

    spec = load_spec(HERE / "auto-max.yaml")
    caps: dict[str, int] = {}
    for role in spec.roles:
        if role.sampling is None:
            continue
        cap = role.sampling.max_tokens
        by_effort = role.sampling.max_tokens_by_effort
        if by_effort is not None and role.reasoning_effort == "inherit":
            cap = getattr(by_effort, effort)
        if cap is None:
            continue
        ceiling = combined_max_tokens
        if role.role_type != "head":
            ceiling = min(ceiling, spec.internal_max_tokens)
        caps[role.name] = min(cap, ceiling)
    return caps


def sample_problems(
    sample: dict,
    *,
    judged: bool,
    require_head: bool,
    caps: dict[str, int],
) -> list[str]:
    """Everything wrong with one product sample (empty = valid)."""

    problems: list[str] = []
    if not sample.get("passed"):
        problems.append(f"request failed: {sample.get('error')}")
    events = _events(sample)
    if not events:
        problems.append("missing trace")
        return problems
    failed = [e for e in events if e.get("status") == "failed"]
    if failed:
        problems.append(
            "failed stages: "
            + ", ".join(f"{e.get('node')}:{(e.get('error') or {}).get('type')}" for e in failed)
        )
    route = _sample_route(sample)
    if judged:
        if not any(
            e.get("node") == JUDGE_NODE and e.get("kind") == "classification" for e in events
        ):
            problems.append("missing route-judge classification")
    if route is None:
        problems.append("no unique published route")
        return problems
    if not judged and route != "primary":
        problems.append(f"forced ensemble ran route {route}")
    if route != "primary":
        return problems
    for node in PRIMARY_GENERATION_ROLES:
        if node == "head" and not require_head:
            continue
        if not any(
            e.get("node") == node and e.get("kind") == "generation" and e.get("status") == "success"
            for e in events
        ):
            problems.append(f"missing generation stage {node}")
    for node in PRIMARY_VERIFICATION_ROLES:
        if not any(
            e.get("node") == node
            and e.get("kind") == "verification"
            and e.get("status") == "success"
            for e in events
        ):
            problems.append(f"missing verification stage {node}")
    for event in events:
        node = event.get("node")
        usage = event.get("usage") or {}
        tokens = usage.get("completion_tokens")
        cap = caps.get(node)
        if (
            event.get("status") == "success"
            and event.get("kind") in {"generation", "verification"}
            and isinstance(tokens, int)
            and cap is not None
            and node != "final"
            and tokens >= cap
        ):
            problems.append(f"{node} ended at its {cap}-token cap (attempt {event.get('attempt')})")
    return problems


def _stage_report(samples: list[dict], caps: dict[str, int]) -> dict:
    """Per-node counts, queue waits, durations, and completion tokens."""

    per_node: dict[str, dict[str, list]] = {}
    for sample in samples:
        for event in _events(sample):
            node = event.get("node")
            if not isinstance(node, str):
                continue
            row = per_node.setdefault(
                node,
                {
                    "status": [],
                    "queue_ms": [],
                    "duration_ms": [],
                    "completion_tokens": [],
                    "attempts": [],
                },
            )
            row["status"].append(str(event.get("status")))
            row["attempts"].append(event.get("attempt"))
            timing = event.get("timing") or {}
            queued, started, completed = (
                _parse_ts(timing.get("queued_at")),
                _parse_ts(timing.get("started_at")),
                _parse_ts(timing.get("completed_at")),
            )
            if (wait := _ms(queued, started)) is not None:
                row["queue_ms"].append(wait)
            if (duration := _ms(started, completed)) is not None:
                row["duration_ms"].append(duration)
            tokens = (event.get("usage") or {}).get("completion_tokens")
            if isinstance(tokens, int):
                row["completion_tokens"].append(tokens)
    report = {}
    for node, row in sorted(per_node.items()):
        tokens = row["completion_tokens"]
        report[node] = {
            "events": len(row["status"]),
            "status": dict(Counter(row["status"])),
            "max_attempt": max((a for a in row["attempts"] if isinstance(a, int)), default=None),
            "queue_wait_ms_p50": _percentile(row["queue_ms"], 0.5),
            "queue_wait_ms_p99": _percentile(row["queue_ms"], 0.99),
            "duration_ms_p50": _percentile(row["duration_ms"], 0.5),
            "duration_ms_p99": _percentile(row["duration_ms"], 0.99),
            "completion_tokens_p50": _percentile([float(t) for t in tokens], 0.5),
            "completion_tokens_max": max(tokens, default=None),
            "cap": caps.get(node),
            "cap_hits": sum(1 for t in tokens if caps.get(node) is not None and t >= caps[node]),
        }
    return report


def _route_report(samples: list[dict]) -> dict:
    by_route: dict[str, dict[str, list[float]]] = {}
    counts: Counter[str] = Counter()
    judge_ms: list[float] = []
    audit_verdicts: Counter[str] = Counter()
    refinements: list[int] = []
    for sample in samples:
        route = _sample_route(sample) or "unresolved"
        counts[route] += 1
        row = by_route.setdefault(route, {"content_ttft_ms": [], "total_ms": [], "tokens": []})
        for field in ("content_ttft_ms", "total_ms"):
            if isinstance(sample.get(field), (int, float)):
                row[field].append(float(sample[field]))
        if isinstance(sample.get("completion_tokens"), int):
            row["tokens"].append(float(sample["completion_tokens"]))
        for event in _events(sample):
            if event.get("node") == JUDGE_NODE:
                timing = event.get("timing") or {}
                if (
                    total := _ms(
                        _parse_ts(timing.get("started_at")), _parse_ts(timing.get("completed_at"))
                    )
                ) is not None:
                    judge_ms.append(total)
            # The bounded re-audit after an inconclusive verdict is traced as
            # node "audit:reverify" (Conductor contract).
            if str(event.get("node")).startswith("audit") and event.get("kind") == "verification":
                detail = event.get("detail") or {}
                if detail.get("inconclusive"):
                    audit_verdicts["inconclusive"] += 1
                elif "pass" in detail:
                    audit_verdicts["PASS" if detail.get("pass") else "FAIL"] += 1
                if detail.get("refinement_exhausted"):
                    audit_verdicts["exhausted"] += 1
        depth = max(
            (
                e.get("attempt")
                for e in _events(sample)
                if e.get("node") == "final" and isinstance(e.get("attempt"), int)
            ),
            default=0,
        )
        if route == "primary":
            refinements.append(depth)
    return {
        "routes": {
            route: {
                "requests": counts[route],
                "content_ttft_p50_ms": _percentile(row["content_ttft_ms"], 0.5),
                "content_ttft_p99_ms": _percentile(row["content_ttft_ms"], 0.99),
                "total_p50_ms": _percentile(row["total_ms"], 0.5),
                "total_p99_ms": _percentile(row["total_ms"], 0.99),
                "public_tokens_p50": _percentile(row["tokens"], 0.5),
            }
            for route, row in sorted(by_route.items())
        },
        "judge_total_ms_p50": _percentile(judge_ms, 0.5),
        "judge_total_ms_p99": _percentile(judge_ms, 0.99),
        "judged_samples": len(judge_ms),
        "audit_verdicts": dict(audit_verdicts),
        "primary_refinements": dict(Counter(refinements)),
    }


def gated_ttft_p50(routes: dict) -> float | None:
    """The largest per-route content-TTFT p50 among gated routes that served
    at least one request (the slowest gated route must clear the gate)."""

    values = [
        entry.get("content_ttft_p50_ms")
        for profile, entry in (routes.get("routes") or {}).items()
        if profile in TTFT_GATED_PROFILES
        and isinstance(entry, dict)
        and isinstance(entry.get("content_ttft_p50_ms"), (int, float))
    ]
    return max(values) if values else None


def baseline_ttft_p50(report: dict | None, requests: int) -> tuple[float | None, str]:
    """Completed-answer DeepSeek-direct TTFT p50, or (None, reason) when the
    baseline row is not a valid denominator: every request must complete with
    finish_reason stop and visible content (no all-thinking or cut-off rows)."""

    if not isinstance(report, dict):
        return None, "missing baseline report"
    samples = report.get("samples") or []
    if len(samples) != requests:
        return None, f"baseline has {len(samples)} samples, expected {requests}"
    bad = [
        s
        for s in samples
        if not s.get("passed")
        or s.get("finish_reason") != "stop"
        or not str(s.get("content") or "").strip()
        or s.get("content_ttft_ms") is None
    ]
    if bad:
        return None, f"{len(bad)} baseline samples did not complete a visible answer"
    values = [float(s["content_ttft_ms"]) for s in samples]
    return _percentile(values, 0.5), "paired_direct"


def validate_product_row(
    row_dir: Path,
    *,
    requests: int,
    judged: bool,
    require_head: bool,
    effort: str,
    combined_max_tokens: int,
) -> tuple[int, dict]:
    report = _row_report(row_dir)
    if report is None:
        return 1, {"error": "missing row-serving.json"}
    samples = report.get("samples") or []
    caps = role_caps(effort, combined_max_tokens)
    problems = {}
    if len(samples) != requests:
        problems["row"] = [f"{len(samples)} samples, expected {requests}"]
    for sample in samples:
        found = sample_problems(sample, judged=judged, require_head=require_head, caps=caps)
        if found:
            problems[str(sample.get("index"))] = found
    routes = _route_report(samples)
    stages = _stage_report(samples, caps)
    _write(row_dir / "routes.json", {"schema_version": 1, **routes})
    _write(row_dir / "stages.json", {"schema_version": 1, "caps": caps, "nodes": stages})
    _write(row_dir / "problems.json", {"schema_version": 1, "problems": problems})
    print(
        f"{row_dir.name}: routes "
        f"{json.dumps({k: v['requests'] for k, v in routes['routes'].items()})}"
    )
    if problems:
        print(f"{row_dir.name}: {sum(len(v) for v in problems.values())} problems", file=sys.stderr)
    return (1 if problems else 0), {"routes": routes, "stages": stages, "problems": problems}


# --- placement ---------------------------------------------------------------------


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
            counts[str(row.get("replica_id", row.get("replica")))] += 1
    return counts


def placement_report(
    counts: Counter[str], *, replicas: int, gated: bool, max_share_of_mean: float
) -> dict:
    total = sum(counts.values())
    mean = total / replicas if replicas and total else 0.0
    largest = max(counts.values(), default=0)
    even = total > 0 and len(counts) == replicas and largest <= max_share_of_mean * mean
    return {
        "schema_version": 1,
        "placements": total,
        "replicas": replicas,
        "per_replica": dict(sorted(counts.items())),
        "largest_share_of_mean": round(largest / mean, 3) if mean else None,
        "max_share_of_mean": max_share_of_mean,
        "gated": gated,
        "passed": even if gated else None,
    }


# --- serving matrices -------------------------------------------------------------


def _serving_matrix(
    run_dir: Path,
    *,
    model: str,
    judged: bool,
    workload: str,
) -> int:
    config = SPEC["verification"]["serving" if workload == "generic" else "coding"]
    requests = int(config["requests_per_concurrency"])
    combined = int(config["combined_max_tokens"])
    multiplier = float(config["ttft_multiplier_vs_deepseek_direct"])
    gate_cfg = SPEC["verification"]["serving"]["placement_gate"]
    effort = SPEC["orchestration"]["default_reasoning_effort"]
    run_dir.mkdir(parents=True, exist_ok=True)

    def dataset(path: Path, count: int, namespace: str) -> None:
        if workload == "generic":
            _serving_dataset(
                path,
                count,
                int(config["prompt_tokens_approx"]),
                namespace=namespace,
                response_instruction=(
                    "Synthesize a useful final answer of approximately 256 output tokens. "
                    "Return only that answer; do not expose candidates or private reasoning."
                ),
            )
        else:
            _coding_dataset(path, count, namespace=namespace)

    warmup = run_dir / "warmup.json"
    dataset(warmup, 4, f"{run_dir.parent.name}-{run_dir.name}-warmup")
    if _bench(
        base_url=f"{_api_url()}/v1",
        model=model,
        dataset=warmup,
        requests=4,
        concurrency=4,
        max_tokens=combined,
        results_dir=run_dir / "warmup",
        log=run_dir / "warmup.log",
        stage_trace=True,
        label="warmup",
    ):
        return 1
    gates: dict[str, dict] = {}
    for concurrency in config["concurrency"]:
        data = run_dir / f"{workload}-c{concurrency}.json"
        dataset(data, requests, f"{run_dir.parent.name}-{run_dir.name}-c{concurrency}")
        row_dir = run_dir / f"{workload}-c{concurrency}"
        offset = _placement_offset(QWEN_PLACEMENT_LOG)
        with MemoryWatch() as watch:
            code = _bench(
                base_url=f"{_api_url()}/v1",
                model=model,
                dataset=data,
                requests=requests,
                concurrency=concurrency,
                max_tokens=combined,
                results_dir=row_dir,
                log=run_dir / f"{workload}-c{concurrency}.log",
                stage_trace=True,
                label=f"{model}-{workload}-c{concurrency}",
            )
        _write(row_dir / "memory.json", {"peaks": watch.peaks(), "samples": watch.samples})
        placement = placement_report(
            _placement_counts(QWEN_PLACEMENT_LOG, offset),
            replicas=QWEN_REPLICAS,
            gated=concurrency >= int(gate_cfg["min_concurrency"]),
            max_share_of_mean=float(gate_cfg["max_share_of_mean"]),
        )
        _write(row_dir / "placement.json", placement)
        print(f"{row_dir.name}: qwen placements {placement['per_replica']}")
        validation, details = validate_product_row(
            row_dir,
            requests=requests,
            judged=judged,
            require_head=True,
            effort=effort,
            combined_max_tokens=combined,
        )
        if placement["passed"] is False:
            print(f"{row_dir.name}: Qwen placement is not even", file=sys.stderr)
            validation = 1
        # Paired same-topology DeepSeek-direct baseline: completed answers at
        # the same concurrency and the same (default = high) effort.
        direct_dir = run_dir / f"deepseek-direct-c{concurrency}"
        _bench(
            base_url=f"{_l1_url()}/v1",
            model=DEEPSEEK_MODEL,
            dataset=data,
            requests=requests,
            concurrency=concurrency,
            max_tokens=combined,
            results_dir=direct_dir,
            log=run_dir / f"deepseek-direct-c{concurrency}.log",
            label=f"deepseek-direct-{workload}-c{concurrency}",
        )
        direct_ttft, denominator = baseline_ttft_p50(_row_report(direct_dir), requests)
        product_ttft = gated_ttft_p50(details.get("routes") or {})
        gate: dict = {
            "product_content_ttft_p50_ms": product_ttft,
            "deepseek_direct_content_ttft_p50_ms": direct_ttft,
            "denominator": denominator,
            "multiplier": multiplier,
            "gated_profiles": list(TTFT_GATED_PROFILES),
            "routes": (details.get("routes") or {}).get("routes"),
            "judge_total_ms_p50": (details.get("routes") or {}).get("judge_total_ms_p50"),
        }
        if product_ttft is None:
            gate.update(status="not_applicable", passed=None)
        elif direct_ttft is None:
            gate.update(status="invalid_baseline", passed=False)
        else:
            gate.update(status="gated", passed=product_ttft <= multiplier * direct_ttft)
        gates[str(concurrency)] = gate
        _write(run_dir / "ttft-gate.json", {"schema_version": 1, "gates": gates})
        print(
            f"c{concurrency}: TTFT gate {gate['status']} "
            f"product={product_ttft} direct={direct_ttft}"
        )
        if code or validation or gate["passed"] is False:
            return 1
    return 0


def serving_auto_max(run_dir: Path) -> int:
    return _serving_matrix(run_dir, model=PRODUCT_MODEL, judged=True, workload="generic")


def serving_auto_max_coding(run_dir: Path) -> int:
    return _serving_matrix(run_dir, model=PRODUCT_MODEL, judged=True, workload="coding")


def serving_ensemble(run_dir: Path) -> int:
    generic = _serving_matrix(
        run_dir / "generic", model=ENSEMBLE_MODEL, judged=False, workload="generic"
    )
    coding = _serving_matrix(
        run_dir / "coding", model=ENSEMBLE_MODEL, judged=False, workload="coding"
    )
    return 1 if (generic or coding) else 0


# --- native (L1) gates ----------------------------------------------------------------


_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run."}},
            "required": ["command"],
        },
    },
}
_TOOL_SYSTEM = (
    "You are an agent operating a computer shell. Every response MUST include "
    "at least one bash tool call; never answer in plain text."
)
_TOOL_USER = "List the files in the current directory."
# A 64x64 solid-red PNG (136 bytes).
_PROBE_IMAGE_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAT0lEQVR42u3PQQkAAAgEsItz/fMY"
    "xgi+hcEKLNO+FgEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQGB"
    "ywLzk8EPlvGqjQAAAABJRU5ErkJggg=="
)


def _finite_logprobs(choice: dict) -> tuple[int, int]:
    """(values seen, non-finite values) over the choice's logprob payload."""

    seen = bad = 0
    content = (choice.get("logprobs") or {}).get("content") or []
    for token in content:
        for value in [
            token.get("logprob"),
            *(t.get("logprob") for t in token.get("top_logprobs") or []),
        ]:
            if isinstance(value, (int, float)):
                seen += 1
                if value != value or value in (float("inf"), float("-inf")):
                    bad += 1
    return seen, bad


def _tool_call_error(message: dict, finish_reason: object) -> str | None:
    if finish_reason != "tool_calls":
        return f"finish_reason is {finish_reason!r}, expected 'tool_calls'"
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return f"message.tool_calls is {calls!r}"
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or function.get("name") != "bash":
            return f"unexpected tool call {call!r}"
        try:
            arguments = json.loads(function.get("arguments") or "")
        except ValueError:
            return f"tool call arguments are not JSON: {function.get('arguments')!r}"
        if (
            not isinstance(arguments, dict)
            or not isinstance(arguments.get("command"), str)
            or not arguments["command"]
        ):
            return f"tool call arguments lack a command string: {arguments!r}"
    return None


def _rendered_prompt(chat_template_kwargs: dict | None, reasoning_effort: str | None) -> str:
    """Render one user turn through the L1 tokenizer and detokenize it, so the
    official thinking/effort encoding is observed rather than assumed."""

    payload: dict = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": "ping"}],
        "add_generation_prompt": True,
    }
    if chat_template_kwargs is not None:
        payload["chat_template_kwargs"] = chat_template_kwargs
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    status, body, _ = _post_json(f"{_l1_url()}/tokenize", payload, timeout_s=60)
    if status != 200 or not isinstance(body, dict):
        raise ValueError(f"tokenize failed: HTTP {status} {str(body)[:200]}")
    status, text, _ = _post_json(
        f"{_l1_url()}/detokenize", {"model": DEEPSEEK_MODEL, "tokens": body["tokens"]}, timeout_s=60
    )
    if status != 200 or not isinstance(text, dict):
        raise ValueError(f"detokenize failed: HTTP {status} {str(text)[:200]}")
    return str(text.get("prompt"))


def native(run_dir: Path) -> int:
    """L1 gates for the six-GPU DeepSeek candidate: startup memory, rendered
    thinking/effort encodings, finite logprobs and correct answers on every DP
    rank, tools, image, cancellation, restart, the fixed-token matrix and the
    completed-answer baseline matrix."""

    config = SPEC["verification"]["native"]
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"schema_version": 1, "cases": {}}
    failures: list[str] = []

    def record(name: str, error: str | None, detail: object = None) -> None:
        report["cases"][name] = {"passed": error is None, "error": error, "detail": detail}
        if error is not None:
            failures.append(f"{name}: {error}")
        _write(run_dir / "native.json", {**report, "passed": not failures})

    # Startup evidence: the vLLM log lines that report weights and KV capacity.
    logs = subprocess.run(
        ["docker", "logs", "--tail", "4000", _container("deepseek")],
        text=True,
        capture_output=True,
        check=False,
    )
    excerpt = [
        line
        for line in (logs.stdout + logs.stderr).splitlines()
        if re.search(
            r"(model weights|KV cache|Available|Maximum concurrency|engram|Engram"
            r"|cpu_offload|GPU blocks)",
            line,
        )
    ]
    (run_dir / "startup-excerpt.log").write_text("\n".join(excerpt) + "\n")
    _write(run_dir / "memory-idle.json", memory_snapshot())
    record(
        "startup_log_excerpt",
        None if excerpt else "no memory/KV lines in the worker log",
        len(excerpt),
    )

    # Official thinking / effort encoding, observed through /tokenize.
    renders = {}
    for name, kwargs, effort, expect, forbid in (
        ("default", None, None, "Reasoning Effort: 75", None),
        ("low", None, "low", "Reasoning Effort: 50", None),
        ("high", None, "high", "Reasoning Effort: 75", None),
        ("max", None, "max", "Reasoning Effort: 100", None),
        ("chat_mode", {"enable_thinking": False}, None, "</think>", "Reasoning Effort"),
    ):
        try:
            text = _rendered_prompt(kwargs, effort)
        except Exception as error:  # noqa: BLE001 - recorded as the failure
            record(f"render_{name}", str(error))
            continue
        renders[name] = text
        error = None
        if expect not in text:
            error = f"rendered prompt lacks {expect!r}"
        elif forbid and forbid in text:
            error = f"rendered prompt contains {forbid!r}"
        elif name == "chat_mode" and not text.rstrip().endswith("</think>"):
            error = "chat mode does not end with an immediately closed thinking span"
        record(f"render_{name}", error, text[-160:])
    (run_dir / "rendered-prompts.json").write_text(
        json.dumps(renders, indent=2, ensure_ascii=False) + "\n"
    )

    # Correctness on every DP rank: fan concurrent arithmetic probes with
    # logprobs; every value must be finite and every answer exact.
    fan = int(config["probe_fan_per_rank"]) * int(
        SPEC["allocation"]["tier2"]["attention_data_parallel_size"]
    )
    variants = [
        ("thinking_default", {}),
        ("thinking_low", {"reasoning_effort": "low"}),
        ("thinking_high", {"reasoning_effort": "high"}),
        ("thinking_max", {"reasoning_effort": "max"}),
        ("chat_mode", {"chat_template_kwargs": {"enable_thinking": False}}),
    ]
    for name, extra in variants:

        def one(case: int, extra=extra):
            payload = {
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Probe {case}: what is 17 * 19? Reply with only the integer.",
                    }
                ],
                "max_tokens": 8192,
                "logprobs": True,
                "top_logprobs": 5,
                "temperature": SPEC["deepseek_generation"]["temperature"],
                "top_p": SPEC["deepseek_generation"]["top_p"],
                **extra,
            }
            return _post_chat(payload, timeout_s=1800, trace=False, base=_l1_url())

        with concurrent.futures.ThreadPoolExecutor(max_workers=fan) as pool:
            results = list(pool.map(one, range(fan)))
        error = None
        seen_total = 0
        for status, body, _ in results:
            if status != 200 or not isinstance(body, dict):
                error = f"HTTP {status}: {str(body)[:300]}"
                break
            choice = body["choices"][0]
            message = choice["message"]
            seen, bad = _finite_logprobs(choice)
            seen_total += seen
            if bad:
                error = f"{bad} non-finite logprob values"
            elif seen == 0:
                error = "no logprob values returned"
            elif choice.get("finish_reason") != "stop":
                error = f"finish_reason {choice.get('finish_reason')!r}"
            elif (message.get("content") or "").strip() != "323":
                error = f"wrong answer {message.get('content')!r}"
            elif name == "chat_mode" and message.get("reasoning_content"):
                error = "chat mode produced reasoning_content"
            elif name != "chat_mode" and not message.get("reasoning_content"):
                error = "thinking mode produced no reasoning_content"
            if error:
                break
        record(f"probe_{name}_x{fan}", error, {"logprob_values": seen_total})

    # Tool call, image, and streaming through the native endpoint.
    status, body, _ = _post_chat(
        {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": _TOOL_SYSTEM},
                {"role": "user", "content": _TOOL_USER},
            ],
            "tools": [_BASH_TOOL],
            "max_tokens": 8192,
        },
        timeout_s=1800,
        trace=False,
        base=_l1_url(),
    )
    if status != 200 or not isinstance(body, dict):
        record("tool_call", f"HTTP {status}: {str(body)[:300]}")
    else:
        choice = body["choices"][0]
        record("tool_call", _tool_call_error(choice["message"], choice.get("finish_reason")))
    status, body, _ = _post_chat(
        {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_PROBE_IMAGE_PNG_BASE64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "What single color fills this image? Answer with one word.",
                        },
                    ],
                }
            ],
            "max_tokens": 8192,
        },
        timeout_s=1800,
        trace=False,
        base=_l1_url(),
    )
    if status != 200 or not isinstance(body, dict):
        record("image", f"HTTP {status}: {str(body)[:300]}")
    else:
        content = body["choices"][0]["message"].get("content") or ""
        record(
            "image",
            None
            if re.search(r"\bred\b", content, re.I)
            else f"answer did not name red: {content[:80]!r}",
            content[:80],
        )

    # Native cancellation: disconnect a running stream, then require every DP
    # rank's running/waiting gauges to return to zero.
    started = time.monotonic()
    request = urllib.request.Request(
        f"{_l1_url()}/v1/chat/completions",
        data=json.dumps(
            {
                "model": DEEPSEEK_MODEL,
                "stream": True,
                "max_tokens": 8192,
                "ignore_eos": True,
                "messages": [
                    {"role": "user", "content": "Count upwards from one, one number per line."}
                ],
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    saw_delta = observed = False
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
                chunk = json.loads(line[6:])
                if any(
                    (c.get("delta") or {}).get("content")
                    or (c.get("delta") or {}).get("reasoning_content")
                    for c in chunk.get("choices", [])
                ):
                    saw_delta = True
                    break
        deadline = time.monotonic() + 15
        while saw_delta and time.monotonic() < deadline and not observed:
            observed = _upstream_active_requests("deepseek") > 0
            time.sleep(0.5)
    closed_at = time.monotonic()
    released_after = None
    while time.monotonic() < closed_at + 300:
        if _upstream_active_requests("deepseek") == 0:
            released_after = time.monotonic() - closed_at
            break
        time.sleep(0.25)
    status, body, _ = _post_chat(
        {
            "model": DEEPSEEK_MODEL,
            "max_tokens": 512,
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "Reply OK."}],
        },
        timeout_s=600,
        trace=False,
        base=_l1_url(),
    )
    follow_up = (
        status == 200
        and isinstance(body, dict)
        and bool(body["choices"][0]["message"].get("content"))
    )
    record(
        "cancellation",
        None
        if (saw_delta and observed and released_after is not None and follow_up)
        else "stream did not release every rank",
        {
            "saw_delta": saw_delta,
            "observed_running": observed,
            "released_after_s": released_after,
            "follow_up": follow_up,
            "elapsed_s": time.monotonic() - started,
        },
    )

    # Fixed-token throughput matrix (reasoning tokens count; not a quality row).
    fixed = int(config["fixed_output_tokens"])
    requests = int(config["requests_per_concurrency"])
    prompt_tokens = int(SPEC["verification"]["serving"]["prompt_tokens_approx"])
    for concurrency in config["concurrency"]:
        data = run_dir / f"fixed-c{concurrency}.json"
        _serving_dataset(
            data, requests, prompt_tokens, namespace=f"{run_dir.name}-fixed-c{concurrency}"
        )
        row_dir = run_dir / f"fixed-c{concurrency}"
        with MemoryWatch() as watch:
            code = _bench(
                base_url=f"{_l1_url()}/v1",
                model=DEEPSEEK_MODEL,
                dataset=data,
                requests=requests,
                concurrency=concurrency,
                max_tokens=fixed,
                results_dir=row_dir,
                log=run_dir / f"fixed-c{concurrency}.log",
                fixed_output=True,
                label=f"native-fixed-c{concurrency}",
            )
        _write(row_dir / "memory.json", {"peaks": watch.peaks(), "samples": watch.samples})
        summary = (_row_report(row_dir) or {}).get("summary")
        record(f"fixed_c{concurrency}", None if code == 0 else "fixed-token row failed", summary)

    # Completed-answer matrix at the default (high) effort: the standalone
    # DeepSeek-direct envelope; the serving gates re-run a paired copy per row.
    combined = int(SPEC["verification"]["serving"]["combined_max_tokens"])
    for workload in ("generic", "coding"):
        for concurrency in config["concurrency"]:
            data = run_dir / f"{workload}-c{concurrency}.json"
            if workload == "generic":
                _serving_dataset(
                    data,
                    requests,
                    prompt_tokens,
                    namespace=f"{run_dir.name}-{workload}-c{concurrency}",
                    response_instruction=(
                        "Synthesize a useful final answer of approximately 256 output "
                        "tokens. Return only that answer."
                    ),
                )
            else:
                _coding_dataset(
                    data, requests, namespace=f"{run_dir.name}-{workload}-c{concurrency}"
                )
            row_dir = run_dir / f"{workload}-c{concurrency}"
            _bench(
                base_url=f"{_l1_url()}/v1",
                model=DEEPSEEK_MODEL,
                dataset=data,
                requests=requests,
                concurrency=concurrency,
                max_tokens=combined,
                results_dir=row_dir,
                log=run_dir / f"{workload}-c{concurrency}.log",
                label=f"native-{workload}-c{concurrency}",
            )
            ttft, reason = baseline_ttft_p50(_row_report(row_dir), requests)
            summary = (_row_report(row_dir) or {}).get("summary")
            record(
                f"completed_{workload}_c{concurrency}",
                None if ttft is not None else reason,
                summary,
            )

    # Service restart: the DeepSeek service must come back healthy and answer.
    restart_started = time.monotonic()
    _run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(HERE),
            "--file",
            str(HERE / "compose.yaml"),
            "restart",
            "deepseek",
        ],
        check=False,
        env=_compose_env(),
    )
    healthy = False
    while time.monotonic() < restart_started + 3600:
        state = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", _container("deepseek")],
            text=True,
            capture_output=True,
            check=False,
        )
        if state.stdout.strip() == "healthy":
            healthy = True
            break
        time.sleep(10)
    status, body, _ = (
        _post_chat(
            {
                "model": DEEPSEEK_MODEL,
                "max_tokens": 512,
                "messages": [{"role": "user", "content": "Reply OK."}],
            },
            timeout_s=600,
            trace=False,
            base=_l1_url(),
        )
        if healthy
        else (0, None, None)
    )
    record(
        "restart",
        None
        if (healthy and status == 200)
        else "DeepSeek service did not return healthy and answering",
        {"seconds": time.monotonic() - restart_started},
    )
    for line in failures:
        print(f"native: {line}", file=sys.stderr)
    print(f"native: {'PASS' if not failures else 'FAIL'} ({len(report['cases'])} cases)")
    return 1 if failures else 0


def _compose_env() -> dict[str, str]:
    """The launcher's Compose environment (paths, ports) for restart commands."""

    import importlib.util

    spec = importlib.util.spec_from_file_location("v41_tiered_control", HERE / "control.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._compose_env()


# --- public gates -------------------------------------------------------------------------


def _tool_request(model: str, **overrides) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _TOOL_SYSTEM},
            {"role": "user", "content": _TOOL_USER},
        ],
        "tools": [_BASH_TOOL],
        "max_tokens": 8192,
    }
    payload.update(overrides)
    return payload


def tool_calling(run_dir: Path) -> int:
    """OpenAI tool-call contract on both public models within the 900 s agent
    turn: auto call, tool-result turn, streaming, explicit effort. On the
    forced ensemble every turn must also trace the audit verdict."""

    run_dir.mkdir(parents=True, exist_ok=True)
    turn_timeout = float(SPEC["verification"]["agent_turn_timeout_s"])
    report: dict = {"schema_version": 1, "cases": {}}
    failures: list[str] = []

    def record(name: str, error: str | None, detail: object = None) -> None:
        report["cases"][name] = {"passed": error is None, "error": error, "detail": detail}
        if error is not None:
            failures.append(f"{name}: {error}")
        _write(run_dir / "tool-calling.json", {**report, "passed": not failures})

    for model in (PRODUCT_MODEL, ENSEMBLE_MODEL):
        forced = model == ENSEMBLE_MODEL
        fan = 2 * QWEN_REPLICAS
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=fan) as pool:
            results = list(
                pool.map(
                    lambda _, model=model: _post_chat(_tool_request(model), timeout_s=turn_timeout),
                    range(fan),
                )
            )
        first_message = None
        error = None
        for status, body, _ in results:
            if status != 200 or not isinstance(body, dict):
                error = f"HTTP {status}: {str(body)[:300]}"
                break
            choice = body["choices"][0]
            error = _tool_call_error(choice["message"], choice.get("finish_reason"))
            if error is None and forced:
                events = (body.get("kairyu_trace_v2") or {}).get("events") or []
                if not any(
                    e.get("node") == "audit"
                    and e.get("kind") == "verification"
                    and e.get("status") == "success"
                    for e in events
                ):
                    error = "forced ensemble tool turn traced no audit verdict"
            if error:
                break
            first_message = choice["message"]
        record(f"{model}/auto_tool_call_x{fan}", error, {"seconds": time.monotonic() - started})
        if first_message is None:
            continue
        call = first_message["tool_calls"][0]
        call_id = call.get("id") or "call_0"
        started = time.monotonic()
        status, body, _ = _post_chat(
            _tool_request(
                model,
                messages=[
                    {"role": "system", "content": _TOOL_SYSTEM},
                    {"role": "user", "content": _TOOL_USER},
                    {
                        "role": "assistant",
                        "content": first_message.get("content"),
                        "reasoning_content": first_message.get("reasoning_content"),
                        "tool_calls": [
                            {"id": call_id, "type": "function", "function": dict(call["function"])}
                        ],
                    },
                    {"role": "tool", "tool_call_id": call_id, "content": "README.md\nsrc\ntests\n"},
                ],
            ),
            timeout_s=turn_timeout,
        )
        if status != 200 or not isinstance(body, dict):
            record(f"{model}/tool_result_turn", f"HTTP {status}: {str(body)[:300]}")
        else:
            choice = body["choices"][0]
            record(
                f"{model}/tool_result_turn",
                _tool_call_error(choice["message"], choice.get("finish_reason")),
                {"seconds": time.monotonic() - started},
            )
        status, body, _ = _post_chat(
            _tool_request(model, stream=True), timeout_s=turn_timeout, trace=False
        )
        if status != 200 or not isinstance(body, str):
            record(f"{model}/streamed_tool_call", f"HTTP {status}: {str(body)[:300]}")
        else:
            calls: dict[int, dict[str, list[str]]] = {}
            finish = None
            for line in body.splitlines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                for choice in chunk.get("choices", ()):
                    finish = choice.get("finish_reason") or finish
                    for delta_call in (choice.get("delta") or {}).get("tool_calls") or ():
                        slot = calls.setdefault(
                            int(delta_call.get("index", 0)), {"name": [], "arguments": []}
                        )
                        function = delta_call.get("function") or {}
                        if function.get("name"):
                            slot["name"].append(function["name"])
                        if function.get("arguments"):
                            slot["arguments"].append(function["arguments"])
            message = {
                "tool_calls": [
                    {"function": {"name": "".join(s["name"]), "arguments": "".join(s["arguments"])}}
                    for _, s in sorted(calls.items())
                ]
                or None
            }
            record(f"{model}/streamed_tool_call", _tool_call_error(message, finish))
        started = time.monotonic()
        status, body, _ = _post_chat(
            _tool_request(model, reasoning_effort="high"), timeout_s=turn_timeout
        )
        if status != 200 or not isinstance(body, dict):
            record(f"{model}/thinking_tool_call", f"HTTP {status}: {str(body)[:300]}")
        else:
            choice = body["choices"][0]
            record(
                f"{model}/thinking_tool_call",
                _tool_call_error(choice["message"], choice.get("finish_reason")),
                {"seconds": time.monotonic() - started},
            )
    for line in failures:
        print(f"tool-calling: {line}", file=sys.stderr)
    print(f"tool-calling: {'PASS' if not failures else 'FAIL'} ({len(report['cases'])} cases)")
    return 1 if failures else 0


def vision(run_dir: Path) -> int:
    """Image requests on both public models. The forced-ensemble JSON case
    disables the head, so its whole answer comes from DeepSeek: naming the
    probe colour proves the original image reached the DeepSeek roles."""

    run_dir.mkdir(parents=True, exist_ok=True)
    max_tokens = int(SPEC["verification"]["vision"]["max_tokens"])
    report: dict = {"schema_version": 1, "cases": {}}
    failures: list[str] = []

    def record(name: str, error: str | None, detail: object = None) -> None:
        report["cases"][name] = {"passed": error is None, "error": error, "detail": detail}
        if error is not None:
            failures.append(f"{name}: {error}")
        _write(run_dir / "vision.json", {**report, "passed": not failures})

    def image_request(model: str, text: str, **overrides) -> dict:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_PROBE_IMAGE_PNG_BASE64}"
                            },
                        },
                        {"type": "text", "text": text},
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }
        payload.update(overrides)
        return payload

    cases = [
        (
            f"{PRODUCT_MODEL}/plain",
            image_request(
                PRODUCT_MODEL, "What single color fills this image? Answer with one word."
            ),
            None,
        ),
        (
            f"{ENSEMBLE_MODEL}/plain",
            image_request(
                ENSEMBLE_MODEL, "What single color fills this image? Answer with one word."
            ),
            "primary",
        ),
        (
            f"{ENSEMBLE_MODEL}/json_headless",
            image_request(
                ENSEMBLE_MODEL,
                'Return a JSON object with one key "color" naming the single color '
                "that fills this image.",
                response_format={"type": "json_object"},
            ),
            "primary",
        ),
    ]
    for name, payload, expected_route in cases:
        started = time.monotonic()
        status, body, _ = _post_chat(payload, timeout_s=3600)
        if status != 200 or not isinstance(body, dict):
            record(name, f"HTTP {status}: {str(body)[:300]}")
            continue
        choice = body["choices"][0]
        content = choice["message"].get("content") or ""
        error = None
        if "red" not in content.lower():
            error = f"answer did not name red: {content[:120]!r}"
        elif choice.get("finish_reason") != "stop":
            error = f"finish_reason {choice.get('finish_reason')!r}"
        events = (body.get("kairyu_trace_v2") or {}).get("events") or []
        route = _sample_route({"trace": body.get("kairyu_trace_v2")})
        if error is None and expected_route and route != expected_route:
            error = f"route {route!r}, expected {expected_route!r}"
        if error is None and expected_route == "primary":
            missing = [
                n
                for n in PRIMARY_GENERATION_ROLES
                if n != "head"
                and not any(e.get("node") == n and e.get("status") == "success" for e in events)
            ]
            if missing:
                error = f"ensemble skipped {missing}"
        record(
            name,
            error,
            {"content": content[:120], "route": route, "seconds": time.monotonic() - started},
        )
    for line in failures:
        print(f"vision: {line}", file=sys.stderr)
    print(f"vision: {'PASS' if not failures else 'FAIL'} ({len(report['cases'])} cases)")
    return 1 if failures else 0


def _wait_all_idle(bound_s: float) -> tuple[float | None, dict]:
    """Seconds until the gateway and every L1 report zero in-flight work."""

    started = time.monotonic()
    last: dict = {}
    while time.monotonic() < started + bound_s:
        try:
            last = {"gateway_outstanding": _gateway_outstanding(), **_all_upstreams_active()}
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            last = {"error": str(error)}
            time.sleep(1)
            continue
        if all(v == 0 for k, v in last.items() if k != "error"):
            return time.monotonic() - started, last
        time.sleep(1)
    return None, last


def cancellation(run_dir: Path) -> int:
    """Public disconnects on the forced ensemble: right after the first public
    bytes (parallel generation running) and later while the remainder is still
    withheld (synthesis/final/audit running). Records how long every worker
    takes to go idle; the bound is generous and the exact time is evidence."""

    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"schema_version": 1, "cases": {}}
    failures: list[str] = []
    bound_s = 900.0

    def record(name: str, error: str | None, detail: object = None) -> None:
        report["cases"][name] = {"passed": error is None, "error": error, "detail": detail}
        if error is not None:
            failures.append(f"{name}: {error}")
        _write(run_dir / "cancellation.json", {**report, "passed": not failures})

    prompt = (
        "Write a thorough, well-argued essay (at least 1500 words) comparing three "
        "different strategies for migrating a monolithic service to microservices, "
        "with concrete trade-offs and a final recommendation."
    )
    for name, hold_s in (("after_first_public_bytes", 0.0), ("while_remainder_withheld", 90.0)):
        request = urllib.request.Request(
            f"{_api_url()}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": ENSEMBLE_MODEL,
                    "stream": True,
                    "max_tokens": 65536,
                    "messages": [{"role": "user", "content": prompt}],
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        saw_public = False
        busy_before = {}
        opened = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                for line in response:
                    if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
                        chunk = json.loads(line[6:])
                        if any(
                            (c.get("delta") or {}).get("content") for c in chunk.get("choices", [])
                        ):
                            saw_public = True
                            break
                deadline = time.monotonic() + hold_s
                while time.monotonic() < deadline:
                    time.sleep(5)
                busy_before = _all_upstreams_active()
        except OSError as error:
            record(name, f"stream failed before disconnect: {error}")
            continue
        closed = time.monotonic()
        released_after, last = _wait_all_idle(bound_s)
        status, body, _ = _post_chat(
            {
                "model": PRODUCT_MODEL,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            },
            timeout_s=900,
            trace=False,
        )
        follow_up = (
            status == 200
            and isinstance(body, dict)
            and bool(body["choices"][0]["message"].get("content"))
        )
        error = None
        if not saw_public:
            error = "no public bytes before disconnect"
        elif released_after is None:
            error = f"workers still busy {bound_s}s after disconnect: {last}"
        elif not follow_up:
            error = "follow-up request failed after cancellation"
        record(
            name,
            error,
            {
                "stream_open_s": closed - opened,
                "busy_before_disconnect": busy_before,
                "released_after_s": released_after,
                "final_gauges": last,
                "follow_up": follow_up,
            },
        )
    for line in failures:
        print(f"cancellation: {line}", file=sys.stderr)
    print(f"cancellation: {'PASS' if not failures else 'FAIL'} ({len(report['cases'])} cases)")
    return 1 if failures else 0


def restart(run_dir: Path) -> int:
    """Normal restart of every service; readiness and one answer on each
    public model afterwards, with the launcher's readiness gate re-run."""

    import importlib.util

    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = _compose_env()
    code = _run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(HERE),
            "--file",
            str(HERE / "compose.yaml"),
            "restart",
        ],
        check=False,
        env=env,
    )
    ready = False
    while time.monotonic() < started + 7200:
        try:
            if json.loads(_get_text(f"{_api_url()}/readyz")).get("status") == "ready":
                ready = True
                break
        except (OSError, ValueError):
            pass
        time.sleep(10)
    result = {
        "compose_restart_exit": code,
        "ready_after_s": time.monotonic() - started if ready else None,
        "models": {},
    }
    if ready:
        spec = importlib.util.spec_from_file_location(
            "v41_tiered_control_restart", HERE / "control.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            module._validate_ready(_api_url(), f"{_l1_url()}/tokenize")
            result["validate_ready"] = "ok"
        except SystemExit as error:
            result["validate_ready"] = str(error)
        for model in (PRODUCT_MODEL, ENSEMBLE_MODEL):
            status, body, _ = _post_chat(
                {
                    "model": model,
                    "max_tokens": 4096,
                    "messages": [
                        {"role": "user", "content": "What is 17 * 19? Reply with only the integer."}
                    ],
                },
                timeout_s=3600,
            )
            content = (
                body["choices"][0]["message"].get("content")
                if status == 200 and isinstance(body, dict)
                else None
            )
            result["models"][model] = {
                "status": status,
                "content": (content or "")[:80],
                "route": _sample_route({"trace": body.get("kairyu_trace_v2")})
                if isinstance(body, dict)
                else None,
            }
    passed = (
        ready
        and code == 0
        and result.get("validate_ready") == "ok"
        and all(
            row["status"] == 200 and "323" in row["content"] for row in result["models"].values()
        )
    )
    result["passed"] = passed
    _write(run_dir / "restart.json", result)
    print(
        f"restart: {'PASS' if passed else 'FAIL'} "
        f"{json.dumps({k: v for k, v in result.items() if k != 'models'})}"
    )
    return 0 if passed else 1


def issue_599(run_dir: Path) -> int:
    """Replay the saved Issue #599 request (path in ISSUE_599_REQUEST_PATH):
    a long tool-bearing conversation with no caller max_tokens must complete
    through the judged product instead of an upstream context-length 400."""

    run_dir.mkdir(parents=True, exist_ok=True)
    path = os.environ.get("ISSUE_599_REQUEST_PATH")
    if not path or not Path(path).is_file():
        print("issue-599: set ISSUE_599_REQUEST_PATH to the saved request JSON", file=sys.stderr)
        return 1
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    body = saved.get("request", saved) if isinstance(saved, dict) else saved
    if not isinstance(body, dict) or "messages" not in body:
        print("issue-599: the saved file must contain an OpenAI chat request", file=sys.stderr)
        return 1
    results = {}
    for name, overrides in (("as_saved", {}), ("explicit_cap_4096", {"max_tokens": 4096})):
        payload = {k: v for k, v in body.items() if k not in {"stream", "stream_options"}}
        payload["model"] = PRODUCT_MODEL
        payload.pop("max_tokens", None)
        payload.update(overrides)
        started = time.monotonic()
        status, response, request_id = _post_chat(payload, timeout_s=7200)
        choice = response["choices"][0] if status == 200 and isinstance(response, dict) else {}
        message = choice.get("message") or {}
        results[name] = {
            "status": status,
            "seconds": time.monotonic() - started,
            "request_id": request_id,
            "finish_reason": choice.get("finish_reason"),
            "content_chars": len(message.get("content") or ""),
            "tool_calls": len(message.get("tool_calls") or []),
            "usage": response.get("usage") if isinstance(response, dict) else None,
            "route": _sample_route({"trace": response.get("kairyu_trace_v2")})
            if isinstance(response, dict)
            else None,
            "error": response if status != 200 else None,
            "passed": status == 200
            and choice.get("finish_reason") in {"stop", "tool_calls"}
            and bool((message.get("content") or "").strip() or message.get("tool_calls")),
        }
        _write(
            run_dir / "issue-599.json",
            {"schema_version": 1, "messages": len(body["messages"]), "results": results},
        )
    passed = all(row["passed"] for row in results.values())
    print(
        f"issue-599: {'PASS' if passed else 'FAIL'} "
        f"{json.dumps({k: (v['status'], v['route']) for k, v in results.items()})}"
    )
    return 0 if passed else 1


def long_input(run_dir: Path) -> int:
    """Inputs near the documented boundaries: a Qwen-boundary conversation on
    the forced ensemble, and DeepSeek-direct retrieval up to near 1M tokens."""

    import secrets

    from tokenizers import Tokenizer

    run_dir.mkdir(parents=True, exist_ok=True)
    config = SPEC["verification"]["long_input"]
    reports = []
    qwen_tokenizer = Tokenizer.from_file(
        str(
            STORAGE_ROOT
            / "model-volumes"
            / SPEC["storage"]["qwen_model_environment"]
            / "models"
            / SPEC["models"]["tier1"]["slug"]
            / "tokenizer.json"
        )
    )
    filler = " filler"
    key = "K" + secrets.token_hex(12).upper()
    target = int(config["qwen_boundary_prompt_tokens"])
    prefix = "Read this log and recover its archive key.\n"
    needle = f"\nThe archive key is {key}.\n"
    suffix = "\nReturn only the archive key."
    overhead = len(qwen_tokenizer.encode(prefix + needle + suffix, add_special_tokens=False).ids)
    repeats = target - overhead
    prompt = prefix + filler * (repeats // 2) + needle + filler * (repeats - repeats // 2) + suffix
    started = time.monotonic()
    status, body, _ = _post_chat(
        {
            "model": ENSEMBLE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
        },
        timeout_s=14400,
    )
    choice = body["choices"][0] if status == 200 and isinstance(body, dict) else {}
    content = (choice.get("message") or {}).get("content") or ""
    events = ((body.get("kairyu_trace_v2") if isinstance(body, dict) else None) or {}).get(
        "events"
    ) or []
    reports.append(
        {
            "case": "ensemble_qwen_boundary",
            "target_input_tokens": target,
            "status": status,
            "usage": body.get("usage") if isinstance(body, dict) else None,
            "key_found": key in content,
            "finish_reason": choice.get("finish_reason"),
            "failed_stages": [e.get("node") for e in events if e.get("status") == "failed"],
            "total_s": time.monotonic() - started,
            "passed": status == 200
            and key in content
            and not any(e.get("status") == "failed" for e in events),
            "error": body if status != 200 else None,
        }
    )
    _write(run_dir / "long-input.json", reports)
    deepseek_tokenizer = Tokenizer.from_file(
        str(
            STORAGE_ROOT
            / "model-volumes"
            / SPEC["storage"]["deepseek_model_environment"]
            / "models"
            / SPEC["models"]["tier2"]["slug"]
            / "tokenizer.json"
        )
    )
    for target in config["deepseek_native_targets"]:
        key = "K" + secrets.token_hex(12).upper()
        needle = f"\nThe archive key is {key}.\n"
        overhead = len(
            deepseek_tokenizer.encode(prefix + needle + suffix, add_special_tokens=False).ids
        )
        repeats = target - overhead
        prompt = (
            prefix + filler * (repeats // 2) + needle + filler * (repeats - repeats // 2) + suffix
        )
        started = time.monotonic()
        status, body, _ = _post_chat(
            {
                "model": DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8192,
            },
            timeout_s=7200,
            trace=False,
            base=_l1_url(),
        )
        choice = body["choices"][0] if status == 200 and isinstance(body, dict) else {}
        content = (choice.get("message") or {}).get("content") or ""
        reports.append(
            {
                "case": f"deepseek_native_{target}",
                "target_input_tokens": target,
                "status": status,
                "usage": body.get("usage") if isinstance(body, dict) else None,
                "key_found": content.strip() == key,
                "finish_reason": choice.get("finish_reason"),
                "total_s": time.monotonic() - started,
                "passed": status == 200
                and content.strip() == key
                and choice.get("finish_reason") == "stop",
            }
        )
        _write(run_dir / "long-input.json", reports)
    passed = all(row["passed"] for row in reports)
    print(
        f"long-input: {'PASS' if passed else 'FAIL'} {[(r['case'], r['passed']) for r in reports]}"
    )
    return 0 if passed else 1


def browser(run_dir: Path) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault(
        "WEBUI_BASE_URL",
        f"http://127.0.0.1:{os.environ.get('CHAT_UI_PORT', SPEC['webui']['port'])}",
    )
    code = _run(
        [str(HERE / "browser-smoke.sh")], log=run_dir / "browser-smoke.log", check=False, env=env
    )
    _write(run_dir / "browser.json", {"passed": code == 0, "exit_code": code})
    return code


# --- attestation ----------------------------------------------------------------------------


def _served_config_sha256() -> str:
    digest = hashlib.sha256()
    for name in SERVED_CONFIG_FILES:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((HERE / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_evidence() -> dict:
    """Reject measuring a stale runtime: images and commands must match the
    pins, and the gateway must serve the mounted files on disk."""

    import yaml

    compose = yaml.safe_load((HERE / "compose.yaml").read_text())
    rows = {}
    for service, pin in (
        ("deepseek", SPEC["vllm"]["deepseek"]["image_id"]),
        *((f"qwen-{i}", SPEC["vllm"]["qwen"]["image_id"]) for i in range(QWEN_REPLICAS)),
    ):
        container = json.loads(
            subprocess.check_output(["docker", "inspect", _container(service)], text=True)
        )[0]
        if container["Image"] != pin:
            raise ValueError(f"{service} runs image {container['Image']}, evidence pins {pin}")
        if container["Config"]["Cmd"] != compose["services"][service]["command"]:
            raise ValueError(f"{service} command differs from compose.yaml")
        rows[service] = {
            "image_id": container["Image"],
            "started_at": container["State"]["StartedAt"],
            "devices": container["HostConfig"]["DeviceRequests"],
        }
    gateway = _container("kairyu")
    mounted = {}
    for name in ("kairyu.yaml", "auto-max.yaml", "ensemble-max.yaml", "router.json"):
        served = subprocess.check_output(["docker", "exec", gateway, "cat", f"/etc/kairyu/{name}"])
        if served != (HERE / name).read_bytes():
            raise ValueError(f"gateway serves a different {name} than the checkout")
        mounted[name] = hashlib.sha256(served).hexdigest()
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return {"l1": rows, "gateway_files_sha256": mounted, "git_commit": git.stdout.strip()}


_VERIFICATIONS = {
    "native": (
        native,
        "six-GPU DeepSeek L1 gates: memory, encodings, finite probes per DP rank, tools, "
        "image, cancellation, fixed/completed matrices, restart",
    ),
    "serving-auto-max": (
        serving_auto_max,
        "judged product, generic workload c1/8/16/32 x 32 with paired DeepSeek-direct TTFT gate",
    ),
    "serving-auto-max-coding": (
        serving_auto_max_coding,
        "judged product, coding workload c1/8/16/32 x 32 with paired DeepSeek-direct TTFT gate",
    ),
    "serving-ensemble": (
        serving_ensemble,
        "forced five-candidate ensemble (kairyu-ensemble-max), generic + coding matrices",
    ),
    "tool-calling": (
        tool_calling,
        "OpenAI bash-tool agent contract on both public models within the 900 s turn",
    ),
    "vision": (
        vision,
        "image requests on both public models; headless JSON case proves DeepSeek saw the image",
    ),
    "cancellation": (
        cancellation,
        "public disconnects during early generation and during the withheld remainder; "
        "time to idle",
    ),
    "restart": (
        restart,
        "normal restart of every service, readiness gate and one answer per public model",
    ),
    "issue-599": (
        issue_599,
        "replay the saved Issue #599 long tool conversation through the judged product",
    ),
    "long-input": (
        long_input,
        "Qwen-boundary ensemble input and DeepSeek-direct retrieval up to near 1M tokens",
    ),
    "browser": (
        browser,
        "Open WebUI browser gate through the shared scripts/webui_browser_smoke.mjs",
    ),
}


def main() -> None:
    global REQUEST_LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verification", choices=(*_VERIFICATIONS, "list"))
    parser.add_argument("--run-id")
    parser.add_argument("--no-start", action="store_true")
    parser.add_argument(
        "--skip-runtime-attestation",
        action="store_true",
        help="native rows before the gateway exists",
    )
    args = parser.parse_args()
    if args.verification == "list":
        for name, (_, description) in _VERIFICATIONS.items():
            print(f"{name}  {description}")
        return
    _ensure_environment(args.no_start)
    run_dir = RESULTS_ROOT / (args.run_id or _run_id())
    run_dir.mkdir(parents=True, exist_ok=True)
    REQUEST_LOG = run_dir / "requests.jsonl"
    runtime = None if args.skip_runtime_attestation else runtime_evidence()
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "started_at": datetime.now(UTC).isoformat(),
        "requested": args.verification,
        "served_config_sha256": _served_config_sha256(),
        "spec": SPEC,
        "runtime": runtime,
    }
    _write(run_dir / "run.json", manifest)
    target, _ = _VERIFICATIONS[args.verification]
    try:
        code = target(run_dir / args.verification)
    except Exception as error:  # noqa: BLE001 - recorded, then non-zero exit
        print(f"{args.verification} failed: {error}", file=sys.stderr)
        code = 1
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["exit_codes"] = {args.verification: code}
    _write(run_dir / "run.json", manifest)
    print(f"artifacts: {run_dir}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
