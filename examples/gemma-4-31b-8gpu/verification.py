#!/usr/bin/env python3
"""GPU verification for Gemma 4 31B chat and EmbeddingGemma 2 replicas on eight GPUs.

``l1`` is the correctness gate every L1 candidate must pass before it is
measured: thinking on and off rendered by the checkpoint's own template, exact
finite answers from every chat replica, the checkpoint's sampling defaults,
and every embedding replica against the sentence-transformers reference. The
public gates run through Kairyu (L2 replica pools, L3 API) on ``/v1/responses``
and ``/v1/embeddings``, the Chat UI's APIs; L1-direct checks use vLLM's own
APIs inside each replica's container.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import control

HERE = control.HERE
ROOT = control.ROOT
SPEC = control.SPEC
SERVED_MODEL = control.CHAT_MODEL
EMBED_MODEL = control.EMBED_MODEL
CHAT_CONTAINERS = [f"{control.PROJECT}-{service}-1" for service in control.CHAT_SERVICES]
EMBED_CONTAINERS = [f"{control.PROJECT}-{service}-1" for service in control.EMBED_SERVICES]
CHAT_REPLICAS = len(CHAT_CONTAINERS)
EMBED_REPLICAS = len(EMBED_CONTAINERS)
SERVED_CONFIG_FILES = (
    HERE / "example.json",
    HERE / "compose.yaml",
    HERE / "kairyu.yaml",
    HERE / "webui-responses-filter.py",
    HERE / "webui-file-download-tool.py",
)
ENVIRONMENT_STORAGE = control.environment_storage()
RESULTS_ROOT = Path(
    os.environ.get("VERIFICATION_RESULTS_ROOT", ENVIRONMENT_STORAGE / "verification-results")
)
PLACEMENT_DIR = ENVIRONMENT_STORAGE / "placement-log"
CHAT_PLACEMENT_LOG = PLACEMENT_DIR / Path(SPEC["pool"]["placement_log"]).name
EMBED_PLACEMENT_LOG = PLACEMENT_DIR / Path(SPEC["embedding_pool"]["placement_log"]).name
REQUEST_LOG: Path | None = None


def _api_url() -> str:
    return f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}"


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# --- HTTP helpers ------------------------------------------------------------------


def post_responses(payload: dict, *, timeout_s: float = 900.0) -> tuple[int, object, str | None]:
    """Send a chat-shaped gate payload to Kairyu's /v1/responses.

    The reply comes back in the chat shape the gates check (a dict, or chat
    SSE text for a stream); an error reply stays raw text. The request log
    keeps the Responses request and reply exactly as they were exchanged.
    """
    body_sent = control.responses_request(payload)
    request = urllib.request.Request(
        f"{_api_url()}/v1/responses",
        data=json.dumps(body_sent).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
            status, request_id = response.status, response.headers.get("x-request-id")
            body: object = (
                control.chat_stream_view(raw)
                if payload.get("stream")
                else control.chat_view(json.loads(raw))
            )
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read().decode("utf-8", "replace")
        body = raw
        request_id = error.headers.get("x-request-id") if error.headers else None
    if REQUEST_LOG is not None:
        row = {"request": body_sent, "status": status, "response": raw, "request_id": request_id}
        with REQUEST_LOG.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return status, body, request_id


def post_embeddings(texts: list[str], *, timeout_s: float = 120.0) -> tuple[int, object]:
    """Kairyu's /v1/embeddings; the reply as JSON, or the raw error text."""
    request = urllib.request.Request(
        f"{_api_url()}/v1/embeddings",
        data=json.dumps({"model": EMBED_MODEL, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")
    except (OSError, urllib.error.URLError) as error:
        return 0, repr(error)


def open_responses_stream(payload: dict, *, timeout_s: float = 1800):
    """Open a streamed /v1/responses request for a chat-shaped payload."""
    request = urllib.request.Request(
        f"{_api_url()}/v1/responses",
        data=json.dumps(control.responses_request({**payload, "stream": True})).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout_s)


_REASONING_DELTAS = ("response.reasoning_text.delta", "response.reasoning_summary_text.delta")


def stream_event(line: bytes) -> dict | None:
    """One Responses SSE ``data:`` line as an event (None for other lines)."""
    if not line.startswith(b"data:") or not line[5:].strip():
        return None
    return json.loads(line[5:])


_L1_PROGRAM = r"""
import json, sys, urllib.error, urllib.request
method, path = sys.argv[1:3]
data = sys.stdin.buffer.read() if method == "POST" else None
request = urllib.request.Request("http://127.0.0.1:8000" + path, data=data,
                                 headers={"Content-Type": "application/json"}, method=method)
try:
    with urllib.request.urlopen(request, timeout=1800) as response:
        sys.stdout.write(json.dumps({"status": response.status, "body": response.read().decode()}))
except urllib.error.HTTPError as error:
    body = error.read().decode("utf-8", "replace")
    sys.stdout.write(json.dumps({"status": error.code, "body": body}))
"""


def l1_call(
    container: str, method: str, path: str, payload: dict | None = None
) -> tuple[int, str]:
    """Reach one vLLM replica inside its container (replicas publish no host port)."""
    completed = subprocess.run(
        ["docker", "exec", "-i", container, "python3", "-c", _L1_PROGRAM, method, path],
        input=json.dumps(payload).encode() if payload is not None else b"",
        capture_output=True,
        timeout=1900,
        check=True,
    )
    result = json.loads(completed.stdout)
    return int(result["status"]), result["body"]


def l1_json(container: str, path: str, payload: dict) -> dict:
    status, body = l1_call(container, "POST", path, payload)
    if status != 200:
        raise ValueError(f"L1 {container} {path} returned HTTP {status}: {body[:300]}")
    return json.loads(body)


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norms = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norms if norms else 0.0


# --- L1 correctness gate -------------------------------------------------------------


def rendered_prompt(container: str, variant: str) -> str:
    """The checkpoint template's prompt for one thinking variant, as vLLM renders it."""
    request: dict = {
        "model": SERVED_MODEL,
        "messages": [{"role": "user", "content": "What is 17 * 19?"}],
        "add_generation_prompt": True,
    }
    if variant != "default":
        request["chat_template_kwargs"] = {"enable_thinking": False}
    tokens = l1_json(container, "/tokenize", request)["tokens"]
    return l1_json(container, "/detokenize", {"model": SERVED_MODEL, "tokens": tokens})["prompt"]


def rendering_error(variant: str, prompt: str) -> str | None:
    # The server default (enable_thinking true) puts the think token at the top
    # of the system turn; thinking off closes an empty thought channel instead.
    if variant == "default":
        if "<|think|>" not in prompt or not prompt.endswith("<|turn>model\n"):
            return f"default does not render thinking on: {prompt[-120:]!r}"
        return None
    if "<|think|>" in prompt or not prompt.endswith("<|channel>thought\n<channel|>"):
        return f"{variant} does not render thinking off: {prompt[-120:]!r}"
    return None


def _l1_probe(container: str, variant: str) -> dict:
    payload = control.arithmetic_probe()
    if variant != "default":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    status, body = l1_call(container, "POST", "/v1/chat/completions", payload)
    parsed = json.loads(body) if status == 200 else body
    error = control.arithmetic_answer_error(parsed) if status == 200 else f"HTTP {status}"
    if error is None and variant == "default":
        message = parsed["choices"][0]["message"]
        if not (message.get("reasoning") or message.get("reasoning_content")):
            error = "thinking on returned no reasoning"
    return {"container": container, "variant": variant, "status": status, "error": error,
            "response": parsed}


_STARTUP_PATTERNS = {
    "weights": r"Model loading took ([0-9.]+) GiB",
    "kv_cache_memory": r"Available KV cache memory: ([0-9.-]+) GiB",
    "kv_cache_tokens": r"GPU KV cache size: ([0-9,]+) tokens",
    "max_concurrency": r"Maximum concurrency for ([0-9,]+) tokens per request: ([0-9.]+)x",
    "hybrid_kv": r"(hybrid[^\n]{0,160})",
    "sampling_defaults": (
        r"(Default (?:vLLM )?sampling parameters have been overridden[^\n]*"
        r"|Using default chat sampling params[^\n]*)"
    ),
}


def startup_evidence(container: str) -> dict:
    logs = subprocess.run(["docker", "logs", container], capture_output=True, text=True, check=True)
    text = logs.stdout + logs.stderr
    return {name: re.findall(pattern, text) for name, pattern in _STARTUP_PATTERNS.items()}


def gpu_memory() -> dict[str, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = dict(line.split(", ") for line in output.strip().splitlines())
    return {index: int(rows[str(index)]) for index in map(str, control.GPU_IDS)}


def embedding_texts() -> list[str]:
    """Queries and documents with EmbeddingGemma 2's own instructions, Japanese and English."""
    spec = SPEC["embedding"]
    queries = (
        "KR-7319という試料は、どのような温度で保存するよう決められていますか。",
        "新規顧客は何社増えましたか。",
        "How does a write-ahead log help a database recover after a crash?",
        "最終の納品期限はいつですか。",
    )
    documents = (
        "保存条件：試料KR-7319は摂氏マイナス四十二度の冷凍庫で保管し、解凍は一回までとする。",
        "第3四半期の主な成果として、新規顧客は42社増え、解約率は1.8%に下がった。",
        "A write-ahead log records each change before it is applied, so replay restores state.",
        "納品期限は2026年12月18日で、担当は品質管理課の佐藤とする。",
    )
    return [spec["query_prefix"] + text for text in queries] + [
        spec["document_prefix"] + text for text in documents
    ]


# sentence-transformers is the model card's reference implementation; it runs
# on CPU in fp32 inside the vLLM image (whose transformers it keeps).
_EMBED_REFERENCE = r"""
import json, subprocess, sys
import transformers
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',
                'sentence-transformers==' + sys.argv[1],
                'transformers==' + transformers.__version__], check=True)
import torch
from sentence_transformers import SentenceTransformer
texts = json.loads(sys.stdin.read())
model = SentenceTransformer('/models/' + sys.argv[2], device='cpu',
                            model_kwargs={'torch_dtype': torch.float32})
vectors = model.encode(texts, normalize_embeddings=False, convert_to_numpy=True)
print('REFERENCE ' + json.dumps([[float(v) for v in row] for row in vectors]))
"""


def embedding_reference(texts: list[str]) -> list[list[float]]:
    env = control._compose_env()
    run = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-i",
            "--entrypoint",
            "python3",
            "--env",
            "CUDA_VISIBLE_DEVICES=",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--volume",
            f"{env['MODEL_STORAGE_PATH']}:/models:ro",
            env["VLLM_IMAGE"],
            "-c",
            _EMBED_REFERENCE,
            SPEC["embedding"]["reference"]["sentence_transformers"],
            SPEC["embedding"]["slug"],
        ],
        input=json.dumps(texts),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    line = next((row for row in run.stdout.splitlines() if row.startswith("REFERENCE ")), None)
    if line is None:
        raise RuntimeError(f"sentence-transformers reference failed: {run.stderr[-2000:]}")
    return json.loads(line.removeprefix("REFERENCE "))


def l1_embeddings(container: str, texts: list[str]) -> list[list[float]]:
    body = l1_json(container, "/v1/embeddings", {"model": EMBED_MODEL, "input": texts})
    return [row["embedding"] for row in sorted(body["data"], key=lambda row: row["index"])]


def l1(run_dir: Path) -> int:
    """L1 correctness: thinking renders, exact finite answers per chat replica, embeddings."""
    report: dict = {"schema_version": 1, "cases": {}}
    failures: list[str] = []

    def record(name: str, error: str | None, detail: object = None) -> None:
        report["cases"][name] = {"passed": error is None, "error": error, "detail": detail}
        if error is not None:
            failures.append(f"{name}: {error}")

    report["startup"] = {
        container: startup_evidence(container) for container in CHAT_CONTAINERS + EMBED_CONTAINERS
    }
    report["idle_gpu_memory_mib"] = gpu_memory()
    # Every chat replica must hold at least one full-context request (the
    # replica search's fixed condition).
    maximum = int(SPEC["model"]["max_context_tokens"])
    for container in CHAT_CONTAINERS:
        rows = report["startup"][container]["max_concurrency"]
        concurrency = next(
            (float(value) for tokens, value in rows if int(tokens.replace(",", "")) == maximum),
            None,
        )
        record(
            f"full_context_{container}",
            None if concurrency is not None and concurrency >= 1.0 else f"{rows!r}",
            concurrency,
        )
    defaults = " ".join(report["startup"][CHAT_CONTAINERS[0]]["sampling_defaults"])
    sampling = SPEC["model"]["sampling"]
    record(
        "default_sampling",
        None
        if all(f"'{key}': {value}" in defaults for key, value in sampling.items())
        else defaults,
    )
    for variant in ("default", "none"):
        try:
            prompt = rendered_prompt(CHAT_CONTAINERS[0], variant)
            record(f"render_{variant}", rendering_error(variant, prompt), prompt[-160:])
        except Exception as error:  # noqa: BLE001 - recorded
            record(f"render_{variant}", str(error))

    probes = int(SPEC["verification"]["l1"]["probes_per_variant"])
    for variant in ("default", "none"):
        jobs = [(container, variant) for container in CHAT_CONTAINERS for _ in range(probes)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(jobs), 64)) as pool:
            results = list(pool.map(lambda job: _l1_probe(*job), jobs))
        errors = [f"{row['container']}: {row['error']}" for row in results if row["error"]]
        (run_dir / f"probes-{variant}.json").write_text(json.dumps(results, indent=2) + "\n")
        record(
            f"probes_{variant}",
            f"{len(errors)}/{len(results)} probes failed" if errors else None,
            {"requests": len(results), "errors": errors[:3]},
        )

    texts = embedding_texts()
    minimum = float(SPEC["embedding"]["reference"]["min_cosine"])
    try:
        reference = embedding_reference(texts)
        for container in EMBED_CONTAINERS:
            vectors = l1_embeddings(container, texts)
            similarities = [cosine(a, b) for a, b in zip(vectors, reference, strict=True)]
            norms = [math.sqrt(sum(v * v for v in vector)) for vector in vectors]
            error = None
            if min(similarities) < minimum:
                error = f"cosine to the reference {min(similarities):.5f} < {minimum}"
            elif any(abs(norm - 1.0) > 1e-3 for norm in norms):
                error = f"norms {norms}"
            record(
                f"embedding_reference_{container}",
                error,
                {"min_cosine": min(similarities), "norms": [round(n, 5) for n in norms]},
            )
    except Exception as error:  # noqa: BLE001 - recorded
        record("embedding_reference", str(error))

    for name, check in (
        ("tool_call", control.validate_tool_calling),
        ("image", control.validate_vision),
        ("kairyu_embeddings", control.validate_embeddings),
    ):
        try:
            check(_api_url())
            record(name, None)
        except SystemExit as error:
            record(name, str(error))
    report["loaded_gpu_memory_mib"] = gpu_memory()
    report["passed"] = not failures
    (run_dir / "l1.json").write_text(json.dumps(report, indent=2) + "\n")
    for line in failures:
        print(f"l1: {line}", file=sys.stderr)
    print(f"l1: {'PASS' if not failures else 'FAIL'} ({len(report['cases'])} cases)")
    return 1 if failures else 0


# --- placement evidence -------------------------------------------------------------------


def placement_rows(path: Path, offset: int) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as stream:
        stream.seek(offset)
        return [
            row for row in (json.loads(line) for line in stream if line.strip())
            if row.get("kind") == "replica"
        ]


def wait_placements(path: Path, offset: int, expected: int, settle_s: float = 15.0) -> list[dict]:
    """The pool log is written asynchronously: poll until ``expected`` rows landed."""
    deadline = time.monotonic() + settle_s
    rows = placement_rows(path, offset)
    while len(rows) < expected and time.monotonic() < deadline:
        time.sleep(0.5)
        rows = placement_rows(path, offset)
    return rows


def own_placements(
    path: Path, offset: int, request_ids: set[str], settle_s: float = 15.0
) -> list[dict]:
    """Placements of these requests only.

    Other clients share the pool while a row runs (the Chat UI's background
    title and follow-up tasks after a chat), so a row is judged on the
    placements of the requests it sent, matched by ``X-Request-ID``.
    """
    deadline = time.monotonic() + settle_s
    rows = [row for row in placement_rows(path, offset) if row.get("request_id") in request_ids]
    while len(rows) < len(request_ids) and time.monotonic() < deadline:
        time.sleep(0.5)
        rows = [
            row for row in placement_rows(path, offset) if row.get("request_id") in request_ids
        ]
    return rows


def row_request_ids(row_dir: Path) -> set[str]:
    samples = json.loads((row_dir / "row.json").read_text())["samples"]
    return {sample["request_id"] for sample in samples if sample.get("request_id")}


def log_offset(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def spread_report(rows: list[dict], *, replicas: int, expected: int, gated: bool) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("replica_id"))] = counts.get(str(row.get("replica_id")), 0) + 1
    mean = expected / replicas
    largest = max(counts.values(), default=0)
    limit = float(SPEC["verification"]["serving"]["placement_gate"]["max_share_of_mean"])
    even = len(rows) == expected and len(counts) == replicas and largest <= limit * mean
    return {
        "placements": len(rows),
        "expected": expected,
        "per_replica": dict(sorted(counts.items())),
        "largest_share_of_mean": round(largest / mean, 3),
        "gated": gated,
        "passed": even if gated else len(rows) == expected,
    }


# --- serving rows -----------------------------------------------------------------------

_WORDS = (
    "code", "review", "function", "module", "request", "result", "verify", "runtime",
    "system", "design", "state", "input", "output", "stream", "cache", "token",
)  # fmt: skip


def fixed_dataset(path: Path, requests: int, approximate_tokens: int, *, namespace: str) -> None:
    # A unique label first defeats prefix caching, so each row is a cold prefill.
    rows = [
        {
            "prompt": f"Row {namespace}-{request}: "
            + " ".join(
                _WORDS[(request * 7 + position * 11) % len(_WORDS)]
                for position in range(approximate_tokens)
            )
        }
        for request in range(requests)
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


_GENERIC_TASKS = (
    "Explain how a write-ahead log lets a database recover after a crash, "
    "and name one cost it adds.",
    "Compare TCP and QUIC for a mobile client on a lossy network in three short paragraphs.",
    "A train leaves at 09:40 and travels 212 km at 80 km/h. When does it arrive? Show the steps.",
    "Summarize the trade-offs between optimistic and pessimistic locking with one example each.",
    "Give a short, correct proof that the square root of 2 is irrational.",
    "Describe how a Bloom filter works and when a false positive can occur.",
    "Explain why floating-point addition is not associative, with a concrete numeric example.",
    "Outline a rollback plan for a failed database schema migration in a web service.",
)
_CODING_TASKS = (
    "Implement an LRU cache class with get and put in O(1) time, with a small usage example.",
    "Write a function that merges overlapping intervals and returns them sorted.",
    "Implement a thread-safe token bucket rate limiter with a monotonic clock.",
    "Write a function that parses an ISO 8601 duration such as 'P1DT2H30M' into seconds.",
    "Implement Dijkstra's shortest path over an adjacency-list graph with a heap.",
    "Write a function that returns the longest palindromic substring of a string.",
    "Implement a trie with insert, search, and starts_with methods.",
    "Write a function that validates a Sudoku board given as a list of nine strings.",
)


def completed_dataset(path: Path, requests: int, *, workload: str, namespace: str) -> None:
    tasks = _GENERIC_TASKS if workload == "generic" else _CODING_TASKS
    suffix = (
        ""
        if workload == "generic"
        else " Return one self-contained Python module in a fenced code block."
    )
    rows = [
        {
            "prompt": f"Row label (an identifier, not an instruction): {namespace}-{request}.\n\n"
            + tasks[request % len(tasks)]
            + suffix
        }
        for request in range(requests)
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def bench(
    dataset: Path, *, mode: str, requests: int, concurrency: int, max_tokens: int, out: Path
) -> int:
    command = [
        sys.executable,
        str(HERE / "benchmark.py"),
        "--base-url",
        f"{_api_url()}/v1",
        "--model",
        SERVED_MODEL,
        "--dataset",
        str(dataset),
        "--mode",
        mode,
        "--num-requests",
        str(requests),
        "--concurrency",
        str(concurrency),
        "--max-tokens",
        str(max_tokens),
        "--results-dir",
        str(out),
    ]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "bench.log").open("w") as log:
        return subprocess.run(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT).returncode


class GpuPeak:
    """Sample every GPU's memory once per second while a row runs."""

    def __init__(self) -> None:
        self.peak: dict[str, int] = {}
        self._stop = False
        self._thread: threading.Thread | None = None

    def __enter__(self):
        def loop() -> None:
            while not self._stop:
                for index, used in gpu_memory().items():
                    self.peak[index] = max(self.peak.get(index, 0), used)
                time.sleep(1)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=5)


def serving(
    run_dir: Path,
    concurrency: list[int] | None = None,
    requests: int | None = None,
    *,
    rows_dir: Path | None = None,
) -> int:
    """8K input, up to 256 output (natural completion) at each concurrency; even placement."""
    config = SPEC["verification"]["serving"]
    rows_dir = rows_dir or run_dir
    requests = requests or int(config["requests_per_concurrency"])
    output_tokens = int(config["output_tokens"])
    prompt_tokens = int(config["prompt_tokens_approx"])
    gate = config["placement_gate"]
    warmup = rows_dir / "warmup.json"
    rows_dir.mkdir(parents=True, exist_ok=True)
    fixed_dataset(warmup, CHAT_REPLICAS, prompt_tokens, namespace=f"{rows_dir.name}-warmup")
    if bench(
        warmup,
        mode="capped",
        requests=CHAT_REPLICAS,
        concurrency=CHAT_REPLICAS,
        max_tokens=32,
        out=rows_dir / "warmup",
    ):
        print("warm-up row failed", file=sys.stderr)
        return 1
    for level in concurrency or config["concurrency"]:
        dataset = rows_dir / f"capped-c{level}.json"
        fixed_dataset(dataset, requests, prompt_tokens, namespace=f"{rows_dir.name}-c{level}")
        offset = log_offset(CHAT_PLACEMENT_LOG)
        peak = GpuPeak()
        with peak:
            code = bench(
                dataset,
                mode="capped",
                requests=requests,
                concurrency=level,
                max_tokens=output_tokens,
                out=rows_dir / f"capped-c{level}",
            )
        spread = spread_report(
            own_placements(
                CHAT_PLACEMENT_LOG, offset, row_request_ids(rows_dir / f"capped-c{level}")
            ),
            replicas=CHAT_REPLICAS,
            expected=requests,
            gated=level >= int(gate["min_concurrency"]),
        )
        (rows_dir / f"capped-c{level}" / "resources.json").write_text(
            json.dumps({"gpu_peak_mib": peak.peak, "placement": spread}, indent=2) + "\n"
        )
        if code or not spread["passed"]:
            print(f"capped c{level}: exit {code}, placement {spread}", file=sys.stderr)
            return 1
    return 0


def completed(run_dir: Path) -> int:
    """Generic and coding tasks to a completed visible answer (c=1, 8, 32), thinking on."""
    failures = 0
    for workload in ("generic", "coding"):
        for level in (1, 8, 32):
            dataset = run_dir / f"{workload}-c{level}.json"
            completed_dataset(
                dataset, 32, workload=workload, namespace=f"{run_dir.name}-{workload}-c{level}"
            )
            failures += (
                bench(
                    dataset,
                    mode="completed",
                    requests=32,
                    concurrency=level,
                    max_tokens=32768,
                    out=run_dir / f"{workload}-c{level}",
                )
                != 0
            )
    return int(failures > 0)


# --- public gates --------------------------------------------------------------------------


def tool_call_error(message: dict, finish_reason: object) -> str | None:
    if finish_reason != "tool_calls":
        return f"finish_reason is {finish_reason!r}"
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return f"tool_calls is {calls!r}"
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or function.get("name") != "bash":
            return f"unexpected tool call {call!r}"
        try:
            arguments = json.loads(function.get("arguments") or "")
        except ValueError:
            return f"arguments are not JSON: {function.get('arguments')!r}"
        if not isinstance(arguments, dict) or not isinstance(arguments.get("command"), str):
            return f"arguments lack a command: {arguments!r}"
    return None


def _stream_tool_message(sse: str) -> tuple[dict, object]:
    calls: dict[int, dict[str, str]] = {}
    finish: object = None
    for line in sse.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        for choice in json.loads(line[6:]).get("choices", ()):
            finish = choice.get("finish_reason") or finish
            for delta in (choice.get("delta") or {}).get("tool_calls") or ():
                slot = calls.setdefault(delta.get("index", 0), {"name": "", "arguments": ""})
                function = delta.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
    return {"tool_calls": [{"function": slot} for _, slot in sorted(calls.items())] or None}, finish


def tool_calling(run_dir: Path) -> int:
    """OpenAI bash-tool contract on every chat replica: fan-out, tool-result turn, streaming."""
    cases: dict[str, str | None] = {}
    fan = 2 * CHAT_REPLICAS
    offset = log_offset(CHAT_PLACEMENT_LOG)
    with concurrent.futures.ThreadPoolExecutor(max_workers=fan) as pool:
        results = list(pool.map(lambda _: post_responses(control.tool_probe()), range(fan)))
    first: dict | None = None
    error = None
    for status, body, _ in results:
        if status != 200 or not isinstance(body, dict):
            error = f"HTTP {status}: {str(body)[:200]}"
            break
        choice = body["choices"][0]
        error = tool_call_error(choice["message"], choice["finish_reason"])
        if error:
            break
        first = choice["message"]
    cases[f"auto_tool_call_x{fan}"] = error
    placed = {
        str(row.get("replica_id")) for row in wait_placements(CHAT_PLACEMENT_LOG, offset, fan)
    }
    cases["every_replica_called_tools"] = (
        None if len(placed) == CHAT_REPLICAS else f"tool calls placed on replicas {sorted(placed)}"
    )
    if first is not None:
        call = first["tool_calls"][0]
        call_id = call.get("id") or "call_0"
        probe = control.tool_probe()
        status, body, _ = post_responses(
            {
                **probe,
                "messages": [
                    probe["messages"][0],
                    {
                        "role": "user",
                        "content": (
                            "List the files in the current directory, "
                            "then show the first line of README.md."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": first.get("content"),
                        "reasoning_content": first.get("reasoning_content"),
                        "tool_calls": [
                            {"id": call_id, "type": "function", "function": call["function"]}
                        ],
                    },
                    {"role": "tool", "tool_call_id": call_id, "content": "README.md\nsrc\ntests\n"},
                ],
            }
        )
        cases["tool_result_turn"] = (
            tool_call_error(body["choices"][0]["message"], body["choices"][0]["finish_reason"])
            if status == 200 and isinstance(body, dict)
            else f"HTTP {status}"
        )
    status, body, _ = post_responses({**control.tool_probe(), "stream": True})
    cases["streamed_tool_call"] = (
        tool_call_error(*_stream_tool_message(body)) if status == 200 else f"HTTP {status}"
    )
    report = {"cases": cases, "replicas_placed": sorted(placed), "passed": not any(cases.values())}
    (run_dir / "tool-calling.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"tool-calling: {'PASS' if report['passed'] else 'FAIL'} {cases}")
    return int(not report["passed"])


def vision(run_dir: Path) -> int:
    """Concurrent image requests answered red and completed."""
    config = SPEC["verification"]["vision"]
    fan = int(config["requests"])

    def ask(case: int):
        return post_responses(
            control.image_request(
                f"Vision case {case}: what single color fills this image? Answer with one word.",
                max_tokens=int(config["max_tokens"]),
            ),
            timeout_s=1800,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=fan) as pool:
        results = list(pool.map(ask, range(fan)))
    answers, errors = [], []
    for status, body, _ in results:
        if status != 200 or not isinstance(body, dict):
            errors.append(f"HTTP {status}")
            continue
        choice = body["choices"][0]
        content = choice["message"].get("content") or ""
        answers.append(content.strip()[:80])
        if choice.get("finish_reason") != "stop" or "red" not in content.lower():
            errors.append(f"not a completed red answer: {content[:80]!r}")
    report = {"answers": answers, "errors": errors, "passed": not errors}
    (run_dir / "vision.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"vision: {'PASS' if not errors else 'FAIL'} {answers}")
    return int(bool(errors))


def _stream(payload: dict, *, timeout_s: float = 1800) -> dict:
    """One streamed /v1/responses request: visible text, reasoning, terminal state."""
    start = time.monotonic()
    first_content = None
    content = trace = ""
    finish = usage = None
    done = False
    with open_responses_stream(payload, timeout_s=timeout_s) as response:
        for line in response:
            event = stream_event(line)
            if event is None:
                continue
            kind = event.get("type")
            if kind == "response.output_text.delta":
                if event.get("delta") and first_content is None:
                    first_content = time.monotonic() - start
                content += event.get("delta") or ""
            elif kind in _REASONING_DELTAS:
                trace += event.get("delta") or ""
            elif kind in ("response.completed", "response.incomplete"):
                view = control.chat_view(event["response"])
                finish, usage, done = view["choices"][0]["finish_reason"], view["usage"], True
                break
            elif kind in ("error", "response.failed"):
                break
    return {
        "done": done,
        "finish_reason": finish,
        "content": content,
        "reasoning_chars": len(trace),
        "time_to_content_s": first_content,
        "total_s": time.monotonic() - start,
        "usage": usage,
    }


def reasoning(run_dir: Path) -> int:
    """Thinking on by default (reasoning, then 323); ``none`` answers 323 without reasoning."""
    rows = []
    for effort in (None, "none"):
        payload = {
            "model": SERVED_MODEL,
            "max_tokens": 32768,
            "messages": [
                {"role": "user", "content": "What is 17 * 19? Reply with only the integer."}
            ],
        }
        if effort:
            payload["reasoning_effort"] = effort
        try:
            row = {"effort": effort or "default", **_stream(payload)}
        except urllib.error.HTTPError as error:
            row = {"effort": effort, "done": False, "error": f"HTTP {error.code}: "
                   f"{error.read().decode('utf-8', 'replace')[:300]}"}
            rows.append({**row, "passed": False})
            continue
        thought = row["reasoning_chars"] > 0
        row["passed"] = (
            row["done"]
            and row["finish_reason"] == "stop"
            and row["content"].strip() == "323"
            and (thought if effort is None else not thought)
        )
        rows.append(row)
    (run_dir / "reasoning.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"reasoning: {[(row['effort'], row['passed']) for row in rows]}")
    return int(not all(row["passed"] for row in rows))


def _l1_active_requests() -> float:
    total = 0.0
    for container in CHAT_CONTAINERS:
        _status, metrics = l1_call(container, "GET", "/metrics")
        for gauge in ("running", "waiting"):
            values = re.findall(
                rf"^vllm:num_requests_{gauge}\{{[^\n]*\}} ([0-9.eE+-]+)$", metrics, re.M
            )
            if not values:
                raise ValueError(f"missing L1 {gauge} gauge on {container}")
            total += sum(float(value) for value in values)
    return total


def cancellation(run_dir: Path) -> int:
    """A client disconnect releases the replica slot and L1; a follow-up completes."""
    payload = {
        "model": SERVED_MODEL,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "Count upwards from one, one number per line."}],
    }
    saw_delta = observed = False
    with open_responses_stream(payload) as response:
        for line in response:
            event = stream_event(line)
            if (
                event is not None
                and event.get("type") in ("response.output_text.delta", *_REASONING_DELTAS)
                and event.get("delta")
            ):
                saw_delta = True
                break
        deadline = time.monotonic() + 15
        while saw_delta and time.monotonic() < deadline:
            if _l1_active_requests() > 0:
                observed = True
                break
            time.sleep(0.5)
    released_at = None
    start = time.monotonic()
    while time.monotonic() - start < 30:
        with urllib.request.urlopen(f"{_api_url()}/metrics", timeout=5) as response:
            metrics = response.read().decode()
        values = re.findall(
            r'^kairyu_replica_outstanding\{pool="'
            + re.escape(SERVED_MODEL)
            + r'"[^\n]*\} ([0-9.]+)$',
            metrics,
            re.M,
        )
        if values and all(float(v) == 0 for v in values) and _l1_active_requests() == 0:
            released_at = time.monotonic() - start
            break
        time.sleep(0.25)
    status, body, _ = post_responses(
        {
            "model": SERVED_MODEL,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": "Reply OK."}],
        }
    )
    usable = (
        status == 200
        and isinstance(body, dict)
        and body["choices"][0].get("finish_reason") == "stop"
        and bool(body["choices"][0]["message"].get("content"))
    )
    report = {
        "received_delta": saw_delta,
        "l1_observed_active": observed,
        "released_after_s": released_at,
        "follow_up_completed": usable,
        "passed": bool(saw_delta and observed and released_at is not None and usable),
    }
    (run_dir / "cancellation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"cancellation: {report}")
    return int(not report["passed"])


def long_context(run_dir: Path) -> int:
    """32K / 128K retrieval and a ~250K full-context request through Kairyu.

    Every length must be accepted with the server's prompt-token count at the
    target and end normally. The archive key is judged only at the lengths in
    ``long_context_answer_tokens``: at ~250K, Gemma 4's reasoning at its
    official sampling sometimes copies the one-word haystack until the output
    cap (MEASUREMENTS.md), which is the model's accuracy, not serving behavior.
    With an edited checkpoint served the key is recorded at every length but
    not judged.
    """
    import secrets

    from tokenizers import Tokenizer

    # The served checkpoint's tokenizer (pinned or CHAT_CHECKPOINT_DIR), as
    # recorded by `run.sh up` once the replicas served it.
    record = ENVIRONMENT_STORAGE / control.CHAT_CHECKPOINT_RECORD
    if not record.is_file():
        raise ValueError(f"{record} is missing; start the example with ./run.sh up first")
    served_record = json.loads(record.read_text(encoding="utf-8"))
    served = Path(served_record["path"])
    tokenizer = Tokenizer.from_file(str(served / "tokenizer.json"))
    # An edited checkpoint's answers are what is being measured elsewhere:
    # every length is judged on acceptance and token count only.
    edited = served_record.get("source") != "pinned"
    filler = " filler"
    if len(tokenizer.encode(filler * 100, add_special_tokens=False).ids) != 100:
        raise ValueError("long-context filler is not one token per repetition")
    maximum = int(SPEC["model"]["max_context_tokens"])
    answer_targets = {int(value) for value in SPEC["verification"]["long_context_answer_tokens"]}
    output_cap = 8192
    # The chat template adds a few tokens around the user prompt.
    template_tokens = 64
    rows = []
    for target in SPEC["verification"]["long_context_tokens"]:
        key = "K" + secrets.token_hex(12).upper()
        prefix = "Read this log and recover its archive key.\n"
        needle = f"\nThe archive key is {key}.\n"
        suffix = "\nReturn only the archive key."
        overhead = len(tokenizer.encode(prefix + needle + suffix, add_special_tokens=False).ids)
        repeats = int(target) - overhead
        prompt = (
            prefix + filler * (repeats // 2) + needle + filler * (repeats - repeats // 2) + suffix
        )
        start = time.monotonic()
        status, body, _ = post_responses(
            {
                "model": SERVED_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": output_cap,
            },
            timeout_s=3600,
        )
        choice = body.get("choices", [{}])[0] if isinstance(body, dict) else {}
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        content = (choice.get("message") or {}).get("content")
        prompt_tokens = usage.get("prompt_tokens") or 0
        answer_judged = int(target) in answer_targets and not edited
        passed = (
            status == 200
            and int(target) <= prompt_tokens <= int(target) + template_tokens
            and prompt_tokens + output_cap <= maximum
            and (
                choice.get("finish_reason") == "stop"
                and isinstance(content, str)
                and content.strip() == key
                if answer_judged
                else choice.get("finish_reason") in ("stop", "length")
            )
        )
        rows.append(
            {
                "target_input_tokens": target,
                "answer_judged": answer_judged,
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
                "content": content,
                "expected": key,
                "total_s": time.monotonic() - start,
                "passed": passed,
            }
        )
        (run_dir / "long-context.json").write_text(json.dumps(rows, indent=2) + "\n")
        if not passed:
            return 1
    return 0


def _container_health(container: str) -> str | None:
    state = json.loads(subprocess.check_output(["docker", "inspect", container]))[0]["State"]
    return (state.get("Health") or {}).get("Status")


def _wait_healthy(containers: list[str], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if all(_container_health(container) == "healthy" for container in containers):
            return True
        time.sleep(2)
    return False


def _kairyu_healthy(pool: str, expected: int, timeout_s: float = 120) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{_api_url()}/metrics", timeout=5) as response:
                if control.healthy_replicas(response.read().decode(), pool) == expected:
                    return True
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    return False


def restart(run_dir: Path) -> int:
    """`run.sh up` after restarting every service: ready, and every Chat UI check matches."""
    env = control._compose_env()
    start = time.monotonic()
    subprocess.run(control._compose_command(["restart"]), env=env, check=True)
    error = None
    try:
        control.up()
        passed = True
    except SystemExit as failure:
        passed, error = False, str(failure)
    report = {"ready_after_s": round(time.monotonic() - start, 1), "passed": passed, "error": error}
    (run_dir / "restart.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"restart: {report}")
    return int(not passed)


def api_equivalence(run_dir: Path) -> int:
    """L1 and L3 agree on Responses (and image-tools on chat); L3 keeps the caller's API."""
    metrics = [
        argument
        for container in CHAT_CONTAINERS
        for argument in ("--l1-metrics", f"{container}=http://127.0.0.1:8000/metrics")
    ]
    command = [
        sys.executable,
        str(ROOT / "verification/product/correctness/upstream_api_equivalence.py"),
        "--l1",
        "http://127.0.0.1:8000/v1",
        "--l1-docker",
        CHAT_CONTAINERS[0],
        "--l1-model",
        SERVED_MODEL,
        *metrics,
        "--l3",
        f"{_api_url()}/v1",
        "--l3-model",
        SERVED_MODEL,
        "--cases",
        "text,stream,effort,tools,image,image-tools,cap,openwebui",
        # Gemma 4 has no effort levels: any explicit effort but none thinks.
        "--efforts",
        "low,high",
        "--expect-reasoning",
        "--exact-input-tokens",
        "--out",
        str(run_dir / "api-equivalence.json"),
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


# --- embeddings through L2/L3 ---------------------------------------------------------------


def embeddings(run_dir: Path) -> int:
    """Kairyu's /v1/embeddings: equal to L1, spread over every replica, survives an outage."""
    config = SPEC["verification"]["embeddings"]
    report: dict = {}
    failures: list[str] = []
    texts = (embedding_texts() * (int(config["texts"]) // len(embedding_texts()) + 1))[
        : int(config["texts"])
    ]
    texts = [f"{text} ({index})" for index, text in enumerate(texts)]

    # 1. The same vectors as the replica itself (one batch through Kairyu).
    status, body = post_embeddings(texts)
    if status != 200 or not isinstance(body, dict):
        failures.append(f"Kairyu batch: HTTP {status}: {str(body)[:200]}")
    else:
        served = [row["embedding"] for row in sorted(body["data"], key=lambda row: row["index"])]
        direct = l1_embeddings(EMBED_CONTAINERS[0], texts)
        lowest = min(cosine(a, b) for a, b in zip(served, direct, strict=True))
        report["min_cosine_kairyu_vs_l1"] = lowest
        if lowest < float(config["min_cosine_kairyu_vs_l1"]):
            failures.append(f"Kairyu and L1 vectors differ: cosine {lowest:.6f}")
        report["usage"] = body.get("usage")

    # 2. Concurrent batches reach every embedding replica.
    def spread_round(label: str) -> dict:
        batches = [texts[index::8] for index in range(8)]
        offset = log_offset(EMBED_PLACEMENT_LOG)
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(config["concurrency"])) as pool:
            results = list(pool.map(post_embeddings, batches))
        rows = wait_placements(EMBED_PLACEMENT_LOG, offset, len(batches))
        replicas = sorted({str(row.get("replica_id")) for row in rows})
        errors = [f"HTTP {code}" for code, _ in results if code != 200]
        if errors or len(replicas) != EMBED_REPLICAS:
            failures.append(f"{label}: errors {errors}, replicas reached {replicas}")
        return {"batches": len(batches), "errors": errors, "replicas": replicas}

    report["spread"] = spread_round("spread")

    # 3. One replica stops: at most ``outage_max_failures`` requests fail before
    # the pool ejects it, /readyz stays ready, the others carry every request;
    # after a restart the prober returns it and batches spread again.
    stopped = EMBED_CONTAINERS[0]
    outcomes: list[int] = []
    ready_codes: list[int] = []
    subprocess.run(["docker", "stop", stopped], check=True, capture_output=True)
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            outcomes.append(post_embeddings(texts[:4], timeout_s=30)[0])
            try:
                with urllib.request.urlopen(f"{_api_url()}/readyz", timeout=5) as response:
                    ready_codes.append(response.status)
            except urllib.error.HTTPError as error:
                ready_codes.append(error.code)
            time.sleep(0.25)
    finally:
        subprocess.run(["docker", "start", stopped], check=True, capture_output=True)
    failed = [code for code in outcomes if code != 200]
    first_failure_tail = outcomes[outcomes.index(failed[0]) + 1 :] if failed else outcomes
    report["outage"] = {
        "requests": len(outcomes),
        "failed": len(failed),
        "all_after_ejection_succeeded": all(code == 200 for code in first_failure_tail),
        "readyz": sorted(set(ready_codes)),
    }
    if len(failed) > int(config["outage_max_failures"]) or set(ready_codes) != {200}:
        failures.append(f"outage: {report['outage']}")
    restored = _wait_healthy([stopped], 1800) and _kairyu_healthy(EMBED_MODEL, EMBED_REPLICAS)
    report["restored"] = restored
    if not restored:
        failures.append("the stopped embedding replica did not return to the pool")
    else:
        report["spread_after_restore"] = spread_round("after restore")
    report["passed"] = not failures
    report["failures"] = failures
    (run_dir / "embeddings.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "usage"}))
    return 0 if not failures else 1


# --- evidence ----------------------------------------------------------------------------------


def served_config_sha256() -> str:
    digest = hashlib.sha256()
    for path in SERVED_CONFIG_FILES:
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def runtime_evidence(*, allow_override: bool = False) -> dict:
    """Refuse to measure a runtime that differs from the committed configuration."""
    env = control._compose_env()
    rendered = json.loads(
        subprocess.check_output(
            control._compose_command(["config", "--format", "json"]), env=env, cwd=ROOT
        )
    )
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", env["VLLM_IMAGE"]], text=True
    ).strip()
    replicas = {}
    for service in control.L1_SERVICES:
        container = json.loads(
            subprocess.check_output(["docker", "inspect", f"{control.PROJECT}-{service}-1"])
        )[0]
        expected = [str(part) for part in rendered["services"][service]["command"]]
        if not allow_override:
            if container["Config"]["Cmd"] != expected:
                raise ValueError(f"running {service} command differs from compose.yaml")
            if container["Image"] != image_id or env["VLLM_IMAGE"] != SPEC["vllm"]["image"]:
                raise ValueError(f"running {service} image differs from the example.json pin")
        replicas[service] = {
            "image_id": container["Image"],
            "command": container["Config"]["Cmd"],
            "started_at": container["State"]["StartedAt"],
            "cpuset": container["HostConfig"]["CpusetCpus"],
        }
    gateway = f"{control.PROJECT}-kairyu-1"
    mounted = subprocess.check_output(["docker", "exec", gateway, "cat", "/etc/kairyu/kairyu.yaml"])
    if mounted != (HERE / "kairyu.yaml").read_bytes():
        raise ValueError("gateway serves a different kairyu.yaml")
    return {
        "vllm_image": env["VLLM_IMAGE"],
        "replicas": replicas,
        "gateway_config_sha256": hashlib.sha256(mounted).hexdigest(),
    }


# --- Japanese knowledge bases (Chat UI) --------------------------------------------
#
# The Chat UI extracts documents with Docling (RapidOCR for bitmaps), embeds
# them with EmbeddingGemma 2 through Kairyu's /v1/embeddings and reranks with
# bge-reranker-v2-m3 (control.py). These gates use generated Japanese
# documents: an image-only "scanned" PDF rendered with a pinned Noto Sans JP,
# a text-layer PDF printed by Chromium, and plain-text distractors that share
# the scanned document's vocabulary.

KNOWLEDGE = SPEC["verification"]["knowledge"]
KNOWLEDGE_NAME = "kairyu-knowledge-gate"
KNOWLEDGE_SCAN = "kairyu-gate-scan.pdf"
KNOWLEDGE_KEY_SENTENCE = (
    "保存条件：試料KR-7319は摂氏マイナス四十二度の冷凍庫で保管し、解凍は一回までとする。"
)
# Fragments without glyph twins: OCR may read the kanji "一" as the long-vowel "ー".
KNOWLEDGE_KEY_FRAGMENTS = ("KR-7319", "マイナス四十二度", "冷凍庫で保管")
KNOWLEDGE_QUERY = "KR-7319という試料は、どのような温度で保存するよう決められていますか。"
_JA_SUBJECTS = (
    "第一分析室",
    "第二分析室",
    "品質管理課",
    "設備保全班",
    "試料受付窓口",
    "安全衛生委員会",
    "製剤開発グループ",
    "物性評価チーム",
)
_JA_TEMPLATES = (
    "{subject}では、{month}月の点検で{device}の校正記録を{count}件確認した。",
    "{subject}は、試料{code}の受入時に外観と重量を記録し、異常がないことを確かめた。",
    "{device}の稼働時間は月平均{count}時間で、前年同月より{delta}時間増えた。",
    "{subject}の報告によると、試料{code}は摂氏{temp}度の保管庫に移された。",
    "手順書の改訂案では、{device}の清掃を{count}日ごとに行うことが提案されている。",
    "{month}月の会議で、{subject}は廃液の分別方法について{count}項目の見直しを求めた。",
    "試料{code}の分析結果は、規格の範囲内であったため再試験は行わない。",
    "{subject}は、{device}の消耗品を{count}個発注し、納期を{month}月末とした。",
)
_JA_DEVICES = ("液体クロマトグラフ", "分光光度計", "恒温槽", "遠心分離機", "電子天秤", "乾燥機")
_JA_TEMPS = ("四", "五", "マイナス二十", "マイナス八十", "二十五", "マイナス四十")


def _normalized(text: str) -> str:
    """NFKC, no whitespace, every dash-like character as ``-`` (OCR spacing varies)."""
    import unicodedata

    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    text = re.sub("[‐-―−]", "-", text)
    # OCR may read the sample code's hyphen as a long-vowel mark or a kanji one.
    return re.sub("KR[-ー一]?(?=\\d)", "KR-", text)


def _has_key(text: str) -> bool:
    flat = _normalized(text)
    return all(_normalized(fragment) in flat for fragment in KNOWLEDGE_KEY_FRAGMENTS)


def knowledge_lines(page: int, *, key: bool, namespace: str) -> list[str]:
    """Deterministic Japanese report lines for one page; ``key`` plants the key sentence."""
    lines = [f"技術報告書 {namespace} 第{page}頁"]
    for index in range(16):
        digest = hashlib.sha256(f"{namespace}:{page}:{index}".encode()).digest()
        code = f"KR-{7300 + digest[1] % 90 + (1 if 7300 + digest[1] % 90 >= 7319 else 0)}"
        lines.append(
            _JA_TEMPLATES[digest[0] % len(_JA_TEMPLATES)].format(
                subject=_JA_SUBJECTS[digest[2] % len(_JA_SUBJECTS)],
                device=_JA_DEVICES[digest[3] % len(_JA_DEVICES)],
                month=1 + digest[4] % 12,
                count=2 + digest[5] % 30,
                delta=1 + digest[6] % 9,
                temp=_JA_TEMPS[digest[7] % len(_JA_TEMPS)],
                code=code,
            )
        )
        if key and index == 7:
            lines.append(KNOWLEDGE_KEY_SENTENCE)
    return lines


def knowledge_font() -> Path:
    """The pinned Noto Sans JP (OFL) used to render the Japanese fixtures."""
    target = ENVIRONMENT_STORAGE / "bench-tmp" / "NotoSansJP-Regular.otf"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(KNOWLEDGE["font_url"], timeout=120) as response:
            target.write_bytes(response.read())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != KNOWLEDGE["font_sha256"]:
        target.unlink()
        raise RuntimeError(f"Noto Sans JP digest {digest} differs from the pinned font")
    return target


def scan_pdf(path: Path, pages: int, key_pages: set[int], *, namespace: str) -> None:
    """An image-only A4 PDF at 150 dpi: there is no text layer, only pixels to OCR."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(knowledge_font()), 28)
    width, height, margin, line_height = 1240, 1754, 110, 46
    images = []
    for page in range(1, pages + 1):
        image = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(image)
        y = margin
        for line in knowledge_lines(page, key=page in key_pages, namespace=namespace):
            row = ""
            for character in line:
                if draw.textlength(row + character, font=font) > width - 2 * margin:
                    draw.text((margin, y), row, font=font, fill=0)
                    y, row = y + line_height, ""
                row += character
            draw.text((margin, y), row, font=font, fill=0)
            y += line_height + 10
        images.append(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], resolution=150)


def browser_image() -> str:
    """Build the pinned Playwright image (the repository's Chat UI browser Dockerfile)."""
    image = KNOWLEDGE["browser_image"]
    build = subprocess.run(
        [
            "docker",
            "build",
            "--file",
            str(ROOT / "deploy/compose/Dockerfile.webui-browser"),
            "--tag",
            image,
            str(ROOT),
        ],
        capture_output=True,
        text=True,
    )
    if build.returncode:
        raise RuntimeError(f"building {image} failed: {build.stderr[-2000:]}")
    return image


def text_pdf(path: Path, lines: list[str]) -> None:
    """A text-layer PDF of ``lines``, printed by Chromium with the pinned font embedded."""
    import html

    work = path.parent
    font = knowledge_font()
    (work / font.name).write_bytes(font.read_bytes())
    body = "".join(f"<p>{html.escape(line)}</p>" for line in lines)
    (work / "text.html").write_text(
        "<!doctype html><meta charset='utf-8'><style>"
        f"@font-face{{font-family:jp;src:url('{font.name}')}}"
        "body{font-family:jp;font-size:14pt;margin:2cm}</style>" + body,
        encoding="utf-8",
    )
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--env",
            "HOME=/tmp",
            "--volume",
            f"{work}:/fixtures",
            "--entrypoint",
            "/ms-playwright/chromium-1223/chrome-linux64/chrome",
            browser_image(),
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--disable-crash-reporter",
            "--no-pdf-header-footer",
            f"--print-to-pdf=/fixtures/{path.name}",
            "file:///fixtures/text.html",
        ],
        check=True,
        capture_output=True,
    )


def distractor_documents(count: int, *, namespace: str) -> list[tuple[str, str]]:
    return [
        (
            f"kairyu-gate-note-{index:02d}.md",
            "\n".join(knowledge_lines(100 + index, key=False, namespace=namespace)) + "\n",
        )
        for index in range(1, count + 1)
    ]


def _compose_service(name: str) -> dict:
    rendered = subprocess.check_output(
        control._compose_command(["config", "--format", "json"]),
        env=control._compose_env(),
        cwd=ROOT,
    )
    return json.loads(rendered)["services"][name]


_LOADER_PROGRAM = r"""
import json, os, sys, time
sys.path.insert(0, '/app/backend')
os.makedirs(os.environ['DATA_DIR'], exist_ok=True)
from open_webui.retrieval.loaders.main import DoclingLoader
params = json.loads(os.environ['DOCLING_PARAMS'])
report = {}
for name in sys.argv[1:]:
    started = time.monotonic()
    documents = DoclingLoader(url=os.environ['DOCLING_SERVER_URL'], file_path='/fixtures/' + name,
                              mime_type='application/pdf', params=params).load()
    report[name] = {'seconds': time.monotonic() - started,
                    'pages': [document.page_content for document in documents]}
print('LOADER-REPORT ' + json.dumps(report, ensure_ascii=False))
"""


def docling(run_dir: Path) -> int:
    """Docling alone, offline: Japanese image-only and text-layer PDFs via Open WebUI's loader."""
    env = control._compose_env()
    fixtures = run_dir / "fixtures"
    pages = int(KNOWLEDGE["probe_pages"])
    scan_pdf(fixtures / "scan.pdf", pages, set(range(1, pages + 1)), namespace=run_dir.name)
    text_lines = knowledge_lines(1, key=True, namespace=f"{run_dir.name}-text")
    text_pdf(fixtures / "text.pdf", text_lines)
    service = _compose_service("docling")
    network = f"kairyu-docling-probe-{run_dir.name.lower()}"
    container = f"{network}-docling"
    environment = [
        item
        for key, value in service["environment"].items()
        for item in ("--env", f"{key}={value}")
    ]
    report: dict = {"docling_image": env["DOCLING_IMAGE"], "docling_params": env["DOCLING_PARAMS"]}
    subprocess.run(
        ["docker", "network", "create", "--internal", network], check=True, capture_output=True
    )
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--name",
                container,
                "--network",
                network,
                "--network-alias",
                "docling",
                "--cpus",
                str(service["cpus"]),
                *environment,
                env["DOCLING_IMAGE"],
            ],
            check=True,
            capture_output=True,
        )
        deadline = time.monotonic() + 300
        while subprocess.run(
            [
                "docker",
                "exec",
                container,
                "python3",
                "-c",
                "import urllib.request; "
                "urllib.request.urlopen('http://127.0.0.1:5001/health', timeout=2)",
            ],
            capture_output=True,
        ).returncode:
            if time.monotonic() > deadline:
                raise RuntimeError("Docling did not become healthy")
            time.sleep(2)
        loaded = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                network,
                "--volume",
                f"{fixtures}:/fixtures:ro",
                "--env",
                "DATA_DIR=/tmp/open-webui",
                "--env",
                # A throwaway key: only the loader is imported, no server runs.
                f"WEBUI_SECRET_KEY={hashlib.sha256(run_dir.name.encode()).hexdigest()}",
                "--env",
                f"DOCLING_SERVER_URL={env['DOCLING_SERVER_URL']}",
                "--env",
                f"DOCLING_PARAMS={env['DOCLING_PARAMS']}",
                "--entrypoint",
                "python3",
                env["OPEN_WEBUI_IMAGE"],
                "-c",
                _LOADER_PROGRAM,
                "scan.pdf",
                "text.pdf",
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        (run_dir / "loader.log").write_text(loaded.stdout + loaded.stderr)
        line = next(
            (row for row in loaded.stdout.splitlines() if row.startswith("LOADER-REPORT ")), None
        )
        if line is None:
            print("Open WebUI's Docling loader returned no report (loader.log)", file=sys.stderr)
            return 1
        extracted = json.loads(line.removeprefix("LOADER-REPORT "))
    finally:
        subprocess.run(
            ["docker", "logs", container],
            stdout=(run_dir / "docling.log").open("w"),
            stderr=subprocess.STDOUT,
        )
        subprocess.run(["docker", "rm", "--force", container], capture_output=True)
        subprocess.run(["docker", "network", "rm", network], capture_output=True)
    scan = extracted["scan.pdf"]
    text = extracted["text.pdf"]
    flat_text = _normalized("".join(text["pages"]))
    report.update(
        scan_pages=len(scan["pages"]),
        scan_seconds=round(scan["seconds"], 1),
        scan_seconds_per_page=round(scan["seconds"] / pages, 1),
        scan_pages_with_key=[
            index + 1 for index, page in enumerate(scan["pages"]) if _has_key(page)
        ],
        text_seconds=round(text["seconds"], 1),
        text_missing_lines=[line for line in text_lines if _normalized(line) not in flat_text],
        scan_excerpt=scan["pages"][:1],
    )
    passed = (
        report["scan_pages"] == pages
        and report["scan_pages_with_key"] == list(range(1, pages + 1))
        and report["scan_seconds_per_page"] <= float(KNOWLEDGE["seconds_per_page"])
        and not report["text_missing_lines"]
    )
    report["passed"] = passed
    (run_dir / "docling.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "scan_excerpt"}, ensure_ascii=False))
    return 0 if passed else 1


def _chat_ui_url() -> str:
    return f"http://127.0.0.1:{os.environ.get('CHAT_UI_PORT', SPEC['webui']['port'])}"


def _chat_ui_token() -> str:
    """The admin session (WEBUI_ADMIN_EMAIL / WEBUI_ADMIN_PASSWORD) the gates act as."""
    return control.chat_ui_admin_token(_chat_ui_url())


def _webui_request(method: str, path: str, token: str, *, payload=None, timeout_s: float = 600):
    request = urllib.request.Request(
        f"{_chat_ui_url()}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read() or b"null")


def _webui_upload(token: str, name: str, data: bytes, content_type: str) -> dict:
    boundary = f"kairyu{hashlib.sha256(data).hexdigest()[:24]}"
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
        + data
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = urllib.request.Request(
        f"{_chat_ui_url()}/api/v1/files/?process=true&process_in_background=true",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def _knowledge_by_name(token: str, name: str) -> dict | None:
    listed = _webui_request("GET", "/api/v1/knowledge/", token)
    rows = listed.get("items", listed) if isinstance(listed, dict) else listed
    return next((row for row in rows or [] if row.get("name") == name), None)


def _knowledge_files(token: str, knowledge_id: str) -> list[dict]:
    # The knowledge detail's ``files`` is empty in v0.12.3; the list has its own route.
    listed = _webui_request("GET", f"/api/v1/knowledge/{knowledge_id}/files", token)
    return listed.get("items", []) if isinstance(listed, dict) else listed or []


# Every file a gate uploads is recorded under its knowledge base, so cleanup
# deletes exactly those (failed uploads included) and never a user's file.
GATE_UPLOADS = RESULTS_ROOT / "knowledge-gate-uploads.json"
_GATE_UPLOADS_LOCK = threading.Lock()


def _gate_uploads() -> dict[str, list[str]]:
    return json.loads(GATE_UPLOADS.read_text()) if GATE_UPLOADS.exists() else {}


def _record_upload(knowledge_id: str, file_id: str) -> None:
    with _GATE_UPLOADS_LOCK:
        uploads = _gate_uploads()
        uploads.setdefault(knowledge_id, []).append(file_id)
        GATE_UPLOADS.parent.mkdir(parents=True, exist_ok=True)
        GATE_UPLOADS.write_text(json.dumps(uploads, indent=2) + "\n")


def _delete_knowledge(token: str, knowledge_id: str) -> None:
    """Delete a gate knowledge base and the files the gates recorded for it."""
    try:
        _webui_request("DELETE", f"/api/v1/knowledge/{knowledge_id}/delete", token)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    with _GATE_UPLOADS_LOCK:
        uploads = _gate_uploads()
        for file_id in uploads.pop(knowledge_id, []):
            try:
                _webui_request("DELETE", f"/api/v1/files/{file_id}", token)
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
        if GATE_UPLOADS.exists():
            GATE_UPLOADS.write_text(json.dumps(uploads, indent=2) + "\n")


def _ingest(
    token: str, knowledge_id: str, name: str, data: bytes, content_type: str
) -> tuple[str, float]:
    """Upload, wait for Open WebUI's extraction, add to the knowledge base; seconds taken."""
    started = time.monotonic()
    file_id = _webui_upload(token, name, data, content_type)["id"]
    _record_upload(knowledge_id, file_id)
    deadline = started + 2 * float(KNOWLEDGE["ingestion_seconds"])
    while True:
        status = _webui_request("GET", f"/api/v1/files/{file_id}/process/status", token).get(
            "status"
        )
        if status == "completed":
            break
        if status == "failed" or time.monotonic() > deadline:
            detail = _webui_request("GET", f"/api/v1/files/{file_id}", token)
            raise RuntimeError(
                f"{name}: processing {status}: {json.dumps(detail.get('data'))[:500]}"
            )
        time.sleep(1)
    _webui_request(
        "POST",
        f"/api/v1/knowledge/{knowledge_id}/file/add",
        token,
        payload={"file_id": file_id},
        timeout_s=1800,
    )
    return file_id, time.monotonic() - started


def knowledge(run_dir: Path) -> int:
    """Japanese knowledge base through the Chat UI, embedded by EmbeddingGemma 2 via Kairyu.

    ``KNOWLEDGE_KEEP=1`` keeps the knowledge base for the ``chat-ui`` gate;
    ``KNOWLEDGE_REUSE=1`` checks a kept one (after a restart) without ingesting.
    """
    token = _chat_ui_token()
    existing = _knowledge_by_name(token, KNOWLEDGE_NAME)
    report: dict = {"name": KNOWLEDGE_NAME, "reused": False}
    ingest_offset = log_offset(EMBED_PLACEMENT_LOG)
    before = _kairyu_api_counts()
    if existing is not None and os.environ.get("KNOWLEDGE_REUSE") == "1":
        knowledge_id = existing["id"]
        report["reused"] = True
        scan_id = next(
            row["id"]
            for row in _knowledge_files(token, knowledge_id)
            if row["filename"] == KNOWLEDGE_SCAN
        )
    else:
        if existing is not None:
            _delete_knowledge(token, existing["id"])
        knowledge_id = _webui_request(
            "POST",
            "/api/v1/knowledge/create",
            token,
            payload={
                "name": KNOWLEDGE_NAME,
                "description": "Kairyu gate: 試料の保存条件を含む日本語の技術報告書",
            },
        )["id"]
        scan = run_dir / KNOWLEDGE_SCAN
        scan_pdf(
            scan, int(KNOWLEDGE["scan_pages"]), {int(KNOWLEDGE["key_page"])}, namespace=run_dir.name
        )
        scan_id, seconds = _ingest(
            token, knowledge_id, KNOWLEDGE_SCAN, scan.read_bytes(), "application/pdf"
        )
        report["scan_ingestion_seconds"] = round(seconds, 1)
        distractor_seconds = []
        for name, text in distractor_documents(
            int(KNOWLEDGE["distractor_documents"]), namespace=run_dir.name
        ):
            distractor_seconds.append(
                _ingest(token, knowledge_id, name, text.encode(), "text/markdown")[1]
            )
        report["distractor_ingestion_seconds"] = round(sum(distractor_seconds), 1)
    ingested = placement_rows(EMBED_PLACEMENT_LOG, ingest_offset)
    content = (
        _webui_request("GET", f"/api/v1/files/{scan_id}/data/content", token).get("content") or ""
    )
    report["ocr_has_key"] = _has_key(content)
    report["ocr_characters"] = len(content)
    query_offset = log_offset(EMBED_PLACEMENT_LOG)
    started = time.monotonic()
    result = _webui_request(
        "POST",
        "/api/v1/retrieval/query/collection",
        token,
        payload={"collection_names": [knowledge_id], "query": KNOWLEDGE_QUERY},
    )
    report["query_seconds"] = round(time.monotonic() - started, 2)
    queried = wait_placements(EMBED_PLACEMENT_LOG, query_offset, 1)
    after = _kairyu_api_counts()
    documents = (result.get("documents") or [[]])[0]
    sources = [
        meta.get("name") or meta.get("source") for meta in (result.get("metadatas") or [[]])[0]
    ]
    top_n = int(KNOWLEDGE["top_n"])
    report["returned"] = len(documents)
    report["key_rank"] = next(
        (rank + 1 for rank, text in enumerate(documents) if _has_key(text)), None
    )
    report["sources"] = sources
    report["scores"] = (result.get("distances") or [[]])[0]
    # The embeddings went through Kairyu's pool: every replica took ingestion
    # batches (a fresh ingestion only) and the query reached the pool too.
    report["kairyu_embeddings"] = {
        "requests": after["/v1/embeddings"] - before["/v1/embeddings"],
        "ingestion_replicas": sorted({str(row.get("replica_id")) for row in ingested}),
        "query_placements": len(queried),
    }
    through_kairyu = report["kairyu_embeddings"]["query_placements"] >= 1 and (
        report["reused"]
        or len(report["kairyu_embeddings"]["ingestion_replicas"]) == EMBED_REPLICAS
    )
    passed = (
        report["ocr_has_key"]
        and report["key_rank"] is not None
        and report["key_rank"] <= top_n
        and report["query_seconds"] <= float(KNOWLEDGE["query_seconds"])
        and report.get("scan_ingestion_seconds", 0) <= float(KNOWLEDGE["ingestion_seconds"])
        and through_kairyu
    )
    report["passed"] = passed
    (run_dir / "knowledge.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    if os.environ.get("KNOWLEDGE_KEEP") != "1":
        _delete_knowledge(token, knowledge_id)
    return 0 if passed else 1


def _kairyu_api_counts() -> dict[str, int]:
    logs = subprocess.run(
        ["docker", "logs", f"{control.PROJECT}-kairyu-1"], capture_output=True, text=True
    )
    text = logs.stdout + logs.stderr
    return {
        path: len(re.findall(rf'"path": "{re.escape(path)}"', text))
        for path in ("/v1/responses", "/v1/chat/completions", "/v1/embeddings")
    }


def _browser(env: dict[str, str], volumes: list[str]) -> subprocess.CompletedProcess:
    """Run this example's browser script against the Chat UI as the admin user."""
    arguments = [
        item for key, value in env.items() for item in ("--env", f"{key}={value}")
    ] + [item for volume in volumes for item in ("--volume", volume)]
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--init",
            "--network",
            "host",
            "--env",
            f"WEBUI_SMOKE_BASE_URL={_chat_ui_url()}",
            "--env",
            f"WEBUI_SMOKE_MODEL={SERVED_MODEL}",
            "--env",
            f"WEBUI_SMOKE_TOKEN={_chat_ui_token()}",
            *arguments,
            "--volume",
            f"{HERE / 'webui-browser-smoke.mjs'}:/work/webui_browser_smoke.mjs:ro",
            browser_image(),
            "node",
            "/work/webui_browser_smoke.mjs",
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )


def chat_ui(run_dir: Path) -> int:
    """The Chat UI in Chromium: 323, then 324 in one chat, Red for an image, a knowledge answer.

    The knowledge-base turn needs the base kept by ``KNOWLEDGE_KEEP=1 verify.sh knowledge``.
    """
    fixtures = run_dir / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.new("RGB", (256, 256), (220, 20, 20)).save(fixtures / "red.png")
    before = _kairyu_api_counts()
    browser = _browser(
        {
            "WEBUI_SMOKE_KNOWLEDGE": KNOWLEDGE_NAME,
            "WEBUI_SMOKE_KNOWLEDGE_QUERY": KNOWLEDGE_QUERY,
            "WEBUI_SMOKE_KNOWLEDGE_SOURCE": KNOWLEDGE_SCAN,
        },
        [f"{fixtures}:/fixtures:ro"],
    )
    (run_dir / "browser.log").write_text(browser.stdout + browser.stderr)
    after = _kairyu_api_counts()
    report = {"before": before, "after": after, "browser_exit": browser.returncode}
    passed = (
        browser.returncode == 0
        and after["/v1/responses"] > before["/v1/responses"]
        and after["/v1/chat/completions"] == before["/v1/chat/completions"]
        # The knowledge turn's query embedding went through Kairyu.
        and after["/v1/embeddings"] > before["/v1/embeddings"]
    )
    report["passed"] = passed
    (run_dir / "chat-ui.json").write_text(json.dumps(report, indent=2) + "\n")
    print(browser.stdout[-2000:] + browser.stderr[-2000:])
    print(json.dumps(report))
    token = _chat_ui_token()
    kept = _knowledge_by_name(token, KNOWLEDGE_NAME)
    if kept is not None and os.environ.get("KNOWLEDGE_KEEP") != "1":
        _delete_knowledge(token, kept["id"])
    return 0 if passed else 1


class DoclingCpu:
    """Docling's own CPU use, sampled once a second from its cgroup counter.

    This is the OCR load itself: Open WebUI's upload, queue and embedding time
    does not count, only CPU time the Docling container spends.
    """

    def __init__(self, interval_s: float = 1.0) -> None:
        container = f"{control.PROJECT}-docling-1"
        identity = subprocess.run(
            ["docker", "inspect", "--format", "{{.Id}}", container],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        candidates = (
            Path(f"/sys/fs/cgroup/system.slice/docker-{identity}.scope/cpu.stat"),
            Path(f"/sys/fs/cgroup/docker/{identity}/cpu.stat"),
        )
        self.path = next((path for path in candidates if path.exists()), None)
        if self.path is None:
            raise RuntimeError(f"no cgroup CPU counter for {container}")
        self.interval_s = interval_s
        # (start, end, cores used) per sample, monotonic seconds.
        self.samples: list[tuple[float, float, float]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _usage_s(self) -> float:
        for line in self.path.read_text().splitlines():
            key, value = line.split()
            if key == "usage_usec":
                return int(value) / 1e6
        raise RuntimeError(f"{self.path} has no usage_usec")

    def _run(self) -> None:
        # A failed read ends sampling; ``error`` fails the gate and the time
        # after the last sample counts as unobserved, never as busy.
        try:
            last_time, last_usage = time.monotonic(), self._usage_s()
            while not self._stop.wait(self.interval_s):
                now, usage = time.monotonic(), self._usage_s()
                self.samples.append((last_time, now, (usage - last_usage) / (now - last_time)))
                last_time, last_usage = now, usage
        except Exception as error:  # noqa: BLE001 - reported through ``error``
            self.error = repr(error)

    def __enter__(self) -> DoclingCpu:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    def wait_busy(self, cores: float, seconds: int, timeout_s: float) -> bool:
        """True once the last ``seconds`` samples each used at least ``cores``."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and self.error is None:
            recent = self.samples[-seconds:]
            if len(recent) == seconds and all(used >= cores for *_, used in recent):
                return True
            time.sleep(self.interval_s / 2)
        return False

    def wait_idle(self, cores: float, seconds: int, timeout_s: float) -> bool:
        """True once the last ``seconds`` samples each used less than ``cores``."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and self.error is None:
            recent = self.samples[-seconds:]
            if len(recent) == seconds and all(used < cores for *_, used in recent):
                return True
            time.sleep(self.interval_s / 2)
        return False

    def wait_past(self, moment: float) -> None:
        """Wait (up to three intervals) for a sample that ends after ``moment``."""
        deadline = time.monotonic() + 3 * self.interval_s
        while time.monotonic() < deadline and self.error is None:
            if self.samples and self.samples[-1][1] >= moment:
                return
            time.sleep(self.interval_s / 4)

    def summary(self, started: float, ended: float, cores: float) -> dict:
        """Busy and observed shares of [started, ended], weighted by time.

        Seconds without a sample (a stopped sampler) are unobserved and so not
        busy: the busy share's denominator is the whole window.
        """
        window = max(ended - started, 0.0)
        observed = busy = core_seconds = 0.0
        for first, last, used in self.samples:
            overlap = max(0.0, min(last, ended) - max(first, started))
            observed += overlap
            core_seconds += overlap * used
            if used >= cores:
                busy += overlap
        return {
            "busy_share": busy / window if window else 0.0,
            "observed_share": observed / window if window else 0.0,
            "mean_cores": core_seconds / observed if observed else 0.0,
        }


def _row_tok_s(rows_dir: Path, level: int) -> float | None:
    path = rows_dir / f"capped-c{level}" / "row.json"
    if not path.exists():
        return None
    return float(json.loads(path.read_text())["summary"]["output_tokens_per_s"])


def knowledge_serving(run_dir: Path) -> int:
    """Serving rows (c1/c8) while Docling OCRs and Kairyu embeds: at least 95 % of unloaded rows.

    The same rows first run with the Chat UI idle (the reference, this run),
    then again while three ingestion streams, one more than Docling's two
    conversion workers, keep Docling at its CPU cap and send their chunks to
    the embedding replicas. The gate fails unless Docling used at least
    ``docling_busy_cores`` in ``docling_busy_share`` of the loaded rows' seconds.
    """
    import queue

    levels = [int(level) for level in KNOWLEDGE["serving_concurrency"]]
    if serving(run_dir, concurrency=levels, rows_dir=run_dir / "unloaded"):
        print("unloaded reference rows failed", file=sys.stderr)
        return 1
    token = _chat_ui_token()
    name = f"{KNOWLEDGE_NAME}-load"
    stale = _knowledge_by_name(token, name)
    if stale is not None:
        _delete_knowledge(token, stale["id"])
    knowledge_id = _webui_request(
        "POST",
        "/api/v1/knowledge/create",
        token,
        payload={"name": name, "description": "Kairyu gate load"},
    )["id"]
    # Open WebUI refuses a document whose content it already holds, so every
    # ingestion uploads a different scan; enough for the streams to outlast the rows.
    pending: queue.Queue[Path] = queue.Queue()
    for index in range(1, int(KNOWLEDGE["load_documents"]) + 1):
        scan = run_dir / f"load-{index:02d}-{KNOWLEDGE_SCAN}"
        scan_pdf(
            scan,
            int(KNOWLEDGE["scan_pages"]),
            {int(KNOWLEDGE["key_page"])},
            namespace=f"{run_dir.name}-load-{index}",
        )
        pending.put(scan)
    stop = threading.Event()
    ingestions: list[dict] = []
    errors: list[str] = []

    def load(stream: int) -> None:
        while not stop.is_set():
            try:
                scan = pending.get_nowait()
            except queue.Empty:
                return
            started = time.monotonic()
            try:
                _ingest(token, knowledge_id, scan.name, scan.read_bytes(), "application/pdf")
            except Exception as error:  # noqa: BLE001 - recorded and fails the gate
                errors.append(f"stream {stream}: {error!r}")
                stop.set()
                return
            ingestions.append({"stream": stream, "start": started, "end": time.monotonic()})

    busy_cores = float(KNOWLEDGE["docling_busy_cores"])
    report: dict = {"load_streams": int(KNOWLEDGE["load_streams"])}
    code = 1
    stuck = 0
    with DoclingCpu() as cpu:
        workers = [
            threading.Thread(target=load, args=(stream,), daemon=True)
            for stream in range(1, int(KNOWLEDGE["load_streams"]) + 1)
        ]
        # The first stream starts alone; the others start when its conversion
        # ends (Docling goes idle while it embeds and adds its document).
        busy_seconds = int(KNOWLEDGE["docling_start_busy_seconds"])
        start_timeout = float(KNOWLEDGE["docling_start_seconds"])
        workers[0].start()
        started_busy = cpu.wait_busy(busy_cores, busy_seconds, start_timeout)
        if len(workers) > 1:
            cpu.wait_idle(busy_cores, 2, float(KNOWLEDGE["ingestion_seconds"]))
            for worker in workers[1:]:
                worker.start()
            started_busy = started_busy and cpu.wait_busy(busy_cores, busy_seconds, start_timeout)
        rows_started = time.monotonic()
        try:
            if started_busy and not errors:
                code = serving(run_dir, concurrency=levels, rows_dir=run_dir / "loaded")
        finally:
            rows_ended = time.monotonic()
            stop.set()
            for worker in workers:
                worker.join(timeout=float(KNOWLEDGE["ingestion_seconds"]) * 2)
            stuck = sum(worker.is_alive() for worker in workers)
        cpu.wait_past(rows_ended)
        load_summary = cpu.summary(rows_started, rows_ended, busy_cores)
    if not stuck:
        _delete_knowledge(token, knowledge_id)
    sampler_error = cpu.error
    rows = {}
    for level in levels:
        loaded, reference = _row_tok_s(run_dir / "loaded", level), _row_tok_s(
            run_dir / "unloaded", level
        )
        if loaded is None or not reference:
            continue
        rows[str(level)] = {
            "output_tokens_per_s": round(loaded, 1),
            "reference": round(reference, 1),
            "fraction": round(loaded / reference, 3),
        }
    report.update(
        docling_started_busy=started_busy,
        docling_sampler_error=sampler_error,
        docling_busy_share=round(load_summary["busy_share"], 3),
        docling_observed_share=round(load_summary["observed_share"], 3),
        docling_mean_cores=round(load_summary["mean_cores"], 2),
        load_workers_stuck=stuck,
        rows_seconds=round(rows_ended - rows_started, 1),
        rows=rows,
        ingestions=[
            {
                "stream": row["stream"],
                "start_s": round(row["start"] - rows_started, 1),
                "end_s": round(row["end"] - rows_started, 1),
            }
            for row in sorted(ingestions, key=lambda row: row["start"])
        ],
        errors=errors,
    )
    passed = (
        code == 0
        and started_busy
        and sampler_error is None
        and not errors
        and not stuck
        and load_summary["busy_share"] >= float(KNOWLEDGE["docling_busy_share"])
        and len(rows) == len(levels)
        and all(
            row["fraction"] >= float(KNOWLEDGE["serving_min_fraction"]) for row in rows.values()
        )
    )
    report["passed"] = passed
    (run_dir / "knowledge-serving.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if passed else 1


# --- Word / Excel / PowerPoint (Chat UI) ------------------------------------------
#
# Attached files are read through Docling; edits run in the browser's Pyodide
# (code interpreter, on by default), which receives the chat's attachments in
# /mnt/uploads. Fixtures are built and results checked with python-docx,
# python-pptx and openpyxl inside the pinned Docling image, offline.

OFFICE_CASES = {
    "docx": {
        "fixture": "kairyu-office-report.docx",
        "result": "kairyu-office-report-revised.docx",
        "question": "添付の報告書によると、納品期限はいつですか。日付だけ答えてください。",
        "answer": r"(2026\s*年\s*)?12\s*月\s*18\s*日|2026[-/.]12[-/.]18|12/18",
        "edit": (
            "添付の kairyu-office-report.docx を編集してください。本文の「東京本社」をすべて"
            "「大阪支社」に置き換え、文書の最後に「改訂日: 2026年10月22日」という段落を追加し、"
            "/mnt/uploads/kairyu-office-report-revised.docx として保存してください。"
            "表やほかの本文は変えないでください。"
        ),
    },
    "xlsx": {
        "fixture": "kairyu-office-sales.xlsx",
        "result": "kairyu-office-sales-total.xlsx",
        "question": "添付の売上表で、2026年9月の売上はいくらですか。数値だけ答えてください。",
        "answer": r"\b161\b",
        "edit": (
            "添付の kairyu-office-sales.xlsx の「売上」シートの最終行の下に、A列が「合計」、"
            "B列がB2からB7の合計（SUM 関数）の行を追加し、ほかのセルは変えずに "
            "/mnt/uploads/kairyu-office-sales-total.xlsx として保存してください。"
        ),
    },
    "pptx": {
        "fixture": "kairyu-office-review.pptx",
        "result": "kairyu-office-review-final.pptx",
        "question": "添付の資料によると、新規顧客は何社増えましたか。数だけ答えてください。",
        "answer": r"\b42\b|四十二",
        "edit": (
            "添付の kairyu-office-review.pptx の最後に、タイトルが「まとめ」で、本文に"
            "箇条書きが3行あるスライドを1枚追加し、既存のスライドは変えずに "
            "/mnt/uploads/kairyu-office-review-final.pptx として保存してください。"
        ),
    },
}

_OFFICE_FIXTURES = r"""
import docx, openpyxl, pptx, sys
out = sys.argv[1]
d = docx.Document()
d.add_heading('出張報告書', 1)
d.add_paragraph('出張先は東京本社で、期間は2026年11月4日から6日までとする。')
d.add_paragraph('納品期限は2026年12月18日で、担当は品質管理課の佐藤とする。')
d.add_paragraph('東京本社の会議室は第3会議室を予約済みである。')
table = d.add_table(rows=1, cols=3)
for cell, text in zip(table.rows[0].cells, ('試料', '数量', '保管場所')):
    cell.text = text
for row in (('試料A', '12', '東京本社'), ('試料B', '8', '名古屋'), ('試料C', '5', '福岡')):
    for cell, text in zip(table.add_row().cells, row):
        cell.text = text
d.save(f'{out}/kairyu-office-report.docx')
book = openpyxl.Workbook()
sheet = book.active
sheet.title = '売上'
sheet.append(['月', '売上（万円）'])
for month, value in zip(range(4, 10), (120, 135, 128, 142, 150, 161)):
    sheet.append([f'2026年{month}月', value])
book.save(f'{out}/kairyu-office-sales.xlsx')
deck = pptx.Presentation()
for title, lines in (('第3四半期の振り返り', ['2026年7月から9月']),
                     ('主な成果', ['新規顧客は42社増えた', '解約率は1.8%に下がった']),
                     ('課題', ['納期遅延が3件あった', '問い合わせ対応の待ち時間'])):
    slide = deck.slides.add_slide(deck.slide_layouts[1])
    slide.shapes.title.text = title
    body = slide.placeholders[1].text_frame
    body.text = lines[0]
    for line in lines[1:]:
        body.add_paragraph().text = line
deck.save(f'{out}/kairyu-office-review.pptx')
print('fixtures ok')
"""

_OFFICE_CHECK = r"""
import docx, json, openpyxl, os, pptx, sys
out = sys.argv[1]
report = {}
def check(name, function):
    path = os.path.join(out, name)
    if not os.path.exists(path):
        report[name] = {'exists': False, 'passed': False}
        return
    try:
        report[name] = {'exists': True, **function(path)}
    except Exception as error:
        report[name] = {'exists': True, 'passed': False, 'error': repr(error)}
def revised_docx(path):
    d = docx.Document(path)
    text = '\n'.join(p.text for p in d.paragraphs)
    rows = [[c.text for c in r.cells] for r in d.tables[0].rows] if d.tables else []
    last = next((p.text for p in reversed(d.paragraphs) if p.text.strip()), '')
    result = {
        'replaced': '大阪支社' in text and '東京本社' not in text,
        'appended': '改訂日' in last and '2026年10月22日' in last,
        'kept_sentence': '納品期限は2026年12月18日' in text,
        'kept_table_rows': len(rows) == 4 and rows[0] == ['試料', '数量', '保管場所'],
    }
    return {**result, 'passed': all(result.values())}
def total_xlsx(path):
    sheet = openpyxl.load_workbook(path)['売上']
    rows = list(sheet.iter_rows(values_only=True))
    kept = rows[:7] == [('月', '売上（万円）')] + [
        (f'2026年{m}月', v) for m, v in zip(range(4, 10), (120, 135, 128, 142, 150, 161))]
    total = rows[7] if len(rows) > 7 else (None, None)
    value = str(total[1]).replace(' ', '').upper()
    result = {
        'kept_rows': kept,
        'total_label': total[0] == '合計',
        'total_sum': value in ('=SUM(B2:B7)', '836', '836.0'),
    }
    return {**result, 'total_cell': str(total[1]), 'passed': all(result.values())}
def final_pptx(path):
    deck = pptx.Presentation(path)
    titles = [s.shapes.title.text if s.shapes.title else '' for s in deck.slides]
    last = deck.slides[-1]
    lines = [p.text for shape in last.shapes if shape.has_text_frame and shape != last.shapes.title
             for p in shape.text_frame.paragraphs if p.text.strip()]
    result = {
        'kept_titles': titles[:3] == ['第3四半期の振り返り', '主な成果', '課題'],
        'added_slide': len(titles) == 4 and titles[3] == 'まとめ',
        'three_bullets': len(lines) >= 3,
    }
    return {**result, 'titles': titles, 'passed': all(result.values())}
check('kairyu-office-report-revised.docx', revised_docx)
check('kairyu-office-sales-total.xlsx', total_xlsx)
check('kairyu-office-review-final.pptx', final_pptx)
print('OFFICE-CHECK ' + json.dumps(report, ensure_ascii=False))
"""


def _docling_python(program: str, directory: Path, *arguments: str) -> str:
    """Run ``program`` offline in the pinned Docling image with ``directory`` at /work."""
    run = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--volume",
            f"{directory}:/work",
            "--entrypoint",
            "python3",
            control._compose_env()["DOCLING_IMAGE"],
            "-c",
            program,
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if run.returncode:
        raise RuntimeError(f"Docling-image program failed: {run.stderr[-2000:]}")
    return run.stdout


def office(run_dir: Path) -> int:
    """Word, Excel and PowerPoint in the Chat UI: read each attachment, then edit it to a new file.

    Chromium attaches each fixture with the clip, asks about its key fact, then
    asks for an edit. Every result is saved from the chat's Files panel (the
    code interpreter's file browser) and checked, whatever the reply says; a
    result the reply also delivers must be the same file. A reply that
    delivers nothing is the model not calling the download tool at its
    official sampling: recorded as a model miss, but at least one format must
    be delivered. Last, an image attached to that chat (after its tool calls)
    must be answered.
    """
    fixtures = run_dir / "fixtures"
    results = run_dir / "results"
    fixtures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    os.chmod(results, 0o777)  # the browser container writes downloads here
    _docling_python(_OFFICE_FIXTURES, fixtures, "/work")
    from PIL import Image

    Image.new("RGB", (256, 256), (220, 20, 20)).save(fixtures / "red.png")
    before = _kairyu_api_counts()
    browser = _browser(
        {
            "WEBUI_SMOKE_MODE": "office",
            "WEBUI_SMOKE_OFFICE": json.dumps(OFFICE_CASES, ensure_ascii=False),
        },
        [f"{fixtures}:/fixtures:ro", f"{results}:/results"],
    )
    (run_dir / "browser.log").write_text(browser.stdout + browser.stderr)
    after = _kairyu_api_counts()
    line = _docling_python(_OFFICE_CHECK, results, "/work")
    checks = json.loads(line.split("OFFICE-CHECK ", 1)[1])
    misses = _model_misses(browser.stdout)
    # A delivered copy must be the very file the code interpreter wrote.
    delivered = {
        case["result"]: (results / case["result"]).exists()
        and (results / f"{case['result']}.delivered").read_bytes()
        == (results / case["result"]).read_bytes()
        for case in OFFICE_CASES.values()
        if (results / f"{case['result']}.delivered").exists()
    }
    report = {
        "browser_exit": browser.returncode,
        "checks": checks,
        "delivered_equals_workspace": delivered,
        "model_misses": misses,
        "kairyu": {"before": before, "after": after},
    }
    passed = (
        browser.returncode == 0
        # Every edit is checked on the code interpreter's own file.
        and all(row.get("passed") for row in checks.values())
        and bool(delivered)
        and all(delivered.values())
        and len(delivered) + len(misses) == len(OFFICE_CASES)
        and after["/v1/responses"] > before["/v1/responses"]
        and after["/v1/chat/completions"] == before["/v1/chat/completions"]
    )
    report["passed"] = passed
    (run_dir / "office.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(browser.stdout[-3000:] + browser.stderr[-2000:])
    print(json.dumps(report, ensure_ascii=False))
    return 0 if passed else 1


def _model_misses(stdout: str) -> list:
    """Replies the browser recorded without a download link (``{"model_miss": ...}``)."""
    misses = []
    for line in stdout.splitlines():
        if line.startswith('{"model_miss"'):
            misses.append(json.loads(line)["model_miss"])
    return misses


# The owner's own request (2026-10-22, on a phone) that the 1017 Chat UI could
# not hand over as a file.
DOWNLOAD_REQUEST = "一年の事業計画のフォーマットをエクセルで作って"
_DOWNLOAD_CHECK = r"""
import hashlib, json, openpyxl, os, sys
out = sys.argv[1]
report = {}
for name in ("desktop.xlsx", "desktop-after-reload.xlsx", "phone.xlsx"):
    path = os.path.join(out, name)
    if not os.path.exists(path):
        report[name] = {"exists": False, "passed": False}
        continue
    try:
        book = openpyxl.load_workbook(path)
        cells = sum(
            1 for sheet in book.worksheets for row in sheet.iter_rows(values_only=True)
            for value in row if value not in (None, "")
        )
        report[name] = {
            "exists": True,
            "sheets": book.sheetnames,
            "filled_cells": cells,
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "passed": bool(book.sheetnames) and cells >= 10,
        }
    except Exception as error:
        report[name] = {"exists": True, "passed": False, "error": repr(error)}
first, again = report["desktop.xlsx"], report["desktop-after-reload.xlsx"]
digest = first.get("sha256")
report["same_after_reload"] = bool(digest) and digest == again.get("sha256")
print("DOWNLOAD-CHECK " + json.dumps(report, ensure_ascii=False))
"""


def file_download(run_dir: Path) -> int:
    """A file the model makes is saved from the reply's link: desktop, after a reload, phone.

    Desktop and phone each ask in up to three new chats until a reply links the
    file; replies without a link (the model not calling the download tool at
    its official sampling) are recorded as model misses.
    """
    results = run_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    os.chmod(results, 0o777)  # the browser container writes downloads here
    before = _kairyu_api_counts()
    browser = _browser(
        {"WEBUI_SMOKE_MODE": "download", "WEBUI_SMOKE_DOWNLOAD_REQUEST": DOWNLOAD_REQUEST},
        [f"{results}:/results"],
    )
    (run_dir / "browser.log").write_text(browser.stdout + browser.stderr)
    after = _kairyu_api_counts()
    line = _docling_python(_DOWNLOAD_CHECK, results, "/work")
    checks = json.loads(line.split("DOWNLOAD-CHECK ", 1)[1])
    report = {
        "browser_exit": browser.returncode,
        "checks": checks,
        "model_misses": _model_misses(browser.stdout),
        "kairyu": {"before": before, "after": after},
    }
    passed = (
        browser.returncode == 0
        and all(
            checks[name].get("passed")
            for name in ("desktop.xlsx", "desktop-after-reload.xlsx", "phone.xlsx")
        )
        and checks["same_after_reload"]
        and after["/v1/responses"] > before["/v1/responses"]
        and after["/v1/chat/completions"] == before["/v1/chat/completions"]
    )
    report["passed"] = passed
    (run_dir / "file-download.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(browser.stdout[-3000:] + browser.stderr[-2000:])
    print(json.dumps(report, ensure_ascii=False))
    return 0 if passed else 1


GATES = {
    "l1": l1,
    "serving": serving,
    "completed": completed,
    "tool-calling": tool_calling,
    "vision": vision,
    "reasoning": reasoning,
    "cancellation": cancellation,
    "long-context": long_context,
    "embeddings": embeddings,
    "restart": restart,
    "api-equivalence": api_equivalence,
    "docling": docling,
    "knowledge": knowledge,
    "chat-ui": chat_ui,
    "knowledge-serving": knowledge_serving,
    "office": office,
    "file-download": file_download,
}


def main() -> None:
    global REQUEST_LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", choices=[*GATES, "list"])
    parser.add_argument("--run-id")
    parser.add_argument("--no-start", action="store_true")
    args = parser.parse_args()
    if args.gate == "list":
        for name, function in GATES.items():
            print(f"{name:17} {(function.__doc__ or '-').strip().splitlines()[0]}")
        return
    if not args.no_start:
        control.up()
    run_dir = RESULTS_ROOT / f"{args.run_id or _run_id()}-{args.gate}"
    run_dir.mkdir(parents=True, exist_ok=True)
    REQUEST_LOG = run_dir / "requests.jsonl"
    manifest = {
        "schema_version": 1,
        "gate": args.gate,
        "started_at": datetime.now(UTC).isoformat(),
        "served_config_sha256": served_config_sha256(),
        "runtime": runtime_evidence(),
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        code = GATES[args.gate](run_dir)
    except Exception as error:  # noqa: BLE001 - a crashed gate is a failed gate
        print(f"{args.gate} failed: {error!r}", file=sys.stderr)
        code = 1
    manifest.update(completed_at=datetime.now(UTC).isoformat(), exit_code=code)
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"artifacts: {run_dir}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
