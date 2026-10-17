#!/usr/bin/env python3
"""Measured verification over /v1/responses: serving matrix, placement, tools, API equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = json.loads((HERE / "example.json").read_text(encoding="utf-8"))
REPLICAS = int(SPEC["allocation"]["replicas"])
TENSOR_PARALLEL = int(SPEC["allocation"]["tensor_parallel_size"])
SERVED_MODEL = SPEC["model"]["served_name"]
# Files whose bytes define the served configuration (the Qwen chat template
# is shared with the single-GPU example and mounted from there).
SERVED_CONFIG_FILES = (
    HERE / "example.json",
    HERE / "compose.yaml",
    HERE / "kairyu.yaml",
    HERE / "../qwen3.8-27b-1gpu/chat_template.jinja",
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
# Host-side view of the pool's placement_log_path bind mount (compose.yaml).
PLACEMENT_LOG = ENVIRONMENT_STORAGE / "placement-log" / Path(SPEC["pool"]["placement_log"]).name


def _run(command: list[str], *, log: Path | None = None, check: bool = True) -> int:
    print("+ " + " ".join(command), flush=True)
    if log is None:
        return subprocess.run(command, cwd=ROOT, check=check).returncode
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if check and process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return process.returncode


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _ensure_environment(no_start: bool) -> None:
    if not no_start:
        _run([sys.executable, str(HERE / "control.py"), "up"])


def _serving_dataset(
    path: Path, requests: int, approximate_tokens: int, *, namespace: str
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
        # The run/row identity and the unique case come first, so neither
        # vLLM prefix caching nor Kairyu's prefix-aware placement can turn a
        # row into a shared-prefix microbenchmark. Server-reported usage
        # remains the source of truth for the actual token count.
        prompt = f"Run {namespace}, case {request}: " + " ".join(words)
        rows.append({"conversations": [{"from": "human", "value": prompt}]})
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def _bench(
    dataset: Path,
    *,
    requests: int,
    concurrency: int,
    max_tokens: int,
    results_dir: Path,
    log: Path,
) -> int:
    return _run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "verification/l1/performance/serving_bench.py"),
            "--base-url",
            f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}/v1",
            "--model",
            SERVED_MODEL,
            "--dataset",
            str(dataset),
            "--num-requests",
            str(requests),
            "--concurrency",
            str(concurrency),
            # /v1/responses, the API the Chat UI uses, has no ignore_eos,
            # min_tokens or seed: each request ends naturally within
            # --max-tokens (m11 D7 amendment 2026-10-17).
            "--api",
            "responses",
            "--max-tokens",
            str(max_tokens),
            "--temperature",
            "1.0",
            "--timeout",
            "86400",
            "--results-dir",
            str(results_dir),
            "--tensor-parallel",
            str(TENSOR_PARALLEL),
            "--dp-replicas",
            str(REPLICAS),
        ],
        log=log,
        check=False,
    )


def _placement_offset(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _placement_counts(
    path: Path,
    offset: int,
    *,
    request_ids: set[str] | None = None,
) -> Counter[str]:
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
            if request_ids is not None and row.get("request_id") not in request_ids:
                continue
            replica = row.get("replica_id", row.get("replica"))
            counts[str(replica)] += 1
    return counts


def _wait_for_placement_counts(
    path: Path,
    offset: int,
    *,
    expected_requests: int,
    request_ids: set[str] | None = None,
    settle_s: float = 10.0,
) -> Counter[str]:
    deadline = time.monotonic() + settle_s
    counts = _placement_counts(path, offset, request_ids=request_ids)
    while sum(counts.values()) < expected_requests and time.monotonic() < deadline:
        time.sleep(0.5)
        counts = _placement_counts(path, offset, request_ids=request_ids)
    return counts


def _placement_report(
    path: Path,
    offset: int,
    *,
    expected_requests: int,
    replicas: int,
    gated: bool,
    max_share_of_mean: float,
    settle_s: float = 10.0,
) -> dict:
    """Per-replica placement counts for one row, from the pool's JSONL log.

    The log is written asynchronously, so poll until the row's placements
    have landed (or the settle window expires). The gate requires exactly
    ``expected_requests`` placements, every replica to receive traffic, and
    no replica to exceed ``max_share_of_mean``
    times the ideal even share; a serial row (c1) is reported only, because
    least-outstanding ties resolve to the lowest replica id by design.
    """

    counts = _wait_for_placement_counts(
        path,
        offset,
        expected_requests=expected_requests,
        settle_s=settle_s,
    )
    total = sum(counts.values())
    mean = expected_requests / replicas if replicas else 0.0
    largest = max(counts.values(), default=0)
    even = (
        total == expected_requests
        and len(counts) == replicas
        and largest <= max_share_of_mean * mean
    )
    return {
        "schema_version": 1,
        "placements": total,
        "expected_requests": expected_requests,
        "replicas": replicas,
        "per_replica": dict(sorted(counts.items())),
        "largest_share_of_mean": round(largest / mean, 3) if mean else None,
        "max_share_of_mean": max_share_of_mean,
        "gated": gated,
        "passed": even if gated else None,
    }


def _validate_serving_row(row_dir: Path, requests: int, max_output_tokens: int) -> int:
    """Reject partial streams and nominally successful zero-token runs.

    Rows run on /v1/responses, so each request ends naturally within
    ``max_output_tokens``: every sample must report 1..max_output_tokens
    generated tokens and the row total must be their sum.
    """

    artifacts = list(row_dir.glob("*-serving.json"))
    if len(artifacts) != 1:
        print(
            f"serving row produced {len(artifacts)} result files; expected exactly one",
            file=sys.stderr,
        )
        return 1
    try:
        result = json.loads(artifacts[0].read_text(encoding="utf-8"))
        summary = result["summary"]
        samples = result["samples"]
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"invalid serving result: {error}", file=sys.stderr)
        return 1
    tokens = [sample.get("completion_tokens") for sample in samples]
    bounded = all(
        isinstance(count, int) and 1 <= count <= max_output_tokens for count in tokens
    )
    complete = (
        summary.get("requests") == requests
        and len(samples) == requests
        and bounded
        and summary.get("completion_tokens_total") == sum(tokens)
        and isinstance(summary.get("output_tokens_per_s"), (int, float))
        and summary["output_tokens_per_s"] > 0
    )
    if not complete:
        print(
            "serving row did not produce complete evidence: "
            f"requests={summary.get('requests')!r}, "
            f"completion_tokens_total={summary.get('completion_tokens_total')!r}, "
            f"output_tokens_per_s={summary.get('output_tokens_per_s')!r}, "
            f"samples={len(samples)!r}, completion_tokens={tokens!r}",
            file=sys.stderr,
        )
        return 1
    return 0


def serving(run_dir: Path) -> int:
    config = SPEC["verification"]["serving"]
    gate = config["placement_gate"]
    requests = int(config["requests_per_concurrency"])
    max_output_tokens = int(config["max_output_tokens"])
    prompt_tokens = int(config["prompt_tokens_approx"])
    run_dir.mkdir(parents=True, exist_ok=True)

    # One short request per replica so every replica's first-request work
    # (autotune, graph capture) is done before the measured rows.
    warmup_dataset = run_dir / "warmup-8k.json"
    _serving_dataset(warmup_dataset, REPLICAS, prompt_tokens, namespace=f"{run_dir.name}-warmup")
    if _bench(
        warmup_dataset,
        requests=REPLICAS,
        concurrency=REPLICAS,
        max_tokens=32,
        results_dir=run_dir / "warmup",
        log=run_dir / "warmup.log",
    ):
        print("warm-up row failed", file=sys.stderr)
        return 1

    failures = 0
    for concurrency in config["concurrency"]:
        row_dir = run_dir / f"serving-c{concurrency}"
        dataset = run_dir / f"serving-8k-c{concurrency}.json"
        _serving_dataset(
            dataset, requests, prompt_tokens, namespace=f"{run_dir.name}-c{concurrency}"
        )
        offset = _placement_offset(PLACEMENT_LOG)
        code = _bench(
            dataset,
            requests=requests,
            concurrency=concurrency,
            max_tokens=max_output_tokens,
            results_dir=row_dir,
            log=run_dir / f"serving-c{concurrency}.log",
        )
        if code == 0:
            code = _validate_serving_row(row_dir, requests, max_output_tokens)
        if code == 0:
            report = _placement_report(
                PLACEMENT_LOG,
                offset,
                expected_requests=requests,
                replicas=REPLICAS,
                gated=concurrency >= int(gate["min_concurrency"]),
                max_share_of_mean=float(gate["max_share_of_mean"]),
            )
            (row_dir / "placement.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"{row_dir.name}: placements {json.dumps(report['per_replica'])}")
            if report["passed"] is False:
                print(
                    f"{row_dir.name}: replica placement is not even "
                    f"(largest share {report['largest_share_of_mean']}x mean)",
                    file=sys.stderr,
                )
                code = 1
        failures += code != 0
        if code:
            break
    return 1 if failures else 0


# --- Tool-calling gate (PR #584 review: a served example must emit OpenAI
# tool calls, or tool-driven agents such as SWE-bench Pro's mini-swe-agent
# fail every turn). The request shape mirrors that agent: an auto-choice
# `bash` function tool, sent through /v1/responses (the API the Chat UI uses,
# m11 D7 amendment 2026-10-17) and read back into the chat shape checked here.
_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run."}
            },
            "required": ["command"],
        },
    },
}
_TOOL_SYSTEM = (
    "You are an agent operating a computer shell. Every response MUST include "
    "at least one bash tool call; never answer in plain text."
)
_TOOL_USER = "List the files in the current directory."


def _validate_tool_call_message(message: dict, finish_reason: object) -> str | None:
    """Reject responses a bash-tool agent loop cannot execute (None = valid)."""

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
        if not isinstance(arguments.get("command"), str) or not arguments["command"]:
            return f"tool call arguments lack a command string: {arguments!r}"
    return None


def _tool_placement_error(
    counts: Counter[str],
    *,
    expected_requests: int,
    replicas: int,
) -> str | None:
    total = sum(counts.values())
    if total != expected_requests:
        return (
            f"recorded {total} correlated placements, expected {expected_requests}: "
            f"{dict(counts)}"
        )
    if len(counts) != replicas:
        return f"only {len(counts)} of {replicas} replicas served tool calls: {dict(counts)}"
    return None


def _reassemble_stream_tool_calls(sse_body: str) -> tuple[dict, object]:
    """Fold SSE chat chunks into (message-like dict, final finish_reason)."""

    calls: dict[int, dict[str, list[str]]] = {}
    finish_reason: object = None
    for line in sse_body.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[len("data: "):])
        for choice in chunk.get("choices", ()):
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            for delta_call in (choice.get("delta") or {}).get("tool_calls") or ():
                index = delta_call.get("index")
                if not isinstance(index, int):
                    return {"tool_calls": None}, "missing tool_call delta index"
                slot = calls.setdefault(index, {"name": [], "arguments": []})
                function = delta_call.get("function") or {}
                if function.get("name"):
                    slot["name"].append(function["name"])
                if function.get("arguments"):
                    slot["arguments"].append(function["arguments"])
    message = {
        "tool_calls": [
            {
                "function": {
                    "name": "".join(slot["name"]),
                    "arguments": "".join(slot["arguments"]),
                }
            }
            for _, slot in sorted(calls.items())
        ]
        or None
    }
    return message, finish_reason


# --- Responses gate adapter (m11 D7 amendment 2026-10-17) ---------------------
# The example gates are written against chat-shaped payloads and replies. These
# helpers send each payload to Kairyu's public /v1/responses instead (the API
# the Chat UI uses) and read the Responses reply back into the chat shape the
# gate checks, so a gate's assertions stay as they were.

# Responses fields a payload may already carry (the Chat UI filter's output
# has reasoning and store) pass through as they are.
_RESPONSES_PASSTHROUGH = frozenset(
    {
        "model",
        "stream",
        "temperature",
        "top_p",
        "parallel_tool_calls",
        "metadata",
        "user",
        "reasoning",
        "store",
    }
)


def _responses_content(content: object, *, assistant: bool) -> object:
    if isinstance(content, str) or content is None:
        return content or ""
    parts = []
    for part in content:
        if part.get("type") == "text":
            kind = "output_text" if assistant else "input_text"
            parts.append({"type": kind, "text": part["text"]})
        elif part.get("type") == "image_url" and not assistant:
            image = part["image_url"]
            url = image["url"] if isinstance(image, dict) else image
            parts.append({"type": "input_image", "image_url": url, "detail": "auto"})
        else:
            raise ValueError(f"chat content part {part.get('type')!r} has no Responses form")
    return parts


def _responses_tool(tool: dict) -> dict:
    function = tool["function"]
    flat = {"type": "function", "name": function["name"]}
    for key in ("description", "parameters", "strict"):
        if key in function:
            flat[key] = function[key]
    return flat


def responses_request(payload: dict) -> dict:
    """The /v1/responses body for one chat-shaped gate payload (stateless)."""

    body: dict = {"store": False}
    for key, value in payload.items():
        if key == "messages":
            continue
        if key in _RESPONSES_PASSTHROUGH:
            body[key] = value
        elif key in ("max_tokens", "max_completion_tokens"):
            body["max_output_tokens"] = value
        elif key == "reasoning_effort":
            body["reasoning"] = {"effort": value}
        elif key == "tools":
            body["tools"] = [_responses_tool(tool) for tool in value]
        elif key == "tool_choice":
            body["tool_choice"] = (
                {"type": "function", "name": value["function"]["name"]}
                if isinstance(value, dict)
                else value
            )
        elif key == "response_format":
            if value.get("type") == "json_schema":
                schema = value["json_schema"]
                body["text"] = {"format": {"type": "json_schema", **schema}}
            else:
                body["text"] = {"format": {"type": value["type"]}}
        elif key == "stream_options":
            continue  # a Responses stream always ends with usage
        else:
            raise ValueError(f"chat field {key!r} has no Responses form")
    items: list[dict] = []
    for message in payload["messages"]:
        role = message["role"]
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
            continue
        if message.get("reasoning_content"):
            text = message["reasoning_content"]
            items.append(
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": text}],
                }
            )
        content = message.get("content")
        if content:
            items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": _responses_content(content, assistant=role == "assistant"),
                }
            )
        for call in message.get("tool_calls") or ():
            items.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                }
            )
    body["input"] = items
    return body


def _finish_reason(response: dict, calls: list) -> str:
    if response.get("status") == "completed":
        return "tool_calls" if calls else "stop"
    if response.get("status") == "incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason")
        return "content_filter" if reason == "content_filter" else "length"
    return str(response.get("status"))


def chat_view(response: dict) -> dict:
    """One Responses object read back as a chat completion (kairyu_* kept)."""

    text: list[str] = []
    reasoning: list[str] = []
    calls: list[dict] = []
    has_message = False
    for item in response.get("output") or ():
        kind = item.get("type")
        if kind == "message":
            has_message = True
            text += [
                part.get("text", "")
                for part in item.get("content") or ()
                if part.get("type") == "output_text"
            ]
        elif kind == "reasoning":
            parts = item.get("content") or item.get("summary") or ()
            reasoning += [part.get("text", "") for part in parts]
        elif kind == "function_call":
            calls.append(
                {
                    "id": item.get("call_id"),
                    "type": "function",
                    "function": {"name": item.get("name"), "arguments": item.get("arguments")},
                }
            )
    usage = response.get("usage") or {}
    chat_usage = {
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "prompt_tokens_details": {
            "cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens")
        },
        "completion_tokens_details": {
            "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
        },
        **{key: value for key, value in usage.items() if key.startswith("orchestration_")},
    }
    return {
        "id": response.get("id"),
        "object": "chat.completion",
        "created": response.get("created_at"),
        "model": response.get("model"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(text) if has_message else None,
                    "reasoning_content": "".join(reasoning) or None,
                    "tool_calls": calls or None,
                },
                "finish_reason": _finish_reason(response, calls),
            }
        ],
        "usage": chat_usage,
        **{key: value for key, value in response.items() if key.startswith("kairyu_")},
    }


def chat_stream_view(sse_text: str) -> str:
    """A Responses SSE body read back as chat SSE chunks ending in [DONE]."""

    chunks: list[dict] = []
    indexes: dict[int, int] = {}
    streamed_arguments: set[int] = set()

    def delta(**fields) -> None:
        chunks.append({"choices": [{"index": 0, "delta": fields, "finish_reason": None}]})

    for line in sse_text.splitlines():
        if not line.startswith("data:") or not line[len("data:") :].strip():
            continue
        event = json.loads(line[len("data:") :])
        kind = event.get("type")
        if kind == "response.output_text.delta":
            delta(content=event["delta"])
        elif kind in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
            delta(reasoning_content=event["delta"])
        elif kind == "response.output_item.added" and event["item"].get("type") == "function_call":
            index = indexes.setdefault(event["output_index"], len(indexes))
            item = event["item"]
            delta(
                tool_calls=[
                    {
                        "index": index,
                        "id": item.get("call_id"),
                        "type": "function",
                        "function": {"name": item.get("name"), "arguments": ""},
                    }
                ]
            )
        elif kind == "response.function_call_arguments.delta":
            index = indexes[event["output_index"]]
            streamed_arguments.add(index)
            delta(tool_calls=[{"index": index, "function": {"arguments": event["delta"]}}])
        elif kind == "response.output_item.done" and event["item"].get("type") == "function_call":
            index = indexes[event["output_index"]]
            if index not in streamed_arguments:
                arguments = event["item"].get("arguments") or ""
                delta(tool_calls=[{"index": index, "function": {"arguments": arguments}}])
        elif kind in ("response.completed", "response.incomplete"):
            view = chat_view(event["response"])
            extensions = {k: v for k, v in view.items() if k.startswith("kairyu_")}
            finish = view["choices"][0]["finish_reason"]
            chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
            chunks.append({"choices": [], "usage": view["usage"], **extensions})
        elif kind in ("error", "response.failed"):
            error = event if kind == "error" else (event.get("response") or {}).get("error")
            chunks.append({"error": error})
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def _post_responses(
    payload: dict,
    *,
    timeout_s: float = 600.0,
    include_request_id: bool = False,
) -> tuple[int, object] | tuple[int, object, str | None]:
    """One chat-shaped gate payload through /v1/responses, read back as chat."""

    import urllib.error
    import urllib.request

    url = (
        f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}"
        "/v1/responses"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(responses_request(payload)).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            status = response.status
            request_id = response.headers.get("x-request-id")
    except urllib.error.HTTPError as error:
        result: tuple[int, object] = (error.code, error.read().decode("utf-8", "replace"))
        request_id = error.headers.get("x-request-id") if error.headers else None
    else:
        reply = chat_stream_view(body) if payload.get("stream") else chat_view(json.loads(body))
        result = (status, reply)
    return (*result, request_id) if include_request_id else result


def _tool_call_request(**overrides) -> dict:
    payload = {
        "model": SERVED_MODEL,
        "messages": [
            {"role": "system", "content": _TOOL_SYSTEM},
            {"role": "user", "content": _TOOL_USER},
        ],
        "tools": [_BASH_TOOL],
        "parallel_tool_calls": True,
        "max_tokens": 8192,
    }
    payload.update(overrides)
    return payload


def tool_calling(run_dir: Path) -> int:
    import concurrent.futures

    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"schema_version": 1, "cases": {}}
    failures: list[str] = []

    def record(name: str, error: str | None, detail: object = None) -> None:
        report["cases"][name] = {"passed": error is None, "error": error, "detail": detail}
        if error is not None:
            failures.append(f"{name}: {error}")

    # Case 1 (mini-swe-agent shape) fanned to 2x replicas concurrently, so the
    # placement log proves every replica serves tool calls, not just one.
    offset = _placement_offset(PLACEMENT_LOG)
    fan = 2 * REPLICAS
    with concurrent.futures.ThreadPoolExecutor(max_workers=fan) as pool:
        results = list(
            pool.map(
                lambda _: _post_responses(_tool_call_request(), include_request_id=True),
                range(fan),
            )
        )
    first_message: dict | None = None
    request_ids: set[str] = set()
    error = None
    for status, body, request_id in results:
        if isinstance(request_id, str) and request_id:
            request_ids.add(request_id)
        if status != 200 or not isinstance(body, dict):
            error = f"HTTP {status}: {str(body)[:300]}"
            break
        choice = body["choices"][0]
        case_error = _validate_tool_call_message(choice["message"], choice["finish_reason"])
        if case_error is not None:
            error = case_error
            break
        first_message = choice["message"]
    if error is None and len(request_ids) != fan:
        error = f"received {len(request_ids)} unique x-request-id headers, expected {fan}"
    record(f"auto_tool_call_x{fan}", error)
    if len(request_ids) == fan:
        counts = _wait_for_placement_counts(
            PLACEMENT_LOG,
            offset,
            expected_requests=fan,
            request_ids=request_ids,
        )
        placement_error = _tool_placement_error(
            counts,
            expected_requests=fan,
            replicas=REPLICAS,
        )
    else:
        counts = Counter()
        placement_error = (
            f"cannot correlate placements: received {len(request_ids)} unique "
            f"x-request-id headers, expected {fan}"
        )
    record(
        "all_replicas_served_tool_calls",
        placement_error,
        dict(sorted(counts.items())),
    )

    # Case 2: the agent loop's second turn (assistant tool_calls + tool result).
    # The task still needs a call after the listing: once the result answers
    # the request, the model reports it in text (PR #647).
    if first_message is not None:
        call = first_message["tool_calls"][0]
        call_id = call.get("id") or "call_0"
        status, body = _post_responses(
            _tool_call_request(
                messages=[
                    {"role": "system", "content": _TOOL_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            "List the files in the current directory, "
                            "then show the first line of README.md."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": dict(call["function"]),
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": "README.md\nsrc\ntests\n",
                    },
                ]
            )
        )
        if status != 200 or not isinstance(body, dict):
            record("tool_result_turn", f"HTTP {status}: {str(body)[:300]}")
        else:
            choice = body["choices"][0]
            record(
                "tool_result_turn",
                _validate_tool_call_message(choice["message"], choice["finish_reason"]),
            )
    else:
        record("tool_result_turn", "skipped: no tool call from case 1")

    # Case 3: streaming deltas carry the same call.
    status, body = _post_responses(_tool_call_request(stream=True))
    if status != 200 or not isinstance(body, str):
        record("streamed_tool_call", f"HTTP {status}: {str(body)[:300]}")
    else:
        message, finish_reason = _reassemble_stream_tool_calls(body)
        record("streamed_tool_call", _validate_tool_call_message(message, finish_reason))

    # Case 4: explicit reasoning_effort must think AND still emit the call.
    status, body = _post_responses(_tool_call_request(reasoning_effort="high"), timeout_s=1200.0)
    if status != 200 or not isinstance(body, dict):
        record("thinking_tool_call", f"HTTP {status}: {str(body)[:300]}")
    else:
        choice = body["choices"][0]
        error = _validate_tool_call_message(choice["message"], choice["finish_reason"])
        if error is None and not choice["message"].get("reasoning_content"):
            error = "reasoning_content is empty under reasoning_effort=high"
        record("thinking_tool_call", error)

    # Case 5: ordinary chat stays non-thinking by default.
    status, body = _post_responses(
        {
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "max_tokens": 512,
        }
    )
    if status != 200 or not isinstance(body, dict):
        record("default_direct_chat", f"HTTP {status}: {str(body)[:300]}")
    else:
        message = body["choices"][0]["message"]
        error = None
        if not message.get("content"):
            error = "empty content"
        elif message.get("reasoning_content"):
            error = "reasoning_content present without reasoning_effort"
        record("default_direct_chat", error)

    report["passed"] = not failures
    (run_dir / "tool-calling.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for line in failures:
        print(f"tool-calling: {line}", file=sys.stderr)
    print(f"tool-calling: {'PASS' if not failures else 'FAIL'} ({len(report['cases'])} cases)")
    return 1 if failures else 0


# --- api-equivalence gate (m11 D4 amendment 2026-10-17): the same Responses
# requests sent to one L1 replica directly and to Kairyu L3 must agree, and
# every replica's own request counters must show L3 reached it on
# /v1/responses, never /v1/chat/completions.
L1_SERVICE_PREFIX = "qwen"


def _l1_containers() -> list[str]:
    """The running L1 replica containers, in replica order."""

    project = SPEC["environment"].replace(".", "-")
    containers = []
    for index in range(REPLICAS):
        service = f"{L1_SERVICE_PREFIX}-{index}"
        found = subprocess.run(
            [
                "docker",
                "ps",
                "--quiet",
                "--filter",
                f"label=com.docker.compose.project={project}",
                "--filter",
                f"label=com.docker.compose.service={service}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
        if len(found) != 1:
            raise RuntimeError(f"expected one running {service} container, found {len(found)}")
        containers.append(found[0])
    return containers


def api_equivalence(run_dir: Path) -> int:
    containers = _l1_containers()
    metrics = [
        argument
        for container in containers
        for argument in ("--l1-metrics", f"{container}=http://127.0.0.1:8000/metrics")
    ]
    return _run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "verification/product/correctness/upstream_api_equivalence.py"),
            "--l1",
            "http://127.0.0.1:8000/v1",
            "--l1-docker",
            containers[0],
            "--l1-model",
            SERVED_MODEL,
            *metrics,
            "--l3",
            f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}/v1",
            "--l3-model",
            SERVED_MODEL,
            "--cases",
            "text,stream,effort,tools,image,cap,openwebui",
            # L3 max reaches L1 as xhigh; any explicit effort selects thinking.
            "--efforts",
            "low,high,xhigh",
            "--expect-reasoning",
            "--exact-input-tokens",
            "--out",
            str(run_dir / "api-equivalence.json"),
        ],
        log=run_dir / "api-equivalence.log",
        check=False,
    )


def _served_config_sha256() -> str:
    digest = hashlib.sha256()
    for path in SERVED_CONFIG_FILES:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "verification", choices=("serving", "tool-calling", "api-equivalence", "list")
    )
    parser.add_argument("--run-id")
    parser.add_argument("--no-start", action="store_true")
    args = parser.parse_args()
    if args.verification == "list":
        print(
            "serving       Responses 8K-input/up-to-256-output TTFT and throughput at "
            "c=1,8,16,32,64 plus the per-row replica placement gate"
        )
        print(
            "tool-calling  OpenAI bash-tool agent contract (auto call, tool-result turn, "
            "streaming, thinking, non-thinking default) on every replica"
        )
        print(
            "api-equivalence  the same Responses requests to one L1 replica and to L3 "
            "agree; every replica saw them on /v1/responses"
        )
        return

    _ensure_environment(args.no_start)
    run_dir = RESULTS_ROOT / (args.run_id or _run_id())
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "started_at": datetime.now(UTC).isoformat(),
        "requested": args.verification,
        "served_config_sha256": _served_config_sha256(),
        "spec": SPEC,
    }
    (run_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    target = {
        "serving": serving,
        "tool-calling": tool_calling,
        "api-equivalence": api_equivalence,
    }[args.verification]
    try:
        code = target(run_dir)
    except Exception as error:
        print(f"{args.verification} failed: {error}", file=sys.stderr)
        code = 1
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["exit_codes"] = {args.verification: code}
    (run_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"artifacts: {run_dir}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
