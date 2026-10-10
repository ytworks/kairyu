#!/usr/bin/env python3
"""GPU gates for routed DeepSeek-V4.1 answers (VCO-D21, VCO-D22).

Every gate has a time budget and writes its per-request evidence (latency,
tokens, tok/s, route, efforts, form check) to
model-volumes/<environment>/results/<gate>-<UTC>.json, with the Quyet reads of
the gate (count, states cut at Quyet's 6,000 tokens, read times) from the
adapters' logs. One public model, kairyu-verified-tool: Quyet routes TOOL (the
next reply needs a tool call) to the verified tool route and THINK to
deepseek_think.

  l1                  every DeepSeek DP rank (thinking and chat, grammar JSON),
                      each Quyet replica's vLLM and System One, and one
                      structured tool call from the public model
  routing             datasets/tool-routing-set.json: TOOL miss rate < 10 % on
                      the calibration and held-out halves, THINK precision >= 90 %
  think-route         everyday requests stream from deepseek_think at the default effort
  effort              the caller's effort reaches the think route and the verified
                      tool route's candidates, answer and fixes; quyet-route judges
                      the route
  verified-tool-route agent turns (datasets/tool-turns.json) take the verified
                      tool route: DeepSeek's 10-16 candidates, 6 judgments each on
                      quyet-judge, the answer and its form check on quyet-judge,
                      and a structured call to a declared tool
  fallback            quyet-route down: an agent turn is answered on the think
                      route; quyet-judge down: still routed, the answer without
                      judgments and published unverified; both back: judged and
                      form-checked again
  serving             agent turns at c1/c4/c8/c16: every reply a structured call;
                      latency, tokens, tok/s per route
  serving-routed      the routing set at c1/c4/c8/c16: route mix, latency, tokens per route
  browser             Open WebUI answers in a real browser
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import control  # noqa: E402

SPEC = control.SPEC
DATASETS = HERE / "datasets"
MODEL = control.MODEL
# Concurrency -> requests, kept from the previous verified example's serving gates.
PLAN = {int(level): count for level, count in SPEC["verification"]["concurrency_plan"].items()}


def _env() -> dict[str, str]:
    return control._compose_env()


# When the running gate started (UTC, RFC 3339): the adapters' log lines since
# then are the gate's Quyet reads.
GATE_STARTED: str | None = None
_READ_LINE = re.compile(
    r"read questions=(\d+) input_tokens=(\d+) truncated=(True|False) "
    r"state_tokens=(\S+) model_ms=(\d+)"
)


def _quyet_reads(since: str | None) -> dict[str, dict]:
    """Per Quyet adapter: reads, questions, states cut at 6,000 tokens and read
    times since ``since``, from the adapter's ``read`` log lines."""

    report = {}
    for service in control.QUYET_SERVICES:
        command = ["docker", "logs", f"{control.PROJECT}-{service}-1"]
        if since:
            command[2:2] = ["--since", since]
        logs = subprocess.run(command, capture_output=True, text=True, check=False)
        rows = [match.groups() for match in _READ_LINE.finditer(logs.stdout + logs.stderr)]
        model_ms = sorted(int(row[4]) for row in rows)
        cut = [int(row[3]) for row in rows if row[2] == "True"]
        report[service] = {
            "reads": len(rows),
            "questions": sum(int(row[0]) for row in rows),
            "truncated": len(cut),
            "truncated_state_tokens_max": max(cut) if cut else None,
            "model_ms_p50": statistics.median(model_ms) if model_ms else None,
            "model_ms_max": model_ms[-1] if model_ms else None,
        }
    return report


def _api(env: dict[str, str]) -> str:
    return f"http://127.0.0.1:{env['API_PORT']}"


def _results_dir() -> Path:
    path = control.environment_storage() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(gate: str, payload: dict) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    payload = {**payload, "quyet_reads": _quyet_reads(GATE_STARTED)}
    print(f"quyet reads: {json.dumps(payload['quyet_reads'])}", flush=True)
    path = _results_dir() / f"{gate}-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"evidence: {path}", flush=True)
    return path


def _dataset(name: str) -> list[dict]:
    """A dataset's items, each with its named tool set expanded (none: no tools)."""

    body = json.loads((DATASETS / name).read_text(encoding="utf-8"))
    return [
        {**item, "tools": body["tools"][item["tools"]]} if "tools" in item else dict(item)
        for item in body["items"]
    ]


TOOL_ROUTE = "verified-tool"


def _route(trace: dict | None) -> tuple[str, float | None]:
    """The profile that served the request and Quyet's P(TOOL), from the trace."""

    events = (trace or {}).get("events") or []
    judge = next((event for event in events if event.get("node") == "profile_judge"), None)
    if judge is None:
        return "unknown", None
    detail = judge.get("detail") or {}
    verdict = detail.get("verdict")
    p_tool = detail.get("p_TOOL")
    if verdict is None:
        return f"deepseek_think (fallback: {detail.get('fallback')})", p_tool
    return (TOOL_ROUTE if verdict == "primary" else verdict), p_tool


def _judge_seconds(trace: dict | None) -> float | None:
    """Wall time of the route judge's read, from the trace."""

    for event in (trace or {}).get("events") or []:
        if event.get("node") == "profile_judge":
            timing = event.get("timing") or {}
            try:
                start = datetime.fromisoformat(timing["started_at"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(timing["completed_at"].replace("Z", "+00:00"))
            except (KeyError, TypeError, ValueError):
                return None
            return round((end - start).total_seconds(), 3)
    return None


def _efforts(trace: dict | None) -> dict[str, str | None]:
    """The reasoning effort of every successful DeepSeek generation, by node
    (a fixed answer reports its last attempt)."""

    return {
        event.get("node"): (event.get("detail") or {}).get("reasoning_effort")
        for event in (trace or {}).get("events") or []
        if event.get("kind") == "generation"
        and event.get("worker") == "deepseek"
        and event.get("status") == "success"
    }


def _judge_worker(trace: dict | None) -> str | None:
    """The Quyet worker that answered the route judge, from the trace."""

    for event in (trace or {}).get("events") or []:
        if event.get("node") == "profile_judge":
            return event.get("worker")
    return None


def _seconds(timing: dict | None, key: str) -> datetime | None:
    try:
        return datetime.fromisoformat((timing or {})[key].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


STAGES = ("candidates", "judgments", "answer", "format_check")


def _stages(trace: dict | None) -> dict[str, dict]:
    """The verified tool route's stages from the trace: status and worker of
    the last run, runs (the answer and its form check run again per fix),
    start of the first and end of the last run (seconds from the first stage),
    and a verification's question count."""

    events = [
        event
        for event in (trace or {}).get("events") or []
        if event.get("node") in STAGES
        and event.get("kind") in {"generation", "verification"}
        and event.get("status") in {"success", "failed"}
    ]
    starts = [
        moment
        for event in events
        if (moment := _seconds(event.get("timing"), "started_at")) is not None
    ]
    origin = min(starts) if starts else None
    stages: dict[str, dict] = {}
    for event in events:
        start = _seconds(event.get("timing"), "started_at")
        end = _seconds(event.get("timing"), "completed_at")
        detail = event.get("detail") or {}
        earlier = stages.get(event["node"]) or {}
        stages[event["node"]] = {
            "status": event["status"],
            "worker": event.get("worker"),
            "runs": earlier.get("runs", 0) + 1,
            "start_s": earlier.get("start_s")
            or (round((start - origin).total_seconds(), 2) if start and origin else None),
            "end_s": round((end - origin).total_seconds(), 2) if end and origin else None,
            **({"items": detail.get("items")} if event["kind"] == "verification" else {}),
            **({"pass": detail.get("pass")} if event["kind"] == "verification" else {}),
            **({"reason": detail.get("reason")} if detail.get("unavailable") else {}),
        }
    return stages


def _tool_route_problems(row: dict, caller_effort: str | None) -> list[str]:
    """What a verified tool row misses of the route (empty when sound): the
    four stages, 10-16 candidates judged 6 times each on quyet-judge, the
    seven-question form check on quyet-judge with its report, and DeepSeek's
    candidates and answer (fixes included) at the caller's effort (default
    high)."""

    stages = row["stages"]
    problems = []
    for node in STAGES:
        if (stages.get(node) or {}).get("status") != "success":
            problems.append(f"{node}={stages.get(node)}")
    if problems:
        return problems
    items = stages["judgments"].get("items") or 0
    if items % 6 or not 60 <= items <= 96:
        problems.append(f"judgment items={items} (6 per candidate, 10-16 candidates)")
    if stages["format_check"].get("items") != 7:
        problems.append(f"form items={stages['format_check'].get('items')}")
    for node in ("judgments", "format_check"):
        if stages[node]["worker"] != "quyet_judge":
            problems.append(f"{node} on {stages[node]['worker']}")
    if row["verification"] is None:
        problems.append("no form-check report")
    if row["efforts"] != _tool_route_efforts(caller_effort):
        problems.append(f"efforts={row['efforts']}")
    return problems


def _tool_route_summary(rows: list[dict]) -> dict:
    """The verified tool route's form checks, fixes, candidates and calls per reply."""

    tool = [row for row in rows if row["route"] == TOOL_ROUTE and row["status"] == 200]
    return {
        "replies": len(tool),
        "form_passed": sum(1 for row in tool if (row["verification"] or {}).get("guaranteed")),
        "publish_reasons": dict(
            collections.Counter(str((row["verification"] or {}).get("reason")) for row in tool)
        ),
        "answer_attempts": dict(
            collections.Counter((row["verification"] or {}).get("attempts") for row in tool)
        ),
        "candidates": dict(
            collections.Counter(
                ((row["stages"].get("judgments") or {}).get("items") or 0) // 6 for row in tool
            )
        ),
        "calls_per_reply": dict(collections.Counter(len(row["tool_calls"]) for row in tool)),
    }


def _tool_call_problems(row: dict, tools: list[dict]) -> list[str]:
    """What a reply misses of a structured call to a declared tool (empty
    when sound): at least one call, each naming a declared tool with a JSON
    object as its arguments."""

    declared = {tool["function"]["name"] for tool in tools}
    calls = row["tool_calls"]
    if not calls:
        return [f"no tool call; content={row['content'][:160]!r}"]
    problems = []
    for call in calls:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "")
        except ValueError:
            arguments = None
        if function.get("name") not in declared or not isinstance(arguments, dict):
            problems.append(
                f"call {function.get('name')!r} {str(function.get('arguments'))[:120]!r}"
            )
    return problems


def _tool_route_efforts(caller_effort: str | None) -> dict[str, str]:
    """The verified tool route's efforts: the caller's (default high) for the
    candidates and the answer (and so for its fixes)."""

    wanted = caller_effort or "high"
    return {"candidates": wanted, "answer": wanted}


def _post_stream(
    url: str, payload: dict, headers: dict, timeout_s: float
) -> tuple[dict, float | None]:
    """Stream one chat request; returns an assembled body and the time to first token."""

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
    )
    started = time.monotonic()
    first: float | None = None
    content, reasoning = [], []
    calls: dict[int, dict] = {}
    finish = None
    body: dict = {"choices": [{"message": {}}]}
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: {"):
                continue
            chunk = json.loads(line[len("data: ") :])
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if (delta.get("content") or delta.get("reasoning_content")) and first is None:
                    first = time.monotonic() - started
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
                for call in delta.get("tool_calls") or []:
                    if first is None:
                        first = time.monotonic() - started
                    slot = calls.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                    function = call.get("function") or {}
                    slot["name"] += function.get("name") or ""
                    slot["arguments"] += function.get("arguments") or ""
                finish = choice.get("finish_reason") or finish
            for key in ("usage", "kairyu_verification", "kairyu_trace_v2", "kairyu_route"):
                if chunk.get(key) is not None:
                    body[key] = chunk[key]
    body["choices"][0]["message"] = {
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "tool_calls": [{"type": "function", "function": calls[index]} for index in sorted(calls)],
    }
    body["choices"][0]["finish_reason"] = finish
    return body, first


def chat(
    env: dict[str, str],
    content: str | None = None,
    *,
    messages: list[dict] | None = None,
    tools: list[dict] | None = None,
    model: str = MODEL,
    trace: bool = False,
    stream: bool = False,
    timeout_s: float = 7200,
    **extra,
) -> dict:
    """One chat request; returns the per-request evidence row."""

    payload = control.routed_request(content or "", model=model, **extra)
    if messages is not None:
        payload["messages"] = messages
    if tools:
        payload["tools"] = tools
    headers = {"X-Kairyu-Trace": "1"} if trace else {}
    url = f"{_api(env)}/v1/chat/completions"
    started = time.monotonic()
    ttft = None
    try:
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            body, ttft = _post_stream(url, payload, headers, timeout_s)
        else:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json", **headers},
            )
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                body = json.loads(response.read())
        status = 200
    except urllib.error.HTTPError as error:
        body = {"error": error.read().decode("utf-8", "replace")[:500]}
        status = error.code
    except (OSError, urllib.error.URLError) as error:
        body = {"error": repr(error)}
        status = 0
    elapsed = time.monotonic() - started
    usage = body.get("usage") or {}
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output_tokens = usage.get("orchestration_output_tokens") or usage.get("completion_tokens") or 0
    trace_body = body.get("kairyu_trace_v2")
    route, p_tool = _route(trace_body)
    report = body.get("kairyu_verification")
    verification = (
        None
        if report is None
        else {
            "guaranteed": report.get("guaranteed"),
            "reason": report.get("reason"),
            "attempts": report.get("attempts"),
            "failing": [
                item.get("id")
                for item in report.get("requirements") or []
                if not item.get("passed")
            ],
        }
    )
    return {
        "status": status,
        "seconds": round(elapsed, 2),
        "ttft_s": round(ttft, 2) if ttft is not None else None,
        "model": model,
        "route": route,
        "p_tool": p_tool,
        "efforts": _efforts(trace_body),
        "judge_worker": _judge_worker(trace_body),
        "stages": _stages(trace_body),
        "judge_s": _judge_seconds(trace_body),
        "public_completion_tokens": usage.get("completion_tokens"),
        "orchestration_input_tokens": usage.get("orchestration_input_tokens")
        or usage.get("prompt_tokens"),
        "orchestration_output_tokens": output_tokens,
        "orchestration_output_tok_per_s": round(output_tokens / elapsed, 1) if elapsed else None,
        "has_verification": report is not None,
        "verification": verification,
        "content": message.get("content") or "",
        "tool_calls": message.get("tool_calls") or [],
        "finish_reason": choice.get("finish_reason"),
        "reasoning_chars": len(message.get("reasoning_content") or ""),
        "error": body.get("error"),
    }


def _summary(rows: list[dict]) -> dict:
    ok = [row for row in rows if row["status"] == 200]
    seconds = sorted(row["seconds"] for row in ok)
    out_tokens = sum(row["orchestration_output_tokens"] or 0 for row in ok)
    in_tokens = sum(row["orchestration_input_tokens"] or 0 for row in ok)
    return {
        "requests": len(rows),
        "ok": len(ok),
        "latency_p50_s": statistics.median(seconds) if seconds else None,
        "latency_p95_s": seconds[max(0, int(len(seconds) * 0.95) - 1)] if seconds else None,
        "orchestration_input_tokens": in_tokens,
        "orchestration_output_tokens": out_tokens,
    }


def _print_rows(rows: list[dict]) -> None:
    for index, row in enumerate(rows):
        print(
            f"  #{index:02d} status={row['status']} {row['seconds']:7.1f}s "
            f"ttft={row.get('ttft_s')} route={row['route']} p_tool={row.get('p_tool')} "
            f"efforts={row['efforts']} judge={row.get('judge_worker')} "
            f"calls={[(call.get('function') or {}).get('name') for call in row['tool_calls']]} "
            f"judge_s={row.get('judge_s')} form={row.get('verification')} "
            f"in={row['orchestration_input_tokens']} "
            f"out={row['orchestration_output_tokens']} "
            f"({row['orchestration_output_tok_per_s']} tok/s)",
            flush=True,
        )
        if row.get("stages"):
            print(
                "      "
                + " ".join(
                    f"{node}={stage['status']}@{stage['start_s']}-{stage['end_s']}s"
                    + (f"/runs={stage['runs']}" if stage.get("runs", 1) > 1 else "")
                    + (f"/items={stage['items']}" if "items" in stage else "")
                    for node, stage in row["stages"].items()
                ),
                flush=True,
            )


def _run_concurrently(env: dict[str, str], prompts: list[dict], concurrency: int) -> list[dict]:
    def one(prompt: dict) -> dict:
        extra = {key: value for key, value in prompt.items() if key != "content"}
        return chat(env, prompt.get("content"), **extra)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, prompts))


class Deadline:
    def __init__(self, gate: str, budget_s: float) -> None:
        self.gate = gate
        self.end = time.monotonic() + budget_s
        self.budget_s = budget_s

    def check(self) -> None:
        if time.monotonic() > self.end:
            raise SystemExit(f"{self.gate}: time budget of {self.budget_s:.0f} s exceeded")


def gate_l1(env: dict[str, str]) -> None:
    started = time.monotonic()
    control.validate_serving(env)
    print(f"l1: PASS in {time.monotonic() - started:.0f} s", flush=True)
    _write("l1", {"passed": True, "seconds": time.monotonic() - started})


def _routing_probabilities(env: dict[str, str], items: list[dict]) -> list[dict[str, float] | None]:
    """Every route's probability for each routing-set conversation, through
    the served judge.

    The orchestrator is built from verified-tool.yaml with the real System One
    backend pointed at quyet-route, so the request Quyet reads (the
    conversation and whether tools are declared) is the one Kairyu sends in
    serving; no generation or judgment runs.
    """

    import asyncio

    from kairyu.dsl.loader import build_orchestrator, load_spec
    from kairyu.engine.mock import MockBackend
    from kairyu.engine.systemone import HTTPSystemOneBackend
    from kairyu.entrypoints.server.chat_service import validate_orchestration_chat_input
    from kairyu.entrypoints.server.protocol import ChatCompletionRequest
    from kairyu.orchestration.request import OrchestrationRequest
    from kairyu.sampling_params import SamplingParams

    async def judge_all() -> list[dict[str, float] | None]:
        backend = HTTPSystemOneBackend(
            base_urls=(control.quyet_l1_url(env, "quyet-route"),),
            upstream_model=SPEC["systemone"]["upstream_model"],
            max_concurrency=8,
            max_queue=1024,
            queue_wait_s=600,
        )
        orchestrator = build_orchestrator(
            load_spec(HERE / "verified-tool.yaml"),
            engine_refs={SPEC["deepseek"]["served_name"]: MockBackend()},
            systemone_refs={model: backend for model in SPEC["systemone"]["models"]},
        )

        async def one(item: dict) -> dict[str, float] | None:
            chat_request = ChatCompletionRequest(
                model=MODEL, messages=item["messages"], tools=item.get("tools")
            )
            call = OrchestrationRequest(
                prompt=validate_orchestration_chat_input(chat_request).prompt,
                sampling_params=SamplingParams(max_tokens=1024),
                tools=tuple(item.get("tools") or ()),
            )
            judged = await orchestrator.judge_role_profile(call)
            metadata = judged.role_profile_judge_event.metadata
            probabilities = {
                key.removeprefix("p_"): float(value)
                for key, value in metadata.items()
                if key.startswith("p_") and isinstance(value, (int, float))
            }
            return probabilities if "TOOL" in probabilities else None

        try:
            return await asyncio.gather(*(one(item) for item in items))
        finally:
            await backend.shutdown()

    return asyncio.run(judge_all())


def _served_route(probabilities: dict[str, float], tau: float | None) -> str:
    # The served rule: TOOL when preferred (p >= tau, if a floor is
    # configured), else the most probable route.
    if tau is not None and probabilities["TOOL"] >= tau:
        return "TOOL"
    return max(probabilities, key=probabilities.__getitem__)


def gate_routing(env: dict[str, str], *, budget_s: float = 1800) -> None:
    """Quyet routes turns whose next reply needs a tool call to TOOL.

    The configured rule (prefer.min_probability, or the most probable route
    when none is set) must miss fewer than 10 % of the TOOL turns on both the
    calibration half (even items) and the held-out half (odd items), and at
    least 90 % of the conversations it sends to THINK must need no tool call
    (THINK precision). The largest floor meeting the miss rate on the
    calibration half is reported as recommended_tau.
    """

    deadline = Deadline("routing", budget_s)
    items = _dataset("tool-routing-set.json")
    started = time.monotonic()
    probabilities = _routing_probabilities(env, items)
    judge_seconds = time.monotonic() - started
    if any(p is None for p in probabilities):
        raise SystemExit("routing: the judge did not answer every conversation")
    rows = [
        {
            "label": item["label"],
            "category": item["category"],
            "tools": bool(item.get("tools")),
            "messages": len(item["messages"]),
            "p": p,
            "p_tool": p["TOOL"],
        }
        for item, p in zip(items, probabilities, strict=True)
    ]
    import yaml

    prefer = yaml.safe_load((HERE / "verified-tool.yaml").read_text())["profile_judge"].get(
        "prefer"
    )
    configured = prefer["min_probability"] if prefer else None

    def miss_rate(subset: list[dict], tau: float | None) -> float:
        needed = [row for row in subset if row["label"] == "TOOL"]
        missed = [row for row in needed if _served_route(row["p"], tau) != "TOOL"]
        return len(missed) / len(needed)

    def think_precision(subset: list[dict], tau: float | None) -> float:
        sent = [row for row in subset if _served_route(row["p"], tau) == "THINK"]
        return sum(1 for row in sent if row["label"] == "THINK") / len(sent) if sent else 1.0

    def think_recall(subset: list[dict], tau: float | None) -> float:
        easy = [row for row in subset if row["label"] == "THINK"]
        return sum(1 for row in easy if _served_route(row["p"], tau) == "THINK") / len(easy)

    calibration = rows[0::2]
    holdout = rows[1::2]
    candidates = sorted({round(row["p_tool"], 4) for row in calibration} | {0.0, 0.5}, reverse=True)
    recommended = next(
        (tau for tau in candidates if tau <= 0.5 and miss_rate(calibration, tau) < 0.10), 0.0
    )
    by_category: dict[str, list[float]] = {}
    for row in rows:
        by_category.setdefault(f"{row['label']}/{row['category']}", []).append(row["p_tool"])
    summary = {
        "conversations": len(rows),
        "judge_wall_s": round(judge_seconds, 2),
        "configured_tau": configured,
        "recommended_tau": recommended,
        "calibration_miss_rate": round(miss_rate(calibration, configured), 4),
        "holdout_miss_rate": round(miss_rate(holdout, configured), 4),
        "think_precision": round(think_precision(rows, configured), 4),
        "think_recall": round(think_recall(rows, configured), 4),
        "think_with_tools_recall": round(
            think_recall([row for row in rows if row["tools"]], configured), 4
        ),
        "p_tool_by_category": {
            category: {
                "n": len(values),
                "min": round(min(values), 4),
                "median": round(statistics.median(values), 4),
                "max": round(max(values), 4),
            }
            for category, values in by_category.items()
        },
    }
    deadline.check()
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    passed = (
        summary["holdout_miss_rate"] < 0.10
        and summary["calibration_miss_rate"] < 0.10
        and summary["think_precision"] >= 0.90
    )
    _write("routing", {"passed": passed, "summary": summary, "rows": rows})
    print(
        f"routing: {'PASS' if passed else 'FAIL'} "
        f"(TOOL miss rate held-out {summary['holdout_miss_rate']}, "
        f"calibration {summary['calibration_miss_rate']}; "
        f"THINK precision {summary['think_precision']})"
    )
    if not passed:
        raise SystemExit(1)


def _everyday(items: list[dict]) -> list[dict]:
    """Everyday chats without tools: the think route's requests."""

    return [item for item in items if item["category"] == "everyday" and not item.get("tools")]


def gate_think_route(env: dict[str, str], *, budget_s: float = 1800) -> None:
    """Everyday requests stream from the think route at the default effort.

    The publisher's private reasoning is withheld by Kairyu's multi-stage
    contract (as on the sibling ensemble's deepseek_think route), so the gate
    checks the route, a streamed answer with its time to first token, no
    verification field, and that DeepSeek ran at the default effort (high).
    """

    deadline = Deadline("think-route", budget_s)
    items = _everyday(_dataset("tool-routing-set.json"))[:6]
    rows = [chat(env, messages=item["messages"], trace=True, stream=True) for item in items]
    deadline.check()
    findings = []
    for row in rows:
        if row["status"] != 200 or not row["content"].strip() or row["ttft_s"] is None:
            findings.append(f"status={row['status']} empty={not row['content'].strip()}")
        elif row["route"] != "deepseek_think":
            findings.append(f"everyday request routed to {row['route']}")
        elif row["has_verification"] or row["efforts"] != {"deepseek_think_answer": "high"}:
            findings.append(
                f"think route: verification={row['has_verification']} efforts={row['efforts']}"
            )
    _print_rows(rows)
    ttfts = [row["ttft_s"] for row in rows if row["ttft_s"] is not None]
    summary = {
        **_summary(rows),
        "think_routed": sum(1 for row in rows if row["route"] == "deepseek_think"),
        "ttft_p50_s": statistics.median(ttfts) if ttfts else None,
    }
    print(json.dumps(summary, indent=2), flush=True)
    _write(
        "think-route",
        {"passed": not findings, "findings": findings, "summary": summary, "rows": rows},
    )
    print(f"think-route: {'PASS' if not findings else 'FAIL'} {findings}")
    if findings:
        raise SystemExit(1)


def gate_effort(env: dict[str, str], *, budget_s: float = 5400) -> None:
    """The caller's effort (default high) reaches the think route and the
    verified tool route's candidates and answer (fixes included); quyet-route
    judges the route."""

    deadline = Deadline("effort", budget_s)
    easy = _everyday(_dataset("tool-routing-set.json"))[0]
    turn = _dataset("tool-turns.json")[0]
    findings = []
    report = {}
    for effort in (None, "low", "high", "max"):
        extra = {} if effort is None else {"reasoning_effort": effort}
        think = chat(env, messages=easy["messages"], trace=True, **extra)
        tool = chat(env, messages=turn["messages"], tools=turn["tools"], trace=True, **extra)
        expected = effort or "high"
        for row, route, wanted in (
            (think, "deepseek_think", {"deepseek_think_answer": expected}),
            (tool, TOOL_ROUTE, _tool_route_efforts(effort)),
        ):
            if row["status"] != 200 or row["route"] != route or row["efforts"] != wanted:
                findings.append(
                    f"{effort}: {row['route']} status={row['status']} efforts={row['efforts']} "
                    f"(expected {route} {wanted})"
                )
            if row["judge_worker"] != "quyet_route":
                findings.append(f"{effort}: route judged on {row['judge_worker']}")
        print(f"effort={expected}:")
        _print_rows([think, tool])
        report[expected if effort else "default"] = {"rows": [think, tool]}
        deadline.check()
    _write("effort", {"passed": not findings, "findings": findings, "report": report})
    print(f"effort: {'PASS' if not findings else 'FAIL'} {findings}")
    if findings:
        raise SystemExit(1)


def gate_verified_tool_route(env: dict[str, str], *, budget_s: float = 7200) -> None:
    """Agent turns take the verified tool route into one structured call.

    Six turns of datasets/tool-turns.json, unary and streamed, with the
    caller's effort cycling through none, low, high and max: each must be
    routed TOOL and return 200 after DeepSeek listed 10-16 candidates, Quyet
    (quyet-judge) judged each six times, the answer was written and its form
    checked on quyet-judge (fixes and the publish reason are recorded), at the
    caller's effort; the reply carries a structured call to a declared tool
    with JSON-object arguments.
    """

    deadline = Deadline("verified-tool-route", budget_s)
    items = _dataset("tool-turns.json")[:6]
    efforts = [None, "low", "high", "max"]
    findings = []
    rows = []
    for index, item in enumerate(items):
        for stream in (False, True):
            effort = efforts[(2 * index + stream) % len(efforts)]
            extra = {"reasoning_effort": effort} if effort else {}
            row = chat(
                env,
                messages=item["messages"],
                tools=item["tools"],
                trace=True,
                stream=stream,
                **extra,
            )
            row.update(id=item["id"], stream=stream, caller_effort=effort)
            rows.append(row)
            problems = ([f"route={row['route']}"] if row["route"] != TOOL_ROUTE else []) + (
                _tool_route_problems(row, effort) + _tool_call_problems(row, item["tools"])
            )
            if row["status"] != 200 or problems:
                findings.append(
                    f"{item['id']} stream={stream} effort={effort}: status={row['status']} "
                    f"{problems}"
                )
        deadline.check()
    _print_rows(rows)
    summary = {**_summary(rows), "tool_route": _tool_route_summary(rows)}
    print(json.dumps(summary, indent=2), flush=True)
    _write(
        "verified-tool-route",
        {"passed": not findings, "findings": findings, "summary": summary, "rows": rows},
    )
    print(f"verified-tool-route: {'PASS' if not findings else 'FAIL'} {findings}")
    if findings:
        raise SystemExit(1)


def _compose(env: dict[str, str], *arguments: str) -> None:
    control._compose(list(arguments), env=env)


def _wait_healthy(service: str, timeout_s: float = 1800) -> None:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        state = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Health.Status}}",
                f"{control.PROJECT}-{service}-1",
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if state == "healthy":
            return
        time.sleep(5)
    raise SystemExit(f"{service} did not become healthy again")


def _replica_services(service: str) -> tuple[str, str]:
    """A Quyet replica's containers: its vLLM, then its adapter."""

    replica = next(item for item in control.QUYET_REPLICAS if item["service"] == service)
    return replica["vllm_service"], service


def gate_fallback(env: dict[str, str], *, budget_s: float = 3600) -> None:
    """Each Quyet replica (its vLLM and adapter) fails alone: quyet-route down
    sends an agent turn to the think route; quyet-judge down leaves routing
    intact, and the verified tool route answers without judgments and is
    published unverified, still with a structured call; both back: judged
    and form-checked again."""

    deadline = Deadline("fallback", budget_s)
    turn = _dataset("tool-turns.json")[0]

    def ask() -> dict:
        return chat(env, messages=turn["messages"], tools=turn["tools"], trace=True)

    def restart(service: str) -> None:
        for name in _replica_services(service):
            _compose(env, "start", name)
            _wait_healthy(name)

    findings = []
    phases = {}
    try:
        _compose(env, "stop", *_replica_services("quyet-route"))
        down = [ask() for _ in range(2)]
        phases["route_down"] = down
        for row in down:
            if (
                row["status"] != 200
                or not (row["content"].strip() or row["tool_calls"])
                or not row["route"].startswith("deepseek_think (fallback")
            ):
                findings.append(f"route down: status={row['status']} route={row['route']}")
    finally:
        restart("quyet-route")
    deadline.check()
    try:
        _compose(env, "stop", *_replica_services("quyet-judge"))
        unjudged = ask()
        phases["judge_down"] = [unjudged]
        judgments = unjudged["stages"].get("judgments") or {}
        verification = unjudged["verification"] or {}
        if (
            unjudged["status"] != 200
            or unjudged["route"] != TOOL_ROUTE
            or (unjudged["stages"].get("answer") or {}).get("status") != "success"
            or "reason" not in judgments
            or verification.get("guaranteed") is not False
            or _tool_call_problems(unjudged, turn["tools"])
        ):
            findings.append(
                f"judge down: status={unjudged['status']} route={unjudged['route']} "
                f"judgments={judgments} form={verification} "
                f"{_tool_call_problems(unjudged, turn['tools'])}"
            )
    finally:
        restart("quyet-judge")
    recovered = ask()
    phases["recovered"] = [recovered]
    problems = _tool_route_problems(recovered, None) + _tool_call_problems(recovered, turn["tools"])
    if recovered["status"] != 200 or recovered["route"] != TOOL_ROUTE or problems:
        findings.append(
            f"after restart: status={recovered['status']} route={recovered['route']} {problems}"
        )
    deadline.check()
    for name, rows in phases.items():
        print(f"{name}:")
        _print_rows(rows)
    _write("fallback", {"passed": not findings, "findings": findings, "phases": phases})
    print(f"fallback: {'PASS' if not findings else 'FAIL'} {findings}")
    if findings:
        raise SystemExit(1)


def _load(
    env: dict[str, str],
    gate: str,
    batches: dict[int, list[dict]],
    *,
    per_route: bool,
    require_calls: bool = False,
) -> dict:
    """Each level's requests at once; every request must be answered, and with
    ``require_calls`` every reply must carry a structured call to a declared tool."""

    report = {}
    # Each verified tool request carries the candidates, the answer and its fixes (6 h).
    deadline = Deadline(gate, 21600)
    for concurrency, prompts in batches.items():
        started = time.monotonic()
        rows = _run_concurrently(env, prompts, concurrency)
        wall = time.monotonic() - started
        summary = {
            **_summary(rows),
            "wall_s": round(wall, 1),
            "requests_per_min": round(60 * len(rows) / wall, 2),
            "tool_route": _tool_route_summary(rows),
        }
        if require_calls:
            summary["without_call"] = [
                index
                for index, (row, prompt) in enumerate(zip(rows, prompts, strict=True))
                if row["status"] == 200 and _tool_call_problems(row, prompt["tools"])
            ]
        summary["orchestration_output_tok_per_s"] = round(
            summary["orchestration_output_tokens"] / wall, 1
        )
        if per_route:
            summary["per_route"] = {}
            for route in sorted({row["route"] for row in rows}):
                subset = [row for row in rows if row["route"] == route]
                judge = sorted(row["judge_s"] for row in subset if row.get("judge_s") is not None)
                summary["per_route"][route] = {
                    **_summary(subset),
                    "judge_p50_s": statistics.median(judge) if judge else None,
                }
        print(f"c{concurrency}:")
        _print_rows(rows)
        print(json.dumps(summary, indent=2), flush=True)
        report[f"c{concurrency}"] = {"summary": summary, "rows": rows}
        deadline.check()
    failures = [
        name
        for name, entry in report.items()
        if entry["summary"]["ok"] != entry["summary"]["requests"]
        or entry["summary"].get("without_call")
    ]
    _write(gate, {"passed": not failures, "failed": failures, "report": report})
    print(
        f"{gate}: {'PASS' if not failures else 'FAIL'} (every request answered"
        + (", every reply a structured call)" if require_calls else ")")
    )
    if failures:
        raise SystemExit(1)
    return report


def gate_serving(env: dict[str, str]) -> None:
    """Agent turns (datasets/tool-turns.json) under load: every reply a
    structured call; latency, tokens and tok/s per route."""

    items = _dataset("tool-turns.json")
    batches, offset = {}, 0
    for concurrency, count in PLAN.items():
        batches[concurrency] = [
            {
                "messages": items[(offset + i) % len(items)]["messages"],
                "tools": items[(offset + i) % len(items)]["tools"],
                "trace": True,
            }
            for i in range(count)
        ]
        offset += count
    _load(env, "serving", batches, per_route=True, require_calls=True)


def gate_serving_routed(env: dict[str, str]) -> None:
    """The routing set under load: route mix, latency and tokens per route."""

    items = _dataset("tool-routing-set.json")
    random.Random(3).shuffle(items)
    batches, offset = {}, 0
    for concurrency, count in PLAN.items():
        batches[concurrency] = [
            {
                "messages": items[(offset + i) % len(items)]["messages"],
                "tools": items[(offset + i) % len(items)].get("tools"),
                "trace": True,
            }
            for i in range(count)
        ]
        offset += count
    _load(env, "serving-routed", batches, per_route=True)


def gate_browser(_env: dict[str, str]) -> None:
    subprocess.run([str(HERE / "browser-smoke.sh")], check=True)
    _write("browser", {"passed": True})


GATES = {
    "l1": gate_l1,
    "routing": gate_routing,
    "think-route": gate_think_route,
    "effort": gate_effort,
    "verified-tool-route": gate_verified_tool_route,
    "fallback": gate_fallback,
    "serving": gate_serving,
    "serving-routed": gate_serving_routed,
    "browser": gate_browser,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", choices=(*GATES, "list"))
    args = parser.parse_args()
    if args.gate == "list":
        print("\n".join(GATES))
        return
    global GATE_STARTED
    GATE_STARTED = datetime.now(UTC).isoformat()
    GATES[args.gate](_env())


if __name__ == "__main__":
    main()
