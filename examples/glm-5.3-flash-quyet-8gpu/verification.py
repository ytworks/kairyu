#!/usr/bin/env python3
"""GPU gates for GLM-5.3-Flash answers decided by Quyet (VCO-D27).

The gates guarantee behaviour, not the accuracy of Quyet's decision: whether a
fix cuts API calls is measured by the benchmark. Every gate has a time budget
and writes its per-request evidence (latency, tokens, tok/s, efforts, the
decision, fixes) to model-volumes/<environment>/results/<gate>-<UTC>.json, with
the Quyet reads of the gate (count, states cut at Quyet's 6,000 tokens, strict
refusals, read times) from the adapters' logs. One public model,
kairyu-verified-tool: GLM answers as it is, Quyet (quyet-judge, strict)
reads five questions once, and a failing answer is written once more.

The decision rule every gate checks on every traced request: the answer runs,
then the five-question decision on quyet-judge; when the first decision passes
the answer is published with no fix (one attempt); when it fails the answer
runs exactly once more (two attempts) and its fix is decided again; a decision
Quyet refused to read whole publishes the answer unverified.

  l1             every GLM DP rank (grammar JSON at the default and low effort), each
                 Quyet replica's vLLM and System One, the decision's strict name
                 refusing a state Quyet would cut, and one structured tool call
                 from the public model
  decision       12 agent turns (datasets/tool-turns.json), unary and streamed:
                 the decision rule, a structured call to a declared tool, the
                 caller's effort, every request under 900 s
  text           text conversations (datasets/text-set.json) stream an answer,
                 under the decision rule, at the default effort
  effort         the caller's effort reaches the answer and its fix
  fallback       quyet-judge down: the answer is published unverified, still a
                 structured call; back: decided again
  serving        agent turns at c1/c4/c8/c16: every reply a structured call under
                 900 s and the decision rule; fixes and stage times recorded
  serving-mixed  agent turns and text conversations mixed at c1/c4/c8/c16:
                 every request answered under the decision rule
  browser        Open WebUI answers in a real browser
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
_REFUSED_LINE = re.compile(r"refused questions=(\d+) strict=(True|False): (.*)")


def _quyet_reads(since: str | None) -> dict[str, dict]:
    """Per Quyet adapter: reads, questions, states cut at 6,000 tokens, strict
    reads refused (a state too long for the decision) and read times since
    ``since``, from the adapter's ``read`` and ``refused`` log lines."""

    report = {}
    for service in control.QUYET_SERVICES:
        command = ["docker", "logs", f"{control.PROJECT}-{service}-1"]
        if since:
            command[2:2] = ["--since", since]
        logs = subprocess.run(command, capture_output=True, text=True, check=False)
        text = logs.stdout + logs.stderr
        rows = [match.groups() for match in _READ_LINE.finditer(text)]
        refused = [match.groups() for match in _REFUSED_LINE.finditer(text)]
        model_ms = sorted(int(row[4]) for row in rows)
        cut = [int(row[3]) for row in rows if row[2] == "True"]
        report[service] = {
            "reads": len(rows),
            "questions": sum(int(row[0]) for row in rows),
            "truncated": len(cut),
            "truncated_state_tokens_max": max(cut) if cut else None,
            "refused_strict": sum(1 for row in refused if row[1] == "True"),
            "refused_other": sum(1 for row in refused if row[1] == "False"),
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


def _efforts(trace: dict | None) -> dict[str, str | None]:
    """The reasoning effort of every successful GLM generation, by node
    (a fixed answer reports its last attempt)."""

    return {
        event.get("node"): (event.get("detail") or {}).get("reasoning_effort")
        for event in (trace or {}).get("events") or []
        if event.get("kind") == "generation"
        and event.get("worker") == "glm"
        and event.get("status") == "success"
    }


def _seconds(timing: dict | None, key: str) -> datetime | None:
    try:
        return datetime.fromisoformat((timing or {})[key].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


STAGES = ("answer", "decision")
# The decision's questions (verified-tool.yaml): E1-E5.
DECISION_ITEMS = 5
# verified-tool.yaml's default_reasoning_effort: GLM's official default.
DEFAULT_EFFORT = "max"


def _stages(trace: dict | None) -> dict[str, dict]:
    """The two stages from the trace: status and worker of the last run, runs
    (the answer and its decision run again on a fix), start of the first and
    end of the last run (seconds from the first stage), and for the decision
    every read's result in order (True pass, False fail, None unavailable),
    its question count and an unavailable read's reason."""

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
        stage = {
            "status": event["status"],
            "worker": event.get("worker"),
            "runs": earlier.get("runs", 0) + 1,
            "start_s": earlier.get("start_s")
            or (round((start - origin).total_seconds(), 2) if start and origin else None),
            "end_s": round((end - origin).total_seconds(), 2) if end and origin else None,
        }
        if event["kind"] == "verification":
            unavailable = bool(detail.get("unavailable"))
            stage["passes"] = [
                *earlier.get("passes", []),
                None if unavailable else detail.get("pass"),
            ]
            stage["items"] = detail.get("items", earlier.get("items"))
            if unavailable:
                stage["reason"] = detail.get("reason")
        stages[event["node"]] = stage
    return stages


def _rule_problems(row: dict, caller_effort: str | None) -> list[str]:
    """What a traced row misses of the decision rule (empty when sound): the
    answer, then the five-question decision read strict on quyet-judge
    (quyet_form); a first decision that passes publishes the answer with no fix
    (one attempt), one that fails runs the answer exactly once more (two
    attempts) and decides the fix again; a first decision Quyet could not read
    publishes the answer unverified with no fix. GLM answers (and fixes)
    at the caller's effort (default max)."""

    stages = row["stages"]
    answer = stages.get("answer") or {}
    decision = stages.get("decision") or {}
    passes = decision.get("passes") or []
    report = row["verification"] or {}
    problems = []
    if row["status"] != 200:
        return [f"status={row['status']} {row.get('error')}"]
    if answer.get("status") != "success":
        problems.append(f"answer={answer}")
    if not passes:
        return [*problems, f"no decision: {decision}"]
    if decision.get("worker") != "quyet_form":
        problems.append(f"decision on {decision.get('worker')}")
    if any(result is not None for result in passes) and decision.get("items") != DECISION_ITEMS:
        problems.append(f"decision items={decision.get('items')}")
    if row["verification"] is None:
        problems.append("no decision report")
    first = passes[0]
    if first is False:
        expected = {"answer_runs": 2, "decisions": 2, "attempts": 2}
    else:
        # A pass, or an unavailable read: the answer as it is.
        expected = {"answer_runs": 1, "decisions": 1, "attempts": 1}
    seen = {
        "answer_runs": answer.get("runs"),
        "decisions": len(passes),
        "attempts": report.get("attempts"),
    }
    if seen != expected:
        problems.append(f"first decision {first}: {seen} (expected {expected})")
    if first is None and report.get("guaranteed") is not False:
        problems.append(f"unread answer reported {report}")
    if row["efforts"] != {"answer": caller_effort or DEFAULT_EFFORT}:
        problems.append(f"efforts={row['efforts']}")
    return problems


def _decision_summary(rows: list[dict]) -> dict:
    """Decisions and fixes over answered rows: first decisions passed, failed
    and unread, the questions that sent answers to a fix, fixes the second
    read passed, and calls per reply. Observations only: the gates do not
    judge the decision's accuracy."""

    ok = [row for row in rows if row["status"] == 200]
    firsts = [((row["stages"].get("decision") or {}).get("passes") or [None])[0] for row in ok]
    fixed = [row for row, first in zip(ok, firsts, strict=True) if first is False]
    return {
        "answers": len(ok),
        "first_passed": sum(1 for first in firsts if first is True),
        "fixed": len(fixed),
        "unread": sum(1 for first in firsts if first is None),
        "fix_share": round(len(fixed) / len(ok), 4) if ok else None,
        "fix_passed_second_read": sum(
            1 for row in fixed if (row["verification"] or {}).get("guaranteed")
        ),
        "failing_after_last_read": dict(
            collections.Counter(
                item for row in ok for item in (row["verification"] or {}).get("failing") or []
            )
        ),
        "calls_per_reply": dict(collections.Counter(len(row["tool_calls"]) for row in ok)),
        "stage_seconds": {
            node: _spread(
                [
                    stage["end_s"] - (stage["start_s"] or 0)
                    for row in ok
                    if (stage := row["stages"].get(node)) and stage.get("end_s") is not None
                ]
            )
            for node in STAGES
        },
    }


def _spread(values: list[float]) -> dict:
    values = sorted(values)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "p50": round(statistics.median(values), 2),
        "p95": round(values[max(0, int(len(values) * 0.95) - 1)], 2),
        "max": round(values[-1], 2),
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
        "efforts": _efforts(trace_body),
        "stages": _stages(trace_body),
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
        "latency_max_s": seconds[-1] if seconds else None,
        "orchestration_input_tokens": in_tokens,
        "orchestration_output_tokens": out_tokens,
    }


def _print_rows(rows: list[dict]) -> None:
    for index, row in enumerate(rows):
        print(
            f"  #{index:02d} status={row['status']} {row['seconds']:7.1f}s "
            f"ttft={row.get('ttft_s')} efforts={row['efforts']} "
            f"calls={[(call.get('function') or {}).get('name') for call in row['tool_calls']]} "
            f"decision={row.get('verification')} "
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
                    + (f"/passes={stage['passes']}" if "passes" in stage else "")
                    for node, stage in row["stages"].items()
                ),
                flush=True,
            )


def _run_concurrently(env: dict[str, str], prompts: list[dict], concurrency: int) -> list[dict]:
    def one(prompt: dict) -> dict:
        # "kind" labels the prompt for the gate; it is not sent.
        extra = {key: value for key, value in prompt.items() if key not in {"content", "kind"}}
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


def _check(
    name: str, rows: list[dict], findings: list[str], *, efforts: list[str | None], tools: bool
) -> None:
    """Add each row's departures from the decision rule (and, for agent turns,
    from a structured call to a declared tool under 900 s) to ``findings``."""

    for row, effort in zip(rows, efforts, strict=True):
        problems = _rule_problems(row, effort)
        if tools:
            problems += _tool_call_problems(row, row["declared_tools"])
            if row["seconds"] >= 900:
                problems.append(f"{row['seconds']} s (900 s or more)")
        elif row["status"] == 200 and not (row["content"].strip() or row["tool_calls"]):
            problems.append("empty answer")
        if problems:
            findings.append(f"{name} {row.get('id', '')} stream={row.get('stream')}: {problems}")


def _unread_findings(rows: list[dict]) -> list[str]:
    """A decision Quyet did not read must be one of the adapter's strict
    refusals (a state over 6,000 tokens), never a failure of the replica."""

    unread = sum(
        1 for row in rows if None in ((row["stages"].get("decision") or {}).get("passes") or [])
    )
    refusals = _quyet_reads(GATE_STARTED).get("quyet-judge", {}).get("refused_strict", 0)
    return [f"{unread} decisions unread, {refusals} strict refusals"] if unread > refusals else []


def gate_decision(env: dict[str, str], *, budget_s: float = 3600) -> None:
    """Agent turns follow the decision rule into one structured call.

    Twelve turns of datasets/tool-turns.json, unary and streamed, with the
    caller's effort cycling through none, low, high and max: each returns 200
    under the decision rule (a passing first decision publishes the answer with
    no fix, a failing one runs exactly one fix that is decided again), at the
    caller's effort; the reply carries a structured call to a declared tool
    with JSON-object arguments, and no request takes 900 s (the cut the DeepSWE
    agent's proxy makes). Fixes and the questions behind them are recorded,
    not judged.
    """

    deadline = Deadline("decision", budget_s)
    items = _dataset("tool-turns.json")[:12]
    cycle = [None, "low", "high", "max"]
    rows, efforts = [], []
    for index, item in enumerate(items):
        for stream in (False, True):
            effort = cycle[(2 * index + stream) % len(cycle)]
            extra = {"reasoning_effort": effort} if effort else {}
            row = chat(
                env,
                messages=item["messages"],
                tools=item["tools"],
                trace=True,
                stream=stream,
                **extra,
            )
            row.update(
                id=item["id"], stream=stream, caller_effort=effort, declared_tools=item["tools"]
            )
            rows.append(row)
            efforts.append(effort)
        deadline.check()
    _print_rows(rows)
    findings: list[str] = []
    _check("decision", rows, findings, efforts=efforts, tools=True)
    findings += _unread_findings(rows)
    summary = {**_summary(rows), "decision": _decision_summary(rows)}
    print(json.dumps(summary, indent=2), flush=True)
    _write(
        "decision", {"passed": not findings, "findings": findings, "summary": summary, "rows": rows}
    )
    print(f"decision: {'PASS' if not findings else 'FAIL'} {findings}")
    if findings:
        raise SystemExit(1)


def gate_text(env: dict[str, str], *, budget_s: float = 1800) -> None:
    """Text conversations (datasets/text-set.json: everyday, expert, and tools
    declared but no call needed) stream an answer under the decision rule at
    the default effort (max)."""

    deadline = Deadline("text", budget_s)
    items = _dataset("text-set.json")
    picked = [item for item in items if not item.get("tools")][:4] + [
        item for item in items if item.get("tools")
    ][:2]
    rows = [
        chat(env, messages=item["messages"], tools=item.get("tools"), trace=True, stream=True)
        for item in picked
    ]
    deadline.check()
    _print_rows(rows)
    findings: list[str] = []
    _check("text", rows, findings, efforts=[None] * len(rows), tools=False)
    findings += [
        f"text: no first token (status={row['status']})" for row in rows if row["ttft_s"] is None
    ]
    findings += _unread_findings(rows)
    ttfts = [row["ttft_s"] for row in rows if row["ttft_s"] is not None]
    summary = {
        **_summary(rows),
        "decision": _decision_summary(rows),
        "ttft_p50_s": statistics.median(ttfts) if ttfts else None,
    }
    print(json.dumps(summary, indent=2), flush=True)
    _write("text", {"passed": not findings, "findings": findings, "summary": summary, "rows": rows})
    print(f"text: {'PASS' if not findings else 'FAIL'} {findings}")
    if findings:
        raise SystemExit(1)


def gate_effort(env: dict[str, str], *, budget_s: float = 5400) -> None:
    """The caller's effort (default max) reaches the answer and its fix, for a
    text conversation and an agent turn."""

    deadline = Deadline("effort", budget_s)
    text = [item for item in _dataset("text-set.json") if not item.get("tools")][0]
    turn = _dataset("tool-turns.json")[0]
    findings: list[str] = []
    report = {}
    for effort in (None, "low", "high", "max"):
        extra = {} if effort is None else {"reasoning_effort": effort}
        answer = chat(env, messages=text["messages"], trace=True, **extra)
        tool = chat(env, messages=turn["messages"], tools=turn["tools"], trace=True, **extra)
        tool["declared_tools"] = turn["tools"]
        _check(f"effort={effort} text", [answer], findings, efforts=[effort], tools=False)
        _check(f"effort={effort} tool", [tool], findings, efforts=[effort], tools=True)
        print(f"effort={effort or DEFAULT_EFFORT}:")
        _print_rows([answer, tool])
        report[effort or "default"] = {"rows": [answer, tool]}
        deadline.check()
    _write("effort", {"passed": not findings, "findings": findings, "report": report})
    print(f"effort: {'PASS' if not findings else 'FAIL'} {findings}")
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
    """quyet-judge (its vLLM and adapter) down: an agent turn's answer is
    published unverified, with no fix and still a structured call; back: the
    decision rule holds again."""

    deadline = Deadline("fallback", budget_s)
    turn = _dataset("tool-turns.json")[0]

    def ask() -> dict:
        row = chat(env, messages=turn["messages"], tools=turn["tools"], trace=True)
        row["declared_tools"] = turn["tools"]
        return row

    findings: list[str] = []
    phases = {}
    try:
        for name in _replica_services("quyet-judge"):
            _compose(env, "stop", name)
        unjudged = ask()
        phases["judge_down"] = [unjudged]
        stages = unjudged["stages"]
        passes = (stages.get("decision") or {}).get("passes") or []
        problems = _tool_call_problems(unjudged, turn["tools"])
        if (
            unjudged["status"] != 200
            or (stages.get("answer") or {}).get("runs") != 1
            or passes != [None]
            or (unjudged["verification"] or {}).get("guaranteed") is not False
            or problems
        ):
            findings.append(
                f"judge down: status={unjudged['status']} stages={stages} "
                f"report={unjudged['verification']} {problems}"
            )
    finally:
        for name in _replica_services("quyet-judge"):
            _compose(env, "start", name)
            _wait_healthy(name)
    deadline.check()
    recovered = ask()
    phases["recovered"] = [recovered]
    _check("after restart", [recovered], findings, efforts=[None], tools=True)
    if None in ((recovered["stages"].get("decision") or {}).get("passes") or [None]):
        findings.append("after restart: the decision was not read")
    for name, rows in phases.items():
        print(f"{name}:")
        _print_rows(rows)
    _write("fallback", {"passed": not findings, "findings": findings, "phases": phases})
    print(f"fallback: {'PASS' if not findings else 'FAIL'} {findings}")
    if findings:
        raise SystemExit(1)


def _load(env: dict[str, str], gate: str, batches: dict[int, list[dict]]) -> dict:
    """Each level's requests at once. Every request must be answered under the
    decision rule; an agent turn (declared tools) must reply with a structured
    call to a declared tool in under 900 s."""

    report = {}
    findings: list[str] = []
    # Each request carries the answer, its decision and at most one fix (6 h).
    deadline = Deadline(gate, 21600)
    for concurrency, prompts in batches.items():
        started = time.monotonic()
        rows = _run_concurrently(env, prompts, concurrency)
        wall = time.monotonic() - started
        for row, prompt in zip(rows, prompts, strict=True):
            row["declared_tools"] = prompt.get("tools") or []
            row["kind"] = prompt["kind"]
        level: list[str] = []
        for kind in ("tool", "text"):
            subset = [row for row in rows if row["kind"] == kind]
            if subset:
                _check(
                    f"c{concurrency} {kind}",
                    subset,
                    level,
                    efforts=[None] * len(subset),
                    tools=kind == "tool",
                )
        summary = {
            **_summary(rows),
            "wall_s": round(wall, 1),
            "requests_per_min": round(60 * len(rows) / wall, 2),
            "decision": _decision_summary(rows),
            "findings": level,
        }
        summary["orchestration_output_tok_per_s"] = round(
            summary["orchestration_output_tokens"] / wall, 1
        )
        if len({row["kind"] for row in rows}) > 1:
            summary["per_kind"] = {
                kind: {
                    **_summary([row for row in rows if row["kind"] == kind]),
                    "decision": _decision_summary([row for row in rows if row["kind"] == kind]),
                }
                for kind in ("tool", "text")
            }
        print(f"c{concurrency}:")
        _print_rows(rows)
        print(json.dumps(summary, indent=2), flush=True)
        report[f"c{concurrency}"] = {"summary": summary, "rows": rows}
        findings += level
        deadline.check()
    findings += _unread_findings([row for entry in report.values() for row in entry["rows"]])
    _write(gate, {"passed": not findings, "findings": findings, "report": report})
    print(f"{gate}: {'PASS' if not findings else 'FAIL'} {findings}")
    if findings:
        raise SystemExit(1)
    return report


def _turn(item: dict) -> dict:
    return {"messages": item["messages"], "tools": item["tools"], "trace": True, "kind": "tool"}


def _text(item: dict) -> dict:
    return {"messages": item["messages"], "tools": item.get("tools"), "trace": True, "kind": "text"}


def gate_serving(env: dict[str, str]) -> None:
    """Agent turns (datasets/tool-turns.json) under load: every reply a
    structured call under the decision rule; fixes, latency, tokens, tok/s and
    stage times recorded."""

    items = _dataset("tool-turns.json")
    batches, offset = {}, 0
    for concurrency, count in PLAN.items():
        batches[concurrency] = [_turn(items[(offset + i) % len(items)]) for i in range(count)]
        offset += count
    _load(env, "serving", batches)


def gate_serving_mixed(env: dict[str, str]) -> None:
    """Agent turns and text conversations mixed under load: every request
    answered under the decision rule, per kind latency, tokens and fixes."""

    turns = _dataset("tool-turns.json")
    texts = _dataset("text-set.json")
    random.Random(3).shuffle(texts)
    mixed = [
        prompt
        for pair in zip(turns, texts[: len(turns)], strict=True)
        for prompt in (_turn(pair[0]), _text(pair[1]))
    ]
    batches, offset = {}, 0
    for concurrency, count in PLAN.items():
        batches[concurrency] = [mixed[(offset + i) % len(mixed)] for i in range(count)]
        offset += count
    _load(env, "serving-mixed", batches)


def gate_browser(_env: dict[str, str]) -> None:
    subprocess.run([str(HERE / "browser-smoke.sh")], check=True)
    _write("browser", {"passed": True})


GATES = {
    "l1": gate_l1,
    "decision": gate_decision,
    "text": gate_text,
    "effort": gate_effort,
    "fallback": gate_fallback,
    "serving": gate_serving,
    "serving-mixed": gate_serving_mixed,
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
