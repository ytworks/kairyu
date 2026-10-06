#!/usr/bin/env python3
"""GPU gates for routed DeepSeek-V4.1 answers (VCO-D18, VCO-D19).

Every gate has a time budget and writes its per-request evidence (latency,
tokens, tok/s, route, efforts) to model-volumes/<environment>/results/<gate>-<UTC>.json.

  l1              every DeepSeek DP rank (thinking and chat, grammar JSON),
                  Qwen chat, Winnow chat and Winnow System One on its L1
  routing         Winnow routes accuracy-critical conversations to VERIFIED
                  (miss rate < 10 % on the calibration and held-out halves)
  think-route     everyday requests stream from deepseek_think at the default effort
  effort          the caller's effort reaches the think route and the verified
                  route's drafts and answer; Qwen's requirements always think at low
  verified-route  a verified request runs the three waves: five DeepSeek drafts
                  beside Qwen's requirements, one Winnow judgment read (5 + 5 x N
                  items), then the answer, at the efforts the effort gate checks
  fallback        Winnow down: 200 on the think route; Winnow back: routed again
  serving         kairyu-verified-always at c1/c4/c8/c16: latency, tokens, tok/s
  serving-routed  kairyu-verified at c1/c4/c8/c16: route mix, latency, tokens per route
  browser         the answer page and Open WebUI answer in a real browser
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
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
ROUTED = control.ROUTED_MODEL
ALWAYS = control.ALWAYS_MODEL
# Concurrency -> requests, kept from the previous verified example's serving gates.
PLAN = {int(level): count for level, count in SPEC["verification"]["concurrency_plan"].items()}
INFOBENCH_URL = (
    "https://huggingface.co/datasets/kqsong/InFoBench/resolve/"
    "cef03a2830944bfb0d201107895ddd0e0e90bf0e/InfoBench.json"
)
INFOBENCH_SHA256 = "66a2ee8d70208a7a879a17f471ba0ffd097b698265ea4796be99a273fdcc6559"


def _env() -> dict[str, str]:
    return control._compose_env()


def _api(env: dict[str, str]) -> str:
    return f"http://127.0.0.1:{env['API_PORT']}"


def _results_dir() -> Path:
    path = control.environment_storage() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(gate: str, payload: dict) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = _results_dir() / f"{gate}-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"evidence: {path}", flush=True)
    return path


def _dataset(name: str) -> list[dict]:
    return json.loads((DATASETS / name).read_text(encoding="utf-8"))


def _route(trace: dict | None, model: str) -> tuple[str, float | None]:
    """The profile that served the request and Winnow's P(VERIFIED), from the trace."""

    if model == ALWAYS:
        return "verified (always)", None
    events = (trace or {}).get("events") or []
    judge = next((event for event in events if event.get("node") == "profile_judge"), None)
    if judge is None:
        return "unknown", None
    detail = judge.get("detail") or {}
    verdict = detail.get("verdict")
    p_verified = detail.get("p_VERIFIED")
    if verdict is None:
        return f"deepseek_think (fallback: {detail.get('fallback')})", p_verified
    return ("verified" if verdict == "primary" else verdict), p_verified


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


def _efforts(trace: dict | None, worker: str = "deepseek") -> list[str | None]:
    """The reasoning effort of every generation of ``worker`` in the trace."""

    return [
        (event.get("detail") or {}).get("reasoning_effort")
        for event in (trace or {}).get("events") or []
        if event.get("kind") == "generation"
        and event.get("worker") == worker
        and event.get("status") == "success"
    ]


def _seconds(timing: dict | None, key: str) -> datetime | None:
    try:
        return datetime.fromisoformat((timing or {})[key].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def _stages(trace: dict | None) -> dict[str, dict]:
    """The verified route's stages from the trace: status, wall time, start
    and end (seconds from the first stage), and the judgment's item count."""

    events = [
        event
        for event in (trace or {}).get("events") or []
        if event.get("node") in {"drafts", "requirements", "judgments", "answer"}
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
        stages[event["node"]] = {
            "status": event["status"],
            "start_s": round((start - origin).total_seconds(), 2) if start and origin else None,
            "end_s": round((end - origin).total_seconds(), 2) if end and origin else None,
            **({"items": detail.get("items")} if event["kind"] == "verification" else {}),
            **({"reason": detail.get("reason")} if detail.get("unavailable") else {}),
        }
    return stages


def _three_waves(row: dict, caller_effort: str | None) -> list[str]:
    """What a verified row misses of the three-wave route (empty when sound):
    the four stages, wave-1 overlap, 5 + 5 x N judgments, and the efforts
    (DeepSeek's drafts and answer at the caller's, default high; Qwen at low)."""

    stages = row["stages"]
    problems = []
    for node in ("drafts", "requirements", "judgments", "answer"):
        if (stages.get(node) or {}).get("status") != "success":
            problems.append(f"{node}={stages.get(node)}")
    if problems:
        return problems
    items = stages["judgments"].get("items") or 0
    # 5 adoption judgments plus 5 per requirement.
    if items < 10 or items % 5:
        problems.append(f"judgment items={items}")
    drafts, requirements = stages["drafts"], stages["requirements"]
    if None in (drafts["end_s"], requirements["start_s"]) or (
        requirements["start_s"] >= drafts["end_s"]
    ):
        problems.append(f"wave 1 not parallel: drafts={drafts} requirements={requirements}")
    wanted = caller_effort or "high"
    if row["efforts"] != [wanted, wanted] or row["qwen_efforts"] != ["low"]:
        problems.append(f"efforts={row['efforts']} qwen={row['qwen_efforts']}")
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
    model: str = ALWAYS,
    trace: bool = False,
    stream: bool = False,
    timeout_s: float = 7200,
    **extra,
) -> dict:
    """One chat request; returns the per-request evidence row."""

    payload = control.routed_request(content or "", model=model, **extra)
    if messages is not None:
        payload["messages"] = messages
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
    route, p_verified = _route(trace_body, model)
    return {
        "status": status,
        "seconds": round(elapsed, 2),
        "ttft_s": round(ttft, 2) if ttft is not None else None,
        "model": model,
        "route": route,
        "p_verified": p_verified,
        "efforts": _efforts(trace_body),
        "qwen_efforts": _efforts(trace_body, "qwen"),
        "stages": _stages(trace_body),
        "judge_s": _judge_seconds(trace_body),
        "public_completion_tokens": usage.get("completion_tokens"),
        "orchestration_input_tokens": usage.get("orchestration_input_tokens")
        or usage.get("prompt_tokens"),
        "orchestration_output_tokens": output_tokens,
        "orchestration_output_tok_per_s": round(output_tokens / elapsed, 1) if elapsed else None,
        "has_verification": "kairyu_verification" in body,
        "content": message.get("content") or "",
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
            f"ttft={row.get('ttft_s')} route={row['route']} p_verified={row.get('p_verified')} "
            f"efforts={row['efforts']} qwen={row.get('qwen_efforts')} judge_s={row.get('judge_s')} "
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


def infobench(limit: int, seed: int) -> list[dict]:
    path = control.environment_storage() / "calibration" / "InfoBench.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        with urllib.request.urlopen(INFOBENCH_URL, timeout=120) as response:
            path.write_bytes(response.read())
    import hashlib

    if hashlib.sha256(path.read_bytes()).hexdigest() != INFOBENCH_SHA256:
        raise SystemExit(f"{path} does not match the pinned InFoBench revision")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def _infobench_content(row: dict) -> str:
    return "\n\n".join(part for part in (row["instruction"], row.get("input")) if part)


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


def _routing_probabilities(env: dict[str, str]) -> list[dict[str, float] | None]:
    """Every route's probability for each routing-set conversation, through
    the served judge.

    The orchestrator is built from verified.yaml with the real System One
    backend pointed at Winnow, so the request Winnow reads is the one Kairyu
    sends in serving; no generation runs.
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
            base_urls=(f"http://127.0.0.1:{env['WINNOW_L1_PORT']}",),
            upstream_model=SPEC["systemone"]["upstream_model"],
            max_concurrency=8,
            max_queue=1024,
            queue_wait_s=600,
        )
        orchestrator = build_orchestrator(
            load_spec(HERE / "verified.yaml"),
            engine_refs={SPEC["deepseek"]["served_name"]: MockBackend()},
            systemone_refs={SPEC["systemone"]["model"]: backend},
        )

        async def one(item: dict) -> dict[str, float] | None:
            chat_request = ChatCompletionRequest(model=ROUTED, messages=item["messages"])
            call = OrchestrationRequest(
                prompt=validate_orchestration_chat_input(chat_request).prompt,
                sampling_params=SamplingParams(max_tokens=1024),
            )
            judged = await orchestrator.judge_role_profile(call)
            metadata = judged.role_profile_judge_event.metadata
            probabilities = {
                key.removeprefix("p_"): float(value)
                for key, value in metadata.items()
                if key.startswith("p_") and isinstance(value, (int, float))
            }
            return probabilities if "VERIFIED" in probabilities else None

        try:
            return await asyncio.gather(*(one(item) for item in _dataset("routing-set.json")))
        finally:
            await backend.shutdown()

    return asyncio.run(judge_all())


def _served_route(probabilities: dict[str, float], tau: float | None) -> str:
    # The served rule: VERIFIED when preferred (p >= tau, if a floor is
    # configured), else the most probable route.
    if tau is not None and probabilities["VERIFIED"] >= tau:
        return "VERIFIED"
    return max(probabilities, key=probabilities.__getitem__)


def gate_routing(env: dict[str, str], *, budget_s: float = 1800) -> None:
    """Winnow routes accuracy-critical conversations to VERIFIED.

    The configured rule (prefer.min_probability, or the most probable route
    when none is set) must miss fewer than 10 % of the VERIFIED conversations
    on both the calibration half (even items) and the held-out half (odd
    items). The largest floor meeting that on the calibration half is
    reported as recommended_tau.
    """

    deadline = Deadline("routing", budget_s)
    items = _dataset("routing-set.json")
    started = time.monotonic()
    probabilities = _routing_probabilities(env)
    judge_seconds = time.monotonic() - started
    if any(p is None for p in probabilities):
        raise SystemExit("routing: the judge did not answer every conversation")
    rows = [
        {**item, "p": p, "p_verified": p["VERIFIED"], "messages": len(item["messages"])}
        for item, p in zip(items, probabilities, strict=True)
    ]
    import yaml

    prefer = yaml.safe_load((HERE / "verified.yaml").read_text())["profile_judge"].get("prefer")
    configured = prefer["min_probability"] if prefer else None

    def miss_rate(subset: list[dict], tau: float | None) -> float:
        needed = [row for row in subset if row["label"] == "VERIFIED"]
        missed = [row for row in needed if _served_route(row["p"], tau) != "VERIFIED"]
        return len(missed) / len(needed)

    def easy_to_think(subset: list[dict], tau: float | None) -> float:
        easy = [row for row in subset if row["label"] == "THINK"]
        return sum(1 for row in easy if _served_route(row["p"], tau) == "THINK") / len(easy)

    calibration = rows[0::2]
    holdout = rows[1::2]
    candidates = sorted(
        {round(row["p_verified"], 4) for row in calibration} | {0.0, 0.5}, reverse=True
    )
    recommended = next(
        (tau for tau in candidates if tau <= 0.5 and miss_rate(calibration, tau) < 0.10), 0.0
    )
    by_category: dict[str, list[float]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row["p_verified"])
    summary = {
        "conversations": len(rows),
        "judge_wall_s": round(judge_seconds, 2),
        "configured_tau": configured,
        "recommended_tau": recommended,
        "calibration_miss_rate": round(miss_rate(calibration, configured), 4),
        "holdout_miss_rate": round(miss_rate(holdout, configured), 4),
        "everyday_to_think": round(easy_to_think(rows, configured), 4),
        "p_verified_by_category": {
            category: {
                "min": round(min(values), 4),
                "median": round(statistics.median(values), 4),
                "max": round(max(values), 4),
            }
            for category, values in by_category.items()
        },
    }
    deadline.check()
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    passed = summary["holdout_miss_rate"] < 0.10 and summary["calibration_miss_rate"] < 0.10
    _write("routing", {"passed": passed, "summary": summary, "rows": rows})
    print(
        f"routing: {'PASS' if passed else 'FAIL'} "
        f"(held-out miss rate {summary['holdout_miss_rate']}, "
        f"calibration {summary['calibration_miss_rate']})"
    )
    if not passed:
        raise SystemExit(1)


def gate_think_route(env: dict[str, str], *, budget_s: float = 1800) -> None:
    """Everyday requests stream from the think route at the default effort.

    The publisher's private reasoning is withheld by Kairyu's multi-stage
    contract (as on the sibling ensemble's deepseek_think route), so the gate
    checks the route, a streamed answer with its time to first token, no
    verification field, and that DeepSeek ran at the default effort (high).
    """

    deadline = Deadline("think-route", budget_s)
    items = [item for item in _dataset("routing-set.json") if item["label"] == "THINK"][:6]
    rows = [
        chat(env, messages=item["messages"], model=ROUTED, trace=True, stream=True)
        for item in items
    ]
    deadline.check()
    findings = []
    for row in rows:
        if row["status"] != 200 or not row["content"].strip() or row["ttft_s"] is None:
            findings.append(f"status={row['status']} empty={not row['content'].strip()}")
        elif row["route"] != "deepseek_think":
            findings.append(f"everyday request routed to {row['route']}")
        elif row["has_verification"] or row["efforts"] != ["high"]:
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
    verified route's drafts and answer; Qwen's requirements think at low."""

    deadline = Deadline("effort", budget_s)
    easy = next(item for item in _dataset("routing-set.json") if item["label"] == "THINK")
    verified_prompt = "List three primary colors as a comma-separated line, nothing else."
    findings = []
    report = {}
    for effort in (None, "low", "high", "max"):
        extra = {} if effort is None else {"reasoning_effort": effort}
        think = chat(env, messages=easy["messages"], model=ROUTED, trace=True, **extra)
        verified = chat(env, verified_prompt, model=ALWAYS, trace=True, **extra)
        expected = effort or "high"
        for row, wanted, qwen in (
            (think, [expected], []),
            (verified, [expected, expected], ["low"]),
        ):
            if row["status"] != 200 or row["efforts"] != wanted or row["qwen_efforts"] != qwen:
                findings.append(
                    f"{effort}: {row['route']} status={row['status']} efforts={row['efforts']} "
                    f"qwen={row['qwen_efforts']} (expected {wanted}, qwen {qwen})"
                )
        if think["route"] != "deepseek_think":
            findings.append(f"{effort}: everyday request routed to {think['route']}")
        print(f"effort={expected}:")
        _print_rows([think, verified])
        report[expected if effort else "default"] = {"rows": [think, verified]}
        deadline.check()
    _write("effort", {"passed": not findings, "findings": findings, "report": report})
    print(f"effort: {'PASS' if not findings else 'FAIL'} {findings}")
    if findings:
        raise SystemExit(1)


def gate_verified_route(env: dict[str, str], *, budget_s: float = 7200) -> None:
    """A verified request runs the three waves into one answer.

    Six VERIFIED conversations of the routing set go to the always-verified
    model, unary and streamed, with the caller's effort cycling through none,
    low, high and max: each must return 200 with a non-empty answer after
    DeepSeek's drafts and Qwen's requirements ran in parallel, one Winnow
    judgment read covered 5 + 5 x N items, and the efforts were the caller's
    (DeepSeek) and low (Qwen).
    """

    deadline = Deadline("verified-route", budget_s)
    items = [item for item in _dataset("routing-set.json") if item["label"] == "VERIFIED"][:6]
    efforts = [None, "low", "high", "max"]
    findings = []
    rows = []
    for index, item in enumerate(items):
        name = f"{item['category']}#{index}"
        for stream in (False, True):
            effort = efforts[(2 * index + stream) % len(efforts)]
            extra = {"reasoning_effort": effort} if effort else {}
            row = chat(
                env, messages=item["messages"], model=ALWAYS, trace=True, stream=stream, **extra
            )
            row.update(id=name, stream=stream, caller_effort=effort)
            rows.append(row)
            problems = _three_waves(row, effort)
            if row["status"] != 200 or not row["content"].strip() or problems:
                findings.append(
                    f"{name} stream={stream} effort={effort}: status={row['status']} "
                    f"empty={not row['content'].strip()} {problems}"
                )
        deadline.check()
    _print_rows(rows)
    summary = _summary(rows)
    print(json.dumps(summary, indent=2), flush=True)
    _write(
        "verified-route",
        {"passed": not findings, "findings": findings, "summary": summary, "rows": rows},
    )
    print(f"verified-route: {'PASS' if not findings else 'FAIL'} {findings}")
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


def gate_fallback(env: dict[str, str], *, budget_s: float = 3600) -> None:
    """Winnow down: the routed model answers on the think route; Winnow back:
    routed again."""

    deadline = Deadline("fallback", budget_s)
    prompt = "In two sentences, explain why the sky looks blue."
    findings = []
    phases = {}
    try:
        _compose(env, "stop", "winnow")
        down = [chat(env, prompt, model=ROUTED, trace=True) for _ in range(2)]
        phases["winnow_down"] = down
        for row in down:
            if (
                row["status"] != 200
                or not row["content"].strip()
                or not row["route"].startswith("deepseek_think (fallback")
            ):
                findings.append(f"winnow down: status={row['status']} route={row['route']}")
        always = chat(env, prompt, model=ALWAYS, trace=True)
        phases["winnow_down_always"] = [always]
        if always["status"] != 200 or not always["content"].strip():
            findings.append(f"winnow down, always model: status={always['status']}")
    finally:
        _compose(env, "start", "winnow")
        _wait_healthy("winnow")
    recovered = chat(env, prompt, model=ROUTED, trace=True)
    phases["recovered"] = [recovered]
    if recovered["status"] != 200 or "fallback" in recovered["route"]:
        findings.append(f"after restart: status={recovered['status']} route={recovered['route']}")
    deadline.check()
    for name, rows in phases.items():
        print(f"{name}:")
        _print_rows(rows)
    _write("fallback", {"passed": not findings, "findings": findings, "phases": phases})
    print(f"fallback: {'PASS' if not findings else 'FAIL'} {findings}")
    if findings:
        raise SystemExit(1)


def _load(
    env: dict[str, str], gate: str, batches: dict[int, list[dict]], *, per_route: bool
) -> dict:
    report = {}
    # Each verified request carries five drafts and the answer (plan: 6 h).
    deadline = Deadline(gate, 21600)
    for concurrency, prompts in batches.items():
        started = time.monotonic()
        rows = _run_concurrently(env, prompts, concurrency)
        wall = time.monotonic() - started
        summary = {
            **_summary(rows),
            "wall_s": round(wall, 1),
            "requests_per_min": round(60 * len(rows) / wall, 2),
        }
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
    ]
    _write(gate, {"passed": not failures, "failed": failures, "report": report})
    print(f"{gate}: {'PASS' if not failures else 'FAIL'} (every request answered)")
    if failures:
        raise SystemExit(1)
    return report


def gate_serving(env: dict[str, str]) -> None:
    """kairyu-verified-always under load, on InFoBench instructions."""

    pool = infobench(sum(PLAN.values()), seed=2)
    batches, offset = {}, 0
    for concurrency, count in PLAN.items():
        batches[concurrency] = [
            {"content": _infobench_content(row), "trace": True}
            for row in pool[offset : offset + count]
        ]
        offset += count
    _load(env, "serving", batches, per_route=False)


def gate_serving_routed(env: dict[str, str]) -> None:
    """kairyu-verified under load: route mix, latency and tokens per route."""

    items = _dataset("routing-set.json")
    random.Random(3).shuffle(items)
    batches, offset = {}, 0
    for concurrency, count in PLAN.items():
        batches[concurrency] = [
            {
                "messages": items[(offset + i) % len(items)]["messages"],
                "model": ROUTED,
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
    "verified-route": gate_verified_route,
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
    GATES[args.gate](_env())


if __name__ == "__main__":
    main()
