"""L1/L3 OpenAI Responses equivalence gate (m11 D4 amendment 2026-10-17).

Sends the same ``/v1/responses`` requests to an L1 server directly and to
Kairyu L3, then requires the same Responses structure from both:

- every L3 event and final response validates against the openai-python
  Responses types; an L1 event outside them (vLLM's
  ``response.reasoning_part.*``) is recorded as an L1 extension, and vLLM's
  reasoning part events are compared as the ``response.content_part.*``
  events the OpenAI types (and Kairyu) emit for the same part;
- each stream is a well-formed sequence (``response.created`` first, one
  terminal event last, increasing ``sequence_number``) whose output-text
  deltas join to the final output text;
- per case, L1 and L3 agree on status, output item types, the collapsed event
  types, the usage keys, function names and the case's own check (an exact
  answer, a reasoning item, a tool round trip, an image answer, a length cap);
- with ``--l1-metrics`` (once per L1 replica, ``CONTAINER=URL`` when the
  replica publishes no host port), the replicas' own request counters show
  that L3 reached them on ``/v1/responses``;
- with ``--l3-orchestrated`` the L3 model is an AUTO model whose output items
  come from its roles, so L1 and L3 are compared on status and usage keys
  only; both sides are still validated in full.

Requests are captured by a stdlib program that runs locally, or inside the L1
container (``--l1-docker``) when L1 publishes no host port.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import struct
import subprocess
import sys
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

_FIXTURE = Path(__file__).with_name("fixtures") / "openwebui_v0.12.3_responses_body.json"
_TERMINAL = {"response.completed", "response.incomplete", "response.failed"}
# vLLM opens and closes a reasoning item's content part with its own event names.
_SAME_EVENT = {
    "response.reasoning_part.added": "response.content_part.added",
    "response.reasoning_part.done": "response.content_part.done",
}
_CAPTURE_PROGRAM = r"""
import json, sys, urllib.error, urllib.request
url, key = sys.argv[1], sys.argv[2]
headers = {"Content-Type": "application/json"}
if key:
    headers["Authorization"] = "Bearer " + key
request = urllib.request.Request(url, data=sys.stdin.buffer.read(), headers=headers)
try:
    with urllib.request.urlopen(request, timeout=1800) as response:
        sys.stdout.write(json.dumps({"status": response.status, "body": response.read().decode()}))
except urllib.error.HTTPError as error:
    body = error.read().decode("utf-8", "replace")
    sys.stdout.write(json.dumps({"status": error.code, "body": body}))
"""
_METRICS_PROGRAM = (
    "import sys, urllib.request; "
    "sys.stdout.write(urllib.request.urlopen(sys.argv[1]).read().decode())"
)


@dataclass(frozen=True)
class Target:
    name: str
    base_url: str
    model: str
    api_key: str = ""
    docker: str | None = None


@dataclass
class Capture:
    status: int
    events: list[dict] = field(default_factory=list)
    response: dict | None = None
    raw: str = ""


_SENT = {"l1": 0, "l3": 0}


def capture(target: Target, body: dict) -> Capture:
    _SENT[target.name] = _SENT.get(target.name, 0) + 1
    url = f"{target.base_url.rstrip('/')}/responses"
    payload = json.dumps({**body, "model": target.model}).encode()
    command = (
        [
            "docker",
            "exec",
            "-i",
            target.docker,
            "python3",
            "-c",
            _CAPTURE_PROGRAM,
            url,
            target.api_key,
        ]
        if target.docker
        else [sys.executable, "-c", _CAPTURE_PROGRAM, url, target.api_key]
    )
    completed = subprocess.run(command, input=payload, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    raw = result["body"]
    if body.get("stream"):
        events = [
            json.loads(line[len("data:") :])
            for line in raw.splitlines()
            if line.startswith("data:") and line[len("data:") :].strip()
        ]
        terminal = next((e for e in reversed(events) if e.get("type") in _TERMINAL), None)
        response = terminal.get("response") if terminal else None
        return Capture(result["status"], events, response, raw)
    try:
        response = json.loads(raw)
    except ValueError:
        response = None
    return Capture(result["status"], [], response, raw)


def _validators() -> tuple[Callable[[dict], None], Callable[[dict], None]] | None:
    try:
        from openai.types.responses import Response, ResponseStreamEvent
        from pydantic import TypeAdapter
    except ImportError:
        return None
    event_adapter = TypeAdapter(ResponseStreamEvent)
    return event_adapter.validate_python, Response.model_validate


def _output_text(response: Mapping) -> str:
    return "".join(
        part.get("text", "")
        for item in response.get("output") or ()
        if item.get("type") == "message"
        for part in item.get("content") or ()
        if part.get("type") == "output_text"
    )


def summarize(cap: Capture, *, strict: bool) -> tuple[dict, list[str]]:
    """The comparable structure of one capture and its own findings."""

    findings: list[str] = []
    if cap.status != 200 or cap.response is None:
        return {"status": cap.status}, [f"HTTP {cap.status}: {cap.raw[:300]}"]
    validators = _validators()
    extensions: list[str] = []
    if cap.events:
        kinds = [event.get("type") for event in cap.events]
        if kinds[0] != "response.created":
            findings.append(f"first event is {kinds[0]!r}, not response.created")
        terminals = [kind for kind in kinds if kind in _TERMINAL]
        if len(terminals) != 1 or kinds[-1] not in _TERMINAL:
            findings.append(f"stream needs exactly one terminal event last; got {terminals}")
        numbers = [e["sequence_number"] for e in cap.events if "sequence_number" in e]
        if numbers != sorted(set(numbers)):
            findings.append("sequence_number is not strictly increasing")
        streamed = "".join(
            e.get("delta", "") for e in cap.events if e.get("type") == "response.output_text.delta"
        )
        if streamed != _output_text(cap.response):
            findings.append("output_text deltas do not join to the final output text")
        if validators is not None:
            for event in cap.events:
                try:
                    validators[0](event)
                except Exception:  # noqa: BLE001 - recorded as a finding or an extension
                    if strict:
                        findings.append(f"event {event.get('type')!r} fails the OpenAI types")
                    else:
                        extensions.append(str(event.get("type")))
        collapsed: list[str] = []
        for kind in kinds:
            if kind in _SAME_EVENT:
                kind = _SAME_EVENT[kind]
            elif kind in extensions or kind == "response.in_progress":
                continue
            if not collapsed or collapsed[-1] != kind or not str(kind).endswith(".delta"):
                collapsed.append(str(kind))
    else:
        collapsed = []
    if validators is not None:
        try:
            validators[1](cap.response)
        except Exception as error:  # noqa: BLE001
            findings.append(f"response fails the OpenAI types: {str(error)[:200]}")
    usage = cap.response.get("usage") or {}
    summary = {
        "status": cap.response.get("status"),
        "incomplete_reason": (cap.response.get("incomplete_details") or {}).get("reason"),
        "items": [item.get("type") for item in cap.response.get("output") or ()],
        "events": collapsed,
        "usage_keys": sorted(key for key in usage if not key.startswith("orchestration_")),
        "functions": [
            item.get("name")
            for item in cap.response.get("output") or ()
            if item.get("type") == "function_call"
        ],
        "text": _output_text(cap.response),
        "input_tokens": usage.get("input_tokens"),
        "l1_extension_events": sorted(set(extensions)),
    }
    return summary, findings


_COMPARED = ("status", "incomplete_reason", "items", "events", "usage_keys", "functions")
_COMPARED_ORCHESTRATED = ("status", "incomplete_reason", "usage_keys")


def compare(
    l1: dict, l3: dict, *, exact_input_tokens: bool = False, orchestrated: bool = False
) -> list[str]:
    findings = [
        f"{key}: L1 {l1.get(key)!r} != L3 {l3.get(key)!r}"
        for key in (_COMPARED_ORCHESTRATED if orchestrated else _COMPARED)
        if l1.get(key) != l3.get(key)
    ]
    if exact_input_tokens and l1.get("input_tokens") != l3.get("input_tokens"):
        findings.append(f"input_tokens: L1 {l1.get('input_tokens')} != L3 {l3.get('input_tokens')}")
    return findings


def _red_png(side: int = 32) -> str:
    row = b"\x00" + b"\xff\x00\x00" * side
    raw = zlib.compress(row * side)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", raw) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


_ARITHMETIC = "What is 17 * 19? Reply with the number only."
_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def _answer_check(text: str) -> str | None:
    return None if re.search(r"\b323\b", text) else f"answer {text[:80]!r} is not 323"


def run_case(case: str, targets: tuple[Target, Target], args) -> dict:
    """Run one case on (L1, L3); returns its record with findings."""

    record: dict = {"case": case, "findings": []}
    base = {"max_output_tokens": args.max_output_tokens, "store": False}

    def both(body: dict, *, check=None, exact=False) -> tuple[dict, dict]:
        summaries = []
        for target in targets:
            summary, own = summarize(capture(target, body), strict=target.name == "l3")
            record["findings"] += [f"{target.name}: {f}" for f in own]
            if check is not None and summary.get("status") is not None:
                problem = check(summary)
                if problem:
                    record["findings"].append(f"{target.name}: {problem}")
            summaries.append(summary)
        record["findings"] += compare(
            *summaries, exact_input_tokens=exact, orchestrated=args.l3_orchestrated
        )
        record.setdefault("summaries", []).append(summaries)
        return summaries[0], summaries[1]

    if case in {"text", "stream"}:
        both(
            {**base, "input": _ARITHMETIC, "stream": case == "stream"},
            check=lambda s: _answer_check(s["text"]),
            exact=args.exact_input_tokens,
        )
    elif case == "effort":
        for effort in args.efforts:
            both(
                {**base, "input": _ARITHMETIC, "stream": True, "reasoning": {"effort": effort}},
                check=lambda s: (
                    None
                    if "reasoning" in s["items"] or not args.expect_reasoning
                    else "no reasoning item"
                ),
            )
    elif case == "tools":
        first_turn = {
            **base,
            "input": "Use the tool to get the weather in Paris.",
            "tools": [_TOOL],
            "tool_choice": "required",
            "stream": True,
        }
        for target in targets:
            cap = capture(target, first_turn)
            summary, own = summarize(cap, strict=target.name == "l3")
            record["findings"] += [f"{target.name}: {f}" for f in own]
            call = next(
                (
                    i
                    for i in (cap.response or {}).get("output") or ()
                    if i.get("type") == "function_call"
                ),
                None,
            )
            if call is None:
                record["findings"].append(f"{target.name}: no function_call")
                continue
            second = {
                **base,
                "input": [
                    {"role": "user", "content": first_turn["input"]},
                    {k: call[k] for k in ("type", "call_id", "name", "arguments")},
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": "18C, sunny",
                    },
                ],
                "tools": [_TOOL],
                "stream": True,
            }
            second_summary, own = summarize(capture(target, second), strict=target.name == "l3")
            record["findings"] += [f"{target.name}: {f}" for f in own]
            if second_summary.get("status") != "completed" or "message" not in second_summary.get(
                "items", []
            ):
                record["findings"].append(f"{target.name}: the tool result turn did not answer")
            record.setdefault("summaries", []).append([target.name, summary, second_summary])
    elif case == "image":
        both(
            {
                **base,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "What single color fills this image? One word.",
                            },
                            {"type": "input_image", "image_url": _red_png(), "detail": "auto"},
                        ],
                    }
                ],
                "stream": True,
            },
            check=lambda s: (
                None if "red" in s["text"].lower() else f"answer {s['text'][:60]!r} is not red"
            ),
        )
    elif case == "cap":
        both(
            {
                "store": False,
                "input": "Count from 1 to 500, one number per line.",
                "max_output_tokens": 16,
            },
            check=lambda s: (
                None
                if s["status"] == "incomplete" and s["incomplete_reason"] == "max_output_tokens"
                else f"status {s['status']} / {s['incomplete_reason']}"
            ),
        )
    elif case == "openwebui":
        body = json.loads(_FIXTURE.read_text())["body"]
        both(body, check=lambda s: _answer_check(s["text"]))
    else:
        raise ValueError(f"unknown case {case!r}")
    return record


def _metric_counts(endpoints: list[str]) -> dict[str, float]:
    """Request counts per handler, summed over every L1 replica's /metrics."""

    counts: dict[str, float] = {}
    for endpoint in endpoints:
        container, _, url = endpoint.partition("=")
        if "://" in container:
            container, url = "", endpoint
        command = (
            ["docker", "exec", "-i", container, "python3", "-c", _METRICS_PROGRAM, url]
            if container
            else [sys.executable, "-c", _METRICS_PROGRAM, url]
        )
        text = subprocess.run(command, capture_output=True, check=True, text=True).stdout
        for line in text.splitlines():
            if not line.startswith("http_requests_total{"):
                continue
            handler = re.search(r'handler="([^"]+)"', line)
            if handler:
                counts[handler.group(1)] = counts.get(handler.group(1), 0.0) + float(
                    line.rsplit(" ", 1)[1]
                )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--l1", required=True, help="L1 OpenAI base URL (…/v1)")
    parser.add_argument("--l1-model", required=True)
    parser.add_argument("--l1-docker", default=None, help="run L1 captures inside this container")
    parser.add_argument(
        "--l1-metrics",
        action="append",
        default=[],
        help="an L1 replica's Prometheus URL (…/metrics), or CONTAINER=URL to read it "
        "inside that container; repeat once per replica L3 can reach",
    )
    parser.add_argument("--l3", required=True, help="Kairyu OpenAI base URL (…/v1)")
    parser.add_argument("--l3-model", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--cases",
        default="text,stream,effort,tools,cap,openwebui",
        help="comma list of text, stream, effort, tools, image, cap, openwebui",
    )
    parser.add_argument("--efforts", default="low,high,xhigh")
    parser.add_argument("--expect-reasoning", action="store_true")
    parser.add_argument("--exact-input-tokens", action="store_true")
    parser.add_argument(
        "--l3-orchestrated",
        action="store_true",
        help="the L3 model is an AUTO model: compare status and usage keys only",
    )
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    args.efforts = [effort for effort in args.efforts.split(",") if effort]
    targets = (
        Target("l1", args.l1, args.l1_model, args.api_key, args.l1_docker),
        Target("l3", args.l3, args.l3_model, args.api_key),
    )
    before = _metric_counts(args.l1_metrics) if args.l1_metrics else None
    records = [run_case(case, targets, args) for case in args.cases.split(",") if case]
    report: dict = {"cases": records}
    if before is not None:
        after = _metric_counts(args.l1_metrics)
        delta = {h: after.get(h, 0.0) - before.get(h, 0.0) for h in after}
        report["l1_request_delta"] = delta
        report["requests_sent"] = dict(_SENT)
        if delta.get("/v1/responses", 0.0) < _SENT["l1"] + _SENT["l3"]:
            report.setdefault("findings", []).append(
                "L1 counted fewer /v1/responses requests than L1 and L3 sent together"
            )
        if delta.get("/v1/chat/completions", 0.0) > 0:
            report.setdefault("findings", []).append(
                "L3 reached L1 on /v1/chat/completions during a Responses-only run"
            )
    failures = [f for record in records for f in record["findings"]] + report.get("findings", [])
    report["passed"] = not failures
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for failure in failures:
        print(f"FAIL {failure}")
    print("api-equivalence " + ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
