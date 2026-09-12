#!/usr/bin/env python3
"""Native vLLM and bounded L2 smoke; endpoints must already be running.

Example: python gpu_smoke.py --base-url http://127.0.0.1:8009/v1 \
    --output /tmp/v41-smoke --provenance /tmp/v41-container-inspect.json

Use --suite l2 with the Kairyu /v1 endpoint for traced route probes. The judge
chooses routes; a coverage_gap means a requested route was not exercised. Add
--l2-effort-matrix --case ^l2-route-primary for four API efforts.

This writes requests, raw responses, timing/errors, config hashes, and incremental
results.json. Protocol passes do not establish semantic checklist quality, effort
preamble selection, or server-side cancellation cleanup; inspect worker logs too.
No service lifecycle operations are performed. Requires PyYAML (same as the hook).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import yaml

REQUIREMENTS_QUERY = (
    "Compare option A (cost 10, latency 20 ms) with option B (cost 15, latency 8 ms). "
    "Recommend exactly one option when latency must be less than 10 ms. "
    "Use fewer than 100 words and end with exactly: Ready."
)
RED_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP8z8DAwMDA"
    "xMDAwMDAAAANHQEDasKb6QAAAABJRU5ErkJggg=="
)
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Return the value for a key.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
    },
}


# Source-audited 2026-09-13 without GPU execution. Native single-user encoding
# ends exactly at <think>; there are no prompt reasoning tokens to deduct.
# The sampler forces the next token at budget exhaustion (no speculation in
# this deployment). The parser counts REASONING_CHUNK only; the single-token
# </think> boundary emits REASONING_END, so marker allowance is ZERO, not +3.
THINKING_BUDGET = 16
BUDGET_ACCOUNTING = {
    "configured_budget": THINKING_BUDGET,
    "prompt_reasoning_tokens": 0,
    "marker_allowance_tokens": 0,
    "reasoning_tokens_upper_bound": THINKING_BUDGET,
    "think_start_token_id": 128821,
    "think_end_token_id": 128822,
    "usage_field": "usage.completion_tokens_details.reasoning_tokens",
    "scope": "native single-user thinking prompt, no speculative decoding",
    "source_evidence": {
        "vllm/tokenizers/deepseek_v41_encoding.py:494-502": {
            "sha256": "e4948074c299b46b7e24c8875026adddb92c9f01153248d01196030e419c260c",
        },
        "vllm/v1/sample/thinking_budget_state.py:416-440": {
            "sha256": "962f8f55210eb0a431cb9c78b013e35f7a7dd58d06d6fb2e7fcff1b457356f8e",
        },
        "vllm/parser/engine/parser_engine.py:646-648": {
            "sha256": "ef42fb624d7a3fe0fada2c6235b0bea7de54136dcc3289e28ae2e3bfa6649e64",
        },
        "vllm/parser/engine/streaming_parser_engine.py:179-184,267-271": {
            "sha256": "497fffefd9da3accd0c4b2097949265952f233f3dad92b08cad9b93db0be8223",
        },
        "checkpoint/tokenizer.json:added_tokens": {
            "sha256": "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b",
        },
    },
}


def build_cases(directory: Path, model: str) -> list[dict]:
    def base(text: str) -> dict:
        return {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 8192,
            "temperature": 1.0,
            "top_p": 1.0,
            "stream": False,
        }

    cases = []
    for effort in (None, "low", "high", "max", "nonthinking"):
        body = base("What is 17 times 19? Return only the integer.")
        if effort == "nonthinking":
            body["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
        elif effort:
            body["reasoning_effort"] = effort
        cases.append(
            {
                "name": f"native-{effort or 'default'}",
                "kind": "arithmetic",
                "payload": body,
                "expect_nonthinking": effort == "nonthinking",
            }
        )
    image = base("Name the solid color in the image. Return one color word only.")
    image["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
    image["messages"][0]["content"] = [
        {"type": "text", "text": image["messages"][0]["content"]},
        {"type": "image_url", "image_url": {"url": RED_IMAGE}},
    ]
    cases.append({"name": "native-image-red", "kind": "image", "payload": image})
    tool = base("Call lookup with key exactly probe. Do not answer without calling it.")
    tool.update(
        tools=[TOOL],
        tool_choice="required",
        chat_template_kwargs={"thinking": False, "enable_thinking": False},
    )
    cases.append({"name": "native-tool", "kind": "tool", "payload": tool})
    spec = yaml.safe_load((directory / "auto-max.yaml").read_text())
    role = next(r for r in spec["roles"] if r["name"] == "requirements")
    for effort in (None, "low", "high", "max"):
        for nested in (None, "low", "max"):
            body = base(role["prompt"].format(query=REQUIREMENTS_QUERY))
            body["max_tokens"] = role["sampling"]["max_tokens"]
            if effort:
                body["reasoning_effort"] = effort
            if nested:
                body["chat_template_kwargs"] = {
                    "reasoning_effort": nested,
                    "thinking": False,
                    "enable_thinking": False,
                }
            cases.append(
                {
                    "name": f"requirements-{effort or 'omitted'}-nested-{nested or 'omitted'}",
                    "kind": "requirements",
                    "payload": body,
                    "expected_effective_effort": "high",
                    "effort_observation": "Requires worker hook log/preamble evidence",
                }
            )
    stream = base("What is 17 times 19? Return only the integer.")
    stream.update(stream=True, chat_template_kwargs={"thinking": False, "enable_thinking": False})
    cases.append({"name": "native-stream", "kind": "arithmetic", "payload": stream})
    cancel = base("Write a long explanation of prime numbers, at least 1000 words.")
    cancel.update(stream=True, chat_template_kwargs={"thinking": False, "enable_thinking": False})
    cases.append({"name": "native-cancel", "kind": "cancel", "payload": cancel})
    cases.append(
        {
            "name": "native-after-cancel",
            "kind": "arithmetic",
            "payload": {**stream, "stream": False},
        }
    )
    budget = base(
        "Calculate 19 times 23 with deliberate care. In private reasoning, independently "
        "derive the result using (20-1)*23 and (20+3)*19, then verify the two calculations "
        "agree. Your public answer must contain only the resulting integer."
    )
    budget.update(
        max_tokens=512,
        thinking_token_budget=THINKING_BUDGET,
        reasoning_effort="high",
        chat_template_kwargs={"thinking": True, "enable_thinking": True},
    )
    cases.append({"name": "native-thinking-budget", "kind": "thinking-budget", "payload": budget})
    return cases


ROUTE_FINAL_NODES = {
    "primary": "synthesis",
    "qwen_direct": "qwen_answer",
    "qwen_think_medium": "qwen_think_answer",
    "deepseek_direct": "deepseek_answer",
    "deepseek_think": "deepseek_think_answer",
}


def build_l2_cases(directory: Path, *, effort_matrix: bool = False) -> list[dict]:
    """Route-targeted prompts: the public judge decides, so misses are coverage gaps."""
    quality = json.loads((directory / "requirements-quality-cases.json").read_text())["cases"]
    primary = next(c["request"] for c in quality if c["id"] == "headed-comparison")
    prompts = {
        "qwen_direct": "What is 17 times 19? Return only the integer.",
        "qwen_think_medium": "Solve this small scheduling puzzle carefully: "
        "A precedes C, B follows A, "
        "C precedes D, and B follows D. Give the unique ordering and explain briefly.",
        "deepseek_direct": "Give an expert overview comparing serializability, snapshot isolation, "
        "and read committed in database systems, including their typical anomalies. This is "
        "a conceptual knowledge question, not a proof or design exercise. Under 250 words.",
        "deepseek_think": "Prove or disprove: for every positive integer n, the sum of the first n "
        "positive cubes is the square of the sum of the first n positive integers. Derive "
        "the identity without assuming it, give an independent verification, and precisely "
        "state the induction invariant. Prioritize rigorous mathematical reasoning.",
    }
    cases = []
    for profile, prompt in prompts.items():
        body = {
            "model": "kairyu-auto-max",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16384,
            "stream": True,
            "stream_options": {"include_usage": True},
            "reasoning_effort": "low",
        }
        cases.append(
            {
                "name": "l2-route-" + profile,
                "kind": "answer",
                "payload": body,
                "trace": True,
                "expected_profile": profile,
            }
        )
    cases.append(
        {
            "name": "l2-route-primary",
            "kind": "answer",
            "payload": primary,
            "trace": True,
            "expected_profile": "primary",
            "expect_headless": False,
        }
    )
    if effort_matrix:
        for effort in (None, "high", "max"):
            body = {**primary}
            if effort is None:
                body.pop("reasoning_effort", None)
            else:
                body["reasoning_effort"] = effort
            cases.append(
                {
                    "name": f"l2-route-primary-{effort or 'omitted'}",
                    "kind": "answer",
                    "payload": body,
                    "trace": True,
                    "expected_profile": "primary",
                    "expect_headless": False,
                    "expected_effective_effort": "high",
                }
            )
    native = {c["name"]: c for c in build_cases(directory, "kairyu-auto-max")}
    for name in ("native-image-red", "native-tool", "native-cancel", "native-after-cancel"):
        case = native[name]
        body = {**case["payload"]}
        body.pop("chat_template_kwargs", None)
        body["reasoning_effort"] = "low"
        cases.append(
            {
                **case,
                "name": name.replace("native-", "l2-"),
                "payload": body,
                "trace": True,
                "expect_headless": True if name == "native-tool" else None,
            }
        )
    cases.append(
        {
            "name": "l2-headless-json",
            "kind": "json-answer",
            "trace": True,
            "expect_headless": True,
            "payload": {
                "model": "kairyu-auto-max",
                "stream": False,
                "max_tokens": 16384,
                "reasoning_effort": "low",
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "user", "content": 'Return exactly one JSON object: {"answer":323}.'}
                ],
            },
        }
    )
    return cases


def validate_l2_trace(body: dict, *, expected_profile=None, expect_headless=None) -> dict:
    events = (body.get("kairyu_trace_v2") or {}).get("events", [])
    successful = {e.get("node") for e in events if e.get("status") == "success"}
    profiles = [profile for profile, node in ROUTE_FINAL_NODES.items() if node in successful]
    profile = profiles[0] if len(profiles) == 1 else None
    result = {"passed": False, "observed_profile": profile, "coverage_gap": False}
    if not profile or "profile_judge" not in successful:
        return {**result, "detail": "missing judge or unambiguous final route trace"}
    if "image_description" in successful:
        return {**result, "detail": "obsolete image_description stage executed"}
    if profile == "primary":
        required = {
            "requirements",
            "draft",
            "policies",
            "answer_1",
            "answer_2",
            "critique",
            "audit",
        }
        if not required <= successful or successful & {"answer_3", "answer_4"}:
            return {**result, "detail": "primary DAG stage coverage failed"}
        if expect_headless is False and "head" not in successful:
            return {**result, "detail": "headed primary omitted its head"}
    elif successful & {"requirements", "audit", "policies", "answer_1", "answer_2"}:
        return {**result, "detail": "direct route executed ensemble stages"}
    if expect_headless is True and "head" in successful:
        return {**result, "detail": "headless request executed head"}
    if expected_profile and profile != expected_profile:
        return {
            **result,
            "coverage_gap": True,
            "detail": f"judge chose {profile}; requested coverage target {expected_profile}",
        }
    return {**result, "passed": True, "detail": "route and DAG trace contract satisfied"}


def validate_logprobs(value: object) -> None:
    """Reject nonfinite sampled or alternative token probabilities, including SSE."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "logprob":
                if not isinstance(item, (int, float)) or isinstance(item, bool):
                    raise ValueError("malformed returned logprob")
                if not math.isfinite(item):
                    raise ValueError("nonfinite returned logprob")
            else:
                validate_logprobs(item)
    elif isinstance(value, list):
        for item in value:
            validate_logprobs(item)


def validate_response(kind: str, body: object) -> dict:
    verdict = {"passed": False, "truncated": False, "detail": "malformed response"}
    try:
        validate_logprobs(body)
    except ValueError as exc:
        return {**verdict, "detail": str(exc)}
    try:
        choice = body["choices"][0]
        message = choice["message"]
        finish = choice.get("finish_reason")
        if kind == "thinking-budget":
            observed = ((body.get("usage") or {}).get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            )
            verdict.update(
                reasoning_tokens_observed=observed,
                reasoning_tokens_upper_bound=THINKING_BUDGET,
                marker_allowance_tokens=0,
                configured_budget=THINKING_BUDGET,
                budget_reached=type(observed) is int and observed >= THINKING_BUDGET,
            )
        verdict["finish_reason"] = finish
        if finish == "length":
            return {**verdict, "truncated": True, "detail": "output token limit reached"}
        if finish != ("tool_calls" if kind == "tool" else "stop"):
            return {**verdict, "detail": f"unexpected finish_reason: {finish}"}
        content = message.get("content") or ""
        valid = False
        if kind == "thinking-budget":
            if type(observed) is not int or observed < THINKING_BUDGET:
                return {
                    **verdict,
                    "outcome": "not_exercised",
                    "detail": "reported reasoning did not reach the budget (or usage missing)",
                }
            if observed > THINKING_BUDGET:
                return {**verdict, "outcome": "fail", "detail": "reasoning budget exceeded"}
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            if not isinstance(reasoning, str) or not reasoning.strip():
                return {
                    **verdict,
                    "outcome": "not_exercised",
                    "detail": "no reasoning text observed despite reported budget usage",
                }
            valid = content.strip() == "437"
            return {
                **verdict,
                "passed": valid,
                "outcome": "pass" if valid else "fail",
                "detail": "budget reached; public 437 completed"
                if valid
                else "incorrect public answer after budget exhaustion",
            }
        if kind == "answer":
            valid = bool(content.strip())
        elif kind == "json-answer":
            valid = json.loads(content) == {"answer": 323}
        elif kind == "arithmetic":
            valid = content.strip() == "323"
        elif kind == "image":
            valid = content.strip().lower().rstrip(".") == "red"
        elif kind == "tool-result":
            valid = content.strip() == "SMOKE_VALUE_731"
        elif kind == "tool":
            calls = message.get("tool_calls") or []
            valid = len(calls) == 1 and bool(calls[0].get("id"))
            if valid:
                call = calls[0]
                valid = (
                    call.get("type") == "function"
                    and call["function"]["name"] == "lookup"
                    and json.loads(call["function"]["arguments"]) == {"key": "probe"}
                )
        elif kind == "requirements":
            rows = json.loads(content)
            keys = {"id", "priority", "requirement", "acceptance_criterion", "source"}
            valid = isinstance(rows, list) and bool(rows)
            if valid:
                valid = all(
                    isinstance(row, dict)
                    and set(row) == keys
                    and all(isinstance(value, str) and value.strip() for value in row.values())
                    and row["id"] == f"R{i}"
                    and row["priority"] in {"minimum", "optional"}
                    for i, row in enumerate(rows, 1)
                )
        return {
            **verdict,
            "passed": valid,
            "detail": "protocol contract satisfied" if valid else "content contract failed",
        }
    except (KeyError, IndexError, TypeError, ValueError):
        return verdict


def parse_stream(raw: str) -> tuple[dict, bool]:
    content, reasoning, finish, done, usage = "", "", None, False, None
    trace = None
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        text = line[5:].strip()
        if text == "[DONE]":
            done = True
            continue
        event = json.loads(text)
        validate_logprobs(event)
        if event.get("error"):
            raise ValueError(f"stream error: {event['error']}")
        if "kairyu_trace_v2" in event:
            if trace is not None:
                raise ValueError("multiple trace envelopes")
            trace = event["kairyu_trace_v2"]
        usage = event.get("usage") or usage
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or delta.get("reasoning") or ""
            finish = choice.get("finish_reason") or finish
    return {
        "choices": [
            {
                "message": {"content": content, "reasoning_content": reasoning},
                "finish_reason": finish,
            }
        ],
        "usage": usage,
        "kairyu_trace_v2": trace,
    }, done


def execute_case(
    case: dict,
    *,
    endpoint: str,
    output: Path,
    timeout: float,
    api_key: str | None = None,
    data_parallel_rank: int | None = None,
    logprobs: bool = False,
) -> dict:
    if data_parallel_rank is not None and data_parallel_rank not in (0, 1, 2):
        raise ValueError("data_parallel_rank must be 0, 1, or 2")
    if logprobs:
        case = {**case, "payload": {**case["payload"], "logprobs": True, "top_logprobs": 2}}
    name = case["name"]
    target = output / name
    target.mkdir()
    (target / "request.json").write_text(json.dumps(case["payload"], indent=2) + "\n")
    result = {
        "name": name,
        "kind": case["kind"],
        "data_parallel_rank": data_parallel_rank,
        "logprobs_requested": logprobs,
        "passed": False,
        "truncated": False,
        "started_at": datetime.now(UTC).isoformat(),
        "timeout_seconds": timeout,
        "expected_effective_effort": case.get("expected_effective_effort"),
    }
    result["messages_sha256"] = hashlib.sha256(
        json.dumps(case["payload"].get("messages"), ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    started = time.monotonic()
    headers = {"Content-Type": "application/json"}
    if case.get("trace"):
        headers["X-Kairyu-Trace"] = "1"
    if data_parallel_rank is not None:
        headers["X-data-parallel-rank"] = str(data_parallel_rank)
    (target / "request-headers.json").write_text(json.dumps(headers, indent=2) + "\n")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, json.dumps(case["payload"]).encode(), headers)
    raw = bytearray()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result["http_status"] = response.status
            result["response_headers"] = dict(response.headers)
            if case["payload"].get("stream"):
                for line in response:
                    raw.extend(line)
                    if time.monotonic() - started > timeout:
                        raise TimeoutError("total case deadline exceeded")
                    if case["kind"] == "cancel" and line.startswith(b"data:"):
                        chunk = line[5:].strip()
                        if chunk and chunk != b"[DONE]":
                            event = json.loads(chunk)
                            validate_logprobs(event)
                            if any(
                                c.get("delta", {}).get("content") for c in event.get("choices", [])
                            ):
                                result.update(
                                    passed=True,
                                    client_cancelled=True,
                                    detail=(
                                        "closed client stream after first content; "
                                        "server cleanup unverified"
                                    ),
                                )
                                break
            else:
                while chunk := response.read1(65536):
                    raw.extend(chunk)
                    if time.monotonic() - started > timeout:
                        raise TimeoutError("total case deadline exceeded")
        (target / "response.raw").write_bytes(raw)
        if case["kind"] == "cancel":
            if not result.get("client_cancelled"):
                result["detail"] = "no content arrived before stream ended"
        else:
            if case["payload"].get("stream"):
                body, done = parse_stream(raw.decode())
                result["stream_done"] = done
            else:
                body, done = json.loads(raw), True
            (target / "response.json").write_text(json.dumps(body, indent=2) + "\n")
            result.update(validate_response(case["kind"], body))
            if case.get("trace"):
                route = validate_l2_trace(
                    body,
                    expected_profile=case.get("expected_profile"),
                    expect_headless=case.get("expect_headless"),
                )
                result["route_validation"] = route
                result["observed_profile"] = route["observed_profile"]
                if not route["passed"]:
                    result.update(passed=False, detail=route["detail"])
            if not done:
                result.update(passed=False, detail="stream ended without [DONE]")
            if case.get("expect_nonthinking"):
                message = body["choices"][0]["message"]
                if message.get("reasoning_content") or message.get("reasoning"):
                    result.update(passed=False, detail="nonthinking response emitted reasoning")
    except urllib.error.HTTPError as exc:
        raw.extend(exc.read())
        result.update(http_status=exc.code, detail=str(exc), error_type=type(exc).__name__)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        result.update(
            detail=str(exc),
            error_type=type(exc).__name__,
            timed_out=isinstance(getattr(exc, "reason", exc), (TimeoutError, socket.timeout)),
        )
    finally:
        (target / "response.raw").write_bytes(raw)
        result["elapsed_seconds"] = time.monotonic() - started
        (target / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Native worker URL ending in /v1")
    parser.add_argument("--output", required=True, type=Path, help="New evidence directory")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--suite", choices=("native", "l2"), default="native")
    parser.add_argument(
        "--l2-effort-matrix",
        action="store_true",
        help="Add primary requests at omitted/high/max API effort (low already present)",
    )
    parser.add_argument("--case", action="append", default=[], help="Regex filter; repeatable")
    parser.add_argument(
        "--provenance",
        action="append",
        default=[],
        type=Path,
        help="Existing container/model attestation JSON to copy",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--data-parallel-rank",
        type=int,
        choices=(0, 1, 2),
        help="Route to one native DP rank via X-data-parallel-rank",
    )
    parser.add_argument(
        "--logprobs",
        action="store_true",
        help="Request logprobs and top_logprobs=2; reject nonfinite values",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    directory = Path(__file__).resolve().parent
    source_cases = (
        build_l2_cases(directory, effort_matrix=args.l2_effort_matrix)
        if args.suite == "l2"
        else build_cases(directory, args.model)
    )
    cases = [
        c
        for c in source_cases
        if not args.case or any(re.search(pattern, c["name"]) for pattern in args.case)
    ]
    if not cases:
        parser.error("case filter selected no cases")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "started_at": datetime.now(UTC).isoformat(),
        "endpoint": args.base_url,
        "host": platform.node(),
        "python": platform.python_version(),
        "model": "kairyu-auto-max" if args.suite == "l2" else args.model,
        "suite": args.suite,
        "l2_effort_matrix": args.l2_effort_matrix,
        "timeout_seconds": args.timeout,
        "data_parallel_rank": args.data_parallel_rank,
        "logprobs_requested": args.logprobs,
        "selected_cases": [c["name"] for c in cases],
        "files": {},
        "thinking_budget_accounting": BUDGET_ACCOUNTING,
    }
    evidence_files = ["gpu_smoke.py", "auto-max.yaml", "requirements_budget.py", "example.json"]
    if args.suite == "l2":
        evidence_files.append("requirements-quality-cases.json")
    for name in evidence_files:
        data = (directory / name).read_bytes()
        manifest["files"][name] = hashlib.sha256(data).hexdigest()
        (args.output / name).write_bytes(data)
    for index, path in enumerate(args.provenance):
        data = path.read_bytes()
        destination = f"provenance-{index}-{path.name}"
        (args.output / destination).write_bytes(data)
        manifest["files"][destination] = hashlib.sha256(data).hexdigest()
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=directory, capture_output=True, text=True
    )
    manifest["git_head"] = git.stdout.strip() if git.returncode == 0 else None
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    endpoint = args.base_url.rstrip("/") + "/chat/completions"

    def run(case):
        result = execute_case(
            case,
            endpoint=endpoint,
            output=args.output,
            timeout=args.timeout,
            api_key=os.environ.get(args.api_key_env),
            data_parallel_rank=args.data_parallel_rank,
            logprobs=args.logprobs,
        )
        results.append(result)
        summary = {"passed": all(r["passed"] for r in results), "cases": results, "complete": False}
        (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"{case['name']}: {'PASS' if result['passed'] else 'FAIL'} "
            f"({result['elapsed_seconds']:.1f}s) {result.get('detail', '')}",
            flush=True,
        )
        return result

    for case in cases:
        result = run(case)
        if case["kind"] == "tool" and result["passed"]:
            response = json.loads((args.output / case["name"] / "response.json").read_text())
            message = response["choices"][0]["message"]
            body = {
                **case["payload"],
                "tool_choice": "none",
                "messages": [
                    *case["payload"]["messages"],
                    message,
                    {
                        "role": "tool",
                        "tool_call_id": message["tool_calls"][0]["id"],
                        "content": "SMOKE_VALUE_731",
                    },
                    {"role": "user", "content": "Return exactly the value returned by lookup."},
                ],
            }
            run(
                {
                    "name": "l2-tool-result" if case.get("trace") else "native-tool-result",
                    "kind": "tool-result",
                    "payload": body,
                    "trace": case.get("trace", False),
                    "expect_headless": True,
                }
            )
    summary = {"passed": all(r["passed"] for r in results), "complete": True, "cases": results}
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
