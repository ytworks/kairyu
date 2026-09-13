#!/usr/bin/env python3
"""Native critical-ensemble serving verification (head/synthesis stream, TTFT gate)."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = json.loads((HERE / "example.json").read_text(encoding="utf-8"))

# Mandatory internal generation nodes on every primary request. The four
# Qwen candidates are four calls, independently of the two-replica placement.
_PRIMARY_INTERNAL_NODES = (
    "requirements",
    "deepseek_candidate",
    "policies",
    "answer_1",
    "answer_2",
    "answer_3",
    "answer_4",
    "review",
)
# Verification nodes every primary-profile request must trace: the audit
# verdict on the synthesis final unit (DTO-D10).
_PRIMARY_VERIFICATION_NODES = ("audit",)
# DTO-D13: the Qwen route judge selects one profile per request; every
# sample must trace the judge classification stage and exactly one profile's
# final unit. Direct routes have no head and no internal stages.
_ROUTE_FINAL_NODES: dict[str, str] = dict(SPEC["orchestration"]["profile_final_roles"])
_JUDGE_NODE = "profile_judge"


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
    os.environ.get(
        "VERIFICATION_RESULTS_ROOT",
        ENVIRONMENT_STORAGE / "verification-results",
    )
)


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




def _requirement_ids(text: str) -> tuple[str, ...]:
    """Validate the generated checklist itself, without trusting a model label."""

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Requirement JSON contains a duplicate object key")
            result[key] = value
        return result

    rows = json.loads(text, object_pairs_hook=unique_object)
    if type(rows) is not list or not rows:
        raise ValueError("Requirement output must be a nonempty JSON array")
    fields = {"id", "priority", "requirement", "acceptance_criterion", "source"}
    for index, row in enumerate(rows, 1):
        if (type(row) is not dict or set(row) != fields
                or any(type(value) is not str or not value.strip() for value in row.values())
                or row["id"] != f"R{index}" or row["priority"] not in {"minimum", "optional"}):
            raise ValueError("Requirement output violates the five-field/ID/priority contract")
    return tuple(row["id"] for row in rows)


class _PrimaryEvidence:
    """Observe the existing opt-in trace/intermediate stream without storing text."""

    def __init__(self):
        self.requirements: tuple[str, ...] = ()
        self.requirements_sha256: str | None = None
        self.audit_outputs: list[dict] = []
        self.events: list[dict] = []

    def observe(self, chunk: dict) -> None:
        if "kairyu_trace_v2" in chunk:
            self.events = chunk["kairyu_trace_v2"].get("events", [])
        for choice in chunk.get("choices", []):
            text = (choice.get("delta") or {}).get("reasoning_content")
            if not isinstance(text, str):
                continue
            header = re.match(r"\A### (requirements|audit) — attempt ([1-9][0-9]*)\n", text)
            if header is None:
                continue
            if not text.endswith("\n\n---\n\n"):
                raise ValueError("completed intermediate output has no expected boundary")
            body = text[:-7]
            _prefix, separator, output = body.rpartition("\n#### Stage output\n\n")
            if not separator:
                raise ValueError("completed intermediate output is missing its stage output")
            if header[1] == "requirements":
                if self.requirements:
                    raise ValueError("Requirement extractor unexpectedly ran more than once")
                self.requirements = _requirement_ids(output)
                self.requirements_sha256 = hashlib.sha256(output.encode()).hexdigest()
            else:
                first_line = output.splitlines()[0] if output.splitlines() else ""
                if first_line not in {"PASS", "FAIL"}:
                    raise ValueError("audit output has no conclusive first-line verdict")
                mentioned = set(re.findall(r"\bR[1-9][0-9]*\b", output))
                if not self.requirements or not set(self.requirements) <= mentioned:
                    raise ValueError("audit output omits a Requirement ID")
                index_match = re.search(r"^- Choice: `([0-9]+)`$", _prefix, re.MULTILINE)
                self.audit_outputs.append({
                    "choice_index": int(index_match[1]) if index_match else 0,
                    "attempt": int(header[2]) - 1,
                    "verdict": first_line,
                    "requirement_ids": list(self.requirements),
                    "sha256": hashlib.sha256(output.encode()).hexdigest(),
                })

    def validate(self, *, effort: str, choices: int) -> dict:
        if not self.requirements:
            raise ValueError("no generated Requirement JSON was observed")
        successful = [event for event in self.events if event.get("status") == "success"]
        for node in _PRIMARY_INTERNAL_NODES:
            expected_worker = "tier1" if node.startswith("answer_") else "tier2"
            if not any(event.get("node") == node and event.get("kind") == "generation"
                       and event.get("worker") == expected_worker for event in successful):
                raise ValueError(f"missing successful {node} generation on {expected_worker}")
        expected_effort = "max" if effort == "max" else "high"
        extractor = next(event for event in successful if event.get("node") == "requirements")
        if extractor.get("detail", {}).get("reasoning_effort") != expected_effort:
            raise ValueError("Requirement trace lacks its effective high-floor/max effort")
        for index in range(choices):
            audits = [event for event in successful
                      if event.get("node") == "audit" and event.get("kind") == "verification"
                      and (event.get("detail", {}).get("choice_index") or 0) == index]
            if not audits or audits[-1].get("worker") != "tier2":
                raise ValueError("a final choice has no independent DeepSeek audit")
            latest = audits[-1]
            detail = latest.get("detail", {})
            if detail.get("inconclusive") is not False:
                raise ValueError("final audit is inconclusive")
            if not (detail.get("pass") is True or (
                detail.get("refinement_exhausted") is True and latest.get("attempt") == 2
            )):
                raise ValueError("final failed audit has not completed the configured repairs")
            outputs = [row for row in self.audit_outputs
                       if row["choice_index"] == index and row["attempt"] == latest["attempt"]]
            if len(outputs) != 1 or (outputs[0]["verdict"] == "PASS") != detail.get("pass"):
                raise ValueError("audit text and trusted choice/attempt trace disagree")
        return {
            "requirement_ids": list(self.requirements),
            "requirements_sha256": self.requirements_sha256,
            "effective_requirement_effort": expected_effort,
            "audits": self.audit_outputs,
            "judgment_correctness_claimed": False,
        }


def functional_primary(run_dir: Path) -> int:
    """Observe generated checklists and all-choice audits on the real primary DAG."""

    async def probes():
        from verification.l1.performance import serving_bench

        model = SPEC["orchestration"]["auto_max_model"]
        base_url = f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}/v1/"
        async with httpx.AsyncClient(base_url=base_url, timeout=900) as client:
            for effort, choices in (("low", 1), ("max", 2)):
                evidence = _PrimaryEvidence()
                async with asyncio.timeout(900):
                    result = await serving_bench.run_one(
                        client, model,
                        "Explain why 2 + 3 = 5 in exactly two short Japanese sentences. "
                        "Do not use a list, heading or code block.",
                        32768, temperature=1.0, top_p=1.0,
                        reasoning_effort=effort, n=choices,
                        request_trace=True, capture_response=True, on_chunk=evidence.observe,
                    )
                if (not result.stream_complete or result.finish_reasons != ("stop",) * choices
                        or not result.response_text or result.trace_status != "valid"):
                    raise ValueError("functional primary did not complete every final choice")
                report = evidence.validate(effort=effort, choices=choices)
                if not any(stage.node == _JUDGE_NODE and stage.kind == "classification"
                           and stage.status == "success" for stage in result.trace_stages):
                    raise ValueError("functional primary request did not run its real judge")
                report.update({"request_effort": effort, "choices": choices,
                               "total_ms": result.total_s * 1000,
                               "stream_complete": result.stream_complete,
                               "finish_reasons": list(result.finish_reasons)})
                (run_dir / f"functional-{effort}-n{choices}.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
        return 0

    with _primary_profile_override(run_dir):
        return asyncio.run(probes())


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
        # Put the run/row identity before the repeated body so another
        # concurrency row cannot become a full-prefix-cache microbenchmark.
        prompt = f"Run {namespace}, case {request}: " + " ".join(words)
        if response_instruction:
            prompt += "\n\n" + response_instruction
        rows.append({"conversations": [{"from": "human", "value": prompt}]})
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def _row_summary(row_dir: Path) -> dict | None:
    artifacts = list(row_dir.glob("*-serving.json"))
    if len(artifacts) != 1:
        return None
    try:
        result = json.loads(artifacts[0].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    summary = result.get("summary")
    return summary if isinstance(summary, dict) else None


def _sample_route(sample: dict) -> str | None:
    """The profile a traced sample ran: the unique route whose final unit
    succeeded as a publisher generation stage; None when ambiguous."""

    stages = sample.get("trace", {}).get("stages", [])
    routes = [
        profile
        for profile, node in _ROUTE_FINAL_NODES.items()
        if any(
            stage.get("node") == node
            and stage.get("role") == "publisher"
            and stage.get("kind") == "generation"
            and stage.get("status") == "success"
            for stage in stages
        )
    ]
    return routes[0] if len(routes) == 1 else None


def _route_report(samples: list[dict]) -> dict:
    """Route distribution, per-route TTFT p50, and judge latency for one row."""

    by_route: dict[str, list[float]] = {}
    requests_by_route: dict[str, int] = {}
    judge_ms: list[float] = []
    for sample in samples:
        route = _sample_route(sample) or "unresolved"
        requests_by_route[route] = requests_by_route.get(route, 0) + 1
        by_route.setdefault(route, [])
        ttft = sample.get("ttft_ms")
        if isinstance(ttft, (int, float)):
            by_route[route].append(float(ttft))
        for stage in sample.get("trace", {}).get("stages", []):
            if stage.get("node") == _JUDGE_NODE and isinstance(
                stage.get("total_ms"), (int, float)
            ):
                judge_ms.append(float(stage["total_ms"]))

    def p50(values: list[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        return round(ordered[(len(ordered) + 1) // 2 - 1], 2)

    return {
        "routes": {
            route: {"requests": requests_by_route[route], "ttft_p50_ms": p50(values)}
            for route, values in sorted(by_route.items())
        },
        "judge_total_ms_p50": p50(judge_ms),
        "judged_samples": len(judge_ms),
    }


def _validate_serving_row(
    row_dir: Path,
    requests: int,
    output_tokens: int,
    *,
    expected_route: str | None = None,
    expected_role: str = "direct",
    expected_kind: str = "generation",
    public_tokens: bool = False,
    require_head: bool = False,
    expected_generation_nodes: tuple[str, ...] = (),
    expected_verification_nodes: tuple[str, ...] = (),
    judged_routes: bool = False,
    require_primary: bool = False,
    expected_choices: int = 1,
) -> int:
    artifacts = list(row_dir.glob("*-serving.json"))
    if len(artifacts) != 1:
        print(f"serving row produced {len(artifacts)} result files", file=sys.stderr)
        return 1
    try:
        result = json.loads(artifacts[0].read_text(encoding="utf-8"))
        summary = result["summary"]
        samples = result["samples"]
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"invalid serving result: {error}", file=sys.stderr)
        return 1
    if public_tokens:
        complete = (
            summary.get("requests") == requests
            and isinstance(summary.get("public_completion_tokens_total"), int)
            and summary["public_completion_tokens_total"] > 0
            and isinstance(summary.get("public_output_tokens_per_s"), (int, float))
            and summary["public_output_tokens_per_s"] > 0
            and len(samples) == requests
            and all(
                isinstance(sample.get("public_completion_tokens"), int)
                and sample["public_completion_tokens"] > 0
                and sample.get("stream_complete") is True
                and sample.get("finish_reasons") == ["stop"] * expected_choices
                for sample in samples
            )
        )
    else:
        expected_total = requests * output_tokens
        complete = (
            summary.get("requests") == requests
            and summary.get("completion_tokens_total") == expected_total
            and isinstance(summary.get("output_tokens_per_s"), (int, float))
            and summary["output_tokens_per_s"] > 0
            and len(samples) == requests
            and all(sample.get("completion_tokens") == output_tokens for sample in samples)
        )
    if complete and expected_route is not None:
        def stage_ok(sample: dict) -> bool:
            stages = sample.get("trace", {}).get("stages", [])
            if sample.get("trace", {}).get("status") != "valid":
                return False
            if judged_routes:
                # DTO-D13: the judge stage must be traced and exactly one
                # profile's final unit must have published. Only the primary
                # (ensemble) route carries the head, internal, and audit
                # stage contracts below.
                if not any(
                    stage.get("node") == _JUDGE_NODE
                    and stage.get("kind") == "classification"
                    and stage.get("status") == "success"
                    for stage in stages
                ):
                    return False
                route = _sample_route(sample)
                if route is None:
                    return False
                if route != "primary":
                    return not require_primary
            if not any(
                stage.get("node") == expected_route
                and stage.get("role") == expected_role
                and stage.get("kind") == expected_kind
                and stage.get("status") == "success"
                for stage in stages
            ):
                return False
            if require_head and not any(
                stage.get("node") == "head"
                and stage.get("kind") == "generation"
                and stage.get("status") == "success"
                for stage in stages
            ):
                return False
            if any(
                not any(
                    stage.get("node") == node
                    and stage.get("kind") == "generation"
                    and stage.get("status") == "success"
                    for stage in stages
                )
                for node in expected_generation_nodes
            ):
                return False
            if any(
                not any(
                    stage.get("node") == node
                    and stage.get("kind") == "verification"
                    and stage.get("status") == "success"
                    for stage in stages
                )
                for node in expected_verification_nodes
            ):
                return False
            if require_primary:
                for index in range(expected_choices):
                    audits = [stage for stage in stages
                              if stage.get("node") in expected_verification_nodes
                              and (stage.get("choice_index") or 0) == index
                              and stage.get("kind") == "verification"
                              and stage.get("status") == "success"]
                    if not audits:
                        return False
                    latest = audits[-1]
                    if latest.get("verification_inconclusive") is not False:
                        return False
                    if not (latest.get("verification_pass") is True or (
                        latest.get("refinement_exhausted") is True and latest.get("attempt") == 2
                    )):
                        return False
            return True

        complete = all(stage_ok(sample) for sample in samples)
        if judged_routes:
            report = _route_report(samples)
            (row_dir / "routes.json").write_text(
                json.dumps({"schema_version": 1, **report}, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            print(f"{row_dir.name}: routes {json.dumps(report['routes'], sort_keys=True)}")
    if not complete:
        print("serving row did not produce complete response and stage evidence", file=sys.stderr)
        return 1
    return 0


def _serving(
    model: str,
    run_dir: Path,
    *,
    tensor_parallel: int,
    replicas: int,
    expected_route: str | None = None,
    expected_role: str = "direct",
    expected_kind: str = "generation",
    warmup_requests: int | None = None,
    natural_completion: bool = False,
    require_head: bool = False,
    expected_generation_nodes: tuple[str, ...] = (),
    expected_verification_nodes: tuple[str, ...] = (),
    judged_routes: bool = False,
) -> int:
    config = SPEC["verification"]["serving"]
    requests = int(config["requests_per_concurrency"])
    output_tokens = int(config["output_tokens"])
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_tokens = int(config["prompt_tokens_approx"])
    warmup_requests = warmup_requests or max(1, replicas)
    warmup_dataset = run_dir / "warmup-8k.json"
    response_instruction = (
        "Synthesize a useful final answer of approximately 256 output tokens. "
        "Return only that answer; do not expose candidates or private reasoning."
        if natural_completion
        else ""
    )
    _serving_dataset(
        warmup_dataset,
        warmup_requests,
        prompt_tokens,
        namespace=f"{run_dir.parent.name}-{run_dir.name}-warmup",
        response_instruction=response_instruction,
    )
    request_max_tokens = (
        int(config["auto_max_combined_max_tokens"])
        if natural_completion
        else output_tokens
    )
    fixed_output_args = (
        []
        if natural_completion
        else ["--min-tokens", "32", "--ignore-eos"]
    )
    public_tokenizer_args = (
        [
            "--public-tokenizer-url",
            f"http://127.0.0.1:{os.environ.get('DEEPSEEK_L1_PORT', 8005)}/tokenize",
            "--public-tokenizer-model",
            SPEC["models"]["tier2"]["served_name"],
        ]
        if natural_completion
        else []
    )
    warmup_code = _run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "verification/l1/performance/serving_bench.py"),
            "--base-url",
            f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}/v1",
            "--model",
            model,
            "--dataset",
            str(warmup_dataset),
            "--num-requests",
            str(warmup_requests),
            "--concurrency",
            str(warmup_requests),
            "--max-tokens",
            str(request_max_tokens if natural_completion else 32),
            *fixed_output_args,
            "--temperature",
            "0.0" if natural_completion else "1.0",
            "--seed",
            "0",
            "--timeout",
            "86400",
            "--results-dir",
            str(run_dir / "warmup"),
            "--tensor-parallel",
            str(tensor_parallel),
            "--dp-replicas",
            str(replicas),
            *public_tokenizer_args,
            *(["--stage-trace"] if expected_route is not None else []),
        ],
        log=run_dir / "warmup.log",
        check=False,
    )
    if warmup_code:
        return warmup_code
    for concurrency in config["concurrency"]:
        dataset = run_dir / f"serving-c{concurrency}-8k.json"
        _serving_dataset(
            dataset,
            requests,
            prompt_tokens,
            namespace=f"{run_dir.parent.name}-{run_dir.name}-c{concurrency}",
            response_instruction=response_instruction,
        )
        row_dir = run_dir / f"serving-c{concurrency}"
        code = _run(
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "verification/l1/performance/serving_bench.py"),
                "--base-url",
                f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}/v1",
                "--model",
                model,
                "--dataset",
                str(dataset),
                "--num-requests",
                str(requests),
                "--concurrency",
                str(concurrency),
                "--max-tokens",
                str(request_max_tokens),
                *(
                    []
                    if natural_completion
                    else ["--min-tokens", str(output_tokens), "--ignore-eos"]
                ),
                "--temperature",
                "0.0" if natural_completion else "1.0",
                "--seed",
                "0",
                "--timeout",
                "86400",
                "--results-dir",
                str(row_dir),
                "--tensor-parallel",
                str(tensor_parallel),
                "--dp-replicas",
                str(replicas),
                *public_tokenizer_args,
                *(["--stage-trace"] if expected_route is not None else []),
            ],
            log=run_dir / f"serving-c{concurrency}.log",
            check=False,
        )
        if code == 0:
            code = _validate_serving_row(
                row_dir,
                requests,
                output_tokens,
                expected_route=expected_route,
                expected_role=expected_role,
                expected_kind=expected_kind,
                public_tokens=natural_completion,
                require_head=require_head,
                expected_generation_nodes=expected_generation_nodes,
                expected_verification_nodes=expected_verification_nodes,
                judged_routes=judged_routes,
            )
        if code:
            return code
    return 0


def serving_auto_max(run_dir: Path) -> int:
    """Generic-workload regression envelope: every request traces the route
    judge and exactly one profile's final unit; requests the judge sends to
    the primary ensemble keep the head/synthesis split public stream and
    the audit verdict (DTO-D13)."""

    return _serving(
        SPEC["orchestration"]["auto_max_model"],
        run_dir,
        tensor_parallel=int(SPEC["allocation"]["tier2"]["tensor_parallel_size"]),
        replicas=1,
        expected_route="synthesis",
        expected_role="publisher",
        expected_kind="generation",
        warmup_requests=4,
        natural_completion=True,
        require_head=True,
        expected_generation_nodes=_PRIMARY_INTERNAL_NODES,
        expected_verification_nodes=_PRIMARY_VERIFICATION_NODES,
        judged_routes=True,
    )

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
    """Deterministic self-contained Python tasks (~1.5K prompt tokens).

    A namespaced context paragraph precedes each task so a later concurrency
    row cannot become a full-prefix-cache benchmark, mirroring
    ``_serving_dataset``.
    """

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


def _bench_row(
    *,
    base_url: str,
    model: str,
    dataset: Path,
    requests: int,
    concurrency: int,
    max_tokens: int,
    results_dir: Path,
    log: Path,
    tensor_parallel: int,
    replicas: int,
    stage_trace: bool,
    public_tokenizer: bool,
) -> int:
    public_tokenizer_args = (
        [
            "--public-tokenizer-url",
            f"http://127.0.0.1:{os.environ.get('DEEPSEEK_L1_PORT', 8005)}/tokenize",
            "--public-tokenizer-model",
            SPEC["models"]["tier2"]["served_name"],
        ]
        if public_tokenizer
        else []
    )
    return _run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "verification/l1/performance/serving_bench.py"),
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
            "1.0",
            "--top-p",
            "1.0",
            "--seed",
            "0",
            "--timeout",
            "86400",
            "--results-dir",
            str(results_dir),
            "--tensor-parallel",
            str(tensor_parallel),
            "--dp-replicas",
            str(replicas),
            *public_tokenizer_args,
            *(["--stage-trace"] if stage_trace else []),
        ],
        log=log,
        check=False,
    )


def _control_module():
    module_spec = importlib.util.spec_from_file_location("tiered_control", HERE / "control.py")
    assert module_spec is not None and module_spec.loader is not None
    control = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(control)
    return control


@contextmanager
def _primary_profile_override(run_dir: Path):
    """Temporarily clone the primary DAG behind every real judge label."""

    from kairyu.dsl.loader import load_spec

    run_dir.mkdir(parents=True, exist_ok=True)
    source = HERE / "auto-max.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    for profile in config["profiles"]:
        profile["roles"] = copy.deepcopy(config["roles"])
    override = run_dir / "primary-comparison.yaml"
    override.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    load_spec(override)
    control = _control_module()
    previous = os.environ.get("KAIRYU_ORCHESTRATOR_SPEC_PATH")
    evidence = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "override_sha256": hashlib.sha256(override.read_bytes()).hexdigest(),
        "override_path": str(override.resolve()),
        "method": "clone primary roles into every named profile; retain actual serial judge",
        "restored": False,
    }
    evidence_path = run_dir / "primary-override.json"
    command = ["up", "--detach", "--no-deps", "--force-recreate", "--wait",
               "--wait-timeout", "120", "kairyu"]
    try:
        os.environ["KAIRYU_ORCHESTRATOR_SPEC_PATH"] = str(override.resolve())
        control._compose(command)
        api_url = f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}"
        policy = control._json_url(f"{api_url}/routing")["models"][
            SPEC["orchestration"]["auto_max_model"]
        ]
        expected_names = [role["name"] for role in config["roles"]]
        if [role.get("name") for role in policy["roles"]] != expected_names:
            raise RuntimeError("verification gateway does not serve the primary DAG")
        if set(policy["profiles"]) != {profile["name"] for profile in config["profiles"]}:
            raise RuntimeError("verification gateway profile inventory differs")
        if any([role.get("name") for role in roles] != expected_names
               for roles in policy["profiles"].values()):
            raise RuntimeError("a verification judge choice still selects a direct DAG")
        if policy["profile_judge"] != config["profile_judge"]:
            # Routing exposes defaults as well as the configured fields.
            if any(policy["profile_judge"].get(key) != value
                   for key, value in config["profile_judge"].items()):
                raise RuntimeError("verification did not retain the configured real judge")
        evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
        yield
    finally:
        if previous is None:
            os.environ.pop("KAIRYU_ORCHESTRATOR_SPEC_PATH", None)
        else:
            os.environ["KAIRYU_ORCHESTRATOR_SPEC_PATH"] = previous
        control._compose(command)
        evidence["restored"] = True
        evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")


def serving_auto_max_coding(run_dir: Path) -> int:
    """Mandatory primary matrix, with the real serial judge and fresh L1 rows."""

    with _primary_profile_override(run_dir):
        return _coding_matrix(run_dir, require_primary=True)


def serving_natural_coding(run_dir: Path) -> int:
    """Record natural routing separately; it cannot satisfy the primary gate."""

    return _coding_matrix(run_dir, require_primary=False)


def _coding_matrix(run_dir: Path, *, require_primary: bool) -> int:
    config = SPEC["verification"]["coding"]
    requests = int(config["requests_per_concurrency"])
    multiplier = float(config["ttft_multiplier_vs_deepseek_direct"])
    run_dir.mkdir(parents=True, exist_ok=True)
    api_url = f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}/v1"
    deepseek_url = f"http://127.0.0.1:{os.environ.get('DEEPSEEK_L1_PORT', 8005)}/v1"
    model = SPEC["orchestration"]["auto_max_model"]
    max_tokens = int(config["auto_max_combined_max_tokens"])
    parallel = int(SPEC["allocation"]["tier2"]["tensor_parallel_size"])
    warmup_dataset = run_dir / "warmup-coding.json"
    _coding_dataset(warmup_dataset, 4, namespace=f"{run_dir.name}-warmup")
    code = _bench_row(
        base_url=api_url, model=model, dataset=warmup_dataset, requests=4,
        concurrency=4, max_tokens=max_tokens, results_dir=run_dir / "warmup",
        log=run_dir / "warmup.log", tensor_parallel=parallel, replicas=1,
        stage_trace=True, public_tokenizer=True,
    )
    if code:
        return code
    gates = {}
    for concurrency in config["concurrency"]:
        dataset = run_dir / f"coding-c{concurrency}.json"
        _coding_dataset(dataset, requests,
                        namespace=f"{run_dir.parent.name}-{run_dir.name}-c{concurrency}")
        row_dir = run_dir / f"coding-c{concurrency}"
        code = _bench_row(
            base_url=api_url, model=model, dataset=dataset, requests=requests,
            concurrency=concurrency, max_tokens=max_tokens, results_dir=row_dir,
            log=run_dir / f"coding-c{concurrency}.log", tensor_parallel=parallel,
            replicas=1, stage_trace=True, public_tokenizer=True,
        )
        if code == 0:
            code = _validate_serving_row(
                row_dir, requests, 0, expected_route="synthesis", expected_role="publisher",
                public_tokens=True, require_head=True,
                expected_generation_nodes=_PRIMARY_INTERNAL_NODES,
                expected_verification_nodes=_PRIMARY_VERIFICATION_NODES,
                judged_routes=True, require_primary=require_primary,
            )
        if code:
            return code
        routes = _row_routes(row_dir)
        if not require_primary:
            gates[str(concurrency)] = {
                "status": "natural_routing_diagnostic", "passed": None,
                "routes": routes, "satisfies_primary_gate": False,
            }
            _write_ttft_gate(run_dir, gates)
            continue
        direct_dir = run_dir / f"deepseek-direct-c{concurrency}"
        code = _bench_row(
            base_url=deepseek_url, model=SPEC["models"]["tier2"]["served_name"],
            dataset=dataset, requests=requests, concurrency=concurrency,
            max_tokens=max_tokens, results_dir=direct_dir,
            log=run_dir / f"deepseek-direct-c{concurrency}.log",
            tensor_parallel=parallel, replicas=1, stage_trace=False, public_tokenizer=True,
        )
        if code == 0:
            code = _validate_serving_row(direct_dir, requests, 0, public_tokens=True)
        if code:
            print("fresh native baseline did not complete every answer", file=sys.stderr)
            return code
        product = _row_summary(row_dir)
        direct = _row_summary(direct_dir)
        product_ttft = product.get("ttft_p50_ms") if isinstance(product, dict) else None
        direct_ttft = direct.get("ttft_p50_ms") if isinstance(direct, dict) else None
        if (type(product_ttft) not in (int, float)
                or type(direct_ttft) not in (int, float)
                or not math.isfinite(product_ttft) or product_ttft <= 0
                or not math.isfinite(direct_ttft) or direct_ttft <= 0
                or routes is None or routes.get("judged_samples") != requests
                or routes.get("routes", {}).get("primary", {}).get("requests") != requests):
            print("mandatory primary TTFT gate is missing complete measurements", file=sys.stderr)
            return 1
        passed = product_ttft <= multiplier * direct_ttft
        gates[str(concurrency)] = {
            "product_semantic_ttft_p50_ms": product_ttft,
            "deepseek_direct_ttft_p50_ms": direct_ttft,
            "denominator_source": "fresh_completed_native_six_gpu",
            "completed_product_requests": requests,
            "completed_baseline_requests": requests,
            "multiplier": multiplier, "passed": passed, "status": "mandatory_primary",
            "routes": routes["routes"], "judge_total_ms_p50": routes["judge_total_ms_p50"],
        }
        _write_ttft_gate(run_dir, gates)
        if not passed:
            return 1
    return 0


def _write_ttft_gate(run_dir: Path, gates: dict) -> None:
    (run_dir / "ttft-gate.json").write_text(
        json.dumps(
            {"schema_version": 1, "gates": gates},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _row_routes(row_dir: Path) -> dict | None:
    """The per-row route report written by ``_validate_serving_row``."""

    path = row_dir / "routes.json"
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return report if isinstance(report, dict) else None



def _served_config_sha256() -> str:
    digest = hashlib.sha256()
    for name in (
        "example.json",
        "compose.yaml",
        "kairyu.yaml",
        "auto-max.yaml",
        "router.json",
        "qwen3.8-chat.jinja",
        "sandbox/Dockerfile",
        "sandbox/runner.py",
    ):
        path = HERE / name
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


_VERIFICATIONS = {
    "functional-primary": (
        functional_primary, "strict Requirement JSON, native effort and per-choice audit",
    ),
    "serving-auto-max": (
        serving_auto_max,
        "generic-workload product DAG serving matrix (head/synthesis stream)",
    ),
    "serving-auto-max-coding": (
        serving_auto_max_coding,
        "mandatory primary+real-judge TTFT <= 2x fresh completed native six-GPU baseline",
    ),
    "serving-natural-coding": (serving_natural_coding, "natural coding routes, diagnostic only"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verification", choices=(*_VERIFICATIONS, "list"))
    parser.add_argument("--run-id")
    parser.add_argument("--no-start", action="store_true")
    args = parser.parse_args()
    if args.verification == "list":
        for name, (_, description) in _VERIFICATIONS.items():
            print(f"{name}  {description}")
        return

    _ensure_environment(args.no_start)
    run_dir = RESULTS_ROOT / (args.run_id or _run_id())
    run_dir.mkdir(parents=True, exist_ok=True)
    target, _ = _VERIFICATIONS[args.verification]
    target_dir = run_dir / args.verification
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
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        code = target(target_dir)
    except KeyboardInterrupt:
        print(f"{args.verification} interrupted", file=sys.stderr)
        code = 130
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
