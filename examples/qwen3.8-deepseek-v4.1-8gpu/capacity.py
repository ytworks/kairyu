"""Bounded native DS6 capacity probes; never starts/stops services.

Example (on the GPU host after startup):
  python capacity.py --container CONTAINER --tokenizer /models/tokenizer.json \
    --results-dir /tmp/ds6-capacity --cases serving retrieval-8k \
    --concurrency 1 8 16 32

Fixed-budget rows count all generated tokens, including reasoning. Retrieval
requires a completed exact-key answer. Neither substitutes for quality gates.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import yaml

HERE = Path(__file__).resolve().parent
HELPER = HERE.parent / "deepseek-v4.1-flash-8gpu/benchmark.py"
_spec = importlib.util.spec_from_file_location("v41_fixed_benchmark", HELPER)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)
_control_spec = importlib.util.spec_from_file_location("v41_capacity_control", HERE / "control.py")
control = importlib.util.module_from_spec(_control_spec)
_control_spec.loader.exec_module(control)
CASES = {
    "retrieval-8k": 8192,
    "retrieval-32k": 32768,
    "retrieval-128k": 131072,
    "retrieval-256k": 262144,
    "retrieval-near1m": 1048576 - 8704,
}


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_runtime_command():
    return yaml.safe_load((HERE / "compose.yaml").read_text())["services"]["deepseek"]["command"]


def validate_runtime(record, image):
    command = record.get("Config", {}).get("Cmd", [])

    def arg(flag, default=None):
        if flag not in command:
            return default
        index = command.index(flag) + 1
        return command[index] if index < len(command) else None

    devices = record.get("HostConfig", {}).get("DeviceRequests", [])
    ids = [value for request in devices for value in request.get("DeviceIDs", [])]
    return (
        record.get("Image") == image
        and record.get("State", {}).get("Running") is True
        and runtime_identity(record)["config_sha256"] == control._requirements_config_sha256()
        and command == expected_runtime_command()
        and arg("--tensor-parallel-size") == "2"
        and arg("--data-parallel-size") == "3"
        and arg("--pipeline-parallel-size", "1") == "1"
        and "--enable-expert-parallel" in command
        and "--speculative-config" not in command
        and sorted(ids) == list(map(str, range(6)))
    )


def runtime_identity(record):
    """Retain useful DS-only identity without recording arbitrary container env."""
    env = dict(
        entry.split("=", 1) for entry in record.get("Config", {}).get("Env", []) if "=" in entry
    )
    return {
        "container_id": record.get("Id"),
        "container_name": record.get("Name"),
        "image_id": record.get("Image"),
        "running": record.get("State", {}).get("Running"),
        "started_at": record.get("State", {}).get("StartedAt"),
        "config_sha256": env.get("KAIRYU_REQUIREMENTS_CONFIG_SHA256"),
    }


def endpoint_matches(record, endpoint):
    """Bind measurement traffic to the inspected local container's published API."""
    url = urlsplit(endpoint)
    if (
        url.scheme != "http"
        or url.hostname != "127.0.0.1"
        or url.path.rstrip("/") != "/v1"
        or url.query
        or url.fragment
        or url.username is not None
        or url.password is not None
    ):
        return False
    bindings = record.get("NetworkSettings", {}).get("Ports", {}).get("8000/tcp") or []
    return any(
        binding.get("HostIp") in {"127.0.0.1", "0.0.0.0"}
        and binding.get("HostPort") == str(url.port or 80)
        for binding in bindings
    )


def fixed_passed(result, output_tokens, raw_tokens):
    return (
        result.get("completion_tokens") == output_tokens
        and result.get("finish_reason") == "length"
        and isinstance(result.get("prompt_tokens"), int)
        and raw_tokens <= result["prompt_tokens"] <= raw_tokens + 512
    )


def retrieval_passed(result, key, raw_tokens, output_budget, context):
    actual = result.get("prompt_tokens")
    return (
        result.get("finish_reason") == "stop"
        and result.get("content", "").strip() == key
        and isinstance(actual, int)
        and raw_tokens <= actual <= raw_tokens + 512
        and actual + output_budget <= context
    )


def prompt_for(tokenizer, target, key, request_id):
    prefix = f"Request {request_id}. Read this archive and recover its key.\n"
    needle = f"\nThe archive key is {key}.\n"
    suffix = "\nReturn only the archive key, with no explanation."

    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    if count(" filler" * 100) != 100:
        raise ValueError("Pinned tokenizer does not encode filler as one token per repetition")
    repeats = target - count(prefix + needle + suffix)
    for _ in range(4):
        prompt = prefix + " filler" * (repeats // 2) + needle
        prompt += " filler" * (repeats - repeats // 2) + suffix
        actual = count(prompt)
        if actual == target:
            return prompt, actual
        repeats += target - actual
    raise ValueError("Cannot construct the requested exact raw-token prompt size")


async def normalize_reasoning_alias(lines):
    """Adapt native V4.1's reasoning field to the pinned shared collector ABI."""
    async for line in lines:
        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
            event = json.loads(line[6:])
            changed = False
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if not delta.get("reasoning_content") and delta.get("reasoning"):
                    delta["reasoning_content"] = delta["reasoning"]
                    changed = True
            if changed:
                line = "data: " + json.dumps(event)
        yield line


async def request(client, body, directory, timeout):
    directory.mkdir()
    dump(directory / "request.json", body)
    start = time.perf_counter()
    try:
        # Outer deadline bounds the full stream, unlike an idle-only read timeout.
        async with asyncio.timeout(timeout):
            async with client.stream("POST", "chat/completions", json=body) as response:
                dump(
                    directory / "http.json",
                    {"status": response.status_code, "headers": dict(response.headers)},
                )
                if response.status_code != 200:
                    (directory / "response.txt").write_bytes(await response.aread())
                    response.raise_for_status()
                with (directory / "response.sse").open("w") as raw:

                    async def lines():
                        async for line in response.aiter_lines():
                            raw.write(line + "\n")
                            raw.flush()
                            yield line

                    result = await benchmark.collect(normalize_reasoning_alias(lines()), start)
        result["transport_passed"] = True
    except Exception as error:
        result = {
            "transport_passed": False,
            "error": f"{type(error).__name__}: {error}",
            "total_ms": (time.perf_counter() - start) * 1000,
        }
    dump(directory / "response.json", result)
    return result


def summarize(samples, elapsed, concurrency):
    good = [s for s in samples if s["passed"]]
    summary = {
        "requests": len(samples),
        "successful_requests": len(good),
        "passed": len(good) == len(samples),
        "concurrency": concurrency,
        "wall_s": elapsed,
        "completion_tokens_total": sum(s["completion_tokens"] for s in good),
    }
    summary["output_tokens_per_s"] = summary["completion_tokens_total"] / elapsed
    summary["requests_per_s"] = len(good) / elapsed
    for field in ("model_ttft_ms", "content_ttft_ms", "total_ms", "tpot_ms"):
        values = [s[field] for s in good if s.get(field) is not None]
        summary[field] = {
            "p50": benchmark.percentile(values, 0.5),
            "p99": benchmark.percentile(values, 0.99),
            "samples": len(values),
        }
    return summary


async def measure(args, tokenizer):
    report = {
        "measurement_scope": "native_deepseek_six_gpu_only",
        "fixed_budget_is_completed_answer_gate": False,
        "rows": [],
        "retrieval": [],
        "passed": True,
    }
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/") + "/",
        timeout=httpx.Timeout(args.timeout, connect=10),
        limits=httpx.Limits(max_connections=max(args.concurrency)),
    ) as client:
        response = await client.get("models")
        dump(
            args.results_dir / "models.json",
            {"status": response.status_code, "body": response.text},
        )
        response.raise_for_status()
        for case in args.cases:
            concurrencies = args.concurrency if case == "serving" else [1]
            for concurrency in concurrencies:
                semaphore = asyncio.Semaphore(concurrency)
                count = args.requests if case == "serving" else 1
                # Tokenization runs before timing; requests have unique prefixes.
                work = []
                for index in range(count):
                    key, request_id = "K" + uuid.uuid4().hex.upper(), uuid.uuid4().hex
                    target = 8192 if case == "serving" else CASES[case]
                    prompt, raw_tokens = prompt_for(tokenizer, target, key, request_id)
                    body = {
                        "model": args.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "chat_template_kwargs": {"thinking": True, "reasoning_effort": args.effort},
                        "max_tokens": 256 if case == "serving" else args.retrieval_output,
                    }
                    if case == "serving":
                        body.update(min_tokens=256, ignore_eos=True)
                    work.append((index, body, key, raw_tokens))

                async def run(item, semaphore=semaphore, case=case, concurrency=concurrency):
                    index, body, key, raw_tokens = item
                    async with semaphore:
                        directory = args.results_dir / f"{case}-c{concurrency}-{index:03d}"
                        result = await request(client, body, directory, args.timeout)
                        check = (
                            fixed_passed(result, 256, raw_tokens)
                            if case == "serving"
                            else (
                                retrieval_passed(
                                    result, key, raw_tokens, args.retrieval_output, 1048576
                                )
                            )
                        )
                        result.update(
                            passed=result["transport_passed"] and check,
                            raw_input_tokens=raw_tokens,
                            expected_key=key,
                            evidence_directory=directory.name,
                        )
                        dump(directory / "result.json", result)
                        return result

                start = time.perf_counter()
                samples = await asyncio.gather(*(run(item) for item in work))
                elapsed = time.perf_counter() - start
                if case == "serving":
                    row = {
                        "measurement": "fixed_length_including_reasoning",
                        "summary": summarize(samples, elapsed, concurrency),
                        "samples": samples,
                    }
                    report["rows"].append(row)
                    print(json.dumps(row["summary"]), flush=True)
                else:
                    report["retrieval"].append({"case": case, **samples[0]})
                    print(
                        json.dumps(
                            {
                                "case": case,
                                "passed": samples[0]["passed"],
                                "usage": samples[0].get("usage"),
                            }
                        ),
                        flush=True,
                    )
                report["passed"] = report["passed"] and all(s["passed"] for s in samples)
                dump(args.results_dir / "measurements.json", report)
                if not report["passed"] and not args.continue_on_failure:
                    return 1
    return int(not report["passed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8009/v1")
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument(
        "--container", required=True, help="Running DS6 container for read-only inspect"
    )
    parser.add_argument("--tokenizer", type=Path, required=True, help="Pinned tokenizer.json")
    parser.add_argument(
        "--results-dir", type=Path, required=True, help="New, nonexistent directory"
    )
    parser.add_argument(
        "--cases", nargs="+", choices=["serving", *CASES], default=["serving", *CASES]
    )
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--requests", type=int, default=32, help="Requests per serving row")
    parser.add_argument(
        "--timeout", type=float, default=1800, help="Total deadline per request seconds"
    )
    parser.add_argument("--retrieval-output", type=int, default=8192)
    parser.add_argument("--effort", choices=["low", "high", "max"], default="high")
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()
    if (
        min(args.concurrency) < 1
        or max(args.concurrency) > 32
        or args.requests < max(args.concurrency)
        or args.requests > 128
        or not 0 < args.timeout <= 3600
        or not 1 <= args.retrieval_output <= 8192
        or len(set(args.cases)) != len(args.cases)
        or len(set(args.concurrency)) != len(args.concurrency)
    ):
        parser.error(
            "Require unique cases/concurrency, c1..32, c<=requests<=128, "
            "0<timeout<=3600, and retrieval output1..8192"
        )
    args.results_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((HERE / "example.json").read_text())
    model_manifest = json.loads(
        (HERE.parent / "deepseek-v4.1-flash-8gpu/model-manifest.json").read_text()
    )
    tokenizer_pin = next(
        row["sha256"] for row in model_manifest["files"] if row["path"] == "tokenizer.json"
    )
    inspected = json.loads(
        subprocess.check_output(["docker", "inspect", args.container], timeout=30)
    )[0]
    identity = runtime_identity(inspected)
    dump(args.results_dir / "runtime.json", identity)
    provenance = {
        "started_at": datetime.now(UTC).isoformat(),
        "container": args.container,
        "endpoint": args.base_url,
        "model": args.model,
        "topology_attested": validate_runtime(inspected, manifest["vllm"]["deepseek"]["image_id"]),
        "runtime_attestation": identity,
        "endpoint_attested": endpoint_matches(inspected, args.base_url),
        "expected_runtime_config_sha256": control._requirements_config_sha256(),
        "expected_runtime_command": expected_runtime_command(),
        "config_sha256": {
            name: digest(HERE / name)
            for name in (
                "compose.yaml",
                "kairyu.yaml",
                "example.json",
                "auto-max.yaml",
                "requirements_budget.py",
                "capacity.py",
            )
        },
        "tokenizer_sha256": digest(args.tokenizer),
        "tokenizer_attested": digest(args.tokenizer) == tokenizer_pin,
        "model_revision": model_manifest["revision"],
        "tokenizer_path": str(args.tokenizer),
        "benchmark_helper_sha256": digest(HELPER),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    dump(args.results_dir / "provenance.json", provenance)
    if not provenance["tokenizer_attested"]:
        raise SystemExit("Tokenizer does not match pinned V4.1 checkpoint manifest")
    if not provenance["topology_attested"]:
        raise SystemExit(
            "Running container image/configuration/TP2/DP3/EP6/devices do not match candidate"
        )
    if not provenance["endpoint_attested"]:
        raise SystemExit("Endpoint must be the inspected container's published 127.0.0.1 API")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    raise SystemExit(asyncio.run(measure(args, tokenizer)))


if __name__ == "__main__":
    main()
