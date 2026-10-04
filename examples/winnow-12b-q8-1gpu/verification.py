#!/usr/bin/env python3
"""GPU verification gates for the Winnow-12B (llama.cpp) environment."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = json.loads((HERE / "example.json").read_text(encoding="utf-8"))
RESULTS_ROOT = ROOT / "verification/results/examples" / SPEC["environment"]
RUNTIME = SPEC["runtime"]
REPLICAS = SPEC["replicas"]
MODEL = SPEC["model"]["served_name"]
SYSTEMONE_MODEL = "winnow-12b-systemone"
PYTHON = str(ROOT / ".venv/bin/python")
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}
TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current local time for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _api() -> str:
    return f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}"


def _replica_url(replica: dict) -> str:
    return f"http://127.0.0.1:{replica['host_port']}"


def _http(method: str, url: str, payload: object | None = None, timeout: float = 900.0):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace")


def _json(method: str, url: str, payload: object | None = None) -> tuple[int, object]:
    status, body = _http(method, url, payload)
    try:
        return status, json.loads(body)
    except ValueError:
        return status, body


def _run(command: list[str], *, log: Path) -> int:
    print("+ " + " ".join(command[:6]) + (" ..." if len(command) > 6 else ""), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        return subprocess.run(
            command, cwd=ROOT, text=True, stdout=stream, stderr=subprocess.STDOUT
        ).returncode


def _report(run_dir: Path, gate: str, cases: dict[str, str | None]) -> int:
    failed = {name: error for name, error in cases.items() if error}
    (run_dir / f"{gate}.json").write_text(
        json.dumps({"gate": gate, "cases": cases, "passed": not failed}, indent=2) + "\n",
        encoding="utf-8",
    )
    for name, error in cases.items():
        verdict = "PASS" if error is None else f"FAIL: {error}"
        print(f"{gate}/{name} {verdict}")
    return 1 if failed else 0


def attest(run_dir: Path) -> int:
    """The L1 contract Kairyu relies on (LCP-D5), read from each replica."""

    cases: dict[str, str | None] = {}
    expected = RUNTIME["sampling_defaults"]
    for replica in REPLICAS:
        url = _replica_url(replica)
        status, _ = _http("GET", f"{url}/health", timeout=5)
        problems = [] if status == 200 else [f"/health {status}"]
        status, props = _json("GET", f"{url}/props")
        if status != 200 or not isinstance(props, dict):
            cases[replica["service"]] = f"/props {status}"
            continue
        settings = props.get("default_generation_settings", {})
        params = settings.get("params", {})
        commit = str(props.get("build_info", "")).rpartition("-")[2]
        if not commit or not RUNTIME["llama_cpp_commit"].startswith(commit):
            problems.append(f"build_info {props.get('build_info')!r}")
        if props.get("total_slots") != RUNTIME["chat_slots"]:
            problems.append(f"total_slots {props.get('total_slots')}")
        if settings.get("n_ctx") != RUNTIME["slot_context_tokens"]:
            problems.append(f"slot n_ctx {settings.get('n_ctx')}")
        for name, value in expected.items():
            if abs(float(params.get(name, float("nan"))) - value) > 1e-6:
                problems.append(f"default {name} {params.get(name)}")
        if props.get("chat_template_caps", {}).get("supports_tool_calls") is not True:
            problems.append("chat template without tool calls")
        if props.get("modalities", {}).get("vision") is not True:
            problems.append("no vision projector")
        (run_dir / f"props-{replica['service']}.json").write_text(
            json.dumps(props, indent=2) + "\n", encoding="utf-8"
        )
        cases[replica["service"]] = "; ".join(problems) or None
    status, models = _json("GET", f"{_api()}/v1/models")
    served = (
        {row.get("id") for row in models.get("data", [])} if isinstance(models, dict) else set()
    )
    cases["kairyu_models"] = None if MODEL in served else f"/v1/models {status}: {sorted(served)}"
    return _report(run_dir, "attest", cases)


def contract(run_dir: Path) -> int:
    """Run l1.correctness.llamacpp_upstream_contract on every replica."""

    cases: dict[str, str | None] = {}
    for replica in REPLICAS:
        name = replica["service"]
        code = _run(
            [
                PYTHON,
                str(ROOT / "verification/l1/correctness/llamacpp_upstream_contract.py"),
                "--base-url",
                _replica_url(replica),
                "--model",
                MODEL,
                "--expect-commit",
                RUNTIME["llama_cpp_commit"][:7],
                "--expect-slots",
                str(RUNTIME["chat_slots"]),
                "--expect-ctx",
                str(RUNTIME["slot_context_tokens"]),
                "--tool-max-tokens",
                "512",
                "--output",
                str(run_dir / f"contract-{name}.json"),
                "--assert-gate",
            ],
            log=run_dir / f"contract-{name}.log",
        )
        cases[name] = None if code == 0 else f"exit {code} (see contract-{name}.log)"
    return _report(run_dir, "contract", cases)


def _serving_dataset(path: Path, requests: int, approximate_tokens: int) -> None:
    words = ("river", "stone", "cloud", "field", "signal", "lantern", "harbor", "orbit")
    rows = []
    for request in range(requests):
        text = " ".join(
            words[(request * 3 + i * 5) % len(words)] for i in range(approximate_tokens)
        )
        prompt = f"Case {request}: continue this list. {text}"
        rows.append({"conversations": [{"from": "human", "value": prompt}]})
    path.write_text(json.dumps(rows), encoding="utf-8")


def serving(run_dir: Path) -> int:
    """TTFT/throughput through Kairyu. ``min_tokens`` fails closed on
    llama.cpp, so the fixed output length comes from ``ignore_eos``."""

    config = SPEC["verification"]["serving"]
    requests = int(config["requests_per_concurrency"])
    output_tokens = int(config["output_tokens"])
    dataset = run_dir / "serving.json"
    _serving_dataset(dataset, requests, int(config["prompt_tokens_approx"]))
    cases: dict[str, str | None] = {}
    for concurrency in config["concurrency"]:
        row_dir = run_dir / f"serving-c{concurrency}"
        code = _run(
            [
                PYTHON,
                str(ROOT / "verification/l1/performance/serving_bench.py"),
                "--base-url",
                f"{_api()}/v1",
                "--model",
                MODEL,
                "--dataset",
                str(dataset),
                "--num-requests",
                str(requests),
                "--concurrency",
                str(concurrency),
                "--max-tokens",
                str(output_tokens),
                "--ignore-eos",
                "--temperature",
                "1.0",
                "--seed",
                "0",
                "--timeout",
                "3600",
                "--results-dir",
                str(row_dir),
                "--tensor-parallel",
                "1",
                "--dp-replicas",
                str(len(REPLICAS)),
            ],
            log=run_dir / f"serving-c{concurrency}.log",
        )
        error = None if code == 0 else f"exit {code}"
        if error is None:
            error = _serving_row_error(row_dir, requests, output_tokens)
        cases[f"c{concurrency}"] = error
    return _report(run_dir, "serving", cases)


def _serving_row_error(row_dir: Path, requests: int, output_tokens: int) -> str | None:
    artifacts = list(row_dir.glob("*-serving.json"))
    if len(artifacts) != 1:
        return f"{len(artifacts)} result files"
    result = json.loads(artifacts[0].read_text(encoding="utf-8"))
    summary, samples = result["summary"], result["samples"]
    complete = (
        summary.get("requests") == requests
        and summary.get("completion_tokens_total") == requests * output_tokens
        and len(samples) == requests
        and all(sample.get("completion_tokens") == output_tokens for sample in samples)
    )
    return None if complete else f"incomplete evidence: {summary}"


def _chat(payload: dict) -> tuple[int, object]:
    return _json("POST", f"{_api()}/v1/chat/completions", {"model": MODEL, **payload})


def _tool_call_error(status: int, body: object, name: str) -> str | None:
    if status != 200 or not isinstance(body, dict):
        return f"HTTP {status}: {str(body)[:200]}"
    message = body["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    if not calls or calls[0]["function"]["name"] != name:
        return f"expected a {name} call, got {json.dumps(message)[:300]}"
    try:
        arguments = json.loads(calls[0]["function"]["arguments"])
    except (TypeError, ValueError):
        return "tool arguments are not JSON"
    return None if isinstance(arguments.get("city"), str) else f"arguments {arguments}"


def _placement_rows() -> list[dict]:
    sys.path.insert(0, str(HERE))
    import control

    path = control.placement_log()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def tool_calling(run_dir: Path) -> int:
    """OpenAI tool calls through Kairyu L3 (auto, named, result turn, stream)."""

    cases: dict[str, str | None] = {}
    ask = [{"role": "user", "content": "What is the weather in Tokyo right now? Use a tool."}]
    status, body = _chat({"messages": ask, "tools": [WEATHER_TOOL, TIME_TOOL], "max_tokens": 512})
    cases["auto"] = _tool_call_error(status, body, "get_weather")
    status, named = _chat(
        {
            "messages": [{"role": "user", "content": "Tell me about Tokyo."}],
            "tools": [WEATHER_TOOL, TIME_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_time"}},
            "max_tokens": 512,
        }
    )
    cases["named"] = _tool_call_error(status, named, "get_time")
    if cases["auto"] is None:
        call = body["choices"][0]["message"]["tool_calls"][0]
        status, final = _chat(
            {
                "messages": [
                    *ask,
                    {"role": "assistant", "content": None, "tool_calls": [call]},
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": '{"temp_c": 21, "sky": "clear"}',
                    },
                ],
                "tools": [WEATHER_TOOL, TIME_TOOL],
                "max_tokens": 512,
            }
        )
        message = final["choices"][0]["message"] if status == 200 else {}
        cases["result_turn"] = (
            None
            if status == 200 and "21" in (message.get("content") or "")
            else f"HTTP {status}: {str(final)[:300]}"
        )
    else:
        cases["result_turn"] = "skipped: auto call failed"
    status, sse = _http(
        "POST",
        f"{_api()}/v1/chat/completions",
        {
            "model": MODEL,
            "messages": ask,
            "tools": [WEATHER_TOOL],
            "stream": True,
            "max_tokens": 512,
        },
    )
    cases["stream"] = (
        None
        if status == 200 and '"tool_calls"' in sse and sse.rstrip().endswith("data: [DONE]")
        else f"HTTP {status}: {sse[-300:]}"
    )
    if len(REPLICAS) > 1:
        before = len(_placement_rows())
        burst = 2 * len(REPLICAS)
        payload = {"messages": ask, "tools": [WEATHER_TOOL], "max_tokens": 512}
        with ThreadPoolExecutor(burst) as pool:
            results = list(pool.map(lambda _: _chat(payload), range(burst)))
        errors = [_tool_call_error(s, b, "get_weather") for s, b in results]
        placed = Counter(row.get("replica_id") for row in _placement_rows()[before:])
        cases["every_replica"] = (
            None
            if not any(errors) and len(placed) == len(REPLICAS)
            else f"errors={[e for e in errors if e][:2]} placement={dict(placed)}"
        )
    return _report(run_dir, "tool-calling", cases)


def _image_url(fmt: str, color: tuple[int, int, int]) -> str:
    from PIL import Image

    output = BytesIO()
    Image.new("RGB", (256, 256), color).save(output, format=fmt)
    mime = {"PNG": "image/png", "WEBP": "image/webp"}[fmt]
    return f"data:{mime};base64,{base64.b64encode(output.getvalue()).decode()}"


def vision(run_dir: Path) -> int:
    """Image chat through Kairyu; WebP reaches llama.cpp as PNG (LCP-D3)."""

    cases: dict[str, str | None] = {}
    for fmt, color, word in (("PNG", (220, 20, 20), "red"), ("WEBP", (20, 60, 220), "blue")):
        status, body = _chat(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": _image_url(fmt, color)}},
                            {
                                "type": "text",
                                "text": "What single color fills this image? One word.",
                            },
                        ],
                    }
                ],
                "max_tokens": 32,
                "temperature": 0.0,
            }
        )
        if status != 200:
            cases[fmt.lower()] = f"HTTP {status}: {str(body)[:200]}"
            continue
        answer = body["choices"][0]["message"].get("content") or ""
        usage = body.get("usage") or {}
        cases[fmt.lower()] = (
            None
            if word in answer.lower() and usage.get("prompt_tokens", 0) > 0
            else f"answer {answer!r}, usage {usage}"
        )
    return _report(run_dir, "vision", cases)


def systemone(run_dir: Path) -> int:
    """Winnow's typed decisions through Kairyu's System One forwarder."""

    request = {
        "state": {"paid": True, "priority": "high"},
        "questions": {
            "paid": {"type": "noul", "instructions": "Has the customer paid?"},
            "route": {
                "type": "choice",
                "instructions": "Which priority is recorded?",
                "criteria": {"low": None, "high": None},
            },
            "rating": {
                "type": "score",
                "instructions": "How urgent is this priority?",
                "criteria": ["not urgent", "moderately urgent", "very urgent"],
            },
        },
    }
    cases: dict[str, str | None] = {}
    status, via_kairyu = _json(
        "POST", f"{_api()}/v1/systemone", {"model": SYSTEMONE_MODEL, **request}
    )
    status_direct, direct = _json(
        "POST", f"{_replica_url(REPLICAS[0])}/v1/systemone", {"model": "Winnow-12B", **request}
    )
    if status != 200 or not isinstance(via_kairyu, dict):
        cases["kairyu"] = f"HTTP {status}: {str(via_kairyu)[:200]}"
        return _report(run_dir, "systemone", cases)
    answers = via_kairyu.get("answers", {})
    usage = via_kairyu.get("usage", {})
    paid = answers.get("paid", {}).get("noul")
    cases["kairyu"] = (
        None
        if set(answers) == set(request["questions"])
        and isinstance(paid, (int, float))
        and paid > 0.5
        and answers.get("route", {}).get("choice") == "high"
        and usage.get("input_tokens", 0) > 0
        else f"answers {json.dumps(answers)[:300]}, usage {usage}"
    )
    cases["same_as_direct"] = (
        None
        if status_direct == 200
        and isinstance(direct, dict)
        and direct.get("answers", {}).get("route", {}).get("choice")
        == answers.get("route", {}).get("choice")
        else f"direct HTTP {status_direct}: {str(direct)[:200]}"
    )
    return _report(run_dir, "systemone", cases)


GATES = {
    "attest": (attest, "/props: build, slots, per-slot context, sampling defaults, tools, vision"),
    "contract": (contract, "l1.correctness.llamacpp_upstream_contract on every replica"),
    "tool-calling": (tool_calling, "auto/named/result-turn/stream tool calls through Kairyu"),
    "vision": (vision, "PNG and WebP image chat through Kairyu"),
    "systemone": (systemone, "typed decisions through Kairyu's System One forwarder"),
    "serving": (serving, "1K-input/256-output TTFT and throughput at the configured concurrency"),
}


def _served_config_sha256() -> str:
    digest = hashlib.sha256()
    for name in ("compose.yaml", "kairyu.yaml", "example.json"):
        digest.update(name.encode() + b"\0" + (HERE / name).read_bytes() + b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verification", choices=(*GATES, "all", "list"))
    parser.add_argument("--run-id")
    parser.add_argument("--no-start", action="store_true")
    args = parser.parse_args()
    if args.verification == "list":
        for name, (_gate, summary) in GATES.items():
            print(f"{name:13} {summary}")
        return
    if not args.no_start:
        subprocess.run([sys.executable, str(HERE / "control.py"), "up"], cwd=ROOT, check=True)
    run_dir = RESULTS_ROOT / (args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    run_dir.mkdir(parents=True, exist_ok=True)
    selected = list(GATES) if args.verification == "all" else [args.verification]
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "started_at": datetime.now(UTC).isoformat(),
        "requested": selected,
        "served_config_sha256": _served_config_sha256(),
        "spec": SPEC,
        "exit_codes": {},
    }
    for name in selected:
        try:
            manifest["exit_codes"][name] = GATES[name][0](run_dir)
        except Exception as error:
            print(f"{name} failed: {type(error).__name__}: {error}", file=sys.stderr)
            manifest["exit_codes"][name] = 1
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"artifacts: {run_dir}")
    raise SystemExit(1 if any(manifest["exit_codes"].values()) else 0)


if __name__ == "__main__":
    main()
