#!/usr/bin/env python3
"""GPU verification gates for Quyet-1.0-Large on one GPU (vLLM chat + System One adapter)."""

from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import control  # noqa: E402

SPEC = control.SPEC
RESULTS_ROOT = ROOT / "verification/results/examples" / SPEC["environment"]
PYTHON = str(ROOT / ".venv/bin/python")
SERVED = control.SERVED
SYSTEMONE = control.SYSTEMONE
DATASET = HERE / SPEC["verification"]["reference"]["dataset"]
REFERENCE_ANSWERS = "reference-answers.jsonl"
SERVED_CONFIG_FILES = ("compose.yaml", "kairyu.yaml", "example.json", "quyet_systemone.py")
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


def _api() -> str:
    return f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}"


def _vllm() -> str:
    return f"http://127.0.0.1:{SPEC['vllm']['host_port']}"


def _adapter() -> str:
    return f"http://127.0.0.1:{SYSTEMONE['host_port']}"


def _http(method: str, url: str, payload: object | None = None, timeout: float = 900.0):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace"), dict(error.headers)


def _json(method: str, url: str, payload: object | None = None, timeout: float = 900.0):
    status, body, headers = _http(method, url, payload, timeout)
    try:
        return status, json.loads(body), headers
    except ValueError:
        return status, body, headers


def _run(command: list[str], *, log: Path, stdout: Path | None = None) -> int:
    print("+ " + " ".join(command[:6]) + (" ..." if len(command) > 6 else ""), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as errors:
        if stdout is None:
            return subprocess.run(
                command, cwd=ROOT, text=True, stdout=errors, stderr=subprocess.STDOUT
            ).returncode
        with stdout.open("w", encoding="utf-8") as out:
            return subprocess.run(
                command, cwd=ROOT, text=True, stdout=out, stderr=errors
            ).returncode


def _report(
    run_dir: Path, gate: str, cases: dict[str, str | None], extra: dict | None = None
) -> int:
    failed = {name: error for name, error in cases.items() if error}
    record = {"gate": gate, "cases": cases, "passed": not failed, **(extra or {})}
    (run_dir / f"{gate}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for name, error in cases.items():
        print(f"{gate}/{name} {'PASS' if error is None else f'FAIL: {error}'}")
    return 1 if failed else 0


def _dataset_sha256() -> str:
    return hashlib.sha256(DATASET.read_bytes()).hexdigest()


def _dataset() -> list[dict]:
    return [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- reference -----------------------------------------------------------------------------


def reference(run_dir: Path) -> int:
    """The official package (transformers, bf16, GPU) answers the reference set, stack down."""

    env = control.compose_env()
    if control._vllm_holds(int(env["GPU_ID"])):
        print("stopping this example's stack: the reference run needs the whole GPU", flush=True)
        control._compose(["down"], env=env)
    row = control.selected_gpu(env, allow_own=False)
    control.ensure_images()
    control.ensure_model(env)
    work = run_dir / "reference"
    work.mkdir(parents=True, exist_ok=True)
    (work / DATASET.name).write_bytes(DATASET.read_bytes())
    started = time.perf_counter()
    code = _run(
        ["docker", "run", "--rm", "--gpus", f"device={env['GPU_ID']}",
         "--cpuset-cpus", control._node_cpulist(control.numa_node(str(row["pci_bus_id"]))),
         "--env", "HF_HUB_OFFLINE=1", "--volume", f"{env['MODEL_STORAGE_PATH']}:/models:ro",
         "--volume", f"{work}:/work:ro", "--entrypoint", "python3", env["QUYET_SYSTEMONE_IMAGE"],
         "-m", "quyet", "predict", f"/models/{SPEC['model']['slug']}",
         "--input", f"/work/{DATASET.name}", "--device", "cuda"],
        log=run_dir / "reference.log",
        stdout=run_dir / REFERENCE_ANSWERS,
    )  # fmt: skip
    wall = time.perf_counter() - started
    lines = (run_dir / REFERENCE_ANSWERS).read_text(encoding="utf-8").splitlines()
    answers = [json.loads(line) for line in lines if line.strip()]
    expected = len(_dataset())
    errors = [answer["error"] for answer in answers if "error" in answer]
    cases = {
        "exit": None if code == 0 else f"exit {code} (see reference.log)",
        "answers": None
        if len(answers) == expected and not errors
        else f"{len(answers)}/{expected}, errors {errors[:3]}",
        "truncation_exercised": None
        if any(answer.get("warnings") for answer in answers)
        else "no request was truncated",
    }
    return _report(
        run_dir, "reference", cases, {"dataset_sha256": _dataset_sha256(), "wall_s": wall}
    )


def _reference_answers(run_dir: Path) -> tuple[list[dict], Path]:
    """This run's reference answers, else the newest earlier run's for the same dataset."""

    candidates = [run_dir] + sorted(
        (path for path in RESULTS_ROOT.iterdir() if path.is_dir() and path != run_dir), reverse=True
    )
    for candidate in candidates:
        record = candidate / "reference.json"
        if not record.exists():
            continue
        meta = json.loads(record.read_text())
        if meta.get("passed") and meta.get("dataset_sha256") == _dataset_sha256():
            lines = (candidate / REFERENCE_ANSWERS).read_text(encoding="utf-8").splitlines()
            return [json.loads(line) for line in lines if line.strip()], candidate
    raise RuntimeError("no passed reference run for this dataset; run `verify.sh reference` first")


# --- attest --------------------------------------------------------------------------------


def _container(name: str) -> dict:
    return json.loads(subprocess.check_output(["docker", "inspect", name]))[0]


def attest(run_dir: Path) -> int:
    """The pinned stack: images, checkpoint, vLLM settings, adapter calibration, Kairyu models."""

    cases: dict[str, str | None] = {}
    vllm = _container(control.VLLM_CONTAINER)
    adapter = _container(control.SYSTEMONE_CONTAINER)
    cases["vllm_image"] = (
        None if vllm["Image"] == SPEC["vllm"]["image_id"] else f"running {vllm['Image']}"
    )
    pinned = SYSTEMONE["image_id"]
    allowed_unpinned = os.environ.get("QUYET_ALLOW_UNPINNED_IMAGE") == "1"
    cases["systemone_image"] = (
        None
        if adapter["Image"] == pinned or (allowed_unpinned and pinned is None)
        else f"running {adapter['Image']}"
    )
    args = vllm["Args"]
    settings = SPEC["vllm"]["settings"]
    wanted = {
        "--max-model-len": str(settings["VLLM_MAX_MODEL_LEN"]),
        "--max-num-seqs": str(settings["VLLM_MAX_NUM_SEQS"]),
        "--gpu-memory-utilization": str(settings["VLLM_GPU_MEMORY_UTILIZATION"]),
    }
    drift = {flag: args[args.index(flag) + 1] if flag in args else None for flag in wanted}
    cases["vllm_settings"] = None if drift == wanted else f"running {drift}, expected {wanted}"
    status, version, _ = _json("GET", f"{_vllm()}/version")
    release = SPEC["vllm"]["release"].lstrip("v")
    cases["vllm_version"] = (
        None
        if isinstance(version, dict) and version.get("version") == release
        else f"{status} {version}"
    )
    status, models, _ = _json("GET", f"{_vllm()}/v1/models")
    rows = models.get("data", []) if isinstance(models, dict) else []
    row = rows[0] if len(rows) == 1 else {}
    cases["vllm_model"] = (
        None
        if row.get("id") == SERVED
        and row.get("root") == f"/models/{SPEC['model']['slug']}"
        and row.get("max_model_len") == SPEC["model"]["max_context_tokens"]
        else f"{status} {rows}"
    )
    logs = subprocess.run(
        ["docker", "logs", control.VLLM_CONTAINER], capture_output=True, text=True
    )
    text = logs.stdout + logs.stderr
    (run_dir / "vllm.log").write_text(text, encoding="utf-8")
    defaults = SPEC["model"]["generation_defaults"]
    sampling = re.findall(r"default (?:chat )?sampling params from model: (\{.*\})", text)
    cases["generation_defaults"] = (
        None
        if sampling
        and all(f"'{name}': {value}" in sampling[-1] for name, value in defaults.items())
        else f"vLLM logged {sampling[-1:] or 'no model sampling defaults'}"
    )
    concurrency = re.findall(
        r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", text
    )
    kv_tokens = re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)
    capacity = float(concurrency[-1][1]) if concurrency else None
    # The plan's context rule: 65,536 only while the KV cache holds 8 such sequences.
    cases["kv_capacity"] = (
        None
        if capacity is not None and capacity >= 8
        else f"{capacity}x at {settings['VLLM_MAX_MODEL_LEN']} tokens"
    )
    status, health, _ = _json("GET", f"{_adapter()}/health")
    health = health if isinstance(health, dict) else {}
    cases["adapter"] = (
        None
        if status == 200
        and health.get("quyet") == SYSTEMONE["quyet_version"]
        and health.get("prompt_version") == SYSTEMONE["prompt_version"]
        and health.get("temperatures") == SYSTEMONE["temperatures"]
        and health.get("model") == SYSTEMONE["upstream_model"]
        and len(health.get("letter_ids") or []) == 10
        else f"{status} {health}"
    )
    env = dict(item.split("=", 1) for item in adapter["Config"]["Env"] if "=" in item)
    adapter_drift = {
        key: env.get(key)
        for key, value in SYSTEMONE["settings"].items()
        if env.get(key) != str(value)
    }
    cases["adapter_settings"] = None if not adapter_drift else f"running {adapter_drift}"
    status, listing, _ = _json("GET", f"{_api()}/v1/models")
    listing = listing if isinstance(listing, dict) else {}
    chat = {row.get("id") for row in listing.get("data", [])}
    jev = {row.get("name") for row in listing.get("models", [])}
    names = {SYSTEMONE["model"], *SYSTEMONE["aliases"]}
    cases["kairyu_models"] = (
        None if chat == {SERVED} and names <= jev else f"{sorted(chat)} / {sorted(jev)}"
    )
    previous = os.environ.get("VERIFY_MODEL")
    os.environ["VERIFY_MODEL"] = "1"
    try:
        control.ensure_model(control.compose_env())
        cases["checkpoint"] = None
    except (SystemExit, subprocess.CalledProcessError) as error:
        cases["checkpoint"] = f"re-hash failed: {error}"
    finally:
        if previous is None:
            os.environ.pop("VERIFY_MODEL", None)
        else:
            os.environ["VERIFY_MODEL"] = previous
    extra = {
        "vllm_image": vllm["Image"],
        "systemone_image": adapter["Image"],
        "kv_cache_tokens": kv_tokens[-1] if kv_tokens else None,
        "max_concurrency_at_context": capacity,
        "adapter_health": health,
    }
    return _report(run_dir, "attest", cases, extra)


# --- systemone -----------------------------------------------------------------------------


def _distribution(answer: dict) -> dict[str, float]:
    if answer["type"] == "noul":
        return {"true": float(answer["noul"]), "false": 1.0 - float(answer["noul"])}
    return {key: float(value) for key, value in answer["probabilities"].items()}


def compare_answers(official: dict, served: dict, margin: float) -> tuple[list[float], list[str]]:
    """Absolute probability differences and the disagreements that fail the gate."""

    diffs: list[float] = []
    problems: list[str] = []
    if served.get("usage", {}).get("input_tokens") != official["usage"]["input_tokens"]:
        problems.append(f"input_tokens {served.get('usage')} vs {official['usage']}")
    if served.get("warnings") != official.get("warnings"):
        problems.append(f"warnings {served.get('warnings')} vs {official.get('warnings')}")
    for qid, expected in official["answers"].items():
        actual = served.get("answers", {}).get(qid)
        if actual is None or actual.get("type") != expected["type"]:
            problems.append(f"{qid}: missing or wrong type")
            continue
        if bool(actual.get("truncated")) != bool(expected.get("truncated")):
            problems.append(
                f"{qid}: truncated {actual.get('truncated')} vs {expected.get('truncated')}"
            )
        want, got = _distribution(expected), _distribution(actual)
        if set(want) != set(got):
            problems.append(f"{qid}: options {sorted(got)} vs {sorted(want)}")
            continue
        diffs.extend(abs(want[key] - got[key]) for key in want)
        ranked = sorted(want.values(), reverse=True)
        if ranked[0] - ranked[1] >= margin and max(want, key=want.get) != max(got, key=got.get):
            problems.append(f"{qid}: top {max(got, key=got.get)!r} vs {max(want, key=want.get)!r}")
    return diffs, problems


def systemone(run_dir: Path) -> int:
    """Kairyu's System One answers vs the official package, plus names and error shapes."""

    config = SPEC["verification"]["reference"]
    official, source = _reference_answers(run_dir)
    requests = _dataset()
    cases: dict[str, str | None] = {}
    served_rows, diffs, problems = [], [], []

    def read(row):
        return _json(
            "POST", f"{_api()}/v1/systemone", {"model": SYSTEMONE["model"], **row}, timeout=300
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(read, requests))
    for index, ((status, body, headers), expected) in enumerate(
        zip(results, official, strict=True)
    ):
        served_rows.append(
            {"status": status, "body": body, "server_timing": headers.get("server-timing")}
        )
        if status != 200 or not isinstance(body, dict):
            problems.append(f"request {index}: HTTP {status} {str(body)[:200]}")
            continue
        row_diffs, row_problems = compare_answers(
            expected, body, float(config["top_choice_margin"])
        )
        diffs.extend(row_diffs)
        problems.extend(f"request {index}: {problem}" for problem in row_problems)
    (run_dir / "systemone-served.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in served_rows), encoding="utf-8"
    )
    median = statistics.median(diffs) if diffs else math.inf
    worst = max(diffs) if diffs else math.inf
    cases["answers_match_official"] = "; ".join(problems[:5]) or None
    cases["median_abs_diff"] = (
        None
        if median <= float(config["max_median_abs_diff"])
        else f"{median:.4f} > {config['max_median_abs_diff']}"
    )
    cases["max_abs_diff"] = (
        None
        if worst <= float(config["max_abs_diff"])
        else f"{worst:.4f} > {config['max_abs_diff']}"
    )
    cases["server_timing"] = (
        None
        if all(row["server_timing"] for row in served_rows if row["status"] == 200)
        else "missing Server-Timing"
    )
    probe = {"state": control.SYSTEMONE_STATE, "questions": control.SYSTEMONE_QUESTIONS}
    for name in SYSTEMONE["aliases"]:
        status, body, _ = _json("POST", f"{_api()}/v1/systemone", {"model": name, **probe})
        error = control.systemone_answer_error(body) if status == 200 else f"HTTP {status}"
        cases[f"alias_{name}"] = error
    eleven = {f"o{i}": None for i in range(11)}
    refusals = {
        "eleven_options": (
            {"q": {"type": "choice", "instructions": "Pick one.", "criteria": eleven}},
            {},
            400,
        ),
        "images": (control.SYSTEMONE_QUESTIONS, {"images": ["data:image/png;base64,AAAA"]}, 400),
        "think": (control.SYSTEMONE_QUESTIONS, {"think": 64}, 400),
        "unknown_type": ({"q": {"type": "rank", "instructions": "x"}}, {}, 400),
        "wrong_shape": ("not an object", {}, 422),
    }
    for case, (questions, options, expected) in refusals.items():
        payload = {
            "model": SYSTEMONE["model"],
            "state": "Hello.",
            "questions": questions,
            **options,
        }
        status, body, _ = _json("POST", f"{_api()}/v1/systemone", payload)
        shape_ok = isinstance(body, dict) and "detail" in body
        cases[f"refuses_{case}"] = (
            None if status == expected and shape_ok else f"HTTP {status} {str(body)[:200]}"
        )
    status, body, _ = _json("POST", f"{_api()}/v1/systemone", {**probe, "model": "no-such-model"})
    cases["refuses_unknown_model"] = None if status == 400 else f"HTTP {status} {str(body)[:200]}"
    status, body, _ = _json(
        "POST", f"{_api()}/v1/systemone", {"model": SYSTEMONE["model"], **probe}
    )
    cases["reads_after_refusals"] = (
        control.systemone_answer_error(body) if status == 200 else f"HTTP {status}"
    )
    extra = {
        "reference_run": source.name,
        "probabilities_compared": len(diffs),
        "median_abs_diff": median,
        "max_abs_diff": worst,
    }
    print(f"compared {len(diffs)} probabilities: median {median:.5f}, max {worst:.5f}")
    return _report(run_dir, "systemone", cases, extra)


# --- chat ----------------------------------------------------------------------------------


def _chat(payload: dict, timeout: float = 600):
    return _json("POST", f"{_api()}/v1/chat/completions", {"model": SERVED, **payload}, timeout)


def _weather_call_error(status: int, body: object) -> str | None:
    if status != 200 or not isinstance(body, dict):
        return f"HTTP {status}: {str(body)[:200]}"
    message = body["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    if not calls or calls[0]["function"]["name"] != "get_weather":
        return f"expected a get_weather call, got {json.dumps(message)[:300]}"
    try:
        arguments = json.loads(calls[0]["function"]["arguments"])
    except (TypeError, ValueError):
        return "tool arguments are not JSON"
    return None if isinstance(arguments.get("city"), str) else f"arguments {arguments}"


def _stream_weather_call_error(status: int, sse: str) -> str | None:
    """Assemble streamed tool-call deltas the way an OpenAI client does."""

    if status != 200:
        return f"HTTP {status}: {sse[-300:]}"
    events = [line[5:].strip() for line in sse.splitlines() if line.startswith("data:")]
    if not events or events[-1] != "[DONE]":
        return f"stream did not end with [DONE]: {sse[-300:]}"
    calls: dict[int, dict[str, str]] = {}
    finish_reason = None
    for event in events[:-1]:
        try:
            chunk = json.loads(event)
            if "error" in chunk:
                return f"stream error: {event[:300]}"
            for choice in chunk.get("choices") or []:
                finish_reason = choice.get("finish_reason") or finish_reason
                for delta in (choice.get("delta") or {}).get("tool_calls") or []:
                    call = calls.setdefault(delta["index"], {"name": "", "arguments": ""})
                    function = delta.get("function") or {}
                    call["name"] += function.get("name") or ""
                    call["arguments"] += function.get("arguments") or ""
        except (AttributeError, KeyError, TypeError, ValueError):
            return f"malformed stream chunk: {event[:300]}"
    if finish_reason != "tool_calls":
        return f"finish_reason {finish_reason!r}, tool calls {calls}"
    message = {"tool_calls": [{"function": calls[index]} for index in sorted(calls)]}
    return _weather_call_error(200, {"choices": [{"message": message}]})


def tool_calling(run_dir: Path) -> int:
    """OpenAI tool calls through Kairyu: auto, named, a tool-result turn and a stream."""

    ask = [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]
    cases: dict[str, str | None] = {}
    status, body, _ = _chat({"messages": ask, "tools": [WEATHER_TOOL], "max_tokens": 512})
    cases["auto"] = _weather_call_error(status, body)
    status, body, _ = _chat({
        "messages": ask, "tools": [WEATHER_TOOL], "max_tokens": 512,
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    })  # fmt: skip
    cases["named"] = _weather_call_error(status, body)
    turn = ask + [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})}}]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": json.dumps({"city": "Paris", "temp_c": 17, "sky": "rain"})},
    ]  # fmt: skip
    status, body, _ = _chat({"messages": turn, "tools": [WEATHER_TOOL], "max_tokens": 512})
    if status != 200 or not isinstance(body, dict):
        cases["result_turn"] = f"HTTP {status}: {str(body)[:200]}"
    else:
        content = body["choices"][0]["message"].get("content") or ""
        cases["result_turn"] = (
            None if "17" in content else f"answer does not use the tool result: {content[:200]!r}"
        )
    status, sse, _ = _http(
        "POST",
        f"{_api()}/v1/chat/completions",
        {
            "model": SERVED,
            "messages": ask,
            "tools": [WEATHER_TOOL],
            "max_tokens": 512,
            "stream": True,
        },
    )
    cases["stream"] = _stream_weather_call_error(status, sse)
    return _report(run_dir, "tool-calling", cases)


def _image_url(fmt: str, color: tuple[int, int, int]) -> str:
    from PIL import Image

    output = BytesIO()
    Image.new("RGB", (256, 256), color).save(output, format=fmt)
    mime = {"PNG": "image/png", "WEBP": "image/webp"}[fmt]
    return f"data:{mime};base64,{base64.b64encode(output.getvalue()).decode()}"


def vision(run_dir: Path) -> int:
    """Image chat through Kairyu: a PNG and a WebP image."""

    cases: dict[str, str | None] = {}
    for fmt, color, word in (("PNG", (220, 20, 20), "red"), ("WEBP", (20, 60, 220), "blue")):
        status, body, _ = _chat(
            control.image_request(_image_url(fmt, color)) | {"temperature": 0.0}
        )
        if status != 200 or not isinstance(body, dict):
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


def _serving_dataset(path: Path, requests: int, approximate_tokens: int) -> None:
    words = ("river", "stone", "cloud", "field", "signal", "lantern", "harbor", "orbit")
    rows = []
    for request in range(requests):
        text = " ".join(
            words[(request * 3 + i * 5) % len(words)] for i in range(approximate_tokens)
        )
        rows.append(
            {
                "conversations": [
                    {"from": "human", "value": f"Case {request}: continue this list. {text}"}
                ]
            }
        )
    path.write_text(json.dumps(rows), encoding="utf-8")


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


def serving(run_dir: Path) -> int:
    """Fixed-length chat (1K in, 256 out with ignore_eos) through Kairyu at c1/4/8."""

    config = SPEC["verification"]["serving"]
    requests, output_tokens = int(config["requests_per_concurrency"]), int(config["output_tokens"])
    dataset = run_dir / "serving.json"
    _serving_dataset(dataset, requests, int(config["prompt_tokens_approx"]))
    cases: dict[str, str | None] = {}
    for concurrency in config["concurrency"]:
        row_dir = run_dir / f"serving-c{concurrency}"
        code = _run(
            [PYTHON, str(ROOT / "verification/l1/performance/serving_bench.py"),
             "--base-url", f"{_api()}/v1", "--model", SERVED, "--dataset", str(dataset),
             "--num-requests", str(requests), "--concurrency", str(concurrency),
             "--max-tokens", str(output_tokens), "--ignore-eos", "--temperature", "1.0",
             "--seed", "0", "--timeout", "3600", "--results-dir", str(row_dir),
             "--tensor-parallel", "1", "--dp-replicas", "1"],
            log=run_dir / f"serving-c{concurrency}.log",
        )  # fmt: skip
        cases[f"c{concurrency}"] = (
            f"exit {code}" if code else _serving_row_error(row_dir, requests, output_tokens)
        )
    return _report(run_dir, "serving", cases)


# --- System One load -----------------------------------------------------------------------


def _long_state(namespace: str, index: int, approximate_tokens: int) -> str:
    sentences = (
        "The customer says the dashboard has been unreachable since the morning deploy.",
        "They mention a contract renewal meeting with their board later this week.",
        "Two earlier tickets about slow exports were closed without a fix.",
        "Their admin tried clearing the cache and switching browsers without success.",
    )
    body = " ".join(
        sentences[(index + i) % len(sentences)] for i in range(approximate_tokens // 14)
    )
    return f"[{namespace}-{index}] {body}"


async def _systemone_burst(
    base_url: str, model: str, requests: int, concurrency: int, namespace: str, state_tokens: int
):
    """Cache-busted System One requests; per request (status, seconds, answer error)."""

    import httpx

    gate = asyncio.Semaphore(concurrency)

    async def one(client, index: int):
        payload = control.systemone_request(
            _long_state(namespace, index, state_tokens), model=model
        )
        async with gate:
            start = time.perf_counter()
            response = await client.post(f"{base_url}/v1/systemone", json=payload)
            elapsed = time.perf_counter() - start
        if response.status_code != 200:
            return response.status_code, elapsed, None
        return 200, elapsed, control.systemone_answer_error(response.json())

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=300, limits=limits) as client:
        start = time.perf_counter()
        rows = await asyncio.gather(*(one(client, i) for i in range(requests)))
        return rows, time.perf_counter() - start


def _nearest_rank(values: list[float], fraction: float) -> float | None:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else None


def systemone_serving(run_dir: Path) -> int:
    """Cache-busted 1K-token states with 3 questions: req/s and p50/p95 at c1/16/32/64."""

    config = SPEC["verification"]["systemone_serving"]
    requests, state_tokens = (
        int(config["requests_per_concurrency"]),
        int(config["state_tokens_approx"]),
    )
    asyncio.run(
        _systemone_burst(_api(), SYSTEMONE["model"], 8, 8, f"{run_dir.name}-warmup", state_tokens)
    )
    # The adapter directly, at c1 only: Kairyu's own overhead. Above its queue it answers 529.
    targets = [("kairyu", _api(), SYSTEMONE["model"], level) for level in config["concurrency"]]
    targets.append(("direct", _adapter(), SYSTEMONE["upstream_model"], 1))
    rows, cases = [], {}
    for target, url, model, level in targets:
        name = f"{target}-c{level}"
        results, wall = asyncio.run(
            _systemone_burst(url, model, requests, level, f"{run_dir.name}-{name}", state_tokens)
        )
        latencies = [elapsed for status, elapsed, _ in results if status == 200]
        bad = [(status, error) for status, _, error in results if status != 200 or error]
        row = {
            "row": name, "requests": requests, "ok": requests - len(bad), "wall_s": wall,
            "requests_per_s": len(latencies) / wall,
            "p50_s": _nearest_rank(latencies, 0.5), "p95_s": _nearest_rank(latencies, 0.95),
        }  # fmt: skip
        rows.append(row)
        print(json.dumps(row), flush=True)
        cases[name] = f"{len(bad)} failed: {bad[:3]}" if bad else None
    limit = float(config["c1_p50_limit_s"])
    c1 = next(row for row in rows if row["row"] == "kairyu-c1")
    cases["kairyu_c1_p50"] = (
        None if c1["p50_s"] is not None and c1["p50_s"] <= limit else f"{c1['p50_s']} s > {limit} s"
    )
    (run_dir / "systemone-serving-rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    return _report(run_dir, "systemone-serving", cases)


def systemone_isolation(run_dir: Path) -> int:
    """A System One burst past every limit gets Kairyu's 429, never the adapter's 529, while
    chat keeps answering and the chat replica stays healthy."""

    config = SPEC["verification"]["isolation"]
    reads, chats = int(config["reads"]), int(config["chats"])
    chat = {"max_tokens": 256, "messages": [{"role": "user", "content": control.ARITHMETIC}]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=chats) as pool:
        chat_results = [pool.submit(_chat, chat) for _ in range(chats)]
        burst, wall = asyncio.run(
            _systemone_burst(_api(), SYSTEMONE["model"], reads, reads, f"{run_dir.name}-burst", 256)
        )
        chat_bodies = [future.result() for future in chat_results]
    statuses: dict[int, int] = {}
    for status, _, _ in burst:
        statuses[status] = statuses.get(status, 0) + 1
    metrics = urllib.request.urlopen(f"{_api()}/metrics", timeout=5).read().decode()
    healthy = control.healthy_replicas(metrics, SERVED)
    chat_errors = [
        f"HTTP {status}" if status != 200 else control.chat_answer_error(body, expected="323")
        for status, body, _ in chat_bodies
    ]
    cases = {
        "burst_only_200_or_429": None if set(statuses) <= {200, 429} else f"statuses {statuses}",
        "burst_reaches_kairyu_limit": None if statuses.get(429) else f"no 429: {statuses}",
        "answers_valid": next(
            (error for status, _, error in burst if status == 200 and error), None
        ),
        "chat_answers_during_burst": next((error for error in chat_errors if error), None),
        "chat_replica_healthy": None if healthy == 1 else f"healthy replicas {healthy}",
    }
    (run_dir / "isolation.json").write_text(
        json.dumps({"statuses": statuses, "wall_s": wall, "healthy": healthy}, indent=2) + "\n"
    )
    print(f"burst statuses {statuses} in {wall:.1f} s; healthy chat replicas {healthy}")
    return _report(run_dir, "systemone-isolation", cases)


GATES = {
    "reference": (
        reference,
        "the official quyet package (transformers, GPU) answers the reference set; stack down",
    ),
    "attest": (
        attest,
        "images, checkpoint re-hash, vLLM settings and KV capacity, adapter calibration, models",
    ),
    "systemone": (
        systemone,
        "Kairyu's System One answers vs the official package; names; error shapes",
    ),
    "tool-calling": (tool_calling, "auto/named/result-turn/stream tool calls through Kairyu"),
    "vision": (vision, "PNG and WebP image chat through Kairyu"),
    "serving": (serving, "1K-input/256-output chat at c1/4/8"),
    "systemone-serving": (systemone_serving, "cache-busted decisions at c1/16/32/64; c1 p50 limit"),
    "systemone-isolation": (
        systemone_isolation,
        "a 640-read burst gets 429s, never 529, while chat answers",
    ),
}


def _served_config_sha256() -> str:
    digest = hashlib.sha256()
    for name in SERVED_CONFIG_FILES:
        digest.update(name.encode() + b"\0" + (HERE / name).read_bytes() + b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verification", choices=(*GATES, "all", "list"))
    parser.add_argument("--run-id")
    parser.add_argument(
        "--no-start", action="store_true", help="do not run control.py up before stack gates"
    )
    args = parser.parse_args()
    if args.verification == "list":
        for name, (_gate, summary) in GATES.items():
            print(f"{name:20} {summary}")
        return
    run_dir = RESULTS_ROOT / (args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    run_dir.mkdir(parents=True, exist_ok=True)
    selected = list(GATES) if args.verification == "all" else [args.verification]
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "started_at": datetime.now(UTC).isoformat(),
        "requested": selected,
        "served_config_sha256": _served_config_sha256(),
        "dataset_sha256": _dataset_sha256(),
        "spec": SPEC,
        "exit_codes": {},
    }
    started = False
    for name in selected:
        if name != "reference" and not started and not args.no_start:
            subprocess.run([sys.executable, str(HERE / "control.py"), "up"], cwd=ROOT, check=True)
            started = True
        try:
            manifest["exit_codes"][name] = GATES[name][0](run_dir)
        except Exception as error:  # a gate that cannot run is a failed gate
            print(f"{name} failed: {type(error).__name__}: {error}", file=sys.stderr)
            manifest["exit_codes"][name] = 1
        if manifest["exit_codes"][name]:
            print(f"stopping at the first failed gate: {name}", file=sys.stderr)
            break
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"artifacts: {run_dir}")
    raise SystemExit(1 if any(manifest["exit_codes"].values()) else 0)


if __name__ == "__main__":
    main()
