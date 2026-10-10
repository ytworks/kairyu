#!/usr/bin/env python3
"""GPU verification gates for Quyet-1.0-Large on one GPU, built around how Jev is used.

A System One model is a decision API for software (docs.typesafe.ai): typed questions
about a state, calibrated probabilities, many questions per call, consistent answers,
called through TypeSafe's SDK. The gates check those properties through Kairyu, against
the quyet package's own run and JevBench, the independent benchmark for Jev-compatible
systems. No chat model is published.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import control  # noqa: E402

SPEC = control.SPEC
RESULTS_ROOT = ROOT / "verification/results/examples" / SPEC["environment"]
SERVED = control.SERVED
SYSTEMONE = control.SYSTEMONE
CHECKS = SPEC["verification"]
AUTHORED = HERE / CHECKS["reference"]["dataset"]
REFERENCE_REQUESTS = "reference-requests.jsonl"
REFERENCE_ANSWERS = "reference-answers.jsonl"
SERVED_CONFIG_FILES = ("compose.yaml", "kairyu.yaml", "example.json", "quyet_systemone.py")
FANOUT_QUESTIONS = [
    {"type": "noul", "instructions": "The customer is asking for a refund."},
    {"type": "noul", "instructions": "The customer mentions a deadline."},
    {
        "type": "choice",
        "instructions": "Which team should handle it?",
        "criteria": {"outage": "service down", "billing": "charges", "feature": "how-to"},
    },
    {
        "type": "score",
        "instructions": "How upset is the customer?",
        "criteria": ["calm", "annoyed", "furious"],
    },
]


def _api() -> str:
    return f"http://127.0.0.1:{os.environ.get('API_PORT', SPEC['api_port'])}"


def _vllm() -> str:
    return f"http://127.0.0.1:{SPEC['vllm']['host_port']}"


def _adapter() -> str:
    return f"http://127.0.0.1:{SYSTEMONE['host_port']}"


def _scratch() -> Path:
    """Large, disposable verification files live on NVMe, never on the root disk."""

    path = control.environment_storage() / "bench-tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def _systemone(payload: dict, timeout: float = 300):
    return _json(
        "POST", f"{_api()}/v1/systemone", {"model": SYSTEMONE["model"], **payload}, timeout
    )


def _run(command: list[str], *, log: Path, stdout: Path | None = None, env=None) -> int:
    print("+ " + " ".join(command[:6]) + (" ..." if len(command) > 6 else ""), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as errors:
        if stdout is None:
            return subprocess.run(
                command, cwd=ROOT, text=True, stdout=errors, stderr=subprocess.STDOUT, env=env
            ).returncode
        with stdout.open("w", encoding="utf-8") as out:
            return subprocess.run(
                command, cwd=ROOT, text=True, stdout=out, stderr=errors, env=env
            ).returncode


def _report(run_dir: Path, gate: str, cases: dict[str, str | None], extra: dict | None = None):
    failed = {name: error for name, error in cases.items() if error}
    record = {"gate": gate, "cases": cases, "passed": not failed, **(extra or {})}
    (run_dir / f"{gate}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for name, error in cases.items():
        print(f"{gate}/{name} {'PASS' if error is None else f'FAIL: {error}'}")
    return 1 if failed else 0


def _nearest_rank(values: list[float], fraction: float) -> float | None:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else None


# --- JevBench --------------------------------------------------------------------------------


def jevbench_checkout() -> Path:
    """JevBench at the pinned revision, on NVMe, with its public splits hash-checked."""

    config = CHECKS["jevbench"]
    target = _scratch() / f"jevbench-{config['revision'][:12]}"
    if not (target / "jevbench").is_dir():
        url = f"https://codeload.github.com/fstandhartinger/jevbench/tar.gz/{config['revision']}"
        with urllib.request.urlopen(url, timeout=120) as response:
            archive = response.read()
        staging = target.with_suffix(".partial")
        staging.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(staging, filter="data")
        (root,) = staging.iterdir()
        root.rename(target)
        staging.rmdir()
    for name, split in config["splits"].items():
        digest = hashlib.sha256((target / split["path"]).read_bytes()).hexdigest()
        if digest != split["sha256"]:
            raise RuntimeError(f"JevBench {name} split hash {digest} differs from example.json")
    return target


def _jevbench_modules(checkout: Path):
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    from jevbench import scoring, summarize, tasks
    from jevbench.adapters.base import build_question

    return tasks, scoring, summarize, build_question


def _jevbench_tasks(checkout: Path) -> dict[str, list]:
    tasks, _, _, _ = _jevbench_modules(checkout)
    return {
        name: tasks.load_jsonl(str(checkout / split["path"]))
        for name, split in CHECKS["jevbench"]["splits"].items()
    }


def _reference_requests(checkout: Path) -> list[dict]:
    """The authored edge cases, then every public JevBench item as its typesafe adapter asks it."""

    _, _, _, build_question = _jevbench_modules(checkout)
    rows = [
        {"id": f"authored-{index}", **json.loads(line)}
        for index, line in enumerate(AUTHORED.read_text(encoding="utf-8").splitlines())
        if line.strip()
    ]
    for name, items in _jevbench_tasks(checkout).items():
        for task in items:
            rows.append(
                {
                    "id": f"jevbench-{name}:{task.id}",
                    "state": task.state,
                    "questions": {"decision": build_question(task)},
                }
            )
    return rows


def _probabilities(answer: dict) -> dict[str, float]:
    """JevBench's typesafe adapter mapping: noul -> yes/no, choice and score as returned."""

    if answer["type"] == "noul":
        return {"yes": float(answer["noul"]), "no": 1.0 - float(answer["noul"])}
    return {key: float(value) for key, value in answer["probabilities"].items()}


def _official_records(checkout: Path, items: list, answers: dict[str, dict], split: str):
    """The official package's answers, scored by JevBench's own scorer."""

    _, scoring, _, _ = _jevbench_modules(checkout)
    records = []
    for task in items:
        answer = answers[f"jevbench-{split}:{task.id}"]["answers"]["decision"]
        scored = scoring.score_task(_probabilities(answer), task)
        records.append(
            {
                "task_id": task.id, "family": task.family, "split": task.split,
                "group": task.group, "ok": True, "valid": scored["valid"],
                "correct": scored["correct"], "predicted": scored.get("predicted"),
                "probs": scored.get("probs"), "strict_valid": scored.get("strict_valid", False),
                "renormalized": scored.get("renormalized", False), "latency_s": None,
                "cost_usd": None, "cost_basis": "official_package_reference",
                "probs_source": "native", "model": "quyet-package",
            }
        )  # fmt: skip
    return records


# --- reference -------------------------------------------------------------------------------


def reference_fingerprint(requests: list[dict]) -> str:
    """Everything a reference answer depends on: each request body, the checkpoint, the
    package and the image it ran in, and the JevBench revision the items came from.

    Key order is kept, not sorted: the package letters a question's options in the
    order its criteria arrive, so reordering them is a different prompt."""

    material = [
        requests,
        [SPEC["model"]["revision"], SPEC["model"]["tree_sha256"]],
        list(control.adapter_labels().items()),
        CHECKS["jevbench"]["revision"],
    ]
    canonical = json.dumps(material, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def reference(run_dir: Path) -> int:
    """The official package (transformers, bf16, GPU) answers every reference request."""

    env = control.compose_env()
    if control._vllm_holds(int(env["GPU_ID"])):
        print("stopping this example's stack: the reference run needs the whole GPU", flush=True)
        control._compose(["down"], env=env)
    row = control.selected_gpu(env, allow_own=False)
    control.ensure_images()
    control.ensure_model(env)
    requests = _reference_requests(jevbench_checkout())
    work = _scratch() / f"reference-{run_dir.name}"
    work.mkdir(parents=True, exist_ok=True)
    (work / REFERENCE_REQUESTS).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests), encoding="utf-8"
    )
    (run_dir / REFERENCE_REQUESTS).write_bytes((work / REFERENCE_REQUESTS).read_bytes())
    cpus = control._node_cpulist(control.numa_node(str(row["pci_bus_id"])))
    started = time.perf_counter()
    code = _run(
        ["docker", "run", "--rm", "--gpus", f"device={env['GPU_ID']}", "--cpuset-cpus", cpus,
         "--env", "HF_HUB_OFFLINE=1", "--volume", f"{env['MODEL_STORAGE_PATH']}:/models:ro",
         "--volume", f"{work}:/work:ro", "--entrypoint", "python3", env["QUYET_SYSTEMONE_IMAGE"],
         "-m", "quyet", "predict", f"/models/{SPEC['model']['slug']}",
         "--input", f"/work/{REFERENCE_REQUESTS}", "--device", "cuda"],
        log=run_dir / "reference.log",
        stdout=run_dir / REFERENCE_ANSWERS,
    )  # fmt: skip
    wall = time.perf_counter() - started
    lines = (run_dir / REFERENCE_ANSWERS).read_text(encoding="utf-8").splitlines()
    answers = [json.loads(line) for line in lines if line.strip()]
    errors = [answer["error"] for answer in answers if "error" in answer]
    cases = {
        "exit": None if code == 0 else f"exit {code} (see reference.log)",
        "answers": None
        if len(answers) == len(requests) and not errors
        else f"{len(answers)}/{len(requests)}, errors {errors[:3]}",
        "truncation_exercised": None
        if any(answer.get("warnings") for answer in answers)
        else "no request was truncated",
    }
    extra = {
        "requests": len(requests),
        "wall_s": wall,
        "fingerprint": reference_fingerprint(requests),
    }
    return _report(run_dir, "reference", cases, extra)


def _reference(run_dir: Path) -> tuple[list[dict], dict[str, dict], Path]:
    """This run's reference, else the newest passed one made from the same requests and
    conditions; the requests returned are always the current ones."""

    requests = _reference_requests(jevbench_checkout())
    fingerprint = reference_fingerprint(requests)
    candidates = [run_dir] + sorted(
        (path for path in RESULTS_ROOT.iterdir() if path.is_dir() and path != run_dir),
        reverse=True,
    )
    for candidate in candidates:
        record = candidate / "reference.json"
        if not record.exists():
            continue
        meta = json.loads(record.read_text())
        if not meta.get("passed") or meta.get("fingerprint") != fingerprint:
            continue
        lines = (candidate / REFERENCE_ANSWERS).read_text(encoding="utf-8").splitlines()
        answers = [json.loads(line) for line in lines if line.strip()]
        ids = [r["id"] for r in requests]
        return requests, dict(zip(ids, answers, strict=True)), candidate
    raise RuntimeError(
        "no passed reference for the current requests, checkpoint and image; "
        "run `verify.sh reference`"
    )


# --- attest ----------------------------------------------------------------------------------


def _container(name: str) -> dict:
    return json.loads(subprocess.check_output(["docker", "inspect", name]))[0]


def attest(run_dir: Path) -> int:
    """The pinned stack: images, checkpoint, vLLM settings, adapter calibration, Kairyu models."""

    cases: dict[str, str | None] = {}
    vllm = _container(control.VLLM_CONTAINER)
    adapter = _container(control.SYSTEMONE_CONTAINER)
    cases["vllm_image"] = (
        None
        if control.vllm_image_matches(vllm["Image"])
        else f"running {vllm['Image']}, not {SPEC['vllm']['repo_digest']}"
    )
    cases["systemone_image"] = (
        None
        if control.adapter_image_matches(adapter["Image"])
        else f"running {adapter['Image']}, built from other sources"
    )
    args = vllm["Args"]
    settings = SPEC["vllm"]["settings"]
    wanted = {
        "--max-model-len": str(settings["VLLM_MAX_MODEL_LEN"]),
        "--max-num-seqs": str(settings["VLLM_MAX_NUM_SEQS"]),
        "--gpu-memory-utilization": str(settings["VLLM_GPU_MEMORY_UTILIZATION"]),
    }
    drift = {flag: args[args.index(flag) + 1] if flag in args else None for flag in wanted}
    vllm_env = dict(item.split("=", 1) for item in vllm["Config"]["Env"] if "=" in item)
    if vllm_env.get("VLLM_BATCH_INVARIANT") != str(settings["VLLM_BATCH_INVARIANT"]):
        drift["VLLM_BATCH_INVARIANT"] = vllm_env.get("VLLM_BATCH_INVARIANT")
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
    (_scratch() / f"vllm-{run_dir.name}.log").write_text(text, encoding="utf-8")
    kv_tokens = re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)
    tokens = int(kv_tokens[-1].replace(",", "")) if kv_tokens else None
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
        None if not chat and names <= jev else f"{sorted(chat)} / {sorted(jev)}"
    )
    payload = {"model": SERVED, "messages": [{"role": "user", "content": "Hello"}]}
    status, body, _ = _json("POST", f"{_api()}/v1/chat/completions", payload)
    cases["chat_not_public"] = None if status == 404 else f"HTTP {status} {str(body)[:200]}"
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
        "kv_cache_tokens": tokens,
        "adapter_health": health,
    }
    return _report(run_dir, "attest", cases, extra)


# --- systemone: the official package's answers -----------------------------------------------


def _distribution(answer: dict) -> dict[str, float]:
    if answer["type"] == "noul":
        return {"true": float(answer["noul"]), "false": 1.0 - float(answer["noul"])}
    return {key: float(value) for key, value in answer["probabilities"].items()}


def official_confidence(distribution: dict[str, float]) -> float:
    """TypeSafe's Choice confidence, (K * p_max - 1) / (K - 1); |2p - 1| for a noul."""

    k = len(distribution)
    return (k * max(distribution.values()) - 1) / (k - 1)


def compare_answers(
    official: dict, served: dict, min_confidence: float
) -> tuple[list[float], list[str], int, list[str]]:
    """(probability differences, structural problems, confident official answers, those
    whose top option the served answer changed). Confident means TypeSafe's confidence
    floor of 0.5; below it the official answer is, in TypeSafe's words, genuinely
    uncertain. Any structural problem (prompt length, truncation, shape) fails the gate."""

    diffs: list[float] = []
    problems: list[str] = []
    confident, flipped = 0, []
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
        if official_confidence(want) >= min_confidence:
            confident += 1
            if max(want, key=want.get) != max(got, key=got.get):
                flipped.append(
                    f"{qid}: top {max(got, key=got.get)!r} vs {max(want, key=want.get)!r}"
                )
    return diffs, problems, confident, flipped


def systemone(run_dir: Path) -> int:
    """Kairyu's System One answers vs the official package on every reference request, plus
    the model names and the error shapes TypeSafe documents."""

    config = CHECKS["reference"]
    requests, official, source = _reference(run_dir)
    cases: dict[str, str | None] = {}

    def read(row):
        return _systemone({"state": row["state"], "questions": row["questions"]})

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(read, requests))
    served_rows, diffs, problems, flips = [], [], [], []
    confident = 0
    for row, (status, body, headers) in zip(requests, results, strict=True):
        served_rows.append({"id": row["id"], "status": status, "body": body,
                            "server_timing": headers.get("server-timing")})  # fmt: skip
        if status != 200 or not isinstance(body, dict):
            problems.append(f"{row['id']}: HTTP {status} {str(body)[:200]}")
            continue
        row_diffs, row_problems, row_confident, row_flips = compare_answers(
            official[row["id"]], body, float(config["same_top_min_official_confidence"])
        )
        diffs.extend(row_diffs)
        problems.extend(f"{row['id']}: {problem}" for problem in row_problems)
        flips.extend(f"{row['id']}: {flip}" for flip in row_flips)
        confident += row_confident
    (run_dir / "systemone-served.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in served_rows), encoding="utf-8"
    )
    median = statistics.median(diffs) if diffs else math.inf
    p99 = _nearest_rank(diffs, 0.99) if diffs else math.inf
    worst = max(diffs) if diffs else math.inf
    cases["prompts_and_shapes_match_official"] = "; ".join(problems[:5]) or None
    agreement = 1 - len(flips) / confident if confident else 0.0
    cases["confident_answers_agree"] = (
        None
        if agreement >= float(config["min_confident_agreement"])
        else f"{agreement:.4f} < {config['min_confident_agreement']}: {flips[:3]}"
    )
    cases["median_abs_diff"] = (
        None
        if median <= float(config["max_median_abs_diff"])
        else f"{median:.4f} > {config['max_median_abs_diff']}"
    )
    cases["server_timing"] = (
        None
        if all(r["server_timing"] for r in served_rows if r["status"] == 200)
        else "missing Server-Timing"
    )
    probe = {"state": control.SYSTEMONE_STATE, "questions": control.SYSTEMONE_QUESTIONS}
    for name in SYSTEMONE["aliases"]:
        status, body, _ = _json("POST", f"{_api()}/v1/systemone", {"model": name, **probe})
        cases[f"alias_{name}"] = (
            control.systemone_answer_error(body) if status == 200 else f"HTTP {status}"
        )
    eleven = {f"o{i}": None for i in range(11)}
    refusals = {
        "eleven_options": (
            {"q": {"type": "choice", "instructions": "Pick.", "criteria": eleven}}, {}, 400
        ),
        "images": (control.SYSTEMONE_QUESTIONS, {"images": ["data:image/png;base64,AAAA"]}, 400),
        "think": (control.SYSTEMONE_QUESTIONS, {"think": 64}, 400),
        "unknown_type": ({"q": {"type": "rank", "instructions": "x"}}, {}, 400),
        "malformed": ("not an object", {}, 422),
    }  # fmt: skip
    for case, (questions, options, expected) in refusals.items():
        status, body, _ = _systemone({"state": "Hello.", "questions": questions, **options})
        shaped = isinstance(body, dict) and "detail" in body
        cases[f"refuses_{case}"] = (
            None if status == expected and shaped else f"HTTP {status} {str(body)[:200]}"
        )
    status, body, _ = _json("POST", f"{_api()}/v1/systemone", {**probe, "model": "no-such-model"})
    cases["refuses_unknown_model"] = None if status == 400 else f"HTTP {status} {str(body)[:200]}"
    status, body, _ = _systemone(probe)
    cases["reads_after_refusals"] = (
        control.systemone_answer_error(body) if status == 200 else f"HTTP {status}"
    )
    extra = {"reference_run": source.name, "requests": len(requests),
             "probabilities_compared": len(diffs), "median_abs_diff": median,
             "p99_abs_diff": p99, "max_abs_diff": worst, "confident_answers": confident,
             "confident_flips": flips}  # fmt: skip
    print(f"compared {len(diffs)} probabilities: median {median:.5f}, p99 {p99:.4f}, "
          f"max {worst:.4f}")  # fmt: skip
    return _report(run_dir, "systemone", cases, extra)


# --- jevbench --------------------------------------------------------------------------------


def _headline(summary: dict) -> dict:
    """The numbers the gate compares; JevBench reports ECE with its bins."""

    ece = summary.get("ece")
    return {**summary, "ece": ece.get("ece") if isinstance(ece, dict) else ece}


def jevbench(run_dir: Path) -> int:
    """JevBench's own runner against Kairyu (typesafe adapter, one request at a time, as the
    board measures), scored beside the official package's answers on the same items."""

    config = CHECKS["jevbench"]
    checkout = jevbench_checkout()
    _, _, summarize, _ = _jevbench_modules(checkout)
    _, official, source = _reference(run_dir)
    all_tasks = _jevbench_tasks(checkout)
    out = run_dir / "jevbench"
    out.mkdir(exist_ok=True)
    # JevBench never overwrites raw evidence, so every invocation gets its own directory.
    raw = _scratch() / f"jevbench-raw-{run_dir.name}-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    env = {**os.environ, "PYTHONPATH": str(checkout)}
    cases: dict[str, str | None] = {}
    rows = {}
    for name, items in all_tasks.items():
        results = out / f"{name}-results.jsonl"
        if results.exists():
            results.unlink()
        code = _run(
            [sys.executable, "-m", "jevbench.cli", "run",
             "--tasks", str(checkout / config["splits"][name]["path"]),
             "--adapter", "typesafe", "--endpoint", _api(), "--key-env", "",
             "--model", SYSTEMONE["model"], "--results", str(results),
             "--raw-dir", str(raw / name), "--ledger", str(raw / f"{name}-ledger.jsonl"),
             "--reserve-usd", "0", "--cost-basis", "self_hosted_no_tariff",
             "--run-label", f"kairyu-{run_dir.name}"],
            log=out / f"{name}-run.log", env=env,
        )  # fmt: skip
        served = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
        reference_records = _official_records(checkout, items, official, name)
        served_summary = summarize.summarize(items, served)
        official_summary = summarize.summarize(items, reference_records)
        rows[name] = {"served": served_summary, "official": official_summary}
        (out / f"{name}-summary.json").write_text(json.dumps(rows[name], indent=2) + "\n")
        s, o = _headline(served_summary), _headline(official_summary)
        problems = []
        if code != 0 or s["n_attempted"] != len(items):
            problems.append(f"runner exit {code}, {s['n_attempted']}/{len(items)} attempted")
        if s["schema_validity"] != 1.0:
            problems.append(f"schema validity {s['schema_validity']}")
        if abs(s["n_correct"] - o["n_correct"]) > int(config["max_accuracy_items_off"]):
            problems.append(f"correct {s['n_correct']} vs official {o['n_correct']}")
        if s["brier_mean"] is None or abs(s["brier_mean"] - o["brier_mean"]) > float(
            config["max_brier_diff"]
        ):
            problems.append(f"brier {s['brier_mean']} vs official {o['brier_mean']}")
        cases[name] = "; ".join(problems) or None
        if s["accuracy"] is not None and s["latency"]["p50_s"] is not None:
            print(
                f"jevbench {name}: accuracy {s['accuracy']:.3f} (official {o['accuracy']:.3f}), "
                f"brier {s['brier_mean']:.4f} ({o['brier_mean']:.4f}), ece {s['ece']:.4f} "
                f"({o['ece']:.4f}), p50 {s['latency']['p50_s']:.3f} s",
                flush=True,
            )
    # ECE uses 10 bins: on a 48- or 72-item split one near-even answer crossing a bin moves
    # it by about 0.01, so calibration is compared over all public items at once.
    items_all = [task for items in all_tasks.values() for task in items]
    served_all = [
        json.loads(line)
        for name in all_tasks
        for line in (out / f"{name}-results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    official_all = [
        record
        for name, items in all_tasks.items()
        for record in _official_records(checkout, items, official, name)
    ]
    s_all = _headline(summarize.summarize(items_all, served_all))
    o_all = _headline(summarize.summarize(items_all, official_all))
    rows["all"] = {"served": s_all, "official": o_all}
    ece_gap = abs(s_all["ece"] - o_all["ece"]) if s_all["ece"] is not None else math.inf
    if s_all["ece"] is None:
        return _report(run_dir, "jevbench", {**cases, "overall_ece": "no served answers"})
    cases["overall_ece"] = (
        None
        if ece_gap <= float(config["max_overall_ece_diff"])
        else f"ece {s_all['ece']} vs official {o_all['ece']}"
    )
    print(
        f"jevbench all {len(items_all)}: correct {s_all['n_correct']} (official "
        f"{o_all['n_correct']}), brier {s_all['brier_mean']:.4f} ({o_all['brier_mean']:.4f}), "
        f"ece {s_all['ece']:.4f} ({o_all['ece']:.4f})",
        flush=True,
    )
    latencies = [
        json.loads(line)["latency_s"]
        for name in all_tasks
        for line in (out / f"{name}-results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    p50 = statistics.median(latencies) if latencies else math.inf
    cases["sequential_p50"] = (
        None if p50 <= float(config["p50_limit_s"]) else f"{p50:.3f} s > {config['p50_limit_s']} s"
    )
    extra = {"reference_run": source.name, "p50_s": p50, "splits": rows,
             "jevbench_revision": config["revision"]}  # fmt: skip
    return _report(run_dir, "jevbench", cases, extra)


# --- fan-out and consistency -----------------------------------------------------------------


def _long_state(namespace: str, index: int, approximate_tokens: int) -> str:
    sentences = (
        "The customer says the dashboard has been unreachable since the morning deploy.",
        "They mention a contract renewal meeting with their board later this week.",
        "Two earlier tickets about slow exports were closed without a fix.",
        "Their admin tried clearing the cache and switching browsers without success.",
    )
    body = " ".join(
        sentences[(index + i) % len(sentences)] for i in range(max(1, approximate_tokens // 14))
    )
    return f"[{namespace}-{index}] {body}"


def _questions(count: int) -> dict:
    return {f"q{i}": FANOUT_QUESTIONS[i % len(FANOUT_QUESTIONS)] for i in range(count)}


def fanout(run_dir: Path) -> int:
    """Many questions in one call (TypeSafe's speculative fan-out): the state is read once and
    the questions in parallel, so 32 questions cost far less than 32 calls."""

    config = CHECKS["fanout"]
    rows, cases = {}, {}
    for count in config["questions"]:
        latencies, errors = [], []
        for index in range(int(config["requests_per_size"])):
            state = _long_state(f"{run_dir.name}-fan{count}", index, config["state_tokens_approx"])
            started = time.perf_counter()
            status, body, _ = _systemone({"state": state, "questions": _questions(count)})
            latencies.append(time.perf_counter() - started)
            answered = isinstance(body, dict) and set(body.get("answers", {})) == set(
                _questions(count)
            )
            if status != 200 or not answered:
                errors.append(f"HTTP {status} {str(body)[:120]}")
        rows[count] = {"p50_s": statistics.median(latencies), "errors": errors}
        cases[f"q{count}_answered"] = errors[0] if errors else None
        print(f"fanout {count} questions: p50 {rows[count]['p50_s']:.3f} s", flush=True)
    smallest, largest = min(config["questions"]), max(config["questions"])
    ratio = rows[largest]["p50_s"] / rows[smallest]["p50_s"]
    limit = float(config["max_ratio_32_to_1"])
    cases["one_call_beats_separate_calls"] = (
        None if ratio <= limit else f"{largest} questions take {ratio:.1f}x one question"
    )
    return _report(run_dir, "fanout", cases, {"rows": rows, "ratio": ratio})


def consistency(run_dir: Path) -> int:
    """The same request answers the same way: repeated alone, and while other reads load vLLM."""

    config = CHECKS["consistency"]
    repeats = int(config["repeats"])
    request = {"state": control.SYSTEMONE_STATE, "questions": control.SYSTEMONE_QUESTIONS}
    alone = [_systemone(request) for _ in range(repeats)]
    load_level = int(config["load_concurrency"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as single:
        load = single.submit(
            asyncio.run,
            _systemone_burst(_api(), SYSTEMONE["model"], load_level * 8, load_level,
                             f"{run_dir.name}-load", 1024),
        )  # fmt: skip
        loaded = [_systemone(request) for _ in range(repeats)]
        load.result()
    bodies = [body for status, body, _ in alone + loaded if status == 200]
    cases: dict[str, str | None] = {
        "all_answered": None if len(bodies) == 2 * repeats else f"{len(bodies)}/{2 * repeats}"
    }
    spread = 0.0
    for qid in control.SYSTEMONE_QUESTIONS:
        dists = [_distribution(body["answers"][qid]) for body in bodies]
        tops = [max(d, key=d.get) for d in dists]
        cases[f"{qid}_top_stable"] = None if len(set(tops)) == 1 else f"top answers {tops}"
        for key in dists[0] if dists else ():
            values = [d[key] for d in dists]
            spread = max(spread, max(values) - min(values))
    limit = float(config["max_abs_diff"])
    cases["probability_spread"] = None if spread <= limit else f"{spread:.4f} > {limit}"
    print(f"consistency: probability spread {spread:.5f} over {len(bodies)} reads", flush=True)
    return _report(run_dir, "consistency", cases, {"spread": spread})


# --- TypeSafe's SDK --------------------------------------------------------------------------

_SDK_PROGRAM = r"""
import json, sys
from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient
base = sys.argv[1]
state = {"message": "Everything is down and we have a demo with our biggest client at noon."}
questions = {
    "team": Choice(instructions="Which team should handle it?",
                   criteria={"outage": "service down", "billing": "charges", "feature": "how-to"}),
    "urgent": Noul(instructions="Does the customer need a reply within the hour?"),
    "tone": Score(instructions="How upset is the customer?",
                  criteria=["calm", "annoyed", "furious"]),
}
out = {}
for model in (None, "quyet-latest", "quyet-1.0-large-systemone"):
    with TypeSafeClient(api_key="unused", base_url=base, model=model,
                        retry=RetryPolicy(max_retries=0)) as client:
        try:
            r = client.system_one(state=state, questions=questions)
            out[str(model)] = {"model": r.model, "choice": r.choices["team"].choice,
                               "probabilities": r.choices["team"].probabilities,
                               "noul": r.nouls["urgent"].noul, "score": r.scores["tone"].score}
        except Exception as error:
            out[str(model)] = {"error": f"{type(error).__name__}: {error}"[:300]}
with TypeSafeClient(api_key="unused", base_url=base, retry=RetryPolicy(max_retries=0)) as client:
    try:
        client.system_one(state="x", questions={"q": Choice(instructions="Pick one.",
                          criteria={f"o{i}": None for i in range(11)})})
        out["eleven_options"] = "no error"
    except Exception as error:
        out["eleven_options"] = type(error).__name__
    try:
        out["models_list"] = [m.name for m in client.models.list().models]
    except Exception as error:
        out["models_list"] = f"{type(error).__name__}: {error}"[:300]
print(json.dumps(out))
"""


def sdk(run_dir: Path) -> int:
    """TypeSafe's Python SDK, pinned, used the way its docs show, against Kairyu."""

    config = CHECKS["sdk"]
    scratch = _scratch()
    venv = scratch / f"sdk-{config['version']}"
    env = {**os.environ, "UV_CACHE_DIR": str(scratch / "uv-cache")}
    python = venv / "bin/python"
    if not python.exists():
        subprocess.run(["uv", "venv", "-q", "--python", "3.12", str(venv)], check=True, env=env)
        subprocess.run(
            ["uv", "pip", "install", "-q", "--python", str(python),
             f"{config['package']}=={config['version']}"],
            check=True, env=env,
        )  # fmt: skip
    result = subprocess.run(
        [str(python), "-c", _SDK_PROGRAM, _api()], capture_output=True, text=True, timeout=600
    )
    (run_dir / "sdk.log").write_text(result.stdout + result.stderr)
    try:
        out = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return _report(run_dir, "sdk", {"ran": f"exit {result.returncode}: {result.stderr[-300:]}"})
    cases: dict[str, str | None] = {}
    for model in ("None", "quyet-latest", "quyet-1.0-large-systemone"):
        answer = out.get(model, {})
        cases[f"system_one_{model}"] = (
            None
            if "error" not in answer
            and answer.get("choice") in ("outage", "billing", "feature")
            and 0 <= answer.get("noul", -1) <= 1
            and 0 <= answer.get("score", -1) <= 2
            else str(answer)[:300]
        )
    cases["eleven_options_is_bad_request"] = (
        None
        if out.get("eleven_options") == "TypeSafeBadRequestError"
        else out.get("eleven_options")
    )
    # Recorded, not gated: Kairyu's Jev model list (m11 D8) carries no release_date.
    return _report(run_dir, "sdk", cases, {"sdk": out, "version": config["version"]})


# --- throughput and isolation ----------------------------------------------------------------


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


def systemone_serving(run_dir: Path) -> int:
    """Throughput by state length (OpenJev's method): cache-busted states, 3 questions.
    Up to Kairyu's forwarding limit every request is answered; beyond it Kairyu queues
    for queue_wait_s and then sheds with 429, and nothing else may come back."""

    config = CHECKS["systemone_serving"]
    requests = int(config["requests_per_level"])
    asyncio.run(_systemone_burst(_api(), SYSTEMONE["model"], 8, 8, f"{run_dir.name}-warm", 50))
    rows, cases = [], {}
    for tokens in config["state_tokens"]:
        for level in config["concurrency"]:
            name = f"state{tokens}-c{level}"
            results, wall = asyncio.run(
                _systemone_burst(_api(), SYSTEMONE["model"], requests, level,
                                 f"{run_dir.name}-{name}", int(tokens))
            )  # fmt: skip
            latencies = [elapsed for status, elapsed, _ in results if status == 200]
            shed = sum(1 for status, _, _ in results if status == 429)
            if level <= int(SYSTEMONE["max_concurrency"]):
                bad = [(status, err) for status, _, err in results if status != 200 or err]
            else:
                bad = [
                    (status, err) for status, _, err in results if status not in (200, 429) or err
                ]
            row = {
                "row": name, "state_tokens": tokens, "concurrency": level, "requests": requests,
                "ok": len(latencies), "shed_429": shed, "wall_s": wall,
                "requests_per_s": len(latencies) / wall,
                "p50_s": _nearest_rank(latencies, 0.5), "p95_s": _nearest_rank(latencies, 0.95),
            }  # fmt: skip
            rows.append(row)
            print(json.dumps(row), flush=True)
            cases[name] = f"{len(bad)} failed: {bad[:3]}" if bad else None
    (run_dir / "systemone-serving-rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    return _report(run_dir, "systemone-serving", cases)


def systemone_isolation(run_dir: Path) -> int:
    """A System One burst past every limit gets Kairyu's 429, never the adapter's 529, and
    Kairyu, the model server and the adapter answer normally right after it."""

    reads = int(CHECKS["isolation"]["reads"])
    burst, wall = asyncio.run(
        _systemone_burst(_api(), SYSTEMONE["model"], reads, reads, f"{run_dir.name}-burst", 256)
    )
    statuses: dict[int, int] = {}
    for status, _, _ in burst:
        statuses[status] = statuses.get(status, 0) + 1
    ready_status, ready, _ = _json("GET", f"{_api()}/readyz")
    metrics = urllib.request.urlopen(f"{_api()}/metrics", timeout=5).read().decode()
    healthy = control.healthy_replicas(metrics, SERVED)
    status, body, _ = _systemone({"state": "Hello.", "questions": control.SYSTEMONE_QUESTIONS})
    cases = {
        "burst_only_200_or_429": None if set(statuses) <= {200, 429} else f"statuses {statuses}",
        "burst_reaches_kairyu_limit": None if statuses.get(429) else f"no 429: {statuses}",
        "answers_valid": next(
            (error for status, _, error in burst if status == 200 and error), None
        ),
        "ready_after_burst": None
        if ready_status == 200 and healthy == 1
        else f"/readyz {ready_status} {ready}, healthy {healthy}",
        "reads_after_burst": control.systemone_answer_error(body)
        if status == 200
        else f"HTTP {status}",
    }
    (run_dir / "isolation.json").write_text(
        json.dumps({"statuses": statuses, "wall_s": wall, "healthy": healthy}, indent=2) + "\n"
    )
    print(f"burst statuses {statuses} in {wall:.1f} s; healthy model server {healthy}")
    return _report(run_dir, "systemone-isolation", cases)


GATES = {
    "reference": (
        reference,
        "the quyet package itself (GPU, stack down) answers 48 + 231 requests",
    ),
    "attest": (attest, "images, checkpoint re-hash, vLLM, calibration, no public chat"),
    "systemone": (systemone, "Kairyu's answers vs the official package; names; error shapes"),
    "jevbench": (jevbench, "JevBench's runner on Kairyu, scored beside the official answers"),
    "fanout": (fanout, "1/8/32 questions per call: the state read once, questions in parallel"),
    "consistency": (consistency, "the same request answers the same way, alone and under load"),
    "sdk": (sdk, "TypeSafe's Python SDK against Kairyu: typed answers and a bad request"),
    "systemone-serving": (systemone_serving, "req/s and p50/p95 by state length and concurrency"),
    "systemone-isolation": (systemone_isolation, "a 640-read burst gets 429s, never 529"),
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
    parser.add_argument("--no-start", action="store_true", help="do not run control.py up")
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
