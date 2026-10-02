#!/usr/bin/env python3
"""Calibrate the Conductor threshold tau_hi on InFoBench expert labels.

InFoBench's expert annotation pairs model answers with decomposed yes/no
requirements and a human pass/fail label for each. Every requirement is
judged through this example's own production path (VCO-D15): DeepSeek turns
the question into a point statement (like the extractor writes) and runs the
example's history role, and OpenJev reads the example's coverage question for
that point over the request, the history summary and the answer, through the
same Kairyu checklist code as serving. tau_hi is the smallest
threshold whose accepted requirements have a one-sided 95 % Clopper-Pearson
upper bound on the violation rate <= alpha on the calibration half; the
held-out half is reported unchanged.

tau_accept (VCO-D15 item 7) is chosen the same way for the acceptance read,
one per response, labelled acceptable when every requirement label is yes;
the checklist runs whole (coverage then acceptance), as in serving.

Usage: ./verify.sh calibrate   (after ./run.sh up)
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import csv
import dataclasses
import hashlib
import json
import math
import random
import re
import sys
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import control  # noqa: E402

from kairyu.dsl.loader import load_spec, role_spec  # noqa: E402
from kairyu.engine.systemone import HTTPSystemOneBackend  # noqa: E402
from kairyu.entrypoints.server.chat_service import (  # noqa: E402
    validate_orchestration_chat_input,
)
from kairyu.entrypoints.server.protocol import ChatCompletionRequest  # noqa: E402
from kairyu.orchestration.checklist import ChecklistConfig  # noqa: E402
from kairyu.orchestration.checklist import judge as judge_checklist  # noqa: E402
from kairyu.orchestration.request import conversation_text  # noqa: E402

SPEC = control.SPEC
# InFoBench expert annotation (Easy and Hard subsets), Google Drive file ids
# from the official repository's "Generation and Annotation" folder.
SOURCES = {
    "161wLlIQzuHofbgkVvvSIn8cH5y6f4jlk": (
        "ac2b5b342188e24b6e781ad7c7a595945f8dbd5ef15d021c2056f11be5bb3da6"
    ),
    "1IKIRSLR3aPnBLhTd99nO09QQ72qiyKZc": (
        "d3e4c9f2220443647118e9a80ffb9519df7c43979ac5c3ca3afacc7424c01d54"
    ),
}
OPENJEV_URLS = ("http://127.0.0.1:8015", "http://127.0.0.1:8016")
ANNOTATED_MODELS = ("gpt-3.5-turbo", "gpt-4", "claude-v1", "alpaca-7b", "vicuna-13b")


def _post(url: str, payload: dict, timeout_s: float = 1800) -> dict:
    return control.post_json(url, payload, timeout_s=timeout_s)


def download(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for file_id, digest in SOURCES.items():
        path = directory / f"{file_id}.csv"
        if not path.is_file():
            url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
            with urllib.request.urlopen(url, timeout=120) as response:
                path.write_bytes(response.read())
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"{path.name} has sha256 {actual}, expected {digest}")
        paths.append(path)
    return paths


_NUMBERED = re.compile(r"^\s*\d+\.\s*")
_CATEGORY = re.compile(r"\s*\([^()]*\)\s*$")


def samples(paths: list[Path]) -> list[dict]:
    """One sample per (instruction, model) with a complete expert label row."""

    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if not row.get("id"):
                    continue
                questions = [
                    _CATEGORY.sub("", _NUMBERED.sub("", line)).strip()
                    for line in row["decomposed_questions"].splitlines()
                    if line.strip()
                ]
                for model in ANNOTATED_MODELS:
                    output = (row.get(model) or "").strip()
                    labels = [
                        part
                        for part in (row.get(f"{model}-annotation") or "").strip().split(".")
                        if part.strip()
                    ]
                    if not output or len(labels) != len(questions) or not questions:
                        continue
                    if set(labels) - {"0", "1"}:
                        continue
                    rows.append(
                        {
                            "id": row["id"],
                            "model": model,
                            "request": "\n\n".join(
                                part for part in (row["instruction"], row.get("input", "")) if part
                            ),
                            "answer": output,
                            "questions": questions,
                            "labels": [int(label) for label in labels],
                        }
                    )
    return rows


def _roles() -> dict[str, dict]:
    spec = yaml.safe_load((HERE / "verified.yaml").read_text(encoding="utf-8"))
    return {role["name"]: role for role in spec["roles"]}


def _query(request: str) -> str:
    """The exact L2 {query} Kairyu renders for a one-turn chat request."""

    chat = ChatCompletionRequest(
        model=SPEC["public_models"][1], messages=[{"role": "user", "content": request}]
    )
    return validate_orchestration_chat_input(chat).prompt


def _deepseek(l1_url: str, prompt: str, *, response_format: dict | None = None) -> str:
    payload: dict = {
        "model": SPEC["deepseek"]["served_name"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16384,
        "temperature": 0.0,
        # chat mode, like the serving state builder (V4.1 thinks otherwise)
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if response_format is not None:
        payload["response_format"] = response_format
    body = _post(f"{l1_url}/v1/chat/completions", payload)
    return body["choices"][0]["message"]["content"]


_STATEMENT_PROMPT = (
    "Rewrite this yes/no question about a generated answer as one declarative "
    "condition the ANSWER must satisfy, in the same language. Output only the "
    "condition.\nQuestion: {question}"
)


def prepare(sample: dict, l1_url: str, roles: dict[str, dict]) -> dict:
    """Point statements and the history summary for one sample."""

    statements = [
        _deepseek(l1_url, _STATEMENT_PROMPT.format(question=question)).strip()
        for question in sample["questions"]
    ]
    conversation = conversation_text(_query(sample["request"]))
    history = _deepseek(
        l1_url, roles["history"]["prompt"].format_map({"conversation": conversation})
    ).strip()
    return {**sample, "statements": statements, "history": history}


def _requirement_checklist() -> ChecklistConfig:
    """The production coverage checklist, reduced to its explicit-point question.

    The same Kairyu code builds the Jev state and questions as in serving;
    InFoBench's labelled requirements stand in for the explicit points.
    """

    spec = load_spec(HERE / "verified-always.yaml")
    node = next(role for role in spec.roles if role.name == "checklist")
    config = role_spec(node).checklist
    assert config is not None
    point = next(
        question
        for question in config.questions
        if question.foreach is not None and question.foreach.role == "extract"
    )
    return dataclasses.replace(config, questions=(point,))


def _outputs(sample: dict) -> tuple[list[dict], dict[str, str]]:
    requirements = [
        {"id": f"Q{index}", "point": text}
        for index, text in enumerate(sample["statements"], start=1)
    ]
    outputs = {
        "extract": json.dumps({"points": requirements}, ensure_ascii=False),
        "answer": sample["answer"],
        "history": sample["history"],
    }
    return requirements, outputs


def _read(config: ChecklistConfig, sample: dict):
    _requirements, outputs = _outputs(sample)

    async def read():
        # The serving path: the L2 reads both OpenJev replicas directly.
        backend = HTTPSystemOneBackend(
            base_urls=OPENJEV_URLS, upstream_model=SPEC["systemone"]["model"], timeout_s=600
        )
        try:
            return await judge_checklist(config, backend, outputs, _query(sample["request"]))
        finally:
            await backend.shutdown()

    return asyncio.run(read())


def judge(sample: dict, api_url: str, roles: dict[str, dict]) -> list[float]:
    """P(requirement satisfied) per requirement, via the production checklist."""

    del roles
    config = dataclasses.replace(_requirement_checklist(), acceptance=None)
    requirements, _ = _outputs(sample)
    verdict = _read(config, sample)
    by_id = {item.id: item.p for item in verdict.items}
    return [by_id[item["id"]] for item in requirements]


def accept(sample: dict) -> float:
    """P(accepted as the reply), via the production checklist with acceptance."""

    config = _requirement_checklist()
    assert config.acceptance is not None
    verdict = _read(config, sample)
    assert verdict.acceptance is not None
    return verdict.acceptance


def _binomial_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson_upper(k: int, n: int, confidence: float) -> float:
    """One-sided upper confidence bound of a binomial proportion."""

    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    low, high = k / n, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        if _binomial_cdf(k, n, middle) > 1 - confidence:
            low = middle
        else:
            high = middle
    return high


def accepted_stats(pairs: list[tuple[float, int]], tau: float, confidence: float) -> dict:
    accepted = [label for p, label in pairs if p >= tau]
    violations = accepted.count(0)
    return {
        "tau": tau,
        "accepted": len(accepted),
        "violations": violations,
        "violation_rate": violations / len(accepted) if accepted else None,
        "upper_bound": clopper_pearson_upper(violations, len(accepted), confidence),
        "acceptance": len(accepted) / len(pairs) if pairs else 0.0,
    }


def choose_tau(pairs: list[tuple[float, int]], alpha: float, confidence: float) -> dict:
    for tau in sorted({p for p, _label in pairs}):
        stats = accepted_stats(pairs, tau, confidence)
        if stats["accepted"] and stats["upper_bound"] <= alpha:
            return stats
    return {"tau": None, "accepted": 0, "violations": 0, "upper_bound": None}


def response_level(rows: list[dict], tau: float) -> dict:
    """A response passes when all its requirements pass; violated if any label is 0."""

    passed = [row for row in rows if all(p >= tau for p in row["p"])]
    violated = [row for row in passed if 0 in row["labels"]]
    return {"responses": len(rows), "passed": len(passed), "violated": len(violated)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    calibration = SPEC["calibration"]
    directory = control.environment_storage() / "calibration"
    rows = samples(download(directory))
    env = control._compose_env()
    api_url = f"http://127.0.0.1:{env['API_PORT']}"
    l1_url = f"http://127.0.0.1:{env['DEEPSEEK_L1_PORT']}"
    roles = _roles()
    # VCO-D15 question and state; reads of earlier formats are not reused.
    cache = directory / "judged-coverage-v2.jsonl"
    done = {}
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[(row["id"], row["model"])] = row

    def work(sample: dict) -> dict:
        key = (sample["id"], sample["model"])
        if key in done:
            return done[key]
        prepared = prepare(sample, l1_url, roles)
        return {**prepared, "p": judge(prepared, api_url, roles)}

    pending = [row for row in rows if (row["id"], row["model"]) not in done]
    print(f"{len(rows)} labelled responses; {len(pending)} to judge", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(work, pending):
            done[(row["id"], row["model"])] = row
            with cache.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    judged = [done[(row["id"], row["model"])] for row in rows]
    instructions = sorted({row["id"] for row in judged})
    random.Random(calibration["split_seed"]).shuffle(instructions)
    calibration_ids = set(instructions[: len(instructions) // 2])
    halves = {
        "calibration": [row for row in judged if row["id"] in calibration_ids],
        "holdout": [row for row in judged if row["id"] not in calibration_ids],
    }
    pairs = {
        name: [(p, label) for row in part for p, label in zip(row["p"], row["labels"], strict=True)]
        for name, part in halves.items()
    }
    alpha, confidence = float(calibration["alpha"]), float(calibration["confidence"])
    chosen = choose_tau(pairs["calibration"], alpha, confidence)
    report = {
        "alpha": alpha,
        "confidence": confidence,
        "instructions": {name: len({row["id"] for row in part}) for name, part in halves.items()},
        "labels": {name: len(values) for name, values in pairs.items()},
        "violations": {
            name: [label for _p, label in values].count(0) for name, values in pairs.items()
        },
        "calibration": chosen,
    }
    if chosen["tau"] is not None:
        report["holdout"] = accepted_stats(pairs["holdout"], chosen["tau"], confidence)
        report["holdout_responses"] = response_level(halves["holdout"], chosen["tau"])
    (directory / "tau.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    accept_cache = directory / "judged-acceptance-v1.jsonl"
    accepted = {}
    if accept_cache.is_file():
        for line in accept_cache.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            accepted[(row["id"], row["model"])] = row["acceptance"]
    pending = [row for row in judged if (row["id"], row["model"]) not in accepted]
    print(f"{len(pending)} responses to read for acceptance", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, value in zip(pending, pool.map(accept, pending), strict=True):
            accepted[(row["id"], row["model"])] = value
            with accept_cache.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps({"id": row["id"], "model": row["model"], "acceptance": value})
                    + "\n"
                )
    responses = {
        name: [(accepted[(row["id"], row["model"])], int(0 not in row["labels"])) for row in part]
        for name, part in halves.items()
    }
    chosen = choose_tau(responses["calibration"], alpha, confidence)
    acceptance = {
        "responses": {name: len(values) for name, values in responses.items()},
        "unacceptable": {
            name: [label for _p, label in values].count(0) for name, values in responses.items()
        },
        "calibration": chosen,
    }
    if chosen["tau"] is not None:
        acceptance["holdout"] = accepted_stats(responses["holdout"], chosen["tau"], confidence)
    (directory / "tau-accept.json").write_text(json.dumps(acceptance, indent=2), encoding="utf-8")
    print(json.dumps(acceptance, indent=2))


if __name__ == "__main__":
    main()
