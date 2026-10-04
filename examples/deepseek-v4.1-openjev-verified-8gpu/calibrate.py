#!/usr/bin/env python3
"""Measure the guarantee on InFoBench expert labels (Q1, Q2) and choose tau_hi.

The guarantee is the acceptance read (VCO-D15 item 7): Jev reads the
per-point results, the request and the answer and decides whether the
answer can be adopted. InFoBench's expert annotation labels every
decomposed requirement of a model answer; an answer is correct when every
label is yes. Each answer is judged through the example's production
checklist code: DeepSeek turns each question into a point statement and
runs the example's history role, then OpenJev reads the coverage questions
and the acceptance question, as in serving.

tau_hi only marks the points a repair should fix, but the acceptance read
sees those marks, so the guarantee is measured for every candidate tau_hi:
  Q1 error  guaranteed answers (acceptance >= tau_accept) that miss a label
  Q2 miss   correct answers that get no guarantee
tau_hi is chosen on the calibration half as the candidate with the most
correct answers guaranteed net of wrong ones guaranteed (a tie goes to the
lower error); the held-out half
is reported for it and decides the gate: error at most 15.2 % and miss at
most 36 % (the held-out values measured when the owner set tau_accept 0.99).

Usage: ./verify.sh calibrate   (after ./run.sh up)
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import csv
import dataclasses
import functools
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


TAU_HI_CANDIDATES = (0.5, 0.9, 0.99, 0.99894, 0.999733)
MAX_ERROR = 0.152
MAX_MISS = 0.36


def _with_tau_hi(config: ChecklistConfig, tau_hi: float) -> ChecklistConfig:
    return dataclasses.replace(config, threshold=tau_hi)


def accept_at(sample: dict, tau_hi: float) -> float:
    """P(accepted) with the coverage marks of ``tau_hi``."""

    config = _with_tau_hi(_requirement_checklist(), tau_hi)
    assert config.acceptance is not None
    verdict = _read(config, sample)
    assert verdict.acceptance is not None
    return verdict.acceptance


def guarantee_stats(rows: list[dict], tau_accept: float, confidence: float) -> dict:
    """Q1 (error among guaranteed) and Q2 (miss among correct) for ``rows``."""

    guaranteed = [row for row in rows if row["acceptance"] >= tau_accept]
    wrong = [row for row in guaranteed if 0 in row["labels"]]
    correct = [row for row in rows if 0 not in row["labels"]]
    missed = [row for row in correct if row["acceptance"] < tau_accept]
    return {
        "responses": len(rows),
        "correct": len(correct),
        "guaranteed": len(guaranteed),
        "guaranteed_wrong": len(wrong),
        "error_rate": round(len(wrong) / len(guaranteed), 4) if guaranteed else None,
        "error_upper_bound": round(
            clopper_pearson_upper(len(wrong), len(guaranteed), confidence), 4
        ),
        "correct_missed": len(missed),
        "miss_rate": round(len(missed) / len(correct), 4) if correct else None,
        "net_correct_guaranteed": len(guaranteed) - 2 * len(wrong),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    calibration = SPEC["calibration"]
    directory = control.environment_storage() / "calibration"
    rows = samples(download(directory))
    env = control._compose_env()
    l1_url = f"http://127.0.0.1:{env['DEEPSEEK_L1_PORT']}"
    roles = _roles()
    spec = yaml.safe_load((HERE / "verified-always.yaml").read_text())
    checklist = next(role for role in spec["roles"] if role["name"] == "checklist")
    tau_accept = float(checklist["checklist"]["acceptance"]["threshold"])
    confidence = float(calibration["confidence"])

    # Point statements and summaries are made once per build of the prompts.
    prompts = hashlib.sha256((_STATEMENT_PROMPT + roles["history"]["prompt"]).encode()).hexdigest()[
        :12
    ]
    prepared_cache = directory / f"prepared-{prompts}.jsonl"
    prepared = {}
    if prepared_cache.is_file():
        for line in prepared_cache.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prepared[(row["id"], row["model"])] = row
    pending = [row for row in rows if (row["id"], row["model"]) not in prepared]
    print(f"{len(rows)} labelled responses; {len(pending)} to prepare", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(lambda sample: prepare(sample, l1_url, roles), pending):
            prepared[(row["id"], row["model"])] = row
            with prepared_cache.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    ready = [prepared[(row["id"], row["model"])] for row in rows]

    instructions = sorted({row["id"] for row in ready})
    random.Random(calibration["split_seed"]).shuffle(instructions)
    calibration_ids = set(instructions[: len(instructions) // 2])

    report: dict = {"tau_accept": tau_accept, "candidates": {}}
    for tau_hi in TAU_HI_CANDIDATES:
        cache = directory / f"acceptance-{prompts}-{tau_hi}.jsonl"
        done = {}
        if cache.is_file():
            for line in cache.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                done[(row["id"], row["model"])] = row["acceptance"]
        todo = [row for row in ready if (row["id"], row["model"]) not in done]
        print(f"tau_hi {tau_hi}: {len(todo)} acceptance reads", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for row, value in zip(
                todo, pool.map(functools.partial(accept_at, tau_hi=tau_hi), todo), strict=True
            ):
                done[(row["id"], row["model"])] = value
                with cache.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps({"id": row["id"], "model": row["model"], "acceptance": value})
                        + "\n"
                    )
        scored = [{**row, "acceptance": done[(row["id"], row["model"])]} for row in ready]
        report["candidates"][str(tau_hi)] = {
            "calibration": guarantee_stats(
                [row for row in scored if row["id"] in calibration_ids], tau_accept, confidence
            ),
            "holdout": guarantee_stats(
                [row for row in scored if row["id"] not in calibration_ids], tau_accept, confidence
            ),
        }
    # Most correct answers guaranteed net of wrong ones; a tie goes to the
    # candidate with the lower error.
    chosen = max(
        TAU_HI_CANDIDATES,
        key=lambda tau: (
            report["candidates"][str(tau)]["calibration"]["net_correct_guaranteed"],
            -(report["candidates"][str(tau)]["calibration"]["error_rate"] or 1.0),
        ),
    )
    holdout = report["candidates"][str(chosen)]["holdout"]
    report["chosen_tau_hi"] = chosen
    report["passed"] = (
        holdout["error_rate"] is not None
        and holdout["error_rate"] <= MAX_ERROR
        and holdout["miss_rate"] is not None
        and holdout["miss_rate"] <= MAX_MISS
    )
    (directory / "guarantee.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(
        f"calibrate: {'PASS' if report['passed'] else 'FAIL'} (tau_hi {chosen}: held-out "
        f"error {holdout['error_rate']} <= {MAX_ERROR}, miss {holdout['miss_rate']} <= {MAX_MISS})"
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
