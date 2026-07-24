"""Tracked compatibility layer for EvalScope 1.8.1 GPQA semantics.

This module intentionally has no EvalScope or dataset-client import.  It mirrors
the pinned adapter's text preprocessing, sequential Python-MT shuffle, prompt,
answer extraction, and exact-match Accuracy behavior for an offline synthetic
smoke.  The source-index target fix is declared as a protocol compatibility
patch because EvalScope 1.8.1 uses text equality, which is ambiguous for
duplicate choices.

Upstream: modelscope/evalscope@fce1d21391dc2d7b45c9cf0edb9b9e40d526aed3
License: Apache-2.0
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROMPT_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'ANSWER: [LETTER]' (without "
    "quotes) where [LETTER] is one of {letters}. Think step by step before "
    "answering.\n\n{question}\n\n{choices}"
)
PROMPT_SHA256 = "f759b2cf67b53500e906bc321c88285351eadd9918a8479505626828f04e6aba"
REQUIRED_FIELDS = (
    "Record ID",
    "Question",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
    "Correct Answer",
)
MAX_DATASET_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class GPQARecord:
    record_id: str
    question: str
    incorrect_answers: tuple[str | None, str | None, str | None]
    correct_answer: str | None


@dataclass(frozen=True, slots=True)
class PreparedGPQARecord:
    record_id: str
    ordinal: int
    prompt: str
    target: str
    choice_permutation: tuple[int, ...]
    input_sha256: str


def preprocess_choice(value: str | None) -> str:
    """Apply EvalScope v1.8.1 GPQA's exact choice normalization order."""

    if value is None:
        return " "
    text = value.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def render_prompt(question: str, choices: tuple[str, ...]) -> str:
    letters = ",".join(chr(65 + index) for index in range(len(choices)))
    choices_text = "\n".join(f"{chr(65 + index)}) {choice}" for index, choice in enumerate(choices))
    return PROMPT_TEMPLATE.format(
        letters=letters,
        question=question,
        choices=choices_text,
    )


def prepare_records(
    records: tuple[GPQARecord, ...],
    *,
    seed: int = 42,
) -> tuple[PreparedGPQARecord, ...]:
    """Precompute choice permutations sequentially in canonical record order."""

    generator = random.Random(seed)
    prepared: list[PreparedGPQARecord] = []
    for ordinal, record in enumerate(records):
        source_choices = tuple(
            preprocess_choice(choice)
            for choice in (*record.incorrect_answers, record.correct_answer)
        )
        permutation = list(range(len(source_choices)))
        generator.shuffle(permutation)
        choices = tuple(source_choices[source_index] for source_index in permutation)
        correct_position = permutation.index(3)
        target = chr(65 + correct_position)
        prompt = render_prompt(record.question, choices)
        input_payload = {
            "choice_permutation": permutation,
            "correct_source_index": 3,
            "item_id": record.record_id,
            "ordinal": ordinal,
            "prompt": prompt,
            "source_choices": [
                {"source_index": index, "text": choice}
                for index, choice in enumerate(source_choices)
            ],
            "target": target,
        }
        input_sha256 = hashlib.sha256(_canonical_json(input_payload)).hexdigest()
        prepared.append(
            PreparedGPQARecord(
                record_id=record.record_id,
                ordinal=ordinal,
                prompt=prompt,
                target=target,
                choice_permutation=tuple(permutation),
                input_sha256=input_sha256,
            )
        )
    return tuple(prepared)


def extract_answer(completion: str, *, choice_count: int = 4) -> str | None:
    """Match EvalScope v1.8.1's permissive single-answer extraction."""

    match = re.search(
        r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)",
        completion,
        flags=re.MULTILINE,
    )
    if match is None:
        match = re.search(
            r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)",
            completion,
        )
    if match is None:
        for character in reversed(completion):
            if character.isupper():
                return character
        return None

    answer = match.group(1).strip().rstrip(".")
    allowed = {chr(65 + index) for index in range(choice_count)}
    return answer if answer in allowed else None


def load_records(path: Path) -> tuple[GPQARecord, ...]:
    """Read an explicitly approved local JSONL or CSV snapshot."""

    content = _read_snapshot_bytes(path)
    return _parse_snapshot_bytes(path, content)


def load_records_with_sha256(
    path: Path,
) -> tuple[tuple[GPQARecord, ...], str]:
    """Bind parsing and SHA-256 to one bounded, no-follow byte snapshot."""

    content = _read_snapshot_bytes(path)
    return _parse_snapshot_bytes(path, content), hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(_read_snapshot_bytes(path)).hexdigest()


def manifest_sha256(records: tuple[PreparedGPQARecord, ...]) -> str:
    manifest = [
        {"input_sha256": record.input_sha256, "item_id": record.record_id} for record in records
    ]
    return hashlib.sha256(_canonical_json(manifest)).hexdigest()


def _parse_snapshot_bytes(
    path: Path,
    content: bytes,
) -> tuple[GPQARecord, ...]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("GPQA local snapshot must be valid UTF-8") from exc
    suffix = path.suffix.casefold()
    if suffix == ".jsonl":
        rows = _load_jsonl(text)
    elif suffix == ".csv":
        rows = _load_csv(text)
    else:
        raise ValueError("GPQA local snapshot must be .jsonl or .csv")
    records = tuple(_parse_record(row, ordinal) for ordinal, row in enumerate(rows))
    if not records:
        raise ValueError("GPQA local snapshot is empty")
    identifiers = [record.record_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("GPQA local snapshot contains duplicate Record ID values")
    return records


def _reject_duplicate_json_fields(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError("GPQA JSONL contains duplicate object field names")
        parsed[key] = value
    return parsed


def _load_jsonl(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(io.StringIO(text), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(
                raw_line,
                object_pairs_hook=_reject_duplicate_json_fields,
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"GPQA JSONL contains invalid JSON at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"GPQA JSONL row {line_number} is not an object")
        rows.append(row)
    return rows


def _load_csv(text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fieldnames = reader.fieldnames
    if fieldnames is None:
        return []
    if len(set(fieldnames)) != len(fieldnames):
        raise ValueError("GPQA CSV contains duplicate header field names")
    return [dict(row) for row in reader]


def _parse_record(row: dict[str, Any], ordinal: int) -> GPQARecord:
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"GPQA row {ordinal} is missing {len(missing)} required field(s)")
    record_id = row["Record ID"]
    question = row["Question"]
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError(f"GPQA row {ordinal} has an invalid Record ID")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"GPQA row {ordinal} has an invalid Question")
    answer_values = (
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
        row["Correct Answer"],
    )
    if any(value is not None and not isinstance(value, str) for value in answer_values):
        raise ValueError(f"GPQA row {ordinal} has a non-text answer")
    return GPQARecord(
        record_id=record_id,
        question=question,
        incorrect_answers=answer_values[:3],
        correct_answer=answer_values[3],
    )


def _read_snapshot_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("GPQA local snapshot must be a regular non-symlink file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("GPQA local snapshot must be a regular non-symlink file")
        if metadata.st_size > MAX_DATASET_BYTES:
            raise ValueError("GPQA local snapshot exceeds the configured size limit")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            content = handle.read(MAX_DATASET_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(content) > MAX_DATASET_BYTES:
        raise ValueError("GPQA local snapshot exceeds the configured size limit")
    return content


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


__all__ = [
    "GPQARecord",
    "MAX_DATASET_BYTES",
    "PROMPT_SHA256",
    "PROMPT_TEMPLATE",
    "PreparedGPQARecord",
    "extract_answer",
    "load_records",
    "load_records_with_sha256",
    "manifest_sha256",
    "prepare_records",
    "preprocess_choice",
    "render_prompt",
    "sha256_file",
]
