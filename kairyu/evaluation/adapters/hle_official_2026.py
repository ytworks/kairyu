"""Pinned compatibility helpers for the CAIS Humanity's Last Exam evaluator.

The source semantics mirrored here were reviewed at
``centerforaisafety/simple-evals@8e53435ff2985b0f32ea7ceb7e92c3a175f2c0f3``.
This module intentionally has no network, dataset-library, or model-client
dependency.  Gated HLE data must be supplied as an approved local JSONL
snapshot; the helpers never fetch or redistribute it.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

UPSTREAM_REPOSITORY = "https://github.com/centerforaisafety/simple-evals"
UPSTREAM_COMMIT = "8e53435ff2985b0f32ea7ceb7e92c3a175f2c0f3"
UPSTREAM_SOURCE_SHA256 = "d276f725ecc5ea2c08f73e161f97760881332c3d33d20b197c2ffbac5f55edfe"
MAX_ATTEMPTS = 5

SYSTEM_PROMPT = """Your response should be in the following format:
Explanation: {your explanation for your answer choice}
Answer: {your chosen answer}
Confidence: {your confidence score between 0% and 100% for your answer}"""

JUDGE_PROMPT = (
    "Judge whether the following [response] to [question] is correct or not based "
    "on the precise and unambiguous [correct_answer] below.\n\n"
    "[question]: {question}\n\n"
    "[response]: {response}\n\n"
    "[correct_answer]: {correct_answer}\n\n"
    "Your judgement must be in XML format with the following tags:\n\n"
    "<extracted_final_answer>The final exact answer extracted from the [response]. "
    "Put \u0027None\u0027 if there is no exact, final answer to extract from the response."
    "</extracted_final_answer>\n\n"
    "<reasoning>Explain why the extracted_final_answer is correct or incorrect based "
    "on [correct_answer], focusing only on if there are meaningful differences "
    "between [correct_answer] and the extracted_final_answer. Do not comment on any "
    "background to the problem, do not attempt to solve the problem, do not argue "
    "for any answer different than [correct_answer], focus only on whether the "
    "answers match.</reasoning>\n\n"
    "<correct>Answer \u0027yes\u0027 if extracted_final_answer matches the [correct_answer] "
    "given above, or is within a small margin of error for numerical problems. "
    "Answer \u0027no\u0027 otherwise, i.e. if there is any inconsistency, ambiguity, "
    "non-equivalency, or if the extracted answer is incorrect.</correct>\n\n"
    "<confidence>The extracted confidence score between 0 and 100 from [response]. "
    "Put 100 if there is no confidence score available.</confidence>"
)

SYSTEM_PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
JUDGE_PROMPT_SHA256 = hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest()
PROMPT_BUNDLE_SHA256 = hashlib.sha256(
    json.dumps(
        {"judge_prompt": JUDGE_PROMPT, "system_prompt": SYSTEM_PROMPT},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

_MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024
_MAX_INLINE_IMAGE_BYTES = 20 * 1024 * 1024
_DATA_IMAGE = re.compile(r"data:(image/(?:png|jpeg|webp|gif));base64,([A-Za-z0-9+/]*={0,2})\Z")
_EXTRACTED_ANSWER = re.compile(
    r"<extracted_final_answer>(.*?)</extracted_final_answer>",
    re.DOTALL | re.IGNORECASE,
)
_REASONING = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)
_CORRECT = re.compile(r"<correct>(yes|no)</correct>", re.DOTALL | re.IGNORECASE)
_CONFIDENCE = re.compile(r"<confidence>(\d+)</confidence>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class HLERecord:
    """Minimal official HLE fields needed by the target and judge calls."""

    record_id: str
    question: str
    answer: str
    image: str | None
    answer_type: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedHLERecord:
    """Validated item with a stable input identity and no provider state."""

    record_id: str
    question: str
    answer: str
    image: str | None
    answer_type: str | None
    input_sha256: str


@dataclass(frozen=True, slots=True)
class TargetPrompt:
    """Provider-neutral representation of the official target messages."""

    system_prompt: str
    question: str
    image_url: str | None


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """The four fields parsed by the upstream judge-response regular expressions."""

    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int


@dataclass(frozen=True, slots=True)
class ScoredPrediction:
    """One successfully parsed judge response used by aggregate scoring."""

    correct: bool
    confidence: int


@dataclass(frozen=True, slots=True)
class HLEMetrics:
    """Exact aggregate fields emitted by the reviewed upstream evaluator."""

    accuracy: float
    confidence_interval: float
    accuracy_success_only: float
    confidence_interval_success_only: float
    calibration_error: float
    evaluated_questions: int
    failed_questions: int
    total_questions: int


def sha256_file(path: Path) -> str:
    """Hash one regular, non-symlink snapshot after a bounded single read."""

    payload = _read_snapshot_bytes(path)
    return hashlib.sha256(payload).hexdigest()


def load_records(path: Path) -> tuple[HLERecord, ...]:
    """Load a local JSONL snapshot while rejecting ambiguous JSON objects."""

    payload = _read_snapshot_bytes(path)
    return _decode_records(payload)


def load_records_with_sha256(path: Path) -> tuple[tuple[HLERecord, ...], str]:
    """Decode and hash the same immutable byte snapshot."""

    payload = _read_snapshot_bytes(path)
    return _decode_records(payload), hashlib.sha256(payload).hexdigest()


def prepare_records(records: Sequence[HLERecord]) -> tuple[PreparedHLERecord, ...]:
    """Validate image payloads and derive stable per-item hashes."""

    if not records:
        raise ValueError("HLE snapshot must contain at least one record")
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("HLE record IDs must be unique")

    prepared: list[PreparedHLERecord] = []
    for record in records:
        if record.image is not None:
            validate_inline_image(record.image)
        identity = {
            "answer": record.answer,
            "answer_type": record.answer_type,
            "id": record.record_id,
            "image": record.image,
            "question": record.question,
        }
        prepared.append(
            PreparedHLERecord(
                record_id=record.record_id,
                question=record.question,
                answer=record.answer,
                image=record.image,
                answer_type=record.answer_type,
                input_sha256=hashlib.sha256(_canonical_json(identity)).hexdigest(),
            )
        )
    return tuple(prepared)


def manifest_sha256(records: Sequence[PreparedHLERecord]) -> str:
    """Hash ordered source IDs and input hashes for resume identity."""

    return hashlib.sha256(
        _canonical_json(
            [
                {
                    "id": record.record_id,
                    "input_sha256": record.input_sha256,
                    "ordinal": ordinal,
                }
                for ordinal, record in enumerate(records)
            ]
        )
    ).hexdigest()


def target_prompt(record: PreparedHLERecord) -> TargetPrompt:
    """Return the system/user content used by the reviewed target formatter."""

    return TargetPrompt(
        system_prompt=SYSTEM_PROMPT,
        question=record.question,
        image_url=record.image,
    )


def build_judge_prompt(*, question: str, response: str, correct_answer: str) -> str:
    """Substitute the official HLE judge prompt without normalization."""

    return JUDGE_PROMPT.format(
        question=question,
        response=response,
        correct_answer=correct_answer,
    )


def parse_judge_response(response: str) -> JudgeVerdict | None:
    """Mirror the upstream four independent XML-tag regular expressions."""

    extracted_match = _EXTRACTED_ANSWER.search(response)
    reasoning_match = _REASONING.search(response)
    correct_match = _CORRECT.search(response)
    confidence_match = _CONFIDENCE.search(response)
    if not all((extracted_match, reasoning_match, correct_match, confidence_match)):
        return None

    assert extracted_match is not None
    assert reasoning_match is not None
    assert correct_match is not None
    assert confidence_match is not None
    return JudgeVerdict(
        extracted_final_answer=extracted_match.group(1).strip(),
        reasoning=reasoning_match.group(1).strip(),
        correct=correct_match.group(1).lower(),  # type: ignore[arg-type]
        confidence=int(confidence_match.group(1)),
    )


def calibration_error(
    confidence: Sequence[float],
    correct: Sequence[bool | int | float],
    *,
    p: Literal["1", "2", "infty", "infinity", "max"] = "2",
    beta: int = 100,
) -> float:
    """Mirror the reviewed upstream ``calib_err`` binning semantics.

    The upstream implementation excludes its final bin when ``n >= beta``;
    that behavior is intentionally preserved rather than silently corrected.
    A deterministic stable sort replaces NumPy's unspecified tie ordering.
    This difference matters only when equal-confidence rows straddle a bin
    boundary and is recorded as a compatibility patch in the profile.
    """

    if len(confidence) != len(correct):
        raise ValueError("confidence and correctness lengths differ")
    if type(beta) is not int or beta <= 0:
        raise ValueError("beta must be a positive integer")
    if p not in {"1", "2", "infty", "infinity", "max"}:
        raise ValueError("p must be '1', '2', or 'infty'")
    if any(not math.isfinite(float(value)) for value in confidence):
        raise ValueError("confidence values must be finite")
    if any(not math.isfinite(float(value)) for value in correct):
        raise ValueError("correctness values must be finite")

    ordered = sorted(
        zip(
            (float(value) for value in confidence),
            (float(value) for value in correct),
            strict=True,
        ),
        key=lambda pair: pair[0],
    )
    count = len(ordered)
    if count < beta:
        if count:
            mean_confidence = sum(pair[0] for pair in ordered) / count
            mean_correct = sum(pair[1] for pair in ordered) / count
            return abs(mean_confidence - mean_correct)
        return 0.0

    bins = [[index * beta, (index + 1) * beta] for index in range(count // beta)]
    bins[-1] = [bins[-1][0], count]
    error = 0.0
    for start, end in bins[:-1]:
        rows = ordered[start:end]
        if not rows:
            continue
        mean_confidence = sum(pair[0] for pair in rows) / len(rows)
        mean_correct = sum(pair[1] for pair in rows) / len(rows)
        difference = abs(mean_confidence - mean_correct)
        if p == "2":
            error += len(rows) / count * difference**2
        elif p == "1":
            error += len(rows) / count * difference
        else:
            error = max(error, difference)
    return math.sqrt(error) if p == "2" else error


def compute_metrics(
    predictions: Sequence[ScoredPrediction],
    *,
    total_questions: int,
    num_failed: int,
) -> HLEMetrics:
    """Compute upstream overall/success-only accuracy, Wald CI, and CE."""

    if type(total_questions) is not int or total_questions < 0:
        raise ValueError("total_questions must be a non-negative integer")
    if type(num_failed) is not int or num_failed < 0:
        raise ValueError("num_failed must be a non-negative integer")
    if len(predictions) + num_failed > total_questions:
        raise ValueError("evaluated and failed counts exceed total questions")

    num_evaluated = len(predictions)
    num_correct = sum(prediction.correct for prediction in predictions)
    accuracy_success_only = 100.0 * num_correct / num_evaluated if num_evaluated > 0 else 0.0
    confidence_half_width_success = (
        1.96 * math.sqrt(accuracy_success_only * (100.0 - accuracy_success_only) / num_evaluated)
        if num_evaluated > 0
        else 0.0
    )
    calibration = (
        100.0
        * calibration_error(
            [prediction.confidence / 100.0 for prediction in predictions],
            [prediction.correct for prediction in predictions],
            p="2",
            beta=100,
        )
        if num_evaluated > 0
        else 0.0
    )
    accuracy_overall = 100.0 * num_correct / total_questions if total_questions > 0 else 0.0
    confidence_half_width_overall = (
        1.96 * math.sqrt(accuracy_overall * (100.0 - accuracy_overall) / total_questions)
        if total_questions > 0
        else 0.0
    )
    return HLEMetrics(
        accuracy=round(accuracy_overall, 2),
        confidence_interval=round(confidence_half_width_overall, 2),
        accuracy_success_only=round(accuracy_success_only, 2),
        confidence_interval_success_only=round(confidence_half_width_success, 2),
        calibration_error=round(calibration, 2),
        evaluated_questions=num_evaluated,
        failed_questions=num_failed,
        total_questions=total_questions,
    )


def encode_item_payload(record: PreparedHLERecord) -> str:
    """Keep question/image content in worker memory through ``AdapterItem``."""

    return _canonical_json(
        {
            "image": record.image,
            "question": record.question,
        }
    ).decode("utf-8")


def decode_item_payload(payload: str) -> tuple[str, str | None]:
    """Recover a validated question and optional inline image."""

    try:
        value = json.loads(payload, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("HLE item payload is not valid JSON") from exc
    if not isinstance(value, Mapping) or set(value) != {"image", "question"}:
        raise ValueError("HLE item payload has the wrong fields")
    question = value["question"]
    image = value["image"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("HLE item question must be a non-blank string")
    if image is not None:
        if not isinstance(image, str):
            raise ValueError("HLE item image must be a string or null")
        validate_inline_image(image)
    return question, image


def validate_inline_image(value: str) -> tuple[str, int]:
    """Match connector canonical-base64, raster-signature, and size validation."""

    match = _DATA_IMAGE.fullmatch(value)
    if match is None:
        raise ValueError("HLE images must be inline PNG, JPEG, WebP, or GIF data URLs")
    encoded = match.group(2)
    decoded_size = _decoded_base64_size(encoded)
    if decoded_size == 0:
        raise ValueError("HLE image data URL is empty")
    if decoded_size > _MAX_INLINE_IMAGE_BYTES:
        raise ValueError("HLE image exceeds the 20 MiB connector limit")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("HLE image data URL has invalid base64") from exc
    if len(payload) != decoded_size or base64.b64encode(payload).decode("ascii") != encoded:
        raise ValueError("HLE image base64 must use canonical padding")
    _validate_image_signature(match.group(1), payload)
    return match.group(1), len(payload)


def _decoded_base64_size(encoded: str) -> int:
    if len(encoded) % 4:
        raise ValueError("HLE image base64 must use canonical padding")
    padding = len(encoded) - len(encoded.rstrip("="))
    return (len(encoded) // 4) * 3 - padding


def _validate_image_signature(mime_type: str, payload: bytes) -> None:
    valid = {
        "image/png": payload.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": payload.startswith(b"\xff\xd8\xff"),
        "image/gif": payload.startswith((b"GIF87a", b"GIF89a")),
        "image/webp": (
            len(payload) >= 12 and payload.startswith(b"RIFF") and payload[8:12] == b"WEBP"
        ),
    }
    if not valid[mime_type]:
        raise ValueError("HLE image bytes do not match the declared MIME type")


def _read_snapshot_bytes(path: Path) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int) or nofollow == 0:
        raise ValueError("HLE snapshot loading requires O_NOFOLLOW support")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("HLE snapshot must be a regular non-symlink file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("HLE snapshot must be a regular non-symlink file")
        if metadata.st_size > _MAX_SNAPSHOT_BYTES:
            raise ValueError("HLE snapshot exceeds the 512 MiB input limit")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(_MAX_SNAPSHOT_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) > _MAX_SNAPSHOT_BYTES:
        raise ValueError("HLE snapshot exceeds the 512 MiB input limit")
    return payload


def _decode_records(payload: bytes) -> tuple[HLERecord, ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("HLE snapshot must be UTF-8 JSONL") from exc

    records: list[HLERecord] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as exc:
            raise ValueError(f"HLE JSONL line {line_number} is invalid") from exc
        except ValueError as exc:
            raise ValueError(f"HLE JSONL line {line_number}: {exc}") from exc
        if not isinstance(row, Mapping):
            raise ValueError(f"HLE JSONL line {line_number} must be an object")
        required = {"id", "question", "answer"}
        if not required.issubset(row):
            raise ValueError(f"HLE JSONL line {line_number} is missing required fields")
        record_id = row["id"]
        question = row["question"]
        answer = row["answer"]
        image = row.get("image")
        answer_type = row.get("answer_type")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"HLE JSONL line {line_number} has an invalid id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"HLE JSONL line {line_number} has an invalid question")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"HLE JSONL line {line_number} has an invalid answer")
        if image == "":
            image = None
        if image is not None and not isinstance(image, str):
            raise ValueError(f"HLE JSONL line {line_number} has an invalid image")
        if answer_type is not None and not isinstance(answer_type, str):
            raise ValueError(f"HLE JSONL line {line_number} has an invalid answer_type")
        records.append(
            HLERecord(
                record_id=record_id,
                question=question,
                answer=answer,
                image=image,
                answer_type=answer_type,
            )
        )
    if not records:
        raise ValueError("HLE snapshot must contain at least one record")
    return tuple(records)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object field names are not allowed")
        result[key] = value
    return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


__all__ = [
    "HLEMetrics",
    "HLERecord",
    "JUDGE_PROMPT",
    "JUDGE_PROMPT_SHA256",
    "JudgeVerdict",
    "MAX_ATTEMPTS",
    "PROMPT_BUNDLE_SHA256",
    "PreparedHLERecord",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_SHA256",
    "ScoredPrediction",
    "TargetPrompt",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPOSITORY",
    "UPSTREAM_SOURCE_SHA256",
    "build_judge_prompt",
    "calibration_error",
    "compute_metrics",
    "decode_item_payload",
    "encode_item_payload",
    "load_records",
    "load_records_with_sha256",
    "manifest_sha256",
    "parse_judge_response",
    "prepare_records",
    "sha256_file",
    "target_prompt",
    "validate_inline_image",
]
