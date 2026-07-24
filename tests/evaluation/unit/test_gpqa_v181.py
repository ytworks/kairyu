import hashlib
import json
from pathlib import Path

import pytest

import kairyu.evaluation.adapters.gpqa_v181 as gpqa_v181
from kairyu.evaluation.adapters.gpqa_v181 import (
    PROMPT_SHA256,
    PROMPT_TEMPLATE,
    extract_answer,
    load_records,
    load_records_with_sha256,
    manifest_sha256,
    prepare_records,
    preprocess_choice,
    sha256_file,
)

ROOT = Path(__file__).parents[3]
SMOKE_FIXTURE = (
    ROOT / "kairyu" / "evaluation" / "resources" / "fixtures" / "gpqa-diamond-smoke.jsonl"
)


def test_synthetic_fixture_and_prompt_template_have_pinned_hashes():
    assert sha256_file(SMOKE_FIXTURE) == (
        "d00d5ff92cf97f99b66af968abb7b247494d2f8f79a434a91e1ce45172683eed"
    )
    assert hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest() == PROMPT_SHA256
    assert PROMPT_SHA256 == ("f759b2cf67b53500e906bc321c88285351eadd9918a8479505626828f04e6aba")


def test_seed_42_sequential_shuffle_prompt_and_input_hash_match_golden():
    prepared = prepare_records(load_records(SMOKE_FIXTURE), seed=42)

    assert [
        (
            item.record_id,
            item.choice_permutation,
            item.target,
            item.input_sha256,
        )
        for item in prepared
    ] == [
        (
            "synthetic-orbit-001",
            (2, 1, 3, 0),
            "C",
            "fdf314bdf3e9bb304402c8a71bcaef3ba8c6613b32a99fff685956fb16da1a0e",
        ),
        (
            "synthetic-lab-002",
            (3, 2, 0, 1),
            "A",
            "9458782398a62aaa34208347311efb0577d93f31949997b4ccc1f423f81e9100",
        ),
    ]
    assert prepared[0].prompt == (
        "Answer the following multiple choice question. The last line of your "
        "response should be of the following format: 'ANSWER: [LETTER]' (without "
        "quotes) where [LETTER] is one of A,B,C,D. Think step by step before "
        "answering.\n\n"
        "A fictional moon completes one orbit every 12 local days. How many orbits "
        "does it complete in 36 local days?\n\n"
        "A) Four orbits\n"
        "B) Two orbits\n"
        "C) Three orbits\n"
        "D) One orbit"
    )
    assert prepared[1].prompt == (
        "Answer the following multiple choice question. The last line of your "
        "response should be of the following format: 'ANSWER: [LETTER]' (without "
        "quotes) where [LETTER] is one of A,B,C,D. Think step by step before "
        "answering.\n\n"
        "In a made-up laboratory notation, ZIM means add two. Starting from five, "
        "what does one ZIM operation produce?\n\n"
        "A) Seven\n"
        "B) Eight\n"
        "C) Five\n"
        "D) Six"
    )
    assert manifest_sha256(prepared) == (
        "396b978e4af4fb42eed17d5e5213f2ba00b768edfdc8c659a5cb7bc480a2ef9a"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, " "),
        ("  A [title] B  ", "A. B"),
        ("A [synthetic citation] B", "A B"),
        (" already clean ", "already clean"),
    ],
)
def test_choice_preprocessing_matches_evalscope_181_order(raw, expected):
    assert preprocess_choice(raw) == expected


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("worked carefully\nANSWER: C", "C"),
        ("ANSWER: B.", "B"),
        ("ANSWER: c", None),
        ("ANSWER: E", None),
        ("ANSWER: A, B", None),
        ("fallback has no tag and ends with D", "D"),
        ("fallback can return out-of-range Z", "Z"),
        ("all lowercase", None),
    ],
)
def test_answer_parser_matches_evalscope_181_permissive_semantics(completion, expected):
    assert extract_answer(completion) == expected


def test_local_snapshot_loader_rejects_symlinks(tmp_path):
    link = tmp_path / "gpqa.jsonl"
    link.symlink_to(SMOKE_FIXTURE)

    with pytest.raises(ValueError, match="regular non-symlink"):
        load_records(link)
    with pytest.raises(ValueError, match="regular non-symlink"):
        sha256_file(link)


def test_combined_loader_reads_once_and_hashes_the_same_snapshot_bytes(
    tmp_path,
    monkeypatch,
):
    snapshot = tmp_path / "gpqa.jsonl"
    original = SMOKE_FIXTURE.read_bytes()
    replacement = original.replace(b"synthetic-orbit-001", b"synthetic-orbit-999")
    snapshot.write_bytes(original)
    real_read = gpqa_v181._read_snapshot_bytes
    reads = 0

    def read_then_replace(path):
        nonlocal reads
        reads += 1
        content = real_read(path)
        path.write_bytes(replacement)
        return content

    monkeypatch.setattr(gpqa_v181, "_read_snapshot_bytes", read_then_replace)

    records, digest = load_records_with_sha256(snapshot)

    assert reads == 1
    assert records[0].record_id == "synthetic-orbit-001"
    assert digest == hashlib.sha256(original).hexdigest()
    assert digest != hashlib.sha256(snapshot.read_bytes()).hexdigest()


def test_jsonl_loader_rejects_duplicate_object_fields(tmp_path):
    snapshot = tmp_path / "duplicate.jsonl"
    row = {
        "Record ID": "duplicate-json",
        "Question": "first",
        "Incorrect Answer 1": "a",
        "Incorrect Answer 2": "b",
        "Incorrect Answer 3": "c",
        "Correct Answer": "d",
    }
    payload = json.dumps(row)
    needle = json.dumps("Question") + ": " + json.dumps("first")
    duplicate = needle + ", " + json.dumps("Question") + ": " + json.dumps("second")
    snapshot.write_text(payload.replace(needle, duplicate) + chr(10), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate object field names"):
        load_records(snapshot)


def test_csv_loader_rejects_duplicate_header_fields(tmp_path):
    snapshot = tmp_path / "duplicate.csv"
    headers = (
        "Record ID",
        "Question",
        "Question",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
        "Correct Answer",
    )
    values = ("duplicate-csv", "first", "second", "a", "b", "c", "d")
    snapshot.write_text(
        ",".join(headers) + chr(10) + ",".join(values) + chr(10),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate header field names"):
        load_records(snapshot)
