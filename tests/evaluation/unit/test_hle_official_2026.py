import base64
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import kairyu.evaluation.adapters.hle_official_2026 as hle
from kairyu.evaluation.adapters.hle_official_2026 import (
    JUDGE_PROMPT_SHA256,
    MAX_ATTEMPTS,
    PROMPT_BUNDLE_SHA256,
    SYSTEM_PROMPT_SHA256,
    HLEMetrics,
    HLERecord,
    ScoredPrediction,
    build_judge_prompt,
    calibration_error,
    compute_metrics,
    decode_item_payload,
    encode_item_payload,
    load_records,
    load_records_with_sha256,
    manifest_sha256,
    parse_judge_response,
    prepare_records,
    target_prompt,
    validate_inline_image,
)

ROOT = Path(__file__).parents[3]
SMOKE_FIXTURE = (
    ROOT / "kairyu" / "evaluation" / "resources" / "fixtures" / "humanitys-last-exam-smoke.jsonl"
)
FIXTURE_SHA256 = "0f85f6fffc09191181b42ac327a66ed97746f661696d1d24798a8f1b396fb194"


def test_official_source_prompt_and_synthetic_fixture_pins():
    assert hle.UPSTREAM_REPOSITORY == "https://github.com/centerforaisafety/simple-evals"
    assert hle.UPSTREAM_COMMIT == "8e53435ff2985b0f32ea7ceb7e92c3a175f2c0f3"
    assert MAX_ATTEMPTS == 5
    assert hle.UPSTREAM_SOURCE_SHA256 == (
        "d276f725ecc5ea2c08f73e161f97760881332c3d33d20b197c2ffbac5f55edfe"
    )
    assert SYSTEM_PROMPT_SHA256 == (
        "40f399fe8c277e2cb98ad8ca663461141d1abbf01a18da3c4129eccc706b4326"
    )
    assert JUDGE_PROMPT_SHA256 == (
        "bf081baac98cc6e45b7b2e18330920a64c0747b1915b346043694305cd396172"
    )
    assert PROMPT_BUNDLE_SHA256 == (
        "94fd163680229c67352a8c4f38bff04dae4183cea13e6afc7674904e21ae6502"
    )
    assert hashlib.sha256(SMOKE_FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


def test_prepare_fixture_covers_text_and_inline_image_with_golden_identity():
    records, source_sha256 = load_records_with_sha256(SMOKE_FIXTURE)
    prepared = prepare_records(records)

    assert source_sha256 == FIXTURE_SHA256
    assert [(record.record_id, record.image is not None) for record in records] == [
        ("synthetic-hle-text-001", False),
        ("synthetic-hle-image-002", True),
    ]
    assert [(record.record_id, record.input_sha256) for record in prepared] == [
        (
            "synthetic-hle-text-001",
            "2f11751493f78194fa4764b6b79689dc05a38bf9f01a94bae5273546a1c196a7",
        ),
        (
            "synthetic-hle-image-002",
            "215ed8474828baaf6e4318607a31385e59703695af9c28bfbc256cf8e538c3c6",
        ),
    ]
    assert manifest_sha256(prepared) == (
        "dea3843aded911fc16c141ea57da35183ed64ff3cbcfa30a6b20048d6ad36064"
    )

    text_prompt = target_prompt(prepared[0])
    image_prompt = target_prompt(prepared[1])
    assert text_prompt.system_prompt == hle.SYSTEM_PROMPT
    assert text_prompt.image_url is None
    assert image_prompt.image_url is not None
    assert validate_inline_image(image_prompt.image_url) == ("image/png", 79)
    assert decode_item_payload(encode_item_payload(prepared[1])) == (
        prepared[1].question,
        prepared[1].image,
    )


@pytest.mark.parametrize(
    ("mime_type", "image_bytes"),
    (
        ("image/png", b"\x89PNG\r\n\x1a\n"),
        ("image/jpeg", b"\xff\xd8\xff\xe0"),
        ("image/gif", b"GIF89a"),
        ("image/webp", b"RIFF\x00\x00\x00\x00WEBP"),
    ),
)
def test_inline_image_validation_matches_connector_raster_signatures(
    mime_type,
    image_bytes,
):
    encoded = base64.b64encode(image_bytes).decode("ascii")

    assert validate_inline_image(f"data:{mime_type};base64,{encoded}") == (
        mime_type,
        len(image_bytes),
    )


def test_prepare_rejects_noncanonical_mismatched_and_oversized_images(monkeypatch):
    png = b"\x89PNG\r\n\x1a\n"
    canonical = base64.b64encode(png).decode("ascii")
    invalid_images = (
        f"data:image/png;base64,{canonical.rstrip(chr(61))}",
        "data:image/png;base64,aGVsbG8=",
        f"data:image/jpeg;base64,{canonical}",
    )

    for ordinal, image in enumerate(invalid_images):
        with pytest.raises(ValueError):
            prepare_records(
                (
                    HLERecord(
                        record_id=f"invalid-image-{ordinal}",
                        question="q",
                        answer="a",
                        image=image,
                    ),
                )
            )

    assert hle._MAX_INLINE_IMAGE_BYTES == 20 * 1024 * 1024
    monkeypatch.setattr(hle, "_MAX_INLINE_IMAGE_BYTES", len(png))
    oversized = base64.b64encode(png + b"x").decode("ascii")
    with pytest.raises(ValueError, match="20 MiB"):
        validate_inline_image(f"data:image/png;base64,{oversized}")


def test_judge_prompt_substitution_preserves_all_text():
    prompt = build_judge_prompt(
        question="synthetic {question}",
        response="Explanation: x\nAnswer: 7\nConfidence: 80%",
        correct_answer="7",
    )

    assert "[question]: synthetic {question}" in prompt
    assert "[response]: Explanation: x\nAnswer: 7\nConfidence: 80%" in prompt
    assert "[correct_answer]: 7" in prompt
    assert prompt.endswith(
        "<confidence>The extracted confidence score between 0 and 100 from [response]. "
        "Put 100 if there is no confidence score available.</confidence>"
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            (
                "<extracted_final_answer> 7 </extracted_final_answer>"
                "<reasoning> equivalent </reasoning>"
                "<correct>YES</correct><confidence>080</confidence>"
            ),
            ("7", "equivalent", "yes", 80),
        ),
        (
            (
                "<confidence>101</confidence><correct>no</correct>"
                "<reasoning>different</reasoning>"
                "<extracted_final_answer>8</extracted_final_answer>"
            ),
            ("8", "different", "no", 101),
        ),
        ("<correct>yes</correct><confidence>50</confidence>", None),
        (
            (
                "<extracted_final_answer>7</extracted_final_answer>"
                "<reasoning>same</reasoning><correct>maybe</correct>"
                "<confidence>50</confidence>"
            ),
            None,
        ),
    ],
)
def test_judge_parser_mirrors_independent_upstream_xml_regexes(response, expected):
    parsed = parse_judge_response(response)
    if expected is None:
        assert parsed is None
    else:
        assert parsed is not None
        assert (
            parsed.extracted_final_answer,
            parsed.reasoning,
            parsed.correct,
            parsed.confidence,
        ) == expected


def test_calibration_error_preserves_upstream_final_bin_behavior():
    assert calibration_error([0.9, 0.8], [True, False]) == pytest.approx(0.35)
    # The reviewed upstream code intentionally/accidentally excludes its final
    # bin for n >= beta. Pin that behavior instead of "fixing" the score.
    assert calibration_error([1.0] * 100, [False] * 100) == 0.0
    assert calibration_error([1.0] * 200, [False] * 200) == pytest.approx(2**-0.5)
    assert calibration_error([1.0] * 200, [False] * 200, p="1") == 0.5
    assert calibration_error([1.0] * 200, [False] * 200, p="infty") == 1.0


def test_compute_metrics_matches_upstream_percent_scales_and_failed_denominator():
    metrics = compute_metrics(
        (ScoredPrediction(correct=True, confidence=80), ScoredPrediction(False, 40)),
        total_questions=3,
        num_failed=1,
    )

    assert metrics == HLEMetrics(
        accuracy=33.33,
        confidence_interval=53.34,
        accuracy_success_only=50.0,
        confidence_interval_success_only=69.3,
        calibration_error=10.0,
        evaluated_questions=2,
        failed_questions=1,
        total_questions=3,
    )
    assert compute_metrics((), total_questions=2, num_failed=2) == HLEMetrics(
        accuracy=0.0,
        confidence_interval=0.0,
        accuracy_success_only=0.0,
        confidence_interval_success_only=0.0,
        calibration_error=0.0,
        evaluated_questions=0,
        failed_questions=2,
        total_questions=2,
    )


def test_snapshot_reader_uses_nofollow_cloexec_and_same_open_descriptor(
    tmp_path,
    monkeypatch,
):
    original = SMOKE_FIXTURE.read_bytes()
    replacement = original.replace(
        b"synthetic-hle-text-001",
        b"synthetic-hle-text-999",
    )
    snapshot = tmp_path / "snapshot.jsonl"
    replacement_path = tmp_path / "replacement.jsonl"
    snapshot.write_bytes(original)
    replacement_path.write_bytes(replacement)
    real_open = hle.os.open
    observed_flags = []

    def open_then_replace(path, flags):
        descriptor = real_open(path, flags)
        observed_flags.append(flags)
        replacement_path.replace(snapshot)
        return descriptor

    monkeypatch.setattr(hle.os, "open", open_then_replace)
    records, digest = load_records_with_sha256(snapshot)

    assert records[0].record_id == "synthetic-hle-text-001"
    assert digest == hashlib.sha256(original).hexdigest()
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == (
        hashlib.sha256(replacement).hexdigest()
    )
    if getattr(hle.os, "O_NOFOLLOW", 0):
        assert observed_flags[0] & hle.os.O_NOFOLLOW
    if getattr(hle.os, "O_CLOEXEC", 0):
        assert observed_flags[0] & hle.os.O_CLOEXEC
    if getattr(hle.os, "O_NONBLOCK", 0):
        assert observed_flags[0] & hle.os.O_NONBLOCK


def test_snapshot_reader_fails_closed_without_nofollow(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot.jsonl"
    snapshot.write_bytes(SMOKE_FIXTURE.read_bytes())
    monkeypatch.delattr(hle.os, "O_NOFOLLOW", raising=False)

    with pytest.raises(ValueError, match="requires O_NOFOLLOW support"):
        load_records(snapshot)


@pytest.mark.skipif(
    not hasattr(hle.os, "mkfifo")
    or not getattr(hle.os, "O_NONBLOCK", 0)
    or not Path("/tmp").is_dir(),
    reason="FIFO and O_NONBLOCK support are required",
)
def test_snapshot_reader_rejects_fifo_without_waiting_for_writer():
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        fifo = Path(directory) / "snapshot.fifo"
        hle.os.mkfifo(fifo)

        with pytest.raises(ValueError, match="regular non-symlink"):
            load_records(fifo)


def test_snapshot_reader_rejects_growth_beyond_bounded_read(tmp_path, monkeypatch):
    limit = 16
    snapshot = tmp_path / "grown.jsonl"
    snapshot.write_bytes(b"x" * (limit + 1))
    real_fstat = hle.os.fstat

    def understated_size(descriptor):
        metadata = real_fstat(descriptor)
        return SimpleNamespace(st_mode=metadata.st_mode, st_size=limit)

    monkeypatch.setattr(hle, "_MAX_SNAPSHOT_BYTES", limit)
    monkeypatch.setattr(hle.os, "fstat", understated_size)

    with pytest.raises(ValueError, match="512 MiB input limit"):
        load_records(snapshot)


def test_loader_rejects_symlinks_duplicate_fields_ids_and_remote_images(tmp_path):
    link = tmp_path / "linked.jsonl"
    link.symlink_to(SMOKE_FIXTURE)
    with pytest.raises(ValueError, match="regular non-symlink"):
        load_records(link)

    duplicate_field = tmp_path / "duplicate-field.jsonl"
    duplicate_field.write_text(
        '{"id":"one","id":"two","question":"q","answer":"a","image":null}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate object field"):
        load_records(duplicate_field)

    duplicate_id = tmp_path / "duplicate-id.jsonl"
    row = {"id": "same", "question": "q", "answer": "a", "image": None}
    duplicate_id.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="IDs must be unique"):
        prepare_records(load_records(duplicate_id))

    remote_image = tmp_path / "remote-image.jsonl"
    remote_image.write_text(
        json.dumps(
            {
                "id": "remote",
                "question": "q",
                "answer": "a",
                "image": "https://example.invalid/private.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inline"):
        prepare_records(load_records(remote_image))


def test_metric_helpers_reject_ambiguous_shapes_and_nonfinite_values():
    with pytest.raises(ValueError, match="lengths differ"):
        calibration_error([0.5], [])
    with pytest.raises(ValueError, match="finite"):
        calibration_error([float("nan")], [True])
    with pytest.raises(ValueError, match="exceed"):
        compute_metrics((ScoredPrediction(True, 50),), total_questions=1, num_failed=1)
