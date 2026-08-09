"""Shared fail-closed evidence-artifact mechanics (#382)."""

from __future__ import annotations

import json

import pytest

from kairyu.bench.evidence import (
    EvidenceError,
    artifact_paths,
    canonical_json,
    read_json_object,
    read_jsonl,
    sha256_file,
    strict_json_loads,
    verify_replayed_manifest,
    write_jsonl,
)


def test_jsonl_roundtrip_is_canonical_atomic_and_indexed(tmp_path):
    path = tmp_path / "raw.jsonl"
    write_jsonl(path, [{"z": 1, "row_index": 99}, {"text": "龍"}])

    assert path.read_text(encoding="utf-8") == (
        '{"row_index":0,"z":1}\n'
        '{"row_index":1,"text":"龍"}\n'
    )
    assert read_jsonl(path) == [
        {"row_index": 0, "z": 1},
        {"row_index": 1, "text": "龍"},
    ]


@pytest.mark.parametrize(
    "payload",
    [
        b'{"row_index":0,"x":1,"x":2}\n',
        b'{"row_index":0,"x":NaN}\n',
        b'{"row_index":0}',
        b'[]\n',
        b'{"row_index":1}\n',
        b'\n',
        b'{"row_index":0,"text":"\xff"}\n',
    ],
)
def test_jsonl_rejects_ambiguous_or_malformed_evidence(tmp_path, payload):
    path = tmp_path / "raw.jsonl"
    path.write_bytes(payload)

    with pytest.raises(EvidenceError):
        read_jsonl(path)


def test_strict_object_rejects_duplicate_nonfinite_and_non_object(tmp_path):
    with pytest.raises(EvidenceError, match="duplicate JSON key"):
        strict_json_loads('{"x":1,"x":2}')
    with pytest.raises(EvidenceError, match="non-finite JSON constant"):
        strict_json_loads('{"x":Infinity}')
    path = tmp_path / "manifest.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(EvidenceError, match="not a JSON object"):
        read_json_object(path)


def test_verify_replays_raw_and_rejects_manifest_or_raw_tamper(tmp_path):
    raw_name = "gate-raw.jsonl"
    manifest_name = "gate-manifest.json"
    raw_path = tmp_path / raw_name
    manifest_path = tmp_path / manifest_name
    write_jsonl(raw_path, [{"value": 7}])

    def recompute(rows, raw_sha256):
        return {
            "schema_version": 1,
            "raw": {
                "sha256": raw_sha256,
                "row_count": len(rows),
            },
            "sum": sum(row["value"] for row in rows),
        }

    manifest = recompute(read_jsonl(raw_path), sha256_file(raw_path))
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    assert verify_replayed_manifest(
        tmp_path,
        raw_name=raw_name,
        manifest_name=manifest_name,
        recompute=recompute,
    ) == manifest
    assert artifact_paths(
        raw_path,
        raw_name=raw_name,
        manifest_name=manifest_name,
    ) == (raw_path, manifest_path)

    retained = json.loads(manifest_path.read_text(encoding="utf-8"))
    retained["sum"] = 8
    manifest_path.write_text(json.dumps(retained), encoding="utf-8")
    with pytest.raises(EvidenceError, match="manifest differs"):
        verify_replayed_manifest(
            manifest_path,
            raw_name=raw_name,
            manifest_name=manifest_name,
            recompute=recompute,
        )

    write_jsonl(raw_path, [{"value": 8}])
    with pytest.raises(EvidenceError, match="manifest differs"):
        verify_replayed_manifest(
            tmp_path,
            raw_name=raw_name,
            manifest_name=manifest_name,
            recompute=recompute,
        )


def test_custom_gate_error_type_is_preserved(tmp_path):
    class GateError(ValueError):
        pass

    path = tmp_path / "raw.jsonl"
    path.write_text('{"row_index":0,"x":NaN}\n', encoding="utf-8")

    with pytest.raises(GateError, match="non-finite"):
        read_jsonl(path, error_type=GateError)
