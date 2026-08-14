"""Formal verification evidence linkage."""

from __future__ import annotations

import json

import pytest

from evidence import canonical_json, sha256_file
from evidence.formal import (
    FormalEvidenceError,
    bind_correctness,
    load_correctness_reference,
    validate_correctness_reference,
)


def _write_correctness(path, *, verdict: str = "pass") -> None:
    path.write_text(
        canonical_json(
            {
                "gate_id": "l1.parity-hf",
                "kind": "correctness",
                "schema_version": 1,
                "verdict": verdict,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_performance_evidence_binds_exact_accepted_correctness_bytes(tmp_path):
    correctness = tmp_path / "correctness.json"
    _write_correctness(correctness)

    reference = load_correctness_reference(correctness)
    payload = bind_correctness(
        {
            "gate_id": "l1.serving-throughput",
            "kind": "performance",
            "schema_version": 1,
        },
        correctness,
    )

    assert reference.sha256 == sha256_file(correctness)
    assert payload["correctness_reference"] == reference.as_mapping()
    assert validate_correctness_reference(
        payload["correctness_reference"]
    ) == reference


@pytest.mark.parametrize("verdict", ["fail", "diagnostic", None])
def test_unaccepted_correctness_cannot_authorize_performance(tmp_path, verdict):
    correctness = tmp_path / "correctness.json"
    _write_correctness(correctness, verdict=verdict)

    with pytest.raises(FormalEvidenceError, match="not accepted"):
        bind_correctness({"kind": "performance"}, correctness)


def test_link_contract_rejects_wrong_kind_duplicate_and_ambiguous_fields(tmp_path):
    correctness = tmp_path / "correctness.json"
    _write_correctness(correctness)

    with pytest.raises(FormalEvidenceError, match="only performance"):
        bind_correctness({"kind": "correctness"}, correctness)
    with pytest.raises(FormalEvidenceError, match="already"):
        bind_correctness(
            {"kind": "performance", "correctness_reference": {}},
            correctness,
        )

    value = json.loads(
        json.dumps(load_correctness_reference(correctness).as_mapping())
    )
    value["extra"] = True
    with pytest.raises(FormalEvidenceError, match="fields"):
        validate_correctness_reference(value)
