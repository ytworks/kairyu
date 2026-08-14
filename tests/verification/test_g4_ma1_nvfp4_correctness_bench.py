from __future__ import annotations

import copy

import pytest

from verification.l1.correctness import g4_ma1_nvfp4_correctness_bench as accuracy


def _report(name: str, memory: int, agreement: float) -> dict[str, object]:
    return {
        "schema_version": accuracy.REPORT_SCHEMA,
        "gate": "G4-M-A1",
        "profile": {"name": name, "sha256": "a" * 64, "config": {}},
        "world_size": 4,
        "model": "model",
        "model_revision": "revision",
        "source_commit": "b" * 40,
        "checkpoint": {"config_sha256": "c" * 64},
        "score": {
            "agreement_rate": agreement,
            "substantive_disagreements": int(agreement < 1.0),
        },
        "passed": agreement == 1.0,
        "memory": {
            "resident_model_bytes_total": memory,
            "resident_model_bytes_max_rank": memory // 4,
        },
    }


def test_curve_orders_profiles_by_measured_resident_memory() -> None:
    curve = accuracy.build_curve(
        [_report("fp8-last", 140, 1.0), _report("baseline", 100, 0.9)]
    )
    assert curve["schema_version"] == accuracy.CURVE_SCHEMA
    assert [point["name"] for point in curve["points"]] == [
        "baseline",
        "fp8-last",
    ]
    assert curve["points"][1]["agreement_rate"] == 1.0


def test_curve_rejects_duplicate_names_and_identity_drift() -> None:
    baseline = _report("baseline", 100, 0.9)
    duplicate = _report("baseline", 120, 1.0)
    with pytest.raises(ValueError, match="unique"):
        accuracy.build_curve([baseline, duplicate])

    drifted = copy.deepcopy(duplicate)
    drifted["profile"]["name"] = "other"
    drifted["source_commit"] = "d" * 40
    with pytest.raises(ValueError, match="run identity"):
        accuracy.build_curve([baseline, drifted])
