"""Fail-closed paired-gate contracts for opt-in FP32 logits (#364)."""

from __future__ import annotations

import copy

import pytest

from bench import gate_logits_dtype

_CODE = {"commit": "a" * 40, "dirty": False}


def _raw_positions() -> list[dict[str, object]]:
    return [
        {
            "prompt": f"p{prompt}",
            "position": position,
            "reference_token": prompt * 16 + position,
            "engine_token": prompt * 16 + position,
            "reference_token_logprob": -0.1,
            "engine_token_logprob": -0.1,
            "reference_top_logprobs": {
                str(prompt * 16 + position + offset): -0.1 - offset
                for offset in range(20)
            },
            "engine_top_logprobs": {
                str(prompt * 16 + position + offset): -0.1 - offset
                for offset in range(20)
            },
            "absolute_logprob_delta": 0.0,
        }
        for prompt in range(64)
        for position in range(16)
    ]


def _measurement_config(arm: str, tp: int | None = None) -> dict[str, object]:
    config: dict[str, object] = {
        "logits_dtype_requested": arm,
        "logits_dtype_resolved": (
            "float32" if arm == "float32" else "bfloat16"
        ),
        "code": copy.deepcopy(_CODE),
        "hardware": {
            "arch": "cuda",
            "device_name": "NVIDIA RTX PRO 6000",
            "device_count": tp or 2,
            "runtime": {
                "torch": "2.test",
                "torch_cuda": "13.0",
                "nccl": "2.test",
                "nvidia_smi": ["same physical GPU inventory"],
                "nvidia_smi_topology": ["same physical topology"],
            },
        },
        "dtype": "bfloat16",
        "checkpoint_revision": "b" * 40,
        "num_prompts": 64,
        "positions": 16,
        "num_pages": 2048,
        "page_size": 16,
    }
    if tp is not None:
        config["tensor_parallel_size"] = tp
    return config


def _artifact(gate: str, arm: str) -> dict[str, object]:
    degrees = (1, 2) if gate == "A1" else (2, 4, 8)
    evidence: dict[str, object] = {
        "reference": {
            "reference_runtime": {"code": copy.deepcopy(_CODE)},
            "reference": {},
        },
        **{
            f"teacher_tp{tp}": {
                "config": _measurement_config(arm, tp),
                "reference_noise_floor": {"raw_self_agreement_rate": 1.0},
                "logprob_tolerance": {"tie_gap_nats": 0.125, "top_k": 20},
                "raw_positions": _raw_positions(),
            }
            for tp in degrees
        },
    }
    if gate == "A1":
        continuation_config = _measurement_config(arm)
        continuation_config["checkpoint"] = {"weights_sha256": "c" * 64}
        evidence["continuations"] = {
            "config": continuation_config,
            "prompts": [{"request_id": f"p{prompt}"} for prompt in range(64)],
        }
    return {
        "gate": f"G2 {gate}",
        "verdict": "PASS",
        "code": copy.deepcopy(_CODE),
        "evidence": evidence,
    }


def _artifacts() -> dict[str, dict[str, object]]:
    return {
        "a1_model": _artifact("A1", "model"),
        "a1_float32": _artifact("A1", "float32"),
        "a2_model": _artifact("A2", "model"),
        "a2_float32": _artifact("A2", "float32"),
    }


def _paired_checks(
    monkeypatch: pytest.MonkeyPatch,
    artifacts: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    # The focused tests exercise the paired operator. Formal A1/A2 recomputation
    # has its own unit suites and requires full model-specific fixture envelopes.
    monkeypatch.setattr(gate_logits_dtype, "_validate_arm", lambda *_args: [])
    checks, _summaries, _quality = gate_logits_dtype.evaluate(**artifacts)
    return {str(check["name"]): check for check in checks}


def test_paired_gate_rejects_cross_arm_hardware_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _artifacts()
    baseline = _paired_checks(monkeypatch, artifacts)
    assert all(check["passed"] for check in baseline.values())

    config = artifacts["a2_float32"]["evidence"]["teacher_tp4"]["config"]
    config["hardware"]["device_name"] = "DIFFERENT-GPU"

    checks = _paired_checks(monkeypatch, artifacts)
    assert checks[
        "A2 TP4 arms use identical non-logits measurement config"
    ]["passed"] is False


@pytest.mark.parametrize(
    ("artifact_name", "teacher_name", "row"),
    (
        ("a1_float32", "teacher_tp1", None),
        (
            "a2_model",
            "teacher_tp8",
            {
                "prompt": "p64",
                "position": 0,
                "reference_token": 0,
                "engine_token": 0,
                "absolute_logprob_delta": 0.0,
            },
        ),
    ),
    ids=("duplicate", "unexpected-extra-key"),
)
def test_paired_gate_rejects_duplicate_or_extra_raw_position(
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
    teacher_name: str,
    row: dict[str, object] | None,
) -> None:
    artifacts = _artifacts()
    teacher = artifacts[artifact_name]["evidence"][teacher_name]
    rows = teacher["raw_positions"]
    rows.append(copy.deepcopy(rows[0]) if row is None else row)

    checks = _paired_checks(monkeypatch, artifacts)
    gate = "A1" if artifact_name.startswith("a1") else "A2"
    tp = teacher_name.removeprefix("teacher_tp")
    assert checks[f"{gate} TP{tp} has complete paired raw positions"][
        "passed"
    ] is False


def test_paired_gate_rejects_missing_engine_top_logprob_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _artifacts()
    rows = artifacts["a2_float32"]["evidence"]["teacher_tp4"][
        "raw_positions"
    ]
    for row in rows:
        del row["engine_top_logprobs"]

    checks = _paired_checks(monkeypatch, artifacts)
    assert checks[
        "A2 TP4 retains complete paired raw logprob observations"
    ]["passed"] is False


def test_paired_gate_recomputes_agreeing_logprob_deltas_from_raw_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _artifacts()
    for artifact in artifacts.values():
        for name, result in artifact["evidence"].items():
            if not name.startswith("teacher_tp"):
                continue
            for row in result["raw_positions"]:
                row.pop("absolute_logprob_delta", None)

    checks = _paired_checks(monkeypatch, artifacts)
    assert all(check["passed"] for check in checks.values())
    _checks, summaries, _quality = gate_logits_dtype.evaluate(**artifacts)
    assert summaries["a2_tp4"]["agreeing_position_logprob_delta"]["model"] == {
        "count": 1024,
        "mean": 0.0,
        "p50": 0.0,
        "p95": 0.0,
        "max": 0.0,
    }


def test_paired_gate_rejects_selected_logprob_map_inconsistency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _artifacts()
    row = artifacts["a1_model"]["evidence"]["teacher_tp1"]["raw_positions"][0]
    row["engine_token_logprob"] = -9.0

    checks = _paired_checks(monkeypatch, artifacts)
    assert checks[
        "A1 TP1 retains complete paired raw logprob observations"
    ]["passed"] is False


def test_paired_gate_rejects_assembler_from_a_different_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _artifacts()
    monkeypatch.setattr(gate_logits_dtype, "_validate_arm", lambda *_args: [])
    checks, _summaries, _quality = gate_logits_dtype.evaluate(
        **artifacts,
        assembly_code={"commit": "b" * 40, "dirty": False},
    )
    by_name = {check["name"]: check for check in checks}
    assert by_name[
        "paired assembler itself uses the same clean measurement commit"
    ]["passed"] is False
