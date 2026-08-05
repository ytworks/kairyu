"""Assemble the fail-closed paired A1/A2 logits-dtype experiment (#364).

Inputs are four self-contained formal gate artifacts: the ``model`` and
``float32`` arms of G2 A1 and G2 A2.  A non-positive quality result is valid,
but incomplete, mixed-provenance, mislabelled, or independently failing arms
make this assembler fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from bench import gate_a1, gate_a2
from bench.parity_tp import _code_provenance

_ARMS = ("model", "float32")
_RESOLVED = {"model": "bfloat16", "float32": "float32"}
_A1_TP = (1, 2)
_A2_TP = (2, 4, 8)
_PROMPTS = 64
_POSITIONS = 16
_EXPECTED_ROWS = _PROMPTS * _POSITIONS
_EXPECTED_POSITION_KEYS = {
    (f"p{prompt}", position)
    for prompt in range(_PROMPTS)
    for position in range(_POSITIONS)
}
_ARM_CONFIG_FIELDS = frozenset(
    {"logits_dtype_requested", "logits_dtype_resolved"}
)


def _record(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(config: dict[str, Any]) -> tuple[object, object]:
    return (
        config.get("logits_dtype_requested"),
        config.get("logits_dtype_resolved"),
    )


def _non_arm_config(config: object) -> dict[str, Any]:
    """Return the measurement identity with only the treatment removed."""

    if not isinstance(config, dict):
        return {}
    return {
        key: value
        for key, value in config.items()
        if key not in _ARM_CONFIG_FIELDS
    }


def _same_non_arm_config(model: object, float32: object) -> bool:
    return (
        isinstance(model, dict)
        and bool(model)
        and isinstance(float32, dict)
        and bool(float32)
        and _non_arm_config(model) == _non_arm_config(float32)
    )


def _position_map(result: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    rows = result.get("raw_positions") or []
    return {
        (str(row.get("prompt")), int(row.get("position"))): row
        for row in rows
        if isinstance(row, dict)
        and row.get("prompt") is not None
        and isinstance(row.get("position"), int)
    }


def _raw_position_counts(result: dict[str, Any]) -> tuple[int, int, bool]:
    rows = result.get("raw_positions")
    raw_count = len(rows) if isinstance(rows, list) else 0
    position_map = _position_map(result)
    unique_count = len(position_map)
    exact = (
        isinstance(rows, list)
        and raw_count == _EXPECTED_ROWS
        and unique_count == _EXPECTED_ROWS
        and set(position_map) == _EXPECTED_POSITION_KEYS
    )
    return raw_count, unique_count, exact


def _is_finite_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _raw_logprob_counts(
    result: dict[str, Any],
    expected_top_k: int | None,
) -> tuple[int, int, bool]:
    """Count complete selected/top-logprob rows retained by one teacher run."""

    rows = result.get("raw_positions")
    if (
        not isinstance(rows, list)
        or type(expected_top_k) is not int
        or expected_top_k <= 0
    ):
        return len(rows) if isinstance(rows, list) else 0, 0, False
    complete = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        engine_token = row.get("engine_token")
        reference_token = row.get("reference_token")
        engine_top = row.get("engine_top_logprobs")
        reference_top = row.get("reference_top_logprobs")
        if (
            type(engine_token) is not int
            or type(reference_token) is not int
            or not _is_finite_number(row.get("engine_token_logprob"))
            or not _is_finite_number(row.get("reference_token_logprob"))
            or not isinstance(engine_top, dict)
            or not isinstance(reference_top, dict)
            or len(engine_top) != expected_top_k
            or len(reference_top) != expected_top_k
            or str(engine_token) not in engine_top
            or str(reference_token) not in reference_top
            or engine_top[str(engine_token)] != row.get("engine_token_logprob")
            or reference_top[str(reference_token)]
            != row.get("reference_token_logprob")
            or not all(
                isinstance(token_id, str) and _is_finite_number(value)
                for token_id, value in engine_top.items()
            )
            or not all(
                isinstance(token_id, str) and _is_finite_number(value)
                for token_id, value in reference_top.items()
            )
        ):
            continue
        complete += 1
    exact = len(rows) == _EXPECTED_ROWS and complete == _EXPECTED_ROWS
    return len(rows), complete, exact


def _arm_configs(
    artifact: dict[str, Any], gate: str
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    evidence = artifact.get("evidence") or {}
    if gate == "A1":
        continuation = (evidence.get("continuations") or {}).get("config") or {}
        degrees = _A1_TP
    else:
        continuation = {}
        degrees = _A2_TP
    teachers = {
        tp: ((evidence.get(f"teacher_tp{tp}") or {}).get("config") or {})
        for tp in degrees
    }
    return continuation, teachers


def _validate_arm(
    artifact: dict[str, Any], gate: str, arm: str
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    expected_gate = f"G2 {gate}"
    _record(
        checks,
        f"{gate} {arm} carries a passing formal gate artifact",
        artifact.get("gate") == expected_gate and artifact.get("verdict") == "PASS",
        f"gate={artifact.get('gate')!r}, verdict={artifact.get('verdict')!r}",
    )
    evidence = artifact.get("evidence") or {}
    if gate == "A1":
        recomputed = gate_a1.evaluate(
            evidence.get("continuations") or {},
            evidence.get("reference") or {},
            [evidence.get("teacher_tp1") or {}, evidence.get("teacher_tp2") or {}],
        )
        comparisons = None
    else:
        recomputed, comparisons = gate_a2.evaluate(
            evidence.get("reference") or {},
            [
                evidence.get("teacher_tp2") or {},
                evidence.get("teacher_tp4") or {},
                evidence.get("teacher_tp8") or {},
            ],
        )
    failed = [row["name"] for row in recomputed if not row["passed"]]
    _record(
        checks,
        f"{gate} {arm} independently recomputes its formal safety gate",
        not failed,
        "PASS" if not failed else f"failed={failed}",
    )
    if gate == "A1":
        reference = (evidence.get("reference") or {}).get("reference") or {}
        noise_floor = gate_a2._reference_noise_floor(reference)
        tie_gap = max(
            gate_a2._MIN_TIE_GAP,
            noise_floor["max_gap_at_self_disagreement"],
        )
        raw_scores = {
            tp: gate_a2._recompute_hf_score(
                evidence.get(f"teacher_tp{tp}") or {},
                reference,
                noise_floor["raw_self_agreement_rate"],
                tie_gap,
            )
            for tp in _A1_TP
        }
        raw_pass = all(
            score["verdict"] == "PASS"
            and score["max_abs_logprob_delta"] <= gate_a2._MAX_LOGPROB_DELTA
            for score in raw_scores.values()
        )
        _record(
            checks,
            f"{gate} {arm} raw rows recompute agreement and binding logprob tolerance",
            raw_pass,
            ", ".join(
                f"TP{tp}={score['verdict']}, max_delta="
                f"{score['max_abs_logprob_delta']:.6f}"
                for tp, score in raw_scores.items()
            ),
        )
    if gate == "A2":
        expected_comparisons = {
            "tp2_vs_hf",
            "tp4_vs_hf",
            "tp8_vs_hf",
            "tp4_vs_tp2",
            "tp8_vs_tp2",
        }
        comparison_pass = set(comparisons or {}) == expected_comparisons and all(
            row.get("verdict") == "PASS" for row in (comparisons or {}).values()
        )
        _record(
            checks,
            f"{gate} {arm} recomputes every HF and direct-TP comparison",
            comparison_pass,
            f"comparisons={sorted((comparisons or {}).keys())}",
        )

    continuation, teachers = _arm_configs(artifact, gate)
    identities = [_identity(config) for config in teachers.values()]
    if gate == "A1":
        identities.append(_identity(continuation))
    expected = (arm, _RESOLVED[arm])
    _record(
        checks,
        f"{gate} {arm} uses one attested logits mode at every observation",
        identities and all(identity == expected for identity in identities),
        f"expected={expected}, found={identities}",
    )
    return checks


def _code_rows(artifact: dict[str, Any], gate: str) -> list[dict[str, Any]]:
    evidence = artifact.get("evidence") or {}
    reference_code = (
        ((evidence.get("reference") or {}).get("reference_runtime") or {}).get("code")
        or {}
    )
    continuation, teachers = _arm_configs(artifact, gate)
    rows = [
        artifact.get("code") or {},
        reference_code,
        *(config.get("code") or {} for config in teachers.values()),
    ]
    if gate == "A1":
        rows.append(continuation.get("code") or {})
    return rows


def _delta_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def paired_position_summary(
    model_result: dict[str, Any], float32_result: dict[str, Any]
) -> dict[str, Any]:
    """Compare one TP cell position-by-position against its shared reference."""

    model_rows = _position_map(model_result)
    float_rows = _position_map(float32_result)
    complete = bool(model_rows) and set(model_rows) == set(float_rows)
    toward = away = changed_other = 0
    changed: list[dict[str, Any]] = []
    model_agreed = float_agreed = 0
    model_deltas: list[float] = []
    float_deltas: list[float] = []
    selected_logprob_deltas: list[float] = []
    top_logprob_deltas: list[float] = []
    model_missing = float_missing = 0
    model_substantive = float_substantive = 0
    reference_mismatches: list[dict[str, Any]] = []
    for key in sorted(set(model_rows) | set(float_rows)):
        model_row = model_rows.get(key) or {}
        float_row = float_rows.get(key) or {}
        reference = model_row.get("reference_token")
        if reference != float_row.get("reference_token"):
            reference_mismatches.append({"prompt": key[0], "position": key[1]})
            continue
        model_token = model_row.get("engine_token")
        float_token = float_row.get("engine_token")
        model_missing += int(model_token is None)
        float_missing += int(float_token is None)
        model_substantive += int(
            model_token is not None
            and model_token != reference
            and not model_row.get("tie_break", False)
        )
        float_substantive += int(
            float_token is not None
            and float_token != reference
            and not float_row.get("tie_break", False)
        )
        model_hit = model_token == reference and model_token is not None
        float_hit = float_token == reference and float_token is not None
        model_agreed += int(model_hit)
        float_agreed += int(float_hit)
        model_selected_logprob = model_row.get("engine_token_logprob")
        float_selected_logprob = float_row.get("engine_token_logprob")
        model_reference_logprob = model_row.get("reference_token_logprob")
        float_reference_logprob = float_row.get("reference_token_logprob")
        if (
            model_hit
            and _is_finite_number(model_selected_logprob)
            and _is_finite_number(model_reference_logprob)
        ):
            model_deltas.append(
                abs(float(model_selected_logprob) - float(model_reference_logprob))
            )
        if (
            float_hit
            and _is_finite_number(float_selected_logprob)
            and _is_finite_number(float_reference_logprob)
        ):
            float_deltas.append(
                abs(float(float_selected_logprob) - float(float_reference_logprob))
            )
        if (
            model_token == float_token
            and model_token is not None
            and isinstance(model_selected_logprob, (int, float))
            and isinstance(float_selected_logprob, (int, float))
        ):
            selected_logprob_deltas.append(
                abs(float(model_selected_logprob) - float(float_selected_logprob))
            )
        model_top = model_row.get("engine_top_logprobs") or {}
        float_top = float_row.get("engine_top_logprobs") or {}
        if isinstance(model_top, dict) and isinstance(float_top, dict):
            for token_id in set(model_top) & set(float_top):
                model_value = model_top[token_id]
                float_value = float_top[token_id]
                if isinstance(model_value, (int, float)) and isinstance(
                    float_value, (int, float)
                ):
                    top_logprob_deltas.append(
                        abs(float(model_value) - float(float_value))
                    )
        if model_token == float_token:
            continue
        if not model_hit and float_hit:
            movement = "toward_reference"
            toward += 1
        elif model_hit and not float_hit:
            movement = "away_from_reference"
            away += 1
        else:
            movement = "changed_other"
            changed_other += 1
        changed.append(
            {
                "prompt": key[0],
                "position": key[1],
                "reference_token": reference,
                "model_token": model_token,
                "float32_token": float_token,
                "movement": movement,
            }
        )
    delta = float_agreed - model_agreed
    model_floor = (model_result.get("reference_noise_floor") or {}).get(
        "raw_self_agreement_rate"
    )
    float_floor = (float32_result.get("reference_noise_floor") or {}).get(
        "raw_self_agreement_rate"
    )
    model_tie_gap = (model_result.get("logprob_tolerance") or {}).get(
        "tie_gap_nats"
    )
    float_tie_gap = (float32_result.get("logprob_tolerance") or {}).get(
        "tie_gap_nats"
    )
    model_top_k = (model_result.get("logprob_tolerance") or {}).get("top_k")
    float_top_k = (float32_result.get("logprob_tolerance") or {}).get("top_k")
    reference_contract_match = (
        isinstance(model_floor, (int, float))
        and model_floor == float_floor
        and isinstance(model_tie_gap, (int, float))
        and model_tie_gap == float_tie_gap
        and type(model_top_k) is int
        and model_top_k > 0
        and model_top_k == float_top_k
    )
    return {
        "complete": complete and not reference_mismatches,
        "positions": len(model_rows),
        "reference_mismatches": reference_mismatches,
        "reference_contract_match": reference_contract_match,
        "reference_self_agreement_floor": (
            model_floor if reference_contract_match else None
        ),
        "tie_gap_nats": model_tie_gap if reference_contract_match else None,
        "top_logprobs_k": model_top_k if reference_contract_match else None,
        "model_agreed": model_agreed,
        "float32_agreed": float_agreed,
        "agreement_delta": delta,
        "classification": "improved" if delta > 0 else "worsened" if delta < 0 else "same",
        "toward_reference": toward,
        "away_from_reference": away,
        "changed_other": changed_other,
        "substantive": {
            "model": model_substantive,
            "float32": float_substantive,
        },
        "missing": {"model": model_missing, "float32": float_missing},
        "changed_positions": changed,
        "agreeing_position_logprob_delta": {
            "model": _delta_summary(model_deltas),
            "float32": _delta_summary(float_deltas),
        },
        "same_selected_token_logprob_delta": _delta_summary(
            selected_logprob_deltas
        ),
        "shared_top_token_logprob_delta": _delta_summary(top_logprob_deltas),
    }


def evaluate(
    a1_model: dict[str, Any],
    a1_float32: dict[str, Any],
    a2_model: dict[str, Any],
    a2_float32: dict[str, Any],
    *,
    assembly_code: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    checks = []
    artifacts = {
        ("A1", "model"): a1_model,
        ("A1", "float32"): a1_float32,
        ("A2", "model"): a2_model,
        ("A2", "float32"): a2_float32,
    }
    for (gate, arm), artifact in artifacts.items():
        checks.extend(_validate_arm(artifact, gate, arm))

    for gate in ("A1", "A2"):
        model_evidence = artifacts[(gate, "model")].get("evidence") or {}
        float_evidence = artifacts[(gate, "float32")].get("evidence") or {}
        _record(
            checks,
            f"{gate} arms use the exact same immutable HF reference",
            model_evidence.get("reference") == float_evidence.get("reference"),
            "reference envelopes compared byte-for-byte as parsed JSON",
        )
        if gate == "A1":
            model_cont = model_evidence.get("continuations") or {}
            float_cont = float_evidence.get("continuations") or {}
            _record(
                checks,
                "A1 arms use identical checkpoint and prompt token IDs",
                (model_cont.get("config") or {}).get("checkpoint")
                == (float_cont.get("config") or {}).get("checkpoint")
                and model_cont.get("prompts") == float_cont.get("prompts"),
                "continuation checkpoint/prompt envelopes compared exactly",
            )

        model_continuation, model_teachers = _arm_configs(
            artifacts[(gate, "model")], gate
        )
        float_continuation, float_teachers = _arm_configs(
            artifacts[(gate, "float32")], gate
        )
        if gate == "A1":
            _record(
                checks,
                "A1 continuation arms use identical non-logits measurement config",
                _same_non_arm_config(model_continuation, float_continuation),
                "parsed configs compared exactly after removing only "
                "logits_dtype_requested/logits_dtype_resolved",
            )
        for tp in (_A1_TP if gate == "A1" else _A2_TP):
            _record(
                checks,
                f"{gate} TP{tp} arms use identical non-logits measurement config",
                _same_non_arm_config(
                    model_teachers.get(tp),
                    float_teachers.get(tp),
                ),
                "parsed configs compared exactly after removing only "
                "logits_dtype_requested/logits_dtype_resolved",
            )

    code_rows = [
        code
        for (gate, _arm), artifact in artifacts.items()
        for code in _code_rows(artifact, gate)
    ]
    commits = {code.get("commit") for code in code_rows}
    same_clean_commit = (
        len(commits) == 1
        and None not in commits
        and code_rows
        and all(code.get("dirty") is False for code in code_rows)
    )
    _record(
        checks,
        "all four arms and both shared references use one clean commit",
        same_clean_commit,
        f"commits={sorted(str(commit) for commit in commits)}, rows={len(code_rows)}",
    )
    if assembly_code is not None:
        _record(
            checks,
            "paired assembler itself uses the same clean measurement commit",
            assembly_code.get("dirty") is False
            and assembly_code.get("commit") in commits
            and len(commits) == 1,
            f"assembler={assembly_code}, measurement_commits="
            f"{sorted(str(commit) for commit in commits)}",
        )

    summaries: dict[str, Any] = {}
    for gate, degrees in (("A1", _A1_TP), ("A2", _A2_TP)):
        model_evidence = artifacts[(gate, "model")].get("evidence") or {}
        float_evidence = artifacts[(gate, "float32")].get("evidence") or {}
        for tp in degrees:
            name = f"{gate.lower()}_tp{tp}"
            model_result = model_evidence.get(f"teacher_tp{tp}") or {}
            float_result = float_evidence.get(f"teacher_tp{tp}") or {}
            summary = paired_position_summary(model_result, float_result)
            model_raw, model_unique, model_exact = _raw_position_counts(
                model_result
            )
            float_raw, float_unique, float_exact = _raw_position_counts(
                float_result
            )
            summary["raw_position_integrity"] = {
                "model": {
                    "rows": model_raw,
                    "unique_positions": model_unique,
                    "exact_expected_keys": model_exact,
                },
                "float32": {
                    "rows": float_raw,
                    "unique_positions": float_unique,
                    "exact_expected_keys": float_exact,
                },
            }
            model_logprob_rows, model_logprob_complete, model_logprob_exact = (
                _raw_logprob_counts(model_result, summary["top_logprobs_k"])
            )
            float_logprob_rows, float_logprob_complete, float_logprob_exact = (
                _raw_logprob_counts(float_result, summary["top_logprobs_k"])
            )
            summary["raw_logprob_integrity"] = {
                "top_k": summary["top_logprobs_k"],
                "model": {
                    "rows": model_logprob_rows,
                    "complete_rows": model_logprob_complete,
                    "exact": model_logprob_exact,
                },
                "float32": {
                    "rows": float_logprob_rows,
                    "complete_rows": float_logprob_complete,
                    "exact": float_logprob_exact,
                },
            }
            summaries[name] = summary
            _record(
                checks,
                f"{gate} TP{tp} has complete paired raw positions",
                summary["complete"]
                and summary["positions"] == _EXPECTED_ROWS
                and summary["reference_contract_match"]
                and model_exact
                and float_exact,
                f"paired_positions={summary['positions']}, "
                f"model_rows/unique={model_raw}/{model_unique}, "
                f"float32_rows/unique={float_raw}/{float_unique}, "
                f"reference_mismatches={len(summary['reference_mismatches'])}, "
                f"reference_contract={summary['reference_contract_match']}",
            )
            _record(
                checks,
                f"{gate} TP{tp} retains complete paired raw logprob observations",
                model_logprob_exact
                and float_logprob_exact
                and summary["agreeing_position_logprob_delta"]["model"]["count"]
                == summary["model_agreed"]
                and summary["agreeing_position_logprob_delta"]["float32"]["count"]
                == summary["float32_agreed"],
                f"top_k={summary['top_logprobs_k']}, "
                f"model_complete={model_logprob_complete}/{model_logprob_rows}, "
                f"float32_complete={float_logprob_complete}/{float_logprob_rows}, "
                "agreeing_delta_counts="
                f"{summary['agreeing_position_logprob_delta']['model']['count']}/"
                f"{summary['model_agreed']} and "
                f"{summary['agreeing_position_logprob_delta']['float32']['count']}/"
                f"{summary['float32_agreed']}",
            )

    deltas = [row["agreement_delta"] for row in summaries.values()]
    if any(delta < 0 for delta in deltas):
        quality = "mixed" if any(delta > 0 for delta in deltas) else "negative"
    elif any(delta > 0 for delta in deltas):
        quality = "positive"
    else:
        quality = "no_measurable_improvement"
    return checks, summaries, quality


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-model", type=Path, required=True)
    parser.add_argument("--a1-float32", type=Path, required=True)
    parser.add_argument("--a2-model", type=Path, required=True)
    parser.add_argument("--a2-float32", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "a1_model": args.a1_model,
        "a1_float32": args.a1_float32,
        "a2_model": args.a2_model,
        "a2_float32": args.a2_float32,
    }
    artifacts = {name: json.loads(path.read_text()) for name, path in paths.items()}
    assembly_code = _code_provenance()
    checks, summaries, quality = evaluate(
        **artifacts,
        assembly_code=assembly_code,
    )
    passed = all(check["passed"] for check in checks)
    payload = {
        "schema": 1,
        "gate": "issue #364 paired logits dtype A1/A2",
        "verdict": "PASS" if passed else "FAIL",
        "quality_classification": quality,
        "code": assembly_code,
        "checks": checks,
        "paired_tp_summaries": summaries,
        "source_files": {
            name: {"path": str(path), "sha256": _file_sha256(path)}
            for name, path in paths.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "gate": payload["gate"],
                "verdict": payload["verdict"],
                "quality_classification": quality,
            },
            indent=2,
        )
    )
    for check in checks:
        print(f"[{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}")
    print(f"written: {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
