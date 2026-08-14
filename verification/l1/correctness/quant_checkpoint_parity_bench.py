"""Assemble the retained INT8/AWQ/GPTQ real-checkpoint parity gate.

Each arm is produced by ``verification/l1/correctness/parity_hf.py`` from the same pinned BF16
Qwen2.5-1.5B-Instruct reference with 64 prompts, 16 teacher-forced positions,
and top-64 logprobs.  This verifier recomputes the binding completeness and
substantive-disagreement verdict from the raw positions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from verification.l1.correctness.parity_hf import _load_reference, _reference_noise_floor

REFERENCE_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
REFERENCE_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PINS = {
    "int8": (
        "RedHatAI/Qwen2.5-1.5B-Instruct-quantized.w8a8",
        "876ae8e2b9982d9603afd3c71add6b3b4b0bc0b6",
    ),
    "awq": (
        "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        "3ecffa0ceb27851800f45519bab9c457a04405e1",
    ),
    "gptq": (
        "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
        "4f5a0e31008e2966e8f787964bcc4e1105e03dc3",
    ),
}
PROMPTS = 64
POSITIONS = 16
EXPECTED_POSITIONS = PROMPTS * POSITIONS
TOP_LOGPROBS = 64


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _validate_quantization(scheme: str, value: object) -> None:
    _require(isinstance(value, dict), f"{scheme} quantization_config is missing")
    if scheme == "awq":
        _require(
            value.get("quant_method") == "awq"
            and value.get("bits") == 4
            and value.get("group_size") == 128
            and value.get("version") == "gemm"
            and value.get("zero_point") is True,
            "AWQ checkpoint ABI differs from the pinned GEMM W4A16 contract",
        )
    elif scheme == "gptq":
        _require(
            value.get("quant_method") == "gptq"
            and value.get("bits") == 4
            and value.get("group_size") == 128
            and value.get("desc_act") is False
            and value.get("sym") is True,
            "GPTQ checkpoint ABI differs from the pinned v1 W4A16 contract",
        )
    else:
        group = (value.get("config_groups") or {}).get("group_0") or {}
        weights = group.get("weights") or {}
        activations = group.get("input_activations") or {}
        _require(
            value.get("quant_method") == "compressed-tensors"
            and value.get("format") == "int-quantized"
            and weights.get("num_bits") == 8
            and weights.get("strategy") == "channel"
            and weights.get("symmetric") is True
            and activations.get("num_bits") == 8
            and activations.get("strategy") == "token"
            and activations.get("dynamic") is True
            and activations.get("symmetric") is True,
            "INT8 checkpoint ABI differs from the pinned dynamic W8A8 contract",
        )


def _score_arm(scheme: str, payload: dict, reference_provenance: dict) -> dict:
    config = payload.get("config") or {}
    repo, revision = PINS[scheme]
    _require(payload.get("schema_version") == 5, f"{scheme} schema differs")
    _require(
        config.get("comparison_kind")
        == "quantized-candidate-vs-unquantized-reference",
        f"{scheme} is not a cross-checkpoint parity arm",
    )
    _require(config.get("checkpoint_repo") == repo, f"{scheme} repo differs")
    _require(config.get("checkpoint_revision") == revision, f"{scheme} revision differs")
    _require(
        config.get("reference_checkpoint_repo") == REFERENCE_REPO
        and config.get("reference_checkpoint_revision") == REFERENCE_REVISION,
        f"{scheme} BF16 reference pin differs",
    )
    _require(
        config.get("reference_provenance") == reference_provenance,
        f"{scheme} reference provenance differs",
    )
    _require(
        config.get("num_prompts") == PROMPTS
        and config.get("positions") == POSITIONS
        and config.get("tensor_parallel_size") == 1,
        f"{scheme} workload differs from the 64x16 TP1 contract",
    )
    code = config.get("code") or {}
    _require(code.get("dirty") is False, f"{scheme} source is dirty")
    candidate = config.get("candidate_provenance") or {}
    _require(
        candidate.get("recorded_top_logprobs") == TOP_LOGPROBS,
        f"{scheme} did not retain top-64 distributions",
    )
    contract = candidate.get("checkpoint_contract") or {}
    _validate_quantization(scheme, contract.get("quantization_config"))

    rows = payload.get("raw_positions")
    _require(isinstance(rows, list), f"{scheme} raw positions are missing")
    expected = {
        (f"p{prompt}", position)
        for prompt in range(PROMPTS)
        for position in range(POSITIONS)
    }
    keys = {(row.get("prompt"), row.get("position")) for row in rows}
    _require(len(rows) == len(keys), f"{scheme} raw positions contain duplicates")
    _require(keys == expected, f"{scheme} raw position inventory differs")
    gate = payload.get("cross_checkpoint_gate") or {}
    tie_gap = gate.get("tie_gap_nats")
    _require(
        isinstance(tie_gap, (int, float)) and math.isfinite(tie_gap) and tie_gap >= 0.125,
        f"{scheme} tie floor is invalid",
    )
    agreed = 0
    substantive = 0
    missing = 0
    for row in rows:
        reference_token = row.get("reference_token")
        candidate_token = row.get("engine_token")
        if candidate_token is None:
            missing += 1
            substantive += 1
            continue
        if candidate_token == reference_token:
            agreed += 1
            continue
        top = row.get("reference_top_logprobs") or {}
        selected = top.get(str(reference_token))
        alternative = top.get(str(candidate_token))
        if not isinstance(selected, (int, float)) or not isinstance(
            alternative, (int, float)
        ):
            substantive += 1
            continue
        substantive += int(selected - alternative > tie_gap)
    passed = missing == 0 and substantive == 0
    _require(
        gate.get("expected_positions") == EXPECTED_POSITIONS
        and gate.get("captured_positions") == EXPECTED_POSITIONS
        and gate.get("missing_predictions") == missing
        and gate.get("substantive_disagreements") == substantive
        and gate.get("passed") is passed,
        f"{scheme} stored gate differs from raw replay",
    )
    return {
        "checkpoint_repo": repo,
        "checkpoint_revision": revision,
        "checkpoint_weight_files": candidate.get("checkpoint_weight_files"),
        "agreed": agreed,
        "agreement_rate": agreed / EXPECTED_POSITIONS,
        "substantive_disagreements": substantive,
        "missing_predictions": missing,
        "tie_gap_nats": tie_gap,
        "max_agreeing_logprob_abs_delta_diagnostic": (
            payload.get("logprob_tolerance") or {}
        ).get("agreeing_positions_max_abs_delta"),
        "passed": passed,
    }


def verify(reference_path: Path, arm_paths: dict[str, Path]) -> dict:
    reference_payload = _load(reference_path)
    provenance = reference_payload.get("provenance") or {}
    _require(
        (reference_payload.get("reference_runtime") or {}).get("checkpoint_repo")
        == REFERENCE_REPO,
        "reference repo differs",
    )
    _require(
        (reference_payload.get("reference_runtime") or {}).get("checkpoint_revision")
        == REFERENCE_REVISION,
        "reference revision differs",
    )
    _require(
        provenance.get("recorded_top_logprobs") == TOP_LOGPROBS,
        "reference is not top-64",
    )
    reference = _load_reference(reference_path, provenance)
    noise_floor = _reference_noise_floor(reference)
    arms = {
        scheme: _score_arm(scheme, _load(arm_paths[scheme]), provenance)
        for scheme in ("int8", "awq", "gptq")
    }
    _require(
        {arm["tie_gap_nats"] for arm in arms.values()}
        == {max(0.125, noise_floor["max_gap_at_self_disagreement"])},
        "arm tie floors differ from the measured reference floor",
    )
    commits = {
        (_load(arm_paths[scheme]).get("config") or {}).get("code", {}).get("commit")
        for scheme in arms
    }
    _require(len(commits) == 1 and None not in commits, "arm source commits differ")
    passed = all(arm["passed"] for arm in arms.values())
    return {
        "schema_version": 1,
        "gate": "issue-356-real-checkpoint-quant-parity",
        "workload": {
            "prompts": PROMPTS,
            "positions_per_prompt": POSITIONS,
            "expected_positions_per_arm": EXPECTED_POSITIONS,
            "top_logprobs": TOP_LOGPROBS,
        },
        "reference": {
            "checkpoint_repo": REFERENCE_REPO,
            "checkpoint_revision": REFERENCE_REVISION,
            "checkpoint_weight_files": provenance.get("checkpoint_weight_files"),
            "noise_floor": {
                "self_agreement_rate": noise_floor["raw_self_agreement_rate"],
                "tie_gap_nats": next(iter(arms.values()))["tie_gap_nats"],
            },
        },
        "source_commit": next(iter(commits)),
        "arms": arms,
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--awq", type=Path, required=True)
    parser.add_argument("--gptq", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = verify(
        args.reference,
        {"int8": args.int8, "awq": args.awq, "gptq": args.gptq},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
