"""Assemble and enforce the formal G2 A2 70B TP correctness gate.

The expensive inputs come from one HF compressed-FP8 reference and three
``parity_hf.py`` runs at TP=2/4/8. This command fails closed on a partial model,
mixed checkpoint, missing per-position evidence, dirty/mixed code, an amended
HF-relative failure, or a substantive TP4/8 disagreement against TP2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from verification.l1.correctness.parity_hf import (
    _MAX_LOGPROB_DELTA,
    _MIN_TIE_GAP,
    _reference_noise_floor,
)
from verification.l1.correctness.parity_tp import _code_provenance

_PROMPTS = 64
_POSITIONS = 16
_TP_DEGREES = (2, 4, 8)
_EXPECTED_ROWS = _PROMPTS * _POSITIONS


def _record(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _position_map(result: dict[str, Any]) -> dict[tuple[str, int], dict]:
    rows = result.get("raw_positions") or []
    return {
        (str(row.get("prompt")), int(row.get("position"))): row
        for row in rows
        if isinstance(row, dict)
        and row.get("prompt") is not None
        and isinstance(row.get("position"), int)
    }


def _compare_to_tp2(
    base: dict[str, Any],
    candidate: dict[str, Any],
    reference_self_agreement: float,
    tie_gap: float,
) -> dict[str, Any]:
    base_rows = _position_map(base)
    candidate_rows = _position_map(candidate)
    expected_keys = {
        (f"p{prompt}", position)
        for prompt in range(_PROMPTS)
        for position in range(_POSITIONS)
    }
    disagreements = []
    agreed = 0
    for key in sorted(expected_keys):
        base_row = base_rows.get(key) or {}
        candidate_row = candidate_rows.get(key) or {}
        base_token = base_row.get("engine_token")
        candidate_token = candidate_row.get("engine_token")
        if base_token == candidate_token and base_token is not None:
            agreed += 1
            continue
        reference_row = base_row.get("reference_top_logprobs") or {}
        base_value = reference_row.get(str(base_token))
        candidate_value = reference_row.get(str(candidate_token))
        gap = (
            round(abs(base_value - candidate_value), 5)
            if base_value is not None and candidate_value is not None
            else None
        )
        disagreements.append(
            {
                "prompt": key[0],
                "position": key[1],
                "tp2_token": base_token,
                "candidate_token": candidate_token,
                "reference_logprob_gap": gap,
                "outside_reference_top_k": (
                    base_value is None or candidate_value is None
                ),
                "tie_break": gap is not None and gap <= tie_gap,
            }
        )
    substantive = [row for row in disagreements if not row["tie_break"]]
    raw_rate = agreed / _EXPECTED_ROWS
    complete = set(base_rows) == expected_keys and set(candidate_rows) == expected_keys
    threshold_met = raw_rate >= reference_self_agreement
    return {
        "positions": _EXPECTED_ROWS,
        "agreed": agreed,
        "rate": raw_rate,
        "threshold": reference_self_agreement,
        # Informative for the direct topology comparison. A2's two binding
        # criteria are each TP degree vs the shared HF reference (the path
        # parity_hf.py measures); the direct TP comparison additionally fails
        # only on incomplete or substantive divergence.
        "reference_noise_floor_met": threshold_met,
        "tie_gap_nats": tie_gap,
        "disagreements": disagreements,
        "substantive": len(substantive),
        "complete": complete,
        "verdict": (
            "PASS"
            if complete and not substantive
            else "FAIL"
        ),
    }


def _recompute_hf_score(
    result: dict[str, Any],
    reference: dict[str, Any],
    reference_self_agreement: float,
    tie_gap: float,
) -> dict[str, Any]:
    """Recompute both amended criteria from raw rows, never reported verdicts."""
    rows = _position_map(result)
    expected_keys = {
        (f"p{prompt}", position)
        for prompt in range(_PROMPTS)
        for position in range(_POSITIONS)
    }
    agreed = 0
    substantive = []
    missing_logprobs = []
    logprob_deltas = []
    reference_mismatches = []
    for key in sorted(expected_keys):
        row = rows.get(key) or {}
        entry = reference.get(key[0]) or {}
        continuations = entry.get("continuation") or []
        reference_rows = entry.get("hf_top_logprobs") or []
        if key[1] >= len(continuations) or key[1] >= len(reference_rows):
            reference_mismatches.append(key)
            continue
        reference_token = continuations[key[1]]
        reference_row = reference_rows[key[1]]
        if (
            row.get("reference_token") != reference_token
            or row.get("reference_top_logprobs") != reference_row
        ):
            reference_mismatches.append(key)
        engine_token = row.get("engine_token")
        if engine_token == reference_token and engine_token is not None:
            agreed += 1
            reference_logprob = reference_row.get(str(reference_token))
            engine_logprob = row.get("engine_token_logprob")
            if reference_logprob is None or engine_logprob is None:
                missing_logprobs.append(key)
            else:
                logprob_deltas.append(abs(engine_logprob - reference_logprob))
            continue
        reference_value = reference_row.get(str(reference_token))
        engine_value = reference_row.get(str(engine_token))
        gap = (
            reference_value - engine_value
            if reference_value is not None and engine_value is not None
            else None
        )
        if gap is None or gap > tie_gap:
            substantive.append(
                {
                    "prompt": key[0],
                    "position": key[1],
                    "engine_token": engine_token,
                    "reference_token": reference_token,
                    "reference_logprob_gap": gap,
                }
            )
    raw_rate = agreed / _EXPECTED_ROWS
    max_delta = max(logprob_deltas) if logprob_deltas else 0.0
    complete = (
        set(rows) == expected_keys
        and not reference_mismatches
        and all(row.get("engine_token") is not None for row in rows.values())
    )
    return {
        "positions": _EXPECTED_ROWS,
        "agreed": agreed,
        "rate": raw_rate,
        "threshold": reference_self_agreement,
        "tie_gap_nats": tie_gap,
        "substantive": substantive,
        "missing_logprobs": [
            {"prompt": prompt, "position": position}
            for prompt, position in missing_logprobs
        ],
        "max_abs_logprob_delta": max_delta,
        "max_abs_logprob_delta_bound": _MAX_LOGPROB_DELTA,
        "distribution_delta_verdict": (
            "PASS" if max_delta <= _MAX_LOGPROB_DELTA else "DIAGNOSTIC_EXCEEDANCE"
        ),
        "reference_mismatches": [
            {"prompt": prompt, "position": position}
            for prompt, position in reference_mismatches
        ],
        "complete": complete,
        "verdict": (
            "PASS"
            if complete
            and raw_rate >= reference_self_agreement
            and not substantive
            and not missing_logprobs
            else "FAIL"
        ),
    }


def evaluate(
    reference_cache: dict[str, Any],
    teacher_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return fail-closed checks and TP4/8-vs-TP2 comparisons."""
    checks: list[dict[str, Any]] = []
    provenance = reference_cache.get("provenance") or {}
    contract = provenance.get("checkpoint_contract") or {}
    quant = contract.get("quantization_config") or {}
    groups = quant.get("config_groups") or {}
    first_group = next(iter(groups.values()), {})
    weights = first_group.get("weights") or {}
    activations = first_group.get("input_activations") or {}
    is_70b_fp8 = (
        contract.get("architectures") == ["LlamaForCausalLM"]
        and contract.get("hidden_size") == 8192
        and contract.get("intermediate_size") == 28672
        and contract.get("num_hidden_layers") == 80
        and contract.get("num_attention_heads") == 64
        and contract.get("num_key_value_heads") == 8
        and contract.get("vocab_size") == 128256
        and quant.get("quant_method") == "compressed-tensors"
        and weights.get("type") == "float"
        and weights.get("num_bits") == 8
        and activations.get("type") == "float"
        and activations.get("num_bits") == 8
        and activations.get("dynamic") is True
    )
    _record(
        checks,
        "checkpoint is Llama-3.3-70B FP8 W8A8",
        is_70b_fp8,
        json.dumps(contract, sort_keys=True),
    )

    digests = provenance.get("checkpoint_weight_files") or {}
    tensor_digests = {
        name: digest
        for name, digest in digests.items()
        if str(name).endswith(".safetensors")
    }
    _record(
        checks,
        "complete checkpoint digests are pinned",
        len(tensor_digests) == 15
        and all(
            isinstance(digest, str) and len(digest) == 64
            for digest in tensor_digests.values()
        ),
        f"{len(tensor_digests)} safetensors digest(s)",
    )

    reference = reference_cache.get("reference") or {}
    expected_ids = {f"p{index}" for index in range(_PROMPTS)}
    reference_complete = (
        provenance.get("num_prompts") == _PROMPTS
        and provenance.get("positions") == _POSITIONS
        and set(reference) == expected_ids
        and all(
            len(row.get("continuation") or []) == _POSITIONS
            and len(row.get("hf_top_logprobs") or []) == _POSITIONS
            for row in reference.values()
        )
    )
    _record(
        checks,
        "HF compressed-FP8 reference is complete",
        reference_complete,
        f"{len(reference)} prompt(s) x {provenance.get('positions')!r} positions",
    )

    reference_runtime = reference_cache.get("reference_runtime") or {}
    repo = reference_runtime.get("checkpoint_repo")
    revision = reference_runtime.get("checkpoint_revision")
    source_pinned = (
        isinstance(repo, str)
        and bool(repo)
        and isinstance(revision, str)
        and len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision)
    )
    _record(
        checks,
        "Hub checkpoint source and immutable revision are pinned",
        source_pinned,
        f"{repo}@{revision}",
    )

    by_tp = {
        (row.get("config") or {}).get("tensor_parallel_size"): row
        for row in teacher_results
        if isinstance(row, dict)
    }
    complete_runs = set(by_tp) == set(_TP_DEGREES)
    if complete_runs:
        complete_runs = all(
            (row.get("config") or {}).get("reference_provenance") == provenance
            and (row.get("config") or {}).get("checkpoint_repo") == repo
            and (row.get("config") or {}).get("checkpoint_revision") == revision
            and (row.get("config") or {}).get("num_prompts") == _PROMPTS
            and (row.get("config") or {}).get("positions") == _POSITIONS
            and (row.get("config") or {}).get("dtype") == "bfloat16"
            and (row.get("agreement") or {}).get("positions") == _EXPECTED_ROWS
            and len(row.get("raw_positions") or []) == _EXPECTED_ROWS
            and len(_position_map(row)) == _EXPECTED_ROWS
            for row in by_tp.values()
        )
    _record(
        checks,
        "TP2, TP4, and TP8 retain every raw position",
        complete_runs,
        f"TP degrees={sorted(value for value in by_tp if value is not None)}",
    )

    amended_pass = set(by_tp) == set(_TP_DEGREES) and all(
        (row.get("agreement") or {}).get("verdict") == "PASS"
        and (row.get("logprob_tolerance") or {}).get("substantive") == 0
        and not (row.get("logprob_tolerance") or {}).get("missing_samples")
        for row in by_tp.values()
    )
    _record(
        checks,
        "every TP degree passes the amended HF-relative criteria",
        amended_pass,
        ", ".join(
            f"TP{tp}: agreement="
            f"{(by_tp.get(tp, {}).get('agreement') or {}).get('verdict')}, "
            f"substantive="
            f"{(by_tp.get(tp, {}).get('logprob_tolerance') or {}).get('substantive')}"
            for tp in _TP_DEGREES
        ),
    )

    runtime_complete = set(by_tp) == set(_TP_DEGREES) and all(
        ((row.get("config") or {}).get("hardware") or {}).get("arch") == "cuda"
        and bool(
            (
                ((row.get("config") or {}).get("hardware") or {}).get("runtime")
                or {}
            ).get("nccl")
        )
        and bool(
            (
                ((row.get("config") or {}).get("hardware") or {}).get("runtime")
                or {}
            ).get("nvidia_smi")
        )
        and bool(
            (
                ((row.get("config") or {}).get("hardware") or {}).get("runtime")
                or {}
            ).get("nvidia_smi_topology")
        )
        for row in by_tp.values()
    )
    _record(
        checks,
        "CUDA, NCCL, and physical topology are recorded",
        runtime_complete,
        "all TP result configs carry runtime and nvidia-smi topology",
    )

    reference_code = reference_runtime.get("code") or {}
    codes = [reference_code] + [
        (row.get("config") or {}).get("code") or {} for row in by_tp.values()
    ]
    commits = {code.get("commit") for code in codes}
    clean_code = (
        len(commits) == 1
        and None not in commits
        and len(codes) == len(_TP_DEGREES) + 1
        and all(code.get("dirty") is False for code in codes)
    )
    _record(
        checks,
        "all measurements use one clean commit",
        clean_code,
        f"commits={sorted(str(commit) for commit in commits)}",
    )

    noise_floor = _reference_noise_floor(reference)
    tie_gap = max(_MIN_TIE_GAP, noise_floor["max_gap_at_self_disagreement"])
    comparisons = {}
    if set(by_tp) == set(_TP_DEGREES):
        for tp in _TP_DEGREES:
            comparisons[f"tp{tp}_vs_hf"] = _recompute_hf_score(
                by_tp[tp],
                reference,
                noise_floor["raw_self_agreement_rate"],
                tie_gap,
            )
        for tp in (4, 8):
            comparisons[f"tp{tp}_vs_tp2"] = _compare_to_tp2(
                by_tp[2],
                by_tp[tp],
                noise_floor["raw_self_agreement_rate"],
                tie_gap,
            )
    recomputed_hf_pass = all(
        comparisons.get(f"tp{tp}_vs_hf", {}).get("verdict") == "PASS"
        for tp in _TP_DEGREES
    )
    _record(
        checks,
        "raw rows recompute both amended criteria at every TP degree",
        recomputed_hf_pass,
        ", ".join(
            f"TP{tp}={comparisons.get(f'tp{tp}_vs_hf', {}).get('verdict')}"
            for tp in _TP_DEGREES
        ),
    )
    direct_pass = all(
        comparisons.get(f"tp{tp}_vs_tp2", {}).get("verdict") == "PASS"
        for tp in (4, 8)
    )
    _record(
        checks,
        "TP4 and TP8 pass directly against TP2",
        direct_pass,
        ", ".join(
            f"{name}={row['verdict']} ({row['agreed']}/{row['positions']}, "
            f"substantive={row['substantive']})"
            for name, row in comparisons.items()
            if name.endswith("_vs_tp2")
        ),
    )
    return checks, comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--teacher-tp2", type=Path, required=True)
    parser.add_argument("--teacher-tp4", type=Path, required=True)
    parser.add_argument("--teacher-tp8", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    inputs = {
        "reference": args.reference,
        "teacher_tp2": args.teacher_tp2,
        "teacher_tp4": args.teacher_tp4,
        "teacher_tp8": args.teacher_tp8,
    }
    evidence = {name: json.loads(path.read_text()) for name, path in inputs.items()}
    checks, comparisons = evaluate(
        evidence["reference"],
        [
            evidence["teacher_tp2"],
            evidence["teacher_tp4"],
            evidence["teacher_tp8"],
        ],
    )
    passed = all(check["passed"] for check in checks)
    payload = {
        "schema": 1,
        "gate": "G2 A2",
        "verdict": "PASS" if passed else "FAIL",
        "code": _code_provenance(),
        "reference_noise_floor": _reference_noise_floor(
            evidence["reference"].get("reference") or {}
        ),
        "tp_comparisons": comparisons,
        "source_files": {
            name: {"path": str(path), "sha256": _file_sha256(path)}
            for name, path in inputs.items()
        },
        "checks": checks,
        "evidence": evidence,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"gate": payload["gate"], "verdict": payload["verdict"]}, indent=2))
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] {check['name']}: {check['detail']}")
    print(f"written: {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
