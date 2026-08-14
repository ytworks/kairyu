"""Assemble and enforce the formal G2 A1 correctness gate.

The expensive measurements stay in their focused harnesses:

* ``parity_tp.py`` records full free-running continuations for TP=1/2 with
  overlap OFF/ON.
* ``parity_hf.py`` records teacher-forced agreement and logprob tolerance for
  TP=1 and TP=2 against one HF reference.

This command rejects partial, stale, or cross-checkpoint inputs and writes one
self-contained review artifact. It exits non-zero unless every amended A1
criterion passes.

Run:
  uv run python verification/l1/correctness/gate_a1.py \
    --continuations bench/results/a1-continuations.json \
    --reference bench/results/a1-hf-reference.json \
    --teacher-tp1 bench/results/a1-teacher-tp1.json \
    --teacher-tp2 bench/results/a1-teacher-tp2.json \
    --out bench/results/g2-a1-llama31-8b-<date>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from verification.l1.correctness.parity_tp import _code_provenance

_PROMPTS = 64
_POSITIONS = 16
_TP_DEGREES = [1, 2]
_RUN_KEYS = {
    f"tp{tp}_overlap_{mode}" for tp in _TP_DEGREES for mode in ("off", "on")
}


def _record(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _weight_map(checkpoint: dict[str, Any]) -> dict[str, str]:
    return {
        str(entry.get("file")): str(entry.get("sha256"))
        for entry in checkpoint.get("weights", [])
        if isinstance(entry, dict) and entry.get("file") and entry.get("sha256")
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(
    continuations: dict[str, Any],
    reference_cache: dict[str, Any],
    teacher_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return every A1 check; callers gate on all entries being true."""
    checks: list[dict[str, Any]] = []
    config = continuations.get("config") or {}
    checkpoint = config.get("checkpoint") or {}
    architecture = checkpoint.get("architecture") or {}
    hardware = config.get("hardware") or {}
    runtime = hardware.get("runtime") or {}

    llama31_8b = (
        architecture.get("model_type") == "llama"
        and architecture.get("hidden_size") == 4096
        and architecture.get("num_hidden_layers") == 32
        and architecture.get("num_attention_heads") == 32
        and architecture.get("num_key_value_heads") == 8
        and architecture.get("vocab_size") == 128256
    )
    _record(
        checks,
        "checkpoint is Llama-3.1-8B",
        llama31_8b,
        f"name={checkpoint.get('name')!r}, architecture={architecture!r}",
    )

    weight_map = _weight_map(checkpoint)
    _record(
        checks,
        "checkpoint has exact weight digests",
        bool(weight_map) and all(len(value) == 64 for value in weight_map.values()),
        f"{len(weight_map)} safetensors digest(s)",
    )
    _record(
        checks,
        "real CUDA/NCCL runtime is recorded",
        hardware.get("arch") == "cuda"
        and bool(runtime.get("torch_cuda"))
        and bool(runtime.get("nccl"))
        and bool(runtime.get("nvidia_smi")),
        (
            f"arch={hardware.get('arch')!r}, CUDA={runtime.get('torch_cuda')!r}, "
            f"NCCL={runtime.get('nccl')!r}, GPUs={hardware.get('device_count')!r}"
        ),
    )
    _record(
        checks,
        "TP and overlap sweep is complete",
        config.get("tp_degrees") == _TP_DEGREES
        and config.get("overlap_modes") == ["off", "on"],
        (
            f"tp={config.get('tp_degrees')!r}, "
            f"overlap={config.get('overlap_modes')!r}"
        ),
    )
    _record(
        checks,
        "greedy generation contract is pinned",
        config.get("num_prompts") == _PROMPTS
        and config.get("max_new_tokens") == _POSITIONS
        and (config.get("sampling") or {}).get("method") == "greedy",
        (
            f"prompts={config.get('num_prompts')!r}, "
            f"tokens={config.get('max_new_tokens')!r}, "
            f"sampling={config.get('sampling')!r}"
        ),
    )

    prompt_rows = continuations.get("prompts") or []
    expected_ids = {f"p{index}" for index in range(_PROMPTS)}
    found_ids = {
        str(row.get("request_id"))
        for row in prompt_rows
        if isinstance(row, dict) and row.get("request_id") is not None
    }
    prompts_complete = (
        len(prompt_rows) == _PROMPTS
        and found_ids == expected_ids
        and all(
            isinstance(row.get("text"), str)
            and bool(row["text"])
            and isinstance(row.get("token_ids"), list)
            and bool(row["token_ids"])
            for row in prompt_rows
            if isinstance(row, dict)
        )
    )
    _record(
        checks,
        "fixed prompts and token IDs are retained",
        prompts_complete,
        f"{len(prompt_rows)} prompt row(s), {len(found_ids)} unique request ID(s)",
    )

    raw_runs = continuations.get("continuations") or {}
    raw_complete = set(raw_runs) == _RUN_KEYS
    if raw_complete:
        for run in raw_runs.values():
            raw_complete = raw_complete and set(run) == expected_ids and all(
                isinstance(tokens, list)
                and len(tokens) == _POSITIONS
                and all(isinstance(token, int) for token in tokens)
                for tokens in run.values()
            )
    _record(
        checks,
        "all raw continuations are retained",
        raw_complete,
        f"runs={sorted(raw_runs)}, expected={sorted(_RUN_KEYS)}",
    )

    overlap = continuations.get("overlap_equality") or {}
    overlap_exact = set(overlap) == {
        "tp1_overlap_on_vs_off",
        "tp2_overlap_on_vs_off",
    } and all(isinstance(row, dict) for row in overlap.values()) and all(
        row.get("prompts") == _PROMPTS
        and row.get("exact_match") == _PROMPTS
        and row.get("first_mismatch") is None
        for row in overlap.values()
    )
    _record(
        checks,
        "overlap ON reproduces OFF at TP1 and TP2",
        overlap_exact,
        json.dumps(overlap, sort_keys=True),
    )

    ref_provenance = reference_cache.get("provenance") or {}
    reference = reference_cache.get("reference") or {}
    reference_complete = (
        ref_provenance.get("num_prompts") == _PROMPTS
        and ref_provenance.get("positions") == _POSITIONS
        and set(reference) == expected_ids
        and all(
            isinstance(row, dict)
            and len(row.get("continuation") or []) == _POSITIONS
            and len(row.get("hf_top_logprobs") or []) == _POSITIONS
            for row in reference.values()
        )
    )
    _record(
        checks,
        "HF reference is complete",
        reference_complete,
        (
            f"prompts={ref_provenance.get('num_prompts')!r}, "
            f"positions={ref_provenance.get('positions')!r}, "
            f"rows={len(reference)}"
        ),
    )

    prompt_by_id = {
        row["request_id"]: row
        for row in prompt_rows
        if isinstance(row, dict) and "request_id" in row
    }
    same_prompts = set(prompt_by_id) == set(reference)
    if same_prompts:
        same_prompts = all(
            prompt_by_id[name].get("text") == reference[name].get("prompt")
            and prompt_by_id[name].get("token_ids") == reference[name].get("prompt_ids")
            for name in reference
        )
    _record(
        checks,
        "continuation and teacher-forced runs use identical prompts",
        same_prompts,
        "natural-language text and token IDs match per request",
    )

    reference_weights = ref_provenance.get("checkpoint_weight_files") or {}
    reference_tensor_weights = {
        name: digest
        for name, digest in reference_weights.items()
        if str(name).endswith(".safetensors")
    }
    _record(
        checks,
        "continuation and HF evidence use identical weights",
        bool(weight_map) and weight_map == reference_tensor_weights,
        (
            f"continuation={sorted(weight_map)}, "
            f"reference={sorted(reference_tensor_weights)}"
        ),
    )

    by_tp = {
        (row.get("config") or {}).get("tensor_parallel_size"): row
        for row in teacher_results
        if isinstance(row, dict)
    }
    teacher_complete = set(by_tp) == set(_TP_DEGREES)
    if teacher_complete:
        teacher_complete = all(
            (row.get("config") or {}).get("num_prompts") == _PROMPTS
            and (row.get("config") or {}).get("positions") == _POSITIONS
            and (row.get("config") or {}).get("dtype") == "bfloat16"
            and (row.get("agreement") or {}).get("positions")
            == _PROMPTS * _POSITIONS
            and (row.get("config") or {}).get("reference_provenance")
            == ref_provenance
            for row in by_tp.values()
        )
    _record(
        checks,
        "teacher-forced TP1 and TP2 evidence is complete",
        teacher_complete,
        f"TP degrees found={sorted(value for value in by_tp if value is not None)}",
    )

    amended_verdicts = set(by_tp) == set(_TP_DEGREES) and all(
        (row.get("agreement") or {}).get("verdict") == "PASS"
        and (row.get("logprob_tolerance") or {}).get("verdict") == "PASS"
        for row in by_tp.values()
    )
    _record(
        checks,
        "amended teacher-forced verdict passes at TP1 and TP2",
        amended_verdicts,
        ", ".join(
            f"TP{tp}: agreement={(by_tp.get(tp, {}).get('agreement') or {}).get('verdict')}, "
            f"logprobs={(by_tp.get(tp, {}).get('logprob_tolerance') or {}).get('verdict')}"
            for tp in _TP_DEGREES
        ),
    )

    continuation_code = config.get("code") or {}
    teacher_codes = [(row.get("config") or {}).get("code") or {} for row in by_tp.values()]
    commits = {continuation_code.get("commit")} | {
        code.get("commit") for code in teacher_codes
    }
    code_reproducible = (
        len(commits) == 1
        and None not in commits
        and continuation_code.get("dirty") is False
        and all(code.get("dirty") is False for code in teacher_codes)
    )
    _record(
        checks,
        "all measurements use one clean commit",
        code_reproducible,
        f"commits={sorted(str(commit) for commit in commits)}",
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuations", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--teacher-tp1", type=Path, required=True)
    parser.add_argument("--teacher-tp2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    inputs = {
        "continuations": args.continuations,
        "reference": args.reference,
        "teacher_tp1": args.teacher_tp1,
        "teacher_tp2": args.teacher_tp2,
    }
    evidence = {
        name: json.loads(path.read_text()) for name, path in inputs.items()
    }
    checks = evaluate(
        evidence["continuations"],
        evidence["reference"],
        [evidence["teacher_tp1"], evidence["teacher_tp2"]],
    )
    passed = all(check["passed"] for check in checks)
    payload = {
        "schema": 1,
        "gate": "G2 A1",
        "verdict": "PASS" if passed else "FAIL",
        "code": _code_provenance(),
        "source_files": {
            name: {"path": str(path), "sha256": _file_sha256(path)}
            for name, path in inputs.items()
        },
        "checks": checks,
        # The committed result is self-contained: raw Kairyu/HF continuations,
        # prompt tokens, per-position logprobs, config, and digests all travel
        # with the verdict instead of depending on mutable machine-local files.
        "evidence": evidence,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"gate": payload["gate"], "verdict": payload["verdict"]}, indent=2))
    for check in checks:
        print(f"[{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}")
    print(f"written: {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
