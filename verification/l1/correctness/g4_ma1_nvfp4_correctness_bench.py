#!/usr/bin/env python3
"""Run and aggregate opt-in NVFP4 accuracy profiles against G4 M-A1.

This companion leaves the retained formal M-A1 schema unchanged. Each ``run``
reuses its pinned reference teacher rollout, runs the same 1,024 positions on
one Kairyu EP profile, and records agreement, resident model bytes, and
projection-local activation saturation. ``curve`` combines those immutable
profile reports into the requested accuracy/memory trade-off artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if __package__ in {None, ""}:
    sys.path.insert(0, str(_REPO_ROOT))

from evidence.reporting import atomic_write_json  # noqa: E402
from kairyu.engine.core.nvfp4_accuracy import NvFp4AccuracyProfile  # noqa: E402
from verification.l1.correctness import g4_ma1_qwen3_235b_nvfp4_bench as gate  # noqa: E402
from verification.l1.correctness import g4_ma1_qwen3_235b_nvfp4_capture as capture  # noqa: E402

REPORT_SCHEMA = "kairyu.g4.ma1.nvfp4-accuracy-profile.v1"
CURVE_SCHEMA = "kairyu.g4.ma1.nvfp4-accuracy-curve.v1"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(), object_pairs_hook=gate._strict_object)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _profile_digest(profile: NvFp4AccuracyProfile) -> str:
    return gate.sha256_json(profile.as_dict())


def _teacher_index(rows: Sequence[Mapping[str, object]]):
    return {
        (row["prompt_id"], row["position"]): row
        for row in rows
    }


def _gate_passed(score: Mapping[str, object]) -> bool:
    selected = score.get("max_selected_logprob_abs_delta")
    reciprocal = score.get("max_reciprocal_cross_selected_abs_delta")
    return bool(
        score.get("complete") is True
        and score.get("agreed", 0) >= gate.AGREEMENT_COUNT_MIN
        and score.get("substantive_disagreements") == 0
        and isinstance(selected, (int, float))
        and selected <= gate.LOGPROB_ABS_DELTA_MAX
        and isinstance(reciprocal, (int, float))
        and reciprocal <= gate.LOGPROB_ABS_DELTA_MAX
    )


def run_profile(
    *,
    name: str,
    profile: NvFp4AccuracyProfile,
    model_path: Path,
    world_size: int,
    host_snapshot: Mapping[str, object],
    reference_fragment: Mapping[str, object],
    container_observation: Mapping[str, object],
    launcher_type=None,
    tokenizer=None,
) -> dict[str, object]:
    if not name or any(character.isspace() for character in name):
        raise ValueError("profile name must be non-empty and contain no whitespace")
    if world_size not in gate.WORLD_SIZES:
        raise ValueError("profile world_size must be 2 or 4")
    reference_arm = gate.REFERENCE_ARM[world_size]
    if not capture._host_snapshot_valid(host_snapshot):
        raise ValueError("invalid G4 M-A1 host snapshot")
    if not capture._fragment_valid(
        reference_fragment,
        arm=reference_arm,
        host_snapshot=host_snapshot,
    ) or reference_fragment["failures"]:
        raise ValueError(f"valid successful {reference_arm} fragment is required")
    if (
        not gate._container_observation_valid(container_observation)
        or container_observation["image_repo_digest"]
        != host_snapshot["image_repo_digest"]
        or container_observation["image_config_id"]
        != host_snapshot["image_config_id"]
    ):
        raise ValueError("profile run requires matching container evidence")
    if tokenizer is None:
        from kairyu.engine.tokenizer import resolve_tokenizer

        tokenizer = resolve_tokenizer(str(model_path))
    if launcher_type is None:
        from kairyu.engine.core.worker import DistEPLauncher

        launcher_type = DistEPLauncher

    prompt_rows = host_snapshot["prompts"]
    prompt_tokens = [list(row["token_ids"]) for row in prompt_rows]
    canonical = capture._reference_teacher_canonical(
        reference_fragment=reference_fragment,
        prompt_tokens=prompt_tokens,
    )
    num_pages = capture._formal_num_pages(
        prompts=prompt_tokens,
        max_tokens=gate.POSITIONS,
        page_size=gate.KAIRYU_KV_PAGE_SIZE,
    )
    checkpoint_start = capture._checkpoint_snapshot(model_path)
    if checkpoint_start != host_snapshot["checkpoint"]:
        raise RuntimeError("profile checkpoint differs from the prepared snapshot")
    started_at = _now()
    launcher = None
    try:
        launcher = launcher_type(
            str(model_path),
            world_size,
            num_pages,
            gate.KAIRYU_KV_PAGE_SIZE,
            tokenizer.vocab(),
            pipeline_depth=1,
            decode_mode="eager",
            kv_cache_dtype="bfloat16",
            nvfp4_accuracy_profile=profile.as_dict() if profile.active else None,
        )
        before = launcher.nvfp4_accuracy_metadata(reset=True)
        free_outputs = capture._run_kairyu_batch(
            tokenizer=tokenizer,
            runner=launcher.runner,
            prompts=prompt_tokens,
            max_tokens=gate.POSITIONS,
            num_pages=num_pages,
            page_size=gate.KAIRYU_KV_PAGE_SIZE,
            request_prefix=f"accuracy-{name}-free",
        )
        free_rows, _ = capture._free_rows_from_outputs(
            arm=gate.KAIRYU_ARM[world_size],
            outputs=free_outputs,
        )
        teacher_rows: list[dict[str, object]] = []
        for position in range(gate.POSITIONS):
            prefixes = [
                [*prompt_tokens[index], *canonical[index][:position]]
                for index in range(gate.PROMPT_COUNT)
            ]
            outputs = capture._run_kairyu_batch(
                tokenizer=tokenizer,
                runner=launcher.runner,
                prompts=prefixes,
                max_tokens=1,
                num_pages=num_pages,
                page_size=gate.KAIRYU_KV_PAGE_SIZE,
                request_prefix=f"accuracy-{name}-teacher-{position}",
            )
            teacher_rows.extend(
                capture._teacher_rows_from_outputs(
                    arm=gate.KAIRYU_ARM[world_size],
                    position=position,
                    outputs=outputs,
                    prompt_tokens=prompt_tokens,
                    canonical=canonical,
                )
            )
        after = launcher.nvfp4_accuracy_metadata()
    finally:
        if launcher is not None:
            launcher.shutdown()
    checkpoint_end = capture._checkpoint_snapshot(model_path)
    if checkpoint_end != checkpoint_start:
        raise RuntimeError("profile checkpoint changed during execution")

    score = gate._score_cell(
        _teacher_index(reference_fragment["teacher"]),
        _teacher_index(teacher_rows),
    )
    resident = [int(row["resident_model_bytes"]) for row in after]
    return {
        "schema_version": REPORT_SCHEMA,
        "gate": gate.GATE,
        "profile": {
            "name": name,
            "config": profile.as_dict(),
            "sha256": _profile_digest(profile),
        },
        "world_size": world_size,
        "model": gate.MODEL,
        "model_revision": gate.MODEL_REVISION,
        "source_commit": host_snapshot["source"]["commit"],
        "checkpoint": checkpoint_start,
        "started_at": started_at,
        "finished_at": _now(),
        "score": score,
        "passed": _gate_passed(score),
        "memory": {
            "resident_model_bytes_per_rank": resident,
            "resident_model_bytes_total": sum(resident),
            "resident_model_bytes_max_rank": max(resident),
        },
        "runtime_before": list(before),
        "runtime_after": list(after),
        "free_running": free_rows,
        "teacher": teacher_rows,
        "container_observation": dict(container_observation),
    }


def build_curve(reports: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not reports:
        raise ValueError("accuracy curve requires at least one profile report")
    first = reports[0]
    identity = (
        first.get("gate"),
        first.get("model"),
        first.get("model_revision"),
        first.get("source_commit"),
        first.get("checkpoint"),
        first.get("world_size"),
    )
    names: set[str] = set()
    points: list[dict[str, object]] = []
    for report in reports:
        if report.get("schema_version") != REPORT_SCHEMA:
            raise ValueError("accuracy curve input has an unsupported schema")
        current = (
            report.get("gate"),
            report.get("model"),
            report.get("model_revision"),
            report.get("source_commit"),
            report.get("checkpoint"),
            report.get("world_size"),
        )
        if current != identity:
            raise ValueError("accuracy curve reports do not share one run identity")
        profile = report.get("profile")
        score = report.get("score")
        memory = report.get("memory")
        if (
            not isinstance(profile, Mapping)
            or not isinstance(score, Mapping)
            or not isinstance(memory, Mapping)
        ):
            raise ValueError("accuracy profile report is malformed")
        name = profile.get("name")
        if not isinstance(name, str) or name in names:
            raise ValueError("accuracy curve profile names must be unique strings")
        names.add(name)
        points.append(
            {
                "name": name,
                "profile_sha256": profile.get("sha256"),
                "passed": report.get("passed"),
                "agreement_rate": score.get("agreement_rate"),
                "substantive_disagreements": score.get("substantive_disagreements"),
                "resident_model_bytes_total": memory.get("resident_model_bytes_total"),
                "resident_model_bytes_max_rank": memory.get("resident_model_bytes_max_rank"),
            }
        )
    points.sort(key=lambda point: (point["resident_model_bytes_total"], point["name"]))
    return {
        "schema_version": CURVE_SCHEMA,
        "generated_at": _now(),
        "gate": identity[0],
        "model": identity[1],
        "model_revision": identity[2],
        "source_commit": identity[3],
        "checkpoint": identity[4],
        "world_size": identity[5],
        "points": points,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--name", required=True)
    run.add_argument("--profile", type=Path, required=True)
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--world-size", type=int, choices=gate.WORLD_SIZES, required=True)
    run.add_argument("--host-snapshot", type=Path, required=True)
    run.add_argument("--reference-fragment", type=Path, required=True)
    run.add_argument("--container-inspect", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--assert-gate", action="store_true")
    curve = subparsers.add_parser("curve")
    curve.add_argument("--report", type=Path, action="append", required=True)
    curve.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "curve":
        artifact = build_curve([_read_object(path) for path in args.report])
        atomic_write_json(args.output, artifact)
        print(gate.canonical_json({"curve": str(args.output), "points": len(artifact["points"])}))
        return 0
    profile = NvFp4AccuracyProfile.parse(_read_object(args.profile))
    host = _read_object(args.host_snapshot)
    observation = capture._container_observation(
        args.container_inspect,
        image_repo_digest=host["image_repo_digest"],
        image_config_id=host["image_config_id"],
    )
    report = run_profile(
        name=args.name,
        profile=profile,
        model_path=args.model_path.resolve(),
        world_size=args.world_size,
        host_snapshot=host,
        reference_fragment=_read_object(args.reference_fragment),
        container_observation=observation,
    )
    atomic_write_json(args.output, report)
    print(gate.canonical_json({"report": str(args.output), "passed": report["passed"]}))
    return 0 if report["passed"] or not args.assert_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
