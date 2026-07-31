"""Replay F2 evidence and produce the G5 F4c global-KV-pool decision artifact.

The F2a trace records every initial cache seed and every subsequent replica
placement.  That is enough to reconstruct resident shared-prefix copies in
benchmark chunk units without estimating KV bytes.  F2c then validates the
same placement effect with exact engine-reported prompt/cached token counts on
Qwen3-32B.  This script binds both retained artifacts into one reviewable F4c
decision; it never reruns either measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from bench import kv_aware_ttft_f2c_bench as f2c
except ModuleNotFoundError:  # direct ``python bench/<script>.py`` execution
    import kv_aware_ttft_f2c_bench as f2c  # type: ignore[no-redef]

SCHEMA_VERSION = "kairyu.f4c.global-kv-pool-decision.v1"
GATE = "G5-F4c"

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_F2A_ARTIFACT = (
    _REPO_ROOT / "bench/results/f2a-prefix-routing-500-2026-07-28"
)
DEFAULT_F2C_ARTIFACT = (
    _REPO_ROOT
    / "bench/results/f2c-kv-aware-ttft-qwen3-32b-2026-07-29"
)

_F2A_SCHEMA = "kairyu.f2a.prefix-routing.v3"
_F2A_MANIFEST = "prefix-routing-f2a-manifest.json"
_F2A_RAW = "prefix-routing-f2a-raw.jsonl"
_F2C_MANIFEST = f2c.MANIFEST_NAME
_F2C_RAW = f2c.RAW_NAME

EXPECTED_F2A_MANIFEST_SHA256 = (
    "fb2858ed213be3f6b98463d30623e937d74403a7a7dd97356311439a7f7585f4"
)
EXPECTED_F2A_RAW_SHA256 = (
    "82bb2a2ff420dbd4e244685ce2a83f38379028604be3c1077e85daf5b31cd0f3"
)
EXPECTED_F2A_SOURCE_COMMIT = "c067cb8a6d1aa661c82a56a007fc325011426459"
EXPECTED_F2A_RAW_ROWS = 24_709
EXPECTED_F2C_MANIFEST_SHA256 = (
    "fcb29c3c5675e58a5bafbcd3c6e1ba737006ec145c731c27cef82e2d2011ad03"
)
EXPECTED_F2C_RAW_SHA256 = (
    "4cfcdeba2b7473aa6c2b28409dbf21de23d775d9b08e971beed6bdab875abe64"
)
EXPECTED_F2C_TRACE_SHA256 = (
    "51d188671432bf791c02d66d91e6a7d785eb2bd01f64e29a41a62e74f9957dad"
)
EXPECTED_F2C_SOURCE_COMMIT = "80b039b5d429c656871a480c2740740951b29b97"
EXPECTED_F2C_RAW_ROWS = 770

REVISIT_REMOTE_REUSABLE_PROMPT_FRACTION = 0.05
REVISIT_REMOTE_RECOMPUTE_TIME_FRACTION = 0.10
REVISIT_CONSECUTIVE_WINDOWS = 3
REVISIT_REQUESTS_PER_WINDOW = 10_000


class EvidenceError(ValueError):
    """The retained evidence is incomplete, inconsistent, or tampered."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvidenceError(f"{path} must contain one JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n") or not line.strip():
                raise EvidenceError(
                    f"{path}: invalid JSONL framing at line {line_number}"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise EvidenceError(
                    f"{path}: line {line_number} is not a JSON object"
                )
            rows.append(value)
    return rows


def _artifact_paths(
    path: Path,
    *,
    manifest_name: str,
    raw_name: str,
) -> tuple[Path, Path]:
    path = path.resolve()
    if path.is_dir():
        manifest_path = path / manifest_name
        raw_path = path / raw_name
    elif path.name == manifest_name:
        manifest_path = path
        raw_path = path.with_name(raw_name)
    elif path.name == raw_name:
        manifest_path = path.with_name(manifest_name)
        raw_path = path
    else:
        raise EvidenceError(
            f"artifact must be a directory, {manifest_name}, or {raw_name}: {path}"
        )
    _require(manifest_path.is_file(), f"missing manifest: {manifest_path}")
    _require(raw_path.is_file(), f"missing raw evidence: {raw_path}")
    return manifest_path, raw_path


def _repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise EvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{label} must be a finite number")
    return result


def _ratio(numerator: int | float, denominator: int | float, label: str) -> float:
    if denominator <= 0:
        raise EvidenceError(f"{label} denominator must be positive")
    result = numerator / denominator
    if not math.isfinite(result):
        raise EvidenceError(f"{label} is not finite")
    return result


def _one_row(rows: list[dict[str, Any]], row_type: str, label: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get("type") == row_type]
    _require(len(matches) == 1, f"{label} needs exactly one {row_type!r} row")
    return matches[0]


def _source_identity(run: dict[str, Any]) -> dict[str, Any]:
    source = run.get("source")
    source_end = run.get("source_end")
    _require(isinstance(source, dict), "F2a run source descriptor is missing")
    _require(isinstance(source_end, dict), "F2a run source-end descriptor is missing")
    commit = source.get("commit")
    _require(_valid_commit(commit), "F2a source commit is not an exact SHA")
    _require(
        source.get("expected_commit") == commit
        and source.get("expected_commit_matches") is True,
        "F2a source did not match its predeclared commit",
    )
    _require(
        source.get("git_clean_at_start") is True,
        "F2a source was not clean at measurement start",
    )
    _require(
        source_end.get("commit") == commit
        and source_end.get("expected_commit") == commit
        and source_end.get("expected_commit_matches") is True,
        "F2a source commit changed during measurement",
    )
    _require(
        source.get("files") == source_end.get("files"),
        "F2a source file hashes changed during measurement",
    )
    environment = run.get("environment")
    github = environment.get("github") if isinstance(environment, dict) else None
    return {
        "source_commit": commit,
        "github_run_id": (
            github.get("GITHUB_RUN_ID") if isinstance(github, dict) else None
        ),
    }


def derive_f2a_shared(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct F2a shared-prefix residency from raw seed/placement rows."""

    replicas = _exact_int(config.get("replicas"), "F2a replicas", minimum=1)
    prefix_chunks = _exact_int(
        config.get("prefix_chunks"), "F2a prefix_chunks", minimum=1
    )
    family_count = _exact_int(
        config.get("shared_families"), "F2a shared_families", minimum=1
    )
    requests_per_family = _exact_int(
        config.get("shared_requests_per_family"),
        "F2a shared_requests_per_family",
        minimum=1,
    )
    expected_requests = family_count * requests_per_family
    replica_ids = {f"replica-{index:03d}" for index in range(replicas)}

    seed_rows = [row for row in rows if row.get("type") == "cache_seed"]
    _require(len(seed_rows) == family_count, "F2a cache-seed count is incomplete")
    seeds: dict[str, str] = {}
    for row in seed_rows:
        family = row.get("family_id")
        replica = row.get("replica_id")
        _require(
            isinstance(family, str) and family not in seeds,
            "F2a cache seeds need unique string family IDs",
        )
        _require(replica in replica_ids, "F2a cache seed names an unknown replica")
        seeds[family] = replica

    shared_rows = [
        row
        for row in rows
        if row.get("type") == "placement" and row.get("trace") == "shared"
    ]
    _require(
        len(shared_rows) == expected_requests * 2,
        "F2a shared placement count is incomplete",
    )
    by_sequence: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in shared_rows:
        sequence = _exact_int(row.get("sequence"), "F2a shared sequence")
        policy = row.get("policy")
        _require(
            policy in {"prefix_aware", "session_hrw"},
            "F2a shared row has an unknown policy",
        )
        _require(
            policy not in by_sequence[sequence],
            "F2a shared policy/sequence pair is duplicated",
        )
        by_sequence[sequence][policy] = row
    _require(
        set(by_sequence) == set(range(expected_requests)),
        "F2a shared sequence is not contiguous",
    )

    identity_fields = (
        "request_id",
        "family_id",
        "prompt_sha256",
        "session_sha256",
        "prefix_fingerprint_sha256",
        "prompt_chunks",
    )
    for sequence, pair in by_sequence.items():
        _require(
            set(pair) == {"prefix_aware", "session_hrw"},
            f"F2a sequence {sequence} is not a complete policy pair",
        )
        candidate = pair["prefix_aware"]
        baseline = pair["session_hrw"]
        _require(
            all(candidate.get(field) == baseline.get(field) for field in identity_fields),
            f"F2a sequence {sequence} policies did not receive identical work",
        )

    policy_results: dict[str, dict[str, Any]] = {}
    for policy, expected_reason in (
        ("prefix_aware", "prefix_match"),
        ("session_hrw", "session_affinity"),
    ):
        residency = {family: {replica} for family, replica in seeds.items()}
        cached_chunks = 0
        hit_count = 0
        duplicate_copy_request_steps = 0
        for sequence in range(expected_requests):
            duplicate_copy_request_steps += sum(
                len(copies) - 1 for copies in residency.values()
            )
            row = by_sequence[sequence][policy]
            family = row.get("family_id")
            selected = row.get("selected_replica_id")
            _require(family in residency, "F2a placement names an unseeded family")
            _require(selected in replica_ids, "F2a placement names an unknown replica")
            _require(
                row.get("backend_replica_id") == selected,
                "F2a selected and executing replicas differ",
            )
            _require(
                row.get("reason") == expected_reason,
                f"F2a {policy} row used an unexpected routing reason",
            )
            _require(
                row.get("prompt_chunks") == prefix_chunks + 1,
                "F2a shared prompt geometry changed",
            )
            expected_hit = selected in residency[family]
            _require(
                row.get("cache_hit") is expected_hit,
                "F2a recorded cache hit disagrees with replayed residency",
            )
            expected_cached = prefix_chunks if expected_hit else 0
            _require(
                row.get("cached_prefix_chunks_before") == expected_cached,
                "F2a cached-prefix chunks disagree with replayed residency",
            )
            residency[family].add(selected)
            cached_chunks += expected_cached
            hit_count += int(expected_hit)

        total_copies = sum(len(copies) for copies in residency.values())
        duplicate_copies = total_copies - family_count
        policy_results[policy] = {
            "request_count": expected_requests,
            "prompt_chunks": expected_requests * (prefix_chunks + 1),
            "reusable_prefix_chunks": expected_requests * prefix_chunks,
            "cached_prefix_chunks": cached_chunks,
            "cache_hit_requests": hit_count,
            "final_resident_prefix_copies": total_copies,
            "duplicate_resident_prefix_copies": duplicate_copies,
            "duplicate_resident_prefix_chunk_copies": duplicate_copies
            * prefix_chunks,
            "duplicate_prefix_copy_request_steps": duplicate_copy_request_steps,
            "duplicate_prefix_chunk_copy_request_steps": (
                duplicate_copy_request_steps * prefix_chunks
            ),
            "mean_duplicate_prefix_copies_before_request": _ratio(
                duplicate_copy_request_steps,
                expected_requests,
                f"F2a {policy} duplicate-copy request-step mean",
            ),
        }

    candidate = policy_results["prefix_aware"]
    baseline = policy_results["session_hrw"]
    _require(
        candidate["prompt_chunks"] == baseline["prompt_chunks"],
        "F2a policy prompt work differs",
    )
    avoided = candidate["cached_prefix_chunks"] - baseline["cached_prefix_chunks"]
    _require(avoided >= 0, "F2a prefix routing cached less work than session HRW")
    _require(
        avoided
        == baseline["duplicate_resident_prefix_chunk_copies"]
        - candidate["duplicate_resident_prefix_chunk_copies"],
        "F2a avoided work and reconstructed duplicate residency disagree",
    )
    return {
        "unit": "256-character benchmark chunks; not KV bytes or model tokens",
        "family_count": family_count,
        "prefix_chunks_per_family": prefix_chunks,
        "candidate": candidate,
        "baseline": baseline,
        "matched_avoidable_cross_replica_recomputation_chunks": avoided,
        "avoidable_fraction_of_all_prompt_chunks": _ratio(
            avoided,
            candidate["prompt_chunks"],
            "F2a avoidable prompt fraction",
        ),
        "avoidable_fraction_of_reusable_prefix_opportunity": _ratio(
            avoided,
            candidate["reusable_prefix_chunks"],
            "F2a avoidable reusable-prefix fraction",
        ),
        "candidate_reusable_prefix_coverage": _ratio(
            candidate["cached_prefix_chunks"],
            candidate["reusable_prefix_chunks"],
            "F2a candidate reusable-prefix coverage",
        ),
    }


def analyze_f2a(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path, raw_path = _artifact_paths(
        path,
        manifest_name=_F2A_MANIFEST,
        raw_name=_F2A_RAW,
    )
    manifest = _load_json(manifest_path)
    rows = _load_jsonl(raw_path)
    raw = manifest.get("raw")
    manifest_sha256 = _sha256_file(manifest_path)
    raw_sha256 = _sha256_file(raw_path)
    _require(
        manifest_sha256 == EXPECTED_F2A_MANIFEST_SHA256,
        "F2a manifest does not match the frozen retained artifact",
    )
    _require(
        raw_sha256 == EXPECTED_F2A_RAW_SHA256,
        "F2a raw evidence does not match the frozen retained artifact",
    )
    _require(
        len(rows) == EXPECTED_F2A_RAW_ROWS,
        "F2a raw row count does not match the frozen retained artifact",
    )
    _require(manifest.get("schema_version") == _F2A_SCHEMA, "unsupported F2a schema")
    _require(manifest.get("passed") is True, "retained F2a artifact did not pass")
    _require(isinstance(raw, dict), "F2a raw descriptor is missing")
    _require(raw.get("path") == _F2A_RAW, "F2a raw filename changed")
    _require(raw.get("sha256") == raw_sha256, "F2a raw SHA-256 mismatch")
    _require(raw.get("rows") == len(rows), "F2a raw row count mismatch")

    run = _one_row(rows, "run", "F2a")
    _require(run.get("schema_version") == _F2A_SCHEMA, "F2a run schema changed")
    _require(run.get("config") == manifest.get("config"), "F2a config mismatch")
    _require(run.get("source") == manifest.get("source"), "F2a source mismatch")
    _require(
        run.get("source_end") == manifest.get("source_end"),
        "F2a source-end mismatch",
    )
    _require(
        run.get("environment") == manifest.get("environment"),
        "F2a environment mismatch",
    )
    identity = _source_identity(run)
    _require(
        identity["source_commit"] == EXPECTED_F2A_SOURCE_COMMIT,
        "F2a source commit does not match the frozen retained artifact",
    )
    config = manifest.get("config")
    _require(isinstance(config, dict), "F2a config is missing")
    metrics = derive_f2a_shared(rows, config)

    recorded_shared = manifest.get("metrics")
    recorded_shared = (
        recorded_shared.get("shared") if isinstance(recorded_shared, dict) else None
    )
    _require(isinstance(recorded_shared, dict), "F2a shared metrics are missing")
    expected_recorded = {
        "request_count_per_policy": metrics["candidate"]["request_count"],
        "prefix_prompt_chunks": metrics["candidate"]["prompt_chunks"],
        "hrw_prompt_chunks": metrics["baseline"]["prompt_chunks"],
        "prefix_cached_prefix_chunks": metrics["candidate"]["cached_prefix_chunks"],
        "hrw_cached_prefix_chunks": metrics["baseline"]["cached_prefix_chunks"],
        "prefix_request_hits": metrics["candidate"]["cache_hit_requests"],
        "hrw_request_hits": metrics["baseline"]["cache_hit_requests"],
    }
    _require(
        all(recorded_shared.get(key) == value for key, value in expected_recorded.items()),
        "F2a manifest shared metrics differ from raw residency replay",
    )
    descriptor = {
        "artifact": _repo_path(manifest_path.parent),
        "schema_version": manifest["schema_version"],
        "manifest_sha256": manifest_sha256,
        "raw_sha256": raw_sha256,
        "raw_row_count": len(rows),
        **identity,
    }
    return descriptor, metrics


def derive_f2c_binding(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pair the real-engine F2c arms and derive exact token reuse mass."""

    binding = [
        row
        for row in rows
        if row.get("type") == "request" and row.get("binding") is True
    ]
    by_key: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in binding:
        key = (
            _exact_int(row.get("round_index"), "F2c round index"),
            _exact_int(row.get("family_index"), "F2c family index"),
            _exact_int(row.get("turn"), "F2c turn", minimum=1),
        )
        policy = row.get("policy")
        _require(
            policy in {f2c.CANDIDATE_POLICY, f2c.BASELINE_POLICY},
            "F2c binding row has an unknown policy",
        )
        _require(
            policy not in by_key[key],
            "F2c binding policy/key pair is duplicated",
        )
        by_key[key][policy] = row
    _require(by_key, "F2c has no binding request pairs")

    totals = {
        f2c.CANDIDATE_POLICY: {"prompt": 0, "cached": 0, "requests": 0},
        f2c.BASELINE_POLICY: {"prompt": 0, "cached": 0, "requests": 0},
    }
    positive_pairs = 0
    equal_pairs = 0
    negative_pairs = 0
    for key, pair in sorted(by_key.items()):
        _require(
            set(pair) == {f2c.CANDIDATE_POLICY, f2c.BASELINE_POLICY},
            f"F2c key {key} is not a complete policy pair",
        )
        candidate = pair[f2c.CANDIDATE_POLICY]
        baseline = pair[f2c.BASELINE_POLICY]
        for row, expected_reason in (
            (candidate, "prefix_match"),
            (baseline, "session_affinity"),
        ):
            _require(
                row.get("status") == "success"
                and row.get("terminal") is True
                and row.get("error_type") is None,
                f"F2c key {key} contains an unsuccessful request",
            )
            _require(
                row.get("reason") == expected_reason,
                f"F2c key {key} used an unexpected routing reason",
            )
        _require(
            candidate.get("prompt_sha256") == baseline.get("prompt_sha256")
            and candidate.get("prompt_chars") == baseline.get("prompt_chars"),
            f"F2c key {key} policies did not receive identical prompts",
        )
        candidate_usage = candidate.get("usage")
        baseline_usage = baseline.get("usage")
        _require(
            isinstance(candidate_usage, dict) and isinstance(baseline_usage, dict),
            f"F2c key {key} is missing engine usage",
        )
        candidate_prompt = _exact_int(
            candidate_usage.get("prompt_tokens"), "F2c candidate prompt tokens", minimum=1
        )
        baseline_prompt = _exact_int(
            baseline_usage.get("prompt_tokens"), "F2c baseline prompt tokens", minimum=1
        )
        candidate_cached = _exact_int(
            candidate_usage.get("cached_tokens"), "F2c candidate cached tokens"
        )
        baseline_cached = _exact_int(
            baseline_usage.get("cached_tokens"), "F2c baseline cached tokens"
        )
        _require(
            candidate_prompt == baseline_prompt,
            f"F2c key {key} policy prompt-token work differs",
        )
        _require(
            candidate_cached <= candidate_prompt
            and baseline_cached <= baseline_prompt,
            f"F2c key {key} cached tokens exceed prompt tokens",
        )
        _require(
            candidate_usage.get("completion_tokens")
            == baseline_usage.get("completion_tokens"),
            f"F2c key {key} completion work differs",
        )
        for policy, prompt_tokens, cached_tokens in (
            (f2c.CANDIDATE_POLICY, candidate_prompt, candidate_cached),
            (f2c.BASELINE_POLICY, baseline_prompt, baseline_cached),
        ):
            totals[policy]["prompt"] += prompt_tokens
            totals[policy]["cached"] += cached_tokens
            totals[policy]["requests"] += 1
        delta = candidate_cached - baseline_cached
        positive_pairs += int(delta > 0)
        equal_pairs += int(delta == 0)
        negative_pairs += int(delta < 0)

    candidate = totals[f2c.CANDIDATE_POLICY]
    baseline = totals[f2c.BASELINE_POLICY]
    _require(
        candidate["prompt"] == baseline["prompt"],
        "F2c policy prompt totals differ",
    )
    _require(negative_pairs == 0, "F2c prefix routing regressed a paired cache result")
    avoided = candidate["cached"] - baseline["cached"]
    baseline_miss = baseline["prompt"] - baseline["cached"]
    candidate_miss = candidate["prompt"] - candidate["cached"]
    _require(avoided >= 0, "F2c prefix routing cached less work than session HRW")
    return {
        "unit": "engine-reported model tokens",
        "pair_count": len(by_key),
        "positive_cache_delta_pairs": positive_pairs,
        "equal_cache_delta_pairs": equal_pairs,
        "negative_cache_delta_pairs": negative_pairs,
        "candidate": {
            "request_count": candidate["requests"],
            "prompt_tokens": candidate["prompt"],
            "cached_tokens": candidate["cached"],
            "uncached_tokens": candidate_miss,
            "token_weighted_cache_rate": _ratio(
                candidate["cached"],
                candidate["prompt"],
                "F2c candidate cache rate",
            ),
        },
        "baseline": {
            "request_count": baseline["requests"],
            "prompt_tokens": baseline["prompt"],
            "cached_tokens": baseline["cached"],
            "uncached_tokens": baseline_miss,
            "token_weighted_cache_rate": _ratio(
                baseline["cached"],
                baseline["prompt"],
                "F2c baseline cache rate",
            ),
        },
        "matched_avoidable_cross_replica_recomputation_tokens": avoided,
        "avoidable_fraction_of_all_prompt_tokens": _ratio(
            avoided,
            candidate["prompt"],
            "F2c avoidable prompt fraction",
        ),
        "avoidable_fraction_of_baseline_uncached_tokens": _ratio(
            avoided,
            baseline_miss,
            "F2c avoidable baseline-miss fraction",
        ),
        "post_routing_uncached_prompt_fraction_upper_bound": _ratio(
            candidate_miss,
            candidate["prompt"],
            "F2c post-routing residual upper bound",
        ),
    }


def _nested_dict(value: object, *keys: str) -> dict[str, Any] | None:
    current: object = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, dict) else None


def analyze_f2c(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path, raw_path = _artifact_paths(
        path,
        manifest_name=_F2C_MANIFEST,
        raw_name=_F2C_RAW,
    )
    manifest_sha256 = _sha256_file(manifest_path)
    raw_sha256 = _sha256_file(raw_path)
    _require(
        manifest_sha256 == EXPECTED_F2C_MANIFEST_SHA256,
        "F2c manifest does not match the frozen retained artifact",
    )
    _require(
        raw_sha256 == EXPECTED_F2C_RAW_SHA256,
        "F2c raw evidence does not match the frozen retained artifact",
    )
    verified = f2c.verify_artifact(path, assert_gate=True)
    _require(verified.get("passed") is True, "retained F2c artifact did not pass")
    manifest = _load_json(manifest_path)
    rows = _load_jsonl(raw_path)
    _require(
        len(rows) == EXPECTED_F2C_RAW_ROWS,
        "F2c raw row count does not match the frozen retained artifact",
    )
    metrics = derive_f2c_binding(rows)

    cache = _nested_dict(verified, "metrics", "cache")
    _require(cache is not None, "F2c verified cache metrics are missing")
    expected_cache = {
        "candidate_prompt_tokens": metrics["candidate"]["prompt_tokens"],
        "candidate_cached_tokens": metrics["candidate"]["cached_tokens"],
        "baseline_prompt_tokens": metrics["baseline"]["prompt_tokens"],
        "baseline_cached_tokens": metrics["baseline"]["cached_tokens"],
    }
    _require(
        all(cache.get(key) == value for key, value in expected_cache.items()),
        "F2c verified metrics differ from binding pair replay",
    )
    _require(
        verified.get("raw", {}).get("sha256") == raw_sha256,
        "F2c verified raw SHA-256 mismatch",
    )
    _require(
        verified.get("trace_sha256") == EXPECTED_F2C_TRACE_SHA256,
        "F2c trace does not match the frozen retained artifact",
    )
    run = _one_row(rows, "run", "F2c")
    source = _nested_dict(run, "provenance_start", "source", "value")
    environment = _nested_dict(run, "provenance_start", "environment", "value")
    _require(source is not None, "F2c source provenance is missing")
    commit = source.get("commit")
    _require(_valid_commit(commit), "F2c source commit is not an exact SHA")
    _require(
        commit == EXPECTED_F2C_SOURCE_COMMIT,
        "F2c source commit does not match the frozen retained artifact",
    )
    github = environment.get("github") if isinstance(environment, dict) else None
    ttft = _nested_dict(verified, "metrics", "ttft")
    goodput = _nested_dict(verified, "metrics", "goodput")
    _require(ttft is not None and goodput is not None, "F2c performance metrics missing")
    metrics["candidate_over_baseline_ttft_p95_ratio"] = _finite_number(
        ttft.get("pooled_ratio"), "F2c pooled TTFT ratio"
    )
    metrics["candidate_over_baseline_goodput_ratio"] = _finite_number(
        goodput.get("pooled_ratio"), "F2c pooled goodput ratio"
    )
    descriptor = {
        "artifact": _repo_path(manifest_path.parent),
        "schema_version": manifest.get("schema_version"),
        "manifest_sha256": manifest_sha256,
        "raw_sha256": raw_sha256,
        "raw_row_count": len(rows),
        "trace_sha256": verified.get("trace_sha256"),
        "source_commit": commit,
        "github_run_id": (
            github.get("GITHUB_RUN_ID") if isinstance(github, dict) else None
        ),
        "model": run.get("model"),
        "model_revision": run.get("model_revision"),
        "model_digests": run.get("model_digests"),
    }
    return descriptor, metrics


def build_decision_artifact(
    f2a_artifact: Path = DEFAULT_F2A_ARTIFACT,
    f2c_artifact: Path = DEFAULT_F2C_ARTIFACT,
) -> dict[str, Any]:
    f2a_input, f2a_metrics = analyze_f2a(f2a_artifact)
    f2c_input, f2c_metrics = analyze_f2c(f2c_artifact)
    current_upper_bound = f2c_metrics[
        "post_routing_uncached_prompt_fraction_upper_bound"
    ]

    decision = {
        "choice": "defer_global_pool",
        "current_architecture": "per_replica_radix_kv_plus_f2_prefix_routing",
        "current_next_steps": [
            "complete_native_dram_tiering_f4a_issue_187",
            "measure_native_tiering_value_f4b_issue_188",
        ],
        "if_revisit_triggers_fire": (
            "prototype_mooncake_store_behind_a_separate_global_kv_object_store_adapter"
        ),
        "ownership_boundaries": {
            "kairyu": (
                "token_and_model_namespace, radix_index, request_and_replica_routing, "
                "logical_object_naming, cache_truth, admission, shard_commit, fallback"
            ),
            "mooncake_store": (
                "physical_object_allocation, immutable_object_storage, and transfer only"
            ),
        },
        "not_selected": {
            "lmcache_full_adoption": (
                "duplicates Kairyu radix/index/tiering/connector policy and has "
                "no Kairyu engine connector"
            ),
            "native_global_kvtransport_store": (
                "would make Kairyu own global metadata, replication, eviction, "
                "recovery, and storage operations"
            ),
        },
    }
    revisit_policy = {
        "prerequisites": [
            "F4a_DRAM_crossover_is_published",
            "F4b_agentic_tiering_trace_is_published",
        ],
        "window_contract": (
            "three consecutive non-overlapping windows of exactly 10000 eligible "
            "requests from one predeclared deployment/model/TP/hardware cohort; "
            "all consecutive eligible requests are included and missing telemetry "
            "invalidates rather than excludes a window"
        ),
        "consecutive_windows": REVISIT_CONSECUTIVE_WINDOWS,
        "requests_per_window": REVISIT_REQUESTS_PER_WINDOW,
        "eligible_request": (
            "successfully admitted text-generation request with nonempty prompt, "
            "cache enabled; eligibility is independent of telemetry availability"
        ),
        "trigger_logic": (
            "after prerequisites, revisit only when the same branch clears its "
            "threshold in all three windows: min(remote_token_fraction) >= 0.05 "
            "OR min(remote_recompute_time_fraction) >= 0.10"
        ),
        "metrics": {
            "remote_token_fraction": (
                "sum of model tokens in exact full blocks that miss locally at routing "
                "time and have a committed namespace-compatible copy on another "
                "replica, divided by all model prompt tokens in eligible requests"
            ),
            "remote_recompute_time_fraction": (
                "sum of counterfactual recompute GPU time for those same exact blocks, "
                "using the predeclared same-cohort F4a token-count curve, divided by "
                "measured total prefill GPU-active time for all eligible requests"
            ),
        },
        "thresholds": {
            "post_routing_remote_resident_reusable_prompt_fraction": (
                REVISIT_REMOTE_REUSABLE_PROMPT_FRACTION
            ),
            "post_routing_remote_recompute_time_fraction": (
                REVISIT_REMOTE_RECOMPUTE_TIME_FRACTION
            ),
        },
        "current_f2c_uncached_prompt_fraction_upper_bound": current_upper_bound,
        "current_evidence_crosses_prompt_fraction_trigger": (
            current_upper_bound >= REVISIT_REMOTE_REUSABLE_PROMPT_FRACTION
        ),
        "prototype_acceptance": [
            "same_checkpoint_tokenizer_rope_lora_kv_layout_page_size_tp_namespace",
            "all_shards_and_commit_manifest_verified_before_injection",
            "timeout_missing_object_digest_or_fingerprint_mismatch_recomputes",
            "tenant_namespace_and_authorization_isolation",
            "lookup_plus_transfer_beats_measured_recompute_above_f4a_crossover",
            "greedy_output_parity_and_no_tpot_p99_regression",
            "master_or_client_loss_degrades_to_cache_miss_without_failed_requests",
        ],
    }
    checks = {
        "f2a_reconstructs_positive_session_hrw_duplicate_mass": (
            f2a_metrics["baseline"]["duplicate_resident_prefix_chunk_copies"] > 0
        ),
        "f2a_prefix_routing_creates_no_duplicate_copy": (
            f2a_metrics["candidate"]["duplicate_resident_prefix_chunk_copies"] == 0
        ),
        "f2a_prefix_routing_covers_all_seeded_reusable_prefixes": (
            f2a_metrics["candidate_reusable_prefix_coverage"] == 1.0
        ),
        "f2c_policy_pairs_have_identical_prompt_work": (
            f2c_metrics["candidate"]["prompt_tokens"]
            == f2c_metrics["baseline"]["prompt_tokens"]
        ),
        "f2c_has_no_pairwise_cache_regression": (
            f2c_metrics["negative_cache_delta_pairs"] == 0
        ),
        "f2c_real_engine_cache_mass_strictly_improves": (
            f2c_metrics["matched_avoidable_cross_replica_recomputation_tokens"] > 0
        ),
        "f2c_goodput_is_noninferior": (
            f2c_metrics["candidate_over_baseline_goodput_ratio"] >= 0.99
        ),
        "current_gross_upper_bound_is_below_revisit_trigger": (
            current_upper_bound < REVISIT_REMOTE_REUSABLE_PROMPT_FRACTION
        ),
        "decision_and_future_provider_are_explicit": (
            decision["choice"] == "defer_global_pool"
            and "mooncake_store" in decision["if_revisit_triggers_fire"]
        ),
        "revisit_policy_is_predeclared_and_bounded": (
            revisit_policy["consecutive_windows"] >= 3
            and revisit_policy["requests_per_window"] == 10_000
            and len(revisit_policy["prerequisites"]) == 2
            and len(revisit_policy["prototype_acceptance"]) >= 7
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "definitions": {
            "duplicate_prefix_mass": (
                "additional resident copies of an already-seeded shared prefix, "
                "reported as logical F2a family/chunk copies and request-steps; "
                "physical KV bytes and byte-seconds were not recorded"
            ),
            "matched_avoidable_cross_replica_recomputation": (
                "candidate cached work minus session-HRW cached work for identical "
                "paired prompts"
            ),
            "post_routing_uncached_prompt_fraction_upper_bound": (
                "all candidate uncached prompt tokens; a conservative ceiling because "
                "novel suffixes are not remotely reusable"
            ),
            "incremental_global_pool_mass": (
                "post-routing local misses with a compatible remote-resident copy whose "
                "measured restore cost beats recompute; not directly measured by F2a/F2c"
            ),
        },
        "inputs": {"f2a": f2a_input, "f2c": f2c_input},
        "evidence": {
            "f2a_500_replica_shared_trace": f2a_metrics,
            "f2c_qwen3_32b_real_engine_trace": f2c_metrics,
        },
        "decision": decision,
        "revisit_policy": revisit_policy,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _render_json(value: dict[str, Any]) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def write_decision_artifact(
    output: Path,
    *,
    f2a_artifact: Path = DEFAULT_F2A_ARTIFACT,
    f2c_artifact: Path = DEFAULT_F2C_ARTIFACT,
) -> dict[str, Any]:
    result = build_decision_artifact(f2a_artifact, f2c_artifact)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(_render_json(result), encoding="utf-8")
    temporary.replace(output)
    return result


def verify_decision_artifact(
    artifact: Path,
    *,
    f2a_artifact: Path = DEFAULT_F2A_ARTIFACT,
    f2c_artifact: Path = DEFAULT_F2C_ARTIFACT,
    assert_gate: bool = False,
) -> dict[str, Any]:
    recorded = _load_json(artifact.resolve())
    recomputed = build_decision_artifact(f2a_artifact, f2c_artifact)
    _require(recorded == recomputed, "F4c artifact differs from independent replay")
    if assert_gate:
        _require(recomputed.get("passed") is True, "F4c decision checks did not pass")
    return recomputed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify-artifact", type=Path)
    parser.add_argument("--f2a-artifact", type=Path, default=DEFAULT_F2A_ARTIFACT)
    parser.add_argument("--f2c-artifact", type=Path, default=DEFAULT_F2C_ARTIFACT)
    parser.add_argument("--assert-gate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output is not None:
            result = write_decision_artifact(
                args.output,
                f2a_artifact=args.f2a_artifact,
                f2c_artifact=args.f2c_artifact,
            )
        else:
            result = verify_decision_artifact(
                args.verify_artifact,
                f2a_artifact=args.f2a_artifact,
                f2c_artifact=args.f2c_artifact,
                assert_gate=args.assert_gate,
            )
    except (ValueError, RuntimeError, OSError, TypeError) as error:
        print(
            _render_json(
                {
                    "schema_version": SCHEMA_VERSION,
                    "gate": GATE,
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            ),
            end="",
        )
        return 1
    print(_render_json(result), end="")
    return 0 if result["passed"] is True or not args.assert_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
