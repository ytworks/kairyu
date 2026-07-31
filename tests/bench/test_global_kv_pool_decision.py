"""Focused tests for the replayable G5 F4c global-KV-pool decision."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from bench import global_kv_pool_decision as f4c

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DECISION_ARTIFACT = (
    _REPO_ROOT / "bench/results/f4c-global-kv-pool-decision-2026-07-31.json"
)
_F2A_ARTIFACT = _REPO_ROOT / "bench/results/f2a-prefix-routing-500-2026-07-28"
_F2A_MANIFEST = _F2A_ARTIFACT / "prefix-routing-f2a-manifest.json"
_F2A_RAW = _F2A_ARTIFACT / "prefix-routing-f2a-raw.jsonl"


def _f2a_fixture() -> tuple[list[dict[str, object]], dict[str, object]]:
    config = {
        "replicas": 4,
        "prefix_chunks": 2,
        "shared_families": 2,
        "shared_requests_per_family": 2,
    }
    rows: list[dict[str, object]] = [
        {"type": "cache_seed", "family_id": "a", "replica_id": "replica-000"},
        {"type": "cache_seed", "family_id": "b", "replica_id": "replica-001"},
    ]
    requests = [
        ("a", "replica-000", "replica-002", False),
        ("b", "replica-001", "replica-003", False),
        ("a", "replica-000", "replica-002", True),
        ("b", "replica-001", "replica-001", True),
    ]
    for sequence, (family, candidate_replica, baseline_replica, baseline_hit) in enumerate(
        requests
    ):
        identity = {
            "type": "placement",
            "trace": "shared",
            "sequence": sequence,
            "request_id": f"request-{sequence}",
            "family_id": family,
            "prompt_sha256": f"prompt-{sequence}",
            "session_sha256": f"session-{sequence}",
            "prefix_fingerprint_sha256": f"prefix-{family}",
            "prompt_chunks": 3,
        }
        rows.append(
            {
                **identity,
                "policy": "prefix_aware",
                "selected_replica_id": candidate_replica,
                "backend_replica_id": candidate_replica,
                "reason": "prefix_match",
                "cache_hit": True,
                "cached_prefix_chunks_before": 2,
            }
        )
        rows.append(
            {
                **identity,
                "policy": "session_hrw",
                "selected_replica_id": baseline_replica,
                "backend_replica_id": baseline_replica,
                "reason": "session_affinity",
                "cache_hit": baseline_hit,
                "cached_prefix_chunks_before": 2 if baseline_hit else 0,
            }
        )
    return rows, config


def _f2c_row(
    *,
    key: tuple[int, int, int],
    policy: str,
    prompt_tokens: int,
    cached_tokens: int,
) -> dict[str, object]:
    round_index, family_index, turn = key
    return {
        "type": "request",
        "binding": True,
        "round_index": round_index,
        "family_index": family_index,
        "turn": turn,
        "policy": policy,
        "status": "success",
        "terminal": True,
        "error_type": None,
        "reason": "prefix_match" if policy == "kv_aware" else "session_affinity",
        "prompt_sha256": f"prompt-{round_index}-{family_index}-{turn}",
        "prompt_chars": 100,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "completion_tokens": 8,
        },
    }


def _f2c_fixture() -> list[dict[str, object]]:
    return [
        _f2c_row(
            key=(0, 0, 1),
            policy="kv_aware",
            prompt_tokens=10,
            cached_tokens=8,
        ),
        _f2c_row(
            key=(0, 0, 1),
            policy="session_hrw",
            prompt_tokens=10,
            cached_tokens=0,
        ),
        _f2c_row(
            key=(0, 0, 2),
            policy="kv_aware",
            prompt_tokens=10,
            cached_tokens=4,
        ),
        _f2c_row(
            key=(0, 0, 2),
            policy="session_hrw",
            prompt_tokens=10,
            cached_tokens=4,
        ),
    ]


def _tampered_f2a(tmp_path: Path, *, update_manifest_sha: bool) -> Path:
    artifact = tmp_path / "f2a"
    artifact.mkdir()
    raw_text = _F2A_RAW.read_text(encoding="utf-8")
    tampered_raw = raw_text.replace(
        '"placement_slo_success_count":512,"placement_sum_ns":67050690',
        '"placement_slo_success_count":511,"placement_sum_ns":67050690',
        1,
    )
    assert tampered_raw != raw_text
    (artifact / _F2A_RAW.name).write_text(tampered_raw, encoding="utf-8")
    manifest_text = _F2A_MANIFEST.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    if update_manifest_sha:
        manifest["raw"]["sha256"] = hashlib.sha256(
            tampered_raw.encode("utf-8")
        ).hexdigest()
        manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (artifact / _F2A_MANIFEST.name).write_text(manifest_text, encoding="utf-8")
    return artifact


def test_f2a_replay_reconstructs_resident_duplicate_mass() -> None:
    rows, config = _f2a_fixture()
    result = f4c.derive_f2a_shared(rows, config)

    assert result["candidate"]["final_resident_prefix_copies"] == 2
    assert result["candidate"]["duplicate_resident_prefix_chunk_copies"] == 0
    assert result["baseline"]["final_resident_prefix_copies"] == 4
    assert result["baseline"]["duplicate_resident_prefix_copies"] == 2
    assert result["baseline"]["duplicate_resident_prefix_chunk_copies"] == 4
    assert result["baseline"]["duplicate_prefix_copy_request_steps"] == 5
    assert result["matched_avoidable_cross_replica_recomputation_chunks"] == 4
    assert result["avoidable_fraction_of_reusable_prefix_opportunity"] == 0.5


def test_f2a_replay_rejects_cache_truth_that_disagrees_with_residency() -> None:
    rows, config = _f2a_fixture()
    broken = deepcopy(rows)
    placement = next(
        row
        for row in broken
        if row.get("type") == "placement"
        and row.get("policy") == "session_hrw"
        and row.get("sequence") == 0
    )
    placement["cache_hit"] = True

    with pytest.raises(f4c.EvidenceError, match="replayed residency"):
        f4c.derive_f2a_shared(broken, config)


def test_f2c_pair_replay_derives_exact_engine_token_opportunity() -> None:
    result = f4c.derive_f2c_binding(_f2c_fixture())

    assert result["pair_count"] == 2
    assert result["positive_cache_delta_pairs"] == 1
    assert result["equal_cache_delta_pairs"] == 1
    assert result["negative_cache_delta_pairs"] == 0
    assert result["candidate"]["prompt_tokens"] == 20
    assert result["baseline"]["prompt_tokens"] == 20
    assert result["matched_avoidable_cross_replica_recomputation_tokens"] == 8
    assert result["avoidable_fraction_of_all_prompt_tokens"] == 0.4
    assert result["post_routing_uncached_prompt_fraction_upper_bound"] == 0.4


def test_f2c_pair_replay_rejects_mismatched_prompts() -> None:
    rows = _f2c_fixture()
    rows[1]["prompt_sha256"] = "different"

    with pytest.raises(f4c.EvidenceError, match="identical prompts"):
        f4c.derive_f2c_binding(rows)


def test_f2c_pair_replay_rejects_pairwise_cache_regression() -> None:
    rows = _f2c_fixture()
    rows[0]["usage"]["cached_tokens"] = 0  # type: ignore[index]
    rows[1]["usage"]["cached_tokens"] = 8  # type: ignore[index]

    with pytest.raises(f4c.EvidenceError, match="regressed"):
        f4c.derive_f2c_binding(rows)


def test_f2a_analysis_rejects_raw_tampering(tmp_path: Path) -> None:
    artifact = _tampered_f2a(tmp_path, update_manifest_sha=False)

    with pytest.raises(f4c.EvidenceError, match="raw evidence.*frozen"):
        f4c.analyze_f2a(artifact)


def test_f2a_analysis_rejects_co_tampered_raw_and_manifest(tmp_path: Path) -> None:
    artifact = _tampered_f2a(tmp_path, update_manifest_sha=True)

    with pytest.raises(f4c.EvidenceError, match="manifest.*frozen"):
        f4c.analyze_f2a(artifact)


def test_retained_decision_artifact_replays_exactly() -> None:
    result = f4c.verify_decision_artifact(_DECISION_ARTIFACT, assert_gate=True)

    f2a = result["evidence"]["f2a_500_replica_shared_trace"]
    f2c = result["evidence"]["f2c_qwen3_32b_real_engine_trace"]
    assert result["passed"] is True
    assert f2a["baseline"]["duplicate_resident_prefix_copies"] == 997
    assert f2a["baseline"]["duplicate_prefix_copy_request_steps"] == 513_809
    assert f2a["candidate"]["duplicate_resident_prefix_copies"] == 0
    assert f2c["matched_avoidable_cross_replica_recomputation_tokens"] == 319_696
    assert f2c["post_routing_uncached_prompt_fraction_upper_bound"] == pytest.approx(
        0.015608267376142558
    )
    assert result["decision"]["choice"] == "defer_global_pool"
    assert result["revisit_policy"][
        "current_evidence_crosses_prompt_fraction_trigger"
    ] is False
    assert result["revisit_policy"]["requests_per_window"] == 10_000
    assert "independent of telemetry" in result["revisit_policy"]["eligible_request"]
    assert "all model prompt tokens" in result["revisit_policy"]["metrics"][
        "remote_token_fraction"
    ]


def test_decision_verifier_rejects_a_tampered_choice(tmp_path: Path) -> None:
    recorded = json.loads(_DECISION_ARTIFACT.read_text(encoding="utf-8"))
    recorded["decision"]["choice"] = "adopt_without_trigger"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(recorded), encoding="utf-8")

    with pytest.raises(f4c.EvidenceError, match="independent replay"):
        f4c.verify_decision_artifact(tampered, assert_gate=True)


def test_cli_verifies_the_committed_artifact(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        f4c.main(
            [
                "--verify-artifact",
                str(_DECISION_ARTIFACT),
                "--assert-gate",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["passed"] is True
    assert output["gate"] == "G5-F4c"


@pytest.mark.parametrize(
    "upstream_error",
    (ValueError("tampered upstream artifact"), RuntimeError("upstream gate failed")),
)
def test_cli_reports_upstream_verification_errors_as_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    upstream_error: Exception,
) -> None:
    def fail_verification(*args: object, **kwargs: object) -> None:
        raise upstream_error

    monkeypatch.setattr(f4c.f2c, "verify_artifact", fail_verification)

    assert (
        f4c.main(
            [
                "--output",
                str(tmp_path / "decision.json"),
                "--assert-gate",
            ]
        )
        == 1
    )
    output = json.loads(capsys.readouterr().out)
    assert output["passed"] is False
    assert output["gate"] == "G5-F4c"
    assert str(upstream_error) in output["error"]


def test_design_and_goal_bind_the_decision_and_measurement_limits() -> None:
    design = (_REPO_ROOT / "docs/design/m7-productionization.md").read_text(
        encoding="utf-8"
    )
    goal = (_REPO_ROOT / "docs/goals/g5-fleet-scale.md").read_text(
        encoding="utf-8"
    )

    for document in (design, goal):
        assert "f4c-global-kv-pool-decision-2026-07-31.json" in document
        assert "Mooncake Store" in document
        assert "physical KV bytes" in document
    assert "513,809" in design
    assert "319,696" in design
    assert "defer" in design.lower()
