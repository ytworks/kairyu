"""Focused gates for the deterministic F2d full-policy replay artifact."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from verification.fleet.correctness import prefix_weight_f2d_bench as f2d


def test_trace_is_deterministic_family_disjoint_and_reuses_retained_inputs() -> None:
    config = f2d.profile_config("smoke")
    first = f2d.build_episodes(config)

    assert len(first) == config.families
    assert sum(episode.split == "train" for episode in first) == config.train_families
    assert (
        sum(episode.split == "heldout" for episode in first)
        == config.heldout_families
    )
    train = {episode.family_id for episode in first if episode.split == "train"}
    heldout = {episode.family_id for episode in first if episode.split == "heldout"}
    assert train.isdisjoint(heldout)
    assert all(len(episode.items) == config.requests_per_family for episode in first)
    assert all(
        {item.prefix_fingerprint for item in episode.items}
        == {episode.prefix_fingerprint}
        for episode in first
    )
    assert f2d.input_descriptor(config)["all_pinned_digests_match"]
    assert f2d.trace_descriptor(config)["sha256"] == (
        "db55bb59855ab31e50e46ffd5d6e7d07dca49b834da18027e711e93c8a6ab2ae"
    )


@pytest.fixture(scope="module")
def passing_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("f2d-passing")
    f2d.run_benchmark(output, profile="smoke", assert_gate=True)
    return output


def test_smoke_artifact_replays_and_only_mean_improvement_is_binding(
    passing_artifact: Path,
) -> None:
    replay = f2d.verify_artifact(passing_artifact, assert_gate=True)

    assert replay["passed"]
    assert all(replay["checks"].values())
    metrics = replay["metrics"]
    assert metrics["selected"] == {"alpha": 1.0, "beta": 1.0}
    assert metrics["baseline"] == {"alpha": 1.0, "beta": 0.25}
    assert (
        metrics["heldout"]["tuned"]["mean_ttft_ticks"]
        < metrics["heldout"]["baseline"]["mean_ttft_ticks"]
    )
    # Tail and action effects are reported, not silently promoted into
    # acceptance criteria that issue #182 never requested.
    assert metrics["heldout"]["tuned"]["p95_ttft_ticks"] is not None
    assert metrics["heldout"]["baseline"]["p95_ttft_ticks"] is not None
    assert isinstance(metrics["heldout"]["action_disagreement_count"], int)
    manifest = json.loads(
        (passing_artifact / f2d.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert all("p95" not in name for name in manifest["gate_checks"])
    assert all("disagreement" not in name for name in manifest["gate_checks"])


def _copy_artifact(source: Path, destination: Path) -> tuple[Path, dict]:
    destination.mkdir()
    for name in (f2d.RAW_NAME, f2d.ROUTER_NAME, f2d.MANIFEST_NAME):
        shutil.copy2(source / name, destination / name)
    manifest_path = destination / f2d.MANIFEST_NAME
    return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "target",
    [
        "source",
        "run",
        "split",
        "order",
        "route",
        "outcome",
        "tuning",
        "hash",
        "passed",
    ],
)
def test_verifier_rejects_tampering_even_when_sidecar_hashes_are_recomputed(
    passing_artifact: Path,
    tmp_path: Path,
    target: str,
) -> None:
    manifest_path, manifest = _copy_artifact(
        passing_artifact,
        tmp_path / target,
    )
    if target in {"source", "run", "split", "order", "outcome", "tuning"}:
        path = manifest_path.parent / f2d.RAW_NAME
        rows = f2d._read_jsonl(path)
        if target == "source":
            forged_commit = "0" * 40
            manifest["source"]["commit"] = forged_commit
            manifest["source_end"]["commit"] = forged_commit
            run = rows[0]
            run["source"] = manifest["source"]
            run["source_end"] = manifest["source_end"]
        elif target == "run":
            rows[0]["gate"] = "forged-gate"
        elif target == "split":
            outcome = next(
                row
                for row in rows
                if row.get("kind") == "placement_outcome"
                and row.get("split") == "heldout"
            )
            outcome["split"] = "train"
        elif target == "order":
            tuning_index = next(
                index for index, row in enumerate(rows) if row.get("type") == "tuning"
            )
            heldout_index = next(
                index
                for index, row in enumerate(rows)
                if row.get("kind") == "placement_outcome"
                and row.get("split") == "heldout"
            )
            rows[tuning_index], rows[heldout_index] = (
                rows[heldout_index],
                rows[tuning_index],
            )
            for index, row in enumerate(rows):
                row["row_index"] = index
        elif target == "outcome":
            outcome = next(
                row for row in rows if row.get("kind") == "placement_outcome"
            )
            outcome["ttft_ticks"] = int(outcome["ttft_ticks"]) + 1
            outcome["ttft_s"] = float(outcome["ttft_ticks"])
        else:
            tuning = next(row for row in rows if row.get("type") == "tuning")
            tuning["selected_beta"] = 2.0
        f2d._write_jsonl(path, rows)
        manifest["raw"]["sha256"] = f2d.sha256_file(path)
    elif target == "route":
        path = manifest_path.parent / f2d.ROUTER_NAME
        rows = f2d._read_jsonl(path)
        route = next(row for row in rows if row.get("reason") == "prefix_match")
        route["reason"] = "session_affinity"
        f2d._write_jsonl(path, rows)
        manifest["router"]["sha256"] = f2d.sha256_file(path)
    elif target == "hash":
        manifest["raw"]["sha256"] = "0" * 64
    else:
        manifest["passed"] = False
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    replay = f2d.verify_artifact(manifest_path)
    assert not replay["passed"]


def test_manifest_passed_is_online_verdict_offline_requires_integrity(
    passing_artifact: Path,
) -> None:
    manifest = json.loads(
        (passing_artifact / f2d.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    replay = f2d.verify_artifact(passing_artifact)

    assert manifest["passed"] is True
    assert manifest["pass_contract"] == f2d._PASS_CONTRACT
    assert replay["checks"]["manifest_passed_recomputes_exactly"]
    assert replay["passed"] is all(replay["checks"].values())
