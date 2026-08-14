"""Machine-readable inventory for retained checkout-only benchmark evidence."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from evidence.legacy_index import (
    INDEX_RELATIVE_PATH,
    ResultArtifact,
    ResultsIndex,
    ResultsIndexError,
    parse_results_index,
    render_results_index,
    validate_results_repository,
)

ROOT = Path(__file__).resolve().parents[2]


def _artifact(
    name: str = "alpha.json",
    *,
    gate_id: str | None = "g1-a1",
    summary_path: str | None = None,
    verdict: str = "pass",
) -> ResultArtifact:
    return ResultArtifact(
        artifact_path=f"bench/results/{name}",
        commit="a" * 40,
        date="2026-08-04",
        gate_id=gate_id,
        hardware="cpu",
        summary_path=summary_path,
        verdict=verdict,
    )


def _text(*artifacts: ResultArtifact) -> str:
    return render_results_index(ResultsIndex(artifacts=artifacts or (_artifact(),)))


def test_checked_in_index_exactly_covers_tracked_top_level_artifacts() -> None:
    index = validate_results_repository(ROOT)
    paths = [artifact.artifact_path for artifact in index.artifacts]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    assert INDEX_RELATIVE_PATH not in paths


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda value: value.update(extra=True), "unknown extra"),
        (lambda value: value.pop("date"), "missing date"),
        (lambda value: value.update(date="2026-02-30"), "real calendar date"),
        (lambda value: value.update(commit="abc"), "full lowercase Git SHA"),
        (lambda value: value.update(gate_id="G2 A1"), "lowercase slug"),
        (lambda value: value.update(verdict="maybe"), "verdict must be"),
        (
            lambda value: value.update(artifact_path="../outside.json"),
            "one top-level path",
        ),
        (
            lambda value: value.update(summary_path="bench/results/else/manifest.json"),
            "must be a descendant",
        ),
    ),
)
def test_artifact_schema_fails_closed(mutate, message: str) -> None:
    payload = json.loads(_text())
    mutate(payload["artifacts"][0])
    with pytest.raises(ResultsIndexError, match=message):
        parse_results_index(json.dumps(payload))


@pytest.mark.parametrize(
    ("text", "message"),
    (
        ('{"schema_version":1,"schema_version":1,"artifacts":[]}', "duplicate"),
        ('{"schema_version":1,"artifacts":[],"value":NaN}', "non-finite"),
        ('{"schema_version":true,"artifacts":[]}', "schema_version"),
        ('{"schema_version":1,"artifacts":[]}', "non-empty"),
    ),
)
def test_json_envelope_is_strict(text: str, message: str) -> None:
    with pytest.raises(ResultsIndexError, match=message):
        parse_results_index(text)


def test_paths_must_be_unique_and_sorted_but_gate_ids_may_repeat() -> None:
    alpha = _artifact()
    beta = _artifact("beta.json")
    parsed = parse_results_index(_text(alpha, beta))
    assert [artifact.gate_id for artifact in parsed.artifacts] == ["g1-a1", "g1-a1"]

    with pytest.raises(ResultsIndexError, match="sorted"):
        parse_results_index(_text(beta, alpha))
    with pytest.raises(ResultsIndexError, match="duplicate artifact paths"):
        parse_results_index(_text(alpha, alpha))


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        timeout=30,
    )


def _repository(tmp_path: Path) -> tuple[Path, ResultArtifact, ResultArtifact]:
    results = tmp_path / "bench" / "results"
    bundle = results / "bundle"
    bundle.mkdir(parents=True)
    (results / "alpha.json").write_text("{}\n", encoding="utf-8")
    (bundle / "manifest.json").write_text("{}\n", encoding="utf-8")
    alpha = _artifact()
    bundled = _artifact(
        "bundle",
        summary_path="bench/results/bundle/manifest.json",
    )
    (results / "index.json").write_text(_text(alpha, bundled), encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "bench/results")
    return tmp_path, alpha, bundled


def test_repository_validation_uses_git_coverage_not_filesystem_globs(
    tmp_path: Path,
) -> None:
    repo, _, _ = _repository(tmp_path)
    ignored = repo / "bench" / "results" / "ignored-runtime.json"
    ignored.write_text("{}\n", encoding="utf-8")

    index = validate_results_repository(repo)
    assert [artifact.artifact_path for artifact in index.artifacts] == [
        "bench/results/alpha.json",
        "bench/results/bundle",
    ]


def test_repository_validation_rejects_missing_and_stale_records(tmp_path: Path) -> None:
    repo, alpha, bundled = _repository(tmp_path)
    index_path = repo / INDEX_RELATIVE_PATH
    index_path.write_text(_text(alpha), encoding="utf-8")
    with pytest.raises(ResultsIndexError, match="missing records.*bundle"):
        validate_results_repository(repo)

    (repo / "bench" / "results" / "bundle" / "manifest.json").unlink()
    _git(repo, "add", "-u", "bench/results")
    index_path.write_text(_text(alpha, bundled), encoding="utf-8")
    with pytest.raises(ResultsIndexError, match="stale records.*bundle"):
        validate_results_repository(repo)


def test_directory_summary_must_be_tracked(tmp_path: Path) -> None:
    repo, alpha, bundled = _repository(tmp_path)
    untracked = repo / "bench" / "results" / "bundle" / "summary.json"
    untracked.write_text("{}\n", encoding="utf-8")
    record = replace(
        bundled,
        summary_path="bench/results/bundle/summary.json",
    )
    (repo / INDEX_RELATIVE_PATH).write_text(_text(alpha, record), encoding="utf-8")
    with pytest.raises(ResultsIndexError, match="tracked regular file"):
        validate_results_repository(repo)


def test_index_must_use_canonical_json(tmp_path: Path) -> None:
    repo, _, _ = _repository(tmp_path)
    index_path = repo / INDEX_RELATIVE_PATH
    index_path.write_text(
        json.dumps(json.loads(index_path.read_text(encoding="utf-8"))),
        encoding="utf-8",
    )
    with pytest.raises(ResultsIndexError, match="not canonical JSON"):
        validate_results_repository(repo)


def test_tracked_artifact_must_not_be_a_symlink(tmp_path: Path) -> None:
    repo, alpha, bundled = _repository(tmp_path)
    outside = repo / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    linked = repo / "bench" / "results" / "linked.json"
    linked.symlink_to(outside)
    _git(repo, "add", "bench/results/linked.json")
    linked_record = _artifact("linked.json")
    (repo / INDEX_RELATIVE_PATH).write_text(
        _text(alpha, bundled, linked_record),
        encoding="utf-8",
    )
    with pytest.raises(ResultsIndexError, match="must not contain a symlink"):
        validate_results_repository(repo)
