from importlib import resources

import pytest
from pydantic import ValidationError

from kairyu.evaluation.references import (
    GPQA_REFERENCE_RESOURCE,
    ReferenceDataError,
    evidence_hash,
    load_reference_snapshot,
)
from kairyu.evaluation.schemas import Comparability, SourceType


def test_packaged_gpqa_snapshot_has_the_reviewed_rows_and_provenance():
    snapshot = load_reference_snapshot()

    assert snapshot.schema_version == 1
    assert snapshot.snapshot_id == "sakana-fugu-technical-report-2026-v2-gpqa"
    assert snapshot.benchmark_id == "gpqa-diamond"
    assert snapshot.source.source_type is SourceType.PAPER_COMPILATION
    assert snapshot.source.url == "https://arxiv.org/pdf/2606.21228"
    assert snapshot.source.locator == "Table 1 (GPQA Diamond); Appendix A"
    assert snapshot.source.release_page == "https://sakana.ai/fugu-release/"
    assert snapshot.source.retrieved_at.isoformat() == "2026-07-24T00:00:00+00:00"

    assert [(result.model_name, result.score) for result in snapshot.results] == [
        ("Fugu Ultra", 95.5),
        ("Fugu", 95.5),
        ("Claude Opus 4.8", 92.0),
        ("Gemini 3.1 Pro", 94.3),
        ("GPT-5.5", 93.6),
    ]
    for result in snapshot.results:
        assert result.score_scale == 100.0
        assert 0.0 <= result.score <= 100.0
        assert result.source_type is SourceType.PAPER_COMPILATION
        assert result.provider_reported is None
        assert result.protocol_hash is None
        assert result.comparability is Comparability.INCOMPATIBLE
        assert result.evidence_hash == evidence_hash(result)
    assert snapshot.source.evidence_hash == evidence_hash(snapshot.source)


def test_reference_loading_is_offline(monkeypatch):
    def fail_network(*_args, **_kwargs):
        raise AssertionError("reference loading attempted network access")

    monkeypatch.setattr("socket.socket", fail_network)

    assert len(load_reference_snapshot().results) == 5


def test_reference_snapshot_rejects_duplicate_yaml_keys(tmp_path):
    resource = resources.files("kairyu.evaluation").joinpath(GPQA_REFERENCE_RESOURCE)
    text = resource.read_text(encoding="utf-8")
    path = tmp_path / "duplicate.yaml"
    path.write_text(text + "\nbenchmark_id: gpqa-diamond\n", encoding="utf-8")

    with pytest.raises(ReferenceDataError, match="strict YAML"):
        load_reference_snapshot(path)


def test_reference_snapshot_rejects_tampered_evidence(tmp_path):
    resource = resources.files("kairyu.evaluation").joinpath(GPQA_REFERENCE_RESOURCE)
    text = resource.read_text(encoding="utf-8")
    path = tmp_path / "tampered.yaml"
    path.write_text(text.replace("score: 95.5", "score: 95.4", 1), encoding="utf-8")

    with pytest.raises(ReferenceDataError, match="schema validation"):
        load_reference_snapshot(path)


def test_reference_snapshot_rejects_unreviewed_provider_claim(tmp_path):
    resource = resources.files("kairyu.evaluation").joinpath(GPQA_REFERENCE_RESOURCE)
    text = resource.read_text(encoding="utf-8")
    path = tmp_path / "provider-claim.yaml"
    path.write_text(
        text.replace("provider_reported: null", "provider_reported: true", 1),
        encoding="utf-8",
    )

    with pytest.raises(ReferenceDataError):
        load_reference_snapshot(path)


def test_snapshot_models_are_frozen():
    snapshot = load_reference_snapshot()

    with pytest.raises(ValidationError):
        snapshot.results[0].score = 0.0
