from importlib import resources

import pytest
import yaml
from pydantic import ValidationError

from kairyu.evaluation.references import (
    GPQA_REFERENCE_RESOURCE,
    HLE_REFERENCE_RESOURCE,
    REFERENCE_RESOURCES_BY_BENCHMARK,
    ReferenceDataError,
    available_reference_benchmark_ids,
    evidence_hash,
    load_reference_snapshot,
    reference_resource_for_benchmark,
)
from kairyu.evaluation.schemas import Comparability, ReferenceResult, SourceType


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


def test_packaged_hle_snapshot_has_the_reviewed_rows_and_provenance():
    snapshot = load_reference_snapshot(benchmark_id="humanitys-last-exam")

    assert snapshot.schema_version == 1
    assert snapshot.snapshot_id == "sakana-fugu-technical-report-2026-v2-hle"
    assert snapshot.benchmark_id == "humanitys-last-exam"
    assert snapshot.source.locator == "Table 1 (Humanity's Last Exam); Appendix A"
    assert [(result.model_name, result.score) for result in snapshot.results] == [
        ("Fugu Ultra", 50.0),
        ("Fugu", 47.2),
        ("Claude Opus 4.8", 49.8),
        ("Gemini 3.1 Pro", 44.4),
        ("GPT-5.5", 41.4),
    ]
    assert all(result.sample_count == 2500 for result in snapshot.results)
    assert all(result.evidence_hash == evidence_hash(result) for result in snapshot.results)
    assert snapshot.source.evidence_hash == evidence_hash(snapshot.source)


def test_reference_resources_are_indexed_by_benchmark():
    assert available_reference_benchmark_ids() == (
        "gpqa-diamond",
        "humanitys-last-exam",
    )
    assert dict(REFERENCE_RESOURCES_BY_BENCHMARK) == {
        "gpqa-diamond": GPQA_REFERENCE_RESOURCE,
        "humanitys-last-exam": HLE_REFERENCE_RESOURCE,
    }
    assert reference_resource_for_benchmark("gpqa-diamond") == GPQA_REFERENCE_RESOURCE
    assert reference_resource_for_benchmark("humanitys-last-exam") == HLE_REFERENCE_RESOURCE


def test_reference_loading_is_offline(monkeypatch):
    def fail_network(*_args, **_kwargs):
        raise AssertionError("reference loading attempted network access")

    monkeypatch.setattr("socket.socket", fail_network)

    assert len(load_reference_snapshot().results) == 5
    assert len(load_reference_snapshot(benchmark_id="humanitys-last-exam").results) == 5


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


def test_reference_snapshot_cannot_cross_benchmark_boundaries():
    gpqa_resource = resources.files("kairyu.evaluation").joinpath(GPQA_REFERENCE_RESOURCE)

    with pytest.raises(ReferenceDataError, match="requested benchmark"):
        load_reference_snapshot(
            gpqa_resource,
            benchmark_id="humanitys-last-exam",
        )


def test_reference_snapshot_rejects_unsupported_schema_version(tmp_path):
    resource = resources.files("kairyu.evaluation").joinpath(GPQA_REFERENCE_RESOURCE)
    text = resource.read_text(encoding="utf-8")
    path = tmp_path / "future-schema.yaml"
    path.write_text(text.replace("schema_version: 1", "schema_version: 2", 1))

    with pytest.raises(ReferenceDataError, match="schema version"):
        load_reference_snapshot(path)


def test_reference_snapshot_rejects_unsupported_record_schema_version(tmp_path):
    resource = resources.files("kairyu.evaluation").joinpath(GPQA_REFERENCE_RESOURCE)
    payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
    result_payload = payload["results"][0]
    result_payload["schema_version"] = 2
    result_payload["evidence_hash"] = evidence_hash(ReferenceResult.model_validate(result_payload))
    path = tmp_path / "future-record-schema.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ReferenceDataError, match="schema version"):
        load_reference_snapshot(path)


def test_reference_index_rejects_unknown_and_unsafe_benchmark_ids(tmp_path):
    with pytest.raises(KeyError, match="invalid benchmark ID"):
        load_reference_snapshot(tmp_path / "unused.yaml", benchmark_id="../gpqa-diamond")
    with pytest.raises(KeyError, match="no reference snapshot"):
        load_reference_snapshot(tmp_path / "unused.yaml", benchmark_id="unreviewed-bench")


def test_snapshot_models_are_frozen():
    snapshot = load_reference_snapshot()

    with pytest.raises(ValidationError):
        snapshot.results[0].score = 0.0
