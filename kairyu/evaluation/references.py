"""Offline, versioned reference-result snapshots for evaluation reports.

Reference data is deliberately loaded from package resources.  Rendering a report
must never turn into a web lookup, and a changed source row must fail its evidence
hash instead of silently changing a comparison.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator
from yaml.constructor import ConstructorError

from kairyu.evaluation.schemas import (
    Comparability,
    FrozenModel,
    Identifier,
    ReferenceResult,
    Source,
    SourceType,
)

GPQA_REFERENCE_RESOURCE = "resources/references/sakana-fugu-technical-report-2026-v2-gpqa.yaml"

_EXPECTED_GPQA_ROWS = (
    ("fugu-ultra-gpqa-diamond-2026-v2", "Fugu Ultra", 95.5),
    ("fugu-gpqa-diamond-2026-v2", "Fugu", 95.5),
    ("claude-opus-4.8-gpqa-diamond-2026-v2", "Claude Opus 4.8", 92.0),
    ("gemini-3.1-pro-gpqa-diamond-2026-v2", "Gemini 3.1 Pro", 94.3),
    ("gpt-5.5-gpqa-diamond-2026-v2", "GPT-5.5", 93.6),
)


class ReferenceDataError(ValueError):
    """A packaged reference snapshot is malformed or has lost provenance."""


class ReferenceSnapshot(FrozenModel):
    """One immutable, reviewable source snapshot and its reported results."""

    schema_version: int = Field(default=1, ge=1)
    snapshot_id: Identifier
    benchmark_id: Identifier
    source: Source
    results: tuple[ReferenceResult, ...]

    @model_validator(mode="after")
    def _records_are_coherent(self) -> ReferenceSnapshot:
        if not self.results:
            raise ValueError("reference snapshot must contain at least one result")
        reference_ids = [result.reference_id for result in self.results]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("reference IDs must be unique")
        model_names = [result.model_name for result in self.results]
        if len(model_names) != len(set(model_names)):
            raise ValueError("reference model names must be unique")

        for result in self.results:
            if result.benchmark_id != self.benchmark_id:
                raise ValueError("reference benchmark ID must match its snapshot")
            if result.source_id != self.source.source_id:
                raise ValueError("reference source ID must match its snapshot source")
            if result.source_type is not self.source.source_type:
                raise ValueError("reference source type must match its snapshot source")
            if not 0.0 <= result.score <= result.score_scale:
                raise ValueError("reference score must be within its declared scale")
            if result.evidence_hash != evidence_hash(result):
                raise ValueError(f"reference evidence hash mismatch for {result.reference_id}")

        if self.source.evidence_hash != evidence_hash(self.source):
            raise ValueError("source evidence hash mismatch")
        return self


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that also rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def evidence_hash(record: Source | ReferenceResult) -> str:
    """Hash all stored evidence fields except the hash itself."""

    payload = record.model_dump(
        mode="json",
        exclude={"evidence_hash"},
        exclude_none=False,
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_reference_snapshot(path: str | Path | None = None) -> ReferenceSnapshot:
    """Load and strictly validate the pinned GPQA reference snapshot.

    ``path`` exists for review tooling and tests.  With no path, only the packaged
    resource is read; this function performs no network access.
    """

    try:
        text = (
            Path(path).read_text(encoding="utf-8")
            if path is not None
            else resources.files("kairyu.evaluation")
            .joinpath(GPQA_REFERENCE_RESOURCE)
            .read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError) as exc:
        raise ReferenceDataError("reference snapshot could not be read") from exc

    try:
        payload = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ReferenceDataError("reference snapshot is not valid strict YAML") from exc
    if not isinstance(payload, Mapping):
        raise ReferenceDataError("reference snapshot root must be a mapping")

    try:
        snapshot = ReferenceSnapshot.model_validate(payload)
    except (TypeError, ValueError) as exc:
        raise ReferenceDataError("reference snapshot failed schema validation") from exc
    _validate_gpqa_snapshot(snapshot)
    return snapshot


def _validate_gpqa_snapshot(snapshot: ReferenceSnapshot) -> None:
    if snapshot.schema_version != 1:
        raise ReferenceDataError("unsupported GPQA reference snapshot schema version")
    if snapshot.snapshot_id != "sakana-fugu-technical-report-2026-v2-gpqa":
        raise ReferenceDataError("unexpected GPQA reference snapshot identity")
    if snapshot.benchmark_id != "gpqa-diamond":
        raise ReferenceDataError("GPQA reference snapshot has the wrong benchmark")
    if snapshot.source.source_type is not SourceType.PAPER_COMPILATION:
        raise ReferenceDataError("GPQA references must be a paper compilation")
    if snapshot.source.source_id != "sakana-fugu-technical-report-2026-v2":
        raise ReferenceDataError("unexpected GPQA source identity")
    if snapshot.source.url != "https://arxiv.org/pdf/2606.21228":
        raise ReferenceDataError("GPQA source must retain its reviewed arXiv URL")
    if snapshot.source.locator != "Table 1 (GPQA Diamond); Appendix A":
        raise ReferenceDataError("GPQA source must retain its reviewed locator")
    if snapshot.source.release_page != "https://sakana.ai/fugu-release/":
        raise ReferenceDataError("GPQA source must retain its reviewed release page")

    actual_rows = tuple(
        (result.reference_id, result.model_name, result.score) for result in snapshot.results
    )
    if actual_rows != _EXPECTED_GPQA_ROWS:
        raise ReferenceDataError("GPQA reference rows differ from the reviewed snapshot")
    for result in snapshot.results:
        if result.score_scale != 100.0:
            raise ReferenceDataError("GPQA reference scores must use the 0-100 scale")
        if result.source_type is not SourceType.PAPER_COMPILATION:
            raise ReferenceDataError("GPQA reference source type must be paper_compilation")
        if result.benchmark_version != "fugu-2026" or result.profile != "fugu-2026":
            raise ReferenceDataError("GPQA references must retain the reviewed profile")
        if result.metric_name != "accuracy" or result.sample_count != 198:
            raise ReferenceDataError("GPQA references must retain metric and sample evidence")
        if result.provider_reported is not None:
            raise ReferenceDataError("unverified provider reporting must remain null")
        if result.independently_reproduced:
            raise ReferenceDataError("paper-compiled GPQA scores are not reproduced results")
        if result.protocol_hash is not None:
            raise ReferenceDataError("unresolved GPQA references cannot claim a protocol hash")
        if result.comparability is not Comparability.INCOMPATIBLE:
            raise ReferenceDataError("GPQA references must remain incompatible")
