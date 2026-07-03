"""HF Hub download helpers. The ONLY module that touches `datasets`/`huggingface_hub`.

Imports are lazy so the core package and the offline-fixtures path work with
zero extras installed; failures surface as the typed errors in types.py and
degrade to skipped cells at run time (the one-command guarantee).
"""

from __future__ import annotations

import os
from pathlib import Path

from kairyu.bench.types import BenchExtrasMissing, DatasetGated, DatasetUnavailable


def _import_datasets():
    try:
        import datasets
    except ImportError as error:
        raise BenchExtrasMissing("bench", "downloading benchmark datasets") from error
    return datasets


def _classify(dataset: str, error: Exception) -> Exception:
    text = str(error).lower()
    if "gated" in text or "401" in text or "403" in text or "authenticat" in text:
        return DatasetGated(dataset)
    return DatasetUnavailable(f"dataset {dataset!r} could not be fetched: {error}")


def load_hf_rows(
    dataset: str,
    *,
    name: str | None = None,
    split: str,
    revision: str | None = None,
    gated: bool = False,
) -> list[dict]:
    """Load one split as a list of dicts. Raises the typed errors on failure."""
    datasets = _import_datasets()
    token = os.environ.get("HF_TOKEN")
    if gated and not token:
        raise DatasetGated(dataset)
    try:
        loaded = datasets.load_dataset(
            dataset, name=name, split=split, revision=revision, token=token
        )
        return [dict(row) for row in loaded]
    except (BenchExtrasMissing, DatasetGated, DatasetUnavailable):
        raise
    except Exception as error:  # noqa: BLE001 - network/hub errors are library-specific
        raise _classify(dataset, error) from error


def save_asset(data: bytes, assets_dir: Path, filename: str) -> str:
    """Write a binary asset (e.g. an image) and return its cache-relative name."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / filename).write_bytes(data)
    return f"assets/{filename}"
