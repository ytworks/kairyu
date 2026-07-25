"""HF Hub download helpers. The ONLY module that touches `datasets`/`huggingface_hub`.

Imports are lazy so the core package and the offline-fixtures path work with
zero extras installed; failures surface as the typed errors in types.py and
degrade to skipped cells at run time (the one-command guarantee).
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
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


def _import_hub():
    try:
        import huggingface_hub
    except ImportError as error:
        raise BenchExtrasMissing("bench", "downloading benchmark assets") from error
    return huggingface_hub


def load_jsonl_files(
    repo_id: str,
    filenames: Sequence[str],
    *,
    revision: str | None = None,
    gated: bool = False,
) -> list[dict]:
    """Read named JSONL files from a dataset repo, concatenated in file order.

    Used for repos that only expose raw JSONL behind a loading script:
    `datasets` requires `trust_remote_code` for those from 2.20 and dropped
    script execution entirely in 4.x, so reading the files directly keeps the
    slot working across versions AND lets `revision` pin a real commit
    (a config name such as "release_v6" is not a git ref).
    """
    hub = _import_hub()
    token = os.environ.get("HF_TOKEN")
    if gated and not token:
        raise DatasetGated(repo_id)
    rows: list[dict] = []
    for filename in filenames:
        try:
            path = hub.hf_hub_download(
                repo_id,
                filename,
                repo_type="dataset",
                revision=revision,
                token=token,
            )
        except Exception as error:  # noqa: BLE001 - hub errors are library-specific
            raise _classify(repo_id, error) from error
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise DatasetUnavailable(
                        f"{repo_id}/{filename} line {line_number} is not valid JSON: {error}"
                    ) from error
    return rows


def download_file(
    repo_id: str,
    filename: str,
    dest: Path,
    *,
    repo_type: str = "dataset",
    revision: str | None = None,
) -> Path | None:
    """Fetch one repo file into `dest` (None if it doesn't exist upstream)."""
    hub = _import_hub()
    try:
        cached = hub.hf_hub_download(
            repo_id,
            filename,
            repo_type=repo_type,
            revision=revision,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception:  # noqa: BLE001 - missing file/network: caller degrades
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(Path(cached).read_bytes())
    return dest


def save_asset(data: bytes, assets_dir: Path, filename: str) -> str:
    """Write a binary asset (e.g. an image) and return its cache-relative name."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / filename).write_bytes(data)
    return f"assets/{filename}"
