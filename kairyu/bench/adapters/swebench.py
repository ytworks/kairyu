"""Shared, fail-closed evidence contracts for official SWE-bench adapters."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from kairyu.bench.types import ItemResult

_COUNT_TO_IDS = {
    "submitted_instances": "submitted_ids",
    "completed_instances": "completed_ids",
    "resolved_instances": "resolved_ids",
    "unresolved_instances": "unresolved_ids",
    "empty_patch_instances": "empty_patch_ids",
    "error_instances": "error_ids",
}


def _required_count(report: dict, name: str) -> int:
    value = report.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _required_id_set(report: dict, name: str) -> set[str]:
    value = report.get(name)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicate IDs")
    return set(value)


def _selected_id_set(selected_ids: Sequence[str] | None) -> set[str] | None:
    if selected_ids is None:
        return None
    if any(not isinstance(item, str) or not item for item in selected_ids):
        raise ValueError("selected IDs must be non-empty strings")
    selected = set(selected_ids)
    if len(selected) != len(selected_ids):
        raise ValueError("selected IDs contain duplicates")
    return selected


def parse_swebench_report(
    report: dict,
    *,
    selected_ids: Sequence[str] | None = None,
) -> tuple[list[ItemResult], int]:
    """Validate schema-v2 partitions and retain every official outcome."""

    if not isinstance(report, dict):
        raise ValueError("SWE-bench report must be an object")
    if report.get("schema_version") != 2:
        raise ValueError("SWE-bench report schema_version must be 2")

    counts = {
        name: _required_count(report, name)
        for name in ("total_instances", *_COUNT_TO_IDS)
    }
    ids = {
        name: _required_id_set(report, name)
        for name in (*_COUNT_TO_IDS.values(), "incomplete_ids")
    }
    for count_name, ids_name in _COUNT_TO_IDS.items():
        if counts[count_name] != len(ids[ids_name]):
            raise ValueError(f"{count_name} does not match {ids_name}")

    if ids["resolved_ids"] & ids["unresolved_ids"]:
        raise ValueError("resolved_ids and unresolved_ids must be disjoint")
    if ids["completed_ids"] != ids["resolved_ids"] | ids["unresolved_ids"]:
        raise ValueError("completed_ids must equal resolved_ids plus unresolved_ids")

    terminal = (ids["completed_ids"], ids["empty_patch_ids"], ids["error_ids"])
    if any(
        left & right
        for index, left in enumerate(terminal)
        for right in terminal[index + 1 :]
    ):
        raise ValueError("submitted outcome IDs must be disjoint")
    if ids["submitted_ids"] != set().union(*terminal):
        raise ValueError(
            "submitted_ids must equal completed, empty-patch, and error IDs"
        )
    if ids["submitted_ids"] & ids["incomplete_ids"]:
        raise ValueError("submitted_ids and incomplete_ids must be disjoint")

    official_ids = ids["submitted_ids"] | ids["incomplete_ids"]
    selected = (
        official_ids if selected_ids is None else _selected_id_set(selected_ids)
    )
    assert selected is not None
    if selected != official_ids:
        raise ValueError("selected IDs must equal submitted_ids plus incomplete_ids")
    if counts["total_instances"] != len(selected):
        raise ValueError("total_instances does not match selected IDs")

    items = []
    for item_id in sorted(selected):
        if item_id in ids["resolved_ids"]:
            items.append(
                ItemResult(
                    item_id=item_id,
                    status="completed",
                    score=1.0,
                    details={"outcome": "resolved"},
                )
            )
        elif item_id in ids["unresolved_ids"]:
            items.append(
                ItemResult(
                    item_id=item_id,
                    status="completed",
                    score=0.0,
                    details={"outcome": "unresolved"},
                )
            )
        elif item_id in ids["empty_patch_ids"]:
            items.append(
                ItemResult(
                    item_id=item_id,
                    status="completed",
                    score=0.0,
                    details={"outcome": "empty_patch"},
                )
            )
        elif item_id in ids["error_ids"]:
            items.append(
                ItemResult(
                    item_id=item_id,
                    status="failed",
                    error="SWE-bench harness error",
                    details={"outcome": "error"},
                )
            )
        else:
            items.append(
                ItemResult(
                    item_id=item_id,
                    status="failed",
                    error="SWE-bench evaluation incomplete",
                    details={"outcome": "incomplete"},
                )
            )
    return items, counts["total_instances"]


def load_prediction_ids(path: Path) -> tuple[str, ...]:
    """Read mini-SWE-agent's prediction mapping and validate its row identity."""

    try:
        predictions = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"prediction file is not readable JSON: {error}") from error
    if not isinstance(predictions, dict) or not predictions:
        raise ValueError("prediction file must be a non-empty object")

    for instance_id, prediction in predictions.items():
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("prediction IDs must be non-empty strings")
        if not isinstance(prediction, dict):
            raise ValueError(f"prediction {instance_id!r} must be an object")
        if prediction.get("instance_id") != instance_id:
            raise ValueError(f"prediction {instance_id!r} has a mismatched instance_id")
        model = prediction.get("model_name_or_path")
        if not isinstance(model, str) or not model:
            raise ValueError(f"prediction {instance_id!r} has an invalid model_name_or_path")
        patch = prediction.get("model_patch")
        if patch is not None and not isinstance(patch, str):
            raise ValueError(f"prediction {instance_id!r} has an invalid model_patch")
    return tuple(sorted(predictions))


def find_swebench_report(workdir: Path) -> dict:
    """Return the sole root schema-v2 report produced by the official harness."""

    candidates = []
    for path in sorted(workdir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("schema_version") == 2:
            candidates.append(data)
    if not candidates:
        raise ValueError("no SWE-bench schema-v2 report produced")
    if len(candidates) > 1:
        raise ValueError("multiple SWE-bench schema-v2 reports produced")
    return candidates[0]


__all__ = [
    "find_swebench_report",
    "load_prediction_ids",
    "parse_swebench_report",
]
