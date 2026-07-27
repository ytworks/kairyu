"""Fleet usage aggregation and independent request-log reconciliation.

Gateways retain ownership of their append-only ledgers.  This module is the
offline boundary that combines those immutable inputs; it is deliberately not
called from the request path.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

USAGE_FIELDS = (
    "requests",
    "prompt_tokens",
    "uncached_tokens",
    "cached_tokens",
    "completion_tokens",
)
TOKEN_FIELDS = USAGE_FIELDS[1:]


class UsageReconciliationError(ValueError):
    """A replay input is malformed or cannot be reconciled safely."""


def _empty_totals() -> dict[str, int]:
    return {field: 0 for field in USAGE_FIELDS}


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        handle = path.open(encoding="utf-8")
    except OSError as error:
        raise UsageReconciliationError(f"cannot read {path}: {error}") from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise UsageReconciliationError(
                    f"{path}:{line_number}: invalid JSON"
                ) from error
            if not isinstance(value, dict):
                raise UsageReconciliationError(
                    f"{path}:{line_number}: record must be an object"
                )
            yield line_number, value


def _usage_values(
    record: Mapping[str, Any],
    *,
    source: str,
) -> tuple[str, dict[str, int]]:
    tenant = record.get("tenant")
    if not isinstance(tenant, str) or not tenant:
        raise UsageReconciliationError(f"{source}: tenant must be a non-empty string")
    prompt = record.get("prompt_tokens")
    cached = record.get("cached_tokens", 0)
    values: dict[str, int] = {
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "uncached_tokens": record.get(
            "uncached_tokens",
            (
                prompt - cached
                if type(prompt) is int and type(cached) is int
                else None
            ),
        ),
        "completion_tokens": record.get("completion_tokens"),
    }
    for field, value in values.items():
        if type(value) is not int or value < 0:
            raise UsageReconciliationError(
                f"{source}: {field} must be a non-negative integer"
            )
    if (
        values["cached_tokens"] + values["uncached_tokens"]
        != values["prompt_tokens"]
    ):
        raise UsageReconciliationError(
            f"{source}: cached_tokens + uncached_tokens must equal prompt_tokens"
        )
    return tenant, values


def _add(
    totals: dict[str, dict[str, int]],
    tenant: str,
    values: Mapping[str, int],
) -> None:
    bucket = totals.setdefault(tenant, _empty_totals())
    bucket["requests"] += 1
    for field in TOKEN_FIELDS:
        bucket[field] += values[field]


def aggregate_gateway_ledgers(
    ledgers: Mapping[str, str | Path],
) -> dict[str, dict[str, int]]:
    """Aggregate independent gateway ledgers without sharing their write path."""
    if not ledgers:
        raise UsageReconciliationError("at least one gateway ledger is required")
    resolved: set[Path] = set()
    totals: dict[str, dict[str, int]] = {}
    for gateway, raw_path in sorted(ledgers.items()):
        if not gateway:
            raise UsageReconciliationError("gateway name must not be empty")
        path = Path(raw_path).resolve()
        if path in resolved:
            raise UsageReconciliationError(
                f"ledger path is assigned to more than one gateway: {path}"
            )
        resolved.add(path)
        for line_number, record in _read_jsonl(path):
            tenant, values = _usage_values(
                record,
                source=f"{path}:{line_number}",
            )
            _add(totals, tenant, values)
    return totals


def aggregate_request_logs(
    request_logs: Iterable[str | Path],
) -> dict[str, dict[str, int]]:
    """Independently aggregate metered outcomes from request audit logs."""
    paths = tuple(Path(path) for path in request_logs)
    if not paths:
        raise UsageReconciliationError("at least one request log is required")
    totals: dict[str, dict[str, int]] = {}
    event_ids: set[tuple[str, str]] = set()
    for path in paths:
        for line_number, record in _read_jsonl(path):
            source = f"{path}:{line_number}"
            gateway = record.get("gateway")
            event_id = record.get("event_id")
            metered = record.get("metered")
            if not isinstance(gateway, str) or not gateway:
                raise UsageReconciliationError(
                    f"{source}: gateway must be a non-empty string"
                )
            if not isinstance(event_id, str) or not event_id:
                raise UsageReconciliationError(
                    f"{source}: event_id must be a non-empty string"
                )
            if type(metered) is not bool:
                raise UsageReconciliationError(f"{source}: metered must be boolean")
            event_key = (gateway, event_id)
            if event_key in event_ids:
                raise UsageReconciliationError(
                    f"{source}: duplicate gateway event {gateway}/{event_id}"
                )
            event_ids.add(event_key)
            if not metered:
                continue
            tenant, values = _usage_values(record, source=source)
            _add(totals, tenant, values)
    return totals


def _relative_error_percent(expected: int, actual: int) -> float:
    if expected == 0:
        return 0.0 if actual == 0 else 100.0
    return abs(actual - expected) / expected * 100.0


def reconcile_usage(
    *,
    ledgers: Mapping[str, str | Path],
    request_logs: Iterable[str | Path],
    threshold_percent: float = 0.1,
) -> dict[str, Any]:
    """Return a deterministic fleet report against independent observations."""
    if threshold_percent <= 0:
        raise UsageReconciliationError("threshold_percent must be positive")
    ledger_totals = aggregate_gateway_ledgers(ledgers)
    request_totals = aggregate_request_logs(request_logs)
    if not request_totals:
        raise UsageReconciliationError(
            "request logs contain no metered usage; refusing an empty pass"
        )
    tenants: dict[str, Any] = {}
    maximum = 0.0
    for tenant in sorted(set(ledger_totals) | set(request_totals)):
        expected = request_totals.get(tenant, _empty_totals())
        actual = ledger_totals.get(tenant, _empty_totals())
        errors = {
            field: _relative_error_percent(expected[field], actual[field])
            for field in USAGE_FIELDS
        }
        tenant_maximum = max(errors.values())
        maximum = max(maximum, tenant_maximum)
        tenants[tenant] = {
            "request_log": expected,
            "ledger": actual,
            "relative_error_percent": errors,
            "max_relative_error_percent": tenant_maximum,
        }
    return {
        "schema_version": 1,
        "gate": "G5 F5e",
        "threshold_percent": threshold_percent,
        "gateway_count": len(ledgers),
        "tenants": tenants,
        "max_relative_error_percent": maximum,
        "passed": maximum < threshold_percent,
    }
