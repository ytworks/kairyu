"""Deterministic cached-token pricing and invoice CSV export."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections.abc import Mapping
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_MILLION = Decimal(1_000_000)
_MONEY_QUANTUM = Decimal("0.000001")
_INVOICE_FIELDS = (
    "invoice_id",
    "price_sheet_version",
    "currency",
    "source_sha256",
    "tenant",
    "billing_start_ts",
    "billing_end_ts",
    "requests",
    "uncached_input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "uncached_input_rate_per_million",
    "cached_input_rate_per_million",
    "output_rate_per_million",
    "tenant_discount_fraction",
    "uncached_input_charge",
    "cached_input_charge",
    "output_charge",
    "subtotal",
    "tenant_discount_amount",
    "total",
)


class InvoiceExportError(ValueError):
    """A ledger snapshot cannot safely produce an invoice."""


class _FrozenDict(dict):
    def _immutable(self, *_args, **_kwargs) -> None:
        raise TypeError("price-sheet mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


class PriceRates(BaseModel):
    """Blended base rates and the cached-input discount rule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    uncached_input_per_million: Decimal = Field(ge=0)
    cached_input_discount: Decimal = Field(ge=0, le=1)
    output_per_million: Decimal = Field(ge=0)

    @field_validator(
        "uncached_input_per_million",
        "cached_input_discount",
        "output_per_million",
        mode="before",
    )
    @classmethod
    def _not_boolean(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("price values must not be boolean")
        return value

    @field_validator(
        "uncached_input_per_million",
        "cached_input_discount",
        "output_per_million",
    )
    @classmethod
    def _finite(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("price values must be finite")
        return value

    @property
    def cached_input_per_million(self) -> Decimal:
        return self.uncached_input_per_million * (
            Decimal(1) - self.cached_input_discount
        )


class PriceSheet(BaseModel):
    """Versioned price configuration applied outside the request path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    rates: PriceRates
    tenant_discounts: dict[str, Decimal] = Field(default_factory=dict)

    @field_validator("tenant_discounts", mode="before")
    @classmethod
    def _discounts_not_boolean(cls, discounts: Any) -> Any:
        if isinstance(discounts, Mapping) and any(
            isinstance(value, bool) for value in discounts.values()
        ):
            raise ValueError("tenant discounts must not be boolean")
        return discounts

    @field_validator("tenant_discounts")
    @classmethod
    def _valid_tenant_discounts(
        cls,
        discounts: Mapping[str, Decimal],
    ) -> dict[str, Decimal]:
        for tenant, discount in discounts.items():
            if not tenant.strip():
                raise ValueError("tenant discount names must not be empty")
            if not discount.is_finite() or discount < 0 or discount > 1:
                raise ValueError(
                    "tenant discounts must be finite fractions between 0 and 1"
                )
        return _FrozenDict(discounts)


def _source_bytes(source: bytes | str | Path) -> bytes:
    if isinstance(source, bytes):
        return source
    try:
        return Path(source).read_bytes()
    except OSError as error:
        raise InvoiceExportError(f"cannot read ledger source: {error}") from error


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _record_usage(
    record: Mapping[str, Any],
    *,
    source: str,
) -> tuple[str, int | float, int, int, int]:
    tenant = record.get("tenant")
    model = record.get("model")
    timestamp = record.get("ts")
    if not isinstance(tenant, str) or not tenant:
        raise InvoiceExportError(f"{source}: tenant must be a non-empty string")
    if not isinstance(model, str) or not model:
        raise InvoiceExportError(f"{source}: model must be a non-empty string")
    if not _is_finite_number(timestamp):
        raise InvoiceExportError(f"{source}: ts must be a finite number")
    prompt = record.get("prompt_tokens")
    cached = record.get("cached_tokens", 0)
    completion = record.get("completion_tokens")
    uncached = record.get(
        "uncached_tokens",
        prompt - cached
        if type(prompt) is int and type(cached) is int
        else None,
    )
    fields = {
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "uncached_tokens": uncached,
        "completion_tokens": completion,
    }
    for field, value in fields.items():
        if type(value) is not int or value < 0:
            raise InvoiceExportError(
                f"{source}: {field} must be a non-negative integer"
            )
    if cached + uncached != prompt:
        raise InvoiceExportError(
            f"{source}: cached_tokens + uncached_tokens must equal prompt_tokens"
        )
    return tenant, timestamp, uncached, cached, completion


def _aggregate(
    sources: Mapping[str, bytes],
    *,
    tenant: str | None,
    start_ts: float | None,
    end_ts: float | None,
) -> tuple[dict[str, dict[str, int]], dict[str, str]]:
    totals: dict[str, dict[str, int]] = {}
    digests: dict[str, Any] = {}
    for source_name, payload in sorted(sources.items()):
        if payload and not payload.endswith(b"\n"):
            raise InvoiceExportError(
                f"{source_name}: ledger snapshot has a truncated final line"
            )
        for line_number, raw_line in enumerate(payload.splitlines(), start=1):
            if not raw_line.strip():
                continue
            source = f"{source_name}:{line_number}"
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise InvoiceExportError(f"{source}: invalid JSON") from error
            if not isinstance(record, dict):
                raise InvoiceExportError(f"{source}: record must be an object")
            (
                record_tenant,
                timestamp,
                uncached,
                cached,
                completion,
            ) = _record_usage(record, source=source)
            if tenant is not None and record_tenant != tenant:
                continue
            if start_ts is not None and timestamp < start_ts:
                continue
            if end_ts is not None and timestamp >= end_ts:
                continue
            digest = digests.setdefault(record_tenant, hashlib.sha256())
            encoded_source = source_name.encode()
            digest.update(len(encoded_source).to_bytes(8, "big"))
            digest.update(encoded_source)
            digest.update(line_number.to_bytes(8, "big"))
            digest.update(len(raw_line).to_bytes(8, "big"))
            digest.update(raw_line)
            bucket = totals.setdefault(
                record_tenant,
                {
                    "requests": 0,
                    "uncached_input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                },
            )
            bucket["requests"] += 1
            bucket["uncached_input_tokens"] += uncached
            bucket["cached_input_tokens"] += cached
            bucket["output_tokens"] += completion
    return totals, {
        record_tenant: digest.hexdigest()
        for record_tenant, digest in digests.items()
    }


def _money(tokens: int, rate: Decimal) -> Decimal:
    return (Decimal(tokens) * rate / _MILLION).quantize(
        _MONEY_QUANTUM,
        rounding=ROUND_HALF_EVEN,
    )


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _timestamp_text(value: int | float | None) -> str:
    if value is None:
        return ""
    decimal = Decimal(str(value))
    if decimal == 0:
        return "0"
    return format(decimal.normalize(), "f")


def export_invoice_csv(
    ledger_sources: Mapping[str, bytes | str | Path],
    price_sheet: PriceSheet,
    *,
    tenant: str | None = None,
    start_ts: float | None = None,
    end_ts: float | None = None,
) -> str:
    """Export deterministic tenant invoices from immutable ledger snapshots."""
    if not ledger_sources:
        raise InvoiceExportError("at least one ledger source is required")
    for name in ledger_sources:
        if not name:
            raise InvoiceExportError("ledger source names must not be empty")
    for name, value in (("start_ts", start_ts), ("end_ts", end_ts)):
        if value is not None and not _is_finite_number(value):
            raise InvoiceExportError(f"{name} must be a finite number")
        if value is not None and value < 0:
            raise InvoiceExportError(f"{name} must be non-negative")
    if start_ts is not None and end_ts is not None and start_ts >= end_ts:
        raise InvoiceExportError("start_ts must be less than end_ts")

    resolved_paths: set[Path] = set()
    sources: dict[str, bytes] = {}
    for name, source in ledger_sources.items():
        if not isinstance(source, bytes):
            path = Path(source).resolve()
            if path in resolved_paths:
                raise InvoiceExportError(
                    f"ledger path is assigned to more than one source: {path}"
                )
            resolved_paths.add(path)
        sources[name] = _source_bytes(source)
    totals, source_digests = _aggregate(
        sources,
        tenant=tenant,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=_INVOICE_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    cached_rate = price_sheet.rates.cached_input_per_million
    for record_tenant, usage in sorted(totals.items()):
        tenant_discount = price_sheet.tenant_discounts.get(
            record_tenant,
            Decimal(0),
        )
        uncached_charge = _money(
            usage["uncached_input_tokens"],
            price_sheet.rates.uncached_input_per_million,
        )
        cached_charge = _money(
            usage["cached_input_tokens"],
            cached_rate,
        )
        output_charge = _money(
            usage["output_tokens"],
            price_sheet.rates.output_per_million,
        )
        subtotal = uncached_charge + cached_charge + output_charge
        discount_amount = (subtotal * tenant_discount).quantize(
            _MONEY_QUANTUM,
            rounding=ROUND_HALF_EVEN,
        )
        total = subtotal - discount_amount
        invoice_seed = "|".join(
            (
                price_sheet.version,
                record_tenant,
                _timestamp_text(start_ts),
                _timestamp_text(end_ts),
                source_digests[record_tenant],
            )
        )
        invoice_id = "kinv_" + hashlib.sha256(invoice_seed.encode()).hexdigest()[:24]
        writer.writerow(
            {
                "invoice_id": invoice_id,
                "price_sheet_version": price_sheet.version,
                "currency": price_sheet.currency,
                "source_sha256": source_digests[record_tenant],
                "tenant": record_tenant,
                "billing_start_ts": _timestamp_text(start_ts),
                "billing_end_ts": _timestamp_text(end_ts),
                **usage,
                "uncached_input_rate_per_million": _decimal_text(
                    price_sheet.rates.uncached_input_per_million
                ),
                "cached_input_rate_per_million": _decimal_text(cached_rate),
                "output_rate_per_million": _decimal_text(
                    price_sheet.rates.output_per_million
                ),
                "tenant_discount_fraction": _decimal_text(tenant_discount),
                "uncached_input_charge": _decimal_text(uncached_charge),
                "cached_input_charge": _decimal_text(cached_charge),
                "output_charge": _decimal_text(output_charge),
                "subtotal": _decimal_text(subtotal),
                "tenant_discount_amount": _decimal_text(discount_amount),
                "total": _decimal_text(total),
            }
        )
    return output.getvalue()
