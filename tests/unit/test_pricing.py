import csv
import io
import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from kairyu.pricing import (
    InvoiceExportError,
    PriceRates,
    PriceSheet,
    export_invoice_csv,
)


def _sheet(
    *,
    version: str = "2026-07-27",
    tenant_discounts: dict[str, str] | None = None,
) -> PriceSheet:
    return PriceSheet(
        version=version,
        currency="USD",
        rates=PriceRates(
            uncached_input_per_million="2.00",
            cached_input_discount="0.75",
            output_per_million="10.00",
        ),
        tenant_discounts=tenant_discounts or {"tenant-a": "0.10"},
    )


def _jsonl(*records: dict) -> bytes:
    return b"".join(
        json.dumps(record, sort_keys=True).encode() + b"\n"
        for record in records
    )


def _rows(payload: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(payload)))


def test_price_sheet_validates_rates_discounts_version_and_currency():
    sheet = _sheet()

    assert sheet.rates.cached_input_per_million == Decimal("0.5000")

    invalid = (
        {"version": "contains space"},
        {"currency": "usd"},
        {"rates": {"cached_input_discount": "1.01"}},
        {"rates": {"uncached_input_per_million": "NaN"}},
        {"rates": {"cached_input_discount": True}},
        {"tenant_discounts": {"tenant-a": "-0.01"}},
        {"tenant_discounts": {"tenant-a": False}},
        {"tenant_discounts": {"": "0.1"}},
    )
    base = {
        "version": "v1",
        "currency": "USD",
        "rates": {
            "uncached_input_per_million": "2",
            "cached_input_discount": "0.5",
            "output_per_million": "10",
        },
        "tenant_discounts": {},
    }
    for override in invalid:
        candidate = {**base, **override}
        if "rates" in override:
            candidate["rates"] = {**base["rates"], **override["rates"]}
        with pytest.raises(ValidationError):
            PriceSheet.model_validate(candidate)
    with pytest.raises(TypeError):
        sheet.tenant_discounts["tenant-a"] = Decimal("0.50")


def test_invoice_csv_is_deterministic_reconciled_and_versioned():
    gateway_a = _jsonl(
        {
            "tenant": "tenant-a",
            "model": "qwen",
            "prompt_tokens": 1_000_000,
            "cached_tokens": 400_000,
            "uncached_tokens": 600_000,
            "completion_tokens": 100_000,
            "ts": 10,
        },
        {
            "tenant": "tenant-b",
            "model": "embed",
            "prompt_tokens": 250_000,
            "cached_tokens": 0,
            "uncached_tokens": 250_000,
            "completion_tokens": 0,
            "ts": 15,
        },
    )
    gateway_b = _jsonl(
        {
            "tenant": "tenant-a",
            "model": "qwen",
            "prompt_tokens": 500_000,
            "cached_tokens": 100_000,
            # Legacy rows derive uncached input deterministically.
            "completion_tokens": 50_000,
            "ts": 20,
        }
    )
    sources = {"gateway-b": gateway_b, "gateway-a": gateway_a}

    first = export_invoice_csv(sources, _sheet())
    second = export_invoice_csv(
        {"gateway-a": gateway_a, "gateway-b": gateway_b},
        _sheet(),
    )

    assert first == second
    rows = _rows(first)
    assert [row["tenant"] for row in rows] == ["tenant-a", "tenant-b"]
    tenant_a = rows[0]
    assert tenant_a["requests"] == "2"
    assert tenant_a["uncached_input_tokens"] == "1000000"
    assert tenant_a["cached_input_tokens"] == "500000"
    assert tenant_a["output_tokens"] == "150000"
    assert tenant_a["uncached_input_rate_per_million"] == "2.00"
    assert tenant_a["cached_input_rate_per_million"] == "0.5000"
    assert tenant_a["output_rate_per_million"] == "10.00"
    assert tenant_a["uncached_input_charge"] == "2.000000"
    assert tenant_a["cached_input_charge"] == "0.250000"
    assert tenant_a["output_charge"] == "1.500000"
    assert tenant_a["subtotal"] == "3.750000"
    assert tenant_a["tenant_discount_amount"] == "0.375000"
    assert tenant_a["total"] == "3.375000"
    assert len(tenant_a["source_sha256"]) == 64

    changed_version = _rows(
        export_invoice_csv(sources, _sheet(version="2026-08-01"))
    )[0]
    assert changed_version["price_sheet_version"] == "2026-08-01"
    assert changed_version["invoice_id"] != tenant_a["invoice_id"]
    assert changed_version["total"] == tenant_a["total"]


def test_invoice_period_and_tenant_filters_are_start_inclusive_end_exclusive():
    source = _jsonl(
        {
            "tenant": "tenant-a",
            "model": "qwen",
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "cached_tokens": 4,
            "ts": 10,
        },
        {
            "tenant": "tenant-a",
            "model": "qwen",
            "prompt_tokens": 20,
            "completion_tokens": 2,
            "cached_tokens": 8,
            "ts": 20,
        },
        {
            "tenant": "tenant-b",
            "model": "qwen",
            "prompt_tokens": 30,
            "completion_tokens": 3,
            "cached_tokens": 0,
            "ts": 20,
        },
    )

    rows = _rows(
        export_invoice_csv(
            {"gateway": source},
            _sheet(),
            tenant="tenant-a",
            start_ts=20,
            end_ts=21,
        )
    )

    assert len(rows) == 1
    assert rows[0]["requests"] == "1"
    assert rows[0]["uncached_input_tokens"] == "12"
    assert rows[0]["cached_input_tokens"] == "8"
    assert rows[0]["billing_start_ts"] == "20"
    assert rows[0]["billing_end_ts"] == "21"


def test_invoice_identity_is_stable_across_equivalent_period_types_and_later_rows():
    record = {
        "tenant": "tenant-a",
        "model": "qwen",
        "prompt_tokens": 20,
        "completion_tokens": 2,
        "cached_tokens": 8,
        "ts": 20,
    }
    later = {
        "tenant": "tenant-a",
        "model": "qwen",
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "cached_tokens": 0,
        "ts": 30,
    }

    before_append = export_invoice_csv(
        {"gateway": _jsonl(record)},
        _sheet(),
        start_ts=10,
        end_ts=21,
    )
    after_append = export_invoice_csv(
        {"gateway": _jsonl(record, later)},
        _sheet(),
        start_ts=10.0,
        end_ts=21.0,
    )

    assert before_append == after_append


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"tenant":"tenant-a"', "truncated final line"),
        (b"not-json\n", "invalid JSON"),
        (
            _jsonl(
                {
                    "tenant": "tenant-a",
                    "model": "qwen",
                    "prompt_tokens": 10,
                    "cached_tokens": 8,
                    "uncached_tokens": 8,
                    "completion_tokens": 1,
                    "ts": 10,
                }
            ),
            "must equal prompt_tokens",
        ),
        (
            _jsonl(
                {
                    "tenant": "tenant-a",
                    "prompt_tokens": 10,
                    "cached_tokens": 0,
                    "completion_tokens": 1,
                    "ts": 10,
                }
            ),
            "model must be",
        ),
    ],
)
def test_invoice_export_fails_closed_on_malformed_records(payload, message):
    with pytest.raises(InvoiceExportError, match=message):
        export_invoice_csv({"gateway": payload}, _sheet())


@pytest.mark.parametrize(
    ("start_ts", "end_ts"),
    [(10, 10), (11, 10), (float("nan"), None), (None, float("inf"))],
)
def test_invoice_export_rejects_invalid_periods(start_ts, end_ts):
    with pytest.raises(InvoiceExportError):
        export_invoice_csv(
            {"gateway": b""},
            _sheet(),
            start_ts=start_ts,
            end_ts=end_ts,
        )


def test_invoice_export_rejects_duplicate_ledger_paths(tmp_path):
    ledger = tmp_path / "usage.jsonl"
    ledger.write_bytes(b"")

    with pytest.raises(InvoiceExportError, match="more than one source"):
        export_invoice_csv(
            {"gateway-a": ledger, "gateway-b": ledger},
            _sheet(),
        )
