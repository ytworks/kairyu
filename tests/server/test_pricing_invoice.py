import csv
import io
import json
from dataclasses import replace
from decimal import Decimal

import httpx
import pytest

from kairyu.deploy.builder import build_app_from_spec
from kairyu.deploy.spec import load_deployment_spec
from kairyu.engine.backend import GenerationUsage
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.extra_routes import MockEmbeddingBackend
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import TenantConfig
from kairyu.pricing import PriceRates, PriceSheet
from tests.server._legacy_chat import create_legacy_app


def _sheet() -> PriceSheet:
    return PriceSheet(
        version="2026-07-27",
        currency="USD",
        rates=PriceRates(
            uncached_input_per_million="2.00",
            cached_input_discount="0.50",
            output_per_million="10.00",
        ),
        tenant_discounts={"tenant-a": "0.10"},
    )


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


def _csv_rows(response: httpx.Response) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(response.text)))


def test_deployment_price_sheet_is_validated_with_tenant_scope(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("KAIRYU_PRICE_SPEC_KEYS", "key-a")
    spec = load_deployment_spec(
        f"""
server:
  api_keys_env: KAIRYU_PRICE_SPEC_KEYS
  usage_ledger_path: {tmp_path / "usage.jsonl"}
engines:
  m: {{ backend: mock }}
tenants:
  key_tenants:
    key-a: tenant-a
pricing:
  version: "2026-07-27"
  currency: USD
  rates:
    uncached_input_per_million: "2.00"
    cached_input_discount: "0.50"
    output_per_million: "10.00"
  tenant_discounts:
    tenant-a: "0.10"
"""
    )

    assert spec.pricing is not None
    assert spec.pricing.rates.cached_input_per_million == 1
    assert spec.pricing.tenant_discounts["tenant-a"] == Decimal("0.10")
    app = build_app_from_spec(spec)
    assert app.state.price_sheet is spec.pricing
    assert any(route.path == "/admin/usage.csv" for route in app.routes)
    app.state.usage_ledger.close()

    with pytest.raises(ValueError, match="usage_ledger_path"):
        load_deployment_spec(
            """
engines:
  m: { backend: mock }
pricing:
  version: v1
  currency: USD
  rates:
    uncached_input_per_million: 2
    cached_input_discount: 0.5
    output_per_million: 10
"""
        )
    with pytest.raises(ValueError, match="unknown tenants"):
        load_deployment_spec(
            f"""
server:
  usage_ledger_path: {tmp_path / "usage.jsonl"}
engines:
  m: {{ backend: mock }}
pricing:
  version: v1
  currency: USD
  rates:
    uncached_input_per_million: 2
    cached_input_discount: 0.5
    output_per_million: 10
  tenant_discounts:
    unknown: 0.1
"""
        )


async def test_invoice_export_covers_stream_responses_embeddings_and_scope(
    tmp_path,
    monkeypatch,
):
    class CachedBackend(MockBackend):
        async def generate(self, request):
            return replace(
                await super().generate(request),
                usage=GenerationUsage(
                    prompt_tokens=20,
                    completion_tokens=5,
                    cached_tokens=8,
                ),
            )

        async def stream(self, request):
            result = await self.generate(request)
            yield result

    monkeypatch.setenv("KAIRYU_PRICING_KEYS", "key-a,key-b")
    monkeypatch.setenv("KAIRYU_PRICING_ADMIN_KEYS", "admin")
    ledger_path = tmp_path / "usage.jsonl"
    app = create_legacy_app(
        {"m": CachedBackend()},
        settings=ServerSettings(
            api_keys_env="KAIRYU_PRICING_KEYS",
            admin_keys_env="KAIRYU_PRICING_ADMIN_KEYS",
            usage_ledger_path=str(ledger_path),
        ),
        tenant_config=TenantConfig(
            key_tenants={"key-a": "tenant-a", "key-b": "tenant-b"}
        ),
        embedding_backends={"embed": MockEmbeddingBackend(dimensions=4)},
        price_sheet=_sheet(),
    )
    tenant_a = {"Authorization": "Bearer key-a"}
    tenant_b = {"Authorization": "Bearer key-b"}
    admin = {"Authorization": "Bearer admin"}

    async with app.router.lifespan_context(app):
        async with _client(app) as client:
            sync = await client.post(
                "/v1/chat/completions",
                headers=tenant_a,
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "sync"}],
                },
            )
            stream = await client.post(
                "/v1/chat/completions",
                headers=tenant_a,
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "stream"}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
            responses = await client.post(
                "/v1/responses",
                headers=tenant_a,
                json={"model": "m", "input": "response"},
            )
            embeddings = await client.post(
                "/v1/embeddings",
                headers=tenant_a,
                json={
                    "model": "embed",
                    "input": "two words",
                    "encoding_format": "float",
                },
            )
            own_invoice = await client.get(
                "/admin/usage.csv",
                headers=tenant_a,
            )
            forbidden = await client.get(
                "/admin/usage.csv?tenant=tenant-b",
                headers=tenant_a,
            )
            empty_tenant = await client.get(
                "/admin/usage.csv",
                headers=tenant_b,
            )
            fleet_invoice = await client.get(
                "/admin/usage.csv",
                headers=admin,
            )

    assert [response.status_code for response in (
        sync,
        stream,
        responses,
        embeddings,
    )] == [200, 200, 200, 200]
    assert own_invoice.status_code == 200
    assert own_invoice.headers["content-type"].startswith("text/csv")
    assert (
        own_invoice.headers["content-disposition"]
        == 'attachment; filename="kairyu-usage-invoice.csv"'
    )
    rows = _csv_rows(own_invoice)
    assert len(rows) == 1
    row = rows[0]
    assert row["tenant"] == "tenant-a"
    assert row["requests"] == "4"
    assert row["uncached_input_tokens"] == "38"
    assert row["cached_input_tokens"] == "24"
    assert row["output_tokens"] == "15"
    assert row["uncached_input_charge"] == "0.000076"
    assert row["cached_input_charge"] == "0.000024"
    assert row["output_charge"] == "0.000150"
    assert row["subtotal"] == "0.000250"
    assert row["tenant_discount_amount"] == "0.000025"
    assert row["total"] == "0.000225"
    assert forbidden.status_code == 403
    assert _csv_rows(empty_tenant) == []
    assert fleet_invoice.text == own_invoice.text

    ledger_records = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(ledger_records) == 4
    assert all("uncached_tokens" in record for record in ledger_records)
    assert all(
        record["cached_tokens"] + record["uncached_tokens"]
        == record["prompt_tokens"]
        for record in ledger_records
    )


async def test_invoice_endpoint_fails_closed_on_corrupt_ledger(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    app = create_legacy_app(
        {"m": MockBackend()},
        settings=ServerSettings(usage_ledger_path=str(ledger_path)),
        price_sheet=_sheet().model_copy(update={"tenant_discounts": {}}),
    )
    app.state.usage_ledger.record(
        "default",
        "m",
        prompt_tokens=10,
        completion_tokens=2,
        cached_tokens=4,
    )
    app.state.usage_ledger.flush()
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write("not-json\n")

    async with app.router.lifespan_context(app):
        async with _client(app) as client:
            response = await client.get("/admin/usage.csv")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "invoice_ledger_invalid"
    assert "invalid JSON" in response.json()["error"]["message"]
