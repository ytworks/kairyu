"""m11 gates: streaming orchestrator usage, tiers, tenancy, responses,
embeddings, vision wire, F5 logic, bench schema."""

import base64
import struct

import pytest
from fastapi.testclient import TestClient

from kairyu.engine.mock import MockBackend
from kairyu.engine.registry import create_backend
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.extra_routes import MockEmbeddingBackend
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import (
    TenantConfig,
    TenantLimiter,
    TenantLimits,
    UsageLedger,
)
from kairyu.orchestration.orchestrator import Orchestrator


def _auto_app(tmp_path, **kwargs):
    engine = create_backend("mock")
    orchestrator = Orchestrator({"tier1": engine, "tier2": engine})
    deep = Orchestrator({"tier1": engine, "tier2": engine}, moa_samples=2)
    return create_app(
        {"m": engine},
        orchestrators={"kairyu-auto": orchestrator, "kairyu-auto-max": deep},
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
        embedding_backend=MockEmbeddingBackend(dimensions=8),
        **kwargs,
    )


class TestOrchestratorSurface:
    def test_tiers_listed_and_auto_usage_real(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            models = [m["id"] for m in client.get("/v1/models").json()["data"]]
            assert "kairyu-auto" in models and "kairyu-auto-max" in models

            response = client.post(
                "/v1/chat/completions",
                json={"model": "kairyu-auto",
                      "messages": [{"role": "user", "content": "hello"}]},
            )
            assert response.status_code == 200
            usage = response.json()["usage"]
            assert usage["completion_tokens"] > 0  # m11 A1: real, not zero
            assert "kairyu_trace" not in response.json() or response.json()["kairyu_trace"] is None

    def test_auto_model_rejects_unsupported_params(self, tmp_path):
        # M4: params the orchestrator can't honor (n>1, logprobs, tools,
        # response_format) must 400, not be silently dropped.
        with TestClient(_auto_app(tmp_path)) as client:
            for extra in (
                {"n": 2},
                {"logprobs": True},
                {"tools": [{"type": "function", "function": {"name": "f"}}]},
                {"response_format": {"type": "json_object"}},
            ):
                resp = client.post(
                    "/v1/chat/completions",
                    json={"model": "kairyu-auto",
                          "messages": [{"role": "user", "content": "hi"}], **extra},
                )
                assert resp.status_code == 400, extra

    def test_trace_header_opt_in(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"X-Kairyu-Trace": "1"},
                json={"model": "kairyu-auto",
                      "messages": [{"role": "user", "content": "hello"}]},
            )
            trace = response.json().get("kairyu_trace")
            assert trace and any("route:" in line for line in trace)

    def test_auto_stream_chunks_and_usage(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            with client.stream(
                "POST", "/v1/chat/completions",
                json={"model": "kairyu-auto", "stream": True,
                      "stream_options": {"include_usage": True},
                      "messages": [{"role": "user", "content": "hello"}]},
            ) as response:
                body = "".join(response.iter_text())
        assert "data: [DONE]" in body
        import json as _json

        data_lines = [
            line[len("data: "):]
            for line in body.splitlines()
            if line.startswith("data: ") and "[DONE]" not in line
        ]
        chunks = [_json.loads(line) for line in data_lines]
        assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
        usage_chunks = [c for c in chunks if c.get("usage")]
        assert usage_chunks and usage_chunks[-1]["usage"]["completion_tokens"] > 0

    def test_moa_tier_synthesizes(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"X-Kairyu-Trace": "1"},
                json={"model": "kairyu-auto-max",
                      "messages": [{"role": "user", "content":
                                    "analyze compare and plan: " + "x" * 2500}]},
            )
            trace = response.json().get("kairyu_trace") or []
            assert any("moa" in line for line in trace), trace


class TestTenancy:
    def test_stream_usage_owner_finalizes_once(self, tmp_path):
        from kairyu.engine.backend import GenerationUsage
        from kairyu.entrypoints.server.metering import StreamUsageOwner
        from kairyu.outputs import CompletionOutput

        ledger = UsageLedger(tmp_path / "usage.jsonl")
        owner = StreamUsageOwner(
            tenant="tenant-a",
            model="model-a",
            prompt="ignored prompt",
            ledger=ledger,
        )
        owner.mark_dispatched()
        owner.observe(
            GenerationUsage(prompt_tokens=7, completion_tokens=5),
            (CompletionOutput(index=0, text="ignored", token_ids=()),),
        )

        owner.finalize()
        owner.finalize()

        assert ledger.totals()["tenant-a"] == {
            "requests": 1,
            "prompt_tokens": 7,
            "completion_tokens": 5,
        }

    def test_stream_usage_owner_skips_undispatched_stream(self, tmp_path):
        from kairyu.entrypoints.server.metering import StreamUsageOwner

        ledger_path = tmp_path / "usage.jsonl"
        owner = StreamUsageOwner(
            tenant="tenant-a",
            model="model-a",
            prompt="unstarted prompt",
            ledger=UsageLedger(ledger_path),
        )

        owner.finalize()

        assert not ledger_path.exists()

    @pytest.mark.parametrize(
        ("with_ledger", "with_limiter"),
        [(True, True), (True, False), (False, True), (False, False)],
    )
    def test_explicit_tenant_metering_keeps_optional_sinks_independent(
        self, tmp_path, with_ledger, with_limiter
    ):
        from kairyu.entrypoints.server.metering import record_tenant_usage

        class RecordingLimiter:
            def __init__(self):
                self.charges = []

            def charge_tokens(self, tenant, tokens):
                self.charges.append((tenant, tokens))

        ledger_path = tmp_path / "usage.jsonl"
        ledger = UsageLedger(ledger_path) if with_ledger else None
        limiter = RecordingLimiter() if with_limiter else None

        record_tenant_usage(
            tenant="tenant-a",
            model="model-a",
            prompt_tokens=7,
            completion_tokens=5,
            ledger=ledger,
            limiter=limiter,
        )

        if ledger is not None:
            assert ledger.totals()["tenant-a"] == {
                "requests": 1,
                "prompt_tokens": 7,
                "completion_tokens": 5,
            }
        else:
            assert not ledger_path.exists()
        if limiter is not None:
            assert limiter.charges == [("tenant-a", 12)]

    def test_usage_counts_prefer_backend_and_openai_usage(self):
        from kairyu.engine.backend import GenerationUsage
        from kairyu.entrypoints.server.metering import resolve_usage_counts
        from kairyu.entrypoints.server.protocol import Usage

        assert resolve_usage_counts(
            GenerationUsage(prompt_tokens=7, completion_tokens=5),
            prompt="ignored prompt",
            completions=(),
        ) == (7, 5)
        assert resolve_usage_counts(
            Usage(prompt_tokens=11, completion_tokens=3, total_tokens=14),
            prompt="ignored prompt",
            completions=(),
        ) == (11, 3)

    def test_usage_counts_derive_multiple_choices_with_wire_approximation(self):
        from kairyu.entrypoints.server.app import _wire_usage
        from kairyu.entrypoints.server.metering import resolve_usage_counts
        from kairyu.outputs import CompletionOutput

        completions = (
            CompletionOutput(index=0, text="ignored text", token_ids=(101, 102)),
            CompletionOutput(index=1, text="three more words", token_ids=()),
        )

        counts = resolve_usage_counts(
            None,
            prompt="rendered prompt words",
            completions=completions,
        )
        wire = _wire_usage("rendered prompt words", completions, None)

        assert counts == (3, 5)
        assert (wire.prompt_tokens, wire.completion_tokens) == counts

    def test_config_repr_excludes_api_key_mapping(self):
        api_secret = "tenant-config-api-secret"
        config = TenantConfig(key_tenants={api_secret: "tenant-a"})

        assert config.tenant_for_key(api_secret) == "tenant-a"
        assert api_secret not in repr(config)
        assert "key_tenants" not in repr(config)

    def test_from_mapping_builds_distinct_tenants_and_copies_inputs(self):
        key_tenants = {"key-a": "tenant-a", "key-b": "tenant-b"}
        limits = {
            "tenant-a": TenantLimits(
                requests_per_minute=10, tokens_per_minute=1_000
            ),
            "tenant-b": TenantLimits(
                requests_per_minute=20, tokens_per_minute=2_000
            ),
            "default": TenantLimits(
                requests_per_minute=30, tokens_per_minute=3_000
            ),
        }

        config = TenantConfig.from_mapping(
            key_tenants=key_tenants,
            limits=limits,
            default_tenant="default",
            resolved_api_keys=frozenset({"key-a", "key-b"}),
        )

        assert config.tenant_for_key("key-a") == "tenant-a"
        assert config.tenant_for_key("key-b") == "tenant-b"
        assert config.limits_for("tenant-a") == limits["tenant-a"]
        assert config.limits_for("tenant-b") == limits["tenant-b"]
        assert config.limits_for("default") == limits["default"]
        key_tenants["key-a"] = "changed"
        limits.clear()
        assert config.tenant_for_key("key-a") == "tenant-a"
        assert config.limits_for("tenant-a") == TenantLimits(
            requests_per_minute=10, tokens_per_minute=1_000
        )

    def test_from_mapping_rejects_empty_key(self):
        with pytest.raises(ValueError, match="mapping key must not be empty"):
            TenantConfig.from_mapping(
                key_tenants={"": "tenant-a"},
                resolved_api_keys=frozenset({"key-a"}),
            )

    def test_from_mapping_rejects_empty_tenant_name(self):
        with pytest.raises(ValueError, match="tenant name must not be empty"):
            TenantConfig.from_mapping(
                key_tenants={"key-a": ""},
                resolved_api_keys=frozenset({"key-a"}),
            )

    def test_from_mapping_rejects_empty_default_tenant(self):
        with pytest.raises(ValueError, match="default tenant must not be empty"):
            TenantConfig.from_mapping(
                key_tenants={},
                default_tenant="",
                resolved_api_keys=frozenset(),
            )

    def test_from_mapping_rejects_key_outside_resolved_api_keys(self):
        with pytest.raises(ValueError, match="unknown API key 'key-b'") as exc_info:
            TenantConfig.from_mapping(
                key_tenants={"key-b": "tenant-b"},
                resolved_api_keys=frozenset({"valid-secret"}),
            )
        assert "valid-secret" not in str(exc_info.value)

    def test_from_mapping_unmapped_resolved_key_uses_default_tenant(self):
        config = TenantConfig.from_mapping(
            key_tenants={"key-a": "tenant-a"},
            default_tenant="fallback",
            resolved_api_keys=frozenset({"key-a", "unmapped-key"}),
        )

        assert config.tenant_for_key("unmapped-key") == "fallback"

    def test_from_mapping_rejects_raw_string_resolved_keys(self):
        with pytest.raises(ValueError, match="must not be a string"):
            TenantConfig.from_mapping(
                key_tenants={"key-a": "tenant-a"},
                resolved_api_keys="key-a",
            )

    def test_from_mapping_allows_multiple_keys_for_one_tenant(self):
        config = TenantConfig.from_mapping(
            key_tenants={"key-a": "shared", "key-b": "shared"},
            limits={"shared": TenantLimits(requests_per_minute=12)},
            resolved_api_keys=frozenset({"key-a", "key-b"}),
        )

        assert config.tenant_for_key("key-a") == "shared"
        assert config.tenant_for_key("key-b") == "shared"
        assert config.limits_for("shared").requests_per_minute == 12

    def test_from_mapping_rejects_orphan_limit_tenant(self):
        with pytest.raises(ValueError, match="limits reference unknown tenant 'orphan'"):
            TenantConfig.from_mapping(
                key_tenants={"key-a": "tenant-a"},
                limits={"orphan": TenantLimits(requests_per_minute=12)},
                default_tenant="default",
                resolved_api_keys=frozenset({"key-a"}),
            )

    def test_admin_only_usage_is_not_mapped_to_default_tenant(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("KAIRYU_DATA_KEYS", "data")
        monkeypatch.setenv("KAIRYU_ADMIN_KEYS", "admin")
        app = create_app(
            {"m": MockBackend()},
            settings=ServerSettings(
                api_keys_env="KAIRYU_DATA_KEYS",
                admin_keys_env="KAIRYU_ADMIN_KEYS",
                usage_ledger_path=str(tmp_path / "usage.jsonl"),
            ),
            tenant_config=TenantConfig(key_tenants={"data": "tenant-a"}),
        )
        payload = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }
        with TestClient(app) as client:
            data = {"Authorization": "Bearer data"}
            admin = {"Authorization": "Bearer admin"}
            assert (
                client.post("/v1/chat/completions", json=payload, headers=data).status_code
                == 200
            )
            admin_usage = client.get("/admin/usage", headers=admin)

        assert admin_usage.status_code == 200
        assert "tenant-a" in admin_usage.json()["usage"]
        assert set(app.state.tenant_limiter._buckets) == {"tenant-a"}

    def test_rate_isolation_and_ledger(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAIRYU_M11_KEYS", "key-a,key-b")
        config = TenantConfig(
            key_tenants={"key-a": "tenant-a", "key-b": "tenant-b"},
            limits={"tenant-a": TenantLimits(requests_per_minute=2)},
        )
        engine = create_backend("mock")
        app = create_app(
            {"m": engine},
            settings=ServerSettings(
                api_keys_env="KAIRYU_M11_KEYS",
                usage_ledger_path=str(tmp_path / "usage.jsonl"),
            ),
            tenant_config=config,
        )
        payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        with TestClient(app) as client:
            a = {"Authorization": "Bearer key-a"}
            b = {"Authorization": "Bearer key-b"}
            assert client.post("/v1/chat/completions", json=payload, headers=a).status_code == 200
            assert client.post("/v1/chat/completions", json=payload, headers=a).status_code == 200
            third = client.post("/v1/chat/completions", json=payload, headers=a)
            assert third.status_code == 429  # tenant-a exhausted
            assert third.json()["error"]["code"] == "tenant_rate_limited"
            assert client.post("/v1/chat/completions", json=payload, headers=b).status_code == 200

            # unauthenticated: 401 wins, bucket untouched (A6)
            assert client.post("/v1/chat/completions", json=payload).status_code == 401

            # security review: usage is scoped to the CALLER's tenant
            usage_a = client.get("/admin/usage", headers=a).json()["usage"]
            assert usage_a["tenant-a"]["requests"] == 2
            assert "tenant-b" not in usage_a  # no cross-tenant disclosure
            assert usage_a["tenant-a"]["completion_tokens"] > 0
            forbidden = client.get("/admin/usage?tenant=tenant-b", headers=a)
            assert forbidden.status_code == 403
            usage_b = client.get("/admin/usage", headers=b).json()["usage"]
            assert usage_b["tenant-b"]["requests"] == 1

    def test_streaming_and_completions_are_metered(self, tmp_path):
        # S3: streaming chat and /v1/completions were never written to the
        # ledger (billing bypass). Both must now record usage.
        ledger_path = tmp_path / "usage.jsonl"
        app = create_app(
            {"m": create_backend("mock")},
            settings=ServerSettings(usage_ledger_path=str(ledger_path)),
        )
        with TestClient(app) as client:
            with client.stream(
                "POST", "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            ) as response:
                for _ in response.iter_lines():
                    pass
            client.post("/v1/completions", json={"model": "m", "prompt": "hello"})
        from kairyu.entrypoints.server.tenancy import UsageLedger

        totals = UsageLedger(ledger_path).totals()["default"]
        assert totals["requests"] == 2  # both the stream and the completion metered
        assert totals["completion_tokens"] > 0

    def test_ledger_reconciles_with_returned_usage(self, tmp_path):
        ledger = UsageLedger(tmp_path / "ledger.jsonl")
        returned = []
        for i in range(20):
            ledger.record("t", "m", prompt_tokens=10 + i, completion_tokens=5 + i)
            returned.append((10 + i, 5 + i))
        totals = ledger.totals()["t"]
        assert totals["prompt_tokens"] == sum(p for p, _ in returned)  # exact (< 0.1%)
        assert totals["completion_tokens"] == sum(c for _, c in returned)

    def test_bucket_refills(self):
        clock = {"t": 0.0}
        limiter = TenantLimiter(
            TenantConfig(limits={"t": TenantLimits(requests_per_minute=60)}),
            now=lambda: clock["t"],
        )
        for _ in range(60):
            assert limiter.admit("t")
        assert not limiter.admit("t")
        clock["t"] = 2.0  # 2 s -> 2 tokens refilled
        assert limiter.admit("t")
        assert limiter.admit("t")
        assert not limiter.admit("t")

    def test_token_budget_is_enforced(self):
        # S4: a tenant that burns its per-minute token budget is refused the next
        # request, even while its request-rate bucket still has room.
        clock = {"t": 0.0}
        limiter = TenantLimiter(
            TenantConfig(
                limits={"t": TenantLimits(requests_per_minute=600, tokens_per_minute=100)}
            ),
            now=lambda: clock["t"],
        )
        assert limiter.admit("t")
        limiter.charge_tokens("t", 150)  # overspend the 100-token budget
        assert not limiter.admit("t")  # refused despite request-rate room
        clock["t"] = 60.0  # a full minute refills the token bucket
        assert limiter.admit("t")


class TestResponseStore:
    def test_lru_cap_and_tenant_scope(self):
        # M2: the in-memory store is LRU-capped and tenant-scoped — a leaked id
        # from another tenant reads as not-found.
        from kairyu.entrypoints.server.extra_routes import ResponseStore

        store = ResponseStore(max_items=2)
        store.save("r1", [{"a": 1}], owner="tenant-a")
        assert store.get("r1", owner="tenant-a") == [{"a": 1}]
        assert store.get("r1", owner="tenant-b") is None  # cross-tenant -> not found
        store.save("r2", [{"b": 2}], owner="tenant-a")
        store.save("r3", [{"c": 3}], owner="tenant-a")  # evicts the LRU (r1)
        assert store.get("r1", owner="tenant-a") is None  # evicted


class TestResponsesApi:
    def test_responses_and_embeddings_meter_authenticated_tenant_with_wire_counts(
        self, tmp_path, monkeypatch
    ):
        from dataclasses import replace

        from kairyu.engine.backend import GenerationUsage

        class ReportedUsageBackend(MockBackend):
            async def generate(self, request):
                result = await super().generate(request)
                return replace(
                    result,
                    usage=GenerationUsage(prompt_tokens=17, completion_tokens=9),
                )

        class DerivedUsageBackend(MockBackend):
            async def generate(self, request):
                return replace(await super().generate(request), usage=None)

        monkeypatch.setenv("KAIRYU_EXTRA_ROUTE_KEYS", "key-a")
        ledger_path = tmp_path / "usage.jsonl"
        app = create_app(
            {"reported": ReportedUsageBackend(), "derived": DerivedUsageBackend()},
            settings=ServerSettings(
                api_keys_env="KAIRYU_EXTRA_ROUTE_KEYS",
                usage_ledger_path=str(ledger_path),
            ),
            tenant_config=TenantConfig(key_tenants={"key-a": "tenant-a"}),
            embedding_backend=MockEmbeddingBackend(dimensions=8),
        )
        headers = {"Authorization": "Bearer key-a"}

        with TestClient(app) as client:
            reported = client.post(
                "/v1/responses",
                headers=headers,
                json={"model": "reported", "input": "reported input"},
            )
            derived = client.post(
                "/v1/responses",
                headers=headers,
                json={"model": "derived", "input": "derived input"},
            )
            embedding = client.post(
                "/v1/embeddings",
                headers=headers,
                json={
                    "model": "embedding-model",
                    "input": ["two words", "one"],
                    "encoding_format": "float",
                },
            )

        assert reported.status_code == 200
        assert reported.json()["usage"] == {
            "input_tokens": 17,
            "output_tokens": 9,
            "total_tokens": 26,
        }
        assert derived.status_code == 200
        derived_usage = derived.json()["usage"]
        assert derived_usage["input_tokens"] > 0
        assert derived_usage["output_tokens"] > 0
        assert embedding.status_code == 200
        assert embedding.json()["usage"] == {"prompt_tokens": 3, "total_tokens": 3}

        totals = UsageLedger(ledger_path).totals()
        assert set(totals) == {"tenant-a"}
        assert totals["tenant-a"] == {
            "requests": 3,
            "prompt_tokens": 17 + derived_usage["input_tokens"] + 3,
            "completion_tokens": 9 + derived_usage["output_tokens"],
        }

    def test_extra_route_failures_before_usable_results_are_unmetered(
        self, tmp_path, monkeypatch
    ):
        class FailingBackend(MockBackend):
            async def generate(self, request):
                raise RuntimeError("backend unavailable")

        class FailingEmbeddingBackend(MockEmbeddingBackend):
            async def embed(self, texts):
                raise RuntimeError("embedding backend unavailable")

        monkeypatch.setenv("KAIRYU_EXTRA_ROUTE_KEYS", "key-a")
        ledger_path = tmp_path / "usage.jsonl"
        app = create_app(
            {"m": FailingBackend()},
            settings=ServerSettings(
                api_keys_env="KAIRYU_EXTRA_ROUTE_KEYS",
                usage_ledger_path=str(ledger_path),
            ),
            tenant_config=TenantConfig(key_tenants={"key-a": "tenant-a"}),
            embedding_backend=FailingEmbeddingBackend(dimensions=8),
        )
        headers = {"Authorization": "Bearer key-a"}

        with TestClient(app, raise_server_exceptions=False) as client:
            invalid_response = client.post(
                "/v1/responses",
                headers=headers,
                json={"model": "m", "input": "x", "stream": True},
            )
            failed_response = client.post(
                "/v1/responses",
                headers=headers,
                json={"model": "m", "input": "x"},
            )
            invalid_embedding = client.post(
                "/v1/embeddings",
                headers=headers,
                json={"model": "m", "input": []},
            )
            failed_embedding = client.post(
                "/v1/embeddings",
                headers=headers,
                json={"model": "m", "input": "x"},
            )

        assert invalid_response.status_code == 400
        assert failed_response.status_code == 502
        assert invalid_embedding.status_code == 400
        assert failed_embedding.status_code == 500
        assert not ledger_path.exists()

    def test_sdk_round_trip_with_previous_response_id(self, tmp_path):
        import openai

        app = _auto_app(tmp_path)
        with TestClient(app) as http:
            client = openai.OpenAI(
                base_url=str(http.base_url) + "/v1",
                api_key="sk-local",
                http_client=http,
            )
            first = client.responses.create(model="m", input="hello")
            assert first.status == "completed"
            assert first.output_text  # computed from the exact item shapes (A8)
            assert first.usage.input_tokens >= 0

            second = client.responses.create(
                model="m", input="and again", previous_response_id=first.id,
                instructions="be brief",
            )
            assert second.output_text

    def test_unknown_previous_id_404(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "m", "input": "x", "previous_response_id": "resp_nope"},
            )
            assert response.status_code == 404

    def test_stream_descoped_cleanly(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            response = client.post(
                "/v1/responses", json={"model": "m", "input": "x", "stream": True}
            )
            assert response.status_code == 400


class TestEmbeddings:
    def test_sdk_round_trip_base64_default(self, tmp_path):
        import openai

        with TestClient(_auto_app(tmp_path)) as http:
            client = openai.OpenAI(
                base_url=str(http.base_url) + "/v1", api_key="sk-local",
                http_client=http,
            )
            result = client.embeddings.create(model="m", input=["hello", "world"])
            assert len(result.data) == 2
            assert len(result.data[0].embedding) == 8  # SDK decodes base64 (A9)
            assert result.usage.prompt_tokens > 0

    def test_float_and_base64_agree(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            as_float = client.post(
                "/v1/embeddings",
                json={"model": "m", "input": "hello", "encoding_format": "float"},
            ).json()["data"][0]["embedding"]
            as_b64 = client.post(
                "/v1/embeddings",
                json={"model": "m", "input": "hello", "encoding_format": "base64"},
            ).json()["data"][0]["embedding"]
            decoded = struct.unpack(
                f"<{len(as_float)}f", base64.b64decode(as_b64)
            )
            assert list(decoded) == pytest.approx(as_float)

    def test_invalid_encoding_format_is_400(self, tmp_path):
        # M6: an unknown encoding_format (e.g. the typo "Base64") must be a 400,
        # not silently served as float.
        with TestClient(_auto_app(tmp_path)) as client:
            resp = client.post(
                "/v1/embeddings",
                json={"model": "m", "input": "hello", "encoding_format": "Base64"},
            )
            assert resp.status_code == 400


class TestVisionWire:
    def test_content_parts_accepted_and_flattened(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "text": "hello parts"}],
                }]},
            )
            assert response.status_code == 200

    def test_image_parts_rejected_cleanly(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look:"},
                        {"type": "image_url", "image_url": {"url": "http://x/i.png"}},
                    ],
                }]},
            )
            assert response.status_code == 400
            assert "image" in response.json()["error"]["message"]


class TestF5Logic:
    def test_priority_admission_orders_and_ages(self):
        from kairyu.engine.core.radix_kv import RadixKVCache
        from kairyu.engine.core.scheduler import EngineRequest, Scheduler

        clock = {"t": 0.0}
        cache = RadixKVCache(num_pages=64, page_size=4)
        scheduler = Scheduler(
            cache, max_num_batched_tokens=8, page_size=4,
            priority_age_s=10.0, clock=lambda: clock["t"],
        )
        scheduler.add_request(EngineRequest("low", (1, 2, 3, 4), max_new_tokens=1, priority=0))
        clock["t"] = 1.0
        scheduler.add_request(EngineRequest("high", (5, 6, 7, 8), max_new_tokens=1, priority=5))
        plan = scheduler.schedule()
        assert plan.scheduled[0].request_id == "high"  # priority beats FIFO

        # aging: a very old low-priority request overtakes a fresh mid one
        scheduler2 = Scheduler(
            cache, max_num_batched_tokens=4, page_size=4,
            priority_age_s=1.0, clock=lambda: clock["t"],
        )
        clock["t"] = 0.0
        scheduler2.add_request(
            EngineRequest("old-low", (11, 12, 13, 14), max_new_tokens=1, priority=0)
        )
        clock["t"] = 10.0
        scheduler2.add_request(
            EngineRequest("new-mid", (15, 16, 17, 18), max_new_tokens=1, priority=3)
        )
        plan = scheduler2.schedule()
        assert plan.scheduled[0].request_id == "old-low"  # aged 10 > priority 3

    def test_admission_controller_shed_and_defer(self):
        from kairyu.entrypoints.server.slo import AdmissionController

        clock = {"t": 0.0}
        controller = AdmissionController(
            ttft_slo_s=0.1, defer_threshold_s=0.3, now=lambda: clock["t"]
        )
        # feed slow observations to raise the EMA
        for _ in range(20):
            started = controller.started()
            clock["t"] += 0.2
            controller.finished_first_token(started)
            controller.completed()
        assert controller.decide().action in ("defer", "shed")
        # pile on in-flight -> shed
        for _ in range(10):
            controller.started()
        assert controller.decide().action == "shed"

    def test_autoscale_hysteresis_table(self):
        from kairyu.entrypoints.server.slo import autoscale_decision

        assert autoscale_decision([0.9, 0.95, 0.9], queue_depth=4).action == "scale_up"
        assert autoscale_decision([0.1, 0.2, 0.1], queue_depth=0).action == "scale_down"
        assert autoscale_decision([0.5, 0.6, 0.5], queue_depth=0).action == "hold"
        assert autoscale_decision([0.9], queue_depth=9).action == "hold"  # window
        assert autoscale_decision([0.9, 0.9, 0.9], queue_depth=0).action == "hold"


def test_frontier_scoreboard_schema():
    from bench.frontier_compare import TargetReport, TrialResult, build_scoreboard

    report = TargetReport(name="kairyu", model="m")
    report.trials.append(TrialResult(ttft_s=0.05, tpot_s=0.01, output_chars=100))
    scoreboard = build_scoreboard([report])
    assert scoreboard["methodology"]["metric_definitions"]["ttft"]
    assert scoreboard["results"][0]["ttft_p50_s"] == 0.05
