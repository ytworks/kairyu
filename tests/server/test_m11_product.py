"""m11 gates: streaming orchestrator usage, tiers, tenancy, responses,
embeddings, vision wire, F5 logic, bench schema."""

import base64
import contextlib
import json
import logging
import struct
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from kairyu.batch.store import BatchStore
from kairyu.engine.backend import GenerationResult, GenerationUsage
from kairyu.engine.embedding import EmbeddingResult
from kairyu.engine.mock import MockBackend
from kairyu.engine.registry import create_backend
from kairyu.entrypoints.server.extra_routes import MockEmbeddingBackend
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import (
    TenantConfig,
    TenantLimiter,
    TenantLimits,
    UsageLedger,
)
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.outputs import CompletionOutput
from tests.server._legacy_chat import LegacyBatchWorker, create_legacy_app


def _auto_app(tmp_path, **kwargs):
    engine = create_backend("mock")
    orchestrator = Orchestrator({"tier1": engine, "tier2": engine})
    deep = Orchestrator({"tier1": engine, "tier2": engine}, moa_samples=2)
    return create_legacy_app(
        {"m": engine},
        orchestrators={"kairyu-auto": orchestrator, "kairyu-auto-max": deep},
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
        embedding_backends={"embedding-model": MockEmbeddingBackend(dimensions=8)},
        **kwargs,
    )


def test_mixed_tool_choice_rejection_is_metered_once(tmp_path):
    class MixedToolBackend(MockBackend):
        async def generate(self, request):
            tool_call = '<tool_call>{"name":"get_weather","arguments":{}}</tool_call>'
            return GenerationResult(
                request_id=request.request_id,
                prompt=request.prompt,
                completions=(
                    CompletionOutput(index=0, text=tool_call, token_ids=()),
                    CompletionOutput(index=1, text="plain", token_ids=()),
                ),
                usage=GenerationUsage(prompt_tokens=19, completion_tokens=7),
            )

    ledger_path = tmp_path / "usage.jsonl"
    app = create_legacy_app(
        {"m": MixedToolBackend()},
        settings=ServerSettings(usage_ledger_path=str(ledger_path)),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "weather"}],
                "n": 2,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": {}},
                    }
                ],
                "tool_choice": "required",
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "tool_choice_not_satisfied"
    assert UsageLedger(ledger_path).totals()["default"] == {
        "requests": 1,
        "prompt_tokens": 19,
        "completion_tokens": 7,
        "cached_tokens": 0,
        "uncached_tokens": 19,
    }


class TestOrchestratorSurface:
    def test_tiers_listed_and_auto_usage_real(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            models = [m["id"] for m in client.get("/v1/models").json()["data"]]
            assert "kairyu-auto" in models and "kairyu-auto-max" in models

            response = client.post(
                "/v1/chat/completions",
                json={"model": "kairyu-auto", "messages": [{"role": "user", "content": "hello"}]},
            )
            assert response.status_code == 200
            payload = response.json()
            usage = payload["usage"]
            assert usage["completion_tokens"] > 0  # m11 A1: real, not zero
            assert "kairyu_trace" not in payload
            assert "kairyu_trace_v2" not in payload
            assert "kairyu_route" not in payload

    def test_auto_model_accepts_orchestration_params(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            for extra in (
                {"n": 2},
                {"logprobs": True},
                {"tools": [{"type": "function", "function": {"name": "f"}}]},
                {"response_format": {"type": "json_object"}},
            ):
                resp = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "kairyu-auto",
                        "messages": [{"role": "user", "content": "hi"}],
                        **extra,
                    },
                )
                assert resp.status_code == 200, extra
                if extra.get("n"):
                    assert [choice["index"] for choice in resp.json()["choices"]] == [
                        0,
                        1,
                    ]

    def test_trace_header_opt_in(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"X-Kairyu-Trace": "1"},
                json={"model": "kairyu-auto", "messages": [{"role": "user", "content": "hello"}]},
            )
            trace = response.json().get("kairyu_trace")
            assert trace and any("route:" in line for line in trace)
            trace_v2 = response.json().get("kairyu_trace_v2")
            assert trace_v2["trace_version"] == "2.0"
            assert trace_v2["request_id"] == response.json()["id"]
            assert trace_v2["events"][0]["kind"] == "routing"
            assert trace_v2["events"][0]["seq"] == 1
            assert trace_v2["events"][-1]["kind"] == "generation"
            assert trace_v2["events"][-1]["timing"]["completed_at"]
            assert "hello" not in str(trace_v2)
            assert response.json()["kairyu_route"]["target"] == "tier1"

    def test_structured_trace_is_declared_in_openapi(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            schemas = client.get("/openapi.json").json()["components"]["schemas"]
        response_schema = schemas["ChatCompletionResponse"]
        assert "kairyu_trace_v2" in response_schema["properties"]
        assert "kairyu_route" in response_schema["properties"]
        assert "KairyuTraceV2" in schemas
        assert "RouteDecisionPayload" in schemas

    def test_auto_stream_chunks_and_usage(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={"X-Kairyu-Trace": "1"},
                json={
                    "model": "kairyu-auto",
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": "hello"}],
                },
            ) as response:
                body = "".join(response.iter_text())
        assert "data: [DONE]" in body
        assert ": trace route:" in body
        import json as _json

        data_lines = [
            line[len("data: ") :]
            for line in body.splitlines()
            if line.startswith("data: ") and "[DONE]" not in line
        ]
        chunks = [_json.loads(line) for line in data_lines]
        assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
        usage_chunks = [c for c in chunks if c.get("usage")]
        assert usage_chunks and usage_chunks[-1]["usage"]["completion_tokens"] > 0

    def test_auto_partial_stream_failure_finalizes_usage_once(self, tmp_path):
        class PartialFailureBackend(MockBackend):
            def __init__(self):
                super().__init__()
                self.closed = False

            async def stream(self, request):
                try:
                    yield GenerationResult(
                        request_id=request.request_id,
                        prompt=request.prompt,
                        completions=(
                            CompletionOutput(
                                index=0,
                                text="partial answer",
                                token_ids=(1, 2),
                            ),
                        ),
                        finished=False,
                    )
                    raise RuntimeError("backend failed after first delta")
                finally:
                    self.closed = True

        backend = PartialFailureBackend()
        ledger_path = tmp_path / "usage.jsonl"
        app = create_legacy_app(
            {},
            orchestrators={"kairyu-auto": Orchestrator({"tier1": backend, "tier2": backend})},
            settings=ServerSettings(usage_ledger_path=str(ledger_path)),
        )

        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "kairyu-auto",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            ) as response:
                body = "".join(response.iter_text())

        assert "partial answer" in body
        assert '"error"' in body
        assert "data: [DONE]" in body
        assert backend.closed is True
        totals = UsageLedger(ledger_path).totals()["default"]
        assert totals["requests"] == 1
        assert totals["completion_tokens"] > 0

    def test_moa_tier_synthesizes(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"X-Kairyu-Trace": "1"},
                json={
                    "model": "kairyu-auto-max",
                    "messages": [
                        {"role": "user", "content": "analyze compare and plan: " + "x" * 2500}
                    ],
                },
            )
            trace = response.json().get("kairyu_trace") or []
            assert any("moa" in line for line in trace), trace


class TestTenancy:
    @pytest.mark.parametrize(
        ("surface", "model", "stream"),
        [
            pytest.param("chat", "m", False, id="sync-chat"),
            pytest.param("chat", "m", True, id="stream-chat"),
            pytest.param("completions", "m", False, id="sync-completions"),
            pytest.param("completions", "m", True, id="stream-completions"),
            pytest.param("chat", "kairyu-auto", False, id="sync-orchestrator"),
            pytest.param("chat", "kairyu-auto", True, id="stream-orchestrator"),
            pytest.param("responses", "m", False, id="responses"),
            pytest.param("embeddings", "embedding-model", False, id="embeddings"),
            pytest.param("batch", "m", False, id="batch"),
        ],
    )
    async def test_successful_usage_endpoint_matrix_records_exactly_once(
        self, tmp_path, surface, model, stream
    ):
        class RecordingLimiter:
            def __init__(self):
                self.charges = []

            def charge_tokens(self, tenant, tokens):
                self.charges.append((tenant, tokens))

        app = _auto_app(tmp_path)
        limiter = RecordingLimiter()
        app.state.tenant_limiter = limiter

        if surface == "batch":
            store = BatchStore(tmp_path / "batch")
            worker = LegacyBatchWorker(
                store,
                {"m": MockBackend()},
                max_concurrency=1,
                metrics=app.state.metrics,
                usage_ledger=app.state.usage_ledger,
                tenant_limiter=limiter,
            )
            content = json.dumps(
                {
                    "custom_id": "matrix-batch",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [{"role": "user", "content": "matrix prompt"}],
                    },
                }
            ).encode()
            file = store.save_file(content, "matrix.jsonl", "batch")
            job = store.create_batch(file.id, "/v1/chat/completions")

            await worker.process(job.id)

            finished = store.get_batch(job.id)
            assert finished.status == "completed"
            assert finished.request_counts.completed == 1
        else:
            if surface == "chat":
                path = "/v1/chat/completions"
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": "matrix prompt"}],
                    "stream": stream,
                }
            elif surface == "completions":
                path = "/v1/completions"
                body = {"model": model, "prompt": "matrix prompt", "stream": stream}
            elif surface == "responses":
                path = "/v1/responses"
                body = {"model": model, "input": "matrix prompt"}
            else:
                path = "/v1/embeddings"
                body = {"model": model, "input": "matrix prompt"}

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(path, json=body)

            assert response.status_code == 200
            if stream:
                assert "data: [DONE]" in response.text

        totals = app.state.usage_ledger.totals()["default"]
        assert totals["requests"] == 1
        rendered_metrics = app.state.metrics.render()[0].decode()
        assert 'kairyu_usage_requests_total{tenant="default"} 1.0' in rendered_metrics
        assert (
            'kairyu_usage_tokens_total{tenant="default",type="prompt"} '
            f"{float(totals['prompt_tokens'])}"
        ) in rendered_metrics
        assert (
            'kairyu_usage_tokens_total{tenant="default",type="completion"} '
            f"{float(totals['completion_tokens'])}"
        ) in rendered_metrics
        assert limiter.charges == [
            (
                "default",
                totals["prompt_tokens"] + totals["completion_tokens"],
            )
        ]

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
            "cached_tokens": 0,
            "uncached_tokens": 7,
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
                "cached_tokens": 0,
                "uncached_tokens": 7,
            }
        else:
            assert not ledger_path.exists()
        if limiter is not None:
            assert limiter.charges == [("tenant-a", 12)]

    def test_quota_charge_survives_ledger_admission_failure(self):
        from kairyu.entrypoints.server.metering import record_tenant_usage

        class FailingLedger:
            def record(self, *args, **kwargs):
                raise RuntimeError("ledger full")

        class RecordingLimiter:
            def __init__(self):
                self.charges = []

            def charge_tokens(self, tenant, tokens):
                self.charges.append((tenant, tokens))

        limiter = RecordingLimiter()
        with pytest.raises(RuntimeError, match="ledger full"):
            record_tenant_usage(
                tenant="tenant-a",
                model="model-a",
                prompt_tokens=7,
                completion_tokens=5,
                ledger=FailingLedger(),
                limiter=limiter,
            )

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
            "tenant-a": TenantLimits(requests_per_minute=10, tokens_per_minute=1_000),
            "tenant-b": TenantLimits(requests_per_minute=20, tokens_per_minute=2_000),
            "default": TenantLimits(requests_per_minute=30, tokens_per_minute=3_000),
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

    def test_admin_only_usage_is_not_mapped_to_default_tenant(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAIRYU_DATA_KEYS", "data")
        monkeypatch.setenv("KAIRYU_ADMIN_KEYS", "admin")
        app = create_legacy_app(
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
                client.post("/v1/chat/completions", json=payload, headers=data).status_code == 200
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
        app = create_legacy_app(
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
        app = create_legacy_app(
            {"m": create_backend("mock")},
            settings=ServerSettings(usage_ledger_path=str(ledger_path)),
        )
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
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

    def test_ledger_rejects_negative_usage_before_sink_updates(self, tmp_path):
        ledger = UsageLedger(tmp_path / "ledger.jsonl")

        with pytest.raises(ValueError, match="non-negative integers"):
            ledger.record(
                "tenant-a",
                "m",
                prompt_tokens=-1,
                completion_tokens=2,
            )

        assert not (tmp_path / "ledger.jsonl").exists()

    def test_ledger_skips_truncated_tail_and_warns_once(self, tmp_path, caplog):
        ledger_path = tmp_path / "ledger.jsonl"
        ledger = UsageLedger(ledger_path)
        ledger.record("tenant-a", "m", prompt_tokens=7, completion_tokens=3)
        ledger.flush()
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write('{"tenant":"tenant-a","prompt_tokens":11')

        with caplog.at_level(
            logging.WARNING,
            logger="kairyu.entrypoints.server.tenancy",
        ):
            totals = ledger.totals()

        assert totals == {
            "tenant-a": {
                "requests": 1,
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "cached_tokens": 0,
                "uncached_tokens": 7,
            }
        }
        assert ledger.malformed_lines == 1
        warnings = [
            record for record in caplog.records if "truncated usage ledger tail" in record.message
        ]
        assert len(warnings) == 1

    def test_ledger_skips_complete_corruption_and_whitespace(self, tmp_path, caplog):
        ledger_path = tmp_path / "ledger.jsonl"
        lines = [
            json.dumps(
                {
                    "tenant": "tenant-a",
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                }
            ),
            "not-json",
            json.dumps(
                {
                    "tenant": "tenant-a",
                    "prompt_tokens": "4",
                    "completion_tokens": 5,
                }
            ),
            json.dumps({"prompt_tokens": 6, "completion_tokens": 7}),
            json.dumps(
                {
                    "tenant": "tenant-a",
                    "prompt_tokens": -1,
                    "completion_tokens": 7,
                }
            ),
            "   ",
            json.dumps(
                {
                    "tenant": "tenant-a",
                    "prompt_tokens": 11,
                    "completion_tokens": 13,
                }
            ),
        ]
        ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ledger = UsageLedger(ledger_path)

        with caplog.at_level(
            logging.ERROR,
            logger="kairyu.entrypoints.server.tenancy",
        ):
            totals = ledger.totals()

        assert totals == {
            "tenant-a": {
                "requests": 2,
                "prompt_tokens": 13,
                "completion_tokens": 16,
                "cached_tokens": 0,
                "uncached_tokens": 13,
            }
        }
        assert ledger.malformed_lines == 4
        errors = [
            record.message
            for record in caplog.records
            if "malformed usage ledger record" in record.message
        ]
        assert len(errors) == 4
        assert all(f"line {line_number}" in " ".join(errors) for line_number in (2, 3, 4, 5))

    def test_admin_usage_returns_partial_totals_for_corrupt_ledger(self, tmp_path, caplog):
        ledger_path = tmp_path / "ledger.jsonl"
        ledger_path.write_text(
            "\n".join(
                (
                    json.dumps(
                        {
                            "tenant": "tenant-a",
                            "prompt_tokens": 5,
                            "completion_tokens": 8,
                        }
                    ),
                    "not-json",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        app = create_legacy_app(
            {"m": MockBackend()},
            settings=ServerSettings(usage_ledger_path=str(ledger_path)),
        )

        with (
            caplog.at_level(
                logging.ERROR,
                logger="kairyu.entrypoints.server.tenancy",
            ),
            TestClient(app) as client,
        ):
            response = client.get("/admin/usage")

        assert response.status_code == 200
        assert response.json() == {
            "usage": {
                "tenant-a": {
                    "requests": 1,
                    "prompt_tokens": 5,
                    "completion_tokens": 8,
                    "cached_tokens": 0,
                    "uncached_tokens": 5,
                }
            }
        }
        assert app.state.usage_ledger.malformed_lines == 1

    def test_create_app_closes_and_can_reopen_ledger_after_shutdown(self, tmp_path):
        app = create_legacy_app(
            {"m": MockBackend()},
            settings=ServerSettings(usage_ledger_path=str(tmp_path / "ledger.jsonl")),
        )
        ledger = app.state.usage_ledger

        with TestClient(app):
            ledger.record("tenant-a", "m", prompt_tokens=1, completion_tokens=2)
            ledger.flush()
            first_handle = ledger._handle
            assert first_handle is not None
            assert not first_handle.closed

        assert first_handle.closed
        ledger.record("tenant-a", "m", prompt_tokens=3, completion_tokens=4)
        ledger.flush()
        assert ledger._handle is not first_handle
        assert not ledger._handle.closed
        ledger.close()

    def test_create_app_closes_ledger_when_caller_lifespan_shutdown_fails(self, tmp_path):
        @contextlib.asynccontextmanager
        async def failing_lifespan(_app):
            yield
            raise RuntimeError("caller shutdown failed")

        app = create_legacy_app(
            {"m": MockBackend()},
            settings=ServerSettings(usage_ledger_path=str(tmp_path / "ledger.jsonl")),
            lifespan=failing_lifespan,
        )
        ledger = app.state.usage_ledger

        with pytest.raises(RuntimeError, match="caller shutdown failed"):
            with TestClient(app):
                ledger.record("tenant-a", "m", prompt_tokens=1, completion_tokens=2)
                ledger.flush()
                handle = ledger._handle

        assert handle is not None
        assert handle.closed

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

    def test_fractional_refill_admits_at_exact_boundary(self):
        clock = {"t": 0.0}
        limiter = TenantLimiter(
            TenantConfig(
                limits={
                    "t": TenantLimits(
                        requests_per_minute=6,
                        request_burst=1,
                    )
                }
            ),
            now=lambda: clock["t"],
        )

        assert limiter.admit("t")
        clock["t"] = 10.0
        assert limiter.admit("t")

    def test_legacy_positional_tenant_limit_priorities_remain_compatible(self):
        limits = TenantLimits(10, 20, -1, 1)

        assert limits.interactive_priority == -1
        assert limits.batch_priority == 1
        assert limits.request_burst is None

    def test_tenant_in_flight_lease_bounds_concurrent_burst(self):
        limiter = TenantLimiter(
            TenantConfig(
                limits={
                    "t": TenantLimits(
                        requests_per_minute=600,
                        request_burst=10,
                        max_in_flight=1,
                    )
                }
            ),
            now=lambda: 0.0,
        )

        first = limiter.acquire("t")
        blocked = limiter.acquire("t")
        assert first.admitted
        assert blocked.reason == "in_flight"
        assert limiter.in_flight("t") == 1

        first.release()
        first.release()
        assert limiter.in_flight("t") == 0
        replacement = limiter.acquire("t")
        assert replacement.admitted
        replacement.release()

    def test_explicit_request_burst_is_independent_of_refill_rate(self):
        clock = {"t": 0.0}
        limiter = TenantLimiter(
            TenantConfig(
                limits={
                    "t": TenantLimits(
                        requests_per_minute=60,
                        request_burst=2,
                    )
                }
            ),
            now=lambda: clock["t"],
        )

        assert limiter.admit("t")
        assert limiter.admit("t")
        assert not limiter.admit("t")
        clock["t"] = 1.0
        assert limiter.admit("t")

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

    def test_token_reservation_rejects_before_dispatch_and_refunds_preflight(self):
        limiter = TenantLimiter(
            TenantConfig(
                limits={
                    "t": TenantLimits(
                        requests_per_minute=600,
                        tokens_per_minute=100,
                        token_burst=100,
                    )
                }
            ),
            now=lambda: 0.0,
        )

        rejected = limiter.acquire("t")
        assert not rejected.reserve_tokens(101)
        rejected.release()
        assert limiter.token_balance("t") == 100
        assert limiter.reservation_snapshot()["t"] == 0

        refunded = limiter.acquire("t")
        assert refunded.reserve_tokens(40)
        refunded.release()
        assert limiter.token_balance("t") == 100
        assert limiter.reservation_snapshot()["t"] == 0

    def test_token_reservation_settles_exact_or_consumes_unknown_work(self):
        limiter = TenantLimiter(
            TenantConfig(
                limits={
                    "t": TenantLimits(
                        requests_per_minute=600,
                        tokens_per_minute=100,
                        token_burst=100,
                    )
                }
            ),
            now=lambda: 0.0,
        )

        exact = limiter.acquire("t")
        assert exact.reserve_tokens(40)
        exact.mark_dispatched()
        exact.settle_tokens(10, exact=True)
        exact.release()
        assert limiter.token_balance("t") == 90

        unknown = limiter.acquire("t")
        assert unknown.reserve_tokens(40)
        unknown.mark_dispatched()
        unknown.settle_tokens(1, exact=False)
        unknown.release()
        assert limiter.token_balance("t") == 50
        assert limiter.reservation_snapshot()["t"] == 0

    def test_dispatched_failure_consumes_full_reservation_without_leak(self):
        limiter = TenantLimiter(
            TenantConfig(
                limits={
                    "t": TenantLimits(
                        requests_per_minute=600,
                        tokens_per_minute=100,
                        token_burst=100,
                    )
                }
            ),
            now=lambda: 0.0,
        )
        admission = limiter.acquire("t")
        assert admission.reserve_tokens(40)
        admission.mark_dispatched()

        admission.release()

        assert limiter.token_balance("t") == 60
        assert limiter.in_flight("t") == 0
        assert limiter.reservation_snapshot()["t"] == 0

    def test_actual_work_over_reservation_is_debited_and_reported(self):
        limiter = TenantLimiter(
            TenantConfig(
                limits={
                    "t": TenantLimits(
                        requests_per_minute=600,
                        tokens_per_minute=100,
                        token_burst=100,
                    )
                }
            ),
            now=lambda: 0.0,
        )
        admission = limiter.acquire("t")
        assert admission.reserve_tokens(20)
        admission.mark_dispatched()

        admission.settle_tokens(30, exact=True)
        admission.release()

        assert limiter.token_balance("t") == 70
        assert limiter.bound_violation_snapshot()["t"] == 1

    @pytest.mark.parametrize(
        ("exact", "refundable"),
        [(False, True), (True, False), (False, False)],
    )
    def test_non_refundable_settlement_still_debits_and_reports_overage(
        self,
        exact,
        refundable,
    ):
        limiter = TenantLimiter(
            TenantConfig(
                limits={
                    "t": TenantLimits(
                        requests_per_minute=600,
                        tokens_per_minute=100,
                        token_burst=100,
                    )
                }
            ),
            now=lambda: 0.0,
        )
        admission = limiter.acquire("t")
        assert admission.reserve_tokens(
            20,
            refundable_on_exact_usage=refundable,
        )
        admission.mark_dispatched()

        admission.settle_tokens(30, exact=exact)
        admission.release()

        assert limiter.token_balance("t") == 70
        assert limiter.bound_violation_snapshot()["t"] == 1
        assert limiter.reservation_snapshot()["t"] == 0


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
                    usage=GenerationUsage(
                        prompt_tokens=17,
                        completion_tokens=9,
                        cached_tokens=11,
                    ),
                )

        class DerivedUsageBackend(MockBackend):
            async def generate(self, request):
                return replace(await super().generate(request), usage=None)

        monkeypatch.setenv("KAIRYU_EXTRA_ROUTE_KEYS", "key-a")
        ledger_path = tmp_path / "usage.jsonl"
        app = create_legacy_app(
            {"reported": ReportedUsageBackend(), "derived": DerivedUsageBackend()},
            settings=ServerSettings(
                api_keys_env="KAIRYU_EXTRA_ROUTE_KEYS",
                usage_ledger_path=str(ledger_path),
            ),
            tenant_config=TenantConfig(key_tenants={"key-a": "tenant-a"}),
            embedding_backends={"embedding-model": MockEmbeddingBackend(dimensions=8)},
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
            "input_tokens_details": {"cached_tokens": 11},
            "output_tokens": 9,
            "output_tokens_details": {"reasoning_tokens": 0},
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
            "cached_tokens": 11,
            "uncached_tokens": derived_usage["input_tokens"] + 9,
        }

    def test_extra_route_failures_have_truthful_dispatch_metering(self, tmp_path, monkeypatch):
        class FailingBackend(MockBackend):
            async def generate(self, request):
                raise RuntimeError("backend unavailable")

        class FailingEmbeddingBackend(MockEmbeddingBackend):
            async def embed(self, texts):
                raise RuntimeError("embedding backend unavailable")

        monkeypatch.setenv("KAIRYU_EXTRA_ROUTE_KEYS", "key-a")
        ledger_path = tmp_path / "usage.jsonl"
        app = create_legacy_app(
            {"m": FailingBackend()},
            settings=ServerSettings(
                api_keys_env="KAIRYU_EXTRA_ROUTE_KEYS",
                usage_ledger_path=str(ledger_path),
            ),
            tenant_config=TenantConfig(key_tenants={"key-a": "tenant-a"}),
            embedding_backends={"embedding-model": FailingEmbeddingBackend(dimensions=8)},
        )
        headers = {"Authorization": "Bearer key-a"}

        with TestClient(app, raise_server_exceptions=False) as client:
            streamed_failure = client.post(
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
                json={"model": "embedding-model", "input": []},
            )
            failed_embedding = client.post(
                "/v1/embeddings",
                headers=headers,
                json={"model": "embedding-model", "input": "x"},
            )

        assert streamed_failure.status_code == 200
        assert "event: response.failed" in streamed_failure.text
        assert failed_response.status_code == 502
        assert invalid_embedding.status_code == 400
        assert failed_embedding.status_code == 502
        totals = UsageLedger(ledger_path).totals()["tenant-a"]
        # The streaming request crossed the dispatch boundary, so it is counted
        # exactly once even though the backend failed before its first chunk.
        # Unary/embedding failures before usable results remain unmetered.
        assert totals["requests"] == 1
        assert totals["completion_tokens"] == 0

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
                model="m",
                input="and again",
                previous_response_id=first.id,
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

    def test_stream_is_typed_responses_sse(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            response = client.post(
                "/v1/responses", json={"model": "m", "input": "x", "stream": True}
            )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "event: response.created" in response.text
        assert "event: response.output_text.delta" in response.text
        assert "event: response.completed" in response.text


class TestEmbeddings:
    def test_embedding_work_reserves_before_backend_dispatch(self):
        class RecordingEmbeddings(MockEmbeddingBackend):
            def __init__(self):
                super().__init__(dimensions=8)
                self.calls = 0

            async def embed(self, texts):
                self.calls += 1
                return await super().embed(texts)

        backend = RecordingEmbeddings()
        config = TenantConfig(
            limits={
                "default": TenantLimits(
                    requests_per_minute=60,
                    tokens_per_minute=1,
                    token_burst=1,
                    max_in_flight=1,
                )
            }
        )
        app = create_legacy_app(
            {"m": MockBackend()},
            tenant_config=config,
            embedding_backends={"embedding-model": backend},
        )

        with TestClient(app) as client:
            response = client.post(
                "/v1/embeddings",
                json={"model": "embedding-model", "input": "hello"},
            )
            metrics = client.get("/metrics").text

        assert response.status_code == 429
        assert backend.calls == 0
        assert (
            'kairyu_tenant_in_flight_requests{source="http",tenant="default"} 0.0'
        ) in metrics
        assert 'kairyu_tenant_reserved_tokens{tenant="default"} 0.0' in metrics

    def test_embedding_unknown_usage_consumes_full_reservation(self):
        config = TenantConfig(
            limits={
                "default": TenantLimits(
                    requests_per_minute=60,
                    tokens_per_minute=100,
                    token_burst=100,
                )
            }
        )
        app = create_legacy_app(
            {"m": MockBackend()},
            tenant_config=config,
            embedding_backends={
                "embedding-model": MockEmbeddingBackend(dimensions=8)
            },
        )

        with TestClient(app) as client:
            response = client.post(
                "/v1/embeddings",
                json={"model": "embedding-model", "input": "hello"},
            )

        assert response.status_code == 200
        assert app.state.tenant_limiter.token_balance("default") == pytest.approx(
            63,
            abs=0.1,
        )
        assert app.state.tenant_limiter.reservation_snapshot()["default"] == 0

    def test_embedding_exact_usage_refunds_conservative_reservation(self):
        class ExactEmbeddingBackend(MockEmbeddingBackend):
            def __init__(self, dimensions):
                super().__init__(dimensions=dimensions)
                self.reports_exact_usage = True

            async def embed(self, texts):
                result = await super().embed(texts)
                return EmbeddingResult(
                    vectors=result.vectors,
                    prompt_tokens=1,
                    usage_exact=True,
                )

        config = TenantConfig(
            limits={
                "default": TenantLimits(
                    requests_per_minute=60,
                    tokens_per_minute=100,
                    token_burst=100,
                )
            }
        )
        app = create_legacy_app(
            {"m": MockBackend()},
            tenant_config=config,
            embedding_backends={
                "embedding-model": ExactEmbeddingBackend(dimensions=8)
            },
        )

        with TestClient(app) as client:
            response = client.post(
                "/v1/embeddings",
                json={"model": "embedding-model", "input": "hello"},
            )

        assert response.status_code == 200
        assert response.json()["usage"] == {
            "prompt_tokens": 1,
            "total_tokens": 1,
        }
        assert app.state.tenant_limiter.token_balance("default") == pytest.approx(
            99,
            abs=0.1,
        )
        assert app.state.tenant_limiter.reservation_snapshot()["default"] == 0

    def test_sdk_round_trip_base64_default(self, tmp_path):
        import openai

        with TestClient(_auto_app(tmp_path)) as http:
            client = openai.OpenAI(
                base_url=str(http.base_url) + "/v1",
                api_key="sk-local",
                http_client=http,
            )
            result = client.embeddings.create(model="embedding-model", input=["hello", "world"])
            assert len(result.data) == 2
            assert len(result.data[0].embedding) == 8  # SDK decodes base64 (A9)
            assert result.usage.prompt_tokens > 0

    def test_float_and_base64_agree(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            as_float = client.post(
                "/v1/embeddings",
                json={
                    "model": "embedding-model",
                    "input": "hello",
                    "encoding_format": "float",
                },
            ).json()["data"][0]["embedding"]
            as_b64 = client.post(
                "/v1/embeddings",
                json={
                    "model": "embedding-model",
                    "input": "hello",
                    "encoding_format": "base64",
                },
            ).json()["data"][0]["embedding"]
            decoded = struct.unpack(f"<{len(as_float)}f", base64.b64decode(as_b64))
            assert list(decoded) == pytest.approx(as_float)

    def test_invalid_encoding_format_is_400(self, tmp_path):
        # M6: an unknown encoding_format (e.g. the typo "Base64") must be a 400,
        # not silently served as float.
        with TestClient(_auto_app(tmp_path)) as client:
            resp = client.post(
                "/v1/embeddings",
                json={
                    "model": "embedding-model",
                    "input": "hello",
                    "encoding_format": "Base64",
                },
            )
            assert resp.status_code == 400


class TestVisionWire:
    def test_content_parts_accepted_and_flattened(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "hello parts"}],
                        }
                    ],
                },
            )
            assert response.status_code == 200

    def test_image_parts_rejected_cleanly(self, tmp_path):
        with TestClient(_auto_app(tmp_path)) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "look:"},
                                {"type": "image_url", "image_url": {"url": "http://x/i.png"}},
                            ],
                        }
                    ],
                },
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
            cache,
            max_num_batched_tokens=8,
            page_size=4,
            priority_age_s=10.0,
            clock=lambda: clock["t"],
        )
        scheduler.add_request(EngineRequest("low", (1, 2, 3, 4), max_new_tokens=1, priority=5))
        clock["t"] = 1.0
        scheduler.add_request(EngineRequest("high", (5, 6, 7, 8), max_new_tokens=1, priority=0))
        plan = scheduler.schedule()
        assert plan.scheduled[0].request_id == "high"  # priority beats FIFO

        # aging: a very old low-priority request overtakes a fresh mid one
        scheduler2 = Scheduler(
            cache,
            max_num_batched_tokens=4,
            page_size=4,
            priority_age_s=1.0,
            clock=lambda: clock["t"],
        )
        clock["t"] = 0.0
        scheduler2.add_request(
            EngineRequest("old-low", (11, 12, 13, 14), max_new_tokens=1, priority=5)
        )
        clock["t"] = 10.0
        scheduler2.add_request(
            EngineRequest("new-mid", (15, 16, 17, 18), max_new_tokens=1, priority=0)
        )
        plan = scheduler2.schedule()
        assert plan.scheduled[0].request_id == "old-low"  # aging removes the 5-point gap

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

    def test_admission_controller_atomic_lease_lifecycle(self):
        from kairyu.entrypoints.server.slo import AdmissionController

        clock = {"t": 0.0}
        controller = AdmissionController(
            ttft_slo_s=1.0,
            defer_threshold_s=2.0,
            now=lambda: clock["t"],
        )
        lease = controller.begin()

        assert lease.decision.action == "admit"
        assert lease.started_at == 0.0
        assert lease.concurrency_at_start == 1
        assert controller.snapshot().in_flight == 1

        clock["t"] = 0.5
        lease.finished_first_token()
        lease.completed()
        assert controller.in_flight == 0

        with pytest.raises(RuntimeError, match="already completed"):
            lease.completed()
        with pytest.raises(RuntimeError, match="after completion"):
            lease.finished_first_token()

    def test_admission_elapsed_time_does_not_backdate_feedback_window(self):
        from kairyu.entrypoints.server.slo import AdmissionController

        clock = {"t": 0.0}
        controller = AdmissionController(
            ttft_slo_s=0.6,
            defer_threshold_s=1.0,
            now=lambda: clock["t"],
        )
        lease = controller.begin(elapsed_s=0.5)

        assert lease.decision.action == "admit"
        assert lease.decision.predicted_ttft_s == pytest.approx(0.51)
        assert lease.started_at == 0.0

        clock["t"] = 0.2
        lease.finished_first_token()
        lease.completed()
        # The EMA sees only post-admission latency: 0.8 * 0.01 + 0.2 * 0.2.
        assert controller.snapshot().ttft_per_unit_ema_s == pytest.approx(0.048)

    def test_admission_lease_freezes_concurrency_and_shed_does_not_reserve(self):
        from kairyu.entrypoints.server.slo import AdmissionController

        clock = {"t": 0.0}
        controller = AdmissionController(
            ttft_slo_s=0.25,
            defer_threshold_s=0.5,
            now=lambda: clock["t"],
        )
        first = controller.begin()
        second = controller.begin()
        assert first.concurrency_at_start == 1
        assert second.concurrency_at_start == 2

        clock["t"] = 1.0
        first.finished_first_token()
        # 0.8 * optimistic 0.01 + 0.2 * (1 second / frozen concurrency 1)
        assert controller.snapshot().ttft_per_unit_ema_s == pytest.approx(0.208)
        shed = controller.begin()
        assert shed.decision.action == "shed"
        assert not shed.active
        assert controller.in_flight == 2

        second.finished_first_token()
        first.completed()
        second.completed()
        assert controller.in_flight == 0

    def test_deferred_work_is_bounded_but_does_not_poison_interactive_ema(self):
        from kairyu.entrypoints.server.slo import AdmissionController

        clock = {"t": 0.0}
        controller = AdmissionController(
            ttft_slo_s=0.25,
            defer_threshold_s=1.0,
            now=lambda: clock["t"],
        )
        first = controller.begin()
        second = controller.begin()
        clock["t"] = 1.0
        first.finished_first_token()
        ema = controller.snapshot().ttft_per_unit_ema_s

        deferred = controller.begin()
        assert deferred.decision.action == "defer"
        assert controller.snapshot().deferred_in_flight == 1
        clock["t"] = 10.0
        deferred.finished_first_token()
        assert controller.snapshot().ttft_per_unit_ema_s == ema

        another_deferred = controller.begin()
        assert another_deferred.decision.action == "defer"
        shed = controller.begin()
        assert shed.decision.action == "shed"
        assert controller.snapshot().in_flight == 4

        second.finished_first_token()
        for lease in (first, second, deferred, another_deferred):
            lease.completed()
        assert controller.snapshot().in_flight == 0
        assert controller.snapshot().interactive_in_flight == 0
        assert controller.snapshot().deferred_in_flight == 0

    @pytest.mark.parametrize(
        ("slo", "threshold"),
        [
            (float("nan"), None),
            (float("inf"), None),
            (1.0, 1.0),
            (1.0, 0.5),
            (1.0, float("nan")),
            (1.0, float("inf")),
        ],
    )
    def test_admission_controller_rejects_invalid_thresholds(
        self, slo, threshold
    ):
        from kairyu.entrypoints.server.slo import AdmissionController

        with pytest.raises(ValueError):
            AdmissionController(slo, threshold)

    def test_autoscale_hysteresis_table(self):
        from kairyu.entrypoints.server.slo import autoscale_decision

        assert autoscale_decision([0.9, 0.95, 0.9], queue_depth=4).action == "scale_up"
        assert autoscale_decision([0.1, 0.2, 0.1], queue_depth=0).action == "scale_down"
        assert autoscale_decision([0.5, 0.6, 0.5], queue_depth=0).action == "hold"
        assert autoscale_decision([0.9], queue_depth=9).action == "hold"  # window
        assert autoscale_decision([0.9, 0.9, 0.9], queue_depth=0).action == "hold"


def test_frontier_scoreboard_schema():
    from verification.product.performance.frontier_compare import (
        TargetReport,
        TrialResult,
        build_scoreboard,
    )

    report = TargetReport(name="kairyu", model="m")
    report.trials.append(
        TrialResult(
            ttft_s=0.05,
            tpot_s=0.01,
            output_chars=100,
            completion_tokens=3,
        )
    )
    scoreboard = build_scoreboard([report])
    assert scoreboard["methodology"]["metric_definitions"]["ttft"]
    assert "completion_tokens" in scoreboard["methodology"]["metric_definitions"]["tpot"]
    assert scoreboard["results"][0]["ttft_p50_s"] == 0.05
    assert scoreboard["results"][0]["tpot_missing_usage_trials"] == 0


class _FakeFrontierCompletions:
    def __init__(self, chunks):
        self._chunks = chunks
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs

        async def stream():
            for chunk in self._chunks:
                yield chunk

        return stream()


class _FakeFrontierClient:
    def __init__(self, chunks):
        completions = _FakeFrontierCompletions(chunks)
        self.chat = SimpleNamespace(completions=completions)


def _frontier_chunk(content=None, *, completion_tokens=None):
    choices = (
        [SimpleNamespace(delta=SimpleNamespace(content=content))] if content is not None else []
    )
    usage = (
        SimpleNamespace(completion_tokens=completion_tokens)
        if completion_tokens is not None
        else None
    )
    return SimpleNamespace(choices=choices, usage=usage)


async def test_frontier_tpot_uses_final_completion_tokens(monkeypatch):
    from verification.product.performance import frontier_compare

    client = _FakeFrontierClient(
        [
            _frontier_chunk("three tokens"),
            _frontier_chunk(" one chunk"),
            _frontier_chunk(completion_tokens=4),
        ]
    )
    clock = iter([0.0, 1.0, 4.0, 5.0])
    monkeypatch.setattr(frontier_compare.time, "perf_counter", lambda: next(clock))

    result = await frontier_compare.run_trial(
        client,
        frontier_compare.Target("kairyu", "http://localhost/v1", "m"),
        "prompt",
    )

    assert client.chat.completions.kwargs["stream_options"] == {"include_usage": True}
    assert result.ttft_s == 1.0
    assert result.output_chars == len("three tokens one chunk")
    assert result.completion_tokens == 4
    assert result.tpot_s == 1.0  # (last content 4 - first content 1) / (4 - 1)


async def test_frontier_missing_usage_never_substitutes_chunk_count(monkeypatch):
    from verification.product.performance import frontier_compare

    client = _FakeFrontierClient([_frontier_chunk("first"), _frontier_chunk("second")])
    clock = iter([0.0, 1.0, 4.0, 5.0])
    monkeypatch.setattr(frontier_compare.time, "perf_counter", lambda: next(clock))

    result = await frontier_compare.run_trial(
        client,
        frontier_compare.Target("legacy", "http://legacy/v1", "m"),
        "prompt",
    )
    report = frontier_compare.TargetReport("legacy", "m", trials=[result])

    assert result.ttft_s == 1.0
    assert result.output_chars == len("firstsecond")
    assert result.completion_tokens is None
    assert result.tpot_s is None
    assert report.summary()["tpot_missing_usage_trials"] == 1


def test_auto_head_stream_emits_public_text_exactly_once(tmp_path):
    """EO-D7 regression: the split head/continuation stream synthesizes its
    delta completions server-side; the terminal result chunk must emit only a
    genuine tail, never the whole already-streamed public text again."""

    from kairyu.orchestration.conductor import RoleSpec
    from kairyu.orchestration.features import extract_features
    from kairyu.orchestration.router import RouteDecision

    class ForceMultiRouter:
        def route(self, prompt):
            return RouteDecision(
                target="multi_agent",
                confidence=1.0,
                features=extract_features(prompt),
                reason="test",
            )

    class StreamBackend(MockBackend):
        def __init__(self, text):
            super().__init__()
            self._text = text

        async def generate(self, request):
            return GenerationResult(
                request_id=request.request_id,
                prompt=request.prompt,
                completions=(
                    CompletionOutput(index=0, text=self._text, token_ids=(1,)),
                ),
                usage=GenerationUsage(prompt_tokens=2, completion_tokens=2),
            )

        async def stream(self, request):
            for end in range(1, len(self._text) + 1):
                yield GenerationResult(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    completions=(
                        CompletionOutput(
                            index=0,
                            text=self._text[:end],
                            token_ids=(1,),
                            finish_reason="stop" if end == len(self._text) else None,
                        ),
                    ),
                    finished=end == len(self._text),
                )

    roles = (
        RoleSpec(name="head", worker="hw", role_type="head", prompt="[h] {query}"),
        RoleSpec(name="draft", worker="dw", prompt="[d] {query} {head}",
                 depends_on=("head",)),
        RoleSpec(
            name="continuation",
            worker="cw",
            role_type="publisher",
            prompt="[c] {head} {draft}",
            depends_on=("head", "draft"),
        ),
    )
    orchestrator = Orchestrator(
        {
            "hw": StreamBackend("Intro. "),
            "dw": StreamBackend("draft"),
            "cw": StreamBackend("Body."),
        },
        router=ForceMultiRouter(),
        roles=roles,
    )
    app = create_legacy_app(
        {},
        orchestrators={"kairyu-auto": orchestrator},
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
    )
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "kairyu-auto",
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        ) as response:
            body = "".join(response.iter_text())
    assert "data: [DONE]" in body
    contents = []
    for line in body.splitlines():
        if not line.startswith("data: ") or "[DONE]" in line:
            continue
        chunk = json.loads(line[len("data: ") :])
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                contents.append(delta["content"])
    assert "".join(contents) == "Intro. Body."
