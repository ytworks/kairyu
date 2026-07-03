"""m11 gates: streaming orchestrator usage, tiers, tenancy, responses,
embeddings, vision wire, F5 logic, bench schema."""

import base64
import struct

import pytest
from fastapi.testclient import TestClient

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

            usage = client.get(
                "/admin/usage", headers=a
            ).json()["usage"]
            assert usage["tenant-a"]["requests"] == 2
            assert usage["tenant-b"]["requests"] == 1
            assert usage["tenant-a"]["completion_tokens"] > 0

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


class TestResponsesApi:
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
