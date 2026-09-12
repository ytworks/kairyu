import importlib.util
from pathlib import Path

import pytest

PATH = (
    Path(__file__).resolve().parents[2] / "examples/qwen3.8-deepseek-v4.1-8gpu/gpu_cancellation.py"
)
spec = importlib.util.spec_from_file_location("v41_gpu_cancellation", PATH)
probe = importlib.util.module_from_spec(spec)


def test_metrics_require_both_gauges_and_requested_rank():
    spec.loader.exec_module(probe)
    text = "\n".join(
        f'vllm:num_requests_{gauge}{{engine="{rank}"}} {int(rank == 1 and gauge == "running")}'
        for gauge in ("running", "waiting")
        for rank in range(3)
    )
    result = probe.parse_metrics(text, rank_label="engine")
    assert probe.idle(result) is False
    assert probe.rank_running(result, 1)
    assert not probe.rank_running(result, 0)
    with pytest.raises(ValueError, match="rank"):
        probe.rank_running(result, 3)
    with pytest.raises(ValueError, match="gauges"):
        probe.parse_metrics('vllm:num_requests_running{engine="0"} 0', rank_label="engine")


def test_nan_and_aggregate_only_metrics_cannot_prove_rank_cleanup():
    spec.loader.exec_module(probe)
    with pytest.raises(ValueError):
        probe.parse_metrics(
            'vllm:num_requests_running{engine="0"} NaN\nvllm:num_requests_waiting{engine="0"} 0',
            rank_label="engine",
        )
    totals = probe.parse_metrics(
        "vllm:num_requests_running 0\nvllm:num_requests_waiting 0", rank_label="engine"
    )
    assert probe.idle(totals)
    with pytest.raises(ValueError, match="rank"):
        probe.rank_running(totals, 0)


def test_cleanup_verdict_requires_server_observation_and_recovery():
    spec.loader.exec_module(probe)
    result = {
        "first_visible": True,
        "observed_active": True,
        "client_closed": True,
        "stream_ended_before_close": False,
        "server_idle_after_close": True,
        "recovery_passed": True,
    }
    assert probe.verdict(result)
    for key in ("first_visible", "observed_active", "server_idle_after_close", "recovery_passed"):
        assert not probe.verdict(dict(result, **{key: False}))
    assert not probe.verdict(dict(result, stream_ended_before_close=True))


@pytest.mark.parametrize("cleanup_pass", [True, False])
def test_client_close_is_followed_by_server_cleanup_gate(tmp_path, monkeypatch, cleanup_pass):
    import asyncio
    import json
    from types import SimpleNamespace

    import httpx

    spec.loader.exec_module(probe)

    class EndlessStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"First token"}}]}\n\n'
            await asyncio.Event().wait()

        async def aclose(self):
            self.closed = True

    raw = EndlessStream()

    async def mocked_idle(client, sources, directory, phase, args):
        if phase == "after":
            assert raw.closed, "Client must close before cleanup is observed"
            if not cleanup_pass:
                raise TimeoutError("Server still running")
        return probe.stamp()

    async def mocked_snapshot(*args):
        return {"deepseek": {"0": {"running": 1, "waiting": 0}}}

    def handler(request):
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(200, stream=raw)
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": "323"}}]}
        )

    monkeypatch.setattr(probe, "wait_idle", mocked_idle)
    monkeypatch.setattr(probe, "snapshot", mocked_snapshot)
    args = SimpleNamespace(
        output=tmp_path,
        suite="native",
        model="deepseek-v4.1-flash",
        base_url="http://test/v1",
        activation_timeout=1,
        rank_label="engine",
        poll_interval=0.001,
        recovery_timeout=1,
    )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await asyncio.wait_for(
                probe.run_case(client, args, {"deepseek": "unused"}, 0), 2
            )

    result = asyncio.run(run())
    assert raw.closed
    assert result["passed"] is cleanup_pass
    assert (tmp_path / "native-rank0" / "response.sse").exists()
    assert bool(result.get("recovery_passed")) is cleanup_pass


def test_public_requires_deepseek_and_qwen_overlap_but_not_both_replicas():
    spec.loader.exec_module(probe)
    assert probe.public_active(["deepseek", "qwen-0"])
    assert probe.public_active(["deepseek", "qwen-1"])
    assert not probe.public_active(["qwen-0", "qwen-1"])
    assert not probe.public_active(["deepseek"])
    assert probe.verdict(
        {
            "suite": "public",
            "observed_active": True,
            "client_closed": True,
            "server_idle_after_close": True,
            "recovery_passed": True,
        }
    )


def test_snapshot_rejects_missing_rank_and_changed_inventory(tmp_path, monkeypatch):
    import asyncio

    spec.loader.exec_module(probe)
    ranks = {"deepseek": ["0", "1", "2"], "qwen-0": ["0"]}

    async def metrics(client, source):
        return "\n".join(
            f'vllm:num_requests_{gauge}{{engine="{rank}"}} 0'
            for rank in ranks[source]
            for gauge in ("running", "waiting")
        )

    monkeypatch.setattr(probe, "metric_text", metrics)

    async def capture():
        return await probe.snapshot(None, {key: key for key in ranks}, tmp_path, "after", "engine")

    asyncio.run(capture())
    ranks["deepseek"] = ["0", "1"]
    with pytest.raises(ValueError, match="exactly ranks"):
        asyncio.run(capture())
    ranks["deepseek"] = ["0", "1", "2"]
    ranks["qwen-0"] = ["1"]
    with pytest.raises(ValueError, match="inventory changed"):
        asyncio.run(capture())
