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


def test_audit_hook_requires_fresh_timestamp_and_exact_fields():
    spec.loader.exec_module(probe)
    line = (
        "2026-09-13T01:00:01.123456789Z INFO Kairyu role hook: role=audit "
        "max_tokens=4096 effort=high messages_sha256=" + "a" * 64 + " thinking_token_budget=2048\n"
    )
    hooks = probe.audit_hooks(line, "qwen-0", "2026-09-13T01:00:00+00:00")
    assert len(hooks) == 1
    assert hooks[0]["messages_sha256"] == "a" * 64
    assert hooks[0]["worker"] == "qwen-0"
    assert hooks[0]["raw"] == line.rstrip("\n")
    assert not probe.audit_hooks(line, "qwen-0", "2026-09-13T01:00:02+00:00")
    with pytest.raises(ValueError, match="timestamp"):
        probe.audit_hooks(line.split(" ", 1)[1], "qwen-0", "2026-09-13T01:00:00+00:00")
    with pytest.raises(ValueError, match="hash"):
        probe.audit_hooks(line, "qwen-0", "2026-09-13T01:00:00+00:00", "b" * 64)


def test_live_audit_requires_unique_hook_matching_running_worker_and_later_keepalive():
    spec.loader.exec_module(probe)
    hook = {"worker": "qwen-0", "observed": {"monotonic_s": 3}}
    values = {
        "deepseek": {str(i): {"running": 0, "waiting": 0} for i in range(3)},
        "qwen-0": {"0": {"running": 1, "waiting": 0}},
        "qwen-1": {"0": {"running": 0, "waiting": 0}},
    }
    assert probe.audit_active([hook], values, {"monotonic_s": 4})
    assert not probe.audit_active([hook], values, {"monotonic_s": 2})
    assert not probe.audit_active([hook], values, {"monotonic_s": 4}, active_since=5)
    assert probe.audit_active([hook], values, {"monotonic_s": 6}, active_since=5)
    assert not probe.audit_active([], values, {"monotonic_s": 4})
    assert not probe.audit_active([dict(hook, worker="qwen-1")], values, {"monotonic_s": 4})
    values["deepseek"]["0"]["running"] = 1
    assert not probe.audit_active([hook], values, {"monotonic_s": 4})
    with pytest.raises(ValueError, match="unique"):
        probe.audit_active([hook, hook], values, {"monotonic_s": 4})


def test_public_audit_verdict_cannot_pass_without_stage_proof():
    spec.loader.exec_module(probe)
    result = dict(
        suite="public-audit",
        first_visible=True,
        observed_active=True,
        client_closed=True,
        server_idle_after_close=True,
        recovery_passed=True,
    )
    assert not probe.verdict(result)
    assert probe.verdict(dict(result, live_audit=True))


async def test_audit_file_cursor_ignores_existing_and_partial_records(tmp_path):
    spec.loader.exec_module(probe)
    log = tmp_path / "worker.log"
    line = (
        "2026-09-13T01:00:01Z Kairyu role hook: role=audit max_tokens=4096 "
        "effort=high messages_sha256=" + "a" * 64 + " thinking_token_budget=2048\n"
    )
    log.write_text(line)
    sources = {"qwen-0": str(log)}
    cursors = probe.audit_cursors(sources)
    assert not await probe.fresh_audit_hooks(sources, cursors, "2026-09-13T01:00:00Z", tmp_path)
    with log.open("a") as out:
        out.write(line[:80])
    assert not await probe.fresh_audit_hooks(sources, cursors, "2026-09-13T01:00:00Z", tmp_path)
    with log.open("a") as out:
        out.write(line[80:])
    hooks = await probe.fresh_audit_hooks(sources, cursors, "2026-09-13T01:00:00Z", tmp_path)
    assert len(hooks) == 1
    assert not await probe.fresh_audit_hooks(sources, cursors, "2026-09-13T01:00:00Z", tmp_path)
    log.write_text("")
    with pytest.raises(ValueError, match="truncated"):
        await probe.fresh_audit_hooks(sources, cursors, "2026-09-13T01:00:00Z", tmp_path)


async def test_public_audit_waits_past_initial_parallel_work_then_checks_cleanup(
    tmp_path, monkeypatch
):
    import asyncio
    import json
    from types import SimpleNamespace

    import httpx

    spec.loader.exec_module(probe)
    raw_closed = asyncio.Event()
    calls = []

    class PublicStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"Opening."}}]}\n\n'
            while True:
                await asyncio.sleep(0.001)
                yield b": status working\n\n"

        async def aclose(self):
            raw_closed.set()

    async def observe_idle(client, sources, directory, phase, args):
        calls.append(phase)
        if phase == "after":
            assert raw_closed.is_set()
        return probe.stamp()

    async def hooks(sources, cursors, started, directory, expected):
        calls.append("logs")
        if calls.count("logs") == 2:
            return [{"worker": "qwen-1", "observed": probe.stamp(), "messages_sha256": "a" * 64}]
        return []

    async def metrics(*args):
        calls.append("metrics")
        parallel = calls.count("metrics") == 1
        return {
            "deepseek": {
                str(i): {"running": int(parallel and i == 0), "waiting": 0} for i in range(3)
            },
            "qwen-0": {"0": {"running": 0, "waiting": 0}},
            "qwen-1": {"0": {"running": 1, "waiting": 0}},
        }

    def handler(request):
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, stream=PublicStream())
        assert raw_closed.is_set()
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": "323"}}]}
        )

    request = tmp_path / "input.json"
    request.write_text(json.dumps({"model": "kairyu-auto-max", "messages": []}))
    args = SimpleNamespace(
        output=tmp_path,
        suite="public-audit",
        public_request=request,
        model="kairyu-auto-max",
        base_url="http://test/v1",
        activation_timeout=1,
        rank_label="engine",
        poll_interval=0.002,
        recovery_timeout=1,
        audit_logs={"qwen-0": "docker://q0", "qwen-1": "docker://q1"},
        expected_audit_sha256=None,
    )
    monkeypatch.setattr(probe, "wait_idle", observe_idle)
    monkeypatch.setattr(probe, "fresh_audit_hooks", hooks)
    monkeypatch.setattr(probe, "snapshot", metrics)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await asyncio.wait_for(probe.run_case(client, args, {}, None), 2)
    assert result["passed"]
    assert calls.count("logs") >= 3  # next iteration receives a post-hook keepalive
    assert result["workers_active_at_close"] == ["qwen-1"]
    assert (
        result["live_audit"]["keepalive"]["monotonic_s"]
        > result["live_audit"]["hook"]["observed"]["monotonic_s"]
    )
    assert (
        result["audit_active_before_keepalive"]["monotonic_s"]
        < result["live_audit"]["keepalive"]["monotonic_s"]
        < result["live_audit"]["monotonic_s"]
    )
    assert calls[-2:] == ["after", "recovery"]
