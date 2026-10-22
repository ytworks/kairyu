"""Cross-file contracts of the replica-pool scale-out examples (FN-D9).

Each example is three files that must agree — ``example.json`` (allocation),
``compose.yaml`` (which GPUs each vLLM service gets), ``kairyu.yaml`` (which
services the pool places onto) — plus the placement gate its ``verification.py``
applies to the pool's JSONL log. Nothing else validates that agreement.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from kairyu.deploy.spec import load_deployment_spec
from kairyu.engine.config_validation import validate_backend_options

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = {
    "qwen3.8-27b-dp8-8gpu": "qwen",
    "deepseek-v4.1-flash-8gpu": "deepseek",
}
# vLLM-side tool parser and non-thinking default kwarg per example.
TOOL_PARSER = {
    "qwen3.8-27b-dp8-8gpu": "qwen3_coder",
    "deepseek-v4.1-flash-8gpu": "deepseek_v41",
}
THINKING_KEY = {
    "qwen3.8-27b-dp8-8gpu": "enable_thinking",
    "deepseek-v4.1-flash-8gpu": "thinking",
}


def _load(path: Path, name: str):
    module_spec = importlib.util.spec_from_file_location(name, path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("environment", sorted(EXAMPLES))
def test_allocation_compose_and_pool_agree(environment: str) -> None:
    example = ROOT / "examples" / environment
    spec = json.loads((example / "example.json").read_text())
    compose = yaml.safe_load((example / "compose.yaml").read_text())
    deployment = load_deployment_spec((example / "kairyu.yaml").read_text())
    allocation = spec["allocation"]
    prefix = EXAMPLES[environment]
    replicas = int(allocation["replicas"])

    # compose: one vLLM service per replica, GPUs 0..7 partitioned exactly once.
    services = compose["services"]
    l1 = [f"{prefix}-{index}" for index in range(replicas)]
    assert set(services) == {*l1, "kairyu"}
    # A DeepSeek-V4 effort overlay (xhigh -> max) is the image L1 serves.
    served = spec["vllm"].get("effort_overlay", spec["vllm"])
    served_image = served["image"]
    claimed: list[int] = []
    for name in l1:
        service = services[name]
        assert service["image"].endswith(f":-{served_image}}}")
        devices = service["deploy"]["resources"]["reservations"]["devices"]
        assert len(devices) == 1
        ids = [int(value) for value in devices[0]["device_ids"]]
        assert len(ids) == int(allocation["tensor_parallel_size"])
        claimed.extend(ids)
        assert services["kairyu"]["depends_on"][name] == {"condition": "service_healthy"}
    gpu_count = int(spec["hardware"]["gpu_count"])
    assert sorted(claimed) == list(allocation["gpu_ids"]) == list(range(gpu_count))
    if "replica_gpu_ids" in allocation:
        groups = [
            services[name]["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"]
            for name in l1
        ]
        assert [[int(v) for v in group] for group in groups] == allocation["replica_gpu_ids"]
    if "kv_cache_memory_bytes" in spec["vllm"]:
        # A cold torch.compile cache inflates vLLM's start-up memory profile,
        # so without the pinned budget a first boot serves a ~5x smaller KV
        # cache than the sibling replica (Qwen3.8-Flash-Next: 741K vs 3.47M).
        for name in l1:
            command = services[name]["command"]
            budget = command[command.index("--kv-cache-memory") + 1]
            assert int(budget) == spec["vllm"]["kv_cache_memory_bytes"]

    # kairyu.yaml: one pool, one replica per compose service, even-spread knobs.
    model = allocation["model"]
    assert set(deployment.pools) == {model}
    assert not deployment.engines and not deployment.orchestrators
    pool = deployment.pools[model]
    assert len(pool.replicas) == replicas
    assert pool.queue_depth_threshold == spec["pool"]["queue_depth_threshold"] == 0
    assert pool.prefix_index is spec["pool"]["prefix_index"] is True
    assert pool.unhealthy_after == spec["pool"]["unhealthy_after"]
    assert pool.placement_log_path == spec["pool"]["placement_log"]
    hosts = []
    for replica in pool.replicas:
        validate_backend_options(replica.backend, replica.options)
        assert replica.options["tensor_parallel_size"] == allocation["tensor_parallel_size"]
        assert replica.options["container_image_digest"] == served["image_id"]
        hosts.append(replica.options["base_url"].removeprefix("http://").split(":")[0])
    assert hosts == l1
    assert deployment.server.max_concurrency == replicas * spec["vllm"]["max_num_seqs"]
    # vLLM owns the chat template and the tool-call parser, and each request
    # reaches it in its caller's API (m9 D2 amendment 2026-10-17). The
    # passthrough-to-/completions shape drops tools entirely (PR #584).
    assert deployment.upstream_chat_models == frozenset({model})
    assert deployment.upstream_responses_models == frozenset({model})
    assert deployment.upstream_messages_models == frozenset({model})
    assert not deployment.legacy_chat_models
    assert deployment.chat_templates == {}
    assert "allow_templated_chat_passthrough" not in pool.replicas[0].options
    if prefix == "deepseek":
        expert_parallel = pool.replicas[0].options["expert_parallel_size"]
        assert expert_parallel == allocation["expert_parallel_size"]

    # Without vLLM's tool parser the DSML/tool markup leaks into
    # message.content and tool_calls stays null — the reviewed live failure.
    # Without the non-thinking default kwargs the reasoning parser files a
    # plain answer (no </think>) as reasoning_content and content is empty.
    parser = TOOL_PARSER[environment]
    thinking_key = THINKING_KEY[environment]
    for name in l1:
        command = services[name]["command"]
        assert "--enable-auto-tool-choice" in command
        assert command[command.index("--tool-call-parser") + 1] == parser
        default_kwargs = command[command.index("--default-chat-template-kwargs") + 1]
        assert json.loads(default_kwargs)[thinking_key] is (
            spec["model"].get("default_reasoning_effort") is not None
        )


def _write_log(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_placement_gate_reads_only_the_row_delta_and_rejects_skew(tmp_path: Path) -> None:
    module = _load(ROOT / "examples/qwen3.8-27b-dp8-8gpu/verification.py", "qwen_dp8_verification")
    log = tmp_path / "placement.jsonl"
    # Earlier traffic (warm-up) must not count toward the row.
    _write_log(log, [{"kind": "replica", "replica_id": "0", "reason": "least_outstanding"}] * 5)
    offset = module._placement_offset(log)

    even = [
        {"kind": "replica", "replica_id": str(index % 4), "reason": "least_outstanding"}
        for index in range(16)
    ] + [{"kind": "membership", "replica_ids": ["0", "1", "2", "3"]}]
    _write_log(log, even)
    gate = {"expected_requests": 16, "replicas": 4, "max_share_of_mean": 2.0, "settle_s": 0}
    report = module._placement_report(log, offset, gated=True, **gate)
    assert report["per_replica"] == {"0": 4, "1": 4, "2": 4, "3": 4}
    assert report["placements"] == 16 and report["passed"] is True

    offset = module._placement_offset(log)
    skewed = [{"kind": "replica", "replica_id": "0", "reason": "prefix_match"}] * 15 + [
        {"kind": "replica", "replica_id": "1", "reason": "least_outstanding"}
    ]
    _write_log(log, skewed)
    report = module._placement_report(log, offset, gated=True, **gate)
    assert report["passed"] is False  # two replicas idle, one above 2x the mean
    ungated = module._placement_report(log, offset, gated=False, **gate)
    assert ungated["passed"] is None and ungated["per_replica"] == {"0": 15, "1": 1}


def test_qwen_eight_replica_gate_rejects_material_skew(tmp_path: Path) -> None:
    module = _load(
        ROOT / "examples/qwen3.8-27b-dp8-8gpu/verification.py",
        "qwen_dp8_gate_verification",
    )
    log = tmp_path / "placement.jsonl"
    skewed_counts = (16, 16, 16, 8, 2, 2, 2, 2)
    skewed = [
        {"kind": "replica", "replica_id": str(replica)}
        for replica, count in enumerate(skewed_counts)
        for _ in range(count)
    ]
    _write_log(log, skewed)
    gate = module.SPEC["verification"]["serving"]["placement_gate"]
    report = module._placement_report(
        log,
        0,
        expected_requests=64,
        replicas=8,
        gated=True,
        max_share_of_mean=float(gate["max_share_of_mean"]),
        settle_s=0,
    )
    assert report["per_replica"] == {
        "0": 16,
        "1": 16,
        "2": 16,
        "3": 8,
        "4": 2,
        "5": 2,
        "6": 2,
        "7": 2,
    }
    assert report["largest_share_of_mean"] == 2.0
    assert report["passed"] is False

    offset = module._placement_offset(log)
    balanced = [
        {"kind": "replica", "replica_id": str(replica)} for replica in range(8) for _ in range(8)
    ]
    _write_log(log, balanced)
    report = module._placement_report(
        log,
        offset,
        expected_requests=64,
        replicas=8,
        gated=True,
        max_share_of_mean=float(gate["max_share_of_mean"]),
        settle_s=0,
    )
    assert report["per_replica"] == {str(replica): 8 for replica in range(8)}
    assert report["largest_share_of_mean"] == 1.0
    assert report["passed"] is True


@pytest.mark.parametrize(
    ("environment", "replicas", "benchmark_counts", "unrelated_counts"),
    [
        (
            "qwen3.8-27b-dp8-8gpu",
            8,
            (16, 16, 16, 8, 2, 2, 2, 2),
            (0, 0, 0, 8, 14, 14, 14, 14),
        ),
    ],
)
def test_placement_gate_rejects_extra_requests_that_mask_skew(
    tmp_path: Path,
    environment: str,
    replicas: int,
    benchmark_counts: tuple[int, ...],
    unrelated_counts: tuple[int, ...],
) -> None:
    module = _load(
        ROOT / "examples" / environment / "verification.py",
        f"{environment}_extra_placements_verification",
    )
    log = tmp_path / "placement.jsonl"
    rows = [
        {"kind": "replica", "replica_id": str(replica)}
        for counts in (benchmark_counts, unrelated_counts)
        for replica, count in enumerate(counts)
        for _ in range(count)
    ]
    _write_log(log, rows)
    gate = module.SPEC["verification"]["serving"]["placement_gate"]
    report = module._placement_report(
        log,
        0,
        expected_requests=sum(benchmark_counts),
        replicas=replicas,
        gated=True,
        max_share_of_mean=float(gate["max_share_of_mean"]),
        settle_s=0,
    )

    assert report["placements"] == sum(benchmark_counts) + sum(unrelated_counts)
    assert report["largest_share_of_mean"] == 2.0
    assert report["passed"] is False


@pytest.mark.parametrize("environment", sorted(EXAMPLES))
def test_tool_result_turn_requires_a_parsed_tool_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
) -> None:
    module = _load(
        ROOT / "examples" / environment / "verification.py",
        f"{environment}_tool_result_gate",
    )
    good_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": "ls"})},
    }
    good_choice = {
        "message": {"tool_calls": [good_call]},
        "finish_reason": "tool_calls",
    }
    leaked_markup_choice = {
        "message": {
            "content": "<tool_calls><invoke name='bash'>ls</invoke></tool_calls>",
            "tool_calls": None,
        },
        "finish_reason": "stop",
    }
    streamed_call = "\n".join(
        [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"name":"bash","arguments":"{\\"command\\":\\"ls\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]
    )
    thinking_choice = {
        "message": {"tool_calls": [good_call], "reasoning_content": "I should list files."},
        "finish_reason": "tool_calls",
    }
    responses = iter(
        [(200, {"choices": [good_choice]})] * (2 * module.REPLICAS)
        + [
            (200, {"choices": [leaked_markup_choice]}),
            (200, streamed_call),
            (200, {"choices": [thinking_choice]}),
            (200, {"choices": [{"message": {"content": "OK"}}]}),
        ]
        + (
            [(200, {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]})]
            if module.SPEC["model"].get("default_reasoning_effort")
            else []
        )
    )
    request_ids = iter(f"req-{index}" for index in range(2 * module.REPLICAS))

    def post_responses(payload, *, include_request_id=False, **kwargs):
        response = next(responses)
        return (*response, next(request_ids)) if include_request_id else response

    monkeypatch.setattr(module, "_post_responses", post_responses)
    monkeypatch.setattr(module, "_placement_offset", lambda path: 0)
    monkeypatch.setattr(
        module,
        "_placement_counts",
        lambda path, offset, **kwargs: {str(replica): 2 for replica in range(module.REPLICAS)},
    )

    assert module.tool_calling(tmp_path) == 1
    report = json.loads((tmp_path / "tool-calling.json").read_text())
    assert report["passed"] is False
    assert report["cases"]["tool_result_turn"] == {
        "detail": None,
        "error": "finish_reason is 'stop', expected 'tool_calls'",
        "passed": False,
    }


@pytest.mark.parametrize("environment", sorted(EXAMPLES))
def test_tool_placement_gate_ignores_unrelated_requests(
    tmp_path: Path,
    environment: str,
) -> None:
    module = _load(
        ROOT / "examples" / environment / "verification.py",
        f"{environment}_tool_placement_gate",
    )
    fan = 2 * module.REPLICAS
    tool_request_ids = {f"tool-{index}" for index in range(fan)}
    rows = [
        {
            "kind": "replica",
            "request_id": request_id,
            "replica_id": "0",
        }
        for request_id in tool_request_ids
    ] + [
        {
            "kind": "replica",
            "request_id": f"unrelated-{replica}",
            "replica_id": str(replica),
        }
        for replica in range(1, module.REPLICAS)
    ]
    log = tmp_path / "placement.jsonl"
    _write_log(log, rows)

    counts = module._placement_counts(log, 0, request_ids=tool_request_ids)

    assert counts == {"0": fan}
    error = module._tool_placement_error(
        counts,
        expected_requests=fan,
        replicas=module.REPLICAS,
    )
    assert error == (
        None
        if module.REPLICAS == 1
        else f"only 1 of {module.REPLICAS} replicas served tool calls: {{'0': {fan}}}"
    )


@pytest.mark.parametrize("environment", sorted(EXAMPLES))
def test_tool_placement_gate_waits_for_async_log_writes(
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
) -> None:
    module = _load(
        ROOT / "examples" / environment / "verification.py",
        f"{environment}_tool_placement_settle",
    )
    fan = 2 * module.REPLICAS
    final_counts = {str(replica): 2 for replica in range(module.REPLICAS)}
    snapshots = iter([{}, final_counts])
    monkeypatch.setattr(module, "_placement_counts", lambda *args, **kwargs: next(snapshots))
    monkeypatch.setattr(module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    counts = module._wait_for_placement_counts(
        Path("unused.jsonl"),
        0,
        expected_requests=fan,
        request_ids={f"tool-{index}" for index in range(fan)},
        settle_s=1.0,
    )

    assert counts == final_counts
    assert (
        module._tool_placement_error(
            counts,
            expected_requests=fan,
            replicas=module.REPLICAS,
        )
        is None
    )
