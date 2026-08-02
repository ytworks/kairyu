"""Production wiring for replicated-QKV/KV EP correctness mode (#166)."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kairyu.engine.core import worker as worker_module
from kairyu.engine.core.comm import FakeCommunicator
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.scheduler import EngineRequest, ScheduledChunk, Scheduler
from kairyu.engine.core.step_input import RequestSnapshot


def _write_ep_checkpoint(directory: Path) -> tuple[str, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text('{"model_type":"qwen3_moe"}')
    (directory / "hf_quant_config.json").write_text(
        '{"producer":{"name":"modelopt","version":"0.33.0"}}'
    )
    shard_names = tuple(
        f"model-{index:05d}-of-00027.safetensors"
        for index in range(1, 28)
    )
    weight_map = {
        f"model.layers.{index}.weight": shard_name
        for index, shard_name in enumerate(shard_names)
    }
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 378}, "weight_map": weight_map})
    )
    for index, shard_name in enumerate(shard_names, start=1):
        (directory / shard_name).write_bytes(bytes([index]) * index)
    return shard_names


def _kernel_probe_runner(
    rank: int,
    world_size: int,
    *,
    malformed_selection: bool = False,
):
    from kairyu.engine.core.quant_config import QuantConfig, QuantMethod
    from kairyu.models.moe_parallel import ExpertParallelLoadInfo
    from kairyu.models.parallel import ReplicatedInputRowParallelLinear
    from kairyu.quant.linear import (
        LinearKernel,
        LinearRole,
        LinearSelection,
        linear_factory,
        make_linear,
    )

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()

    class _RowComm:
        def __init__(self) -> None:
            self.rank = rank
            self.world_size = world_size

        @staticmethod
        def tensor_all_reduce(partial):
            return partial

    comm = _RowComm()
    factory = linear_factory(
        QuantConfig(QuantMethod.NVFP4, weight_bits=4, group_size=16),
        dtype=torch.bfloat16,
        tensor_parallel_size=world_size,
        tensor_parallel_rank=rank,
    )
    for layer_index in range(94):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        name = f"model.layers.{layer_index}.self_attn.o_proj"
        module = make_linear(
            factory,
            16,
            4,
            False,
            qualified_name=name,
            role=LinearRole.ATTENTION_OUTPUT,
            layer_index=layer_index,
            shard_dim=1,
        )
        module.linear_selection = LinearSelection(
            QuantConfig(QuantMethod.NVFP4, weight_bits=4, group_size=16),
            LinearKernel.FLASHINFER_NVFP4,
            "unit-test",
        )
        ReplicatedInputRowParallelLinear(module, comm)
        layer.self_attn.o_proj = module
        model.model.layers.append(layer)
    if malformed_selection:
        del model.model.layers[0].self_attn.o_proj.linear_selection
    experts_per_rank = 128 // world_size
    owned = tuple(
        range(rank * experts_per_rank, (rank + 1) * experts_per_rank)
    )
    load_info = ExpertParallelLoadInfo(
        ep_rank=rank,
        ep_size=world_size,
        owned_expert_indices=owned,
        quantization_source="hf_quant_config.json",
        quantization_method="nvfp4",
        kv_cache_quant_algo="FP8",
        producer_name="modelopt",
        producer_version="0.33.0",
        checkpoint_tensor_count=2242,
        rank_loaded_tensor_count=2745,
        auxiliary_kv_scale_count=188,
    )
    return SimpleNamespace(
        _model=model,
        expert_parallel_load_info=load_info,
    )


class _AccountingRunner:
    def execute(self, scheduled, states):
        return {}

    def execute_passive(self, scheduled, states):
        return {}

    def make_sampling_token_packet(self, scheduled, states, sampled=None):
        return torch.full((len(scheduled),), -1, dtype=torch.int64)

    def adopt_sampling_token_packet(self, scheduled, states, packet) -> None:
        pass

    def release(self, request_id: str) -> None:
        pass


def _accounting_snapshot(request_id: str) -> RequestSnapshot:
    return RequestSnapshot(
        request_id=request_id,
        prompt_token_ids=tuple(range(20)),
        computed_prompt=20,
        outputs=(),
        in_flight=0,
        page_ids=(7, 11),
        decode_page_ids=(),
        eos_token_id=None,
        max_new_tokens=1,
        num_cached_tokens=16,
    )


@pytest.mark.parametrize("expert_parallel_size", [2, 4])
def test_ep_correctness_mode_accepts_only_the_declared_topologies(
    expert_parallel_size,
):
    worker_module._validate_ep_correctness_mode(
        expert_parallel_size=expert_parallel_size,
        pipeline_depth=1,
        decode_mode="eager",
        kv_cache_dtype="bfloat16",
        pd_separation=False,
        graph_scratch_page=None,
        dram_kv_tier_capacity_pages=0,
        dram_kv_tier_profile=None,
        speculative=None,
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"expert_parallel_size": 1}, "expert_parallel_size"),
        ({"expert_parallel_size": 8}, "expert_parallel_size"),
        ({"expert_parallel_size": True}, "expert_parallel_size"),
        ({"pipeline_depth": 2}, "pipeline_depth=1"),
        ({"decode_mode": "cuda_graph"}, "CUDA graph"),
        ({"kv_cache_dtype": "auto"}, "bfloat16"),
        ({"pd_separation": True}, "P-D separation"),
        ({"graph_scratch_page": 9}, "CUDA graphs"),
        ({"dram_kv_tier_capacity_pages": 8}, "DRAM KV tier"),
        ({"dram_kv_tier_profile": "/profile.json"}, "DRAM KV tier"),
        ({"speculative": "ngram"}, "speculative decoding"),
    ],
)
def test_ep_correctness_mode_rejects_unsupported_execution_before_launch(
    override,
    message,
):
    options = {"expert_parallel_size": 2}
    options.update(override)
    with pytest.raises((TypeError, ValueError), match=message):
        worker_module._validate_ep_correctness_mode(**options)


def test_ep_launcher_rejects_unsupported_mode_before_probe_or_spawn(monkeypatch):
    import torch.multiprocessing as mp

    touched: list[str] = []
    monkeypatch.setattr(
        worker_module,
        "_ep_placement",
        lambda *args, **kwargs: touched.append("placement"),
    )
    monkeypatch.setattr(
        mp,
        "spawn",
        lambda *args, **kwargs: touched.append("spawn"),
    )

    with pytest.raises(ValueError, match="CUDA graph"):
        worker_module.DistEPLauncher(
            "/model",
            expert_parallel_size=2,
            num_pages=64,
            page_size=16,
            vocab=["token"],
            decode_mode="cuda_graph",
        )
    assert touched == []


def test_ep_handshake_names_the_topology_and_fails_on_size_mismatch(tmp_path):
    shard_names = _write_ep_checkpoint(tmp_path)
    handshake = worker_module._make_ep_handshake(
        str(tmp_path),
        expert_parallel_size=4,
        num_pages=64,
        page_size=16,
        attention_backend_identity="flashinfer-sm120",
        kv_cache_dtype_resolved="bfloat16",
    )

    assert handshake["parallelism"] == "expert_parallel"
    assert handshake["expert_parallel_size"] == 4
    assert handshake["attention_placement"] == "replicated"
    assert handshake["attention_output_placement"] == "row_parallel"
    assert handshake["attention_output_parallel_size"] == 4
    assert handshake["attention_output_partial_dtype"] == "bfloat16"
    assert handshake["execution_mode"] == "replicated-attention-correctness"
    assert handshake["kv_cache_dtype_requested"] == "bfloat16"
    assert "tensor_parallel_size" not in handshake
    identity = handshake["checkpoint_identity"]
    assert identity["resolved_model_path"] == str(tmp_path.resolve())
    assert tuple(name for name, _digest in identity["metadata_sha256"]) == (
        "config.json",
        "hf_quant_config.json",
        "model.safetensors.index.json",
    )
    assert tuple(name for name, _size in identity["indexed_shards"]) == shard_names
    worker_module._validate_ep_handshake(
        handshake,
        str(tmp_path),
        expert_parallel_size=4,
        num_pages=64,
        page_size=16,
        attention_backend_identity="flashinfer-sm120",
        kv_cache_dtype_resolved="bfloat16",
    )
    with pytest.raises(RuntimeError, match="EP worker mismatch"):
        worker_module._validate_ep_handshake(
            handshake,
            str(tmp_path),
            expert_parallel_size=2,
            num_pages=64,
            page_size=16,
            attention_backend_identity="flashinfer-sm120",
            kv_cache_dtype_resolved="bfloat16",
        )


def test_ep_checkpoint_identity_reads_metadata_but_only_stats_shards(
    tmp_path,
    monkeypatch,
):
    shard_names = _write_ep_checkpoint(tmp_path)
    original_read_bytes = Path.read_bytes
    read_names: list[str] = []

    def guarded_read_bytes(path):
        read_names.append(path.name)
        if path.suffix == ".safetensors":
            raise AssertionError("startup identity must not read shard payloads")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    identity = worker_module._ep_checkpoint_identity(str(tmp_path))

    assert read_names == [
        "config.json",
        "hf_quant_config.json",
        "model.safetensors.index.json",
    ]
    assert identity["indexed_shards"] == tuple(
        (name, index)
        for index, name in enumerate(shard_names, start=1)
    )


@pytest.mark.parametrize("mutation", ["metadata", "shard-size"])
def test_ep_handshake_detects_lightweight_checkpoint_identity_drift(
    tmp_path,
    mutation,
):
    shard_names = _write_ep_checkpoint(tmp_path)
    handshake = worker_module._make_ep_handshake(
        str(tmp_path),
        expert_parallel_size=2,
        num_pages=64,
        page_size=16,
        attention_backend_identity="flashinfer-sm120",
        kv_cache_dtype_resolved="bfloat16",
    )
    if mutation == "metadata":
        (tmp_path / "hf_quant_config.json").write_text(
            '{"producer":{"name":"modelopt","version":"changed"}}'
        )
    else:
        with (tmp_path / shard_names[0]).open("ab") as stream:
            stream.write(b"x")

    with pytest.raises(RuntimeError, match="EP worker mismatch"):
        worker_module._validate_ep_handshake(
            handshake,
            str(tmp_path),
            expert_parallel_size=2,
            num_pages=64,
            page_size=16,
            attention_backend_identity="flashinfer-sm120",
            kv_cache_dtype_resolved="bfloat16",
        )


def test_ep_startup_identity_leaves_same_size_shard_hashing_to_formal_gate(tmp_path):
    shard_names = _write_ep_checkpoint(tmp_path)
    handshake = worker_module._make_ep_handshake(
        str(tmp_path),
        expert_parallel_size=2,
        num_pages=64,
        page_size=16,
        attention_backend_identity="flashinfer-sm120",
        kv_cache_dtype_resolved="bfloat16",
    )
    first_shard = tmp_path / shard_names[0]
    first_shard.write_bytes(b"z" * first_shard.stat().st_size)

    # Startup intentionally avoids 134 GB of duplicate I/O. The formal gate
    # hashes every shard at measurement start/end; this seam binds name/size.
    worker_module._validate_ep_handshake(
        handshake,
        str(tmp_path),
        expert_parallel_size=2,
        num_pages=64,
        page_size=16,
        attention_backend_identity="flashinfer-sm120",
        kv_cache_dtype_resolved="bfloat16",
    )


def test_ep_checkpoint_identity_normalizes_the_shared_model_path(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_ep_checkpoint(checkpoint)
    alias = tmp_path / "alias"
    alias.symlink_to(checkpoint, target_is_directory=True)
    from_alias = worker_module._ep_checkpoint_identity(str(alias))
    from_real = worker_module._ep_checkpoint_identity(str(checkpoint))

    assert from_alias == from_real


def test_ep_runner_metadata_never_reports_expert_parallelism_as_tp():
    comms = FakeCommunicator.create_group(2)
    runner = worker_module.DistEPModelRunner(
        comms[0],
        object(),
        expert_parallel_size=2,
    )

    assert runner.parallelism_metadata() == {
        "parallelism": "expert_parallel",
        "expert_parallel_size": 2,
        "attention_placement": "replicated",
        "attention_output_placement": "row_parallel",
        "attention_output_parallel_size": 2,
        "attention_output_partial_dtype": "bfloat16",
        "execution_mode": "replicated-attention-correctness",
        "pipeline_depth": 1,
        "decode_mode": "eager",
        "kv_cache_dtype": "bfloat16",
    }
    assert not hasattr(runner, "tensor_parallel_size")
    assert not isinstance(runner, worker_module.DistTPModelRunner)
    assert not issubclass(worker_module.DistEPLauncher, worker_module.DistTPLauncher)
    with pytest.raises(RuntimeError, match="does not support a DRAM KV tier"):
        runner.available_prefix(("page",))


def test_ep_runner_rejects_a_communicator_with_the_wrong_world_size():
    comms = FakeCommunicator.create_group(2)
    with pytest.raises(ValueError, match="world size"):
        worker_module.DistEPModelRunner(
            comms[0],
            object(),
            expert_parallel_size=4,
        )


@pytest.mark.parametrize("world_size", [2, 4])
def test_ep_kv_accounting_gathers_identical_first_request_views_and_disarms(
    world_size,
):
    control = FakeCommunicator.create_group(world_size)
    model = FakeCommunicator.create_group(world_size)
    local = tuple(_AccountingRunner() for _ in range(world_size))
    driver = worker_module.DistEPModelRunner(
        control[0],
        local[0],
        model[0],
        expert_parallel_size=world_size,
    )
    snapshot = _accounting_snapshot("request-0")

    with ThreadPoolExecutor(max_workers=world_size - 1) as pool:
        workers = tuple(
            pool.submit(
                worker_module.worker_step_loop,
                control[rank],
                local[rank],
                model[rank],
                parallelism_prefix="EP",
            )
            for rank in range(1, world_size)
        )
        started = driver.start_ep_kv_accounting()
        driver.execute(
            (ScheduledChunk("request-0", 4, True, 16),),
            {"request-0": snapshot},
        )
        rows = driver.finish_ep_kv_accounting()
        driver.shutdown()
        assert [worker.result(timeout=2) for worker in workers] == [
            1
        ] * (world_size - 1)

    assert [row["rank"] for row in started] == list(range(world_size))
    assert all(row["capture_action"] == "start" for row in started)
    assert all(row["request_count"] == 0 for row in started)
    assert all(row["capture_error"] is None for row in started)
    assert [row["rank"] for row in rows] == list(range(world_size))
    assert all(row["capture_action"] == "finish" for row in rows)
    assert all(row["overflow"] is False for row in rows)
    assert all(row["capture_error"] is None for row in rows)
    assert all(row["request_count"] == 1 for row in rows)
    assert all(row["prompt_tokens"] == 20 for row in rows)
    assert all(row["cached_tokens"] == 16 for row in rows)
    invariant = {
        key: value
        for key, value in rows[0].items()
        if key not in {"rank"}
    }
    assert all(
        {
            key: value
            for key, value in row.items()
            if key not in {"rank"}
        }
        == invariant
        for row in rows[1:]
    )
    observation = rows[0]["observations"][0]
    assert observation == {
        "sequence": 0,
        "request_id": "request-0",
        "prompt_tokens": 20,
        "cached_tokens": 16,
        "prefill_start": 16,
        "first_num_tokens": 4,
        "first_is_prefill": True,
        "prompt_token_ids_sha256": worker_module._json_sha256(list(range(20))),
        "page_count": 2,
        "page_ids_sha256": worker_module._json_sha256([7, 11]),
    }


def test_ep_kv_accounting_uses_scheduler_cache_truth_for_prefill_start():
    cache = RadixKVCache(num_pages=32, page_size=4)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=64,
        max_num_seqs=1,
    )
    shared = tuple(range(8))
    scheduler.add_request(
        EngineRequest(
            "seed",
            prompt_token_ids=shared + tuple(range(100, 104)),
            max_new_tokens=1,
        )
    )
    assert scheduler.schedule().scheduled[0].num_tokens == 12
    assert scheduler.update({"seed": 42}) == ("seed",)

    scheduler.add_request(
        EngineRequest(
            "reuse",
            prompt_token_ids=shared + tuple(range(200, 204)),
            max_new_tokens=1,
        )
    )
    scheduled = scheduler.schedule().scheduled
    recorder = worker_module._EPKVAccountingRecorder()
    recorder.start()
    recorder.observe(scheduled, scheduler.states)
    row = recorder.snapshot(rank=0, world_size=4, action="finish")

    assert row["capture_error"] is None
    assert row["request_count"] == 1
    assert row["cached_tokens"] == 8
    assert row["observations"][0]["prefill_start"] == 8
    assert row["observations"][0]["first_num_tokens"] == 4
    assert scheduled[0].position == 0


def test_ep_kv_accounting_recomputes_last_token_on_a_full_prompt_hit():
    cache = RadixKVCache(num_pages=32, page_size=4)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=64,
        max_num_seqs=1,
    )
    prompt = tuple(range(12))
    scheduler.add_request(
        EngineRequest("seed", prompt_token_ids=prompt, max_new_tokens=1)
    )
    scheduler.schedule()
    assert scheduler.update({"seed": 42}) == ("seed",)

    scheduler.add_request(
        EngineRequest("exact", prompt_token_ids=prompt, max_new_tokens=1)
    )
    scheduled = scheduler.schedule().scheduled
    recorder = worker_module._EPKVAccountingRecorder()
    recorder.start()
    recorder.observe(scheduled, scheduler.states)
    row = recorder.snapshot(rank=0, world_size=4, action="finish")

    assert row["capture_error"] is None
    assert row["cached_tokens"] == len(prompt)
    assert row["observations"][0]["prefill_start"] == len(prompt) - 1
    assert row["observations"][0]["first_num_tokens"] == 1


def test_ep_kv_accounting_observation_failure_is_reported_after_collective():
    control = FakeCommunicator.create_group(2)
    model = FakeCommunicator.create_group(2)
    local = (_AccountingRunner(), _AccountingRunner())
    driver = worker_module.DistEPModelRunner(
        control[0],
        local[0],
        model[0],
        expert_parallel_size=2,
    )
    malformed = _accounting_snapshot("request-0")

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(
            worker_module.worker_step_loop,
            control[1],
            local[1],
            model[1],
            parallelism_prefix="EP",
        )
        driver.start_ep_kv_accounting()
        driver.execute(
            (ScheduledChunk("request-0", 4, False),),
            {"request-0": malformed},
        )
        rows = driver.finish_ep_kv_accounting()
        driver.shutdown()
        assert worker.result(timeout=2) == 1

    assert [row["request_count"] for row in rows] == [0, 0]
    assert all(
        isinstance(row["capture_error"], str)
        and "malformed first-prefill allocation" in row["capture_error"]
        for row in rows
    )


def test_ep_kv_accounting_finish_disarms_and_bounds_seen_state():
    recorder = worker_module._EPKVAccountingRecorder()
    recorder.start()
    recorder.snapshot(rank=0, world_size=4, action="finish")
    recorder.observe(
        (ScheduledChunk("missing", 1, True),),
        {},
    )
    recorder.start()
    row = recorder.snapshot(rank=0, world_size=4, action="finish")

    assert row["request_count"] == 0
    assert row["capture_error"] is None
    assert row["overflow"] is False


def test_ep_kernel_inventory_gathers_rank_local_runtime_evidence_without_names():
    control = FakeCommunicator.create_group(2)
    model = FakeCommunicator.create_group(2)
    local = tuple(_kernel_probe_runner(rank, 2) for rank in range(2))
    driver = worker_module.DistEPModelRunner(
        control[0],
        local[0],
        model[0],
        expert_parallel_size=2,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(
            worker_module.worker_step_loop,
            control[1],
            local[1],
            model[1],
            parallelism_prefix="EP",
        )
        rows = driver.ep_kernel_inventory_metadata()
        driver.shutdown()
        assert worker.result(timeout=2) == 0

    assert [row["rank"] for row in rows] == [0, 1]
    assert all(set(row) == worker_module._EP_KERNEL_INVENTORY_FIELDS for row in rows)
    assert all(row["projection_count"] == 94 for row in rows)
    assert all(
        row["kernel_counts"] == {"nvfp4:flashinfer_nvfp4": 94}
        for row in rows
    )
    for rank, row in enumerate(rows):
        assert row["attention_output"] == {
            "backend": "flashinfer.mm_fp4",
            "placement": "row_parallel",
            "parallel_size": 2,
            "rank": rank,
            "shard_axis": 1,
            "global_in_features": 32,
            "local_in_features": 16,
            "out_features": 4,
            "partial_dtype": "bfloat16",
            "block_count": 94,
            "successful_forward_block_count": 0,
        }
    assert all(row["fused_moe"] is None for row in rows)
    assert all(
        row["meta_count"] == {"parameters": 0, "buffers": 0, "total": 0}
        for row in rows
    )
    for rank, row in enumerate(rows):
        names = sorted(
            f"model.layers.{layer}.self_attn.o_proj" for layer in range(94)
        )
        assert row["projection_inventory_sha256"] == worker_module._json_sha256(names)
        assert "projection_names" not in row
        assert row["load_info"] == {
            "ep_rank": rank,
            "ep_size": 2,
            "owned_expert_count": 64,
            "owned_expert_indices_sha256": worker_module._json_sha256(
                list(range(rank * 64, (rank + 1) * 64))
            ),
            "first_owned_expert": rank * 64,
            "last_owned_expert": (rank + 1) * 64 - 1,
            "quantization_source": "hf_quant_config.json",
            "quantization_method": "nvfp4",
            "kv_cache_quant_algo": "FP8",
            "producer_name": "modelopt",
            "producer_version": "0.33.0",
            "checkpoint_tensor_count": 2242,
            "rank_loaded_tensor_count": 2745,
            "auxiliary_kv_scale_count": 188,
        }


def test_ep_kernel_inventory_measures_unresolved_meta_state():
    (control, _peer) = FakeCommunicator.create_group(2)
    local = _kernel_probe_runner(0, 2)
    local._model.register_parameter(
        "unresolved_parameter",
        torch.nn.Parameter(torch.empty(1, device="meta")),
    )
    local._model.register_buffer(
        "unresolved_buffer",
        torch.empty(1, device="meta"),
    )

    row = worker_module._ep_kernel_inventory_row(control, local)

    assert row["kernel_counts"] == {"nvfp4:flashinfer_nvfp4": 94}
    assert row["meta_count"] == {"parameters": 1, "buffers": 1, "total": 2}


def test_ep_kernel_inventory_rejects_attention_output_kernel_fallback():
    from kairyu.engine.core.quant_config import QuantConfig, QuantMethod
    from kairyu.quant.linear import LinearKernel, LinearSelection

    (control, _peer) = FakeCommunicator.create_group(2)
    local = _kernel_probe_runner(0, 2)
    local._model.model.layers[0].self_attn.o_proj.linear_selection = LinearSelection(
        QuantConfig(QuantMethod.NVFP4, weight_bits=4, group_size=16),
        LinearKernel.CPU_REFERENCE,
        "unit-test-fallback",
    )

    with pytest.raises(RuntimeError, match="does not select FlashInfer NVFP4"):
        worker_module._ep_kernel_inventory_row(control, local)


def test_ep_kernel_inventory_counts_only_successful_row_reductions():
    (control, _peer) = FakeCommunicator.create_group(2)
    local = _kernel_probe_runner(0, 2)
    output = local._model.model.layers[0].self_attn.o_proj

    before = worker_module._ep_kernel_inventory_row(control, local)
    assert before["attention_output"]["successful_forward_block_count"] == 0

    hidden = torch.zeros(2, 32, dtype=torch.bfloat16)
    result = output(hidden)

    assert result.shape == (2, 4)
    after = worker_module._ep_kernel_inventory_row(control, local)
    assert after["attention_output"]["successful_forward_block_count"] == 1


def test_ep_kernel_inventory_rank_failure_is_gathered_without_hanging():
    control = FakeCommunicator.create_group(2, timeout_s=0.2)
    model = FakeCommunicator.create_group(2, timeout_s=0.2)
    local = (
        _kernel_probe_runner(0, 2),
        _kernel_probe_runner(1, 2, malformed_selection=True),
    )
    driver = worker_module.DistEPModelRunner(
        control[0],
        local[0],
        model[0],
        expert_parallel_size=2,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(
            worker_module.worker_step_loop,
            control[1],
            local[1],
            model[1],
            parallelism_prefix="EP",
        )
        with pytest.raises(
            RuntimeError,
            match=r"rank 1: RuntimeError: .* has no quant method",
        ):
            driver.ep_kernel_inventory_metadata()
        assert isinstance(driver.fatal_error, RuntimeError)
        control[0].broadcast(None, src=0)
        assert worker.result(timeout=2) == 0


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (
            lambda reply: (reply,),
            "gathered malformed rank count",
        ),
        (
            lambda reply: (
                reply,
                worker_module._EPKernelInventoryReply(
                    rank=1,
                    row={**reply.row, "rank": 1, "projection_count": "two"},
                ),
            ),
            "malformed projection_count",
        ),
        (
            lambda reply: (
                reply,
                worker_module._EPKernelInventoryReply(
                    rank=1,
                    row={
                        **reply.row,
                        "rank": 1,
                        "attention_output": {
                            **reply.row["attention_output"],
                            "rank": 1,
                            "local_in_features": 17,
                        },
                    },
                ),
            ),
            "malformed attention_output",
        ),
    ],
)
def test_ep_kernel_inventory_driver_fails_closed_on_malformed_replies(
    monkeypatch,
    corrupt,
    message,
):
    control = FakeCommunicator.create_group(2)
    model = FakeCommunicator.create_group(2)
    local = _kernel_probe_runner(0, 2)
    driver = worker_module.DistEPModelRunner(
        control[0],
        local,
        model[0],
        expert_parallel_size=2,
    )
    monkeypatch.setattr(control[0], "all_gather", lambda reply: corrupt(reply))

    with pytest.raises(RuntimeError, match=message):
        driver.ep_kernel_inventory_metadata()
    assert isinstance(driver.fatal_error, RuntimeError)


def test_ep_launcher_exposes_the_runner_kernel_inventory_method():
    expected = ({"rank": 0}, {"rank": 1})
    launcher = object.__new__(worker_module.DistEPLauncher)
    launcher.runner = SimpleNamespace(
        ep_kernel_inventory_metadata=lambda: expected,
    )

    assert launcher.ep_kernel_inventory_metadata() is expected


@pytest.mark.parametrize("rank", [0, 3])
def test_build_ep_runner_replicates_attention_and_binds_rank_local_experts(
    monkeypatch,
    rank,
):
    from kairyu.engine.core import attention as attention_module
    from kairyu.engine.core import (
        attention_selector,
        hw_profile,
        kv_pool,
        model_runner,
        sampler,
    )
    from kairyu.models import moe_parallel

    placement = worker_module._EPPlacement(
        f"cuda:{rank}",
        torch.bfloat16,
        "nccl",
    )
    profile = SimpleNamespace(
        arch="cuda",
        device_count=8,
        device_name=f"gpu-{rank}",
        sm=120,
        kernel_tier="fa2",
    )
    decision = SimpleNamespace(
        requested="auto",
        resolved="flashinfer",
        source="hw_profile",
        components={
            "prefill": "flashinfer",
            "decode": "flashinfer",
            "kv_mode": "paged-direct",
        },
        architecture={
            "arch": "cuda",
            "device_name": f"gpu-{rank}",
            "sm": 120,
            "kernel_tier": "fa2",
        },
    )
    backend = SimpleNamespace(selection_decision=decision)
    full_config = SimpleNamespace(
        num_hidden_layers=94,
        kv_cache_num_heads=4,
        kv_cache_head_dim=128,
    )
    load_info = SimpleNamespace(
        ep_rank=rank,
        ep_size=4,
        owned_expert_indices=tuple(range(rank * 32, (rank + 1) * 32)),
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr(hw_profile, "probe", lambda device: profile)
    monkeypatch.setattr(
        attention_module,
        "select_backend",
        lambda actual, *, device: backend,
    )
    monkeypatch.setattr(
        attention_selector,
        "attention_backend_identity",
        lambda actual: "flashinfer-sm120",
    )

    def fake_build_ep_model(
        model_dir,
        ep_size,
        ep_rank,
        comm,
        *,
        dtype,
        device,
        attention_backend,
    ):
        seen["model"] = {
            "model_dir": model_dir,
            "ep_size": ep_size,
            "ep_rank": ep_rank,
            "comm": comm,
            "dtype": dtype,
            "device": device,
            "attention_backend": attention_backend,
        }
        return "rank-local-model", full_config, load_info

    def fake_pool(**kwargs):
        seen["pool"] = kwargs
        return "rank-local-pool"

    class FakePagedModelRunner:
        def __init__(self, model, pool, **kwargs):
            seen["runner"] = {"model": model, "pool": pool, **kwargs}

    monkeypatch.setattr(moe_parallel, "build_ep_model", fake_build_ep_model)
    monkeypatch.setattr(kv_pool, "PagedKVPool", fake_pool)
    monkeypatch.setattr(model_runner, "PagedModelRunner", FakePagedModelRunner)
    monkeypatch.setattr(
        sampler,
        "Sampler",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    comm = object()
    runner, returned_config, returned_info = worker_module.build_ep_runner(
        "/model",
        expert_parallel_size=4,
        rank=rank,
        comm=comm,
        num_pages=128,
        page_size=16,
        vocab=["token"],
        placement=placement,
    )

    assert seen["model"] == {
        "model_dir": "/model",
        "ep_size": 4,
        "ep_rank": rank,
        "comm": comm,
        "dtype": torch.bfloat16,
        "device": f"cuda:{rank}",
        "attention_backend": backend,
    }
    assert seen["pool"] == {
        "num_layers": 94,
        "num_pages": 128,
        "page_size": 16,
        # Attention is replicated: EP must not divide the four KV heads.
        "num_kv_heads": 4,
        "head_dim": 128,
        "dtype": torch.bfloat16,
        "device": f"cuda:{rank}",
    }
    assert seen["runner"]["sampling_owner"] is (rank == 0)
    assert (seen["runner"]["sampler"] is not None) is (rank == 0)
    assert runner.parallelism == "expert_parallel"
    assert runner.expert_parallel_size == 4
    assert runner.expert_parallel_rank == rank
    assert runner.attention_placement == "replicated"
    assert runner.attention_output_placement == "row_parallel"
    assert runner.attention_output_parallel_size == 4
    assert runner.attention_output_partial_dtype == "bfloat16"
    assert runner.kv_cache_dtype_requested == "bfloat16"
    assert runner.kv_cache_dtype_resolved == "bfloat16"
    assert runner.dram_kv_tier is None
    assert returned_config is full_config
    assert returned_info is load_info


def test_ep_placement_fails_closed_without_enough_cuda_devices(monkeypatch):
    from kairyu.engine.core import hw_profile

    monkeypatch.setattr(
        hw_profile,
        "probe",
        lambda: SimpleNamespace(arch="cuda", device_count=2),
    )
    with pytest.raises(RuntimeError, match="expert_parallel_size=4"):
        worker_module._ep_placement(4, 0)
