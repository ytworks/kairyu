"""Production diagnostics specific to request-owned attention-DP EP4."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kairyu.engine.core import worker as worker_module
from kairyu.engine.core.comm import FakeCommunicator
from kairyu.engine.core.quant_config import QuantConfig, QuantMethod
from kairyu.models.moe_parallel import EpMoeBlock, ExpertParallelLoadInfo
from kairyu.quant.linear import (
    LinearKernel,
    LinearRole,
    LinearSelection,
    linear_factory,
    make_linear,
)


class _SamplingRunner:
    def __init__(self, rank: int, *, sampler_present: bool = True) -> None:
        self.sampling_owner = True
        self._sampler = object() if sampler_present else None
        self._device = f"cuda:{rank}"

    def release(self, _request_id: str) -> None:
        pass


def test_attention_dp_sampling_diagnostics_accept_every_request_owner_sampler() -> None:
    control = FakeCommunicator.create_group(4)
    model = FakeCommunicator.create_group(4)
    local = tuple(_SamplingRunner(rank) for rank in range(4))
    driver = worker_module.DistEPModelRunner(
        control[0],
        local[0],
        model[0],
        expert_parallel_size=4,
        attention_dp=True,
    )

    with ThreadPoolExecutor(max_workers=3) as executor:
        workers = tuple(
            executor.submit(
                worker_module.worker_step_loop,
                control[rank],
                local[rank],
                model[rank],
                parallelism_prefix="EP",
            )
            for rank in range(1, 4)
        )
        try:
            rows = driver.sampling_ownership_metadata()
        finally:
            driver.shutdown()
        assert tuple(worker.result(timeout=2) for worker in workers) == (0, 0, 0)

    assert tuple(row["rank"] for row in rows) == (0, 1, 2, 3)
    assert all(row["sampling_owner"] is True for row in rows)
    assert all(row["sampler_present"] is True for row in rows)

    with pytest.raises(RuntimeError, match="rank-0-only sampler contract"):
        worker_module._validate_sampling_ownership_rows(
            rows,
            control_world_size=4,
            model_world_size=4,
            control_backend="fake",
            model_backend="fake",
        )


def test_attention_dp_sampling_diagnostics_reject_missing_owner_sampler() -> None:
    rows = tuple(
        {
            "rank": rank,
            "control_world_size": 4,
            "control_backend": "fake",
            "model_world_size": 4,
            "model_backend": "fake",
            "sampling_owner": True,
            "sampler_present": rank != 2,
            "device": f"cuda:{rank}",
        }
        for rank in range(4)
    )

    with pytest.raises(RuntimeError, match="all-rank request-owner sampler contract"):
        worker_module._validate_sampling_ownership_rows(
            rows,
            control_world_size=4,
            model_world_size=4,
            control_backend="fake",
            model_backend="fake",
            request_owner_sampling=True,
        )


class _AttentionDpComm:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.world_size = 4

    @staticmethod
    def tensor_all_gather(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod
    def tensor_reduce_scatter(tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class _SparseBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList([nn.Identity() for _ in range(128)])

    @staticmethod
    def _route(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(hidden.shape[0], 1, dtype=torch.int64),
            torch.ones(hidden.shape[0], 1, dtype=torch.float32),
        )


def _attention_dp_kernel_runner(rank: int, *, completed: bool = True):
    comm = _AttentionDpComm(rank)
    quant = QuantConfig(QuantMethod.NVFP4, weight_bits=4, group_size=16)
    factory = linear_factory(quant, dtype=torch.bfloat16)
    selection = LinearSelection(
        quant,
        LinearKernel.FLASHINFER_NVFP4,
        "attention-dp-diagnostics-test",
    )
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList()
    layout = (2, 1, 3, 2)
    for layer_index in range(94):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        name = f"model.layers.{layer_index}.self_attn.o_proj"
        output = make_linear(
            factory,
            16,
            4,
            False,
            qualified_name=name,
            role=LinearRole.ATTENTION_OUTPUT,
            layer_index=layer_index,
            shard_dim=None,
        )
        output.linear_selection = selection
        layer.self_attn.o_proj = output
        block = EpMoeBlock(
            _SparseBlock(),
            comm,
            ep_rank=rank,
            ep_size=4,
            attention_dp=True,
        )
        object.__setattr__(
            block,
            "_prepared_nvfp4_moe",
            SimpleNamespace(
                num_global_experts=128,
                local_expert_start=rank * 32,
                fc2_weight=torch.empty(32, 4, 1),
            ),
        )
        if completed:
            object.__setattr__(block, "_fused_nvfp4_executed", True)
            object.__setattr__(block, "_attention_dp_completed_steps", 3)
            object.__setattr__(block, "_attention_dp_last_layout", layout)
            object.__setattr__(
                block,
                "_attention_dp_last_local_rows",
                layout[rank],
            )
            object.__setattr__(block, "_attention_dp_last_padded_rows", 3)
            object.__setattr__(block, "_attention_dp_last_total_rows", 12)
        layer.mlp = block
        model.model.layers.append(layer)

    owned = tuple(range(rank * 32, (rank + 1) * 32))
    return SimpleNamespace(
        _model=model,
        execution_mode=worker_module._EP_ATTENTION_DP_MODE,
        moe_dispatcher=worker_module._EP_ATTENTION_DP_DISPATCHER,
        expert_parallel_load_info=ExpertParallelLoadInfo(
            ep_rank=rank,
            ep_size=4,
            owned_expert_indices=owned,
            quantization_source="hf_quant_config.json",
            quantization_method="nvfp4",
            kv_cache_quant_algo="FP8",
            producer_name="modelopt",
            producer_version="0.33.0",
            checkpoint_tensor_count=146_549,
            rank_loaded_tensor_count=36_000,
            auxiliary_kv_scale_count=188,
        ),
    )


def test_attention_dp_kernel_inventory_validates_replicated_output_and_dispatcher() -> None:
    control = FakeCommunicator.create_group(4)
    runners = tuple(_attention_dp_kernel_runner(rank) for rank in range(4))
    rows = tuple(
        worker_module._ep_kernel_inventory_row(control[rank], runners[rank])
        for rank in range(4)
    )

    validated = worker_module._validate_ep_kernel_inventory_rows(
        rows,
        world_size=4,
    )

    assert validated == rows
    for rank, row in enumerate(rows):
        assert row["attention_output"] == {
            "backend": "flashinfer.mm_fp4",
            "placement": "replicated",
            "parallel_size": 1,
            "rank": rank,
            "shard_axis": None,
            "global_in_features": 16,
            "local_in_features": 16,
            "out_features": 4,
            "partial_dtype": None,
            "block_count": 94,
            "successful_forward_block_count": None,
        }
        fused = row["fused_moe"]
        assert set(fused) == worker_module._EP_ATTENTION_DP_FUSED_MOE_SUMMARY_FIELDS
        assert fused["dispatcher"] == "nvfp4_allgather_reduce_scatter"
        assert fused["activation_collective"] == "tensor_all_gather_many"
        assert fused["combine_collective"] == "tensor_reduce_scatter"
        assert fused["post_combine_all_reduce"] is False
        assert fused["block_count"] == 94
        assert fused["successful_forward_block_count"] == 94
        assert fused["last_rank_row_layout"] == (2, 1, 3, 2)
        assert fused["last_local_real_rows"] == (2, 1, 3, 2)[rank]
        assert fused["last_padded_rows_per_rank"] == 3
        assert fused["total_collective_rows"] == 12

    malformed = tuple(dict(row) for row in rows)
    malformed[2]["fused_moe"] = {
        **malformed[2]["fused_moe"],
        "dispatcher": "replicated_allreduce",
    }
    with pytest.raises(RuntimeError, match="rank 2 has malformed fused_moe"):
        worker_module._validate_ep_kernel_inventory_rows(
            malformed,
            world_size=4,
        )


def test_attention_dp_kernel_inventory_rejects_row_binding_on_replicated_output() -> None:
    control = FakeCommunicator.create_group(4)
    runner = _attention_dp_kernel_runner(0, completed=False)
    output = runner._model.model.layers[0].self_attn.o_proj
    object.__setattr__(output, "_kairyu_row_parallel", object())

    with pytest.raises(RuntimeError, match="unexpectedly has row-parallel bindings"):
        worker_module._ep_kernel_inventory_row(control[0], runner)


def test_attention_dp_assignment_evidence_retains_rolling_tail_and_total_sequence() -> None:
    control = FakeCommunicator.create_group(4)
    model = FakeCommunicator.create_group(4)
    driver = worker_module.DistEPModelRunner(
        control[0],
        _SamplingRunner(0),
        model[0],
        expert_parallel_size=4,
        attention_dp=True,
    )
    limit = worker_module._EP_ATTENTION_DP_ASSIGNMENT_LOG_LIMIT
    total = limit + 7
    assignments = tuple(
        {
            "sequence": sequence,
            "request_id": f"request-{sequence}",
            "owner_rank": sequence % 4,
            "cache_affine": sequence % 3 == 0,
        }
        for sequence in range(total)
    )

    driver._retain_attention_dp_assignments(assignments)
    metadata = driver.attention_dp_ownership_metadata()

    assert metadata["assignment_count"] == total
    assert metadata["assignment_retention_limit"] == limit
    assert metadata["retained_assignment_count"] == limit
    assert metadata["dropped_assignment_count"] == 7
    assert len(metadata["assignments"]) == limit
    assert metadata["assignments"][0]["sequence"] == 7
    assert metadata["assignments"][-1]["sequence"] == total - 1
