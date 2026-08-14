"""CPU-only producer tests for the G4 M-A1 real-engine capture adapter."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kairyu.engine.core.sampling_types import SampledToken
from verification.l1.correctness import g4_ma1_qwen3_235b_nvfp4_bench as gate
from verification.l1.correctness import g4_ma1_qwen3_235b_nvfp4_capture as capture


class _Tokenizer:
    eos_token_id = None

    def encode(self, text: str) -> tuple[int, ...]:
        return (100 + gate.PROMPTS.index(text),)

    def decode(self, token_ids) -> str:
        return " ".join(str(token) for token in token_ids)

    def vocab(self) -> list[str]:
        return [str(index) for index in range(4096)]


def _source() -> dict[str, object]:
    return {
        "commit": "a" * 40,
        "tracked_dirty": False,
        "files": {name: "b" * 64 for name in gate.BOUND_SOURCE_FILES},
    }


def test_source_snapshot_rejects_untracked_bound_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bound = "bound.py"
    (tmp_path / bound).write_text("tracked = False\n", encoding="utf-8")
    monkeypatch.setattr(gate, "BOUND_SOURCE_FILES", (bound,))
    monkeypatch.setattr(capture, "_REPO_ROOT", tmp_path)

    def fake_git(*args: str) -> str:
        if args[0] == "ls-files":
            raise subprocess.CalledProcessError(1, ("git", *args))
        raise AssertionError(args)

    monkeypatch.setattr(capture, "_git", fake_git)

    with pytest.raises(ValueError, match="not tracked by git"):
        capture._source_snapshot()


def test_source_snapshot_checks_all_untracked_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bound = "bound.py"
    (tmp_path / bound).write_text("tracked = True\n", encoding="utf-8")
    monkeypatch.setattr(gate, "BOUND_SOURCE_FILES", (bound,))
    monkeypatch.setattr(capture, "_REPO_ROOT", tmp_path)

    def fake_git(*args: str) -> str:
        if args[0] == "ls-files":
            return bound
        if args[0] == "rev-parse":
            return "a" * 40
        assert args == ("status", "--porcelain", "--untracked-files=all")
        return "?? unrelated-runtime-input"

    monkeypatch.setattr(capture, "_git", fake_git)

    with pytest.raises(ValueError, match="clean bound source commit"):
        capture._source_snapshot()


def _checkpoint() -> dict[str, object]:
    weights = [
        {"name": name, "size": size, "sha256": digest}
        for name, size, digest in gate.EXPECTED_WEIGHT_FILES
    ]
    tokenizer = [
        {"name": name, "size": size, "sha256": digest}
        for name, size, digest in gate.EXPECTED_TOKENIZER_FILES
    ]
    return {
        "model": gate.MODEL,
        "revision": gate.MODEL_REVISION,
        "config_sha256": gate.EXPECTED_CONFIG_SHA256,
        "index_sha256": gate.EXPECTED_INDEX_SHA256,
        "hf_quant_config_sha256": gate.EXPECTED_HF_QUANT_CONFIG_SHA256,
        "index_total_size": gate.EXPECTED_INDEX_TOTAL_SIZE,
        "tensor_count": gate.EXPECTED_TENSOR_COUNT,
        "weight_shards": weights,
        "weights_rollup_sha256": gate.sha256_json(weights),
        "tokenizer_files": tokenizer,
        "tokenizer_rollup_sha256": gate.sha256_json(tokenizer),
        "architecture": dict(gate.EXPECTED_ARCHITECTURE),
        "quantization": dict(gate.EXPECTED_QUANTIZATION),
    }


def _prompts(checkpoint: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for index, text in enumerate(gate.PROMPTS):
        tokens = [100 + index]
        rows.append(
            {
                "schema_version": gate.SCHEMA_VERSION,
                "type": "prompt",
                "prompt_id": f"p{index}",
                "prompt_index": index,
                "text": text,
                "text_sha256": gate.sha256_bytes(text.encode()),
                "token_ids": tokens,
                "token_ids_sha256": gate.sha256_json(tokens),
                "tokenizer_rollup_sha256": checkpoint[
                    "tokenizer_rollup_sha256"
                ],
            }
        )
    return rows


@pytest.fixture
def host_snapshot() -> dict[str, object]:
    checkpoint = _checkpoint()
    value = {
        "schema_version": capture.CAPTURE_SCHEMA,
        "type": capture.HOST_SNAPSHOT_TYPE,
        "prepared_at": "2026-08-01T00:00:00+00:00",
        "image_repo_digest": "kairyu:test@sha256:" + "e" * 64,
        "image_config_id": "sha256:" + "f" * 64,
        "source": _source(),
        "checkpoint": checkpoint,
        "prompts": _prompts(checkpoint),
    }
    assert capture._host_snapshot_valid(value)
    return value


def _environment(arm: str) -> dict[str, object]:
    world_size = gate._arm_world_size(arm)
    reference = gate._arm_stack(arm) == "reference"
    gpus = [
        {
            "rank": rank,
            "local_device_index": rank,
            "uuid": f"GPU-{rank:032x}",
            "pci_bus_id": f"00000000:{rank + 1:02X}:00.0",
            "name": gate.EXPECTED_GPU_NAME,
            "sm": gate.EXPECTED_SM,
            "total_memory_bytes": 96 * 1024**3,
            "numa_node": rank // 2,
        }
        for rank in range(world_size)
    ]
    raw = "GPU0 GPU1 CPU Affinity NUMA Affinity\nfixture"
    packages = {"python": "3.12", "torch": "2.12"}
    packages.update(
        {"tensorrt_llm": "1.2.1"}
        if reference
        else {"kairyu": "0.1.0", "flashinfer": "0.6.14"}
    )
    return {
        "packages": packages,
        "cuda": {"runtime": "13.0", "driver": "590", "nccl": "2.29"},
        "visibility": {
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(world_size))),
            "NVIDIA_VISIBLE_DEVICES": ",".join(row["uuid"] for row in gpus),
            "resolved_rank_uuid_order": [row["uuid"] for row in gpus],
        },
        "gpus": gpus,
        "topology": {
            "gpu_uuid_order": [row["uuid"] for row in gpus],
            "gpu_pci_bus_id_order": [row["pci_bus_id"] for row in gpus],
            "gpu_numa_node_order": [row["numa_node"] for row in gpus],
            "nvidia_smi_topology_raw": raw,
            "nvidia_smi_topology_sha256": gate.sha256_bytes(raw.encode()),
        },
    }


def _checkpoint_view(checkpoint: dict[str, object]) -> dict[str, object]:
    return {
        "kind": "bf16-kv-metadata-overlay",
        "source_model_path": "/models/source",
        "runtime_model_path": "/tmp/runtime",
        "source_hf_quant_config_sha256": gate.EXPECTED_HF_QUANT_CONFIG_SHA256,
        "runtime_hf_quant_config_sha256": (
            gate.EXPECTED_TRT_RUNTIME_HF_QUANT_CONFIG_SHA256
        ),
        "removed_json_path": "quantization.kv_cache_quant_algo",
        "removed_value": "FP8",
        "kv_cache_config_dtype_argument": "auto",
        "effective_kv_cache_dtype": "bfloat16",
        "weight_links": [
            {
                "name": row["name"],
                "symlink_target": f"/models/source/{row['name']}",
                "source_device": 1,
                "source_inode": index + 1,
                "runtime_device": 1,
                "runtime_inode": index + 1,
                "size_bytes": row["size"],
            }
            for index, row in enumerate(checkpoint["weight_shards"])
        ],
    }


def _observation(
    repo_digest: str, config_id: str, source_revision: str | None = None
) -> dict[str, object]:
    return {
        "source": "docker-container-inspect",
        "container_id": "1" * 64,
        "image_repo_digest": repo_digest,
        "image_config_id": config_id,
        "running": True,
        "started_at": "2026-08-01T00:00:01+00:00",
        "inspect_sha256": "2" * 64,
        "source_revision": source_revision,
    }


def _logprobs(selected: int) -> dict[int, float]:
    result = {selected: -0.1}
    candidate = 2000
    while len(result) < gate.TOP_LOGPROBS:
        if candidate != selected:
            result[candidate] = -1.0 - 0.01 * len(result)
        candidate += 1
    return result


def _canonical_tokens(prompt_index: int) -> list[int]:
    first = 1000 + prompt_index * gate.POSITIONS
    return list(range(first, first + gate.POSITIONS))


class _ReferenceParams:
    values: list[dict[str, object]] = []

    def __init__(self, **values) -> None:
        self.values.append(values)
        self.__dict__.update(values)


class _Config:
    def __init__(self, **values) -> None:
        self.values = values


class _ReferenceLLM:
    calls: list[int] = []
    init_values: dict[str, object] = {}
    last_instance = None

    def __init__(self, **values) -> None:
        type(self).init_values = values
        type(self).last_instance = self
        self.shutdown_called = False

    def generate(self, prompts, params, *, use_tqdm):
        assert use_tqdm is False
        self.calls.append(params.max_tokens)
        outputs = []
        for prompt in prompts:
            if isinstance(prompt, str):
                prompt_index = gate.PROMPTS.index(prompt)
                prompt_ids = [100 + prompt_index]
                tokens = _canonical_tokens(prompt_index)
            else:
                prompt_ids = list(prompt)
                prompt_index = prompt_ids[0] - 100
                tokens = [_canonical_tokens(prompt_index)[len(prompt_ids) - 1]]
            completion = SimpleNamespace(
                token_ids=tokens,
                logprobs=[_logprobs(token) for token in tokens],
                finish_reason="length",
            )
            outputs.append(
                SimpleNamespace(
                    outputs=(completion,),
                    prompt_token_ids=tuple(prompt_ids),
                )
            )
        return outputs

    def shutdown(self) -> None:
        self.shutdown_called = True


class _DivergentReferenceLLM(_ReferenceLLM):
    def generate(self, prompts, params, *, use_tqdm):
        assert use_tqdm is False
        self.calls.append(params.max_tokens)
        outputs = []
        for prompt in prompts:
            if isinstance(prompt, str):
                prompt_index = gate.PROMPTS.index(prompt)
                prompt_ids = [100 + prompt_index]
                tokens = _canonical_tokens(prompt_index)
                tokens[1] += 50_000
            else:
                prompt_ids = list(prompt)
                prompt_index = prompt_ids[0] - 100
                tokens = [_canonical_tokens(prompt_index)[len(prompt_ids) - 1]]
            completion = SimpleNamespace(
                token_ids=tokens,
                logprobs=[_logprobs(token) for token in tokens],
                finish_reason="length",
            )
            outputs.append(
                SimpleNamespace(
                    outputs=(completion,),
                    prompt_token_ids=tuple(prompt_ids),
                )
            )
        return outputs


def _projection_names(arm: str) -> list[list[str]]:
    world_size = gate._arm_world_size(arm)
    experts_per_rank = gate.NUM_EXPERTS // world_size
    names_by_rank = []
    for rank in range(world_size):
        names = [
            f"model.layers.{layer}.self_attn.{projection}"
            for layer in range(gate.NUM_LAYERS)
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        ]
        names.extend(
            f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
            for layer in range(gate.NUM_LAYERS)
            for expert in range(
                rank * experts_per_rank,
                (rank + 1) * experts_per_rank,
            )
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
        names_by_rank.append(names)
    return names_by_rank


def _patch_common(monkeypatch, host_snapshot) -> dict[str, int]:
    calls = {"checkpoint": 0}

    def checkpoint_snapshot(_path):
        calls["checkpoint"] += 1
        return copy.deepcopy(host_snapshot["checkpoint"])

    monkeypatch.setattr(capture, "_environment", _environment)
    monkeypatch.setattr(capture, "_checkpoint_snapshot", checkpoint_snapshot)
    monkeypatch.setattr(
        capture,
        "_reference_overlay",
        lambda _source, _runtime: _checkpoint_view(host_snapshot["checkpoint"]),
    )
    monkeypatch.setattr(
        capture,
        "_ownership_rows",
        lambda **_kw: (
            [],
            _projection_names(_kw["arm"]),
        ),
    )
    return calls


def test_reference_arm_produces_64_plus_16_waves(
    monkeypatch: pytest.MonkeyPatch,
    host_snapshot: dict[str, object],
) -> None:
    _ReferenceLLM.calls = []
    _ReferenceParams.values = []
    calls = _patch_common(monkeypatch, host_snapshot)
    fragment = capture.capture_reference_arm(
        model_path=Path("/models/source"),
        world_size=2,
        host_snapshot=host_snapshot,
        bindings={
            "LLM": _ReferenceLLM,
            "SamplingParams": _ReferenceParams,
            "KvCacheConfig": _Config,
            "MoeConfig": _Config,
            "Nvfp4GemmConfig": _Config,
            "version": "1.2.1",
        },
        container_observation=_observation(
            gate.REFERENCE_RUNTIME["image_repo_digest"],
            gate.REFERENCE_RUNTIME["image_config_id"],
        ),
    )

    assert _ReferenceLLM.calls == [gate.POSITIONS, *([1] * gate.POSITIONS)]
    init = _ReferenceLLM.init_values
    assert init["tensor_parallel_size"] == 2
    assert init["moe_tensor_parallel_size"] == 1
    assert init["moe_expert_parallel_size"] == 2
    assert init["disable_overlap_scheduler"] is True
    assert init["cuda_graph_config"] is None
    assert init["attn_backend"] == "TRTLLM"
    assert init["kv_cache_config"].values == {
        "enable_block_reuse": False,
        "enable_partial_reuse": False,
        "copy_on_partial_reuse": False,
        "max_tokens": gate.KV_CACHE_CAPACITY_TOKENS,
        "free_gpu_memory_fraction": (
            gate.REFERENCE_KV_FREE_GPU_MEMORY_FRACTION
        ),
        "dtype": "auto",
    }
    assert init["moe_config"].values["backend"] == "CUTLASS"
    assert init["nvfp4_gemm_config"].values == {"allowed_backends": ["cutlass"]}
    assert all(values["top_k"] == 0 for values in _ReferenceParams.values)
    assert _ReferenceParams.values[0]["add_special_tokens"] is True
    assert all(
        values["add_special_tokens"] is False
        for values in _ReferenceParams.values[1:]
    )
    assert len(fragment["free_run"]) == gate.PROMPT_COUNT
    assert len(fragment["teacher"]) == gate.EXPECTED_POSITIONS
    assert calls["checkpoint"] == 2
    assert capture._fragment_valid(
        fragment, arm="reference_ep2", host_snapshot=host_snapshot
    )


def test_reference_teacher_rollout_is_independent_of_free_decode(
    monkeypatch: pytest.MonkeyPatch,
    host_snapshot: dict[str, object],
) -> None:
    _DivergentReferenceLLM.calls = []
    _patch_common(monkeypatch, host_snapshot)
    fragment = capture.capture_reference_arm(
        model_path=Path("/models/source"),
        world_size=2,
        host_snapshot=host_snapshot,
        bindings={
            "LLM": _DivergentReferenceLLM,
            "SamplingParams": _ReferenceParams,
            "KvCacheConfig": _Config,
            "MoeConfig": _Config,
            "Nvfp4GemmConfig": _Config,
            "version": "1.2.1",
        },
        container_observation=_observation(
            gate.REFERENCE_RUNTIME["image_repo_digest"],
            gate.REFERENCE_RUNTIME["image_config_id"],
        ),
    )

    assert fragment["free_run"][0]["output_token_ids"][1] == 51_001
    canonical = capture._reference_teacher_canonical(
        reference_fragment=fragment,
        prompt_tokens=[row["token_ids"] for row in host_snapshot["prompts"]],
    )
    assert canonical == [
        _canonical_tokens(index) for index in range(gate.PROMPT_COUNT)
    ]
    assert len(fragment["teacher"]) == gate.EXPECTED_POSITIONS
    assert capture._fragment_valid(
        fragment, arm="reference_ep2", host_snapshot=host_snapshot
    )


class _BatchRunner:
    def __init__(self, world_size: int = 2) -> None:
        self.first_scheduled = None
        self.world_size = world_size

    def execute(self, scheduled, states):
        if self.first_scheduled is None:
            self.first_scheduled = len(scheduled)
        result = {}
        for chunk in scheduled:
            state = states[chunk.request_id]
            if not chunk.is_prefill or state.prefill_done:
                prompt_index = state.prompt_token_ids[0] - 100
                if not 0 <= prompt_index < gate.PROMPT_COUNT:
                    prompt_index = state.prompt_token_ids[0] - 1
                selected = _canonical_tokens(prompt_index)[len(state.outputs)]
                result[chunk.request_id] = (
                    SampledToken(
                        selected,
                        -0.1,
                        tuple(_logprobs(selected).items()),
                    ),
                )
        return result

    def release(self, _request_id: str) -> None:
        pass

    def sampling_ownership_metadata(self):
        return tuple(
            {
                "rank": rank,
                "control_world_size": self.world_size,
                "control_backend": "gloo",
                "model_world_size": self.world_size,
                "model_backend": "nccl",
                "sampling_owner": rank == 0,
                "sampler_present": rank == 0,
                "device": f"cuda:{rank}",
            }
            for rank in range(self.world_size)
        )


def test_kairyu_batch_submits_all_64_before_first_step() -> None:
    runner = _BatchRunner()
    outputs = capture._run_kairyu_batch(
        tokenizer=_Tokenizer(),
        runner=runner,
        prompts=[[index + 1] for index in range(gate.PROMPT_COUNT)],
        max_tokens=1,
        num_pages=128,
        page_size=16,
        request_prefix="fixture",
    )

    assert runner.first_scheduled == gate.PROMPT_COUNT
    assert len(outputs) == gate.PROMPT_COUNT
    rows = capture._teacher_rows_from_outputs(
        arm="kairyu_ep2",
        position=0,
        outputs=outputs,
        prompt_tokens=[[index + 1] for index in range(gate.PROMPT_COUNT)],
        canonical=[
            _canonical_tokens(index) for index in range(gate.PROMPT_COUNT)
        ],
    )
    assert len(rows) == gate.PROMPT_COUNT


def _ownership_fixture(arm: str) -> list[dict[str, object]]:
    world_size = gate._arm_world_size(arm)
    experts_per_rank = gate.NUM_EXPERTS // world_size
    fully_replicated_bytes = 4_000
    total_expert_bytes = 8_000
    attention_output_full_bytes = 4_000
    attention_output_shards = []
    for rank in range(world_size):
        shard = {
            "placement": "row_parallel",
            "checkpoint_axis": 1,
            "rank": rank,
            "world_size": world_size,
            "tensor_count": gate.ATTENTION_OUTPUT_SHARDED_TENSOR_COUNT,
            "tensor_names_sha256": "6" * 64,
            "full_geometry_sha256": "7" * 64,
            "full_bytes": attention_output_full_bytes,
            "slice_start_numerator": rank,
            "slice_end_numerator": rank + 1,
            "slice_denominator": world_size,
            "rank_slice_bytes": attention_output_full_bytes // world_size,
        }
        shard["rank_slice_sha256"] = gate._attention_output_rank_slice_sha256(
            shard
        )
        attention_output_shards.append(shard)
    partition_sha256 = gate.sha256_json(
        [row["rank_slice_sha256"] for row in attention_output_shards]
    )
    for shard in attention_output_shards:
        shard["partition_sha256"] = partition_sha256
    rows = []
    for rank in range(world_size):
        owned = list(
            range(rank * experts_per_rank, (rank + 1) * experts_per_rank)
        )
        owned_expert_bytes = total_expert_bytes // world_size
        attention_output_slice_bytes = attention_output_full_bytes // world_size
        rows.append(
            {
                "schema_version": gate.SCHEMA_VERSION,
                "type": "ownership",
                "arm": arm,
                "rank": rank,
                "world_size": world_size,
                "layers": gate.NUM_LAYERS,
                "num_experts": gate.NUM_EXPERTS,
                "owned_expert_indices": owned,
                "owned_expert_tensor_count": (
                    12 * gate.NUM_LAYERS * experts_per_rank
                ),
                "owned_expert_tensor_names_sha256": gate.sha256_json(
                    [arm, rank, "owned"]
                ),
                "owned_expert_bytes": owned_expert_bytes,
                "fully_replicated_tensor_count": (
                    gate.FULLY_REPLICATED_TENSOR_COUNT
                ),
                "fully_replicated_bytes": fully_replicated_bytes,
                "fully_replicated_sha256": "2" * 64,
                "attention_output_shard": attention_output_shards[rank],
                "rank_loaded_tensor_count": (
                    gate.LOADED_NON_EXPERT_TENSOR_COUNT
                    + 12 * gate.NUM_LAYERS * experts_per_rank
                ),
                "rank_loaded_bytes": (
                    fully_replicated_bytes
                    + owned_expert_bytes
                    + attention_output_slice_bytes
                ),
                "loaded_tensor_names_sha256": gate.sha256_json(
                    [arm, rank, "loaded"]
                ),
                "dense_replication_sha256": "3" * 64,
                "router_replication_sha256": "4" * 64,
                "shared_expert_replication_sha256": "5" * 64,
                "remote_expert_tensor_reads": 0,
                "unresolved_meta_tensors": 0,
                "evidence_source": "checkpoint-index-and-build-contract",
                "load_contract": {
                    "quantization_source": "hf_quant_config.json",
                    "quantization_method": "nvfp4",
                    "checkpoint_kv_cache_quant_algo": "FP8",
                    "runtime_kv_cache_dtype": "bfloat16",
                    "producer_name": "modelopt",
                    "producer_version": "0.33.0",
                    "checkpoint_tensor_count": gate.EXPECTED_TENSOR_COUNT,
                    "auxiliary_kv_scale_count": 188,
                    "ownership": "contiguous-global-expert-blocks",
                },
            }
        )
    return rows


class _Launcher:
    def __init__(
        self,
        runner: _BatchRunner,
        *,
        arm: str,
        ownership: list[dict[str, object]],
        projections: list[list[str]],
    ) -> None:
        self.runner = runner
        self.arm = arm
        self.ownership = ownership
        self.projections = projections
        self.expert_parallel_load_info = object()
        self.attention_backend_decision = SimpleNamespace(
            requested="auto",
            resolved="flashinfer",
            components={
                "prefill": "flashinfer",
                "decode": "flashinfer",
                "kv_mode": "paged-direct",
            },
        )
        self.attention_backend_identity = "fixture-attention"
        self.shutdown_called = False

    def parallelism_metadata(self):
        return {
            "parallelism": "expert_parallel",
            "expert_parallel_size": gate._arm_world_size(self.arm),
            "attention_placement": "replicated",
            "attention_output_placement": "row_parallel",
            "attention_output_parallel_size": gate._arm_world_size(self.arm),
            "attention_output_partial_dtype": "bfloat16",
            "execution_mode": "replicated-attention-correctness",
            "pipeline_depth": 1,
            "decode_mode": "eager",
            "kv_cache_dtype": "bfloat16",
        }

    def shutdown(self) -> None:
        self.shutdown_called = True

    def ep_kernel_inventory_metadata(self):
        return _kernel_probe_fixture(self.projections, self.ownership)


@pytest.mark.parametrize("world_size", gate.WORLD_SIZES)
def test_kairyu_arm_uses_reference_prefix_and_fresh_wave_caches(
    monkeypatch: pytest.MonkeyPatch,
    host_snapshot: dict[str, object],
    world_size: int,
) -> None:
    _ReferenceLLM.calls = []
    calls = _patch_common(monkeypatch, host_snapshot)
    arm = gate.KAIRYU_ARM[world_size]
    reference = capture.capture_reference_arm(
        model_path=Path("/models/source"),
        world_size=world_size,
        host_snapshot=host_snapshot,
        bindings={
            "LLM": _DivergentReferenceLLM,
            "SamplingParams": _ReferenceParams,
            "KvCacheConfig": _Config,
            "MoeConfig": _Config,
            "Nvfp4GemmConfig": _Config,
            "version": "1.2.1",
        },
        container_observation=_observation(
            gate.REFERENCE_RUNTIME["image_repo_digest"],
            gate.REFERENCE_RUNTIME["image_config_id"],
        ),
    )
    runner = _BatchRunner(world_size)
    ownership = _ownership_fixture(arm)
    projections = _projection_names(arm)
    launcher = _Launcher(
        runner,
        arm=arm,
        ownership=ownership,
        projections=projections,
    )
    monkeypatch.setattr(
        capture,
        "_ownership_rows",
        lambda **_kw: (ownership, projections),
    )
    monkeypatch.setattr(capture, "_package_version", lambda name: "0.6.14")
    fragment = capture.capture_kairyu_arm(
        model_path=Path("/models/source"),
        world_size=world_size,
        host_snapshot=host_snapshot,
        reference_fragment=reference,
        launcher_type=lambda *_args, **_kwargs: launcher,
        tokenizer=_Tokenizer(),
        container_observation=_observation(
            host_snapshot["image_repo_digest"],
            host_snapshot["image_config_id"],
            host_snapshot["source"]["commit"],
        ),
    )

    assert launcher.shutdown_called is True
    assert len(fragment["free_run"]) == gate.PROMPT_COUNT
    assert len(fragment["teacher"]) == gate.EXPECTED_POSITIONS
    assert calls["checkpoint"] == 4
    assert [
        row["output_token_ids"] for row in fragment["free_run"]
    ] == [
        _canonical_tokens(index) for index in range(gate.PROMPT_COUNT)
    ]
    assert [
        row["canonical_token_id"] for row in fragment["teacher"]
    ] == [
        _canonical_tokens(prompt)[position]
        for position in range(gate.POSITIONS)
        for prompt in range(gate.PROMPT_COUNT)
    ]
    assert capture._fragment_valid(
        fragment, arm=arm, host_snapshot=host_snapshot
    )


def test_top64_rejects_missing_selected_and_nonfinite() -> None:
    values = _logprobs(7)
    values.pop(7)
    values[9000] = -9.0
    with pytest.raises(RuntimeError, match="absent"):
        capture._ranked_logprobs(7, values)
    values = _logprobs(7)
    values[7] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        capture._ranked_logprobs(7, values)


def _kernel_probe_fixture(
    projections: list[list[str]], ownership: list[dict[str, object]]
) -> list[dict[str, object]]:
    rows = []
    for rank, (names, owned) in enumerate(
        zip(projections, ownership, strict=True)
    ):
        owned_indices = list(owned["owned_expert_indices"])
        routed_matches = [
            capture._ROUTED_EXPERT.match(f"{name}.") for name in names
        ]
        routed_projection_count = sum(
            match is not None for match in routed_matches
        )
        dense_projection_count = len(names) - routed_projection_count
        fused_block_count = len(
            {
                int(match.group(1))
                for match in routed_matches
                if match is not None
            }
        )
        kernel_counts = {
            "nvfp4:flashinfer_nvfp4": dense_projection_count,
        }
        fused_moe = None
        if routed_projection_count:
            kernel_counts[
                "nvfp4:flashinfer_cutlass_fused_moe"
            ] = routed_projection_count
            fused_moe = {
                "backend": "flashinfer.cutlass_fused_moe",
                "global_expert_count": gate.NUM_EXPERTS,
                "local_expert_count": len(owned_indices),
                "local_expert_start": owned_indices[0],
                "weight_order": "up_gate",
                "scale_layout": "128x4",
                "output_dtype": "bfloat16",
                "ep_partial": True,
                "enable_alltoall": False,
                "block_count": fused_block_count,
                "successful_forward_block_count": fused_block_count,
            }
        rows.append(
            {
                "rank": rank,
                "projection_count": len(names),
                "projection_inventory_sha256": gate.sha256_json(names),
                "kernel_counts": dict(sorted(kernel_counts.items())),
                "fused_moe": fused_moe,
                "attention_output": {
                    "backend": "flashinfer.mm_fp4",
                    "placement": "row_parallel",
                    "parallel_size": len(projections),
                    "rank": rank,
                    "shard_axis": 1,
                    "global_in_features": 8_192,
                    "local_in_features": 8_192 // len(projections),
                    "out_features": 4_096,
                    "partial_dtype": "bfloat16",
                    "block_count": gate.NUM_LAYERS,
                    "successful_forward_block_count": gate.NUM_LAYERS,
                },
                "meta_count": {"parameters": 0, "buffers": 0, "total": 0},
                "load_info": {
                    "ep_rank": rank,
                    "ep_size": len(projections),
                    "owned_expert_count": len(owned_indices),
                    "owned_expert_indices_sha256": gate.sha256_json(
                        owned_indices
                    ),
                    "first_owned_expert": owned_indices[0],
                    "last_owned_expert": owned_indices[-1],
                    "quantization_source": "hf_quant_config.json",
                    "quantization_method": "nvfp4",
                    "kv_cache_quant_algo": "FP8",
                    "producer_name": "modelopt",
                    "producer_version": "0.33.0",
                    "checkpoint_tensor_count": gate.EXPECTED_TENSOR_COUNT,
                    "rank_loaded_tensor_count": owned[
                        "rank_loaded_tensor_count"
                    ],
                    "auxiliary_kv_scale_count": 188,
                },
            }
        )
    return rows


@pytest.mark.parametrize(
    "field", ["digest", "kernel", "fused_moe", "attention", "meta", "load"]
)
def test_all_rank_kernel_probe_mismatch_fails_closed(field: str) -> None:
    ownership = _ownership_fixture("kairyu_ep2")
    projections = [
        ["model.layers.0.mlp.experts.0.gate_proj", "a"],
        ["model.layers.0.mlp.experts.64.gate_proj", "c"],
    ]
    rows = _kernel_probe_fixture(projections, ownership)
    capture._validate_ep_kernel_probe(
        rows, projection_names=projections, ownership=ownership
    )
    broken = json.loads(json.dumps(rows))
    if field == "digest":
        broken[1]["projection_inventory_sha256"] = "0" * 64
    elif field == "kernel":
        broken[1]["kernel_counts"] = {"nvfp4:cpu_reference": 1}
    elif field == "fused_moe":
        broken[1]["fused_moe"]["successful_forward_block_count"] -= 1
    elif field == "attention":
        broken[1]["attention_output"]["placement"] = "replicated"
    elif field == "meta":
        broken[1]["meta_count"]["total"] = 1
    else:
        broken[1]["load_info"]["rank_loaded_tensor_count"] += 1
    with pytest.raises(RuntimeError, match="runtime kernel/load probe"):
        capture._validate_ep_kernel_probe(
            broken, projection_names=projections, ownership=ownership
        )


def test_attention_output_shard_contract_is_exact_and_complete() -> None:
    candidates = [
        f"model.layers.{layer}.self_attn.o_proj.{member}"
        for layer in range(gate.NUM_LAYERS)
        for member in ("weight", "weight_scale", "weight_scale_2", "input_scale")
    ]
    names = capture._attention_output_sharded_names(candidates)
    assert len(names) == gate.ATTENTION_OUTPUT_SHARDED_TENSOR_COUNT
    assert all(name.endswith((".weight", ".weight_scale")) for name in names)
    assert not any(name.endswith((".weight_scale_2", ".input_scale")) for name in names)

    sizes = {name: 16 for name in names}
    contracts = capture._attention_output_shard_contracts(
        names, sizes, world_size=4
    )
    rank_digests = [row["rank_slice_sha256"] for row in contracts]
    assert [row["slice_start_numerator"] for row in contracts] == [0, 1, 2, 3]
    assert [row["slice_end_numerator"] for row in contracts] == [1, 2, 3, 4]
    assert len(set(rank_digests)) == 4
    assert all(
        row["partition_sha256"] == gate.sha256_json(rank_digests)
        for row in contracts
    )
    assert all(row["full_bytes"] == row["rank_slice_bytes"] * 4 for row in contracts)

    with pytest.raises(ValueError, match="exactly 94"):
        capture._attention_output_sharded_names(candidates[1:])
    broken_sizes = dict(sizes)
    broken_sizes[names[0]] = 15
    with pytest.raises(ValueError, match="split evenly"):
        capture._attention_output_shard_contracts(
            names, broken_sizes, world_size=4
        )


def test_failure_fragment_is_formal(
    host_snapshot: dict[str, object],
) -> None:
    fragment = capture._failure_fragment(
        arm="reference_ep4",
        host_snapshot=host_snapshot,
        error=RuntimeError("boom"),
    )
    assert fragment["session"] is None
    assert fragment["failures"][0]["message"] == "boom"
    assert capture._fragment_valid(
        fragment, arm="reference_ep4", host_snapshot=host_snapshot
    )


def test_container_inspect_binds_running_process_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container_id = "a" * 64
    repo = "kairyu@sha256:" + "b" * 64
    config_id = "sha256:" + "c" * 64
    raw = [
        {
            "Id": container_id,
            "Image": config_id,
            "Config": {"Image": repo},
            "State": {
                "Running": True,
                "StartedAt": "2026-08-01T00:00:00Z",
            },
        }
    ]
    path = tmp_path / "inspect.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setenv("HOSTNAME", container_id[:12])

    observed = capture._container_observation(
        path, image_repo_digest=repo, image_config_id=config_id
    )

    assert observed["container_id"] == container_id
    assert observed["image_repo_digest"] == repo
    assert observed["image_config_id"] == config_id


@pytest.mark.parametrize("field", ["id", "image", "repo", "running"])
def test_container_inspect_rejects_identity_mismatch(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container_id = "a" * 64
    repo = "kairyu@sha256:" + "b" * 64
    config_id = "sha256:" + "c" * 64
    raw = {
        "Id": ("d" * 64 if field == "id" else container_id),
        "Image": ("sha256:" + "e" * 64 if field == "image" else config_id),
        "Config": {"Image": ("wrong" if field == "repo" else repo)},
        "State": {
            "Running": field != "running",
            "StartedAt": "2026-08-01T00:00:00Z",
        },
    }
    path = tmp_path / "inspect.json"
    path.write_text(json.dumps([raw]), encoding="utf-8")
    monkeypatch.setenv("HOSTNAME", container_id[:12])

    with pytest.raises(ValueError, match="docker inspect"):
        capture._container_observation(
            path, image_repo_digest=repo, image_config_id=config_id
        )


class _FailingReferenceLLM(_ReferenceLLM):
    def generate(self, _prompts, _params, *, use_tqdm):
        assert use_tqdm is False
        raise RuntimeError("generate failed")


def test_reference_shutdown_runs_when_generate_fails(
    monkeypatch: pytest.MonkeyPatch,
    host_snapshot: dict[str, object],
) -> None:
    _patch_common(monkeypatch, host_snapshot)
    with pytest.raises(RuntimeError, match="generate failed"):
        capture.capture_reference_arm(
            model_path=Path("/models/source"),
            world_size=2,
            host_snapshot=host_snapshot,
            bindings={
                "LLM": _FailingReferenceLLM,
                "SamplingParams": _ReferenceParams,
                "KvCacheConfig": _Config,
                "MoeConfig": _Config,
                "Nvfp4GemmConfig": _Config,
                "version": "1.2.1",
            },
            container_observation=_observation(
                gate.REFERENCE_RUNTIME["image_repo_digest"],
                gate.REFERENCE_RUNTIME["image_config_id"],
            ),
        )
    assert _FailingReferenceLLM.last_instance.shutdown_called is True
