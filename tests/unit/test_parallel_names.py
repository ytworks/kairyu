"""#233: canonical checkpoint names survive TP, SP, and EP binding."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
import torch
from safetensors.torch import save_file

from kairyu.engine.core.quant_config import QuantConfig, QuantMethod
from kairyu.models.config import parse_model_config
from kairyu.models.llama import DenseDecoder
from kairyu.models.moe_parallel import (
    EpMoeBlock,
    RemoteExpertOwnershipError,
)
from kairyu.models.parallel import (
    GatherBeforeNorm,
    ParallelShardInfo,
    RowParallelLinear,
    ScatterAfterEmbedding,
    SequenceParallelContext,
    SequenceParallelNorm,
    UnsupportedParallelSerializationError,
    build_tp_model,
    checkpoint_member_specs,
    parallel_shard_info,
    parallel_state_dict,
    tp_view,
)
from kairyu.quant.linear import (
    ExpertScope,
    Fp8Linear,
    LinearRole,
    TensorParallelMode,
    linear_factory,
)

_DENSE_RAW = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 64,
    "vocab_size": 64,
    "max_position_embeddings": 128,
    "tie_word_embeddings": False,
}

_QWEN_MOE_RAW = {
    **_DENSE_RAW,
    "architectures": ["Qwen3MoeForCausalLM"],
    "head_dim": 8,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "norm_topk_prob": True,
}

_DEEPSEEK_MOE_RAW = {
    **_DENSE_RAW,
    "architectures": ["DeepseekV3ForCausalLM"],
    "num_key_value_heads": 4,
    "kv_lora_rank": 8,
    "q_lora_rank": None,
    "qk_nope_head_dim": 4,
    "qk_rope_head_dim": 4,
    "v_head_dim": 6,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "n_group": 2,
    "topk_group": 1,
    "n_shared_experts": 1,
    "first_k_dense_replace": 0,
}


@dataclass
class _TopologyComm:
    rank: int
    world_size: int

    def tensor_all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def tensor_all_reduce_max(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def tensor_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def tensor_reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class _UnusedEpComm:
    """EP naming tests never dispatch tokens."""


def _write_dense_checkpoint(path, raw: dict) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(raw))
    model = DenseDecoder(parse_model_config(raw))
    state = {
        name: tensor.detach().clone().contiguous()
        for name, tensor in model.state_dict().items()
    }
    if raw.get("tie_word_embeddings"):
        state.pop("lm_head.weight")
    save_file(state, path / "model.safetensors")


def _tp_factory(
    quant: QuantConfig | None = None,
    *,
    rank: int = 1,
):
    return linear_factory(
        quant or QuantConfig(QuantMethod.NONE),
        tensor_parallel_size=2,
        tensor_parallel_rank=rank,
    )


def _local_dense(
    *,
    quant: QuantConfig | None = None,
    rank: int = 1,
) -> DenseDecoder:
    config = tp_view(parse_model_config(_DENSE_RAW), 2, rank)
    return DenseDecoder(config, linear_factory=_tp_factory(quant, rank=rank))


def _bind_dense_execution(
    model: DenseDecoder,
    comm: _TopologyComm,
    *,
    sequence_parallel: bool,
) -> None:
    context = SequenceParallelContext(comm) if sequence_parallel else None
    for layer in model.model.layers:
        RowParallelLinear(layer.self_attn.o_proj, comm, context)
        RowParallelLinear(layer.mlp.down_proj, comm, context)
        if context is not None:
            SequenceParallelNorm(layer.input_layernorm, context)
            SequenceParallelNorm(layer.post_attention_layernorm, context)
    if context is not None:
        ScatterAfterEmbedding(model.model.embed_tokens, context)
        GatherBeforeNorm(model.model.norm, context)


def _name_contract(model: torch.nn.Module) -> dict[str, tuple[str, ...]]:
    return {
        "state": tuple(model.state_dict()),
        "parameters": tuple(name for name, _ in model.named_parameters()),
        "buffers": tuple(name for name, _ in model.named_buffers()),
        "modules": tuple(name for name, _ in model.named_modules()),
    }


def _all_names(model: torch.nn.Module) -> set[str]:
    contract = _name_contract(model)
    return set().union(*contract.values())


def _assert_no_wrapper_paths(names: set[str]) -> None:
    forbidden_segments = {"block", "embedding", "local", "local_experts"}
    for name in names:
        assert forbidden_segments.isdisjoint(name.split(".")), name
        assert ".norm.norm." not in f".{name}.", name


def _assert_named_objects_resolve(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        assert model.get_parameter(name) is parameter
    for name, buffer in model.named_buffers():
        assert model.get_buffer(name) is buffer
    for name, module in model.named_modules():
        if name:
            assert model.get_submodule(name) is module


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in parallel_state_dict(model).items()
    }


def _ep_model(
    raw: dict,
    *,
    rank: int = 1,
    factory=None,
) -> tuple[DenseDecoder, EpMoeBlock]:
    config = parse_model_config(raw)
    model = DenseDecoder(
        config,
        linear_factory=factory or linear_factory(QuantConfig(QuantMethod.NONE)),
    )
    block = model.model.layers[0].mlp
    ep_block = EpMoeBlock(
        block,
        _UnusedEpComm(),
        ep_rank=rank,
        ep_size=2,
    )
    model.model.layers[0].mlp = ep_block
    return model, ep_block


def test_tp_and_sp_binding_before_or_after_load_preserves_canonical_apis() -> None:
    torch.manual_seed(233)
    source = _local_dense()
    source_state = {
        name: tensor.detach().clone()
        for name, tensor in source.state_dict().items()
    }
    canonical = _name_contract(source)

    bind_after_load = _local_dense()
    bind_after_load.load_state_dict(source_state, strict=True)
    _bind_dense_execution(
        bind_after_load,
        _TopologyComm(rank=1, world_size=2),
        sequence_parallel=True,
    )

    bind_before_load = _local_dense()
    _bind_dense_execution(
        bind_before_load,
        _TopologyComm(rank=1, world_size=2),
        sequence_parallel=True,
    )
    bind_before_load.load_state_dict(source_state, strict=True)

    for model in (bind_after_load, bind_before_load):
        assert _name_contract(model) == canonical
        _assert_named_objects_resolve(model)
        _assert_no_wrapper_paths(_all_names(model))
        for name, expected in source_state.items():
            torch.testing.assert_close(model.state_dict()[name], expected)

        output = model.get_submodule("model.layers.0.self_attn.o_proj")
        assert output is model.model.layers[0].self_attn.o_proj
        assert (
            model.get_parameter("model.layers.0.self_attn.o_proj.weight")
            is output.weight
        )
        assert (
            model.get_submodule("model.embed_tokens")
            is model.model.embed_tokens
        )
        assert model.get_submodule("model.norm") is model.model.norm

    assert set(bind_before_load.state_dict()) == set(source_state)


def test_dense_row_parallel_bias_is_registered_canonically_and_added_once() -> None:
    torch.manual_seed(0)
    full_weight = (torch.randn(3, 6) * 0.25).to(torch.bfloat16)
    bias = (torch.randn(3) * 8).to(torch.bfloat16)
    hidden = (torch.randn(5, 6) * 0.25).to(torch.bfloat16)

    local = torch.nn.Linear(3, 3, bias=True, dtype=torch.bfloat16)
    with torch.no_grad():
        local.weight.copy_(full_weight[:, :3])
        local.bias.copy_(bias)
    canonical_bias = local.bias
    local_partial = torch.nn.functional.linear(
        hidden[:, :3],
        full_weight[:, :3],
        bias=None,
    )
    peer_partial = torch.nn.functional.linear(
        hidden[:, 3:],
        full_weight[:, 3:],
        bias=None,
    )
    expected = (local_partial + peer_partial) + bias

    # This gate is intentionally sensitive to where BF16 rounding occurs.
    # Adding bias inside the rank-local GEMM and removing it again cannot
    # reconstruct the unbiased partial exactly.
    recovered_after_local_bias = (
        torch.nn.functional.linear(
            hidden[:, :3],
            full_weight[:, :3],
            bias,
        )
        - bias
        + peer_partial
    ) + bias
    assert not torch.equal(recovered_after_local_bias, expected)

    class _PeerReduce:
        rank = 0
        world_size = 2

        def __init__(self) -> None:
            self.seen: torch.Tensor | None = None

        def tensor_all_reduce(self, partial: torch.Tensor) -> torch.Tensor:
            self.seen = partial.detach().clone()
            return partial + peer_partial

    comm = _PeerReduce()
    bound = RowParallelLinear(local, comm)
    actual = bound(hidden[:, :3])

    assert bound is local
    assert local.bias is canonical_bias
    assert tuple(local.state_dict()) == ("weight", "bias")
    assert tuple(dict(local.named_parameters())) == ("weight", "bias")
    assert torch.equal(comm.seen, local_partial)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("activation_dynamic", [True, False], ids=("dynamic", "static"))
def test_fp8_tp_context_and_buffers_keep_canonical_paths(
    activation_dynamic: bool,
) -> None:
    quant = QuantConfig(
        QuantMethod.FP8,
        weight_bits=8,
        activation_bits=8,
        activation_dynamic=activation_dynamic,
    )
    model = _local_dense(quant=quant)
    before = _name_contract(model)
    comm = _TopologyComm(rank=1, world_size=2)
    output = model.model.layers[0].self_attn.o_proj
    RowParallelLinear(output, comm)

    query_name = "model.layers.0.self_attn.q_proj"
    output_name = "model.layers.0.self_attn.o_proj"
    query = model.get_submodule(query_name)
    assert isinstance(query, Fp8Linear)
    assert isinstance(output, Fp8Linear)
    assert _name_contract(model) == before
    assert query.linear_context.qualified_name == query_name
    assert query.linear_context.tensor_parallel.mode is TensorParallelMode.COLUMN
    assert query.linear_context.tensor_parallel.rank == 1
    assert query.linear_context.tensor_parallel.world_size == 2
    assert output.linear_context.qualified_name == output_name
    assert output.linear_context.tensor_parallel.mode is TensorParallelMode.ROW

    buffer_names = dict(model.named_buffers())
    assert f"{query_name}.weight" in buffer_names
    assert f"{query_name}.weight_scale" in buffer_names
    assert f"{output_name}.weight" in buffer_names
    assert f"{output_name}.weight_scale" in buffer_names
    input_scale_name = f"{output_name}.input_scale"
    assert (input_scale_name in buffer_names) is not activation_dynamic
    if activation_dynamic:
        assert output._row_parallel_comm is comm
    else:
        assert not hasattr(output, "_row_parallel_comm")

    specs = checkpoint_member_specs(model)
    assert specs[f"{query_name}.weight"].source_name == f"{query_name}.weight"
    assert specs[f"{query_name}.weight"].shard_dim == 0
    assert specs[f"{query_name}.weight_scale"].shard_dim == 0
    assert specs[f"{output_name}.weight"].shard_dim == 1
    assert specs[f"{output_name}.weight_scale"].shard_dim is None
    if not activation_dynamic:
        assert specs[input_scale_name].shard_dim is None
    _assert_named_objects_resolve(model)
    _assert_no_wrapper_paths(_all_names(model))


@pytest.mark.parametrize(
    ("quant", "member_dims"),
    [
        pytest.param(
            QuantConfig(
                QuantMethod.FP8,
                weight_bits=8,
                activation_bits=8,
                activation_dynamic=False,
            ),
            {
                "weight": (0, 1),
                "weight_scale": (0, None),
                "input_scale": (None, None),
            },
            id="fp8",
        ),
        pytest.param(
            QuantConfig(
                QuantMethod.INT8,
                weight_bits=8,
                activation_bits=8,
                activation_dynamic=True,
            ),
            {
                "weight": (0, 1),
                "weight_scale": (0, None),
            },
            id="int8",
        ),
        pytest.param(
            QuantConfig(
                QuantMethod.AWQ,
                weight_bits=4,
                group_size=16,
            ),
            {
                "qweight": (1, 0),
                "qzeros": (1, 0),
                "scales": (1, 0),
            },
            id="awq",
        ),
        pytest.param(
            QuantConfig(
                QuantMethod.GPTQ,
                weight_bits=4,
                group_size=16,
            ),
            {
                "qweight": (1, 0),
                "qzeros": (1, 0),
                "scales": (1, 0),
                "g_idx": (None, 0),
            },
            id="gptq",
        ),
        pytest.param(
            QuantConfig(
                QuantMethod.NVFP4,
                weight_bits=4,
                group_size=16,
            ),
            {
                "weight": (0, 1),
                "weight_scale": (0, 1),
                "weight_scale_2": (None, None),
                "input_scale": (None, None),
            },
            id="nvfp4",
        ),
    ],
)
def test_quantized_checkpoint_members_have_exact_physical_shard_axes(
    quant: QuantConfig,
    member_dims: dict[str, tuple[int | None, int | None]],
) -> None:
    model = _local_dense(quant=quant)
    specs = checkpoint_member_specs(model)
    query = "model.layers.0.self_attn.q_proj"
    output = "model.layers.0.self_attn.o_proj"
    state_names = set(model.state_dict())

    query_members = {
        name.removeprefix(f"{query}.")
        for name in state_names
        if name.startswith(f"{query}.")
        and "." not in name.removeprefix(f"{query}.")
    }
    output_members = {
        name.removeprefix(f"{output}.")
        for name in state_names
        if name.startswith(f"{output}.")
        and "." not in name.removeprefix(f"{output}.")
    }
    assert query_members == {*member_dims, "bias"}
    assert output_members == set(member_dims)

    for member, (column_dim, row_dim) in member_dims.items():
        query_name = f"{query}.{member}"
        output_name = f"{output}.{member}"
        assert specs[query_name].source_name == query_name
        assert specs[query_name].shard_dim == column_dim
        assert specs[output_name].source_name == output_name
        assert specs[output_name].shard_dim == row_dim
    assert specs[f"{query}.bias"].source_name == f"{query}.bias"
    assert specs[f"{query}.bias"].shard_dim == 0


def test_tp_rank_local_strict_roundtrip_and_full_hf_error(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "dense"
    _write_dense_checkpoint(checkpoint, _DENSE_RAW)
    comm = _TopologyComm(rank=1, world_size=2)
    source, _, _ = build_tp_model(
        str(checkpoint),
        tp=2,
        rank=1,
        comm=comm,
        sequence_parallel=True,
    )
    state = _clone_state(source)
    assert parallel_shard_info(source) == (
        ("<root>", ParallelShardInfo("tensor", 1, 2)),
    )

    destination, _, _ = build_tp_model(
        str(checkpoint),
        tp=2,
        rank=1,
        comm=_TopologyComm(rank=1, world_size=2),
        sequence_parallel=True,
    )
    first_name, first_parameter = next(iter(destination.named_parameters()))
    with torch.no_grad():
        first_parameter.add_(1)
    assert not torch.equal(destination.get_parameter(first_name), state[first_name])
    result = destination.load_state_dict(state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert tuple(destination.state_dict()) == tuple(state)
    for name, expected in state.items():
        torch.testing.assert_close(destination.state_dict()[name], expected)
    _assert_named_objects_resolve(destination)
    _assert_no_wrapper_paths(_all_names(destination))

    messages = []
    errors = []
    for _ in range(2):
        with pytest.raises(UnsupportedParallelSerializationError) as caught:
            parallel_state_dict(source, mode="full-hf")
        messages.append(str(caught.value))
        errors.append(caught.value)
    assert messages == [
        "full-hf serialization is unsupported from one parallel rank "
        "(tensor@<root>:rank=1/2); use mode='rank-local' for a same-topology "
        "round-trip or gather canonical shards from every rank"
    ] * 2
    assert errors[0].mode == "full-hf"
    assert errors[0].topology == (
        ("<root>", ParallelShardInfo("tensor", 1, 2)),
    )


def test_tp_tied_weights_keep_alias_and_canonical_state_names(tmp_path) -> None:
    raw = {**_DENSE_RAW, "tie_word_embeddings": True}
    checkpoint = tmp_path / "tied"
    _write_dense_checkpoint(checkpoint, raw)
    model, _, _ = build_tp_model(
        str(checkpoint),
        tp=2,
        rank=1,
        comm=_TopologyComm(rank=1, world_size=2),
    )

    embedding = model.get_parameter("model.embed_tokens.weight")
    head = model.get_parameter("lm_head.weight")
    assert head is embedding
    rank_state = parallel_state_dict(model, keep_vars=True)
    assert tuple(name for name in rank_state if name.endswith("embed_tokens.weight")) == (
        "model.embed_tokens.weight",
    )
    assert "lm_head.weight" in rank_state
    assert rank_state["lm_head.weight"] is rank_state["model.embed_tokens.weight"]
    assert "lm_head.weight" not in dict(model.named_parameters())

    snapshot = {
        name: tensor.detach().clone()
        for name, tensor in rank_state.items()
    }
    result = model.load_state_dict(snapshot, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert (
        model.get_parameter("lm_head.weight")
        is model.get_parameter("model.embed_tokens.weight")
    )
    _assert_no_wrapper_paths(_all_names(model))


@pytest.mark.parametrize(
    ("raw", "has_shared"),
    [
        (_QWEN_MOE_RAW, False),
        (_DEEPSEEK_MOE_RAW, True),
    ],
    ids=("qwen", "deepseek"),
)
def test_ep_global_expert_paths_context_and_rank_local_roundtrip(
    raw: dict,
    has_shared: bool,
) -> None:
    model, ep_block = _ep_model(raw)
    prefix = "model.layers.0.mlp"
    assert len(ep_block.experts) == 4
    assert ep_block.owned_expert_indices == (2, 3)
    assert ep_block.experts[0] is None
    assert ep_block.experts[1] is None
    assert ep_block.experts[2] is not None
    assert ep_block.experts[3] is not None
    assert ep_block.owner_rank(0) == 0
    assert ep_block.owner_rank(3) == 1
    assert parallel_shard_info(model) == (
        (prefix, ParallelShardInfo("expert", 1, 2)),
    )

    state = _clone_state(model)
    state_names = set(state)
    member_specs = checkpoint_member_specs(model)
    expert_indices = {
        int(name.removeprefix(f"{prefix}.experts.").split(".", 1)[0])
        for name in state_names
        if name.startswith(f"{prefix}.experts.")
    }
    assert expert_indices == {2, 3}
    assert f"{prefix}.gate.weight" in state_names
    assert ep_block.gate.linear_context.qualified_name == f"{prefix}.gate"
    assert ep_block.gate.linear_context.role is LinearRole.MOE_ROUTER

    expert = model.get_submodule(f"{prefix}.experts.2")
    assert expert is ep_block.local_expert(2)
    expert_context = expert.down_proj.linear_context
    assert expert_context.qualified_name == f"{prefix}.experts.2.down_proj"
    assert expert_context.expert_index == 2
    assert expert_context.expert_scope is ExpertScope.ROUTED
    assert ep_block.local_experts == (
        ep_block.experts[2],
        ep_block.experts[3],
    )

    if has_shared:
        correction = f"{prefix}.gate.e_score_correction_bias"
        shared = f"{prefix}.shared_experts.up_proj"
        assert correction in state_names
        assert member_specs[correction].source_name == correction
        assert member_specs[correction].shard_dim is None
        assert f"{shared}.weight" in state_names
        shared_module = model.get_submodule(shared)
        assert shared_module.linear_context.qualified_name == shared
        assert shared_module.linear_context.expert_scope is ExpertScope.SHARED
        assert shared_module.linear_context.expert_index is None
    else:
        assert ep_block.shared_experts is None
        assert not any(".shared_experts." in name for name in state_names)

    messages = []
    for _ in range(2):
        with pytest.raises(RemoteExpertOwnershipError) as caught:
            ep_block.local_expert(0)
        messages.append(str(caught.value))
    assert messages == [
        "expert 0 is owned by rank 0, not rank 1",
        "expert 0 is owned by rank 0, not rank 1",
    ]

    destination, _ = _ep_model(raw)
    result = destination.load_state_dict(state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert tuple(destination.state_dict()) == tuple(state)
    for name, expected in state.items():
        torch.testing.assert_close(destination.state_dict()[name], expected)

    _assert_named_objects_resolve(model)
    _assert_no_wrapper_paths(_all_names(model))
    with pytest.raises(
        UnsupportedParallelSerializationError,
        match=r"expert@model\.layers\.0\.mlp:rank=1/2",
    ):
        parallel_state_dict(model, mode="full-hf")


def test_nested_row_parallel_owned_expert_keeps_global_expert_name() -> None:
    factory = _tp_factory(rank=1)
    model, ep_block = _ep_model(_QWEN_MOE_RAW, factory=factory)
    expert = ep_block.local_expert(2)
    projection = expert.down_proj
    canonical = "model.layers.0.mlp.experts.2.down_proj"
    assert projection.linear_context.qualified_name == canonical
    assert projection.linear_context.tensor_parallel.mode is TensorParallelMode.ROW
    before = _name_contract(model)

    comm = _TopologyComm(rank=1, world_size=2)
    bound = RowParallelLinear(projection, comm)

    assert bound is projection
    assert model.get_submodule(canonical) is projection
    assert model.get_parameter(f"{canonical}.weight") is projection.weight
    assert _name_contract(model) == before
    assert f"{canonical}.weight" in model.state_dict()
    assert not any(".local." in name for name in _all_names(model))
    assert not any(".local_experts." in name for name in _all_names(model))
    _assert_named_objects_resolve(model)
    _assert_no_wrapper_paths(_all_names(model))
