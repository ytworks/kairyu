from __future__ import annotations

import importlib
import sys
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from kairyu.kernels import nvfp4_moe_gpu
from kairyu.kernels.nvfp4_moe_gpu import PreparedNvFp4Moe, fused_moe_forward


def _prepared(*, local_experts: int = 2, global_experts: int = 4) -> PreparedNvFp4Moe:
    hidden_size = 128
    intermediate_size = 128
    return PreparedNvFp4Moe(
        fc1_weight=torch.zeros(
            local_experts,
            2 * intermediate_size,
            hidden_size // 16,
            dtype=torch.int64,
        ),
        fc2_weight=torch.zeros(
            local_experts,
            hidden_size,
            intermediate_size // 16,
            dtype=torch.int64,
        ),
        fc1_act_scale=torch.tensor(1.0, dtype=torch.float32),
        fc1_weight_scale=torch.zeros(
            local_experts,
            2 * intermediate_size,
            hidden_size // 64,
            dtype=torch.int32,
        ),
        fc1_dequant_scale=torch.ones(local_experts, dtype=torch.float32),
        fc2_act_scale=torch.tensor(1.0, dtype=torch.float32),
        fc2_weight_scale=torch.zeros(
            local_experts,
            hidden_size,
            intermediate_size // 64,
            dtype=torch.int32,
        ),
        fc2_dequant_scale=torch.ones(local_experts, dtype=torch.float32),
        num_global_experts=global_experts,
        local_expert_start=2,
    )


def _request() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    PreparedNvFp4Moe,
    torch.Tensor,
]:
    hidden = torch.randn(3, 128, dtype=torch.bfloat16)
    # Rank 1 owns [2, 4), so every route below is a valid zero-local-route case.
    selected = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int32)
    routing = torch.full((3, 2), 0.5, dtype=torch.float32)
    prepared = _prepared()
    output = torch.full_like(hidden, 17)
    return hidden, selected, routing, prepared, output


@pytest.fixture
def cpu_contract(monkeypatch):
    monkeypatch.setattr(nvfp4_moe_gpu, "_require_sm120", lambda _hidden: None)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())


def test_lazy_flashinfer_dispatch_uses_global_ids_ep_and_preallocated_output(
    monkeypatch,
    cpu_contract,
) -> None:
    hidden, selected, routing, prepared, output = _request()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    swiglu = object()

    def fake_cutlass(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs["output"].zero_()
        return [kwargs["output"], torch.tensor(0)]

    fake = SimpleNamespace(
        __version__="0.6.14",
        cutlass_fused_moe=fake_cutlass,
        ActivationType=SimpleNamespace(Swiglu=swiglu),
    )
    monkeypatch.setitem(sys.modules, "flashinfer", fake)

    actual = fused_moe_forward(
        hidden,
        selected,
        routing,
        prepared,
        output=output,
        ep_size=2,
        ep_rank=1,
    )

    assert actual is output
    assert torch.count_nonzero(output) == 0
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:5] == (
        hidden,
        selected,
        routing,
        prepared.fc1_weight,
        prepared.fc2_weight,
    )
    assert args[5] == torch.bfloat16
    assert kwargs["quant_scales"] == prepared.quant_scales()
    assert kwargs["output"] is output
    assert kwargs["ep_size"] == 2
    assert kwargs["ep_rank"] == 1
    assert kwargs["enable_alltoall"] is False
    assert kwargs["tp_size"] == 1
    assert kwargs["tp_rank"] == 0
    assert kwargs["activation_type"] is swiglu
    assert kwargs["use_packed_weights"] is False
    assert kwargs["input_sf"] is None


def test_attention_dp_quantizes_before_communication_and_interleaves_after_gather(
    monkeypatch,
    cpu_contract,
) -> None:
    hidden = torch.randn(3, 128, dtype=torch.bfloat16)
    global_scale = torch.tensor(1.0, dtype=torch.float32)
    packed = torch.zeros(3, 64, dtype=torch.uint8)
    scale_rows = torch.zeros(3, 8, dtype=torch.uint8)
    swizzled = torch.zeros(128 * 8, dtype=torch.uint8)
    calls: list[tuple[str, object]] = []

    def fake_quantize(value, scale, *, is_sf_swizzled_layout):
        calls.append(
            (
                "quantize",
                (value, scale, is_sf_swizzled_layout),
            )
        )
        return packed, scale_rows

    def fake_interleave(value):
        calls.append(("interleave", value))
        return swizzled

    fake = SimpleNamespace(
        __version__="0.6.14",
        fp4_quantize=fake_quantize,
        nvfp4_block_scale_interleave=fake_interleave,
    )
    monkeypatch.setitem(sys.modules, "flashinfer", fake)

    actual_packed, actual_rows = nvfp4_moe_gpu.quantize_moe_input(
        hidden,
        global_scale,
    )
    actual_swizzled = nvfp4_moe_gpu.interleave_moe_input_scales(actual_rows)

    assert actual_packed is packed
    assert actual_rows is scale_rows
    assert actual_swizzled is swizzled
    assert calls == [
        ("quantize", (hidden, global_scale, False)),
        ("interleave", scale_rows),
    ]


def test_attention_dp_prequantized_dispatch_passes_swizzled_input_scale(
    monkeypatch,
    cpu_contract,
) -> None:
    _hidden, selected, routing, prepared, output = _request()
    packed = torch.zeros(3, 64, dtype=torch.uint8)
    input_sf = torch.zeros(128 * 8, dtype=torch.uint8)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_cutlass(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs["output"].zero_()
        return [kwargs["output"]]

    fake = SimpleNamespace(
        __version__="0.6.14",
        cutlass_fused_moe=fake_cutlass,
        ActivationType=SimpleNamespace(Swiglu=object()),
    )
    monkeypatch.setitem(sys.modules, "flashinfer", fake)

    actual = nvfp4_moe_gpu.fused_moe_forward_quantized(
        packed,
        input_sf,
        selected,
        routing,
        prepared,
        output=output,
        ep_size=2,
        ep_rank=1,
    )

    assert actual is output
    assert calls[0][0][0] is packed
    assert calls[0][1]["input_sf"] is input_sf
    assert calls[0][1]["enable_alltoall"] is False


def test_module_import_does_not_import_flashinfer(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "flashinfer", raising=False)

    importlib.reload(nvfp4_moe_gpu)

    assert "flashinfer" not in sys.modules


def test_missing_flashinfer_is_actionable(monkeypatch, cpu_contract) -> None:
    hidden, selected, routing, prepared, output = _request()

    def missing(_name: str):
        raise ModuleNotFoundError("no flashinfer")

    monkeypatch.setattr(nvfp4_moe_gpu.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="requires FlashInfer with cutlass_fused_moe"):
        fused_moe_forward(
            hidden,
            selected,
            routing,
            prepared,
            output=output,
            ep_size=2,
            ep_rank=1,
        )


@pytest.mark.parametrize("capability", [(11, 0), (12, 1)])
def test_device_contract_rejects_unverified_sms(monkeypatch, capability) -> None:
    hidden = SimpleNamespace(device=torch.device("cuda:7"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)

    with pytest.raises(RuntimeError, match="verified only on SM120"):
        nvfp4_moe_gpu._require_sm120(hidden)


def test_device_contract_queries_the_selected_sm120_device(monkeypatch) -> None:
    hidden = SimpleNamespace(device=torch.device("cuda:7"))
    seen: list[torch.device] = []

    def capability(device):
        seen.append(device)
        return (12, 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)

    nvfp4_moe_gpu._require_sm120(hidden)

    assert seen == [torch.device("cuda:7")]


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("num_global_experts", 6, r"local_experts \* ep_size"),
        ("local_expert_start", 0, "contiguous global range"),
        (
            "fc1_weight_scale",
            torch.zeros(2, 128, 2, dtype=torch.int32),
            "fc1_weight_scale shape",
        ),
        (
            "fc2_dequant_scale",
            torch.ones(2, dtype=torch.bfloat16),
            "fc2_dequant_scale must have dtype",
        ),
    ],
)
def test_prepared_contract_fails_closed(
    monkeypatch,
    cpu_contract,
    field,
    replacement,
    message,
) -> None:
    hidden, selected, routing, prepared, output = _request()
    prepared = replace(prepared, **{field: replacement})

    with pytest.raises((TypeError, ValueError), match=message):
        fused_moe_forward(
            hidden,
            selected,
            routing,
            prepared,
            output=output,
            ep_size=2,
            ep_rank=1,
        )


@pytest.mark.parametrize(
    ("target", "replacement", "message"),
    [
        ("hidden", torch.zeros(3, 128, dtype=torch.float16), "hidden must have dtype"),
        (
            "hidden",
            torch.zeros(128, 3, dtype=torch.bfloat16).T,
            "hidden must be contiguous",
        ),
        (
            "selected",
            torch.zeros(3, 2, dtype=torch.int64),
            "selected_experts must have dtype",
        ),
        (
            "routing",
            torch.zeros(3, 2, dtype=torch.bfloat16),
            "routing_weights must have dtype",
        ),
        (
            "output",
            torch.zeros(128, 3, dtype=torch.bfloat16).T,
            "output must be contiguous",
        ),
    ],
)
def test_dynamic_tensor_contract_fails_closed(
    monkeypatch,
    cpu_contract,
    target,
    replacement,
    message,
) -> None:
    hidden, selected, routing, prepared, output = _request()
    values = {
        "hidden": hidden,
        "selected": selected,
        "routing": routing,
        "output": output,
    }
    values[target] = replacement

    with pytest.raises((TypeError, ValueError), match=message):
        fused_moe_forward(
            values["hidden"],
            values["selected"],
            values["routing"],
            prepared,
            output=values["output"],
            ep_size=2,
            ep_rank=1,
        )


def test_global_route_ids_fail_closed_without_rejecting_zero_local_routes(
    monkeypatch,
    cpu_contract,
) -> None:
    hidden, selected, routing, prepared, output = _request()
    selected[0, 0] = prepared.num_global_experts

    with pytest.raises(RuntimeError, match="global IDs"):
        fused_moe_forward(
            hidden,
            selected,
            routing,
            prepared,
            output=output,
            ep_size=2,
            ep_rank=1,
        )


def test_output_must_not_alias_hidden(cpu_contract) -> None:
    hidden, selected, routing, prepared, _output = _request()

    with pytest.raises(ValueError, match="must not alias hidden"):
        fused_moe_forward(
            hidden,
            selected,
            routing,
            prepared,
            output=hidden,
            ep_size=2,
            ep_rank=1,
        )
