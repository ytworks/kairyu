"""GPU oracles for every advertised production quantized-linear scheme."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def cuda():
    if not torch.cuda.is_available():  # pragma: no cover - deploy-day only
        pytest.skip("CUDA required")
    return "cuda"


def _fp8_module(in_features=64, out_features=32):
    from kairyu.quant.fp8 import quantize_fp8
    from kairyu.quant.linear import Fp8Linear

    module = Fp8Linear(in_features, out_features, bias=True)
    qweight, scale = quantize_fp8(torch.randn(out_features, in_features))
    module.weight.copy_(qweight)
    module.weight_scale.copy_(scale)
    return module


def _int8_module(in_features=64, out_features=32):
    from kairyu.quant.int8 import quantize_int8_weight
    from kairyu.quant.linear import Int8Linear

    module = Int8Linear(in_features, out_features, bias=True)
    qweight, scale = quantize_int8_weight(torch.randn(out_features, in_features))
    module.weight.copy_(qweight)
    module.weight_scale.copy_(scale)
    return module


@pytest.mark.parametrize("k_size", [1536, 8960])
def test_int8_activation_quantization_matches_torch_rounding(cuda, k_size):
    from kairyu.kernels.quant_gemm_gpu import _quantize_activation
    from kairyu.quant.int8 import quantize_int8_activation

    torch.manual_seed(356)
    values = torch.randn(4, k_size, device=cuda, dtype=torch.bfloat16)
    values[0].zero_()
    expected, expected_scale = quantize_int8_activation(values.cpu().float())
    actual, actual_scale = _quantize_activation(values, torch.int8, 127.0)

    torch.testing.assert_close(actual_scale.cpu(), expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("k_size", "seed"), [(64, 1174), (1536, 357), (8960, 357)]
)
def test_int8_fused_output_is_bit_exact_with_biased_oracle(cuda, k_size, seed):
    torch.manual_seed(seed)
    module = _int8_module(k_size, 128)
    module.bias.data.normal_()
    module.bias.data = module.bias.data.to(torch.bfloat16)
    values = torch.randn(4, k_size, dtype=torch.bfloat16)
    expected = module.forward_reference(values)
    actual = module.to(cuda)(values.to(cuda)).cpu()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _awq_module(in_features=64, out_features=32):
    from kairyu.quant.awq import quantize_awq
    from kairyu.quant.linear import AwqLinear

    module = AwqLinear(in_features, out_features, bias=True, group_size=16)
    qweight, qzeros, scales = quantize_awq(
        torch.randn(out_features, in_features), group_size=16
    )
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    return module


def _gptq_module(in_features=64, out_features=32):
    from kairyu.quant.gptq import quantize_gptq
    from kairyu.quant.linear import GptqLinear

    module = GptqLinear(in_features, out_features, bias=True, group_size=16)
    qweight, qzeros, scales, g_idx = quantize_gptq(
        torch.randn(out_features, in_features), group_size=16
    )
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.g_idx.copy_(g_idx)
    return module


def _nvfp4_module(in_features=64, out_features=32):
    from kairyu.quant.linear import NvFp4Linear
    from kairyu.quant.nvfp4 import quantize_nvfp4

    module = NvFp4Linear(in_features, out_features, bias=True)
    weight, weight_scale, weight_scale_2 = quantize_nvfp4(
        torch.randn(out_features, in_features)
    )
    module.weight.copy_(weight)
    module.weight_scale.copy_(weight_scale)
    module.weight_scale_2.copy_(weight_scale_2)
    module.input_scale.copy_(torch.tensor(4.0 / (6.0 * 448.0)))
    return module


@pytest.mark.parametrize(
    ("scheme", "factory", "atol"),
    [
        ("fp8", _fp8_module, 0.40),
        ("int8", _int8_module, 0.08),
        ("awq", _awq_module, 0.08),
        ("gptq", _gptq_module, 0.08),
        ("nvfp4", _nvfp4_module, 0.08),
    ],
)
@pytest.mark.parametrize(
    ("m_size", "k_size", "n_size"),
    [(1, 64, 32), (17, 96, 40), (32, 64, 32)],
)
def test_fused_kernel_matches_cpu_oracle_without_dequantizing_weight(
    cuda,
    monkeypatch,
    scheme: str,
    factory: Callable,
    atol: float,
    m_size: int,
    k_size: int,
    n_size: int,
):
    """Production forward must dispatch to the fused kernel, never dequantize."""
    torch.manual_seed(0)
    module = factory(k_size, n_size)
    module.bias.data.normal_()
    x = torch.randn(m_size, k_size)
    # Both paths see the identical values that the production BF16 model emits.
    cpu_out = module.forward_reference(x.to(torch.bfloat16).float())

    def forbidden_dequantize(*_args, **_kwargs):
        raise AssertionError(f"{scheme} CUDA forward materialized its full weight")

    monkeypatch.setattr(type(module), "dequantize", forbidden_dequantize)
    module = module.to(cuda)
    gpu_out = module(x.to(device=cuda, dtype=torch.bfloat16)).float().cpu()

    if scheme == "fp8":
        error = gpu_out - cpu_out
        relative_rmse = error.square().mean().sqrt() / cpu_out.square().mean().sqrt()
        assert relative_rmse.item() < 0.035
        assert error.abs().max().item() < 0.12 * k_size**0.5
    else:
        torch.testing.assert_close(gpu_out, cpu_out, rtol=0.03, atol=atol)


@pytest.mark.parametrize("factory", [_awq_module, _gptq_module], ids=["awq", "gptq"])
@pytest.mark.parametrize("activation_dtype", [torch.float16, torch.bfloat16])
def test_w4a16_uses_fp16_checkpoint_scale_precision(cuda, factory, activation_dtype):
    """W4A16 always preserves the checkpoint's FP16 scale precision."""
    torch.manual_seed(366)
    module = factory(64, 32)
    module.bias.data.zero_()
    # Exactly halfway between adjacent BF16 values around one, but exactly
    # representable in FP16.  The old activation-dtype dequant path loses it.
    module.scales.fill_(1.00390625)
    activation = torch.ones(3, 64, dtype=activation_dtype)
    weight = module.dequantize()
    expected = torch.mm(
        activation.to(device=cuda, dtype=torch.float16),
        weight.to(device=cuda, dtype=torch.float16).T.contiguous(),
        out_dtype=torch.float32,
    ).to(activation_dtype)

    actual = module.to(cuda)(activation.to(cuda))

    if activation_dtype is torch.bfloat16:
        bf16_rounded = torch.mm(
            activation.to(cuda),
            weight.to(device=cuda, dtype=torch.bfloat16).T.contiguous(),
            out_dtype=torch.float32,
        ).to(torch.bfloat16)
        assert not torch.equal(expected, bf16_rounded)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("factory", [_awq_module, _gptq_module], ids=["awq", "gptq"])
def test_w4a16_saturates_bf16_activations_to_finite_fp16_range(cuda, factory):
    torch.manual_seed(367)
    module = factory(64, 32)
    module.bias.data.zero_()
    # Power-of-two weights keep every FP32 partial sum exact independent of
    # the reduction order used by Triton and cuBLAS.
    module.scales.fill_(0.5)
    activation = torch.full((3, 64), 70_000.0, dtype=torch.bfloat16)
    activation[1].neg_()
    weight = module.dequantize().to(device=cuda, dtype=torch.float16)
    expected = torch.mm(
        activation.float().clamp(-65_504.0, 65_504.0).to(
            device=cuda, dtype=torch.float16
        ),
        weight.T.contiguous(),
        out_dtype=torch.float32,
    ).to(torch.bfloat16)

    actual = module.to(cuda)(activation.to(cuda))

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("factory", [_awq_module, _gptq_module], ids=["awq", "gptq"])
def test_w4a16_does_not_hide_fp16_infinite_activations(cuda, factory):
    module = factory(64, 32).to(cuda)
    activation = torch.full((1, 64), torch.inf, device=cuda, dtype=torch.float16)

    actual = module(activation)

    assert not torch.isfinite(actual).all()


def test_nvfp4_qkv_pack_matches_separate_and_replays_cuda_graph(cuda):
    """One activation quantization must preserve Q/K/V values and graph safety."""
    from kairyu.models.packed_linear import NvFp4LinearPack

    torch.manual_seed(168)
    modules = tuple(_nvfp4_module(64, rows).to(cuda) for rows in (64, 32, 32))
    for module in modules[1:]:
        module.input_scale.copy_(modules[0].input_scale)
        module.weight_scale_2.copy_(modules[0].weight_scale_2)
    pack = NvFp4LinearPack.create(modules)
    assert pack is not None
    hidden = torch.randn(3, 64, device=cuda, dtype=torch.bfloat16)
    expected = tuple(module(hidden) for module in modules)
    actual = pack(hidden)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)

    for _ in range(3):
        pack(hidden)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = pack(hidden)
    graph.replay()
    torch.cuda.synchronize()
    for result, reference in zip(replayed, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_fused_kernel_rejects_unsupported_activation_dtype(cuda):
    module = _awq_module().to(cuda)
    with pytest.raises(RuntimeError, match="float16/bfloat16"):
        module(torch.randn(2, 64, device=cuda, dtype=torch.float32))


@pytest.mark.parametrize(
    ("scheme", "factory"),
    [
        ("fp8", _fp8_module),
        ("int8", _int8_module),
        ("awq", _awq_module),
        ("gptq", _gptq_module),
        ("nvfp4", _nvfp4_module),
    ],
)
def test_fused_kernel_rejects_unsupported_compute_capability(
    cuda, monkeypatch, scheme: str, factory: Callable
):
    module = factory().to(cuda)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (7, 5))
    with pytest.raises(RuntimeError, match=rf"{scheme} fused kernel requires SM"):
        module(torch.randn(2, 64, device=cuda, dtype=torch.bfloat16))


def test_static_fp8_uses_checkpoint_input_scale(cuda):
    from kairyu.quant.fp8 import quantize_fp8
    from kairyu.quant.linear import Fp8Linear

    torch.manual_seed(2)
    module = Fp8Linear(64, 32, bias=False, activation_dynamic=False)
    qweight, weight_scale = quantize_fp8(torch.randn(32, 64), per_channel=False)
    module.weight.copy_(qweight)
    module.weight_scale = weight_scale
    module.input_scale.copy_(torch.tensor(0.01))
    x = torch.randn(7, 64).to(torch.bfloat16).float()
    expected = module.forward_reference(x)
    actual = module.to(cuda)(x.to(device=cuda, dtype=torch.bfloat16)).float().cpu()
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.40)


def test_dynamic_fp8_accepts_global_per_token_scale(cuda):
    torch.manual_seed(19)
    module = _fp8_module(in_features=64, out_features=32)
    x = torch.randn(7, 64).to(torch.bfloat16).float()
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = (amax * 1.25 / 448.0).clamp(min=1e-12)
    expected = module.forward_with_activation_scale(x, scale)

    module = module.to(cuda)
    x_cuda = x.to(device=cuda, dtype=torch.bfloat16)
    torch.testing.assert_close(
        module.activation_amax(x_cuda).cpu(),
        amax,
        rtol=0,
        atol=0,
    )
    actual = module.forward_with_activation_scale(
        x_cuda, scale.to(cuda)
    ).float().cpu()
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.40)
