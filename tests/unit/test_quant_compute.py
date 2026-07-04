"""m14 gates: bit-pattern pins, round-trips, and the flagship
quantized-checkpoint-runs-on-CPU integration per scheme."""

import json

import pytest
import torch

from kairyu.quant import awq, fp8, gptq, int8, nvfp4

transformers = pytest.importorskip("transformers")


class TestBitPatterns:
    def test_awq_pack_order_hand_computed(self):
        # one row, cols 0..7 hold values equal to their index: nibble i of the
        # packed word must hold column ORDER[i] = [0,2,4,6,1,3,5,7]
        values = torch.arange(8, dtype=torch.int32).reshape(1, 8)
        packed = awq.pack_awq(values)
        assert packed.shape == (1, 1)
        word = packed[0, 0].item() & 0xFFFFFFFF
        nibbles = [(word >> (4 * i)) & 0xF for i in range(8)]
        assert nibbles == [0, 2, 4, 6, 1, 3, 5, 7]
        assert torch.equal(awq.unpack_awq(packed), values)

    def test_gptq_row_pack_lsb_first(self):
        values = torch.arange(8, dtype=torch.int32).reshape(8, 1)
        packed = gptq.pack_gptq_rows(values)
        word = packed[0, 0].item() & 0xFFFFFFFF
        nibbles = [(word >> (4 * i)) & 0xF for i in range(8)]
        assert nibbles == [0, 1, 2, 3, 4, 5, 6, 7]  # sequential, no reorder
        assert torch.equal(gptq.unpack_gptq_rows(packed), values)

    def test_gptq_zero_offset_convention(self):
        torch.manual_seed(0)
        weight = torch.randn(16, 16)  # out >= 8: qzeros pack along OUT axis
        qweight, qzeros_stored, scales, g_idx = gptq.quantize_gptq(weight, group_size=8)
        # stored zeros are z-1: restoring +1 must round-trip the weight
        restored = gptq.dequantize_gptq(qweight, qzeros_stored, scales, g_idx, 8)
        assert (restored - weight).abs().max().item() < 0.2  # 4-bit tolerance

    def test_nvfp4_nibble_layout(self):
        # weight [1, 2]: values 1.0 (LUT idx 2) and -6.0 (sign|idx7 = 0xF)
        weight = torch.tensor([[1.0, -6.0]])
        packed, scale, global_scale = nvfp4.quantize_nvfp4(
            torch.cat([weight, torch.zeros(1, 14)], dim=1)
        )
        byte0 = packed[0, 0].item()
        assert byte0 & 0xF == 2  # LOW nibble = even element (1.0)
        assert (byte0 >> 4) & 0xF == 0xF  # HIGH nibble = odd element (-6.0)

    def test_nvfp4_rne_boundaries(self):
        # M1: each boundary rounds to the neighbor with the EVEN LUT index.
        # LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        # 0.25->0(0.0) 0.75->2(1.0) 1.25->2(1.0) 1.75->4(2.0)
        # 2.5->4(2.0) 3.5->6(4.0) 5.0->6(4.0)
        values = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        indices = nvfp4._cast_to_fp4_indices(values)
        assert indices.tolist() == [0, 2, 2, 4, 4, 6, 6]
        # the packed value is the LUT entry at that index (magnitude preserved)
        assert nvfp4.E2M1_LUT[indices].tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]

    def test_fp8_quantize_clamps_before_cast(self):
        # torch CPU fp8 cast is non-saturating: without the clamp this is NaN
        weight = torch.tensor([[1000.0, -1000.0]])
        q, scale = fp8.quantize_fp8(weight, per_channel=False)
        assert not torch.isnan(q.to(torch.float32)).any()
        restored = fp8.dequantize_fp8(q, scale)
        assert (restored - weight).abs().max().item() / 1000.0 < 0.05


class TestRoundTrips:
    @pytest.mark.parametrize("per_channel", [True, False])
    def test_fp8(self, per_channel):
        torch.manual_seed(1)
        weight = torch.randn(8, 32)
        q, scale = fp8.quantize_fp8(weight, per_channel=per_channel)
        assert q.dtype == torch.float8_e4m3fn
        # e4m3 relative precision is ~2^-3 near the top of each binade
        assert (fp8.dequantize_fp8(q, scale) - weight).abs().max().item() < 0.15

    def test_int8_exact_int32_accumulation(self):
        torch.manual_seed(2)
        weight = torch.randn(8, 16)
        x = torch.randn(3, 16)
        w_q, w_scale = int8.quantize_int8_weight(weight)
        x_q, x_scale = int8.quantize_int8_activation(x)
        ours = int8.int8_w8a8_matmul(x_q, x_scale, w_q, w_scale)
        # reference computed the same way in fp64 proves int32 accumulation exact
        expected = (
            x_q.to(torch.float64) @ w_q.to(torch.float64).t()
        ) * x_scale.to(torch.float64) * w_scale.t().to(torch.float64)
        assert torch.allclose(ours.to(torch.float64), expected)

    def test_awq(self):
        torch.manual_seed(3)
        weight = torch.randn(16, 32)  # out, in
        qweight, qzeros, scales, = awq.quantize_awq(weight, group_size=16)
        assert qweight.shape == (32, 2)  # [in, out//8]
        assert qzeros.shape == (2, 2)  # [in//g, out//8]
        assert scales.shape == (2, 16)  # [in//g, out]
        restored = awq.dequantize_awq(qweight, qzeros, scales, 16)
        assert restored.shape == weight.shape
        assert (restored - weight).abs().max().item() < 0.3

    def test_nvfp4(self):
        torch.manual_seed(4)
        weight = torch.randn(8, 32)
        packed, scale, global_scale = nvfp4.quantize_nvfp4(weight)
        assert packed.shape == (8, 16) and packed.dtype == torch.uint8
        assert scale.shape == (8, 2) and scale.dtype == torch.float8_e4m3fn
        restored = nvfp4.dequantize_nvfp4(packed, scale, global_scale)
        assert (restored - weight).abs().max().item() < 0.6  # 4-bit blockwise


# --- flagship D5: quantized checkpoint loads and RUNS through the engine ---

TINY = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=128,
    vocab_size=256,
    max_position_embeddings=512,
)
_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

_QUANT_CONFIGS = {
    "fp8": {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "weights": {"type": "float", "num_bits": 8, "strategy": "channel"},
                "input_activations": {"type": "float", "num_bits": 8, "dynamic": True},
            }
        },
        "ignore": ["lm_head"],
    },
    "int8": {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "weights": {"type": "int", "num_bits": 8, "strategy": "channel"},
                "input_activations": {"type": "int", "num_bits": 8, "dynamic": True},
            }
        },
        "ignore": ["lm_head"],
    },
    "awq": {"quant_method": "awq", "bits": 4, "group_size": 16, "version": "gemm",
            "zero_point": True},
    "gptq": {"quant_method": "gptq", "bits": 4, "group_size": 16, "desc_act": False,
             "sym": True},
    "nvfp4": {"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16},
}


def _quantize_projection(scheme: str, weight: torch.Tensor) -> dict[str, torch.Tensor]:
    if scheme == "fp8":
        q, scale = fp8.quantize_fp8(weight)
        return {"weight": q, "weight_scale": scale}
    if scheme == "int8":
        q, scale = int8.quantize_int8_weight(weight)
        return {"weight": q, "weight_scale": scale}
    if scheme == "awq":
        qweight, qzeros, scales = awq.quantize_awq(weight, group_size=16)
        return {"qweight": qweight, "qzeros": qzeros, "scales": scales}
    if scheme == "gptq":
        qweight, qzeros, scales, g_idx = gptq.quantize_gptq(weight, group_size=16)
        return {"qweight": qweight, "qzeros": qzeros, "scales": scales, "g_idx": g_idx}
    q, scale, global_scale = nvfp4.quantize_nvfp4(weight)
    return {
        "weight": q,
        "weight_scale": scale,
        "weight_scale_2": global_scale,
        "input_scale": torch.ones(()),
    }


@pytest.fixture(scope="module")
def fp32_reference(tmp_path_factory):
    torch.manual_seed(41)
    hf_model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY))
    hf_model = hf_model.to(torch.float32).eval()
    path = tmp_path_factory.mktemp("fp32")
    hf_model.save_pretrained(path, safe_serialization=True)
    return path, hf_model


def _write_quantized_checkpoint(scheme: str, source_path, hf_model, target) -> None:
    from safetensors.torch import save_file

    state = dict(hf_model.state_dict())
    quantized: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if any(f".{proj}.weight" in name for proj in _PROJECTIONS):
            prefix = name.rsplit(".", 1)[0]
            for suffix, packed in _quantize_projection(scheme, tensor).items():
                quantized[f"{prefix}.{suffix}"] = packed.contiguous()
        else:
            quantized[name] = tensor.contiguous()
    save_file(quantized, target / "model.safetensors")
    config = json.loads((source_path / "config.json").read_text())
    config["quantization_config"] = _QUANT_CONFIGS[scheme]
    (target / "config.json").write_text(json.dumps(config))


def _engine_greedy(path, prompt, max_new=12):
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.models.loader import load_model

    model, config, _ = load_model(path)
    cache = RadixKVCache(num_pages=128, page_size=4)
    scheduler = Scheduler(cache, max_num_batched_tokens=64, page_size=4)
    pool = PagedKVPool.for_cache(cache, config)
    engine = EngineCore(scheduler, PagedModelRunner(model, pool, sampler=Sampler()))
    engine.add_request(
        EngineRequest("a", prompt, max_new_tokens=max_new, sampling=EngineSampling())
    )
    return engine.run_to_completion()["a"]


@pytest.mark.parametrize("scheme", ["fp8", "int8", "awq", "gptq", "nvfp4"])
def test_quantized_checkpoint_runs_full_engine(scheme, fp32_reference, tmp_path):
    source_path, hf_model = fp32_reference
    target = tmp_path / scheme
    target.mkdir()
    _write_quantized_checkpoint(scheme, source_path, hf_model, target)

    torch.manual_seed(43)
    prompt = tuple(torch.randint(0, 256, (10,)).tolist())
    reference = _engine_greedy(source_path, prompt)
    quantized = _engine_greedy(target, prompt)
    assert len(quantized) == len(reference)
    if scheme in ("fp8", "int8"):  # 8-bit: greedy tokens should mostly survive
        agreement = sum(
            a == b for a, b in zip(quantized, reference, strict=True)
        ) / len(reference)
        assert agreement >= 0.5, f"{scheme}: agreement {agreement}"
    # 4-bit at hidden-64 is lossy by construction: assert non-degenerate output
    assert len(set(quantized)) > 1, f"{scheme}: degenerate output {quantized}"
