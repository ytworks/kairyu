"""D5: quant detection extensions, hardware profiles, checkpoint reader."""

import pytest
import torch

from kairyu.engine.core.hw_profile import (
    EnvRecord,
    HardwareProfile,
    load_env_record,
    probe,
    write_env_record,
)
from kairyu.engine.core.quant_config import QuantConfig, QuantMethod, detect_quantization
from kairyu.engine.core.weights import CheckpointReader


class TestQuantDetection:
    def test_modelopt_nvfp4(self):
        # nvidia/*-NVFP4 checkpoint schema (roadmap §7 refs)
        config = {
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "group_size": 16,
            }
        }
        quant = detect_quantization(config)
        assert quant.method is QuantMethod.NVFP4
        assert quant.weight_bits == 4
        assert quant.group_size == 16

    def test_modelopt_fp8(self):
        config = {"quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"}}
        assert detect_quantization(config).method is QuantMethod.FP8

    def test_modelopt_unknown_algo_fails(self):
        config = {"quantization_config": {"quant_method": "modelopt", "quant_algo": "INT3"}}
        with pytest.raises(ValueError, match="modelopt"):
            detect_quantization(config)

    def test_compressed_tensors_int8_w8a8(self):
        config = {
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "weights": {"type": "int", "num_bits": 8},
                        "input_activations": {"type": "int", "num_bits": 8},
                    }
                },
            }
        }
        quant = detect_quantization(config)
        assert quant.method is QuantMethod.INT8
        assert quant.activation_bits == 8

    def test_compressed_tensors_fp4_rejected(self):
        # m14 A8: CT-FP4 uses different names + an inverted global scale vs
        # modelopt — flowing it into the modelopt module would corrupt weights
        config = {
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {"weights": {"type": "float", "num_bits": 4, "group_size": 16}}
                },
            }
        }
        with pytest.raises(ValueError, match="modelopt"):
            detect_quantization(config)

    def test_existing_schemes_unchanged(self):
        assert detect_quantization({}).method is QuantMethod.NONE
        awq = {"quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128}}
        assert detect_quantization(awq).method is QuantMethod.AWQ


class TestHardwareProfile:
    def test_probe_returns_cpu_profile_without_cuda(self):
        profile = probe()
        if not torch.cuda.is_available():
            assert profile.arch == "cpu"
            assert profile.kernel_tier == "torch"

    def test_best_format_matrix(self):
        sm120 = HardwareProfile(arch="cuda", sm=120, formats=("bf16", "fp8", "nvfp4"))
        assert sm120.best_format(QuantConfig(QuantMethod.NVFP4)) == "nvfp4"
        assert sm120.best_format(QuantConfig(QuantMethod.FP8)) == "fp8"
        assert sm120.kernel_tier == "fa2"

        sm90 = HardwareProfile(arch="cuda", sm=90, formats=("bf16", "fp8"))
        assert sm90.best_format(QuantConfig(QuantMethod.FP8)) == "fp8"
        assert sm90.kernel_tier == "full"
        with pytest.raises(ValueError, match="nvfp4"):
            sm90.best_format(QuantConfig(QuantMethod.NVFP4))

        sm80 = HardwareProfile(arch="cuda", sm=80, formats=("bf16", "int8"))
        assert sm80.best_format(QuantConfig(QuantMethod.AWQ)) == "w4a16"  # always ok
        assert sm80.best_format(QuantConfig(QuantMethod.INT8)) == "int8"
        with pytest.raises(ValueError, match="fp8"):
            sm80.best_format(QuantConfig(QuantMethod.FP8))

    def test_bf16_always_supported(self):
        cpu = HardwareProfile(arch="cpu")
        assert cpu.best_format(QuantConfig(QuantMethod.NONE)) == "bf16"

    def test_env_record_roundtrip(self, tmp_path):
        record = EnvRecord(
            date="2026-07-03",
            profile=HardwareProfile(arch="cpu"),
            library_versions={"torch": torch.__version__},
        )
        path = write_env_record(record, tmp_path)
        loaded = load_env_record(path)
        assert loaded["date"] == "2026-07-03"
        assert loaded["profile"]["arch"] == "cpu"
        assert "torch" in loaded["library_versions"]


@pytest.fixture()
def sharded_checkpoint(tmp_path):
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(0)
    tensors = {
        "model.embed.weight": torch.randn(8, 4, generator=generator),
        "model.layers.0.w": torch.randn(6, 6, generator=generator),
        "model.layers.1.w": torch.randn(6, 6, generator=generator),
        "lm_head.weight": torch.randn(8, 4, generator=generator),
    }
    shard0 = {k: tensors[k] for k in ("model.embed.weight", "model.layers.0.w")}
    shard1 = {k: tensors[k] for k in ("model.layers.1.w", "lm_head.weight")}
    save_file(shard0, tmp_path / "model-00001-of-00002.safetensors")
    save_file(shard1, tmp_path / "model-00002-of-00002.safetensors")
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {
            **{k: "model-00001-of-00002.safetensors" for k in shard0},
            **{k: "model-00002-of-00002.safetensors" for k in shard1},
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(__import__("json").dumps(index))
    return tmp_path, tensors


class TestCheckpointReader:
    def test_reads_all_tensors_across_shards(self, sharded_checkpoint):
        path, tensors = sharded_checkpoint
        reader = CheckpointReader(path)
        assert set(reader.names()) == set(tensors)
        for name, expected in tensors.items():
            assert torch.equal(reader.tensor(name), expected)

    def test_items_iterates_everything(self, sharded_checkpoint):
        path, tensors = sharded_checkpoint
        loaded = dict(CheckpointReader(path).items())
        assert set(loaded) == set(tensors)

    def test_get_slice_matches_full_tensor(self, sharded_checkpoint):
        path, tensors = sharded_checkpoint
        reader = CheckpointReader(path)
        sliced = reader.get_slice("model.layers.0.w", dim=0, start=2, end=5)
        assert torch.equal(sliced, tensors["model.layers.0.w"][2:5])
        sliced = reader.get_slice("model.layers.1.w", dim=1, start=0, end=3)
        assert torch.equal(sliced, tensors["model.layers.1.w"][:, 0:3])

    def test_single_file_checkpoint(self, tmp_path):
        from safetensors.torch import save_file

        tensor = torch.ones(3, 3)
        file = tmp_path / "model.safetensors"
        save_file({"w": tensor}, file)
        assert torch.equal(CheckpointReader(file).tensor("w"), tensor)
        # directory without index: glob fallback
        assert torch.equal(CheckpointReader(tmp_path).tensor("w"), tensor)

    def test_missing_tensor_and_path_fail_loudly(self, sharded_checkpoint, tmp_path):
        path, _ = sharded_checkpoint
        with pytest.raises(KeyError):
            CheckpointReader(path).tensor("ghost")
        with pytest.raises(ValueError, match="checkpoint"):
            CheckpointReader(tmp_path / "nope")
