import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/qwen3.8-deepseek-v4.1-8gpu"
spec = importlib.util.spec_from_file_location("masked_kv_patch", EXAMPLE / "patch_masked_kv.py")
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)


def test_source_drift_rejected_without_partial_mutation(tmp_path):
    for name in patcher.SOURCE_SHA256:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text("changed upstream source")
    with pytest.raises(ValueError, match="Unrecognized FlashInfer source"):
        patcher.patch(tmp_path)
    assert all(
        (tmp_path / name).read_text() == "changed upstream source" for name in patcher.SOURCE_SHA256
    )


def test_ambiguous_anchor_fails_closed():
    with pytest.raises(ValueError, match="exactly one"):
        patcher.replace_once("duplicate duplicate", "duplicate", "replacement")
    with pytest.raises(ValueError, match="exactly one"):
        patcher.replace_once("missing", "anchor", "replacement")


def test_jit_workspace_cannot_reuse_mounted_unpatched_cache():
    dockerfile = (EXAMPLE / "vllm-sm120.Dockerfile").read_text()
    assert "ENV FLASHINFER_WORKSPACE_BASE=/opt/kairyu/flashinfer-masked-kv-v1" in dockerfile
    assert "RUN python3 /opt/kairyu/patch_masked_kv.py" in dockerfile


def test_decode_declares_its_zero_row_without_prefill_header():
    # Decode is compiled separately; common/kv_cache_io.cuh is not included.
    source = (
        "namespace flashinfer::sparse_mla_sm120 {\n"
        "          section_kv + (size_t)block_idx_g * section_stride + "
        "(size_t)local_idx_g * IO_STRIDE;\n      cp_async_bulk_g2s(kv_fp8_dst"
    )
    result = patcher.transform("decode_dsv4_kernel.cuh", source)
    declaration = "uint8_t kKairyuDecodeInvalidKVZeroRow[2048] = {};"
    assert declaration in result
    assert result.index(declaration) < result.index(": kKairyuDecodeInvalidKVZeroRow;")
    assert "cp_async_bulk_g2s(kv_fp8_dst" in result
