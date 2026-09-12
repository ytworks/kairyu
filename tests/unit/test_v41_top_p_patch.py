import importlib.util
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/qwen3.8-deepseek-v4.1-8gpu"


def load():
    spec = importlib.util.spec_from_file_location("top_p_patch", EXAMPLE / "patch_top_p.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinned_source_changes_once_and_second_apply_is_noop(tmp_path, monkeypatch):
    patcher = load()
    source = "before\n" + patcher.ANCHOR + "\nafter\n"
    path = tmp_path / "topk_topp_triton.py"
    path.write_text(source)
    monkeypatch.setattr(patcher, "SOURCE_SHA256", patcher.digest(source.encode()))
    monkeypatch.setattr(
        patcher,
        "PATCHED_SHA256",
        patcher.digest(source.replace(patcher.ANCHOR, patcher.REPLACEMENT).encode()),
    )
    assert patcher.patch(path) is True
    after = path.read_bytes()
    assert patcher.patch(path) is False
    assert path.read_bytes() == after
    assert "if not (pivot_logit < M):" in after.decode()


def test_drift_fails_without_writing_even_if_patch_marker_exists(tmp_path):
    patcher = load()
    path = tmp_path / "source.py"
    for source in ["changed upstream source", patcher.REPLACEMENT + "\nforeign change"]:
        path.write_text(source)
        with pytest.raises(ValueError, match="Unrecognized"):
            patcher.patch(path)
        assert path.read_text() == source


def test_anchor_missing_or_duplicated_fails():
    patcher = load()
    for source in ["", patcher.ANCHOR * 2]:
        with pytest.raises(ValueError, match="exactly one"):
            patcher.transform(source)


def test_child_runs_top_p_patch_without_changing_thinking_force():
    patcher = load()
    assert "thinking_budget" not in patcher.TARGET
    assert "1.0e9" not in patcher.REPLACEMENT
    dockerfile = (EXAMPLE / "vllm-sm120.Dockerfile").read_text()
    assert "COPY patch_top_p.py /opt/kairyu/patch_top_p.py" in dockerfile
    assert "RUN python3 /opt/kairyu/patch_top_p.py" in dockerfile
