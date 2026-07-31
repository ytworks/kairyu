"""Static guards for the production FlashInfer SM120 AOT matrix."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = (_REPO_ROOT / "Dockerfile.cuda").read_text(encoding="utf-8")


def _aot_override() -> str:
    return next(line for line in _DOCKERFILE.splitlines() if "config={" in line)


def test_sm120_aot_covers_bf16_query_with_e4m3_kv() -> None:
    override = _aot_override()

    assert '"fa2_head_dim": [(64, 64), (128, 128)]' in override
    assert (
        '"f16_dtype": [__import__("torch").float16, '
        '__import__("torch").bfloat16]'
    ) in override
    assert (
        '"f8_dtype": [__import__("torch").float8_e4m3fn]'
        in override
    )
    assert '"f8_dtype": []' not in override
    assert '"fa3_head_dim": []' in override
    assert "float8_e5m2" not in override


def test_docker_build_checks_the_scoped_aot_override_was_applied() -> None:
    assert (
        """grep -F -n '"f8_dtype": """
        """[__import__("torch").float8_e4m3fn]'"""
    ) in _DOCKERFILE


def test_runtime_payload_is_revision_independent_and_nonroot_cache_safe() -> None:
    payload = "FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04 AS runtime-payload"
    final = "FROM runtime-payload AS base"

    assert payload in _DOCKERFILE
    assert final in _DOCKERFILE
    assert _DOCKERFILE.index(final) > _DOCKERFILE.index('CMD ["/etc/kairyu/config.yaml"]')
    assert _DOCKERFILE.index("ARG KAIRYU_VCS_REF") > _DOCKERFILE.index(final)
    assert _DOCKERFILE.count("ARG KAIRYU_VCS_REF") == 1
    assert "FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer" in _DOCKERFILE
