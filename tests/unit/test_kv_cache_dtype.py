"""CPU-only contracts for #170 KV-cache dtype startup plumbing.

These tests prove validation and propagation only.  They do not label mocked
SM120 objects as GPU correctness evidence; the FP8 kernel/attention bake stays
in the real-GPU suite.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from kairyu.engine.config_validation import validate_backend_options
from kairyu.engine.core.kv_cache_dtype import (
    resolve_kv_cache_dtype,
    validate_kv_cache_dtype,
)
from kairyu.engine.core.worker import (
    DistTPLauncher,
    make_handshake,
    validate_handshake,
)
from kairyu.engine.kairyu_backend import KairyuBackend, build_engine_loop
from kairyu.engine.tokenizer import ToyTokenizer
from kairyu.engine.zmq_backend import (
    EngineServiceError,
    ZmqEngineBackend,
    _kv_cache_dtype_from_startup_frame,
    _validate_startup_kv_cache_dtype_identity,
)


@pytest.mark.parametrize("value", ("auto", "bfloat16"))
def test_public_kv_cache_dtype_names(value):
    assert validate_kv_cache_dtype(value) == value


def test_public_fp8_request_reports_failed_bake_and_stays_disabled():
    with pytest.raises(RuntimeError, match="G4 E-KV.*bake failed"):
        validate_kv_cache_dtype("fp8_e4m3")


@pytest.mark.parametrize("value", (None, "", "bf16", "fp8", torch.bfloat16))
def test_public_kv_cache_dtype_rejects_unknown_values(value):
    with pytest.raises(ValueError, match="kv_cache_dtype must be one of"):
        validate_kv_cache_dtype(value)


@pytest.mark.parametrize("backend", ("kairyu", "kairyu-proc"))
def test_native_config_default_remains_auto_and_explicit_needs_real_model(backend):
    validate_backend_options(backend, {})
    with pytest.raises(ValueError, match="real model_path"):
        validate_backend_options(backend, {"kv_cache_dtype": "bfloat16"})


@pytest.mark.parametrize("backend", ("kairyu", "kairyu-proc"))
@pytest.mark.parametrize("value", ("auto", "bfloat16"))
def test_native_config_accepts_public_kv_dtype_with_real_model(backend, value):
    validate_backend_options(
        backend,
        {
            "model_path": "/model",
            "kv_cache_dtype": value,
        },
    )


@pytest.mark.parametrize("backend", ("kairyu", "kairyu-proc"))
def test_native_config_rejects_failed_fp8_bake_before_startup(backend):
    with pytest.raises(RuntimeError, match="G4 E-KV.*bake failed"):
        validate_backend_options(
            backend,
            {
                "model_path": "/model",
                "kv_cache_dtype": "fp8_e4m3",
            },
        )


def test_pd_rejects_explicit_bfloat16_kv_cache_dtype():
    with pytest.raises(ValueError, match="does not support P-D separation"):
        validate_backend_options(
            "kairyu",
            {
                "model_path": "/model",
                "pd_separation": True,
                "kv_cache_dtype": "bfloat16",
            },
        )


def test_fp8_rejects_before_other_candidate_specific_options():
    with pytest.raises(RuntimeError, match="G4 E-KV.*bake failed"):
        validate_backend_options(
            "kairyu",
            {
                "model_path": "/model",
                "speculative": "ngram",
                "kv_cache_dtype": "fp8_e4m3",
            },
        )


def test_custom_runner_rejects_explicit_kv_cache_dtype_before_hardware_probe():
    with pytest.raises(ValueError, match="real model_path"):
        build_engine_loop(runner=object(), kv_cache_dtype="bfloat16")


def test_single_rank_rejects_failed_fp8_bake_before_model_load():
    with pytest.raises(RuntimeError, match="G4 E-KV.*bake failed"):
        build_engine_loop(
            model_path="/missing-model",
            kv_cache_dtype="fp8_e4m3",
        )


def test_tp_rejects_failed_fp8_bake_before_process_spawn():
    with pytest.raises(RuntimeError, match="G4 E-KV.*bake failed"):
        DistTPLauncher(
            "/missing-model",
            tp=2,
            num_pages=64,
            page_size=16,
            vocab=[],
            kv_cache_dtype="fp8_e4m3",
        )


def test_auto_preserves_compute_dtype():
    assert (
        resolve_kv_cache_dtype(
            "auto",
            torch.float32,
            SimpleNamespace(arch="cpu", sm=None),
            object(),
            None,
        )
        is torch.float32
    )


def test_explicit_bfloat16_requires_real_bfloat16_compute():
    model_config = SimpleNamespace(is_mla=False)
    profile = SimpleNamespace(arch="cuda", sm=120)
    assert (
        resolve_kv_cache_dtype(
            "bfloat16",
            torch.bfloat16,
            profile,
            object(),
            model_config,
        )
        is torch.bfloat16
    )
    with pytest.raises(RuntimeError, match="BF16 model compute"):
        resolve_kv_cache_dtype(
            "bfloat16",
            torch.float32,
            SimpleNamespace(arch="cpu", sm=None),
            object(),
            model_config,
        )


@pytest.mark.parametrize(
    "profile",
    (
        SimpleNamespace(arch="cpu", sm=None),
        SimpleNamespace(arch="cuda", sm=100),
        SimpleNamespace(arch="cuda", sm=120),
    ),
)
def test_fp8_resolver_stays_disabled_on_every_runtime(profile):
    with pytest.raises(RuntimeError, match="G4 E-KV.*bake failed"):
        resolve_kv_cache_dtype(
            "fp8_e4m3",
            torch.bfloat16,
            profile,
            SimpleNamespace(supports_fp8_kv=True),
            SimpleNamespace(is_mla=False),
        )


def test_single_rank_passes_resolved_dtype_to_pool(monkeypatch):
    """Structural wiring test only; no mocked GPU result is treated as a bake."""

    from kairyu.engine.core import attention, hw_profile, kv_pool, model_runner, sampler
    from kairyu.models import loader

    seen = {}
    profile = SimpleNamespace(arch="cuda", sm=120)
    decision = SimpleNamespace(requested="auto", resolved="flashinfer")
    backend = SimpleNamespace(selection_decision=decision)
    config = SimpleNamespace(
        is_mla=False,
        num_hidden_layers=1,
        num_key_value_heads=1,
        kv_cache_num_heads=1,
        kv_cache_head_dim=8,
        kv_cache_v_head_dim=8,
        vocab_size=50_000,
    )
    generation = SimpleNamespace(eos_token_id=None, stop_token_ids=())

    class FakeModel:
        def to(self, device):
            seen["model_device"] = device
            return self

    class FakePool:
        @classmethod
        def for_cache(cls, cache, model_config, *, dtype, device):
            seen["pool"] = {
                "cache": cache,
                "model_config": model_config,
                "dtype": dtype,
                "device": device,
            }
            return object()

    class FakeRunner:
        def __init__(self, model, pool, **kwargs):
            seen["runner"] = (model, pool, kwargs)

    monkeypatch.setattr(hw_profile, "probe", lambda: profile)
    monkeypatch.setattr(
        attention,
        "select_backend",
        lambda selected, *, device: backend,
    )
    monkeypatch.setattr(
        loader,
        "load_model",
        lambda *args, **kwargs: (FakeModel(), config, generation),
    )
    monkeypatch.setattr(
        loader,
        "load_generation_defaults",
        lambda *args, **kwargs: generation,
    )
    monkeypatch.setattr(kv_pool, "PagedKVPool", FakePool)
    monkeypatch.setattr(model_runner, "PagedModelRunner", FakeRunner)
    monkeypatch.setattr(sampler, "Sampler", lambda **kwargs: object())

    loop, _, _ = build_engine_loop(
        model_path="/model",
        tokenizer=ToyTokenizer(),
        kv_cache_dtype="bfloat16",
    )

    assert seen["pool"]["dtype"] is torch.bfloat16
    assert seen["pool"]["device"] == "cuda"
    assert loop.kv_cache_dtype_requested == "bfloat16"
    assert loop.kv_cache_dtype_resolved == "bfloat16"


def test_toy_backend_reports_no_managed_kv_cache():
    backend = KairyuBackend()
    assert backend.kv_cache_dtype_requested == "auto"
    assert backend.kv_cache_dtype_resolved == "not-applicable"


def test_tp_handshake_rejects_kv_dtype_mismatch(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "test"}')
    handshake = make_handshake(
        str(tmp_path),
        64,
        16,
        kv_cache_dtype_requested="fp8_e4m3",
        kv_cache_dtype_resolved="fp8_e4m3",
    )
    with pytest.raises(RuntimeError, match="KV dtype identity"):
        validate_handshake(
            handshake,
            str(tmp_path),
            64,
            16,
            kv_cache_dtype_requested="fp8_e4m3",
            kv_cache_dtype_resolved="bfloat16",
        )


def test_zmq_config_and_startup_wire_reject_failed_fp8_bake():
    with pytest.raises(RuntimeError, match="G4 E-KV.*bake failed"):
        ZmqEngineBackend(
            model_path="/model",
            kv_cache_dtype="fp8_e4m3",
        )
    with pytest.raises(EngineServiceError, match="G4 E-KV.*bake failed"):
        _kv_cache_dtype_from_startup_frame(
            {
                "kv_cache_dtype_requested": "fp8_e4m3",
                "kv_cache_dtype_resolved": "fp8_e4m3",
            }
        )


def test_zmq_startup_wire_rejects_partial_kv_dtype_metadata():
    with pytest.raises(EngineServiceError, match="resolved"):
        _kv_cache_dtype_from_startup_frame(
            {"kv_cache_dtype_requested": "auto"}
        )


def test_zmq_startup_identity_preserves_auto_legacy_compatibility():
    _validate_startup_kv_cache_dtype_identity("auto", None, None)
    for resolved in ("float16", "float32", "bfloat16", "not-applicable"):
        _validate_startup_kv_cache_dtype_identity(
            "auto",
            "auto",
            resolved,
        )


@pytest.mark.parametrize(
    ("configured", "requested", "resolved", "match"),
    (
        ("fp8_e4m3", None, None, "did not report"),
        ("fp8_e4m3", "auto", "bfloat16", "request mismatch"),
        ("fp8_e4m3", "fp8_e4m3", "bfloat16", "inconsistent"),
        ("auto", "auto", "fp8_e4m3", "inconsistent"),
    ),
)
def test_zmq_startup_identity_fails_closed(
    configured,
    requested,
    resolved,
    match,
):
    with pytest.raises(EngineServiceError, match=match):
        _validate_startup_kv_cache_dtype_identity(
            configured,
            requested,
            resolved,
        )
