"""CPU startup/observability contracts for opt-in FP32 logits (#364)."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
import torch

from kairyu.deploy.spec import load_deployment_spec
from kairyu.engine.config_validation import validate_backend_options
from kairyu.engine.core.engine_service import _startup_ready_frame
from kairyu.engine.core.logits_dtype import resolve_logits_dtype
from kairyu.engine.mock import MockBackend
from kairyu.engine.zmq_backend import (
    EngineServiceError,
    ZmqEngineBackend,
    _logits_dtype_from_startup_frame,
    _validate_startup_logits_dtype_identity,
)
from kairyu.entrypoints.server.app import create_app
from kairyu.orchestration.replica import ReplicaPool


def test_model_logits_dtype_rejects_an_unservable_resolution() -> None:
    with pytest.raises(ValueError, match="model logits dtype must resolve"):
        resolve_logits_dtype("model", torch.float64)


@pytest.mark.parametrize(
    ("backend", "options"),
    (
        ("mock", {"logits_dtype": "float32"}),
        ("openai", {"logits_dtype": "float32"}),
    ),
)
def test_non_native_builtin_backends_reject_logits_dtype(
    backend: str,
    options: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        validate_backend_options(backend, options)


def test_deployment_spec_preserves_native_logits_dtype_option() -> None:
    spec = load_deployment_spec(
        """
engines:
  precise:
    backend: kairyu-proc
    options:
      model_path: /model
      logits_dtype: float32
"""
    )

    assert spec.engines["precise"].options["logits_dtype"] == "float32"


def test_zmq_constructor_pins_logits_dtype_in_child_config() -> None:
    backend = ZmqEngineBackend(model_path="/model", logits_dtype="float32")

    assert backend._config["logits_dtype"] == "float32"
    assert backend.logits_dtype_requested == "float32"
    assert backend.logits_dtype_resolved is None


def test_zmq_constructor_rejects_fp32_without_real_model() -> None:
    with pytest.raises(ValueError, match="real model_path"):
        ZmqEngineBackend(logits_dtype="float32")


@pytest.mark.parametrize(
    ("requested", "resolved"),
    (
        ("model", "float16"),
        ("model", "bfloat16"),
        ("model", "float32"),
        ("model", "not-applicable"),
        ("float32", "float32"),
    ),
)
def test_logits_startup_metadata_accepts_truthful_pairs(
    requested: str,
    resolved: str,
) -> None:
    frame = {
        "logits_dtype_requested": requested,
        "logits_dtype_resolved": resolved,
    }
    assert _logits_dtype_from_startup_frame(frame) == (requested, resolved)
    _validate_startup_logits_dtype_identity(requested, requested, resolved)


@pytest.mark.parametrize(
    "frame",
    (
        {"logits_dtype_requested": "model"},
        {"logits_dtype_resolved": "float32"},
        {
            "logits_dtype_requested": "fp32",
            "logits_dtype_resolved": "float32",
        },
        {
            "logits_dtype_requested": "model",
            "logits_dtype_resolved": "float64",
        },
    ),
)
def test_logits_startup_metadata_rejects_malformed_pairs(
    frame: dict[str, object],
) -> None:
    with pytest.raises(EngineServiceError):
        _logits_dtype_from_startup_frame(frame)


def test_logits_startup_identity_is_fail_closed_for_explicit_fp32() -> None:
    _validate_startup_logits_dtype_identity("model", None, None)
    with pytest.raises(EngineServiceError, match="did not report"):
        _validate_startup_logits_dtype_identity("float32", None, None)
    with pytest.raises(EngineServiceError, match="request mismatch"):
        _validate_startup_logits_dtype_identity("float32", "model", "float32")
    with pytest.raises(EngineServiceError, match="resolution is inconsistent"):
        _validate_startup_logits_dtype_identity("float32", "float32", "bfloat16")


def test_child_startup_ready_frame_reports_resolved_logits_dtype() -> None:
    frame = _startup_ready_frame(
        SimpleNamespace(
            generation_defaults=None,
            logits_dtype_requested="float32",
            logits_dtype_resolved="float32",
        )
    )

    assert frame["logits_dtype_requested"] == "float32"
    assert frame["logits_dtype_resolved"] == "float32"


@pytest.mark.asyncio
async def test_backends_reports_requested_and_resolved_logits_dtype() -> None:
    backend = ZmqEngineBackend()
    backend.logits_dtype_resolved = "not-applicable"
    app = create_app(engines={"native": backend})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        payload = (await client.get("/backends")).json()

    entry = payload["engines"][0]
    assert entry["requested_logits_dtype"] == "model"
    assert entry["logits_dtype"] == "not-applicable"
    await backend.shutdown()


@pytest.mark.asyncio
async def test_backends_adopts_consistent_replica_logits_dtype() -> None:
    pool = ReplicaPool([MockBackend()])
    pool._backends_cache = {
        "engines": [
            {
                "requested_logits_dtype": "float32",
                "logits_dtype": "float32",
            },
            {
                "requested_logits_dtype": "float32",
                "logits_dtype": "float32",
            },
        ]
    }
    app = create_app(engines={"pool": pool})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        payload = (await client.get("/backends")).json()

    entry = payload["engines"][0]
    assert entry["requested_logits_dtype"] == "float32"
    assert entry["logits_dtype"] == "float32"
    assert entry["via_replica"]["requested_logits_dtype"] == "float32"
    assert entry["via_replica"]["logits_dtype"] == "float32"
