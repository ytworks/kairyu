"""Contracts of the Gemma 4 31B + EmbeddingGemma 2 example (chat and embedding replica pools)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import yaml

from kairyu.deploy.spec import load_deployment_spec

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/gemma-4-31b-8gpu"


def test_served_configuration_loads_and_agrees(monkeypatch):
    """The gateway starts on the committed YAML; both pools name the compose replicas."""
    monkeypatch.syspath_prepend(str(EXAMPLE))
    sys.modules.pop("control", None)
    control = importlib.import_module("control")  # import-time allocation check
    spec = json.loads((EXAMPLE / "example.json").read_text())
    compose = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())
    deployment = load_deployment_spec((EXAMPLE / "kairyu.yaml").read_text())

    assert control.compose_layout(compose) == control.expected_layout()
    chat = deployment.pools[spec["model"]["served_name"]]
    assert [replica.options["base_url"] for replica in chat.replicas] == [
        f"http://{service}:8000/v1" for service in control.CHAT_SERVICES
    ]
    embedding = deployment.embeddings[spec["embedding"]["served_name"]]
    assert embedding.dimensions == spec["embedding"]["dimensions"]
    assert [replica.base_url for replica in embedding.replicas] == [
        f"http://{service}:8000/v1" for service in control.EMBED_SERVICES
    ]
    assert deployment.upstream_responses_models == frozenset({spec["model"]["served_name"]})
