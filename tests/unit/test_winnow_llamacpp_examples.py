"""Cross-file contract of the Winnow-12B llama.cpp examples (LCP-D5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from kairyu.deploy.spec import load_deployment_spec
from kairyu.engine.config_validation import validate_backend_options

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


@pytest.mark.parametrize("environment", ["winnow-12b-q8-1gpu", "winnow-12b-q8-dp8-8gpu"])
def test_l1_geometry_matches_what_kairyu_admits(environment):
    """Kairyu's per-request context and admission must be the slots
    winnow-server actually runs; a drift silently mis-sizes admission, and a
    default /readyz health URL would never mark a llama-server replica ready."""

    root = EXAMPLES / environment
    spec = json.loads((root / "example.json").read_text())
    runtime = spec["runtime"]
    compose = yaml.safe_load((root / "compose.yaml").read_text())
    deployment = load_deployment_spec((root / "kairyu.yaml").read_text())
    entries = (
        list(deployment.engines.values())
        or list(deployment.pools[spec["model"]["served_name"]].replicas)
    )

    assert len(entries) == len(spec["replicas"])
    for replica, entry in zip(spec["replicas"], entries, strict=True):
        validate_backend_options(entry.backend, entry.options)
        assert entry.options["upstream"] == "llamacpp"
        assert entry.options["max_model_len"] == runtime["slot_context_tokens"]
        assert entry.resolved_health_url() == f"http://{replica['service']}:8091/health"
        command = compose["services"][replica["service"]]["command"]
        slots = int(_flag(command, "--chat-parallel"))
        assert slots == runtime["chat_slots"]
        assert int(_flag(command, "--context")) == slots * runtime["slot_context_tokens"]
        sampling = runtime["sampling_defaults"]
        for flag, name in (
            ("--temp", "temperature"),
            ("--top-k", "top_k"),
            ("--top-p", "top_p"),
            ("--min-p", "min_p"),
            ("--repeat-penalty", "repeat_penalty"),
        ):
            assert float(_flag(command, flag)) == sampling[name]
    assert deployment.server.max_concurrency == runtime["chat_slots"] * len(spec["replicas"])
